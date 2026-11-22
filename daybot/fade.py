"""Gap-FADE (short) backtest: the opposite of gap-and-go.

Idea: many small caps that gap up hard give the gain back during the day. Short them
once they lose VWAP, stop above the high of day.

Setup (pre-chosen, not tuned):
  - Stock gapped up >= GAP_MIN, $1-$20, first 5-min bar traded >= $500k (liquidity/borrow proxy).
  - After 9:45 ET and before 11:00 ET: first 5-min close BELOW VWAP while still above
    yesterday's close (gap not already filled) -> short at next bar's open.
  - Stop: high of day so far (+0.5%). Skip if the stop is more than 10% away.
  - Exit: 2R target, OR hold to 15:50 (both versions tested).

Reuses the market-wide gap data cached by daybot.gapscan (run that first, same dates).

Run:  python -m daybot.fade --start 2021-01-01 --end 2025-06-30

Big real-world catch printed in the report: you can only short stocks your broker can
borrow. Alpaca only allows shorting "easy to borrow" stocks, and many hot small caps aren't.
The --etb-only rows use TODAY's easy-to-borrow list as a rough stand-in.
"""
import argparse
from dataclasses import dataclass
from datetime import datetime, time, timedelta
from typing import Optional

import pandas as pd

from .backtest import Trade, report, run_trades
from .config import Settings
from .gapscan import intraday_for_day, scan_events
from .strategy import vwap


@dataclass
class FadeParams:
    gap_min: float = 0.10
    bar_minutes: int = 5
    first_entry: str = "09:45"
    last_entry: str = "11:00"
    flatten: str = "15:50"
    stop_buffer: float = 0.005
    min_stop_pct: float = 0.01
    max_stop_pct: float = 0.10
    reward_risk: Optional[float] = 2.0   # None = hold to close
    min_first_bar_dollar_volume: float = 500_000


def _t(s):
    h, m = s.split(":")
    return time(int(h), int(m))


def simulate_fade(symbol, bars, prev_close, p: FadeParams, slippage=0.01) -> Optional[Trade]:
    if len(bars) < 10:
        return None
    first = bars.iloc[0]
    if first["open"] / prev_close - 1 < p.gap_min:
        return None
    if first["close"] * first["volume"] < p.min_first_bar_dollar_volume:
        return None
    vw = vwap(bars)
    t0, t1, tf = _t(p.first_entry), _t(p.last_entry), _t(p.flatten)
    for i in range(1, len(bars) - 1):
        ts, bar = bars.index[i], bars.iloc[i]
        if ts.time() < t0:
            continue
        if ts.time() > t1:
            return None
        if bar["close"] < vw.iloc[i] and bar["close"] > prev_close:
            hod = float(bars["high"].iloc[:i + 1].max())
            entry = float(bars.iloc[i + 1]["open"]) - slippage       # short fills at next open
            stop = hod * (1 + p.stop_buffer)
            risk = stop - entry
            if risk <= 0:
                return None
            if risk / entry < p.min_stop_pct:
                stop, risk = entry * (1 + p.min_stop_pct), entry * p.min_stop_pct
            if risk / entry > p.max_stop_pct:
                return None
            target = entry - p.reward_risk * risk if p.reward_risk else None
            exit_px, reason, exit_t = None, "", None
            for j in range(i + 1, len(bars)):
                b, t = bars.iloc[j], bars.index[j]
                if t.time() >= tf:
                    exit_px, reason, exit_t = float(b["open"]) + slippage, "eod", t
                    break
                if b["high"] >= stop:                                  # worst case first
                    exit_px, reason, exit_t = max(stop, float(b["open"])) + slippage, "stop", t
                    break
                if target is not None and b["low"] <= target:
                    exit_px, reason, exit_t = target + slippage, "target", t
                    break
            if exit_px is None:
                exit_px, reason, exit_t = float(bars.iloc[-1]["close"]) + slippage, "eod", bars.index[-1]
            return Trade(symbol, ts.date(), bars.index[i + 1], exit_t, entry, stop, exit_px,
                         (entry - exit_px) / risk, reason, "fade", "short")
    return None


VARIANTS = [
    (f"gap {int(g*100)}%+, {'2R target' if rr else 'hold to close'}", FadeParams(gap_min=g, reward_risk=rr))
    for g in (0.10, 0.20) for rr in (2.0, None)
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--equity", type=float, default=10_000.0)
    ap.add_argument("--slippage", type=float, default=0.01)
    a = ap.parse_args()
    s = Settings()
    s.validate()
    from .broker import ET, Broker
    b = Broker(s.api_key, s.api_secret, paper=True, feed="sip")
    start = datetime.fromisoformat(a.start).replace(tzinfo=ET)
    end = datetime.fromisoformat(a.end).replace(tzinfo=ET)

    print("Loading gap days (cached from gapscan)...", flush=True)
    syms = b.listed_symbols(include_delisted=True)
    ev = scan_events(b, syms, start - timedelta(days=45), end, 0.10, s.risk.min_price, s.risk.max_price)
    ev = ev[ev["day"] >= start.date()]
    days = sorted(ev["day"].unique())
    print(f"  {len(ev)} gap days on {len(days)} trading days", flush=True)

    print("Checking which stocks Alpaca lists as easy to borrow today...", flush=True)
    from alpaca.trading.enums import AssetClass, AssetStatus
    from alpaca.trading.requests import GetAssetsRequest
    etb = {x.symbol for x in b.trading.get_all_assets(
        GetAssetsRequest(status=AssetStatus.ACTIVE, asset_class=AssetClass.US_EQUITY))
        if x.shortable and x.easy_to_borrow}

    cands = {name: [] for name, _ in VARIANTS}
    for k, day in enumerate(days, 1):
        todays = ev[ev["day"] == day]
        try:
            bars_by_sym = intraday_for_day(b, day, list(todays["symbol"]), 5)
        except Exception as e:
            print(f"  {day}: failed ({e})", flush=True)
            continue
        for r in todays.itertuples():
            bars = bars_by_sym.get(r.symbol)
            if bars is None:
                continue
            for name, p in VARIANTS:
                tr = simulate_fade(r.symbol, bars, float(r.prev_close), p, a.slippage)
                if tr and s.risk.min_price <= tr.entry <= s.risk.max_price:
                    cands[name].append(tr)
        if k % 200 == 0 or k == len(days):
            print(f"  {k}/{len(days)} days", flush=True)

    rows = []
    for name, _ in VARIANTS:
        for etb_only in (False, True):
            c = [t for t in cands[name] if (t.symbol in etb) or not etb_only]
            for pdt in (False, True):
                trades, curve, dd = run_trades(c, s.risk, a.equity, pdt=pdt)
                rs = pd.Series([t.r for t in trades], dtype=float)
                allr = pd.Series([t.r for t in c], dtype=float)
                rows.append(dict(
                    variant=name, borrow="easy-to-borrow only" if etb_only else "all",
                    pdt_limit="yes" if pdt else "no", setups=len(c),
                    setups_avg_R=round(allr.mean(), 3) if len(allr) else None,
                    trades=len(trades), win_rate=f"{(rs > 0).mean():.0%}" if len(rs) else "-",
                    avg_R=round(rs.mean(), 3) if len(rs) else None,
                    end_equity=f"${curve[-1][1]:,.0f}" if curve else f"${a.equity:,.0f}",
                    max_dd=f"{dd:.0%}"))
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    print(f"\n===== GAP FADE (SHORT) {a.start} to {a.end}, ${a.equity:,.0f} start, 1% risk/trade =====")
    print(df.to_string(index=False))
    df.to_csv("fade_results.csv", index=False)
    best = max(VARIANTS, key=lambda v: pd.Series([t.r for t in cands[v[0]]] or [0]).mean())[0]
    pd.DataFrame([t.__dict__ for t in cands[best]]).to_csv("fade_best_setups.csv", index=False)
    print("\nsetups_avg_R = every setup the rules found. avg_R = trades actually taken under the")
    print("account rules (2/day, 1 at a time, and with pdt_limit=yes: 3 day trades per 5 days under $25k).")
    print("Worth paper trading only if avg_R > +0.05 with 100+ trades, on the easy-to-borrow rows.")


if __name__ == "__main__":
    main()
