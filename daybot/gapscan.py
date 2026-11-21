"""Market-wide gap-and-go backtest.

Step 1: scan DAILY bars of every NYSE/NASDAQ/AMEX stock (including delisted ones) for
        days that opened 10%+ above the prior close in the $1-$20 range.
Step 2: download 5-min bars only for those stock-days and run the gap-and-go setup on them.

Uses split-adjusted prices (so reverse splits don't show up as fake gaps).
Everything is cached in data_cache/, so reruns are fast and an interrupted run resumes.

Run:  python -m daybot.gapscan --start 2021-01-01 --end 2025-06-30
"""
import argparse
import os
import time
from datetime import datetime, timedelta

import pandas as pd

from .backtest import report, run_trades, simulate_symbol_day
from .config import Settings
from .strategy import DayContext, find_gap_signal

CACHE = os.path.join("data_cache", "gapscan")
PAUSE = 0.35  # stay under Alpaca's free-plan rate limit (200 requests/min)


def find_gap_days(daily: pd.DataFrame, gap_min: float, min_price: float, max_price: float,
                  lookback: int = 20) -> pd.DataFrame:
    """daily: one symbol's daily bars (index = date). Returns qualifying gap days with context.
    Uses only info known at the open: prior close, prior average volume, today's open."""
    d = daily.sort_index()
    out = pd.DataFrame({
        "open": d["open"],
        "prev_close": d["close"].shift(1),
        "avg_daily_volume": d["volume"].shift(1).rolling(lookback, min_periods=10).mean(),
    })
    out["gap"] = out["open"] / out["prev_close"] - 1
    mask = (out["gap"] >= gap_min) & out["open"].between(min_price, max_price) & \
        out["avg_daily_volume"].notna()
    return out[mask]


def scan_events(b, syms, start, end, gap_min, min_price, max_price, batch=50):
    path = os.path.join(CACHE, f"events_{start:%Y%m%d}_{end:%Y%m%d}_{gap_min}.pkl")
    if os.path.exists(path):
        return pd.read_pickle(path)
    rows = []
    for i in range(0, len(syms), batch):
        chunk = syms[i:i + batch]
        try:
            df = b.daily_bars(chunk, start, end)
        except Exception as e:
            print(f"  daily batch {i//batch+1} failed: {e}", flush=True)
            time.sleep(5)
            continue
        if df is not None and not df.empty:
            for sym, sdf in df.groupby(level=0):
                sdf = sdf.droplevel(0)
                sdf.index = sdf.index.tz_convert("America/New_York").date
                ev = find_gap_days(sdf, gap_min, min_price, max_price)
                for day, r in ev.iterrows():
                    rows.append(dict(symbol=sym, day=day, prev_close=r.prev_close,
                                     avg_daily_volume=r.avg_daily_volume, gap=r.gap))
        print(f"  scanned {min(i+batch, len(syms))}/{len(syms)} stocks | gap days so far: {len(rows)}",
              flush=True)
        time.sleep(PAUSE)
    ev = pd.DataFrame(rows)
    pd.to_pickle(ev, path)
    return ev


def intraday_for_day(b, day, symbols, bar_minutes):
    """5-min split-adjusted bars for several symbols on one day (cached per day)."""
    path = os.path.join(CACHE, "days", f"{day}.pkl")
    if os.path.exists(path):
        return pd.read_pickle(path)
    from .broker import ET
    s = datetime(day.year, day.month, day.day, 9, 30, tzinfo=ET)
    out = {}
    for i in range(0, len(symbols), 100):
        res = b.bars(symbols[i:i + 100], s, s + timedelta(hours=6, minutes=30), bar_minutes,
                     adjustment="split")
        for sym, days in res.items():
            if day in days:
                out[sym] = days[day].astype("float32")
        time.sleep(PAUSE)
    pd.to_pickle(out, path)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--equity", type=float, default=100.0)
    ap.add_argument("--slippage", type=float, default=0.01)
    ap.add_argument("--active-only", action="store_true", help="skip delisted stocks (not recommended)")
    a = ap.parse_args()
    s = Settings()
    s.validate()
    from .broker import ET, Broker
    b = Broker(s.api_key, s.api_secret, paper=True, feed="sip")
    os.makedirs(os.path.join(CACHE, "days"), exist_ok=True)
    start = datetime.fromisoformat(a.start).replace(tzinfo=ET)
    end = datetime.fromisoformat(a.end).replace(tzinfo=ET)
    gp = s.gap

    print("Step 1/2: listing stocks...", flush=True)
    syms = b.listed_symbols(include_delisted=not a.active_only)
    print(f"  {len(syms)} stocks. Scanning daily bars for {gp.gap_min:.0%}+ gaps "
          f"(${s.risk.min_price:g}-${s.risk.max_price:g})...", flush=True)
    ev = scan_events(b, syms, start - timedelta(days=45), end, gp.gap_min,
                     s.risk.min_price, s.risk.max_price)
    first = start.date()
    ev = ev[ev["day"] >= first] if len(ev) else ev
    if ev.empty:
        print("No gap days found.")
        return
    days = sorted(ev["day"].unique())
    print(f"  {len(ev)} gap days across {ev['symbol'].nunique()} stocks on {len(days)} trading days",
          flush=True)

    market = None
    if gp.market_filter:
        from .backtest import load_symbol
        market = load_symbol(b, "SPY", start - timedelta(days=5), end, gp.bar_minutes)

    print("Step 2/2: testing gap-and-go on each gap day (5-min bars)...", flush=True)
    cands, tested = [], 0
    for k, day in enumerate(days, 1):
        todays = ev[ev["day"] == day]
        try:
            bars_by_sym = intraday_for_day(b, day, list(todays["symbol"]), gp.bar_minutes)
        except Exception as e:
            print(f"  {day}: failed ({e})", flush=True)
            time.sleep(5)
            continue
        for r in todays.itertuples():
            bars = bars_by_sym.get(r.symbol)
            if bars is None or len(bars) < 10:
                continue
            tested += 1
            ctx = DayContext(prev_close=float(r.prev_close), avg_daily_volume=float(r.avg_daily_volume),
                             market_bars=market.get(day) if market else None)
            sig = find_gap_signal(r.symbol, bars, gp, ctx)
            tr = simulate_symbol_day(r.symbol, bars, gp, a.slippage, sig) if sig else None
            if tr and s.risk.min_price <= tr.entry <= s.risk.max_price:
                cands.append(tr)
        if k % 50 == 0 or k == len(days):
            print(f"  {k}/{len(days)} days | gap days tested {tested} | setups {len(cands)}", flush=True)

    trades, curve, dd = run_trades(cands, s.risk, a.equity)
    print("\n===== GAP-AND-GO (market-wide, incl. delisted) =====")
    print(f"Gap days found: {len(ev)} | with intraday data: {tested} | setups triggered: {len(cands)}")
    print(report(trades, curve, dd, a.equity))
    pd.DataFrame([t.__dict__ for t in trades]).to_csv("backtest_gapscan.csv", index=False)
    pd.DataFrame([t.__dict__ for t in cands]).to_csv("backtest_gapscan_all_setups.csv", index=False)
    if cands:
        r = pd.Series([t.r for t in cands])
        print(f"All {len(cands)} setups (before the 2-trades/day limit): avg R {r.mean():+.3f}, "
              f"win rate {(r > 0).mean():.0%}")


if __name__ == "__main__":
    main()
