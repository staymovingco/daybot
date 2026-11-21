"""Test a SMALL, pre-chosen set of ORB variants side by side (no endless tuning).

Run:  python -m daybot.sweep --start 2021-01-01 --end 2025-06-30 --universe smallcaps.txt

Uses the same cached data as the backtester, so after the first run it's fast.
Pick the winner here, then test it ONCE on the held-out period with daybot.backtest.
"""
import argparse
from dataclasses import replace
from datetime import datetime, timedelta

import pandas as pd

from .backtest import load_symbol, run_trades, symbol_candidates
from .config import Settings

# 8 variants: range length x stop placement x market filter. Chosen up front, not mined.
VARIANTS = [
    (f"{orm}m range, stop {stop}, SPY filter {'on' if mf else 'off'}",
     dict(or_minutes=orm, stop_mode=stop, market_filter=mf))
    for orm in (15, 30) for stop in ("mid", "low") for mf in (False, True)
]


def summarize(name, trades, curve, dd, start_equity):
    if not trades:
        return dict(variant=name, trades=0, win_rate="-", avg_R=float("nan"),
                    profit_factor=float("nan"), end_equity=start_equity, max_dd="0%")
    rs = pd.Series([t.r for t in trades])
    wins, losses = rs[rs > 0], rs[rs <= 0]
    pf = wins.sum() / abs(losses.sum()) if losses.sum() != 0 else float("inf")
    return dict(variant=name, trades=len(trades), win_rate=f"{len(wins)/len(rs):.0%}",
                avg_R=round(rs.mean(), 3), profit_factor=round(pf, 2),
                end_equity=round(curve[-1][1], 2), max_dd=f"{dd:.0%}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--universe", default="smallcaps.txt")
    ap.add_argument("--equity", type=float, default=100.0)
    ap.add_argument("--slippage", type=float, default=0.01)
    a = ap.parse_args()
    s = Settings()
    s.validate()
    from .broker import ET, Broker
    b = Broker(s.api_key, s.api_secret, paper=True, feed="sip")
    syms = [l.strip().upper() for l in open(a.universe) if l.strip() and not l.startswith("#")]
    start = datetime.fromisoformat(a.start).replace(tzinfo=ET) - timedelta(days=35)
    end = datetime.fromisoformat(a.end).replace(tzinfo=ET)
    first_day = datetime.fromisoformat(a.start).date()

    print("Loading SPY for the market filter...", flush=True)
    market = load_symbol(b, "SPY", start, end, s.orb.bar_minutes)
    variants = [(name, replace(s.orb, **kw)) for name, kw in VARIANTS]
    cands = {name: [] for name, _ in variants}
    for k, sym in enumerate(syms, 1):
        try:
            days = load_symbol(b, sym, start, end, s.orb.bar_minutes)
        except Exception as e:
            print(f"  [{k}/{len(syms)}] {sym}: skipped ({e})", flush=True)
            continue
        for name, p in variants:
            cands[name] += symbol_candidates(sym, days, [("orb", p)], s.risk, a.slippage,
                                             first_day, market)
        print(f"  [{k}/{len(syms)}] {sym} done", flush=True)
        del days

    rows = []
    for name, _ in variants:
        trades, curve, dd = run_trades(cands[name], s.risk, a.equity)
        rows.append(summarize(name, trades, curve, dd, a.equity))
    df = pd.DataFrame(rows).sort_values("avg_R", ascending=False, na_position="last")
    pd.set_option("display.width", 200)
    print("\n===== ORB VARIANTS (sorted by avg R per trade) =====")
    print(df.to_string(index=False))
    df.to_csv("sweep_orb.csv", index=False)
    print("\nA variant is worth holdout-testing only if avg_R > +0.05 with 100+ trades.")


if __name__ == "__main__":
    main()
