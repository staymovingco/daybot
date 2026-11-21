"""Backtester using the exact same strategy + risk rules as the live bot.

Honest assumptions: enter at the NEXT bar's open (no look-ahead), slippage both ways,
if stop and target are hit in the same bar we assume the stop (worst case), gaps through
the stop fill at the worse price, everything closed at flatten_time.

Run:  python -m daybot.backtest --start 2021-01-01 --end 2025-06-30 --strategy both
      python -m daybot.backtest ... --strategy gap --universe smallcaps.txt
      (hold out the most recent ~200 trading days and test on them last, once)
"""
import argparse
import math
from dataclasses import dataclass
from datetime import datetime, timedelta

import pandas as pd

from .config import RiskConfig, Settings, StrategyParams
from .strategy import STRATEGIES, Signal, _t, context_from_history, find_orb_signal


@dataclass
class Trade:
    symbol: str
    day: object
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry: float
    stop: float
    exit: float
    r: float
    exit_reason: str
    strategy: str = "orb"


def simulate_symbol_day(symbol, bars, p: StrategyParams, slippage: float = 0.01,
                        sig: Signal = None):
    if sig is None:
        sig = find_orb_signal(symbol, bars, p)
    if not sig:
        return None
    i = bars.index.get_loc(sig.time)
    if i + 1 >= len(bars):
        return None
    entry_bar = bars.iloc[i + 1]
    entry = float(entry_bar["open"]) + slippage
    risk = entry - sig.stop
    if risk <= 0:
        return None
    target = entry + p.reward_risk * risk
    flat = _t(p.flatten_time)
    exit_px, reason, exit_t = None, "", None
    for j in range(i + 1, len(bars)):
        b, t = bars.iloc[j], bars.index[j]
        if t.time() >= flat:
            exit_px, reason, exit_t = float(b["open"]) - slippage, "eod", t
            break
        if b["low"] <= sig.stop:
            exit_px, reason, exit_t = min(sig.stop, float(b["open"])) - slippage, "stop", t
            break
        if b["high"] >= target:
            exit_px, reason, exit_t = target - slippage, "target", t
            break
    if exit_px is None:
        exit_px, reason, exit_t = float(bars.iloc[-1]["close"]) - slippage, "eod", bars.index[-1]
    return Trade(symbol, bars.index[0].date(), bars.index[i + 1], exit_t,
                 entry, sig.stop, exit_px, (exit_px - entry) / risk, reason, sig.strategy)


def run_portfolio(sessions: dict, strategies, rc: RiskConfig,
                  start_equity: float = 100.0, slippage: float = 0.01, first_day=None):
    """sessions = {symbol: {date: bars}}; strategies = [(name, params), ...] or one params object
    (ORB). Compounds equity with the live risk rules."""
    if isinstance(strategies, StrategyParams):
        strategies = [("orb", strategies)]
    days = sorted({d for s in sessions.values() for d in s if first_day is None or d >= first_day})
    equity, peak, max_dd = start_equity, start_equity, 0.0
    taken, curve = [], []
    for d in days:
        cands = []
        for sym, sd in sessions.items():
            bars = sd.get(d)
            if bars is None or len(bars) < 10:
                continue
            ctx = context_from_history(sd, d)
            for name, p in strategies:
                sig = STRATEGIES[name](sym, bars, p, ctx)
                tr = simulate_symbol_day(sym, bars, p, slippage, sig) if sig else None
                if tr and rc.min_price <= tr.entry <= rc.max_price:
                    cands.append(tr)
                    break  # one setup per symbol per day
        cands.sort(key=lambda t: t.entry_time)
        day_start, n, busy_until = equity, 0, None
        for tr in cands:
            if n >= rc.max_trades_per_day:
                break
            if (equity - day_start) / day_start <= -rc.daily_loss_limit_pct:
                break
            if busy_until is not None and tr.entry_time < busy_until:
                continue  # max 1 open position
            qty = math.floor(min(equity * rc.risk_pct / (tr.entry - tr.stop),
                                 equity * rc.cash_buffer / tr.entry))
            if qty < 1:
                continue
            equity += qty * (tr.exit - tr.entry)
            taken.append(tr)
            n += 1
            busy_until = tr.exit_time
        peak = max(peak, equity)
        max_dd = min(max_dd, (equity - peak) / peak)
        curve.append((d, equity))
    return taken, curve, max_dd


def report(trades, curve, max_dd, start_equity):
    if not trades:
        return "No trades. Loosen filters or add symbols/dates."
    rs = pd.Series([t.r for t in trades])
    wins, losses = rs[rs > 0], rs[rs <= 0]
    pf = wins.sum() / abs(losses.sum()) if len(losses) and losses.sum() != 0 else float("inf")
    end = curve[-1][1]
    lines = [
        f"Trades:          {len(trades)}  over {len(curve)} days",
        f"Win rate:        {len(wins)/len(rs):.1%}",
        f"Avg R per trade: {rs.mean():+.3f}   (expectancy; must be > 0 after costs)",
        f"Profit factor:   {pf:.2f}   (want > 1.3)",
        f"Exits:           " + ", ".join(f"{k} {v}" for k, v in pd.Series([t.exit_reason for t in trades]).value_counts().items()),
        f"Equity:          ${start_equity:,.2f} -> ${end:,.2f}  ({end/start_equity-1:+.1%})",
        f"Max drawdown:    {max_dd:.1%}",
        "PASS" if len(trades) >= 100 and rs.mean() > 0.05 and max_dd > -0.10
        else "NOT READY: need 100+ trades, avg R > +0.05, drawdown under 10%",
    ]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--symbols", default="")
    ap.add_argument("--universe", default="", help="text file with one symbol per line")
    ap.add_argument("--strategy", default="both", choices=["orb", "gap", "both"])
    ap.add_argument("--equity", type=float, default=100.0)
    ap.add_argument("--slippage", type=float, default=0.01, help="$ per share each way")
    ap.add_argument("--feed", default="sip", choices=["sip", "iex"],
                    help="sip = full-market history (free plan allows data older than 15 min)")
    a = ap.parse_args()
    s = Settings()
    s.validate()
    from .broker import ET, Broker
    b = Broker(s.api_key, s.api_secret, paper=True, feed=a.feed)
    print(f"Downloading 5-min bars ({a.feed.upper()} feed)... this can take a few minutes.", flush=True)
    syms = [x.strip().upper() for x in a.symbols.split(",") if x.strip()]
    if a.universe:
        syms += [l.strip().upper() for l in open(a.universe) if l.strip() and not l.startswith("#")]
    syms = list(dict.fromkeys(syms)) or s.watchlist
    start = datetime.fromisoformat(a.start).replace(tzinfo=ET) - timedelta(days=35)  # history for gap context
    sessions = b.bars(syms, start, datetime.fromisoformat(a.end).replace(tzinfo=ET), s.orb.bar_minutes)
    first_day = datetime.fromisoformat(a.start).date()

    names = ["orb", "gap"] if a.strategy == "both" else [a.strategy]
    runs = [[n] for n in names] + ([names] if len(names) > 1 else [])
    for run in runs:
        strats = [(n, s.params(n)) for n in run]
        trades, curve, dd = run_portfolio(sessions, strats, s.risk, a.equity, a.slippage,
                                          first_day=first_day)
        print(f"\n===== {' + '.join(n.upper() for n in run)} =====")
        print(report(trades, curve, dd, a.equity))
        out = f"backtest_{'_'.join(run)}.csv"
        pd.DataFrame([t.__dict__ for t in trades]).to_csv(out, index=False)
        print(f"Trade list saved to {out}")


if __name__ == "__main__":
    main()
