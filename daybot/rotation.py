"""Sector momentum rotation ("dual momentum" style), judged against buy & hold SPY.

Pre-chosen rules (not tuned):
  - Once a month, at the last close of the month, rank the 11 SPDR sector ETFs by their
    past return (6-month or 12-month lookback - both tested).
  - Hold the top 3, equal weight, from the next day's open until next month.
  - Trend filter (tested on and off): if SPY closes below its 200-day average,
    hold T-bills (BIL) instead.
Assumptions: next-open fills, 0.05% slippage on every dollar traded, dividend-adjusted
prices, fractional shares.

Run:  python -m daybot.rotation --start 2016-01-01 --end 2025-06-30
"""
import argparse
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

SECTORS = ["XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB", "XLRE", "XLC"]
CASH = "BIL"
SLIPPAGE = 0.0005
LOOKBACK_DAYS = {6: 126, 12: 252}


def month_ends(index: pd.DatetimeIndex) -> list:
    """Positions of the last trading day of each month."""
    m = index.to_period("M")
    return [i for i in range(len(index) - 1) if m[i] != m[i + 1]]


def target_weights(closes: pd.DataFrame, t: int, lookback: int, top_n: int, trend_filter: bool):
    w = pd.Series(0.0, index=closes.columns)
    spy = closes["SPY"]
    if trend_filter:
        sma = spy.iloc[max(0, t - 199):t + 1].mean()
        if t < 199 or spy.iloc[t] < sma:
            w[CASH] = 1.0
            return w
    if t < lookback:
        w[CASH] = 1.0
        return w
    past, now = closes.iloc[t - lookback], closes.iloc[t]
    score = (now / past - 1)[[s for s in SECTORS if s in closes.columns]].dropna()
    picks = score.sort_values(ascending=False).index[:top_n]
    if len(picks) == 0:
        w[CASH] = 1.0
    else:
        w[picks] = 1.0 / len(picks)
    return w


def backtest(closes: pd.DataFrame, opens: pd.DataFrame, lookback: int, top_n=3, trend_filter=True):
    """Returns daily portfolio returns (open-to-open) and list of monthly holdings."""
    idx = closes.index
    rets = (opens.shift(-1) / opens - 1).fillna(0.0)
    W = pd.DataFrame(0.0, index=idx, columns=closes.columns)
    current = pd.Series(0.0, index=closes.columns)
    current[CASH] = 1.0
    holdings, costs = [], pd.Series(0.0, index=idx)
    ends = set(month_ends(idx))
    for t in range(len(idx)):
        if t - 1 in ends:                       # decided at previous close, trade at this open
            new = target_weights(closes, t - 1, lookback, top_n, trend_filter)
            costs.iloc[t] = (new - current).abs().sum() * SLIPPAGE
            current = new
            holdings.append((idx[t], ",".join(current[current > 0].index)))
        W.iloc[t] = current
    daily = (W * rets).sum(axis=1) - costs
    return daily, holdings


def metrics(name, daily: pd.Series, start_equity: float, n_rebalances: int):
    curve = (1 + daily).cumprod()
    years = len(daily) / 252
    cagr = curve.iloc[-1] ** (1 / years) - 1
    dd = (curve / curve.cummax() - 1).min()
    sharpe = daily.mean() / daily.std() * np.sqrt(252) if daily.std() > 0 else 0
    monthly = (1 + daily).groupby(daily.index.to_period("M")).prod() - 1
    return dict(strategy=name, rebalances=n_rebalances, months_up=f"{(monthly > 0).mean():.0%}",
                yearly=f"{cagr:+.1%}", max_dd=f"{dd:.0%}", sharpe=round(sharpe, 2),
                end_equity=f"${start_equity * curve.iloc[-1]:,.0f}")


def run(data: dict, first_day, start_equity=10_000.0):
    closes = pd.DataFrame({s: d["close"] for s, d in data.items()}).sort_index()
    opens = pd.DataFrame({s: d["open"] for s, d in data.items()}).sort_index()
    keep = closes.index >= pd.Timestamp(first_day)
    rows = []
    for months, lb in LOOKBACK_DAYS.items():
        for tf in (True, False):
            daily, hold = backtest(closes, opens, lb, 3, tf)
            n = sum(1 for d, _ in hold if d >= pd.Timestamp(first_day))
            rows.append(metrics(f"Top 3 sectors, {months}-mo momentum, trend filter {'on' if tf else 'off'}",
                                daily[keep], start_equity, n))
    spy_r = (opens["SPY"].shift(-1) / opens["SPY"] - 1).fillna(0.0)[keep]
    rows.append(metrics("Buy & hold SPY (benchmark)", spy_r, start_equity, 0))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--equity", type=float, default=10_000.0)
    a = ap.parse_args()
    from .config import Settings
    from .swing import load_daily
    s = Settings()
    s.validate()
    from .broker import ET, Broker
    b = Broker(s.api_key, s.api_secret, paper=True, feed="sip")
    start = datetime.fromisoformat(a.start).replace(tzinfo=ET)
    end = datetime.fromisoformat(a.end).replace(tzinfo=ET)
    print("Downloading daily bars for 11 sector ETFs + SPY + BIL...", flush=True)
    data = load_daily(b, SECTORS + ["SPY", CASH], start - timedelta(days=400), end)
    first_data = min(d.index.min() for d in data.values())
    first_day = max(start.date(), (first_data + pd.Timedelta(days=380)).date())
    if first_day > start.date():
        print(f"(Data starts {first_data.date()}; results begin {first_day} so the 12-month lookback "
              f"and 200-day average have history.)")
    df = run(data, first_day, a.equity)
    pd.set_option("display.width", 200)
    print(f"\n===== SECTOR MOMENTUM ROTATION {first_day} to {a.end} (${a.equity:,.0f} start) =====")
    print(df.to_string(index=False))
    df.to_csv("rotation_results.csv", index=False)
    print("\nWorth trading only if it beats buy & hold on sharpe AND has a smaller max drawdown.")


if __name__ == "__main__":
    main()
