"""Swing-trading backtest on DAILY bars, judged against simply buying and holding SPY.

Pre-chosen strategies (no tuning):
  1. RSI(2) pullback (Larry Connors): ETF above its 200-day average and RSI(2) < 10 -> buy.
     Sell when it closes above its 5-day average, or after 10 days.
     Run on SPY, QQQ, IWM, DIA with 1/4 of the account per ETF.
  2. Same RSI(2) pullback on SPY only.
  3. Trend filter: hold SPY while it closes above its 200-day average, else cash.
  4. Benchmark: buy and hold SPY.

Honest assumptions: signals use the CLOSE, trades happen at the NEXT day's OPEN,
0.05% slippage per side, fractional shares (Alpaca supports them), dividend-adjusted prices.

Run:  python -m daybot.swing --start 2016-01-01 --end 2025-06-30
      (then once on the held-out period: --start 2025-07-01 --end 2026-11-20)
"""
import argparse
import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

ETFS = ["SPY", "QQQ", "IWM", "DIA"]
SLIPPAGE = 0.0005


def rsi(close: pd.Series, n: int = 2) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(100)


def rsi2_positions(df: pd.DataFrame, entry_rsi=10, exit_ma=5, trend_ma=200, max_hold=10):
    """Returns (position series aligned to days: 1 = held from that day's open to next open,
    list of (entry_idx, exit_idx))."""
    c = df["close"]
    sma_t, sma_x, r = c.rolling(trend_ma).mean(), c.rolling(exit_ma).mean(), rsi(c)
    n = len(df)
    pos = np.zeros(n)
    trades, held, entry = [], False, None
    for t in range(n - 1):            # decision at close t -> acts at open t+1
        if not held:
            if pd.notna(sma_t.iloc[t]) and c.iloc[t] > sma_t.iloc[t] and r.iloc[t] < entry_rsi:
                held, entry = True, t + 1
        else:
            if c.iloc[t] > sma_x.iloc[t] or (t + 1 - entry) >= max_hold:
                held = False
                trades.append((entry, t + 1))
        pos[t + 1] = 1.0 if held else 0.0
    if held:
        trades.append((entry, n - 1))
    return pd.Series(pos, index=df.index), trades


def trend_positions(df: pd.DataFrame, trend_ma=200):
    c = df["close"]
    above = (c > c.rolling(trend_ma).mean()).astype(float)
    pos = above.shift(1).fillna(0.0)   # decide at close, hold from next open
    trades, start = [], None
    for i, p in enumerate(pos.values):
        if p and start is None:
            start = i
        elif not p and start is not None:
            trades.append((start, i))
            start = None
    if start is not None:
        trades.append((start, len(pos) - 1))
    return pos, trades


def portfolio(data: dict, positions: dict, weight: float):
    """Daily equity curve from open-to-open returns with slippage on each position change."""
    rets = []
    for sym, pos in positions.items():
        o = data[sym]["open"]
        r = (o.shift(-1) / o - 1).fillna(0.0)
        cost = pos.diff().abs().fillna(pos.iloc[0]) * SLIPPAGE
        rets.append(weight * (pos * r - cost))
    daily = pd.concat(rets, axis=1).sum(axis=1)
    return (1 + daily).cumprod(), daily


def trade_returns(df, trades):
    o = df["open"].values
    return [o[x] / o[e] - 1 - 2 * SLIPPAGE for e, x in trades if x > e]


def stats(name, curve, daily, trade_rets, exposure, start_equity):
    years = len(curve) / 252
    cagr = curve.iloc[-1] ** (1 / years) - 1 if years > 0 else 0
    dd = (curve / curve.cummax() - 1).min()
    sharpe = daily.mean() / daily.std() * np.sqrt(252) if daily.std() > 0 else 0
    tr = pd.Series(trade_rets, dtype=float)
    return dict(strategy=name, trades=len(tr),
                win_rate=f"{(tr > 0).mean():.0%}" if len(tr) else "-",
                avg_trade=f"{tr.mean():+.2%}" if len(tr) else "-",
                yearly=f"{cagr:+.1%}", max_dd=f"{dd:.0%}", sharpe=round(sharpe, 2),
                in_market=f"{exposure:.0%}",
                end_equity=f"${start_equity * curve.iloc[-1]:,.2f}")


def run(data: dict, first_day, start_equity=100.0):
    rows, pos_all = [], {}
    for s in ETFS:
        if s in data:
            pos_all[s] = rsi2_positions(data[s])[0]

    def clip(x):
        return x[x.index >= pd.Timestamp(first_day)]

    # Indicators use pre-start history; results count only from first_day
    def eval_(name, positions, weight, trs):
        clipped = {k: clip(v) for k, v in positions.items()}
        cd = {k: data[k].loc[clipped[k].index] for k in clipped}
        curve, daily = portfolio(cd, clipped, weight)
        exposure = pd.concat(clipped.values(), axis=1).max(axis=1).mean()
        return stats(name, curve, daily, trs, exposure, start_equity)

    def trs_for(sym_trades):
        out = []
        for s, (df, t) in sym_trades.items():
            kept = [(e, x) for e, x in t if df.index[e] >= pd.Timestamp(first_day)]
            out += trade_returns(df, kept)
        return out

    rsi_trades = {s: (data[s], rsi2_positions(data[s])[1]) for s in pos_all}
    rows.append(eval_("RSI(2) pullback, 4 ETFs", pos_all, 1 / len(pos_all), trs_for(rsi_trades)))
    rows.append(eval_("RSI(2) pullback, SPY only", {"SPY": pos_all["SPY"]}, 1.0,
                      trs_for({"SPY": rsi_trades["SPY"]})))
    tp, tt = trend_positions(data["SPY"])
    rows.append(eval_("Trend: SPY above 200-day", {"SPY": tp}, 1.0, trs_for({"SPY": (data["SPY"], tt)})))
    bh = pd.Series(1.0, index=data["SPY"].index)
    rows.append(eval_("Buy & hold SPY (benchmark)", {"SPY": bh}, 1.0, []))
    return pd.DataFrame(rows)


def load_daily(b, syms, start, end, cache_dir="data_cache"):
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"daily_{'_'.join(syms)}_{start:%Y%m%d}_{end:%Y%m%d}.pkl")
    if os.path.exists(path):
        return pd.read_pickle(path)
    df = b.daily_bars(syms, start, end, adjustment="all")
    out = {}
    for sym, sdf in df.groupby(level=0):
        sdf = sdf.droplevel(0)[["open", "high", "low", "close", "volume"]]
        sdf.index = pd.DatetimeIndex(sdf.index.tz_convert("America/New_York").date)
        out[sym] = sdf
    pd.to_pickle(out, path)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--equity", type=float, default=100.0)
    a = ap.parse_args()
    from .config import Settings
    s = Settings()
    s.validate()
    from .broker import ET, Broker
    b = Broker(s.api_key, s.api_secret, paper=True, feed="sip")
    start = datetime.fromisoformat(a.start).replace(tzinfo=ET)
    end = datetime.fromisoformat(a.end).replace(tzinfo=ET)
    print("Downloading daily bars (with ~1 year of extra history for the 200-day average)...", flush=True)
    data = load_daily(b, ETFS, start - timedelta(days=400), end)
    df = run(data, start.date(), a.equity)
    pd.set_option("display.width", 200)
    print(f"\n===== SWING STRATEGIES {a.start} to {a.end} (${a.equity:g} start) =====")
    print(df.to_string(index=False))
    df.to_csv("swing_results.csv", index=False)
    print("\nWorth trading only if it beats buy & hold on sharpe AND has a smaller max drawdown.")


if __name__ == "__main__":
    main()
