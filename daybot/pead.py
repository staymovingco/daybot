"""Post-Earnings Announcement Drift (PEAD) backtest, judged against SPY.

Idea (decades of academic research): stocks that beat earnings estimates by a lot tend to
keep drifting up for weeks after the report.

Pre-chosen rules (not tuned):
  - Earnings beat: actual EPS beats the estimate by >= 10% (estimate must be meaningful).
  - Report day D. To avoid needing the exact before/after-market time, we wait:
    the "reaction" is close of D+1 vs close of D-1, and we BUY at the open of D+2.
  - Versions tested: hold 20 or 60 trading days; require a positive price reaction or not.
  - Up to 10 positions at once, 10% of the account each (fractional shares), 0.05% slippage.
Key number: avg_excess = average trade return MINUS what SPY did over the same days.

Data: earnings from Alpha Vantage (ALPHAVANTAGE_API_KEY, free, full history, 25 stocks/day)
or FMP (FMP_API_KEY; free plan only returns the last 5 reports). Prices from Alpaca.
Free FMP plan allows ~250 requests/day; one request per stock, cached in data_cache/fmp/.

Run:  python -m daybot.pead --probe            (check what your FMP plan returns)
      python -m daybot.pead --start 2016-01-01 --end 2025-06-30
"""
import argparse
import json
import os
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests

FMP = "https://financialmodelingprep.com"
CACHE = os.path.join("data_cache", "fmp")
SLIPPAGE = 0.0005


def fmp_get(path: str, key: str, **params):
    r = requests.get(f"{FMP}{path}", params={**params, "apikey": key}, timeout=30)
    try:
        data = r.json()
    except Exception:
        data = r.text
    return r.status_code, data


def parse_alphavantage(data) -> pd.DataFrame:
    """Alpha Vantage EARNINGS response -> date, eps, est."""
    out = []
    for x in (data or {}).get("quarterlyEarnings", []) if isinstance(data, dict) else []:
        try:
            out.append(dict(date=pd.Timestamp(x["reportedDate"]), eps=float(x["reportedEPS"]),
                            est=float(x["estimatedEPS"])))
        except (KeyError, TypeError, ValueError):
            continue   # "None" values etc.
    df = pd.DataFrame(out)
    if len(df):
        df = df.drop_duplicates("date").sort_values("date").reset_index(drop=True)
    return df


class DailyLimit(Exception):
    pass


def get_earnings_av(sym: str, key: str) -> pd.DataFrame:
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, f"av_{sym}.json")
    if os.path.exists(path):
        return parse_alphavantage(json.load(open(path)))
    r = requests.get("https://www.alphavantage.co/query",
                     params=dict(function="EARNINGS", symbol=sym, apikey=key), timeout=30)
    data = r.json()
    if isinstance(data, dict) and ("Information" in data or "Note" in data) and "quarterlyEarnings" not in data:
        raise DailyLimit(str(data.get("Information") or data.get("Note"))[:160])
    json.dump(data, open(path, "w"))
    time.sleep(13)   # free plan also limits requests per minute
    return parse_alphavantage(data)


def parse_earnings(rows) -> pd.DataFrame:
    """Accepts FMP stable or v3 shapes. Returns date, eps, est columns."""
    out = []
    for x in rows if isinstance(rows, list) else []:
        eps = x.get("epsActual", x.get("eps"))
        est = x.get("epsEstimated")
        d = x.get("date")
        if d is None or eps is None or est is None:
            continue
        out.append(dict(date=pd.Timestamp(d[:10]), eps=float(eps), est=float(est)))
    df = pd.DataFrame(out)
    if len(df):
        df = df.drop_duplicates("date").sort_values("date").reset_index(drop=True)
    return df


def get_earnings(sym: str, key: str) -> pd.DataFrame:
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, f"{sym}.json")
    if os.path.exists(path):
        return parse_earnings(json.load(open(path)))
    status, data = fmp_get("/stable/earnings", key, symbol=sym, limit=5)   # free plan max = 5
    if status != 200 or not isinstance(data, list):
        status2, data2 = fmp_get(f"/api/v3/historical/earning_calendar/{sym}", key, limit=200)
        if status2 == 200 and isinstance(data2, list):
            data = data2
        else:
            raise RuntimeError(f"FMP {status}: {str(data)[:160]}")
    json.dump(data, open(path, "w"))
    time.sleep(0.3)
    return parse_earnings(data)


def find_events(sym, earn: pd.DataFrame, px: pd.DataFrame, beat_min=0.10, min_abs_est=0.05):
    """Earnings beats mapped onto trading days. Returns dicts with entry index etc."""
    idx = px.index
    ev = []
    for r in earn.itertuples():
        if abs(r.est) < min_abs_est:
            continue
        surprise = (r.eps - r.est) / abs(r.est)
        if surprise < beat_min:
            continue
        d = idx.searchsorted(r.date)          # first trading day on/after report date
        if d < 1 or d + 2 >= len(idx):
            continue
        reaction = px["close"].iloc[d + 1] / px["close"].iloc[d - 1] - 1
        ev.append(dict(symbol=sym, report=r.date, surprise=surprise, reaction=reaction, entry=d + 2))
    return ev


def simulate(events, prices: dict, hold: int, need_up: bool, first_day, max_pos=10):
    """Equal-slot portfolio: up to max_pos positions, each 1/max_pos of equity at entry day."""
    spy = prices["SPY"]
    days = spy.index[spy.index >= pd.Timestamp(first_day)]
    trades = []
    for e in events:
        px = prices[e["symbol"]]
        if need_up and e["reaction"] <= 0:
            continue
        i0, i1 = e["entry"], min(e["entry"] + hold, len(px) - 1)
        if px.index[i0] < pd.Timestamp(first_day) or i1 <= i0:
            continue
        o = px["open"]
        ret = o.iloc[i1] / o.iloc[i0] - 1 - 2 * SLIPPAGE
        s0, s1 = spy.index.searchsorted(px.index[i0]), spy.index.searchsorted(px.index[i1])
        spy_ret = spy["open"].iloc[min(s1, len(spy) - 1)] / spy["open"].iloc[s0] - 1
        trades.append(dict(symbol=e["symbol"], entry=px.index[i0], exit=px.index[i1], ret=ret,
                           excess=ret - spy_ret, surprise=e["surprise"]))
    trades.sort(key=lambda t: t["entry"])

    # Daily portfolio returns with slot limit
    open_pos, taken = [], []
    daily = pd.Series(0.0, index=days)
    sym_rets = {s: (p["open"].shift(-1) / p["open"] - 1).fillna(0.0) for s, p in prices.items()}
    ti = 0
    for d in days:
        open_pos = [t for t in open_pos if t["exit"] > d]
        while ti < len(trades) and trades[ti]["entry"] <= d:
            t = trades[ti]
            ti += 1
            if t["entry"] == d and len(open_pos) < max_pos:
                open_pos.append(t)
                taken.append(t)
                daily[d] -= SLIPPAGE / max_pos
        for t in open_pos:
            daily[d] += sym_rets[t["symbol"]].get(d, 0.0) / max_pos
        for t in open_pos:
            if t["exit"] == d:
                daily[d] -= SLIPPAGE / max_pos
    return pd.DataFrame(taken), daily


def metrics(name, trades: pd.DataFrame, daily: pd.Series, equity: float):
    curve = (1 + daily).cumprod()
    years = len(daily) / 252
    cagr = curve.iloc[-1] ** (1 / years) - 1 if years > 0 else 0
    dd = (curve / curve.cummax() - 1).min()
    sharpe = daily.mean() / daily.std() * np.sqrt(252) if daily.std() > 0 else 0
    n = len(trades)
    return dict(strategy=name, trades=n,
                win_rate=f"{(trades['ret'] > 0).mean():.0%}" if n else "-",
                avg_trade=f"{trades['ret'].mean():+.2%}" if n else "-",
                avg_excess=f"{trades['excess'].mean():+.2%}" if n else "-",
                yearly=f"{cagr:+.1%}", max_dd=f"{dd:.0%}", sharpe=round(sharpe, 2),
                end_equity=f"${equity * curve.iloc[-1]:,.0f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2016-01-01")
    ap.add_argument("--end", default="2025-06-30")
    ap.add_argument("--universe", default="largecaps.txt")
    ap.add_argument("--equity", type=float, default=10_000.0)
    ap.add_argument("--probe", action="store_true")
    a = ap.parse_args()
    from dotenv import load_dotenv
    load_dotenv()
    key = os.getenv("FMP_API_KEY", "").strip()
    av_key = os.getenv("ALPHAVANTAGE_API_KEY", "").strip()
    if not key and not av_key:
        raise SystemExit("Add ALPHAVANTAGE_API_KEY=... (recommended) or FMP_API_KEY=... to .env first.")

    if a.probe:
        if av_key:
            r = requests.get("https://www.alphavantage.co/query",
                             params=dict(function="EARNINGS", symbol="AAPL", apikey=av_key), timeout=30)
            df = parse_alphavantage(r.json())
            print(f"Alpha Vantage EARNINGS: rows {len(df)}"
                  + (f" | {df['date'].min().date()} to {df['date'].max().date()}" if len(df)
                     else f" | message: {str(r.json())[:200]}"))
        if not key:
            return
        for path, params in [("/stable/earnings", dict(symbol="AAPL", limit=200)),
                             ("/api/v3/historical/earning_calendar/AAPL", dict(limit=200))]:
            st, data = fmp_get(path, key, **params)
            df = parse_earnings(data)
            print(f"{path}: HTTP {st} | rows with EPS+estimate: {len(df)}"
                  + (f" | {df['date'].min().date()} to {df['date'].max().date()}" if len(df) else
                     f" | message: {str(data)[:200]}"))
        return

    from .broker import ET, Broker
    from .config import Settings
    from .swing import load_daily
    s = Settings()
    s.validate()
    b = Broker(s.api_key, s.api_secret, paper=True, feed="sip")
    syms = [l.strip().upper() for l in open(a.universe) if l.strip() and not l.startswith("#")]
    start = datetime.fromisoformat(a.start).replace(tzinfo=ET)
    end = datetime.fromisoformat(a.end).replace(tzinfo=ET)

    source = "Alpha Vantage" if av_key else "FMP (free plan: last 5 reports only)"
    print(f"Step 1/2: earnings history for {len(syms)} stocks from {source} (cached after first run)...",
          flush=True)
    earn, failed, stopped = {}, [], False
    for k, sym in enumerate(syms, 1):
        try:
            earn[sym] = get_earnings_av(sym, av_key) if av_key else get_earnings(sym, key)
        except DailyLimit as e:
            print(f"  Daily request limit reached after {len(earn)} stocks ({e}).", flush=True)
            stopped = True
            break
        except Exception as e:
            failed.append(sym)
            if len(failed) <= 3:
                print(f"  {sym}: {e}", flush=True)
            if "limit" in str(e).lower():
                stopped = True
                break
        if k % 25 == 0:
            print(f"  {k}/{len(syms)}", flush=True)
    if failed:
        print(f"  {len(failed)} stocks had no earnings data on your plan (skipped).")
    if stopped:
        cached = sum(1 for f in os.listdir(CACHE) if f.startswith("av_" if av_key else ""))
        print(f"  {cached} of {len(syms)} stocks downloaded so far. Run the same command again tomorrow")
        print("  to continue (finished stocks are saved). Results below use only what's downloaded.")
    got = [x for x in earn if len(earn[x])]
    if not got:
        raise SystemExit("No earnings data returned. Run with --probe and tell Claude what it prints.")
    span = pd.concat([earn[x]["date"] for x in got])
    print(f"  earnings for {len(got)} stocks, {span.min().date()} to {span.max().date()}", flush=True)

    print("Step 2/2: daily prices from Alpaca...", flush=True)
    prices = {}
    for i in range(0, len(got), 25):
        prices.update(load_daily(b, got[i:i + 25], start - timedelta(days=30), end + timedelta(days=100)))
    prices.update(load_daily(b, ["SPY"], start - timedelta(days=30), end + timedelta(days=100)))

    events = []
    for sym in got:
        if sym in prices:
            events += [e for e in find_events(sym, earn[sym], prices[sym])
                       if e["report"] <= pd.Timestamp(a.end)]
    print(f"  {len(events)} earnings beats of 10%+ found", flush=True)

    rows = []
    for hold in (20, 60):
        for need_up in (False, True):
            tr, daily = simulate(events, prices, hold, need_up, start.date())
            daily = daily[daily.index <= pd.Timestamp(a.end)]
            rows.append(metrics(f"Beat 10%+, hold {hold}d, {'price up on report' if need_up else 'any reaction'}",
                                tr, daily, a.equity))
    spy = prices["SPY"]
    spy_r = (spy["open"].shift(-1) / spy["open"] - 1).fillna(0.0)
    spy_r = spy_r[(spy_r.index >= pd.Timestamp(start.date())) & (spy_r.index <= pd.Timestamp(a.end))]
    rows.append(metrics("Buy & hold SPY (benchmark)", pd.DataFrame(columns=["ret", "excess"]), spy_r, a.equity))
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    print(f"\n===== POST-EARNINGS DRIFT {a.start} to {a.end} (${a.equity:,.0f} start) =====")
    print(df.to_string(index=False))
    df.to_csv("pead_results.csv", index=False)
    print("\navg_excess = average trade return minus SPY over the same days (the real edge).")
    print("Note: the stock list is TODAY's large caps, which flatters results (survivorship bias).")
    print("Worth paper trading only if avg_excess is clearly positive AND it beats SPY on sharpe.")


if __name__ == "__main__":
    main()
