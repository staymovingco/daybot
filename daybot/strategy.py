"""Trading setups. The same functions drive the backtester and the live bot,
so what you test is what trades.

Bars: one symbol, one regular session, tz-aware America/New_York index (bar start time),
columns open/high/low/close/volume.

Every setup has the signature:  fn(symbol, bars, ctx, params) -> Signal | None
"""
from dataclasses import dataclass
from datetime import time
from typing import Optional

import pandas as pd

from .config import GapParams, StrategyParams

SESSION_MINUTES = 390


@dataclass
class Signal:
    symbol: str
    time: pd.Timestamp   # start time of the bar that triggered
    entry: float         # trigger bar close (live fills near this; backtest uses next open)
    stop: float
    target: float
    reason: str
    strategy: str = "orb"


@dataclass
class DayContext:
    """Info from before today's open."""
    prev_close: Optional[float] = None
    avg_daily_volume: Optional[float] = None
    has_news: Optional[bool] = None   # None = unknown / not checked


def _t(s: str) -> time:
    h, m = s.split(":")
    return time(int(h), int(m))


def vwap(bars: pd.DataFrame) -> pd.Series:
    typical = (bars["high"] + bars["low"] + bars["close"]) / 3
    return (typical * bars["volume"]).cumsum() / bars["volume"].cumsum()


def _clamp_stop(entry, stop, p):
    """Enforce min/max stop distance. Returns adjusted stop, or None if too wide."""
    risk_pct = (entry - stop) / entry
    if risk_pct < p.min_stop_pct:
        return entry * (1 - p.min_stop_pct)
    if risk_pct > p.max_stop_pct:
        return None
    return stop


# ---------------------------------------------------------------- 1) Opening Range Breakout
def find_orb_signal(symbol: str, bars: pd.DataFrame, p: StrategyParams,
                    ctx: Optional[DayContext] = None) -> Optional[Signal]:
    """Buy the FIRST close above the opening range, if above VWAP on strong volume."""
    n_or = p.or_minutes // p.bar_minutes
    if len(bars) <= n_or:
        return None
    or_bars = bars.iloc[:n_or]
    or_high, or_low = or_bars["high"].max(), or_bars["low"].min()
    if (or_bars["close"] * or_bars["volume"]).sum() < p.min_or_dollar_volume:
        return None

    vw = vwap(bars)
    last_entry = _t(p.last_entry_time)
    for i in range(n_or, len(bars)):
        ts = bars.index[i]
        if ts.time() > last_entry:
            return None
        bar = bars.iloc[i]
        avg_prior_vol = bars["volume"].iloc[:i].mean()
        if bar["close"] <= or_high:
            continue
        # First close above the range: take it or skip the day (no chasing later breaks)
        if bar["close"] <= vw.iloc[i] or bar["volume"] < p.rvol_min * avg_prior_vol:
            return None
        entry = float(bar["close"])
        stop = _clamp_stop(entry, float((or_high + or_low) / 2), p)
        if stop is None:
            return None
        return Signal(symbol, ts, entry, stop, entry + p.reward_risk * (entry - stop),
                      f"ORB break {or_high:.2f}, above VWAP, rvol {bar['volume']/avg_prior_vol:.1f}x", "orb")
    return None


# ---------------------------------------------------------------- 2) Small-cap Gap-and-Go
def find_gap_signal(symbol: str, bars: pd.DataFrame, p: GapParams,
                    ctx: Optional[DayContext] = None) -> Optional[Signal]:
    """Stock gaps up big on heavy volume. Buy when it breaks to a new high of day
    above VWAP. Stop under the recent pullback low."""
    if ctx is None or not ctx.prev_close or not ctx.avg_daily_volume or len(bars) < 2:
        return None
    if p.require_news and ctx.has_news is False:
        return None
    first = bars.iloc[0]
    gap = first["open"] / ctx.prev_close - 1
    if gap < p.gap_min:
        return None
    if first["close"] * first["volume"] < p.min_first_bar_dollar_volume:
        return None
    if first["close"] <= ctx.prev_close:   # gap already fully faded
        return None

    vw = vwap(bars)
    last_entry = _t(p.last_entry_time)
    hod = float(first["high"])
    for i in range(1, len(bars)):
        ts, bar = bars.index[i], bars.iloc[i]
        if ts.time() > last_entry:
            return None
        if bar["close"] > hod:
            elapsed = (i + 1) * p.bar_minutes / SESSION_MINUTES
            rvol = bars["volume"].iloc[:i + 1].sum() / (ctx.avg_daily_volume * elapsed)
            if bar["close"] > vw.iloc[i] and rvol >= p.rvol_min:
                entry = float(bar["close"])
                pullback_low = float(bars["low"].iloc[max(0, i - 3):i].min())
                stop = _clamp_stop(entry, pullback_low, p)
                if stop is not None:
                    return Signal(symbol, ts, entry, stop, entry + p.reward_risk * (entry - stop),
                                  f"Gap {gap:+.0%}, new HOD {hod:.2f}, rvol {rvol:.1f}x, above VWAP", "gap")
        hod = max(hod, float(bar["high"]))
    return None


STRATEGIES = {"orb": find_orb_signal, "gap": find_gap_signal}


def context_from_history(sessions_for_symbol: dict, day, lookback: int = 20) -> DayContext:
    """Build DayContext from earlier regular-session bars (used by the backtester)."""
    prior = sorted(d for d in sessions_for_symbol if d < day)[-lookback:]
    if not prior:
        return DayContext()
    vols = [sessions_for_symbol[d]["volume"].sum() for d in prior]
    return DayContext(prev_close=float(sessions_for_symbol[prior[-1]]["close"].iloc[-1]),
                      avg_daily_volume=float(sum(vols) / len(vols)))
