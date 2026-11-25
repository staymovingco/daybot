import numpy as np
import pandas as pd

from daybot.config import RiskConfig, StrategyParams
from daybot.risk import DayState, RiskManager
from daybot.strategy import find_orb_signal
from daybot.backtest import simulate_symbol_day, run_portfolio

ET = "America/New_York"


def make_day(date="2025-03-03", breakout=True, run_to=None, crash=False):
    idx = pd.date_range(f"{date} 09:30", f"{date} 15:55", freq="5min", tz=ET)
    n = len(idx)
    close = np.full(n, 10.0)
    close[:3] = [10.0, 10.1, 9.9]            # opening range ~9.85-10.15
    vol = np.full(n, 100_000.0)
    if breakout:
        close[3:] = 10.0
        close[6] = 10.40                      # breakout bar at 10:00
        vol[6] = 400_000
        after = run_to if run_to else 10.40
        close[7:] = np.linspace(10.45, after, n - 7)
        if crash:
            close[7:] = np.linspace(10.2, 9.0, n - 7)
    df = pd.DataFrame({"open": close, "close": close, "volume": vol}, index=idx)
    df["open"] = df["close"].shift(1).fillna(df["close"])
    df["high"] = df[["open", "close"]].max(axis=1) + 0.05
    df["low"] = df[["open", "close"]].min(axis=1) - 0.05
    return df


def test_signal_found():
    sig = find_orb_signal("TEST", make_day(), StrategyParams())
    assert sig is not None
    assert sig.entry == 10.40 and sig.stop < sig.entry < sig.target


def test_no_signal_without_breakout():
    assert find_orb_signal("TEST", make_day(breakout=False), StrategyParams()) is None


def test_no_signal_on_weak_volume():
    df = make_day()
    df.iloc[6, df.columns.get_loc("volume")] = 100_000
    assert find_orb_signal("TEST", df, StrategyParams()) is None


def test_sizing_risks_one_percent():
    rm = RiskManager(RiskConfig())
    qty = rm.position_size(equity=1000, buying_power=1000, entry=10.0, stop=9.5)
    assert qty == 20  # $10 risk / $0.50


def test_sizing_capped_by_cash():
    rm = RiskManager(RiskConfig())
    assert rm.position_size(equity=100, buying_power=100, entry=10.0, stop=9.99) == 9


def test_daily_loss_lock():
    rm = RiskManager(RiskConfig())
    st = DayState(start_equity=100, peak_equity=100)
    rm.update(st, 97.9)
    assert st.locked and not rm.can_open(st, 0)[0]


def test_backtest_target_and_stop():
    p = StrategyParams()
    win = simulate_symbol_day("W", make_day(run_to=12.0), p)
    loss = simulate_symbol_day("L", make_day(crash=True), p)
    assert win.exit_reason == "target" and win.r > 1.5
    assert loss.exit_reason == "stop" and loss.r < 0


def test_portfolio_compounds():
    sessions = {"W": {pd.Timestamp("2025-03-03").date(): make_day(run_to=12.0)}}
    trades, curve, dd = run_portfolio(sessions, StrategyParams(), RiskConfig(), 1000)
    assert len(trades) == 1 and curve[-1][1] > 1000


# ---------------- gap-and-go ----------------
from daybot.config import GapParams
from daybot.strategy import DayContext, find_gap_signal, context_from_history


def make_gap_day(date="2025-03-04", open_px=6.0, then=None, vol0=400_000):
    """Opens at open_px, small pullback, then pushes to new highs (unless `then` given)."""
    idx = pd.date_range(f"{date} 09:30", f"{date} 15:55", freq="5min", tz=ET)
    n = len(idx)
    path = [open_px + 0.20, open_px + 0.05, open_px + 0.10, open_px + 0.45]   # bar3 = new HOD
    tail = then if then is not None else np.linspace(open_px + 0.5, open_px + 1.5, n - 4)
    close = np.array(path + list(tail))
    df = pd.DataFrame({"close": close}, index=idx)
    df["open"] = df["close"].shift(1).fillna(open_px)
    df["high"] = df[["open", "close"]].max(axis=1) + 0.03
    df["low"] = df[["open", "close"]].min(axis=1) - 0.03
    df["volume"] = [vol0, 200_000, 150_000, 300_000] + [80_000] * (n - 4)
    return df[["open", "high", "low", "close", "volume"]]


CTX = DayContext(prev_close=5.0, avg_daily_volume=2_000_000)


def test_gap_signal_found():
    sig = find_gap_signal("GAP", make_gap_day(), GapParams(), CTX)
    assert sig is not None and sig.strategy == "gap"
    assert sig.stop < sig.entry < sig.target
    assert sig.time == make_gap_day().index[3]


def test_gap_too_small():
    assert find_gap_signal("GAP", make_gap_day(open_px=5.2), GapParams(), CTX) is None


def test_gap_needs_relative_volume():
    quiet = DayContext(prev_close=5.0, avg_daily_volume=50_000_000)
    assert find_gap_signal("GAP", make_gap_day(), GapParams(), quiet) is None


def test_gap_news_filter():
    no_news = DayContext(prev_close=5.0, avg_daily_volume=2_000_000, has_news=False)
    assert find_gap_signal("GAP", make_gap_day(), GapParams(require_news=True), no_news) is None
    assert find_gap_signal("GAP", make_gap_day(), GapParams(require_news=False), no_news) is not None


def test_context_from_history():
    d1, d2 = pd.Timestamp("2025-03-03").date(), pd.Timestamp("2025-03-04").date()
    hist = {d1: make_day(), d2: make_gap_day()}
    ctx = context_from_history(hist, d2)
    assert ctx.prev_close == float(make_day()["close"].iloc[-1])
    assert ctx.avg_daily_volume == make_day()["volume"].sum()


def test_portfolio_both_strategies():
    d1, d2 = pd.Timestamp("2025-03-03").date(), pd.Timestamp("2025-03-04").date()
    # day1 ends ~5.0 so day2 opening at 6.0 is a 20% gap
    base = make_day(breakout=False)
    base[["open", "high", "low", "close"]] *= 0.5
    base["volume"] = 2_000_000 / len(base)
    sessions = {"GAP": {d1: base, d2: make_gap_day()}}
    strats = [("orb", StrategyParams()), ("gap", GapParams())]
    trades, curve, dd = run_portfolio(sessions, strats, RiskConfig(), 1000)
    assert any(t.strategy == "gap" for t in trades)


# ---------------- variants + gap scan ----------------
from dataclasses import replace
from daybot.gapscan import find_gap_days


def test_orb_stop_modes():
    mid = find_orb_signal("T", make_day(), StrategyParams(stop_mode="mid"))
    low = find_orb_signal("T", make_day(), StrategyParams(stop_mode="low"))
    assert low.stop < mid.stop and low.target > mid.target


def test_orb_30_minute_range():
    sig = find_orb_signal("T", make_day(), StrategyParams(or_minutes=30))
    assert sig is not None and sig.time == make_day().index[6]


def test_market_filter():
    day = make_day()
    p = StrategyParams(market_filter=True)
    assert find_orb_signal("T", day, p, DayContext()) is None          # no SPY data -> no trade
    spy_up = make_day(run_to=12.0)                                       # SPY above VWAP at 10:00
    assert find_orb_signal("T", day, p, DayContext(market_bars=spy_up)) is not None
    spy_down = make_day(crash=True)
    spy_down["close"] = np.linspace(10, 9, len(spy_down))                # SPY sliding all day
    spy_down["open"] = spy_down["close"]
    assert find_orb_signal("T", day, p, DayContext(market_bars=spy_down)) is None


def test_find_gap_days():
    idx = pd.date_range("2025-01-01", periods=30, freq="B").date
    d = pd.DataFrame({"open": 5.0, "close": 5.0, "volume": 1e6}, index=idx)
    d.loc[idx[25], "open"] = 6.0          # 20% gap
    d.loc[idx[27], "open"] = 30.0         # big gap but above $20
    ev = find_gap_days(d, 0.10, 1, 20)
    assert list(ev.index) == [idx[25]]
    assert ev.iloc[0]["prev_close"] == 5.0 and abs(ev.iloc[0]["gap"] - 0.2) < 1e-9


# ---------------- swing ----------------
from daybot import swing


def make_daily(n=600, seed=1, drift=0.0004):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2018-01-01", periods=n)
    c = 100 * np.exp(np.cumsum(drift + 0.01 * rng.standard_normal(n)))
    o = np.r_[c[0], c[:-1]] * (1 + 0.001 * rng.standard_normal(n))
    return pd.DataFrame({"open": o, "high": np.maximum(o, c) * 1.002, "low": np.minimum(o, c) * 0.998,
                         "close": c, "volume": 1e6}, index=idx)


def test_rsi_bounds():
    r = swing.rsi(make_daily()["close"])
    assert r.between(0, 100).all()


def test_rsi2_no_lookahead_and_trades():
    df = make_daily()
    pos, trades = swing.rsi2_positions(df)
    assert pos.iloc[:200].sum() == 0          # needs 200 days of history before any trade
    assert len(trades) > 0 and all(x > e for e, x in trades)
    assert all(x - e <= 10 for e, x in trades)  # max hold respected


def test_swing_run_table():
    data = {s: make_daily(seed=i) for i, s in enumerate(swing.ETFS)}
    out = swing.run(data, data["SPY"].index[250].date(), 100)
    assert len(out) == 4 and "Buy & hold SPY (benchmark)" in set(out["strategy"])


def test_buy_and_hold_matches_price():
    df = make_daily()
    pos = pd.Series(1.0, index=df.index)
    curve, _ = swing.portfolio({"SPY": df}, {"SPY": pos}, 1.0)
    expected = df["open"].iloc[-1] / df["open"].iloc[0] * (1 - swing.SLIPPAGE)
    assert abs(curve.iloc[-1] - expected) / expected < 1e-3


# ---------------- auto-invest ----------------
from daybot.autoinvest import plan_buy


def test_plan_buy():
    assert plan_buy(100, 0, 500, 1, "", "2026-W48") == 100
    assert plan_buy(100, 0, 500, 1, "2026-W48", "2026-W48") == 0      # once per week
    assert plan_buy(900, 0, 500, 1, "", "2026-W48") == 500            # weekly cap
    assert plan_buy(30, 10, 500, 1, "", "2026-W48") == 20             # keeps reserve
    assert plan_buy(0.5, 0, 500, 1, "", "2026-W48") == 0              # below minimum


# ---------------- gap fade (short) ----------------
from daybot.fade import FadeParams, simulate_fade
from daybot.backtest import run_trades as _run_trades, Trade as _Trade


def make_fade_day(date="2025-03-05", fade=True):
    """Opens at 6.00 (20% gap over 5.00), pops to ~6.4, then fades below VWAP."""
    idx = pd.date_range(f"{date} 09:30", f"{date} 15:55", freq="5min", tz=ET)
    n = len(idx)
    up = [6.1, 6.3, 6.4, 6.35]                      # 9:30-9:45
    down = list(np.linspace(6.0, 5.2, n - 4)) if fade else list(np.linspace(6.45, 7.5, n - 4))
    close = np.array(up + down)
    df = pd.DataFrame({"close": close}, index=idx)
    df["open"] = df["close"].shift(1).fillna(6.0)
    df["high"] = df[["open", "close"]].max(axis=1) + 0.02
    df["low"] = df[["open", "close"]].min(axis=1) - 0.02
    df["volume"] = [500_000, 300_000, 250_000, 200_000] + [100_000] * (n - 4)
    return df[["open", "high", "low", "close", "volume"]]


def test_fade_short_wins_on_fade():
    tr = simulate_fade("F", make_fade_day(), 5.0, FadeParams(reward_risk=None))
    assert tr is not None and tr.side == "short" and tr.r > 0 and tr.exit < tr.entry


def test_fade_no_trade_when_it_runs():
    assert simulate_fade("F", make_fade_day(fade=False), 5.0, FadeParams()) is None


def test_fade_small_gap_skipped():
    assert simulate_fade("F", make_fade_day(), 5.9, FadeParams(gap_min=0.10)) is None


def test_short_pnl_and_pdt():
    t0 = pd.Timestamp("2025-03-03 10:00", tz=ET)
    mk = lambda d: _Trade("S", pd.Timestamp(d).date(), pd.Timestamp(f"{d} 10:00", tz=ET),
                          pd.Timestamp(f"{d} 11:00", tz=ET), 10.0, 10.5, 9.0, 2.0, "target", "fade", "short")
    trades = [mk(d) for d in ["2025-03-03", "2025-03-04", "2025-03-05", "2025-03-06", "2025-03-07"]]
    taken, curve, _ = _run_trades(trades, RiskConfig(), 10_000)
    assert len(taken) == 5 and curve[-1][1] > 10_000          # shorts profit when price falls
    taken_pdt, _, _ = _run_trades(trades, RiskConfig(), 10_000, pdt=True)
    assert len(taken_pdt) == 3                                  # 3 day trades per 5 days


# ---------------- sector rotation ----------------
from daybot import rotation


def _rot_data(n=700):
    data = {}
    for i, s in enumerate(rotation.SECTORS + ["SPY", rotation.CASH]):
        drift = 0.0001 * i if s != rotation.CASH else 0.00015
        data[s] = make_daily(n=n, seed=i, drift=drift)
    return data


def test_rotation_picks_strongest_and_cash_when_trend_down():
    data = _rot_data()
    closes = pd.DataFrame({s: d["close"] for s, d in data.items()})
    w = rotation.target_weights(closes, 600, 252, 3, trend_filter=False)
    assert abs(w.sum() - 1) < 1e-9 and (w > 0).sum() == 3 and w[rotation.CASH] == 0
    closes2 = closes.copy()
    closes2.loc[closes2.index[590:601], "SPY"] = closes2["SPY"].iloc[:600].min() * 0.5   # crash
    w2 = rotation.target_weights(closes2, 600, 252, 3, trend_filter=True)
    assert w2[rotation.CASH] == 1.0


def test_rotation_run_table():
    data = _rot_data()
    out = rotation.run(data, data["SPY"].index[300].date(), 10_000)
    assert len(out) == 5 and out["rebalances"].iloc[0] > 10


def test_month_ends():
    idx = pd.bdate_range("2025-01-01", "2025-03-31")
    me = rotation.month_ends(idx)
    assert [idx[i].month for i in me] == [1, 2]


# ---------------- post-earnings drift ----------------
from daybot import pead


def test_parse_earnings_both_shapes():
    stable = [{"date": "2024-05-02", "epsActual": 1.5, "epsEstimated": 1.3},
              {"date": "2024-08-01", "epsActual": None, "epsEstimated": 1.4}]
    v3 = [{"date": "2024-05-02", "eps": 1.5, "epsEstimated": 1.3}]
    assert len(pead.parse_earnings(stable)) == 1 and len(pead.parse_earnings(v3)) == 1
    assert len(pead.parse_earnings({"Error Message": "x"})) == 0


def test_find_events_and_simulate():
    px = make_daily(n=400, seed=3)
    spy = make_daily(n=400, seed=4)
    earn = pd.DataFrame({"date": [px.index[100], px.index[200], px.index[300]],
                         "eps": [1.3, 1.0, 1.0], "est": [1.0, 1.0, 0.01]})
    ev = pead.find_events("X", earn, px)
    assert len(ev) == 1 and ev[0]["entry"] == 102            # only the 30% beat; tiny estimate skipped
    tr, daily = pead.simulate(ev, {"X": px, "SPY": spy}, 20, False, px.index[0].date())
    assert len(tr) == 1 and tr.iloc[0]["exit"] == px.index[122]
    assert abs(tr.iloc[0]["ret"] - (px["open"].iloc[122] / px["open"].iloc[102] - 1 - 2 * pead.SLIPPAGE)) < 1e-9


def test_parse_alphavantage():
    data = {"quarterlyEarnings": [
        {"reportedDate": "2024-05-02", "reportedEPS": "1.53", "estimatedEPS": "1.5"},
        {"reportedDate": "2024-08-01", "reportedEPS": "None", "estimatedEPS": "1.4"}]}
    df = pead.parse_alphavantage(data)
    assert len(df) == 1 and df.iloc[0]["eps"] == 1.53
    assert len(pead.parse_alphavantage({"Information": "rate limit"})) == 0


# ---------------- options demo ----------------
from types import SimpleNamespace
from datetime import date as _date, timedelta as _td
from daybot.options_demo import pick_contract, explain


def test_pick_contract():
    today = _date(2026, 11, 24)
    mk = lambda strike, days, tradable=True: SimpleNamespace(strike_price=str(strike), tradable=tradable,
                                                             expiration_date=today + _td(days=days))
    cs = [mk(600, 37), mk(605, 37), mk(601, 20), mk(600, 60), mk(602, 35, tradable=False)]
    c = pick_contract(cs, 602.0, today)
    assert float(c.strike_price) == 600 and (c.expiration_date - today).days == 37


def test_explain_text():
    out = explain("SPY261231C00600000", 1, 15.0, 12.0, -300, 598.0, 600.0, _date(2026, 12, 31),
                  _date(2026, 11, 24))
    assert "break-even at expiry: $615.00" in out and "-300" in out and "time value" in out


def test_options_verdict():
    from daybot.options_demo import verdict
    assert verdict(0.55, 30)[0] == "SELL" and "take-profit" in verdict(0.55, 30)[1]
    assert verdict(-0.6, 30)[0] == "SELL" and "stop-loss" in verdict(-0.6, 30)[1]
    assert verdict(0.1, 14)[0] == "SELL" and "time stop" in verdict(0.1, 14)[1]
    assert verdict(0.1, 30)[0] == "HOLD"


# ---------------- strangle demo ----------------
from daybot.strangle_demo import pick_strangle, explain as strangle_explain


def test_pick_strangle():
    today = _date(2026, 11, 25)
    mk = lambda strike, days: SimpleNamespace(strike_price=str(strike), tradable=True,
                                              expiration_date=today + _td(days=days))
    calls = [mk(s, d) for s in (715, 720, 725, 730) for d in (36, 44)]
    puts = [mk(s, d) for s in (670, 675, 680, 685) for d in (36, 44)]
    c, p = pick_strangle(calls, puts, 700.0, today)
    assert float(c.strike_price) == 720 and float(p.strike_price) == 680
    assert c.expiration_date == p.expiration_date == today + _td(days=36)


def test_strangle_explain():
    out = strangle_explain(700.0, 721.0, 679.0, 20.0, 18.0, -200, _date(2026, 12, 31), _date(2026, 11, 25))
    assert "above $741.00" in out and "below $659.00" in out and "-200" in out
