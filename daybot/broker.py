"""Thin wrapper around Alpaca so the rest of the bot doesn't depend on their SDK details."""
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
from alpaca.data.enums import Adjustment, DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.historical.screener import ScreenerClient
from alpaca.data.historical.news import NewsClient
from alpaca.data.requests import MarketMoversRequest, MostActivesRequest, NewsRequest, StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderClass, OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest, StopLossRequest, TakeProfitRequest

from .strategy import DayContext

ET = ZoneInfo("America/New_York")


def round_price(p: float) -> float:
    """Alpaca tick rules: $1+ -> 2 decimals, under $1 -> 4 decimals."""
    return round(p, 2) if p >= 1 else round(p, 4)


def to_sessions(df: pd.DataFrame) -> dict:
    """Alpaca multi-index bars -> {symbol: {date: regular-session DataFrame}}."""
    out = {}
    if df is None or df.empty:
        return out
    for symbol, sdf in df.groupby(level=0):
        sdf = sdf.droplevel(0).tz_convert(ET)
        sdf = sdf.between_time("09:30", "15:59")[["open", "high", "low", "close", "volume"]]
        out[symbol] = {d: g for d, g in sdf.groupby(sdf.index.date)}
    return out


class Broker:
    def __init__(self, key: str, secret: str, paper: bool = True, feed: str = "iex"):
        self.trading = TradingClient(key, secret, paper=paper)
        self.data = StockHistoricalDataClient(key, secret)
        self.screener = ScreenerClient(key, secret)
        self.news = NewsClient(key, secret)
        self.feed = DataFeed.SIP if feed.lower() == "sip" else DataFeed.IEX

    # ---- account / market state ----
    def account(self) -> dict:
        a = self.trading.get_account()
        bp = min(float(a.buying_power), float(a.non_marginable_buying_power or a.buying_power))
        return {"equity": float(a.equity), "last_equity": float(a.last_equity), "buying_power": bp,
                "cash": float(a.cash)}

    def clock(self):
        return self.trading.get_clock()

    def open_positions(self) -> list:
        return [p.symbol for p in self.trading.get_all_positions()]

    # ---- data ----
    def bars(self, symbols, start: datetime, end: datetime, bar_minutes: int = 5,
             adjustment: str = "raw") -> dict:
        req = StockBarsRequest(symbol_or_symbols=symbols, start=start, end=end,
                               timeframe=TimeFrame(bar_minutes, TimeFrameUnit.Minute), feed=self.feed,
                               adjustment=Adjustment(adjustment))
        return to_sessions(self.data.get_stock_bars(req).df)

    def todays_completed_bars(self, symbols, bar_minutes: int = 5) -> dict:
        now = datetime.now(ET)
        start = now.replace(hour=9, minute=30, second=0, microsecond=0)
        sessions = self.bars(symbols, start, now, bar_minutes)
        out = {}
        for sym, days in sessions.items():
            df = days.get(now.date())
            if df is None:
                continue
            done = df[df.index + timedelta(minutes=bar_minutes) <= now]  # drop the bar still forming
            if len(done):
                out[sym] = done
        return out

    def most_actives(self, top: int = 20) -> list:
        try:
            res = self.screener.get_most_actives(MostActivesRequest(top=top))
            return [m.symbol for m in res.most_actives]
        except Exception as e:
            print(f"screener unavailable: {e}", flush=True)
            return []

    def top_gainers(self, top: int = 20) -> list:
        try:
            res = self.screener.get_market_movers(MarketMoversRequest(top=top))
            return [m.symbol for m in res.gainers]
        except Exception as e:
            print(f"movers unavailable: {e}", flush=True)
            return []

    def daily_context(self, symbols, lookback: int = 20) -> dict:
        """{symbol: DayContext} with yesterday's close and average daily volume."""
        now = datetime.now(ET)
        req = StockBarsRequest(symbol_or_symbols=symbols, start=now - timedelta(days=lookback * 2),
                               end=now, timeframe=TimeFrame.Day, feed=self.feed)
        df = self.data.get_stock_bars(req).df
        out = {}
        if df is None or df.empty:
            return out
        for sym, sdf in df.groupby(level=0):
            sdf = sdf.droplevel(0)
            sdf = sdf[sdf.index.tz_convert(ET).date < now.date()].tail(lookback)
            if len(sdf):
                out[sym] = DayContext(prev_close=float(sdf["close"].iloc[-1]),
                                      avg_daily_volume=float(sdf["volume"].mean()))
        return out

    def has_recent_news(self, symbol: str, hours: int = 24) -> bool:
        try:
            now = datetime.now(ET)
            res = self.news.get_news(NewsRequest(symbols=symbol, start=now - timedelta(hours=hours),
                                                 end=now, limit=5))
            return len(res.data.get("news", [])) > 0
        except Exception as e:
            print(f"news check failed for {symbol}: {e}", flush=True)
            return False

    def daily_bars(self, symbols, start: datetime, end: datetime, adjustment: str = "split"):
        """Raw multi-index daily DataFrame (symbol, timestamp)."""
        req = StockBarsRequest(symbol_or_symbols=symbols, start=start, end=end,
                               timeframe=TimeFrame.Day, feed=self.feed, adjustment=Adjustment(adjustment))
        return self.data.get_stock_bars(req).df

    def listed_symbols(self, include_delisted: bool = True) -> list:
        """Every NYSE/NASDAQ/AMEX common-stock-style symbol, optionally including delisted ones
        (reduces survivorship bias in backtests)."""
        from alpaca.trading.enums import AssetClass, AssetStatus
        from alpaca.trading.requests import GetAssetsRequest
        statuses = [AssetStatus.ACTIVE] + ([AssetStatus.INACTIVE] if include_delisted else [])
        out = set()
        for st in statuses:
            for a in self.trading.get_all_assets(GetAssetsRequest(status=st, asset_class=AssetClass.US_EQUITY)):
                ex = str(getattr(a.exchange, "value", a.exchange))
                sym = a.symbol
                warrant_unit_right = len(sym) == 5 and sym[-1] in "WUR"
                if ex in ("NASDAQ", "NYSE", "AMEX") and sym.isalpha() and len(sym) <= 5 \
                        and not warrant_unit_right:
                    out.add(a.symbol)
        return sorted(out)

    # ---- orders ----
    def buy_bracket(self, symbol: str, qty: int, stop: float, target: float) -> str:
        """Market buy with stop-loss and take-profit held AT THE BROKER (survives bot crashes)."""
        order = MarketOrderRequest(
            symbol=symbol, qty=qty, side=OrderSide.BUY, time_in_force=TimeInForce.DAY,
            order_class=OrderClass.BRACKET,
            take_profit=TakeProfitRequest(limit_price=round_price(target)),
            stop_loss=StopLossRequest(stop_price=round_price(stop)))
        return str(self.trading.submit_order(order_data=order).id)

    def buy_notional(self, symbol: str, dollars: float) -> str:
        """Fractional market buy for a dollar amount (Alpaca requires DAY for fractional)."""
        order = MarketOrderRequest(symbol=symbol, notional=round(dollars, 2), side=OrderSide.BUY,
                                   time_in_force=TimeInForce.DAY)
        return str(self.trading.submit_order(order_data=order).id)

    def position_summary(self, symbol: str):
        try:
            p = self.trading.get_open_position(symbol)
            return float(p.qty), float(p.market_value), float(p.unrealized_pl)
        except Exception:
            return 0.0, 0.0, 0.0

    def flatten_all(self):
        self.trading.close_all_positions(cancel_orders=True)
