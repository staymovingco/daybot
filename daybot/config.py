"""All settings come from the .env file. Defaults are conservative."""
import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()


def _f(name, default):
    return float(os.getenv(name, default))


def _i(name, default):
    return int(os.getenv(name, default))


@dataclass
class StrategyParams:
    bar_minutes: int = 5
    or_minutes: int = 15            # opening range = first 15 minutes
    rvol_min: float = 1.5           # breakout bar volume vs. average of earlier bars today
    reward_risk: float = 2.0        # target = 2x the risk
    last_entry_time: str = "11:30"  # no new trades after this (ET)
    flatten_time: str = "15:50"     # close everything by this (ET)
    min_stop_pct: float = 0.005     # stop at least 0.5% away
    max_stop_pct: float = 0.06      # skip if stop would be more than 6% away
    min_or_dollar_volume: float = 250_000  # liquidity filter on the opening range


@dataclass
class GapParams(StrategyParams):
    """Small-cap gap-and-go (Ross Cameron style), adapted to 5-min bars."""
    gap_min: float = 0.10                 # opened 10%+ above yesterday's close
    rvol_min: float = 3.0                 # volume so far vs. normal pace for this time of day
    last_entry_time: str = "10:30"        # momentum fades; first hour only
    min_stop_pct: float = 0.01
    max_stop_pct: float = 0.08
    min_first_bar_dollar_volume: float = 500_000
    require_news: bool = False            # True = only trade gappers with news in last 24h


@dataclass
class RiskConfig:
    risk_pct: float = 0.01              # risk 1% of equity per trade
    max_trades_per_day: int = 2
    daily_loss_limit_pct: float = 0.02  # lock trading for the day at -2%
    max_open_positions: int = 1
    min_price: float = 1.0              # $1+ only (sub-$1 = sub-penny ticks, wide spreads)
    max_price: float = 20.0
    cash_buffer: float = 0.95           # never use more than 95% of buying power
    max_drawdown_pct: float = 0.15      # stop bot entirely 15% below peak equity


@dataclass
class Settings:
    api_key: str = field(default_factory=lambda: os.getenv("ALPACA_API_KEY", ""))
    api_secret: str = field(default_factory=lambda: os.getenv("ALPACA_API_SECRET", ""))
    mode: str = field(default_factory=lambda: os.getenv("TRADING_MODE", "paper").lower())
    live_confirm: str = field(default_factory=lambda: os.getenv("LIVE_CONFIRM", ""))
    data_feed: str = field(default_factory=lambda: os.getenv("DATA_FEED", "iex"))
    watchlist: list = field(default_factory=lambda: [
        s.strip().upper() for s in os.getenv("WATCHLIST", "SOFI,PLTR,F,NIO,RIVN,LCID,SNAP,AAL").split(",") if s.strip()
    ])
    use_screener: bool = field(default_factory=lambda: os.getenv("USE_SCREENER", "true").lower() == "true")
    telegram_token: str = field(default_factory=lambda: os.getenv("TELEGRAM_BOT_TOKEN", ""))
    telegram_chat_id: str = field(default_factory=lambda: os.getenv("TELEGRAM_CHAT_ID", ""))
    db_path: str = field(default_factory=lambda: os.getenv("DB_PATH", "journal.db"))
    strategies: list = field(default_factory=lambda: [
        s.strip().lower() for s in os.getenv("STRATEGIES", "orb,gap").split(",") if s.strip()])
    orb: StrategyParams = field(default_factory=lambda: StrategyParams(
        rvol_min=_f("RVOL_MIN", 1.5), reward_risk=_f("REWARD_RISK", 2.0)))
    gap: GapParams = field(default_factory=lambda: GapParams(
        gap_min=_f("GAP_MIN", 0.10), rvol_min=_f("GAP_RVOL_MIN", 3.0),
        reward_risk=_f("GAP_REWARD_RISK", 2.0),
        require_news=os.getenv("GAP_REQUIRE_NEWS", "false").lower() == "true"))

    def params(self, name: str) -> StrategyParams:
        return {"orb": self.orb, "gap": self.gap}[name]
    risk: RiskConfig = field(default_factory=lambda: RiskConfig(
        risk_pct=_f("RISK_PCT", 0.01),
        max_trades_per_day=_i("MAX_TRADES_PER_DAY", 2),
        daily_loss_limit_pct=_f("DAILY_LOSS_LIMIT_PCT", 0.02),
        min_price=_f("MIN_PRICE", 1.0),
        max_price=_f("MAX_PRICE", 20.0)))

    @property
    def paper(self) -> bool:
        return self.mode != "live"

    def validate(self):
        if not self.api_key or not self.api_secret:
            raise SystemExit("Missing ALPACA_API_KEY / ALPACA_API_SECRET in .env")
        if self.mode == "live" and self.live_confirm != "I_ACCEPT_THE_RISK":
            raise SystemExit("Live mode needs LIVE_CONFIRM=I_ACCEPT_THE_RISK in .env")
