"""Risk manager: sizing and the rules that can veto any trade."""
import math
from dataclasses import dataclass

from .config import RiskConfig


@dataclass
class DayState:
    start_equity: float
    peak_equity: float
    trades_today: int = 0
    locked: bool = False
    lock_reason: str = ""


class RiskManager:
    def __init__(self, cfg: RiskConfig):
        self.cfg = cfg

    def position_size(self, equity: float, buying_power: float, entry: float, stop: float) -> int:
        """Whole shares so that a stop-out loses ~risk_pct of equity, capped by cash."""
        per_share_risk = entry - stop
        if per_share_risk <= 0 or entry <= 0:
            return 0
        by_risk = (equity * self.cfg.risk_pct) / per_share_risk
        by_cash = (buying_power * self.cfg.cash_buffer) / entry
        return max(0, math.floor(min(by_risk, by_cash)))

    def price_ok(self, price: float) -> bool:
        return self.cfg.min_price <= price <= self.cfg.max_price

    def update(self, state: DayState, equity: float) -> DayState:
        """Check loss limits against current equity; lock if breached."""
        state.peak_equity = max(state.peak_equity, equity)
        day_pnl = (equity - state.start_equity) / state.start_equity
        if day_pnl <= -self.cfg.daily_loss_limit_pct and not state.locked:
            state.locked, state.lock_reason = True, f"daily loss limit hit ({day_pnl:.1%})"
        dd = (equity - state.peak_equity) / state.peak_equity
        if dd <= -self.cfg.max_drawdown_pct:
            state.locked, state.lock_reason = True, f"max drawdown hit ({dd:.1%}) - back to paper"
        return state

    def can_open(self, state: DayState, open_positions: int) -> tuple[bool, str]:
        if state.locked:
            return False, state.lock_reason
        if state.trades_today >= self.cfg.max_trades_per_day:
            return False, "max trades for today"
        if open_positions >= self.cfg.max_open_positions:
            return False, "max open positions"
        return True, "ok"
