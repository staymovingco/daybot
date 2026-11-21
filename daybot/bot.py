"""Live / paper trading loop. Run:  python -m daybot.bot

Kill switch: create a file named KILL in the working folder -> flattens everything and exits.
"""
import os
import time as _time
from datetime import datetime

from .broker import ET, Broker
from .config import Settings
from .journal import Alerts, Journal
from .risk import DayState, RiskManager
from .strategy import STRATEGIES, _t

LOOP_SECONDS = 30


class Bot:
    def __init__(self, s: Settings):
        s.validate()
        self.s = s
        self.broker = Broker(s.api_key, s.api_secret, paper=s.paper, feed=s.data_feed)
        self.risk = RiskManager(s.risk)
        self.journal = Journal(s.db_path)
        self.alerts = Alerts(s.telegram_token, s.telegram_chat_id)
        self.day = None
        self.state: DayState | None = None
        self.universe: list = []
        self.contexts: dict = {}
        self.news_cache: dict = {}
        self.flattened = False

    # ---------- daily setup ----------
    def start_day(self, today: str):
        acct = self.broker.account()
        peak = max(float(self.journal.get_meta("peak_equity", acct["equity"])), acct["equity"])
        self.state = DayState(start_equity=acct["last_equity"], peak_equity=peak,
                              trades_today=self.journal.trades_on(today))
        extra = []
        if self.s.use_screener:
            extra += self.broker.most_actives(25)
            if "gap" in self.s.strategies:
                extra += self.broker.top_gainers(20)
        self.universe = list(dict.fromkeys(self.s.watchlist + extra))
        self.contexts = self.broker.daily_context(self.universe)
        self.news_cache = {}
        self.day, self.flattened = today, False
        self.alerts.send(f"[{'PAPER' if self.s.paper else 'LIVE'}] Day start {today} | equity "
                         f"${acct['equity']:.2f} | watching {len(self.universe)} symbols")

    def end_day(self):
        acct = self.broker.account()
        self.journal.save_day(self.day, self.state.start_equity, acct["equity"],
                              self.state.trades_today, self.state.lock_reason)
        self.journal.set_meta("peak_equity", self.state.peak_equity)
        pnl = acct["equity"] - self.state.start_equity
        self.alerts.send(f"Day end {self.day} | P&L ${pnl:+.2f} ({pnl/self.state.start_equity:+.2%}) | "
                         f"equity ${acct['equity']:.2f} | trades {self.state.trades_today}")

    # ---------- one pass ----------
    def tick(self, now: datetime):
        p = self.s.orb  # shared timing settings (flatten time)
        acct = self.broker.account()
        was_locked = self.state.locked
        self.risk.update(self.state, acct["equity"])
        if self.state.locked and not was_locked:
            self.broker.flatten_all()
            self.alerts.send(f"LOCKED: {self.state.lock_reason}. Flattened. No more trades today.")

        if now.time() >= _t(p.flatten_time):
            if not self.flattened:
                self.broker.flatten_all()
                self.flattened = True
                self.alerts.send("End-of-day flatten done.")
            return
        latest_entry = max(_t(self.s.params(n).last_entry_time) for n in self.s.strategies)
        if now.time() > latest_entry:
            return

        open_pos = self.broker.open_positions()
        ok, why = self.risk.can_open(self.state, len(open_pos))
        if not ok:
            return

        traded = self.journal.symbols_traded_on(self.day)
        candidates = [x for x in self.universe if x not in traded and x not in open_pos]
        if not candidates:
            return
        bars = self.broker.todays_completed_bars(candidates, p.bar_minutes)
        for sym, df in bars.items():
            sig = self._signal(sym, df)
            if not sig:
                continue
            qty = self.risk.position_size(acct["equity"], acct["buying_power"], sig.entry, sig.stop)
            if qty < 1:
                continue
            try:
                oid = self.broker.buy_bracket(sym, qty, sig.stop, sig.target)
            except Exception as e:
                self.alerts.send(f"Order failed {sym}: {e}")
                continue
            self.state.trades_today += 1
            self.journal.log_trade(self.day, sym, qty, sig.entry, sig.stop, sig.target, oid,
                                   f"[{sig.strategy}] {sig.reason}")
            self.alerts.send(f"[{sig.strategy.upper()}] BUY {qty} {sym} ~${sig.entry:.2f} | stop ${sig.stop:.2f} | "
                             f"target ${sig.target:.2f} | risk ${qty*(sig.entry-sig.stop):.2f}\n{sig.reason}")
            break  # one entry per pass; risk rules re-checked next pass

    def _signal(self, sym, df):
        """First enabled setup that fires on the most recent completed bar."""
        ctx = self.contexts.get(sym)
        for name in self.s.strategies:
            params = self.s.params(name)
            if name == "gap" and ctx is not None and params.require_news:
                if sym not in self.news_cache:
                    self.news_cache[sym] = self.broker.has_recent_news(sym)
                ctx.has_news = self.news_cache[sym]
            sig = STRATEGIES[name](sym, df, params, ctx)
            # Only act if the trigger is the latest completed bar (no stale entries)
            if sig and sig.time == df.index[-1] and self.risk.price_ok(sig.entry):
                return sig
        return None

    # ---------- main loop ----------
    def run(self):
        self.alerts.send(f"Bot started in {'PAPER' if self.s.paper else 'LIVE'} mode.")
        while True:
            try:
                if os.path.exists("KILL"):
                    self.broker.flatten_all()
                    self.alerts.send("KILL file found: flattened everything, bot stopped.")
                    return
                clock = self.broker.clock()
                now = datetime.now(ET)
                today = now.date().isoformat()
                if clock.is_open:
                    if self.day != today:
                        self.start_day(today)
                    self.tick(now)
                elif self.day == today and self.state is not None:
                    self.end_day()
                    self.day = None
                    self.state = None
                _time.sleep(LOOP_SECONDS if clock.is_open else 300)
            except KeyboardInterrupt:
                print("Stopped by user (positions left as-is; broker stops still active).")
                return
            except Exception as e:
                self.alerts.send(f"Error (bot keeps running): {e}")
                _time.sleep(60)


if __name__ == "__main__":
    Bot(Settings()).run()
