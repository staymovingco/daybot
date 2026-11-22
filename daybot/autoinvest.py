"""Auto-investor: once a week, put the cash sitting in your Alpaca account into an index fund.

- Buys only. Never sells. Fractional shares, so any amount from $1 works.
- You add money to Alpaca yourself (e.g. $25/week by recurring bank transfer);
  this just makes sure it gets invested instead of sitting as cash.
- Runs from a daily timer; buys at most once per calendar week, only when the market is open.
- Paper by default. Live needs TRADING_MODE=live and LIVE_CONFIRM=I_ACCEPT_THE_RISK.

Run once by hand:   python -m daybot.autoinvest            (add --dry-run to just preview)
"""
import argparse
import os
from datetime import datetime

from dotenv import load_dotenv

load_dotenv()


def plan_buy(cash: float, reserve: float, max_per_week: float, min_order: float,
             last_buy_week: str, this_week: str) -> float:
    """Dollar amount to invest now (0 = nothing to do)."""
    if last_buy_week == this_week:
        return 0.0
    amount = min(cash - reserve, max_per_week)
    amount = round(amount, 2)
    return amount if amount >= min_order else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    from .broker import ET, Broker
    from .config import Settings
    from .journal import Alerts, Journal

    s = Settings()
    s.validate()
    symbol = os.getenv("AUTOINVEST_SYMBOL", "SPY").upper()
    reserve = float(os.getenv("AUTOINVEST_CASH_RESERVE", "0"))
    max_week = float(os.getenv("AUTOINVEST_MAX_PER_WEEK", "500"))
    min_order = float(os.getenv("AUTOINVEST_MIN_ORDER", "1"))

    b = Broker(s.api_key, s.api_secret, paper=s.paper, feed=s.data_feed)
    j = Journal(s.db_path)
    alerts = Alerts(s.telegram_token, s.telegram_chat_id)
    mode = "PAPER" if s.paper else "LIVE"

    now = datetime.now(ET)
    y, w, _ = now.isocalendar()
    this_week = f"{y}-W{w:02d}"
    if not b.clock().is_open:
        print(f"[{mode}] {now:%Y-%m-%d %H:%M} ET market closed; nothing to do.")
        return

    acct = b.account()
    cash = min(acct["cash"], acct["buying_power"])
    amount = plan_buy(cash, reserve, max_week, min_order, j.get_meta("autoinvest_week", ""), this_week)
    qty, value, pl = b.position_summary(symbol)
    if amount <= 0:
        print(f"[{mode}] Nothing to buy (cash ${cash:.2f}, already bought this week: "
              f"{j.get_meta('autoinvest_week', '') == this_week}). {symbol} holding ${value:,.2f}.")
        return
    if a.dry_run:
        print(f"[{mode}] DRY RUN: would buy ${amount:.2f} of {symbol} (cash ${cash:.2f}).")
        return

    oid = b.buy_notional(symbol, amount)
    j.set_meta("autoinvest_week", this_week)
    j.log_trade(now.date().isoformat(), symbol, 0, amount, 0, 0, oid, f"autoinvest ${amount:.2f}")
    alerts.send(f"[{mode}] Auto-invest: bought ${amount:.2f} of {symbol}. "
                f"Holding before this buy: ${value:,.2f} ({pl:+,.2f} gain/loss).")


if __name__ == "__main__":
    main()
