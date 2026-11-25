"""Long strangle learning trade (PAPER ONLY), in its own paper account so it can be compared
with the single-call trade.

Buys 1 SPY call ~3% ABOVE today's price and 1 SPY put ~3% BELOW, same expiration 30-45 days out.
Profits if SPY makes a big move either way; loses if SPY stays quiet (time decay on both legs).

Keys (separate paper account): STRANGLE_ALPACA_API_KEY / STRANGLE_ALPACA_API_SECRET in .env.
Sell rules are shared with the call trade: OPTIONS_TAKE_PROFIT, OPTIONS_STOP_LOSS, OPTIONS_EXIT_DAYS.

  python -m daybot.strangle_demo --preview
  python -m daybot.strangle_demo --buy
  python -m daybot.strangle_demo              # status + HOLD/SELL verdict
  python -m daybot.strangle_demo --sell       # sell both legs
  python -m daybot.strangle_demo --auto-sell  # sell both legs only if a rule says SELL
"""
import argparse
import os
from datetime import date, timedelta

from dotenv import load_dotenv

from .options_demo import verdict

load_dotenv()
WIDTH = 0.03


def pick_strangle(calls, puts, spot: float, today: date, width=WIDTH, min_days=30, max_days=45):
    """Choose one expiration (closest to ~37 days) that has both legs, then the call strike nearest
    spot*(1+width) and the put strike nearest spot*(1-width)."""
    def ok(c):
        return c.tradable and min_days <= (c.expiration_date - today).days <= max_days
    exps = sorted({c.expiration_date for c in calls if ok(c)} & {p.expiration_date for p in puts if ok(p)},
                  key=lambda d: abs((d - today).days - 37))
    if not exps:
        return None, None
    exp = exps[0]
    call = min((c for c in calls if c.expiration_date == exp and c.tradable),
               key=lambda c: abs(float(c.strike_price) - spot * (1 + width)))
    put = min((p for p in puts if p.expiration_date == exp and p.tradable),
              key=lambda p: abs(float(p.strike_price) - spot * (1 - width)))
    return call, put


def explain(spot, call_strike, put_strike, total_entry, total_now, pl, expiry, today):
    dte = (expiry - today).days
    cost, value = total_entry * 100, total_now * 100
    up, down = call_strike + total_entry, put_strike - total_entry
    return "\n".join([
        f"You own a long strangle on SPY, expiring {expiry} ({dte} days left):",
        f"  call strike ${call_strike:.2f} (above)  +  put strike ${put_strike:.2f} (below)",
        f"  SPY now: ${spot:.2f}",
        f"  Paid ${cost:,.0f} for both  ->  worth ${value:,.0f} now   Profit/loss: {pl:+,.0f} "
        f"({pl / cost:+.0%})" if cost else "",
        f"  Break-even at expiry: SPY above ${up:.2f} ({up / spot - 1:+.1%}) or below ${down:.2f} "
        f"({down / spot - 1:+.1%})",
        "",
        "What's going on:",
        "  - You win if SPY makes a BIG move either way. A quiet SPY slowly drains both legs",
        "    (time decay x2) - that's the cost of not having to guess the direction.",
    ])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--buy", action="store_true")
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--sell", action="store_true")
    ap.add_argument("--auto-sell", action="store_true")
    a = ap.parse_args()

    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockLatestTradeRequest
    from alpaca.trading.client import TradingClient
    from alpaca.trading.enums import AssetStatus, ContractType, OrderSide, TimeInForce
    from alpaca.trading.requests import GetOptionContractsRequest, MarketOrderRequest

    key, secret = os.getenv("STRANGLE_ALPACA_API_KEY", ""), os.getenv("STRANGLE_ALPACA_API_SECRET", "")
    if not key or not secret:
        raise SystemExit("Add STRANGLE_ALPACA_API_KEY and STRANGLE_ALPACA_API_SECRET "
                         "(a separate PAPER account) to .env.")
    tc = TradingClient(key, secret, paper=True)          # hard-wired to paper
    dc = StockHistoricalDataClient(key, secret)
    spot = float(dc.get_stock_latest_trade(StockLatestTradeRequest(symbol_or_symbols="SPY"))["SPY"].price)
    today = date.today()
    held = [p for p in tc.get_all_positions() if str(getattr(p.asset_class, "value", p.asset_class)) == "us_option"]

    if not a.buy and not a.preview:
        if not held:
            print(f"No strangle in this paper account (cash ${float(tc.get_account().cash):,.0f}). "
                  "Run with --preview, then --buy.")
            return
        legs = {}
        for p in held:
            c = tc.get_option_contract(p.symbol)
            legs[str(getattr(c.type, "value", c.type))] = (p, c)
        entry = sum(float(p.avg_entry_price) * float(p.qty) for p, _ in legs.values())
        now = sum(float(p.current_price) * float(p.qty) for p, _ in legs.values())
        pl = sum(float(p.unrealized_pl) for p, _ in legs.values())
        expiry = min(c.expiration_date for _, c in legs.values())
        cs = float(legs["call"][1].strike_price) if "call" in legs else float("nan")
        ps = float(legs["put"][1].strike_price) if "put" in legs else float("nan")
        print(explain(spot, cs, ps, entry, now, pl, expiry, today))
        if len(legs) < 2:
            print("  NOTE: only one leg is open right now.")
        pl_pct = pl / (entry * 100) if entry else 0.0
        action, why = verdict(pl_pct, (expiry - today).days,
                              float(os.getenv("OPTIONS_TAKE_PROFIT", "0.5")),
                              float(os.getenv("OPTIONS_STOP_LOSS", "0.5")),
                              int(os.getenv("OPTIONS_EXIT_DAYS", "14")))
        print(f"\n>>> {action}: {why}")
        if a.sell or (a.auto_sell and action == "SELL"):
            if not tc.get_clock().is_open:
                print("Market is closed - can't sell now. Try between 7:30am and 2:00pm Mountain.")
                return
            for p, _ in legs.values():
                o = tc.submit_order(MarketOrderRequest(symbol=p.symbol, qty=float(p.qty), side=OrderSide.SELL,
                                                       time_in_force=TimeInForce.DAY))
                print(f"PAPER sell order placed: {p.symbol} (order {o.id})")
            print(f"Result: about {pl:+,.0f} ({pl_pct:+.0%}).")
        elif action == "SELL":
            print("    To follow the rule, run:  .venv/bin/python -m daybot.strangle_demo --sell")
        return

    def contracts(kind, lo, hi):
        req = GetOptionContractsRequest(underlying_symbols=["SPY"], type=kind, status=AssetStatus.ACTIVE,
                                        expiration_date_gte=today + timedelta(days=30),
                                        expiration_date_lte=today + timedelta(days=45),
                                        strike_price_gte=str(round(lo)), strike_price_lte=str(round(hi)),
                                        limit=1000)
        return tc.get_option_contracts(req).option_contracts
    calls = contracts(ContractType.CALL, spot * 1.01, spot * 1.05)
    puts = contracts(ContractType.PUT, spot * 0.95, spot * 0.99)
    call, put = pick_strangle(calls, puts, spot, today)
    if call is None:
        raise SystemExit("Couldn't find matching SPY calls and puts 30-45 days out.")
    est = sum(float(c.close_price) for c in (call, put) if c.close_price)
    print(f"SPY is ${spot:.2f}. Picked expiration {call.expiration_date} "
          f"({(call.expiration_date - today).days} days):")
    print(f"  CALL {call.symbol}  strike ${float(call.strike_price):.2f} ({float(call.strike_price)/spot-1:+.1%})")
    print(f"  PUT  {put.symbol}  strike ${float(put.strike_price):.2f} ({float(put.strike_price)/spot-1:+.1%})")
    if est:
        print(f"  Last prices add up to ~${est:.2f}/share = about ${est*100:,.0f} for the pair. "
              f"SPY must move about {est/spot:.1%} past a strike to break even at expiry.")
    if a.preview:
        print("Preview only - nothing bought. Run with --buy to buy both legs in paper.")
        return
    if held:
        raise SystemExit("This account already holds options - one strangle at a time.")
    if not tc.get_clock().is_open:
        raise SystemExit("Market is closed. Run --buy between 7:30am and 2:00pm Mountain time on a weekday.")
    for c in (call, put):
        o = tc.submit_order(MarketOrderRequest(symbol=c.symbol, qty=1, side=OrderSide.BUY,
                                               time_in_force=TimeInForce.DAY))
        print(f"PAPER order placed: buy 1 {c.symbol} (order {o.id})")
    print("Check it with: .venv/bin/python -m daybot.strangle_demo")


if __name__ == "__main__":
    main()
