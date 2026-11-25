"""Options learning trade (PAPER ONLY): buy 1 SPY call ~30-45 days out, strike near today's price,
then check on it whenever you like with a plain-English report.

Uses a SEPARATE paper account so it doesn't mix with the auto-investor:
put that account's keys in .env as OPTIONS_ALPACA_API_KEY / OPTIONS_ALPACA_API_SECRET.

  python -m daybot.options_demo --preview   # show the contract it would buy, buy nothing
  python -m daybot.options_demo --buy       # buy 1 contract (paper, market hours only)
  python -m daybot.options_demo             # status report on what you own
"""
import argparse
import os
from datetime import date, timedelta

from dotenv import load_dotenv

load_dotenv()


def pick_contract(contracts, spot: float, today: date, min_days=30, max_days=45):
    """Nearest-to-the-money call expiring 30-45 days out (prefers ~37 days)."""
    best = None
    for c in contracts:
        dte = (c.expiration_date - today).days
        if not (min_days <= dte <= max_days) or not c.tradable:
            continue
        score = (abs(float(c.strike_price) - spot) / spot, abs(dte - 37))
        if best is None or score < best[0]:
            best = (score, c)
    return best[1] if best else None


def explain(sym, qty, entry, now, pl, spot, strike, expiry, today):
    dte = (expiry - today).days
    cost = entry * 100 * qty
    value = now * 100 * qty
    breakeven = strike + entry
    lines = [
        f"You own {qty:g} call contract(s): {sym}",
        f"  = the right to buy {int(100*qty)} SPY shares at ${strike:.2f} until {expiry} ({dte} days left)",
        f"  SPY now: ${spot:.2f}   |  strike: ${strike:.2f}   |  break-even at expiry: ${breakeven:.2f}",
        f"  Paid ${cost:,.0f} (${entry:.2f}/share)  ->  worth ${value:,.0f} now (${now:.2f}/share)",
        f"  Profit/loss: {pl:+,.0f} ({pl / cost:+.0%})" if cost else "",
        "",
        "What's going on:",
    ]
    intrinsic = max(0.0, spot - strike)
    time_value = max(0.0, now - intrinsic)
    lines.append(f"  - ${intrinsic:.2f}/share is 'real' value (SPY is ${spot - strike:+.2f} vs the strike).")
    lines.append(f"  - ${time_value:.2f}/share is 'time value' - this melts toward $0 as expiry gets closer,")
    lines.append("    faster in the last few weeks. That melt is the cost of owning an option.")
    if dte <= 7:
        lines.append("  - Under a week left: time value is nearly gone. Most traders sell before this point.")
    if spot < strike:
        lines.append(f"  - SPY needs to rise above ${strike:.2f} for the option to have real value at expiry.")
    return "\n".join(x for x in lines if x is not None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--buy", action="store_true")
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--use-main-account", action="store_true",
                    help="use ALPACA_API_KEY instead of the separate options paper account")
    a = ap.parse_args()

    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockLatestTradeRequest
    from alpaca.trading.client import TradingClient
    from alpaca.trading.enums import AssetStatus, ContractType, OrderSide, TimeInForce
    from alpaca.trading.requests import GetOptionContractsRequest, MarketOrderRequest

    if a.use_main_account:
        key, secret = os.getenv("ALPACA_API_KEY", ""), os.getenv("ALPACA_API_SECRET", "")
    else:
        key, secret = os.getenv("OPTIONS_ALPACA_API_KEY", ""), os.getenv("OPTIONS_ALPACA_API_SECRET", "")
    if not key or not secret:
        raise SystemExit("Add OPTIONS_ALPACA_API_KEY and OPTIONS_ALPACA_API_SECRET (a separate PAPER "
                         "account) to .env, or run with --use-main-account.")
    tc = TradingClient(key, secret, paper=True)          # hard-wired to paper
    dc = StockHistoricalDataClient(key, secret)
    acct = tc.get_account()
    spot = float(dc.get_stock_latest_trade(StockLatestTradeRequest(symbol_or_symbols="SPY"))["SPY"].price)
    today = date.today()

    held = [p for p in tc.get_all_positions() if str(getattr(p.asset_class, "value", p.asset_class)) == "us_option"]
    if not a.buy and not a.preview:
        if not held:
            print(f"No options yet in this paper account (cash ${float(acct.cash):,.0f}). "
                  "Run with --preview, then --buy.")
            return
        for p in held:
            c = tc.get_option_contract(p.symbol)
            print(explain(p.symbol, float(p.qty), float(p.avg_entry_price), float(p.current_price),
                          float(p.unrealized_pl), spot, float(c.strike_price), c.expiration_date, today))
            print()
        return

    req = GetOptionContractsRequest(underlying_symbols=["SPY"], type=ContractType.CALL,
                                    status=AssetStatus.ACTIVE,
                                    expiration_date_gte=today + timedelta(days=30),
                                    expiration_date_lte=today + timedelta(days=45),
                                    strike_price_gte=str(round(spot * 0.97)),
                                    strike_price_lte=str(round(spot * 1.03)), limit=500)
    c = pick_contract(tc.get_option_contracts(req).option_contracts, spot, today)
    if c is None:
        raise SystemExit("No suitable SPY call found 30-45 days out.")
    est = float(c.close_price) if c.close_price else None
    print(f"SPY is ${spot:.2f}. Picked: {c.symbol}")
    print(f"  Call, strike ${float(c.strike_price):.2f}, expires {c.expiration_date} "
          f"({(c.expiration_date - today).days} days)")
    if est:
        print(f"  Last price ~${est:.2f}/share = about ${est * 100:,.0f} for 1 contract (100 shares)")
    if a.preview:
        print("Preview only - nothing bought. Run with --buy to buy 1 contract in paper.")
        return
    if held:
        raise SystemExit("You already own an option in this account - one learning trade at a time.")
    if not tc.get_clock().is_open:
        raise SystemExit("Market is closed. Run --buy between 7:30am and 2:00pm Mountain time on a weekday.")
    o = tc.submit_order(MarketOrderRequest(symbol=c.symbol, qty=1, side=OrderSide.BUY,
                                           time_in_force=TimeInForce.DAY))
    print(f"PAPER order placed: buy 1 {c.symbol} (order {o.id}). Check it with: "
          f"python -m daybot.options_demo")


if __name__ == "__main__":
    main()
