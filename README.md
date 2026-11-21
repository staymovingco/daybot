# daybot: a $100 → $1,000 day-trading bot (paper first)

Automated Opening Range Breakout bot for Alpaca. Small, rule-based, risk-first.

## What it does
Two setups, switch either on/off with `STRATEGIES=` in `.env`:

**1. ORB: Opening Range Breakout** (Toby Crabel / Brian Shannon style)
- First 15 minutes set the range. Buys the first 5-min close above it **if** above VWAP and
  volume is 1.5x+ the day's average so far. Stop: middle of the range. Entries until 11:30 ET.

**2. GAP: Small-cap Gap-and-Go** (Ross Cameron style)
- Stock opened 10%+ above yesterday's close with heavy first-bar dollar volume, and the gap is holding.
- Buys a 5-min close at a **new high of day**, above VWAP, with volume running 3x+ its normal pace.
- Stop: low of the last few bars (the pullback). Entries only until 10:30 ET; momentum fades fast.
- Optional news filter (`GAP_REQUIRE_NEWS=true`). Not included: float (Alpaca has no float data),
  1-minute/pre-market entries. Those are known differences from how Cameron trades it.

Both: **target 2x the risk**; stop and target held **at the broker** (bracket order),
so they still protect you if the bot or server crashes.
- **Universe:** your watchlist + today's most-active stocks + top gainers, $1–$20 by default.
- **No new trades after 11:30 ET. Everything closed by 15:50 ET.** Never holds overnight.

## Risk rules (enforced automatically)
| Rule | Default |
|---|---|
| Risk per trade | 1% of account |
| Max trades/day | 2 |
| Max open positions | 1 |
| Daily loss lock | -2% → flatten, done for the day |
| Max drawdown stop | -15% from peak → bot locks; go back to paper |
| Kill switch | create a file named `KILL` → flatten + exit |
| Live trading | blocked unless `LIVE_CONFIRM=I_ACCEPT_THE_RISK` |

## Setup (about 15 minutes)
1. Make a free account at alpaca.markets. Generate **Paper** API keys.
2. Install Python 3.11+, then in this folder:
   ```
   python -m venv .venv
   source .venv/bin/activate        # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   cp .env.example .env              # then paste your keys into .env
   python -m pytest -q               # should say 8 passed
   ```

## The process
**Phase 1: backtest (prove the edge exists)**
```
python -m daybot.backtest --start 2021-01-01 --end 2025-06-30 --universe smallcaps.txt --strategy both
```
This prints three reports, **ORB alone, GAP alone, and ORB + GAP together**, and saves each trade list
to CSV. Keep only what passes. Tune only on this period. Then run once on the held-out recent stretch:
```
python -m daybot.backtest --start 2025-07-01 --end 2026-11-20 --universe smallcaps.txt --strategy both
```
**Gap-and-go backtest warning:** it only finds gaps in the symbols you give it. Real gappers are
often random stocks you'd never list. Add more symbols to `smallcaps.txt` (hundreds is better).
Today's list also misses stocks that were delisted, which makes results look better than reality.
Needs: 100+ trades, average R above +0.05, drawdown under 10%. If it says NOT READY, don't trade it.
Change one setting at a time (RVOL_MIN, REWARD_RISK, MIN/MAX_PRICE, symbols) and re-test.

**Phase 2: paper trade 1–2 months**
```
python -m daybot.bot
```
Leave it running on a $5–12/mo cloud server (DigitalOcean/Lightsail) using `tmux` or a systemd service.
Compare paper results to the backtest. Big gap = the backtest was lying.

**Phase 3: live with $100**
Make live keys in a **cash account**, put them in `.env`, set `TRADING_MODE=live`
and `LIVE_CONFIRM=I_ACCEPT_THE_RISK`. Milestones: $200 → $400 → $800 → $1,000.
Risk recalculates automatically as the account grows.

## Files
- `daybot/strategy.py` – the setup (shared by backtest and live)
- `daybot/risk.py` – sizing and loss limits
- `daybot/broker.py` – Alpaca orders and data
- `daybot/bot.py` – the live loop
- `daybot/backtest.py` – honest backtester (next-bar fills, slippage, worst-case same-bar exits)
- `daybot/journal.py` – SQLite trade journal + Telegram alerts

## Honest notes
- This is one sample setup, not a proven money-maker. The backtest decides.
- Cheap stocks have wide spreads and get manipulated. Default `$1` minimum and liquidity filter are there on purpose.
- Free IEX data covers a slice of volume. Paid SIP data (`DATA_FEED=sip`) is more accurate.
- Cash accounts can only trade settled money (next business day), so $100 may only allow ~1 trade/day.
- Not financial advice.
