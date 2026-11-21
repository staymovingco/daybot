"""SQLite trade journal + small helpers for alerts."""
import sqlite3
from datetime import datetime

import requests

SCHEMA = """
CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY, day TEXT, symbol TEXT, qty INTEGER,
    entry_est REAL, stop REAL, target REAL, order_id TEXT, reason TEXT, created_at TEXT);
CREATE TABLE IF NOT EXISTS days (
    day TEXT PRIMARY KEY, start_equity REAL, end_equity REAL, trades INTEGER, note TEXT);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
"""


class Journal:
    def __init__(self, path: str):
        self.db = sqlite3.connect(path)
        self.db.executescript(SCHEMA)

    def log_trade(self, day, symbol, qty, entry, stop, target, order_id, reason):
        self.db.execute(
            "INSERT INTO trades(day,symbol,qty,entry_est,stop,target,order_id,reason,created_at) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            (day, symbol, qty, entry, stop, target, order_id, reason, datetime.utcnow().isoformat()))
        self.db.commit()

    def trades_on(self, day) -> int:
        return self.db.execute("SELECT COUNT(*) FROM trades WHERE day=?", (day,)).fetchone()[0]

    def symbols_traded_on(self, day) -> set:
        return {r[0] for r in self.db.execute("SELECT symbol FROM trades WHERE day=?", (day,))}

    def save_day(self, day, start_equity, end_equity, trades, note=""):
        self.db.execute("INSERT OR REPLACE INTO days VALUES (?,?,?,?,?)",
                        (day, start_equity, end_equity, trades, note))
        self.db.commit()

    def get_meta(self, key, default=None):
        row = self.db.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return row[0] if row else default

    def set_meta(self, key, value):
        self.db.execute("INSERT OR REPLACE INTO meta VALUES (?,?)", (key, str(value)))
        self.db.commit()


class Alerts:
    def __init__(self, token: str, chat_id: str):
        self.token, self.chat_id = token, chat_id

    def send(self, text: str):
        print(f"[ALERT] {text}", flush=True)
        if not (self.token and self.chat_id):
            return
        try:
            requests.post(f"https://api.telegram.org/bot{self.token}/sendMessage",
                          json={"chat_id": self.chat_id, "text": text}, timeout=10)
        except Exception as e:  # alerts must never crash the bot
            print(f"[ALERT FAILED] {e}", flush=True)
