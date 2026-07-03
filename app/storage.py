from __future__ import annotations

import json
import os
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Iterator

from .models import Position


class Storage:
    def __init__(self, path: str, initial_balance: float) -> None:
        self.path = path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._lock = threading.RLock()
        self._init_db(initial_balance)

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            conn = sqlite3.connect(self.path)
            conn.row_factory = sqlite3.Row
            try:
                yield conn
                conn.commit()
            finally:
                conn.close()

    def _init_db(self, initial_balance: float) -> None:
        with self._conn() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS positions (
                    instrument TEXT PRIMARY KEY,
                    side TEXT NOT NULL,
                    units INTEGER NOT NULL,
                    entry_price REAL NOT NULL,
                    stop_price REAL NOT NULL,
                    opened_at TEXT NOT NULL,
                    planned_risk_home REAL NOT NULL,
                    broker_trade_id TEXT
                );
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts TEXT NOT NULL,
                    level TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    instrument TEXT,
                    payload TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS processed_candles (
                    instrument TEXT NOT NULL,
                    candle_time TEXT NOT NULL,
                    PRIMARY KEY (instrument, candle_time)
                );
                CREATE TABLE IF NOT EXISTS kv (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS paper_account (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    balance REAL NOT NULL
                );
                """
            )
            conn.execute(
                "INSERT OR IGNORE INTO paper_account(id, balance) VALUES(1, ?)",
                (initial_balance,),
            )

    def add_event(
        self,
        level: str,
        kind: str,
        payload: dict[str, Any],
        instrument: str | None = None,
    ) -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO events(ts, level, kind, instrument, payload) VALUES(?,?,?,?,?)",
                (
                    datetime.now(timezone.utc).isoformat(),
                    level,
                    kind,
                    instrument,
                    json.dumps(payload, ensure_ascii=False, default=str),
                ),
            )

    def recent_events(self, limit: int = 100) -> list[dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM events ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["payload"] = json.loads(item["payload"])
            result.append(item)
        return result

    def latest_event(self, kind: str) -> dict[str, Any] | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM events WHERE kind=? ORDER BY id DESC LIMIT 1",
                (kind,),
            ).fetchone()
        if row is None:
            return None
        item = dict(row)
        item["payload"] = json.loads(item["payload"])
        return item

    def get_positions(self) -> list[Position]:
        with self._conn() as conn:
            rows = conn.execute("SELECT * FROM positions ORDER BY instrument").fetchall()
        return [Position(**dict(row)) for row in rows]

    def get_position(self, instrument: str) -> Position | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM positions WHERE instrument=?", (instrument,)
            ).fetchone()
        return Position(**dict(row)) if row else None

    def save_position(self, position: Position) -> None:
        with self._conn() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO positions(
                    instrument, side, units, entry_price, stop_price, opened_at,
                    planned_risk_home, broker_trade_id
                ) VALUES(?,?,?,?,?,?,?,?)
                """,
                (
                    position.instrument,
                    position.side,
                    position.units,
                    position.entry_price,
                    position.stop_price,
                    position.opened_at,
                    position.planned_risk_home,
                    position.broker_trade_id,
                ),
            )

    def delete_position(self, instrument: str) -> None:
        with self._conn() as conn:
            conn.execute("DELETE FROM positions WHERE instrument=?", (instrument,))

    def is_processed(self, instrument: str, candle_time: str) -> bool:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT 1 FROM processed_candles WHERE instrument=? AND candle_time=?",
                (instrument, candle_time),
            ).fetchone()
        return row is not None

    def mark_processed(self, instrument: str, candle_time: str) -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO processed_candles(instrument, candle_time) VALUES(?,?)",
                (instrument, candle_time),
            )

    def get_kv(self, key: str, default: str | None = None) -> str | None:
        with self._conn() as conn:
            row = conn.execute("SELECT value FROM kv WHERE key=?", (key,)).fetchone()
        return row["value"] if row else default

    def set_kv(self, key: str, value: str) -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO kv(key, value) VALUES(?,?)", (key, value)
            )

    def paper_balance(self) -> float:
        with self._conn() as conn:
            row = conn.execute("SELECT balance FROM paper_account WHERE id=1").fetchone()
        return float(row["balance"])

    def set_paper_balance(self, balance: float) -> None:
        with self._conn() as conn:
            conn.execute("UPDATE paper_account SET balance=? WHERE id=1", (balance,))
