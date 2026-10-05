from __future__ import annotations

import json
import os
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Iterator

from .models import Position
from .recovery_authority import (
    EVENT_BOOTSTRAP,
    EVENT_LATCHED,
    RECOVERY_EVENT_PREFIX,
    RECOVERY_INITIALIZED_KEY,
    RECOVERY_INITIALIZED_VALUE,
    RECOVERY_LATCH_DRAWDOWN_STOP_TEXT,
    RECOVERY_LATCH_DRAWDOWN_STOP_V1,
    RECOVERY_SQLITE_BUSY_TIMEOUT_MS,
    RECOVERY_STATE_KEY,
    STATE_ACTIVE,
    STATE_LATCHED_DD_STOP,
    RecoveryStateError,
    active_state,
    calculate_drawdown,
    canonical_float_text,
    canonical_json,
    is_governed_db,
    latched_state,
    parse_positive_float_text,
    parse_recovery_state,
    recovery_latch_reached,
    state_block_reason,
    state_entries_blocked,
)


class Storage:
    def __init__(self, path: str, initial_balance: float) -> None:
        self.path = path
        self.initial_balance = float(initial_balance)
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

    @contextmanager
    def _recovery_conn(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            conn = sqlite3.connect(
                self.path,
                timeout=RECOVERY_SQLITE_BUSY_TIMEOUT_MS / 1000.0,
                isolation_level=None,
            )
            conn.row_factory = sqlite3.Row
            conn.execute(
                f"PRAGMA busy_timeout={int(RECOVERY_SQLITE_BUSY_TIMEOUT_MS)}"
            )
            try:
                conn.execute("BEGIN IMMEDIATE")
                yield conn
                conn.execute("COMMIT")
            except Exception:
                if conn.in_transaction:
                    conn.execute("ROLLBACK")
                raise
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

    @staticmethod
    def _insert_event_conn(
        conn: sqlite3.Connection,
        *,
        ts: str,
        level: str,
        kind: str,
        payload: dict[str, Any],
        instrument: str | None = None,
        canonical: bool = False,
    ) -> int:
        payload_text = (
            canonical_json(payload)
            if canonical
            else json.dumps(payload, ensure_ascii=False, default=str)
        )
        cursor = conn.execute(
            "INSERT INTO events(ts, level, kind, instrument, payload) VALUES(?,?,?,?,?)",
            (ts, level, kind, instrument, payload_text),
        )
        return int(cursor.lastrowid)

    @staticmethod
    def _get_kv_conn(
        conn: sqlite3.Connection, key: str, default: str | None = None
    ) -> str | None:
        row = conn.execute("SELECT value FROM kv WHERE key=?", (key,)).fetchone()
        return row["value"] if row else default

    @staticmethod
    def _set_kv_conn(conn: sqlite3.Connection, key: str, value: str) -> None:
        conn.execute(
            "INSERT OR REPLACE INTO kv(key, value) VALUES(?,?)",
            (key, value),
        )

    @staticmethod
    def _has_recovery_event_conn(conn: sqlite3.Connection) -> bool:
        row = conn.execute(
            "SELECT 1 FROM events "
            "WHERE substr(kind, 1, ?) = ? LIMIT 1",
            (len(RECOVERY_EVENT_PREFIX), RECOVERY_EVENT_PREFIX),
        ).fetchone()
        return row is not None

    @staticmethod
    def _latest_recovery_event_conn(
        conn: sqlite3.Connection,
    ) -> dict[str, Any] | None:
        row = conn.execute(
            "SELECT * FROM events "
            "WHERE substr(kind, 1, ?) = ? "
            "ORDER BY id DESC LIMIT 1",
            (len(RECOVERY_EVENT_PREFIX), RECOVERY_EVENT_PREFIX),
        ).fetchone()
        if row is None:
            return None
        item = dict(row)
        try:
            item["payload"] = json.loads(item["payload"])
        except json.JSONDecodeError:
            item["payload"] = None
        return item

    def add_event(
        self,
        level: str,
        kind: str,
        payload: dict[str, Any],
        instrument: str | None = None,
    ) -> None:
        with self._conn() as conn:
            self._insert_event_conn(
                conn,
                ts=datetime.now(timezone.utc).isoformat(),
                level=level,
                kind=kind,
                payload=payload,
                instrument=instrument,
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
            return self._get_kv_conn(conn, key, default)

    def set_kv(self, key: str, value: str) -> None:
        with self._conn() as conn:
            self._set_kv_conn(conn, key, value)

    def paper_balance(self) -> float:
        with self._conn() as conn:
            row = conn.execute("SELECT balance FROM paper_account WHERE id=1").fetchone()
        return float(row["balance"])

    def set_paper_balance(self, balance: float) -> None:
        with self._conn() as conn:
            conn.execute("UPDATE paper_account SET balance=? WHERE id=1", (balance,))

    @staticmethod
    def _recovery_result(
        *,
        governed: bool,
        integrity_ok: bool,
        mode_ok: bool,
        state: dict[str, Any] | None,
        high_water: float | None,
        drawdown: float | None,
        bootstrap_occurred: bool = False,
        transition_occurred: bool = False,
        latest_event: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> dict[str, Any]:
        block_reason = state_block_reason(
            state,
            integrity_ok=integrity_ok,
            mode_ok=mode_ok,
        )
        return {
            "governed": governed,
            "integrity_ok": integrity_ok,
            "mode_ok": mode_ok,
            "state": state,
            "high_water": high_water,
            "drawdown": drawdown,
            "bootstrap_occurred": bootstrap_occurred,
            "transition_occurred": transition_occurred,
            "entries_blocked": (
                governed
                and (
                    state_entries_blocked(state)
                    or not integrity_ok
                    or not mode_ok
                )
            ),
            "block_reason": block_reason if governed else None,
            "latest_event": latest_event,
            "error": error,
        }

    def update_nav_high_water_atomic(self, nav: float) -> tuple[float, float]:
        nav_value = float(nav)
        canonical_float_text(nav_value)
        with self._recovery_conn() as conn:
            raw = self._get_kv_conn(conn, "nav_high_water")
            if raw is None:
                high_water = nav_value
                self._set_kv_conn(
                    conn, "nav_high_water", canonical_float_text(high_water)
                )
            else:
                current = parse_positive_float_text(raw, name="nav_high_water")
                high_water = max(current, nav_value)
                if high_water != current:
                    self._set_kv_conn(
                        conn, "nav_high_water", canonical_float_text(high_water)
                    )
            return high_water, calculate_drawdown(nav_value, high_water)

    def recovery_snapshot(self, *, broker_mode: str) -> dict[str, Any]:
        with self._conn() as conn:
            raw_state = self._get_kv_conn(conn, RECOVERY_STATE_KEY)
            marker = self._get_kv_conn(conn, RECOVERY_INITIALIZED_KEY)
            recovery_event_exists = self._has_recovery_event_conn(conn)
            governed = is_governed_db(
                self.path,
                state_present=raw_state is not None,
                marker_present=marker is not None,
                event_present=recovery_event_exists,
            )
            if not governed:
                return self._recovery_result(
                    governed=False,
                    integrity_ok=True,
                    mode_ok=True,
                    state=None,
                    high_water=None,
                    drawdown=None,
                )

            mode_ok = broker_mode == "mt5_paper"
            if raw_state is None:
                return self._recovery_result(
                    governed=True,
                    integrity_ok=False,
                    mode_ok=mode_ok,
                    state=None,
                    high_water=None,
                    drawdown=None,
                    latest_event=self._latest_recovery_event_conn(conn),
                    error="recovery state is uninitialized or missing",
                )
            if marker != RECOVERY_INITIALIZED_VALUE:
                return self._recovery_result(
                    governed=True,
                    integrity_ok=False,
                    mode_ok=mode_ok,
                    state=None,
                    high_water=None,
                    drawdown=None,
                    latest_event=self._latest_recovery_event_conn(conn),
                    error="recovery initialization marker is missing or invalid",
                )
            try:
                state = parse_recovery_state(raw_state)
                high_water = parse_positive_float_text(
                    self._get_kv_conn(conn, "nav_high_water"),
                    name="nav_high_water",
                )
            except RecoveryStateError as exc:
                return self._recovery_result(
                    governed=True,
                    integrity_ok=False,
                    mode_ok=mode_ok,
                    state=None,
                    high_water=None,
                    drawdown=None,
                    latest_event=self._latest_recovery_event_conn(conn),
                    error=str(exc),
                )
            return self._recovery_result(
                governed=True,
                integrity_ok=True,
                mode_ok=mode_ok,
                state=state,
                high_water=high_water,
                drawdown=None,
                latest_event=self._latest_recovery_event_conn(conn),
            )

    def evaluate_recovery(
        self,
        *,
        nav: float,
        broker_mode: str,
        strategy_profile: str,
        source_commit: str | None,
        observed_at: str,
    ) -> dict[str, Any]:
        try:
            return self._evaluate_recovery_locked(
                nav=nav,
                broker_mode=broker_mode,
                strategy_profile=strategy_profile,
                source_commit=source_commit,
                observed_at=observed_at,
            )
        except (sqlite3.Error, RecoveryStateError, ValueError) as exc:
            return self._recovery_result(
                governed=True,
                integrity_ok=False,
                mode_ok=broker_mode == "mt5_paper",
                state=None,
                high_water=None,
                drawdown=None,
                error=str(exc),
            )

    def _evaluate_recovery_locked(
        self,
        *,
        nav: float,
        broker_mode: str,
        strategy_profile: str,
        source_commit: str | None,
        observed_at: str,
    ) -> dict[str, Any]:
        nav_value = float(nav)
        if not canonical_float_text(nav_value):
            raise RecoveryStateError("NAV is invalid")

        with self._recovery_conn() as conn:
            raw_state = self._get_kv_conn(conn, RECOVERY_STATE_KEY)
            marker = self._get_kv_conn(conn, RECOVERY_INITIALIZED_KEY)
            recovery_event_exists = self._has_recovery_event_conn(conn)
            governed = is_governed_db(
                self.path,
                state_present=raw_state is not None,
                marker_present=marker is not None,
                event_present=recovery_event_exists,
            )
            if not governed:
                return self._recovery_result(
                    governed=False,
                    integrity_ok=True,
                    mode_ok=True,
                    state=None,
                    high_water=None,
                    drawdown=None,
                )

            mode_ok = broker_mode == "mt5_paper"
            if not mode_ok:
                return self._recovery_result(
                    governed=True,
                    integrity_ok=True,
                    mode_ok=False,
                    state=None,
                    high_water=None,
                    drawdown=None,
                    latest_event=self._latest_recovery_event_conn(conn),
                    error="governed Recovery V1 DB requires mt5_paper mode",
                )

            raw_high_water = self._get_kv_conn(conn, "nav_high_water")

            if raw_state is None:
                if marker is not None or recovery_event_exists:
                    return self._recovery_result(
                        governed=True,
                        integrity_ok=False,
                        mode_ok=True,
                        state=None,
                        high_water=None,
                        drawdown=None,
                        latest_event=self._latest_recovery_event_conn(conn),
                        error="recovery state missing after prior initialization evidence",
                    )

                if raw_high_water is not None:
                    high_water = parse_positive_float_text(
                        raw_high_water, name="nav_high_water"
                    )
                    drawdown = calculate_drawdown(nav_value, high_water)
                    if source_commit is None:
                        return self._recovery_result(
                            governed=True,
                            integrity_ok=False,
                            mode_ok=True,
                            state=None,
                            high_water=high_water,
                            drawdown=drawdown,
                            error="source commit unavailable for recovery bootstrap",
                        )
                    if recovery_latch_reached(nav_value, high_water):
                        drawdown = max(drawdown, RECOVERY_LATCH_DRAWDOWN_STOP_V1)
                        state = latched_state(
                            epoch=0,
                            tripped_at=observed_at,
                            nav=nav_value,
                            high_water=high_water,
                            drawdown=drawdown,
                        )
                    else:
                        state = active_state(epoch=0)
                    self._set_kv_conn(
                        conn, RECOVERY_STATE_KEY, canonical_json(state)
                    )
                    self._set_kv_conn(
                        conn,
                        RECOVERY_INITIALIZED_KEY,
                        RECOVERY_INITIALIZED_VALUE,
                    )
                    payload = {
                        "schema_version": state["schema_version"],
                        "epoch": state["epoch"],
                        "previous_state": None,
                        "new_state": state["state"],
                        "latch_drawdown_stop_text": RECOVERY_LATCH_DRAWDOWN_STOP_TEXT,
                        "trip_id": state["trip_id"],
                        "nav": canonical_float_text(nav_value),
                        "high_water": canonical_float_text(high_water),
                        "drawdown": canonical_float_text(drawdown),
                        "position_count": int(
                            conn.execute("SELECT COUNT(*) FROM positions").fetchone()[0]
                        ),
                        "strategy_profile": strategy_profile,
                        "broker_mode": broker_mode,
                        "db_basename": os.path.basename(self.path),
                        "source_commit": source_commit,
                        "timestamp": observed_at,
                    }
                    event_id = self._insert_event_conn(
                        conn,
                        ts=observed_at,
                        level="WARNING"
                        if state["state"] == STATE_LATCHED_DD_STOP
                        else "INFO",
                        kind=EVENT_BOOTSTRAP,
                        payload=payload,
                        canonical=True,
                    )
                    return self._recovery_result(
                        governed=True,
                        integrity_ok=True,
                        mode_ok=True,
                        state=state,
                        high_water=high_water,
                        drawdown=drawdown,
                        bootstrap_occurred=True,
                        transition_occurred=True,
                        latest_event={
                            "id": event_id,
                            "ts": observed_at,
                            "kind": EVENT_BOOTSTRAP,
                            "payload": payload,
                        },
                    )

                position_count = int(
                    conn.execute("SELECT COUNT(*) FROM positions").fetchone()[0]
                )
                event_count = int(
                    conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
                )
                processed_count = int(
                    conn.execute("SELECT COUNT(*) FROM processed_candles").fetchone()[0]
                )
                kv_count = int(conn.execute("SELECT COUNT(*) FROM kv").fetchone()[0])
                paper_balance = float(
                    conn.execute(
                        "SELECT balance FROM paper_account WHERE id=1"
                    ).fetchone()[0]
                )
                fresh = (
                    position_count == 0
                    and event_count == 0
                    and processed_count == 0
                    and kv_count == 0
                    and canonical_float_text(paper_balance)
                    == canonical_float_text(self.initial_balance)
                    and canonical_float_text(nav_value)
                    == canonical_float_text(paper_balance)
                )
                if not fresh:
                    return self._recovery_result(
                        governed=True,
                        integrity_ok=False,
                        mode_ok=True,
                        state=None,
                        high_water=None,
                        drawdown=None,
                        error="historical/non-fresh DB is missing nav_high_water",
                    )
                if source_commit is None:
                    return self._recovery_result(
                        governed=True,
                        integrity_ok=False,
                        mode_ok=True,
                        state=None,
                        high_water=None,
                        drawdown=None,
                        error="source commit unavailable for recovery bootstrap",
                    )

                high_water = nav_value
                drawdown = 0.0
                state = active_state(epoch=0)
                self._set_kv_conn(
                    conn, "nav_high_water", canonical_float_text(high_water)
                )
                self._set_kv_conn(conn, RECOVERY_STATE_KEY, canonical_json(state))
                self._set_kv_conn(
                    conn, RECOVERY_INITIALIZED_KEY, RECOVERY_INITIALIZED_VALUE
                )
                payload = {
                    "schema_version": state["schema_version"],
                    "epoch": state["epoch"],
                    "previous_state": None,
                    "new_state": state["state"],
                    "latch_drawdown_stop_text": RECOVERY_LATCH_DRAWDOWN_STOP_TEXT,
                    "trip_id": None,
                    "nav": canonical_float_text(nav_value),
                    "high_water": canonical_float_text(high_water),
                    "drawdown": canonical_float_text(drawdown),
                    "position_count": position_count,
                    "strategy_profile": strategy_profile,
                    "broker_mode": broker_mode,
                    "db_basename": os.path.basename(self.path),
                    "source_commit": source_commit,
                    "timestamp": observed_at,
                }
                event_id = self._insert_event_conn(
                    conn,
                    ts=observed_at,
                    level="INFO",
                    kind=EVENT_BOOTSTRAP,
                    payload=payload,
                    canonical=True,
                )
                return self._recovery_result(
                    governed=True,
                    integrity_ok=True,
                    mode_ok=True,
                    state=state,
                    high_water=high_water,
                    drawdown=drawdown,
                    bootstrap_occurred=True,
                    transition_occurred=True,
                    latest_event={
                        "id": event_id,
                        "ts": observed_at,
                        "kind": EVENT_BOOTSTRAP,
                        "payload": payload,
                    },
                )

            if marker != RECOVERY_INITIALIZED_VALUE:
                return self._recovery_result(
                    governed=True,
                    integrity_ok=False,
                    mode_ok=True,
                    state=None,
                    high_water=None,
                    drawdown=None,
                    latest_event=self._latest_recovery_event_conn(conn),
                    error="recovery state/initialization marker inconsistency",
                )

            state = parse_recovery_state(raw_state)
            high_water = parse_positive_float_text(
                raw_high_water, name="nav_high_water"
            )

            if (
                state["state"] in {STATE_ACTIVE, STATE_LATCHED_DD_STOP}
                and nav_value > high_water
            ):
                high_water = nav_value
                self._set_kv_conn(
                    conn, "nav_high_water", canonical_float_text(high_water)
                )
            drawdown = calculate_drawdown(nav_value, high_water)

            if (
                state["state"] == STATE_ACTIVE
                and recovery_latch_reached(nav_value, high_water)
            ):
                drawdown = max(drawdown, RECOVERY_LATCH_DRAWDOWN_STOP_V1)
                if source_commit is None:
                    return self._recovery_result(
                        governed=True,
                        integrity_ok=False,
                        mode_ok=True,
                        state=state,
                        high_water=high_water,
                        drawdown=drawdown,
                        latest_event=self._latest_recovery_event_conn(conn),
                        error="source commit unavailable for recovery latch",
                    )
                latched = latched_state(
                    epoch=int(state["epoch"]),
                    tripped_at=observed_at,
                    nav=nav_value,
                    high_water=high_water,
                    drawdown=drawdown,
                )
                self._set_kv_conn(
                    conn, RECOVERY_STATE_KEY, canonical_json(latched)
                )
                payload = {
                    "schema_version": latched["schema_version"],
                    "epoch": latched["epoch"],
                    "previous_state": STATE_ACTIVE,
                    "new_state": STATE_LATCHED_DD_STOP,
                    "latch_drawdown_stop_text": RECOVERY_LATCH_DRAWDOWN_STOP_TEXT,
                    "trip_id": latched["trip_id"],
                    "nav": canonical_float_text(nav_value),
                    "high_water": canonical_float_text(high_water),
                    "drawdown": canonical_float_text(drawdown),
                    "position_count": int(
                        conn.execute("SELECT COUNT(*) FROM positions").fetchone()[0]
                    ),
                    "strategy_profile": strategy_profile,
                    "broker_mode": broker_mode,
                    "db_basename": os.path.basename(self.path),
                    "source_commit": source_commit,
                    "timestamp": observed_at,
                }
                event_id = self._insert_event_conn(
                    conn,
                    ts=observed_at,
                    level="WARNING",
                    kind=EVENT_LATCHED,
                    payload=payload,
                    canonical=True,
                )
                return self._recovery_result(
                    governed=True,
                    integrity_ok=True,
                    mode_ok=True,
                    state=latched,
                    high_water=high_water,
                    drawdown=drawdown,
                    transition_occurred=True,
                    latest_event={
                        "id": event_id,
                        "ts": observed_at,
                        "kind": EVENT_LATCHED,
                        "payload": payload,
                    },
                )

            return self._recovery_result(
                governed=True,
                integrity_ok=True,
                mode_ok=True,
                state=state,
                high_water=high_water,
                drawdown=drawdown,
                latest_event=self._latest_recovery_event_conn(conn),
            )