from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.config import Settings
from app.engine import TradingEngine
from app.models import StrategyDecision
from app.recovery_authority import (
    EVENT_BOOTSTRAP,
    EVENT_LATCHED,
    RECOVERY_GOVERNED_DB_BASENAME,
    RECOVERY_INITIALIZED_KEY,
    RECOVERY_INITIALIZED_VALUE,
    RECOVERY_LATCH_DRAWDOWN_STOP_TEXT,
    RECOVERY_LATCH_DRAWDOWN_STOP_V1,
    RECOVERY_SCHEMA_VERSION,
    RECOVERY_SQLITE_BUSY_TIMEOUT_MS,
    RECOVERY_STATE_KEY,
    STATE_ACTIVE,
    STATE_LATCHED_DD_STOP,
    RecoveryStateError,
    active_state,
    canonical_float_text,
    canonical_json,
    governed_db_basename,
    is_governed_db,
    latched_state,
    parse_recovery_state,
    stable_trip_id,
)
from app.storage import Storage


SOURCE_COMMIT = "a" * 40
OBSERVED_AT = "2026-10-05T00:00:00+00:00"


def storage_for(
    tmp_path: Path,
    *,
    name: str = RECOVERY_GOVERNED_DB_BASENAME,
    initial_balance: float = 1_000_000.0,
) -> Storage:
    return Storage(str(tmp_path / name), initial_balance)


def db_rows(storage: Storage, query: str, params: tuple = ()) -> list[tuple]:
    with sqlite3.connect(storage.path) as conn:
        return [tuple(row) for row in conn.execute(query, params).fetchall()]


def recovery_events(storage: Storage, kind: str | None = None) -> list[tuple]:
    if kind is None:
        return db_rows(
            storage,
            "SELECT kind, payload FROM events "
            "WHERE substr(kind,1,18)='drawdown_recovery_' ORDER BY id",
        )
    return db_rows(
        storage,
        "SELECT kind, payload FROM events WHERE kind=? ORDER BY id",
        (kind,),
    )


def evaluate(
    storage: Storage,
    *,
    nav: float = 1_000_000.0,
    broker_mode: str = "mt5_paper",
    strategy_profile: str = "adaptive_dual_regime_v1",
    source_commit: str | None = SOURCE_COMMIT,
) -> dict:
    return storage.evaluate_recovery(
        nav=nav,
        broker_mode=broker_mode,
        strategy_profile=strategy_profile,
        source_commit=source_commit,
        observed_at=OBSERVED_AT,
    )


def dummy_engine(storage: Storage, settings: Settings) -> TradingEngine:
    engine = object.__new__(TradingEngine)
    engine.settings = settings
    engine.storage = storage
    return engine


def adaptive_settings(**changes) -> Settings:
    base = Settings.from_env()
    defaults = {
        "broker_mode": "mt5_paper",
        "strategy_profile": "adaptive_dual_regime_v1",
        "trading_armed": True,
        "instruments": (),
    }
    defaults.update(changes)
    return replace(base, **defaults)


def test_fixed_recovery_constants() -> None:
    assert RECOVERY_SCHEMA_VERSION == 1
    assert RECOVERY_LATCH_DRAWDOWN_STOP_V1 == pytest.approx(0.10)
    assert RECOVERY_LATCH_DRAWDOWN_STOP_TEXT == "0.1"
    assert RECOVERY_SQLITE_BUSY_TIMEOUT_MS == 5000
    assert RECOVERY_GOVERNED_DB_BASENAME == "adaptive_paper.sqlite3"


def test_threshold_is_independent_of_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DRAWDOWN_STOP", "0.99")
    monkeypatch.setenv("ADAPTIVE_DRAWDOWN_STOP", "0.88")
    monkeypatch.setenv("STRATEGY_PROFILE", "legacy_trend_v1")
    settings = Settings.from_env()
    assert settings.drawdown_stop == pytest.approx(0.99)
    assert settings.adaptive_drawdown_stop == pytest.approx(0.88)
    assert RECOVERY_LATCH_DRAWDOWN_STOP_V1 == pytest.approx(0.10)


def test_governed_basename_is_case_insensitive_and_identity_can_govern() -> None:
    assert governed_db_basename(r"C:\x\ADAPTIVE_PAPER.SQLITE3")
    assert is_governed_db(
        r"C:\x\other.sqlite3",
        state_present=True,
        marker_present=False,
    )
    assert is_governed_db(
        r"C:\x\other.sqlite3",
        state_present=False,
        marker_present=True,
    )


def test_canonical_json_and_float_are_deterministic() -> None:
    assert canonical_json({"b": 2, "a": 1}) == '{"a":1,"b":2}'
    assert canonical_float_text(0.1) == repr(0.1)
    assert canonical_float_text(1_000_000.0) == "1000000.0"


def test_trip_id_is_stable_and_state_round_trips() -> None:
    state = latched_state(
        epoch=0,
        tripped_at=OBSERVED_AT,
        nav=900_000.0,
        high_water=1_000_000.0,
        drawdown=0.10,
    )
    assert state["trip_id"] == stable_trip_id(
        epoch=0,
        tripped_at=OBSERVED_AT,
        nav=900_000.0,
        high_water=1_000_000.0,
        drawdown=0.10,
    )
    parsed = parse_recovery_state(canonical_json(state))
    assert parsed == state


@pytest.mark.parametrize(
    "mutator",
    [
        lambda x: {**x, "schema_version": 99},
        lambda x: {**x, "schema_version": True},
        lambda x: {**x, "epoch": False},
        lambda x: {**x, "state": "UNKNOWN"},
        lambda x: {**x, "state": STATE_ACTIVE},
        lambda x: {**x, "latch_drawdown_stop_text": "0.12"},
        lambda x: {**x, "trip_id": "bad"},
        lambda x: {**x, "rebaseline_receipt_sha256": "unexpected"},
        lambda x: {**x, "rearmed_at": OBSERVED_AT},
    ],
)
def test_state_validation_rejects_unknown_or_tampered(mutator) -> None:
    state = latched_state(
        epoch=0,
        tripped_at=OBSERVED_AT,
        nav=899_000.0,
        high_water=1_000_000.0,
        drawdown=0.101,
    )
    with pytest.raises(RecoveryStateError):
        parse_recovery_state(canonical_json(mutator(state)))


def test_active_state_rejects_stale_trip_or_future_authority_fields() -> None:
    state = active_state(epoch=0)
    for key, value in (
        ("trip_id", "stale-trip"),
        ("tripped_at", OBSERVED_AT),
        ("trip_nav", "900000.0"),
        ("trip_high_water", "1000000.0"),
        ("trip_drawdown", "0.1"),
        ("rebaseline_receipt_sha256", "unexpected"),
        ("rearmed_at", OBSERVED_AT),
    ):
        corrupted = {**state, key: value}
        with pytest.raises(RecoveryStateError):
            parse_recovery_state(canonical_json(corrupted))


def test_fresh_bootstrap_is_active_atomic_and_idempotent(tmp_path: Path) -> None:
    storage = storage_for(tmp_path)
    first = evaluate(storage)
    assert first["integrity_ok"] is True
    assert first["bootstrap_occurred"] is True
    assert first["state"]["state"] == STATE_ACTIVE
    assert storage.get_kv(RECOVERY_INITIALIZED_KEY) == RECOVERY_INITIALIZED_VALUE
    assert float(storage.get_kv("nav_high_water")) == pytest.approx(1_000_000.0)
    assert len(recovery_events(storage, EVENT_BOOTSTRAP)) == 1

    second = evaluate(storage)
    assert second["bootstrap_occurred"] is False
    assert second["state"]["state"] == STATE_ACTIVE
    assert len(recovery_events(storage, EVENT_BOOTSTRAP)) == 1


def test_bootstrap_is_first_kv_mutation(tmp_path: Path) -> None:
    storage = storage_for(tmp_path)
    assert db_rows(storage, "SELECT key,value FROM kv") == []
    result = evaluate(storage)
    assert result["bootstrap_occurred"] is True
    keys = {row[0] for row in db_rows(storage, "SELECT key,value FROM kv")}
    assert keys == {
        "nav_high_water",
        RECOVERY_STATE_KEY,
        RECOVERY_INITIALIZED_KEY,
    }


def test_legacy_high_water_below_stop_bootstraps_active_unchanged(
    tmp_path: Path,
) -> None:
    storage = storage_for(tmp_path)
    storage.set_kv("nav_high_water", "1000000.0")
    result = evaluate(storage, nav=950_000.0)
    assert result["state"]["state"] == STATE_ACTIVE
    assert storage.get_kv("nav_high_water") == "1000000.0"


@pytest.mark.parametrize("nav", [900_000.0, 899_999.0])
def test_legacy_high_water_at_or_above_fixed_stop_bootstraps_latched(
    tmp_path: Path, nav: float
) -> None:
    storage = storage_for(tmp_path)
    storage.set_kv("nav_high_water", "1000000.0")
    result = evaluate(
        storage,
        nav=nav,
        strategy_profile="legacy_trend_v1",
    )
    assert result["state"]["state"] == STATE_LATCHED_DD_STOP
    assert result["entries_blocked"] is True
    assert storage.get_kv("nav_high_water") == "1000000.0"


def test_diagnostic7_snapshot_latches_even_legacy_profile(tmp_path: Path) -> None:
    storage = storage_for(tmp_path)
    storage.set_kv("nav_high_water", "1070356.2228955375")
    result = evaluate(
        storage,
        nav=962578.1170207597,
        strategy_profile="legacy_trend_v1",
    )
    assert result["state"]["state"] == STATE_LATCHED_DD_STOP
    assert result["drawdown"] == pytest.approx(0.10069367895411063)


def test_governed_non_mt5_mode_fails_closed_without_bootstrap(
    tmp_path: Path,
) -> None:
    storage = storage_for(tmp_path)
    result = evaluate(storage, broker_mode="paper")
    assert result["entries_blocked"] is True
    assert result["mode_ok"] is False
    assert storage.get_kv(RECOVERY_STATE_KEY) is None
    assert storage.get_kv(RECOVERY_INITIALIZED_KEY) is None
    assert recovery_events(storage) == []


def test_missing_state_with_marker_fails_closed(tmp_path: Path) -> None:
    storage = storage_for(tmp_path)
    storage.set_kv(RECOVERY_INITIALIZED_KEY, RECOVERY_INITIALIZED_VALUE)
    result = evaluate(storage)
    assert result["entries_blocked"] is True
    assert result["integrity_ok"] is False
    assert storage.get_kv(RECOVERY_STATE_KEY) is None


def test_missing_state_with_recovery_event_fails_closed(tmp_path: Path) -> None:
    storage = storage_for(tmp_path)
    storage.add_event("INFO", EVENT_BOOTSTRAP, {"x": 1})
    result = evaluate(storage)
    assert result["entries_blocked"] is True
    assert result["integrity_ok"] is False
    assert storage.get_kv(RECOVERY_STATE_KEY) is None

def test_recovery_event_alone_keeps_renamed_db_governed(tmp_path: Path) -> None:
    storage = storage_for(tmp_path, name="renamed.sqlite3")
    storage.add_event("INFO", EVENT_BOOTSTRAP, {"x": 1})
    result = evaluate(storage)
    assert result["governed"] is True
    assert result["entries_blocked"] is True
    assert result["integrity_ok"] is False
    assert storage.get_kv(RECOVERY_STATE_KEY) is None


def test_historical_db_with_missing_high_water_fails_closed(tmp_path: Path) -> None:
    storage = storage_for(tmp_path)
    storage.add_event("INFO", "ordinary_event", {"x": 1})
    result = evaluate(storage)
    assert result["entries_blocked"] is True
    assert result["integrity_ok"] is False
    assert storage.get_kv("nav_high_water") is None


@pytest.mark.parametrize("value", ["oops", "nan", "inf", "0", "-1"])
def test_present_invalid_high_water_fails_closed_without_reseed(
    tmp_path: Path, value: str
) -> None:
    storage = storage_for(tmp_path)
    storage.set_kv("nav_high_water", value)
    result = evaluate(storage)
    assert result["entries_blocked"] is True
    assert result["integrity_ok"] is False
    assert storage.get_kv("nav_high_water") == value
    assert storage.get_kv(RECOVERY_STATE_KEY) is None


def test_active_state_then_missing_high_water_fails_closed(tmp_path: Path) -> None:
    storage = storage_for(tmp_path)
    assert evaluate(storage)["state"]["state"] == STATE_ACTIVE
    with sqlite3.connect(storage.path) as conn:
        conn.execute("DELETE FROM kv WHERE key='nav_high_water'")
        conn.commit()
    result = evaluate(storage)
    assert result["entries_blocked"] is True
    assert result["integrity_ok"] is False
    assert storage.get_kv("nav_high_water") is None


def test_active_state_then_invalid_high_water_fails_closed_no_reseed(
    tmp_path: Path,
) -> None:
    storage = storage_for(tmp_path)
    assert evaluate(storage)["state"]["state"] == STATE_ACTIVE
    storage.set_kv("nav_high_water", "nan")
    result = evaluate(storage)
    assert result["entries_blocked"] is True
    assert result["integrity_ok"] is False
    assert storage.get_kv("nav_high_water") == "nan"


def test_state_marker_inconsistency_fails_closed(tmp_path: Path) -> None:
    storage = storage_for(tmp_path)
    storage.set_kv(RECOVERY_STATE_KEY, canonical_json(active_state()))
    storage.set_kv("nav_high_water", "1000000.0")
    result = evaluate(storage)
    assert result["entries_blocked"] is True
    assert result["integrity_ok"] is False


def test_source_commit_required_for_bootstrap_transition(tmp_path: Path) -> None:
    storage = storage_for(tmp_path)
    result = evaluate(storage, source_commit=None)
    assert result["entries_blocked"] is True
    assert result["integrity_ok"] is False
    assert storage.get_kv(RECOVERY_STATE_KEY) is None


def test_event_failure_rolls_back_fresh_bootstrap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    storage = storage_for(tmp_path)

    def boom(*_args, **_kwargs):
        raise sqlite3.OperationalError("event insert failed")

    monkeypatch.setattr(Storage, "_insert_event_conn", staticmethod(boom))
    result = evaluate(storage)
    assert result["entries_blocked"] is True
    assert result["integrity_ok"] is False
    assert db_rows(storage, "SELECT key,value FROM kv") == []
    assert db_rows(storage, "SELECT kind FROM events") == []


def test_state_write_failure_rolls_back_high_water(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    storage = storage_for(tmp_path)
    original = Storage._set_kv_conn

    def fail_state(conn, key, value):
        if key == RECOVERY_STATE_KEY:
            raise sqlite3.OperationalError("state write failed")
        return original(conn, key, value)

    monkeypatch.setattr(Storage, "_set_kv_conn", staticmethod(fail_state))
    result = evaluate(storage)
    assert result["entries_blocked"] is True
    assert db_rows(storage, "SELECT key,value FROM kv") == []
    assert db_rows(storage, "SELECT kind FROM events") == []


def test_lock_failure_is_fail_closed_without_fallback_reads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    storage = storage_for(tmp_path)

    @contextmanager
    def locked():
        raise sqlite3.OperationalError("database is locked")
        yield  # pragma: no cover

    monkeypatch.setattr(storage, "_recovery_conn", locked)
    result = evaluate(storage)
    assert result["entries_blocked"] is True
    assert result["integrity_ok"] is False
    assert "locked" in result["error"]


def test_competing_fresh_bootstrap_creates_one_bootstrap_event(
    tmp_path: Path,
) -> None:
    path = tmp_path / RECOVERY_GOVERNED_DB_BASENAME
    one = Storage(str(path), 1_000_000.0)
    two = Storage(str(path), 1_000_000.0)
    barrier = threading.Barrier(2)
    results: list[dict] = []

    def run(storage: Storage) -> None:
        barrier.wait()
        results.append(evaluate(storage))

    threads = [threading.Thread(target=run, args=(one,)), threading.Thread(target=run, args=(two,))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert len(results) == 2
    assert sum(bool(item["bootstrap_occurred"]) for item in results) == 1
    assert len(recovery_events(one, EVENT_BOOTSTRAP)) == 1
    assert parse_recovery_state(one.get_kv(RECOVERY_STATE_KEY))["state"] == STATE_ACTIVE


def test_competing_latch_creates_one_trip_and_event(tmp_path: Path) -> None:
    path = tmp_path / RECOVERY_GOVERNED_DB_BASENAME
    one = Storage(str(path), 1_000_000.0)
    two = Storage(str(path), 1_000_000.0)
    assert evaluate(one)["state"]["state"] == STATE_ACTIVE

    barrier = threading.Barrier(2)
    results: list[dict] = []

    def run(storage: Storage) -> None:
        barrier.wait()
        results.append(evaluate(storage, nav=899_000.0))

    threads = [threading.Thread(target=run, args=(one,)), threading.Thread(target=run, args=(two,))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert len(results) == 2
    assert len(recovery_events(one, EVENT_LATCHED)) == 1
    state = parse_recovery_state(one.get_kv(RECOVERY_STATE_KEY))
    assert state["state"] == STATE_LATCHED_DD_STOP
    assert all(item["state"]["trip_id"] == state["trip_id"] for item in results)


def test_latched_state_does_not_self_clear_after_nav_recovery(tmp_path: Path) -> None:
    storage = storage_for(tmp_path)
    storage.set_kv("nav_high_water", "1000000.0")
    latched = evaluate(storage, nav=899_000.0)
    assert latched["state"]["state"] == STATE_LATCHED_DD_STOP

    recovered = evaluate(storage, nav=1_010_000.0)
    assert recovered["state"]["state"] == STATE_LATCHED_DD_STOP
    assert recovered["entries_blocked"] is True
    assert recovered["high_water"] == pytest.approx(1_010_000.0)
    assert float(storage.get_kv("nav_high_water")) == pytest.approx(1_010_000.0)
    assert len(recovery_events(storage, EVENT_LATCHED)) == 0
    assert len(recovery_events(storage, EVENT_BOOTSTRAP)) == 1


def test_active_to_latched_transition_is_idempotent(tmp_path: Path) -> None:
    storage = storage_for(tmp_path)
    assert evaluate(storage)["state"]["state"] == STATE_ACTIVE
    first = evaluate(storage, nav=899_000.0)
    second = evaluate(storage, nav=899_000.0)
    assert first["state"]["state"] == STATE_LATCHED_DD_STOP
    assert second["state"]["trip_id"] == first["state"]["trip_id"]
    assert len(recovery_events(storage, EVENT_LATCHED)) == 1


def test_recovery_entries_override_trading_armed_and_profile(tmp_path: Path) -> None:
    storage = storage_for(tmp_path)
    storage.set_kv("nav_high_water", "1000000.0")
    recovery = evaluate(storage, nav=899_000.0, strategy_profile="legacy_trend_v1")
    settings = adaptive_settings(strategy_profile="legacy_trend_v1", trading_armed=True)
    engine = dummy_engine(storage, settings)
    enabled, reason = engine._entries_enabled(recovery)
    assert enabled is False
    assert "LATCHED_DD_STOP" in reason


def test_final_open_path_rechecks_persistent_recovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    storage = storage_for(tmp_path)
    storage.set_kv("nav_high_water", "1000000.0")
    evaluate(storage, nav=899_000.0)
    settings = adaptive_settings(strategy_profile="legacy_trend_v1", trading_armed=True)
    engine = dummy_engine(storage, settings)

    class Market:
        @staticmethod
        def mid_price(_instrument):
            return 1.10

        @staticmethod
        def conversion_rate(_quote, _home):
            return 1.0

    class Paper:
        @staticmethod
        def market_order(*_args, **_kwargs):
            raise AssertionError("market_order must not run while recovery is latched")

    engine.market_data = Market()
    engine.paper = Paper()
    engine.oanda = SimpleNamespace()
    engine._portfolio_metrics = lambda: (0.0, 0.0)

    monkeypatch.setattr(
        "app.engine.calculate_units",
        lambda **_kwargs: SimpleNamespace(units=1, reason="ok"),
    )
    decision = StrategyDecision(
        action="enter_long",
        candle_time="2026-10-05",
        close=1.10,
        atr=0.01,
        ema=1.0,
        reason="test",
        score=90,
        regime="trend",
        entry_kind="breakout",
        stop_atr_multiple=2.0,
        risk_fraction=0.005,
    )
    result = engine._open_candidate(
        instrument="EURUSD",
        decision=decision,
        nav=899_000.0,
        drawdown=0.0,
        monthly_loss=0.0,
        entries_enabled=True,
        armed_reason="armed",
    )
    assert result["status"].startswith("entry_blocked: recovery_state:LATCHED_DD_STOP")


def test_status_is_read_only_and_does_not_bootstrap(tmp_path: Path) -> None:
    storage = storage_for(tmp_path)
    settings = adaptive_settings()
    engine = dummy_engine(storage, settings)
    engine.nav = lambda: 1_000_000.0

    before_kv = db_rows(storage, "SELECT key,value FROM kv ORDER BY key")
    before_events = db_rows(storage, "SELECT kind,payload FROM events ORDER BY id")
    status = engine.status()
    after_kv = db_rows(storage, "SELECT key,value FROM kv ORDER BY key")
    after_events = db_rows(storage, "SELECT kind,payload FROM events ORDER BY id")

    assert before_kv == after_kv == []
    assert before_events == after_events == []
    assert status["recovery"]["recovery_integrity_ok"] is False
    assert status["recovery"]["recovery_entries_blocked"] is True


def test_status_corrupt_state_is_read_only(tmp_path: Path) -> None:
    storage = storage_for(tmp_path)
    storage.set_kv(RECOVERY_STATE_KEY, "{bad")
    storage.set_kv(RECOVERY_INITIALIZED_KEY, RECOVERY_INITIALIZED_VALUE)
    storage.set_kv("nav_high_water", "1000000.0")
    settings = adaptive_settings()
    engine = dummy_engine(storage, settings)
    engine.nav = lambda: 950_000.0

    before = db_rows(storage, "SELECT key,value FROM kv ORDER BY key")
    event_before = db_rows(storage, "SELECT kind,payload FROM events ORDER BY id")
    status = engine.status()
    after = db_rows(storage, "SELECT key,value FROM kv ORDER BY key")
    event_after = db_rows(storage, "SELECT kind,payload FROM events ORDER BY id")

    assert before == after
    assert event_before == event_after
    assert status["recovery"]["recovery_integrity_ok"] is False


def test_run_once_source_orders_recovery_before_monthly_and_before_open() -> None:
    source = Path(__import__("app.engine", fromlist=["x"]).__file__).read_text(
        encoding="utf-8"
    )
    run = source[source.index("    def run_once("):source.index("    def status(")]
    first_recovery = run.index("recovery = self._evaluate_recovery")
    monthly = run.index("self._monthly_metrics")
    second_recovery = run.index(
        "recovery = self._evaluate_recovery",
        first_recovery + 1,
    )
    open_candidate = run.index("self._open_candidate")
    stop_processing = run.index("self.paper.process_stop")
    assert first_recovery < monthly
    assert stop_processing < second_recovery < open_candidate


def test_all_engine_high_water_writes_route_through_storage_atomic_primitive() -> None:
    source = Path(__import__("app.engine", fromlist=["x"]).__file__).read_text(
        encoding="utf-8"
    )
    assert 'set_kv("nav_high_water"' not in source
    assert "update_nav_high_water_atomic" in source


def test_no_rebaseline_rearm_probation_mutation_is_implemented() -> None:
    for module_name in ("app.recovery_authority", "app.storage", "app.engine"):
        path = Path(__import__(module_name, fromlist=["x"]).__file__)
        source = path.read_text(encoding="utf-8")
        tree = __import__("ast").parse(source)
        names = {
            node.name.lower()
            for node in __import__("ast").walk(tree)
            if isinstance(node, (__import__("ast").FunctionDef, __import__("ast").AsyncFunctionDef))
        }
        assert not any("rebaseline" in name for name in names)
        assert not any("rearm" in name for name in names)
    assert recovery_events  # keep the negative-capability helper referenced