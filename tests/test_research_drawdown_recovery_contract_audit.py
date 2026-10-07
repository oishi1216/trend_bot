from __future__ import annotations

import ast
from pathlib import Path

import pytest

from app.config import Settings
from app import research_drawdown_recovery_contract_audit as audit


def state(
    *,
    nav: float,
    high_water: float,
    drawdown: float,
    start_open: list[str] | None = None,
    gate_open: list[str] | None = None,
    realized: float | None = None,
) -> dict:
    return {
        "current_nav": nav,
        "high_water": high_water,
        "drawdown": drawdown,
        "open_at_date_start": list(start_open or []),
        "open_at_gate": list(gate_open or []),
        "realized_equity_before_date_exits": nav if realized is None else realized,
    }


def gate(reason: str) -> dict:
    return {"gate": {"reason": reason}}


def terminal_lifecycle() -> dict:
    return {
        "first_drawdown_stop_crossing": "2024-01-02",
        "self_recovery_observed": False,
        "terminal_flat_exists": True,
        "terminal_flat_state": {
            "at_or_below_recovery_boundary": True,
        },
        "post_terminal": {
            "new_entries": 0,
            "nav_constant": True,
            "high_water_constant": True,
            "drawdown_constant": True,
            "all_drawdown_at_or_above_stop": True,
            "all_candidate_days_blocked_dd_stop": True,
        },
    }


def test_control_settings_uses_exact_experiment4_control_only() -> None:
    control = audit.control_settings(Settings.from_env())
    assert control.strategy_profile == "adaptive_dual_regime_v1"
    assert control.adaptive_breakeven_trigger_r == 1.0
    assert control.adaptive_drawdown_stop == pytest.approx(0.10)


def test_no_treatment_constants_or_new_thresholds_are_defined() -> None:
    source = Path(audit.__file__).read_text(encoding="utf-8")
    assert "TREATMENT_BREAKEVEN_TRIGGER_R" not in source
    assert "adaptive_drawdown_stop=1.0" not in source
    assert "0.5R" not in source


def test_lifecycle_detects_crossing_terminal_flat_and_constant_lock() -> None:
    dates = [
        "2024-01-01T00:00:00+00:00",
        "2024-01-02T00:00:00+00:00",
        "2024-01-03T00:00:00+00:00",
        "2024-01-04T00:00:00+00:00",
    ]
    daily = {
        dates[0]: state(nav=100.0, high_water=100.0, drawdown=0.0, start_open=["EURUSD"], gate_open=["EURUSD"]),
        dates[1]: state(nav=89.9, high_water=100.0, drawdown=0.101, start_open=["EURUSD"], gate_open=[]),
        dates[2]: state(nav=89.9, high_water=100.0, drawdown=0.101),
        dates[3]: state(nav=89.9, high_water=100.0, drawdown=0.101),
    }
    simulation = {
        "dates": dates,
        "trades": [
            {
                "instrument": "EURUSD",
                "side": "long",
                "entry_date": dates[0],
                "exit_date": dates[1],
            }
        ],
    }
    gate_trace = {
        "by_date": {
            dates[1]: gate("blocked_dd_stop"),
            dates[2]: gate("blocked_dd_stop"),
            dates[3]: gate("blocked_dd_stop"),
        }
    }

    result = audit.audit_lifecycle(
        simulation,
        daily,
        gate_trace,
        drawdown_stop=0.10,
    )

    assert result["first_drawdown_stop_crossing"] == dates[1]
    assert result["first_candidate_blocked_dd_stop"] == dates[1]
    assert result["last_open_position_exit_date_after_crossing"] == dates[1]
    assert result["first_flat_at_gate_date"] == dates[1]
    assert result["terminal_flat_date"] == dates[2]
    assert result["terminal_flat_state"]["recovery_boundary_nav"] == pytest.approx(90.0)
    assert result["terminal_flat_state"]["gap_to_recovery_boundary"] == pytest.approx(0.1)
    assert result["terminal_flat_state"]["at_or_below_recovery_boundary"] is True
    assert result["post_terminal"]["new_entries"] == 0
    assert result["post_terminal"]["candidate_days"] == 2
    assert result["post_terminal"]["candidate_gate_reasons"] == {"blocked_dd_stop": 2}
    assert result["post_terminal"]["nav_constant"] is True
    assert result["post_terminal"]["high_water_constant"] is True
    assert result["post_terminal"]["drawdown_constant"] is True
    assert result["post_terminal"]["all_candidate_days_blocked_dd_stop"] is True
    assert result["self_recovery_observed"] is False


def test_lifecycle_requires_entry_after_dd_recovery_for_self_recovery() -> None:
    dates = [
        "2024-01-01T00:00:00+00:00",
        "2024-01-02T00:00:00+00:00",
        "2024-01-03T00:00:00+00:00",
        "2024-01-04T00:00:00+00:00",
    ]
    daily = {
        dates[0]: state(nav=100.0, high_water=100.0, drawdown=0.0),
        dates[1]: state(nav=89.0, high_water=100.0, drawdown=0.11),
        dates[2]: state(nav=91.0, high_water=100.0, drawdown=0.09),
        dates[3]: state(nav=92.0, high_water=100.0, drawdown=0.08),
    }
    no_entry = audit.audit_lifecycle(
        {"dates": dates, "trades": []},
        daily,
        {"by_date": {dates[1]: gate("blocked_dd_stop")}},
        drawdown_stop=0.10,
    )
    assert no_entry["first_later_recovery_below_stop"] == dates[2]
    assert no_entry["first_entry_after_recovery"] is None
    assert no_entry["self_recovery_observed"] is False

    with_entry = audit.audit_lifecycle(
        {
            "dates": dates,
            "trades": [
                {
                    "instrument": "EURUSD",
                    "side": "long",
                    "entry_date": dates[2],
                    "exit_date": dates[3],
                }
            ],
        },
        daily,
        {"by_date": {dates[1]: gate("blocked_dd_stop"), dates[2]: gate("PASS")}},
        drawdown_stop=0.10,
    )
    assert with_entry["first_later_recovery_below_stop"] == dates[2]
    assert with_entry["first_entry_after_recovery"] == dates[2]
    assert with_entry["self_recovery_observed"] is True


def test_recovery_boundary_strict_semantics() -> None:
    high_water = 1_000_000.0
    stop = 0.10
    boundary = high_water * (1.0 - stop)
    assert boundary == 900_000.0
    assert 1.0 - boundary / high_water == pytest.approx(stop)
    assert (1.0 - (boundary - 0.01) / high_water) > stop
    assert (1.0 - (boundary + 0.01) / high_water) < stop


def test_classification_terminal_lock() -> None:
    label, _ = audit.classify_contract(
        anchor_ok=True,
        lifecycle=terminal_lifecycle(),
        production_contract_ok=True,
    )
    assert label == "TERMINAL_LOCK_CONFIRMED"


def test_classification_self_recovery_observed() -> None:
    life = terminal_lifecycle()
    life["self_recovery_observed"] = True
    label, _ = audit.classify_contract(
        anchor_ok=True,
        lifecycle=life,
        production_contract_ok=True,
    )
    assert label == "SELF_RECOVERY_OBSERVED"


def test_classification_contract_divergence() -> None:
    label, _ = audit.classify_contract(
        anchor_ok=True,
        lifecycle=terminal_lifecycle(),
        production_contract_ok=False,
    )
    assert label == "CONTRACT_DIVERGENCE"


def test_classification_inconclusive_when_terminal_conditions_incomplete() -> None:
    life = terminal_lifecycle()
    life["post_terminal"]["nav_constant"] = False
    label, _ = audit.classify_contract(
        anchor_ok=True,
        lifecycle=life,
        production_contract_ok=True,
    )
    assert label == "INCONCLUSIVE"


def engine_source(
    *,
    high_water_delegate: bool = True,
    recovery_delegate: bool = True,
    risk_zeroing: bool = True,
) -> str:
    update = (
        "return self.storage.update_nav_high_water_atomic(nav)"
        if high_water_delegate
        else "return nav, 0.0"
    )
    recovery = (
        "return self.storage.evaluate_recovery(nav=nav, broker_mode='mt5_paper', "
        "strategy_profile='adaptive_dual_regime_v1', source_commit='abc', "
        "observed_at='now')"
        if recovery_delegate
        else "return {'governed': False}"
    )
    risk_return = "return 0.0" if risk_zeroing else "return risk_fraction"
    return f"""
class TradingEngine:
    def _update_drawdown(self, nav: float):
        {update}

    def _evaluate_recovery(self, nav: float, source_commit):
        {recovery}

    def run_once(self):
        nav = 100.0
        first = self._evaluate_recovery(nav, "abc")
        second = self._evaluate_recovery(nav, "abc")
        return first, second

def adaptive_risk_after_drawdown(risk_fraction, drawdown, settings):
    if drawdown >= settings.adaptive_drawdown_stop:
        {risk_return}
    return risk_fraction
"""


def storage_source(
    *,
    monotonic: bool = True,
    self_clear: bool = False,
    extra: str = "",
) -> str:
    update = (
        "high_water = max(current, nav_value)"
        if monotonic
        else "high_water = nav_value"
    )
    self_clear_line = "state = active_state(epoch=1)" if self_clear else ""
    return f"""
from .recovery_authority import RECOVERY_STATE_KEY

class Storage:
    @staticmethod
    def _set_kv_conn(conn, key, value):
        conn.execute(
            "INSERT OR REPLACE INTO kv(key, value) VALUES(?,?)",
            (key, value),
        )

    def update_nav_high_water_atomic(self, nav: float):
        nav_value = float(nav)
        with self._recovery_conn() as conn:
            raw = self._get_kv_conn(conn, "nav_high_water")
            current = parse_positive_float_text(raw, name="nav_high_water")
            {update}
            self._set_kv_conn(
                conn, "nav_high_water", canonical_float_text(high_water)
            )
            return high_water, calculate_drawdown(nav_value, high_water)

    def _evaluate_recovery_locked(self, nav: float):
        nav_value = float(nav)
        with self._recovery_conn() as conn:
            raw_state = self._get_kv_conn(conn, RECOVERY_STATE_KEY)
            raw_high_water = self._get_kv_conn(conn, "nav_high_water")
            if raw_state is None:
                state = active_state(epoch=0)
                self._set_kv_conn(
                    conn, RECOVERY_STATE_KEY, canonical_json(state)
                )
                return state
            state = parse_recovery_state(raw_state)
            {self_clear_line}
            high_water = parse_positive_float_text(
                raw_high_water, name="nav_high_water"
            )
            if (
                state["state"] in {{STATE_ACTIVE, STATE_LATCHED_DD_STOP}}
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
                latched = latched_state(
                    epoch=0,
                    tripped_at="now",
                    nav=nav_value,
                    high_water=high_water,
                    drawdown=drawdown,
                )
                self._set_kv_conn(
                    conn, RECOVERY_STATE_KEY, canonical_json(latched)
                )
                return latched
            return state

{extra}
"""


def recovery_source(*, threshold: str = "0.10") -> str:
    return f"""
RECOVERY_LATCH_DRAWDOWN_STOP_V1 = {threshold}
RECOVERY_LATCH_DRAWDOWN_STOP_TEXT = "0.1"
RECOVERY_STATE_KEY = "drawdown_recovery_state_v1"
STATE_ACTIVE = "ACTIVE"
STATE_LATCHED_DD_STOP = "LATCHED_DD_STOP"

def recovery_latch_reached(nav: float, high_water: float) -> bool:
    nav_value = float(nav)
    high_value = float(high_water)
    recovery_boundary = high_value * (1.0 - RECOVERY_LATCH_DRAWDOWN_STOP_V1)
    return nav_value <= recovery_boundary

def active_state(*, epoch=0):
    return {{"state": STATE_ACTIVE, "epoch": epoch}}

def latched_state(**kwargs):
    return {{"state": STATE_LATCHED_DD_STOP, **kwargs}}

def calculate_drawdown(nav_value, high_value):
    return max(0.0, 1.0 - nav_value / high_value)

def parse_positive_float_text(value, *, name):
    return float(value)

def canonical_float_text(value):
    return repr(float(value))

def canonical_json(value):
    return str(value)

def parse_recovery_state(value):
    return value
"""


def write_repo(
    tmp_path: Path,
    *,
    engine: str | None = None,
    storage: str | None = None,
    recovery: str | None = None,
) -> Path:
    app = tmp_path / "app"
    app.mkdir()
    (app / "engine.py").write_text(
        engine if engine is not None else engine_source(),
        encoding="utf-8",
    )
    (app / "storage.py").write_text(
        storage if storage is not None else storage_source(),
        encoding="utf-8",
    )
    (app / "recovery_authority.py").write_text(
        recovery if recovery is not None else recovery_source(),
        encoding="utf-8",
    )
    return tmp_path


def test_production_source_contract_accepts_current_r1_repo() -> None:
    repo = Path(audit.__file__).resolve().parents[1]
    result = audit.audit_production_source_contract(repo)
    assert result["contract_ok"] is True
    assert result["engine_high_water_delegation"] is True
    assert result["engine_recovery_delegation"] is True
    assert result["engine_run_recovery_checks"] is True
    assert result["reads_persisted_nav_high_water"] is True
    assert result["monotonic_high_water_max"] is True
    assert result["persists_high_water"] is True
    assert result["fixed_recovery_latch_10pct"] is True
    assert result["persistent_latch_transition"] is True
    assert result["persistent_latch_no_self_clear"] is True
    assert result["all_high_water_writers_governed"] is True
    assert result["unresolved_kv_mutators"] == []
    assert result["no_future_recovery_mutation"] is True
    assert result["no_automatic_reset_path_found"] is True


def test_production_source_contract_accepts_minimal_r1_shape(tmp_path: Path) -> None:
    repo = write_repo(tmp_path)
    result = audit.audit_production_source_contract(repo)
    assert result["contract_ok"] is True


def test_production_source_contract_rejects_rogue_high_water_writer(
    tmp_path: Path,
) -> None:
    repo = write_repo(tmp_path)
    (repo / "app" / "rogue.py").write_text(
        'def reset(storage):\n'
        '    storage.set_kv("nav_high_water", "0")\n',
        encoding="utf-8",
    )
    result = audit.audit_production_source_contract(repo)
    assert result["contract_ok"] is False
    assert result["all_high_water_writers_governed"] is False


def test_production_source_contract_rejects_alias_high_water_writer(
    tmp_path: Path,
) -> None:
    repo = write_repo(tmp_path)
    (repo / "app" / "alias_writer.py").write_text(
        'KEY = "nav_high_water"\n'
        'def mutate(storage):\n'
        '    storage.set_kv(KEY, "0")\n',
        encoding="utf-8",
    )
    result = audit.audit_production_source_contract(repo)
    assert result["contract_ok"] is False
    assert result["all_high_water_writers_governed"] is False
    assert any(
        row["path"] == "app/alias_writer.py"
        and row["resolved_key"] == "nav_high_water"
        for row in result["nav_high_water_writer_calls"]
    )


def test_production_source_contract_rejects_writer_outside_app(
    tmp_path: Path,
) -> None:
    repo = write_repo(tmp_path)
    scripts = repo / "scripts"
    scripts.mkdir()
    (scripts / "reset_high_water.py").write_text(
        'def mutate(storage):\n'
        '    storage.set_kv("nav_high_water", "0")\n',
        encoding="utf-8",
    )
    result = audit.audit_production_source_contract(repo)
    assert result["contract_ok"] is False
    assert result["all_high_water_writers_governed"] is False
    assert any(
        row["path"] == "scripts/reset_high_water.py"
        for row in result["nav_high_water_writer_calls"]
    )


def test_production_source_contract_fails_closed_on_unresolved_dynamic_mutator(
    tmp_path: Path,
) -> None:
    repo = write_repo(tmp_path)
    scripts = repo / "scripts"
    scripts.mkdir()
    (scripts / "dynamic_writer.py").write_text(
        'def mutate(storage, key):\n'
        '    storage.set_kv(key, "0")\n',
        encoding="utf-8",
    )
    result = audit.audit_production_source_contract(repo)
    assert result["contract_ok"] is False
    assert result["all_high_water_writers_governed"] is False
    assert any(
        row["path"] == "scripts/dynamic_writer.py"
        and row["reason"] == "unresolved_key_expression"
        for row in result["unresolved_kv_mutators"]
    )


def test_production_source_contract_fails_closed_on_parameter_shadowing(
    tmp_path: Path,
) -> None:
    repo = write_repo(tmp_path)
    (repo / "app" / "shadowed_key.py").write_text(
        'KEY = "scheduled_run"\n'
        'def mutate(storage, KEY):\n'
        '    storage.set_kv(KEY, "0")\n',
        encoding="utf-8",
    )
    result = audit.audit_production_source_contract(repo)
    assert result["contract_ok"] is False
    assert result["all_high_water_writers_governed"] is False
    assert any(
        row["path"] == "app/shadowed_key.py"
        and row["reason"] == "unresolved_key_expression"
        for row in result["unresolved_kv_mutators"]
    )


def test_production_source_contract_fails_closed_on_ambiguous_local_shadowing(
    tmp_path: Path,
) -> None:
    repo = write_repo(tmp_path)
    (repo / "app" / "ambiguous_key.py").write_text(
        'KEY = "scheduled_run"\n'
        'def mutate(storage, source):\n'
        '    KEY = "scheduled_run"\n'
        '    KEY = source\n'
        '    storage.set_kv(KEY, "0")\n',
        encoding="utf-8",
    )
    result = audit.audit_production_source_contract(repo)
    assert result["contract_ok"] is False
    assert result["all_high_water_writers_governed"] is False
    assert any(
        row["path"] == "app/ambiguous_key.py"
        and row["reason"] == "unresolved_key_expression"
        for row in result["unresolved_kv_mutators"]
    )


def test_production_source_contract_rejects_class_body_high_water_writer(
    tmp_path: Path,
) -> None:
    repo = write_repo(tmp_path)
    scripts = repo / "scripts"
    scripts.mkdir()
    (scripts / "class_body_writer.py").write_text(
        'class Rogue:\n'
        '    storage.set_kv("nav_high_water", "0")\n',
        encoding="utf-8",
    )
    result = audit.audit_production_source_contract(repo)
    assert result["contract_ok"] is False
    assert result["all_high_water_writers_governed"] is False
    assert any(
        row["path"] == "scripts/class_body_writer.py"
        and row["function"] == "<class:Rogue>"
        and row["resolved_key"] == "nav_high_water"
        for row in result["nav_high_water_writer_calls"]
    )


def test_production_source_contract_fails_closed_on_class_body_dynamic_writer(
    tmp_path: Path,
) -> None:
    repo = write_repo(tmp_path)
    scripts = repo / "scripts"
    scripts.mkdir()
    (scripts / "class_body_dynamic.py").write_text(
        'class Rogue:\n'
        '    storage.set_kv(key, "0")\n',
        encoding="utf-8",
    )
    result = audit.audit_production_source_contract(repo)
    assert result["contract_ok"] is False
    assert result["all_high_water_writers_governed"] is False
    assert any(
        row["path"] == "scripts/class_body_dynamic.py"
        and row["function"] == "<class:Rogue>"
        and row["reason"] == "unresolved_key_expression"
        for row in result["unresolved_kv_mutators"]
    )


def test_production_source_contract_rejects_nested_function_capture(
    tmp_path: Path,
) -> None:
    repo = write_repo(tmp_path)
    scripts = repo / "scripts"
    scripts.mkdir()
    (scripts / "nested_function_writer.py").write_text(
        'KEY = "scheduled_run"\n'
        'def outer(storage):\n'
        '    KEY = "nav_high_water"\n'
        '    def inner():\n'
        '        storage.set_kv(KEY, "0")\n'
        '    inner()\n',
        encoding="utf-8",
    )
    result = audit.audit_production_source_contract(repo)
    assert result["contract_ok"] is False
    assert result["all_high_water_writers_governed"] is False
    assert any(
        row["path"] == "scripts/nested_function_writer.py"
        and row["function"] == "inner"
        and row["resolved_key"] == "nav_high_water"
        for row in result["nav_high_water_writer_calls"]
    )


def test_production_source_contract_rejects_nested_class_capture(
    tmp_path: Path,
) -> None:
    repo = write_repo(tmp_path)
    scripts = repo / "scripts"
    scripts.mkdir()
    (scripts / "nested_class_writer.py").write_text(
        'KEY = "scheduled_run"\n'
        'def outer(storage):\n'
        '    KEY = "nav_high_water"\n'
        '    class Rogue:\n'
        '        storage.set_kv(KEY, "0")\n',
        encoding="utf-8",
    )
    result = audit.audit_production_source_contract(repo)
    assert result["contract_ok"] is False
    assert result["all_high_water_writers_governed"] is False
    assert any(
        row["path"] == "scripts/nested_class_writer.py"
        and row["function"] == "<class:Rogue>"
        and row["resolved_key"] == "nav_high_water"
        for row in result["nav_high_water_writer_calls"]
    )


def test_production_source_contract_fails_closed_on_mixed_loop_binding(
    tmp_path: Path,
) -> None:
    repo = write_repo(tmp_path)
    scripts = repo / "scripts"
    scripts.mkdir()
    (scripts / "mixed_loop_writer.py").write_text(
        'KEY = "scheduled_run"\n'
        'def mutate(storage, source):\n'
        '    KEY = "scheduled_run"\n'
        '    for KEY in source:\n'
        '        storage.set_kv(KEY, "0")\n',
        encoding="utf-8",
    )
    result = audit.audit_production_source_contract(repo)
    assert result["contract_ok"] is False
    assert result["all_high_water_writers_governed"] is False
    assert any(
        row["path"] == "scripts/mixed_loop_writer.py"
        and row["reason"] == "unresolved_key_expression"
        for row in result["unresolved_kv_mutators"]
    )


def test_production_source_contract_fails_closed_on_global_key_mutation(
    tmp_path: Path,
) -> None:
    repo = write_repo(tmp_path)
    scripts = repo / "scripts"
    scripts.mkdir()
    (scripts / "global_writer.py").write_text(
        'KEY = "scheduled_run"\n'
        'def set_key(source):\n'
        '    global KEY\n'
        '    KEY = source\n'
        'def mutate(storage):\n'
        '    global KEY\n'
        '    storage.set_kv(KEY, "0")\n',
        encoding="utf-8",
    )
    result = audit.audit_production_source_contract(repo)
    assert result["contract_ok"] is False
    assert result["all_high_water_writers_governed"] is False
    assert any(
        row["path"] == "scripts/global_writer.py"
        and row["function"] == "mutate"
        and row["reason"] == "unresolved_key_expression"
        for row in result["unresolved_kv_mutators"]
    )


def test_production_source_contract_fails_closed_on_nonlocal_key_mutation(
    tmp_path: Path,
) -> None:
    repo = write_repo(tmp_path)
    scripts = repo / "scripts"
    scripts.mkdir()
    (scripts / "nonlocal_writer.py").write_text(
        'KEY = "scheduled_run"\n'
        'def outer(storage, source):\n'
        '    KEY = "scheduled_run"\n'
        '    def set_key():\n'
        '        nonlocal KEY\n'
        '        KEY = source\n'
        '    def mutate():\n'
        '        nonlocal KEY\n'
        '        storage.set_kv(KEY, "0")\n'
        '    set_key()\n'
        '    mutate()\n',
        encoding="utf-8",
    )
    result = audit.audit_production_source_contract(repo)
    assert result["contract_ok"] is False
    assert result["all_high_water_writers_governed"] is False
    assert any(
        row["path"] == "scripts/nonlocal_writer.py"
        and row["function"] == "mutate"
        and row["reason"] == "unresolved_key_expression"
        for row in result["unresolved_kv_mutators"]
    )


def test_production_source_contract_fails_closed_on_global_setter_reader(
    tmp_path: Path,
) -> None:
    repo = write_repo(tmp_path)
    scripts = repo / "scripts"
    scripts.mkdir()
    (scripts / "global_setter_reader.py").write_text(
        'KEY = "scheduled_run"\n'
        'def set_key(source):\n'
        '    global KEY\n'
        '    KEY = source\n'
        'def mutate(storage, source):\n'
        '    set_key(source)\n'
        '    storage.set_kv(KEY, "0")\n',
        encoding="utf-8",
    )
    result = audit.audit_production_source_contract(repo)
    assert result["contract_ok"] is False
    assert any(
        row["path"] == "scripts/global_setter_reader.py"
        and row["reason"] == "unresolved_key_expression"
        for row in result["unresolved_kv_mutators"]
    )


def test_production_source_contract_fails_closed_on_nonlocal_setter_sibling_reader(
    tmp_path: Path,
) -> None:
    repo = write_repo(tmp_path)
    scripts = repo / "scripts"
    scripts.mkdir()
    (scripts / "nonlocal_setter_reader.py").write_text(
        'def outer(storage, source):\n'
        '    KEY = "scheduled_run"\n'
        '    def set_key():\n'
        '        nonlocal KEY\n'
        '        KEY = source\n'
        '    def mutate():\n'
        '        storage.set_kv(KEY, "0")\n'
        '    set_key()\n'
        '    mutate()\n',
        encoding="utf-8",
    )
    result = audit.audit_production_source_contract(repo)
    assert result["contract_ok"] is False
    assert any(
        row["path"] == "scripts/nonlocal_setter_reader.py"
        and row["reason"] == "unresolved_key_expression"
        for row in result["unresolved_kv_mutators"]
    )


def test_production_source_contract_freezes_alias_in_defining_scope(
    tmp_path: Path,
) -> None:
    repo = write_repo(tmp_path)
    scripts = repo / "scripts"
    scripts.mkdir()
    (scripts / "captured_alias.py").write_text(
        'def outer(storage):\n'
        '    KEY = "nav_high_water"\n'
        '    ALIAS = KEY\n'
        '    def inner():\n'
        '        KEY = "scheduled_run"\n'
        '        storage.set_kv(ALIAS, "0")\n'
        '    inner()\n',
        encoding="utf-8",
    )
    result = audit.audit_production_source_contract(repo)
    assert result["contract_ok"] is False
    assert any(
        row["path"] == "scripts/captured_alias.py"
        and row["resolved_key"] == "nav_high_water"
        for row in result["nav_high_water_writer_calls"]
    )


def test_production_source_contract_fails_closed_on_import_alias_ambiguity(
    tmp_path: Path,
) -> None:
    repo = write_repo(tmp_path)
    scripts = repo / "scripts"
    scripts.mkdir()
    (scripts / "import_alias.py").write_text(
        'from app.recovery_authority import RECOVERY_STATE_KEY as KEY\n'
        'KEY = "nav_high_water"\n'
        'def mutate(storage):\n'
        '    storage.set_kv(KEY, "0")\n',
        encoding="utf-8",
    )
    result = audit.audit_production_source_contract(repo)
    assert result["contract_ok"] is False
    assert any(
        row["path"] == "scripts/import_alias.py"
        and row["reason"] == "unresolved_key_expression"
        for row in result["unresolved_kv_mutators"]
    )


def test_production_source_contract_fails_closed_on_match_pattern_capture(
    tmp_path: Path,
) -> None:
    repo = write_repo(tmp_path)
    scripts = repo / "scripts"
    scripts.mkdir()
    (scripts / "match_capture.py").write_text(
        'def mutate(storage, source):\n'
        '    KEY = "scheduled_run"\n'
        '    match source:\n'
        '        case {"key": KEY}:\n'
        '            storage.set_kv(KEY, "0")\n',
        encoding="utf-8",
    )
    result = audit.audit_production_source_contract(repo)
    assert result["contract_ok"] is False
    assert any(
        row["path"] == "scripts/match_capture.py"
        and row["reason"] == "unresolved_key_expression"
        for row in result["unresolved_kv_mutators"]
    )


def test_production_source_contract_fails_closed_on_global_repository_fallback(
    tmp_path: Path,
) -> None:
    repo = write_repo(tmp_path)
    scripts = repo / "scripts"
    scripts.mkdir()
    (scripts / "safe_key.py").write_text(
        'KEY = "scheduled_run"\n',
        encoding="utf-8",
    )
    (scripts / "global_only.py").write_text(
        'def set_key(source):\n'
        '    global KEY\n'
        '    KEY = source\n'
        'def mutate(storage, source):\n'
        '    set_key(source)\n'
        '    storage.set_kv(KEY, "0")\n',
        encoding="utf-8",
    )
    result = audit.audit_production_source_contract(repo)
    assert result["contract_ok"] is False
    assert any(
        row["path"] == "scripts/global_only.py"
        and row["reason"] == "unresolved_key_expression"
        for row in result["unresolved_kv_mutators"]
    )


def test_production_source_contract_fails_closed_on_wrong_module_explicit_import(
    tmp_path: Path,
) -> None:
    repo = write_repo(tmp_path)
    scripts = repo / "scripts"
    scripts.mkdir()
    (scripts / "safe.py").write_text(
        'KEY = "scheduled_run"\n',
        encoding="utf-8",
    )
    (scripts / "dynamic.py").write_text(
        'KEY = input()\n',
        encoding="utf-8",
    )
    (scripts / "target.py").write_text(
        'from scripts.dynamic import KEY\n'
        'def reader(storage):\n'
        '    storage.set_kv(KEY, "0")\n',
        encoding="utf-8",
    )
    result = audit.audit_production_source_contract(repo)
    assert result["contract_ok"] is False
    assert any(
        row["path"] == "scripts/target.py"
        and row["reason"] == "unresolved_key_expression"
        for row in result["unresolved_kv_mutators"]
    )


def test_production_source_contract_fails_closed_on_wildcard_import_provenance(
    tmp_path: Path,
) -> None:
    repo = write_repo(tmp_path)
    scripts = repo / "scripts"
    scripts.mkdir()
    (scripts / "safe.py").write_text(
        'KEY = "scheduled_run"\n',
        encoding="utf-8",
    )
    (scripts / "dynamic.py").write_text(
        'KEY = input()\n',
        encoding="utf-8",
    )
    (scripts / "target.py").write_text(
        'from scripts.dynamic import *\n'
        'def reader(storage):\n'
        '    storage.set_kv(KEY, "0")\n',
        encoding="utf-8",
    )
    result = audit.audit_production_source_contract(repo)
    assert result["contract_ok"] is False
    assert any(
        row["path"] == "scripts/target.py"
        and row["reason"] == "unresolved_key_expression"
        for row in result["unresolved_kv_mutators"]
    )


def test_production_source_contract_resolves_helper_call_arguments(
    tmp_path: Path,
) -> None:
    repo = write_repo(tmp_path)
    scripts = repo / "scripts"
    scripts.mkdir()
    (scripts / "helper_argument.py").write_text(
        'KEY = "scheduled_run"\n'
        'def choose(KEY):\n'
        '    return KEY\n'
        'def reader(storage):\n'
        '    storage.set_kv(choose("nav_high_water"), "0")\n',
        encoding="utf-8",
    )
    result = audit.audit_production_source_contract(repo)
    assert result["contract_ok"] is False
    assert any(
        row["path"] == "scripts/helper_argument.py"
        and row["resolved_key"] == "nav_high_water"
        for row in result["nav_high_water_writer_calls"]
    )


def test_production_source_contract_fails_closed_on_callable_shadowing(
    tmp_path: Path,
) -> None:
    repo = write_repo(tmp_path)
    scripts = repo / "scripts"
    scripts.mkdir()
    (scripts / "helper_shadow.py").write_text(
        'def choose():\n'
        '    return "scheduled_run"\n'
        'def reader(storage, choose):\n'
        '    storage.set_kv(choose(), "0")\n',
        encoding="utf-8",
    )
    result = audit.audit_production_source_contract(repo)
    assert result["contract_ok"] is False
    assert any(
        row["path"] == "scripts/helper_shadow.py"
        and row["reason"] == "unresolved_key_expression"
        for row in result["unresolved_kv_mutators"]
    )


def test_production_source_contract_fails_closed_on_starred_mutator_args(
    tmp_path: Path,
) -> None:
    repo = write_repo(tmp_path)
    scripts = repo / "scripts"
    scripts.mkdir()
    (scripts / "starred_mutator.py").write_text(
        'def reader(storage, conn):\n'
        '    storage._set_kv_conn(*[conn, "nav_high_water"], "scheduled_run")\n',
        encoding="utf-8",
    )
    result = audit.audit_production_source_contract(repo)
    assert result["contract_ok"] is False
    assert any(
        row["path"] == "scripts/starred_mutator.py"
        and row["reason"] == "ambiguous_argument_unpacking"
        for row in result["unresolved_kv_mutators"]
    )


def test_production_source_contract_fails_closed_on_unpacked_mutator_kwargs(
    tmp_path: Path,
) -> None:
    repo = write_repo(tmp_path)
    scripts = repo / "scripts"
    scripts.mkdir()
    (scripts / "kwargs_mutator.py").write_text(
        'def reader(storage, payload):\n'
        '    storage.set_kv("scheduled_run", "0", **payload)\n',
        encoding="utf-8",
    )
    result = audit.audit_production_source_contract(repo)
    assert result["contract_ok"] is False
    assert any(
        row["path"] == "scripts/kwargs_mutator.py"
        and row["reason"] == "ambiguous_argument_unpacking"
        for row in result["unresolved_kv_mutators"]
    )


def test_production_source_contract_fails_closed_on_mutable_import_donor(
    tmp_path: Path,
) -> None:
    repo = write_repo(tmp_path)
    scripts = repo / "scripts"
    scripts.mkdir()
    (scripts / "donor.py").write_text(
        'KEY = "scheduled_run"\n'
        'def set_key(value):\n'
        '    global KEY\n'
        '    KEY = value\n',
        encoding="utf-8",
    )
    (scripts / "target.py").write_text(
        'from scripts.donor import KEY\n'
        'def reader(storage):\n'
        '    storage.set_kv(KEY, "0")\n',
        encoding="utf-8",
    )
    result = audit.audit_production_source_contract(repo)
    assert result["contract_ok"] is False
    assert any(
        row["path"] == "scripts/target.py"
        and row["reason"] == "unresolved_key_expression"
        for row in result["unresolved_kv_mutators"]
    )


def test_production_source_contract_fails_closed_on_module_helper_rebinding(
    tmp_path: Path,
) -> None:
    repo = write_repo(tmp_path)
    scripts = repo / "scripts"
    scripts.mkdir()
    (scripts / "helper_rebound.py").write_text(
        'def choose():\n'
        '    return "scheduled_run"\n'
        'choose = lambda: "nav_high_water"\n'
        'def reader(storage):\n'
        '    storage.set_kv(choose(), "0")\n',
        encoding="utf-8",
    )
    result = audit.audit_production_source_contract(repo)
    assert result["contract_ok"] is False
    assert any(
        row["path"] == "scripts/helper_rebound.py"
        and row["reason"] == "unresolved_key_expression"
        for row in result["unresolved_kv_mutators"]
    )


def test_production_source_contract_fails_closed_on_method_helper_attribute_rebinding(
    tmp_path: Path,
) -> None:
    repo = write_repo(tmp_path)
    scripts = repo / "scripts"
    scripts.mkdir()
    (scripts / "method_helper_rebound.py").write_text(
        'class C:\n'
        '    def choose(self):\n'
        '        return "scheduled_run"\n'
        '    def reader(self, storage):\n'
        '        self.choose = lambda: "nav_high_water"\n'
        '        storage.set_kv(self.choose(), "0")\n',
        encoding="utf-8",
    )
    result = audit.audit_production_source_contract(repo)
    assert result["contract_ok"] is False
    assert any(
        row["path"] == "scripts/method_helper_rebound.py"
        and row["reason"] == "unresolved_key_expression"
        for row in result["unresolved_kv_mutators"]
    )


def test_production_source_contract_freezes_helper_dependencies_in_defining_scope(
    tmp_path: Path,
) -> None:
    repo = write_repo(tmp_path)
    scripts = repo / "scripts"
    scripts.mkdir()
    (scripts / "helper_scope.py").write_text(
        'SAFE = "nav_high_water"\n'
        'def choose():\n'
        '    return SAFE\n'
        'def reader(storage):\n'
        '    SAFE = "scheduled_run"\n'
        '    storage.set_kv(choose(), "0")\n',
        encoding="utf-8",
    )
    result = audit.audit_production_source_contract(repo)
    assert result["contract_ok"] is False
    assert any(
        row["path"] == "scripts/helper_scope.py"
        and row["resolved_key"] == "nav_high_water"
        for row in result["nav_high_water_writer_calls"]
    )


def test_production_source_contract_rejects_mutator_callable_alias(
    tmp_path: Path,
) -> None:
    repo = write_repo(tmp_path)
    scripts = repo / "scripts"
    scripts.mkdir()
    (scripts / "mutator_alias.py").write_text(
        'def reader(storage):\n'
        '    write = storage.set_kv\n'
        '    write("nav_high_water", "0")\n',
        encoding="utf-8",
    )
    result = audit.audit_production_source_contract(repo)
    assert result["contract_ok"] is False
    assert any(
        row["path"] == "scripts/mutator_alias.py"
        and row["call"] == "set_kv"
        and row["resolved_key"] == "nav_high_water"
        for row in result["nav_high_water_writer_calls"]
    )


def test_production_source_contract_rejects_mutator_alias_chain(
    tmp_path: Path,
) -> None:
    repo = write_repo(tmp_path)
    scripts = repo / "scripts"
    scripts.mkdir()
    (scripts / "mutator_alias_chain.py").write_text(
        'def reader(storage):\n'
        '    write = storage.set_kv\n'
        '    again = write\n'
        '    again("nav_high_water", "0")\n',
        encoding="utf-8",
    )
    result = audit.audit_production_source_contract(repo)
    assert result["contract_ok"] is False
    assert any(
        row["path"] == "scripts/mutator_alias_chain.py"
        and row["resolved_key"] == "nav_high_water"
        for row in result["nav_high_water_writer_calls"]
    )


def test_production_source_contract_fails_closed_on_alias_in_lambda(
    tmp_path: Path,
) -> None:
    repo = write_repo(tmp_path)
    scripts = repo / "scripts"
    scripts.mkdir()
    (scripts / "mutator_alias_lambda.py").write_text(
        'def reader(storage):\n'
        '    write = storage.set_kv\n'
        '    run = lambda: write("nav_high_water", "0")\n'
        '    run()\n',
        encoding="utf-8",
    )
    result = audit.audit_production_source_contract(repo)
    assert result["contract_ok"] is False
    assert any(
        row["path"] == "scripts/mutator_alias_lambda.py"
        and row["reason"] == "unscanned_mutator_alias"
        for row in result["unresolved_kv_mutators"]
    )


def test_production_source_contract_fails_closed_on_helper_callable_parameter(
    tmp_path: Path,
) -> None:
    repo = write_repo(tmp_path)
    scripts = repo / "scripts"
    scripts.mkdir()
    (scripts / "helper_callable_parameter.py").write_text(
        'def safe():\n'
        '    return "scheduled_run"\n'
        'def choose(safe):\n'
        '    return safe()\n'
        'def reader(storage):\n'
        '    storage.set_kv(choose(lambda: "nav_high_water"), "0")\n',
        encoding="utf-8",
    )
    result = audit.audit_production_source_contract(repo)
    assert result["contract_ok"] is False
    assert any(
        row["path"] == "scripts/helper_callable_parameter.py"
        and row["reason"] == "unresolved_key_expression"
        for row in result["unresolved_kv_mutators"]
    )


def test_production_source_contract_fails_closed_on_decorated_helper(
    tmp_path: Path,
) -> None:
    repo = write_repo(tmp_path)
    scripts = repo / "scripts"
    scripts.mkdir()
    (scripts / "decorated_helper.py").write_text(
        'def decorate(fn):\n'
        '    return lambda: "nav_high_water"\n'
        '@decorate\n'
        'def choose():\n'
        '    return "scheduled_run"\n'
        'def reader(storage):\n'
        '    storage.set_kv(choose(), "0")\n',
        encoding="utf-8",
    )
    result = audit.audit_production_source_contract(repo)
    assert result["contract_ok"] is False
    assert any(
        row["path"] == "scripts/decorated_helper.py"
        and row["reason"] == "unresolved_key_expression"
        for row in result["unresolved_kv_mutators"]
    )



def _repair9_assert_contract_false(
    repo: Path,
    relative: str,
    source: str,
) -> dict:
    path = repo / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")
    result = audit.audit_production_source_contract(repo)
    assert result["contract_ok"] is False
    return result


def test_production_source_contract_fails_closed_on_mutator_alias_reassignment(tmp_path: Path) -> None:
    repo = write_repo(tmp_path)
    result = _repair9_assert_contract_false(
        repo, "scripts/alias_reassignment.py",
        'def run(storage):\n'
        '    write = storage.set_kv\n'
        '    write = storage.set_kv\n'
        '    write("nav_high_water", "0")\n',
    )
    assert any(row.get("reason") == "mutator_callable_escape" for row in result["unresolved_kv_mutators"])


def test_production_source_contract_fails_closed_on_mutator_conditional_alias(tmp_path: Path) -> None:
    repo = write_repo(tmp_path)
    _repair9_assert_contract_false(
        repo, "scripts/alias_conditional.py",
        'def run(storage, flag):\n'
        '    if flag:\n'
        '        write = storage.set_kv\n'
        '    else:\n'
        '        write = storage.set_kv\n'
        '    write("nav_high_water", "0")\n',
    )


def test_production_source_contract_fails_closed_on_mutator_destructuring_alias(tmp_path: Path) -> None:
    repo = write_repo(tmp_path)
    _repair9_assert_contract_false(
        repo, "scripts/alias_destructure.py",
        'def run(storage):\n'
        '    write, other = storage.set_kv, None\n'
        '    write("nav_high_water", "0")\n',
    )


def test_production_source_contract_fails_closed_on_mutator_attribute_escape(tmp_path: Path) -> None:
    repo = write_repo(tmp_path)
    _repair9_assert_contract_false(
        repo, "scripts/alias_attribute.py",
        'def run(storage, box):\n'
        '    box.write = storage.set_kv\n'
        '    box.write("nav_high_water", "0")\n',
    )


def test_production_source_contract_fails_closed_on_mutator_parameter_escape(tmp_path: Path) -> None:
    repo = write_repo(tmp_path)
    _repair9_assert_contract_false(
        repo, "scripts/alias_parameter.py",
        'def invoke(write):\n'
        '    write("nav_high_water", "0")\n'
        'def run(storage):\n'
        '    invoke(storage.set_kv)\n',
    )


def test_production_source_contract_fails_closed_on_mutator_default_escape(tmp_path: Path) -> None:
    repo = write_repo(tmp_path)
    _repair9_assert_contract_false(
        repo, "scripts/alias_default.py",
        'def run(storage):\n'
        '    def invoke(write=storage.set_kv):\n'
        '        write("nav_high_water", "0")\n'
        '    invoke()\n',
    )


def test_production_source_contract_fails_closed_on_mutator_return_escape(tmp_path: Path) -> None:
    repo = write_repo(tmp_path)
    _repair9_assert_contract_false(
        repo, "scripts/alias_return.py",
        'def obtain(storage):\n'
        '    return storage.set_kv\n'
        'def run(storage):\n'
        '    write = obtain(storage)\n'
        '    write("nav_high_water", "0")\n',
    )


def test_production_source_contract_fails_closed_on_mutator_expression_escape(tmp_path: Path) -> None:
    repo = write_repo(tmp_path)
    _repair9_assert_contract_false(
        repo, "scripts/alias_expression.py",
        'def run(storage, flag):\n'
        '    write = storage.set_kv if flag else storage.set_kv\n'
        '    write("nav_high_water", "0")\n',
    )


def test_production_source_contract_fails_closed_on_helper_local_lambda(tmp_path: Path) -> None:
    repo = write_repo(tmp_path)
    _repair9_assert_contract_false(
        repo, "scripts/helper_local_lambda.py",
        'def safe():\n'
        '    return "scheduled_run"\n'
        'def choose():\n'
        '    safe = lambda: "nav_high_water"\n'
        '    return safe()\n'
        'def run(storage):\n'
        '    storage.set_kv(choose(), "0")\n',
    )


def test_production_source_contract_fails_closed_on_helper_local_nested_function(tmp_path: Path) -> None:
    repo = write_repo(tmp_path)
    _repair9_assert_contract_false(
        repo, "scripts/helper_nested_function.py",
        'def safe():\n'
        '    return "scheduled_run"\n'
        'def choose():\n'
        '    def safe():\n'
        '        return "nav_high_water"\n'
        '    return safe()\n'
        'def run(storage):\n'
        '    storage.set_kv(choose(), "0")\n',
    )


def test_production_source_contract_fails_closed_on_decorated_class_method(tmp_path: Path) -> None:
    repo = write_repo(tmp_path)
    _repair9_assert_contract_false(
        repo, "scripts/decorated_class.py",
        'def decorate(cls):\n'
        '    return cls\n'
        '@decorate\n'
        'class C:\n'
        '    def choose(self):\n'
        '        return "scheduled_run"\n'
        '    def run(self, storage):\n'
        '        storage.set_kv(self.choose(), "0")\n',
    )


def test_production_source_contract_fails_closed_on_wrong_method_receiver(tmp_path: Path) -> None:
    repo = write_repo(tmp_path)
    _repair9_assert_contract_false(
        repo, "scripts/wrong_receiver.py",
        'class C:\n'
        '    def choose(self):\n'
        '        return "scheduled_run"\n'
        'def run(self, storage):\n'
        '    storage.set_kv(self.choose(), "0")\n',
    )


def test_production_source_contract_fails_closed_on_subclass_method_override(tmp_path: Path) -> None:
    repo = write_repo(tmp_path)
    _repair9_assert_contract_false(
        repo, "scripts/subclass_override.py",
        'class C:\n'
        '    def choose(self):\n'
        '        return "scheduled_run"\n'
        '    def run(self, storage):\n'
        '        storage.set_kv(self.choose(), "0")\n'
        'class D(C):\n'
        '    choose = lambda self: "nav_high_water"\n',
    )


def test_production_source_contract_rejects_missing_adaptive_zeroing(
    tmp_path: Path,
) -> None:
    repo = write_repo(
        tmp_path,
        engine=engine_source(risk_zeroing=False),
    )
    result = audit.audit_production_source_contract(repo)
    assert result["contract_ok"] is False
    assert result["adaptive_risk_zero_at_or_above_stop"] is False


def test_production_source_contract_rejects_non_monotonic_high_water(
    tmp_path: Path,
) -> None:
    repo = write_repo(
        tmp_path,
        storage=storage_source(monotonic=False),
    )
    result = audit.audit_production_source_contract(repo)
    assert result["contract_ok"] is False
    assert result["monotonic_high_water_max"] is False


def test_production_source_contract_rejects_non_10pct_latch(
    tmp_path: Path,
) -> None:
    repo = write_repo(
        tmp_path,
        recovery=recovery_source(threshold="0.20"),
    )
    result = audit.audit_production_source_contract(repo)
    assert result["contract_ok"] is False
    assert result["fixed_recovery_latch_10pct"] is False


def test_production_source_contract_rejects_latched_self_clear(
    tmp_path: Path,
) -> None:
    repo = write_repo(
        tmp_path,
        storage=storage_source(self_clear=True),
    )
    result = audit.audit_production_source_contract(repo)
    assert result["contract_ok"] is False
    assert result["persistent_latch_no_self_clear"] is False


def test_production_source_contract_rejects_future_rearm_mutator(
    tmp_path: Path,
) -> None:
    repo = write_repo(
        tmp_path,
        storage=storage_source(
            extra="""
def rearm_recovery(storage):
    storage.set_kv(RECOVERY_STATE_KEY, "ACTIVE")
"""
        ),
    )
    result = audit.audit_production_source_contract(repo)
    assert result["contract_ok"] is False
    assert result["no_future_recovery_mutation"] is False
    assert result["no_automatic_reset_path_found"] is False


def test_production_source_contract_rejects_missing_engine_recovery_delegation(
    tmp_path: Path,
) -> None:
    repo = write_repo(
        tmp_path,
        engine=engine_source(recovery_delegate=False),
    )
    result = audit.audit_production_source_contract(repo)
    assert result["contract_ok"] is False
    assert result["engine_recovery_delegation"] is False


def test_diagnostic6_corroboration(tmp_path: Path) -> None:
    payload = {
        "classification": "TARGET_GATE_BLOCK_DOMINANT",
        "target_records": [
            {
                "control_selected_gate": {
                    "reason": "blocked_dd_stop",
                    "drawdown": audit.EXPECTED_DIAGNOSTIC6_DD,
                }
            }
            for _ in range(audit.EXPECTED_DIAGNOSTIC6_TARGETS)
        ],
    }
    path = tmp_path / "d6.json"
    path.write_text(__import__("json").dumps(payload), encoding="utf-8")
    result = audit.corroborate_diagnostic6(path)
    assert result["matches"] is True
    assert result["target_count"] == 18


def test_diagnostic6_corroboration_fails_on_wrong_reason(tmp_path: Path) -> None:
    payload = {
        "classification": "TARGET_GATE_BLOCK_DOMINANT",
        "target_records": [
            {
                "control_selected_gate": {
                    "reason": "blocked_max_positions",
                    "drawdown": audit.EXPECTED_DIAGNOSTIC6_DD,
                }
            }
            for _ in range(audit.EXPECTED_DIAGNOSTIC6_TARGETS)
        ],
    }
    path = tmp_path / "d6.json"
    path.write_text(__import__("json").dumps(payload), encoding="utf-8")
    assert audit.corroborate_diagnostic6(path)["matches"] is False


def test_module_does_not_import_engine() -> None:
    source_path = Path(audit.__file__)
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module or "")
    assert "app.engine" not in imports
    assert "engine" not in imports


def test_write_report_is_deterministic(tmp_path: Path) -> None:
    payload = {
        "classification": "TERMINAL_LOCK_CONFIRMED",
        "classification_rationale": "test",
        "control_anchor": {"matches": True},
        "lifecycle": {
            "first_drawdown_stop_crossing": "2024-01-02",
            "first_candidate_blocked_dd_stop": "2024-01-02",
            "last_open_position_exit_date_after_crossing": "2024-01-02",
            "first_flat_at_gate_date": "2024-01-02",
            "terminal_flat_date": "2024-01-03",
            "terminal_flat_state": {"nav": 89.9},
            "first_later_recovery_below_stop": None,
            "post_terminal": {"candidate_days": 2},
        },
        "production_paper_contract": {
            "contract_ok": True,
            "engine_high_water_delegation": True,
            "engine_recovery_delegation": True,
            "monotonic_high_water_max": True,
            "all_high_water_writers_governed": True,
            "fixed_recovery_latch_10pct": True,
            "persistent_latch_no_self_clear": True,
            "no_future_recovery_mutation": True,
            "single_expected_writer": True,
            "no_automatic_reset_path_found": True,
            "adaptive_risk_zero_at_or_above_stop": True,
            "evidence_lines": [],
        },
        "diagnostic6_corroboration": {"matches": True},
        "policy_implication": "test",
        "limitations": ["test"],
    }
    ja = tmp_path / "a.json"
    jb = tmp_path / "b.json"
    ma = tmp_path / "a.md"
    mb = tmp_path / "b.md"
    audit.write_report(payload, ja, ma)
    audit.write_report(payload, jb, mb)
    assert ja.read_bytes() == jb.read_bytes()
    assert ma.read_bytes() == mb.read_bytes()


def test_repair10_capability_escapes(tmp_path: Path) -> None:
    cases = {
        "class_alias": 'class Box:\n    write = storage.set_kv\nBox.write("nav_high_water", "0")\n',
        "unknown_lookup": 'write = getattr(storage, name)\nwrite("nav_high_water", "0")\n',
        "getattr": 'write = getattr(storage, "set_kv")\nwrite("nav_high_water", "0")\n',
        "getattribute": 'write = storage.__getattribute__("delete_kv")\nwrite("nav_high_water")\n',
        "dynamic_lookup": 'name = "set_kv"\nwrite = getattr(storage, name)\nwrite("nav_high_water", "0")\n',
    }
    for name, source in cases.items():
        case_root = tmp_path / name
        case_root.mkdir()
        repo = write_repo(case_root)
        _repair9_assert_contract_false(repo, "scripts/escape.py", source)


def test_repair10_cross_module_mutator_export(tmp_path: Path) -> None:
    repo = write_repo(tmp_path)
    donor = repo / "app" / "donor.py"
    donor.write_text("write = Storage.set_kv\n", encoding="utf-8")
    _repair9_assert_contract_false(
        repo, "scripts/imported.py",
        'from app.donor import write\nwrite("nav_high_water", "0")\n',
    )


def test_repair10_method_receiver_frames(tmp_path: Path) -> None:
    cases = {
        "rebound": '    def run(self, storage):\n        self = other\n        storage.set_kv(self.choose(), "0")\n',
        "nested": '    def run(self, storage):\n        def invoke(self):\n            storage.set_kv(self.choose(), "0")\n        invoke(other)\n',
        "unrelated_cls": '    def run(self, cls, storage):\n        storage.set_kv(cls.choose(), "0")\n',
        "unrelated_self": '    def run(cls, self, storage):\n        storage.set_kv(self.choose(), "0")\n',
        "nonlocal": '    def run(self, storage):\n        def replace():\n            nonlocal self\n            self = other\n        storage.set_kv(self.choose(), "0")\n',
        "global": '    def run(self, storage):\n        global self\n        self = other\n        storage.set_kv(self.choose(), "0")\n',
    }
    for name, body in cases.items():
        case_root = tmp_path / name
        case_root.mkdir()
        repo = write_repo(case_root)
        _repair9_assert_contract_false(
            repo, "scripts/receiver.py",
            'class C:\n    def choose(self):\n        return "scheduled_run"\n' + body,
        )


def test_repair10_subclass_dispatch(tmp_path: Path) -> None:
    cases = {
        "alias_base": 'Base = C\nAlias = Base\nclass D(Alias):\n    choose = lambda self: "nav_high_water"\n',
        "decorated": 'def replace(cls):\n    cls.choose = lambda self: "nav_high_water"\n    return cls\n@replace\nclass D(C):\n    pass\n',
        "decorated_replacement": 'def replace(cls):\n    return type("Replacement", (C,), {"choose": lambda self: "nav_high_water"})\n@replace\nclass D(C):\n    pass\n',
        "metaclass": 'class Meta(type):\n    def __new__(meta, name, bases, ns):\n        ns["choose"] = lambda self: "nav_high_water"\n        return type.__new__(meta, name, bases, ns)\nclass D(C, metaclass=Meta):\n    pass\n',
    }
    for name, suffix in cases.items():
        case_root = tmp_path / name
        case_root.mkdir()
        repo = write_repo(case_root)
        _repair9_assert_contract_false(
            repo, "scripts/dispatch.py",
            'class C:\n    def choose(self):\n        return "scheduled_run"\n'
            '    def run(self, storage):\n        storage.set_kv(self.choose(), "0")\n' + suffix,
        )

def test_repair11_cross_module_subclass_override_fails_closed(
    tmp_path: Path,
) -> None:
    repo = write_repo(tmp_path)
    (repo / "app" / "base_escape.py").write_text(
        'class C:\n'
        '    def choose(self):\n'
        '        return "scheduled_run"\n'
        '    def run(self, storage):\n'
        '        storage.set_kv(self.choose(), "0")\n',
        encoding="utf-8",
    )
    scripts = repo / "scripts"
    scripts.mkdir()
    (scripts / "evil_subclass.py").write_text(
        'from app.base_escape import C\n'
        'class D(C):\n'
        '    def choose(self):\n'
        '        return "nav_high_water"\n'
        'def execute(storage):\n'
        '    D().run(storage)\n',
        encoding="utf-8",
    )
    assert audit.audit_production_source_contract(repo)["contract_ok"] is False


def test_repair11_cross_module_monkeypatch_fails_closed(
    tmp_path: Path,
) -> None:
    repo = write_repo(tmp_path)
    (repo / "app" / "base_patch.py").write_text(
        'class C:\n'
        '    def choose(self):\n'
        '        return "scheduled_run"\n'
        '    def run(self, storage):\n'
        '        storage.set_kv(self.choose(), "0")\n',
        encoding="utf-8",
    )
    scripts = repo / "scripts"
    scripts.mkdir()
    (scripts / "evil_patch.py").write_text(
        'from app.base_patch import C\n'
        'C.choose = lambda self: "nav_high_water"\n'
        'def execute(storage):\n'
        '    C().run(storage)\n',
        encoding="utf-8",
    )
    assert audit.audit_production_source_contract(repo)["contract_ok"] is False


@pytest.mark.parametrize(
    "source",
    [
        (
            'class C:\n'
            '    def choose(self):\n'
            '        return "scheduled_run"\n'
            '    @staticmethod\n'
            '    def run(self, storage):\n'
            '        storage.set_kv(self.choose(), "0")\n'
        ),
        (
            'class C:\n'
            '    def __getattribute__(self, name):\n'
            '        if name == "choose":\n'
            '            return lambda: "nav_high_water"\n'
            '        return object.__getattribute__(self, name)\n'
            '    def choose(self):\n'
            '        return "scheduled_run"\n'
            '    def run(self, storage):\n'
            '        storage.set_kv(self.choose(), "0")\n'
        ),
        (
            'class C:\n'
            '    def choose(self):\n'
            '        return "scheduled_run"\n'
            '    def run(self, storage):\n'
            '        vars(self)["choose"] = lambda: "nav_high_water"\n'
            '        storage.set_kv(self.choose(), "0")\n'
        ),
        (
            'class C:\n'
            '    def choose(self):\n'
            '        return "scheduled_run"\n'
            '    def run(self, storage):\n'
            '        self.__dict__["choose"] = lambda: "nav_high_water"\n'
            '        storage.set_kv(self.choose(), "0")\n'
        ),
        (
            'class C:\n'
            '    def choose(self):\n'
            '        return "scheduled_run"\n'
            '    def run(self, storage, name):\n'
            '        setattr(self, name, lambda: "nav_high_water")\n'
            '        storage.set_kv(self.choose(), "0")\n'
        ),
    ],
)
def test_repair11_receiver_dispatch_uncertainty_fails_closed(
    tmp_path: Path,
    source: str,
) -> None:
    repo = write_repo(tmp_path)
    scripts = repo / "scripts"
    scripts.mkdir()
    (scripts / "receiver_escape.py").write_text(source, encoding="utf-8")
    assert audit.audit_production_source_contract(repo)["contract_ok"] is False


@pytest.mark.parametrize(
    "source",
    [
        (
            'class Box:\n'
            '    pass\n'
            'def execute(storage, name):\n'
            '    box = Box()\n'
            '    box.w = getattr(storage, name)\n'
            '    box.w("nav_high_water", "0")\n'
        ),
        (
            'def execute(storage, name):\n'
            '    return getattr(storage, name)\n'
        ),
        (
            'def sink(write):\n'
            '    return write\n'
            'def execute(storage, name):\n'
            '    return sink(getattr(storage, name))\n'
        ),
        (
            'def execute(storage, name):\n'
            '    write = getattr(storage, name)\n'
            '    again = write\n'
            '    again("nav_high_water", "0")\n'
        ),
        (
            'import operator\n'
            'def execute(storage):\n'
            '    operator.methodcaller("set_kv", "nav_high_water", "0")(storage)\n'
        ),
        (
            'def execute(storage):\n'
            '    Storage.__dict__["set_kv"](storage, "nav_high_water", "0")\n'
        ),
    ],
)
def test_repair11_mutator_capability_escape_fails_closed(
    tmp_path: Path,
    source: str,
) -> None:
    repo = write_repo(tmp_path)
    scripts = repo / "scripts"
    scripts.mkdir()
    (scripts / "capability_escape.py").write_text(source, encoding="utf-8")
    result = audit.audit_production_source_contract(repo)
    assert result["contract_ok"] is False
    assert result["unresolved_kv_mutators"]


def test_repair11_raw_sql_kv_write_fails_closed(tmp_path: Path) -> None:
    repo = write_repo(tmp_path)
    scripts = repo / "scripts"
    scripts.mkdir()
    (scripts / "raw_sql_escape.py").write_text(
        'def execute(conn):\n'
        '    conn.execute('
        '"UPDATE kv SET value=\'0\' WHERE key=\'nav_high_water\'")\n',
        encoding="utf-8",
    )
    result = audit.audit_production_source_contract(repo)
    assert result["contract_ok"] is False
    assert any(
        row["reason"] == "raw_sql_kv_write"
        for row in result["unresolved_kv_mutators"]
    )


def test_repair11_dynamic_sql_surface_fails_closed(tmp_path: Path) -> None:
    repo = write_repo(tmp_path)
    scripts = repo / "scripts"
    scripts.mkdir()
    (scripts / "dynamic_sql_escape.py").write_text(
        'def execute(conn, sql):\n'
        '    conn.execute(sql)\n',
        encoding="utf-8",
    )
    result = audit.audit_production_source_contract(repo)
    assert result["contract_ok"] is False
    assert any(
        row["reason"] == "dynamic_sql_mutation_surface"
        for row in result["unresolved_kv_mutators"]
    )

def test_repair11_module_constant_monkeypatch_fails_closed(
    tmp_path: Path,
) -> None:
    repo = write_repo(tmp_path)
    (repo / "app" / "base_mod.py").write_text(
        'KEY = "scheduled_run"\n'
        'def run(storage):\n'
        '    storage.set_kv(KEY, "0")\n',
        encoding="utf-8",
    )
    scripts = repo / "scripts"
    scripts.mkdir()
    (scripts / "evil.py").write_text(
        'import app.base_mod as b\n'
        'b.KEY = "nav_high_water"\n'
        'def go(storage):\n'
        '    b.run(storage)\n',
        encoding="utf-8",
    )
    result = audit.audit_production_source_contract(repo)
    assert result["contract_ok"] is False
    assert result["unresolved_kv_mutators"]


def test_repair11_module_helper_monkeypatch_fails_closed(
    tmp_path: Path,
) -> None:
    repo = write_repo(tmp_path)
    (repo / "app" / "base_mod.py").write_text(
        'def choose():\n'
        '    return "scheduled_run"\n'
        'def run(storage):\n'
        '    storage.set_kv(choose(), "0")\n',
        encoding="utf-8",
    )
    scripts = repo / "scripts"
    scripts.mkdir()
    (scripts / "evil.py").write_text(
        'import app.base_mod as b\n'
        'b.choose = lambda: "nav_high_water"\n'
        'def go(storage):\n'
        '    b.run(storage)\n',
        encoding="utf-8",
    )
    assert audit.audit_production_source_contract(repo)["contract_ok"] is False


def test_repair11_foreign_receiver_cross_module_fails_closed(
    tmp_path: Path,
) -> None:
    repo = write_repo(tmp_path)
    (repo / "app" / "base_recv.py").write_text(
        'class C:\n'
        '    def choose(self):\n'
        '        return "scheduled_run"\n'
        '    def run(self, storage):\n'
        '        storage.set_kv(self.choose(), "0")\n',
        encoding="utf-8",
    )
    scripts = repo / "scripts"
    scripts.mkdir()
    (scripts / "evil.py").write_text(
        'from app.base_recv import C\n'
        'class E:\n'
        '    def choose(self):\n'
        '        return "nav_high_water"\n'
        'def go(storage):\n'
        '    C.run(E(), storage)\n',
        encoding="utf-8",
    )
    assert audit.audit_production_source_contract(repo)["contract_ok"] is False


@pytest.mark.parametrize(
    "subclass_source",
    [
        (
            'from app.base_alias import C as Parent\n'
            'class D(Parent):\n'
            '    def choose(self):\n'
            '        return "nav_high_water"\n'
        ),
        (
            'from app.base_alias import C\n'
            'Base = C\n'
            'class D(Base):\n'
            '    def choose(self):\n'
            '        return "nav_high_water"\n'
        ),
        (
            'from app.base_alias import C\n'
            'D = type("D", (C,), {"choose": lambda self: "nav_high_water"})\n'
        ),
    ],
)
def test_repair11_aliased_or_dynamic_subclass_fails_closed(
    tmp_path: Path,
    subclass_source: str,
) -> None:
    repo = write_repo(tmp_path)
    (repo / "app" / "base_alias.py").write_text(
        'class C:\n'
        '    def choose(self):\n'
        '        return "scheduled_run"\n'
        '    def run(self, storage):\n'
        '        storage.set_kv(self.choose(), "0")\n',
        encoding="utf-8",
    )
    scripts = repo / "scripts"
    scripts.mkdir()
    (scripts / "evil.py").write_text(
        subclass_source
        + 'def go(storage):\n'
        + '    D().run(storage)\n',
        encoding="utf-8",
    )
    assert audit.audit_production_source_contract(repo)["contract_ok"] is False


@pytest.mark.parametrize(
    "source",
    [
        (
            'class Box: pass\n'
            'def go(st, name):\n'
            '    box = Box()\n'
            '    box.w = getattr(st, name)\n'
            '    box.w("nav_high_water", "0")\n'
        ),
        (
            'def go(st, name):\n'
            '    return getattr(st, name)\n'
        ),
        (
            'def go(st):\n'
            '    st.__class__.__dict__["set_kv"]('
            'st, "nav_high_water", "0")\n'
        ),
        (
            'def go(st):\n'
            '    type(st).__dict__["set_kv"]('
            'st, "nav_high_water", "0")\n'
        ),
        (
            'from operator import methodcaller as mc\n'
            'def go(st):\n'
            '    mc("set_kv", "nav_high_water", "0")(st)\n'
        ),
    ],
)
def test_repair11_capability_escape_without_storage_name_fails_closed(
    tmp_path: Path,
    source: str,
) -> None:
    repo = write_repo(tmp_path)
    scripts = repo / "scripts"
    scripts.mkdir()
    (scripts / "evil.py").write_text(source, encoding="utf-8")
    result = audit.audit_production_source_contract(repo)
    assert result["contract_ok"] is False
    assert any(
        row["reason"] == "mutator_dynamic_attribute_escape"
        for row in result["unresolved_kv_mutators"]
    )


@pytest.mark.parametrize(
    "source, expected_reason",
    [
        (
            'def go(conn):\n'
            '    ex = conn.execute\n'
            '    ex("UPDATE kv SET value=\'0\' '
            'WHERE key=\'nav_high_water\'")\n',
            "raw_sql_kv_write",
        ),
        (
            'def go(conn):\n'
            '    conn.execute("INSERT OR IGNORE INTO main.kv'
            '(key,value) VALUES(\'nav_high_water\',\'0\')")\n',
            "raw_sql_kv_write",
        ),
        (
            'def go(conn):\n'
            '    conn.execute("DROP TABLE [kv]")\n',
            "raw_sql_kv_write",
        ),
        (
            'from functools import partial\n'
            'def go(conn):\n'
            '    run = partial(conn.execute, '
            '"UPDATE kv SET value=\'0\'")\n'
            '    run()\n',
            "sql_callable_escape",
        ),
    ],
)
def test_repair11_sql_alias_and_variants_fail_closed(
    tmp_path: Path,
    source: str,
    expected_reason: str,
) -> None:
    repo = write_repo(tmp_path)
    scripts = repo / "scripts"
    scripts.mkdir()
    (scripts / "evil.py").write_text(source, encoding="utf-8")
    result = audit.audit_production_source_contract(repo)
    assert result["contract_ok"] is False
    assert any(
        row["reason"] == expected_reason
        for row in result["unresolved_kv_mutators"]
    )


@pytest.mark.parametrize(
    "source",
    [
        (
            'class C:\n'
            '    def choose(self):\n'
            '        return "scheduled_run"\n'
            'c = C()\n'
            'c.__dict__["choose"] = lambda: "nav_high_water"\n'
            'def go(storage):\n'
            '    storage.set_kv(c.choose(), "0")\n'
        ),
        (
            'class C:\n'
            '    __getattribute__ = lambda self, name: '
            '(lambda: "nav_high_water")\n'
            '    def choose(self):\n'
            '        return "scheduled_run"\n'
            '    def run(self, storage):\n'
            '        storage.set_kv(self.choose(), "0")\n'
        ),
    ],
)
def test_repair11_additional_lookup_override_shapes_fail_closed(
    tmp_path: Path,
    source: str,
) -> None:
    repo = write_repo(tmp_path)
    scripts = repo / "scripts"
    scripts.mkdir()
    (scripts / "evil.py").write_text(source, encoding="utf-8")
    assert audit.audit_production_source_contract(repo)["contract_ok"] is False
@pytest.mark.parametrize(
    "source, expected_reason",
    [
        (
            'def go(conn):\n'
            '    (lambda: conn.execute('
            '"DELETE FROM kv WHERE key=\\\'nav_high_water\\\'"))()\n',
            "raw_sql_kv_write",
        ),
        (
            'def sink(f):\n'
            '    f("DELETE FROM kv WHERE key=\\\'nav_high_water\\\'")\n'
            'def go(conn):\n'
            '    ex = conn.execute\n'
            '    sink(ex)\n',
            "sql_callable_escape",
        ),
        (
            'def go(conn):\n'
            '    name = "execute"\n'
            '    getattr(conn, name)('
            '"DELETE FROM kv WHERE key=\\\'nav_high_water\\\'")\n',
            "raw_sql_kv_write",
        ),
        (
            'import operator\n'
            'def go(conn):\n'
            '    operator.methodcaller('
            '"execute", "DELETE FROM kv WHERE key=\\\'nav_high_water\\\'")(conn)\n',
            "sql_callable_escape",
        ),
        (
            'def go(conn):\n'
            '    type(conn).__dict__["execute"]('
            'conn, "DELETE FROM kv WHERE key=\\\'nav_high_water\\\'")\n',
            "dynamic_sql_mutation_surface",
        ),
    ],
)
def test_repair12_sql_closed_world_surfaces_fail_closed(
    tmp_path: Path,
    source: str,
    expected_reason: str,
) -> None:
    repo = write_repo(tmp_path)
    scripts = repo / "scripts"
    scripts.mkdir()
    (scripts / "escape.py").write_text(source, encoding="utf-8")
    result = audit.audit_production_source_contract(repo)
    assert result["contract_ok"] is False
    assert any(
        row["reason"] == expected_reason
        for row in result["unresolved_kv_mutators"]
    )


def test_repair12_module_qualified_foreign_receiver_fails_closed(
    tmp_path: Path,
) -> None:
    repo = write_repo(tmp_path)
    (repo / "app" / "base_recv.py").write_text(
        'class C:\n'
        '    def choose(self):\n'
        '        return "scheduled_run"\n'
        '    def run(self, storage):\n'
        '        storage.set_kv(self.choose(), "0")\n',
        encoding="utf-8",
    )
    scripts = repo / "scripts"
    scripts.mkdir()
    (scripts / "escape.py").write_text(
        'import app.base_recv as m\n'
        'class E:\n'
        '    def choose(self):\n'
        '        return "nav_high_water"\n'
        'def go(storage):\n'
        '    m.C.run(E(), storage)\n',
        encoding="utf-8",
    )
    assert audit.audit_production_source_contract(repo)["contract_ok"] is False


@pytest.mark.parametrize(
    "source",
    [
        (
            'from app.storage import Storage\n'
            'def go(storage):\n'
            '    w = Storage.__dict__.__getitem__("set_kv")\n'
            '    w(storage, "nav_high_water", "0")\n'
        ),
        (
            'from app.storage import Storage\n'
            'def go(storage):\n'
            '    w = dict(vars(Storage))["set_kv"]\n'
            '    w(storage, "nav_high_water", "0")\n'
        ),
        (
            'import operator\n'
            'from app.storage import Storage\n'
            'def go(storage):\n'
            '    w = operator.getitem(vars(Storage), "set_kv")\n'
            '    w(storage, "nav_high_water", "0")\n'
        ),
        (
            'from functools import reduce\n'
            'def go(storage):\n'
            '    w = reduce(getattr, ["set_kv"], storage)\n'
            '    w("nav_high_water", "0")\n'
        ),
    ],
)
def test_repair12_mutator_capability_factories_fail_closed(
    tmp_path: Path,
    source: str,
) -> None:
    repo = write_repo(tmp_path)
    scripts = repo / "scripts"
    scripts.mkdir()
    (scripts / "escape.py").write_text(source, encoding="utf-8")
    result = audit.audit_production_source_contract(repo)
    assert result["contract_ok"] is False
    assert any(
        row["reason"] == "mutator_dynamic_attribute_escape"
        for row in result["unresolved_kv_mutators"]
    )


@pytest.mark.parametrize(
    "source, expected_reason",
    [
        (
            'def go(storage):\n'
            '    exec("storage.set_kv(\\\'nav_high_water\\\', \\\'0\\\')")\n',
            "dynamic_code_execution_surface",
        ),
        (
            'def go(storage):\n'
            '    eval("storage.set_kv(\\\'nav_high_water\\\', \\\'0\\\')")\n',
            "dynamic_code_execution_surface",
        ),
        (
            'import subprocess\n'
            'def go():\n'
            '    subprocess.run(["sqlite3", "paper.db", '
            '"DELETE FROM kv WHERE key=\\\'nav_high_water\\\'"])\n',
            "out_of_process_mutation_surface",
        ),
    ],
)
def test_repair12_dynamic_execution_surfaces_fail_closed(
    tmp_path: Path,
    source: str,
    expected_reason: str,
) -> None:
    repo = write_repo(tmp_path)
    scripts = repo / "scripts"
    scripts.mkdir()
    (scripts / "escape.py").write_text(source, encoding="utf-8")
    result = audit.audit_production_source_contract(repo)
    assert result["contract_ok"] is False
    assert any(
        row["reason"] == expected_reason
        for row in result["unresolved_kv_mutators"]
    )
def test_repair13_recovery_state_clear_fails_closed(tmp_path: Path) -> None:
    repo = write_repo(tmp_path)
    scripts = repo / "scripts"
    scripts.mkdir()
    (scripts / "state_clear.py").write_text(
        'from app.recovery_authority import RECOVERY_STATE_KEY, active_state, canonical_json\n'
        'def apply(storage):\n'
        '    storage.set_kv(RECOVERY_STATE_KEY, canonical_json(active_state()))\n',
        encoding="utf-8",
    )
    result = audit.audit_production_source_contract(repo)
    assert result["contract_ok"] is False
    assert any(
        row["reason"] == "ungoverned_recovery_authority_write"
        for row in result["unresolved_kv_mutators"]
    )


def test_repair13_subprocess_alias_fails_closed(tmp_path: Path) -> None:
    repo = write_repo(tmp_path)
    scripts = repo / "scripts"
    scripts.mkdir()
    (scripts / "spawn_escape.py").write_text(
        'import subprocess as sp\n'
        'def apply():\n'
        '    sp.run(["sqlite3", "data/adaptive_paper.sqlite3", '
        '"DELETE FROM kv WHERE key=\\\'nav_high_water\\\'"])\n',
        encoding="utf-8",
    )
    result = audit.audit_production_source_contract(repo)
    assert result["contract_ok"] is False
    assert any(
        row["reason"] == "out_of_process_mutation_surface"
        for row in result["unresolved_kv_mutators"]
    )


def test_repair13_governed_db_unlink_fails_closed(tmp_path: Path) -> None:
    repo = write_repo(tmp_path)
    scripts = repo / "scripts"
    scripts.mkdir()
    (scripts / "db_replace.py").write_text(
        'from pathlib import Path\n'
        'def apply():\n'
        '    Path("data/adaptive_paper.sqlite3").unlink()\n',
        encoding="utf-8",
    )
    result = audit.audit_production_source_contract(repo)
    assert result["contract_ok"] is False
    assert any(
        row["reason"] == "governed_db_replacement_surface"
        for row in result["unresolved_kv_mutators"]
    )


def test_repair13_source_commit_subprocess_kwargs_fails_closed(
    tmp_path: Path,
) -> None:
    repo = write_repo(tmp_path)
    path = repo / "app" / "recovery_authority.py"
    path.write_text(
        path.read_text(encoding="utf-8")
        + '\nimport subprocess\n'
        + 'def source_commit_escape(repo_root, extra):\n'
        + '    return subprocess.run(["git", "-C", str(repo_root), "rev-parse", "HEAD"], '
        + 'check=True, capture_output=True, text=True, timeout=5, **extra)\n',
        encoding="utf-8",
    )
    result = audit.audit_production_source_contract(repo)
    assert result["contract_ok"] is False
    assert any(
        row["reason"] == "out_of_process_mutation_surface"
        for row in result["unresolved_kv_mutators"]
    )


def test_repair13_duplicate_storage_set_helper_fails_closed(
    tmp_path: Path,
) -> None:
    repo = write_repo(tmp_path)
    path = repo / "app" / "storage.py"
    path.write_text(
        path.read_text(encoding="utf-8")
        + '\nclass OtherWriter:\n'
        + '    @staticmethod\n'
        + '    def _set_kv_conn(conn, key, value):\n'
        + '        conn.execute("INSERT OR REPLACE INTO kv(key, value) VALUES(?,?)", '
        + '("nav_high_water", "0"))\n',
        encoding="utf-8",
    )
    result = audit.audit_production_source_contract(repo)
    assert result["contract_ok"] is False
    assert any(
        row["reason"] == "storage_set_helper_identity_ambiguous"
        for row in result["unresolved_kv_mutators"]
    )


def test_repair13_reused_sql_alias_in_lambda_fails_closed(
    tmp_path: Path,
) -> None:
    repo = write_repo(tmp_path)
    scripts = repo / "scripts"
    scripts.mkdir()
    (scripts / "alias_lambda.py").write_text(
        'def first(conn):\n'
        '    ex = conn.execute\n'
        '    ex("SELECT 1")\n'
        'def go(conn):\n'
        '    ex = conn.execute\n'
        '    (lambda: ex("DELETE FROM kv WHERE key=\\\'nav_high_water\\\'"))()\n',
        encoding="utf-8",
    )
    result = audit.audit_production_source_contract(repo)
    assert result["contract_ok"] is False
    assert any(
        row["reason"] == "raw_sql_kv_write"
        for row in result["unresolved_kv_mutators"]
    )


def test_repair13_file_order_foreign_receiver_fails_closed(
    tmp_path: Path,
) -> None:
    repo = write_repo(tmp_path)
    (repo / "app" / "base_recv.py").write_text(
        'class C:\n'
        '    def choose(self):\n'
        '        return "scheduled_run"\n'
        '    def run(self, storage):\n'
        '        storage.set_kv(self.choose(), "0")\n',
        encoding="utf-8",
    )
    (repo / "app" / "_evil.py").write_text(
        'import app.base_recv as m\n'
        'class E:\n'
        '    def choose(self):\n'
        '        return "nav_high_water"\n'
        'def go(storage):\n'
        '    m.C.run(E(), storage)\n',
        encoding="utf-8",
    )
    assert audit.audit_production_source_contract(repo)["contract_ok"] is False


@pytest.mark.parametrize(
    "source",
    [
        (
            'from app.storage import Storage\n'
            'def go(storage):\n'
            '    w = dict(vars(Storage)).get("set_kv")\n'
            '    w(storage, "nav_high_water", "0")\n'
        ),
        (
            'from app.storage import Storage\n'
            'def go(storage):\n'
            '    w = next(v for k, v in vars(Storage).items() if k == "set_kv")\n'
            '    w(storage, "nav_high_water", "0")\n'
        ),
        (
            'import inspect\n'
            'def go(conn):\n'
            '    inspect.getattr_static(type(conn), "execute")('
            'conn, "DELETE FROM kv WHERE key=\\\'nav_high_water\\\'")\n'
        ),
    ],
)
def test_repair13_reflection_capability_factories_fail_closed(
    tmp_path: Path,
    source: str,
) -> None:
    repo = write_repo(tmp_path)
    scripts = repo / "scripts"
    scripts.mkdir()
    (scripts / "reflect.py").write_text(source, encoding="utf-8")
    result = audit.audit_production_source_contract(repo)
    assert result["contract_ok"] is False
    assert result["unresolved_kv_mutators"]


def test_repair13_dynamic_getattr_call_on_conn_fails_closed(
    tmp_path: Path,
) -> None:
    repo = write_repo(tmp_path)
    scripts = repo / "scripts"
    scripts.mkdir()
    (scripts / "dynamic_call.py").write_text(
        'def go(conn, name, sql):\n'
        '    getattr(conn, name).__call__(sql)\n',
        encoding="utf-8",
    )
    result = audit.audit_production_source_contract(repo)
    assert result["contract_ok"] is False
    assert any(
        row["reason"] == "dynamic_persistent_capability_call"
        for row in result["unresolved_kv_mutators"]
    )


def test_repair13_nested_app_tests_path_is_in_scope(tmp_path: Path) -> None:
    repo = write_repo(tmp_path)
    nested = repo / "app" / "tests"
    nested.mkdir()
    (nested / "reset.py").write_text(
        'def go(storage):\n'
        '    storage.set_kv("nav_high_water", "0")\n',
        encoding="utf-8",
    )
    assert audit.audit_production_source_contract(repo)["contract_ok"] is False


def test_repair13_powershell_sqlite_mutation_fails_closed(
    tmp_path: Path,
) -> None:
    repo = write_repo(tmp_path)
    scripts = repo / "scripts"
    scripts.mkdir()
    (scripts / "evil.ps1").write_text(
        'sqlite3.exe "data/adaptive_paper.sqlite3" '
        '"DELETE FROM kv WHERE key=''nav_high_water''"\n',
        encoding="utf-8",
    )
    result = audit.audit_production_source_contract(repo)
    assert result["contract_ok"] is False
    assert any(
        row["reason"] == "non_python_recovery_mutation_surface"
        for row in result["unresolved_kv_mutators"]
    )


def test_repair13_legitimate_powershell_db_reference_is_allowed(
    tmp_path: Path,
) -> None:
    repo = write_repo(tmp_path)
    scripts = repo / "scripts"
    scripts.mkdir()
    (scripts / "run.ps1").write_text(
        '$env:DB_PATH = "data/adaptive_paper.sqlite3"\n'
        '& $pythonPath $runnerPath\n',
        encoding="utf-8",
    )
    assert audit.audit_production_source_contract(repo)["contract_ok"] is True


def test_repair14_constructed_recovery_key_fails_closed(tmp_path: Path) -> None:
    repo = write_repo(tmp_path)
    scripts = repo / "scripts"
    scripts.mkdir()
    (scripts / "constructed.py").write_text(
        'from app.recovery_authority import active_state, canonical_json\n'
        'def go(storage):\n'
        '    key = f"drawdown_recovery_state_{chr(118)}1"\n'
        '    storage.set_kv(key, canonical_json(active_state()))\n',
        encoding="utf-8",
    )
    assert audit.audit_production_source_contract(repo)["contract_ok"] is False


def test_repair14_active_rewrite_in_authority_method_fails_closed(
    tmp_path: Path,
) -> None:
    repo = write_repo(tmp_path)
    path = repo / "app" / "storage.py"
    text = path.read_text(encoding="utf-8")
    needle = "            state = parse_recovery_state(raw_state)\n"
    replacement = needle + "            state = dict(state, state=STATE_ACTIVE)\n"
    assert needle in text
    path.write_text(text.replace(needle, replacement, 1), encoding="utf-8")
    result = audit.audit_production_source_contract(repo)
    assert result["contract_ok"] is False
    assert any(
        row["reason"] == "recovery_state_value_provenance_ambiguous"
        for row in result["unresolved_kv_mutators"]
    )




def test_repair14_foreign_receiver_variable_fails_closed(
    tmp_path: Path,
) -> None:
    repo = write_repo(tmp_path)
    (repo / "app" / "base_recv.py").write_text(
        'class C:\n'
        '    def choose(self):\n'
        '        return "scheduled_run"\n'
        '    def run(self, storage):\n'
        '        storage.set_kv(self.choose(), "0")\n',
        encoding="utf-8",
    )
    (repo / "app" / "_evil.py").write_text(
        'import app.base_recv as m\n'
        'class E:\n'
        '    def choose(self):\n'
        '        return "nav_high_water"\n'
        'def go(storage):\n'
        '    e = E()\n'
        '    m.C.run(e, storage)\n',
        encoding="utf-8",
    )
    result = audit.audit_production_source_contract(repo)
    assert result["contract_ok"] is False




def test_repair14_audit_module_direct_write_fails_closed(tmp_path: Path) -> None:
    repo = write_repo(tmp_path)
    target = repo / "app" / "research_drawdown_recovery_contract_audit.py"
    target.write_text(
        Path(audit.__file__).read_text(encoding="utf-8")
        + '\n\ndef extra_writer(storage):\n'
        + '    storage.set_kv("nav_high_water", "0")\n',
        encoding="utf-8",
    )
    result = audit.audit_production_source_contract(repo)
    assert result["contract_ok"] is False
    assert any(
        row["reason"] == "audit_module_mutation_surface"
        for row in result["unresolved_kv_mutators"]
    )


def test_repair14_script_variable_db_reuse_fails_closed(tmp_path: Path) -> None:
    repo = write_repo(tmp_path)
    scripts = repo / "scripts"
    scripts.mkdir()
    (scripts / "rewrite.ps1").write_text(
        '$db = "data/adaptive_paper.sqlite3"\n'
        '# gap\n# gap\n# gap\n# gap\n'
        'Invoke-DbReplacement $db\n',
        encoding="utf-8",
    )
    result = audit.audit_production_source_contract(repo)
    assert result["contract_ok"] is False
    assert any(
        row["reason"] == "non_python_recovery_mutation_surface"
        for row in result["unresolved_kv_mutators"]
    )


def test_repair14_duplicate_async_recovery_method_fails_closed(
    tmp_path: Path,
) -> None:
    repo = write_repo(tmp_path)
    path = repo / "app" / "storage.py"
    path.write_text(
        path.read_text(encoding="utf-8")
        + '\nclass Other:\n'
        + '    async def _evaluate_recovery_locked(self, nav):\n'
        + '        return nav\n',
        encoding="utf-8",
    )
    result = audit.audit_production_source_contract(repo)
    assert result["contract_ok"] is False
    assert any(
        row["reason"] == "recovery_authority_identity_ambiguous"
        for row in result["unresolved_kv_mutators"]
    )


def test_repair14_composed_governed_db_path_fails_closed(tmp_path: Path) -> None:
    repo = write_repo(tmp_path)
    scripts = repo / "scripts"
    scripts.mkdir()
    (scripts / "path_case.py").write_text(
        'from pathlib import Path\n'
        'def go():\n'
        '    db = Path("data") / "adaptive_paper.sqlite3"\n'
        '    db.unlink()\n',
        encoding="utf-8",
    )
    result = audit.audit_production_source_contract(repo)
    assert result["contract_ok"] is False
    assert any(
        row["reason"] == "governed_db_replacement_surface"
        for row in result["unresolved_kv_mutators"]
    )


def test_repair14_function_local_import_is_resolved_without_execution() -> None:
    tree = ast.Module(
        body=[
            ast.FunctionDef(
                name="go",
                args=ast.arguments(
                    posonlyargs=[],
                    args=[],
                    kwonlyargs=[],
                    kw_defaults=[],
                    defaults=[],
                ),
                body=[
                    ast.Import(
                        names=[ast.alias(name="subprocess", asname="sp")]
                    )
                ],
                decorator_list=[],
            )
        ],
        type_ignores=[],
    )
    bindings = audit._qualified_bindings(tree)
    assert bindings["sp"] == "subprocess"
    symbol = audit._qualified_symbol(
        ast.Attribute(
            value=ast.Name(id="sp", ctx=ast.Load()),
            attr="run",
            ctx=ast.Load(),
        ),
        bindings,
    )
    assert symbol == "subprocess.run"


def test_repair14_import_alias_is_not_hidden_by_unrelated_shadow() -> None:
    tree = ast.Module(
        body=[
            ast.Import(names=[ast.alias(name="subprocess", asname="sp")]),
            ast.Import(names=[ast.alias(name="os", asname=None)]),
            ast.FunctionDef(
                name="noise",
                args=ast.arguments(
                    posonlyargs=[],
                    args=[],
                    kwonlyargs=[],
                    kw_defaults=[],
                    defaults=[],
                ),
                body=[
                    ast.Assign(
                        targets=[ast.Name(id="sp", ctx=ast.Store())],
                        value=ast.Attribute(
                            value=ast.Name(id="os", ctx=ast.Load()),
                            attr="path",
                            ctx=ast.Load(),
                        ),
                    )
                ],
                decorator_list=[],
            ),
        ],
        type_ignores=[],
    )
    bindings = audit._qualified_bindings(tree)
    assert bindings["sp"] == "subprocess"
