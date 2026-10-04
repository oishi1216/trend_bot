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


def engine_source(extra: str = "") -> str:
    return f'''
class TradingEngine:
    def _update_drawdown(self, nav: float):
        current = float(self.storage.get_kv("nav_high_water", str(nav)) or nav)
        high_water = max(current, nav)
        self.storage.set_kv("nav_high_water", str(high_water))
        drawdown = 0.0 if high_water <= 0 else max(0.0, 1.0 - nav / high_water)
        return high_water, drawdown

def adaptive_risk_after_drawdown(risk_fraction, drawdown, settings):
    if drawdown >= settings.adaptive_drawdown_stop:
        return 0.0
    return risk_fraction

{extra}
'''


def write_repo(tmp_path: Path, engine: str) -> Path:
    app = tmp_path / "app"
    app.mkdir()
    (app / "engine.py").write_text(engine, encoding="utf-8")
    return tmp_path


def test_production_source_contract_detects_monotonic_latched_contract(tmp_path: Path) -> None:
    repo = write_repo(tmp_path, engine_source())
    result = audit.audit_production_source_contract(repo)
    assert result["contract_ok"] is True
    assert result["reads_persisted_nav_high_water"] is True
    assert result["monotonic_high_water_max"] is True
    assert result["persists_high_water"] is True
    assert result["adaptive_risk_zero_at_or_above_stop"] is True
    assert result["single_expected_writer"] is True
    assert result["no_automatic_reset_path_found"] is True


def test_production_source_contract_rejects_second_writer(tmp_path: Path) -> None:
    repo = write_repo(
        tmp_path,
        engine_source(
            '''
def reset_high_water(storage):
    storage.set_kv("nav_high_water", "0")
'''
        ),
    )
    result = audit.audit_production_source_contract(repo)
    assert result["contract_ok"] is False
    assert result["single_expected_writer"] is False
    assert result["no_automatic_reset_path_found"] is False


def test_production_source_contract_rejects_missing_adaptive_zeroing(tmp_path: Path) -> None:
    source = engine_source().replace(
        'if drawdown >= settings.adaptive_drawdown_stop:\n        return 0.0',
        'if drawdown >= settings.adaptive_drawdown_stop:\n        return risk_fraction',
    )
    repo = write_repo(tmp_path, source)
    result = audit.audit_production_source_contract(repo)
    assert result["contract_ok"] is False
    assert result["adaptive_risk_zero_at_or_above_stop"] is False


def test_production_source_contract_rejects_non_monotonic_high_water(tmp_path: Path) -> None:
    source = engine_source().replace(
        "high_water = max(current, nav)",
        "high_water = nav",
    )
    repo = write_repo(tmp_path, source)
    result = audit.audit_production_source_contract(repo)
    assert result["contract_ok"] is False
    assert result["monotonic_high_water_max"] is False


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
            "monotonic_high_water_max": True,
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
