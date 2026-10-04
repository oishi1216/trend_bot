from __future__ import annotations

import copy
import json
import math
from dataclasses import replace
from pathlib import Path

import pytest

import app.research_no_trade_counterfactual_audit as audit
from app.config import Settings
from app.research_backtest import run_research
from app.research_no_trade_counterfactual_audit import (
    ANCHOR,
    ANCHOR_DIAGNOSTICS,
    ARM_SPECS,
    BLOCK_KEYS,
    CONTROL_REPORTED_FIELDS,
    FOLD_MIN_TRADES,
    GATE_ORDER,
    _anchor_comparison,
    _classify_gate,
    _gate_evidence,
    _json_safe,
    _walk_forward_evidence,
    arm_specifications,
    build_arm_settings,
    changed_fields,
    normalize_control,
    run_experiment,
    select_next_candidate,
    simulated_gate_diagnostics,
    write_report,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

FORWARD_EXECUTION_FILES = (
    "app/paper.py",
    "app/engine.py",
    "app/main.py",
    "app/mt5_client.py",
    "app/strategy.py",
    "app/adaptive_strategy.py",
    "app/risk.py",
)

DEFAULT_CONTROL_BLOCKS = {
    "blocked_max_positions": 144,
    "blocked_dd_stop": 55,
    "blocked_aggregate_risk": 4,
    "blocked_currency_risk": 14,
    "blocked_monthly_loss": 0,
}

EXPECTED_KEY_BY_FIELD = {spec.field: spec.expected_diagnostic_key for spec in ARM_SPECS}

BASE = Settings.from_env()
CONTROL_REF = normalize_control(BASE)

# Arm payload that is clearly overrestrictive on every check.
OVER_KW = dict(
    full_pf=0.85,
    validation_pf=1.1,
    test_pf=1.0,
    test_max_dd=0.05,
    high_cost_pf=0.9,
    folds=[{"metrics": {"trades": 5, "pf": 1.2, "max_dd": 0.05}} for _ in range(5)],
)
# Arm payload with Validation and Test PF both worse and high-cost not improving.
HARM_KW = dict(validation_pf=0.3, test_pf=0.3, high_cost_pf=0.6)


def _metrics(trades, pf, profit, cagr=0.0, max_dd=0.0, win_rate=0.0, avg_r=0.0):
    return {
        "trades": trades,
        "pf": pf,
        "profit": profit,
        "cagr": cagr,
        "max_dd": max_dd,
        "win_rate": win_rate,
        "avg_r": avg_r,
    }


def _fold(pf, trades=5, max_dd=0.05):
    return {"metrics": _metrics(trades, pf, 0.0, max_dd=max_dd)}


def _payload(
    *,
    full_trades=133,
    full_pf=ANCHOR["pf"],
    full_profit=ANCHOR["profit"],
    cagr=ANCHOR["cagr"],
    full_max_dd=ANCHOR["max_dd"],
    test_trades=13,
    test_pf=ANCHOR["test_pf"],
    test_max_dd=0.05,
    validation_pf=0.5,
    train_pf=1.0,
    high_cost_pf=0.7,
    folds=None,
):
    if folds is None:
        folds = [_fold(0.5) for _ in range(5)]
    return {
        "metrics": _metrics(full_trades, full_pf, full_profit, cagr=cagr, max_dd=full_max_dd, win_rate=0.4, avg_r=0.1),
        "train": {"metrics": _metrics(full_trades, train_pf, 0.0, max_dd=full_max_dd)},
        "validation": {"metrics": _metrics(full_trades, validation_pf, 0.0)},
        "test": {"metrics": _metrics(test_trades, test_pf, 0.0, max_dd=test_max_dd)},
        "sensitivity": {
            "base": {"metrics": _metrics(full_trades, full_pf, full_profit, cagr=cagr, max_dd=full_max_dd)},
            "cost_x2": {"metrics": _metrics(full_trades, high_cost_pf, full_profit, max_dd=full_max_dd)},
        },
        "walk_forward": {"folds": list(folds), "fold_positive": 3, "fold_total": len(folds)},
    }


def _pair(control_kw=None, arm_kw=None):
    return _gate_evidence(_payload(**(control_kw or {})), _payload(**(arm_kw or {})))


def _canned(monkeypatch, *, control=None, arms=None, control_blocks=None):
    """Patch run_research and the gate-diagnostics probe with deterministic fakes.

    Returns (calls, fake_probe). `calls` records every Settings passed to
    run_research; the arm is identified by the single field that differs from
    the normalized control.
    """
    calls = []
    arms = arms or {}
    control_payload = control if control is not None else _payload()
    blocks = dict(DEFAULT_CONTROL_BLOCKS if control_blocks is None else control_blocks)

    def fake_run(candles_by_instrument, settings, initial_equity=1_000_000.0):
        calls.append(settings)
        changed = changed_fields(CONTROL_REF, settings)
        if not changed:
            return copy.deepcopy(control_payload)
        return copy.deepcopy(arms.get(changed[0], control_payload))

    def fake_probe(candles_by_instrument, settings, initial_equity, expected_trades):
        changed = changed_fields(CONTROL_REF, settings)
        if not changed:
            return dict(blocks)
        diagnostics = dict(blocks)
        diagnostics[EXPECTED_KEY_BY_FIELD[changed[0]]] = 0
        return diagnostics

    monkeypatch.setattr(audit, "run_research", fake_run)
    return calls, fake_probe


def _run(probe):
    return run_experiment({}, BASE, initial_equity=1_000_000.0, probe=probe)


# --- normalized control -------------------------------------------------------


def test_normalized_control_resets_experimental_values():
    base = replace(
        BASE,
        adaptive_score_min=85.0,
        adaptive_strength_gap_min=0.5,
        adaptive_max_open_positions=1,
        adaptive_fast_ema_days=10,
        adaptive_mid_ema_days=75,
        adaptive_slow_ema_days=100,
    )
    control = normalize_control(base)

    assert control.adaptive_score_min == 75.0
    assert control.adaptive_strength_gap_min == 0.3
    assert control.adaptive_max_open_positions == 2
    assert control.adaptive_fast_ema_days == 20
    assert control.adaptive_mid_ema_days == 50
    assert control.adaptive_slow_ema_days == 200


def test_normalized_control_preserves_other_production_defaults():
    base = replace(BASE, adaptive_score_min=85.0)
    control = normalize_control(base)

    assert set(changed_fields(base, control)) <= {
        "adaptive_score_min",
        "adaptive_strength_gap_min",
        "adaptive_max_open_positions",
        "adaptive_fast_ema_days",
        "adaptive_mid_ema_days",
        "adaptive_slow_ema_days",
    }
    assert control.adaptive_drawdown_stop == BASE.adaptive_drawdown_stop
    assert control.adaptive_max_aggregate_risk == BASE.adaptive_max_aggregate_risk
    assert control.adaptive_max_single_currency_risk == BASE.adaptive_max_single_currency_risk
    assert control.adaptive_monthly_loss_limit == BASE.adaptive_monthly_loss_limit
    assert control.adaptive_drawdown_reduce_1 == BASE.adaptive_drawdown_reduce_1
    assert control.adaptive_drawdown_reduce_2 == BASE.adaptive_drawdown_reduce_2


def test_no_score85_maxpos1_or_ema75_leakage_into_control_or_arms():
    base = replace(
        BASE,
        adaptive_score_min=85.0,
        adaptive_max_open_positions=1,
        adaptive_mid_ema_days=75,
    )
    control = normalize_control(base)

    assert control.adaptive_score_min != 85.0
    assert control.adaptive_max_open_positions != 1
    assert control.adaptive_mid_ema_days != 75
    for spec in ARM_SPECS:
        arm = build_arm_settings(control, spec)
        assert arm.adaptive_score_min == 75.0
        assert arm.adaptive_max_open_positions in {2, 10}
        assert arm.adaptive_mid_ema_days == 50


def test_control_reported_fields_are_the_normalized_and_gate_fields():
    for field in CONTROL_REPORTED_FIELDS:
        assert hasattr(BASE, field)


# --- arm specifications and single-field difference --------------------------


def test_arm_specifications_are_fixed_and_expose_expected_keys():
    assert arm_specifications() == [
        {
            "key": "A_max_positions",
            "field": "adaptive_max_open_positions",
            "audit_value": 10,
            "expected_diagnostic_key": "blocked_max_positions",
        },
        {
            "key": "B_drawdown_stop",
            "field": "adaptive_drawdown_stop",
            "audit_value": 1.0,
            "expected_diagnostic_key": "blocked_dd_stop",
        },
        {
            "key": "C_aggregate_risk",
            "field": "adaptive_max_aggregate_risk",
            "audit_value": 1.0,
            "expected_diagnostic_key": "blocked_aggregate_risk",
        },
        {
            "key": "D_currency_risk",
            "field": "adaptive_max_single_currency_risk",
            "audit_value": 1.0,
            "expected_diagnostic_key": "blocked_currency_risk",
        },
        {
            "key": "E_monthly_loss",
            "field": "adaptive_monthly_loss_limit",
            "audit_value": 1.0,
            "expected_diagnostic_key": "blocked_monthly_loss",
        },
    ]


@pytest.mark.parametrize("spec", ARM_SPECS, ids=lambda spec: spec.key)
def test_each_arm_changes_exactly_one_settings_field(spec):
    control = normalize_control(BASE)
    arm = build_arm_settings(control, spec)

    assert changed_fields(control, arm) == [spec.field]
    assert getattr(arm, spec.field) == spec.audit_value


def test_build_arm_settings_rejects_noop_arm():
    control = replace(normalize_control(BASE), adaptive_max_open_positions=10)
    with pytest.raises(ValueError):
        build_arm_settings(control, ARM_SPECS[0])


def test_single_field_assertion_rejects_extra_change():
    control = normalize_control(BASE)
    arm = replace(control, adaptive_max_open_positions=10, adaptive_score_min=85.0)
    with pytest.raises(RuntimeError):
        audit.assert_single_field_difference(control, arm, "adaptive_max_open_positions")


# --- run_research reuse --------------------------------------------------------


def test_reuses_run_research_directly():
    assert audit.run_research is run_research


def test_run_experiment_calls_run_research_once_control_then_once_per_active_arm(monkeypatch):
    calls, probe = _canned(monkeypatch)

    _run(probe)

    assert len(calls) == 5
    assert changed_fields(CONTROL_REF, calls[0]) == []
    active_fields = [changed_fields(CONTROL_REF, call) for call in calls[1:]]
    assert active_fields == [
        ["adaptive_max_open_positions"],
        ["adaptive_drawdown_stop"],
        ["adaptive_max_aggregate_risk"],
        ["adaptive_max_single_currency_risk"],
    ]


def test_run_experiment_is_deterministic(monkeypatch):
    _calls, probe = _canned(monkeypatch)
    first = _run(probe)
    second = _run(probe)
    assert first == second


# --- monthly-loss zero-block skip ---------------------------------------------


def test_monthly_zero_block_skips_arm_and_run_call(monkeypatch):
    calls, probe = _canned(monkeypatch)

    payload = _run(probe)

    assert payload["control_block_counts"]["blocked_monthly_loss"] == 0
    assert all(changed_fields(CONTROL_REF, call) != ["adaptive_monthly_loss_limit"] for call in calls)
    monthly = payload["gates"]["E_monthly_loss"]
    assert monthly["classification"] == "NOT_APPLICABLE"
    assert monthly["executed"] is False
    assert monthly["active"] is False


def test_nonzero_monthly_block_runs_monthly_arm_and_anchor_mismatch_keeps_it_ambiguous(monkeypatch):
    blocks = dict(DEFAULT_CONTROL_BLOCKS, blocked_monthly_loss=3)
    calls, probe = _canned(monkeypatch, control_blocks=blocks)

    payload = _run(probe)

    assert any(changed_fields(CONTROL_REF, call) == ["adaptive_monthly_loss_limit"] for call in calls)
    monthly = payload["gates"]["E_monthly_loss"]
    assert monthly["executed"] is True
    assert payload["audit_valid"] is False
    assert monthly["classification"] == "AMBIGUOUS"


# --- anchor and authority ------------------------------------------------------


def test_anchor_comparison_matches_expected_baseline():
    control = {**_payload(), "gate_diagnostics": dict(ANCHOR_DIAGNOSTICS)}
    anchor = _anchor_comparison(control)
    assert anchor["matches"] is True
    assert all(field["matches"] for field in anchor["fields"].values())


def test_anchor_comparison_detects_diagnostic_drift():
    diagnostics = dict(ANCHOR_DIAGNOSTICS, blocked_dd_stop=54)
    control = {**_payload(), "gate_diagnostics": diagnostics}
    anchor = _anchor_comparison(control)
    assert anchor["matches"] is False
    assert anchor["fields"]["blocked_dd_stop"]["matches"] is False


def test_anchor_mismatch_forces_audit_invalid_and_blocks_protective_and_overrestrictive(monkeypatch):
    control = _payload(full_pf=0.9)
    arms = {"adaptive_max_open_positions": _payload(**OVER_KW), "adaptive_drawdown_stop": _payload(**HARM_KW)}
    _calls, probe = _canned(monkeypatch, control=control, arms=arms)

    payload = _run(probe)

    assert payload["audit_valid"] is False
    classifications = {gate["classification"] for gate in payload["gates"].values()}
    assert classifications <= {"AMBIGUOUS", "NOT_APPLICABLE"}
    assert "POTENTIALLY_OVERRESTRICTIVE" not in classifications
    assert "PROTECTIVE" not in classifications
    assert payload["next_experiment_candidate"] is None


def test_production_defaults_drift_forces_audit_invalid(monkeypatch):
    snapshots = iter([{"ADAPTIVE_SCORE_MIN": None}, {"ADAPTIVE_SCORE_MIN": "85"}])
    monkeypatch.setattr(audit, "_env_snapshot", lambda: next(snapshots))
    arms = {"adaptive_max_open_positions": _payload(**OVER_KW)}
    _calls, probe = _canned(monkeypatch, arms=arms)

    payload = _run(probe)

    assert payload["production_defaults_unchanged"] is False
    assert payload["audit_valid"] is False
    assert payload["gates"]["A_max_positions"]["classification"] == "AMBIGUOUS"
    assert payload["next_experiment_candidate"] is None


# --- classifications ----------------------------------------------------------


def test_all_four_classifications_are_deterministic():
    over = _pair(arm_kw=OVER_KW)
    harm = _pair(arm_kw=HARM_KW)
    neutral = _pair()

    cases = {
        "NOT_APPLICABLE": _classify_gate(0, True, over),
        "AMBIGUOUS": _classify_gate(5, False, over),
        "POTENTIALLY_OVERRESTRICTIVE": _classify_gate(5, True, over),
        "PROTECTIVE": _classify_gate(5, True, harm),
    }
    for expected, (classification, _reason) in cases.items():
        assert classification == expected
    assert _classify_gate(5, True, neutral)[0] == "AMBIGUOUS"
    assert _classify_gate(5, True, over) == _classify_gate(5, True, over)


def test_potentially_overrestrictive_when_all_nine_criteria_hold(monkeypatch):
    arms = {"adaptive_max_open_positions": _payload(**OVER_KW)}
    _calls, probe = _canned(monkeypatch, arms=arms)

    payload = _run(probe)

    assert payload["gates"]["A_max_positions"]["classification"] == "POTENTIALLY_OVERRESTRICTIVE"
    assert payload["next_experiment_candidate"]["key"] == "A_max_positions"


def test_protective_when_validation_and_test_pf_both_worse(monkeypatch):
    arms = {"adaptive_drawdown_stop": _payload(**HARM_KW)}
    _calls, probe = _canned(monkeypatch, arms=arms)

    payload = _run(probe)

    assert payload["gates"]["B_drawdown_stop"]["classification"] == "PROTECTIVE"
    assert payload["next_experiment_candidate"] is None


def test_protective_when_test_maxdd_worse_by_more_than_10pct_and_test_pf_worse():
    classification, _ = _classify_gate(
        5,
        True,
        _pair(arm_kw=dict(test_max_dd=0.06, test_pf=0.3, validation_pf=1.1, high_cost_pf=0.6)),
    )
    assert classification == "PROTECTIVE"


def test_protective_with_contradictory_high_cost_and_walk_forward_is_ambiguous():
    arm_kw = dict(HARM_KW, high_cost_pf=0.9, folds=[_fold(1.2) for _ in range(5)])
    classification, _ = _classify_gate(5, True, _pair(arm_kw=arm_kw))
    assert classification == "AMBIGUOUS"


def test_overrestrictive_requires_high_cost_pf_improvement():
    arm_kw = dict(OVER_KW, high_cost_pf=0.7)
    classification, _ = _classify_gate(5, True, _pair(arm_kw=arm_kw))
    assert classification == "AMBIGUOUS"


def test_validation_improvement_without_test_improvement_is_not_overrestrictive():
    arm_kw = dict(OVER_KW, test_pf=0.4276630536)
    assert _classify_gate(5, True, _pair(arm_kw=arm_kw))[0] == "AMBIGUOUS"


def test_test_improvement_without_validation_improvement_is_not_overrestrictive():
    arm_kw = dict(OVER_KW, validation_pf=0.5)
    assert _classify_gate(5, True, _pair(arm_kw=arm_kw))[0] == "AMBIGUOUS"


def test_test_maxdd_worsening_blocks_overrestrictive():
    arm_kw = dict(OVER_KW, test_max_dd=0.0551)
    assert _classify_gate(5, True, _pair(arm_kw=arm_kw))[0] == "AMBIGUOUS"


# --- 10% PF and DD boundaries ---------------------------------------------------


def test_full_pf_exactly_10pct_worse_passes_and_beyond_fails():
    passes = _pair(control_kw=dict(full_pf=1.0), arm_kw=dict(full_pf=0.9))["checks"]
    fails = _pair(control_kw=dict(full_pf=1.0), arm_kw=dict(full_pf=0.8999))["checks"]
    assert passes["full_pf_not_worse_by_more_than_10pct"] is True
    assert fails["full_pf_not_worse_by_more_than_10pct"] is False


def test_full_maxdd_exactly_10pct_worse_passes_and_beyond_fails():
    passes = _pair(control_kw=dict(full_max_dd=0.10), arm_kw=dict(full_max_dd=0.11))["checks"]
    fails = _pair(control_kw=dict(full_max_dd=0.10), arm_kw=dict(full_max_dd=0.1101))["checks"]
    assert passes["full_maxdd_not_worse_by_more_than_10pct"] is True
    assert fails["full_maxdd_not_worse_by_more_than_10pct"] is False


def test_test_maxdd_10pct_boundary():
    at_boundary = _pair(control_kw=dict(test_max_dd=0.05), arm_kw=dict(test_max_dd=0.055))["checks"]
    beyond = _pair(control_kw=dict(test_max_dd=0.05), arm_kw=dict(test_max_dd=0.0551))["checks"]
    assert at_boundary["test_maxdd_worse_by_more_than_10pct"] is False
    assert beyond["test_maxdd_worse_by_more_than_10pct"] is True


# --- walk-forward rule ---------------------------------------------------------


def test_fold_min_trades_constant_is_at_least_one():
    assert FOLD_MIN_TRADES >= 1
    assert FOLD_MIN_TRADES == 3


def test_walk_forward_zero_folds_is_not_contradiction_evidence():
    evidence = _walk_forward_evidence({"folds": []}, {"folds": []})
    assert evidence["folds_sufficient"] == 0
    assert evidence["non_contradicting"] is False
    assert evidence["ratio"] is None


def test_walk_forward_insufficient_folds_are_excluded_from_ratio():
    control = {"folds": [_fold(0.5, trades=5), _fold(0.5, trades=2), _fold(0.5, trades=5)]}
    arm = {"folds": [_fold(1.2, trades=5), _fold(0.0, trades=2), _fold(1.2, trades=5)]}
    evidence = _walk_forward_evidence(control, arm)

    assert evidence["folds_compared"] == 3
    assert evidence["folds_sufficient"] == 2
    assert evidence["folds_excluded"] == 1
    assert evidence["folds_pf_not_worse"] == 2
    assert evidence["non_contradicting"] is True


def test_walk_forward_all_insufficient_folds_cannot_support_overrestrictive():
    control = {"folds": [_fold(0.5, trades=1) for _ in range(5)]}
    arm = {"folds": [_fold(1.2, trades=1) for _ in range(5)]}
    evidence = _walk_forward_evidence(control, arm)
    assert evidence["folds_sufficient"] == 0
    assert evidence["non_contradicting"] is False


def test_walk_forward_majority_worse_contradicts():
    control = {"folds": [_fold(1.0, trades=5) for _ in range(4)]}
    arm = {"folds": [_fold(0.5, trades=5) for _ in range(3)] + [_fold(1.5, trades=5)]}
    evidence = _walk_forward_evidence(control, arm)
    assert evidence["folds_pf_not_worse"] == 1
    assert evidence["non_contradicting"] is False


# --- next experiment selection -------------------------------------------------


def _gate(key, classification, robustness, blocks):
    return {
        "key": key,
        "field": "field",
        "control_value": 1,
        "audit_value": 2,
        "control_block_count": blocks,
        "classification": classification,
        "robustness_score": robustness,
    }


def test_next_candidate_none_when_no_gate_is_overrestrictive():
    gates = {
        "A_max_positions": _gate("A_max_positions", "PROTECTIVE", 9.0, 144),
        "B_drawdown_stop": _gate("B_drawdown_stop", "AMBIGUOUS", 9.0, 55),
    }
    assert select_next_candidate(gates) is None


def test_next_candidate_never_selects_protective_or_ambiguous():
    gates = {
        "A_max_positions": _gate("A_max_positions", "PROTECTIVE", 9.0, 144),
        "B_drawdown_stop": _gate("B_drawdown_stop", "AMBIGUOUS", 8.0, 55),
        "C_aggregate_risk": _gate("C_aggregate_risk", "POTENTIALLY_OVERRESTRICTIVE", 0.1, 4),
    }
    chosen = select_next_candidate(gates)
    assert chosen["key"] == "C_aggregate_risk"


def test_next_candidate_is_at_most_one_and_ranked_by_robustness_then_blocks_then_order():
    gates = {
        "A_max_positions": _gate("A_max_positions", "POTENTIALLY_OVERRESTRICTIVE", 0.2, 144),
        "B_drawdown_stop": _gate("B_drawdown_stop", "POTENTIALLY_OVERRESTRICTIVE", 0.3, 55),
    }
    assert select_next_candidate(gates)["key"] == "B_drawdown_stop"

    tied = {
        "A_max_positions": _gate("A_max_positions", "POTENTIALLY_OVERRESTRICTIVE", 0.2, 144),
        "B_drawdown_stop": _gate("B_drawdown_stop", "POTENTIALLY_OVERRESTRICTIVE", 0.2, 55),
    }
    assert select_next_candidate(tied)["key"] == "A_max_positions"

    fully_tied = {
        "C_aggregate_risk": _gate("C_aggregate_risk", "POTENTIALLY_OVERRESTRICTIVE", 0.2, 4),
        "B_drawdown_stop": _gate("B_drawdown_stop", "POTENTIALLY_OVERRESTRICTIVE", 0.2, 4),
    }
    assert select_next_candidate(fully_tied)["key"] == "B_drawdown_stop"
    assert GATE_ORDER["B_drawdown_stop"] < GATE_ORDER["C_aggregate_risk"]


# --- probe and output contract -------------------------------------------------


def test_probe_returns_zero_block_counts_for_empty_candles():
    diagnostics = simulated_gate_diagnostics({}, BASE, 1_000_000.0, 0)
    assert {key: diagnostics[key] for key in BLOCK_KEYS} == {key: 0 for key in BLOCK_KEYS}


def test_probe_rejects_trade_count_divergence_from_run_research(monkeypatch):
    fake_simulated = {"trades": [], "diagnostics": {key: 0 for key in BLOCK_KEYS}}
    monkeypatch.setattr(audit, "_simulate", lambda *args, **kwargs: fake_simulated)
    with pytest.raises(RuntimeError):
        simulated_gate_diagnostics({"USDJPY": object()}, BASE, 1_000_000.0, 3)


def test_json_safe_maps_non_finite_to_null():
    assert _json_safe({"a": math.inf, "b": -math.inf, "c": 1.5}) == {"a": None, "b": None, "c": 1.5}


def test_write_report_is_deterministic_and_states_limitations(monkeypatch, tmp_path):
    arms = {"adaptive_max_open_positions": _payload(**OVER_KW)}
    _calls, probe = _canned(monkeypatch, arms=arms)
    payload = _run(probe)

    json_a, md_a = tmp_path / "a.json", tmp_path / "a.md"
    json_b, md_b = tmp_path / "b.json", tmp_path / "b.md"
    write_report(payload, json_a, md_a)
    write_report(payload, json_b, md_b)

    assert json_a.read_text(encoding="utf-8") == json_b.read_text(encoding="utf-8")
    assert md_a.read_text(encoding="utf-8") == md_b.read_text(encoding="utf-8")

    markdown = md_a.read_text(encoding="utf-8")
    assert "policy-level counterfactual limitation" in markdown
    assert "## Limitations" in markdown
    assert "No arm is a production recommendation" in markdown

    written = json.loads(json_a.read_text(encoding="utf-8"))
    assert written["no_arm_is_production_recommendation"] is True
    assert written["next_experiment_candidate"]["key"] == "A_max_positions"


def test_markdown_reports_null_candidate_when_none_qualifies(monkeypatch, tmp_path):
    _calls, probe = _canned(monkeypatch)
    payload = _run(probe)
    markdown_path = tmp_path / "none.md"
    write_report(payload, tmp_path / "none.json", markdown_path)
    assert "next_experiment_candidate: null" in markdown_path.read_text(encoding="utf-8")


# --- output-only artifact and import boundary ---------------------------------


def test_default_artifact_paths_are_output_only_under_data_research():
    assert audit.DEFAULT_JSON_PATH == "data/research/adaptive_v2_no_trade_counterfactual_audit.json"
    assert audit.DEFAULT_MARKDOWN_PATH == "data/research/adaptive_v2_no_trade_counterfactual_audit.md"


def test_forward_execution_modules_never_import_this_audit_module():
    for rel_path in FORWARD_EXECUTION_FILES:
        source = (REPO_ROOT / rel_path).read_text(encoding="utf-8")
        assert "research_no_trade_counterfactual_audit" not in source


def test_audit_module_has_no_forward_live_paper_or_scheduler_imports():
    source = (REPO_ROOT / "app" / "research_no_trade_counterfactual_audit.py").read_text(encoding="utf-8")
    for forbidden in (
        "from .paper",
        "from .engine",
        "from .main",
        "from .mt5_client",
        "from .scheduler",
        "import app.paper",
        "import app.engine",
        "import app.main",
        "import app.mt5_client",
    ):
        assert forbidden not in source


def test_main_writes_output_only_artifacts_under_given_paths(monkeypatch, tmp_path, capsys):
    _calls, probe = _canned(monkeypatch)
    monkeypatch.setattr(audit, "simulated_gate_diagnostics", probe)

    data_dir = tmp_path / "fx_research"
    data_dir.mkdir()
    (data_dir / "USDJPY.json").write_text(
        json.dumps(
            {
                "instrument": "USDJPY",
                "synced_at": "2026-08-16T00:00:00+00:00",
                "bars": [
                    {"time": "2020-01-01T00:00:00+00:00", "open": 100.0, "high": 100.5, "low": 99.5, "close": 100.2},
                    {"time": "2020-01-02T00:00:00+00:00", "open": 100.2, "high": 100.7, "low": 99.7, "close": 100.4},
                ],
            }
        ),
        encoding="utf-8",
    )
    json_path = tmp_path / "out.json"
    markdown_path = tmp_path / "out.md"

    exit_code = audit.main(
        [
            "run",
            "--data-dir",
            str(data_dir),
            "--json-path",
            str(json_path),
            "--markdown-path",
            str(markdown_path),
        ]
    )

    assert exit_code == 0
    assert json_path.exists()
    assert markdown_path.exists()
    assert "NO_TRADE_COUNTERFACTUAL_AUDIT_OK" in capsys.readouterr().out
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["experiment"] == "adaptive_v2_no_trade_counterfactual_audit_4"


def test_main_reports_no_data_without_writing_outputs(tmp_path):
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    exit_code = audit.main(
        [
            "run",
            "--data-dir",
            str(empty_dir),
            "--json-path",
            str(tmp_path / "x.json"),
            "--markdown-path",
            str(tmp_path / "x.md"),
        ]
    )
    assert exit_code == 1
    assert not (tmp_path / "x.json").exists()


# --- PowerShell runner contract (text only, no execution) ----------------------


def test_powershell_runner_is_utf8_with_bom_and_invokes_worktree_module():
    runner = REPO_ROOT / "scripts" / "Run-AdaptiveV2NoTradeCounterfactualAudit.ps1"
    raw = runner.read_bytes()
    assert raw[:3] == b"\xef\xbb\xbf"

    text = raw.decode("utf-8-sig")
    assert "[Parameter(Mandatory = $true)]" in text
    assert "[string]$DataDir" in text
    assert "$PSScriptRoot" in text
    assert "-m app.research_no_trade_counterfactual_audit run" in text
    assert "[switch]$ValidateOnly" in text


def test_powershell_runner_has_no_network_order_or_database_commands():
    runner = REPO_ROOT / "scripts" / "Run-AdaptiveV2NoTradeCounterfactualAudit.ps1"
    text = runner.read_text(encoding="utf-8-sig")
    for forbidden in (
        "Invoke-WebRequest",
        "Invoke-RestMethod",
        "Start-Process",
        "sqlite3",
        "app.main",
        "app.paper",
        "app.engine",
        "app.mt5_client",
        "send_order",
    ):
        assert forbidden not in text
