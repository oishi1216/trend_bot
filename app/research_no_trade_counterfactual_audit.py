from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Mapping

import pandas as pd

from .config import Settings
from .research_backtest import (
    DEFAULT_SLIPPAGE_PIPS,
    DEFAULT_SPREAD_PIPS,
    _simulate,
    run_research,
)

# Adaptive v2 Diagnostic #4 (GitHub Issue #14): research-only, policy-level
# counterfactual audit of the existing no-trade portfolio/risk gates. Every
# arm reruns app.research_backtest.run_research on the same normalized control
# with exactly one Settings field changed. This is NOT event-level PnL
# attribution: no shadow trade engine is built, and downstream portfolio state
# can differ after a gate is relaxed. No arm is a production recommendation.
# This module must never be imported by forward/paper execution code
# (app.engine, app.paper, app.main, app.mt5_client).

# Normalized control: production/default strategy settings for this audit.
SCORE_MIN = 75.0
STRENGTH_GAP_MIN = 0.3
MAX_OPEN_POSITIONS = 2
FAST_EMA_DAYS = 20
MID_EMA_DAYS = 50
SLOW_EMA_DAYS = 200

CONTROL_REPORTED_FIELDS = (
    "adaptive_score_min",
    "adaptive_strength_gap_min",
    "adaptive_max_open_positions",
    "adaptive_fast_ema_days",
    "adaptive_mid_ema_days",
    "adaptive_slow_ema_days",
    "adaptive_drawdown_stop",
    "adaptive_drawdown_reduce_1",
    "adaptive_drawdown_reduce_2",
    "adaptive_max_aggregate_risk",
    "adaptive_max_single_currency_risk",
    "adaptive_monthly_loss_limit",
)

BLOCK_KEYS = (
    "blocked_max_positions",
    "blocked_monthly_loss",
    "blocked_dd_stop",
    "blocked_aggregate_risk",
    "blocked_currency_risk",
)

# Same control anchor as Experiment #2 (adaptive_max_open_positions=2,
# adaptive_score_min=75, EMA 20/50/200), plus the expected control gate block
# counts. A mismatch means the repository/data state has drifted from the
# anchor; policy comparisons are then not authoritative.
ANCHOR: dict[str, float] = {
    "trades": 133,
    "cagr": -0.0030874892,
    "max_dd": 0.1006936790,
    "pf": 0.8137793529,
    "profit": -37421.8830,
    "test_trades": 13,
    "test_pf": 0.4276630536,
}
ANCHOR_DIAGNOSTICS: dict[str, int] = {
    "blocked_max_positions": 144,
    "blocked_dd_stop": 55,
    "blocked_aggregate_risk": 4,
    "blocked_currency_risk": 14,
    "blocked_monthly_loss": 0,
}

ANCHOR_FLOAT_REL_TOL = 1e-6
ANCHOR_FLOAT_ABS_TOL = 1e-6
ANCHOR_PROFIT_ABS_TOL = 0.01

# Relative tolerance for PF and max-DD materiality (10% convention shared with
# the earlier adaptive v2 experiments).
MATERIALITY_TOLERANCE = 0.10

# Tolerance for strict "improves"/"not worse" max-DD comparisons, to absorb
# floating point noise without treating a genuine reversal as a tie.
MAX_DD_EPSILON = 1e-12

# Walk-forward non-contradiction rule (deterministic, disclosed in output).
# A fold is sufficiently sampled only when both arms closed at least this many
# trades in it; insufficient folds are excluded from the ratio.
FOLD_MIN_TRADES = 3
WALK_FORWARD_NOT_WORSE_MIN_RATIO = 0.5

if FOLD_MIN_TRADES < 1:
    raise RuntimeError("FOLD_MIN_TRADES must be at least 1 so that empty folds never decide a gate")

# Environment keys watched for persisted overrides; only in-memory
# dataclasses.replace() is allowed by this audit.
_ENV_WATCH_KEYS = (
    "ADAPTIVE_MAX_OPEN_POSITIONS",
    "ADAPTIVE_SCORE_MIN",
    "ADAPTIVE_STRENGTH_GAP_MIN",
    "ADAPTIVE_FAST_EMA_DAYS",
    "ADAPTIVE_MID_EMA_DAYS",
    "ADAPTIVE_SLOW_EMA_DAYS",
    "ADAPTIVE_DRAWDOWN_STOP",
    "ADAPTIVE_MAX_AGGREGATE_RISK",
    "ADAPTIVE_MAX_SINGLE_CURRENCY_RISK",
    "ADAPTIVE_MONTHLY_LOSS_LIMIT",
    "STRATEGY_PROFILE",
)

DEFAULT_JSON_PATH = "data/research/adaptive_v2_no_trade_counterfactual_audit.json"
DEFAULT_MARKDOWN_PATH = "data/research/adaptive_v2_no_trade_counterfactual_audit.md"

AUDIT_SCOPE = "policy_level_counterfactual_not_event_level_pnl_attribution"

SELECTION_RULE = (
    "At most one gate. Only POTENTIALLY_OVERRESTRICTIVE gates are eligible; PROTECTIVE, AMBIGUOUS "
    "and NOT_APPLICABLE gates are never selected. Eligible gates are ordered by (1) robustness_score "
    "= min(Validation PF delta, Test PF delta) descending, (2) control block count descending "
    "(the more binding gate first), (3) fixed gate order A->E ascending. If no gate is eligible, "
    "next_experiment_candidate is null."
)

LIMITATIONS = (
    "Policy-level counterfactual only: each active arm reruns run_research with one Settings field "
    "relaxed. Individual blocked signals are not replayed or attributed a PnL.",
    "Downstream portfolio state can differ after relaxation (open positions, equity and drawdown "
    "path, risk budgets, and therefore later candidates). Arm trade sets are not supersets of the "
    "control, and trade-count deltas mix blocked-entry effects with path effects.",
    "Gate block counts are sequential first-block-wins diagnostics. Relaxing one gate lets candidates "
    "reach later gates, so other gates' counts can shift inside an arm.",
    "run_research does not expose simulator diagnostics. Block counts come from a read-only base-cost "
    "_simulate probe with the same settings, checked to reproduce run_research's base trade count.",
    "Walk-forward non-contradiction rule: a fold is sufficiently sampled when both arms have at least "
    f"FOLD_MIN_TRADES={FOLD_MIN_TRADES} closed trades in it. The clause passes only when at least one "
    "sufficiently sampled fold exists and at least "
    f"{WALK_FORWARD_NOT_WORSE_MIN_RATIO:.0%} of them have arm PF >= control PF.",
    "No arm is a production recommendation. Production Settings are not changed, and this module is "
    "not imported by forward execution code.",
)

POLICY_LIMITATION = (
    "Policy-level counterfactual: the arm reruns the full Research simulator with only this gate "
    "relaxed. Relaxing it changes downstream open positions, equity path, drawdown state and risk "
    "budgets, so the arm's trade set is not a superset of control and blocked-signal PnL is not "
    "attributed. Block counts are sequential first-block-wins diagnostics."
)


@dataclass(frozen=True)
class ArmSpec:
    key: str
    field: str
    audit_value: float
    expected_diagnostic_key: str


# Fixed audit arms, each relaxing exactly one gate threshold. Arm E (monthly
# loss) runs only when the control block count is non-zero; on the anchor it
# is zero, so it is NOT_APPLICABLE and never executed.
ARM_SPECS: tuple[ArmSpec, ...] = (
    ArmSpec("A_max_positions", "adaptive_max_open_positions", 10, "blocked_max_positions"),
    ArmSpec("B_drawdown_stop", "adaptive_drawdown_stop", 1.0, "blocked_dd_stop"),
    ArmSpec("C_aggregate_risk", "adaptive_max_aggregate_risk", 1.0, "blocked_aggregate_risk"),
    ArmSpec("D_currency_risk", "adaptive_max_single_currency_risk", 1.0, "blocked_currency_risk"),
    ArmSpec("E_monthly_loss", "adaptive_monthly_loss_limit", 1.0, "blocked_monthly_loss"),
)

GATE_ORDER: dict[str, int] = {spec.key: index for index, spec in enumerate(ARM_SPECS)}


def arm_specifications() -> list[dict[str, Any]]:
    return [dataclasses.asdict(spec) for spec in ARM_SPECS]


def normalize_control(base: Settings) -> Settings:
    """Reset inherited experimental values to the audit's production defaults.

    Everything else is inherited unchanged from ``base``. The result is built
    in memory with dataclasses.replace(); no environment variable is written.
    """
    return replace(
        base,
        adaptive_score_min=SCORE_MIN,
        adaptive_strength_gap_min=STRENGTH_GAP_MIN,
        adaptive_max_open_positions=MAX_OPEN_POSITIONS,
        adaptive_fast_ema_days=FAST_EMA_DAYS,
        adaptive_mid_ema_days=MID_EMA_DAYS,
        adaptive_slow_ema_days=SLOW_EMA_DAYS,
    )


def changed_fields(first: Settings, second: Settings) -> list[str]:
    return [
        f.name
        for f in dataclasses.fields(first)
        if getattr(first, f.name) != getattr(second, f.name)
    ]


def assert_single_field_difference(control: Settings, arm: Settings, field: str) -> None:
    differing = changed_fields(control, arm)
    if differing != [field]:
        raise RuntimeError(
            f"audit arm must differ from control in exactly {field!r}; differs in {differing}"
        )


def build_arm_settings(control: Settings, spec: ArmSpec) -> Settings:
    if getattr(control, spec.field) == spec.audit_value:
        raise ValueError(
            f"{spec.key}: control {spec.field} already equals audit value {spec.audit_value}"
        )
    arm = replace(control, **{spec.field: spec.audit_value})
    assert_single_field_difference(control, arm, spec.field)
    return arm


GateDiagnosticsProbe = Callable[[Mapping[str, pd.DataFrame], Settings, float, int], dict[str, Any]]


def simulated_gate_diagnostics(
    candles_by_instrument: Mapping[str, pd.DataFrame],
    settings: Settings,
    initial_equity: float,
    expected_trades: int,
) -> dict[str, Any]:
    """Read-only gate-diagnostics probe.

    run_research does not expose the simulator's gate counters, so this runs
    the same base-cost simulator once more and returns only its diagnostics.
    The base-cost trade count must reproduce run_research's base trade count,
    otherwise the counters would not describe the audited run.
    """
    if not candles_by_instrument:
        return {key: 0 for key in BLOCK_KEYS}
    simulated = _simulate(
        candles_by_instrument,
        settings,
        initial_equity=initial_equity,
        spread_pips=DEFAULT_SPREAD_PIPS,
        slippage_pips=DEFAULT_SLIPPAGE_PIPS,
    )
    if len(simulated["trades"]) != expected_trades:
        raise RuntimeError(
            "gate-diagnostics probe diverged from run_research base-cost trade count: "
            f"probe={len(simulated['trades'])} run_research={expected_trades}"
        )
    return dict(simulated["diagnostics"])


def _run_arm(
    candles_by_instrument: Mapping[str, pd.DataFrame],
    settings: Settings,
    initial_equity: float,
    probe: GateDiagnosticsProbe,
) -> dict[str, Any]:
    result = run_research(candles_by_instrument, settings, initial_equity=initial_equity)
    expected_trades = int(result.get("metrics", {}).get("trades", 0))
    diagnostics = probe(candles_by_instrument, settings, initial_equity, expected_trades)
    return {**result, "gate_diagnostics": dict(diagnostics)}


def _block_counts(diagnostics: Mapping[str, Any]) -> dict[str, int]:
    return {key: int(diagnostics.get(key, 0)) for key in BLOCK_KEYS}


def _pf_rank(pf: float | None) -> float:
    # _metrics() returns None for "infinite" PF (winners, no losers) rather
    # than float("inf") to stay JSON-serializable; recover the ordering here.
    return math.inf if pf is None else float(pf)


def _pf_delta(arm_pf: float | None, control_pf: float | None) -> float:
    arm = _pf_rank(arm_pf)
    control = _pf_rank(control_pf)
    if math.isinf(arm) and math.isinf(control):
        return 0.0
    return arm - control


def _field_matches(key: str, expected: float, actual: Any) -> bool:
    if key in {"trades", "test_trades"} or key.startswith("blocked_"):
        return actual == expected
    if actual is None:
        return False
    if key == "profit":
        return math.isclose(actual, expected, rel_tol=ANCHOR_FLOAT_REL_TOL, abs_tol=ANCHOR_PROFIT_ABS_TOL)
    return math.isclose(actual, expected, rel_tol=ANCHOR_FLOAT_REL_TOL, abs_tol=ANCHOR_FLOAT_ABS_TOL)


def _anchor_comparison(control: Mapping[str, Any]) -> dict[str, Any]:
    full = control["metrics"]
    test = control["test"]["metrics"]
    diagnostics = control["gate_diagnostics"]
    observed: dict[str, Any] = {
        "trades": full["trades"],
        "cagr": full["cagr"],
        "max_dd": full["max_dd"],
        "pf": full["pf"],
        "profit": full["profit"],
        "test_trades": test["trades"],
        "test_pf": test["pf"],
    }
    fields: dict[str, dict[str, Any]] = {}
    all_match = True
    for key, expected in ANCHOR.items():
        matches = _field_matches(key, expected, observed[key])
        fields[key] = {"expected": expected, "actual": observed[key], "matches": matches}
        all_match = all_match and matches
    for key, expected in ANCHOR_DIAGNOSTICS.items():
        actual = diagnostics.get(key)
        matches = _field_matches(key, expected, actual)
        fields[key] = {"expected": expected, "actual": actual, "matches": matches}
        all_match = all_match and matches
    return {"matches": all_match, "fields": fields}


def _walk_forward_evidence(control_wf: Mapping[str, Any], arm_wf: Mapping[str, Any]) -> dict[str, Any]:
    control_folds = control_wf.get("folds") or []
    arm_folds = arm_wf.get("folds") or []
    total = min(len(control_folds), len(arm_folds))
    sufficient = [
        (control_fold["metrics"], arm_fold["metrics"])
        for control_fold, arm_fold in zip(control_folds[:total], arm_folds[:total])
        if min(control_fold["metrics"]["trades"], arm_fold["metrics"]["trades"]) >= FOLD_MIN_TRADES
    ]
    not_worse = sum(
        1 for control_metrics, arm_metrics in sufficient
        if _pf_rank(arm_metrics["pf"]) >= _pf_rank(control_metrics["pf"])
    )
    ratio = (not_worse / len(sufficient)) if sufficient else None
    non_contradicting = bool(sufficient) and ratio >= WALK_FORWARD_NOT_WORSE_MIN_RATIO
    return {
        "folds_compared": total,
        "folds_sufficient": len(sufficient),
        "folds_excluded": total - len(sufficient),
        "folds_pf_not_worse": not_worse,
        "ratio": ratio,
        "min_ratio": WALK_FORWARD_NOT_WORSE_MIN_RATIO,
        "min_fold_trades": FOLD_MIN_TRADES,
        "non_contradicting": bool(non_contradicting),
    }


def _gate_evidence(control: Mapping[str, Any], arm: Mapping[str, Any]) -> dict[str, Any]:
    c_full = control["metrics"]
    a_full = arm["metrics"]
    c_val = control["validation"]["metrics"]
    a_val = arm["validation"]["metrics"]
    c_test = control["test"]["metrics"]
    a_test = arm["test"]["metrics"]
    c_high = control["sensitivity"]["cost_x2"]["metrics"]
    a_high = arm["sensitivity"]["cost_x2"]["metrics"]

    validation_delta = _pf_delta(a_val["pf"], c_val["pf"])
    test_delta = _pf_delta(a_test["pf"], c_test["pf"])
    high_cost_delta = _pf_delta(a_high["pf"], c_high["pf"])
    walk_forward = _walk_forward_evidence(control["walk_forward"], arm["walk_forward"])

    checks = {
        "validation_pf_improves": validation_delta > 0,
        "validation_pf_worse": validation_delta < 0,
        "test_pf_improves": test_delta > 0,
        "test_pf_worse": test_delta < 0,
        "test_maxdd_not_worse": a_test["max_dd"] <= c_test["max_dd"] + MAX_DD_EPSILON,
        "test_maxdd_worse_by_more_than_10pct": (
            a_test["max_dd"] > c_test["max_dd"] * (1 + MATERIALITY_TOLERANCE) + MAX_DD_EPSILON
        ),
        "full_pf_not_worse_by_more_than_10pct": (
            _pf_rank(a_full["pf"]) >= _pf_rank(c_full["pf"]) * (1 - MATERIALITY_TOLERANCE)
        ),
        "full_maxdd_not_worse_by_more_than_10pct": (
            a_full["max_dd"] <= c_full["max_dd"] * (1 + MATERIALITY_TOLERANCE) + MAX_DD_EPSILON
        ),
        "high_cost_pf_improves": high_cost_delta > 0,
        "walk_forward_non_contradicting": walk_forward["non_contradicting"],
    }
    return {
        "checks": {key: bool(value) for key, value in checks.items()},
        "walk_forward": walk_forward,
        "robustness_score": min(validation_delta, test_delta),
    }


def _classify_gate(
    block_count: int,
    audit_valid: bool,
    evidence: Mapping[str, Any],
) -> tuple[str, str]:
    """Classify one active gate. Priority: block count, validity, overrestrictive, protective."""
    if block_count <= 0:
        return (
            "NOT_APPLICABLE",
            "control block count is zero; the gate never bound on the control, so no relaxation arm was executed",
        )
    if not audit_valid:
        return (
            "AMBIGUOUS",
            "audit_valid=false (control anchor or production-default drift); policy comparisons are not "
            "authoritative, so PROTECTIVE and POTENTIALLY_OVERRESTRICTIVE claims are withheld",
        )

    checks = evidence["checks"]
    overrestrictive_checks = {
        "block_count_positive": block_count > 0,
        "validation_pf_improves": checks["validation_pf_improves"],
        "test_pf_improves": checks["test_pf_improves"],
        "test_maxdd_not_worse": checks["test_maxdd_not_worse"],
        "full_pf_not_worse_by_more_than_10pct": checks["full_pf_not_worse_by_more_than_10pct"],
        "full_maxdd_not_worse_by_more_than_10pct": checks["full_maxdd_not_worse_by_more_than_10pct"],
        "high_cost_pf_improves": checks["high_cost_pf_improves"],
        "walk_forward_non_contradicting": checks["walk_forward_non_contradicting"],
        "production_defaults_unchanged": audit_valid,
    }
    if all(overrestrictive_checks.values()):
        return (
            "POTENTIALLY_OVERRESTRICTIVE",
            "relaxing the gate improved Validation and Test PF without worsening Test max DD, without a "
            "more-than-10% full-sample PF or max-DD deterioration, with high-cost PF still improving, "
            "with walk-forward not contradicting, and with production defaults unchanged",
        )

    oos_pf_both_worse = checks["validation_pf_worse"] and checks["test_pf_worse"]
    test_dd_harm = checks["test_maxdd_worse_by_more_than_10pct"] and checks["test_pf_worse"]
    if oos_pf_both_worse or test_dd_harm:
        contradicted = checks["high_cost_pf_improves"] and checks["walk_forward_non_contradicting"]
        if not contradicted:
            return (
                "PROTECTIVE",
                "relaxing the gate caused clear out-of-sample harm (Validation and Test PF both worse, or "
                "Test max DD worse by more than 10% with Test PF worse) with no strong contradictory "
                "high-cost and walk-forward improvement",
            )
        return (
            "AMBIGUOUS",
            "out-of-sample harm was observed, but high-cost PF and walk-forward both improved, so the "
            "result is ambiguous rather than protective",
        )
    return (
        "AMBIGUOUS",
        "active gate did not meet the overrestrictive or protective criteria",
    )


def _full_view(metrics: Mapping[str, Any]) -> dict[str, Any]:
    return {key: metrics[key] for key in ("trades", "pf", "avg_r", "cagr", "max_dd", "profit", "win_rate")}


def _split_view(metrics: Mapping[str, Any]) -> dict[str, Any]:
    return {key: metrics[key] for key in ("trades", "pf", "avg_r", "max_dd", "profit")}


def _inactive_gate(spec: ArmSpec, control_value: Any, block_count: int) -> dict[str, Any]:
    return {
        "key": spec.key,
        "field": spec.field,
        "control_value": control_value,
        "audit_value": spec.audit_value,
        "expected_control_diagnostic": spec.expected_diagnostic_key,
        "control_block_count": block_count,
        "active": False,
        "executed": False,
        "classification": "NOT_APPLICABLE",
        "classification_reason": (
            "control block count is zero; the gate never bound on the control, so no relaxation arm "
            "was executed"
        ),
        "robustness_score": None,
    }


def _active_gate(
    spec: ArmSpec,
    control_value: Any,
    block_count: int,
    control: Mapping[str, Any],
    arm: Mapping[str, Any],
    audit_valid: bool,
) -> dict[str, Any]:
    evidence = _gate_evidence(control, arm)
    classification, reason = _classify_gate(block_count, audit_valid, evidence)
    control_trades = control["metrics"]["trades"]
    arm_trades = arm["metrics"]["trades"]
    return {
        "key": spec.key,
        "field": spec.field,
        "control_value": control_value,
        "audit_value": spec.audit_value,
        "expected_control_diagnostic": spec.expected_diagnostic_key,
        "control_block_count": block_count,
        "active": True,
        "executed": True,
        "classification": classification,
        "classification_reason": reason,
        "robustness_score": evidence["robustness_score"],
        "checks": evidence["checks"],
        "trade_count": {
            "control": control_trades,
            "audit": arm_trades,
            "delta": arm_trades - control_trades,
            "ratio": (arm_trades / control_trades) if control_trades else None,
        },
        "metrics": {
            "full": {"control": _full_view(control["metrics"]), "audit": _full_view(arm["metrics"])},
            "train": {
                "control": _split_view(control["train"]["metrics"]),
                "audit": _split_view(arm["train"]["metrics"]),
            },
            "validation": {
                "control": _split_view(control["validation"]["metrics"]),
                "audit": _split_view(arm["validation"]["metrics"]),
            },
            "test": {
                "control": _split_view(control["test"]["metrics"]),
                "audit": _split_view(arm["test"]["metrics"]),
            },
            "base": {
                "control": _full_view(control["sensitivity"]["base"]["metrics"]),
                "audit": _full_view(arm["sensitivity"]["base"]["metrics"]),
            },
            "high_cost": {
                "control": _full_view(control["sensitivity"]["cost_x2"]["metrics"]),
                "audit": _full_view(arm["sensitivity"]["cost_x2"]["metrics"]),
            },
        },
        "walk_forward": {
            "control_fold_positive": control["walk_forward"].get("fold_positive"),
            "audit_fold_positive": arm["walk_forward"].get("fold_positive"),
            **evidence["walk_forward"],
        },
        "arm_gate_diagnostics": dict(arm["gate_diagnostics"]),
        "policy_limitation": POLICY_LIMITATION,
    }


def select_next_candidate(gates: Mapping[str, Mapping[str, Any]]) -> dict[str, Any] | None:
    eligible = [gate for gate in gates.values() if gate["classification"] == "POTENTIALLY_OVERRESTRICTIVE"]
    if not eligible:
        return None
    ordered = sorted(
        eligible,
        key=lambda gate: (
            -gate["robustness_score"],
            -gate["control_block_count"],
            GATE_ORDER[gate["key"]],
        ),
    )
    chosen = ordered[0]
    return {
        "key": chosen["key"],
        "field": chosen["field"],
        "control_value": chosen["control_value"],
        "audit_value": chosen["audit_value"],
        "robustness_score": chosen["robustness_score"],
        "control_block_count": chosen["control_block_count"],
    }


def _env_snapshot() -> dict[str, str | None]:
    return {key: os.environ.get(key) for key in _ENV_WATCH_KEYS}


def run_experiment(
    candles_by_instrument: Mapping[str, pd.DataFrame],
    base_settings: Settings,
    initial_equity: float = 1_000_000.0,
    probe: GateDiagnosticsProbe | None = None,
) -> dict[str, Any]:
    probe_fn = simulated_gate_diagnostics if probe is None else probe
    env_before = _env_snapshot()

    control_settings = normalize_control(base_settings)
    control = _run_arm(candles_by_instrument, control_settings, initial_equity, probe_fn)
    block_counts = _block_counts(control["gate_diagnostics"])

    # Only arms whose control block count is non-zero are executed. The gate
    # set is fixed and each arm changes exactly one field; there is no sweep.
    arm_runs: dict[str, dict[str, Any]] = {}
    for spec in ARM_SPECS:
        if block_counts[spec.expected_diagnostic_key] <= 0:
            continue
        arm_settings = build_arm_settings(control_settings, spec)
        arm_runs[spec.key] = _run_arm(candles_by_instrument, arm_settings, initial_equity, probe_fn)

    env_unchanged = _env_snapshot() == env_before
    anchor = _anchor_comparison(control)
    audit_valid = bool(anchor["matches"]) and env_unchanged

    gates: dict[str, dict[str, Any]] = {}
    for spec in ARM_SPECS:
        control_value = getattr(control_settings, spec.field)
        block_count = block_counts[spec.expected_diagnostic_key]
        arm = arm_runs.get(spec.key)
        if arm is None:
            gates[spec.key] = _inactive_gate(spec, control_value, block_count)
        else:
            gates[spec.key] = _active_gate(spec, control_value, block_count, control, arm, audit_valid)

    return {
        "experiment": "adaptive_v2_no_trade_counterfactual_audit_4",
        "audit_scope": AUDIT_SCOPE,
        "audit_valid": audit_valid,
        "production_defaults_unchanged": env_unchanged,
        "score_min": SCORE_MIN,
        "control_settings": {
            field: getattr(control_settings, field) for field in CONTROL_REPORTED_FIELDS
        },
        "control": control,
        "control_block_counts": block_counts,
        "anchor": anchor,
        "gates": gates,
        "next_experiment_candidate": select_next_candidate(gates),
        "next_experiment_selection_rule": SELECTION_RULE,
        "walk_forward_policy": {
            "min_fold_trades": FOLD_MIN_TRADES,
            "min_ratio": WALK_FORWARD_NOT_WORSE_MIN_RATIO,
            "rule": "sufficient folds only; at least one required; PF not-worse ratio >= min_ratio",
        },
        "limitations": list(LIMITATIONS),
        "no_arm_is_production_recommendation": True,
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def dump_json(payload: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(payload), ensure_ascii=False, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )


def _fmt(key: str, value: Any) -> str:
    if value is None:
        return "inf" if key == "pf" else "n/a"
    if isinstance(value, float):
        return f"{value:.10f}"
    return str(value)


def _metric_text(view: Mapping[str, Any]) -> str:
    return ", ".join(f"{key}={_fmt(key, value)}" for key, value in view.items())


def _fmt_num(value: Any) -> str:
    return "n/a" if value is None else f"{value:.10f}"


def write_report(payload: dict[str, Any], json_path: str | Path, markdown_path: str | Path) -> None:
    dump_json(payload, json_path)

    control = payload["control"]
    gates = payload["gates"]
    candidate = payload["next_experiment_candidate"]

    lines = [
        "# Adaptive v2 Diagnostic #4 — no-trade gate counterfactual audit (research-only, policy-level)",
        "",
        "Policy-level counterfactual audit, not event-level PnL attribution. No shadow trade engine was "
        "built; every arm reruns app.research_backtest.run_research with exactly one Settings field "
        "changed. Downstream portfolio state can differ after relaxation. No arm is a production "
        "recommendation, and production settings are unchanged.",
        "",
        f"audit_valid: **{payload['audit_valid']}**",
        f"production_defaults_unchanged: {payload['production_defaults_unchanged']}",
        "",
        "## Gate classifications",
        "",
        "| Gate | Field | Control value | Audit value | Control blocks | Classification |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for key, gate in gates.items():
        lines.append(
            f"| {key} | {gate['field']} | {gate['control_value']} | {gate['audit_value']} | "
            f"{gate['control_block_count']} | {gate['classification']} |"
        )

    lines += ["", "## Next Experiment #4 candidate", ""]
    if candidate is None:
        lines.append("- next_experiment_candidate: null — no gate-relaxation experiment is recommended from this audit.")
    else:
        lines.append(
            f"- next_experiment_candidate: {candidate['key']} ({candidate['field']}: "
            f"{candidate['control_value']} -> {candidate['audit_value']}, "
            f"control blocks={candidate['control_block_count']}, "
            f"robustness_score={_fmt_num(candidate['robustness_score'])})"
        )
    lines.append(f"- selection rule: {payload['next_experiment_selection_rule']}")

    lines += ["", "## Anchor check (control)"]
    for key, field in payload["anchor"]["fields"].items():
        lines.append(f"- {key}: expected={field['expected']} actual={field['actual']} matches={field['matches']}")

    lines += ["", "## Control metrics"]
    lines.append(f"- full: {_metric_text(_full_view(control['metrics']))}")
    for split in ("train", "validation", "test"):
        lines.append(f"- {split}: {_metric_text(_split_view(control[split]['metrics']))}")
    lines.append(f"- base-cost: {_metric_text(_full_view(control['sensitivity']['base']['metrics']))}")
    lines.append(f"- high-cost: {_metric_text(_full_view(control['sensitivity']['cost_x2']['metrics']))}")
    wf = control["walk_forward"]
    lines.append(f"- walk-forward fold_positive/fold_total={wf.get('fold_positive')}/{wf.get('fold_total')}")

    lines += ["", "## Control gate block counts"]
    for key, value in payload["control_block_counts"].items():
        lines.append(f"- {key}: {value}")

    for key, gate in gates.items():
        lines += ["", f"## {key}"]
        if not gate["active"]:
            lines.append(f"- classification: **{gate['classification']}** — {gate['classification_reason']}")
            lines.append(
                f"- control block count ({gate['expected_control_diagnostic']}): {gate['control_block_count']}; "
                "no relaxation arm executed"
            )
            continue
        lines.append(
            f"- changed field: {gate['field']} control={gate['control_value']} audit={gate['audit_value']}"
        )
        lines.append(f"- control block count ({gate['expected_control_diagnostic']}): {gate['control_block_count']}")
        lines.append(f"- classification: **{gate['classification']}** — {gate['classification_reason']}")
        lines.append(f"- robustness_score: {_fmt_num(gate['robustness_score'])}")
        metrics = gate["metrics"]
        lines.append(f"- full control: {_metric_text(metrics['full']['control'])}")
        lines.append(f"- full audit: {_metric_text(metrics['full']['audit'])}")
        for split in ("train", "validation", "test"):
            lines.append(f"- {split} control: {_metric_text(metrics[split]['control'])}")
            lines.append(f"- {split} audit: {_metric_text(metrics[split]['audit'])}")
        lines.append(f"- base-cost control: {_metric_text(metrics['base']['control'])}")
        lines.append(f"- base-cost audit: {_metric_text(metrics['base']['audit'])}")
        lines.append(f"- high-cost control: {_metric_text(metrics['high_cost']['control'])}")
        lines.append(f"- high-cost audit: {_metric_text(metrics['high_cost']['audit'])}")
        wf_gate = gate["walk_forward"]
        lines.append(
            f"- walk-forward: folds_compared={wf_gate['folds_compared']} "
            f"folds_sufficient={wf_gate['folds_sufficient']} folds_excluded={wf_gate['folds_excluded']} "
            f"folds_pf_not_worse={wf_gate['folds_pf_not_worse']} ratio={_fmt_num(wf_gate['ratio'])} "
            f"min_fold_trades={wf_gate['min_fold_trades']} non_contradicting={wf_gate['non_contradicting']} "
            f"fold_positive control/audit={wf_gate['control_fold_positive']}/{wf_gate['audit_fold_positive']}"
        )
        trades = gate["trade_count"]
        lines.append(
            f"- trade count: control={trades['control']} audit={trades['audit']} "
            f"delta={trades['delta']} ratio={_fmt_num(trades['ratio'])}"
        )
        diagnostics = gate["arm_gate_diagnostics"]
        lines.append(
            "- resulting diagnostic gate counts: "
            + ", ".join(f"{name}={diagnostics[name]}" for name in sorted(diagnostics))
        )
        lines.append(f"- policy-level counterfactual limitation: {gate['policy_limitation']}")

    lines += ["", "## Limitations"]
    lines += [f"- {text}" for text in payload["limitations"]]

    Path(markdown_path).parent.mkdir(parents=True, exist_ok=True)
    Path(markdown_path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    from .fx_research_data import DEFAULT_RESEARCH_DATA_DIR, load_history

    parser = argparse.ArgumentParser(
        description="Research-only Adaptive v2 Diagnostic #4: no-trade gate counterfactual audit"
    )
    parser.add_argument("command", choices=["run"])
    parser.add_argument("--data-dir", default=DEFAULT_RESEARCH_DATA_DIR)
    parser.add_argument("--json-path", default=DEFAULT_JSON_PATH)
    parser.add_argument("--markdown-path", default=DEFAULT_MARKDOWN_PATH)
    args = parser.parse_args(argv)

    settings = Settings.from_env()
    history = load_history(args.data_dir)
    if not history:
        print(f"NO_DATA: no synced history found under {args.data_dir}")
        return 1

    payload = run_experiment(history, settings, initial_equity=settings.paper_initial_balance)
    write_report(payload, args.json_path, args.markdown_path)
    candidate = payload["next_experiment_candidate"]
    candidate_key = "none" if candidate is None else candidate["key"]
    print(
        "NO_TRADE_COUNTERFACTUAL_AUDIT_OK "
        f"audit_valid={payload['audit_valid']} "
        f"control_trades={payload['control']['metrics']['trades']} "
        f"candidate={candidate_key}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
