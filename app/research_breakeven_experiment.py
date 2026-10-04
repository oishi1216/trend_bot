from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from .config import Settings
from .research_backtest import (
    DEFAULT_SLIPPAGE_PIPS,
    DEFAULT_SPREAD_PIPS,
    _simulate,
    run_research,
)

CONTROL_BREAKEVEN_TRIGGER_R = 1.0
TREATMENT_BREAKEVEN_TRIGGER_R = 0.5
STRATEGY_PROFILE = "adaptive_dual_regime_v1"
FAST_EMA_DAYS = 20
MID_EMA_DAYS = 50
SLOW_EMA_DAYS = 200
SCORE_MIN = 75.0
MAX_OPEN_POSITIONS = 2
STRENGTH_GAP_MIN = 0.3

DEFAULT_JSON_PATH = "data/research/adaptive_v2_breakeven_experiment.json"
DEFAULT_MARKDOWN_PATH = "data/research/adaptive_v2_breakeven_experiment.md"

ANCHOR: dict[str, float] = {
    "trades": 133,
    "cagr": -0.0030874892,
    "max_dd": 0.1006936790,
    "pf": 0.8137793529,
    "profit": -37421.8830,
    "test_trades": 13,
    "test_pf": 0.4276630536,
}
ANCHOR_FLOAT_REL_TOL = 1e-6
ANCHOR_FLOAT_ABS_TOL = 1e-6
ANCHOR_PROFIT_ABS_TOL = 0.01
FLOAT_EPSILON = 1e-9
RETENTION_MIN_RATIO = 0.70
FULL_MAX_DD_MAX_WORSEN_RATIO = 0.10
FOLD_MIN_TRADES = 3
MIN_SUFFICIENT_FOLDS = 2
WALK_FORWARD_PF_NOT_WORSE_MIN_RATIO = 0.50

_ENV_WATCH_KEYS = (
    "ADAPTIVE_BREAKEVEN_TRIGGER_R",
    "ADAPTIVE_FAST_EMA_DAYS",
    "ADAPTIVE_MID_EMA_DAYS",
    "ADAPTIVE_SLOW_EMA_DAYS",
    "ADAPTIVE_SCORE_MIN",
    "ADAPTIVE_MAX_OPEN_POSITIONS",
    "ADAPTIVE_STRENGTH_GAP_MIN",
    "STRATEGY_PROFILE",
)

FORWARD_EXECUTION_MODULES = (
    "app.paper",
    "app.engine",
    "app.main",
    "app.mt5_client",
)

SUPPORTED_GATE_NAMES = (
    "anchor_match",
    "full_pf_improves",
    "full_avg_r_improves",
    "validation_pf_improves",
    "test_pf_improves",
    "test_avg_r_not_worse",
    "test_maxdd_not_worse",
    "full_maxdd_not_more_than_10pct_worse",
    "validation_retention",
    "test_retention",
    "high_cost_pf_direction_preserved",
    "walk_forward_non_contradicting",
    "production_defaults_unchanged",
    "single_variable_isolation",
)


def build_experiment_settings(base: Settings) -> tuple[Settings, Settings]:
    pinned = {
        "strategy_profile": STRATEGY_PROFILE,
        "adaptive_fast_ema_days": FAST_EMA_DAYS,
        "adaptive_mid_ema_days": MID_EMA_DAYS,
        "adaptive_slow_ema_days": SLOW_EMA_DAYS,
        "adaptive_score_min": SCORE_MIN,
        "adaptive_max_open_positions": MAX_OPEN_POSITIONS,
        "adaptive_strength_gap_min": STRENGTH_GAP_MIN,
    }
    control = replace(
        base,
        adaptive_breakeven_trigger_r=CONTROL_BREAKEVEN_TRIGGER_R,
        **pinned,
    )
    treatment = replace(
        base,
        adaptive_breakeven_trigger_r=TREATMENT_BREAKEVEN_TRIGGER_R,
        **pinned,
    )
    return control, treatment


def changed_fields(left: Settings, right: Settings) -> list[str]:
    return [
        field.name
        for field in dataclasses.fields(left)
        if getattr(left, field.name) != getattr(right, field.name)
    ]


def settings_diff(left: Settings, right: Settings) -> list[dict[str, Any]]:
    return [
        {
            "field": name,
            "control": getattr(left, name),
            "treatment": getattr(right, name),
        }
        for name in changed_fields(left, right)
    ]


def _production_snapshot() -> dict[str, Any]:
    return {
        "env": {key: os.environ.get(key) for key in _ENV_WATCH_KEYS},
        "settings": dataclasses.asdict(Settings.from_env()),
    }


def _anchor_comparison(control_payload: Mapping[str, Any]) -> dict[str, Any]:
    full = control_payload["metrics"]
    test = control_payload["test"]["metrics"]
    observed = {
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
        actual = observed[key]
        if key in {"trades", "test_trades"}:
            matches = actual == expected
        elif key == "profit":
            matches = actual is not None and math.isclose(
                actual,
                expected,
                rel_tol=ANCHOR_FLOAT_REL_TOL,
                abs_tol=ANCHOR_PROFIT_ABS_TOL,
            )
        else:
            matches = actual is not None and math.isclose(
                actual,
                expected,
                rel_tol=ANCHOR_FLOAT_REL_TOL,
                abs_tol=ANCHOR_FLOAT_ABS_TOL,
            )
        fields[key] = {"expected": expected, "actual": actual, "matches": matches}
        all_match = all_match and matches
    return {"matches": all_match, "fields": fields}


def _metric_direction(
    control_metrics: Mapping[str, Any],
    treatment_metrics: Mapping[str, Any],
    key: str,
) -> str | None:
    if int(control_metrics.get("trades", 0)) <= 0 or int(treatment_metrics.get("trades", 0)) <= 0:
        return None
    control = control_metrics.get(key)
    treatment = treatment_metrics.get(key)
    if key == "pf":
        control = math.inf if control is None else float(control)
        treatment = math.inf if treatment is None else float(treatment)
    elif control is not None and treatment is not None:
        control = float(control)
        treatment = float(treatment)
    else:
        return None
    if math.isinf(control) and math.isinf(treatment):
        return "unchanged"
    if treatment > control + FLOAT_EPSILON:
        return "improved"
    if treatment < control - FLOAT_EPSILON:
        return "worsened"
    return "unchanged"


def _retention(control_trades: int, treatment_trades: int) -> dict[str, Any]:
    ratio = treatment_trades / control_trades if control_trades else None
    return {
        "control_trades": control_trades,
        "treatment_trades": treatment_trades,
        "ratio": ratio,
        "min_ratio": RETENTION_MIN_RATIO,
        "passes": ratio is not None and ratio >= RETENTION_MIN_RATIO,
    }


def _walk_forward_evidence(
    control_walk_forward: Mapping[str, Any],
    treatment_walk_forward: Mapping[str, Any],
) -> dict[str, Any]:
    control_folds = control_walk_forward.get("folds") or []
    treatment_folds = treatment_walk_forward.get("folds") or []
    compared = min(len(control_folds), len(treatment_folds))
    sufficient: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
    for control_fold, treatment_fold in zip(
        control_folds[:compared], treatment_folds[:compared]
    ):
        control_metrics = control_fold["metrics"]
        treatment_metrics = treatment_fold["metrics"]
        if min(
            int(control_metrics["trades"]),
            int(treatment_metrics["trades"]),
        ) >= FOLD_MIN_TRADES:
            sufficient.append((control_metrics, treatment_metrics))

    not_worse = 0
    improved = 0
    for control_metrics, treatment_metrics in sufficient:
        direction = _metric_direction(control_metrics, treatment_metrics, "pf")
        if direction in {"improved", "unchanged"}:
            not_worse += 1
        if direction == "improved":
            improved += 1

    ratio = not_worse / len(sufficient) if sufficient else None
    control_positive = int(control_walk_forward.get("fold_positive", 0))
    treatment_positive = int(treatment_walk_forward.get("fold_positive", 0))
    enough = len(sufficient) >= MIN_SUFFICIENT_FOLDS
    positive_ok = treatment_positive >= control_positive
    pf_ok = (
        enough
        and ratio is not None
        and ratio >= WALK_FORWARD_PF_NOT_WORSE_MIN_RATIO
    )
    return {
        "folds_compared": compared,
        "folds_sufficient": len(sufficient),
        "min_sufficient_folds": MIN_SUFFICIENT_FOLDS,
        "folds_pf_not_worse": not_worse,
        "folds_pf_improved": improved,
        "ratio": ratio,
        "min_ratio": WALK_FORWARD_PF_NOT_WORSE_MIN_RATIO,
        "min_fold_trades": FOLD_MIN_TRADES,
        "control_fold_positive": control_positive,
        "treatment_fold_positive": treatment_positive,
        "positive_fold_count_ok": positive_ok,
        "non_contradicting": bool(pf_ok and positive_ok),
    }


def _exit_diagnostics(
    candles_by_instrument: Mapping[str, pd.DataFrame],
    settings: Settings,
    initial_equity: float,
    expected_trades: int,
) -> dict[str, Any]:
    if not candles_by_instrument:
        return {
            "exit_reason_counts": {},
            "hard_stop_count": 0,
            "zero_r_exit_count": 0,
            "breakeven_specific_count": None,
            "breakeven_specific_note": (
                "Existing simulator labels all stop exits as hard_stop; "
                "breakeven-specific stop exits are not separately identifiable."
            ),
        }

    simulated = _simulate(
        candles_by_instrument,
        settings,
        initial_equity=initial_equity,
        spread_pips=DEFAULT_SPREAD_PIPS,
        slippage_pips=DEFAULT_SLIPPAGE_PIPS,
    )
    trades = simulated["trades"]
    if len(trades) != expected_trades:
        raise RuntimeError(
            "exit diagnostics diverged from run_research base-cost trade count: "
            f"diagnostic={len(trades)} run_research={expected_trades}"
        )
    counts = Counter(str(trade["exit_reason"]) for trade in trades)
    return {
        "exit_reason_counts": dict(sorted(counts.items())),
        "hard_stop_count": counts.get("hard_stop", 0),
        "zero_r_exit_count": sum(
            1 for trade in trades if abs(float(trade.get("r", 0.0))) <= FLOAT_EPSILON
        ),
        "breakeven_specific_count": None,
        "breakeven_specific_note": (
            "Existing simulator labels initial, trailing, and moved-to-entry stops "
            "as hard_stop; no new forward semantics were added to distinguish them."
        ),
    }


def _evaluate(
    control: Mapping[str, Any],
    treatment: Mapping[str, Any],
    anchor: Mapping[str, Any],
    production_defaults_unchanged: bool,
    single_variable_isolation: bool,
) -> dict[str, Any]:
    c_full = control["metrics"]
    t_full = treatment["metrics"]
    c_val = control["validation"]["metrics"]
    t_val = treatment["validation"]["metrics"]
    c_test = control["test"]["metrics"]
    t_test = treatment["test"]["metrics"]
    c_high = control["sensitivity"]["cost_x2"]["metrics"]
    t_high = treatment["sensitivity"]["cost_x2"]["metrics"]

    full_pf = _metric_direction(c_full, t_full, "pf")
    full_avg_r = _metric_direction(c_full, t_full, "avg_r")
    val_pf = _metric_direction(c_val, t_val, "pf")
    test_pf = _metric_direction(c_test, t_test, "pf")
    test_avg_r = _metric_direction(c_test, t_test, "avg_r")
    high_pf = _metric_direction(c_high, t_high, "pf")

    validation_retention = _retention(c_val["trades"], t_val["trades"])
    test_retention = _retention(c_test["trades"], t_test["trades"])
    walk_forward = _walk_forward_evidence(
        control["walk_forward"], treatment["walk_forward"]
    )

    test_maxdd_not_worse = (
        t_test["max_dd"] <= c_test["max_dd"] + FLOAT_EPSILON
    )
    full_maxdd_limit = (
        c_full["max_dd"] * (1 + FULL_MAX_DD_MAX_WORSEN_RATIO) + FLOAT_EPSILON
    )
    full_maxdd_ok = t_full["max_dd"] <= full_maxdd_limit

    supported_gates = {
        "anchor_match": bool(anchor["matches"]),
        "full_pf_improves": full_pf == "improved",
        "full_avg_r_improves": full_avg_r == "improved",
        "validation_pf_improves": val_pf == "improved",
        "test_pf_improves": test_pf == "improved",
        "test_avg_r_not_worse": test_avg_r in {"improved", "unchanged"},
        "test_maxdd_not_worse": bool(test_maxdd_not_worse),
        "full_maxdd_not_more_than_10pct_worse": bool(full_maxdd_ok),
        "validation_retention": bool(validation_retention["passes"]),
        "test_retention": bool(test_retention["passes"]),
        "high_cost_pf_direction_preserved": high_pf == "improved",
        "walk_forward_non_contradicting": bool(walk_forward["non_contradicting"]),
        "production_defaults_unchanged": bool(production_defaults_unchanged),
        "single_variable_isolation": bool(single_variable_isolation),
    }

    sample_adequate = bool(
        validation_retention["passes"]
        and test_retention["passes"]
        and walk_forward["folds_sufficient"] >= MIN_SUFFICIENT_FOLDS
    )
    robustness_mixed = bool(
        high_pf == "improved"
        or (
            walk_forward["folds_sufficient"] >= MIN_SUFFICIENT_FOLDS
            and walk_forward["folds_pf_improved"] > 0
            and walk_forward["treatment_fold_positive"]
            > walk_forward["control_fold_positive"]
        )
    )
    falsified_primary = bool(
        (full_pf == "worsened" and full_avg_r == "worsened")
        or (val_pf == "worsened" and test_pf == "worsened")
    )

    return {
        **supported_gates,
        "directions": {
            "full_pf": full_pf,
            "full_avg_r": full_avg_r,
            "validation_pf": val_pf,
            "test_pf": test_pf,
            "test_avg_r": test_avg_r,
            "high_cost_pf": high_pf,
        },
        "validation_retention_details": validation_retention,
        "test_retention_details": test_retention,
        "walk_forward_details": walk_forward,
        "full_maxdd_limit": full_maxdd_limit,
        "sample_adequate": sample_adequate,
        "robustness_mixed": robustness_mixed,
        "falsified_primary": falsified_primary,
    }


def _classify(evaluation: Mapping[str, Any]) -> tuple[str, str]:
    if not evaluation["anchor_match"]:
        return "INCONCLUSIVE", "Control anchor drifted; treatment conclusion is invalid."
    if not evaluation["production_defaults_unchanged"]:
        return "INCONCLUSIVE", "Production/default Settings changed during the research run."
    if not evaluation["single_variable_isolation"]:
        return "INCONCLUSIVE", "Control and treatment were not isolated to one Settings field."

    if all(bool(evaluation[name]) for name in SUPPORTED_GATE_NAMES):
        return (
            "SUPPORTED",
            "All predeclared support gates passed for the single 1.0R -> 0.5R treatment.",
        )

    if (
        evaluation["sample_adequate"]
        and evaluation["falsified_primary"]
        and not evaluation["robustness_mixed"]
    ):
        return (
            "FALSIFIED",
            "Adequate samples show the predeclared primary performance direction worsened "
            "without material robustness evidence making the result mixed.",
        )

    return (
        "INCONCLUSIVE",
        "The result is mixed, underpowered, or fails one or more robustness/safety gates.",
    )


def run_experiment(
    candles_by_instrument: Mapping[str, pd.DataFrame],
    base_settings: Settings,
    initial_equity: float = 1_000_000.0,
) -> dict[str, Any]:
    before = _production_snapshot()
    control_settings, treatment_settings = build_experiment_settings(base_settings)
    diff = settings_diff(control_settings, treatment_settings)
    isolation = diff == [
        {
            "field": "adaptive_breakeven_trigger_r",
            "control": CONTROL_BREAKEVEN_TRIGGER_R,
            "treatment": TREATMENT_BREAKEVEN_TRIGGER_R,
        }
    ]

    control = run_research(
        candles_by_instrument,
        control_settings,
        initial_equity=initial_equity,
    )
    treatment = run_research(
        candles_by_instrument,
        treatment_settings,
        initial_equity=initial_equity,
    )

    control_exit = _exit_diagnostics(
        candles_by_instrument,
        control_settings,
        initial_equity,
        int(control["metrics"]["trades"]),
    )
    treatment_exit = _exit_diagnostics(
        candles_by_instrument,
        treatment_settings,
        initial_equity,
        int(treatment["metrics"]["trades"]),
    )

    production_defaults_unchanged = _production_snapshot() == before
    anchor = _anchor_comparison(control)
    evaluation = _evaluate(
        control,
        treatment,
        anchor,
        production_defaults_unchanged,
        isolation,
    )
    classification, rationale = _classify(evaluation)

    return {
        "experiment": "adaptive_v2_breakeven_experiment_4",
        "research_only": True,
        "control_breakeven_trigger_r": CONTROL_BREAKEVEN_TRIGGER_R,
        "treatment_breakeven_trigger_r": TREATMENT_BREAKEVEN_TRIGGER_R,
        "pinned_settings": {
            "strategy_profile": STRATEGY_PROFILE,
            "adaptive_fast_ema_days": FAST_EMA_DAYS,
            "adaptive_mid_ema_days": MID_EMA_DAYS,
            "adaptive_slow_ema_days": SLOW_EMA_DAYS,
            "adaptive_score_min": SCORE_MIN,
            "adaptive_max_open_positions": MAX_OPEN_POSITIONS,
            "adaptive_strength_gap_min": STRENGTH_GAP_MIN,
        },
        "settings_diff": diff,
        "control": control,
        "treatment": treatment,
        "control_exit_diagnostics": control_exit,
        "treatment_exit_diagnostics": treatment_exit,
        "anchor": anchor,
        "evaluation": evaluation,
        "classification": classification,
        "classification_rationale": rationale,
        "disclosures": {
            "test_holdout": (
                "Test has already been inspected by prior research and is not a pristine untouched holdout."
            ),
            "equity_curve_limitation": (
                "Pre-existing simulator limitation: adaptive-gate blocked dates can omit equity-curve "
                "points, so DD/CAGR are secondary safety metrics until separately repaired/re-anchored."
            ),
            "causal_input_boundary": (
                "Post-trade MFE/MAE are diagnostics only and are not consumed by treatment decision logic."
            ),
            "paper_state_verification": (
                "Paper DB hash/mtime and primary checkout state are verified externally before publication."
            ),
        },
    }


def dump_json(payload: dict[str, Any], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )


def _format_metrics(label: str, metrics: Mapping[str, Any]) -> str:
    pf = metrics["pf"]
    pf_text = f"{pf:.10f}" if pf is not None else "inf"
    return (
        f"- {label}: trades={metrics['trades']} CAGR={metrics['cagr']:.10f} "
        f"maxDD={metrics['max_dd']:.10f} PF={pf_text} "
        f"win_rate={metrics['win_rate']:.4f} avg_r={metrics['avg_r']:.10f} "
        f"profit={metrics['profit']:.4f}"
    )


def write_report(
    payload: dict[str, Any],
    json_path: str | Path,
    markdown_path: str | Path,
) -> None:
    dump_json(payload, json_path)
    control = payload["control"]
    treatment = payload["treatment"]
    evaluation = payload["evaluation"]

    lines = [
        "# Adaptive v2 Experiment #4 — breakeven trigger 1.0R -> 0.5R",
        "",
        f"Classification: **{payload['classification']}**",
        "",
        payload["classification_rationale"],
        "",
        "## Exact Settings diff",
        f"- {payload['settings_diff']}",
        "",
        "## Anchor check",
    ]
    for key, field in payload["anchor"]["fields"].items():
        lines.append(
            f"- {key}: expected={field['expected']} actual={field['actual']} matches={field['matches']}"
        )

    lines += ["", "## Full"]
    lines.append(_format_metrics("Control 1.0R", control["metrics"]))
    lines.append(_format_metrics("Treatment 0.5R", treatment["metrics"]))

    lines += ["", "## Train / Validation / Test"]
    for split in ("train", "validation", "test"):
        lines.append(f"### {split.capitalize()}")
        lines.append(_format_metrics("Control 1.0R", control[split]["metrics"]))
        lines.append(_format_metrics("Treatment 0.5R", treatment[split]["metrics"]))

    lines += ["", "## High-cost sensitivity"]
    lines.append(
        _format_metrics("Control high-cost", control["sensitivity"]["cost_x2"]["metrics"])
    )
    lines.append(
        _format_metrics("Treatment high-cost", treatment["sensitivity"]["cost_x2"]["metrics"])
    )

    lines += ["", "## Walk-forward"]
    wf = evaluation["walk_forward_details"]
    lines.append(
        f"- sufficiently_sampled_folds={wf['folds_sufficient']} "
        f"(minimum={wf['min_sufficient_folds']}, min_trades_per_arm={wf['min_fold_trades']})"
    )
    lines.append(
        f"- PF not-worse={wf['folds_pf_not_worse']}/{wf['folds_sufficient']} "
        f"ratio={wf['ratio']} required={wf['min_ratio']}"
    )
    lines.append(
        f"- positive folds control={wf['control_fold_positive']} "
        f"treatment={wf['treatment_fold_positive']}"
    )

    lines += ["", "## Validation/Test trade-count retention"]
    for name in ("validation", "test"):
        item = evaluation[f"{name}_retention_details"]
        lines.append(
            f"- {name}: control={item['control_trades']} treatment={item['treatment_trades']} "
            f"ratio={item['ratio']} minimum={item['min_ratio']} passes={item['passes']}"
        )

    lines += ["", "## Exit diagnostics"]
    for label, diagnostic in (
        ("Control 1.0R", payload["control_exit_diagnostics"]),
        ("Treatment 0.5R", payload["treatment_exit_diagnostics"]),
    ):
        lines.append(f"### {label}")
        lines.append(f"- exit_reason_counts={diagnostic['exit_reason_counts']}")
        lines.append(f"- hard_stop_count={diagnostic['hard_stop_count']}")
        lines.append(f"- zero_r_exit_count={diagnostic['zero_r_exit_count']}")
        lines.append(f"- breakeven_specific_count={diagnostic['breakeven_specific_count']}")
        lines.append(f"- note={diagnostic['breakeven_specific_note']}")

    lines += ["", "## Support gates"]
    for name in SUPPORTED_GATE_NAMES:
        lines.append(f"- {name}: {evaluation[name]}")

    lines += ["", "## Disclosures"]
    for key, value in payload["disclosures"].items():
        lines.append(f"- {key}: {value}")

    output = Path(markdown_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    from .fx_research_data import DEFAULT_RESEARCH_DATA_DIR, load_history

    parser = argparse.ArgumentParser(
        description="Research-only Adaptive v2 Experiment #4: breakeven trigger 1.0R vs 0.5R"
    )
    parser.add_argument("command", choices=["run"])
    parser.add_argument("--data-dir", default=DEFAULT_RESEARCH_DATA_DIR)
    parser.add_argument("--json-path", default=DEFAULT_JSON_PATH)
    parser.add_argument("--markdown-path", default=DEFAULT_MARKDOWN_PATH)
    args = parser.parse_args(argv)

    history = load_history(args.data_dir)
    if not history:
        print(f"NO_DATA: no synced history found under {args.data_dir}")
        return 1

    settings = Settings.from_env()
    payload = run_experiment(
        history,
        settings,
        initial_equity=settings.paper_initial_balance,
    )
    write_report(payload, args.json_path, args.markdown_path)
    print(
        "BREAKEVEN_EXPERIMENT_OK "
        f"classification={payload['classification']} "
        f"control_trades={payload['control']['metrics']['trades']} "
        f"treatment_trades={payload['treatment']['metrics']['trades']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
