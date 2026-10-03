from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from .config import Settings
from .research_backtest import run_research

# Adaptive v2 Experiment #2 (GitHub Issue #9): research-only comparison of
# exactly one variable, adaptive_max_open_positions (2 -> 1 selectivity),
# holding every other setting fixed. Both arms use adaptive_score_min=75
# (the production default); this experiment does not carry forward Issue
# #7's treatment score_min=85. This module must never be imported by
# forward/paper execution code (app.engine, app.paper, app.main,
# app.mt5_client).
SCORE_MIN = 75.0
CONTROL_MAX_OPEN_POSITIONS = 2
TREATMENT_MAX_OPEN_POSITIONS = 1

# Anchor observed on real synced MT5 history for the control arm
# (adaptive_max_open_positions=2, adaptive_score_min=75), captured at the
# time this experiment was authored. A mismatch means the repository/data
# state has drifted from the anchor and treatment conclusions must not be
# claimed.
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

# Gate 6: treatment must retain at least half of control's full-sample trade
# count, otherwise the sample is underpowered and the run is INCONCLUSIVE.
RETENTION_MIN_RATIO = 0.5

# Gate 5: full-sample PF/profit are allowed to move against the treatment by
# at most this fraction before being considered "material deterioration"
# (same 10% convention as Issue #7's adaptive v2 experiment #1).
MATERIALITY_TOLERANCE = 0.10

# Minimum fraction of comparable walk-forward folds whose treatment max DD
# must not be worse than control's for Gate 8 (walk-forward does not
# materially contradict the max-DD direction).
WALK_FORWARD_NOT_WORSE_MIN_RATIO = 0.5

# Tolerance for strict "improves"/"not worse" max-DD comparisons, to absorb
# floating point noise without treating a genuine reversal as a tie.
MAX_DD_EPSILON = 1e-12

# Environment keys that would indicate a persisted override if changed by
# running the experiment; only in-memory dataclasses.replace() is allowed.
_ENV_WATCH_KEYS = ("ADAPTIVE_MAX_OPEN_POSITIONS", "ADAPTIVE_SCORE_MIN", "STRATEGY_PROFILE")


def build_experiment_settings(base: Settings) -> tuple[Settings, Settings]:
    """Return (control, treatment) Settings that differ only in adaptive_max_open_positions.

    Both are derived in-memory from the same base Settings via
    dataclasses.replace(); no environment variable or config file is read
    or written, so Settings.from_env() defaults are unaffected. Both arms
    explicitly pin adaptive_score_min=75 so this experiment never drifts
    onto Issue #7's treatment score_min=85.
    """
    control = replace(
        base,
        adaptive_score_min=SCORE_MIN,
        adaptive_max_open_positions=CONTROL_MAX_OPEN_POSITIONS,
    )
    treatment = replace(
        base,
        adaptive_score_min=SCORE_MIN,
        adaptive_max_open_positions=TREATMENT_MAX_OPEN_POSITIONS,
    )
    return control, treatment


def _pf_rank(pf: float | None) -> float:
    # _metrics() returns None for "infinite" PF (winners, no losers) rather
    # than float("inf") to stay JSON-serializable; recover the ordering here.
    return math.inf if pf is None else float(pf)


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
                actual, expected, rel_tol=ANCHOR_FLOAT_REL_TOL, abs_tol=ANCHOR_PROFIT_ABS_TOL
            )
        else:
            matches = actual is not None and math.isclose(
                actual, expected, rel_tol=ANCHOR_FLOAT_REL_TOL, abs_tol=ANCHOR_FLOAT_ABS_TOL
            )
        fields[key] = {"expected": expected, "actual": actual, "matches": matches}
        all_match = all_match and matches
    return {"matches": all_match, "fields": fields}


def _retention(control_trades: int, treatment_trades: int) -> dict[str, Any]:
    ratio = (treatment_trades / control_trades) if control_trades else None
    passes = ratio is not None and ratio >= RETENTION_MIN_RATIO
    return {
        "control_trades": control_trades,
        "treatment_trades": treatment_trades,
        "ratio": ratio,
        "min_ratio": RETENTION_MIN_RATIO,
        "passes_min_ratio": passes,
    }


def _full_sample_not_materially_worse(
    control_metrics: Mapping[str, Any], treatment_metrics: Mapping[str, Any]
) -> bool:
    control_pf = _pf_rank(control_metrics["pf"])
    treatment_pf = _pf_rank(treatment_metrics["pf"])
    pf_ok = treatment_pf >= control_pf * (1 - MATERIALITY_TOLERANCE)

    control_profit = control_metrics["profit"]
    buffer = abs(control_profit) * MATERIALITY_TOLERANCE
    profit_ok = treatment_metrics["profit"] >= control_profit - buffer
    return bool(pf_ok and profit_ok)


def _walk_forward_no_contradiction(
    control_walk_forward: Mapping[str, Any], treatment_walk_forward: Mapping[str, Any]
) -> tuple[bool, dict[str, Any]]:
    """Gate 8: walk-forward does not materially contradict the max-DD direction.

    Compares each already-exposed fold's "metrics"."max_dd" (per-fold output
    of app.research_backtest.run_research's walk_forward.folds) between
    control and treatment; passes if at least half of the comparable folds
    have a treatment max DD that is not worse than control's.
    """
    control_folds = control_walk_forward.get("folds") or []
    treatment_folds = treatment_walk_forward.get("folds") or []
    total = min(len(control_folds), len(treatment_folds))
    if total == 0:
        return True, {
            "folds_compared": 0,
            "folds_not_worse": 0,
            "ratio": None,
            "sample_sufficient": False,
        }
    not_worse = sum(
        1
        for control_fold, treatment_fold in zip(control_folds[:total], treatment_folds[:total])
        if treatment_fold["metrics"]["max_dd"] <= control_fold["metrics"]["max_dd"] + MAX_DD_EPSILON
    )
    ratio = not_worse / total
    return ratio >= WALK_FORWARD_NOT_WORSE_MIN_RATIO, {
        "folds_compared": total,
        "folds_not_worse": not_worse,
        "ratio": ratio,
        "sample_sufficient": True,
    }


def _evaluate_gates(
    control: Mapping[str, Any],
    treatment: Mapping[str, Any],
    anchor: Mapping[str, Any],
    env_unchanged: bool,
) -> dict[str, Any]:
    control_full = control["metrics"]
    treatment_full = treatment["metrics"]
    control_test = control["test"]["metrics"]
    treatment_test = treatment["test"]["metrics"]
    control_validation = control["validation"]["metrics"]
    treatment_validation = treatment["validation"]["metrics"]
    control_train = control["train"]["metrics"]
    treatment_train = treatment["train"]["metrics"]
    control_high = control["sensitivity"]["cost_x2"]["metrics"]
    treatment_high = treatment["sensitivity"]["cost_x2"]["metrics"]

    retention = _retention(control_full["trades"], treatment_full["trades"])

    # Gate 2: treatment full-sample max DD improves (strictly lower).
    full_maxdd_improves = treatment_full["max_dd"] < control_full["max_dd"] - MAX_DD_EPSILON
    # Gate 3: treatment Test max DD does not worsen.
    test_maxdd_not_worse = treatment_test["max_dd"] <= control_test["max_dd"] + MAX_DD_EPSILON

    # Gate 4: Validation and Test PF are not both worse than control, and
    # the result is not Train-only (Train PF improving while both
    # out-of-sample splits are worse would be exactly a Train-only result,
    # which is a subset of "both OOS splits worse" and therefore already
    # covered by the first condition below).
    validation_pf_worse = _pf_rank(treatment_validation["pf"]) < _pf_rank(control_validation["pf"])
    test_pf_worse = _pf_rank(treatment_test["pf"]) < _pf_rank(control_test["pf"])
    oos_pf_both_worse = validation_pf_worse and test_pf_worse
    train_pf_improves = _pf_rank(treatment_train["pf"]) > _pf_rank(control_train["pf"])
    train_only_result = bool(train_pf_improves and oos_pf_both_worse)
    validation_and_test_pf_not_both_worse = not oos_pf_both_worse

    # Gate 5: full-sample PF/profit do not materially deteriorate.
    full_sample_not_materially_worse = _full_sample_not_materially_worse(control_full, treatment_full)

    # Gate 7: high-cost does not reverse the robustness conclusion (the
    # max-DD direction observed at base cost).
    high_cost_direction_improves = treatment_high["max_dd"] < control_high["max_dd"] - MAX_DD_EPSILON
    high_cost_no_reversal = (not full_maxdd_improves) or high_cost_direction_improves

    # Gate 8: walk-forward does not materially contradict direction.
    walk_forward_no_contradiction, walk_forward_details = _walk_forward_no_contradiction(
        control["walk_forward"], treatment["walk_forward"]
    )

    return {
        "anchor_match": bool(anchor["matches"]),
        "full_maxdd_improves": bool(full_maxdd_improves),
        "test_maxdd_not_worse": bool(test_maxdd_not_worse),
        "validation_and_test_pf_not_both_worse": bool(validation_and_test_pf_not_both_worse),
        "full_sample_not_materially_worse": bool(full_sample_not_materially_worse),
        "retention": bool(retention["passes_min_ratio"]),
        "high_cost_no_reversal": bool(high_cost_no_reversal),
        "walk_forward_no_contradiction": bool(walk_forward_no_contradiction),
        "production_defaults_unchanged": bool(env_unchanged),
        "details": {
            "retention": retention,
            "train_only_result": train_only_result,
            "walk_forward": walk_forward_details,
            "validation": {
                "control_pf": control_validation["pf"],
                "treatment_pf": treatment_validation["pf"],
            },
            "test": {
                "control_pf": control_test["pf"],
                "treatment_pf": treatment_test["pf"],
                "control_max_dd": control_test["max_dd"],
                "treatment_max_dd": treatment_test["max_dd"],
            },
            "full_sample": {
                "control_pf": control_full["pf"],
                "treatment_pf": treatment_full["pf"],
                "control_profit": control_full["profit"],
                "treatment_profit": treatment_full["profit"],
                "control_max_dd": control_full["max_dd"],
                "treatment_max_dd": treatment_full["max_dd"],
            },
            "high_cost": {
                "control_max_dd": control_high["max_dd"],
                "treatment_max_dd": treatment_high["max_dd"],
                "base_direction_improves": full_maxdd_improves,
            },
        },
    }


def _classify(gates: Mapping[str, Any]) -> tuple[str, str]:
    """Apply the 9 predeclared Issue #9 gates in priority order.

    Gates 1 (anchor), 6 (retention) and 9 (production defaults) are
    validity preconditions: if any fails, the run is INCONCLUSIVE regardless
    of the other gates. Gates 2/3 (full and Test max DD) are the primary
    falsification condition for this max-open-positions experiment. Gates
    4/5/7/8 are additionally required for SUPPORTED: lower DD alone is not
    enough if PF/profit/out-of-sample robustness collapses.
    """
    if not gates["anchor_match"]:
        return (
            "INCONCLUSIVE",
            "Control anchor did not match the repository tolerance; treatment cannot be "
            "evaluated against a trusted baseline.",
        )
    if not gates["retention"]:
        return (
            "INCONCLUSIVE",
            "Treatment trade count fell below 50% of control; sample is underpowered for a conclusion.",
        )
    if not gates["production_defaults_unchanged"]:
        return (
            "INCONCLUSIVE",
            "Production defaults or forward semantics changed during experiment execution.",
        )
    if not gates["full_maxdd_improves"] or not gates["test_maxdd_not_worse"]:
        return (
            "FALSIFIED",
            "Treatment did not improve full-sample max DD over control, or treatment Test max "
            "DD worsened.",
        )
    if (
        gates["validation_and_test_pf_not_both_worse"]
        and gates["full_sample_not_materially_worse"]
        and gates["high_cost_no_reversal"]
        and gates["walk_forward_no_contradiction"]
    ):
        return (
            "SUPPORTED",
            "Treatment reduced full-sample max DD without worsening Test max DD, Validation and "
            "Test PF were not both worse (so the result is not Train-only), full-sample PF/profit "
            "did not materially deteriorate, high-cost did not reverse the max-DD conclusion, and "
            "walk-forward did not materially contradict the direction.",
        )
    return (
        "INCONCLUSIVE",
        "Treatment reduced max DD, but one or more secondary robustness gates (Validation/Test "
        "PF, full-sample materiality, high-cost reversal, or walk-forward direction) did not "
        "pass; lower DD alone is not sufficient to claim support.",
    )


def run_experiment(
    candles_by_instrument: Mapping[str, pd.DataFrame],
    base_settings: Settings,
    initial_equity: float = 1_000_000.0,
) -> dict[str, Any]:
    env_before = {key: os.environ.get(key) for key in _ENV_WATCH_KEYS}

    control_settings, treatment_settings = build_experiment_settings(base_settings)

    control = run_research(candles_by_instrument, control_settings, initial_equity=initial_equity)
    treatment = run_research(candles_by_instrument, treatment_settings, initial_equity=initial_equity)

    env_after = {key: os.environ.get(key) for key in _ENV_WATCH_KEYS}
    env_unchanged = env_before == env_after

    anchor = _anchor_comparison(control)
    gates = _evaluate_gates(control, treatment, anchor, env_unchanged)
    classification, rationale = _classify(gates)

    return {
        "experiment": "adaptive_v2_max_open_positions_experiment_2",
        "control_max_open_positions": CONTROL_MAX_OPEN_POSITIONS,
        "treatment_max_open_positions": TREATMENT_MAX_OPEN_POSITIONS,
        "score_min": SCORE_MIN,
        "control": control,
        "treatment": treatment,
        "anchor": anchor,
        "gates": gates,
        "classification": classification,
        "classification_rationale": rationale,
    }


def dump_json(payload: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )


def _format_metrics(label: str, metrics: Mapping[str, Any]) -> str:
    pf = metrics["pf"]
    pf_text = f"{pf:.10f}" if pf is not None else "inf"
    return (
        f"- {label}: trades={metrics['trades']} CAGR={metrics['cagr']:.10f} "
        f"maxDD={metrics['max_dd']:.10f} PF={pf_text} win_rate={metrics['win_rate']:.4f} "
        f"avg_r={metrics['avg_r']:.4f} profit={metrics['profit']:.4f}"
    )


def write_report(payload: dict[str, Any], json_path: str | Path, markdown_path: str | Path) -> None:
    dump_json(payload, json_path)

    control = payload["control"]
    treatment = payload["treatment"]
    gates = payload["gates"]

    lines = [
        "# Adaptive v2 Experiment #2 — adaptive_max_open_positions 2 -> 1 (research-only, "
        "score_min=75 both arms)",
        "",
        f"Classification: **{payload['classification']}**",
        "",
        payload["classification_rationale"],
        "",
        "## Anchor check (control, adaptive_max_open_positions=2, score_min=75)",
    ]
    for key, field in payload["anchor"]["fields"].items():
        lines.append(f"- {key}: expected={field['expected']} actual={field['actual']} matches={field['matches']}")

    lines += ["", "## Full sample"]
    lines.append(_format_metrics("Control", control["metrics"]))
    lines.append(_format_metrics("Treatment", treatment["metrics"]))

    lines += ["", "## Train / Validation / Test"]
    for split in ("train", "validation", "test"):
        lines.append(f"### {split.capitalize()}")
        lines.append(_format_metrics("Control", control[split]["metrics"]))
        lines.append(_format_metrics("Treatment", treatment[split]["metrics"]))

    lines += ["", "## Base-cost / high-cost sensitivity"]
    lines.append(_format_metrics("Control base-cost", control["sensitivity"]["base"]["metrics"]))
    lines.append(_format_metrics("Control high-cost", control["sensitivity"]["cost_x2"]["metrics"]))
    lines.append(_format_metrics("Treatment base-cost", treatment["sensitivity"]["base"]["metrics"]))
    lines.append(_format_metrics("Treatment high-cost", treatment["sensitivity"]["cost_x2"]["metrics"]))

    lines += ["", "## Walk-forward"]
    lines.append(
        f"- Control fold_positive/fold_total={control['walk_forward']['fold_positive']}/"
        f"{control['walk_forward']['fold_total']}"
    )
    lines.append(
        f"- Treatment fold_positive/fold_total={treatment['walk_forward']['fold_positive']}/"
        f"{treatment['walk_forward']['fold_total']}"
    )
    wf_details = gates["details"]["walk_forward"]
    lines.append(
        f"- Max-DD not-worse folds: {wf_details['folds_not_worse']}/{wf_details['folds_compared']} "
        f"ratio={wf_details['ratio']} sample_sufficient={wf_details['sample_sufficient']}"
    )

    lines += ["", "## Trade-count retention"]
    retention = gates["details"]["retention"]
    lines.append(
        f"- control={retention['control_trades']} treatment={retention['treatment_trades']} "
        f"ratio={retention['ratio']} min_ratio={retention['min_ratio']} passes={retention['passes_min_ratio']}"
    )

    lines += ["", "## Gates"]
    for name in (
        "anchor_match",
        "full_maxdd_improves",
        "test_maxdd_not_worse",
        "validation_and_test_pf_not_both_worse",
        "full_sample_not_materially_worse",
        "retention",
        "high_cost_no_reversal",
        "walk_forward_no_contradiction",
        "production_defaults_unchanged",
    ):
        lines.append(f"- {name}: {gates[name]}")

    Path(markdown_path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    from .fx_research_data import DEFAULT_RESEARCH_DATA_DIR, load_history

    parser = argparse.ArgumentParser(
        description="Research-only Adaptive v2 Experiment #2: adaptive_max_open_positions 2 vs 1"
    )
    parser.add_argument("command", choices=["run"])
    parser.add_argument("--data-dir", default=DEFAULT_RESEARCH_DATA_DIR)
    parser.add_argument(
        "--json-path",
        default="data/research/adaptive_v2_max_positions_experiment.json",
    )
    parser.add_argument(
        "--markdown-path",
        default="data/research/adaptive_v2_max_positions_experiment.md",
    )
    args = parser.parse_args(argv)

    settings = Settings.from_env()
    history = load_history(args.data_dir)
    if not history:
        print(f"NO_DATA: no synced history found under {args.data_dir}")
        return 1

    payload = run_experiment(history, settings, initial_equity=settings.paper_initial_balance)
    write_report(payload, args.json_path, args.markdown_path)
    print(
        "MAX_POSITIONS_EXPERIMENT_OK "
        f"classification={payload['classification']} "
        f"control_trades={payload['control']['metrics']['trades']} "
        f"treatment_trades={payload['treatment']['metrics']['trades']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
