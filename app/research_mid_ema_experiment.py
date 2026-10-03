from __future__ import annotations

import argparse
import dataclasses
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

# Adaptive v2 Experiment #3 (GitHub Issue #12): research-only comparison of
# exactly one Settings field, adaptive_mid_ema_days (50 -> 75 direction
# horizon), holding every other setting fixed. This is a D1 in-memory
# experiment on the existing Research simulator only: no 1H data path, no
# VWAP, and no env/config persistence. This module must never be imported by
# forward/paper execution code (app.engine, app.paper, app.main,
# app.mt5_client).
CONTROL_MID_EMA_DAYS = 50
TREATMENT_MID_EMA_DAYS = 75
FAST_EMA_DAYS = 20
SLOW_EMA_DAYS = 200
SCORE_MIN = 75.0
MAX_OPEN_POSITIONS = 2
STRENGTH_GAP_MIN = 0.3

# Anchor observed on real synced MT5 history for the control arm
# (adaptive_mid_ema_days=50 with every other setting at its default), which is
# the same arm as Issue #9's control. A mismatch means the repository/data
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

# Gate 7: treatment must retain at least half of control's full-sample trade
# count, otherwise the sample is underpowered and the run is INCONCLUSIVE.
RETENTION_MIN_RATIO = 0.5

# Gate 6: treatment full-sample max DD may be at most 10% worse (relative)
# than control's.
FULL_MAX_DD_MAX_WORSEN_RATIO = 0.10

# Gate 9: at least this fraction of sufficiently sampled walk-forward folds
# must have treatment PF not worse than control PF.
WALK_FORWARD_PF_NOT_WORSE_MIN_RATIO = 0.5

# A walk-forward fold is "sufficiently sampled" for Gate 9 only when both arms
# closed at least this many trades in it. Not specified in Issue #12; chosen
# here so that PF on a fold with one or two trades cannot decide the gate.
FOLD_MIN_TRADES = 3

# Tolerance for strict "improves"/"not worse" max-DD comparisons, to absorb
# floating point noise without treating a genuine reversal as a tie.
MAX_DD_EPSILON = 1e-12

# Environment keys that would indicate a persisted override if changed by
# running the experiment; only in-memory dataclasses.replace() is allowed.
_ENV_WATCH_KEYS = (
    "ADAPTIVE_MID_EMA_DAYS",
    "ADAPTIVE_FAST_EMA_DAYS",
    "ADAPTIVE_SLOW_EMA_DAYS",
    "ADAPTIVE_SCORE_MIN",
    "ADAPTIVE_MAX_OPEN_POSITIONS",
    "ADAPTIVE_STRENGTH_GAP_MIN",
    "STRATEGY_PROFILE",
)

# Gate names in Issue #12 predeclared order (gate N is GATE_NAMES[N - 1]).
GATE_NAMES = (
    "anchor_match",
    "full_pf_improves",
    "full_avg_r_improves",
    "validation_and_test_pf_improve",
    "test_maxdd_not_worse",
    "full_maxdd_not_more_than_10pct_worse",
    "retention",
    "high_cost_pf_direction_preserved",
    "walk_forward_positive_and_pf_not_worse",
    "production_defaults_unchanged",
)


def build_experiment_settings(base: Settings) -> tuple[Settings, Settings]:
    """Return (control, treatment) Settings that differ only in adaptive_mid_ema_days.

    Both are derived in-memory from the same base Settings via
    dataclasses.replace(); no environment variable or config file is read
    or written, so Settings.from_env() defaults are unaffected. Every other
    field this experiment depends on is pinned explicitly in both arms so the
    comparison cannot drift onto a carried-forward Issue #7 or #9 value.
    """
    pinned = {
        "adaptive_fast_ema_days": FAST_EMA_DAYS,
        "adaptive_slow_ema_days": SLOW_EMA_DAYS,
        "adaptive_score_min": SCORE_MIN,
        "adaptive_max_open_positions": MAX_OPEN_POSITIONS,
        "adaptive_strength_gap_min": STRENGTH_GAP_MIN,
    }
    control = replace(base, adaptive_mid_ema_days=CONTROL_MID_EMA_DAYS, **pinned)
    treatment = replace(base, adaptive_mid_ema_days=TREATMENT_MID_EMA_DAYS, **pinned)
    return control, treatment


def _pf_rank(pf: float | None) -> float:
    # _metrics() returns None for "infinite" PF (winners, no losers) rather
    # than float("inf") to stay JSON-serializable; recover the ordering here.
    return math.inf if pf is None else float(pf)


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


def _walk_forward_gate(
    control_walk_forward: Mapping[str, Any], treatment_walk_forward: Mapping[str, Any]
) -> tuple[bool, dict[str, Any]]:
    """Gate 9: positive-fold count >= control AND >=50% of sufficiently sampled folds have PF not worse.

    Both checks read only the already-exposed per-fold output of
    app.research_backtest.run_research (walk_forward.folds[*].metrics and
    walk_forward.fold_positive). Insufficient folds never pass the PF clause.
    """
    control_folds = control_walk_forward.get("folds") or []
    treatment_folds = treatment_walk_forward.get("folds") or []
    total = min(len(control_folds), len(treatment_folds))
    control_positive = int(control_walk_forward.get("fold_positive", 0))
    treatment_positive = int(treatment_walk_forward.get("fold_positive", 0))
    positive_ok = treatment_positive >= control_positive

    sufficient = [
        (control_fold["metrics"], treatment_fold["metrics"])
        for control_fold, treatment_fold in zip(control_folds[:total], treatment_folds[:total])
        if min(control_fold["metrics"]["trades"], treatment_fold["metrics"]["trades"]) >= FOLD_MIN_TRADES
    ]
    not_worse = sum(
        1
        for control_metrics, treatment_metrics in sufficient
        if _pf_rank(treatment_metrics["pf"]) >= _pf_rank(control_metrics["pf"])
    )
    ratio = (not_worse / len(sufficient)) if sufficient else None
    pf_ok = bool(sufficient) and ratio >= WALK_FORWARD_PF_NOT_WORSE_MIN_RATIO
    return bool(positive_ok and pf_ok), {
        "control_fold_positive": control_positive,
        "treatment_fold_positive": treatment_positive,
        "fold_positive_ok": bool(positive_ok),
        "folds_compared": total,
        "folds_sufficient": len(sufficient),
        "folds_pf_not_worse": not_worse,
        "ratio": ratio,
        "min_ratio": WALK_FORWARD_PF_NOT_WORSE_MIN_RATIO,
        "min_fold_trades": FOLD_MIN_TRADES,
        "pf_not_worse_ok": bool(pf_ok),
    }


def _evaluate_gates(
    control: Mapping[str, Any],
    treatment: Mapping[str, Any],
    anchor: Mapping[str, Any],
    env_unchanged: bool,
) -> dict[str, Any]:
    control_full = control["metrics"]
    treatment_full = treatment["metrics"]
    control_validation = control["validation"]["metrics"]
    treatment_validation = treatment["validation"]["metrics"]
    control_test = control["test"]["metrics"]
    treatment_test = treatment["test"]["metrics"]
    control_high = control["sensitivity"]["cost_x2"]["metrics"]
    treatment_high = treatment["sensitivity"]["cost_x2"]["metrics"]

    retention = _retention(control_full["trades"], treatment_full["trades"])

    # Gate 2: full-sample PF strictly improves.
    full_pf_improves = _pf_rank(treatment_full["pf"]) > _pf_rank(control_full["pf"])
    # Gate 3: full-sample average R strictly improves.
    full_avg_r_improves = treatment_full["avg_r"] > control_full["avg_r"]
    # Gate 4: Validation PF and Test PF both strictly improve (Full-only or
    # Test-only improvement is insufficient).
    validation_pf_improves = _pf_rank(treatment_validation["pf"]) > _pf_rank(control_validation["pf"])
    test_pf_improves = _pf_rank(treatment_test["pf"]) > _pf_rank(control_test["pf"])
    # Gate 5: Test max DD is not worse.
    test_maxdd_not_worse = treatment_test["max_dd"] <= control_test["max_dd"] + MAX_DD_EPSILON
    # Gate 6: full-sample max DD is at most 10% worse than control's.
    full_maxdd_limit = control_full["max_dd"] * (1 + FULL_MAX_DD_MAX_WORSEN_RATIO) + MAX_DD_EPSILON
    full_maxdd_not_more_than_10pct_worse = treatment_full["max_dd"] <= full_maxdd_limit
    # Gate 8: high-cost PF keeps the same improvement direction as high-cost control.
    high_cost_pf_direction_preserved = _pf_rank(treatment_high["pf"]) > _pf_rank(control_high["pf"])
    # Gate 9: walk-forward positive-fold count and sampled-fold PF direction.
    walk_forward_ok, walk_forward_details = _walk_forward_gate(
        control["walk_forward"], treatment["walk_forward"]
    )

    return {
        "anchor_match": bool(anchor["matches"]),
        "full_pf_improves": bool(full_pf_improves),
        "full_avg_r_improves": bool(full_avg_r_improves),
        "validation_and_test_pf_improve": bool(validation_pf_improves and test_pf_improves),
        "test_maxdd_not_worse": bool(test_maxdd_not_worse),
        "full_maxdd_not_more_than_10pct_worse": bool(full_maxdd_not_more_than_10pct_worse),
        "retention": bool(retention["passes_min_ratio"]),
        "high_cost_pf_direction_preserved": bool(high_cost_pf_direction_preserved),
        "walk_forward_positive_and_pf_not_worse": bool(walk_forward_ok),
        "production_defaults_unchanged": bool(env_unchanged),
        "details": {
            "retention": retention,
            "walk_forward": walk_forward_details,
            "oos_pf_both_not_improving": bool(not validation_pf_improves and not test_pf_improves),
            "validation": {
                "control_pf": control_validation["pf"],
                "treatment_pf": treatment_validation["pf"],
                "pf_improves": bool(validation_pf_improves),
            },
            "test": {
                "control_pf": control_test["pf"],
                "treatment_pf": treatment_test["pf"],
                "control_max_dd": control_test["max_dd"],
                "treatment_max_dd": treatment_test["max_dd"],
                "pf_improves": bool(test_pf_improves),
            },
            "full_sample": {
                "control_pf": control_full["pf"],
                "treatment_pf": treatment_full["pf"],
                "control_avg_r": control_full["avg_r"],
                "treatment_avg_r": treatment_full["avg_r"],
                "control_max_dd": control_full["max_dd"],
                "treatment_max_dd": treatment_full["max_dd"],
                "max_dd_limit": full_maxdd_limit,
            },
            "high_cost": {
                "control_pf": control_high["pf"],
                "treatment_pf": treatment_high["pf"],
            },
        },
    }


def _classify(gates: Mapping[str, Any]) -> tuple[str, str]:
    """Apply the 10 predeclared Issue #12 gates.

    Gates 1 (anchor), 10 (production defaults) and 7 (retention) are validity
    preconditions: if any fails, the run is INCONCLUSIVE and no treatment
    support or falsification is claimed. FALSIFIED requires contrary evidence
    on the primary direction: full-sample PF or avg R not improving, or both
    Validation and Test PF not improving. SUPPORTED requires every gate.
    """
    if not gates["anchor_match"]:
        return (
            "INCONCLUSIVE",
            "Control anchor did not match the repository tolerance; treatment cannot be "
            "evaluated against a trusted baseline.",
        )
    if not gates["production_defaults_unchanged"]:
        return (
            "INCONCLUSIVE",
            "Production defaults or forward semantics changed during experiment execution.",
        )
    if not gates["retention"]:
        return (
            "INCONCLUSIVE",
            "Treatment trade count fell below 50% of control; sample is underpowered for a conclusion.",
        )
    if (
        not gates["full_pf_improves"]
        or not gates["full_avg_r_improves"]
        or gates["details"]["oos_pf_both_not_improving"]
    ):
        return (
            "FALSIFIED",
            "Treatment (adaptive_mid_ema_days=75) did not improve full-sample PF or average R over "
            "control (adaptive_mid_ema_days=50), or neither Validation nor Test PF improved.",
        )
    if all(gates[name] for name in GATE_NAMES):
        return (
            "SUPPORTED",
            "Treatment improved full-sample PF and average R, improved both Validation and Test PF, "
            "kept Test and full-sample max DD within limits, retained at least 50% of trades, "
            "preserved the PF direction at high cost, and passed the walk-forward checks.",
        )
    return (
        "INCONCLUSIVE",
        "Treatment improved the primary PF/avg-R direction, but one or more predeclared robustness "
        "gates did not pass; full-sample-only or Test-only improvement is not sufficient to claim support.",
    )


def run_experiment(
    candles_by_instrument: Mapping[str, pd.DataFrame],
    base_settings: Settings,
    initial_equity: float = 1_000_000.0,
) -> dict[str, Any]:
    before = _production_snapshot()

    control_settings, treatment_settings = build_experiment_settings(base_settings)

    control = run_research(candles_by_instrument, control_settings, initial_equity=initial_equity)
    treatment = run_research(candles_by_instrument, treatment_settings, initial_equity=initial_equity)

    env_unchanged = _production_snapshot() == before

    anchor = _anchor_comparison(control)
    gates = _evaluate_gates(control, treatment, anchor, env_unchanged)
    classification, rationale = _classify(gates)

    return {
        "experiment": "adaptive_v2_mid_ema_experiment_3",
        "control_mid_ema_days": CONTROL_MID_EMA_DAYS,
        "treatment_mid_ema_days": TREATMENT_MID_EMA_DAYS,
        "pinned_settings": {
            "adaptive_fast_ema_days": FAST_EMA_DAYS,
            "adaptive_slow_ema_days": SLOW_EMA_DAYS,
            "adaptive_score_min": SCORE_MIN,
            "adaptive_max_open_positions": MAX_OPEN_POSITIONS,
            "adaptive_strength_gap_min": STRENGTH_GAP_MIN,
        },
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


def _format_groups(label: str, groups: Mapping[str, Any]) -> list[str]:
    if not groups:
        return [f"- {label}: none"]
    lines = []
    for key in sorted(groups):
        group = groups[key]
        pf = group["pf"]
        pf_text = f"{pf:.4f}" if pf is not None else "inf"
        lines.append(
            f"- {label} {key}: trades={group['trades']} PF={pf_text} "
            f"avg_r={group['avg_r']:.4f} profit={group['profit']:.4f}"
        )
    return lines


def write_report(payload: dict[str, Any], json_path: str | Path, markdown_path: str | Path) -> None:
    dump_json(payload, json_path)

    control = payload["control"]
    treatment = payload["treatment"]
    gates = payload["gates"]
    details = gates["details"]

    lines = [
        "# Adaptive v2 Experiment #3 — adaptive_mid_ema_days 50 -> 75 (research-only, D1 in-memory, "
        "score_min=75 / max_positions=2 / strength_gap_min=0.3 / fast=20 / slow=200 both arms)",
        "",
        f"Classification: **{payload['classification']}**",
        "",
        payload["classification_rationale"],
        "",
        "## Anchor check (control, adaptive_mid_ema_days=50)",
    ]
    for key, field in payload["anchor"]["fields"].items():
        lines.append(f"- {key}: expected={field['expected']} actual={field['actual']} matches={field['matches']}")

    lines += ["", "## Full sample"]
    lines.append(_format_metrics("Control (mid EMA 50)", control["metrics"]))
    lines.append(_format_metrics("Treatment (mid EMA 75)", treatment["metrics"]))

    lines += ["", "## Train / Validation / Test"]
    for split in ("train", "validation", "test"):
        lines.append(f"### {split.capitalize()}")
        lines.append(_format_metrics("Control (mid EMA 50)", control[split]["metrics"]))
        lines.append(_format_metrics("Treatment (mid EMA 75)", treatment[split]["metrics"]))

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
    wf = details["walk_forward"]
    lines.append(
        f"- PF not-worse folds (sufficiently sampled, min {wf['min_fold_trades']} trades per arm): "
        f"{wf['folds_pf_not_worse']}/{wf['folds_sufficient']} of {wf['folds_compared']} compared "
        f"ratio={wf['ratio']}"
    )

    lines += ["", "## Trade-count retention"]
    retention = details["retention"]
    lines.append(
        f"- control={retention['control_trades']} treatment={retention['treatment_trades']} "
        f"ratio={retention['ratio']} min_ratio={retention['min_ratio']} passes={retention['passes_min_ratio']}"
    )

    lines += ["", "## By regime (diagnostic, base cost)"]
    lines += _format_groups("Control", control["by_regime"])
    lines += _format_groups("Treatment", treatment["by_regime"])

    lines += ["", "## By score band (diagnostic, base cost)"]
    lines += _format_groups("Control", control["by_score_band"])
    lines += _format_groups("Treatment", treatment["by_score_band"])

    lines += ["", "## Gates"]
    for number, name in enumerate(GATE_NAMES, start=1):
        lines.append(f"{number}. {name}: {gates[name]}")

    Path(markdown_path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    from .fx_research_data import DEFAULT_RESEARCH_DATA_DIR, load_history

    parser = argparse.ArgumentParser(
        description="Research-only Adaptive v2 Experiment #3: adaptive_mid_ema_days 50 vs 75"
    )
    parser.add_argument("command", choices=["run"])
    parser.add_argument("--data-dir", default=DEFAULT_RESEARCH_DATA_DIR)
    parser.add_argument(
        "--json-path",
        default="data/research/adaptive_v2_mid_ema_experiment.json",
    )
    parser.add_argument(
        "--markdown-path",
        default="data/research/adaptive_v2_mid_ema_experiment.md",
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
        "MID_EMA_EXPERIMENT_OK "
        f"classification={payload['classification']} "
        f"control_trades={payload['control']['metrics']['trades']} "
        f"treatment_trades={payload['treatment']['metrics']['trades']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
