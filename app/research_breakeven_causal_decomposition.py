from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence

from .config import Settings
from .research_backtest import (
    DEFAULT_SLIPPAGE_PIPS,
    DEFAULT_SPREAD_PIPS,
    _simulate,
    _split_dates,
)
from .research_breakeven_experiment import (
    CONTROL_BREAKEVEN_TRIGGER_R,
    FAST_EMA_DAYS,
    MAX_OPEN_POSITIONS,
    MID_EMA_DAYS,
    SCORE_MIN,
    SLOW_EMA_DAYS,
    STRENGTH_GAP_MIN,
    STRATEGY_PROFILE,
    TREATMENT_BREAKEVEN_TRIGGER_R,
    _production_snapshot,
    build_experiment_settings,
    settings_diff,
)

DEFAULT_JSON_PATH = "data/research/adaptive_v2_breakeven_causal_decomposition.json"
DEFAULT_MARKDOWN_PATH = "data/research/adaptive_v2_breakeven_causal_decomposition.md"

FLOAT_EPSILON = 1e-9
DOMINANCE_MIN_SHARE = 0.60
MATERIAL_MIN_SHARE = 0.40

ASSOCIATION_CATEGORIES = (
    "control_same_instrument_open",
    "control_capacity_full",
    "control_candidate_day_state_divergence",
    "other_state_divergence",
)

DIAGNOSTIC_LABELS = (
    "MATCHED_OUTCOME_DOMINANT",
    "PATH_DIVERGENCE_DOMINANT",
    "MIXED",
    "INSUFFICIENT_EVIDENCE",
)


def entry_identity(trade: Mapping[str, Any]) -> tuple[str, str, str]:
    return (
        str(trade["instrument"]),
        str(trade["side"]),
        str(trade["entry_date"]),
    )


def index_unique_trades(
    trades: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str, str], Mapping[str, Any]]:
    indexed: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    for trade in trades:
        identity = entry_identity(trade)
        if identity in indexed:
            raise ValueError(f"duplicate entry identity: {identity!r}")
        indexed[identity] = trade
    return indexed


def trade_metrics(trades: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    gross_profit = sum(float(t.get("pnl", 0.0)) for t in trades if float(t.get("pnl", 0.0)) > 0)
    gross_loss = abs(
        sum(float(t.get("pnl", 0.0)) for t in trades if float(t.get("pnl", 0.0)) < 0)
    )
    pf = gross_profit / gross_loss if gross_loss else (None if gross_profit else 0.0)
    wins = sum(1 for t in trades if float(t.get("pnl", 0.0)) > 0)
    return {
        "trades": len(trades),
        "pf": pf,
        "avg_r": mean(float(t.get("r", 0.0)) for t in trades) if trades else 0.0,
        "profit": sum(float(t.get("pnl", 0.0)) for t in trades),
        "win_rate": wins / len(trades) if trades else 0.0,
    }


def partition_trades(
    control_trades: Sequence[Mapping[str, Any]],
    treatment_trades: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    control = index_unique_trades(control_trades)
    treatment = index_unique_trades(treatment_trades)

    control_ids = set(control)
    treatment_ids = set(treatment)
    matched_ids = sorted(control_ids & treatment_ids)
    control_only_ids = sorted(control_ids - treatment_ids)
    treatment_only_ids = sorted(treatment_ids - control_ids)

    matched = [
        {
            "identity": list(identity),
            "control": control[identity],
            "treatment": treatment[identity],
        }
        for identity in matched_ids
    ]
    control_only = [control[identity] for identity in control_only_ids]
    treatment_only = [treatment[identity] for identity in treatment_only_ids]

    reconciliation = {
        "control_total": len(control_trades),
        "treatment_total": len(treatment_trades),
        "matched_entries": len(matched),
        "control_only_entries": len(control_only),
        "treatment_only_entries": len(treatment_only),
        "control_reconciles": len(control_trades) == len(matched) + len(control_only),
        "treatment_reconciles": len(treatment_trades) == len(matched) + len(treatment_only),
    }
    reconciliation["all_reconcile"] = bool(
        reconciliation["control_reconciles"] and reconciliation["treatment_reconciles"]
    )
    if not reconciliation["all_reconcile"]:
        raise RuntimeError("trade identity partition failed to reconcile")

    return {
        "matched": matched,
        "control_only": control_only,
        "treatment_only": treatment_only,
        "reconciliation": reconciliation,
    }


def classify_treatment_only_association(
    trade: Mapping[str, Any],
    control_simulation: Mapping[str, Any],
    max_open_positions: int,
) -> str:
    entry_date = str(trade["entry_date"])
    instrument = str(trade["instrument"])
    open_positions = list(
        (control_simulation.get("open_positions_before_date") or {}).get(entry_date, [])
    )

    if instrument in open_positions:
        return "control_same_instrument_open"
    if len(open_positions) >= max_open_positions:
        return "control_capacity_full"
    if entry_date in set(control_simulation.get("candidate_dates") or []):
        return "control_candidate_day_state_divergence"
    return "other_state_divergence"


def _association_evidence(
    trades: Sequence[Mapping[str, Any]],
    control_simulation: Mapping[str, Any],
    max_open_positions: int,
) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = {
        category: [] for category in ASSOCIATION_CATEGORIES
    }
    rows: list[dict[str, Any]] = []
    candidate_dates = set(control_simulation.get("candidate_dates") or [])
    open_by_date = control_simulation.get("open_positions_before_date") or {}

    for trade in sorted(trades, key=entry_identity):
        category = classify_treatment_only_association(
            trade, control_simulation, max_open_positions
        )
        grouped[category].append(trade)
        entry_date = str(trade["entry_date"])
        rows.append(
            {
                "identity": list(entry_identity(trade)),
                "category": category,
                "control_open_positions_before_date": sorted(
                    list(open_by_date.get(entry_date, []))
                ),
                "control_candidate_day": entry_date in candidate_dates,
            }
        )

    return {
        "counts": {category: len(grouped[category]) for category in ASSOCIATION_CATEGORIES},
        "metrics": {
            category: trade_metrics(grouped[category])
            for category in ASSOCIATION_CATEGORIES
        },
        "evidence": rows,
    }


def _parse_datetime(value: Any) -> datetime:
    text = str(value)
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    return datetime.fromisoformat(text)


def holding_calendar_days(trade: Mapping[str, Any]) -> int:
    return max(
        0,
        (_parse_datetime(trade["exit_date"]) - _parse_datetime(trade["entry_date"])).days,
    )


def outcome_direction(
    control_trade: Mapping[str, Any],
    treatment_trade: Mapping[str, Any],
    epsilon: float = FLOAT_EPSILON,
) -> str:
    delta = float(treatment_trade.get("r", 0.0)) - float(control_trade.get("r", 0.0))
    if delta > epsilon:
        return "improved"
    if delta < -epsilon:
        return "worsened"
    return "unchanged"


def _sign(value: float, epsilon: float = FLOAT_EPSILON) -> int:
    if value > epsilon:
        return 1
    if value < -epsilon:
        return -1
    return 0


def period_date_sets(dates: Sequence[str]) -> dict[str, set[str] | None]:
    train, validation, test = _split_dates(list(dates))
    return {
        "full": None,
        "train": train,
        "validation": validation,
        "test": test,
    }


def _filter_by_exit(
    trades: Sequence[Mapping[str, Any]],
    period_dates: set[str] | None,
) -> list[Mapping[str, Any]]:
    if period_dates is None:
        return list(trades)
    return [trade for trade in trades if str(trade["exit_date"]) in period_dates]


def _filter_pairs_by_control_exit(
    matched: Sequence[Mapping[str, Any]],
    period_dates: set[str] | None,
) -> list[Mapping[str, Any]]:
    if period_dates is None:
        return list(matched)
    return [
        pair
        for pair in matched
        if str(pair["control"]["exit_date"]) in period_dates
    ]


def _matched_pair_stats(matched: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    directions = {"improved": 0, "worsened": 0, "unchanged": 0}
    delta_profit = 0.0
    delta_r = 0.0
    exit_date_changed = 0
    exit_reason_changed = 0
    control_holding: list[int] = []
    treatment_holding: list[int] = []

    for pair in matched:
        control = pair["control"]
        treatment = pair["treatment"]
        directions[outcome_direction(control, treatment)] += 1
        delta_profit += float(treatment.get("pnl", 0.0)) - float(control.get("pnl", 0.0))
        delta_r += float(treatment.get("r", 0.0)) - float(control.get("r", 0.0))
        exit_date_changed += int(
            str(control.get("exit_date")) != str(treatment.get("exit_date"))
        )
        exit_reason_changed += int(
            str(control.get("exit_reason")) != str(treatment.get("exit_reason"))
        )
        control_holding.append(holding_calendar_days(control))
        treatment_holding.append(holding_calendar_days(treatment))

    return {
        "pairs": len(matched),
        "delta_profit": delta_profit,
        "delta_r": delta_r,
        "avg_delta_r": delta_r / len(matched) if matched else 0.0,
        "outcome_direction_counts": directions,
        "exit_date_changed": exit_date_changed,
        "exit_reason_changed": exit_reason_changed,
        "avg_control_holding_calendar_days": (
            mean(control_holding) if control_holding else 0.0
        ),
        "avg_treatment_holding_calendar_days": (
            mean(treatment_holding) if treatment_holding else 0.0
        ),
    }


def _safe_share(numerator: float, denominator: float) -> float | None:
    if abs(denominator) <= FLOAT_EPSILON:
        return None
    return numerator / denominator


def decompose_period(
    *,
    period_dates: set[str] | None,
    partition: Mapping[str, Any],
    control_simulation: Mapping[str, Any],
    treatment_simulation: Mapping[str, Any],
    max_open_positions: int,
) -> dict[str, Any]:
    control_all = list(control_simulation["trades"])
    treatment_all = list(treatment_simulation["trades"])
    control_period = _filter_by_exit(control_all, period_dates)
    treatment_period = _filter_by_exit(treatment_all, period_dates)

    matched = list(partition["matched"])
    matched_control_legs = _filter_by_exit(
        [pair["control"] for pair in matched], period_dates
    )
    matched_treatment_legs = _filter_by_exit(
        [pair["treatment"] for pair in matched], period_dates
    )
    matched_pairs = _filter_pairs_by_control_exit(matched, period_dates)

    control_only = _filter_by_exit(partition["control_only"], period_dates)
    treatment_only = _filter_by_exit(partition["treatment_only"], period_dates)

    control_metrics = trade_metrics(control_period)
    treatment_metrics = trade_metrics(treatment_period)
    matched_control_metrics = trade_metrics(matched_control_legs)
    matched_treatment_metrics = trade_metrics(matched_treatment_legs)
    control_only_metrics = trade_metrics(control_only)
    treatment_only_metrics = trade_metrics(treatment_only)
    matched_pair_stats = _matched_pair_stats(matched_pairs)
    associations = _association_evidence(
        treatment_only, control_simulation, max_open_positions
    )

    full_profit_delta = treatment_metrics["profit"] - control_metrics["profit"]
    matched_profit_delta = float(matched_pair_stats["delta_profit"])
    path_divergence_net_profit = (
        treatment_only_metrics["profit"] - control_only_metrics["profit"]
    )
    interaction_residual = (
        full_profit_delta - matched_profit_delta - path_divergence_net_profit
    )

    trade_count_delta = treatment_metrics["trades"] - control_metrics["trades"]
    net_path_trade_delta = (
        treatment_only_metrics["trades"] - control_only_metrics["trades"]
    )

    matched_abs_profit_share = _safe_share(
        abs(matched_profit_delta), abs(full_profit_delta)
    )
    path_abs_profit_share = _safe_share(
        abs(path_divergence_net_profit), abs(full_profit_delta)
    )
    matched_signed_profit_share = _safe_share(
        matched_profit_delta, full_profit_delta
    )
    path_signed_profit_share = _safe_share(
        path_divergence_net_profit, full_profit_delta
    )
    path_trade_count_abs_share_net = _safe_share(
        abs(net_path_trade_delta), abs(trade_count_delta)
    )
    treatment_only_vs_abs_trade_delta_share = _safe_share(
        float(treatment_only_metrics["trades"]), abs(float(trade_count_delta))
    )

    control_reconciles = (
        control_metrics["trades"]
        == matched_control_metrics["trades"] + control_only_metrics["trades"]
    )
    treatment_reconciles = (
        treatment_metrics["trades"]
        == matched_treatment_metrics["trades"] + treatment_only_metrics["trades"]
    )

    return {
        "control_metrics": control_metrics,
        "treatment_metrics": treatment_metrics,
        "matched_control_metrics": matched_control_metrics,
        "matched_treatment_metrics": matched_treatment_metrics,
        "control_only_metrics": control_only_metrics,
        "treatment_only_metrics": treatment_only_metrics,
        "matched_pairwise_control_exit_attribution": matched_pair_stats,
        "treatment_only_associations": associations,
        "counts": {
            "control_total": control_metrics["trades"],
            "treatment_total": treatment_metrics["trades"],
            "matched_control_legs": matched_control_metrics["trades"],
            "matched_treatment_legs": matched_treatment_metrics["trades"],
            "matched_pairwise_control_exit": matched_pair_stats["pairs"],
            "control_only": control_only_metrics["trades"],
            "treatment_only": treatment_only_metrics["trades"],
            "treatment_only_share_of_treatment": (
                treatment_only_metrics["trades"] / treatment_metrics["trades"]
                if treatment_metrics["trades"]
                else None
            ),
        },
        "contributions": {
            "full_profit_delta": full_profit_delta,
            "matched_profit_delta": matched_profit_delta,
            "path_divergence_net_profit": path_divergence_net_profit,
            "interaction_residual": interaction_residual,
            "matched_signed_profit_share": matched_signed_profit_share,
            "matched_abs_profit_share": matched_abs_profit_share,
            "path_signed_profit_share": path_signed_profit_share,
            "path_abs_profit_share": path_abs_profit_share,
            "trade_count_delta": trade_count_delta,
            "net_path_trade_delta": net_path_trade_delta,
            "path_trade_count_abs_share_net": path_trade_count_abs_share_net,
            "treatment_only_vs_abs_trade_delta_share": (
                treatment_only_vs_abs_trade_delta_share
            ),
        },
        "reconciliation": {
            "control_reconciles": control_reconciles,
            "treatment_reconciles": treatment_reconciles,
            "all_reconcile": bool(control_reconciles and treatment_reconciles),
            "note": (
                "Arm metrics use each arm's exit_date. Pairwise matched deltas are "
                "attributed by control exit_date, matching the predeclared diagnostic rule."
            ),
        },
    }


def classify_diagnostic(
    periods: Mapping[str, Mapping[str, Any]],
) -> tuple[str, str]:
    full = periods["full"]
    validation = periods["validation"]
    test = periods["test"]

    if not all(
        bool(periods[name]["reconciliation"]["all_reconcile"])
        for name in ("full", "train", "validation", "test")
    ):
        return (
            "INSUFFICIENT_EVIDENCE",
            "Trade partitions did not reconcile under deterministic entry identity.",
        )

    contributions = full["contributions"]
    profit_delta = float(contributions["full_profit_delta"])
    trade_delta = int(contributions["trade_count_delta"])
    if abs(profit_delta) <= FLOAT_EPSILON and trade_delta == 0:
        return (
            "INSUFFICIENT_EVIDENCE",
            "Both Full profit delta and trade-count delta are effectively zero.",
        )

    matched_share = contributions["matched_abs_profit_share"]
    path_share = contributions["path_abs_profit_share"]
    treatment_only_trade_share = contributions[
        "treatment_only_vs_abs_trade_delta_share"
    ]

    matched_share_value = float(matched_share) if matched_share is not None else 0.0
    path_share_value = float(path_share) if path_share is not None else 0.0
    treatment_only_trade_share_value = (
        float(treatment_only_trade_share)
        if treatment_only_trade_share is not None
        else 0.0
    )

    full_path_sign = _sign(float(contributions["path_divergence_net_profit"]))
    split_path_signs = [
        _sign(float(validation["contributions"]["path_divergence_net_profit"])),
        _sign(float(test["contributions"]["path_divergence_net_profit"])),
    ]
    path_split_support = (
        full_path_sign != 0 and any(sign == full_path_sign for sign in split_path_signs)
    )

    full_matched_sign = _sign(float(contributions["matched_profit_delta"]))
    split_matched_signs = [
        _sign(float(validation["contributions"]["matched_profit_delta"])),
        _sign(float(test["contributions"]["matched_profit_delta"])),
    ]
    matched_non_contradictory = (
        full_matched_sign != 0
        and all(sign in {0, full_matched_sign} for sign in split_matched_signs)
    )

    if (
        matched_share is not None
        and path_share is not None
        and matched_share_value >= MATERIAL_MIN_SHARE
        and path_share_value >= MATERIAL_MIN_SHARE
    ):
        return (
            "MIXED",
            "Matched-entry and path-divergence profit contributions are both material.",
        )

    path_dominance = (
        path_share_value >= DOMINANCE_MIN_SHARE
        or treatment_only_trade_share_value >= DOMINANCE_MIN_SHARE
    )
    if (
        path_dominance
        and path_split_support
        and matched_share_value < MATERIAL_MIN_SHARE
    ):
        return (
            "PATH_DIVERGENCE_DOMINANT",
            "Path-divergence contribution meets the predeclared dominance threshold "
            "and has same-direction Validation/Test evidence.",
        )

    if (
        matched_share_value >= DOMINANCE_MIN_SHARE
        and path_share_value < MATERIAL_MIN_SHARE
        and matched_non_contradictory
    ):
        return (
            "MATCHED_OUTCOME_DOMINANT",
            "Matched-entry outcome contribution meets the predeclared dominance threshold "
            "without contradictory Validation/Test direction.",
        )

    if (
        matched_share_value >= MATERIAL_MIN_SHARE
        or path_share_value >= MATERIAL_MIN_SHARE
        or treatment_only_trade_share_value >= MATERIAL_MIN_SHARE
        or any(
            sign != 0 and full_path_sign != 0 and sign != full_path_sign
            for sign in split_path_signs
        )
        or any(
            sign != 0 and full_matched_sign != 0 and sign != full_matched_sign
            for sign in split_matched_signs
        )
    ):
        return (
            "MIXED",
            "Both mechanisms are material, split evidence conflicts, or no single mechanism "
            "satisfies all dominance requirements.",
        )

    return (
        "INSUFFICIENT_EVIDENCE",
        "Attribution denominators or mechanism contributions are too small for dominance.",
    )


def build_diagnostic_settings(base: Settings) -> tuple[Settings, Settings, list[dict[str, Any]]]:
    control, treatment = build_experiment_settings(base)
    diff = settings_diff(control, treatment)
    expected = [
        {
            "field": "adaptive_breakeven_trigger_r",
            "control": CONTROL_BREAKEVEN_TRIGGER_R,
            "treatment": TREATMENT_BREAKEVEN_TRIGGER_R,
        }
    ]
    if diff != expected:
        raise RuntimeError(f"Experiment #4 settings isolation drifted: {diff!r}")
    return control, treatment, diff


def run_diagnostic(
    candles_by_instrument: Mapping[str, Any],
    base_settings: Settings,
    initial_equity: float = 1_000_000.0,
) -> dict[str, Any]:
    before = _production_snapshot()
    control_settings, treatment_settings, diff = build_diagnostic_settings(base_settings)

    control_simulation = _simulate(
        candles_by_instrument,
        control_settings,
        initial_equity=initial_equity,
        spread_pips=DEFAULT_SPREAD_PIPS,
        slippage_pips=DEFAULT_SLIPPAGE_PIPS,
    )
    treatment_simulation = _simulate(
        candles_by_instrument,
        treatment_settings,
        initial_equity=initial_equity,
        spread_pips=DEFAULT_SPREAD_PIPS,
        slippage_pips=DEFAULT_SLIPPAGE_PIPS,
    )

    if list(control_simulation["dates"]) != list(treatment_simulation["dates"]):
        raise RuntimeError("control/treatment common-date identity drifted")

    partition = partition_trades(
        control_simulation["trades"], treatment_simulation["trades"]
    )
    periods: dict[str, Any] = {}
    for name, dates in period_date_sets(control_simulation["dates"]).items():
        periods[name] = decompose_period(
            period_dates=dates,
            partition=partition,
            control_simulation=control_simulation,
            treatment_simulation=treatment_simulation,
            max_open_positions=control_settings.adaptive_max_open_positions,
        )

    if not periods["full"]["reconciliation"]["all_reconcile"]:
        raise RuntimeError("Full decomposition failed to reconcile")

    after = _production_snapshot()
    if after != before:
        raise RuntimeError("production/default Settings or watched environment changed")

    classification, rationale = classify_diagnostic(periods)

    return {
        "diagnostic": "adaptive_v2_breakeven_causal_decomposition_5",
        "research_only": True,
        "experiment_4_classification": "INCONCLUSIVE",
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
        "global_partition": partition["reconciliation"],
        "periods": periods,
        "classification": classification,
        "classification_rationale": rationale,
        "disclosures": {
            "path_association": (
                "Treatment-only labels are observational path associations, not proof "
                "of exclusive causality."
            ),
            "position_sizing": (
                "Entry-identity matching does not freeze position sizing; earlier PnL "
                "can alter later risk dollars."
            ),
            "split_attribution": (
                "Arm metrics use each arm's exit_date; pairwise matched deltas use "
                "control exit_date so each matched identity is counted once."
            ),
            "ohlc_ordering": (
                "D1 OHLC cannot establish intraday high/low ordering on the same candle."
            ),
            "equity_curve_limitation": (
                "Existing adaptive-gate blocked-date equity-curve omission remains; "
                "DD/CAGR are not primary quantities in this diagnostic."
            ),
            "test_holdout": (
                "Test has already been inspected by prior research and is not pristine."
            ),
        },
    }


def dump_json(payload: Mapping[str, Any], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str)
        + "\n",
        encoding="utf-8",
    )


def _fmt_metric(metrics: Mapping[str, Any]) -> str:
    pf = metrics["pf"]
    pf_text = "inf" if pf is None else f"{float(pf):.6f}"
    return (
        f"trades={metrics['trades']} PF={pf_text} avgR={float(metrics['avg_r']):.6f} "
        f"profit={float(metrics['profit']):.2f} win_rate={float(metrics['win_rate']):.4f}"
    )


def write_markdown(payload: Mapping[str, Any], path: str | Path) -> None:
    lines = [
        "# Adaptive v2 Diagnostic #5 — breakeven causal/path decomposition",
        "",
        f"Classification: **{payload['classification']}**",
        "",
        str(payload["classification_rationale"]),
        "",
        "## Settings",
        f"- diff={payload['settings_diff']}",
        "- control=1.0R treatment=0.5R; no additional threshold tested",
        "",
    ]
    for period_name in ("full", "validation", "test"):
        period = payload["periods"][period_name]
        lines.extend(
            [
                f"## {period_name.capitalize()}",
                f"- control: {_fmt_metric(period['control_metrics'])}",
                f"- treatment: {_fmt_metric(period['treatment_metrics'])}",
                f"- matched control: {_fmt_metric(period['matched_control_metrics'])}",
                f"- matched treatment: {_fmt_metric(period['matched_treatment_metrics'])}",
                f"- control-only: {_fmt_metric(period['control_only_metrics'])}",
                f"- treatment-only: {_fmt_metric(period['treatment_only_metrics'])}",
                f"- contributions={period['contributions']}",
                f"- treatment-only associations={period['treatment_only_associations']['counts']}",
                "",
            ]
        )

    lines.extend(["## Disclosures"])
    for key, value in payload["disclosures"].items():
        lines.append(f"- {key}: {value}")

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_report(
    payload: Mapping[str, Any],
    json_path: str | Path,
    markdown_path: str | Path,
) -> None:
    dump_json(payload, json_path)
    write_markdown(payload, markdown_path)


def main(argv: list[str] | None = None) -> int:
    from .fx_research_data import DEFAULT_RESEARCH_DATA_DIR, load_history

    parser = argparse.ArgumentParser(
        description="Research-only Adaptive v2 Diagnostic #5: breakeven causal decomposition"
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
    payload = run_diagnostic(
        history,
        settings,
        initial_equity=settings.paper_initial_balance,
    )
    write_report(payload, args.json_path, args.markdown_path)
    full = payload["periods"]["full"]
    print(
        "BREAKEVEN_CAUSAL_DECOMPOSITION_OK "
        f"classification={payload['classification']} "
        f"control_trades={full['control_metrics']['trades']} "
        f"treatment_trades={full['treatment_metrics']['trades']} "
        f"treatment_only={full['counts']['treatment_only']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
