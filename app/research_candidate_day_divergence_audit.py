from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence

from . import research_backtest as research_backtest
from .config import Settings
from .instruments import base_currency, quote_currency
from .research_backtest import DEFAULT_SLIPPAGE_PIPS, DEFAULT_SPREAD_PIPS, _adverse_cost, _split_dates
from .research_breakeven_causal_decomposition import (
    classify_treatment_only_association,
    entry_identity,
    partition_trades,
    trade_metrics,
)
from .research_breakeven_experiment import (
    CONTROL_BREAKEVEN_TRIGGER_R,
    TREATMENT_BREAKEVEN_TRIGGER_R,
    _production_snapshot,
    build_experiment_settings,
    settings_diff,
)

DEFAULT_JSON_PATH = "data/research/adaptive_v2_candidate_day_divergence_audit.json"
DEFAULT_MARKDOWN_PATH = "data/research/adaptive_v2_candidate_day_divergence_audit.md"

EPSILON = 1e-9
DOMINANCE_SHARE = 0.60
MIXED_SHARE = 0.25

BLOCK_REASONS = (
    "blocked_monthly_loss",
    "blocked_dd_stop",
    "blocked_max_positions",
    "blocked_aggregate_risk",
    "blocked_currency_risk",
)

PRIMARY_MECHANISMS = (
    "TARGET_NOT_CONTROL_CANDIDATE",
    "TARGET_SELECTED_CONTROL_BUT_BLOCKED",
    "TARGET_RANKED_BELOW_CONTROL_SELECTION",
    "UNEXPLAINED",
)

SECONDARY_ASSOCIATIONS = (
    "CONTROL_SELECTED_COMPETITOR_OPEN_IN_TREATMENT",
    "CONTROL_SELECTED_COMPETITOR_NOT_IN_TREATMENT_CANDIDATES",
    "CONTROL_SELECTED_COMPETITOR_SHARED",
    "OTHER",
)

CONCLUSIONS = (
    "RANKING_DISPLACEMENT_DOMINANT",
    "TARGET_GATE_BLOCK_DOMINANT",
    "TARGET_CANDIDATE_ABSENCE_DOMINANT",
    "MIXED",
    "INCONCLUSIVE",
)

TARGET_ASSOCIATION = "control_candidate_day_state_divergence"
EXPECTED_TARGET_COUNT = 18
EXPECTED_VALIDATION_TARGETS = 0
EXPECTED_TEST_TARGETS = 18
EXPECTED_TARGET_PROFIT = -10060.514478756802


def _side_from_action(action: str) -> str:
    if action == "enter_long":
        return "long"
    if action == "enter_short":
        return "short"
    raise ValueError(f"candidate action is not an entry action: {action!r}")


def _candidate_key(candidate: Mapping[str, Any]) -> tuple[str, str]:
    return str(candidate["instrument"]), str(candidate["side"])


def _entry_key_for_candidate(candidate: Mapping[str, Any], date: str) -> tuple[str, str, str]:
    instrument, side = _candidate_key(candidate)
    return instrument, side, date


def _serialize_candidates(candidates: Sequence[tuple[str, Any, float, float]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, (instrument, decision, score, nav_snapshot) in enumerate(candidates):
        action = str(decision.action)
        rows.append(
            {
                "original_index": index,
                "instrument": str(instrument),
                "action": action,
                "side": _side_from_action(action),
                "score": float(score),
                "risk_fraction": (
                    None
                    if decision.risk_fraction is None
                    else float(decision.risk_fraction)
                ),
                "regime": str(decision.regime),
                "entry_kind": decision.entry_kind,
                "nav_snapshot": float(nav_snapshot),
            }
        )
    order = sorted(
        range(len(rows)),
        key=lambda index: (-float(rows[index]["score"]), int(rows[index]["original_index"])),
    )
    rank_by_index = {index: rank for rank, index in enumerate(order, start=1)}
    for index, row in enumerate(rows):
        row["rank"] = rank_by_index[index]
    return rows


def simulate_with_candidate_trace(
    candles_by_instrument: Mapping[str, Any],
    settings: Settings,
    *,
    initial_equity: float,
) -> dict[str, Any]:
    original_selector = research_backtest._select_highest_candidate
    calls: list[dict[str, Any]] = []

    def traced_selector(candidates: list[tuple[str, Any, float, float]]):
        selected = original_selector(candidates)
        serialized = _serialize_candidates(candidates)
        selected_index = None
        if selected is not None:
            for index, candidate in enumerate(candidates):
                if candidate is selected:
                    selected_index = index
                    break
            if selected_index is None:
                for index, candidate in enumerate(candidates):
                    if candidate == selected:
                        selected_index = index
                        break
        calls.append(
            {
                "candidates": serialized,
                "selected": (
                    None
                    if selected_index is None
                    else dict(serialized[selected_index])
                ),
            }
        )
        return selected

    research_backtest._select_highest_candidate = traced_selector
    try:
        simulation = research_backtest._simulate(
            candles_by_instrument,
            settings,
            initial_equity=initial_equity,
            spread_pips=DEFAULT_SPREAD_PIPS,
            slippage_pips=DEFAULT_SLIPPAGE_PIPS,
        )
    finally:
        research_backtest._select_highest_candidate = original_selector

    candidate_dates = list(simulation.get("candidate_dates") or [])
    if len(calls) != len(candidate_dates):
        raise RuntimeError(
            "candidate trace call count does not match simulator candidate_dates: "
            f"calls={len(calls)} dates={len(candidate_dates)}"
        )
    if len(candidate_dates) != len(set(candidate_dates)):
        raise RuntimeError("candidate_dates unexpectedly contains duplicates")

    trace_by_date: dict[str, dict[str, Any]] = {}
    for date, call in zip(candidate_dates, calls):
        trace_by_date[str(date)] = {
            "date": str(date),
            "candidates": call["candidates"],
            "selected": call["selected"],
        }

    return {
        "simulation": simulation,
        "trace_by_date": trace_by_date,
        "trace_calls": len(calls),
        "selector_restored": research_backtest._select_highest_candidate is original_selector,
    }


def recover_trade_runtime(
    trade: Mapping[str, Any],
    *,
    spread_pips: float = DEFAULT_SPREAD_PIPS,
    slippage_pips: float = DEFAULT_SLIPPAGE_PIPS,
) -> dict[str, float]:
    pnl = float(trade["pnl"])
    r_value = float(trade["r"])
    if not math.isfinite(pnl) or not math.isfinite(r_value) or abs(r_value) <= EPSILON:
        raise ValueError(
            f"cannot recover positive risk dollars from zero/non-finite R for {entry_identity(trade)!r}"
        )
    risk_dollars = pnl / r_value
    if not math.isfinite(risk_dollars) or risk_dollars <= 0:
        raise ValueError(
            f"recovered risk dollars are not positive for {entry_identity(trade)!r}: {risk_dollars}"
        )

    instrument = str(trade["instrument"])
    side = str(trade["side"])
    entry_price = float(trade["entry_price"])
    initial_stop = float(trade["initial_stop_price"])
    cost = _adverse_cost(instrument, spread_pips, slippage_pips)
    if side == "long":
        entry_close = entry_price - cost
        direction = 1.0
    elif side == "short":
        entry_close = entry_price + cost
        direction = -1.0
    else:
        raise ValueError(f"unsupported trade side: {side!r}")

    stop_distance = abs(entry_close - initial_stop)
    if not math.isfinite(stop_distance) or stop_distance <= EPSILON:
        raise ValueError(
            f"invalid initial stop distance for {entry_identity(trade)!r}: {stop_distance}"
        )
    price_per_risk_unit = risk_dollars / stop_distance
    if not math.isfinite(price_per_risk_unit) or price_per_risk_unit <= 0:
        raise ValueError(
            f"invalid price-per-risk-unit for {entry_identity(trade)!r}"
        )

    exit_price = float(trade["exit_price"])
    reconstructed_pnl = (exit_price - entry_price) * direction * price_per_risk_unit
    if not math.isclose(reconstructed_pnl, pnl, rel_tol=1e-8, abs_tol=1e-6):
        raise ValueError(
            "trade-ledger runtime reconstruction does not reproduce pnl for "
            f"{entry_identity(trade)!r}: reconstructed={reconstructed_pnl} actual={pnl}"
        )

    return {
        "risk_dollars": risk_dollars,
        "price_per_risk_unit": price_per_risk_unit,
        "entry_close": entry_close,
        "stop_distance": stop_distance,
    }


def _trade_open_at_date_start(trade: Mapping[str, Any], date: str) -> bool:
    return str(trade["entry_date"]) < date <= str(trade["exit_date"])


def _trade_open_at_gate(trade: Mapping[str, Any], date: str) -> bool:
    return str(trade["entry_date"]) < date < str(trade["exit_date"])


def _close_lookup(candles_by_instrument: Mapping[str, Any]) -> dict[str, dict[str, float]]:
    lookup: dict[str, dict[str, float]] = {}
    for instrument, frame in candles_by_instrument.items():
        lookup[str(instrument)] = {
            str(row.time): float(row.close)
            for row in frame[["time", "close"]].itertuples(index=False)
        }
    return lookup


def reconstruct_daily_state(
    candles_by_instrument: Mapping[str, Any],
    simulation: Mapping[str, Any],
    *,
    initial_equity: float,
) -> dict[str, dict[str, Any]]:
    trades = list(simulation["trades"])
    runtime_by_identity = {
        entry_identity(trade): recover_trade_runtime(trade) for trade in trades
    }
    closes = _close_lookup(candles_by_instrument)
    pnl_by_exit: dict[str, float] = defaultdict(float)
    for trade in trades:
        pnl_by_exit[str(trade["exit_date"])] += float(trade["pnl"])

    realized_equity = float(initial_equity)
    high_water = float(initial_equity)
    month_start_nav: dict[str, float] = {}
    state_by_date: dict[str, dict[str, Any]] = {}

    for date_value in simulation["dates"]:
        date = str(date_value)
        start_open_trades = [
            trade for trade in trades if _trade_open_at_date_start(trade, date)
        ]
        mtm = 0.0
        for trade in start_open_trades:
            instrument = str(trade["instrument"])
            if date not in closes.get(instrument, {}):
                raise RuntimeError(
                    f"missing close for active trade {entry_identity(trade)!r} on {date}"
                )
            runtime = runtime_by_identity[entry_identity(trade)]
            direction = 1.0 if str(trade["side"]) == "long" else -1.0
            mtm += (
                closes[instrument][date] - float(trade["entry_price"])
            ) * direction * float(runtime["price_per_risk_unit"])

        current_nav = realized_equity + mtm
        high_water = max(high_water, current_nav)
        month = date[:7]
        month_start_nav.setdefault(month, current_nav)
        month_start = month_start_nav[month]
        monthly_loss = (
            max(0.0, 1.0 - current_nav / month_start)
            if month_start > 0
            else 0.0
        )
        drawdown = (
            max(0.0, 1.0 - current_nav / high_water)
            if high_water > 0
            else 0.0
        )
        gate_open_trades = [
            trade for trade in trades if _trade_open_at_gate(trade, date)
        ]
        state_by_date[date] = {
            "realized_equity_before_date_exits": realized_equity,
            "mark_to_market_at_date_start": mtm,
            "current_nav": current_nav,
            "high_water": high_water,
            "month_start_nav": month_start,
            "monthly_loss": monthly_loss,
            "drawdown": drawdown,
            "open_at_date_start": sorted(str(t["instrument"]) for t in start_open_trades),
            "open_at_gate": sorted(str(t["instrument"]) for t in gate_open_trades),
        }
        realized_equity += pnl_by_exit.get(date, 0.0)

    return state_by_date


def evaluate_selected_candidate_gate(
    selected: Mapping[str, Any],
    date: str,
    settings: Settings,
    simulation: Mapping[str, Any],
    daily_state: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    state = dict(daily_state[date])
    trades = list(simulation["trades"])
    gate_open_trades = [
        trade for trade in trades if _trade_open_at_gate(trade, date)
    ]
    runtime_rows = [
        (trade, recover_trade_runtime(trade)) for trade in gate_open_trades
    ]

    risk_fraction_value = selected.get("risk_fraction")
    if risk_fraction_value is None:
        raise RuntimeError(f"selected candidate has no risk_fraction on {date}")
    original_risk_fraction = float(risk_fraction_value)
    adjusted_risk_fraction = original_risk_fraction
    monthly_loss = float(state["monthly_loss"])
    drawdown = float(state["drawdown"])
    nav_snapshot = float(selected["nav_snapshot"])

    reason = "PASS"
    if monthly_loss >= settings.adaptive_monthly_loss_limit:
        reason = "blocked_monthly_loss"
    elif drawdown >= settings.adaptive_drawdown_stop:
        reason = "blocked_dd_stop"
    else:
        if drawdown >= settings.adaptive_drawdown_reduce_2:
            adjusted_risk_fraction = min(adjusted_risk_fraction, 0.0025)
        elif drawdown >= settings.adaptive_drawdown_reduce_1:
            adjusted_risk_fraction *= 0.5

        if len(gate_open_trades) >= settings.adaptive_max_open_positions:
            reason = "blocked_max_positions"
        else:
            aggregate_risk = sum(
                float(runtime["risk_dollars"]) for _, runtime in runtime_rows
            )
            proposed_risk = adjusted_risk_fraction * nav_snapshot
            if (
                nav_snapshot > 0
                and aggregate_risk + proposed_risk
                > nav_snapshot * settings.adaptive_max_aggregate_risk
            ):
                reason = "blocked_aggregate_risk"
            else:
                candidate_currencies = {
                    base_currency(str(selected["instrument"])),
                    quote_currency(str(selected["instrument"])),
                }
                same_currency_risk = sum(
                    float(runtime["risk_dollars"])
                    for trade, runtime in runtime_rows
                    if {
                        base_currency(str(trade["instrument"])),
                        quote_currency(str(trade["instrument"])),
                    }
                    & candidate_currencies
                )
                if (
                    nav_snapshot > 0
                    and same_currency_risk + proposed_risk
                    > nav_snapshot * settings.adaptive_max_single_currency_risk
                ):
                    reason = "blocked_currency_risk"

    aggregate_risk = sum(float(runtime["risk_dollars"]) for _, runtime in runtime_rows)
    candidate_currencies = {
        base_currency(str(selected["instrument"])),
        quote_currency(str(selected["instrument"])),
    }
    same_currency_risk = sum(
        float(runtime["risk_dollars"])
        for trade, runtime in runtime_rows
        if {
            base_currency(str(trade["instrument"])),
            quote_currency(str(trade["instrument"])),
        }
        & candidate_currencies
    )
    return {
        "reason": reason,
        "original_risk_fraction": original_risk_fraction,
        "adjusted_risk_fraction": adjusted_risk_fraction,
        "monthly_loss": monthly_loss,
        "drawdown": drawdown,
        "nav_snapshot": nav_snapshot,
        "open_at_gate": sorted(str(t["instrument"]) for t in gate_open_trades),
        "open_count": len(gate_open_trades),
        "aggregate_risk_dollars": aggregate_risk,
        "same_currency_risk_dollars": same_currency_risk,
        "proposed_risk_dollars": adjusted_risk_fraction * nav_snapshot,
        "aggregate_risk_limit_dollars": (
            nav_snapshot * settings.adaptive_max_aggregate_risk
        ),
        "single_currency_risk_limit_dollars": (
            nav_snapshot * settings.adaptive_max_single_currency_risk
        ),
        "state": state,
    }


def reconstruct_gate_trace(
    trace_by_date: Mapping[str, Mapping[str, Any]],
    simulation: Mapping[str, Any],
    settings: Settings,
    daily_state: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    ledger_identities = {
        entry_identity(trade) for trade in simulation["trades"]
    }
    by_date: dict[str, dict[str, Any]] = {}
    counts = Counter({reason: 0 for reason in BLOCK_REASONS})
    pass_count = 0

    for date in simulation["candidate_dates"]:
        date_text = str(date)
        trace = trace_by_date[date_text]
        selected = trace.get("selected")
        if selected is None:
            raise RuntimeError(f"candidate day has no selected candidate: {date_text}")
        gate = evaluate_selected_candidate_gate(
            selected,
            date_text,
            settings,
            simulation,
            daily_state,
        )
        selected_identity = _entry_key_for_candidate(selected, date_text)
        opened = selected_identity in ledger_identities
        if gate["reason"] == "PASS":
            pass_count += 1
            if not opened:
                raise RuntimeError(
                    "reconstructed PASS candidate did not open: "
                    f"{selected_identity!r}"
                )
        else:
            counts[gate["reason"]] += 1
            if opened:
                raise RuntimeError(
                    "reconstructed blocked candidate appears in ledger: "
                    f"{selected_identity!r} reason={gate['reason']}"
                )
        by_date[date_text] = {
            "selected": dict(selected),
            "selected_identity": list(selected_identity),
            "opened": opened,
            "gate": gate,
        }

    expected = {
        reason: int((simulation.get("diagnostics") or {}).get(reason, 0))
        for reason in BLOCK_REASONS
    }
    observed = {reason: int(counts.get(reason, 0)) for reason in BLOCK_REASONS}
    matches = observed == expected
    if not matches:
        raise RuntimeError(
            "reconstructed gate block counts do not match simulator diagnostics: "
            f"observed={observed} expected={expected}"
        )

    return {
        "by_date": by_date,
        "observed_block_counts": observed,
        "expected_block_counts": expected,
        "block_counts_match": True,
        "pass_count": pass_count,
        "candidate_days": len(by_date),
    }


def _candidate_map(trace: Mapping[str, Any]) -> dict[tuple[str, str], Mapping[str, Any]]:
    result: dict[tuple[str, str], Mapping[str, Any]] = {}
    for candidate in trace.get("candidates") or []:
        key = _candidate_key(candidate)
        if key in result:
            raise RuntimeError(f"duplicate candidate identity on one day: {key!r}")
        result[key] = candidate
    return result


def _split_name(exit_date: str, all_dates: Sequence[str]) -> str:
    train, validation, test = _split_dates(list(all_dates))
    if exit_date in train:
        return "train"
    if exit_date in validation:
        return "validation"
    if exit_date in test:
        return "test"
    raise RuntimeError(f"trade exit date is outside simulator dates: {exit_date}")


def derive_target_trades(
    partition: Mapping[str, Any],
    control_simulation: Mapping[str, Any],
    settings: Settings,
) -> list[Mapping[str, Any]]:
    targets = [
        trade
        for trade in partition["treatment_only"]
        if classify_treatment_only_association(
            trade,
            control_simulation,
            settings.adaptive_max_open_positions,
        )
        == TARGET_ASSOCIATION
    ]
    return sorted(targets, key=entry_identity)


def target_anchor(
    targets: Sequence[Mapping[str, Any]],
    all_dates: Sequence[str],
) -> dict[str, Any]:
    split_counts = Counter(
        _split_name(str(trade["exit_date"]), all_dates) for trade in targets
    )
    metrics = trade_metrics(targets)
    observed = {
        "target_count": len(targets),
        "validation_targets": int(split_counts.get("validation", 0)),
        "test_targets": int(split_counts.get("test", 0)),
        "target_profit": float(metrics["profit"]),
    }
    expected = {
        "target_count": EXPECTED_TARGET_COUNT,
        "validation_targets": EXPECTED_VALIDATION_TARGETS,
        "test_targets": EXPECTED_TEST_TARGETS,
        "target_profit": EXPECTED_TARGET_PROFIT,
    }
    matches = (
        observed["target_count"] == expected["target_count"]
        and observed["validation_targets"] == expected["validation_targets"]
        and observed["test_targets"] == expected["test_targets"]
        and math.isclose(
            observed["target_profit"],
            expected["target_profit"],
            rel_tol=0.0,
            abs_tol=EPSILON,
        )
    )
    return {"matches": matches, "observed": observed, "expected": expected}


def classify_primary_mechanism(
    target: Mapping[str, Any],
    control_trace: Mapping[str, Any],
    control_gate: Mapping[str, Any],
) -> str:
    key = (str(target["instrument"]), str(target["side"]))
    candidates = _candidate_map(control_trace)
    if key not in candidates:
        return "TARGET_NOT_CONTROL_CANDIDATE"
    selected = control_trace.get("selected")
    if selected is None:
        return "UNEXPLAINED"
    if _candidate_key(selected) == key:
        if str(control_gate["reason"]) != "PASS":
            return "TARGET_SELECTED_CONTROL_BUT_BLOCKED"
        return "UNEXPLAINED"
    return "TARGET_RANKED_BELOW_CONTROL_SELECTION"


def classify_secondary_association(
    primary: str,
    control_trace: Mapping[str, Any],
    treatment_trace: Mapping[str, Any],
    treatment_open_before: Sequence[str],
) -> str:
    if primary != "TARGET_RANKED_BELOW_CONTROL_SELECTION":
        return "OTHER"
    selected = control_trace.get("selected")
    if selected is None:
        return "OTHER"
    selected_key = _candidate_key(selected)
    if str(selected["instrument"]) in set(map(str, treatment_open_before)):
        return "CONTROL_SELECTED_COMPETITOR_OPEN_IN_TREATMENT"
    treatment_candidates = _candidate_map(treatment_trace)
    if selected_key not in treatment_candidates:
        return "CONTROL_SELECTED_COMPETITOR_NOT_IN_TREATMENT_CANDIDATES"
    if selected_key in treatment_candidates:
        return "CONTROL_SELECTED_COMPETITOR_SHARED"
    return "OTHER"


def _score_gap(
    target_candidate: Mapping[str, Any] | None,
    selected_candidate: Mapping[str, Any] | None,
) -> float | None:
    if target_candidate is None or selected_candidate is None:
        return None
    return float(selected_candidate["score"]) - float(target_candidate["score"])


def build_target_records(
    targets: Sequence[Mapping[str, Any]],
    all_dates: Sequence[str],
    control_trace_by_date: Mapping[str, Mapping[str, Any]],
    treatment_trace_by_date: Mapping[str, Mapping[str, Any]],
    control_gate_by_date: Mapping[str, Mapping[str, Any]],
    treatment_gate_by_date: Mapping[str, Mapping[str, Any]],
    control_simulation: Mapping[str, Any],
    treatment_simulation: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for target in targets:
        date = str(target["entry_date"])
        if date not in control_trace_by_date or date not in treatment_trace_by_date:
            raise RuntimeError(f"target date missing candidate trace: {date}")
        c_trace = control_trace_by_date[date]
        t_trace = treatment_trace_by_date[date]
        c_candidates = _candidate_map(c_trace)
        t_candidates = _candidate_map(t_trace)
        target_key = (str(target["instrument"]), str(target["side"]))
        c_target = c_candidates.get(target_key)
        t_target = t_candidates.get(target_key)
        if t_target is None:
            raise RuntimeError(
                f"treatment target is absent from treatment candidate trace: {entry_identity(target)!r}"
            )
        t_selected = t_trace.get("selected")
        if t_selected is None or _candidate_key(t_selected) != target_key:
            raise RuntimeError(
                f"treatment target was not selected on its entry date: {entry_identity(target)!r}"
            )
        t_gate = treatment_gate_by_date[date]
        if str(t_gate["gate"]["reason"]) != "PASS":
            raise RuntimeError(
                f"treatment target selected but reconstructed as blocked: {entry_identity(target)!r}"
            )

        c_gate_row = control_gate_by_date[date]
        primary = classify_primary_mechanism(
            target,
            c_trace,
            c_gate_row["gate"],
        )
        treatment_open_before = list(
            (treatment_simulation.get("open_positions_before_date") or {}).get(date, [])
        )
        secondary = classify_secondary_association(
            primary,
            c_trace,
            t_trace,
            treatment_open_before,
        )
        c_keys = set(c_candidates)
        t_keys = set(t_candidates)
        selected_control = c_trace.get("selected")
        selected_treatment = t_trace.get("selected")

        rows.append(
            {
                "identity": list(entry_identity(target)),
                "split": _split_name(str(target["exit_date"]), all_dates),
                "treatment_pnl": float(target["pnl"]),
                "treatment_r": float(target["r"]),
                "primary_mechanism": primary,
                "secondary_association": secondary,
                "control_open_positions_before_date": list(
                    (control_simulation.get("open_positions_before_date") or {}).get(date, [])
                ),
                "treatment_open_positions_before_date": treatment_open_before,
                "target_control_rank": (
                    None if c_target is None else int(c_target["rank"])
                ),
                "target_treatment_rank": int(t_target["rank"]),
                "control_selected": selected_control,
                "treatment_selected": selected_treatment,
                "control_selected_gate": c_gate_row["gate"],
                "treatment_selected_gate": t_gate["gate"],
                "control_selected_minus_target_score": _score_gap(
                    c_target, selected_control
                ),
                "treatment_selected_minus_target_score": _score_gap(
                    t_target, selected_treatment
                ),
                "candidates_only_in_control": [
                    list(key) for key in sorted(c_keys - t_keys)
                ],
                "candidates_only_in_treatment": [
                    list(key) for key in sorted(t_keys - c_keys)
                ],
                "shared_candidates": [
                    list(key) for key in sorted(c_keys & t_keys)
                ],
                "control_candidates": list(c_trace.get("candidates") or []),
                "treatment_candidates": list(t_trace.get("candidates") or []),
                "control_selected_competitor_open_in_treatment": (
                    bool(selected_control)
                    and str(selected_control["instrument"]) in set(treatment_open_before)
                ),
                "treatment_target_instrument_open_in_control": (
                    str(target["instrument"])
                    in set(
                        (control_simulation.get("open_positions_before_date") or {}).get(
                            date, []
                        )
                    )
                ),
            }
        )
    return rows


def classify_conclusion(
    primary_counts: Mapping[str, int],
    total_targets: int,
    *,
    anchor_valid: bool,
    reconstruction_valid: bool,
) -> tuple[str, str]:
    if not anchor_valid or not reconstruction_valid or total_targets <= 0:
        return (
            "INCONCLUSIVE",
            "Target anchor or gate reconstruction did not validate.",
        )
    known = {
        "RANKING_DISPLACEMENT_DOMINANT": int(
            primary_counts.get("TARGET_RANKED_BELOW_CONTROL_SELECTION", 0)
        ),
        "TARGET_GATE_BLOCK_DOMINANT": int(
            primary_counts.get("TARGET_SELECTED_CONTROL_BUT_BLOCKED", 0)
        ),
        "TARGET_CANDIDATE_ABSENCE_DOMINANT": int(
            primary_counts.get("TARGET_NOT_CONTROL_CANDIDATE", 0)
        ),
    }
    material = [
        label for label, count in known.items()
        if count / total_targets >= MIXED_SHARE
    ]
    if len(material) >= 2:
        return (
            "MIXED",
            "At least two known absence mechanisms each explain at least 25% of targets.",
        )
    for label, count in known.items():
        if count / total_targets >= DOMINANCE_SHARE:
            return (
                label,
                f"{count}/{total_targets} targets meet the predeclared >=60% dominance threshold.",
            )
    return (
        "INCONCLUSIVE",
        "No single known mechanism reaches 60% and fewer than two reach 25%.",
    )


def _metrics_for_records(
    records: Sequence[Mapping[str, Any]],
    key: str,
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record[key])].append(
            {
                "pnl": float(record["treatment_pnl"]),
                "r": float(record["treatment_r"]),
            }
        )
    return {
        group: trade_metrics(trades)
        for group, trades in sorted(grouped.items())
    }


def summarize_records(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    primary_counts = Counter(str(record["primary_mechanism"]) for record in records)
    secondary_counts = Counter(
        str(record["secondary_association"]) for record in records
    )
    ranking = [
        record
        for record in records
        if record["primary_mechanism"] == "TARGET_RANKED_BELOW_CONTROL_SELECTION"
    ]
    competitor_open = sum(
        int(bool(record["control_selected_competitor_open_in_treatment"]))
        for record in ranking
    )
    competitor_blocked = sum(
        int(str(record["control_selected_gate"]["reason"]) != "PASS")
        for record in ranking
    )
    gate_reasons = Counter(
        str(record["control_selected_gate"]["reason"]) for record in ranking
    )
    gaps = [
        float(record["control_selected_minus_target_score"])
        for record in ranking
        if record["control_selected_minus_target_score"] is not None
    ]
    return {
        "primary_counts": {
            mechanism: int(primary_counts.get(mechanism, 0))
            for mechanism in PRIMARY_MECHANISMS
        },
        "secondary_counts": {
            association: int(secondary_counts.get(association, 0))
            for association in SECONDARY_ASSOCIATIONS
        },
        "ranking_count": len(ranking),
        "ranking_competitor_open_in_treatment_count": competitor_open,
        "ranking_competitor_open_in_treatment_ratio": (
            competitor_open / len(ranking) if ranking else None
        ),
        "ranking_selected_competitor_blocked_count": competitor_blocked,
        "ranking_selected_competitor_blocked_ratio": (
            competitor_blocked / len(ranking) if ranking else None
        ),
        "ranking_selected_competitor_gate_reasons": dict(sorted(gate_reasons.items())),
        "score_gap": {
            "count": len(gaps),
            "min": min(gaps) if gaps else None,
            "mean": mean(gaps) if gaps else None,
            "max": max(gaps) if gaps else None,
        },
        "metrics_by_primary": _metrics_for_records(records, "primary_mechanism"),
        "metrics_by_secondary": _metrics_for_records(records, "secondary_association"),
    }


def build_audit_settings(base: Settings) -> tuple[Settings, Settings, list[dict[str, Any]]]:
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


def run_audit(
    candles_by_instrument: Mapping[str, Any],
    base_settings: Settings,
    initial_equity: float = 1_000_000.0,
) -> dict[str, Any]:
    before = _production_snapshot()
    control_settings, treatment_settings, diff = build_audit_settings(base_settings)

    control_run = simulate_with_candidate_trace(
        candles_by_instrument,
        control_settings,
        initial_equity=initial_equity,
    )
    treatment_run = simulate_with_candidate_trace(
        candles_by_instrument,
        treatment_settings,
        initial_equity=initial_equity,
    )
    control_sim = control_run["simulation"]
    treatment_sim = treatment_run["simulation"]
    if list(control_sim["dates"]) != list(treatment_sim["dates"]):
        raise RuntimeError("control/treatment common dates diverged")

    control_state = reconstruct_daily_state(
        candles_by_instrument,
        control_sim,
        initial_equity=initial_equity,
    )
    treatment_state = reconstruct_daily_state(
        candles_by_instrument,
        treatment_sim,
        initial_equity=initial_equity,
    )
    control_gate = reconstruct_gate_trace(
        control_run["trace_by_date"],
        control_sim,
        control_settings,
        control_state,
    )
    treatment_gate = reconstruct_gate_trace(
        treatment_run["trace_by_date"],
        treatment_sim,
        treatment_settings,
        treatment_state,
    )

    partition = partition_trades(control_sim["trades"], treatment_sim["trades"])
    targets = derive_target_trades(partition, control_sim, control_settings)
    anchor = target_anchor(targets, control_sim["dates"])
    if not anchor["matches"]:
        raise RuntimeError(f"accepted target anchor drifted: {anchor!r}")

    records = build_target_records(
        targets,
        control_sim["dates"],
        control_run["trace_by_date"],
        treatment_run["trace_by_date"],
        control_gate["by_date"],
        treatment_gate["by_date"],
        control_sim,
        treatment_sim,
    )
    summary = summarize_records(records)
    reconstruction_valid = bool(
        control_gate["block_counts_match"] and treatment_gate["block_counts_match"]
    )
    conclusion, rationale = classify_conclusion(
        summary["primary_counts"],
        len(records),
        anchor_valid=bool(anchor["matches"]),
        reconstruction_valid=reconstruction_valid,
    )

    after = _production_snapshot()
    if before != after:
        raise RuntimeError("production/default Settings or watched environment changed")

    split_summary: dict[str, Any] = {}
    for split in ("train", "validation", "test"):
        split_records = [record for record in records if record["split"] == split]
        split_summary[split] = {
            "targets": len(split_records),
            "metrics": trade_metrics(
                [
                    {"pnl": record["treatment_pnl"], "r": record["treatment_r"]}
                    for record in split_records
                ]
            ),
            "primary_counts": dict(
                Counter(record["primary_mechanism"] for record in split_records)
            ),
        }

    return {
        "diagnostic": "adaptive_v2_candidate_day_divergence_root_cause_6",
        "research_only": True,
        "settings_diff": diff,
        "control_breakeven_trigger_r": CONTROL_BREAKEVEN_TRIGGER_R,
        "treatment_breakeven_trigger_r": TREATMENT_BREAKEVEN_TRIGGER_R,
        "candidate_trace_reconciliation": {
            "control_trace_calls": control_run["trace_calls"],
            "control_candidate_dates": len(control_sim["candidate_dates"]),
            "control_selector_restored": control_run["selector_restored"],
            "treatment_trace_calls": treatment_run["trace_calls"],
            "treatment_candidate_dates": len(treatment_sim["candidate_dates"]),
            "treatment_selector_restored": treatment_run["selector_restored"],
        },
        "gate_reconstruction": {
            "control": {
                key: value
                for key, value in control_gate.items()
                if key != "by_date"
            },
            "treatment": {
                key: value
                for key, value in treatment_gate.items()
                if key != "by_date"
            },
        },
        "target_anchor": anchor,
        "target_records": records,
        "summary": summary,
        "split_summary": split_summary,
        "classification": conclusion,
        "classification_rationale": rationale,
        "limitations": [
            "Candidate-day attribution is path-dependent and observational.",
            "Ranking displacement does not imply a lower-ranked candidate should replace a higher-ranked candidate.",
            "Gate attribution follows current policy's sequential first-block-wins semantics.",
            "Trade-ledger risk reconstruction fails closed when required risk cannot be recovered.",
            "D1 OHLC cannot establish intraday event ordering.",
            "Test has already been inspected and is not pristine.",
            "This audit evaluates no new production policy.",
        ],
    }


def dump_json(payload: Mapping[str, Any], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str)
        + "\n",
        encoding="utf-8",
    )


def write_markdown(payload: Mapping[str, Any], path: str | Path) -> None:
    summary = payload["summary"]
    lines = [
        "# Adaptive v2 Diagnostic #6 — candidate-day divergence root-cause audit",
        "",
        f"Classification: **{payload['classification']}**",
        "",
        str(payload["classification_rationale"]),
        "",
        "## Anchor / reconstruction",
        f"- target anchor: {payload['target_anchor']}",
        f"- control gate reconstruction: {payload['gate_reconstruction']['control']}",
        f"- treatment gate reconstruction: {payload['gate_reconstruction']['treatment']}",
        "",
        "## Mechanisms",
        f"- primary counts: {summary['primary_counts']}",
        f"- secondary counts: {summary['secondary_counts']}",
        f"- ranking competitor open in treatment ratio: {summary['ranking_competitor_open_in_treatment_ratio']}",
        f"- ranking selected competitor blocked ratio: {summary['ranking_selected_competitor_blocked_ratio']}",
        f"- ranking selected competitor gate reasons: {summary['ranking_selected_competitor_gate_reasons']}",
        f"- score gap: {summary['score_gap']}",
        "",
        "## Split summary",
        f"- Validation: {payload['split_summary']['validation']}",
        f"- Test: {payload['split_summary']['test']}",
        "",
        "## Per-target records",
    ]
    for record in payload["target_records"]:
        lines.append(
            "- "
            f"{record['identity']} primary={record['primary_mechanism']} "
            f"secondary={record['secondary_association']} "
            f"control_rank={record['target_control_rank']} "
            f"treatment_rank={record['target_treatment_rank']} "
            f"control_gate={record['control_selected_gate']['reason']} "
            f"pnl={record['treatment_pnl']:.2f} r={record['treatment_r']:.6f}"
        )
    lines.extend(["", "## Limitations"])
    lines.extend(f"- {item}" for item in payload["limitations"])
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
        description="Research-only Adaptive v2 Diagnostic #6 candidate-day divergence audit"
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
    payload = run_audit(
        history,
        settings,
        initial_equity=settings.paper_initial_balance,
    )
    write_report(payload, args.json_path, args.markdown_path)
    print(
        "CANDIDATE_DAY_DIVERGENCE_AUDIT_OK "
        f"classification={payload['classification']} "
        f"targets={payload['target_anchor']['observed']['target_count']} "
        f"ranking={payload['summary']['primary_counts']['TARGET_RANKED_BELOW_CONTROL_SELECTION']} "
        f"gate_block={payload['summary']['primary_counts']['TARGET_SELECTED_CONTROL_BUT_BLOCKED']} "
        f"candidate_absence={payload['summary']['primary_counts']['TARGET_NOT_CONTROL_CANDIDATE']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
