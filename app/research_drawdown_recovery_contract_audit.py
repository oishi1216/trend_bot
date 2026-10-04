from __future__ import annotations

import argparse
import ast
import json
import math
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

from .config import Settings
from .research_backtest import _metrics
from .research_breakeven_experiment import (
    CONTROL_BREAKEVEN_TRIGGER_R,
    _production_snapshot,
    build_experiment_settings,
)
from .research_candidate_day_divergence_audit import (
    reconstruct_daily_state,
    reconstruct_gate_trace,
    simulate_with_candidate_trace,
)

DEFAULT_JSON_PATH = "data/research/adaptive_v2_drawdown_recovery_contract_audit.json"
DEFAULT_MARKDOWN_PATH = "data/research/adaptive_v2_drawdown_recovery_contract_audit.md"

EPSILON = 1e-9
EXPECTED_TRADES = 133
EXPECTED_PF = 0.8137793529369636
EXPECTED_PROFIT = -37421.88297924094
EXPECTED_MAX_DD = 0.10069367895411063
EXPECTED_BLOCKED_DD_STOP = 55
EXPECTED_DIAGNOSTIC6_TARGETS = 18
EXPECTED_DIAGNOSTIC6_DD = 0.100693678954111

CLASSIFICATIONS = (
    "TERMINAL_LOCK_CONFIRMED",
    "SELF_RECOVERY_OBSERVED",
    "CONTRACT_DIVERGENCE",
    "INCONCLUSIVE",
)


def control_settings(base: Settings) -> Settings:
    control, _treatment = build_experiment_settings(base)
    if not math.isclose(
        float(control.adaptive_breakeven_trigger_r),
        CONTROL_BREAKEVEN_TRIGGER_R,
        rel_tol=0.0,
        abs_tol=EPSILON,
    ):
        raise RuntimeError("Experiment #4 control breakeven trigger drifted")
    return control


def _parse_date(value: str) -> datetime:
    text = str(value)
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    return datetime.fromisoformat(text)


def _first_date(
    dates: Sequence[str],
    predicate,
) -> str | None:
    for date in dates:
        if predicate(str(date)):
            return str(date)
    return None


def control_anchor(
    simulation: Mapping[str, Any],
    *,
    initial_equity: float,
) -> dict[str, Any]:
    dates = list(simulation["dates"])
    start_date = dates[0] if dates else None
    end_date = dates[-1] if dates else None
    curve = list(simulation["equity_curve"])
    metrics = _metrics(
        list(simulation["trades"]),
        curve,
        start_date,
        end_date,
        equity_first=initial_equity,
        equity_last=curve[-1] if curve else initial_equity,
    )
    observed = {
        "trades": int(metrics["trades"]),
        "pf": float(metrics["pf"]),
        "profit": float(metrics["profit"]),
        "max_dd": float(metrics["max_dd"]),
        "blocked_dd_stop": int(simulation["diagnostics"]["blocked_dd_stop"]),
    }
    expected = {
        "trades": EXPECTED_TRADES,
        "pf": EXPECTED_PF,
        "profit": EXPECTED_PROFIT,
        "max_dd": EXPECTED_MAX_DD,
        "blocked_dd_stop": EXPECTED_BLOCKED_DD_STOP,
    }
    matches = (
        observed["trades"] == expected["trades"]
        and observed["blocked_dd_stop"] == expected["blocked_dd_stop"]
        and math.isclose(observed["pf"], expected["pf"], rel_tol=0.0, abs_tol=EPSILON)
        and math.isclose(
            observed["profit"], expected["profit"], rel_tol=0.0, abs_tol=EPSILON
        )
        and math.isclose(
            observed["max_dd"], expected["max_dd"], rel_tol=0.0, abs_tol=EPSILON
        )
    )
    return {"matches": matches, "observed": observed, "expected": expected}


def audit_lifecycle(
    simulation: Mapping[str, Any],
    daily_state: Mapping[str, Mapping[str, Any]],
    gate_trace: Mapping[str, Any],
    *,
    drawdown_stop: float,
) -> dict[str, Any]:
    dates = [str(date) for date in simulation["dates"]]
    if not dates:
        raise RuntimeError("simulation has no common dates")

    crossing = _first_date(
        dates,
        lambda date: float(daily_state[date]["drawdown"]) >= drawdown_stop,
    )
    if crossing is None:
        return {
            "first_drawdown_stop_crossing": None,
            "self_recovery_observed": False,
            "terminal_flat_exists": False,
        }

    gate_by_date = gate_trace["by_date"]
    first_dd_block = _first_date(
        [date for date in dates if date in gate_by_date and date >= crossing],
        lambda date: str(gate_by_date[date]["gate"]["reason"]) == "blocked_dd_stop",
    )

    crossing_state = daily_state[crossing]
    trades = list(simulation["trades"])
    open_at_crossing = [
        trade
        for trade in trades
        if str(trade["entry_date"]) < crossing <= str(trade["exit_date"])
    ]
    crossing_open_exit_dates = sorted(
        {str(trade["exit_date"]) for trade in open_at_crossing}
    )
    last_open_exit = crossing_open_exit_dates[-1] if crossing_open_exit_dates else None

    first_flat_gate = _first_date(
        [date for date in dates if date >= crossing],
        lambda date: len(daily_state[date]["open_at_gate"]) == 0,
    )
    terminal_flat = None
    if first_flat_gate is not None:
        terminal_flat = _first_date(
            [date for date in dates if date >= first_flat_gate],
            lambda date: (
                len(daily_state[date]["open_at_date_start"]) == 0
                and len(daily_state[date]["open_at_gate"]) == 0
            ),
        )

    first_recovery = _first_date(
        [date for date in dates if date > crossing],
        lambda date: float(daily_state[date]["drawdown"]) < drawdown_stop,
    )
    first_entry_after_recovery = None
    if first_recovery is not None:
        later_entries = sorted(
            str(trade["entry_date"])
            for trade in trades
            if str(trade["entry_date"]) >= first_recovery
        )
        first_entry_after_recovery = later_entries[0] if later_entries else None
    self_recovery_observed = (
        first_recovery is not None and first_entry_after_recovery is not None
    )

    result: dict[str, Any] = {
        "first_drawdown_stop_crossing": crossing,
        "first_candidate_blocked_dd_stop": first_dd_block,
        "crossing_open_positions_date_start": list(
            crossing_state["open_at_date_start"]
        ),
        "crossing_open_positions_gate": list(crossing_state["open_at_gate"]),
        "crossing_open_trade_exit_dates": crossing_open_exit_dates,
        "last_open_position_exit_date_after_crossing": last_open_exit,
        "first_flat_at_gate_date": first_flat_gate,
        "terminal_flat_date": terminal_flat,
        "first_later_recovery_below_stop": first_recovery,
        "first_entry_after_recovery": first_entry_after_recovery,
        "self_recovery_observed": self_recovery_observed,
        "terminal_flat_exists": terminal_flat is not None,
        "final_dataset_date": dates[-1],
    }
    if terminal_flat is None:
        return result

    terminal_state = daily_state[terminal_flat]
    high_water = float(terminal_state["high_water"])
    nav = float(terminal_state["current_nav"])
    dd = float(terminal_state["drawdown"])
    recovery_boundary = high_water * (1.0 - drawdown_stop)
    gap = recovery_boundary - nav

    post_dates = [date for date in dates if date >= terminal_flat]
    post_candidate_dates = [
        date for date in post_dates if date in gate_by_date
    ]
    post_reasons = Counter(
        str(gate_by_date[date]["gate"]["reason"])
        for date in post_candidate_dates
    )
    post_entries = [
        trade for trade in trades if str(trade["entry_date"]) >= terminal_flat
    ]

    nav_values = [float(daily_state[date]["current_nav"]) for date in post_dates]
    high_values = [float(daily_state[date]["high_water"]) for date in post_dates]
    dd_values = [float(daily_state[date]["drawdown"]) for date in post_dates]

    nav_constant = max(nav_values) - min(nav_values) <= EPSILON
    high_constant = max(high_values) - min(high_values) <= EPSILON
    dd_constant = max(dd_values) - min(dd_values) <= EPSILON
    all_dd_at_or_above_stop = all(value >= drawdown_stop for value in dd_values)
    all_post_candidates_dd_blocked = all(
        str(gate_by_date[date]["gate"]["reason"]) == "blocked_dd_stop"
        for date in post_candidate_dates
    )

    calendar_days = (
        _parse_date(dates[-1]) - _parse_date(terminal_flat)
    ).days

    result.update(
        {
            "terminal_flat_state": {
                "nav": nav,
                "realized_equity_before_date_exits": float(
                    terminal_state["realized_equity_before_date_exits"]
                ),
                "high_water": high_water,
                "drawdown": dd,
                "recovery_boundary_nav": recovery_boundary,
                "strict_reentry_condition": f"NAV > {recovery_boundary:.12f}",
                "gap_to_recovery_boundary": gap,
                "at_or_below_recovery_boundary": nav <= recovery_boundary + EPSILON,
            },
            "post_terminal": {
                "common_date_observations": len(post_dates),
                "calendar_days_to_dataset_end": calendar_days,
                "candidate_days": len(post_candidate_dates),
                "candidate_gate_reasons": dict(sorted(post_reasons.items())),
                "new_entries": len(post_entries),
                "new_entry_identities": [
                    [
                        str(trade["instrument"]),
                        str(trade["side"]),
                        str(trade["entry_date"]),
                    ]
                    for trade in post_entries
                ],
                "nav_min": min(nav_values),
                "nav_max": max(nav_values),
                "high_water_min": min(high_values),
                "high_water_max": max(high_values),
                "drawdown_min": min(dd_values),
                "drawdown_max": max(dd_values),
                "nav_constant": nav_constant,
                "high_water_constant": high_constant,
                "drawdown_constant": dd_constant,
                "all_drawdown_at_or_above_stop": all_dd_at_or_above_stop,
                "all_candidate_days_blocked_dd_stop": all_post_candidates_dd_blocked,
            },
        }
    )
    return result


def _function_node(tree: ast.AST, name: str) -> ast.FunctionDef:
    matches = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected exactly one function {name!r}, found {len(matches)}")
    return matches[0]


def _call_name(call: ast.Call) -> str:
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    if isinstance(call.func, ast.Name):
        return call.func.id
    return ""


def audit_production_source_contract(repo_root: str | Path) -> dict[str, Any]:
    root = Path(repo_root)
    engine_path = root / "app" / "engine.py"
    if not engine_path.is_file():
        raise RuntimeError(f"engine source not found: {engine_path}")

    engine_text = engine_path.read_text(encoding="utf-8")
    engine_tree = ast.parse(engine_text)
    update_node = _function_node(engine_tree, "_update_drawdown")
    risk_node = _function_node(engine_tree, "adaptive_risk_after_drawdown")

    update_source = ast.get_source_segment(engine_text, update_node) or ""
    risk_source = ast.get_source_segment(engine_text, risk_node) or ""

    update_normalized = " ".join(update_source.split())
    risk_normalized = " ".join(risk_source.split())

    reads_persisted = (
        'get_kv("nav_high_water"' in update_normalized
        or "get_kv('nav_high_water'" in update_normalized
    )
    monotonic_max = "max(current, nav)" in update_normalized
    persists_high_water = (
        'set_kv("nav_high_water", str(high_water))' in update_normalized
        or "set_kv('nav_high_water', str(high_water))" in update_normalized
    )
    drawdown_formula = "1.0 - nav / high_water" in update_normalized
    risk_zeroing = (
        "drawdown >= settings.adaptive_drawdown_stop" in risk_normalized
        and "return 0.0" in risk_normalized
    )

    nav_high_water_calls: list[dict[str, Any]] = []
    writer_calls: list[dict[str, Any]] = []
    reset_named_functions: list[str] = []

    for path in sorted(root.rglob("*.py")):
        if any(part in {".git", ".venv", "__pycache__"} for part in path.parts):
            continue
        text = path.read_text(encoding="utf-8")
        try:
            tree = ast.parse(text)
        except SyntaxError as exc:
            raise RuntimeError(f"cannot parse source during contract audit: {path}: {exc}")
        relative = path.relative_to(root).as_posix()

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                lower = node.name.lower()
                if ("reset" in lower or "recover" in lower or "rebaseline" in lower):
                    segment = ast.get_source_segment(text, node) or ""
                    if "nav_high_water" in segment:
                        reset_named_functions.append(f"{relative}:{node.lineno}:{node.name}")
            if not isinstance(node, ast.Call):
                continue
            contains_key = any(
                isinstance(arg, ast.Constant)
                and arg.value == "nav_high_water"
                for arg in node.args
            )
            if not contains_key:
                continue
            name = _call_name(node)
            row = {
                "path": relative,
                "line": int(getattr(node, "lineno", 0)),
                "call": name,
            }
            nav_high_water_calls.append(row)
            if name not in {"get_kv"}:
                writer_calls.append(row)

    expected_writer = {
        "path": "app/engine.py",
        "line": next(
            (
                int(node.lineno)
                for node in ast.walk(update_node)
                if isinstance(node, ast.Call)
                and _call_name(node) == "set_kv"
                and any(
                    isinstance(arg, ast.Constant)
                    and arg.value == "nav_high_water"
                    for arg in node.args
                )
            ),
            -1,
        ),
        "call": "set_kv",
    }
    single_writer = writer_calls == [expected_writer]
    no_reset_path = len(reset_named_functions) == 0 and single_writer

    source_lines = engine_text.splitlines()
    evidence_lines = [
        {
            "path": "app/engine.py",
            "line": index,
            "text": line.strip(),
        }
        for index, line in enumerate(source_lines, start=1)
        if "nav_high_water" in line
        or "adaptive_drawdown_stop" in line
        or "high_water = max(current, nav)" in line
    ]

    contract_ok = all(
        (
            reads_persisted,
            monotonic_max,
            persists_high_water,
            drawdown_formula,
            risk_zeroing,
            single_writer,
            no_reset_path,
        )
    )
    return {
        "contract_ok": contract_ok,
        "reads_persisted_nav_high_water": reads_persisted,
        "monotonic_high_water_max": monotonic_max,
        "persists_high_water": persists_high_water,
        "drawdown_formula_current_nav_vs_high_water": drawdown_formula,
        "adaptive_risk_zero_at_or_above_stop": risk_zeroing,
        "nav_high_water_calls": nav_high_water_calls,
        "nav_high_water_writer_calls": writer_calls,
        "single_expected_writer": single_writer,
        "reset_named_functions_touching_nav_high_water": reset_named_functions,
        "no_automatic_reset_path_found": no_reset_path,
        "evidence_lines": evidence_lines,
    }


def corroborate_diagnostic6(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {"provided": False, "matches": False}
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    records = list(payload.get("target_records") or [])
    dd_values = [
        float(record["control_selected_gate"]["drawdown"])
        for record in records
    ]
    reasons = [
        str(record["control_selected_gate"]["reason"])
        for record in records
    ]
    matches = (
        len(records) == EXPECTED_DIAGNOSTIC6_TARGETS
        and all(reason == "blocked_dd_stop" for reason in reasons)
        and all(
            math.isclose(
                value,
                EXPECTED_DIAGNOSTIC6_DD,
                rel_tol=0.0,
                abs_tol=EPSILON,
            )
            for value in dd_values
        )
    )
    return {
        "provided": True,
        "matches": matches,
        "target_count": len(records),
        "all_control_selected_blocked_dd_stop": all(
            reason == "blocked_dd_stop" for reason in reasons
        ),
        "control_dd_min": min(dd_values) if dd_values else None,
        "control_dd_max": max(dd_values) if dd_values else None,
        "source_classification": payload.get("classification"),
    }


def classify_contract(
    *,
    anchor_ok: bool,
    lifecycle: Mapping[str, Any],
    production_contract_ok: bool,
) -> tuple[str, str]:
    crossing = lifecycle.get("first_drawdown_stop_crossing")
    if crossing is None:
        return "INCONCLUSIVE", "The exact control never crossed the drawdown stop."

    if lifecycle.get("self_recovery_observed"):
        if production_contract_ok:
            return (
                "SELF_RECOVERY_OBSERVED",
                "The exact control crossed the stop and later recovered below it autonomously.",
            )
        return (
            "CONTRACT_DIVERGENCE",
            "Research self-recovery was observed but production/paper source contract could not be proven equivalent.",
        )

    terminal = lifecycle.get("terminal_flat_state") or {}
    post = lifecycle.get("post_terminal") or {}
    terminal_conditions = all(
        (
            anchor_ok,
            lifecycle.get("terminal_flat_exists") is True,
            terminal.get("at_or_below_recovery_boundary") is True,
            int(post.get("new_entries", -1)) == 0,
            post.get("nav_constant") is True,
            post.get("high_water_constant") is True,
            post.get("drawdown_constant") is True,
            post.get("all_drawdown_at_or_above_stop") is True,
            post.get("all_candidate_days_blocked_dd_stop") is True,
        )
    )

    if terminal_conditions and production_contract_ok:
        return (
            "TERMINAL_LOCK_CONFIRMED",
            "After the DD-stop crossing the control reaches a flat state below the recovery boundary, then NAV/high-water/DD remain constant and every later candidate is blocked; production/paper uses the same monotonic persisted high-water with no automatic reset path.",
        )
    if terminal_conditions and not production_contract_ok:
        return (
            "CONTRACT_DIVERGENCE",
            "Research exhibits a terminal lock but production/paper source contract is not equivalent or cannot prove the same recovery behavior.",
        )
    return (
        "INCONCLUSIVE",
        "The frozen terminal-lock conditions were not all established.",
    )


def run_audit(
    candles_by_instrument: Mapping[str, Any],
    base_settings: Settings,
    *,
    initial_equity: float = 1_000_000.0,
    repo_root: str | Path,
    diagnostic6_json_path: str | Path | None = None,
) -> dict[str, Any]:
    before = _production_snapshot()
    settings = control_settings(base_settings)

    replay = simulate_with_candidate_trace(
        candles_by_instrument,
        settings,
        initial_equity=initial_equity,
    )
    simulation = replay["simulation"]
    state = reconstruct_daily_state(
        candles_by_instrument,
        simulation,
        initial_equity=initial_equity,
    )
    gates = reconstruct_gate_trace(
        replay["trace_by_date"],
        simulation,
        settings,
        state,
    )
    anchor = control_anchor(simulation, initial_equity=initial_equity)
    if not anchor["matches"]:
        raise RuntimeError(f"authoritative control anchor drifted: {anchor!r}")

    lifecycle = audit_lifecycle(
        simulation,
        state,
        gates,
        drawdown_stop=float(settings.adaptive_drawdown_stop),
    )
    source_contract = audit_production_source_contract(repo_root)
    corroboration = corroborate_diagnostic6(diagnostic6_json_path)

    classification, rationale = classify_contract(
        anchor_ok=bool(anchor["matches"]),
        lifecycle=lifecycle,
        production_contract_ok=bool(source_contract["contract_ok"]),
    )

    after = _production_snapshot()
    if before != after:
        raise RuntimeError("production/default Settings or watched environment changed")

    return {
        "diagnostic": "adaptive_v2_drawdown_stop_recovery_contract_7",
        "research_only": True,
        "control_breakeven_trigger_r": CONTROL_BREAKEVEN_TRIGGER_R,
        "drawdown_stop": float(settings.adaptive_drawdown_stop),
        "control_anchor": anchor,
        "candidate_trace_reconciliation": {
            "trace_calls": replay["trace_calls"],
            "candidate_dates": len(simulation["candidate_dates"]),
            "selector_restored": replay["selector_restored"],
        },
        "gate_reconstruction": {
            key: value for key, value in gates.items() if key != "by_date"
        },
        "lifecycle": lifecycle,
        "production_paper_contract": source_contract,
        "diagnostic6_corroboration": corroboration,
        "classification": classification,
        "classification_rationale": rationale,
        "policy_implication": (
            "The current contract behaves as a latched circuit breaker under closed-system conditions and therefore requires an explicit recovery-authority design before any further DD-related experiment."
            if classification == "TERMINAL_LOCK_CONFIRMED"
            else "No DD recovery policy change is authorized by this diagnostic."
        ),
        "non_executed_future_design_candidates": [
            "manual reset authority",
            "governed epoch/rebaseline rules",
            "explicit recovery state machine",
        ],
        "limitations": [
            "Terminal means terminal only under the closed Research system: no external cash flows, manual reset, or operator intervention.",
            "This audit does not evaluate whether relaxing the 10% drawdown stop is beneficial.",
            "The known Research equity-curve blocked-date omission remains pre-existing; lifecycle state is reconstructed from the trade ledger and common dates.",
            "Production/paper equivalence is established by static source-contract inspection, not by executing paper/live code.",
            "Test data has already been inspected and is not pristine.",
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
    lifecycle = payload["lifecycle"]
    terminal = lifecycle.get("terminal_flat_state") or {}
    post = lifecycle.get("post_terminal") or {}
    source = payload["production_paper_contract"]
    lines = [
        "# Adaptive v2 Diagnostic #7 — drawdown-stop recovery contract audit",
        "",
        f"Classification: **{payload['classification']}**",
        "",
        str(payload["classification_rationale"]),
        "",
        "## Control anchor",
        f"- {payload['control_anchor']}",
        "",
        "## Drawdown lifecycle",
        f"- first crossing: {lifecycle.get('first_drawdown_stop_crossing')}",
        f"- first DD-blocked candidate: {lifecycle.get('first_candidate_blocked_dd_stop')}",
        f"- last crossing-open position exit: {lifecycle.get('last_open_position_exit_date_after_crossing')}",
        f"- first flat at gate: {lifecycle.get('first_flat_at_gate_date')}",
        f"- terminal flat date: {lifecycle.get('terminal_flat_date')}",
        f"- terminal state: {terminal}",
        f"- first later recovery below stop: {lifecycle.get('first_later_recovery_below_stop')}",
        f"- post-terminal: {post}",
        "",
        "## Production / paper source contract",
        f"- contract_ok: {source['contract_ok']}",
        f"- monotonic high-water: {source['monotonic_high_water_max']}",
        f"- single expected writer: {source['single_expected_writer']}",
        f"- no automatic reset path: {source['no_automatic_reset_path_found']}",
        f"- adaptive risk zero at/above stop: {source['adaptive_risk_zero_at_or_above_stop']}",
        f"- evidence lines: {source['evidence_lines']}",
        "",
        "## Diagnostic #6 corroboration",
        f"- {payload['diagnostic6_corroboration']}",
        "",
        "## Policy boundary",
        f"- {payload['policy_implication']}",
        "- No reset, rebaseline, cooldown, or DD-threshold change was tested or implemented.",
        "",
        "## Limitations",
    ]
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
        description="Research-only Adaptive v2 Diagnostic #7 drawdown recovery contract audit"
    )
    parser.add_argument("command", choices=["run"])
    parser.add_argument("--data-dir", default=DEFAULT_RESEARCH_DATA_DIR)
    parser.add_argument("--diagnostic6-json")
    parser.add_argument("--json-path", default=DEFAULT_JSON_PATH)
    parser.add_argument("--markdown-path", default=DEFAULT_MARKDOWN_PATH)
    args = parser.parse_args(argv)

    history = load_history(args.data_dir)
    if not history:
        print(f"NO_DATA: no synced history found under {args.data_dir}")
        return 1

    repo_root = Path(__file__).resolve().parents[1]
    settings = Settings.from_env()
    payload = run_audit(
        history,
        settings,
        initial_equity=settings.paper_initial_balance,
        repo_root=repo_root,
        diagnostic6_json_path=args.diagnostic6_json,
    )
    write_report(payload, args.json_path, args.markdown_path)
    lifecycle = payload["lifecycle"]
    print(
        "DRAWDOWN_RECOVERY_CONTRACT_AUDIT_OK "
        f"classification={payload['classification']} "
        f"crossing={lifecycle.get('first_drawdown_stop_crossing')} "
        f"terminal_flat={lifecycle.get('terminal_flat_date')} "
        f"post_candidates={(lifecycle.get('post_terminal') or {}).get('candidate_days')}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
