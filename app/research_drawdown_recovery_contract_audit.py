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


def _normalized_node_source(text: str, node: ast.AST) -> str:
    return " ".join((ast.get_source_segment(text, node) or "").split())


def _call_has_string_arg(call: ast.Call, value: str) -> bool:
    return any(
        isinstance(arg, ast.Constant) and arg.value == value
        for arg in call.args
    )


def _calls_with_key(
    node: ast.AST,
    *,
    call_names: set[str],
    key: str,
) -> list[ast.Call]:
    return [
        call
        for call in ast.walk(node)
        if isinstance(call, ast.Call)
        and _call_name(call) in call_names
        and _call_has_string_arg(call, key)
    ]


def _module_constant(tree: ast.Module, name: str) -> Any:
    values: list[ast.AST] = []
    for statement in tree.body:
        if isinstance(statement, ast.Assign):
            if any(
                isinstance(target, ast.Name) and target.id == name
                for target in statement.targets
            ):
                values.append(statement.value)
        elif (
            isinstance(statement, ast.AnnAssign)
            and isinstance(statement.target, ast.Name)
            and statement.target.id == name
            and statement.value is not None
        ):
            values.append(statement.value)
    if len(values) != 1:
        return None
    try:
        return ast.literal_eval(values[0])
    except (ValueError, TypeError):
        return None


_NAV_HIGH_WATER_KEY = "nav_high_water"
_MUTATOR_KEY_INDEX = {
    "set_kv": 0,
    "delete_kv": 0,
    "_set_kv_conn": 1,
    "_delete_kv_conn": 1,
}
_EXCLUDED_SOURCE_PARTS = {
    ".git",
    ".venv",
    "__pycache__",
    ".pytest_cache",
    "tests",
}
_AUDIT_SOURCE_PATH = "app/research_drawdown_recovery_contract_audit.py"


def _unique_expression_bindings(node: ast.AST) -> dict[str, ast.AST]:
    values: dict[str, list[ast.AST]] = {}

    class BindingCollector(ast.NodeVisitor):
        def visit_FunctionDef(self, child: ast.FunctionDef) -> None:
            if child is node:
                self.generic_visit(child)

        def visit_AsyncFunctionDef(self, child: ast.AsyncFunctionDef) -> None:
            if child is node:
                self.generic_visit(child)

        def visit_ClassDef(self, child: ast.ClassDef) -> None:
            if child is node:
                self.generic_visit(child)

        def visit_Assign(self, child: ast.Assign) -> None:
            for target in child.targets:
                if isinstance(target, ast.Name):
                    values.setdefault(target.id, []).append(child.value)
            self.generic_visit(child.value)

        def visit_AnnAssign(self, child: ast.AnnAssign) -> None:
            if isinstance(child.target, ast.Name) and child.value is not None:
                values.setdefault(child.target.id, []).append(child.value)
                self.generic_visit(child.value)

    BindingCollector().visit(node)
    return {
        name: expressions[0]
        for name, expressions in values.items()
        if len(expressions) == 1
    }


def _assignment_target_names(target: ast.AST) -> set[str]:
    if isinstance(target, ast.Name):
        return {target.id}
    if isinstance(target, (ast.Tuple, ast.List)):
        names: set[str] = set()
        for element in target.elts:
            names.update(_assignment_target_names(element))
        return names
    if isinstance(target, ast.Starred):
        return _assignment_target_names(target.value)
    return set()


def _scope_bound_names(node: ast.AST) -> set[str]:
    names: set[str] = set()

    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        args = node.args
        names.update(arg.arg for arg in args.posonlyargs)
        names.update(arg.arg for arg in args.args)
        names.update(arg.arg for arg in args.kwonlyargs)
        if args.vararg is not None:
            names.add(args.vararg.arg)
        if args.kwarg is not None:
            names.add(args.kwarg.arg)

    class BindingCollector(ast.NodeVisitor):
        def visit_FunctionDef(self, child: ast.FunctionDef) -> None:
            if child is node:
                for statement in child.body:
                    self.visit(statement)
            else:
                names.add(child.name)

        def visit_AsyncFunctionDef(self, child: ast.AsyncFunctionDef) -> None:
            if child is node:
                for statement in child.body:
                    self.visit(statement)
            else:
                names.add(child.name)

        def visit_ClassDef(self, child: ast.ClassDef) -> None:
            if child is node:
                for statement in child.body:
                    self.visit(statement)
            else:
                names.add(child.name)

        def visit_Lambda(self, child: ast.Lambda) -> None:
            return

        def visit_Assign(self, child: ast.Assign) -> None:
            for target in child.targets:
                names.update(_assignment_target_names(target))
            self.visit(child.value)

        def visit_AnnAssign(self, child: ast.AnnAssign) -> None:
            names.update(_assignment_target_names(child.target))
            if child.value is not None:
                self.visit(child.value)

        def visit_AugAssign(self, child: ast.AugAssign) -> None:
            names.update(_assignment_target_names(child.target))
            self.visit(child.value)

        def visit_For(self, child: ast.For) -> None:
            names.update(_assignment_target_names(child.target))
            self.visit(child.iter)
            for statement in child.body:
                self.visit(statement)
            for statement in child.orelse:
                self.visit(statement)

        def visit_AsyncFor(self, child: ast.AsyncFor) -> None:
            self.visit_For(child)

        def visit_With(self, child: ast.With) -> None:
            for item in child.items:
                self.visit(item.context_expr)
                if item.optional_vars is not None:
                    names.update(_assignment_target_names(item.optional_vars))
            for statement in child.body:
                self.visit(statement)

        def visit_AsyncWith(self, child: ast.AsyncWith) -> None:
            self.visit_With(child)

        def visit_ExceptHandler(self, child: ast.ExceptHandler) -> None:
            if child.name:
                names.add(child.name)
            if child.type is not None:
                self.visit(child.type)
            for statement in child.body:
                self.visit(statement)

        def visit_NamedExpr(self, child: ast.NamedExpr) -> None:
            names.update(_assignment_target_names(child.target))
            self.visit(child.value)

        def visit_Import(self, child: ast.Import) -> None:
            for alias in child.names:
                names.add(alias.asname or alias.name.split(".", 1)[0])

        def visit_ImportFrom(self, child: ast.ImportFrom) -> None:
            for alias in child.names:
                if alias.name != "*":
                    names.add(alias.asname or alias.name)

        def visit_ListComp(self, child: ast.ListComp) -> None:
            return

        def visit_SetComp(self, child: ast.SetComp) -> None:
            return

        def visit_DictComp(self, child: ast.DictComp) -> None:
            return

        def visit_GeneratorExp(self, child: ast.GeneratorExp) -> None:
            return

    BindingCollector().visit(node)
    return names


def _imported_repository_bindings(
    tree: ast.Module,
    *,
    repository_bindings: Mapping[str, ast.AST],
) -> dict[str, ast.AST]:
    imported: dict[str, ast.AST] = {}
    for statement in tree.body:
        if not isinstance(statement, ast.ImportFrom):
            continue
        for alias in statement.names:
            if alias.name == "*":
                continue
            expression = repository_bindings.get(alias.name)
            if expression is None:
                continue
            imported[alias.asname or alias.name] = expression
    return imported


def _executable_calls(statements: Sequence[ast.stmt]) -> list[ast.Call]:
    calls: list[ast.Call] = []

    class ExecutableCallCollector(ast.NodeVisitor):
        def visit_Call(self, node: ast.Call) -> None:
            calls.append(node)
            self.generic_visit(node)

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            return

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            return

        def visit_Lambda(self, node: ast.Lambda) -> None:
            return

        def visit_ListComp(self, node: ast.ListComp) -> None:
            return

        def visit_SetComp(self, node: ast.SetComp) -> None:
            return

        def visit_DictComp(self, node: ast.DictComp) -> None:
            return

        def visit_GeneratorExp(self, node: ast.GeneratorExp) -> None:
            return

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            return

    collector = ExecutableCallCollector()
    for statement in statements:
        collector.visit(statement)
    return calls


def _helper_return_expressions(tree: ast.Module) -> dict[str, ast.AST]:
    candidates: dict[str, list[ast.AST]] = {}
    for function in (
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ):
        returns = [
            child.value
            for child in ast.walk(function)
            if isinstance(child, ast.Return) and child.value is not None
        ]
        if len(returns) == 1:
            candidates.setdefault(function.name, []).append(returns[0])
    return {
        name: expressions[0]
        for name, expressions in candidates.items()
        if len(expressions) == 1
    }


def _resolve_key_expression(
    expression: ast.AST,
    *,
    bindings: Mapping[str, ast.AST],
    helper_returns: Mapping[str, ast.AST],
    seen_names: frozenset[str] = frozenset(),
) -> tuple[str, str | None]:
    if isinstance(expression, ast.Constant) and isinstance(expression.value, str):
        return "exact", expression.value

    if isinstance(expression, ast.Name):
        if expression.id in seen_names:
            return "unresolved", None
        bound = bindings.get(expression.id)
        if bound is None:
            return "unresolved", None
        return _resolve_key_expression(
            bound,
            bindings=bindings,
            helper_returns=helper_returns,
            seen_names=seen_names | {expression.id},
        )

    if isinstance(expression, ast.JoinedStr):
        prefix = ""
        suffix = ""
        before_dynamic = True
        after_dynamic_parts: list[str] = []
        has_dynamic = False
        for value in expression.values:
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                if before_dynamic:
                    prefix += value.value
                else:
                    after_dynamic_parts.append(value.value)
            else:
                before_dynamic = False
                has_dynamic = True
                after_dynamic_parts = []
        suffix = "".join(after_dynamic_parts)
        if not has_dynamic:
            return "exact", prefix
        if prefix and not _NAV_HIGH_WATER_KEY.startswith(prefix):
            return "safe_non_target", prefix
        if suffix and not _NAV_HIGH_WATER_KEY.endswith(suffix):
            return "safe_non_target", suffix
        return "unresolved", None

    if isinstance(expression, ast.BinOp) and isinstance(expression.op, ast.Add):
        left_status, left_value = _resolve_key_expression(
            expression.left,
            bindings=bindings,
            helper_returns=helper_returns,
            seen_names=seen_names,
        )
        right_status, right_value = _resolve_key_expression(
            expression.right,
            bindings=bindings,
            helper_returns=helper_returns,
            seen_names=seen_names,
        )
        if left_status == right_status == "exact":
            return "exact", f"{left_value}{right_value}"
        if left_status == "exact" and left_value and not _NAV_HIGH_WATER_KEY.startswith(left_value):
            return "safe_non_target", left_value
        if right_status == "exact" and right_value and not _NAV_HIGH_WATER_KEY.endswith(right_value):
            return "safe_non_target", right_value
        if left_status == "safe_non_target" or right_status == "safe_non_target":
            return "safe_non_target", None
        return "unresolved", None

    if isinstance(expression, ast.Call):
        helper = helper_returns.get(_call_name(expression))
        if helper is not None:
            return _resolve_key_expression(
                helper,
                bindings=bindings,
                helper_returns=helper_returns,
                seen_names=seen_names,
            )

    return "unresolved", None


def _source_file_is_in_scope(root: Path, path: Path) -> bool:
    relative = path.relative_to(root)
    if relative.as_posix() == _AUDIT_SOURCE_PATH:
        return False
    return not any(part in _EXCLUDED_SOURCE_PARTS for part in relative.parts)


def audit_production_source_contract(repo_root: str | Path) -> dict[str, Any]:
    root = Path(repo_root)
    paths = {
        "engine": root / "app" / "engine.py",
        "storage": root / "app" / "storage.py",
        "recovery": root / "app" / "recovery_authority.py",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise RuntimeError(f"production source not found: {missing}")

    texts = {name: path.read_text(encoding="utf-8") for name, path in paths.items()}
    try:
        trees = {name: ast.parse(text) for name, text in texts.items()}
    except SyntaxError as exc:
        raise RuntimeError(f"cannot parse production source during contract audit: {exc}") from exc

    engine_update = _function_node(trees["engine"], "_update_drawdown")
    engine_evaluate = _function_node(trees["engine"], "_evaluate_recovery")
    engine_run = _function_node(trees["engine"], "run_once")
    risk_node = _function_node(trees["engine"], "adaptive_risk_after_drawdown")
    storage_update = _function_node(trees["storage"], "update_nav_high_water_atomic")
    storage_recovery = _function_node(trees["storage"], "_evaluate_recovery_locked")
    latch_reached = _function_node(trees["recovery"], "recovery_latch_reached")
    drawdown_node = _function_node(trees["recovery"], "calculate_drawdown")

    engine_update_source = _normalized_node_source(texts["engine"], engine_update)
    engine_evaluate_source = _normalized_node_source(texts["engine"], engine_evaluate)
    risk_source = _normalized_node_source(texts["engine"], risk_node)
    storage_update_source = _normalized_node_source(texts["storage"], storage_update)
    storage_recovery_source = _normalized_node_source(texts["storage"], storage_recovery)
    latch_source = _normalized_node_source(texts["recovery"], latch_reached)
    drawdown_source = _normalized_node_source(texts["recovery"], drawdown_node)

    engine_high_water_delegation = (
        "self.storage.update_nav_high_water_atomic(nav)" in engine_update_source
    )
    engine_recovery_delegation = (
        "self.storage.evaluate_recovery(" in engine_evaluate_source
    )
    engine_run_recovery_checks = (
        sum(
            1
            for node in ast.walk(engine_run)
            if isinstance(node, ast.Call) and _call_name(node) == "_evaluate_recovery"
        )
        >= 2
    )

    storage_update_reads = bool(
        _calls_with_key(
            storage_update,
            call_names={"_get_kv_conn"},
            key="nav_high_water",
        )
    )
    storage_recovery_reads = bool(
        _calls_with_key(
            storage_recovery,
            call_names={"_get_kv_conn"},
            key="nav_high_water",
        )
    )
    storage_update_writes = bool(
        _calls_with_key(
            storage_update,
            call_names={"_set_kv_conn"},
            key="nav_high_water",
        )
    )
    storage_recovery_writes = bool(
        _calls_with_key(
            storage_recovery,
            call_names={"_set_kv_conn"},
            key="nav_high_water",
        )
    )

    reads_persisted = storage_update_reads and storage_recovery_reads
    monotonic_max = (
        "high_water = max(current, nav_value)" in storage_update_source
        and "state[\"state\"] in {STATE_ACTIVE, STATE_LATCHED_DD_STOP}"
        in storage_recovery_source
        and "nav_value > high_water" in storage_recovery_source
        and "high_water = nav_value" in storage_recovery_source
    )
    persists_high_water = storage_update_writes and storage_recovery_writes
    drawdown_formula = (
        "calculate_drawdown(nav_value, high_water)" in storage_update_source
        and "calculate_drawdown(nav_value, high_water)" in storage_recovery_source
        and "1.0 - nav_value / high_value" in drawdown_source
    )
    risk_zeroing = (
        "drawdown >= settings.adaptive_drawdown_stop" in risk_source
        and "return 0.0" in risk_source
    )

    fixed_recovery_latch_10pct = (
        _module_constant(
            trees["recovery"], "RECOVERY_LATCH_DRAWDOWN_STOP_V1"
        )
        == 0.10
        and _module_constant(
            trees["recovery"], "RECOVERY_LATCH_DRAWDOWN_STOP_TEXT"
        )
        == "0.1"
        and "RECOVERY_LATCH_DRAWDOWN_STOP_V1" in latch_source
        and "nav_value <= recovery_boundary" in latch_source
    )

    persistent_latch_transition = (
        "state[\"state\"] == STATE_ACTIVE" in storage_recovery_source
        and "recovery_latch_reached(nav_value, high_water)"
        in storage_recovery_source
        and "latched = latched_state(" in storage_recovery_source
        and "RECOVERY_STATE_KEY, canonical_json(latched)"
        in storage_recovery_source
        and "STATE_LATCHED_DD_STOP" in storage_recovery_source
    )

    parse_state_lines = [
        int(node.lineno)
        for node in ast.walk(storage_recovery)
        if isinstance(node, ast.Call) and _call_name(node) == "parse_recovery_state"
    ]
    active_state_lines = [
        int(node.lineno)
        for node in ast.walk(storage_recovery)
        if isinstance(node, ast.Call) and _call_name(node) == "active_state"
    ]
    persistent_latch_no_self_clear = (
        len(parse_state_lines) == 1
        and bool(active_state_lines)
        and all(line < parse_state_lines[0] for line in active_state_lines)
    )

    nav_high_water_calls: list[dict[str, Any]] = []
    writer_calls: list[dict[str, Any]] = []
    unresolved_kv_mutators: list[dict[str, Any]] = []
    passthrough_mutators: list[dict[str, Any]] = []
    reset_named_functions: list[str] = []
    processed_mutators: set[tuple[str, int, int, str]] = set()

    source_modules: list[
        tuple[Path, str, str, ast.Module, dict[str, ast.AST], dict[str, ast.AST]]
    ] = []
    repository_constant_candidates: dict[str, set[str]] = {}

    for source_path in sorted(root.rglob("*.py")):
        if not _source_file_is_in_scope(root, source_path):
            continue
        source_text = source_path.read_text(encoding="utf-8")
        try:
            source_tree = ast.parse(source_text)
        except SyntaxError as exc:
            raise RuntimeError(
                f"cannot parse source during contract audit: {source_path}: {exc}"
            ) from exc
        relative = source_path.relative_to(root).as_posix()
        module_bindings = _unique_expression_bindings(source_tree)
        helper_returns = _helper_return_expressions(source_tree)
        source_modules.append(
            (
                source_path,
                relative,
                source_text,
                source_tree,
                module_bindings,
                helper_returns,
            )
        )
        for name, expression in module_bindings.items():
            status, value = _resolve_key_expression(
                expression,
                bindings=module_bindings,
                helper_returns=helper_returns,
            )
            if status == "exact" and value is not None:
                repository_constant_candidates.setdefault(name, set()).add(value)

    repository_bindings = {
        name: ast.Constant(value=next(iter(values)))
        for name, values in repository_constant_candidates.items()
        if len(values) == 1
    }

    allowed_passthrough = {
        ("app/storage.py", "set_kv", "_set_kv_conn", "key"),
    }

    def inspect_mutator(
        *,
        call: ast.Call,
        relative: str,
        function_name: str,
        bindings: Mapping[str, ast.AST],
        helper_returns: Mapping[str, ast.AST],
    ) -> None:
        call_name = _call_name(call)
        key_index = _MUTATOR_KEY_INDEX.get(call_name)
        if key_index is None:
            return
        line = int(getattr(call, "lineno", 0))
        column = int(getattr(call, "col_offset", 0))
        processed_mutators.add((relative, line, column, call_name))
        row = {
            "path": relative,
            "line": line,
            "function": function_name,
            "call": call_name,
        }
        if len(call.args) <= key_index:
            unresolved_kv_mutators.append({**row, "reason": "missing_key_argument"})
            return

        key_expression = call.args[key_index]
        status, resolved_key = _resolve_key_expression(
            key_expression,
            bindings=bindings,
            helper_returns=helper_returns,
        )
        row["key_status"] = status
        row["resolved_key"] = resolved_key

        if status == "exact" and resolved_key == _NAV_HIGH_WATER_KEY:
            nav_high_water_calls.append(row)
            writer_calls.append(row)
            return
        if status in {"exact", "safe_non_target"}:
            return

        passthrough_key = (
            key_expression.id
            if isinstance(key_expression, ast.Name)
            else ""
        )
        passthrough_identity = (
            relative,
            function_name,
            call_name,
            passthrough_key,
        )
        if passthrough_identity in allowed_passthrough:
            passthrough_mutators.append(row)
            return
        unresolved_kv_mutators.append(
            {**row, "reason": "unresolved_key_expression"}
        )

    for (
        source_path,
        relative,
        source_text,
        source_tree,
        module_bindings,
        helper_returns,
    ) in source_modules:
        module_bound_names = _scope_bound_names(source_tree)
        combined_module_bindings = {
            name: expression
            for name, expression in repository_bindings.items()
            if name not in module_bound_names
        }
        combined_module_bindings.update(
            _imported_repository_bindings(
                source_tree,
                repository_bindings=repository_bindings,
            )
        )
        combined_module_bindings.update(module_bindings)

        for function in (
            node
            for node in ast.walk(source_tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        ):
            lower = function.name.lower()
            segment = ast.get_source_segment(source_text, function) or ""
            if (
                any(
                    token in lower
                    for token in ("reset", "rebaseline", "rearm", "probation")
                )
                and (
                    _NAV_HIGH_WATER_KEY in segment
                    or "RECOVERY_STATE_KEY" in segment
                )
            ):
                reset_named_functions.append(
                    f"{relative}:{function.lineno}:{function.name}"
                )

            function_bound_names = _scope_bound_names(function)
            function_bindings = {
                name: expression
                for name, expression in combined_module_bindings.items()
                if name not in function_bound_names
            }
            function_bindings.update(_unique_expression_bindings(function))
            for call in _executable_calls(function.body):
                inspect_mutator(
                    call=call,
                    relative=relative,
                    function_name=function.name,
                    bindings=function_bindings,
                    helper_returns=helper_returns,
                )

        for class_node in (
            node for node in ast.walk(source_tree) if isinstance(node, ast.ClassDef)
        ):
            class_bound_names = _scope_bound_names(class_node)
            class_bindings = {
                name: expression
                for name, expression in combined_module_bindings.items()
                if name not in class_bound_names
            }
            class_bindings.update(_unique_expression_bindings(class_node))
            for call in _executable_calls(class_node.body):
                inspect_mutator(
                    call=call,
                    relative=relative,
                    function_name=f"<class:{class_node.name}>",
                    bindings=class_bindings,
                    helper_returns=helper_returns,
                )

        for call in _executable_calls(source_tree.body):
            inspect_mutator(
                call=call,
                relative=relative,
                function_name="<module>",
                bindings=combined_module_bindings,
                helper_returns=helper_returns,
            )

    for (
        _source_path,
        relative,
        _source_text,
        source_tree,
        _module_bindings,
        _helper_returns,
    ) in source_modules:
        for call in (
            node for node in ast.walk(source_tree) if isinstance(node, ast.Call)
        ):
            call_name = _call_name(call)
            if call_name not in _MUTATOR_KEY_INDEX:
                continue
            identity = (
                relative,
                int(getattr(call, "lineno", 0)),
                int(getattr(call, "col_offset", 0)),
                call_name,
            )
            if identity in processed_mutators:
                continue
            unresolved_kv_mutators.append(
                {
                    "path": relative,
                    "line": identity[1],
                    "function": "<unscanned-lexical-scope>",
                    "call": call_name,
                    "key_status": "unresolved",
                    "resolved_key": None,
                    "reason": "unscanned_lexical_scope",
                }
            )

    allowed_writer_functions = {
        ("app/storage.py", "update_nav_high_water_atomic", "_set_kv_conn"),
        ("app/storage.py", "_evaluate_recovery_locked", "_set_kv_conn"),
    }
    all_high_water_writers_governed = (
        bool(writer_calls)
        and not unresolved_kv_mutators
        and all(
            (row["path"], row["function"], row["call"])
            in allowed_writer_functions
            for row in writer_calls
        )
    )

    future_transition_event_names = (
        "drawdown_recovery_rebaselined",
        "drawdown_recovery_rearmed_probation",
        "drawdown_recovery_relatch",
    )
    combined_r1_source = "\n".join(
        (texts["engine"], texts["storage"], texts["recovery"])
    )
    no_future_recovery_mutation = (
        not reset_named_functions
        and not any(name in combined_r1_source for name in future_transition_event_names)
    )

    no_reset_path = (
        all_high_water_writers_governed
        and persistent_latch_no_self_clear
        and no_future_recovery_mutation
    )

    evidence_lines: list[dict[str, Any]] = []
    evidence_tokens = (
        "nav_high_water",
        "update_nav_high_water_atomic",
        "evaluate_recovery",
        "RECOVERY_LATCH_DRAWDOWN_STOP_V1",
        "recovery_latch_reached",
        "STATE_LATCHED_DD_STOP",
        "adaptive_drawdown_stop",
    )
    for name, path in paths.items():
        for index, line in enumerate(texts[name].splitlines(), start=1):
            if any(token in line for token in evidence_tokens):
                evidence_lines.append(
                    {
                        "path": path.relative_to(root).as_posix(),
                        "line": index,
                        "text": line.strip(),
                    }
                )

    contract_ok = all(
        (
            engine_high_water_delegation,
            engine_recovery_delegation,
            engine_run_recovery_checks,
            reads_persisted,
            monotonic_max,
            persists_high_water,
            drawdown_formula,
            risk_zeroing,
            fixed_recovery_latch_10pct,
            persistent_latch_transition,
            persistent_latch_no_self_clear,
            all_high_water_writers_governed,
            no_future_recovery_mutation,
            no_reset_path,
        )
    )
    return {
        "contract_ok": contract_ok,
        "engine_high_water_delegation": engine_high_water_delegation,
        "engine_recovery_delegation": engine_recovery_delegation,
        "engine_run_recovery_checks": engine_run_recovery_checks,
        "reads_persisted_nav_high_water": reads_persisted,
        "monotonic_high_water_max": monotonic_max,
        "persists_high_water": persists_high_water,
        "drawdown_formula_current_nav_vs_high_water": drawdown_formula,
        "adaptive_risk_zero_at_or_above_stop": risk_zeroing,
        "fixed_recovery_latch_10pct": fixed_recovery_latch_10pct,
        "persistent_latch_transition": persistent_latch_transition,
        "persistent_latch_no_self_clear": persistent_latch_no_self_clear,
        "all_high_water_writers_governed": all_high_water_writers_governed,
        "nav_high_water_calls": nav_high_water_calls,
        "nav_high_water_writer_calls": writer_calls,
        "unresolved_kv_mutators": unresolved_kv_mutators,
        "passthrough_kv_mutators": passthrough_mutators,
        "single_expected_writer": all_high_water_writers_governed,
        "reset_named_functions_touching_nav_high_water": reset_named_functions,
        "no_future_recovery_mutation": no_future_recovery_mutation,
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
        f"- engine high-water delegation: {source['engine_high_water_delegation']}",
        f"- engine Recovery delegation: {source['engine_recovery_delegation']}",
        f"- monotonic high-water: {source['monotonic_high_water_max']}",
        f"- all high-water writers governed: {source['all_high_water_writers_governed']}",
        f"- fixed Recovery latch = 10%: {source['fixed_recovery_latch_10pct']}",
        f"- persistent latch / no self-clear: {source['persistent_latch_no_self_clear']}",
        f"- no future R1 transition mutation: {source['no_future_recovery_mutation']}",
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
