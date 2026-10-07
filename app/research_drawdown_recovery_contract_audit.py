from __future__ import annotations

import argparse
import ast
import json
import math
import re
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
_SQL_CALL_NAMES = {"execute", "executemany", "executescript"}
_EXCLUDED_SOURCE_PARTS = {
    ".git",
    ".venv",
    "__pycache__",
    ".pytest_cache",
    "tests",
}
_AUDIT_SOURCE_PATH = "app/research_drawdown_recovery_contract_audit.py"


def _resolved_mutator_callable_name(
    expression: ast.AST,
    aliases: Mapping[str, str],
) -> str | None:
    if isinstance(expression, ast.Attribute):
        return (
            expression.attr
            if expression.attr in _MUTATOR_KEY_INDEX
            else None
        )
    if isinstance(expression, ast.Name):
        if expression.id in _MUTATOR_KEY_INDEX:
            return expression.id
        return aliases.get(expression.id)
    return None


def _extend_mutator_aliases(
    base: Mapping[str, str],
    unique_bindings: Mapping[str, ast.AST],
) -> dict[str, str]:
    aliases = dict(base)
    pending = dict(unique_bindings)
    while pending:
        progressed = False
        for name, expression in list(pending.items()):
            resolved = _resolved_mutator_callable_name(expression, aliases)
            if resolved is None:
                continue
            aliases[name] = resolved
            del pending[name]
            progressed = True
        if not progressed:
            break
    return aliases


def _static_string_value(expression: ast.AST) -> str | None:
    if isinstance(expression, ast.Constant) and isinstance(expression.value, str):
        return expression.value
    if isinstance(expression, ast.BinOp) and isinstance(expression.op, ast.Add):
        left = _static_string_value(expression.left)
        right = _static_string_value(expression.right)
        if left is not None and right is not None:
            return left + right
    if isinstance(expression, ast.JoinedStr):
        parts: list[str] = []
        for value in expression.values:
            if not isinstance(value, ast.Constant) or not isinstance(value.value, str):
                return None
            parts.append(value.value)
        return "".join(parts)
    return None


def _resolved_sql_callable_name(
    expression: ast.AST,
    aliases: Mapping[str, str],
) -> str | None:
    if isinstance(expression, ast.Attribute):
        return expression.attr if expression.attr in _SQL_CALL_NAMES else None
    if isinstance(expression, ast.Name):
        if expression.id in _SQL_CALL_NAMES:
            return expression.id
        return aliases.get(expression.id)
    if (
        isinstance(expression, ast.Call)
        and _call_name(expression) in {"getattr", "getattribute", "__getattribute__"}
    ):
        method_lookup = (
            isinstance(expression.func, ast.Attribute)
            and _call_name(expression) in {"getattribute", "__getattribute__"}
            and len(expression.args) == 1
        )
        index = 0 if method_lookup else 1
        attribute = (
            expression.args[index]
            if len(expression.args) > index
            else next(
                (kw.value for kw in expression.keywords if kw.arg == "name"),
                None,
            )
        )
        if (
            isinstance(attribute, ast.Constant)
            and isinstance(attribute.value, str)
            and attribute.value in _SQL_CALL_NAMES
        ):
            return attribute.value
    return None


def _extend_sql_aliases(
    base: Mapping[str, str],
    unique_bindings: Mapping[str, ast.AST],
) -> dict[str, str]:
    aliases = dict(base)
    pending = dict(unique_bindings)
    while pending:
        progressed = False
        for name, expression in list(pending.items()):
            resolved = _resolved_sql_callable_name(expression, aliases)
            if resolved is None:
                continue
            aliases[name] = resolved
            del pending[name]
            progressed = True
        if not progressed:
            break
    return aliases


def _normalized_sql(sql_text: str) -> str:
    without_comments = re.sub(
        r"--[^\n]*|/\*.*?\*/",
        " ",
        sql_text,
        flags=re.DOTALL,
    )
    unquoted_identifiers = (
        without_comments.replace('"', "")
        .replace(chr(96), "")
        .replace("[", "")
        .replace("]", "")
    )
    return " ".join(unquoted_identifiers.casefold().split())


def _is_allowed_busy_timeout_pragma(expression: ast.AST) -> bool:
    if not isinstance(expression, ast.JoinedStr) or len(expression.values) != 2:
        return False
    prefix, formatted = expression.values
    if (
        not isinstance(prefix, ast.Constant)
        or prefix.value != "PRAGMA busy_timeout="
        or not isinstance(formatted, ast.FormattedValue)
        or not isinstance(formatted.value, ast.Call)
        or not isinstance(formatted.value.func, ast.Name)
        or formatted.value.func.id != "int"
        or len(formatted.value.args) != 1
        or not isinstance(formatted.value.args[0], ast.Name)
        or formatted.value.args[0].id != "RECOVERY_SQLITE_BUSY_TIMEOUT_MS"
    ):
        return False
    return True


def _sql_mutation_escape_reason(
    call: ast.Call,
    *,
    relative: str,
    scope_name: str,
    resolved_call_name: str | None = None,
) -> str | None:
    call_name = resolved_call_name or _call_name(call)
    if call_name not in _SQL_CALL_NAMES:
        return None
    if not call.args:
        return "dynamic_sql_mutation_surface"

    sql_text = _static_string_value(call.args[0])
    if sql_text is None:
        if (
            call_name == "execute"
            and relative == "app/storage.py"
            and scope_name == "_recovery_conn"
            and _is_allowed_busy_timeout_pragma(call.args[0])
        ):
            return None
        return "dynamic_sql_mutation_surface"

    normalized = _normalized_sql(sql_text)
    mentions_kv = bool(
        re.search(r"(?<![\w])(?:[a-z_][\w]*\.)?kv(?![\w])", normalized)
    )
    mutates_schema_or_rows = bool(
        re.search(
            r"\b(?:insert|replace|update|delete|drop|alter)\b"
            r"|\bcreate\s+trigger\b",
            normalized,
        )
    )
    if not (mentions_kv and mutates_schema_or_rows):
        return None

    allowed_set = bool(
        call_name == "execute"
        and relative == "app/storage.py"
        and scope_name == "_set_kv_conn"
        and re.fullmatch(
            r"insert\s+or\s+replace\s+into\s+kv\s*"
            r"\(\s*key\s*,\s*value\s*\)\s*"
            r"values\s*\(\s*\?\s*,\s*\?\s*\)\s*;?",
            normalized,
        )
    )
    return None if allowed_set else "raw_sql_kv_write"

def _pattern_capture_names(pattern: ast.pattern) -> set[str]:
    names: set[str] = set()
    if isinstance(pattern, ast.MatchAs):
        if pattern.name is not None:
            names.add(pattern.name)
        if pattern.pattern is not None:
            names.update(_pattern_capture_names(pattern.pattern))
    elif isinstance(pattern, ast.MatchStar):
        if pattern.name is not None:
            names.add(pattern.name)
    elif isinstance(pattern, ast.MatchMapping):
        for child in pattern.patterns:
            names.update(_pattern_capture_names(child))
        if pattern.rest is not None:
            names.add(pattern.rest)
    elif isinstance(pattern, ast.MatchSequence):
        for child in pattern.patterns:
            names.update(_pattern_capture_names(child))
    elif isinstance(pattern, ast.MatchClass):
        for child in pattern.patterns:
            names.update(_pattern_capture_names(child))
        for child in pattern.kwd_patterns:
            names.update(_pattern_capture_names(child))
    elif isinstance(pattern, ast.MatchOr):
        for child in pattern.patterns:
            names.update(_pattern_capture_names(child))
    return names


def _scope_binding_facts(node: ast.AST) -> dict[str, Any]:
    counts: dict[str, int] = {}
    candidates: dict[str, list[ast.AST]] = {}
    global_names: set[str] = set()
    nonlocal_names: set[str] = set()

    def bind(name: str, expression: ast.AST | None = None) -> None:
        counts[name] = counts.get(name, 0) + 1
        if expression is not None:
            candidates.setdefault(name, []).append(expression)

    def bind_target(
        target: ast.AST,
        *,
        expression: ast.AST | None = None,
    ) -> None:
        names = _assignment_target_names(target)
        if isinstance(target, ast.Name) and expression is not None:
            bind(target.id, expression)
            return
        for name in names:
            bind(name)

    def visit_function_definition_expressions(
        collector: ast.NodeVisitor,
        child: ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda,
    ) -> None:
        args = child.args
        for value in args.defaults:
            collector.visit(value)
        for value in args.kw_defaults:
            if value is not None:
                collector.visit(value)
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for decorator in child.decorator_list:
                collector.visit(decorator)
            if child.returns is not None:
                collector.visit(child.returns)

    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        args = node.args
        for arg in (*args.posonlyargs, *args.args, *args.kwonlyargs):
            bind(arg.arg)
        if args.vararg is not None:
            bind(args.vararg.arg)
        if args.kwarg is not None:
            bind(args.kwarg.arg)

    class BindingCollector(ast.NodeVisitor):
        def visit_Module(self, child: ast.Module) -> None:
            for statement in child.body:
                self.visit(statement)

        def visit_FunctionDef(self, child: ast.FunctionDef) -> None:
            if child is node:
                for statement in child.body:
                    self.visit(statement)
                return
            bind(child.name)
            visit_function_definition_expressions(self, child)

        def visit_AsyncFunctionDef(self, child: ast.AsyncFunctionDef) -> None:
            if child is node:
                for statement in child.body:
                    self.visit(statement)
                return
            bind(child.name)
            visit_function_definition_expressions(self, child)

        def visit_ClassDef(self, child: ast.ClassDef) -> None:
            if child is node:
                for statement in child.body:
                    self.visit(statement)
                return
            bind(child.name)
            for decorator in child.decorator_list:
                self.visit(decorator)
            for base in child.bases:
                self.visit(base)
            for keyword in child.keywords:
                self.visit(keyword.value)

        def visit_Lambda(self, child: ast.Lambda) -> None:
            visit_function_definition_expressions(self, child)

        def visit_Assign(self, child: ast.Assign) -> None:
            for target in child.targets:
                bind_target(target, expression=child.value)
            self.visit(child.value)

        def visit_AnnAssign(self, child: ast.AnnAssign) -> None:
            bind_target(child.target, expression=child.value)
            if child.value is not None:
                self.visit(child.value)

        def visit_AugAssign(self, child: ast.AugAssign) -> None:
            bind_target(child.target)
            self.visit(child.value)

        def visit_For(self, child: ast.For) -> None:
            bind_target(child.target)
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
                    bind_target(item.optional_vars)
            for statement in child.body:
                self.visit(statement)

        def visit_AsyncWith(self, child: ast.AsyncWith) -> None:
            self.visit_With(child)

        def visit_ExceptHandler(self, child: ast.ExceptHandler) -> None:
            if child.name:
                bind(child.name)
            if child.type is not None:
                self.visit(child.type)
            for statement in child.body:
                self.visit(statement)

        def visit_NamedExpr(self, child: ast.NamedExpr) -> None:
            bind_target(child.target)
            self.visit(child.value)

        def visit_Delete(self, child: ast.Delete) -> None:
            for target in child.targets:
                bind_target(target)

        def visit_Match(self, child: ast.Match) -> None:
            self.visit(child.subject)
            for case in child.cases:
                for name in _pattern_capture_names(case.pattern):
                    bind(name)
                if case.guard is not None:
                    self.visit(case.guard)
                for statement in case.body:
                    self.visit(statement)

        def visit_Import(self, child: ast.Import) -> None:
            for alias in child.names:
                bind(alias.asname or alias.name.split(".", 1)[0])

        def visit_ImportFrom(self, child: ast.ImportFrom) -> None:
            for alias in child.names:
                if alias.name != "*":
                    bind(alias.asname or alias.name)

        def visit_Global(self, child: ast.Global) -> None:
            for name in child.names:
                global_names.add(name)
                bind(name)

        def visit_Nonlocal(self, child: ast.Nonlocal) -> None:
            for name in child.names:
                nonlocal_names.add(name)
                bind(name)

        def _visit_comprehension(
            self,
            generators: list[ast.comprehension],
            values: list[ast.AST],
        ) -> None:
            # Comprehension iteration variables live in the comprehension's
            # implicit scope. Named expressions inside the expressions bind
            # to the containing scope, so visit expressions but not targets.
            for generator in generators:
                self.visit(generator.iter)
                for condition in generator.ifs:
                    self.visit(condition)
            for value in values:
                self.visit(value)

        def visit_ListComp(self, child: ast.ListComp) -> None:
            self._visit_comprehension(child.generators, [child.elt])

        def visit_SetComp(self, child: ast.SetComp) -> None:
            self._visit_comprehension(child.generators, [child.elt])

        def visit_DictComp(self, child: ast.DictComp) -> None:
            self._visit_comprehension(
                child.generators,
                [child.key, child.value],
            )

        def visit_GeneratorExp(self, child: ast.GeneratorExp) -> None:
            self._visit_comprehension(child.generators, [child.elt])

    BindingCollector().visit(node)
    unique_bindings = {
        name: expressions[0]
        for name, expressions in candidates.items()
        if counts.get(name) == 1
        and len(expressions) == 1
        and name not in global_names
        and name not in nonlocal_names
    }
    return {
        "bound_names": set(counts),
        "binding_counts": dict(counts),
        "unique_bindings": unique_bindings,
        "global_names": global_names,
        "nonlocal_names": nonlocal_names,
    }


def _unique_expression_bindings(node: ast.AST) -> dict[str, ast.AST]:
    return dict(_scope_binding_facts(node)["unique_bindings"])


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
    return set(_scope_binding_facts(node)["bound_names"])


def _module_name_from_relative(relative: str) -> str:
    module = relative[:-3].replace("/", ".") if relative.endswith(".py") else relative.replace("/", ".")
    if module.endswith(".__init__"):
        module = module[: -len(".__init__")]
    return module


def _import_source_module(
    current_module: str,
    statement: ast.ImportFrom,
) -> str | None:
    if statement.level == 0:
        return statement.module
    package_parts = current_module.split(".")[:-1]
    up = statement.level - 1
    if up > len(package_parts):
        return None
    base_parts = package_parts[: len(package_parts) - up]
    if statement.module:
        base_parts.extend(statement.module.split("."))
    return ".".join(base_parts) if base_parts else None


def _imported_repository_bindings(
    tree: ast.Module,
    *,
    current_module: str,
    module_bindings_by_name: Mapping[str, Mapping[str, ast.AST]],
) -> tuple[dict[str, ast.AST], bool]:
    imported: dict[str, ast.AST] = {}
    has_wildcard_import = False
    for statement in tree.body:
        if not isinstance(statement, ast.ImportFrom):
            continue
        source_module = _import_source_module(current_module, statement)
        for alias in statement.names:
            if alias.name == "*":
                has_wildcard_import = True
                continue
            if source_module is None:
                continue
            source_bindings = module_bindings_by_name.get(source_module)
            if source_bindings is None:
                continue
            expression = source_bindings.get(alias.name)
            if expression is None:
                continue
            imported[alias.asname or alias.name] = expression
    return imported, has_wildcard_import


def _global_declared_names(tree: ast.Module) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Global):
            names.update(node.names)
    return names


def _descendant_nonlocal_names(node: ast.AST) -> set[str]:
    names: set[str] = set()
    for child in ast.walk(node):
        if child is node:
            continue
        if isinstance(child, ast.Nonlocal):
            names.update(child.names)
    return names


def _freeze_bindings(
    bindings: Mapping[str, ast.AST],
    *,
    helper_returns: Mapping[str, ast.AST],
) -> dict[str, ast.AST]:
    frozen: dict[str, ast.AST] = {}
    for name, expression in bindings.items():
        status, value = _resolve_key_expression(
            expression,
            bindings=bindings,
            helper_returns=helper_returns,
        )
        if status == "exact" and value is not None:
            frozen[name] = ast.Constant(value=value)
        elif status == "safe_non_target":
            frozen[name] = ast.Constant(
                value=f"__audit_safe_non_target__:{name}"
            )
    return frozen


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


def _sql_callable_escape_rows(
    statements: Sequence[ast.stmt],
    *,
    aliases: Mapping[str, str],
    relative: str,
    scope_name: str,
) -> list[dict[str, Any]]:
    references: list[tuple[ast.AST, ast.AST | None]] = []

    class Collector(ast.NodeVisitor):
        def __init__(self) -> None:
            self.parents: list[ast.AST] = []

        def visit(self, node: ast.AST) -> Any:
            parent = self.parents[-1] if self.parents else None
            if isinstance(node, (ast.Attribute, ast.Call)):
                references.append((node, parent))
            self.parents.append(node)
            try:
                return super().visit(node)
            finally:
                self.parents.pop()

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            return

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            return

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            return

    collector = Collector()
    for statement in statements:
        collector.visit(statement)

    rows: list[dict[str, Any]] = []
    seen: set[tuple[int, int]] = set()

    def add(node: ast.AST) -> None:
        identity = (
            int(getattr(node, "lineno", 0)),
            int(getattr(node, "col_offset", 0)),
        )
        if identity in seen:
            return
        seen.add(identity)
        rows.append(
            {
                "path": relative,
                "line": identity[0],
                "function": scope_name,
                "call": "sql_callable",
                "key_status": "unresolved",
                "resolved_key": None,
                "reason": "sql_callable_escape",
            }
        )

    for node, parent in references:
        resolved = _resolved_sql_callable_name(node, aliases)
        if resolved is None:
            if (
                isinstance(node, ast.Call)
                and _call_name(node) == "partial"
                and node.args
                and _resolved_sql_callable_name(node.args[0], aliases)
                is not None
            ):
                add(node)
            continue
        if isinstance(parent, ast.Call) and parent.func is node:
            continue
        if (
            isinstance(parent, ast.Assign)
            and parent.value is node
            and len(parent.targets) == 1
            and isinstance(parent.targets[0], ast.Name)
            and aliases.get(parent.targets[0].id) == resolved
        ):
            continue
        if (
            isinstance(parent, ast.AnnAssign)
            and parent.value is node
            and isinstance(parent.target, ast.Name)
            and aliases.get(parent.target.id) == resolved
        ):
            continue
        add(node)
    return rows


def _mutator_callable_escape_rows(
    statements: Sequence[ast.stmt],
    *,
    aliases: Mapping[str, str],
    binding_counts: Mapping[str, int],
    relative: str,
    scope_name: str,
    allow_local_aliases: bool = False,
    bindings: Mapping[str, ast.AST] | None = None,
) -> list[dict[str, Any]]:
    references: list[
        tuple[ast.AST, ast.AST | None, ast.AST | None]
    ] = []
    imports: list[ast.ImportFrom] = []

    class Collector(ast.NodeVisitor):
        def __init__(self) -> None:
            self.parents: list[ast.AST] = []

        def visit(self, node: ast.AST) -> Any:
            parent = self.parents[-1] if self.parents else None
            grandparent = self.parents[-2] if len(self.parents) >= 2 else None
            if isinstance(node, (ast.Name, ast.Attribute, ast.Call, ast.Subscript)):
                references.append((node, parent, grandparent))
            if isinstance(node, ast.ImportFrom):
                imports.append(node)
            self.parents.append(node)
            try:
                return super().visit(node)
            finally:
                self.parents.pop()

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            for decorator in node.decorator_list:
                self.visit(decorator)
            for value in node.args.defaults:
                self.visit(value)
            for value in node.args.kw_defaults:
                if value is not None:
                    self.visit(value)
            if node.returns is not None:
                self.visit(node.returns)

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            self.visit_FunctionDef(node)

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            for decorator in node.decorator_list:
                self.visit(decorator)
            for base in node.bases:
                self.visit(base)
            for keyword in node.keywords:
                self.visit(keyword.value)

    collector = Collector()
    for statement in statements:
        collector.visit(statement)

    rows: list[dict[str, Any]] = []
    seen: set[tuple[int, int, str]] = set()

    def add(node: ast.AST, call_name: str, reason: str) -> None:
        line = int(getattr(node, "lineno", 0))
        column = int(getattr(node, "col_offset", 0))
        identity = (line, column, reason)
        if identity in seen:
            return
        seen.add(identity)
        rows.append(
            {
                "path": relative,
                "line": line,
                "function": scope_name,
                "call": call_name,
                "key_status": "unresolved",
                "resolved_key": None,
                "reason": reason,
            }
        )

    operator_lookup_names = {"attrgetter", "methodcaller"}
    for statement in imports:
        for alias in statement.names:
            if alias.name in _MUTATOR_KEY_INDEX:
                add(
                    statement,
                    alias.asname or alias.name,
                    "mutator_callable_import_escape",
                )
            if (
                statement.module == "operator"
                and alias.name in {"attrgetter", "methodcaller"}
            ):
                operator_lookup_names.add(alias.asname or alias.name)

    invoked_names = {
        call.func.id
        for statement in statements
        for call in ast.walk(statement)
        if isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
    }
    alias_sources: dict[str, str] = {}
    invoked_attributes = {
        (ast.unparse(call.func.value), call.func.attr)
        for statement in statements
        for call in ast.walk(statement)
        if isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
    }
    for statement in statements:
        for node in ast.walk(statement):
            if (
                isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and isinstance(node.value, ast.Name)
            ):
                alias_sources[node.targets[0].id] = node.value.id
            elif (
                isinstance(node, ast.AnnAssign)
                and isinstance(node.target, ast.Name)
                and isinstance(node.value, ast.Name)
            ):
                alias_sources[node.target.id] = node.value.id
    changed = True
    while changed:
        changed = False
        for target, source_name in alias_sources.items():
            if target in invoked_names and source_name not in invoked_names:
                invoked_names.add(source_name)
                changed = True

    def assigned_targets(
        parent: ast.AST | None,
    ) -> tuple[set[str], set[tuple[str, str]]]:
        names: set[str] = set()
        attrs: set[tuple[str, str]] = set()
        targets: list[ast.AST] = []
        if isinstance(parent, ast.Assign):
            targets.extend(parent.targets)
        elif isinstance(parent, ast.AnnAssign):
            targets.append(parent.target)
        for target in targets:
            names.update(_assignment_target_names(target))
            if isinstance(target, ast.Attribute):
                attrs.add((ast.unparse(target.value), target.attr))
        return names, attrs

    def dict_lookup_surface(expression: ast.AST) -> bool:
        return (
            isinstance(expression, ast.Attribute)
            and expression.attr == "__dict__"
        ) or (
            isinstance(expression, ast.Call)
            and _call_name(expression) in {"vars", "globals"}
        )

    for node, parent, grandparent in references:
        if isinstance(node, ast.Call):
            call_name = _call_name(node)
            if (
                isinstance(parent, ast.Call)
                and parent.func is node
                and node.args
                and call_name not in _MUTATOR_KEY_INDEX
            ):
                status, key = _resolve_key_expression(
                    node.args[0],
                    bindings=bindings or {},
                    helper_returns={},
                )
                if status == "exact" and key in _MUTATOR_KEY_INDEX:
                    add(
                        node,
                        call_name or "<callable-factory>",
                        "mutator_dynamic_attribute_escape",
                    )
                    continue
            effective_lookup_name = (
                node.func.id
                if isinstance(node.func, ast.Name)
                and node.func.id in operator_lookup_names
                else call_name
            )
            if effective_lookup_name in operator_lookup_names:
                attribute = node.args[0] if node.args else None
                status, key = _resolve_key_expression(
                    attribute, bindings=bindings or {}, helper_returns={}
                )
                if status != "exact" or key in _MUTATOR_KEY_INDEX:
                    add(node, effective_lookup_name, "mutator_dynamic_attribute_escape")
                continue

            if (
                call_name == "get"
                and isinstance(node.func, ast.Attribute)
                and dict_lookup_surface(node.func.value)
            ):
                attribute = node.args[0] if node.args else None
                status, key = _resolve_key_expression(
                    attribute, bindings=bindings or {}, helper_returns={}
                )
                if status != "exact" or key in _MUTATOR_KEY_INDEX:
                    add(node, "dict.get", "mutator_dynamic_attribute_escape")
                continue

            if call_name in {"getattr", "getattribute", "__getattribute__"}:
                method_lookup = (
                    isinstance(node.func, ast.Attribute)
                    and call_name in {"getattribute", "__getattribute__"}
                    and len(node.args) == 1
                )
                index = 0 if method_lookup else 1
                attribute = node.args[index] if len(node.args) > index else next(
                    (kw.value for kw in node.keywords if kw.arg == "name"), None
                )
                status, key = _resolve_key_expression(
                    attribute, bindings=bindings or {}, helper_returns={}
                )
                assigned_names, assigned_attrs = assigned_targets(parent)
                callable_use = (
                    isinstance(parent, ast.Call) and parent.func is node
                ) or bool(assigned_names & invoked_names) or bool(
                    assigned_attrs & invoked_attributes
                )
                no_default_dynamic_lookup = method_lookup or (
                    len(node.args) <= 2
                    and not any(kw.arg == "default" for kw in node.keywords)
                )
                escapes_scope = (
                    isinstance(parent, ast.Return)
                    or (
                        isinstance(parent, ast.Call)
                        and parent.func is not node
                    )
                    or (
                        isinstance(parent, ast.keyword)
                        and isinstance(grandparent, ast.Call)
                    )
                )
                if (status == "exact" and key in _MUTATOR_KEY_INDEX) or (
                    status != "exact"
                    and (
                        callable_use
                        or (no_default_dynamic_lookup and escapes_scope)
                    )
                ):
                    add(node, call_name, "mutator_dynamic_attribute_escape")
            continue

        if isinstance(node, ast.Subscript) and dict_lookup_surface(node.value):
            status, key = _resolve_key_expression(
                node.slice, bindings=bindings or {}, helper_returns={}
            )
            callable_use = isinstance(parent, ast.Call) and parent.func is node
            if (status == "exact" and key in _MUTATOR_KEY_INDEX) or (
                status != "exact" and callable_use
            ):
                add(node, "__dict__/vars", "mutator_dynamic_attribute_escape")
            continue

        if isinstance(node, ast.Name) and not isinstance(node.ctx, ast.Load):
            continue
        resolved = _resolved_mutator_callable_name(node, aliases)
        if resolved is None:
            continue
        if isinstance(parent, ast.Call) and parent.func is node:
            continue
        if (
            allow_local_aliases
            and isinstance(parent, ast.Assign)
            and parent.value is node
            and len(parent.targets) == 1
            and isinstance(parent.targets[0], ast.Name)
            and binding_counts.get(parent.targets[0].id) == 1
        ):
            continue
        if (
            allow_local_aliases
            and isinstance(parent, ast.AnnAssign)
            and parent.value is node
            and isinstance(parent.target, ast.Name)
            and binding_counts.get(parent.target.id) == 1
        ):
            continue
        add(node, resolved, "mutator_callable_escape")

    return rows

def _nested_scope_nodes(statements: Sequence[ast.stmt]) -> list[ast.AST]:
    scopes: list[ast.AST] = []

    class NestedScopeCollector(ast.NodeVisitor):
        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            scopes.append(node)

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            scopes.append(node)

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            scopes.append(node)

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

    collector = NestedScopeCollector()
    for statement in statements:
        collector.visit(statement)
    return scopes


def _attribute_mutated_names(tree: ast.AST) -> set[str]:
    names: set[str] = set()

    def collect_target(target: ast.AST) -> None:
        if isinstance(target, ast.Attribute):
            names.add(target.attr)
        elif isinstance(target, ast.Subscript):
            receiver_dict = (
                isinstance(target.value, ast.Attribute)
                and target.value.attr == "__dict__"
            )
            receiver_vars = (
                isinstance(target.value, ast.Call)
                and _call_name(target.value) in {"vars", "globals"}
            )
            if receiver_dict or receiver_vars:
                if (
                    isinstance(target.slice, ast.Constant)
                    and isinstance(target.slice.value, str)
                ):
                    names.add(target.slice.value)
                else:
                    names.add("*")
        elif isinstance(target, (ast.Tuple, ast.List)):
            for element in target.elts:
                collect_target(element)
        elif isinstance(target, ast.Starred):
            collect_target(target.value)

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                collect_target(target)
        elif isinstance(node, ast.AnnAssign):
            collect_target(node.target)
        elif isinstance(node, ast.AugAssign):
            collect_target(node.target)
        elif isinstance(node, ast.Delete):
            for target in node.targets:
                collect_target(target)
        elif (
            isinstance(node, ast.Call)
            and _call_name(node) == "setattr"
            and len(node.args) >= 2
        ):
            if (
                isinstance(node.args[1], ast.Constant)
                and isinstance(node.args[1].value, str)
            ):
                names.add(node.args[1].value)
            else:
                names.add("*")
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "__setattr__"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id in {"object", "type"}
            and len(node.args) >= 2
        ):
            if (
                isinstance(node.args[1], ast.Constant)
                and isinstance(node.args[1].value, str)
            ):
                names.add(node.args[1].value)
            else:
                names.add("*")
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in {"update", "setdefault"}
        ):
            mapping = node.func.value
            receiver_dict = (
                isinstance(mapping, ast.Attribute)
                and mapping.attr == "__dict__"
            )
            receiver_vars = (
                isinstance(mapping, ast.Call)
                and _call_name(mapping) in {"vars", "globals"}
            )
            if receiver_dict or receiver_vars:
                if node.func.attr == "setdefault" and node.args:
                    key = node.args[0]
                    if isinstance(key, ast.Constant) and isinstance(key.value, str):
                        names.add(key.value)
                    else:
                        names.add("*")
                elif node.args and isinstance(node.args[0], ast.Dict):
                    for key in node.args[0].keys:
                        if isinstance(key, ast.Constant) and isinstance(key.value, str):
                            names.add(key.value)
                        else:
                            names.add("*")
                else:
                    names.add("*")
    return names


def _helper_return_expressions(
    tree: ast.Module,
    *,
    repository_base_names: set[str] | frozenset[str] = frozenset(),
    repository_attribute_unstable: set[str] | frozenset[str] = frozenset(),
    repository_class_qualified_names: set[str] | frozenset[str] = frozenset(),
) -> dict[str, dict[str, Any]]:
    candidates: dict[str, list[dict[str, Any]]] = {}

    def direct_returns(
        function: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> list[ast.AST]:
        values: list[ast.AST] = []

        class ReturnCollector(ast.NodeVisitor):
            def visit_Return(self, node: ast.Return) -> None:
                if node.value is not None:
                    values.append(node.value)

            def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
                return

            def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
                return

            def visit_ClassDef(self, node: ast.ClassDef) -> None:
                return

            def visit_Lambda(self, node: ast.Lambda) -> None:
                return

        collector = ReturnCollector()
        for statement in function.body:
            collector.visit(statement)
        return values

    module_facts = _scope_binding_facts(tree)
    classes = {
        statement.name: statement for statement in tree.body
        if isinstance(statement, ast.ClassDef)
    }
    class_aliases = {
        name: name for name in classes
        if module_facts["binding_counts"].get(name) == 1
        and name not in _global_declared_names(tree)
    }
    pending = {
        name: value for name, value in module_facts["unique_bindings"].items()
        if name not in _global_declared_names(tree)
    }
    while pending:
        progressed = False
        for name, value in list(pending.items()):
            if isinstance(value, ast.Name) and value.id in class_aliases:
                class_aliases[name] = class_aliases[value.id]
                del pending[name]
                progressed = True
        if not progressed:
            break

    subclass_overrides: set[tuple[str, str]] = set()
    unsafe_dispatch: set[str] = set()
    bases_by_class: dict[str, set[str]] = {}
    for name, statement in classes.items():
        bases: set[str] = set()
        for base in statement.bases:
            resolved = class_aliases.get(base.id) if isinstance(base, ast.Name) else None
            if resolved is None:
                # Unknown bases can hide a subclass relationship.
                unsafe_dispatch.update(classes)
            else:
                bases.add(resolved)
        bases_by_class[name] = bases
        if statement.decorator_list or statement.keywords:
            unsafe_dispatch.add(name)

    for name, statement in classes.items():
        ancestors = set(bases_by_class[name])
        pending_bases = list(ancestors)
        while pending_bases:
            for base in bases_by_class.get(pending_bases.pop(), set()):
                if base not in ancestors:
                    ancestors.add(base)
                    pending_bases.append(base)
        for base in ancestors:
            for method in _scope_bound_names(statement):
                subclass_overrides.add((base, method))
        if name in unsafe_dispatch:
            unsafe_dispatch.update(ancestors)

    def add_helper(
        function: ast.FunctionDef | ast.AsyncFunctionDef,
        *,
        kind: str,
        owner: str | None = None,
    ) -> None:
        if function.decorator_list:
            return
        returns = direct_returns(function)
        if len(returns) != 1:
            return
        args = function.args
        if (
            args.vararg is not None
            or args.kwarg is not None
            or args.kwonlyargs
            or args.defaults
            or any(value is not None for value in args.kw_defaults)
        ):
            return
        params = tuple(arg.arg for arg in (*args.posonlyargs, *args.args))

        receiver = params[0] if params else None
        receiver_facts = _scope_binding_facts(function)
        if kind == "method" and (
            receiver not in {"self", "cls"}
            or receiver_facts["binding_counts"].get(receiver) != 1
            or receiver in _descendant_nonlocal_names(function)
            or receiver in _global_declared_names(tree)
        ):
            return
        local_bound_names = _scope_bound_names(function)
        if any(
            isinstance(call.func, ast.Name)
            and call.func.id in local_bound_names
            for call in ast.walk(returns[0])
            if isinstance(call, ast.Call)
        ):
            return

        candidates.setdefault(function.name, []).append(
            {
                "return": returns[0],
                "params": params,
                "kind": kind,
                "owner": owner,
                "receiver": receiver,
            }
        )

    module_facts = _scope_binding_facts(tree)
    module_global_unstable = _global_declared_names(tree)
    attribute_unstable = (
        _attribute_mutated_names(tree) | set(repository_attribute_unstable)
    )

    for statement in tree.body:
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if (
                module_facts["binding_counts"].get(statement.name) == 1
                and statement.name not in module_global_unstable
                and statement.name not in repository_attribute_unstable
                and "*" not in repository_attribute_unstable
            ):
                add_helper(statement, kind="module")
        elif isinstance(statement, ast.ClassDef):
            class_bound_names = _scope_bound_names(statement)
            class_lookup_unstable = bool(
                class_bound_names & {"__getattribute__", "__getattr__"}
            )
            if (
                statement.name in unsafe_dispatch
                or statement.name in repository_base_names
                or statement.name in repository_class_qualified_names
                or bool(statement.bases)
                or class_lookup_unstable
                or "*" in attribute_unstable
            ):
                continue
            class_facts = _scope_binding_facts(statement)
            for child in statement.body:
                if (
                    isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and class_facts["binding_counts"].get(child.name) == 1
                    and child.name not in attribute_unstable
                    and (statement.name, child.name) not in subclass_overrides
                ):
                    add_helper(
                        child,
                        kind="method",
                        owner=statement.name,
                    )

    return {
        name: specs[0]
        for name, specs in candidates.items()
        if len(specs) == 1
    }

def _resolve_key_expression(
    expression: ast.AST,
    *,
    bindings: Mapping[str, ast.AST],
    helper_returns: Mapping[str, Mapping[str, Any]],
    seen_names: frozenset[str] = frozenset(),
    seen_helpers: frozenset[str] = frozenset(),
    method_owner: tuple[str, str] | None = None,
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
            seen_helpers=seen_helpers,
            method_owner=method_owner,
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
            seen_helpers=seen_helpers,
            method_owner=method_owner,
        )
        right_status, right_value = _resolve_key_expression(
            expression.right,
            bindings=bindings,
            helper_returns=helper_returns,
            seen_names=seen_names,
            seen_helpers=seen_helpers,
            method_owner=method_owner,
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
        helper_name = _call_name(expression)
        if (
            isinstance(expression.func, ast.Name)
            and expression.func.id in bindings
        ):
            return "unresolved", None
        helper = helper_returns.get(helper_name)
        if helper is None or helper_name in seen_helpers:
            return "unresolved", None
        if any(isinstance(arg, ast.Starred) for arg in expression.args):
            return "unresolved", None
        if any(keyword.arg is None for keyword in expression.keywords):
            return "unresolved", None

        kind = str(helper["kind"])
        params = list(helper["params"])
        if isinstance(expression.func, ast.Name):
            if kind != "module":
                return "unresolved", None
        elif isinstance(expression.func, ast.Attribute):
            helper_owner = helper.get("owner")
            if (
                kind != "method"
                or not isinstance(expression.func.value, ast.Name)
                or method_owner != (helper_owner, expression.func.value.id)
                or helper_owner is None
            ):
                return "unresolved", None
            if params and params[0] in {"self", "cls"}:
                params = params[1:]
        else:
            return "unresolved", None

        argument_bindings: dict[str, ast.AST] = {}
        if len(expression.args) > len(params):
            return "unresolved", None
        for name, value in zip(params, expression.args):
            argument_bindings[name] = value
        for keyword in expression.keywords:
            if keyword.arg not in params or keyword.arg in argument_bindings:
                return "unresolved", None
            argument_bindings[keyword.arg] = keyword.value
        if set(argument_bindings) != set(params):
            return "unresolved", None

        helper_bindings = dict(helper.get("defining_bindings", {}))
        helper_bindings.update(argument_bindings)
        return _resolve_key_expression(
            helper["return"],
            bindings=helper_bindings,
            helper_returns=helper_returns,
            seen_names=seen_names,
            seen_helpers=seen_helpers | {helper_name},
            method_owner=(
                (str(helper["owner"]), str(helper["receiver"]))
                if kind == "method" and helper.get("owner") is not None
                else None
            ),
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
    processed_mutator_sites: set[tuple[str, int, int]] = set()
    known_mutator_alias_names: dict[str, set[str]] = {}

    source_modules: list[
        tuple[
            Path,
            str,
            str,
            ast.Module,
            dict[str, ast.AST],
            dict[str, dict[str, Any]],
        ]
    ] = []
    repository_constant_candidates: dict[str, set[str]] = {}
    repository_base_names: set[str] = set()
    repository_attribute_unstable: set[str] = set()
    repository_class_qualified_names: set[str] = set()

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

        repository_attribute_unstable.update(
            _attribute_mutated_names(source_tree)
        )

        symbols: dict[str, str] = {}
        pending_symbols: dict[str, ast.AST] = {}
        for statement in source_tree.body:
            if isinstance(statement, ast.ClassDef):
                symbols[statement.name] = statement.name
            elif isinstance(statement, ast.ImportFrom):
                for alias in statement.names:
                    symbols[alias.asname or alias.name] = alias.name
            elif (
                isinstance(statement, ast.Assign)
                and len(statement.targets) == 1
                and isinstance(statement.targets[0], ast.Name)
            ):
                pending_symbols[statement.targets[0].id] = statement.value
            elif (
                isinstance(statement, ast.AnnAssign)
                and isinstance(statement.target, ast.Name)
                and statement.value is not None
            ):
                pending_symbols[statement.target.id] = statement.value

        while pending_symbols:
            progressed = False
            for name, expression in list(pending_symbols.items()):
                resolved: str | None = None
                if isinstance(expression, ast.Name):
                    resolved = symbols.get(expression.id)
                elif isinstance(expression, ast.Attribute):
                    resolved = expression.attr
                if resolved is None:
                    continue
                symbols[name] = resolved
                del pending_symbols[name]
                progressed = True
            if not progressed:
                break

        def resolved_symbol(expression: ast.AST) -> str | None:
            if isinstance(expression, ast.Name):
                return symbols.get(expression.id, expression.id)
            if isinstance(expression, ast.Attribute):
                return expression.attr
            return None

        for class_node in (
            node for node in ast.walk(source_tree)
            if isinstance(node, ast.ClassDef)
        ):
            for base in class_node.bases:
                resolved = resolved_symbol(base)
                if resolved is not None:
                    repository_base_names.add(resolved)
                else:
                    for child in ast.walk(base):
                        resolved_child = resolved_symbol(child)
                        if resolved_child is not None:
                            repository_base_names.add(resolved_child)

        for call in (
            node for node in ast.walk(source_tree)
            if isinstance(node, ast.Call)
        ):
            call_name = _call_name(call)
            if call_name == "type" and len(call.args) >= 2:
                for child in ast.walk(call.args[1]):
                    resolved = resolved_symbol(child)
                    if resolved is not None:
                        repository_base_names.add(resolved)
            elif call_name == "new_class" and len(call.args) >= 2:
                for child in ast.walk(call.args[1]):
                    resolved = resolved_symbol(child)
                    if resolved is not None:
                        repository_base_names.add(resolved)
            if call_name in {"getattr", "vars"} and call.args:
                resolved = resolved_symbol(call.args[0])
                if resolved is not None:
                    repository_class_qualified_names.add(resolved)

        for node in ast.walk(source_tree):
            if (
                isinstance(node, ast.Attribute)
                and isinstance(node.ctx, ast.Load)
                and isinstance(node.value, ast.Name)
            ):
                resolved = symbols.get(node.value.id)
                if resolved is not None:
                    repository_class_qualified_names.add(resolved)

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
        module_bindings = {
            name: expression
            for name, expression in _unique_expression_bindings(source_tree).items()
            if name not in repository_attribute_unstable
            and "*" not in repository_attribute_unstable
        }
        helper_returns = _helper_return_expressions(
            source_tree,
            repository_base_names=repository_base_names,
            repository_attribute_unstable=repository_attribute_unstable,
            repository_class_qualified_names=repository_class_qualified_names,
        )
        helper_defining_bindings = _freeze_bindings(
            {
                name: expression
                for name, expression in module_bindings.items()
                if name not in _global_declared_names(source_tree)
            },
            helper_returns={},
        )
        for helper_spec in helper_returns.values():
            helper_spec["defining_bindings"] = helper_defining_bindings
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

    module_bindings_by_name: dict[str, dict[str, ast.AST]] = {}
    for (
        _source_path,
        relative,
        _source_text,
        _source_tree,
        module_bindings,
        helper_returns,
    ) in source_modules:
        donor_global_unstable = _global_declared_names(_source_tree)
        stable_module_bindings = {
            name: expression
            for name, expression in module_bindings.items()
            if name not in donor_global_unstable
        }
        module_bindings_by_name[_module_name_from_relative(relative)] = (
            _freeze_bindings(
                stable_module_bindings,
                helper_returns=helper_returns,
            )
        )

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
        resolved_call_name: str | None = None,
        method_owner: tuple[str, str] | None = None,
    ) -> None:
        call_name = resolved_call_name or _call_name(call)
        key_index = _MUTATOR_KEY_INDEX.get(call_name)
        if key_index is None:
            return
        line = int(getattr(call, "lineno", 0))
        column = int(getattr(call, "col_offset", 0))
        processed_mutators.add((relative, line, column, call_name))
        processed_mutator_sites.add((relative, line, column))
        row = {
            "path": relative,
            "line": line,
            "function": function_name,
            "call": call_name,
        }
        if (
            any(isinstance(arg, ast.Starred) for arg in call.args)
            or any(keyword.arg is None for keyword in call.keywords)
        ):
            unresolved_kv_mutators.append(
                {**row, "reason": "ambiguous_argument_unpacking"}
            )
            return
        if len(call.args) <= key_index:
            unresolved_kv_mutators.append({**row, "reason": "missing_key_argument"})
            return

        key_expression = call.args[key_index]
        status, resolved_key = _resolve_key_expression(
            key_expression,
            bindings=bindings,
            helper_returns=helper_returns,
            method_owner=method_owner,
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
        imported_module_bindings, has_wildcard_import = (
            _imported_repository_bindings(
                source_tree,
                current_module=_module_name_from_relative(relative),
                module_bindings_by_name=module_bindings_by_name,
            )
        )
        module_global_unstable_names = _global_declared_names(source_tree)

        def scan_scope(
            scope_node: ast.AST,
            *,
            inherited_bindings: Mapping[str, ast.AST],
            inherited_mutator_aliases: Mapping[str, str],
            scope_name: str,
            owner_class: str | None,
        ) -> None:
            facts = _scope_binding_facts(scope_node)
            method_identity = None
            if owner_class is not None and isinstance(
                scope_node, (ast.FunctionDef, ast.AsyncFunctionDef)
            ):
                positional = [*scope_node.args.posonlyargs, *scope_node.args.args]
                receiver = positional[0].arg if positional else None
                if (
                    receiver in {"self", "cls"}
                    and not scope_node.decorator_list
                    and facts["binding_counts"].get(receiver) == 1
                    and receiver not in _descendant_nonlocal_names(scope_node)
                    and receiver not in module_global_unstable_names
                ):
                    method_identity = (owner_class, receiver)
            descendant_nonlocal_names = (
                _descendant_nonlocal_names(scope_node)
                if isinstance(
                    scope_node,
                    (ast.FunctionDef, ast.AsyncFunctionDef),
                )
                else set()
            )
            blocked_names = (
                set(facts["bound_names"])
                | descendant_nonlocal_names
            )
            if isinstance(scope_node, ast.Module):
                blocked_names |= module_global_unstable_names
            scope_bindings = {
                name: expression
                for name, expression in inherited_bindings.items()
                if name not in blocked_names
            }
            scope_mutator_aliases = {
                name: call_name
                for name, call_name in inherited_mutator_aliases.items()
                if name not in blocked_names
            }

            if isinstance(scope_node, ast.Module):
                if not has_wildcard_import:
                    for name, expression in imported_module_bindings.items():
                        if (
                            facts["binding_counts"].get(name) == 1
                            and name not in module_global_unstable_names
                        ):
                            scope_bindings[name] = expression
                    for name, expression in module_bindings.items():
                        if name not in module_global_unstable_names:
                            scope_bindings[name] = expression
                statements = scope_node.body
            elif isinstance(scope_node, ast.ClassDef):
                # Class-body names use LOAD_NAME semantics rather than a
                # lexical closure. Do not retroactively treat a class-local
                # assignment as proof for every executable statement.
                statements = scope_node.body
            else:
                for name, expression in facts["unique_bindings"].items():
                    if name not in descendant_nonlocal_names:
                        scope_bindings[name] = expression
                statements = scope_node.body

            scope_mutator_aliases = _extend_mutator_aliases(
                scope_mutator_aliases,
                facts["unique_bindings"],
            )
            scope_sql_aliases = _extend_sql_aliases(
                {},
                scope_bindings,
            )
            scope_sql_aliases = _extend_sql_aliases(
                scope_sql_aliases,
                facts["unique_bindings"],
            )
            known_mutator_alias_names.setdefault(relative, set()).update(
                scope_mutator_aliases
            )
            unresolved_kv_mutators.extend(
                _sql_callable_escape_rows(
                    statements,
                    aliases=scope_sql_aliases,
                    relative=relative,
                    scope_name=scope_name,
                )
            )
            unresolved_kv_mutators.extend(
                _mutator_callable_escape_rows(
                    statements,
                    aliases=scope_mutator_aliases,
                    bindings=scope_bindings,
                    binding_counts=facts["binding_counts"],
                    relative=relative,
                    scope_name=scope_name,
                    allow_local_aliases=isinstance(
                        scope_node, (ast.FunctionDef, ast.AsyncFunctionDef)
                    ),
                )
            )

            scope_helper_returns = helper_returns
            if not isinstance(scope_node, ast.Module):
                scope_helper_returns = {
                    name: spec
                    for name, spec in helper_returns.items()
                    if name not in facts["bound_names"]
                }

            scope_bindings = _freeze_bindings(
                scope_bindings,
                helper_returns=scope_helper_returns,
            )

            if isinstance(scope_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                lower = scope_node.name.lower()
                segment = ast.get_source_segment(source_text, scope_node) or ""
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
                        f"{relative}:{scope_node.lineno}:{scope_node.name}"
                    )

            for call in _executable_calls(statements):
                resolved_sql_call_name = _resolved_sql_callable_name(
                    call.func,
                    scope_sql_aliases,
                )
                sql_escape_reason = _sql_mutation_escape_reason(
                    call,
                    relative=relative,
                    scope_name=scope_name,
                    resolved_call_name=resolved_sql_call_name,
                )
                if sql_escape_reason is not None:
                    unresolved_kv_mutators.append(
                        {
                            "path": relative,
                            "line": int(getattr(call, "lineno", 0)),
                            "function": scope_name,
                            "call": _call_name(call),
                            "key_status": "unresolved",
                            "resolved_key": None,
                            "reason": sql_escape_reason,
                        }
                    )

                resolved_call_name = _call_name(call)
                if resolved_call_name not in _MUTATOR_KEY_INDEX:
                    resolved_call_name = _resolved_mutator_callable_name(
                        call.func,
                        scope_mutator_aliases,
                    )
                if resolved_call_name is None:
                    continue
                inspect_mutator(
                    call=call,
                    relative=relative,
                    function_name=scope_name,
                    bindings=scope_bindings,
                    helper_returns=scope_helper_returns,
                    resolved_call_name=resolved_call_name,
                    method_owner=method_identity,
                )

            for child_scope in _nested_scope_nodes(statements):
                if isinstance(scope_node, ast.ClassDef):
                    # Function/class bodies nested in a class do not close
                    # over the class namespace. They may still close over the
                    # lexical scope that contains the class itself.
                    child_inherited = inherited_bindings
                    child_mutator_aliases = inherited_mutator_aliases
                else:
                    child_inherited = scope_bindings
                    child_mutator_aliases = scope_mutator_aliases

                if isinstance(child_scope, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    child_name = child_scope.name
                    child_owner = (
                        scope_node.name
                        if isinstance(scope_node, ast.ClassDef)
                        else None
                    )
                else:
                    child_name = f"<class:{child_scope.name}>"
                    child_owner = child_scope.name
                scan_scope(
                    child_scope,
                    inherited_bindings=child_inherited,
                    inherited_mutator_aliases=child_mutator_aliases,
                    scope_name=child_name,
                    owner_class=child_owner,
                )

        scan_scope(
            source_tree,
            inherited_bindings={},
            inherited_mutator_aliases={},
            scope_name="<module>",
            owner_class=None,
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
            is_alias_call = (
                isinstance(call.func, ast.Name)
                and call.func.id
                in known_mutator_alias_names.get(relative, set())
            )
            if call_name not in _MUTATOR_KEY_INDEX and not is_alias_call:
                continue
            line = int(getattr(call, "lineno", 0))
            column = int(getattr(call, "col_offset", 0))
            if (relative, line, column) in processed_mutator_sites:
                continue
            unresolved_kv_mutators.append(
                {
                    "path": relative,
                    "line": line,
                    "function": "<unscanned-lexical-scope>",
                    "call": call_name,
                    "key_status": "unresolved",
                    "resolved_key": None,
                    "reason": (
                        "unscanned_mutator_alias"
                        if is_alias_call
                        else "unscanned_lexical_scope"
                    ),
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
