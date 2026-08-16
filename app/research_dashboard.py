from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .research_backtest import dump_results, load_universe, run_backtest


@dataclass(frozen=True)
class ResearchDashboardData:
    status: dict[str, Any]
    strategies: dict[str, Any]
    portfolio: dict[str, Any]
    meta: dict[str, Any]


def _format_pct(value: float | None) -> str:
    return "-" if value is None else f"{value:.1f}%"


def _format_num(value: float | None) -> str:
    return "-" if value is None else f"{value:.2f}"


def _grade_gte(actual: float | None, threshold: float) -> str:
    if actual is None:
        return "WARN"
    if actual >= threshold:
        return "PASS"
    if actual >= threshold * 0.9:
        return "WARN"
    return "FAIL"


def _grade_lte(actual: float | None, threshold: float) -> str:
    if actual is None:
        return "WARN"
    if actual <= threshold:
        return "PASS"
    if actual <= threshold * 1.1:
        return "WARN"
    return "FAIL"


def _api_estimate() -> dict[str, Any]:
    estimated_requests = 402
    rate_limit = 60
    return {
        "target_symbols": 200,
        "contract_plan": "Light",
        "api_limit_per_min": rate_limit,
        "estimated_requests": estimated_requests,
        "theoretical_min_minutes": round(estimated_requests / rate_limit, 1),
        "estimated_time_range_minutes": [8, 12],
        "available_period": "2021-xx-xx \u301c 2026-xx-xx",
        "note": "J-Quants\u540c\u671f200\u92fc\u67c4\u306f\u5951\u7d04\u4e0a\u9650\u306b\u5408\u308f\u305b\u3066\u5206\u5272\u53d6\u5f97\u3057\u3066\u304f\u3060\u3055\u3044",
    }


def _judge_payload(payload: dict[str, Any]) -> dict[str, Any]:
    walk_forward = payload.get("walk_forward", {})
    out_of_sample = walk_forward.get("out_of_sample", {})
    metrics = out_of_sample.get("metrics", {})
    annual = out_of_sample.get("annual", {})
    sensitivity = payload.get("sensitivity", {})

    cagr = (metrics.get("cagr") or 0.0) * 100.0
    max_dd = (metrics.get("max_dd") or 0.0) * 100.0
    top1 = (walk_forward.get("top1_share") or 0.0) * 100.0
    positive_year_ratio = None if not annual else sum(1 for row in annual.values() if row.get("pnl", 0.0) > 0) / len(annual) * 100.0
    cost_x2_cagr = (sensitivity.get("cost_x2", {}).get("cagr") or 0.0) * 100.0
    fold_runs = walk_forward.get("folds", [])
    fold_positive = sum(1 for fold in fold_runs if fold.get("metrics", {}).get("cagr", 0.0) > 0)
    fold_total = len(fold_runs) or 5

    checks = [
        {"label": "譛滄俣螟砲AGR", "actual": _format_pct(cagr), "basis": ">=20%", "grade": _grade_gte(cagr, 20.0)},
        {"label": "譛螟ｧDD", "actual": _format_pct(max_dd), "basis": "<=12%", "grade": _grade_lte(max_dd, 12.0)},
        {"label": "PF", "actual": _format_num(metrics.get("pf")), "basis": ">=1.40", "grade": _grade_gte(metrics.get("pf"), 1.40)},
        {"label": "譛滄俣螟門叙蠑墓焚", "actual": str(metrics.get("trades", 0)), "basis": ">=150", "grade": _grade_gte(float(metrics.get("trades", 0)), 150.0)},
        {"label": "繝励Λ繧ｹ蟷ｴ邇・", "actual": _format_pct(positive_year_ratio), "basis": ">=70%", "grade": _grade_gte(positive_year_ratio, 70.0)},
        {"label": "譛螟ｧ1驫俶氛萓晏ｭ・", "actual": _format_pct(top1), "basis": "<=15%", "grade": _grade_lte(top1, 15.0)},
        {"label": "繧ｳ繧ｹ繝・蛟・", "actual": _format_pct(cost_x2_cagr), "basis": ">0%", "grade": _grade_gte(cost_x2_cagr, 0.0)},
        {"label": "Walk-forward", "actual": f"{fold_positive}/{fold_total} fold\u30d7\u30e9\u30b9", "basis": "guideline", "grade": "PASS" if fold_positive >= (fold_total // 2 + 1) else "WARN"},
    ]

    overall = "PASS"
    if any(check["grade"] == "FAIL" for check in checks):
        overall = "FAIL"
    elif any(check["grade"] == "WARN" for check in checks):
        overall = "WARN"

    reasons = [f"・{c['label']} {c['actual']} → {c['grade']}" for c in checks]
    return {
        "overall": overall,
        "headline": (
            "\u30d5\u30a9\u30ef\u30fc\u30c9\u7dcf\u5408\u306f\u6ce8\u610f\u3001\u8ffd\u52a0\u78ba\u8a8d\u304c\u5fc5\u8981\u3067\u3059"
            if overall == "WARN"
            else "\u30d5\u30a9\u30ef\u30fc\u30c9\u7dcf\u5408\u306f\u826f\u597d\u3067\u3059"
            if overall == "PASS"
            else "\u7dcf\u5408\u8a55\u4fa1\u304c\u57fa\u6e96\u672a\u9054\u3067\u3059"
        ),
        "reasons": reasons,
        "checks": checks,
        "api_estimate": _api_estimate(),
        "sensitivity": sensitivity,
        "walk_forward": walk_forward,
    }


def load_research_dashboard(data_dir: str, cache_path: str, initial_equity: float = 1_000_000) -> ResearchDashboardData:
    data_root = Path(data_dir)
    cache = Path(cache_path)
    universe = load_universe(data_root) if data_root.exists() else {}
    payload = json.loads(cache.read_text(encoding="utf-8")) if cache.exists() else None
    if universe:
        payload = run_backtest(universe, initial_equity=initial_equity)
        dump_results(payload, cache)
    payload = payload or {"universe_size": len(universe), "strategies": {}, "portfolio": {}}
    judgement = _judge_payload(payload)
    return ResearchDashboardData(
        status={
            "mode": "annual_20_research_v0.6.0",
            "universe_size": payload.get("universe_size", len(universe)),
            "data_dir": str(data_root),
            "cache_path": str(cache),
            "loaded": bool(universe),
        },
        strategies=payload.get("strategies", {}),
        portfolio=payload.get("portfolio", {}),
        meta={
            "api_budget_note": "J-Quants\u540c\u671f200\u92fc\u67c4\u306f\u5951\u7d04\u4e0a\u9650\u306b\u5408\u308f\u305b\u3066\u5206\u5272\u53d6\u5f97\u3057\u3066\u304f\u3060\u3055\u3044",
            "cache_exists": cache.exists(),
            "walk_forward": payload.get("walk_forward", {}),
            "sensitivity": payload.get("sensitivity", {}),
            "annual_stability": payload.get("annual_stability", {}),
            "train": payload.get("train", {}),
            "validation": payload.get("validation", {}),
            "test": payload.get("test", {}),
            "judgement": judgement,
        },
    )
