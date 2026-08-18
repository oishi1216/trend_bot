from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .config import Settings
from .fx_research_data import FX_RESEARCH_INSTRUMENTS, data_status, load_history
from .research_backtest import dump_results, load_cached_results, run_research


@dataclass(frozen=True)
class ResearchDashboardData:
    status: dict[str, Any]
    headline: dict[str, Any]
    annual: dict[str, Any]
    symbols: dict[str, Any]
    regimes: dict[str, Any]
    score_bands: dict[str, Any]
    sensitivity: dict[str, Any]
    splits: dict[str, Any]
    walk_forward: dict[str, Any]
    verdict: dict[str, Any]
    meta: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _format_pct(value: float | None) -> str:
    return "-" if value is None else f"{value * 100:.1f}%"


def _format_num(value: float | None) -> str:
    if value is None:
        return "-"
    if value == float("inf"):
        return "inf"
    return f"{value:.2f}"


def _grade_gte(actual: float | None, threshold: float) -> str:
    if actual is None:
        return "WARN"
    if actual >= threshold:
        return "PASS"
    if actual >= threshold - abs(threshold) * 0.1 - 0.001:
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


def _judge(payload: dict[str, Any]) -> dict[str, Any]:
    metrics = payload.get("metrics", {})
    sensitivity = payload.get("sensitivity", {})
    walk_forward = payload.get("walk_forward", {})

    cagr = metrics.get("cagr")
    max_dd = metrics.get("max_dd")
    pf = metrics.get("pf")
    trades = metrics.get("trades", 0)
    cost_x2_cagr = sensitivity.get("cost_x2", {}).get("metrics", {}).get("cagr")
    fold_positive = walk_forward.get("fold_positive", 0)
    fold_total = walk_forward.get("fold_total", 0) or 1

    checks = [
        {"label": "CAGR", "actual": _format_pct(cagr), "basis": ">=0%", "grade": _grade_gte(cagr, 0.0)},
        {"label": "Max Drawdown", "actual": _format_pct(max_dd), "basis": "<=20%", "grade": _grade_lte(max_dd, 0.20)},
        {"label": "Profit Factor", "actual": _format_num(pf), "basis": ">=1.20", "grade": _grade_gte(pf, 1.20)},
        {"label": "Trade Count", "actual": str(trades), "basis": ">=30", "grade": _grade_gte(float(trades), 30.0)},
        {"label": "Cost x2 CAGR", "actual": _format_pct(cost_x2_cagr), "basis": ">=-5%", "grade": _grade_gte(cost_x2_cagr, -0.05)},
        {
            "label": "Walk-forward",
            "actual": f"{fold_positive}/{fold_total} folds positive",
            "basis": "majority",
            "grade": "PASS" if fold_positive >= (fold_total // 2 + 1) else "WARN",
        },
    ]
    overall = "PASS"
    if any(c["grade"] == "FAIL" for c in checks):
        overall = "FAIL"
    elif any(c["grade"] == "WARN" for c in checks):
        overall = "WARN"

    return {
        "overall": overall,
        "checks": checks,
        "reasons": [f"- {c['label']} {c['actual']} -> {c['grade']}" for c in checks],
    }


def load_research_dashboard(
    data_dir: str,
    cache_path: str,
    settings: Settings,
) -> ResearchDashboardData:
    status = data_status(data_dir, FX_RESEARCH_INSTRUMENTS)
    history = load_history(data_dir, FX_RESEARCH_INSTRUMENTS)
    cache = Path(cache_path)

    payload = load_cached_results(cache, data_dir)
    if payload is None and history:
        payload = run_research(
            history, settings, initial_equity=settings.paper_initial_balance
        )
        dump_results(payload, cache)
    payload = payload or {}

    if payload:
        verdict = _judge(payload)
    else:
        verdict = {
            "overall": "NO_DATA",
            "checks": [],
            "reasons": [
                "No synced FX history yet. Run scripts/Run-AdaptiveResearch.ps1"
            ],
        }

    return ResearchDashboardData(
        status={
            "mode": "fx_adaptive_research_v1",
            "strategy_profile": "adaptive_dual_regime_v1",
            "data_dir": status["data_dir"],
            "cache_path": str(cache),
            "cache_exists": cache.exists(),
            "instruments": status["instruments"],
            "last_data_update": status["last_update"],
            "all_synced": status["all_synced"],
            "backtest_period": payload.get("period", {}),
            "loaded": bool(history),
        },
        headline=payload.get("metrics", {}),
        annual=payload.get("annual", {}),
        symbols=payload.get("by_symbol", {}),
        regimes=payload.get("by_regime", {}),
        score_bands=payload.get("by_score_band", {}),
        sensitivity=payload.get("sensitivity", {}),
        splits={
            "train": payload.get("train", {}),
            "validation": payload.get("validation", {}),
            "test": payload.get("test", {}),
        },
        walk_forward=payload.get("walk_forward", {}),
        verdict=verdict,
        meta={
            "cost_assumptions": payload.get("cost_assumptions", {}),
            "baseline_config": payload.get("baseline_config", {}),
            "trade_count": payload.get("metrics", {}).get("trades", 0),
        },
    )
