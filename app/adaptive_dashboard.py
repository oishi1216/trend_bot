from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class AdaptiveDashboardData:
    status: dict[str, Any]
    runs: list[dict[str, Any]]
    latest_run: dict[str, Any] | None
    task_status: dict[str, Any]


def _read_json_file(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _format_no_entry_reason(decision: dict[str, Any]) -> tuple[str, str]:
    regime = decision.get("regime", "unknown")
    action = decision.get("action")
    status = str(decision.get("status") or "").lower()
    score = float(decision.get("score") or 0)

    regime_label = {
        "unclear": "\u76f8\u5834\u65b9\u5411\u304c\u4e0d\u660e\u78ba",
        "trend": "\u30c8\u30ec\u30f3\u30c9\u5224\u5b9a\u3001\u30a8\u30f3\u30c8\u30ea\u30fc\u6761\u4ef6\u672a\u9054",
        "range": "\u30ec\u30f3\u30b8\u5224\u5b9a\u3001\u53cd\u8ee2\u6761\u4ef6\u672a\u9054",
    }.get(regime, "\u30a8\u30f3\u30c8\u30ea\u30fc\u6761\u4ef6\u672a\u9054")

    if action not in (None, "none"):
        if action == "enter_long":
            return "\u5019\u88dc\u3042\u308a\uff1a\u8cb7\u3044\u6761\u4ef6\u3092\u6e80\u305f\u3057\u3066\u3044\u307e\u3059", regime_label
        if action == "enter_short":
            return "\u5019\u88dc\u3042\u308a\uff1a\u58f2\u308a\u6761\u4ef6\u3092\u6e80\u305f\u3057\u3066\u3044\u307e\u3059", regime_label
        return f"\u5019\u88dc\u3042\u308a\uff1a{action} \u6761\u4ef6\u3092\u6e80\u305f\u3057\u3066\u3044\u307e\u3059", regime_label

    if regime == "trend":
        if score > 0 or status in {"entry_blocked", "blocked"}:
            return "\u898b\u9001\u308a\uff1a\u30c8\u30ec\u30f3\u30c9\u5224\u5b9a\u3001\u5019\u88dc\u6761\u4ef6\u306f\u3042\u308b\u304c\u6700\u7d42\u6761\u4ef6\u672a\u9054", regime_label
        return "\u898b\u9001\u308a\uff1a\u30c8\u30ec\u30f3\u30c9\u5224\u5b9a\u3001\u30a8\u30f3\u30c8\u30ea\u30fc\u6761\u4ef6\u672a\u9054", regime_label
    if regime == "range":
        if score > 0:
            return "\u898b\u9001\u308a\uff1a\u30ec\u30f3\u30b8\u5224\u5b9a\u3001\u5019\u88dc\u6761\u4ef6\u306f\u3042\u308b\u304c\u6700\u7d42\u6761\u4ef6\u672a\u9054", regime_label
        return "\u898b\u9001\u308a\uff1a\u30ec\u30f3\u30b8\u5224\u5b9a\u3001\u53cd\u8ee2\u6761\u4ef6\u672a\u9054", regime_label
    if score > 0:
        return "\u898b\u9001\u308a\uff1a\u76f8\u5834\u65b9\u5411\u304c\u4e0d\u660e\u78ba\u3060\u304c\u5019\u88dc\u6761\u4ef6\u306f\u3042\u308b", regime_label
    return "\u898b\u9001\u308a\uff1a\u76f8\u5834\u65b9\u5411\u304c\u4e0d\u660e\u78ba", regime_label


def _latest_run_summary(event_row: sqlite3.Row) -> dict[str, Any]:
    payload = json.loads(event_row["payload"])
    results = payload.get("results", [])
    latest_dt = payload.get("started_at") or event_row["ts"]
    task_result = {
        "started_at": payload.get("started_at"),
        "finished_at": payload.get("finished_at"),
        "status": "success" if payload.get("errors") in (None, [], {}) else "failure",
        "mode": payload.get("mode"),
        "strategy_profile": payload.get("strategy_profile"),
        "nav": payload.get("nav"),
        "drawdown": payload.get("drawdown"),
        "monthly_loss": payload.get("monthly_loss"),
        "results_count": len(results),
        "errors": payload.get("errors", []),
        "positions": payload.get("positions", []),
        "currency_strength": payload.get("currency_strength", {}),
        "results": results,
    }
    task_result["has_candidate"] = any(
        float(item.get("decision", {}).get("score") or 0) > 0
        or item.get("decision", {}).get("action") not in (None, "none")
        for item in results
    )
    task_result["latest_action"] = next(
        (
            item
            for item in results
            if item.get("decision", {}).get("action") not in (None, "none")
        ),
        None,
    )
    task_result["latest_run_time"] = latest_dt
    return task_result


def _currency_rows(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in results:
        decision = item.get("decision", {})
        metadata = decision.get("metadata", {}) or {}
        score = float(decision.get("score") or 0)
        action = decision.get("action")
        summary, regime_hint = _format_no_entry_reason(decision)
        rows.append(
            {
                "instrument": item.get("instrument"),
                "regime": decision.get("regime", "unknown"),
                "score": score,
                "strength_gap": metadata.get("strength_gap"),
                "status": item.get("status"),
                "action": action,
                "no_entry_reason": decision.get("reason"),
                "no_entry_reason_summary": summary,
                "is_candidate": score > 0 or action not in (None, "none"),
                "is_action_candidate": action not in (None, "none"),
            }
        )
    return rows


def load_adaptive_dashboard(db_path: str, task_log_dir: str | None = None) -> AdaptiveDashboardData:
    path = Path(db_path)
    status: dict[str, Any] = {
        "db_path": str(path),
        "exists": path.exists(),
        "strategy_profile": "adaptive_dual_regime_v1",
        "nav": None,
        "drawdown": None,
        "monthly_loss": None,
        "positions": [],
        "currency_rows": [],
    }
    runs: list[dict[str, Any]] = []
    latest_run: dict[str, Any] | None = None

    if path.exists():
        conn = sqlite3.connect(path)
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                """
                SELECT id, ts, payload
                FROM events
                WHERE kind = 'engine_run'
                ORDER BY id DESC
                LIMIT 5
                """
            ).fetchall()
        finally:
            conn.close()

        for row in rows:
            run = _latest_run_summary(row)
            runs.append(
                {
                    "started_at": run["started_at"],
                    "finished_at": run["finished_at"],
                    "status": run["status"],
                    "has_candidate": run["has_candidate"],
                    "nav": run["nav"],
                    "drawdown": run["drawdown"],
                    "monthly_loss": run["monthly_loss"],
                    "results_count": run["results_count"],
                }
            )

        if rows:
            latest_run = _latest_run_summary(rows[0])
            status.update(
                {
                    "nav": latest_run["nav"],
                    "drawdown": latest_run["drawdown"],
                    "monthly_loss": latest_run["monthly_loss"],
                    "positions": latest_run["positions"],
                    "currency_rows": _currency_rows(latest_run["results"]),
                    "latest_started_at": latest_run["started_at"],
                    "latest_finished_at": latest_run["finished_at"],
                }
            )

    task_status: dict[str, Any] = {
        "task_registered": False,
        "last_run_time": None,
        "last_result": None,
        "last_result_meaning": None,
        "latest_log_updated": None,
    }
    if task_log_dir:
        log_dir = Path(task_log_dir)
        latest_json = _read_json_file(log_dir / "adaptive_paper_daily_latest.json")
        if latest_json:
            task_status.update(
                {
                    "task_registered": True,
                    "last_run_time": latest_json.get("started_at"),
                    "last_result": "success" if not latest_json.get("errors") else "failure",
                    "last_result_meaning": "Success"
                    if not latest_json.get("errors")
                    else "Runtime or instrument error",
                    "latest_log_updated": (
                        datetime.fromtimestamp(
                            (log_dir / "adaptive_paper_daily_latest.json").stat().st_mtime
                        ).isoformat()
                        if (log_dir / "adaptive_paper_daily_latest.json").exists()
                        else None
                    ),
                    "latest_log": latest_json,
                }
            )

    return AdaptiveDashboardData(
        status=status,
        runs=runs,
        latest_run=latest_run,
        task_status=task_status,
    )
