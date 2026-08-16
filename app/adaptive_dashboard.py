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
        rows.append(
            {
                "instrument": item.get("instrument"),
                "regime": decision.get("regime", "unknown"),
                "score": decision.get("score", 0.0),
                "strength_gap": metadata.get("strength_gap"),
                "status": item.get("status"),
                "action": decision.get("action"),
                "no_entry_reason": decision.get("reason"),
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
                    "last_result": "success"
                    if not latest_json.get("errors")
                    else "failure",
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
