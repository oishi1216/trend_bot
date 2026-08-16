from __future__ import annotations

import importlib
import json
import sqlite3

from fastapi.testclient import TestClient

from app.adaptive_dashboard import load_adaptive_dashboard


def _create_dashboard_db(path):
    conn = sqlite3.connect(path)
    try:
        conn.execute(
            """
            CREATE TABLE events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                kind TEXT NOT NULL,
                ts TEXT NOT NULL,
                payload TEXT NOT NULL
            )
            """
        )
        results = []
        for idx, instrument in enumerate(
            [
                "EURUSD",
                "GBPUSD",
                "USDJPY",
                "AUDUSD",
                "NZDUSD",
                "USDCAD",
                "USDCHF",
                "EURJPY",
                "GBPJPY",
                "EURAUD",
            ]
        ):
            results.append(
                {
                    "instrument": instrument,
                    "status": "entry_blocked" if idx else "ready",
                    "decision": {
                        "regime": "trend" if idx % 2 == 0 else "range",
                        "score": 78.0 + idx,
                        "action": "none" if idx else "enter_long",
                        "reason": "Adaptive trend no-entry; score=0.0, strength_gap=0.31, ADX=25.0, vol_multiplier=1.00"
                        if idx
                        else "entry candidate",
                        "metadata": {"strength_gap": 0.31 + idx / 100},
                    },
                }
            )
        payload = {
            "started_at": "2026-08-16T22:15:00+00:00",
            "finished_at": "2026-08-16T22:16:30+00:00",
            "errors": [],
            "nav": 12345.67,
            "drawdown": 0.0123,
            "monthly_loss": 0.0345,
            "positions": [{"instrument": "EURUSD", "side": "long"}],
            "currency_strength": {"USD": 1.2},
            "results": results,
        }
        for idx in range(2):
            conn.execute(
                "INSERT INTO events(kind, ts, payload) VALUES (?, ?, ?)",
                (
                    "engine_run",
                    f"2026-08-16T22:1{idx}:00+00:00",
                    json.dumps(payload),
                ),
            )
        conn.commit()
    finally:
        conn.close()


def _create_latest_log(path):
    path.write_text(
        json.dumps(
            {
                "started_at": "2026-08-16T22:15:00+00:00",
                "errors": [],
            }
        ),
        encoding="utf-8",
    )


def test_load_adaptive_dashboard_reads_sqlite_and_latest_json(tmp_path):
    db_path = tmp_path / "adaptive_paper.sqlite3"
    log_dir = tmp_path / "task_logs"
    log_dir.mkdir()
    _create_dashboard_db(db_path)
    _create_latest_log(log_dir / "adaptive_paper_daily_latest.json")

    dashboard = load_adaptive_dashboard(str(db_path), str(log_dir))

    assert dashboard.status["strategy_profile"] == "adaptive_dual_regime_v1"
    assert dashboard.status["nav"] == 12345.67
    assert dashboard.status["drawdown"] == 0.0123
    assert dashboard.status["monthly_loss"] == 0.0345
    assert len(dashboard.status["positions"]) == 1
    assert len(dashboard.status["currency_rows"]) == 10
    assert dashboard.status["currency_rows"][0]["regime"] == "trend"
    assert dashboard.status["currency_rows"][0]["strength_gap"] == 0.31
    assert dashboard.status["currency_rows"][0]["is_candidate"] is True
    assert dashboard.status["currency_rows"][0]["is_action_candidate"] is True
    assert dashboard.status["currency_rows"][0]["no_entry_reason_summary"] == "\u5019\u88dc\u3042\u308a\uff1a\u8cb7\u3044\u6761\u4ef6\u3092\u6e80\u305f\u3057\u3066\u3044\u307e\u3059"
    assert dashboard.status["currency_rows"][1]["no_entry_reason_summary"] == "\u898b\u9001\u308a\uff1a\u30ec\u30f3\u30b8\u5224\u5b9a\u3001\u5019\u88dc\u6761\u4ef6\u306f\u3042\u308b\u304c\u6700\u7d42\u6761\u4ef6\u672a\u9054"
    assert dashboard.status["currency_rows"][2]["no_entry_reason_summary"] == "\u898b\u9001\u308a\uff1a\u30c8\u30ec\u30f3\u30c9\u5224\u5b9a\u3001\u5019\u88dc\u6761\u4ef6\u306f\u3042\u308b\u304c\u6700\u7d42\u6761\u4ef6\u672a\u9054"
    assert dashboard.runs and len(dashboard.runs) == 2
    assert dashboard.runs[0]["has_candidate"] is True
    assert dashboard.latest_run["results"][0]["decision"]["action"] == "enter_long"
    assert dashboard.task_status["task_registered"] is True
    assert dashboard.task_status["last_result"] == "success"


def test_api_adaptive_status_uses_dashboard_loader(monkeypatch, tmp_path):
    db_path = tmp_path / "adaptive_paper.sqlite3"
    log_dir = tmp_path / "task_logs"
    log_dir.mkdir()
    _create_dashboard_db(db_path)
    _create_latest_log(log_dir / "adaptive_paper_daily_latest.json")

    monkeypatch.setenv("DB_PATH", str(tmp_path / "legacy.sqlite3"))
    monkeypatch.setenv("PAPER_INITIAL_BALANCE", "1000000")
    monkeypatch.setenv("STRATEGY_PROFILE", "adaptive_dual_regime_v1")
    app_main = importlib.import_module("app.main")
    monkeypatch.setattr(app_main, "adaptive_db_path", str(db_path))
    monkeypatch.setattr(app_main, "adaptive_task_log_dir", str(log_dir))

    client = TestClient(app_main.app)
    response = client.get("/api/adaptive/status")

    assert response.status_code == 200
    body = response.json()
    assert body["status"]["nav"] == 12345.67
    assert body["status"]["currency_rows"][0]["instrument"] == "EURUSD"
    assert body["task_status"]["last_result"] == "success"
