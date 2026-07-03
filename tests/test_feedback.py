from __future__ import annotations

from app.config import Settings
from app.feedback import FeedbackService, redact_sensitive
from app.storage import Storage


def test_redact_sensitive_removes_secrets_and_raw():
    data = {
        "api_token": "secret-token",
        "account_id": "abc123",
        "nested": {"Authorization": "Bearer secret", "raw": {"huge": "payload"}},
        "safe": 42,
    }
    cleaned = redact_sensitive(data)
    assert cleaned["api_token"] == "[redacted]"
    assert cleaned["account_id"] == "[redacted]"
    assert cleaned["nested"]["Authorization"] == "[redacted]"
    assert cleaned["nested"]["raw"] == "[omitted]"
    assert cleaned["safe"] == 42


def test_build_report_does_not_include_unselected_events(tmp_path, monkeypatch):
    monkeypatch.setenv("DB_PATH", str(tmp_path / "test.sqlite3"))
    settings = Settings.from_env()
    storage = Storage(settings.db_path, settings.paper_initial_balance)
    storage.add_event("INFO", "engine_run", {"nav": 1_000_000})
    storage.add_event("INFO", "openai_feedback", {"should": "not be resent"})
    service = FeedbackService(settings, storage)

    report = service.build_report({"mode": "paper", "results": []})
    kinds = {event["kind"] for event in report["recent_operational_events"]}
    assert "engine_run" in kinds
    assert "openai_feedback" not in kinds
