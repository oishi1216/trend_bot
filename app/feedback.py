from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field

from .config import Settings
from .storage import Storage

logger = logging.getLogger(__name__)

_ALLOWED_EVENT_KINDS = {
    "engine_run",
    "position_opened",
    "position_closed",
    "stop_filled",
    "instrument_run_failed",
    "scheduled_run_failed",
    "reconcile_removed_local_position",
    "untracked_broker_position",
}

# These keys are removed before anything leaves the application.
_SENSITIVE_KEY_PARTS = {
    "api_key",
    "apikey",
    "token",
    "secret",
    "authorization",
    "password",
    "account_id",
    "accountid",
    "broker_trade_id",
}
_OMITTED_KEYS = {"raw"}


class FeedbackFinding(BaseModel):
    severity: Literal["info", "warning", "error", "critical"]
    category: Literal[
        "execution",
        "data_quality",
        "risk_control",
        "reliability",
        "observability",
        "strategy_evidence",
    ]
    title: str
    evidence: list[str]
    recommendation: str
    code_change_candidate: bool


class TradingFeedback(BaseModel):
    overall_status: Literal["healthy", "watch", "action_required", "stop_and_review"]
    summary: str
    insufficient_data: bool
    findings: list[FeedbackFinding]
    safe_code_changes: list[str]
    human_review_items: list[str]
    prohibited_automatic_changes: list[str]


def redact_sensitive(value: Any) -> Any:
    """Recursively remove secrets, account identifiers, and verbose broker payloads."""
    if isinstance(value, dict):
        cleaned: dict[str, Any] = {}
        for key, item in value.items():
            normalized = str(key).lower()
            if normalized in _OMITTED_KEYS:
                cleaned[str(key)] = "[omitted]"
                continue
            if any(part in normalized for part in _SENSITIVE_KEY_PARTS):
                cleaned[str(key)] = "[redacted]"
                continue
            cleaned[str(key)] = redact_sensitive(item)
        return cleaned
    if isinstance(value, list):
        return [redact_sensitive(item) for item in value]
    if isinstance(value, tuple):
        return [redact_sensitive(item) for item in value]
    return value


class FeedbackService:
    def __init__(self, settings: Settings, storage: Storage) -> None:
        self.settings = settings
        self.storage = storage

    def _is_due(self) -> bool:
        last_sent = self.storage.get_kv("openai_feedback_last_sent_at")
        if not last_sent:
            return True
        try:
            last_dt = datetime.fromisoformat(last_sent)
            if last_dt.tzinfo is None:
                last_dt = last_dt.replace(tzinfo=timezone.utc)
        except ValueError:
            return True
        return datetime.now(timezone.utc) - last_dt >= timedelta(
            hours=self.settings.openai_feedback_min_interval_hours
        )

    def build_report(self, run_summary: dict[str, Any]) -> dict[str, Any]:
        events = [
            event
            for event in self.storage.recent_events(
                self.settings.openai_feedback_event_limit
            )
            if event.get("kind") in _ALLOWED_EVENT_KINDS
        ]
        report = {
            "report_version": "1.0",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "environment": {
                "broker_mode": self.settings.broker_mode,
                "instruments": list(self.settings.instruments),
                "trading_armed": self.settings.trading_armed,
            },
            "protected_risk_rules": {
                "risk_per_trade": self.settings.risk_per_trade,
                "max_aggregate_risk": self.settings.max_aggregate_risk,
                "max_gross_leverage": self.settings.max_gross_leverage,
                "drawdown_half_risk": self.settings.drawdown_half_risk,
                "drawdown_stop": self.settings.drawdown_stop,
            },
            "strategy_parameters": {
                "ema_days": self.settings.ema_days,
                "ema_slope_lookback": self.settings.ema_slope_lookback,
                "entry_channel_days": self.settings.entry_channel_days,
                "exit_channel_days": self.settings.exit_channel_days,
                "atr_days": self.settings.atr_days,
                "atr_stop_multiple": self.settings.atr_stop_multiple,
            },
            "latest_run": run_summary,
            "recent_operational_events": events,
        }
        return redact_sensitive(report)

    def analyze_run(
        self, run_summary: dict[str, Any], *, force: bool = False
    ) -> dict[str, Any]:
        if not self.settings.openai_feedback_enabled:
            return {"status": "disabled"}
        if not self.settings.openai_api_key:
            result = {"status": "error", "error": "OPENAI_API_KEY is not configured"}
            self.storage.add_event("ERROR", "openai_feedback_failed", result)
            return result
        if self.settings.broker_mode == "oanda_live" and not self.settings.openai_feedback_allow_live:
            return {"status": "skipped_live_mode"}
        if not force and not self._is_due():
            return {"status": "skipped_interval"}

        report = self.build_report(run_summary)
        try:
            # Import lazily so the bot can still start when feedback is disabled.
            from openai import OpenAI

            client = OpenAI(
                api_key=self.settings.openai_api_key,
                timeout=self.settings.openai_timeout_seconds,
                max_retries=2,
            )
            response = client.responses.parse(
                model=self.settings.openai_model,
                store=False,
                input=[
                    {
                        "role": "system",
                        "content": (
                            "You are reviewing an automated FX test-trading system. "
                            "Analyze only the supplied report. Separate software defects, "
                            "execution anomalies, missing data, and weak statistical evidence. "
                            "Never claim guaranteed profitability. Never recommend automatically "
                            "increasing risk, leverage, trade frequency, widening stops, disabling "
                            "drawdown controls, or changing protected strategy/risk parameters. "
                            "A code_change_candidate may be true only for technical fixes such as "
                            "logging, reconciliation, duplicate-order prevention, data validation, "
                            "error handling, tests, or observability. Strategy changes must be listed "
                            "only under human_review_items. Be conservative when sample size is small."
                        ),
                    },
                    {
                        "role": "user",
                        "content": json.dumps(report, ensure_ascii=False, separators=(",", ":")),
                    },
                ],
                text_format=TradingFeedback,
            )
            parsed = response.output_parsed
            if parsed is None:
                raise RuntimeError("The model returned no structured feedback")

            usage = getattr(response, "usage", None)
            if usage is not None and hasattr(usage, "model_dump"):
                usage = usage.model_dump()

            result = {
                "status": "completed",
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "model": getattr(response, "model", self.settings.openai_model),
                "response_id": getattr(response, "id", None),
                "usage": usage,
                "feedback": parsed.model_dump(),
            }
            self.storage.add_event("INFO", "openai_feedback", result)
            self.storage.set_kv(
                "openai_feedback_last_sent_at", datetime.now(timezone.utc).isoformat()
            )
            return result
        except Exception as exc:
            logger.exception("OpenAI feedback analysis failed")
            result = {
                "status": "error",
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "error": str(exc),
            }
            self.storage.add_event("ERROR", "openai_feedback_failed", result)
            return result
