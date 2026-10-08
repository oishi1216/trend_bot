from __future__ import annotations

import hashlib
import json
import math
import subprocess
from pathlib import Path
from typing import Any, Mapping

RECOVERY_SCHEMA_VERSION = 1
RECOVERY_STATE_KEY = "drawdown_recovery_state_v1"
RECOVERY_INITIALIZED_KEY = "drawdown_recovery_initialized_v1"
RECOVERY_INITIALIZED_VALUE = "1"
RECOVERY_LATCH_DRAWDOWN_STOP_V1 = 0.10
RECOVERY_LATCH_DRAWDOWN_STOP_TEXT = "0.1"
RECOVERY_SQLITE_BUSY_TIMEOUT_MS = 5000
RECOVERY_GOVERNED_DB_BASENAME = "adaptive_paper.sqlite3"

STATE_ACTIVE = "ACTIVE"
STATE_LATCHED_DD_STOP = "LATCHED_DD_STOP"
STATE_REBASELINED_PENDING_REARM = "REBASELINED_PENDING_REARM"
STATE_RECOVERY_PROBATION = "RECOVERY_PROBATION"
KNOWN_STATES = {
    STATE_ACTIVE,
    STATE_LATCHED_DD_STOP,
    STATE_REBASELINED_PENDING_REARM,
    STATE_RECOVERY_PROBATION,
}

EVENT_BOOTSTRAP = "drawdown_recovery_bootstrap"
EVENT_LATCHED = "drawdown_recovery_latched"
RECOVERY_EVENT_PREFIX = "drawdown_recovery_"


class RecoveryStateError(ValueError):
    pass


def canonical_float_text(value: float) -> str:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise RecoveryStateError("recovery numeric value must be finite")
    return repr(parsed)


def parse_positive_float_text(value: str | None, *, name: str) -> float:
    if value is None:
        raise RecoveryStateError(f"{name} is missing")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise RecoveryStateError(f"{name} is not a float") from exc
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise RecoveryStateError(f"{name} must be finite and > 0")
    return parsed


def canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(
        dict(payload),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def governed_db_basename(path: str | Path) -> bool:
    return Path(path).name.casefold() == RECOVERY_GOVERNED_DB_BASENAME.casefold()


def is_governed_db(
    path: str | Path,
    *,
    state_present: bool,
    marker_present: bool,
    event_present: bool = False,
) -> bool:
    return (
        governed_db_basename(path)
        or state_present
        or marker_present
        or event_present
    )


def calculate_drawdown(nav: float, high_water: float) -> float:
    nav_value = float(nav)
    high_value = float(high_water)
    if not math.isfinite(nav_value) or nav_value <= 0.0:
        raise RecoveryStateError("NAV must be finite and > 0")
    if not math.isfinite(high_value) or high_value <= 0.0:
        raise RecoveryStateError("high-water must be finite and > 0")
    return max(0.0, 1.0 - nav_value / high_value)
def recovery_latch_reached(nav: float, high_water: float) -> bool:
    nav_value = float(nav)
    high_value = float(high_water)
    if not math.isfinite(nav_value) or nav_value <= 0.0:
        raise RecoveryStateError("NAV must be finite and > 0")
    if not math.isfinite(high_value) or high_value <= 0.0:
        raise RecoveryStateError("high-water must be finite and > 0")
    recovery_boundary = high_value * (1.0 - RECOVERY_LATCH_DRAWDOWN_STOP_V1)
    return nav_value <= recovery_boundary


def active_state(*, epoch: int = 0) -> dict[str, Any]:
    return {
        "schema_version": RECOVERY_SCHEMA_VERSION,
        "epoch": int(epoch),
        "state": STATE_ACTIVE,
        "latch_drawdown_stop_text": RECOVERY_LATCH_DRAWDOWN_STOP_TEXT,
        "trip_id": None,
        "tripped_at": None,
        "trip_nav": None,
        "trip_high_water": None,
        "trip_drawdown": None,
        "rebaseline_receipt_sha256": None,
        "rearmed_at": None,
    }


def trip_identity_payload(
    *,
    epoch: int,
    tripped_at: str,
    nav: float,
    high_water: float,
    drawdown: float,
) -> dict[str, Any]:
    return {
        "schema_version": RECOVERY_SCHEMA_VERSION,
        "epoch": int(epoch),
        "latch_drawdown_stop_text": RECOVERY_LATCH_DRAWDOWN_STOP_TEXT,
        "tripped_at": str(tripped_at),
        "nav": canonical_float_text(nav),
        "high_water": canonical_float_text(high_water),
        "drawdown": canonical_float_text(drawdown),
    }


def stable_trip_id(
    *,
    epoch: int,
    tripped_at: str,
    nav: float,
    high_water: float,
    drawdown: float,
) -> str:
    payload = trip_identity_payload(
        epoch=epoch,
        tripped_at=tripped_at,
        nav=nav,
        high_water=high_water,
        drawdown=drawdown,
    )
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def latched_state(
    *,
    epoch: int,
    tripped_at: str,
    nav: float,
    high_water: float,
    drawdown: float,
) -> dict[str, Any]:
    trip_id = stable_trip_id(
        epoch=epoch,
        tripped_at=tripped_at,
        nav=nav,
        high_water=high_water,
        drawdown=drawdown,
    )
    return {
        "schema_version": RECOVERY_SCHEMA_VERSION,
        "epoch": int(epoch),
        "state": STATE_LATCHED_DD_STOP,
        "latch_drawdown_stop_text": RECOVERY_LATCH_DRAWDOWN_STOP_TEXT,
        "trip_id": trip_id,
        "tripped_at": str(tripped_at),
        "trip_nav": canonical_float_text(nav),
        "trip_high_water": canonical_float_text(high_water),
        "trip_drawdown": canonical_float_text(drawdown),
        "rebaseline_receipt_sha256": None,
        "rearmed_at": None,
    }


def parse_recovery_state(raw: str) -> dict[str, Any]:
    try:
        payload = json.loads(raw)
    except (TypeError, json.JSONDecodeError) as exc:
        raise RecoveryStateError("recovery state is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise RecoveryStateError("recovery state must be an object")

    required = {
        "schema_version",
        "epoch",
        "state",
        "latch_drawdown_stop_text",
        "trip_id",
        "tripped_at",
        "trip_nav",
        "trip_high_water",
        "trip_drawdown",
        "rebaseline_receipt_sha256",
        "rearmed_at",
    }
    if set(payload) != required:
        raise RecoveryStateError("recovery state field set is invalid")
    if (
        type(payload["schema_version"]) is not int
        or payload["schema_version"] != RECOVERY_SCHEMA_VERSION
    ):
        raise RecoveryStateError("unknown recovery schema version")
    if type(payload["epoch"]) is not int or payload["epoch"] < 0:
        raise RecoveryStateError("recovery epoch is invalid")
    if not isinstance(payload["state"], str) or payload["state"] not in KNOWN_STATES:
        raise RecoveryStateError("unknown recovery state")
    if payload["latch_drawdown_stop_text"] != RECOVERY_LATCH_DRAWDOWN_STOP_TEXT:
        raise RecoveryStateError("recovery latch threshold identity mismatch")

    if payload["state"] == STATE_ACTIVE:
        for key in (
            "trip_id",
            "tripped_at",
            "trip_nav",
            "trip_high_water",
            "trip_drawdown",
            "rebaseline_receipt_sha256",
            "rearmed_at",
        ):
            if payload[key] is not None:
                raise RecoveryStateError(f"active state has unexpected {key}")

    if payload["state"] == STATE_LATCHED_DD_STOP:
        for key in (
            "trip_id",
            "tripped_at",
            "trip_nav",
            "trip_high_water",
            "trip_drawdown",
        ):
            if payload[key] in (None, ""):
                raise RecoveryStateError(f"latched state missing {key}")
        nav = parse_positive_float_text(payload["trip_nav"], name="trip_nav")
        high_water = parse_positive_float_text(
            payload["trip_high_water"], name="trip_high_water"
        )
        try:
            drawdown = float(payload["trip_drawdown"])
        except (TypeError, ValueError) as exc:
            raise RecoveryStateError("trip_drawdown is invalid") from exc
        if not math.isfinite(drawdown) or drawdown < 0.0:
            raise RecoveryStateError("trip_drawdown is invalid")
        expected = stable_trip_id(
            epoch=payload["epoch"],
            tripped_at=payload["tripped_at"],
            nav=nav,
            high_water=high_water,
            drawdown=drawdown,
        )
        if payload["trip_id"] != expected:
            raise RecoveryStateError("trip_id does not match canonical trip identity")
        if payload["rebaseline_receipt_sha256"] is not None:
            raise RecoveryStateError("latched state has unexpected rebaseline receipt")
        if payload["rearmed_at"] is not None:
            raise RecoveryStateError("latched state has unexpected rearmed timestamp")
    return payload


def state_entries_blocked(state: Mapping[str, Any] | None) -> bool:
    return state is None or state.get("state") != STATE_ACTIVE


def state_block_reason(
    state: Mapping[str, Any] | None,
    *,
    integrity_ok: bool,
    mode_ok: bool,
) -> str | None:
    if not mode_ok:
        return "recovery_mode_mismatch"
    if not integrity_ok or state is None:
        return "recovery_integrity_hold"
    if state.get("state") != STATE_ACTIVE:
        return f"recovery_state:{state.get('state')}"
    return None


def resolve_source_commit(repo_root: str | Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    value = completed.stdout.strip()
    if len(value) != 40 or any(char not in "0123456789abcdefABCDEF" for char in value):
        return None
    return value.lower()