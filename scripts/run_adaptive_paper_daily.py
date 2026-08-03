from __future__ import annotations

import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from app.config import Settings
from app.engine import TradingEngine
from app.storage import Storage

EXPECTED_INSTRUMENTS = {
    "USDJPY",
    "EURUSD",
    "GBPUSD",
    "AUDUSD",
    "NZDUSD",
    "USDCAD",
    "USDCHF",
    "EURJPY",
    "GBPJPY",
    "AUDJPY",
}


def fail(message: str, exit_code: int) -> int:
    print(f"SAFETY_CHECK_FAILED: {message}")
    return exit_code


def main() -> int:
    settings = Settings.from_env()

    if settings.broker_mode != "mt5_paper":
        return fail(f"unexpected broker mode: {settings.broker_mode}", 3)
    if not settings.adaptive_enabled:
        return fail(f"unexpected strategy profile: {settings.strategy_profile}", 3)
    if not settings.trading_armed:
        return fail("TRADING_ARMED must be true inside this paper-only task", 3)
    if Path(settings.db_path).name != "adaptive_paper.sqlite3":
        return fail(f"unexpected database: {settings.db_path}", 3)
    if set(settings.instruments) != EXPECTED_INSTRUMENTS:
        return fail(f"unexpected instruments: {settings.instruments}", 3)
    if settings.market_data_candle_count != 3200:
        return fail(
            f"unexpected candle count: {settings.market_data_candle_count}", 3
        )

    storage = Storage(settings.db_path, settings.paper_initial_balance)
    engine = TradingEngine(settings, storage)

    started_at = datetime.now(timezone.utc).isoformat()
    summary = engine.run_once(force=False)
    finished_at = datetime.now(timezone.utc).isoformat()

    statuses = Counter()
    opened: list[str] = []
    errors: list[str] = []

    print("=== ADAPTIVE PAPER DAILY RUN ===")
    print(f"started_at={started_at}")
    print(f"finished_at={finished_at}")
    print(f"db={settings.db_path}")
    print(f"mode={settings.broker_mode}")
    print(f"profile={settings.strategy_profile}")
    print(f"armed={settings.trading_armed}")
    print(f"nav={summary['nav']}")
    print(f"drawdown={summary['drawdown']}")
    print(f"monthly_loss={summary['monthly_loss']}")
    print()

    print("=== CURRENCY STRENGTH ===")
    strengths = summary.get("currency_strength", {})
    for currency, score in sorted(
        strengths.items(), key=lambda item: item[1], reverse=True
    ):
        print(f"{currency}: {score:.6f}")
    print()

    print("=== INSTRUMENT RESULTS ===")
    for result in summary["results"]:
        instrument = result["instrument"]
        status = result["status"]
        decision = result.get("decision", {})
        statuses[status] += 1

        print(
            f"{instrument}: status={status}, "
            f"action={decision.get('action', '-')}, "
            f"regime={decision.get('regime', '-')}, "
            f"score={decision.get('score', 0)}"
        )

        reason = decision.get("reason")
        if reason:
            print(f"  reason={reason}")

        if status == "opened":
            opened.append(instrument)
        if status == "error":
            errors.append(
                f"{instrument}: {result.get('error', 'unknown error')}"
            )

    positions = [position.to_dict() for position in storage.get_positions()]

    print()
    print("=== SUMMARY ===")
    for status, count in sorted(statuses.items()):
        print(f"status[{status}]={count}")
    print(f"opened_count={len(opened)}")
    print(f"opened={opened}")
    print(f"errors={len(errors)}")
    print(f"positions_open={len(positions)}")

    report = {
        "started_at": started_at,
        "finished_at": finished_at,
        "mode": summary["mode"],
        "strategy_profile": summary["strategy_profile"],
        "armed": summary["armed"],
        "nav": summary["nav"],
        "drawdown": summary["drawdown"],
        "monthly_loss": summary["monthly_loss"],
        "currency_strength": strengths,
        "status_counts": dict(statuses),
        "opened": opened,
        "errors": errors,
        "positions": positions,
    }

    json_path = os.environ.get("ADAPTIVE_DAILY_JSON_PATH")
    if json_path:
        output_path = Path(json_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        print(f"json_report={output_path}")

    if errors:
        print("DAILY_RUN_FAILED: instrument errors detected")
        return 1
    if len(opened) > 1:
        print(f"DAILY_RUN_FAILED: more than one position opened: {opened}")
        return 2

    print("ADAPTIVE PAPER DAILY RUN PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
