from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .config import Settings
from .mt5_client import Mt5Client, Mt5Error

FX_RESEARCH_INSTRUMENTS: tuple[str, ...] = (
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
)

DEFAULT_RESEARCH_DATA_DIR = "data/fx_research"
DEFAULT_CANDLE_COUNT = 3200


def _instrument_path(data_dir: str | Path, instrument: str) -> Path:
    return Path(data_dir) / f"{instrument}.json"


def _load_instrument_file(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    bars = payload.get("bars", [])
    return bars if isinstance(bars, list) else []


def _merge_bars(
    existing: list[dict[str, Any]], new: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    by_time = {bar["time"]: bar for bar in existing}
    for bar in new:
        by_time[bar["time"]] = bar
    return [by_time[key] for key in sorted(by_time)]


def sync_history(
    client: Mt5Client,
    data_dir: str | Path = DEFAULT_RESEARCH_DATA_DIR,
    instruments: tuple[str, ...] = FX_RESEARCH_INSTRUMENTS,
    count: int = DEFAULT_CANDLE_COUNT,
) -> dict[str, dict[str, Any]]:
    """Fetch D1 history for the given instruments and persist it locally as JSON.

    No manual CSV authoring is required: history is fetched from MT5 and
    merged into the existing dedicated research data directory, which is
    isolated from the live/paper-forward sqlite state.
    """
    root = Path(data_dir)
    root.mkdir(parents=True, exist_ok=True)
    candles_by_instrument, errors_by_instrument = client.candles_batch(
        instruments, count=count
    )

    results: dict[str, dict[str, Any]] = {}
    synced_at = datetime.now(timezone.utc).isoformat()
    for instrument in instruments:
        path = _instrument_path(root, instrument)
        if instrument not in candles_by_instrument:
            results[instrument] = {
                "synced": False,
                "error": errors_by_instrument.get(
                    instrument, "Instrument missing from MT5 batch response"
                ),
                "count": len(_load_instrument_file(path)),
            }
            continue

        candles = candles_by_instrument[instrument]
        new_bars = candles.to_dict(orient="records")
        merged = _merge_bars(_load_instrument_file(path), new_bars)
        path.write_text(
            json.dumps(
                {
                    "instrument": instrument,
                    "synced_at": synced_at,
                    "bars": merged,
                },
                ensure_ascii=False,
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )
        results[instrument] = {
            "synced": True,
            "count": len(merged),
            "first_time": merged[0]["time"] if merged else None,
            "last_time": merged[-1]["time"] if merged else None,
        }
    return results


def load_history(
    data_dir: str | Path = DEFAULT_RESEARCH_DATA_DIR,
    instruments: tuple[str, ...] = FX_RESEARCH_INSTRUMENTS,
) -> dict[str, pd.DataFrame]:
    root = Path(data_dir)
    history: dict[str, pd.DataFrame] = {}
    for instrument in instruments:
        bars = _load_instrument_file(_instrument_path(root, instrument))
        if not bars:
            continue
        history[instrument] = (
            pd.DataFrame(bars).sort_values("time").reset_index(drop=True)
        )
    return history


def data_status(
    data_dir: str | Path = DEFAULT_RESEARCH_DATA_DIR,
    instruments: tuple[str, ...] = FX_RESEARCH_INSTRUMENTS,
) -> dict[str, Any]:
    root = Path(data_dir)
    per_symbol: dict[str, Any] = {}
    last_updates: list[datetime] = []
    for instrument in instruments:
        path = _instrument_path(root, instrument)
        if not path.exists():
            per_symbol[instrument] = {
                "available": False,
                "count": 0,
                "first_time": None,
                "last_time": None,
                "updated_at": None,
            }
            continue
        bars = _load_instrument_file(path)
        updated_at = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        last_updates.append(updated_at)
        per_symbol[instrument] = {
            "available": bool(bars),
            "count": len(bars),
            "first_time": bars[0]["time"] if bars else None,
            "last_time": bars[-1]["time"] if bars else None,
            "updated_at": updated_at.isoformat(),
        }
    return {
        "data_dir": str(root),
        "instruments": per_symbol,
        "last_update": max(last_updates).isoformat() if last_updates else None,
        "all_synced": (
            all(v["available"] for v in per_symbol.values()) if per_symbol else False
        ),
    }


def _cmd_verify(settings: Settings) -> int:
    client = Mt5Client(settings)
    try:
        info = client.connection_check()
    except Mt5Error as exc:
        print(f"MT5_CONNECTION=FAILED: {exc}")
        return 1
    print(f"MT5_CONNECTION=OK: {json.dumps(info, ensure_ascii=False, default=str)}")
    return 0


def _cmd_sync(settings: Settings, data_dir: str, count: int) -> int:
    client = Mt5Client(settings)
    results = sync_history(client, data_dir=data_dir, count=count)
    print(json.dumps(results, ensure_ascii=False, indent=2, default=str))
    failed = [instrument for instrument, r in results.items() if not r.get("synced")]
    if failed:
        print(f"SYNC_FAILED_INSTRUMENTS={','.join(failed)}")
        return 1
    print("SYNC=OK")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="FX adaptive research data sync")
    parser.add_argument("command", choices=["verify", "sync"])
    parser.add_argument("--data-dir", default=DEFAULT_RESEARCH_DATA_DIR)
    parser.add_argument("--count", type=int, default=DEFAULT_CANDLE_COUNT)
    args = parser.parse_args(argv)

    settings = Settings.from_env()
    if args.command == "verify":
        return _cmd_verify(settings)
    return _cmd_sync(settings, args.data_dir, args.count)


if __name__ == "__main__":
    sys.exit(main())
