from __future__ import annotations

import json

import pandas as pd
import pytest

from app.config import Settings
from app.fx_research_data import (
    DEFAULT_RESEARCH_DATA_DIR,
    FX_RESEARCH_INSTRUMENTS,
    data_status,
    load_history,
    sync_history,
)
from app.mt5_client import Mt5Client, Mt5Error


def _settings():
    return Settings.from_env()


def _fake_candles(count=5, start_price=100.0):
    rows = []
    price = start_price
    for i in range(count):
        price += 0.5
        rows.append(
            {
                "time": f"2026-01-{i + 1:02d}T00:00:00+00:00",
                "open": price - 0.1,
                "high": price + 0.2,
                "low": price - 0.2,
                "close": price,
                "volume": 100,
            }
        )
    return pd.DataFrame(rows)


def test_fx_research_instruments_matches_ten_pairs():
    assert FX_RESEARCH_INSTRUMENTS == (
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


def test_data_isolated_from_paper_forward_db_path():
    assert DEFAULT_RESEARCH_DATA_DIR != "data/adaptive_paper.sqlite3"
    assert "adaptive_paper" not in DEFAULT_RESEARCH_DATA_DIR


def test_sync_history_writes_json_without_manual_csv(tmp_path, monkeypatch):
    client = Mt5Client(_settings())
    fake = {"USDJPY": _fake_candles(5), "EURUSD": _fake_candles(5)}
    monkeypatch.setattr(
        client, "candles_batch", lambda instruments, count=None: (fake, {})
    )

    results = sync_history(
        client, data_dir=tmp_path, instruments=("USDJPY", "EURUSD"), count=5
    )

    assert results["USDJPY"]["synced"] is True
    assert (tmp_path / "USDJPY.json").exists()
    assert not list(tmp_path.glob("*.csv"))
    payload = json.loads((tmp_path / "USDJPY.json").read_text(encoding="utf-8"))
    assert payload["instrument"] == "USDJPY"
    assert len(payload["bars"]) == 5


def test_sync_history_reports_error_without_deleting_existing_data(tmp_path, monkeypatch):
    client = Mt5Client(_settings())
    monkeypatch.setattr(
        client,
        "candles_batch",
        lambda instruments, count=None: ({"USDJPY": _fake_candles(5)}, {}),
    )
    sync_history(client, data_dir=tmp_path, instruments=("USDJPY", "EURUSD"), count=5)

    monkeypatch.setattr(
        client,
        "candles_batch",
        lambda instruments, count=None: (
            {},
            {"USDJPY": "boom", "EURUSD": "boom"},
        ),
    )
    results = sync_history(
        client, data_dir=tmp_path, instruments=("USDJPY", "EURUSD"), count=5
    )

    assert results["USDJPY"]["synced"] is False
    assert (tmp_path / "USDJPY.json").exists()
    existing = json.loads((tmp_path / "USDJPY.json").read_text(encoding="utf-8"))
    assert len(existing["bars"]) == 5


def test_sync_history_merges_and_extends_history(tmp_path, monkeypatch):
    client = Mt5Client(_settings())
    first = {"USDJPY": _fake_candles(5, start_price=100.0)}
    monkeypatch.setattr(
        client, "candles_batch", lambda instruments, count=None: (first, {})
    )
    sync_history(client, data_dir=tmp_path, instruments=("USDJPY",), count=5)

    second_rows = _fake_candles(5, start_price=100.0)
    extra = pd.DataFrame(
        [
            {
                "time": "2026-01-06T00:00:00+00:00",
                "open": 102.5,
                "high": 103.0,
                "low": 102.0,
                "close": 102.8,
                "volume": 100,
            }
        ]
    )
    second = {"USDJPY": pd.concat([second_rows, extra], ignore_index=True)}
    monkeypatch.setattr(
        client, "candles_batch", lambda instruments, count=None: (second, {})
    )
    results = sync_history(client, data_dir=tmp_path, instruments=("USDJPY",), count=5)

    assert results["USDJPY"]["count"] == 6


def test_load_history_reads_synced_json(tmp_path):
    (tmp_path / "USDJPY.json").write_text(
        json.dumps(
            {
                "instrument": "USDJPY",
                "synced_at": "2026-08-16T00:00:00+00:00",
                "bars": _fake_candles(5).to_dict(orient="records"),
            }
        ),
        encoding="utf-8",
    )
    history = load_history(tmp_path, instruments=("USDJPY", "EURUSD"))
    assert "USDJPY" in history
    assert "EURUSD" not in history
    assert list(history["USDJPY"]["time"]) == sorted(history["USDJPY"]["time"])


def test_data_status_reports_counts_and_last_update(tmp_path):
    (tmp_path / "USDJPY.json").write_text(
        json.dumps(
            {
                "instrument": "USDJPY",
                "synced_at": "2026-08-16T00:00:00+00:00",
                "bars": _fake_candles(5).to_dict(orient="records"),
            }
        ),
        encoding="utf-8",
    )
    status = data_status(tmp_path, instruments=("USDJPY", "EURUSD"))
    assert status["instruments"]["USDJPY"]["count"] == 5
    assert status["instruments"]["EURUSD"]["available"] is False
    assert status["all_synced"] is False
    assert status["last_update"] is not None


def test_mt5_client_connection_check_reports_status(monkeypatch):
    client = Mt5Client(_settings())

    class FakeAccount:
        login = 12345
        server = "OANDA-Demo"

    class FakeTerminal:
        connected = True

    class FakeMt5:
        def account_info(self):
            return FakeAccount()

        def terminal_info(self):
            return FakeTerminal()

        def shutdown(self):
            self.shutdown_called = True

    fake = FakeMt5()
    monkeypatch.setattr(client, "connect", lambda: fake)

    info = client.connection_check()

    assert info["connected"] is True
    assert info["login"] == 12345
    assert info["server"] == "OANDA-Demo"


def test_mt5_client_connection_check_propagates_failure(monkeypatch):
    client = Mt5Client(_settings())

    def _fail():
        raise Mt5Error("MT5 initialize failed")

    monkeypatch.setattr(client, "connect", _fail)

    with pytest.raises(Mt5Error):
        client.connection_check()


def test_mt5_client_candles_batch_collects_per_instrument_errors(monkeypatch):
    client = Mt5Client(_settings())

    class FakeMt5:
        TIMEFRAME_D1 = "D1"

        def symbol_select(self, instrument, enabled):
            return instrument != "BADPAIR"

        def copy_rates_from_pos(self, instrument, timeframe, start, count):
            return [
                {
                    "time": 1700000000 + i * 86400,
                    "open": 1.0,
                    "high": 1.1,
                    "low": 0.9,
                    "close": 1.05,
                    "tick_volume": 10,
                }
                for i in range(count)
            ]

        def shutdown(self):
            return None

        def last_error(self):
            return "mock error"

    fake = FakeMt5()
    monkeypatch.setattr(client, "connect", lambda: fake)

    candles, errors = client.candles_batch(["USDJPY", "BADPAIR"], count=5)

    assert "USDJPY" in candles
    assert "BADPAIR" in errors
