from dataclasses import replace
from datetime import datetime, timezone

from app.config import Settings
from app.mt5_client import Mt5Client


def sample_rates():
    return [
        {
            "time": int(datetime(2026, 7, 1, tzinfo=timezone.utc).timestamp()),
            "open": 1.0,
            "high": 1.2,
            "low": 0.9,
            "close": 1.1,
            "tick_volume": 10,
        },
        {
            "time": int(datetime(2026, 7, 2, tzinfo=timezone.utc).timestamp()),
            "open": 1.1,
            "high": 1.3,
            "low": 1.0,
            "close": 1.2,
            "tick_volume": 20,
        },
        {
            "time": int(datetime(2026, 7, 3, tzinfo=timezone.utc).timestamp()),
            "open": 1.2,
            "high": 1.4,
            "low": 1.1,
            "close": 1.3,
            "tick_volume": 30,
        },
    ]


def test_rates_to_dataframe_drops_newest_bar():
    candles = Mt5Client.rates_to_dataframe(sample_rates())

    assert len(candles) == 2
    assert candles.iloc[-1]["time"] == "2026-07-02T00:00:00+00:00"
    assert candles.iloc[-1]["close"] == 1.2


def test_candles_uses_configured_history_count(monkeypatch):
    settings = replace(
        Settings.from_env(),
        market_data_candle_count=3200,
    )
    client = Mt5Client(settings)

    class FakeMt5:
        TIMEFRAME_D1 = "D1"

        def __init__(self):
            self.requested_count = None

        def symbol_select(self, instrument, enabled):
            return True

        def copy_rates_from_pos(self, instrument, timeframe, start, count):
            self.requested_count = count
            return sample_rates()

        def shutdown(self):
            return None

        def last_error(self):
            return None

    fake = FakeMt5()
    monkeypatch.setattr(client, "connect", lambda: fake)

    client.candles("USDJPY")

    assert fake.requested_count == 3201


def test_spread_pips_uses_jpy_pip_size(monkeypatch):
    settings = Settings.from_env()
    client = Mt5Client(settings)
    monkeypatch.setattr(client, "_tick", lambda instrument: (150.000, 150.020))

    assert client.spread_pips("USDJPY") == 2.0
