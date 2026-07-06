from datetime import datetime, timezone

from app.mt5_client import Mt5Client


def test_rates_to_dataframe_drops_newest_bar():
    rates = [
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

    candles = Mt5Client.rates_to_dataframe(rates)

    assert len(candles) == 2
    assert candles.iloc[-1]["time"] == "2026-07-02T00:00:00+00:00"
    assert candles.iloc[-1]["close"] == 1.2
