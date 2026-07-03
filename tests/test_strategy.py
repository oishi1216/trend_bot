from dataclasses import replace

import pandas as pd

from app.config import Settings
from app.strategy import decide


def settings():
    base = Settings.from_env()
    return replace(
        base,
        ema_days=20,
        ema_slope_lookback=5,
        entry_channel_days=10,
        exit_channel_days=5,
        atr_days=5,
    )


def test_long_breakout_signal():
    rows = []
    price = 100.0
    for i in range(40):
        price += 0.2
        rows.append({"time": f"2025-01-{i+1:02d}", "open": price-0.1, "high": price+0.1, "low": price-0.2, "close": price})
    rows[-1]["close"] = max(r["high"] for r in rows[:-1]) + 1.0
    rows[-1]["high"] = rows[-1]["close"] + 0.1
    decision = decide(pd.DataFrame(rows), settings(), None)
    assert decision.action == "enter_long"
