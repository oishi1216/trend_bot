from dataclasses import replace

import numpy as np
import pandas as pd

from app.adaptive_strategy import currency_strength_scores, decide
from app.config import Settings


def adaptive_settings():
    return replace(
        Settings.from_env(),
        strategy_profile="adaptive_dual_regime_v1",
        adaptive_slow_ema_days=120,
        adaptive_mid_ema_days=40,
        adaptive_fast_ema_days=15,
        adaptive_slope_lookback=8,
        adaptive_strength_gap_min=0.3,
        adaptive_score_min=70.0,
        adaptive_score_medium=80.0,
        adaptive_score_high=90.0,
    )


def trend_candles(direction=1, rows=260):
    data = []
    price = 100.0
    for index in range(rows):
        drift = 0.20 * direction
        wave = np.sin(index / 6) * 0.04
        price += drift + wave
        data.append(
            {
                "time": f"2025-{index:03d}",
                "open": price - 0.10 * direction,
                "high": price + 0.25,
                "low": price - 0.25,
                "close": price,
            }
        )
    if direction > 0:
        prior_high = max(row["high"] for row in data[:-1])
        data[-1]["close"] = prior_high + 1.0
        data[-1]["high"] = data[-1]["close"] + 0.2
        data[-1]["low"] = data[-1]["close"] - 0.4
    else:
        prior_low = min(row["low"] for row in data[:-1])
        data[-1]["close"] = prior_low - 1.0
        data[-1]["low"] = data[-1]["close"] - 0.2
        data[-1]["high"] = data[-1]["close"] + 0.4
    return pd.DataFrame(data)


def test_adaptive_trend_breakout_signal():
    decision = decide(
        trend_candles(1),
        adaptive_settings(),
        None,
        pair_strength_gap=2.0,
    )
    assert decision.action == "enter_long"
    assert decision.regime == "trend"
    assert decision.entry_kind == "breakout"
    assert decision.score >= 70
    assert 0 < decision.risk_fraction <= 0.008
    assert decision.stop_atr_multiple == 2.2


def test_adaptive_blocks_weak_cross_currency_alignment():
    decision = decide(
        trend_candles(1),
        adaptive_settings(),
        None,
        pair_strength_gap=-0.29,
    )
    assert decision.action == "none"


def test_adaptive_allows_strength_gap_at_new_threshold():
    decision = decide(
        trend_candles(1),
        adaptive_settings(),
        None,
        pair_strength_gap=0.3,
    )
    assert decision.action == "enter_long"


def test_currency_strength_aggregates_base_and_quote():
    settings = adaptive_settings()
    strengths = currency_strength_scores(
        {
            "EURUSD": trend_candles(1),
            "USDJPY": trend_candles(1),
            "EURJPY": trend_candles(1),
        },
        settings,
    )
    assert strengths["EUR"] > strengths["JPY"]
