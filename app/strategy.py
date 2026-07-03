from __future__ import annotations

import pandas as pd

from .config import Settings
from .models import Position, StrategyDecision


REQUIRED_COLUMNS = {"time", "open", "high", "low", "close"}


def indicators(candles: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    missing = REQUIRED_COLUMNS - set(candles.columns)
    if missing:
        raise ValueError(f"Missing candle columns: {sorted(missing)}")

    df = candles.copy().sort_values("time").reset_index(drop=True)
    for col in ("open", "high", "low", "close"):
        df[col] = pd.to_numeric(df[col], errors="raise")

    df["ema"] = df["close"].ewm(span=settings.ema_days, adjust=False).mean()
    df["ema_past"] = df["ema"].shift(settings.ema_slope_lookback)

    previous_close = df["close"].shift(1)
    true_range = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - previous_close).abs(),
            (df["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    # Wilder-style ATR.
    df["atr"] = true_range.ewm(
        alpha=1 / settings.atr_days,
        adjust=False,
        min_periods=settings.atr_days,
    ).mean()

    # Shift by one so today's breakout is compared with prior days only.
    df["entry_high"] = (
        df["high"].rolling(settings.entry_channel_days).max().shift(1)
    )
    df["entry_low"] = (
        df["low"].rolling(settings.entry_channel_days).min().shift(1)
    )
    df["exit_high"] = (
        df["high"].rolling(settings.exit_channel_days).max().shift(1)
    )
    df["exit_low"] = (
        df["low"].rolling(settings.exit_channel_days).min().shift(1)
    )
    return df


def decide(
    candles: pd.DataFrame,
    settings: Settings,
    position: Position | None,
) -> StrategyDecision:
    df = indicators(candles, settings)
    minimum = max(
        settings.ema_days + settings.ema_slope_lookback,
        settings.entry_channel_days + 1,
        settings.exit_channel_days + 1,
        settings.atr_days + 1,
    )
    if len(df) < minimum:
        raise ValueError(f"Need at least {minimum} complete daily candles; got {len(df)}")

    row = df.iloc[-1]
    needed = ["ema", "ema_past", "atr", "entry_high", "entry_low", "exit_high", "exit_low"]
    if row[needed].isna().any():
        raise ValueError("Latest indicators contain NaN")

    close = float(row["close"])
    candle_time = str(row["time"])
    atr = float(row["atr"])
    ema = float(row["ema"])

    if position:
        if position.side == "long" and close < float(row["exit_low"]):
            return StrategyDecision(
                "exit", candle_time, close, atr, ema, "Long exit: close below prior 20-day low"
            )
        if position.side == "short" and close > float(row["exit_high"]):
            return StrategyDecision(
                "exit", candle_time, close, atr, ema, "Short exit: close above prior 20-day high"
            )
        return StrategyDecision("hold", candle_time, close, atr, ema, "Existing position held")

    trend_up = close > ema and ema > float(row["ema_past"])
    trend_down = close < ema and ema < float(row["ema_past"])

    if trend_up and close > float(row["entry_high"]):
        return StrategyDecision(
            "enter_long", candle_time, close, atr, ema,
            "Long entry: rising 200-day EMA and close above prior 55-day high",
        )
    if trend_down and close < float(row["entry_low"]):
        return StrategyDecision(
            "enter_short", candle_time, close, atr, ema,
            "Short entry: falling 200-day EMA and close below prior 55-day low",
        )
    return StrategyDecision("none", candle_time, close, atr, ema, "No valid entry")
