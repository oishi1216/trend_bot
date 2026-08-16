from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .instruments import base_currency, quote_currency
from .models import Position, StrategyDecision

REQUIRED_COLUMNS = {"time", "open", "high", "low", "close"}


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _true_range(df: pd.DataFrame) -> pd.Series:
    previous_close = df["close"].shift(1)
    return pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - previous_close).abs(),
            (df["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)


def indicators(candles: pd.DataFrame, settings: Any) -> pd.DataFrame:
    missing = REQUIRED_COLUMNS - set(candles.columns)
    if missing:
        raise ValueError(f"Missing candle columns: {sorted(missing)}")

    df = candles.copy().sort_values("time").reset_index(drop=True)
    for column in ("open", "high", "low", "close"):
        df[column] = pd.to_numeric(df[column], errors="raise")

    fast = settings.adaptive_fast_ema_days
    middle = settings.adaptive_mid_ema_days
    slow = settings.adaptive_slow_ema_days
    adx_days = settings.adaptive_adx_days
    breakout_days = settings.adaptive_breakout_days
    bollinger_days = settings.adaptive_bollinger_days

    df["ema_fast"] = df["close"].ewm(span=fast, adjust=False).mean()
    df["ema_mid"] = df["close"].ewm(span=middle, adjust=False).mean()
    df["ema_slow"] = df["close"].ewm(span=slow, adjust=False).mean()
    df["ema_mid_past"] = df["ema_mid"].shift(settings.adaptive_slope_lookback)

    true_range = _true_range(df)
    df["atr"] = true_range.ewm(
        alpha=1 / adx_days,
        adjust=False,
        min_periods=adx_days,
    ).mean()

    up_move = df["high"].diff()
    down_move = -df["low"].diff()
    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0.0),
        index=df.index,
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0.0),
        index=df.index,
    )
    plus_smoothed = plus_dm.ewm(
        alpha=1 / adx_days, adjust=False, min_periods=adx_days
    ).mean()
    minus_smoothed = minus_dm.ewm(
        alpha=1 / adx_days, adjust=False, min_periods=adx_days
    ).mean()
    denominator = df["atr"].replace(0, np.nan)
    plus_di = 100 * plus_smoothed / denominator
    minus_di = 100 * minus_smoothed / denominator
    di_sum = (plus_di + minus_di).replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / di_sum
    df["adx"] = dx.ewm(
        alpha=1 / adx_days, adjust=False, min_periods=adx_days
    ).mean()

    df["breakout_high"] = df["high"].rolling(breakout_days).max().shift(1)
    df["breakout_low"] = df["low"].rolling(breakout_days).min().shift(1)
    df["trail_high"] = df["high"].rolling(10).max().shift(1)
    df["trail_low"] = df["low"].rolling(10).min().shift(1)

    df["bb_mean"] = df["close"].rolling(bollinger_days).mean()
    df["bb_std"] = df["close"].rolling(bollinger_days).std(ddof=0)
    df["zscore"] = (df["close"] - df["bb_mean"]) / df["bb_std"].replace(0, np.nan)
    df["band_width"] = (4 * df["bb_std"]) / df["bb_mean"].replace(0, np.nan)

    daily_vol = df["close"].pct_change().rolling(20).std(ddof=0)
    for lookback in (5, 20, 60):
        raw_return = np.log(df["close"] / df["close"].shift(lookback))
        df[f"momentum_{lookback}"] = raw_return / (
            daily_vol * math.sqrt(lookback)
        ).replace(0, np.nan)

    return df


def pair_momentum(candles: pd.DataFrame, settings: Any) -> float:
    df = indicators(candles, settings)
    row = df.iloc[-1]
    values = [row["momentum_5"], row["momentum_20"], row["momentum_60"]]
    if any(pd.isna(value) for value in values):
        return 0.0
    return float(values[0] * 0.25 + values[1] * 0.40 + values[2] * 0.35)


def currency_strength_scores(
    candles_by_instrument: Mapping[str, pd.DataFrame], settings: Any
) -> dict[str, float]:
    totals: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)
    for instrument, candles in candles_by_instrument.items():
        momentum = pair_momentum(candles, settings)
        base = base_currency(instrument)
        quote = quote_currency(instrument)
        totals[base] += momentum
        totals[quote] -= momentum
        counts[base] += 1
        counts[quote] += 1
    return {
        currency: totals[currency] / counts[currency]
        for currency in totals
        if counts[currency]
    }


def strength_gap(instrument: str, strengths: Mapping[str, float]) -> float:
    return float(
        strengths.get(base_currency(instrument), 0.0)
        - strengths.get(quote_currency(instrument), 0.0)
    )


def _band_width_percentile(df: pd.DataFrame) -> float:
    latest = float(df.iloc[-1]["band_width"])
    history = df["band_width"].dropna().tail(252)
    if history.empty or not math.isfinite(latest):
        return 0.5
    return float((history <= latest).mean())


def _risk_for_score(
    score: float, regime: str, volatility_multiplier: float, settings: Any
) -> float:
    if score >= settings.adaptive_score_high:
        base_risk = settings.adaptive_risk_high
    elif score >= settings.adaptive_score_medium:
        base_risk = settings.adaptive_risk_medium
    elif score >= settings.adaptive_score_min:
        base_risk = settings.adaptive_risk_low
    else:
        return 0.0
    if regime == "range":
        base_risk = min(base_risk, settings.adaptive_range_risk_cap)
    return base_risk * volatility_multiplier


def _score_trend(
    *,
    row: pd.Series,
    direction: int,
    gap: float,
    entry_kind: str,
    settings: Any,
) -> float:
    aligned_gap = direction * gap
    strength = _clamp(aligned_gap / 2.5, 0.0, 1.0) * 25
    adx = float(row["adx"])
    regime = _clamp((adx - settings.adaptive_trend_adx) / 17.0, 0.0, 1.0) * 20
    shape = 20.0 if entry_kind == "breakout" else 17.0
    alignment = 15.0
    momentum = direction * float(row["momentum_20"])
    persistence = _clamp(momentum / 2.0, 0.0, 1.0) * 10
    cost_allowance = 10.0
    return strength + regime + shape + alignment + persistence + cost_allowance


def _score_range(
    *, row: pd.Series, direction: int, gap: float, settings: Any
) -> float:
    zscore = abs(float(row["zscore"]))
    excursion = _clamp(
        (zscore - settings.adaptive_range_z) / 1.2, 0.0, 1.0
    )
    shape = 15.0 + excursion * 5.0
    adx = float(row["adx"])
    regime = _clamp(
        (settings.adaptive_range_adx - adx) / settings.adaptive_range_adx,
        0.0,
        1.0,
    ) * 20
    conflict = max(0.0, -(direction * gap) - 1.0)
    strength = max(0.0, 20.0 - conflict * 8.0)
    alignment = 10.0
    timing = 15.0
    cost_allowance = 10.0
    return strength + regime + shape + alignment + timing + cost_allowance


def _position_bars(
    df: pd.DataFrame, metadata: Mapping[str, Any] | None
) -> int:
    if not metadata:
        return 0
    opened_candle_time = metadata.get("opened_candle_time")
    if not opened_candle_time:
        return 0
    matches = df.index[
        df["time"].astype(str) == str(opened_candle_time)
    ].tolist()
    if not matches:
        return 0
    return max(0, len(df) - 1 - matches[-1])


def _manage_position(
    df: pd.DataFrame,
    position: Position,
    metadata: Mapping[str, Any] | None,
) -> StrategyDecision:
    row = df.iloc[-1]
    close = float(row["close"])
    atr = float(row["atr"])
    candle_time = str(row["time"])
    ema = float(row["ema_slow"])
    direction = 1 if position.side == "long" else -1
    initial_stop = float(
        (metadata or {}).get("initial_stop_price", position.stop_price)
    )
    initial_risk = abs(position.entry_price - initial_stop)
    reward_r = (
        0.0
        if initial_risk <= 0
        else direction * (close - position.entry_price) / initial_risk
    )
    regime = str((metadata or {}).get("regime", "trend"))
    bars_held = _position_bars(df, metadata)

    updated_stop: float | None = None
    if reward_r >= 1.0:
        breakeven = position.entry_price
        updated_stop = (
            max(position.stop_price, breakeven)
            if direction > 0
            else min(position.stop_price, breakeven)
        )
    if reward_r >= 1.5:
        atr_trail = close - 2.5 * atr if direction > 0 else close + 2.5 * atr
        channel_trail = float(
            row["trail_low"] if direction > 0 else row["trail_high"]
        )
        candidate = (
            max(atr_trail, channel_trail)
            if direction > 0
            else min(atr_trail, channel_trail)
        )
        updated_stop = (
            max(position.stop_price, candidate)
            if direction > 0
            else min(position.stop_price, candidate)
        )

    if regime == "range":
        mean = float(row["bb_mean"])
        mean_reached = close >= mean if direction > 0 else close <= mean
        if mean_reached or bars_held >= 5:
            return StrategyDecision(
                "exit",
                candle_time,
                close,
                atr,
                ema,
                "Adaptive range exit: mean reached or five-day time stop",
                regime=regime,
                updated_stop_price=updated_stop,
                metadata={"reward_r": reward_r, "bars_held": bars_held},
            )
    else:
        channel_broken = (
            close < float(row["trail_low"])
            if direction > 0
            else close > float(row["trail_high"])
        )
        stale = (
            bars_held >= 20
            and reward_r < 0.5
            and float(row["adx"]) < 20
        )
        if channel_broken or stale:
            return StrategyDecision(
                "exit",
                candle_time,
                close,
                atr,
                ema,
                "Adaptive trend exit: 10-day channel break or stale trend",
                regime=regime,
                updated_stop_price=updated_stop,
                metadata={"reward_r": reward_r, "bars_held": bars_held},
            )

    return StrategyDecision(
        "hold",
        candle_time,
        close,
        atr,
        ema,
        "Adaptive position held",
        regime=regime,
        updated_stop_price=updated_stop,
        metadata={"reward_r": reward_r, "bars_held": bars_held},
    )


def decide(
    candles: pd.DataFrame,
    settings: Any,
    position: Position | None,
    pair_strength_gap: float = 0.0,
    position_metadata: Mapping[str, Any] | None = None,
) -> StrategyDecision:
    df = indicators(candles, settings)
    minimum = max(
        settings.adaptive_slow_ema_days + settings.adaptive_slope_lookback,
        settings.adaptive_breakout_days + 2,
        settings.adaptive_bollinger_days + 2,
        70,
    )
    if len(df) < minimum:
        raise ValueError(
            f"Need at least {minimum} complete daily candles; got {len(df)}"
        )

    row = df.iloc[-1]
    previous = df.iloc[-2]
    needed = [
        "ema_fast",
        "ema_mid",
        "ema_slow",
        "ema_mid_past",
        "atr",
        "adx",
        "breakout_high",
        "breakout_low",
        "bb_mean",
        "zscore",
        "momentum_20",
        "momentum_60",
        "trail_high",
        "trail_low",
    ]
    if row[needed].isna().any():
        raise ValueError("Latest adaptive indicators contain NaN")

    if position:
        return _manage_position(df, position, position_metadata)

    close = float(row["close"])
    atr = float(row["atr"])
    candle_time = str(row["time"])
    ema_fast = float(row["ema_fast"])
    ema_mid = float(row["ema_mid"])
    ema_slow = float(row["ema_slow"])
    ema_mid_past = float(row["ema_mid_past"])
    adx = float(row["adx"])
    momentum_20 = float(row["momentum_20"])
    momentum_60 = float(row["momentum_60"])
    band_percentile = _band_width_percentile(df)

    trend_distance_ok = abs(ema_mid - ema_slow) >= 0.8 * atr
    momentum_agrees = momentum_20 * momentum_60 > 0
    slope_direction = 1 if ema_mid > ema_mid_past else -1
    momentum_direction = 1 if momentum_20 > 0 else -1
    is_trend = (
        adx >= settings.adaptive_trend_adx
        and trend_distance_ok
        and momentum_agrees
        and slope_direction == momentum_direction
    )
    is_range = (
        adx <= settings.adaptive_range_adx
        and abs(close - ema_slow) <= 1.5 * atr
        and 0.15 <= band_percentile <= 0.55
        and abs(momentum_20) <= 1.75
    )

    action = "none"
    regime = "unclear"
    entry_kind: str | None = None
    direction = 0
    stop_multiple: float | None = None
    score = 0.0

    if is_trend:
        regime = "trend"
        strength_gap_ok = abs(pair_strength_gap) >= settings.adaptive_strength_gap_min
        long_alignment = (
            strength_gap_ok
            and ema_fast > ema_mid > ema_slow
            and pair_strength_gap > 0
        )
        short_alignment = (
            strength_gap_ok
            and ema_fast < ema_mid < ema_slow
            and pair_strength_gap < 0
        )
        long_breakout = (
            long_alignment and close > float(row["breakout_high"])
        )
        short_breakout = (
            short_alignment and close < float(row["breakout_low"])
        )
        long_pullback = (
            long_alignment
            and float(row["low"]) <= ema_fast + 0.5 * atr
            and close > float(previous["high"])
            and close > ema_slow
        )
        short_pullback = (
            short_alignment
            and float(row["high"]) >= ema_fast - 0.5 * atr
            and close < float(previous["low"])
            and close < ema_slow
        )
        if long_breakout or short_breakout or long_pullback or short_pullback:
            direction = 1 if long_breakout or long_pullback else -1
            action = "enter_long" if direction > 0 else "enter_short"
            entry_kind = (
                "breakout"
                if long_breakout or short_breakout
                else "pullback"
            )
            stop_multiple = 2.2 if entry_kind == "breakout" else 1.7
            score = _score_trend(
                row=row,
                direction=direction,
                gap=pair_strength_gap,
                entry_kind=entry_kind,
                settings=settings,
            )
    elif is_range:
        regime = "range"
        zscore = float(row["zscore"])
        long_reversal = (
            zscore <= -settings.adaptive_range_z
            and close > float(previous["close"])
            and pair_strength_gap > -1.5
        )
        short_reversal = (
            zscore >= settings.adaptive_range_z
            and close < float(previous["close"])
            and pair_strength_gap < 1.5
        )
        if long_reversal or short_reversal:
            direction = 1 if long_reversal else -1
            action = "enter_long" if direction > 0 else "enter_short"
            entry_kind = "mean_reversion"
            stop_multiple = 1.3
            score = _score_range(
                row=row,
                direction=direction,
                gap=pair_strength_gap,
                settings=settings,
            )

    atr_history = df["atr"].dropna().tail(60)
    median_atr = float(atr_history.median()) if not atr_history.empty else atr
    volatility_multiplier = (
        _clamp(median_atr / atr, 0.5, 1.0) if atr > 0 else 0.5
    )
    risk_fraction = _risk_for_score(
        score, regime, volatility_multiplier, settings
    )

    if (
        action != "none"
        and (
            score < settings.adaptive_score_min
            or risk_fraction <= 0
        )
    ):
        action = "none"
        entry_kind = None
        stop_multiple = None

    reason = (
        f"Adaptive {regime} {entry_kind or 'no-entry'}; "
        f"score={score:.1f}, strength_gap={pair_strength_gap:.2f}, "
        f"ADX={adx:.1f}, vol_multiplier={volatility_multiplier:.2f}"
    )
    return StrategyDecision(
        action,
        candle_time,
        close,
        atr,
        ema_slow,
        reason,
        score=score,
        regime=regime,
        entry_kind=entry_kind,
        stop_atr_multiple=stop_multiple,
        risk_fraction=risk_fraction if action != "none" else None,
        metadata={
            "strength_gap": pair_strength_gap,
            "adx": adx,
            "zscore": float(row["zscore"]),
            "band_width_percentile": band_percentile,
            "volatility_multiplier": volatility_multiplier,
        },
    )
