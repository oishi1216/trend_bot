from __future__ import annotations

import json
import math
from dataclasses import replace
from datetime import datetime, timedelta

import pandas as pd

import app.adaptive_strategy as adaptive_strategy
import app.research_backtest as research_backtest
from app.config import Settings
from app.research_backtest import _group_metrics, _score_band, _simulate, run_research
from app.research_dashboard import load_research_dashboard


def _baseline_settings():
    return replace(Settings.from_env(), strategy_profile="adaptive_dual_regime_v1")


def _fast_settings():
    # Mirrors the proven fast-converging EMA parameters used in
    # tests/test_adaptive_strategy.py so entries/exits are deterministic
    # over a few hundred synthetic bars.
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


def _bar(dt, open_, high, low, close):
    return {
        "time": dt.strftime("%Y-%m-%dT00:00:00+00:00"),
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
    }


def _trend_rows(direction, rows, breakout_at, crash_at=None):
    price = 100.0
    data = []
    for index in range(rows):
        drift = 0.20 * direction
        wave = math.sin(index / 6) * 0.04
        price += drift + wave
        data.append(
            {
                "open": price - 0.10 * direction,
                "high": price + 0.25,
                "low": price - 0.25,
                "close": price,
            }
        )

    prior_high = max(row["high"] for row in data[:breakout_at])
    prior_low = min(row["low"] for row in data[:breakout_at])
    if direction > 0:
        data[breakout_at]["close"] = prior_high + 1.0
        data[breakout_at]["high"] = data[breakout_at]["close"] + 0.2
        data[breakout_at]["low"] = data[breakout_at]["close"] - 0.4
    else:
        data[breakout_at]["close"] = prior_low - 1.0
        data[breakout_at]["low"] = data[breakout_at]["close"] - 0.2
        data[breakout_at]["high"] = data[breakout_at]["close"] + 0.4

    if crash_at is not None:
        entry_close = data[breakout_at]["close"]
        if direction > 0:
            data[crash_at]["low"] = entry_close - 80.0
            data[crash_at]["close"] = data[crash_at]["low"] + 0.1
            data[crash_at]["open"] = entry_close - 1.0
            data[crash_at]["high"] = entry_close
        else:
            data[crash_at]["high"] = entry_close + 80.0
            data[crash_at]["close"] = data[crash_at]["high"] - 0.1
            data[crash_at]["open"] = entry_close + 1.0
            data[crash_at]["low"] = entry_close

        # Flatten everything after the crash so the underlying trend
        # formula's continuation cannot re-gap back up/down and trigger
        # spurious follow-on entries; only the engineered crash-exit matters.
        flat_price = data[crash_at]["close"]
        for idx in range(crash_at + 1, rows):
            data[idx]["open"] = flat_price
            data[idx]["high"] = flat_price + 0.1
            data[idx]["low"] = flat_price - 0.1
            data[idx]["close"] = flat_price
    return data


def _frame(rows_data, start=datetime(2020, 1, 1)):
    out = []
    d = start
    for row in rows_data:
        out.append(_bar(d, row["open"], row["high"], row["low"], row["close"]))
        d += timedelta(days=1)
    return pd.DataFrame(out)


def _multi_instrument_candles(rows=300, breakout_at=220, crash_at=221):
    usdjpy = _trend_rows(1, rows, breakout_at, crash_at)
    eurusd = _trend_rows(-1, rows, breakout_at, crash_at=None)
    gbpusd = _trend_rows(-1, rows, breakout_at, crash_at=None)
    return {
        "USDJPY": _frame(usdjpy),
        "EURUSD": _frame(eurusd),
        "GBPUSD": _frame(gbpusd),
    }


def test_default_settings_preserve_baseline_config():
    settings = Settings.from_env()
    assert settings.adaptive_strength_gap_min == 0.3
    assert settings.adaptive_score_min == 75.0


def test_decide_is_reused_not_reimplemented():
    assert research_backtest.decide is adaptive_strategy.decide
    assert (
        research_backtest.currency_strength_scores
        is adaptive_strategy.currency_strength_scores
    )


def test_simulate_has_no_lookahead():
    settings = _fast_settings()
    full = _multi_instrument_candles(rows=300, breakout_at=220, crash_at=221)
    truncated = {
        instrument: df.iloc[:225].reset_index(drop=True)
        for instrument, df in full.items()
    }

    full_result = _simulate(
        full, settings, initial_equity=1_000_000.0, spread_pips=1.0, slippage_pips=0.3
    )
    truncated_result = _simulate(
        truncated,
        settings,
        initial_equity=1_000_000.0,
        spread_pips=1.0,
        slippage_pips=0.3,
    )

    cutoff = truncated["USDJPY"]["time"].iloc[-1]
    full_before_cutoff = sorted(
        (t for t in full_result["trades"] if t["exit_date"] <= cutoff),
        key=lambda t: (t["instrument"], t["entry_date"]),
    )
    truncated_trades = sorted(
        truncated_result["trades"], key=lambda t: (t["instrument"], t["entry_date"])
    )

    assert truncated_trades  # sanity: the engineered scenario produced a trade
    assert full_before_cutoff == truncated_trades


def test_run_research_reports_required_breakdowns():
    settings = _fast_settings()
    candles = _multi_instrument_candles(rows=300, breakout_at=220, crash_at=221)
    payload = run_research(candles, settings, initial_equity=1_000_000.0)

    for key in ("cagr", "max_dd", "pf", "win_rate", "trades"):
        assert key in payload["metrics"]
    assert isinstance(payload["annual"], dict)
    assert "USDJPY" in payload["by_symbol"]
    assert set(payload["sensitivity"]) == {"base", "cost_x2"}
    assert (
        payload["sensitivity"]["cost_x2"]["cost_pips"]["spread"]
        > payload["sensitivity"]["base"]["cost_pips"]["spread"]
    )
    for split in ("train", "validation", "test"):
        assert "metrics" in payload[split]
    assert "folds" in payload["walk_forward"]
    assert payload["baseline_config"]["adaptive_strength_gap_min"] == 0.3
    assert payload["baseline_config"]["adaptive_score_min"] == 70.0


def test_score_band_uses_configured_thresholds():
    settings = _fast_settings()
    assert _score_band(75.0, settings) == "70-80"
    assert _score_band(85.0, settings) == "80-90"
    assert _score_band(95.0, settings) == ">=90"


def test_group_metrics_by_symbol():
    trades = [
        {"instrument": "USDJPY", "pnl": 100.0, "r": 1.0},
        {"instrument": "USDJPY", "pnl": -50.0, "r": -0.5},
        {"instrument": "EURUSD", "pnl": 30.0, "r": 0.3},
    ]
    grouped = _group_metrics(trades, lambda t: t["instrument"])
    assert grouped["USDJPY"]["trades"] == 2
    assert grouped["EURUSD"]["trades"] == 1


def test_load_research_dashboard_without_history_reports_no_data(tmp_path):
    dashboard = load_research_dashboard(
        str(tmp_path / "fx_research"), str(tmp_path / "cache.json"), _baseline_settings()
    )
    assert dashboard.status["mode"] == "fx_adaptive_research_v1"
    assert dashboard.status["loaded"] is False
    assert dashboard.verdict["overall"] == "NO_DATA"


def test_load_research_dashboard_uses_synced_history(tmp_path):
    settings = _fast_settings()
    data_dir = tmp_path / "fx_research"
    data_dir.mkdir()
    candles = _multi_instrument_candles(rows=260, breakout_at=220, crash_at=221)
    for instrument, df in candles.items():
        (data_dir / f"{instrument}.json").write_text(
            json.dumps(
                {
                    "instrument": instrument,
                    "synced_at": "2026-08-16T00:00:00+00:00",
                    "bars": df.to_dict(orient="records"),
                }
            ),
            encoding="utf-8",
        )

    dashboard = load_research_dashboard(
        str(data_dir), str(tmp_path / "cache.json"), settings
    )
    assert dashboard.status["loaded"] is True
    assert dashboard.status["mode"] == "fx_adaptive_research_v1"
    assert dashboard.headline["trades"] >= 0
    assert (tmp_path / "cache.json").exists()
