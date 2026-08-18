from __future__ import annotations

from app.research_failure_analysis import _excursions, _group_metrics, analyze_failure
from tests.test_research import _fast_settings, _frame, _multi_instrument_candles


def test_failure_analysis_reuses_simulator_and_records_entry_gap():
    payload = analyze_failure(_multi_instrument_candles(rows=300, breakout_at=220, crash_at=221), _fast_settings())
    assert payload["baseline"]["metrics"]["trades"] == len(payload["trades"])
    assert payload["trades"]
    assert all("strength_gap" in trade for trade in payload["trades"])
    assert set(payload["by"]["period"]) == {"train", "validation", "test"}


def test_excursions_are_directional_and_holding_days_are_deterministic():
    frame = _frame([
        {"open": 100, "high": 102, "low": 99, "close": 101},
        {"open": 101, "high": 104, "low": 100, "close": 103},
        {"open": 103, "high": 104, "low": 98, "close": 99},
    ])
    long_trade = {"entry_date": frame.iloc[0]["time"], "exit_date": frame.iloc[2]["time"], "entry_price": 100, "initial_stop_price": 98, "side": "long", "pnl": -1}
    short_trade = {**long_trade, "side": "short"}
    long = _excursions(long_trade, frame)
    short = _excursions(short_trade, frame)
    assert long["holding_trading_days"] == short["holding_trading_days"] == 2
    assert long["mfe_r"] == 2.0
    assert short["mfe_r"] == 1.0
    assert long["positive_excursion_closed_negative"] is True


def test_group_metrics_reconcile_loss_contribution_and_tiny_sample():
    grouped = _group_metrics([
        {"instrument": "A", "pnl": -10, "r": -1},
        {"instrument": "A", "pnl": 5, "r": 0.5},
        {"instrument": "B", "pnl": -5, "r": -0.5},
    ], "instrument")
    assert grouped["A"]["gross_loss"] + grouped["B"]["gross_loss"] == 15
    assert grouped["A"]["total_loss_contribution"] == 10 / 15
    assert grouped["B"]["median_r"] == -0.5


def test_repair_outputs_split_cost_score_holding_and_all_mechanisms():
    payload = analyze_failure(_multi_instrument_candles(rows=300, breakout_at=220, crash_at=221), _fast_settings())
    assert {item["id"] for item in payload["mechanism_evaluation"]} == set("ABCDEFGHI")
    assert "inspect_by_split" not in str(payload)
    assert set(payload["cost_sensitivity"]) == {"BASE", "HIGH_COST", "DELTA"}
    assert payload["score_summary"]["count"] == len(payload["trades"])
    assert payload["holding_time"]["winner"]["summary"]["count"] >= 0
    assert all("risk_fraction" in trade for trade in payload["trades"])
    assert all("train" in item and "validation" in item and "test" in item for item in payload["hypotheses"])
