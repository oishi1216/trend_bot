from __future__ import annotations

import ast
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from app.config import Settings
from app.models import StrategyDecision
from app import research_candidate_day_divergence_audit as audit


def decision(
    action: str,
    *,
    score: float,
    risk_fraction: float = 0.005,
    regime: str = "trend",
    entry_kind: str = "breakout",
) -> StrategyDecision:
    return StrategyDecision(
        action=action,
        candle_time="2024-01-01",
        close=1.1,
        atr=0.01,
        ema=1.0,
        reason="test",
        score=score,
        regime=regime,
        entry_kind=entry_kind,
        stop_atr_multiple=2.0,
        risk_fraction=risk_fraction,
    )


def ledger_trade(
    instrument: str = "EURUSD",
    *,
    side: str = "long",
    entry_date: str = "2024-01-01",
    exit_date: str = "2024-01-03",
    risk_dollars: float = 1000.0,
    r_value: float = 0.5,
    entry_close: float = 1.1000,
    initial_stop: float = 1.0900,
) -> dict:
    cost = audit._adverse_cost(
        instrument,
        audit.DEFAULT_SPREAD_PIPS,
        audit.DEFAULT_SLIPPAGE_PIPS,
    )
    if side == "long":
        entry_price = entry_close + cost
        direction = 1.0
    else:
        entry_price = entry_close - cost
        direction = -1.0
    stop_distance = abs(entry_close - initial_stop)
    price_per_risk_unit = risk_dollars / stop_distance
    pnl = risk_dollars * r_value
    exit_price = entry_price + (pnl / price_per_risk_unit) * direction
    return {
        "instrument": instrument,
        "side": side,
        "entry_date": entry_date,
        "exit_date": exit_date,
        "entry_price": entry_price,
        "initial_stop_price": initial_stop,
        "exit_price": exit_price,
        "pnl": pnl,
        "r": r_value,
        "exit_reason": "test",
    }


def adaptive_settings() -> Settings:
    return audit.build_audit_settings(Settings.from_env())[0]


def selected_candidate(
    instrument: str = "EURUSD",
    *,
    side: str = "long",
    score: float = 90.0,
    risk_fraction: float = 0.005,
    nav_snapshot: float = 100_000.0,
) -> dict:
    return {
        "original_index": 0,
        "instrument": instrument,
        "action": "enter_long" if side == "long" else "enter_short",
        "side": side,
        "score": score,
        "risk_fraction": risk_fraction,
        "regime": "trend",
        "entry_kind": "breakout",
        "nav_snapshot": nav_snapshot,
        "rank": 1,
    }


def test_exact_experiment4_settings_only_1r_vs_05r() -> None:
    control, treatment, diff = audit.build_audit_settings(Settings.from_env())
    assert control.adaptive_breakeven_trigger_r == 1.0
    assert treatment.adaptive_breakeven_trigger_r == 0.5
    assert diff == [
        {
            "field": "adaptive_breakeven_trigger_r",
            "control": 1.0,
            "treatment": 0.5,
        }
    ]


def test_candidate_rank_preserves_first_tie_wins() -> None:
    candidates = [
        ("EURUSD", decision("enter_long", score=90), 90.0, 100_000.0),
        ("GBPUSD", decision("enter_short", score=90), 90.0, 100_000.0),
        ("AUDUSD", decision("enter_long", score=80), 80.0, 100_000.0),
    ]
    rows = audit._serialize_candidates(candidates)
    assert [row["rank"] for row in rows] == [1, 2, 3]
    assert audit.research_backtest._select_highest_candidate(candidates) is candidates[0]


def test_selector_trace_preserves_result_and_restores(monkeypatch: pytest.MonkeyPatch) -> None:
    original = audit.research_backtest._select_highest_candidate

    def fake_simulate(*_args, **_kwargs):
        candidates = [
            ("EURUSD", decision("enter_long", score=90), 90.0, 100_000.0),
            ("GBPUSD", decision("enter_short", score=90), 90.0, 100_000.0),
        ]
        selected = audit.research_backtest._select_highest_candidate(candidates)
        assert selected is candidates[0]
        return {
            "candidate_dates": ["2024-01-02"],
            "trades": [],
            "dates": ["2024-01-02"],
            "diagnostics": {},
        }

    monkeypatch.setattr(audit.research_backtest, "_simulate", fake_simulate)
    result = audit.simulate_with_candidate_trace(
        {},
        adaptive_settings(),
        initial_equity=1_000_000.0,
    )
    assert result["trace_by_date"]["2024-01-02"]["selected"]["instrument"] == "EURUSD"
    assert result["selector_restored"] is True
    assert audit.research_backtest._select_highest_candidate is original


def test_selector_restored_after_simulator_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    original = audit.research_backtest._select_highest_candidate

    def fake_simulate(*_args, **_kwargs):
        candidates = [
            ("EURUSD", decision("enter_long", score=90), 90.0, 100_000.0)
        ]
        audit.research_backtest._select_highest_candidate(candidates)
        raise RuntimeError("boom")

    monkeypatch.setattr(audit.research_backtest, "_simulate", fake_simulate)
    with pytest.raises(RuntimeError, match="boom"):
        audit.simulate_with_candidate_trace(
            {}, adaptive_settings(), initial_equity=1_000_000.0
        )
    assert audit.research_backtest._select_highest_candidate is original


def test_trace_candidate_date_mismatch_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_simulate(*_args, **_kwargs):
        candidates = [
            ("EURUSD", decision("enter_long", score=90), 90.0, 100_000.0)
        ]
        audit.research_backtest._select_highest_candidate(candidates)
        return {"candidate_dates": [], "trades": [], "dates": [], "diagnostics": {}}

    monkeypatch.setattr(audit.research_backtest, "_simulate", fake_simulate)
    with pytest.raises(RuntimeError, match="trace call count"):
        audit.simulate_with_candidate_trace(
            {}, adaptive_settings(), initial_equity=1_000_000.0
        )


def test_recover_trade_runtime_reproduces_risk_and_pnl() -> None:
    trade = ledger_trade(risk_dollars=1000.0, r_value=0.5)
    runtime = audit.recover_trade_runtime(trade)
    assert runtime["risk_dollars"] == pytest.approx(1000.0)
    assert runtime["stop_distance"] == pytest.approx(0.01)
    assert runtime["price_per_risk_unit"] == pytest.approx(100_000.0)


def test_recover_trade_runtime_zero_r_fails_closed() -> None:
    trade = ledger_trade(risk_dollars=1000.0, r_value=0.5)
    trade["r"] = 0.0
    trade["pnl"] = 0.0
    with pytest.raises(ValueError, match="zero/non-finite R"):
        audit.recover_trade_runtime(trade)


def test_daily_state_reconstructs_nav_highwater_month_and_dd() -> None:
    trade = ledger_trade(
        entry_date="2024-01-01",
        exit_date="2024-01-03",
        risk_dollars=1000.0,
        r_value=0.5,
    )
    candles = {
        "EURUSD": pd.DataFrame(
            {
                "time": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"],
                "close": [1.1000, 1.1020, 1.1050, 1.1050],
            }
        )
    }
    simulation = {
        "trades": [trade],
        "dates": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"],
    }
    state = audit.reconstruct_daily_state(
        candles,
        simulation,
        initial_equity=1_000_000.0,
    )

    assert state["2024-01-01"]["current_nav"] == pytest.approx(1_000_000.0)
    assert state["2024-01-02"]["mark_to_market_at_date_start"] == pytest.approx(192.0)
    assert state["2024-01-02"]["current_nav"] == pytest.approx(1_000_192.0)
    assert state["2024-01-03"]["mark_to_market_at_date_start"] == pytest.approx(492.0)
    assert state["2024-01-03"]["high_water"] == pytest.approx(1_000_492.0)
    assert state["2024-01-04"]["realized_equity_before_date_exits"] == pytest.approx(1_000_500.0)
    assert state["2024-01-04"]["current_nav"] == pytest.approx(1_000_500.0)
    assert state["2024-01-04"]["monthly_loss"] == 0.0
    assert state["2024-01-04"]["drawdown"] == 0.0


def _gate_state(monthly_loss: float = 0.0, drawdown: float = 0.0) -> dict:
    return {
        "2024-01-02": {
            "monthly_loss": monthly_loss,
            "drawdown": drawdown,
            "realized_equity_before_date_exits": 100_000.0,
            "mark_to_market_at_date_start": 0.0,
            "current_nav": 100_000.0,
            "high_water": 100_000.0,
            "month_start_nav": 100_000.0,
            "open_at_date_start": [],
            "open_at_gate": [],
        }
    }


def test_gate_order_monthly_then_dd(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = adaptive_settings()
    selected = selected_candidate()
    simulation = {"trades": []}
    result = audit.evaluate_selected_candidate_gate(
        selected,
        "2024-01-02",
        settings,
        simulation,
        _gate_state(monthly_loss=0.06, drawdown=0.11),
    )
    assert result["reason"] == "blocked_monthly_loss"

    result = audit.evaluate_selected_candidate_gate(
        selected,
        "2024-01-02",
        settings,
        simulation,
        _gate_state(monthly_loss=0.0, drawdown=0.11),
    )
    assert result["reason"] == "blocked_dd_stop"


def test_gate_order_max_positions_before_aggregate(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = adaptive_settings()
    active = [
        {"instrument": "GBPUSD", "entry_date": "2024-01-01", "exit_date": "2024-01-03"},
        {"instrument": "AUDUSD", "entry_date": "2024-01-01", "exit_date": "2024-01-03"},
    ]
    monkeypatch.setattr(
        audit,
        "recover_trade_runtime",
        lambda _trade: {"risk_dollars": 1000.0},
    )
    result = audit.evaluate_selected_candidate_gate(
        selected_candidate(risk_fraction=0.02),
        "2024-01-02",
        settings,
        {"trades": active},
        _gate_state(),
    )
    assert result["reason"] == "blocked_max_positions"


def test_dd_risk_reduction_changes_downstream_aggregate_check(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = replace(
        adaptive_settings(),
        adaptive_max_open_positions=5,
        adaptive_max_aggregate_risk=0.006,
        adaptive_max_single_currency_risk=1.0,
    )
    active = [
        {"instrument": "GBPUSD", "entry_date": "2024-01-01", "exit_date": "2024-01-03"}
    ]
    monkeypatch.setattr(
        audit,
        "recover_trade_runtime",
        lambda _trade: {"risk_dollars": 100.0},
    )
    result = audit.evaluate_selected_candidate_gate(
        selected_candidate(instrument="AUDUSD", risk_fraction=0.008, nav_snapshot=100_000.0),
        "2024-01-02",
        settings,
        {"trades": active},
        _gate_state(drawdown=0.05),
    )
    assert result["adjusted_risk_fraction"] == pytest.approx(0.004)
    assert result["reason"] == "PASS"


def test_aggregate_then_currency_gate_order(monkeypatch: pytest.MonkeyPatch) -> None:
    active = [
        {"instrument": "EURUSD", "entry_date": "2024-01-01", "exit_date": "2024-01-03"}
    ]
    monkeypatch.setattr(
        audit,
        "recover_trade_runtime",
        lambda _trade: {"risk_dollars": 500.0},
    )
    aggregate_settings = replace(
        adaptive_settings(),
        adaptive_max_open_positions=5,
        adaptive_max_aggregate_risk=0.008,
        adaptive_max_single_currency_risk=1.0,
    )
    result = audit.evaluate_selected_candidate_gate(
        selected_candidate(instrument="AUDUSD", risk_fraction=0.005, nav_snapshot=100_000.0),
        "2024-01-02",
        aggregate_settings,
        {"trades": active},
        _gate_state(),
    )
    assert result["reason"] == "blocked_aggregate_risk"

    currency_settings = replace(
        adaptive_settings(),
        adaptive_max_open_positions=5,
        adaptive_max_aggregate_risk=1.0,
        adaptive_max_single_currency_risk=0.008,
    )
    result = audit.evaluate_selected_candidate_gate(
        selected_candidate(instrument="EURJPY", risk_fraction=0.005, nav_snapshot=100_000.0),
        "2024-01-02",
        currency_settings,
        {"trades": active},
        _gate_state(),
    )
    assert result["reason"] == "blocked_currency_risk"


def test_gate_trace_self_check_matches_and_mismatch_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    trace = {
        "2024-01-02": {
            "selected": selected_candidate(),
            "candidates": [selected_candidate()],
        },
        "2024-01-03": {
            "selected": selected_candidate(instrument="GBPUSD"),
            "candidates": [selected_candidate(instrument="GBPUSD")],
        },
    }
    pass_trade = {
        "instrument": "GBPUSD",
        "side": "long",
        "entry_date": "2024-01-03",
    }

    def fake_gate(selected, date, *_args, **_kwargs):
        return {
            "reason": "blocked_dd_stop" if date == "2024-01-02" else "PASS",
        }

    monkeypatch.setattr(audit, "evaluate_selected_candidate_gate", fake_gate)
    simulation = {
        "trades": [pass_trade],
        "candidate_dates": ["2024-01-02", "2024-01-03"],
        "diagnostics": {
            "blocked_monthly_loss": 0,
            "blocked_dd_stop": 1,
            "blocked_max_positions": 0,
            "blocked_aggregate_risk": 0,
            "blocked_currency_risk": 0,
        },
    }
    result = audit.reconstruct_gate_trace(
        trace,
        simulation,
        adaptive_settings(),
        {},
    )
    assert result["block_counts_match"] is True
    assert result["pass_count"] == 1

    simulation["diagnostics"]["blocked_dd_stop"] = 2
    with pytest.raises(RuntimeError, match="block counts"):
        audit.reconstruct_gate_trace(
            trace,
            simulation,
            adaptive_settings(),
            {},
        )


def test_target_derivation_uses_diagnostic5_association() -> None:
    control = {
        "candidate_dates": ["2024-01-02"],
        "open_positions_before_date": {"2024-01-02": []},
    }
    target = {
        "instrument": "EURUSD",
        "side": "long",
        "entry_date": "2024-01-02",
        "exit_date": "2024-01-03",
        "pnl": -10.0,
        "r": -0.1,
    }
    other = {
        "instrument": "GBPUSD",
        "side": "long",
        "entry_date": "2024-01-04",
        "exit_date": "2024-01-05",
        "pnl": 5.0,
        "r": 0.05,
    }
    partition = {"treatment_only": [target, other]}
    targets = audit.derive_target_trades(partition, control, adaptive_settings())
    assert targets == [target]


def _trace_with_candidates(candidates: list[dict], selected: dict | None) -> dict:
    return {"candidates": candidates, "selected": selected}


def test_primary_mechanisms() -> None:
    target = {"instrument": "EURUSD", "side": "long"}
    eur = selected_candidate("EURUSD")
    gbp = selected_candidate("GBPUSD", score=95.0)

    assert (
        audit.classify_primary_mechanism(
            target,
            _trace_with_candidates([gbp], gbp),
            {"reason": "PASS"},
        )
        == "TARGET_NOT_CONTROL_CANDIDATE"
    )
    assert (
        audit.classify_primary_mechanism(
            target,
            _trace_with_candidates([eur], eur),
            {"reason": "blocked_currency_risk"},
        )
        == "TARGET_SELECTED_CONTROL_BUT_BLOCKED"
    )
    assert (
        audit.classify_primary_mechanism(
            target,
            _trace_with_candidates([eur, gbp], gbp),
            {"reason": "PASS"},
        )
        == "TARGET_RANKED_BELOW_CONTROL_SELECTION"
    )


def test_secondary_ranking_associations() -> None:
    eur = selected_candidate("EURUSD", score=90.0)
    gbp = selected_candidate("GBPUSD", score=95.0)
    primary = "TARGET_RANKED_BELOW_CONTROL_SELECTION"

    assert audit.classify_secondary_association(
        primary,
        _trace_with_candidates([eur, gbp], gbp),
        _trace_with_candidates([eur], eur),
        ["GBPUSD"],
    ) == "CONTROL_SELECTED_COMPETITOR_OPEN_IN_TREATMENT"

    assert audit.classify_secondary_association(
        primary,
        _trace_with_candidates([eur, gbp], gbp),
        _trace_with_candidates([eur], eur),
        [],
    ) == "CONTROL_SELECTED_COMPETITOR_NOT_IN_TREATMENT_CANDIDATES"

    assert audit.classify_secondary_association(
        primary,
        _trace_with_candidates([eur, gbp], gbp),
        _trace_with_candidates([eur, gbp], eur),
        [],
    ) == "CONTROL_SELECTED_COMPETITOR_SHARED"


def test_conclusion_thresholds_and_mixed_precedence() -> None:
    label, _ = audit.classify_conclusion(
        {"TARGET_RANKED_BELOW_CONTROL_SELECTION": 7},
        10,
        anchor_valid=True,
        reconstruction_valid=True,
    )
    assert label == "RANKING_DISPLACEMENT_DOMINANT"

    label, _ = audit.classify_conclusion(
        {
            "TARGET_RANKED_BELOW_CONTROL_SELECTION": 6,
            "TARGET_SELECTED_CONTROL_BUT_BLOCKED": 3,
        },
        10,
        anchor_valid=True,
        reconstruction_valid=True,
    )
    assert label == "MIXED"

    label, _ = audit.classify_conclusion(
        {"TARGET_SELECTED_CONTROL_BUT_BLOCKED": 6},
        10,
        anchor_valid=True,
        reconstruction_valid=True,
    )
    assert label == "TARGET_GATE_BLOCK_DOMINANT"

    label, _ = audit.classify_conclusion(
        {"TARGET_NOT_CONTROL_CANDIDATE": 6},
        10,
        anchor_valid=True,
        reconstruction_valid=True,
    )
    assert label == "TARGET_CANDIDATE_ABSENCE_DOMINANT"

    label, _ = audit.classify_conclusion(
        {"TARGET_RANKED_BELOW_CONTROL_SELECTION": 5},
        10,
        anchor_valid=False,
        reconstruction_valid=True,
    )
    assert label == "INCONCLUSIVE"


def test_summary_metrics_are_deterministic() -> None:
    records = [
        {
            "primary_mechanism": "TARGET_RANKED_BELOW_CONTROL_SELECTION",
            "secondary_association": "CONTROL_SELECTED_COMPETITOR_OPEN_IN_TREATMENT",
            "control_selected_competitor_open_in_treatment": True,
            "control_selected_gate": {"reason": "PASS"},
            "control_selected_minus_target_score": 5.0,
            "treatment_pnl": -10.0,
            "treatment_r": -0.1,
        },
        {
            "primary_mechanism": "TARGET_RANKED_BELOW_CONTROL_SELECTION",
            "secondary_association": "CONTROL_SELECTED_COMPETITOR_SHARED",
            "control_selected_competitor_open_in_treatment": False,
            "control_selected_gate": {"reason": "blocked_currency_risk"},
            "control_selected_minus_target_score": 2.0,
            "treatment_pnl": 5.0,
            "treatment_r": 0.05,
        },
    ]
    summary = audit.summarize_records(records)
    assert summary["ranking_count"] == 2
    assert summary["ranking_competitor_open_in_treatment_ratio"] == 0.5
    assert summary["ranking_selected_competitor_blocked_ratio"] == 0.5
    assert summary["score_gap"]["mean"] == pytest.approx(3.5)
    assert summary["metrics_by_primary"]["TARGET_RANKED_BELOW_CONTROL_SELECTION"]["profit"] == -5.0


def test_direct_imports_exclude_forward_paper_live_modules() -> None:
    source_path = Path(audit.__file__)
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            imported.append(module)
    forbidden = ("paper", "engine", "main", "mt5_client")
    assert not any(
        name == item or name.startswith(item + ".")
        for name in imported
        for item in forbidden
    )


def test_json_markdown_output_is_deterministic(tmp_path: Path) -> None:
    payload = {
        "classification": "RANKING_DISPLACEMENT_DOMINANT",
        "classification_rationale": "test",
        "target_anchor": {"matches": True},
        "gate_reconstruction": {
            "control": {"block_counts_match": True},
            "treatment": {"block_counts_match": True},
        },
        "summary": {
            "primary_counts": {},
            "secondary_counts": {},
            "ranking_competitor_open_in_treatment_ratio": 1.0,
            "ranking_selected_competitor_blocked_ratio": 0.0,
            "ranking_selected_competitor_gate_reasons": {"PASS": 1},
            "score_gap": {"count": 1, "min": 1.0, "mean": 1.0, "max": 1.0},
        },
        "split_summary": {
            "validation": {"targets": 0},
            "test": {"targets": 1},
        },
        "target_records": [
            {
                "identity": ["EURUSD", "long", "2024-01-02"],
                "primary_mechanism": "TARGET_RANKED_BELOW_CONTROL_SELECTION",
                "secondary_association": "CONTROL_SELECTED_COMPETITOR_OPEN_IN_TREATMENT",
                "target_control_rank": 2,
                "target_treatment_rank": 1,
                "control_selected_gate": {"reason": "PASS"},
                "treatment_pnl": -10.0,
                "treatment_r": -0.1,
            }
        ],
        "limitations": ["test"],
    }
    json_a = tmp_path / "a.json"
    json_b = tmp_path / "b.json"
    md_a = tmp_path / "a.md"
    md_b = tmp_path / "b.md"
    audit.write_report(payload, json_a, md_a)
    audit.write_report(payload, json_b, md_b)
    assert json_a.read_bytes() == json_b.read_bytes()
    assert md_a.read_bytes() == md_b.read_bytes()
