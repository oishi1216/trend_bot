from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from app.config import Settings
from app import research_breakeven_causal_decomposition as diagnostic


def trade(
    instrument: str,
    *,
    side: str = "long",
    entry_date: str = "2024-01-01T00:00:00+00:00",
    exit_date: str = "2024-01-02T00:00:00+00:00",
    pnl: float = 0.0,
    r: float = 0.0,
    exit_reason: str = "hard_stop",
) -> dict:
    return {
        "instrument": instrument,
        "side": side,
        "entry_date": entry_date,
        "exit_date": exit_date,
        "pnl": pnl,
        "r": r,
        "exit_reason": exit_reason,
    }


def simulation(trades: list[dict], dates: list[str]) -> dict:
    return {
        "trades": trades,
        "dates": dates,
        "candidate_dates": [],
        "open_positions_before_date": {date: [] for date in dates},
    }


def classification_period(
    *,
    matched_share: float | None,
    path_share: float | None,
    matched_delta: float,
    path_delta: float,
    full_profit_delta: float,
    trade_delta: int = 1,
    treatment_only_trade_share: float | None = 1.0,
    reconcile: bool = True,
) -> dict:
    return {
        "contributions": {
            "matched_abs_profit_share": matched_share,
            "path_abs_profit_share": path_share,
            "matched_profit_delta": matched_delta,
            "path_divergence_net_profit": path_delta,
            "full_profit_delta": full_profit_delta,
            "trade_count_delta": trade_delta,
            "treatment_only_vs_abs_trade_delta_share": treatment_only_trade_share,
        },
        "reconciliation": {"all_reconcile": reconcile},
    }


def make_classification_periods(full: dict, validation: dict, test: dict) -> dict:
    return {
        "full": full,
        "train": full,
        "validation": validation,
        "test": test,
    }


def test_exact_experiment4_settings_are_reused() -> None:
    control, treatment, diff = diagnostic.build_diagnostic_settings(Settings.from_env())

    assert control.adaptive_breakeven_trigger_r == 1.0
    assert treatment.adaptive_breakeven_trigger_r == 0.5
    assert diff == [
        {
            "field": "adaptive_breakeven_trigger_r",
            "control": 1.0,
            "treatment": 0.5,
        }
    ]
    assert control.strategy_profile == "adaptive_dual_regime_v1"
    assert treatment.strategy_profile == "adaptive_dual_regime_v1"
    assert control.adaptive_fast_ema_days == treatment.adaptive_fast_ema_days == 20
    assert control.adaptive_mid_ema_days == treatment.adaptive_mid_ema_days == 50
    assert control.adaptive_slow_ema_days == treatment.adaptive_slow_ema_days == 200
    assert control.adaptive_score_min == treatment.adaptive_score_min == 75.0
    assert control.adaptive_max_open_positions == treatment.adaptive_max_open_positions == 2
    assert control.adaptive_strength_gap_min == treatment.adaptive_strength_gap_min == 0.3


def test_entry_identity_and_duplicate_fail_closed() -> None:
    item = trade("EURUSD")
    assert diagnostic.entry_identity(item) == (
        "EURUSD",
        "long",
        "2024-01-01T00:00:00+00:00",
    )
    with pytest.raises(ValueError, match="duplicate entry identity"):
        diagnostic.index_unique_trades([item, dict(item)])


def test_partition_reconciles_matched_and_only_entries() -> None:
    a = trade("EURUSD", pnl=10, r=1)
    b = trade("GBPUSD", entry_date="2024-01-03T00:00:00+00:00", pnl=-5, r=-0.5)
    a_treatment = dict(a, exit_date="2024-01-04T00:00:00+00:00", pnl=20, r=2)
    c = trade("AUDUSD", entry_date="2024-01-05T00:00:00+00:00", pnl=2, r=0.2)

    result = diagnostic.partition_trades([a, b], [a_treatment, c])

    assert len(result["matched"]) == 1
    assert [x["instrument"] for x in result["control_only"]] == ["GBPUSD"]
    assert [x["instrument"] for x in result["treatment_only"]] == ["AUDUSD"]
    assert result["reconciliation"]["all_reconcile"] is True


def test_treatment_only_association_priority() -> None:
    same_date = "2024-01-03T00:00:00+00:00"
    candidate_date = "2024-01-04T00:00:00+00:00"
    other_date = "2024-01-05T00:00:00+00:00"
    control = {
        "open_positions_before_date": {
            same_date: ["EURUSD", "GBPUSD"],
            candidate_date: ["EURUSD"],
            other_date: [],
        },
        "candidate_dates": [same_date, candidate_date],
    }

    same = trade("EURUSD", entry_date=same_date)
    capacity = trade("AUDUSD", entry_date=same_date)
    candidate = trade("AUDUSD", entry_date=candidate_date)
    other = trade("AUDUSD", entry_date=other_date)

    assert (
        diagnostic.classify_treatment_only_association(same, control, 2)
        == "control_same_instrument_open"
    )
    assert (
        diagnostic.classify_treatment_only_association(capacity, control, 2)
        == "control_capacity_full"
    )
    assert (
        diagnostic.classify_treatment_only_association(candidate, control, 2)
        == "control_candidate_day_state_divergence"
    )
    assert (
        diagnostic.classify_treatment_only_association(other, control, 2)
        == "other_state_divergence"
    )


def test_trade_metrics_matches_research_pf_semantics() -> None:
    assert diagnostic.trade_metrics([]) == {
        "trades": 0,
        "pf": 0.0,
        "avg_r": 0.0,
        "profit": 0,
        "win_rate": 0.0,
    }

    winners = [trade("EURUSD", pnl=10, r=1.0)]
    assert diagnostic.trade_metrics(winners)["pf"] is None

    mixed = [
        trade("EURUSD", pnl=10, r=1.0),
        trade(
            "GBPUSD",
            entry_date="2024-01-03T00:00:00+00:00",
            pnl=-5,
            r=-0.5,
        ),
    ]
    metrics = diagnostic.trade_metrics(mixed)
    assert metrics["pf"] == 2.0
    assert metrics["avg_r"] == 0.25
    assert metrics["profit"] == 5.0
    assert metrics["win_rate"] == 0.5


def test_outcome_direction_uses_one_e_minus_nine_epsilon() -> None:
    control = trade("EURUSD", r=1.0)
    assert diagnostic.outcome_direction(control, dict(control, r=1.0 + 1e-10)) == "unchanged"
    assert diagnostic.outcome_direction(control, dict(control, r=1.0 + 2e-9)) == "improved"
    assert diagnostic.outcome_direction(control, dict(control, r=1.0 - 2e-9)) == "worsened"


def test_period_date_sets_reuses_existing_split_semantics() -> None:
    dates = [f"2024-01-{day:02d}" for day in range(1, 11)]
    expected_train, expected_validation, expected_test = diagnostic._split_dates(dates)
    actual = diagnostic.period_date_sets(dates)

    assert actual["full"] is None
    assert actual["train"] == expected_train
    assert actual["validation"] == expected_validation
    assert actual["test"] == expected_test
    assert (len(expected_train), len(expected_validation), len(expected_test)) == (6, 2, 2)


def test_decomposition_contribution_arithmetic_and_reconciliation() -> None:
    dates = [f"2024-01-{day:02d}T00:00:00+00:00" for day in range(1, 11)]
    a = trade("EURUSD", exit_date=dates[2], pnl=10, r=1)
    b = trade("GBPUSD", entry_date=dates[1], exit_date=dates[3], pnl=-5, r=-0.5)
    a_treatment = dict(a, exit_date=dates[4], pnl=20, r=2)
    c = trade("AUDUSD", entry_date=dates[5], exit_date=dates[6], pnl=2, r=0.2)

    control = simulation([a, b], dates)
    treatment = simulation([a_treatment, c], dates)
    partition = diagnostic.partition_trades(control["trades"], treatment["trades"])

    result = diagnostic.decompose_period(
        period_dates=None,
        partition=partition,
        control_simulation=control,
        treatment_simulation=treatment,
        max_open_positions=2,
    )

    assert result["reconciliation"]["all_reconcile"] is True
    assert result["contributions"]["full_profit_delta"] == 17.0
    assert result["contributions"]["matched_profit_delta"] == 10.0
    assert result["contributions"]["path_divergence_net_profit"] == 7.0
    assert result["contributions"]["interaction_residual"] == 0.0
    assert result["contributions"]["trade_count_delta"] == 0
    assert result["contributions"]["path_trade_count_abs_share_net"] is None
    assert result["contributions"]["treatment_only_vs_abs_trade_delta_share"] is None


def test_path_divergence_dominant_classification() -> None:
    full = classification_period(
        matched_share=0.2,
        path_share=0.8,
        matched_delta=2,
        path_delta=8,
        full_profit_delta=10,
        treatment_only_trade_share=1.0,
    )
    validation = classification_period(
        matched_share=0.1,
        path_share=0.9,
        matched_delta=1,
        path_delta=3,
        full_profit_delta=4,
    )
    test = classification_period(
        matched_share=0.1,
        path_share=0.9,
        matched_delta=0,
        path_delta=0,
        full_profit_delta=1,
    )

    label, _ = diagnostic.classify_diagnostic(
        make_classification_periods(full, validation, test)
    )
    assert label == "PATH_DIVERGENCE_DOMINANT"


def test_matched_outcome_dominant_classification() -> None:
    full = classification_period(
        matched_share=0.8,
        path_share=0.2,
        matched_delta=8,
        path_delta=2,
        full_profit_delta=10,
        treatment_only_trade_share=0.2,
    )
    validation = classification_period(
        matched_share=0.8,
        path_share=0.2,
        matched_delta=2,
        path_delta=1,
        full_profit_delta=3,
    )
    test = classification_period(
        matched_share=0.8,
        path_share=0.2,
        matched_delta=0,
        path_delta=0,
        full_profit_delta=1,
    )

    label, _ = diagnostic.classify_diagnostic(
        make_classification_periods(full, validation, test)
    )
    assert label == "MATCHED_OUTCOME_DOMINANT"


def test_mixed_classification_when_both_profit_mechanisms_material() -> None:
    full = classification_period(
        matched_share=0.5,
        path_share=0.5,
        matched_delta=5,
        path_delta=5,
        full_profit_delta=10,
    )
    validation = classification_period(
        matched_share=0.5,
        path_share=0.5,
        matched_delta=1,
        path_delta=-1,
        full_profit_delta=1,
    )
    test = classification_period(
        matched_share=0.5,
        path_share=0.5,
        matched_delta=1,
        path_delta=1,
        full_profit_delta=2,
    )

    label, _ = diagnostic.classify_diagnostic(
        make_classification_periods(full, validation, test)
    )
    assert label == "MIXED"


def test_insufficient_evidence_for_zero_denominators() -> None:
    zero = classification_period(
        matched_share=None,
        path_share=None,
        matched_delta=0,
        path_delta=0,
        full_profit_delta=0,
        trade_delta=0,
        treatment_only_trade_share=None,
    )
    label, _ = diagnostic.classify_diagnostic(
        make_classification_periods(zero, zero, zero)
    )
    assert label == "INSUFFICIENT_EVIDENCE"


def test_run_diagnostic_calls_simulator_exactly_once_per_arm(monkeypatch: pytest.MonkeyPatch) -> None:
    dates = [f"2024-01-{day:02d}T00:00:00+00:00" for day in range(1, 6)]
    control_trade = trade("EURUSD", exit_date=dates[2], pnl=-10, r=-1)
    treatment_trade = dict(control_trade, exit_date=dates[1], pnl=0, r=0)
    calls: list[float] = []

    def fake_simulate(_candles, settings, **_kwargs):
        calls.append(settings.adaptive_breakeven_trigger_r)
        trades = [control_trade] if settings.adaptive_breakeven_trigger_r == 1.0 else [treatment_trade]
        return simulation(trades, dates)

    monkeypatch.setattr(diagnostic, "_simulate", fake_simulate)
    payload = diagnostic.run_diagnostic({}, Settings.from_env())

    assert calls == [1.0, 0.5]
    assert payload["research_only"] is True
    assert payload["experiment_4_classification"] == "INCONCLUSIVE"
    assert payload["settings_diff"][0]["field"] == "adaptive_breakeven_trigger_r"


def test_diagnostic_module_has_no_forward_execution_imports() -> None:
    source_path = Path(diagnostic.__file__)
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.append(node.module or "")

    forbidden = ("app.paper", "app.engine", "app.main", "app.mt5_client")
    assert not any(name.startswith(forbidden) for name in imported)


def test_report_output_is_deterministic(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dates = [f"2024-01-{day:02d}T00:00:00+00:00" for day in range(1, 6)]
    control_trade = trade("EURUSD", exit_date=dates[2], pnl=-10, r=-1)
    treatment_trade = dict(control_trade, exit_date=dates[1], pnl=0, r=0)

    def fake_simulate(_candles, settings, **_kwargs):
        trades = [control_trade] if settings.adaptive_breakeven_trigger_r == 1.0 else [treatment_trade]
        return simulation(trades, dates)

    monkeypatch.setattr(diagnostic, "_simulate", fake_simulate)
    payload = diagnostic.run_diagnostic({}, Settings.from_env())

    json_a = tmp_path / "a.json"
    md_a = tmp_path / "a.md"
    json_b = tmp_path / "b.json"
    md_b = tmp_path / "b.md"
    diagnostic.write_report(payload, json_a, md_a)
    diagnostic.write_report(payload, json_b, md_b)

    assert json_a.read_bytes() == json_b.read_bytes()
    assert md_a.read_bytes() == md_b.read_bytes()
    loaded = json.loads(json_a.read_text(encoding="utf-8"))
    assert loaded["classification"] in diagnostic.DIAGNOSTIC_LABELS
