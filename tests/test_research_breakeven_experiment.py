from __future__ import annotations

import dataclasses
import json
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

import app.research_breakeven_experiment as rbe
from app.adaptive_strategy import _manage_position
from app.config import Settings
from app.models import Position
from app.research_backtest import run_research
from app.research_breakeven_experiment import (
    ANCHOR,
    CONTROL_BREAKEVEN_TRIGGER_R,
    DEFAULT_JSON_PATH,
    DEFAULT_MARKDOWN_PATH,
    FOLD_MIN_TRADES,
    MIN_SUFFICIENT_FOLDS,
    RETENTION_MIN_RATIO,
    SUPPORTED_GATE_NAMES,
    TREATMENT_BREAKEVEN_TRIGGER_R,
    _anchor_comparison,
    _classify,
    _evaluate,
    _exit_diagnostics,
    _metric_direction,
    _retention,
    _walk_forward_evidence,
    build_experiment_settings,
    changed_fields,
    run_experiment,
    settings_diff,
    write_report,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

PINNED = {
    "strategy_profile": "adaptive_dual_regime_v1",
    "adaptive_fast_ema_days": 20,
    "adaptive_mid_ema_days": 50,
    "adaptive_slow_ema_days": 200,
    "adaptive_score_min": 75.0,
    "adaptive_max_open_positions": 2,
    "adaptive_strength_gap_min": 0.3,
}


def _base() -> Settings:
    return replace(Settings.from_env(), strategy_profile="adaptive_dual_regime_v1")


def _metrics(
    trades=10,
    pf=1.0,
    avg_r=0.1,
    cagr=0.01,
    max_dd=0.1,
    profit=100.0,
    win_rate=0.5,
):
    return {
        "trades": trades,
        "pf": pf,
        "avg_r": avg_r,
        "cagr": cagr,
        "max_dd": max_dd,
        "profit": profit,
        "win_rate": win_rate,
    }


def _fold(pf=1.0, trades=5, cagr=0.01):
    return {"metrics": _metrics(trades=trades, pf=pf, cagr=cagr)}


def _arm(
    *,
    full_pf=1.0,
    full_avg_r=0.1,
    full_trades=100,
    full_max_dd=0.1,
    val_pf=1.0,
    val_trades=20,
    test_pf=1.0,
    test_avg_r=0.1,
    test_trades=20,
    test_max_dd=0.1,
    high_pf=1.0,
    fold_positive=3,
    folds=None,
):
    if folds is None:
        folds = [_fold(full_pf, trades=5) for _ in range(5)]
    return {
        "metrics": _metrics(
            trades=full_trades,
            pf=full_pf,
            avg_r=full_avg_r,
            max_dd=full_max_dd,
        ),
        "train": {"metrics": _metrics(trades=60, pf=full_pf, avg_r=full_avg_r)},
        "validation": {"metrics": _metrics(trades=val_trades, pf=val_pf, avg_r=full_avg_r)},
        "test": {
            "metrics": _metrics(
                trades=test_trades,
                pf=test_pf,
                avg_r=test_avg_r,
                max_dd=test_max_dd,
            )
        },
        "sensitivity": {
            "base": {"metrics": _metrics(trades=full_trades, pf=full_pf)},
            "cost_x2": {"metrics": _metrics(trades=full_trades, pf=high_pf)},
        },
        "walk_forward": {
            "folds": list(folds),
            "fold_positive": fold_positive,
            "fold_total": len(folds),
        },
        "baseline_config": {},
        "by_regime": {},
        "by_score_band": {},
    }


def _passing_pair():
    control = _arm(
        full_pf=1.0,
        full_avg_r=0.10,
        val_pf=1.0,
        test_pf=1.0,
        test_avg_r=0.10,
        high_pf=1.0,
        fold_positive=2,
        folds=[_fold(1.0) for _ in range(5)],
    )
    treatment = _arm(
        full_pf=1.2,
        full_avg_r=0.20,
        val_pf=1.2,
        test_pf=1.2,
        test_avg_r=0.11,
        high_pf=1.2,
        fold_positive=3,
        folds=[_fold(1.2) for _ in range(5)],
    )
    return control, treatment


def _evaluation(control, treatment, *, anchor=True, defaults=True, isolation=True):
    return _evaluate(
        control,
        treatment,
        {"matches": anchor, "fields": {}},
        defaults,
        isolation,
    )


def _position(side="long", entry=100.0, stop=90.0):
    return Position(
        instrument="USDJPY",
        side=side,
        units=1,
        entry_price=entry,
        stop_price=stop,
        opened_at="2020-01-01",
        planned_risk_home=1000.0,
    )


def _manage_df(close: float, atr: float = 2.0) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "time": "2020-01-02",
                "close": close,
                "atr": atr,
                "ema_slow": 100.0,
                "trail_low": 50.0,
                "trail_high": 150.0,
                "bb_mean": 100.0,
                "adx": 30.0,
            }
        ]
    )


# Settings / isolation

def test_production_default_breakeven_trigger_is_1_0(monkeypatch):
    monkeypatch.delenv("ADAPTIVE_BREAKEVEN_TRIGGER_R", raising=False)
    assert Settings.from_env().adaptive_breakeven_trigger_r == 1.0


def test_control_1_0_treatment_0_5_and_exactly_one_field_differs():
    control, treatment = build_experiment_settings(_base())
    assert CONTROL_BREAKEVEN_TRIGGER_R == 1.0
    assert TREATMENT_BREAKEVEN_TRIGGER_R == 0.5
    assert control.adaptive_breakeven_trigger_r == 1.0
    assert treatment.adaptive_breakeven_trigger_r == 0.5
    assert changed_fields(control, treatment) == ["adaptive_breakeven_trigger_r"]
    assert settings_diff(control, treatment) == [
        {
            "field": "adaptive_breakeven_trigger_r",
            "control": 1.0,
            "treatment": 0.5,
        }
    ]


def test_research_control_normalizes_adaptive_profile_from_legacy_base():
    legacy = replace(Settings.from_env(), strategy_profile="legacy_trend_v1")
    control, treatment = build_experiment_settings(legacy)
    assert control.strategy_profile == "adaptive_dual_regime_v1"
    assert treatment.strategy_profile == "adaptive_dual_regime_v1"
    assert control.adaptive_enabled is True
    assert treatment.adaptive_enabled is True


def test_prior_experiment_values_do_not_leak():
    drifted = replace(
        _base(),
        adaptive_score_min=85.0,
        adaptive_max_open_positions=1,
        adaptive_fast_ema_days=15,
        adaptive_mid_ema_days=75,
        adaptive_slow_ema_days=120,
        adaptive_strength_gap_min=0.7,
    )
    control, treatment = build_experiment_settings(drifted)
    for settings in (control, treatment):
        for field, expected in PINNED.items():
            assert getattr(settings, field) == expected


def test_build_does_not_mutate_base():
    base = _base()
    before = dataclasses.asdict(base)
    build_experiment_settings(base)
    assert dataclasses.asdict(base) == before


# Existing strategy path / close-based threshold

def test_reuses_run_research_identity():
    assert rbe.run_research is run_research


def test_strategy_uses_close_reward_r_and_no_mfe_mae_feedback():
    source = (REPO_ROOT / "app" / "adaptive_strategy.py").read_text(encoding="utf-8")
    assert "reward_r >= settings.adaptive_breakeven_trigger_r" in source
    assert "reward_r >= 1.5" in source
    assert "mfe" not in source.lower()
    assert "mae" not in source.lower()


def test_default_1_0_regression_below_threshold_does_not_move_stop():
    decision = _manage_position(
        _manage_df(107.5),
        _position(),
        {"initial_stop_price": 90.0, "regime": "trend"},
        replace(_base(), adaptive_breakeven_trigger_r=1.0),
    )
    assert decision.metadata["reward_r"] == pytest.approx(0.75)
    assert decision.updated_stop_price is None


def test_treatment_threshold_below_at_above_is_deterministic():
    settings = replace(_base(), adaptive_breakeven_trigger_r=0.5)
    below = _manage_position(
        _manage_df(104.9),
        _position(),
        {"initial_stop_price": 90.0, "regime": "trend"},
        settings,
    )
    at = _manage_position(
        _manage_df(105.0),
        _position(),
        {"initial_stop_price": 90.0, "regime": "trend"},
        settings,
    )
    above = _manage_position(
        _manage_df(105.1),
        _position(),
        {"initial_stop_price": 90.0, "regime": "trend"},
        settings,
    )
    assert below.updated_stop_price is None
    assert at.updated_stop_price == 100.0
    assert above.updated_stop_price == 100.0


def test_trail_activation_remains_1_5r():
    settings = replace(_base(), adaptive_breakeven_trigger_r=0.5)
    df = _manage_df(115.0, atr=2.0)
    df.loc[0, "trail_low"] = 108.0
    decision = _manage_position(
        df,
        _position(),
        {"initial_stop_price": 90.0, "regime": "trend"},
        settings,
    )
    assert decision.metadata["reward_r"] == pytest.approx(1.5)
    assert decision.updated_stop_price == pytest.approx(110.0)


# Numeric/zero-trade reviewer carry-forward

def test_float_delta_at_or_below_1e_9_is_unchanged():
    control = _metrics(trades=10, pf=1.0)
    treatment = _metrics(trades=10, pf=1.0 + 1e-9)
    assert _metric_direction(control, treatment, "pf") == "unchanged"
    treatment["pf"] = 1.0 + 1.1e-9
    assert _metric_direction(control, treatment, "pf") == "improved"


def test_zero_trade_pf_cannot_create_directional_claim():
    assert _metric_direction(
        _metrics(trades=0, pf=0.0),
        _metrics(trades=0, pf=999.0),
        "pf",
    ) is None


# Retention / high-cost / walk-forward

def test_validation_and_test_retention_floor_is_70_percent():
    assert RETENTION_MIN_RATIO == 0.70
    assert _retention(10, 7)["passes"] is True
    assert _retention(10, 6)["passes"] is False


def test_walk_forward_requires_min_three_trades_and_two_sufficient_folds():
    assert FOLD_MIN_TRADES == 3
    assert MIN_SUFFICIENT_FOLDS == 2
    insufficient = _walk_forward_evidence(
        {"folds": [_fold(1.0, trades=3)], "fold_positive": 1},
        {"folds": [_fold(1.1, trades=3)], "fold_positive": 1},
    )
    assert insufficient["folds_sufficient"] == 1
    assert insufficient["non_contradicting"] is False
    sufficient = _walk_forward_evidence(
        {"folds": [_fold(1.0, trades=3), _fold(1.0, trades=3)], "fold_positive": 1},
        {"folds": [_fold(1.1, trades=3), _fold(0.9, trades=3)], "fold_positive": 1},
    )
    assert sufficient["folds_sufficient"] == 2
    assert sufficient["ratio"] == 0.5
    assert sufficient["non_contradicting"] is True


def test_high_cost_reversal_blocks_supported():
    control, treatment = _passing_pair()
    treatment["sensitivity"]["cost_x2"]["metrics"]["pf"] = 0.9
    evaluation = _evaluation(control, treatment)
    assert evaluation["high_cost_pf_direction_preserved"] is False
    assert _classify(evaluation)[0] == "INCONCLUSIVE"


# Classification

def test_supported_requires_all_predeclared_gates():
    control, treatment = _passing_pair()
    evaluation = _evaluation(control, treatment)
    assert all(evaluation[name] for name in SUPPORTED_GATE_NAMES)
    assert _classify(evaluation)[0] == "SUPPORTED"


def test_mixed_validation_test_is_inconclusive():
    control, treatment = _passing_pair()
    treatment["test"]["metrics"]["pf"] = 0.9
    evaluation = _evaluation(control, treatment)
    assert evaluation["validation_pf_improves"] is True
    assert evaluation["test_pf_improves"] is False
    assert _classify(evaluation)[0] == "INCONCLUSIVE"


def test_falsified_when_adequate_full_pf_and_avg_r_both_worsen_without_robustness_mix():
    control, treatment = _passing_pair()
    treatment["metrics"]["pf"] = 0.8
    treatment["metrics"]["avg_r"] = 0.05
    treatment["sensitivity"]["cost_x2"]["metrics"]["pf"] = 0.8
    treatment["walk_forward"] = {
        "folds": [_fold(0.8, trades=5) for _ in range(5)],
        "fold_positive": 1,
        "fold_total": 5,
    }
    evaluation = _evaluation(control, treatment)
    assert evaluation["sample_adequate"] is True
    assert evaluation["falsified_primary"] is True
    assert evaluation["robustness_mixed"] is False
    assert _classify(evaluation)[0] == "FALSIFIED"


def test_anchor_drift_forces_inconclusive():
    control, treatment = _passing_pair()
    assert _classify(_evaluation(control, treatment, anchor=False))[0] == "INCONCLUSIVE"


def test_classification_is_deterministic():
    control, treatment = _passing_pair()
    first = _classify(_evaluation(control, treatment))
    second = _classify(_evaluation(control, treatment))
    assert first == second


# Anchor

def test_anchor_comparison_matches_authoritative_control():
    payload = {
        "metrics": _metrics(
            trades=int(ANCHOR["trades"]),
            pf=ANCHOR["pf"],
            cagr=ANCHOR["cagr"],
            max_dd=ANCHOR["max_dd"],
            profit=ANCHOR["profit"],
        ),
        "test": {
            "metrics": _metrics(
                trades=int(ANCHOR["test_trades"]),
                pf=ANCHOR["test_pf"],
            )
        },
    }
    assert _anchor_comparison(payload)["matches"] is True


# Exit diagnostics reuse simulator without new semantics

def test_exit_diagnostics_reports_reason_hard_stop_and_zero_r(monkeypatch):
    monkeypatch.setattr(
        rbe,
        "_simulate",
        lambda *args, **kwargs: {
            "trades": [
                {"exit_reason": "hard_stop", "r": 0.0},
                {"exit_reason": "hard_stop", "r": -1.0},
                {"exit_reason": "Adaptive trend exit", "r": 0.2},
            ]
        },
    )
    result = _exit_diagnostics({"USDJPY": pd.DataFrame()}, _base(), 1_000_000.0, 3)
    assert result["exit_reason_counts"] == {
        "Adaptive trend exit": 1,
        "hard_stop": 2,
    }
    assert result["hard_stop_count"] == 2
    assert result["zero_r_exit_count"] == 1
    assert result["breakeven_specific_count"] is None


def test_exit_diagnostics_rejects_trade_count_divergence(monkeypatch):
    monkeypatch.setattr(rbe, "_simulate", lambda *args, **kwargs: {"trades": []})
    with pytest.raises(RuntimeError):
        _exit_diagnostics({"USDJPY": pd.DataFrame()}, _base(), 1_000_000.0, 1)


# Orchestration exactly two run_research calls

def test_run_experiment_calls_run_research_exactly_twice(monkeypatch):
    calls = []

    def fake_run(_candles, settings, initial_equity=1_000_000.0):
        calls.append(settings)
        if settings.adaptive_breakeven_trigger_r == 1.0:
            arm = _arm(
                full_pf=ANCHOR["pf"],
                full_avg_r=-0.1,
                full_trades=int(ANCHOR["trades"]),
                full_max_dd=ANCHOR["max_dd"],
                val_pf=0.8,
                val_trades=20,
                test_pf=ANCHOR["test_pf"],
                test_avg_r=-0.1,
                test_trades=int(ANCHOR["test_trades"]),
                high_pf=0.8,
                fold_positive=2,
                folds=[_fold(0.8, trades=5) for _ in range(5)],
            )
            arm["metrics"]["cagr"] = ANCHOR["cagr"]
            arm["metrics"]["profit"] = ANCHOR["profit"]
            return arm
        return _arm(
            full_pf=1.0,
            full_avg_r=0.1,
            full_trades=120,
            val_pf=1.0,
            val_trades=18,
            test_pf=0.8,
            test_avg_r=0.0,
            test_trades=12,
            high_pf=1.0,
            fold_positive=3,
            folds=[_fold(1.0, trades=5) for _ in range(5)],
        )

    monkeypatch.setattr(rbe, "run_research", fake_run)
    monkeypatch.setattr(
        rbe,
        "_exit_diagnostics",
        lambda *args, **kwargs: {
            "exit_reason_counts": {},
            "hard_stop_count": 0,
            "zero_r_exit_count": 0,
            "breakeven_specific_count": None,
            "breakeven_specific_note": "n/a",
        },
    )
    payload = run_experiment({}, _base())
    assert len(calls) == 2
    assert calls[0].adaptive_breakeven_trigger_r == 1.0
    assert calls[1].adaptive_breakeven_trigger_r == 0.5
    assert changed_fields(calls[0], calls[1]) == ["adaptive_breakeven_trigger_r"]
    assert payload["settings_diff"][0]["field"] == "adaptive_breakeven_trigger_r"


# Deterministic output / output-only / runner safety

def test_report_is_deterministic_and_discloses_holdout_and_equity_limitation(tmp_path):
    control, treatment = _passing_pair()
    evaluation = _evaluation(control, treatment)
    payload = {
        "classification": _classify(evaluation)[0],
        "classification_rationale": _classify(evaluation)[1],
        "settings_diff": [{"field": "adaptive_breakeven_trigger_r", "control": 1.0, "treatment": 0.5}],
        "anchor": {"fields": {}},
        "control": control,
        "treatment": treatment,
        "evaluation": evaluation,
        "control_exit_diagnostics": {
            "exit_reason_counts": {"hard_stop": 1},
            "hard_stop_count": 1,
            "zero_r_exit_count": 0,
            "breakeven_specific_count": None,
            "breakeven_specific_note": "not separately identifiable",
        },
        "treatment_exit_diagnostics": {
            "exit_reason_counts": {"hard_stop": 1},
            "hard_stop_count": 1,
            "zero_r_exit_count": 0,
            "breakeven_specific_count": None,
            "breakeven_specific_note": "not separately identifiable",
        },
        "disclosures": {
            "test_holdout": "Test has already been inspected by prior research and is not a pristine untouched holdout.",
            "equity_curve_limitation": "adaptive-gate blocked dates can omit equity-curve points",
            "causal_input_boundary": "MFE/MAE are diagnostics only",
            "paper_state_verification": "external",
        },
    }
    a_json, a_md = tmp_path / "a.json", tmp_path / "a.md"
    b_json, b_md = tmp_path / "b.json", tmp_path / "b.md"
    write_report(payload, a_json, a_md)
    write_report(payload, b_json, b_md)
    assert a_json.read_text(encoding="utf-8") == b_json.read_text(encoding="utf-8")
    assert a_md.read_text(encoding="utf-8") == b_md.read_text(encoding="utf-8")
    text = a_md.read_text(encoding="utf-8")
    assert "not a pristine untouched holdout" in text
    assert "equity-curve" in text


def test_default_artifacts_are_output_only_under_data_research():
    assert DEFAULT_JSON_PATH == "data/research/adaptive_v2_breakeven_experiment.json"
    assert DEFAULT_MARKDOWN_PATH == "data/research/adaptive_v2_breakeven_experiment.md"


def test_forward_execution_modules_never_import_experiment():
    for rel in (
        "app/paper.py",
        "app/engine.py",
        "app/main.py",
        "app/mt5_client.py",
    ):
        assert "research_breakeven_experiment" not in (REPO_ROOT / rel).read_text(
            encoding="utf-8"
        )


def test_runner_is_bom_safe_and_has_no_network_order_or_db_commands():
    runner = REPO_ROOT / "scripts" / "Run-AdaptiveV2BreakevenExperiment.ps1"
    raw = runner.read_bytes()
    assert raw[:3] == b"\xef\xbb\xbf"
    text = raw.decode("utf-8-sig")
    assert "-m app.research_breakeven_experiment run" in text
    assert "rev-parse --path-format=absolute --git-common-dir" in text
    assert "repository's main checkout" in text
    assert '$python = "python"' not in text
    for forbidden in (
        "Invoke-WebRequest",
        "Invoke-RestMethod",
        "sqlite3",
        "app.main",
        "app.paper",
        "app.engine",
        "app.mt5_client",
        "send_order",
    ):
        assert forbidden not in text
