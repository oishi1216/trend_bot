from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import app.research_max_positions_experiment as rmpe
from app.config import Settings
from app.research_backtest import run_research
from app.research_max_positions_experiment import (
    ANCHOR,
    CONTROL_MAX_OPEN_POSITIONS,
    RETENTION_MIN_RATIO,
    SCORE_MIN,
    TREATMENT_MAX_OPEN_POSITIONS,
    _anchor_comparison,
    _classify,
    _evaluate_gates,
    _full_sample_not_materially_worse,
    _retention,
    _walk_forward_no_contradiction,
    build_experiment_settings,
    main,
    run_experiment,
)
from app.research_max_positions_experiment import run_research as reexported_run_research
from tests.test_research import _fast_settings

FORWARD_EXECUTION_FILES = (
    "app/paper.py",
    "app/engine.py",
    "app/main.py",
    "app/mt5_client.py",
    "app/strategy.py",
    "app/adaptive_strategy.py",
    "app/risk.py",
)


def _metrics(trades, pf, profit, cagr=0.0, max_dd=0.0, win_rate=0.0, avg_r=0.0):
    return {
        "trades": trades,
        "pf": pf,
        "profit": profit,
        "cagr": cagr,
        "max_dd": max_dd,
        "win_rate": win_rate,
        "avg_r": avg_r,
    }


def _fold(max_dd, trades=2):
    return {"metrics": _metrics(trades, 1.0, 0.0, max_dd=max_dd)}


def _payload(
    *,
    full_trades=100,
    full_pf=1.0,
    full_profit=1000.0,
    full_max_dd=0.1,
    test_trades=20,
    test_pf=1.0,
    test_max_dd=0.1,
    validation_pf=1.0,
    train_pf=1.0,
    high_cost_pf=1.0,
    high_cost_max_dd=0.1,
    folds=(),
):
    return {
        "metrics": _metrics(full_trades, full_pf, full_profit, max_dd=full_max_dd),
        "train": {"metrics": _metrics(full_trades, train_pf, 0.0)},
        "validation": {"metrics": _metrics(full_trades, validation_pf, 0.0)},
        "test": {"metrics": _metrics(test_trades, test_pf, 0.0, max_dd=test_max_dd)},
        "sensitivity": {
            "base": {"metrics": _metrics(full_trades, full_pf, full_profit, max_dd=full_max_dd)},
            "cost_x2": {"metrics": _metrics(full_trades, high_cost_pf, full_profit, max_dd=high_cost_max_dd)},
        },
        "walk_forward": {"folds": list(folds), "fold_positive": 3, "fold_total": 5},
    }


def _gates(control, treatment, anchor_matches=True, env_unchanged=True):
    anchor = {"matches": anchor_matches, "fields": {}}
    return _evaluate_gates(control, treatment, anchor, env_unchanged)


def _canned_run_research():
    """Deterministic, instant stand-in for app.research_backtest.run_research.

    Returns (fake, calls): `fake` is monkeypatched over
    app.research_max_positions_experiment.run_research so run_experiment()
    exercises its own orchestration/gate logic without re-running the real
    multi-day simulator (which is what made the focused test file slow).
    `calls` records every Settings instance each arm was invoked with, so
    tests can still prove control=2/treatment=1 (both score_min=75) reached
    the (mocked) simulator and that exactly one field differed between the
    two calls. The canned payload shape mirrors
    app.research_backtest.run_research's real return value (metrics/train/
    validation/test/sensitivity/walk_forward/baseline_config) so it
    exercises the same consumer code.
    """
    calls: list[Settings] = []

    def _fake(candles_by_instrument, settings, initial_equity=1_000_000.0):
        calls.append(settings)
        is_control = settings.adaptive_max_open_positions == CONTROL_MAX_OPEN_POSITIONS
        full_metrics = _metrics(
            133 if is_control else 70,
            0.8137793529 if is_control else 1.5,
            -37421.8830 if is_control else 500.0,
            cagr=-0.0030874892 if is_control else 0.01,
            max_dd=0.1006936790 if is_control else 0.05,
            win_rate=0.4,
            avg_r=0.1,
        )
        test_metrics = _metrics(
            13 if is_control else 8,
            0.4276630536 if is_control else 1.2,
            10.0,
            max_dd=0.05,
            win_rate=0.4,
            avg_r=0.1,
        )
        validation_metrics = dict(test_metrics, pf=0.5 if is_control else 1.1)
        train_metrics = dict(full_metrics)
        high_cost_metrics = dict(full_metrics, pf=full_metrics["pf"] * 0.9)
        folds = [
            {"fold": idx, "metrics": _metrics(2, 1.0, 0.0, max_dd=0.08 if is_control else 0.04)}
            for idx in range(1, 6)
        ]
        return {
            "period": {
                "start": "2020-01-01T00:00:00+00:00",
                "end": "2020-12-01T00:00:00+00:00",
                "trading_days": 260,
            },
            "metrics": full_metrics,
            "annual": {},
            "by_symbol": {},
            "by_regime": {},
            "by_score_band": {},
            "sensitivity": {
                "base": {"cost_pips": {"spread": 1.0, "slippage": 0.3}, "metrics": full_metrics},
                "cost_x2": {"cost_pips": {"spread": 3.0, "slippage": 0.6}, "metrics": high_cost_metrics},
            },
            "train": {"trades": train_metrics["trades"], "metrics": train_metrics, "annual": {}},
            "validation": {"trades": validation_metrics["trades"], "metrics": validation_metrics, "annual": {}},
            "test": {"trades": test_metrics["trades"], "metrics": test_metrics, "annual": {}},
            "walk_forward": {
                "folds": folds,
                "fold_positive": 3,
                "fold_total": 5,
                "mode": "continuous_state_period_slice",
                "trade_attribution": "exit_date",
                "state_reset_between_folds": False,
            },
            "cost_assumptions": {
                "base_spread_pips": 1.0,
                "base_slippage_pips": 0.3,
                "high_spread_pips": 3.0,
                "high_slippage_pips": 0.6,
            },
            "baseline_config": {
                "strategy_profile": settings.strategy_profile,
                "adaptive_strength_gap_min": settings.adaptive_strength_gap_min,
                "adaptive_score_min": settings.adaptive_score_min,
                "adaptive_score_medium": settings.adaptive_score_medium,
                "adaptive_score_high": settings.adaptive_score_high,
                "adaptive_trend_adx": settings.adaptive_trend_adx,
                "adaptive_range_adx": settings.adaptive_range_adx,
                "adaptive_max_open_positions": settings.adaptive_max_open_positions,
            },
        }

    return _fake, calls


# --- control=2, treatment=1, both score_min=75, exactly one variable differs


def test_control_is_2_and_treatment_is_1_both_score_min_75():
    assert CONTROL_MAX_OPEN_POSITIONS == 2
    assert TREATMENT_MAX_OPEN_POSITIONS == 1
    assert SCORE_MIN == 75.0


def test_build_experiment_settings_differs_only_in_max_open_positions():
    base = Settings.from_env()
    control, treatment = build_experiment_settings(base)

    assert control.adaptive_max_open_positions == 2
    assert treatment.adaptive_max_open_positions == 1
    assert control.adaptive_score_min == 75.0
    assert treatment.adaptive_score_min == 75.0

    differing_fields = [
        f.name
        for f in dataclasses.fields(control)
        if getattr(control, f.name) != getattr(treatment, f.name)
    ]
    assert differing_fields == ["adaptive_max_open_positions"]


def test_build_experiment_settings_does_not_carry_forward_issue_7_treatment_score():
    base = replace_score_min(Settings.from_env(), 85.0)
    control, treatment = build_experiment_settings(base)
    assert control.adaptive_score_min == 75.0
    assert treatment.adaptive_score_min == 75.0


def replace_score_min(settings, score_min):
    return dataclasses.replace(settings, adaptive_score_min=score_min)


# --- Settings.from_env defaults unchanged -----------------------------------


def test_settings_from_env_defaults_unchanged_before_and_after_experiment(monkeypatch):
    fake, _calls = _canned_run_research()
    monkeypatch.setattr(rmpe, "run_research", fake)

    before = dataclasses.asdict(Settings.from_env())
    run_experiment({}, _fast_settings(), initial_equity=1_000_000.0)
    after = dataclasses.asdict(Settings.from_env())

    assert before == after
    assert after["adaptive_score_min"] == 75.0
    assert after["adaptive_max_open_positions"] == 2


def test_build_experiment_settings_does_not_mutate_base_settings():
    base = _fast_settings()
    before = dataclasses.asdict(base)
    build_experiment_settings(base)
    assert dataclasses.asdict(base) == before


# --- same Research simulator path reused, not reimplemented ----------------


def test_reuses_run_research_rather_than_reimplementing():
    assert reexported_run_research is run_research


def test_control_and_treatment_arms_differ_only_in_max_open_positions(monkeypatch):
    fake, calls = _canned_run_research()
    monkeypatch.setattr(rmpe, "run_research", fake)

    base = _fast_settings()
    payload = run_experiment({}, base, initial_equity=1_000_000.0)

    assert len(calls) == 2
    control_settings, treatment_settings = calls
    assert control_settings.adaptive_max_open_positions == 2
    assert treatment_settings.adaptive_max_open_positions == 1
    assert control_settings.adaptive_score_min == 75.0
    assert treatment_settings.adaptive_score_min == 75.0

    differing_fields = [
        f.name
        for f in dataclasses.fields(control_settings)
        if getattr(control_settings, f.name) != getattr(treatment_settings, f.name)
    ]
    assert differing_fields == ["adaptive_max_open_positions"]

    assert payload["control"]["baseline_config"]["adaptive_max_open_positions"] == 2
    assert payload["treatment"]["baseline_config"]["adaptive_max_open_positions"] == 1
    assert payload["control"]["baseline_config"]["adaptive_score_min"] == 75.0
    assert payload["treatment"]["baseline_config"]["adaptive_score_min"] == 75.0


# --- deterministic report ---------------------------------------------------


def test_run_experiment_is_deterministic(monkeypatch):
    fake, calls = _canned_run_research()
    monkeypatch.setattr(rmpe, "run_research", fake)
    settings = _fast_settings()

    first = run_experiment({}, settings, initial_equity=1_000_000.0)
    second = run_experiment({}, settings, initial_equity=1_000_000.0)

    assert first == second
    assert len(calls) == 4
    assert first["classification"] in {"SUPPORTED", "INCONCLUSIVE", "FALSIFIED"}


def test_write_report_produces_deterministic_json_and_markdown(monkeypatch, tmp_path):
    from app.research_max_positions_experiment import write_report

    fake, _calls = _canned_run_research()
    monkeypatch.setattr(rmpe, "run_research", fake)
    payload = run_experiment({}, _fast_settings(), initial_equity=1_000_000.0)

    json_path_a = tmp_path / "a.json"
    md_path_a = tmp_path / "a.md"
    json_path_b = tmp_path / "b.json"
    md_path_b = tmp_path / "b.md"

    write_report(payload, json_path_a, md_path_a)
    write_report(payload, json_path_b, md_path_b)

    assert json_path_a.read_text(encoding="utf-8") == json_path_b.read_text(encoding="utf-8")
    assert md_path_a.read_text(encoding="utf-8") == md_path_b.read_text(encoding="utf-8")
    assert "Classification:" in md_path_a.read_text(encoding="utf-8")


# --- retention gate ----------------------------------------------------------


def test_retention_passes_at_or_above_half():
    retention = _retention(control_trades=100, treatment_trades=50)
    assert retention["ratio"] == 0.5
    assert retention["passes_min_ratio"] is True


def test_retention_fails_below_half_and_forces_inconclusive():
    control = _payload(full_trades=100, full_max_dd=0.2, test_max_dd=0.1)
    treatment = _payload(full_trades=49, full_max_dd=0.1, test_max_dd=0.1)
    gates = _gates(control, treatment)

    assert gates["retention"] is False
    classification, _ = _classify(gates)
    assert classification == "INCONCLUSIVE"


def test_retention_with_zero_control_trades_does_not_pass():
    retention = _retention(control_trades=0, treatment_trades=0)
    assert retention["ratio"] is None
    assert retention["passes_min_ratio"] is False
    assert RETENTION_MIN_RATIO == 0.5


# --- max-DD gates (primary falsification condition) -------------------------


def test_falsified_when_full_max_dd_does_not_improve():
    control = _payload(full_max_dd=0.1, test_max_dd=0.1)
    treatment = _payload(full_max_dd=0.1, test_max_dd=0.1)
    gates = _gates(control, treatment)
    assert gates["full_maxdd_improves"] is False
    classification, _ = _classify(gates)
    assert classification == "FALSIFIED"


def test_falsified_when_test_max_dd_worsens():
    control = _payload(full_max_dd=0.2, test_max_dd=0.1)
    treatment = _payload(full_max_dd=0.1, test_max_dd=0.2)
    gates = _gates(control, treatment)
    assert gates["full_maxdd_improves"] is True
    assert gates["test_maxdd_not_worse"] is False
    classification, _ = _classify(gates)
    assert classification == "FALSIFIED"


def test_supported_requires_full_and_test_maxdd_improvement():
    control = _payload(full_max_dd=0.2, test_max_dd=0.2, high_cost_max_dd=0.2)
    treatment = _payload(full_max_dd=0.1, test_max_dd=0.1, high_cost_max_dd=0.1)
    gates = _gates(control, treatment)
    assert gates["full_maxdd_improves"] is True
    assert gates["test_maxdd_not_worse"] is True
    classification, _ = _classify(gates)
    assert classification == "SUPPORTED"


# --- Validation/Test PF gate: train-only improvement is insufficient -------


def test_validation_and_test_pf_gate_blocks_supported_when_both_worse():
    control = _payload(
        full_max_dd=0.2, test_max_dd=0.2, high_cost_max_dd=0.2, test_pf=2.0, validation_pf=2.0, train_pf=1.0
    )
    treatment = _payload(
        full_max_dd=0.1, test_max_dd=0.1, high_cost_max_dd=0.1, test_pf=1.0, validation_pf=1.0, train_pf=2.0
    )
    gates = _gates(control, treatment)

    assert gates["validation_and_test_pf_not_both_worse"] is False
    assert gates["details"]["train_only_result"] is True
    classification, _ = _classify(gates)
    assert classification == "INCONCLUSIVE"


def test_validation_and_test_pf_gate_passes_when_not_both_worse():
    control = _payload(
        full_max_dd=0.2, test_max_dd=0.2, high_cost_max_dd=0.2, test_pf=1.0, validation_pf=2.0
    )
    treatment = _payload(
        full_max_dd=0.1, test_max_dd=0.1, high_cost_max_dd=0.1, test_pf=1.0, validation_pf=1.0
    )
    # high_cost_max_dd differs (0.2 -> 0.1) alongside full/test max_dd so the
    # high-cost gate also sees an improvement, isolating this assertion to
    # the Validation/Test PF gate rather than tripping on Gate 7.
    gates = _gates(control, treatment)

    assert gates["validation_and_test_pf_not_both_worse"] is True
    classification, _ = _classify(gates)
    assert classification == "SUPPORTED"


# --- high-cost reversal gate --------------------------------------------------


def test_high_cost_reversal_blocks_supported_classification():
    control = _payload(full_max_dd=0.2, test_max_dd=0.2, high_cost_max_dd=0.1)
    treatment = _payload(full_max_dd=0.1, test_max_dd=0.1, high_cost_max_dd=0.2)
    gates = _gates(control, treatment)

    assert gates["high_cost_no_reversal"] is False
    classification, _ = _classify(gates)
    assert classification == "INCONCLUSIVE"


def test_high_cost_consistent_improvement_allows_supported():
    control = _payload(full_max_dd=0.2, test_max_dd=0.2, high_cost_max_dd=0.2)
    treatment = _payload(full_max_dd=0.1, test_max_dd=0.1, high_cost_max_dd=0.1)
    gates = _gates(control, treatment)

    assert gates["high_cost_no_reversal"] is True
    classification, _ = _classify(gates)
    assert classification == "SUPPORTED"


# --- walk-forward gate --------------------------------------------------------


def test_walk_forward_no_contradiction_handles_empty_folds():
    passes, details = _walk_forward_no_contradiction({"folds": []}, {"folds": []})
    assert passes is True
    assert details["sample_sufficient"] is False

    passes, details = _walk_forward_no_contradiction(
        {"folds": [_fold(0.1), _fold(0.1)]}, {"folds": [_fold(0.05), _fold(0.2)]}
    )
    assert details == {"folds_compared": 2, "folds_not_worse": 1, "ratio": 0.5, "sample_sufficient": True}
    assert passes is True


def test_walk_forward_gate_blocks_supported_when_direction_reverses():
    control_folds = [_fold(0.1) for _ in range(5)]
    treatment_folds = [_fold(0.2) for _ in range(4)] + [_fold(0.05)]
    control = _payload(full_max_dd=0.2, test_max_dd=0.2, folds=control_folds)
    treatment = _payload(full_max_dd=0.1, test_max_dd=0.1, folds=treatment_folds)
    gates = _gates(control, treatment)

    assert gates["walk_forward_no_contradiction"] is False
    classification, _ = _classify(gates)
    assert classification == "INCONCLUSIVE"


def test_walk_forward_gate_passes_when_direction_preserved_or_improved():
    control_folds = [_fold(0.1) for _ in range(5)]
    treatment_folds = [_fold(0.05) for _ in range(5)]
    control = _payload(full_max_dd=0.2, test_max_dd=0.2, high_cost_max_dd=0.2, folds=control_folds)
    treatment = _payload(full_max_dd=0.1, test_max_dd=0.1, high_cost_max_dd=0.1, folds=treatment_folds)
    gates = _gates(control, treatment)

    assert gates["walk_forward_no_contradiction"] is True
    classification, _ = _classify(gates)
    assert classification == "SUPPORTED"


# --- deterministic classification -------------------------------------------


def test_classification_is_deterministic_for_same_gates():
    control = _payload(full_max_dd=0.2, test_max_dd=0.2)
    treatment = _payload(full_max_dd=0.1, test_max_dd=0.1)
    gates_a = _gates(control, treatment)
    gates_b = _gates(control, treatment)

    assert gates_a == gates_b
    assert _classify(gates_a) == _classify(gates_b)


def test_classification_is_only_one_of_three_values():
    control = _payload(full_max_dd=0.2, test_max_dd=0.2)
    treatment = _payload(full_max_dd=0.1, test_max_dd=0.1)
    classification, _ = _classify(_gates(control, treatment))
    assert classification in {"SUPPORTED", "INCONCLUSIVE", "FALSIFIED"}


# --- anchor mismatch must not claim treatment support ------------------------


def test_anchor_mismatch_forces_inconclusive_even_if_treatment_looks_better():
    control = _payload(full_max_dd=0.2, test_max_dd=0.2)
    treatment = _payload(full_max_dd=0.01, test_max_dd=0.01)
    gates = _gates(control, treatment, anchor_matches=False)

    classification, rationale = _classify(gates)
    assert classification == "INCONCLUSIVE"
    assert "anchor" in rationale.lower()


def test_anchor_comparison_reports_per_field_matches():
    control_payload = {
        "metrics": _metrics(ANCHOR["trades"], ANCHOR["pf"], ANCHOR["profit"], cagr=ANCHOR["cagr"], max_dd=ANCHOR["max_dd"]),
        "test": {"metrics": _metrics(ANCHOR["test_trades"], ANCHOR["test_pf"], 0.0)},
    }
    anchor = _anchor_comparison(control_payload)
    assert anchor["matches"] is True
    assert all(field["matches"] for field in anchor["fields"].values())


def test_anchor_comparison_detects_mismatch():
    control_payload = {
        "metrics": _metrics(999, 1.5, 1.0, cagr=0.5, max_dd=0.01),
        "test": {"metrics": _metrics(5, 1.5, 0.0)},
    }
    anchor = _anchor_comparison(control_payload)
    assert anchor["matches"] is False
    assert anchor["fields"]["trades"]["matches"] is False


# --- full-sample materiality gate --------------------------------------------


def test_full_sample_materiality_tolerates_small_degradation():
    control_metrics = _metrics(100, 1.0, -1000.0)
    treatment_metrics = _metrics(100, 0.95, -1050.0)
    assert _full_sample_not_materially_worse(control_metrics, treatment_metrics) is True


def test_full_sample_materiality_rejects_large_degradation():
    control_metrics = _metrics(100, 1.0, -1000.0)
    treatment_metrics = _metrics(100, 0.5, -2000.0)
    assert _full_sample_not_materially_worse(control_metrics, treatment_metrics) is False


def test_full_sample_materiality_gate_blocks_supported_despite_maxdd_improvement():
    control = _payload(full_max_dd=0.2, test_max_dd=0.1, full_pf=1.0, full_profit=-1000.0)
    treatment = _payload(full_max_dd=0.05, test_max_dd=0.05, full_pf=0.5, full_profit=-2000.0)
    gates = _gates(control, treatment)

    assert gates["full_maxdd_improves"] is True
    assert gates["full_sample_not_materially_worse"] is False
    classification, rationale = _classify(gates)
    assert classification == "INCONCLUSIVE"
    assert "not sufficient" in rationale.lower()


# --- production defaults / forward semantics gate ----------------------------


def test_production_defaults_unchanged_gate_uses_env_snapshot():
    control = _payload(full_max_dd=0.2, test_max_dd=0.2)
    treatment = _payload(full_max_dd=0.1, test_max_dd=0.1)
    gates_ok = _gates(control, treatment, env_unchanged=True)
    gates_bad = _gates(control, treatment, env_unchanged=False)

    assert gates_ok["production_defaults_unchanged"] is True
    assert gates_bad["production_defaults_unchanged"] is False
    classification, rationale = _classify(gates_bad)
    assert classification == "INCONCLUSIVE"
    assert "production defaults" in rationale.lower()


def test_production_defaults_remain_2_and_75_after_module_import():
    settings = Settings.from_env()
    assert settings.adaptive_max_open_positions == 2
    assert settings.adaptive_score_min == 75.0


# --- output-only artifacts are never consumed by forward execution ----------


def test_forward_execution_modules_never_import_this_experiment_module():
    for rel_path in FORWARD_EXECUTION_FILES:
        source = Path(rel_path).read_text(encoding="utf-8")
        assert "research_max_positions_experiment" not in source
        assert "adaptive_v2_max_open_positions_experiment" not in source


def test_main_writes_output_only_artifacts_under_data_research(monkeypatch, tmp_path, capsys):
    fake, calls = _canned_run_research()
    monkeypatch.setattr(rmpe, "run_research", fake)

    data_dir = tmp_path / "fx_research"
    data_dir.mkdir()
    (data_dir / "USDJPY.json").write_text(
        json.dumps(
            {
                "instrument": "USDJPY",
                "synced_at": "2026-08-16T00:00:00+00:00",
                "bars": [
                    {
                        "time": "2020-01-01T00:00:00+00:00",
                        "open": 100.0,
                        "high": 100.5,
                        "low": 99.5,
                        "close": 100.2,
                    },
                    {
                        "time": "2020-01-02T00:00:00+00:00",
                        "open": 100.2,
                        "high": 100.7,
                        "low": 99.7,
                        "close": 100.4,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    json_path = tmp_path / "out.json"
    markdown_path = tmp_path / "out.md"
    exit_code = main(
        [
            "run",
            "--data-dir",
            str(data_dir),
            "--json-path",
            str(json_path),
            "--markdown-path",
            str(markdown_path),
        ]
    )

    assert exit_code == 0
    assert json_path.exists()
    assert markdown_path.exists()
    assert len(calls) == 2
    out = capsys.readouterr().out
    assert "MAX_POSITIONS_EXPERIMENT_OK" in out

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["classification"] in {"SUPPORTED", "INCONCLUSIVE", "FALSIFIED"}
    assert payload["control_max_open_positions"] == 2
    assert payload["treatment_max_open_positions"] == 1
    assert payload["score_min"] == 75.0


def test_main_reports_no_data_without_touching_production_db(tmp_path):
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    exit_code = main(
        [
            "run",
            "--data-dir",
            str(empty_dir),
            "--json-path",
            str(tmp_path / "x.json"),
            "--markdown-path",
            str(tmp_path / "x.md"),
        ]
    )
    assert exit_code == 1
    assert not (tmp_path / "x.json").exists()
