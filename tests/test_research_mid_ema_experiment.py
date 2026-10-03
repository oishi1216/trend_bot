from __future__ import annotations

import dataclasses
import json
from dataclasses import replace
from pathlib import Path

import app.research_mid_ema_experiment as rmee
from app.config import Settings
from app.research_backtest import run_research
from app.research_mid_ema_experiment import (
    ANCHOR,
    CONTROL_MID_EMA_DAYS,
    FOLD_MIN_TRADES,
    GATE_NAMES,
    RETENTION_MIN_RATIO,
    TREATMENT_MID_EMA_DAYS,
    _anchor_comparison,
    _classify,
    _evaluate_gates,
    _retention,
    _walk_forward_gate,
    build_experiment_settings,
    main,
    run_experiment,
    write_report,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

FORWARD_EXECUTION_FILES = (
    "app/paper.py",
    "app/engine.py",
    "app/main.py",
    "app/mt5_client.py",
    "app/strategy.py",
    "app/adaptive_strategy.py",
    "app/risk.py",
)

PINNED = {
    "adaptive_fast_ema_days": 20,
    "adaptive_slow_ema_days": 200,
    "adaptive_score_min": 75.0,
    "adaptive_max_open_positions": 2,
    "adaptive_strength_gap_min": 0.3,
}


def _base_settings() -> Settings:
    return replace(Settings.from_env(), strategy_profile="adaptive_dual_regime_v1")


def _metrics(trades, pf, profit=0.0, cagr=0.0, max_dd=0.0, win_rate=0.0, avg_r=0.0):
    return {
        "trades": trades,
        "pf": pf,
        "profit": profit,
        "cagr": cagr,
        "max_dd": max_dd,
        "win_rate": win_rate,
        "avg_r": avg_r,
    }


def _fold(pf, trades=5, max_dd=0.0):
    return {"metrics": _metrics(trades, pf, max_dd=max_dd)}


def _arm(
    *,
    trades=100,
    pf=1.0,
    avg_r=0.1,
    profit=1000.0,
    max_dd=0.1,
    cagr=0.0,
    val_pf=1.0,
    test_pf=1.0,
    test_trades=20,
    test_max_dd=0.1,
    high_pf=1.0,
    fold_positive=3,
    folds=None,
):
    if folds is None:
        folds = [_fold(pf) for _ in range(5)]
    return {
        "metrics": _metrics(trades, pf, profit, cagr=cagr, max_dd=max_dd, avg_r=avg_r),
        "train": {"metrics": _metrics(trades, pf, profit)},
        "validation": {"metrics": _metrics(trades, val_pf, 0.0)},
        "test": {"metrics": _metrics(test_trades, test_pf, 0.0, max_dd=test_max_dd)},
        "sensitivity": {
            "base": {"metrics": _metrics(trades, pf, profit, max_dd=max_dd)},
            "cost_x2": {"metrics": _metrics(trades, high_pf, profit, max_dd=max_dd)},
        },
        "walk_forward": {
            "folds": list(folds),
            "fold_positive": fold_positive,
            "fold_total": len(folds),
        },
    }


def _pair(**treatment_overrides):
    """Control (EMA 50) baseline and a treatment (EMA 75) that passes every gate unless overridden."""
    control = _arm(pf=1.0, avg_r=0.1, max_dd=0.1, val_pf=1.0, test_pf=1.0, test_max_dd=0.1, high_pf=1.0)
    treatment_kwargs = dict(
        pf=1.2,
        avg_r=0.2,
        max_dd=0.1,
        val_pf=1.1,
        test_pf=1.1,
        test_max_dd=0.1,
        high_pf=1.1,
        fold_positive=3,
        folds=[_fold(1.2) for _ in range(5)],
    )
    treatment_kwargs.update(treatment_overrides)
    return control, _arm(**treatment_kwargs)


def _gates(control, treatment, anchor_matches=True, env_unchanged=True):
    anchor = {"matches": anchor_matches, "fields": {}}
    return _evaluate_gates(control, treatment, anchor, env_unchanged)


def _classification(control, treatment, **kwargs):
    return _classify(_gates(control, treatment, **kwargs))[0]


def _canned_run_research():
    """Deterministic stand-in for app.research_backtest.run_research (no real simulation).

    Returns (fake, calls): `fake` is monkeypatched over
    app.research_mid_ema_experiment.run_research so run_experiment()
    exercises its own orchestration and gate logic. `calls` records every
    Settings instance each arm was invoked with.
    """
    calls: list[Settings] = []

    def _fake(candles_by_instrument, settings, initial_equity=1_000_000.0):
        calls.append(settings)
        if settings.adaptive_mid_ema_days == CONTROL_MID_EMA_DAYS:
            arm = _arm(
                trades=133,
                pf=ANCHOR["pf"],
                avg_r=-0.01,
                profit=ANCHOR["profit"],
                max_dd=ANCHOR["max_dd"],
                cagr=ANCHOR["cagr"],
                val_pf=0.5,
                test_pf=ANCHOR["test_pf"],
                test_trades=int(ANCHOR["test_trades"]),
                test_max_dd=0.05,
                high_pf=0.7,
                fold_positive=2,
                folds=[_fold(0.5) for _ in range(5)],
            )
        else:
            arm = _arm(
                trades=100,
                pf=1.0,
                avg_r=0.05,
                profit=5000.0,
                max_dd=0.05,
                val_pf=1.1,
                test_pf=0.9,
                test_trades=12,
                test_max_dd=0.05,
                high_pf=0.9,
                fold_positive=3,
                folds=[_fold(0.9) for _ in range(5)],
            )
        arm["period"] = {"start": "2020-01-01", "end": "2020-12-01", "trading_days": 260}
        arm["baseline_config"] = {
            "adaptive_mid_ema_days": settings.adaptive_mid_ema_days,
            "adaptive_fast_ema_days": settings.adaptive_fast_ema_days,
            "adaptive_slow_ema_days": settings.adaptive_slow_ema_days,
            "adaptive_score_min": settings.adaptive_score_min,
            "adaptive_max_open_positions": settings.adaptive_max_open_positions,
            "adaptive_strength_gap_min": settings.adaptive_strength_gap_min,
        }
        arm["by_regime"] = {"trend": _metrics(60, 1.1, avg_r=0.1)}
        arm["by_score_band"] = {"75-82": _metrics(60, 1.1, avg_r=0.1)}
        return arm

    return _fake, calls


# --- control=50, treatment=75, every other field pinned ---------------------


def test_control_is_50_and_treatment_is_75_with_pinned_others():
    assert CONTROL_MID_EMA_DAYS == 50
    assert TREATMENT_MID_EMA_DAYS == 75


def test_build_experiment_settings_differs_only_in_mid_ema():
    control, treatment = build_experiment_settings(_base_settings())

    assert control.adaptive_mid_ema_days == 50
    assert treatment.adaptive_mid_ema_days == 75

    differing_fields = [
        f.name
        for f in dataclasses.fields(control)
        if getattr(control, f.name) != getattr(treatment, f.name)
    ]
    assert differing_fields == ["adaptive_mid_ema_days"]


def test_both_arms_preserve_fast20_slow200_score75_maxpos2_strength03():
    control, treatment = build_experiment_settings(_base_settings())
    for settings in (control, treatment):
        for field, value in PINNED.items():
            assert getattr(settings, field) == value


def test_base_carryover_from_issue_7_or_9_or_other_overrides_is_not_inherited():
    # Simulates a base carrying Issue #7 score=85, Issue #9 max_positions=1,
    # and drifted fast/slow/strength values; both arms must still pin the
    # Issue #12 values.
    base = replace(
        _base_settings(),
        adaptive_score_min=85.0,
        adaptive_max_open_positions=1,
        adaptive_fast_ema_days=15,
        adaptive_slow_ema_days=120,
        adaptive_strength_gap_min=0.5,
    )
    control, treatment = build_experiment_settings(base)
    for settings in (control, treatment):
        for field, value in PINNED.items():
            assert getattr(settings, field) == value


def test_build_experiment_settings_does_not_mutate_base_settings():
    base = _base_settings()
    before = dataclasses.asdict(base)
    build_experiment_settings(base)
    assert dataclasses.asdict(base) == before


# --- production defaults unchanged ------------------------------------------


def test_production_defaults_are_unchanged(monkeypatch):
    for key in rmee._ENV_WATCH_KEYS:
        monkeypatch.delenv(key, raising=False)
    settings = Settings.from_env()
    assert settings.adaptive_mid_ema_days == 50
    assert settings.adaptive_fast_ema_days == 20
    assert settings.adaptive_slow_ema_days == 200
    assert settings.adaptive_score_min == 75.0
    assert settings.adaptive_max_open_positions == 2
    assert settings.adaptive_strength_gap_min == 0.3


def test_run_experiment_leaves_production_settings_unchanged(monkeypatch):
    fake, _calls = _canned_run_research()
    monkeypatch.setattr(rmee, "run_research", fake)

    before = dataclasses.asdict(Settings.from_env())
    run_experiment({}, _base_settings(), initial_equity=1_000_000.0)
    after = dataclasses.asdict(Settings.from_env())

    assert before == after


# --- same Research simulator path reused, not reimplemented -----------------


def test_reuses_app_research_backtest_run_research_identity():
    assert rmee.run_research is run_research


def test_control_and_treatment_arms_differ_only_in_mid_ema(monkeypatch):
    fake, calls = _canned_run_research()
    monkeypatch.setattr(rmee, "run_research", fake)

    payload = run_experiment({}, _base_settings(), initial_equity=1_000_000.0)

    assert len(calls) == 2
    control_settings, treatment_settings = calls
    assert control_settings.adaptive_mid_ema_days == 50
    assert treatment_settings.adaptive_mid_ema_days == 75
    differing_fields = [
        f.name
        for f in dataclasses.fields(control_settings)
        if getattr(control_settings, f.name) != getattr(treatment_settings, f.name)
    ]
    assert differing_fields == ["adaptive_mid_ema_days"]

    assert payload["control"]["baseline_config"]["adaptive_mid_ema_days"] == 50
    assert payload["treatment"]["baseline_config"]["adaptive_mid_ema_days"] == 75
    for key in PINNED:
        assert payload["control"]["baseline_config"][key] == PINNED[key]
        assert payload["treatment"]["baseline_config"][key] == PINNED[key]


# --- deterministic report ---------------------------------------------------


def test_run_experiment_is_deterministic(monkeypatch):
    fake, _calls = _canned_run_research()
    monkeypatch.setattr(rmee, "run_research", fake)
    settings = _base_settings()

    first = run_experiment({}, settings, initial_equity=1_000_000.0)
    second = run_experiment({}, settings, initial_equity=1_000_000.0)

    assert first == second
    assert first["classification"] in {"SUPPORTED", "INCONCLUSIVE", "FALSIFIED"}


def test_write_report_produces_deterministic_json_and_markdown(monkeypatch, tmp_path):
    fake, _calls = _canned_run_research()
    monkeypatch.setattr(rmee, "run_research", fake)
    payload = run_experiment({}, _base_settings(), initial_equity=1_000_000.0)

    json_a, md_a = tmp_path / "a.json", tmp_path / "a.md"
    json_b, md_b = tmp_path / "b.json", tmp_path / "b.md"
    write_report(payload, json_a, md_a)
    write_report(payload, json_b, md_b)

    assert json_a.read_text(encoding="utf-8") == json_b.read_text(encoding="utf-8")
    assert md_a.read_text(encoding="utf-8") == md_b.read_text(encoding="utf-8")


def test_markdown_report_contains_required_sections(monkeypatch, tmp_path):
    fake, _calls = _canned_run_research()
    monkeypatch.setattr(rmee, "run_research", fake)
    payload = run_experiment({}, _base_settings(), initial_equity=1_000_000.0)

    json_path, md_path = tmp_path / "out.json", tmp_path / "out.md"
    write_report(payload, json_path, md_path)
    markdown = md_path.read_text(encoding="utf-8")

    assert f"Classification: **{payload['classification']}**" in markdown
    for heading in (
        "## Anchor check",
        "## Full sample",
        "## Train / Validation / Test",
        "## Base-cost / high-cost sensitivity",
        "## Walk-forward",
        "## Trade-count retention",
        "## By regime",
        "## By score band",
        "## Gates",
    ):
        assert heading in markdown
    assert "trend" in markdown
    assert "75-82" in markdown


# --- anchor gate (gate 1) ----------------------------------------------------


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


def test_anchor_mismatch_forces_inconclusive_even_when_treatment_is_better():
    control, treatment = _pair()
    gates = _gates(control, treatment, anchor_matches=False)

    classification, rationale = _classify(gates)
    assert gates["anchor_match"] is False
    assert classification == "INCONCLUSIVE"
    assert "anchor" in rationale.lower()


# --- gate 2 / 3: full-sample PF and avg R -------------------------------------


def test_gate_2_full_pf_must_improve():
    control, treatment = _pair(pf=1.0)
    gates = _gates(control, treatment)
    assert gates["full_pf_improves"] is False
    assert _classify(gates)[0] == "FALSIFIED"


def test_gate_3_full_avg_r_must_improve():
    control, treatment = _pair(avg_r=0.1)
    gates = _gates(control, treatment)
    assert gates["full_avg_r_improves"] is False
    assert _classify(gates)[0] == "FALSIFIED"


# --- gate 4: Validation and Test PF both improve -----------------------------


def test_gate_4_full_only_improvement_is_not_supported():
    control, treatment = _pair(val_pf=0.9, test_pf=0.9)
    gates = _gates(control, treatment)
    assert gates["full_pf_improves"] is True
    assert gates["validation_and_test_pf_improve"] is False
    assert _classify(gates)[0] == "FALSIFIED"


def test_gate_4_test_only_improvement_is_not_supported():
    control, treatment = _pair(val_pf=0.9, test_pf=1.5)
    gates = _gates(control, treatment)
    assert gates["validation_and_test_pf_improve"] is False
    assert gates["details"]["oos_pf_both_not_improving"] is False
    assert _classify(gates)[0] == "INCONCLUSIVE"


def test_gate_4_validation_only_improvement_is_not_supported():
    control, treatment = _pair(val_pf=1.5, test_pf=0.9)
    gates = _gates(control, treatment)
    assert gates["validation_and_test_pf_improve"] is False
    assert _classify(gates)[0] == "INCONCLUSIVE"


# --- gate 5: Test max DD not worse -------------------------------------------


def test_gate_5_test_maxdd_worse_blocks_supported():
    control, treatment = _pair(test_max_dd=0.2)
    gates = _gates(control, treatment)
    assert gates["test_maxdd_not_worse"] is False
    assert _classify(gates)[0] == "INCONCLUSIVE"


# --- gate 6: full max DD no more than 10% worse -------------------------------


def test_gate_6_full_maxdd_within_10pct_is_allowed():
    control, treatment = _pair(max_dd=0.1)
    treatment_within = _arm(**{**_kwargs_of(treatment), "max_dd": 0.1 * 1.10})
    gates = _gates(control, treatment_within)
    assert gates["full_maxdd_not_more_than_10pct_worse"] is True
    assert _classify(gates)[0] == "SUPPORTED"


def test_gate_6_full_maxdd_more_than_10pct_worse_blocks_supported():
    control, treatment = _pair(max_dd=0.1)
    treatment_worse = _arm(**{**_kwargs_of(treatment), "max_dd": 0.1 * 1.11})
    gates = _gates(control, treatment_worse)
    assert gates["full_maxdd_not_more_than_10pct_worse"] is False
    assert _classify(gates)[0] == "INCONCLUSIVE"


def _kwargs_of(arm):
    """Rebuild _arm kwargs from an already-built arm payload (test helper only)."""
    return {
        "trades": arm["metrics"]["trades"],
        "pf": arm["metrics"]["pf"],
        "avg_r": arm["metrics"]["avg_r"],
        "profit": arm["metrics"]["profit"],
        "max_dd": arm["metrics"]["max_dd"],
        "cagr": arm["metrics"]["cagr"],
        "val_pf": arm["validation"]["metrics"]["pf"],
        "test_pf": arm["test"]["metrics"]["pf"],
        "test_trades": arm["test"]["metrics"]["trades"],
        "test_max_dd": arm["test"]["metrics"]["max_dd"],
        "high_pf": arm["sensitivity"]["cost_x2"]["metrics"]["pf"],
        "fold_positive": arm["walk_forward"]["fold_positive"],
        "folds": arm["walk_forward"]["folds"],
    }


# --- gate 7: retention --------------------------------------------------------


def test_retention_passes_at_or_above_half():
    retention = _retention(control_trades=100, treatment_trades=50)
    assert retention["ratio"] == 0.5
    assert retention["passes_min_ratio"] is True
    assert RETENTION_MIN_RATIO == 0.5


def test_retention_below_half_forces_inconclusive():
    control, treatment = _pair()
    treatment = _arm(**{**_kwargs_of(treatment), "trades": 49})
    gates = _gates(control, treatment)
    assert gates["retention"] is False
    assert _classify(gates)[0] == "INCONCLUSIVE"


def test_retention_with_zero_control_trades_does_not_pass():
    retention = _retention(control_trades=0, treatment_trades=0)
    assert retention["ratio"] is None
    assert retention["passes_min_ratio"] is False


# --- gate 8: high-cost PF direction preserved --------------------------------


def test_gate_8_high_cost_reversal_blocks_supported():
    control, treatment = _pair(high_pf=0.9)
    control = _arm(**{**_kwargs_of(control), "high_pf": 1.0})
    gates = _gates(control, treatment)
    assert gates["high_cost_pf_direction_preserved"] is False
    assert _classify(gates)[0] == "INCONCLUSIVE"


def test_gate_8_high_cost_direction_preserved_allows_supported():
    control, treatment = _pair()
    gates = _gates(control, treatment)
    assert gates["high_cost_pf_direction_preserved"] is True
    assert _classify(gates)[0] == "SUPPORTED"


# --- gate 9: walk-forward ------------------------------------------------------


def test_walk_forward_gate_fails_when_positive_fold_count_below_control():
    control, treatment = _pair(fold_positive=4)
    control = _arm(**{**_kwargs_of(control), "fold_positive": 4})
    treatment = _arm(**{**_kwargs_of(treatment), "fold_positive": 3})
    gates = _gates(control, treatment)
    assert gates["walk_forward_positive_and_pf_not_worse"] is False
    assert gates["details"]["walk_forward"]["fold_positive_ok"] is False
    assert _classify(gates)[0] == "INCONCLUSIVE"


def test_walk_forward_gate_fails_when_majority_of_sampled_folds_worse():
    control_folds = [_fold(1.0) for _ in range(5)]
    treatment_folds = [_fold(0.5) for _ in range(3)] + [_fold(1.5) for _ in range(2)]
    control, treatment = _pair(folds=treatment_folds)
    control = _arm(**{**_kwargs_of(control), "folds": control_folds})
    gates = _gates(control, treatment)
    details = gates["details"]["walk_forward"]
    assert details["folds_sufficient"] == 5
    assert details["folds_pf_not_worse"] == 2
    assert gates["walk_forward_positive_and_pf_not_worse"] is False
    assert _classify(gates)[0] == "INCONCLUSIVE"


def test_walk_forward_gate_ignores_undersampled_folds_for_pf_clause():
    # Only folds with >= FOLD_MIN_TRADES in both arms count; the rest cannot decide the gate.
    control_folds = [_fold(1.0, trades=FOLD_MIN_TRADES) for _ in range(2)] + [
        _fold(1.0, trades=1) for _ in range(3)
    ]
    treatment_folds = [_fold(1.2, trades=FOLD_MIN_TRADES) for _ in range(2)] + [
        _fold(0.1, trades=1) for _ in range(3)
    ]
    passes, details = _walk_forward_gate(
        {"folds": control_folds, "fold_positive": 2},
        {"folds": treatment_folds, "fold_positive": 2},
    )
    assert details["folds_sufficient"] == 2
    assert details["folds_pf_not_worse"] == 2
    assert passes is True


def test_walk_forward_gate_with_no_sufficiently_sampled_folds_does_not_pass():
    passes, details = _walk_forward_gate(
        {"folds": [_fold(1.0, trades=1)], "fold_positive": 1},
        {"folds": [_fold(1.2, trades=1)], "fold_positive": 1},
    )
    assert details["folds_sufficient"] == 0
    assert passes is False


# --- gate 10: production defaults / forward semantics -------------------------


def test_production_defaults_gate_forces_inconclusive():
    control, treatment = _pair()
    gates_ok = _gates(control, treatment, env_unchanged=True)
    gates_bad = _gates(control, treatment, env_unchanged=False)

    assert gates_ok["production_defaults_unchanged"] is True
    assert gates_bad["production_defaults_unchanged"] is False
    classification, rationale = _classify(gates_bad)
    assert classification == "INCONCLUSIVE"
    assert "production defaults" in rationale.lower()


# --- classification -----------------------------------------------------------


def test_supported_when_all_ten_gates_pass():
    control, treatment = _pair()
    gates = _gates(control, treatment)
    assert all(gates[name] for name in GATE_NAMES)
    assert _classify(gates)[0] == "SUPPORTED"


def test_gate_names_follow_issue_12_numbering():
    assert len(GATE_NAMES) == 10
    assert GATE_NAMES[0] == "anchor_match"
    assert GATE_NAMES[-1] == "production_defaults_unchanged"


def test_classification_is_deterministic_for_same_inputs():
    control, treatment = _pair()
    assert _classify(_gates(control, treatment)) == _classify(_gates(control, treatment))


def test_classification_is_only_one_of_three_values():
    control, treatment = _pair(pf=0.9)
    assert _classification(control, treatment) in {"SUPPORTED", "INCONCLUSIVE", "FALSIFIED"}


# --- output-only artifacts ----------------------------------------------------


def test_main_writes_output_only_artifacts_to_requested_paths(monkeypatch, tmp_path, capsys):
    fake, calls = _canned_run_research()
    monkeypatch.setattr(rmee, "run_research", fake)
    monkeypatch.chdir(tmp_path)

    data_dir = tmp_path / "fx_research"
    data_dir.mkdir()
    (data_dir / "USDJPY.json").write_text(
        json.dumps(
            {
                "instrument": "USDJPY",
                "synced_at": "2026-08-16T00:00:00+00:00",
                "bars": [
                    {"time": "2020-01-01T00:00:00+00:00", "open": 100.0, "high": 100.5, "low": 99.5, "close": 100.2},
                    {"time": "2020-01-02T00:00:00+00:00", "open": 100.2, "high": 100.7, "low": 99.7, "close": 100.4},
                ],
            }
        ),
        encoding="utf-8",
    )

    json_path = tmp_path / "out" / "mid.json"
    markdown_path = tmp_path / "out" / "mid.md"
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
    assert "MID_EMA_EXPERIMENT_OK" in capsys.readouterr().out

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["classification"] in {"SUPPORTED", "INCONCLUSIVE", "FALSIFIED"}
    assert payload["control_mid_ema_days"] == 50
    assert payload["treatment_mid_ema_days"] == 75
    assert payload["pinned_settings"] == PINNED
    # Only the explicit output paths are written; nothing lands under the working directory's data/.
    assert not (tmp_path / "data").exists()


def test_main_reports_no_data_without_writing_outputs(tmp_path):
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
    assert not (tmp_path / "x.md").exists()


# --- no forward imports -------------------------------------------------------


def test_forward_execution_modules_never_import_this_experiment_module():
    for rel_path in FORWARD_EXECUTION_FILES:
        source = (REPO_ROOT / rel_path).read_text(encoding="utf-8")
        assert "research_mid_ema_experiment" not in source


def test_runner_script_invokes_only_research_module_and_no_forward_or_network_commands():
    script = (REPO_ROOT / "scripts" / "Run-AdaptiveV2MidEmaExperiment.ps1").read_text(encoding="utf-8")
    assert "app.research_mid_ema_experiment" in script
    for forbidden in (
        "Invoke-WebRequest",
        "Invoke-RestMethod",
        "sqlite3",
        "app.main",
        "app.paper",
        "app.engine",
        "app.mt5_client",
    ):
        assert forbidden not in script
