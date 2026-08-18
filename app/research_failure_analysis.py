from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any, Mapping

import pandas as pd

from .config import Settings
from .fx_research_data import DEFAULT_RESEARCH_DATA_DIR, load_history
from .research_backtest import (
    DEFAULT_SLIPPAGE_PIPS, DEFAULT_SPREAD_PIPS, HIGH_COST_SLIPPAGE_PIPS,
    HIGH_COST_SPREAD_PIPS, _metrics, _simulate, _split_dates,
)


STRENGTH_BANDS = (("<0.3", float("-inf"), 0.3), ("0.3-0.5", 0.3, 0.5), ("0.5-1.0", 0.5, 1.0), (">=1.0", 1.0, float("inf")))
HOLDING_BANDS = (("1 day", 1, 1), ("2-3 days", 2, 3), ("4-7 days", 4, 7), ("8-14 days", 8, 14), ("15+ days", 15, float("inf")))


def _summary(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "min": None, "p25": None, "median": None, "mean": None, "p75": None, "max": None}
    ordered = sorted(values)
    def percentile(fraction: float) -> float:
        index = (len(ordered) - 1) * fraction
        low, high = int(index), min(len(ordered) - 1, int(index) + 1)
        return ordered[low] + (ordered[high] - ordered[low]) * (index - low)
    return {"count": len(values), "min": min(values), "p25": percentile(.25), "median": median(values), "mean": mean(values), "p75": percentile(.75), "max": max(values)}


def _holding_band(days: int) -> str:
    for label, low, high in HOLDING_BANDS:
        if low <= days <= high:
            return label
    return "15+ days"


def _band(value: float, bands: tuple[tuple[str, float, float], ...] = STRENGTH_BANDS) -> str:
    value = abs(float(value))
    for label, low, high in bands:
        if low <= value < high:
            return label
    return bands[-1][0]


def _score_band(score: float) -> str:
    if score >= 90:
        return ">=90"
    if score >= 80:
        return "80-90"
    return "<80"


def _split_for_trade(trade: Mapping[str, Any], dates: list[str]) -> str:
    train, validation, test = _split_dates(dates)
    date = str(trade["exit_date"])
    if date in train:
        return "train"
    if date in validation:
        return "validation"
    if date in test:
        return "test"
    return "other"


def _excursions(trade: Mapping[str, Any], candles: pd.DataFrame) -> dict[str, Any]:
    frame = candles.sort_values("time").reset_index(drop=True)
    times = frame["time"].astype(str).tolist()
    entry = times.index(str(trade["entry_date"]))
    exit_ = times.index(str(trade["exit_date"]))
    risk = abs(float(trade["entry_price"]) - float(trade.get("initial_stop_price", trade["entry_price"])))
    if risk <= 0:
        risk = abs(float(trade["entry_price"]) - float(trade["exit_price"])) or 1.0
    side = str(trade["side"])
    favorable: list[float] = []
    adverse: list[float] = []
    ambiguity = False
    for _, row in frame.iloc[entry + 1 : exit_ + 1].iterrows():
        if side == "long":
            fav = (float(row["high"]) - float(trade["entry_price"])) / risk
            adv = (float(trade["entry_price"]) - float(row["low"])) / risk
        else:
            fav = (float(trade["entry_price"]) - float(row["low"])) / risk
            adv = (float(row["high"]) - float(trade["entry_price"])) / risk
        favorable.append(fav)
        adverse.append(adv)
        ambiguity = ambiguity or (fav > 0 and adv > 0)
    mfe = max(favorable, default=0.0)
    mae = max(adverse, default=0.0)
    return {
        "holding_trading_days": max(0, exit_ - entry),
        "mfe_r": mfe,
        "mae_r": mae,
        "reached_0_5r": mfe >= 0.5,
        "reached_1r": mfe >= 1.0,
        "reached_2r": mfe >= 2.0,
        "positive_excursion_closed_negative": mfe > 0 and float(trade["pnl"]) < 0,
        "giveback_r": mfe - float(trade.get("r", 0.0)),
        "same_candle_ohlc_ambiguous": ambiguity,
    }


def _group_metrics(trades: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for trade in trades:
        groups[str(trade.get(key, "other"))].append(trade)
    out: dict[str, dict[str, Any]] = {}
    total_loss = abs(sum(float(t["pnl"]) for t in trades if float(t["pnl"]) < 0))
    for name in sorted(groups):
        rows = groups[name]
        profits = [float(t["pnl"]) for t in rows]
        gp = sum(x for x in profits if x > 0)
        gl = abs(sum(x for x in profits if x < 0))
        rs = [float(t.get("r", 0.0)) for t in rows]
        out[name] = {
            "trades": len(rows), "wins": sum(x > 0 for x in profits),
            "win_rate": sum(x > 0 for x in profits) / len(rows), "profit": sum(profits),
            "gross_profit": gp, "gross_loss": gl, "pf": gp / gl if gl else (None if gp else 0.0),
            "avg_r": sum(rs) / len(rs), "median_r": median(rs),
            "total_loss_contribution": gl / total_loss if total_loss else 0.0,
        }
    return out


def _split_evidence(trades: list[dict[str, Any]], predicate) -> dict[str, Any]:
    result = {}
    directions = []
    for period in ("train", "validation", "test"):
        rows = [t for t in trades if t["period"] == period]
        value = sum(bool(predicate(t)) for t in rows) / len(rows) if rows else None
        sufficient = len(rows) >= 5
        direction = "positive" if value is not None and value >= .5 else "negative" if value is not None else "unknown"
        result[period] = {"value": value, "trade_count": len(rows), "direction": direction, "sample_sufficient": sufficient}
        if sufficient:
            directions.append(direction)
    consistency = "insufficient_sample" if len(directions) < 2 else "consistent" if len(set(directions)) == 1 else "mixed"
    return {"train": result["train"], "validation": result["validation"], "test": result["test"], "consistency": consistency}


def _mechanism_evaluation(trades: list[dict[str, Any]], base: dict[str, Any], high: dict[str, Any]) -> list[dict[str, Any]]:
    losers = [t for t in trades if t["pnl"] < 0]
    def rate(rows, predicate):
        return sum(bool(predicate(t)) for t in rows) / len(rows) if rows else 0.0
    def evaluate(ident, mechanism, rows, metrics, rationale, predicate, confidence=None):
        split = _split_evidence(trades, predicate)
        confidence = confidence or ("insufficient_sample" if len(rows) < 5 else "moderate")
        return {"id": ident, "mechanism": mechanism, "status": "insufficient_evidence" if len(rows) < 5 else "evaluated", "relevant_trades": len(rows), "metrics": metrics, **split, "confidence": confidence, "rationale": rationale}
    base_m = base["metrics"]; high_m = high["metrics"]
    cost = {"BASE": {k: base_m[k] for k in ("trades", "pf", "profit", "cagr", "max_dd")}, "HIGH_COST": {k: high_m[k] for k in ("trades", "pf", "profit", "cagr", "max_dd")}}
    cost["DELTA"] = {k: cost["HIGH_COST"][k] - cost["BASE"][k] for k in ("pf", "profit", "cagr", "max_dd")}
    return [
        evaluate("A", "poor entry selection", losers, {"hard_stop_rate": rate(losers, lambda t: t["exit_reason"] == "hard_stop"), "near_immediate_loser_rate": rate(losers, lambda t: t["holding_trading_days"] <= 2), "loser_median_mfe": median([t["mfe_r"] for t in losers]), "loser_median_mae": median([t["mae_r"] for t in losers]), "low_mfe_high_mae_rate": rate(losers, lambda t: t["mfe_r"] < .5 and t["mae_r"] > 1), "mfe_below_0_5r_rate": rate(losers, lambda t: t["mfe_r"] < .5)}, "Observed low-excursion and early-loss rates; diagnostic only.", lambda t: t["pnl"] < 0 and t["mfe_r"] < .5),
        evaluate("B", "exit / giveback behavior", losers, {"positive_mfe_then_negative_rate": rate(losers, lambda t: t["positive_excursion_closed_negative"]), "reached_0_5r_then_negative_rate": rate(losers, lambda t: t["reached_0_5r"]), "reached_1r_then_negative_rate": rate(losers, lambda t: t["reached_1r"]), "reached_2r_then_negative_rate": rate(losers, lambda t: t["reached_2r"]), "median_giveback_r": median([t["giveback_r"] for t in losers])}, "Positive excursion is measured after entry and never feeds decisions.", lambda t: t["positive_excursion_closed_negative"]),
        evaluate("C", "regime misclassification", trades, {"by_regime": _group_metrics(trades, "regime")}, "Compare regime metrics and split direction.", lambda t: t["pnl"] < 0),
        evaluate("D", "currency-strength filter weakness", trades, {"by_strength_gap_band": _group_metrics(trades, "strength_gap_band")}, "Compare deterministic absolute strength-gap bands.", lambda t: abs(t["strength_gap"]) >= .5),
        evaluate("E", "score ranking weakness", trades, {"by_score_band": _group_metrics(trades, "score_band"), "score_summary": _summary([t["score"] for t in trades]), "winner_score_summary": _summary([t["score"] for t in trades if t["pnl"] > 0]), "loser_score_summary": _summary([t["score"] for t in losers])}, "Compare observed score bands without threshold changes.", lambda t: t["score"] >= 90),
        evaluate("F", "pair-specific concentration", losers, {"by_pair": _group_metrics(trades, "instrument")}, "Pair evidence is bounded by trade count.", lambda t: t["pnl"] < 0),
        evaluate("G", "time-period / structural instability", trades, {"by_period": _group_metrics(trades, "period"), "by_year": _group_metrics(trades, "year")}, "Compare calendar year and Train/Validation/Test.", lambda t: t["pnl"] < 0),
        evaluate("H", "transaction-cost sensitivity", trades, cost, "Reuses Research v1 base/high-cost simulations; no sweep.", lambda t: t["pnl"] < 0),
        evaluate("I", "portfolio / risk interaction", trades, {"simulator_diagnostics": base["diagnostics"], "risk_fraction_summary": _summary([t["risk_fraction"] for t in trades])}, "Reuses existing gate diagnostics; no counterfactual run.", lambda t: t["risk_fraction"] > 0),
    ]


def _hypotheses(evaluations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ranked = sorted(evaluations, key=lambda e: (-int(e["status"] == "evaluated"), -e["relevant_trades"], e["id"]))
    return [{"id": e["id"], "failure_mechanism": e["mechanism"], "evidence_for": e["metrics"], "evidence_against": {"consistency": e["consistency"]}, "relevant_trade_count": e["relevant_trades"], "train": e["train"], "validation": e["validation"], "test": e["test"], "consistency": e["consistency"], "confidence": e["confidence"], "single_variable_next_experiment": "Test one variable related to this mechanism while freezing all other rules.", "expected_causal_mechanism": e["rationale"], "falsification_condition": "Direction is absent or mixed in at least two sufficiently sampled splits."} for e in ranked]


def analyze_failure(candles_by_instrument: Mapping[str, pd.DataFrame], settings: Settings, initial_equity: float = 1_000_000.0) -> dict[str, Any]:
    result = _simulate(candles_by_instrument, settings, initial_equity=initial_equity, spread_pips=DEFAULT_SPREAD_PIPS, slippage_pips=DEFAULT_SLIPPAGE_PIPS)
    high_cost = _simulate(candles_by_instrument, settings, initial_equity=initial_equity, spread_pips=HIGH_COST_SPREAD_PIPS, slippage_pips=HIGH_COST_SLIPPAGE_PIPS)
    dates = result["dates"]
    trades: list[dict[str, Any]] = []
    for raw in result["trades"]:
        trade = dict(raw)
        trade.update(_excursions(trade, candles_by_instrument[trade["instrument"]]))
        trade["period"] = _split_for_trade(trade, dates)
        trade["strength_gap_band"] = _band(float(trade.get("strength_gap", 0.0)))
        trade["score_band"] = _score_band(float(trade.get("score", 0.0)))
        trade["year"] = str(trade["exit_date"])[:4]
        trade["holding_band"] = _holding_band(trade["holding_trading_days"])
        trades.append(trade)
    metrics = _metrics(trades, result["equity_curve"], dates[0] if dates else None, dates[-1] if dates else None,
                       equity_first=initial_equity, equity_last=result["equity_curve"][-1] if result["equity_curve"] else initial_equity)
    grouped_period = _group_metrics(trades, "period")
    by_period = {name: grouped_period.get(name, {"trades": 0, "wins": 0, "win_rate": 0.0, "profit": 0.0, "gross_profit": 0.0, "gross_loss": 0.0, "pf": 0.0, "avg_r": 0.0, "median_r": 0.0, "total_loss_contribution": 0.0}) for name in ("train", "validation", "test")}
    high_metrics = _metrics(high_cost["trades"], high_cost["equity_curve"], dates[0] if dates else None, dates[-1] if dates else None, equity_first=initial_equity, equity_last=high_cost["equity_curve"][-1] if high_cost["equity_curve"] else initial_equity)
    evaluations = _mechanism_evaluation(trades, {**result, "metrics": metrics}, {**high_cost, "metrics": high_metrics})
    winners = [t for t in trades if t["pnl"] > 0]; losers = [t for t in trades if t["pnl"] < 0]
    return {"baseline": {"metrics": metrics, "trades": len(trades), "test_trades": by_period.get("test", {}).get("trades", 0)},
            "trades": trades, "by": {"instrument": _group_metrics(trades, "instrument"), "side": _group_metrics(trades, "side"),
            "entry_kind": _group_metrics(trades, "entry_kind"), "exit_reason": _group_metrics(trades, "exit_reason"),
            "regime": _group_metrics(trades, "regime"), "strength_gap_band": _group_metrics(trades, "strength_gap_band"),
            "score_band": _group_metrics(trades, "score_band"), "period": by_period, "year": _group_metrics(trades, "year")},
            "score_summary": _summary([t["score"] for t in trades]),
            "winner_score_summary": _summary([t["score"] for t in winners]),
            "loser_score_summary": _summary([t["score"] for t in losers]),
            "holding_time": {"winner": {"summary": _summary([t["holding_trading_days"] for t in winners]), "bands": _group_metrics(winners, "holding_band")}, "loser": {"summary": _summary([t["holding_trading_days"] for t in losers]), "bands": _group_metrics(losers, "holding_band")}},
            "cost_sensitivity": next(e["metrics"] for e in evaluations if e["id"] == "H"),
            "mechanism_evaluation": evaluations,
            "hypotheses": _hypotheses(evaluations),
            "diagnostics": {"immediate_losers": sum(int(t["holding_trading_days"] <= 2 and t["pnl"] < 0) for t in trades),
            "hard_stop_losses": sum(int(t["exit_reason"] == "hard_stop" and t["pnl"] < 0) for t in trades),
            "positive_excursion_closed_negative": sum(int(t["positive_excursion_closed_negative"]) for t in trades),
            "same_candle_ohlc_ambiguous": sum(int(t["same_candle_ohlc_ambiguous"]) for t in trades)},
            "limitations": ["D1 OHLC cannot prove intraday ordering when favorable and adverse extremes share a candle.", "Historical spread is a cost model, not measured historical spread history.", "Gross leverage cannot be reproduced exactly because complete forward nominal-exposure/conversion information is unavailable."]}


def write_report(payload: dict[str, Any], json_path: str | Path, markdown_path: str | Path) -> None:
    Path(json_path).write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    metrics = payload["baseline"]["metrics"]
    lines = ["# Adaptive v1 Failure Analysis", "", f"Baseline: trades={metrics['trades']} CAGR={metrics['cagr']:.10f} maxDD={metrics['max_dd']:.10f} PF={metrics['pf']:.10f} profit={metrics['profit']:.4f}", "", "## Cost sensitivity", json.dumps(payload["cost_sensitivity"], ensure_ascii=False, indent=2), "", "## A-I mechanism evaluation"]
    for evaluation in payload["mechanism_evaluation"]:
        lines.append(f"- {evaluation['id']} {evaluation['mechanism']}: status={evaluation['status']}, confidence={evaluation['confidence']}, consistency={evaluation['consistency']}, n={evaluation['relevant_trades']}")
    lines += ["", "## Largest loss contributors"]
    for name, rows in payload["by"]["instrument"].items():
        lines.append(f"- {name}: trades={rows['trades']}, profit={rows['profit']:.2f}, loss contribution={rows['total_loss_contribution']:.1%}")
    lines += ["", "## Entry vs exit", f"- immediate losers: {payload['diagnostics']['immediate_losers']}", f"- hard-stop losses: {payload['diagnostics']['hard_stop_losses']}", f"- positive excursion then negative close: {payload['diagnostics']['positive_excursion_closed_negative']}", "", "## Ranked hypotheses"]
    for index, hypothesis in enumerate(payload["hypotheses"], 1):
        lines.append(f"{index}. **{hypothesis['failure_mechanism']}** ({hypothesis['confidence']}, n={hypothesis['relevant_trade_count']}, {hypothesis['consistency']}) — {hypothesis['single_variable_next_experiment']}")
    lines += ["", "## Limitations", *[f"- {item}" for item in payload["limitations"]]]
    Path(markdown_path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Adaptive v1 failure analysis")
    parser.add_argument("--data-dir", default=DEFAULT_RESEARCH_DATA_DIR)
    parser.add_argument("--json-path", default="data/research/fx_adaptive_failure_analysis.json")
    parser.add_argument("--markdown-path", default="data/research/fx_adaptive_failure_analysis.md")
    args = parser.parse_args(argv)
    history = load_history(args.data_dir)
    if not history:
        print(f"NO_DATA: no synced history found under {args.data_dir}")
        return 1
    payload = analyze_failure(history, Settings.from_env(), Settings.from_env().paper_initial_balance)
    write_report(payload, args.json_path, args.markdown_path)
    print(f"FAILURE_ANALYSIS_OK trades={payload['baseline']['metrics']['trades']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
