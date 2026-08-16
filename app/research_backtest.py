from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any, Callable, Mapping

import pandas as pd

from .adaptive_strategy import currency_strength_scores, decide, strength_gap
from .config import Settings
from .instruments import quote_currency
from .models import Position

# Stage 1 explicit transaction-cost assumptions (round-trip cost is split
# across entry/exit fills, mirroring app.paper.PaperBroker._adverse_cost).
DEFAULT_SPREAD_PIPS = 1.0
DEFAULT_SLIPPAGE_PIPS = 0.3
HIGH_COST_SPREAD_PIPS = 3.0
HIGH_COST_SLIPPAGE_PIPS = 0.6

_PIP_JPY = 0.01
_PIP_DEFAULT = 0.0001


def _pip_size(instrument: str) -> float:
    return _PIP_JPY if quote_currency(instrument) == "JPY" else _PIP_DEFAULT


def _adverse_cost(instrument: str, spread_pips: float, slippage_pips: float) -> float:
    return _pip_size(instrument) * (spread_pips / 2 + slippage_pips)


@dataclass
class _OpenTrade:
    instrument: str
    side: str
    entry_price: float
    stop_price: float
    initial_stop_price: float
    opened_at: str
    regime: str
    entry_kind: str | None
    score: float
    risk_fraction: float
    risk_dollars: float
    price_per_risk_unit: float


def _common_dates(candles_by_instrument: Mapping[str, pd.DataFrame]) -> list[str]:
    sets = [
        set(df["time"].astype(str)) for df in candles_by_instrument.values() if len(df)
    ]
    if not sets:
        return []
    common = set.intersection(*sets)
    return sorted(common)


def _simulate(
    candles_by_instrument: Mapping[str, pd.DataFrame],
    settings: Settings,
    *,
    initial_equity: float,
    spread_pips: float,
    slippage_pips: float,
) -> dict[str, Any]:
    """Replay bars chronologically and reuse app.adaptive_strategy.decide as-is.

    No-lookahead guarantee: at each simulated date `t`, every instrument's
    dataframe is freshly sliced to rows with time <= t before being handed to
    `decide()`/`currency_strength_scores()`. Nothing beyond `t` is ever
    constructed or read, so a decision at `t` cannot be influenced by bars
    that have not "happened" yet in the replay.
    """
    dates = _common_dates(candles_by_instrument)
    sorted_frames = {
        instrument: df.sort_values("time").reset_index(drop=True)
        for instrument, df in candles_by_instrument.items()
    }

    equity = float(initial_equity)
    equity_curve: list[float] = []
    equity_dates: list[str] = []
    trades: list[dict[str, Any]] = []
    open_trades: dict[str, _OpenTrade] = {}

    def _close_trade(instrument: str, fill: float, current_date: str, reason: str) -> None:
        nonlocal equity
        trade = open_trades.pop(instrument)
        direction = 1 if trade.side == "long" else -1
        pnl = (fill - trade.entry_price) * direction * trade.price_per_risk_unit
        equity += pnl
        trades.append(
            {
                "instrument": instrument,
                "side": trade.side,
                "regime": trade.regime,
                "entry_kind": trade.entry_kind,
                "score": trade.score,
                "entry_date": trade.opened_at,
                "exit_date": current_date,
                "entry_price": trade.entry_price,
                "exit_price": fill,
                "pnl": pnl,
                "r": pnl / trade.risk_dollars if trade.risk_dollars else 0.0,
                "exit_reason": reason,
            }
        )
        equity_curve.append(equity)
        equity_dates.append(current_date)

    for current_date in dates:
        slices: dict[str, pd.DataFrame] = {}
        for instrument, df in sorted_frames.items():
            sliced = df[df["time"] <= current_date]
            if sliced.empty:
                continue
            slices[instrument] = sliced
        if not slices:
            continue

        strengths = currency_strength_scores(slices, settings)
        candidates: list[tuple[str, Any]] = []

        for instrument, sliced in slices.items():
            row = sliced.iloc[-1]
            if str(row["time"]) != current_date:
                continue

            stopped_today = False
            trade = open_trades.get(instrument)
            if trade is not None:
                low = float(row["low"])
                high = float(row["high"])
                open_price = float(row["open"])
                cost = _adverse_cost(instrument, spread_pips, slippage_pips)
                fill = None
                if trade.side == "long" and low <= trade.stop_price:
                    fill = min(trade.stop_price, open_price) - cost
                elif trade.side == "short" and high >= trade.stop_price:
                    fill = max(trade.stop_price, open_price) + cost
                if fill is not None:
                    _close_trade(instrument, fill, current_date, "hard_stop")
                    stopped_today = True

            trade = open_trades.get(instrument)
            position = None
            metadata = None
            if trade is not None:
                position = Position(
                    instrument=instrument,
                    side=trade.side,
                    units=1,
                    entry_price=trade.entry_price,
                    stop_price=trade.stop_price,
                    opened_at=trade.opened_at,
                    planned_risk_home=0.0,
                )
                metadata = {
                    "regime": trade.regime,
                    "entry_kind": trade.entry_kind,
                    "initial_stop_price": trade.initial_stop_price,
                    "opened_candle_time": trade.opened_at,
                    "score": trade.score,
                    "risk_fraction": trade.risk_fraction,
                }

            try:
                decision = decide(
                    sliced,
                    settings,
                    position,
                    pair_strength_gap=strength_gap(instrument, strengths),
                    position_metadata=metadata,
                )
            except ValueError:
                # Not enough warm-up bars yet for this instrument; skip.
                continue

            if position is not None:
                if decision.updated_stop_price is not None:
                    tighter = (
                        decision.updated_stop_price > trade.stop_price
                        if trade.side == "long"
                        else decision.updated_stop_price < trade.stop_price
                    )
                    if tighter:
                        trade.stop_price = float(decision.updated_stop_price)

                if decision.action == "exit":
                    close = float(row["close"])
                    cost = _adverse_cost(instrument, spread_pips, slippage_pips)
                    fill = close - cost if trade.side == "long" else close + cost
                    _close_trade(instrument, fill, current_date, decision.reason)
                continue

            if stopped_today:
                continue

            if decision.action in {"enter_long", "enter_short"}:
                candidates.append((instrument, decision))

        remaining_slots = settings.adaptive_max_open_positions - len(open_trades)
        if remaining_slots > 0 and candidates:
            candidates.sort(key=lambda item: item[1].score, reverse=True)
            for instrument, decision in candidates[:remaining_slots]:
                row = slices[instrument].iloc[-1]
                close = float(row["close"])
                side = "long" if decision.action == "enter_long" else "short"
                stop_multiple = decision.stop_atr_multiple or settings.atr_stop_multiple
                stop_price = (
                    close - stop_multiple * decision.atr
                    if side == "long"
                    else close + stop_multiple * decision.atr
                )
                stop_distance = abs(close - stop_price)
                risk_fraction = float(decision.risk_fraction or 0.0)
                if stop_distance <= 0 or risk_fraction <= 0:
                    continue
                cost = _adverse_cost(instrument, spread_pips, slippage_pips)
                entry_price = close + cost if side == "long" else close - cost
                risk_dollars = risk_fraction * equity
                open_trades[instrument] = _OpenTrade(
                    instrument=instrument,
                    side=side,
                    entry_price=entry_price,
                    stop_price=stop_price,
                    initial_stop_price=stop_price,
                    opened_at=current_date,
                    regime=decision.regime,
                    entry_kind=decision.entry_kind,
                    score=decision.score,
                    risk_fraction=risk_fraction,
                    risk_dollars=risk_dollars,
                    price_per_risk_unit=risk_dollars / stop_distance,
                )

    return {
        "trades": trades,
        "equity_curve": equity_curve,
        "equity_dates": equity_dates,
        "final_equity": equity,
        "dates": dates,
    }


def _max_dd(equity: list[float]) -> float:
    peak = equity[0] if equity else 0.0
    dd = 0.0
    for x in equity:
        peak = max(peak, x)
        if peak:
            dd = max(dd, 1 - x / peak)
    return dd


def _cagr(equity_first: float, equity_last: float, start_date: str | None, end_date: str | None) -> float:
    if equity_first <= 0 or not start_date or not end_date:
        return 0.0
    d0 = datetime.fromisoformat(start_date)
    d1 = datetime.fromisoformat(end_date)
    days = max(1, (d1 - d0).days)
    years = days / 365.25
    if years <= 0:
        return 0.0
    return (equity_last / equity_first) ** (1.0 / years) - 1.0


def _metrics(
    trades: list[dict[str, Any]],
    equity_curve: list[float],
    start_date: str | None,
    end_date: str | None,
) -> dict[str, Any]:
    gp = sum(t["pnl"] for t in trades if t["pnl"] > 0)
    gl = abs(sum(t["pnl"] for t in trades if t["pnl"] < 0))
    wins = [t for t in trades if t["pnl"] > 0]
    equity_first = equity_curve[0] if equity_curve else 0.0
    equity_last = equity_curve[-1] if equity_curve else equity_first
    # pf is left as None (not float("inf")) when there are winners but no
    # losers, since "Infinity" is not valid JSON and would break the
    # dashboard's fetch(...).json() call.
    pf = (gp / gl) if gl else (None if gp else 0.0)
    return {
        "cagr": _cagr(equity_first, equity_last, start_date, end_date),
        "max_dd": _max_dd(equity_curve),
        "pf": pf,
        "win_rate": len(wins) / len(trades) if trades else 0.0,
        "trades": len(trades),
        "avg_r": mean(t["r"] for t in trades) if trades else 0.0,
        "profit": sum(t["pnl"] for t in trades),
    }


def _annual(trades: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for t in trades:
        y = t["exit_date"][:4]
        out.setdefault(y, {"trades": 0, "pnl": 0.0, "wins": 0})
        out[y]["trades"] += 1
        out[y]["pnl"] += t["pnl"]
        out[y]["wins"] += int(t["pnl"] > 0)
    for y, x in out.items():
        x["win_rate"] = x["wins"] / x["trades"] if x["trades"] else 0.0
        del x["wins"]
    return out


def _group_metrics(
    trades: list[dict[str, Any]], key_fn: Callable[[dict[str, Any]], str]
) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for t in trades:
        groups[key_fn(t)].append(t)
    result: dict[str, Any] = {}
    for key, group_trades in groups.items():
        gp = sum(t["pnl"] for t in group_trades if t["pnl"] > 0)
        gl = abs(sum(t["pnl"] for t in group_trades if t["pnl"] < 0))
        wins = [t for t in group_trades if t["pnl"] > 0]
        pf = (gp / gl) if gl else (None if gp else 0.0)
        result[key] = {
            "trades": len(group_trades),
            "win_rate": len(wins) / len(group_trades) if group_trades else 0.0,
            "pf": pf,
            "avg_r": mean(t["r"] for t in group_trades) if group_trades else 0.0,
            "profit": sum(t["pnl"] for t in group_trades),
        }
    return result


def _score_band(score: float, settings: Settings) -> str:
    if score >= settings.adaptive_score_high:
        return f">={settings.adaptive_score_high:.0f}"
    if score >= settings.adaptive_score_medium:
        return f"{settings.adaptive_score_medium:.0f}-{settings.adaptive_score_high:.0f}"
    return f"{settings.adaptive_score_min:.0f}-{settings.adaptive_score_medium:.0f}"


def _split_dates(dates: list[str]) -> tuple[set[str], set[str], set[str]]:
    if len(dates) < 3:
        return set(dates), set(), set()
    a = int(len(dates) * 0.6)
    b = int(len(dates) * 0.8)
    return set(dates[:a]), set(dates[a:b]), set(dates[b:])


def _baseline_config(settings: Settings) -> dict[str, Any]:
    return {
        "strategy_profile": "adaptive_dual_regime_v1",
        "adaptive_strength_gap_min": settings.adaptive_strength_gap_min,
        "adaptive_score_min": settings.adaptive_score_min,
        "adaptive_score_medium": settings.adaptive_score_medium,
        "adaptive_score_high": settings.adaptive_score_high,
        "adaptive_trend_adx": settings.adaptive_trend_adx,
        "adaptive_range_adx": settings.adaptive_range_adx,
        "adaptive_max_open_positions": settings.adaptive_max_open_positions,
    }


def _empty_result(settings: Settings) -> dict[str, Any]:
    return {
        "period": {},
        "metrics": {},
        "annual": {},
        "by_symbol": {},
        "by_regime": {},
        "by_score_band": {},
        "sensitivity": {},
        "train": {},
        "validation": {},
        "test": {},
        "walk_forward": {},
        "cost_assumptions": {},
        "baseline_config": _baseline_config(settings),
    }


def run_research(
    candles_by_instrument: Mapping[str, pd.DataFrame],
    settings: Settings,
    initial_equity: float = 1_000_000.0,
) -> dict[str, Any]:
    if not candles_by_instrument:
        return _empty_result(settings)

    base = _simulate(
        candles_by_instrument,
        settings,
        initial_equity=initial_equity,
        spread_pips=DEFAULT_SPREAD_PIPS,
        slippage_pips=DEFAULT_SLIPPAGE_PIPS,
    )
    high_cost = _simulate(
        candles_by_instrument,
        settings,
        initial_equity=initial_equity,
        spread_pips=HIGH_COST_SPREAD_PIPS,
        slippage_pips=HIGH_COST_SLIPPAGE_PIPS,
    )

    trades = base["trades"]
    dates = base["dates"]
    start_date = dates[0] if dates else None
    end_date = dates[-1] if dates else None
    metrics = _metrics(trades, base["equity_curve"], start_date, end_date)
    high_cost_metrics = _metrics(
        high_cost["trades"], high_cost["equity_curve"], start_date, end_date
    )

    train_dates, validation_dates, test_dates = _split_dates(dates)

    def _period_result(period_dates: set[str]) -> dict[str, Any]:
        subset = [t for t in trades if t["exit_date"] in period_dates]
        subset_equity = [
            e for e, d in zip(base["equity_curve"], base["equity_dates"]) if d in period_dates
        ]
        p_start = min(period_dates) if period_dates else None
        p_end = max(period_dates) if period_dates else None
        return {
            "trades": len(subset),
            "metrics": _metrics(subset, subset_equity, p_start, p_end),
            "annual": _annual(subset),
        }

    train_result = _period_result(train_dates)
    validation_result = _period_result(validation_dates)
    test_result = _period_result(test_dates)

    fold_dates: list[set[str]] = []
    if test_dates:
        test_list = sorted(test_dates)
        fold_size = max(1, len(test_list) // 5)
        for i in range(5):
            start = i * fold_size
            end = len(test_list) if i == 4 else min(len(test_list), (i + 1) * fold_size)
            fold_dates.append(set(test_list[start:end]))
    folds = []
    for idx, fold in enumerate(fold_dates, start=1):
        folds.append({"fold": idx, "dates": len(fold), **_period_result(fold)})
    fold_positive = sum(1 for f in folds if f["metrics"]["cagr"] > 0)

    return {
        "period": {"start": start_date, "end": end_date, "trading_days": len(dates)},
        "metrics": metrics,
        "annual": _annual(trades),
        "by_symbol": _group_metrics(trades, lambda t: t["instrument"]),
        "by_regime": _group_metrics(trades, lambda t: t["regime"]),
        "by_score_band": _group_metrics(trades, lambda t: _score_band(t["score"], settings)),
        "sensitivity": {
            "base": {
                "cost_pips": {
                    "spread": DEFAULT_SPREAD_PIPS,
                    "slippage": DEFAULT_SLIPPAGE_PIPS,
                },
                "metrics": metrics,
            },
            "cost_x2": {
                "cost_pips": {
                    "spread": HIGH_COST_SPREAD_PIPS,
                    "slippage": HIGH_COST_SLIPPAGE_PIPS,
                },
                "metrics": high_cost_metrics,
            },
        },
        "train": train_result,
        "validation": validation_result,
        "test": test_result,
        "walk_forward": {
            "folds": folds,
            "fold_positive": fold_positive,
            "fold_total": len(folds),
        },
        "cost_assumptions": {
            "base_spread_pips": DEFAULT_SPREAD_PIPS,
            "base_slippage_pips": DEFAULT_SLIPPAGE_PIPS,
            "high_spread_pips": HIGH_COST_SPREAD_PIPS,
            "high_slippage_pips": HIGH_COST_SLIPPAGE_PIPS,
        },
        "baseline_config": _baseline_config(settings),
    }


def dump_results(results: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def load_cached_results(
    cache_path: str | Path, data_dir: str | Path
) -> dict[str, Any] | None:
    cache_path = Path(cache_path)
    if not cache_path.exists():
        return None
    data_root = Path(data_dir)
    newest_source = 0.0
    if data_root.exists():
        for f in data_root.glob("*.json"):
            newest_source = max(newest_source, f.stat().st_mtime)
    if newest_source and cache_path.stat().st_mtime < newest_source:
        return None
    try:
        return json.loads(cache_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def main(argv: list[str] | None = None) -> int:
    from .fx_research_data import DEFAULT_RESEARCH_DATA_DIR, load_history

    parser = argparse.ArgumentParser(description="Run FX adaptive research backtest")
    parser.add_argument("command", choices=["run"])
    parser.add_argument("--data-dir", default=DEFAULT_RESEARCH_DATA_DIR)
    parser.add_argument(
        "--cache-path", default="data/research/fx_adaptive_backtest.json"
    )
    args = parser.parse_args(argv)

    settings = Settings.from_env()
    history = load_history(args.data_dir)
    if not history:
        print(f"NO_DATA: no synced history found under {args.data_dir}")
        return 1

    payload = run_research(history, settings, initial_equity=settings.paper_initial_balance)
    dump_results(payload, args.cache_path)
    metrics = payload.get("metrics", {})
    print(
        "RESEARCH_OK "
        f"trades={metrics.get('trades', 0)} "
        f"cagr={metrics.get('cagr')} "
        f"max_dd={metrics.get('max_dd')} "
        f"pf={metrics.get('pf')}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
