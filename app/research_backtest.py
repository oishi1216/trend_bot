from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from statistics import mean
import csv
import json
from typing import Any


@dataclass(frozen=True)
class Bar:
    date: date
    open: float
    high: float
    low: float
    close: float
    volume: float
    value: float


@dataclass(frozen=True)
class StrategySpec:
    name: str
    hold_days: int
    entry_rank_band: float
    exit_rule: str


STRATEGIES = {
    "A Event": StrategySpec("A Event", 5, 0.10, "event"),
    "B Momentum": StrategySpec("B Momentum", 20, 0.20, "momentum"),
    "C Pullback": StrategySpec("C Pullback", 5, 0.30, "pullback"),
}


def load_universe(data_dir: str | Path) -> dict[str, list[Bar]]:
    root = Path(data_dir)
    universe: dict[str, list[Bar]] = {}
    for path in sorted(root.glob("*.csv")):
        rows: list[Bar] = []
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                rows.append(
                    Bar(
                        date=datetime.strptime(row["date"], "%Y-%m-%d").date(),
                        open=float(row["open"]),
                        high=float(row["high"]),
                        low=float(row["low"]),
                        close=float(row["close"]),
                        volume=float(row.get("volume") or 0),
                        value=float(row.get("value") or 0),
                    )
                )
        if rows:
            universe[path.stem] = rows
    return universe


def _sma(values) -> float:
    values = list(values)
    return mean(values) if values else 0.0


def _ret(a: float, b: float) -> float:
    return 0.0 if a <= 0 else b / a - 1.0


def _atr(bars: list[Bar], idx: int, days: int = 20) -> float:
    start = max(1, idx - days + 1)
    trs = []
    prev = bars[start - 1].close
    for bar in bars[start : idx + 1]:
        trs.append(
            max(bar.high - bar.low, abs(bar.high - prev), abs(bar.low - prev))
        )
        prev = bar.close
    return _sma(trs)


def _avg_value(bars: list[Bar], idx: int, days: int = 20) -> float:
    return _sma(bar.value for bar in bars[max(0, idx - days + 1) : idx + 1])


def _max_dd(equity: list[float]) -> float:
    peak = equity[0] if equity else 0.0
    dd = 0.0
    for x in equity:
        peak = max(peak, x)
        if peak:
            dd = max(dd, 1 - x / peak)
    return dd


def _annual(trades: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for t in trades:
        y = t["entry_date"][:4]
        out.setdefault(y, {"trades": 0, "pnl": 0.0, "wins": 0})
        out[y]["trades"] += 1
        out[y]["pnl"] += t["pnl"]
        out[y]["wins"] += int(t["pnl"] > 0)
    for y, x in out.items():
        x["win_rate"] = x["wins"] / x["trades"] if x["trades"] else 0.0
        del x["wins"]
    return out


def _concentration(trades: list[dict[str, Any]]) -> dict[str, Any]:
    prof: dict[str, float] = defaultdict(float)
    total = sum(max(0.0, t["pnl"]) for t in trades)
    for t in trades:
        prof[t["symbol"]] += max(0.0, t["pnl"])
    vals = sorted(prof.values(), reverse=True)
    return {
        "top1_share": (vals[0] / total) if total and vals else 0.0,
        "top3_share": (sum(vals[:3]) / total) if total else 0.0,
    }


def _metrics(trades: list[dict[str, Any]], equity: list[float]) -> dict[str, Any]:
    gp = sum(t["pnl"] for t in trades if t["pnl"] > 0)
    gl = abs(sum(t["pnl"] for t in trades if t["pnl"] < 0))
    wins = [t for t in trades if t["pnl"] > 0]
    cagr = 0.0
    if len(equity) >= 2 and equity[0] > 0:
        cagr = (equity[-1] / equity[0]) ** (252 / max(1, len(equity))) - 1
    return {
        "cagr": cagr,
        "max_dd": _max_dd(equity),
        "pf": (gp / gl) if gl else (float("inf") if gp else 0.0),
        "win_rate": len(wins) / len(trades) if trades else 0.0,
        "trades": len(trades),
        "avg_r": _sma(t["r"] for t in trades) if trades else 0.0,
        "profit": sum(t["pnl"] for t in trades),
    }


def _split_dates(dates: list[date]) -> tuple[set[date], set[date], set[date]]:
    if len(dates) < 3:
        return set(dates), set(), set()
    a = int(len(dates) * 0.6)
    b = int(len(dates) * 0.8)
    return set(dates[:a]), set(dates[a:b]), set(dates[b:])


def _window_for_split(dates: list[date], split: str) -> set[date]:
    train, validation, test = _split_dates(dates)
    return {"train": train, "validation": validation, "test": test}[split]


def _liquid_symbols(universe: dict[str, list[Bar]], min_value: float = 50_000_000) -> list[str]:
    return [
        symbol
        for symbol, bars in universe.items()
        if bars and _avg_value(bars, len(bars) - 1, 20) >= min_value
    ]


def _signal_rows(
    universe: dict[str, list[Bar]],
    allowed_dates: set[date],
    *,
    event_shift: float = 0.0,
    cost_mult: float = 1.0,
) -> dict[str, list[dict[str, Any]]]:
    symbols = _liquid_symbols(universe)
    dates = sorted({bar.date for bars in universe.values() for bar in bars if bar.date in allowed_dates})
    per_strategy: dict[str, list[dict[str, Any]]] = defaultdict(list)
    if not dates:
        return per_strategy

    for current in dates[120:]:
        ranked: list[dict[str, Any]] = []
        for sym in symbols:
            bars = universe[sym]
            idx = next((i for i, b in enumerate(bars) if b.date == current), None)
            if idx is None or idx < 120 or idx + 1 >= len(bars):
                continue
            close = bars[idx].close
            ranked.append(
                {
                    "sym": sym,
                    "idx": idx,
                    "close": close,
                    "r5": _ret(bars[idx - 5].close, close) if idx >= 5 else 0.0,
                    "r20": _ret(bars[idx - 20].close, close),
                    "r120": _ret(bars[idx - 120].close, close),
                    "ma60": _sma(b.close for b in bars[idx - 59 : idx + 1]),
                    "value20": _avg_value(bars, idx, 20),
                    "atr20": _atr(bars, idx, 20) * cost_mult,
                }
            )
        if not ranked:
            continue

        by20 = sorted(ranked, key=lambda x: x["r20"], reverse=True)
        by120 = sorted(ranked, key=lambda x: x["r120"], reverse=True)
        by_value = sorted(ranked, key=lambda x: x["value20"], reverse=True)

        market_weak = mean(x["r20"] for x in by20[: min(20, len(by20))]) < 0
        cutoff_a = by20[max(0, len(by20) // 10 - 1)]["r20"]
        cutoff_c = sorted(x["r5"] for x in ranked)[max(0, len(ranked) // 5 - 1)]

        top_a = {x["sym"] for x in by20[: max(1, len(by20) // 10)]}
        top_b = {x["sym"] for x in by120[: max(1, len(by120) // 5)]}
        top_c = {x["sym"] for x in by120[: max(1, len(by120) * 3 // 10)]}
        liquid = {x["sym"] for x in by_value[: max(1, len(by_value) // 2)]}

        for row in ranked:
            sym = row["sym"]
            bars = universe[sym]
            entry = bars[row["idx"] + 1]
            atr = max(row["atr20"], 1e-9)
            shares = max(1, int((1_000_000 * 0.0035) / (2 * atr)))
            risk_amt = shares * (2 * atr)

            def add(strategy: str, exit_idx: int, reason: str):
                exit_bar = bars[min(exit_idx, len(bars) - 1)]
                pnl = (exit_bar.close - entry.open) * shares
                per_strategy[strategy].append(
                    {
                        "strategy": strategy,
                        "symbol": sym,
                        "entry_date": entry.date.isoformat(),
                        "exit_date": exit_bar.date.isoformat(),
                        "entry_price": entry.open,
                        "exit_price": exit_bar.close,
                        "pnl": pnl,
                        "r": pnl / risk_amt if risk_amt else 0.0,
                        "reason": reason,
                    }
                )

            if (not market_weak) and sym in top_a and row["r20"] < cutoff_a * (1 + event_shift):
                add("A Event", row["idx"] + STRATEGIES["A Event"].hold_days, "event-drift")
            if sym in top_b and sym in liquid and row["close"] > row["ma60"]:
                add("B Momentum", row["idx"] + STRATEGIES["B Momentum"].hold_days, "weekly-momentum")
            if sym in top_c and row["r5"] <= cutoff_c * (1 + event_shift) and row["close"] > row["ma60"]:
                add("C Pullback", row["idx"] + STRATEGIES["C Pullback"].hold_days, "trend-pullback")
    return per_strategy


def _assemble_result(
    per_strategy: dict[str, list[dict[str, Any]]], initial_equity: float
) -> dict[str, Any]:
    strategies: dict[str, Any] = {}
    portfolio: list[dict[str, Any]] = []
    for name, trades in per_strategy.items():
        equity = [initial_equity]
        for t in sorted(trades, key=lambda x: x["exit_date"]):
            equity.append(equity[-1] + t["pnl"])
        strategies[name] = {
            "trades": trades,
            "equity": equity,
            "metrics": _metrics(trades, equity),
            "annual": _annual(trades),
            "concentration": _concentration(trades),
        }
        portfolio.extend(trades)

    peq = [initial_equity]
    for t in sorted(portfolio, key=lambda x: x["exit_date"]):
        peq.append(peq[-1] + t["pnl"])
    return {
        "strategies": strategies,
        "portfolio": {
            "trades": portfolio,
            "equity": peq,
            "metrics": _metrics(portfolio, peq),
            "annual": _annual(portfolio),
            "concentration": _concentration(portfolio),
        },
    }


def _allocation_grid() -> list[dict[str, int]]:
    candidates: list[dict[str, int]] = []
    for a in (0, 20, 40, 60, 80, 100):
        for b in (0, 20, 40, 60, 80, 100 - a):
            c = 100 - a - b
            if c < 0:
                continue
            if c % 20 != 0:
                continue
            candidates.append({"A": a, "B": b, "C": c})
    return candidates


def _portfolio_score(metrics: dict[str, Any]) -> tuple[float, float, float]:
    return (
        metrics.get("cagr", 0.0),
        -metrics.get("max_dd", 0.0),
        metrics.get("pf", 0.0),
    )


def _blend_strategies(result: dict[str, Any], allocation: dict[str, int]) -> dict[str, Any]:
    selected = []
    for name, weight in allocation.items():
        trades = result["strategies"].get(f"{name} Event" if name == "A" else f"{name} Momentum" if name == "B" else f"{name} Pullback", {}).get("trades", [])
        selected.extend((weight, trade) for trade in trades)
    selected.sort(key=lambda item: item[1]["exit_date"])
    equity = [1_000_000.0]
    for weight, trade in selected:
        equity.append(equity[-1] + trade["pnl"] * (weight / 100.0))
    trades = [trade for _, trade in selected]
    return {
        "allocation": allocation,
        "metrics": _metrics(trades, equity),
        "equity": equity,
        "trades": trades,
    }


def _sensitivity_suite(universe: dict[str, list[Bar]], allowed_dates: set[date]) -> dict[str, Any]:
    scenarios = {
        "base": _assemble_result(_signal_rows(universe, allowed_dates), 1_000_000)["portfolio"]["metrics"],
        "cost_x2": _assemble_result(_signal_rows(universe, allowed_dates, cost_mult=2.0), 1_000_000)["portfolio"]["metrics"],
        "loose": _assemble_result(_signal_rows(universe, allowed_dates, event_shift=-0.10), 1_000_000)["portfolio"]["metrics"],
        "tight": _assemble_result(_signal_rows(universe, allowed_dates, event_shift=0.10), 1_000_000)["portfolio"]["metrics"],
    }
    return scenarios


def run_backtest(
    universe: dict[str, list[Bar]],
    initial_equity: float = 1_000_000,
    risk_per_trade: float = 0.0035,
) -> dict[str, Any]:
    del risk_per_trade
    dates = sorted({bar.date for bars in universe.values() for bar in bars})
    if not dates:
        return {"universe_size": 0, "strategies": {}, "portfolio": {}}

    train_dates, validation_dates, test_dates = _split_dates(dates)
    train_raw = _signal_rows(universe, train_dates)
    validation_raw = _signal_rows(universe, validation_dates)
    test_raw = _signal_rows(universe, test_dates)
    train_result = _assemble_result(train_raw, initial_equity)
    validation_result = _assemble_result(validation_raw, initial_equity)
    test_result = _assemble_result(test_raw, initial_equity)

    fold_dates = []
    if test_dates:
        test_dates_list = sorted(test_dates)
        fold_size = max(1, len(test_dates) // 5)
        for i in range(5):
            start = i * fold_size
            end = len(test_dates_list) if i == 4 else min(len(test_dates_list), (i + 1) * fold_size)
            fold_dates.append(set(test_dates_list[start:end]))
    folds = []
    for idx, fold in enumerate(fold_dates, start=1):
        fold_result = _assemble_result(_signal_rows(universe, fold), initial_equity)
        folds.append(
            {
                "fold": idx,
                "dates": len(fold),
                "metrics": fold_result["portfolio"]["metrics"],
            }
        )

    grid = []
    for allocation in _allocation_grid():
        grid.append(
            {
                "allocation": allocation,
                "score": _portfolio_score(_blend_strategies(validation_result, allocation)["metrics"]),
            }
        )
    best_allocation = max(grid, key=lambda item: item["score"])["allocation"] if grid else {"A": 40, "B": 40, "C": 20}

    portfolio_blend = _blend_strategies(test_result, best_allocation)
    if portfolio_blend["trades"]:
        one_symbol = sorted(
            defaultdict(float, {
                trade["symbol"]: sum(max(0.0, t["pnl"]) for t in portfolio_blend["trades"] if t["symbol"] == trade["symbol"])
                for trade in portfolio_blend["trades"]
            }).values(),
            reverse=True,
        )
        total_pos = sum(max(0.0, t["pnl"]) for t in portfolio_blend["trades"])
        top1_share = one_symbol[0] / total_pos if total_pos and one_symbol else 0.0
    else:
        top1_share = 0.0

    combined = {
        "train": train_result,
        "validation": validation_result,
        "test": test_result,
        "walk_forward": {
            "best_allocation": best_allocation,
            "grid": grid,
            "out_of_sample": portfolio_blend,
            "top1_share": top1_share,
            "folds": folds,
        },
        "sensitivity": _sensitivity_suite(universe, test_dates),
        "annual_stability": {
            "train_positive_year_ratio": _positive_year_ratio(train_result["portfolio"]["annual"]),
            "validation_positive_year_ratio": _positive_year_ratio(validation_result["portfolio"]["annual"]),
            "test_positive_year_ratio": _positive_year_ratio(test_result["portfolio"]["annual"]),
        },
    }

    combined["strategies"] = train_result["strategies"]
    combined["portfolio"] = test_result["portfolio"]
    combined["universe_size"] = len(_liquid_symbols(universe))
    return combined


def _positive_year_ratio(annual: dict[str, Any]) -> float:
    if not annual:
        return 0.0
    positives = [1 for stats in annual.values() if stats.get("pnl", 0.0) > 0]
    return len(positives) / len(annual)


def dump_results(results: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
