from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class SizeResult:
    units: int
    risk_budget_home: float
    estimated_risk_home: float
    reason: str


def calculate_units(
    *,
    nav_home: float,
    entry_price: float,
    stop_price: float,
    quote_to_home: float,
    current_planned_risk_home: float,
    current_gross_notional_home: float,
    risk_per_trade: float,
    max_aggregate_risk: float,
    max_gross_leverage: float,
) -> SizeResult:
    if nav_home <= 0 or entry_price <= 0 or quote_to_home <= 0:
        return SizeResult(0, 0, 0, "Invalid NAV, entry price, or conversion rate")

    stop_distance_quote = abs(entry_price - stop_price)
    if stop_distance_quote <= 0:
        return SizeResult(0, 0, 0, "Stop distance is zero")

    per_trade_budget = nav_home * risk_per_trade
    aggregate_remaining = nav_home * max_aggregate_risk - current_planned_risk_home
    risk_budget = min(per_trade_budget, aggregate_remaining)
    if risk_budget <= 0:
        return SizeResult(0, 0, 0, "Aggregate risk cap reached")

    loss_per_unit_home = stop_distance_quote * quote_to_home
    units_by_risk = math.floor(risk_budget / loss_per_unit_home)

    # One base-currency unit is entry_price quote units, then converted to home.
    notional_per_unit_home = entry_price * quote_to_home
    leverage_remaining = nav_home * max_gross_leverage - current_gross_notional_home
    units_by_leverage = math.floor(max(0.0, leverage_remaining) / notional_per_unit_home)

    units = max(0, min(units_by_risk, units_by_leverage))
    if units < 1:
        return SizeResult(0, risk_budget, 0, "Risk or leverage cap leaves less than one unit")

    estimated = units * loss_per_unit_home
    reason = f"Sized by min(risk={units_by_risk}, leverage={units_by_leverage})"
    return SizeResult(units, risk_budget, estimated, reason)
