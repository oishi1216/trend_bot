from dataclasses import replace

from app.config import Settings
from app.engine import adaptive_risk_after_drawdown, shared_currency_planned_risk
from app.models import Position


def settings():
    return replace(
        Settings.from_env(),
        strategy_profile="adaptive_dual_regime_v1",
    )


def position(instrument, planned_risk):
    return Position(
        instrument=instrument,
        side="long",
        units=1,
        entry_price=1.0,
        stop_price=0.9,
        opened_at="2026-01-01T00:00:00+00:00",
        planned_risk_home=planned_risk,
    )


def test_adaptive_risk_reduces_with_drawdown():
    config = settings()

    assert adaptive_risk_after_drawdown(0.008, 0.03, config) == 0.008
    assert adaptive_risk_after_drawdown(0.008, 0.05, config) == 0.004
    assert adaptive_risk_after_drawdown(0.008, 0.08, config) == 0.0025
    assert adaptive_risk_after_drawdown(0.008, 0.10, config) == 0.0


def test_shared_currency_risk_counts_each_position_once():
    positions = [
        position("EURUSD", 1000),
        position("USDJPY", 2000),
        position("AUDNZD", 3000),
    ]

    assert shared_currency_planned_risk("GBPUSD", positions) == 3000
    assert shared_currency_planned_risk("AUDJPY", positions) == 5000
