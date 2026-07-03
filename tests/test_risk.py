from app.risk import calculate_units


def test_risk_position_size():
    result = calculate_units(
        nav_home=1_000_000,
        entry_price=150.0,
        stop_price=148.0,
        quote_to_home=1.0,
        current_planned_risk_home=0,
        current_gross_notional_home=0,
        risk_per_trade=0.0025,
        max_aggregate_risk=0.0075,
        max_gross_leverage=2.0,
    )
    assert result.units == 1250
    assert result.estimated_risk_home == 2500
