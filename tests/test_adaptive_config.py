from app.config import Settings


def test_adaptive_profile_uses_ten_mt5_pairs_and_long_history(monkeypatch):
    monkeypatch.setenv("BROKER_MODE", "mt5_paper")
    monkeypatch.setenv("STRATEGY_PROFILE", "adaptive_dual_regime_v1")
    monkeypatch.delenv("MT5_INSTRUMENTS", raising=False)
    monkeypatch.delenv("MARKET_DATA_CANDLE_COUNT", raising=False)

    settings = Settings.from_env()

    assert len(settings.instruments) == 10
    assert settings.market_data_candle_count == 3200
    assert settings.adaptive_enabled


def test_legacy_defaults_are_unchanged(monkeypatch):
    monkeypatch.setenv("BROKER_MODE", "mt5_paper")
    monkeypatch.delenv("STRATEGY_PROFILE", raising=False)
    monkeypatch.delenv("MT5_INSTRUMENTS", raising=False)
    monkeypatch.delenv("MARKET_DATA_CANDLE_COUNT", raising=False)

    settings = Settings.from_env()

    assert settings.strategy_profile == "legacy_trend_v1"
    assert len(settings.instruments) == 4
    assert settings.market_data_candle_count == 320
