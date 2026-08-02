from __future__ import annotations

import os
from dataclasses import dataclass


def _bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _float(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


def _int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def _tuple_env(name: str, default: str) -> tuple[str, ...]:
    return tuple(
        item.strip().upper()
        for item in os.getenv(name, default).split(",")
        if item.strip()
    )


@dataclass(frozen=True)
class Settings:
    app_name: str
    db_path: str
    host: str
    port: int
    log_level: str
    broker_mode: str
    trading_armed: bool
    allow_live_trading: str
    oanda_account_id: str
    oanda_api_token: str
    oanda_base_url_practice: str
    oanda_base_url_live: str
    account_home_currency: str
    instruments: tuple[str, ...]
    market_data_candle_count: int
    strategy_profile: str
    entry_blocked_currencies: tuple[str, ...]
    ema_days: int
    ema_slope_lookback: int
    entry_channel_days: int
    exit_channel_days: int
    atr_days: int
    atr_stop_multiple: float
    risk_per_trade: float
    max_aggregate_risk: float
    max_gross_leverage: float
    drawdown_half_risk: float
    drawdown_stop: float
    adaptive_fast_ema_days: int
    adaptive_mid_ema_days: int
    adaptive_slow_ema_days: int
    adaptive_slope_lookback: int
    adaptive_adx_days: int
    adaptive_trend_adx: float
    adaptive_range_adx: float
    adaptive_breakout_days: int
    adaptive_bollinger_days: int
    adaptive_range_z: float
    adaptive_score_min: float
    adaptive_score_medium: float
    adaptive_score_high: float
    adaptive_risk_low: float
    adaptive_risk_medium: float
    adaptive_risk_high: float
    adaptive_range_risk_cap: float
    adaptive_max_open_positions: int
    adaptive_max_aggregate_risk: float
    adaptive_max_single_currency_risk: float
    adaptive_max_gross_leverage: float
    adaptive_monthly_loss_limit: float
    adaptive_drawdown_reduce_1: float
    adaptive_drawdown_reduce_2: float
    adaptive_drawdown_stop: float
    adaptive_max_spread_pips: float
    paper_initial_balance: float
    paper_spread_pips: float
    paper_slippage_pips: float
    schedule_hour_utc: int
    schedule_minute_utc: int
    openai_feedback_enabled: bool
    openai_feedback_allow_live: bool
    openai_api_key: str
    openai_model: str
    openai_feedback_event_limit: int
    openai_feedback_min_interval_hours: float
    openai_timeout_seconds: float
    mt5_terminal_path: str
    mt5_timeout_ms: int

    @property
    def oanda_base_url(self) -> str:
        if self.broker_mode == "oanda_live":
            return self.oanda_base_url_live.rstrip("/")
        return self.oanda_base_url_practice.rstrip("/")

    @property
    def live_safety_unlocked(self) -> bool:
        return self.allow_live_trading == "YES_I_ACCEPT_THE_RISK"

    @property
    def adaptive_enabled(self) -> bool:
        return self.strategy_profile == "adaptive_dual_regime_v1"

    @classmethod
    def from_env(cls) -> "Settings":
        mode = os.getenv("BROKER_MODE", "paper").strip().lower()
        if mode not in {"paper", "mt5_paper", "oanda_practice", "oanda_live"}:
            raise ValueError(
                "BROKER_MODE must be paper, mt5_paper, oanda_practice, or oanda_live"
            )

        profile = os.getenv("STRATEGY_PROFILE", "legacy_trend_v1").strip().lower()
        if profile not in {"legacy_trend_v1", "adaptive_dual_regime_v1"}:
            raise ValueError(
                "STRATEGY_PROFILE must be legacy_trend_v1 or adaptive_dual_regime_v1"
            )

        adaptive_mt5_default = (
            "USDJPY,EURUSD,GBPUSD,AUDUSD,NZDUSD,USDCAD,USDCHF,EURJPY,GBPJPY,AUDJPY"
        )
        adaptive_oanda_default = (
            "USD_JPY,EUR_USD,GBP_USD,AUD_USD,NZD_USD,USD_CAD,USD_CHF,EUR_JPY,GBP_JPY,AUD_JPY"
        )
        if mode == "mt5_paper":
            default_instruments = (
                adaptive_mt5_default
                if profile == "adaptive_dual_regime_v1"
                else "USDJPY,EURUSD,GBPUSD,AUDUSD"
            )
            instruments = _tuple_env("MT5_INSTRUMENTS", default_instruments)
        else:
            default_instruments = (
                adaptive_oanda_default
                if profile == "adaptive_dual_regime_v1"
                else "USD_JPY,EUR_USD,GBP_USD,AUD_USD"
            )
            instruments = _tuple_env("INSTRUMENTS", default_instruments)

        default_candle_count = 3200 if profile == "adaptive_dual_regime_v1" else 320

        return cls(
            app_name=os.getenv("APP_NAME", "FX Trend Bot"),
            db_path=os.getenv("DB_PATH", "data/fxbot.sqlite3"),
            host=os.getenv("HOST", "127.0.0.1"),
            port=_int("PORT", 8000),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            broker_mode=mode,
            trading_armed=_bool("TRADING_ARMED", False),
            allow_live_trading=os.getenv("ALLOW_LIVE_TRADING", "NO"),
            oanda_account_id=os.getenv("OANDA_ACCOUNT_ID", "").strip(),
            oanda_api_token=os.getenv("OANDA_API_TOKEN", "").strip(),
            oanda_base_url_practice=os.getenv(
                "OANDA_BASE_URL_PRACTICE", "https://api-fxpractice.oanda.com"
            ),
            oanda_base_url_live=os.getenv(
                "OANDA_BASE_URL_LIVE", "https://api-fxtrade.oanda.com"
            ),
            account_home_currency=os.getenv("ACCOUNT_HOME_CURRENCY", "JPY").upper(),
            instruments=instruments,
            market_data_candle_count=_int(
                "MARKET_DATA_CANDLE_COUNT", default_candle_count
            ),
            strategy_profile=profile,
            entry_blocked_currencies=_tuple_env("ENTRY_BLOCKED_CURRENCIES", ""),
            ema_days=_int("EMA_DAYS", 200),
            ema_slope_lookback=_int("EMA_SLOPE_LOOKBACK", 20),
            entry_channel_days=_int("ENTRY_CHANNEL_DAYS", 55),
            exit_channel_days=_int("EXIT_CHANNEL_DAYS", 20),
            atr_days=_int("ATR_DAYS", 20),
            atr_stop_multiple=_float("ATR_STOP_MULTIPLE", 2.0),
            risk_per_trade=_float("RISK_PER_TRADE", 0.0025),
            max_aggregate_risk=_float("MAX_AGGREGATE_RISK", 0.0075),
            max_gross_leverage=_float("MAX_GROSS_LEVERAGE", 2.0),
            drawdown_half_risk=_float("DRAWDOWN_HALF_RISK", 0.08),
            drawdown_stop=_float("DRAWDOWN_STOP", 0.12),
            adaptive_fast_ema_days=_int("ADAPTIVE_FAST_EMA_DAYS", 20),
            adaptive_mid_ema_days=_int("ADAPTIVE_MID_EMA_DAYS", 50),
            adaptive_slow_ema_days=_int("ADAPTIVE_SLOW_EMA_DAYS", 200),
            adaptive_slope_lookback=_int("ADAPTIVE_SLOPE_LOOKBACK", 10),
            adaptive_adx_days=_int("ADAPTIVE_ADX_DAYS", 14),
            adaptive_trend_adx=_float("ADAPTIVE_TREND_ADX", 23.0),
            adaptive_range_adx=_float("ADAPTIVE_RANGE_ADX", 18.0),
            adaptive_breakout_days=_int("ADAPTIVE_BREAKOUT_DAYS", 20),
            adaptive_bollinger_days=_int("ADAPTIVE_BOLLINGER_DAYS", 20),
            adaptive_range_z=_float("ADAPTIVE_RANGE_Z", 1.8),
            adaptive_score_min=_float("ADAPTIVE_SCORE_MIN", 75.0),
            adaptive_score_medium=_float("ADAPTIVE_SCORE_MEDIUM", 82.0),
            adaptive_score_high=_float("ADAPTIVE_SCORE_HIGH", 90.0),
            adaptive_risk_low=_float("ADAPTIVE_RISK_LOW", 0.0040),
            adaptive_risk_medium=_float("ADAPTIVE_RISK_MEDIUM", 0.0065),
            adaptive_risk_high=_float("ADAPTIVE_RISK_HIGH", 0.0080),
            adaptive_range_risk_cap=_float("ADAPTIVE_RANGE_RISK_CAP", 0.0050),
            adaptive_max_open_positions=_int("ADAPTIVE_MAX_OPEN_POSITIONS", 2),
            adaptive_max_aggregate_risk=_float(
                "ADAPTIVE_MAX_AGGREGATE_RISK", 0.0130
            ),
            adaptive_max_single_currency_risk=_float(
                "ADAPTIVE_MAX_SINGLE_CURRENCY_RISK", 0.0080
            ),
            adaptive_max_gross_leverage=_float(
                "ADAPTIVE_MAX_GROSS_LEVERAGE", 3.0
            ),
            adaptive_monthly_loss_limit=_float(
                "ADAPTIVE_MONTHLY_LOSS_LIMIT", 0.05
            ),
            adaptive_drawdown_reduce_1=_float(
                "ADAPTIVE_DRAWDOWN_REDUCE_1", 0.04
            ),
            adaptive_drawdown_reduce_2=_float(
                "ADAPTIVE_DRAWDOWN_REDUCE_2", 0.07
            ),
            adaptive_drawdown_stop=_float("ADAPTIVE_DRAWDOWN_STOP", 0.10),
            adaptive_max_spread_pips=_float("ADAPTIVE_MAX_SPREAD_PIPS", 3.0),
            paper_initial_balance=_float("PAPER_INITIAL_BALANCE", 1_000_000),
            paper_spread_pips=_float("PAPER_SPREAD_PIPS", 1.0),
            paper_slippage_pips=_float("PAPER_SLIPPAGE_PIPS", 0.2),
            schedule_hour_utc=_int("SCHEDULE_HOUR_UTC", 22),
            schedule_minute_utc=_int("SCHEDULE_MINUTE_UTC", 15),
            openai_feedback_enabled=_bool("OPENAI_FEEDBACK_ENABLED", False),
            openai_feedback_allow_live=_bool(
                "OPENAI_FEEDBACK_ALLOW_LIVE", False
            ),
            openai_api_key=os.getenv("OPENAI_API_KEY", "").strip(),
            openai_model=os.getenv("OPENAI_MODEL", "gpt-5.4-mini").strip(),
            openai_feedback_event_limit=_int(
                "OPENAI_FEEDBACK_EVENT_LIMIT", 100
            ),
            openai_feedback_min_interval_hours=_float(
                "OPENAI_FEEDBACK_MIN_INTERVAL_HOURS", 20.0
            ),
            openai_timeout_seconds=_float("OPENAI_TIMEOUT_SECONDS", 60.0),
            mt5_terminal_path=os.getenv("MT5_TERMINAL_PATH", "").strip(),
            mt5_timeout_ms=_int("MT5_TIMEOUT_MS", 120000),
        )
