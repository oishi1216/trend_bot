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

    @property
    def oanda_base_url(self) -> str:
        if self.broker_mode == "oanda_live":
            return self.oanda_base_url_live.rstrip("/")
        return self.oanda_base_url_practice.rstrip("/")

    @property
    def live_safety_unlocked(self) -> bool:
        return self.allow_live_trading == "YES_I_ACCEPT_THE_RISK"

    @classmethod
    def from_env(cls) -> "Settings":
        instruments = tuple(
            item.strip().upper()
            for item in os.getenv(
                "INSTRUMENTS", "USD_JPY,EUR_USD,GBP_USD,AUD_USD"
            ).split(",")
            if item.strip()
        )
        mode = os.getenv("BROKER_MODE", "paper").strip().lower()
        if mode not in {"paper", "oanda_practice", "oanda_live"}:
            raise ValueError("BROKER_MODE must be paper, oanda_practice, or oanda_live")
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
            paper_initial_balance=_float("PAPER_INITIAL_BALANCE", 1_000_000),
            paper_spread_pips=_float("PAPER_SPREAD_PIPS", 1.0),
            paper_slippage_pips=_float("PAPER_SLIPPAGE_PIPS", 0.2),
            schedule_hour_utc=_int("SCHEDULE_HOUR_UTC", 22),
            schedule_minute_utc=_int("SCHEDULE_MINUTE_UTC", 15),
            openai_feedback_enabled=_bool("OPENAI_FEEDBACK_ENABLED", False),
            openai_feedback_allow_live=_bool("OPENAI_FEEDBACK_ALLOW_LIVE", False),
            openai_api_key=os.getenv("OPENAI_API_KEY", "").strip(),
            openai_model=os.getenv("OPENAI_MODEL", "gpt-5.4-mini").strip(),
            openai_feedback_event_limit=_int("OPENAI_FEEDBACK_EVENT_LIMIT", 100),
            openai_feedback_min_interval_hours=_float(
                "OPENAI_FEEDBACK_MIN_INTERVAL_HOURS", 20.0
            ),
            openai_timeout_seconds=_float("OPENAI_TIMEOUT_SECONDS", 60.0),
        )
