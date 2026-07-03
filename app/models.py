from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal


Side = Literal["long", "short"]
Action = Literal["enter_long", "enter_short", "exit", "hold", "none"]


@dataclass(frozen=True)
class Position:
    instrument: str
    side: Side
    units: int
    entry_price: float
    stop_price: float
    opened_at: str
    planned_risk_home: float
    broker_trade_id: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class StrategyDecision:
    action: Action
    candle_time: str
    close: float
    atr: float
    ema: float
    reason: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class Fill:
    instrument: str
    side: Side
    units: int
    price: float
    trade_id: str | None
    raw: dict
