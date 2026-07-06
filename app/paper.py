from __future__ import annotations

from datetime import datetime, timezone

from .config import Settings
from .instruments import quote_currency
from .models import Fill, Position, Side
from .storage import Storage


def pip_size(instrument: str) -> float:
    return 0.01 if quote_currency(instrument) == "JPY" else 0.0001


class PaperBroker:
    def __init__(self, settings: Settings, storage: Storage, market_data) -> None:
        self.settings = settings
        self.storage = storage
        self.market_data = market_data

    def _adverse_cost(self, instrument: str) -> float:
        pip = pip_size(instrument)
        return pip * (self.settings.paper_spread_pips / 2 + self.settings.paper_slippage_pips)

    def nav(self) -> float:
        nav = self.storage.paper_balance()
        for position in self.storage.get_positions():
            current = self.market_data.mid_price(position.instrument)
            quote = quote_currency(position.instrument)
            quote_to_home = self.market_data.conversion_rate(
                quote, self.settings.account_home_currency
            )
            direction = 1 if position.side == "long" else -1
            nav += (
                (current - position.entry_price)
                * position.units
                * direction
                * quote_to_home
            )
        return nav

    def market_order(
        self,
        instrument: str,
        side: Side,
        units: int,
        stop_price: float,
    ) -> Fill:
        mid = self.market_data.mid_price(instrument)
        cost = self._adverse_cost(instrument)
        fill_price = mid + cost if side == "long" else mid - cost
        return Fill(
            instrument=instrument,
            side=side,
            units=units,
            price=fill_price,
            trade_id=f"paper-{instrument}-{datetime.now(timezone.utc).timestamp()}",
            raw={"mode": "paper", "mid": mid, "adverse_cost": cost},
        )

    def close_position(self, position: Position, reason: str) -> Fill:
        mid = self.market_data.mid_price(position.instrument)
        cost = self._adverse_cost(position.instrument)
        fill_price = mid - cost if position.side == "long" else mid + cost
        self._realize(position, fill_price, reason)
        return Fill(
            instrument=position.instrument,
            side=position.side,
            units=position.units,
            price=fill_price,
            trade_id=None,
            raw={"mode": "paper", "reason": reason, "mid": mid, "adverse_cost": cost},
        )

    def process_stop(self, position: Position, candle: dict) -> Fill | None:
        if position.side == "long" and float(candle["low"]) <= position.stop_price:
            # A gap below the stop fills at the worse opening price.
            raw_fill = min(position.stop_price, float(candle["open"]))
            fill_price = raw_fill - self._adverse_cost(position.instrument)
        elif position.side == "short" and float(candle["high"]) >= position.stop_price:
            raw_fill = max(position.stop_price, float(candle["open"]))
            fill_price = raw_fill + self._adverse_cost(position.instrument)
        else:
            return None
        self._realize(position, fill_price, "hard_stop")
        return Fill(
            instrument=position.instrument,
            side=position.side,
            units=position.units,
            price=fill_price,
            trade_id=None,
            raw={"mode": "paper", "reason": "hard_stop", "candle": candle},
        )

    def _realize(self, position: Position, exit_price: float, reason: str) -> None:
        quote = quote_currency(position.instrument)
        quote_to_home = self.market_data.conversion_rate(
            quote, self.settings.account_home_currency
        )
        direction = 1 if position.side == "long" else -1
        pnl_home = (
            (exit_price - position.entry_price)
            * position.units
            * direction
            * quote_to_home
        )
        self.storage.set_paper_balance(self.storage.paper_balance() + pnl_home)
        self.storage.add_event(
            "INFO",
            "paper_realized_pnl",
            {"pnl_home": pnl_home, "exit_price": exit_price, "reason": reason},
            position.instrument,
        )
