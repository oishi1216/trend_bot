from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pandas as pd

from .config import Settings
from .instruments import quote_currency


class Mt5Error(RuntimeError):
    pass


class Mt5Client:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def _mt5(self):
        try:
            import MetaTrader5 as mt5
        except ImportError as exc:
            raise Mt5Error(
                "MetaTrader5 package is required for mt5_paper. "
                "Run: python -m pip install -r requirements-mt5.txt"
            ) from exc
        return mt5

    def connect(self):
        mt5 = self._mt5()
        path = self.settings.mt5_terminal_path or None
        ok = mt5.initialize(path=path, timeout=self.settings.mt5_timeout_ms)
        if not ok:
            raise Mt5Error(f"MT5 initialize failed: {mt5.last_error()}")
        return mt5

    @staticmethod
    def rates_to_dataframe(rates: Any) -> pd.DataFrame:
        rows = []
        for row in rates[:-1]:
            rows.append(
                {
                    "time": datetime.fromtimestamp(
                        int(row["time"]), tz=timezone.utc
                    ).isoformat(),
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "close": float(row["close"]),
                    "volume": int(row["tick_volume"]),
                }
            )
        return pd.DataFrame(rows)

    def candles(self, instrument: str, count: int | None = None) -> pd.DataFrame:
        requested = count or self.settings.market_data_candle_count
        mt5 = self.connect()
        try:
            if not mt5.symbol_select(instrument, True):
                raise Mt5Error(
                    f"MT5 symbol_select failed for {instrument}: {mt5.last_error()}"
                )

            rates = mt5.copy_rates_from_pos(
                instrument,
                mt5.TIMEFRAME_D1,
                0,
                requested + 1,
            )
            if rates is None:
                raise Mt5Error(
                    f"MT5 copy_rates_from_pos failed for {instrument}: {mt5.last_error()}"
                )

            candles = self.rates_to_dataframe(rates)
            if candles.empty:
                raise Mt5Error(f"No complete MT5 candles for {instrument}")
            return candles
        finally:
            mt5.shutdown()

    def _tick(self, instrument: str):
        mt5 = self.connect()
        try:
            if not mt5.symbol_select(instrument, True):
                raise Mt5Error(
                    f"MT5 symbol_select failed for {instrument}: {mt5.last_error()}"
                )
            tick = mt5.symbol_info_tick(instrument)
            if tick is None:
                raise Mt5Error(
                    f"MT5 tick not available for {instrument}: {mt5.last_error()}"
                )
            bid = float(tick.bid)
            ask = float(tick.ask)
            if bid <= 0 or ask <= 0 or ask < bid:
                raise Mt5Error(
                    f"Invalid MT5 bid/ask for {instrument}: bid={bid}, ask={ask}"
                )
            return bid, ask
        finally:
            mt5.shutdown()

    def mid_price(self, instrument: str) -> float:
        bid, ask = self._tick(instrument)
        return (bid + ask) / 2.0

    def spread_pips(self, instrument: str) -> float:
        bid, ask = self._tick(instrument)
        pip = 0.01 if quote_currency(instrument) == "JPY" else 0.0001
        return (ask - bid) / pip

    def conversion_rate(self, from_currency: str, to_currency: str) -> float:
        from_currency = from_currency.upper()
        to_currency = to_currency.upper()
        if from_currency == to_currency:
            return 1.0

        direct = f"{from_currency}{to_currency}"
        inverse = f"{to_currency}{from_currency}"
        try:
            return self.mid_price(direct)
        except Mt5Error:
            return 1.0 / self.mid_price(inverse)

    def open_positions(self) -> dict[str, dict[str, Any]]:
        return {}
