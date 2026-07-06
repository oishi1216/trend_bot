from __future__ import annotations

import logging
from typing import Any

import httpx
import pandas as pd

from .config import Settings
from .instruments import quote_currency
from .models import Fill, Side

logger = logging.getLogger(__name__)


class OandaError(RuntimeError):
    pass


class OandaClient:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.base_url = settings.oanda_base_url
        self.account_id = settings.oanda_account_id
        self._client = httpx.Client(
            base_url=self.base_url,
            headers={
                "Authorization": f"Bearer {settings.oanda_api_token}",
                "Content-Type": "application/json",
                "Accept-Datetime-Format": "RFC3339",
            },
            timeout=30.0,
        )

    def validate_config(self) -> None:
        if not self.account_id or not self.settings.oanda_api_token:
            raise OandaError("OANDA_ACCOUNT_ID and OANDA_API_TOKEN are required")

    def _request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        self.validate_config()
        response = self._client.request(method, path, **kwargs)
        if response.is_error:
            try:
                detail = response.json()
            except Exception:
                detail = response.text
            raise OandaError(f"OANDA {response.status_code}: {detail}")
        return response.json()

    def candles(self, instrument: str, count: int = 320) -> pd.DataFrame:
        data = self._request(
            "GET",
            f"/v3/instruments/{instrument}/candles",
            params={
                "count": count,
                "granularity": "D",
                "price": "M",
                "dailyAlignment": 17,
                "alignmentTimezone": "America/New_York",
                "smooth": "false",
            },
        )
        rows = []
        for candle in data.get("candles", []):
            if not candle.get("complete"):
                continue
            mid = candle["mid"]
            rows.append(
                {
                    "time": candle["time"],
                    "open": float(mid["o"]),
                    "high": float(mid["h"]),
                    "low": float(mid["l"]),
                    "close": float(mid["c"]),
                    "volume": int(candle.get("volume", 0)),
                }
            )
        if not rows:
            raise OandaError(f"No complete candles returned for {instrument}")
        return pd.DataFrame(rows)

    def account_summary(self) -> dict[str, Any]:
        return self._request("GET", f"/v3/accounts/{self.account_id}/summary")["account"]

    def nav(self) -> float:
        return float(self.account_summary()["NAV"])

    def mid_price(self, instrument: str) -> float:
        data = self._request(
            "GET",
            f"/v3/accounts/{self.account_id}/pricing",
            params={"instruments": instrument},
        )
        prices = data.get("prices", [])
        if not prices:
            raise OandaError(f"No price returned for {instrument}")
        price = prices[0]
        bid = float(price["bids"][0]["price"])
        ask = float(price["asks"][0]["price"])
        return (bid + ask) / 2.0

    def conversion_rate(self, from_currency: str, to_currency: str) -> float:
        from_currency = from_currency.upper()
        to_currency = to_currency.upper()
        if from_currency == to_currency:
            return 1.0
        direct = f"{from_currency}_{to_currency}"
        inverse = f"{to_currency}_{from_currency}"
        try:
            return self.mid_price(direct)
        except OandaError:
            return 1.0 / self.mid_price(inverse)

    @staticmethod
    def _format_price(instrument: str, value: float) -> str:
        decimals = 3 if quote_currency(instrument) == "JPY" else 5
        return f"{value:.{decimals}f}"

    def market_order(
        self,
        instrument: str,
        side: Side,
        units: int,
        stop_price: float,
    ) -> Fill:
        signed_units = units if side == "long" else -units
        payload = {
            "order": {
                "units": str(signed_units),
                "instrument": instrument,
                "timeInForce": "FOK",
                "type": "MARKET",
                "positionFill": "DEFAULT",
                "stopLossOnFill": {
                    "timeInForce": "GTC",
                    "price": self._format_price(instrument, stop_price),
                },
                "clientExtensions": {
                    "tag": "fx_trend_bot",
                    "comment": "EMA200/55-day breakout/2ATR stop",
                },
            }
        }
        data = self._request(
            "POST", f"/v3/accounts/{self.account_id}/orders", json=payload
        )
        fill = data.get("orderFillTransaction")
        if not fill:
            raise OandaError(f"Market order was not filled immediately: {data}")
        trade = fill.get("tradeOpened", {})
        return Fill(
            instrument=instrument,
            side=side,
            units=abs(int(float(fill["units"]))),
            price=float(fill["price"]),
            trade_id=trade.get("tradeID"),
            raw=data,
        )

    def close_position(self, instrument: str, side: Side) -> Fill:
        payload = (
            {"longUnits": "ALL", "shortUnits": "NONE"}
            if side == "long"
            else {"longUnits": "NONE", "shortUnits": "ALL"}
        )
        data = self._request(
            "PUT",
            f"/v3/accounts/{self.account_id}/positions/{instrument}/close",
            json=payload,
        )
        fill = data.get("longOrderFillTransaction") or data.get("shortOrderFillTransaction")
        if not fill:
            raise OandaError(f"Position close was not filled: {data}")
        closed_side: Side = "long" if side == "long" else "short"
        return Fill(
            instrument=instrument,
            side=closed_side,
            units=abs(int(float(fill["units"]))),
            price=float(fill["price"]),
            trade_id=None,
            raw=data,
        )

    def open_positions(self) -> dict[str, dict[str, Any]]:
        data = self._request("GET", f"/v3/accounts/{self.account_id}/openPositions")
        return {position["instrument"]: position for position in data.get("positions", [])}
