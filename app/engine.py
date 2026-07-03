from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from .config import Settings
from .models import Position
from .oanda import OandaClient
from .paper import PaperBroker
from .risk import calculate_units
from .storage import Storage
from .strategy import decide

logger = logging.getLogger(__name__)


class TradingEngine:
    def __init__(self, settings: Settings, storage: Storage) -> None:
        self.settings = settings
        self.storage = storage
        self.oanda = OandaClient(settings)
        self.paper = PaperBroker(settings, storage, self.oanda)

    def _entries_enabled(self) -> tuple[bool, str]:
        if not self.settings.trading_armed:
            return False, "TRADING_ARMED is false"
        if self.settings.broker_mode == "oanda_live" and not self.settings.live_safety_unlocked:
            return False, "Live safety lock is not unlocked"
        return True, "armed"

    def nav(self) -> float:
        if self.settings.broker_mode == "paper":
            return self.paper.nav()
        return self.oanda.nav()

    def _update_drawdown(self, nav: float) -> tuple[float, float]:
        current = float(self.storage.get_kv("nav_high_water", str(nav)) or nav)
        high_water = max(current, nav)
        self.storage.set_kv("nav_high_water", str(high_water))
        drawdown = 0.0 if high_water <= 0 else max(0.0, 1.0 - nav / high_water)
        return high_water, drawdown

    def _portfolio_metrics(self) -> tuple[float, float]:
        planned_risk = sum(p.planned_risk_home for p in self.storage.get_positions())
        gross = 0.0
        for position in self.storage.get_positions():
            quote = position.instrument.split("_")[1]
            quote_to_home = self.oanda.conversion_rate(
                quote, self.settings.account_home_currency
            )
            current_price = self.oanda.mid_price(position.instrument)
            gross += position.units * current_price * quote_to_home
        return planned_risk, gross

    def reconcile(self) -> None:
        if self.settings.broker_mode == "paper":
            return
        remote = self.oanda.open_positions()
        local = {p.instrument: p for p in self.storage.get_positions()}

        for instrument, position in local.items():
            remote_position = remote.get(instrument)
            if not remote_position:
                self.storage.add_event(
                    "WARNING",
                    "reconcile_removed_local_position",
                    {"reason": "No matching open broker position; likely broker stop/close"},
                    instrument,
                )
                self.storage.delete_position(instrument)
                continue
            units = (
                float(remote_position["long"]["units"])
                if position.side == "long"
                else float(remote_position["short"]["units"])
            )
            if units == 0:
                self.storage.add_event(
                    "WARNING",
                    "reconcile_removed_local_position",
                    {"reason": "Matching broker side has zero units"},
                    instrument,
                )
                self.storage.delete_position(instrument)

        for instrument in remote:
            if instrument in self.settings.instruments and instrument not in local:
                self.storage.add_event(
                    "ERROR",
                    "untracked_broker_position",
                    {"reason": "Manual intervention required; new entries for this pair are blocked"},
                    instrument,
                )

    def run_once(self, force: bool = False) -> dict[str, Any]:
        started = datetime.now(timezone.utc).isoformat()
        results: list[dict[str, Any]] = []
        self.reconcile()

        nav = self.nav()
        high_water, drawdown = self._update_drawdown(nav)
        risk_fraction = self.settings.risk_per_trade
        if drawdown >= self.settings.drawdown_half_risk:
            risk_fraction /= 2
        circuit_breaker = drawdown >= self.settings.drawdown_stop
        entries_enabled, armed_reason = self._entries_enabled()

        for instrument in self.settings.instruments:
            try:
                candles = self.oanda.candles(instrument)
                candle = candles.iloc[-1].to_dict()
                candle_time = str(candle["time"])
                if self.storage.is_processed(instrument, candle_time) and not force:
                    results.append({"instrument": instrument, "status": "already_processed", "candle_time": candle_time})
                    continue

                position = self.storage.get_position(instrument)

                # Paper hard stop is simulated using the completed day's OHLC before close-based rules.
                if self.settings.broker_mode == "paper" and position:
                    stop_fill = self.paper.process_stop(position, candle)
                    if stop_fill:
                        self.storage.delete_position(instrument)
                        self.storage.add_event(
                            "INFO", "stop_filled", {"fill": stop_fill.raw, "price": stop_fill.price}, instrument
                        )
                        position = None

                decision = decide(candles, self.settings, position)
                result: dict[str, Any] = {
                    "instrument": instrument,
                    "candle_time": candle_time,
                    "decision": decision.to_dict(),
                }

                if position and decision.action == "exit":
                    if self.settings.broker_mode == "paper":
                        fill = self.paper.close_position(position, decision.reason)
                    else:
                        fill = self.oanda.close_position(instrument, position.side)
                    self.storage.delete_position(instrument)
                    self.storage.add_event(
                        "INFO", "position_closed",
                        {"decision": decision.to_dict(), "fill_price": fill.price, "raw": fill.raw},
                        instrument,
                    )
                    result["status"] = "closed"
                    result["fill_price"] = fill.price

                elif not position and decision.action in {"enter_long", "enter_short"}:
                    remote = self.oanda.open_positions() if self.settings.broker_mode != "paper" else {}
                    if instrument in remote:
                        result["status"] = "blocked_untracked_remote_position"
                    elif not entries_enabled:
                        result["status"] = f"entry_blocked: {armed_reason}"
                    elif circuit_breaker:
                        result["status"] = "entry_blocked: 12% drawdown circuit breaker"
                    else:
                        side = "long" if decision.action == "enter_long" else "short"
                        entry_reference = self.oanda.mid_price(instrument)
                        stop_price = (
                            entry_reference - self.settings.atr_stop_multiple * decision.atr
                            if side == "long"
                            else entry_reference + self.settings.atr_stop_multiple * decision.atr
                        )
                        quote = instrument.split("_")[1]
                        quote_to_home = self.oanda.conversion_rate(
                            quote, self.settings.account_home_currency
                        )
                        planned_risk, gross = self._portfolio_metrics()
                        size = calculate_units(
                            nav_home=nav,
                            entry_price=entry_reference,
                            stop_price=stop_price,
                            quote_to_home=quote_to_home,
                            current_planned_risk_home=planned_risk,
                            current_gross_notional_home=gross,
                            risk_per_trade=risk_fraction,
                            max_aggregate_risk=self.settings.max_aggregate_risk,
                            max_gross_leverage=self.settings.max_gross_leverage,
                        )
                        if size.units < 1:
                            result["status"] = f"entry_blocked: {size.reason}"
                        else:
                            fill = (
                                self.paper.market_order(instrument, side, size.units, stop_price)
                                if self.settings.broker_mode == "paper"
                                else self.oanda.market_order(instrument, side, size.units, stop_price)
                            )
                            actual_risk = abs(fill.price - stop_price) * fill.units * quote_to_home
                            new_position = Position(
                                instrument=instrument,
                                side=side,
                                units=fill.units,
                                entry_price=fill.price,
                                stop_price=stop_price,
                                opened_at=datetime.now(timezone.utc).isoformat(),
                                planned_risk_home=actual_risk,
                                broker_trade_id=fill.trade_id,
                            )
                            self.storage.save_position(new_position)
                            self.storage.add_event(
                                "INFO", "position_opened",
                                {
                                    "decision": decision.to_dict(),
                                    "position": new_position.to_dict(),
                                    "sizing": size.__dict__,
                                    "raw": fill.raw,
                                },
                                instrument,
                            )
                            result["status"] = "opened"
                            result["position"] = new_position.to_dict()
                else:
                    result["status"] = decision.action

                self.storage.mark_processed(instrument, candle_time)
                results.append(result)
            except Exception as exc:
                logger.exception("Failed processing %s", instrument)
                self.storage.add_event(
                    "ERROR", "instrument_run_failed", {"error": str(exc)}, instrument
                )
                results.append({"instrument": instrument, "status": "error", "error": str(exc)})

        summary = {
            "started_at": started,
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "mode": self.settings.broker_mode,
            "armed": self.settings.trading_armed,
            "nav": nav,
            "nav_high_water": high_water,
            "drawdown": drawdown,
            "circuit_breaker": circuit_breaker,
            "results": results,
        }
        self.storage.add_event("INFO", "engine_run", summary)
        return summary

    def status(self) -> dict[str, Any]:
        try:
            nav = self.nav()
            high_water, drawdown = self._update_drawdown(nav)
            nav_error = None
        except Exception as exc:
            nav = None
            high_water = float(self.storage.get_kv("nav_high_water", "0") or 0)
            drawdown = None
            nav_error = str(exc)
        return {
            "mode": self.settings.broker_mode,
            "trading_armed": self.settings.trading_armed,
            "live_safety_unlocked": self.settings.live_safety_unlocked,
            "instruments": self.settings.instruments,
            "nav": nav,
            "nav_error": nav_error,
            "nav_high_water": high_water,
            "drawdown": drawdown,
            "positions": [p.to_dict() for p in self.storage.get_positions()],
            "risk": {
                "per_trade": self.settings.risk_per_trade,
                "aggregate": self.settings.max_aggregate_risk,
                "max_gross_leverage": self.settings.max_gross_leverage,
                "half_risk_drawdown": self.settings.drawdown_half_risk,
                "stop_drawdown": self.settings.drawdown_stop,
            },
        }
