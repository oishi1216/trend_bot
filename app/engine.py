from __future__ import annotations

import json
import logging
from dataclasses import replace
from datetime import datetime, timezone
from typing import Any

from .adaptive_strategy import (
    currency_strength_scores,
    decide as decide_adaptive,
    strength_gap,
)
from .config import Settings
from .instruments import base_currency, quote_currency
from .models import Position, StrategyDecision
from .mt5_client import Mt5Client
from .oanda import OandaClient
from .paper import PaperBroker
from .risk import calculate_units
from .storage import Storage
from .strategy import decide as decide_legacy

logger = logging.getLogger(__name__)


def adaptive_risk_after_drawdown(
    risk_fraction: float, drawdown: float, settings: Settings
) -> float:
    if drawdown >= settings.adaptive_drawdown_stop:
        return 0.0
    if drawdown >= settings.adaptive_drawdown_reduce_2:
        return min(risk_fraction, 0.0025)
    if drawdown >= settings.adaptive_drawdown_reduce_1:
        return risk_fraction * 0.5
    return risk_fraction


def shares_currency(left: str, right: str) -> bool:
    return bool(
        {base_currency(left), quote_currency(left)}
        & {base_currency(right), quote_currency(right)}
    )


def shared_currency_planned_risk(
    instrument: str, positions: list[Position]
) -> float:
    return sum(
        position.planned_risk_home
        for position in positions
        if shares_currency(instrument, position.instrument)
    )


class TradingEngine:
    def __init__(self, settings: Settings, storage: Storage) -> None:
        self.settings = settings
        self.storage = storage
        self.oanda = OandaClient(settings)
        self.market_data = (
            Mt5Client(settings)
            if settings.broker_mode == "mt5_paper"
            else self.oanda
        )
        self.paper = PaperBroker(settings, storage, self.market_data)

    def _is_paper_mode(self) -> bool:
        return self.settings.broker_mode in {"paper", "mt5_paper"}

    def _entries_enabled(self) -> tuple[bool, str]:
        if not self.settings.trading_armed:
            return False, "TRADING_ARMED is false"
        if self.settings.adaptive_enabled and not self._is_paper_mode():
            return (
                False,
                "adaptive_dual_regime_v1 is paper-only until validation passes",
            )
        if (
            self.settings.broker_mode == "oanda_live"
            and not self.settings.live_safety_unlocked
        ):
            return False, "Live safety lock is not unlocked"
        return True, "armed"

    def nav(self) -> float:
        if self._is_paper_mode():
            return self.paper.nav()
        return self.oanda.nav()

    def _update_drawdown(self, nav: float) -> tuple[float, float]:
        current = float(self.storage.get_kv("nav_high_water", str(nav)) or nav)
        high_water = max(current, nav)
        self.storage.set_kv("nav_high_water", str(high_water))
        drawdown = (
            0.0 if high_water <= 0 else max(0.0, 1.0 - nav / high_water)
        )
        return high_water, drawdown

    def _monthly_metrics(self, nav: float) -> tuple[str, float, float]:
        month = datetime.now(timezone.utc).strftime("%Y-%m")
        key = f"month_start_nav:{month}"
        raw = self.storage.get_kv(key)
        if raw is None:
            self.storage.set_kv(key, str(nav))
            start_nav = nav
        else:
            start_nav = float(raw)
        monthly_return = 0.0 if start_nav <= 0 else nav / start_nav - 1.0
        monthly_loss = max(0.0, -monthly_return)
        return month, start_nav, monthly_loss

    def _portfolio_metrics(self) -> tuple[float, float]:
        planned_risk = sum(
            position.planned_risk_home
            for position in self.storage.get_positions()
        )
        gross = 0.0
        for position in self.storage.get_positions():
            quote = quote_currency(position.instrument)
            quote_to_home = self.market_data.conversion_rate(
                quote, self.settings.account_home_currency
            )
            current_price = self.market_data.mid_price(position.instrument)
            gross += position.units * current_price * quote_to_home
        return planned_risk, gross

    def _position_metadata_key(self, instrument: str) -> str:
        return f"position_metadata:{instrument}"

    def _position_metadata(self, instrument: str) -> dict[str, Any]:
        raw = self.storage.get_kv(self._position_metadata_key(instrument))
        if not raw:
            return {}
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}

    def _save_position_metadata(
        self, instrument: str, metadata: dict[str, Any]
    ) -> None:
        self.storage.set_kv(
            self._position_metadata_key(instrument),
            json.dumps(metadata, ensure_ascii=False, default=str),
        )

    def _clear_position_metadata(self, instrument: str) -> None:
        self.storage.set_kv(self._position_metadata_key(instrument), "{}")

    def _spread_pips(self, instrument: str) -> float:
        method = getattr(self.market_data, "spread_pips", None)
        if callable(method):
            return float(method(instrument))
        return float(self.settings.paper_spread_pips)

    def reconcile(self) -> None:
        if self._is_paper_mode():
            return
        remote = self.oanda.open_positions()
        local = {
            position.instrument: position
            for position in self.storage.get_positions()
        }

        for instrument, position in local.items():
            remote_position = remote.get(instrument)
            if not remote_position:
                self.storage.add_event(
                    "WARNING",
                    "reconcile_removed_local_position",
                    {
                        "reason": (
                            "No matching open broker position; "
                            "likely broker stop/close"
                        )
                    },
                    instrument,
                )
                self.storage.delete_position(instrument)
                self._clear_position_metadata(instrument)
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
                self._clear_position_metadata(instrument)

        for instrument in remote:
            if instrument in self.settings.instruments and instrument not in local:
                self.storage.add_event(
                    "ERROR",
                    "untracked_broker_position",
                    {
                        "reason": (
                            "Manual intervention required; new entries "
                            "for this pair are blocked"
                        )
                    },
                    instrument,
                )

    def _decide(
        self,
        instrument: str,
        candles,
        position: Position | None,
        strengths: dict[str, float],
    ) -> StrategyDecision:
        if not self.settings.adaptive_enabled:
            return decide_legacy(candles, self.settings, position)
        return decide_adaptive(
            candles,
            self.settings,
            position,
            pair_strength_gap=strength_gap(instrument, strengths),
            position_metadata=self._position_metadata(instrument),
        )

    def _tighten_paper_stop(
        self, position: Position, decision: StrategyDecision
    ) -> Position:
        new_stop = decision.updated_stop_price
        if not self._is_paper_mode() or new_stop is None:
            return position
        tighter = (
            new_stop > position.stop_price
            if position.side == "long"
            else new_stop < position.stop_price
        )
        if not tighter:
            return position
        quote = quote_currency(position.instrument)
        quote_to_home = self.market_data.conversion_rate(
            quote, self.settings.account_home_currency
        )
        planned_risk = max(
            0.0,
            abs(position.entry_price - new_stop)
            * position.units
            * quote_to_home,
        )
        updated = replace(
            position,
            stop_price=float(new_stop),
            planned_risk_home=planned_risk,
        )
        self.storage.save_position(updated)
        self.storage.add_event(
            "INFO",
            "paper_stop_tightened",
            {"old_stop": position.stop_price, "new_stop": new_stop},
            position.instrument,
        )
        return updated

    def _open_candidate(
        self,
        *,
        instrument: str,
        decision: StrategyDecision,
        nav: float,
        drawdown: float,
        monthly_loss: float,
        entries_enabled: bool,
        armed_reason: str,
    ) -> dict[str, Any]:
        result: dict[str, Any] = {
            "instrument": instrument,
            "candle_time": decision.candle_time,
            "decision": decision.to_dict(),
        }
        if not entries_enabled:
            result["status"] = f"entry_blocked: {armed_reason}"
            return result

        if self.settings.adaptive_enabled:
            if monthly_loss >= self.settings.adaptive_monthly_loss_limit:
                result["status"] = "entry_blocked: monthly loss limit reached"
                return result
            positions = self.storage.get_positions()
            if len(positions) >= self.settings.adaptive_max_open_positions:
                result["status"] = (
                    "entry_blocked: maximum open positions reached"
                )
                return result
            currencies = {
                base_currency(instrument),
                quote_currency(instrument),
            }
            if currencies & set(self.settings.entry_blocked_currencies):
                result["status"] = (
                    "entry_blocked: currency event safety block"
                )
                return result
            spread = self._spread_pips(instrument)
            result["spread_pips"] = spread
            if spread > self.settings.adaptive_max_spread_pips:
                result["status"] = "entry_blocked: spread too wide"
                return result
            risk_fraction = adaptive_risk_after_drawdown(
                float(decision.risk_fraction or 0.0),
                drawdown,
                self.settings,
            )
            if risk_fraction <= 0:
                result["status"] = (
                    "entry_blocked: adaptive drawdown circuit breaker"
                )
                return result
            currency_remaining = (
                nav * self.settings.adaptive_max_single_currency_risk
                - shared_currency_planned_risk(instrument, positions)
            )
            risk_fraction = min(
                risk_fraction,
                max(0.0, currency_remaining) / nav,
            )
            max_aggregate_risk = self.settings.adaptive_max_aggregate_risk
            max_gross_leverage = self.settings.adaptive_max_gross_leverage
        else:
            if drawdown >= self.settings.drawdown_stop:
                result["status"] = "entry_blocked: drawdown circuit breaker"
                return result
            risk_fraction = self.settings.risk_per_trade
            if drawdown >= self.settings.drawdown_half_risk:
                risk_fraction /= 2
            max_aggregate_risk = self.settings.max_aggregate_risk
            max_gross_leverage = self.settings.max_gross_leverage

        if risk_fraction <= 0:
            result["status"] = "entry_blocked: currency risk cap reached"
            return result

        side = "long" if decision.action == "enter_long" else "short"
        entry_reference = self.market_data.mid_price(instrument)
        stop_multiple = (
            decision.stop_atr_multiple
            if decision.stop_atr_multiple is not None
            else self.settings.atr_stop_multiple
        )
        stop_price = (
            entry_reference - stop_multiple * decision.atr
            if side == "long"
            else entry_reference + stop_multiple * decision.atr
        )
        quote = quote_currency(instrument)
        quote_to_home = self.market_data.conversion_rate(
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
            max_aggregate_risk=max_aggregate_risk,
            max_gross_leverage=max_gross_leverage,
        )
        if size.units < 1:
            result["status"] = f"entry_blocked: {size.reason}"
            return result

        fill = (
            self.paper.market_order(
                instrument, side, size.units, stop_price
            )
            if self._is_paper_mode()
            else self.oanda.market_order(
                instrument, side, size.units, stop_price
            )
        )
        actual_risk = (
            abs(fill.price - stop_price) * fill.units * quote_to_home
        )
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
        if self.settings.adaptive_enabled:
            self._save_position_metadata(
                instrument,
                {
                    "profile": self.settings.strategy_profile,
                    "regime": decision.regime,
                    "entry_kind": decision.entry_kind,
                    "initial_stop_price": stop_price,
                    "opened_candle_time": decision.candle_time,
                    "score": decision.score,
                    "risk_fraction": risk_fraction,
                },
            )
        self.storage.add_event(
            "INFO",
            "position_opened",
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
        return result

    def run_once(self, force: bool = False) -> dict[str, Any]:
        started = datetime.now(timezone.utc).isoformat()
        self.reconcile()

        nav = self.nav()
        high_water, drawdown = self._update_drawdown(nav)
        month, month_start_nav, monthly_loss = self._monthly_metrics(nav)
        entries_enabled, armed_reason = self._entries_enabled()

        candles_by_instrument: dict[str, Any] = {}
        fetch_errors: dict[str, str] = {}
        for instrument in self.settings.instruments:
            try:
                candles_by_instrument[instrument] = self.market_data.candles(
                    instrument,
                    count=self.settings.market_data_candle_count,
                )
            except Exception as exc:
                logger.exception("Failed fetching %s", instrument)
                fetch_errors[instrument] = str(exc)
                self.storage.add_event(
                    "ERROR",
                    "instrument_run_failed",
                    {"error": str(exc)},
                    instrument,
                )

        strengths = (
            currency_strength_scores(
                candles_by_instrument, self.settings
            )
            if self.settings.adaptive_enabled and candles_by_instrument
            else {}
        )
        results_by_instrument: dict[str, dict[str, Any]] = {}
        candidates: list[tuple[str, StrategyDecision]] = []

        for instrument in self.settings.instruments:
            if instrument in fetch_errors:
                results_by_instrument[instrument] = {
                    "instrument": instrument,
                    "status": "error",
                    "error": fetch_errors[instrument],
                }
                continue

            candles = candles_by_instrument[instrument]
            candle = candles.iloc[-1].to_dict()
            candle_time = str(candle["time"])
            if self.storage.is_processed(instrument, candle_time) and not force:
                results_by_instrument[instrument] = {
                    "instrument": instrument,
                    "status": "already_processed",
                    "candle_time": candle_time,
                }
                continue

            try:
                position = self.storage.get_position(instrument)
                stopped_today = False

                if self._is_paper_mode() and position:
                    stop_fill = self.paper.process_stop(position, candle)
                    if stop_fill:
                        self.storage.delete_position(instrument)
                        self._clear_position_metadata(instrument)
                        self.storage.add_event(
                            "INFO",
                            "stop_filled",
                            {
                                "fill": stop_fill.raw,
                                "price": stop_fill.price,
                            },
                            instrument,
                        )
                        position = None
                        stopped_today = True

                decision = self._decide(
                    instrument, candles, position, strengths
                )
                result: dict[str, Any] = {
                    "instrument": instrument,
                    "candle_time": candle_time,
                    "decision": decision.to_dict(),
                }

                if position:
                    position = self._tighten_paper_stop(
                        position, decision
                    )

                if position and decision.action == "exit":
                    fill = (
                        self.paper.close_position(
                            position, decision.reason
                        )
                        if self._is_paper_mode()
                        else self.oanda.close_position(
                            instrument, position.side
                        )
                    )
                    self.storage.delete_position(instrument)
                    self._clear_position_metadata(instrument)
                    self.storage.add_event(
                        "INFO",
                        "position_closed",
                        {
                            "decision": decision.to_dict(),
                            "fill_price": fill.price,
                            "raw": fill.raw,
                        },
                        instrument,
                    )
                    result["status"] = "closed"
                    result["fill_price"] = fill.price
                elif (
                    not position
                    and not stopped_today
                    and decision.action
                    in {"enter_long", "enter_short"}
                ):
                    remote = (
                        {}
                        if self._is_paper_mode()
                        else self.oanda.open_positions()
                    )
                    if instrument in remote:
                        result["status"] = (
                            "blocked_untracked_remote_position"
                        )
                    else:
                        result["status"] = "entry_candidate"
                        candidates.append((instrument, decision))
                else:
                    result["status"] = (
                        "stopped_today"
                        if stopped_today
                        else decision.action
                    )

                self.storage.mark_processed(instrument, candle_time)
                results_by_instrument[instrument] = result
            except Exception as exc:
                logger.exception("Failed processing %s", instrument)
                self.storage.add_event(
                    "ERROR",
                    "instrument_run_failed",
                    {"error": str(exc)},
                    instrument,
                )
                results_by_instrument[instrument] = {
                    "instrument": instrument,
                    "status": "error",
                    "error": str(exc),
                }

        selected = candidates
        if self.settings.adaptive_enabled and candidates:
            selected = [
                max(candidates, key=lambda item: item[1].score)
            ]
            selected_instrument = selected[0][0]
            for instrument, _decision in candidates:
                if instrument != selected_instrument:
                    results_by_instrument[instrument]["status"] = (
                        "entry_skipped: lower_ranked_candidate"
                    )

        for instrument, decision in selected:
            try:
                results_by_instrument[instrument] = self._open_candidate(
                    instrument=instrument,
                    decision=decision,
                    nav=nav,
                    drawdown=drawdown,
                    monthly_loss=monthly_loss,
                    entries_enabled=entries_enabled,
                    armed_reason=armed_reason,
                )
            except Exception as exc:
                logger.exception("Failed opening %s", instrument)
                self.storage.add_event(
                    "ERROR",
                    "entry_open_failed",
                    {"error": str(exc)},
                    instrument,
                )
                results_by_instrument[instrument]["status"] = "error"
                results_by_instrument[instrument]["error"] = str(exc)

        results = [
            results_by_instrument[instrument]
            for instrument in self.settings.instruments
        ]
        summary = {
            "started_at": started,
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "mode": self.settings.broker_mode,
            "strategy_profile": self.settings.strategy_profile,
            "armed": self.settings.trading_armed,
            "nav": nav,
            "nav_high_water": high_water,
            "drawdown": drawdown,
            "month": month,
            "month_start_nav": month_start_nav,
            "monthly_loss": monthly_loss,
            "circuit_breaker": (
                drawdown >= self.settings.adaptive_drawdown_stop
                if self.settings.adaptive_enabled
                else drawdown >= self.settings.drawdown_stop
            ),
            "currency_strength": strengths,
            "results": results,
        }
        self.storage.add_event("INFO", "engine_run", summary)
        return summary

    def status(self) -> dict[str, Any]:
        try:
            nav = self.nav()
            high_water, drawdown = self._update_drawdown(nav)
            month, month_start_nav, monthly_loss = (
                self._monthly_metrics(nav)
            )
            nav_error = None
        except Exception as exc:
            nav = None
            high_water = float(
                self.storage.get_kv("nav_high_water", "0") or 0
            )
            drawdown = None
            month = datetime.now(timezone.utc).strftime("%Y-%m")
            month_start_nav = None
            monthly_loss = None
            nav_error = str(exc)

        return {
            "mode": self.settings.broker_mode,
            "strategy_profile": self.settings.strategy_profile,
            "trading_armed": self.settings.trading_armed,
            "live_safety_unlocked": self.settings.live_safety_unlocked,
            "instruments": self.settings.instruments,
            "market_data_candle_count": (
                self.settings.market_data_candle_count
            ),
            "nav": nav,
            "nav_error": nav_error,
            "nav_high_water": high_water,
            "drawdown": drawdown,
            "month": month,
            "month_start_nav": month_start_nav,
            "monthly_loss": monthly_loss,
            "positions": [
                position.to_dict()
                for position in self.storage.get_positions()
            ],
            "risk": {
                "per_trade": (
                    {
                        "low": self.settings.adaptive_risk_low,
                        "medium": self.settings.adaptive_risk_medium,
                        "high": self.settings.adaptive_risk_high,
                    }
                    if self.settings.adaptive_enabled
                    else self.settings.risk_per_trade
                ),
                "aggregate": (
                    self.settings.adaptive_max_aggregate_risk
                    if self.settings.adaptive_enabled
                    else self.settings.max_aggregate_risk
                ),
                "max_gross_leverage": (
                    self.settings.adaptive_max_gross_leverage
                    if self.settings.adaptive_enabled
                    else self.settings.max_gross_leverage
                ),
                "monthly_loss_limit": (
                    self.settings.adaptive_monthly_loss_limit
                    if self.settings.adaptive_enabled
                    else None
                ),
                "stop_drawdown": (
                    self.settings.adaptive_drawdown_stop
                    if self.settings.adaptive_enabled
                    else self.settings.drawdown_stop
                ),
            },
        }
