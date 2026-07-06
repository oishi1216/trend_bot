from __future__ import annotations

KNOWN_CURRENCIES = {"USD", "JPY", "EUR", "GBP", "AUD", "NZD", "CAD", "CHF"}


def normalize_instrument(instrument: str) -> str:
    return instrument.replace("_", "").replace(".", "").upper()


def base_currency(instrument: str) -> str:
    normalized = normalize_instrument(instrument)
    if len(normalized) < 6:
        raise ValueError(f"Unsupported instrument format: {instrument}")
    base = normalized[:3]
    if base not in KNOWN_CURRENCIES:
        raise ValueError(f"Unsupported base currency: {instrument}")
    return base


def quote_currency(instrument: str) -> str:
    normalized = normalize_instrument(instrument)
    if len(normalized) < 6:
        raise ValueError(f"Unsupported instrument format: {instrument}")
    quote = normalized[3:6]
    if quote not in KNOWN_CURRENCIES:
        raise ValueError(f"Unsupported quote currency: {instrument}")
    return quote


def is_jpy_pair(instrument: str) -> bool:
    return quote_currency(instrument) == "JPY"
