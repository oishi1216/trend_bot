from __future__ import annotations

KNOWN_CURRENCIES = {"USD", "JPY", "EUR", "GBP", "AUD", "NZD", "CAD", "CHF"}


def normalize_instrument(instrument: str) -> str:
    return instrument.replace("_", "").replace(".", "").upper()


def _currency_pair(instrument: str) -> tuple[str, str]:
    normalized = normalize_instrument(instrument)
    if len(normalized) < 6:
        raise ValueError(f"Unsupported instrument format: {instrument}")

    base = normalized[:3]
    quote = normalized[3:6]
    if base not in KNOWN_CURRENCIES:
        raise ValueError(f"Unsupported base currency: {instrument}")
    if quote not in KNOWN_CURRENCIES:
        raise ValueError(f"Unsupported quote currency: {instrument}")
    return base, quote


def base_currency(instrument: str) -> str:
    base, _ = _currency_pair(instrument)
    return base


def quote_currency(instrument: str) -> str:
    _, quote = _currency_pair(instrument)
    return quote


def is_jpy_pair(instrument: str) -> bool:
    return quote_currency(instrument) == "JPY"
