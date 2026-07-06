import pytest

from app.instruments import base_currency, is_jpy_pair, quote_currency


def test_currency_helpers_support_oanda_and_mt5_names():
    assert base_currency("USD_JPY") == "USD"
    assert quote_currency("USD_JPY") == "JPY"
    assert base_currency("USDJPY") == "USD"
    assert quote_currency("USDJPY") == "JPY"
    assert quote_currency("EURUSD") == "USD"
    assert is_jpy_pair("USDJPY") is True
    assert is_jpy_pair("EURUSD") is False


def test_currency_helpers_reject_unknown_format():
    with pytest.raises(ValueError):
        quote_currency("BTCUSD")
