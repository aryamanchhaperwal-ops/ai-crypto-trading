from datetime import datetime, timezone

import numpy as np
import pytest

from stonkfly.display import market_frame
from stonkfly.market import CoinbaseMarket, utc_timestamp


class PublicSDK:
    def get_public_candles(self, product, start, end, granularity, limit):
        return {
            "candles": [
                {"start": str(int(end) + 60), "close": "999999"},
                {"start": str(int(end) - 60), "close": "99"},
                {"start": str(int(end) - 120), "close": "101"},
            ]
        }

    def get_public_products(self):
        return {
            "products": [
                {
                    "product_id": "BTC-USDC",
                    "product_type": "SPOT",
                    "quote_currency_id": "USDC",
                    "base_increment": ".00000001",
                    "quote_increment": ".01",
                    "quote_min_size": "1",
                    "base_min_size": ".00000001",
                    "price": "100.05",
                }
            ]
        }

    def get_public_product(self, product):
        return {
            "product_id": product,
            "product_type": "SPOT",
            "quote_currency_id": "USDC",
            "base_increment": ".00000001",
            "quote_increment": ".01",
            "quote_min_size": "1",
            "base_min_size": ".00000001",
        }

    def get_public_product_book(self, product, limit):
        return {
            "pricebook": {
                "product_id": product,
                "time": datetime.now(timezone.utc).isoformat(),
                "bids": [{"price": "100"}],
                "asks": [{"price": "100.1"}],
            }
        }


def test_completed_history_and_observation_clock():
    m = CoinbaseMarket(("BTC-USDC",), PublicSDK())
    quotes = m.snapshot()
    assert m.history["BTC-USDC"] == [101.0, 99.0]
    m.record(quotes)
    assert m.history["BTC-USDC"] == [101.0, 99.0, 100.05]
    m.snapshot()  # Execution quote refresh must not add another neural observation.
    assert len(m.history["BTC-USDC"]) == 3
    q = quotes["BTC-USDC"]
    frame = market_frame(q.product, m.history[q.product], q.bid, q.ask)
    assert frame.shape == (180, 320, 3) and frame.dtype == np.uint8
    other = market_frame(q.product, list(reversed(m.history[q.product])), q.bid, q.ask)
    assert not np.array_equal(frame, other)


def test_explicit_timestamp_zone():
    with pytest.raises(ValueError):
        utc_timestamp("2026-01-01T00:00:00")
    assert utc_timestamp("2026-01-01T00:00:00Z") == 1767225600


def test_public_client_cannot_inherit_credentials(monkeypatch):
    import coinbase.rest

    observed = {}

    def factory(**kwargs):
        observed.update(kwargs)
        return PublicSDK()

    monkeypatch.setattr(coinbase.rest, "RESTClient", factory)
    CoinbaseMarket(("BTC-USDC",)).snapshot()
    assert observed["api_key"] is None and observed["api_secret"] is None

def test_market_discovery():
    from stonkfly.market import discover_markets
    class MockClient:
        def get_public_products(self):
            return {
                "products": [
                    {"product_id": "BTC-USD", "product_type": "SPOT", "quote_currency_id": "USD", "base_currency_id": "BTC", "is_disabled": False, "approximate_quote_24h_volume": "1000"},
                    {"product_id": "ETH-USDC", "product_type": "SPOT", "quote_currency_id": "USDC", "base_currency_id": "ETH", "is_disabled": False, "approximate_quote_24h_volume": "2000"},
                    {"product_id": "DOGE-USD", "product_type": "SPOT", "quote_currency_id": "USD", "is_disabled": True, "approximate_quote_24h_volume": "500"},
                    {"product_id": "FUT-USD", "product_type": "FUTURE", "quote_currency_id": "USD", "base_currency_id": "BTC", "is_disabled": False, "approximate_quote_24h_volume": "5000"}
                ]
            }
    
    markets = discover_markets(MockClient(), limit=50)
    assert markets == ["ETH-USDC", "BTC-USD"]

def test_market_discovery_limit():
    from stonkfly.market import discover_markets
    class MockClient:
        def get_public_products(self):
            return {
                                "products": [
                    {"product_id": f"COIN{i}-USD", "product_type": "SPOT", "quote_currency_id": "USD", "base_currency_id": f"COIN{i}", "is_disabled": False, "approximate_quote_24h_volume": str(100 - i)}
                    for i in range(100)
                ]
            }
    markets = discover_markets(MockClient(), limit=50)
    assert len(markets) == 50
    assert markets[0] == "COIN0-USD"
