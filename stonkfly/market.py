"""Public Coinbase observations. Synthetic fixtures are explicit test inputs."""

import math
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from decimal import Decimal

from .config import D


@dataclass(frozen=True)
class Quote:
    product: str
    bid: Decimal
    ask: Decimal
    timestamp: float
    base_increment: Decimal
    quote_increment: Decimal
    price_increment: Decimal
    minimum_quote: Decimal
    minimum_base: Decimal

    def __post_init__(self):
        if (
            not self.bid.is_finite()
            or not self.ask.is_finite()
            or not 0 < self.bid <= self.ask
            or not math.isfinite(self.timestamp)
        ):
            raise ValueError("Invalid quote")
        for k in [
            "base_increment",
            "quote_increment",
            "price_increment",
            "minimum_quote",
            "minimum_base",
        ]:
            if not getattr(self, k).is_finite() or getattr(self, k) <= 0:
                raise ValueError("Invalid market increment")

    def json(self):
        return {
            k: str(v) if isinstance(v, Decimal) else v for k, v in asdict(self).items()
        }


def unwrap(value):
    return value.to_dict() if hasattr(value, "to_dict") else value


def utc_timestamp(value):
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("Exchange timestamp requires a timezone")
    return parsed.timestamp()



def discover_markets(client, limit=50):
    products = client.get_public_products()
    products = products.to_dict()["products"] if hasattr(products, "to_dict") else products.get("products", [])
    
    eligible = []
    for m in products:
        if m.get("product_type") != "SPOT":
            continue
        if m.get("quote_currency_id") not in ("USD", "USDC"):
            continue
        if any(m.get(k, False) for k in [
            "is_disabled", "trading_disabled", "cancel_only", "view_only",
            "limit_only", "auction_mode"
        ]):
            continue
        vol = float(m.get("approximate_quote_24h_volume", 0) or 0)
        if vol <= 0:
            continue
        eligible.append((vol, m))
        
    bases = {}
    for vol, m in eligible:
        base = m.get("base_currency_id")
        if base not in bases or vol > bases[base][0]:
            bases[base] = (vol, m)
            
    sorted_markets = sorted(bases.values(), key=lambda x: x[0], reverse=True)
    return [m["product_id"] for vol, m in sorted_markets[:limit]]

class CoinbaseMarket:

    def __init__(self, products, client=None):
        if client is None:
            from coinbase.rest import RESTClient

            # Override SDK environment defaults: public observations never need
            # an account key, even when a live broker exists in this process.
            client = RESTClient(api_key=None, api_secret=None, timeout=10)
        self.client = client
        self.products = products
        self.meta = {}
        self.history = {p: [] for p in products}

    def snapshot(self):
        result = {}

        # First ensure history is initialized for all products
        for product in self.products:
            if not self.history[product]:
                import time
                time.sleep(0.05)
                end = int(time.time() // 60) * 60
                candles = unwrap(
                    self.client.get_public_candles(
                        product, str(end - 120 * 60), str(end), "ONE_MINUTE", limit=120
                    )
                ).get("candles", [])
                past = sorted(
                    (c for c in candles if int(c["start"]) < end),
                    key=lambda c: int(c["start"]),
                )
                if not past:
                    raise RuntimeError(f"No historical candles available for {product}")
                self.history[product] = [float(D(c["close"])) for c in past]
                if any(not math.isfinite(v) or v <= 0 for v in self.history[product]):
                    raise RuntimeError("Invalid historical price")

        bulk_resp = self.client.get_public_products()
        bulk = bulk_resp.to_dict()["products"] if hasattr(bulk_resp, "to_dict") else bulk_resp.get("products", [])
        bulk_map = {m["product_id"]: m for m in bulk}
        import time
        now = time.time()
        
        for product in self.products:
            m = bulk_map.get(product)

                    
            m = bulk_map.get(product)
            if not m:
                m = unwrap(self.client.get_public_product(product))
            self.meta[product] = m
            
            if m.get("product_type") != "SPOT" or m.get("quote_currency_id") not in ("USD", "USDC"):
                raise RuntimeError("Unexpected product")
            if any(m.get(k, False) for k in ["is_disabled", "trading_disabled", "cancel_only", "view_only", "limit_only", "auction_mode"]):
                raise RuntimeError("Product unavailable for immediate spot execution")
                
            price = D(m["price"])
            quote = Quote(
                product,
                price * D("0.9999"),
                price * D("1.0001"),
                time.time(),
                D(m["base_increment"]),
                D(m["quote_increment"]),
                D(m.get("price_increment", m["quote_increment"])),
                D(m["quote_min_size"]),
                D(m["base_min_size"]),
            )
            result[product] = quote
        return result

    def record(self, quotes):
        for p, q in quotes.items():
            self.history[p].append(float((q.bid + q.ask) / 2))
            self.history[p] = self.history[p][-120:]


class FixtureMarket:
    """Deterministic prices for offline verification, never used in live mode."""

    def __init__(self, products):
        self.products = products
        self.tick = 0
        self.history = {p: [] for p in products}

    def snapshot(self):
        base = {"BTC-USDC": 60000, "ETH-USDC": 2500, "SOL-USDC": 100}
        quotes = {}
        for j, p in enumerate(self.products):
            price = D(base[p]) * D(1 + 0.025 * math.sin(self.tick * 0.6 + j))
            quotes[p] = Quote(
                p,
                price * D(".9995"),
                price * D("1.0005"),
                time.time(),
                D(".00000001"),
                D(".01"),
                D(".01"),
                D("1"),
                D(".00000001"),
            )
            if not self.history[p]:
                self.history[p] = [
                    float(D(base[p]) * D(1 + 0.01 * math.sin(i * 0.4 + j)))
                    for i in range(80)
                ]
        self.tick += 1
        return quotes

    record = CoinbaseMarket.record
