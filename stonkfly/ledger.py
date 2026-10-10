"""Durable money accounting and order intent. All values use Decimal strings."""

import contextlib
import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .config import D


class Ledger:
    def __init__(self, path, settings, mode):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path, isolation_level=None)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY,value TEXT NOT NULL)"
        )
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS orders (id TEXT PRIMARY KEY,status TEXT NOT NULL,created REAL NOT NULL,plan TEXT NOT NULL,exchange_id TEXT,settlement TEXT)"
        )
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS paper_trades ("
            "id TEXT PRIMARY KEY,timestamp TEXT NOT NULL,asset TEXT NOT NULL,"
            "side TEXT NOT NULL,price TEXT NOT NULL,quantity TEXT NOT NULL,"
            "fee TEXT NOT NULL,signal TEXT NOT NULL,reason TEXT NOT NULL,"
            "cash_balance TEXT NOT NULL,realized_pnl TEXT NOT NULL,"
            "cumulative_realized_pnl TEXT NOT NULL,unrealized_pnl TEXT NOT NULL,"
            "equity TEXT NOT NULL)"
        )
        if self.get("settings") is None:
            with self.transaction():
                for k, v in {
                    "settings": settings.signature(),
                    "mode": mode,
                    "cash": settings.capital,
                    "initial_cash": settings.capital,
                    "positions": {},
                    "anchor": settings.capital,
                    "tick": 0,
                    "checkpoint": None,
                    "halted": None,
                    "last_attempt": 0,
                    "cost_basis": {},
                    "realized_pnl": "0",
                }.items():
                    self.put(k, v)
        elif self.get("settings") != settings.signature() or self.get("mode") != mode:
            raise RuntimeError(
                "Run settings/mode mismatch; use a separate paper run directory"
            )
        if self.get("cost_basis") is None or self.get("realized_pnl") is None:
            self._rebuild_accounting()

    def _rebuild_accounting(self):
        basis = {}
        quantities = {}
        realized = D(0)
        rows = self.db.execute(
            "SELECT plan,settlement FROM orders"
            " WHERE status='SETTLED' AND settlement IS NOT NULL ORDER BY created"
        ).fetchall()
        for plan_json, settlement_json in rows:
            plan, settlement = json.loads(plan_json), json.loads(settlement_json)
            product = plan["product"]
            quantity = D(settlement["base"])
            quote = D(settlement["quote"])
            fee = D(settlement["fee"])
            held = D(basis.get(product, "0"))
            owned = quantities.get(product, D(0))
            if plan["side"] == "BUY":
                basis[product] = str(held + quote + fee)
                quantities[product] = owned + quantity
            elif owned > 0:
                sold = min(quantity, owned)
                sold_basis = held * sold / owned
                realized += quote - fee - sold_basis
                basis[product] = str(held - sold_basis)
                quantities[product] = owned - sold
        with self.transaction():
            self.put("cost_basis", basis)
            self.put("realized_pnl", str(realized))

    @contextlib.contextmanager
    def transaction(self):
        self.db.execute("BEGIN IMMEDIATE")
        try:
            yield
            self.db.execute("COMMIT")
        except BaseException:
            self.db.execute("ROLLBACK")
            raise

    def get(self, key):
        row = self.db.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, key, value):
        self.db.execute(
            "INSERT OR REPLACE INTO meta VALUES (?,?)",
            (key, json.dumps(value, allow_nan=False)),
        )

    @property
    def cash(self):
        return D(self.get("cash"))

    @property
    def positions(self):
        return {k: D(v) for k, v in self.get("positions").items()}

    def equity(self, quotes):
        return self.cash + sum(
            (v * quotes[p].bid for p, v in self.positions.items()), D(0)
        )

    def halt(self, reason):
        self.put("halted", reason)

    def reserve(self, plan, now):
        cid = str(uuid.uuid4())
        plan = {**plan, "client_order_id": cid}
        with self.transaction():
            if self.pending():
                raise RuntimeError("Unreconciled order exists")
            self.db.execute(
                "INSERT INTO orders(id,status,created,plan) VALUES (?,?,?,?)",
                (cid, "PREPARED", now, json.dumps(plan)),
            )
            self.put("last_attempt", now)
        return plan

    def mark(self, cid, status, exchange_id=None):
        self.db.execute(
            "UPDATE orders SET status=?,exchange_id=COALESCE(?,exchange_id) WHERE id=?",
            (status, exchange_id, cid),
        )

    def pending(self):
        rows = self.db.execute(
            "SELECT id,status,created,plan,exchange_id FROM orders WHERE status NOT IN ('SETTLED','REJECTED') ORDER BY created"
        ).fetchall()
        return [
            {
                "id": r[0],
                "status": r[1],
                "created": r[2],
                "plan": json.loads(r[3]),
                "exchange_id": r[4],
            }
            for r in rows
        ]

    def attempts_today(self, now):
        return self.db.execute(
            "SELECT COUNT(*) FROM orders WHERE created>=?", (now - now % 86400,)
        ).fetchone()[0]

    def settle(self, cid, base, quote, fee):
        base, quote, fee = map(D, (base, quote, fee))
        if min(base, quote, fee) < 0:
            raise ValueError("Negative settlement")
        if (base == 0 and (quote or fee)) or (base > 0 and quote == 0):
            raise ValueError("Inconsistent fill quantities")
        with self.transaction():
            row = self.db.execute(
                "SELECT status,plan,settlement FROM orders WHERE id=?", (cid,)
            ).fetchone()
            if not row:
                raise RuntimeError("Unknown order")
            payload = {"base": str(base), "quote": str(quote), "fee": str(fee)}
            if row[0] == "SETTLED":
                if json.loads(row[2]) != payload:
                    raise RuntimeError("Settlement changed after finalization")
                return
            if row[0] == "REJECTED":
                raise RuntimeError("Cannot settle a rejected intent")
            p = json.loads(row[1])
            positions = self.positions
            held = positions.get(p["product"], D(0))
            cost_basis = {
                k: D(v) for k, v in (self.get("cost_basis") or {}).items()
            }
            held_basis = cost_basis.get(p["product"], D(0))
            realized = D(0)
            cash = self.cash
            if base > D(p["base_size"]):
                raise RuntimeError("Fill exceeds requested quantity")
            if p["side"] == "BUY":
                if quote > D(p["limit_price"]) * base + D(".00000001"):
                    raise RuntimeError("Buy fill exceeded limit price")
                cash -= quote + fee
                positions[p["product"]] = held + base
                cost_basis[p["product"]] = held_basis + quote + fee
            else:
                if quote + D(".00000001") < D(p["limit_price"]) * base:
                    raise RuntimeError("Sell fill below limit price")
                if base > held or held <= 0 or held_basis < 0:
                    raise RuntimeError("Fill exceeds owned position or cost basis")
                sold_basis = held_basis * base / held
                realized = quote - fee - sold_basis
                cost_basis[p["product"]] = held_basis - sold_basis
                cash += quote - fee
                positions[p["product"]] = held - base
            if cash < 0 or positions[p["product"]] < 0:
                raise RuntimeError("Fill exceeds reserved account funds")
            cumulative_realized = D(self.get("realized_pnl") or "0") + realized
            marks = {
                k: D(v) for k, v in p.get("market_marks", {}).items()
            }
            marks.setdefault(p["product"], D(p["observed_bid"]))
            unrealized = sum(
                (
                    quantity
                    * marks.get(
                        product,
                        cost_basis.get(product, D(0)) / quantity
                        if quantity
                        else D(0),
                    )
                    - cost_basis.get(product, D(0))
                    for product, quantity in positions.items()
                ),
                D(0),
            )
            equity = cash + sum(
                (
                    quantity
                    * marks.get(
                        product,
                        cost_basis.get(product, D(0)) / quantity
                        if quantity
                        else D(0),
                    )
                    for product, quantity in positions.items()
                ),
                D(0),
            )
            self.put("cash", str(cash))
            self.put("positions", {k: str(v) for k, v in positions.items()})
            self.put("cost_basis", {k: str(v) for k, v in cost_basis.items()})
            self.put("realized_pnl", str(cumulative_realized))
            self.db.execute(
                "UPDATE orders SET status='SETTLED',settlement=? WHERE id=?",
                (json.dumps(payload), cid),
            )
            if self.get("mode") == "paper":
                signal = p.get("signal", {"decision": p["side"]})
                reason = signal.get("reason", f"Stonkfly neural signal: {p['side']}")
                self.db.execute(
                    "INSERT INTO paper_trades VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        cid,
                        datetime.now(timezone.utc).isoformat(),
                        p["product"],
                        p["side"],
                        str(quote / base if base else D(0)),
                        str(base),
                        str(fee),
                        json.dumps(signal, allow_nan=False),
                        reason,
                        str(cash),
                        str(realized),
                        str(cumulative_realized),
                        str(unrealized),
                        str(equity),
                    ),
                )
            if fee > D(p["fee_ceiling"]):
                self.halt(
                    "Actual fee exceeded preview ceiling; fill recorded, further orders stopped"
                )

    def commit_tick(self, anchor, checkpoint, observation=None):
        with self.transaction():
            self.put("anchor", str(anchor))
            self.put("checkpoint", checkpoint)
            self.put("tick", self.get("tick") + 1)
            if observation is not None:
                self.put("observation", observation)

    def close(self):
        self.db.close()
