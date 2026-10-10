

"""Single-worker run loop. Default execution is paper; live must be explicit."""

import argparse
import dataclasses
try:
    import fcntl
except ImportError:
    fcntl = None
import hashlib
import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from .config import D, Settings


def load_model_for_preflight(settings):
    from .data import verify
    from .neural.controller import FlyController

    dataset = verify()
    controller = FlyController(settings)
    brain = controller.brain
    decoder = controller.decoder
    if (
        brain.n != dataset["neurons"]
        or len(brain.post) != dataset["directed_edges"]
        or not len(decoder.left)
        or not len(decoder.right)
        or not len(decoder.gate)
    ):
        raise RuntimeError("MaleCNS model or decision decoder verification failed")
    return {
        "name": "MaleCNS v1.0",
        "loaded": True,
        "neurons": brain.n,
        "directed_edges": len(brain.post),
        "decoder_outputs": {
            "left": len(decoder.left),
            "right": len(decoder.right),
            "gate": len(decoder.gate),
        },
        "dataset_verified": dataset["arrays_verified"],
        "source": "no-trade preflight process; no live worker started",
    }


def run_readonly_preflight(broker, settings, out):
    halt_reason = broker.l.get("halted")
    stop_file = out / "STOP"
    if halt_reason == "RuntimeError":
        try:
            diagnostic = json.loads((out / "error.json").read_text())
            halt_detail = diagnostic.get("reason") or halt_reason
        except (OSError, json.JSONDecodeError):
            halt_detail = halt_reason
    else:
        halt_detail = halt_reason
    checks = [
        {
            "name": "Risk controls",
            "status": "PASS",
            "detail": (
                f"Validated capital cap {settings.capital} USDC; "
                f"loss stop {settings.loss_stop} USDC"
            ),
        },
        {
            "name": "Live run state",
            "status": "BLOCKED" if halt_reason or stop_file.exists() else "PASS",
            "detail": (
                f"Existing live run halt: {halt_detail}"
                if halt_reason
                else "STOP file is present"
                if stop_file.exists()
                else "No existing halt or STOP file"
            ),
        },
        {
            "name": "Order limits",
            "status": "PASS",
            "detail": (
                f"Max order {settings.order_limit} USDC; "
                f"{settings.daily_orders} attempts/day; "
                f"{settings.interval_seconds:g}s minimum interval"
            ),
        },
    ]
    values = {
        "mode": "live preflight-only",
        "live_trading": "OFF",
        "orders_submitted": 0,
        "order_limits": {
            "max_order_usdc": settings.order_limit,
            "max_attempts_per_day": settings.daily_orders,
            "minimum_interval_seconds": settings.interval_seconds,
            "loss_stop_usdc": settings.loss_stop,
            "fee_reserve": settings.fee_reserve,
            "slippage_limit": settings.slippage,
            "spread_limit": settings.spread_limit,
        },
    }

    try:
        model = load_model_for_preflight(settings)
        checks.append(
            {
                "name": "MaleCNS model load",
                "status": "PASS",
                "detail": (
                    f"{model['name']} loaded; {model['neurons']} neurons, "
                    f"{model['directed_edges']} directed edges; fixed decision "
                    "decoder outputs verified. Preflight does not start a live worker."
                ),
            }
        )
        values["model"] = model
    except Exception as exc:
        checks.append(
            {
                "name": "MaleCNS model load",
                "status": "FAIL",
                "detail": f"Existing MaleCNS model failed to load ({type(exc).__name__}): {exc}",
            }
        )
        values["model"] = {
            "name": "MaleCNS v1.0",
            "loaded": False,
            "source": "no-trade preflight process; no live worker started",
        }

    try:
        from .market import CoinbaseMarket

        quotes = CoinbaseMarket(settings.products).snapshot()
        now = time.time()
        quote_details = []
        for product, quote in quotes.items():
            age = now - quote.timestamp
            if age < 0 or age > settings.max_quote_age:
                raise RuntimeError(f"{product} quote is stale")
            quote_details.append(
                f"{product} bid {quote.bid}, ask {quote.ask}, age {age:.2f}s"
            )
        checks.append(
            {
                "name": "Market-data access",
                "status": "PASS",
                "detail": "; ".join(quote_details),
            }
        )
        values["market_data"] = {
            product: {
                "bid": str(quote.bid),
                "ask": str(quote.ask),
                "quote_age_seconds": round(now - quote.timestamp, 3),
            }
            for product, quote in quotes.items()
        }
    except Exception as exc:
        checks.append(
            {
                "name": "Market-data access",
                "status": "FAIL",
                "detail": (
                    str(exc)
                    if isinstance(exc, RuntimeError)
                    and str(exc).endswith("quote is stale")
                    else f"Public Coinbase market-data request failed ({type(exc).__name__})"
                ),
            }
        )

    try:
        broker.preflight()
    except Exception as exc:
        broker_checks = broker.preflight_checks
        checks.extend(broker_checks)
        if not any(check["status"] == "FAIL" for check in broker_checks):
            checks.append(
                {
                    "name": "Coinbase preflight",
                    "status": "FAIL",
                    "detail": f"Preflight stopped ({type(exc).__name__})",
                }
            )
        values.update(broker.preflight_values)
    else:
        checks.extend(broker.preflight_checks)
        values.update(broker.preflight_values)

    result = "PASS" if all(check["status"] == "PASS" for check in checks) else "FAIL"
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "result": result,
        "checks": checks,
        "values": values,
        "orders_submitted": 0,
    }
    path = out / "preflight.json"
    temp = path.with_suffix(".json.partial")
    temp.write_text(json.dumps(report, indent=2) + "\n")
    temp.replace(path)
    return report


def build_run_settings(products, learning, neural_ms, live):
    args = {
        "products": tuple(products),
        "learning": learning,
        "neural_ms": neural_ms,
        "pulse_ms": min(200, neural_ms),
    }
    if live:
        return Settings(**args)
    paper_cash = D(os.environ.get("PAPER_STARTING_USDC", "100"))
    return Settings(
        **args,
        capital=str(paper_cash),
        order_limit=str(min(D("10"), paper_cash)),
        loss_stop=str(min(D("20"), paper_cash)),
    )


def main():
    p = argparse.ArgumentParser(prog="stonkfly")
    sub = p.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--reuse-doomfly", type=Path)
    sub.add_parser("verify")
    run = sub.add_parser("run")
    mode = run.add_mutually_exclusive_group()
    mode.add_argument("--live", action="store_true")
    mode.add_argument("--paper", action="store_true", help="Explicitly use simulated execution")
    run.add_argument(
        "--preflight-only",
        action="store_true",
        help="Read-only exchange checks; never submit an order",
    )
    run.add_argument(
        "--resume-reviewed",
        action="store_true",
        help="After manual review, clear a transient halt only after successful reconciliation",
    )
    run.add_argument(
        "--fixture",
        action="store_true",
        help="Synthetic offline market input; paper only",
    )
    run.add_argument("--steps", type=int, default=0, help="0 keeps running")
    run.add_argument(
        "--fast",
        action="store_true",
        help="Skip waiting in paper mode; execution cooldown still applies",
    )
    run.add_argument(
        "--frozen",
        action="store_true",
        help="Freeze all memory efficacies for a control run",
    )
    run.add_argument("--out", type=Path)
    run.add_argument(
        "--products",
        nargs="+",
        default=[],
        help="List of specific products to trade (e.g. BTC-USDC)",
    )
    run.add_argument(
        "--auto-discover",
        action="store_true",
        help="Dynamically discover top 50 eligible markets",
    )
    run.add_argument("--neural-ms", type=float, default=500)
    status = sub.add_parser("status")
    status.add_argument("--out", type=Path, default=Path("runs/paper"))
    paper_stop = sub.add_parser("paper-stop")
    paper_stop.add_argument("--out", type=Path, default=Path("runs/paper"))
    a = p.parse_args()
    from dotenv import load_dotenv

    # Never search parent projects for unrelated account credentials.
    load_dotenv(dotenv_path=Path.cwd() / ".env", override=False)
    if a.command in ("prepare", "verify"):
        from .data import prepare, verify

        if a.command == "prepare":
            prepare(a.reuse_doomfly)
        else:
            print(json.dumps(verify()))
        return
    if a.command == "paper-stop":
        import sqlite3

        db_path = a.out / "ledger.sqlite"
        if not db_path.exists():
            raise SystemExit(f"No existing run ledger at {db_path}")
        with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as db:
            row = db.execute(
                "SELECT value FROM meta WHERE key='mode'"
            ).fetchone()
        if row is None or json.loads(row[0]) != "paper":
            raise SystemExit("Refusing to stop a run that is not in paper mode")
        a.out.mkdir(parents=True, exist_ok=True)
        (a.out / "STOP").touch()
        print(f"Paper emergency STOP set for {a.out}; no new simulated trades will be accepted.")
        return
    if a.command == "status":
        import sqlite3

        db = sqlite3.connect(f"file:{a.out / 'ledger.sqlite'}?mode=ro", uri=True)
        meta = {k: json.loads(v) for k, v in db.execute("SELECT key,value FROM meta")}
        print(
            json.dumps(
                {
                    k: meta.get(k)
                    for k in [
                        "mode",
                        "tick",
                        "cash",
                        "positions",
                        "initial_cash",
                        "anchor",
                        "halted",
                    ]
                },
                indent=2,
            )
        )
        return
    if a.live and (a.fixture or a.fast):
        p.error("Live mode forbids fixtures and fast replay")
    if a.preflight_only and not a.live:
        p.error("--preflight-only requires --live")
    if a.steps < 0:
        p.error("steps cannot be negative")

    if getattr(a, "auto_discover", False):
        from .market import discover_markets
        from coinbase.rest import RESTClient
        client = RESTClient(api_key=None, api_secret=None, timeout=10)
        a.products = discover_markets(client, limit=50)
        print(f"Auto-discovered {len(a.products)} eligible markets.", flush=True)
    elif not getattr(a, "products", []):
        if getattr(a, "command", "") == "run":
            a.products = ["BTC-USDC"]

    settings = build_run_settings(
        a.products, not a.frozen, a.neural_ms, live=a.live
    )
    out = a.out or Path("runs/live" if a.live else "runs/paper")
    out.mkdir(parents=True, exist_ok=True)
    lock = (out / "worker.lock").open("a")
    try:
        if 'fcntl' in sys.modules and getattr(sys.modules['fcntl'], 'flock', None):
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        else:
            import msvcrt
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    except (BlockingIOError, OSError):
        raise SystemExit("A worker already owns this run directory")
    worker_marker = out / "worker.json"
    marker_temp = worker_marker.with_suffix(".json.partial")
    if worker_marker.exists():
        worker_marker.unlink()
    from .broker import CoinbaseBroker, PaperBroker
    from .ledger import Ledger

    ledger = Ledger(out / "ledger.sqlite", settings, "live" if a.live else "paper")
    try:
        broker = (
            CoinbaseBroker.from_env(settings, ledger)
            if a.live
            else PaperBroker(settings, ledger)
        )
        if a.preflight_only:
            report = run_readonly_preflight(broker, settings, out)
            print(json.dumps(report), flush=True)
            if report["result"] != "PASS":
                raise SystemExit(1)
            return
        result = broker.preflight()
        print(json.dumps(result), flush=True)
        if a.resume_reviewed:
            if (out / "STOP").exists() or ledger.pending():
                raise RuntimeError(
                    "Remove STOP only after review; unresolved orders cannot resume"
                )
            reason = ledger.get("halted")
            if reason and ("Loss stop" in reason or "fee exceeded" in reason):
                raise RuntimeError("A financial stop cannot be cleared by this flag")
            ledger.put("halted", None)
        from .data import verify

        verified = verify()
        from PIL import Image

        from .actions import StonkflyActions
        from .display import market_frame
        from .market import CoinbaseMarket, FixtureMarket
        from .neural.controller import FlyController
        from .reinforcement import reinforcement
        from .risk import Guard, Veto

        market = (
            FixtureMarket(settings.products)
            if a.fixture
            else CoinbaseMarket(settings.products)
        )
        previous = ledger.get("observation")
        if previous:
            market.history = previous["market_history"]
            if a.fixture:
                market.tick = previous["fixture_tick"]
        controller = FlyController(settings)
        cp = ledger.get("checkpoint")
        if cp:
            path = out / cp["file"]
            if hashlib.sha256(path.read_bytes()).hexdigest() != cp["sha256"]:
                raise RuntimeError("Checkpoint integrity mismatch")
            controller.restore(path)
        marker_temp.write_text(
            json.dumps(
                {
                    "mode": broker.mode,
                    "model": "MaleCNS v1.0",
                    "model_loaded": True,
                    "pid": os.getpid(),
                    "started_at": datetime.now(timezone.utc).isoformat(),
                },
                indent=2,
            )
            + "\n"
        )
        marker_temp.replace(worker_marker)
        provenance = {
            "settings": dataclasses.asdict(settings),
            "dataset": verified,
            "circuit": controller.brain.circuit["report"],
            "vision": controller.brain.visual_report,
            "mode": broker.mode,
            "feed": "fixture" if a.fixture else "coinbase-public",
            "decoder": "DNp20 mean R-L: buy/sell; DNpe017 spike gate; otherwise hold. Engineered fixed mapping.",
            "learning_validated": False,
            "pain_receptors_modeled": False,
            "timing": "Each observation advances configured neural_ms regardless of wall-market time; no claim of real-time fly physiology.",
            "source_sha256": {
                str(path.relative_to(Path(__file__).parent)): hashlib.sha256(
                    path.read_bytes()
                ).hexdigest()
                for path in Path(__file__).parent.rglob("*")
                if path.suffix in (".py", ".cpp")
            },
        }
        signature = hashlib.sha256(
            json.dumps(provenance, sort_keys=True).encode()
        ).hexdigest()
        if ledger.get("provenance_sha256") not in (None, signature):
            raise RuntimeError(
                "Run source/protocol changed; use a separate paper run or explicitly review migration"
            )
        ledger.put("provenance_sha256", signature)
        (out / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
        guard = Guard(settings, ledger, out / "STOP")
        provider = StonkflyActions(guard, broker)
        action = provider.get_actions()[0]
        count = 0
        while not a.steps or count < a.steps:
            started = time.monotonic()
            if (out / "STOP").exists() or ledger.get("halted"):
                break
            broker.reconcile()
            broker.verify_balances()
            quotes = market.snapshot()
            guard.check(quotes, time.time())
            market.record(quotes)
            valuation_quotes = quotes
            equity = ledger.equity(quotes)
            kind, delta = reinforcement(
                equity, ledger.get("anchor"), settings.reward_deadband
            )
            
            pre_scan_cp = out / "brain-prescan.npz"
            controller.save(pre_scan_cp)
            
            best_score = -9999
            best_product = None
            rejected = []
            
            for p in settings.products:
                controller.restore(pre_scan_cp)
                q_temp = quotes[p]
                f_temp = market_frame(p, market.history[p], q_temp.bid, q_temp.ask)
                n_temp = controller.observe(f_temp, kind)
                
                if n_temp["side"] != "HOLD":
                    # Estimate costs
                    spread = (q_temp.ask - q_temp.bid) / q_temp.bid if q_temp.bid else D(0)
                    slippage = D(settings.slippage)
                    fee = D(settings.paper_fee) * 2
                    costs = spread + slippage + fee
                    
                    edge = D(abs(n_temp["difference_hz"])) * D(settings.signal_edge_factor)
                    margin = D(settings.safety_margin)
                    
                    if edge <= costs + margin:
                        rejected.append({"product": p, "side": n_temp["side"], "reason": f"Edge {float(edge):.4f} <= costs {float(costs):.4f} + margin {float(margin):.4f}"})
                        score = -9999
                    else:
                        score = float(edge - costs)
                        
                    if score > best_score:
                        best_score = score
                        best_product = p
            
            if rejected:
                with (out / "rejected.jsonl").open("a") as f_rej:
                    for r in rejected:
                        r["tick"] = ledger.get("tick")
                        f_rej.write(json.dumps(r) + "\n")
            
            if best_product is None:
                best_product = settings.products[ledger.get("tick") % len(settings.products)]
                
            controller.restore(pre_scan_cp)
            product = best_product
            q = quotes[product]
            frame = market_frame(product, market.history[product], q.bid, q.ask)
            neural = controller.observe(frame, kind)
            
            # Checkpoint + accounting anchor are committed before any trade.
            # Two slots keep the last committed snapshot safe during a crash.
            slot = ledger.get("tick") % 2
            checkpoint = out / f"brain-{slot}.npz"
            controller.save(checkpoint)
            checkpoint_info = {
                "file": checkpoint.name,
                "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            }
            observation = {
                "neural": neural,
                "product": product,
                "quote": q.json(),
                "pnl_delta_usdc": str(delta),
                "market_history": market.history,
                "fixture_tick": getattr(market, "tick", None),
            }
            ledger.commit_tick(equity, checkpoint_info, observation)
            order = {"status": "HOLD"}
            if neural["side"] != "HOLD":
                try:
                    # Neural integration can be slow; use a fresh execution book.
                    fresh = market.snapshot()
                    latest = fresh[product]
                    if abs(latest.bid - q.bid) / q.bid > D(settings.slippage):
                        raise Veto("Price moved beyond neural observation tolerance")
                    valuation_quotes = fresh
                    provider.quotes = fresh
                    order = action.invoke({"product": product, "side": neural["side"]})
                except Veto as e:
                    order = {"status": "VETO", "reason": str(e)}
            portfolio_equity = ledger.equity(valuation_quotes)
            signal_reason = (
                f"Neural HOLD: gate spikes {neural['gate_spikes']}, "
                f"right-left {neural['difference_hz']:.2f} Hz."
                if neural["side"] == "HOLD"
                else f"Neural {neural['side']}: gate spikes {neural['gate_spikes']}, "
                f"right-left {neural['difference_hz']:.2f} Hz; accepted only if risk checks pass."
            )
            row = {
                "tick": ledger.get("tick"),
                "wall_time": time.time(),
                "product": product,
                "mode": broker.mode,
                "quote": q.json(),
                "equity_usdc": str(portfolio_equity),
                "pnl_delta_usdc": str(delta),
                "neural": neural,
                "execution": order,
                "signal_reason": signal_reason,
            }
            with (out / "events.jsonl").open("a") as f:
                f.write(json.dumps(row, allow_nan=False) + "\n")
                f.flush()
                os.fsync(f.fileno())
            Image.fromarray(frame).save(out / "latest-input.png")
            (out / "latest.json").write_text(json.dumps(row, indent=2) + "\n")
            print(
                json.dumps(
                    {
                        "tick": row["tick"],
                        "side": neural["side"],
                        "execution": order["status"],
                        "equity": str(equity),
                        "stimulus": kind,
                        "plastic_edges_changed": neural["memory"]["changed_edges"],
                    }
                ),
                flush=True,
            )
            count += 1
            if not a.fast and (not a.steps or count < a.steps):
                until = started + settings.interval_seconds
                while time.monotonic() < until and not (out / "STOP").exists():
                    time.sleep(min(1, until - time.monotonic()))
    except KeyboardInterrupt:
        print("Stopped; run state preserved.", flush=True)
    except Exception as e:
        # Never print SDK exception text: it may contain account/request details.
        if not a.preflight_only and not ledger.get("halted"):
            ledger.halt(type(e).__name__)
        frames = traceback.extract_tb(e.__traceback__)
        origin = frames[-1] if frames else None
        internal = origin and Path(origin.filename).is_relative_to(
            Path(__file__).parent
        )
        diagnostic = {
            "type": type(e).__name__,
            "reason": str(e)
            if internal
            else "External dependency error; review connection and account state.",
            "locations": [
                f"{Path(f.filename).name}:{f.lineno} {f.name}" for f in frames
            ],
        }
        (out / "error.json").write_text(json.dumps(diagnostic, indent=2) + "\n")
        print(
            f"Stopped safely: {type(e).__name__}. Inspect local state and reconcile before restarting.",
            file=sys.stderr,
        )
        raise SystemExit(1) from None
    finally:
        worker_marker.unlink(missing_ok=True)
        marker_temp.unlink(missing_ok=True)
        ledger.close()
        lock.close()


if __name__ == "__main__":
    main()
