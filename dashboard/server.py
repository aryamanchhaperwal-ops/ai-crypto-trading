"""Stonkfly neural dashboard with isolated paper controls.

Reads the checkpoints, events and ledger that `python -m stonkfly run` writes into
a run directory and serves them to index.html. Its local paper endpoints start
the existing paper worker or latch its STOP file; it never invokes live execution.

Usage (from the repository root):
    python dashboard/server.py [RUN_DIR]        # default: runs/paper

Environment variables: see README.md ("Dashboard configuration").
"""

import json
import re
import os
import ipaddress
import subprocess
import sqlite3
from decimal import Decimal
from datetime import datetime
import sys
import time
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlsplit

import numpy as np

HERE = Path(__file__).resolve().parent
# Repository root (the directory that contains the `stonkfly` package and `data/`).
ROOT = Path(os.environ.get("STONKFLY_ROOT", HERE.parent)).resolve()
RUN = Path(
    sys.argv[1] if len(sys.argv) > 1 else os.environ.get("STONKFLY_RUN", ROOT / "runs/paper")
).resolve()
PORT = int(os.environ.get("PORT", "8765"))
# Localhost by default. Set HOST=0.0.0.0 to open it to your LAN (read-only, no auth).
HOST = os.environ.get("HOST", "127.0.0.1")
PREFLIGHT_RUN = ROOT / "runs" / "live"
PREFLIGHT_FILE = PREFLIGHT_RUN / "preflight.json"
PREFLIGHT_LOCK = threading.Lock()
CACHE = Path(os.environ.get("STONKFLY_VIZ_CACHE", HERE / "cache"))
ANIM_DIR = Path(os.environ.get("STONKFLY_ANIM", HERE / "anim"))
# Optional links to other dashboards, e.g. "BTC fly:8765,PEPE fly:8766".
FLIES = [
    {"name": name.strip(), "port": int(port)}
    for name, _, port in (
        item.rpartition(":") for item in os.environ.get("STONKFLY_FLIES", "").split(",") if item.strip()
    )
]
os.environ.setdefault("STONKFLY_DATA", str(ROOT / "data"))
DATA = Path(os.environ["STONKFLY_DATA"])
sys.path.insert(0, str(ROOT))


def build_static():
    """Neuron positions, superclasses and identified groups; cached on disk."""
    if (CACHE / "static.npz").exists() and (CACHE / "meta.json").exists():
        z = np.load(CACHE / "static.npz")
        return z["pos"], z["sc"], z["type"], json.loads((CACHE / "meta.json").read_text())
    import pyarrow.feather as feather

    from stonkfly.neural.visual import projection

    g = np.load(DATA / "graph.npz")
    ids, ptr, post, retina = g["ids"], g["ptr"], g["post"], g["retina"]
    n = len(ids)
    cols = [
        "bodyId", "type", "superclass", "somaSide", "rootSide",
        "somaLocation", "tosomaLocation", "assignedOlHex1", "assignedOlHex2",
    ]
    a = (
        feather.read_table(DATA / "annotations.feather", columns=cols)
        .to_pandas()
        .set_index("bodyId")
        .loc[ids]
    )
    pos = np.full((n, 3), np.nan, dtype=np.float64)
    for col in ["somaLocation", "tosomaLocation"]:
        values = a[col].to_numpy()
        for i in np.flatnonzero(np.isnan(pos[:, 0])):
            v = values[i]
            if isinstance(v, (list, tuple, np.ndarray)) and len(v) == 3:
                pos[i] = v
    measured = ~np.isnan(pos[:, 0])
    # Cells without a soma in the volume (e.g. photoreceptors) are placed at
    # the mean position of their synaptic partners. Display only.
    src = np.repeat(np.arange(n, dtype=np.int32), np.diff(ptr))
    for _ in range(4):
        miss = np.isnan(pos[:, 0])
        if not miss.any():
            break
        known = ~miss
        acc = np.zeros((n, 3))
        out_e = miss[src] & known[post]
        in_e = miss[post] & known[src]
        cnt = np.bincount(src[out_e], minlength=n) + np.bincount(post[in_e], minlength=n)
        for k in range(3):
            acc[:, k] = np.bincount(
                src[out_e], weights=pos[post[out_e], k], minlength=n
            ) + np.bincount(post[in_e], weights=pos[src[in_e], k], minlength=n)
        fill = miss & (cnt > 0)
        pos[fill] = acc[fill] / cnt[fill, None]
    del src

    superclass = a.superclass.fillna("unassigned").astype(str).to_numpy()
    names, sc = np.unique(superclass, return_inverse=True)
    t = a.type.fillna("").astype(str).to_numpy(dtype=str)
    side = a.somaSide.fillna("").astype(str).to_numpy(dtype=str)
    type_names, type_code = np.unique(t, return_inverse=True)

    def idx(mask):
        return np.flatnonzero(mask).tolist()

    brain = SimpleNamespace(ptr=ptr, post=post, weight=g["weight"], retina=retina)
    r8, r8_uv, _ = projection(brain, a)
    groups = {
        "dnp20_L": idx((t == "DNp20") & (side == "L")),
        "dnp20_R": idx((t == "DNp20") & (side == "R")),
        "gate": idx(t == "DNpe017"),
        "reward": idx(t == "PAM11"),
        "aversive": idx(t == "PPL101"),
        "mbon": idx(np.isin(t, ["MBON07", "MBON11"])),
        "kc": idx(np.char.startswith(t, "KC")),
        "retina": retina.tolist(),
        "r8": r8.tolist(),
    }
    meta = {
        "n": n,
        "superclasses": names.tolist(),
        "superclass_sizes": np.bincount(sc, minlength=len(names)).tolist(),
        "type_names": type_names.tolist(),
        "groups": groups,
        "retina_uv": np.round(g["uv"], 4).tolist(),
        "r8_uv": np.round(r8_uv, 4).tolist(),
        "r8_channel": np.where(a.type.iloc[r8].eq("R8p"), 2, 1).tolist(),
        "measured_positions": int(measured.sum()),
        "inferred_positions": int((~measured & ~np.isnan(pos[:, 0])).sum()),
        "unplaced": int(np.isnan(pos[:, 0]).sum()),
        "bounds_min": np.nanmin(pos, axis=0).tolist(),
        "bounds_max": np.nanmax(pos, axis=0).tolist(),
    }
    pos = pos.astype(np.float32)
    sc = sc.astype(np.uint8)
    type_code = type_code.astype(np.uint16)
    CACHE.mkdir(exist_ok=True)
    np.savez(CACHE / "static.npz", pos=pos, sc=sc, type=type_code)
    (CACHE / "meta.json").write_text(json.dumps(meta))
    return pos, sc, type_code, meta


def hop_distance():
    """Synaptic hops from the photoreceptors (R1-R6 and R8). 255 = unreachable."""
    path = CACHE / "hop.npy"
    if path.exists():
        return np.load(path)
    g = np.load(DATA / "graph.npz")
    ptr, post = g["ptr"], g["post"]
    n = len(ptr) - 1
    src = np.repeat(np.arange(n, dtype=np.int32), np.diff(ptr))
    hop = np.full(n, 255, dtype=np.uint8)
    seeds = np.r_[g["retina"], np.asarray(META["groups"]["r8"], dtype=np.int32)]
    hop[seeds] = 0
    frontier = hop == 0
    level = 0
    while frontier.any() and level < 254:
        level += 1
        reached = np.zeros(n, dtype=bool)
        reached[post[frontier[src]]] = True
        reached &= hop == 255
        hop[reached] = level
        frontier = reached
    np.save(path, hop)
    return hop


POS, SC, TYPE, META = build_static()
HOP = hop_distance()
_checkpoint = {"key": None}


def checkpoint():
    files = sorted(RUN.glob("brain-*.npz"), key=lambda p: p.stat().st_mtime_ns)
    if not files:
        return None
    key = (str(files[-1]), files[-1].stat().st_mtime_ns)
    if _checkpoint["key"] != key:
        with np.load(files[-1], allow_pickle=False) as z:
            meta = json.loads(str(z["metadata"]))
            efficacy = 1 + z["memory_w"]
            hist, edges = np.histogram(efficacy, bins=40, range=(0.1, 2.0))
            _checkpoint.update(
                key=key,
                counts=np.minimum(z["counts"], 255).astype(np.uint8).tobytes(),
                luminance=np.round(z["luminance"], 3).tolist(),
                r8_light=np.round(z["r8_light"], 3).tolist(),
                efficacy_hist=hist.tolist(),
                efficacy_edges=np.round(edges, 3).tolist(),
                brain_ms=meta["cursor"] * 0.1,
                mtime=files[-1].stat().st_mtime,
            )
    return _checkpoint


def ledger():
    try:
        db = sqlite3.connect(f"file:{RUN / 'ledger.sqlite'}?mode=ro", uri=True)
        meta = {k: json.loads(v) for k, v in db.execute("SELECT key,value FROM meta")}
        db.close()
        return {
            k: meta.get(k)
            for k in [
                "mode",
                "tick",
                "cash",
                "positions",
                "initial_cash",
                "halted",
                "cost_basis",
                "realized_pnl",
            ]
        }
    except sqlite3.Error:
        return {}


def paper_portfolio(events, ledger_state):
    mode = ledger_state.get("mode", "UNKNOWN")
    runtime = runtime_readiness(RUN)
    active = worker_active(RUN)
    if mode != "paper":
        status = (
            "UNAVAILABLE"
            if mode == "live"
            else "STARTING"
            if active is True
            else "UNKNOWN"
        )
        return {
            "mode": mode,
            "live_trading": "UNAVAILABLE" if mode == "live" else "UNKNOWN",
            "status": status,
            "system_detail": runtime.get("system_detail"),
            "starting_balance": None,
            "cash": None,
            "positions": [],
            "equity": None,
            "realized_pnl": None,
            "unrealized_pnl": None,
            "net_pnl": None,
            "latest_decision": None,
            "open_position_count": None,
            "closed_trade_count": None,
            "trades": [],
            "trade_log_error": None,
            "stop_requested": (RUN / "STOP").exists(),
        }
    elif ledger_state.get("halted") or (RUN / "STOP").exists():
        status = "HALTED"
    elif active is True:
        status = "RUNNING"
    elif active is False:
        status = "STOPPED"
    else:
        status = "UNKNOWN"

    paper_trades = []
    log_error = None
    closed_trade_count = 0
    realized = Decimal(str(ledger_state.get("realized_pnl") or "0"))
    cost_basis = ledger_state.get("cost_basis") or {}
    try:
        with sqlite3.connect(f"file:{RUN / 'ledger.sqlite'}?mode=ro", uri=True) as db:
            rows = db.execute(
                "SELECT timestamp,asset,side,price,quantity,fee,signal,reason,"
                "cash_balance,realized_pnl,cumulative_realized_pnl,"
                "unrealized_pnl,equity FROM paper_trades "
                "ORDER BY timestamp DESC LIMIT 100"
            ).fetchall()
            closed_trade_count = db.execute(
                "SELECT COUNT(*) FROM paper_trades WHERE side='SELL'"
            ).fetchone()[0]
        paper_trades = [
            {
                "timestamp": row[0],
                "asset": row[1],
                "side": row[2],
                "price": row[3],
                "quantity": row[4],
                "fee": row[5],
                "signal": json.loads(row[6]),
                "reason": row[7],
                "cash_balance": row[8],
                "realized_pnl": row[9],
                "cumulative_realized_pnl": row[10],
                "unrealized_pnl": row[11],
                "equity": row[12],
            }
            for row in rows
        ]
    except sqlite3.Error as exc:
        log_error = f"Paper trade log unavailable ({type(exc).__name__})"

    positions = ledger_state.get("positions") or {}
    open_positions = [
        {"asset": asset, "quantity": str(quantity)}
        for asset, quantity in sorted(positions.items())
        if Decimal(str(quantity)) > 0
    ]
    marks = {}
    for event in events:
        marks[event["product"]] = Decimal(str(event["bid"]))
    cash = Decimal(str(ledger_state.get("cash") or "0"))
    basis = {k: Decimal(str(v)) for k, v in cost_basis.items()}
    equity = cash + sum(
        (
            Decimal(str(quantity))
            * marks.get(
                asset,
                basis.get(asset, Decimal(0)) / Decimal(str(quantity))
                if Decimal(str(quantity))
                else Decimal(0),
            )
            for asset, quantity in positions.items()
        ),
        Decimal(0),
    )
    if events:
        equity = Decimal(str(events[-1]["equity"]))
    initial = Decimal(str(ledger_state.get("initial_cash") or "0"))
    net_pnl = equity - initial
    latest = events[-1] if events else None
    decision = None
    if latest:
        decision = {
            "timestamp": latest["time"],
            "asset": latest["product"],
            "side": latest["side"],
            "difference_hz": latest["diff_hz"],
            "risk_status": (
                "PASS"
                if latest["execution"] == "FILLED"
                else "BLOCKED"
                if latest["execution"] == "VETO"
                else "NOT APPLICABLE"
            ),
            "execution": latest["execution"],
            "reason": latest["reason"],
        }
    return {
        "mode": mode,
        "live_trading": "OFF" if mode == "paper" else "UNAVAILABLE" if mode == "live" else "UNKNOWN",
        "status": status,
        "system_detail": runtime.get("system_detail"),
        "starting_balance": ledger_state.get("initial_cash"),
        "cash": ledger_state.get("cash"),
        "positions": open_positions,
        "equity": str(equity),
        "realized_pnl": str(realized),
        "unrealized_pnl": str(net_pnl - realized),
        "net_pnl": str(net_pnl),
        "latest_decision": decision,
        "open_position_count": len(open_positions),
        "closed_trade_count": closed_trade_count,
        "trades": paper_trades,
        "trade_log_error": log_error,
        "stop_requested": (RUN / "STOP").exists(),
    }


def worker_active(run):
    path = run / "worker.lock"
    if not path.exists():
        return False
    try:
        with path.open("r+b") as lock:
            if os.name == "nt":
                import msvcrt

                lock.seek(0)
                try:
                    msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
                except OSError as exc:
                    if exc.errno in (13, 36, 10013, 10035):
                        return True
                    return None
                lock.seek(0)
                msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                try:
                    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    return True
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        return False
    except OSError:
        return None


def runtime_readiness(run):
    db_path = run / "ledger.sqlite"
    if not db_path.exists():
        return {
            "system_status": "NOT STARTED",
            "mode": "UNKNOWN",
            "model": {
                "name": "Six-Agent Team",
                "loaded": False,
                "status": "UNKNOWN",
                "detail": "No run ledger is available.",
            },
        }
    try:
        db = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        meta = {k: json.loads(v) for k, v in db.execute("SELECT key,value FROM meta")}
        db.close()
    except (sqlite3.Error, ValueError, json.JSONDecodeError):
        return {
            "system_status": "UNKNOWN",
            "mode": "UNKNOWN",
            "model": {
                "name": "Six-Agent Team",
                "loaded": None,
                "status": "UNKNOWN",
                "detail": "Run ledger could not be read.",
            },
        }
    active = worker_active(run)
    marker = None
    if active is True:
        try:
            marker = json.loads((run / "worker.json").read_text())
        except (OSError, json.JSONDecodeError):
            marker = None
    stopped = bool(meta.get("halted")) or (run / "STOP").exists()
    if stopped:
        status = "HALTED"
        halt_detail = meta.get("halted") or "STOP file is present"
        if halt_detail == "RuntimeError":
            try:
                diagnostic = json.loads((run / "error.json").read_text())
                halt_detail = diagnostic.get("reason") or halt_detail
            except (OSError, json.JSONDecodeError):
                pass
        detail = str(halt_detail)
    elif active is True and marker:
        status = "RUNNING"
        detail = "Live worker holds the run lock and reports a loaded model."
    elif active is True:
        status = "PREFLIGHTING"
        detail = "A process owns the run lock; no active engine marker is available."
    elif active is None:
        status = "UNKNOWN"
        detail = "Could not determine whether the run lock is held."
    else:
        status = "STOPPED"
        detail = "No worker currently holds the run lock."
    checkpoints = sorted(
        run.glob("brain-*.npz"), key=lambda p: p.stat().st_mtime_ns
    )
    if active is True and marker and marker.get("model_loaded") is True:
        model = {
            "name": marker.get("model", "Six-Agent Team"),
            "loaded": True,
            "status": "LOADED",
            "detail": "The active engine reports its model loaded.",
        }
    elif active is True:
        model = {
            "name": "Six-Agent Team",
            "loaded": False,
            "status": "NOT LOADED",
            "detail": "An existing preflight or engine startup currently owns this run.",
        }
    elif checkpoints:
        model = {
            "name": "Six-Agent Team",
            "loaded": False,
            "status": "CHECKPOINT AVAILABLE",
            "detail": "A checkpoint exists, but no active worker has it loaded.",
        }
    else:
        model = {
            "name": "Six-Agent Team",
            "loaded": False,
            "status": "NOT LOADED",
            "detail": "No model checkpoint is available in this run.",
        }
    return {
        "system_status": status,
        "system_detail": detail,
        "mode": meta.get("mode", "UNKNOWN"),
        "model": model,
    }


def readiness_state():
    try:
        report = json.loads(PREFLIGHT_FILE.read_text())
    except (OSError, json.JSONDecodeError):
        report = None
    live_runtime = runtime_readiness(PREFLIGHT_RUN)
    model = (report or {}).get("values", {}).get("model", {})
    if model.get("loaded") is True and live_runtime["model"].get("loaded") is not True:
        live_runtime["model"] = {
            "name": model.get("name", "Six-Agent Team"),
            "loaded": True,
            "status": "LOADED (NO-TRADE PREFLIGHT)",
            "detail": (
                "The verified Six-Agent Team and decoder loaded successfully in the "
                "no-trade preflight. No live worker is running."
            ),
        }
    return {
        "runtime": live_runtime,
        "preflight": report,
        "live_trading": (
            "ON"
            if live_runtime["system_status"] == "RUNNING"
            and live_runtime["mode"] == "live"
            and live_runtime["model"]["loaded"] is True
            else "OFF"
        ),
        "dashboard_run_mode": runtime_readiness(RUN)["mode"],
    }


def trade_pnl():
    """Realized P&L of every settled SELL, keyed by its fill, on average cost."""
    out = {}
    try:
        db = sqlite3.connect(f"file:{RUN / 'ledger.sqlite'}?mode=ro", uri=True)
        rows = db.execute(
            "SELECT plan, settlement FROM orders"
            " WHERE status='SETTLED' AND settlement IS NOT NULL ORDER BY created"
        ).fetchall()
        db.close()
    except sqlite3.Error:
        return out
    held = {}
    for plan, settlement in rows:
        p, s = json.loads(plan), json.loads(settlement)
        base, quote, fee = Decimal(s["base"]), Decimal(s["quote"]), Decimal(s["fee"])
        qty, cost = held.get(p["product"], (Decimal(0), Decimal(0)))
        if p["side"] == "BUY":
            held[p["product"]] = (qty + base, cost + quote + fee)
        elif qty > 0:
            sold = min(base, qty)
            basis = cost * sold / qty
            out[f"{s['base']}|{s['quote']}"] = float(quote - fee - basis)
            held[p["product"]] = (qty - sold, cost - basis)
    return out


def state():
    events = []
    pnl = trade_pnl()
    path = RUN / "events.jsonl"
    if path.exists():
        for line in path.read_text().splitlines()[-400:]:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            nr = r["neural"]
            events.append(
                {
                    "tick": r["tick"],
                    "time": r["wall_time"],
                    "product": r["product"],
                    "bid": r["quote"]["bid"],
                    "equity": float(r["equity_usdc"]),
                    "pnl": float(r["pnl_delta_usdc"]),
                    "side": nr["side"],
                    "left_hz": nr["left_hz"],
                    "right_hz": nr["right_hz"],
                    "diff_hz": nr["difference_hz"],
                    "gate": nr["gate_spikes"],
                    "stimulus": nr["stimulus"],
                    "reward_spikes": nr["reward_spikes"],
                    "aversive_spikes": nr["aversive_spikes"],
                    "kc_spikes": nr["KC_spikes"],
                    "total_spikes": nr["total_spikes"],
                    "compute": nr["compute_seconds"],
                    "changed_edges": nr["memory"]["changed_edges"],
                    "mean_efficacy": nr["memory"]["mean_efficacy"],
                    "min_efficacy": nr["memory"]["minimum_efficacy"],
                    "execution": r["execution"].get("status"),
                    "reason": r["execution"].get("reason") or r.get("signal_reason"),
                    "trade_pnl": pnl.get(f"{r['execution'].get('base')}|{r['execution'].get('quote')}"),
                }
            )
    cp = checkpoint()
    ledger_state = ledger()
    return {
        "run": RUN.name,
        "now": time.time(),
        "events": events,
        "ledger": ledger_state,
        "paper": paper_portfolio(events, ledger_state),
        "stopped": (RUN / "STOP").exists(),
        "checkpoint": None
        if cp is None
        else {k: cp[k] for k in ["luminance", "r8_light", "efficacy_hist", "efficacy_edges", "brain_ms", "mtime"]},
    }


def _is_loopback_name(hostname):
    try:
        return ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        return False


def _failed_preflight(detail):
    return {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "result": "FAIL",
        "checks": [{"name": "Existing Coinbase preflight", "status": "FAIL", "detail": detail}],
        "values": {
            "mode": "live preflight-only",
            "live_trading": "OFF",
            "orders_submitted": 0,
        },
        "orders_submitted": 0,
    }


def _write_preflight(report):
    PREFLIGHT_RUN.mkdir(parents=True, exist_ok=True)
    temp = PREFLIGHT_FILE.with_suffix(".json.partial")
    temp.write_text(json.dumps(report, indent=2) + "\n")
    temp.replace(PREFLIGHT_FILE)


def _report_timestamp(report):
    try:
        return datetime.fromisoformat(report["timestamp"].replace("Z", "+00:00")).timestamp()
    except (KeyError, TypeError, ValueError):
        return 0


class Handler(BaseHTTPRequestHandler):
    def send(self, body, ctype, cache=False, status=200):
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "max-age=3600" if cache else "no-store")
        self.end_headers()
        self.wfile.write(body)

    def send_range(self, path, ctype):
        # <video> needs byte ranges to seek and loop reliably.
        size = path.stat().st_size
        start, end = 0, size - 1
        m = re.fullmatch(r"bytes=(\d*)-(\d*)", self.headers.get("Range", ""))
        if m and (m.group(1) or m.group(2)):
            if m.group(1):
                start = int(m.group(1))
                end = min(int(m.group(2)), size - 1) if m.group(2) else size - 1
            else:
                start = max(0, size - int(m.group(2)))
            if start > end:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self.end_headers()
                return
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        else:
            self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header("Cache-Control", "max-age=3600")
        self.end_headers()
        try:
            with path.open("rb") as f:
                f.seek(start)
                self.wfile.write(f.read(end - start + 1))
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_GET(self):
        route = self.path.split("?")[0]
        if route in ("/", "/index.html"):
            return self.send((HERE / "index.html").read_bytes(), "text/html; charset=utf-8")
        if route == "/flies.json":
            return self.send(json.dumps(FLIES).encode(), "application/json")
        if route == "/meta.json":
            return self.send(json.dumps(META).encode(), "application/json", True)
        if route == "/pos.bin":
            return self.send(POS.tobytes(), "application/octet-stream", True)
        if route == "/sc.bin":
            return self.send(SC.tobytes(), "application/octet-stream", True)
        if route == "/hop.bin":
            return self.send(HOP.tobytes(), "application/octet-stream", True)
        if route == "/type.bin":
            return self.send(TYPE.tobytes(), "application/octet-stream", True)
        if route == "/state.json":
            return self.send(json.dumps(state()).encode(), "application/json")
        if route == "/readiness.json":
            return self.send(
                json.dumps(readiness_state()).encode(), "application/json"
            )
        if route == "/counts.bin":
            cp = checkpoint()
            return self.send(cp["counts"] if cp else b"", "application/octet-stream")
        if route == "/input.png" and (RUN / "latest-input.png").exists():
            return self.send((RUN / "latest-input.png").read_bytes(), "image/png")
        m = re.fullmatch(r"/anim/(buy|sell_profit|sell_loss|hold)\.mp4", route)
        if m and (ANIM_DIR / f"{m.group(1)}.mp4").is_file():
            return self.send_range(ANIM_DIR / f"{m.group(1)}.mp4", "video/mp4")
        self.send_error(404)

    def do_POST(self):
        route = self.path.split("?")[0]
        if route not in ("/api/preflight", "/api/paper/start", "/api/paper/stop"):
            return self.send_error(404)
        try:
            address = ipaddress.ip_address(self.client_address[0])
        except ValueError:
            return self.send_error(403, "Actions are restricted to localhost")
        if not address.is_loopback:
            return self.send_error(403, "Actions are restricted to localhost")
        origin = self.headers.get("Origin")
        if origin:
            try:
                parsed = urlsplit(origin)
                valid_port = parsed.port == PORT
            except ValueError:
                valid_port = False
                parsed = None
            if (
                parsed is None
                or parsed.scheme not in ("http", "https")
                or not valid_port
                or not parsed.hostname
                or not (
                    parsed.hostname == "localhost"
                    or _is_loopback_name(parsed.hostname)
                )
            ):
                return self.send_error(403, "Invalid local action origin")
        if route == "/api/paper/start":
            if RUN.resolve() == PREFLIGHT_RUN.resolve():
                return self.send(
                    b'{"error":"The live run directory cannot be started in paper mode"}',
                    "application/json",
                    status=409,
                )
            run_ledger = ledger()
            if (RUN / "ledger.sqlite").exists() and run_ledger.get("mode") != "paper":
                return self.send(
                    b'{"error":"Refusing to start paper mode in a non-paper or unreadable run directory"}',
                    "application/json",
                    status=409,
                )
            active = worker_active(RUN)
            if active is None:
                return self.send(
                    b'{"error":"Could not verify the paper worker lock"}',
                    "application/json",
                    status=503,
                )
            if active:
                return self.send(
                    b'{"error":"A worker already owns this run directory"}',
                    "application/json",
                    status=409,
                )
            if (RUN / "STOP").exists():
                return self.send(
                    b'{"error":"Emergency STOP is set; review the run and clear STOP explicitly before restarting"}',
                    "application/json",
                    status=409,
                )
            if run_ledger.get("halted"):
                return self.send(
                    b'{"error":"Paper ledger is halted and requires operator review"}',
                    "application/json",
                    status=409,
                )
            try:
                subprocess.Popen(
                    [
                        sys.executable,
                        "-m",
                        "stonkfly",
                        "run",
                        "--paper",
                        "--out",
                        str(RUN),
                    ],
                    cwd=ROOT,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    close_fds=True,
                )
            except OSError as exc:
                return self.send(
                    json.dumps(
                        {"error": f"Could not start the paper worker ({type(exc).__name__})"}
                    ).encode(),
                    "application/json",
                    status=500,
                )
            return self.send(
                json.dumps(
                    {
                        "ok": True,
                        "mode": "paper",
                        "status": "STARTING",
                        "run": str(RUN),
                        "command": [
                            sys.executable,
                            "-m",
                            "stonkfly",
                            "run",
                            "--paper",
                            "--out",
                            str(RUN),
                        ],
                        "live_trading": "OFF",
                    }
                ).encode(),
                "application/json",
                status=202,
            )
        if route == "/api/paper/stop":
            run_ledger = ledger()
            if run_ledger.get("mode") != "paper":
                return self.send(
                    b'{"error":"Refusing to stop a run that is not in paper mode"}',
                    "application/json",
                    status=409,
                )
            active = worker_active(RUN)
            if active is not True:
                return self.send(
                    b'{"error":"No active paper worker owns this run directory"}',
                    "application/json",
                    status=409,
                )
            (RUN / "STOP").touch()
            return self.send(
                json.dumps(
                    {
                        "ok": True,
                        "mode": "paper",
                        "status": "STOP REQUESTED",
                        "run": str(RUN),
                        "live_trading": "OFF",
                    }
                ).encode(),
                "application/json",
            )
        if not PREFLIGHT_LOCK.acquire(blocking=False):
            return self.send(
                b'{"error":"A preflight is already running"}',
                "application/json",
                status=409,
            )
        try:
            started = time.time()
            try:
                process = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "stonkfly",
                        "run",
                        "--live",
                        "--preflight-only",
                    ],
                    cwd=ROOT,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=120,
                    check=False,
                )
                if process.returncode and not PREFLIGHT_FILE.exists():
                    raise RuntimeError("Preflight exited without writing a report")
            except subprocess.TimeoutExpired:
                report = _failed_preflight(
                    "Preflight timed out after 120 seconds"
                )
                _write_preflight(report)
            except Exception as exc:
                report = _failed_preflight(
                    f"Existing preflight command failed ({type(exc).__name__})"
                )
                _write_preflight(report)
            try:
                report = json.loads(PREFLIGHT_FILE.read_text())
            except (OSError, json.JSONDecodeError):
                report = _failed_preflight("Preflight report is unavailable")
                _write_preflight(report)
            if _report_timestamp(report) < started:
                report = _failed_preflight(
                    "Preflight did not produce a fresh result"
                )
                _write_preflight(report)
            response = readiness_state()
            return self.send(json.dumps(response).encode(), "application/json")
        finally:
            PREFLIGHT_LOCK.release()

    # Other non-GET methods cannot modify records or invoke any trading action.
    def refuse(self):
        self.send_error(405, "Read-only dashboard")

    do_PUT = do_PATCH = do_DELETE = refuse

    def log_message(self, *args):
        pass


if __name__ == "__main__":
    if not (RUN / "events.jsonl").exists():
        print(f"Note: {RUN} has no events yet; start `python -m stonkfly run` first.", flush=True)
    print(f"Stonkfly dashboard: http://{HOST}:{PORT}  (run: {RUN.name})", flush=True)
    ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()
