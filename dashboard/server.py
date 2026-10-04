"""Stonkfly live neural dashboard (read-only).

Reads the checkpoints, events and ledger that `python -m stonkfly run` writes into
a run directory and serves them to index.html. It never writes to the run and
refuses every non-GET request.

Usage (from the repository root):
    python dashboard/server.py [RUN_DIR]        # default: runs/paper

Environment variables: see README.md ("Dashboard configuration").
"""

import json
import re
import os
import sqlite3
from decimal import Decimal
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

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
            for k in ["mode", "tick", "cash", "positions", "initial_cash", "halted"]
        }
    except sqlite3.Error:
        return {}


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
                    "reason": r["execution"].get("reason"),
                    "trade_pnl": pnl.get(f"{r['execution'].get('base')}|{r['execution'].get('quote')}"),
                }
            )
    cp = checkpoint()
    return {
        "run": RUN.name,
        "now": time.time(),
        "events": events,
        "ledger": ledger(),
        "stopped": (RUN / "STOP").exists(),
        "checkpoint": None
        if cp is None
        else {k: cp[k] for k in ["luminance", "r8_light", "efficacy_hist", "efficacy_edges", "brain_ms", "mtime"]},
    }


class Handler(BaseHTTPRequestHandler):
    def send(self, body, ctype, cache=False):
        self.send_response(200)
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
        if route == "/counts.bin":
            cp = checkpoint()
            return self.send(cp["counts"] if cp else b"", "application/octet-stream")
        if route == "/input.png" and (RUN / "latest-input.png").exists():
            return self.send((RUN / "latest-input.png").read_bytes(), "image/png")
        m = re.fullmatch(r"/anim/(buy|sell_profit|sell_loss|hold)\.mp4", route)
        if m and (ANIM_DIR / f"{m.group(1)}.mp4").is_file():
            return self.send_range(ANIM_DIR / f"{m.group(1)}.mp4", "video/mp4")
        self.send_error(404)

    # Read-only dashboard: refuse every method that could change state.
    def refuse(self):
        self.send_error(405, "Read-only dashboard")

    do_POST = do_PUT = do_PATCH = do_DELETE = refuse

    def log_message(self, *args):
        pass


if __name__ == "__main__":
    if not (RUN / "events.jsonl").exists():
        print(f"Note: {RUN} has no events yet; start `python -m stonkfly run` first.", flush=True)
    print(f"Stonkfly dashboard: http://{HOST}:{PORT}  (run: {RUN.name})", flush=True)
    ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()
