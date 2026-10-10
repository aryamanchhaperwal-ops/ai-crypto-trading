# AI Crypto Trading

An experimental AI-driven cryptocurrency trading engine with neural decision processing and Coinbase integration.

## Overview

The engine converts public Coinbase price observations into an RGB chart and presents it to a retained MaleCNS v1.0 neural graph containing 166,700 neurons and 25.6 million connections. A fixed neural readout proposes buy, sell, or hold. An execution guard checks limits and may reject a proposal; it does not replace it with a different trade.

Positive portfolio P&L stimulates identified PAM11 dopamine cells, while negative P&L stimulates PPL101 cells. A candidate memory rule adjusts existing KC-to-MBON connections. These are engineered reinforcement signals, not modeled pain receptors. Synaptic changes do not demonstrate profitable learning. See [model and evidence](docs/model.md).

The browser dashboard displays recorded neural activity, the visual input, trading decisions, portfolio data, and charts from a run directory. It includes a paper-portfolio panel with local start/stop controls and a separate LIVE READINESS / PREFLIGHT action that never submits, cancels, or modifies Coinbase orders. See [operations](docs/operations.md).

## Install and run

Python 3.11 and a C++17 compiler are required. Allow several GB for the dataset, dependencies, derived data, and checkpoints; 16 GB RAM is recommended.

```sh
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e '.[test]'
python -m stonkfly prepare
python -m stonkfly run --paper
```

The default is paper trading with real public BTC-USDC prices and a simulated 100 USDC balance. Set `PAPER_STARTING_USDC` locally to configure the simulated starting balance (maximum 100 USDC); it does not access or fund Coinbase. No API key is needed. Runtime logs, simulated fills, sensory images, and resumable neural state are stored locally in `runs/paper/`. Stop through the dashboard's emergency paper stop or `python -m stonkfly paper-stop --out runs/paper`.

To view recorded paper activity and the live-readiness panel, start the dashboard in a second terminal:

```sh
python dashboard/server.py runs/paper
```

Open `http://127.0.0.1:8765` in a browser. Dashboard configuration is described in [operations](docs/operations.md).

## Live trading

Live orders are opt-in and require local credentials. Create a dedicated Coinbase Advanced portfolio with **at most 100 USDC** and a portfolio-scoped **ECDSA API key with View + Trade, no Transfer**. Copy `.env.example` to `.env`, fill it in locally, and run:

```sh
python -m stonkfly run --live --preflight-only
python -m stonkfly run --live
```

Defaults include a $10 maximum order (including reserved fees), 24 attempts per day, and no shorts or leverage. A $20 drawdown stops new orders; **it does not liquidate holdings or cap further losses**. Review [operation and recovery guidance](docs/operations.md) before enabling live execution.

```sh
python -m stonkfly status
python -m pytest -q
```

The project is not funded or connected to anyone's account. Live execution requires your own local credentials and explicit opt-in.
