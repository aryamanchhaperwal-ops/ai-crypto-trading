# Deployment Procedure

This guide outlines the strict requirements for deploying the stonkfly application in a live trading environment.

## 1. Prerequisites
- Python 3.10+
- A dedicated Coinbase account/sub-account with a specific Portfolio ID containing exactly a max of 100 USDC and NO other assets (to satisfy strict risk control requirements).
- Coinbase API Credentials downloaded as a JSON file.

## 2. Installation
```bash
# Clone the repository
git clone <repository_url>
cd AI-Crypto-Trading-Agent

# Create and activate a virtual environment
virtualenv env
source env/bin/activate

# Install the package and dependencies
pip install -e .
```

## 3. Configuration
The application requires explicit opt-in for live trading. You must provide these environment variables (or place them in a `.env` file at the root of the project):
```bash
export STONKFLY_LIVE=I_ACCEPT_REAL_TRADES
export COINBASE_KEY_FILE=/path/to/your/coinbase_key.json
export COINBASE_PORTFOLIO_ID=your_coinbase_portfolio_id
```

## 4. Run Read-Only Preflight
Before executing live trades, you must run the preflight check to verify connectivity, API permissions, account balances, and market discovery:
```bash
python -m stonkfly run --live --preflight-only --auto-discover
```
Review the `preflight.json` output carefully. Ensure all checks indicate `PASS`. It will explicitly report `FAIL` if the API key is invalid, permissions are wrong, open orders exist, or the balance does not adhere strictly to the rules (e.g., more than 100 USDC or non-USDC assets present).

## 5. Live Trading Execution
Once the preflight successfully completes and you have confirmed your environment is safely isolated, you can start the automated trading bot:
```bash
python -m stonkfly run --live --auto-discover
```
