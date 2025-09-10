#!/usr/bin/env bash
# run_pipeline.sh — one-touch pipeline runner
# Usage:
#   chmod +x run_pipeline.sh
#   ./run_pipeline.sh
#
# Optional env vars:
#   START_DATE=2018-01-01 ./run_pipeline.sh
#   UNIVERSE_FILE=data/universe_tickers.txt ./run_pipeline.sh

set -euo pipefail

# ---- Config (override via env) ----
PYTHON_BIN="${PYTHON_BIN:-python}"
START_DATE="${START_DATE:-2018-01-01}"
UNIVERSE_FILE="${UNIVERSE_FILE:-data/universe_tickers.txt}"

echo "==> Using Python: $($PYTHON_BIN -c 'import sys; print(sys.executable)')"
echo "==> Start date:   ${START_DATE}"
echo "==> Universe:     ${UNIVERSE_FILE}"

# ---- 0) Sanity checks ----
if [ ! -f "Data_Pipeline.py" ]; then
  echo "ERROR: Data_Pipeline.py not found in current directory."
  exit 1
fi

# ---- 1) Dependencies ----
echo "==> Installing/validating Python dependencies..."
$PYTHON_BIN -m pip install --upgrade pip >/dev/null
$PYTHON_BIN -m pip install -q pandas numpy pyarrow scikit-learn yfinance fredapi >/dev/null

# ---- 2) Init project structure ----
echo "==> Initializing project folders & config..."
$PYTHON_BIN Data_Pipeline.py init

# ---- 3) Create a default universe file if missing ----
if [ ! -f "${UNIVERSE_FILE}" ]; then
  echo "==> Creating default universe at ${UNIVERSE_FILE}"
  mkdir -p "$(dirname "${UNIVERSE_FILE}")"
  cat > "${UNIVERSE_FILE}" <<'EOF'
AAPL
MSFT
SPY
NVDA
AMZN
GOOGL
META
TSLA
JPM
JNJ
V
MA
HD
XOM
PEP
KO
BAC
AVGO
COST
WMT
NFLX
AMD
INTC
CRM
ORCL
CSCO
QCOM
TXN
UNH
PFE
MRK
ABBV
MCD
CAT
DIS
NKE
TMO
PM
IBM
UPS
BA
HON
SBUX
BKNG
ADP
GILD
LRCX
AMAT
BRK-B
EOF
fi

# ---- 4) Fetch prices via yfinance ----
echo "==> Fetching OHLCV from yfinance..."
$PYTHON_BIN Data_Pipeline.py fetch-yfinance --universe-file "${UNIVERSE_FILE}" --start "${START_DATE}"

# ---- 5) Build trading calendar ----
echo "==> Building trading calendar..."
$PYTHON_BIN Data_Pipeline.py build-calendar

# ---- 6) (Optional) Ingest macro from a CSV if provided ----
# If you have a macro CSV ready, set MACRO_CSV=path.csv and it will be ingested.
if [ "${MACRO_CSV:-}" != "" ]; then
  if [ -f "${MACRO_CSV}" ]; then
    echo "==> Ingesting macro from ${MACRO_CSV}..."
    $PYTHON_BIN Data_Pipeline.py ingest-macro --csv "${MACRO_CSV}"
  else
    echo "WARN: MACRO_CSV=${MACRO_CSV} not found; skipping macro ingest."
  fi
fi

# ---- 7) Make features & labels ----
echo "==> Making features..."
$PYTHON_BIN Data_Pipeline.py make-features

echo "==> Making labels..."
$PYTHON_BIN Data_Pipeline.py make-labels

# ---- 8) Run time-safe baseline (if file exists) ----
if [ -f "train_baseline_timesafe.py" ]; then
  echo "==> Running time-safe baseline..."
  $PYTHON_BIN train_baseline_timesafe.py || true
else
  echo "NOTE: train_baseline_timesafe.py not found; skipping baseline."
fi

# ---- 9) Run walk-forward backtest (if file exists) ----
if [ -f "backtest_walk_forward.py" ]; then
  echo "==> Running walk-forward backtest..."
  $PYTHON_BIN backtest_walk_forward.py || true
else
  echo "NOTE: backtest_walk_forward.py not found; skipping walk-forward."
fi

echo "==> Done."
echo "Artifacts:"
echo "  - data/raw/prices.parquet"
echo "  - data/raw/trading_days.csv"
echo "  - data/features/eq_features.parquet"
echo "  - data/features/labels.parquet"
echo "  - reports/wf_daily.csv (if backtest ran)"
echo "  - reports/wf_yearly.csv (if backtest ran)"
