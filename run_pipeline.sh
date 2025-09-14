#!/bin/bash
set -euo pipefail

# ============================
# Config knobs (override via env)
# ============================
START_DATE="${START_DATE:-2020-01-01}"
END_DATE="${END_DATE:-2025-12-31}"
MAX_TICKERS="${MAX_TICKERS:-80}"
SECTOR_MAP="${SECTOR_MAP:-data/raw/sector_map.csv}"

# If you want a totally fresh run (delete old derived data), set CLEAN=1
CLEAN="${CLEAN:-1}"

FUND_Q_CSV="data/raw/fundamentals_quarterly.csv"
MACRO_CSV="data/raw/sample_macro.csv"

echo "=== 🚀 Starting MarketPrediction Pipeline ==="
echo "Dates: $START_DATE → $END_DATE | Max tickers: $MAX_TICKERS"
echo "Sector map: $SECTOR_MAP"

# ---------------------------
# (Optional) Clean derived data
# ---------------------------
if [[ "$CLEAN" == "1" ]]; then
  echo ">>> Cleaning derived data (parquets/csvs under data/features, data/interim, reports)"
  rm -rf data/features data/interim reports
  mkdir -p data/features data/interim reports
fi

# ---------------------------
# 1) Init
# ---------------------------
echo ">>> Initializing project directories"
python Data_Pipeline.py init

# ---------------------------
# Helper: fetch data with yfinance + FRED
# ---------------------------
fetch_data() {
  echo ">>> Fetching raw data via yfinance (+ FRED for macro)"
  if [[ -f "$SECTOR_MAP" ]]; then
    python fetch_real_data.py \
      --sector-map "$SECTOR_MAP" \
      --start "$START_DATE" \
      --end "$END_DATE" \
      --max-tickers "$MAX_TICKERS"
  else
    echo "⚠️  Sector map not found at '$SECTOR_MAP'. Falling back to a small default ticker set."
    python fetch_real_data.py \
      --tickers AAPL MSFT NVDA AVGO AMD INTC \
      --start "$START_DATE" \
      --end "$END_DATE"
  fi
}

# ---------------------------
# 2) Ensure fundamentals & macro CSVs exist (auto-generate if missing)
# ---------------------------
NEED_FETCH=0
[[ ! -f "$FUND_Q_CSV" ]] && NEED_FETCH=1
[[ ! -f "$MACRO_CSV" ]] && NEED_FETCH=1

if [[ "$NEED_FETCH" == "1" ]]; then
  fetch_data
else
  echo ">>> Fundamentals & macro CSVs already present — skipping fetch"
fi

# ---------------------------
# 3) Ingest fundamentals (if present)
# ---------------------------
if [[ -f "$FUND_Q_CSV" ]]; then
  echo ">>> Ingesting fundamentals (quarterly) → parquet"
  python Data_Pipeline.py ingest-fundamentals --csv "$FUND_Q_CSV"

  echo ">>> Expanding fundamentals to daily"
  python Data_Pipeline.py make-fundamentals-daily
else
  echo "⚠️  '$FUND_Q_CSV' not found — skipping fundamentals ingestion."
fi

# ---------------------------
# 4) Ingest macro (if present)
# ---------------------------
if [[ -f "$MACRO_CSV" ]]; then
  echo ">>> Ingesting macro"
  python Data_Pipeline.py ingest-macro --csv "$MACRO_CSV"
else
  echo "⚠️  '$MACRO_CSV' not found — skipping macro ingestion."
fi

# ---------------------------
# 5) (Optional) Sentiment/regimes
# ---------------------------
if [[ -f "data/raw/sample_sentiment.csv" ]]; then
  echo ">>> Ingesting sentiment"
  python Data_Pipeline.py ingest-sentiment --csv data/raw/sample_sentiment.csv
fi

echo ">>> Building regimes/breadth (skips gracefully if not implemented)"
python Data_Pipeline.py make-regimes || echo "ℹ️  make-regimes not implemented / skipped"

# ---------------------------
# 6) Features & labels
# ---------------------------
echo ">>> Making features"
python Data_Pipeline.py make-features

echo ">>> Making labels"
python Data_Pipeline.py make-labels

# ---------------------------
# 7) Backtest
# ---------------------------
echo ">>> Running walk-forward backtest..."
if ! python backtest_walk_forward.py; then
  echo "⚠️ Backtest failed (commonly: not enough history with current WARMUP/EMBARGO)."
  echo "   Try re-running with a longer history (START_DATE earlier) or lower WARMUP_DAYS."
fi

echo "=== ✅ Pipeline complete ==="
