#!/usr/bin/env python3
"""
Data_Pipeline.py — minimal, leakage-safe starter

Subcommands:
  init                         -> create folders + default config
  ingest-prices --csv FILE     -> load your prices CSV into parquet
  ingest-macro  --csv FILE     -> load macro CSV into parquet (US10Y/US2Y/VIX)
  build-calendar               -> derive trading days from prices
  make-features                -> compute v1 features
  make-labels                  -> compute next-day labels

Expected CSV schemas you provide:
  Prices CSV (wide or long is fine; here we expect LONG):
    date,ticker,open,high,low,close,adj_close,volume
  Macro CSV (daily or lower freq; will be forward-filled):
    date,series,value        # series in {US10Y,US2Y,VIX}

Usage examples (after `python Data_Pipeline.py init`):
  python Data_Pipeline.py ingest-prices --csv data/raw/sample_prices.csv
  python Data_Pipeline.py ingest-macro  --csv data/raw/sample_macro.csv
  python Data_Pipeline.py build-calendar
  python Data_Pipeline.py make-features
  python Data_Pipeline.py make-labels
"""

import argparse
import os
from pathlib import Path
import sys
import json
import pandas as pd
import numpy as np
try:
    import yfinance as yf
except ImportError:
    yf = None

# -------------------------
# Paths & Config
# -------------------------
PROJ = Path(".")
DATA = PROJ / "data"
RAW = DATA / "raw"
INTERIM = DATA / "interim"
FEATURES = DATA / "features"
REPORTS = PROJ / "reports"
CONFIG = PROJ / "configs" / "default.json"

DEFAULT_CFG = {
    "decision_timing": "close",     # decide at close, enter next open
    "entry_timing": "next_open",
    "beta_market": "SPY",           # "SPY" or "median"
    "sector_map_csv": "data/raw/sector_map.csv",  # optional CSV: ticker,sector
    "macro_delta_windows": [1, 5],  # compute deltas over these day windows
    "winsorize": {"p_low": 0.01, "p_high": 0.99}, # per-date winsor for select cols

    "features": {
        "momentum_windows": [5, 20],
        "vol_window": 20,
        "extra_vol_windows": [5, 10],        # new: smaller realized vol windows
        "beta_window": 60,
        "volume_zscore_window": 20,
        "gap": True,
        "high_low_range_window": 20,
        "cross_sectional_ranks": True,
        "sector_ranks": True,                # new: within-sector ranks
        "dollar_volume": True                # new: add dollar volume feature
    },
    "labels": {"horizon": "1d_open_to_close"},
    "files": {
        "prices_raw_parquet": str(RAW / "prices.parquet"),
        "macro_raw_parquet": str(RAW / "macro.parquet"),
        "calendar_csv": str(RAW / "trading_days.csv"),
        "features_parquet": str(FEATURES / "eq_features.parquet"),
        "labels_parquet": str(FEATURES / "labels.parquet")
    }
}


#Create the directories for file organization
def ensure_dirs():
    for p in [DATA, RAW, INTERIM, FEATURES, REPORTS, PROJ / "configs"]:
        p.mkdir(parents=True, exist_ok=True)


#If the file configuration did not work, have user initialize, else load the JSON
def load_cfg():
    if not CONFIG.exists():
        raise FileNotFoundError("Run `init` first to create default config.")
    with open(CONFIG, "r") as f:
        return json.load(f)

def save_cfg(cfg):
    CONFIG.parent.mkdir(parents=True, exist_ok=True)
    with open(CONFIG, "w") as f:
        json.dump(cfg, f, indent=2)

# -------------------------
# Utilities
# -------------------------

#Ensure there are no columns missing from CSV
def _assert_cols(df, req, name):
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing columns: {missing}")
    

#Standardize the date given in CSVs
def _to_datetime(df, col="date"):
    df[col] = pd.to_datetime(df[col], utc=False).dt.tz_localize(None)
    return df

#Sort the information by date
def _sort(df):
    if "ticker" in df.columns:
        return df.sort_values(["ticker", "date"])
    return df.sort_values("date")


def _winsorize_series(s, p_low=0.01, p_high=0.99):
    if s.isna().all():
        return s
    lo, hi = s.quantile(p_low), s.quantile(p_high)
    return s.clip(lower=lo, upper=hi)

def _winsorize_per_date(df, cols, p_low, p_high):
    # Winsorize columns per date to reduce outliers without leaking future info
    def _w(d):
        for c in cols:
            if c in d.columns:
                d[c] = _winsorize_series(d[c], p_low, p_high)
        return d
    return df.groupby("date", group_keys=False).apply(_w)

def _load_sector_map(path_str):
    if not path_str:
        return None
    p = Path(path_str)
    # Only proceed if it's an existing FILE (not a dir)
    if not (p.exists() and p.is_file()):
        return None
    smap = pd.read_csv(p)
    if not {"ticker","sector"}.issubset(smap.columns):
        raise ValueError("sector_map_csv must have columns: ticker,sector")
    smap["ticker"] = smap["ticker"].astype(str).str.upper()
    return smap[["ticker","sector"]]


# -------------------------
# Commands
# -------------------------
def cmd_init(_args):
    ensure_dirs()
    if not CONFIG.exists():
        save_cfg(DEFAULT_CFG)
    (RAW / "README.txt").write_text("Place raw CSVs here (prices, macro) or use CLI to ingest.\n")
    print(f"✅ Project initialized.\n - Folders under ./data\n - Default config at {CONFIG}")


#takes a CSV file,ensures that it exists and had all of the necessary columns. It then drops any rows that are missing values 
#Finally it converts the CSV to a parquet which is more efficient than a CSV
def cmd_ingest_prices(args):
    cfg = load_cfg()
    ensure_dirs()
    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    df = pd.read_csv(csv_path)
    req = ["date","ticker","open","high","low","close","adj_close","volume"]
    _assert_cols(df, req, "prices CSV")
    df = _to_datetime(df, "date")
    df = _sort(df).reset_index(drop=True)
    # Basic cleaning
    df = df.dropna(subset=["date","ticker","adj_close"]).copy()
    df["ticker"] = df["ticker"].astype(str).str.upper()
    # Save to parquet
    out = Path(cfg["files"]["prices_raw_parquet"])
    df.to_parquet(out, index=False)
    print(f"✅ Ingested prices → {out} ({len(df):,} rows, {df['ticker'].nunique()} tickers)")

#takes in a CSV of macroeconimoc factors and cleans it in a similar way to cmd_ingest_prices
def cmd_ingest_macro(args):
    cfg = load_cfg()
    ensure_dirs()
    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    df = pd.read_csv(csv_path)
    req = ["date","series","value"]
    _assert_cols(df, req, "macro CSV")
    df = _to_datetime(df, "date")
    df["series"] = df["series"].astype(str).str.upper()
    df = df.dropna(subset=["date","series","value"]).copy()
    df = _sort(df)
    out = Path(cfg["files"]["macro_raw_parquet"])
    df.to_parquet(out, index=False)
    print(f"✅ Ingested macro → {out} ({len(df):,} rows, {df['series'].nunique()} series)")


#takes unique dates from stocks and uses them to figure out what days were trading days. 
def cmd_build_calendar(_args):
    cfg = load_cfg()
    prices = pd.read_parquet(cfg["files"]["prices_raw_parquet"])
    prices = _sort(prices)
    trading_days = (
        prices[["date"]].drop_duplicates().sort_values("date")
    )
    out = Path(cfg["files"]["calendar_csv"])
    trading_days.to_csv(out, index=False)
    print(f"✅ Built trading calendar → {out} ({len(trading_days):,} days)")


#fills in macro values from last known value to ensure there are no missing values caused by values not being updated day of
def _forward_fill_macro_to_daily(macro_df, calendar_df):
    # Pivot to wide: date x series
    wide = macro_df.pivot(index="date", columns="series", values="value").sort_index()
    # Reindex to trading calendar and forward-fill
    cal = calendar_df["date"].sort_values().unique()
    wide = wide.reindex(cal).ffill()
    wide = wide.reset_index()
    return wide  # columns like US10Y, US2Y, VIX

#calculates the beta value for each stock over a 60 day rolling window to measure the inherent risk of the stock
def _rolling_beta(stock_ret, mkt_ret, window):
    """
    Simple rolling beta via cov/var.
    Returns series aligned to stock_ret index.
    """
    cov = stock_ret.rolling(window).cov(mkt_ret)
    var = mkt_ret.rolling(window).var()
    beta = cov / var
    return beta

#creates the features from the cleaned data that will then be put into the machine learning model. Ensures that it is leak proof by only using
#data that is up to day t if we are on day t + 1. This ensures that we are not using data that would not have been known yet to train the model
#on the past outcomes.
def cmd_make_features(_args):
    cfg = load_cfg()
    f = cfg["files"]
    prices = pd.read_parquet(f["prices_raw_parquet"])
    prices = _sort(prices).reset_index(drop=True)

    _assert_cols(prices, ["date","ticker","open","high","low","close","adj_close","volume"], "prices")

    # Returns
    prices["log_ret"] = np.log(prices.groupby("ticker")["adj_close"].pct_change() + 1.0)
    prices["ret_1"]   = prices.groupby("ticker")["adj_close"].pct_change()

    # Momentum windows
    w_moms = cfg["features"]["momentum_windows"]
    for w in w_moms:
        prices[f"mom_{w}"] = prices.groupby("ticker")["adj_close"].pct_change(periods=w)

    # Volatility (annualized) + extra short windows
    vol_w = cfg["features"]["vol_window"]
    prices["vol_20"] = (
        prices.groupby("ticker")["log_ret"]
              .rolling(vol_w).std().reset_index(level=0, drop=True) * np.sqrt(252)
    )
    for vw in cfg["features"].get("extra_vol_windows", []):
        prices[f"vol_{vw}"] = (
            prices.groupby("ticker")["log_ret"]
                  .rolling(vw).std().reset_index(level=0, drop=True) * np.sqrt(252)
        )

    # Dollar volume (liquidity)
    if cfg["features"].get("dollar_volume", True):
        prices["dollar_vol"] = prices["adj_close"] * prices["volume"]

    # Volume zscore
    vz_w = cfg["features"]["volume_zscore_window"]
    grp = prices.groupby("ticker")["volume"]
    mean_v = grp.rolling(vz_w).mean().reset_index(level=0, drop=True)
    std_v  = grp.rolling(vz_w).std().reset_index(level=0, drop=True)
    prices["vol_z20"] = (prices["volume"] - mean_v) / std_v

    # Gap & high-low range
    prices["prev_close"] = prices.groupby("ticker")["adj_close"].shift(1)
    prices["gap"] = (prices["open"] - prices["prev_close"]) / prices["prev_close"]
    hl_w = cfg["features"]["high_low_range_window"]
    prices["hl_range"] = (prices["high"] - prices["low"]) / prices["adj_close"]
    prices["high_low_range20"] = (
        prices.groupby("ticker")["hl_range"].rolling(hl_w).mean().reset_index(level=0, drop=True)
    )

    # -------------------
    # Market proxy & beta / idio vol
    # -------------------
    ret = prices[["date","ticker","log_ret"]].dropna()
    beta_w = cfg["features"]["beta_window"]
    beta_market = cfg.get("beta_market", "SPY").upper()

    if beta_market == "SPY":
        # Use SPY as market; if not present, fallback to cross-sectional median
        spy = prices.loc[prices["ticker"] == "SPY", ["date","log_ret"]].rename(columns={"log_ret":"mkt_ret"})
        if len(spy):
            tmp = ret.merge(spy, on="date", how="left").dropna(subset=["log_ret","mkt_ret"])
        else:
            mkt = ret.groupby("date")["log_ret"].median().rename("mkt_ret").reset_index()
            tmp = ret.merge(mkt, on="date", how="left").dropna(subset=["log_ret","mkt_ret"])
    else:
        # Cross-sectional median return as market
        mkt = ret.groupby("date")["log_ret"].median().rename("mkt_ret").reset_index()
        tmp = ret.merge(mkt, on="date", how="left").dropna(subset=["log_ret","mkt_ret"])

    # Ensure stable order for alignment
    tmp = tmp.sort_values(["ticker", "date"]).reset_index(drop=True)

    beta_series = (
        tmp.groupby("ticker", group_keys=False)
        .apply(lambda d: _rolling_beta(d["log_ret"], d["mkt_ret"], beta_w),
                include_groups=False)
        .reset_index(drop=True)
    )

    tmp["beta_rolling"] = beta_series.values

    tmp["resid"] = tmp["log_ret"] - tmp["beta_rolling"] * tmp["mkt_ret"]
    tmp["idio_vol_60"] = tmp.groupby("ticker")["resid"].rolling(beta_w).std().reset_index(level=0, drop=True)

    features = prices.merge(
        tmp[["date","ticker","beta_rolling","idio_vol_60"]],
        on=["date","ticker"], how="left"
    )

    # -------------------
    # Cross-sectional ranks (universe-wide)
    # -------------------
    if cfg["features"]["cross_sectional_ranks"]:
        rank_cols = [f"mom_{w}" for w in w_moms] + ["vol_20","idio_vol_60","vol_z20"]
        rank_cols += [c for c in features.columns if c.startswith("vol_") and c not in {"vol_20"}]
        if cfg["features"].get("dollar_volume", True):
            rank_cols += ["dollar_vol"]
        for col in rank_cols:
            if col in features.columns:
                features[f"rank_{col}_pct"] = (
                    features.groupby("date")[col].rank(pct=True, method="average")
                )

    # -------------------
    # Sector ranks (optional, requires sector_map.csv)
    # -------------------
    if cfg["features"].get("sector_ranks", True):
        smap = _load_sector_map(cfg.get("sector_map_csv", ""))
        if smap is not None:
            features = features.merge(smap, on="ticker", how="left")
            if "sector" in features.columns:
                for col in [f"mom_{w}" for w in w_moms] + ["vol_20","idio_vol_60"]:
                    if col in features.columns:
                        features[f"rank_sector_{col}_pct"] = (
                            features.groupby(["date","sector"])[col].rank(pct=True, method="average")
                        )

    # -------------------
    # Macro merge (optional) + macro deltas
    # -------------------
    macro_cols_added = []
    if Path(f["macro_raw_parquet"]).exists() and Path(f["calendar_csv"]).exists():
        macro = pd.read_parquet(f["macro_raw_parquet"])
        cal = pd.read_csv(f["calendar_csv"])
        macro = _to_datetime(macro, "date")
        cal = _to_datetime(cal, "date")
        macro_wide = _forward_fill_macro_to_daily(macro, cal)
        # Term spread if both present
        if {"US10Y","US2Y"}.issubset(macro_wide.columns):
            macro_wide["TERM_SPREAD"] = macro_wide["US10Y"] - macro_wide["US2Y"]
        # Macro deltas
        delta_ws = cfg.get("macro_delta_windows", [1,5])
        for c in [col for col in macro_wide.columns if col != "date"]:
            for w in delta_ws:
                macro_wide[f"{c}_chg_{w}d"] = macro_wide[c].pct_change(w)
        macro_cols_added = [c for c in macro_wide.columns if c != "date"]
        features = features.merge(macro_wide, on="date", how="left")

    # -------------------
    # Winsorize selected columns per date (reduce outliers)
    # -------------------
    wz = cfg.get("winsorize", {"p_low": 0.01, "p_high": 0.99})
    wlow, whigh = wz.get("p_low", 0.01), wz.get("p_high", 0.99)
    to_winsor = []
    to_winsor += [f"mom_{w}" for w in w_moms if f"mom_{w}" in features.columns]
    to_winsor += [c for c in ["vol_20","idio_vol_60","gap","high_low_range20","vol_z20","dollar_vol"] if c in features.columns]
    to_winsor += [c for c in features.columns if c.startswith("vol_") and c not in {"vol_20"}]
    if to_winsor:
        features = _winsorize_per_date(features, to_winsor, wlow, whigh)

    # -------------------
    # Final select & save
    # -------------------
    keep_cols = [
        "date","ticker","ret_1",
        *[f"mom_{w}" for w in w_moms],
        "vol_20", *[f"vol_{vw}" for vw in cfg["features"].get("extra_vol_windows", []) if f"vol_{vw}" in features.columns],
        "vol_z20","gap","high_low_range20",
        "beta_rolling","idio_vol_60",
    ]
    if cfg["features"].get("dollar_volume", True) and "dollar_vol" in features.columns:
        keep_cols.append("dollar_vol")

    # ranks
    keep_cols += [c for c in features.columns if c.startswith("rank_")]

    # macro cols if present
    keep_cols += [c for c in macro_cols_added if c in features.columns]

    keep_cols = [c for c in keep_cols if c in features.columns]
    out_df = features[keep_cols].dropna().sort_values(["ticker","date"]).reset_index(drop=True)

    out_path = Path(f["features_parquet"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out_path, index=False)
    print(f"✅ Features written → {out_path} ({len(out_df):,} rows)")



#creates the data that the ML model will learn from. Calculates the trade outcomes for each day of each trade being made
#then places this data into a parquet. 
def cmd_make_labels(_args):
    cfg = load_cfg()
    f = cfg["files"]
    prices = pd.read_parquet(f["prices_raw_parquet"])
    prices = _sort(prices)
    # Decide at close (t), enter next open (t+1) → label is next-day open→close return
    # y = (close_{t+1} - open_{t+1}) / open_{t+1}
    prices["open_t1"]  = prices.groupby("ticker")["open"].shift(-1)
    prices["close_t1"] = prices.groupby("ticker")["close"].shift(-1)
    y = (prices["close_t1"] - prices["open_t1"]) / prices["open_t1"]
    labels = prices[["date","ticker"]].copy()
    labels["y_next_1d"] = y
    labels["target_1d"] = (labels["y_next_1d"] > 0).astype(int)
    labels = labels.dropna().sort_values(["ticker","date"]).reset_index(drop=True)
    out = Path(f["labels_parquet"])
    out.parent.mkdir(parents=True, exist_ok=True)
    labels.to_parquet(out, index=False)
    print(f"✅ Labels written → {out} ({len(labels):,} rows)")



def _write_prices_parquet_from_df(df_long, out_path):
    req = ["date","ticker","open","high","low","close","adj_close","volume"]
    _assert_cols(df_long, req, "yfinance result")
    df_long["ticker"] = df_long["ticker"].astype(str).str.upper()
    _to_datetime(df_long, "date")
    df_long = _sort(df_long).dropna(subset=["date","ticker","adj_close"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_long.to_parquet(out_path, index=False)


def cmd_fetch_yfinance(args):
    """
    Fetch daily OHLCV via yfinance and save to the configured parquet:
      prices_raw_parquet
    Usage:
      python Data_Pipeline.py fetch-yfinance --tickers AAPL MSFT SPY --start 2018-01-01 --end 2025-09-08
    You can also supply --universe-file pointing to a CSV/TXT with one ticker per line or a 'ticker' column.
    """
    if yf is None:
        raise RuntimeError("yfinance is not installed. Run: pip install yfinance")

    cfg = load_cfg()
    tickers = list(args.tickers or [])
    if args.universe_file:
        p = Path(args.universe_file)
        if not p.exists():
            raise FileNotFoundError(p)
        if p.suffix.lower() in {".csv", ".tsv"}:
            dfu = pd.read_csv(p)
            if "ticker" in dfu.columns:
                tickers.extend(dfu["ticker"].astype(str).tolist())
            else:
                # assume first column has tickers
                tickers.extend(dfu.iloc[:,0].astype(str).tolist())
        else:
            # txt with one ticker per line
            tickers.extend([line.strip() for line in p.read_text().splitlines() if line.strip()])

    tickers = sorted(set([t.strip().upper() for t in tickers if t.strip()]))
    if not tickers:
        raise ValueError("No tickers provided. Use --tickers ... or --universe-file FILE")

    start = args.start or "2018-01-01"
    end = args.end or None  # yfinance will default to today if None

    print(f"⏬ Fetching {len(tickers)} tickers from yfinance "
          f"({tickers[:8]}{'...' if len(tickers)>8 else ''}) "
          f"start={start} end={end}")
    
    # If config requests SPY as market, ensure it's included
    cfg = load_cfg()   # (already loaded at top of function in your file)
    if cfg.get("beta_market", "SPY").upper() == "SPY" and "SPY" not in tickers:
        tickers.append("SPY")


    # yfinance supports batching with space-joined tickers
    data = yf.download(" ".join(tickers), start=start, end=end, group_by="ticker", auto_adjust=False, progress=True)

    # If a single ticker, yfinance returns a single-level columns DF
    records = []
    if len(tickers) == 1:
        t = tickers[0]
        dft = data.reset_index()
        dft["ticker"] = t
        dft = dft.rename(columns={
            "Date": "date",
            "Open": "open", "High": "high", "Low": "low",
            "Close": "close", "Adj Close": "adj_close", "Volume": "volume"
        })
        records.append(dft)
    else:
        # MultiIndex columns: (field level 1) under each ticker
        for t in tickers:
            if t not in data.columns.get_level_values(0):
                # Sometimes a ticker fails; skip it
                continue
            dft = data[t].reset_index()
            dft["ticker"] = t
            dft = dft.rename(columns={
                "Date": "date",
                "Open": "open", "High": "high", "Low": "low",
                "Close": "close", "Adj Close": "adj_close", "Volume": "volume"
            })
            records.append(dft)

    if not records:
        raise RuntimeError("yfinance returned no data (check tickers and date range).")

    long_df = pd.concat(records, ignore_index=True)
    # Keep only required cols
    long_df = long_df[["date","ticker","open","high","low","close","adj_close","volume"]]

    out = Path(cfg["files"]["prices_raw_parquet"])
    _write_prices_parquet_from_df(long_df, out)
    print(f"✅ Saved prices → {out} ({len(long_df):,} rows, {long_df['ticker'].nunique()} tickers)")


# -------------------------
# CLI
# -------------------------
def main():
    parser = argparse.ArgumentParser(description="Data pipeline CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p0 = sub.add_parser("init", help="Create folders + default config")

    p1 = sub.add_parser("ingest-prices", help="Ingest prices CSV")
    p1.add_argument("--csv", required=True)

    p2 = sub.add_parser("ingest-macro", help="Ingest macro CSV")
    p2.add_argument("--csv", required=True)

    p3 = sub.add_parser("build-calendar", help="Build trading calendar from prices")

    p4 = sub.add_parser("make-features", help="Compute v1 features")

    p5 = sub.add_parser("make-labels", help="Compute next-day labels")

    p_fetch = sub.add_parser("fetch-yfinance", help="Fetch OHLCV via yfinance into prices parquet")
    p_fetch.add_argument("--tickers", nargs="*", help="List of tickers (e.g., AAPL MSFT SPY)")
    p_fetch.add_argument("--universe-file", help="CSV/TXT file with tickers or a 'ticker' column")
    p_fetch.add_argument("--start", help="YYYY-MM-DD (default 2018-01-01)")
    p_fetch.add_argument("--end", help="YYYY-MM-DD (default today)")

    args = parser.parse_args()

    if args.cmd == "init":
        cmd_init(args)
    elif args.cmd == "ingest-prices":
        cmd_ingest_prices(args)
    elif args.cmd == "ingest-macro":
        cmd_ingest_macro(args)
    elif args.cmd == "build-calendar":
        cmd_build_calendar(args)
    elif args.cmd == "make-features":
        cmd_make_features(args)
    elif args.cmd == "make-labels":
        cmd_make_labels(args)
    elif args.cmd == "fetch-yfinance":
        cmd_fetch_yfinance(args)

    else:
        parser.print_help()

    


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    main()