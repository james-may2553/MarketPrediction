#!/usr/bin/env python3
"""
Data_Pipeline.py — leakage-safe, factor-rich starter

Subcommands:
  init                           -> create folders + default config
  ingest-prices --csv FILE       -> load your prices CSV into parquet
  ingest-macro  --csv FILE       -> load macro CSV into parquet (e.g., US10Y/US2Y/VIX/PMI/etc.)
  ingest-fundamentals --csv FILE -> load fundamentals CSV (Value/Quality factors)
  ingest-news --csv FILE         -> load news sentiment CSV (FinBERT/etc.)
  ingest-trends --csv FILE       -> load Google Trends CSV
  build-calendar                 -> derive trading days from prices
  make-features                  -> compute features (price/macros/fundamentals/alt-data)
  make-labels                    -> compute next-day labels
  fetch-yfinance [--tickers ... | --universe-file FILE] [--start YYYY-MM-DD] [--end YYYY-MM-DD]
                                  -> fetch OHLCV via yfinance into prices parquet

Expected CSV schemas you provide:
  Prices CSV (wide or long is fine; here we expect LONG):
    date,ticker,open,high,low,close,adj_close,volume
  Macro CSV (daily or lower freq; will be forward-filled):
    date,series,value        # series like {US10Y,US2Y,VIX,PMI,UNEMP_RATE,...}
  Fundamentals CSV (panel by date,ticker; we will ffill to calendar and lag by shift_days):
    date,ticker,pe,ev_ebitda,pb,roe,roa,gross_margin,op_margin,net_margin
  News CSV (panel by date,ticker):
    date,ticker,sentiment_mean,sentiment_std,article_count
  Trends CSV (panel by date,ticker):
    date,ticker,trends_score

Usage examples (after `python Data_Pipeline.py init`):
  python Data_Pipeline.py ingest-prices --csv data/raw/sample_prices.csv
  python Data_Pipeline.py ingest-macro  --csv data/raw/sample_macro.csv
  python Data_Pipeline.py ingest-fundamentals --csv data/raw/fundamentals.csv
  python Data_Pipeline.py ingest-news         --csv data/raw/news_sentiment.csv
  python Data_Pipeline.py ingest-trends       --csv data/raw/google_trends.csv
  python Data_Pipeline.py build-calendar
  python Data_Pipeline.py make-features
  python Data_Pipeline.py make-labels
  python Data_Pipeline.py fetch-yfinance --tickers AAPL MSFT SPY --start 2018-01-01
"""

import argparse
from pathlib import Path
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
        "extra_vol_windows": [5, 10],        # extra realized vol windows
        "beta_window": 60,
        "volume_zscore_window": 20,
        "gap": True,
        "high_low_range_window": 20,
        "cross_sectional_ranks": True,
        "sector_ranks": True,                # within-sector ranks
        "dollar_volume": True,               # add dollar volume feature
        "higher_moments": True,              # skew/kurtosis on log returns
        "adv_windows": [20],                 # ADV windows in days
        "roll_spread_window": 20             # Roll spread estimator window
    },
    "macro": {
        # series whose changes should be diffs, not pct changes
        "diff_series": ["UNEMP_RATE", "PMI", "CREDIT_SPREAD_BAA10Y"]
    },
    "fundamentals": {
        "shift_days": 1,  # safety lag before ffill/merge
        "rank_columns": ["pe","ev_ebitda","pb","roe","roa","gross_margin","op_margin","net_margin"]
    },
    "alt_data": {
        "news_shift_days": 0,     # shift if needed (e.g., publication cutoff)
        "trends_shift_days": 0
    },
    "labels": {"horizon": "1d_open_to_close"},
    "files": {
        "prices_raw_parquet": str(RAW / "prices.parquet"),
        "macro_raw_parquet": str(RAW / "macro.parquet"),
        "calendar_csv": str(RAW / "trading_days.csv"),
        "features_parquet": str(FEATURES / "eq_features.parquet"),
        "labels_parquet": str(FEATURES / "labels.parquet"),
        "fundamentals_parquet": str(RAW / "fundamentals.parquet"),
        "news_parquet": str(RAW / "news_sentiment.parquet"),
        "trends_parquet": str(RAW / "google_trends.parquet")
    }
}

# -------------------------
# Dir & Config helpers
# -------------------------
def ensure_dirs():
    for p in [DATA, RAW, INTERIM, FEATURES, REPORTS, PROJ / "configs"]:
        p.mkdir(parents=True, exist_ok=True)

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
def _shift_by_days(df, group_cols, date_col, shift_days):
    """
    Shift each group's rows forward in time (positive = move into the future),
    to ensure no lookahead. After shifting, later ffill will align to calendar.
    """
    if not shift_days:
        return df
    return (df.sort_values(group_cols + [date_col])
              .groupby(group_cols, group_keys=False)
              .apply(lambda d: d.assign(**{date_col: d[date_col] + pd.Timedelta(days=shift_days)})))

def _ffill_panel_to_calendar(df, group_cols, date_col, calendar_dates):
    """
    Reindex (group × calendar) and forward-fill within each group.
    """
    df = df.sort_values(group_cols + [date_col])
    frames = []
    cal = pd.Index(calendar_dates, name=date_col)
    for g, d in df.groupby(group_cols):
        d = d.set_index(date_col).reindex(cal).ffill()
        # put group cols back
        if isinstance(g, tuple):
            for k, v in zip(group_cols, g):
                d[k] = v
        else:
            d[group_cols[0]] = g
        d = d.reset_index()
        frames.append(d)
    return pd.concat(frames, ignore_index=True) if frames else df

def _assert_cols(df, req, name):
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing columns: {missing}")

def _to_datetime(df, col="date"):
    df[col] = pd.to_datetime(df[col], utc=False).dt.tz_localize(None)
    return df

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
    if not (p.exists() and p.is_file()):
        return None
    smap = pd.read_csv(p)
    if not {"ticker","sector"}.issubset(smap.columns):
        raise ValueError("sector_map_csv must have columns: ticker,sector")
    smap["ticker"] = smap["ticker"].astype(str).str.upper()
    return smap[["ticker","sector"]]

# -------------------------
# Commands: init & ingest
# -------------------------
def _winsorize_cs(s, p):
    lo, hi = s.quantile(p), s.quantile(1-p)
    return s.clip(lo, hi)

def cmd_make_fundamentals_daily(_args):
    cfg = load_cfg(); f = cfg["files"]
    rawp = Path(f["fundamentals_raw_parquet"])
    calp = Path(f["calendar_csv"])
    if not rawp.exists() or not calp.exists():
        raise FileNotFoundError("Need fundamentals_raw_parquet and trading calendar.")

    # --- Load & normalize upstream tables ---
    fund = pd.read_parquet(rawp)
    if "ticker" not in fund.columns:
        raise ValueError("fundamentals_raw_parquet is missing a 'ticker' column.")
    fund["ticker"] = fund["ticker"].astype(str).str.upper()
    fund["date"] = pd.to_datetime(fund["date"])  # ensure datetime

    cal = pd.read_csv(calp)
    cal = _to_datetime(cal, "date")

    # Expand each ticker to trading days and forward-fill up to N days
    maxff = int(cfg["fundamentals"]["ffill_days"])
    tickers = fund["ticker"].astype(str).str.upper().unique()

    # Keep fundamentals columns minimal to avoid merge collisions
    keep_cols = [
        "date","ticker","pe","pb","ps","ev_ebitda",
        "roe","roa","gross_margin","oper_margin",
        "market_cap","shares","debt_to_equity"
    ]
    keep_cols = [c for c in keep_cols if c in fund.columns]

    out = []
    for t in tickers:
        dft = fund.loc[fund["ticker"] == t, keep_cols].sort_values("date").copy()
        if dft.empty:
            continue

        tmp = cal.copy()
        tmp["ticker"] = t
        # Validate merge cardinality: one calendar row per (date,ticker), many fundamentals rows allowed only if your source has duplicates
        tmp = tmp.merge(dft, on=["date","ticker"], how="left", validate="many_to_one")
        tmp = tmp.sort_values("date")

        # Forward-fill per ticker with limit using column-wise ffill to preserve 'ticker' column
        num_cols = [c for c in tmp.columns if c not in ("date", "ticker")]
        tmp[num_cols] = tmp.groupby("ticker")[num_cols].ffill(limit=maxff)
        # Ensure ticker column survives any groupby/index shenanigans
        tmp["ticker"] = t

        out.append(tmp)

    if not out:
        raise RuntimeError("No fundamentals rows produced after expansion/ffill.")

    daily = pd.concat(out, ignore_index=True)

    # --- Robustly ensure plain 'ticker' and 'date' columns exist before using them ---
    cols = set(daily.columns)

    if "ticker" not in cols:
        for cand in ("ticker_x", "ticker_y", "Ticker", "level_0", "level_1", "group"):
            if cand in cols:
                daily["ticker"] = daily[cand]
                break
        else:
            raise KeyError(f"'ticker' missing. Columns: {list(daily.columns)[:50]}")

    if "date" not in daily.columns:
        for cand in ("date_x", "date_y", "Date"):
            if cand in daily.columns:
                daily["date"] = daily[cand]
                break
        else:
            raise KeyError(f"'date' missing. Columns: {list(daily.columns)[:50]}")

    if "ticker_x" in daily.columns and "ticker_y" in daily.columns:
        daily["ticker"] = daily["ticker_x"].fillna(daily["ticker_y"]) 
        daily = daily.drop(columns=["ticker_x","ticker_y"])
    if "date_x" in daily.columns and "date_y" in daily.columns:
        daily["date"] = pd.to_datetime(daily["date_x"]).fillna(pd.to_datetime(daily["date_y"]))
        daily = daily.drop(columns=["date_x","date_y"])

    # Drop duplicated column names (keep first)
    daily = daily.loc[:, ~daily.columns.duplicated()]

    # Normalize types
    daily["ticker"] = daily["ticker"].astype(str).str.upper()
    daily["date"] = pd.to_datetime(daily["date"]) 

    # Engineering: invert some ratios so "bigger is better"
    for col in ["pe","pb","ps","ev_ebitda"]:
        if col in daily.columns:
            daily[f"inv_{col}"] = 1.0 / daily[col].replace(0, np.nan)

    # Quality/size
    if "market_cap" in daily.columns:
        daily["log_mcap"] = np.log(daily["market_cap"].where(daily["market_cap"] > 0))
    if "shares_out" in daily.columns and "float_turnover" not in daily.columns:
        daily["float_turnover"] = np.nan  # placeholder for ADV-based metric later

    # Cross-sectional winsorize per day (numeric cols only)
    p = float(cfg["fundamentals"]["winsor_pct"])  # e.g., 0.01 means clip to [1%,99%]
    numcols = daily.select_dtypes(include=[np.number]).columns.tolist()
    for c in [x for x in numcols if x not in []]:  # keep list of identifiers empty
        daily[c] = daily.groupby("date")[c].transform(lambda s: _winsorize_cs(s, p))

    # Time z-scores per ticker for curated list
    z_targets = [
        "inv_pe","inv_pb","inv_ps","inv_ev_ebitda",
        "roe","roa","gross_margin","oper_margin","log_mcap","debt_to_equity"
    ]
    for w in cfg["fundamentals"]["zscore_windows"]:
        for c in z_targets:
            if c in daily.columns:
                m = daily.groupby("ticker")[c].rolling(w).mean().reset_index(level=0, drop=True)
                s = daily.groupby("ticker")[c].rolling(w).std().reset_index(level=0, drop=True)
                daily[f"z_{c}_{w}"] = (daily[c] - m) / s

    # Final select/sort
    keep = ["date","ticker"] + [c for c in daily.columns if c not in {"market_cap","shares_out"}]
    seen = set(); ordered_keep = []
    for c in keep:
        if c not in seen and c in daily.columns:
            ordered_keep.append(c); seen.add(c)

    daily = (
        daily[ordered_keep]
        .dropna(subset=["date","ticker"])
        .sort_values(["ticker","date"]) 
        .reset_index(drop=True)
    )

    outp = Path(f["fundamentals_daily_parquet"])
    outp.parent.mkdir(parents=True, exist_ok=True)
    daily.to_parquet(outp, index=False)
    print(f"✅ Fundamentals (daily) → {outp} ({len(daily):,} rows)")

def cmd_ingest_fundamentals(args):
    """
    Ingest quarterly fundamentals from CSV into parquet.
    Expects at minimum: date, ticker, plus numeric fundamental columns.
    """
    cfg = load_cfg()
    f = cfg["files"]

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    df = pd.read_csv(csv_path)
    # Basic sanity
    req = ["date","ticker"]
    _assert_cols(df, req, "fundamentals CSV")

    df = _to_datetime(df, "date")
    df["ticker"] = df["ticker"].astype(str).str.upper()

    # Drop rows with no fundamentals at all
    keep_cols = [c for c in df.columns if c not in {"date","ticker"}]
    df = df.dropna(subset=keep_cols, how="all")

    out = Path(f["fundamentals_raw_parquet"])
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"✅ Ingested fundamentals → {out} ({len(df):,} rows, {df['ticker'].nunique()} tickers)")



def cmd_ingest_sentiment(args):
    cfg = load_cfg()
    path = Path(args.csv)
    if not path.exists(): raise FileNotFoundError(path)
    df = pd.read_csv(path)
    _assert_cols(df, ["date","ticker","sentiment"], "sentiment CSV")
    df = _to_datetime(df, "date")
    df["ticker"] = df["ticker"].astype(str).str.upper()
    df["sentiment"] = pd.to_numeric(df["sentiment"], errors="coerce")
    out = Path(cfg["files"]["sentiment_raw_parquet"])
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"✅ Ingested sentiment → {out} ({len(df):,} rows)")

def cmd_make_regimes(_args):
    cfg = load_cfg(); f = cfg["files"]
    need = [f["prices_raw_parquet"], f["calendar_csv"]]
    for p in need:
        if not Path(p).exists(): raise FileNotFoundError(p)

    prices = pd.read_parquet(f["prices_raw_parquet"])
    prices = _sort(prices)
    cal = pd.read_csv(f["calendar_csv"]); cal = _to_datetime(cal, "date")

    # Market proxy: cross-sectional median log_ret already in make-features; recompute quickly here
    prices["log_ret"] = np.log(prices.groupby("ticker")["adj_close"].pct_change() + 1.0)
    # Spy/Market daily return (use cross-sec median to avoid dependency)
    mkt = prices.groupby("date")["log_ret"].median().rename("mkt_log_ret").reset_index()

    # Breadth: % of tickers up each day
    up = prices.groupby(["date"]).apply(lambda d: (d["adj_close"].pct_change().fillna(0) > 0).mean()).rename("breadth_up").reset_index()

    # 5/20-day breadth averages
    for w in cfg["regimes"]["breadth_lookbacks"]:
        up[f"breadth_up_{w}"] = up["breadth_up"].rolling(w).mean()

    # Merge macro (US10Y/US2Y/VIX) if present
    macro_wide = None
    if Path(f["macro_raw_parquet"]).exists():
        macro = pd.read_parquet(f["macro_raw_parquet"])
        macro = _to_datetime(macro, "date")
        macro_wide = macro.pivot(index="date", columns="series", values="value").reset_index()

    df = cal.merge(mkt, on="date", how="left").merge(up, on="date", how="left")
    if macro_wide is not None:
        df = df.merge(macro_wide, on="date", how="left")
        if {"US10Y","US2Y"}.issubset(df.columns):
            df["TERM_SPREAD"] = df["US10Y"] - df["US2Y"]

    # Realized vol of market proxy
    rv_win = int(cfg["regimes"]["spy_realized_vol_win"])
    df["mkt_rvol"] = df["mkt_log_ret"].rolling(rv_win).std() * np.sqrt(252)

    # Regime dummies
    vhi, vlo = cfg["regimes"]["vix_high_thresh"], cfg["regimes"]["vix_low_thresh"]
    if "VIX" in df.columns:
        df["regime_high_vol"] = (df["VIX"] >= vhi).astype(int)
        df["regime_low_vol"]  = (df["VIX"] <= vlo).astype(int)
    else:
        df["regime_high_vol"] = (df["mkt_rvol"] >= df["mkt_rvol"].quantile(0.7)).astype(int)
        df["regime_low_vol"]  = (df["mkt_rvol"] <= df["mkt_rvol"].quantile(0.3)).astype(int)

    if "TERM_SPREAD" in df.columns:
        hi, lo = cfg["regimes"]["term_spread_high"], cfg["regimes"]["term_spread_low"]
        df["regime_steep"]  = (df["TERM_SPREAD"] >= hi).astype(int)
        df["regime_flat"]   = (df["TERM_SPREAD"] <= lo).astype(int)

    outp = Path(f["regimes_parquet"])
    outp.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(outp, index=False)
    print(f"✅ Regimes written → {outp} ({len(df):,} days)")



def cmd_init(_args):
    ensure_dirs()
    if not CONFIG.exists():
        save_cfg(DEFAULT_CFG)
    (RAW / "README.txt").write_text("Place raw CSVs here (prices, macro, fundamentals, news, trends) or use CLI to ingest.\n")
    print(f"✅ Project initialized.\n - Folders under ./data\n - Default config at {CONFIG}")

def cmd_ingest_fundamentals(args):
    cfg = load_cfg()
    ensure_dirs()
    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    df = pd.read_csv(csv_path)
    _assert_cols(df, ["date","ticker"], "fundamentals CSV")
    df["ticker"] = df["ticker"].astype(str).str.upper()
    df = _to_datetime(df, "date")
    df = _sort(df).dropna(subset=["date","ticker"]).reset_index(drop=True)
    out = Path(cfg["files"]["fundamentals_parquet"])
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"✅ Ingested fundamentals → {out} ({len(df):,} rows, {df['ticker'].nunique()} tickers)")

def cmd_ingest_news(args):
    cfg = load_cfg()
    ensure_dirs()
    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    df = pd.read_csv(csv_path)
    _assert_cols(df, ["date","ticker"], "news CSV")
    df["ticker"] = df["ticker"].astype(str).str.upper()
    df = _to_datetime(df, "date")
    df = _sort(df).dropna(subset=["date","ticker"]).reset_index(drop=True)
    out = Path(cfg["files"]["news_parquet"])
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"✅ Ingested news sentiment → {out} ({len(df):,} rows, {df['ticker'].nunique()} tickers)")

def cmd_ingest_trends(args):
    cfg = load_cfg()
    ensure_dirs()
    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    df = pd.read_csv(csv_path)
    _assert_cols(df, ["date","ticker","trends_score"], "trends CSV")
    df["ticker"] = df["ticker"].astype(str).str.upper()
    df = _to_datetime(df, "date")
    df = _sort(df).dropna(subset=["date","ticker"]).reset_index(drop=True)
    out = Path(cfg["files"]["trends_parquet"])
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"✅ Ingested Google Trends → {out} ({len(df):,} rows, {df['ticker'].nunique()} tickers)")

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
    df = df.dropna(subset=["date","ticker","adj_close"]).copy()
    df["ticker"] = df["ticker"].astype(str).str.upper()
    out = Path(cfg["files"]["prices_raw_parquet"])
    out.to_parquet  # (lint calm)
    out = Path(cfg["files"]["prices_raw_parquet"])
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"✅ Ingested prices → {out} ({len(df):,} rows, {df['ticker'].nunique()} tickers)")

def cmd_ingest_macro(args):
    cfg = load_cfg()
    ensure_dirs()
    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    df = pd.read_csv(csv_path)
    _assert_cols(df, ["date","series","value"], "macro CSV")
    df = _to_datetime(df, "date")
    df["series"] = df["series"].astype(str).str.upper()
    df = df.dropna(subset=["date","series","value"]).copy()
    df = _sort(df)
    out = Path(cfg["files"]["macro_raw_parquet"])
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"✅ Ingested macro → {out} ({len(df):,} rows, {df['series'].nunique()} series)")

# -------------------------
# Calendar & Macro helpers
# -------------------------
def cmd_build_calendar(_args):
    cfg = load_cfg()
    prices = pd.read_parquet(cfg["files"]["prices_raw_parquet"])
    prices = _sort(prices)
    trading_days = prices[["date"]].drop_duplicates().sort_values("date")
    out = Path(cfg["files"]["calendar_csv"])
    out.parent.mkdir(parents=True, exist_ok=True)
    trading_days.to_csv(out, index=False)
    print(f"✅ Built trading calendar → {out} ({len(trading_days):,} days)")

def _forward_fill_macro_to_daily(macro_df, calendar_df):
    # Pivot to wide: date x series
    wide = macro_df.pivot(index="date", columns="series", values="value").sort_index()
    # Reindex to trading calendar and forward-fill
    cal = calendar_df["date"].sort_values().unique()
    wide = wide.reindex(cal).ffill()
    wide = wide.reset_index()
    return wide  # columns like US10Y, US2Y, VIX, ...

# -------------------------
# Beta helper
# -------------------------
def _rolling_beta(stock_ret, mkt_ret, window):
    """
    Simple rolling beta via cov/var.
    Returns series aligned to stock_ret index.
    """
    cov = stock_ret.rolling(window).cov(mkt_ret)
    var = mkt_ret.rolling(window).var()
    beta = cov / var
    return beta

# -------------------------
# Feature builder
# -------------------------
def cmd_make_features(_args):
    cfg = load_cfg()
    f = cfg["files"]
    prices = pd.read_parquet(f["prices_raw_parquet"])
    prices = _sort(prices).reset_index(drop=True)

    _assert_cols(prices, ["date","ticker","open","high","low","close","adj_close","volume"], "prices")

<<<<<<< Updated upstream
    # -------------------
    # Returns & core tech features (leak-safe; only past info)
    # -------------------
    prices["log_ret"] = np.log(prices.groupby("ticker")["adj_close"].pct_change() + 1.0)
=======
    # Returns
    prices["log_ret"] = np.log1p(prices.groupby("ticker")["adj_close"].pct_change())
>>>>>>> Stashed changes
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
    # Market proxy & beta / idiosyncratic vol
    # -------------------
    ret = prices[["date","ticker","log_ret"]].dropna()
    beta_w = cfg["features"]["beta_window"]
    beta_market = cfg.get("beta_market", "SPY").upper()

    if beta_market == "SPY":
        spy = prices.loc[prices["ticker"] == "SPY", ["date","log_ret"]].rename(columns={"log_ret":"mkt_ret"})
        if len(spy):
            tmp = ret.merge(spy, on="date", how="left").dropna(subset=["log_ret","mkt_ret"])
        else:
            mkt = ret.groupby("date")["log_ret"].median().rename("mkt_ret").reset_index()
            tmp = ret.merge(mkt, on="date", how="left").dropna(subset=["log_ret","mkt_ret"])
    else:
        mkt = ret.groupby("date")["log_ret"].median().rename("mkt_ret").reset_index()
        tmp = ret.merge(mkt, on="date", how="left").dropna(subset=["log_ret","mkt_ret"])

    tmp = tmp.sort_values(["ticker", "date"]).reset_index(drop=True)
    beta_series = (
        tmp.groupby("ticker", group_keys=False)
<<<<<<< Updated upstream
           .apply(lambda d: _rolling_beta(d["log_ret"], d["mkt_ret"], beta_w), include_groups=False)
           .reset_index(drop=True)
=======
          .apply(lambda d: _rolling_beta(d["log_ret"], d["mkt_ret"], beta_w),
                 include_groups=False)
          .reset_index(drop=True)
>>>>>>> Stashed changes
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
    # Sector ranks (optional)
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
    # Macro merge (optional) + macro deltas (pct or diff)
    # -------------------
    macro_cols_added = []
    if Path(f["macro_raw_parquet"]).exists() and Path(f["calendar_csv"]).exists():
        macro = pd.read_parquet(f["macro_raw_parquet"])
        cal = pd.read_csv(f["calendar_csv"])
        macro = _to_datetime(macro, "date")
        cal = _to_datetime(cal, "date")
        macro_wide = _forward_fill_macro_to_daily(macro, cal)
        if {"US10Y","US2Y"}.issubset(macro_wide.columns):
            macro_wide["TERM_SPREAD"] = macro_wide["US10Y"] - macro_wide["US2Y"]
<<<<<<< Updated upstream
=======
        # Macro deltas (pct-change or diffs per config)
        diff_series = set(cfg.get("macro", {}).get("diff_series", []))
>>>>>>> Stashed changes
        delta_ws = cfg.get("macro_delta_windows", [1,5])
        for c in [col for col in macro_wide.columns if col != "date"]:
            for w in delta_ws:
                if c in diff_series:
                    macro_wide[f"{c}_diff_{w}d"] = macro_wide[c].diff(w)
                else:
                    macro_wide[f"{c}_chg_{w}d"] = macro_wide[c].pct_change(w)
        macro_cols_added = [c for c in macro_wide.columns if c != "date"]
        features = features.merge(macro_wide, on="date", how="left")

    # --- Fundamentals (daily) ---
    fundp = Path(f.get("fundamentals_daily_parquet", ""))
    if fundp.exists():
        features = features.merge(pd.read_parquet(fundp), on=["date","ticker"], how="left")

    # --- Sentiment (optional) ---
    sentp = Path(f.get("sentiment_raw_parquet", ""))
    if sentp.exists():
        sent = pd.read_parquet(sentp)
        features = features.merge(sent, on=["date","ticker"], how="left")

    # --- Regimes (date-level) ---
    regimep = Path(f.get("regimes_parquet", ""))
    if regimep.exists():
        regimes = pd.read_parquet(regimep)
        features = features.merge(regimes, on="date", how="left")

    # -------------------
    # Higher-moment volatility (skew/kurtosis)
    # -------------------
    if cfg["features"].get("higher_moments", True):
        wv = cfg["features"]["vol_window"]

        def _moment3(s, window):
            m1 = s.rolling(window).mean()
            sd = s.rolling(window).std()
            num = (s - m1).rolling(window).apply(lambda x: np.mean(x**3), raw=False)
            with np.errstate(invalid="ignore", divide="ignore"):
                return num / (sd**3)

        def _moment4(s, window):
            m1 = s.rolling(window).mean()
            sd = s.rolling(window).std()
            num = (s - m1).rolling(window).apply(lambda x: np.mean(x**4), raw=False)
            with np.errstate(invalid="ignore", divide="ignore"):
                return num / (sd**4)

        features[f"skew_{wv}"] = (
            features.groupby("ticker")["log_ret"]
                    .apply(lambda s: _moment3(s, wv))
                    .reset_index(level=0, drop=True)
        )
        features[f"kurt_{wv}"] = (
            features.groupby("ticker")["log_ret"]
                    .apply(lambda s: _moment4(s, wv))
                    .reset_index(level=0, drop=True)
        )

    # -------------------
    # Liquidity: ADV (dollar & shares) + Roll spread estimator
    # -------------------
    for aw in cfg["features"].get("adv_windows", [20]):
        features[f"adv{aw}_dollar"] = (
            (features["adj_close"] * features["volume"])
                .groupby(features["ticker"]).rolling(aw).mean()
                .reset_index(level=0, drop=True)
        )
        features[f"adv{aw}_share"] = (
            features.groupby("ticker")["volume"].rolling(aw).mean()
                .reset_index(level=0, drop=True)
        )

    rsw = cfg["features"].get("roll_spread_window", 20)
    def _roll_spread(series, w):
        r = series
        cov = r.rolling(w).cov(r.shift(1))
        # Roll spread ≈ 2 * sqrt(-cov), clipped at 0
        return 2.0 * np.sqrt(np.clip(-cov, 0, None))

    features[f"roll_spread{rsw}"] = (
        features.groupby("ticker")["log_ret"]
                .apply(lambda s: _roll_spread(s, rsw))
                .reset_index(level=0, drop=True)
    )

    # -------------------
    # Fundamentals merge (Value/Quality), with safety lag + ffill to trading calendar
    # -------------------
    cal = None
    if Path(f["calendar_csv"]).exists():
        cal = pd.read_csv(f["calendar_csv"])
        cal = _to_datetime(cal, "date")

    funda_parq = Path(f["fundamentals_parquet"])
    if funda_parq.exists() and cal is not None:
        funda = pd.read_parquet(funda_parq)
        funda["ticker"] = funda["ticker"].astype(str).str.upper()
        funda = _to_datetime(funda, "date")
        shift_days = int(cfg.get("fundamentals", {}).get("shift_days", 1))
        funda = _shift_by_days(funda, ["ticker"], "date", shift_days)
        funda = _ffill_panel_to_calendar(funda, ["ticker"], "date", cal["date"].unique())
        features = features.merge(funda, on=["date","ticker"], how="left")

        for col in cfg.get("fundamentals", {}).get("rank_columns", []):
            if col in features.columns:
                features[f"rank_{col}_pct"] = features.groupby("date")[col].rank(pct=True)

    # -------------------
    # Alternative data: News sentiment & Google Trends (optional), lag + ffill
    # -------------------
    news_parq = Path(f["news_parquet"])
    if news_parq.exists() and cal is not None:
        news = pd.read_parquet(news_parq)
        news["ticker"] = news["ticker"].astype(str).str.upper()
        news = _to_datetime(news, "date")
        nshift = int(cfg.get("alt_data", {}).get("news_shift_days", 0))
        news = _shift_by_days(news, ["ticker"], "date", nshift)
        news = _ffill_panel_to_calendar(news, ["ticker"], "date", cal["date"].unique())
        features = features.merge(news, on=["date","ticker"], how="left")

    tr_parq = Path(f["trends_parquet"])
    if tr_parq.exists() and cal is not None:
        tr = pd.read_parquet(tr_parq)
        tr["ticker"] = tr["ticker"].astype(str).str.upper()
        tr = _to_datetime(tr, "date")
        tshift = int(cfg.get("alt_data", {}).get("trends_shift_days", 0))
        tr = _shift_by_days(tr, ["ticker"], "date", tshift)
        tr = _ffill_panel_to_calendar(tr, ["ticker"], "date", cal["date"].unique())
        features = features.merge(tr, on=["date","ticker"], how="left")

    # -------------------
    # Winsorize selected columns per date (reduce outliers)
    # -------------------
    wz = cfg.get("winsorize", {"p_low": 0.01, "p_high": 0.99})
    wlow, whigh = wz.get("p_low", 0.01), wz.get("p_high", 0.99)
    to_winsor = []
    to_winsor += [f"mom_{w}" for w in w_moms if f"mom_{w}" in features.columns]
    to_winsor += [c for c in ["vol_20","idio_vol_60","gap","high_low_range20","vol_z20","dollar_vol"] if c in features.columns]
    to_winsor += [c for c in features.columns if c.startswith("vol_") and c not in {"vol_20"}]
    to_winsor += [c for c in features.columns if c.startswith("skew_") or c.startswith("kurt_")]
    to_winsor += [c for c in features.columns if c.startswith("adv") or c.startswith("roll_spread")]
    to_winsor += [c for c in ["pe","ev_ebitda","pb","roe","roa","gross_margin","op_margin","net_margin"] if c in features.columns]
    if to_winsor:
        features = _winsorize_per_date(features, list(set(to_winsor)), wlow, whigh)

    # -------------------
    # 🔒 Sanitize non-finites (∞ → NaN)  # NEW
    # -------------------
    num_cols_all = features.select_dtypes(include=[np.number]).columns
    features[num_cols_all] = features[num_cols_all].replace([np.inf, -np.inf], np.nan)  # NEW

    # -------------------
    # Final select & save (lenient NA policy; impute later in model)
    # -------------------
    core_cols = [
        "date","ticker","ret_1","vol_20","vol_z20","gap","high_low_range20",
        "beta_rolling","idio_vol_60",
        *[f"mom_{w}" for w in w_moms if f"mom_{w}" in features.columns],
    ]

    # keep everything numeric (fundamentals/regimes/sentiment) in addition to the core
    num_new = [c for c in features.select_dtypes(include=[np.number]).columns
               if c not in {"y_next_1d","target_1d"}]
    keep_cols = sorted(set(core_cols + num_new))

    feat_df = features[keep_cols].copy()

<<<<<<< Updated upstream
    # Require only essential core technicals; allow NaNs elsewhere
    required = [c for c in ["ret_1","vol_20","beta_rolling"] if c in feat_df.columns]
    if required:
        feat_df = feat_df.dropna(subset=required)

    # Light clipping to avoid huge spikes in tail values
    for c in feat_df.select_dtypes(include=[np.number]).columns:
        q1 = feat_df[c].quantile(0.001)
        q9 = feat_df[c].quantile(0.999)
        feat_df[c] = feat_df[c].clip(q1, q9)

    out_df = feat_df.sort_values(["ticker","date"]).reset_index(drop=True)
=======
    # higher moments, ADV, roll spread, fundamentals (levels + ranks), alt-data
    keep_cols += [c for c in [
        # higher moments
        f"skew_{cfg['features']['vol_window']}",
        f"kurt_{cfg['features']['vol_window']}",
        # ADV windows
        *[f"adv{w}_dollar" for w in cfg['features'].get('adv_windows', [])],
        *[f"adv{w}_share"  for w in cfg['features'].get('adv_windows', [])],
        # Roll spread
        f"roll_spread{cfg['features'].get('roll_spread_window', 20)}",
        # Fundamentals (levels)
        "pe","ev_ebitda","pb","roe","roa","gross_margin","op_margin","net_margin",
        # Fundamental ranks
        *[f"rank_{c}_pct" for c in cfg.get("fundamentals", {}).get("rank_columns", [])],
        # Alt data
        "sentiment_mean","sentiment_std","article_count","trends_score",
    ] if c in features.columns]

    keep_cols = [c for c in keep_cols if c in features.columns]
    out_df = features[keep_cols].dropna().sort_values(["ticker","date"]).reset_index(drop=True)
>>>>>>> Stashed changes

    out_path = Path(f["features_parquet"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out_path, index=False)
    print(f"✅ Features written → {out_path} ({len(out_df):,} rows)")

<<<<<<< Updated upstream




#creates the data that the ML model will learn from. Calculates the trade outcomes for each day of each trade being made
#then places this data into a parquet. 
=======
# -------------------------
# Labels
# -------------------------
>>>>>>> Stashed changes
def cmd_make_labels(_args):
    cfg = load_cfg()
    f = cfg["files"]
    prices = pd.read_parquet(f["prices_raw_parquet"])
    prices = _sort(prices)
<<<<<<< Updated upstream

    # Decide at close (t), enter next open (t+1) → label is next 10-day open→close return
    # y = (close_{t+10} - open_{t+10}) / open_{t+10}
    horizon = 10
    prices[f"open_t{horizon}"]  = prices.groupby("ticker")["open"].shift(-horizon)
    prices[f"close_t{horizon}"] = prices.groupby("ticker")["close"].shift(-horizon)
    y = (prices[f"close_t{horizon}"] - prices[f"open_t{horizon}"]) / prices[f"open_t{horizon}"]

=======
    # Decide at close (t), enter next open (t+1) → label is next-day open→close return
    prices["open_t1"]  = prices.groupby("ticker")["open"].shift(-1)
    prices["close_t1"] = prices.groupby("ticker")["close"].shift(-1)
    y = (prices["close_t1"] - prices["open_t1"]) / prices["open_t1"]
>>>>>>> Stashed changes
    labels = prices[["date","ticker"]].copy()
    labels["y_next_10d"] = y
    labels["target_10d"] = (labels["y_next_10d"] > 0).astype(int)

    labels = labels.dropna().sort_values(["ticker","date"]).reset_index(drop=True)

    out = Path(f["labels_parquet"])
    out.parent.mkdir(parents=True, exist_ok=True)
    labels.to_parquet(out, index=False)
    print(f"✅ Labels written → {out} ({len(labels):,} rows, horizon={horizon}d)")


# -------------------------
# yfinance fetcher
# -------------------------
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
                tickers.extend(dfu.iloc[:,0].astype(str).tolist())
        else:
            tickers.extend([line.strip() for line in p.read_text().splitlines() if line.strip()])

    tickers = sorted(set([t.strip().upper() for t in tickers if t.strip()]))
    if not tickers:
        raise ValueError("No tickers provided. Use --tickers ... or --universe-file FILE")

    start = args.start or "2018-01-01"
    end = args.end or None  # yfinance will default to today if None

    # Ensure SPY included if config uses SPY as market proxy
    if cfg.get("beta_market", "SPY").upper() == "SPY" and "SPY" not in tickers:
        tickers.append("SPY")

    print(f"⏬ Fetching {len(tickers)} tickers from yfinance "
          f"({tickers[:8]}{'...' if len(tickers)>8 else ''}) "
          f"start={start} end={end}")

    data = yf.download(" ".join(tickers), start=start, end=end, group_by="ticker", auto_adjust=False, progress=True)

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
        for t in tickers:
            if t not in data.columns.get_level_values(0):
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

    sub.add_parser("init", help="Create folders + default config")

    p1 = sub.add_parser("ingest-prices", help="Ingest prices CSV")
    p1.add_argument("--csv", required=True)

    p2 = sub.add_parser("ingest-macro", help="Ingest macro CSV")
    p2.add_argument("--csv", required=True)

    p_fun = sub.add_parser("ingest-fundamentals", help="Ingest fundamentals CSV")
    p_fun.add_argument("--csv", required=True)

    p_news = sub.add_parser("ingest-news", help="Ingest news sentiment CSV")
    p_news.add_argument("--csv", required=True)

    p_tr = sub.add_parser("ingest-trends", help="Ingest Google Trends CSV")
    p_tr.add_argument("--csv", required=True)

    sub.add_parser("build-calendar", help="Build trading calendar from prices")
    sub.add_parser("make-features", help="Compute features")
    sub.add_parser("make-labels", help="Compute next-day labels")

    p_fund_in = sub.add_parser("ingest-fundamentals", help="Ingest fundamentals CSV")
    p_fund_in.add_argument("--csv", required=True)

    p_fund_daily = sub.add_parser("make-fundamentals-daily", help="Forward-fill fundamentals to daily + engineer features")

    p_sent_in = sub.add_parser("ingest-sentiment", help="Ingest per-ticker sentiment CSV")
    p_sent_in.add_argument("--csv", required=True)

    p_reg = sub.add_parser("make-regimes", help="Build macro/breadth regime features")


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
    elif args.cmd == "ingest-fundamentals":
        cmd_ingest_fundamentals(args)
    elif args.cmd == "ingest-news":
        cmd_ingest_news(args)
    elif args.cmd == "ingest-trends":
        cmd_ingest_trends(args)
    elif args.cmd == "build-calendar":
        cmd_build_calendar(args)
    elif args.cmd == "make-features":
        cmd_make_features(args)
    elif args.cmd == "make-labels":
        cmd_make_labels(args)
    elif args.cmd == "fetch-yfinance":
        cmd_fetch_yfinance(args)
<<<<<<< Updated upstream
    elif args.cmd == "ingest-fundamentals":
        cmd_ingest_fundamentals(args)
    elif args.cmd == "make-fundamentals-daily":
        cmd_make_fundamentals_daily(args)
    elif args.cmd == "ingest-sentiment":
        cmd_ingest_sentiment(args)
    elif args.cmd == "make-regimes":
        cmd_make_regimes(args)
    elif args.cmd == "ingest-fundamentals":
        cmd_ingest_fundamentals(args)


=======
>>>>>>> Stashed changes
    else:
        parser.print_help()

if __name__ == "__main__":
    pd.set_option("display.width", 200)
    main()
