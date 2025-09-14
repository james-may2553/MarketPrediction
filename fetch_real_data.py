#!/usr/bin/env python3
"""
Fetch data for the pipeline using yfinance (+ FRED for macro).

Outputs under data/raw/:
  - fundamentals_quarterly.csv  (quarterly wide per ticker)
  - sample_macro.csv            (daily long table: date,series,value with US10Y, US2Y, VIX)

Usage:
  python fetch_real_data.py --sector-map data/raw/sector_map.csv --start 2019-01-01 --end 2025-12-31
  # or
  python fetch_real_data.py --tickers AAPL MSFT NVDA --start 2019-01-01 --end 2025-12-31

Notes:
  - yfinance fundamentals coverage can be patchy; we compute ratios when fields exist.
  - Macro comes from FRED (stable & well-documented).
"""

import argparse
import os
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yfinance as yf

# ---------- Paths / keys ----------
OUT_DIR = Path("data/raw")
OUT_DIR.mkdir(parents=True, exist_ok=True)

FRED_KEY = os.getenv("FRED_API_KEY")
FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"

# ---------- Helpers ----------
TICKER_RE = re.compile(r"^[A-Z][A-Z0-9.\-]{0,9}$")

def to_dt(x): return pd.to_datetime(x, errors="coerce")

def load_tickers(args) -> list[str]:
    if args.tickers:
        raw = [t.strip().upper() for t in args.tickers]
    elif args.sector_map:
        df = pd.read_csv(args.sector_map)
        col = "ticker" if "ticker" in df.columns else df.columns[0]
        raw = [str(x).strip().upper() for x in df[col].dropna().tolist()]
    else:
        raise SystemExit("Provide --tickers ... or --sector-map")

    # drop comments / headers
    tickers = [t for t in raw if t and not t.startswith("#") and TICKER_RE.match(t)]
    # de-dupe, keep order
    seen, clean = set(), []
    for t in tickers:
        if t not in seen:
            seen.add(t); clean.append(t)
    if args.max_tickers and len(clean) > args.max_tickers:
        clean = clean[:args.max_tickers]
    if not clean:
        raise SystemExit("No valid tickers after cleaning.")
    return clean

# ---------- Macro (FRED) ----------
def fetch_fred_series(series_id: str, start: str, end: str) -> pd.DataFrame:
    if not FRED_KEY:
        raise SystemExit("Missing FRED_API_KEY env var.")
    r = requests.get(FRED_BASE, params={
        "series_id": series_id, "api_key": FRED_KEY, "file_type": "json",
        "observation_start": start, "observation_end": end
    }, timeout=30)
    r.raise_for_status()
    obs = r.json().get("observations", [])
    df = pd.DataFrame(obs)[["date","value"]] if obs else pd.DataFrame(columns=["date","value"])
    df["date"] = to_dt(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df

def build_macro_csv(start: str, end: str) -> Path:
    print("==> Fetching macro from FRED (US10Y, US2Y, VIX)...")
    ten = fetch_fred_series("DGS10", start, end).assign(series="US10Y")
    two = fetch_fred_series("DGS2",  start, end).assign(series="US2Y")
    vix = fetch_fred_series("VIXCLS", start, end).assign(series="VIX")
    macro = pd.concat([ten, two, vix], ignore_index=True).dropna(subset=["date"]).sort_values("date")
    out = OUT_DIR / "sample_macro.csv"
    macro.to_csv(out, index=False)
    print(f"✅ Macro → {out} ({len(macro):,} rows)")
    return out

# ---------- Prices ----------
def load_prices_parquet_or_yf(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """Use existing prices.parquet if present; else fetch via yfinance."""
    pp = Path("data/raw/prices.parquet")
    if pp.exists():
        px = pd.read_parquet(pp)
        # normalize schema
        if "date" not in px.columns or "ticker" not in px.columns:
            raise SystemExit("prices.parquet must have at least 'date' and 'ticker' columns.")

        px["date"] = to_dt(px["date"])
        px["ticker"] = px["ticker"].astype(str).str.upper()   # <-- fix here

        # keep only requested tickers and date range
        px = px[px["ticker"].isin(tickers)]
        px = px[(px["date"] >= to_dt(start)) & (px["date"] <= to_dt(end))]

        need = {"date","ticker","open","high","low","close","adj_close","volume"}
        miss = need - set(px.columns)
        if miss:
            raise SystemExit(f"prices.parquet missing columns: {miss}")
        return px

    print("ℹ️  data/raw/prices.parquet not found; fetching prices via yfinance...")
    data = yf.download(" ".join(tickers), start=start, end=end, auto_adjust=False, progress=False, group_by="ticker")
    recs = []
    if len(tickers) == 1:
        t = tickers[0]
        dft = data.reset_index().rename(columns={
            "Date":"date","Open":"open","High":"high","Low":"low","Close":"close","Adj Close":"adj_close","Volume":"volume"
        })
        dft["ticker"] = t; recs.append(dft)
    else:
        for t in tickers:
            if t not in data.columns.get_level_values(0): continue
            dft = data[t].reset_index().rename(columns={
                "Date":"date","Open":"open","High":"high","Low":"low","Close":"close","Adj Close":"adj_close","Volume":"volume"
            })
            dft["ticker"] = t; recs.append(dft)
    px = pd.concat(recs, ignore_index=True) if recs else pd.DataFrame()
    if px.empty:
        raise SystemExit("yfinance returned no prices (check tickers/dates).")
    px["date"] = to_dt(px["date"])
    return px

# ---------- Fundamentals (yfinance) ----------
FIN_MAP_INCOME = {
    "Total Revenue": "totalRevenue",
    "Gross Profit": "grossProfit",
    "Operating Income": "operatingIncome",
    "Net Income": "netIncome",
}
FIN_MAP_BAL = {
    "Total Stockholder Equity": "totalShareholderEquity",
    "Total Assets": "totalAssets",
    "Total Liabilities Net Minority Interest": "totalLiabilities",
    "Ordinary Shares Number": "commonStockSharesOutstanding",  # often exists
}

def yf_quarterly_table(df: pd.DataFrame, rename_map: dict) -> pd.DataFrame:
    if df is None or df.empty: return pd.DataFrame()
    t = df.copy()
    t.index = t.index.astype(str)   # rows = items
    t = t.T                         # rows = dates, cols = items
    t.index.name = "date"
    t = t.reset_index()
    keep = {src: dst for src, dst in rename_map.items() if src in t.columns}
    if not keep: return pd.DataFrame()
    t = t[["date", *keep.keys()]].rename(columns=keep)
    t["date"] = to_dt(t["date"])
    return t

def fetch_yf_quarterlies(ticker: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    tk = yf.Ticker(ticker)
    fin = tk.quarterly_financials
    bal = tk.quarterly_balance_sheet
    inc = yf_quarterly_table(fin, FIN_MAP_INCOME)
    bal = yf_quarterly_table(bal, FIN_MAP_BAL)

    # Shares fallback from fast_info/info if column missing
    shares_static = None
    try: shares_static = getattr(tk, "fast_info", {}).get("shares_outstanding", None)
    except Exception: pass
    if shares_static is None:
        try: shares_static = tk.info.get("sharesOutstanding", None)
        except Exception: pass

    if not bal.empty and "commonStockSharesOutstanding" not in bal.columns and shares_static is not None:
        bal["commonStockSharesOutstanding"] = float(shares_static)

    if not inc.empty: inc["ticker"] = ticker
    if not bal.empty: bal["ticker"] = ticker
    return inc, bal

def compute_ratios(inc: pd.DataFrame, bal: pd.DataFrame, prices: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if inc.empty or bal.empty:
        return pd.DataFrame()
    q = inc.merge(bal, on=["ticker","date"], how="outer").dropna(subset=["date"]).sort_values("date")
    px_t = prices[prices["ticker"] == ticker].rename(columns={"date":"date_px"}).sort_values("date_px")

    # get quarter price on/after fiscal date; fallback to before
    q = pd.merge_asof(q.rename(columns={"date":"date_q"}).sort_values("date_q"),
                      px_t.sort_values("date_px"),
                      left_on="date_q", right_on="date_px",
                      by="ticker", direction="forward")
    if q["close"].isna().any():
        q_missing = q[q["close"].isna()].drop(columns=["close","adj_close"])
        q_back = pd.merge_asof(q_missing, px_t.sort_values("date_px"),
                               left_on="date_q", right_on="date_px",
                               by="ticker", direction="backward")
        q.loc[q["close"].isna(), ["close","adj_close","date_px"]] = q_back[["close","adj_close","date_px"]].values

    q["shares"] = q.get("commonStockSharesOutstanding", np.nan)
    # If still NA, try median of any available
    if q["shares"].isna().all() and "commonStockSharesOutstanding" in bal.columns:
        q["shares"] = q["shares"].fillna(bal["commonStockSharesOutstanding"].median())

    eps_q   = (q.get("netIncome") / q["shares"]).replace({0: np.nan}) if "netIncome" in q.columns else np.nan
    sales_q = (q.get("totalRevenue") / q["shares"]).replace({0: np.nan}) if "totalRevenue" in q.columns else np.nan
    book_q  = (q.get("totalShareholderEquity") / q["shares"]).replace({0: np.nan}) if "totalShareholderEquity" in q.columns else np.nan

    q["pe"] = (q["close"] / eps_q).replace([np.inf, -np.inf], np.nan) if isinstance(eps_q, pd.Series) else np.nan
    q["ps"] = (q["close"] / sales_q).replace([np.inf, -np.inf], np.nan) if isinstance(sales_q, pd.Series) else np.nan
    q["pb"] = (q["close"] / book_q).replace([np.inf, -np.inf], np.nan) if isinstance(book_q, pd.Series) else np.nan

    if "grossProfit" in q.columns and "totalRevenue" in q.columns:
        q["gross_margin"] = (q["grossProfit"] / q["totalRevenue"]).replace([np.inf,-np.inf], np.nan)
    else:
        q["gross_margin"] = np.nan
    if "operatingIncome" in q.columns and "totalRevenue" in q.columns:
        q["oper_margin"] = (q["operatingIncome"] / q["totalRevenue"]).replace([np.inf,-np.inf], np.nan)
    else:
        q["oper_margin"] = np.nan

    if "netIncome" in q.columns and "totalShareholderEquity" in q.columns:
        q["roe"] = (q["netIncome"] / q["totalShareholderEquity"]).replace([np.inf,-np.inf], np.nan)
    else:
        q["roe"] = np.nan
    if "netIncome" in q.columns and "totalAssets" in q.columns:
        q["roa"] = (q["netIncome"] / q["totalAssets"]).replace([np.inf,-np.inf], np.nan)
    else:
        q["roa"] = np.nan

    if "totalLiabilities" in q.columns and "totalShareholderEquity" in q.columns:
        q["debt_to_equity"] = (q["totalLiabilities"] / q["totalShareholderEquity"]).replace([np.inf,-np.inf], np.nan)
    else:
        q["debt_to_equity"] = np.nan

    q["market_cap"] = q["close"] * q["shares"]
    out = q[["date_q","ticker","pe","pb","ps","roe","roa","gross_margin","oper_margin","market_cap","shares","debt_to_equity"]]
    out = out.rename(columns={"date_q":"date"}).dropna(subset=["date"])
    return out.sort_values("date")

def build_fundamentals_csv(tickers: list[str], start: str, end: str) -> Path:
    print(f"==> Fetching fundamentals via yfinance for {len(tickers)} tickers ...")
    prices = load_prices_parquet_or_yf(tickers, start, end)[["date","ticker","close","adj_close"]]
    prices = prices.sort_values(["ticker","date"])
    rows = []
    for i, t in enumerate(tickers, 1):
        try:
            inc, bal = fetch_yf_quarterlies(t)
            if inc.empty or bal.empty:
                print(f"  · {t}: missing quarterly tables; skipping")
                continue
            rec = compute_ratios(inc.assign(ticker=t), bal.assign(ticker=t), prices, t)
            if not rec.empty:
                rows.append(rec)
            else:
                print(f"  · {t}: could not compute ratios; skipping")
        except Exception as e:
            print(f"  · {t}: ERROR {e}")
        # gentle throttle
        time.sleep(0.5)

    if not rows:
        raise SystemExit("No fundamentals could be constructed from yfinance.")
    allq = pd.concat(rows, ignore_index=True).sort_values(["ticker","date"])
    out = OUT_DIR / "fundamentals_quarterly.csv"
    allq.to_csv(out, index=False)
    print(f"✅ Fundamentals → {out} ({len(allq):,} rows, {allq['ticker'].nunique()} tickers)")
    return out

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", nargs="*", help="Tickers (e.g., AAPL MSFT NVDA)")
    ap.add_argument("--sector-map", help="CSV with 'ticker' column (headers/comments allowed)")
    ap.add_argument("--start", default="2019-01-01")
    ap.add_argument("--end", default="2025-12-31")
    ap.add_argument("--max-tickers", type=int, default=30)
    args = ap.parse_args()

    tickers = load_tickers(args)
    print(f"Tickers ({len(tickers)}): {tickers[:12]}{'...' if len(tickers)>12 else ''}")

    fund_csv  = build_fundamentals_csv(tickers, args.start, args.end)
    macro_csv = build_macro_csv(args.start, args.end)

    print("\nAll done:")
    print(f"  Fundamentals → {fund_csv}")
    print(f"  Macro        → {macro_csv}")

if __name__ == "__main__":
    main()
