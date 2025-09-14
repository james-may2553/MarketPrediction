#!/usr/bin/env python3
"""
Gather_External.py — fetch Fundamentals (Alpha Vantage), News Sentiment (Alpha Vantage),
and Google Trends (pytrends) and save to the parquet paths defined in configs/default.json.

Env:
  ALPHAVANTAGE_API_KEY=...   # required for fundamentals + news

Usage:
  python Gather_External.py fundamentals --start 2018-01-01 [--end 2025-09-12]
  python Gather_External.py news         --start 2018-01-01 [--end 2025-09-12]
  python Gather_External.py trends       --start 2018-01-01 [--end 2025-09-12]
"""

import argparse
import json
import os
from pathlib import Path
from time import sleep
from datetime import datetime, date

import numpy as np
import pandas as pd
import requests

# Optional (install with: pip install pytrends)
try:
    from pytrends.request import TrendReq
except Exception:
    TrendReq = None

PROJ   = Path(".")
DATA   = PROJ / "data"
RAW    = DATA / "raw"
CONFIG = PROJ / "configs" / "default.json"

# ---------------- helpers ----------------
def load_cfg():
    if not CONFIG.exists():
        raise FileNotFoundError("Config not found. Run: python Data_Pipeline.py init")
    with open(CONFIG, "r") as f:
        return json.load(f)

def ensure_dirs():
    RAW.mkdir(parents=True, exist_ok=True)

def _to_datetime_col(df, col="date"):
    df[col] = pd.to_datetime(df[col], utc=False).dt.tz_localize(None)
    return df

def _date_str(d):
    if d is None:
        return None
    if isinstance(d, (datetime, date)):
        return d.strftime("%Y-%m-%d")
    return str(d)

def _load_universe_tickers(cfg):
    p = Path(cfg["files"]["prices_raw_parquet"])
    if not p.exists():
        raise FileNotFoundError(f"Prices parquet not found: {p}. Ingest or fetch prices first.")
    df = pd.read_parquet(p)
    tks = sorted(set(df["ticker"].astype(str).str.upper()))
    # Drop obvious index/ETF tickers from fundamentals/news if you want:
    # etfs = {"SPY","QQQ","IWM","DIA","XLK","XLF","XLY","XLC","XLI","XLE","XLP","XLV","XLU","XLB","XLRE"}
    # tks = [t for t in tks if t not in etfs]
    return tks

# --------------- FUNDAMENTALS (Alpha Vantage Overview) ---------------
def fetch_alpha_vantage_fundamentals(cfg, start, end, tickers, pause=12.0):
    """
    Uses Alpha Vantage 'OVERVIEW' endpoint (per ticker) to extract:
      pe (PERatio), pb (PriceToBookRatio), ev_ebitda (EVToEBITDA),
      roe (ReturnOnEquityTTM), roa (ReturnOnAssetsTTM),
      op_margin (OperatingMarginTTM), net_margin (ProfitMargin),
      gross_margin (GrossProfitTTM / RevenueTTM).
    Assigns 'date' = latest reported quarter in 'LatestQuarter' (or today if missing).

    Writes parquet to cfg['files']['fundamentals_parquet'] with rows (date,ticker,columns...).

    NOTE: This gives a TTM snapshot series keyed to reporting dates; your main pipeline
    forward-fills it to trading days safely.
    """
    api_key = os.getenv("ALPHAVANTAGE_API_KEY")
    if not api_key:
        raise RuntimeError("ALPHAVANTAGE_API_KEY not set in environment.")

    start = pd.to_datetime(_date_str(start)) if start else None
    end   = pd.to_datetime(_date_str(end))   if end   else None

    base = "https://www.alphavantage.co/query?function=OVERVIEW&symbol={tk}&apikey={key}"
    rows = []

    for i, tk in enumerate(tickers, 1):
        url = base.format(tk=tk, key=api_key)
        try:
            r = requests.get(url, timeout=30)
            if r.status_code != 200:
                print(f"[AV-Fund] {tk} HTTP {r.status_code}: {r.text[:120]}")
                sleep(pause)
                continue
            data = r.json()
            if not isinstance(data, dict) or len(data) == 0 or "Symbol" not in data:
                # Often AV returns a note for throttling; keep going with sleep
                msg = data.get("Note") if isinstance(data, dict) else None
                if msg:
                    print(f"[AV-Fund] {tk} throttled: {msg[:90]}")
                sleep(pause)
                continue

            # Date anchor: latest reported quarter or today
            dt = data.get("LatestQuarter") or data.get("MostRecentQuarter")
            d  = pd.to_datetime(dt) if dt else pd.to_datetime(date.today())

            if start and d < start:
                # If reporting date before start window, still include (rare),
                # but you can skip to reduce size.
                pass
            if end and d > end:
                # Also usually include; skip only if strictly enforcing end date.
                pass

            # Parse and coerce helpers
            def fnum(k):
                v = data.get(k, None)
                return pd.to_numeric(v, errors="coerce")

            pe          = fnum("PERatio")
            pb          = fnum("PriceToBookRatio")
            ev_ebitda   = fnum("EVToEBITDA")
            roe         = fnum("ReturnOnEquityTTM")
            roa         = fnum("ReturnOnAssetsTTM")
            op_margin   = fnum("OperatingMarginTTM")
            net_margin  = fnum("ProfitMargin")
            gp_ttm      = fnum("GrossProfitTTM")
            rev_ttm     = fnum("RevenueTTM")
            gross_margin = (gp_ttm / rev_ttm) if (rev_ttm and pd.notna(rev_ttm) and rev_ttm != 0) else np.nan

            rows.append({
                "date": d, "ticker": tk,
                "pe": pe, "pb": pb, "ev_ebitda": ev_ebitda,
                "roe": roe, "roa": roa,
                "gross_margin": gross_margin,
                "op_margin": op_margin, "net_margin": net_margin
            })
        except Exception as e:
            print(f"[AV-Fund] {tk} error: {e}")

        # progress + rate limit
        if i % 20 == 0:
            print(f"[AV-Fund] processed {i}/{len(tickers)}")
        sleep(pause)

    if not rows:
        raise RuntimeError("No fundamentals fetched from Alpha Vantage (check API key / limits).")

    df = pd.DataFrame(rows)
    df["ticker"] = df["ticker"].astype(str).str.upper()
    df = _to_datetime_col(df, "date")
    df = df.sort_values(["ticker","date"]).reset_index(drop=True)

    out = Path(cfg["files"]["fundamentals_parquet"])
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"✅ Fundamentals written → {out} ({len(df):,} rows, {df['ticker'].nunique()} tickers)")

# --------------- NEWS (Alpha Vantage NEWS_SENTIMENT) ---------------
def fetch_alpha_vantage_news(cfg, start, end, tickers, batch=10, pause=12.0):
    """
    Alpha Vantage NEWS_SENTIMENT endpoint.
    Aggregates to (date,ticker): sentiment_mean, sentiment_std, article_count.
    Writes parquet to cfg['files']['news_parquet'].
    """
    api_key = os.getenv("ALPHAVANTAGE_API_KEY")
    if not api_key:
        raise RuntimeError("ALPHAVANTAGE_API_KEY not set in environment.")

    start = pd.to_datetime(_date_str(start)) if start else pd.to_datetime("2018-01-01")
    end   = pd.to_datetime(_date_str(end))   if end   else None

    rows = []
    base = "https://www.alphavantage.co/query?function=NEWS_SENTIMENT"

    symbols = list(tickers)
    for i in range(0, len(symbols), batch):
        group = symbols[i:i+batch]
        tick_param = ",".join(group)
        url = f"{base}&tickers={tick_param}&apikey={api_key}&sort=LATEST"
        try:
            r = requests.get(url, timeout=45)
            if r.status_code != 200:
                print(f"[AV-News] HTTP {r.status_code}: {r.text[:120]}")
                sleep(pause)
                continue
            payload = r.json()
            feed = payload.get("feed", [])
            if not feed:
                note = payload.get("Note") or payload.get("Information")
                if note:
                    print(f"[AV-News] Throttle/info: {note[:120]}")
                else:
                    print(f"[AV-News] Empty feed for batch {group}")
                sleep(pause)
                continue

            for item in feed:
                ts = item.get("time_published")  # e.g. "20250115T123000"
                if not ts:
                    continue
                d = pd.to_datetime(ts[:8], format="%Y%m%d")
                if d < start or (end is not None and d > end):
                    continue
                overall = item.get("overall_sentiment_score", None)
                tick_sent = item.get("ticker_sentiment", [])
                # For each ticker in item, record a sentiment observation
                for tsent in tick_sent:
                    tk = (tsent.get("ticker") or "").upper()
                    if not tk:
                        continue
                    score = pd.to_numeric(tsent.get("ticker_sentiment_score"), errors="coerce")
                    if pd.isna(score):
                        score = pd.to_numeric(overall, errors="coerce")
                    rows.append({"date": d, "ticker": tk, "sent": score})
        except Exception as e:
            print(f"[AV-News] error: {e}")

        print(f"[AV-News] processed {min(i+batch, len(symbols))}/{len(symbols)} tickers...")
        sleep(pause)  # rate limit

    if not rows:
        raise RuntimeError("No news sentiment fetched from Alpha Vantage (rate limits?).")

    df = pd.DataFrame(rows)
    df["ticker"] = df["ticker"].astype(str).str.upper()
    df = df.dropna(subset=["ticker"])
    df = df.groupby(["date","ticker"])["sent"].agg(
        sentiment_mean="mean",
        sentiment_std="std",
        article_count="count"
    ).reset_index()

    out = Path(cfg["files"]["news_parquet"])
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"✅ News sentiment written → {out} ({len(df):,} rows, {df['ticker'].nunique()} tickers)")

# --------------- TRENDS (Google Trends via pytrends) ---------------
def fetch_google_trends(cfg, start, end, tickers, tz=0, pause=2.0):
    """
    Google Trends interest for each ticker keyword.
    Output columns: date,ticker,trends_score
    Writes parquet to cfg['files']['trends_parquet'].
    """
    if TrendReq is None:
        raise RuntimeError("pytrends is not installed. Run: pip install pytrends")

    start_s = _date_str(start) or "2018-01-01"
    end_s   = _date_str(end) if end else _date_str(date.today())

    pytrends = TrendReq(hl="en-US", tz=tz)
    rows = []

    for i, tk in enumerate(tickers, 1):
        try:
            kw = [tk]  # simple: ticker as keyword; you can map to company names if desired
            timeframe = f"{start_s} {end_s}"
            pytrends.build_payload(kw_list=kw, timeframe=timeframe, geo="")
            df = pytrends.interest_over_time()
            if df is None or df.empty:
                continue
            s = df[tk].rename("trends_score").to_frame()
            s = s.reset_index().rename(columns={"date": "date"})
            s["ticker"] = tk
            rows.append(s[["date","ticker","trends_score"]])
        except Exception as e:
            print(f"[Trends] {tk} error: {e}")
        if i % 25 == 0:
            print(f"[Trends] processed {i}/{len(tickers)} tickers...")
        sleep(pause)

    if not rows:
        raise RuntimeError("No trends data fetched (pytrends returned empty).")

    out_df = pd.concat(rows, ignore_index=True)
    out_df = _to_datetime_col(out_df, "date")
    out_df = out_df.sort_values(["ticker","date"]).reset_index(drop=True)

    out = Path(cfg["files"]["trends_parquet"])
    out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out, index=False)
    print(f"✅ Google Trends written → {out} ({len(out_df):,} rows, {out_df['ticker'].nunique()} tickers)")

# ---------------- main ----------------
def main():
    parser = argparse.ArgumentParser(description="Gather external datasets for the pipeline")
    sub = parser.add_subparsers(dest="cmd", required=True)

    for name in ["fundamentals","news","trends"]:
        p = sub.add_parser(name, help=f"Fetch {name}")
        p.add_argument("--start", required=False, help="YYYY-MM-DD (default 2018-01-01)")
        p.add_argument("--end",   required=False, help="YYYY-MM-DD (default today)")

    args = parser.parse_args()
    cfg = load_cfg()
    ensure_dirs()

    start = args.start or "2018-01-01"
    end   = args.end or None

    tickers = _load_universe_tickers(cfg)
    tickers = sorted(set([t for t in tickers if t and isinstance(t, str)]))

    if args.cmd == "fundamentals":
        fetch_alpha_vantage_fundamentals(cfg, start, end, tickers)
    elif args.cmd == "news":
        fetch_alpha_vantage_news(cfg, start, end, tickers)
    elif args.cmd == "trends":
        fetch_google_trends(cfg, start, end, tickers)
    else:
        parser.print_help()

if __name__ == "__main__":
    pd.set_option("display.width", 200)
    main()
