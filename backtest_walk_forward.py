import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# =====================
# Config (tweak freely)
# =====================
FEATURES_PATH = "data/features/eq_features.parquet"
LABELS_PATH   = "data/features/labels.parquet"
SECTOR_MAP_CSV = "data/raw/sector_map.csv"   # used for sector caps

# Walk-forward settings
WARMUP_DAYS        = 504      # require ~2Y before first test
TRAIN_WINDOW_DAYS  = 504      # rolling window length
EMBARGO_DAYS       = 10       # bleed protection
REBALANCE_EVERY    = 1        # refit every N days (weekly-ish)

# Trading rules
THRESHOLD          = 0.5     # higher -> fewer, higher conviction longs
MIN_STOCKS         = 5        # skip day if too few candidates
MAX_POSITIONS      = 80       # cap portfolio breadth
SECTOR_CAP_PCT     = 0.35     # max fraction of names from any one sector
VOL_SCALE_WEIGHTS  = False     # 1/(vol_20+eps) scaling of weights
EXCLUDE_TICKERS    = {"SPY","QQQ","IWM","DIA","XLK","XLF","XLY","XLC","XLI","XLE","XLP","XLV","XLU","XLB","XLRE"}  # avoid ETFs

# Cost model (simple)
COST_BPS           = 2.0      # per day, approximate via turnover

RANDOM_STATE       = 0

# =====================
# Metrics helpers
# =====================
def sharpe(returns, ann_factor=252):
    r = np.asarray(returns, dtype=float)
    if r.size == 0: return np.nan
    mu, sigma = r.mean(), r.std()
    return np.sqrt(ann_factor) * (mu / sigma) if sigma > 0 else np.nan

def max_drawdown(equity_curve):
    ec = np.asarray(equity_curve, dtype=float)
    peaks = np.maximum.accumulate(ec)
    dd = (ec - peaks) / peaks
    return dd.min() if len(dd) else np.nan

# =====================
# Data prep
# =====================
def load_sector_map(path):
    p = Path(path)
    if p.exists() and p.is_file():
        smap = pd.read_csv(p)
        if {"ticker","sector"}.issubset(smap.columns):
            smap["ticker"] = smap["ticker"].astype(str).str.upper()
            return smap[["ticker","sector"]]
    return None

def prepare_data():
    X = pd.read_parquet(FEATURES_PATH)
    y = pd.read_parquet(LABELS_PATH)

    # Merge features & labels
    df = X.merge(y, on=["date","ticker"], how="inner")
    df["date"] = pd.to_datetime(df["date"])
    df["ticker"] = df["ticker"].astype(str).str.upper()
    df = df.sort_values(["date","ticker"]).reset_index(drop=True)

    # Attach sector info (for caps) if available
    smap = load_sector_map(SECTOR_MAP_CSV)
    if smap is not None:
        df = df.merge(smap, on="ticker", how="left")
    else:
        df["sector"] = "UNKNOWN"

    # Exclude ETFs/benchmarks from trading universe (keep them in df in case used as features, but we won’t trade them)
    df["is_excluded"] = df["ticker"].isin(EXCLUDE_TICKERS)

    # Feature columns: all numeric except metadata/labels
    meta = {"date","ticker","sector","is_excluded","y_next_1d","target_1d"}
    numerics = df.select_dtypes(include=[np.number]).columns
    feature_cols = [c for c in numerics if c not in {"y_next_1d","target_1d"}]

    return df, feature_cols

# =====================
# Model
# =====================
def make_model():
    return Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("lr", LogisticRegression(
            solver="saga",
            max_iter=500,
            C=1.0,
            n_jobs=-1,
            random_state=RANDOM_STATE
        ))
    ])

# =====================
# Portfolio helpers
# =====================
def apply_sector_cap(candidates, sector_cap_pct=0.25, max_positions=50):
    """
    candidates: DataFrame with columns [ticker, sector, prob_up] (sorted by prob_up desc)
    Returns the tickers that pass sector caps up to max_positions.
    """
    if candidates.empty:
        return []

    total_cap = max(1, max_positions)
    per_sector_cap = max(1, int(np.floor(sector_cap_pct * total_cap)))

    kept = []
    sector_counts = {}

    for _, row in candidates.iterrows():
        sec = row.get("sector", "UNKNOWN")
        if sector_counts.get(sec, 0) < per_sector_cap:
            kept.append(row["ticker"])
            sector_counts[sec] = sector_counts.get(sec, 0) + 1
            if len(kept) >= total_cap:
                break
    return kept

def vol_scale_weights(df_sub):
    """
    df_sub: rows for selected tickers on a single day, must have vol_20 (or any vol column).
    Return normalized weights (sum to 1).
    """
    eps = 1e-6
    if "vol_20" not in df_sub.columns or df_sub["vol_20"].isna().all():
        # fallback: equal weight
        w = np.ones(len(df_sub), dtype=float)
    else:
        inv = 1.0 / (df_sub["vol_20"].fillna(df_sub["vol_20"].median()) + eps)
        w = inv.to_numpy()
    w = w / w.sum() if w.sum() > 0 else np.ones_like(w) / len(w)
    return w

# =====================
# Walk-forward backtest
# =====================
def walk_forward_backtest(df, feature_cols):
    dates = pd.Index(sorted(df["date"].unique()))
    if len(dates) <= WARMUP_DAYS + 1:
        raise ValueError("Not enough history for the chosen WARMUP_DAYS.")

    daily = []
    model = None

    for t_idx in range(WARMUP_DAYS + EMBARGO_DAYS, len(dates) - 1):
        # Train window indices
        train_end_idx   = t_idx - EMBARGO_DAYS
        train_start_idx = max(0, train_end_idx - TRAIN_WINDOW_DAYS)
        train_dates = dates[train_start_idx:train_end_idx]
        test_date   = dates[t_idx]
        next_date   = dates[t_idx + 1]

        train_df = df[df["date"].isin(train_dates)]
        test_df  = df[df["date"] == test_date]

        if train_df.empty or test_df["ticker"].nunique() < MIN_STOCKS:
            continue

        # Train set
        X_train = train_df[feature_cols].astype(float).values
        y_train = train_df["target_1d"].astype(int).values

        # Refit only every N days (and always the first usable day)
        need_refit = (model is None) or ((t_idx - (WARMUP_DAYS + EMBARGO_DAYS)) % REBALANCE_EVERY == 0)
        if need_refit:
            model = make_model()
            model.fit(X_train, y_train)

        # Score today's cross-section
        X_test = test_df[feature_cols].astype(float).values
        proba  = model.predict_proba(X_test)[:, 1]
        test_df = test_df.copy()
        test_df["prob_up"] = proba

        # Universe filter: threshold + exclude ETFs + min stocks
        pool = test_df.loc[(test_df["prob_up"] >= THRESHOLD) & (~test_df["is_excluded"])].copy()
        # Sort by conviction
        pool = pool.sort_values("prob_up", ascending=False)

        if pool.empty:
            # no trades
            daily.append({
                "test_date": test_date, "n_longs": 0,
                "gross_ret": 0.0, "cost": 0.0, "net_ret": 0.0,
                "turnover": 0.0, "hit_rate": np.nan, "accuracy": np.nan,
                "signals": {}
            })
            continue

        # Apply sector caps & max positions
        keep_tickers = apply_sector_cap(
            candidates=pool[["ticker","sector","prob_up"]],
            sector_cap_pct=SECTOR_CAP_PCT,
            max_positions=MAX_POSITIONS
        )
        pool = pool[pool["ticker"].isin(keep_tickers)]

        n_longs = len(pool)
        if n_longs < MIN_STOCKS:
            daily.append({
                "test_date": test_date, "n_longs": 0,
                "gross_ret": 0.0, "cost": 0.0, "net_ret": 0.0,
                "turnover": 0.0, "hit_rate": np.nan, "accuracy": np.nan,
                "signals": {}
            })
            continue

        # Weights
        if VOL_SCALE_WEIGHTS:
            w = vol_scale_weights(pool)  # sum to 1
        else:
            w = np.ones(n_longs, dtype=float) / n_longs
        pool["weight"] = w

        # Realize next-day returns (labels live on the decision row)
        pnl = pool[["ticker","weight"]].merge(
            df.loc[df["date"] == test_date, ["ticker","y_next_1d"]],
            on="ticker", how="left"
        ).dropna(subset=["y_next_1d"])

        pnl["contrib"] = pnl["weight"] * pnl["y_next_1d"]
        gross_ret = pnl["contrib"].sum()

        # --- Turnover & costs (compare today vs yesterday's signals) ---
        if len(daily) > 0 and daily[-1]["test_date"] == dates[t_idx - 1]:
            prev_sig = daily[-1]["signals"]  # dict: ticker -> weight
        else:
            prev_sig = {}

        curr_sig = dict(zip(pool["ticker"], pool["weight"]))
        universe = set(curr_sig.keys()) | set(prev_sig.keys())
        # L1 weight change as a proxy for turnover
        turnover = sum(abs(curr_sig.get(tk, 0.0) - prev_sig.get(tk, 0.0)) for tk in universe)

        cost = (COST_BPS / 10000.0) * turnover
        net_ret = gross_ret - cost

        # Hit-rate among traded names (positive y_next_1d)
        hit_rate = (pnl["y_next_1d"] > 0).mean() if len(pnl) else np.nan
        day_acc  = hit_rate

        daily.append({
            "test_date": test_date,
            "n_longs": n_longs,
            "gross_ret": gross_ret,
            "cost": cost,
            "net_ret": net_ret,
            "turnover": turnover,
            "hit_rate": hit_rate,
            "accuracy": day_acc,
            "signals": curr_sig,  # keep for next day's turnover calc
        })

    # Build results
    res = pd.DataFrame(daily)
    if res.empty:
        raise RuntimeError("No backtest rows produced—tune WARMUP/EMBARGO/THRESHOLD/MAX_POSITIONS.")

    res = res.drop(columns=["signals"])
    res = res.sort_values("test_date").reset_index(drop=True)
    res["equity"] = (1.0 + res["net_ret"]).cumprod()

    # Metrics
    metrics = {
        "days": len(res),
        "mean_daily_ret_%": 100 * res["net_ret"].mean(),
        "vol_daily_%": 100 * res["net_ret"].std(),
        "sharpe": sharpe(res["net_ret"]),
        "max_drawdown_%": 100 * max_drawdown(res["equity"]),
        "avg_turnover_%": 100 * res["turnover"].mean(),
        "avg_n_longs": res["n_longs"].mean(),
        "avg_hit_rate_%": 100 * res["hit_rate"].mean(skipna=True),
        "median_hit_rate_%": 100 * res["hit_rate"].median(skipna=True),
    }

    # Yearly breakdown
    res["year"] = pd.DatetimeIndex(res["test_date"]).year
    by_year = res.groupby("year").agg(
        days=("net_ret", "size"),
        ret_pct=("net_ret", lambda x: 100 * (np.prod(1 + x) - 1)),
        sharpe=("net_ret", sharpe),
        hit_pct=("hit_rate", lambda x: 100 * np.nanmean(x)),
        avg_turnover_pct=("turnover", lambda x: 100 * np.nanmean(x)),
    ).reset_index()

    return res, metrics, by_year

# =====================
# Main
# =====================
if __name__ == "__main__":
    df, feature_cols = prepare_data()
    res, metrics, by_year = walk_forward_backtest(df, feature_cols)

    print("\n=== Walk-Forward Summary ===")
    for k, v in metrics.items():
        print(f"{k:>20}: {v:.4f}" if isinstance(v, (int, float, np.floating)) else f"{k:>20}: {v}")

    print("\n=== Yearly Breakdown ===")
    print(by_year.to_string(index=False))

    out_dir = Path("reports"); out_dir.mkdir(parents=True, exist_ok=True)
    res[["test_date","net_ret","equity","n_longs","turnover","hit_rate"]].to_csv(out_dir/"wf_daily.csv", index=False)
    by_year.to_csv(out_dir/"wf_yearly.csv", index=False)
    print("\nSaved daily results → reports/wf_daily.csv")
    print("Saved yearly results → reports/wf_yearly.csv")
