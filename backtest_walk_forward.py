import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# -----------------------
# Configs (tweak freely)
# -----------------------
FEATURES_PATH = "data/features/eq_features.parquet"
LABELS_PATH   = "data/features/labels.parquet"

# Walk-forward settings
WARMUP_DAYS   = 504     # ~2 years of trading days before first test date
EMBARGO_DAYS  = 10      # remove last X days from train to avoid bleed into test
THRESHOLD     = 0.50    # prob threshold for long/flat (try 0.55+ as a sanity check)
MIN_STOCKS    = 5       # require at least N names per day to trade

# Cost model
COST_BPS      = 2.0     # per side, approximated via turnover (e.g., 2 bps)
# -----------------------

def sharpe(returns, ann_factor=252):
    r = np.asarray(returns)
    if r.size == 0: return np.nan
    mu, sigma = r.mean(), r.std()
    return np.sqrt(ann_factor) * (mu / sigma) if sigma > 0 else np.nan

def max_drawdown(equity_curve):
    ec = np.asarray(equity_curve)
    peaks = np.maximum.accumulate(ec)
    dd = (ec - peaks) / peaks
    return dd.min()  # negative number

def prepare_data():
    X = pd.read_parquet(FEATURES_PATH)
    y = pd.read_parquet(LABELS_PATH)
    df = X.merge(y, on=["date", "ticker"], how="inner")

    # Sort + index
    df = df.sort_values(["date", "ticker"]).reset_index(drop=True)

    # Feature columns (everything except keys + labels)
    feats = [c for c in df.columns if c not in ["date", "ticker", "y_next_1d", "target_1d"]]
    return df, feats

def walk_forward_backtest(df, feature_cols):
    dates = pd.Index(sorted(df["date"].unique()))
    if len(dates) <= WARMUP_DAYS + 1:
        raise ValueError("Not enough history for the chosen WARMUP_DAYS.")

    # Storage for daily portfolio results
    daily = []

    # Model pipeline (fit fresh each step on training window ONLY)
    def make_model():
        return Pipeline([
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("lr", LogisticRegression(max_iter=2000, C=1.0, n_jobs=None, random_state=0))
        ])

    for t_idx in range(WARMUP_DAYS + EMBARGO_DAYS, len(dates) - 1):
        # Train window: up to (t_idx - EMBARGO_DAYS - 1)
        train_end_idx = t_idx - EMBARGO_DAYS
        train_dates = dates[:train_end_idx]          # strictly before embargo
        test_date   = dates[t_idx]                   # decide at t
        next_date   = dates[t_idx + 1]               # realize at t+1 (your label is y_next_1d)

        train_df = df[df["date"].isin(train_dates)]
        test_df  = df[df["date"] == test_date]       # cross-section for a single day

        if len(train_df) == 0 or len(test_df) < MIN_STOCKS:
            continue

        # Prepare X/y
        X_train = train_df[feature_cols].astype(float).values
        y_train = train_df["target_1d"].values

        X_test  = test_df[feature_cols].astype(float).values

        
        # Fit & predict
        model = make_model()
        model.fit(X_train, y_train)
        proba  = model.predict_proba(X_test)[:, 1]   # P(up)
        preds  = (proba >= THRESHOLD)                # boolean

        # Build positions for decision day (equal-weight longs)
        test_df = test_df.copy()
        test_df["prob_up"] = proba
        test_df["signal"]  = preds.astype(int)       # ensure 0/1

        n_longs = int(test_df["signal"].sum())
        trade_weight = 0.0 if n_longs < MIN_STOCKS else (1.0 / n_longs)

        # Join next-day realized return (label lives on the decision row)
        pnl_components = test_df[["ticker", "signal"]].merge(
            df.loc[df["date"] == test_date, ["ticker", "y_next_1d"]],
            on="ticker", how="left"
        ).dropna(subset=["y_next_1d"])               # robustness: ensure labels present

        pnl_components["contrib"] = trade_weight * pnl_components["signal"] * pnl_components["y_next_1d"]
        gross_ret = pnl_components["contrib"].sum()

        # --- turnover & costs ---
        if len(daily) > 0 and daily[-1]["test_date"] == dates[t_idx - 1]:
            prev_sig = daily[-1]["signals"]  # dict ticker->0/1
        else:
            prev_sig = {}

        curr_sig = dict(zip(test_df["ticker"], test_df["signal"]))
        universe = set(curr_sig.keys()) | set(prev_sig.keys())
        changed = sum(1 for tk in universe if curr_sig.get(tk, 0) != prev_sig.get(tk, 0))
        turnover = changed / max(1, len(universe))

        cost = (COST_BPS / 10000.0) * turnover
        net_ret = gross_ret - cost

        # --- accuracy / hit-rate on traded names (long/flat) ---
        if pnl_components["signal"].sum() > 0:
            hit_rate = (pnl_components.loc[pnl_components["signal"] == 1, "y_next_1d"] > 0).mean()
        else:
            hit_rate = np.nan
        day_acc = hit_rate


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

    # Build results DataFrame
    res = pd.DataFrame(daily)
    if res.empty:
        raise RuntimeError("No backtest rows produced—check your WARMUP_DAYS/EMBARGO/threshold settings.")

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

    # Yearly breakdown (hit-rate & return)
    res["year"] = pd.DatetimeIndex(res["test_date"]).year
    by_year = res.groupby("year").agg(
        days=("net_ret", "size"),
        ret_pct=("net_ret", lambda x: 100 * (np.prod(1 + x) - 1)),
        sharpe=("net_ret", sharpe),
        hit_pct=("hit_rate", lambda x: 100 * np.nanmean(x)),
        avg_turnover_pct=("turnover", lambda x: 100 * np.nanmean(x)),
    ).reset_index()

    return res, metrics, by_year

if __name__ == "__main__":
    df, feature_cols = prepare_data()
    res, metrics, by_year = walk_forward_backtest(df, feature_cols)

    print("\n=== Walk-Forward Summary ===")
    for k, v in metrics.items():
        print(f"{k:>20}: {v:.4f}" if isinstance(v, (int, float, np.floating)) else f"{k:>20}: {v}")

    print("\n=== Yearly Breakdown ===")
    print(by_year.to_string(index=False))

    # Optional: write results for plotting
    out_dir = Path("reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    res[["test_date", "net_ret", "equity", "n_longs", "turnover", "hit_rate"]].to_csv(out_dir / "wf_daily.csv", index=False)
    by_year.to_csv(out_dir / "wf_yearly.csv", index=False)
    print("\nSaved daily results → reports/wf_daily.csv")
    print("Saved yearly results → reports/wf_yearly.csv")
