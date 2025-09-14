# diagnostics.py
# -*- coding: utf-8 -*-
"""
Lightweight diagnostics for market model pipelines.
All plots are saved under outdir (default: artifacts/diag).
No side effects on your core pipeline.
"""

from __future__ import annotations
import os
import math
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------- utils ----------

def _ensure_dir(d: str | Path) -> Path:
    p = Path(d)
    p.mkdir(parents=True, exist_ok=True)
    return p

def _nan_sharpe(returns: pd.Series, periods_per_year: int = 252) -> float:
    r = pd.Series(returns).dropna()
    if r.empty:
        return float("nan")
    mu = r.mean()
    sig = r.std(ddof=0)
    if sig == 0 or np.isnan(sig):
        return float("nan")
    daily_sr = mu / sig
    return daily_sr * math.sqrt(periods_per_year)

def _savefig(path: Path, tight: bool = True) -> None:
    if tight:
        plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150)
    plt.close()

# ---------- 1) Label sanity ----------

def run_label_sanity(
    df: pd.DataFrame,
    label_col: str = "fwd_ret",
    outdir: str | Path = "artifacts/diag",
) -> Dict[str, float]:
    outdir = _ensure_dir(outdir)

    s = pd.Series(df[label_col]).astype(float)
    stats = dict(
        n=int(s.notna().sum()),
        mean=float(np.nanmean(s)),
        std=float(np.nanstd(s)),
        pct_pos=float(100 * np.nanmean(s > 0)),
        pct_neg=float(100 * np.nanmean(s < 0)),
        pct_zero=float(100 * np.nanmean(s == 0)),
    )

    plt.figure(figsize=(8, 4))
    s.hist(bins=120)
    plt.title(f"Label distribution: {label_col}")
    plt.xlabel(label_col)
    plt.ylabel("count")
    _savefig(Path(outdir) / f"01_label_hist_{label_col}.png")

    # rolling mean to visualize drift (if date index or 'date' column exists)
    if "date" in df.columns:
        tmp = df[["date", label_col]].copy()
        tmp = tmp.sort_values("date")
        tmp[label_col] = tmp[label_col].astype(float)
        tmp["roll_mean_63"] = tmp[label_col].rolling(63).mean()
        plt.figure(figsize=(10, 4))
        plt.plot(tmp["date"], tmp["roll_mean_63"])
        plt.title(f"Rolling mean (63d) of {label_col}")
        plt.xlabel("date")
        plt.ylabel("roll mean")
        _savefig(Path(outdir) / f"02_label_rollmean_{label_col}.png")

    pd.DataFrame([stats]).to_csv(Path(outdir) / "01_label_stats.csv", index=False)
    return stats

# ---------- 2) Feature–target correlations ----------

def run_feature_target_corr(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = "fwd_ret",
    outdir: str | Path = "artifacts/diag",
    method: str = "pearson",           # "pearson" | "spearman"
    sample_rows: int | None = 300_000, # set None to use all rows
    drop_constant: bool = True,
) -> pd.DataFrame:
    outdir = _ensure_dir(outdir)

    use = df[feature_cols + [target_col]].copy()
    if sample_rows is not None and len(use) > sample_rows:
        use = use.sample(sample_rows, random_state=42)

    X = use[feature_cols].astype(float)
    y = use[target_col].astype(float)

    if drop_constant:
        std = X.std(ddof=0)
        keep = std[std > 0].index.tolist()
        if len(keep) < len(feature_cols):
            feature_cols = keep
            X = X[feature_cols]

    # pandas' corrwith handles NaNs and constant cols gracefully (returns NaN, no spammy warnings)
    corr = X.corrwith(y, method=method)
    corr = corr.fillna(0.0).sort_values(ascending=False)

    corr.to_csv(Path(outdir) / "03_feature_corr.csv", header=["corr"])
    plt.figure(figsize=(max(10, len(corr) * 0.25), 4))
    corr.plot.bar()
    plt.axhline(0, linestyle="--")
    plt.title(f"Feature correlations vs {target_col} ({method})")
    _savefig(Path(outdir) / "03_feature_corr_bar.png")
    return corr.to_frame("corr")


# ---------- 3) Feature importance (tree) ----------

def run_feature_importance(
    df: pd.DataFrame,
    feature_cols: List[str],
    label_col: str = "label",
    outdir: str | Path = "artifacts/diag",
    model: Optional[object] = None,
    rows_cap: int = 200_000,      # cap rows for speed
    n_estimators: int = 150,      # fewer trees = faster
    n_jobs: int = -1,
    min_samples_leaf: int = 50,   # stabilize on noisy finance data
) -> pd.DataFrame:
    outdir = _ensure_dir(outdir)

    try:
        from sklearn.ensemble import RandomForestClassifier
    except Exception as e:
        raise RuntimeError("scikit-learn is required for feature importance") from e

    use = df[feature_cols + [label_col]].dropna(subset=[label_col]).copy()
    if rows_cap is not None and len(use) > rows_cap:
        use = use.sample(rows_cap, random_state=42)

    # drop constant features here too (just in case)
    Xf = use[feature_cols].astype(float)
    std = Xf.std(ddof=0)
    keep = std[std > 0].index.tolist()
    if len(keep) < len(feature_cols):
        feature_cols = keep
        Xf = Xf[feature_cols]

    y = use[label_col].astype(int).values
    X = Xf.values

    if model is None:
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=None,
            min_samples_leaf=min_samples_leaf,
            max_features="sqrt",
            n_jobs=n_jobs,
            random_state=42,
            class_weight="balanced_subsample",
        )

    model.fit(X, y)
    imp = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=True)
    imp.to_csv(Path(outdir) / "04_feature_importance.csv", header=["importance"])

    plt.figure(figsize=(max(10, len(imp) * 0.25), 6))
    imp.plot.barh()
    plt.title("RandomForest feature importance")
    _savefig(Path(outdir) / "04_feature_importance_barh.png")
    return imp.to_frame("importance")

# ---------- 4) Probability calibration curve ----------

def run_probability_calibration(
    y_true: Iterable[int],
    y_prob: Iterable[float],
    outdir: str | Path = "artifacts/diag",
    bins: int = 10,
) -> pd.DataFrame:
    outdir = _ensure_dir(outdir)
    y_true = pd.Series(y_true).astype(int)
    y_prob = pd.Series(y_prob).astype(float)

    q = pd.qcut(y_prob, q=bins, duplicates="drop")
    dfb = pd.DataFrame({"bin": q, "y": y_true, "p": y_prob})
    cal = dfb.groupby("bin").agg(
        n=("y", "size"),
        avg_pred=("p", "mean"),
        hit_rate=("y", "mean"),
    ).reset_index(drop=True)

    plt.figure(figsize=(5, 5))
    plt.plot(cal["avg_pred"], cal["hit_rate"], marker="o")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("avg predicted prob")
    plt.ylabel("empirical hit rate")
    plt.title("Calibration curve (deciles)")
    _savefig(Path(outdir) / "05_calibration_curve.png")

    cal.to_csv(Path(outdir) / "05_calibration_table.csv", index=False)
    return cal

# ---------- 5) Backtest breakdown ----------

def run_backtest_breakdown(
    daily_df: pd.DataFrame,
    outdir: str | Path = "artifacts/diag",
    date_col: str = "date",
    ret_col: str = "ret",
    turnover_col: Optional[str] = "turnover_pct",
) -> pd.DataFrame:
    outdir = _ensure_dir(outdir)
    d = daily_df.copy()
    d[date_col] = pd.to_datetime(d[date_col])
    d = d.sort_values(date_col)
    d["year"] = d[date_col].dt.year

    yearly = d.groupby("year").agg(
        days=(ret_col, "count"),
        ret_pct=(ret_col, lambda x: 100 * np.nansum(x)),
        sharpe=(ret_col, _nan_sharpe),
        avg_turnover_pct=(turnover_col, "mean") if turnover_col in d.columns else ("year", "size"),
    ).reset_index()

    yearly.to_csv(Path(outdir) / "06_yearly_breakdown.csv", index=False)

    plt.figure(figsize=(8, 4))
    plt.bar(yearly["year"].astype(str), yearly["ret_pct"])
    plt.title("Yearly return %")
    plt.xlabel("year")
    plt.ylabel("ret %")
    _savefig(Path(outdir) / "06_yearly_returns_bar.png")

    return yearly

# ---------- 6) Permutation test ----------

def run_permutation_test(
    X: np.ndarray,
    y: np.ndarray,
    model_factory: Callable[[], object],
    scorer: Callable[[Iterable[int], Iterable[float]], float],
    n_iter: int = 50,
    outdir: str | Path = "artifacts/diag",
) -> pd.DataFrame:
    """
    Shuffle labels n_iter times, refit, score; compare original score to null distribution.
    scorer should return a single float (e.g., out-of-fold AUC or Sharpe on validation set).
    """
    outdir = _ensure_dir(outdir)

    # original fit/score
    base_model = model_factory()
    base_model.fit(X, y)
    try:
        yhat = base_model.predict_proba(X)[:, 1]
    except Exception:
        yhat = base_model.predict(X)
    base_score = float(scorer(y, yhat))

    # permutations
    null_scores = []
    rng = np.random.default_rng(42)
    for _ in range(n_iter):
        y_shuf = rng.permutation(y)
        m = model_factory()
        m.fit(X, y_shuf)
        try:
            yhat_s = m.predict_proba(X)[:, 1]
        except Exception:
            yhat_s = m.predict(X)
        null_scores.append(float(scorer(y_shuf, yhat_s)))

    out = pd.DataFrame({"null_score": null_scores})
    out["base_score"] = base_score
    out["p_value_right"] = (out["null_score"] >= base_score).mean()

    out.to_csv(Path(outdir) / "07_permutation_scores.csv", index=False)

    plt.figure(figsize=(7, 4))
    plt.hist(out["null_score"], bins=30)
    plt.axvline(base_score, linestyle="--")
    plt.title(f"Permutation test — base={base_score:.4f}, p={out['p_value_right'].iloc[0]:.3f}")
    plt.xlabel("score")
    plt.ylabel("count")
    _savefig(Path(outdir) / "07_permutation_hist.png")

    return out
