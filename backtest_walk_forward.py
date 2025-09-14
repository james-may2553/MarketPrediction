import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer
from diagnostics import (
    run_label_sanity,
    run_feature_target_corr,
    run_feature_importance,
    run_probability_calibration,
    run_backtest_breakdown,
)

try:
    from tqdm import tqdm
except Exception:  # fallback if tqdm isn't installed
    def tqdm(x, **k): 
        return x

# =====================
# Config (tweak freely)
# =====================
FEATURES_PATH = "data/features/eq_features.parquet"
LABELS_PATH   = "data/features/labels.parquet"
SECTOR_MAP_CSV = "data/raw/sector_map.csv"   # used for sector caps

# Walk-forward settings
WARMUP_DAYS        = 504      # require ~2Y before first test
TRAIN_WINDOW_DAYS  = 504      # rolling window length
EMBARGO_DAYS       = 10       # bleed protection (we'll auto-bump to >= label horizon)
REBALANCE_EVERY    = 10       # refit every N days (weekly-ish)

# ===== Model settings =====
MODEL_TYPE = "lgbm"      # "logreg" | "logreg_l1" | "lgbm" | "xgb"
CLASS_WEIGHT = "balanced"  # for linear models: None | "balanced"
MAX_ITER = 2000            # for linear models

# Calibration (probability calibration on a recent validation slice)
CALIBRATE = True
CAL_METHOD = "isotonic"    # "isotonic" (nonlinear) or "sigmoid" (Platt)
VAL_DAYS = 60              # last N days of the training window used for calibration
MIN_VAL_ROWS = 2000        # fallback: skip calibration if not enough rows

# Trading rules
THRESHOLD          = 0.5     # higher -> fewer, higher conviction longs
MIN_STOCKS         = 5       # skip day if too few candidates
MAX_POSITIONS      = 80      # cap portfolio breadth
SECTOR_CAP_PCT     = 0.35    # max fraction of names from any one sector
VOL_SCALE_WEIGHTS  = False   # 1/(vol_20+eps) scaling of weights
EXCLUDE_TICKERS    = {"SPY","QQQ","IWM","DIA","XLK","XLF","XLY","XLC","XLI","XLE","XLP","XLV","XLU","XLB","XLRE"}  # avoid ETFs

# Cost model (simple)
COST_BPS           = 2.0      # per day, approximate via turnover

RANDOM_STATE       = 0

# ---- Label selection (auto-detect horizon) ----
def pick_label_cols(df):
    """
    Returns (y_col, target_col, horizon_days).
    Tries 10d, then 5d, then 1d.
    """
    options = [
        ("y_next_10d", "target_10d", 10),
        ("y_next_5d",  "target_5d",   5),
        ("y_next_1d",  "target_1d",   1),
    ]
    for y_col, t_col, H in options:
        if y_col in df.columns and t_col in df.columns:
            return y_col, t_col, H
    raise KeyError(
        f"No label columns found. Looked for: {options}. "
        f"Got columns: {list(df.columns)[:50]}"
    )

# Optional: quick null-signal sanity check
RUN_PERMUTATION_TEST = False
PERM_N_ITER = 50

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

def _inf_to_nan(X):
    X = np.asarray(X, dtype=float)
    X[~np.isfinite(X)] = np.nan
    return X

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

    # Exclude ETFs/benchmarks from trading universe (keep them in df in case used as features)
    df["is_excluded"] = df["ticker"].isin(EXCLUDE_TICKERS)

    # ---- Determine label columns dynamically ----
    Y_COL, TARGET_COL, H = pick_label_cols(df)

    # ---- Feature columns: numeric only, exclude labels/metadata ----
    numerics = df.select_dtypes(include=[np.number, "bool"]).columns
    skip = {Y_COL, TARGET_COL}
    feature_cols = [c for c in numerics if c not in skip]

    # Replace infs with NaN so stats don’t blow up
    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)

    # Drop ultra-sparse features (e.g., >98% missing)
    na_ratio = df[feature_cols].isna().mean()
    hi_na = na_ratio[na_ratio > 0.98].index.tolist()
    if hi_na:
        feature_cols = [c for c in feature_cols if c not in hi_na]

    # Drop constant/all-NaN features
    col_std = df[feature_cols].astype(float).std(ddof=0)
    const_cols = col_std.index[(col_std == 0.0) | col_std.isna()].tolist()
    if const_cols:
        feature_cols = [c for c in feature_cols if c not in const_cols]

    if not feature_cols:
        raise ValueError("No usable features after cleaning (all dropped).")

    if hi_na or const_cols:
        print(f"⚠️ Dropped {len(hi_na)} high-NaN and {len(const_cols)} constant features.")

    return df, feature_cols, (Y_COL, TARGET_COL, H)

# =====================
# Model
# =====================
def make_model():
    mt = MODEL_TYPE.lower()

    if mt == "logreg":
        # L2 logistic
        return Pipeline([
            ("finite", FunctionTransformer(np.nan_to_num, feature_names_out="one-to-one")),  # inf → NaN/finite
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("clf", LogisticRegression(
                penalty="l2", C=1.0, solver="saga",
                class_weight=CLASS_WEIGHT, max_iter=MAX_ITER,
                n_jobs=-1, random_state=RANDOM_STATE
            ))
        ])

    if mt == "logreg_l1":
        # L1 logistic (sparse)
        return Pipeline([
            ("finite", FunctionTransformer(np.nan_to_num, feature_names_out="one-to-one")),
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("clf", LogisticRegression(
                penalty="l1", C=0.5, solver="saga",
                class_weight=CLASS_WEIGHT, max_iter=MAX_ITER,
                n_jobs=-1, random_state=RANDOM_STATE
            ))
        ])

    if mt == "lgbm":
        from lightgbm import LGBMClassifier
        return Pipeline([
            ("finite", FunctionTransformer(np.nan_to_num, feature_names_out="one-to-one")),
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", LGBMClassifier(
                n_estimators=600, learning_rate=0.03, num_leaves=63,
                subsample=0.8, colsample_bytree=0.8,
                min_child_samples=40, reg_lambda=1.0,
                random_state=RANDOM_STATE, n_jobs=-1
            ))
        ])

    if mt == "xgb":
        from xgboost import XGBClassifier
        return Pipeline([
            ("finite", FunctionTransformer(np.nan_to_num, feature_names_out="one-to-one")),
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", XGBClassifier(
                n_estimators=700, learning_rate=0.03, max_depth=6,
                subsample=0.8, colsample_bytree=0.8,
                reg_lambda=1.0, eval_metric="logloss",
                random_state=RANDOM_STATE, n_jobs=-1
            ))
        ])

    raise ValueError(f"Unknown MODEL_TYPE={MODEL_TYPE}")


def maybe_calibrate(prefit_model, X_val, y_val):
    """Wrap a prefit model with a probability calibrator, if enabled/possible."""
    if not CALIBRATE or len(y_val) < MIN_VAL_ROWS:
        return prefit_model
    try:
        # For Pipelines or estimators; CalibratedClassifierCV handles both.
        calib = CalibratedClassifierCV(prefit_model, method=CAL_METHOD, cv="prefit")
        calib.fit(X_val, y_val)
        return calib
    except Exception:
        # If model type doesn't play nicely, just return original
        return prefit_model
    

def make_base_models_for_ensemble():
    """Return a list of base estimators for voting/stacking."""
    bases = []
    # Linear pair
    bases.append(Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("clf", LogisticRegression(penalty="l2", C=1.0, solver="saga",
                                   class_weight=CLASS_WEIGHT, max_iter=MAX_ITER,
                                   n_jobs=-1, random_state=RANDOM_STATE))
    ]))
    bases.append(Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("clf", LogisticRegression(penalty="l1", C=0.5, solver="saga",
                                   class_weight=CLASS_WEIGHT, max_iter=MAX_ITER,
                                   n_jobs=-1, random_state=RANDOM_STATE))
    ]))
    # Trees (light settings to keep runtime reasonable)
    from lightgbm import LGBMClassifier
    from xgboost import XGBClassifier
    bases.append(LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=63,
                                subsample=0.8, colsample_bytree=0.8,
                                min_child_samples=40, reg_lambda=1.0,
                                random_state=RANDOM_STATE, n_jobs=-1))
    bases.append(XGBClassifier(n_estimators=500, learning_rate=0.05, max_depth=6,
                               subsample=0.8, colsample_bytree=0.8,
                               reg_lambda=1.0, eval_metric="logloss",
                               random_state=RANDOM_STATE, n_jobs=-1))
    return bases

def fit_voting(X_tr, y_tr, X_va=None, y_va=None):
    """Fit base models and return a callable that outputs averaged probs.
       If CALIBRATE, calibrate the averaged probs on validation slice."""
    bases = make_base_models_for_ensemble()
    for m in bases: m.fit(X_tr, y_tr)

    def _avg_proba(X):
        ps = [m.predict_proba(X)[:,1] for m in bases]
        return np.mean(ps, axis=0)

    if CALIBRATE and X_va is not None and y_va is not None and len(y_va) >= MIN_VAL_ROWS:
        # Calibrate the averaged probabilities (single calibrator on the blend)
        p_va = _avg_proba(X_va)
        if CAL_METHOD == "sigmoid":
            from sklearn.linear_model import LogisticRegression as LR
            from sklearn.preprocessing import FunctionTransformer
            from sklearn.pipeline import make_pipeline
            calibrator = make_pipeline(
                FunctionTransformer(lambda z: z.reshape(-1,1), validate=False),
                LR(max_iter=1000)
            )
            calibrator.fit(p_va, y_va)
            return lambda X: calibrator.predict_proba(_avg_proba(X))[:,1]
        else:
            from sklearn.isotonic import IsotonicRegression
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(p_va, y_va)
            return lambda X: iso.predict(_avg_proba(X))
    else:
        return _avg_proba

def fit_stacking(X_tr, y_tr, X_va, y_va):
    """Fit base models on train; train a meta-learner on validation probs."""
    bases = make_base_models_for_ensemble()
    for m in bases: m.fit(X_tr, y_tr)

    # Validation meta-features = base probas
    P_va = np.column_stack([m.predict_proba(X_va)[:,1] for m in bases])

    # Meta-learner (logistic); optional calibration wraps after
    meta = LogisticRegression(max_iter=2000, class_weight=CLASS_WEIGHT, solver="lbfgs")
    meta.fit(P_va, y_va)

    if CALIBRATE and len(y_va) >= MIN_VAL_ROWS:
        meta = CalibratedClassifierCV(meta, method=CAL_METHOD, cv="prefit")
        meta.fit(P_va, y_va)

    def _stacked_proba(X):
        P = np.column_stack([m.predict_proba(X)[:,1] for m in bases])
        return meta.predict_proba(P)[:,1]
    return _stacked_proba

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
        w = np.ones(len(df_sub), dtype=float)
    else:
        inv = 1.0 / (df_sub["vol_20"].fillna(df_sub["vol_20"].median()) + eps)
        w = inv.to_numpy()
    w = w / w.sum() if w.sum() > 0 else np.ones_like(w) / len(w)
    return w

# =====================
# Walk-forward backtest
# =====================
def _build_row_ranges(df):
    """Precompute contiguous row ranges [start,end) per unique date (df must be sorted by ["date","ticker"])."""
    dates = pd.Index(sorted(df["date"].unique()))
    counts = df["date"].value_counts().reindex(dates).to_numpy()
    starts = np.empty(len(dates), dtype=np.int64)
    starts[0] = 0
    if len(dates) > 1:
        starts[1:] = np.cumsum(counts[:-1])
    ends = starts + counts
    return dates, starts, ends, counts

def walk_forward_backtest(df, feature_cols, Y_COL, TARGET_COL, H):
    # ---------- Precompute ----------
    df = df.sort_values(["date","ticker"]).reset_index(drop=True)

    dates, row_starts, row_ends, counts = _build_row_ranges(df)

    X_all   = df[feature_cols].to_numpy(np.float32)
    y_trg   = df[TARGET_COL].to_numpy(np.int8)      # classification label
    y_next  = df[Y_COL].to_numpy(np.float32)        # forward return (float)
    tickers = df["ticker"].astype(str).to_numpy()
    sectors = df["sector"].fillna("UNKNOWN").astype(str).to_numpy()
    excluded = df["is_excluded"].to_numpy(np.bool_)
    vol20   = df.get("vol_20", pd.Series(np.nan, index=df.index)).to_numpy(np.float32)

    # progress bounds
    if len(dates) <= WARMUP_DAYS + 1:
        raise ValueError("Not enough history for the chosen WARMUP_DAYS.")
    local_embargo = max(EMBARGO_DAYS, H)
    start_idx = WARMUP_DAYS + local_embargo
    end_idx   = len(dates) - 1

    # modeling objects (warm-start)
    base_model = None
    proba_fn   = None
    calibrator = None
    last_calib_day = -10**9   # force initial calibration

    daily_rows = []
    prev_w = pd.Series(dtype=np.float32)  # previous day weights by ticker

    # ---- Diagnostics B: collectors for probability calibration (per test fold) ----
    all_probs, all_labels = [], []

    it = tqdm(range(start_idx, end_idx), total=max(0, end_idx - start_idx),
              desc="Walk-forward (fast)", mininterval=0.2, smoothing=0.1, leave=True)

    for t_idx in it:
        # --- compute train/test row slices via contiguous ranges ---
        train_end_idx   = t_idx - local_embargo
        train_start_idx = max(0, train_end_idx - TRAIN_WINDOW_DAYS)

        tr_start_row = int(row_starts[train_start_idx])
        tr_end_row   = int(row_ends[train_end_idx - 1]) if train_end_idx - 1 >= train_start_idx else tr_start_row

        te_start_row = int(row_starts[t_idx])
        te_end_row   = int(row_ends[t_idx])

        if tr_end_row <= tr_start_row or (row_ends[t_idx] - row_starts[t_idx]) < MIN_STOCKS:
            continue

        # validation tail by *days*, not rows
        val_start_day = max(train_start_idx, train_end_idx - VAL_DAYS)
        va_start_row  = int(row_starts[val_start_day])
        va_end_row    = int(row_ends[train_end_idx - 1]) if train_end_idx - 1 >= val_start_day else va_start_row

        # build numpy views (no copies)
        X_tr = X_all[tr_start_row:tr_end_row]
        y_tr = y_trg[tr_start_row:tr_end_row]

        if va_end_row - va_start_row >= MIN_VAL_ROWS:
            X_va = X_all[va_start_row:va_end_row]
            y_va = y_trg[va_start_row:va_end_row]
        else:
            X_va = None; y_va = None

        # --- refit cadence & warm-start ---
        need_refit = (proba_fn is None) or ((t_idx - start_idx) % REBALANCE_EVERY == 0)
        if need_refit:
            if MODEL_TYPE.lower() in {"logreg", "logreg_l1"}:
                penalty = "l1" if MODEL_TYPE.lower() == "logreg_l1" else "l2"
                clf = LogisticRegression(
                    penalty=penalty, C=(0.5 if penalty=="l1" else 1.0),
                    solver="saga", class_weight=CLASS_WEIGHT,
                    max_iter=MAX_ITER, n_jobs=-1, random_state=RANDOM_STATE,
                    warm_start=True,
                )
                pipe = Pipeline([
                    ("finite", FunctionTransformer(np.nan_to_num, feature_names_out="one-to-one")),
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler(with_mean=True, with_std=True)),
                    ("clf", clf),
                ])
                # reuse existing estimator for warm_start
                if base_model is not None:
                    pipe["clf"].coef_ = getattr(base_model["clf"], "coef_", None)
                    pipe["clf"].intercept_ = getattr(base_model["clf"], "intercept_", None)
                pipe.fit(X_tr, y_tr)
                base_model = pipe
                proba_fn = lambda X: base_model.predict_proba(X)[:,1]
            elif MODEL_TYPE.lower() in {"lgbm", "xgb"}:
                base_model = make_model()
                base_model.fit(X_tr, y_tr)
                proba_fn = lambda X: base_model.predict_proba(X)[:,1]
            elif MODEL_TYPE.lower() == "voting":
                proba_fn = fit_voting(X_tr, y_tr, X_va, y_va)
            elif MODEL_TYPE.lower() == "stacking":
                if X_va is None or y_va is None:
                    proba_fn = fit_voting(X_tr, y_tr, None, None)
                else:
                    proba_fn = fit_stacking(X_tr, y_tr, X_va, y_va)
            else:
                raise ValueError(f"Unknown MODEL_TYPE={MODEL_TYPE}")

            # probability calibration (throttled)
            if CALIBRATE and (X_va is not None) and (len(y_va) >= MIN_VAL_ROWS) and (t_idx - last_calib_day >= max(5, REBALANCE_EVERY)):
                try:
                    if CAL_METHOD == "sigmoid":
                        calibrator = CalibratedClassifierCV(base_model, method="sigmoid", cv="prefit")
                        calibrator.fit(X_va, y_va)
                        proba_fn = lambda X: calibrator.predict_proba(X)[:,1]
                    else:
                        from sklearn.isotonic import IsotonicRegression
                        p_va = proba_fn(X_va)
                        iso = IsotonicRegression(out_of_bounds="clip")
                        iso.fit(p_va, y_va)
                        proba_fn = lambda X, _pf=proba_fn, _iso=iso: _iso.predict(_pf(X))
                    last_calib_day = t_idx
                except Exception:
                    pass  # keep uncalibrated if it fails

        # --- score today's cross-section (NumPy slice) ---
        X_te = X_all[te_start_row:te_end_row]
        p    = proba_fn(X_te)

        # ---- Diagnostics B (inside loop): collect probs + true labels for calibration later ----
        all_probs.append(p)
        all_labels.append(y_trg[te_start_row:te_end_row])

        # construct a tiny day-frame just once
        day = pd.DataFrame({
            "ticker":  tickers[te_start_row:te_end_row],
            "sector":  sectors[te_start_row:te_end_row],
            "prob_up": p.astype(np.float32),
            "y_next":  y_next[te_start_row:te_end_row],
            "excluded": excluded[te_start_row:te_end_row],
            "vol_20":  vol20[te_start_row:te_end_row],
        })

        # universe filter
        pool = day[(day["prob_up"] >= THRESHOLD) & (~day["excluded"])]
        if pool.shape[0] < MIN_STOCKS:
            daily_rows.append({"test_date": dates[t_idx], "n_longs": 0,
                               "gross_ret": 0.0, "cost": 0.0, "net_ret": 0.0,
                               "turnover": 0.0, "hit_rate": np.nan, "accuracy": np.nan})
            if len(daily_rows) % 25 == 0:
                it.set_postfix({"date": str(dates[t_idx].date()), "n": 0}, refresh=False)
            continue

        # partial sort then sector cap (vectorized)
        k_take = min(pool.shape[0], MAX_POSITIONS * 3)
        pool = pool.nlargest(k_take, "prob_up")
        per_sec = max(1, int(np.floor(SECTOR_CAP_PCT * MAX_POSITIONS)))
        pool = (pool.sort_values("prob_up", ascending=False)
                    .groupby("sector", sort=False)
                    .head(per_sec)
                    .head(MAX_POSITIONS))

        n_longs = pool.shape[0]
        if n_longs < MIN_STOCKS:
            daily_rows.append({"test_date": dates[t_idx], "n_longs": 0,
                               "gross_ret": 0.0, "cost": 0.0, "net_ret": 0.0,
                               "turnover": 0.0, "hit_rate": np.nan, "accuracy": np.nan})
            if len(daily_rows) % 25 == 0:
                it.set_postfix({"date": str(dates[t_idx].date()), "n": 0}, refresh=False)
            continue

        # weights
        if VOL_SCALE_WEIGHTS and pool["vol_20"].notna().any():
            eps = 1e-6
            inv = 1.0 / (pool["vol_20"].fillna(pool["vol_20"].median()) + eps).to_numpy(np.float32)
            w = inv / inv.sum()
        else:
            w = np.full(n_longs, 1.0 / n_longs, dtype=np.float32)
        pool = pool.assign(weight=w)

        # P&L (no merge)
        gross_ret = float(np.dot(pool["weight"].to_numpy(), pool["y_next"].to_numpy()))

        # turnover and costs (vectorized align)
        curr_w = pd.Series(pool["weight"].to_numpy(), index=pool["ticker"].to_numpy())
        union = curr_w.index.union(prev_w.index)
        turnover = np.abs(curr_w.reindex(union, fill_value=0.0) - prev_w.reindex(union, fill_value=0.0)).sum()
        cost = (COST_BPS / 10000.0) * float(turnover)
        net_ret = gross_ret - cost
        hit_rate = float((pool["y_next"] > 0).mean())

        daily_rows.append({
            "test_date": dates[t_idx],
            "n_longs": int(n_longs),
            "gross_ret": gross_ret,
            "cost": cost,
            "net_ret": net_ret,
            "turnover": float(turnover),
            "hit_rate": hit_rate,
            "accuracy": hit_rate,
        })
        prev_w = curr_w

        if len(daily_rows) % 25 == 0:
            it.set_postfix({"date": str(dates[t_idx].date()), "n": int(n_longs)}, refresh=False)

    # ---- results / metrics (unchanged logic) ----
    res = pd.DataFrame(daily_rows)
    if res.empty:
        raise RuntimeError("No backtest rows produced—tune your config.")
    res = res.sort_values("test_date").reset_index(drop=True)
    res["equity"] = (1.0 + res["net_ret"]).cumprod()

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

    res["year"] = pd.DatetimeIndex(res["test_date"]).year
    by_year = res.groupby("year").agg(
        days=("net_ret", "size"),
        ret_pct=("net_ret", lambda x: 100 * (np.prod(1 + x) - 1)),
        sharpe=("net_ret", sharpe),
        hit_pct=("hit_rate", lambda x: 100 * np.nanmean(x)),
        avg_turnover_pct=("turnover", lambda x: 100 * np.nanmean(x)),
    ).reset_index()

    # ---- Diagnostics C: yearly breakdown artifact ----
    run_backtest_breakdown(
        res.rename(columns={"test_date": "date", "net_ret": "ret", "turnover": "turnover_pct"}),
        outdir="artifacts/diag",
        date_col="date",
        ret_col="ret",
        turnover_col="turnover_pct",
    )

    # ---- Diagnostics B (after loop): probability calibration (across all test folds) ----
    try:
        if len(all_probs) and len(all_labels):
            y_prob = np.concatenate(all_probs)
            y_true = np.concatenate(all_labels).astype(int)
            run_probability_calibration(y_true, y_prob, outdir="artifacts/diag", bins=10)
    except Exception:
        pass  # don't let diagnostics break the run

    return res, metrics, by_year

# =====================
# Main
# =====================
if __name__ == "__main__":
    df, feature_cols, (Y_COL, TARGET_COL, H) = prepare_data()

    # ---- Diagnostics A: dataset-wide checks (once, before training) ----
    try:
        run_label_sanity(df, label_col=Y_COL, outdir="artifacts/diag")
    except Exception:
        pass
    try:
        run_feature_target_corr(df, feature_cols, target_col=Y_COL, outdir="artifacts/diag")
    except Exception:
        pass
    try:
        run_feature_importance(df, feature_cols, label_col=TARGET_COL, outdir="artifacts/diag")
    except Exception:
        pass

    # (Optional) Null-signal sanity check — disabled by default
    if RUN_PERMUTATION_TEST:
        try:
            from diagnostics import run_permutation_test
            from sklearn.linear_model import LogisticRegression as LR
            from sklearn.metrics import roc_auc_score
            X_perm = df[feature_cols].astype(float).values
            y_perm = df[TARGET_COL].astype(int).values
            _ = run_permutation_test(
                X_perm, y_perm,
                model_factory=lambda: LR(max_iter=500, n_jobs=-1),
                scorer=lambda yt, yp: roc_auc_score(yt, yp),
                n_iter=PERM_N_ITER,
                outdir="artifacts/diag",
            )
        except Exception:
            pass

    res, metrics, by_year = walk_forward_backtest(df, feature_cols, Y_COL, TARGET_COL, H)

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
