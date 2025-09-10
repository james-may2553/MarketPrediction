import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

# -------- settings you can tweak ----------
FEATURES_PATH = "data/features/eq_features.parquet"
LABELS_PATH   = "data/features/labels.parquet"
TEST_FRACTION = 0.20        # last 20% of DATES used for test
EMBARGO_DAYS  = 10          # drop last X training days to avoid bleed into test
MAX_ITER      = 1000
# -----------------------------------------

def load_and_merge():
    X = pd.read_parquet(FEATURES_PATH)
    y = pd.read_parquet(LABELS_PATH)
    df = X.merge(y, on=["date", "ticker"], how="inner")
    # ensure proper dtypes & sorting
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "ticker"]).reset_index(drop=True)
    return df

def time_safe_split(df, test_fraction=0.2, embargo_days=0):
    # Split by unique DATES (not rows) to avoid cross-sectional leakage
    all_dates = np.array(sorted(df["date"].unique()))
    split_idx = int(len(all_dates) * (1 - test_fraction))
    train_end_date = all_dates[split_idx - 1]

    # Apply embargo: remove the last `embargo_days` trading dates from the train set
    if embargo_days > 0 and split_idx - embargo_days > 0:
        embargo_cutoff = all_dates[split_idx - embargo_days - 1]
    else:
        embargo_cutoff = train_end_date

    train_mask = df["date"] <= embargo_cutoff
    test_mask  = df["date"] > train_end_date   # strictly later dates only

    train_df = df.loc[train_mask].copy()
    test_df  = df.loc[test_mask].copy()

    # Safety checks
    assert train_df["date"].max() < test_df["date"].min(), "Train/Test date overlap!"
    return train_df, test_df, train_end_date, embargo_cutoff

def main():
    df = load_and_merge()

    # Feature set (exclude metadata + labels)
    feature_cols = [c for c in df.columns if c not in ["date", "ticker", "y_next_1d", "target_1d"]]

    train_df, test_df, split_date, embargo_cutoff = time_safe_split(
        df, test_fraction=TEST_FRACTION, embargo_days=EMBARGO_DAYS
    )

    X_train = train_df[feature_cols].astype(float).fillna(0.0).values
    y_train = train_df["target_1d"].astype(int).values

    X_test  = test_df[feature_cols].astype(float).fillna(0.0).values
    y_test  = test_df["target_1d"].astype(int).values

    model = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("lr", LogisticRegression(max_iter=MAX_ITER))
    ])

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    try:
        auc = roc_auc_score(y_test, y_proba)
    except ValueError:
        auc = float("nan")  # (e.g., if only one class shows up in test)

    # Per-date accuracy (helps spot regime effects)
    per_day = test_df.assign(pred=y_pred).groupby("date").apply(
        lambda d: (d["target_1d"] == d["pred"]).mean()
    ).rename("daily_accuracy").reset_index()

    print("\n=== Time-Safe Baseline (chronological split) ===")
    print(f"Train end date: {split_date.date()}  |  Embargo cutoff used in train: {embargo_cutoff.date()}")
    print(f"Train rows: {len(train_df):,}  |  Test rows: {len(test_df):,}  |  Unique test dates: {test_df['date'].nunique():,}")
    print(f"Accuracy: {acc:.3f}   ROC-AUC: {auc:.3f}")
    print("\nClassification report (test):")
    print(classification_report(y_test, y_pred, digits=3))

    # Optional: save per-day accuracies for plotting
    out_dir = Path("reports"); out_dir.mkdir(parents=True, exist_ok=True)
    per_day.to_csv(out_dir / "baseline_timesafe_daily_accuracy.csv", index=False)
    print("\nSaved per-day accuracy → reports/baseline_timesafe_daily_accuracy.csv")

if __name__ == "__main__":
    main()
