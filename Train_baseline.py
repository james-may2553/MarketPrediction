import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np

# Load features + labels
X = pd.read_parquet("data/features/eq_features.parquet")
y = pd.read_parquet("data/features/labels.parquet")

# Merge on (date, ticker)
df = X.merge(y, on=["date", "ticker"], how="inner")

# Define features and label
feature_cols = [c for c in df.columns if c not in ["date", "ticker", "y_next_1d", "target_1d"]]
X = df[feature_cols].fillna(0).values
y = df["target_1d"].values

# Very simple train/test split (last 20% of data as test)
split = int(len(df) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Fit a logistic regression
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Baseline logistic regression accuracy: {acc:.3f}")
