#!/usr/bin/env python3
"""
ablation_runner.py — run walk-forward backtests across a parameter grid
Usage:
  python ablation_runner.py
  # or narrow the grid:
  python ablation_runner.py --quick
"""

import argparse
import itertools
import time
import numpy as np
import pandas as pd

import backtest_walk_forward as bf  # <— uses your file

# --- helper to run one config ------------------------------------------------
def run_one(cfg_overrides):
    # Apply overrides to the imported module (simple & explicit)
    for k, v in cfg_overrides.items():
        setattr(bf, k, v)

    df, feature_cols = bf.prepare_data()
    res, metrics, by_year = bf.walk_forward_backtest(df, feature_cols)

    # Add overall compounded return %
    overall_ret_pct = 100 * ((1.0 + res["net_ret"]).prod() - 1.0)

    row = {
        **cfg_overrides,
        "days": metrics["days"],
        "sharpe": metrics["sharpe"],
        "mean_daily_ret_%": metrics["mean_daily_ret_%"],
        "vol_daily_%": metrics["vol_daily_%"],
        "max_drawdown_%": metrics["max_drawdown_%"],
        "avg_turnover_%": metrics["avg_turnover_%"],
        "avg_n_longs": metrics["avg_n_longs"],
        "avg_hit_rate_%": metrics["avg_hit_rate_%"],
        "overall_ret_%": overall_ret_pct,
    }
    return row, res, by_year

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="smaller grid (fast sanity)")
    args = ap.parse_args()

    # ---------------- Parameter grids ----------------
    # You can tweak these freely.
    thresholds       = [0.50, 0.55, 0.60] if not args.quick else [0.55]
    vol_scale_flags  = [True, False]      if not args.quick else [True, False]
    sector_caps      = [0.25, 0.35, 1.00] if not args.quick else [0.25, 1.00]  # 1.00 ~ no cap if MAX_POSITIONS limits
    max_positions    = [20, 40, 80]       if not args.quick else [40]
    rebalance_every  = [1, 5]             if not args.quick else [5]
    train_window     = [252, 504]         if not args.quick else [504]
    cost_bps         = [2.0]              if not args.quick else [2.0]
    exclude_etfs     = [True]             if not args.quick else [True]

    # Build grid
    grid = []
    for THRESHOLD, VOL_SCALE_WEIGHTS, SECTOR_CAP_PCT, MAX_POSITIONS, REBALANCE_EVERY, TRAIN_WINDOW_DAYS, COST_BPS, EXCLUDE in itertools.product(
        thresholds, vol_scale_flags, sector_caps, max_positions, rebalance_every, train_window, cost_bps, exclude_etfs
    ):
        overrides = {
            "THRESHOLD": THRESHOLD,
            "VOL_SCALE_WEIGHTS": VOL_SCALE_WEIGHTS,
            "SECTOR_CAP_PCT": SECTOR_CAP_PCT,
            "MAX_POSITIONS": MAX_POSITIONS,
            "REBALANCE_EVERY": REBALANCE_EVERY,
            "TRAIN_WINDOW_DAYS": TRAIN_WINDOW_DAYS,
            "COST_BPS": COST_BPS,
        }
        # Toggle ETF exclusion by swapping list
        if EXCLUDE:
            overrides["EXCLUDE_TICKERS"] = {"SPY","QQQ","IWM","DIA","XLK","XLF","XLY","XLC","XLI","XLE","XLP","XLV","XLU","XLB","XLRE"}
        else:
            overrides["EXCLUDE_TICKERS"] = set()

        grid.append(overrides)

    results = []
    t0 = time.time()

    for i, cfg in enumerate(grid, 1):
        print(f"\n[{i}/{len(grid)}] Running: {cfg}")
        try:
            row, res, by_year = run_one(cfg)
            results.append(row)
        except Exception as e:
            print(f"  ✖ Failed: {e}")
            # Record failure so you see which combo broke
            results.append({**cfg, "sharpe": np.nan, "overall_ret_%": np.nan, "error": str(e)})

    # Collect & sort
    df_res = pd.DataFrame(results)
    # Rank by Sharpe then overall return
    df_res = df_res.sort_values(["sharpe", "overall_ret_%"], ascending=[False, False]).reset_index(drop=True)

    # Save
    out_dir = Path("reports"); out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / ("ablation_quick.csv" if args.quick else "ablation_full.csv")
    df_res.to_csv(out_csv, index=False)

    print(f"\nSaved ablation results → {out_csv}")
    # Show top 10
    print("\nTop 10 by Sharpe:")
    cols = ["sharpe","overall_ret_%","mean_daily_ret_%","max_drawdown_%","avg_turnover_%",
            "THRESHOLD","VOL_SCALE_WEIGHTS","SECTOR_CAP_PCT","MAX_POSITIONS","REBALANCE_EVERY","TRAIN_WINDOW_DAYS","COST_BPS"]
    print(df_res[cols].head(10).to_string(index=False))

if __name__ == "__main__":
    from pathlib import Path
    main()
