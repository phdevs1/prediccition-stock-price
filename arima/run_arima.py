#!/usr/bin/env python3
# -*-Encoding: utf-8 -*-
"""
ARIMA baseline — multi-step stock return prediction.

Mirrors main.py structure: iterates over all CSVs in --root_path,
evaluates on the same train/val/test split as D-Va (data_loader.py),
and saves results in the same format for direct comparison.

Usage:
    # Same dataset and horizon as D-Va:
    python arima/run_arima.py --root_path data_processed_ln_2 --sequence_length 10

    # Other horizons (same --sequence_length convention as main.py):
    python arima/run_arima.py --root_path data_processed_ln_2 --sequence_length 20
    python arima/run_arima.py --root_path data_processed_ln_2 --sequence_length 40

    # Wider ARIMA search space:
    python arima/run_arima.py --root_path data_processed_ln_2 --sequence_length 10 --max_p 8 --max_q 8

Output layout (inside arima/results/):
    {file}_{setting}/
        pred.npy          (n_windows, pred_len, 1)  — standardized forecasts
        true.npy          (n_windows, pred_len, 1)  — standardized actuals
        target_mean.npy   scalar — train mean of target_return
        target_std.npy    scalar — train std  of target_return
    {setting}.csv         per-ticker MSE table (same format as main.py output)
"""
import argparse
import os
import sys
import numpy as np
import pandas as pd

# Allow running from any CWD: resolve paths relative to project root
_SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _PROJECT_ROOT)

from arima.arima_model import evaluate_ticker  # noqa: E402

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="ARIMA baseline for stock return prediction")

parser.add_argument(
    "--root_path",
    type=str,
    default="./data_processed_ln_2",
    help="directory containing CSV files (one per ticker)",
)
parser.add_argument(
    "--sequence_length",
    type=int,
    default=10,
    help="input sequence length — used to align test windows with D-Va",
)
parser.add_argument(
    "--prediction_length",
    type=int,
    default=None,
    help="forecast horizon; defaults to sequence_length (same as main.py)",
)
parser.add_argument(
    "--max_p",
    type=int,
    default=5,
    help="maximum AR order for auto-ARIMA search",
)
parser.add_argument(
    "--max_q",
    type=int,
    default=5,
    help="maximum MA order for auto-ARIMA search",
)
parser.add_argument(
    "--results_dir",
    type=str,
    default=None,
    help="output directory (default: arima/results/ next to this script)",
)
parser.add_argument(
    "--verbose",
    action="store_true",
    help="print selected ARIMA order per ticker",
)

args = parser.parse_args()

if args.prediction_length is None:
    args.prediction_length = args.sequence_length

if args.results_dir is None:
    args.results_dir = os.path.join(_SCRIPT_DIR, "results")

# Resolve root_path relative to project root if not absolute
if not os.path.isabs(args.root_path):
    args.root_path = os.path.join(_PROJECT_ROOT, args.root_path)

# Setting tag mirrors main.py: "tp{year}_sl{seq_len}"
year_tag      = os.path.basename(args.root_path.rstrip("/"))
train_setting = f"tp{year_tag}_sl{args.sequence_length}"

print(f"ARIMA baseline — {train_setting}")
print(f"  data   : {args.root_path}")
print(f"  seq_len: {args.sequence_length}  pred_len: {args.prediction_length}")
print(f"  results: {args.results_dir}\n")

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
csv_files   = sorted(f for f in os.listdir(args.root_path) if f.endswith(".csv"))
results_rows = []

for idx, file in enumerate(csv_files):
    ticker = os.path.splitext(file)[0]
    print(f"[{idx+1:3d}/{len(csv_files)}] {ticker} ... ", end="", flush=True)

    df     = pd.read_csv(os.path.join(args.root_path, file))
    series = df.iloc[:, -1].values.astype(float)  # last column = target_return

    preds, pred_stds, trues, t_mean, t_std, order = evaluate_ticker(
        series,
        seq_len  = args.sequence_length,
        pred_len = args.prediction_length,
        max_p    = args.max_p,
        max_q    = args.max_q,
        verbose  = args.verbose,
    )

    mse = float(np.mean((preds - trues) ** 2))
    print(f"MSE={mse:.6f}  order={order}  windows={preds.shape[0]}")

    # -----------------------------------------------------------------------
    # Save results — same format as Exp_Model.test() for portfolio compatibility
    # pred/true shape: (n_windows, pred_len, 1)
    # -----------------------------------------------------------------------
    folder = os.path.join(args.results_dir, f"{file}_{train_setting}")
    os.makedirs(folder, exist_ok=True)
    np.save(os.path.join(folder, "pred.npy"),        preds[:, :, np.newaxis])
    np.save(os.path.join(folder, "pred_std.npy"),    pred_stds[:, :, np.newaxis])
    np.save(os.path.join(folder, "true.npy"),        trues[:, :, np.newaxis])
    np.save(os.path.join(folder, "target_mean.npy"), np.array([t_mean]))
    np.save(os.path.join(folder, "target_std.npy"),  np.array([t_std]))

    results_rows.append({
        "Ticker":      ticker,
        "MSE":         mse,
        "StdDev":      0.0,          # ARIMA is deterministic — no variance across runs
        "ARIMA_order": str(order),
    })

# ---------------------------------------------------------------------------
# Aggregate CSV
# ---------------------------------------------------------------------------
results = pd.DataFrame(results_rows)
os.makedirs(args.results_dir, exist_ok=True)
out_csv = os.path.join(args.results_dir, f"{train_setting}.csv")
results.to_csv(out_csv, index=False)

print(f"\n{'='*60}")
print(f"Results saved: {out_csv}")
print(f"Mean MSE across {len(results)} tickers: {results['MSE'].mean():.6f}")
print(f"{'='*60}")
print(results.to_string(index=False))
