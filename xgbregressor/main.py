# -*-Encoding: utf-8 -*-
import argparse
import os
import sys
import numpy as np
import random
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_load.data_loader import Dataset_Custom
from xgb_runner import XGBRunner

fix_seed = 100
random.seed(fix_seed)
np.random.seed(fix_seed)

parser = argparse.ArgumentParser(description="XGBoost baseline for DVA comparison")

# Data — same as main.py
parser.add_argument("--root_path", type=str, default="../data/2020_2022", help="root path of the data files")
parser.add_argument("--sequence_length", type=int, default=10, help="length of input sequence")
parser.add_argument("--prediction_length", type=int, default=None, help="prediction sequence length")
parser.add_argument("--target_dim", type=int, default=1, help="dimension of target")

# Training
parser.add_argument("--itr", type=int, default=1, help="number of experiment repetitions (XGBoost is deterministic, 1 es suficiente)")

# XGBoost hyperparameters
parser.add_argument("--n_estimators", type=int, default=200, help="number of boosting rounds")
parser.add_argument("--max_depth", type=int, default=4, help="maximum tree depth")
parser.add_argument("--xgb_lr", type=float, default=0.05, help="XGBoost learning rate")
parser.add_argument("--subsample", type=float, default=0.8, help="subsample ratio of training instances")
parser.add_argument("--colsample_bytree", type=float, default=0.8, help="subsample ratio of columns per tree")
parser.add_argument("--min_child_weight", type=int, default=5, help="minimum sum of instance weight in a child")
parser.add_argument("--early_stopping_rounds", type=int, default=30, help="stop if no improvement for N rounds")
parser.add_argument("--max_bin", type=int, default=64, help="max bins for hist — reduce to save RAM")
parser.add_argument("--nthread", type=int, default=2, help="CPU threads for XGBoost (limit to avoid OOM)")

# Device
parser.add_argument("--use_gpu", action="store_true", help="use GPU (CUDA) for XGBoost")
parser.add_argument("--gpu", type=int, default=0, help="gpu index")

args = parser.parse_args()

if args.prediction_length is None:
    args.prediction_length = args.sequence_length

if args.use_gpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    args.use_gpu = torch.cuda.is_available()

_csv_files = sorted(f for f in os.listdir(args.root_path) if f.endswith(".csv"))

print("Args in experiment:")
print(args)
print(f"Found {len(_csv_files)} CSV files in {args.root_path}")

train_setting = "tp{}_sl{}".format(
    args.root_path.rstrip("/").split(os.sep)[-1], args.sequence_length
)
all_mse = {f: [] for f in _csv_files}

for itr_idx in range(args.itr):
    print(f"\n\n>>>>>>> itr {itr_idx + 1}/{args.itr} — training on {len(_csv_files)} assets >>>>>>>")
    runner = XGBRunner(args)
    runner.train(train_setting, _csv_files)

    for file in _csv_files:
        args.data_path = file
        test_setting = file + "_" + train_setting
        print(f">>>>>>>start testing : {test_setting}>>>>>>>>>>>>>>>>>>>>>>>>>>")
        mse = runner.test(test_setting)
        all_mse[file].append(mse)

results_rows = []
for file in _csv_files:
    ticker = os.path.splitext(file)[0]
    results_rows.append({
        "Ticker": ticker,
        "MSE": np.mean(all_mse[file]),
        "StdDev": np.std(all_mse[file]),
    })

results = pd.DataFrame(results_rows, columns=["Ticker", "MSE", "StdDev"])
results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(results_dir, exist_ok=True)
results.to_csv(os.path.join(results_dir, train_setting + ".csv"), index=False)
print(results)
