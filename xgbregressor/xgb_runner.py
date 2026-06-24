# -*-Encoding: utf-8 -*-
import os
import sys
import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def _make_windows(data_x, data_y, seq_len, pred_len):
    """Build all sliding windows in one vectorized numpy operation."""
    n = len(data_x) - seq_len - pred_len + 1
    # indices para X: (n, seq_len) y para Y: (n, pred_len)
    idx_x = np.arange(seq_len)[None, :] + np.arange(n)[:, None]       # (n, seq_len)
    idx_y = np.arange(seq_len - 1, seq_len - 1 + pred_len)[None, :] + np.arange(n)[:, None]  # (n, pred_len)
    X = data_x[idx_x]          # (n, seq_len, n_feat)
    y = data_y[idx_y]          # (n, pred_len, 1)
    return X, y


def _load_csv(root_path, data_path, seq_len, pred_len):
    """
    Replica la logica de Dataset_Custom pero devuelve arrays numpy directamente,
    sin iteracion Python por muestra.
    """
    df = pd.read_csv(os.path.join(root_path, data_path))
    length = len(df)
    num_train = int(length * 0.7)
    num_vali  = int(length * 0.1)
    num_test  = int(length * 0.2)

    border1s = [0, num_train - seq_len, num_train + num_vali - seq_len]
    border2s = [num_train, num_train + num_vali, num_train + num_vali + num_test]

    all_cols   = list(df.columns[1:])
    target_col = all_cols[-1]
    feat_cols  = all_cols[:-1]

    feat_vals   = df[feat_cols].values.astype(np.float32)
    target_vals = df[[target_col]].values.astype(np.float32)

    # Escalar features con estadisticas de train (igual que StandardScaler en data_loader)
    feat_mean = feat_vals[border1s[0]:border2s[0]].mean(0)
    feat_std  = feat_vals[border1s[0]:border2s[0]].std(0)
    feat_std  = np.where(feat_std < 1e-8, 1.0, feat_std)
    feat_scaled = (feat_vals - feat_mean) / feat_std

    # Escalar target con estadisticas de train
    target_train = target_vals[border1s[0]:border2s[0]]
    target_mean  = target_train.mean(0)
    target_std   = target_train.std(0).clip(min=1e-8)
    target_scaled = (target_vals - target_mean) / target_std

    splits = {}
    for flag, b1, b2 in zip(["train", "val", "test"], border1s, border2s):
        data_x = feat_scaled[b1:b2]
        data_y = target_scaled[b1:b2]
        X_wins, y_wins = _make_windows(data_x, data_y, seq_len, pred_len)
        # X_wins: (n, seq_len, n_feat) → aplanar a (n, seq_len*n_feat)
        n = X_wins.shape[0]
        X_flat = X_wins.reshape(n, -1)
        y_flat = y_wins.reshape(n, -1)       # (n, pred_len)
        inp    = X_wins[:, :, -1:]            # (n, seq_len, 1) — ultima feature
        splits[flag] = (X_flat, y_flat, inp, target_mean, target_std)

    return splits


class XGBRunner:
    def __init__(self, args):
        self.args = args
        self.model = None

    def _get_split(self, flag, data_path=None):
        if data_path is None:
            data_path = self.args.data_path
        splits = _load_csv(
            self.args.root_path, data_path,
            self.args.sequence_length, self.args.prediction_length,
        )
        return splits[flag]

    def _collect_multi(self, flag, csv_files):
        X_all, y_all = [], []
        for f in csv_files:
            X, y, _, _, _ = self._get_split(flag, f)
            X_all.append(X)
            y_all.append(y)
        return np.concatenate(X_all), np.concatenate(y_all)

    def train(self, setting, csv_files=None):
        if csv_files:
            X_train, y_train = self._collect_multi("train", csv_files)
            X_val,   y_val   = self._collect_multi("val",   csv_files)
        else:
            X_train, y_train, _, _, _ = self._get_split("train")
            X_val,   y_val,   _, _, _ = self._get_split("val")

        print(f"Train: {X_train.shape[0]} samples | Val: {X_val.shape[0]} samples | "
              f"Features: {X_train.shape[1]} | Outputs: {y_train.shape[1]}")

        self.model = xgb.XGBRegressor(
            n_estimators=self.args.n_estimators,
            max_depth=self.args.max_depth,
            learning_rate=self.args.xgb_lr,
            subsample=self.args.subsample,
            colsample_bytree=self.args.colsample_bytree,
            min_child_weight=self.args.min_child_weight,
            tree_method="hist",
            max_bin=self.args.max_bin,
            nthread=self.args.nthread,
            device="cuda" if self.args.use_gpu else "cpu",
            early_stopping_rounds=self.args.early_stopping_rounds,
            eval_metric="rmse",
            verbosity=0,
            random_state=42,
        )
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        print(f"  Best iteration: {self.model.best_iteration}  "
              f"Val RMSE: {self.model.best_score:.6f}")

    def test(self, setting):
        X_test, y_test, inp, target_mean, target_std = self._get_split("test")
        preds = self.model.predict(X_test)

        pred_len   = self.args.prediction_length
        target_dim = self.args.target_dim
        preds = preds.reshape(-1, pred_len, target_dim)
        trues = y_test.reshape(-1, pred_len, target_dim)

        mse = np.mean((preds - trues) ** 2)
        print(f"mse:{mse}")

        folder_path = os.path.join(RESULTS_DIR, setting)
        os.makedirs(folder_path, exist_ok=True)
        np.save(os.path.join(folder_path, "pred.npy"),        preds)
        np.save(os.path.join(folder_path, "true.npy"),        trues)
        np.save(os.path.join(folder_path, "noisy.npy"),       preds)
        np.save(os.path.join(folder_path, "input.npy"),       inp)
        np.save(os.path.join(folder_path, "target_mean.npy"), np.asarray(target_mean))
        np.save(os.path.join(folder_path, "target_std.npy"),  np.asarray(target_std))
        return mse
