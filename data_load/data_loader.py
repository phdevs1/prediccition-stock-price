# -*-Encoding: utf-8 -*-
import os
import pandas as pd

import torch
from torch.utils.data import Dataset
from utils.timefeatures import time_features
import warnings

warnings.filterwarnings("ignore")


class StandardScaler(object):
    def __init__(self):
        self.mean = 0.0
        self.std = 1.0

    def fit(self, data):
        self.mean = data.mean(0)
        self.std = data.std(0)

    def _coerce(self, arr, ref):
        if torch.is_tensor(ref):
            return torch.from_numpy(arr).type_as(ref).to(ref.device)
        return arr

    def transform(self, data):
        return (data - self._coerce(self.mean, data)) / self._coerce(self.std, data)

    def inverse_transform(self, data):
        return data * self._coerce(self.std, data) + self._coerce(self.mean, data)


class Dataset_Custom(Dataset):
    def __init__(
        self,
        root_path,
        flag="train",
        size=None,
        data_path="AAPL.csv",
    ):
        # size [seq_len, pred_len]
        self.seq_len = size[0]
        self.pred_len = size[1]
        assert flag in ["train", "test", "val"]
        type_map = {"train": 0, "val": 1, "test": 2}
        self.set_type = type_map[flag]

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        length = len(df_raw)
        num_train = int(length * 0.7)
        num_test = int(length * 0.2)
        num_vali = int(length * 0.1)

        border1s = [0, num_train - self.seq_len, num_train + num_vali - self.seq_len]
        border2s = [num_train, num_train + num_vali, num_train + num_vali + num_test]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        all_cols = list(df_raw.columns[1:])  # todas menos 'date'
        target_col = all_cols[-1]  # target_return (ultima columna)
        feature_cols = all_cols[:-1]  # features de entrada, sin target_return

        df_features = df_raw[feature_cols]
        df_target = df_raw[[target_col]]

        # StandardScaler ajustado SOLO en train (features) y aplicado a toda la serie.
        self.scaler.fit(df_features[border1s[0] : border2s[0]].values)
        data_x = self.scaler.transform(df_features.values)

        # Target estandarizado con su propia estadistica de train. Guardamos
        # mean/std para poder invertir a retornos reales (portafolio / metricas).
        target_train = df_target[border1s[0] : border2s[0]].values
        self.target_mean = target_train.mean(0)
        self.target_std = target_train.std(0)
        self.target_std = self.target_std.clip(min=1e-8)  # evita division por cero
        data_y = (df_target.values - self.target_mean) / self.target_std

        df_stamp = pd.DatetimeIndex(df_raw[border1:border2]["date"])
        data_stamp = time_features(df_stamp)

        self.data_x = data_x[border1:border2]
        self.data_y = data_y[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - 1  # target_return[s_end-1]: primer retorno del horizonte
        r_end = r_begin + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    @staticmethod
    def feature_dim(root_path, data_path):
        """Numero de features de entrada para un CSV dado (excluye date y target_return)."""
        sample = pd.read_csv(os.path.join(root_path, data_path), nrows=1)
        return sum(1 for c in sample.columns if c.lower() not in ("date", "target_return"))
