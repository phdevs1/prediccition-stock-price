# -*-Encoding: utf-8 -*-
import os
import numpy as np
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

    def transform(self, data):
        mean = (
            torch.from_numpy(self.mean).type_as(data).to(data.device)
            if torch.is_tensor(data)
            else self.mean
        )
        std = (
            torch.from_numpy(self.std).type_as(data).to(data.device)
            if torch.is_tensor(data)
            else self.std
        )
        return (data - mean) / std

    def inverse_transform(self, data):
        mean = (
            torch.from_numpy(self.mean).type_as(data).to(data.device)
            if torch.is_tensor(data)
            else self.mean
        )
        std = (
            torch.from_numpy(self.std).type_as(data).to(data.device)
            if torch.is_tensor(data)
            else self.std
        )
        return data * std + mean


class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag="train", size=None, data_path="AAPL.csv"):
        # size [seq_len, label_len, pred_len]
        # info
        self.seq_len = size[0]
        self.pred_len = size[1]
        # init
        assert flag in ["train", "test", "val"]
        type_map = {"train": 0, "val": 1, "test": 2}
        self.set_type = type_map[flag]

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        length = len(df_raw)
        num_train = int(length * 0.7)
        num_test = int(length * 0.2)
        num_vali = int(length * 0.1)

        border1s = [0, num_train - self.seq_len, num_train + num_vali - self.seq_len]
        border2s = [num_train, num_train + num_vali, num_train + num_vali + num_test]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        feature_cols = df_raw.columns[1:-1].tolist()
        target_col = df_raw.columns[-1:].tolist()

        df_x = df_raw[feature_cols]
        df_y = df_raw[target_col]

        # scaler features
        self.scaler = StandardScaler()
        self.scaler.fit(df_x.iloc[:num_train].values)
        data_x = self.scaler.transform(df_x.values)

        # scaler target
        self.target_scaler = StandardScaler()
        self.target_scaler.fit(df_y.iloc[:num_train].values)
        data_y = self.target_scaler.transform(df_y.values)

        df_stamp = pd.DatetimeIndex(df_raw[border1:border2]["date"])
        data_stamp = time_features(df_stamp)

        self.data_x = data_x[border1:border2]
        self.data_y = data_y[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end, -1:]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1
