# -*-Encoding: utf-8 -*-
import numpy as np
import pandas as pd


def time_features(dates) -> np.ndarray:
    """Features temporales para datos diarios de mercado.

    Devuelve array (N, 3) con DayOfWeek, DayOfMonth, MonthOfYear,
    cada uno normalizado a [-0.5, 0.5].
    """
    idx = pd.DatetimeIndex(pd.to_datetime(dates.values))
    return np.column_stack([
        idx.dayofweek / 6.0 - 0.5,           # 0=Lun, 4=Vie en mercado
        (idx.day - 1) / 30.0 - 0.5,
        (idx.month - 1) / 11.0 - 0.5,
    ])
