import numpy as np
import pandas as pd


class TimeFeature:
    def __init__(self):
        pass

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        pass

    def __repr__(self):
        return self.__class__.__name__ + "()"


class MonthOfYear(TimeFeature):
    """Mes del año, codificado como entero 0-11 (compatible con month_embed,
    nn.Embedding(13, d_model))."""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.month - 1).to_numpy()


class DayOfMonth(TimeFeature):
    """Día del mes, codificado como entero 0-30 (compatible con day_embed,
    nn.Embedding(32, d_model))."""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.day - 1).to_numpy()


class DayOfWeek(TimeFeature):
    """Día de la semana, codificado como entero 0-6, Lunes=0 (compatible con
    weekday_embed, nn.Embedding(7, d_model)). En datos de mercado solo
    tomara valores 0-4 (Lunes-Viernes), lo cual es valido."""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.dayofweek.to_numpy()


def time_features(dates, freq="t"):

    dates = pd.DatetimeIndex(pd.to_datetime(dates.values))

    features = [MonthOfYear(), DayOfMonth(), DayOfWeek()]
    return (
        np.vstack([feat(dates) for feat in features]).transpose(1, 0).astype(np.int64)
    )
