# -*-Encoding: utf-8 -*-
"""
ARIMA per-ticker fitting and walk-forward evaluation.

Strategy:
  1. Auto-ARIMA(p, 0, q) on training data (d=0: log-returns are stationary).
     Order selected by AIC via stepwise search (pmdarima).
  2. Walk-forward test evaluation that mirrors D-Va exactly:
     - Same 70 / 10 / 20 split (int multiples, matching data_loader.py).
     - Same align_target=True convention: for input window ending at index
       s_end, the forecast origin is s_end-1 (r_begin = s_end - 1).
  3. Parameters are fitted once on training data; SARIMAX.apply(history,
     refit=False) runs the Kalman filter on the expanding history without
     re-estimating parameters — ~100x faster than refitting every window.
"""
import warnings
import numpy as np
from pmdarima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX


def _split_sizes(n):
    """Replicate data_loader.py split: int(n*0.7) / int(n*0.1) / int(n*0.2)."""
    n_train = int(n * 0.7)
    n_val   = int(n * 0.1)
    n_test  = int(n * 0.2)
    return n_train, n_val, n_test


def find_best_order(train_series, max_p=5, max_q=5):
    """
    Return best ARIMA(p, 0, q) order for a stationary series.
    Uses stepwise AIC search; falls back to (1, 0, 0) if search fails.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            model = auto_arima(
                train_series,
                d=0, D=0,
                max_p=max_p, max_q=max_q,
                max_P=0, max_Q=0,
                information_criterion="aic",
                stepwise=True,
                error_action="ignore",
                suppress_warnings=True,
            )
            return model.order
        except Exception:
            return (1, 0, 0)


def evaluate_ticker(series, seq_len, pred_len, max_p=5, max_q=5, verbose=False):
    """
    Walk-forward ARIMA evaluation for one ticker.

    Parameters
    ----------
    series   : 1-D array-like, the target_return column (raw, not standardized).
    seq_len  : input sequence length (used only to determine window boundaries,
               same value as D-Va's --sequence_length).
    pred_len : forecast horizon (same as D-Va's --prediction_length).

    Returns
    -------
    preds       : (n_windows, pred_len) — forecasts in standardized space
    trues       : (n_windows, pred_len) — actuals in standardized space
    target_mean : float — train-set mean of target (for de-standardization)
    target_std  : float — train-set std  of target (for de-standardization)
    order       : tuple (p, d, q) selected by auto-ARIMA
    """
    series = np.asarray(series, dtype=float)
    n = len(series)
    n_train, n_val, n_test = _split_sizes(n)

    # Standardize with train statistics (same convention as D-Va)
    train_raw    = series[:n_train]
    target_mean  = float(train_raw.mean())
    target_std   = float(train_raw.std()) or 1.0
    series_std   = (series - target_mean) / target_std

    # Step 1: find best ARIMA order on training data
    order = find_best_order(series_std[:n_train], max_p=max_p, max_q=max_q)
    if verbose:
        print(f"    auto-ARIMA order: {order}")

    # Step 2: fit ARIMA on training data — parameters fixed for all test windows
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            base_fit = SARIMAX(series_std[:n_train], order=order).fit(disp=False)
        except Exception:
            # Degenerate fallback: constant-zero forecast (predict mean)
            n_windows = n_test - pred_len + 1
            preds = np.zeros((n_windows, pred_len))
            trues_out = np.zeros((n_windows, pred_len))
            for i in range(n_windows):
                r_begin = n_train + n_val + i - 1
                trues_out[i] = series_std[r_begin : r_begin + pred_len]
            return preds, trues_out, target_mean, target_std, order

    # Step 3: walk-forward evaluation matching D-Va's test windows
    #
    # D-Va test slice: data[n_train + n_val - seq_len : n_train + n_val + n_test]
    # Within the slice, align_target=True: r_begin (in slice) = i + seq_len - 1
    # => r_begin (absolute) = n_train + n_val - seq_len + i + seq_len - 1
    #                       = n_train + n_val + i - 1
    #
    # n_windows = n_test - pred_len + 1   (matches D-Va __len__)
    n_windows  = n_test - pred_len + 1
    preds      = np.zeros((n_windows, pred_len))
    pred_stds  = np.zeros((n_windows, pred_len))
    trues_out  = np.zeros((n_windows, pred_len))

    for i in range(n_windows):
        r_begin = n_train + n_val + i - 1
        r_end   = r_begin + pred_len
        history = series_std[:r_begin]
        trues_out[i] = series_std[r_begin:r_end]

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = base_fit.apply(history, refit=False)
                fc  = res.get_forecast(steps=pred_len)
                preds[i]     = fc.predicted_mean.values
                pred_stds[i] = fc.se_mean.values  # std del forecast en cada horizonte
        except Exception:
            preds[i]    = 0.0
            pred_stds[i] = 1.0  # incertidumbre maxima como fallback

    return preds, pred_stds, trues_out, target_mean, target_std, order
