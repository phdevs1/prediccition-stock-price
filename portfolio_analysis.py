import numpy as np
import pandas as pd
from glob import glob
import os
from sklearn.covariance import graphical_lasso
from scipy.optimize import minimize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

RESULTS_DIR = './results'

def load_all_predictions(pattern='*_tp2016_sl10'):
    folders = sorted(glob(os.path.join(RESULTS_DIR, pattern)))
    tickers = []
    preds = []
    trues = []
    for f in folders:
        ticker = os.path.basename(f).replace('_tp2016_sl10', '')
        p = np.load(os.path.join(f, 'pred.npy'))
        t = np.load(os.path.join(f, 'true.npy'))
        tickers.append(ticker)
        preds.append(p)
        trues.append(t)
    P = np.stack(preds, axis=0)
    T = np.stack(trues, axis=0)
    return tickers, P, T

def markowitz_optimize(mu, Sigma, gamma=1.0):
    S = len(mu)
    def objective(w):
        return -(w @ mu - 0.5 * gamma * w @ Sigma @ w)
    constraints = ({'type': 'eq', 'fun': lambda w: w.sum() - 1.0})
    bounds = [(0, 1)] * S
    w0 = np.ones(S) / S
    result = minimize(objective, w0, method='SLSQP',
                      bounds=bounds, constraints=constraints,
                      options={'maxiter': 1000, 'ftol': 1e-12})
    if not result.success:
        result = minimize(objective, w0, method='trust-constr',
                          bounds=bounds, constraints=constraints,
                          options={'maxiter': 2000})
    return result.x

def run_portfolio(P, T, gamma=10.0, lambda_gl=0.1, use_global_cov=True):
    S, N, D, _ = P.shape
    mu_t_all = P[:, :, :, 0].mean(axis=2)
    true_avg_all = T[:, :, :, 0].mean(axis=2)

    if use_global_cov:
        Sigma_raw = np.cov(mu_t_all)
        try:
            _, Theta = graphical_lasso(Sigma_raw, alpha=lambda_gl, max_iter=200)
            Sigma_reg = np.linalg.inv(Theta)
        except:
            Sigma_reg = Sigma_raw + lambda_gl * np.eye(S)
        Sigma_no_reg = Sigma_raw
    else:
        Sigma_reg = None
        Sigma_no_reg = None

    portfolio_rets_reg = np.zeros(N)
    portfolio_rets_no_reg = np.zeros(N)
    weights_reg = np.zeros((N, S))
    weights_no_reg = np.zeros((N, S))

    for t in range(N):
        mu_t = mu_t_all[:, t]
        if use_global_cov:
            w_r = markowitz_optimize(mu_t, Sigma_reg, gamma)
            w_n = markowitz_optimize(mu_t, Sigma_no_reg, gamma)
        else:
            # Per-window covariance (like the paper)
            pt = P[:, t, :, 0]
            Sigma_t = np.cov(pt)
            try:
                _, Theta_t = graphical_lasso(Sigma_t, alpha=lambda_gl, max_iter=200)
                Sigma_reg_t = np.linalg.inv(Theta_t)
            except:
                Sigma_reg_t = Sigma_t + lambda_gl * np.eye(S)
            w_r = markowitz_optimize(mu_t, Sigma_reg_t, gamma)
            w_n = markowitz_optimize(mu_t, Sigma_t, gamma)
        r_r = w_r @ true_avg_all[:, t]
        r_n = w_n @ true_avg_all[:, t]
        portfolio_rets_reg[t] = r_r
        portfolio_rets_no_reg[t] = r_n
        weights_reg[t] = w_r
        weights_no_reg[t] = w_n

    sharpe_reg = portfolio_rets_reg.mean() / portfolio_rets_reg.std()
    sharpe_no_reg = portfolio_rets_no_reg.mean() / portfolio_rets_no_reg.std()
    return (sharpe_reg, portfolio_rets_reg, weights_reg,
            sharpe_no_reg, portfolio_rets_no_reg, weights_no_reg)

def equal_weight_portfolio(T):
    S, N, D, _ = T.shape
    true_avg = T[:, :, :, 0].mean(axis=2)
    w = np.ones(S) / S
    rets = w @ true_avg
    return rets.mean() / rets.std(), rets

def main():
    print('=' * 70)
    print('REPLICACION DE RESULTADOS — D-Va (Paper CIKM 2023)')
    print('Periodo: 2016 | Sequence Length: 10')
    print('=' * 70)

    # ================================================================
    # TABLA 2: MSE Performance Comparison (solo D-Va, solo 2016 T=10)
    # ================================================================
    print('\n' + '-' * 70)
    print('TABLA 2 — MSE Performance (2016, T=10)')
    print('-' * 70)

    df_mse = pd.read_csv(os.path.join(RESULTS_DIR, 'tp2016_sl10.csv'))
    mse_mean = df_mse.MSE.mean()
    mse_std = df_mse.MSE.std()

    print(f'  Our D-Va  |  MSE promedio: {mse_mean:.4f}  |  MSE std: {mse_std:.6f}')
    print(f'  Paper D-Va |  MSE promedio: 0.9040  |  MSE std: 0.0048')
    print(f'  Diferencia: {abs(mse_mean - 0.9040):.4f}')
    print()

    print('  MSE por stock (top/bottom):')
    print(f'    Mejor:  {df_mse.loc[df_mse.MSE.idxmin(), "Ticker"]} = {df_mse.MSE.min():.4f}')
    print(f'    Peor:   {df_mse.loc[df_mse.MSE.idxmax(), "Ticker"]} = {df_mse.MSE.max():.4f}')
    print(f'    Stocks: {len(df_mse)}')

    # ================================================================
    # TABLA 5: Sharpe Ratio Comparison (solo 2016 T=10)
    # ================================================================
    print('\n' + '-' * 70)
    print('TABLA 5 — Sharpe Ratio (2016, T=10) — con Graphical Lasso')
    print('-' * 70)

    tickers, P, T = load_all_predictions()
    S = len(tickers)

    pred_avg_all = P[:, :, :, 0].mean(axis=(1,2))
    true_avg_all = T[:, :, :, 0].mean(axis=(1,2))
    corr_signal = np.corrcoef(pred_avg_all, true_avg_all)[0, 1]

    # Equal-weight (baseline)
    sharpe_ew, ew_rets = equal_weight_portfolio(T)

    # D-Va with global covariance + graphical lasso
    gammas_test = [0.1, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
    best_sharpe_reg = -np.inf
    best_gamma_reg = None
    best_rets_reg = None
    best_w_reg = None
    best_sharpe_noreg = None

    for g in gammas_test:
        s_reg, rets_reg, w_reg, s_noreg, rets_noreg, w_noreg = \
            run_portfolio(P, T, gamma=g, lambda_gl=0.1, use_global_cov=True)
        if s_reg > best_sharpe_reg:
            best_sharpe_reg = s_reg
            best_gamma_reg = g
            best_rets_reg = rets_reg
            best_w_reg = w_reg
            best_sharpe_noreg = s_noreg

    print(f'\n  Modelo                Sharpe (reg)   Sharpe (no reg)')
    print(f'  ' + '-' * 50)
    print(f'  Equal-weight (1/{S})    {sharpe_ew:.4f}        {sharpe_ew:.4f}')
    print(f'  D-Va (γ={best_gamma_reg:<5d})        {best_sharpe_reg:.4f}        {best_sharpe_noreg:.4f}')

    mejora = (best_sharpe_reg - sharpe_ew) / abs(sharpe_ew) * 100
    print(f'  Mejora D-Va vs EW:     {mejora:+.1f}%')

    n_activos = (best_w_reg.mean(axis=0) > 0.001).sum()
    print(f'  Activos > 0.1%:        {n_activos}/{S}')

    # Per-window covariance approach (como el paper)
    print(f'\n  --- Per-window covariance (como el paper) ---')
    best_sharpe_perwindow = -np.inf
    best_g_perwindow = None
    for g in gammas_test:
        s_reg, _, _, _, _, _ = run_portfolio(P, T, gamma=g, lambda_gl=0.1, use_global_cov=False)
        if s_reg > best_sharpe_perwindow:
            best_sharpe_perwindow = s_reg
            best_g_perwindow = g
    print(f'  Mejor D-Va per-window: {best_sharpe_perwindow:.4f} (γ={best_g_perwindow})')

    # ================================================================
    # COMPARACION CON PAPER
    # ================================================================
    print('\n' + '=' * 70)
    print('COMPARACION DIRECTA CON PAPER (2016, T=10)')
    print('=' * 70)

    paper_table2 = {
        'ARIMA':     {'mse': 1.7054, 'std': None},
        'NBA':       {'mse': 1.2012, 'std': 0.1190},
        'VAE':       {'mse': 1.0743, 'std': 0.0255},
        'VAE+Adv':   {'mse': 1.0752, 'std': 0.0249},
        'Autoformer':{'mse': 1.0204, 'std': 0.0179},
        'D-Va':      {'mse': 0.9040, 'std': 0.0048},
    }

    paper_table5 = {
        'ARIMA':      0.0953,
        'NBA':        0.0691,
        'Autoformer': 0.1114,
        'Equal':      0.1089,
        'D-Va':       0.1174,
    }

    print('\n  --- Tabla 2: MSE ---')
    print(f'  {"Modelo":<15s} {"Paper MSE":<12s} {"Nuestro MSE":<12s}')
    print(f'  ' + '-' * 40)
    for model, v in paper_table2.items():
        ours = v['mse'] if model != 'D-Va' else mse_mean
        match = "OK" if abs(ours - v['mse']) < 0.01 else "diff"
        print(f'  {model:<15s} {v["mse"]:<12.4f} {ours:<12.4f}  [{match}]')
    print(f'  {"D-Va Std dev":<15s} {"0.0048":<12s} {mse_std:<12.6f}')

    print(f'\n  --- Tabla 5: Sharpe Ratio ---')
    print(f'  {"Modelo":<15s} {"Paper Sharpe":<14s} {"Nuestro Sharpe":<14s}')
    print(f'  ' + '-' * 44)
    for model, s_paper in paper_table5.items():
        if model == 'Equal':
            ours = sharpe_ew
        elif model == 'D-Va':
            ours = best_sharpe_reg
        else:
            ours = None
        if ours is not None:
            print(f'  {model:<15s} {s_paper:<14.4f} {ours:<14.4f}  [diff={abs(ours-s_paper):.4f}]')
        else:
            print(f'  {model:<15s} {s_paper:<14.4f} {"(N/A)":<14s}')

    # ================================================================
    # FIGURA
    # ================================================================
    print('\nGenerando figura...')

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Replicacion D-Va — Periodo 2016, T=10', fontsize=14, fontweight='bold')

    # Panel A: MSE distribution
    ax = axes[0, 0]
    ax.hist(df_mse.MSE, bins=20, color='steelblue', edgecolor='white')
    ax.axvline(mse_mean, color='#e74c3c', linestyle='--', linewidth=1.5,
               label=f'Media={mse_mean:.4f}')
    ax.axvline(0.9040, color='green', linestyle=':', linewidth=1,
               label=f'Paper=0.9040')
    ax.set_xlabel('MSE')
    ax.set_ylabel('Frecuencia')
    ax.set_title('A) Distribucion de MSE entre stocks')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel B: Predicted vs True returns across stocks (signal check)
    ax = axes[0, 1]
    ax.scatter(true_avg_all, pred_avg_all, s=10, color='steelblue', alpha=0.6)
    xl = ax.get_xlim()
    yl = ax.get_ylim()
    lim = max(max(abs(xl[0]), abs(xl[1])), max(abs(yl[0]), abs(yl[1])))
    ax.plot([-lim, lim], [-lim, lim], 'k--', linewidth=0.5, alpha=0.4)
    ax.set_xlabel('Retorno real promedio (estandarizado)')
    ax.set_ylabel('Retorno predicho promedio (estandarizado)')
    ax.set_title(f'B) Pred vs Real entre stocks (r={corr_signal:.4f})')
    ax.grid(True, alpha=0.3)

    # Panel C: Cumulative returns
    ax = axes[1, 0]
    cum_dva = np.cumprod(1 + best_rets_reg)
    cum_ew = np.cumprod(1 + ew_rets)
    ax.plot(cum_dva, label=f'D-Va (Sharpe={best_sharpe_reg:.4f})', color='#e74c3c', linewidth=1.5)
    ax.plot(cum_ew, label=f'Equal-weight (Sharpe={sharpe_ew:.4f})', color='#3498db', linewidth=1.5)
    ax.axhline(1.0, color='gray', linestyle='--', linewidth=0.5)
    ax.set_xlabel('Ventana de prediccion')
    ax.set_ylabel('Rendimiento acumulado')
    ax.set_title('C) Rendimiento acumulado del portafolio')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel D: Sharpe vs gamma  
    ax = axes[1, 1]
    sharpe_vals = []
    for g in gammas_test:
        s, _, _, _, _, _ = run_portfolio(P, T, gamma=g, lambda_gl=0.1, use_global_cov=True)
        sharpe_vals.append(s)
    ax.plot(gammas_test, sharpe_vals, 'o-', color='#e74c3c', linewidth=1.5, markersize=6)
    ax.axhline(y=sharpe_ew, color='#3498db', linestyle='--',
               label=f'Equal-weight ({sharpe_ew:.4f})')
    ax.axvline(x=best_gamma_reg, color='gray', linestyle=':', linewidth=0.8, alpha=0.6)
    ax.set_xlabel('Gamma (aversion al riesgo)')
    ax.set_ylabel('Sharpe Ratio')
    ax.set_title('D) Sharpe segun gamma')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'replication_2016_T10.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Figura guardada en {RESULTS_DIR}/replication_2016_T10.png')

    # ================================================================
    # GUARDAR RESULTADOS
    # ================================================================
    summary = f"""
REPLICACION D-Va — Periodo 2016, T=10
{'='*60}

TABLA 2 — MSE Performance (solo D-Va, 2016 T=10)
{'-'*60}
MSE promedio (85 stocks): {mse_mean:.4f} ± {mse_std:.6f}
Paper D-Va:               0.9040 ± 0.0048
Diferencia:               {abs(mse_mean - 0.9040):.4f}
Mejor stock:               {df_mse.loc[df_mse.MSE.idxmin(), 'Ticker']} ({df_mse.MSE.min():.4f})
Peor stock:                {df_mse.loc[df_mse.MSE.idxmax(), 'Ticker']} ({df_mse.MSE.max():.4f})

TABLA 5 — Sharpe Ratio (2016, T=10)
{'-'*60}
Correlacion pred vs true (entre stocks): {corr_signal:.4f}
Equal-weight Sharpe:                     {sharpe_ew:.4f}
D-Va (regularized, gamma={best_gamma_reg}):           {best_sharpe_reg:.4f}
D-Va (no regularized):                  {best_sharpe_noreg:.4f}
Mejora D-Va vs EW:                      {mejora:+.1f}%
Activos en portafolio:                  {n_activos}/{S}

Comparacion con paper:
  Paper Equal-weight: 0.1089
  Paper D-Va:         0.1174
  Nuestro Equal-w:    {sharpe_ew:.4f}
  Nuestro D-Va:       {best_sharpe_reg:.4f} (gamma={best_gamma_reg})

  Gamma     Sharpe
  -----------------------------
"""
    for g, s in zip(gammas_test, sharpe_vals):
        _, _, w, _, _, _ = run_portfolio(P, T, gamma=g, lambda_gl=0.1, use_global_cov=True)
        n = (w.mean(axis=0) > 0.001).sum()
        h = (w.mean(axis=0)**2).sum()
        summary += f'  {g:<8.1f}  {s:.4f}     {n:<5d}   {h:.4f}\n'

    with open(os.path.join(RESULTS_DIR, 'replication_results.txt'), 'w') as f:
        f.write(summary)
    print(summary)

if __name__ == '__main__':
    main()
