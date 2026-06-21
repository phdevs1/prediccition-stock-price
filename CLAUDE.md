# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Diffusion Variational Autoencoder (D-Va)** for multi-step regression stock price prediction. The model combines a bidirectional VAE with a Gaussian diffusion process to handle stochasticity in financial time series.

## Environment Setup

- **Python 3.8** (pinned in `.python-version`)
- **PyTorch 2.1.2** with CUDA 12.1 (installed separately — see `install.sh`):
  ```
  pip install torch==2.1.2 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
  pip install -r requirements.txt
  ```
- Key pinned deps: `gluonts==0.11.7`, `pandas==1.4.1`

## Running Experiments

```bash
# Full experiment sweep over all CSVs in a directory
python main.py --root_path data/2016 --sequence_length 10

# Force CPU (the --use_gpu flag uses type=bool, so "False" is truthy — use env var instead)
CUDA_VISIBLE_DEVICES="" python main.py --root_path data/2016 --sequence_length 10

# Select specific GPU
python main.py --root_path data/2016 --sequence_length 10 --gpu 1
```

`run.sh` contains the full sweep over years (2016, 2019, 2022) × sequence lengths (10, 20, 40, 60).

**To run a single ticker:** point `--root_path` at a directory containing only that one CSV.

## Architecture

```
main.py                        # CLI entrypoint, loops over CSVs, aggregates results
exp/exp_model.py (Exp_Model)   # Training loop, validation, test, checkpoint management
model/model.py                 # Three nn.Module classes (see below)
model/diffusion_process.py     # GaussianDiffusion, beta schedules
model/encoder.py               # Bidirectional VAE encoder (NVAE-style)
model/resnet.py                # Res12_Quadratic score network
model/embedding.py             # DataEmbedding (temporal + value)
model/neural_operations.py     # Building blocks for encoder/decoder cells
data_load/data_loader.py       # Dataset_Custom, StandardScaler
utils/timefeatures.py          # Converts DatetimeIndex to cyclical time features
utils/tools.py                 # EarlyStopping, adjust_learning_rate
```

### Three model classes in `model/model.py`

| Class | Role |
|---|---|
| `diffusion_generate` | GRU → concat → `GaussianDiffusion.log_prob`; produces distribution + noisy sample |
| `denoise_net` | Wraps `diffusion_generate` + `DataEmbedding` + `Res12_Quadratic` score net; computes DSM loss during training |
| `pred_net` | Inherits `denoise_net`; inference path — runs deterministic decoder then applies score gradient correction |

`Exp_Model` initializes all three, trains `denoise_net`, copies weights to `pred_net` via `gluonts.torch.util.copy_parameters` for validation/test.

### Training loss

`loss = MSE + zeta * KL + eta * DSM`

where DSM is the denoising score-matching loss from the `Res12_Quadratic` energy net.

## Data Format

CSVs must have:
- Column 0: `date` (parseable by `pd.DatetimeIndex`)
- Columns 1–N: numeric features (all used); **last column is the prediction target**

`Dataset_Custom` split: 70% train / 10% val / 20% test (deterministic, computed from file length). Sliding windows of `seq_len + pred_len`; `__len__` = `len(data) - seq_len - pred_len + 1`.

`--input_dim` (default 6) must match the number of feature columns in the CSV.

## Data Acquisition

Raw data is downloaded with `idea/get_yahoo.py` (uses `yfinance`) into `pre-processing/raw/`. The processed CSVs in `data/2016/`, `data/2019/`, `data/2022/` are what `main.py` consumes — each subdirectory holds one CSV per ticker for that test period.

## Outputs

- **Checkpoints**: `./checkpoints/{data_path}_{setting}/checkpoint.pth` (written by `EarlyStopping`)
- **Per-run arrays**: `./results/{setting}/pred.npy`, `true.npy`, `noisy.npy`, `input.npy`
- **Aggregated results**: `./results/tp{year}_sl{seq_len}.csv` — per-ticker MSE and StdDev across `--itr` runs (default 5)

## Portfolio Analysis (`portfolio_analysis.py`)

Downstream analysis script that reads `results/` after a full experiment run. It:
1. Loads `pred.npy` / `true.npy` from all per-ticker result folders matching a glob pattern (default `*_tp2016_sl10`)
2. Builds a Markowitz mean-variance portfolio using model predictions as expected returns, with graphical lasso regularization (`sklearn`) on the covariance matrix
3. Outputs portfolio weights, returns, and plots

Run it after `main.py` completes for a given `(year, seq_len)` combination.

## Key Gotchas

- `batch_size` must be small enough that `drop_last=True` still produces at least one batch for every split.
- After training, `Exp_Model.train()` immediately loads `checkpoint.pth` — if early stopping never triggered (no improvement), the file may not exist and will crash.
- `prediction_length` defaults to `None` and is set equal to `sequence_length` in `main.py`; passing it explicitly on the CLI bypasses that logic.
- `torch.randint(0, diff_steps, (batch_size,))` in the training loop uses the fixed `batch_size` arg, not the actual batch length — do not use `drop_last=False`.
