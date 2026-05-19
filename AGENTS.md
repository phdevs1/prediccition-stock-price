Purpose
Help future OpenCode agents run experiments and avoid common pitfalls in this repo.

Quick environment
- Python: the repo targets Python 3.8 (see .python-version).
- Install runtime deps: pip install -r requirements.txt (requirements pin torch==1.11.0, gluonts==0.11.7, pandas==1.4.1).

Main entrypoints
- Run full experiment sweep (iterates every CSV in a directory):
  python main.py --root_path data/2016 --sequence_length 10
  run.sh contains examples that run the script for different years/sequence lengths.
- main.py will loop over every file in --root_path and treat each CSV as a ticker.

Data format and loader behaviour (must-not-miss)
- CSVs must have a date column as the first column and numeric feature columns afterwards.
- Dataset picks columns df_raw.columns[1:] (it ignores the first column) and uses the last feature column as the target (data[..., -1]).
- Train/val/test split is deterministic inside Dataset_Custom: 70% train, 10% val, 20% test (computed from file length). The dataset uses sliding windows; __len__ returns len(data_x) - seq_len - pred_len + 1.

Important arguments and defaults
- --sequence_length (default 10) and --prediction_length (default None -> equals sequence_length).
- --input_dim default 6; ensure CSV has matching number of feature columns used by the model.
- --batch_size default 16; DataLoader uses drop_last=True so batches will always be full size.
- --checkpoints default ./checkpoints/; checkpoints are saved to ./checkpoints/{setting}/checkpoint.pth by EarlyStopping.
- Results: after a full run main.py writes ./results/tp<root_dir_name>_sl<sequence_length>.csv containing per-ticker MSE and StdDev.

GPU / CUDA notes (common source of confusion)
- The code sets args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False. It also sets CUDA_VISIBLE_DEVICES=os.environ inside Exp_Model when use_gpu is True.
- Don't rely on passing --use_gpu False on the command line: argparse uses type=bool which behaves oddly for string values (e.g. "False" is truthy). To force CPU reliably, run with CUDA hidden in the environment:
  CUDA_VISIBLE_DEVICES="" python main.py ...
- To select GPU index use --gpu N (the code sets CUDA_VISIBLE_DEVICES to that index before creating devices).

Checkpoints and running test
- Training creates a folder at {checkpoints}/{setting}/ and EarlyStopping writes checkpoint.pth there. After training the code immediately loads {path}/checkpoint.pth; if that file is missing test will fail. Ensure training completed and produced the checkpoint before running test code.

How to run a single ticker/file
- main.py does not accept a single-file arg from CLI. Options:
  1) Point --root_path at a directory that contains only the single CSV you want to run.
  2) Temporarily edit main.py to set args.root_path and args.data_path for debugging (small, clear change).

Common gotchas and pitfalls
- The dataset expects a 'date' column (or at least a parsable first column) and numeric feature columns — mismatched CSV schema will cause crashes.
- If you modify sequence/prediction lengths, verify Dataset_Custom slicing logic and that seq_len + pred_len < file length for the CSVs used.
- The code expects full batches (drop_last=True) and uses torch.randint with shape (batch_size,) — do not set batch_size larger than what the data can produce without checking drop_last behaviour.
- EarlyStopping will save the best model at checkpoint.pth; training assumes that file exists afterwards (no fallback).

Where to look next
- main.py (top-level orchestration)
- exp/exp_model.py (training loop, device selection, checkpoints)
- data_load/data_loader.py (file format, split, sliding-window logic)
- run.sh (useful example invocations)

If anything here is unclear and you need team conventions (PR/branch rules, long-running GPU queue, dataset provenance), ask a human — the repo has no additional automation or CI documented.
