# Ensemble: LightGBM base + Transformer residual model

Design:
- Train a LightGBM base regressor to predict `next_day_return` (scripts/train_base_lightgbm.py).
- Compute residuals: residual = true_return - base_pred. Produce `data/train_resid.csv`/`data/val_resid.csv` where `next_day_return` is replaced by residual.
- Train the lightweight Transformer on residual datasets using the Python trainer (`scripts/train_transformer.py`).
- Generate predictions for base and transformer on the validation set and fit a small linear fusion model to combine them (scripts/fuse_models.py).

Notes & next steps:
- The repository now includes `scripts/predict_transformer.py` to export per-sample transformer predictions to `artifacts/trans_pred_val.csv` and `artifacts/trans_pred_train.csv`.
- Fusion can be simple linear regression or a learned blend saved in `artifacts/fusion_weights.json`.
- Transformer training moved to Python for easier iteration; the change keeps GPU usage via PyTorch if available.

Quick run (best-effort end-to-end):
1. ./scripts/run_ensemble_train.sh
2. If `artifacts/trans_pred_val.csv` is missing, run batch_predict targeted at the val time range to generate it.
3. python3 scripts/fuse_models.py --base-val artifacts/base_pred_val.csv --trans-val artifacts/trans_pred_val.csv --val-truth data/val.csv

GPU tips & throughput: ✅
- Use CUDA-enabled PyTorch for transformer training (the scripts auto-select GPU when available).
- Reduce `--d-model`, `--n-layers`, and `--n-head` for memory-constrained GPUs (defaults are small: `d_model=32`, `n_layers=1`, `n_head=2`).
- Increase `--batch` as GPU memory allows to improve throughput; prefer power-of-two batch sizes.
- Use `--lr-scheduler cosine` for stable long runs and `--resume` with `--checkpoint` to continue interrupted training.

Developer setup (server/CI) 🛠️
- Create and activate a Python virtual environment, then install deps:

  ```bash
  python3 -m venv .venv && source .venv/bin/activate
  python -m pip install --upgrade pip
  python -m pip install -r tests/requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
  ```

- System packages: On Ubuntu install OpenMP for LightGBM:
  `sudo apt-get update && sudo apt-get install -y libomp-dev`
  On macOS: `brew install libomp`.

- If you encounter errors with compiled extensions and NumPy 2.x, pin to NumPy 1.x:
  `python -m pip install 'numpy<2' --force-reinstall`

- Quick helper: `scripts/setup_dev_env.sh` automates venv + installs and checks compatibility.

Quick one-liner to pull latest changes and set up the dev environment (safe default):

```bash
git pull origin main && ./scripts/setup_dev_env.sh
```

If you'd rather avoid system package installation on the server, run the one-liner with `--no-system-deps`:

```bash
git pull origin main && ./scripts/setup_dev_env.sh --no-system-deps
```

If you'd like, I can:
- Add a small Rust/Python tool to run the transformer checkpoint and write per-sample predictions to CSV automatically (recommended), or
- Change the transformer to be trained in Python so the entire stack is Pythonic.
