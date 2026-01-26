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

If you'd like, I can:
- Add a small Rust/Python tool to run the transformer checkpoint and write per-sample predictions to CSV automatically (recommended), or
- Change the transformer to be trained in Python so the entire stack is Pythonic.
