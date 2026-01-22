# Training README

## Overview ✅
This document explains how to run the Rust training binary (`train`) on a server (CPU or GPU). The code has unit tests covering the training helpers (losses, grad clipping, scheduler, weighted losses); the library tests pass locally. Before launching a full run on the server, perform the short dry-run checks below.

---

## Readiness assessment 🟢
- Unit tests (library) pass: run `cargo test --lib --features pytorch`.
- Training loop and sample-weighting integration are implemented and have a small integration test that runs a single forward/backward step.

Caveats:
- Some bin tests require DB credentials or runtime environment and may fail when run on a bare server. This does not affect the `train` binary functionality.
- Confirm libtorch is installed and compatible with your CUDA/driver stack (if using GPU).

---

## Prerequisites (server) 🔧
- Rust toolchain (stable or nightly as used by the repo). Install via rustup.
- libtorch (PyTorch C++ distribution) installed on the server:
  - For CUDA-enabled GPU training, download a libtorch build matching the CUDA version.
  - Unpack and add the lib directory to `LD_LIBRARY_PATH` (Linux) or `DYLD_LIBRARY_PATH` (macOS).
  - Example (bash):

```bash
# Example on Linux
export LD_LIBRARY_PATH=/path/to/libtorch/lib:$LD_LIBRARY_PATH
```

- Ensure GPU drivers and `nvidia-smi` are available for CUDA training.
- `cargo` build needs the `pytorch` feature enabled for binaries that use `tch`.

Optional:
- `tmux` or `screen` for long runs, or create a `systemd` service.

---

## Build (quick) ⚙️
From repository root:

```bash
# Release build (recommended for production runs)
cargo build --release --features pytorch

# Or run directly (debug) for short tests
cargo run --release --bin train --features pytorch -- --help
```

---

## Small dry-run (recommended) 🧪
1. Prepare small train/val CSV files (same schema as `MlTrainingRecord`). You can use `data/train.csv` and `data/val.csv` with a small subset.
2. Run a short test on CPU:

```bash
cargo run --release --bin train --features pytorch -- --train data/sample_train.csv --val data/sample_val.csv --device cpu --batch 16 --max-epochs 2 --out artifacts/test_run
```

3. If GPU available, test `--device cuda:0` with small batch size (e.g., 32) to check memory usage.

---

## Example production commands (suggestions) 🚀
- GPU run (recommended):

```bash

```

- CPU run (small):

```bash
cargo run --release --bin train --features pytorch -- --train /data/train.csv --val /data/val.csv --device cpu --batch 64 --max-epochs 50 --out artifacts/cpu_test
```

---

## Important flags (summary) 🏷️
- `--train <path>`: training CSV (default `data/train.csv`)
- `--val <path>`: validation CSV (default `data/val.csv`)
- `--device <cpu|cuda|cuda:<idx>>`: device to run on
- `--batch <N>`: batch size (default 256)
- `--lr <f>`: override initial learning rate
- `--wd <f>`: weight decay (L2)
- `--huber-delta <f>`: enable Huber loss with the delta parameter (otherwise MSE used)
- `--grad-clip <f>`: global-norm gradient clipping threshold
- `--sample-weight-method <none|time_decay>`: enable sample weighting
- `--sample-weight-decay <f>`: decay rate for time-decay weights (days multiplier)
- `--sample-weight-normalize <none|mean|sum>`: normalization method applied to sample weights
- `--lr-scheduler <none|cosine|cosine_restart>`: learning rate scheduler
- `--lr-min <f>`: minimum LR for cosine annealing
- `--cosine-t-max <N>`: T_max for cosine annealing (or T0 for restart)
- `--t-mult <f>`: multiplier for warm restarts (SGDR)
- `--dropout <f>`: override model dropout
- `--max-epochs <N>`: max epochs
- `--early-stop <N>`: early stopping patience (epochs)
- `--out <path>`: artifact directory for saved models (`best_model.safetensors`)

---

## Checkpoints & artifacts 💾
- Best model saved to `<out_dir>/best_model.safetensors`.
- Periodic checkpoints saved every 10 epochs as `checkpoint_epoch_<n>.safetensors`.
- Logs print per-epoch metrics including training/validation loss, IC (Pearson & Spearman), and TopK stats. Use `train.log` to monitor.

Note: Resuming training from a checkpoint is not currently automated by the CLI; you can still load saved weights by adding a small loader (I can add a `--resume <path>` flag if you'd like).

---

## Monitoring & tips 📈
- Use `nvidia-smi` to monitor GPU usage.
- Start with small batch sizes if GPU memory is unknown and increase gradually.
- Watch the printed "Estimated peak batch memory" line at start of training.

---

## Troubleshooting 🛠️
- "libtorch not found" / `libtorch.so` loader errors: ensure `LD_LIBRARY_PATH` points to `libtorch/lib`.
- CUDA not found / `Device::Cuda` not available: check driver and `nvidia-smi`.
- If bin tests fail during `cargo test --features pytorch`, run library tests only: `cargo test --lib --features pytorch`.

---

## Validation & sanity checks before long run ✅
1. Run `cargo test --lib --features pytorch` (should pass all tests in `src/training_torch.rs`).
2. Start a small GPU test to validate memory and logs.
3. Inspect `train.log` for NaNs and early indications.

---

## Would you like me to also:
- Add a `--resume <checkpoint>` CLI option and loader? (Yes/No)
- Create a `systemd` unit or a `run_train.sh` wrapper to run supervised on your server? (Yes/No)

---

## Contact
If you want, I can also commit a short `run_train.sh` wrapper and a `systemd` unit file tuned for your server.

---

Good to go? If you want, I can now add a `--resume` flag and a short wrapper script for running under `tmux`/`systemd`.