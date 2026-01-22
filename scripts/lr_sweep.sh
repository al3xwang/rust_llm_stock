#!/usr/bin/env bash
set -euo pipefail

# Learning rate sweep script
# Usage: ./scripts/lr_sweep.sh 1e-5 3e-5 1e-4
# Default list if no args provided
LRS=(${@:-1e-5 3e-5 1e-4 3e-4 1e-3})

mkdir -p logs
mkdir -p artifacts

OUT_CSV=logs/lr_sweep_results.csv
echo "lr,best_valid_loss,model_path,log" > "$OUT_CSV"

for lr in "${LRS[@]}"; do
  echo "=== Running LR=$lr ==="
  OUT_DIR=artifacts/lr_${lr}
  mkdir -p "$OUT_DIR"
  LOG=logs/lr_${lr}.log

  # Run training (sequential to avoid GPU contention)
  cargo run --release --features pytorch --bin train -- --train data/train.csv --val data/val.csv --device cuda --lr "${lr}" --out "$OUT_DIR" > "$LOG" 2>&1 || true

  # Extract best validation loss and model path
  BEST_LINE=$(grep -E "Best validation loss" -m 1 "$LOG" || true)
  if [[ -n "$BEST_LINE" ]]; then
    BEST_LOSS=$(echo "$BEST_LINE" | awk -F":" '{print $2}' | xargs)
  else
    BEST_LOSS="NA"
  fi
  MODEL_PATH="$OUT_DIR/best_model.safetensors"

  echo "${lr},${BEST_LOSS},${MODEL_PATH},${LOG}" >> "$OUT_CSV"
  echo "LR=$lr done. Best loss: ${BEST_LOSS}"
  sleep 5
done

echo "Sweep finished. Results in $OUT_CSV"