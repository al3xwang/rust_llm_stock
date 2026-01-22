#!/usr/bin/env bash
set -euo pipefail

# Hyperparameter sweep: LR x batch x weight_decay
# Usage: ./scripts/hyper_sweep.sh

LRS=(1e-5 3e-5 1e-4)
BATCHES=(64 128 256)
WDS=(0.0 1e-6 1e-5)

mkdir -p logs
mkdir -p artifacts
OUT_CSV=logs/hyper_sweep_results.csv
echo "lr,batch,wd,best_valid_loss,model_path,log" > "$OUT_CSV"

for lr in "${LRS[@]}"; do
  for batch in "${BATCHES[@]}"; do
    for wd in "${WDS[@]}"; do
      echo "=== LR=${lr} BATCH=${batch} WD=${wd} ==="
      OUT_DIR=artifacts/lr_${lr}_bs_${batch}_wd_${wd}
      mkdir -p "$OUT_DIR"
      LOG=logs/lr_${lr}_bs_${batch}_wd_${wd}.log

      cargo run --release --features pytorch --bin train -- --train data/train.csv --val data/val.csv --device cuda --lr "${lr}" --batch "${batch}" --wd "${wd}" --out "$OUT_DIR" > "$LOG" 2>&1 || true

      BEST_LINE=$(grep -E "Best validation loss" -m 1 "$LOG" || true)
      if [[ -n "$BEST_LINE" ]]; then
        BEST_LOSS=$(echo "$BEST_LINE" | awk -F":" '{print $2}' | xargs)
      else
        BEST_LOSS="NA"
      fi
      MODEL_PATH="$OUT_DIR/best_model.safetensors"

      echo "${lr},${batch},${wd},${BEST_LOSS},${MODEL_PATH},${LOG}" >> "$OUT_CSV"
      echo "Done: lr=${lr} batch=${batch} wd=${wd} -> ${BEST_LOSS}"
      sleep 3
    done
  done
done

echo "Sweep finished. Results in $OUT_CSV"