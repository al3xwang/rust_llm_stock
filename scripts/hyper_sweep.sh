#!/usr/bin/env bash
set -euo pipefail

# Hyperparameter sweep: LR x batch x weight_decay
# Usage: ./scripts/hyper_sweep.sh

# Expanded sweep ranges
LRS=(5e-6 1e-5 3e-5 1e-4 3e-4)
BATCHES=(64 128 256 512)
WDS=(0.0 1e-7 1e-6 1e-5)
HUBERS=("none" 0.1 0.5)
DROPOUTS=(0.0 0.1 0.2)

mkdir -p logs
mkdir -p artifacts
OUT_CSV=logs/hyper_sweep_results.csv
echo "lr,batch,wd,best_valid_loss,model_path,log" > "$OUT_CSV"

for lr in "${LRS[@]}"; do
  for batch in "${BATCHES[@]}"; do
    for wd in "${WDS[@]}"; do
      for h in "${HUBERS[@]}"; do
        for d in "${DROPOUTS[@]}"; do
          echo "=== LR=${lr} BATCH=${batch} WD=${wd} HUBER=${h} DROPOUT=${d} ==="
          OUT_DIR=artifacts/lr_${lr}_bs_${batch}_wd_${wd}_huber_${h}_dropout_${d}
          mkdir -p "$OUT_DIR"
          LOG=logs/lr_${lr}_bs_${batch}_wd_${wd}_huber_${h}_dropout_${d}.log

          CMD=(cargo run --release --features pytorch --bin train -- --train data/train.csv --val data/val.csv --device cuda --lr "${lr}" --batch "${batch}" --wd "${wd}" --out "$OUT_DIR")
          if [[ "${h}" != "none" ]]; then
            CMD+=(--huber-delta "${h}")
          fi
          if [[ -n "${d}" ]]; then
            CMD+=(--dropout "${d}")
          fi
          ("${CMD[@]}" > "$LOG" 2>&1) || true

          BEST_LINE=$(grep -E "Best validation loss" -m 1 "$LOG" || true)
          if [[ -n "$BEST_LINE" ]]; then
            BEST_LOSS=$(echo "$BEST_LINE" | awk -F":" '{print $2}' | xargs)
          else
            BEST_LOSS="NA"
          fi
          MODEL_PATH="$OUT_DIR/best_model.safetensors"

          echo "${lr},${batch},${wd},${h},${d},${BEST_LOSS},${MODEL_PATH},${LOG}" >> "$OUT_CSV"
          echo "Done: lr=${lr} batch=${batch} wd=${wd} huber=${h} dropout=${d} -> ${BEST_LOSS}"
          sleep 3
        done
      done
    done
  done
done

echo "Sweep finished. Results in $OUT_CSV"