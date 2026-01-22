#!/usr/bin/env bash
set -euo pipefail

# Hyperparameter sweep: LR x batch x weight_decay
# Usage: ./scripts/hyper_sweep.sh

# Default device & environment (override with DEVICE env var)
# Use GPU 0 by default because it's the most powerful
DEVICE="${DEVICE:-cuda:0}"
export PATH="$HOME/.cargo/bin:$PATH"
export LIBTORCH="/home/alex/libtorch"
export LD_LIBRARY_PATH="${LIBTORCH}/lib:${LD_LIBRARY_PATH:-}"  # ensure libtorch shared libs are found at runtime

# Quick sweep mode: set QUICK=1 to run a much faster, sampled sweep
QUICK="${QUICK:-0}"

if [[ "${QUICK}" == "1" ]]; then
  echo "⚡ QUICK sweep enabled: smaller grid and sampled data"
  LRS=(1e-5 1e-4)
  BATCHES=(128 256)
  WDS=(0.0 1e-6)
  HUBERS=("none" 0.5)
  GRAD_CLIPS=("none" 0.5)
  DROPOUTS=(0.0 0.1)
  SAMPLE_FRAC="${SAMPLE_FRAC:-0.1}"
  MAX_EPOCHS="${MAX_EPOCHS:-20}"
  EARLY_STOP="${EARLY_STOP:-5}"
else
  # Expanded sweep ranges (full)
  LRS=(1e-5 3e-5 1e-4 3e-4)
  BATCHES=(64 128 256 512)
  WDS=(0.0 1e-7 1e-6 1e-5)
  HUBERS=("none" 0.1 0.5)
  GRAD_CLIPS=("none" 0.1 0.5 1.0)
  DROPOUTS=(0.0 0.1 0.2)
  SAMPLE_FRAC="${SAMPLE_FRAC:-1.0}"
  MAX_EPOCHS="${MAX_EPOCHS:-1000}"
  EARLY_STOP="${EARLY_STOP:-20}"
fi

# LR-only sweep: if LR_ONLY=1 then only iterate LRs and fix others to sensible defaults.
LR_ONLY="${LR_ONLY:-0}"
if [[ "${LR_ONLY}" == "1" ]]; then
  echo "🔬 LR-only sweep: sweeping LRs only; other hyperparameters fixed to defaults"
  BATCHES=(128)
  WDS=(0.0)
  HUBERS=("none")
  GRAD_CLIPS=("none")
  DROPOUTS=(0.0)
fi

# Utility: sample CSV by fraction (header preserved). If frac >= 1.0 the file is copied.
sample_csv() {
  local infile="$1"
  local outfile="$2"
  local frac="$3"
  # If frac >= 1.0, copy original
  if awk "BEGIN {exit !($frac >= 1.0)}"; then
    cp "$infile" "$outfile"
  else
    awk -v p="$frac" 'BEGIN{srand()} NR==1 {print; next} {if (rand() < p) print}' "$infile" > "$outfile"
  fi
}

mkdir -p logs
mkdir -p artifacts
OUT_CSV=logs/hyper_sweep_results.csv
echo "lr,batch,wd,huber,grad_clip,dropout,sample_frac,max_epochs,early_stop,best_valid_loss,model_path,log" > "$OUT_CSV"

for lr in "${LRS[@]}"; do
  for batch in "${BATCHES[@]}"; do
    for wd in "${WDS[@]}"; do
      for h in "${HUBERS[@]}"; do
        for g in "${GRAD_CLIPS[@]}"; do
          for d in "${DROPOUTS[@]}"; do
          echo "=== LR=${lr} BATCH=${batch} WD=${wd} HUBER=${h} DROPOUT=${d} ==="
          OUT_DIR=artifacts/lr_${lr}_bs_${batch}_wd_${wd}_huber_${h}_dropout_${d}
          mkdir -p "$OUT_DIR"
          LOG=logs/lr_${lr}_bs_${batch}_wd_${wd}_huber_${h}_dropout_${d}.log

          # Prepare sampled training/validation files (tmp) if SAMPLE_FRAC < 1.0
          if awk "BEGIN {exit !($SAMPLE_FRAC < 1.0)}"; then
            TMP_TRAIN=$(mktemp /tmp/train_sample.XXXX.csv)
            TMP_VAL=$(mktemp /tmp/val_sample.XXXX.csv)
            sample_csv "data/train.csv" "$TMP_TRAIN" "$SAMPLE_FRAC"
            sample_csv "data/val.csv" "$TMP_VAL" "$SAMPLE_FRAC"
            TRAIN_ARG="$TMP_TRAIN"
            VAL_ARG="$TMP_VAL"
          else
            TRAIN_ARG="data/train.csv"
            VAL_ARG="data/val.csv"
          fi

          CMD=(cargo run --release --features pytorch --bin train -- --train "${TRAIN_ARG}" --val "${VAL_ARG}" --device "${DEVICE}" --lr "${lr}" --batch "${batch}" --wd "${wd}" --max-epochs "${MAX_EPOCHS}" --early-stop "${EARLY_STOP}" --out "$OUT_DIR")
          if [[ "${h}" != "none" ]]; then
            CMD+=(--huber-delta "${h}")
          fi
          if [[ "${g}" != "none" ]]; then
            CMD+=(--grad-clip "${g}")
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

          echo "${lr},${batch},${wd},${h},${g},${d},${SAMPLE_FRAC},${MAX_EPOCHS},${EARLY_STOP},${BEST_LOSS},${MODEL_PATH},${LOG}" >> "$OUT_CSV"
          echo "Done: lr=${lr} batch=${batch} wd=${wd} huber=${h} grad_clip=${g} dropout=${d} sample_frac=${SAMPLE_FRAC} max_epochs=${MAX_EPOCHS} -> ${BEST_LOSS}"

          # Clean up temporary sample files if created
          if [[ -n "${TMP_TRAIN:-}" ]]; then
            rm -f "${TMP_TRAIN}" "${TMP_VAL}"
            unset TMP_TRAIN TMP_VAL
          fi

          sleep 3
        done
      done
    done
  done
done

echo "Sweep finished. Results in $OUT_CSV"