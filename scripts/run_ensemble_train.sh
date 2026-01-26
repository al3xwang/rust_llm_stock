#!/bin/bash
# scripts/run_ensemble_train.sh
# Walk-forward (sliding window) ensemble training using generated splits

set -e

SPLIT_LIST="data/walk_forward_splits/split_list.txt"

if [[ ! -f "$SPLIT_LIST" ]]; then
  echo "Split list $SPLIT_LIST not found. Please run scripts/generate_walk_forward_splits.py first."
  exit 1
fi

while IFS=, read -r TRAIN_CSV VAL_CSV; do
  echo "=== Walk-forward window: $TRAIN_CSV | $VAL_CSV ==="

  # Set output suffix based on window name (customize as needed)
  SUFFIX="$(basename "$TRAIN_CSV" .csv)_$(basename "$VAL_CSV" .csv)"

  echo "1) Training LightGBM base model..."
  python3 scripts/train_base_lightgbm.py \
    --train "$TRAIN_CSV" \
    --val "$VAL_CSV" \
    --model-out "artifacts/lgb_base_model_${SUFFIX}.pkl" \
    --pred-out-train "artifacts/base_pred_train_${SUFFIX}.csv" \
    --pred-out-val "artifacts/base_pred_val_${SUFFIX}.csv" \
    --num-boost-round 1000

  echo "2) Training transformer on residuals (Python trainer)..."
  python3 scripts/train_transformer.py \
    --train "data/train_resid_${SUFFIX}.csv" \
    --val "data/val_resid_${SUFFIX}.csv" \
    --seq-len 60 --epochs 10 --batch 256 \
    --model-out "artifacts/transformer_best_${SUFFIX}.pt" \
    --scaler-out "artifacts/transformer_scaler_${SUFFIX}.pkl" \
    --d-model 32 --n-head 2 --n-layers 1 --lr-scheduler cosine --resume \
    --checkpoint "artifacts/transformer_ckpt_${SUFFIX}.pt" || { echo "Transformer training failed; check logs"; exit 1; }

  echo "3) Generate transformer predictions on train/val and save to artifacts/trans_pred_{train|val}_${SUFFIX}.csv"
  python3 scripts/predict_transformer.py \
    --model "artifacts/transformer_best_${SUFFIX}.pt" \
    --scaler "artifacts/transformer_scaler_${SUFFIX}.pkl" \
    --input-csv "data/train_resid_${SUFFIX}.csv" \
    --out "artifacts/trans_pred_train_${SUFFIX}.csv" \
    --seq-len 60 --batch 512
  python3 scripts/predict_transformer.py \
    --model "artifacts/transformer_best_${SUFFIX}.pt" \
    --scaler "artifacts/transformer_scaler_${SUFFIX}.pkl" \
    --input-csv "data/val_resid_${SUFFIX}.csv" \
    --out "artifacts/trans_pred_val_${SUFFIX}.csv" \
    --seq-len 60 --batch 512

  echo "4) Fit fusion weights (requires artifacts/base_pred_val_${SUFFIX}.csv and artifacts/trans_pred_val_${SUFFIX}.csv)"
  if [[ -f "artifacts/base_pred_val_${SUFFIX}.csv" && -f "artifacts/trans_pred_val_${SUFFIX}.csv" ]]; then
    python3 scripts/fuse_models.py \
      --base-val "artifacts/base_pred_val_${SUFFIX}.csv" \
      --trans-val "artifacts/trans_pred_val_${SUFFIX}.csv" \
      --val-truth "$VAL_CSV" \
      --out "artifacts/fusion_weights_${SUFFIX}.json"
  else
    echo "Skipping fusion (missing trans or base predictions for $SUFFIX)."
  fi

done < "$SPLIT_LIST"

echo "All walk-forward windows complete."
