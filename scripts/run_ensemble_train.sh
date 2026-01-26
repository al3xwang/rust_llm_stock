#!/bin/bash
# scripts/run_ensemble_train.sh
# End-to-end orchestration for: train LightGBM base -> compute residuals -> train transformer on residuals -> fit fusion weights

set -e

echo "1) Training LightGBM base model..."
python3 scripts/train_base_lightgbm.py --train data/train.csv --val data/val.csv --model-out artifacts/lgb_base_model.pkl --pred-out-train artifacts/base_pred_train.csv --pred-out-val artifacts/base_pred_val.csv --num-boost-round 1000

echo "2) Training transformer on residuals (Python trainer)..."
python3 scripts/train_transformer.py --train data/train_resid.csv --val data/val_resid.csv --seq-len 60 --epochs 10 --batch 256 --model-out artifacts/transformer_best.pt --scaler-out artifacts/transformer_scaler.pkl --d-model 32 --n-head 2 --n-layers 1 --lr-scheduler cosine --resume --checkpoint artifacts/transformer_ckpt.pt || { echo "Transformer training failed; check logs"; exit 1; }

echo "3) Generate transformer predictions on train/val and save to artifacts/trans_pred_{train|val}.csv"
python3 scripts/predict_transformer.py --model artifacts/transformer_best.pt --scaler artifacts/transformer_scaler.pkl --input-csv data/train_resid.csv --out artifacts/trans_pred_train.csv --seq-len 60 --batch 512
python3 scripts/predict_transformer.py --model artifacts/transformer_best.pt --scaler artifacts/transformer_scaler.pkl --input-csv data/val_resid.csv --out artifacts/trans_pred_val.csv --seq-len 60 --batch 512


echo "4) Fit fusion weights (requires artifacts/base_pred_val.csv and artifacts/trans_pred_val.csv)"
if [[ -f artifacts/base_pred_val.csv && -f artifacts/trans_pred_val.csv ]]; then
  python3 scripts/fuse_models.py --base-val artifacts/base_pred_val.csv --trans-val artifacts/trans_pred_val.csv --val-truth data/val.csv --out artifacts/fusion_weights.json
else
  echo "Skipping fusion (missing trans or base predictions)."
fi

echo "Done."
