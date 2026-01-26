#!/usr/bin/env python3
"""
Lightweight smoke test runner (no pytest dependency).
Creates tiny synthetic data and runs the ensemble pipeline end-to-end.
Exits non-zero on failure.
"""
import subprocess
import sys
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path('.').resolve()
ART = ROOT / 'artifacts'
DATA = ROOT / 'data'
TEST_DIR = ROOT / 'tests' / 'tmp'
TEST_DIR.mkdir(parents=True, exist_ok=True)
ART.mkdir(parents=True, exist_ok=True)
DATA.mkdir(parents=True, exist_ok=True)

TRAIN = TEST_DIR / 'train_small.csv'
VAL = TEST_DIR / 'val_small.csv'

if not TRAIN.exists():
    dates = pd.date_range('2020-01-01', periods=200)
    rows = []
    for t in ['A','B']:
        for i,d in enumerate(dates):
            rows.append({'ts_code': t, 'trade_date': d.strftime('%Y-%m-%d'), 'feat1': np.sin(i/10), 'feat2': i%5, 'next_day_return': ((i%7)-3)/100.0})
    df = pd.DataFrame(rows)
    df.iloc[:300].to_csv(TRAIN, index=False)
    df.iloc[300:].to_csv(VAL, index=False)


def run(cmd):
    print('RUN:', ' '.join(cmd))
    try:
        subprocess.check_call(cmd)
    except subprocess.CalledProcessError as e:
        print('Command failed:', e)
        sys.exit(1)


def main():
    # Train base model
    run(['python3','scripts/train_base_lightgbm.py','--train',str(TRAIN),'--val',str(VAL),'--model-out',str(ART/'test_lgb.pkl'),'--pred-out-train',str(ART/'base_pred_train.csv'),'--pred-out-val',str(ART/'base_pred_val.csv'),'--num-boost-round','5'])
    if not (ART/'base_pred_val.csv').exists():
        print('Base predictions missing')
        sys.exit(2)

    # Create residuals using the same logic as script (re-run to ensure data/train_resid.csv exists)
    run(['python3','scripts/train_base_lightgbm.py','--train',str(TRAIN),'--val',str(VAL),'--model-out',str(ART/'test_lgb.pkl'),'--pred-out-train',str(ART/'base_pred_train.csv'),'--pred-out-val',str(ART/'base_pred_val.csv'),'--num-boost-round','5'])

    # Prepare residual versions for transformer (copy train/val into data/ and create residual CSVs)
    # The train_base_lightgbm creates data/train_resid.csv and data/val_resid.csv
    if not (DATA/'train_resid.csv').exists() or not (DATA/'val_resid.csv').exists():
        print('Residual datasets missing after base training; expected data/train_resid.csv and data/val_resid.csv')
        # attempt to create them by running the train script again
        run(['python3','scripts/train_base_lightgbm.py','--train',str(TRAIN),'--val',str(VAL),'--model-out',str(ART/'test_lgb.pkl'),'--pred-out-train',str(ART/'base_pred_train.csv'),'--pred-out-val',str(ART/'base_pred_val.csv'),'--num-boost-round','5'])

    # Train transformer for 1 epoch
    run(['python3','scripts/train_transformer.py','--train',str(DATA/'train_resid.csv'),'--val',str(DATA/'val_resid.csv'),'--epochs','1','--seq-len','10','--batch','8','--model-out',str(ART/'test_transformer.pt'),'--scaler-out',str(ART/'test_scaler.pkl')])
    if not (ART/'test_transformer.pt').exists():
        print('Transformer model not saved')
        sys.exit(3)

    # Predict and fuse
    run(['python3','scripts/predict_transformer.py','--model',str(ART/'test_transformer.pt'),'--scaler',str(ART/'test_scaler.pkl'),'--input-csv',str(DATA/'val_resid.csv'),'--out',str(ART/'test_trans_pred_val.csv'),'--seq-len','10','--batch','8'])
    if not (ART/'test_trans_pred_val.csv').exists():
        print('Transformer predictions missing')
        sys.exit(4)

    run(['python3','scripts/fuse_models.py','--base-val',str(ART/'base_pred_val.csv'),'--trans-val',str(ART/'test_trans_pred_val.csv'),'--val-truth',str(VAL),'--out',str(ART/'test_fusion.json')])
    if not (ART/'test_fusion.json').exists():
        print('Fusion weights not saved')
        sys.exit(5)

    print('Smoke tests passed ✅')

if __name__ == '__main__':
    main()
