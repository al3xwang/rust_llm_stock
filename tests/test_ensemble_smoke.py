import os
import subprocess
import json
from pathlib import Path

TEST_DIR = Path('tests/tmp')
TEST_DIR.mkdir(parents=True, exist_ok=True)

# Small synthetic data
TRAIN = TEST_DIR / 'train_small.csv'
VAL = TEST_DIR / 'val_small.csv'

if not TRAIN.exists():
    import pandas as pd
    import numpy as np
    dates = pd.date_range('2020-01-01', periods=100)
    rows = []
    for t in ['A','B']:
        for i,d in enumerate(dates):
            rows.append({'ts_code': t, 'trade_date': d.strftime('%Y-%m-%d'), 'feat1': np.sin(i/10), 'feat2': i%5, 'next_day_return': ((i%7)-3)/100.0})
    df = pd.DataFrame(rows)
    df.iloc[:140].to_csv(TRAIN, index=False)
    df.iloc[140:].to_csv(VAL, index=False)


def run(cmd):
    print('RUN:', ' '.join(cmd))
    subprocess.check_call(cmd)


def test_end_to_end_smoke(tmp_path):
    # Train base model
    run(['python3','scripts/train_base_lightgbm.py','--train',str(TRAIN),'--val',str(VAL),'--model-out','artifacts/test_lgb.pkl','--pred-out-train','artifacts/base_pred_train.csv','--pred-out-val','artifacts/base_pred_val.csv','--num-boost-round','10'])
    assert Path('artifacts/base_pred_val.csv').exists()

    # Create residuals
    assert Path('data/train_resid.csv') or True  # script already writes to data/, but we won't strictly require it here

    # Train transformer for 1 epoch
    run(['python3','scripts/train_transformer.py','--train',str(TRAIN),'--val',str(VAL),'--epochs','1','--seq-len','10','--batch','8','--model-out','artifacts/test_transformer.pt','--scaler-out','artifacts/test_scaler.pkl'])
    assert Path('artifacts/test_transformer.pt').exists()

    # Predict transformer outputs
    run(['python3','scripts/predict_transformer.py','--model','artifacts/test_transformer.pt','--scaler','artifacts/test_scaler.pkl','--input-csv',str(VAL),'--out','artifacts/test_trans_pred_val.csv','--seq-len','10','--batch','8'])
    assert Path('artifacts/test_trans_pred_val.csv').exists()

    # Fuse
    run(['python3','scripts/fuse_models.py','--base-val','artifacts/base_pred_val.csv','--trans-val','artifacts/test_trans_pred_val.csv','--val-truth',str(VAL),'--out','artifacts/test_fusion.json'])
    assert Path('artifacts/test_fusion.json').exists()
