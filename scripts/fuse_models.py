#!/usr/bin/env python3
"""
Fit a simple linear fusion of base + transformer predictions on the validation set.
Saves weights as JSON with keys ['intercept','w_base','w_trans'] and prints validation MSE.
"""
import argparse
import json
import pandas as pd
from sklearn.linear_model import LinearRegression
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--base-val', default='artifacts/base_pred_val.csv')
    p.add_argument('--trans-val', default='artifacts/trans_pred_val.csv')
    p.add_argument('--val-truth', default='data/val.csv')
    p.add_argument('--out', default='artifacts/fusion_weights.json')
    args = p.parse_args()

    base = pd.read_csv(args.base_val)
    trans = pd.read_csv(args.trans_val)
    val = pd.read_csv(args.val_truth)

    # join on ts_code and trade_date
    df = val[['ts_code','trade_date','next_day_return']].merge(base, on=['ts_code','trade_date']).merge(trans, on=['ts_code','trade_date'])
    df = df.dropna()
    X = df[['base_pred','trans_pred']].values
    y = df['next_day_return'].values

    lr = LinearRegression().fit(X,y)
    preds = lr.predict(X)
    mse = ((preds - y)**2).mean()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    weights = {'intercept': float(lr.intercept_), 'w_base': float(lr.coef_[0]), 'w_trans': float(lr.coef_[1]), 'val_mse': float(mse)}
    with open(args.out, 'w') as f:
        json.dump(weights, f, indent=2)

    print('Fusion weights saved to', args.out)
    print('Val MSE:', mse)

if __name__ == '__main__':
    main()
