#!/usr/bin/env python3
"""
Load transformer checkpoint and scaler, produce per-sample predictions for a CSV dataset and write CSV with columns: ts_code,trade_date,trans_pred
"""
import argparse
import joblib
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from train_transformer import ResidualSeqDataset, TransformerRegressor, load_dataset, collate_fn
from torch.utils.data import DataLoader


def predict_on_dataset(model, samples, batch_size=512, device=None):
    device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    loader = DataLoader(samples, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    model.eval()
    preds = []
    with torch.no_grad():
        for ts, dates, X, _ in loader:
            X = X.to(device)
            out = model(X).cpu().numpy()
            for t,d,p in zip(ts, dates, out):
                preds.append({'ts_code': t, 'trade_date': d, 'trans_pred': float(p)})
    return pd.DataFrame(preds)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='artifacts/transformer_best.pt')
    p.add_argument('--scaler', default='artifacts/transformer_scaler.pkl')
    p.add_argument('--input-csv', required=True)
    p.add_argument('--seq-len', type=int, default=60)
    p.add_argument('--out', required=True)
    p.add_argument('--batch', type=int, default=512)
    args = p.parse_args()

    Path('artifacts').mkdir(parents=True, exist_ok=True)

    scaler = joblib.load(args.scaler)
    features = scaler['features']
    mean = scaler['mean']
    std = scaler['std']

    samples, _ = load_dataset(args.input_csv, args.seq_len, features=features)
    samples = [(ts,d,(X-mean)/std,y) for (ts,d,X,y) in samples]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TransformerRegressor(n_features=len(features))
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.to(device)

    df_preds = predict_on_dataset(model, samples, batch_size=args.batch, device=device)
    df_preds.to_csv(args.out, index=False)
    print('Saved transformer predictions to', args.out)

if __name__ == '__main__':
    main()
