#!/usr/bin/env python3
"""
Scan features in training CSV and report top correlations with next_day_return.
Usage: scripts/scan_feature_corr.py [--csv data/train.csv] [--sample-rows 100000]
"""
import argparse
import pandas as pd
import numpy as np
# use pandas' corr(method='spearman') to avoid scipy dependency

parser = argparse.ArgumentParser()
parser.add_argument('--csv', default='data/train.csv')
parser.add_argument('--sample-rows', type=int, default=100000)
parser.add_argument('--frac-per-chunk', type=float, default=0.02)
parser.add_argument('--chunksize', type=int, default=20000)
args = parser.parse_args()

csv = args.csv
sample_rows = args.sample_rows
frac = args.frac_per_chunk
chunksize = args.chunksize
print(f'Reading CSV in chunks from {csv} (chunksize={chunksize}, frac={frac})')

reader = pd.read_csv(csv, chunksize=chunksize, dtype=str)
samples = []
for chunk in reader:
    k = max(1, int(len(chunk) * frac))
    samples.append(chunk.sample(n=min(k, len(chunk)), random_state=42))
    if sum(len(df) for df in samples) >= sample_rows:
        break
if not samples:
    raise SystemExit('No data sampled')
df = pd.concat(samples, ignore_index=True)
print('Sampled rows:', len(df))

# target
T = 'next_day_return'
if T not in df.columns:
    raise SystemExit('Target column not found in sampled data')

exclude = {'ts_code','trade_date','next_day_return','next_day_direction','next_3day_return','next_3day_direction'}
features = [c for c in df.columns if c not in exclude]
print('Candidate features:', len(features))

# convert to numeric
numeric = []
for c in features:
    df[c] = pd.to_numeric(df[c], errors='coerce')
    if df[c].notna().sum() > 100:
        numeric.append(c)
print('Numeric features:', len(numeric))

# target numeric
df[T] = pd.to_numeric(df[T], errors='coerce')
mask = df[T].notna()
for c in numeric:
    mask = mask & df[c].notna()
print('Rows with complete data:', mask.sum())
if mask.sum() < 10:
    raise SystemExit('Insufficient complete rows for correlation')

res = []
for c in numeric:
    x = df.loc[mask, c].astype(float)
    y = df.loc[mask, T].astype(float)
    if x.nunique() <= 1:
        continue
    pear = np.corrcoef(x, y)[0,1]
    # compute Spearman via ranks to avoid scipy dependency
    xr = x.rank(method='average')
    yr = y.rank(method='average')
    spe = np.corrcoef(xr, yr)[0,1]
    res.append((c, pear, spe))

res_sorted = sorted(res, key=lambda r: abs(r[1]) if not np.isnan(r[1]) else 0.0, reverse=True)
print('\nTop features by |Pearson|:')
for i, (c, pear, spe) in enumerate(res_sorted[:50], 1):
    print(f'{i:2d}. {c:40s} Pearson={pear:.4f} Spearman={spe:.4f}')

res_sorted_spe = sorted(res, key=lambda r: abs(r[2]) if not np.isnan(r[2]) else 0.0, reverse=True)
print('\nTop features by |Spearman|:')
for i, (c, pear, spe) in enumerate(res_sorted_spe[:50], 1):
    print(f'{i:2d}. {c:40s} Pearson={pear:.4f} Spearman={spe:.4f}')
