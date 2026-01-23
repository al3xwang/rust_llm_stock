#!/usr/bin/env python3
"""Leakage checks for data/train.csv
- Samples rows from data/*/train.csv
- Reports: header suspicious names, duplicates across splits, equality fraction to next_day_return, low-cardinality predictive deltas
Usage: scripts/leakage_checks.py --csv data/*/train.csv --sample 200000
"""
import argparse, glob, pandas as pd, numpy as np, sys

parser = argparse.ArgumentParser()
parser.add_argument('--csv', default='data/*/train.csv')
parser.add_argument('--sample', type=int, default=200000)
parser.add_argument('--chunksize', type=int, default=50000)
args = parser.parse_args()

paths = glob.glob(args.csv)
if not paths:
    print('No train CSV found for pattern', args.csv, file=sys.stderr); sys.exit(1)
train_path = paths[0]
print('Using train CSV:', train_path)

# Read in chunks to sample
reader = pd.read_csv(train_path, chunksize=args.chunksize, dtype=str)
samples=[]
for chunk in reader:
    k = max(1, int(len(chunk) * 0.05))
    samples.append(chunk.sample(n=min(k, len(chunk)), random_state=42))
    if sum(len(df) for df in samples) >= args.sample:
        break
if not samples:
    print('No data sampled', file=sys.stderr); sys.exit(1)
df = pd.concat(samples, ignore_index=True)
print('Sampled rows:', len(df))

# normalize boolean literals
for col in df.columns:
    if df[col].dtype == object:
        df[col] = df[col].replace({'True':'true','False':'false'})

T='next_day_return'
if T not in df.columns:
    print('Target column not present:', T, file=sys.stderr); sys.exit(1)

# basic header scan for suspicious names
suspicious_keywords = ['next','future','pred','predict','target','label','leak']
cols = df.columns.tolist()
print('\nHeader scan (columns matching suspicious keywords):')
for c in cols:
    low = c.lower()
    if any(k in low for k in suspicious_keywords) and c not in (T,'next_day_direction','next_3day_return','next_3day_direction'):
        print(' -', c)

# check duplicates across splits
def path_for(pattern):
    p = glob.glob(pattern)
    return p[0] if p else None
train_glob = 'data/*/train.csv'
val_glob = 'data/*/val.csv'
test_glob = 'data/*/test.csv'
train_p = path_for(train_glob)
val_p = path_for(val_glob)
test_p = path_for(test_glob)
print('\nSplit files found:')
print(' train:', train_p)
print(' val:  ', val_p)
print(' test: ', test_p)

def key_set(path):
    if not path: return set()
    keys=set()
    for chunk in pd.read_csv(path, chunksize=100000, dtype=str):
        if 'ts_code' in chunk.columns and 'trade_date' in chunk.columns:
            keys.update((r[0], r[1]) for r in chunk[['ts_code','trade_date']].values)
    return keys

train_keys = key_set(train_p)
val_keys = key_set(val_p)
test_keys = key_set(test_p)

if val_p:
    inter = train_keys & val_keys
    print('\nOverlaps train<->val:', len(inter))
    if len(inter)>0:
        print('Examples:', list(inter)[:5])
if test_p:
    inter2 = train_keys & test_keys
    print('Overlaps train<->test:', len(inter2))
    if len(inter2)>0:
        print('Examples:', list(inter2)[:5])

# Convert numeric columns and target
numeric_cols=[]
for c in df.columns:
    if c in ('ts_code','trade_date'):
        continue
    df[c] = pd.to_numeric(df[c], errors='coerce')
    if df[c].notna().sum()>100:
        numeric_cols.append(c)
print('\nNumeric columns:', len(numeric_cols))

# target numeric and sign
df[T] = pd.to_numeric(df[T], errors='coerce')
mask = df[T].notna()
base_pos_rate = (df.loc[mask, T] > 0).mean()
print('Base positive next-day return rate:', base_pos_rate)

# equality fraction and pearson/spearman
suspicious_equal=[]
suspicious_cat=[]
for c in numeric_cols:
    if c==T: continue
    # equality fraction (exact float equality)
    try:
        eq_frac = (df.loc[mask, c] == df.loc[mask, T]).mean()
    except Exception:
        eq_frac = 0.0
    # unique
    uniq = df[c].nunique(dropna=True)
    max_delta = 0.0
    max_cat=None
    if uniq<=50:
        # categorical predictive delta
        grp = df.loc[mask, [c, T]].dropna()
        if not grp.empty:
            pos = (grp[T]>0).mean()
            deltas = []
            for val, sub in grp.groupby(c):
                p = (sub[T]>0).mean()
                deltas.append(abs(p - pos))
                if abs(p-pos) > max_delta:
                    max_delta = abs(p-pos); max_cat = val
    if eq_frac>0.01:
        suspicious_equal.append((c, eq_frac))
    if max_delta>0.2:
        suspicious_cat.append((c, uniq, max_delta, max_cat))

print('\nFeatures with >1% exact equality to target:')
for c, v in sorted(suspicious_equal, key=lambda x:-x[1]):
    print(f' - {c}: equality_frac={v:.4f}')

print('\nLow-cardinality features with large predictive delta (>0.2):')
for c, uniq, delta, cat in sorted(suspicious_cat, key=lambda x:-x[2]):
    print(f' - {c}: uniq={uniq}, max_delta={delta:.3f}, value={cat}')

# final quick correlation top-n
res=[]
for c in numeric_cols:
    if c==T: continue
    x = df.loc[mask, c].astype(float)
    y = df.loc[mask, T].astype(float)
    if x.nunique()<=1: continue
    pear = np.corrcoef(x, y)[0,1]
    xr = x.rank(method='average')
    yr = y.rank(method='average')
    spe = np.corrcoef(xr, yr)[0,1]
    res.append((c, pear, spe))
res_sorted = sorted(res, key=lambda r: abs(r[1]) if not np.isnan(r[1]) else 0.0, reverse=True)
print('\nTop 20 features by |Pearson|:')
for i,(c,pear,spe) in enumerate(res_sorted[:20],1):
    print(f'{i:2d}. {c:40s} Pearson={pear:.4f} Spearman={spe:.4f}')

print('\nDone.')
