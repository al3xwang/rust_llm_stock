#!/usr/bin/env python3
import pandas as pd
import numpy as np

CSV = 'train.csv'

def main():
    df = pd.read_csv(CSV, low_memory=False)
    print('Loaded', CSV, 'shape=', df.shape)

    targets = ['next_day_return','next_day_direction','next_3day_return','next_3day_direction']
    present_targets = [t for t in targets if t in df.columns]
    if 'next_day_return' not in df.columns:
        print('next_day_return not found in CSV columns; aborting')
        return

    id_cols = ['ts_code','trade_date']
    exclude = set(id_cols + present_targets)
    feature_cols = [c for c in df.columns if c not in exclude]

    print('Total columns:', len(df.columns))
    print('Feature columns:', len(feature_cols))

    # Check for any feature columns that start with 'next_'
    next_like = [c for c in feature_cols if c.lower().startswith('next_')]
    if next_like:
        print('\nWARNING: feature columns starting with "next_":')
        for c in next_like:
            print('  ', c)
    else:
        print('\nNo feature columns starting with "next_" found.')

    # Basic target statistics
    print('\nnext_day_return stats:')
    print(df['next_day_return'].describe())

    # Duplicate rows checks
    dup_td = df.duplicated(subset=['ts_code','trade_date']).sum()
    dup_full = df.duplicated().sum()
    print(f'\nDuplicate (ts_code,trade_date) rows: {dup_td}')
    print(f'Exact full-row duplicates: {dup_full}')

    # Compute correlations
    y = df['next_day_return'].astype(float)

    results = []
    for col in feature_cols:
        try:
            x = pd.to_numeric(df[col], errors='coerce')
        except Exception:
            continue
        # drop NaNs pairwise
        mask = (~x.isna()) & (~y.isna())
        if mask.sum() < 30:
            continue
        px = x[mask]
        py = y[mask]
        if px.std() == 0 or py.std() == 0:
            continue
        pearson_r = px.corr(py, method='pearson')
        # compute spearman via rank + pearson to avoid scipy dependency
        spearman_r = px.rank(method='average').corr(py.rank(method='average'))
        results.append((col, pearson_r if pearson_r is not None else 0.0, spearman_r if spearman_r is not None else 0.0, mask.sum()))

    # Sort by absolute Pearson
    results.sort(key=lambda r: abs(r[1]) if r[1] is not None else 0, reverse=True)

    print('\nTop 30 features by |Pearson| with next_day_return:')
    for col, p, s, n in results[:30]:
        print(f"{col}: pearson={p:.6f}, spearman={s:.6f}, n={n}")

    # Check for perfect or near-perfect correlations
    near_perfect = [(c,p,s,n) for (c,p,s,n) in results if abs(p) >= 0.9999]
    if near_perfect:
        print('\nPOTENTIAL LEAKAGE: near-perfect Pearson correlations found:')
        for c,p,s,n in near_perfect:
            print(f"  {c}: pearson={p}, spearman={s}, n={n}")
    else:
        print('\nNo near-perfect Pearson correlations (|r|>=0.9999) found.')

    # Check if any feature equals the target numerically for a large fraction of rows
    near_equal_features = []
    for col in feature_cols:
        try:
            a = pd.to_numeric(df[col], errors='coerce')
            b = pd.to_numeric(df['next_day_return'], errors='coerce')
        except Exception:
            continue
        mask = (~a.isna()) & (~b.isna())
        if mask.sum() == 0:
            continue
        proportion = np.isclose(a[mask], b[mask], atol=1e-9).sum() / mask.sum()
        if proportion > 0.01:  # arbitrary threshold to flag
            near_equal_features.append((col, proportion, mask.sum()))
    if near_equal_features:
        print('\nFeatures nearly equal to next_day_return (proportion > 0.01):')
        for c,p,n in sorted(near_equal_features, key=lambda x: x[1], reverse=True):
            print(f'  {c}: proportion={p:.6f}, n={n}')
    else:
        print('\nNo features nearly equal to next_day_return at >1% proportion.')

    # Check if any feature exactly equals the target (after NaN fill)
    exact_matches = []
    for col in feature_cols:
        try:
            a = df[col].fillna(0).astype(str)
            b = df['next_day_return'].fillna(0).astype(str)
        except Exception:
            continue
        if (a == b).all():
            exact_matches.append(col)
    if exact_matches:
        print('\nExact string-equal matches to next_day_return:')
        for c in exact_matches:
            print('  ', c)

    # Check correlation distribution
    import statistics
    prs = [abs(p) for (_,p,_,_) in results if p is not None]
    if prs:
        print('\nCorrelation summary (abs Pearson): min={:.6f}, median={:.6f}, max={:.6f}'.format(min(prs), statistics.median(prs), max(prs)))

if __name__ == '__main__':
    main()
