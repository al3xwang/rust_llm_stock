#!/usr/bin/env python3
import pandas as pd, sys, glob

def sample_csv(src, dst, target_nrows, seed=42, chunksize=20000, frac=0.02):
    reader = pd.read_csv(src, chunksize=chunksize, dtype=str)
    samples=[]
    for chunk in reader:
        k = max(1, int(len(chunk) * frac))
        samples.append(chunk.sample(n=min(k, len(chunk)), random_state=seed))
        if sum(len(df) for df in samples) >= target_nrows:
            break
    if not samples:
        raise RuntimeError('No data sampled')
    df = pd.concat(samples, ignore_index=True)
    # if we sampled more than target, downsample; otherwise keep what we have
    if len(df) > target_nrows:
        df = df.sample(n=target_nrows, random_state=seed)
    df.to_csv(dst, index=False)
    return dst

if __name__ == '__main__':
    paths = glob.glob('rust_llm_stock/data/*/train.csv')
    if not paths:
        print('train.csv not found under rust_llm_stock/data/*', file=sys.stderr); sys.exit(1)
    train_path = paths[0]
    val_path = glob.glob('rust_llm_stock/data/*/val.csv')[0]
    print('train_path', train_path)
    print('val_path', val_path)
    train_sample = sample_csv(train_path, 'rust_llm_stock/data/train_sample_50k.csv', 50000)
    val_sample = sample_csv(val_path, 'rust_llm_stock/data/val_sample_20k.csv', 20000)
    print('wrote', train_sample, val_sample)
    df = pd.read_csv(train_sample)
    if 'next_day_return' not in df.columns:
        print('next_day_return not found', file=sys.stderr); sys.exit(1)
    vals = df['next_day_return'].sample(frac=1.0, random_state=123).reset_index(drop=True)
    df['next_day_return'] = vals
    df.to_csv('rust_llm_stock/data/train_shuffled_sample_50k.csv', index=False)
    print('wrote rust_llm_stock/data/train_shuffled_sample_50k.csv')