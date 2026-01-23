#!/usr/bin/env python3
import pandas as pd
import glob

train_paths = glob.glob('data/*/train_model.csv')
val_paths = glob.glob('data/*/val_model.csv')

if not train_paths:
    print('No train_model.csv found')
    raise SystemExit(1)
train_path = train_paths[0]
val_path = val_paths[0] if val_paths else None

print('Sampling from', train_path)
df_train = pd.read_csv(train_path)
train_sample = df_train.sample(n=min(50000, len(df_train)), random_state=42)
train_sample.to_csv('data/train_model_small.csv', index=False)
print('Wrote data/train_model_small.csv', len(train_sample))

if val_path:
    print('Sampling from', val_path)
    df_val = pd.read_csv(val_path)
    val_sample = df_val.sample(n=min(10000, len(df_val)), random_state=42)
    val_sample.to_csv('data/val_model_small.csv', index=False)
    print('Wrote data/val_model_small.csv', len(val_sample))
else:
    print('No val model file found; skipping')
