#!/usr/bin/env python3
"""
Check date overlaps between train/val/test CSVs
- Loads CSVs (defaults: ./data/kc_train.csv, ./data/kc_val.csv, ./data/kc_test.csv)
- Parses `trade_date` column (supports YYYYMMDD or YYYY-MM-DD)
- Reports date ranges, unique dates counts
- Finds overlaps in date sets and (ts_code, trade_date) pairs
- Validates strict chronological split (train < val < test)
- Saves summary JSON and detailed overlap CSVs under `artifacts/`
"""

import argparse
from pathlib import Path
import pandas as pd
import json

parser = argparse.ArgumentParser()
parser.add_argument('--train', default='./data/kc_train.csv')
parser.add_argument('--val', default='./data/kc_val.csv')
parser.add_argument('--test', default='./data/kc_test.csv')
args = parser.parse_args()

Path('artifacts').mkdir(exist_ok=True)

def load_df(path):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {path}")
    df = pd.read_csv(p)
    if 'trade_date' not in df.columns:
        raise KeyError(f"Expected 'trade_date' column in {path}")
    # normalize trade_date to datetime
    df['trade_date_str'] = df['trade_date'].astype(str)
    # try YYYYMMDD
    try:
        df['trade_date_dt'] = pd.to_datetime(df['trade_date_str'], format='%Y%m%d')
    except Exception:
        # try ISO format
        df['trade_date_dt'] = pd.to_datetime(df['trade_date_str'], errors='coerce')
    if df['trade_date_dt'].isnull().any():
        raise ValueError(f"Some trade_date could not be parsed in {path}")
    return df

# Load
train = load_df(args.train)
val = load_df(args.val)
test = load_df(args.test)

summary = {}
for name, df in [('train', train), ('val', val), ('test', test)]:
    min_date = df['trade_date_dt'].min()
    max_date = df['trade_date_dt'].max()
    unique_dates = df['trade_date_dt'].nunique()
    n_rows = len(df)
    unique_stocks = df['ts_code'].nunique() if 'ts_code' in df.columns else None
    summary[name] = {
        'rows': int(n_rows),
        'min_date': str(min_date.date()),
        'max_date': str(max_date.date()),
        'unique_dates': int(unique_dates),
        'unique_stocks': int(unique_stocks) if unique_stocks is not None else None
    }

# Date-set overlaps
train_dates = set(train['trade_date_dt'].dt.date.unique())
val_dates = set(val['trade_date_dt'].dt.date.unique())
test_dates = set(test['trade_date_dt'].dt.date.unique())

overlap_train_val_dates = sorted(list(train_dates & val_dates))
overlap_val_test_dates = sorted(list(val_dates & test_dates))
overlap_train_test_dates = sorted(list(train_dates & test_dates))

summary['overlap_dates'] = {
    'train_val_count': len(overlap_train_val_dates),
    'val_test_count': len(overlap_val_test_dates),
    'train_test_count': len(overlap_train_test_dates)
}

# Pair overlaps (ts_code, trade_date)
def pairs(df):
    if 'ts_code' in df.columns:
        # Vectorized construction of (ts_code, date) pairs for performance
        return set(zip(df['ts_code'].values, df['trade_date_dt'].dt.date.values))
    else:
        return set()

pairs_train = pairs(train)
pairs_val = pairs(val)
pairs_test = pairs(test)

pair_overlap_train_val = sorted(list(pairs_train & pairs_val))
pair_overlap_val_test = sorted(list(pairs_val & pairs_test))
pair_overlap_train_test = sorted(list(pairs_train & pairs_test))

summary['overlap_pairs'] = {
    'train_val_count': len(pair_overlap_train_val),
    'val_test_count': len(pair_overlap_val_test),
    'train_test_count': len(pair_overlap_train_test)
}

# Chronological checks
chron_ok = True
chron_msgs = []
if pd.to_datetime(summary['train']['max_date']) >= pd.to_datetime(summary['val']['min_date']):
    chron_ok = False
    chron_msgs.append('train.max_date >= val.min_date')
if pd.to_datetime(summary['val']['max_date']) >= pd.to_datetime(summary['test']['min_date']):
    chron_ok = False
    chron_msgs.append('val.max_date >= test.min_date')
summary['chronological_split_ok'] = chron_ok
summary['chronological_messages'] = chron_msgs

# Save overlap details if any
if overlap_train_val_dates:
    pd.DataFrame({'date': overlap_train_val_dates}).to_csv('artifacts/overlap_train_val_dates.csv', index=False)
if overlap_val_test_dates:
    pd.DataFrame({'date': overlap_val_test_dates}).to_csv('artifacts/overlap_val_test_dates.csv', index=False)
if overlap_train_test_dates:
    pd.DataFrame({'date': overlap_train_test_dates}).to_csv('artifacts/overlap_train_test_dates.csv', index=False)

if pair_overlap_train_val:
    pd.DataFrame(pair_overlap_train_val, columns=['ts_code','trade_date']).to_csv('artifacts/overlap_train_val_pairs.csv', index=False)
if pair_overlap_val_test:
    pd.DataFrame(pair_overlap_val_test, columns=['ts_code','trade_date']).to_csv('artifacts/overlap_val_test_pairs.csv', index=False)
if pair_overlap_train_test:
    pd.DataFrame(pair_overlap_train_test, columns=['ts_code','trade_date']).to_csv('artifacts/overlap_train_test_pairs.csv', index=False)

with open('artifacts/check_date_overlaps_summary.json','w') as f:
    json.dump(summary, f, indent=2)

# Print summary
print('\nDate Overlap Summary:')
print(json.dumps(summary, indent=2))

if not chron_ok:
    print('\n⚠️ Chronological split check FAILED: see artifacts/check_date_overlaps_summary.json and overlap CSVs for details')
else:
    print('\n✅ Chronological split OK: no date boundary violations')

if any([len(overlap_train_val_dates), len(overlap_val_test_dates), len(overlap_train_test_dates), len(pair_overlap_train_val), len(pair_overlap_val_test), len(pair_overlap_train_test)]):
    print('\nDetailed overlap files saved to artifacts/*.csv')
else:
    print('\nNo overlaps detected (date sets and (ts_code, date) pairs)')
