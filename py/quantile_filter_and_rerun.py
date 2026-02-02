#!/usr/bin/env python3
"""
Exclude worst instruments and re-evaluate top quantile returns
- Loads previous quantile diagnostics summary to identify worst instruments
- Excludes a specified number of worst instruments (union across quantiles or per-quantile)
- Recomputes quantile performance metrics and daily P&L
- Saves outputs: CSVs and JSON summary
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--exclude-n', type=int, default=20, help='Number of worst instruments to exclude (per quantile union)')
parser.add_argument('--quantiles', nargs='+', type=float, default=[0.05, 0.10, 0.20], help='Quantiles to evaluate')
parser.add_argument('--tc', type=float, default=0.0005, help='Transaction cost')
parser.add_argument('--slippage', type=float, default=0.0005, help='Slippage')
args = parser.parse_args()

Path('artifacts').mkdir(exist_ok=True)

# load ensemble proba
if Path('artifacts/test_predictions_kc_mixed_ensemble_v2.csv').exists():
    ens = pd.read_csv('artifacts/test_predictions_kc_mixed_ensemble_v2.csv')
elif Path('artifacts/test_predictions_kc_ensemble.csv').exists():
    ens = pd.read_csv('artifacts/test_predictions_kc_ensemble.csv')
else:
    raise SystemExit('No ensemble prediction file found')

# pick proba column
if 'predicted_proba_ensemble' in ens.columns:
    proba_col = 'predicted_proba_ensemble'
elif 'predicted_proba' in ens.columns:
    proba_col = 'predicted_proba'
elif 'predicted_proba_calibrated' in ens.columns:
    proba_col = 'predicted_proba_calibrated'
else:
    raise SystemExit('No probability column found in predictions')

# load test and diagnostics
df_test = pd.read_csv('data/kc_test.csv')
with open('artifacts/quantile_diagnostics_summary.json','r') as f:
    diag = json.load(f)

# collect worst instruments (union across quantiles)
worst_union = set()
for qstr, info in diag.items():
    # info['worst_instruments'] is a list of dicts with 'ts_code','net_sum'
    for entry in info.get('worst_instruments', [])[:args.exclude_n if 'args' in globals() else 20]:
        worst_union.add(entry['ts_code'])

# If diag keys are strings like '0.05', convert
if len(worst_union)==0:
    # try parsing keys as floats
    for q in args.quantiles:
        qk = str(q)
        info = diag.get(qk, {})
        for entry in info.get('worst_instruments', [])[:args.exclude_n]:
            worst_union.add(entry['ts_code'])

worst_union = set(list(worst_union)[:args.exclude_n])
print(f"Excluding {len(worst_union)} worst instruments: sample: {list(worst_union)[:5]}")

# build bt
bt = pd.DataFrame({
    'ts_code': df_test['ts_code'].values,
    'trade_date': df_test['trade_date'].values,
    'return': df_test['next_day_return'].values,
    'proba': ens[proba_col].values
})

# filter out worst instruments
bt_filtered = bt[~bt['ts_code'].isin(worst_union)].copy()
bt_filtered = bt_filtered.sort_values(['ts_code','trade_date']).reset_index(drop=True)

results = {}
for q in args.quantiles:
    thr = bt_filtered['proba'].quantile(1 - q)
    sel_full = bt_filtered.copy()
    sel_full['pos'] = (sel_full['proba'] >= thr).astype(int)
    sel_full['prev_pos'] = sel_full.groupby('ts_code')['pos'].shift(1).fillna(0).astype(int)
    sel_full['trade_flag'] = (sel_full['pos'] != sel_full['prev_pos']).astype(int)
    sel_full['net_return'] = sel_full['pos'] * sel_full['return'] - sel_full['trade_flag'] * 2 * (args.tc + args.slippage)

    trades = sel_full[sel_full['pos']==1].copy()
    total_trades = len(trades)
    mean_net = float(trades['net_return'].mean()) if total_trades>0 else 0.0
    win_rate = float((trades['return']>0).sum() / total_trades) if total_trades>0 else 0.0

    # daily aggregated
    daily = sel_full[sel_full['pos']==1].groupby('trade_date').agg({'net_return':'mean'}).rename(columns={'net_return':'daily_net'})
    if len(daily)>0:
        daily['daily_net_clipped'] = np.clip(daily['daily_net'], -0.9, 0.9)
        cum = float(np.exp(np.nansum(np.log1p(daily['daily_net_clipped']))) - 1)
    else:
        cum = 0.0

    results[q] = {'trades': int(total_trades), 'mean_net': mean_net, 'win_rate': win_rate, 'total_daily_net': cum}

    trades.to_csv(f'artifacts/filtered_quantile_{int(q*100)}_trades.csv', index=False)
    daily.to_csv(f'artifacts/filtered_quantile_{int(q*100)}_daily.csv')

with open('artifacts/filtered_quantile_summary.json','w') as f:
    json.dump(results, f, indent=2)

print('Filtered quantile results saved to artifacts/filtered_quantile_summary.json')
print('Summary:')
for q, r in results.items():
    print(f"top {int(q*100)}% -> trades={r['trades']}, mean_net={r['mean_net']:.4%}, win_rate={r['win_rate']:.2%}, total_daily_net={r['total_daily_net']:.2%}")
