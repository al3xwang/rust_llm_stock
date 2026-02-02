#!/usr/bin/env python3
"""
Quantile Cost Sensitivity Sweep
- Sweeps transaction cost (tc) and slippage values and evaluates top quantile performance.
- Outputs CSV/JSON summary and a PNG plot of returns vs total cost per quantile.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

Path('artifacts').mkdir(exist_ok=True)

# load ensemble probs
if Path('artifacts/test_predictions_kc_mixed_ensemble_v2.csv').exists():
    ens = pd.read_csv('artifacts/test_predictions_kc_mixed_ensemble_v2.csv')
elif Path('artifacts/test_predictions_kc_ensemble.csv').exists():
    ens = pd.read_csv('artifacts/test_predictions_kc_ensemble.csv')
else:
    raise SystemExit('Ensemble predictions CSV not found in artifacts')

# pick probability column
if 'predicted_proba_ensemble' in ens.columns:
    proba_col = 'predicted_proba_ensemble'
elif 'predicted_proba' in ens.columns:
    proba_col = 'predicted_proba'
elif 'predicted_proba_calibrated' in ens.columns:
    proba_col = 'predicted_proba_calibrated'
else:
    raise SystemExit('No ensemble probability column found')

# load test returns
df_test = pd.read_csv('data/kc_test.csv')

bt = pd.DataFrame({
    'ts_code': df_test['ts_code'].values,
    'trade_date': df_test['trade_date'].values,
    'return': df_test['next_day_return'].values,
    'proba': ens[proba_col].values
})

bt = bt.sort_values(['ts_code','trade_date']).reset_index(drop=True)

quantiles = [0.05, 0.10, 0.20]
# grid of tc and slippage to test
tc_values = [0.0, 0.0005, 0.001, 0.0025, 0.005]
slippage_values = [0.0, 0.0005, 0.001, 0.0025, 0.005]

rows = []
for tc in tc_values:
    for slippage in slippage_values:
        total_cost = tc + slippage
        for q in quantiles:
            thr = bt['proba'].quantile(1 - q)
            # compute pos per instrument per date
            sel_full = bt.copy()
            sel_full['pos'] = (sel_full['proba'] >= thr).astype(int)
            sel_full['prev_pos'] = sel_full.groupby('ts_code')['pos'].shift(1).fillna(0).astype(int)
            sel_full['trade_flag'] = (sel_full['pos'] != sel_full['prev_pos']).astype(int)
            sel_full['net_return'] = sel_full['pos'] * sel_full['return'] - sel_full['trade_flag'] * 2 * (tc + slippage)
            trades = sel_full[sel_full['pos'] == 1].copy()

            total_trades = len(trades)
            mean_net = float(trades['net_return'].mean()) if total_trades>0 else 0.0
            win_rate = float((trades['return']>0).sum() / total_trades) if total_trades>0 else 0.0

            # daily aggregated returns
            daily = sel_full[sel_full['pos']==1].groupby('trade_date').agg({'net_return':'mean'}).rename(columns={'net_return':'daily_net'})
            if len(daily)>0:
                daily['daily_net_clipped'] = np.clip(daily['daily_net'], -0.9, 0.9)
                cum = float(np.exp(np.nansum(np.log1p(daily['daily_net_clipped']))) - 1)
            else:
                cum = 0.0

            rows.append({
                'tc': tc,
                'slippage': slippage,
                'total_cost': total_cost,
                'quantile': q,
                'trades': int(total_trades),
                'mean_net_per_trade': mean_net,
                'win_rate': win_rate,
                'total_daily_net': cum
            })

res_df = pd.DataFrame(rows)
res_df.to_csv('artifacts/quantile_cost_sensitivity.csv', index=False)
with open('artifacts/quantile_cost_sensitivity.json','w') as f:
    json.dump(rows, f, indent=2)

# Plot: total_daily_net vs total_cost for each quantile
plt.figure(figsize=(8,5))
for q in quantiles:
    dfq = res_df[res_df['quantile']==q].copy()
    # group by total_cost and average
    dfg = dfq.groupby('total_cost').agg({'total_daily_net':'mean','trades':'mean'}).reset_index()
    plt.plot(dfg['total_cost'], dfg['total_daily_net'], marker='o', label=f'top {int(q*100)}%')

plt.xlabel('Total Cost (tc + slippage)')
plt.ylabel('Total Daily Net Return (compounded)')
plt.title('Quantile Cost Sensitivity')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('artifacts/quantile_cost_sensitivity.png')

print('Saved artifacts/quantile_cost_sensitivity.csv, .json and .png')
print('\nTop results sample:')
print(res_df.sort_values(['quantile','total_cost']).groupby('quantile').head(5).to_string(index=False))