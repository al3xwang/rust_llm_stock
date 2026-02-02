#!/usr/bin/env python3
"""
Quantile 1% diagnostics
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd

Path('artifacts').mkdir(exist_ok=True)

# config
q = 0.01
tc = 0.0005
slippage = 0.0005

# load ensemble probs
if Path('artifacts/test_predictions_kc_mixed_ensemble_v2.csv').exists():
    ens = pd.read_csv('artifacts/test_predictions_kc_mixed_ensemble_v2.csv')
elif Path('artifacts/test_predictions_kc_ensemble.csv').exists():
    ens = pd.read_csv('artifacts/test_predictions_kc_ensemble.csv')
else:
    raise SystemExit('No ensemble predictions available')

if 'predicted_proba_ensemble' in ens.columns:
    proba_col = 'predicted_proba_ensemble'
elif 'predicted_proba' in ens.columns:
    proba_col = 'predicted_proba'
elif 'predicted_proba_calibrated' in ens.columns:
    proba_col = 'predicted_proba_calibrated'
else:
    raise SystemExit('No probability column found')

# load test
df_test = pd.read_csv('data/kc_test.csv')

bt = pd.DataFrame({'ts_code': df_test['ts_code'].values,
                   'trade_date': df_test['trade_date'].values,
                   'return': df_test['next_day_return'].values,
                   'proba': ens[proba_col].values})

bt = bt.sort_values(['ts_code','trade_date']).reset_index(drop=True)
thr = bt['proba'].quantile(1 - q)
sel = bt[bt['proba'] >= thr].copy()
sel_full = bt.copy()
sel_full['pos'] = (sel_full['proba'] >= thr).astype(int)
sel_full['prev_pos'] = sel_full.groupby('ts_code')['pos'].shift(1).fillna(0).astype(int)
sel_full['trade_flag'] = (sel_full['pos'] != sel_full['prev_pos']).astype(int)
sel_full['gross_return'] = sel_full['pos'] * sel_full['return']
sel_full['net_return'] = sel_full['gross_return'] - sel_full['trade_flag'] * 2 * (tc + slippage)

trades = sel_full[sel_full['pos'] == 1].copy()
total_trades = len(trades)
mean_gross = float(trades['gross_return'].mean()) if total_trades>0 else 0.0
std_gross = float(trades['gross_return'].std()) if total_trades>0 else 0.0
win_rate = float((trades['gross_return']>0).sum() / total_trades) if total_trades>0 else 0.0
mean_net = float(trades['net_return'].mean()) if total_trades>0 else 0.0
sum_net = float(trades['net_return'].sum()) if total_trades>0 else 0.0

# daily
daily = trades.groupby('trade_date').agg({'net_return':'mean','gross_return':'mean','trade_flag':'sum'})
daily = daily.rename(columns={'trade_flag':'num_trades'})
if len(daily)>0:
    daily['cum_net'] = np.exp(np.nancumsum(np.log1p(np.clip(daily['net_return'], -0.9, 0.9)))) - 1
    total_daily_net = float(daily['cum_net'].iloc[-1])
else:
    total_daily_net = 0.0

# worst instruments
instrument_pnl = trades.groupby('ts_code').agg({'net_return':['sum','mean','count']})
instrument_pnl.columns = ['net_sum','net_mean','count']
worst_instr = instrument_pnl.sort_values('net_sum').head(50).reset_index().to_dict(orient='records')

summary = {
    'quantile': q,
    'threshold': float(thr),
    'total_trades': int(total_trades),
    'mean_gross_return': mean_gross,
    'std_gross_return': std_gross,
    'win_rate': win_rate,
    'mean_net_per_trade': mean_net,
    'sum_net': sum_net,
    'total_daily_net': total_daily_net,
    'num_active_days': int(daily.shape[0])
}

with open('artifacts/quantile_1pct_summary.json','w') as f:
    json.dump({'summary':summary,'worst_instruments':worst_instr}, f, indent=2)

trades.to_csv('artifacts/quantile_1pct_trades.csv', index=False)
daily.to_csv('artifacts/quantile_1pct_daily.csv')

print('Saved quantile 1% diagnostics: artifacts/quantile_1pct_summary.json and trades/daily CSVs')
print('Summary:')
for k,v in summary.items():
    print(f"  {k}: {v}")
print('\nTop 5 worst instruments:')
for r in worst_instr[:5]:
    print(f"  {r['ts_code']}: sum={r['net_sum']:.4f}, count={r['count']}")