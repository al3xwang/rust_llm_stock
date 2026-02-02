#!/usr/bin/env python3
"""
Quantile diagnostics for ensemble predictions
- Loads ensemble probs and test data
- For each quantile (top q%), computes:
  - number of trades
  - per-trade mean return, std, win rate (before costs)
  - per-trade net return after (tc+slippage) round-trip
  - daily aggregated returns and cumulative
  - top instruments by P&L contribution
- Saves diagnostics in artifacts/quantile_diagnostics_*.json and CSVs
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd

Path('artifacts').mkdir(exist_ok=True)

# Config
quantiles = [0.05, 0.1, 0.2]
transaction_cost = 0.0005
slippage = 0.0005

# Load ensemble probabilities (from stacking script)
ens = pd.read_csv('artifacts/test_predictions_kc_mixed_ensemble_v2.csv') if Path('artifacts/test_predictions_kc_mixed_ensemble_v2.csv').exists() else pd.read_csv('artifacts/test_predictions_kc_ensemble.csv')
# Ensure columns present
if 'predicted_proba_ensemble' in ens.columns:
    proba_col = 'predicted_proba_ensemble'
elif 'predicted_proba' in ens.columns:
    proba_col = 'predicted_proba'
elif 'predicted_proba_calibrated' in ens.columns:
    proba_col = 'predicted_proba_calibrated'
else:
    raise SystemExit('No ensemble probability column found')

# Load test returns
df_test = pd.read_csv('data/kc_test.csv')

# Merge
bt = pd.DataFrame({
    'ts_code': df_test['ts_code'].values,
    'trade_date': df_test['trade_date'].values,
    'return': df_test['next_day_return'].values,
    'proba': ens[proba_col].values
})

# Sort for grouping operations
bt = bt.sort_values(['ts_code','trade_date']).reset_index(drop=True)

results = {}

for q in quantiles:
    thr = bt['proba'].quantile(1 - q)
    sel = bt[bt['proba'] >= thr].copy()
    sel['pos'] = 1
    # compute per-instrument prev pos to detect trades
    sel_full = bt.copy()
    sel_full['pos'] = (sel_full['proba'] >= thr).astype(int)
    sel_full['prev_pos'] = sel_full.groupby('ts_code')['pos'].shift(1).fillna(0).astype(int)
    sel_full['trade_flag'] = (sel_full['pos'] != sel_full['prev_pos']).astype(int)
    # per-instrument per-trade returns (for days where pos ==1)
    trades = sel_full[sel_full['pos'] == 1].copy()
    trades['gross_return'] = trades['return']
    trades['net_return'] = trades['gross_return'] - trades['trade_flag'] * 2 * (transaction_cost + slippage)

    # aggregate stats
    total_trades = len(trades)
    mean_gross = float(trades['gross_return'].mean()) if total_trades>0 else 0.0
    std_gross = float(trades['gross_return'].std()) if total_trades>0 else 0.0
    win_rate = float((trades['gross_return']>0).sum() / total_trades) if total_trades>0 else 0.0
    mean_net = float(trades['net_return'].mean()) if total_trades>0 else 0.0
    total_net = float(trades['net_return'].sum()) if total_trades>0 else 0.0

    # daily aggregated returns
    daily = trades.groupby('trade_date').agg({'net_return':'mean','gross_return':'mean','trade_flag':'sum'}).rename(columns={'trade_flag':'num_trades'})
    daily['cum_net'] = np.exp(np.nancumsum(np.log1p(np.clip(daily['net_return'], -0.9, 0.9)))) - 1
    total_daily_net = float(daily['cum_net'].iloc[-1]) if len(daily)>0 else 0.0

    # top losing instruments
    instrument_pnl = trades.groupby('ts_code').agg({'net_return':['sum','mean','count']})
    instrument_pnl.columns = ['net_sum','net_mean','count']
    worst_instruments = instrument_pnl.sort_values('net_sum').head(20).reset_index().to_dict(orient='records')

    results[q] = {
        'threshold': float(thr),
        'total_trades': int(total_trades),
        'mean_gross': mean_gross,
        'std_gross': std_gross,
        'win_rate': win_rate,
        'mean_net_per_trade': mean_net,
        'total_net': total_net,
        'total_daily_net': total_daily_net,
        'num_active_days': int(daily.shape[0]),
        'worst_instruments': worst_instruments
    }

    # Save diagnostics CSVs
    trades.to_csv(f'artifacts/quantile_{int(q*100)}_trades.csv', index=False)
    daily.to_csv(f'artifacts/quantile_{int(q*100)}_daily.csv')

# Save JSON summary
with open('artifacts/quantile_diagnostics_summary.json','w') as f:
    json.dump(results, f, indent=2)

print('Diagnostics saved to artifacts/quantile_diagnostics_summary.json and per-quantile CSVs')
print('Summary:')
for q, r in results.items():
    print(f"Top {int(q*100)}% -> trades={r['total_trades']}, mean_net={r['mean_net_per_trade']:.4%}, win_rate={r['win_rate']:.2%}, total_daily_net={r['total_daily_net']:.2%}")
