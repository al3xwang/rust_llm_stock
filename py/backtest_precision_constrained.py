#!/usr/bin/env python3
"""
Backtest with precision-optimized threshold subject to minimum recall constraint
Loads stacking features, retrains stacker on validation features, searches thresholds
that maximize precision while keeping recall >= R_min. Runs backtest with fees/slippage.
"""

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, accuracy_score

# Params
parser = argparse.ArgumentParser()
parser.add_argument('--min-recall', type=float, default=0.6, help='Minimum recall constraint (0-1)')
parser.add_argument('--tc', type=float, default=0.0005, help='Transaction cost per trade (one-way fraction)')
parser.add_argument('--slippage', type=float, default=0.0005, help='Slippage per trade (one-way fraction)')
parser.add_argument('--quantiles', nargs='+', type=float, default=[0.05,0.1,0.2], help='Quantiles to test')
args = parser.parse_args()

Path('artifacts').mkdir(exist_ok=True)

print('='*70)
print('🔒 PRECISION-CONSTRAINED BACKTEST')
print('='*70)

# Load features & data
val_feat = pd.read_csv('./artifacts/stack_val_features.csv')
test_feat = pd.read_csv('./artifacts/stack_test_features.csv')
df_test = pd.read_csv('./data/kc_test.csv')
df_val = pd.read_csv('./data/kc_val.csv')

y_val = df_val['next_day_direction'].map({-1:0,1:1}).astype(int)

# train stacker
X_val = val_feat.drop(columns=['y']) if 'y' in val_feat.columns else val_feat
stacker = LogisticRegression(solver='lbfgs', max_iter=500)
stacker.fit(X_val, y_val)

ens_val_proba = stacker.predict_proba(X_val)[:,1]
ens_test_proba = stacker.predict_proba(test_feat)[:,1]

# Threshold search: maximize precision subject to recall >= min_recall
min_recall = args.min_recall
best_prec = -1
best_thr = None
best_metrics = None
for thr in np.arange(0.0,1.001,0.001):
    preds = (ens_val_proba >= thr).astype(int)
    rec = recall_score(y_val, preds, zero_division=0)
    if rec >= min_recall:
        prec = precision_score(y_val, preds, zero_division=0)
        acc = accuracy_score(y_val, preds)
        if prec > best_prec:
            best_prec = prec
            best_thr = thr
            best_metrics = {'precision': prec, 'recall': rec, 'accuracy': acc}

if best_thr is None:
    print(f"No threshold satisfies recall >= {min_recall}")
    best_thr = 0.5
    best_metrics = {'precision': precision_score(y_val, (ens_val_proba>=best_thr).astype(int), zero_division=0), 'recall': recall_score(y_val, (ens_val_proba>=best_thr).astype(int), zero_division=0), 'accuracy': accuracy_score(y_val, (ens_val_proba>=best_thr).astype(int))}

print(f"Selected threshold {best_thr:.3f} on val -> precision={best_metrics['precision']:.4f}, recall={best_metrics['recall']:.4f}, acc={best_metrics['accuracy']:.4f}")

# Get test preds using selected threshold
test_preds = (ens_test_proba >= best_thr).astype(int)
# Map to +1/-1 positions
positions = np.where(test_preds==1, 1, -1)

# Build a DataFrame aligned with test set
bt = df_test[['ts_code','trade_date','next_day_return']].copy()
bt['pred_label'] = test_preds
bt['position'] = positions

# compute per-instrument position changes (prev pos) to apply per-instrument transaction costs
bt = bt.sort_values(['ts_code','trade_date']).reset_index(drop=True)
bt['prev_pos'] = bt.groupby('ts_code')['position'].shift(1).fillna(0).astype(int)
bt['trade_flag'] = (bt['position'] != bt['prev_pos']).astype(int)

# apply P&L and costs per row
tc = args.tc
slippage = args.slippage
bt['strategy_return'] = bt['position'] * bt['next_day_return'] - bt['trade_flag'] * 2 * (tc + slippage)

# Aggregate to daily portfolio (equal-weighted average across instruments per date)
daily = bt.groupby('trade_date').agg({
    'strategy_return':'mean',
    'next_day_return':'mean',
    'trade_flag':'sum'
}).rename(columns={'next_day_return':'buyhold_return'})

print(f"Number of unique trading dates: {len(daily)}")
max_ret = float(np.nanmax(daily['buyhold_return']))
min_ret = float(np.nanmin(daily['buyhold_return']))
print(f"Max single-day actual (daily average) return: {max_ret:.4f}, Min: {min_ret:.4f}")

# Clip daily returns to avoid explosion and compute compounding across dates
daily['strategy_return_clipped'] = np.clip(daily['strategy_return'], -0.9, 0.9)
daily['buyhold_return_clipped'] = np.clip(daily['buyhold_return'], -0.9, 0.9)

daily['strategy_cum'] = np.exp(np.nancumsum(np.log1p(daily['strategy_return_clipped'].values)))
daily['buyhold_cum'] = np.exp(np.nancumsum(np.log1p(daily['buyhold_return_clipped'].values)))

# metrics
total_strategy_return = float(daily['strategy_cum'].iloc[-1] - 1)
total_buyhold_return = float(daily['buyhold_cum'].iloc[-1] - 1)
mean_daily = float(daily['strategy_return'].mean())
std_daily = float(daily['strategy_return'].std())
annual_sharpe = float((mean_daily / std_daily) * (252**0.5) if std_daily>0 else 0)

# drawdown
strategy_cum_series = daily['strategy_cum']
running_max = strategy_cum_series.cummax()
drawdown = (strategy_cum_series - running_max) / running_max
mdd = float(drawdown.min())

# Save results
out_df = bt.copy()
# also add daily agg for convenience
out_df.to_csv('artifacts/backtest_precision_constrained_full.csv', index=False)

daily.to_csv('artifacts/backtest_precision_constrained_daily.csv')

summary = {
    'min_recall_constraint': float(min_recall),
    'selected_threshold': float(best_thr),
    'val_precision': float(best_metrics['precision']),
    'val_recall': float(best_metrics['recall']),
    'test_total_strategy_return': float(total_strategy_return),
    'test_total_buyhold_return': float(total_buyhold_return),
    'mean_daily_return': float(mean_daily),
    'std_daily_return': float(std_daily),
    'annual_sharpe': float(annual_sharpe),
    'max_drawdown': float(mdd),
    'n_trades': int(daily['trade_flag'].sum()),
}
with open('artifacts/backtest_precision_constrained_summary.json','w') as f:
    json.dump(summary, f, indent=2)

print('\nBacktest complete. Summary:')
for k,v in summary.items():
    print(f"  {k}: {v}")
# Quantile-based alternative: compute per-date aggregates for top quantiles
print('\nQuantile results (top only):')
for q in args.quantiles:
    thr_q = np.quantile(ens_test_proba, 1-q)
    preds_q = (ens_test_proba >= thr_q).astype(int)
    bt_q = bt.copy()
    bt_q['pos_q'] = preds_q
    bt_q = bt_q.sort_values(['ts_code','trade_date']).reset_index(drop=True)
    bt_q['prev_pos_q'] = bt_q.groupby('ts_code')['pos_q'].shift(1).fillna(0).astype(int)
    bt_q['trade_flag_q'] = (bt_q['pos_q'] != bt_q['prev_pos_q']).astype(int)
    bt_q['strat_q_ret'] = bt_q['pos_q'] * bt_q['next_day_return'] - bt_q['trade_flag_q'] * 2 * (tc + slippage)
    daily_q = bt_q.groupby('trade_date').agg({'strat_q_ret':'mean', 'pos_q':'sum'}).rename(columns={'pos_q':'num_long'})
    daily_q['strat_q_ret_clipped'] = np.clip(daily_q['strat_q_ret'], -0.9, 0.9)
    daily_q['cum'] = np.exp(np.nancumsum(np.log1p(daily_q['strat_q_ret_clipped'].values)))
    tot_q = float(daily_q['cum'].iloc[-1] - 1)
    print(f"  top {int(q*100)}%: trades={int(bt_q['pos_q'].sum())}, total_return={tot_q:.2%}")

print('\nResults saved to artifacts/backtest_precision_constrained.csv and _summary.json')
