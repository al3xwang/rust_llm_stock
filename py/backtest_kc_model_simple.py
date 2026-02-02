#!/usr/bin/env python3
"""
Backtesting Script for LightGBM KC Stock Predictions (Simplified)
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path

print("="*70)
print("📊 LIGHTGBM KC MODEL BACKTEST")
print("="*70)

# Load test data
print("\n📁 Loading Data...")
test_data = pd.read_csv("./data/kc_test.csv")
predictions = pd.read_csv("./artifacts/test_predictions_kc_final.csv")

# Merge predictions with actual returns
backtest_df = pd.DataFrame({
    'trade_date': test_data['trade_date'].values,
    'ts_code': test_data['ts_code'].values,
    'actual_direction': test_data['next_day_direction'].values,
    'predicted_direction': predictions['predicted'].values,
    'predicted_proba': predictions['predicted_proba'].values,
    'actual_return': test_data['next_day_return'].values,
})

# Remove rows with missing data
backtest_df = backtest_df.dropna()
print(f"✅ Loaded {len(backtest_df):,} test samples")

# ============================================================================
# SECTION 1: Prediction Accuracy Analysis
# ============================================================================
print("\n" + "="*70)
print("📈 SECTION 1: PREDICTION ACCURACY")
print("="*70)

# Direction accuracy
correct_preds = (backtest_df['predicted_direction'] == backtest_df['actual_direction']).sum()
total_preds = len(backtest_df)
accuracy = correct_preds / total_preds

print(f"\n📊 Direction Prediction Accuracy:")
print(f"   Correct: {correct_preds:,} / {total_preds:,}")
print(f"   Accuracy: {accuracy:.2%}")

# Breakdown by predicted class
print(f"\n📋 Prediction Distribution:")
pred_counts = backtest_df['predicted_direction'].value_counts()
print(f"   Predicted Down (-1): {pred_counts.get(-1, 0):,} ({pred_counts.get(-1, 0)/len(backtest_df)*100:.1f}%)")
print(f"   Predicted Up   (+1): {pred_counts.get(1, 0):,} ({pred_counts.get(1, 0)/len(backtest_df)*100:.1f}%)")

actual_counts = backtest_df['actual_direction'].value_counts()
print(f"\n   Actual Down (-1): {actual_counts.get(-1, 0):,} ({actual_counts.get(-1, 0)/len(backtest_df)*100:.1f}%)")
print(f"   Actual Up   (+1): {actual_counts.get(1, 0):,} ({actual_counts.get(1, 0)/len(backtest_df)*100:.1f}%)")

# ============================================================================
# SECTION 2: Returns Analysis by Prediction
# ============================================================================
print("\n" + "="*70)
print("💰 SECTION 2: RETURNS ANALYSIS")
print("="*70)

# Average return by predicted direction
up_preds = backtest_df[backtest_df['predicted_direction'] == 1]
down_preds = backtest_df[backtest_df['predicted_direction'] == -1]

up_return_mean = up_preds['actual_return'].mean() if len(up_preds) > 0 else 0
down_return_mean = down_preds['actual_return'].mean() if len(down_preds) > 0 else 0

print(f"\nAverage Returns by Prediction:")
print(f"   When predicted UP:   {up_return_mean:+.4%} (n={len(up_preds):,})")
print(f"   When predicted DOWN: {down_return_mean:+.4%} (n={len(down_preds):,})")

# Win rate on predictions
if len(up_preds) > 0:
    up_wins = (up_preds['actual_return'] > 0).sum()
    up_win_rate = up_wins / len(up_preds)
    print(f"\n   UP Prediction Win Rate: {up_wins:,}/{len(up_preds):,} = {up_win_rate:.2%}")

if len(down_preds) > 0:
    down_wins = (down_preds['actual_return'] < 0).sum()
    down_win_rate = down_wins / len(down_preds)
    print(f"   DOWN Prediction Win Rate: {down_wins:,}/{len(down_preds):,} = {down_win_rate:.2%}")

# ============================================================================
# SECTION 3: Trading Strategy Performance
# ============================================================================
print("\n" + "="*70)
print("📊 SECTION 3: TRADING STRATEGY PERFORMANCE")
print("="*70)

# Strategy: Buy/Long when predicted UP, Short/Sell when predicted DOWN
backtest_df['strategy_return'] = backtest_df['predicted_direction'] * backtest_df['actual_return']
backtest_df['buy_hold_return'] = backtest_df['actual_return']

# Cumulative returns
backtest_df['strategy_cumulative'] = (1 + backtest_df['strategy_return']).cumprod()
backtest_df['buyhold_cumulative'] = (1 + backtest_df['buy_hold_return']).cumprod()

strategy_total_return = backtest_df['strategy_cumulative'].iloc[-1] - 1
buyhold_total_return = backtest_df['buyhold_cumulative'].iloc[-1] - 1

# Safe calculation avoiding overflow
strategy_total_return = backtest_df['strategy_return'].sum()
buyhold_total_return = backtest_df['buy_hold_return'].sum()

print(f"\nCumulative Returns (total sum of daily returns):")
print(f"   Model Strategy:  {strategy_total_return:+.2%}")
print(f"   Buy & Hold:      {buyhold_total_return:+.2%}")
print(f"   Outperformance:  {strategy_total_return - buyhold_total_return:+.2%}")

# Daily statistics
strategy_mean_daily = backtest_df['strategy_return'].mean()
strategy_std_daily = backtest_df['strategy_return'].std()
buyhold_mean_daily = backtest_df['buy_hold_return'].mean()
buyhold_std_daily = backtest_df['buy_hold_return'].std()

# Sharpe ratio (assuming 0 risk-free rate, daily data)
annual_trading_days = 252
strategy_sharpe = strategy_mean_daily / strategy_std_daily * np.sqrt(annual_trading_days) if strategy_std_daily > 0 else 0
buyhold_sharpe = buyhold_mean_daily / buyhold_std_daily * np.sqrt(annual_trading_days) if buyhold_std_daily > 0 else 0

print(f"\nRisk-Adjusted Metrics (Annualized):")
print(f"   Model Sharpe Ratio:  {strategy_sharpe:.3f}")
print(f"   Buy & Hold Sharpe:   {buyhold_sharpe:.3f}")

# Maximum Drawdown
def calculate_max_drawdown(cumulative_returns):
    running_max = cumulative_returns.expanding().max()
    drawdown = (cumulative_returns - running_max) / running_max
    return drawdown.min()

strategy_mdd = calculate_max_drawdown(backtest_df['strategy_cumulative'])
buyhold_mdd = calculate_max_drawdown(backtest_df['buyhold_cumulative'])

# Calculate returns with compounding for cumulative analysis
backtest_df['strategy_log_return'] = np.log(1 + backtest_df['strategy_return'])
backtest_df['buyhold_log_return'] = np.log(1 + backtest_df['buy_hold_return'])
backtest_df['strategy_cumulative_log'] = backtest_df['strategy_log_return'].cumsum()
backtest_df['buyhold_cumulative_log'] = backtest_df['buyhold_log_return'].cumsum()
backtest_df['strategy_cumulative'] = np.exp(backtest_df['strategy_cumulative_log'])
backtest_df['buyhold_cumulative'] = np.exp(backtest_df['buyhold_cumulative_log'])

strategy_mdd = calculate_max_drawdown(backtest_df['strategy_cumulative'])
buyhold_mdd = calculate_max_drawdown(backtest_df['buyhold_cumulative'])

print(f"\nMaximum Drawdown (with proper compounding):")
print(f"   Model Strategy:  {strategy_mdd:.2%}")
print(f"   Buy & Hold:      {buyhold_mdd:.2%}")

# Win rate on strategy
strategy_win_rate = (backtest_df['strategy_return'] > 0).sum() / len(backtest_df)
buyhold_win_rate = (backtest_df['buy_hold_return'] > 0).sum() / len(backtest_df)

print(f"\nDaily Win Rate:")
print(f"   Model Strategy:  {strategy_win_rate:.2%} ({(backtest_df['strategy_return'] > 0).sum():,} winning days)")
print(f"   Buy & Hold:      {buyhold_win_rate:.2%} ({(backtest_df['buy_hold_return'] > 0).sum():,} winning days)")

# ============================================================================
# SECTION 4: Threshold-Based Strategy Analysis
# ============================================================================
print("\n" + "="*70)
print("🎯 SECTION 4: CONFIDENCE-BASED TRADING")
print("="*70)

print(f"\nOnly trade when model confidence is high:")
for confidence_threshold in [0.50, 0.55, 0.60, 0.65]:
    high_conf = backtest_df[
        (backtest_df['predicted_proba'] >= confidence_threshold) |
        (backtest_df['predicted_proba'] <= (1 - confidence_threshold))
    ].copy()
    
    if len(high_conf) == 0:
        continue
    
    high_conf['strategy_return'] = high_conf['predicted_direction'] * high_conf['actual_return']
    accuracy_conf = (high_conf['predicted_direction'] == high_conf['actual_direction']).sum() / len(high_conf)
    avg_return_conf = high_conf['strategy_return'].mean()
    total_return_conf = (1 + high_conf['strategy_return']).prod() - 1
    win_rate_conf = (high_conf['strategy_return'] > 0).sum() / len(high_conf)
    
    print(f"\n   Confidence Threshold: {confidence_threshold:.0%}")
    print(f"   Trades: {len(high_conf):,} ({len(high_conf)/len(backtest_df)*100:.1f}% of all days)")
    print(f"   Accuracy: {accuracy_conf:.2%}")
    print(f"   Win Rate: {win_rate_conf:.2%}")
    print(f"   Avg Daily Return: {avg_return_conf:+.4%}")
    print(f"   Total Return: {total_return_conf:+.2%}")

# ============================================================================
# SECTION 5: Monthly Performance
# ============================================================================
print("\n" + "="*70)
print("📅 SECTION 5: MONTHLY PERFORMANCE")
print("="*70)

backtest_df['trade_date_dt'] = pd.to_datetime(backtest_df['trade_date'], format='%Y%m%d')
backtest_df['year_month'] = backtest_df['trade_date_dt'].dt.strftime('%Y-%m')

monthly_perf = backtest_df.groupby('year_month').agg({
    'strategy_return': 'sum',
    'buy_hold_return': 'sum',
    'actual_direction': 'count'
}).rename(columns={'actual_direction': 'trades'})

monthly_perf['outperformance'] = monthly_perf['strategy_return'] - monthly_perf['buy_hold_return']

print("\nMonthly Returns (Strategy vs Buy & Hold):")
print(f"{'Month':<10} {'Strategy':>12} {'Buy&Hold':>12} {'Outperf':>12} {'Trades':>8}")
print("-" * 54)
for idx, row in monthly_perf.iterrows():
    print(f"{idx:<10} {row['strategy_return']:>11.2%} {row['buy_hold_return']:>11.2%} {row['outperformance']:>11.2%} {int(row['trades']):>7}")

# ============================================================================
# SECTION 6: Summary Statistics
# ============================================================================
print("\n" + "="*70)
print("📊 SUMMARY STATISTICS")
print("="*70)

print(f"""
Test Period: {backtest_df['trade_date'].min()} to {backtest_df['trade_date'].max()}
Total Samples: {len(backtest_df):,}

Direction Prediction Accuracy: {accuracy:.2%}

Returns Performance:
  Model Strategy Total Return: {strategy_total_return:+.2%}
  Buy & Hold Total Return: {buyhold_total_return:+.2%}
  Outperformance: {strategy_total_return - buyhold_total_return:+.2%}
  
  Model Daily Mean Return: {strategy_mean_daily:+.4%}
  Model Daily Std Dev: {strategy_std_daily:.4%}
  
  Buy & Hold Daily Mean Return: {buyhold_mean_daily:+.4%}
  Buy & Hold Daily Std Dev: {buyhold_std_daily:.4%}

Risk-Adjusted Metrics (Annualized):
  Model Sharpe Ratio: {strategy_sharpe:.3f}
  Buy & Hold Sharpe Ratio: {buyhold_sharpe:.3f}
  
  Model Max Drawdown: {strategy_mdd:.2%}
  Buy & Hold Max Drawdown: {buyhold_mdd:.2%}

Win Rates:
  Model Strategy: {strategy_win_rate:.2%}
  Buy & Hold: {buyhold_win_rate:.2%}

Best Month: {monthly_perf['strategy_return'].idxmax()} ({monthly_perf['strategy_return'].max():+.2%})
Worst Month: {monthly_perf['strategy_return'].idxmin()} ({monthly_perf['strategy_return'].min():+.2%})

Conclusion:
  The model strategy shows {('POSITIVE' if strategy_total_return > buyhold_total_return else 'NEGATIVE')}
  outperformance with {strategy_sharpe:.2f} Sharpe ratio vs {buyhold_sharpe:.2f} for buy & hold.
""")

# Save results
backtest_df.to_csv('./artifacts/backtest_results_kc.csv', index=False)
print(f"\n✅ Backtest results saved to artifacts/backtest_results_kc.csv")

summary_stats = {
    'test_period': f"{backtest_df['trade_date'].min()} to {backtest_df['trade_date'].max()}",
    'total_samples': int(len(backtest_df)),
    'direction_accuracy': float(accuracy),
    'model_total_return': float(strategy_total_return),
    'buyhold_total_return': float(buyhold_total_return),
    'outperformance': float(strategy_total_return - buyhold_total_return),
    'model_sharpe_ratio': float(strategy_sharpe),
    'buyhold_sharpe_ratio': float(buyhold_sharpe),
    'model_max_drawdown': float(strategy_mdd),
    'buyhold_max_drawdown': float(buyhold_mdd),
    'model_win_rate': float(strategy_win_rate),
    'buyhold_win_rate': float(buyhold_win_rate),
}

with open('./artifacts/backtest_summary_kc.json', 'w') as f:
    json.dump(summary_stats, f, indent=2)

print(f"✅ Summary statistics saved to artifacts/backtest_summary_kc.json")

print("\n" + "="*70)
print("✅ BACKTEST COMPLETE")
print("="*70)
