#!/usr/bin/env python3
"""
Backtesting Script for LightGBM KC Stock Predictions

Evaluates model predictions against actual returns and generates trading metrics.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime

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

print(f"✅ Loaded {len(backtest_df)} test samples")

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

print(f"\nCumulative Returns (from start of test period):")
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

print(f"\nMaximum Drawdown:")
print(f"   Model Strategy:  {strategy_mdd:.2%}")
print(f"   Buy & Hold:      {buyhold_mdd:.2%}")

# ============================================================================
# SECTION 4: Threshold-Based Strategy Analysis
# ============================================================================
print("\n" + "="*70)
print("🎯 SECTION 4: CONFIDENCE-BASED TRADING")
print("="*70)

# Only trade when confidence is high
for confidence_threshold in [0.50, 0.55, 0.60, 0.65]:
    high_conf = backtest_df[
        (backtest_df['predicted_proba'] >= confidence_threshold) |
        (backtest_df['predicted_proba'] <= (1 - confidence_threshold))
    ].copy()
    
    if len(high_conf) == 0:
        continue
    
    high_conf['strategy_return'] = high_conf['predicted_direction'] * high_conf['actual_return']
    accuracy = (high_conf['predicted_direction'] == high_conf['actual_direction']).sum() / len(high_conf)
    avg_return = high_conf['strategy_return'].mean()
    total_return = (1 + high_conf['strategy_return']).prod() - 1
    win_rate = (high_conf['strategy_return'] > 0).sum() / len(high_conf)
    
    print(f"\nConfidence Threshold: {confidence_threshold:.0%}")
    print(f"   Trades: {len(high_conf):,} ({len(high_conf)/len(backtest_df)*100:.1f}% of all days)")
    print(f"   Accuracy: {accuracy:.2%}")
    print(f"   Win Rate: {win_rate:.2%}")
    print(f"   Avg Daily Return: {avg_return:+.4%}")
    print(f"   Total Return: {total_return:+.2%}")

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
# SECTION 6: Visualizations
# ============================================================================
print("\n📈 Generating Visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Cumulative returns
axes[0, 0].plot(backtest_df['strategy_cumulative'], label='Model Strategy', linewidth=2)
axes[0, 0].plot(backtest_df['buyhold_cumulative'], label='Buy & Hold', linewidth=2)
axes[0, 0].set_title('Cumulative Returns Comparison')
axes[0, 0].set_ylabel('Cumulative Return (x)')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Daily returns distribution
axes[0, 1].hist(backtest_df['strategy_return']*100, bins=50, alpha=0.6, label='Model Strategy', edgecolor='black')
axes[0, 1].hist(backtest_df['buy_hold_return']*100, bins=50, alpha=0.6, label='Buy & Hold', edgecolor='black')
axes[0, 1].set_title('Daily Returns Distribution')
axes[0, 1].set_xlabel('Daily Return (%)')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3, axis='y')

# Monthly returns
monthly_dates = pd.to_datetime(monthly_perf.index)
x_pos = np.arange(len(monthly_perf))
axes[1, 0].bar(x_pos - 0.2, monthly_perf['strategy_return']*100, 0.4, label='Model Strategy', alpha=0.8)
axes[1, 0].bar(x_pos + 0.2, monthly_perf['buy_hold_return']*100, 0.4, label='Buy & Hold', alpha=0.8)
axes[1, 0].set_title('Monthly Returns Comparison')
axes[1, 0].set_ylabel('Monthly Return (%)')
axes[1, 0].set_xticks(x_pos)
axes[1, 0].set_xticklabels(monthly_perf.index, rotation=45)
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3, axis='y')
axes[1, 0].axhline(y=0, color='k', linestyle='-', linewidth=0.5)

# Win rate by prediction type
pred_types = ['Predicted UP', 'Predicted DOWN']
if len(up_preds) > 0:
    up_wins_pct = (up_preds['actual_return'] > 0).sum() / len(up_preds) * 100
else:
    up_wins_pct = 0
    
if len(down_preds) > 0:
    down_wins_pct = (down_preds['actual_return'] < 0).sum() / len(down_preds) * 100
else:
    down_wins_pct = 0

win_rates = [up_wins_pct, down_wins_pct]
colors = ['green' if wr > 50 else 'red' for wr in win_rates]
axes[1, 1].bar(pred_types, win_rates, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
axes[1, 1].axhline(y=50, color='black', linestyle='--', linewidth=2, label='Random Chance')
axes[1, 1].set_title('Win Rate by Prediction Type')
axes[1, 1].set_ylabel('Win Rate (%)')
axes[1, 1].set_ylim([0, 100])
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3, axis='y')

# Add percentage labels on bars
for i, (pred_type, wr) in enumerate(zip(pred_types, win_rates)):
    axes[1, 1].text(i, wr + 2, f'{wr:.1f}%', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('./artifacts/backtest_analysis_kc.png', dpi=150, bbox_inches='tight')
print("✅ Backtest analysis plot saved to artifacts/backtest_analysis_kc.png")
plt.close()

# ============================================================================
# SECTION 7: Summary Report
# ============================================================================
print("\n" + "="*70)
print("📊 BACKTEST SUMMARY REPORT")
print("="*70)

print(f"""
Performance Metrics:
  Total Test Samples: {len(backtest_df):,}
  Date Range: {backtest_df['trade_date'].min()} to {backtest_df['trade_date'].max()}
  
Accuracy:
  Direction Accuracy: {accuracy:.2%}
  Up Prediction Win Rate: {(up_preds['actual_return'] > 0).sum() / len(up_preds) * 100:.2f}%
  Down Prediction Win Rate: {(down_preds['actual_return'] < 0).sum() / len(down_preds) * 100:.2f}%

Returns:
  Model Strategy Total Return: {strategy_total_return:+.2%}
  Buy & Hold Total Return: {buyhold_total_return:+.2%}
  Outperformance: {strategy_total_return - buyhold_total_return:+.2%}
  
  Model Daily Mean Return: {strategy_mean_daily:+.4%}
  Buy & Hold Daily Mean Return: {buyhold_mean_daily:+.4%}

Risk-Adjusted:
  Model Sharpe Ratio: {strategy_sharpe:.3f}
  Buy & Hold Sharpe Ratio: {buyhold_sharpe:.3f}
  
  Model Max Drawdown: {strategy_mdd:.2%}
  Buy & Hold Max Drawdown: {buyhold_mdd:.2%}

Best Performing Month: {monthly_perf['strategy_return'].idxmax()} ({monthly_perf['strategy_return'].max():+.2%})
Worst Performing Month: {monthly_perf['strategy_return'].idxmin()} ({monthly_perf['strategy_return'].min():+.2%})

Key Insight: 
  The model shows {('POSITIVE' if strategy_total_return > buyhold_total_return else 'NEGATIVE')} 
  outperformance vs buy & hold, with {('BETTER' if strategy_sharpe > buyhold_sharpe else 'WORSE')} 
  risk-adjusted returns.
""")

# Save backtest results
backtest_df.to_csv('./artifacts/backtest_results_kc.csv', index=False)
print(f"\n✅ Backtest results saved to artifacts/backtest_results_kc.csv")

summary_stats = {
    'total_samples': len(backtest_df),
    'direction_accuracy': accuracy,
    'model_total_return': strategy_total_return,
    'buyhold_total_return': buyhold_total_return,
    'outperformance': strategy_total_return - buyhold_total_return,
    'model_sharpe_ratio': strategy_sharpe,
    'buyhold_sharpe_ratio': buyhold_sharpe,
    'model_max_drawdown': strategy_mdd,
    'buyhold_max_drawdown': buyhold_mdd,
}

import json
with open('./artifacts/backtest_summary_kc.json', 'w') as f:
    json.dump(summary_stats, f, indent=2, default=float)

print(f"✅ Summary statistics saved to artifacts/backtest_summary_kc.json")

print("\n" + "="*70)
print("✅ BACKTEST COMPLETE")
print("="*70)
