#!/usr/bin/env python3
"""
Enhanced Backtest Analysis for KC Model
Shows actual insights about model performance
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path

print("="*70)
print("🔍 ENHANCED BACKTEST ANALYSIS - KC MODEL")
print("="*70)

# Load data
print("\n📁 Loading Data...")
test_data = pd.read_csv("./data/kc_test.csv")
predictions = pd.read_csv("./artifacts/test_predictions_kc_final.csv")

# Merge
# Determine which probability column to use (prefer calibrated if available)
proba_col_name = None
for cand in ['predicted_proba_calibrated', 'predicted_proba_raw', 'predicted_proba']:
    if cand in predictions.columns:
        proba_col_name = cand
        break
if proba_col_name is None:
    raise KeyError('No predicted probability column found in predictions CSV')

# Build dataframe
df = pd.DataFrame({
    'ts_code': test_data['ts_code'].values,
    'trade_date': test_data['trade_date'].values,
    'actual_return': test_data['next_day_return'].values,
    'actual_direction': test_data['next_day_direction'].values,
    'predicted_direction': predictions['predicted'].values,
    'predicted_proba': predictions[proba_col_name].values,
})
# keep original column name available
if proba_col_name != 'predicted_proba':
    df['proba_source'] = proba_col_name
else:
    df['proba_source'] = 'predicted_proba'

df = df.dropna()
print(f"✅ Loaded {len(df):,} test samples\n")

# ============================================================================
# Key Finding: Why The Model Behaves This Way
# ============================================================================
print("="*70)
print("🎯 KEY FINDING: MODEL BIAS ANALYSIS")
print("="*70)

pred_counts = df['predicted_direction'].value_counts()
actual_counts = df['actual_direction'].value_counts()

print(f"\nPrediction Distribution:")
print(f"  Predicted  -1: {pred_counts.get(-1, 0):>8,} ({pred_counts.get(-1, 0)/len(df)*100:>5.1f}%)")
print(f"  Predicted  +1: {pred_counts.get(1, 0):>8,} ({pred_counts.get(1, 0)/len(df)*100:>5.1f}%)")
print(f"\nActual Distribution:")
print(f"  Actual     -1: {actual_counts.get(-1, 0):>8,} ({actual_counts.get(-1, 0)/len(df)*100:>5.1f}%)")
print(f"  Actual     +1: {actual_counts.get(1, 0):>8,} ({actual_counts.get(1, 0)/len(df)*100:>5.1f}%)")

# Accuracy breakdown
print(f"\n📊 Accuracy Breakdown:")
correct = (df['predicted_direction'] == df['actual_direction']).sum()
total = len(df)
accuracy = correct / total
print(f"  Overall Accuracy: {correct:,} / {total:,} = {accuracy:.2%}")

# What does predicting all +1 get us?
up_correct = ((df['predicted_direction'] == 1) & (df['actual_direction'] == 1)).sum()
print(f"  Predicting +1 when actually +1: {up_correct:,}")

down_wrong = ((df['predicted_direction'] == 1) & (df['actual_direction'] == -1)).sum()
print(f"  Predicting +1 when actually -1: {down_wrong:,}")

# ============================================================================
# Strategy Evaluation: What If We DIDN'T Follow The Model?
# ============================================================================
print("\n" + "="*70)
print("💡 STRATEGIC INSIGHT: Model Effectiveness")
print("="*70)

# Calculate returns under different strategies
df['random_return'] = np.where(df['actual_direction'] == 1, df['actual_return'], -df['actual_return'])
df['actual_return_long'] = df['actual_return']  # Just buy and hold
df['model_return'] = df['predicted_direction'] * df['actual_return']  # Follow model

print(f"\n📈 Strategy Comparison (using actual returns):")
print(f"  Average daily return (buy & hold):  {df['actual_return_long'].mean():+.4%}")
print(f"  Average daily return (model):       {df['model_return'].mean():+.4%}")
print(f"  Difference:                         {(df['model_return'].mean() - df['actual_return_long'].mean()):+.4%}")

# Win rate comparison
buyhold_wins = (df['actual_return_long'] > 0).sum()
model_wins = (df['model_return'] > 0).sum()

print(f"\n  Buy & Hold win rate: {buyhold_wins:,}/{len(df):,} = {buyhold_wins/len(df):.2%}")
print(f"  Model win rate:      {model_wins:,}/{len(df):,} = {model_wins/len(df):.2%}")
print(f"  Difference:          {(model_wins - buyhold_wins):+,} days")

# ============================================================================
# Returns Analysis: Where Model Is Right vs Wrong
# ============================================================================
print("\n" + "="*70)
print("🔬 RETURNS ANALYSIS: When Model Is Correct vs Incorrect")
print("="*70)

# When model predicts UP (which is always in our case)
up_preds = df[df['predicted_direction'] == 1].copy()
down_preds = df[df['predicted_direction'] == -1].copy()

if len(up_preds) > 0:
    up_avg_return = up_preds['actual_return'].mean()
    up_win_pct = (up_preds['actual_return'] > 0).sum() / len(up_preds)
    print(f"\nWhen Model Predicts UP ({len(up_preds):,} times):")
    print(f"  Avg Return:    {up_avg_return:+.4%}")
    print(f"  Win Rate:      {up_win_pct:.2%}")
    print(f"  Correct:       {((up_preds['predicted_direction'] == up_preds['actual_direction']).sum()):,}")
    print(f"  Wrong:         {((up_preds['predicted_direction'] != up_preds['actual_direction']).sum()):,}")

if len(down_preds) > 0:
    down_avg_return = down_preds['actual_return'].mean()
    down_win_pct = (down_preds['actual_return'] < 0).sum() / len(down_preds) if len(down_preds) > 0 else 0
    print(f"\nWhen Model Predicts DOWN ({len(down_preds):,} times):")
    print(f"  Avg Return:    {down_avg_return:+.4%}")
    print(f"  Win Rate:      {down_win_pct:.2%}")

# ============================================================================
# Confidence Analysis
# ============================================================================
print("\n" + "="*70)
print("📊 CONFIDENCE ANALYSIS")
print("="*70)

# Choose calibrated proba if available
proba_col = 'predicted_proba_calibrated' if 'predicted_proba_calibrated' in df.columns else 'predicted_proba'
print(f"\nPredicted Probability Statistics (using: {proba_col}):")
print(f"  Min:    {df[proba_col].min():.4f}")
print(f"  Max:    {df[proba_col].max():.4f}")
print(f"  Mean:   {df[proba_col].mean():.4f}")
print(f"  Median: {df[proba_col].median():.4f}")
print(f"  Std:    {df[proba_col].std():.4f}")

# High confidence trades (>55% or <45%)
high_conf = df[(df[proba_col] > 0.55) | (df[proba_col] < 0.45)]
if len(high_conf) > 0:
    high_acc = (high_conf['predicted_direction'] == high_conf['actual_direction']).sum() / len(high_conf)
    print(f"\n  High Confidence (>55% or <45%): {len(high_conf):,} trades")
    print(f"  Accuracy:                        {high_acc:.2%}")

# Quantile-based strategy (trade top/bottom quantiles)
print("\nQuantile-based trading tests:")
for q in [0.05, 0.10, 0.20, 0.30]:
    low_q = df[proba_col].quantile(q)
    high_q = df[proba_col].quantile(1 - q)
    sel = df[(df[proba_col] <= low_q) | (df[proba_col] >= high_q)].copy()
    if len(sel) == 0:
        print(f"  Quantile {int(q*100)}%: no trades")
        continue
    sel['quant_strategy_return'] = sel['predicted_direction'] * sel['actual_return']
    tot_ret = sel['quant_strategy_return'].sum()
    avg_ret = sel['quant_strategy_return'].mean()
    win_rate = (sel['quant_strategy_return'] > 0).sum() / len(sel)
    print(f"  Quantile {int(q*100)}%: Trades={len(sel):,}, TotalRet={tot_ret:+.2%}, AvgDaily={avg_ret:+.4%}, WinRate={win_rate:.2%}")

# ============================================================================
# Monthly Summary
# ============================================================================
print("\n" + "="*70)
print("📅 MONTHLY PERFORMANCE SUMMARY")
print("="*70)

df['trade_date_dt'] = pd.to_datetime(df['trade_date'], format='%Y%m%d')
df['year_month'] = df['trade_date_dt'].dt.strftime('%Y-%m')

monthly = df.groupby('year_month').agg({
    'actual_return': ['sum', 'mean', 'count'],
    'model_return': ['sum', 'mean'],
    'actual_direction': 'first'
}).round(4)

monthly.columns = ['Return_Sum', 'Avg_Daily_Return', 'Days', 'Model_Return_Sum', 'Model_Avg_Daily_Return', 'FirstDir']

print(f"\n{'Month':<10} {'Total_Ret':>10} {'Avg_Daily':>10} {'Model_Ret':>10} {'Days':>6}")
print("-"*50)

for idx, row in monthly.iterrows():
    print(f"{idx:<10} {row['Return_Sum']:>9.2%} {row['Avg_Daily_Return']:>9.4%} {row['Model_Return_Sum']:>9.2%} {int(row['Days']):>5}")

best_month = monthly['Return_Sum'].idxmax()
worst_month = monthly['Return_Sum'].idxmin()
print(f"\n  Best Month:  {best_month} ({monthly.loc[best_month, 'Return_Sum']:.2%})")
print(f"  Worst Month: {worst_month} ({monthly.loc[worst_month, 'Return_Sum']:.2%})")

# ============================================================================
# Final Verdict
# ============================================================================
print("\n" + "="*70)
print("📋 FINAL VERDICT")
print("="*70)

model_total_return = df['model_return'].sum()
buyhold_total_return = df['actual_return_long'].sum()
model_advantage = model_total_return - buyhold_total_return

print(f"""
The model trained with optimal threshold (0.30) and threshold-based predictions has:

✓ Direction Accuracy: {accuracy:.2%}
✓ Total Return (Daily Sum): {model_total_return:+.2%}
✓ Buy & Hold Return: {buyhold_total_return:+.2%}
✓ Outperformance: {model_advantage:+.2%}

Current Issue: The model predicts 100% UP because it was optimized to maximize
recall with a decision threshold of 0.30. This means it's always long, which
equals buy & hold performance.

Recommendation:
1. The {accuracy:.1%} accuracy is barely better than random (50%)
2. The model is not adding value - it's essentially buy & hold
3. Need to reconsider the optimization strategy:
   - Try different thresholds for different probability scores
   - Use ensemble methods or voting from multiple models
   - Focus on high-confidence predictions only (discard uncertain ones)
   - Consider market microstructure (entry/exit costs)
   
The test period shows high market volatility (returns up to ±{df['actual_return'].abs().max():.1%} per day)
which makes direction prediction difficult at {accuracy:.1%} accuracy.
""")

# Save summary
summary = {
    'test_period': f"{df['trade_date'].min()} to {df['trade_date'].max()}",
    'total_samples': int(len(df)),
    'direction_accuracy': float(accuracy),
    'predicted_all_up': int(pred_counts.get(1, 0)),
    'predicted_all_down': int(pred_counts.get(-1, 0)),
    'model_total_return': float(model_total_return),
    'buyhold_total_return': float(buyhold_total_return),
    'outperformance': float(model_advantage),
    'avg_daily_return': float(df['actual_return'].mean()),
    'return_volatility': float(df['actual_return'].std()),
}

with open('./artifacts/backtest_analysis_kc.json', 'w') as f:
    json.dump(summary, f, indent=2)

df.to_csv('./artifacts/backtest_full_kc.csv', index=False)

print("\n✅ Full backtest saved to artifacts/backtest_full_kc.csv")
print("✅ Analysis summary saved to artifacts/backtest_analysis_kc.json")
