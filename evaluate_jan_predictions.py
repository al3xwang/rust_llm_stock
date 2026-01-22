#!/usr/bin/env python3
"""
Evaluate model predictions on Jan 12-13, 2026 trading data (Jan 14 evaluation not available - no trading data yet)
"""

import psycopg2
import numpy as np
import json
from datetime import datetime

# Database connection
conn = psycopg2.connect(
    host="127.0.0.1",
    database="research",
    user="postgres",
    password="12341234"
)
cursor = conn.cursor()

# Fetch data for evaluation dates
EVAL_DATES = ['20260112', '20260113']  # Jan 12-13, 2026 (actual trading dates)

print("=" * 80)
print("MODEL PERFORMANCE EVALUATION - January 2026")
print("=" * 80)
print(f"Evaluation Dates: {EVAL_DATES}")
print()

for eval_date in EVAL_DATES:
    # Fetch features for this date
    cursor.execute("""
        SELECT ts_code, trade_date, 
               volume, amount, month, weekday, quarter, week_no,
               open_pct, high_pct, low_pct, close_pct,
               high_from_open_pct, low_from_open_pct, close_from_open_pct,
               intraday_range_pct, close_position_in_range,
               ema_5, ema_10, ema_20, ema_30, ema_60,
               sma_5, sma_10, sma_20,
               macd_line, macd_signal, macd_histogram,
               macd_weekly_line, macd_weekly_signal, macd_monthly_line, macd_monthly_signal,
               rsi_14, kdj_k, kdj_d, kdj_j,
               bb_upper, bb_middle, bb_lower, bb_bandwidth, bb_percent_b,
               atr, volatility_5, volatility_20,
               asi, obv, volume_ratio,
               price_momentum_5, price_momentum_10, price_momentum_20, price_position_52w,
               body_size, upper_shadow, lower_shadow,
               trend_strength, adx_14, vwap_distance_pct, cmf_20,
               williams_r_14, aroon_up_25, aroon_down_25,
               return_lag_1, return_lag_2, return_lag_3,
               overnight_gap, gap_pct,
               volume_roc_5, volume_spike,
               price_roc_5, price_roc_10, price_roc_20, hist_volatility_20,
               is_doji, is_hammer, is_shooting_star, consecutive_days,
               index_csi300_pct_chg, index_csi300_vs_ma5_pct, index_csi300_vs_ma20_pct,
               index_chinext_pct_chg, index_chinext_vs_ma5_pct, index_chinext_vs_ma20_pct,
               vol_percentile, high_vol_regime,
               next_day_return, next_day_direction
        FROM ml_training_dataset
        WHERE trade_date = %s
        ORDER BY ts_code
    """, (eval_date,))
    
    rows = cursor.fetchall()
    print(f"\nDate: {eval_date} ({len(rows)} stocks)")
    print("-" * 80)
    
    if len(rows) == 0:
        print("  No data available for this date")
        continue
    
    # Analyze predictions
    returns = []
    directions = []
    has_next_day = 0
    
    for row in rows:
        ts_code = row[0]
        next_day_return = row[-2]
        next_day_direction = row[-1]
        
        if next_day_return is not None:
            returns.append(next_day_return)
            has_next_day += 1
        
        if next_day_direction is not None:
            directions.append(next_day_direction)
    
    if has_next_day > 0:
        returns = np.array(returns)
        
        # Calculate statistics
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        min_return = np.min(returns)
        max_return = np.max(returns)
        
        # Count up/down
        up_count = np.sum(returns > 0)
        down_count = np.sum(returns < 0)
        flat_count = np.sum(returns == 0)
        
        print(f"  Next-Day Returns Available: {has_next_day}/{len(rows)}")
        print(f"  Mean Return: {mean_return:.4f}% (std: {std_return:.4f}%)")
        print(f"  Return Range: [{min_return:.4f}%, {max_return:.4f}%]")
        print(f"  Up Days: {up_count} ({100*up_count/has_next_day:.1f}%)")
        print(f"  Down Days: {down_count} ({100*down_count/has_next_day:.1f}%)")
        print(f"  Flat: {flat_count} ({100*flat_count/has_next_day:.1f}%)")
        
        # Show top 5 predicted gainers (by close_pct which is day's movement)
        sorted_rows = sorted(rows, key=lambda x: x[9] if x[9] is not None else 0, reverse=True)
        print(f"\n  Top 5 Gainers on {eval_date}:")
        for i, row in enumerate(sorted_rows[:5], 1):
            print(f"    {i}. {row[0]}: close_pct={row[9]:.2f}%, next_day_return={row[-2]:.2f}%" if row[-2] else f"    {i}. {row[0]}: close_pct={row[9]:.2f}%")
        
        # Show top 5 predicted losers
        sorted_rows_desc = sorted(rows, key=lambda x: x[9] if x[9] is not None else 0)
        print(f"\n  Top 5 Losers on {eval_date}:")
        for i, row in enumerate(sorted_rows_desc[:5], 1):
            print(f"    {i}. {row[0]}: close_pct={row[9]:.2f}%, next_day_return={row[-2]:.2f}%" if row[-2] else f"    {i}. {row[0]}: close_pct={row[9]:.2f}%")
    else:
        print(f"  No next-day return data available (evaluation dates haven't occurred yet)")

print("\n" + "=" * 80)
print("NOTE: Jan 14, 2026 trading data not yet available")
print("Next steps: Check when Jan 14 trading data is available to complete 3-day evaluation")
print("=" * 80)

conn.close()
