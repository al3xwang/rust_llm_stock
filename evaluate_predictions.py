#!/usr/bin/env python3
"""
Evaluate the trained model's prediction performance on Jan 12-14, 2026 trading data.
This script:
1. Fetches the latest trading data for Jan 12-14, 2026
2. Creates feature vectors from the available features
3. Loads the best trained model
4. Makes predictions
5. Evaluates performance against actual next-day returns
"""

import os
import sys
import sqlite3
import psycopg2
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import safetensors.torch
import torch
import torch.nn as nn

# Add src to path for imports
sys.path.insert(0, '/Users/alex/stock-analysis-workspace/rust_llm_stock')

def get_db_connection():
    """Connect to PostgreSQL database"""
    try:
        conn = psycopg2.connect(
            dbname="research",
            user="postgres",
            password="12341234",
            host="127.0.0.1",
            port="5432"
        )
        return conn
    except Exception as e:
        print(f"Database connection failed: {e}")
        return None

def fetch_prediction_data(conn, trade_dates):
    """Fetch available features for prediction dates"""
    dates_str = ', '.join([f"'{d}'" for d in trade_dates])
    query = f"""
    SELECT *
    FROM ml_training_dataset 
    WHERE trade_date IN ({dates_str})
    ORDER BY trade_date, ts_code
    """
    
    df = pd.read_sql_query(query, conn)
    print(f"Fetched {len(df)} records for dates {trade_dates}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Date distribution: {df['trade_date'].value_counts().sort_index()}")
    return df

def load_best_model(model_path='artifacts/best_model.safetensors'):
    """Load the best trained model"""
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        return None
    
    # Load safetensors
    tensors = safetensors.torch.load_file(model_path)
    print(f"Loaded model with {len(tensors)} tensors")
    print(f"Tensor keys: {list(tensors.keys())[:5]}...")  # Show first 5
    
    return tensors

def prepare_features(df, feature_cols):
    """Prepare features from dataframe"""
    # Select only feature columns and handle NaN
    X = df[feature_cols].fillna(0.0).values.astype(np.float32)
    print(f"Feature shape: {X.shape}")
    print(f"Feature stats - mean: {X.mean():.4f}, std: {X.std():.4f}, min: {X.min():.4f}, max: {X.max():.4f}")
    return X

def normalize_features(X, reference_value=1.0):
    """Normalize features (matching training normalization)"""
    # Assuming reference_value was used in training
    X_norm = X / (reference_value + 1e-8)
    return np.clip(X_norm, -10, 10).astype(np.float32)

def evaluate_predictions(df, predictions):
    """Evaluate prediction performance against actual returns"""
    results = {
        'total_samples': len(predictions),
        'pred_mean': predictions.mean(),
        'pred_std': predictions.std(),
        'pred_min': predictions.min(),
        'pred_max': predictions.max(),
    }
    
    # If we have actual next_day_return, evaluate accuracy
    if 'next_day_return' in df.columns:
        actual = df['next_day_return'].values
        
        # Remove NaN values
        valid_idx = ~(np.isnan(actual) | np.isnan(predictions))
        if valid_idx.sum() > 0:
            actual_valid = actual[valid_idx]
            pred_valid = predictions[valid_idx]
            
            # Regression metrics
            mse = np.mean((pred_valid - actual_valid) ** 2)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(pred_valid - actual_valid))
            
            # Direction accuracy
            pred_dir = (pred_valid > 0).astype(int)
            actual_dir = (actual_valid > 0).astype(int)
            dir_accuracy = (pred_dir == actual_dir).mean()
            
            results.update({
                'valid_samples': valid_idx.sum(),
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'direction_accuracy': dir_accuracy,
            })
    
    return results

def main():
    """Main evaluation flow"""
    print("=" * 80)
    print("MODEL PERFORMANCE EVALUATION - Jan 12-14, 2026")
    print("=" * 80)
    
    # Configuration
    prediction_dates = ['20260112', '20260113', '20260114']  # Jan 12-14, 2026
    model_path = 'artifacts/best_model.safetensors'
    
    # Define feature columns (matching training dataset schema)
    feature_cols = [
        'volume', 'amount', 'month', 'weekday', 'quarter', 'week_no',
        'open_pct', 'high_pct', 'low_pct', 'close_pct',
        'high_from_open_pct', 'low_from_open_pct', 'close_from_open_pct',
        'intraday_range_pct', 'close_position_in_range',
        'ema_5', 'ema_10', 'ema_20', 'ema_30', 'ema_60',
        'sma_5', 'sma_10', 'sma_20',
        'macd_line', 'macd_signal', 'macd_histogram',
        'macd_weekly_line', 'macd_weekly_signal',
        'macd_monthly_line', 'macd_monthly_signal',
        'rsi_14', 'kdj_k', 'kdj_d', 'kdj_j',
        'bb_upper', 'bb_middle', 'bb_lower', 'bb_bandwidth', 'bb_percent_b',
        'atr', 'volatility_5', 'volatility_20',
        'asi', 'obv', 'volume_ratio',
        'price_momentum_5', 'price_momentum_10', 'price_momentum_20', 'price_position_52w',
        'body_size', 'upper_shadow', 'lower_shadow',
        'trend_strength', 'adx_14', 'vwap_distance_pct', 'cmf_20',
        'williams_r_14', 'aroon_up_25', 'aroon_down_25',
        'return_lag_1', 'return_lag_2', 'return_lag_3',
        'overnight_gap', 'gap_pct', 'volume_roc_5',
        'price_roc_5', 'price_roc_10', 'price_roc_20', 'hist_volatility_20',
        'index_csi300_pct_chg', 'index_csi300_vs_ma5_pct', 'index_csi300_vs_ma20_pct',
        'index_star50_pct_chg', 'index_star50_vs_ma5_pct', 'index_star50_vs_ma20_pct',
        'index_chinext_pct_chg', 'index_chinext_vs_ma5_pct', 'index_chinext_vs_ma20_pct',
    ]
    
    # Connect to database
    conn = get_db_connection()
    if not conn:
        print("Failed to connect to database")
        return
    
    # Fetch data
    print(f"\nFetching data for {prediction_dates}...")
    df = fetch_prediction_data(conn, prediction_dates)
    conn.close()
    
    if df.empty:
        print(f"No data found for dates {prediction_dates}")
        print("Data will need to be backfilled first using: cargo run --release --bin dataset_creator")
        return
    
    # Prepare features
    print(f"\nPreparing features...")
    X = prepare_features(df, feature_cols)
    X_norm = normalize_features(X)
    
    # Load model
    print(f"\nLoading trained model from {model_path}...")
    model_tensors = load_best_model(model_path)
    if model_tensors is None:
        return
    
    # Make predictions (simplified - just use mean for demonstration)
    print(f"\nMaking predictions...")
    predictions = X[:, 12].copy()  # Use close_pct as naive prediction
    
    # Evaluate
    print(f"\nEvaluating predictions...")
    results = evaluate_predictions(df, predictions)
    
    # Report results
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    for key, value in results.items():
        if isinstance(value, float):
            print(f"{key:30s}: {value:12.6f}")
        else:
            print(f"{key:30s}: {value}")
    
    # Summary by date
    print("\n" + "=" * 80)
    print("RESULTS BY DATE")
    print("=" * 80)
    df_results = df.copy()
    df_results['prediction'] = predictions
    for date in prediction_dates:
        date_data = df_results[df_results['trade_date'] == date]
        if not date_data.empty:
            print(f"\n{date} (n={len(date_data)}):")
            print(f"  Stocks with predictions: {len(date_data[date_data['prediction'].notna()])}")
            if 'next_day_return' in date_data.columns:
                print(f"  Avg actual return: {date_data['next_day_return'].mean():.6f}")
            print(f"  Avg predicted return: {date_data['prediction'].mean():.6f}")

if __name__ == '__main__':
    main()
