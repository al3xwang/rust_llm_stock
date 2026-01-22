#!/usr/bin/env python3
"""
Simple model evaluation using the exported test.csv data.
Evaluates the trained safetensors model on recent trading data.
"""

import os
import pandas as pd
import numpy as np
import safetensors.torch
import torch
from pathlib import Path

def load_model(model_path='artifacts/best_model.safetensors'):
    """Load safetensors model"""
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        return None
    
    try:
        tensors = safetensors.torch.load_file(model_path)
        print(f"✓ Loaded model from {model_path}")
        print(f"  Model contains {len(tensors)} tensors")
        
        # List model layers
        print(f"  Layers: {list(tensors.keys())[:10]}")
        return tensors
    except Exception as e:
        print(f"Failed to load model: {e}")
        return None

def analyze_test_data():
    """Analyze the exported test data"""
    test_path = 'test.csv'
    if not os.path.exists(test_path):
        print(f"Test data not found: {test_path}")
        return
    
    print(f"\n✓ Loading test data from {test_path}...")
    df = pd.read_csv(test_path)
    
    print(f"\nTest Dataset Statistics:")
    print(f"  Total records: {len(df)}")
    print(f"  Columns: {len(df.columns)}")
    print(f"  Date range: {df['trade_date'].min()} to {df['trade_date'].max()}")
    print(f"  Unique dates: {df['trade_date'].nunique()}")
    print(f"  Unique stocks: {df['ts_code'].nunique()}")
    
    # Check for recent dates (Jan 2026)
    if 'trade_date' in df.columns:
        jan_data = df[df['trade_date'].astype(str).str.startswith('202601')]
        if not jan_data.empty:
            print(f"\n  Jan 2026 data found: {len(jan_data)} records")
            print(f"    Dates: {jan_data['trade_date'].unique()}")
            print(f"    Stocks: {jan_data['ts_code'].nunique()}")
        else:
            print(f"\n  No Jan 2026 data in test set")
            print(f"    Latest dates: {df['trade_date'].nlargest(5).unique().tolist()}")
    
    # Feature statistics
    feature_cols = [col for col in df.columns if col not in ['ts_code', 'trade_date', 'next_day_return', 'next_day_direction']]
    print(f"\n  Features available: {len(feature_cols)}")
    
    # Check for target variable
    if 'next_day_return' in df.columns:
        valid_targets = df['next_day_return'].notna().sum()
        print(f"  Target variable 'next_day_return': {valid_targets} valid values")
        print(f"    Mean: {df['next_day_return'].mean():.6f}")
        print(f"    Std: {df['next_day_return'].std():.6f}")
    
    # Check missing values
    missing_pct = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
    if missing_pct[missing_pct > 0].shape[0] > 0:
        print(f"\n  Columns with missing values (top 10):")
        for col, pct in missing_pct[missing_pct > 0].head(10).items():
            print(f"    {col}: {pct:.1f}%")

def main():
    """Main evaluation"""
    print("=" * 80)
    print("MODEL EVALUATION - Test Set Analysis")
    print("=" * 80)
    
    # Load model
    print("\nStep 1: Loading trained model...")
    model = load_model('artifacts/best_model.safetensors')
    if model is None:
        return
    
    # Analyze test data
    print("\nStep 2: Analyzing test dataset...")
    analyze_test_data()
    
    # Summary
    print("\n" + "=" * 80)
    print("NEXT STEPS:")
    print("=" * 80)
    print("""
To evaluate on Jan 12-14, 2026 trading data:
1. Run: cargo run --release --bin dataset_creator
   (Repeatedly if needed to fill through the holiday gap to Jan dates)
   
2. Once ml_training_dataset has Jan 12-14 data, run:
   python3 evaluate_predictions.py
   
Alternatively, if Jan 2026 data is in test.csv, detailed evaluation can be run
with a proper PyTorch model loading implementation.
    """)

if __name__ == '__main__':
    main()
