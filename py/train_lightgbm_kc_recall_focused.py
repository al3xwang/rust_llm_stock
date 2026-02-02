#!/usr/bin/env python3
"""
LightGBM KC Training - Recall-Focused Approach

Focus on models that actually detect both classes rather than just optimizing AUC.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.utils.class_weight import compute_class_weight

import warnings
warnings.filterwarnings("ignore")

# Setup
workspace_root = Path("/Users/alex/rust_llm_stock")
os.chdir(workspace_root)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

print("="*70)
print("🚀 LIGHTGBM KC - RECALL-FOCUSED TRAINING")
print("="*70)

# ============================================================================
# SECTION 1: Load Data
# ============================================================================
print("\n📊 SECTION 1: Loading Data Files")
print("-"*70)

train_path = Path("./data/kc_train.csv")
val_path = Path("./data/kc_val.csv")
test_path = Path("./data/kc_test.csv")

df_train = pd.read_csv(train_path)
df_val = pd.read_csv(val_path)
df_test = pd.read_csv(test_path) if test_path.exists() else None

# Filter out class 0
target_col = 'next_day_direction'
df_train = df_train[df_train[target_col] != 0].copy()
df_val = df_val[df_val[target_col] != 0].copy()
if df_test is not None:
    df_test = df_test[df_test[target_col] != 0].copy()

# ============================================================================
# SECTION 2: Data Preprocessing
# ============================================================================
print("\n🔧 SECTION 2: Data Preprocessing")
print("-"*70)

id_cols = ['ts_code', 'trade_date']
exclude_cols = {'ts_code', 'trade_date', 'next_day_direction', 'next_3day_direction',
                'next_day_return', 'next_3day_return', 'id', 'created_at'}
exclude_prefixes = ('industry_emb_', 'act_ent_type_emb_')

numeric_cols = df_train.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df_train.select_dtypes(include=['object']).columns.tolist()

feature_cols = [c for c in numeric_cols if c not in exclude_cols and not any(c.startswith(p) for p in exclude_prefixes)]
categorical_feature_cols = [c for c in categorical_cols if c not in exclude_cols]
embedding_cols = [c for c in numeric_cols if any(c.startswith(p) for p in exclude_prefixes)]

all_feature_cols = feature_cols + categorical_feature_cols + embedding_cols

# Impute
imputer = SimpleImputer(strategy='median')
df_train[feature_cols] = imputer.fit_transform(df_train[feature_cols])
df_val[feature_cols] = imputer.transform(df_val[feature_cols])
if df_test is not None:
    df_test[feature_cols] = imputer.transform(df_test[feature_cols])

for emb_col in embedding_cols:
    df_train[emb_col].fillna(0.0, inplace=True)
    df_val[emb_col].fillna(0.0, inplace=True)
    if df_test is not None:
        df_test[emb_col].fillna(0.0, inplace=True)

# Encode categorical
label_encoders = {}
for cat_col in categorical_feature_cols:
    le = LabelEncoder()
    df_train[cat_col] = le.fit_transform(df_train[cat_col].astype(str))
    df_val[cat_col] = le.transform(df_val[cat_col].astype(str))
    if df_test is not None:
        df_test[cat_col] = df_test[cat_col].astype(str).apply(
            lambda x: le.transform([x])[0] if x in le.classes_ else 0
        )
    label_encoders[cat_col] = le

# Prepare data
X_train = df_train[all_feature_cols]
y_train = df_train[target_col]
X_val = df_val[all_feature_cols]
y_val = df_val[target_col]

label_map = {-1: 0, 1: 1}
reverse_label_map = {0: -1, 1: 1}
y_train_encoded = y_train.map(label_map).astype(int)
y_val_encoded = y_val.map(label_map).astype(int)

if df_test is not None:
    X_test = df_test[all_feature_cols]
    y_test = df_test[target_col]
    y_test_encoded = y_test.map(label_map).astype(int)
else:
    X_test = None
    y_test = None
    y_test_encoded = None

print(f"✅ Data ready: {X_train.shape[0]} training samples, {len(all_feature_cols)} features")

# ============================================================================
# SECTION 3: Recall-Focused Models
# ============================================================================
print("\n🧪 SECTION 3: Training Recall-Focused Models")
print("-"*70)

configs = [
    {
        'name': 'LowerThreshold',
        'desc': 'Lower Decision Threshold (0.4)',
        'threshold': 0.4,
        'params': {
            'objective': 'binary',
            'metric': ['auc', 'binary_logloss'],
            'num_leaves': 20,
            'learning_rate': 0.02,
            'feature_fraction': 0.7,
            'bagging_fraction': 0.7,
            'bagging_freq': 3,
            'verbose': -1,
            'seed': RANDOM_SEED,
            'boosting_type': 'gbdt',
            'max_depth': 6,
            'min_data_in_leaf': 70,
            'lambda_l1': 0.8,
            'lambda_l2': 0.8,
            'scale_pos_weight': 1.09,
        }
    },
    {
        'name': 'HighScalePosWeight',
        'desc': 'High Pos Weight (5.0) - Favor Positive Class',
        'threshold': 0.5,
        'params': {
            'objective': 'binary',
            'metric': ['auc', 'binary_logloss'],
            'num_leaves': 20,
            'learning_rate': 0.02,
            'feature_fraction': 0.7,
            'bagging_fraction': 0.7,
            'bagging_freq': 3,
            'verbose': -1,
            'seed': RANDOM_SEED,
            'boosting_type': 'gbdt',
            'max_depth': 6,
            'min_data_in_leaf': 70,
            'lambda_l1': 0.8,
            'lambda_l2': 0.8,
            'scale_pos_weight': 5.0,
        }
    },
    {
        'name': 'HighScalePosWeight_LowerThreshold',
        'desc': 'High Pos Weight (5.0) + Lower Threshold (0.35)',
        'threshold': 0.35,
        'params': {
            'objective': 'binary',
            'metric': ['auc', 'binary_logloss'],
            'num_leaves': 20,
            'learning_rate': 0.02,
            'feature_fraction': 0.7,
            'bagging_fraction': 0.7,
            'bagging_freq': 3,
            'verbose': -1,
            'seed': RANDOM_SEED,
            'boosting_type': 'gbdt',
            'max_depth': 6,
            'min_data_in_leaf': 70,
            'lambda_l1': 0.8,
            'lambda_l2': 0.8,
            'scale_pos_weight': 5.0,
        }
    },
    {
        'name': 'VeryHighScalePosWeight',
        'desc': 'Very High Pos Weight (10.0)',
        'threshold': 0.5,
        'params': {
            'objective': 'binary',
            'metric': ['auc', 'binary_logloss'],
            'num_leaves': 25,
            'learning_rate': 0.025,
            'feature_fraction': 0.75,
            'bagging_fraction': 0.75,
            'bagging_freq': 3,
            'verbose': -1,
            'seed': RANDOM_SEED,
            'boosting_type': 'gbdt',
            'max_depth': 7,
            'min_data_in_leaf': 50,
            'lambda_l1': 0.5,
            'lambda_l2': 0.5,
            'scale_pos_weight': 10.0,
        }
    }
]

results = []

for config in configs:
    print(f"\n{'='*70}")
    print(f"🔬 {config['name']}: {config['desc']}")
    print(f"{'='*70}")
    
    train_data = lgb.Dataset(X_train, label=y_train_encoded)
    val_data = lgb.Dataset(X_val, label=y_val_encoded, reference=train_data)
    
    model = lgb.train(
        config['params'],
        train_data,
        num_boost_round=500,
        valid_sets=[train_data, val_data],
        valid_names=['train', 'valid'],
        callbacks=[
            lgb.log_evaluation(period=100),
            lgb.early_stopping(stopping_rounds=40)
        ]
    )
    
    # Predictions on validation with threshold
    y_pred_val_proba = model.predict(X_val)
    y_pred_val_binary = (y_pred_val_proba >= config['threshold']).astype(int)
    y_pred_val_orig = np.array([reverse_label_map[p] for p in y_pred_val_binary])
    
    val_accuracy = accuracy_score(y_val, y_pred_val_orig)
    val_precision = precision_score(y_val, y_pred_val_orig, zero_division=0)
    val_recall = recall_score(y_val, y_pred_val_orig, zero_division=0)
    val_f1 = f1_score(y_val, y_pred_val_orig, zero_division=0)
    val_auc = roc_auc_score(y_val_encoded, y_pred_val_proba)
    
    val_pred_dist = np.bincount(y_pred_val_binary, minlength=2)
    
    print(f"\n📊 Validation Results (threshold={config['threshold']}):")
    print(f"   AUC-ROC:   {val_auc:.4f}")
    print(f"   Accuracy:  {val_accuracy:.4f}")
    print(f"   Precision: {val_precision:.4f}")
    print(f"   Recall:    {val_recall:.4f}")
    print(f"   F1-Score:  {val_f1:.4f}")
    print(f"   Pred distribution: {val_pred_dist[0]} negatives, {val_pred_dist[1]} positives")
    
    # Test results
    test_metrics = {}
    if y_test is not None:
        y_pred_test_proba = model.predict(X_test)
        y_pred_test_binary = (y_pred_test_proba >= config['threshold']).astype(int)
        y_pred_test_orig = np.array([reverse_label_map[p] for p in y_pred_test_binary])
        
        test_auc = roc_auc_score(y_test_encoded, y_pred_test_proba)
        test_accuracy = accuracy_score(y_test, y_pred_test_orig)
        test_precision = precision_score(y_test, y_pred_test_orig, zero_division=0)
        test_recall = recall_score(y_test, y_pred_test_orig, zero_division=0)
        test_f1 = f1_score(y_test, y_pred_test_orig, zero_division=0)
        
        test_pred_dist = np.bincount(y_pred_test_binary, minlength=2)
        
        test_metrics = {
            'test_auc': test_auc,
            'test_accuracy': test_accuracy,
            'test_precision': test_precision,
            'test_recall': test_recall,
            'test_f1': test_f1,
        }
        
        print(f"\n   Test AUC-ROC:   {test_auc:.4f}")
        print(f"   Test Accuracy:  {test_accuracy:.4f}")
        print(f"   Test Precision: {test_precision:.4f}")
        print(f"   Test Recall:    {test_recall:.4f}")
        print(f"   Test F1-Score:  {test_f1:.4f}")
        print(f"   Test Pred dist: {test_pred_dist[0]} negatives, {test_pred_dist[1]} positives")
    
    results.append({
        'config': config['name'],
        'threshold': config['threshold'],
        'model': model,
        'val_auc': val_auc,
        'val_accuracy': val_accuracy,
        'val_precision': val_precision,
        'val_recall': val_recall,
        'val_f1': val_f1,
        'X_test': X_test,
        'y_test': y_test,
        **test_metrics
    })

# ============================================================================
# SECTION 4: Results Summary
# ============================================================================
print(f"\n{'='*70}")
print("📊 RECALL-FOCUSED RESULTS COMPARISON")
print(f"{'='*70}")

results_df = pd.DataFrame([
    {
        'Config': r['config'],
        'Threshold': r['threshold'],
        'Val AUC': f"{r['val_auc']:.4f}",
        'Val Recall': f"{r['val_recall']:.4f}",
        'Val F1': f"{r['val_f1']:.4f}",
        'Test Recall': f"{r.get('test_recall', 0):.4f}" if 'test_recall' in r else 'N/A',
    }
    for r in results
])

print("\n" + results_df.to_string(index=False))

# Find best based on F1-score (balances precision and recall)
best_result = max(results, key=lambda x: x['val_f1'])
print(f"\n🏆 BEST MODEL (by F1-Score): {best_result['config']} (F1: {best_result['val_f1']:.4f})")

# ============================================================================
# SECTION 5: Best Model Analysis
# ============================================================================
print(f"\n{'='*70}")
print("✅ BEST MODEL ANALYSIS")
print(f"{'='*70}")

print(f"\n📋 Configuration: {best_result['config']}")
print(f"    Decision Threshold: {best_result['threshold']}")
print(f"\n   Validation Metrics:")
print(f"   AUC-ROC:   {best_result['val_auc']:.4f}")
print(f"   Accuracy:  {best_result['val_accuracy']:.4f}")
print(f"   Precision: {best_result['val_precision']:.4f}")
print(f"   Recall:    {best_result['val_recall']:.4f}")
print(f"   F1-Score:  {best_result['val_f1']:.4f}")

if 'test_f1' in best_result:
    print(f"\n   Test Metrics:")
    print(f"   AUC-ROC:   {best_result['test_auc']:.4f}")
    print(f"   Accuracy:  {best_result['test_accuracy']:.4f}")
    print(f"   Precision: {best_result['test_precision']:.4f}")
    print(f"   Recall:    {best_result['test_recall']:.4f}")
    print(f"   F1-Score:  {best_result['test_f1']:.4f}")
    
    # Test confusion matrix
    y_pred_test_proba = best_result['model'].predict(best_result['X_test'])
    y_pred_test_binary = (y_pred_test_proba >= best_result['threshold']).astype(int)
    y_pred_test_orig = np.array([reverse_label_map[p] for p in y_pred_test_binary])
    
    cm = confusion_matrix(best_result['y_test'], y_pred_test_orig, labels=[-1, 1])
    print(f"\n   Test Confusion Matrix:")
    print(f"                  Pred -1  Pred 1")
    print(f"   Actual -1      {cm[0,0]:6d}   {cm[0,1]:6d}")
    print(f"   Actual  1      {cm[1,0]:6d}   {cm[1,1]:6d}")

print("\n" + "="*70)
print("✅ RECALL-FOCUSED TRAINING COMPLETE")
print("="*70)
