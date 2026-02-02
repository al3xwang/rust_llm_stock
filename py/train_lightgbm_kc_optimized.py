#!/usr/bin/env python3
"""
Optimized LightGBM Training Script for KC Stock Data - Multiple Configurations

This script trains multiple LightGBM models with different hyperparameters,
compares results, and selects the best model based on validation metrics.
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
sns.set(style='whitegrid', context='notebook', rc={'figure.figsize': (12, 6)})

print("="*70)
print("🚀 LIGHTGBM KC OPTIMIZED MULTI-CONFIG TRAINING")
print("="*70)

# ============================================================================
# SECTION 1: Load Data
# ============================================================================
print("\n📊 SECTION 1: Loading Data Files")
print("-"*70)

train_path = Path("./data/kc_train.csv")
val_path = Path("./data/kc_val.csv")
test_path = Path("./data/kc_test.csv")

if not train_path.exists() or not val_path.exists():
    print(f"❌ ERROR: Required data files not found!")
    sys.exit(1)

df_train = pd.read_csv(train_path)
df_val = pd.read_csv(val_path)
df_test = pd.read_csv(test_path) if test_path.exists() else None

print(f"✅ Training data: {df_train.shape}")
print(f"✅ Validation data: {df_val.shape}")
if df_test is not None:
    print(f"✅ Test data: {df_test.shape}")

# Filter out class 0 (neutral)
target_col = 'next_day_direction'
df_train = df_train[df_train[target_col] != 0].copy()
df_val = df_val[df_val[target_col] != 0].copy()
if df_test is not None:
    df_test = df_test[df_test[target_col] != 0].copy()

print(f"\n🎯 Target distribution (binary):")
print(f"   Train: {df_train[target_col].value_counts().to_dict()}")
print(f"   Val:   {df_val[target_col].value_counts().to_dict()}")

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

print(f"   Total features: {len(all_feature_cols)}")

# Impute missing values
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

# Encode categorical features
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

print("✅ Preprocessing complete")

# Prepare features and labels
X_train = df_train[all_feature_cols]
y_train = df_train[target_col]
X_val = df_val[all_feature_cols]
y_val = df_val[target_col]

# Binary encoding: -1 -> 0, 1 -> 1
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

# ============================================================================
# SECTION 3: Model Configurations to Test
# ============================================================================
print("\n🧪 SECTION 3: Testing Multiple Configurations")
print("-"*70)

# Calculate class weights
unique_classes = np.array([0, 1])
class_weights = compute_class_weight('balanced', classes=unique_classes, y=y_train_encoded)

configs = [
    {
        'name': 'Config1_HighReg',
        'desc': 'High Regularization (L1=1.5, L2=1.5)',
        'params': {
            'objective': 'binary',
            'metric': ['auc', 'binary_logloss'],
            'num_leaves': 20,
            'learning_rate': 0.015,
            'feature_fraction': 0.6,
            'bagging_fraction': 0.6,
            'bagging_freq': 3,
            'verbose': -1,
            'seed': RANDOM_SEED,
            'boosting_type': 'gbdt',
            'max_depth': 5,
            'min_data_in_leaf': 100,
            'lambda_l1': 1.5,
            'lambda_l2': 1.5,
            'min_gain_to_split': 0.1,
            'scale_pos_weight': class_weights[1] / class_weights[0],
        }
    },
    {
        'name': 'Config2_MediumReg',
        'desc': 'Medium Regularization (L1=0.8, L2=0.8)',
        'params': {
            'objective': 'binary',
            'metric': ['auc', 'binary_logloss'],
            'num_leaves': 25,
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
            'min_gain_to_split': 0.08,
            'scale_pos_weight': class_weights[1] / class_weights[0],
        }
    },
    {
        'name': 'Config3_ShallowTrees',
        'desc': 'Shallow Trees (depth=3, leaves=10)',
        'params': {
            'objective': 'binary',
            'metric': ['auc', 'binary_logloss'],
            'num_leaves': 10,
            'learning_rate': 0.01,
            'feature_fraction': 0.65,
            'bagging_fraction': 0.65,
            'bagging_freq': 3,
            'verbose': -1,
            'seed': RANDOM_SEED,
            'boosting_type': 'gbdt',
            'max_depth': 3,
            'min_data_in_leaf': 150,
            'lambda_l1': 1.0,
            'lambda_l2': 1.0,
            'min_gain_to_split': 0.15,
            'scale_pos_weight': class_weights[1] / class_weights[0],
        }
    },
    {
        'name': 'Config4_HighLR',
        'desc': 'Higher Learning Rate (lr=0.05)',
        'params': {
            'objective': 'binary',
            'metric': ['auc', 'binary_logloss'],
            'num_leaves': 15,
            'learning_rate': 0.05,
            'feature_fraction': 0.7,
            'bagging_fraction': 0.7,
            'bagging_freq': 3,
            'verbose': -1,
            'seed': RANDOM_SEED,
            'boosting_type': 'gbdt',
            'max_depth': 5,
            'min_data_in_leaf': 80,
            'lambda_l1': 1.2,
            'lambda_l2': 1.2,
            'min_gain_to_split': 0.1,
            'scale_pos_weight': class_weights[1] / class_weights[0],
        }
    },
    {
        'name': 'Config5_Balanced',
        'desc': 'Balanced approach',
        'params': {
            'objective': 'binary',
            'metric': ['auc', 'binary_logloss'],
            'num_leaves': 22,
            'learning_rate': 0.025,
            'feature_fraction': 0.75,
            'bagging_fraction': 0.75,
            'bagging_freq': 3,
            'verbose': -1,
            'seed': RANDOM_SEED,
            'boosting_type': 'gbdt',
            'max_depth': 6,
            'min_data_in_leaf': 60,
            'lambda_l1': 0.6,
            'lambda_l2': 0.6,
            'min_gain_to_split': 0.07,
            'scale_pos_weight': class_weights[1] / class_weights[0],
        }
    }
]

results = []

for config in configs:
    print(f"\n{'='*70}")
    print(f"🔬 Testing: {config['name']} - {config['desc']}")
    print(f"{'='*70}")
    
    # Train model
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
    
    # Evaluate on validation
    y_pred_val_proba = model.predict(X_val)
    y_pred_val_binary = (y_pred_val_proba >= 0.5).astype(int)
    y_pred_val_orig = np.array([reverse_label_map[p] for p in y_pred_val_binary])
    
    val_accuracy = accuracy_score(y_val, y_pred_val_orig)
    val_precision = precision_score(y_val, y_pred_val_orig, zero_division=0)
    val_recall = recall_score(y_val, y_pred_val_orig, zero_division=0)
    val_f1 = f1_score(y_val, y_pred_val_orig, zero_division=0)
    val_auc = roc_auc_score(y_val_encoded, y_pred_val_proba)
    
    print(f"\n📊 Validation Results:")
    print(f"   AUC-ROC:  {val_auc:.4f}")
    print(f"   Accuracy: {val_accuracy:.4f}")
    print(f"   Precision: {val_precision:.4f}")
    print(f"   Recall:   {val_recall:.4f}")
    print(f"   F1-Score: {val_f1:.4f}")
    
    # Evaluate on test if available
    test_metrics = {}
    if y_test is not None:
        y_pred_test_proba = model.predict(X_test)
        y_pred_test_binary = (y_pred_test_proba >= 0.5).astype(int)
        y_pred_test_orig = np.array([reverse_label_map[p] for p in y_pred_test_binary])
        
        test_auc = roc_auc_score(y_test_encoded, y_pred_test_proba)
        test_accuracy = accuracy_score(y_test, y_pred_test_orig)
        test_f1 = f1_score(y_test, y_pred_test_orig, zero_division=0)
        
        test_metrics = {
            'test_auc': test_auc,
            'test_accuracy': test_accuracy,
            'test_f1': test_f1
        }
        
        print(f"\n   Test AUC-ROC:  {test_auc:.4f}")
        print(f"   Test Accuracy: {test_accuracy:.4f}")
        print(f"   Test F1:       {test_f1:.4f}")
    
    results.append({
        'config': config['name'],
        'description': config['desc'],
        'model': model,
        'params': config['params'],
        'val_auc': val_auc,
        'val_accuracy': val_accuracy,
        'val_precision': val_precision,
        'val_recall': val_recall,
        'val_f1': val_f1,
        'X_test': X_test,
        'y_test': y_test,
        'y_test_encoded': y_test_encoded,
        **test_metrics
    })

# ============================================================================
# SECTION 4: Results Comparison
# ============================================================================
print(f"\n{'='*70}")
print("📊 RESULTS COMPARISON")
print(f"{'='*70}")

results_df = pd.DataFrame([
    {
        'Config': r['config'],
        'Val AUC': f"{r['val_auc']:.4f}",
        'Val Acc': f"{r['val_accuracy']:.4f}",
        'Val Recall': f"{r['val_recall']:.4f}",
        'Test AUC': f"{r.get('test_auc', 'N/A'):.4f}" if 'test_auc' in r else 'N/A',
    }
    for r in results
])

print("\n" + results_df.to_string(index=False))

# Find best model
best_result = max(results, key=lambda x: x['val_auc'])
print(f"\n🏆 BEST MODEL: {best_result['config']} (Val AUC: {best_result['val_auc']:.4f})")

# ============================================================================
# SECTION 5: Detailed Analysis of Best Model
# ============================================================================
print(f"\n{'='*70}")
print("✅ BEST MODEL DETAILED ANALYSIS")
print(f"{'='*70}")

best_model = best_result['model']

print(f"\n📋 Configuration: {best_result['description']}")
print(f"\n   Learning Rate: {best_result['params']['learning_rate']}")
print(f"   Max Depth: {best_result['params']['max_depth']}")
print(f"   Num Leaves: {best_result['params']['num_leaves']}")
print(f"   Lambda L1: {best_result['params']['lambda_l1']}")
print(f"   Lambda L2: {best_result['params']['lambda_l2']}")
print(f"   Scale Pos Weight: {best_result['params']['scale_pos_weight']:.4f}")

print(f"\n📈 Validation Metrics:")
print(f"   AUC-ROC:  {best_result['val_auc']:.4f}")
print(f"   Accuracy: {best_result['val_accuracy']:.4f}")
print(f"   Precision: {best_result['val_precision']:.4f}")
print(f"   Recall:   {best_result['val_recall']:.4f}")
print(f"   F1-Score: {best_result['val_f1']:.4f}")

if 'test_auc' in best_result:
    print(f"\n   Test AUC-ROC:  {best_result['test_auc']:.4f}")
    print(f"   Test Accuracy: {best_result['test_accuracy']:.4f}")
    print(f"   Test F1:       {best_result['test_f1']:.4f}")
    
    # Confusion matrix on test
    y_pred_test_proba = best_model.predict(best_result['X_test'])
    y_pred_test_binary = (y_pred_test_proba >= 0.5).astype(int)
    y_pred_test_orig = np.array([reverse_label_map[p] for p in y_pred_test_binary])
    
    cm = confusion_matrix(best_result['y_test'], y_pred_test_orig, labels=[-1, 1])
    print(f"\n📋 Test Confusion Matrix:")
    print(f"                  Pred -1  Pred 1")
    print(f"   Actual -1       {cm[0,0]:6d}   {cm[0,1]:6d}")
    print(f"   Actual  1       {cm[1,0]:6d}   {cm[1,1]:6d}")

# Save best model predictions
output_path = Path("./artifacts/test_predictions_kc_best.csv")
output_path.parent.mkdir(parents=True, exist_ok=True)

if best_result['X_test'] is not None:
    y_pred_proba = best_model.predict(best_result['X_test'])
    y_pred_binary = (y_pred_proba >= 0.5).astype(int)
    y_pred_orig = np.array([reverse_label_map[p] for p in y_pred_binary])
    
    pred_df = best_result['y_test'].to_frame(name='actual')
    pred_df['predicted'] = y_pred_orig
    pred_df['predicted_proba'] = y_pred_proba
    pred_df.to_csv(output_path, index=False)
    print(f"\n✅ Best model predictions saved to {output_path}")

# Feature importance
print(f"\n📊 Top 20 Important Features:")
importance = pd.DataFrame({
    'feature': all_feature_cols,
    'importance': best_model.feature_importance(importance_type='gain')
}).sort_values('importance', ascending=False)

print(importance.head(20).to_string(index=False))

# Save results summary
summary = {
    'best_config': best_result['config'],
    'best_description': best_result['description'],
    'val_auc': float(best_result['val_auc']),
    'val_accuracy': float(best_result['val_accuracy']),
    'val_f1': float(best_result['val_f1']),
}
if 'test_auc' in best_result:
    summary['test_auc'] = float(best_result['test_auc'])
    summary['test_accuracy'] = float(best_result['test_accuracy'])

summary_path = Path("./artifacts/model_comparison_results.json")
with open(summary_path, 'w') as f:
    json.dump(summary, f, indent=2)

print(f"\n✅ Results summary saved to {summary_path}")

print("\n" + "="*70)
print("✅ MODEL OPTIMIZATION COMPLETE")
print("="*70)
