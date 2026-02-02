#!/usr/bin/env python3
"""
LightGBM Training Script for KC Stock Data

This script trains a LightGBM model on KC (Shanghai Stock Exchange) stock market data,
evaluates on validation set, and tests on kc_test.csv.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# LightGBM and ML libraries
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve
)

import warnings
warnings.filterwarnings("ignore")

# Setup
workspace_root = Path("/Users/alex/rust_llm_stock")
os.chdir(workspace_root)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

sns.set(style='whitegrid', context='notebook', rc={'figure.figsize': (12, 6)})

print("="*70)
print("🚀 LIGHTGBM KC STOCK MARKET MODEL TRAINING")
print("="*70)

# ============================================================================
# SECTION 1: Load Training, Validation, and Test Data
# ============================================================================
print("\n📊 SECTION 1: Loading Data Files")
print("-"*70)

train_path = Path("./data/kc_train.csv")
val_path = Path("./data/kc_val.csv")
test_path = Path("./data/kc_test.csv")

if not train_path.exists() or not val_path.exists():
    print(f"❌ ERROR: Required data files not found!")
    print(f"   Training: {train_path} {'✅' if train_path.exists() else '❌'}")
    print(f"   Validation: {val_path} {'✅' if val_path.exists() else '❌'}")
    print(f"   Test: {test_path} {'✅' if test_path.exists() else '❌'}")
    sys.exit(1)

df_train = pd.read_csv(train_path)
df_val = pd.read_csv(val_path)

# Load test data if it exists
if test_path.exists():
    df_test = pd.read_csv(test_path)
    print(f"✅ Training data: {df_train.shape}")
    print(f"✅ Validation data: {df_val.shape}")
    print(f"✅ Test data: {df_test.shape}")
else:
    print(f"✅ Training data: {df_train.shape}")
    print(f"✅ Validation data: {df_val.shape}")
    print(f"ℹ️  Test data file not found at {test_path}")
    df_test = None

# Target variable exploration
target_col = 'next_day_direction'
print(f"\n🎯 Target variable distribution (before filtering):")
print(f"   Training: {df_train[target_col].value_counts().to_dict()}")
print(f"   Validation: {df_val[target_col].value_counts().to_dict()}")

# FILTER OUT CLASS 0 (neutral) - convert to binary classification
print(f"\n   Filtering out Class 0 (neutral) for binary classification...")
df_train = df_train[df_train[target_col] != 0].copy()
df_val = df_val[df_val[target_col] != 0].copy()

print(f"\n🎯 Target variable distribution (after filtering):")
print(f"   Training: {df_train[target_col].value_counts().to_dict()}")
print(f"   Validation: {df_val[target_col].value_counts().to_dict()}")

# ============================================================================
# SECTION 2: Data Preprocessing
# ============================================================================
print("\n🔧 SECTION 2: Data Preprocessing")
print("-"*70)

# Identify columns
id_cols = ['ts_code', 'trade_date']
exclude_cols = {'ts_code', 'trade_date', 'next_day_direction', 'next_3day_direction',
                'next_day_return', 'next_3day_return', 'id', 'created_at'}
exclude_prefixes = ('industry_emb_', 'act_ent_type_emb_')

# Get features
numeric_cols = df_train.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df_train.select_dtypes(include=['object']).columns.tolist()

feature_cols = [c for c in numeric_cols if c not in exclude_cols and not any(c.startswith(p) for p in exclude_prefixes)]
categorical_feature_cols = [c for c in categorical_cols if c not in exclude_cols]
embedding_cols = [c for c in numeric_cols if any(c.startswith(p) for p in exclude_prefixes)]

all_feature_cols = feature_cols + categorical_feature_cols + embedding_cols

print(f"   Numeric features: {len(feature_cols)}")
print(f"   Categorical features: {len(categorical_feature_cols)}")
print(f"   Embedding features: {len(embedding_cols)}")
print(f"   Total features: {len(all_feature_cols)}")

# Handle missing values
imputer = SimpleImputer(strategy='median')
df_train[feature_cols] = imputer.fit_transform(df_train[feature_cols])
df_val[feature_cols] = imputer.transform(df_val[feature_cols])

# Fill embeddings with 0
for emb_col in embedding_cols:
    df_train[emb_col].fillna(0.0, inplace=True)
    df_val[emb_col].fillna(0.0, inplace=True)

# Encode categorical features
label_encoders = {}
for cat_col in categorical_feature_cols:
    le = LabelEncoder()
    df_train[cat_col] = le.fit_transform(df_train[cat_col].astype(str))
    df_val[cat_col] = le.transform(df_val[cat_col].astype(str))
    label_encoders[cat_col] = le

print("✅ Preprocessing complete")

# Prepare data for training FIRST
X_train = df_train[all_feature_cols]
y_train = df_train[target_col]
X_val = df_val[all_feature_cols]
y_val = df_val[target_col]

# Binary classification: convert {-1, 1} -> {0, 1}
# -1 (down) -> 0, 1 (up) -> 1
label_map = {-1: 0, 1: 1}
reverse_label_map = {0: -1, 1: 1}
y_train_encoded = y_train.map(label_map).astype(int)
y_val_encoded = y_val.map(label_map).astype(int)

# Prepare for training
y_train_lgb = y_train_encoded
y_val_lgb = y_val_encoded

# ============================================================================
# SECTION 3: Train LightGBM Model
# ============================================================================
print("\n🚀 SECTION 3: Training LightGBM Model")
print("-"*70)

train_data = lgb.Dataset(X_train, label=y_train_lgb)
val_data = lgb.Dataset(X_val, label=y_val_lgb, reference=train_data)

# Check class balance
class_balance = y_train_lgb.value_counts().min() / y_train_lgb.value_counts().max()
is_imbalanced = class_balance < 0.3

# Calculate class weights for imbalanced multiclass
from sklearn.utils.class_weight import compute_class_weight

unique_classes = np.array(sorted(np.unique(y_train_lgb)))
class_weights = compute_class_weight('balanced', classes=unique_classes, y=y_train_lgb)
class_weight_dict = {reverse_label_map[cls]: w for cls, w in zip(unique_classes, class_weights)}

print(f"   Class weights: {class_weight_dict}")

params = {
    'objective': 'binary',
    'metric': ['auc', 'binary_logloss'],
    'num_leaves': 25,
    'learning_rate': 0.02,
    'feature_fraction': 0.75,
    'bagging_fraction': 0.75,
    'bagging_freq': 3,
    'verbose': -1,
    'seed': RANDOM_SEED,
    'boosting_type': 'gbdt',
    'max_depth': 7,
    'min_data_in_leaf': 50,
    'is_unbalance': is_imbalanced,
    'lambda_l1': 0.8,
    'lambda_l2': 0.8,
    'min_gain_to_split': 0.05,
}

print(f"   Parameters: boosting_type={params['boosting_type']}, learning_rate={params['learning_rate']}, objective={params['objective']}")
print(f"   Class balance: {class_balance:.3f} {'(imbalanced)' if is_imbalanced else '(balanced)'}")


# Train with early stopping - increased patience for better convergence
print("   Training...")
model = lgb.train(
    params,
    train_data,
    num_boost_round=500,  # Increased max rounds
    valid_sets=[train_data, val_data],
    valid_names=['train', 'valid'],
    callbacks=[
        lgb.log_evaluation(period=50),
        lgb.early_stopping(stopping_rounds=30)  # Increased patience
    ]
)

print(f"✅ Training complete!")
print(f"   Best iteration: {model.best_iteration}")
print(f"   Training AUC: {model.best_score['train']['auc']:.4f}")
print(f"   Validation AUC: {model.best_score['valid']['auc']:.4f}")

# Validation metrics - for binary
y_pred_val_proba = model.predict(X_val)
y_pred_val = (y_pred_val_proba >= 0.5).astype(int)
# Map back to original class labels (-1, 1)
y_pred_val = np.array([reverse_label_map[p] for p in y_pred_val])

val_accuracy = accuracy_score(y_val, y_pred_val)
val_precision = precision_score(y_val, y_pred_val, zero_division=0)
val_recall = recall_score(y_val, y_pred_val, zero_division=0)
val_f1 = f1_score(y_val, y_pred_val, zero_division=0)
val_auc = roc_auc_score(y_val_lgb, y_pred_val_proba)

print(f"\n📊 Validation Metrics:")
print(f"   Accuracy:  {val_accuracy:.4f}")
print(f"   Precision: {val_precision:.4f}")
print(f"   Recall:    {val_recall:.4f}")
print(f"   F1-Score:  {val_f1:.4f}")
if val_auc > 0:
    print(f"   AUC-ROC:   {val_auc:.4f}")

# ============================================================================
# SECTION 4: Load Test Data
# ============================================================================
print("\n📁 SECTION 4: Loading Test Data")
print("-"*70)

if df_test is not None and df_test.shape[0] > 0:
    print(f"✅ Test data loaded: {df_test.shape}")
    has_test_target = target_col in df_test.columns
    if has_test_target:
        print(f"   Target distribution: {df_test[target_col].value_counts().to_dict()}")
else:
    print(f"ℹ️  Test data not available or empty")
    print(f"   Using validation set (kc_val.csv) as test data instead")
    df_test = df_val.copy()
    has_test_target = True
    test_source = "validation set"

# Preprocess test data - handle 0 rows
if df_test.shape[0] > 0:
    df_test[feature_cols] = imputer.transform(df_test[feature_cols])
for emb_col in embedding_cols:
    df_test[emb_col].fillna(0.0, inplace=True)
for cat_col in categorical_feature_cols:
    # Handle unseen categories by filling with the most common training value
    known_classes = set(label_encoders[cat_col].classes_)
    df_test[cat_col] = df_test[cat_col].astype(str).apply(
        lambda x: label_encoders[cat_col].transform([x])[0] 
        if x in known_classes 
        else 0  # Use 0 for unknown categories
    )

X_test = df_test[all_feature_cols]
if has_test_target and target_col in df_test.columns:
    y_test = df_test[target_col]
else:
    y_test = None

print("✅ Test data preprocessed")

# ============================================================================
# SECTION 5: Predictions on Test Data
# ============================================================================
print("\n🔮 SECTION 5: Generating Test Predictions")
print("-"*70)

y_pred_test_proba = model.predict(X_test)
y_pred_test = (y_pred_test_proba >= 0.5).astype(int)
# Map back to original class labels (-1, 1)
y_pred_test = np.array([reverse_label_map[p] for p in y_pred_test])

print(f"   Total predictions: {len(y_pred_test)}")
print(f"   Class -1 (Down): {(y_pred_test == -1).sum()} ({(y_pred_test == -1).sum() / len(y_pred_test) * 100:.1f}%)")
print(f"   Class  1 (Up):   {(y_pred_test == 1).sum()} ({(y_pred_test == 1).sum() / len(y_pred_test) * 100:.1f}%)")
print(f"   Probability range: [{y_pred_test_proba.min():.4f}, {y_pred_test_proba.max():.4f}]")

# Save predictions
output_path = Path("./artifacts/test_predictions_kc_lgb.csv")
output_path.parent.mkdir(parents=True, exist_ok=True)
pred_df = df_test[['ts_code', 'trade_date']].copy()
pred_df['predicted_proba'] = y_pred_test_proba
pred_df['predicted_class'] = y_pred_test
pred_df.to_csv(output_path, index=False)
print(f"✅ Predictions saved to {output_path}")

# ============================================================================
# SECTION 6: Evaluation Metrics
# ============================================================================
print("\n📈 SECTION 6: Model Evaluation")
print("-"*70)

if y_test is not None and len(y_test) > 0:
    # Encode test targets for AUC calculation
    y_test_encoded = y_test.map(label_map).astype(int)
    
    test_accuracy = accuracy_score(y_test, y_pred_test)
    test_precision = precision_score(y_test, y_pred_test, zero_division=0)
    test_recall = recall_score(y_test, y_pred_test, zero_division=0)
    test_f1 = f1_score(y_test, y_pred_test, zero_division=0)
    test_auc = roc_auc_score(y_test_encoded, y_pred_test_proba)
    
    print(f"\n✅ Test Set Performance:")
    print(f"   Accuracy:  {test_accuracy:.4f}")
    print(f"   Precision: {test_precision:.4f}")
    print(f"   Recall:    {test_recall:.4f}")
    print(f"   F1-Score:  {test_f1:.4f}")
    if test_auc > 0:
        print(f"   AUC-ROC:   {test_auc:.4f}")
    
    # Confusion Matrix - binary
    cm = confusion_matrix(y_test, y_pred_test, labels=[-1, 1])
    print(f"\n📋 Confusion Matrix:")
    print(f"   Classes: -1 (down), 1 (up)")
    print(f"   Predicted | Actual -1   Actual 1")
    print(f"   Pred -1   |  {cm[0,0]:6d}     {cm[0,1]:6d}")
    print(f"   Pred  1   |  {cm[1,0]:6d}     {cm[1,1]:6d}")
    
    # Performance Comparison
    train_pred_proba = model.predict(X_train)
    train_pred = (train_pred_proba >= 0.5).astype(int)
    train_pred = np.array([reverse_label_map[p] for p in train_pred])
    train_auc = roc_auc_score(y_train_lgb, train_pred_proba)
    print(f"\n📊 AUC-ROC Comparison:")
    print(f"   Training:   {train_auc:.4f}")
    print(f"   Validation: {val_auc:.4f}")
    print(f"   Test:       {test_auc:.4f}")
    
    # Generate visualizations
    print("\n📈 Generating visualizations...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Confusion Matrix with proper labels
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0], cbar=False,
                xticklabels=['Down(-1)', 'Up(1)'],
                yticklabels=['Down(-1)', 'Up(1)'])
    axes[0, 0].set_title(f'Confusion Matrix - Test Set (Accuracy: {test_accuracy:.4f})')
    axes[0, 0].set_ylabel('True Label')
    axes[0, 0].set_xlabel('Predicted Label')
    
    # Prediction probability distribution
    axes[0, 1].hist(y_pred_test_proba, bins=40, alpha=0.7, color='steelblue', edgecolor='black')
    axes[0, 1].axvline(0.5, color='red', linestyle='--', linewidth=2, label='Decision Threshold')
    axes[0, 1].set_xlabel('Predicted Probability (Class 1)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Distribution of Predicted Probability')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)
    
    # Per-class prediction distribution
    for cls in [-1, 1]:
        mask = y_test == cls
        if mask.sum() > 0:
            axes[1, 0].hist(y_pred_test_proba[mask], bins=30, alpha=0.5, label=f'Class {cls}')
    
    axes[1, 0].axvline(0.5, color='red', linestyle='--', linewidth=2, label='Threshold')
    axes[1, 0].set_xlabel('Predicted Probability')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Probability Distribution by True Class - Test Set')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)
    
    # Metrics Comparison (only include non-zero metrics)
    metrics_list = ['Accuracy', 'Precision', 'Recall', 'F1']
    val_metrics = [val_accuracy, val_precision, val_recall, val_f1]
    test_metrics = [test_accuracy, test_precision, test_recall, test_f1]
    
    x = np.arange(len(metrics_list))
    width = 0.35
    axes[1, 1].bar(x - width/2, val_metrics, width, label='Val', alpha=0.8)
    axes[1, 1].bar(x + width/2, test_metrics, width, label='Test', alpha=0.8)
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].set_title('Performance Metrics Comparison')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(metrics_list)
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3, axis='y')
    axes[1, 1].set_ylim([0, 1.0])
    
    plt.tight_layout()
    plot_path = Path("./artifacts/test_evaluation_kc_lgb.png")
    plt.savefig(plot_path, dpi=100, bbox_inches='tight')
    print(f"✅ Evaluation plot saved to {plot_path}")
    plt.close()
    
else:
    print("⚠️  Test data does not have target values")
    print(f"   Showing prediction statistics only")
    
    # Distribution plot
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.hist(y_pred_test_proba, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    ax.axvline(0.5, color='red', linestyle='--', linewidth=2, label='Decision Threshold')
    ax.set_xlabel('Predicted Probability')
    ax.set_ylabel('Frequency')
    ax.set_title('Prediction Distribution - Test Set')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    
    plot_path = Path("./artifacts/test_predictions_distribution_kc_lgb.png")
    plt.savefig(plot_path, dpi=100, bbox_inches='tight')
    print(f"✅ Distribution plot saved to {plot_path}")
    plt.close()

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*70)
print("✅ LIGHTGBM TRAINING AND EVALUATION COMPLETE")
print("="*70)
print(f"📊 Results Summary:")
print(f"   - Model saved internally")
print(f"   - Predictions: {output_path}")
print(f"   - Evaluation plot: ./artifacts/test_evaluation_kc_lgb.png")
print(f"   - Feature count: {len(all_feature_cols)}")
if y_test is not None:
    print(f"   - Test AUC-ROC: {test_auc:.4f}")
print("="*70)
