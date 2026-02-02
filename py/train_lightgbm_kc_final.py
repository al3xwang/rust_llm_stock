#!/usr/bin/env python3
"""
LightGBM KC - Single Best Model with Threshold Tuning

Focus on a single well-tuned model with threshold optimization.
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, brier_score_loss
)
from sklearn.linear_model import LogisticRegression

import warnings
warnings.filterwarnings("ignore")

os.chdir("/Users/alex/rust_llm_stock")
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

print("="*70)
print("🚀 LIGHTGBM KC - BEST MODEL WITH THRESHOLD TUNING")
print("="*70)

# Load data
print("\n📊 Loading Data...")
df_train = pd.read_csv("./data/kc_train.csv")
df_val = pd.read_csv("./data/kc_val.csv")
df_test = pd.read_csv("./data/kc_test.csv")

target_col = 'next_day_direction'
df_train = df_train[df_train[target_col] != 0].copy()
df_val = df_val[df_val[target_col] != 0].copy()
df_test = df_test[df_test[target_col] != 0].copy()

# Preprocess
print("🔧 Preprocessing...")
exclude_cols = {'ts_code', 'trade_date', 'next_day_direction', 'next_3day_direction',
                'next_day_return', 'next_3day_return', 'id', 'created_at'}
exclude_prefixes = ('industry_emb_', 'act_ent_type_emb_')

numeric_cols = df_train.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df_train.select_dtypes(include=['object']).columns.tolist()

feature_cols = [c for c in numeric_cols if c not in exclude_cols and not any(c.startswith(p) for p in exclude_prefixes)]
categorical_feature_cols = [c for c in categorical_cols if c not in exclude_cols]
embedding_cols = [c for c in numeric_cols if any(c.startswith(p) for p in exclude_prefixes)]

all_feature_cols = feature_cols + categorical_feature_cols + embedding_cols

imputer = SimpleImputer(strategy='median')
df_train[feature_cols] = imputer.fit_transform(df_train[feature_cols])
df_val[feature_cols] = imputer.transform(df_val[feature_cols])
df_test[feature_cols] = imputer.transform(df_test[feature_cols])

for emb_col in embedding_cols:
    df_train[emb_col].fillna(0.0, inplace=True)
    df_val[emb_col].fillna(0.0, inplace=True)
    df_test[emb_col].fillna(0.0, inplace=True)

label_encoders = {}
for cat_col in categorical_feature_cols:
    le = LabelEncoder()
    df_train[cat_col] = le.fit_transform(df_train[cat_col].astype(str))
    df_val[cat_col] = le.transform(df_val[cat_col].astype(str))
    df_test[cat_col] = df_test[cat_col].astype(str).apply(
        lambda x: le.transform([x])[0] if x in le.classes_ else 0
    )
    label_encoders[cat_col] = le

X_train = df_train[all_feature_cols]
y_train = df_train[target_col]
X_val = df_val[all_feature_cols]
y_val = df_val[target_col]
X_test = df_test[all_feature_cols]
y_test = df_test[target_col]

label_map = {-1: 0, 1: 1}
reverse_label_map = {0: -1, 1: 1}
y_train_enc = y_train.map(label_map).astype(int)
y_val_enc = y_val.map(label_map).astype(int)
y_test_enc = y_test.map(label_map).astype(int)

# Train best model
print("🚀 Training Model...")
params = {
    'objective': 'binary',
    'metric': ['auc', 'binary_logloss'],
    'num_leaves': 20,
    'learning_rate': 0.02,
    'feature_fraction': 0.7,
    'bagging_fraction': 0.7,
    'bagging_freq': 3,
    'verbose': -1,
    'seed': RANDOM_SEED,
    'max_depth': 6,
    'min_data_in_leaf': 70,
    'lambda_l1': 0.8,
    'lambda_l2': 0.8,
    'scale_pos_weight': 2.0,  # Emphasize positive class
}

train_data = lgb.Dataset(X_train, label=y_train_enc)
val_data = lgb.Dataset(X_val, label=y_val_enc, reference=train_data)

model = lgb.train(
    params,
    train_data,
    num_boost_round=500,
    valid_sets=[train_data, val_data],
    valid_names=['train', 'valid'],
    callbacks=[
        lgb.log_evaluation(period=100),
        lgb.early_stopping(stopping_rounds=40)
    ]
)

# Find optimal threshold
print("\n🔍 Finding Optimal Threshold...")
y_val_proba = model.predict(X_val)

best_f1 = 0
best_threshold = 0.5

for threshold in np.arange(0.3, 0.7, 0.05):
    y_pred_binary = (y_val_proba >= threshold).astype(int)
    y_pred_orig = np.array([reverse_label_map[p] for p in y_pred_binary])
    f1 = f1_score(y_val, y_pred_orig, zero_division=0)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

print(f"✅ Optimal threshold: {best_threshold:.2f} (F1: {best_f1:.4f})")

# Evaluate with optimal threshold
print("\n📊 Final Evaluation (with optimal threshold):")

# Validation
y_val_pred_binary = (y_val_proba >= best_threshold).astype(int)
y_val_pred = np.array([reverse_label_map[p] for p in y_val_pred_binary])

val_acc = accuracy_score(y_val, y_val_pred)
val_prec = precision_score(y_val, y_val_pred, zero_division=0)
val_rec = recall_score(y_val, y_val_pred, zero_division=0)
val_f1 = f1_score(y_val, y_val_pred, zero_division=0)
val_auc = roc_auc_score(y_val_enc, y_val_proba)

print(f"\nValidation (threshold={best_threshold:.2f}):")
print(f"  AUC-ROC:   {val_auc:.4f}")
print(f"  Accuracy:  {val_acc:.4f}")
print(f"  Precision: {val_prec:.4f}")
print(f"  Recall:    {val_rec:.4f}")
print(f"  F1-Score:  {val_f1:.4f}")

# Test
y_test_proba = model.predict(X_test)
y_test_pred_binary = (y_test_proba >= best_threshold).astype(int)
y_test_pred = np.array([reverse_label_map[p] for p in y_test_pred_binary])

test_acc = accuracy_score(y_test, y_test_pred)
test_prec = precision_score(y_test, y_test_pred, zero_division=0)
test_rec = recall_score(y_test, y_test_pred, zero_division=0)
test_f1 = f1_score(y_test, y_test_pred, zero_division=0)
test_auc = roc_auc_score(y_test_enc, y_test_proba)

print(f"\nTest (threshold={best_threshold:.2f}):")
print(f"  AUC-ROC:   {test_auc:.4f}")
print(f"  Accuracy:  {test_acc:.4f}")
print(f"  Precision: {test_prec:.4f}")
print(f"  Recall:    {test_rec:.4f}")
print(f"  F1-Score:  {test_f1:.4f}")

# ------------------------------------------------------------------
# Calibrate predicted probabilities using validation set
# ------------------------------------------------------------------
print("\n🔧 Calibrating predicted probabilities (sigmoid/logistic fit on val)")
calib_clf = LogisticRegression(solver='lbfgs')
val_probs_for_calib = y_val_proba.reshape(-1, 1)
calib_clf.fit(val_probs_for_calib, y_val_enc)

# Apply to validation and test probabilities
y_val_proba_cal = calib_clf.predict_proba(val_probs_for_calib)[:, 1]
y_test_proba_cal = calib_clf.predict_proba(y_test_proba.reshape(-1, 1))[:, 1]

# Calibration diagnostics
val_brier = brier_score_loss(y_val_enc, y_val_proba_cal)
test_brier = brier_score_loss(y_test_enc, y_test_proba_cal)
print(f"Calibration Brier score - val: {val_brier:.6f}, test: {test_brier:.6f}")
print(f"Val probs range: {y_val_proba_cal.min():.4f}-{y_val_proba_cal.max():.4f}; Test probs range: {y_test_proba_cal.min():.4f}-{y_test_proba_cal.max():.4f}")

# Update test predictions to include calibrated proba
y_test_proba_for_saving = y_test_proba_cal

# Confusion matrices
cm_val = confusion_matrix(y_val, y_val_pred, labels=[-1, 1])
cm_test = confusion_matrix(y_test, y_test_pred, labels=[-1, 1])

print(f"\nValidation Confusion Matrix:")
print(f"             Pred -1  Pred 1")
print(f"  Actual -1  {cm_val[0,0]:6d}  {cm_val[0,1]:6d}")
print(f"  Actual  1  {cm_val[1,0]:6d}  {cm_val[1,1]:6d}")

print(f"\nTest Confusion Matrix:")
print(f"             Pred -1  Pred 1")
print(f"  Actual -1  {cm_test[0,0]:6d}  {cm_test[0,1]:6d}")
print(f"  Actual  1  {cm_test[1,0]:6d}  {cm_test[1,1]:6d}")

# Feature importance
print("\n📊 Top 15 Important Features:")
importance_df = pd.DataFrame({
    'feature': all_feature_cols,
    'importance': model.feature_importance(importance_type='gain')
}).sort_values('importance', ascending=False)

print(importance_df.head(15).to_string(index=False))

# Save predictions
pred_df = pd.DataFrame({
    'ts_code': df_test['ts_code'],
    'trade_date': df_test['trade_date'],
    'actual': y_test,
    'predicted': y_test_pred,
    'predicted_proba_raw': y_test_proba,
    'predicted_proba_calibrated': y_test_proba_for_saving
})
pred_df.to_csv("./artifacts/test_predictions_kc_final.csv", index=False)

print("\n✅ Predictions (raw + calibrated) saved to artifacts/test_predictions_kc_final.csv")
print("="*70)
print("✅ TRAINING COMPLETE")
print("="*70)
