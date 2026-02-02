#!/usr/bin/env python3
"""
Mixed-Model Ensemble: XGBoost and CatBoost (if available)
- Trains XGBoost and CatBoost classifiers
- Calibrates each model on validation predictions (logistic reg)
- Averages calibrated probabilities for ensemble
- Optimizes threshold on validation set by F1
- Saves per-model predictions and ensemble outputs
"""

import os
import json
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, brier_score_loss

os.chdir('/Users/alex/rust_llm_stock')
print('='*70)
print('⚖️  MIXED-MODEL ENSEMBLE (XGBoost + CatBoost)')
print('='*70)

# Load data
print('\n📥 Loading kc_train / kc_val / kc_test')
df_train = pd.read_csv('./data/kc_train.csv')
df_val = pd.read_csv('./data/kc_val.csv')
df_test = pd.read_csv('./data/kc_test.csv')

# Prepare features
exclude_cols = {'ts_code', 'trade_date', 'next_day_direction', 'next_3day_direction', 'next_day_return', 'next_3day_return', 'id', 'created_at'}
exclude_prefixes = ('industry_emb_', 'act_ent_type_emb_')

numeric_cols = df_train.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df_train.select_dtypes(include=['object']).columns.tolist()
feature_cols = [c for c in numeric_cols if c not in exclude_cols and not any(c.startswith(p) for p in exclude_prefixes)]
cat_cols = [c for c in categorical_cols if c not in exclude_cols]
emb_cols = [c for c in numeric_cols if any(c.startswith(p) for p in exclude_prefixes)]

# Impute numeric
imp = SimpleImputer(strategy='median')
X_train = pd.DataFrame(imp.fit_transform(df_train[feature_cols]), columns=feature_cols)
X_val = pd.DataFrame(imp.transform(df_val[feature_cols]), columns=feature_cols)
X_test = pd.DataFrame(imp.transform(df_test[feature_cols]), columns=feature_cols)

# Embeddings
for c in emb_cols:
    X_train[c] = df_train[c].fillna(0.0)
    X_val[c] = df_val[c].fillna(0.0)
    X_test[c] = df_test[c].fillna(0.0)

# Categorical
label_encoders = {}
for c in cat_cols:
    le = LabelEncoder()
    X_train[c] = le.fit_transform(df_train[c].astype(str))
    X_val[c] = le.transform(df_val[c].astype(str))
    X_test[c] = df_test[c].astype(str).apply(lambda x: le.transform([x])[0] if x in le.classes_ else 0)
    label_encoders[c] = le

# Targets 0/1
y_train = df_train['next_day_direction'].map({-1:0,1:1}).astype(int)
y_val = df_val['next_day_direction'].map({-1:0,1:1}).astype(int)
y_test = df_test['next_day_direction'].map({-1:0,1:1}).astype(int)

Path('artifacts').mkdir(exist_ok=True)

models_info = []

# Helper to try import and return None if not available
def try_import(lib_name, alias=None):
    try:
        module = __import__(lib_name)
        return module
    except Exception as e:
        print(f"⚠️  {lib_name} not available: {e}")
        return None

# XGBoost
xgb = try_import('xgboost')
if xgb is not None:
    from xgboost import XGBClassifier
    print('\n--- Training XGBoost')
    xgb_clf = XGBClassifier(use_label_encoder=False, eval_metric='logloss', verbosity=0, n_estimators=200, max_depth=6, learning_rate=0.02, random_state=42)
    xgb_clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=40, verbose=False)
    y_val_proba = xgb_clf.predict_proba(X_val)[:,1]
    y_test_proba = xgb_clf.predict_proba(X_test)[:,1]
    # calibrate
    calib = LogisticRegression(solver='lbfgs')
    calib.fit(y_val_proba.reshape(-1,1), y_val)
    y_val_cal = calib.predict_proba(y_val_proba.reshape(-1,1))[:,1]
    y_test_cal = calib.predict_proba(y_test_proba.reshape(-1,1))[:,1]
    models_info.append({'name':'xgboost','model':xgb_clf, 'calib':calib, 'val_proba_cal':y_val_cal, 'test_proba_cal':y_test_cal})
    pd.DataFrame({'ts_code':df_test['ts_code'],'trade_date':df_test['trade_date'],'pred_proba_cal':y_test_cal}).to_csv('artifacts/test_pred_xg.csv', index=False)

# CatBoost (optional)
catboost = try_import('catboost')
if catboost is not None:
    from catboost import CatBoostClassifier
    print('\n--- Training CatBoost')
    cat_clf = CatBoostClassifier(iterations=500, learning_rate=0.03, depth=6, verbose=0, random_seed=42)
    # CatBoost can take categorical indices
    cat_inds = [X_train.columns.get_loc(c) for c in cat_cols if c in X_train.columns]
    cat_clf.fit(X_train, y_train, eval_set=(X_val, y_val), cat_features=cat_inds)
    y_val_proba = cat_clf.predict_proba(X_val)[:,1]
    y_test_proba = cat_clf.predict_proba(X_test)[:,1]
    calib = LogisticRegression(solver='lbfgs')
    calib.fit(y_val_proba.reshape(-1,1), y_val)
    y_val_cal = calib.predict_proba(y_val_proba.reshape(-1,1))[:,1]
    y_test_cal = calib.predict_proba(y_test_proba.reshape(-1,1))[:,1]
    models_info.append({'name':'catboost','model':cat_clf,'calib':calib,'val_proba_cal':y_val_cal,'test_proba_cal':y_test_cal})
    pd.DataFrame({'ts_code':df_test['ts_code'],'trade_date':df_test['trade_date'],'pred_proba_cal':y_test_cal}).to_csv('artifacts/test_pred_cat.csv', index=False)

# If no new models available, exit
if len(models_info) == 0:
    print('\nNo alternative model families are available (xgboost/catboost missing). Please install them and re-run.')
    raise SystemExit(1)

# Ensemble averaging
print('\n--- Building ensemble from available models')
all_test = np.vstack([m['test_proba_cal'] for m in models_info])
all_val = np.vstack([m['val_proba_cal'] for m in models_info])
ens_test_proba = all_test.mean(axis=0)
ens_val_proba = all_val.mean(axis=0)

# Optimize threshold on val
best_f1 = 0; best_thr = 0.5
for thr in np.arange(0.25,0.6,0.01):
    preds = (ens_val_proba >= thr).astype(int)
    f1 = f1_score(y_val, preds, zero_division=0)
    if f1 > best_f1:
        best_f1 = f1; best_thr = thr
print(f"Ensemble best threshold on val: {best_thr:.2f} (F1: {best_f1:.4f})")

ens_test_pred = (ens_test_proba >= best_thr).astype(int)
acc = accuracy_score(y_test, ens_test_pred)
prec = precision_score(y_test, ens_test_pred, zero_division=0)
rec = recall_score(y_test, ens_test_pred, zero_division=0)
f1 = f1_score(y_test, ens_test_pred, zero_division=0)
auc = roc_auc_score(y_test, ens_test_proba)

print('\nEnsemble Test Metrics:')
print(f"  Acc={acc:.4f}, Prec={prec:.4f}, Rec={rec:.4f}, F1={f1:.4f}, AUC={auc:.4f}")

# Save ensemble predictions
ens_df = pd.DataFrame({'ts_code':df_test['ts_code'],'trade_date':df_test['trade_date'],'actual':df_test['next_day_direction'],'predicted':ens_test_pred,'predicted_proba_ensemble':ens_test_proba})
ens_df.to_csv('artifacts/test_predictions_kc_mixed_ensemble.csv', index=False)

# Save summary
summary = {'members':[m['name'] for m in models_info],'best_thr':float(best_thr),'test_acc':float(acc),'test_prec':float(prec),'test_rec':float(rec),'test_f1':float(f1),'test_auc':float(auc)}
with open('artifacts/mixed_ensemble_summary.json','w') as f:
    json.dump(summary, f, indent=2)

print('\nSaved ensemble outputs to artifacts/test_predictions_kc_mixed_ensemble.csv and artifacts/mixed_ensemble_summary.json')
print('Done.')
