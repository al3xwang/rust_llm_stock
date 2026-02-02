#!/usr/bin/env python3
"""
Mixed Ensemble (LightGBM + XGBoost + HistGradientBoosting)
- Trains each model, calibrates probabilities on validation set (logistic),
- Averages calibrated probabilities to build ensemble, optimizes threshold on val,
- Evaluates on test and saves outputs.
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
from sklearn.ensemble import HistGradientBoostingClassifier

import lightgbm as lgb
from xgboost import XGBClassifier

os.chdir('/Users/alex/rust_llm_stock')
print('='*70)
print('🔀 MIXED ENSEMBLE V2 (LGBM + XGB + HGB)')
print('='*70)

# Load data
print('\n📥 Loading kc_train / kc_val / kc_test')
df_train = pd.read_csv('./data/kc_train.csv')
df_val = pd.read_csv('./data/kc_val.csv')
df_test = pd.read_csv('./data/kc_test.csv')

# Feature selection / preprocessing
exclude_cols = {'ts_code', 'trade_date', 'next_day_direction', 'next_3day_direction', 'next_day_return', 'next_3day_return', 'id', 'created_at'}
exclude_prefixes = ('industry_emb_', 'act_ent_type_emb_')

numeric_cols = df_train.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df_train.select_dtypes(include=['object']).columns.tolist()
feature_cols = [c for c in numeric_cols if c not in exclude_cols and not any(c.startswith(p) for p in exclude_prefixes)]
cat_cols = [c for c in categorical_cols if c not in exclude_cols]
emb_cols = [c for c in numeric_cols if any(c.startswith(p) for p in exclude_prefixes)]

# Impute numerics
imp = SimpleImputer(strategy='median')
X_train = pd.DataFrame(imp.fit_transform(df_train[feature_cols]), columns=feature_cols)
X_val = pd.DataFrame(imp.transform(df_val[feature_cols]), columns=feature_cols)
X_test = pd.DataFrame(imp.transform(df_test[feature_cols]), columns=feature_cols)

# embeddings
for c in emb_cols:
    X_train[c] = df_train[c].fillna(0.0)
    X_val[c] = df_val[c].fillna(0.0)
    X_test[c] = df_test[c].fillna(0.0)

# categorical
label_encoders = {}
for c in cat_cols:
    le = LabelEncoder()
    X_train[c] = le.fit_transform(df_train[c].astype(str))
    X_val[c] = le.transform(df_val[c].astype(str))
    X_test[c] = df_test[c].astype(str).apply(lambda x: le.transform([x])[0] if x in le.classes_ else 0)
    label_encoders[c] = le

# targets 0/1
y_train = df_train['next_day_direction'].map({-1:0,1:1}).astype(int)
y_val = df_val['next_day_direction'].map({-1:0,1:1}).astype(int)
y_test = df_test['next_day_direction'].map({-1:0,1:1}).astype(int)

Path('artifacts').mkdir(exist_ok=True)

members = []

# LightGBM
print('\n--- Training LightGBM')
lgb_params = {'objective':'binary','metric':['auc','binary_logloss'],'num_leaves':24,'learning_rate':0.02,'verbose':-1,'seed':42}
dtrain = lgb.Dataset(X_train, label=y_train)
dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)
model_lgb = lgb.train(lgb_params, dtrain, num_boost_round=1000, valid_sets=[dtrain,dval], valid_names=['train','valid'], callbacks=[lgb.log_evaluation(period=0), lgb.early_stopping(stopping_rounds=40)])
y_val_proba = model_lgb.predict(X_val)
y_test_proba = model_lgb.predict(X_test)
calib = LogisticRegression(solver='lbfgs')
calib.fit(y_val_proba.reshape(-1,1), y_val)
y_val_cal = calib.predict_proba(y_val_proba.reshape(-1,1))[:,1]
y_test_cal = calib.predict_proba(y_test_proba.reshape(-1,1))[:,1]
members.append({'name':'lightgbm','val_cal':y_val_cal,'test_cal':y_test_cal})
pd.DataFrame({'ts_code':df_test['ts_code'],'pred_proba_cal':y_test_cal}).to_csv('artifacts/test_pred_lgb.csv', index=False)

# XGBoost
print('\n--- Training XGBoost')
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', verbosity=0, n_estimators=300, max_depth=6, learning_rate=0.02, random_state=7)
try:
    # some xgboost versions accept early_stopping_rounds in fit; if not, fallback
    xgb.fit(X_train, y_train, eval_set=[(X_val,y_val)], early_stopping_rounds=40, verbose=False)
except TypeError:
    xgb.fit(X_train, y_train, eval_set=[(X_val,y_val)], verbose=False)

y_val_proba = xgb.predict_proba(X_val)[:,1]
y_test_proba = xgb.predict_proba(X_test)[:,1]
calib = LogisticRegression(solver='lbfgs')
calib.fit(y_val_proba.reshape(-1,1), y_val)
y_val_cal = calib.predict_proba(y_val_proba.reshape(-1,1))[:,1]
y_test_cal = calib.predict_proba(y_test_proba.reshape(-1,1))[:,1]
members.append({'name':'xgboost','val_cal':y_val_cal,'test_cal':y_test_cal})
pd.DataFrame({'ts_code':df_test['ts_code'],'pred_proba_cal':y_test_cal}).to_csv('artifacts/test_pred_xgb.csv', index=False)

# HistGradientBoosting (sklearn)
print('\n--- Training HistGradientBoosting')
hgb = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_depth=8, random_state=13)
hgb.fit(X_train, y_train)
y_val_proba = hgb.predict_proba(X_val)[:,1]
y_test_proba = hgb.predict_proba(X_test)[:,1]
calib = LogisticRegression(solver='lbfgs')
calib.fit(y_val_proba.reshape(-1,1), y_val)
y_val_cal = calib.predict_proba(y_val_proba.reshape(-1,1))[:,1]
y_test_cal = calib.predict_proba(y_test_proba.reshape(-1,1))[:,1]
members.append({'name':'histgb','val_cal':y_val_cal,'test_cal':y_test_cal})
pd.DataFrame({'ts_code':df_test['ts_code'],'pred_proba_cal':y_test_cal}).to_csv('artifacts/test_pred_histgb.csv', index=False)

# Ensemble
print('\n--- Building ensemble (average calibrated probs)')
all_val = np.vstack([m['val_cal'] for m in members])
all_test = np.vstack([m['test_cal'] for m in members])
ens_val = all_val.mean(axis=0)
ens_test = all_test.mean(axis=0)

# threshold search on val
best_f1 = 0; best_thr = 0.5
for thr in np.arange(0.2,0.6,0.01):
    preds = (ens_val >= thr).astype(int)
    f1 = f1_score(y_val, preds, zero_division=0)
    if f1 > best_f1:
        best_f1 = f1; best_thr = thr

print(f"Best threshold on val: {best_thr:.2f} (F1={best_f1:.4f})")

ens_pred = (ens_test >= best_thr).astype(int)
acc = accuracy_score(y_test, ens_pred)
prec = precision_score(y_test, ens_pred, zero_division=0)
rec = recall_score(y_test, ens_pred, zero_division=0)
f1 = f1_score(y_test, ens_pred, zero_division=0)
auc = roc_auc_score(y_test, ens_test)

print('\nEnsemble test metrics:')
print(f"Acc={acc:.4f}, Prec={prec:.4f}, Rec={rec:.4f}, F1={f1:.4f}, AUC={auc:.4f}")

# Save ensemble preds
ens_df = pd.DataFrame({'ts_code':df_test['ts_code'],'trade_date':df_test['trade_date'],'actual':df_test['next_day_direction'],'predicted':ens_pred,'predicted_proba_ensemble':ens_test})
ens_df.to_csv('artifacts/test_predictions_kc_mixed_ensemble_v2.csv', index=False)

summary = {'members':[m['name'] for m in members],'best_thr':float(best_thr),'test_acc':float(acc),'test_prec':float(prec),'test_rec':float(rec),'test_f1':float(f1),'test_auc':float(auc)}
with open('artifacts/mixed_ensemble_v2_summary.json','w') as f:
    json.dump(summary, f, indent=2)

print('\nSaved results to artifacts/test_predictions_kc_mixed_ensemble_v2.csv and artifacts/mixed_ensemble_v2_summary.json')
print('Done.')
