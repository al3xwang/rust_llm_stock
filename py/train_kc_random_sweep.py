#!/usr/bin/env python3
"""
Randomized Hyperparameter Sweep for KC models
- Runs randomized trials for LightGBM and XGBoost (configurable counts)
- Calibrates predicted probabilities on validation set (LogisticRegression)
- Optimizes threshold by F1 on validation set
- Saves per-trial summaries and aggregated results to artifacts

Usage: python py/train_kc_random_sweep.py --lgb-trials 30 --xgb-trials 20 --seed 42
"""

import os
import json
import argparse
from pathlib import Path
import random
import time

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, brier_score_loss

import lightgbm as lgb
from xgboost import XGBClassifier

# Setup
os.chdir('/Users/alex/rust_llm_stock')
Path('artifacts').mkdir(exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument('--lgb-trials', type=int, default=30)
parser.add_argument('--xgb-trials', type=int, default=20)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--max-rounds', type=int, default=500)
args = parser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)

print('='*70)
print('🔬 RANDOMIZED HYPERPARAM SWEEP')
print(f" LGB trials: {args.lgb_trials}, XGB trials: {args.xgb_trials}, seed: {args.seed}")
print('='*70)

# Load data
print('\n📥 Loading kc_train / kc_val / kc_test')
df_train = pd.read_csv('./data/kc_train.csv')
df_val = pd.read_csv('./data/kc_val.csv')
df_test = pd.read_csv('./data/kc_test.csv')

# Preprocess
exclude_cols = {'ts_code', 'trade_date', 'next_day_direction', 'next_3day_direction', 'next_day_return', 'next_3day_return', 'id', 'created_at'}
exclude_prefixes = ('industry_emb_', 'act_ent_type_emb_')
numeric_cols = df_train.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = df_train.select_dtypes(include=['object']).columns.tolist()
feature_cols = [c for c in numeric_cols if c not in exclude_cols and not any(c.startswith(p) for p in exclude_prefixes)]
emb_cols = [c for c in numeric_cols if any(c.startswith(p) for p in exclude_prefixes)]

imputer = SimpleImputer(strategy='median')
X_train = pd.DataFrame(imputer.fit_transform(df_train[feature_cols]), columns=feature_cols)
X_val = pd.DataFrame(imputer.transform(df_val[feature_cols]), columns=feature_cols)
X_test = pd.DataFrame(imputer.transform(df_test[feature_cols]), columns=feature_cols)
for c in emb_cols:
    X_train[c] = df_train[c].fillna(0.0)
    X_val[c] = df_val[c].fillna(0.0)
    X_test[c] = df_test[c].fillna(0.0)

# label encode categoricals
label_encoders = {}
for c in cat_cols:
    le = LabelEncoder()
    X_train[c] = le.fit_transform(df_train[c].astype(str))
    X_val[c] = le.transform(df_val[c].astype(str))
    X_test[c] = df_test[c].astype(str).apply(lambda x: le.transform([x])[0] if x in le.classes_ else 0)
    label_encoders[c] = le

y_train = df_train['next_day_direction'].map({-1:0,1:1}).astype(int)
y_val = df_val['next_day_direction'].map({-1:0,1:1}).astype(int)
y_test = df_test['next_day_direction'].map({-1:0,1:1}).astype(int)

# helpers

def sample_lgb_params():
    return {
        'objective':'binary',
        'metric':['auc','binary_logloss'],
        'num_leaves': int(2 ** random.uniform(3.0, 6.0)),
        'learning_rate': float(10 ** random.uniform(-3.0, -1.0)),
        'min_data_in_leaf': int(2 ** random.uniform(3, 8)),
        'feature_fraction': round(random.uniform(0.5, 1.0), 3),
        'bagging_fraction': round(random.uniform(0.5, 1.0), 3),
        'scale_pos_weight': random.choice([1,2,5,10]),
        'verbose': -1,
        'seed': random.randint(1, 99999)
    }


def sample_xgb_params():
    return {
        'n_estimators': args.max_rounds,
        'max_depth': random.choice([3,4,5,6,8,10]),
        'learning_rate': float(10 ** random.uniform(-3.0, -1.0)),
        'subsample': round(random.uniform(0.5, 1.0), 3),
        'colsample_bytree': round(random.uniform(0.5, 1.0), 3),
        'scale_pos_weight': random.choice([1,2,5,10]),
        'seed': random.randint(1, 99999),
        'verbosity': 0
    }

results = []
trial_id = 0

# Run LGB trials
print('\n=== Starting LightGBM random trials ===')
for i in range(args.lgb_trials):
    trial_id += 1
    params = sample_lgb_params()
    t0 = time.time()
    dtrain = lgb.Dataset(X_train, label=y_train)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)
    try:
        model = lgb.train(params, dtrain, num_boost_round=args.max_rounds, valid_sets=[dtrain,dval], valid_names=['train','valid'], callbacks=[lgb.log_evaluation(period=0), lgb.early_stopping(stopping_rounds=40)])
    except Exception as e:
        print(f"Trial {trial_id} LGB training error: {e}")
        continue
    y_val_proba_raw = model.predict(X_val)
    y_test_proba_raw = model.predict(X_test)
    # calibrate
    calib = LogisticRegression(solver='lbfgs')
    calib.fit(y_val_proba_raw.reshape(-1,1), y_val)
    y_val_proba = calib.predict_proba(y_val_proba_raw.reshape(-1,1))[:,1]
    y_test_proba = calib.predict_proba(y_test_proba_raw.reshape(-1,1))[:,1]
    # find best thr on val
    best_f1 = 0; best_thr = 0.5
    for thr in np.arange(0.2,0.6,0.01):
        preds = (y_val_proba >= thr).astype(int)
        f1 = f1_score(y_val, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1; best_thr = thr
    # eval test
    preds_test = (y_test_proba >= best_thr).astype(int)
    acc = accuracy_score(y_test, preds_test)
    prec = precision_score(y_test, preds_test, zero_division=0)
    rec = recall_score(y_test, preds_test, zero_division=0)
    f1t = f1_score(y_test, preds_test, zero_division=0)
    auc = roc_auc_score(y_test, y_test_proba)
    brier = brier_score_loss(y_test, y_test_proba)
    t1 = time.time()
    out = {
        'trial_id':trial_id,
        'model_family':'lightgbm',
        'params':params,
        'best_thr':float(best_thr),
        'val_f1':float(best_f1),
        'test_acc':float(acc), 'test_prec':float(prec), 'test_rec':float(rec), 'test_f1':float(f1t), 'test_auc':float(auc), 'test_brier':float(brier),
        'time_s': t1-t0
    }
    results.append(out)
    # save predictions
    pd.DataFrame({'ts_code':df_test['ts_code'],'trade_date':df_test['trade_date'],'pred_proba':y_test_proba,'pred_label':preds_test}).to_csv(f'artifacts/rand_trial_{trial_id}_lgb_preds.csv', index=False)
    print(f"LGB trial {trial_id} done: val_f1={best_f1:.4f}, test_f1={f1t:.4f}, auc={auc:.4f}")

# Run XGB trials
print('\n=== Starting XGBoost random trials ===')
for i in range(args.xgb_trials):
    trial_id += 1
    params = sample_xgb_params()
    t0 = time.time()
    try:
        xgb = XGBClassifier(use_label_encoder=False, **params)
        try:
            xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=40, verbose=False)
        except TypeError:
            xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    except Exception as e:
        print(f"Trial {trial_id} XGB training error: {e}")
        continue
    y_val_proba_raw = xgb.predict_proba(X_val)[:,1]
    y_test_proba_raw = xgb.predict_proba(X_test)[:,1]
    calib = LogisticRegression(solver='lbfgs')
    calib.fit(y_val_proba_raw.reshape(-1,1), y_val)
    y_val_proba = calib.predict_proba(y_val_proba_raw.reshape(-1,1))[:,1]
    y_test_proba = calib.predict_proba(y_test_proba_raw.reshape(-1,1))[:,1]
    best_f1 = 0; best_thr = 0.5
    for thr in np.arange(0.2,0.6,0.01):
        preds = (y_val_proba >= thr).astype(int)
        f1 = f1_score(y_val, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1; best_thr = thr
    preds_test = (y_test_proba >= best_thr).astype(int)
    acc = accuracy_score(y_test, preds_test)
    prec = precision_score(y_test, preds_test, zero_division=0)
    rec = recall_score(y_test, preds_test, zero_division=0)
    f1t = f1_score(y_test, preds_test, zero_division=0)
    auc = roc_auc_score(y_test, y_test_proba)
    brier = brier_score_loss(y_test, y_test_proba)
    t1 = time.time()
    out = {
        'trial_id':trial_id,
        'model_family':'xgboost',
        'params':params,
        'best_thr':float(best_thr),
        'val_f1':float(best_f1),
        'test_acc':float(acc), 'test_prec':float(prec), 'test_rec':float(rec), 'test_f1':float(f1t), 'test_auc':float(auc), 'test_brier':float(brier),
        'time_s': t1-t0
    }
    results.append(out)
    pd.DataFrame({'ts_code':df_test['ts_code'],'trade_date':df_test['trade_date'],'pred_proba':y_test_proba,'pred_label':preds_test}).to_csv(f'artifacts/rand_trial_{trial_id}_xgb_preds.csv', index=False)
    print(f"XGB trial {trial_id} done: val_f1={best_f1:.4f}, test_f1={f1t:.4f}, auc={auc:.4f}")

# Save aggregated results
res_df = pd.DataFrame(results)
res_df.to_csv('artifacts/random_sweep_results.csv', index=False)
with open('artifacts/random_sweep_results.json','w') as f:
    json.dump(results, f, indent=2)

# Report top models by test AUC and test F1
print('\n=== Sweep completed ===')
print('Top by test_auc:')
print(res_df.sort_values('test_auc', ascending=False).head(5)[['trial_id','model_family','test_auc','test_f1','test_prec','test_rec']].to_string(index=False))
print('\nTop by test_f1:')
print(res_df.sort_values('test_f1', ascending=False).head(5)[['trial_id','model_family','test_f1','test_auc','test_prec','test_rec']].to_string(index=False))
print('\nResults saved to artifacts/random_sweep_results.csv / .json')
