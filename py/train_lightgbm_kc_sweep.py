#!/usr/bin/env python3
"""
LightGBM KC - Small Hyperparameter Sweep
- Trains a small set of LightGBM configs (fast) on kc_train/kc_val, evaluates on kc_test
- Calibrates predicted probabilities (logistic on validation probs)
- Saves per-model predictions and a sweep results CSV/JSON
"""

import os
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, brier_score_loss
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb

os.chdir('/Users/alex/rust_llm_stock')
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

print('='*70)
print('⚡ LIGHTGBM KC - HYPERPARAM SWEEP (small)')
print('='*70)

# Load data
print('\n📥 Loading kc_train / kc_val / kc_test...')
df_train = pd.read_csv('./data/kc_train.csv')
df_val = pd.read_csv('./data/kc_val.csv')
df_test = pd.read_csv('./data/kc_test.csv')

target_col = 'next_day_direction'
# Ensure binary
for d in (df_train, df_val, df_test):
    if target_col in d.columns:
        d.dropna(subset=[target_col], inplace=True)
        d = d[d[target_col] != 0]

# Prepare features (reuse logic from final script)
exclude_cols = {'ts_code', 'trade_date', 'next_day_direction', 'next_3day_direction', 'next_day_return', 'next_3day_return', 'id', 'created_at'}
exclude_prefixes = ('industry_emb_', 'act_ent_type_emb_')

numeric_cols = df_train.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df_train.select_dtypes(include=['object']).columns.tolist()
feature_cols = [c for c in numeric_cols if c not in exclude_cols and not any(c.startswith(p) for p in exclude_prefixes)]
cat_cols = [c for c in categorical_cols if c not in exclude_cols]
emb_cols = [c for c in numeric_cols if any(c.startswith(p) for p in exclude_prefixes)]
all_features = feature_cols + cat_cols + emb_cols

# Impute numeric
imputer = SimpleImputer(strategy='median')
X_train = pd.DataFrame(imputer.fit_transform(df_train[feature_cols]), columns=feature_cols)
X_val = pd.DataFrame(imputer.transform(df_val[feature_cols]), columns=feature_cols)
X_test = pd.DataFrame(imputer.transform(df_test[feature_cols]), columns=feature_cols)

# Handle embeddings
for c in emb_cols:
    X_train[c] = df_train[c].fillna(0.0)
    X_val[c] = df_val[c].fillna(0.0)
    X_test[c] = df_test[c].fillna(0.0)

# Label encode categoricals
label_encoders = {}
for c in cat_cols:
    le = LabelEncoder()
    X_train[c] = le.fit_transform(df_train[c].astype(str))
    X_val[c] = le.transform(df_val[c].astype(str))
    # map unseen to 0
    X_test[c] = df_test[c].astype(str).apply(lambda x: le.transform([x])[0] if x in le.classes_ else 0)
    label_encoders[c] = le

y_train = df_train[target_col].map({-1:0,1:1}).astype(int)
y_val = df_val[target_col].map({-1:0,1:1}).astype(int)
y_test = df_test[target_col].map({-1:0,1:1}).astype(int)

# Small grid
configs = [
    {'name':'baseline','num_leaves':20,'learning_rate':0.02,'scale_pos_weight':1,'max_depth':6},
    {'name':'shallow_high_lr','num_leaves':20,'learning_rate':0.05,'scale_pos_weight':1,'max_depth':4},
    {'name':'deep_balanced','num_leaves':40,'learning_rate':0.02,'scale_pos_weight':1,'max_depth':10},
    {'name':'pos_w_5','num_leaves':20,'learning_rate':0.02,'scale_pos_weight':5,'max_depth':6},
    {'name':'pos_w_10','num_leaves':20,'learning_rate':0.02,'scale_pos_weight':10,'max_depth':6},
    {'name':'shallow_lr_pos10','num_leaves':20,'learning_rate':0.05,'scale_pos_weight':10,'max_depth':4},
]

results = []
Path('artifacts').mkdir(exist_ok=True)

for cfg in configs:
    name = cfg['name']
    print('\n' + '-'*60)
    print(f"🔁 Training config: {name}")
    params = {
        'objective':'binary', 'metric':['auc','binary_logloss'],
        'num_leaves':cfg['num_leaves'], 'learning_rate':cfg['learning_rate'],
        'scale_pos_weight':cfg['scale_pos_weight'], 'max_depth':cfg['max_depth'],
        'verbose':-1
    }

    dtrain = lgb.Dataset(X_train, label=y_train)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

    model = lgb.train(
        params,
        dtrain,
        num_boost_round=1000,
        valid_sets=[dtrain, dval],
        valid_names=['train','valid'],
        callbacks=[lgb.log_evaluation(period=0), lgb.early_stopping(stopping_rounds=40)]
    )

    # predict
    y_val_proba = model.predict(X_val)
    y_test_proba_raw = model.predict(X_test)

    # calibrate
    clf = LogisticRegression(solver='lbfgs')
    clf.fit(y_val_proba.reshape(-1,1), y_val)
    y_val_proba_cal = clf.predict_proba(y_val_proba.reshape(-1,1))[:,1]
    y_test_proba_cal = clf.predict_proba(y_test_proba_raw.reshape(-1,1))[:,1]

    # find best threshold on val by F1
    best_f1 = 0
    best_thr = 0.5
    for thr in np.arange(0.25,0.6,0.05):
        preds = (y_val_proba_cal >= thr).astype(int)
        f1 = f1_score(y_val, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr

    # evaluate on test
    y_test_pred = (y_test_proba_cal >= best_thr).astype(int)

    acc = accuracy_score(y_test, y_test_pred)
    prec = precision_score(y_test, y_test_pred, zero_division=0)
    rec = recall_score(y_test, y_test_pred, zero_division=0)
    f1 = f1_score(y_test, y_test_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_test_proba_cal)
    brier = brier_score_loss(y_test, y_test_proba_cal)

    # quantile-based quick test
    proba_col = y_test_proba_cal
    summary_quant = {}
    for q in [0.05,0.1,0.2]:
        low = np.quantile(proba_col, q)
        high = np.quantile(proba_col, 1-q)
        mask = (proba_col <= low) | (proba_col >= high)
        if mask.sum() == 0:
            summary_quant[f'q{int(q*100)}'] = None
            continue
        strat_ret = (np.where(y_test_pred[mask]==1, df_test.loc[mask,'next_day_return'].values, -df_test.loc[mask,'next_day_return'].values)).sum()
        summary_quant[f'q{int(q*100)}'] = {'trades':int(mask.sum()), 'tot_return':float(strat_ret)}

    # Save predictions for this model
    pred_df = pd.DataFrame({
        'ts_code': df_test['ts_code'].values,
        'trade_date': df_test['trade_date'].values,
        'actual': df_test['next_day_direction'].values,
        'predicted': y_test_pred,  # mapped as 0/1
        'predicted_proba_raw': y_test_proba_raw,
        'predicted_proba_cal': y_test_proba_cal
    })
    pred_path = f'artifacts/test_predictions_kc_sweep_{name}.csv'
    pred_df.to_csv(pred_path, index=False)

    # Save summary
    summary = {
        'name': name,
        'params': cfg,
        'best_threshold': float(best_thr),
        'test_acc': float(acc), 'test_prec': float(prec), 'test_rec': float(rec), 'test_f1': float(f1), 'test_auc': float(auc), 'test_brier': float(brier),
        'quantile_summary': summary_quant,
        'predictions_file': pred_path
    }
    with open(f'artifacts/lgb_sweep_{name}_summary.json','w') as f:
        json.dump(summary, f, indent=2)

    results.append(summary)
    print(f"  ✅ {name}: f1={f1:.4f}, auc={auc:.4f}, prec={prec:.4f}, rec={rec:.4f}, thr={best_thr}")

# Save aggregated results
res_df = pd.DataFrame([{
    'name': r['name'],
    'test_acc': r['test_acc'],
    'test_prec': r['test_prec'],
    'test_rec': r['test_rec'],
    'test_f1': r['test_f1'],
    'test_auc': r['test_auc'],
    'best_threshold': r['best_threshold']
} for r in results])
res_df.to_csv('artifacts/lgb_sweep_results.csv', index=False)

with open('artifacts/lgb_sweep_results.json','w') as f:
    json.dump(results, f, indent=2)

print('\n' + '='*70)
print('✅ SWEEP COMPLETE — results saved to artifacts/lgb_sweep_results.csv / .json')
print('='*70)
