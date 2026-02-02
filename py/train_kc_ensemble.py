#!/usr/bin/env python3
"""
KC Ensemble Training
- Train multiple LightGBM models with varied seeds/configs (bagging + hyperparam diversity)
- Calibrate each model's probabilities using the validation set
- Average calibrated probabilities to form ensemble score
- Optimize threshold on val (F1) and evaluate on test
- Save ensemble predictions and summary
"""

import os
import json
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, brier_score_loss

os.chdir('/Users/alex/rust_llm_stock')
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

print('='*70)
print('🤝 KC MODEL ENSEMBLE')
print('='*70)

# Load data
print('\n📥 Loading data...')
df_train = pd.read_csv('./data/kc_train.csv')
df_val = pd.read_csv('./data/kc_val.csv')
df_test = pd.read_csv('./data/kc_test.csv')

# Preprocess features similar to other scripts
exclude_cols = {'ts_code', 'trade_date', 'next_day_direction', 'next_3day_direction', 'next_day_return', 'next_3day_return', 'id', 'created_at'}
exclude_prefixes = ('industry_emb_', 'act_ent_type_emb_')

numeric_cols = df_train.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df_train.select_dtypes(include=['object']).columns.tolist()
feature_cols = [c for c in numeric_cols if c not in exclude_cols and not any(c.startswith(p) for p in exclude_prefixes)]
cat_cols = [c for c in categorical_cols if c not in exclude_cols]
emb_cols = [c for c in numeric_cols if any(c.startswith(p) for p in exclude_prefixes)]

# Impute
imp = SimpleImputer(strategy='median')
X_train = pd.DataFrame(imp.fit_transform(df_train[feature_cols]), columns=feature_cols)
X_val = pd.DataFrame(imp.transform(df_val[feature_cols]), columns=feature_cols)
X_test = pd.DataFrame(imp.transform(df_test[feature_cols]), columns=feature_cols)

# Embeddings / fill
for c in emb_cols:
    X_train[c] = df_train[c].fillna(0.0)
    X_val[c] = df_val[c].fillna(0.0)
    X_test[c] = df_test[c].fillna(0.0)

# Categorical encoding (simple mapping)
label_encoders = {}
for c in cat_cols:
    le = LabelEncoder()
    X_train[c] = le.fit_transform(df_train[c].astype(str))
    X_val[c] = le.transform(df_val[c].astype(str))
    X_test[c] = df_test[c].astype(str).apply(lambda x: le.transform([x])[0] if x in le.classes_ else 0)
    label_encoders[c] = le

# Targets (0/1)
y_train = df_train['next_day_direction'].map({-1:0,1:1}).astype(int)
y_val = df_val['next_day_direction'].map({-1:0,1:1}).astype(int)
y_test = df_test['next_day_direction'].map({-1:0,1:1}).astype(int)

# Ensemble configs: mix of seeds and params
ensemble_specs = [
    {'name':'lgb_1','params':{'num_leaves':20,'learning_rate':0.02,'scale_pos_weight':1,'max_depth':6}, 'seed':1},
    {'name':'lgb_2','params':{'num_leaves':24,'learning_rate':0.03,'scale_pos_weight':1,'max_depth':6}, 'seed':7},
    {'name':'lgb_3','params':{'num_leaves':16,'learning_rate':0.02,'scale_pos_weight':5,'max_depth':5}, 'seed':13},
    {'name':'lgb_4','params':{'num_leaves':32,'learning_rate':0.01,'scale_pos_weight':1,'max_depth':8}, 'seed':21},
    {'name':'lgb_5','params':{'num_leaves':20,'learning_rate':0.02,'scale_pos_weight':10,'max_depth':6}, 'seed':42},
]

Path('artifacts').mkdir(exist_ok=True)

model_outputs = []
calib_models = []

for spec in ensemble_specs:
    name = spec['name']
    params = spec['params'].copy()
    params.update({'objective':'binary','metric':['auc','binary_logloss'],'verbose':-1})
    params['seed'] = spec['seed']
    print(f"\n--- Training {name} (seed={spec['seed']})")

    dtrain = lgb.Dataset(X_train, label=y_train)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

    model = lgb.train(params, dtrain, num_boost_round=1000, valid_sets=[dtrain,dval], valid_names=['train','valid'], callbacks=[lgb.log_evaluation(period=0), lgb.early_stopping(stopping_rounds=40)])

    y_val_proba_raw = model.predict(X_val)
    y_test_proba_raw = model.predict(X_test)

    # calibrate
    calib = LogisticRegression(solver='lbfgs')
    calib.fit(y_val_proba_raw.reshape(-1,1), y_val)
    y_val_cal = calib.predict_proba(y_val_proba_raw.reshape(-1,1))[:,1]
    y_test_cal = calib.predict_proba(y_test_proba_raw.reshape(-1,1))[:,1]

    model_outputs.append({'name':name, 'val_proba_cal': y_val_cal, 'test_proba_cal': y_test_cal})
    calib_models.append(calib)

    # save per-model predictions
    pd.DataFrame({
        'ts_code': df_test['ts_code'].values,
        'trade_date': df_test['trade_date'].values,
        'pred_proba_cal': y_test_cal
    }).to_csv(f'artifacts/test_pred_{name}.csv', index=False)
    print(f"Saved artifacts/test_pred_{name}.csv")

# Create ensemble by averaging calibrated probabilities
print('\n--- Building ensemble predictions (average calibrated probs)')
all_test_probas = np.vstack([m['test_proba_cal'] for m in model_outputs])
all_val_probas = np.vstack([m['val_proba_cal'] for m in model_outputs])

ens_test_proba = all_test_probas.mean(axis=0)
ens_val_proba = all_val_probas.mean(axis=0)

# Find best threshold on val
best_f1 = 0
best_thr = 0.5
for thr in np.arange(0.25,0.6,0.01):
    preds = (ens_val_proba >= thr).astype(int)
    f1 = f1_score(y_val, preds, zero_division=0)
    if f1 > best_f1:
        best_f1 = f1
        best_thr = thr

print(f"Best threshold on val (ensemble): {best_thr:.2f} (F1: {best_f1:.4f})")

ens_test_pred = (ens_test_proba >= best_thr).astype(int)

# Evaluate
acc = accuracy_score(y_test, ens_test_pred)
prec = precision_score(y_test, ens_test_pred, zero_division=0)
rec = recall_score(y_test, ens_test_pred, zero_division=0)
f1 = f1_score(y_test, ens_test_pred, zero_division=0)
auc = roc_auc_score(y_test, ens_test_proba)

print('\nEnsemble performance on test:')
print(f"  Acc: {acc:.4f}, Prec: {prec:.4f}, Rec: {rec:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")

# Save ensemble predictions
ens_df = pd.DataFrame({
    'ts_code': df_test['ts_code'].values,
    'trade_date': df_test['trade_date'].values,
    'actual': df_test['next_day_direction'].values,
    'predicted': ens_test_pred,
    'predicted_proba_ensemble': ens_test_proba,
})
ens_df.to_csv('artifacts/test_predictions_kc_ensemble.csv', index=False)

summary = {
    'ensemble_members': [s['name'] for s in ensemble_specs],
    'best_threshold': float(best_thr),
    'test_acc': float(acc),
    'test_prec': float(prec),
    'test_rec': float(rec),
    'test_f1': float(f1),
    'test_auc': float(auc)
}

with open('artifacts/ensemble_summary.json','w') as f:
    json.dump(summary, f, indent=2)

print('\nSaved ensemble predictions to artifacts/test_predictions_kc_ensemble.csv')
print('Saved ensemble summary to artifacts/ensemble_summary.json')
print('\nDone.')
