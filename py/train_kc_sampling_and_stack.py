#!/usr/bin/env python3
"""
Targeted sampling / class-weight experiments and stacking
- Trains LightGBM and XGBoost with upsampling and class-weight variations
- Trains a small MLPClassifier with class weighting and upsampling variants
- Calibrates probabilities on validation set (Logistic Regression)
- Trains a logistic stacker on validation calibrated probs and evaluates on test
- Saves per-model calibrated probs and stacking summary
"""

import os
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import resample
import lightgbm as lgb
from xgboost import XGBClassifier

# Set working directory to project root (use script location as fallback)
proj_root = Path(__file__).resolve().parents[1]
try:
    os.chdir(str(proj_root))
    print(f"Working dir set to {proj_root}")
except Exception as e:
    print(f"Could not change working dir to {proj_root}: {e}")
Path('artifacts').mkdir(exist_ok=True)

print('='*70)
print('🎯 SAMPLING & STACKING EXPERIMENTS')
print('='*70)

# Load data
df_train = pd.read_csv('./data/kc_train.csv')
df_val = pd.read_csv('./data/kc_val.csv')
df_test = pd.read_csv('./data/kc_test.csv')

# Prepare features
exclude_cols = {'ts_code','trade_date','next_day_direction','next_day_return','id','created_at'}
exclude_prefixes = ('industry_emb_','act_ent_type_emb_')
numeric_cols = df_train.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = df_train.select_dtypes(include=['object']).columns.tolist()
feature_cols = [c for c in numeric_cols if c not in exclude_cols and not any(c.startswith(p) for p in exclude_prefixes)]
emb_cols = [c for c in numeric_cols if any(c.startswith(p) for p in exclude_prefixes)]
# detect temporal-only features (month, weekday, quarter, week, day_of, year)
temporal_indicators = ['month','weekday','quarter','week','week_no','day_of','year']
temporal_cols = [c for c in feature_cols if any(t in c.lower() for t in temporal_indicators)]
print(f'Found temporal cols: {temporal_cols}')

imp = SimpleImputer(strategy='median')
X_train_num = pd.DataFrame(imp.fit_transform(df_train[feature_cols]), columns=feature_cols)
X_val_num = pd.DataFrame(imp.transform(df_val[feature_cols]), columns=feature_cols)
X_test_num = pd.DataFrame(imp.transform(df_test[feature_cols]), columns=feature_cols)
for c in emb_cols:
    X_train_num[c] = df_train[c].fillna(0.0)
    X_val_num[c] = df_val[c].fillna(0.0)
    X_test_num[c] = df_test[c].fillna(0.0)
# simple label encode (vectorized mapping for speed and unseen handling)
label_encoders = {}
for c in cat_cols:
    le = LabelEncoder()
    train_vals = df_train[c].astype(str)
    val_vals = df_val[c].astype(str)
    test_vals = df_test[c].astype(str)
    le.fit(train_vals)
    mapping = {cls: i for i, cls in enumerate(le.classes_)}
    X_train_num[c] = train_vals.map(mapping).fillna(0).astype(int)
    X_val_num[c] = val_vals.map(mapping).fillna(0).astype(int)
    X_test_num[c] = test_vals.map(mapping).fillna(0).astype(int)
    label_encoders[c] = le

y_train = df_train['next_day_direction'].map({-1:0,1:1}).astype(int)
y_val = df_val['next_day_direction'].map({-1:0,1:1}).astype(int)
y_test = df_test['next_day_direction'].map({-1:0,1:1}).astype(int)

# Experiment settings
upsample_mults = [1,2,5]
scale_pos_weights = [1,2,5,10]
mlp_configs = [
    {'hidden_layer_sizes':(64,), 'alpha':1e-4},
    {'hidden_layer_sizes':(64,32), 'alpha':1e-4},
]
# Random Forest hyperparameter grid (kept small to limit runtime)
rf_n_estimators = [100, 200]
rf_max_depths = [10, None]
rf_class_weights = [None, 'balanced']

model_records = []
val_features = {}
test_features = {}

# Helper to upsample positive class in training
train_df_full = X_train_num.copy()
train_df_full['y'] = y_train.values

for mult in upsample_mults:
    if mult == 1:
        Xs = X_train_num.copy(); ys = y_train.copy()
        tag = f'upsample_{mult}'
    else:
        pos = train_df_full[train_df_full['y']==1]
        neg = train_df_full[train_df_full['y']==0]
        pos_ups = resample(pos, replace=True, n_samples=int(len(pos)*(mult)), random_state=42)
        up_df = pd.concat([neg, pos_ups], ignore_index=True)
        ys = up_df['y']
        Xs = up_df.drop(columns=['y'])
        tag = f'upsample_{mult}'

    # LightGBM with default params but class weight via scale_pos_weight
    for spw in scale_pos_weights:
        params = {'objective':'binary','metric':['auc'],'num_leaves':24,'learning_rate':0.02,'verbose':-1,'scale_pos_weight':spw}
        dtrain = lgb.Dataset(Xs, label=ys)
        dval = lgb.Dataset(X_val_num, label=y_val, reference=dtrain)
        model = lgb.train(params, dtrain, num_boost_round=500, valid_sets=[dtrain,dval], valid_names=['train','valid'], callbacks=[lgb.early_stopping(stopping_rounds=40), lgb.log_evaluation(period=0)])
        y_val_proba_raw = model.predict(X_val_num)
        # calibrate
        calib = LogisticRegression(solver='lbfgs')
        calib.fit(y_val_proba_raw.reshape(-1,1), y_val)
        y_val_cal = calib.predict_proba(y_val_proba_raw.reshape(-1,1))[:,1]
        y_test_proba_cal = calib.predict_proba(model.predict(X_test_num).reshape(-1,1))[:,1]
        name = f'lgb_up{mult}_spw{spw}'
        model_records.append({'name':name,'family':'lgb','upsample_mult':mult,'scale_pos_weight':spw})
        val_features[name] = y_val_cal
        test_features[name] = y_test_proba_cal
        print(f"Saved feature: {name}")

    # XGBoost
    for spw in scale_pos_weights:
        params = {'n_estimators':300,'max_depth':6,'learning_rate':0.02,'scale_pos_weight':spw,'verbosity':0}
        xgb = XGBClassifier(use_label_encoder=False, **params)
        try:
            xgb.fit(Xs, ys, eval_set=[(X_val_num, y_val)], early_stopping_rounds=40, verbose=False)
        except TypeError:
            xgb.fit(Xs, ys, eval_set=[(X_val_num, y_val)], verbose=False)
        y_val_proba_raw = xgb.predict_proba(X_val_num)[:,1]
        calib = LogisticRegression(solver='lbfgs')
        calib.fit(y_val_proba_raw.reshape(-1,1), y_val)
        y_val_cal = calib.predict_proba(y_val_proba_raw.reshape(-1,1))[:,1]
        y_test_proba_cal = calib.predict_proba(xgb.predict_proba(X_test_num)[:,1].reshape(-1,1))[:,1]
        name = f'xgb_up{mult}_spw{spw}'
        model_records.append({'name':name,'family':'xgb','upsample_mult':mult,'scale_pos_weight':spw})
        val_features[name] = y_val_cal
        test_features[name] = y_test_proba_cal
        print(f"Saved feature: {name}")

    # MLP classifiers
    for cfg in mlp_configs:
        mlp = MLPClassifier(hidden_layer_sizes=cfg['hidden_layer_sizes'], alpha=cfg['alpha'], max_iter=200, random_state=42)
        mlp.fit(Xs, ys)
        y_val_proba_raw = mlp.predict_proba(X_val_num)[:,1]
        calib = LogisticRegression(solver='lbfgs')
        calib.fit(y_val_proba_raw.reshape(-1,1), y_val)
        y_val_cal = calib.predict_proba(y_val_proba_raw.reshape(-1,1))[:,1]
        y_test_proba_cal = calib.predict_proba(mlp.predict_proba(X_test_num)[:,1].reshape(-1,1))[:,1]
        name = f'mlp_up{mult}_hid{cfg["hidden_layer_sizes"]}'
        model_records.append({'name':name,'family':'mlp','upsample_mult':mult,'cfg':cfg})
        val_features[name] = y_val_cal
        test_features[name] = y_test_proba_cal
        print(f"Saved feature: {name}")

    # Temporal features — prefer LSTM predictions if available (artifacts/*temp_lstm*.csv)
    if len(temporal_cols) > 0:
        lval = Path('artifacts/val_predictions_temp_lstm.csv')
        ltest = Path('artifacts/test_predictions_temp_lstm.csv')
        if lval.exists() and ltest.exists():
            # load and align to df_val / df_test
            df_lval = pd.read_csv(lval)
            df_ltest = pd.read_csv(ltest)
            # merge to preserve order
            merged_val = df_val[['ts_code','trade_date']].merge(df_lval, on=['ts_code','trade_date'], how='left')
            merged_test = df_test[['ts_code','trade_date']].merge(df_ltest, on=['ts_code','trade_date'], how='left')
            val_proba = merged_val['proba'].fillna(0.5).values
            test_proba = merged_test['proba'].fillna(0.5).values
            name = f'temp_lstm_up{mult}'
            model_records.append({'name':name,'family':'temporal_lstm','upsample_mult':mult})
            val_features[name] = val_proba
            test_features[name] = test_proba
            print(f"Loaded LSTM temporal predictions and saved feature: {name}")
        else:
            # fallback to classical temporal baselines (LR + small MLP)
            # simple logistic baseline
            temp_lr = LogisticRegression(solver='lbfgs', max_iter=500)
            temp_lr.fit(Xs[temporal_cols], ys)
            y_val_proba_raw = temp_lr.predict_proba(X_val_num[temporal_cols])[:,1]
            calib = LogisticRegression(solver='lbfgs')
            calib.fit(y_val_proba_raw.reshape(-1,1), y_val)
            y_val_cal = calib.predict_proba(y_val_proba_raw.reshape(-1,1))[:,1]
            y_test_proba_cal = calib.predict_proba(temp_lr.predict_proba(X_test_num[temporal_cols])[:,1].reshape(-1,1))[:,1]
            name = f'temp_up{mult}_lr'
            model_records.append({'name':name,'family':'temporal_lr','upsample_mult':mult})
            val_features[name] = y_val_cal
            test_features[name] = y_test_proba_cal
            print(f"Saved feature: {name}")

            # small MLP on temporal features
            temp_mlp = MLPClassifier(hidden_layer_sizes=(32,), alpha=1e-4, max_iter=200, random_state=42)
            temp_mlp.fit(Xs[temporal_cols], ys)
            y_val_proba_raw = temp_mlp.predict_proba(X_val_num[temporal_cols])[:,1]
            calib = LogisticRegression(solver='lbfgs')
            calib.fit(y_val_proba_raw.reshape(-1,1), y_val)
            y_val_cal = calib.predict_proba(y_val_proba_raw.reshape(-1,1))[:,1]
            y_test_proba_cal = calib.predict_proba(temp_mlp.predict_proba(X_test_num[temporal_cols])[:,1].reshape(-1,1))[:,1]
            name = f'temp_up{mult}_mlp'
            model_records.append({'name':name,'family':'temporal_mlp','upsample_mult':mult})
            val_features[name] = y_val_cal
            test_features[name] = y_test_proba_cal
            print(f"Saved feature: {name}")

    # Random Forest hyperparameter grid
    for n in rf_n_estimators:
        for d in rf_max_depths:
            for cw in rf_class_weights:
                rf = RandomForestClassifier(n_estimators=n, max_depth=d, class_weight=cw, random_state=42, n_jobs=-1)
                rf.fit(Xs, ys)
                y_val_proba_raw = rf.predict_proba(X_val_num)[:,1]
                calib = LogisticRegression(solver='lbfgs')
                calib.fit(y_val_proba_raw.reshape(-1,1), y_val)
                y_val_cal = calib.predict_proba(y_val_proba_raw.reshape(-1,1))[:,1]
                y_test_proba_cal = calib.predict_proba(rf.predict_proba(X_test_num)[:,1].reshape(-1,1))[:,1]
                d_label = 'none' if d is None else str(d)
                cw_label = 'bal' if cw == 'balanced' else 'none'
                name = f'rf_up{mult}_n{n}_d{d_label}_cw{cw_label}'
                model_records.append({'name':name,'family':'rf','upsample_mult':mult,'n_estimators':n,'max_depth':d,'class_weight':str(cw)})
                val_features[name] = y_val_cal
                test_features[name] = y_test_proba_cal
                print(f"Saved feature: {name}")

    # Random Forest classifiers (integrated as additional base models)
    for cw in [None, 'balanced']:
        rf = RandomForestClassifier(n_estimators=200, max_depth=10, class_weight=cw, random_state=42, n_jobs=-1)
        rf.fit(Xs, ys)
        y_val_proba_raw = rf.predict_proba(X_val_num)[:,1]
        calib = LogisticRegression(solver='lbfgs')
        calib.fit(y_val_proba_raw.reshape(-1,1), y_val)
        y_val_cal = calib.predict_proba(y_val_proba_raw.reshape(-1,1))[:,1]
        y_test_proba_cal = calib.predict_proba(rf.predict_proba(X_test_num)[:,1].reshape(-1,1))[:,1]
        cw_label = 'bal' if cw == 'balanced' else 'none'
        name = f'rf_up{mult}_cw{cw_label}'
        model_records.append({'name':name,'family':'rf','upsample_mult':mult,'class_weight':str(cw)})
        val_features[name] = y_val_cal
        test_features[name] = y_test_proba_cal
        print(f"Saved feature: {name}")

# Build DataFrames for stacking
val_feat_df = pd.DataFrame(val_features)
test_feat_df = pd.DataFrame(test_features)
val_feat_df['y'] = y_val.values

# Train logistic stacker
stacker = LogisticRegression(solver='lbfgs', max_iter=200)
stacker.fit(val_feat_df.drop(columns=['y']), val_feat_df['y'])

# Predict on test using per-model test features
ens_test_proba = stacker.predict_proba(test_feat_df)[:,1]
# find best threshold on val via cross-validation (use same val set: get stacker val prob on val)
ens_val_proba = stacker.predict_proba(val_feat_df.drop(columns=['y']))[:,1]
best_f1 = 0; best_thr = 0.5
for thr in np.arange(0.2,0.6,0.01):
    preds = (ens_val_proba >= thr).astype(int)
    f1 = f1_score(val_feat_df['y'], preds, zero_division=0)
    if f1 > best_f1:
        best_f1 = f1; best_thr = thr

ens_test_pred = (ens_test_proba >= best_thr).astype(int)
acc = accuracy_score(y_test, ens_test_pred)
prec = precision_score(y_test, ens_test_pred, zero_division=0)
rec = recall_score(y_test, ens_test_pred, zero_division=0)
f1 = f1_score(y_test, ens_test_pred, zero_division=0)
auc = roc_auc_score(y_test, ens_test_proba)

print('\nStacker performance on test:')
print(f'  Acc={acc:.4f}, Prec={prec:.4f}, Rec={rec:.4f}, F1={f1:.4f}, AUC={auc:.4f}, best_thr={best_thr:.2f}')

# Save stack features and summary
val_feat_df.to_csv('artifacts/stack_val_features.csv', index=False)
test_feat_df.to_csv('artifacts/stack_test_features.csv', index=False)
with open('artifacts/stacker_summary.json','w') as f:
    json.dump({'best_thr':float(best_thr),'acc':float(acc),'prec':float(prec),'rec':float(rec),'f1':float(f1),'auc':float(auc),'members':list(val_features.keys())}, f, indent=2)

print('\nSaved stacker feature CSVs and summary in artifacts/')
print('Done.')
