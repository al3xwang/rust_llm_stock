#!/usr/bin/env python3
"""
Train a LightGBM base regressor on `next_day_return` and produce residual datasets
Outputs:
- model saved at --model-out
- predictions on train/val: --pred-out-train, --pred-out-val
- residual CSVs (same format as original, but `next_day_return` replaced with residual): train_resid.csv, val_resid.csv
"""
import argparse
import joblib
import pandas as pd
import lightgbm as lgb
from pathlib import Path


def load_df(path):
    return pd.read_csv(path)


def save_df(df, path):
    df.to_csv(path, index=False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--train', default='data/train.csv')
    p.add_argument('--val', default='data/val.csv')
    p.add_argument('--target', default='next_day_return')
    p.add_argument('--model-out', default='artifacts/lgb_base_model.pkl')
    p.add_argument('--pred-out-train', default='artifacts/base_pred_train.csv')
    p.add_argument('--pred-out-val', default='artifacts/base_pred_val.csv')
    p.add_argument('--num-boost-round', type=int, default=500)
    p.add_argument('--verbose', action='store_true')
    args = p.parse_args()

    train = load_df(args.train)
    val = load_df(args.val)

    features = [c for c in train.columns if c not in ['ts_code','trade_date', args.target, 'next_day_direction']]

    # Identify categorical (string/object) columns
    cat_features = [c for c in features if train[c].dtype == 'object' or str(train[c].dtype).startswith('string')]
    num_features = [c for c in features if c not in cat_features]

    # Prepare feature DataFrames
    X_train = train[features].copy()
    X_val = val[features].copy()

    # Fill numeric features with 0.0
    if num_features:
        X_train[num_features] = X_train[num_features].fillna(0.0)
        X_val[num_features] = X_val[num_features].fillna(0.0)

    # For categorical features, label encode them to integers
    for c in cat_features:
        # Combine train and val to get all categories
        combined = pd.concat([X_train[c], X_val[c]], ignore_index=True).fillna('MISSING')
        le = pd.Categorical(combined).codes
        X_train[c] = pd.Categorical(X_train[c].fillna('MISSING')).codes
        X_val[c] = pd.Categorical(X_val[c].fillna('MISSING')).codes

    y_train = train[args.target].fillna(0.0)
    y_val = val[args.target].fillna(0.0)

    # No categorical_feature needed since we label encoded
    dtrain = lgb.Dataset(X_train, label=y_train)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

    params = {
        'objective': 'regression',
        'metric': 'l2',
        'verbosity': -1,
        'num_threads': 4,
    }

    # Use callbacks for compatibility across LightGBM versions
    callbacks = []
    if args.verbose:
        callbacks.append(lgb.log_evaluation(100))
    else:
        callbacks.append(lgb.log_evaluation(-1))
    callbacks.append(lgb.early_stopping(50))

    model = lgb.train(params, dtrain, num_boost_round=args.num_boost_round, valid_sets=[dval], callbacks=callbacks)

    Path(args.model_out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.model_out)

    # Predictions
    pred_train = pd.Series(model.predict(X_train, num_iteration=model.best_iteration), name='base_pred')
    pred_val = pd.Series(model.predict(X_val, num_iteration=model.best_iteration), name='base_pred')

    save_df(pd.concat([train[['ts_code','trade_date']], pred_train], axis=1), args.pred_out_train)
    save_df(pd.concat([val[['ts_code','trade_date']], pred_val], axis=1), args.pred_out_val)

    # Create residual datasets (replace target with residual = true - base_pred)
    train_res = train.copy()
    val_res = val.copy()
    train_res[args.target] = (y_train - pred_train).values
    val_res[args.target] = (y_val - pred_val).values
    # Recompute direction as sign of residual
    def sign_or_zero(x):
        if x > 0:
            return 1
        elif x < 0:
            return -1
        else:
            return 0

    train_res['next_day_direction'] = train_res[args.target].apply(sign_or_zero)
    val_res['next_day_direction'] = val_res[args.target].apply(sign_or_zero)

    save_df(train_res, 'data/train_resid.csv')
    save_df(val_res, 'data/val_resid.csv')

    print('Base model trained; saved model and residual datasets.')


if __name__ == '__main__':
    main()
