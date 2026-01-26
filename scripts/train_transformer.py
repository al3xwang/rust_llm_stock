#!/usr/bin/env python3
"""
Lightweight PyTorch Transformer to model residual sequences.
Input: CSV of rows with columns including ['ts_code','trade_date', features..., 'next_day_return']
Outputs: saved model at --model-out and scaler at --scaler-out
"""
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import joblib


class ResidualSeqDataset(Dataset):
    def __init__(self, df, seq_len=60, features=None, target_col='next_day_return'):
        self.seq_len = seq_len
        self.target_col = target_col
        if features is None:
            self.features = [c for c in df.columns if c not in ['ts_code','trade_date', target_col, 'next_day_direction']]
        else:
            self.features = features
        self.samples = []  # (ts_code, trade_date, X_seq, y)
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        grouped = df.sort_values(['ts_code','trade_date']).groupby('ts_code')
        for ts, g in grouped:
            arr = g[self.features + [self.target_col]].to_numpy()
            if len(arr) <= seq_len:
                continue
            for i in range(len(arr) - seq_len):
                X = arr[i:i+seq_len, :len(self.features)]
                y = arr[i+seq_len, len(self.features)]
                date = g.iloc[i+seq_len]['trade_date'].strftime('%Y-%m-%d')
                self.samples.append((ts, date, X.astype(np.float32), float(y)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ts, date, X, y = self.samples[idx]
        return ts, date, X, y


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x


class TransformerRegressor(nn.Module):
    def __init__(self, n_features, d_model=32, n_head=2, n_layers=1, d_ff=64, dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(n_features, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head, dim_feedforward=d_ff, dropout=dropout)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.pos = PositionalEncoding(d_model)
        self.out = nn.Linear(d_model, 1)

    def forward(self, x):
        # x: batch, seq_len, n_features
        x = self.proj(x)
        x = self.pos(x)
        # transformer expects seq_len, batch, d_model
        x = x.permute(1,0,2)
        h = self.encoder(x)
        h = h.mean(dim=0)
        return self.out(h).squeeze(-1)


def collate_fn(batch):
    ts = [b[0] for b in batch]
    dates = [b[1] for b in batch]
    X = np.stack([b[2] for b in batch])
    y = np.array([b[3] for b in batch], dtype=np.float32)
    return ts, dates, torch.from_numpy(X), torch.from_numpy(y)


def build_scaler_and_features(df):
    features = [c for c in df.columns if c not in ['ts_code','trade_date','next_day_return','next_day_direction']]
    vals = df[features].fillna(0.0).to_numpy(dtype=np.float32)
    mean = vals.mean(axis=0)
    std = vals.std(axis=0)
    std[std==0]=1.0
    return features, mean, std


def apply_scaler_to_samples(samples, features, mean, std):
    # samples is list of (ts,date,X,y) where X is seq_len x n_features
    scaled = []
    for ts,date,X,y in samples:
        Xs = (X - mean) / std
        scaled.append((ts,date,Xs,y))
    return scaled


def load_dataset(df_path, seq_len, features=None, mean=None, std=None, target_col='next_day_return'):
    df = pd.read_csv(df_path)
    if features is None:
        features = [c for c in df.columns if c not in ['ts_code','trade_date',target_col,'next_day_direction']]
    ds = ResidualSeqDataset(df, seq_len=seq_len, features=features, target_col=target_col)
    samples = ds.samples
    if mean is not None and std is not None:
        samples = apply_scaler_to_samples(samples, features, mean, std)
    return samples, features


def train_loop(model, opt, train_loader, val_loader, epochs, device, scheduler=None, checkpoint_path='artifacts/transformer_ckpt.pt', resume=False, save_every=1, patience=None):
    mse = nn.MSELoss()
    best_val = float('inf')
    start_epoch = 1
    if resume and Path(checkpoint_path).exists():
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt['model'])
        opt.load_state_dict(ckpt['opt'])
        start_epoch = ckpt.get('epoch', 0) + 1
        best_val = ckpt.get('best_val', best_val)
        print(f"Resuming from checkpoint at epoch {start_epoch}")

    no_improve = 0
    for ep in range(start_epoch, epochs+1):
        model.train()
        train_losses = []
        for _, _, X, y in train_loader:
            X = X.to(device)
            y = y.to(device)
            opt.zero_grad()
            out = model(X)
            loss = mse(out, y)
            loss.backward()
            opt.step()
            train_losses.append(loss.item())
        if scheduler is not None:
            scheduler.step()
        model.eval()
        val_losses = []
        with torch.no_grad():
            for _, _, X, y in val_loader:
                X = X.to(device)
                y = y.to(device)
                out = model(X)
                val_losses.append(((out - y)**2).mean().item())
        avg_train = np.mean(train_losses)
        avg_val = np.mean(val_losses) if val_losses else float('nan')
        print(f"Epoch {ep}: train_mse={avg_train:.6f}, val_mse={avg_val:.6f}")
        # checkpointing
        if avg_val < best_val:
            best_val = avg_val
            no_improve = 0
            torch.save({'model': model.state_dict(), 'opt': opt.state_dict(), 'epoch': ep, 'best_val': best_val}, checkpoint_path)
            torch.save(model.state_dict(), 'artifacts/transformer_best.pt')
        else:
            no_improve += 1
        if patience is not None and no_improve >= patience:
            print(f"Early stopping triggered after {no_improve} epochs without improvement")
            break
        # periodic save
        if save_every and ep % save_every == 0:
            torch.save({'model': model.state_dict(), 'opt': opt.state_dict(), 'epoch': ep, 'best_val': best_val}, f"artifacts/transformer_epoch_{ep}.pt")
    return model


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--train', default='data/train_resid.csv')
    p.add_argument('--val', default='data/val_resid.csv')
    p.add_argument('--seq-len', type=int, default=60)
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--batch', type=int, default=256)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--d-model', type=int, default=32)
    p.add_argument('--n-head', type=int, default=2)
    p.add_argument('--n-layers', type=int, default=1)
    p.add_argument('--d-ff', type=int, default=64)
    p.add_argument('--dropout', type=float, default=0.1)
    p.add_argument('--lr-scheduler', choices=['none','cosine','step'], default='cosine')
    p.add_argument('--lr-min', type=float, default=1e-6)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--checkpoint', default='artifacts/transformer_ckpt.pt')
    p.add_argument('--save-every', type=int, default=1)
    p.add_argument('--patience', type=int, default=3)
    p.add_argument('--model-out', default='artifacts/transformer_best.pt')
    p.add_argument('--scaler-out', default='artifacts/transformer_scaler.pkl')
    args = p.parse_args()

    Path('artifacts').mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(args.train)
    features, mean, std = build_scaler_and_features(train_df)
    joblib.dump({'features': features, 'mean': mean, 'std': std}, args.scaler_out)

    train_samples, _ = load_dataset(args.train, args.seq_len, features=features)
    val_samples, _ = load_dataset(args.val, args.seq_len, features=features)

    # convert to tensors/scaled
    train_samples = apply_scaler_to_samples(train_samples, features, mean, std)
    val_samples = apply_scaler_to_samples(val_samples, features, mean, std)

    train_loader = DataLoader(train_samples, batch_size=args.batch, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_samples, batch_size=args.batch, shuffle=False, collate_fn=collate_fn)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TransformerRegressor(n_features=len(features), d_model=args.d_model, n_head=args.n_head, n_layers=args.n_layers, d_ff=args.d_ff, dropout=args.dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    # lr scheduler
    scheduler = None
    if args.lr_scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.epochs), eta_min=args.lr_min)
    elif args.lr_scheduler == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.1)

    model = train_loop(model, opt, train_loader, val_loader, args.epochs, device, scheduler=scheduler, checkpoint_path=args.checkpoint, resume=args.resume, save_every=args.save_every, patience=args.patience)
    torch.save(model.state_dict(), args.model_out)
    print('Transformer training complete; model saved to', args.model_out)


if __name__ == '__main__':
    main()
