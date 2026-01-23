#!/usr/bin/env python3
import pandas as pd, math
CSV='data/train.csv'
print('Loading',CSV)
df=pd.read_csv(CSV, low_memory=False)
print('shape',df.shape)
print('next_day_return stats')
print(df['next_day_return'].describe())
print('dups (ts_code,trade_date)=', df.duplicated(subset=['ts_code','trade_date']).sum())
# features
exclude={'ts_code','trade_date','next_day_return','next_day_direction','next_3day_return','next_3day_direction'}
feat=[c for c in df.columns if c not in exclude]
print('features',len(feat))
res=[]
for c in feat:
    try:
        x=pd.to_numeric(df[c], errors='coerce')
    except Exception:
        continue
    mask=~x.isna() & ~df['next_day_return'].isna()
    if mask.sum()<30: continue
    px=x[mask]; py=df['next_day_return'][mask]
    if px.std()==0 or py.std()==0: continue
    pear=px.corr(py)
    res.append((c,pear,mask.sum()))
res.sort(key=lambda r: abs(r[1]) if r[1] is not None else 0, reverse=True)
print('max |pearson|', res[0] if res else None)
print('top 10')
for r in res[:10]: print(r)
