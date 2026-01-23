#!/usr/bin/env python3
"""Create a small subsequence dataset (top N stocks by record count) and shuffle labels."""
import csv, glob, sys, random
from collections import Counter

SRC = 'data/*/train.csv'
OUT = 'data/train_seq_shuffled_small.csv'
N = 50
SEQ_LEN = 60
REQUIRED = SEQ_LEN + 1

files = glob.glob(SRC)
if not files:
    print('train.csv not found under data/*', file=sys.stderr); sys.exit(1)
src = files[0]
print('source:', src)

counts = Counter()
with open(src, 'r', newline='') as f:
    rdr = csv.reader(f)
    header = next(rdr)
    ts_idx = header.index('ts_code')
    for r in rdr:
        counts[r[ts_idx]] += 1

good = [s for s,c in counts.items() if c >= REQUIRED]
good.sort(key=lambda s: counts[s], reverse=True)
selected = set(good[:N])
print('selected stocks:', len(selected))

# write subset
with open(src, 'r', newline='') as inf, open(OUT, 'w', newline='') as outf:
    rdr = csv.reader(inf)
    w = csv.writer(outf)
    header = next(rdr)
    w.writerow(header)
    ts_idx = header.index('ts_code')
    for r in rdr:
        if r[ts_idx] in selected:
            w.writerow(r)

# shuffle labels globally
with open(OUT, 'r', newline='') as f:
    rdr = csv.reader(f)
    header = next(rdr)
    t_idx = header.index('next_day_return')
    rows = list(rdr)
labels = [r[t_idx] for r in rows]
random.seed(42)
random.shuffle(labels)
for i,r in enumerate(rows):
    r[t_idx] = labels[i]

with open(OUT, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(header)
    for r in rows:
        r = ['true' if x=='True' else ('false' if x=='False' else x) for x in r]
        w.writerow(r)
print('wrote', OUT)