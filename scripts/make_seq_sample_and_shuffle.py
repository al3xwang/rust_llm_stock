#!/usr/bin/env python3
"""Select stocks with enough history (>= seq_len+1 rows), create a subset CSV, and shuffle labels globally."""
import csv, glob, sys, random
from collections import Counter

SEQ_LEN = 60
REQUIRED = SEQ_LEN + 1
SRC_GLOB = 'rust_llm_stock/data/*/train.csv'
# Support being run from repository root or parent directory
ALTERNATE_GLOB = 'data/*/train.csv'
OUT_SUBSET = 'data/train_seq_sample.csv'
OUT_SHUFFLED = 'data/train_seq_shuffled.csv'

# find source file
srcs = glob.glob(ALTERNATE_GLOB) or glob.glob(SRC_GLOB)
if not srcs:
    print('train.csv not found', file=sys.stderr); sys.exit(1)
src = srcs[0]
print('source:', src)

# First pass: count per stock
counts = Counter()
with open(src, 'r', newline='') as f:
    rdr = csv.reader(f)
    header = next(rdr)
    ts_idx = header.index('ts_code')
    for r in rdr:
        counts[r[ts_idx]] += 1

# choose stocks with enough rows
good_stocks = [s for s, c in counts.items() if c >= REQUIRED]
print('stocks with >=', REQUIRED, ':', len(good_stocks))
# sort by count descending and pick top 1000 stocks or so
good_stocks.sort(key=lambda s: counts[s], reverse=True)
selected = set(good_stocks[:1000])
print('selected stocks:', len(selected))

# second pass: write rows for selected stocks
with open(src, 'r', newline='') as inf, open(OUT_SUBSET, 'w', newline='') as outf:
    rdr = csv.reader(inf)
    w = csv.writer(outf)
    header = next(rdr)
    w.writerow(header)
    ts_idx = header.index('ts_code')
    for r in rdr:
        if r[ts_idx] in selected:
            w.writerow(r)

print('Wrote subset to', OUT_SUBSET)

# Third step: shuffle next_day_return globally
# find column index
with open(OUT_SUBSET, 'r', newline='') as f:
    rdr = csv.reader(f)
    header = next(rdr)
    t_idx = header.index('next_day_return')
    rows = list(rdr)

print('subset rows:', len(rows))
labels = [r[t_idx] for r in rows]
random.seed(42)
random.shuffle(labels)
for i, r in enumerate(rows):
    r[t_idx] = labels[i]

# normalize True/False to lowercase, but we only changed numeric column
with open(OUT_SHUFFLED, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(header)
    for r in rows:
        r = ['true' if x=='True' else ('false' if x=='False' else x) for x in r]
        w.writerow(r)

print('Wrote shuffled labels file:', OUT_SHUFFLED)