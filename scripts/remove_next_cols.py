#!/usr/bin/env python3
import glob
import csv
import os
import shutil

patterns = [
    'data/*/train.csv',
    'data/*/val.csv',
    'data/*/test.csv',
    'data/train.csv',
    'data/val.csv',
    'data/test.csv',
]

files = []
for p in patterns:
    files.extend(glob.glob(p))
files = sorted(set(files))

if not files:
    print('No matching CSV files found.')
    raise SystemExit(0)

for f in files:
    print('Processing', f)
    bak = f + '.bak'
    if not os.path.exists(bak):
        shutil.copy2(f, bak)
        print('  backup ->', bak)
    tmp = f + '.tmp'
    changed = False
    with open(f, newline='') as infile:
        reader = csv.reader(infile)
        try:
            header = next(reader)
        except StopIteration:
            print('  empty file, skipping')
            continue
        keep_idx = [i for i,h in enumerate(header) if not h.strip().lower().startswith('next_')]
        if len(keep_idx) == len(header):
            print('  no next_ columns found')
            continue
        # write tmp
        with open(tmp, 'w', newline='') as outfile:
            writer = csv.writer(outfile)
            writer.writerow([header[i] for i in keep_idx])
            for row in reader:
                # pad row if shorter
                if len(row) < len(header):
                    row = row + [''] * (len(header) - len(row))
                writer.writerow([row[i] for i in keep_idx])
        os.replace(tmp, f)
        print('  removed', len(header) - len(keep_idx), 'columns, wrote', f)

print('Done.')
