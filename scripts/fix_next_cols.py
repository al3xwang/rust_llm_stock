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

allowed_targets = set(['next_day_return','next_day_direction','next_3day_return','next_3day_direction'])

files = []
for p in patterns:
    files.extend(glob.glob(p))
files = sorted(set(files))

if not files:
    print('No matching CSV files found.')
    raise SystemExit(0)

for f in files:
    bak = f + '.bak'
    if os.path.exists(bak):
        print('Restoring from backup', bak)
        shutil.copy2(bak, f)
    else:
        print('No backup for', f, ' — proceeding with current file')
    tmp = f + '.tmp'
    with open(f, newline='') as infile:
        reader = csv.reader(infile)
        try:
            header = next(reader)
        except StopIteration:
            print('  empty file, skipping', f)
            continue
        keep_idx = [i for i,h in enumerate(header) if not (h.strip().lower().startswith('next_') and h.strip().lower() not in allowed_targets)]
        if len(keep_idx) == len(header):
            print('  no non-target next_ columns to remove in', f)
            continue
        with open(tmp, 'w', newline='') as outfile:
            writer = csv.writer(outfile)
            writer.writerow([header[i] for i in keep_idx])
            for row in reader:
                if len(row) < len(header):
                    row = row + [''] * (len(header) - len(row))
                writer.writerow([row[i] for i in keep_idx])
        os.replace(tmp, f)
        print('  removed', len(header) - len(keep_idx), 'columns (kept targets) in', f)

print('Done.')
