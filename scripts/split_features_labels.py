#!/usr/bin/env python3
import glob, csv, os

patterns = [
    'data/*/train.csv',
    'data/*/val.csv',
    'data/*/test.csv',
    'data/train.csv',
    'data/val.csv',
    'data/test.csv',
]

targets = ['next_day_return','next_day_direction','next_3day_return','next_3day_direction']

files = sorted(set(sum([glob.glob(p) for p in patterns], [])))
if not files:
    print('No CSVs found to process.')
    raise SystemExit(0)

for f in files:
    print('Processing', f)
    with open(f, newline='') as infile:
        reader = csv.reader(infile)
        try:
            header = next(reader)
        except StopIteration:
            print('  empty, skipping')
            continue
        low = [h.strip().lower() for h in header]
        target_idx = [i for i,h in enumerate(low) if h in targets]
        id_idx = []
        for name in ('ts_code','trade_date'):
            if name in low:
                id_idx.append(low.index(name))
        # feature indices: all except target indices
        feat_idx = [i for i in range(len(header)) if i not in target_idx]
        # prepare filenames
        base = os.path.dirname(f)
        name = os.path.basename(f)
        feat_file = os.path.join(base, name.replace('.csv','') + '_features.csv')
        label_file = os.path.join(base, name.replace('.csv','') + '_labels.csv')
        # write features (exclude target columns)
        with open(feat_file, 'w', newline='') as fout:
            writer = csv.writer(fout)
            writer.writerow([header[i] for i in feat_idx])
            for row in reader:
                if len(row) < len(header):
                    row = row + [''] * (len(header) - len(row))
                writer.writerow([row[i] for i in feat_idx])
        # write labels (ts_code, trade_date, targets)
        # need to re-open original to iterate again
    with open(f, newline='') as infile:
        reader = csv.reader(infile)
        header = next(reader)
        low = [h.strip().lower() for h in header]
        id_idx = [low.index(name) for name in ('ts_code','trade_date') if name in low]
        target_idx = [i for i,h in enumerate(low) if h in targets]
        with open(label_file, 'w', newline='') as fout:
            writer = csv.writer(fout)
            hdr = [header[i] for i in id_idx + target_idx]
            writer.writerow(hdr)
            for row in reader:
                if len(row) < len(header):
                    row = row + [''] * (len(header) - len(row))
                writer.writerow([row[i] for i in id_idx + target_idx])
    print('  wrote', feat_file, 'and', label_file)

print('Done.')
