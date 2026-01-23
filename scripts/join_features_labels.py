#!/usr/bin/env python3
import glob, csv, os

patterns = [
    ('data/*/train_features.csv','data/*/train_labels.csv','data/*/train_model.csv'),
    ('data/*/val_features.csv','data/*/val_labels.csv','data/*/val_model.csv'),
    ('data/*/test_features.csv','data/*/test_labels.csv','data/*/test_model.csv'),
    ('data/train_features.csv','data/train_labels.csv','data/train_model.csv'),
    ('data/val_features.csv','data/val_labels.csv','data/val_model.csv'),
    ('data/test_features.csv','data/test_labels.csv','data/test_model.csv'),
]

processed = []
for feat_pat, lab_pat, out_pat in patterns:
    feat_files = glob.glob(feat_pat)
    for feat_file in feat_files:
        base = os.path.dirname(feat_file)
        name = os.path.basename(feat_file)
        # derive label filename in same dir
        label_file = os.path.join(base, name.replace('_features.csv','_labels.csv'))
        out_file = os.path.join(base, name.replace('_features.csv','_model.csv'))
        if not os.path.exists(label_file):
            print('Skipping', feat_file, '- labels not found at', label_file)
            continue
        print('Joining', feat_file, '+', label_file, '->', out_file)
        # load labels into map
        labels = {}
        with open(label_file, newline='') as lf:
            r = csv.DictReader(lf)
            for row in r:
                key = (row.get('ts_code','').strip(), row.get('trade_date','').strip())
                labels[key] = {k:v for k,v in row.items()}
        # join and write
        with open(feat_file, newline='') as ff, open(out_file, 'w', newline='') as of:
            rf = csv.DictReader(ff)
            feat_fields = rf.fieldnames or []
            # determine label fields (exclude ts_code/trade_date duplicate)
            lab_fields = []
            if labels:
                sample_key = next(iter(labels))
                lab_fields = [f for f in labels[sample_key].keys() if f not in ('ts_code','trade_date')]
            out_fields = list(feat_fields) + lab_fields
            writer = csv.DictWriter(of, fieldnames=out_fields)
            writer.writeheader()
            total = 0
            written = 0
            missing = 0
            for row in rf:
                total += 1
                key = (row.get('ts_code','').strip(), row.get('trade_date','').strip())
                lab = labels.get(key)
                if lab is None:
                    missing += 1
                    continue
                out_row = dict(row)
                for f in lab_fields:
                    out_row[f] = lab.get(f,'')
                writer.writerow(out_row)
                written += 1
        print(f'  total={total} written={written} missing_labels={missing}')
        processed.append(out_file)

if not processed:
    print('No model files created.')
else:
    print('Created:', '\n'.join(processed))
