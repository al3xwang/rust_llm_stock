#!/usr/bin/env python3
import glob, csv, os

# Find backup train.csv.bak and corresponding train_model.csv
bak_paths = glob.glob('data/*/train.csv.bak') + glob.glob('data/train.csv.bak')
model_paths = glob.glob('data/*/train_model.csv') + glob.glob('data/train_model.csv')

pairs = []
for bak in bak_paths:
    base = os.path.dirname(bak)
    model = os.path.join(base, 'train_model.csv')
    if model in model_paths:
        pairs.append((bak, model))

if not pairs:
    print('No matching backup/model pairs found. Looked for:', bak_paths, model_paths)
    raise SystemExit(0)

for bak, model in pairs:
    print('Comparing:')
    print('  backup:', bak)
    print('  model :', model)
    # load model into dict
    model_map = {}
    with open(model, newline='') as mf:
        mr = csv.DictReader(mf)
        for r in mr:
            key = (r.get('ts_code','').strip(), r.get('trade_date','').strip())
            model_map[key] = r
    # iterate backup and compare
    total = 0
    matched = 0
    diffs = 0
    difflist = []
    with open(bak, newline='') as bf:
        br = csv.DictReader(bf)
        for r in br:
            total += 1
            key = (r.get('ts_code','').strip(), r.get('trade_date','').strip())
            m = model_map.get(key)
            if m is None:
                continue
            matched += 1
            # compare fields present in both
            diff_cols = []
            for col in set(list(r.keys()) + list(m.keys())):
                a = (r.get(col) or '').strip()
                b = (m.get(col) or '').strip()
                if a != b:
                    diff_cols.append((col, a, b))
            if diff_cols:
                diffs += 1
                if len(difflist) < 10:
                    difflist.append((key, diff_cols))
    print(f'  total rows in backup: {total}')
    print(f'  matched keys in model : {matched}')
    print(f'  rows with any differences: {diffs} (showing up to 10)')
    for key, diffscols in difflist:
        print('\n--- diff for', key, '---')
        for col,a,b in diffscols:
            print(f"{col}: backup='{a}'  model='{b}'")
    print('\nDone for pair.\n')
#!/usr/bin/env python3
import glob, csv, os

# Find backup train.csv.bak and corresponding train_model.csv
bak_paths = glob.glob('data/*/train.csv.bak') + glob.glob('data/train.csv.bak')
model_paths = glob.glob('data/*/train_model.csv') + glob.glob('data/train_model.csv')

pairs = []
for bak in bak_paths:
    base = os.path.dirname(bak)
    model = os.path.join(base, 'train_model.csv')
    if model in model_paths:
        pairs.append((bak, model))

if not pairs:
    print('No matching backup/model pairs found. Looked for:', bak_paths, model_paths)
    raise SystemExit(0)

for bak, model in pairs:
    print('Comparing:')
    print('  backup:', bak)
    print('  model :', model)
    # load model into dict
    model_map = {}
    with open(model, newline='') as mf:
        mr = csv.DictReader(mf)
        for r in mr:
            key = (r.get('ts_code','').strip(), r.get('trade_date','').strip())
            model_map[key] = r
    # iterate backup and compare
    total = 0
    matched = 0
    diffs = 0
    difflist = []
    with open(bak, newline='') as bf:
        br = csv.DictReader(bf)
        for r in br:
            total += 1
            key = (r.get('ts_code','').strip(), r.get('trade_date','').strip())
            m = model_map.get(key)
            if m is None:
                continue
            matched += 1
            # compare fields present in both
            diff_cols = []
            for col in set(list(r.keys()) + list(m.keys())):
                a = (r.get(col) or '').strip()
                b = (m.get(col) or '').strip()
                if a != b:
                    diff_cols.append((col, a, b))
            if diff_cols:
                diffs += 1
                if len(difflist) < 10:
                    difflist.append((key, diff_cols))
    print(f'  total rows in backup: {total}')
    print(f'  matched keys in model : {matched}')
    print(f'  rows with any differences: {diffs} (showing up to 10)')
    for key, diffscols in difflist:
        print('\n--- diff for', key, '---')
        for col,a,b in diffscols:
            print(f"{col}: backup='{a}'  model='{b}'")
    print('\nDone for pair.\n')
