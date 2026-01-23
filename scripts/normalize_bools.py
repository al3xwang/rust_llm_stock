#!/usr/bin/env python3
import glob, shutil, os
files = ['data/train_model_small.csv','data/val_model_small.csv']
for f in files:
    if not os.path.exists(f):
        print('Missing', f)
        continue
    bak = f + '.bak'
    if not os.path.exists(bak):
        shutil.copy2(f, bak)
    with open(bak, 'r') as inf, open(f, 'w') as outf:
        for line in inf:
            line = line.replace('True', 'true').replace('False', 'false').replace('TRUE','true').replace('FALSE','false')
            outf.write(line)
    print('Normalized booleans in', f)
