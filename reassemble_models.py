#!/usr/bin/env python3
"""Reassemble the chunked XGBoost model JSONs (split for GitHub's per-file
upload limit). Run once after cloning:
    python3 reassemble_models.py
"""
import glob, os
HERE = os.path.dirname(os.path.abspath(__file__))
for parts_dir in sorted(glob.glob(os.path.join(HERE, '*.json.parts'))):
    out = parts_dir[:-len('.parts')]
    chunks = sorted(glob.glob(os.path.join(parts_dir, '*')))
    with open(out, 'w') as fh:
        for c in chunks:
            fh.write(open(c).read())
    print('wrote', out)
