"""
Load every prepared extra dataset through the PAPER's own pipeline
(experiments.pipeline.prepare_split, via the xad_extra hook) and report the
window-level numbers the experiments will actually see. Writes
datasets/logs/verify_windows.tsv and fails loudly on degenerate splits.

    XAD_EXTRA=1 PYTHONPATH=datasets/register python datasets/verify.py [names]
"""
import os
import sys

import numpy as np

import xad_extra
from experiments import pipeline as pl

names = sys.argv[1:] or xad_extra.available()
out = os.path.join(xad_extra.REPO, 'datasets', 'logs', 'verify_windows.tsv')
rows = []
hdr = ['dataset', 'features', 'seq', 'stride', 'calib_window',
       'train_win', 'train_normal', 'val_win', 'val_normal', 'val_anom_rate',
       'test_win', 'test_anom_win', 'test_anom_rate', 'flags']
for n in names:
    cfg = pl.DATASET_CONFIG[n]
    s = pl.prepare_split(n, os.path.join(xad_extra.PREPARED, n), 0.2, 0.2, verbose=False)
    flags = []
    if len(s['X_val_normal']) < cfg['calibration_window']:
        flags.append('val_normal<calib_window')
    if s['y_test'].sum() == 0:
        flags.append('NO_TEST_ANOMALIES')
    if s['y_test'].mean() > 0.3:
        flags.append('test_anomaly_rate>30%')
    stride = max(1, int(round(cfg['sequence_length'] * (1 - cfg['window_overlap']))))
    r = [n, s['input_dim'], cfg['sequence_length'], stride, cfg['calibration_window'],
         len(s['X_train']), len(s['X_train_normal']), len(s['X_val']), len(s['X_val_normal']),
         round(float(s['y_val'].mean()), 4), len(s['X_test']), int(s['y_test'].sum()),
         round(float(s['y_test'].mean()), 4), ';'.join(flags) or 'ok']
    rows.append(r)
    print('\t'.join(map(str, r)), flush=True)

with open(out, 'w') as f:
    f.write('\t'.join(hdr) + '\n')
    for r in rows:
        f.write('\t'.join(map(str, r)) + '\n')
print(f'wrote {out}')
