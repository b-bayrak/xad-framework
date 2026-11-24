"""
P1 (uncensored sliding control), P2 (matched false-alarm recall) and the threshold
positions needed by P3, for one dataset and detector over all five seeds.

For each seed: load the cached 5-seed detector, score the calibration (normal
validation) and test streams once (lambda_temp = 0, as deployed), and replay
static, sliding (censored, as in the paper), sliding_all (uncensored control) and
ACI at every alpha. The static/sliding/ACI rows must reproduce exp_6 exactly; the
aggregation step checks this.

Outputs (runs/posthoc/results/policies/):
  <ds>_<det>.csv            one row per seed x policy x alpha
  <ds>_<det>_s<seed>.npz    scores, labels, ACI decisions at alpha=0.05 (for P6),
                            and per-step thresholds of every policy at alpha 0.05 and 0.20
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

import posthoc_common as C
from experiments.exp_6_cp_ablation import score_stream

POLICIES = ['static', 'sliding', 'sliding_all', 'aci']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--detector', required=True)
    ap.add_argument('--seeds', type=int, nargs='+', default=None, help='testing only')
    ap.add_argument('--out_sub', default='policies', help='testing only')
    a = ap.parse_args()
    out = os.path.join(C.OUT, a.out_sub)
    os.makedirs(out, exist_ok=True)
    split = C.load_split(a.dataset)
    y = split['y_test'].astype(int)
    rows = []
    for seed in (a.seeds or C.SEEDS):
        t0 = time.time()
        det = C.load_detector(a.dataset, a.detector, seed, split)
        cal = score_stream(det, split['X_val_normal'], 0.0)
        test = score_stream(det, split['X_test'], 0.0)
        arrays = {'cal_scores': cal, 'test_scores': test, 'y_test': y}
        for pol in POLICIES:
            for alpha in C.ALPHAS:
                cp = C.make_cp(pol, alpha, a.dataset, cal)
                preds, thr = C.replay(cp, test)
                final_thr = C.current_threshold(cp)
                cov = float(np.mean(preds[y == 0] == 0)) if np.any(y == 0) else np.nan
                fpr = 1 - cov
                rec = recall_score(y, preds, zero_division=0)
                orc = C.oracle_recall_at_fpr(test, y, fpr)
                rows.append({
                    'dataset': a.dataset, 'detector': a.detector, 'seed': seed, 'policy': pol, 'alpha': alpha,
                    'coverage': cov, 'gap': cov - (1 - alpha), 'recall': rec,
                    'precision': precision_score(y, preds, zero_division=0), 'f1': f1_score(y, preds, zero_division=0),
                    'fpr': fpr, 'oracle_recall_same_fpr': orc,
                    'recall_ratio_to_oracle': rec / orc if orc and orc > 0 else np.nan,
                    'final_threshold': final_thr,
                    # P3: share of normal windows that would pass the final threshold
                    'pass_rate_normal_test': float(np.mean(test[y == 0] <= final_thr)),
                    'pass_rate_normal_cal': float(np.mean(cal <= final_thr)),
                    'n_test': len(test), 'n_test_anom': int(y.sum()),
                })
                if alpha in (0.05, 0.20) and seed == 42:
                    arrays[f'thr_{pol}_a{int(alpha * 100):02d}'] = thr
                if pol == 'aci' and alpha == 0.05:
                    arrays['aci_preds_a05'] = preds
        np.savez_compressed(os.path.join(out, f'{a.dataset}_{a.detector}_s{seed}.npz'), **arrays)
        print(f'{a.dataset}/{a.detector}/seed{seed}: {len(test)} test windows, {time.time() - t0:.0f}s', flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(out, f'{a.dataset}_{a.detector}.csv'), index=False)
    print('DONE', a.dataset, a.detector)


if __name__ == '__main__':
    main()
