"""
Exploratory diagnostic (ANALYSIS_PLAN deviations, 2026-10-21, found during P6): how
much of the test stream ACI spends with an infinite threshold.

ACI's update is label-free (err_t = 1 iff the window was flagged), so a long run of
flagged windows pushes alpha_t below 1/(m+1), the threshold becomes +inf, and no
window can be flagged until alpha_t has climbed back at gamma * alpha per step. For
one dataset, all detectors, seeds and alphas, the ACI replay is recomputed from the
P1 scores (no detector is run) and summarised. The replay must reproduce the
P1 coverage and recall (and, at alpha = 0.05, every stored decision); the script
stops otherwise.

Output (runs/posthoc/results/aci_trace/<ds>.csv), one row per
detector x seed x alpha:
  share_inf_pos / share_inf_neg  share of test steps with threshold +inf / -inf
  anom_under_inf                 share of anomalous test windows that arrive while
                                 the threshold is +inf (cannot be detected)
  longest_inf_run, inf_episodes  in test steps
  alpha_t_min                    lowest alpha_t reached
  coverage_finite, recall_finite coverage and recall over steps with a finite threshold
"""
import argparse
import os

import numpy as np
import pandas as pd

import posthoc_common as C


def runs(mask):
    """Lengths of consecutive True runs."""
    if not mask.any():
        return np.array([], int)
    d = np.diff(np.concatenate([[0], mask.astype(int), [0]]))
    return np.where(d == -1)[0] - np.where(d == 1)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True)
    a = ap.parse_args()
    out = os.path.join(C.OUT, 'aci_trace')
    os.makedirs(out, exist_ok=True)
    rows = []
    for det in C.DETS:
        p1 = pd.read_csv(os.path.join(C.OUT, 'policies', f'{a.dataset}_{det}.csv'))
        for seed in C.SEEDS:
            z = np.load(os.path.join(C.OUT, 'policies', f'{a.dataset}_{det}_s{seed}.npz'))
            y, test = z['y_test'].astype(int), z['test_scores']
            for alpha in C.ALPHAS:
                cp = C.make_cp('aci', alpha, a.dataset, z['cal_scores'])
                preds, thr = C.replay(cp, test)
                ref = p1[(p1.seed == seed) & (p1.policy == 'aci') & np.isclose(p1.alpha, alpha)].iloc[0]
                cov = float(np.mean(preds[y == 0] == 0))
                rec = float(preds[y == 1].mean()) if y.any() else np.nan
                # CSV round trip changes the last bit, hence 1e-12; decisions are compared exactly below
                assert abs(cov - ref.coverage) < 1e-12 and (abs(rec - ref.recall) < 1e-12 or (np.isnan(rec) and ref.recall == 0)), \
                    f'replay differs from P1: {a.dataset} {det} {seed} {alpha}'
                if alpha == 0.05:
                    assert (preds == z['aci_preds_a05']).all()
                pos, neg = np.isposinf(thr), np.isneginf(thr)
                fin = ~(pos | neg)
                r = runs(pos)
                rows.append({
                    'dataset': a.dataset, 'detector': det, 'seed': seed, 'alpha': alpha, 'n_test': len(test),
                    'n_anom': int(y.sum()), 'coverage': cov, 'recall': rec,
                    'share_inf_pos': float(pos.mean()), 'share_inf_neg': float(neg.mean()),
                    'anom_under_inf': float(pos[y == 1].mean()) if y.any() else np.nan,
                    'longest_inf_run': int(r.max()) if len(r) else 0, 'inf_episodes': len(r),
                    'alpha_t_min': float(np.min(cp.alpha_history)),
                    'coverage_finite': float(np.mean(preds[fin & (y == 0)] == 0)) if (fin & (y == 0)).any() else np.nan,
                    'recall_finite': float(preds[fin & (y == 1)].mean()) if (fin & (y == 1)).any() else np.nan,
                })
        print(f'{a.dataset}/{det}: done', flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(out, f'{a.dataset}.csv'), index=False)
    print('DONE', a.dataset)


if __name__ == '__main__':
    main()
