"""
P4: counterfactuals under the time-of-detection protocol, with plausibility.

For one dataset, detector, seed, and threshold policy (alpha = 0.05):
  1. score the calibration and test streams once and replay the policy;
  2. sample up to 100 flagged windows exactly as exp_4 does (same seed, same rule);
  3. replay again and keep a snapshot of the conformal predictor at the moment each
     sampled window was flagged, with the search threshold set to the threshold
     used for that decision (posthoc_common.replay_with_snapshots);
  4. run GA-CE, DICE, and PerCE for each sampled window against its own snapshot.

exp_4 instead checked every counterfactual against the threshold at the END of the
stream, which can be +inf under ACI (ANALYSIS_PLAN, P4 note).

Plausibility: 1-NN distance of a window to the normal training windows divided by
the median 1-NN distance of normal validation windows to the same training
windows (near or below 1 = as close to the training data as typical normal data),
for the counterfactual and, as a reference, the anomaly it explains.

Outputs (runs/posthoc/results/cf/):
  <ds>_<det>_s<seed>_<policy>.csv       one row per anomaly x method
  <ds>_<det>_s<seed>_<policy>_cf.npz    anomalies and counterfactuals (float32)
"""
import argparse
import os
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

import posthoc_common as C
from experiments.exp_4_counterfactual_quality import make_explainer, METHOD_NAMES
from experiments.exp_6_cp_ablation import score_stream

METHODS = ['ga_ce', 'dice', 'perce']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--detector', required=True)
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--policy', choices=['sliding', 'aci'], required=True)
    ap.add_argument('--max_anom', type=int, default=100, help='testing only')
    ap.add_argument('--out_sub', default=None, help='testing only')
    a = ap.parse_args()
    np.random.seed(a.seed)
    out = os.path.join(C.OUT, a.out_sub or 'cf')
    os.makedirs(out, exist_ok=True)
    split = C.load_split(a.dataset)
    det = C.load_detector(a.dataset, a.detector, a.seed, split)
    cal = score_stream(det, split['X_val_normal'], 0.0)
    test = score_stream(det, split['X_test'], 0.0)
    preds, _ = C.replay(C.make_cp(a.policy, 0.05, a.dataset, cal), test)
    idx = np.where(preds == 1)[0]
    if len(idx) > a.max_anom:                                   # same sampling rule as exp_4
        idx = idx[np.random.RandomState(a.seed).choice(len(idx), a.max_anom, replace=False)]
    snaps = C.replay_with_snapshots(C.make_cp(a.policy, 0.05, a.dataset, cal), test, idx)
    X_anom = split['X_test'][idx]
    y_anom = split['y_test'][idx].astype(int)

    rs = np.random.RandomState(0)                               # plausibility reference, separate RNG
    tr = split['X_train_normal']
    tr = tr[rs.permutation(len(tr))[:5000]].reshape(min(len(tr), 5000), -1)
    va = split['X_val_normal']
    va = va[rs.permutation(len(va))[:1000]].reshape(min(len(va), 1000), -1)
    nn = NearestNeighbors(n_neighbors=1).fit(tr)
    ref = float(np.median(nn.kneighbors(va)[0][:, 0]))
    plaus = lambda W: nn.kneighbors(W.reshape(len(W), -1))[0][:, 0] / ref
    anom_plaus = plaus(X_anom) if len(X_anom) else np.array([])

    args = SimpleNamespace(n_generations=100, population_size=50)
    rows, arrays = [], {'anomalies': X_anom.astype(np.float32), 'test_index': idx, 'y_anom': y_anom}
    for method in METHODS:
        t0 = time.time()
        first = snaps[int(idx[0])] if len(idx) else C.make_cp(a.policy, 0.05, a.dataset, cal)
        ex = make_explainer(method, det, first, args)
        ex.fit(split['X_val_normal'])
        cfs = np.full(X_anom.shape, np.nan, dtype=np.float32)
        start = len(rows)
        for i, j in enumerate(idx):
            snap = snaps[int(j)]
            ex.conformal_predictor = snap
            thr = float(snap.threshold)
            r = {'method': METHOD_NAMES[method], 'sample_id': i, 'test_index': int(j), 'is_true_anomaly': int(y_anom[i]),
                 'threshold_at_detection': thr, 'alpha_t': float(getattr(snap, 'alpha_t', np.nan)),
                 'cal_pass_rate_at_threshold': float(np.mean(cal <= thr)), 'orig_score': float(test[j]),
                 'anomaly_plausibility': float(anom_plaus[i])}
            try:
                res = ex.generate_counterfactual(X_anom[i], verbose=False)
                cf = res.get('counterfactual')
                r.update(valid=bool(res['validity']), proximity=res['proximity'], sparsity=res['sparsity'])
                if cf is not None:
                    cfs[i] = np.asarray(cf, dtype=np.float32).reshape(X_anom[i].shape)
                    r['cf_score'] = snap._compute_nonconformity(cfs[i], det)
            except Exception as e:                              # recorded, never dropped
                r.update(valid=False, error=str(e)[:200])
            rows.append(r)
        ok = ~np.isnan(cfs).any(axis=(1, 2)) if len(cfs) else np.array([], bool)
        if ok.any():
            p = np.full(len(cfs), np.nan)
            p[ok] = plaus(cfs[ok])
            for k, r in enumerate(rows[start:]):
                r['cf_plausibility'] = float(p[k])
        arrays[f'cf_{method}'] = cfs
        print(f'{a.dataset}/{a.detector}/s{a.seed}/{a.policy}/{method}: {len(idx)} windows, {time.time() - t0:.0f}s', flush=True)
    df = pd.DataFrame(rows)
    for k, v in dict(dataset=a.dataset, detector=a.detector, seed=a.seed, policy=a.policy,
                     n_detected=int((preds == 1).sum()), plausibility_ref=ref).items():
        df[k] = v
    tag = f'{a.dataset}_{a.detector}_s{a.seed}_{a.policy}'
    df.to_csv(os.path.join(out, f'{tag}.csv'), index=False)
    np.savez_compressed(os.path.join(out, f'{tag}_cf.npz'), **arrays)
    print('DONE', tag)


if __name__ == '__main__':
    main()
