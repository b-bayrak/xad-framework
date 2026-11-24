"""
P5: Isolation Forest as the detector inside the same conformal and explanation
pipeline, for one dataset and seed. IsolationForest(200 trees) is fitted on the
flattened normal training windows (as exp_8); its negated score is the
nonconformity score. All four threshold policies are replayed at every alpha;
GA-CE and PerCE (score-based) are run under sliding and ACI at alpha = 0.05 on
up to 100 detected windows, sampled as in exp_4, with the time-of-detection
protocol of posthoc_cf.py. DICE needs gradients and is not
run with this detector.

Outputs (runs/posthoc/results/iforest/):
  <ds>_s<seed>_policies.csv   one row per policy x alpha
  <ds>_s<seed>_cf.csv         one row per anomaly x method x policy
"""
import argparse
import os
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

import posthoc_common as C
from experiments.exp_4_counterfactual_quality import make_explainer, METHOD_NAMES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--max_anom', type=int, default=100, help='testing only')
    ap.add_argument('--out_sub', default=None, help='testing only')
    a = ap.parse_args()
    np.random.seed(a.seed)
    out = os.path.join(C.OUT, a.out_sub or 'iforest')
    os.makedirs(out, exist_ok=True)
    split = C.load_split(a.dataset)
    y = split['y_test'].astype(int)
    Xtr = split['X_train_normal'].reshape(len(split['X_train_normal']), -1)
    t0 = time.time()
    forest = IsolationForest(n_estimators=200, random_state=a.seed, n_jobs=-1).fit(Xtr)
    forest.set_params(n_jobs=1)             # single-window scoring inside the explainers
    det = C.IForestDetector(forest)
    flat = lambda X: X.reshape(len(X), -1)
    cal = -forest.score_samples(flat(split['X_val_normal']))
    test = -forest.score_samples(flat(split['X_test']))
    roc = roc_auc_score(y, test)
    rows = []
    for pol in ['static', 'sliding', 'sliding_all', 'aci']:
        for alpha in C.ALPHAS:
            cp = C.make_cp(pol, alpha, a.dataset, cal)
            preds, _ = C.replay(cp, test)
            cov = float(np.mean(preds[y == 0] == 0))
            rec = recall_score(y, preds, zero_division=0)
            orc = C.oracle_recall_at_fpr(test, y, 1 - cov)
            final_thr = C.current_threshold(cp)
            rows.append({'dataset': a.dataset, 'detector': 'iforest', 'seed': a.seed, 'policy': pol, 'alpha': alpha,
                         'coverage': cov, 'gap': cov - (1 - alpha), 'recall': rec,
                         'precision': precision_score(y, preds, zero_division=0), 'f1': f1_score(y, preds, zero_division=0),
                         'oracle_recall_same_fpr': orc, 'roc_auc': roc, 'final_threshold': final_thr,
                         'pass_rate_normal_test': float(np.mean(test[y == 0] <= final_thr))})
    pd.DataFrame(rows).to_csv(os.path.join(out, f'{a.dataset}_s{a.seed}_policies.csv'), index=False)
    print(f'{a.dataset}/s{a.seed}: policies done ({time.time() - t0:.0f}s), ROC-AUC {roc:.3f}', flush=True)

    cf_rows = []
    args = SimpleNamespace(n_generations=100, population_size=50)
    for pol in ['sliding', 'aci']:
        cp = C.make_cp(pol, 0.05, a.dataset, cal)
        preds, _ = C.replay(cp, test)
        idx = np.where(preds == 1)[0]
        if len(idx) > a.max_anom:
            idx = idx[np.random.RandomState(a.seed).choice(len(idx), a.max_anom, replace=False)]
        snaps = C.replay_with_snapshots(C.make_cp(pol, 0.05, a.dataset, cal), test, idx)   # time-of-detection protocol
        for method in ['ga_ce', 'perce']:
            t1 = time.time()
            ex = make_explainer(method, det, snaps[int(idx[0])] if len(idx) else cp, args)
            ex.fit(split['X_val_normal'])
            for i, j in enumerate(idx):
                ex.conformal_predictor = snaps[int(j)]
                thr = float(snaps[int(j)].threshold)
                r = {'dataset': a.dataset, 'detector': 'iforest', 'seed': a.seed, 'policy': pol,
                     'method': METHOD_NAMES[method], 'sample_id': i, 'test_index': int(j), 'is_true_anomaly': int(y[j]),
                     'n_detected': int(preds.sum()), 'threshold_at_detection': thr,
                     'cal_pass_rate_at_threshold': float(np.mean(cal <= thr))}
                try:
                    res = ex.generate_counterfactual(split['X_test'][j], verbose=False)
                    r.update(valid=bool(res['validity']), proximity=res['proximity'], sparsity=res['sparsity'])
                except Exception as e:
                    r.update(valid=False, error=str(e))
                cf_rows.append(r)
            print(f'{a.dataset}/s{a.seed}/{pol}/{method}: {len(idx)} anomalies, {time.time() - t1:.0f}s', flush=True)
    pd.DataFrame(cf_rows).to_csv(os.path.join(out, f'{a.dataset}_s{a.seed}_cf.csv'), index=False)
    print('DONE', a.dataset, a.seed)


if __name__ == '__main__':
    main()
