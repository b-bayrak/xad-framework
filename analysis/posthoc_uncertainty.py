"""
P6: does the four-factor uncertainty decomposition carry information? For one
dataset, on a fixed random sample of up to 1000 test windows:

  distributional  mean k-NN distance to training windows      (detector-free)
  density         LOF score                                    (detector-free)
  temporal        percentile of first-difference variance      (detector-free)
  feature_oor     largest per-feature out-of-range rate        (pre-specified, detector-free)
  feature         mean per-feature uncertainty, computed with per-feature
                  reconstruction errors of the detector (shape F). The framework
                  passes errors of shape (1, T, F), which _compute_feature_uncertainty
                  indexes by time step; see ANALYSIS_PLAN P6 note. Per detector x seed.

The framework's UncertaintyExplainer is fitted on up to 5000 normal training
windows. For each detector and seed, the ACI decisions at alpha = 0.05 from the P1
run give the errors to predict; each factor's AUC for predicting a wrong decision
is reported, plus Spearman correlations between factors. Needs P1 outputs.

Outputs (runs/posthoc/results/uncertainty/):
  <ds>_factors.csv   detector-free factor values for the sampled windows
  <ds>_auc.csv       correlations and AUCs (rows: kind, detector, seed, factor, value)
"""
import argparse
import os

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

import posthoc_common as C
from xad.uncertainty_explainer import UncertaintyExplainer

FREE = ['distributional', 'density', 'temporal', 'feature_oor']


def per_feature_errors(det, X):
    """Mean squared reconstruction error per feature, shape (len(X), F). One window
    at a time, because compute_anomaly_score expects a single window."""
    errs = np.empty((len(X), X.shape[2]), dtype=np.float64)
    dev = next(det.parameters()).device
    with torch.no_grad():
        for i in range(len(X)):
            _, d = det.compute_anomaly_score(torch.from_numpy(X[i:i + 1]).float().to(dev))
            rec = d['reconstruction']
            rec = rec.detach().cpu().numpy() if hasattr(rec, 'detach') else np.asarray(rec)
            errs[i] = ((rec.reshape(X.shape[1:]) - X[i]) ** 2).mean(axis=0)
    return errs


def auc(wrong, v):
    v = np.asarray(v, float)
    if not (0 < wrong.sum() < len(wrong)) or np.nanmax(v) - np.nanmin(v) == 0:
        return np.nan
    return roc_auc_score(wrong, v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--dets', nargs='+', default=None, help='testing only')
    ap.add_argument('--seeds', type=int, nargs='+', default=None, help='testing only')
    ap.add_argument('--policies_sub', default='policies', help='testing only')
    ap.add_argument('--out_sub', default='uncertainty', help='testing only')
    a = ap.parse_args()
    out = os.path.join(C.OUT, a.out_sub)
    os.makedirs(out, exist_ok=True)
    split = C.load_split(a.dataset)
    rs = np.random.RandomState(0)
    tr = split['X_train_normal']
    tr = tr[rs.permutation(len(tr))[:5000]]
    X = split['X_test']
    sample = np.sort(rs.permutation(len(X))[:1000])
    Xs = X[sample]
    np.random.seed(0)                                    # UncertaintyExplainer subsamples training statistics
    ue = UncertaintyExplainer(training_data=tr)
    F = pd.DataFrame({'test_index': sample,
                      'distributional': [ue._compute_epistemic_uncertainty(x)['distance'] for x in Xs],
                      'density': [ue._compute_aleatoric_uncertainty(x)['lof_score'] for x in Xs],
                      'temporal': [ue._compute_temporal_uncertainty(x)['percentile'] for x in Xs],
                      'feature_oor': [float(np.max(ue._compute_feature_uncertainty(x))) for x in Xs]})
    F['y'] = split['y_test'][sample].astype(int)
    F.to_csv(os.path.join(out, f'{a.dataset}_factors.csv'), index=False)

    rows = []
    for det_name in (a.dets or C.DETS):
        for seed in (a.seeds or C.SEEDS):
            det = C.load_detector(a.dataset, det_name, seed, split)
            pfe = per_feature_errors(det, Xs)
            feat = np.array([np.mean(ue._compute_feature_uncertainty(x, e)) for x, e in zip(Xs, pfe)])
            vals = {**{f: F[f].values for f in FREE}, 'feature': feat}
            z = np.load(os.path.join(C.OUT, a.policies_sub, f'{a.dataset}_{det_name}_s{seed}.npz'))
            wrong = (z['aci_preds_a05'][sample] != z['y_test'][sample]).astype(int)
            for f, v in vals.items():
                rows.append({'dataset': a.dataset, 'kind': 'auc_error', 'detector': det_name, 'seed': seed,
                             'factor': f, 'value': auc(wrong, v), 'error_rate': float(wrong.mean()),
                             'constant': bool(np.nanmax(v) - np.nanmin(v) == 0)})
            names = list(vals)
            M = np.column_stack([vals[n] for n in names])
            for i in range(len(names)):
                for k in range(i + 1, len(names)):
                    if np.ptp(M[:, i]) > 0 and np.ptp(M[:, k]) > 0:
                        r = spearmanr(M[:, i], M[:, k]).correlation
                    else:
                        r = np.nan
                    rows.append({'dataset': a.dataset, 'kind': 'corr', 'detector': det_name, 'seed': seed,
                                 'factor': f'{names[i]}~{names[k]}', 'value': r})
    pd.DataFrame(rows).to_csv(os.path.join(out, f'{a.dataset}_auc.csv'), index=False)
    print('DONE', a.dataset)


if __name__ == '__main__':
    main()
