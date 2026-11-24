"""
Shared pieces for the post-hoc analyses P1-P6 (ANALYSIS_PLAN.md). The paper
code (xad/, experiments/) is imported, never modified. Run with XAD_EXTRA=1 and
PYTHONPATH=datasets/register so the ten added datasets are registered.
"""
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import REPO, RUNS_BENCHMARK as E, RUNS_POSTHOC, PREPARED  # noqa: E402
sys.path.insert(0, REPO)

from experiments import pipeline as pl                       # noqa: E402
from xad.temporal_conformal_predictor import TemporalConformalPredictor  # noqa: E402

DS = ['threew', 'daphnet', 'exathlon', 'gecco', 'ghl', 'metropt3', 'msl', 'psm', 'skab', 'smap', 'smd', 'swat']
DETS = ['conv_ae', 'lstm_ae', 'lstm_vae']
SEEDS = [42, 43, 44, 45, 46]
ALPHAS = [0.05, 0.10, 0.15, 0.20]
ACI_GAMMA = 0.005
OUT = os.path.join(RUNS_POSTHOC, 'results')

# identical to the arguments of the 5-seed runs (runs/skab_swat and benchmark sbatch files)
HP = SimpleNamespace(hidden_dim=128, latent_dim=32, num_layers=2, dropout=0.2, learning_rate=0.001,
                     weight_decay=1e-5, kl_weight=0.001, batch_size=32, epochs=200, patience=20)


def data_path(ds):
    return {'skab': os.path.join(REPO, 'data', 'SKAB') + '/', 'swat': os.path.join(REPO, 'data', 'SWaT') + '/'}.get(
        ds, os.path.join(PREPARED, ds))


def cache_dir(ds):
    return os.path.join(REPO, 'runs', 'skab_swat', 'cache') if ds in ('skab', 'swat') else os.path.join(E, 'cache')


def load_split(ds):
    return pl.prepare_split(ds, data_path(ds), 0.2, 0.2, verbose=False)


def load_detector(ds, det, seed, split):
    """Load the cached detector of the 5-seed run. Never trains: a missing checkpoint is an error."""
    ckpt = os.path.join(cache_dir(ds), f'{ds}_{det}_seed{seed}.pt')
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f'cached detector missing: {ckpt}')
    detector, _ = pl.get_or_train_detector(det, ds, split['X_train_normal'], split['X_val_normal'],
                                           split['input_dim'], split['sequence_length'], pl.default_config(HP),
                                           cache_dir=cache_dir(ds), force_retrain=False, seed=seed, verbose=False)
    detector.eval()
    return detector


def window_size(ds):
    return pl.DATASET_CONFIG[ds]['calibration_window']


class UncensoredSlidingCP(TemporalConformalPredictor):
    """P1 control: the sliding window of Eq. (sliding), except that EVERY test score
    enters the window, not only scores of windows classified normal. Everything
    else (seeding from the calibration tail, quantile, window length) is unchanged."""

    def predict_with_confidence(self, x, model=None, update_state=True, precomputed_score=None):
        mode = self.mode
        self.mode = 'sliding'
        try:
            r = super().predict_with_confidence(x, model, update_state=False, precomputed_score=precomputed_score)
        finally:
            self.mode = mode
        if update_state:
            self.threshold = r['threshold']
            self.window.append(r['score'])          # admitted whatever the decision was
            self.threshold_history.append(r['threshold'])
        return r


def make_cp(mode, alpha, ds, cal_scores):
    if mode == 'sliding_all':
        cp = UncensoredSlidingCP(alpha=alpha, window_size=window_size(ds), mode='sliding', aci_gamma=ACI_GAMMA,
                                 smoothness_weight=0.0)
    else:
        cp = TemporalConformalPredictor(alpha=alpha, window_size=window_size(ds), mode=mode, aci_gamma=ACI_GAMMA,
                                        smoothness_weight=0.0)
    cp.calibrate_from_scores(np.asarray(cal_scores, float))
    return cp


def replay(cp, test_scores):
    """Chronological replay; returns predictions and the threshold used at each step."""
    preds = np.empty(len(test_scores), dtype=np.int8)
    thr = np.empty(len(test_scores), dtype=np.float32)
    for i, s in enumerate(test_scores):
        r = cp.predict_with_confidence(None, None, update_state=True, precomputed_score=float(s))
        preds[i] = r['prediction'] == 'anomaly'
        thr[i] = r['threshold']
    return preds, thr


def current_threshold(cp):
    """Threshold a new (synthetic) window would be checked against now, without changing state."""
    return cp.predict_with_confidence(None, None, update_state=False, precomputed_score=0.0)['threshold']


def oracle_recall_at_fpr(test_scores, y, fpr):
    """Recall of the best FIXED threshold with the same false-alarm rate on the test stream (oracle reference)."""
    ns, an = test_scores[y == 0], test_scores[y == 1]
    if len(an) == 0:
        return np.nan
    t = np.quantile(ns, 1 - fpr) if fpr > 0 else np.max(ns)
    return float(np.mean(an > t))


class IForestDetector(torch.nn.Module):
    """Adapter that lets the conformal predictor and the score-based explainers
    (GA-CE, PerCE) use an Isolation Forest: compute_anomaly_score() returns the
    negated sklearn score of the flattened window. No reconstruction, so the TVD
    term is zero, as for the deployed autoencoders (lambda_temp = 0)."""

    def __init__(self, iforest):
        super().__init__()
        self.iforest = iforest
        self._p = torch.nn.Parameter(torch.zeros(1), requires_grad=False)

    def compute_anomaly_score(self, x):
        a = x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)
        a = a.reshape(a.shape[0], -1) if a.ndim == 3 else a.reshape(1, -1)
        return torch.from_numpy(-self.iforest.score_samples(a)), {}

    def forward(self, x):
        return self.compute_anomaly_score(x)[0]


def replay_with_snapshots(cp, test_scores, idx):
    """Replay chronologically and, for each index in idx, keep a copy of the
    predictor exactly as it was when that window was decided, with `threshold`
    set to the threshold used for that decision. The explainers steer their
    search with `cp.threshold` and check validity through predict_with_confidence
    (update_state=False); with the snapshot both refer to the same threshold
    (time-of-detection protocol, ANALYSIS_PLAN P4)."""
    import copy
    want = set(int(i) for i in idx)
    snaps = {}
    for i, s in enumerate(test_scores):
        snap = copy.deepcopy(cp) if i in want else None
        r = cp.predict_with_confidence(None, None, update_state=True, precomputed_score=float(s))
        if snap is not None:
            snap.threshold = float(r['threshold'])
            snap.threshold_history = []          # not needed, keeps snapshots small
            snaps[i] = snap
    return snaps
