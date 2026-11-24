#!/usr/bin/env python
"""
    Experiment 6: Conformal threshold-policy and nonconformity ablation.

    Head-to-head comparison of the three threshold policies
        static  - plain split conformal (threshold fixed at calibration)
        sliding - this paper's sliding-window recalibration (Eq. 8)
        aci     - Adaptive Conformal Inference (Gibbs & Candes, 2021)
    crossed with alpha in {0.05, 0.10, 0.15, 0.20} and with the TVD term
    (Eq. 9-10) on/off (smoothness_weight in {0.0, 0.1}).

    For each configuration it reports:
        - coverage (true-normal specificity - the guaranteed quantity)
        - calibration error |coverage - (1-alpha)|
        - accuracy, F1/precision/recall at the conformal threshold
        - ROC-AUC / PR-AUC of the nonconformity scores (policy-independent;
          repeated per smoothness_weight so the TVD term's effect on ranking
          power is isolated explicitly)

    Model forward passes are done ONCE per smoothness_weight (scores depend
    on the nonconformity measure, not on the threshold policy or alpha);
    every policy x alpha cell is then replayed through the real
    TemporalConformalPredictor logic on the same precomputed scores, so the
    comparison is exact, not a reimplementation.

    Uses the SAME cached detector as exp_1..exp_5 for a given
    (dataset, detector).

    Usage:
        python experiments/exp_6_cp_ablation.py --dataset swat --data_path ./data/SWaT/ --detector lstm_ae
"""

import os
import sys
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments import pipeline as pl
from xad.temporal_conformal_predictor import TemporalConformalPredictor

ALPHAS = [0.05, 0.10, 0.15, 0.20]
MODES = ['static', 'sliding', 'aci']
# 0.1 is the paper's stated lambda_temp; 0.01 probes whether the TVD term's
# raw magnitude (a sum over T-1 timesteps, much larger than the mean
# reconstruction error) simply swamps the base score at 0.1 - the SKAB
# diagnosis showed exactly that (raw-score ROC-AUC 0.61 vs 0.52 with +0.1*TVD).
SMOOTHNESS_WEIGHTS = [0.0, 0.01, 0.1]


def score_stream(detector, X, smoothness_weight: float) -> np.ndarray:
    """Nonconformity scores for X, in order (one forward pass per window)."""
    scorer = TemporalConformalPredictor(smoothness_weight=smoothness_weight, mode='static')
    return np.array([scorer._compute_nonconformity(X[i], detector) for i in range(len(X))])


def replay_policy(cal_scores, test_scores, mode, alpha, window_size, aci_gamma):
    """Run one threshold policy over precomputed scores, chronologically."""
    cp = TemporalConformalPredictor(alpha=alpha, window_size=window_size,
                                    mode=mode, aci_gamma=aci_gamma)
    cp.calibrate_from_scores(cal_scores)
    preds = np.empty(len(test_scores), dtype=int)
    for i, s in enumerate(test_scores):
        r = cp.predict_with_confidence(None, None, update_state=True, precomputed_score=s)
        preds[i] = 1 if r['prediction'] == 'anomaly' else 0
    return preds


def run_experiment(args):
    print("\n" + "-" * 70)
    print("EXPERIMENT 6: CP Threshold-Policy x TVD Ablation")
    print("-" * 70)

    split = pl.prepare_split(args.dataset, args.data_path, args.test_size, args.val_size)
    config = pl.default_config(args)
    window_size = pl.DATASET_CONFIG[args.dataset]['calibration_window']

    all_rows = []
    detectors = [args.detector] if getattr(args, 'detector', None) else pl.DETECTORS
    for detector_type in detectors:
        print(f"\n{'-'*60}\n{detector_type.upper()}\n{'-'*60}")
        detector, _ = pl.get_or_train_detector(
            detector_type, args.dataset,
            split['X_train_normal'], split['X_val_normal'],
            split['input_dim'], split['sequence_length'],
            config, cache_dir=args.cache_dir, force_retrain=args.force_retrain,
            seed=args.seed, verbose=True,
        )

        for sw in SMOOTHNESS_WEIGHTS:
            print(f"\n  Scoring calibration + test streams (smoothness_weight={sw}) ...")
            cal_scores = score_stream(detector, split['X_val_normal'], sw)
            test_scores = score_stream(detector, split['X_test'], sw)

            roc = roc_auc_score(split['y_test'], test_scores)
            pr = average_precision_score(split['y_test'], test_scores)
            print(f"  score ranking power: ROC-AUC={roc:.4f}, PR-AUC={pr:.4f}")

            for mode in MODES:
                for alpha in ALPHAS:
                    preds = replay_policy(cal_scores, test_scores, mode, alpha,
                                          window_size, args.aci_gamma)
                    cov = pl.coverage_and_accuracy(split['y_test'], preds, alpha)
                    row = {
                        'detector': detector_type,
                        'dataset': args.dataset,
                        'smoothness_weight': sw,
                        'cp_mode': mode,
                        'aci_gamma': args.aci_gamma if mode == 'aci' else None,
                        'alpha': alpha,
                        'expected_coverage': 1 - alpha,
                        'coverage': cov['coverage'],
                        'calibration_error': cov['calibration_error'],
                        'accuracy': cov['accuracy'],
                        'f1': f1_score(split['y_test'], preds, zero_division=0),
                        'precision': precision_score(split['y_test'], preds, zero_division=0),
                        'recall': recall_score(split['y_test'], preds, zero_division=0),
                        'roc_auc_scores': roc,
                        'pr_auc_scores': pr,
                    }
                    all_rows.append(row)
                    print(f"    sw={sw} mode={mode:7s} alpha={alpha:.2f}: "
                          f"cov={row['coverage']:.3f} (err={row['calibration_error']:.3f}) "
                          f"F1={row['f1']:.3f} P={row['precision']:.3f} R={row['recall']:.3f}")

    df = pd.DataFrame(all_rows)
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = args.detector if args.detector else 'all'
    df['seed'] = args.seed
    out = os.path.join(args.output_dir, f'exp6_cp_ablation_{args.dataset}_{tag}_s{args.seed}_{timestamp}.csv')
    df.to_csv(out, index=False)
    print(f"\nResults saved to: {out}")
    return df


def main():
    parser = argparse.ArgumentParser(description='Experiment 6: CP mode / TVD ablation')
    parser.add_argument('--dataset', type=str, default='skab', choices=['skab', 'swat'])
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--test_size', type=float, default=0.2)
    parser.add_argument('--val_size', type=float, default=0.2)
    parser.add_argument('--detector', type=str, default=None,
                        choices=['conv_ae', 'lstm_ae', 'lstm_vae'])
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--latent_dim', type=int, default=32)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--kl_weight', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--aci_gamma', type=float, default=0.005)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_dir', type=str, default='./results/')
    parser.add_argument('--cache_dir', type=str, default='./results/cache/')
    parser.add_argument('--force_retrain', action='store_true')
    args = parser.parse_args()

    run_experiment(args)


if __name__ == '__main__':
    main()
