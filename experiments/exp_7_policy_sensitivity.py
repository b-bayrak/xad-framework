#!/usr/bin/env python
"""
    Experiment 7: sensitivity of the two adaptive threshold policies to
    their single hyperparameter.

    - ACI: step size gamma in {0.001, 0.002, 0.005, 0.01, 0.02, 0.05}
    - Sliding window: window size w in {100, 250, 500, 1000, 2000}

    Like exp_6, the calibration and test streams are scored ONCE per
    (dataset, detector, seed) and every hyperparameter cell replays the same
    chronological score sequence through the real TemporalConformalPredictor
    logic, so differences are attributable to the hyperparameter alone.
    Reported at alpha in {0.05, 0.20}: coverage, calibration error, F1,
    recall. Deployed nonconformity configuration (lambda_temp per
    DATASET_CONFIG).

    Usage:
        python experiments/exp_7_policy_sensitivity.py --dataset skab --data_path ./data/SKAB/ --detector lstm_ae
"""

import os
import sys
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, recall_score, precision_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments import pipeline as pl
from experiments.exp_6_cp_ablation import score_stream, replay_policy

GAMMAS = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05]
WINDOWS = [100, 250, 500, 1000, 2000]
ALPHAS = [0.05, 0.20]


def run_experiment(args):
    print("\n" + "-" * 70)
    print("EXPERIMENT 7: Policy Hyperparameter Sensitivity (gamma / window size)")
    print("-" * 70)

    split = pl.prepare_split(args.dataset, args.data_path, args.test_size, args.val_size)
    config = pl.default_config(args)
    sw = pl.DATASET_CONFIG[args.dataset]['smoothness_weight']

    rows = []
    detectors = [args.detector] if getattr(args, 'detector', None) else pl.DETECTORS
    for detector_type in detectors:
        detector, _ = pl.get_or_train_detector(
            detector_type, args.dataset,
            split['X_train_normal'], split['X_val_normal'],
            split['input_dim'], split['sequence_length'],
            config, cache_dir=args.cache_dir, force_retrain=args.force_retrain,
            seed=args.seed, verbose=True,
        )
        print(f"  scoring streams ({detector_type}, lambda_temp={sw}) ...")
        cal_scores = score_stream(detector, split['X_val_normal'], sw)
        test_scores = score_stream(detector, split['X_test'], sw)

        def record(mode, param_name, param_value, preds, alpha):
            cov = pl.coverage_and_accuracy(split['y_test'], preds, alpha)
            rows.append({
                'detector': detector_type, 'dataset': args.dataset,
                'cp_mode': mode, 'param': param_name, 'value': param_value,
                'alpha': alpha, 'coverage': cov['coverage'],
                'calibration_error': cov['calibration_error'],
                'accuracy': cov['accuracy'],
                'f1': f1_score(split['y_test'], preds, zero_division=0),
                'precision': precision_score(split['y_test'], preds, zero_division=0),
                'recall': recall_score(split['y_test'], preds, zero_division=0),
                'seed': args.seed,
            })
            r = rows[-1]
            print(f"    {mode} {param_name}={param_value:<7} alpha={alpha:.2f}: "
                  f"cov={r['coverage']:.3f} err={r['calibration_error']:.3f} "
                  f"F1={r['f1']:.3f} R={r['recall']:.3f}")

        for alpha in ALPHAS:
            for g in GAMMAS:
                preds = replay_policy(cal_scores, test_scores, 'aci', alpha,
                                      pl.DATASET_CONFIG[args.dataset]['calibration_window'], g)
                record('aci', 'gamma', g, preds, alpha)
            for w in WINDOWS:
                preds = replay_policy(cal_scores, test_scores, 'sliding', alpha, w, 0.005)
                record('sliding', 'window', w, preds, alpha)

    df = pd.DataFrame(rows)
    os.makedirs(args.output_dir, exist_ok=True)
    tag = args.detector if args.detector else 'all'
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = os.path.join(args.output_dir, f'exp7_sensitivity_{args.dataset}_{tag}_s{args.seed}_{ts}.csv')
    df.to_csv(out, index=False)
    print(f"\nResults saved to: {out}")
    return df


def main():
    parser = argparse.ArgumentParser(description='Experiment 7: policy hyperparameter sensitivity')
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
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_dir', type=str, default='./results/')
    parser.add_argument('--cache_dir', type=str, default='./results/cache/')
    parser.add_argument('--force_retrain', action='store_true')
    args = parser.parse_args()
    run_experiment(args)


if __name__ == '__main__':
    main()
