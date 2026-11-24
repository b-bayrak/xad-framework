#!/usr/bin/env python
"""
    Experiment 1: Detection Performance Across Detector Architectures

    Evaluates the framework's applicability across different detector
    architectures (Conv-AE, LSTM-AE, LSTM-VAE) on SKAB and SWaT.

    Metrics reported:
        - F1-Score (optimal threshold), ROC-AUC, PR-AUC
        - Coverage (true-normal specificity - NOT accuracy, see pipeline.py)
        - Calibration Error at the default alpha

    Trains (or loads a cached copy of) each detector via
    experiments/pipeline.py, so these numbers come from the exact same
    trained model used by exp_2/exp_3/exp_4 for the same (dataset, detector).

    Usage:
        python experiments/exp_1_detection_performance.py --dataset skab --data_path ./data/SKAB/
        python experiments/exp_1_detection_performance.py --dataset swat --data_path ./data/SWaT/
"""

import os
import sys
import argparse
import json
import time
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments import pipeline as pl


def compute_optimal_f1(y_true, scores):
    precisions, recalls, thresholds = precision_recall_curve(y_true, scores)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    best_idx = np.argmax(f1_scores[:-1])
    return {
        'f1': f1_scores[best_idx],
        'threshold': thresholds[best_idx],
        'precision': precisions[best_idx],
        'recall': recalls[best_idx]
    }


def run_experiment(args):
    print("\n" + "-" * 70)
    print("EXPERIMENT 1: Detection Performance Across Architectures")
    print("-" * 70)

    print(f"\nLoading {args.dataset.upper()} dataset from {args.data_path}")
    split = pl.prepare_split(args.dataset, args.data_path, args.test_size, args.val_size)

    config = pl.default_config(args)
    all_results = []

    detectors = [args.detector] if getattr(args, 'detector', None) else pl.DETECTORS
    for detector_type in detectors:
        print(f"\n{'-'*60}\nEvaluating {detector_type.upper()}\n{'-'*60}")

        start_time = time.time()
        detector, history = pl.get_or_train_detector(
            detector_type, args.dataset,
            split['X_train_normal'], split['X_val_normal'],
            split['input_dim'], split['sequence_length'],
            config, cache_dir=args.cache_dir, force_retrain=args.force_retrain,
            seed=args.seed, verbose=True,
        )
        training_time = time.time() - start_time

        cp = pl.make_conformal_predictor(
            detector, split['X_val_normal'], args.dataset,
            alpha=args.alpha, adaptive=not args.no_adaptive,
            cp_mode=args.cp_mode, aci_gamma=args.aci_gamma, verbose=True,
        )

        predictions, scores, confidences = pl.predict_all(cp, detector, split['X_test'])
        optimal_f1 = compute_optimal_f1(split['y_test'], scores)
        cov = pl.coverage_and_accuracy(split['y_test'], predictions, args.alpha)

        metrics = {
            'detector': detector_type,
            'f1_optimal': optimal_f1['f1'],
            'precision': optimal_f1['precision'],
            'recall': optimal_f1['recall'],
            'roc_auc': roc_auc_score(split['y_test'], scores),
            'pr_auc': average_precision_score(split['y_test'], scores),
            'accuracy': cov['accuracy'],
            'coverage': cov['coverage'],
            'calibration_error': cov['calibration_error'],
            'training_time': training_time,
            'final_train_loss': history['train_loss'][-1] if history and history.get('train_loss') else None,
            'final_val_loss': history['val_loss'][-1] if history and history.get('val_loss') else None,
        }
        print(f"\nResults for {detector_type}:")
        print(f"  F1 (optimal):     {metrics['f1_optimal']:.4f}")
        print(f"  ROC-AUC:          {metrics['roc_auc']:.4f}")
        print(f"  PR-AUC:           {metrics['pr_auc']:.4f}")
        print(f"  Accuracy:         {metrics['accuracy']:.4f}  (both classes)")
        print(f"  Coverage:         {metrics['coverage']:.4f}  (true-normal specificity)")
        print(f"  Calibration Err:  {metrics['calibration_error']:.4f}")

        metrics['dataset'] = args.dataset
        all_results.append(metrics)

    df = pd.DataFrame(all_results)

    print("\n" + "-" * 70)
    print("SUMMARY: Detection Performance")
    print("-" * 70)
    print(f"\nDataset: {args.dataset.upper()}  Alpha: {args.alpha}")
    print("\n" + df[['detector', 'f1_optimal', 'roc_auc', 'pr_auc',
                     'accuracy', 'coverage', 'calibration_error']].to_string(index=False))

    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Detector tag in the filename is load-bearing, not cosmetic: when
    # several SLURM array tasks (one per detector) share one --output_dir
    # and finish within the same second, a timestamp-only filename collides
    # and one detector's CSV silently overwrites another's.
    detector_tag = getattr(args, 'detector', None) or 'all'
    df['seed'] = args.seed
    df['cp_mode'] = args.cp_mode or ('static' if args.no_adaptive else 'sliding')
    csv_path = os.path.join(args.output_dir, f'exp1_{args.dataset}_{detector_tag}_s{args.seed}_{timestamp}.csv')
    df.to_csv(csv_path, index=False)

    config_path = os.path.join(args.output_dir, f'exp1_{args.dataset}_{timestamp}_config.json')
    with open(config_path, 'w') as f:
        json.dump({'config': config, 'args': vars(args), 'timestamp': timestamp}, f, indent=2)

    print(f"\nResults saved to: {csv_path}")
    return df


def parse_args():
    parser = argparse.ArgumentParser(description='Experiment 1: Detection Performance')

    parser.add_argument('--dataset', type=str, default='skab', choices=['skab', 'swat'])
    parser.add_argument('--data_path', type=str, default='./data/SKAB/')
    parser.add_argument('--test_size', type=float, default=0.2)
    parser.add_argument('--val_size', type=float, default=0.2)

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

    parser.add_argument('--alpha', type=float, default=0.05)
    parser.add_argument('--detector', type=str, default=None, choices=pl.DETECTORS,
                        help='Restrict to a single detector (for parallel array jobs); default: all three')
    parser.add_argument('--no_adaptive', action='store_true',
                        help='Disable sliding-window recalibration (Eq. 8); use static split-CP instead')
    parser.add_argument('--cp_mode', type=str, default=None,
                        choices=['static', 'sliding', 'aci'],
                        help='Conformal threshold policy; default: sliding '
                             '(or static if --no_adaptive is set)')
    parser.add_argument('--aci_gamma', type=float, default=0.005,
                        help='ACI step size gamma (only used with --cp_mode aci)')

    parser.add_argument('--output_dir', type=str, default='./results/')
    parser.add_argument('--cache_dir', type=str, default='./results/cache/')
    parser.add_argument('--force_retrain', action='store_true',
                        help='Ignore any cached checkpoint and retrain from scratch')
    parser.add_argument('--seed', type=int, default=42)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    run_experiment(args)
