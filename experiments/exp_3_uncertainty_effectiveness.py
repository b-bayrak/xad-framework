#!/usr/bin/env python
"""
    Experiment 3: Uncertainty Effectiveness

    Evaluates whether confidence estimates are informative for prediction
    quality: precision/accuracy stratified by confidence tertile (low/medium/high).

    Metrics reported: precision by confidence level,
    accuracy by confidence level.

    Uses the SAME cached detector as exp_1/exp_2 for a given (dataset, detector).

    Usage:
        python experiments/exp_3_uncertainty_effectiveness.py --dataset skab --data_path ./data/SKAB/
"""

import os
import sys
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments import pipeline as pl


def stratify_by_confidence(confidences: np.ndarray, n_bins: int = 3):
    quantiles = np.quantile(confidences, [1 / 3, 2 / 3])
    return np.digitize(confidences, quantiles)


def compute_reliability_by_confidence(predictions, y_true, confidences):
    bins = stratify_by_confidence(confidences)
    bin_labels = ['low', 'medium', 'high']
    results = {}

    for bin_idx, label in enumerate(bin_labels):
        mask = bins == bin_idx
        if mask.sum() == 0:
            results[label] = {'n_samples': 0, 'normal_accuracy': None, 'anomaly_precision': None}
            continue

        bin_predictions = predictions[mask]
        bin_true = y_true[mask]

        normal_mask = bin_predictions == 0
        normal_accuracy = float((bin_true[normal_mask] == 0).mean()) if normal_mask.sum() > 0 else None

        anomaly_mask = bin_predictions == 1
        anomaly_precision = float((bin_true[anomaly_mask] == 1).mean()) if anomaly_mask.sum() > 0 else None

        results[label] = {
            'n_samples': int(mask.sum()),
            'mean_confidence': float(confidences[mask].mean()),
            'normal_accuracy': normal_accuracy,
            'anomaly_precision': anomaly_precision,
            'overall_accuracy': float((bin_predictions == bin_true).mean()),
        }

    return results


def run_experiment(args):
    print("\n" + "-" * 70)
    print("EXPERIMENT 3: Uncertainty Effectiveness")
    print("-" * 70)

    print(f"\nLoading {args.dataset.upper()} dataset from {args.data_path}")
    split = pl.prepare_split(args.dataset, args.data_path, args.test_size, args.val_size)

    config = pl.default_config(args)
    all_results = []
    detailed_results = []

    detectors = [args.detector] if getattr(args, 'detector', None) else pl.DETECTORS
    for detector_type in detectors:
        print(f"\n{'-'*60}\n{detector_type.upper()}\n{'-'*60}")

        detector, _ = pl.get_or_train_detector(
            detector_type, args.dataset,
            split['X_train_normal'], split['X_val_normal'],
            split['input_dim'], split['sequence_length'],
            config, cache_dir=args.cache_dir, force_retrain=args.force_retrain,
            seed=args.seed, verbose=(detector_type == detectors[0]),
        )
        cp = pl.make_conformal_predictor(
            detector, split['X_val_normal'], args.dataset,
            alpha=args.alpha, adaptive=not args.no_adaptive,
            cp_mode=args.cp_mode, aci_gamma=args.aci_gamma, verbose=False,
        )
        predictions, scores, confidences = pl.predict_all(cp, detector, split['X_test'])
        y_test = split['y_test']

        for idx in range(len(y_test)):
            detailed_results.append({
                'detector': detector_type, 'sample_id': idx,
                'prediction': 'anomaly' if predictions[idx] else 'normal',
                'true_label': int(y_test[idx]), 'confidence': float(confidences[idx]),
                'score': float(scores[idx]),
                'correct': bool(predictions[idx] == y_test[idx]),
            })

        reliability = compute_reliability_by_confidence(predictions, y_test, confidences)
        correct = (predictions == y_test).astype(int)
        correlation, p_value = stats.pearsonr(confidences, correct)

        print("\nReliability by Confidence Level:")
        for level in ['low', 'medium', 'high']:
            r = reliability[level]
            if r['n_samples'] > 0:
                na = f"{r['normal_accuracy']:.3f}" if r['normal_accuracy'] is not None else "N/A"
                ap = f"{r['anomaly_precision']:.3f}" if r['anomaly_precision'] is not None else "N/A"
                print(f"  {level.upper():8s}: n={r['n_samples']:4d}, Normal Acc={na:>6}, Anomaly Prec={ap:>6}")

        print(f"\nCorrelation (confidence vs correctness): rho={correlation:.4f}, p={p_value:.6f}")

        for level in ['low', 'medium', 'high']:
            r = reliability[level]
            all_results.append({
                'detector': detector_type, 'dataset': args.dataset, 'confidence_level': level,
                'n_samples': r['n_samples'], 'normal_accuracy': r['normal_accuracy'],
                'anomaly_precision': r['anomaly_precision'], 'overall_accuracy': r.get('overall_accuracy'),
                'correlation': correlation, 'p_value': p_value,
            })

    df = pd.DataFrame(all_results)
    df_detailed = pd.DataFrame(detailed_results)

    print("\n" + "-" * 70)
    print("SUMMARY: Anomaly Prediction Precision by Confidence")
    print("-" * 70)
    pivot_anomaly = df.pivot_table(index='detector', columns='confidence_level',
                                    values='anomaly_precision', aggfunc='mean')
    cols = [c for c in ['low', 'medium', 'high'] if c in pivot_anomaly.columns]
    print(f"\nDataset: {args.dataset.upper()}")
    print(pivot_anomaly[cols].to_string())

    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # See exp_1's identical comment: detector tag avoids filename collisions
    # between concurrent per-detector SLURM array tasks sharing --output_dir.
    detector_tag = getattr(args, 'detector', None) or 'all'
    df['seed'] = args.seed
    df['cp_mode'] = args.cp_mode or ('static' if args.no_adaptive else 'sliding')
    df_detailed['seed'] = args.seed
    csv_path = os.path.join(args.output_dir, f'exp3_{args.dataset}_{detector_tag}_s{args.seed}_{timestamp}.csv')
    df.to_csv(csv_path, index=False)
    df_detailed.to_csv(os.path.join(args.output_dir, f'exp3_{args.dataset}_{detector_tag}_s{args.seed}_{timestamp}_detailed.csv'), index=False)
    print(f"\nResults saved to: {csv_path}")
    return df


def parse_args():
    parser = argparse.ArgumentParser(description='Experiment 3: Uncertainty Effectiveness')

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
    parser.add_argument('--no_adaptive', action='store_true')
    parser.add_argument('--cp_mode', type=str, default=None,
                        choices=['static', 'sliding', 'aci'],
                        help='Conformal threshold policy; default: sliding '
                             '(or static if --no_adaptive is set)')
    parser.add_argument('--aci_gamma', type=float, default=0.005,
                        help='ACI step size gamma (only used with --cp_mode aci)')
    parser.add_argument('--output_dir', type=str, default='./results/')
    parser.add_argument('--cache_dir', type=str, default='./results/cache/')
    parser.add_argument('--force_retrain', action='store_true')
    parser.add_argument('--seed', type=int, default=42)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    run_experiment(args)
