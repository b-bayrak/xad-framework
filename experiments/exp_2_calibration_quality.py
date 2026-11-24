#!/usr/bin/env python
"""
    Experiment 2: Calibration Quality Across Confidence Levels

    Evaluates calibration performance of the temporal-aware conformal
    predictor across significance levels (alpha in {0.05, 0.10, 0.15, 0.20}).

    Metrics reported:
        - Expected Coverage (1 - alpha)
        - Observed Coverage (true-normal specificity, see pipeline.py)
        - Calibration Error |observed - expected|

    Uses the SAME cached detector as exp_1 for a given (dataset, detector) -
    only the conformal calibration step (cheap) is repeated per alpha, so
    its alpha=0.05 results agree with Experiment 1.

    Usage:
        python experiments/exp_2_calibration_quality.py --dataset skab --data_path ./data/SKAB/
"""

import os
import sys
import argparse
from datetime import datetime

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments import pipeline as pl


def run_experiment(args):
    print("\n" + "-" * 70)
    print("EXPERIMENT 2: Calibration Quality Across Confidence Levels")
    print("-" * 70)

    print(f"\nLoading {args.dataset.upper()} dataset from {args.data_path}")
    split = pl.prepare_split(args.dataset, args.data_path, args.test_size, args.val_size)

    config = pl.default_config(args)
    alpha_values = [0.05, 0.10, 0.15, 0.20]
    all_results = []

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

        for alpha in alpha_values:
            cp = pl.make_conformal_predictor(
                detector, split['X_val_normal'], args.dataset,
                alpha=alpha, adaptive=not args.no_adaptive,
                cp_mode=args.cp_mode, aci_gamma=args.aci_gamma, verbose=False,
            )
            predictions, scores, _ = pl.predict_all(cp, detector, split['X_test'])
            cov = pl.coverage_and_accuracy(split['y_test'], predictions, alpha)

            all_results.append({
                'detector': detector_type,
                'dataset': args.dataset,
                'alpha': alpha,
                'expected_coverage': cov['expected_coverage'],
                'observed_coverage': cov['coverage'],
                'accuracy': cov['accuracy'],
                'calibration_error': cov['calibration_error'],
            })
            print(f"  alpha={alpha:.2f}: Expected={1-alpha:.3f}, "
                  f"Observed coverage={cov['coverage']:.3f}, "
                  f"Error={cov['calibration_error']:.3f}  "
                  f"(accuracy={cov['accuracy']:.3f})")

    df = pd.DataFrame(all_results)

    print("\n" + "-" * 70)
    print("SUMMARY: Calibration Performance")
    print("-" * 70)
    for detector_type in df['detector'].unique():
        print(f"\n{detector_type.upper()}:")
        for _, row in df[df['detector'] == detector_type].iterrows():
            print(f"  alpha={row['alpha']:.2f}: 1-alpha={row['expected_coverage']:.3f}, "
                  f"Obs.={row['observed_coverage']:.3f}, Error={row['calibration_error']:.3f}")

    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # See exp_1's identical comment: detector tag avoids filename collisions
    # between concurrent per-detector SLURM array tasks sharing --output_dir.
    detector_tag = getattr(args, 'detector', None) or 'all'
    df['seed'] = args.seed
    df['cp_mode'] = args.cp_mode or ('static' if args.no_adaptive else 'sliding')
    csv_path = os.path.join(args.output_dir, f'exp2_{args.dataset}_{detector_tag}_s{args.seed}_{timestamp}.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")
    return df


def parse_args():
    parser = argparse.ArgumentParser(description='Experiment 2: Calibration Quality')

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
