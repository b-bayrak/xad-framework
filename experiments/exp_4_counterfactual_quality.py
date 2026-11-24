#!/usr/bin/env python
"""
    Experiment 4: Counterfactual Explanation Quality

    Evaluates counterfactual quality: Validity, Proximity
    (L2), Sparsity, for GA-CE (Section 3.5.1 / Algorithm 1), DICE (Appendix
    B.1), and PerCE (Appendix B.2).

    Uses the SAME cached detector as exp_1-3 for a given (dataset, detector).
    The normal reference set for k-NN/DTW initialization is the normal
    subset of validation, matching what the conformal predictor was
    calibrated on.

    Usage:
        python experiments/exp_4_counterfactual_quality.py --dataset skab --data_path ./data/SKAB/
        python experiments/exp_4_counterfactual_quality.py --dataset skab --methods ga_ce dice perce
"""

import os
import sys
import argparse
import time
from datetime import datetime

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments import pipeline as pl
from xad.counterfactual_explainer import CounterfactualExplainer
from xad.dice_explainer import DICEExplainer
from xad.perce_explainer import PerCEExplainer


METHOD_NAMES = {'ga_ce': 'GA-CE', 'dice': 'DICE', 'perce': 'PerCE'}


def make_explainer(method: str, detector, conformal, args):
    if method == 'ga_ce':
        return CounterfactualExplainer(
            model=detector, conformal_predictor=conformal,
            n_generations=args.n_generations, population_size=args.population_size,
        )
    elif method == 'dice':
        return DICEExplainer(model=detector, conformal_predictor=conformal)
    elif method == 'perce':
        return PerCEExplainer(model=detector, conformal_predictor=conformal)
    else:
        raise ValueError(f"Unknown method: {method}")


def evaluate_counterfactuals(explainer, X_anomalies, X_normal_ref, verbose=True,
                             conformal=None, detector=None):
    """
    If `conformal` and `detector` are given, each per-sample detail row also
    records the original and counterfactual nonconformity scores and the
    threshold in force at CF-check time (plus ACI's alpha_t where relevant).
    This is what makes post-hoc cross-policy validity analysis possible
    ("would this same counterfactual be valid under the other policy's
    threshold?") without regenerating counterfactuals.
    """
    explainer.fit(X_normal_ref)

    valid_count, total_proximity, total_sparsity = 0, 0.0, 0.0
    generation_times = []
    results = []

    iterator = range(len(X_anomalies))
    if verbose:
        iterator = tqdm(iterator, desc="Generating counterfactuals")

    for i in iterator:
        x_anomaly = X_anomalies[i]
        start_time = time.time()
        try:
            cf_result = explainer.generate_counterfactual(x_anomaly, verbose=False)
            gen_time = time.time() - start_time

            is_valid = cf_result['validity']
            proximity = cf_result['proximity']
            sparsity = cf_result['sparsity']

            valid_count += int(is_valid)
            total_proximity += proximity
            total_sparsity += sparsity
            generation_times.append(gen_time)

            row = {'sample_id': i, 'valid': is_valid, 'proximity': proximity,
                   'sparsity': sparsity, 'generation_time': gen_time}
            if conformal is not None and detector is not None:
                cf_arr = cf_result.get('counterfactual')
                row['orig_score'] = conformal._compute_nonconformity(x_anomaly, detector)
                row['cf_score'] = (conformal._compute_nonconformity(cf_arr, detector)
                                   if cf_arr is not None else None)
                row['threshold_at_check'] = conformal.threshold
                row['cp_mode'] = conformal.mode
                row['alpha_t'] = getattr(conformal, 'alpha_t', None)
            results.append(row)
        except Exception as e:
            if verbose:
                print(f"\nFailed for sample {i}: {e}")
            results.append({'sample_id': i, 'valid': False, 'proximity': None,
                             'sparsity': None, 'error': str(e)})

    n_successful = len([r for r in results if r.get('proximity') is not None])
    return {
        'n_anomalies': len(X_anomalies),
        'n_successful': n_successful,
        'validity_rate': valid_count / n_successful if n_successful > 0 else 0,
        'mean_proximity': total_proximity / n_successful if n_successful > 0 else 0,
        'mean_sparsity': total_sparsity / n_successful if n_successful > 0 else 0,
        'mean_gen_time': float(np.mean(generation_times)) if generation_times else 0,
        'detailed_results': results,
    }


def run_experiment(args):
    print("\n" + "-" * 70)
    print("EXPERIMENT 4: Counterfactual Explanation Quality")
    print("-" * 70)

    print(f"\nLoading {args.dataset.upper()} dataset from {args.data_path}")
    split = pl.prepare_split(args.dataset, args.data_path, args.test_size, args.val_size)

    config = pl.default_config(args)
    all_results = []
    all_details = []

    detectors = [args.detector] if getattr(args, 'detector', None) else pl.DETECTORS
    for detector_type in detectors:
        print(f"\n{'-'*60}\nEvaluating {detector_type.upper()}\n{'-'*60}")

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

        print("\nIdentifying anomalies in test set...")
        predictions, scores, _ = pl.predict_all(cp, detector, split['X_test'])
        detected_indices = np.where(predictions == 1)[0]
        detected_anomalies = split['X_test'][detected_indices]
        print(f"Detected {len(detected_anomalies)} anomalies")

        if len(detected_anomalies) == 0:
            for method in args.methods:
                all_results.append({'detector': detector_type, 'dataset': args.dataset, 'method': METHOD_NAMES[method],
                                     'n_anomalies': 0, 'validity_rate': None, 'mean_proximity': None, 'mean_sparsity': None})
            continue

        if len(detected_anomalies) > args.max_anomalies:
            rng = np.random.RandomState(args.seed)
            idx = rng.choice(len(detected_anomalies), args.max_anomalies, replace=False)
            detected_anomalies = detected_anomalies[idx]

        # Same anomaly subset reused across methods for a fair comparison
        # (avoids "method A got an easier random sample than method B").
        for method in args.methods:
            print(f"\nGenerating {METHOD_NAMES[method]} counterfactuals for {len(detected_anomalies)} anomalies...")
            explainer = make_explainer(method, detector, cp, args)
            cf_metrics = evaluate_counterfactuals(
                explainer=explainer, X_anomalies=detected_anomalies,
                X_normal_ref=split['X_val_normal'], verbose=True,
                conformal=cp, detector=detector,
            )
            for r in cf_metrics['detailed_results']:
                r.update({'detector': detector_type, 'method': METHOD_NAMES[method]})
                all_details.append(r)

            print(f"\n{METHOD_NAMES[method]} quality for {detector_type}:")
            print(f"  Validity Rate:   {cf_metrics['validity_rate']:.2%}")
            print(f"  Mean Proximity:  {cf_metrics['mean_proximity']:.6f}")
            print(f"  Mean Sparsity:   {cf_metrics['mean_sparsity']:.2%}")
            print(f"  Avg Gen Time:    {cf_metrics['mean_gen_time']:.2f}s")

            all_results.append({
                'detector': detector_type, 'dataset': args.dataset, 'method': METHOD_NAMES[method],
                'n_anomalies': cf_metrics['n_anomalies'], 'n_successful': cf_metrics['n_successful'],
                'validity_rate': cf_metrics['validity_rate'], 'mean_proximity': cf_metrics['mean_proximity'],
                'mean_sparsity': cf_metrics['mean_sparsity'], 'mean_gen_time': cf_metrics['mean_gen_time'],
            })

    df = pd.DataFrame(all_results)

    print("\n" + "-" * 70)
    print("SUMMARY: Counterfactual Explanation Quality")
    print("-" * 70)
    print(f"\nDataset: {args.dataset.upper()}")
    print("\n" + df[['detector', 'method', 'validity_rate', 'mean_proximity', 'mean_sparsity']].to_string(index=False))

    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # See exp_1's identical comment: detector tag avoids filename collisions
    # between concurrent per-detector SLURM array tasks sharing --output_dir.
    detector_tag = getattr(args, 'detector', None) or 'all'
    df['seed'] = args.seed
    cp_mode_tag = args.cp_mode or ('static' if args.no_adaptive else 'sliding')
    df['cp_mode'] = cp_mode_tag
    csv_path = os.path.join(args.output_dir, f'exp4_{args.dataset}_{detector_tag}_{cp_mode_tag}_s{args.seed}_{timestamp}.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")

    if all_details:
        df_det = pd.DataFrame(all_details)
        df_det['seed'] = args.seed
        df_det['dataset'] = args.dataset
        det_path = os.path.join(
            args.output_dir,
            f'exp4_{args.dataset}_{detector_tag}_{cp_mode_tag}_s{args.seed}_{timestamp}_detailed.csv')
        df_det.to_csv(det_path, index=False)
        print(f"Per-anomaly details saved to: {det_path}")
    return df


def parse_args():
    parser = argparse.ArgumentParser(description='Experiment 4: Counterfactual Quality')

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

    parser.add_argument('--methods', type=str, nargs='+', default=['ga_ce', 'dice', 'perce'],
                        choices=['ga_ce', 'dice', 'perce'], help='Which counterfactual methods to evaluate')
    parser.add_argument('--n_generations', type=int, default=100, help='GA-CE only')
    parser.add_argument('--population_size', type=int, default=50, help='GA-CE only')
    parser.add_argument('--max_anomalies', type=int, default=100)

    parser.add_argument('--output_dir', type=str, default='./results/')
    parser.add_argument('--cache_dir', type=str, default='./results/cache/')
    parser.add_argument('--force_retrain', action='store_true')
    parser.add_argument('--seed', type=int, default=42)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    np.random.seed(args.seed)
    run_experiment(args)
