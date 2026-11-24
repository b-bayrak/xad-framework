#!/usr/bin/env python
"""
    Experiment 5: GA-CE Loss Term Ablation (Appendix)

    Ablates each term of the GA-CE objective (Eq. 17, Eq. 22-23) individually:
    Full GA-CE, No L_smooth, No L_spar, No L_prox. Reports Validity,
    Proximity, Sparsity, and Smoothness for each variant.

    This reuses experiments/exp_4_counterfactual_quality.py's
    evaluate_counterfactuals() on the exact same cached detector, calibration,
    and detected-anomaly sample as Experiment 4, so the ablation is directly
    comparable to the GA-CE results of Experiment 4 for the same
    (dataset, detector).

    Usage:
        python experiments/exp_5_gace_ablation.py --dataset skab --detector conv_ae --data_path ./data/SKAB/
"""

import os
import sys
import argparse
import json
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments import pipeline as pl
from experiments.exp_4_counterfactual_quality import evaluate_counterfactuals


def mean_smoothness(detected_anomalies, cf_arrays):
    """Smoothness diagnostic: mean squared temporal-dynamics mismatch (Eq. 22),
    lower is more temporally coherent. Reported alongside validity/proximity/
    sparsity since that's the trade-off the ablation is meant to expose."""
    vals = []
    for x, x_cf in zip(detected_anomalies, cf_arrays):
        if x_cf is None:
            continue
        true_diff = x[1:] - x[:-1]
        cf_diff = x_cf[1:] - x_cf[:-1]
        vals.append(float(np.mean((cf_diff - true_diff) ** 2)))
    return float(np.mean(vals)) if vals else float('nan')


VARIANTS = {
    'Full GA-CE': dict(use_proximity=True, use_sparsity=True, use_smooth=True),
    'No L_smooth': dict(use_proximity=True, use_sparsity=True, use_smooth=False),
    'No L_spar': dict(use_proximity=True, use_sparsity=False, use_smooth=True),
    'No L_prox': dict(use_proximity=False, use_sparsity=True, use_smooth=True),
}


def run_experiment(args):
    print("\n" + "-" * 70)
    print("EXPERIMENT 5: GA-CE Loss Term Ablation")
    print("-" * 70)

    split = pl.prepare_split(args.dataset, args.data_path, args.test_size, args.val_size)
    config = pl.default_config(args)

    detector, _ = pl.get_or_train_detector(
        args.detector, args.dataset,
        split['X_train_normal'], split['X_val_normal'],
        split['input_dim'], split['sequence_length'],
        config, cache_dir=args.cache_dir, force_retrain=args.force_retrain,
        seed=args.seed, verbose=True,
    )
    cp = pl.make_conformal_predictor(
        detector, split['X_val_normal'], args.dataset,
        alpha=args.alpha, adaptive=not args.no_adaptive,
        cp_mode=args.cp_mode, aci_gamma=args.aci_gamma, verbose=False,
    )

    predictions, scores, _ = pl.predict_all(cp, detector, split['X_test'])
    detected_indices = np.where(predictions == 1)[0]
    detected_anomalies = split['X_test'][detected_indices]
    print(f"Detected {len(detected_anomalies)} anomalies on {args.dataset}/{args.detector}")

    if len(detected_anomalies) > args.max_anomalies:
        rng = np.random.RandomState(args.seed)
        idx = rng.choice(len(detected_anomalies), args.max_anomalies, replace=False)
        detected_anomalies = detected_anomalies[idx]
    n_used = len(detected_anomalies)
    print(f"Using n={n_used} anomalies for the ablation (same subset across all variants)")

    from xad.counterfactual_explainer import CounterfactualExplainer

    rows = []
    for variant_name, flags in VARIANTS.items():
        print(f"\nRunning variant: {variant_name} ({flags})")
        variant_explainer = CounterfactualExplainer(
            model=detector, conformal_predictor=cp,
            n_generations=args.n_generations, population_size=args.population_size,
            **flags,
        )
        cf_metrics = evaluate_counterfactuals(
            explainer=variant_explainer, X_anomalies=detected_anomalies,
            X_normal_ref=split['X_val_normal'], verbose=True,
        )
        # Smoothness is a diagnostic only reported when --with_smoothness is
        # passed, since it requires regenerating the counterfactual arrays
        # (evaluate_counterfactuals only returns aggregate metrics), roughly
        # doubling the cost of this script.
        smoothness = float('nan')
        if args.with_smoothness:
            from xad.counterfactual_explainer import CounterfactualExplainer
            explainer = CounterfactualExplainer(
                model=detector, conformal_predictor=cp,
                n_generations=args.n_generations, population_size=args.population_size, **flags,
            )
            explainer.fit(split['X_val_normal'])
            cfs = [explainer.generate_counterfactual(x)['counterfactual'] for x in detected_anomalies]
            smoothness = mean_smoothness(detected_anomalies, cfs)

        rows.append({
            'variant': variant_name,
            'dataset': args.dataset,
            'detector': args.detector,
            'n_anomalies': n_used,
            'validity': cf_metrics['validity_rate'],
            'proximity': cf_metrics['mean_proximity'],
            'sparsity': cf_metrics['mean_sparsity'],
            'smoothness': smoothness,
        })
        print(f"  -> validity={cf_metrics['validity_rate']:.3f}, "
              f"proximity={cf_metrics['mean_proximity']:.3f}, "
              f"sparsity={cf_metrics['mean_sparsity']:.3f}, smoothness={smoothness}")

    df = pd.DataFrame(rows)

    print("\n" + "-" * 70)
    print(f"SUMMARY: GA-CE Ablation ({args.dataset}/{args.detector}, n={n_used})")
    print("-" * 70)
    print(df.to_string(index=False))
    print("\nNOTE: This table is only valid for the exact (dataset, detector, n_anomalies) "
          "printed above. Do not present it as a general multi-architecture/multi-dataset "
          "finding - re-run per (dataset, detector) if you need broader coverage.")

    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    df['seed'] = args.seed
    df['cp_mode'] = args.cp_mode or ('static' if args.no_adaptive else 'sliding')
    csv_path = os.path.join(args.output_dir, f'exp5_gace_ablation_{args.dataset}_{args.detector}_s{args.seed}_{timestamp}.csv')
    df.to_csv(csv_path, index=False)
    with open(csv_path.replace('.csv', '_meta.json'), 'w') as f:
        json.dump({'dataset': args.dataset, 'detector': args.detector, 'n_anomalies': n_used,
                    'alpha': args.alpha, 'n_generations': args.n_generations,
                    'population_size': args.population_size, 'seed': args.seed}, f, indent=2)
    print(f"\nResults saved to: {csv_path}")
    return df


def parse_args():
    parser = argparse.ArgumentParser(description='Experiment 5: GA-CE Loss Term Ablation')

    parser.add_argument('--dataset', type=str, default='skab', choices=['skab', 'swat'])
    parser.add_argument('--detector', type=str, default='conv_ae', choices=pl.DETECTORS)
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
    parser.add_argument('--no_adaptive', action='store_true')
    parser.add_argument('--cp_mode', type=str, default=None,
                        choices=['static', 'sliding', 'aci'],
                        help='Conformal threshold policy; default: sliding '
                             '(or static if --no_adaptive is set)')
    parser.add_argument('--aci_gamma', type=float, default=0.005,
                        help='ACI step size gamma (only used with --cp_mode aci)')

    parser.add_argument('--n_generations', type=int, default=100)
    parser.add_argument('--population_size', type=int, default=50)
    parser.add_argument('--max_anomalies', type=int, default=100)
    parser.add_argument('--with_smoothness', action='store_true',
                        help='Also compute the smoothness diagnostic (reruns GA-CE once more per variant)')

    parser.add_argument('--output_dir', type=str, default='./results/')
    parser.add_argument('--cache_dir', type=str, default='./results/cache/')
    parser.add_argument('--force_retrain', action='store_true')
    parser.add_argument('--seed', type=int, default=42)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    np.random.seed(args.seed)
    run_experiment(args)
