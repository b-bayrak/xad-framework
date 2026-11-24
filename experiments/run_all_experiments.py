#!/usr/bin/env python
"""
    Generates results for Tables 1-5 (plus the Appendix GA-CE ablation, exp 5).

    All five scripts share one trained+calibrated model per (dataset,
    detector) via experiments/pipeline.py's on-disk cache (--cache_dir), so
    running exp 1-5 back to back for the same dataset produces mutually
    consistent numbers. Re-running this script a second time with
    the same --cache_dir reuses the cached weights instead of retraining;
    pass --force_retrain to force a clean retrain.

    Usage:
        python experiments/run_all_experiments.py --dataset skab --data_path ./data/SKAB/
        python experiments/run_all_experiments.py --dataset swat --data_path ./data/SWaT/ --experiments 1 2
"""

import os
import sys
import argparse
import subprocess
from datetime import datetime
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


EXP_SCRIPTS = {
    1: 'exp_1_detection_performance.py',
    2: 'exp_2_calibration_quality.py',
    3: 'exp_3_uncertainty_effectiveness.py',
    4: 'exp_4_counterfactual_quality.py',
    5: 'exp_5_gace_ablation.py',
    6: 'exp_6_cp_ablation.py',
}

EXP_NAMES = {
    1: 'Detection Performance',
    2: 'Calibration Quality',
    3: 'Uncertainty Effectiveness',
    4: 'Counterfactual Quality',
    5: 'GA-CE Loss Term Ablation (Appendix)',
    6: 'CP Threshold-Policy x TVD Ablation (static vs sliding vs ACI)',
}

# CLI flags every exp_*.py script accepts and that should be forwarded
# identically so all five stay consistent with each other.
SHARED_FLAGS = [
    'dataset', 'data_path', 'test_size', 'val_size',
    'hidden_dim', 'latent_dim', 'num_layers', 'dropout',
    'learning_rate', 'weight_decay', 'kl_weight', 'batch_size', 'epochs', 'patience',
    'seed', 'cache_dir',
]

# exp_2 sweeps alpha in {0.05, 0.10, 0.15, 0.20} internally
# and does not accept a single --alpha - forwarding it there fails argparse
# ("unrecognized arguments: --alpha ...") before the script does anything.
# Every other script wants exactly one --alpha.
EXPS_WITH_ALPHA_FLAG = {1, 3, 4, 5}


def run_experiment(exp_num: int, args, output_dir: str) -> bool:
    script_name = EXP_SCRIPTS.get(exp_num)
    if script_name is None:
        print(f"Unknown experiment: {exp_num}")
        return False

    script_path = os.path.join(os.path.dirname(__file__), script_name)
    if not os.path.exists(script_path):
        print(f"Script not found: {script_path}")
        return False

    cmd = [sys.executable, script_path, '--output_dir', output_dir]
    for flag in SHARED_FLAGS:
        cmd += [f'--{flag}', str(getattr(args, flag))]
    if exp_num in EXPS_WITH_ALPHA_FLAG:
        cmd += ['--alpha', str(args.alpha)]
    if exp_num != 6:
        # exp_6 sweeps threshold policies and alphas internally; it accepts
        # neither --no_adaptive nor --cp_mode (forwarding them would fail
        # argparse, like the exp_2 --alpha incident).
        if args.no_adaptive:
            cmd.append('--no_adaptive')
        if args.cp_mode:
            cmd += ['--cp_mode', args.cp_mode]
    cmd += ['--aci_gamma', str(args.aci_gamma)]
    # --force_retrain must only reach the FIRST experiment that runs: each
    # exp_*.py script passes it straight to get_or_train_detector(), so
    # forwarding it to every sub-script made exp_1, exp_2, and exp_4 each
    # retrain (GPU training is non-deterministic even when seeded) and
    # overwrite the cache - Tables 1 and 2 then described different models,
    # the exact inconsistency the shared cache exists to prevent. Confirmed
    # in the conv_ae refit runs: SKAB coverage 0.9057 (exp1) vs 0.9013
    # (exp2), SWaT 0.740 vs 0.788, while never-force-retrained detectors
    # matched bit-exactly. Caller (main) clears the flag after the first
    # experiment via args._force_retrain_pending.
    if getattr(args, '_force_retrain_pending', False):
        cmd.append('--force_retrain')
    if exp_num in (1, 2, 3, 4, 6) and args.detector:
        cmd += ['--detector', args.detector]
    if exp_num == 5:
        # exp 5 always takes a single detector; --gace_detector overrides
        # --detector for the ablation specifically if you want a different
        # one than the main sweep, otherwise it follows --detector (or
        # conv_ae if neither is set). exp 5 runs GA-CE generation 4x (once
        # per ablated variant) over --max_anomalies anomalies each, so it is
        # the single most expensive part of the sweep on SWaT - budget
        # walltime accordingly or lower --max_anomalies.
        gace_detector = args.gace_detector or args.detector or 'conv_ae'
        cmd += ['--detector', gace_detector,
                '--n_generations', str(args.n_generations),
                '--population_size', str(args.population_size),
                '--max_anomalies', str(args.max_anomalies)]
        if args.with_smoothness:
            cmd.append('--with_smoothness')
    if exp_num == 4:
        cmd += ['--n_generations', str(args.n_generations),
                '--population_size', str(args.population_size),
                '--max_anomalies', str(args.max_anomalies)]

    print(f"\n{'-'*70}\nRunning Experiment {exp_num}: {script_name}\n{'-'*70}\n")
    print(' '.join(cmd))
    result = subprocess.run(cmd, capture_output=False)
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description='Run all XAD experiments')

    parser.add_argument('--dataset', type=str, default='skab', choices=['skab', 'swat'])
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--experiments', type=int, nargs='+', default=[1, 2, 3, 4, 5, 6])
    parser.add_argument('--output_dir', type=str, default='./results/')
    parser.add_argument('--cache_dir', type=str, default='./results/cache/')
    parser.add_argument('--force_retrain', action='store_true')
    parser.add_argument('--run_id', type=str, default=None,
                        help='Shared output-folder suffix (e.g. a SLURM job ID) so parallel '
                             'per-detector array tasks for the same dataset write into the '
                             'same results/<dataset>_<run_id>/ directory instead of each '
                             'getting their own timestamp')

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
    parser.add_argument('--no_adaptive', action='store_true',
                        help='Disable sliding-window recalibration (Eq. 8) across all experiments')
    parser.add_argument('--cp_mode', type=str, default=None,
                        choices=['static', 'sliding', 'aci'],
                        help='Conformal threshold policy forwarded to all experiments; '
                             'default: sliding (or static if --no_adaptive is set)')
    parser.add_argument('--aci_gamma', type=float, default=0.005,
                        help='ACI step size gamma (only used with --cp_mode aci)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--detector', type=str, default=None,
                        choices=['conv_ae', 'lstm_ae', 'lstm_vae'],
                        help='Restrict experiments 1-4 to a single detector (for SLURM '
                             'array jobs, one task per detector); default: all three. '
                             'Also used as exp 5\'s detector unless --gace_detector is set.')

    parser.add_argument('--n_generations', type=int, default=100)
    parser.add_argument('--population_size', type=int, default=50)
    parser.add_argument('--max_anomalies', type=int, default=100,
                        help='Cap on anomalies used for GA-CE generation in exp 4/5 (dominant '
                             'cost on SWaT - lower this to bound walltime, e.g. 30)')
    parser.add_argument('--gace_detector', type=str, default=None,
                        choices=['conv_ae', 'lstm_ae', 'lstm_vae'],
                        help='Detector to use for exp 5 (GA-CE ablation); defaults to --detector, or conv_ae')
    parser.add_argument('--with_smoothness', action='store_true')

    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = args.run_id or timestamp
    output_dir = os.path.join(args.output_dir, f'{args.dataset}_{run_id}')
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "-" * 70)
    print("XAD FRAMEWORK - RUNNING ALL EXPERIMENTS")
    print("-" * 70)
    print(f"\nDataset: {args.dataset.upper()}")
    print(f"Data path: {args.data_path}")
    print(f"Model cache: {args.cache_dir}  (force_retrain={args.force_retrain})")
    print(f"Output directory: {output_dir}")
    print(f"Experiments to run: {args.experiments}")

    config = vars(args)
    config['timestamp'] = timestamp
    config['output_dir'] = output_dir
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    results = {}
    # Only the first experiment gets --force_retrain (see run_experiment);
    # every later one must load the cache that first run wrote, or the
    # tables come from different models.
    args._force_retrain_pending = args.force_retrain
    for exp_num in args.experiments:
        success = run_experiment(exp_num, args, output_dir)
        results[exp_num] = 'SUCCESS' if success else 'FAILED'
        if success:
            args._force_retrain_pending = False

    print("\n" + "-" * 70)
    print("EXPERIMENT SUMMARY")
    print("-" * 70)
    for exp_num, status in results.items():
        print(f"  Experiment {exp_num}: {EXP_NAMES.get(exp_num, 'Unknown'):40s} [{status}]")

    print(f"\nResults saved to: {output_dir}")
    print("-" * 70)


if __name__ == '__main__':
    main()
