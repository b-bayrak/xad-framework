#!/usr/bin/env python
"""
    Aggregate multi-seed result CSVs into mean +/- std tables.

    Scans a results directory (recursively) for exp{1,2,4,5,6}_*.csv files,
    groups rows by every identifying column (detector, dataset, alpha,
    method, variant, cp_mode, smoothness_weight, confidence_level - whichever
    are present) across seeds, and writes one aggregated CSV per experiment
    with <metric>_mean and <metric>_std columns plus n_seeds.

    Rows are deduplicated per (identity columns + seed) keeping the most
    recent file first, so re-running an experiment for one seed supersedes
    its older CSV without double-counting.

    Usage:
        python experiments/aggregate_seeds.py --results_dir results/skab_final --out_dir results/skab_final/aggregated
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd

# NOTE: n_anomalies is deliberately NOT an identity column - the number of
# detected anomalies varies per seed (different trained models), so keying on
# it would split each (detector, method) group into per-seed singletons and
# silently produce std=NaN rows instead of a 3-seed aggregate.
ID_CANDIDATES = ['dataset', 'detector', 'alpha', 'expected_coverage', 'method',
                 'variant', 'cp_mode', 'smoothness_weight', 'aci_gamma',
                 'confidence_level']


def load_experiment(results_dir: str, prefix: str) -> "pd.DataFrame | None":
    paths = sorted(glob.glob(os.path.join(results_dir, '**', f'{prefix}_*.csv'), recursive=True))
    paths = [p for p in paths if not p.endswith('_detailed.csv') and os.sep + 'aggregated' + os.sep not in p]
    if not paths:
        return None
    frames = []
    for p in paths:
        df = pd.read_csv(p)
        df['_file'] = p
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    if 'seed' not in df.columns:
        df['seed'] = 42  # pre-multi-seed CSVs were all seed 42
    id_cols = [c for c in ID_CANDIDATES if c in df.columns]
    # newest file wins per (identity, seed)
    df = df.drop_duplicates(subset=id_cols + ['seed'], keep='last')
    return df, id_cols


def aggregate(df: pd.DataFrame, id_cols):
    metric_cols = [c for c in df.columns
                   if c not in id_cols + ['seed', '_file']
                   and np.issubdtype(df[c].dtype, np.number)]
    g = df.groupby(id_cols, dropna=False)
    agg = g[metric_cols].agg(['mean', 'std'])
    agg.columns = [f'{m}_{s}' for m, s in agg.columns]
    agg['n_seeds'] = g['seed'].nunique()
    return agg.reset_index()


def main():
    parser = argparse.ArgumentParser(description='Aggregate multi-seed CSVs into mean +/- std')
    parser.add_argument('--results_dir', type=str, required=True)
    parser.add_argument('--out_dir', type=str, default=None,
                        help='Default: <results_dir>/aggregated')
    args = parser.parse_args()

    out_dir = args.out_dir or os.path.join(args.results_dir, 'aggregated')
    os.makedirs(out_dir, exist_ok=True)

    for prefix in ['exp1', 'exp2', 'exp3', 'exp4', 'exp5_gace_ablation', 'exp6_cp_ablation']:
        loaded = load_experiment(args.results_dir, prefix)
        if loaded is None:
            print(f"[skip] no {prefix}_*.csv under {args.results_dir}")
            continue
        df, id_cols = loaded
        agg = aggregate(df, id_cols)
        out = os.path.join(out_dir, f'{prefix}_aggregated.csv')
        agg.to_csv(out, index=False)
        n_seeds = df['seed'].nunique()
        print(f"[ok] {prefix}: {len(df)} rows, {n_seeds} seed(s), "
              f"{len(agg)} aggregated rows -> {out}")


if __name__ == '__main__':
    main()
