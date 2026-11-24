#!/usr/bin/env python
"""
    Consistency checks on the output of run_all_experiments.py:

    1. exp_1 and exp_2 must agree per detector at alpha = 0.05. A mismatch
       means a detector was retrained instead of loaded from the cache.
    2. Accuracy and coverage are reported side by side; they are expected
       to differ.
    3. The "Full GA-CE" row of the ablation (exp5_gace_ablation_*.csv) must
       be of the same order of magnitude as the GA-CE row of exp_4 for the
       same (dataset, detector).

    Usage:
        python experiments/check_consistency.py --results_dir results/skab_<run_id>
"""

import argparse
import glob
import os
import sys

import pandas as pd


def _load_all(pattern: str) -> "pd.DataFrame | None":
    """
    Load and concatenate every CSV matching `pattern`, deduplicated by
    (detector, dataset) keeping the most recent file. A shared results
    directory (e.g. one per --run_id, written into by several SLURM array
    tasks, one per detector) accumulates one exp*_<dataset>_<timestamp>.csv
    per run - taking only the single newest file (as an earlier version of
    this script did) silently drops every detector except whichever task
    happened to finish last, which looks like "missing data" for the others
    rather than the real, present result.
    """
    # recursive=True + '**' so results written into nested run subfolders
    # (e.g. results/skab_<run_id>/skab_<timestamp>/ produced by a refit job
    # whose --output_dir pointed at the run folder) are checked too - a
    # non-recursive glob silently skipped conv_ae's nested CSVs, which is
    # exactly how an exp1-vs-exp2 coverage mismatch went undetected.
    dirname, basename = os.path.split(pattern)
    matches = sorted(glob.glob(os.path.join(dirname, '**', basename), recursive=True))
    matches += sorted(glob.glob(pattern))
    # exclude derived artifacts (aggregate_seeds.py outputs) - they match the
    # exp*_ glob but have _mean/_std columns, not raw metric columns
    matches = [m for m in dict.fromkeys(sorted(matches))
               if 'aggregated' not in m and not m.endswith('_detailed.csv')]
    if not matches:
        return None
    frames = []
    for path in matches:
        df = pd.read_csv(path)
        df['_source_file'] = path
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)
    key_cols = [c for c in ('detector', 'alpha', 'method', 'seed', 'variant', 'cp_mode') if c in combined.columns]
    if 'detector' in combined.columns:
        # keep the latest file's rows per detector (files are already in
        # sorted/chronological order from glob+sort, so "last occurrence" wins)
        combined = combined.drop_duplicates(subset=key_cols, keep='last') if key_cols else combined
    return combined


def check_table1_vs_table2(results_dir: str, alpha: float = 0.05, tol: float = 1e-6) -> bool:
    df1 = _load_all(os.path.join(results_dir, 'exp1_*.csv'))
    df2 = _load_all(os.path.join(results_dir, 'exp2_*.csv'))
    if df1 is None or df2 is None:
        print(f"[SKIP] Table1-vs-Table2 check: missing exp1 or exp2 CSV in {results_dir}")
        return True

    df2_alpha = df2[abs(df2['alpha'] - alpha) < 1e-9]

    ok = True
    print(f"\n--- exp_1 vs exp_2 (alpha={alpha}) ---")
    for _, row1 in df1.iterrows():
        det = row1['detector']
        seed = row1.get('seed', None)
        row2 = df2_alpha[df2_alpha['detector'] == det]
        # multi-seed runs: rows must be compared within the same seed, or a
        # seed-42 exp1 row gets checked against a seed-43 exp2 row and fails
        # spuriously (different trained models, correctly so).
        if seed is not None and 'seed' in df2_alpha.columns:
            row2 = row2[row2['seed'] == seed]
        if row2.empty:
            print(f"  [SKIP] {det} (seed={seed}): no matching exp_2 row at alpha={alpha}")
            continue
        row2 = row2.iloc[0]
        cov_diff = abs(row1['coverage'] - row2['observed_coverage'])
        err_diff = abs(row1['calibration_error'] - row2['calibration_error'])
        status = "OK" if (cov_diff < tol and err_diff < tol) else "MISMATCH"
        if status == "MISMATCH":
            ok = False
        print(f"  [{status}] {det} (seed={seed}): Table1 coverage={row1['coverage']:.6f} vs "
              f"Table2 observed={row2['observed_coverage']:.6f} (diff={cov_diff:.2e}); "
              f"Table1 cal_err={row1['calibration_error']:.6f} vs "
              f"Table2 cal_err={row2['calibration_error']:.6f} (diff={err_diff:.2e})")
        print(f"           accuracy={row1['accuracy']:.4f} vs coverage={row1['coverage']:.4f} "
              f"(gap={abs(row1['accuracy']-row1['coverage']):.4f} - expected to be nonzero)")

    return ok


def check_gace_ablation_vs_table5(results_dir: str, max_ratio: float = 10.0) -> bool:
    df4 = _load_all(os.path.join(results_dir, 'exp4_*.csv'))
    exp5_paths = [p for p in glob.glob(os.path.join(results_dir, '**', 'exp5_gace_ablation_*.csv'), recursive=True)
                  if 'aggregated' not in p]
    if df4 is None or not exp5_paths:
        print(f"\n[SKIP] GA-CE ablation vs exp_4 check: missing exp4 or exp5 CSV in {results_dir}")
        return True

    ok = True
    print("\n--- GA-CE ablation ('Full GA-CE') vs exp_4 ---")
    for exp5_path in sorted(exp5_paths):
        df5 = pd.read_csv(exp5_path)
        if df5.empty:
            continue
        detector = df5['detector'].iloc[0]
        # exp4 may contain one row per method (GA-CE/DICE/
        # PerCE) for this detector - the ablation table only concerns GA-CE,
        # so filter to that method explicitly rather than risk comparing
        # against an arbitrary row.
        row4 = df4[(df4['detector'] == detector) & (df4.get('method', 'GA-CE') == 'GA-CE')]
        if 'seed' in df5.columns and 'seed' in df4.columns:
            row4 = row4[row4['seed'] == df5['seed'].iloc[0]]
        if 'cp_mode' in df5.columns and 'cp_mode' in df4.columns:
            row4 = row4[row4['cp_mode'] == df5['cp_mode'].iloc[0]]
        row5 = df5[df5['variant'] == 'Full GA-CE']
        if row4.empty or row5.empty:
            print(f"  [SKIP] {detector}: missing exp_4 or 'Full GA-CE' row")
            continue
        row4, row5 = row4.iloc[0], row5.iloc[0]

        def ratio(a, b):
            lo, hi = sorted([abs(a), abs(b)])
            return float('inf') if lo == 0 else hi / lo

        prox_ratio = ratio(row4['mean_proximity'], row5['proximity'])
        status = "OK" if prox_ratio <= max_ratio else "MISMATCH"
        if status == "MISMATCH":
            ok = False
        print(f"  [{status}] {detector}: Table5 proximity={row4['mean_proximity']:.4f} vs "
              f"ablation 'Full GA-CE' proximity={row5['proximity']:.4f} (ratio={prox_ratio:.1f}x, "
              f"flag if > {max_ratio}x)")
    return ok


def main():
    parser = argparse.ArgumentParser(description="Sanity-check results before trusting them")
    parser.add_argument('--results_dir', type=str, required=True)
    parser.add_argument('--alpha', type=float, default=0.05)
    args = parser.parse_args()

    if not os.path.isdir(args.results_dir):
        print(f"No such directory: {args.results_dir}")
        sys.exit(2)

    ok1 = check_table1_vs_table2(args.results_dir, alpha=args.alpha)
    ok2 = check_gace_ablation_vs_table5(args.results_dir)

    print("\n" + "-" * 60)
    if ok1 and ok2:
        print("ALL CHECKS PASSED")
    else:
        print("ONE OR MORE CHECKS FAILED - do not trust these numbers yet, see messages above")
    print("-" * 60)
    sys.exit(0 if (ok1 and ok2) else 1)


if __name__ == '__main__':
    main()
