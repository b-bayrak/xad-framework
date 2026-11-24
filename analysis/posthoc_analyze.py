"""
Summaries and tests for the post-hoc analyses P1-P6 (ANALYSIS_PLAN.md). Each part
runs only if its results are complete, and every summary is written to
results/posthoc/. Statistics follow the main analysis: unit =
dataset (seeds averaged first), two-sided Wilcoxon over the twelve datasets, Holm
over the detector x alpha (or detector x explainer) combinations of a question.

    bash analysis/run.sh posthoc_analyze.py [--parts p1 p3 p4 p5 p6]
"""
import argparse
import glob
import json
import os
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
from scipy import stats as st
from sklearn.metrics import roc_auc_score

import analyze as A
import posthoc_common as C

R = C.OUT
S = os.path.join(C.REPO, 'results', 'posthoc')
os.makedirs(S, exist_ok=True)
NUM = {}                                  # values exported to the paper build
PARTIAL = False                           # --test_partial only: layout tests on unfinished runs, never the real summary


def put(k, v):
    if v is not None and not (isinstance(v, float) and np.isnan(v)):
        NUM[k] = v if isinstance(v, str) else float(v)


def wilc(x, y):
    return A.wilcoxon_row(np.asarray(x, float), np.asarray(y, float))


def exp6_reference():
    """Deployed-setting exp_6 rows of the 5-seed runs, for the reproduction check."""
    fs = glob.glob(os.path.join(C.E, 'results', '*_5seed_*', 'exp6_cp_ablation_*.csv')) + \
        glob.glob(os.path.join(C.REPO, 'runs', 'skab_swat', 'results', '*_5seed_*', 'exp6_cp_ablation_*.csv'))
    e = pd.concat(pd.read_csv(f).assign(_f=os.path.basename(f),
                                        main_job=os.path.basename(os.path.dirname(f)).split('_')[-1]) for f in fs)
    e = e[(e.smoothness_weight == 0) & ((e.cp_mode != 'aci') | np.isclose(e.aci_gamma.astype(float), 0.005))]
    return e.sort_values('_f').drop_duplicates(['dataset', 'detector', 'seed', 'cp_mode', 'alpha'], keep='last')


def gpu_of(jobs):
    """GPU model of each array task (sacct), cached in summary/job_gpus.csv so the
    check can be rerun after the accounting records expire."""
    f = os.path.join(S, 'job_gpus.csv')
    old = pd.read_csv(f, dtype=str) if os.path.exists(f) else pd.DataFrame(columns=['job', 'task', 'gpu'])
    need = sorted(set(jobs) - set(old.job))
    rows = []
    for j in need:
        out = subprocess.run(['sacct', '-j', j, '-X', '-n', '-P', '--format=JobID,AllocTRES'],
                             capture_output=True, text=True).stdout
        for line in out.split():
            jid, tres = line.split('|', 1)
            if '_' in jid and 'gres/gpu:' in tres:
                rows.append(dict(job=j, task=jid.split('_')[1], gpu=tres.split('gres/gpu:')[1].split('=')[0]))
    new = pd.concat([old, pd.DataFrame(rows)]).drop_duplicates(['job', 'task'])
    new.to_csv(f, index=False)
    return {(r.job, int(r.task)): r.gpu for r in new.itertuples()}


def p1_tasks():
    """(dataset, detector) -> (job, task) of the P1 run, from the job logs."""
    out = {}
    for f in glob.glob(os.path.join(C.RUNS_POSTHOC, 'logs', 'policies_*_*.out')):
        job, task = os.path.basename(f)[len('policies_'):-4].split('_')
        for line in open(f):
            if line.startswith('DONE '):
                _, ds, det = line.split()
                out[(ds, det)] = (job, int(task))
    return out


def part_p1():
    fs = sorted(glob.glob(os.path.join(R, 'policies', '*.csv')))
    d = pd.concat(pd.read_csv(f) for f in fs)
    assert len(fs) == 36, f'P1 incomplete: {len(fs)}/36 files'
    # reproduction check against exp_6
    e = exp6_reference().rename(columns={'cp_mode': 'policy'})
    m = d.merge(e[['dataset', 'detector', 'seed', 'policy', 'alpha', 'coverage', 'recall', 'f1', 'main_job']],
                on=['dataset', 'detector', 'seed', 'policy', 'alpha'], suffixes=('', '_exp6'))
    diff = {k: float((m[k] - m[f'{k}_exp6']).abs().max()) for k in ['coverage', 'recall', 'f1']}
    put('p1.repro.rows', len(m)); put('p1.repro.max_abs_diff', max(diff.values()))
    for k, v in diff.items():
        put(f'p1.repro.max_abs_diff.{k}', v)
    # P1 re-scores the cached detectors. Scores agree with the original run only to
    # float precision when the GPU model differs; the censored sliding window feeds
    # its own decisions back into the threshold and amplifies such differences.
    # Original task index = detector + 3 * seed (all 5-seed sbatch files).
    p1t = p1_tasks()
    orig = {(r.main_job, C.DETS.index(r.detector) + 3 * C.SEEDS.index(r.seed)) for r in m.itertuples()}
    gpu = gpu_of({j for j, _ in orig} | {j for j, _ in p1t.values()})
    m['gpu_exp6'] = [gpu.get((r.main_job, C.DETS.index(r.detector) + 3 * C.SEEDS.index(r.seed)), '?') for r in m.itertuples()]
    m['gpu_p1'] = [gpu.get(p1t.get((r.dataset, r.detector), ('', -1)), '?') for r in m.itertuples()]
    m['cov_absdiff'] = (m.coverage - m.coverage_exp6).abs()
    m.to_csv(os.path.join(S, 'p1_reproduction_check.csv'), index=False)
    put('p1.repro.gpu_unknown', ((m.gpu_exp6 == '?') | (m.gpu_p1 == '?')).sum())
    for p in ['static', 'sliding', 'aci']:
        for same, tag in [(True, 'samegpu'), (False, 'diffgpu')]:
            x = m[(m.policy == p) & ((m.gpu_exp6 == m.gpu_p1) == same)]
            put(f'p1.repro.{p}.{tag}.runs', len(x))
            put(f'p1.repro.{p}.{tag}.exact', (x.cov_absdiff == 0).mean())
            put(f'p1.repro.{p}.{tag}.max', x.cov_absdiff.max())
    # what reaches the paper: seed-averaged cells
    sa = m.assign(gap=m.coverage - (1 - m.alpha), gap6=m.coverage_exp6 - (1 - m.alpha)) \
        .groupby(['dataset', 'detector', 'policy', 'alpha'])[['gap', 'gap6', 'recall', 'recall_exp6']].mean().reset_index()
    for p in ['static', 'sliding', 'aci']:
        x = sa[sa.policy == p]
        put(f'p1.repro.{p}.cell.max_cov', (x.gap - x.gap6).abs().max())
        put(f'p1.repro.{p}.cell.max_recall', (x.recall - x.recall_exp6).abs().max())
        put(f'p1.repro.{p}.within3.exp6', (x.gap6.abs() <= A.CRIT_PTS).sum())
        put(f'p1.repro.{p}.within3.p1', (x.gap.abs() <= A.CRIT_PTS).sum())
    flip = sa[(sa.gap.abs() <= A.CRIT_PTS) != (sa.gap6.abs() <= A.CRIT_PTS)]
    flip.to_csv(os.path.join(S, 'p1_reproduction_boundary_cells.csv'), index=False)
    print(f'P1 reproduction vs exp_6: {len(m)} rows, max |diff| {diff}; '
          f'{len(flip)} seed-averaged cell(s) cross the 3-point boundary')
    print(m.groupby(['policy', m.gpu_exp6 == m.gpu_p1]).cov_absdiff
          .agg(runs='size', exact=lambda v: (v == 0).mean(), max='max').round(5).to_string())

    # total alarm rate (normal and anomalous windows) minus alpha, per run. Label-free
    # ACI counts every flag as an error, so its long-run alarm rate is held at alpha.
    nn = d.n_test - d.n_test_anom
    d['alarm_dev'] = ((1 - d.coverage) * nn + d.recall * d.n_test_anom) / d.n_test - d.alpha
    for p, x in d.groupby('policy'):
        put(f'p1.{p}.alarm_dev.median', x.alarm_dev.median())
        put(f'p1.{p}.alarm_dev.min', x.alarm_dev.min()); put(f'p1.{p}.alarm_dev.max', x.alarm_dev.max())
        put(f'p1.{p}.alarm_dev.absmax', x.alarm_dev.abs().max())
        put(f'p1.{p}.alarm_dev.within1', (x.alarm_dev.abs() <= 0.01).mean())
        put(f'p1.{p}.runs', len(x))
    d[['dataset', 'detector', 'seed', 'policy', 'alpha', 'alarm_dev']].to_csv(os.path.join(S, 'p1_alarm_rate.csv'), index=False)
    # with the alarm rate held at alpha, ACI's coverage gap is pi * (recall - alpha) / (1 - pi), pi = anomaly share
    x = d[d.policy == 'aci']
    pi = x.n_test_anom / x.n_test
    res = x.gap - pi * (x.recall - x.alpha) / (1 - pi)
    put('p1.aci.identity.absmax', res.abs().max()); put('p1.aci.identity.median_abs', res.abs().median())
    put('p1.aci.identity.corr', np.corrcoef(x.gap, pi * (x.recall - x.alpha) / (1 - pi))[0, 1])
    cres = x.assign(res=res).groupby(['dataset', 'detector', 'alpha']).res.mean()
    put('p1.aci.identity.cell_absmax', cres.abs().max())
    put('p1.pi.min', pi.min()); put('p1.pi.max', pi.max())
    # per dataset x detector x policy x alpha, seeds averaged
    g = d.groupby(['dataset', 'detector', 'policy', 'alpha']).agg(
        coverage=('coverage', 'mean'), gap=('gap', 'mean'), recall=('recall', 'mean'), f1=('f1', 'mean'),
        oracle=('oracle_recall_same_fpr', 'mean'), fpr=('fpr', 'mean'), n=('seed', 'nunique')).reset_index()
    g['recall_minus_oracle'] = g.recall - g.oracle
    g.to_csv(os.path.join(S, 'p1_policies.csv'), index=False)
    pol = ['static', 'sliding', 'sliding_all', 'aci']
    cells = g[g.policy == 'sliding_all']
    put('p1.cells', len(cells))
    for p in pol:
        x = g[g.policy == p]
        put(f'p1.{p}.within3', (x.gap.abs() <= A.CRIT_PTS).sum())
        put(f'p1.{p}.under3', (x.gap < -A.CRIT_PTS).sum())
        put(f'p1.{p}.over3', (x.gap > A.CRIT_PTS).sum())
        put(f'p1.{p}.gap.median', x.gap.median())
        put(f'p1.{p}.gap.min', x.gap.min()); put(f'p1.{p}.gap.max', x.gap.max())
        put(f'p1.{p}.recall.median', x.recall.median())
        put(f'p1.{p}.recall_minus_oracle.median', x.recall_minus_oracle.median())
        put(f'p1.{p}.recall_minus_oracle.lo', x.recall_minus_oracle.quantile(0.25))
        put(f'p1.{p}.recall_minus_oracle.hi', x.recall_minus_oracle.quantile(0.75))
        put(f'p1.{p}.beats_oracle', (x.recall_minus_oracle > 0).sum())
        put(f'p1.{p}.below_oracle', (x.recall_minus_oracle < 0).sum())
        a05 = x[np.isclose(x.alpha, 0.05)]
        put(f'p1.{p}.a05.cov.min', a05.coverage.min()); put(f'p1.{p}.a05.cov.max', a05.coverage.max())
    # uncensored closer than censored?
    piv = g.pivot_table(index=['dataset', 'detector', 'alpha'], columns='policy', values='gap')
    put('p1.uncens_closer_than_cens', (piv.sliding_all.abs() < piv.sliding.abs()).sum())
    put('p1.uncens_closer_than_aci', (piv.sliding_all.abs() < piv.aci.abs()).sum())
    u3 = g[(g.policy == 'sliding_all') & (g.gap < -A.CRIT_PTS)]
    put('p1.unc_under3.gecco_msl', u3.dataset.isin(['gecco', 'msl']).sum())
    put('p1.unc_under3.top2', ','.join(u3.dataset.value_counts().index[:2]))
    # tests over datasets: uncensored vs censored |gap|, uncensored vs ACI |gap|, recall
    rows = []
    for q, (p_a, p_b, col, ab) in {'unc_vs_cens_gap': ('sliding_all', 'sliding', 'gap', True),
                                    'unc_vs_aci_gap': ('sliding_all', 'aci', 'gap', True),
                                    'unc_vs_aci_rec': ('sliding_all', 'aci', 'recall', False),
                                    'unc_vs_cens_rec': ('sliding_all', 'sliding', 'recall', False)}.items():
        keys = []
        for det in C.DETS:
            for a in C.ALPHAS:
                sub = g[(g.detector == det) & np.isclose(g.alpha, a)].pivot(index='dataset', columns='policy', values=col)
                x, y = (sub[p_a].abs(), sub[p_b].abs()) if ab else (sub[p_a], sub[p_b])
                w = wilc(x, y)
                rows.append(dict(test=q, detector=det, alpha=a, **w))
        sub = pd.DataFrame([r for r in rows if r['test'] == q])
        h = A.holm(sub.p.values)
        for r, hh in zip([r for r in rows if r['test'] == q], h):
            r['p_holm'] = hh
            put(f'p1.stat.{q}.{r["detector"]}.a{int(r["alpha"] * 100):02d}.median_diff', r['median_diff'])
            put(f'p1.stat.{q}.{r["detector"]}.a{int(r["alpha"] * 100):02d}.pholm', hh)
        put(f'p1.stat.{q}.pholm.max', max(h)); put(f'p1.stat.{q}.pholm.min', min(h))
        put(f'p1.stat.{q}.median_diff.min', sub.median_diff.min()); put(f'p1.stat.{q}.median_diff.max', sub.median_diff.max())
    pd.DataFrame(rows).to_csv(os.path.join(S, 'p1_tests.csv'), index=False)
    # threshold positions at the end of the stream (P3, original exp_4 protocol)
    fin = d[np.isclose(d.alpha, 0.05) & d.policy.isin(['sliding', 'aci'])]
    fin = fin.assign(final_inf=~np.isfinite(fin.final_threshold))
    fin.groupby(['dataset', 'detector', 'policy']).agg(pass_rate=('pass_rate_normal_test', 'mean'),
                                                       inf_runs=('final_inf', 'sum')).reset_index() \
        .to_csv(os.path.join(S, 'p3_end_of_stream_thresholds.csv'), index=False)
    put('p3.aci.end_inf_runs', fin[fin.policy == 'aci'].final_inf.sum())
    for ds, v in fin[fin.policy == 'aci'].groupby('dataset').final_inf.sum().items():
        put(f'p3.aci.end_inf.{ds}', v)
    put('p3.aci.runs', (fin.policy == 'aci').sum())


METHODS = ['GA-CE', 'DICE', 'PerCE']
MKEY = {'GA-CE': 'gace', 'DICE': 'dice', 'PerCE': 'perce'}
AK = {0.05: '05', 0.10: '10', 0.15: '15', 0.20: '20'}


def exp4_rows():
    """exp_4 rows of the 5-seed runs (end-of-stream protocol), newest file per seed."""
    import paper_outputs as MP
    frames = [A.load('exp4', dd['main'], A.EXPS['exp4']['ids'], []) for dd in MP.result_dirs().values()]
    return pd.concat([f for f in frames if f is not None], ignore_index=True)


def as01(v):
    return v.map({True: 1.0, False: 0.0, 'True': 1.0, 'False': 0.0}).astype(float)


def cf_runs(d, keys):
    """One row per run (keys + method). validity = valid / successfully generated,
    the exp_4 definition; generation errors are counted in n - n_ok."""
    d = d.copy()
    d['ok'] = d.proximity.notna()
    d['v'] = as01(d.valid).where(d.ok)
    if 'cf_plausibility' in d:
        d['cfp_valid'] = d.cf_plausibility.where(d.v == 1)
        d['cfp_le1'] = (d.cf_plausibility <= 1).astype(float).where(d.cf_plausibility.notna())
    agg = dict(n=('sample_id', 'size'), n_ok=('ok', 'sum'), validity=('v', 'mean'), proximity=('proximity', 'mean'),
               sparsity=('sparsity', 'mean'), pass_rate=('cal_pass_rate_at_threshold', 'mean'),
               thr_inf=('threshold_at_detection', lambda x: float(np.isinf(x).mean())), n_detected=('n_detected', 'first'))
    if 'cf_plausibility' in d:
        agg.update(cf_plaus=('cf_plausibility', 'median'), cf_plaus_valid=('cfp_valid', 'median'),
                   cf_plaus_le1=('cfp_le1', 'mean'), anom_plaus=('anomaly_plausibility', 'median'))
    return d.groupby(keys + ['method']).agg(**agg).reset_index()


def ols_cluster(df, y, x):
    """validity ~ pass_rate + aci indicator, OLS with standard errors clustered by
    dataset; same estimator as statsmodels OLS(cov_type='cluster') with its default
    small-sample correction and normal p-values (statsmodels is not in the xad env;
    checked against it on the same data, see AUDIT D2)."""
    df = df.dropna(subset=[y, x])
    X = np.column_stack([np.ones(len(df)), df[x].values, (df.policy == 'aci').astype(float).values])
    Y = df[y].values.astype(float)
    g = pd.factorize(df.dataset)[0]
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ X.T @ Y
    u = Y - X @ beta
    meat = sum(np.outer(X[g == k].T @ u[g == k], X[g == k].T @ u[g == k]) for k in np.unique(g))
    n, k, G = len(Y), X.shape[1], len(np.unique(g))
    V = XtX_inv @ meat @ XtX_inv * (G / (G - 1)) * ((n - 1) / (n - k))
    se = np.sqrt(np.diag(V))
    p = 2 * st.norm.sf(np.abs(beta / se))
    rho = st.spearmanr(df[x], df[y])
    return dict(n=n, b_pass=beta[1], p_pass=p[1], b_aci=beta[2], p_aci=p[2], rho=rho.correlation, rho_p=rho.pvalue)


def part_p4():
    fs = sorted(f for f in glob.glob(os.path.join(R, 'cf', '*.csv')))
    assert len(fs) == 12 * 3 * 5 * 2 or PARTIAL, f'P4 incomplete: {len(fs)}/360 files'
    frames, empty = [], []
    for f in fs:
        x = pd.read_csv(f)
        (frames if len(x) else empty).append(x if len(x) else os.path.basename(f))
    d = pd.concat(frames, ignore_index=True)
    put('p4.runs', len(fs)); put('p4.runs_no_detection', len(empty))
    put('p4.gen_errors', int(d.proximity.isna().sum())); put('p4.rows', len(d))
    runs = cf_runs(d, ['dataset', 'detector', 'seed', 'policy'])
    runs.to_csv(os.path.join(S, 'p4_runs.csv'), index=False)
    put('p4.thr_inf_rows', int(np.isinf(d.threshold_at_detection).sum()))
    cell = runs.groupby(['dataset', 'detector', 'policy', 'method']).mean(numeric_only=True).reset_index()
    cell.to_csv(os.path.join(S, 'p4_cells.csv'), index=False)
    # ACI vs sliding validity, unit = dataset, Holm over detector x explainer (as the main analysis)
    rows = []
    for det in C.DETS:
        for meth in METHODS:
            pv = cell[(cell.detector == det) & (cell.method == meth)].pivot(index='dataset', columns='policy', values='validity') \
                .reindex(columns=['aci', 'sliding'])
            rows.append(dict(detector=det, method=meth, **wilc(pv['aci'], pv['sliding'])))
    t = pd.DataFrame(rows)
    t['p_holm'] = A.holm(t.p.values)
    t.to_csv(os.path.join(S, 'p4_validity_tests.csv'), index=False)
    for r in t.itertuples():
        for c in ['median_diff', 'ci95_lo', 'ci95_hi', 'p', 'p_holm', 'n_datasets']:
            put(f'p4.stat.{r.detector}.{MKEY[r.method]}.{c}', getattr(r, c))
    put('p4.stat.pholm.min', t.p_holm.min()); put('p4.stat.pholm.max', t.p_holm.max())
    put('p4.stat.median_diff.min', t.median_diff.min()); put('p4.stat.median_diff.max', t.median_diff.max())
    # Friedman over explainers, per detector x policy
    for det in C.DETS:
        for pol in ['sliding', 'aci']:
            pv = cell[(cell.detector == det) & (cell.policy == pol)].pivot(index='dataset', columns='method', values='validity').dropna()
            if pv.shape[0] < 3 or pv.shape[1] < 3:
                assert PARTIAL, f'P4 Friedman: too few datasets for {det} {pol}'
                continue
            r = st.friedmanchisquare(*[pv[c] for c in pv.columns])
            k, n = pv.shape[1], pv.shape[0]
            put(f'p4.fried.{det}.{pol}.p', r.pvalue); put(f'p4.fried.{det}.{pol}.n', n)
            put(f'p4.fried.{det}.{pol}.cd', st.studentized_range.ppf(0.95, k, np.inf) / np.sqrt(2) * np.sqrt(k * (k + 1) / (6 * n)))
            for c, v in pv.rank(axis=1, ascending=False).mean().items():
                put(f'p4.fried.{det}.{pol}.rank.{MKEY[c]}', v)
    # descriptive: validity, proximity, sparsity, plausibility per policy x method (medians over datasets)
    dsm = cell.groupby(['dataset', 'policy', 'method']).mean(numeric_only=True).reset_index()
    for (pol, meth), g in dsm.groupby(['policy', 'method']):
        b = f'p4.{pol}.{MKEY[meth]}'
        for c in ['validity', 'proximity', 'sparsity', 'cf_plaus', 'cf_plaus_valid', 'cf_plaus_le1', 'anom_plaus', 'pass_rate']:
            put(f'{b}.{c}.median', g[c].median()); put(f'{b}.{c}.min', g[c].min()); put(f'{b}.{c}.max', g[c].max())
    put('p4.anom_plaus.median', dsm.drop_duplicates(['dataset', 'policy']).anom_plaus.median())
    # per detector under ACI (medians over datasets) and per dataset (means over detectors)
    for (det, meth), y in cell[cell.policy == 'aci'].groupby(['detector', 'method']):
        b_ = f'p4.aci.{det}.{MKEY[meth]}'
        put(f'{b_}.validity.median', y.validity.median()); put(f'{b_}.cf_plaus_valid.median', y.cf_plaus_valid.median())
        put(f'{b_}.anom_plaus.median', y.anom_plaus.median())
        put(f'{b_}.plaus_worse_than_anom', (y.cf_plaus_valid > y.anom_plaus).sum()); put(f'{b_}.cells', len(y))
    for (ds, pol, meth), y in cell.groupby(['dataset', 'policy', 'method']):
        put(f'p4.{ds}.{pol}.{MKEY[meth]}.validity', y.validity.mean())
    dv = cell.groupby(['dataset', 'policy', 'method']).validity.mean().unstack('policy')
    for meth in METHODS:
        x = dv.xs(meth, level='method')
        over = sorted(x.index[x.sliding > 0.2])
        put(f'p4.sliding.{MKEY[meth]}.ds_over20', len(over)); put(f'p4.sliding.{MKEY[meth]}.ds_over20_list', ','.join(over))
        put(f'p4.{MKEY[meth]}.aci_higher_ds', (x.aci > x.sliding).sum())
    # which windows are explained: share of true anomalies among the sampled windows (same windows for every explainer)
    w = d[d.method == 'GA-CE']
    for pol, y in w.groupby('policy'):
        put(f'p4.{pol}.true_anom_share', y.is_true_anomaly.mean())
        ratio = y.orig_score / y.threshold_at_detection
        put(f'p4.{pol}.score_ratio.median', ratio.median()); put(f'p4.{pol}.score_ratio.below_1p1', (ratio < 1.1).mean())
    for (ds, pol), y in w.groupby(['dataset', 'policy']):
        put(f'p4.{ds}.{pol}.true_anom_share', y.is_true_anomaly.mean())
    # end-of-stream protocol (exp_4) next to time-of-detection, same cells
    e4 = exp4_rows()
    e4c = e4.groupby(['dataset', 'detector', 'cp_mode', 'method']).validity_rate.mean().rename('validity_exp4').reset_index() \
        .rename(columns={'cp_mode': 'policy'})
    both = cell.merge(e4c, on=['dataset', 'detector', 'policy', 'method'], how='outer', indicator=True)
    put('p4.vs_exp4.unmatched', (both._merge != 'both').sum())
    both = both[both._merge == 'both']
    both.drop(columns='_merge').to_csv(os.path.join(S, 'p4_vs_exp4.csv'), index=False)
    # datasets whose end-of-stream ACI threshold is infinite in every run (validity 100% there)
    infds = ['gecco', 'exathlon']
    for meth in METHODS:
        x = both[(both.policy == 'aci') & both.dataset.isin(infds) & (both.method == meth)]
        put(f'p4.infds.{MKEY[meth]}.validity', x.validity.mean()); put(f'p4.infds.{MKEY[meth]}.validity_exp4', x.validity_exp4.mean())
        put(f'p4.infds.{MKEY[meth]}.validity.min', x.validity.min()); put(f'p4.infds.{MKEY[meth]}.validity.max', x.validity.max())
        y = cell[(cell.policy == 'aci') & (cell.method == meth)]
        put(f'p4.aci.{MKEY[meth]}.plaus_worse_than_anom', (y.cf_plaus_valid > y.anom_plaus).sum())
        put(f'p4.aci.{MKEY[meth]}.plaus_cells', y.cf_plaus_valid.notna().sum())
    for pol in ['sliding', 'aci']:
        x = both[both.policy == pol]
        put(f'p4.vs_exp4.{pol}.median_diff', (x.validity - x.validity_exp4).median())
        put(f'p4.vs_exp4.{pol}.exp4_median', x.validity_exp4.median()); put(f'p4.vs_exp4.{pol}.p4_median', x.validity.median())
    # reduced reproduction check (ANALYSIS_PLAN deviation): number of sampled windows per run
    n4 = e4.groupby(['dataset', 'detector', 'seed', 'cp_mode']).n_anomalies.first().rename('n_exp4').reset_index() \
        .rename(columns={'cp_mode': 'policy'})
    n = runs.groupby(['dataset', 'detector', 'seed', 'policy']).n.first().reset_index() \
        .merge(n4, on=['dataset', 'detector', 'seed', 'policy'], how='right')
    n['n'] = n.n.fillna(0)
    put('p4.nsampled.runs', len(n)); put('p4.nsampled.match', (n.n == n.n_exp4).sum())
    n[n.n != n.n_exp4].to_csv(os.path.join(S, 'p4_nsampled_mismatch.csv'), index=False)
    print('P4 validity ACI - sliding (time of detection):'); print(t[['detector', 'method', 'median_diff', 'p_holm']].round(3).to_string(index=False))


def part_p3():
    # (a) end-of-stream protocol: exp_4 validity vs share of normal test windows below the final threshold (P1)
    p1 = pd.concat(pd.read_csv(f) for f in glob.glob(os.path.join(R, 'policies', '*.csv')))
    p1 = p1[np.isclose(p1.alpha, 0.05) & p1.policy.isin(['sliding', 'aci'])]
    e4 = exp4_rows().rename(columns={'cp_mode': 'policy'})
    a = e4.merge(p1[['dataset', 'detector', 'seed', 'policy', 'pass_rate_normal_test', 'final_threshold']],
                 on=['dataset', 'detector', 'seed', 'policy'])
    a = a.assign(validity=a.validity_rate.astype(float), final_inf=~np.isfinite(a.final_threshold))
    put('p3.eos.configs', len(a) // 3); put('p3.eos.inf_configs', a.final_inf.sum() // 3)
    put('p3.eos.inf_validity.median', a[a.final_inf].validity.median())
    rows = []
    for meth in METHODS:
        for subset, x in [('all', a), ('finite', a[~a.final_inf])]:
            r = ols_cluster(x[x.method == meth], 'validity', 'pass_rate_normal_test')
            rows.append(dict(protocol='end_of_stream', subset=subset, method=meth, **r))
    # (b) time-of-detection protocol: P4 validity vs share of calibration windows below the threshold used
    if os.path.exists(os.path.join(S, 'p4_runs.csv')):
        runs = pd.read_csv(os.path.join(S, 'p4_runs.csv'))
        for meth in METHODS:
            r = ols_cluster(runs[runs.method == meth], 'validity', 'pass_rate')
            rows.append(dict(protocol='time_of_detection', subset='all', method=meth, **r))
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(S, 'p3_validity_vs_pass_rate.csv'), index=False)
    for r in t.itertuples():
        b = f'p3.{r.protocol}.{r.subset}.{MKEY[r.method]}'
        for c in ['n', 'b_pass', 'p_pass', 'b_aci', 'p_aci', 'rho', 'rho_p']:
            put(f'{b}.{c}', getattr(r, c))
    print('P3 validity vs pass rate:'); print(t.round(3).to_string(index=False))


def part_p5():
    fs = sorted(glob.glob(os.path.join(R, 'iforest', '*_policies.csv')))
    assert len(fs) == 60, f'P5 policies incomplete: {len(fs)}/60'
    d = pd.concat(pd.read_csv(f) for f in fs)
    d['rmo'] = d.recall - d.oracle_recall_same_fpr
    # test-stream sizes from P1 (same split and windows); total alarm rate minus alpha
    sz = pd.concat(pd.read_csv(f) for f in glob.glob(os.path.join(R, 'policies', '*.csv'))) \
        .groupby('dataset')[['n_test', 'n_test_anom']].first()
    d = d.join(sz, on='dataset')
    d['alarm_dev'] = ((1 - d.coverage) * (d.n_test - d.n_test_anom) + d.recall * d.n_test_anom) / d.n_test - d.alpha
    for p, x in d.groupby('policy'):
        put(f'p5.{p}.alarm_dev.absmax', x.alarm_dev.abs().max()); put(f'p5.{p}.alarm_dev.median', x.alarm_dev.median())
    g = d.groupby(['dataset', 'policy', 'alpha']).mean(numeric_only=True).reset_index()
    g.to_csv(os.path.join(S, 'p5_policies.csv'), index=False)
    roc = d.groupby('dataset').roc_auc.mean()
    put('p5.roc.median', roc.median()); put('p5.roc.min', roc.min()); put('p5.roc.max', roc.max())
    for ds, v in roc.items():
        put(f'p5.{ds}.roc', v)
    for pol in ['static', 'sliding', 'sliding_all', 'aci']:
        x = g[g.policy == pol]
        put(f'p5.{pol}.cells', len(x))
        put(f'p5.{pol}.within3', (x.gap.abs() <= A.CRIT_PTS).sum())
        put(f'p5.{pol}.under3', (x.gap < -A.CRIT_PTS).sum()); put(f'p5.{pol}.over3', (x.gap > A.CRIT_PTS).sum())
        put(f'p5.{pol}.gap.median', x.gap.median()); put(f'p5.{pol}.rmo.median', x.rmo.median())
        for a in C.ALPHAS:
            y = x[np.isclose(x.alpha, a)]
            put(f'p5.{pol}.a{AK[a]}.gap.median', y.gap.median()); put(f'p5.{pol}.a{AK[a]}.recall.median', y.recall.median())
    rows = []
    for q, (pa, pb, col, ab) in {'aci_vs_sliding_gap': ('aci', 'sliding', 'gap', True),
                                 'aci_vs_sliding_rec': ('aci', 'sliding', 'recall', False),
                                 'aci_vs_sliding_f1': ('aci', 'sliding', 'f1', False),
                                 'unc_vs_cens_gap': ('sliding_all', 'sliding', 'gap', True),
                                 'unc_vs_aci_gap': ('sliding_all', 'aci', 'gap', True)}.items():
        part = []
        for a in C.ALPHAS:
            pv = g[np.isclose(g.alpha, a)].pivot(index='dataset', columns='policy', values=col)
            x, y = (pv[pa].abs(), pv[pb].abs()) if ab else (pv[pa], pv[pb])
            part.append(dict(test=q, alpha=a, **wilc(x, y)))
        for r, h in zip(part, A.holm([r['p'] for r in part])):
            r['p_holm'] = h
            put(f'p5.stat.{q}.a{AK[r["alpha"]]}.median_diff', r['median_diff']); put(f'p5.stat.{q}.a{AK[r["alpha"]]}.pholm', h)
        put(f'p5.stat.{q}.pholm.max', max(r['p_holm'] for r in part)); put(f'p5.stat.{q}.pholm.min', min(r['p_holm'] for r in part))
        put(f'p5.stat.{q}.median_diff.min', min(r['median_diff'] for r in part))
        put(f'p5.stat.{q}.median_diff.max', max(r['median_diff'] for r in part))
        rows += part
    pd.DataFrame(rows).to_csv(os.path.join(S, 'p5_policy_tests.csv'), index=False)
    print('P5 policy tests:'); print(pd.DataFrame(rows)[['test', 'alpha', 'median_diff', 'p_holm']].round(4).to_string(index=False))

    fs = sorted(glob.glob(os.path.join(R, 'iforest', '*_cf.csv')))
    assert len(fs) == 60, f'P5 counterfactuals incomplete: {len(fs)}/60'
    frames = [pd.read_csv(f) for f in fs]
    put('p5.cf.runs_no_detection', sum(len(x) == 0 for x in frames))
    c = pd.concat([x for x in frames if len(x)], ignore_index=True)
    put('p5.cf.gen_errors', int(c.proximity.isna().sum()))
    runs = cf_runs(c, ['dataset', 'seed', 'policy'])
    runs.to_csv(os.path.join(S, 'p5_cf_runs.csv'), index=False)
    cell = runs.groupby(['dataset', 'policy', 'method']).mean(numeric_only=True).reset_index()
    rows = []
    for meth in ['GA-CE', 'PerCE']:
        pv = cell[cell.method == meth].pivot(index='dataset', columns='policy', values='validity')
        rows.append(dict(method=meth, **wilc(pv['aci'], pv['sliding'])))
        for pol in ['sliding', 'aci']:
            v = pv[pol]
            put(f'p5.cf.{pol}.{MKEY[meth]}.validity.median', v.median())
            put(f'p5.cf.{pol}.{MKEY[meth]}.validity.min', v.min()); put(f'p5.cf.{pol}.{MKEY[meth]}.validity.max', v.max())
    t = pd.DataFrame(rows)
    t['p_holm'] = A.holm(t.p.values)
    t.to_csv(os.path.join(S, 'p5_cf_tests.csv'), index=False)
    for r in t.itertuples():
        put(f'p5.cf.stat.{MKEY[r.method]}.median_diff', r.median_diff); put(f'p5.cf.stat.{MKEY[r.method]}.pholm', r.p_holm)
    for meth in ['GA-CE', 'PerCE']:
        r = ols_cluster(runs[runs.method == meth], 'validity', 'pass_rate')
        for k in ['rho', 'b_pass', 'p_pass', 'b_aci', 'p_aci']:
            put(f'p5.cf.pass.{MKEY[meth]}.{k}', r[k])
    print('P5 CF tests:'); print(t.round(4).to_string(index=False))


def part_aci():
    """Exploratory: time ACI spends with an infinite threshold (posthoc_aci_trace.py)."""
    fs = sorted(glob.glob(os.path.join(R, 'aci_trace', '*.csv')))
    assert len(fs) == 12, f'ACI trace incomplete: {len(fs)}/12'
    d = pd.concat(pd.read_csv(f) for f in fs)
    assert len(d) == 12 * 3 * 5 * 4
    put('aci.runs', len(d)); put('aci.inf_neg.max', d.share_inf_neg.max())
    g = d.groupby(['dataset', 'alpha']).mean(numeric_only=True).reset_index()     # detectors and seeds averaged
    g.to_csv(os.path.join(S, 'aci_trace_per_dataset.csv'), index=False)
    for a in C.ALPHAS:
        x = g[np.isclose(g.alpha, a)].set_index('dataset')
        put(f'aci.a{AK[a]}.runs_any_inf', (d[np.isclose(d.alpha, a)].share_inf_pos > 0).sum())
        put(f'aci.a{AK[a]}.ds_over1pct', (x.share_inf_pos > 0.01).sum())
        put(f'aci.a{AK[a]}.ds_under1pct', (x.share_inf_pos <= 0.01).sum())
        put(f'aci.a{AK[a]}.ds_over10pct', (x.share_inf_pos > 0.10).sum())
        for ds, r in x.iterrows():
            for c in ['share_inf_pos', 'anom_under_inf', 'coverage', 'coverage_finite', 'recall', 'recall_finite',
                      'inf_episodes', 'longest_inf_run']:
                put(f'aci.{ds}.a{AK[a]}.{c}', r[c])
    print('ACI +inf share at alpha 0.05 (per dataset):')
    print(g[np.isclose(g.alpha, 0.05)].set_index('dataset')[['share_inf_pos', 'anom_under_inf', 'recall', 'recall_finite']].round(3).to_string())


def short_name(n):
    """Readable sensor labels for figures and text (SKAB uses long CamelCase names)."""
    return {'Accelerometer1RMS': 'Accel. 1 RMS', 'Accelerometer2RMS': 'Accel. 2 RMS',
            'Volume Flow RateRMS': 'Volume flow RMS'}.get(n, n)


def part_case():
    """Descriptive case study (case_study.py): selection of the stream window shown in the
    uncertainty figure and the numbers of the worked examples."""
    d = os.path.join(R, 'case_study')
    assert os.path.exists(os.path.join(d, 'case_study.json')), 'case study missing'
    cs = json.load(open(os.path.join(d, 'case_study.json')))
    z = np.load(os.path.join(d, 'stream_swat.npz'))
    y, flag = z['y'], z['flag']
    sel = int(np.where((y == 1) & (flag == 1))[0][0])        # rule: first true anomaly flagged by ACI
    names = [str(n) for n in z['names']]
    fs = z['feature_scores'][sel]
    order = np.argsort(-fs, kind='mergesort')
    st_ = cs['stream']
    for k in ['n', 'anomalies', 'flagged', 'wrong', 'inf_steps']:
        put(f'cs.stream.{k}', st_[k])
    put('cs.stream.sel', sel); put('cs.stream.sel_score', float(z['score'][sel])); put('cs.stream.sel_threshold', float(z['threshold'][sel]))
    put('cs.stream.sel_ratio', float(z['score'][sel] / z['threshold'][sel])); put('cs.stream.sel_conf', float(z['confidence'][sel]))
    for k in ['dist_score', 'dens_score', 'temp_score', 'total', 'dist_pct', 'temp_pct', 'lof']:
        put(f'cs.stream.sel_{k}', float(z[k][sel]))
    for r in range(3):
        put(f'cs.stream.sel_feat{r + 1}', names[order[r]]); put(f'cs.stream.sel_feat{r + 1}_score', float(fs[order[r]]))
    wrong = flag != y
    put('cs.stream.conf_wrong.median', float(np.median(z['confidence'][wrong])))
    put('cs.stream.conf_right.median', float(np.median(z['confidence'][~wrong])))
    put('cs.stream.total_wrong.mean', float(z['total'][wrong].mean())); put('cs.stream.total_right.mean', float(z['total'][~wrong].mean()))
    put('cs.stream.conf_lt055', float((z['confidence'] < 0.55).mean()))
    tp, fp = int(((flag == 1) & (y == 1)).sum()), int(((flag == 1) & (y == 0)).sum())
    put('cs.stream.tp', tp); put('cs.stream.fp', fp); put('cs.stream.fn', int(((flag == 0) & (y == 1)).sum()))
    put('cs.stream.recall', tp / max(1, int(y.sum())))
    put('cs.stream.conf_fp.median', float(np.median(z['confidence'][(flag == 1) & (y == 0)])))
    put('cs.stream.fp_score.median', float(np.median(z['score'][(flag == 1) & (y == 0)])))
    put('cs.stream.normal_score.median', float(np.median(z['score'][y == 0])))
    put('cs.stream.sel_pvalue', float(z['p_value'][sel]))
    put('cs.stream.cal_sd_over_median', float(np.std(z['cal']) / np.median(z['cal'])))
    cf = pd.read_csv(os.path.join(R, 'cf', 'swat_lstm_ae_s42_aci.csv'))
    cf = cf[cf.test_index == sel].set_index('method')
    put('cs.stream.sel_in_cf_sample', float(len(cf) > 0))
    for mm in METHODS if len(cf) else []:
        put(f'cs.stream.sel_cf.{MKEY[mm]}.valid_word', 'valid' if as01(pd.Series([cf.loc[mm, 'valid']])).iloc[0] == 1 else 'not valid')
        put(f'cs.stream.sel_cf.{MKEY[mm]}.prox', float(cf.loc[mm, 'proximity'])); put(f'cs.stream.sel_cf.{MKEY[mm]}.spar', float(cf.loc[mm, 'sparsity']))
        put(f'cs.stream.sel_cf.{MKEY[mm]}.plaus', float(cf.loc[mm, 'cf_plausibility']))
        put(f'cs.stream.sel_cf.{MKEY[mm]}.cf_ratio', float(cf.loc[mm, 'cf_score'] / cf.loc[mm, 'threshold_at_detection']))
    sel_info = {'stream_sel': sel}
    e = np.load(os.path.join(d, 'examples.npz'))
    enames = [str(n) for n in e['names']]
    for key, m in cs['examples'].items():
        b = f'cs.ex{key}'
        put(f'{b}.test_index', m['test_index']); put(f'{b}.candidates', m['candidates']); put(f'{b}.true', m['true_anomaly'])
        put(f'{b}.score', m['score']); put(f'{b}.threshold', m['threshold']); put(f'{b}.anom_plaus', m['anomaly_plausibility'])
        for k in ['dist_score', 'dens_score', 'temp_score', 'total']:
            put(f'{b}.{k}', m['factors'][k])
        x = e[f'{key}_x']
        delta = np.mean([np.abs(e[f'{key}_{mm}'] - x).mean(axis=0) for mm in METHODS], axis=0)
        top = np.argsort(-delta, kind='mergesort')[:3]
        sel_info[f'ex{key}_sensors'] = [int(t) for t in top]
        for r, t in enumerate(top):
            put(f'{b}.sensor{r + 1}', short_name(enames[t]))
        for mm, v in m['methods'].items():
            mk = MKEY[mm]
            put(f'{b}.{mk}.valid', float(v['valid'])); put(f'{b}.{mk}.prox', v['proximity']); put(f'{b}.{mk}.spar', v['sparsity'])
            put(f'{b}.{mk}.plaus', v['plausibility'])
            put(f'{b}.{mk}.valid_word', 'valid' if v['valid'] else 'not valid')
        for mm in METHODS:
            dx = np.abs(e[f'{key}_{mm}'] - x)
            changed = dx > 0.01 * np.std(x)                        # sparsity definition of the explainers
            put(f'{b}.{MKEY[mm]}.nchanged', int(changed.sum())); put(f'{b}.{MKEY[mm]}.nsensors', int(changed.any(axis=0).sum()))
            put(f'{b}.{MKEY[mm]}.top_sensor', short_name(enames[int(np.argmax(dx.mean(axis=0)))]))
            put(f'{b}.{MKEY[mm]}.range_ratio', float((e[f'{key}_{mm}'].max(axis=0) - e[f'{key}_{mm}'].min(axis=0)).sum()
                                                     / (x.max(axis=0) - x.min(axis=0)).sum()))
        put(f'{b}.entries', int(x.size)); put(f'{b}.nsensor_total', int(x.shape[1]))
        # confidence of the decision, with the framework's own function and the P1 calibration scores
        zz = np.load(os.path.join(R, 'policies', f"skab_{m['detector']}_s42.npz"))
        cp = C.make_cp('aci', 0.05, 'skab', zz['cal_scores'])
        put(f'{b}.confidence', cp._compute_confidence(m['score'], m['threshold'], zz['cal_scores']))
        fsx = e[f'{key}_feature_scores']
        o = np.argsort(-fsx, kind='mergesort')
        for r in range(2):
            put(f'{b}.feat{r + 1}', short_name(enames[o[r]])); put(f'{b}.feat{r + 1}_score', float(fsx[o[r]]))
    json.dump(sel_info, open(os.path.join(S, 'case_selection.json'), 'w'), indent=1)
    print('case study:', json.dumps(sel_info))


P6_FACTORS = ['distributional', 'density', 'temporal', 'feature_oor', 'feature']


def p6_margin_reference():
    """Exploratory reference for P6 (added after the P6 results were seen, see
    ANALYSIS_PLAN): AUC of the conformal margin -|score - threshold| for the same
    wrong decisions. The ACI replay at alpha = 0.05 is recomputed from the P1 scores
    and must give the stored decisions."""
    f = os.path.join(S, 'p6_margin_reference.csv')
    if os.path.exists(f):
        return pd.read_csv(f)
    rows = []
    for ds in C.DS:
        idx = pd.read_csv(os.path.join(R, 'uncertainty', f'{ds}_factors.csv')).test_index.values
        for det in C.DETS:
            for seed in C.SEEDS:
                z = np.load(os.path.join(R, 'policies', f'{ds}_{det}_s{seed}.npz'))
                preds, thr = C.replay(C.make_cp('aci', 0.05, ds, z['cal_scores']), z['test_scores'])
                assert (preds == z['aci_preds_a05']).all(), f'ACI replay differs from P1: {ds} {det} {seed}'
                wrong = (z['aci_preds_a05'][idx] != z['y_test'][idx]).astype(int)
                margin = np.abs(z['test_scores'][idx] - thr[idx].astype(np.float64))
                finite = np.isfinite(margin)
                v = np.where(finite, -margin, -np.nanmax(margin[finite]) - 1)   # +inf threshold = most certain
                rows.append(dict(dataset=ds, detector=det, seed=seed, factor='margin',
                                 value=roc_auc_score(wrong, v) if 0 < wrong.sum() < len(wrong) else np.nan,
                                 inf_threshold=int((~finite).sum())))
    out = pd.DataFrame(rows)
    out.to_csv(f, index=False)
    return out


def part_p6():
    fs = sorted(glob.glob(os.path.join(R, 'uncertainty', '*_auc.csv')))
    assert len(fs) == 12, f'P6 incomplete: {len(fs)}/12 files'
    d = pd.concat(pd.read_csv(f) for f in fs)
    a = d[d.kind == 'auc_error']
    assert len(a) == 12 * 3 * 5 * len(P6_FACTORS), 'P6: unexpected number of AUC rows'
    put('p6.runs', a.groupby(['dataset', 'detector', 'seed']).ngroups)
    put('p6.error_rate.min', a.error_rate.min()); put('p6.error_rate.max', a.error_rate.max())
    const = a[a.factor == 'feature_oor'].groupby('dataset').constant.apply(lambda v: v.astype(str).eq('True').all())
    put('p6.feature_oor.constant_datasets', const.sum())
    put('p6.feature_oor.constant_list', ','.join(sorted(const[const].index)))
    ref = p6_margin_reference()
    put('p6.margin.inf_threshold_windows', ref.inf_threshold.sum())
    allf = pd.concat([a[['dataset', 'detector', 'seed', 'factor', 'value']], ref[['dataset', 'detector', 'seed', 'factor', 'value']]])
    per_ds = allf.groupby(['dataset', 'factor']).value.mean().unstack()       # unit = dataset
    per_ds.to_csv(os.path.join(S, 'p6_auc_per_dataset.csv'))
    rows = []
    for f in P6_FACTORS:                                                     # pre-specified factors: Holm over these
        v = per_ds[f].dropna()
        rows.append(dict(factor=f, n=len(v), median=v.median(), min=v.min(), max=v.max(), above=(v > 0.5).sum(),
                         p=st.wilcoxon(v - 0.5).pvalue if len(v) >= 6 else np.nan))
    h = A.holm(np.array([r['p'] for r in rows]))
    for r, hh in zip(rows, h):
        r['p_holm'] = hh
    v = per_ds['margin']
    rows.append(dict(factor='margin (exploratory)', n=len(v), median=v.median(), min=v.min(), max=v.max(),
                     above=(v > 0.5).sum(), p=st.wilcoxon(v - 0.5).pvalue, p_holm=np.nan))
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(S, 'p6_auc_tests.csv'), index=False)
    for r in t.itertuples():
        k = 'margin' if r.factor.startswith('margin') else r.factor
        for c in ['n', 'median', 'min', 'max', 'above', 'p', 'p_holm']:
            put(f'p6.auc.{k}.{c}', getattr(r, c))
    # does any factor add to the margin? per dataset, factor AUC minus margin AUC
    for f in P6_FACTORS:
        diff = (per_ds[f] - per_ds['margin']).dropna()
        put(f'p6.auc.{f}.minus_margin.median', diff.median())
        put(f'p6.auc.{f}.beats_margin', (diff > 0).sum())
    # descriptive view of the same question: error rate of ACI's decisions by tenth of each
    # detector-free factor (within-dataset deciles), relative to the dataset's overall error rate
    rows = []
    for ds in C.DS:
        Fd = pd.read_csv(os.path.join(R, 'uncertainty', f'{ds}_factors.csv'))
        idx = Fd.test_index.values
        dec = {f: np.minimum(9, (Fd[f].rank(method='first', pct=True).values * 10 - 1e-9).astype(int))
               for f in ['distributional', 'density', 'temporal'] if Fd[f].nunique() > 1}
        wr = []
        for det in C.DETS:
            for seed in C.SEEDS:
                z = np.load(os.path.join(R, 'policies', f'{ds}_{det}_s{seed}.npz'))
                wr.append(z['aci_preds_a05'][idx] != z['y_test'][idx])
        wrong = np.mean(wr, axis=0)                               # share of the 15 runs that decide this window wrongly
        for f, k in dec.items():
            for q in range(10):
                rows.append(dict(dataset=ds, factor=f, decile=q + 1, rel_error=wrong[k == q].mean() / wrong.mean()))
    dec_df = pd.DataFrame(rows)
    dec_df.to_csv(os.path.join(S, 'p6_decile_error.csv'), index=False)
    for f, g in dec_df.groupby('factor'):
        put(f'p6.decile.{f}.top.median', g[g.decile == 10].rel_error.median())
        put(f'p6.decile.{f}.bottom.median', g[g.decile == 1].rel_error.median())
    # correlations between factors, unit = dataset
    c = d[d.kind == 'corr'].groupby(['dataset', 'factor']).value.mean().unstack()
    c.to_csv(os.path.join(S, 'p6_corr_per_dataset.csv'))
    for pair in c.columns:
        k = pair.replace('~', '_')
        put(f'p6.corr.{k}.median', c[pair].median())
        put(f'p6.corr.{k}.min', c[pair].min()); put(f'p6.corr.{k}.max', c[pair].max())
        put(f'p6.corr.{k}.n', c[pair].notna().sum())
    print('P6 AUC (unit = dataset):'); print(t.round(3).to_string(index=False))
    print('P6 Spearman medians:'); print(c.median().round(2).to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--parts', nargs='+', default=['p1', 'aci', 'p4', 'p3', 'p5', 'p6', 'case'])   # p3 reads p4_runs.csv
    ap.add_argument('--test_partial', metavar='DIR', help='testing only: summarise unfinished runs into DIR')
    a = ap.parse_args()
    env = A.check_env()
    global S, PARTIAL
    if a.test_partial:
        S, PARTIAL = a.test_partial, True
        os.makedirs(S, exist_ok=True)
        for f in glob.glob(os.path.join(S, '*.csv')):
            if not os.path.exists(os.path.join(S, os.path.basename(f))):
                shutil.copy(f, S)
    for p in a.parts:
        fn = globals().get(f'part_{p}')
        if fn is None:
            continue
        try:
            fn()
        except AssertionError as e:
            print(f'{p}: skipped ({e})')
    old = json.load(open(os.path.join(S, 'numbers.json'))) if os.path.exists(os.path.join(S, 'numbers.json')) else {}
    old.update(NUM)
    old['_environment'] = f"{sys.executable} scipy {env['scipy']} numpy {env['numpy']} pandas {env['pandas']}"
    json.dump(old, open(os.path.join(S, 'numbers.json'), 'w'), indent=1, sort_keys=True)
    print(f'{len(NUM)} values written to {S}/numbers.json')


if __name__ == '__main__':
    main()
