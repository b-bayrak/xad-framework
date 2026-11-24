"""
Build every number, table and figure of the paper from the run outputs.

All twelve datasets are treated identically in every table, figure and test.
The script

  * loads the results of the main and supplementary runs (runs/skab_swat/ for
    SKAB and SWaT, runs/benchmark/ for the ten added datasets, jobs as
    recorded in docs/RUNLOG.md) and the post-hoc summaries in
    results/posthoc_summary/;
  * refuses to finish if any experiment lacks one of the five seeds;
  * checks every qualitative statement of the paper listed in CLAIMS against
    the data;
  * writes results/paper/numbers.json (every value), results/paper/tables/
    (LaTeX table bodies) and results/paper/figures/ (PDF figures).

    bash analysis/run.sh paper_outputs.py [--allow_partial]
"""
import argparse
import glob
import json
import os
import re
import shutil
import sys

import numpy as np
import pandas as pd
from scipy import stats as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze as A  # noqa: E402  same loading, dedup, Wilcoxon and Holm code as the report

E, REPO = A.E, A.REPO
from paths import PREPARED, PREP_LOGS, RAW, RUNS_SKAB_SWAT  # noqa: E402
OUT = os.path.join(REPO, 'results', 'paper')
AUDIT = os.path.join(OUT, 'numbers.json')
DETS = A.DETS
DET = {'conv_ae': 'Conv-AE', 'lstm_ae': 'LSTM-AE', 'lstm_vae': 'LSTM-VAE'}
POL = {'static': 'Static', 'sliding': 'Sliding', 'aci': 'ACI'}
METH = {'GA-CE': 'gace', 'DICE': 'dice', 'PerCE': 'perce'}
VARIANT = {'Full GA-CE': 'full', 'No L_smooth': 'nosmooth', 'No L_spar': 'nospar', 'No L_prox': 'noprox'}
AK = ['05', '10', '15', '20']
AKEY = {0.05: '05', 0.10: '10', 0.15: '15', 0.20: '20'}
NAME = {'threew': '3W', 'daphnet': 'Daphnet', 'exathlon': 'Exathlon', 'gecco': 'GECCO', 'ghl': 'GHL',
        'metropt3': 'MetroPT-3', 'msl': 'MSL', 'psm': 'PSM', 'skab': 'SKAB', 'smap': 'SMAP',
        'smd': 'SMD', 'swat': 'SWaT'}
DS = list(NAME)                       # alphabetical by display name; one list, one order
DEV = ['skab', 'swat']                # used while developing the framework
PRE = [d for d in DS if d not in DEV]  # the ten datasets of the pre-specified analysis
DOMAIN = {'threew': 'Oil wells', 'daphnet': 'Wearable gait sensors', 'exathlon': 'Spark cluster',
          'gecco': 'Drinking-water quality', 'ghl': 'Gasoil heating loop (sim.)',
          'metropt3': 'Metro air compressor', 'msl': 'Mars rover telemetry', 'psm': 'Server metrics',
          'skab': 'Water-circulation testbed', 'smap': 'Satellite telemetry', 'smd': 'Server machines',
          'swat': 'Water treatment'}
SPLIT = {'threew': 'by instance', 'daphnet': 'official files', 'exathlon': 'official files',
         'gecco': 'official files', 'ghl': 'official files', 'metropt3': 'by date', 'msl': 'official files',
         'psm': 'official files', 'skab': 'by file, 60/20/20', 'smap': 'official files',
         'smd': 'official files', 'swat': 'chronological 60/20/20'}


# ------------------------------------------------------------------ data ---

def result_dirs():
    d = {}
    for ds in DEV:
        main = sorted(glob.glob(os.path.join(RUNS_SKAB_SWAT, 'results', f'{ds}_5seed_*')))[-1]
        job = main.rsplit('_', 1)[1]
        d[ds] = {'main': [main], 'supp': [os.path.join(RUNS_SKAB_SWAT, 'results', f'{ds}_supp_{job}')]}
    for ds, (m, _, _) in A.runlog_jobs().items():
        if ds in DS:
            d[ds] = {'main': [os.path.join(E, 'results', f'{ds}_5seed_{m}')],
                     'supp': [os.path.join(E, 'results', f'{ds}_supp_{m}')]}
    return d


def load_all():
    dirs, data, dups = result_dirs(), {}, []
    for prefix, spec in A.EXPS.items():
        frames = []
        for ds, dd in dirs.items():
            src = dd['main'] + dd['supp'] if spec['dir'] == 'both' else dd[spec['dir']]
            f = A.load(prefix, src, spec['ids'], dups)
            if f is not None:
                frames.append(f)
        data[prefix] = pd.concat(frames, ignore_index=True) if frames else None
    cls = sorted(glob.glob(os.path.join(E, 'results', 'classical_baselines', 'exp8_classical_*.csv')) +
                 glob.glob(os.path.join(RUNS_SKAB_SWAT, 'results', 'classical_baselines', 'exp8_classical_*.csv')))
    c = pd.concat([pd.read_csv(p).assign(_file=os.path.basename(p)) for p in cls])
    data['exp8'] = c.sort_values('_file').drop_duplicates(['dataset', 'method', 'seed'], keep='last')
    return data, dups


def completeness(data):
    gaps = []
    for prefix, col in [('exp1', None), ('exp4', 'cp_mode'), ('exp6_cp_ablation', 'cp_mode'), ('exp3', 'cp_mode'),
                        ('exp5_gace_ablation', 'cp_mode'), ('exp7_sensitivity', 'cp_mode')]:
        df = data[prefix]
        for ds in DS:
            for d in DETS:
                sub = df[(df.dataset == ds) & (df.detector == d)] if df is not None else pd.DataFrame()
                modes = ['aci'] if prefix == 'exp3' else (sorted(sub[col].dropna().unique()) if col and len(sub) else [None])
                for m in modes or [None]:
                    ss = sub if m is None or not len(sub) else sub[sub[col] == m]
                    n = ss.seed.nunique() if len(ss) else 0
                    if n < 5:
                        gaps.append(f'{prefix}/{ds}/{d}/{m}: {n} seeds')
    return gaps


def dataset_meta():
    cache = os.path.join(REPO, 'results', 'paper', 'dataset_windows.json')
    if os.path.exists(cache):
        return json.load(open(cache))
    from experiments import pipeline as pl
    out = {}
    for ds in DS:
        dp = {'skab': os.path.join(REPO, 'data', 'SKAB'), 'swat': os.path.join(REPO, 'data', 'SWaT')}.get(
            ds, os.path.join(PREPARED, ds))
        s = pl.prepare_split(ds, dp + '/', 0.2, 0.2, verbose=False)
        cfg = pl.DATASET_CONFIG[ds]
        out[ds] = dict(features=int(s['input_dim']), seq=cfg['sequence_length'],
                       stride=max(1, int(round(cfg['sequence_length'] * (1 - cfg['window_overlap'])))),
                       calib_window=cfg['calibration_window'], train_win=len(s['X_train_normal']),
                       val_normal=len(s['X_val_normal']), test_win=len(s['X_test']),
                       test_rate=float(s['y_test'].mean()))
    json.dump(out, open(cache, 'w'), indent=1)
    return out


# --------------------------------------------------------------- numbers ---

def build_numbers(data, meta):
    N = {}

    def put(k, v):
        if isinstance(v, str):
            N[k] = v
        elif v is not None and not (isinstance(v, float) and np.isnan(v)):
            N[k] = float(v)

    for ds, m in meta.items():
        for k, v in m.items():
            put(f'{ds}.meta.{k}', v)

    # detection
    e1 = A.agg(data['exp1'], ['dataset', 'detector'], ['f1_optimal', 'roc_auc', 'pr_auc', 'training_time'])
    for _, r in e1.iterrows():
        for m, k in [('f1_optimal', 'f1s'), ('roc_auc', 'roc'), ('pr_auc', 'pr'), ('training_time', 'train_time')]:
            put(f'{r.dataset}.{r.detector}.{k}', r[f'{m}_mean'])
            put(f'{r.dataset}.{r.detector}.{k}_sd', r[f'{m}_std'])
    t8 = data['exp8'].groupby(['dataset', 'method']).agg(
        roc=('roc_auc', 'mean'), roc_sd=('roc_auc', 'std'), pr=('pr_auc', 'mean'), pr_sd=('pr_auc', 'std'),
        f1s=('f1_optimal', 'mean')).reset_index()
    mk = {'IsolationForest': 'iforest', 'LOF': 'lof', 'OneClassSVM': 'ocsvm'}
    for _, r in t8.iterrows():
        for k in ['roc', 'roc_sd', 'pr', 'pr_sd', 'f1s']:
            put(f'{r.dataset}.{mk[r.method]}.{k}', r[k])

    # threshold policies (deployed settings)
    e6 = data['exp6_cp_ablation']
    dep = e6[(e6.smoothness_weight == A.DEPLOYED_SMOOTHNESS) &
             ((e6.cp_mode != 'aci') | np.isclose(e6.aci_gamma.astype(float), A.ACI_GAMMA))].copy()
    dep['gap'] = dep.coverage - dep.expected_coverage
    t6 = A.agg(dep, ['dataset', 'detector', 'cp_mode', 'alpha'],
               ['coverage', 'gap', 'calibration_error', 'f1', 'precision', 'recall', 'accuracy'])
    for _, r in t6.iterrows():
        b = f'{r.dataset}.{r.detector}.{r.cp_mode}.a{AKEY[round(r.alpha, 2)]}'
        for m, k in [('coverage', 'cov'), ('gap', 'gap'), ('calibration_error', 'err'), ('f1', 'f1'),
                     ('precision', 'prec'), ('recall', 'rec'), ('accuracy', 'acc')]:
            put(f'{b}.{k}', r[f'{m}_mean']); put(f'{b}.{k}_sd', r[f'{m}_std'])
    for (ds, pol, a), g_ in t6.groupby(['dataset', 'cp_mode', 'alpha']):
        for m, k in [('coverage', 'cov'), ('gap', 'gap'), ('f1', 'f1'), ('recall', 'rec'), ('accuracy', 'acc')]:
            put(f'{ds}.avg.{pol}.a{AKEY[round(a, 2)]}.{k}', g_[f'{m}_mean'].mean())
    # TVD term (ranking, policy-independent)
    tv = e6[(e6.cp_mode == 'static') & np.isclose(e6.alpha, 0.05)]
    tv = A.agg(tv, ['dataset', 'detector', 'smoothness_weight'], ['roc_auc_scores'])
    for _, r in tv.iterrows():
        put(f'{r.dataset}.{r.detector}.tvd.{str(r.smoothness_weight).replace(".", "p")}.roc', r.roc_auc_scores_mean)

    # counterfactuals
    t4 = A.agg(data['exp4'], ['dataset', 'detector', 'method', 'cp_mode'],
               ['validity_rate', 'mean_proximity', 'mean_sparsity', 'n_anomalies', 'mean_gen_time'])
    for _, r in t4.iterrows():
        b = f'{r.dataset}.{r.detector}.{r.cp_mode}.{METH[r.method]}'
        for m, k in [('validity_rate', 'valid'), ('mean_proximity', 'prox'), ('mean_sparsity', 'spar'),
                     ('n_anomalies', 'n'), ('mean_gen_time', 'time')]:
            put(f'{b}.{k}', r[f'{m}_mean']); put(f'{b}.{k}_sd', r[f'{m}_std'])
    for (ds, pol, meth), g_ in t4.groupby(['dataset', 'cp_mode', 'method']):
        put(f'{ds}.avg.{pol}.{METH[meth]}.valid', g_.validity_rate_mean.mean())

    # confidence tertiles under ACI
    if data['exp3'] is not None:
        e3 = data['exp3'][data['exp3'].cp_mode == 'aci']
        t3 = A.agg(e3, ['dataset', 'detector', 'confidence_level'], ['anomaly_precision', 'normal_accuracy'])
        for _, r in t3.iterrows():
            b = f'{r.dataset}.{r.detector}.conf.{r.confidence_level}'
            put(f'{b}.prec', r.anomaly_precision_mean); put(f'{b}.prec_sd', r.anomaly_precision_std)
            put(f'{b}.nacc', r.normal_accuracy_mean); put(f'{b}.nacc_sd', r.normal_accuracy_std)
        for (ds, lvl), g_ in t3.groupby(['dataset', 'confidence_level']):
            put(f'{ds}.avg.conf.{lvl}.prec', g_.anomaly_precision_mean.mean())
            put(f'{ds}.avg.conf.{lvl}.nacc', g_.normal_accuracy_mean.mean())

    # GA-CE ablation and policy sensitivity
    if data['exp5_gace_ablation'] is not None:
        t5 = A.agg(data['exp5_gace_ablation'], ['dataset', 'detector', 'cp_mode', 'variant'],
                   ['validity', 'proximity', 'sparsity', 'smoothness'])
        for _, r in t5.iterrows():
            b = f'{r.dataset}.{r.detector}.{r.cp_mode}.abl.{VARIANT[r.variant]}'
            for m in ['validity', 'proximity', 'sparsity', 'smoothness']:
                put(f'{b}.{m}', r[f'{m}_mean'])
    if data['exp7_sensitivity'] is not None:
        for (ds, prm, val, a), g_ in data['exp7_sensitivity'].groupby(['dataset', 'param', 'value', 'alpha']):
            put(f'{ds}.sens.{prm}.{str(float(val)).replace(".", "p")}.a{AKEY[round(a, 2)]}.err', g_.calibration_error.mean())
        for ds in DS:
            ws = [v for k, v in N.items() if re.match(rf'^{ds}\.sens\.window\.[^.]+\.a05\.err$', k)]
            if ws:
                put(f'{ds}.sens.window.best.a05', min(ws))

    # ---- statistics. Unit = dataset, seeds averaged first. 'all': twelve datasets (main text);
    # 'pre': the ten datasets of the analysis specified before the runs (appendix).
    iso = t8[t8.method == 'IsolationForest'].set_index('dataset')
    for G, lst in [('all', DS), ('pre', PRE)]:
        sub6 = t6[t6.dataset.isin(lst)]
        for q in ['cov', 'rec', 'f1']:
            keys = []
            for d in DETS:
                for a in A.ALPHAS:
                    pv = sub6[(sub6.detector == d) & np.isclose(sub6.alpha, a)].pivot(
                        index='dataset', columns='cp_mode', values=['gap_mean', 'recall_mean', 'f1_mean'])
                    x, y = {'cov': (pv[('gap_mean', 'aci')].abs(), pv[('gap_mean', 'sliding')].abs()),
                            'rec': (pv[('recall_mean', 'aci')], pv[('recall_mean', 'sliding')]),
                            'f1': (pv[('f1_mean', 'aci')], pv[('f1_mean', 'sliding')])}[q]
                    w = A.wilcoxon_row(x, y)
                    k = f'stat.{G}.{q}.{d}.a{AKEY[a]}'
                    keys.append(k)
                    for kk in ['median_diff', 'ci95_lo', 'ci95_hi', 'p', 'n_datasets']:
                        put(f'{k}.{kk}', w[kk])
                    if q == 'cov':
                        put(f'crit.{G}.{d}.a{AKEY[a]}.slide_under', (pv[('gap_mean', 'sliding')] < -A.CRIT_PTS).sum())
                        put(f'crit.{G}.{d}.a{AKEY[a]}.aci_within', (pv[('gap_mean', 'aci')].abs() <= A.CRIT_PTS).sum())
            for k, h in zip(keys, A.holm([N.get(k + '.p', np.nan) for k in keys])):
                put(k + '.pholm', h)
        sub4 = t4[t4.dataset.isin(lst)]
        keys = []
        for d in DETS:
            for meth, m in METH.items():
                pv = sub4[(sub4.detector == d) & (sub4.method == meth)].pivot(index='dataset', columns='cp_mode', values='validity_rate_mean')
                w = A.wilcoxon_row(pv['aci'], pv['sliding'])
                k = f'stat.{G}.cf.{d}.{m}'
                keys.append(k)
                for kk in ['median_diff', 'ci95_lo', 'ci95_hi', 'p', 'n_datasets']:
                    put(f'{k}.{kk}', w[kk])
            for pol in ['sliding', 'aci']:
                pv = sub4[(sub4.detector == d) & (sub4.cp_mode == pol)].pivot(index='dataset', columns='method', values='validity_rate_mean').dropna()
                r = st.friedmanchisquare(*[pv[c] for c in pv.columns])
                kk, n = pv.shape[1], pv.shape[0]
                put(f'stat.{G}.fried.{d}.{pol}.chi2', r.statistic); put(f'stat.{G}.fried.{d}.{pol}.p', r.pvalue)
                put(f'stat.{G}.fried.{d}.{pol}.cd', st.studentized_range.ppf(0.95, kk, np.inf) / np.sqrt(2) * np.sqrt(kk * (kk + 1) / (6 * n)))
                for c, v in pv.rank(axis=1, ascending=False).mean().items():
                    put(f'stat.{G}.fried.{d}.{pol}.rank.{METH[c]}', v)
        for k, h in zip(keys, A.holm([N.get(k + '.p', np.nan) for k in keys])):
            put(k + '.pholm', h)
        keys = []
        for d in DETS:
            e = e1[(e1.detector == d) & e1.dataset.isin(lst)].set_index('dataset')
            for m, km in [('roc_auc', 'roc'), ('pr_auc', 'pr')]:
                w = A.wilcoxon_row(e[f'{m}_mean'], iso.loc[e.index, km])
                k = f'stat.{G}.iso.{d}.{km}'
                keys.append(k)
                for kk in ['median_diff', 'ci95_lo', 'ci95_hi', 'p']:
                    put(f'{k}.{kk}', w[kk])
        for k, h in zip(keys, A.holm([N.get(k + '.p', np.nan) for k in keys])):
            put(k + '.pholm', h)

    # ---- descriptive counts over all twelve datasets
    cells = [(ds, d, a) for ds in DS for d in DETS for a in AK]
    gap = lambda ds, d, pol, a: N[f'{ds}.{d}.{pol}.a{a}.gap']
    put('count.cells', len(cells))
    put('count.slide_under3', sum(gap(ds, d, 'sliding', a) < -A.CRIT_PTS for ds, d, a in cells))
    put('count.aci_within3', sum(abs(gap(ds, d, 'aci', a)) <= A.CRIT_PTS for ds, d, a in cells))
    put('count.aci_over3', sum(gap(ds, d, 'aci', a) > A.CRIT_PTS for ds, d, a in cells))
    put('count.aci_under3', sum(gap(ds, d, 'aci', a) < -A.CRIT_PTS for ds, d, a in cells))
    put('count.aci_maxunder', -min(0, min(gap(ds, d, 'aci', a) for ds, d, a in cells)))
    put('count.aci_closer', sum(abs(gap(ds, d, 'aci', a)) < abs(gap(ds, d, 'sliding', a)) for ds, d, a in cells))
    put('count.slide_rec_higher', sum(N[f'{ds}.{d}.sliding.a{a}.rec'] > N[f'{ds}.{d}.aci.a{a}.rec'] for ds, d, a in cells))
    over = sorted({ds for ds, d, a in cells if gap(ds, d, 'aci', a) > A.CRIT_PTS})
    put('names.aci_over3', ' and '.join(NAME[x] for x in over))
    put('count.static_within3_avg.a05', sum(abs(N[f'{ds}.avg.static.a05.gap']) <= A.CRIT_PTS for ds in DS))
    for a in ['05', '20']:
        best = {ds: max(['static', 'sliding', 'aci'], key=lambda p: N[f'{ds}.avg.{p}.a{a}.f1']) for ds in DS}
        for pol in POL:
            put(f'count.bestf1.a{a}.{pol}', sum(v == pol for v in best.values()))
    for ds in DS:
        put(f'{ds}.best_deep_roc', max(N[f'{ds}.{d}.roc'] for d in DETS))
        put(f'{ds}.best_classical_roc', max(N[f'{ds}.{m}.roc'] for m in ['iforest', 'lof', 'ocsvm']))
    clsb = [ds for ds in DS if N[f'{ds}.best_classical_roc'] > N[f'{ds}.best_deep_roc']]
    put('count.classical_best', len(clsb))
    put('names.classical_best', ', '.join(NAME[x] for x in clsb))
    for d in DETS:
        put(f'count.{d}.beats_iforest', sum(N[f'{ds}.{d}.roc'] > N[f'{ds}.iforest.roc'] for ds in DS))
    put('count.lstm_vae_weakest', sum(min(DETS, key=lambda d: N[f'{ds}.{d}.roc']) == 'lstm_vae' for ds in DS))
    up = [N[f'{ds}.{d}.tvd.0p1.roc'] - N[f'{ds}.{d}.tvd.0p0.roc'] for ds in DS for d in DETS]
    put('count.tvd.improves', sum(x > 0.005 for x in up)); put('count.tvd.hurts', sum(x < -0.005 for x in up))
    put('count.tvd.pairs', len(up)); put('median.tvd.delta', float(np.median(up)))
    cfc = [(ds, d) for ds in DS for d in DETS]
    put('count.cf.cells', len(cfc))
    for m in METH.values():
        put(f'count.cf.{m}.aci_higher', sum(N[f'{ds}.{d}.aci.{m}.valid'] > N[f'{ds}.{d}.sliding.{m}.valid'] for ds, d in cfc))
    bestm = lambda ds, d, k, f: f(METH.values(), key=lambda m: N[f'{ds}.{d}.aci.{m}.{k}'])
    put('count.cf.perce_prox_min', sum(bestm(ds, d, 'prox', min) == 'perce' for ds, d in cfc))
    put('count.cf.gace_prox_max', sum(bestm(ds, d, 'prox', max) == 'gace' for ds, d in cfc))
    put('count.cf.dice_spar_min', sum(bestm(ds, d, 'spar', min) == 'dice' for ds, d in cfc))
    put('count.cf.gace_time_max', sum(bestm(ds, d, 'time', max) == 'gace' for ds, d in cfc))
    cc = [(ds, d) for ds, d in cfc if f'{ds}.{d}.conf.high.prec' in N and f'{ds}.{d}.conf.low.prec' in N]
    put('count.conf.cells', len(cc))
    put('count.conf.high_gt_low', sum(N[f'{ds}.{d}.conf.high.prec'] > N[f'{ds}.{d}.conf.low.prec'] for ds, d in cc))
    lv = ['low', 'medium', 'high']
    flat = [ds for ds in DS if all(f'{ds}.avg.conf.{l}.nacc' in N for l in lv)
            and min(N[f'{ds}.avg.conf.{l}.nacc'] for l in lv) >= 0.80
            and max(N[f'{ds}.avg.conf.{l}.nacc'] for l in lv) - min(N[f'{ds}.avg.conf.{l}.nacc'] for l in lv) <= 0.10]
    put('count.conf.nacc_flat', len(flat))
    put('count.sens.window_never_close', sum(N.get(f'{ds}.sens.window.best.a05', 0) > A.CRIT_PTS for ds in DS))
    tw = sorted(N[f'{ds}.meta.test_win'] for ds in DS)
    put('meta.test_win.min', tw[0]); put('meta.test_win.max', tw[-1])
    tt = [N[f'{ds}.{d}.train_time'] for ds in DS for d in DETS]
    put('train_time.min', min(tt)); put('train_time.max', max(tt)); put('train_time.median', float(np.median(tt)))

    # ---- leave-one-dataset-out (exploratory) over the twelve datasets
    def loo(drop):
        keep = [d for d in DS if d not in drop]

        def wp(x, y):
            x, y = np.asarray(x, float), np.asarray(y, float)
            return float(st.wilcoxon(x, y).pvalue) if len(x) >= 5 and np.any(x != y) else np.nan
        pc, pr_ = [], []
        for d in DETS:
            for a in A.ALPHAS:
                pv = t6[(t6.detector == d) & np.isclose(t6.alpha, a) & t6.dataset.isin(keep)].pivot(
                    index='dataset', columns='cp_mode', values=['gap_mean', 'recall_mean'])
                pc.append(wp(pv[('gap_mean', 'aci')].abs(), pv[('gap_mean', 'sliding')].abs()))
                pr_.append(wp(pv[('recall_mean', 'aci')], pv[('recall_mean', 'sliding')]))
        cf = []
        for d in DETS:
            for meth in METH:
                pv = t4[(t4.detector == d) & (t4.method == meth) & t4.dataset.isin(keep)].pivot(
                    index='dataset', columns='cp_mode', values='validity_rate_mean')
                cf.append((meth, wp(pv['aci'], pv['sliding'])))
        hc = A.holm([x[1] for x in cf])
        row = {'n': len(keep), 'cov': float(np.nanmax(A.holm(pc))), 'rec': float(np.nanmax(A.holm(pr_))),
               'gace': float(max(h for (m, _), h in zip(cf, hc) if m == 'GA-CE'))}
        for d in DETS:
            e = e1[(e1.detector == d) & e1.dataset.isin(keep)].set_index('dataset')
            row[f'iso_{d}'] = float(np.median(e.roc_auc_mean - iso.loc[e.index, 'roc']))
        return row
    rows_loo = [('none', loo([]))] + [(ds, loo([ds])) for ds in DS] + [('smap+daphnet', loo(['smap', 'daphnet']))]
    for name, r in rows_loo:
        for k, v in r.items():
            put(f'loo.{name}.{k}', v)
    single = [r for name, r in rows_loo if name in DS]
    put('loo.single.cov_max', max(r['cov'] for r in single)); put('loo.single.rec_max', max(r['rec'] for r in single))
    put('loo.single.gace_max', max(r['gace'] for r in single)); put('loo.single.gace_min', min(r['gace'] for r in single))
    put('loo.single.gace_sig', sum(r['gace'] < 0.05 for r in single))

    # ---- facts about data preparation (from logs and raw files)
    v1 = pd.read_csv(os.path.join(PREP_LOGS, 'verify_windows_v1.tsv'), sep='\t').set_index('dataset')
    put('excl.swan.test_anom_rate', v1.loc['swan_sf', 'test_anom_rate'])
    shares, nrows = [], 0
    for f in sorted(glob.glob(os.path.join(RAW, 'pronto', 'PRONTO benchmark case study', 'Pre-processed data',
                                           'Aligned and labelled alarm and process data', 'Testday*_merged.csv'))):
        dd = pd.read_csv(f, usecols=['Fault'])
        nrows += len(dd)
        sh = (~dd.Fault.isin(['Normal', 'Slugging'])).mean()
        if sh > 0:
            shares.append(sh)
    if shares:   # PRONTO raw files present (bash datasets/download.sh pronto)
        put('excl.pronto.rows', nrows); put('excl.pronto.fault_min', min(shares)); put('excl.pronto.fault_max', max(shares))
    log = open(os.path.join(PREP_LOGS, 'threew.log')).read()
    put('threew.masked_cells', int(re.search(r'implausible cells set to missing: (\d+)', log).group(1)))
    put('threew.truncated_rows', int(re.search(r'onset\): (\d+)', log).group(1)))
    mt = json.load(open(os.path.join(PREPARED, 'threew', 'meta.json')))
    put('threew.imputed_frac', mt['nan_cells_filled'] / (mt['n_rows'] * mt['n_features']))
    put('threew.instances', mt['n_recordings'])
    put('threew.raw_anomalous_share', json.load(open(os.path.join(REPO, 'results', 'paper', 'threew_raw_stats.json')))['raw_anomalous_share'])
    put('exathlon.meta.dropped', len(json.load(open(os.path.join(PREPARED, 'exathlon', 'meta.json')))['dropped_features_constant_on_train']))
    return N


# --------------------------------------------------------------- post hoc ---
# Post-hoc analyses P1-P6 and the exploratory diagnostics (ANALYSIS_PLAN.md), summarised
# by posthoc_analyze.py into results/posthoc/. A final build needs every part.

PHS = os.path.join(REPO, 'results', 'posthoc')
PH_PARTS = {'p1': 'p1.cells', 'aci': 'aci.runs', 'p3': 'p3.time_of_detection.all.gace.n', 'p4': 'p4.runs',
            'p5': 'p5.cf.stat.gace.pholm', 'p6': 'p6.runs', 'case': 'cs.stream.n'}


def posthoc_numbers(N):
    ph = json.load(open(os.path.join(PHS, 'numbers.json')))
    clash = set(ph) & set(N)
    if clash:
        raise SystemExit(f'post-hoc keys collide with main keys: {sorted(clash)[:5]}')
    N.update(ph)
    return [p for p, k in PH_PARTS.items() if k not in N]


# ---------------------------------------------------------------- tables ---

def pfmt(p):
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return '--'
    return '$<$0.001' if p < 0.001 else f'{p:.3f}'


def fv(v, d=3, sc=1.0):
    return '--' if v is None or (isinstance(v, float) and np.isnan(v)) else f'{v * sc:.{d}f}'


def fpm(N, k, d=3, sc=1.0):
    m, s = N.get(k), N.get(k + '_sd')
    if m is None:
        return '--'
    if s is None or np.isnan(s):
        return f'{m * sc:.{d}f}'
    return f'{m * sc:.{d}f}{{\\scriptsize$\\pm${s * sc:.{d}f}}}'


def rows(fn, per_detector=False):
    out = []
    for i, ds in enumerate(DS):
        if per_detector:
            for j, d in enumerate(DETS):
                lead = f'\\multirow{{3}}{{*}}{{{NAME[ds]}}}' if j == 0 else ''
                out.append(f'{lead} & {DET[d]} & {fn(ds, d)} \\\\')
            if i < len(DS) - 1:
                out.append('\\midrule')
        else:
            out.append(f'{NAME[ds]} & {fn(ds)} \\\\')
    return '\n'.join(out)


def conf_cell(N, b):
    v, s = N.get(b), N.get(b + '_sd')
    if v is None:
        return '--'
    return f'{100 * v:.1f}' if s is None or np.isnan(s) else f'{100 * v:.1f}{{\\scriptsize$\\pm${100 * s:.1f}}}'


def tables(N):
    T = {}
    avg = lambda ks: np.nanmean([N.get(k, np.nan) for k in ks])
    T['datasets'] = rows(lambda ds: f'{DOMAIN[ds]} & {int(N[f"{ds}.meta.features"])} & '
                         f'{int(N[f"{ds}.meta.seq"])}/{int(N[f"{ds}.meta.stride"])} & {SPLIT[ds]} & '
                         f'{int(N[f"{ds}.meta.train_win"]):,} & {int(N[f"{ds}.meta.val_normal"]):,} & '
                         f'{int(N[f"{ds}.meta.test_win"]):,} & {100 * N[f"{ds}.meta.test_rate"]:.1f}')
    T['det'] = rows(lambda ds: f'{100 * N[f"{ds}.meta.test_rate"]:.1f} & ' +
                    ' & '.join(fpm(N, f'{ds}.{d}.roc') for d in DETS) + ' & ' +
                    ' & '.join(fpm(N, f'{ds}.{d}.pr') for d in DETS) + ' & ' +
                    f'{fv(N.get(f"{ds}.iforest.roc"))} & {fv(N.get(f"{ds}.lof.roc"))}')
    T['f1s'] = rows(lambda ds: ' & '.join(fpm(N, f'{ds}.{d}.f1s') for d in DETS) + f' & {fv(N.get(f"{ds}.iforest.f1s"))}')

    def cal(ds):
        c = [fv(avg([f'{ds}.{d}.aci.a{a}.cov' for d in DETS])) for a in AK]
        errs = [N[f'{ds}.{d}.aci.a{a}.err'] for d in DETS for a in AK]
        gaps = [N[f'{ds}.{d}.aci.a{a}.gap'] for d in DETS for a in AK]
        return ' & '.join(c) + f' & {fv(max(errs))} & {fv(-min(0, min(gaps)))} & {fv(avg([f"{ds}.{d}.aci.a05.acc" for d in DETS]))}'
    T['cal'] = rows(cal)
    T['cal_full'] = rows(lambda ds, d: ' & '.join(fpm(N, f'{ds}.{d}.aci.a{a}.cov') for a in AK), True)
    for a in ['05', '20']:
        T[f'pol_a{a}'] = rows(lambda ds, a=a: ' & '.join(fv(N.get(f'{ds}.avg.{p}.a{a}.{m}')) for p in POL for m in ['cov', 'rec', 'f1']))
        T[f'pol_full_a{a}'] = rows(lambda ds, d, a=a: ' & '.join(fpm(N, f'{ds}.{d}.{p}.a{a}.{m}') for p in POL for m in ['cov', 'f1']), True)

    def statcells(G, d, a, ci=True):
        c = []
        for q in ['cov', 'rec', 'f1']:
            b = f'stat.{G}.{q}.{d}.a{a}'
            md = f'${N[b + ".median_diff"]:+.3f}$'
            if ci:
                md += f' [${N[b + ".ci95_lo"]:+.3f}$, ${N[b + ".ci95_hi"]:+.3f}$]'
            c.append(f'{md} & {pfmt(N[b + ".pholm"])}')
        return ' & '.join(c)
    T['stats'] = '\n'.join(f'{DET[d]} & {statcells("all", d, "05")} & {int(N[f"crit.all.{d}.a05.slide_under"])} & '
                           f'{int(N[f"crit.all.{d}.a05.aci_within"])} \\\\' for d in DETS)
    out = []
    for d in DETS:
        for i, a in enumerate(AK):
            lead = f'\\multirow{{4}}{{*}}{{{DET[d]}}}' if i == 0 else ''
            out.append(f'{lead} & 0.{a} & {statcells("all", d, a, False)} & {int(N[f"crit.all.{d}.a{a}.slide_under"])} & '
                       f'{int(N[f"crit.all.{d}.a{a}.aci_within"])} \\\\')
        out.append('\\midrule')
    T['stats_all'] = '\n'.join(out[:-1])
    T['pre'] = '\n'.join(f'{DET[d]} & {statcells("pre", d, "05", False)} & ' +
                         ' & '.join(pfmt(N[f'stat.pre.cf.{d}.{m}.pholm']) for m in METH.values()) + ' \\\\' for d in DETS)
    T['conf'] = rows(lambda ds: ' & '.join(fv(N.get(f'{ds}.avg.conf.{l}.prec'), 1, 100) for l in ['low', 'medium', 'high']) + ' & ' +
                     ' & '.join(fv(N.get(f'{ds}.avg.conf.{l}.nacc'), 1, 100) for l in ['low', 'medium', 'high']))
    T['conf_full'] = rows(lambda ds, d: ' & '.join(conf_cell(N, f'{ds}.{d}.conf.{l}.prec') for l in ['low', 'medium', 'high']) + ' & ' +
                          ' & '.join(conf_cell(N, f'{ds}.{d}.conf.{l}.nacc') for l in ['low', 'medium', 'high']), True)
    T['cf'] = rows(lambda ds: ' & '.join(fv(N.get(f'{ds}.avg.{p}.{m}.valid'), 1, 100) for m in METH.values() for p in ['aci', 'sliding']) +
                   f' & {fv(avg([f"{ds}.{d}.aci.{m}.n" for d in DETS for m in METH.values()]), 0)}')
    T['cf_stats'] = '\n'.join(f'{DET[d]} & ' + ' & '.join(f'${100 * N[f"stat.all.cf.{d}.{m}.median_diff"]:+.1f}$ & '
                                                          f'{pfmt(N[f"stat.all.cf.{d}.{m}.pholm"])}' for m in METH.values()) + ' & \\\\' for d in DETS)

    def cffull(ds, d):
        dp = 1 if ds == 'swat' else 2
        c = []
        for m in METH.values():
            c += [fv(N.get(f'{ds}.{d}.aci.{m}.valid'), 1, 100), fv(N.get(f'{ds}.{d}.aci.{m}.prox'), dp), fv(N.get(f'{ds}.{d}.aci.{m}.spar'), 1, 100)]
        return ' & '.join(c + [fv(N.get(f'{ds}.{d}.sliding.{m}.valid'), 1, 100) for m in METH.values()])
    T['cf_full'] = rows(cffull, True)
    for pol in ['sliding', 'aci']:
        def abl(ds, pol=pol):
            k = lambda v, m, d: N.get(f'{ds}.{d}.{pol}.abl.{v}.{m}', np.nan)
            ok = lambda m: [d for d in DETS if k('full', m, d) > 0]
            rel = lambda v, m: np.nanmedian([k(v, m, d) / k('full', m, d) - 1 for d in ok(m)]) if ok(m) else np.nan
            sg = lambda v: '--' if np.isnan(v) else f'${100 * v:+.1f}$'
            return (f'{fv(avg([f"{ds}.{d}.{pol}.abl.full.validity" for d in DETS]), 1, 100)} & '
                    f'{fv(avg([f"{ds}.{d}.{pol}.abl.full.sparsity" for d in DETS]), 1, 100)} & '
                    f'{sg(np.nanmean([k("nospar", "sparsity", d) - k("full", "sparsity", d) for d in DETS]))} & '
                    f'{sg(rel("noprox", "proximity"))}\\% & '
                    f'{sg(np.nanmean([k("noprox", "validity", d) - k("full", "validity", d) for d in DETS]))} & '
                    f'{sg(rel("nosmooth", "smoothness"))}\\%')
        T[f'abl_{pol}'] = rows(abl)
        cl = [(ds, d) for ds in DS for d in DETS if f'{ds}.{d}.{pol}.abl.full.validity' in N]
        for v, (m, sgn) in {'nosmooth': ('smoothness', 1), 'nospar': ('sparsity', -1), 'noprox': ('proximity', 1)}.items():
            N[f'abl.{pol}.{v}.target_worse'] = float(sum(
                sgn * (N[f'{ds}.{d}.{pol}.abl.{v}.{m}'] - N[f'{ds}.{d}.{pol}.abl.full.{m}']) > 0 for ds, d in cl))
        N[f'abl.{pol}.cells'] = float(len(cl))
        if cl:
            N[f'abl.{pol}.noprox.dvalid'] = float(np.mean([N[f'{ds}.{d}.{pol}.abl.noprox.validity'] -
                                                           N[f'{ds}.{d}.{pol}.abl.full.validity'] for ds, d in cl]))
    gam, win = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05], [100, 250, 500, 1000, 2000]
    for a in ['05', '20']:
        T[f'sens_a{a}'] = rows(lambda ds, a=a: ' & '.join(fv(N.get(f'{ds}.sens.gamma.{str(float(v)).replace(".", "p")}.a{a}.err')) for v in gam) +
                               ' & ' + ' & '.join(fv(N.get(f'{ds}.sens.window.{str(float(v)).replace(".", "p")}.a{a}.err')) for v in win))
    T['classical'] = rows(lambda ds: f'{fpm(N, f"{ds}.iforest.roc")} & {fv(N.get(f"{ds}.lof.roc"))} & {fv(N.get(f"{ds}.ocsvm.roc"))} & '
                          f'{fpm(N, f"{ds}.iforest.pr")} & {fv(N.get(f"{ds}.lof.pr"))} & {fv(N.get(f"{ds}.ocsvm.pr"))}')
    out = []
    for name in ['none'] + DS + ['smap+daphnet']:
        lab = 'none' if name == 'none' else ('SMAP and Daphnet' if name == 'smap+daphnet' else NAME[name])
        b = f'loo.{name}'
        out.append(f'{lab} & {int(N[b + ".n"])} & {pfmt(N[b + ".cov"])} & {pfmt(N[b + ".rec"])} & {pfmt(N[b + ".gace"])} & ' +
                   ' & '.join(f'${N[b + f".iso_{d}"]:+.3f}$' for d in DETS) + ' \\\\')
    T['loo'] = '\n'.join(out)
    posthoc_tables(T)
    return T


def sg2(v, d=2):
    return '--' if v is None or (isinstance(v, float) and np.isnan(v)) else f'${v:+.{d}f}$'


def posthoc_tables(T):
    """Appendix tables of the post-hoc analyses, read from results/posthoc/. A table is
    only produced when its summary exists."""
    f = lambda n: os.path.join(PHS, n)
    if os.path.exists(f('p1_policies.csv')):
        g = pd.read_csv(f('p1_policies.csv')).groupby(['dataset', 'policy', 'alpha']).mean(numeric_only=True)
        for a, ak in [(0.05, '05'), (0.20, '20')]:
            def r(ds, a=a):
                v = lambda p, c: g.loc[(ds, p, a), c] if (ds, p, a) in g.index else np.nan
                return (' & '.join(fv(v(p, 'coverage'), 2) for p in ['static', 'sliding', 'sliding_all', 'aci']) + ' & ' +
                        ' & '.join(sg2(v(p, 'recall_minus_oracle')) for p in ['sliding', 'sliding_all', 'aci']))
            T[f'p1_a{ak}'] = rows(r)
    if os.path.exists(f('aci_trace_per_dataset.csv')):
        g = pd.read_csv(f('aci_trace_per_dataset.csv')).set_index(['dataset', 'alpha'])
        T['aci_trace'] = rows(lambda ds: ' & '.join(fv(g.loc[(ds, a), 'share_inf_pos'], 1, 100) for a in A.ALPHAS) + ' & ' +
                              f"{fv(g.loc[(ds, 0.05), 'anom_under_inf'], 1, 100)} & {fv(g.loc[(ds, 0.05), 'recall'], 3)} & "
                              f"{fv(g.loc[(ds, 0.05), 'recall_finite'], 3)}")
    if os.path.exists(f('p5_policies.csv')) and os.path.exists(f('p5_cf_runs.csv')):
        g = pd.read_csv(f('p5_policies.csv')).set_index(['dataset', 'policy', 'alpha'])
        c = pd.read_csv(f('p5_cf_runs.csv')).groupby(['dataset', 'policy', 'method']).validity.mean()
        T['iforest'] = rows(lambda ds: f"{fv(g.loc[(ds, 'static', 0.05), 'roc_auc'], 3)} & " +
                            ' & '.join(fv(g.loc[(ds, p, 0.05), 'coverage'], 2) for p in ['static', 'sliding', 'sliding_all', 'aci']) + ' & ' +
                            ' & '.join(fv(c.get((ds, p, m), np.nan), 1, 100) for m in ['GA-CE', 'PerCE'] for p in ['sliding', 'aci']))
    if os.path.exists(f('p6_auc_per_dataset.csv')):
        g = pd.read_csv(f('p6_auc_per_dataset.csv'), index_col=0)
        T['p6'] = rows(lambda ds: ' & '.join(fv(g.loc[ds, k], 2).replace('--', '') for k in ['distributional', 'density', 'temporal', 'feature_oor', 'feature', 'margin']))
    if os.path.exists(f('p4_cells.csv')):
        g = pd.read_csv(f('p4_cells.csv')).groupby(['dataset', 'policy', 'method']).mean(numeric_only=True)
        v = lambda ds, p, m, k: g.loc[(ds, p, m), k] if (ds, p, m) in g.index else np.nan
        T['cf_tod'] = rows(lambda ds: ' & '.join(fv(v(ds, p, m, 'validity'), 1, 100) for m in METH for p in ['aci', 'sliding']) +
                           f" & {fv(np.nanmean([v(ds, 'aci', m, 'n') for m in METH]), 0)}")
        T['cf_plaus'] = rows(lambda ds: ' & '.join(fv(v(ds, 'aci', m, 'cf_plaus_valid'), 2) for m in METH) + ' & ' +
                             ' & '.join(fv(v(ds, 'aci', m, 'cf_plaus_le1'), 0, 100) for m in METH) +
                             f" & {fv(v(ds, 'aci', 'GA-CE', 'anom_plaus'), 2)}")
        t = pd.read_csv(f('p4_validity_tests.csv'))
        T['cf_tod_stats'] = '\n'.join(f'{DET[d]} & ' + ' & '.join(
            f"${100 * t[(t.detector == d) & (t.method == m)].median_diff.iloc[0]:+.1f}$ & {pfmt(t[(t.detector == d) & (t.method == m)].p_holm.iloc[0])}"
            for m in METH) + ' & \\\\' for d in DETS)
    if os.path.exists(f('p3_validity_vs_pass_rate.csv')):
        t = pd.read_csv(f('p3_validity_vs_pass_rate.csv'))
        lab = {('end_of_stream', 'all'): 'End of stream', ('end_of_stream', 'finite'): 'End of stream, finite threshold',
               ('time_of_detection', 'all'): 'Time of detection'}
        out = []
        for (pr, sub), L in lab.items():
            x = t[(t.protocol == pr) & (t.subset == sub)]
            for i, m in enumerate(METH):
                r = x[x.method == m]
                if not len(r):
                    continue
                r = r.iloc[0]
                out.append(f"{L if i == 0 else ''} & {m} & {int(r.n)} & {r.rho:.2f} & ${r.b_pass:+.2f}$ & {pfmt(r.p_pass)} & "
                           f"${r.b_aci:+.2f}$ & {pfmt(r.p_aci)} \\\\")
            out.append('\\midrule')
        T['p3'] = '\n'.join(out[:-1])


# --------------------------------------------------------------- figures ---

def figures(N, outdir):
    import paper_figures as F
    F.style()
    F.fig_detection(N, outdir, DS, NAME, DETS, DET)
    F.fig_mechanism(N, outdir, NAME, DS)
    F.fig_policies(N, outdir)
    F.fig_tests(N, outdir, DETS, DET)
    F.fig_confidence(N, outdir, DS, NAME)
    F.fig_uncertainty(N, outdir, NAME)
    if os.path.exists(os.path.join(PHS, 'p4_cells.csv')):
        F.fig_cf(N, outdir, DS, NAME, DETS, DET)
    if os.path.exists(os.path.join(PHS, 'case_selection.json')):
        F.fig_stream(N, outdir)
        F.fig_cf_example(N, outdir)


# ---------------------------------------------------------------- claims ---
# Every qualitative statement in the text that is not itself a rendered number.

CLAIMS = [
    ('SKAB LSTM-VAE below chance', lambda N: N['skab.lstm_vae.roc'] < 0.5),
    ('SMAP: every detector and Isolation Forest below chance', lambda N: all(N[f'smap.{d}.roc'] < 0.5 for d in DETS) and N['smap.iforest.roc'] < 0.5),
    ('GECCO recurrent detectors at chance', lambda N: all(0.45 <= N[f'gecco.{d}.roc'] <= 0.55 for d in ['lstm_ae', 'lstm_vae'])),
    ('MetroPT-3 easiest dataset', lambda N: max(DS, key=lambda ds: N[f'{ds}.best_deep_roc']) == 'metropt3'),
    ('SWaT, SMD, GHL/LSTM-AE separated well', lambda N: all(N[f'{ds}.{d}.roc'] >= 0.75 for ds in ['swat', 'smd'] for d in DETS) and N['ghl.lstm_ae.roc'] >= 0.75),
    ('deep vs IF not significant (12 datasets)', lambda N: all(N[f'stat.all.iso.{d}.{m}.pholm'] > 0.05 for d in DETS for m in ['roc', 'pr'])),
    ('LSTM-VAE weakest on most datasets', lambda N: N['count.lstm_vae_weakest'] > 6),
    ('RQ1: each architecture ranks best on at least one dataset', lambda N:
     {max(DETS, key=lambda d: N[f'{ds}.{d}.roc']) for ds in DS} == set(DETS)),
    ('sliding under-covers >3 pts in every cell', lambda N: N['count.slide_under3'] == N['count.cells']),
    ('ACI misses are all over-coverage', lambda N: N['count.aci_under3'] == 0),
    ('coverage and recall differences significant everywhere (12)', lambda N: all(N[f'stat.all.{q}.{d}.a{a}.pholm'] < 0.05 for q in ['cov', 'rec'] for d in DETS for a in AK)),
    ('F1 difference never significant (12)', lambda N: all(N[f'stat.all.f1.{d}.a{a}.pholm'] > 0.05 for d in DETS for a in AK)),
    ('same conclusions on the pre-specified ten', lambda N: all(N[f'stat.pre.{q}.{d}.a{a}.pholm'] < 0.05 for q in ['cov', 'rec'] for d in DETS for a in AK)
     and all(N[f'stat.pre.f1.{d}.a{a}.pholm'] > 0.05 for d in DETS for a in AK) and all(N[f'stat.pre.cf.{d}.gace.pholm'] < 0.05 for d in DETS)),
    ('GA-CE validity gain significant for every detector (12)', lambda N: all(N[f'stat.all.cf.{d}.gace.pholm'] < 0.05 for d in DETS)),
    ('DICE least sparse everywhere', lambda N: N['count.cf.dice_spar_min'] == N['count.cf.cells']),
    ('only SKAB below the 100-anomaly cap', lambda N: all(min(N[f'{ds}.{d}.{p}.{m}.n'] for d in DETS for m in METH.values()) >= 100
                                                       for ds in DS if ds != 'skab' for p in ['aci', 'sliding'])),
    ('static collapses on GECCO and Exathlon', lambda N: N['gecco.avg.static.a05.cov'] < 0.5 and N['exathlon.avg.static.a05.cov'] < 0.5),
    ('static best F1 on MetroPT-3; SWaT LSTMs static best, Conv-AE near zero', lambda N:
     N['metropt3.avg.static.a05.f1'] > max(N['metropt3.avg.aci.a05.f1'], N['metropt3.avg.sliding.a05.f1']) and
     all(N[f'swat.{d}.static.a05.f1'] > max(N[f'swat.{d}.sliding.a05.f1'], N[f'swat.{d}.aci.a05.f1']) for d in ['lstm_ae', 'lstm_vae']) and
     N['swat.conv_ae.static.a05.f1'] < 0.05),
    ('confidence gradients clear on SMD, PSM, 3W', lambda N: all(N[f'{ds}.avg.conf.high.prec'] - N[f'{ds}.avg.conf.low.prec'] > 0.3 for ds in ['smd', 'psm', 'threew'])),
    ('alert precision < 25% in every tertile on GHL, MetroPT-3, GECCO', lambda N: all(N[f'{ds}.avg.conf.{l}.prec'] < 0.25 for ds in ['ghl', 'metropt3', 'gecco'] for l in ['low', 'medium', 'high'])),
    ('MetroPT-3 detectors rank well', lambda N: all(N[f'metropt3.{d}.roc'] > 0.9 for d in DETS)),
    ('high > low alert precision in most cells', lambda N: N['count.conf.high_gt_low'] > N['count.conf.cells'] / 2),
    ('normal accuracy flat and high on most datasets', lambda N: N['count.conf.nacc_flat'] > 6),
    ('SWaT LSTM high-confidence normals unreliable', lambda N: all(N[f'swat.{d}.conf.high.nacc'] < 0.60 for d in ['lstm_ae', 'lstm_vae'])),
    ('SKAB: Conv-AE and LSTM-AE top-third alerts all correct in every seed; LSTM-VAE has no top-third alert', lambda N:
     all(N[f'skab.{d}.conf.high.prec'] == 1.0 and N[f'skab.{d}.conf.high.prec_sd'] == 0 for d in ['conv_ae', 'lstm_ae'])
     and 'skab.lstm_vae.conf.high.prec' not in N and 'skab.lstm_vae.conf.low.prec' in N),
    ('TVD: SKAB ROC decreases monotonically', lambda N: all(N[f'skab.{d}.tvd.0p0.roc'] > N[f'skab.{d}.tvd.0p01.roc'] > N[f'skab.{d}.tvd.0p1.roc'] for d in DETS)),
    ('sliding never close on most datasets', lambda N: N['count.sens.window_never_close'] > 6),
    ('ACI error small for every gamma', lambda N: max(v for k, v in N.items() if re.match(r'^[a-z0-9]+\.sens\.gamma\.[^.]+\.a05\.err$', k)) <= 0.07),
    ('removing L_prox raises validity on average', lambda N: N['abl.sliding.noprox.dvalid'] > 0 and N['abl.aci.noprox.dvalid'] > 0),
    ('LOO: coverage stays significant after any single removal', lambda N: N['loo.single.cov_max'] < 0.05),
    ('LOO: GA-CE stays significant after any single removal', lambda N: N['loo.single.gace_max'] < 0.05),
    ('LOO: SMAP+Daphnet removal keeps cov, rec, GA-CE significant; AEs fall further behind IF', lambda N:
     all(N[f'loo.smap+daphnet.{k}'] < 0.05 for k in ['cov', 'rec', 'gace']) and
     all(N[f'loo.smap+daphnet.iso_{d}'] < N[f'loo.none.iso_{d}'] for d in DETS)),
    ('sliding gap smallest on SKAB, SKAB test stream shortest', lambda N:
     min(DS, key=lambda ds: np.mean([abs(N[f'{ds}.{d}.sliding.a{a}.gap']) for d in DETS for a in AK])) == 'skab' and
     min(DS, key=lambda ds: N[f'{ds}.meta.test_win']) == 'skab'),
    ('SKAB: clear confidence gradient (detector average)', lambda N: N['skab.avg.conf.high.prec'] - N['skab.avg.conf.low.prec'] > 0.3),
    ('confidence caption: exactly two empty thirds, SKAB/LSTM-VAE high (alerts) and GECCO/LSTM-AE medium', lambda N:
     sorted((ds, d, l, k) for ds in DS for d in DETS for l in ['low', 'medium', 'high'] for k in ['prec', 'nacc']
            if f'{ds}.{d}.conf.{l}.{k}' not in N) ==
     [('gecco', 'lstm_ae', 'medium', 'nacc'), ('gecco', 'lstm_ae', 'medium', 'prec'), ('skab', 'lstm_vae', 'high', 'prec')]),
    ('SWaT under ACI: PerCE largest proximity, sparsity >= 89%', lambda N: all(max(METH.values(), key=lambda m: N[f'swat.{d}.aci.{m}.prox']) == 'perce' and N[f'swat.{d}.aci.perce.spar'] >= 0.89 for d in DETS)),
    ('SWaT sliding at 0.20 flags about half of normal (cov 0.40-0.55)', lambda N: all(0.40 <= N[f'swat.{d}.sliding.a20.cov'] <= 0.55 for d in DETS)),
    ('SWaT drift: static Conv-AE fails while ACI error small', lambda N: N['swat.conv_ae.static.a05.f1'] < 0.05),
    # ---- post-hoc analyses (P1-P6) and exploratory diagnostics
    ('P1: uncensored closer than censored in most cells, more often within 3 pts', lambda N:
     N['p1.uncens_closer_than_cens'] > N['p1.cells'] / 2 and N['p1.sliding_all.within3'] > N['p1.sliding.within3']),
    ('P1: uncensored vs censored |gap| significant in all 12, uncensored smaller', lambda N:
     N['p1.stat.unc_vs_cens_gap.pholm.max'] < 0.05 and N['p1.stat.unc_vs_cens_gap.median_diff.max'] < 0),
    ('P1: uncensored vs ACI |gap| not significant in any of 12', lambda N: N['p1.stat.unc_vs_aci_gap.pholm.min'] > 0.05),
    ('P1: uncensored recall higher than ACI, significant in all 12', lambda N:
     N['p1.stat.unc_vs_aci_rec.pholm.max'] < 0.05 and N['p1.stat.unc_vs_aci_rec.median_diff.min'] > 0),
    ('P1: uncensored raises more alarms (median)', lambda N: N['p1.sliding_all.alarm_dev.median'] > 0),
    ('P2: ACI below matched-FPR reference (median); uncensored loses less', lambda N:
     N['p1.aci.recall_minus_oracle.median'] < 0 and N['p1.sliding_all.recall_minus_oracle.median'] > N['p1.aci.recall_minus_oracle.median']),
    ('P1: ACI over-coverage cells: SKAB highest anomaly share, SMD highest ACI recall at 0.05', lambda N:
     max(DS, key=lambda d: N[f'{d}.meta.test_rate']) == 'skab' and max(DS, key=lambda d: N[f'{d}.avg.aci.a05.rec']) == 'smd'
     and N['names.aci_over3'] == 'SKAB and SMD'),
    ('exploratory: GECCO and Exathlon are the datasets with the most infinite-threshold time', lambda N:
     sorted(DS, key=lambda d: -N[f'aci.{d}.a05.share_inf_pos'])[:2] == ['gecco', 'exathlon']),
    ('exploratory: where infinite periods cover > 0.1% of windows, they cover a larger share of anomalies; MetroPT-3 most', lambda N:
     all(N[f'aci.{d}.a05.anom_under_inf'] > N[f'aci.{d}.a05.share_inf_pos'] for d in DS if N[f'aci.{d}.a05.share_inf_pos'] > 0.001) and
     max(DS, key=lambda d: N[f'aci.{d}.a05.anom_under_inf'] / max(N[f'aci.{d}.a05.share_inf_pos'], 1e-9)
         if N[f'aci.{d}.a05.share_inf_pos'] > 0.001 else 0) == 'metropt3'),
    ('P5: IF ACI vs sliding |gap| and recall significant (ACI closer, lower recall), F1 not', lambda N:
     N['p5.stat.aci_vs_sliding_gap.pholm.max'] < 0.05 and N['p5.stat.aci_vs_sliding_gap.median_diff.max'] < 0 and
     N['p5.stat.aci_vs_sliding_rec.pholm.max'] <= N['p5.stat.aci_vs_sliding_gap.pholm.max'] and N['p5.stat.aci_vs_sliding_rec.median_diff.max'] < 0 and
     N['p5.stat.aci_vs_sliding_f1.pholm.min'] > 0.05),
    ('P5: uncensored further than ACI only at 0.05', lambda N: N['p5.stat.unc_vs_aci_gap.a05.pholm'] < 0.05 and
     N['p5.stat.unc_vs_aci_gap.a05.median_diff'] > 0 and all(N[f'p5.stat.unc_vs_aci_gap.a{a}.pholm'] > 0.05 for a in ['10', '15', '20'])),
    ('SWaT Conv-AE static threshold above almost all test windows: recall near zero', lambda N: N['swat.conv_ae.static.a05.rec'] < 0.05),
    ('P3/P4: end-of-stream ACI threshold infinite in all GECCO and Exathlon runs and one PSM run, no other', lambda N:
     N['p3.aci.end_inf.gecco'] == 15 and N['p3.aci.end_inf.exathlon'] == 15 and N['p3.aci.end_inf.psm'] == 1 and
     sum(N[f'p3.aci.end_inf.{d}'] for d in DS) == 31),
    ('re-scoring: one ACI cell crosses the 3-point boundary, PSM LSTM-VAE 0.20', lambda N:
     N['p1.repro.aci.within3.exp6'] - N['p1.repro.aci.within3.p1'] == 1),
    ('P6: distributional > temporal > density by median AUC, all three significant', lambda N:
     N['p6.auc.distributional.median'] > N['p6.auc.temporal.median'] > N['p6.auc.density.median'] and
     all(N[f'p6.auc.{f}.p_holm'] < 0.05 for f in ['distributional', 'temporal', 'density'])),
    ('P6: pre-specified feature scalar constant on Exathlon, GECCO, PSM', lambda N: N['p6.feature_oor.constant_list'] == 'exathlon,gecco,psm'),
    ('P6: pre-specified feature scalar weakly significant (median AUC < 0.6, p_Holm < 0.05)', lambda N:
     N['p6.auc.feature_oor.p_holm'] < 0.05 and N['p6.auc.feature_oor.median'] < 0.6),
    ('P2: ACI below the fixed-threshold reference in most combinations', lambda N: N['p1.aci.below_oracle'] > N['p1.cells'] / 2),
    ('P1: uncensored vs ACI |gap|: corrected p equal in all 12 (min == max)', lambda N:
     N['p1.stat.unc_vs_aci_gap.pholm.min'] == N['p1.stat.unc_vs_aci_gap.pholm.max']),
    ('sliding has the higher recall in all combinations', lambda N: N['count.slide_rec_higher'] == N['count.cells']),
    ('P5: censored window under-covers in all IF combinations', lambda N: N['p5.sliding.under3'] == N['p5.sliding.cells']),
    ('P6: feature score with per-feature errors not different from chance', lambda N: N['p6.auc.feature.p_holm'] > 0.05),
    ('P6: distributional and temporal above the margin reference (medians)', lambda N:
     min(N['p6.auc.distributional.median'], N['p6.auc.temporal.median']) > N['p6.auc.margin.median']),
    ('P6: distributional correlates with density and temporal; density~temporal near 0; feature weakly related', lambda N:
     N['p6.corr.distributional_density.median'] > 0.5 and N['p6.corr.distributional_temporal.median'] > 0.5 and
     abs(N['p6.corr.density_temporal.median']) < 0.2 and
     all(abs(N[f'p6.corr.{k}.median']) <= 0.3 for k in ['distributional_feature', 'density_feature', 'temporal_feature', 'feature_oor_feature'])),
    # ---- counterfactuals, time-of-detection protocol (P3/P4/P5)
    ('P4: ACI validity higher for all nine detector x explainer', lambda N: all(N[f'p4.stat.{d}.{m}.median_diff'] > 0 for d in DETS for m in METH.values())),
    ('P4: significant exactly for GA-CE (all), PerCE (Conv-AE, LSTM-VAE), DICE (Conv-AE)', lambda N:
     {(d, m) for d in DETS for m in METH.values() if N[f'p4.stat.{d}.{m}.p_holm'] < 0.05} ==
     {('conv_ae', 'gace'), ('lstm_ae', 'gace'), ('lstm_vae', 'gace'), ('conv_ae', 'perce'), ('lstm_vae', 'perce'), ('conv_ae', 'dice')}),
    ('P4: GA-CE gain smaller than at end of stream for every detector', lambda N:
     max(N[f'p4.stat.{d}.gace.median_diff'] for d in DETS) < min(N[f'stat.all.cf.{d}.gace.median_diff'] for d in DETS)),
    ('P4: ACI thresholds at detection above the sliding ones (pass rate)', lambda N: N['p4.aci.gace.pass_rate.median'] > N['p4.sliding.gace.pass_rate.median']),
    ('P3: pass rate correlates with validity for each explainer (time of detection)', lambda N:
     all(N[f'p3.time_of_detection.all.{m}.rho'] > 0 and N[f'p3.time_of_detection.all.{m}.rho_p'] < 0.05 for m in METH.values())),
    ('P3: PerCE gain explained by pass rate; GA-CE keeps an ACI effect; DICE unrelated (time of detection)', lambda N:
     N['p3.time_of_detection.all.perce.p_pass'] < 0.05 and N['p3.time_of_detection.all.perce.p_aci'] > 0.05 and
     N['p3.time_of_detection.all.gace.p_aci'] < 0.05 and N['p3.time_of_detection.all.gace.b_aci'] > 0 and
     N['p3.time_of_detection.all.dice.p_pass'] > 0.05 and N['p3.time_of_detection.all.dice.rho'] < 0.2),
    ('P3: end of stream, pass rate explains GA-CE (ACI n.s.)', lambda N:
     N['p3.end_of_stream.all.gace.p_pass'] < 0.05 and N['p3.end_of_stream.all.gace.p_aci'] > 0.05),
    ('P4: Friedman under ACI significant for every detector; best GA-CE/Conv-AE, PerCE/LSTM-AE, DICE/LSTM-VAE', lambda N:
     all(N[f'p4.fried.{d}.aci.p'] < 0.05 for d in DETS) and
     [min(METH.values(), key=lambda m: N[f'p4.fried.{d}.aci.rank.{m}']) for d in DETS] == ['gace', 'perce', 'dice']),
    ('P4: no explainer as plausible as typical normal; GA-CE/PerCE rarely worse than anomaly; DICE in at least half', lambda N:
     all(N[f'p4.aci.{m}.cf_plaus_valid.median'] > 1 for m in METH.values()) and
     N['p4.aci.gace.plaus_worse_than_anom'] <= 4 and N['p4.aci.perce.plaus_worse_than_anom'] <= 4 and
     N['p4.aci.dice.plaus_worse_than_anom'] >= N['p4.aci.dice.plaus_cells'] / 2),
    ('P4: DICE with LSTM-VAE valid almost always but less plausible than the anomaly in most datasets', lambda N:
     N['p4.aci.lstm_vae.dice.validity.median'] > 0.9 and N['p4.aci.lstm_vae.dice.plaus_worse_than_anom'] > N['p4.aci.lstm_vae.dice.cells'] / 2),
    ('P4: DICE rarely valid with Conv-AE and LSTM-AE', lambda N: N['p4.aci.conv_ae.dice.validity.median'] < 0.15 and N['p4.aci.lstm_ae.dice.validity.median'] < 0.15),
    ('P4: PerCE second for LSTM-VAE and plausible there', lambda N:
     N['p4.fried.lstm_vae.aci.rank.perce'] < N['p4.fried.lstm_vae.aci.rank.gace'] and N['p4.aci.lstm_vae.perce.plaus_worse_than_anom'] <= 2),
    ('P4: PerCE smallest proximity and most unchanged entries, GA-CE largest proximity, DICE fewest unchanged', lambda N:
     min(METH.values(), key=lambda m: N[f'p4.aci.{m}.proximity.median']) == 'perce' and
     max(METH.values(), key=lambda m: N[f'p4.aci.{m}.sparsity.median']) == 'perce' and
     max(METH.values(), key=lambda m: N[f'p4.aci.{m}.proximity.median']) == 'gace' and
     min(METH.values(), key=lambda m: N[f'p4.aci.{m}.sparsity.median']) == 'dice'),
    ('P4: most explained windows are false alarms; SMAP few true anomalies; SMAP GA-CE/PerCE about half valid', lambda N:
     N['p4.aci.true_anom_share'] < 0.5 and N['p4.sliding.true_anom_share'] < 0.5 and N['p4.smap.aci.true_anom_share'] < 0.2 and
     all(0.4 <= N[f'p4.smap.aci.{m}.validity'] <= 0.6 for m in ['gace', 'perce'])),
    ('P4: SWaT validity low for every explainer under ACI', lambda N: max(N[f'p4.swat.aci.{m}.validity'] for m in METH.values()) < 0.25),
    ('P4: GA-CE above 20% under sliding only on MSL, PSM, SKAB', lambda N: N['p4.sliding.gace.ds_over20_list'] == 'msl,psm,skab'),
    ('P4: infinite-threshold datasets drop below 100% for every explainer', lambda N: all(N[f'p4.infds.{m}.validity'] < 0.9 for m in METH.values())),
    # ---- case study (descriptive) and RQ answers
    ('stream: ACI flags about alpha of the windows', lambda N: abs(N['cs.stream.flagged'] / N['cs.stream.n'] - 0.05) < 0.01),
    ('stream: false alarms are windows with very large scores', lambda N: N['cs.stream.fp_score.median'] > 100 * N['cs.stream.normal_score.median']),
    ('stream: confidence below 0.55 for most decisions; extreme calibration spread', lambda N:
     N['cs.stream.conf_lt055'] > 0.9 and N['cs.stream.cal_sd_over_median'] > 1000),
    ('stream: factors separate wrong from correct; marked window at maximum on all three', lambda N:
     N['cs.stream.total_wrong.mean'] > N['cs.stream.total_right.mean'] + 0.2 and
     N['cs.stream.sel_dist_score'] == N['cs.stream.sel_dens_score'] == N['cs.stream.sel_temp_score'] == 1.0),
    ('RQ3: top tenth about twice the average error rate (distributional, temporal)', lambda N:
     all(1.5 < N[f'p6.decile.{f}.top.median'] < 2.5 for f in ['distributional', 'temporal'])),
    ('RQ3: distributional and temporal above 0.5 on most datasets', lambda N:
     N['p6.auc.distributional.above'] > 6 and N['p6.auc.temporal.above'] > 6),
    ('example A: PerCE one sensor, GA-CE all sensors, DICE not valid, GA-CE and PerCE valid, true anomaly', lambda N:
     N['cs.exA.perce.nsensors'] == 1 and N['cs.exA.gace.nsensors'] == N['cs.exA.nsensor_total'] and N['cs.exA.dice.valid'] == 0
     and N['cs.exA.gace.valid'] == 1 and N['cs.exA.perce.valid'] == 1 and N['cs.exA.true'] == 1),
    ('example B: DICE valid changing every entry, PerCE valid in one sensor and as plausible as the window, false alarm', lambda N:
     N['cs.exB.dice.valid'] == 1 and N['cs.exB.dice.nchanged'] == N['cs.exB.entries'] and N['cs.exB.perce.valid'] == 1 and
     N['cs.exB.perce.nsensors'] == 1 and abs(N['cs.exB.perce.plaus'] - N['cs.exB.anom_plaus']) < 0.1 and N['cs.exB.true'] == 0
     and N['cs.exB.dice.plaus'] > N['cs.exB.anom_plaus']),
    ('SWaT example: window in CF sample, no explainer valid', lambda N: N['cs.stream.sel_in_cf_sample'] == 1 and
     all(N[f'cs.stream.sel_cf.{m}.valid_word'] == 'not valid' for m in METH.values())),
    ('SKAB example A: high distributional, typical temporal', lambda N: N['cs.exA.dist_score'] == 1.0 and N['cs.exA.temp_score'] <= 0.1),
    ('examples: both windows just above the threshold', lambda N:
     all(1 < N[f'cs.ex{k}.score'] / N[f'cs.ex{k}.threshold'] < 1.01 for k in ['A', 'B'])),
    ('P5: IF ACI gain significant for GA-CE and PerCE (equal p), explained by pass rate', lambda N:
     N['p5.cf.stat.gace.pholm'] == N['p5.cf.stat.perce.pholm'] < 0.05 and
     all(N[f'p5.cf.pass.{m}.p_aci'] > 0.05 and N[f'p5.cf.pass.{m}.p_pass'] < 0.05 for m in ['gace', 'perce'])),
]


def check_claims(N, draft):
    bad = []
    for desc, f in CLAIMS:
        try:
            ok = bool(f(N))
        except KeyError as e:
            ok, desc = None, desc + f' (missing {e})'
        if ok is not True:
            bad.append(('MISSING ' if ok is None else 'FALSE ') + desc)
    for b in bad:
        print('CLAIM', b)
    print(f'claims: {len(CLAIMS) - len(bad)}/{len(CLAIMS)} hold')
    if bad and not draft:
        raise SystemExit('a statement in the text no longer matches the data')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--allow_partial', action='store_true',
                    help='build from incomplete results (missing seeds or post-hoc parts)')
    args = ap.parse_args()
    env = A.check_env()
    data, dups = load_all()
    gaps = completeness(data)
    if gaps:
        print(f'INCOMPLETE: {len(gaps)} cells below 5 seeds, e.g. ' + '; '.join(gaps[:5]))
        if not args.allow_partial:
            raise SystemExit('refusing to build from incomplete results')
    N = build_numbers(data, dataset_meta())
    ph_missing = posthoc_numbers(N)
    if ph_missing:
        print(f'INCOMPLETE: post-hoc parts without results: {", ".join(ph_missing)}')
        if not args.allow_partial:
            raise SystemExit('refusing to build without every post-hoc analysis')
    for sub in ('figures', 'tables'):
        d = os.path.join(OUT, sub)
        if os.path.isdir(d):
            shutil.rmtree(d)
        os.makedirs(d)
    figures(N, os.path.join(OUT, 'figures'))
    T = tables(N)
    for name, body in T.items():
        open(os.path.join(OUT, 'tables', f'{name}.tex'), 'w').write(body + '\n')
    check_claims(N, args.allow_partial)
    old = json.load(open(AUDIT)) if os.path.exists(AUDIT) else {}
    rec = {'environment': env, 'all_values': N}
    if 'used_in_text' in old:
        rec['used_in_text'] = old['used_in_text']
    json.dump(rec, open(AUDIT, 'w'), indent=1, sort_keys=True)
    print(f'wrote {OUT}: {len(N)} values, {len(T)} tables, superseded duplicate rows: {sum(len(d) for d in dups)}')


if __name__ == '__main__':
    main()
