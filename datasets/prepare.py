"""
Convert each raw extra dataset into the common prepared format read by
datasets/register/xad_extra (data/prepared/<name>/data.npz + meta.json). Every decision that
affects results (split policy, label definition, resampling, dropped
columns, NaN handling, windowing preset) is written into meta.json and the
per-dataset log, so the run is reproducible and auditable.

Needs pyarrow for the 3W parquet files.
    python datasets/prepare.py gecco psm ...   # or: all

Shared rules
  * Label: point label 1 = anomalous/faulty/attack, 0 = normal. A window is
    anomalous if any point in it is (same as the paper's create_sequences).
  * Features: numeric sensor columns only. Columns that are constant or
    entirely NaN on the TRAIN rows are dropped (they carry no information
    for a reconstruction model and break scaling). Remaining NaNs: forward
    fill within each recording, then fill with the train median.
  * Windowing preset (same two presets as the paper), chosen by the size of
    the VAL split, because val normal windows are the conformal calibration
    set and must comfortably exceed the sliding calibration buffer:
        val >= 100k rows -> SWaT preset (seq 100, overlap 0.5, calib window 1000)
                            (>= ~2k val windows at stride 50, 2x the buffer)
        otherwise        -> SKAB preset (seq 50,  overlap 0.8, calib window 500)
  * smoothness_weight (lambda_temp) = 0.0, the paper's deployed value.
"""

import glob
import json
import os
import re
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(REPO, 'data', 'raw')
OUT = os.path.join(REPO, 'data', 'prepared')
LOGDIR = os.path.join(REPO, 'datasets', 'logs')

SKAB_PRESET = {'sequence_length': 50, 'window_overlap': 0.8, 'calibration_window': 500, 'smoothness_weight': 0.0}
SWAT_PRESET = {'sequence_length': 100, 'window_overlap': 0.5, 'calibration_window': 1000, 'smoothness_weight': 0.0}
SPLIT_ID = {'train': 0, 'val': 1, 'test': 2}


class Log:
    def __init__(self, name):
        os.makedirs(LOGDIR, exist_ok=True)
        self.f = open(os.path.join(LOGDIR, f'{name}.log'), 'w')

    def __call__(self, *a):
        s = ' '.join(str(x) for x in a)
        print(s)
        self.f.write(s + '\n')
        self.f.flush()


# ---------------------------------------------------------------- writer ---

def write_prepared(name, runs, feature_names, info, log, preset=None):
    """
    runs: list of dicts {df: DataFrame[feature_names], y: int array,
                         split: 'train'|'val'|'test', seg: recording id (str)}
    Rows of one run must be chronological. Runs are stored in list order.
    """
    seg_ids = {}
    for r in runs:
        seg_ids.setdefault(r['seg'], len(seg_ids))

    # Drop columns constant / all-NaN on train rows.
    train = pd.concat([r['df'] for r in runs if r['split'] == 'train'], ignore_index=True)
    nun = train[feature_names].nunique(dropna=True)
    dropped = [c for c in feature_names if nun[c] <= 1]
    keep = [c for c in feature_names if c not in dropped]
    med = train[keep].median()

    Xs, ys, sps, sgs = [], [], [], []
    n_nan = 0
    for r in runs:
        d = r['df'][keep].astype('float64')
        n_nan += int(d.isna().sum().sum())
        d = d.ffill().fillna(med)
        Xs.append(d.to_numpy(np.float32))
        ys.append(np.asarray(r['y'], dtype=np.int8))
        sps.append(np.full(len(d), SPLIT_ID[r['split']], np.int8))
        sgs.append(np.full(len(d), seg_ids[r['seg']], np.int32))
    X, y, sp, sg = (np.concatenate(a) for a in (Xs, ys, sps, sgs))
    assert np.isfinite(X).all(), 'non-finite values remain after filling'

    if preset is None:
        preset = SWAT_PRESET if int((sp == SPLIT_ID['val']).sum()) >= 100_000 else SKAB_PRESET
    else:
        info = {**info, 'preset_override': 'set explicitly in prepare.py (see preset_reason)'}
    stats = {}
    for s, i in SPLIT_ID.items():
        m = sp == i
        stats[s] = {'rows': int(m.sum()), 'anomalous_rows': int(y[m].sum()),
                    'anomaly_rate': round(float(y[m].mean()) if m.any() else float('nan'), 5),
                    'recordings': int(len(np.unique(sg[m])))}

    d = os.path.join(OUT, name)
    os.makedirs(d, exist_ok=True)
    np.savez_compressed(os.path.join(d, 'data.npz'), X=X, y=y, split=sp, seg=sg)
    meta = {
        'name': name,
        'prepared_utc': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
        'feature_names': keep,
        'dropped_features_constant_on_train': dropped,
        'n_features': len(keep),
        'n_rows': int(len(X)),
        'nan_cells_filled': n_nan,
        'n_recordings': len(seg_ids),
        'config': preset,
        'split_stats_points': stats,
        **info,
    }
    with open(os.path.join(d, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)
    log(f'[{name}] rows={len(X)} features={len(keep)} (dropped {len(dropped)}: {dropped}) '
        f'recordings={len(seg_ids)} nan_filled={n_nan} preset=seq{preset["sequence_length"]}')
    for s in SPLIT_ID:
        log(f'[{name}]   {s:5s} {stats[s]}')
    return meta


# ------------------------------------------------------------- mTSBench ---

def _label(df):
    return (df['is_anomaly'].to_numpy() != 0).astype(np.int8) if 'is_anomaly' in df else np.zeros(len(df), np.int8)


def mtsbench(name, raw_dir, info, entity_filter=None, log=None, preset=None):
    """
    mTSBench policy (uniform for all 6 subsets): per entity,
      official *_train.csv  -> first 80% train, last 20% val (chronological;
                               train files are anomaly-free),
      official *_val.csv    -> appended to val (if the entity has one),
      official *_test.csv   -> test.
    """
    files = sorted(glob.glob(os.path.join(RAW, raw_dir, '*.csv')))
    ents = sorted({re.sub(r'_(train|val|test)\.csv$', '', os.path.basename(f)) for f in files})
    if entity_filter:
        ents = [e for e in ents if entity_filter(e)]
    runs, feats = [], None
    for e in ents:
        parts = {s: os.path.join(RAW, raw_dir, f'{e}_{s}.csv') for s in ('train', 'val', 'test')}
        if not (os.path.exists(parts['train']) and os.path.exists(parts['test'])):
            log(f'[{name}] skip entity {e}: train or test file missing in the source')
            continue
        tr = pd.read_csv(parts['train'])
        cols = [c for c in tr.columns if c not in ('timestamp', 'is_anomaly')]
        if feats is None:
            feats = cols
        elif set(cols) != set(feats):
            log(f'[{name}] skip entity {e}: feature columns differ')
            continue
        cut = int(round(len(tr) * 0.8))
        runs.append({'df': tr.iloc[:cut], 'y': _label(tr)[:cut], 'split': 'train', 'seg': e})
        runs.append({'df': tr.iloc[cut:], 'y': _label(tr)[cut:], 'split': 'val', 'seg': e})
        if os.path.exists(parts['val']):
            va = pd.read_csv(parts['val'])
            runs.append({'df': va, 'y': _label(va), 'split': 'val', 'seg': e + '#val'})
        te = pd.read_csv(parts['test'])
        runs.append({'df': te, 'y': _label(te), 'split': 'test', 'seg': e + '#test'})
    info = {**info, 'entities': ents,
            'split_policy': 'mTSBench official files; per entity: train file -> first 80% train / '
                            'last 20% val (chronological); official val file -> val; test file -> test'}
    return write_prepared(name, runs, feats, info, log, preset=preset)


# --------------------------------------------------------------- MetroPT-3 --

METROPT_FAILURES = [  # Davari et al. 2021, failure reports shipped with the dataset
    ('2020-04-18 00:00', '2020-04-18 23:59'),
    ('2020-05-29 23:30', '2020-05-30 06:00'),
    ('2020-06-05 10:00', '2020-06-07 14:30'),
    ('2020-07-15 14:30', '2020-07-15 19:00'),
]


def metropt(log):
    df = pd.read_csv(os.path.join(RAW, 'metropt3', 'MetroPT3(AirCompressor).csv'))
    df = df.drop(columns=[c for c in df.columns if c.startswith('Unnamed')])
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    feats = [c for c in df.columns if c != 'timestamp']
    y = np.zeros(len(df), np.int8)
    for a, b in METROPT_FAILURES:
        y[(df['timestamp'] >= a) & (df['timestamp'] <= b)] = 1
    # Split by date so all 4 failures fall in test; train/val are failure-free.
    t = df['timestamp']
    split = np.where(t < '2020-03-20', 'train', np.where(t < '2020-04-10', 'val', 'test'))
    # New recording at sampling gaps > 5 min, so no window spans a gap.
    gap = t.diff().dt.total_seconds().fillna(0).to_numpy() > 300
    rec = np.cumsum(gap)
    runs = []
    key = pd.Series(split).astype(str) + '|' + pd.Series(rec).astype(str)
    bounds = np.flatnonzero(key.to_numpy()[1:] != key.to_numpy()[:-1]) + 1
    for idx in np.split(np.arange(len(df)), bounds):
        runs.append({'df': df.iloc[idx][feats], 'y': y[idx], 'split': split[idx[0]], 'seg': f'rec{rec[idx[0]]}'})
    log(f'[metropt3] {int(gap.sum())} sampling gaps > 5 min -> {rec.max() + 1} contiguous recordings')
    info = {'source': 'UCI ML repo id 791 (MetroPT-3, Veloso et al. 2022), CC BY 4.0',
            'domain': 'Railway / air compressor (APU) of a metro train',
            'label_definition': 'point is anomalous inside one of the 4 reported air-leak failure '
                                'windows (failure report table in the dataset description)',
            'failure_windows': METROPT_FAILURES,
            'split_policy': 'by date: train < 2020-03-20, val 2020-03-20..2020-04-10, '
                            'test >= 2020-04-10 (all 4 failures in test); recordings split at '
                            'sampling gaps > 5 min'}
    return write_prepared('metropt3', runs, feats, info, log)


# ---------------------------------------------------------------- 3W (real) --

def PLAUSIBLE_3W(col):
    if col.startswith('P-') or col == 'PT-P':
        return -1e8, 1e8
    if col.startswith('T-'):
        return -50.0, 500.0
    if col.startswith('Q'):
        return -100.0, 100.0
    if col.startswith('ABER-'):
        # small overshoot (100.2-100.4 %) is real calibration slack, keep it;
        # -99.99 is a missing-value code
        return -5.0, 110.0
    if col.startswith('ESTADO-'):
        return 0.0, 1.0
    return -np.inf, np.inf


def threew(log, resample_s=10, fault_tail_min=30):
    """
    Petrobras 3W v2, REAL instances only. Each instance (one well, one
    period) is one recording. Resampled to `resample_s` seconds (mean for
    signals, max for the label) to keep the series length comparable to
    SWaT. Label: class != 0 (covers transient 1xx and steady-state codes).
    Split: instances assigned to train/val/test by a seeded shuffle,
    stratified by event type, 60/20/20 by instance count. Normal-only
    instances (type 0) and the normal head of fault instances are what the
    detector trains/calibrates on (pipeline uses normal-only windows).
    """
    import configparser
    cp = configparser.ConfigParser()
    cp.read(os.path.join(RAW, 'threew', 'dataset.ini'))
    desc = dict(cp['PARQUET_FILE_PROPERTIES'])
    rng = np.random.RandomState(42)
    inst = []
    for c in range(10):
        fs = sorted(glob.glob(os.path.join(RAW, 'threew', str(c), 'WELL-*.parquet')))
        order = rng.permutation(len(fs))
        n = len(fs)
        n_tr, n_va = int(round(0.6 * n)), int(round(0.2 * n))
        for rank, i in enumerate(order):
            s = 'train' if rank < n_tr else ('val' if rank < n_tr + n_va else 'test')
            inst.append((c, fs[i], s))
    frames = []
    n_sentinel = [0]
    n_sentinel_by_col = {}
    n_trunc = [0]
    for c, f, s in inst:
        d = pd.read_parquet(f)
        if 'timestamp' in d.columns:
            d = d.set_index('timestamp')
        d.index = pd.to_datetime(d.index)
        lab = d['class'].fillna(0)
        num = d.drop(columns=[x for x in ('class', 'state') if x in d.columns]).apply(pd.to_numeric, errors='coerce')
        # Physically impossible readings and missing-value codes -> missing
        # (inspected file by file before choosing the limits). Found:
        # P-PDG 1.2e42 Pa and T-PDG 1.7e38 C (WELL-00006, 2018), T-PDG 2.9e15 C
        # (WELL-00019), P-TPT 2.9e9 Pa (WELL-00015). Limits by unit
        # (dataset.ini): legitimate maxima are 4.3e7 Pa and 120 C.
        sentinel = pd.DataFrame(False, index=num.index, columns=num.columns)
        for col in num.columns:
            lo, hi = PLAUSIBLE_3W(col)
            bad = (num[col] < lo) | (num[col] > hi)
            n_sentinel_by_col[col] = n_sentinel_by_col.get(col, 0) + int(bad.sum())
            sentinel[col] = bad
        n_sentinel[0] += int(sentinel.sum().sum())
        num = num.mask(sentinel)
        r = num.resample(f'{resample_s}s').mean()
        yl = (lab != 0).astype(int).resample(f'{resample_s}s').max().reindex(r.index).fillna(0).to_numpy(np.int8)
        # Keep the normal head of a fault instance plus the first
        # `fault_tail_min` minutes after onset; the hours/days of steady-state
        # fault that follow would otherwise make 40-50% of all points
        # anomalous (fails the rare-anomaly criterion) and add nothing to
        # onset detection.
        if yl.any():
            cut = int(np.argmax(yl)) + fault_tail_min * 60 // resample_s
            n_trunc[0] += max(0, len(yl) - cut)
            r, yl = r.iloc[:cut], yl[:cut]
        frames.append((c, os.path.basename(f), s, r, yl))
    cols = sorted(set().union(*[set(fr[3].columns) for fr in frames]))
    # Keep a column only if it is observed (non-NaN) in >= 50% of train rows.
    trn = pd.concat([fr[3].reindex(columns=cols) for fr in frames if fr[2] == 'train'])
    cover = trn.notna().mean()
    feats = [c for c in cols if cover[c] >= 0.5]
    log(f'[threew] physically implausible cells set to missing: {n_sentinel[0]} '
        f'by column: { {k: v for k, v in n_sentinel_by_col.items() if v} }')
    log(f'[threew] steady-state fault rows dropped (> {fault_tail_min} min after onset): {n_trunc[0]}')
    log(f'[threew] {len(frames)} real instances; columns kept by >=50% train coverage: '
        f'{len(feats)}/{len(cols)}; dropped low-coverage: {[c for c in cols if c not in feats]}')
    runs = [{'df': fr[3].reindex(columns=feats), 'y': fr[4], 'split': fr[2], 'seg': f'{fr[0]}/{fr[1]}'}
            for fr in frames]
    by = pd.DataFrame([(fr[0], fr[2]) for fr in frames], columns=['event', 'split'])
    log('[threew] instances per event type x split:\n' + by.value_counts().unstack(fill_value=0).to_string())
    info = {'source': 'Petrobras 3W v2.0.0 (Vargas et al. 2019), github.com/petrobras/3W, CC BY 4.0',
            'domain': 'Offshore oil wells (subsea production)',
            'instances': 'REAL only (WELL-*), simulated and hand-drawn excluded',
            'resample_seconds': resample_s,
            'fault_truncation': f'fault instances kept up to {fault_tail_min} min after the first '
                                'anomalous label (normal head + onset); later steady-state fault rows dropped',
            'implausible_value_rule': 'set to missing outside unit limits: pressure [Pa] |P|<=1e8, '
                                      'temperature [C] -50..500, flow [m3/s] |Q|<=100, '
                                      'choke opening [%] -5..110, valve state 0..1. Masked values found: T-PDG -999 and ABER-CKP/T-MON-CKP -99.99 (missing codes), P-JUS-CKP 1.2-1.5e9 Pa (WELL-00026, 3 files, unit error), P-PDG <= -1e8 Pa and 1.2e42, T-PDG 1.7e38/2.9e15, P-TPT 2.9e9 Pa',
            'label_definition': "class != 0 (transient 1xx and steady-state fault codes); "
                                "window label = max over the resampling bin",
            'feature_descriptions': {k.upper(): v for k, v in desc.items() if k.upper() in feats},
            'split_policy': 'instances (not rows) assigned by seeded shuffle (seed 42), stratified '
                            'by event type, 60/20/20 by count'}
    return write_prepared('threew', runs, feats, info, log)


# -------------------------------------------------------------- registry ---

def run(name):
    log = Log(name)
    log(f'=== prepare {name}  {datetime.now(timezone.utc):%Y-%m-%dT%H:%M:%SZ}')
    if name == 'gecco':
        return mtsbench('gecco', 'gecco', {
            'source': 'mTSBench/GECCO (GECCO 2018 IC, water quality), CC BY-NC 4.0',
            'domain': 'Drinking-water quality monitoring'}, log=log)
    if name == 'swan_sf':
        return mtsbench('swan_sf', 'swan_sf', {
            'source': 'mTSBench/SWAN-SF (Angryk et al. 2020), CC BY-NC 4.0',
            'domain': 'Space weather (solar flare magnetogram features)'}, log=log)
    if name == 'psm':
        return mtsbench('psm', 'psm', {
            'source': 'mTSBench/PSM (eBay, Abdulaal et al. 2021), CC BY-NC 4.0',
            'domain': 'IT server metrics (anonymised features)'}, log=log)
    if name == 'smd':
        return mtsbench('smd', 'smd', {
            'source': 'mTSBench/SMD (Su et al. 2019), CC BY-NC 4.0',
            'domain': 'IT server machines (anonymised features), 18 machines pooled'}, log=log)
    if name == 'ghl':
        return mtsbench('ghl', 'ghl', {
            'source': 'mTSBench/GHL (Filonov et al. 2016, gasoil heating loop, Modelica simulation), CC BY-NC 4.0',
            'domain': 'Industrial process control (simulated gasoil heating loop), 14 runs pooled'}, log=log)
    if name == 'exathlon':
        # Each Exathlon trace has its own 21-column feature set, so traces
        # cannot be pooled; use the largest trace (the one mTSBench also
        # ships a val file for).
        return mtsbench('exathlon', 'exathlon', {
            'source': 'mTSBench/Exathlon (Jacob et al. 2021), CC BY-NC 4.0',
            'domain': 'Spark cluster monitoring',
            'trace_choice': 'single trace 4_1_100000_61: per-trace feature sets differ, so traces '
                            'cannot be pooled; this is the largest trace and the one with an official val file'},
            entity_filter=lambda e: e == 'Exathlon_4_1_100000_61', log=log)
    if name == 'daphnet':
        return mtsbench('daphnet', 'daphnet', {
            'source': 'mTSBench/Daphnet (Bachlin et al. 2010, freezing of gait), CC BY-NC 4.0',
            'domain': 'Healthcare wearable accelerometers, 26 sessions pooled'}, log=log)
    if name == 'smap':
        return mtsbench('smap', 'smap', {
            'source': 'mTSBench/SMAP (NASA, Hundman et al. 2018), CC BY-NC 4.0',
            'domain': 'Spacecraft telemetry (anonymised command/telemetry channels), all channels pooled',
            'replaces': 'swan_sf: 76% of SWAN-SF test windows are anomalous (fails the rare-anomaly '
                        'criterion), see logs/prepare/verify_windows_v1.tsv'}, log=log)
    if name == 'msl':
        return mtsbench('msl', 'msl', {
            'source': 'mTSBench/MSL (NASA Mars Science Laboratory, Hundman et al. 2018), CC BY-NC 4.0',
            'domain': 'Spacecraft telemetry (anonymised command/telemetry channels), all channels pooled',
            'replaces': 'pronto: labelled PRONTO data is ~45k rows (< 100k volume criterion) and 81-94% '
                        'of each test day is fault (fails rare-anomaly criterion); see AUDIT.md B9'}, log=log)
    if name == 'metropt3':
        return metropt(log)
    if name == 'threew':
        return threew(log)
    if name == 'pronto':
        import prepare_pronto
        return prepare_pronto.run(log, write_prepared)
    raise SystemExit(f'unknown dataset {name}')


ALL = ['gecco', 'psm', 'smd', 'ghl', 'exathlon', 'daphnet', 'smap', 'msl', 'metropt3', 'threew']
# pronto was evaluated and excluded (AUDIT.md B9).
# swan_sf was prepared once and then excluded (see 'replaces' in smap's meta); still runnable by name.

if __name__ == '__main__':
    names = ALL if sys.argv[1:] == ['all'] else sys.argv[1:]
    for n in names:
        run(n)
