"""
Figures of the paper. Every value is read from the generated numbers (N, built by
paper_outputs.build_numbers) or from the post-hoc summaries in results/posthoc_summary/;
nothing is typed by hand. Called from paper_outputs.figures().

One theme for every figure (font, ink, grid, panel titles), and colour carries one
meaning per family, never as the only cue:
  threshold policies  static #2a78d6 (square), sliding #eb6834 (down triangle),
                      uncensored sliding = same orange, hollow / dashed, ACI #1baf7a (circle)
  explainers          GA-CE #6250d6, DICE #eda100, PerCE #e34948, with their own markers
  detector families   autoencoders #8e3b8e (plum, shape per architecture), classical #c47f17 (amber, open)
  confidence          aqua ramp (confidence is ACI's): low #8fd3b8, middle #1baf7a, high #0b6e4f
  uncertainty         violet ramp for factor magnitudes; wrong decisions #c0392b, correct #a9a7a0
  anomalous periods   coral tint #fbe3d9, as the anomaly lane of the framework figure
The 3-colour policy and explainer sets and the 2-colour family pair pass the
all-pairs CVD and normal-vision checks of the dataviz validator; the aqua and
yellow slots are below 3:1 contrast, so they always come with a marker shape and a
legend or direct label.
"""
import json
import os

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                                  # noqa: E402
from matplotlib import font_manager as fm                        # noqa: E402
from matplotlib.lines import Line2D                              # noqa: E402
import matplotlib.patches                                         # noqa: E402,F401
from matplotlib.colors import LinearSegmentedColormap             # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))

from paths import RUNS_POSTHOC as PH, RESULTS  # noqa: E402
PS = os.path.join(RESULTS, 'posthoc')

INK, INK2, MUTED, GRID, BAND = '#1b2333', '#52514e', '#8a8880', '#e8e6dc', '#f1efe8'
ANOM = '#fbe3d9'                                                  # anomalous periods
FAM = {'deep': '#8e3b8e', 'classical': '#c47f17'}
CONFC = {'low': '#8fd3b8', 'medium': '#1baf7a', 'high': '#0b6e4f'}
WRONG, RIGHT = '#c0392b', '#a9a7a0'
UNC_RAMP = ['#f6f4ff', '#cfc6f7', '#8b7be3', '#6250d6', '#2e1f80']
POLC = {'static': '#2a78d6', 'sliding': '#eb6834', 'sliding_all': '#eb6834', 'aci': '#1baf7a'}
POLM = {'static': 's', 'sliding': 'v', 'sliding_all': 'v', 'aci': 'o'}
POLN = {'static': 'Static', 'sliding': 'Sliding', 'sliding_all': 'Sliding, uncensored', 'aci': 'ACI'}
POLFILL = {'static': True, 'sliding': True, 'sliding_all': False, 'aci': True}
EXC = {'GA-CE': '#6250d6', 'DICE': '#eda100', 'PerCE': '#e34948'}
EXM = {'GA-CE': 'o', 'DICE': 's', 'PerCE': 'D'}
DETM = {'conv_ae': 'o', 'lstm_ae': 'D', 'lstm_vae': '^'}
META = {'Creator': None, 'Producer': None, 'CreationDate': None}
W2 = 7.0                                   # full text width (in)


def style():
    for w in ['Regular', 'Bold', 'Italic']:
        fm.fontManager.addfont(os.path.join(HERE, 'fonts', f'NimbusSansXAD-{w}.ttf'))
    fam = 'Nimbus Sans XAD'                     # renamed TrueType copy (fonts/README.txt), embedded as Type 42
    assert fm.findfont(fm.FontProperties(family=fam)).endswith('NimbusSansXAD-Regular.ttf')
    plt.rcParams.update({
        'font.family': fam, 'font.size': 7.5, 'axes.titlesize': 7.5, 'axes.labelsize': 7.5,
        'xtick.labelsize': 6.8, 'ytick.labelsize': 6.8, 'legend.fontsize': 6.8,
        'axes.linewidth': 0.6, 'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'text.color': INK,
        'xtick.color': INK2, 'ytick.color': INK2, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
        'xtick.major.size': 2.5, 'ytick.major.size': 2.5, 'axes.titleweight': 'bold', 'axes.titlelocation': 'left',
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'axes.unicode_minus': True, 'legend.frameon': False,
        'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02})


def clean(ax, grid='x'):
    for s in ['top', 'right']:
        ax.spines[s].set_visible(False)
    if grid:
        ax.grid(axis=grid, color=GRID, lw=0.5, zorder=0)
    ax.set_axisbelow(True)


def save(fig, out, name):
    fig.savefig(os.path.join(out, f'{name}.pdf'), metadata=META)
    fig.savefig(os.path.join(out, f'{name}.png'), dpi=220)
    plt.close(fig)


def pol_scatter(ax, x, y, pol, s=16, **kw):
    c = POLC[pol]
    ax.scatter(x, y, s=s, marker=POLM[pol], facecolor=c if POLFILL[pol] else 'white', edgecolor=c if not POLFILL[pol] else 'white',
               linewidth=0.9 if not POLFILL[pol] else 0.4, zorder=3, **kw)


def pol_handle(pol, ls=None):
    c = POLC[pol]
    return Line2D([], [], marker=POLM[pol], ls=ls or 'none', color=c, markerfacecolor=c if POLFILL[pol] else 'white',
                  markeredgecolor=c if not POLFILL[pol] else 'white', markersize=5, label=POLN[pol])


def letter(ax, s):
    ax.set_title(s, loc='left', fontsize=7.5, fontweight='bold', pad=4)


# ------------------------------------------------------------------ detection

def fig_detection(N, out, DS, NAME, DETS, DETN):
    """(a) ROC-AUC per dataset, (b) PR-AUC as a multiple of the anomaly rate."""
    order = sorted(DS, key=lambda d: max(N[f'{d}.{m}.roc'] for m in DETS))
    fig, axs = plt.subplots(1, 2, figsize=(W2, 2.75), sharey=True, gridspec_kw=dict(wspace=0.08))
    y = np.arange(len(order))
    for ax, k in zip(axs, ['roc', 'lift']):
        for i, ds in enumerate(order):
            rate = N[f'{ds}.meta.test_rate']
            val = lambda m: N[f'{ds}.{m}.roc'] if k == 'roc' else N[f'{ds}.{m}.pr'] / rate
            deep = [val(m) for m in DETS]
            ax.plot([min(deep), max(deep)], [i, i], color=FAM['deep'], alpha=0.45, lw=1.6, zorder=2, solid_capstyle='round')
            for m in DETS:
                ax.scatter(val(m), i, marker=DETM[m], s=19, color=FAM['deep'], edgecolor='white', lw=0.4, zorder=4)
            for m, mk in [('iforest', 's'), ('lof', 'P')]:
                if f'{ds}.{m}.{"roc" if k == "roc" else "pr"}' in N:
                    ax.scatter(val(m), i, marker=mk, s=16, facecolor='white', edgecolor=FAM['classical'], lw=0.9, zorder=3)
        clean(ax)
    axs[0].axvline(0.5, color=MUTED, lw=0.7, ls=(0, (3, 2)), zorder=1)
    axs[0].text(0.49, len(order) - 0.45, 'chance', color=MUTED, fontsize=6.3, ha='right', va='center')
    axs[0].set_xlim(0.25, 1.0)
    axs[0].set_xlabel('ROC-AUC')
    axs[1].set_xscale('log')
    axs[1].axvline(1, color=MUTED, lw=0.7, ls=(0, (3, 2)), zorder=1)
    axs[1].text(0.97, len(order) - 0.45, 'chance', color=MUTED, fontsize=6.3, ha='right', va='center')
    axs[1].set_xticks([0.5, 1, 2, 5, 10, 20])
    axs[1].set_xticklabels(['0.5', '1', '2', '5', '10', '20'])
    axs[1].minorticks_off()
    axs[1].set_xlim(0.45, 30)
    axs[1].set_xlabel('PR-AUC ÷ anomaly rate of the test windows (log scale)')
    axs[0].set_yticks(y)
    axs[0].set_yticklabels([f'{NAME[d]}  ' for d in order])
    axs[0].tick_params(axis='y', length=0)
    axs[0].set_ylim(-0.6, len(order) - 0.1)
    letter(axs[0], '(a) Ranking of anomalies')
    letter(axs[1], '(b) Precision gain over the anomaly rate')
    h = [Line2D([], [], marker=DETM[m], ls='none', color=FAM['deep'], markeredgecolor='white', markersize=5.5, label=DETN[m]) for m in DETS]
    h += [Line2D([], [], marker='s', ls='none', markerfacecolor='white', markeredgecolor=FAM['classical'], markersize=5.5, label='Isolation Forest'),
          Line2D([], [], marker='P', ls='none', markerfacecolor='white', markeredgecolor=FAM['classical'], markersize=5.5, label='LOF')]
    fig.legend(handles=h, loc='upper center', ncol=5, bbox_to_anchor=(0.5, 1.07), handletextpad=0.2, columnspacing=1.4)
    save(fig, out, 'fig_detection')


# ------------------------------------------------------------------ mechanism

def _trajectory(ax, ds, NAME, det='conv_ae', seed=42, alpha='05', legend=False):
    z = np.load(os.path.join(PH, 'results', 'policies', f'{ds}_{det}_s{seed}.npz'))
    s, yv = z['test_scores'].astype(float), z['y_test'].astype(int)
    x = np.arange(len(s))
    # anomalous stretches as light bands, scores as small marks (anomalous darker)
    d = np.diff(np.concatenate([[0], yv, [0]]))
    for a, b in zip(np.where(d == 1)[0], np.where(d == -1)[0]):
        ax.axvspan(a - 0.5, b - 0.5, color=ANOM, lw=0, zorder=0)
    ax.scatter(x[yv == 0], s[yv == 0], s=0.6, color='#c8c6bd', lw=0, zorder=1, rasterized=True)
    ax.scatter(x[yv == 1], s[yv == 1], s=0.8, color=INK2, lw=0, zorder=1, rasterized=True)
    top = np.nanmax(s) * 3
    for pol in ['static', 'sliding', 'sliding_all', 'aci']:
        t = z[f'thr_{pol}_a{alpha}'].astype(float)
        fin = np.isfinite(t)
        ax.plot(x, np.where(fin, t, np.nan), color=POLC[pol], lw={'aci': 0.45}.get(pol, 1.15),
                ls=(0, (3, 1.5)) if pol == 'sliding_all' else '-', zorder={'aci': 2}.get(pol, 4), solid_joinstyle='miter')
        if pol == 'aci' and (~fin).any():
            ax.scatter(x[~fin], np.full((~fin).sum(), top), marker='|', s=10, lw=0.6, color=POLC['aci'], zorder=3, rasterized=True)
            ax.text(-0.012, top, '∞', transform=matplotlib.transforms.blended_transform_factory(ax.transAxes, ax.transData),
                    color=INK2, fontsize=8, ha='right', va='center')
    ax.set_yscale('log')
    ax.set_xlim(-len(s) * 0.01, len(s) * 1.01)
    clean(ax, grid='y')
    ax.set_ylabel('Nonconformity score')
    return len(s), float(np.mean(~np.isfinite(z[f'thr_aci_a{alpha}'])))


def fig_mechanism(N, out, NAME, DS):
    fig = plt.figure(figsize=(W2, 3.9))
    gs = fig.add_gridspec(2, 2, width_ratios=[2.35, 1], hspace=0.55, wspace=0.42)
    axa, axb, axc = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[:, 1])
    n, _ = _trajectory(axa, 'exathlon', NAME)
    letter(axa, '(a) Exathlon, gradual drift of normal scores')
    n, _ = _trajectory(axb, 'swat', NAME)
    letter(axb, '(b) SWaT, attack periods (shaded)')
    axb.set_xlabel('Test window (time order)')
    h = [pol_handle(p, ls='-' if p != 'sliding_all' else (0, (3, 1.5))) for p in ['static', 'sliding', 'sliding_all', 'aci']]
    for hh in h:
        hh.set_marker('')
        hh.set_linewidth(1.1)
    h += [Line2D([], [], marker='o', ls='none', color='#c8c6bd', markersize=2.5, label='normal window'),
          Line2D([], [], marker='o', ls='none', color=INK2, markersize=2.5, label='anomalous window'),
          matplotlib.patches.Patch(color=ANOM, label='anomalous period')]
    axa.legend(handles=h, loc='lower left', bbox_to_anchor=(0, 1.13), ncol=7, handlelength=1.5, columnspacing=0.9, fontsize=6.2)
    # (c) share of the test stream with ACI threshold at +inf
    g = pd.read_csv(os.path.join(PS, 'aci_trace_per_dataset.csv'))
    order = sorted(DS, key=lambda d: g[(g.dataset == d) & np.isclose(g.alpha, 0.05)].share_inf_pos.iloc[0])
    for i, ds in enumerate(order):
        r5 = g[(g.dataset == ds) & np.isclose(g.alpha, 0.05)].iloc[0]
        r20 = g[(g.dataset == ds) & np.isclose(g.alpha, 0.20)].iloc[0]
        axc.plot([r5.share_inf_pos * 100, r5.anom_under_inf * 100], [i, i], color=GRID, lw=2.2, zorder=1, solid_capstyle='round')
        axc.scatter(r5.share_inf_pos * 100, i, marker='o', s=18, color=POLC['aci'], edgecolor='white', lw=0.4, zorder=3)
        axc.scatter(r5.anom_under_inf * 100, i, marker='o', s=18, facecolor='white', edgecolor=POLC['aci'], lw=1.0, zorder=3)
        axc.scatter(r20.share_inf_pos * 100, i, marker='|', s=26, color=INK2, lw=0.9, zorder=2)
    axc.set_yticks(range(len(order)))
    axc.set_yticklabels([NAME[d] for d in order])
    axc.tick_params(axis='y', length=0)
    axc.set_xlim(-3, 100)
    axc.set_xlabel('Share of windows (%)')
    clean(axc)
    letter(axc, '(c) ACI threshold at ∞')
    hc = [Line2D([], [], marker='o', ls='none', color=POLC['aci'], markeredgecolor='white', markersize=5, label='all test windows, α = 0.05'),
          Line2D([], [], marker='o', ls='none', markerfacecolor='white', markeredgecolor=POLC['aci'], markersize=5, label='anomalous windows, α = 0.05'),
          Line2D([], [], marker='|', ls='none', color=INK2, markersize=6, label='all test windows, α = 0.20')]
    axc.legend(handles=hc, loc='upper left', bbox_to_anchor=(-0.05, -0.09), handletextpad=0.2, fontsize=6.2, labelspacing=0.3)
    save(fig, out, 'fig_mechanism')


# ------------------------------------------------------------- policy grid

def _strip(ax, vals_by_pol, pols, rng, jitter=0.16):
    for i, p in enumerate(pols):
        v = np.asarray(vals_by_pol[p], float)
        v = v[~np.isnan(v)]
        yy = i + rng.uniform(-jitter, jitter, len(v))
        pol_scatter(ax, v, yy, p, s=7, alpha=0.85)
        med = np.median(v)
        ax.plot([med, med], [i - 0.32, i + 0.32], color=INK, lw=1.3, zorder=5, solid_capstyle='butt')
    ax.set_yticks(range(len(pols)))
    ax.set_ylim(len(pols) - 0.45, -0.55)
    ax.tick_params(axis='y', length=0)
    clean(ax)


def fig_policies(N, out):
    """Rows: deep detectors (144 cells), Isolation Forest (48 cells). Columns: coverage gap,
    total alarm rate minus alpha, recall minus the fixed threshold with the same false-alarm rate."""
    pols = ['static', 'sliding', 'sliding_all', 'aci']
    p1 = pd.read_csv(os.path.join(PS, 'p1_policies.csv'))
    al = pd.read_csv(os.path.join(PS, 'p1_alarm_rate.csv')).groupby(['dataset', 'detector', 'policy', 'alpha']).alarm_dev.mean().reset_index()
    p1 = p1.merge(al, on=['dataset', 'detector', 'policy', 'alpha'])
    p5 = pd.read_csv(os.path.join(PS, 'p5_policies.csv'))
    rows = [('Autoencoders', p1, 'recall_minus_oracle'), ('Isolation Forest', p5, 'rmo')]
    fig, axs = plt.subplots(2, 3, figsize=(W2, 3.55), gridspec_kw=dict(hspace=0.55, wspace=0.12, height_ratios=[1, 1]))
    rng = np.random.default_rng(0)
    def lim(cols, lo_min, hi_min):
        v = np.concatenate([d[c].values for (_, d, rk) in rows for c in ([rk] if cols == 'rk' else [cols])])
        v = v[~np.isnan(v)]
        pad = 0.04 * (v.max() - v.min())
        return min(v.min() - pad, lo_min), max(v.max() + pad, hi_min)
    for r, (lab, d, rk) in enumerate(rows):
        n = int(d.groupby('policy').size().iloc[0])
        for c, (col, xl, xr) in enumerate([('gap', 'Coverage − (1 − α)', lim('gap', -0.1, 0.1)),
                                           ('alarm_dev', 'Alarm rate − α  (all test windows)', lim('alarm_dev', -0.1, 0.1)),
                                           (rk, 'Recall − recall of a fixed threshold\nwith the same false-alarm rate', lim('rk', -0.1, 0.1))]):
            ax = axs[r, c]
            if col == 'gap':
                ax.axvspan(-0.03, 0.03, color=BAND, lw=0, zorder=0)
            ax.axvline(0, color=MUTED, lw=0.6, zorder=1)
            _strip(ax, {p: d[d.policy == p][col] for p in pols}, pols, rng)
            ax.set_xlim(*xr)
            if c == 0:
                ax.set_yticklabels([POLN[p] for p in pols])
            else:
                ax.set_yticklabels([])
            ax.set_xlabel(xl if r == 1 else '')
        axs[r, 0].set_title(f'({"ab"[r]}) {lab}, {n} cells', loc='left', fontsize=7.5, fontweight='bold', pad=4)
    save(fig, out, 'fig_policies')


# ------------------------------------------------------------------- tests

def _forest(ax, rows, title, xlabel, col=INK, mk='o'):
    """rows: list of (label, median, lo, hi, p_holm) top to bottom; filled = p_Holm < 0.05."""
    for i, (lab, m, lo, hi, p) in enumerate(rows):
        sig = p < 0.05
        ax.plot([lo, hi], [i, i], color=col, alpha=0.75, lw=1.0, zorder=2, solid_capstyle='butt')
        ax.scatter(m, i, s=19, marker=mk, facecolor=col if sig else 'white', edgecolor='white' if sig else col,
                   lw=0.4 if sig else 0.9, zorder=3)
    ax.axvline(0, color=MUTED, lw=0.6, zorder=1)
    ax.set_ylim(len(rows) - 0.4, -0.6)
    ax.set_title(title, loc='left', fontsize=7.0, fontweight='bold', pad=3)
    ax.set_xlabel(xlabel, fontsize=6.6)
    clean(ax)
    ax.tick_params(axis='y', length=0)


def fig_tests(N, out, DETS, DETN):
    """Paired differences over the twelve datasets, per detector x alpha (median, bootstrap 95% CI)."""
    t1 = pd.read_csv(os.path.join(PS, 'p1_tests.csv'))
    alphas = [0.05, 0.10, 0.15, 0.20]
    ak = {0.05: '05', 0.10: '10', 0.15: '15', 0.20: '20'}
    labels = [(d, a) for d in DETS for a in alphas]

    def main(q):
        return [(None, N[f'stat.all.{q}.{d}.a{ak[a]}.median_diff'], N[f'stat.all.{q}.{d}.a{ak[a]}.ci95_lo'],
                 N[f'stat.all.{q}.{d}.a{ak[a]}.ci95_hi'], N[f'stat.all.{q}.{d}.a{ak[a]}.pholm']) for d, a in labels]

    def post(q):
        out_ = []
        for d, a in labels:
            r = t1[(t1.test == q) & (t1.detector == d) & np.isclose(t1.alpha, a)].iloc[0]
            out_.append((None, r.median_diff, r.ci95_lo, r.ci95_hi, r.p_holm))
        return out_
    panels = [(main('cov'), 'ACI − sliding\n|coverage error|', 'aci'), (main('rec'), 'ACI − sliding\nrecall', 'aci'),
              (main('f1'), 'ACI − sliding\nF1', 'aci'),
              (post('unc_vs_cens_gap'), 'Uncensored − sliding\n|coverage error|', 'sliding_all'),
              (post('unc_vs_aci_gap'), 'Uncensored − ACI\n|coverage error|', 'sliding_all'),
              (post('unc_vs_aci_rec'), 'Uncensored − ACI\nrecall', 'sliding_all')]
    fig, axs = plt.subplots(2, 3, figsize=(W2, 4.5), sharey=True, gridspec_kw=dict(wspace=0.1, hspace=0.38))
    for ax, (rows, t, pol) in zip(axs.flat, panels):
        _forest(ax, rows, t, '', col=POLC[pol], mk=POLM[pol])
        lo = min(r[2] for r in rows); hi = max(r[3] for r in rows)
        pad = 0.08 * max(hi - lo, 0.05)
        ax.set_xlim(min(lo - pad, -pad), max(hi + pad, pad))
    for ax in axs[:, 0]:
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels([(f'{DETN[d]}   ' if a == alphas[0] else '') + f'α = {a:.2f}' for d, a in labels], fontsize=6.3)
    for ax in axs.flat:
        for k in range(1, len(DETS)):
            ax.axhline(4 * k - 0.5, color=GRID, lw=0.6, zorder=0)
    fig.supxlabel('Median paired difference over the twelve datasets (seeds averaged per dataset)', fontsize=7, y=0.035)
    h = [Line2D([], [], marker='o', ls='none', color=POLC['aci'], markeredgecolor='white', markersize=5, label='ACI comparison'),
         Line2D([], [], marker='v', ls='none', color=POLC['sliding_all'], markeredgecolor='white', markersize=5, label='uncensored-window comparison'),
         Line2D([], [], marker='o', ls='none', color=INK2, markeredgecolor='white', markersize=4.5, label='filled: Holm-corrected p < 0.05'),
         Line2D([], [], marker='o', ls='none', markerfacecolor='white', markeredgecolor=INK2, markersize=4.5, label='open: not significant'),
         Line2D([], [], ls='-', color=INK2, lw=0.9, label='bootstrap 95% CI')]
    fig.legend(handles=h, loc='upper center', ncol=5, bbox_to_anchor=(0.5, 0.995), handletextpad=0.3, columnspacing=1.2, fontsize=6.4)
    save(fig, out, 'fig_tests')


# ---------------------------------------------------------------- confidence

def fig_confidence(N, out, DS, NAME):
    """Per dataset (mean over detectors, ACI, alpha = 0.05): alert precision and accuracy of
    normal predictions in the low / medium / high confidence tertile."""
    fig, axs = plt.subplots(1, 2, figsize=(W2, 2.6), sharey=True, gridspec_kw=dict(wspace=0.08))
    order = sorted(DS, key=lambda d: N.get(f'{d}.avg.conf.high.prec', np.nan) - N.get(f'{d}.avg.conf.low.prec', np.nan))
    for ax, k, xl in [(axs[0], 'prec', 'Precision of anomaly alerts (%)'), (axs[1], 'nacc', 'Accuracy of normal predictions (%)')]:
        for i, ds in enumerate(order):
            v = {l: 100 * N.get(f'{ds}.avg.conf.{l}.{k}', np.nan) for l in ['low', 'medium', 'high']}
            if not np.isnan(v['low']) and not np.isnan(v['high']):
                up = v['high'] >= v['low']
                ax.annotate('', xy=(v['high'], i), xytext=(v['low'], i),
                            arrowprops=dict(arrowstyle='-|>', color=CONFC['high'] if up else MUTED, ls='-' if up else (0, (2, 1.5)),
                                            lw=0.9, shrinkA=3.5, shrinkB=3.5, mutation_scale=6))
            ax.scatter(v['low'], i, s=20, color=CONFC['low'], edgecolor=CONFC['medium'], lw=0.6, zorder=3)
            ax.scatter(v['medium'], i, s=11, color=CONFC['medium'], edgecolor='white', lw=0.3, zorder=3)
            ax.scatter(v['high'], i, s=20, color=CONFC['high'], edgecolor='white', lw=0.4, zorder=4)
        ax.set_xlim(-3, 103)
        ax.set_xlabel(xl)
        clean(ax)
    axs[0].set_yticks(range(len(order)))
    axs[0].set_yticklabels([NAME[d] for d in order])
    axs[0].tick_params(axis='y', length=0)
    letter(axs[0], '(a) Alerts')
    letter(axs[1], '(b) Normal predictions')
    h = [Line2D([], [], marker='o', ls='none', markerfacecolor=CONFC['low'], markeredgecolor=CONFC['medium'], markersize=5.5, label='low-confidence third'),
         Line2D([], [], marker='o', ls='none', color=CONFC['medium'], markeredgecolor='white', markersize=4.2, label='middle third'),
         Line2D([], [], marker='o', ls='none', color=CONFC['high'], markeredgecolor='white', markersize=5.5, label='high-confidence third'),
         Line2D([], [], color=CONFC['high'], lw=0.9, label='low to high'),
         Line2D([], [], color=MUTED, lw=0.9, ls=(0, (2, 1.5)), label='high below low')]
    fig.legend(handles=h, loc='upper center', ncol=5, bbox_to_anchor=(0.5, 1.07), handletextpad=0.3, columnspacing=1.3)
    save(fig, out, 'fig_confidence')


# ------------------------------------------------------- uncertainty factors

FACN = {'distributional': 'Distributional\n(k-NN distance)', 'density': 'Density-based\n(LOF score)',
        'temporal': 'Temporal\n(first-difference variance)', 'feature_oor': 'Feature, pre-specified\n(max, no reconstruction term)',
        'feature': 'Feature, with per-feature\nreconstruction errors (mean)', 'margin': 'Reference: conformal\nmargin |s − threshold|'}


def fig_uncertainty(N, out, NAME):
    """Top: error rate of ACI's decisions by tenth of each detector-free factor, relative to the
    dataset's error rate (one thin line per dataset, median bold). Bottom: AUC of every factor
    for the wrong decisions (one mark per dataset) and Spearman correlation between factors."""
    a = pd.read_csv(os.path.join(PS, 'p6_auc_per_dataset.csv'), index_col=0)
    t = pd.read_csv(os.path.join(PS, 'p6_auc_tests.csv'))
    c = pd.read_csv(os.path.join(PS, 'p6_corr_per_dataset.csv'), index_col=0)
    dec = pd.read_csv(os.path.join(PS, 'p6_decile_error.csv'))
    ramp = LinearSegmentedColormap.from_list('unc', UNC_RAMP)
    fig = plt.figure(figsize=(W2, 5.1))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.15], hspace=0.62, wspace=0.42)
    top = [('distributional', '(a) Distributional factor'), ('density', '(b) Density-based factor'), ('temporal', '(c) Temporal factor')]
    for k, (f, title) in enumerate(top):
        ax = fig.add_subplot(gs[0, k])
        g = dec[dec.factor == f]
        for ds, gg in g.groupby('dataset'):
            ax.plot(gg.decile, gg.rel_error, color=UNC_RAMP[2], alpha=0.35, lw=0.7, zorder=2)
        med = g.groupby('decile').rel_error.median()
        ax.plot(med.index, med.values, color=UNC_RAMP[4], lw=1.8, marker='o', ms=3.5, markeredgecolor='white', mew=0.4, zorder=4)
        ax.axhline(1, color=MUTED, lw=0.7, ls=(0, (3, 2)), zorder=1)
        ax.set_yscale('log', base=2)
        ax.set_yticks([0.125, 0.25, 0.5, 1, 2, 4, 8]); ax.set_yticklabels(['⅛', '¼', '½', '1', '2', '4', '8'])
        ax.minorticks_off()
        ax.set_ylim(0.09, 9)
        ax.set_xticks([1, 5, 10]); ax.set_xticklabels(['1\nlowest', '5', '10\nhighest'])
        ax.set_xlabel('Tenth of the factor (within dataset)', fontsize=6.6)
        if k == 0:
            ax.set_ylabel('Error rate ÷ dataset error rate')
        clean(ax, grid='y')
        letter(ax, title)
    ax.legend(handles=[Line2D([], [], color=UNC_RAMP[2], alpha=0.6, lw=0.8, label='one dataset'),
                       Line2D([], [], color=UNC_RAMP[4], lw=1.8, marker='o', ms=3.5, markeredgecolor='white', label='median')],
              loc='upper left', fontsize=6.2, handlelength=1.6)
    gb = gs[1, :].subgridspec(1, 2, width_ratios=[2.0, 1.0], wspace=0.9)
    ax = fig.add_subplot(gb[0])
    facs = ['distributional', 'density', 'temporal', 'feature_oor', 'feature', 'margin']
    rng = np.random.default_rng(1)
    for i, f in enumerate(facs):
        v = a[f].dropna().values
        ref = f == 'margin'
        col = MUTED if ref else ramp(0.55 + 0.4 * (np.median(v) - 0.5) / 0.2)
        ax.scatter(v, i + rng.uniform(-0.14, 0.14, len(v)), s=13, marker='o', facecolor='white' if ref else col,
                   edgecolor=MUTED if ref else 'white', lw=0.8 if ref else 0.4, zorder=3)
        ax.plot([np.median(v)] * 2, [i - 0.32, i + 0.32], color=INK, lw=1.5, zorder=4)
        r = t[t.factor == (f if not ref else 'margin (exploratory)')]
        assert len(r) == 1, f
        r = r.iloc[0]
        p = r.p_holm if not ref else r.p
        lab = (f'Holm p = {p:.3f}' if not ref else f'p = {p:.2f}\n(not corrected)') + (f'\nn = {int(r.n)}' if int(r.n) != 12 else '')
        ax.text(1.005, i, lab, transform=matplotlib.transforms.blended_transform_factory(ax.transAxes, ax.transData),
                fontsize=6.0, color=INK2, va='center')
    ax.axvline(0.5, color=MUTED, lw=0.7, ls=(0, (3, 2)), zorder=1)
    ax.axhline(4.5, color=GRID, lw=0.8)
    ax.set_yticks(range(len(facs)))
    ax.set_yticklabels([FACN[f] for f in facs], fontsize=6.3)
    ax.set_ylim(len(facs) - 0.4, -0.6)
    ax.tick_params(axis='y', length=0)
    ax.set_xlim(0.15, 1.0)
    ax.set_xlabel('AUC for wrong ACI decisions (α = 0.05), one mark per dataset')
    clean(ax)
    letter(ax, '(d) Does the factor point to wrong decisions?')
    ax2 = fig.add_subplot(gb[1])
    f5 = ['distributional', 'density', 'temporal', 'feature_oor', 'feature']
    short = {'distributional': 'Distrib.', 'density': 'Density', 'temporal': 'Temporal', 'feature_oor': 'Feat., pre-spec.', 'feature': 'Feat., per-f. err.'}
    M = np.full((5, 5), np.nan)
    for i in range(5):
        for j in range(i):
            k = f'{f5[j]}~{f5[i]}' if f'{f5[j]}~{f5[i]}' in c.columns else f'{f5[i]}~{f5[j]}'
            M[i, j] = c[k].median()
    cmap = LinearSegmentedColormap.from_list('div', ['#c0392b', '#f0efec', UNC_RAMP[3]])
    ax2.imshow(np.ma.masked_invalid(M), cmap=cmap, vmin=-1, vmax=1)
    for i in range(5):
        for j in range(i):
            ax2.text(j, i, f'{M[i, j]:.2f}'.replace('-', '−'), ha='center', va='center', fontsize=6.2,
                     color='white' if abs(M[i, j]) > 0.6 else INK)
    ax2.set_xticks(range(4)); ax2.set_xticklabels([short[f] for f in f5[:4]], rotation=35, ha='right', fontsize=6.2)
    ax2.set_yticks(range(1, 5)); ax2.set_yticklabels([short[f] for f in f5[1:]], fontsize=6.2)
    ax2.set_xlim(-0.5, 3.5); ax2.set_ylim(4.5, 0.5)
    for sp in ax2.spines.values():
        sp.set_visible(False)
    ax2.tick_params(length=0)
    letter(ax2, '(e) Spearman ρ, median')
    save(fig, out, 'fig_uncertainty')


# ----------------------------------------------------------- counterfactuals

def fig_cf(N, out, DS, NAME, DETS, DETN):
    """(a) validity vs share of normal calibration windows below the threshold used
    (time-of-detection protocol, one mark per run); (b) the two protocols side by side;
    (c) ACI minus sliding validity per detector and explainer; (d) plausibility."""
    runs = pd.read_csv(os.path.join(PS, 'p4_runs.csv'))
    both = pd.read_csv(os.path.join(PS, 'p4_vs_exp4.csv'))
    tests = pd.read_csv(os.path.join(PS, 'p4_validity_tests.csv'))
    cells = pd.read_csv(os.path.join(PS, 'p4_cells.csv'))
    meths = ['GA-CE', 'DICE', 'PerCE']
    fig, axs = plt.subplots(2, 2, figsize=(W2, 5.2), gridspec_kw=dict(hspace=0.5, wspace=0.32, top=0.91))
    # (a)
    ax = axs[0, 0]
    bins = np.linspace(0, 1, 11)
    for m in meths:
        x = runs[runs.method == m].dropna(subset=['validity', 'pass_rate'])
        ax.scatter(100 * x.pass_rate, 100 * x.validity, s=5, marker=EXM[m], color=EXC[m], alpha=0.35, lw=0, zorder=2, rasterized=True)
        b = pd.cut(x.pass_rate, bins)
        gb = x.groupby(b, observed=True).validity.agg(['mean', 'size'])
        gb = gb[gb['size'] >= 5]                                   # bins with at least 5 runs
        mid = [100 * (i.left + i.right) / 2 for i in gb.index]
        ax.plot(mid, 100 * gb['mean'].values, color=EXC[m], lw=1.3, marker=EXM[m], ms=3.5, markeredgecolor='white', mew=0.4,
                zorder=4, label=m)
    ax.set_xlim(-2, 102); ax.set_ylim(-3, 103)
    ax.set_xlabel('Normal calibration windows below the threshold used (%)')
    ax.set_ylabel('Validity (%)')
    clean(ax, grid='both')
    letter(ax, '(a) Validity and threshold position')
    # (b)
    ax = axs[0, 1]
    ax.plot([0, 100], [0, 100], color=MUTED, lw=0.7, zorder=1)
    for m in meths:
        for pol, fill in [('aci', True), ('sliding', False)]:
            x = both[(both.method == m) & (both.policy == pol)]
            ax.scatter(100 * x.validity_exp4, 100 * x.validity, s=11, marker=EXM[m], facecolor=EXC[m] if fill else 'white',
                       edgecolor='white' if fill else EXC[m], lw=0.4 if fill else 0.8, zorder=3)
    ax.set_xlim(-3, 103); ax.set_ylim(-3, 103)
    ax.set_xlabel('Validity, end-of-stream threshold (%)')
    ax.set_ylabel('Validity, time-of-detection threshold (%)')
    clean(ax, grid='both')
    letter(ax, '(b) The two protocols (filled: ACI, open: sliding)')
    # (c)
    ax = axs[1, 0]
    rows = [(d, m) for d in DETS for m in meths]
    for i, (d, m) in enumerate(rows):
        r = tests[(tests.detector == d) & (tests.method == m)].iloc[0]
        sig = bool(r.p_holm < 0.05)
        ax.plot([100 * r.ci95_lo, 100 * r.ci95_hi], [i, i], color=EXC[m], lw=0.9, zorder=2)
        ax.scatter(100 * r.median_diff, i, s=18, marker=EXM[m], facecolor=EXC[m] if sig else 'white',
                   edgecolor='white' if sig else EXC[m], lw=0.4 if sig else 0.9, zorder=3)
    ax.axvline(0, color=MUTED, lw=0.6)
    for k in range(1, len(DETS)):
        ax.axhline(3 * k - 0.5, color=GRID, lw=0.6)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([(f'{DETN[d]}   ' if m == meths[0] else '') + m for d, m in rows], fontsize=6.3)
    ax.set_ylim(len(rows) - 0.4, -0.6)
    ax.tick_params(axis='y', length=0)
    ax.set_xlabel('Validity difference, median over datasets (points)')
    clean(ax)
    letter(ax, '(c) ACI − sliding (filled: Holm-corrected p < 0.05)')
    # (d)
    ax = axs[1, 1]
    rng = np.random.default_rng(2)
    yl = []
    for i, m in enumerate(meths):
        v = cells[(cells.method == m) & (cells.policy == 'aci')].cf_plaus_valid.dropna().values
        ax.scatter(v, i + rng.uniform(-0.15, 0.15, len(v)), s=8, marker=EXM[m], color=EXC[m], edgecolor='white', lw=0.3, zorder=3)
        if len(v):
            ax.plot([np.median(v)] * 2, [i - 0.3, i + 0.3], color=INK, lw=1.4, zorder=4)
        yl.append(f'{m} (valid)')
    v = cells[(cells.method == 'GA-CE') & (cells.policy == 'aci')].anom_plaus.dropna().values
    ax.scatter(v, 3 + rng.uniform(-0.15, 0.15, len(v)), s=8, facecolor='white', edgecolor=MUTED, lw=0.7, zorder=3)
    if len(v):
        ax.plot([np.median(v)] * 2, [2.7, 3.3], color=INK, lw=1.4, zorder=4)
    yl.append('explained anomalies')
    ax.axvline(1, color=MUTED, lw=0.7, ls=(0, (3, 2)))
    ax.set_xscale('log')
    ax.set_yticks(range(4)); ax.set_yticklabels(yl, fontsize=6.4)
    ax.set_ylim(3.5, -0.6)
    ax.tick_params(axis='y', length=0)
    ax.set_xticks([0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500])
    ax.set_xticklabels(['0.5', '1', '2', '5', '10', '20', '50', '100', '200', '500'])
    ax.minorticks_off()
    ax.set_xlabel('Plausibility (1 = typical normal window; log scale)')
    clean(ax)
    letter(ax, '(d) Plausibility under ACI')
    h = [Line2D([], [], marker=EXM[m], ls='-', lw=1.1, color=EXC[m], markeredgecolor='white', markersize=4.5, label=m) for m in meths]
    fig.legend(handles=h, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.0), handletextpad=0.3, columnspacing=1.6)
    save(fig, out, 'fig_cf')


# ------------------------------------------------------ case study: uncertainty

CS = os.path.join(PH, 'results', 'case_study')


def short_name(n):
    return {'Accelerometer1RMS': 'Accel. 1 RMS', 'Accelerometer2RMS': 'Accel. 2 RMS',
            'Volume Flow RateRMS': 'Volume flow RMS'}.get(n, n)


def _anom_spans(ax, yv):
    d = np.diff(np.concatenate([[0], yv, [0]]))
    for a, b in zip(np.where(d == 1)[0], np.where(d == -1)[0]):
        ax.axvspan(a - 0.5, b - 0.5, color=ANOM, lw=0, zorder=0)


def fig_stream(N, out):
    """What the operator sees along one test stream (SWaT, LSTM-AE, seed 42, ACI, alpha = 0.05):
    (a) score and threshold, (b) confidence of each decision, (c) the uncertainty factors,
    (d) the feature-level scores of one flagged window (first true anomaly flagged by ACI)."""
    z = np.load(os.path.join(CS, 'stream_swat.npz'))
    sel = json.load(open(os.path.join(PS, 'case_selection.json')))['stream_sel']
    y, flag, s, thr, conf = z['y'].astype(int), z['flag'].astype(int), z['score'], z['threshold'], z['confidence']
    x = np.arange(len(s))
    wrong = flag != y
    ramp = LinearSegmentedColormap.from_list('unc', UNC_RAMP)
    fig = plt.figure(figsize=(W2, 4.6))
    gs = fig.add_gridspec(3, 2, width_ratios=[2.6, 1], height_ratios=[1.35, 0.85, 0.95], hspace=0.38, wspace=0.42)
    a = fig.add_subplot(gs[0, 0])
    _anom_spans(a, y)
    a.scatter(x[y == 0], s[y == 0], s=0.7, color='#c8c6bd', lw=0, zorder=1, rasterized=True)
    a.scatter(x[y == 1], s[y == 1], s=0.9, color=INK2, lw=0, zorder=1, rasterized=True)
    fin = np.isfinite(thr)
    a.plot(x, np.where(fin, thr, np.nan), color=POLC['aci'], lw=0.6, zorder=3)
    a.set_yscale('log')
    a.axvline(sel, color=UNC_RAMP[4], lw=0.9, ls=(0, (3, 2)), zorder=4)
    a.set_ylabel('Score')
    clean(a, grid='y')
    plt.setp(a.get_xticklabels(), visible=False)
    letter(a, '(a) Nonconformity score and ACI threshold')
    a.legend(handles=[Line2D([], [], color=POLC['aci'], lw=1.0, label='ACI threshold'),
                      Line2D([], [], marker='o', ls='none', color='#c8c6bd', markersize=2.5, label='normal'),
                      Line2D([], [], marker='o', ls='none', color=INK2, markersize=2.5, label='anomalous'),
                      matplotlib.patches.Patch(color=ANOM, label='attack period'),
                      Line2D([], [], color=UNC_RAMP[4], lw=0.9, ls=(0, (3, 2)), label='window in (d)')],
             loc='lower left', bbox_to_anchor=(0, 1.1), ncol=5, fontsize=6.1, handlelength=1.4, columnspacing=0.9)
    b = fig.add_subplot(gs[1, 0], sharex=a)
    _anom_spans(b, y)
    for f_, c_, lab in [(0, '#c8c6bd', 'normal decision'), (1, POLC['aci'], 'alert')]:
        m = (flag == f_) & ~wrong
        b.scatter(x[m], conf[m], s=1.0, color=c_, lw=0, zorder=2, rasterized=True)
    b.scatter(x[wrong], conf[wrong], s=2.2, color=WRONG, lw=0, zorder=3, rasterized=True)
    b.axvline(sel, color=UNC_RAMP[4], lw=0.9, ls=(0, (3, 2)), zorder=4)
    b.set_ylim(0.48, 1.01)
    b.set_ylabel('Confidence')
    clean(b, grid='y')
    plt.setp(b.get_xticklabels(), visible=False)
    letter(b, '(b) Confidence')
    b.legend(handles=[Line2D([], [], marker='o', ls='none', color='#c8c6bd', markersize=3, label='correct normal'),
                      Line2D([], [], marker='o', ls='none', color=POLC['aci'], markersize=3, label='correct alert'),
                      Line2D([], [], marker='o', ls='none', color=WRONG, markersize=3, label='wrong decision')],
             loc='lower right', bbox_to_anchor=(1.0, 0.98), ncol=3, fontsize=6.0, handletextpad=0.2, columnspacing=0.8, borderaxespad=0)
    c = fig.add_subplot(gs[2, 0], sharex=a)
    M = np.vstack([z['dist_score'], z['dens_score'], z['temp_score'], z['total']])
    c.imshow(M, aspect='auto', cmap=ramp, vmin=0, vmax=1, interpolation='nearest', extent=(-0.5, len(s) - 0.5, 3.5, -0.5))
    c.set_yticks(range(4)); c.set_yticklabels(['Distributional', 'Density-based', 'Temporal', 'Overall u'], fontsize=6.3)
    c.axvline(sel, color=INK, lw=0.9, ls=(0, (3, 2)))
    c.axhline(2.5, color='white', lw=1.2)
    c.tick_params(axis='y', length=0)
    for sp in c.spines.values():
        sp.set_visible(False)
    c.set_xlabel('Test window (time order)')
    letter(c, '(c) Uncertainty factors, score in [0, 1]')
    cax = c.inset_axes([1.01, 0.0, 0.012, 1.0])
    cb = fig.colorbar(matplotlib.cm.ScalarMappable(cmap=ramp, norm=matplotlib.colors.Normalize(0, 1)), cax=cax)
    cb.set_ticks([0, 0.5, 1]); cb.ax.tick_params(labelsize=6, length=2); cb.outline.set_visible(False)
    d = fig.add_subplot(gs[:, 1])
    names = [short_name(str(n)) for n in z['names']]
    fs = z['feature_scores'][sel]
    o = np.argsort(-fs, kind='mergesort')[:10]
    d.barh(range(len(o)), fs[o], color=[ramp(0.35 + 0.65 * v) for v in fs[o]], height=0.68, zorder=3)
    d.set_yticks(range(len(o))); d.set_yticklabels([names[i] for i in o], fontsize=6.4)
    d.set_ylim(len(o) - 0.4, -0.6)
    d.set_xlim(0, 1)
    d.set_xlabel('Feature-level score')
    clean(d)
    d.tick_params(axis='y', length=0)
    letter(d, '(d) Sensors behind window ' + str(sel))
    info = (f"score {s[sel]:.3g}, threshold {thr[sel]:.3g}\n"
            f"confidence {conf[sel]:.2f}\n"
            f"distributional {z['dist_score'][sel]:.1f}, density {z['dens_score'][sel]:.1f},\n"
            f"temporal {z['temp_score'][sel]:.1f}, overall u {z['total'][sel]:.2f}")
    d.text(0.98, 0.02, info, transform=d.transAxes, ha='right', va='bottom', fontsize=6.0, color=INK2,
           bbox=dict(boxstyle='round,pad=0.35', fc='white', ec=GRID, lw=0.6))
    save(fig, out, 'fig_stream')


# ------------------------------------------------- case study: counterfactuals

def fig_cf_example(N, out):
    """Counterfactuals of the three explainers for two SKAB windows (seed 42, ACI, time of
    detection; selection rules in case_study.py). Left: the three sensors that change most,
    with the window, its nearest normal training window and the counterfactuals. Right:
    absolute change per sensor and time step for each explainer, on a common scale per row."""
    e = np.load(os.path.join(CS, 'examples.npz'))
    meta = json.load(open(os.path.join(CS, 'case_study.json')))['examples']
    sel = json.load(open(os.path.join(PS, 'case_selection.json')))
    names = [short_name(str(n)) for n in e['names']]
    meths = ['GA-CE', 'DICE', 'PerCE']
    detn = {'lstm_ae': 'LSTM-AE', 'lstm_vae': 'LSTM-VAE'}
    fig = plt.figure(figsize=(W2, 6.0))
    outer = fig.add_gridspec(2, 1, hspace=0.5, top=0.91)
    for r, key in enumerate(['A', 'B']):
        m = meta[key]
        x, near = e[f'{key}_x'], e[f'{key}_near']
        g = outer[r].subgridspec(3, 5, width_ratios=[2.25, 0.62, 1, 1, 1], hspace=0.12, wspace=0.12)
        top = sel[f'ex{key}_sensors']
        t = np.arange(x.shape[0])
        for k, j in enumerate(top):
            ax = fig.add_subplot(g[k, 0])
            ax.plot(t, near[:, j], color=MUTED, lw=1.0, ls=(0, (3, 1.5)), zorder=2)
            for mm in meths:
                cf = e[f'{key}_{mm}']
                ok = m['methods'][mm]['valid']
                ax.plot(t, cf[:, j], color=EXC[mm], lw=1.5 if ok else 0.9, ls='-' if ok else (0, (1.2, 1.2)), zorder=3)
            ax.plot(t, x[:, j], color=INK, lw=0.9, zorder=5)
            ax.set_ylabel(names[j], fontsize=6.2, labelpad=2)
            ax.tick_params(labelsize=5.8)
            clean(ax, grid='y')
            if k < 2:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel('Time step in the window', fontsize=6.4)
            if k == 0:
                kind = 'true anomaly' if m['true_anomaly'] else 'false alarm'
                ax.set_title(f"({'ab'[r]}) {detn[m['detector']]}, window {m['test_index']}, {kind}",
                             loc='left', fontsize=7.2, fontweight='bold', pad=4)
        diffs = {mm: np.abs(e[f'{key}_{mm}'] - x) for mm in meths}
        vmax = np.quantile(np.concatenate([v.ravel() for v in diffs.values()]), 0.99)
        for k, mm in enumerate(meths):
            ax = fig.add_subplot(g[:, k + 2])
            cmap = LinearSegmentedColormap.from_list(mm, ['#ffffff', EXC[mm]])
            ax.imshow(diffs[mm].T, aspect='auto', cmap=cmap, vmin=0, vmax=vmax, interpolation='nearest')
            v = m['methods'][mm]
            ax.set_title(mm, fontsize=7, color=EXC[mm] if mm != 'DICE' else '#b07800', fontweight='bold', loc='center', pad=3)
            ax.set_xlabel(f"{'valid' if v['valid'] else 'not valid'}\nprox {v['proximity']:.2f} · plaus {v['plausibility']:.1f}\n"
                          f"{100 * v['sparsity']:.0f}% unchanged", fontsize=6.0,
                          color=INK if v['valid'] else MUTED, labelpad=2)
            ax.set_yticks(range(len(names)))
            ax.set_yticklabels(names if k == 0 else [], fontsize=5.9)
            ax.set_xticks([])
            ax.tick_params(length=0)
            for sp in ax.spines.values():
                sp.set_color(GRID)
            for j in top:
                ax.add_patch(matplotlib.patches.Rectangle((-0.5, j - 0.5), x.shape[0], 1, fill=False, ec=INK2, lw=0.5))
    h = [Line2D([], [], color=INK, lw=0.9, label='flagged window'), Line2D([], [], color=MUTED, lw=1.0, ls=(0, (3, 1.5)), label='nearest normal training window')]
    h += [Line2D([], [], color=EXC[mm], lw=1.5, label=mm) for mm in meths]
    h += [Line2D([], [], color=INK2, lw=0.9, ls=(0, (1.2, 1.2)), label='not valid (dotted)')]
    fig.legend(handles=h, loc='upper center', ncol=6, bbox_to_anchor=(0.5, 0.995), fontsize=6.3, handlelength=1.8, columnspacing=1.1)
    save(fig, out, 'fig_cf_example')
