# Leakage and fairness audit (written 2026-10-14, before any full run)

Purpose: list every place where labels, test data or results could influence
a number, and what we do about it. Nothing here was found by looking at
results; the full runs had not started when this was written.

## A. Paper code (`xad/`, `experiments/`), unchanged and reused as is

| # | what | where | verdict |
|---|---|---|---|
| A1 | **F1\* uses the oracle threshold**: the F1-maximising cut chosen on TEST labels | `exp_1_detection_performance.py:39-46,82`; `exp_8_classical_baselines.py:32-35` | Optimistic by construction. The paper already labels it (Sec. 4 setup, Table 1 caption: "oracle score threshold ... upper bound"). Extended tables must keep the F1\* name and that caption. The deployable F1 is the one at the conformal threshold (exp_6, per policy), and it is reported next to F1\* for every dataset. |
| A2 | Training uses only normal training windows (needs train labels) | `pipeline.prepare_split` -> `BaseDataLoader.normal_only` | Standard semi-supervised AD assumption, same as the paper. For the 7 mTSBench sets the train files are anomaly-free anyway. |
| A3 | Conformal calibration and early stopping use only normal VAL windows (needs val labels) | same | Same assumption as the paper (a clean normal calibration set). Disclosed. |
| A4 | Online threshold updates (sliding window, ACI) during the test stream | `temporal_conformal_predictor.py:323-340` | **Label-free.** ACI's `err_t` is 1 if the window was *flagged*, not if it was truly anomalous (line 332); the sliding buffer is fed by the model's own decisions. No test label reaches the threshold. |
| A5 | Counterfactual experiments explain the windows the detector *flagged* | `exp_4_counterfactual_quality.py:149-163` | No label filter, so false alarms are explained too (realistic, not cherry-picked). The subsample of <= 100 flagged windows is drawn with the run's seed and shared by all explainers. `n_anomalies` (how many were explained) is reported, since it can be < 100. |
| A6 | Hyperparameters | sbatch files | Copied unchanged from the SKAB/SWaT 5-seed arrays. **No tuning on any extra dataset.** lambda_temp = 0 (paper default, not re-selected). |
| A7 | LOF / OC-SVM fit on a 5,000-window subsample of normal training windows | `exp_8_classical_baselines.py:42-45` | Part of the paper's script; disclosed in the paper; applies here too. |
| A8 | Seeds | sbatch | 42-46 for every dataset and detector; no seed is dropped or re-drawn. A failed task is reported as a failure (n_seeds < 5), never silently rerun with another seed. |

## B. Dataset preparation (`code/prepare.py`), decisions that touch labels

None of these used model outputs or results; each is recorded in that
dataset's `meta.json` and log.

| # | dataset | decision | label use |
|---|---|---|---|
| B1 | metropt3 | split dates (train < 03-20, val < 04-10, test after) chosen so all 4 reported failures fall in test | failure dates (public metadata) set the split; train/val contain no failure |
| B2 | threew | fault instances truncated 30 min after onset; steady-state tail dropped | onset = first labelled point; applied identically to train/val/test; done to meet the rare-anomaly criterion (otherwise 43-51% anomalous) |
| B3 | threew | instance-level 60/20/20 split stratified by event type (seed 42) | event type used only for stratification |
| B4 | swan_sf | excluded | 76% of test windows anomalous (label statistics only), fails the rare-anomaly criterion |
| B5 | all | preset (window length) chosen by VAL size | no labels |
| B6 | all | constant / low-coverage features dropped, medians for NaN fill | computed on TRAIN rows only |
| B7 | psm | kept although 32% of test points are anomalous | standard benchmark; flagged in every table |
| B8 | threew | physically impossible readings and missing-value codes set to missing, by unit limits (pressure 1e8 Pa, temperature -50..500 C, flow 100 m3/s, choke -5..110 %, state 0..1). Found because the smoke run produced NaN scores (train inputs up to 1.4e20 after scaling). Masked: T-PDG -999 and ABER-CKP/T-MON-CKP -99.99 codes, P-JUS-CKP 1.2-1.5e9 Pa in 3 WELL-00026 files (unit error), P-PDG <= -1e8 / 1.2e42, T-PDG 1.7e38 / 2.9e15, P-TPT 2.9e9. 4.39M cells total, counts per column in logs/prepare/threew.log | no labels; limits chosen by inspecting the masked values file by file (a first 0..100 % choke limit wrongly masked 334 real 100.2-100.4 % readings and was widened). Re-prepared before any full run; checksum re-frozen |
| B9 | pronto | excluded; replaced by msl | label statistics only: the authors' aligned+labelled process data (Testday2-4_merged.csv) is 45,420 rows (< 100k volume criterion of Dataset_notes) and fault rows are 94% and 80% of the two induced-fault test days, the third day containing only normal operation and slugging (fails the rare-anomaly criterion; correction 2026-10-21: earlier text said 81-94%, recomputed exactly from Testday2-4_merged.csv; the same wrong range remains in the frozen prepared/msl/meta.json 'replaces' field); slugging is an operating regime, not a fault. Shrinking fault episodes until anomalies look rare would tailor the data to the method, so it was not done. Decided before any full run |

## C. Process safeguards

* Prepared data frozen with md5 in `PREPARED_CHECKSUMS.txt` before submission. Every SLURM task re-checks the md5 at start and refuses to run on a mismatch (added after the B8 smoke incident, where re-preparing during a running smoke task paired a cached 13-feature model with 12-feature data).
* Every submission (job ids, code md5) is appended to `RUNLOG.md` automatically.
* Smoke-test outputs (`smoke/`) use 2 epochs and are never used for any number.
* The dataset list and the analysis are fixed in `ANALYSIS_PLAN.md` before results exist; all 10 datasets are reported whatever their results.

## D. Post-hoc analyses (P1-P6), findings that affect how results are read

* D1 (2026-10-21) P1 does not reproduce exp_6 bit for bit. P1 re-scores the cached
  5-seed detectors; exp_6 scored them in the training job. Over 2160 runs (12 datasets
  x 3 detectors x 5 seeds x {static, sliding, ACI} x 4 alpha), per-run coverage
  differs by at most 0.0010 (static), 0.0004 (ACI) and 0.083 (censored sliding;
  SMAP conv_ae seed 45, alpha 0.15: 0.110 vs 0.193). Cause, checked rather than
  assumed: GPU model (sacct AllocTRES, cached in posthoc/summary/job_gpus.csv).
  When the original task and the P1 task ran on the same GPU model (128 runs per
  policy), 99% of static, 100% of ACI and 95% of sliding runs are identical
  (largest sliding difference 0.0019); on different models (592 runs per policy)
  59% of sliding runs are identical. Scores therefore agree to float precision, and
  the censored window, which feeds its own decisions back into the threshold,
  amplifies the last-digit differences. Seed-averaged cells, the unit of every
  table: largest coverage difference 0.0002 (static), 0.0001 (ACI), 0.017
  (sliding); sliding is within 3 points of target in 0/144 cells in both; static
  54/144 in both; ACI 127/144 (exp_6) vs 126/144 (P1), the one cell being PSM
  LSTM-VAE alpha 0.20 with gap 0.02996 vs 0.03000. Handling: the main tables keep
  exp_6; every P1 comparison (uncensored vs censored vs ACI) uses P1's own scores,
  so all policies in a comparison see identical scores. Both counts are reported
  where they appear. Table: posthoc/summary/p1_reproduction_check.csv.
* D2 (2026-10-21) Analysis environment. analyze.py, make_paper.py and
  posthoc_analyze.py must run in the conda env "xad" (scipy 1.15.3, numpy 2.2.6,
  pandas 2.3.3). One interactive draft build of the paper was run by mistake with the
  cluster's base Python (scipy 1.13.1). For Wilcoxon tests with zero or tied paired
  differences the two scipy versions return different p-values; this changed the
  Holm-corrected p of the DICE and PerCE validity tests (e.g. Conv-AE DICE 0.035 in xad,
  0.056 in base). The draft was discarded. The three scripts now refuse to run unless
  the pinned versions are loaded (analyze.check_env), the paper's numbers.json records
  the environment, and code/run_xad.sh runs a script in the right environment. The
  post-hoc summaries were regenerated from scratch in xad; all 920 values agree with the
  base-Python run (copy kept in posthoc/summary_basepy_DO_NOT_USE). statsmodels is not
  in xad, so the clustered regression of P3 is implemented in posthoc_analyze.ols_cluster;
  on a test data set it agrees with statsmodels OLS(cov_type='cluster') to 10 decimals.
* D3 (2026-10-21) Node failure. Task 14 of the SWaT supplementary array 25752064
  (lstm_vae, seed 46) ended NODE_FAIL (sacct) during its last step, exp_5 under ACI,
  after exp_3, exp_7 and exp_5 (sliding) had written their results. Only that step was
  rerun, with identical arguments (FIVE_SEED_RUN/supp_5seed_rerun_exp5aci.sbatch, job in
  RUNLOG.md). Nothing else of the task was repeated or replaced.
* D4 (2026-10-22) Paper presentation checks, after compiling the paper with tectonic
  (scratch environment, not part of the project). Found and fixed: (a) the figure PDFs
  embedded the system's CFF Nimbus Sans as TrueType ("Mismatch between font type and
  embedded font file" in poppler); the converted fonts are now renamed "Nimbus Sans XAD"
  (code/fonts) so matplotlib cannot pick the system OTF, and every figure embeds Type 42
  TrueType without warnings; (b) \citet printed "(author?)" with elsarticle-num; the
  build now uses elsarticle-num-names; (c) "Appendix Appendix X" from \ref to appendix
  labels; (d) the counterfactual figure floated into the appendix; float parameters and a
  \FloatBarrier before the back matter keep all ten figures in the main text;
  (e) references.bib: duplicate darban2024deep (identical) and wang2025distributed
  (second copy had wrong first names; DBLP confirms the first copy, volume 645 added)
  removed, sun2024model completed (TMLR 2025), xu2021conformal typed as ICML paper.
  Figure 1 (overview.png) copied from the author's CAS-template folder
  What__How_Confident__and_Why/figs/.
