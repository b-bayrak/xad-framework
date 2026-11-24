# Analysis plan (fixed 2026-10-14, before any full result exists)

Written before the full 5-seed runs were submitted, so the analysis cannot be
shaped by the results. Any later deviation gets added at the bottom under
"Deviations" with a date and reason; nothing above is edited.

## Scope

All 10 extra datasets (daphnet, exathlon, gecco, ghl, metropt3, msl, psm, smap, smd, threew), x
3 detectors (Conv-AE, LSTM-AE, LSTM-VAE), x 5 seeds (42-46), with the paper's
hyperparameters. Every dataset is reported, including ones where the method
does poorly. SKAB and SWaT 5-seed results (`FIVE_SEED_RUN/`) are shown
alongside for reference, not pooled into the new-dataset statistics.

## Questions, each mapped to the paper claim it re-tests

| Q | paper claim | measured by | source |
|---|---|---|---|
| Q1 | Sliding-window recalibration under-covers (censoring ratchet) | coverage and calibration error per policy (static / sliding / ACI), alpha in {0.05, 0.10, 0.15, 0.20} | exp_2, exp_6 |
| Q2 | ACI holds coverage (within ~3 pts) | same | exp_6 |
| Q3 | ACI pays in recall during anomaly bursts; no policy dominates | F1 / precision / recall **at the conformal threshold**, per policy | exp_6 |
| Q4 | Counterfactual validity inherits the threshold policy's calibration | validity, proximity, sparsity per explainer (GA-CE, DICE, PerCE) under sliding vs ACI; n explained | exp_4 (both passes) |
| Q5 | Confidence is informative | precision / accuracy by confidence tertile under ACI | exp_3 (supp) |
| Q6 | GA-CE loss terms each matter | ablation variants, sliding and ACI | exp_5 (supp) |
| Q7 | Policy hyperparameters (gamma, w) are not cherry-picked | sensitivity curves | exp_7 (supp) |
| Q8 | Detectors have real ranking power | ROC-AUC, PR-AUC (with prevalence as the chance line), F1\* (oracle, labelled as upper bound); vs IsolationForest, LOF, OC-SVM | exp_1, exp_8 |

## Reporting rules

1. Per dataset x detector: mean +/- std over seeds **and n_seeds**. 95% t-intervals in the appendix tables.
2. F1\* is always named F1\* and footnoted "oracle threshold on test labels, upper bound". The deployable F1 (conformal threshold) sits next to it.
3. PR-AUC is printed with the test prevalence, since its chance level differs per dataset (1.6% GHL vs 34% PSM).
4. Flags carried in every table: PSM (anomalies not rare, 34% of test windows), Exathlon (single trace, 5 informative features), GHL (simulated plant), 3W (35% of cells imputed; faults truncated after 30 min).
5. Failures (task crashed, no anomalies flagged so counterfactuals undefined, NaN metrics) are shown as such, with the reason, never dropped or imputed.
6. `experiments/check_consistency.py` is run on every result folder; its output is saved next to the aggregates and any inconsistency is reported.

## Statistics (decided now)

* Unit of analysis for cross-dataset claims: dataset (n = 10), averaging the 5 seeds and reporting detectors separately (and the detector-average).
* Q1/Q2 policy comparison: per dataset x detector, coverage gap to target (1 - alpha) for sliding vs ACI; across datasets, two-sided Wilcoxon signed-rank on the paired gaps, with Holm correction over the 3 detectors x 4 alphas. Effect size: median paired difference plus its bootstrap 95% CI.
* Q3: same test on recall and F1 at the conformal threshold.
* Q4/Q6 explainer comparison: Friedman test over datasets, Nemenyi post-hoc, critical-difference diagram; validity reported both on its own and jointly with proximity (validity alone can favour large edits).
* Q8: Wilcoxon of each deep detector vs IsolationForest on ROC-AUC and PR-AUC across datasets.
* Per-dataset seed-level tests (n = 5) are descriptive only; no significance claims from 5 seeds on one dataset.
* Every test is reported with its statistic and p, including non-significant ones.

## What would count against the paper's claims (stated in advance)

* Q1 fails if sliding is within 3 pts of target on most new datasets.
* Q2 fails if ACI misses target by > 3 pts on more than a few datasets.
* Q4 fails if explainer validity does not differ between sliding and ACI.
These outcomes are reported as found.

## Deviations

* 2026-10-14 (before any full run): pronto excluded and msl added as dataset #10 (AUDIT.md B9). The dataset list is now closed. If a Kaggle token becomes available, HAI is added as an 11th dataset and reported in full; it does not replace any of the 10.
* 2026-10-14 (before any full run): threew re-prepared after the smoke test exposed corrupted raw values (AUDIT.md B8).
* 2026-10-14 (before any full run; additive only): the Q2 verdict table additionally splits ACI misses into over-coverage and under-coverage. Reason: the paper's "within 3 pts" figure is SWaT-specific (Sec. 4: <= 2.8 pts on SWaT; on SKAB ACI over-covers by 1.6-6.4 pts), so direction matters. The pre-stated criterion itself (|gap| > 3 pts) is unchanged and still reported.
* 2026-10-21 (AFTER results were seen; exploratory): Section 8 of the report adds a sensitivity check that repeats the main tests with datasets removed (leave-one-dataset-out for all 10, plus SMAP and Daphnet together). Prompted by the question whether SMAP (detection below chance) and Daphnet should be dropped. Decision: neither is dropped; all 10 remain in every main table and test. The check is reported as exploratory and does not replace any pre-registered result.
* 2026-10-21 (correction of a stated fact, no change to any decision): PRONTO's fault share is 94% and 80% on its two induced-fault test days (third day: no faults), not 81-94% as first written in AUDIT.md B9.
* 2026-10-21 (AFTER results; presentation and main analysis): at the author's request all twelve datasets are treated identically in the paper (one alphabetical list, no primary/extended split). The main-text tests are therefore run over all twelve datasets (paper build v2, `code/make_paper.py`). The pre-specified ten-dataset analysis is unchanged, still reported in full (paper Appendix "Pre-specified Analysis", and `analysis/REPORT.md`), and reaches the same conclusions; the paper says so. The leave-one-dataset-out check in the paper is likewise run over twelve datasets.

## Post-hoc analyses (specified 2026-10-21, BEFORE any of them was run)

Added in response to anticipated reviewer objections. They are post hoc relative to
the main results (which were known), but each analysis below, its metrics, and what
will be reported were fixed before running it. All twelve datasets, all three
detectors, seeds 42-46, no tuning; every result is reported whatever it shows.

* P1 Uncensored sliding control. A sliding window that admits every test score
  (not only scores classified normal). Report coverage, recall, F1 for static,
  sliding (censored), sliding (uncensored), ACI at all alphas. Test: |coverage gap|
  uncensored vs censored, Wilcoxon over 12 datasets, Holm over detector x alpha.
  Expectation stated in advance: removing censoring removes most of the
  under-coverage; if it does not, the censoring explanation in the paper is wrong
  and the text must say so.
* P2 Matched false-alarm recall. For each policy, the recall an oracle fixed
  threshold would reach at the same realised false-alarm rate (labelled as an
  oracle reference). Report achieved / oracle recall per policy.
* P3 Counterfactual validity vs threshold position. For each configuration, the
  share of normal test windows that pass the final threshold the counterfactuals
  are checked against. Report Spearman correlation with validity per explainer and
  a regression of validity on pass rate with a policy indicator.
* P4 Plausibility. Re-generate the exp_4 counterfactuals with the same seeds,
  save them, and report the 1-NN distance to normal training windows divided by
  the median 1-NN distance of normal validation windows (<=1 = as close to the
  training data as typical normal data). Also check that regenerated validity
  matches exp_4.
* P5 Isolation Forest backbone. IsolationForest (200 trees, seeds 42-46) scores
  through the same conformal pipeline: all four policies, and GA-CE and PerCE
  under sliding and ACI (DICE needs gradients and is not run). Same tests as P1.
* P6 Uncertainty decomposition. On up to 1000 test windows per configuration:
  Spearman correlations between the continuous factor values (k-NN distance, LOF
  score, temporal-variance percentile, max feature out-of-range rate) and the AUC
  of each factor for predicting wrong ACI decisions at alpha = 0.05.
* 2026-10-21, found while smoke-testing P1 (BEFORE P3-P5 were run): the counterfactual
  protocol of exp_4 checks validity against the threshold in force at the END of the
  test stream. Under ACI, alpha_t can drift below 1/(m+1) by then, which makes the
  threshold +inf and every counterfactual trivially valid: this happens in 31 of 180
  ACI configurations (all GECCO and Exathlon runs, one PSM run; mean validity 100%
  there vs 47/39/24% for GA-CE/PerCE/DICE elsewhere). SKAB and SWaT are not affected.
  A second inconsistency: the explainers steer their search with `cp.threshold`,
  which ACI never updates (it stays the static calibration threshold), while
  validity is checked against ACI's current threshold. Under the sliding window both
  agree, but the end-of-stream threshold is its most ratcheted point.
  Change to P4 and P5 (decided before running them): each counterfactual is generated
  and checked against a snapshot of the conformal predictor taken at the moment its
  window was flagged, with the search threshold set to the threshold actually used for
  that decision ("time-of-detection protocol"). P3 uses the share of normal
  calibration windows that pass that same threshold. The paper reports both the
  original end-of-stream protocol (exp_4) and the time-of-detection protocol, and
  states the problem with the former.
* 2026-10-21, found while smoke-testing P6 (BEFORE P6 was submitted): (a) the
  pre-specified scalar for the feature factor (largest per-feature out-of-range rate)
  is constant on GECCO, so its correlations are undefined and its AUC is 0.5. (b) A
  bug in xad/uncertainty_explainer.py: xad.py passes reconstruction errors of shape
  (1, T, F) and `_compute_feature_uncertainty` indexes `reconstruction_errors[:, i]`,
  which selects TIME STEP i, not feature i, so the reconstruction part (weight 0.4)
  of each feature's uncertainty is taken from an unrelated time step. No reported
  experiment uses this function (only the worked examples in the paper appendix do).
  xad/ is not modified here (results are tied to its checksum); the fix is proposed
  to the author. P6 therefore reports: the pre-specified feature scalar as planned,
  and in addition the mean feature uncertainty computed with per-feature
  reconstruction errors (shape F, the evident intent), per detector and seed.

* 2026-10-21, after P1 ran: P1 does not reproduce exp_6 exactly because the cached
  detectors were re-scored on a different GPU model for most runs (AUDIT.md D1). Not a
  change of analysis; recorded because the P1 script (posthoc_policies.py) states that
  the static/sliding/ACI rows "must reproduce exp_6 exactly". P1 comparisons use P1's own scores throughout.
* 2026-10-21, before P4 results exist: exp_4 stored only per-configuration aggregates
  (n_anomalies, validity_rate, mean proximity, mean sparsity), not per-window results,
  and P4 uses the time-of-detection protocol. The pre-specified check "regenerated
  validity matches exp_4" is therefore reduced to: same number of sampled windows
  per configuration. Validity under the two protocols is reported side by side.
* 2026-10-21, exploratory, added after the P6 results were seen: (a) P6 reports, as a
  reference for the factor AUCs, the AUC of the conformal margin -|score - threshold|
  for the same wrong ACI decisions (+inf thresholds counted as most certain). Not part
  of the pre-specified P6 and not in its Holm family. (b) Computing (a) showed that
  ACI's threshold is +inf for a large share of the test stream on some datasets (on
  the P6 sample at alpha = 0.05: 69% of GECCO and 36% of Exathlon windows). Because
  ACI's update is label-free, every flagged window counts as an error and a long
  flagged stretch drives alpha_t below 1/(m+1); no window can be flagged until alpha_t
  recovers at gamma * alpha per step. posthoc_aci_trace.py measures this on every
  run (share of steps with +inf threshold, share of anomalies arriving then,
  coverage and recall over finite-threshold steps). Reported in full as an
  exploratory diagnostic.
* 2026-10-21, P4 compute only: the 352 P4 tasks that had not started on GPUQ were
  cancelled and resubmitted unchanged on CPUQ (RUNLOG), after a timing test showed
  GA-CE at 3.4 s per window on 8 CPU cores vs 5-7 s on the GPU nodes for these small
  detectors. threew tasks 0-9 ran on GPUQ. Each P4 run scores, detects, takes its
  snapshots and explains on one device, so every run is self-contained; the device
  of each run follows from its job id (RUNLOG) and is listed by posthoc_analyze.py.
* 2026-10-21, finding while preparing the P1 write-up (exploratory, all 720 P1 runs):
  under ACI the TOTAL alarm rate (flagged windows / all test windows, anomalies
  included) equals alpha within +-0.018 in every run (98% within +-0.01). This is the
  long-run guarantee of Gibbs and Candes applied to err_t = "flagged". It means ACI's
  coverage near 1 - alpha follows by construction when few windows are true anomalies,
  and is not evidence of a well-calibrated detector. The paper states this and uses
  recall at matched false-alarm rate (P2) as the informative comparison.
  Consequence, also checked on all 720 runs: with the alarm rate equal to alpha,
  ACI's coverage gap equals pi * (recall - alpha) / (1 - pi) for anomaly share pi;
  the identity holds with residual median 0.002, max 0.031 per run (correlation
  0.96), and explains ACI's over-coverage on the datasets with many anomalies.
* 2026-10-21, wording correction (no change of analysis): the P6 text above calls the
  pre-specified feature scalar the "max feature out-of-range rate". What
  posthoc_uncertainty.py computes, as specified in its code, is the maximum over
  features of the framework's per-feature score without the detector's errors, i.e.
  max_j (0.3 * share outside the widened training range + 0.3 * share beyond 2 sd).
  The paper describes it in these terms. While checking this, the methodology section
  of the paper was found to describe the four factors differently from the code
  (latent-space Mahalanobis and k-NN, score variance over neighbouring windows, plain
  per-feature MSE, four-term sum); the code computes input-space k-NN distance, LOF,
  first-difference variance, the mixed per-feature score, and the mean of three
  binned scores. The paper text was changed to describe the code, which produced
  every reported number; templates_v2 keeps the earlier text.
* 2026-10-22, descriptive additions (no tests), at the author's request for figures on
  uncertainty and counterfactuals: (a) error rate of ACI's decisions by tenth of each
  detector-free uncertainty factor (posthoc_analyze part_p6, p6_decile_error.csv), the
  same data as the P6 AUCs; (b) a case study (case_study.py, job in RUNLOG): the outputs
  of the deployed pipeline along the SWaT test stream (LSTM-AE, seed 42, P1 scores
  replayed, decisions identical to P1) and two SKAB counterfactual examples taken from
  the P4 outputs. Selection rules were written into case_study.py / part_case before the
  windows were looked at: first attack window flagged by ACI; for SKAB, the median-
  proximity GA-CE+PerCE-valid true anomaly (LSTM-AE) and the median-plausibility valid
  DICE window (LSTM-VAE). The case study found that the confidence value is close to 0.5
  for 93% of SWaT decisions because a few extreme calibration scores inflate the
  standard deviation it divides by; reported in the paper as an observation.
