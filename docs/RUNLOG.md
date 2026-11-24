# Run log

Every SLURM submission for the ten added datasets and for the post-hoc analyses, appended automatically by `jobs/benchmark_submit.sh` and `jobs/posthoc_submit.sh` (written by the scripts under their earlier names). Rows with mode `smoke` are short plumbing checks whose outputs are not used. The analysis reads the job ids of the last `full` row per dataset.

| submitted (UTC) | mode | dataset | job id | walltime / mem | code checksums |
|---|---|---|---|---|---|
| 2026-10-14T09:00:10Z | smoke | daphnet | 25693991 | 01:00:00 / 32G | paper-code md5 3e94282ad8e3, extra-code md5 599a426731f2 |
| 2026-10-14T09:00:10Z | smoke | exathlon | 25693992 | 01:00:00 / 32G | paper-code md5 3e94282ad8e3, extra-code md5 599a426731f2 |
| 2026-10-14T09:00:10Z | smoke | gecco | 25693993 | 01:00:00 / 32G | paper-code md5 3e94282ad8e3, extra-code md5 599a426731f2 |
| 2026-10-14T09:00:10Z | smoke | ghl | 25693994 | 01:00:00 / 32G | paper-code md5 3e94282ad8e3, extra-code md5 599a426731f2 |
| 2026-10-14T09:00:10Z | smoke | metropt3 | 25693995 | 01:00:00 / 32G | paper-code md5 3e94282ad8e3, extra-code md5 599a426731f2 |
| 2026-10-14T09:00:10Z | smoke | psm | 25693996 | 01:00:00 / 32G | paper-code md5 3e94282ad8e3, extra-code md5 599a426731f2 |
| 2026-10-14T09:00:10Z | smoke | smap | 25693997 | 01:00:00 / 32G | paper-code md5 3e94282ad8e3, extra-code md5 599a426731f2 |
| 2026-10-14T09:00:10Z | smoke | smd | 25693998 | 01:00:00 / 32G | paper-code md5 3e94282ad8e3, extra-code md5 599a426731f2 |
| 2026-10-14T09:00:10Z | smoke | threew | 25693999 | 01:00:00 / 32G | paper-code md5 3e94282ad8e3, extra-code md5 599a426731f2 |
| 2026-10-14T09:07:40Z | smoke (supp only) | exathlon | supp 25694026 (vs smoke main 25693992) | 01:00:00 / 32G | manual sbatch |
| 2026-10-14T09:16:42Z | smoke | threew | main 25694093, supp 25694094, classical - | 01:00:00 / 32G | paper-code md5 3e94282ad8e3, extra-code md5 163b72b683e2 |
| 2026-10-14T09:20:38Z | smoke | msl | main 25694110, supp 25694111, classical - | 01:00:00 / 32G | paper-code md5 3e94282ad8e3, extra-code md5 163b72b683e2 |
| 2026-10-14T09:39:12Z | full | daphnet | main 25694191, supp 25694192, classical 25694193 | 24:00:00 / 64G | paper-code md5 3e94282ad8e3, extra-code md5 163b72b683e2 |
| 2026-10-14T09:39:12Z | full | exathlon | main 25694194, supp 25694195, classical 25694196 | 12:00:00 / 32G | paper-code md5 3e94282ad8e3, extra-code md5 163b72b683e2 |
| 2026-10-14T09:39:12Z | full | gecco | main 25694197, supp 25694198, classical 25694199 | 12:00:00 / 32G | paper-code md5 3e94282ad8e3, extra-code md5 163b72b683e2 |
| 2026-10-14T09:39:12Z | full | ghl | main 25694200, supp 25694201, classical 25694202 | 24:00:00 / 64G | paper-code md5 3e94282ad8e3, extra-code md5 163b72b683e2 |
| 2026-10-14T09:39:12Z | full | metropt3 | main 25694203, supp 25694204, classical 25694205 | 24:00:00 / 64G | paper-code md5 3e94282ad8e3, extra-code md5 163b72b683e2 |
| 2026-10-14T09:39:12Z | full | msl | main 25694206, supp 25694207, classical 25694208 | 12:00:00 / 32G | paper-code md5 3e94282ad8e3, extra-code md5 163b72b683e2 |
| 2026-10-14T09:39:12Z | full | psm | main 25694209, supp 25694210, classical 25694211 | 12:00:00 / 32G | paper-code md5 3e94282ad8e3, extra-code md5 163b72b683e2 |
| 2026-10-14T09:39:12Z | full | smap | main 25694212, supp 25694213, classical 25694214 | 24:00:00 / 64G | paper-code md5 3e94282ad8e3, extra-code md5 163b72b683e2 |
| 2026-10-14T09:39:12Z | full | smd | main 25694215, supp 25694216, classical 25694217 | 24:00:00 / 64G | paper-code md5 3e94282ad8e3, extra-code md5 163b72b683e2 |
| 2026-10-14T09:39:12Z | full | threew | main 25694218, supp 25694219, classical 25694220 | 24:00:00 / 64G | paper-code md5 3e94282ad8e3, extra-code md5 163b72b683e2 |
| 2026-10-21T11:25:47Z | reference-supp | skab | supp 25752063 (reuses FIVE_SEED_RUN main 25339219 cache) | 48:00:00 / 64G | slurm/five_seed/supp_5seed_array.sbatch |
| 2026-10-21T11:25:47Z | reference-supp | swat | supp 25752064 (reuses FIVE_SEED_RUN main 25339220 cache) | 48:00:00 / 64G | slurm/five_seed/supp_5seed_array.sbatch |
| 2026-10-21T15:11:23Z | posthoc | P1-P3 policies (12 ds x 3 det, 5 seeds each) | 25753704 | GPUQ 10h | 16d0654c5e0b |
| 2026-10-21T15:11:23Z | posthoc | P5 Isolation Forest backbone (12 ds x 5 seeds) | 25753705 | CPUQ 72h | 16d0654c5e0b |
| 2026-10-21T16:28:28Z | posthoc | P6 uncertainty (12 ds) | 25754184 | CPUQ 12h, after 25753704 | 0ddf6a7dc888 |
| 2026-10-21T16:28:28Z | posthoc | P4 counterfactual regeneration threew (3 det x 5 seeds x 2 policies) | 25754185 | GPUQ 36h | 0ddf6a7dc888 |
| 2026-10-21T16:28:28Z | posthoc | P4 counterfactual regeneration daphnet (3 det x 5 seeds x 2 policies) | 25754186 | GPUQ 36h | 0ddf6a7dc888 |
| 2026-10-21T16:28:29Z | posthoc | P4 counterfactual regeneration exathlon (3 det x 5 seeds x 2 policies) | 25754187 | GPUQ 36h | 0ddf6a7dc888 |
| 2026-10-21T16:28:29Z | posthoc | P4 counterfactual regeneration gecco (3 det x 5 seeds x 2 policies) | 25754188 | GPUQ 36h | 0ddf6a7dc888 |
| 2026-10-21T16:28:29Z | posthoc | P4 counterfactual regeneration ghl (3 det x 5 seeds x 2 policies) | 25754189 | GPUQ 36h | 0ddf6a7dc888 |
| 2026-10-21T16:28:29Z | posthoc | P4 counterfactual regeneration metropt3 (3 det x 5 seeds x 2 policies) | 25754190 | GPUQ 36h | 0ddf6a7dc888 |
| 2026-10-21T16:28:29Z | posthoc | P4 counterfactual regeneration msl (3 det x 5 seeds x 2 policies) | 25754191 | GPUQ 36h | 0ddf6a7dc888 |
| 2026-10-21T16:28:29Z | posthoc | P4 counterfactual regeneration psm (3 det x 5 seeds x 2 policies) | 25754192 | GPUQ 36h | 0ddf6a7dc888 |
| 2026-10-21T16:28:29Z | posthoc | P4 counterfactual regeneration skab (3 det x 5 seeds x 2 policies) | 25754193 | GPUQ 36h | 0ddf6a7dc888 |
| 2026-10-21T16:28:29Z | posthoc | P4 counterfactual regeneration smap (3 det x 5 seeds x 2 policies) | 25754194 | GPUQ 36h | 0ddf6a7dc888 |
| 2026-10-21T16:28:29Z | posthoc | P4 counterfactual regeneration smd (3 det x 5 seeds x 2 policies) | 25754195 | GPUQ 36h | 0ddf6a7dc888 |
| 2026-10-21T16:28:29Z | posthoc | P4 counterfactual regeneration swat (3 det x 5 seeds x 2 policies) | 25754196 | GPUQ 36h | 0ddf6a7dc888 |
| 2026-10-21T16:42:03Z | posthoc | ACI infinite-threshold trace (exploratory, 12 ds, CPU replay of P1 scores) | 25754291 | CPUQ 2h | 6462b2e282ea |
| 2026-10-21T16:49:56Z | posthoc | P4 counterfactual regeneration threew on CPUQ, tasks 10-29 (pending GPUQ tasks cancelled) | 25754347 | CPUQ 24h | 6e7e208202ad |
| 2026-10-21T16:49:56Z | posthoc | P4 counterfactual regeneration daphnet on CPUQ, tasks 0-29 (pending GPUQ tasks cancelled) | 25754348 | CPUQ 24h | 6e7e208202ad |
| 2026-10-21T16:49:56Z | posthoc | P4 counterfactual regeneration exathlon on CPUQ, tasks 0-29 (pending GPUQ tasks cancelled) | 25754349 | CPUQ 24h | 6e7e208202ad |
| 2026-10-21T16:49:56Z | posthoc | P4 counterfactual regeneration gecco on CPUQ, tasks 0-29 (pending GPUQ tasks cancelled) | 25754350 | CPUQ 24h | 6e7e208202ad |
| 2026-10-21T16:49:56Z | posthoc | P4 counterfactual regeneration ghl on CPUQ, tasks 0-29 (pending GPUQ tasks cancelled) | 25754351 | CPUQ 24h | 6e7e208202ad |
| 2026-10-21T16:49:56Z | posthoc | P4 counterfactual regeneration metropt3 on CPUQ, tasks 0-29 (pending GPUQ tasks cancelled) | 25754352 | CPUQ 24h | 6e7e208202ad |
| 2026-10-21T16:49:56Z | posthoc | P4 counterfactual regeneration msl on CPUQ, tasks 0-29 (pending GPUQ tasks cancelled) | 25754353 | CPUQ 24h | 6e7e208202ad |
| 2026-10-21T16:49:56Z | posthoc | P4 counterfactual regeneration psm on CPUQ, tasks 0-29 (pending GPUQ tasks cancelled) | 25754354 | CPUQ 24h | 6e7e208202ad |
| 2026-10-21T16:49:56Z | posthoc | P4 counterfactual regeneration skab on CPUQ, tasks 0-29 (pending GPUQ tasks cancelled) | 25754355 | CPUQ 24h | 6e7e208202ad |
| 2026-10-21T16:49:56Z | posthoc | P4 counterfactual regeneration smap on CPUQ, tasks 0-29 (pending GPUQ tasks cancelled) | 25754356 | CPUQ 24h | 6e7e208202ad |
| 2026-10-21T16:49:56Z | posthoc | P4 counterfactual regeneration smd on CPUQ, tasks 0-29 (pending GPUQ tasks cancelled) | 25754357 | CPUQ 24h | 6e7e208202ad |
| 2026-10-21T16:49:56Z | posthoc | P4 counterfactual regeneration swat on CPUQ, tasks 0-29 (pending GPUQ tasks cancelled) | 25754358 | CPUQ 24h | 6e7e208202ad |
| 2026-10-21T20:19:17Z | rerun | swat | exp_5 under ACI for task 14 (lstm_vae seed 46) after NODE_FAIL of 25752064_14 -> 25755783 | 48:00:00 / 64G | slurm/five_seed/supp_5seed_rerun_exp5aci.sbatch |
| 2026-10-22T12:31:15Z | posthoc | Case-study illustrations (descriptive): SWaT LSTM-AE s42 uncertainty stream, SKAB CF examples | 25767587 | CPUQ 4h | 2126c8015793 |
