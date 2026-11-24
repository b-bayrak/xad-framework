# Reproducing the paper

All commands run from the repository root. The jobs are written for a SLURM cluster with GPU nodes. Before submitting, set `#SBATCH --account`, the partition names (`GPUQ`, `CPUQ`) and the `module load` line in [`jobs/`](../jobs) to match your cluster, and create a conda environment called `xad` with the packages of `requirements.txt`.

Every run uses seeds 42 to 46 and the hyperparameters in the job files. Nothing is tuned on the test data.

## 1. Environment

```bash
conda create -n xad python=3.10
conda activate xad
pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install -e .
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

## 2. Data

```bash
git clone https://github.com/waico/SKAB.git data/SKAB      # SKAB
# SWaT: request access from iTrust and place the files in data/SWaT/
bash datasets/download.sh                                   # ten added datasets -> data/raw/
python datasets/prepare.py all                              # -> data/prepared/
XAD_EXTRA=1 PYTHONPATH=datasets/register python datasets/verify.py
```

`verify.py` loads every prepared dataset through the experiment pipeline and writes the window counts to `datasets/logs/verify_windows.tsv`. Compare `md5sum data/prepared/*/data.npz` with `datasets/checksums.txt`. The jobs refuse to run on data that does not match.

## 3. Runs on SKAB and SWaT

```bash
mkdir -p runs/skab_swat/logs
sbatch jobs/skab_main.sbatch          # 15 GPU tasks: 3 detectors x 5 seeds
sbatch jobs/swat_main.sbatch          # 15 GPU tasks
sbatch jobs/skab_swat_classical.sbatch
```

When the main jobs have finished, run the supplementary experiments with their job ids:

```bash
sbatch --export=ALL,DATASET=skab,MAIN_JOB=<skab job id> jobs/skab_swat_supp.sbatch
sbatch --export=ALL,DATASET=swat,MAIN_JOB=<swat job id> jobs/skab_swat_supp.sbatch
```

| Job | Experiments | Output |
|---|---|---|
| `skab_main`, `swat_main` | 1 detection, 2 calibration, 3 confidence, 4 counterfactuals (sliding window and ACI), 6 threshold policies, and 5 GA-CE ablation on SKAB | `runs/skab_swat/results/<ds>_5seed_<job>/` |
| `skab_swat_supp` | 3 under ACI, 7 sensitivity, 5 GA-CE ablation under both policies | `runs/skab_swat/results/<ds>_supp_<job>/` |
| `skab_swat_classical` | 8 classical baselines | `runs/skab_swat/results/classical_baselines/` |

Trained detectors are cached in `runs/skab_swat/cache/` and reused by every later step.

## 4. Runs on the ten added datasets

```bash
bash jobs/benchmark_submit.sh full all
```

For each dataset this submits the main array, the supplementary array (which waits for the main array) and the classical baselines, with the same experiments and hyperparameters as for SKAB and SWaT. Every submission is appended to `docs/RUNLOG.md`. The analysis reads the job ids from there. Outputs go to `runs/benchmark/`.

`bash jobs/benchmark_submit.sh smoke gecco` runs a short check with tiny budgets, written to a separate folder.

## 5. Post-hoc analyses

```bash
bash jobs/posthoc_submit.sh
bash jobs/posthoc_submit.sh acitrace casestudy
```

| Part | Script | Question |
|---|---|---|
| P1 to P3 | `analysis/posthoc_policies.py` | Uncensored sliding window, recall at a matched false-alarm rate, threshold positions |
| P4 | `analysis/posthoc_cf.py` | Counterfactuals checked against the threshold that raised the alarm, and their plausibility |
| P5 | `analysis/posthoc_iforest.py` | Isolation Forest as the detector inside the same pipeline |
| P6 | `analysis/posthoc_uncertainty.py` | Whether the uncertainty factors point to wrong decisions |
| Trace | `analysis/posthoc_aci_trace.py` | Time ACI spends with an infinite threshold |
| Figures | `analysis/case_study.py` | Inputs of Figures 5 and 8 |

These jobs reuse the cached detectors and never train. P1 must finish before P6. Outputs go to `runs/posthoc/`.

## 6. Analysis, tables and figures

```bash
bash analysis/run.sh analyze.py          # main tests            -> results/analysis/
bash analysis/run.sh posthoc_analyze.py  # P1 to P6 summaries    -> results/posthoc/
bash analysis/run.sh paper_outputs.py    # numbers, tables, figures -> results/paper/
```

`analysis/run.sh` activates the `xad` environment and registers the prepared datasets. `analyze.py` and `paper_outputs.py` stop if a run is missing a seed. `paper_outputs.py` also checks every qualitative statement of the paper against the data and stops if one no longer holds.

The statistical unit is the dataset: seeds are averaged first, then a two-sided Wilcoxon signed-rank test runs over the twelve datasets, with Holm correction within each question.

## Compute

The main runs need one GPU per task. The time limits in the job files are those used for the paper: 12 hours per SKAB task, 48 hours per SWaT task, and 12 or 24 hours per task on the added datasets, with 24 hours and 64 GB for the largest ones (GHL, MetroPT-3, 3W, Daphnet, SMD, SMAP). The analysis steps in section 6 run on a CPU in a few minutes.

## Records

`ANALYSIS_PLAN.md`, `AUDIT.md` and `RUNLOG.md` are kept as they were written during the runs. They refer to two folders by their earlier names: `EXTRA_DATASETS/` is now split into `datasets/`, `analysis/`, `jobs/` and `runs/benchmark/`, and `FIVE_SEED_RUN/` is now `runs/skab_swat/`.
