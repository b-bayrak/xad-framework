#!/bin/bash
# exp_8 classical baselines (IsolationForest x3 seeds, LOF, OC-SVM) on ONE
# extra dataset, same splits/windows as the deep detectors. CPU only.
# Unchanged experiments/exp_8_classical_baselines.py (its LOF/OC-SVM 5,000
# training-window subsample is part of that script and applies here too).
#
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=CPUQ
#SBATCH --cpus-per-task=16

set -euo pipefail
: "${DATASET:?}"
REPO=${XAD_REPO:-${SLURM_SUBMIT_DIR:-$(pwd)}}
BASE=$REPO/runs/benchmark
cd "$REPO"
module purge
module load Anaconda3/2022.10
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate xad
export XAD_EXTRA=1
export PYTHONPATH=$REPO/datasets/register${PYTHONPATH:+:$PYTHONPATH}
echo "=== $(date -u +%FT%TZ) host=$(hostname) job=$SLURM_JOB_ID dataset=$DATASET"
python3 experiments/exp_8_classical_baselines.py --dataset "$DATASET" \
    --data_path "$REPO/data/prepared/$DATASET" --output_dir "$BASE/results/classical_baselines"
echo "=== DONE $(date -u +%FT%TZ)"
