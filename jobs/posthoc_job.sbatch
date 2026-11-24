#!/bin/bash
# Runner for the post-hoc analyses P1-P6 (docs/ANALYSIS_PLAN.md). Submitted by
# jobs/posthoc_submit.sh, which sets KIND and the array, partition and time per
# analysis. Reuses the cached detectors of the main runs and never trains.
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --cpus-per-task=8
set -euo pipefail
: "${KIND:?}"
REPO=${XAD_REPO:-${SLURM_SUBMIT_DIR:-$(pwd)}}
cd "$REPO/analysis"
module purge
module load Anaconda3/2022.10
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate xad
export XAD_EXTRA=1 PYTHONPATH=$REPO/datasets/register:$REPO/analysis
DS=(threew daphnet exathlon gecco ghl metropt3 msl psm skab smap smd swat)
DET=(conv_ae lstm_ae lstm_vae)
SEEDS=(42 43 44 45 46)
i=$SLURM_ARRAY_TASK_ID
echo "=== $(date -u +%FT%TZ) host=$(hostname) kind=$KIND task=$i"
case $KIND in
  policies)     python posthoc_policies.py --dataset ${DS[$((i / 3))]} --detector ${DET[$((i % 3))]} ;;
  cf)           D=${DATASET:?}; POL=(sliding aci)
                python posthoc_cf.py --dataset $D --detector ${DET[$((i % 3))]} --seed ${SEEDS[$(((i / 3) % 5))]} --policy ${POL[$((i / 15))]} ;;
  iforest)      python posthoc_iforest.py --dataset ${DS[$((i / 5))]} --seed ${SEEDS[$((i % 5))]} ;;
  uncertainty)  python posthoc_uncertainty.py --dataset ${DS[$i]} ;;
  acitrace)     python posthoc_aci_trace.py --dataset ${DS[$i]} ;;
  casestudy)    python case_study.py ;;
  *) echo "unknown KIND"; exit 2 ;;
esac
echo "=== DONE $(date -u +%FT%TZ)"
