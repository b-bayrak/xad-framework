#!/bin/bash
# Classical baselines (experiment 8: Isolation Forest, LOF, One-Class SVM) on
# SKAB and SWaT, with the same splits and windows as the autoencoders. CPU only.
# Writes to runs/skab_swat/results/classical_baselines/.
#
# Submit from the repository root:
#   mkdir -p runs/skab_swat/logs
#   sbatch jobs/skab_swat_classical.sbatch
#
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --job-name=xad_skab_swat_classical
#SBATCH --partition=CPUQ
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=runs/skab_swat/logs/classical_%j.out
#SBATCH --error=runs/skab_swat/logs/classical_%j.err

set -euo pipefail
REPO=${XAD_REPO:-${SLURM_SUBMIT_DIR:-$(pwd)}}
cd "$REPO"
module purge
module load Anaconda3/2022.10
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate xad

OUT=runs/skab_swat/results/classical_baselines
python3 experiments/exp_8_classical_baselines.py --dataset skab --data_path ./data/SKAB/ --output_dir "$OUT"
python3 experiments/exp_8_classical_baselines.py --dataset swat --data_path ./data/SWaT/ --output_dir "$OUT"
