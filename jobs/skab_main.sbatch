#!/bin/bash
# SKAB main runs: 3 detectors x 5 seeds (42-46) = 15 GPU array tasks.
# Runs experiments 1-6, plus experiment 4 under ACI, and writes to runs/skab_swat/{results,cache,logs}.
#
# Submit from the repository root:
#   mkdir -p runs/skab_swat/logs
#   sbatch jobs/skab_main.sbatch
#
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --job-name=xad_skab_5seed
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --array=0-14
#SBATCH --output=runs/skab_swat/logs/skab_%A_%a.out
#SBATCH --error=runs/skab_swat/logs/skab_%A_%a.err

set -euo pipefail

REPO=${XAD_REPO:-${SLURM_SUBMIT_DIR:-$(pwd)}}
BASE=$REPO/runs/skab_swat
cd "$REPO"                       # so ./data/ and experiments/ resolve
mkdir -p "$BASE/logs" "$BASE/cache" "$BASE/results"

module purge
module load Anaconda3/2022.10
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate xad

DETECTORS=(conv_ae lstm_ae lstm_vae)
SEEDS=(42 43 44 45 46)
DETECTOR=${DETECTORS[$((SLURM_ARRAY_TASK_ID % 3))]}
SEED=${SEEDS[$((SLURM_ARRAY_TASK_ID / 3))]}
RUN_ID="5seed_${SLURM_ARRAY_JOB_ID}"
OUTDIR="$BASE/results/skab_${RUN_ID}"

echo "Task $SLURM_ARRAY_TASK_ID / detector=$DETECTOR / seed=$SEED / dataset=skab"
echo "Output dir: $OUTDIR   Cache: $BASE/cache"
nvidia-smi --query-gpu=name,memory.total --format=csv

python3 experiments/run_all_experiments.py \
    --dataset skab --data_path ./data/SKAB/ \
    --output_dir "$BASE/results/" --cache_dir "$BASE/cache/" \
    --run_id "${RUN_ID}" \
    --detector "${DETECTOR}" --seed "${SEED}" \
    --hidden_dim 128 --latent_dim 32 --num_layers 2 --dropout 0.2 \
    --epochs 200 --patience 20 --batch_size 32 \
    --alpha 0.05 \
    --n_generations 100 --population_size 50 \
    --max_anomalies 100 --with_smoothness \
    --experiments 1 2 3 4 5 6

# Second counterfactual pass under ACI.
python3 experiments/exp_4_counterfactual_quality.py \
    --dataset skab --data_path ./data/SKAB/ \
    --output_dir "$OUTDIR" --cache_dir "$BASE/cache/" \
    --detector "${DETECTOR}" --seed "${SEED}" \
    --hidden_dim 128 --latent_dim 32 --num_layers 2 --dropout 0.2 \
    --epochs 200 --patience 20 --batch_size 32 \
    --alpha 0.05 --max_anomalies 100 \
    --n_generations 100 --population_size 50 \
    --cp_mode aci

echo "DONE task $SLURM_ARRAY_TASK_ID (detector=$DETECTOR seed=$SEED)"
