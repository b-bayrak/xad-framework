#!/bin/bash
# Supplementary runs for SKAB and SWaT, same recipe as
# jobs/benchmark_supp.sbatch: experiment 3 under ACI, experiment 7
# (sensitivity) and experiment 5 (GA-CE ablation, sliding window and ACI).
# Reuses the detectors cached by the main run (no retraining; the job stops if
# a checkpoint is missing) and writes to runs/skab_swat/results/<ds>_supp_<main job>/.
#
# Submit from the repository root, with the job id of the main run:
#   sbatch --export=ALL,DATASET=skab,MAIN_JOB=<job id> jobs/skab_swat_supp.sbatch
#   sbatch --export=ALL,DATASET=swat,MAIN_JOB=<job id> jobs/skab_swat_supp.sbatch
#
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --array=0-14
#SBATCH --output=runs/skab_swat/logs/supp_%x_%A_%a.out
#SBATCH --error=runs/skab_swat/logs/supp_%x_%A_%a.err

set -euo pipefail
: "${DATASET:?}"; : "${MAIN_JOB:?}"
REPO=${XAD_REPO:-${SLURM_SUBMIT_DIR:-$(pwd)}}
BASE=$REPO/runs/skab_swat
cd "$REPO"
module purge
module load Anaconda3/2022.10
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate xad

case $DATASET in skab) DP=./data/SKAB/ ;; swat) DP=./data/SWaT/ ;; *) echo "skab|swat only"; exit 2 ;; esac
DETECTORS=(conv_ae lstm_ae lstm_vae)
SEEDS=(42 43 44 45 46)
DETECTOR=${DETECTORS[$((SLURM_ARRAY_TASK_ID % 3))]}
SEED=${SEEDS[$((SLURM_ARRAY_TASK_ID / 3))]}
OUT=$BASE/results/${DATASET}_supp_${MAIN_JOB}
mkdir -p "$OUT"
CKPT=$BASE/cache/${DATASET}_${DETECTOR}_seed${SEED}.pt
[ -f "$CKPT" ] || { echo "MISSING cached detector $CKPT"; exit 3; }
echo "=== $(date -u +%FT%TZ) host=$(hostname) job=${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} dataset=$DATASET detector=$DETECTOR seed=$SEED"
md5sum "$CKPT"

COMMON=(--dataset "$DATASET" --data_path "$DP" --output_dir "$OUT" --cache_dir "$BASE/cache/"
        --detector "$DETECTOR" --seed "$SEED"
        --hidden_dim 128 --latent_dim 32 --num_layers 2 --dropout 0.2
        --epochs 200 --patience 20 --batch_size 32)
echo "== exp_3 under ACI"
python3 experiments/exp_3_uncertainty_effectiveness.py "${COMMON[@]}" --alpha 0.05 --cp_mode aci
echo "== exp_7 policy sensitivity"
python3 experiments/exp_7_policy_sensitivity.py "${COMMON[@]}"
echo "== exp_5 GA-CE ablation (default policy = sliding)"
python3 experiments/exp_5_gace_ablation.py "${COMMON[@]}" --alpha 0.05 \
    --max_anomalies 100 --n_generations 100 --population_size 50 --with_smoothness
echo "== exp_5 GA-CE ablation under ACI"
python3 experiments/exp_5_gace_ablation.py "${COMMON[@]}" --alpha 0.05 --cp_mode aci \
    --max_anomalies 100 --n_generations 100 --population_size 50 --with_smoothness
echo "=== DONE $(date -u +%FT%TZ) task $SLURM_ARRAY_TASK_ID"
