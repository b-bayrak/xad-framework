#!/bin/bash
# Main runs on one of the ten added datasets: 3 detectors x 5 seeds = 15 GPU
# array tasks, with the same hyperparameters and experiments as the SKAB and
# SWaT runs (jobs/swat_main.sbatch). The datasets are registered at
# runtime through the xad_extra hook (XAD_EXTRA=1 and PYTHONPATH); xad/ and
# experiments/ are used unchanged. Writes to runs/benchmark/{results,cache,logs}.
#
# Submitted by jobs/benchmark_submit.sh, which sets DATASET, walltime and
# memory and records the job id in docs/RUNLOG.md.
#
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --array=0-14

set -euo pipefail
: "${DATASET:?DATASET must be exported by benchmark_submit.sh}"
: "${MODE:=full}"   # full | smoke

REPO=${XAD_REPO:-${SLURM_SUBMIT_DIR:-$(pwd)}}
BASE=$REPO/runs/benchmark
cd "$REPO"
mkdir -p "$BASE/logs" "$BASE/cache" "$BASE/results"

module purge
module load Anaconda3/2022.10
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate xad
export XAD_EXTRA=1
export PYTHONPATH=$REPO/datasets/register${PYTHONPATH:+:$PYTHONPATH}

DETECTORS=(conv_ae lstm_ae lstm_vae)
SEEDS=(42 43 44 45 46)
DETECTOR=${DETECTORS[$((SLURM_ARRAY_TASK_ID % 3))]}
SEED=${SEEDS[$((SLURM_ARRAY_TASK_ID / 3))]}
DATA_PATH=$REPO/data/prepared/$DATASET
# Guard: prepared data must match the frozen checksum, so a cached detector
# can never be paired with re-prepared data.
WANT=$(awk -v d="$DATASET" '$NF==d {print $1}' "$REPO/datasets/checksums.txt" | tail -1)
HAVE=$(md5sum "$DATA_PATH/data.npz" | cut -c1-32)
[ -n "$WANT" ] && [ "$WANT" = "$HAVE" ] || { echo "DATA CHECKSUM MISMATCH for $DATASET: frozen=$WANT now=$HAVE"; exit 4; }
echo "data md5 $HAVE (matches datasets/checksums.txt)"

if [ "$MODE" = smoke ]; then
    # Plumbing check only: tiny budgets, separate dirs, numbers meaningless.
    RUN_ID="smoke_${SLURM_ARRAY_JOB_ID}"; RES=$BASE/smoke/results; CACHE=$BASE/smoke/cache
    EPOCHS=2; PATIENCE=2; NGEN=3; POP=8; MAXA=5
else
    RUN_ID="5seed_${SLURM_ARRAY_JOB_ID}"; RES=$BASE/results; CACHE=$BASE/cache
    EPOCHS=200; PATIENCE=20; NGEN=100; POP=50; MAXA=100
fi
mkdir -p "$RES" "$CACHE"
OUTDIR="$RES/${DATASET}_${RUN_ID}"

echo "=== $(date -u +%FT%TZ) host=$(hostname) job=${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "dataset=$DATASET detector=$DETECTOR seed=$SEED mode=$MODE"
echo "data=$DATA_PATH out=$OUTDIR cache=$CACHE"
nvidia-smi --query-gpu=name,memory.total --format=csv

python3 experiments/run_all_experiments.py \
    --dataset "$DATASET" --data_path "$DATA_PATH" \
    --output_dir "$RES/" --cache_dir "$CACHE/" \
    --run_id "${RUN_ID}" \
    --detector "${DETECTOR}" --seed "${SEED}" \
    --hidden_dim 128 --latent_dim 32 --num_layers 2 --dropout 0.2 \
    --epochs $EPOCHS --patience $PATIENCE --batch_size 32 \
    --alpha 0.05 \
    --n_generations $NGEN --population_size $POP \
    --max_anomalies $MAXA \
    --experiments 1 2 3 4 6

# Second counterfactual pass under ACI.
python3 experiments/exp_4_counterfactual_quality.py \
    --dataset "$DATASET" --data_path "$DATA_PATH" \
    --output_dir "$OUTDIR" --cache_dir "$CACHE/" \
    --detector "${DETECTOR}" --seed "${SEED}" \
    --hidden_dim 128 --latent_dim 32 --num_layers 2 --dropout 0.2 \
    --epochs $EPOCHS --patience $PATIENCE --batch_size 32 \
    --alpha 0.05 --max_anomalies $MAXA \
    --n_generations $NGEN --population_size $POP \
    --cp_mode aci

echo "=== DONE $(date -u +%FT%TZ) task $SLURM_ARRAY_TASK_ID (dataset=$DATASET detector=$DETECTOR seed=$SEED)"
