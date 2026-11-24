#!/bin/bash
# Supplementary runs on one of the ten added datasets, after the main array and
# with its cached detectors (no retraining):
#   experiment 3 under ACI, experiment 7 (sensitivity to the ACI step size and
#   the window size) and experiment 5 (GA-CE ablation, sliding window and ACI).
# Task i depends on main task i (benchmark_submit.sh uses --dependency=aftercorr), so
# detector and seed match. Writes to runs/benchmark/results/<ds>_supp_<main job>/.
#
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --array=0-14

set -euo pipefail
: "${DATASET:?}"; : "${MAIN_JOB:?}"; : "${MODE:=full}"

REPO=${XAD_REPO:-${SLURM_SUBMIT_DIR:-$(pwd)}}
BASE=$REPO/runs/benchmark
cd "$REPO"

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
    RES=$BASE/smoke/results; CACHE=$BASE/smoke/cache; EPOCHS=2; PATIENCE=2; NGEN=3; POP=8; MAXA=5
else
    RES=$BASE/results; CACHE=$BASE/cache; EPOCHS=200; PATIENCE=20; NGEN=100; POP=50; MAXA=100
fi
OUT=$RES/${DATASET}_supp_${MAIN_JOB}
mkdir -p "$OUT"
COMMON=(--dataset "$DATASET" --data_path "$DATA_PATH" --output_dir "$OUT" --cache_dir "$CACHE/"
        --detector "$DETECTOR" --seed "$SEED"
        --hidden_dim 128 --latent_dim 32 --num_layers 2 --dropout 0.2
        --epochs $EPOCHS --patience $PATIENCE --batch_size 32)

echo "=== $(date -u +%FT%TZ) host=$(hostname) job=${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} main=$MAIN_JOB"
echo "dataset=$DATASET detector=$DETECTOR seed=$SEED mode=$MODE out=$OUT"
# Guard: the cached detector from the main array must exist, so these
# numbers describe the same model as the main tables.
CKPT=$CACHE/${DATASET}_${DETECTOR}_seed${SEED}.pt
[ -f "$CKPT" ] || { echo "MISSING cached detector $CKPT (main task failed?)"; exit 3; }
md5sum "$CKPT"

echo "== exp_3 under ACI"
python3 experiments/exp_3_uncertainty_effectiveness.py "${COMMON[@]}" --alpha 0.05 --cp_mode aci
echo "== exp_7 policy sensitivity"
python3 experiments/exp_7_policy_sensitivity.py "${COMMON[@]}"
echo "== exp_5 GA-CE ablation (default policy = sliding, as SKAB array)"
python3 experiments/exp_5_gace_ablation.py "${COMMON[@]}" --alpha 0.05 \
    --max_anomalies $MAXA --n_generations $NGEN --population_size $POP --with_smoothness
echo "== exp_5 GA-CE ablation under ACI"
python3 experiments/exp_5_gace_ablation.py "${COMMON[@]}" --alpha 0.05 --cp_mode aci \
    --max_anomalies $MAXA --n_generations $NGEN --population_size $POP --with_smoothness

echo "=== DONE $(date -u +%FT%TZ) task $SLURM_ARRAY_TASK_ID (dataset=$DATASET detector=$DETECTOR seed=$SEED)"
