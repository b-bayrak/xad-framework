#!/bin/bash
# Submit the runs on the ten added datasets: for each dataset the main array,
# the supplementary array and the classical baselines. Every submission (UTC
# time, mode, dataset, job ids, code checksums) is appended to docs/RUNLOG.md.
#
#   bash jobs/benchmark_submit.sh full  all              # all prepared datasets
#   bash jobs/benchmark_submit.sh full  gecco psm        # a subset
#   bash jobs/benchmark_submit.sh smoke gecco            # short plumbing check

set -euo pipefail
REPO=${XAD_REPO:-$(pwd)}
RUN=$REPO/runs/benchmark
PREP=$REPO/data/prepared
mkdir -p "$RUN/logs"
MODE=$1; shift
[ "$MODE" = full ] || [ "$MODE" = smoke ] || { echo "mode must be full|smoke"; exit 1; }
if [ "${1:-}" = all ]; then
    set -- $(ls "$PREP")
fi

# Walltime/memory per dataset (big = long test streams or >1M rows).
res() {
    case $1 in
        ghl|metropt3|threew|daphnet|smd|smap) echo "24:00:00 64G" ;;
        *)                                    echo "12:00:00 32G" ;;
    esac
}

# Checksums of the code that produces the numbers, so a result folder can be
# tied to the exact code version.
CODE_SUM=$(cat "$REPO"/xad/*.py "$REPO"/experiments/*.py | md5sum | cut -c1-12)
EXTRA_SUM=$(cat "$REPO"/datasets/register/xad_extra/__init__.py "$REPO"/datasets/register/sitecustomize.py \
            "$REPO"/jobs/benchmark_*.sbatch | md5sum | cut -c1-12)

LOG=$REPO/docs/RUNLOG.md
for DS in "$@"; do
    [ -f "$PREP/$DS/meta.json" ] || { echo "not prepared: $DS"; continue; }
    read -r T M <<< "$(res "$DS")"
    if [ "$MODE" = smoke ]; then T=01:00:00; M=32G; ARR="--array=0-2"; ST=01:00:00; else ARR=""; ST=$T; fi
    # 1) main array: exp 1 2 3 4 6 + exp 4 under ACI (trains + caches detectors)
    JID=$(sbatch --parsable --job-name="xad_x_${DS}_${MODE}" --time="$T" --mem="$M" $ARR \
          --output="$RUN/logs/${DS}_${MODE}_%A_%a.out" \
          --error="$RUN/logs/${DS}_${MODE}_%A_%a.err" \
          --export=ALL,DATASET="$DS",MODE="$MODE" \
          "$REPO/jobs/benchmark_main.sbatch")
    # 2) supplementary array: exp 3 (ACI), 7, 5 (sliding + ACI); task i waits for main task i
    SJID=$(sbatch --parsable --job-name="xad_xs_${DS}_${MODE}" --time="$ST" --mem="$M" $ARR \
          --dependency=aftercorr:"$JID" --kill-on-invalid-dep=yes \
          --output="$RUN/logs/${DS}_supp_${MODE}_%A_%a.out" \
          --error="$RUN/logs/${DS}_supp_${MODE}_%A_%a.err" \
          --export=ALL,DATASET="$DS",MODE="$MODE",MAIN_JOB="$JID" \
          "$REPO/jobs/benchmark_supp.sbatch")
    # 3) classical baselines (CPU, independent of the deep models); full mode only
    CJID=-
    if [ "$MODE" = full ]; then
        CJID=$(sbatch --parsable --job-name="xad_xc_${DS}" --time=24:00:00 --mem=64G \
              --output="$RUN/logs/${DS}_classical_%j.out" \
              --error="$RUN/logs/${DS}_classical_%j.err" \
              --export=ALL,DATASET="$DS" "$REPO/jobs/benchmark_classical.sbatch")
    fi
    echo "| $(date -u +%FT%TZ) | $MODE | $DS | main $JID, supp $SJID, classical $CJID | ${T} / ${M} | paper-code md5 $CODE_SUM, extra-code md5 $EXTRA_SUM |" >> "$LOG"
    echo "submitted $MODE $DS -> main $JID supp $SJID classical $CJID"
done
