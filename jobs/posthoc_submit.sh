#!/bin/bash
# Submit the post-hoc analyses P1 to P6 and log every job in docs/RUNLOG.md.
# They reuse the detectors cached by the main runs and never train.
#
#   bash jobs/posthoc_submit.sh                          # P1-P3, P6, P5, P4
#   bash jobs/posthoc_submit.sh acitrace casestudy       # trace and figure inputs
set -euo pipefail
REPO=${XAD_REPO:-$(pwd)}
L=$REPO/runs/posthoc/logs
S=$REPO/jobs/posthoc_job.sbatch
mkdir -p "$L"
PARTS="${*:-policies uncertainty iforest cf}"   # acitrace only when named
has() { [[ " $PARTS " == *" $1 "* ]]; }
log() { echo "| $(date -u +%FT%TZ) | posthoc | $1 | $2 | $3 | $(md5sum $REPO/analysis/posthoc_*.py | md5sum | cut -c1-12) |" >> $REPO/docs/RUNLOG.md; echo "$1 -> $2"; }
if has policies; then
P=$(sbatch --parsable --job-name=xad_ph_pol --partition=GPUQ --gres=gpu:1 --time=10:00:00 --mem=48G --array=0-35 \
     --output=$L/policies_%A_%a.out --error=$L/policies_%A_%a.err --export=ALL,KIND=policies $S)
log "P1-P3 policies (12 ds x 3 det, 5 seeds each)" "$P" "GPUQ 10h"
echo $P > $L/policies_jobid
fi
if has uncertainty; then
P=${P:-$(cat $L/policies_jobid)}
DEP=""; squeue -h -j "$P" 2>/dev/null | grep -q . && DEP="--dependency=afterok:$P"   # only wait if P1 is still queued
U=$(sbatch --parsable --job-name=xad_ph_unc --partition=CPUQ --time=12:00:00 --mem=32G --array=0-11 $DEP \
     --output=$L/uncertainty_%A_%a.out --error=$L/uncertainty_%A_%a.err --export=ALL,KIND=uncertainty $S)
log "P6 uncertainty (12 ds)" "$U" "CPUQ 12h, after $P"
fi
if has iforest; then
F=$(sbatch --parsable --job-name=xad_ph_if --partition=CPUQ --time=72:00:00 --mem=32G --array=0-59 \
     --output=$L/iforest_%A_%a.out --error=$L/iforest_%A_%a.err --export=ALL,KIND=iforest $S)
log "P5 Isolation Forest backbone (12 ds x 5 seeds)" "$F" "CPUQ 72h"
fi
if has cf; then
for D in threew daphnet exathlon gecco ghl metropt3 msl psm skab smap smd swat; do
  C=$(sbatch --parsable --job-name=xad_ph_cf_$D --partition=GPUQ --gres=gpu:1 --time=36:00:00 --mem=48G --array=0-29 \
       --output=$L/cf_${D}_%A_%a.out --error=$L/cf_${D}_%A_%a.err --export=ALL,KIND=cf,DATASET=$D $S)
  log "P4 counterfactual regeneration $D (3 det x 5 seeds x 2 policies)" "$C" "GPUQ 36h"
done
fi
if has acitrace; then
T=$(sbatch --parsable --job-name=xad_ph_aci --partition=CPUQ --time=02:00:00 --mem=16G --array=0-11 \
     --output=$L/acitrace_%A_%a.out --error=$L/acitrace_%A_%a.err --export=ALL,KIND=acitrace $S)
log "ACI infinite-threshold trace (exploratory, 12 ds, CPU replay of P1 scores)" "$T" "CPUQ 2h"
fi
if has cfcpu; then
# P4 runs on CPU nodes: GA-CE took 3.4 s per window on 8 CPU cores and 5-7 s on a GPU
# for these small detectors.
# Each run detects, snapshots and explains on one device, so runs are self-contained.
# CF_ARRAY_<ds> overrides the array range (default 0-29), e.g. for tasks already run on GPUQ.
for D in ${CF_DATASETS:-threew daphnet exathlon gecco ghl metropt3 msl psm skab smap smd swat}; do
  V=CF_ARRAY_$D; R=${!V:-0-29}
  C=$(sbatch --parsable --job-name=xad_ph_cfc_$D --partition=CPUQ --time=24:00:00 --mem=32G --array=$R \
       --output=$L/cf_${D}_%A_%a.out --error=$L/cf_${D}_%A_%a.err --export=ALL,KIND=cf,DATASET=$D $S)
  log "P4 counterfactual regeneration $D on CPUQ, tasks $R (pending GPUQ tasks cancelled)" "$C" "CPUQ 24h"
done
fi
if has casestudy; then
X=$(sbatch --parsable --job-name=xad_ph_case --partition=CPUQ --time=04:00:00 --mem=48G --cpus-per-task=16 --array=0 \
     --output=$L/casestudy_%A_%a.out --error=$L/casestudy_%A_%a.err --export=ALL,KIND=casestudy $S)
log "Case-study illustrations (descriptive): SWaT LSTM-AE s42 uncertainty stream, SKAB CF examples" "$X" "CPUQ 4h"
fi
