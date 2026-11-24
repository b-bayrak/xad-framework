#!/bin/bash
# Run a script of this folder in the conda environment "xad", with the ten added
# datasets registered, e.g.  bash analysis/run.sh paper_outputs.py
set -euo pipefail
D=$(cd "$(dirname "$0")" && pwd)
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate xad
cd "$D"
XAD_EXTRA=1 PYTHONPATH=$(dirname "$D")/datasets/register:$D exec python "$@"
