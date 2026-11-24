#!/bin/bash
# Download the raw files of the ten added datasets into data/raw/, in parallel,
# from public sources that need no credentials. Each dataset logs to
# data/raw/logs/<name>.log and gets a SOURCE.txt with URL and UTC timestamp.
# swan and pronto were evaluated and excluded (docs/AUDIT.md, B4 and B9).
#
#   bash datasets/download.sh            # all datasets
#   bash datasets/download.sh gecco smd  # a subset
#
# Re-running skips files that already exist with a non-zero size.

set -uo pipefail
REPO=${XAD_REPO:-$(pwd)}
RAW=$REPO/data/raw
LOG=$RAW/logs
mkdir -p "$RAW" "$LOG"

HF=https://huggingface.co/datasets/PLAN-Lab/mTSBench/resolve/main

fetch() {  # fetch <url> <dest>
    local url=$1 dest=$2
    if [ -s "$dest" ]; then echo "skip (exists) $dest"; return 0; fi
    mkdir -p "$(dirname "$dest")"
    curl -sSL --retry 5 --retry-delay 5 -o "$dest.part" "$url" && mv "$dest.part" "$dest" \
        && echo "ok   $(du -h "$dest" | cut -f1) $dest" || { echo "FAIL $url"; rm -f "$dest.part"; return 1; }
}

provenance() {  # provenance <dir> <text>
    mkdir -p "$1"; { echo "$2"; echo "downloaded_utc: $(date -u +%FT%TZ)"; } > "$1/SOURCE.txt"
}

# mTSBench subsets (Wang et al. 2025, arXiv:2506.21550, CC BY-NC 4.0).
# Official per-entity train/val/test CSVs are kept as shipped.
mtsbench() {  # mtsbench <hf_dir> <local_name>
    local d=$1 name=$2 dest=$RAW/$2
    provenance "$dest" "source: $HF/$d  (mTSBench, PLAN-Lab, CC BY-NC 4.0)"
    curl -s "https://huggingface.co/api/datasets/PLAN-Lab/mTSBench/tree/main/$d" \
      | python3 -c "import json,sys; [print(x['path']) for x in json.load(sys.stdin) if x['path'].endswith('.csv')]" \
      | while read -r p; do fetch "$HF/$p" "$dest/$(basename "$p")"; done
}

threew() {  # Petrobras 3W v2, REAL instances only (WELL-*.parquet), CC BY 4.0
    local dest=$RAW/threew
    provenance "$dest" "source: https://github.com/petrobras/3W/tree/main/dataset (v2.0.0, REAL WELL-* instances only, CC BY 4.0)"
    fetch https://raw.githubusercontent.com/petrobras/3W/main/dataset/dataset.ini "$dest/dataset.ini"
    for c in 0 1 2 3 4 5 6 7 8 9; do
        curl -s "https://api.github.com/repos/petrobras/3W/contents/dataset/$c" \
          | python3 -c "import json,sys; [print(x['download_url']) for x in json.load(sys.stdin) if x['name'].startswith('WELL')]" \
          | while read -r u; do fetch "$u" "$dest/$c/$(basename "$u")"; done
    done
}

metropt() {  # UCI ML repository id 791, CC BY 4.0
    local dest=$RAW/metropt3
    provenance "$dest" "source: https://archive.ics.uci.edu/dataset/791/metropt+3+dataset (CC BY 4.0). Labels: 4 air-leak failure reports (Davari et al. 2021, Table in dataset description)."
    fetch "https://archive.ics.uci.edu/static/public/791/metropt+3+dataset.zip" "$dest/metropt3.zip" \
      && (cd "$dest" && unzip -oq metropt3.zip && echo "unzipped: $(ls)")
}

pronto() {  # Zenodo record 1341583 (Stief et al. 2019), CC BY 4.0
    local dest=$RAW/pronto
    provenance "$dest" "source: https://zenodo.org/records/1341583 (PRONTO heterogeneous benchmark, Stief et al. 2019)"
    fetch "https://zenodo.org/records/1341583/files/README.txt?download=1" "$dest/README.txt"
    fetch "https://zenodo.org/records/1341583/files/Multiphase%20flow%20facility%20case%20study%20technical%20report.pdf?download=1" "$dest/technical_report.pdf"
    fetch "https://zenodo.org/records/1341583/files/PRONTO%20benchmark%20case%20study.zip?download=1" "$dest/pronto.zip" \
      && (cd "$dest" && unzip -oq pronto.zip && echo "unzipped: $(ls)")
}

run() {  # run <name>  -> dispatch, logging to data/raw/logs/<name>.log
    local n=$1
    {
        echo "=== $n  start $(date -u +%FT%TZ)"
        case $n in
            ghl)      mtsbench GHL ghl ;;
            gecco)    mtsbench GECCO gecco ;;
            swan)     mtsbench swan swan_sf ;;
            smd)      mtsbench SMD smd ;;
            psm)      mtsbench PSM psm ;;
            exathlon) mtsbench Exathlon exathlon ;;
            daphnet)  mtsbench Daphnet daphnet ;;
            smap)     mtsbench SMAP smap ;;
            msl)      mtsbench MSL msl ;;
            threew)   threew ;;
            metropt)  metropt ;;
            pronto)   pronto ;;
            *) echo "unknown dataset $n"; return 1 ;;
        esac
        echo "=== $n  end   $(date -u +%FT%TZ)"
    } > "$LOG/$n.log" 2>&1
}

ALL=(ghl gecco swan smd psm exathlon daphnet smap msl threew metropt pronto)
TARGETS=("${@:-${ALL[@]}}")
for n in "${TARGETS[@]}"; do run "$n" & done
wait
echo "all downloads finished $(date -u +%FT%TZ)"
grep -h "FAIL\|=== .* end" "$LOG"/*.log
