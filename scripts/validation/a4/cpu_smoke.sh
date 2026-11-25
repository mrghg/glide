#!/bin/bash
# Task A4 CPU acceptance runs on the smoke config, one after another, $A4_THREADS
# threads (default 16), with CUDA hidden.
#
#   cpu_smoke.sh <main src dir> <A4 src dir> <out dir> [path suffix, e.g. -static]
#
# Extra environment (e.g. GLIDE_STATIC_SUBSTEPS=1) is inherited. Writes run
# directories and logs <out dir>/cpu-<variant><suffix>{,.log}.
set -u
MAIN_SRC=$1; A4_SRC=$2; OUT=$3; SUF=${4:-}
PY=/home/u6za/chxmr.u6za/code/glide/.venv/bin/python
HERE=$(dirname "$(readlink -f "$0")")
CFG=$(readlink -f "$HERE/../../../configs/smoke_mhd_single_release.yaml")
T=${A4_THREADS:-16}
export GLIDE_NUM_THREADS=$T OMP_NUM_THREADS=$T CUDA_VISIBLE_DEVICES=""
run() {  # name src variant [VAR=value...]
  local name=$1 src=$2 variant=$3; shift 3
  if [ -d "$OUT/$name$SUF" ]; then echo "skip $name$SUF (exists)"; return; fi
  echo "start $(date -Is) $name$SUF"
  ( for kv in "$@"; do export "${kv?}"; done
    cd "$(dirname "$src")" && /usr/bin/time -v "$PY" "$HERE/run_variant.py" --src "$src" \
      --variant "$variant" -- --config "$CFG" --device cpu --output-uri "$OUT/$name$SUF" ) \
    > "$OUT/$name$SUF.log" 2>&1
  echo "end $(date -Is) $name$SUF rc=$?"
}
run cpu-main "$MAIN_SRC" plain
run cpu-a4-degrees "$A4_SRC" plain GLIDE_POSITION_FRAME=degrees
run cpu-a4-metric "$A4_SRC" plain
run cpu-a4-metric-shifted "$A4_SRC" shifted_origin
run cpu-main-ulp "$MAIN_SRC" ulp
