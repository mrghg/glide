#!/bin/bash
# Task A4 integration with main (#66 trapezoid, #73, #75): CPU smoke runs of main
# and the merged branch, endpoint and trapezoid (multi-bin), for the bit-identity
# check in the degrees frame and the rounding-level check in the metric frame.
#
#   integrate_smoke.sh <out dir> <main snapshot> <A4 snapshot> <path suffix> <run>...
#
# A snapshot is a directory holding src/, configs/ and data/ (a symlink to the
# met), exported with `git archive`; runs start in it so the config's relative
# `data/` paths resolve. Each <run> is name:snapshot(main|a4):config-stem:variant
# [:VAR=value...], with the variant one of run_variant.py's (plain,
# shifted_origin, ulp). Extra environment (GLIDE_STATIC_SUBSTEPS=1) is inherited.
# Threads: $A4_THREADS (default 8), CUDA hidden. Writes <out>/<name><suffix>{,.log}.
set -u
OUT=$1; MAIN=$2; A4=$3; SUF=$4; shift 4
PY=/home/u6za/chxmr.u6za/code/glide/.venv/bin/python
HERE=$(dirname "$(readlink -f "$0")")
T=${A4_THREADS:-8}
export GLIDE_NUM_THREADS=$T OMP_NUM_THREADS=$T CUDA_VISIBLE_DEVICES=""
status=0
for spec in "$@"; do
  IFS=: read -r name which stem variant envs <<<"$spec"
  snap=$MAIN; [ "$which" = a4 ] && snap=$A4
  # Skip only a run that finished: run_metadata.json is written at the end.
  if [ -f "$OUT/$name$SUF/run_metadata.json" ]; then echo "skip $name$SUF (complete)"; continue; fi
  echo "start $(date -Is) $name$SUF"
  ( IFS=:; for kv in $envs; do export "${kv?}"; done
    cd "$snap" && /usr/bin/time -v "$PY" "$HERE/run_variant.py" --src "$snap/src" \
      --variant "$variant" -- --config "configs/$stem.yaml" --device cpu \
      --output-uri "$OUT/$name$SUF" ) > "$OUT/$name$SUF.log" 2>&1
  rc=$?
  echo "end $(date -Is) $name$SUF rc=$rc"
  [ "$rc" -eq 0 ] || status=$rc
done
exit "$status"
