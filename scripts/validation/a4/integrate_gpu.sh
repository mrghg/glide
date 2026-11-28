#!/bin/bash
# Task A4 integration with main: compiled CUDA smoke runs of a snapshot in the
# local metric frame, endpoint and trapezoid multi-bin (configs as in
# integrate_smoke.sh). Run it under flock on the shared GPU lock.
#
#   integrate_gpu.sh <snapshot> <out dir> <config stem>...
#
# Writes <out>/gpu-metric-<stem>{,.log}. Check the logs for `compiles=2` and no
# "WON'T CONVERT".
set -u
SNAP=$1; OUT=$2; shift 2
HERE=$(dirname "$(readlink -f "$0")")
source "${GLIDE_ENV_ROOT:-/home/u6za/chxmr.u6za/code/glide}/scripts/gpu_env.sh" || exit 1
export GLIDE_COMPILE=1
unset GLIDE_POSITION_FRAME
PY=${GLIDE_PYTHON:-/home/u6za/chxmr.u6za/code/glide/.venv/bin/python}
cd "$SNAP" || exit 1
for stem in "$@"; do
  name=gpu-metric-$stem
  if [ -d "$OUT/$name" ]; then echo "skip $name (exists)"; continue; fi
  echo "start $(date -Is) $name"
  /usr/bin/time -v "$PY" "$HERE/run_variant.py" --src "$SNAP/src" --variant plain -- \
    --config "configs/$stem.yaml" --device cuda --output-uri "$OUT/$name" \
    > "$OUT/$name.log" 2>&1
  echo "end $(date -Is) $name rc=$?"
done
