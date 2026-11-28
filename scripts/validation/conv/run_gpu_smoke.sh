#!/bin/bash
# Decision 0033: compiled CUDA smoke runs (configs/smoke_mhd_single_release.yaml)
# with the new defaults and with the legacy options. Run under the shared GPU lock:
#   flock <main>/outputs/agents/gpu.lock scripts/validation/conv/run_gpu_smoke.sh
# Check the logs for `compiles=2` and no "WON'T CONVERT".
set -u
W=$(cd "$(dirname "$0")/../../.." && pwd)
MAIN=${GLIDE_MAIN:-/home/u6za/chxmr.u6za/code/glide}
source "$MAIN/scripts/gpu_env.sh" || exit 1
export GLIDE_COMPILE=1 PYTHONPATH=$W/src OMP_NUM_THREADS=2
cd "$MAIN" || exit 1
O=$MAIN/outputs/agents/CONV/pr/gpu
mkdir -p "$O"
E=convection.emanuel
status=0
for v in default legacy; do
  if [ "$v" = legacy ]; then SET="$E.closure=cape_velocity $E.application=cfl_scaled $E.trigger=glide"; else SET=""; fi
  /usr/bin/time -v .venv/bin/python "$W/scripts/validation/conv/run_footprint.py" \
    "$W/configs/smoke_mhd_single_release.yaml" "$O/$v" --device cuda --set $SET > "$O/$v.log" 2>&1
  rc=$?
  echo "$v rc=$rc"
  [ "$rc" -eq 0 ] || status=$rc
done
exit "$status"
