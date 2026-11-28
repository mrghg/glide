#!/bin/bash
# Decision 0034: compiled CUDA smoke runs (configs/smoke_mhd_single_release.yaml)
# with the default trigger (flexpart_cloud_base) and with trigger 'glide'.
# Expected: ~2.5 min and < 1.5 GiB of device memory each (decision 0033's runs).
# Run under the shared GPU lock, from the main checkout:
#   flock <main>/outputs/agents/gpu.lock scripts/validation/conv/trigger/run_gpu_smoke.sh
# Check the logs for `compiles=2` and no "WON'T CONVERT".
set -u
W=$(cd "$(dirname "$0")/../../../.." && pwd)
MAIN=${GLIDE_MAIN:-/home/u6za/chxmr.u6za/code/glide}
source "$MAIN/scripts/gpu_env.sh" || exit 1
export GLIDE_COMPILE=1 PYTHONPATH=$W/src OMP_NUM_THREADS=2
cd "$MAIN" || exit 1
O=$MAIN/outputs/agents/CONV/trigpr/gpu
mkdir -p "$O"
status=0
for v in default glide; do
  if [ "$v" = glide ]; then SET="convection.emanuel.trigger=glide"; else SET=""; fi
  /usr/bin/time -v .venv/bin/python "$W/scripts/validation/conv/run_footprint.py" \
    "$W/configs/smoke_mhd_single_release.yaml" "$O/$v" --device cuda --set $SET > "$O/$v.log" 2>&1
  rc=$?
  echo "$v rc=$rc"
  [ "$rc" -eq 0 ] || status=$rc
done
exit "$status"
