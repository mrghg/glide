#!/bin/bash
# Decision 6 study: footprint runs of one case, both triggers, paired seeds, on the
# GPU under the shared lock (one lock per run). Outputs:
# outputs/agents/CONV/trigger/footprints/<case>/<trigger>_s<seed>{,.log,.yaml,.exposure.json}
# usage: run_case.sh <case> <config.yaml> [seeds...]
set -u
W=$(cd "$(dirname "$0")/../../../.." && pwd)
MAIN=${GLIDE_MAIN:-/home/u6za/chxmr.u6za/code/glide}
CASE=$1; CFG=$2; shift 2
SEEDS=${*:-42 43 44}
cd "$MAIN" || exit 1
O=$MAIN/outputs/agents/CONV/trigger/footprints/$CASE
mkdir -p "$O"
for s in $SEEDS; do
  for tr in glide flexpart_cloud_base; do
    name=${tr%%_*}_s$s
    [ "$tr" = flexpart_cloud_base ] && name=fpcb_s$s
    if [ -f "$O/$name.exposure.json" ] && [ -d "$O/$name/footprints.zarr" ]; then echo "skip $name"; continue; fi
    flock "$MAIN/outputs/agents/gpu.lock" bash -c "
      source '$MAIN/scripts/gpu_env.sh' || exit 1
      export GLIDE_COMPILE=1 PYTHONPATH='$W/src' OMP_NUM_THREADS=2
      /usr/bin/time -v .venv/bin/python '$W/scripts/validation/conv/trigger/run_footprint_exposure.py' \
        '$CFG' '$O/$name' --device cuda --set convection.emanuel.trigger=$tr release.seed=$s
    " > "$O/$name.log" 2>&1
    echo "$name rc=$?"
  done
done
