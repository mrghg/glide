#!/bin/bash
# Task CONV: production effect on CPU. One run per variant, 3 threads each.
# usage: run_footprints.sh <case: smoke|jfj> <variant> [variant ...]
# variants: legacy | legacy_s43 (seed 43: noise) | buoy_exp | buoy_exp_trig | c003_exp | legacy_sub4
set -e
W=$(cd "$(dirname "$0")/../../.." && pwd)
MAIN=${GLIDE_MAIN:-/home/u6za/chxmr.u6za/code/glide}
cd "$MAIN"
CASE=$1
shift
case $CASE in
smoke) CFG=$W/configs/smoke_mhd_single_release.yaml; EXTRA="" ;;
jfj) CFG=$W/scripts/validation/zs-repair/case2-jfj.yaml; EXTRA="simulation.length_seconds=259200" ;;
*) echo "unknown case $CASE"; exit 1 ;;
esac
O=$MAIN/outputs/agents/CONV/footprints/$CASE
mkdir -p $O
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 PYTHONPATH=$W/src
E=convection.emanuel
for v in "$@"; do
  case $v in
  legacy) SET="" ;;
  buoy_exp) SET="$E.closure=buoyancy $E.application=exponential" ;;
  buoy_exp_trig) SET="$E.closure=buoyancy $E.application=exponential $E.trigger_height_above_lcl_m=${TRIG_DZ:-150}" ;;
  c003_exp) SET="$E.closure_c=0.003 $E.application=exponential" ;;
  legacy_sub4) SET="$E.application=substeps $E.n_substeps=4" ;;
  legacy_s43) SET="release.seed=43" ;;
  *) echo "unknown variant $v"; exit 1 ;;
  esac
  /usr/bin/time -v .venv/bin/python $W/scripts/validation/conv/run_footprint.py $CFG $O/$v --set $EXTRA $SET > $O/$v.log 2>&1 &
done
wait
echo DONE
