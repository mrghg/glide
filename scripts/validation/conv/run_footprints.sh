#!/bin/bash
# Decision 0033: production effect on CPU. One run per variant, 3 threads each,
# all variants of a call in parallel.
# usage: [SRC=<lpdm src dir>] [OUT=<subdir>] run_footprints.sh <case: smoke|jfj> <variant> [variant ...]
#   SRC: the code to run (default: this checkout's src; main's src for "before").
#   OUT: output subdirectory under outputs/agents/CONV/ (default footprints).
# variants (branch code; on main use only default / default_s43):
#   default      the configured defaults (on this branch: buoyancy + exponential)
#   default_s43  the same with release seed 43 (Monte Carlo noise)
#   legacy       closure cape_velocity + application cfl_scaled (the pre-0033 scheme)
#   legacy_s43   legacy with seed 43
#   buoy_exp_trig, c003_exp, legacy_sub4   prototype variants of task CONV
set -e
W=$(cd "$(dirname "$0")/../../.." && pwd)
MAIN=${GLIDE_MAIN:-/home/u6za/chxmr.u6za/code/glide}
SRC=${SRC:-$W/src}
cd "$MAIN"
CASE=$1
shift
case $CASE in
smoke) CFG=$W/configs/smoke_mhd_single_release.yaml; EXTRA="" ;;
jfj) CFG=$W/scripts/validation/zs-repair/case2-jfj.yaml; EXTRA="simulation.length_seconds=259200" ;;
*) echo "unknown case $CASE"; exit 1 ;;
esac
O=$MAIN/outputs/agents/CONV/${OUT:-footprints}/$CASE
mkdir -p $O
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 PYTHONPATH=$SRC
E=convection.emanuel
L="$E.closure=cape_velocity $E.application=cfl_scaled $E.trigger=glide"
for v in "$@"; do
  case $v in
  default) SET="" ;;
  default_s43) SET="release.seed=43" ;;
  legacy) SET="$L" ;;
  legacy_s43) SET="$L release.seed=43" ;;
  buoy_exp_trig) SET="$E.closure=buoyancy $E.application=exponential $E.trigger=glide $E.trigger_height_above_lcl_m=${TRIG_DZ:-150}" ;;
  c003_exp) SET="$E.closure_c=0.003 $E.application=exponential" ;;
  legacy_sub4) SET="$E.closure=cape_velocity $E.application=substeps $E.n_substeps=4 $E.trigger=glide" ;;
  *) echo "unknown variant $v"; exit 1 ;;
  esac
  /usr/bin/time -v .venv/bin/python $W/scripts/validation/conv/run_footprint.py $CFG $O/$v --set $EXTRA $SET > $O/$v.log 2>&1 &
done
wait
echo DONE
