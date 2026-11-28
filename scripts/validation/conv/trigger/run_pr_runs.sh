#!/bin/bash
# Decision 0034: CPU footprint runs before (the pre-0034 code, PR #78's head) and
# after (this branch) the change of default trigger. One run per argument, all in
# parallel, THREADS threads each (default 4).
# usage: [SRC=<lpdm src dir>] [THREADS=n] run_pr_runs.sh <case>:<variant> [...]
#   case: smoke (configs/smoke_mhd_single_release.yaml) or jfj
#         (scripts/validation/zs-repair/case2-jfj.yaml, 3 days back).
#   variant: default | default_s43 | glide | glide_s43 (glide: trigger 'glide').
#   SRC: the code to run (default this checkout's src; for "before", PR #78's src,
#        written to outputs/agents/CONV/trigpr/pre0034_src/src). Outputs go to
#        outputs/agents/CONV/trigpr/<before|after>/<case>/<variant>.
set -e
W=$(cd "$(dirname "$0")/../../../.." && pwd)
MAIN=${GLIDE_MAIN:-/home/u6za/chxmr.u6za/code/glide}
SRC=${SRC:-$W/src}
WHEN=${WHEN:-after}
T=${THREADS:-4}
cd "$MAIN"
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=$T MKL_NUM_THREADS=$T PYTHONPATH=$SRC
for arg in "$@"; do
  CASE=${arg%%:*}
  v=${arg#*:}
  case $CASE in
  smoke) CFG=$W/configs/smoke_mhd_single_release.yaml; EXTRA="" ;;
  jfj) CFG=$W/scripts/validation/zs-repair/case2-jfj.yaml; EXTRA="simulation.length_seconds=259200" ;;
  *) echo "unknown case $CASE"; exit 1 ;;
  esac
  case $v in
  default) SET="" ;;
  default_s43) SET="release.seed=43" ;;
  glide) SET="convection.emanuel.trigger=glide" ;;
  glide_s43) SET="convection.emanuel.trigger=glide release.seed=43" ;;
  *) echo "unknown variant $v"; exit 1 ;;
  esac
  O=$MAIN/outputs/agents/CONV/trigpr/$WHEN/$CASE
  mkdir -p "$O"
  /usr/bin/time -v .venv/bin/python "$W/scripts/validation/conv/run_footprint.py" "$CFG" "$O/$v" \
    --set $EXTRA $SET > "$O/$v.log" 2>&1 &
done
wait
echo DONE
