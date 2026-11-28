#!/bin/bash
# Task A4: the GPU work after the first A/B pair, in three lock holds, so that the
# runs compared for timing go back to back under the same conditions.
#
#   gpu_groups.sh <out dir> <main checkout> <A4 checkout>
#
# Uses gpu_queue.sh and gather_microbench.py from the same directory as this script.
set -u
O=$1; MAIN=$2; A4=$3
HERE=$(dirname "$(readlink -f "$0")")
LOCK=/home/u6za/chxmr.u6za/code/glide/outputs/agents/gpu.lock
Q=$HERE/gpu_queue.sh
echo "group1 wait $(date -Is)"
flock "$LOCK" bash -c "
  echo group1 start \$(date -Is)
  source /home/u6za/chxmr.u6za/code/glide/scripts/gpu_env.sh >/dev/null 2>&1
  /home/u6za/chxmr.u6za/code/glide/.venv/bin/python $HERE/gather_microbench.py > $O/gather_microbench.txt 2>&1
  A4_HOLD_LOCK=1 $Q $O $MAIN $A4 cut-a4-degrees-eager cut-a4-fp32-eager cut-a4-bf16-eager
  echo group1 end \$(date -Is)"
echo "group2 wait $(date -Is)"
flock "$LOCK" bash -c "echo group2 start \$(date -Is); A4_HOLD_LOCK=1 $Q $O $MAIN $A4 a4-fp32-s43; echo group2 end \$(date -Is)"
echo "group3 wait $(date -Is)"
flock "$LOCK" bash -c "echo group3 start \$(date -Is); A4_HOLD_LOCK=1 $Q $O $MAIN $A4 main-s42.r2 a4-fp32-s42.r2 a4-bf16-s42.r2; echo group3 end \$(date -Is)"
echo "all done $(date -Is)"
