#!/bin/bash
# Task A4 GPU runs: benchmark.py (compiled, synchronised phase timers) on one
# checkout, one config. Run it under flock on the shared GPU lock.
#
#   run_bench.sh <checkout dir> <out dir> <run name> [VAR=value ...]
#
# <checkout dir> is a GLIDE checkout (its src/ is what runs, see
# scripts/agent/golden_common.py); $A4_CONFIG (default
# configs/ab_multisite_perf.yaml) is relative to it. VAR=value pairs are exported
# first (e.g. GLIDE_A4_SEED=43). Writes <out dir>/<run name>{,.log,.bench.md,.bench.json}.
set -u
CHECKOUT=$1; OUT=$2; NAME=$3; shift 3
# The GPU environment of the checkout that holds the venv (GLIDE_ENV_ROOT, default
# the main checkout of this worktree layout).
source "${GLIDE_ENV_ROOT:-/home/u6za/chxmr.u6za/code/glide}/scripts/gpu_env.sh" || exit 1
export GLIDE_COMPILE=1 GLIDE_PHASE_TIMERS_SYNC=1
for kv in "$@"; do export "${kv?}"; done
CFG=${A4_CONFIG:-configs/ab_multisite_perf.yaml}
cd "$CHECKOUT" || exit 1
echo "start $(date -Is) $NAME $CFG $*"
"$GLIDE_PYTHON" scripts/agent/benchmark.py \
  --config "$CFG" --device cuda --output "$OUT/$NAME" --json "$OUT/$NAME.bench.json" \
  > "$OUT/$NAME.bench.md"
rc=$?
echo "end $(date -Is) rc=$rc"
exit $rc
