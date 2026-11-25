#!/bin/bash
# Reference GPU benchmark for the sub-stepping change (decision 0026).
# Usage: run_bench.sh <source checkout> <config yaml> <output dir>
# Runs scripts/agent/benchmark.py from <source checkout> (its own src/ on
# PYTHONPATH, via golden_common) with graph capture (GLIDE_COMPILE=1) and
# GPU-synchronised phase timers, so the "step" phase is the step's GPU time.
set -u
SRC=$1
CFG=$2
OUT=$3
cd "$SRC" || exit 1
# shellcheck disable=SC1091
source scripts/gpu_env.sh
export GLIDE_COMPILE=1 GLIDE_PHASE_TIMERS_SYNC=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
"$GLIDE_PYTHON" scripts/agent/benchmark.py \
	--config "$CFG" --device cuda --output "$OUT" --json "$OUT.bench.json" \
	>"$OUT.bench.md" 2>&1
echo "exit $?" >>"$OUT.bench.md"
