#!/bin/bash
# Task A01: run the two comparison configs one after the other on a GH200.
# Usage, from the GLIDE root on a compute node: bash scripts/validation/a01/run.sh
set -euo pipefail
source scripts/gpu_env.sh
export GLIDE_PHASE_TIMERS=1
mkdir -p outputs
for name in model pressure23; do
	/usr/bin/time -v "${GLIDE_PYTHON}" -m lpdm.main \
		--config "scripts/validation/a01/${name}.yaml" --device cuda \
		--output-uri "outputs/a01-${name}" > "outputs/a01-${name}.log" 2>&1
done
