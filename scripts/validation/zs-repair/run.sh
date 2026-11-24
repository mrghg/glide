#!/bin/bash
# Run both standard cases, tagged before|after: bash scripts/validation/zs-repair/run.sh before
set -euo pipefail
source scripts/gpu_env.sh
tag=$1
for name in case2-jfj case3-cbw20; do
	"${GLIDE_PYTHON}" -m lpdm.main --config "scripts/validation/zs-repair/${name}.yaml" --device cuda \
		--output-uri "outputs/zs-repair-${name}-${tag}" > "outputs/zs-repair-${name}-${tag}.log" 2>&1 &
done
wait
