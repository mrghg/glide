#!/bin/bash
# Task A04: standard cases 3 and 4, weighted by dry-air density (dry_air) and in
# plain seconds (none), same seed, one GPU run at a time:
#   bash scripts/validation/a04/run.sh
# Then: .venv/bin/python scripts/validation/a04/compare.py outputs
set -euo pipefail
source scripts/gpu_env.sh
mkdir -p outputs
for name in case3-cbw20 case4-lef; do
	cfg="scripts/validation/a04/${name}.yaml"
	sed -e 's/density_weighting: dry_air/density_weighting: none/' "${cfg}" > "outputs/a04-${name}-none.yaml"
	for weighting in dry_air none; do
		src="${cfg}"
		[[ "${weighting}" == none ]] && src="outputs/a04-${name}-none.yaml"
		"${GLIDE_PYTHON}" -m lpdm.main --config "${src}" --device cuda \
			--output-uri "outputs/a04-${name}-${weighting}" > "outputs/a04-${name}-${weighting}.log" 2>&1
	done
done
