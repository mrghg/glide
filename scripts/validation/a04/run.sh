#!/bin/bash
# Task A04: standard cases 3 and 4, weighted by dry-air density (dry_air) and in
# plain seconds (none), same seed, one GPU run at a time:
#   bash scripts/validation/a04/run.sh
# Then: .venv/bin/python scripts/validation/a04/compare.py outputs
# NAM stores: until the reader relabels their 0-360 longitudes (PR #56), a NAM
# run configured in -180..180 is driven by the winds at the grid's western edge.
# A04's measurement therefore ran case 4 from a 0-360 twin of case4-lef.yaml
# (lon bounds 240..305 and 240.125..304.875, release lon 269.7268), tagged
# case4-lef360: compare.py outputs case4-lef=case4-lef360
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
