#!/bin/bash
# =============================================================================
# Build ERA5 ARCO stores on Isambard AI: for each recipe, create the store, fill
# its times with several processes at once, then finish (fill rule, validation).
#
# Lives in GLIDE (validation/met-inputs/); runs met-bazarr from its own checkout.
# Run from the GLIDE repository root, inside an allocation or with sbatch
# (submitted by a person, never by an agent):
#
#   export METBAZARR_OUT=/projects/<project>/glide-data/era5-arco
#   export METBAZARR_REPO=~/code/met-bazarr      # a checkout at the pinned tag
#   R=validation/met-inputs/era5-arco/recipes
#   sbatch validation/met-inputs/jobs/isambard/build_era5.sh $R/EUR-Jan.yaml [more.yaml ...]
#   NPROC=4 FINISH_FLAGS=--validation-input sbatch validation/met-inputs/jobs/isambard/build_era5.sh ...
#
# Environment: met-bazarr's .venv with the arco extra (`uv sync --extra arco`
# in $METBAZARR_REPO).
# NPROC (default 8) fill processes run at once, each reading its 13 variables in
# parallel threads. Measured on a GH200 node (19 Nov 2026): about 12 s per hour
# of data and 2 GB per process; with 8 processes, 24 EUR hours in 2.5 min, so a
# 24-day window takes about an hour. A one-GPU allocation has 115 GB of memory.
# A store that already exists is kept and only its unwritten times are filled
# (met-bazarr >= v0.3.0, --missing-only), so resubmitting resumes after a
# time-out; OVERWRITE=1 recreates it. Exits non-zero if any step or validation fails.
# =============================================================================

#SBATCH --job-name=metbazarr-era5
#SBATCH --output=slurm-%x-%j.out
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --gres=gpu:1                        # Isambard AI has no CPU-only nodes

set -euo pipefail
# GLIDE root: the submit dir under sbatch (the script is copied to a spool dir), else from the script's path.
if [[ -n "${SLURM_JOB_ID:-}" && "$0" == */slurm_script ]]; then cd "${SLURM_SUBMIT_DIR}"; else cd "$(dirname "$0")/../../../.."; fi
[[ $# -ge 1 ]] || { echo "usage: build_era5.sh <recipe.yaml> [...]" >&2; exit 2; }
: "${METBAZARR_OUT:?export METBAZARR_OUT, the directory the recipes write to}"
NPROC="${NPROC:-8}"
METBAZARR_REPO="${METBAZARR_REPO:-${HOME}/code/met-bazarr}"
MB="${METBAZARR_REPO}/.venv/bin/metbazarr"
PY="${METBAZARR_REPO}/.venv/bin/python"
[[ -x "${MB}" ]] || { echo "ERROR: ${MB} not found; run: (cd ${METBAZARR_REPO} && uv sync --extra arco)" >&2; exit 1; }

echo "host $(hostname), job ${SLURM_JOB_ID:-none}, glide $(git rev-parse --short HEAD), met-bazarr $(git -C "${METBAZARR_REPO}" describe --tags --always --dirty), NPROC ${NPROC}, out ${METBAZARR_OUT}"
status=0
for recipe in "$@"; do
	echo "=== ${recipe}  $(date -u +%FT%TZ)"
	out="$("${PY}" -c "from metbazarr.recipe import load_recipe; print(load_recipe('${recipe}').output)")"
	n="$("${PY}" -c "from metbazarr.recipe import load_recipe; print(len(load_recipe('${recipe}').times(1)))")"
	if [[ ! -d "${out}" || -n "${OVERWRITE:-}" ]]; then
		"${MB}" build "${recipe}" --step create ${OVERWRITE:+--overwrite} || { status=1; continue; }
	else
		echo "store exists: filling into it (${out})"
	fi
	block=$(( (n + NPROC - 1) / NPROC ))
	pids=()
	for (( k = 0; k < NPROC; k++ )); do
		a=$(( k * block )); b=$(( a + block < n ? a + block : n ))
		(( a < b )) || continue
		"${MB}" build "${recipe}" --step fill --missing-only --times "${a}:${b}" >"${out%.zarr}.fill-${k}.log" 2>&1 &
		pids+=($!)
	done
	fill_ok=1
	for pid in "${pids[@]}"; do wait "${pid}" || fill_ok=0; done
	if (( ! fill_ok )); then
		echo "FAILED: a fill process failed; see ${out%.zarr}.fill-*.log" >&2
		status=1
		continue
	fi
	rm -f "${out%.zarr}".fill-*.log
	# shellcheck disable=SC2086
	"${MB}" build "${recipe}" --step finish ${FINISH_FLAGS:-} || status=1
done
echo "finished $(date -u +%FT%TZ), status ${status}"
exit "${status}"
