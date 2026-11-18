#!/bin/bash
# =============================================================================
# Download ERA5 windows from ARCO (GCS) as GLIDE meteorology stores.
#
# Wraps scripts/download_sample_cube.py. One Zarr store per window, named
#     <DOMAIN>_<YYYYMMDD of first time>-<YYYYMMDD of last time>[_ml].zarr
# (`_ml` for model levels). Stores that already exist are skipped, so the script
# can be re-run after a job runs out of time; an unfinished window leaves only a
# *.tmp-download directory, which is removed and redone.
#
# Usage (from the repo root; inside an allocation, or submitted with sbatch,
# which only the human does: agents never submit jobs):
#
#   scripts/download_era5.sh --preset validation              # validation-plan §4, 13 windows
#   scripts/download_era5.sh --preset reference-dev           # EUROPE Dec 2023 + Jan 2024 (task A01)
#   scripts/download_era5.sh --domain EUR --start 2019-01-08T00:00 --end 2019-01-08T23:00
#   sbatch scripts/download_era5.sh --preset reference-dev --out-dir /some/dir
#
# Options:
#   --preset NAME          validation | reference-dev
#   --domain D --start T --end T
#                          one window instead of a preset; D is a name in the
#                          table below, T is ISO UTC; both ends included
#   --levels L             model (default) | pressure
#   --out-dir DIR          default: $GLIDE_DATA/era5-model (model) or
#                          $GLIDE_DATA/era5-pressure (pressure)
#   --parallel N           windows at once (default 3). Each process peaks at
#                          about 25 GB, and a one-GPU allocation on Isambard AI
#                          has 115 GB, so more than 4 is killed for memory. The
#                          allocation's CPUs are shared among dask's threads.
#   --filter REGEX         only windows whose store name matches
#   --dry-run              list what would be done, download nothing
#
# A window can fail its final check if the source holds a missing value, as
# ARCO's ERA5 heat flux does at one cell on 2018-12-28 21:00. The incomplete
# store is kept as *.tmp-download; see docs/met_schema.md ("Validation inputs")
# for the documented fill before installing it by hand.
#
# Measured on a GH200 node (17 Nov 2026): a 24-day EUR window takes about
# 70 minutes with 24 dask threads; time is set by the number of whole-globe
# source chunks read, so a larger box costs little more time.
# =============================================================================

#SBATCH --job-name=glide-era5
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --gres=gpu:1                        # Isambard AI has no CPU-only nodes

set -euo pipefail
# Repo root: the submit dir under sbatch (the script is copied to a spool dir), else the script's parent.
if [[ -n "${SLURM_JOB_ID:-}" && "$0" == */slurm_script ]]; then cd "${SLURM_SUBMIT_DIR}"; else cd "$(dirname "$0")/.."; fi

usage() { sed -n '/^# Usage/,/^# A window/p' "$0" | sed 's/^# \{0,1\}//' | head -n -1; exit "${1:-0}"; }

PRESET="" DOMAIN="" T0="" T1="" LEVELS="model" OUT_DIR="" PARALLEL=3 FILTER="" DRY_RUN=0
while [[ $# -gt 0 ]]; do
	case "$1" in
		--preset) PRESET="$2"; shift 2 ;;
		--domain) DOMAIN="$2"; shift 2 ;;
		--start) T0="$2"; shift 2 ;;
		--end) T1="$2"; shift 2 ;;
		--levels) LEVELS="$2"; shift 2 ;;
		--out-dir) OUT_DIR="$2"; shift 2 ;;
		--parallel) PARALLEL="$2"; shift 2 ;;
		--filter) FILTER="$2"; shift 2 ;;
		--dry-run) DRY_RUN=1; shift ;;
		-h | --help) usage 0 ;;
		*) echo "unknown option: $1" >&2; usage 1 ;;
	esac
done

# Domains: lon_min lon_max lat_min lat_max (degrees; ERA5 grid points).
declare -A BOX=(
	[EUR]="-30 40 32 72"           # validation-plan §4
	[NAM]="-120 -55 22 58"         # validation-plan §4
	[EUROPE]="-98 39.5 10.6 79.2"  # the development domain of the reference config
)

# Presets: domain start end (UTC, both ends included). Validation: case library first.
preset_windows() {
	case "$1" in
		validation)
			cat <<'EOF'
EUR 2018-12-22T00:00 2019-01-15T00:00
NAM 2018-12-22T00:00 2019-01-15T00:00
EUR 2019-06-21T00:00 2019-07-15T00:00
NAM 2019-06-21T00:00 2019-07-15T00:00
NAM 1987-01-05T00:00 1987-01-18T00:00
NAM 1980-07-08T00:00 1980-07-12T00:00
EUR 1994-10-23T00:00 1994-10-28T00:00
EUR 1994-11-14T00:00 1994-11-19T00:00
NAM 1983-09-18T00:00 1983-09-22T00:00
NAM 1983-09-25T00:00 1983-09-29T00:00
NAM 1983-10-02T00:00 1983-10-06T00:00
NAM 1983-10-14T00:00 1983-10-17T00:00
NAM 1983-10-26T00:00 1983-11-01T00:00
EOF
			;;
		reference-dev)
			cat <<'EOF'
EUROPE 2023-12-01T00:00 2023-12-31T23:00
EUROPE 2024-01-01T00:00 2024-01-31T23:00
EOF
			;;
		*) echo "unknown preset: $1 (validation | reference-dev)" >&2; exit 1 ;;
	esac
}

if [[ -n "${PRESET}" && -n "${DOMAIN}${T0}${T1}" ]]; then
	echo "give either --preset or --domain/--start/--end, not both" >&2; exit 1
fi
if [[ -n "${PRESET}" ]]; then
	WINDOWS="$(preset_windows "${PRESET}")"
elif [[ -n "${DOMAIN}" && -n "${T0}" && -n "${T1}" ]]; then
	WINDOWS="${DOMAIN} ${T0} ${T1}"
else
	echo "give --preset, or all of --domain, --start and --end" >&2; usage 1
fi

case "${LEVELS}" in
	model) SUFFIX="_ml"; SUBDIR="era5-model" ;;
	pressure) SUFFIX=""; SUBDIR="era5-pressure" ;;
	*) echo "--levels must be model or pressure" >&2; exit 1 ;;
esac

if [[ -z "${OUT_DIR}" ]]; then
	if [[ -z "${GLIDE_DATA:-}" && -f dev/agent/isambard/env.local.sh ]]; then
		# shellcheck disable=SC1091
		source dev/agent/isambard/env.local.sh
	fi
	: "${GLIDE_DATA:?set GLIDE_DATA or pass --out-dir}"
	OUT_DIR="${GLIDE_DATA}/${SUBDIR}"
fi
LOG_DIR="${OUT_DIR}/logs"

PYTHON=".venv/bin/python"
[[ -x "${PYTHON}" ]] || { echo "ERROR: ${PYTHON} not found" >&2; exit 1; }

# Share the allocation's CPUs among the parallel downloads (dask reads DASK_*).
export DASK_NUM_WORKERS="$(( ${SLURM_CPUS_ON_NODE:-$(nproc)} / PARALLEL ))"

store_name() {  # domain start end -> store name
	local d0=${2%%T*} d1=${3%%T*}
	echo "${1}_${d0//-/}-${d1//-/}${SUFFIX}"
}

download_one() {
	local name=$1 dom=$2 t0=$3 t1=$4
	local out="${OUT_DIR}/${name}.zarr"
	if [[ -d "${out}" ]]; then
		echo "skip   ${name} (exists)"
		return 0
	fi
	rm -rf "${out}.tmp-download"
	read -r lon0 lon1 lat0 lat1 <<<"${BOX[$dom]}"
	echo "start  ${name}  $(date -u +%H:%M:%S)"
	if "${PYTHON}" -W ignore scripts/download_sample_cube.py --levels "${LEVELS}" \
		--out-path "${out}" --time-start "${t0}" --time-end "${t1}" \
		--lon-min "${lon0}" --lon-max "${lon1}" --lat-min "${lat0}" --lat-max "${lat1}" \
		>"${LOG_DIR}/${name}.log" 2>&1; then
		echo "done   ${name}  $(date -u +%H:%M:%S)  $(du -sh "${out}" | cut -f1)"
	else
		echo "FAILED ${name}  (see ${LOG_DIR}/${name}.log)"
		return 1
	fi
}

echo "=== ERA5 download: $(hostname) job ${SLURM_JOB_ID:-none} git $(git rev-parse --short HEAD) ==="
echo "levels: ${LEVELS}  out: ${OUT_DIR}  parallel: ${PARALLEL}  dask threads each: ${DASK_NUM_WORKERS}  filter: ${FILTER:-none}"

# Validate every window before starting any download.
while read -r dom t0 t1; do
	[[ -n "${BOX[$dom]:-}" ]] || { echo "unknown domain: ${dom} (known: ${!BOX[*]})" >&2; exit 1; }
	[[ "${t0}" < "${t1}" ]] || { echo "start must be before end: ${t0} ${t1}" >&2; exit 1; }
done <<<"${WINDOWS}"

if (( DRY_RUN )); then
	while read -r dom t0 t1; do
		name="$(store_name "${dom}" "${t0}" "${t1}")"
		[[ -n "${FILTER}" && ! "${name}" =~ ${FILTER} ]] && continue
		state="download"; [[ -d "${OUT_DIR}/${name}.zarr" ]] && state="exists, skip"
		echo "${name}.zarr  ${dom} [${BOX[$dom]}]  ${t0} .. ${t1}  (${state})"
	done <<<"${WINDOWS}"
	exit 0
fi

mkdir -p "${OUT_DIR}" "${LOG_DIR}"
status=0
while read -r dom t0 t1; do
	name="$(store_name "${dom}" "${t0}" "${t1}")"
	[[ -n "${FILTER}" && ! "${name}" =~ ${FILTER} ]] && continue
	while (( $(jobs -rp | wc -l) >= PARALLEL )); do wait -n || status=1; done
	download_one "${name}" "${dom}" "${t0}" "${t1}" &
done <<<"${WINDOWS}"
while (( $(jobs -rp | wc -l) > 0 )); do wait -n || status=1; done

echo "=== finished $(date -u +%H:%M:%S), status ${status} ==="
exit "${status}"
