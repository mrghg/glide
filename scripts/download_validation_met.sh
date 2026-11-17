#!/bin/bash
# =============================================================================
# Download the ERA5 model-level windows of docs/validation-plan.md §4
# ("Meteorology windows and sizes") from ARCO ERA5 on GCS.
#
# One Zarr store per window, written to $GLIDE_DATA/era5-model/ as
#   <DOMAIN>_<start YYYYMMDD>-<end YYYYMMDD>_ml.zarr
# Stores that already exist are skipped, so the script is safe to re-run after
# a job runs out of time (an unfinished window leaves only a *.tmp-download
# directory, which is removed and redone).
#
# Run from the repo root, either inside an allocation:
#     scripts/download_validation_met.sh [PARALLEL] [FILTER]
# or as a job (submit it yourself; agents never submit jobs):
#     mkdir -p logs && sbatch scripts/download_validation_met.sh
#
# PARALLEL: windows downloaded at once (default 3). Each process peaks at about
#   25 GB of memory, and a one-GPU allocation on Isambard AI gets 115 GB, so
#   more than 4 at once is killed for running out of memory. The node's CPUs
#   are shared among the processes' dask threads. FILTER: optional grep pattern on the window name, e.g.
#   'EUR_2018' or '1983'.
#
# Measured on a GH200 node (17 Nov 2026): about 25 s per hour of EUR data,
# about 10 minutes per day of meteorology for one process.
# =============================================================================

#SBATCH --job-name=glide-era5-validation
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

PARALLEL="${1:-3}"
# Share the allocation's CPUs among the parallel downloads (dask reads DASK_*).
export DASK_NUM_WORKERS="$(( ${SLURM_CPUS_ON_NODE:-$(nproc)} / PARALLEL ))"
FILTER="${2:-}"

if [[ -z "${GLIDE_DATA:-}" && -f dev/agent/isambard/env.local.sh ]]; then
	# shellcheck disable=SC1091
	source dev/agent/isambard/env.local.sh
fi
: "${GLIDE_DATA:?set GLIDE_DATA (dev/agent/isambard/env.local.sh)}"
OUT_DIR="${GLIDE_DATA}/era5-model"
LOG_DIR="${OUT_DIR}/logs"
mkdir -p "${OUT_DIR}" "${LOG_DIR}"

PYTHON=".venv/bin/python"
[[ -x "${PYTHON}" ]] || { echo "ERROR: ${PYTHON} not found" >&2; exit 1; }

# Domains: ERA5 grid points, validation-plan §4.
declare -A BOX=(
	[EUR]="-30 40 32 72"
	[NAM]="-120 -55 22 58"
)

# name domain start end   (UTC, end inclusive). Case library first.
WINDOWS=$(cat <<'EOF'
EUR_20181222-20190115 EUR 2018-12-22T00:00 2019-01-15T00:00
NAM_20181222-20190115 NAM 2018-12-22T00:00 2019-01-15T00:00
EUR_20190621-20190715 EUR 2019-06-21T00:00 2019-07-15T00:00
NAM_20190621-20190715 NAM 2019-06-21T00:00 2019-07-15T00:00
NAM_19870105-19870118 NAM 1987-01-05T00:00 1987-01-18T00:00
NAM_19800708-19800712 NAM 1980-07-08T00:00 1980-07-12T00:00
EUR_19941023-19941028 EUR 1994-10-23T00:00 1994-10-28T00:00
EUR_19941114-19941119 EUR 1994-11-14T00:00 1994-11-19T00:00
NAM_19830918-19830922 NAM 1983-09-18T00:00 1983-09-22T00:00
NAM_19830925-19830929 NAM 1983-09-25T00:00 1983-09-29T00:00
NAM_19831002-19831006 NAM 1983-10-02T00:00 1983-10-06T00:00
NAM_19831014-19831017 NAM 1983-10-14T00:00 1983-10-17T00:00
NAM_19831026-19831101 NAM 1983-10-26T00:00 1983-11-01T00:00
EOF
)

download_one() {
	local name=$1 dom=$2 t0=$3 t1=$4
	local out="${OUT_DIR}/${name}_ml.zarr"
	if [[ -d "${out}" ]]; then
		echo "skip   ${name} (exists)"
		return 0
	fi
	rm -rf "${out}.tmp-download"
	read -r lon0 lon1 lat0 lat1 <<<"${BOX[$dom]}"
	echo "start  ${name}  $(date -u +%H:%M:%S)"
	if "${PYTHON}" -W ignore scripts/download_sample_cube.py --levels model \
		--out-path "${out}" --time-start "${t0}" --time-end "${t1}" \
		--lon-min "${lon0}" --lon-max "${lon1}" --lat-min "${lat0}" --lat-max "${lat1}" \
		>"${LOG_DIR}/${name}.log" 2>&1; then
		echo "done   ${name}  $(date -u +%H:%M:%S)  $(du -sh "${out}" | cut -f1)"
	else
		echo "FAILED ${name}  (see ${LOG_DIR}/${name}.log)"
		return 1
	fi
}

echo "=== ERA5 validation met: $(hostname) job ${SLURM_JOB_ID:-none} git $(git rev-parse --short HEAD) ==="
echo "out: ${OUT_DIR}  parallel: ${PARALLEL}  dask threads each: ${DASK_NUM_WORKERS}  filter: ${FILTER:-none}"

status=0
while read -r name dom t0 t1; do
	[[ -n "${FILTER}" && ! "${name}" =~ ${FILTER} ]] && continue
	while (( $(jobs -rp | wc -l) >= PARALLEL )); do wait -n || status=1; done
	download_one "${name}" "${dom}" "${t0}" "${t1}" &
done <<<"${WINDOWS}"
while (( $(jobs -rp | wc -l) > 0 )); do wait -n || status=1; done

echo "=== finished $(date -u +%H:%M:%S), status ${status} ==="
exit "${status}"
