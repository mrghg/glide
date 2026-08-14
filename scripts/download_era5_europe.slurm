#!/bin/bash
# =============================================================================
# Download ERA5 EUROPE domain cubes for GLIDE testing — Isambard AI
#
# Downloads months sequentially, either on 37 pressure levels (default) or
# ERA5's 137 native model (hybrid) levels — see docs/met_schema.md.
#
# Network-bound job; the GPU is not used but Isambard AI has no pure CPU nodes
# so one GPU must be requested to get an allocation. The EUROPE domain is
# ~3 GB/month uncompressed on pressure levels; model levels carry ~3.7x the
# vertical data (137 vs 37 levels), so budget more wall time, memory, and
# egress for --levels model.
#
# SUBMIT FROM THE REPO ROOT:
#     mkdir -p logs
#     sbatch scripts/download_era5_europe.slurm             # pressure levels (default)
#     sbatch scripts/download_era5_europe.slurm model        # native model levels
#
# Pressure-level cubes land in ${HOME}/data/arco-era5/; model-level cubes in
# ${HOME}/data/arco-era5-model/ (kept separate so the two met types are never
# confused on disk, on top of the _ml filename suffix download_sample_cube.py
# already adds).
#
# BEFORE FIRST USE, fill in the lines marked  # >>> EDIT (account / partition).
# =============================================================================

#SBATCH --job-name=glide-era5-download
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=01:00:00                     # >>> EDIT: adjust after a first run (longer for model levels)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4                   # xarray/dask decompression threads
#SBATCH --mem=32G                           # headroom for in-memory subset before write (bump for model levels)
#SBATCH --gres=gpu:1                        # required — Isambard AI has no pure CPU nodes
# #SBATCH --account=CHEM007981                # >>> EDIT: set to your Isambard AI project code
# #SBATCH --partition=workq                   # >>> EDIT: set to the Isambard AI GPU partition

set -euo pipefail

# Run from the directory the job was submitted from (expected: repo root).
cd "${SLURM_SUBMIT_DIR:-$PWD}"

# --- Vertical coordinate (first sbatch argument) ----------------------------
LEVELS="${1:-pressure}"
case "${LEVELS}" in
	pressure) OUT_DIR="${HOME}/data/arco-era5" ;;
	model) OUT_DIR="${HOME}/data/arco-era5-model" ;;
	*)
		echo "ERROR: unknown levels '${LEVELS}' (expected 'pressure' or 'model')." >&2
		exit 1
		;;
esac

PYTHON=".venv/bin/python"
if [[ ! -x "${PYTHON}" ]]; then
	echo "ERROR: ${PYTHON} not found. Create the venv and editable-install first:" >&2
	echo "  uv venv .venv && uv pip install --python .venv/bin/python -e ." >&2
	exit 1
fi

# Limit dask/OMP threads to the allocated CPUs.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${OMP_NUM_THREADS}"
export OPENBLAS_NUM_THREADS="${OMP_NUM_THREADS}"

# --- Provenance ------------------------------------------------------------
echo "=== GLIDE ERA5 download ==="
echo "host:    $(hostname)"
echo "date:    $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "job:     ${SLURM_JOB_ID:-<none>} on ${SLURM_JOB_NODELIST:-localhost}"
echo "git rev: $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "levels:  ${LEVELS}"
echo "out-dir: ${OUT_DIR}"
echo "==========================="

# --- Downloads -------------------------------------------------------------
for YEAR_MONTH in 202312 202401; do
	echo ""
	echo "--- Downloading EUROPE ${YEAR_MONTH} (${LEVELS}) ---"
	"${PYTHON}" scripts/download_sample_cube.py \
		--domain EUROPE \
		--year-month "${YEAR_MONTH}" \
		--out-dir "${OUT_DIR}" \
		--levels "${LEVELS}"
	echo "--- Done: EUROPE ${YEAR_MONTH} ---"
done

echo ""
echo "=== All downloads complete ==="
