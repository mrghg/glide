#!/bin/bash
# =============================================================================
# Download ERA5 EUROPE domain cubes for GLIDE testing — Isambard AI
#
# Downloads two months sequentially:
#   data/era5/EUROPE_202312.zarr
#   data/era5/EUROPE_202401.zarr
#
# Network-bound job; the GPU is not used but Isambard AI has no pure CPU nodes
# so one GPU must be requested to get an allocation. The EUROPE domain is
# ~3 GB/month uncompressed, so allow generous wall time for slow GCS egress.
#
# SUBMIT FROM THE REPO ROOT:
#     mkdir -p logs
#     sbatch scripts/download_era5_europe.slurm
#
# BEFORE FIRST USE, fill in the lines marked  # >>> EDIT (account / partition).
# =============================================================================

#SBATCH --job-name=glide-era5-download
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=04:00:00                     # >>> EDIT: adjust after a first run
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4                   # xarray/dask decompression threads
#SBATCH --mem=32G                           # headroom for in-memory subset before write
#SBATCH --gres=gpu:1                        # required — Isambard AI has no pure CPU nodes
# #SBATCH --account=CHEM007981                # >>> EDIT: set to your Isambard AI project code
# #SBATCH --partition=workq                   # >>> EDIT: set to the Isambard AI GPU partition

set -euo pipefail

# Run from the directory the job was submitted from (expected: repo root).
cd "${SLURM_SUBMIT_DIR:-$PWD}"

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
echo "==========================="

# --- Downloads -------------------------------------------------------------
for YEAR_MONTH in 202312 202401; do
	echo ""
	echo "--- Downloading EUROPE ${YEAR_MONTH} ---"
	"${PYTHON}" scripts/download_sample_cube.py \
		--domain EUROPE \
		--year-month "${YEAR_MONTH}" \
		--out-dir "${HOME}/data/arco-era5"
	echo "--- Done: EUROPE ${YEAR_MONTH} ---"
done

echo ""
echo "=== All downloads complete ==="
