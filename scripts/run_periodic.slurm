#!/bin/bash
# =============================================================================
# GLIDE periodic (multi-release) run as a SLURM batch job.
#
# Runs `python -m lpdm.main` on a multi-release YAML config (default:
# configs/example_mhd_january_periodic.yaml) on a single compute node.
# The model is a single Python process: it parallelises via torch CPU threads
# OR a single GPU — it is NOT MPI, so this is a 1-node / 1-task job.
#
# SUBMIT FROM THE REPO ROOT:
#     mkdir -p slurm_logs
#     sbatch scripts/run_periodic.slurm
# Override the config / device without editing this file:
#     sbatch scripts/run_periodic.slurm configs/example_mhd_january.yaml
#     GLIDE_DEVICE=cuda sbatch scripts/run_periodic.slurm   # see GPU block below
#
# BEFORE FIRST USE, fill in the lines marked  # >>> EDIT:
#   - account / partition (cluster-specific; delete if your cluster has defaults)
#   - module load / venv activation for your cluster's Python
#   - confirm the met data is readable at the path in the YAML `io.zarr_store`
#     (default ~/data/arco-era5/EUROPE_*.zarr) from the compute node's filesystem
# =============================================================================

#SBATCH --job-name=glide-periodic
#SBATCH --output=slurm_logs/%x-%j.out      # %x=job name, %j=job id (dir must exist at submit)
#SBATCH --error=slurm_logs/%x-%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16                  # see threading note below: ~16 is the
                                            # sweet spot; 48 is ~25% SLOWER here.
#SBATCH --mem=32G                           # >>> EDIT: steady-state RSS ~3-8 GB; headroom for met cache
#SBATCH --account=CHEM007981                # >>> EDIT: uncomment + set if your cluster requires it
#SBATCH --partition=compute

# --- GPU alternative -------------------------------------------------------
# The model is device-agnostic (cuda -> mps -> cpu). A single GPU is much faster
# for the interpolation-heavy step. To run on GPU instead of CPU:
#   1. comment out the CPU --partition above,
#   2. uncomment the two lines below (set the GPU partition),
#   3. submit with  GLIDE_DEVICE=cuda sbatch scripts/run_periodic.slurm
#      (and make sure the venv's torch is a CUDA build).
##SBATCH --partition=CHANGE_ME_GPU
##SBATCH --gres=gpu:1
# ---------------------------------------------------------------------------

set -euo pipefail

# Run from the directory the job was submitted from (expected: repo root).
cd "${SLURM_SUBMIT_DIR:-$PWD}"

# --- Python environment ----------------------------------------------------
# >>> EDIT: load your cluster's Python (delete if the venv is self-contained).
# module load python/3.11
PYTHON=".venv/bin/python"
if [[ ! -x "${PYTHON}" ]]; then
	echo "ERROR: ${PYTHON} not found. Create the venv and editable-install first:" >&2
	echo "  uv venv .venv && uv pip install --python .venv/bin/python -e ." >&2
	exit 1
fi

# --- CPU threading --------------------------------------------------------
# Benchmarking the real engine on a 48-core node (20k AND 200k particles) found
# ~16 torch threads is the sweet spot; one-thread-per-core (48) is ~25% SLOWER
# because the per-step tensors are small enough that thread lock-contention
# dominates before the cores saturate. GLIDE_NUM_THREADS drives torch's intra-op
# pool directly (lpdm.main._configure_cpu_threads); the *_NUM_THREADS exports
# pin the BLAS/OpenMP backends to match so nothing oversubscribes.
#
# THROUGHPUT TIP: because one run only needs ~16 cores, a 48-core node fits ~3
# concurrent runs. For parameter sweeps, submit 3 jobs with --cpus-per-task=16
# (or one job array) instead of one 48-thread job — ~3x aggregate throughput.
export GLIDE_NUM_THREADS="${GLIDE_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-16}}"
export OMP_NUM_THREADS="${GLIDE_NUM_THREADS}"
export MKL_NUM_THREADS="${GLIDE_NUM_THREADS}"
export OPENBLAS_NUM_THREADS="${GLIDE_NUM_THREADS}"

# --- Run parameters --------------------------------------------------------
CONFIG="${1:-configs/example_mhd_january_periodic.yaml}"   # 1st arg overrides
DEVICE="${GLIDE_DEVICE:-cpu}"                               # cpu | cuda | mps | auto
OUTPUT_URI="outputs/${SLURM_JOB_NAME:-glide}-${SLURM_JOB_ID:-local}"

# --- Provenance (lands in the .out log) ------------------------------------
echo "=== GLIDE periodic run ==="
echo "host:     $(hostname)"
echo "date:     $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "job:      ${SLURM_JOB_ID:-<none>} on ${SLURM_JOB_NODELIST:-localhost}"
echo "cpus:     ${SLURM_CPUS_PER_TASK:-?}  mem: ${SLURM_MEM_PER_NODE:-?} MB"
echo "git rev:  $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "config:   ${CONFIG}"
echo "device:   ${DEVICE}"
echo "output:   ${OUTPUT_URI}"
echo "=========================="

# Direct invocation (not srun) — GLIDE is a single-process, non-MPI job.
# srun wraps the launcher in SLURM's PMI/PMIx task-launch machinery which can
# fail with "Unspecified error" on some cluster configurations even for ntasks=1.
"${PYTHON}" -m lpdm.main \
	--config "${CONFIG}" \
	--device "${DEVICE}" \
	--output-uri "${OUTPUT_URI}"

echo "=== GLIDE run complete: ${OUTPUT_URI} ==="

# =============================================================================
# Sizing notes
#   - Walltime scales with (n_releases x n_particles_per_release x backward
#     steps). The shipped config (24 releases x 2000 particles x 30-day backward
#     at dt=60 = 43,200 steps) is heavy on CPU — budget several hours and tune
#     --time after a first run. A GPU run is typically far faster.
#   - Memory: the in-RAM footprint tensor is one batch's worth
#     (max_releases_per_batch x n_time_bins x n_z x n_y x n_x x 4 B); the shipped
#     config is < 50 MB. The dominant RSS is the stitched met cache (a few GB).
#   - To make a run finish faster for a smoke test: lower n_releases /
#     n_particles_per_release / simulation.length_seconds in a copy of the YAML.
# =============================================================================
