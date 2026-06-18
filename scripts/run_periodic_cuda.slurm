#!/bin/bash
# =============================================================================
# GLIDE periodic (multi-release) run on a single GPU — tuned for Isambard AI
# (NVIDIA GH200 Grace-Hopper superchips, 4 per node; ARM64 / aarch64).
#
# GLIDE is device-agnostic: the same code runs on CPU, CUDA, or MPS. The hot
# path (grid_sample interpolation + elementwise OU turbulence) is exactly what a
# GPU accelerates, so a single GH200 should be far faster than the 16-thread CPU
# run (run_periodic.slurm). Use this for the full 30-day production runs; use the
# CPU script for quick local iteration / parameter sweeps.
#
# PREREQUISITES (do these once, on Isambard AI):
#   module load cuda/12.6
#   # Build the venv with an ARM64 + CUDA torch wheel (NOT the x86 default):
#   uv venv .venv
#   uv pip install --python .venv/bin/python -e .
#   # then install a CUDA-enabled aarch64 torch matching cuda/12.6, e.g.:
#   uv pip install --python .venv/bin/python torch --index-url https://download.pytorch.org/whl/cu126
#   # sanity check (must print True):
#   .venv/bin/python -c "import torch; print(torch.cuda.is_available())"
#
# SUBMIT FROM THE REPO ROOT:
#     mkdir -p slurm_logs
#     sbatch scripts/run_periodic_cuda.slurm
#     sbatch scripts/run_periodic_cuda.slurm configs/smoke_mhd_single_release.yaml  # smoke test first!
#
# BEFORE FIRST USE, fill in the lines marked  # >>> EDIT (account / partition).
# =============================================================================

#SBATCH --job-name=glide-periodic-gpu
#SBATCH --output=slurm_logs/%x-%j.out      # %x=job name, %j=job id (dir must exist at submit)
#SBATCH --error=slurm_logs/%x-%j.err
#SBATCH --time=04:00:00                     # GPU run is much faster; trim after a first run
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1                        # one GH200 (a node has 4; we use 1)
#SBATCH --cpus-per-task=16                  # CPU-side work (met I/O, numpy) + dataloader threads
#SBATCH --mem=64G                           # >>> EDIT: Grace has unified memory; headroom for met cache
# #SBATCH --account=CHEM007981                # >>> EDIT: set to your Isambard AI project code
# #SBATCH --partition=workq                   # >>> EDIT: set to the Isambard AI GPU partition

set -euo pipefail

# Run from the directory the job was submitted from (expected: repo root).
cd "${SLURM_SUBMIT_DIR:-$PWD}"

# --- CUDA + Python environment ---------------------------------------------
# Isambard AI needs the CUDA module loaded before the CUDA torch wheel will find
# its runtime. Keep this version in sync with the torch build in the venv.
module load cuda/12.6

PYTHON=".venv/bin/python"
if [[ ! -x "${PYTHON}" ]]; then
	echo "ERROR: ${PYTHON} not found. Create the venv and editable-install first:" >&2
	echo "  uv venv .venv && uv pip install --python .venv/bin/python -e ." >&2
	exit 1
fi

# Fail fast (before the long preflight) if the venv torch can't see the GPU —
# usually means an x86 or CPU-only wheel, or a missing `module load cuda/12.6`.
if ! "${PYTHON}" -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then
	echo "ERROR: torch.cuda.is_available() is False. Check that:" >&2
	echo "  - 'module load cuda/12.6' ran (it is in this script), and" >&2
	echo "  - the venv has an aarch64 CUDA torch wheel (see header PREREQUISITES)." >&2
	exit 1
fi

# CPU threads still matter for the host-side ops (met decode, numpy). The GPU
# carries the per-step physics, so a modest pool is plenty; match the CPU-script
# default. (No effect on the CUDA kernels themselves.)
export GLIDE_NUM_THREADS="${GLIDE_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-16}}"
export OMP_NUM_THREADS="${GLIDE_NUM_THREADS}"
export MKL_NUM_THREADS="${GLIDE_NUM_THREADS}"
export OPENBLAS_NUM_THREADS="${GLIDE_NUM_THREADS}"

# --- Run parameters --------------------------------------------------------
CONFIG="${1:-configs/example_mhd_january_periodic.yaml}"   # 1st arg overrides
DEVICE="${GLIDE_DEVICE:-cuda}"                              # cuda by default here
OUTPUT_URI="outputs/${SLURM_JOB_NAME:-glide}-${SLURM_JOB_ID:-local}"

# --- Provenance (lands in the .out log) ------------------------------------
echo "=== GLIDE periodic GPU run ==="
echo "host:     $(hostname)"
echo "date:     $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "job:      ${SLURM_JOB_ID:-<none>} on ${SLURM_JOB_NODELIST:-localhost}"
echo "gpu:      ${SLURM_GPUS_ON_NODE:-?} x $(${PYTHON} -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")')"
echo "cuda:     $(${PYTHON} -c 'import torch; print(torch.version.cuda)')"
echo "git rev:  $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "config:   ${CONFIG}"
echo "device:   ${DEVICE}"
echo "output:   ${OUTPUT_URI}"
echo "============================="

# Direct invocation (not srun) — GLIDE is a single-process, non-MPI job. srun
# wraps the launcher in SLURM's PMI/PMIx machinery which can fail with
# "Unspecified error" on some clusters even for ntasks=1.
"${PYTHON}" -m lpdm.main \
	--config "${CONFIG}" \
	--device "${DEVICE}" \
	--output-uri "${OUTPUT_URI}"

echo "=== GLIDE run complete: ${OUTPUT_URI} ==="

# =============================================================================
# Sizing notes
#   - Benchmark a SINGLE release first (configs/smoke_mhd_single_release.yaml) to
#     confirm the GPU path works and to time one release before committing the
#     full 720-release / 30-day job. Scale --time from that measurement.
#   - GH200 has large unified (CPU+GPU) memory; the per-batch footprint tensor
#     and met cache that bound CPU runs are not the constraint here. Particle
#     count per batch is the main GPU-memory knob (state is (N, 4) float32 plus a
#     few per-particle work tensors).
#   - If you see the substep-cap WARNING in the log ("hit max_substeps"), the
#     near-surface Δt/τ bias is only partly controlled — lower simulation.dt_seconds
#     or raise turbulence max_substeps; unrelated to device.
# =============================================================================
