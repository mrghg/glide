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
#   module load cudatoolkit/24.11_12.6
#   # Build the venv with an ARM64 + CUDA 12.6 torch wheel (NOT the x86 default):
#   uv venv .venv
#   uv pip install --python .venv/bin/python -e .
#   # torch is pinned to the cu126 index via pyproject.toml [tool.uv.sources],
#   # so 'uv pip install -e .' picks the right wheel automatically.
#   # sanity check (must print True  12.6):
#   .venv/bin/python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
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
# cudatoolkit/24.11_12.6 provides the full CUDA 12.6 stack (runtime, nvcc,
# ptxas) and supersedes the older cuda/12.6 module — the two conflict on
# Isambard AI.  Loading cudatoolkit unconditionally avoids the conflict when it
# is already inherited from the login shell, and means a single module load
# covers both eager and torch.compile paths.
module load cudatoolkit/24.11_12.6

PYTHON=".venv/bin/python"
if [[ ! -x "${PYTHON}" ]]; then
	echo "ERROR: ${PYTHON} not found. Create the venv and editable-install first:" >&2
	echo "  uv venv .venv && uv pip install --python .venv/bin/python -e ." >&2
	exit 1
fi

# CPU threads still matter for the host-side ops (met decode, numpy). The GPU
# carries the per-step physics, so a modest pool is plenty; match the CPU-script
# default. (No effect on the CUDA kernels themselves.)
export GLIDE_NUM_THREADS="${GLIDE_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-16}}"
export OMP_NUM_THREADS="${GLIDE_NUM_THREADS}"
export MKL_NUM_THREADS="${GLIDE_NUM_THREADS}"
export OPENBLAS_NUM_THREADS="${GLIDE_NUM_THREADS}"

# torch.compile / CUDA-graph capture of the per-step physics. On CUDA the runtime
# uses the static-shape per-step path (full particle buffer, mask-gated, no per-step
# host sync — docs/architecture.md §5), and GLIDE_COMPILE wraps the whole fixed-count
# Hanna substep loop with `torch.compile(mode="reduce-overhead")` so it is captured
# as ONE CUDA graph and replayed per step. This is THE lever for the launch/
# orchestration-bound regime (low GPU sm% in the telemetry below): it collapses the
# per-step kernel-launch overhead, not just per-kernel cost.
#
# Inductor/Triton codegen on Isambard AI needs TWO toolchain pieces; loading only
# one makes compilation silently fall back to eager (no graph capture):
#   - cudatoolkit/24.11_12.6 : nvcc/ptxas for the Triton GPU backend
#   - gcc-native/14.2        : a C++20 host compiler for Inductor's C++ codegen.
#     The system GCC (8.x) rejects `-std=c++20` (only knows `-std=c++2a`), so the
#     precompiled-header build fails and compilation never happens. Inductor
#     honours $CC/$CXX, so we point them at the freshly-loaded GCC 14.
#
# It can't hard-fail a run (eager fallback is on); the FIRST step pays a one-time
# compile cost (seconds–tens of seconds). Set GLIDE_COMPILE=0 at submit to skip it.
#
# WHAT TO CHECK after a run (this is the phase-4 measurement — report these back):
#   - the .out log shows "torch.compile(mode='reduce-overhead') enabled" and NO
#     "WON'T CONVERT" warnings (⇒ the loop captured rather than fell back);
#   - the end-of-run sm% summary climbs well above the ~0–37% launch-bound range;
#   - per-batch wall time drops.
# TUNE: on the graph path `turbulence.max_substeps` is the FIXED per-step iteration
# count (the loop is unrolled to it), so set it to the smallest physically-converged
# value (≈15–25), not the default 50 — it is now a direct per-step cost.
export GLIDE_COMPILE="${GLIDE_COMPILE:-1}"
if [[ "${GLIDE_COMPILE}" != "0" ]]; then
	module load gcc-native/14.2
	export CC=gcc
	export CXX=g++
	echo "compile:  ON   CC=$(command -v gcc)  CXX=$(command -v g++)"
else
	echo "compile:  OFF  (GLIDE_COMPILE=0)"
fi

# Prefer PyTorch's bundled NCCL over the system one loaded by cudatoolkit.
# MUST appear after ALL module loads: modules prepend their lib paths, so if
# this is applied earlier a subsequent module load re-seats the older NCCL.
# cudatoolkit ships a libnccl.so that predates ncclCommResume (added in 2.19);
# torch's bundled copy has it and must win the dynamic-linker search.
TORCH_LIB="$("${PYTHON}" -c 'import os, torch; print(os.path.dirname(torch.__file__))')/lib"
export LD_LIBRARY_PATH="${TORCH_LIB}:${LD_LIBRARY_PATH:-}"

# Fail fast before the long preflight if the venv torch can't see the GPU.
if ! "${PYTHON}" -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then
	echo "ERROR: torch.cuda.is_available() is False. Check that:" >&2
	echo "  - the venv has an aarch64 CUDA 12.6 torch wheel (see PREREQUISITES)," >&2
	echo "  - torch.__version__ ends in '+cu126' (run: ${PYTHON} -c 'import torch; print(torch.version.cuda)')." >&2
	exit 1
fi

# PROFILING PASS (when sm% stays low and you need to know WHY the GPU idles):
# submit with GLIDE_PROFILE=1 — the runtime captures a window of cursor-loop steps
# (default: 20 steps after 5 warmup) with torch.profiler, prints a CPU/GPU summary
# (GPU-busy %, host syncs, and top ops with their CALL COUNTS) and writes a Chrome
# trace, then exits early (no full run). Open the trace at https://ui.perfetto.dev,
# or just paste the printed summary. Knobs: GLIDE_PROFILE_STEPS / _WARMUP / _TRACE;
# GLIDE_PROFILE_CONTINUE=1 to finish the run instead of exiting. Best on the smoke
# config:  GLIDE_PROFILE=1 sbatch scripts/run_periodic_cuda.slurm configs/smoke_mhd_single_release.yaml
#
# WHOLE-RUN PHASE TIMERS (when the per-step capture looks fast but the JOB is slow —
# i.e. the GPU bursts then idles): submit with GLIDE_PHASE_TIMERS=1. Unlike GLIDE_PROFILE
# (which samples ~20 steps inside ONE met window), this accumulates wall time across EVERY
# window/batch and prints a phase breakdown (met_fetch / step / gridder / advect / convection
# + residual) at the end, AND logs a running breakdown every 25 met fetches — so even a job
# that TIMES OUT leaves the split in the .out log. The met_fetch bucket is the GPU-idle time
# spent on synchronous met I/O. Compare met_fetch% to wall; check the fetch COUNT vs the
# unique hours your run spans (re-fetch thrash → bump memory.met_cache_max_hours in the
# config). Knobs: GLIDE_PHASE_TIMERS_EVERY (log cadence); GLIDE_PHASE_TIMERS_SYNC=1 to
# cuda-sync each phase for clean step-vs-gridder attribution (serialises — perturbs totals).
# Leave GLIDE_PROFILE OFF for this (it exits early). Runs a normal full job.

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

# --- GPU telemetry (background, auto-stopped) ------------------------------
# Sample GPU utilisation (sm%) + memory every 10 s to a sidecar log for the whole
# run — nothing to launch by hand. This is the key diagnostic for whether the
# GPU is actually busy: sustained sm% > ~70 means compute-bound (near the limit);
# sm% persistently low means launch/sync-bound (try GLIDE_COMPILE=1 below). The
# sampler is killed automatically when the job exits (success, failure, or
# timeout) via the EXIT trap.
GPU_LOG="slurm_logs/${SLURM_JOB_NAME:-glide}-${SLURM_JOB_ID:-local}.gpu.log"
if command -v nvidia-smi >/dev/null 2>&1; then
	nvidia-smi dmon -s um -d 10 -o DT > "${GPU_LOG}" 2>&1 &
	DMON_PID=$!
	trap 'kill "${DMON_PID}" 2>/dev/null || true' EXIT
	echo "gpu telemetry -> ${GPU_LOG} (nvidia-smi dmon, 10 s cadence)"
fi

# Direct invocation (not srun) — GLIDE is a single-process, non-MPI job. srun
# wraps the launcher in SLURM's PMI/PMIx machinery which can fail with
# "Unspecified error" on some clusters even for ntasks=1.
"${PYTHON}" -m lpdm.main \
	--config "${CONFIG}" \
	--device "${DEVICE}" \
	--output-uri "${OUTPUT_URI}"

echo "=== GLIDE run complete: ${OUTPUT_URI} ==="
if [[ -f "${GPU_LOG:-}" ]]; then
	echo "--- GPU utilisation summary (sm% column) ---"
	# Mean/Max of the sm% column (3rd field on data rows; skip the two header lines).
	awk 'NR>2 && $3 ~ /^[0-9]+$/ {n++; s+=$3; if ($3>m) m=$3} END {if (n) printf "  samples=%d  mean_sm=%.1f%%  max_sm=%d%%\n", n, s/n, m}' "${GPU_LOG}" || true
fi

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
