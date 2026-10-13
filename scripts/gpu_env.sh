# shellcheck shell=bash
# =============================================================================
# GPU runtime environment for GLIDE on Isambard AI (GH200, aarch64).
#
# SOURCE this (don't execute it), from an interactive GPU shell or a SLURM job:
#
#     source scripts/gpu_env.sh
#
# Shared by scripts/run_periodic_cuda.slurm and the README's interactive recipe
# so the two can't drift. Safe to source more than once. Never calls `exit` or
# `set -e`: it returns non-zero on error, leaving an interactive shell intact.
#
# Respects GLIDE_COMPILE if already set (GLIDE_COMPILE=0 skips the GCC load);
# otherwise defaults it to 1, as the code itself defaults to eager.
# =============================================================================

_glide_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GLIDE_PYTHON="${_glide_root}/.venv/bin/python"
unset _glide_root

if [[ ! -x "${GLIDE_PYTHON}" ]]; then
	echo "ERROR: ${GLIDE_PYTHON} not found. Create the venv and editable-install first:" >&2
	echo "  uv venv .venv && uv pip install --python .venv/bin/python -e ." >&2
	return 1
fi
export GLIDE_PYTHON

# 1. CUDA. cudatoolkit/24.11_12.6 provides the full CUDA 12.6 stack (runtime,
#    nvcc, ptxas) and supersedes the older cuda/12.6 module; the two conflict,
#    so this must be the only CUDA module loaded.
module load cudatoolkit/24.11_12.6 || return 1

# 2. torch.compile (CUDA-graph capture of the per-step physics). Inductor's C++
#    codegen needs a C++20 host compiler; the system GCC 8 rejects -std=c++20,
#    and compilation then silently falls back to eager. Inductor honours
#    $CC/$CXX, so point them at GCC 14.
export GLIDE_COMPILE="${GLIDE_COMPILE:-1}"
if [[ "${GLIDE_COMPILE}" != "0" ]]; then
	module load gcc-native/14.2 || return 1
	export CC=gcc
	export CXX=g++
fi

# 3. Prefer PyTorch's bundled NCCL over cudatoolkit's. MUST come after ALL
#    module loads: modules prepend their lib paths, so a later load re-seats the
#    older NCCL, which predates ncclCommResume (added in 2.19). Any existing
#    copy is dropped first, so re-sourcing moves it to the front without
#    stacking duplicates.
_glide_torch_lib="$("${GLIDE_PYTHON}" -c 'import os, torch; print(os.path.dirname(torch.__file__))')/lib" || return 1
_glide_ld=""
IFS=: read -ra _glide_ld_parts <<<"${LD_LIBRARY_PATH:-}"
for _glide_p in "${_glide_ld_parts[@]}"; do
	if [[ -n "${_glide_p}" && "${_glide_p}" != "${_glide_torch_lib}" ]]; then
		_glide_ld+=":${_glide_p}"
	fi
done
export LD_LIBRARY_PATH="${_glide_torch_lib}${_glide_ld}"
unset _glide_torch_lib _glide_ld _glide_ld_parts _glide_p

# CUDA caching allocator: expandable segments cut reserved-but-unused HBM from
# the per-batch alloc/free churn of the particle + footprint tensors.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [[ "${GLIDE_COMPILE}" != "0" ]]; then
	echo "gpu_env: cudatoolkit 12.6, compile ON (CC=$(command -v gcc)), torch NCCL first"
else
	echo "gpu_env: cudatoolkit 12.6, compile OFF (GLIDE_COMPILE=0), torch NCCL first"
fi
