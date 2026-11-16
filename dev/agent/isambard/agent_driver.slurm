#!/bin/bash
# =============================================================================
# Long-running agent allocation for GLIDE development on Isambard AI.
#
# The human submits this with dev/agent/isambard/submit.sh <A|B> [loop|shell]
# [walltime]. The scheduler chooses the account and partition. UNTESTED
# TEMPLATE: run it once with a short walltime before relying on it.
#
# Two modes, both in a dedicated tmux server on the compute node, so they
# inherit the modules and variables set here:
#   loop   the unattended driver loop (run_task.sh), one session per task;
#   shell  a shell for a watched session: attach and start `claude` yourself.
# To attach:
#     ssh <node>
#     tmux -L glide-<jobid> attach
# Resubmit when the allocation ends; the repository is the only state.
#
# Agents must not submit, alter or cancel jobs (Isambard AI asks that agents do
# not submit jobs). This script puts refusing stand-ins for the SLURM
# job-control commands first on PATH as a guard; the instructions say the same.
# =============================================================================
#SBATCH --job-name=glide-agent
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
ENV_FILE="dev/agent/isambard/env.local.sh"
[[ -f "${ENV_FILE}" ]] || { echo "missing ${ENV_FILE}" >&2; exit 1; }
# shellcheck source=/dev/null
source "${ENV_FILE}"
for v in GLIDE_REPO GLIDE_DATA GLIDE_AGENT_OUTPUTS GLIDE_SCRATCH; do
  [[ -n "${!v:-}" ]] || { echo "${v} is not set in ${ENV_FILE}" >&2; exit 1; }
done
cd "${GLIDE_REPO}"
QUEUE="${1:-A}"
MODE="${2:-loop}"

# Modules, CC/CXX, torch's NCCL first on LD_LIBRARY_PATH and the allocator
# config: shared with run_periodic_cuda.slurm. gpu_env.sh exports
# GLIDE_COMPILE=1; unset it so tests and runs keep the code's eager default,
# and a task that wants compilation sets it explicitly. GCC 14 stays loaded.
source scripts/gpu_env.sh
unset GLIDE_COMPILE
export UV_CACHE_DIR="${GLIDE_SCRATCH}/uv"
export TRITON_CACHE_DIR="${GLIDE_SCRATCH}/triton"
mkdir -p "${UV_CACHE_DIR}" "${TRITON_CACHE_DIR}" "${GLIDE_AGENT_OUTPUTS}" slurm_logs

case "${QUEUE}" in
  A) export GLIDE_AGENT_MODEL="${GLIDE_AGENT_MODEL_A:-claude-opus-5-5}" ;;
  B) export GLIDE_AGENT_MODEL="${GLIDE_AGENT_MODEL_B:-claude-sonnet-5-5}" ;;
  *) echo "unknown queue ${QUEUE}" >&2; exit 1 ;;
esac

# Guard: refusing stand-ins for the job-control commands, first on PATH. Read-only
# queries (squeue, sinfo) stay available, so a session can check its remaining
# walltime with: squeue -j "$SLURM_JOB_ID" -h -o %L
GUARD_DIR="${GLIDE_SCRATCH}/agent-guard"
mkdir -p "${GUARD_DIR}"
for cmd in sbatch srun salloc scancel scontrol; do
  printf '#!/bin/bash\necho "%s is disabled in agent allocations: agents must not submit, alter or cancel jobs. Run work directly on this node." >&2\nexit 1\n' "${cmd}" > "${GUARD_DIR}/${cmd}"
  chmod +x "${GUARD_DIR}/${cmd}"
done
export PATH="${GUARD_DIR}:${PATH}"
export GLIDE_AGENT_GUARD_DIR="${GUARD_DIR}"

SOCKET="glide-${SLURM_JOB_ID:-local}"
echo "node: $(hostname)  queue: ${QUEUE}  mode: ${MODE}  model: ${GLIDE_AGENT_MODEL}  job: ${SLURM_JOB_ID:-none}"
echo "attach with: ssh $(hostname) then tmux -L ${SOCKET} attach"

# Pre-flight: GPU, API egress, GitHub.
.venv/bin/python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)"
claude -p "Reply with the single word OK." --model "${GLIDE_AGENT_MODEL}" >/dev/null
gh auth status >/dev/null

# A dedicated tmux server (-L) is started fresh by this job, so it inherits this
# shell's environment: the loaded modules and everything from env.local.sh.
case "${MODE}" in
  loop)
    tmux -L "${SOCKET}" new-session -d -s "agent-${QUEUE}" \
      "bash dev/agent/isambard/run_task.sh ${QUEUE} 2>&1 | tee -a slurm_logs/agent-${QUEUE}-${SLURM_JOB_ID:-local}.log"
    ;;
  shell)
    # An interactive shell that reads ~/.bashrc, then puts the guard back first
    # on PATH in case .bashrc reset it.
    RCFILE="${GLIDE_SCRATCH}/agent-shell-${SLURM_JOB_ID:-local}.rc"
    cat > "${RCFILE}" <<RC
[[ -f ~/.bashrc ]] && source ~/.bashrc
export PATH="${GUARD_DIR}:\${PATH}"
cd "${GLIDE_REPO}"
echo "GLIDE queue ${QUEUE}, watched session. Start with:"
echo "  claude --model ${GLIDE_AGENT_MODEL}"
echo "and ask it to follow dev/agent/INSTRUCTIONS.md for queue ${QUEUE}."
echo "Job control (sbatch, srun, salloc, scancel, scontrol) is disabled here; use the login node."
RC
    tmux -L "${SOCKET}" new-session -d -s "agent-${QUEUE}" "bash --rcfile ${RCFILE} -i"
    ;;
  *) echo "unknown mode ${MODE}" >&2; exit 1 ;;
esac
while tmux -L "${SOCKET}" has-session -t "agent-${QUEUE}" 2>/dev/null; do sleep 60; done
echo "session ended"
