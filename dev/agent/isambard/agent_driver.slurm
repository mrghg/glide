#!/bin/bash
# =============================================================================
# Long-running agent allocation for GLIDE development on Isambard AI.
#
# Submit with dev/agent/isambard/submit.sh <A|B> [walltime], which supplies the
# account, partition and walltime from env.local.sh. UNTESTED TEMPLATE: run it
# once with a short walltime before relying on it.
#
# The driver loop runs in a dedicated tmux server on the compute node, so it
# inherits the modules and variables set here. To watch or intervene:
#     ssh <node>
#     tmux -L glide-<jobid> attach
# Resubmit when the allocation ends; the repository is the only state.
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

module load cudatoolkit/24.11_12.6
module load gcc-native/14.2
export CC=gcc CXX=g++
export UV_CACHE_DIR="${GLIDE_SCRATCH}/uv"
export TRITON_CACHE_DIR="${GLIDE_SCRATCH}/triton"
mkdir -p "${UV_CACHE_DIR}" "${TRITON_CACHE_DIR}" "${GLIDE_AGENT_OUTPUTS}" slurm_logs

case "${QUEUE}" in
  A) export GLIDE_AGENT_MODEL="${GLIDE_AGENT_MODEL_A:-claude-fable-5-1}" ;;
  B) export GLIDE_AGENT_MODEL="${GLIDE_AGENT_MODEL_B:-claude-sonnet-5}" ;;
  *) echo "unknown queue ${QUEUE}" >&2; exit 1 ;;
esac

SOCKET="glide-${SLURM_JOB_ID:-local}"
echo "node: $(hostname)  queue: ${QUEUE}  model: ${GLIDE_AGENT_MODEL}  job: ${SLURM_JOB_ID:-none}"
echo "attach with: ssh $(hostname) then tmux -L ${SOCKET} attach"

# Pre-flight: GPU, API egress, GitHub.
.venv/bin/python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)"
claude -p "Reply with the single word OK." --model "${GLIDE_AGENT_MODEL}" >/dev/null
gh auth status >/dev/null

# A dedicated tmux server (-L) is started fresh by this job, so it inherits this
# shell's environment: the loaded modules and everything from env.local.sh.
tmux -L "${SOCKET}" new-session -d -s "agent-${QUEUE}" \
  "bash dev/agent/isambard/run_task.sh ${QUEUE} 2>&1 | tee -a slurm_logs/agent-${QUEUE}-${SLURM_JOB_ID:-local}.log"
while tmux -L "${SOCKET}" has-session -t "agent-${QUEUE}" 2>/dev/null; do sleep 60; done
echo "driver loop finished"
