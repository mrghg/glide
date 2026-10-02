#!/bin/bash
# =============================================================================
# Long-running agent allocation for GLIDE development on Isambard AI.
#
# TEMPLATE. Fill the >>> EDIT lines, then test with a short --time before
# relying on it. Submit from the repo root:
#     sbatch --time=24:00:00 dev/agent/isambard/agent_driver.slurm A
#     sbatch --time=24:00:00 dev/agent/isambard/agent_driver.slurm B
# The positional argument is the queue letter (A or B). The driver loop runs
# inside a tmux session on the compute node so you can attach:
#     ssh <node>; tmux attach -t glide-agent-<queue>
# Resubmit when the allocation ends; PROGRESS.md and QUEUE.md are the state.
# =============================================================================
#SBATCH --job-name=glide-agent
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
# #SBATCH --account=<<PLACEHOLDER>>      # >>> EDIT
# #SBATCH --partition=<<PLACEHOLDER>>    # >>> EDIT

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
QUEUE="${1:-A}"

module load cudatoolkit/24.11_12.6
module load gcc-native/14.2
export CC=gcc CXX=g++
export UV_CACHE_DIR="<<PLACEHOLDER: scratch>>/uv"          # >>> EDIT
export TRITON_CACHE_DIR="<<PLACEHOLDER: scratch>>/triton"  # >>> EDIT
export GLIDE_DATA="<<PLACEHOLDER: cubes root>>"            # >>> EDIT
export GLIDE_AGENT_OUTPUTS="<<PLACEHOLDER: outputs root>>" # >>> EDIT

# Model per queue. IDs: claude-fable-5-1, claude-opus-5, claude-sonnet-5.
case "${QUEUE}" in
  A) export GLIDE_AGENT_MODEL="claude-fable-5-1" ;;
  B) export GLIDE_AGENT_MODEL="claude-sonnet-5" ;;
  *) echo "unknown queue ${QUEUE}" >&2; exit 1 ;;
esac

echo "node: $(hostname)  queue: ${QUEUE}  model: ${GLIDE_AGENT_MODEL}  job: ${SLURM_JOB_ID}"

# Pre-flight: GPU, API egress, git.
.venv/bin/python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)"
claude -p "Reply with the single word OK." --model "${GLIDE_AGENT_MODEL}" >/dev/null
gh auth status >/dev/null

# Run the driver loop inside tmux so a human can attach; keep the job alive
# until the loop exits.
SESSION="glide-agent-${QUEUE}"
tmux new-session -d -s "${SESSION}" \
  "bash dev/agent/isambard/run_task.sh ${QUEUE} 2>&1 | tee -a slurm_logs/agent-${QUEUE}-${SLURM_JOB_ID}.log"
while tmux has-session -t "${SESSION}" 2>/dev/null; do sleep 60; done
echo "driver loop finished"
