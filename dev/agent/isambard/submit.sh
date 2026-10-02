#!/bin/bash
# Submit a long-running agent allocation. Run on the login node from anywhere:
#     dev/agent/isambard/submit.sh A            # queue A, default walltime
#     dev/agent/isambard/submit.sh B 12:00:00   # queue B, 12 hours
# Reads account, partition and paths from env.local.sh next to this script.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ENV_FILE="${HERE}/env.local.sh"
[[ -f "${ENV_FILE}" ]] || { echo "missing ${ENV_FILE}; copy env.local.sh.example and fill it" >&2; exit 1; }
# shellcheck source=/dev/null
source "${ENV_FILE}"
QUEUE="${1:?usage: submit.sh <A|B> [walltime]}"
TIME="${2:-${GLIDE_AGENT_WALLTIME:-24:00:00}}"
for v in GLIDE_SLURM_ACCOUNT GLIDE_SLURM_PARTITION GLIDE_REPO; do
  [[ -n "${!v:-}" ]] || { echo "${v} is not set in ${ENV_FILE}" >&2; exit 1; }
done
cd "${GLIDE_REPO}"
mkdir -p slurm_logs
sbatch --account="${GLIDE_SLURM_ACCOUNT}" --partition="${GLIDE_SLURM_PARTITION}" \
  --time="${TIME}" --job-name="glide-agent-${QUEUE}" \
  dev/agent/isambard/agent_driver.slurm "${QUEUE}"
