#!/bin/bash
# Submit a long-running agent allocation. Run by the human on the login node,
# from anywhere. Agents never run this or any other job-submission command.
#     dev/agent/isambard/submit.sh A              # queue A, unattended loop
#     dev/agent/isambard/submit.sh A shell        # queue A, a shell for a watched session
#     dev/agent/isambard/submit.sh B loop 12:00:00
# The scheduler chooses the account and partition. Paths come from env.local.sh
# next to this script.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ENV_FILE="${HERE}/env.local.sh"
[[ -f "${ENV_FILE}" ]] || { echo "missing ${ENV_FILE}; copy env.local.sh.example and fill it" >&2; exit 1; }
# shellcheck source=/dev/null
source "${ENV_FILE}"
QUEUE="${1:?usage: submit.sh <A|B> [loop|shell] [walltime]}"
MODE="${2:-loop}"
TIME="${3:-${GLIDE_AGENT_WALLTIME:-24:00:00}}"
case "${MODE}" in loop|shell) ;; *) echo "mode must be loop or shell" >&2; exit 1 ;; esac
[[ -n "${GLIDE_REPO:-}" ]] || { echo "GLIDE_REPO is not set in ${ENV_FILE}" >&2; exit 1; }
cd "${GLIDE_REPO}"
mkdir -p slurm_logs
sbatch --time="${TIME}" --job-name="glide-agent-${QUEUE}-${MODE}" \
  dev/agent/isambard/agent_driver.slurm "${QUEUE}" "${MODE}"
