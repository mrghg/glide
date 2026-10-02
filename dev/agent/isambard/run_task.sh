#!/bin/bash
# =============================================================================
# Driver loop: one fresh Claude Code session per piece of work.
#
# Started by agent_driver.slurm inside a dedicated tmux server, so it inherits
# the job's modules and the variables from env.local.sh. UNTESTED TEMPLATE.
#
# The session selects work itself (INSTRUCTIONS.md §1) and either delivers one
# piece of work or ends with a line containing only:
#   WAITING  - the next task depends on a PR not yet merged; sleep and retry
#   NO_TASK  - every task in the queue is merged; exit
#
# Controls (from another shell on the node):
#   touch dev/agent/STOP_AFTER_TASK   # finish the current session, then exit
#   Ctrl-C in the tmux pane           # abort
#
# Each WAITING check is a short session and costs a little; raise
# GLIDE_AGENT_WAIT_SECONDS, or stop the loop, before a long absence.
# =============================================================================
set -uo pipefail
QUEUE="${1:?queue letter required}"
MODEL="${GLIDE_AGENT_MODEL:?GLIDE_AGENT_MODEL not set}"
MAX_TASKS="${GLIDE_AGENT_MAX_TASKS:-20}"
WAIT_S="${GLIDE_AGENT_WAIT_SECONDS:-1800}"
STOP_FILE="dev/agent/STOP_AFTER_TASK"
PROMPT="Follow dev/agent/INSTRUCTIONS.md exactly. Your queue is ${QUEUE}. Select work with the \
procedure in section 1 (resume requests and hand-offs first, then the first available task whose \
dependencies are merged). If that procedure says so, print a line containing only WAITING or only \
NO_TASK and stop. Otherwise deliver that one piece of work per section 2 and stop."

delivered=0
while (( delivered < MAX_TASKS )); do
  if [[ -f "${STOP_FILE}" ]]; then echo "stop file present; exiting"; rm -f "${STOP_FILE}"; exit 0; fi
  if ! git diff --quiet || ! git diff --cached --quiet; then
    echo "uncommitted changes to tracked files; stopping so nothing is lost" >&2; exit 1
  fi
  if ! { git fetch -q --prune origin && git switch -q main && git pull -q --ff-only; }; then
    echo "could not sync main; stopping" >&2; exit 1
  fi

  LOG="$(mktemp)"
  echo "=== session start $(date -u +%FT%TZ) queue=${QUEUE} model=${MODEL} ==="
  claude -p "${PROMPT}" --model "${MODEL}" --dangerously-skip-permissions 2>&1 | tee "${LOG}"
  RC=${PIPESTATUS[0]}
  echo "=== session end $(date -u +%FT%TZ) rc=${RC} ==="

  if grep -qxE '[[:space:]]*NO_TASK[[:space:]]*' "${LOG}"; then
    rm -f "${LOG}"; echo "queue ${QUEUE} complete"; exit 0
  fi
  if grep -qxE '[[:space:]]*WAITING[[:space:]]*' "${LOG}"; then
    rm -f "${LOG}"; echo "waiting ${WAIT_S}s for a merge"; sleep "${WAIT_S}"; continue
  fi
  rm -f "${LOG}"
  delivered=$((delivered + 1))
  if (( RC != 0 )); then echo "session exited non-zero; pausing 5 min"; sleep 300; fi
done
echo "reached GLIDE_AGENT_MAX_TASKS=${MAX_TASKS}"
