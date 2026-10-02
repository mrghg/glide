#!/bin/bash
# =============================================================================
# Driver loop: run one fresh Claude Code session per task until the queue has
# no OPEN task whose dependencies are MERGED, or until a stop file appears.
#
# TEMPLATE. Assumes: repo root is the working directory; .venv exists; the
# `claude` and `gh` CLIs are on PATH and authenticated; GLIDE_AGENT_MODEL is set.
#
# Controls while running (from another shell on the node):
#   touch dev/agent/STOP_AFTER_TASK   # finish the current task, then exit
#   Ctrl-C in the tmux pane           # abort the current session; loop exits
#
# Task selection: this script does NOT parse QUEUE/PROGRESS itself. It hands
# the queue letter to the agent, whose instructions say how to pick the next
# OPEN task. A session that finds nothing to do prints NO_TASK and the loop
# exits. Keep it this simple until it has run cleanly a few times.
# =============================================================================
set -uo pipefail
QUEUE="${1:?queue letter required}"
MODEL="${GLIDE_AGENT_MODEL:?GLIDE_AGENT_MODEL not set}"
MAX_TASKS="${GLIDE_AGENT_MAX_TASKS:-20}"
STOP_FILE="dev/agent/STOP_AFTER_TASK"

for i in $(seq 1 "${MAX_TASKS}"); do
  git switch -q main && git pull -q --ff-only
  if [[ -f "${STOP_FILE}" ]]; then echo "stop file present; exiting"; rm -f "${STOP_FILE}"; exit 0; fi

  PROMPT="Follow dev/agent/INSTRUCTIONS.md exactly. Work queue ${QUEUE} in dev/agent/QUEUE.md: \
take the first task whose status in dev/agent/PROGRESS.md is OPEN (or absent) and whose dependencies \
are MERGED. If there is no such task, print exactly NO_TASK and stop. Otherwise deliver that one task \
as a pull request per the session protocol, then stop."

  echo "=== task ${i} start $(date -u +%FT%TZ) ==="
  OUT="$(claude -p "${PROMPT}" --model "${MODEL}" --dangerously-skip-permissions 2>&1 | tee /dev/stderr)"
  RC=$?
  echo "=== task ${i} end rc=${RC} $(date -u +%FT%TZ) ==="
  if grep -q "NO_TASK" <<<"${OUT}"; then echo "queue ${QUEUE} empty"; exit 0; fi
  if [[ ${RC} -ne 0 ]]; then echo "session exited non-zero; pausing 5 min"; sleep 300; fi
done
echo "reached MAX_TASKS=${MAX_TASKS}"
