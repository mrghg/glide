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
# If a session fails (for example the subscription's usage limit is reached
# mid-task), the loop saves the session's uncommitted work as a WIP commit on
# the task branch, pushes it, and marks the task HANDOFF so the next session
# resumes it. It then sends one tiny test prompt: if the API does not answer,
# the loop exits and the allocation is released rather than held idle until the
# limit resets. Two failed sessions in a row also end the loop.
#
# Every session is logged as one JSON line in
# $GLIDE_AGENT_OUTPUTS/session_log.jsonl.
#
# Controls (from another shell on the node):
#   touch dev/agent/STOP_AFTER_TASK   # finish the current session, then exit
#   Ctrl-C in the tmux pane           # abort
#
# Each WAITING check is a short session and costs a little; raise
# GLIDE_AGENT_WAIT_SECONDS, or stop the loop, before a long absence.
# =============================================================================
set -uo pipefail
# Only inside a job allocation: never on a login node.
[[ -n "${SLURM_JOB_ID:-}" ]] || { echo "SLURM_JOB_ID is unset: this is not a compute-node allocation; refusing to run" >&2; exit 1; }
QUEUE="${1:?queue letter required}"
MODEL="${GLIDE_AGENT_MODEL:?GLIDE_AGENT_MODEL not set}"
MAX_TASKS="${GLIDE_AGENT_MAX_TASKS:-20}"
WAIT_S="${GLIDE_AGENT_WAIT_SECONDS:-1800}"
STOP_FILE="dev/agent/STOP_AFTER_TASK"
SESSION_LOG="${GLIDE_AGENT_OUTPUTS:?GLIDE_AGENT_OUTPUTS not set}/session_log.jsonl"
PROMPT="Follow dev/agent/INSTRUCTIONS.md exactly. Your queue is ${QUEUE}. Select work with the \
procedure in section 1 (resume requests and hand-offs first, then the first available task whose \
dependencies are merged). If that procedure says so, print a line containing only WAITING or only \
NO_TASK and stop. Otherwise deliver that one piece of work per section 2 and stop."

# One JSON line per session. Values are simple strings and integers, so no
# escaping is needed beyond what the fields themselves guarantee.
log_session() { # start end rc outcome branch
  printf '{"job":"%s","queue":"%s","model":"%s","start":"%s","end":"%s","rc":%d,"outcome":"%s","branch":"%s"}\n' \
    "${SLURM_JOB_ID}" "${QUEUE}" "${MODEL}" "$1" "$2" "$3" "$4" "$5" >> "${SESSION_LOG}"
}

# After a failed session on a task branch: commit any uncommitted work as WIP
# (pre-commit hooks still run), mark the task HANDOFF so the next session
# resumes it, and push. A task already at PR OPEN, BLOCKED or HANDOFF with a
# clean tree is left alone. Returns non-zero if the state could not be saved.
save_interrupted() { # rc
  local rc="$1" branch id pf dirty status msg
  branch="$(git branch --show-current)"
  dirty="$(git status --porcelain)"
  if [[ "${branch}" != agent/* ]]; then
    [[ -z "${dirty}" ]] && return 0
    echo "uncommitted changes on ${branch:-a detached HEAD}, not a task branch; leaving them for the human" >&2
    return 1
  fi
  id="${branch#agent/}"; id="${id%%-*}"
  pf="dev/agent/progress/${id}.md"
  if [[ ! -f "${pf}" ]]; then
    [[ -z "${dirty}" ]] && return 0
    echo "no progress file ${pf} on ${branch}; leaving the changes for the human" >&2
    return 1
  fi
  status="$(grep -m1 '^- status:' "${pf}" || true)"
  if [[ -z "${dirty}" && "${status}" != *CLAIMED* ]]; then
    git push -q origin HEAD 2>/dev/null || true
    return 0
  fi
  sed -i.bak -E 's/^- status: .*/- status: HANDOFF/' "${pf}" && rm -f "${pf}.bak"
  if [[ -n "${dirty}" ]]; then
    msg="wip(${id}): session interrupted (driver, exit ${rc})"
    printf -- '- interrupted: %s, session exit code %s. The driver committed the uncommitted work as a WIP commit; it may be incomplete or fail tests. Review it before continuing.\n' \
      "$(date -u +%FT%TZ)" "${rc}" >> "${pf}"
  else
    msg="chore(${id}): mark HANDOFF after interrupted session (driver, exit ${rc})"
    printf -- '- interrupted: %s, session exit code %s. No uncommitted work; continue from the last commit and the notes above.\n' \
      "$(date -u +%FT%TZ)" "${rc}" >> "${pf}"
  fi
  git add -A || return 1
  if ! git commit -q -m "${msg}"; then
    echo "commit failed (pre-commit hooks?); leaving the changes for the human" >&2
    return 1
  fi
  git push -q origin HEAD || { echo "push failed" >&2; return 1; }
  echo "saved the interrupted session on ${branch} and marked ${id} HANDOFF"
}

delivered=0
failures=0
while (( delivered < MAX_TASKS )); do
  if [[ -f "${STOP_FILE}" ]]; then echo "stop file present; exiting"; rm -f "${STOP_FILE}"; exit 0; fi
  if ! git diff --quiet || ! git diff --cached --quiet; then
    echo "uncommitted changes to tracked files; stopping so nothing is lost" >&2; exit 1
  fi
  if ! { git fetch -q --prune origin && git switch -q main && git pull -q --ff-only; }; then
    echo "could not sync main; stopping" >&2; exit 1
  fi

  LOG="$(mktemp)"
  START="$(date -u +%FT%TZ)"
  echo "=== session start ${START} queue=${QUEUE} model=${MODEL} ==="
  claude -p "${PROMPT}" --model "${MODEL}" --dangerously-skip-permissions 2>&1 | tee "${LOG}"
  RC=${PIPESTATUS[0]}
  END="$(date -u +%FT%TZ)"
  BRANCH="$(git branch --show-current)"
  echo "=== session end ${END} rc=${RC} ==="

  if (( RC == 0 )) && grep -qxE '[[:space:]]*NO_TASK[[:space:]]*' "${LOG}"; then
    rm -f "${LOG}"; log_session "${START}" "${END}" "${RC}" no_task "${BRANCH}"
    echo "queue ${QUEUE} complete"; exit 0
  fi
  if (( RC == 0 )) && grep -qxE '[[:space:]]*WAITING[[:space:]]*' "${LOG}"; then
    rm -f "${LOG}"; log_session "${START}" "${END}" "${RC}" waiting "${BRANCH}"
    failures=0; echo "waiting ${WAIT_S}s for a merge"; sleep "${WAIT_S}"; continue
  fi
  rm -f "${LOG}"

  if (( RC != 0 )); then
    log_session "${START}" "${END}" "${RC}" failed "${BRANCH}"
    failures=$((failures + 1))
    save_interrupted "${RC}" || { echo "stopping; the allocation will be released" >&2; exit 1; }
    if (( failures >= 2 )); then
      echo "two failed sessions in a row; stopping so the allocation is released" >&2; exit 1
    fi
    if ! claude -p "Reply with the single word OK." --model "${MODEL}" >/dev/null 2>&1; then
      echo "the API does not answer (usage limit or outage); stopping so the allocation is released" >&2; exit 1
    fi
    echo "the API answers, so the failure was not a usage limit; continuing"
    continue
  fi

  log_session "${START}" "${END}" "${RC}" delivered "${BRANCH}"
  failures=0
  delivered=$((delivered + 1))
done
echo "reached GLIDE_AGENT_MAX_TASKS=${MAX_TASKS}"
