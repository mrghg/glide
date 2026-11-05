# 0015 — Agent-driven development: one task per session, PR per task, human merges

**Context.** Most of the remaining work will be done by coding agents running
inside long GPU allocations on Isambard AI. The failure modes are silent
physics changes, scope creep, context exhaustion and drift between code and
documentation.

**Decision.** Work is organised as task queues (dev/agent/QUEUE.md). Durable state
lives in the repository itself: each task has a progress file under
dev/agent/progress/, created and pushed on the task branch as the claim, so a
file on main means the task is merged and an open PR or `agent/<ID>-*` branch
means it is in flight. dev/agent/PROGRESS.md is maintained by the human only
(current state, data inventory, resume requests), so PRs never conflict on it.
Environment-specific values (SLURM account, paths) live in an untracked
`env.local.sh` and are referred to by variable name. Each session delivers one task on its
own branch as a pull request and stops. Agents never commit to main. A fresh
session reviews every PR before the human merges. Numerics-touching tasks use
the strongest model and are verified against golden fixtures, static/dynamic
parity, graph-capture guards and the well-mixed gates; infrastructure tasks may
use a cheaper model. Written stop conditions define when an agent halts and
asks. The agent runs inside the allocation; the login node carries only ssh and
tmux.

**Rationale.** Bounded sessions limit the blast radius of any one mistake and
survive allocations ending. Golden fixtures and the eager CPU reference make
silent physics changes detectable. Model choice matches the cost of an
undetected error.

**Rejected alternatives.** One long interactive session (dies with the job,
loses context). Agent on the login node submitting GPU jobs (queue wait before
every test; login-node policy). Per-command permission prompts (not
long-running).

**Status.** Adopted 2026-09-16; claim and state mechanics revised 2026-10-02. Docs: dev/agent-operations.md,
dev/agent/INSTRUCTIONS.md.
