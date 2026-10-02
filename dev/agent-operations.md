# Running agent-driven development

_For the human directing the work. The agents read
[dev/agent/INSTRUCTIONS.md](agent/INSTRUCTIONS.md); this page is what you do
before, during and after their sessions. Written 2026-09-16, revised
2026-10-02._

**Contents**

1. [The model of work](#1-the-model-of-work)
2. [Phase 0: what to prepare](#2-phase-0-what-to-prepare)
3. [Running sessions on Isambard AI](#3-running-sessions-on-isambard-ai)
4. [Choosing the model per task](#4-choosing-the-model-per-task)
5. [Reviewing and merging](#5-reviewing-and-merging)
6. [Escalations and interventions](#6-escalations-and-interventions)
7. [Cadence](#7-cadence)
8. [Failure modes to watch for](#8-failure-modes-to-watch-for)

---

## 1. The model of work

- The plan is [dev/roadmap.md](roadmap.md). The tasks are
  [dev/agent/QUEUE.md](agent/QUEUE.md), in two queues: **A** (numerics-touching
  and validation, sequential, strongest model) and **B** (infrastructure,
  parallel, cheaper model). The GATES repository has its own queue in its code
  review document.
- **One piece of work per session.** A session starts cold, works out from the
  repository what to do next, does it on its own branch, tests on CPU and GPU,
  opens a PR and stops. The next session starts from the repository, not from
  the previous conversation. This bounds the damage any one session can do and
  survives allocations ending.
- **State lives in the repository.** Each task has a progress file,
  `dev/agent/progress/<ID>.md`, which the agent creates on its branch and pushes
  at once as its claim. The file reaches `main` only when you merge the PR, so
  a file on `main` means the task is merged. An open PR or an `agent/<ID>-*`
  branch means it is in flight.
- **PROGRESS.md is yours.** [dev/agent/PROGRESS.md](agent/PROGRESS.md) holds
  the current state, the data inventory and your resume requests. Agents read it
  and never edit it, so pull requests never conflict on it.
- **Branch per task, PR per task, human merges.** Agents never commit to
  `main`. CI runs the CPU suite on every PR. A fresh agent session reviews each
  PR before you do.
- **Stop conditions are written down** in the instructions. The agent halts
  and asks on physics changes outside scope, output-contract changes,
  validation-threshold changes, data deletion, met downloads, and unexpected
  GPU parity failures. Everything else it decides alone.
- **Environment-specific values stay out of the repository.** The SLURM account,
  data paths and scratch paths live in an untracked
  `dev/agent/isambard/env.local.sh`. Agents refer to them by variable name.

---

## 2. Phase 0: what to prepare

Work through this once. The queue assumes all of it.

### Repository

- [ ] Review and merge the plan PR.
- [ ] On Isambard, copy `dev/agent/isambard/env.local.sh.example` to
      `env.local.sh` in the same directory and fill it in. It is gitignored.
- [ ] Fill the data inventory in `dev/agent/PROGRESS.md`, with paths relative to
      `$GLIDE_DATA`.

### Environment on Isambard AI

- [ ] A working venv in the repo checkout on the shared filesystem, built per
      the README "Running on GPU" section (`cudatoolkit/24.11_12.6`, aarch64
      cu126 torch). Confirm `torch.cuda.is_available()` inside a GPU job.
- [ ] `gcc-native/14.2` loads and `GLIDE_COMPILE=1` produces a captured graph
      (no "WON'T CONVERT" in a smoke-test log).
- [ ] Claude Code installed for linux-arm64 and runnable from a compute node.
- [ ] Outbound HTTPS to the Anthropic API from a **compute** node. The driver's
      pre-flight checks this, so a short test allocation is enough.
- [ ] `git push` to GitHub and `gh pr create` work from a compute node
      (deploy key or token with repo scope; `gh auth status`).
- [ ] `GLIDE_SCRATCH` points at fast project scratch, not home; the uv and
      Triton caches go there.
- [ ] Outbound HTTPS to GCS from a compute node only if you want tasks to read
      ARCO ERA5 directly. Otherwise download every cube yourself in advance.

### Data inventory

- [ ] Every local ERA5 cube listed in PROGRESS.md, with domain, period, level
      type, cadence and size.
- [ ] The reference multi-site configuration
      (`configs/example_multisite_january.yaml` or its successor) points at a
      cube that exists.
- [ ] A small cube for smoke tests and a medium cube for benchmarks, both local.
- [ ] Decide which additional cubes you will download before the tasks that
      need them: a model-level cube (existing item 1), an ETEX Oct–Nov 1994
      European cube (Tier 3), a UK cube (v1.x).

### Golden fixtures and benchmarks

- [ ] Task A0 generates these; you only choose the configurations. The defaults
      in its spec are the smoke config on CPU and the reference multi-site config
      on GPU.

### CI and repository settings

- [ ] GitHub Actions runs `pytest -q` on every PR (CPU). Branch protection on
      `main`: PR required, CI green required, no force push.
- [ ] Optional: the nightly GPU parity job from task B8, posting results to a
      pinned issue.

### Reference runs that need a person

- [ ] Start the FLEXPART v11 on ERA5 reference runs for the case library now,
      on the current physics. They are the critical path for Tier 2.
- [ ] Obtain ETEX release and station data; confirm ERA5 1994 download.
- [ ] Locate the NAME footprints, EDGAR maps and radon data on the group
      archive, and set their variables in `env.local.sh`. They are restricted
      and never enter the repository.

### GATES

- [ ] Merge the GATES code review PR. Copy `dev/agent/` into the GATES repo with
      paths adjusted; its queue is §5 of the review document.

---

## 3. Running sessions on Isambard AI

Queue times are long, the login node is policed, and you have GPU hours. So the
agent runs **inside a long GPU allocation**, and the login node only submits
the job and holds your ssh and tmux.

**Submitting.** From the login node:

```bash
dev/agent/isambard/submit.sh A
```

An optional second argument sets the walltime. The script reads the account,
partition and paths from `env.local.sh` and submits
`agent_driver.slurm`. The job loads modules, runs pre-flight checks, and starts
the driver loop in a dedicated tmux server on the compute node.

**The driver loop.** Each iteration checks that the working tree has no
uncommitted changes (it stops rather than discard anything), syncs `main`, and
starts a fresh session. The session does one of three things:

- delivers one piece of work as a PR, and the loop continues;
- prints `WAITING`, because the next task depends on a PR you have not merged.
  The loop sleeps (`GLIDE_AGENT_WAIT_SECONDS`, default 30 minutes) and tries
  again. Each check is a short session, so raise the wait or stop the loop
  before a long absence;
- prints `NO_TASK`, because every task in the queue is merged. The loop exits.

Queue A is sequential, so in practice it delivers a task and then waits for you
to merge it. Queue B tasks are mostly independent, so it keeps going.

**Attaching.** The job log prints the node and the tmux socket name. Then:

```bash
ssh <node>
tmux -L glide-<jobid> attach
```

`touch dev/agent/STOP_AFTER_TASK` ends the loop after the current session.
Ctrl-C in the pane aborts at once.

**Restarting.** When the allocation ends, resubmit. The repository is the only
state. Test a cold restart once deliberately before relying on it.

**First runs: supervised.**

1. Submit Queue A and stay attached while it does A00. Review and merge the PR
   yourself. This shows whether the instructions, the claim, the progress file
   and the PR template work in practice, on a small task.
2. Do the same for A0, since every later task is checked against its fixtures.
3. Then leave Queue A unattended. Start Queue B in a second allocation once
   Queue A runs smoothly.

**Releasing a stuck claim.** If you close a PR without merging and want the
task redone from scratch, delete its `agent/<ID>-*` branch. The task becomes
available again.

---

## 4. Choosing the model per task

The rule: match model cost to the cost of an undetected error.

| Queue | Tasks | Model | Why |
| --- | --- | --- | --- |
| A | Everything before the physics freeze; validation analysis; all PR reviews | Fable (`claude-fable-5-1`), high effort | failure modes are silent: the code runs, tests pass, the physics has changed |
| B | Met cache, sparse output, batching, packaging, docs, figure tooling | Sonnet (`claude-sonnet-5`) | well-specified, cheap to check |
| either | Column and satellite releases, composite met source | Opus or Fable | design-heavy but not physics-changing |

Set `GLIDE_AGENT_MODEL_A` and `GLIDE_AGENT_MODEL_B` in `env.local.sh`. If in
doubt, use the stronger model; the marginal cost is small against a bad merge.

---

## 5. Reviewing and merging

1. When a PR appears, start a **fresh** session with `/code-review` on the PR
   number, using the strongest model. The author session is blind to its own
   errors; a cold reviewer is not.
2. For numerics PRs (Queue A), check in person, in this order:
   - the golden comparison output in the PR description: which fixtures,
     which tolerance, whether it passed;
   - CPU/GPU parity (static vs dynamic path) on the smoke config;
   - no test tolerance was loosened and no test was skipped or deleted;
   - the docs page and STATUS were updated; a decision record exists if the
     task said one was required;
   - the benchmark number moved in the direction claimed.
3. For infrastructure PRs, check the acceptance criteria from the task spec
   and the storage or throughput numbers reported.
4. Merge with squash, keeping the task ID in the title, and delete the branch.
   The task's progress file arrives on `main` with the merge, which is what
   marks it merged. Update "Current state" in PROGRESS.md from its hand-off
   notes if needed.
5. To ask for changes, comment on the PR **and** add a resume request to
   PROGRESS.md on `main`, for example `R4 — A2: address the review comments on
   the bandwidth default`. The next session on that queue acts on it before
   starting any new task. Sessions do not see PR comments until a resume
   request points them there.

---

## 6. Escalations and interventions

**BLOCKED.** When an agent hits a stop condition it sets its progress file to
`BLOCKED`, pushes the branch, and opens a draft PR titled
`<ID>: BLOCKED — <question>`, or comments on the existing PR. Draft PRs titled
BLOCKED are your inbox (`gh pr list --draft`). Answer by adding a resume
request with the answer, for example
`R5 — A9: FLEXPART source is now at $GLIDE_FLEXPART_SRC; use the f2py route`.
The agent resumes on the next session, completes the task and marks the PR
ready.

**HANDOFF.** When a session runs low on context mid-task it writes hand-off
notes, sets status `HANDOFF` and stops. The next session continues
automatically; you need do nothing.

**Intervene directly** (attach to tmux) when a session is looping, when a GPU
run it launched has gone far past its stated estimate, or when you see it
heading into a stop-condition area without stopping.

---

## 7. Cadence

- **Daily, 20 minutes:** review and merge PRs; answer BLOCKED drafts with
  resume requests; update "Current state".
- **Weekly:** harvest `proposed follow-ups` from the week's merged progress
  files into QUEUE.md; reprioritise; update `dev/roadmap.md` if scope moved;
  check the benchmark and storage numbers against the roadmap targets; confirm
  the FLEXPART and ETEX reference work is progressing; delete resume requests
  for merged tasks.
- **At the freeze:** you and the external reviewer sign the physics audit;
  tag; update STATUS.

---

## 8. Failure modes to watch for

- **Silent physics change.** Numbers still look plausible, tests pass because
  tolerances are wide. Defence: the golden fixtures, the well-mixed gates, and
  the rule that tolerances are never loosened in a task that changes physics.
- **Scope creep.** A task that "while it was there" refactored something else.
  Defence: the task's "out of scope" list, and reviews that reject unrelated
  diffs.
- **Duplicate work.** Two sessions on one task. Defence: the claim is a pushed
  branch, checked before any session starts work.
- **Stale claims.** A branch left behind after an abandoned PR keeps its task
  in flight forever. Defence: delete the branch to release it.
- **Context exhaustion mid-task.** Defence: progress files written early, tasks
  sized to hours, and the HANDOFF status.
- **Test suite drift.** Skipped or xfailed tests accumulating. Defence: CI
  reports the skip count; reviews check it.
- **Runaway compute.** A convergence sweep launched at ten times the intended
  size. Defence: task specs state expected runtimes; the agent must stop and
  report if a run exceeds twice the estimate.
- **Leaked environment details.** Account codes or paths committed. Defence:
  `env.local.sh` is gitignored and agents refer to variable names only.
- **Documentation lag.** Code merged without docs and STATUS updates. Defence:
  it is in the acceptance criteria of every task, and reviews check it.
