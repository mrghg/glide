# Running agent-driven development

_For the human directing the work. The agents read
[dev/agent/INSTRUCTIONS.md](agent/INSTRUCTIONS.md); this page is what you do
before, during and after their sessions. Written 2026-09-16._

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
- **One task per session.** A session starts cold, reads the durable state
  ([dev/agent/PROGRESS.md](agent/PROGRESS.md)), claims the next task, works on
  its own branch, tests on CPU and GPU, opens a PR, writes its progress entry
  and stops. The next session picks up from the file, not from the previous
  conversation. This bounds the damage any one session can do and survives
  allocations ending.
- **Branch per task, PR per task, human merges.** Agents never commit to
  `main`. CI runs the CPU suite on every PR. A fresh agent session reviews each
  PR before you do.
- **Stop conditions are written down** in the instructions. The agent halts
  and asks on physics changes outside scope, output-contract changes,
  validation-threshold changes, data deletion, met downloads, and unexpected
  GPU parity failures. Everything else it decides alone.

---

## 2. Phase 0: what to prepare

Work through this once. Everything here is either done by the documents in this
PR or is something only you can do. The queue assumes all of it.

### Repository (done by this PR, check they are merged)

- [ ] `dev/roadmap.md`, `docs/validation-plan.md`, `dev/agent/*`, decision
      records 0011–0015, pointer edits in `AGENTS.md`, `STATUS.md`,
      `README.md`, `docs/README.md`.
- [ ] Fill the `<<PLACEHOLDER>>` fields in `dev/agent/INSTRUCTIONS.md` §6 (data
      paths, account, partition) and in `dev/agent/isambard/*`.

### Environment on Isambard AI

- [ ] A working venv in the repo checkout on the shared filesystem, built per
      the README "Running on GPU" section (`cudatoolkit/24.11_12.6`, aarch64
      cu126 torch). Confirm `torch.cuda.is_available()` inside a GPU job.
- [ ] `gcc-native/14.2` loads and `GLIDE_COMPILE=1` produces a captured graph
      (no "WON'T CONVERT" in a smoke-test log).
- [ ] Claude Code installed for linux-arm64 and runnable from a compute node.
- [ ] Outbound HTTPS to the Anthropic API from a **compute** node (run
      `claude -p "say ok"` inside a short GPU job).
- [ ] Outbound HTTPS to GCS from a compute node, for the few tasks that need a
      new cube (or decide that all cubes are downloaded by you in advance).
- [ ] `git push` to GitHub and `gh pr create` work from a compute node
      (deploy key or token with repo scope; `gh auth status`).
- [ ] `uv` cache and Triton cache pointed at fast local or project scratch, not
      home, via `UV_CACHE_DIR` and `TRITON_CACHE_DIR`.

### Data inventory

- [ ] Every local ERA5 cube listed in `dev/agent/PROGRESS.md` §"Data inventory"
      with path, domain, period, level type, cadence, size.
- [ ] The reference multi-site configuration (`configs/example_multisite_january.yaml`
      or its successor) points at a cube that exists.
- [ ] A small cube for smoke tests and a medium cube for benchmarks, both local.
- [ ] Decide which additional cubes you will download yourself before the tasks
      that need them: model-level cube (existing item 1), ETEX Oct–Nov 1994
      European cube (Tier 3), UK cube (v1.x).

### Golden fixtures and benchmarks

- [ ] Task A0 in the queue generates these; you only need to choose the
      configurations. Defaults proposed there are the smoke config on CPU and
      the reference multi-site config on GPU.

### CI and repository settings

- [ ] GitHub Actions runs `pytest -q` on every PR (CPU). Branch protection on
      `main`: PR required, CI green required, no force push.
- [ ] Optional but valuable: a nightly sbatch on Isambard that runs the GPU
      parity tests and posts the result to the PROGRESS file or a GitHub issue.

### Reference runs that need a person

- [ ] Start the FLEXPART v11 on ERA5 reference runs for the case library now,
      on the current physics. They are the critical path for Tier 2.
- [ ] Obtain ETEX release and station data; confirm ERA5 1994 download.
- [ ] Locate the NAME footprints, EDGAR maps and radon data on the group
      archive; note paths in PROGRESS.md (restricted, never in the repo).

### GATES

- [ ] Merge the GATES code review PR. Copy `dev/agent/INSTRUCTIONS.md` into the
      GATES repo with paths adjusted; its queue is §5 of the review document.

---

## 3. Running sessions on Isambard AI

Queue times are long, the login node is policed, and you have GPU hours. So the
agent runs **inside a long GPU allocation**, and the login node holds only ssh
and tmux.

1. Submit `dev/agent/isambard/agent_driver.slurm` with the longest wall time
   the partition allows. It starts a tmux session on the compute node and runs
   the driver loop.
2. The driver loop (`dev/agent/isambard/run_task.sh`) takes the next unclaimed
   task from the chosen queue, launches a fresh Claude Code session for it in
   headless mode with the task ID as the prompt, waits for it to finish, and
   moves to the next. Each task gets a clean context.
3. Attach when you want to watch or redirect: `ssh <node>` from the login
   node, then `tmux attach -t glide-agent-<queue>`. Ctrl-C in the pane stops
   the current session; the driver moves on or exits depending on the flag you
   set.
4. When the allocation ends, resubmit. The queue and PROGRESS file are the only
   state. Test this cold-restart once deliberately before relying on it.
5. Run Queue A and Queue B as two separate allocations with different models if
   you want them to overlap.

The idle-GPU cost while the agent thinks is accepted; it is far cheaper than a
queue wait before every test. The agent runs GPU tests directly on the node (no
inner sbatch) because the allocation already holds the GPU.

Before the first real task, run the three checks in §2 "Environment" from
inside an allocation: API egress, arm64 binary, git push.

---

## 4. Choosing the model per task

The rule: match model cost to the cost of an undetected error.

| Queue | Tasks | Model | Why |
| --- | --- | --- | --- |
| A | Everything before the physics freeze; validation analysis; all PR reviews | Fable (`claude-fable-5-1`), high effort | failure modes are silent: the code runs, tests pass, the physics has changed |
| B | Met cache, sparse output, batching, packaging, docs, figure tooling | Sonnet (`claude-sonnet-5`) | well-specified, cheap to check |
| either | Column and satellite releases, composite met source | Opus or Fable | design-heavy but not physics-changing |

Set the model in `run_task.sh` per queue. If in doubt, use the stronger model;
the marginal cost is small against a bad merge.

---

## 5. Reviewing and merging

1. When a PR appears, the driver (or you) starts a **fresh** session with
   `/code-review` on the PR number, with the strongest model. The author
   session is blind to its own errors; a cold reviewer is not.
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
4. Merge with squash, keeping the task ID in the title. Delete the branch.
5. If a PR is not mergeable, comment with what to change and mark the task
   `REWORK` in PROGRESS.md; the next session on that queue picks it up.

---

## 6. Escalations and interventions

The agent signals by writing a `BLOCKED` entry in PROGRESS.md with the question
and, if a PR exists, a comment on it. It then stops that task and, if the
driver allows, moves to the next unblocked task.

You respond by editing the task in QUEUE.md (clarify scope, add a decision) or
by writing the answer under the BLOCKED entry and clearing the flag. Keep
answers in the files, not in chat, so the next session sees them.

Intervene directly (attach to tmux) when a session is looping, when a GPU job
it launched has run far past its expected time, or when you see it heading into
a stop-condition area without stopping.

---

## 7. Cadence

- **Daily, 20 minutes:** read new PROGRESS entries, review and merge PRs,
  answer BLOCKED items.
- **Weekly:** reprioritise QUEUE.md, update `dev/roadmap.md` if scope moved,
  check the benchmark and storage numbers against the roadmap targets, confirm
  the reference-run pipeline (FLEXPART, ETEX) is progressing.
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
- **Context exhaustion mid-task.** The session forgets its own plan. Defence:
  PROGRESS entries written early and updated, tasks sized to hours, and the
  instruction to split rather than push on.
- **Test suite drift.** Skipped or xfailed tests accumulating. Defence: CI
  reports the skip count; reviews check it.
- **Runaway compute.** A convergence sweep launched at ten times the intended
  size. Defence: task specs state expected runtimes; the agent must stop and
  report if a run exceeds twice the estimate.
- **Documentation lag.** Code merged without docs and STATUS updates. Defence:
  it is in the acceptance criteria of every task, and reviews check it.
