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
- **Agents never submit jobs.** Isambard AI asks that agents do not submit
  jobs. You submit every allocation; the agent works only on the node it was
  given, and the driver disables the SLURM job-control commands inside it.
- **Environment-specific values stay out of the repository.** Data paths,
  scratch paths and the default walltime live in an untracked
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
      type, cadence and size. Model levels are the default from task A01 on
      ([decision 0017](decisions/0017-model-levels-default.md)), so download
      new cubes on model levels.
- [ ] The reference multi-site configuration
      (`configs/example_multisite_january.yaml` or its successor) points at a
      cube that exists. The shipped configs read `data/era5/EUROPE_*.zarr`
      (model levels: `data/era5-model/`), so symlink those directories in the
      checkout to the cubes under `$GLIDE_DATA`.
- [ ] A small cube for smoke tests and a medium cube for benchmarks, both local.
- [ ] Decide which additional cubes you will download before the tasks that
      need them: a model-level cube matching the reference configuration's
      pressure-level cube (needed by A01, the second task); the Tier 3 cubes
      for ETEX (Oct–Nov 1994, Europe), CAPTEX (Sep–Oct 1983) and ANATEX
      (Jan–Mar 1987, North America), on model levels; a UK cube (v1.x).

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
- [ ] Download the ETEX and DATEM tracer data per `data/tracer-experiments.md`
      and set `GLIDE_ETEX` and `GLIDE_DATEM`. ETEX downloads by script; DATEM
      needs a browser.
- [ ] Create an OSF account for the pre-registration (needed after the freeze;
      see §7).
- [ ] Locate the NAME footprints, EDGAR maps and radon data on the group
      archive, and set their variables in `env.local.sh`. They are restricted
      and never enter the repository.

### GATES

- [ ] Merge the GATES code review PR. Copy `dev/agent/` into the GATES repo with
      paths adjusted; its queue is §5 of the review document.

---

## 3. Running sessions on Isambard AI

Queue times are long, the login node is policed, and you have GPU hours. So
each agent works **inside a long allocation that you submit**: one GPU and 16
CPU cores of a GH200 node (every Isambard AI compute node has both). The
allocation is the only queue wait. Everything the agent does, from CPU tests
to GPU runs, happens directly on that node.

**Agents never submit jobs.** Isambard AI asks that agents do not submit jobs,
so:
- only you run `submit.sh` or any other `sbatch`, from the login node;
- inside the allocation, the driver puts refusing stand-ins for `sbatch`,
  `srun`, `salloc`, `scancel` and `scontrol` first on `PATH`. Read-only
  `squeue` and `sinfo` still work;
- the instructions forbid job control outright, and a task that needs a job
  (more GPUs, a longer run than the walltime left, a scheduled job) stops and
  asks you;
- the driver loop refuses to start outside an allocation, and a session that
  finds no `SLURM_JOB_ID` stops at once.

The guard catches mistakes; it is not a sandbox. The instructions are the rule.

**Submitting.** From the login node:

```bash
dev/agent/isambard/submit.sh A shell      # queue A, watched
dev/agent/isambard/submit.sh B            # queue B, unattended loop
dev/agent/isambard/submit.sh A loop 12:00:00
```

The arguments are the queue, the mode (`loop`, the default, or `shell`) and
an optional walltime. The scheduler chooses the account and partition. The job
loads modules, runs pre-flight checks (GPU, Claude API, GitHub) and starts a
dedicated tmux server on the compute node. The job log prints the node and the
tmux socket. To attach:

```bash
ssh <node>
tmux -L glide-<jobid> attach
```

Detaching (Ctrl-b d) leaves everything running. The allocation lives entirely
on the compute node, so it survives your connection dropping. Nothing
long-lived runs on the login node.

### Which mode, when

| Phase | Mode | Why |
| --- | --- | --- |
| Phase 1, Queue A (A00 to A8) | **shell**, watched | These tasks change results and are the riskiest. Queue A waits for your merge after every task anyway, so an unattended loop would mostly sit idle. Watching lets you answer questions at once instead of through a BLOCKED draft and a resume request. |
| Queue B, throughout | **loop**, unattended, in a second allocation | Independent, lower-risk infrastructure tasks; the loop keeps going without you. |
| Pre-registration (A8b) | shell or loop | A drafting task; you edit the PR anyway. |
| Validation, A9 to A11 | **loop**, unattended | Deliberately. Steering an agent during a registered validation run is itself a choice made after seeing results. With the loop, every intervention goes through a resume request in PROGRESS.md and is logged. The work is mostly long compute bound by fixed rules. |

**A watched session.** Attach, then start Claude in the tmux shell. The shell
already has the modules, the environment and the guard:

```bash
claude --model "$GLIDE_AGENT_MODEL"
```

Give it the same instruction the loop uses: follow
`dev/agent/INSTRUCTIONS.md`, your queue is A, select work with §1 and deliver
one piece of work per §2. When it opens the PR, review and merge, then start a
fresh session for the next task (`/clear` or a new `claude`), so each task
starts cold as it would in the loop. If you redirect it or answer a question
mid-task, it records your instruction in its progress file
(INSTRUCTIONS.md §1). Long runs, such as A5's sweeps, can be left running
while you detach. Log your watching time in the paper repository's
`metrics/human_time.csv`.

**The unattended loop.** Each iteration checks that the working tree has no
uncommitted changes (it stops rather than discard anything), syncs `main`,
and starts a fresh session. The session does one of three things:

- delivers one piece of work as a PR, and the loop continues;
- prints `WAITING`, because the next task depends on a PR you have not merged.
  The loop sleeps (`GLIDE_AGENT_WAIT_SECONDS`, default 30 minutes) and tries
  again. Each check is a short session, so raise the wait or stop the loop
  before a long absence;
- prints `NO_TASK`, because every task in the queue is merged. The loop exits.

`touch dev/agent/STOP_AFTER_TASK` ends the loop after the current session.
Ctrl-C in the pane aborts at once.

**Walltime.** Sessions check the time left before long runs and hand off if a
run will not fit. When the allocation ends, resubmit. The repository is the
only state. Test a cold restart once deliberately before relying on it.

**First runs.** Do A00, A01 and A0 in shell mode and watch closely. They show
whether the instructions, the claim, the progress file and the PR template
work in practice, A01 switches the default meteorology, and every later task
is checked against A0's fixtures. Start the Queue B loop in a second
allocation once Queue A runs smoothly.

**Releasing a stuck claim.** If you close a PR without merging and want the
task redone from scratch, delete its `agent/<ID>-*` branch. The task becomes
available again.

---

## 4. Choosing the model per task

The rule: match model cost to the cost of an undetected error.

| Queue | Tasks | Model | Why |
| --- | --- | --- | --- |
| A | Everything before the physics freeze; validation analysis; all PR reviews | Opus 5.5 (`claude-opus-5-5`), high effort | failure modes are silent: the code runs, tests pass, the physics has changed |
| B | Met cache, sparse output, batching, packaging, docs, figure tooling | Sonnet 5.5 (`claude-sonnet-5-5`) | well-specified, cheap to check |
| either | Column and satellite releases, composite met source | Opus 5.5 | design-heavy but not physics-changing |

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
  the FLEXPART runs and the tracer data are progressing; delete resume requests
  for merged tasks.
- **At the freeze:** you and the external reviewer sign the physics audit;
  tag; update STATUS.
- **Before validation runs:** when the A8b PR arrives, set the final
  tolerances and merge it. Register the record on
  [OSF Registries](https://osf.io/registries) (open-ended template; attach the
  record, the release lists in `docs/validation/preregistration/` and the two
  commit hashes; embargo if wanted). Then replace `**Registration.** pending`
  in the record with the URL, DOI and date, commit to main, and add the DOI to
  "Current state" in PROGRESS.md. Tasks A9 to A11 stop until you do. Any later
  change to the registered settings needs a new decision record and an OSF
  update.

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
