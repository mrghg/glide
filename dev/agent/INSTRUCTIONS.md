# Standing instructions for coding-agent sessions

You are a coding agent working on GLIDE, a backward-in-time Lagrangian particle
dispersion model in pure PyTorch. These instructions apply to every session. Read
them fully, then [AGENTS.md](../../AGENTS.md), which remains authoritative for
code style, memory safeguards and I/O contracts. Where these instructions and
AGENTS.md overlap, AGENTS.md wins.

**Contents**

1. [What you are here to do](#1-what-you-are-here-to-do)
2. [Session protocol](#2-session-protocol)
3. [Verification standard](#3-verification-standard)
4. [Stop conditions](#4-stop-conditions)
5. [Never](#5-never)
6. [Environment](#6-environment)
7. [Managing your own context](#7-managing-your-own-context)
8. [Templates](#8-templates)

---

## 1. What you are here to do

Deliver one piece of work from [QUEUE.md](QUEUE.md), completely, verified and
documented, as a pull request, then stop. The prompt names your queue (A or B)
and sometimes a task ID. If it names an ID, do that task. Otherwise select work
with the procedure below.

The plan behind the queue is [dev/roadmap.md](../roadmap.md). The physics is in
[docs/physics.md](../../docs/physics.md) and its siblings; major decisions are in
[dev/decisions/](../decisions/). Read the decision records referenced by your
task before designing anything.

You work on the model that a sceptical community will judge. Correctness and
traceability matter more than speed of delivery. A task delivered late with
every check passed is a success; a task delivered on time with a silent change
to the physics is a failure that costs weeks.

### Selecting work

Task state is derived from the repository, never from memory or chat.

1. **Synchronise.** `git fetch --prune origin`, `git switch main`,
   `git pull --ff-only`.
2. **Resume requests come first.** [PROGRESS.md](PROGRESS.md) "Resume
   requests" lists numbered entries (`R1`, `R2`, …) written by the human, each
   naming a task and giving an instruction or an answer to a BLOCKED question.
   An entry is pending if the task's branch exists and the progress file on
   that branch (`git show origin/agent/<ID>-<slug>:dev/agent/progress/<ID>.md`)
   does not list the entry under `handled:`. A task whose progress file on its
   branch has status `HANDOFF` is also pending. Take the first pending item
   for a task in your queue.
3. **Otherwise classify each task in your queue:**
   - **MERGED** — `dev/agent/progress/<ID>.md` exists on `main`. Progress files
     reach main only through a merged PR.
   - **IN FLIGHT** — not merged, and either an open PR whose title starts with
     `<ID>:` exists (`gh pr list --state open --json number,title`) or a remote
     branch `agent/<ID>-*` exists (`git ls-remote --heads origin "agent/<ID>-*"`).
   - **AVAILABLE** — neither.
4. **Take the first AVAILABLE task**, in the order listed in QUEUE.md, whose
   `Depends on` task IDs are all MERGED (in either queue). Prerequisites that
   are not task IDs (data, source code, a human's reference run) are checked
   after you start; if one is missing, that is a stop condition (§4).
5. If nothing qualifies but some task in your queue is IN FLIGHT, or AVAILABLE
   with unmerged dependencies, print a line containing only `WAITING` and stop.
6. If every task in your queue is MERGED, print a line containing only
   `NO_TASK` and stop.

Never start work on an IN FLIGHT task unless a pending resume request or a
`HANDOFF` status names it.

---

## 2. Session protocol

### A new task

1. **Orient.** Read PROGRESS.md (current state, data inventory, resume
   requests), your task's spec in QUEUE.md, the files the spec names, and the
   progress files of the tasks it depends on (their hand-off notes). Do not
   start coding until you can state the acceptance criteria in your own words.
2. **Branch.** `git switch -c agent/<ID>-<short-slug>` from an up-to-date
   `main`. You never commit to `main`.
3. **Claim.** Create `dev/agent/progress/<ID>.md` from the template in §8 with
   status `CLAIMED` and your plan: the files you expect to touch, the tests you
   will add, the figures you will produce, the expected runtime and memory of
   any GPU runs. Commit it and **push the branch immediately**, so that other
   sessions see the claim.
4. **Implement** within the scope in the spec. If the task turns out to be
   larger than a few hours, split it: deliver the first coherent part, and list
   the remainder under `proposed follow-ups` in your progress file.
5. **Test.** Run the targeted tests for changed modules, then the full CPU
   suite. Run the GPU checks in §3 if the task touches the runtime, the step
   path, the gridder, the reader or the release generator.
6. **Document.** Update the relevant `docs/` page, STATUS.md (if a milestone,
   number or priority changed) and the README (if a flag or behaviour changed).
   Write a decision record if the spec says so. Do not edit PROGRESS.md; if its
   "Current state" should change, say so in your progress file's hand-off notes.
7. **Commit** in small, conventional commits (`feat(gridder): ...`,
   `fix(met): ...`, `test: ...`, `docs: ...`), each ending with the attribution
   line the harness gives you. Run `ruff format .` and `ruff check --fix .`
   before every commit; pre-commit runs gitleaks and must pass.
8. **Open the PR** with the template in §8, titled `<ID>: <task title>`. Then
   update your progress file: status `PR OPEN #<n>`, what was done, what was
   verified, hand-off notes. Commit and push.
9. **Stop.** Do not start another piece of work in the same session.

### A resume request or a hand-off

1. Check out the task branch and merge `origin/main` into it. Never rebase or
   force-push a pushed branch. Resolve conflicts.
2. Read the resume entry (or the hand-off notes), the PR and its review
   comments (`gh pr view <n> --comments`).
3. Do the work asked for, verify per §3, commit and push.
4. Add the entry ID under `handled:` in the progress file with a one-line note,
   and update its status. If the PR is a draft opened for a BLOCKED stop and the
   task is now complete, retitle it `<ID>: <task title>` and mark it ready
   (`gh pr ready <n>`).
5. **Stop.**

---

## 3. Verification standard

Every task has acceptance criteria in its spec. The following apply on top for
any change to `src/lpdm/`:

- **Full CPU suite green**, no new skips, no loosened tolerances, no deleted
  tests. If a test must change because behaviour legitimately changed, the
  spec must say so and the PR must explain it.
- **Golden comparison.** Run `scripts/agent/compare_golden.py` (created by task
  A0) against the fixtures named in the spec, at the tolerance the spec
  declares. Paste the output in the PR. For tasks the roadmap marks
  "numerics", a golden difference within tolerance is the acceptance; for all
  other tasks the golden comparison must be bit-identical on CPU.
- **Static/dynamic parity.** The CUDA static path and the eager CPU path must
  agree on the smoke configuration to the parity tolerance the existing tests
  use. The eager CPU path is the numerical reference
  ([decision 0002](../decisions/0002-pure-pytorch-device-agnostic.md)).
- **Graph capture intact.** The two CPU-only capture guard tests must pass, and
  a GPU smoke run with `GLIDE_COMPILE=1` must log the compiled path with no
  "WON'T CONVERT" warnings ([decision 0004](../decisions/0004-cuda-graph-static-step-path.md)).
- **Well-mixed gates** pass for any change to turbulence, reflection, the
  drift, the gridder or time stepping.
- **Benchmark.** If the task claims a performance change, run
  `scripts/agent/benchmark.py` (task A0) before and after on the same node and
  report footprints per second and the phase breakdown.
- **Figures.** If the spec lists figures, produce them with a script under
  `scripts/validation/`, commit the script, and attach the images to the PR.
  Do not commit large binaries; put them under `$GLIDE_AGENT_OUTPUTS` and link.

Expected runtimes are stated in each spec. If a run exceeds twice its estimate,
stop it, record why in your progress file, and continue only if the cause is
understood.

---

## 4. Stop conditions

Stop when any of these arises:

- The task requires changing physics outside its stated scope, or changing
  any physics after the freeze tag exists.
- The task requires changing the output contract (`footprints.zarr`,
  `endpoint_particles.parquet`, `trajectory_diagnostics.parquet`,
  `run_metadata.json`) in a way the spec did not authorise.
- A validation threshold, tolerance or pre-registered metric would need to
  change.
- GPU parity, a well-mixed gate or the golden comparison fails in a way the
  task did not predict, and you cannot explain it within an hour of
  investigation.
- The task needs meteorology that is not in the data inventory.
- The task needs to delete or overwrite anything outside the repository, the
  scratch directory, or `$GLIDE_AGENT_OUTPUTS`.
- The task needs credentials, licences or data you do not have, or an
  environment variable from §6 that is unset.

How to stop:

1. Set your progress file's status to `BLOCKED` and write the question, and
   what you would do under each answer.
2. Commit and push the branch.
3. If no PR exists, open a **draft** PR titled `<ID>: BLOCKED — <question in a
   few words>` with the question in the body. If a PR exists, comment on it
   with the question.
4. End the session.

The human answers by adding a resume request to PROGRESS.md. A stop is a normal
outcome, not a failure.

**Running out of context** is handled differently. If you are more than half
through your context budget and less than half through the task, write
thorough hand-off notes, set status `HANDOFF`, commit, push and stop. The next
session continues automatically; no human action is needed.

---

## 5. Never

- Commit to `main`, force-push, rewrite history on a shared branch, or delete a
  branch you did not create.
- Edit PROGRESS.md. The human maintains it.
- Commit `dev/agent/isambard/env.local.sh`, or write the values of the
  `GLIDE_*` environment variables (paths, account codes) into any tracked file.
  Refer to the variable name instead.
- Run anything heavier than `git`, `tmux` or an editor on a login node. All
  compute happens inside the allocation.
- Download meteorology, or any dataset, without a task that authorises it.
- Loosen a test tolerance, skip a test, or mark one xfail to make a run pass.
- Change default configuration values silently. Defaults change only through a
  task that says so, with a decision record.
- Put restricted data (NAME, FLEXPART reference runs, EDGAR, station data)
  into the repository or the tests. See [data/README.md](../../data/README.md).
- Remove or weaken the memory guards, the fail-fast behaviour, or the
  diagnostic metadata described in AGENTS.md.
- Launch a GPU run without an estimate of its runtime and memory.
- Report a result you did not observe. If a step was skipped, say so.

---

## 6. Environment

The driver sources `dev/agent/isambard/env.local.sh` before starting you. That
file is untracked; the operator writes it from
[env.local.sh.example](isambard/env.local.sh.example). The variables below are
therefore set in your environment. If a variable a task needs is unset, that is
a stop condition.

| Variable | Meaning |
| --- | --- |
| `GLIDE_REPO` | the repository checkout, which is your working directory |
| `GLIDE_DATA` | root of the local met cubes; the inventory in PROGRESS.md gives paths relative to it |
| `GLIDE_AGENT_OUTPUTS` | run outputs, figures and long logs |
| `GLIDE_SCRATCH` | scratch space; `UV_CACHE_DIR` and `TRITON_CACHE_DIR` are under it |
| `GLIDE_FLEXPART_SRC`, `GLIDE_FLEXPART_OUTPUTS`, `GLIDE_NAME_FOOTPRINTS`, `GLIDE_EDGAR`, `GLIDE_ETEX` | restricted reference data, never copied into the repository |
| `GLIDE_SLURM_ACCOUNT`, `GLIDE_SLURM_PARTITION` | only for an sbatch a task explicitly asks for |

Fixed facts:

- **Machine:** Isambard AI, NVIDIA GH200 nodes (aarch64). You are running
  inside a SLURM allocation that already holds one GPU. Run GPU tests directly;
  do not submit inner jobs.
- **Modules:** already loaded by the driver: `cudatoolkit/24.11_12.6` (never
  `cuda/12.6`) and `gcc-native/14.2` with `CC=gcc CXX=g++`. See
  `scripts/run_periodic_cuda.slurm` for the full environment logic.
- **Python:** `.venv/bin/python` in the repo checkout. Tests:
  `.venv/bin/python -m pytest -q`. Install:
  `uv pip install --python .venv/bin/python -e ".[dev]"`.
- **Reference configs:** smoke `configs/smoke_mhd_single_release.yaml`;
  reference `configs/example_multisite_january.yaml`.
- **Git:** push to `origin` on GitHub; open PRs with `gh pr create`.

---

## 7. Managing your own context

- Write the progress file early and update it as you go; it is the only thing
  that survives you.
- Prefer reading the specific functions the spec names over whole files.
- Keep long outputs (test logs, benchmark tables) in files under
  `$GLIDE_AGENT_OUTPUTS` and quote only the lines that matter.
- When a decision is yours to make, make it and record it in the PR
  description; do not ask the human questions that the spec, the docs or the
  decision records already answer.

---

## 8. Templates

### Progress file, `dev/agent/progress/<ID>.md`

There is no MERGED status: the file's presence on `main` is what marks a task
merged.

```
# <ID> — <title>

- status: CLAIMED | PR OPEN #<n> | BLOCKED | HANDOFF
- queue: <A|B>   model: <model>   branch: agent/<ID>-<slug>
- started: YYYY-MM-DD   updated: YYYY-MM-DD
- plan: <files, tests, figures, expected GPU runtime and memory>
- done: <what changed, one line per item>
- verified: <CPU suite; golden fixtures + tolerance + result; parity; capture; benchmark before/after>
- docs: <pages updated; decision record if any>
- handoff: <for the next session and the human, including suggested updates to PROGRESS.md "Current state">
- proposed follow-ups: <new tasks discovered, one line each>
- question: <if BLOCKED: the question, and what you would do under each answer>
- handled: <resume entry IDs, one per line, with a note>
```

### PR description

```
## <ID>: <title>

Roadmap items: <IDs>. Task spec: dev/agent/QUEUE.md#<ID>.

### What changed
- ...

### Verification
- CPU suite: <n> passed, <n> skipped (was <n>), 0 loosened tolerances
- Golden: <fixtures>, tolerance <x>, result <max abs / rel diff>
- Parity (static vs dynamic, smoke config): <result>
- Graph capture: <compiled, no WON'T CONVERT> / n/a
- Benchmark (<config>, <node>): before <x> fp/s, after <y> fp/s; phase table attached
- Figures: <paths / attached>

### Docs
- <pages updated>; decision record <n> if any

### Out of scope / follow-ups proposed
- ...

🤖 Generated with [Claude Code](https://claude.com/claude-code)
```
