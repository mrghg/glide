# Standing instructions for coding-agent sessions

You are a coding agent working on GLIDE, a backward-in-time Lagrangian particle
dispersion model in pure PyTorch. These instructions apply to every session. Read
them fully, then [AGENTS.md](../../AGENTS.md), which remains authoritative for
code style, memory safeguards and I/O contracts. Where these instructions and
AGENTS.md overlap, AGENTS.md wins.

`<<PLACEHOLDER>>` fields in §6 are filled in by the human operator before the
first session.

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

Deliver one task from [QUEUE.md](QUEUE.md), completely, verified, documented,
as a pull request, then stop. The task is given to you in the prompt by ID
(for example `A3`). If no ID is given, take the first task in the named queue
whose status in [PROGRESS.md](PROGRESS.md) is `OPEN` (or absent) and whose
dependencies are `MERGED`.

The plan behind the queue is [dev/roadmap.md](../roadmap.md). The physics is in
[docs/physics.md](../../docs/physics.md) and its siblings; major decisions are in
[dev/decisions/](../decisions/). Read the decision records referenced by your
task before designing anything.

You work on the model that a sceptical community will judge. Correctness and
traceability matter more than speed of delivery. A task delivered late with
every check passed is a success; a task delivered on time with a silent change
to the physics is a failure that costs weeks.

---

## 2. Session protocol

1. **Orient.** Read PROGRESS.md (current state, data inventory, open BLOCKED
   items, hand-off notes) and your task's spec in QUEUE.md. Read the files the
   spec names. Do not start coding until you can state the acceptance criteria
   in your own words.
2. **Branch.** `git switch -c agent/<ID>-<short-slug>` from an up-to-date `main`.
   You never commit to `main`.
3. **Claim.** Append a `CLAIMED` entry for the task to PROGRESS.md with the date,
   your model name and the branch name, as the first commit on your branch.
4. **Plan briefly** in the PROGRESS entry: the files you expect to touch, the
   tests you will add, the figures you will produce, the expected runtimes of
   any GPU runs. Update it if the plan changes.
5. **Implement** within the scope in the spec. If the task turns out to be
   larger than a few hours, split it: deliver the first coherent part, and
   append a proposed follow-up task to the "Proposed tasks" section of
   PROGRESS.md for the human to accept into the queue.
6. **Test.** Run the targeted tests for changed modules, then the full CPU
   suite. Run the GPU checks in §3 if the task touches the runtime, the step
   path, the gridder, the reader or the release generator.
7. **Document.** Update the relevant `docs/` page, STATUS.md (if a milestone,
   number or priority changed) and the README (if a flag or behaviour changed).
   Write a decision record if the spec says so.
8. **Commit** in small, conventional commits (`feat(gridder): ...`,
   `fix(met): ...`, `test: ...`, `docs: ...`), each ending with the attribution
   line the harness gives you. Run `ruff format .` and `ruff check --fix .`
   before every commit; pre-commit runs gitleaks and must pass.
9. **Open the PR** with the template in §8. Title: `<ID>: <task title>`.
10. **Record.** Replace your `CLAIMED` entry in PROGRESS.md with a `PR OPEN`
    entry containing the PR number, what was done, what was verified and any
    hand-off notes. This edit goes in the PR too.
11. **Stop.** Do not start another task in the same session.

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
  Do not commit large binaries; put them under the outputs directory named in
  §6 and link.

Expected runtimes are stated in each spec. If a run exceeds twice its estimate,
stop it, record why in PROGRESS.md, and continue only if the cause is
understood.

---

## 4. Stop conditions

Stop the task, write a `BLOCKED` entry in PROGRESS.md stating the question and
what you would do under each answer, comment on the PR if one exists, and end
the session, when any of these arises:

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
  scratch directory, or the outputs directory named in §6.
- The task needs credentials, licences or data you do not have.
- You are more than half through your context budget and less than half
  through the task: write hand-off notes and stop, so the next session can
  continue from the file.

A stop is a normal outcome, not a failure. An answered BLOCKED entry is how the
human gives you a decision.

---

## 5. Never

- Commit to `main`, force-push, rewrite history on a shared branch, or delete a
  branch you did not create.
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

Filled in by the operator. Keep this section short and factual.

- **Machine:** Isambard AI, NVIDIA GH200 nodes (aarch64). You are running
  inside a SLURM allocation that already holds one GPU; run GPU tests directly,
  do not submit inner jobs.
- **Modules:** `module load cudatoolkit/24.11_12.6` (never `cuda/12.6`);
  `module load gcc-native/14.2` with `CC=gcc CXX=g++` for compiled runs. See
  `scripts/run_periodic_cuda.slurm` for the full environment logic.
- **Python:** `.venv/bin/python` in the repo checkout. Tests:
  `.venv/bin/python -m pytest -q`. Install: `uv pip install --python .venv/bin/python -e ".[dev]"`.
- **Repository:** `<<PLACEHOLDER: absolute path of the checkout>>`
- **Data root (cubes):** `<<PLACEHOLDER: GLIDE_DATA path>>`; the inventory is in
  PROGRESS.md.
- **Outputs and figures:** `<<PLACEHOLDER: outputs path on project scratch>>`
- **Scratch for caches:** `<<PLACEHOLDER>>` (`UV_CACHE_DIR`, `TRITON_CACHE_DIR`)
- **Reference configs:** smoke `configs/smoke_mhd_single_release.yaml`;
  reference `configs/example_multisite_january.yaml`.
- **Git:** push to `origin` on GitHub; open PRs with `gh pr create`. Account and
  partition for any sbatch you are explicitly told to use:
  `<<PLACEHOLDER>>`.

---

## 7. Managing your own context

- Write the PROGRESS entry early and update it as you go; it is the only thing
  that survives you.
- Prefer reading the specific functions the spec names over whole files.
- Keep long outputs (test logs, benchmark tables) in files under the outputs
  directory and quote only the lines that matter.
- When a decision is yours to make, make it and record it in the PR
  description; do not ask the human questions that the spec, the docs or the
  decision records already answer.

---

## 8. Templates

### PROGRESS.md entry

```
### <ID> — <title>
- status: CLAIMED | PR OPEN #<n> | BLOCKED | REWORK | MERGED
- date: YYYY-MM-DD   model: <model>   branch: agent/<ID>-<slug>
- plan: <files, tests, figures, expected GPU runtime>
- done: <what changed, one line per item>
- verified: <CPU suite; golden fixtures + tolerance + result; parity; capture; benchmark before/after>
- docs: <pages updated; decision record if any>
- handoff: <anything the next session needs to know>
- question (if BLOCKED): <the question, and what you would do under each answer>
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
