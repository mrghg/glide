# GLIDE — agent & contributor guide

Project-specific defaults for coding agents (and humans) working in this repo.
GLIDE is a backward-in-time Lagrangian Particle Dispersion Model in pure PyTorch;
see [README.md](README.md) for the user-facing overview.

This file is the agent-agnostic contributor guide (the `AGENTS.md` convention;
[CLAUDE.md](CLAUDE.md) is a pointer to it). Keep operational guidance here.

**Where things live**
- Current status — what works, what's pending, latest results: [STATUS.md](STATUS.md).
- Physics & systems docs: [docs/](docs/) (architecture, turbulence, convection,
  validation, LPDM spec).
- Major design/physics decisions (why, with rejected alternatives):
  [dev/decisions/](dev/decisions/).
- Historical dated work-orders (physics/test reviews): [dev/](dev/). The
  stream-of-consciousness journal was retired 2026-07-24; its durable content
  moved to STATUS / decisions / docs, and full history is in git.

## Scope and priorities
- Prioritise memory safety and predictable runtime behaviour over raw speed.
- Keep changes small, explicit, and testable; avoid broad refactors unless asked.
- Keep the physics interrogable — don't obscure the model behind premature abstraction.

## Environment and execution
- Use the project venv at `.venv`. Editable-install for imports:
  `uv pip install --python .venv/bin/python -e ".[dev]"`.
- Run tests from repo root (no `PYTHONPATH`): `.venv/bin/python -m pytest -q`.
  Targeted: `.venv/bin/python -m pytest -q tests/<module>.py`.
- Primary entrypoint: `python -m lpdm.main --config <yaml>`.
- This is a device-agnostic codebase (CUDA / MPS / CPU). GPU-specific work must be
  device-gated; the local dev box is CPU-only (GPU runs happen on Isambard AI).

## Runtime memory safeguards (required for run-related changes)
Memory controls live in the `memory:` section of the run config (not CLI flags):
`met_cache_max_hours`, `met_cache_on_host`, `met_prefetch`, `log_every_steps`,
`gc_every_steps`, `guard_check_every_steps`, and the `guard_max_*_gib` limits.
When editing runtime/orchestration logic (`src/lpdm/main.py`, met reads, tensor
loops):
- Do not remove fail-fast memory-guard behaviour unless explicitly requested.
- Do not introduce unbounded caches in long-running loops.
- Release large temporary tensors as soon as practical.
- Preserve diagnostic metadata output on guard-triggered aborts.

## Data and I/O behavior
- Keep output paths compatible with local and remote (e.g. `gs://`) storage.
- Maintain existing output contracts unless a breaking change is requested:
  `footprints.zarr`, `endpoint_particles.parquet`, `trajectory_diagnostics.parquet`,
  `run_metadata.json`.
- Restricted validation data (NAME/FLEXPART/EDGAR) is not in the repo; see
  [data/README.md](data/README.md). Tests must not depend on it.

## Code style
- Match existing formatting and naming; comment only non-obvious logic.
- Prefer explicit validation for new config fields / CLI flags.
- Keep memory defaults conservative.

## Validation expectations
After runtime-impacting edits: run diagnostics on edited files, run at least the
targeted tests for changed modules, and update [README.md](README.md) (and the
relevant `docs/` page) for new flags or behaviour changes.

## Keeping the docs current
- Record a **major** design or physics decision as a short record in
  [dev/decisions/](dev/decisions/) (see its README for the format). Reflect the
  resulting behaviour in the relevant `docs/` page — decisions capture *why*,
  `docs/` capture *how*.
- Keep [STATUS.md](STATUS.md) honest: update it when a milestone lands, a
  validation number changes, or the "what's next" priorities shift.
- Don't reopen a bloated chronological journal; use git history for narrative.

## Non-goals
- Don't add unrelated dependencies without strong justification.
- Don't reintroduce deployment infra (containers/cloud) until the architecture
  has settled (see the Next Steps in the README).
