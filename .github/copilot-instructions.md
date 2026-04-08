# GLIDE Copilot Instructions

These instructions define project-specific defaults for coding agents working in this repository.

Project vision, architecture intent, and milestone history are tracked in `CHECKPOINT.md`.
Keep detailed operational guidance here and avoid duplicating rules across both files.

## Scope and priorities

- Prioritize memory safety and predictable runtime behavior over raw speed.
- Keep changes small, explicit, and testable.
- Avoid broad refactors unless explicitly requested.

## Environment and execution

- Use the project virtual environment at `.venv`.
- Prefer editable install for imports: `uv pip install --python .venv/bin/python -e .`.
- Run tests from repo root without `PYTHONPATH`: `.venv/bin/python -m pytest -q`.
- For targeted runs, prefer `.venv/bin/python -m pytest -q tests/<module>.py`.
- Primary entrypoint is `python -m lpdm.main`.

## Runtime memory safeguards (required for run-related changes)

When editing runtime/orchestration logic (`src/lpdm/main.py`, meteorology reads, tensor loops), preserve and account for these controls:

- `--met-cache-max-hours`
- `--memory-log-every-steps`
- `--gc-every-steps`
- `--memory-guard-max-rss-gib`
- `--memory-guard-max-device-allocated-gib`
- `--memory-guard-max-device-reserved-gib`
- `--memory-guard-check-every-steps`

Rules:

- Do not remove fail-fast memory guard behavior unless explicitly requested.
- Do not introduce unbounded caches in long-running loops.
- If adding large temporary tensors/arrays, release references as soon as practical.
- Preserve diagnostic metadata output on guard-triggered aborts.

## Data and I/O behavior

- Keep output paths compatible with local and remote (`gs://`) storage.
- Maintain existing output contracts unless user asks for a breaking change:
  - `endpoint_particles.parquet`
  - `trajectory_diagnostics.parquet`
  - `run_metadata.json`

## Code style

- Match existing formatting and naming conventions.
- Add comments only for non-obvious logic.
- Prefer explicit validation for new CLI flags.
- Keep defaults conservative for memory usage in cloud VM scenarios.

## Validation expectations

After runtime-impacting edits:

1. Run diagnostics/error checks on edited files.
2. If feasible, run at least targeted tests related to changed modules.
3. Update `README.md` for new flags or behavior changes.

## Non-goals

- Do not add unrelated dependencies without strong justification.
- Do not modify deployment infra (`Dockerfile`, `deploy.sh`) unless requested.
