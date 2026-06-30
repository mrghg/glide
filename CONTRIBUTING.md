# Contributing to GLIDE

Thanks for your interest. GLIDE is research code, shared for input from the
research community — bug reports, questions, physics feedback, and pull requests
are all welcome. It is a work in progress and not yet validated for production
use, so expect rough edges and please flag anything that looks wrong.

For larger changes, please open an issue first to discuss the approach before
investing time in a PR.

## Development setup

```bash
uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install -e ".[dev]"        # core + pytest
# add the notebook/plotting stack with ".[viz,dev]"
```

`uv` (>= 0.4.0) is recommended; see the [README](README.md) for the full install
matrix and the GPU (Isambard AI) setup.

## Running the tests

```bash
.venv/bin/python -m pytest -q                  # full suite
.venv/bin/python -m pytest -q tests/test_hanna.py   # a single module
```

The suite runs in well under a minute with no network access — all tests use
synthetic met via `AnalyticMetReader`. Please make sure it passes before opening
a PR, and add tests for new behaviour.

## Conventions

- **Device-agnostic.** The model runs on CPU, CUDA, and MPS. Keep it that way:
  gate GPU-specific code behind device checks, and don't assume a GPU is present
  (CI and most development happen on CPU).
- **Memory safety first.** Preserve the fail-fast memory guards and avoid
  unbounded caches in long-running loops. Release large temporary tensors
  promptly. See the `memory:` config section.
- **Keep the physics interrogable.** Prefer clear, inspectable code over
  premature abstraction — a reader should be able to follow the model.
- **Match the surrounding style.** Follow existing formatting and naming; comment
  only non-obvious logic. Prefer explicit validation for new config fields.
- **Small, focused changes.** Avoid broad refactors unless they're the point of
  the PR.

## Documentation

Update the docs in the same PR as the code:

- [README.md](README.md) for user-facing flags, configs, or behaviour changes.
- The relevant page under [docs/](docs/) for physics or architecture changes.

## Data

The validation datasets (NAME, FLEXPART, EDGAR) are **not** redistributed with
this repo and are gitignored — see [data/README.md](data/README.md). Do not
commit data files, and do not make tests depend on restricted data; tests must
run from the synthetic fixtures alone.

## Pull requests

1. Branch off `main`.
2. Make your change with tests and docs updated.
3. Ensure `pytest` passes locally.
4. Open the PR with a clear description of what changed and why.

## License

By contributing, you agree that your contributions will be licensed under the
project's [Apache License 2.0](LICENSE).

## Questions

For access to the validation data, or anything else, contact
**Matt Rigby (matt.rigby@bristol.ac.uk)**.
