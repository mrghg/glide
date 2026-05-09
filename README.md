# GPU-accelerated Lagrangian Inversion & Dispersion Engine (GLIDE)

Backward-in-time Lagrangian Particle Dispersion Model (LPDM) scaffold for GPU-first execution.

## Project Layout

- `src/lpdm/met_reader.py`: Meteorological I/O interface and ARCO ERA5 reader skeleton.
- `src/lpdm/gpu_engine.py`: GPU physics/advection utilities and turbulence stepping helpers.
- `src/lpdm/footprint_gridder.py`: Footprint accumulation placeholder.
- `src/lpdm/release_generator.py`: Particle release generators.
- `src/lpdm/output_writer.py`: Storage/output placeholder.
- `src/lpdm/main.py`: Orchestrator placeholder.
- `src/lpdm/visualize.py`: Plotly and Matplotlib diagnostics.
- `tests/test_physics.py`: Physics unit tests with analytical/mock met fields.
- `Dockerfile`: GPU-enabled runtime image.
- `deploy.sh`: Artifact Registry build + Cloud Run GPU deployment.

## Documentation Governance

Use the following source-of-truth split to avoid drift:

- `.github/copilot-instructions.md`: operational coding-agent behavior, guardrails, and workflow expectations.
- `CHECKPOINT.md`: project goal, architecture intent, milestone history, and next recommended technical priorities.
- `README.md`: user-facing setup, run commands, flags, and output contracts.
- `VALIDATION.md`: validation suite scope, test tolerances and seeds, and which metrics are placeholder pending later milestones.
- `docs/turbulence.md`: turbulence parameterization spec — modular architecture, scheme math, and the M1 implementation/validation plan.

If behavior or interfaces change, update both the implementation and the matching documentation source above in the same PR.

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m lpdm.main --config configs/local_smoke_test.yaml
```

Run configs are YAML; the schema lives in [src/lpdm/config.py](src/lpdm/config.py). Two examples ship with the repo:
- `configs/local_smoke_test.yaml` — small backward run against `data/sample_met.zarr`.
- `configs/example_mhd_january.yaml` — Hanna scheme, FLEXPART-aligned grid for the comparison fixture in `data/FLEXPART/`.

The CLI is intentionally tiny: `--config <path>` plus `--device`, `--output-uri`, `--start-time` overrides. Everything else is set in the YAML.

## One-Command Bootstrap (Recommended)

After cloning on any machine:

```bash
chmod +x scripts/setup.sh
./scripts/setup.sh --run-tests
```

What this does:
- Uses `uv` if available (falls back to `venv + pip` automatically).
- Creates `.venv` with Python 3.11 by default.
- Installs `requirements.txt`.
- Optionally runs `tests/test_physics.py` when `--run-tests` is passed.

Custom Python executable:

```bash
./scripts/setup.sh --python python3.11 --run-tests
```

## Install With UV

Install UV (if not already installed):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Create a virtual environment and install dependencies from `requirements.txt`:

```bash
uv venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt
uv pip install -e .
```

Run the model entrypoint:

```bash
python -m lpdm.main --config configs/local_smoke_test.yaml
```

## Physics Tests

```bash
.venv/bin/python -m pytest -q tests/test_physics.py
```

Recommended once per environment (so imports work without `PYTHONPATH`):

```bash
uv pip install --python .venv/bin/python -e .
```

Then run all tests from repo root:

```bash
.venv/bin/python -m pytest -q
```

The suite includes:
- Uniform wind RK2 advection precision test (zero turbulence).
- Zero-wind Langevin diffusion Gaussianity test.
- Well-mixed periodic turbulence uniformity test.

Each test also verifies particle mass conservation by checking total weight.

## Minimal Trajectory Run (Local or Vertex Notebook)

### Downloading Local Sample Data

For local development or rapid testing, reading from the remote ARCO ERA5 Zarr store can be slow or memory-intensive. You can download a cropped subset of the dataset (a "data cube") that spans only your area and time of interest:

```bash
.venv/bin/python scripts/download_sample_cube.py \
    --out-path data/sample_met.zarr \
    --store-uri gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3 \
	--time-start 2023-12-29T18:00:00 \
    --time-end 2024-01-01T06:00:00 \
	--lon-min -127.0 \
	--lon-max -117.0 \
	--lat-min 33.0 \
	--lat-max 43.0
```

Notes:
- Public ARCO buckets are opened anonymously in the helper script, so ADC credentials are not required for that default source URI.
- To choose output store format explicitly, pass `--zarr-version 2` (default) or `--zarr-version 3`.

After downloading the sample data, either:
- run `python -m lpdm.main --config configs/local_smoke_test.yaml` directly, or
- use VS Code Run/Debug profile `GLIDE: lpdm.main (Local Sample Met)` from `.vscode/launch.json` (which uses that same config).

### Local Smoke Test (Using Downloaded Sample)

`configs/local_smoke_test.yaml` is wired to `data/sample_met.zarr` with a 3 h backward run. Use it to validate the local meteorology path end-to-end:

```bash
.venv/bin/python -m lpdm.main --config configs/local_smoke_test.yaml
```

Expected outputs under `outputs/demo-run-local`:
- `endpoint_particles.parquet`
- `trajectory_diagnostics.parquet`
- `footprints.zarr`
- `run_metadata.json`

### Running the Model

Author a YAML config (start from one of the examples in `configs/`) and pass it via `--config`:

```bash
.venv/bin/python -m lpdm.main --config configs/example_mhd_january.yaml
```

CLI overrides are intentionally limited to the three knobs that change between runs of the same physics config:

```bash
.venv/bin/python -m lpdm.main \
	--config configs/example_mhd_january.yaml \
	--device cuda \
	--output-uri outputs/run-A \
	--start-time 2024-01-10T00:00:00Z
```

YAML schema is defined by [src/lpdm/config.py](src/lpdm/config.py). The top-level sections are `io`, `simulation`, `release`, `turbulence`, `output_grid`, `met_domain`, and `memory`. Validation includes: `simulation.length_seconds > release.duration_seconds`, strictly ascending `output_grid.z_edges_m`, and the release point lying inside `met_domain`.

Memory controls live in the `memory:` section of the YAML — `met_cache_max_hours`, `log_every_steps`, `gc_every_steps`, and three optional guard thresholds (`guard_max_rss_gib`, `guard_max_device_allocated_gib`, `guard_max_device_reserved_gib`) plus `guard_check_every_steps`. If a guard fires the run exits with a `MemoryError` and writes diagnostic metadata to `run_metadata.json`.

Outputs written under `io.output_uri`:
- `endpoint_particles.parquet`
- `trajectory_diagnostics.parquet`
- `footprints.zarr`
- `run_metadata.json`

## Deploy

```bash
chmod +x deploy.sh
PROJECT_ID=my-project REGION=us-central1 ./deploy.sh
```
