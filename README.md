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

If behavior or interfaces change, update both the implementation and the matching documentation source above in the same PR.

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m lpdm.main
```

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
python -m lpdm.main
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
    --time-start 2023-12-31T18:00:00 \
    --time-end 2024-01-01T06:00:00 \
    --lon-min -125.0 \
    --lon-max -119.0 \
    --lat-min 35.0 \
    --lat-max 41.0
```

After downloading the sample data, specify `--zarr-store data/sample_met.zarr` in your run commands or update `.vscode/launch.json` to use this local store instead of the remote `gs://...` URI.

### Running the Model

The entrypoint supports a minimal end-to-end backward trajectory run,
including met fetch, advection stepping, and output persistence.

Example:

```bash
PYTHONPATH=src .venv/bin/python -m lpdm.main \
	--zarr-store gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3 \
	--start-time 2024-01-01T00:00:00Z \
	--release-duration-seconds 3600 \
	--simulation-length-seconds 10800 \
	--release-seed 42 \
	--n-particles 2048 \
	--release-lon -122.30 \
	--release-lat 37.90 \
	--release-alt-agl-m 500 \
	--dt-seconds 300 \
	--output-uri outputs/demo-run
```

Equivalent environment-variable configuration is also supported:

```bash
export LPDM_ZARR_STORE=gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3
export LPDM_START_TIME=2024-01-01T00:00:00Z
export LPDM_RELEASE_DURATION_SECONDS=3600
export LPDM_SIMULATION_LENGTH_SECONDS=10800
export LPDM_RELEASE_SEED=42
export LPDM_OUTPUT_URI=gs://<your-bucket>/lpdm/demo-run
PYTHONPATH=src .venv/bin/python -m lpdm.main
```

Runtime timing semantics:
- `start-time`: start of the particle release window.
- `release-duration-seconds`: release window length; particles are released uniformly across this window.
- `simulation-length-seconds`: total backward integration length measured from the end of the release window.
- `release-seed` (optional): makes temporal release sampling deterministic and reproducible.

Memory controls and profiling logs:
- `--met-cache-max-hours`: cap in-memory hourly met tensors (default `2`, set `0` to disable cache).
- `--memory-log-every-steps`: emit process/device memory logs every N steps (default `10`, set `0` to disable).
- `--gc-every-steps`: run `gc.collect()` every N steps (default `50`, set `0` to disable).
- `--memory-guard-max-rss-gib`: abort early if RSS exceeds this threshold (unset by default).
- `--memory-guard-max-device-allocated-gib`: abort early if device allocated memory exceeds this threshold (unset by default).
- `--memory-guard-max-device-reserved-gib`: abort early if CUDA reserved memory or MPS driver memory exceeds this threshold (unset by default).
- `--memory-guard-check-every-steps`: check guard every N steps (default `1`).

Example with aggressive memory limits:

```bash
PYTHONPATH=src .venv/bin/python -m lpdm.main \
	--zarr-store gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3 \
	--start-time 2024-01-01T00:00:00Z \
	--release-duration-seconds 3600 \
	--simulation-length-seconds 10800 \
	--met-cache-max-hours 1 \
	--memory-log-every-steps 5 \
	--memory-guard-max-rss-gib 10 \
	--memory-guard-max-device-allocated-gib 14 \
	--memory-guard-max-device-reserved-gib 15 \
	--memory-guard-check-every-steps 1 \
	--gc-every-steps 25 \
	--output-uri outputs/memory-profiled-run
```

If the guard is triggered, the run exits with a `MemoryError` and writes
diagnostic metadata to `run_metadata.json`.

Validation requires `simulation-length-seconds > release-duration-seconds`.

Outputs written by default:
- `endpoint_particles.parquet`
- `trajectory_diagnostics.parquet`
- `run_metadata.json`

## Deploy

```bash
chmod +x deploy.sh
PROJECT_ID=my-project REGION=us-central1 ./deploy.sh
```
