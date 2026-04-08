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
```

Run the model entrypoint:

```bash
python -m lpdm.main
```

## Physics Tests

```bash
PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_physics.py
```

The suite includes:
- Uniform wind RK2 advection precision test (zero turbulence).
- Zero-wind Langevin diffusion Gaussianity test.
- Well-mixed periodic turbulence uniformity test.

Each test also verifies particle mass conservation by checking total weight.

## Minimal Trajectory Run (Local or Vertex Notebook)

The entrypoint now supports a minimal end-to-end backward trajectory run,
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
