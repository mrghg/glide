# GLIDE checkpoint and project context

## Instruction ownership (rationalized)

- Canonical operational coding-agent instructions live in `.github/copilot-instructions.md`.
- This file is for project history, architecture intent, and milestone context.
- Avoid duplicating detailed operational rules in this file.

## Project goal (north star)

Build a modern, highly optimized, backward-in-time LPDM for greenhouse-gas footprints that bypasses legacy CPU/I/O bottlenecks by streaming ARCO ERA5 Zarr data directly into a pure PyTorch physics engine.

## Core principles

1. Device-agnostic development with dynamic fallback (cuda -> mps -> cpu), while targeting NVIDIA L4 on GCP Cloud Run for production.
2. Batch optimization: vectorized tensor operations only in runtime-critical paths.
3. Meters, not pressure: internal particle physics must operate in geometric meters (AGL) before GPU stepping.

## Architecture summary

### `src/lpdm/release_generator.py`

- Converts measurement geometry to `(N, 4)` particle tensors `[x, y, z, weight]`.
- Column-style releases should use importance sampling with a vertical PDF based on pressure weighting and averaging kernel.

### `src/lpdm/met_reader.py`

- Streams spatiotemporal subsets from ARCO ERA5 Zarr on GCS.
- Converts geopotential to height AGL.
- Converts vertical wind from omega (`Pa/s`) to geometric vertical velocity (`m/s`) via hydrostatic approximation.
- Returns `hour_start` and `hour_end` 3D met tensors for temporal interpolation.

### `src/lpdm/gpu_engine.py`

- Pure PyTorch backward-in-time stepping on batched particle tensors.
- Uses normalized coordinates for interpolation space.
- Core methods include RK2 advection, Langevin turbulence updates, and boundary reflection.

### `src/lpdm/footprint_gridder.py`

- Accumulates on-the-fly Eulerian footprints into binned tensor grids.
- Intended to use scatter-style reductions for batched accumulation.

### `src/lpdm/output_writer.py`

- Writes model artifacts for local and remote storage.
- Primary contracts: `endpoint_particles.parquet`, `trajectory_diagnostics.parquet`, `run_metadata.json`.

### `tests/test_physics.py`

- Local validation of numerical behavior and conservation before cloud deployment.
- Includes uniform wind advection and stochastic diffusion checks.

## Milestone timeline

### 2026-04-02 baseline

- Scaffold complete: Dockerfile, deploy.sh, module stubs, README updates.
- Runtime device selection added: cuda -> mps -> cpu.
- Release classes implemented with device-aware tensor output in `src/lpdm/release_generator.py`.
- GPU engine utilities implemented in `src/lpdm/gpu_engine.py` (RK2, Langevin update, reflection, periodic diffusion helpers).
- Physics tests in `tests/test_physics.py` pass in project venv.
- Visualization helpers and starter notebook added.

### 2026-04-08 memory-hardening update

- Runtime now includes bounded met cache controls and periodic memory telemetry.
- Fail-fast memory guards added for RSS and device memory thresholds.
- Guard-triggered abort path writes diagnostic metadata before exit.

### 2026-04-28 local testing and memory-optimization update

- Fixed severe memory explosions in `src/lpdm/met_reader.py` by lazily evaluating spatial and temporal slices before executing Dask `.compute()` calls.
- Created `scripts/download_sample_cube.py` utility to download localized ARCO ERA5 data cubes for faster local testing and debugging.
- Updated documentation for local workflow and data subsetting.

### 2026-04-29 local met debug and zarr-compat update

- `scripts/download_sample_cube.py` now opens public ARCO GCS data with anonymous access via `storage_options={"token": "anon"}`.
- Added optional output format switch (`--zarr-version {2,3}`) and v3-safe write preparation by clearing inherited per-variable encodings before Zarr v3 writes.
- Updated write API usage to avoid deprecation warnings by using `zarr_format` instead of deprecated `zarr_version` keyword in `xarray.Dataset.to_zarr`.
- Added VS Code debug profile for local meteorology smoke tests in `.vscode/launch.json`:
	- `GLIDE: lpdm.main (Local Sample Met)`
	- Uses `${workspaceFolder}/data/sample_met.zarr` as `--zarr-store`.
	- Writes to `outputs/demo-run-local`.

## Next recommended task

- Replace domain-mean wind placeholder usage in runtime stepping with full spatial-temporal interpolation path.

## Known gotchas

- Running bare `pytest` may use the wrong interpreter; prefer `.venv/bin/python -m pytest`.
- In a fresh environment, install the package editable first: `uv pip install --python .venv/bin/python -e .`.
- Ensure shell PATH is refreshed if `uv` was installed in-session.
- For public GCS Zarr buckets, ADC is not required when opening with anonymous token mode (`token="anon"`).
- Zarr v3 writes can fail if source dataset encodings contain v2 codec objects (for example `numcodecs.Blosc`); clear inherited encodings before writing v3.
