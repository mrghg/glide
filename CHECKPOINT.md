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

### 2026-04-29 physics engine and footprint accumulation update

- Replaced domain-mean wind placeholder in `src/lpdm/main.py` with full spatial-temporal interpolation utilizing `torch.nn.functional.grid_sample`.
- Wired up `engine.update_langevin_velocity` and `engine.apply_vertical_turbulence` to apply vertical turbulence diffusion.
- Fully implemented Eulerian footprint accumulation via `FootprintGridder.accumulate()` in `src/lpdm/footprint_gridder.py` using parallel `scatter_add_` on fractional indices.
- Integrated the footprint gridder into the main event loop to generate and serialize `footprints.zarr` alongside other run artifacts.

### 2026-04-29 runtime validation and local met integrity update

- Fixed footprint time-bin indexing in `src/lpdm/main.py`; bins are now keyed from `release_end` rather than `sim_start`, so multi-hour footprint output populates all expected time slices.
- Refactored `src/lpdm/main.py` to isolate output-path handling, memory telemetry/guard metadata, hourly met fetching, and active-particle transport from the main stepping loop.
- Added a preflight meteorology time-coverage check so runs fail early with a clear error if the requested backward window exceeds available met data.
- Expanded local sample met defaults to cover two extra days of history and a wider lat/lon box for SF-area debugging.
- Hardened `scripts/download_sample_cube.py` to write into a temporary Zarr store, validate that all required variables are fully finite, and only then replace the live `data/sample_met.zarr`.
- Tightened `src/lpdm/met_reader.py` to reject any non-finite geopotential-derived AGL values instead of attempting to continue with inconsistent meteorology.
- Persisted spatial and time-bin coordinate metadata into `footprints.zarr`, and updated the demo notebook to consume those coordinates directly while remaining compatible with older stores.

### 2026-05-08 Milestone 0 closure

- Rewrote `ColumnRelease` in `src/lpdm/release_generator.py` to take altitudes (m AGL) directly and use the averaging kernel as the only sampling weight. The old API's `levels` argument was internally inconsistent (used both as a pressure-mass weight and as the particle altitude column); pressure-mass weighting is now the caller's responsibility, passed in via `averaging_kernel`. Added `tests/test_release_generator.py` covering uniform sampling, AK-biased sampling, and three input-validation paths.
- Documented footprint output units. The `FootprintGridder` module docstring and `accumulate` docstring now describe the raw accumulator value (`Σ over active particles of (weight_i × dt_step)`, dimensionality `(mass fraction) × seconds`) and the conversion path to physical sensitivity. The same description is persisted into `run_metadata.json` under `footprint_units` for both successful and memory-guard-aborted runs.
- Added `VALIDATION.md` at the repo root: full table of tests with their tolerances and seeds, an explicit list of placeholder metrics pending Milestone 1, the canonical pytest command, the sample-met end-to-end command pointer, and a workflow for adding new physics regression tests.
- Total test coverage now stands at 37 cases across `tests/test_physics.py`, `tests/test_footprint.py`, `tests/test_release_generator.py`, `tests/test_main_runtime.py`, and the pre-existing `tests/test_main_config.py`, `tests/test_output_writer.py`, `tests/test_met_reader.py`, `tests/test_download_sample_cube.py`. Suite runs in ~12 s with no network.

### 2026-05-07 Milestone 0 validation suite first wave

- Added engine-level regression tests in `tests/test_physics.py`: RK2 second-order accuracy under a linear (spatially-varying) wind, and surface-reflection edge cases including non-zero `z_surface`. The pre-existing constant-wind test only proved the scheme was exact for trivial wind; it did not distinguish RK1 from RK2.
- Added `tests/test_footprint.py` with seven unit tests for `FootprintGridder` covering binning, mass conservation, active-mask handling, out-of-bounds rejection, repeat accumulation, and the silent no-op contract for invalid time-bin indices.
- Built a synthetic `AnalyticMetReader` test fixture and refactored `_run` in `src/lpdm/main.py` to accept an injectable `reader: MetReader | None = None`. Added `tests/test_main_runtime.py` exercising the full main loop without remote ERA5: smoke run, met-coverage preflight failure, analytic backward-trajectory match for constant wind, and footprint conservation against the trajectory's active-particle-time integral.
- Fixed a precision bug in `src/lpdm/main.py`: `release_times_ts` was stored in `torch.float32`, which only resolves to ~128 s near Unix timestamps of 1.7e9. For release windows shorter than that resolution the staggered release collapsed to a few discrete times and some samples landed outside the intended window. Initially promoted to `torch.float64`, but that broke MPS (which only supports float32); the final fix stores offsets-from-release_start in float32 instead, which keeps values in `[0, release_duration_seconds)` where float32 precision is plenty.
- Retracted the earlier "RK2 temporal-order" claim. Empirical convergence and Taylor analysis confirm the call-site scheme remains globally second-order despite reusing one `alpha` for both substeps; the corresponding fix was removed from M1 scope and the M0 time-varying-wind regression case was reframed as a confirmation rather than a bug-exposure test.

## Milestone roadmap

### Milestone 0: Validation baseline and reference cases

- Build a small scientific validation suite before changing core transport physics.
- Add deterministic seeded regression cases in `tests/test_physics.py` and focused end-to-end smoke cases around `src/lpdm/main.py`.
- Define comparison metrics for plume spread, time-height structure, and footprint agreement against analytic cases and, where practical, FLEXPART/STILT-style reference behavior.
- Treat all dispersion-related metrics (vertical spread, footprint extent, plume width) as PLACEHOLDER baselines only. The current turbulence step uses hard-coded `t_lagrangian=300 s` and `sigma_w2=1.0 m^2/s^2` constants in `src/lpdm/main.py`, and there is no horizontal turbulence at all. Meaningful dispersion baselines must wait until Milestone 1 wires real turbulence parameters from the met fields.
- Implementation checklist:
	- Add synthetic met fixtures or in-memory harnesses so core transport tests do not depend on remote ERA5 access.
	- Separate validation into three layers: unit physics checks, end-to-end runtime checks, and external reference comparisons.
	- Record one canonical validation command set in this file or README so future physics changes can be checked consistently.
	- Keep all stochastic cases seeded and treat seed stability as part of the contract.
	- Resolve `ColumnRelease` semantic ambiguity in `src/lpdm/release_generator.py`: `levels` is currently used both as a pressure-weighting input and as the particle altitude column, which cannot both be correct. Decide whether the input is altitudes (m AGL) or pressures (hPa) and align the variable name, docstring, and `alt` assignment accordingly.
	- Document footprint output units explicitly in `src/lpdm/footprint_gridder.py` and `run_metadata.json`. The current accumulator sums `weight * dt_seconds` per bin; describe the resulting units and the conversion path to a sensitivity.
- Unit and regression cases to add first:
	- Constant horizontal wind backward advection: particle mean position should match the analytic trajectory within a tight tolerance.
	- Time-varying analytic wind backward advection (for example a wind that ramps linearly across the hour): confirm RK2 stays globally second-order under temporal wind variation, not just under time-invariant wind. Note: the call-site `wind_fn` in `src/lpdm/main.py` reuses one `alpha` (timestep midpoint) for both RK2 substeps, which is non-standard but Taylor analysis and empirical convergence tests show it remains globally second-order with comparable error to the textbook midpoint method.
	- Zero-wind, zero-turbulence persistence: particles should remain stationary aside from explicitly enabled physics.
	- Vertical OU/Langevin variance check: confirm the discrete OU step matches the closed-form integrated OU variance for the supplied `t_lagrangian` and `sigma_w2`. This validates the numerical implementation only; it does not validate physical realism of dispersion.
	- Surface reflection check: particles crossing below `z=0` should reflect correctly with no negative post-step altitudes.
	- Footprint conservation check: integrated footprint mass should scale consistently with active particle weights and residence time.
	- Footprint time-bin assignment check: occupancy should land in the expected hourly bins across a multi-hour run.
- End-to-end smoke cases to maintain:
	- Short local sample-met point release run that verifies `endpoint_particles.parquet`, `trajectory_diagnostics.parquet`, `footprints.zarr`, and `run_metadata.json` are all produced.
	- Local run with persisted footprint coordinates, confirming the `footprints.zarr` schema contains time-bin bounds, vertical bin bounds, and horizontal edge coordinates.
	- Invalid met coverage case that confirms preflight failure happens before stepping begins and emits a useful error.
	- Local corrupted-met rejection case that confirms non-finite geopotential-derived AGL values fail loudly rather than degrading silently.
- Reference comparison targets:
	- Analytic baselines first: constant wind, time-varying wind, no-turbulence, and simple diffusion-growth cases.
	- One canonical offline comparison case against FLEXPART or STILT once a matching forcing/release setup is defined. Defer this until after Milestone 1, since the placeholder turbulence makes any current external comparison meaningless.
	- If exact external intercomparison is not yet practical, define acceptance against self-consistent surrogate metrics first and promote to cross-model comparison later.
- Metrics to compute and track:
	- Endpoint mean position in longitude, latitude, and altitude (baselineable in M0).
	- Endpoint spread in longitude, latitude, and altitude (placeholder until M1).
	- Time-height integrated footprint structure (placeholder until M1).
	- Column-integrated footprint magnitude and spatial extent (placeholder until M1).
	- Particle survival counts and any future domain-exit or merge counts.
	- Runtime and peak memory for representative local smoke cases.
- Deliverables for Milestone 0 completion:
	- Expanded regression coverage in `tests/test_physics.py` and neighboring focused test modules.
	- At least one reproducible end-to-end validation command using `data/sample_met.zarr`.
	- A short validation note documenting expected tolerances, seeds, and which metrics are placeholders pending Milestone 1.
- Exit criteria:
	- Reproducible seeded regression fixtures for advection, surface reflection, and footprint binning/conservation.
	- Documented placeholder status for dispersion-related metrics, with reasons.
	- A documented validation workflow for future physics changes.

### Milestone 1: Full turbulence scheme

- Replace the placeholder vertical OU/Langevin path in `src/lpdm/gpu_engine.py` and the hard-coded `t_lagrangian=300` / `sigma_w2=1.0` arguments at the call site in `src/lpdm/main.py` with a scientifically grounded turbulence parameterization driven by the met fields.
- Choose and document the target formulation first (likely a FLEXPART-like or STILT-like vertical mixing scheme), then implement a first validated version before adding extensions. Do not pre-empt the formulation choice in this document.
- Add horizontal stochastic diffusion alongside the vertical scheme. The current pipeline has no horizontal turbulence, so transport spread is governed only by mean-wind shear and is systematically under-dispersed laterally. Treat horizontal diffusion as in-scope for this milestone, not a follow-on item.
- Wire the `blh` and `sp` channels (already fetched into the hourly met tensors but currently sliced off at `met_window.hour_start[:3]` in `src/lpdm/main.py`) into the turbulence parameterization. Add the additional inputs the chosen scheme requires (for example friction velocity, Obukhov length, convective velocity scale) to `src/lpdm/met_reader.py`.
- Preserve batch-friendly tensor interfaces so the runtime loop in `src/lpdm/main.py` stays vectorized.
- Exit criteria:
	- Stable long-run behavior across different boundary-layer regimes.
	- Validation against Milestone 0 reference cases, plus re-baselined dispersion metrics that previously carried placeholder status.
	- Clear documentation of assumptions, required met fields, and known limitations.

### Milestone 2: Adaptive particle aggregation

- Implement a conservative far-field particle aggregation methodology to reduce compute cost once particles move away from the release region.
- Keep strict no-merge zones near the receptor, near the surface, anywhere flow gradients or footprint sensitivity are high, and anywhere within the boundary layer. Use the `blh` channel from the met tensors so the merge boundary tracks diurnal mixed-layer evolution rather than a fixed altitude threshold.
- Preserve at minimum total mass and low-order moments during merges, and add diagnostics to quantify approximation error.
- Integrate aggregation controls into the runtime loop without breaking memory guards or output contracts.
- Exit criteria:
	- Demonstrated runtime or memory reduction versus an unaggregated baseline.
	- Bounded footprint error against Milestone 0 reference runs.
	- Diagnostics for merge counts, effective sample size, and approximation drift.

### Milestone 3: Production GPU execution

- Harden the existing device-aware runtime for real CUDA execution rather than just device selection and local MPS compatibility.
- Profile interpolation, met staging, footprint accumulation, and any host-device transfers in the current runtime.
- Address the vertical-interpolation approximation in `src/lpdm/main.py`: the `grid_sample` call treats the vertical tensor axis as linear in AGL meters using only the first/last level's spatially-averaged AGL, but ARCO ERA5 data sits on pressure levels that are roughly log-linear in altitude. Either resample the met tensor onto a uniform AGL grid before interpolation, or perform vertical interpolation in log-pressure (or pressure-level-index) space and convert at the end. Profile both options before committing.
- Benchmark on a representative NVIDIA target and tune memory behavior under the existing guardrails.
- Ensure the turbulence and aggregation implementations remain GPU-efficient and do not reintroduce hidden CPU bottlenecks.
- Exit criteria:
	- Documented single-GPU benchmark for a representative run.
	- Verified memory telemetry and guard behavior on CUDA.
	- A short performance note identifying the dominant remaining bottlenecks.

### Milestone 4: Versioned input schema and domain semantics

- Replace the current thin CLI-first configuration surface in `src/lpdm/main.py` with a versioned run schema loaded from YAML or JSON.
- Make output grid definition explicit, including horizontal domain, vertical bins, and time-bin resolution, instead of deriving them from bbox padding and fixed defaults.
- Add explicit regional-domain semantics, including what happens when particles leave the domain laterally or vertically.
- Keep preflight validation strong and user-facing so invalid or incomplete configs fail early with clear messages.
- Exit criteria:
	- One validated config model owns run definition.
	- Output grid, domain limits, release definition, and physics options are all schema-controlled.
	- The CLI becomes a thin wrapper around schema loading and overrides.

### Milestone 5: Observation and release generalization

- Expand `src/lpdm/release_generator.py` beyond single-point releases to support column releases, multi-point releases, and observation-linked release definitions.
- Connect release definitions to the new run schema so user inputs map cleanly into particle initialization.
- Ensure the output metadata clearly records which release strategy was used.
- Exit criteria:
	- Generic release definitions supported without ad hoc runtime branching.
	- Tests cover point, column, and multi-point initialization paths.
	- Run metadata is sufficient to reproduce a release setup from config alone.

### Milestone 6: Usability and operational hardening

- Revisit notebook defaults and plotting helpers so they adapt automatically to arbitrary run durations and output paths.
- Add restart/checkpoint strategy for longer or more expensive runs once turbulence and aggregation increase runtime complexity.
- Improve run metadata, diagnostics, and artifact discoverability for debugging and review.
- Exit criteria:
	- Demo notebook works across arbitrary run lengths without brittle manual edits.
	- Long runs can be resumed or at least checkpointed safely.
	- Diagnostics are sufficient to explain failures, approximation decisions, and performance regressions.

## Immediate recommendation

- Milestone 0 is complete: validation suite landed, `ColumnRelease` ambiguity resolved, footprint units documented, `VALIDATION.md` published. See the 2026-05-07 and 2026-05-08 entries above for what was added and the placeholder list.
- Move directly to Milestone 1. Treat horizontal stochastic diffusion as in-scope for M1, not a separate follow-on. The first sub-task is choosing and documenting the turbulence parameterization (Hanna, Degrazia, STILT-style, etc.) before any code change, so we can baseline the new behaviour against `VALIDATION.md`'s placeholder metrics.
- Once Milestone 1 lands, prioritize Milestone 2.
- Defer substantial GPU tuning until the turbulence and aggregation interfaces are stable enough to benchmark meaningfully.

## Operational notes

- Running bare `pytest` may use the wrong interpreter; prefer `.venv/bin/python -m pytest`.
- In a fresh environment, install the package editable first: `uv pip install --python .venv/bin/python -e .`.
- Ensure shell PATH is refreshed if `uv` was installed in-session.
- For public GCS Zarr buckets, ADC is not required when opening with anonymous token mode (`token="anon"`).
- Zarr v3 writes can fail if source dataset encodings contain v2 codec objects (for example `numcodecs.Blosc`); clear inherited encodings before writing v3.
