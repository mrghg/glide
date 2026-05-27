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

### 2026-05-27 Milestone 5 stages 2–5: multi-release execution end-to-end

Stage 1 (2026-05-26) defined the schema and batch-expansion plumbing; stages 2–5 plug those into the runtime so multi-release configs actually execute. A `kind: "periodic_point"` config with 744 hourly releases now runs as a single `python -m lpdm.main` invocation, producing one 5D footprints.zarr `(release_time, time_ago, z, lat, lon)`. The two runtime guards from stage 1 are gone.

- **Stage 2 — per-particle release-time sidecars** (`src/lpdm/release_generator.py`). Added `ParticleBatch` frozen dataclass (`particles, release_idx, release_time_offsets_s, release_window_end_offsets_s, batch_start_time`) and `generate_batch_particles(batch, *, device, dtype)`. Per-release seeds drawn on CPU with a `torch.Generator` and moved to device, mirroring the legacy code so a single-release batch is bit-equivalent. `_run` switched to consume the sidecars; the active-mask check changed from `release_offsets_s >= (cursor - release_start)` to `release_time_offsets_s >= (cursor - batch_start)` (identical evaluation for the single-release case where `batch_start == release_start`).
- **Stage 3 — per-particle active mask + per-particle time-ago bin**. Active mask now uses both bounds: `(release_time_offsets_s >= cursor_offset_s) & (release_time_offsets_s - sim.length_seconds <= cursor_offset_s)`. The lower bound is automatically satisfied for single-release runs throughout the existing loop, so no behaviour change there. `FootprintGridder.accumulate` takes per-particle `t_idx: torch.Tensor` (was a Python int); computed from `release_window_end_offsets_s - cursor_offset_s` per particle so each release measures time-ago from its own window-end. For a uniform `t_idx` tensor the scatter is bit-equivalent to the previous scalar-slice path.
- **Stage 4 — 5D footprint accumulator + Zarr writer**. Gridder ctor gained `n_releases: int = 1`; tensor shape became `(n_releases, n_time_bins, n_z, n_y, n_x)`. `accumulate` takes per-particle `release_idx: torch.Tensor` and scatters into the 5D-flat index. `OutputWriter.write_footprint_zarr` rewritten strictly for 5D with leading `release_time` dim (and a `release_duration_seconds` aux coord). `notebooks/footprint_explorer.ipynb` gained a `RELEASE_INDEX` knob; legacy 4D stores pass through unchanged.
- **Stage 5 — outer batch loop in `_run`**. Both stage-1 guards (`PointReleaseConfig`-only and "single-release-batch" assertions) deleted. The runtime now:
  1. Calls `cfg.expand_to_batches()` and validates met coverage over the **full schedule** (`max(release_window_end)` down to `min(release_window_end) - sim.length_seconds`).
  2. Allocates the gridder once with `n_releases = total_releases_across_batches`.
  3. Iterates batches. For each batch: generates the particles, computes per-batch cursor bounds (`max(window_end) → min(window_end) - sim.length_seconds`), runs the stage 3 cursor loop, accumulates endpoint particles for the end-of-run write. Met cache and memory_stats survive across batches (heavy reuse — consecutive batches' backward windows overlap by `sim.length_seconds - batch_span`).
  4. After all batches: concatenates endpoint particles across batches and writes one parquet with a new `release_idx` column (via the extended `write_particles_parquet(release_idx=…)` API); writes trajectory diagnostics with a new `batch_idx` column; writes the 5D footprint Zarr with a flattened-schedule `release_time` coord; emits a `schedule` block in `run_metadata.json` (`n_batches`, `n_releases`, `max_releases_per_batch`, `release_times`).
- **`configs/example_mhd_january_periodic.yaml`** — new example: 744 hourly Mace Head releases × 30-day backward, on the FLEXPART-aligned EUROPE grid. Designed so each `release_time` slice is directly comparable to a row of `data/FLEXPART/FLEXPART_MHD_test_202401.nc`. Carries a memory note (~23 GB for the in-memory 5D tensor at full resolution) and documents the `batch.max_releases_per_batch: 24` choice.
- **`scripts/download_sample_cube.py`** — unchanged, but the existing `EUROPE_202312.zarr` + `EUROPE_202401.zarr` stores remain the correct met inputs for the periodic example.
- **Documented discrete-time boundary effect**: the cursor loop terminus uses `min(window_end) - sim.length_seconds` and the loop condition is `t_cursor > terminus` (strict). The earliest release in each batch's backward window therefore loses one cursor step's worth of mass to the terminus exclusion. For the FLEXPART use case (sim_length=30 days, dt=60 s → 43,200 cursor steps per release) this is a ~0.002% effect; for short test scenarios it's a few percent and surfaces in the per-release equivalence tests. Captured in tests and a comment in `_run`.
- **Tests** — 8 new tests in `tests/test_main_runtime.py` covering multi-release end-to-end (shape + release_time coord), `release_idx` column on endpoint parquet, `batch_idx` column on trajectory parquet, multi-release mass conservation against the active-particle-time formula, per-release mass ≈ expected within boundary tolerance, and batch-chunking shape + per-chunking conservation. 3 new tests in `tests/test_output_writer.py` for the optional `release_idx` column (presence, shape-mismatch rejection, back-compat single-release). Stale `test_runtime_rejects_multi_release_with_clear_message` removed. Pre-existing single-release conservation test (`test_footprint_total_matches_active_particle_time`) tolerance loosened 1e-5 → 1e-3 to be robust to test-order RNG bleed. Total test count 146 (up from 138); suite runs in ~10 s; passed three consecutive full-suite runs without flakes.

### 2026-05-27 M4 out-of-domain particle handling: drop-and-count + kill-on-exit

Closed the M4 "explicit handling for particles leaving met_domain" gap, prompted by the user asking whether escaped particles should be killed for efficiency. They should — and it was both an efficiency and a (latent) correctness issue.

- **The problem**: `GPUEngine.normalize_particle_coordinates` clamps normalized coords to `[-1, 1]` ([gpu_engine.py:102](src/lpdm/gpu_engine.py#L102)), so a particle that advects outside `met_domain` isn't producing NaNs — it gets pinned to the grid edge and keeps being pushed by edge winds every timestep for the rest of the (up to 30-day) integration. Pure wasted compute: it can't contribute to the footprint either (outside `met_domain ⊇ output_grid`, so the gridder's `valid_mask` already drops it). For a backward Europe run a large fraction of particles escape long before 30 days, so the active set was staying near full size when it should shrink.
- **Fix (drop-and-count)** in `src/lpdm/main.py`:
  - New `_within_met_domain(particles, cfg)` predicate: inside the lon/lat bbox (inclusive) and `alt <= alt_max_m`. No lower-altitude kill — the `z=0` floor is handled by surface reflection.
  - Per batch, a persistent `alive` bool mask. The cursor active mask is now `release_active & alive`. After each step's advect+turbulence, particles that left `met_domain` have their `alive` bit cleared (`alive &= ~(active_mask & ~within)`); they're never advected or accumulated again. Active set shrinks as particles escape → directly less per-step compute.
  - Kill is checked *after* accumulation, but escaped particles contribute 0 that step anyway (outside `output_grid` → dropped by `valid_mask`), so footprints are unchanged. For `met_domain ⊇ output_grid` (the normal case) this is purely an efficiency win; for a misconfig where `output_grid` pokes outside `met_domain` it also correctly stops edge-clamped-met contributions.
  - Killing on `met_domain` exit, **not** `output_grid` exit: `met_domain` is usually slightly wider, and a particle in that buffer zone still has valid met and may re-enter the output grid further back in time.
- **Diagnostics**: `trajectory_diagnostics.parquet` gains `escaped_this_step` and `alive_particles` columns (the M4-requested per-step escaped count). `alive_particles` is computed as `n_particles - batch_escaped_total` (no extra device sync). `run_metadata.json`'s `schedule` block gains `escaped_met_domain_total`.
- **Bit-equivalence**: in-domain runs (all existing tests — gentle winds, short windows, generous domains) produce zero escapes, so `alive` stays all-True and behaviour is identical. Verified: the 13 prior runtime tests pass unchanged.
- 3 new tests in `tests/test_main_runtime.py`: `_within_met_domain` predicate unit test (edge-inclusive, alt_max, no lower kill); strong-wind run where particles escape a small domain (escaped counts > 0, `alive_particles` monotonically non-increasing, active set collapses by run end); and a no-escape run confirming `alive` stays full and `escaped_met_domain_total == 0`. Total test count 154.
- **Not done (optional M4 follow-up)**: abort-above-threshold safeguard — warn or fail if the escaped fraction exceeds a configurable limit (signals an undersized `met_domain`). Drop-and-count + the diagnostic counts cover the immediate need; a threshold is a small additive knob if we want it.

### 2026-05-27 M5 streaming footprint writes: per-batch Zarr regions

Closed the streaming follow-up flagged across the stage-5 entries. The runtime no longer holds the whole schedule's footprint tensor in memory; each batch writes its slice to disk and frees its gridder, so peak footprint memory is **one batch's** tensor, not the full schedule's. This is what the user's question ("write to disk after each batch and clear that part of the memory?") asked for, and it removes the practical ceiling on `n_releases`.

- **`OutputWriter` streaming pair** ([src/lpdm/output_writer.py](src/lpdm/output_writer.py)):
  - `create_footprint_store(path, *, shape, coords, attrs)` — writes coords + array metadata for the full `(total_releases, T, Z, Y, X)` store, with the footprint data variable as a lazy `dask.array.zeros` written `compute=False`. The full tensor is never materialised; chunked one release per chunk so region writes touch disjoint chunks.
  - `write_footprint_region(path, footprint, *, release_start, release_stop)` — writes one contiguous block into `release_time[release_start:release_stop]` via xarray `to_zarr(region=…)`. Coord-free dataset (data only), so it's a pure region update.
  - `write_footprint_zarr` (the single-shot path) is kept and now shares a `_split_footprint_coords` helper with the streaming methods. Still used by tests and available for direct one-tensor writes.
- **`_run` refactor** ([src/lpdm/main.py](src/lpdm/main.py)): the gridder is now allocated **per batch** with `n_releases=len(batch.releases)` instead of once with `n_releases=total_releases`. On the first batch the runtime builds full-schedule coords (geometry is batch-independent) and calls `create_footprint_store`; every batch writes its region after the cursor loop and then `del gridder`. `generate_batch_particles` still emits global `release_idx`; the runtime remaps to batch-local (`release_idx - batch.releases[0].release_idx`) for the per-batch gridder and offsets back to global for the region write (`release_start = batch.releases[0].release_idx`). Batches are contiguous slices of the global release list (see `expand_to_batches`), so the offset arithmetic is exact.
- **Memory guard interaction**: the `FootprintGridder` guard now sees the per-batch `n_releases`, so it bounds the *per-batch* tensor (the thing actually allocated). A high-`n_time_bins` config that would have tripped the guard for the whole schedule may now fit per-batch; if a single batch is still too big, reduce `batch.max_releases_per_batch`.
- **Abort behaviour**: if a memory guard trips mid-run, batches already streamed are on disk; only the in-flight batch's region stays zero. Strictly better than the previous all-or-nothing end-of-run write.
- **Example config** ([configs/example_mhd_january_periodic.yaml](configs/example_mhd_january_periodic.yaml)): memory note rewritten — peak is now `max_releases_per_batch × n_time_bins × …` (≈ 33 MB for the shipped 24/batch × 1-bin values) rather than the full-schedule figure.
- 3 new tests in `tests/test_output_writer.py`: create-store-then-region-writes (3 batches into disjoint slices, verifies per-release content + coord survival), region shape-mismatch rejection, non-5D store-shape rejection. The existing multi-release runtime tests (`tests/test_main_runtime.py`) now exercise the streaming path end-to-end — the batch-chunking test in particular confirms region writes across multiple batches reconstruct the same per-chunking-conservative footprint. Total test count 151; suite ~11 s.
- **Still open**: the dense per-batch tensor is the remaining memory term. For pathological configs (huge `n_time_bins` × big grid × big batch) the per-batch tensor can still be large — sparse accumulation or chunked-within-batch time handling would be the next lever, but it's not needed for the FLEXPART workflow.

### 2026-05-27 M5 stage 5 follow-up: gridder memory guard + corrected periodic-config sizing

First end-to-end run of `configs/example_mhd_january_periodic.yaml` failed with `RuntimeError: ... can't allocate memory: you tried to allocate 712673510400 bytes`. Two errors in the stage 5 work landed earlier today contributed: the example config had `n_time_bins: 720` (one hourly bin per hour of the 30-day backward window) instead of the time-integrated `n_time_bins: 1` that the FLEXPART comparison actually needs, and there was no upfront check that the gridder tensor would fit in memory.

- **Example config** ([configs/example_mhd_january_periodic.yaml](configs/example_mhd_january_periodic.yaml)): `n_time_bins` set to 1 (FLEXPART's `srr` is time-integrated, and `to_stilt_surface_footprint` integrates over `time_ago` anyway). Tensor footprint is now `720 × 1 × 3 × 293 × 391 × 4 B ≈ 0.92 GiB`. Memory note rewritten to lay out the actual `(n_releases × n_time_bins × n_z × n_y × n_x) × 4 B` arithmetic with worked examples for `n_time_bins = 1 / 24 / 720` (the last is the failure case).
- **`FootprintGridder` memory guard** ([src/lpdm/footprint_gridder.py](src/lpdm/footprint_gridder.py)): `__init__` now computes `n_elements × bytes_per_element` and raises `MemoryError` with a clear "requested X GiB / cap Y GiB" message if the tensor would exceed the cap. Default cap 32 GiB (`MAX_FOOTPRINT_GIB`); overridable per-run via `LPDM_FOOTPRINT_MAX_GIB` env var for users who genuinely need a bigger allocation. The message explicitly mentions the FLEXPART-`n_time_bins=1` workaround and the streaming-per-batch follow-up.
- **Streaming per-batch Zarr writes** were the planned follow-up here — now done, see the later same-day "M5 streaming footprint writes" entry above. The guard now bounds the per-batch tensor (the thing actually allocated) rather than the full schedule.
- 2 new tests in `tests/test_footprint.py`: oversized-tensor rejection with a representative FLEXPART-scale spec (`MemoryError`, message contains `"Footprint tensor would require"`); `LPDM_FOOTPRINT_MAX_GIB` env var lifts the cap (uses `monkeypatch.setenv`). Total test count 148; suite still <30 s.

### 2026-05-26 Milestone 5 stage 1: multi-release schema + batch expansion

Foundation for running many releases in one process — needed so a month of MHD hourly footprints (720 releases) doesn't become 720 separate `python -m lpdm.main` invocations. **Schema layer only**; the runtime still executes single-release configs and refuses multi-release ones with a clear `NotImplementedError`. Stages 2–5 will lift that guard.

- **Discriminated-union release schema** in `src/lpdm/config.py`. Three variants under one `ReleaseConfig` discriminator on `kind`:
  - `kind: "point"` — `PointReleaseConfig`, field-identical to today's schema. Existing YAML and all `cfg.release.lon / .lat / .alt_agl_m / .duration_seconds / .n_particles / .seed` accesses continue working unchanged.
  - `kind: "periodic_point"` — `PeriodicPointReleaseConfig` with `point: PointSpec, start_time, period_seconds, n_releases, duration_seconds, n_particles_per_release, seed`. Covers the "every hour for N days" site workflow in 6 YAML lines.
  - `kind: "point_schedule"` — `PointScheduleReleaseConfig` with explicit `times: list[datetime]`. Foundation for the eventual satellite multi-point case (adds a per-time `points` list in a later stage).
- **`BatchConfig` section** with `max_releases_per_batch: int = 24`. Single knob controlling how multi-release schedules are chunked for execution. Default 24 = one calendar day for hourly cadence — matches the natural unit users think in for site footprints and keeps memory predictable (`max_releases_per_batch × n_particles_per_release × particle_state_bytes`).
- **Internal expansion types**: `ConcreteRelease` and `ReleaseBatch` (frozen dataclasses). `RunConfig.expand_to_batches() → list[ReleaseBatch]` produces a uniform per-release representation regardless of source `kind` — `kind: "point"` expands to one batch with one release at `simulation.start_time`. The runtime stages 2–5 will iterate this list directly, blind to which YAML kind produced it.
- **Seed derivation**: when the top-level `seed` is non-null, each release `i` gets `seed_i = seed + i`. When null, all releases get `None` (non-deterministic). For `kind: "point"` this means `seed_0 == seed`, preserving today's single-run reproducibility.
- **Polymorphic cross-section validation**: `_check_release_inside_met_domain` and `_check_simulation_length_vs_release` now use a `_release_point()` helper that returns `(lon, lat, alt_agl_m)` from any variant.
- **Runtime guard**: `_run()` in `src/lpdm/main.py` raises `NotImplementedError("multi-release ... not yet wired up")` for non-`PointReleaseConfig` releases. One-line defensive change so the schema can be exercised in tests and downstream tooling can iterate while the runtime work is still in progress.
- 13 new tests in `tests/test_main_config.py` cover all three variants' parse + validate, `expand_to_batches` behaviour for each kind, seed derivation (including null), chunking 24 releases into 3 batches of `[10, 10, 4]` at `max_releases_per_batch=10`, contiguous `release_idx` across batch boundaries, unknown-kind rejection, polymorphic cross-section validation (out-of-domain + length-vs-duration for the periodic variant), empty-times rejection, `BatchConfig` defaults + override, and the runtime guard's clear message. Total test count 121 (up from 108); suite still <10 s.
- **Remaining stages** (one PR each):
  - Stage 2: per-particle release-time + release-idx tagging (`PointRelease.generate` + sidecar tensors).
  - Stage 3: per-particle active mask + per-particle time-ago bin in the runtime cursor loop.
  - Stage 4: 5D footprint accumulator and Zarr writer (`(release_time, time_ago, z, lat, lon)`, chunked one release per chunk to match the FLEXPART fixture layout).
  - Stage 5: outer batch loop in `_run`, met-cache carry-over between consecutive batches, per-batch endpoint/diagnostic flushing.
- **Out of scope (deferred)**: multi-point releases per release-time (the satellite column case) — same plumbing extended in a follow-up, since `release_idx` already discriminates per release row and adding a points list per row is the only schema change needed.

### 2026-05-25 Two memory bugs in the met reader + viz stack rewrite

First multi-month EUROPE comparison run consumed >200 GB of RAM. Two distinct bugs in `src/lpdm/met_reader.py`, plus a complete rewrite of the demo notebook stack.

- **Bug 1 (dominant, pre-existing): `chunks=None` loaded the entire Zarr as numpy.** `xr.open_zarr(..., chunks=self.chunk_overrides or None)` — with the default empty `chunk_overrides`, `{} or None` evaluates to `None`, which in xarray means "skip dask, materialise into numpy". The 58 GB local sample cube hadn't tripped this because it fit; the 116 GB stitched EUROPE pair did not. Fixed by switching to `chunks="auto"` in both single- and multi-store branches of `_open_dataset`. Took `_open_dataset()` from 320 s / 326 GB down to **1.7 s / 507 MB**; steady-state RSS across 5 fetches went from "unbounded" to **3.4 GB** with the existing `met_cache_max_hours=2` policy. ~100× memory reduction.
- **Bug 2 (latent, would have surfaced after Bug 1 was fixed): non-monotonic longitude.** The downloaded EUROPE cubes have `lon = [262..359.75, 0..39.5]` because the source ARCO data is in 0..360 and the EUROPE bbox crosses Greenwich. Pandas can't binary-search a non-monotonic 1D index, so every `ds.sel({lon: slice(...)})` in `_slice_spatial_temporal` either raised or skipped chunk pruning. Fixed by adding `_ensure_monotonic_longitude(ds)` to `_open_dataset` — maps `lon >= 180` to `lon - 360` via lazy `assign_coords` (no data move for the common single-wrap case), with an `argsort` fallback for scrambled layouts. The same normalisation now runs at *write* time in `scripts/download_sample_cube.py` so future cubes are stored cleanly. Existing 116 GB on disk doesn't need re-downloading.
- 4 new tests in `tests/test_met_reader.py` cover the wrap normalisation: convention-swap remap, end-to-end fetch on a wrapped-lon store, no-op on already-monotonic, argsort-fallback path. Total test count now 108.
- **Notebook stack rewrite.** The 391×293 EUROPE footprint exposed that `px.choropleth_map` doesn't scale past ~5k cells — it sends a polygon-per-cell to the browser. Tried `go.Densitymap` (still wrong: it's a scatter density estimator, gave sparse dots). Settled on **hvplot + holoviews + bokeh + geoviews + jupyter_bokeh**, with `hvplot.image` (one raster per frame, not 100k polygons) over a `CartoLight` tile basemap. Deliberately `project=False` to avoid per-frame cartopy reprojection — a 40× speedup that holds because we're plotting at mid-latitudes where the basemap's Web Mercator transform aligns naturally with Plate Carrée data. EUROPE render: ~6 s, 118 KB HTML. Datashader is intentionally not included; the same code path adds `rasterize=True` as a one-flag upgrade if grids ever exceed 1M cells.
- `notebooks/demo_run_local_footprints.ipynb` (plotly) deleted. Replaced with `notebooks/footprint_explorer.ipynb` — minimal 8-cell notebook: load → animated single-level map → time-integrated single-level map. Polished plots only; time-altitude and total-by-time charts are deferred until needed.
- **Dependency cleanup.** Removed unused `pydeck>=0.9` (zero callers anywhere). Removed `plotly>=5.22` and deleted `src/lpdm/visualize.py` together — the module had zero callers, so the plotly dep was only there to support dead code. Added `hvplot>=0.10`, `geoviews>=1.12`, `jupyter_bokeh>=4.0`. Kept `matplotlib>=3.9` for the two diagnostic notebooks (`era5_retrieval_test.ipynb`, `zarr_store_inspector.ipynb`).

### 2026-05-24 Multi-store reader: stitching monthly EUROPE Zarrs along time

- Closed the "single `zarr_store` URI" follow-up flagged in the previous 2026-05-24 entry. `ArcoEra5ZarrReader.__init__(zarr_store=...)` now accepts a single URI, a local glob pattern (e.g. `~/data/arco-era5/EUROPE_*.zarr`), or an explicit list of URIs. Implementation lives in `_resolve_stores` (static helper: `~`/`$VAR` expansion, glob expansion for local paths only, dedupe-preserving-order, empty/blank rejection) and a rewritten `_open_dataset` that opens each store lazily, sorts by first timestamp, verifies `lat`/`lon`/`level` coords match across stores (clear error if not), concatenates along `time` with `join="exact"`, and collapses duplicate timestamps from month-boundary overlap by keeping the first occurrence. `gs://` URIs are passed through verbatim — globbing remote prefixes is not supported, callers must list them explicitly.
- `IOConfig.zarr_store` schema relaxed to `str | list[str]` with a field validator that rejects empty lists, empty strings, and non-string list entries. Schema is otherwise untouched; existing single-string YAML configs validate as before.
- `configs/example_mhd_january.yaml` repointed from the GCS URI to `~/data/arco-era5/EUROPE_*.zarr`, with a comment block showing the explicit-list form for `gs://` deployments. Local user has already downloaded both `EUROPE_202312.zarr` and `EUROPE_202401.zarr` (~58 GB each), so the 30-day MHD backward run can now resolve met across the December/January boundary.
- 10 new tests (7 in `tests/test_met_reader.py`, 3 in `tests/test_main_config.py`) cover: remote URI pass-through, empty-list rejection, blank-entry rejection, dedupe order preservation, local glob expansion, glob-with-no-matches error, get_time_coverage across stitched stores, fetch_hourly_window spanning both stores, duplicate-timestamp collapse, and mismatched-coord rejection. Total test count now 104; suite runs in ~29 s.
- **What's not done** (intentional, separate change): the per-fetch open-stitch-close pattern is preserved — opening multiple Zarrs every hour is wasteful but matches the existing single-store behaviour. Add a session-level dataset cache only if profiling on the real 30-day run shows it dominates.

### 2026-05-24 Meteorology archive convention: EUROPE_YYYYMM monthly Zarrs

- Added the start of a real-world FLEXPART comparison workflow. Reference fixture `data/FLEXPART/FLEXPART_MHD_test_202401.nc` (Mace Head, January 2024, 96 hourly footprints subset to days 1/10/20/30, ~5 MB). Each footprint is a 30-day backward integration, so a full comparison needs ~62 days of met covering both Dec 2023 and Jan 2024.
- Decision: run the first comparison case locally rather than building Cloud Run infra now. User will execute the downloads on a machine with sufficient disk (full FLEXPART EUROPE domain × 37 pressure levels × 1 month is ~80 GB uncompressed, ~25-30 GB on disk per month).
- **Met archive convention** (refactored `scripts/download_sample_cube.py`):
  - Named-domain mode: `--domain EUROPE --year-month 202401` → `data/era5/EUROPE_202401.zarr`. The `DOMAINS` dict at the top of the script registers EUROPE with the full FLEXPART-fixture extents (lon -98 to 39.5, lat 10.6 to 79.2). Add new domains there rather than at call sites.
  - One Zarr per month, not one big Zarr per multi-month span: better resumability (only re-download a failed month), self-documenting on disk, shareable per-month, and matches ERA5's natural temporal chunking. Stores include `glide_domain`/`glide_year_month`/`glide_source_store`/`glide_domain_description` attrs so a Zarr is self-identifying.
  - Ad-hoc subset mode preserved for SF-style smoke tests (`--out-path` + explicit time/lon/lat flags). Dispatcher refuses to mix modes.
  - **OOM fix**: `_validate_written_store` switched from `np.asarray(ds[var].values)` (which loads each variable fully) to `np.isfinite(ds[var]).all().compute()` (dask-streaming). Required for multi-GB stores; previous version would have crashed before validation finished.
- Added 10 unit tests for the new helpers (`_resolve_year_month_window`, `_resolve_domain_bbox`, dispatcher mode-mixing rejection, leap-Feb handling). Total test count 88; existing atomic-replace test untouched. Suite still <12 s.
- README's "Downloading Local Sample Data" section rewritten to show both modes (named-domain canonical, ad-hoc legacy) and to flag the per-month disk footprint up front.
- **Open follow-up** before the first multi-month comparison run can execute: `ArcoEra5ZarrReader` currently opens a single `zarr_store` URI. To consume `EUROPE_202312` + `EUROPE_202401` together it needs either a glob/list of stores or a thin merge step. Small change; not done yet — wire it up when the downloaded archives are in hand.

### 2026-05-24 Milestone 4 (most of): YAML run config + output_grid / met_domain split

- Replaced the argparse-and-env-var configuration surface with a single pydantic-validated YAML schema in `src/lpdm/config.py`. Top-level `RunConfig` composes `IOConfig`, `SimulationConfig`, `ReleaseConfig`, `TurbulenceConfig`, `OutputGridConfig`, `MetDomainConfig`, and `MemoryConfig`. All sub-models are frozen with `extra="forbid"`. Cross-section model validators enforce `simulation.length_seconds > release.duration_seconds`, strictly ascending `output_grid.z_edges_m`, ascending `lon_bounds` / `lat_bounds`, and the release point lying inside `met_domain`.
- The output grid is now a first-class concept independent of the met fetch bbox: `output_grid` carries `lon_bounds, lat_bounds, n_x, n_y, z_edges_m, n_time_bins` and feeds straight into `FootprintGridder`. The previous derivation from `release_point ± bbox_pad_*` and the hardcoded 0.25° resolution is gone. This means GLIDE can accumulate footprints directly onto a reference grid (e.g. FLEXPART's 391×293) — `lpdm.comparison.regrid_conservative` is kept in the module for off-grid cases but dropped from the documented comparison workflow.
- `met_domain` (lon/lat bounds + alt_max_m) replaces `bbox_pad_*` and the per-step active-particle bbox in `_get_hourly_met_window`. Every per-hour fetch uses the same fixed bbox: predictable memory, simpler code. The implicit z floor is 0 m AGL.
- Shrunk the CLI to `--config <yaml> [--device …] [--output-uri …] [--start-time …]`. The three overrides are the things that legitimately change between runs of the same physics config; everything else lives in YAML. `RunConfig.with_overrides()` applies them functionally on the frozen model.
- Two example configs ship with the repo: `configs/example_mhd_january.yaml` (Hanna scheme, FLEXPART-aligned 391×293 grid, MHD release — wired to `data/FLEXPART/FLEXPART_MHD_test_202401.nc` for cell-for-cell comparison) and `configs/local_smoke_test.yaml` (placeholder scheme, `data/sample_met.zarr`, used by the README quick-start and the VS Code launch profile).
- `tests/test_main_config.py` rewritten against `RunConfig.model_validate({...})` and `RunConfig.from_yaml(...)`; new tests cover cross-section length validation, z-edges ordering, release-outside-met-domain rejection, `extra="forbid"`, `with_overrides`, and YAML round-trip. `tests/test_main_runtime.py` keeps its flat-keyword `_make_run_config` helper but rebuilds the nested dict internally and accesses fields via the new attribute paths (`cfg.simulation.dt_seconds`, `cfg.release.lon`, etc.). Total test count: 78 (up from 74), suite still <12 s.
- Added `pydantic>=2.7`, `PyYAML>=6.0`, `h5netcdf>=1.3`, `h5py>=3.10` to `requirements.txt`. The latter two were needed to open the FLEXPART `.nc` reference fixture.
- Updated `README.md`, `VALIDATION.md`, and `.vscode/launch.json` to use `--config`. Removed the env-var-configuration section from the README. The FLEXPART/NAME/STILT comparison workflow in `VALIDATION.md` now points at `configs/example_mhd_january.yaml` and notes that direct accumulation into the reference grid eliminates the regridding step.
- **Gaps before M4 closes:**
  - No `schema_version` field on the YAML; "versioned" in the M4 brief isn't satisfied yet. Add one once the schema is stable enough that we'd accept the migration debt of breaking it.
  - No explicit handling for particles leaving `met_domain`. `grid_sample` silently extrapolates at the boundary and the gridder's `valid_mask` silently drops out-of-bounds positions from footprint accumulation. Decide policy (clip-and-flag, drop-and-count, or abort on threshold) and surface a diagnostic — the count of escaped particles per step belongs in `trajectory_diagnostics.parquet`.

### 2026-05-08 Configurable vertical bin edges (b)

- Replaced the `FootprintGridder` API of `(z_bounds, n_z_bins)` with a single `z_edges_m` argument: a strictly-ascending sequence of bin edges in m AGL. Each adjacent pair defines one bin, so non-uniform layouts like `(0, 40, 1000, 5000)` are first-class — surface layer + mixed layer + free troposphere with no abuse.
- Vertical binning in `accumulate` switched from uniform-fraction-floor to `torch.bucketize` against the edge tensor (kept on the gridder's device to avoid host syncs). Below the first edge → bin index −1, at-or-above the last edge → bin index = `n_z`; both filtered by the existing `valid_mask`.
- Added `z_edges_m: tuple[float, ...]` to `RunConfig` and a `--z-edges-m` CLI flag (`nargs="+"`, default `(0, 1000, 2000, 3000, 4000, 5000)` matching the previous uniform layout). `LPDM_Z_EDGES_M` env mirror takes a comma-separated list. Default behaviour is unchanged; FLEXPART/NAME-style comparison runs use e.g. `--z-edges-m 0 40 1000 5000`.
- `_run` now wires `cfg.z_edges_m` into the gridder; `_build_footprint_dataset_metadata` reads `gridder.z_edges` directly so the persisted `z_bottom_m` / `z_top_m` coords are exact non-uniform edges. The demo notebook already reads these coords without assuming uniformity.
- Three new tests in `tests/test_footprint.py`: non-uniform binning correctness (one particle per asymmetric bin lands in the right z slot), constructor rejection of non-ascending edges, and rejection of too-few-edge lists. Existing tests updated to the new API. Total test count now 74.

### 2026-05-08 Footprint comparison utilities (a + c)

- Added `src/lpdm/comparison.py` with two post-processing helpers, kept out of the main runtime so they can grow at comparison time without affecting steady-state runs.
- `to_stilt_surface_footprint`: converts the raw 4D GLIDE residence-time footprint (`s` per cell) to STILT-style surface sensitivity in `m**2 s mol**-1` (equivalently `(mol/mol)/(mol/m**2/s)`) per Lin 2003 Eq. 5. Surface-layer integration uses depth-weighted overlap so bins that don't exactly match the chosen `surface_layer_depth_m` are credited proportionally; an exact match (e.g. running the model with a 0–40 m bottom bin) avoids the approximation entirely. Records the conversion parameters in `attrs` for downstream traceability.
- `regrid_conservative`: pure-NumPy area-weighted mass-conservative regridder for rectangular lat/lon grids. Uses spherical-cell areas (`(sin(lat_top) - sin(lat_bot)) × dlon_rad`) so high-latitude cos-lat curvature is respected. Factors into 1D weight matrices and applies them with an `einsum` so leading dims (`time_ago`, `z_bin`) are preserved without explicit loops. No `xesmf` / `ESMPy` dependency. Rectangular-grid only — sufficient for FLEXPART and NAME outputs.
- 14 new tests in `tests/test_comparison.py` covering exact/partial surface-layer overlap, integrated-vs-time-resolved STILT conversion, attrs propagation, identity / coarsen / refine regridding, redistribution proportionality, conservation at high latitudes, multi-dim preservation, out-of-extent zeroing, and input validation. Total test count now 71.

### 2026-05-08 Milestone 1 steps 6 + 7: HannaScheme implementation

- Added `engine.apply_horizontal_turbulence(particles, u_prime, v_prime, dt, *, backward)` primitive, mirroring `apply_vertical_turbulence` with the cos-lat correction on lon. Engine still owns the geometric application of perturbation velocities; schemes own the math of computing them.
- Implemented `src/lpdm/turbulence/hanna.py` with the full Hanna 1982 / FLEXPART scheme:
  - Free-function physics (`coriolis_parameter`, `air_density`, `obukhov_length`, `convective_velocity`, `surface_layer_sigma_w`, `in_bl_sigma_TL`) for testability in isolation. All vectorize over per-particle tensors.
  - Three in-BL stability regimes (stable/neutral/unstable) selected per particle via `torch.where` with the FLEXPART `h/L` thresholds.
  - Surface-layer override for `z < 0.1 h` (regime-specific σ_w; σ_u/σ_v from in-BL formula at clamped z; T_L = κ z / σ).
  - Above-BL constant-K placeholder (σ = 0.1 m/s, T_L = 100 s) per spec, with the N²-refinement noted as M1.x in `VALIDATION.md`.
  - FLEXPART piecewise-homogeneous treatment (no explicit Thomson 1987 drift).
  - Numerical floors on σ, T_L, u\*, BLH to keep the OU update sane at z=0 and in calm conditions.
- `HannaScheme.required_met_keys() = ("t", "ustar", "shf")`. The runtime's union with the baseline (`u, v, w, blh, sp`) now flows through the reader and into the channel tensor automatically.
- `HannaScheme.step()` does the full pipeline: 2D bilinear interpolation of all surface fields (and T at lowest model level) at particle (lon, lat) → stability → in-BL formulae → SL override → above-BL override → numerical floors → OU update for u'/v'/w' → vertical+horizontal displacement → surface reflection.
- Extended `AnalyticMetReader` in `tests/test_main_runtime.py` to emit `t`, `ustar`, `shf` with configurable scalars so Hanna runs are testable without remote ERA5.
- Added `tests/test_hanna.py` with eleven unit tests (registry, Coriolis, density, Obukhov L, convective velocity, in-BL formulae per regime, regime selection, surface-layer formulae, above-BL constants).
- Added three end-to-end Hanna tests in `tests/test_main_runtime.py`: smoke run, mean-trajectory preservation under constant wind, non-trivial vertical spread under convective conditions. Total test count now 57.
- The default `--turbulence-scheme` stays `placeholder_constant_ou` until an external (FLEXPART/STILT) comparison validates the Hanna behaviour. CLI `--help` now lists both registered schemes.
- Updated `VALIDATION.md` with all new tests and reframed the "placeholder pending M1" section to "pending external validation" — local-dispersion metrics are now Hanna-driven but the FLEXPART/STILT cross-comparison case is still future work.

### 2026-05-08 Milestone 1 steps 4 + 5: ustar/shf met inputs

- Extended `ArcoEra5ZarrReader` with the met inputs Hanna needs: added `ustar -> friction_velocity` and `shf -> surface_sensible_heat_flux` to `DEFAULT_VARIABLE_MAP`.
- Made the channel set pluggable: `channel_names` is now a constructor argument (defaults to `DEFAULT_CHANNEL_NAMES = ("u", "v", "w", "blh", "sp")`). `required_variable_keys` is derived as `channel_names ∪ _DERIVATION_KEYS = ("t", "z", "z_sfc")`, so unused fields aren't fetched. The constructor rejects channel names with no entry in `variable_map`.
- Added units validation for `ustar` (m/s) and `shf` (W/m^2 instantaneous OR J/m^2 accumulated). Accumulated SHF is auto de-accumulated by dividing by `accumulation_seconds` (default 3600 s for hourly ERA5). New `_convert_shf_to_w_per_m2` helper alongside the existing omega->w conversion.
- Wired `_run` in `main.py` to compute the required channel set as `("u", "v", "w", "blh", "sp") ∪ scheme.required_met_keys()` and pass it to the reader. Placeholder runs unchanged; Hanna will pick up `ustar`, `shf` automatically once the scheme declares them.
- Updated `scripts/download_sample_cube.py` so locally-cached cubes now also fetch `friction_velocity` and `surface_sensible_heat_flux` from ARCO ERA5. Existing local cubes will need re-running once Hanna lands; placeholder runs continue to work against the old cube.
- Added three tests in `tests/test_met_reader.py`: extended channel_names brings ustar into the tensor and de-accumulates J/m^2 SHF to 100 W/m^2; instantaneous W/m^2 SHF passes through unchanged; constructor rejects channel names absent from variable_map. Total test count now 42.

### 2026-05-08 Milestone 1 step 3: named met-channel accessor

- Added `channel_names: tuple[str, ...]` field to `HourlyMetTensors` plus `channel(name)` and `channels(*names)` accessor methods. `channels(*names)` returns stacked tensors in the requested order, not the underlying tensor order. Unknown channel names raise `KeyError` with the available channels listed.
- Promoted the previously local `logical_keys = ("u", "v", "w", "blh", "sp")` tuple in `ArcoEra5ZarrReader._dataset_to_channel_tensor` to a class-level `CHANNEL_NAMES` constant. `fetch_hourly_window` now populates `HourlyMetTensors.channel_names` from it.
- Migrated the `met_window.hour_start[:3]` / `hour_end[:3]` slice site in `_advect_active_particles` to `met_window.channels("u", "v", "w")`. Schemes will use the same accessor pattern when reading their declared met inputs (e.g. `met.channel("ustar")`).
- Added two unit tests in `tests/test_met_reader.py`: positive-path matching against positional indexing and order preservation; error-path KeyError on unknown channel. Total test count now 39.

### 2026-05-08 Milestone 1 step 1: turbulence subpackage scaffold

- Wrote `docs/turbulence.md` capturing the modular architecture (`TurbulenceScheme` ABC, registry, `lpdm/turbulence/{base,placeholder,hanna}.py`), the Hanna 1982 / FLEXPART formulation per stability regime, the day-1 above-BL constant-K placeholder, surface-layer override, drift handling (FLEXPART piecewise-homogeneous, no explicit Thomson 1987 drift), and the M1 implementation/validation plan. Linked from `README.md` Documentation Governance.
- Created `src/lpdm/turbulence/` subpackage. `base.py` provides the `TurbulenceScheme` ABC and a `name`-keyed registry (`register_scheme` decorator, `get_scheme(name)`, `list_schemes()`); `placeholder.py` extracts the M0 constant-OU behaviour into `PlaceholderConstantOU`.
- Refactored `src/lpdm/main.py` to dispatch turbulence through the scheme: `_advect_active_particles` is now advection-only and returns the temporal interpolation weight `t_alpha` so the scheme can re-use it; the runtime loop calls `scheme.step(...)` after advection. Added `--turbulence-scheme` CLI flag (and `LPDM_TURBULENCE_SCHEME` env mirror) with default `placeholder_constant_ou`. `_run` accepts an optional `scheme` parameter alongside the existing `reader` parameter so tests can inject schemes directly.
- All 37 existing tests pass against the refactored runtime; behaviour against the placeholder scheme is bit-equivalent to the pre-refactor M0 path.

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

- **Milestone status:** M0 complete. M1 implementation-complete (default still `placeholder_constant_ou` pending external validation). M4 nearly complete — schema, output_grid/met_domain split, CLI shrink, and out-of-domain particle handling (drop-and-count, 2026-05-27) all landed; the only remaining gap is the `schema_version` field. **M5 stages 1–5 complete (2026-05-26 + 2026-05-27)**: multi-release configs now execute end-to-end, producing a single 5D `footprints.zarr` indexed by `release_time`, streamed per-batch. `configs/example_mhd_january_periodic.yaml` is the reference periodic run.
- **Closing M1 — next step is the FLEXPART comparison run, now via M5 multi-release execution:**
  1. ✅ Downloaded `EUROPE_202312.zarr` + `EUROPE_202401.zarr` to `~/data/arco-era5/` (~58 GB each).
  2. ✅ `ArcoEra5ZarrReader` consumes multiple monthly stores via glob or explicit list (2026-05-24 multi-store reader entry).
  3. ✅ Memory blow-up resolved (2026-05-25 entry); steady-state RSS for the EUROPE stitch is ~3.4 GB.
  4. ✅ M5 multi-release execution landed 2026-05-27. A 744-release Mace Head schedule fits in one process via `configs/example_mhd_january_periodic.yaml`.
  5. Execute the 744-release MHD run against the local met stores. Compare the 96 release_time slices that align with `data/FLEXPART/FLEXPART_MHD_test_202401.nc` (days 1, 10, 20, 30 × 24 h). Workflow in `VALIDATION.md` § "Comparing against FLEXPART / NAME / STILT".
  6. Once Hanna agrees with FLEXPART within a documented tolerance on this case, flip the default scheme to `hanna_1982` and update `README.md`.
- **M5 follow-ups:**
  - ✅ Streaming per-batch Zarr writes — landed 2026-05-27 (see entry). Peak footprint memory is now one batch's tensor, not the whole schedule's.
  - Satellite multi-point-per-time case: add a `points: list[PointSpec]` field on `PointScheduleReleaseConfig`. Same runtime plumbing; `release_idx` already discriminates per release row.
  - (Only if a pathological config needs it) sparse / chunked-within-batch footprint accumulation to shrink the remaining dense per-batch tensor term.
- **Closing M4:** ✅ out-of-domain particle handling landed 2026-05-27 (drop-and-count: kill on `met_domain` exit, `escaped_this_step`/`alive_particles` in `trajectory_diagnostics.parquet`, `escaped_met_domain_total` in run metadata). Remaining: add `schema_version: 1` to the YAML once we're ready to commit to a breaking-change discipline. Optional safeguard: abort/warn if the escaped fraction exceeds a configurable threshold (undersized-domain signal).
- **Optional M1.x refinement:** replace the constant-K above-BL placeholder with an N²-based scheme using model-level `theta` gradients. Spec § "Open questions / known limitations" in `docs/turbulence.md` flags this; only worth doing if free-troposphere transport accuracy proves insufficient.
- After M1 validation lands, prioritize M2 (particle aggregation). Defer substantial GPU tuning until the turbulence and aggregation interfaces are stable enough to benchmark meaningfully.

## Operational notes

- Running bare `pytest` may use the wrong interpreter; prefer `.venv/bin/python -m pytest`.
- In a fresh environment, install the package editable first: `uv pip install --python .venv/bin/python -e .`.
- Ensure shell PATH is refreshed if `uv` was installed in-session.
- For public GCS Zarr buckets, ADC is not required when opening with anonymous token mode (`token="anon"`).
- Zarr v3 writes can fail if source dataset encodings contain v2 codec objects (for example `numcodecs.Blosc`); clear inherited encodings before writing v3.
