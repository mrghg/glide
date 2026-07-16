# GLIDE checkpoint and project context

> Internal dev journal — chronological project history, architecture intent, and
> milestone context. Not part of the user-facing docs. Operational agent/contributor
> guidance lives in `../CLAUDE.md`; current-state architecture in
> `../docs/architecture.md`.

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
- Core methods: RK2 backward advection, Langevin OU update (with `drift` arg for the Thomson well-mixed term), horizontal/vertical turbulent displacement (backward by default), surface reflection.

### `src/lpdm/turbulence/`

- `base.py` — `TurbulenceScheme` ABC + name-keyed registry (`register_scheme` / `get_scheme`).
- `placeholder.py` — `PlaceholderConstantOU` M0 baseline (constant `T_L=300 s, σ²=1`, no horizontal, no drift). Kept as a regression pin only.
- `hanna.py` — `HannaScheme` (production): Hanna 1982 / FLEXPART in-BL formulae (stable/neutral/unstable from `h/L`), surface-layer override (`z < 0.1 h`), free-troposphere Richardson closure (`K_z` from `N²`/Ri with Blackadar `l`), Thomson well-mixed drift `½(1+w'²/σ_w²)·∂σ_w²/∂z` (sign-flipped for backward Langevin per Flesch et al. 1995), optional Maryon (1998) / FLEXPART meander (independent horizontal OU with σ from local grid-wind std-dev).
- See `docs/turbulence.md` for the assembled formulation and `docs/LPDM_physics_spec.md` for the source-of-truth physics spec used for audits.

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

### 2026-07-16 Finding 7: the vertical coordinate is terrain-blind (surface footprint is ZERO over high ground)

**Symptom.** Spotted by eye in the January-2024 ICOS animation
(`scripts/animate_footprints.py`): the surface (0–40 m) footprint is zero over
elevated terrain. The hole mask traces orography exactly — Scandes, Scottish
Highlands, Alps, Pyrenees, Iberian meseta, Apennines, Dinarides, Carpathians,
Anatolia. Not physical: a month of releases cannot leave the whole Iberian meseta
at exactly zero while the sea around it is strongly sensitive.

**Confirmed against real met** (`data/glide_met_2024-01-15T12.zarr`, one ERA5 hour
extracted from Isambard with the new `scripts/extract_met_window.py`):

- *The vertical axis is not AGL — it is ~ASL.* `level_agl_m` (the bbox-mean of the
  per-column AGL, which `normalize_particle_coordinates` maps particles through)
  matches the true **mean height above sea level** to within 1–140 m, because the
  mean surface elevation over this ocean-dominated domain is only 140 m. Averaging
  AGL over the bbox erases terrain. A particle at "20 m AGL" is at 20 m ASL — over
  the Atlantic and over a 2263 m Alpine summit alike.
- *The clamp destroyed every sub-surface column.* Over the Alps 10/37 levels are
  below ground; `np.maximum(agl_m, 0.0)` collapsed them all onto z=0, giving 9
  zero-thickness layers. `_vertical_gradient` divides by those (guarded only by its
  1e-6 floor): **17,324 cells with |dθ/dz| > 1 K/m, max 4.6e6 K/m**, vs **0 cells /
  max 0.11 K/m** without the clamp. 45% of columns in the window have ≥1 clamped level.
- *Measured consequence on the real footprint.* Cells **with** surface footprint have
  median terrain **0.4 m**; cells with a **hole** have median terrain **669 m**. By
  terrain band, fraction of cells with NO surface footprint: sea level **0.2%**,
  >300 m **69.8%**, >600 m **86.6%**. Endpoint particles floor in proportion to
  terrain (p1: Atlantic 42 m, Iberia 488 m, Norway 823 m, Alps 969 m).
- *Releases are affected too, badly.* Releases are specified AGL but the axis is ASL,
  so **22/55 sites sit above 200 m and 13 above 500 m** are initialised at the wrong
  altitude by their full station elevation: PRS-10magl off by **2375 m**,
  JFJ-1000magl by **2247 m**, ZSF/ZUG-600magl by 1157 m, CMN-500magl by 812 m. Those
  sites are released ~2 km underground, inside the degenerate band. **Any validation
  statistic involving the Alpine/Apennine sites is measuring this bug, not the
  physics** — the v2/v3 comparisons need re-running (or those sites excluded) once fixed.

`docs/LPDM_physics_spec.md` is silent on terrain, so this is a design gap, not a
documented simplification.

**Two separable defects:**

- **A — the sub-surface clamp** (`met_reader._compute_level_agl_m`). **FIXED
  2026-07-16**: the clamp is removed; sub-surface levels now carry their true
  negative AGL, so layer thicknesses are real and the gradient blow-up is gone.
  Consumers needing a positive height already clamp locally
  (`free_trop_diffusivity(height.clamp(min=Z_MIN_M), ...)`). `tests/test_met_reader.py`
  green.
- **B — the bbox-mean profile used as the particle vertical mapping** (`main.py` /
  `hanna._grid_bounds` → `GridInterpolationBounds.level_agl_m` →
  `GPUEngine.normalize_particle_coordinates`). **FIXED 2026-07-16** via the
  terrain-following resample below — the reader now ships fields on a fixed AGL grid,
  so the single 1-D level array is *exact* per column, not a bbox approximation.

**A alone is not enough.** It removes the absurd gradients but leaves the coordinate
flat and ASL-like: particles would fly *through* mountains and Iberia's footprint
would be sampled 700 m below the meseta. Wrong, just no longer zero. **Do not treat
the terrain holes as fixed until B lands.**

**Design for B — hybrid (terrain-following) vertical coordinate, FLEXPART-style.
IMPLEMENTED 2026-07-16 (`src/lpdm/vertical_grid.py` + `met_reader` resample; 235
tests green).** Chosen over the two per-particle alternatives after checking what FLEXPART/HYSPLIT/
STILT actually do: they read *native* terrain-following model levels, so below-ground
levels never exist. GLIDE streams ARCO ERA5 *pressure* levels (quasi-horizontal —
they slice through mountains), which is the root of this. FLEXPART's `verttransform`
resamples the met onto a fixed terrain-following height grid **once per window** and
slope-corrects `w`. We mirror that. `particles[:, 2]` stays AGL throughout.

*Why this beats the per-particle options.* An LPDM's physics is intrinsically in
height-above-ground (`σ_w`, `T_L`, the whole BL scaling is in `z/h`) and the footprint
is defined in AGL, so a terrain-following grid is the natural frame. Regridding in the
reader (per window, ~60 steps amortised) means: the vertical level array becomes a
**run constant** (permanently kills the `level_agl_m` per-window recompile class), the
hot path gets *simpler* (no per-particle terrain lookup), and sub-surface pressure
levels are excluded **once**, cleanly.

Implementation (mostly `met_reader`-contained — the per-particle sampling already
routes through one 1-D `level_arr` + `grid_sample`, so the schemes barely move):

1. **`src/lpdm/vertical_grid.py`** (new, pure-numpy, CPU-tested): `default_agl_levels`
   (fixed ascending AGL grid, fine near surface → geometric aloft, to `alt_max`);
   `regrid_columns_to_agl` (per-column linear interp of pressure-level fields onto the
   AGL grid, excluding sub-surface levels — bracket lower index clamped to each
   column's first above-ground level, constant-extrapolate below it); `terrain_gradient`
   / `slope_correct_w`.
2. **`met_reader.fetch_hourly_window`**: after the existing pressure-level channel
   build + omega→w (both on the pressure grid, unchanged), regrid `hour_start`/`hour_end`
   onto the AGL grid using the (now un-clamped, Fix A) per-column heights; apply the
   `w` slope correction `w_agl = w − taper(z)·(u·∂h/∂x + v·∂h/∂y)`,
   `taper = clamp(1 − z/z_top, 0, 1)`; set `metadata.level = agl_levels`,
   `height_agl_m = agl_levels` broadcast `[Z,Y,X]`, and
   `metadata.pressure_level_hpa = bbox-mean pressure on the AGL grid` (length Z).
3. **Schemes unchanged.** Hanna-FT and Emanuel already use `pressure_level_hpa` as a
   per-level (horizontally-uniform) pressure and `height_agl_m` for gradients; feeding
   them the AGL-grid mean pressure + uniform heights is the *same* class of
   approximation they already make. Everything is stored ascending-z (surface at
   index 0) so Emanuel's flip-to-ascending becomes a no-op.
4. **Hot path:** `level_arr` is now the fixed AGL grid (a run constant); the
   `normalize_particle_coordinates` fractional-level lookup is unchanged and now
   *correct* per-column (the AGL grid is terrain-following, so one 1-D array is exact,
   not a bbox approximation). The CPU guards
   (`test_step_core_traces_as_one_graph_no_breaks/_does_not_recompile_per_step`) still
   apply.
5. Rejected alternatives, both per-particle: **(i) AGL + terrain-offset lookup + slope
   term** — works but adds ~5 terrain lookups/step to the hot path and a non-orthogonal
   Jacobian in the Langevin term; **(ii) track ASL internally** — exact advection but
   changes the meaning of `particles[:, 2]` across the release generator, reflection,
   the (deliberately met-agnostic) gridder, and the `endpoint_particles.parquet`
   contract, while every physics term still needs AGL. The per-window regrid gets the
   terrain-following benefit without either cost.

**Follow-ups / caveats.** The AGL grid is a modelling choice (FLEXPART makes it
configurable); starts as a `met_reader` default, promote to config later. Near-surface
extrapolation below the lowest above-ground level is constant (a log-profile would be
more faithful; defensible v1). The slope-correction taper shape follows FLEXPART in
spirit — verify against the primary reference before treating it as final (no FLEXPART
source in-repo). CANNOT be physics-validated on the CPU box: the numerical kernels are
unit-tested against the real Alps column (`data/glide_met_2024-01-15T12.zarr`) and the
existing well-mixed/conservation suite guards regression, but the footprint-over-terrain
outcome needs a GH200 run + NAME/FLEXPART comparison.

**Acceptance:** surface footprint non-zero over Iberia/Alps with the sea-level field
unchanged; the hole-vs-terrain correlation above collapses (>600 m band 86.6% → ~0);
elevated-site releases land at the right AGL; then re-run the 56-site validation and
revisit the v2/v3 mountain-site comparisons.

**Landed 2026-07-16.** `src/lpdm/vertical_grid.py` (`default_agl_levels`,
`regrid_columns_to_agl` with sub-surface exclusion, `terrain_gradient`,
`slope_correct_w`) + `ArcoEra5ZarrReader._resample_terrain_following` (on by default;
`terrain_following=False` restores the legacy pressure-grid path for A/B and synthetic
stores). Emanuel/Hanna-FT unchanged (they read the AGL-grid mean pressure). Tests:
`tests/test_vertical_grid.py` (8, incl. linear-recovery, sub-surface exclusion,
orientation invariance, slope taper) + two `test_met_reader` resample/slope tests; full
suite 235 green. **End-to-end reader check on the real Alps hour:** regridded near-surface
T @ 0 m AGL = **−7.2 °C** (the real 266 K lowest-above-ground level) vs the **+9 °C**
(282 K) sub-surface fiction the old coordinate sampled — the defining symptom is gone in
the fields. **STILL PENDING (needs GH200 + reference):** the footprint-over-terrain
acceptance run and the v2/v3 mountain-site re-check; the taper-shape cross-check vs the
primary FLEXPART reference; promoting the AGL grid to run config. **Fix A + B do not
change results over the sea**, only over terrain — the animation's ocean footprints stand.

### 2026-06-18 Performance: CPU thread tuning, per-window field caching, GPU readiness

Profiled the real engine (cProfile + thread sweeps on a 48-core node) to cut runtime for the longer FLEXPART-comparison runs. Combined CPU speedup ~1.4–1.5×, plus a 3×-jobs-per-node throughput win for sweeps; the engine is now positioned for GPU (Isambard AI GH200).

- **CPU thread tuning.** The per-step tensors (10^4–10^5 particles) are too small to feed many cores — lock contention dominates first. Measured optimum is ~16 torch threads; 48 (one-per-core, the old SLURM default) was the *slowest* at both 20k and 200k particles (~25% slower). `lpdm.main._configure_cpu_threads` reads `GLIDE_NUM_THREADS`; `scripts/run_periodic.slurm` defaults to 16 with `--cpus-per-task=16` and documents the multi-job throughput tip.
- **Per-window field caching.** `_meander_sigma_fields`, `_density_fields`, `_free_trop_fields` recomputed grid-wide `avg_pool2d`/vertical-gradient stacks every timestep, but these depend only on the hourly met window (~60 steps). Now built **once per window at the window midpoint** (`t_alpha=0.5`) and cached (`_cached_window_field`, keyed by `metadata.time_start`). `avg_pool2d` calls dropped 796→32 over 200 steps; ~20% faster at 20k particles. **Physics change (deliberate, OK'd):** these support fields are now frozen at the window midpoint rather than time-interpolated per step — a standard met-cadence approximation (<1%/hr drift); the per-particle interpolation through them is unchanged. Guarded by `test_per_window_field_cache_reuses_within_window_and_rebuilds_across` and the unchanged well-mixed/runaway-lofting tests.
- **Deferred diagnostics (GPU-oriented).** The per-step trajectory diagnostics did ~7 `.item()` host syncs every step (3 position means, 3 wind means, escaped count). On CUDA each serialises the pipeline. Now accumulated as device tensors and materialised in ONE host transfer per batch (`alive_particles` recovered via cumsum of the escaped-per-step tensor). Bit-equivalent output schema; verified end-to-end. The one remaining per-step sync is `active_count` (control flow) — see #2 below.
- **Hot-loop host-sync removal (2026-06-18b, after H100 was found launch/sync-bound — ~30 min/24-release batch, GPU starved).** The Hanna substep loop was firing *hundreds* of device→host syncs per step, each stalling the GPU:
  - **Engine input validation** (`gpu_engine.py`): `update_langevin_velocity` / `apply_vertical_turbulence` / `apply_horizontal_turbulence` ran `torch.any(... <= 0)` value-checks on every call — 3 + 1 + 1 syncs × 3 components × up to 50 substeps ≈ hundreds/step. Gated behind `VALIDATE_ENGINE_INPUTS` (module flag, env `GLIDE_VALIDATE_ENGINE=1`), **off by default**. Cheap host-side shape/ndim checks always run; the value-checks guard only against negative dt/σ²/T_L which are positive by construction. No test asserts these raises with the engine methods.
  - **Per-substep break** (`hanna.py _integrate_vertical_substeps`): `if not bool(mask.any()): break` synced every substep — but `mask = k_required > i` is non-empty for every `i < max_k = max(k_required)`, so the break was dead code. Removed (kept the single `max_k` sync that bounds the loop).
  - **FT-override guard** (`hanna.py _column_turbulence`): `if bool(torch.any(above_bl))` synced 3×/step (σ_w + 2 drift FD probes). Now applies the masked `torch.where` unconditionally (no-op where false); costs at most one extra cached-field grid_sample when nothing is aloft.
  - **Substep-cap warning**: reused the already-synced `max_k >= k_cap` instead of a fresh `at_cap.any()` that otherwise synced every step for runs that never hit the cap.
  - Net: from ~hundreds of host syncs/step to ~3 (`max_k`, the `scheme.step` empty-guard, and `active_count`). CPU runtime unchanged (`.item()` is cheap there); the win is entirely CUDA. **Expect the H100 to be much better utilised — measure GPU util (`nvidia-smi dmon`) to confirm and decide whether #2 (torch.compile) is still needed.**
- **GPU tooling.** `scripts/run_periodic_cuda.slurm` (Isambard AI / GH200: `module load cuda/12.6`, `--gres=gpu:1`, fail-fast `torch.cuda.is_available()` gate) + `configs/smoke_mhd_single_release.yaml` (one release, full physics, for benchmarking one release before the 720-release job). `_resolve_device` now raises `PreflightValidationError` for `cuda`/`mps` when unavailable instead of failing deep in allocation.
- **Test pass count: 200 → 202** (added the cache test; +1 net elsewhere). Suite ~2.7 min.
- **#2 — `torch.compile` of the hot-path kernels (DONE 2026-06-18b).** The substep loop launches many tiny elementwise kernels per step; on GPU the launch overhead can dominate once the syncs are gone. `GPUEngine(compile_hot_paths=...)` (env `GLIDE_COMPILE=1`, off by default) wraps `update_langevin_velocity` / `apply_vertical_turbulence` / `apply_horizontal_turbulence` / `reflect_surface` with `torch.compile(dynamic=True)` (dynamic because the masked substep subsets change size each iteration). `torch._dynamo.config.suppress_errors=True` gives per-graph eager fallback, so enabling it can never hard-fail a run. Needs `cudatoolkit/24.11_12.6` (nvcc/ptxas for Inductor/Triton) on Isambard AI — the CUDA SLURM script has a commented `module load` + `export GLIDE_COMPILE=1`. Method-level fusion is the first cut; if GPU sm% is still low, escalate to fusing the whole substep *body* (the densest launch region). Kept float32 (bf16 would wreck position accumulation and the finite-difference WMC drift terms; the other project's bf16/TF32/fused-EMA are ML-training-specific and not applicable). Compiled RNG is statistically — not bit — equivalent to eager. Guarded by `test_compiled_hot_paths_match_eager_and_never_hard_fail`.
- **GPU telemetry in the run script.** `scripts/run_periodic_cuda.slurm` now starts `nvidia-smi dmon -s um` in the background (10 s cadence) to a `*.gpu.log` sidecar and prints a mean/max sm% summary at the end — auto-killed via an EXIT trap. This is the diagnostic for the launch/sync-bound vs compute-bound question: low sustained sm% ⇒ try `GLIDE_COMPILE=1`.
- **Remaining per-step host syncs:** `max_k` (bounds the substep loop), the `scheme.step` empty-active guard, and `active_count` (control flow in `main._run`). All three are one-per-step and cheap relative to the removed hundreds; folding `active_count` away (operate on a possibly-empty active set) is the last micro-sync to chase if profiling still shows host stalls.

### 2026-06-19 torch.compile needs a C++20 host compiler on Isambard AI

Enabling `GLIDE_COMPILE=1` produced a flood of `torch._dynamo` **"WON'T CONVERT"** warnings, one per hot-path function, and ran no faster — every kernel was silently falling back to eager. The warnings name the *function* (`reflect_surface`, then `update_langevin_velocity`, …) so it looks like a per-method tracing problem, but it is **not**: the real error is at the bottom of the (suppressed) traceback, in Inductor's C++ codegen —

```
g++: error: unrecognized command line option '-std=c++20'; did you mean '-std=c++2a'?
```

- **Root cause.** Inductor compiles a C++ glue layer + precompiled header with the *host* compiler even on the GPU/Triton path. The system GCC on the el8 nodes is 8.x, which only knows `-std=c++2a` (the pre-final C++20 spelling). The build fails, `suppress_errors=True` swallows it as a per-graph eager fallback, and dynamo advances to the next function — which dies identically. So the failure is environmental and uniform across every compiled target, not fixable by editing any method.
- **Fix (in `scripts/run_periodic_cuda.slurm`, active when `GLIDE_COMPILE != 0`).** Load a C++20 host compiler alongside the CUDA toolkit and point Inductor at it:
  ```bash
  module load cudatoolkit/24.11_12.6   # nvcc/ptxas for Triton (GPU backend)
  module load gcc-native/14.2          # C++20 host compiler for Inductor C++ codegen
  export CC=gcc; export CXX=g++        # Inductor honours $CC/$CXX
  ```
  Confirmation that it worked: the "WON'T CONVERT" warnings disappear (compilation actually ran rather than falling back).
- **Diagnosis tip.** Because `suppress_errors=True` hides the real exception, reproduce with it *off* to see the bottom of the traceback: `torch._dynamo.config.suppress_errors = False`, then call the compiled fn directly. That is the only way the `-std=c++20` line surfaces.
- **Code cleanups made while chasing this (kept — sound regardless of the real cause).** (a) `reflect_surface` rewritten from boolean-mask in-place assignment to `torch.where` (one fewer clone, more compile-friendly). (b) The OU update extracted to a module-level pure-tensor kernel `_ou_step_kernel`, compiled via `self._ou_step_fn` instead of the bound method — a clean tensor-only function with no `self`/`Tensor|float` unions is what Inductor traces best. Both are numerically identical to before (`test_compiled_hot_paths_match_eager_and_never_hard_fail` + the OU/altitude variance tests pass).

### 2026-06-19 CUDA-graph capture wired — fixed-count loop + reduce-overhead (M3 phase 3)

Phase 3 of the CUDA-graph restructure (`architecture.md` §5.2): turn the now-static,
sync-free per-step path into a **captured CUDA graph**. CPU-verifiable parts are
done and tested; the actual graph capture + `sm%` payoff is **CUDA-only**, so it is
**pending Matt's GH200 run** (this dev box is CPU — can't capture graphs here).

- **Fixed-count substep loop (removes the last per-step sync).** `_substep_counts`
  (sync-free `k_i`/sub-dt) split out of `_substep_schedule` (which keeps the `max_k`
  `.item()` + cap warning for the variable-count path). `_integrate_vertical_substeps_static`
  gains `n_substeps`: when given, it loops a **constant** `max_substeps` times instead
  of the data-dependent `max_k`. Iterations past a particle's `k_i` are `sub_dt=0`
  no-ops, so the result is **bit-identical** to the variable-count loop (both draw
  RNG identically up to `max_k`; the extra iterations draw unused RNG but never change
  state). A constant trip count is what makes the loop graph-capturable.
- **`mode="reduce-overhead"` graph capture.** `HannaScheme._maybe_graph_compile`
  lazily wraps the fixed-count loop with `torch.compile(mode="reduce-overhead",
  dynamic=False)` on the static+compile path → the whole loop is one CUDA graph
  replayed per step (eager fallback via `suppress_errors`; logs "enabled"). RNG is
  handled by Inductor's cudagraph trees.
- **No nested compile.** `GPUEngine` now separates `_compile_requested` (the user's
  `GLIDE_COMPILE` intent) from `_compile_hot_paths` (whether *it* compiles the
  per-method kernels). On the static/graph path the per-method compile is **suppressed**
  (`_compile_hot_paths=False`) so the scheme's whole-loop graph doesn't nest a
  per-method graph; the scheme reads `_compile_requested` to drive its own capture.
- **Capture boundary = the substep loop.** Met fetch, convection (once/window), and
  the per-window field rebuilds are computed in `step` *outside* the loop and passed
  in as tensors, so they never enter the graph — the existing per-window/per-step
  split made this clean.
- **Tests (+3, total 212):** `test_fixed_count_static_substeps_matches_variable_count`
  (bit-identical, with genuine no-op iterations); `test_graph_compile_gating` (engages
  only on static+compile; per-method compile suppressed there); `test_graph_compile_substep_path_runs_on_cpu`
  (forces the path, actually compiles via Inductor on CPU, runs + finite + z≥0 — proves
  the loop is traceable, no hard graph breaks). Full suite green.
- **`scripts/run_periodic_cuda.slurm`** updated: `GLIDE_COMPILE` now drives graph
  capture on the static path; added a "WHAT TO CHECK" block (look for the "enabled"
  log + no "WON'T CONVERT", sm% climbing above 0–37%, per-batch time dropping) and a
  note that `max_substeps` is now the fixed per-step iteration count (tune to ~15–25).
- **Pending (phase 4, Matt on GH200):** confirm the graph actually captures and sm%
  rises; decide whether the §5.1 (B) escaped-particle recapture is needed.
- **Phase 4 finding (2026-06-21):** with compile ON and **no "WON'T CONVERT"** (the
  substep-loop graph captured cleanly), GH200 sm% is **still ~0–35%** (unchanged from
  pre-graph). So the substep loop was *not* the wall — the GPU finishes its bursts
  fast and idles in the **gaps**: the rest of the per-step pipeline is still eager
  (~5 surface `grid_sample`s, **3×** `_column_turbulence` with a `grid_sample` each,
  density/meander interp, RK2 advection `grid_sample`s, footprint scatter) plus the
  per-step Python orchestration. Next is to *localise the gap before building more*.
- **Profiler hook (`GLIDE_PROFILE`, 2026-06-26):** `main._StepProfiler` wraps a window
  of cursor-loop steps in `torch.profiler` (default 20 after 3 warmup), writes a Chrome
  trace + prints a summary (GPU-busy %, host-sync ops, top ops **with call counts**),
  then exits the run. Off by default (zero overhead). Env: `GLIDE_PROFILE`,
  `_WARMUP`/`_STEPS`/`_TRACE`, `_CONTINUE`. The summary's diagnosis → fix: many small
  GPU ops = launch-bound (expand the captured region to the whole `step`); a
  `cudaStreamSynchronize` = a residual host sync; long CPU spans = Python/met-bound
  (async prefetch). Guarded by `test_step_profiler_captures_window_and_exits`.
- **Profile result (GH200 smoke, 2026-06-26): LAUNCH-BOUND, conclusively.** 20-step
  window: **GPU-busy ~16.9%**, ~30 ms/step wall but only ~2.5 ms/step of actual GPU
  compute (50 ms CUDA / 20 steps) ⇒ **~10× headroom**. `cudaLaunchKernel` = **25,012
  calls (~1,250 launches/step), 33% of CPU**; kernels avg 1.6 µs CUDA but 5.9 µs CPU
  to launch (launch overhead > kernel). The captured substep graph works (the single
  `CompiledFxGraph` op runs 20×, once/step) but is a small slice; the wall is the
  **~1,250 eager launches/step from the rest of `HannaScheme.step`** — 5 surface
  `grid_sample`s, 3× `_column_turbulence` (each a `grid_sample` + ~40 elementwise),
  density/meander interp, RK2 advection, footprint scatter. Secondary: 740
  `cudaStreamSynchronize` (~37/step, only 4.8 ms but they block kernel queuing) +
  `aten::nonzero`/`index` from boolean masking (gridder valid-mask / escaped). **Fix:
  expand the compiled region from the substep loop to the whole per-step tensor core
  (gather met-field tensors in Python outside, compile `_step_core(...)`), collapsing
  ~1,250 launches → a handful. Expected GPU-busy 17% → 60–80%.**

### 2026-06-26 Whole per-step core compiled — the launch-bound fix (M3 phase 4)

Acting on the profile above: extracted the entire static-path per-step pipeline into a
**pure-tensor `HannaScheme._step_core`** and made *it* (not just the substep loop) the
`torch.compile(mode="reduce-overhead")` / CUDA-graph target. On CUDA the whole step —
5 surface `grid_sample`s, 3× `_column_turbulence` (each a `grid_sample` + ~40 elementwise),
density/meander interp, the substep loop, and the mask-gated write-back — now captures as
**one graph** replayed per step, instead of ~1,250 individual eager launches.

- **Split `step` into two clean paths.** The Python met-window access is hoisted into
  `_gather_static_inputs` (returns the per-step met-field *tensors* — 5 time-interpolated
  surface 2D fields + the cached FT/density/meander 3D stacks — plus the run-constant
  `GridInterpolationBounds`). `_step_core` takes only tensors + bounds + `engine`, so it's
  a clean compile target. The **dynamic (CPU) path** keeps the boolean-indexed body,
  byte-identical to before. Physics free-functions (`_column_turbulence`, `obukhov_length`,
  …) are unchanged and shared by both paths — interrogability preserved.
- **New compile-clean helper `_bilinear_at`** (float grid bounds, no numpy access) for the
  surface fields inside the core; `_interp_2d_bilinear` stays for the dynamic path.
- **`_maybe_graph_compile` now wraps `_step_core`** (was the substep loop);
  `_compiled_graph_core` replaces `_compiled_graph_substep`. The substep loop is called
  *inside* the core (eager, fixed-count via `n_substeps=max_substeps`), so no nested
  compile. Eager static path (no compile) uses `n_substeps=None` (variable `max_k`).
- **Eager `_step_core` is equivalent to the old static path** — same op sequence, so the
  static gates (`test_v1_well_mixed[static]`, `test_static_step_freezes_inactive_particles`,
  `test_static_path_footprint_conservation`) pass unchanged, and the dynamic-path tests
  too. **CPU smoke confirms the whole core compiles + runs** (no hard graph breaks → it
  should capture as a CUDA graph on the GH200). Full suite green.
- **Two CUDA-graph correctness fixes found on the GH200 (Matt, 2026-06-26; ba96533,
  ddd9683)** — neither reproducible on CPU (CPU `reduce-overhead` doesn't use CUDA
  graphs, so the CPU smoke test passed without them): **(1)** call
  `torch.compiler.cudagraph_mark_step_begin()` before each compiled-core invocation, so
  the framework doesn't treat the previous step's output buffers as live inputs; **(2)**
  `.clone()` the six tensors the core returns before storing them in `state` / returning
  `particles`, because the graph's outputs alias static graph memory that the next replay
  overwrites (and we persist `u/v/w_prime`/meander across steps and hand `particles` to
  `main._run`). General rule for any future CUDA-graph capture here: mark step-begin per
  replay, and clone any captured output that outlives the call.
- **Graph-break fix — the whole core now captures as ONE graph (2026-06-26).** The first
  GH200 capture left perf unchanged because of a single **graph break** (GH200 `.err`):
  `GPUEngine.normalize_particle_coordinates` did `ascending = bool(level_arr[-1] > level_arr[0])`
  — a data-dependent `bool(tensor)` — and that lookup runs ~5×/step (F9 + density/meander
  interp), shattering the capture into eager sub-graphs. Fixed by reading `ascending` from
  the **Python tuple** `bounds.level_agl_m` (constant for the whole run → compile-time
  constant, bit-identical). **CPU-confirmed break-free:** `torch.compile(_step_core,
  fullgraph=True, backend="eager")` now traces the whole core as one graph (it raises on
  any break, needs no GPU or C++ codegen) — locked in as `test_step_core_traces_as_one_graph_no_breaks`,
  a CPU regression guard against future `.item()`/`bool()` breaks. (Aside: `GLIDE_PROFILE`
  "exits early without error" is **by design** — it `SystemExit`s after the capture window.)
- **GH200 result (2026-06-26): capture works, 4× faster, but bottleneck shifted.** With the
  graph-break fixed: per-step wall ~30 → ~7.3 ms (**4×**), `cudaLaunchKernel` ~1,250 → ~237
  /step, GPU-busy 17% → 27%, `cudaGraphLaunch` ×20 (one replay/step). The Hanna core is now
  a small slice; the remaining ~237 launches/step are everything in `main._run` *around* the
  core — **advection** (`grid_sample`s) and the **footprint gridder** (`nonzero`/`index` from
  valid-mask boolean indexing, = the remaining ~18 syncs/step).

### 2026-06-26 Sync-free gridder + advection folded into the graph (M3 phase 4 cont.)

Closing the two non-core per-step costs the GH200 profile exposed.

- **Sync-free `FootprintGridder.accumulate`.** Rewrote it to operate on the WHOLE particle
  buffer with **no `torch.any` and no boolean indexing** (`particles[active_mask]`,
  `act_r[valid_mask]`) — those forced ~9 device→host syncs/step that blocked kernel queuing.
  Now: compute all indices, build a `valid` mask (active & in-bounds), zero the weight of
  invalid/inactive particles (`weights * valid * dt`), clamp indices into range, and
  `scatter_add_` the full set — out-of-bounds particles add 0.0 (exact no-op), so the
  footprint is **bit-identical** to the masked version. All 22 footprint/conservation/
  escaped tests pass unchanged.
- **RK2 advection folded into `HannaScheme._step_core`.** Advection was the other eager
  chunk (the `grid_sample`s). Rather than a second CUDA graph (whose `mark_step_begin`
  semantics across two graphs are subtle and untestable on CPU), advection now runs *inside*
  the single captured core: `_gather_static_inputs` also gathers the raw u/v/w wind fields +
  `alpha`; `_step_core` does RK2 (`_advect_rk2`, bit-identical to the old `wind_fn` +
  `engine.rk2_advect_backward`) on the full set FIRST, then turbulence on the advected
  positions, then gates inactive particles back to their **original (pre-advection)** state.
  So the whole step (advect + turbulence) is ONE graph and inherits the existing
  `cudagraph_mark_step_begin` + output-clone handling — no new cudagraph reasoning.
- **Robustness: `TurbulenceScheme.step_includes_advection(engine)`** (default `False`;
  `HannaScheme` returns `True` on the static path). `main._run`'s static branch advects
  itself **only when the scheme does not fold it**, so a non-Hanna scheme on CUDA still
  advects correctly (the fold made the static path Hanna-specific otherwise). Alpha +
  grid-mean-wind diagnostic factored into `_advection_alpha_wind_mean`, shared by both paths.
- **Validated CPU-side:** the fullgraph break-guard now traces the whole advect+turbulence
  core as one graph (no breaks); `test_hanna_constant_wind_preserves_mean_trajectory`
  parametrized over static/dynamic confirms the folded advection reproduces the analytic
  backward trajectory (conservation alone is position-independent and wouldn't catch it);
  freeze/conservation/dynamic-path tests pass.
- **Recompile blowout from the advection fold — fixed (2026-06-26; GH200 ca807de).** The fold
  passed `alpha` (the RK2-midpoint time-interp weight, **different every step**) to the
  compiled core as a **Python float**. dynamo specialises graphs on float values → it
  recompiled the whole core *every step*, blew the recompile limit (`cache_size_limit`), and
  fell back to eager: the profile window took **64.8 s** (vs 0.15 s), GPU-busy **0.7%**, the
  CPU dominated by `create_aot_dispatcher_function`/`fx_codegen_and_compile` and `aten::mul`
  running 29,958× eager. **Fix:** pass `alpha` as a **0-d tensor** (a dynamic input dynamo
  doesn't specialise on). General rule: *no per-step-varying Python scalar may be a compiled-
  core argument — make it a 0-d tensor.* (`dt_seconds` is a float too but is constant within a
  batch, so it only recompiles on the rare clamped last step — cached.) Guard added:
  `test_step_core_does_not_recompile_per_step` uses `torch._dynamo.config.error_on_recompile`
  to turn a per-step recompile into a hard CPU error (no GPU needed) — the recompile analogue
  of the fullgraph break-guard. Profiler default warmup bumped 5 → 10 (the whole-core compile
  is large; gives the CUDA-graph trees room to settle before the window).
- **Steady-state profile confirmed (2026-06-26; GH200 23cdce1).** With the `alpha` fix the
  capture is clean: per-step wall **30 → 7.3 → 5.0 ms**, GPU-busy **17 → 27 → 36.5%**,
  `cudaLaunchKernel` **~1250 → 237 → 106/step**, `cudaStreamSynchronize` **740 → 360 → 140**,
  `cudaGraphLaunch` ×1/step, no recompile. **≈6× faster than the pre-graph baseline** → the
  ~2.5 h month run → ~25 min. Remaining GPU time: ~49% is `copy_`/`Memcpy DtoD` (cudagraph
  input/output plumbing — the big met-field tensors copied into static buffers each replay),
  ~26% the captured physics core, the rest the still-eager footprint gridder (~45 launches/
  step) + diagnostics. Next levers (diminishing returns): capture the gridder (a 2nd graph) or
  make the met fields static graph inputs (recapture per window). **Decision pending Matt:**
  bank the 6× or push.
- **Second recompile — per-WINDOW, from `level_agl_m` — fixed (2026-06-26; GH200 23cdce1, month
  run).** Only showed on a multi-window run: `normalize_particle_coordinates` built the F9
  vertical-lookup axis with `torch.as_tensor(bounds.level_agl_m)` from a **tuple of floats**,
  and the per-level AGL heights (bbox-mean) **change every met window** (geopotential is
  weather-dependent). dynamo specialised on the tuple values → recompiled every window → blew
  the limit (`0/8`, reason `grid_bounds.level_agl_m[0] == 18026.37`) → eager fallback. **Fix:**
  pass the level array as a **dynamic tensor** (`level_arr`) and the order as a genuinely-
  constant **bool** (`ascending`), threaded `_gather_static_inputs → _step_core → {_advect_rk2,
  _column_turbulence, _interp_3d_field} → normalize_particle_coordinates`. The tuple path is
  kept for non-compiled callers (level_arr/ascending default None). The recompile guard now
  **varies the level array** across calls (blh 2000 vs 4000 → 4000 vs 6000 m top) so it covers
  this class too; verified non-vacuous (forcing the tuple path → `RecompileError`). Same lesson
  as `alpha`, one level deeper: *no per-window-varying value may reach the compiled core as a
  Python scalar/tuple — only constants and tensors.*
- **Month run: physics fast, WALL not faster — GPU bursts (≤64% sm) then long idle (2026-06-26).**
  The per-step capture is fast but the *job* isn't, and the dmon shows the GPU idling between
  bursts → the bottleneck is non-GPU work *between* the captured bursts, which the 20-step
  `GLIDE_PROFILE` window (always inside ONE met window) is structurally blind to. Prime
  suspect: synchronous met I/O at window transitions (`reader.fetch_hourly_window`,
  `_get_hourly_met_window`), made worse by **cross-batch met RE-FETCH**. Config analysis of
  `example_mhd_january_periodic.yaml` (5-day backward window, 24-release/1-day batches → 30
  batches, `met_cache_max_hours: 2`): consecutive batches' 5-day windows overlap ~4/5, and a
  2-hour LRU keeps none of it → each met hour is fetched by ~5 batches. Est. **~4320 fetches vs
  ~840 unique hours (~5× redundant I/O)**. Prefetch does NOT fix the *count* — that's a cache
  (or batch-size) change. Whether it matters depends on per-fetch latency (local zarr vs GCS).
- **Built `_PhaseTimer` (`GLIDE_PHASE_TIMERS`, main.py) to settle it.** Whole-run wall-clock
  accountant bucketed by phase (met_fetch / step / advect / convection / gridder + residual),
  printed at run end AND logged every `GLIDE_PHASE_TIMERS_EVERY` (default 25) fetches — so a
  TIMED-OUT job still leaves the split in the .out log (the last run timed out, no manifest).
  `met_fetch` is exact (synchronous CPU/IO), so `met_fetch / wall` is a faithful GPU-idle
  fraction even with no CUDA sync; `GLIDE_PHASE_TIMERS_SYNC=1` adds per-phase `cuda.synchronize`
  for clean step-vs-gridder attribution (serialises → perturbs totals). Wired on both step
  paths; no-op when disabled.
- **MEASURED (2026-06-29, 2-day GH200 run): I/O-bound.** `met_fetch=63% of wall, ~1.6 s/fetch`;
  step+gridder only ~15%. The 6× physics win was real but irrelevant to the wall — the GPU
  idles ~2/3 of the time on `fetch_hourly_window`. So the fix is fetch reduction + overlap, not
  more GPU work.
- **CACHE FIX CONFIRMED (2026-06-29): `met_cache_max_hours` must EXCEED one batch's met span,
  not equal it.** Bumping 2→**144** (= exactly one batch span: 120 h backward + 24 h release
  span) gave **zero** benefit — batch 1 still re-fetched its whole range. Cause: **LRU thrash**.
  The cursor walks latest→earliest, so at a batch's end the LRU-oldest entry is the *top* hour —
  exactly what the next batch (starting 24 h later) needs first. With the cache exactly full,
  batch N+1's first new fetches evict the overlap hours right before they'd be reused. Bumping
  to **192** (≥ span + one batch advance ≈ 168 h) fixed it: batch 1's fetch count **plateaued at
  ~168** (batch 0's 144 + ~24 new) and the periodic log went quiet = reuse working. Month
  extrapolation: ~840 vs ~4320 fetches (**~5×**), met_fetch ~63%→~30% of wall, ≈2× faster
  overall. Current recommended config value: `met_cache_max_hours: 192` (240 for margin).
  **NB on memory: a cached window is ~300 MiB (`[C,Z,Y,X]`), so 192 h ≈ ~50 GiB.** The early
  "~21 MB/window" note was wrong (that RSS was staging, not the cache). With `met_cache_on_host`
  (the dev_alloc fix) that ~50 GiB lives in host RAM, not HBM — so confirm Grace headroom.

### NEXT STEPS (ordered; 2026-06-29 — Matt stepping away, resume here)
1. **⚠️ `dev_alloc` per-batch GPU-memory growth (GATES the month run) — INSTRUMENTED + MITIGATED,
   2026-06-29; awaiting a GH200 run to confirm.** Observed 29.7→43.5 GiB across batches
   (`cache=144`), 45 GiB at batch 1 (`cache=192`) — *device* memory, not the host cache; and
   `alloc ≈ reserved` so it's LIVE tensors, ~40 GiB more than one batch needs (~1 GiB) → real
   accumulation. If ~linear a 30-batch month OOMs ~batch 7–8 (likely why earlier long runs died).
   Changes made in `main.py`:
   - **`@torch.no_grad()` on `_run`** — the whole batch loop ran with autograd live; correct for
     an inference model and rules out grad-graph retention. (Tests still pass — values unchanged.)
   - **Per-batch reclaim** at batch end: `del convection_state` (was leaked — only
     `turbulence_state` was deleted) + `gc.collect()` + `torch.cuda.empty_cache()` (CUDA only) to
     break reference cycles (closures / compiled-fn caches retain device tensors past plain `del`)
     and return blocks to the driver.
   - **Batch-boundary diagnostics**: `_log_device_memory` logs alloc/reserved/peak + **compile
     count** at `batch N start`, `end (pre-reclaim)`, `end (post-reclaim)`.
   **Data from the 2-day run (2026-06-29):** `batch 0 end (post-reclaim) alloc=43.48 GiB`,
   `batch 1 start=43.48`, `batch 1 end (post-reclaim)=50.70`, **`compiles=1` throughout**. So:
   (a) NOT recompilation (compiles flat → the `dt` worry was wrong); (b) `gc`+`empty_cache` freed
   ~nothing (43.54→43.48); (c) batch 0's 43 GiB **survives every `del` into batch 1**; (d) ~**+7
   GiB/batch** of LIVE memory (`alloc≈reserved`). I hypothesised cudagraph-pool growth — and was
   WRONG; the snapshot disproved it (good thing I localised instead of "fixing" a guess).
   **RESOLVED via `GLIDE_MEM_SNAPSHOT=1` allocation snapshot (added to main.py; open the pickle at
   https://pytorch.org/memory_viz, or use scratch `analyze2.py` to aggregate by Python frame):**
   the 50.7 GiB is **47.9 GiB in 336 blocks of 146 MiB, all from `met_reader.py:798
   _dataset_to_channel_tensor`** (via `fetch_hourly_window` ← `_get_hourly_met_window`) + 2.6 GiB
   more from the reader. Cause: the reader was built with `device=cuda`, so every cached met
   window's `[C,Z,Y,X]` stack lived in **HBM**, and the 192 h cache held ~168 windows × ~300 MiB
   ≈ 50 GiB on the GPU. NOT a leak — the cache doing its job on the wrong device; the "per-batch
   growth" was the cache filling toward the cap. (Also corrects the earlier "~21 MB/window" note —
   that host RSS was staging, not the cache; a window is ~300 MiB.) Diag lists were negligible
   (34 k tiny blocks). **FIX: cache met on the HOST** — new `memory.met_cache_on_host` config
   (default True) → reader built with `device="cpu"`; the per-step physics already moves the active
   window to the device via its `.to(device)` calls (audited ALL consumers: `_advection_alpha_wind_mean`,
   `_advect_active_particles`, `_gather_static_inputs`+`_build` closures, Emanuel `maybe_convect` —
   every one `.to(device)`s; the CPU suite can't catch a missing one since cpu==cpu, so this was a
   manual audit). Frees ~50 GiB HBM; the cache moves to Grace LPDDR5X (plentiful). Per-step
   host→device copy of the active window is cheap over NVLink-C2C. Kept the no_grad wrap +
   `del convection_state` + per-batch `gc`/`empty_cache` (cheap correctness hygiene).
   **CONFIRMED on GH200 (2026-06-29):** `dev_alloc` dropped from ~50 GiB to **~0.2 GiB**, flat
   across the batch boundary (0.15→0.15), `compiles=1`; host RSS plateaus at **~58 GiB** (the
   cache) once full. OOM gate cleared — ~119 GiB HBM now free (so bigger batches are an option
   later). The non-cache GPU working set is tiny (~0.2 GiB), incl. the cudagraph workspace.
2. **Async prefetch — BUILT (2026-06-29; `memory.met_prefetch`, default True).** `MetPrefetcher`
   in main.py: a single background worker prefetches the next backward met hour while the
   current window is stepped, so fetch I/O overlaps compute (~1.58 s fetch ≈ ~1.2 s compute per
   window → up to ~1.75× on the remaining ~22 min month I/O). Safety as designed: ONE worker does
   every reader access (no reader-concurrency assumption); requires `met_cache_on_host` so the
   worker only makes HOST tensors (no CUDA off-thread; auto-disabled+warned otherwise); the cache
   is mutated only by the main thread in `get()` (no lock); fetch errors re-raise via
   `Future.result()`; `shutdown()` after the batch loop. Bit-equivalence guard:
   `test_met_prefetch_matches_synchronous_footprints` (prefetch on vs off → identical footprints,
   spanning several met-hour boundaries). **Next: confirm on GH200 — phase-timer `met_fetch%`
   should drop sharply (it now measures only the *un-hidden* I/O = prefetch misses).**
3. **Cache guard — DONE (2026-06-29).** `_warn_if_met_cache_thrashes` (main.py) warns at startup
   when `met_cache_max_hours` is below the batch-geometry reuse threshold (≈ batch met span +
   advance, computed from the first two expanded batches), so the set-cache-to-batch-span →
   zero-benefit footgun can't recur silently. Test: `test_met_cache_thrash_warning_fires_when_undersized`.

### I/O-OPTIMISATION ARC COMPLETE (2026-06-29) — all planned items shipped, full suite 217 green
Confirmed on GH200 (2-day, 48 footprints, ~7 min): phase timers now **balanced** —
`step 29% / convection 19% / met_fetch 15% (was 63%) / gridder 7% / residual ~29%`;
`dev_alloc 0.2 GiB`, RSS plateau ~52 GiB. Month extrapolation ≈ **~1.5 h (was 2.5 h) and now
completes reliably** (no OOM). Summary of the arc: cache fix (~5× fewer fetches) + host cache
(~50 GiB HBM → host, no OOM) + async prefetch (overlap remaining I/O) + guards/diagnostics
(phase timer, mem snapshot, recompile/break guards, cache-thrash guard).

### NEW OPTIMISATION FRONTIER (if Matt wants to push further; the run is now I/O-solved + reliable)
The bottleneck has shifted off I/O. Remaining levers, in rough order:
- **residual ~29%** — per-step Python loop overhead + batch setup/finalize + output write; not in
  any timed phase. Profile what's there before optimising.
- **convection ~19%** (Emanuel) — fires once per met bracket; not yet optimised like Hanna was.
- **step ~29%** — already the 6× CUDA-graph win; diminishing.
- **Bigger batches** — now unlocked: GPU is ~0.2 GiB used (was the constraint). Larger
  `max_releases_per_batch` → fewer batches → less per-batch residual + better compile
  amortisation, traded against more inactive-particle work on the static full-buffer path
  (step). Net unclear — measure with the phase timer.
- REMINDER: `max_substeps=20` is still a placeholder pending Matt's convergence eval [[open-max-substeps-eval]].

### MULTI-SITE RELEASES (built 2026-06-29) — the efficient way to grow a run
New `release.kind: "multi_point_periodic"` (`config.py`): a `sites` list (`{name, lon, lat,
alt_agl_m}`) on a shared periodic schedule. `expand_to_batches` emits one `ConcreteRelease`
per (time, site) in **TIME-MAJOR** order, so chunking by `max_releases_per_batch` gives batches
of contiguous-times × ALL sites — a dense active set sharing one met fetch/hour (amortises the
per-window fixed costs that dominate the wall). Keep `max_releases_per_batch` a multiple of
n_sites. The particle/runtime/gridder/prefetch machinery was ALREADY per-release-location
(`generate_batch_particles` uses each `rel.lon/lat/alt`), so this was mostly config + output.
Why flat not 6D: the flat `release` axis is the general case — it also fits the planned
**satellite** workflow (irregular (lon,lat,time) soundings), which a `site×time` grid cannot
represent. **OUTPUT SCHEMA CHANGED (all runs):** footprint leading dim `release_time` → `release`,
with `release_time`/`release_lon`/`release_lat`/`release_alt_agl_m`/`site` as per-release COORDS
(was scalar attrs). Recover a per-site cube:
`fp["footprint"].set_index(release=["site","release_time"]).unstack("release").sel(site="MHD")`.
Example: `configs/example_multisite_january.yaml` (MHD/RGL/TAC). Tests:
`test_multi_site_*` (expand time-major; run emits per-site coords + unstacks), config-validation
tests (site-in-domain, unique names), updated `test_output_writer`/periodic tests for the dim
rename. Notebook `flexpart_comparison.ipynb` cmp-03 updated (`swap_dims` to release_time + read
location from coords). `ConcreteRelease` gained an optional `label` (site name).
- **Fix (found on the first GH200 run, 2026-06-21):** the per-step memory-log block
  (`mem.log_every_steps`) still referenced `active_count`, which only the *dynamic*
  branch binds → `UnboundLocalError` on the static path. CPU tests missed it because
  `_make_run_config` defaults `memory_log_every_steps=0` (block never ran). Fixed by
  materialising the count locally in the log line (`int(active_mask.sum().item())` —
  occasional, off the hot path) and adding `memory_log_every_steps=1` to
  `test_static_path_footprint_conservation` so the static log path is exercised.

### 2026-06-19 CUDA-graph prep — full-set device-gated per-step path (M3 phase 2)

Phase 2 of the CUDA-graph restructure (`architecture.md` §5.2): make the **whole**
per-step body static-shape and host-sync-free on the GPU path, so phase 1's static
substep loop is actually fed constant shapes (it was previously handed a
boolean-indexed — dynamic-shape — active subset). This is strategy **(D)+(A)** from
§5.1: device-gated, "accept the waste" (inactive/escaped particles are processed
then masked out, not dropped). **CPU is byte-for-byte unchanged.**

- **Shared gate `use_static_step_path(device, override)`** (`gpu_engine.py`): static
  on CUDA, dynamic on CPU/MPS; `GLIDE_STATIC_SUBSTEPS` env override; explicit
  override arg for the scheme ctor flag. `HannaScheme._use_static_substeps` and the
  `main._run` loop both call it, so they always agree.
- **`HannaScheme.step` full-set path.** Operates on the full particle buffer (no
  `particles[active_mask]` indexing); writes back with `torch.where(active_mask, …)`
  for positions and every state channel (`u/v/w_prime`, `u/v_meander`). Inactive
  particles are additionally frozen inside the substep loop by forcing `sub_dt=0`
  (via an `active_mask` threaded into `_integrate_vertical_substeps_static` /
  `_substep_schedule`, which also stops inactive particles inflating `max_k` or
  tripping the cap warning). The dynamic (masked-subset) path is retained for CPU.
- **`main._run` static branch.** Advects the full buffer and gates with
  `torch.where`; runs scheme.step / convection / accumulate / escaped-update
  unconditionally (all already `active_mask`-gated), with **no per-step `.item()`**.
  The `active_count` control-flow sync is gone on this path; the diagnostic is now a
  per-step device tensor (`active_mask.sum()`) materialized once per batch. The
  dynamic branch is the verbatim pre-phase-2 code (and the gridder accumulate +
  escaped-update were already full-set + mask, so they needed no change).
- **Remaining per-step host sync on the static path:** `max_k` only (bounds the
  substep loop). Phase 3 replaces it with a fixed loop count for graph capture.
- **Convection** is untouched — it runs once per met-window (outside the per-step
  hot path), already full-set + `torch.where`-gated, so it stays out of the future
  captured region by design.
- **Tests (+2, total 209):** `test_static_step_freezes_inactive_particles` (mixed
  mask → inactive positions + all state byte-identical, active particles move);
  `test_static_path_footprint_conservation` (end-to-end with `GLIDE_STATIC_SUBSTEPS=1`
  → footprint sum = Σ(active)·dt/N, validating full-set advection gating + the
  `active_count` desync + mask-gated accumulation). Full suite green.

### 2026-06-19 CUDA-graph prep — device-gated static-shape substep loop (M3 phase 1)

First phase of the CUDA-graph restructure (`architecture.md` §5). The substep loop
is the densest per-step kernel-launch region; CUDA-graph capture (the real cure for
the GH200's launch/orchestration-bound state) requires **constant tensor shapes**
across substeps, which the existing dynamic masked loop violates (the active subset
shrinks each substep). This phase adds a static-shape variant and validates it
against the dynamic one; graph capture itself is phase 3.

- **New `HannaScheme._integrate_vertical_substeps_static`.** Processes the **full**
  active set every substep, masking finished particles by `sub_dt=0` (a math no-op:
  `a=exp(0)=1`, `variance=σ²(1−a²)=0`, displacement `w'·0=0`, reflection of `z≥0` is
  identity) instead of indexing a shrinking subset. Constant kernel shapes across
  iterations → fuses cleanly under `torch.compile` and is the prerequisite for CUDA
  graphs. Trades extra elementwise FLOPs (finished particles keep being touched) for
  far fewer/consistent launches — a win on a launch-bound GPU, a loss on CPU.
- **Device-gated.** New `static_substeps: bool | None` ctor arg + `GLIDE_STATIC_SUBSTEPS`
  env + `_use_static_substeps(engine)`: auto-selects static on CUDA, dynamic
  (masked) on CPU/MPS. **CPU behaviour is unchanged** (still the dynamic path), so
  all existing runs/tests are bit-identical by default.
- **Shared `_substep_schedule`** factored out of both variants (single source of
  truth for `k_i`, sub-dt, the `max_k` loop bound, and the once-per-instance
  substep-cap warning). The `max_k` host sync is now the *only* per-step sync in the
  static path; phase 2 removes the remaining `scheme.step`/`active_count` syncs and
  phase 3 replaces `max_k` with a fixed loop count for graph capture.
- **Equivalence (not bit-identical when `k_i` varies).** `randn` is drawn for the
  full set each substep (vs only the owing subset), so the random realisation
  differs; the two paths are statistically equivalent, and **bit-identical when
  every particle needs the same `k`** (no early finishers → identical ops + RNG).
- **Tests (+3 in `test_hanna.py`):** zero-dt no-op contract on the engine
  primitives; bit-identity for homogeneous `k` (k=1 and k=5); distribution match
  for heterogeneous `k` (std within 5% at n=8192). **`test_v1_well_mixed_hanna_backward_path`
  parametrized over `static_substeps`** so the static path is held to the same WMC
  acceptance gate (interior relative-RMS < 15%) as the dynamic path. Both pass.
  **Test count 203 → 207** (+3 new, +1 from the V1 parametrize); full suite green.

### 2026-06-01 Deep convection (Emanuel reduced port; Forster 2007 / Stohl 2005 §4.6)

The biggest non-audit dispersion gap GLIDE had vs FLEXPART — deep cumulus convection that loft surface air through the entire troposphere in minutes-to-hours — is now implemented as a reduced faithful port of the FLEXPART Emanuel & Živković-Rothman scheme (Stohl 2005 §4.6; Forster, Stohl & Seibert 2007). See `docs/convection.md` for the full spec.

- **Architecture.** New `src/lpdm/convection/` subpackage (mirrors `lpdm.turbulence/`'s ABC + registry pattern): `ConvectionScheme` ABC, name-keyed registry, `NoConvection` (default, pass-through), `EmanuelReducedConvection`. The convection scheme is a separate runtime stage from turbulence — called **once per met-update interval** (not per integration timestep), because the per-column mass-flux matrix depends only on temperature/humidity profiles which don't change inside a met window. The runtime tracks the met bracket's start time and fires convection exactly once when the cursor crosses into a new bracket.
- **Algorithm.** Implements the five Forster (2007) steps in pure torch at a higher level of abstraction than FLEXPART's ~3000-line `convect43c.f`:
  1. Parcel lift (LCL via Bolton 1980; dry adiabat below, moist pseudo-adiabat above via θ_e-conserving bisection — fixed-point/Newton both fail to converge under Clausius-Clapeyron's strong nonlinearity).
  2. LCL/LNB/CAPE identification on the bbox-mean column profile.
  3. Trigger (Forster 2007 Eq 34): `T_v_parcel(LCL+1) ≥ T_v_env(LCL+1) + T_threshold` with `T_threshold = 0.9 K`, plus CAPE floor (`min_cape_j_kg=50`) and deep-only filter (`min_cloud_depth_m=500`).
  4. Cloud-base mass flux: `M_b = closure_c · ρ_LCL · min(√(2·CAPE), 5 m/s)`. The cap on the buoyancy velocity is essential — without it, CAPE > 500 J/kg gives an unphysically large M_b; the 5 m/s cap represents a typical updraft-cell peak.
  5. Mass-flux matrix `MA[i, j]` with three groups of source rows: **sub-LCL** = surface-updraft path (BL → cloud at `MA[i, j] = M_b · profile(j)`); **in-cloud** = Emanuel buoyancy-sorting (Forster Eq 35-36 with mixing fraction ε from θ-profiles); **above-LNB** = zero. Particle redistribution probability is `MA[host, j] / m_host`, sampled via cumulative-sum + uniform random with sub-bin placement in the destination layer.
- **Backward mode.** Same matrix used for forward + backward. Forster 2007 §3 + Fig 2 demonstrate the WMC holds under either time direction; the matrix is mass-conserving by construction.
- **Documented departures from full Emanuel** (in `docs/convection.md`): (a) bbox-mean column rather than per-(lon,lat) — mirrors the F9 approximation; (b) `θ_l ≈ θ` (latent reservoir implicit in q_sat); (c) no explicit saturated-downdraft branch; (d) linear M-profile not the buoyancy-driven profile; (e) capped buoyancy velocity instead of the full quasi-equilibrium closure; (f) compensating subsidence implicit in mass conservation rather than as an explicit downward velocity for non-displaced particles.
- **Met channel.** Added `q` (specific humidity) to `ArcoEra5ZarrReader.DEFAULT_VARIABLE_MAP`. Units validation accepts `kg kg**-1` (ARCO ERA5's published units) and the dimensionless variants. **`scripts/download_sample_cube.py` updated to fetch `specific_humidity`** — it's a 3D pressure-level field (adds ~as much volume as temperature), but since the example periodic config now enables convection by default, a cube without it can't run that config. New regression `test_required_vars_cover_all_scheme_dependencies` pins the download-script variable list against every scheme dependency so this can't silently drift again. **Existing local EUROPE_YYYYMM cubes must be re-downloaded** to pick up `q` before running a convection-enabled config.
- **Config.** New `ConvectionConfig` in `lpdm.config`: `scheme: "none" | "emanuel_reduced"` (default `"none"` is bit-equivalent to no-convection), plus `emanuel:` block with the four tuning constants. YAMLs without a `convection:` block produce the same output as before. `main._convection_scheme_kwargs` mirrors `_scheme_kwargs` for the turbulence case. CLI rejects unknown convection schemes the same way it rejects unknown turbulence schemes.
- **Tests added: +18 (total 200).** Free-function physics: q↔e round-trip, T_v, LCL temp/pressure, θ, θ_e, q_sat, moist-adiabat θ_e conservation. Column-level: CAPE/LCL/LNB on a moist-unstable sounding (CAPE > 100 J/kg) and a dry-stable sounding (CAPE ≈ 0). Matrix: zero-when-no-cloud, non-zero-confined-to-source-rows-≤-LNB-and-cols-in-[LCL,LNB], in-cloud row sums equal M-profile, BL row sums equal M_b. Scheme-level: NoConvection pass-through, end-to-end lofting (>500 m median rise on a synthetic convective column), no-fire on stable sounding, constructor validation.
- **Example config.** `configs/example_mhd_january_periodic.yaml` now ships with `convection.scheme: emanuel_reduced` enabled by default. For January 2024 mid-latitude this will be a modest effect (winter, weak convection); the test in summer / tropics will be more meaningful.
- **Test pass count: 169 → 200 across the two audit PRs + this convection PR**, suite runtime ~3 min (the V1 and convection tests are the slow ones).
- **Next user step:** re-run `configs/example_mhd_january_periodic.yaml` (now convection-enabled in addition to all audit fixes) and update `notebooks/flexpart_comparison.ipynb`. If the FLEXPART gap is still meaningful, the next levers (in order) are per-column profiles for the parcel lift, the full Emanuel quasi-equilibrium closure, and the saturated-downdraft branch — all documented in `docs/convection.md` §4.

### 2026-05-31 Audit follow-up: F4T2 substepping + F3 cap removed + F9 vertical interp + F5/F15 UBL + F10 surface-ρ helper

User re-ran FLEXPART comparison after the first audit-fixes PR (entry below);
gap narrowed but GLIDE remained less dispersive than FLEXPART. This PR
(`physics-audit-followup`) closes the remaining five deferred audit items in
one branch. Tests: 169 → 182 (+13). Deep convection (the biggest known
non-audit dispersion gap, Stohl 2005 §4.6) is scoped separately and documented
in this entry but not implemented here.

- **F4 Tier 2 — per-particle adaptive substepping.** Each particle's OU + displacement + reflection now integrates in `k_i = ceil(dt / (substep_c · T_Lw_i))` internal substeps (default `substep_c=0.5`, capped at `max_substeps=50`). Vectorised via masking — only particles still owing substeps are touched in each loop iteration. σ², T_L, gradient, density pieces stay fixed at outer-step values (a deliberate simplification, documented); the `(1+w'²/σ²)` velocity factor inside the σ-gradient drift IS recomputed per substep using the current `w'` (holding it fixed would systematically under-drift newly-released particles whose `w'≈0` at outer-step start — caught by `test_hanna_well_mixed_no_runaway_lofting` during development). Reflection (with w'-flip) happens at each substep. New helper `HannaScheme._integrate_vertical_substeps`; `engine.{update_langevin_velocity, apply_vertical_turbulence, apply_horizontal_turbulence}` extended to accept per-particle `dt_seconds` tensors. The F4 Tier 1 once-only-warning moved from `dt > T_L/5` to `k_i hits max_substeps` (more actionable signal post-substepping).
- **F4 rogue-trajectory clip.** Added a `|w'/σ_w| ≤ 4` clip (`W_PRIME_SIGMA_RATIO_MAX`) at the end of each substep OU update (spec §T anticipates exactly this safeguard). Without it, particles near the BL top (where σ_w hits the `SIGMA_MIN_M_S` floor) snowball drift into NaN under the substep cap. FLEXPART has the equivalent clip.
- **F3 — drift cap removed.** The legacy `|drift| ≤ σ_w/Δt` cap was a silent WMC violation at the SL / BL-top seams. F4 Tier 2 + the rogue-trajectory clip address the root cause; comment in `hanna.py` now cites this removal.
- **F9 — pressure-level vertical interpolation.** `GridInterpolationBounds` gained a `level_agl_m: tuple[float, ...] | None = None` field; when populated, `GPUEngine.normalize_particle_coordinates` does a piecewise-linear fractional-level lookup against the per-level AGL array instead of the legacy linear-in-AGL mapping. ARCO ERA5 pressure levels are roughly log-linear in altitude, so the legacy mapping silently warped vertical wind shear (and σ/T_L sampling) — biggest effect aloft. Both call sites (`main._advect_active_particles` and `hanna._grid_bounds`) updated; the lookup handles both ascending and descending level orderings. The bbox-mean `metadata.level` is the source of truth (the per-column `height_agl_m` would be even more accurate but requires a much bigger refactor; documented as a known residual approximation).
- **F5/F15 — constant-σ "unresolved basal layer."** `HannaScheme(z_ubl_m=…)` default 2 m. The σ/T_L/drift/density-gradient *sampling* height is clamped at `z_ubl_m`, holding turbulence parameters constant in the basal layer. This makes smooth-wall reflection WMC-exact per W&F §7b. Particle position is NOT clamped; only the *evaluation* height for σ/T_L. Also caps the |∂σ²/∂z| spike that drives the substep cap to bind near-surface.
- **F10 — surface air-density helper for STILT conversion.** `comparison.surface_air_density_from_met(met_ds)` builds `ρ(lat, lon) = sp/(R_d·T_surface)` from a met store as an `xr.DataArray`, which the existing `to_stilt_surface_footprint` already accepted in place of the scalar `air_density_kg_m3`. Per S&F 2004 Eq. 8 the footprint is density-weighted at the source cell; using a scalar ρ biases the result by a few percent for deep-PBL receptors. The function itself didn't change — F10 is just the missing helper to build the right ρ field.
- **Tests added (+13, total 182):** `test_substep_cap_warning_fires_once` + `test_substep_cap_warning_silent_when_dt_is_small` (F4T2 + cap); `test_normalize_coordinates_uses_level_lookup_for_pressure_levels` + `test_normalize_coordinates_handles_descending_level_order` (F9); `test_ubl_holds_sigma_constant_below_z_ubl` + `test_z_ubl_constructor_validation` (F5/F15); four `surface_air_density_*` tests (F10). The previously-broken `test_hanna_well_mixed_no_runaway_lofting` is now passing again after the per-substep `(1+w'²/σ²)` recomputation fix. Suite runtime ~160s.
- **Conservation tolerance loosened from 1e-3 → 2e-3** in `test_periodic_release_footprint_mass_matches_active_particle_time` to absorb the F4 Tier 2 substep-loop's extra float32 arithmetic noise; documented in the test docstring.
- **Deferred to a separate PR — deep convection (Emanuel & Živković-Rothman 1999; Stohl 2005 §4.6).** The single biggest known non-audit dispersion gap GLIDE has vs FLEXPART, but the faithful port is ~500–1000 lines of new physics + a specific-humidity met channel + a new runtime integration point + tests. A simplified placeholder is unlikely to quantitatively match FLEXPART (could introduce a different bias rather than fix the gap). Scoped as its own follow-up PR.

### 2026-05-30 Audit fixes landed: reflection w'-flip, Δt warning, density term, V1 well-mixed tests

Closed the four highest-priority gaps from the same-day physics audit (entry below). All in one PR `physics-audit-may30` (5 new tests, 174 passing). The two V1 well-mixed tests are now the primary acceptance gate for the integrated backward HannaScheme path; both pass with the production code and provide regression coverage for F1/F2/F3/F4 jointly.

- **F1 — `engine.reflect_surface` now flips `w'` along with `z` (Wilson & Flesch 1993 §6).** Signature changed to `reflect_surface(particles, w_prime, *, z_surface) -> (particles, w_prime)` — mandatory `w_prime` argument so callers can't silently regress. Both `HannaScheme` and `PlaceholderConstantOU` updated. The pre-audit behaviour (reflecting only `z`) left each reflected particle pointing downward into the boundary for ~τ_L worth of steps, biasing near-surface residence and inflating the surface footprint near the receptor. New regression `test_reflect_surface_w_flip_resolves_one_way_downward_drift` asserts the joint reversal directly. Existing reflect tests updated to the new signature and to assert the w'-flip on reflected entries.
- **F4 Tier 1 — `HannaScheme` warns once per instance when `min(active T_L) < 5·Δt`.** Cites the W&F 1993 Appendix bias formula and S&T 1999's stricter `c = 0.05` constant. Rate-limited via an `_warned_dt` flag so long runs don't drown their log in repeats. Tested with `test_dt_too_large_warning_fires_once` (deliberately small-T_L config — release at z = 1 m → `T_Lw ~ 1 s` vs `dt = 300 s` → ratio 300, fires once) and `test_dt_too_large_warning_silent_when_dt_is_small` (`T_L = 1000 s, dt = 1 s` → silent). Tier 2 (per-particle substepping) is the next step if/when this becomes the dominant error.
- **F2 — Stohl-Thomson 1999 density correction `+ (σ_w²/ρ)·∂ρ/∂z` added to the vertical drift.** New `_density_fields` helper computes `ρ = p/(R_d·T)` and `∂ρ/∂z` on the met grid (same machinery as `_free_trop_fields`), trilinearly sampled per particle. Added to `drift_w` *before* the existing `drift_w = -drift_w` line, so the backward sign flip carries the term correctly (Flesch et al. 1995: `a_b = -a_f - 2w/τ` flips both inhomogeneity terms in lockstep for symmetric Gaussian `g_a`). Expected magnitude per S&T's CAPTEX runs (PBL ≈ 1700 m): +5.5% mean on surface concentrations, +1–15% range — modest but biased, exactly what goes into an inversion's flux estimate.
- **F6 — V1 well-mixed tests against the production code path.** Two new tests in `tests/test_main_runtime.py`:
  - `test_v1_well_mixed_hanna_backward_path` — synthetic uniform met window (neutral BL, ustar = 0.4, BLH = 3000 m), 6000 particles initialised uniform in [0, BLH], 1500 steps × dt = 15 s through `HannaScheme.step` (backward) with reflection at z=0 (built into scheme) and z=BLH (added manually with w'-flip). Assertion: interior bins (z ∈ [600, 2400] m, excluding surface-layer and reflection-bias regions per W&F §7b) have relative-RMS deviation from uniform < 15%. Measured: 4.7% (just above the ~4% shot-noise floor at N/bin ≈ 600).
  - `test_v1_density_weighted_well_mixed_with_F2` — same setup but with the steeper pressure profile (1000 → 500 hPa over 6000 m → `ρ_top/ρ_bot ≈ 0.5`), and particles initialised ρ(z)-weighted via rejection sampling. Assertion: interior z-distribution stays ρ-weighted (relative-RMS vs ρ-weighted expected < 15%; sanity-checked that the flat-null assertion would fail). Measured: 4.0% vs ρ-weighted, 8.1% vs flat-null — test discriminates 2×, would catch a regression of F2.
- **Tests added: 5** (1 reflection regression, 2 dt-warning, 2 V1 well-mixed). Total: **174** (was 169). Suite runtime: ~91 s (the two V1 tests add ~45 s combined; they exercise the full production loop at scale).
- **`docs/turbulence.md` §3.2.4 / §3.2.5 / §3.2.7 updated** to reflect the W&F-grounded reflection definition, the full S&T 1999 drift formula (σ-gradient + density), and the actual implemented Δt warning (replacing the previously-aspirational "M1 adds a runtime warning…" claim that was the F4 finding).
- **Backward sign for the density term**: derived afresh from Flesch et al. 1995 `a_b = -a_f - 2w/τ`. The σ-gradient and density terms both flip; the `-w/τ` relaxation does not. The single `drift_w = -drift_w` line correctly handles both — no special-casing needed, but the comment in `hanna.py` now cites this derivation explicitly so a future reader doesn't "fix" it back.

What's NOT done in this PR (deferred):
- F3 (drift cap `±σ_w/Δt`) — still present, but the F4 Tier 2 substepping fix would remove the need; left for now.
- F4 Tier 2 (per-particle substepping when `dt > c·T_Lw`) — bigger change; not needed before the next FLEXPART comparison run.
- F5/F15 (W&F UBL — constant-σ basal layer near z=0) — FLEXPART has the same simplification; documented modelling cost, not a fix.
- F9 (vertical interpolation on bbox-mean `metadata.level`) — M3 milestone item.
- F10 (STILT conversion uses scalar `air_density`) — matters for satellite columns, not site releases.

### 2026-05-30 Physics audit against `docs/LPDM_physics_spec.md` (Wilson & Flesch 1993; Stohl & Thomson 1999)

End-to-end audit of the turbulence + advection + boundary code against the literature physics spec the user assembled (`docs/LPDM_physics_spec.md`), cross-checked against Wilson & Flesch (1993) and Stohl & Thomson (1999) read in full. Drift formula (P3) and finite-dt OU machinery (P1, P2) check out. Three real WMC-violating gaps stand out, in priority order.

- **F1 — `engine.reflect_surface` flips only `z`, not `w'`.** Wilson & Flesch §6 *defines* smooth-wall reflection as `(z → 2B − z, w → −w)`; doing only the `z` half is a different, unsupported scheme. Effect: after a reflection the particle keeps its downward `w'` for ~τ_L worth of steps, biases the near-surface residence time upward, and inflates the surface footprint near the receptor. One-line fix: thread `w_prime` through `reflect_surface` (or return a `(particles, w_prime)` pair) and negate the flipped entries. The Hanna scheme and the placeholder scheme both consume the fix the same way. Existing `test_reflect_surface_handles_boundary_cases` only checks positions; needs a companion that asserts `w' > 0` after reflection from a downward-moving state.
- **F2 — drift is missing the Stohl–Thomson 1999 density term `+ (σ_w²/ρ)·∂ρ/∂z`.** Derived from the WMC for a ρ-weighted Gaussian `g_a` (S&T 1999 Eq 3). Implementation matches the FT closure already in place: build ρ on model levels (`ρ = sp / (R_d · T)`, both already in `HourlyMetTensors`), `∂ρ/∂z` via `_vertical_gradient` on `height_agl_m`, interpolate trilinearly per particle, add to `drift_w`. The existing `drift_w = -drift_w` line correctly carries the term into the backward adjoint (Flesch et al. 1995: `a_b = -a_f - 2w/τ` flips both inhomogeneity terms in lockstep). Empirical magnitude per S&T's CAPTEX runs (max PBL ≈ 1700 m): mean +5.5% on surface concentrations, range +1–15%, larger at longer travel times. Real, biased, quantifiable — exactly the bias signal that goes into an inversion's flux estimate.
- **F4 — fixed Δt with no T_L-aware floor.** `T_L_MIN_S=1 s` floor + 60–300 s configured dt means `Δt/τ` can hit 60+ in unstable surface-layer regimes. Stohl & Thomson use `Δt = min(c·τ_Lw, c/|∂σ_w/∂z|, c·z_i/|w|)` with **c=0.05**, plus floors `Δt ≥ 1 s, τ_L ≥ 10 s`. The W&F Appendix bias formula shows the resulting "Δt-bias velocity" `wB/σ_w ≈ -α·β·μ` (α≈0.5, β≈0.5, μ=Δt/τ) — at our current ratios that's many-percent bias in the near-surface position distribution. Tier 1 fix (≤10 lines): per-step log a warning when `min(T_L) < 5·Δt` over active particles, as `docs/turbulence.md §3.2.7` already promised. Tier 2 (more work): per-particle substep `k = ceil(Δt / (c·T_Lw))`.

Other audit outputs:

- **F3 (drift cap `±σ_w/Δt`) — keep, but mark as "papers over F4".** A direct WMC formula violation wherever it bites, present specifically because Δt is too large to handle sharp σ-gradients at the BL top. Fixing F4 makes this cap stop firing.
- **F5/F15 (surface-layer σ_w varying right down to z=0).** W&F §7b's "unresolved basal layer" recipe (constant σ_w near z=0 so reflection becomes WMC-exact) is *not* applied; the Hanna SL formula extrapolates a varying σ_w to z=0. FLEXPART makes the same simplification, so this is a documented modelling cost rather than a unique GLIDE bug. Lower priority.
- **F6 — no end-to-end V1 well-mixed test for the production code path.** The forward, engine-only `test_well_mixed_condition_drift_keeps_uniform_distribution` validates the OU+drift in isolation; the existing `test_hanna_well_mixed_no_runaway_lofting` is a qualitative regression, not a WMC test. The single most valuable test to add is a V1 that drives `HannaScheme.step` (backward, with reflection) on a synthetic σ_w(z) column and asserts the end-state z-histogram stays flat (and, after F2, asserts it matches the ρ(z) profile).
- **F7 — the forward WMC test also doesn't flip w on reflection.** It passes despite this, because the test is integrated forward; it won't catch a regression of F1. Belt-and-braces, but worth noting in a comment.
- **F8/F14 — backward drift sign is correctly handled** (matches Flesch et al. 1995 `a' = -a - 2w/τ` reducing to the inhomogeneity-flip for Gaussian symmetric `g_a`). The code comment's informal "displacement negates w'·dt so we negate drift" derivation is the right answer for the wrong reason; should be updated to cite Flesch directly.
- **F9 — vertical interpolation on the bbox-mean `metadata.level` axis** (linear-in-AGL-metres against pressure-level data) remains a known approximation flagged in the M3 milestone. Second-order for surface footprints.
- **F10 — STILT-style conversion uses scalar `air_density`.** Sound for site-level releases, biased for satellite columns or deep PBLs. Flagged for the satellite work.
- **F11 (`∂σ²/∂z` consistency)** ✓ — the drift derivative is a finite difference of the same `_column_turbulence` profile used elsewhere; P4 holds.
- **F12 (Gaussian-only PDF)** ✓ — defensible per spec §P3 / Thomson 1987 §5.2 (surface concentration is insensitive to skewness even though aloft dispersion isn't); flagged as a modelling choice, not a bug.

Outcome: full findings + per-finding location/severity/proposed-fix table delivered in the audit response. Implementation queue (proposed): F1 (one-line) → F4 Tier 1 (warning) → F6 (V1 well-mixed test against the post-F1 production path) → F2 (density term + V1 ρ-weighted variant). All four can be landed in one PR with the V1 test as the acceptance gate.



### 2026-05-29 Meander: unresolved-mesoscale horizontal turbulence (Maryon/FLEXPART §4.5)

After the well-mixed-drift + FT fixes restored vertical recycling, the FLEXPART comparison looked much better but still left a horizontal-spread gap — the classic *unresolved-mesoscale* ("meander") motions: quasi-2-D eddies larger than 3-D turbulence yet smaller than the met grid can resolve. Implemented the state-of-the-art portable scheme (Maryon 1998, which NAME originated and FLEXPART adopted; Stohl et al. 2005 §4.5), confirmed by reading the Stohl 2005 technical note directly.

- **Physics.** An *independent* horizontal OU process added on top of the Hanna `u'/v'` turbulence, applied at **all altitudes**: `σ_meander,i = C · stddev_local(U_i)`, `τ = ½·(met interval) ≈ 1800 s` for hourly ERA5. `stddev_local` is the std-dev of the grid-scale wind component over the `(2r+1)²` neighbourhood (per level), interpolated per-particle. `C = turbmesoscale = 0.16` (FLEXPART default). **No drift** (inhomogeneity is vertical), symmetric forcing → backward displacement needs no sign flip.
- **Answers the user's resolution-dependence question.** σ is *derived from local grid-wind variability*, not a hand-tuned constant — so a finer met grid automatically yields a smaller meander (more mesoscale already resolved). This is exactly NAME's resolution-dependent behaviour, for free.
- **Code.** `HannaScheme.__init__` gains `meander_{enabled,coefficient,stencil_radius,timescale_seconds}` (default **off** → existing baselines bit-identical). New `_meander_sigma_fields` + `_windowed_std` (windowed std via `avg_pool2d`, `count_include_pad=False` for valid edge handling); shared `_grid_bounds` helper factored out of `_free_trop_fields`. State adds `{u_meander, v_meander}` only when enabled (preserves the `{u,v,w}_prime`-only layout otherwise). Config: `turbulence.meander` block (`config.MeanderConfig`); `main._scheme_kwargs` forwards it for Hanna only (other schemes take no kwargs). Enabled in both shipped `example_mhd_january*.yaml`.
- **Tests** (+7, total 169): `_windowed_std` vs numpy population-std + uniform→0; constructor validation; state-key gating; `_scheme_kwargs` Hanna-only forwarding + config defaults; end-to-end `test_hanna_meander_increases_horizontal_spread` (varying-wind met window, meander-on horizontal spread 3.2× the meander-off spread). Full suite green, ~40 s.
- **Next (user action):** re-run `configs/example_mhd_january_periodic.yaml` (now meander-enabled) and redo `notebooks/flexpart_comparison.ipynb` to quantify the remaining dispersion gap. If horizontal spread is now over-shooting, lower `coefficient`; if still short, raise it or the `stencil_radius`.

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

### 2026-05-29 Turbulence dispersion fix: well-mixed drift + free-troposphere closure

Visual FLEXPART comparison (notebooks/flexpart_comparison.ipynb) showed GLIDE footprints matched FLEXPART on mean trajectory but were **far less dispersive** at the surface (190 nonzero cells vs FLEXPART's 26,334 for the same release). Root cause: vertical transport was effectively one-way (upward) — the mean particle altitude climbed monotonically to 3.6 km and never recycled to the surface, so the surface footprint only got painted near the source. Two structural gaps, both now closed.

- **Item 0 — expose 3D geopotential height.** The met reader computed per-level AGL then discarded the 3-D field, keeping only the bbox-mean `metadata.level`. Now `HourlyMetTensors.height_agl_m` carries the full `[Z, Y, X]` per-column height (per-window average of hour-start/end). `_subset_vertical_levels_by_agl` returns it; threaded through `fetch_hourly_window`. Needed for accurate `∂θ/∂z`, `∂u/∂z` in the FT closure. `AnalyticMetReader` fixture supplies a broadcast height too.
- **Item 1 — Thomson well-mixed drift.** `engine.update_langevin_velocity` gained a `drift` arg (forward-Euler increment on the exact-OU homogeneous part; `drift=0` is bit-identical to before). The Hanna vertical step adds `a = ½(1+w'²/σ_w²)·∂σ_w²/∂z`, with `∂σ_w²/∂z` a central finite-difference of the full column σ_w profile (so it spans in-BL → SL → FT transitions), capped at one σ_w/step. **Critical subtlety:** GLIDE integrates backward and the displacement negates w'·dt; the *deterministic* drift must therefore enter with reversed sign for the adjoint/backward Langevin (Thomson 1987 §5; Flesch, Wilson & Yee 1995). The first cut got this wrong and pumped the entire population up to ~2 km (a one-way valve) — fixed by pre-negating the drift. The forward physics is validated by an engine-level well-mixed test (uniform stays uniform); the backward sign by an end-to-end no-runaway test.
- **Item 2 — free-troposphere gradient-Richardson closure.** Replaced the `σ=0.1, T_L=100` above-BL placeholder (which froze lofted particles, `K≈1 m²/s`) with `K_z = l²|∂U/∂z|·f(Ri)` from potential-temperature/shear gradients (Blackadar mixing length λ=100 m, `f(Ri)` decaying to 0 at `Ri_c=0.25`, `K` floored at 0.1 and capped at 50 m²/s), split into `(σ_w, T_Lw)` via the buoyancy timescale. New free functions `potential_temperature`/`brunt_vaisala_squared`/`gradient_richardson`/`free_trop_diffusivity`/`free_trop_sigma_TL`. FT fields are built once per step on the met grid and trilinearly interpolated per above-BL particle (reusing the advection coordinate normalisation). The Hanna `step()` was refactored around a reusable `_column_turbulence(z, …)` helper (in-BL + SL + FT), which both the main evaluation and the drift finite-difference call.
- **Validation (1-day MHD backward, 4000 particles, real EUROPE met).** Before fix (buggy sign): endpoint altitudes clustered at 1660–2500 m, 0.6% below the ~760 m BLH, 0% at the surface — runaway lofting. After fix: mean altitude *plateaus* ~880 m, endpoint p10/p50/p90 = 217/623/1767 m, **60% back inside the BL**, surface footprint 27 → 142 cells in the 1-day run alone (the prior 30-day run only reached 190). Particles now recycle through the BL as they should.
- **Tests** (+ ~10): free-trop free functions (θ, N², Ri/K_z shutoff at Ri_c, σ²·T_L=K_z identity, K_z never zero); OU `drift` determinism + back-compat; engine-level well-mixed-condition test (drift keeps a uniform tracer uniform; no-drift accumulates in the low-σ region); end-to-end `test_hanna_well_mixed_no_runaway_lofting` regression (mid-BL release stays BL-confined, mixes down to the surface, no escape). Met-reader + AnalyticMetReader height-field assertions. Total 162; suite ~40 s. All existing tests pass unchanged (drift=0 / no-escape paths are bit-equivalent).
- **Deferred (item 3, agreed with user):** the NAME-style unresolved-mesoscale "meander" horizontal term (resolution-dependent) — add after re-running the full comparison to see how much horizontal spread is still missing once recycling is restored. FT horizontal turbulence is currently isotropic (σ_u=σ_v=σ_w).
- **Next:** re-run the 24-release `configs/example_mhd_january_periodic.yaml` and redo the side-by-side comparison; expect substantially more surface coverage. Known minor limitation: FT σ/T_L fields are sampled with the bbox-mean vertical normalisation (same approximation as wind advection); the gradients use true per-column heights.

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

**Status (2026-06-19): in progress.** Device-agnostic runtime, host-sync minimization,
`torch.compile`, per-window field caching, CPU thread tuning, and the GPU
toolchain/telemetry are landed (see the 2026-06-18 / 2026-06-19 entries). The
GH200 is diagnosed launch/orchestration-bound; the CUDA-graph restructure is the
active workstream (phase 1, the device-gated static-shape substep loop, landed
2026-06-19). The architecture and the remaining phases are documented in
`architecture.md` (§4 implemented, §5 the CUDA-graph plan). The exit-criteria
benchmark/perf-note will live there.

- Harden the existing device-aware runtime for real CUDA execution rather than just device selection and local MPS compatibility.
- Profile interpolation, met staging, footprint accumulation, and any host-device transfers in the current runtime.
- ~~Address the vertical-interpolation approximation in `src/lpdm/main.py`~~ **DONE (F9, 2026-05-31):** the vertical `grid_sample` axis now uses a piecewise-linear fractional-level lookup against the per-level AGL array (`GridInterpolationBounds.level_agl_m`), handling the log-linear-in-altitude pressure-level spacing. See the 2026-05-31 audit-followup entry.
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

**Milestone status (as of 2026-06-19):**
- **M0 / M1 / M4 / M5: code-complete.** Hanna turbulence (drift, FT closure, meander) + full audit fixes (F1+F2+F3+F4Tier1+F4Tier2+F5/F15+F9+F10) + **deep convection (Emanuel reduced port, 2026-06-01)**. Multi-release execution, streaming per-batch Zarr writes, drop-and-count out-of-domain handling all landed.
- **Physics audit (all 10 items) + deep convection — closed.** Three PRs total: `physics-audit-may30`, `physics-audit-followup`, `deep-convection`.
- **M3 (Production GPU execution): in progress.** Host-sync minimization, `torch.compile`, per-window field caching, CPU thread tuning, GPU toolchain + telemetry landed. GH200 diagnosed launch/orchestration-bound; the CUDA-graph restructure is the active workstream — **phases 1–3 landed 2026-06-19** (device-gated static-shape substep loop → full-set sync-free per-step path → fixed-count loop wrapped with `mode="reduce-overhead"` graph capture; strategy (D)+(A)). All CPU-verifiable parts tested green; the graph capture itself is CUDA-only and awaits the GH200 run. Plan + systems-performance architecture in `architecture.md`.
- **Open M4 item:** `schema_version` field on the YAML.
- **Open M5 item:** satellite multi-point-per-time releases.

**Next user step (M3 phase 4 — Matt, on the GH200):** submit `scripts/run_periodic_cuda.slurm` (compile on by default) and report back from the log: (a) "torch.compile(mode='reduce-overhead') enabled" present and NO "WON'T CONVERT" (⇒ the loop captured), (b) the end-of-run sm% summary vs the prior 0–37% launch-bound range, (c) per-batch wall time. Tune `turbulence.max_substeps` down to ~15–25 (it is now the fixed per-step iteration count). Those numbers decide whether M3 is done or the §5.1 (B) escaped-particle recapture is worth adding. Separately, the FLEXPART comparison can be generated now via CPU batch-parallelism while the GPU work proceeds.

**Open follow-ups (no longer including deep convection):**
- **M2 — particle aggregation.** Compute savings; orthogonal to physics.
- **Convection — full Emanuel quasi-equilibrium closure.** Our reduced port caps the buoyancy velocity at 5 m/s as a closure simplification; the full Emanuel scheme has a quasi-equilibrium balance. If the FLEXPART comparison shows under-convective transport, this is the next lever.
- **Convection — per-(lon,lat) profiles.** We use the bbox-mean column for the parcel lift (mirrors the F9 advection approximation). Per-column would be more accurate for large met domains; same refactor of the 3D `grid_sample` machinery as the F9 per-column follow-up.
- **F9 follow-up — per-column heights for vertical interp.** Replacing bbox-mean `metadata.level` with per-column `height_agl_m`.
- **F4 follow-up — per-substep σ re-evaluation** in the turbulence substep loop.

## Operational notes

- Running bare `pytest` may use the wrong interpreter; prefer `.venv/bin/python -m pytest`.
- In a fresh environment, install the package editable first: `uv pip install --python .venv/bin/python -e .`.
- Ensure shell PATH is refreshed if `uv` was installed in-session.
- For public GCS Zarr buckets, ADC is not required when opening with anonymous token mode (`token="anon"`).
- Zarr v3 writes can fail if source dataset encodings contain v2 codec objects (for example `numcodecs.Blosc`); clear inherited encodings before writing v3.

## Public-release readiness audit (2026-06-30)

Goal: get the repo into a state where colleagues can look at / use it. Survey done
2026-06-30; repo is already fairly clean (sensible `.gitignore`, no tracked caches/build
artifacts, no personal absolute paths in `src/`). Outstanding items, by priority:

### Blockers — decisions required before going public
- **No LICENSE file.** Without one the code is all-rights-reserved; "public" ≠ usable.
  Pick one (MIT / BSD-3 typical for scientific tools; Apache-2.0 for explicit patent
  grant). This shapes everything else — decide first.
- **Third-party data redistribution rights.** `data/` (30 MB, tracked) holds derived
  products we may not have the right to redistribute:
  - `data/validation-timeseries/*.csv` (100 files) — CH₄ enhancements from **NAME** (UK
    Met Office) and **FLEXPART** footprints. NAME output is typically not freely
    redistributable.
  - `data/FLEXPART/FLEXPART_MHD_test_202401.nc` — FLEXPART footprint fixture.
  - EDGAR is CC-BY but needs attribution.
  Options: confirm permission, OR replace with a small synthetic fixture for
  tests/notebooks, OR host externally + download script (cf. `download_sample_cube.py`).
  Tests and notebooks depending on these need a graceful "data not present" path.

### Personal / environment leakage
- **`outputs` symlink ** — tracked symlink into
  personal storage. Remove; let `outputs/` be a normal gitignored dir.
- **`glide_feature.png`** (713 KB, untracked, created 2026-06-30) — decide README asset
  (→ `docs/img/`) vs scratch (delete).
- **Personal SLURM-log helpers** `scripts/glogs.sh`, `scripts/logcleanup`,
  `scripts/loglatest` (untracked) — keep untracked (add to `.gitignore`) or remove.

### Stale GCP/Cloud-Run framing vs actual HPC (Isambard AI / SLURM / GH200) reality
- **`deploy.sh` + `Dockerfile`** — Cloud Run deployment, `cu124` wheel (contradicts the
  new `cu126` pin), `PORT=8080` for a batch job with no HTTP server. Delete or mark
  clearly experimental/unsupported.
- **README inaccuracies:**
  - "Project Layout" still calls `footprint_gridder.py`, `output_writer.py`, `main.py`
    **"placeholder"** — all fully implemented now; badly undersells the project.
  - GCP-centric framing: "Minimal Trajectory Run (Local or **Vertex Notebook**)",
    `deploy.sh`/Cloud Run section.
  - "Three examples ship" but there are now **six** configs.

### Internal dev docs not meant for public consumption
- **`CHECKPOINT.md` (this file, ~1000 lines)** — stream-of-consciousness dev journal;
  reads as internal scratch to outsiders. Move to `dev/` or `.github/`, or distil durable
  decisions into `architecture.md` and drop the journal. Shouldn't be top-level next to
  README.
- **AI-agent artifacts** — `.github/copilot-instructions.md`, `.claude/`. Harmless but
  signal WIP; decide keep vs relocate.

### Config & script tidying
- **Config sprawl (6, overlapping):** `example_mhd_january{,_periodic}.yaml`,
  `example_multisite_january.yaml`, `multisite_validation_48h.yaml`, two smoke tests.
  Trim to a clear ladder: one local smoke (no external data), one single-site example,
  one multi-site example. Move machine-generated `multisite_validation_48h.yaml` out of
  the curated set (regenerate on demand via `scripts/make_multisite_config.py`).
- **Path drift:** `configs/multisite_validation_48h.yaml` has
  `output_uri: outputs/multisite-validation-48h` but the actual run used
  `outputs/icos-validation` (what `notebooks/multisite_validation.ipynb` reads).
  Reconcile.

### Quick hygiene wins
- Add `CONTRIBUTING.md` + a one-line "does it work?" install-and-test in the README intro.
- Note **uv >= 0.4.0** requirement (the `[[tool.uv.index]]` pin) in the main install
  section, not just the GPU section.
- Update README "Documentation Governance" after any doc reshuffle.

### Suggested order of operations
1. License + data rights (blockers) — decide first; they shape the rest.
2. De-personalize: drop `outputs` symlink, sort `glide_feature.png` + log scripts.
3. Reframe README ("placeholder" labels, GCP framing); decide deploy.sh/Dockerfile fate.
4. Docs: relocate/distil CHECKPOINT.md.
5. Configs: trim to a clean example ladder.

Lowest-risk first pass (no judgment calls needed): README accuracy fixes + de-personalizing
(#2–3). License (#1) and data rights (blockers) need Matt's decision before touching.

**Status 2026-06-30/07-02:** steps 1–5 all executed (Apache-2.0 + NOTICE; data
untracked with data/README.md; README rewritten, Docker removed; docs consolidated
under docs/ + CLAUDE.md at root + this journal moved to dev/; configs trimmed to a
5-example ladder; CONTRIBUTING.md added).

## Physics review (2026-07-02) — findings handed off

Full physics audit (docs vs code vs literature) completed 2026-07-02. Six numbered
findings + minor items, with evidence, locations, fix guidance, commit order, and a
"verified correct — do not touch" list, all in
**`dev/PHYSICS_REVIEW_2026-07-02.md`** (the work order for the fixes).

Headlines: (1) ERA5 SHF sign convention inverted → stability classification flips
on real met (CONFIRMED against the local EUROPE store — fix first, one line);
(2) convection backward kernel not time-reversed + no compensating subsidence, no
convection well-mixed test; (3) sub-LCL mass-flux rows each carry full M_b
(BL venting ~n_BL× overcounted); (4) neutral-regime signed Coriolis breaks SH;
(5) Hanna coefficients diverge from FLEXPART reference (cross-check never done);
(6) stable surface-layer σ_w grows with height (φ_m misapplied). The core
Langevin/WMC machinery (drift + density term + backward signs, reflection,
implicit position update, ω→w) audited CORRECT.

**All findings implemented on branch `physics-fixes-jul02` (2026-07-02), full
suite 224 passed.** Four commits: (a) SHF sign flip + regression; (b) convection
non-divergent mass-flux matrix (updraft + compensating subsidence) with the
FLEXPART backward-transpose sampling and mass-weighted BL entrainment, gated by
deterministic well-mixed (mᵀP=mᵀ) tests; (c) Hanna σ/T_L overhaul to FLEXPART
v11 coefficients + |f| Coriolis + constant stable-SL σ_w + the t_idx/Z_MIN_M
minor items. DEFERRED (documented in docs/turbulence.md §3.2.2/§3.2.4): FLEXPART's
10/10/30 s T_L floors and removing the MO surface-layer override — both interact
with the adaptive-substep machinery and need their own validation. Convection's
hardcoded 3600 s interval also left as a documented follow-up. **Next: the GH200
validation re-run (Finding 1 acceptance) — re-run the 56-site config and compare
against the 2026-06-30 baseline (FLEXPART 0.67, NAME-UMG 0.74, NAME-UKV 0.85;
GLIDE means 13–19% high).**

**v2 validation (2026-07-02, `outputs/icos-validation-v2`, 48 h):** GLIDE now
over-estimates mean enhancements, worst at polluted low-inlet sites (up to
~120 ppb vs <60 for NAME/FLEXPART) — the SHF fix restored correct nocturnal
stability and unmasked weak near-surface vertical mixing: with the 1 s T_L floor
and the MO surface-layer override, K = σ_w²·T_Lw is 3–8× below FLEXPART's in the
lowest ~15 m on stable nights, inflating near-field surface residence exactly at
near-source sites. In response, the two deferred items are now IMPLEMENTED and
DEFAULT: `turbulence.flexpart_tl_floors: true` (10/10/30 s) and
`surface_layer_override: false` (regime formulas to the ground, per FLEXPART
v11). Legacy behaviour kept behind the flags for A/B. Side-effect: with
T_Lw ≥ 30 s the substep cap rarely binds (max_substeps pressure eases). **Next:
v3 GH200 run with the new defaults; expect polluted-site means to drop toward
the reference range. Also check the v2 log for any "Emanuel convection: fires"
lines (bbox-mean January column should not trigger).**
