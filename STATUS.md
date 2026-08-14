# GLIDE — project status

A snapshot of what works, what's pending, and the latest results. Keep it honest
and current; narrative history is in git. For the user-facing overview see
[README.md](README.md); for design rationale see [dev/decisions/](dev/decisions/);
for the physics/systems reference see [docs/](docs/).

> **Research code, under active development. The physics has NOT been fully
> validated — not for production use.**

_Last updated: 2026-07-24._

## What GLIDE is

A backward-in-time Lagrangian Particle Dispersion Model for greenhouse-gas
footprints, in pure PyTorch. It bypasses the two traditional LPDM bottlenecks by
**streaming analysis-ready ARCO ERA5 from a Zarr store** (fetching only the chunks
a run needs) and running the **per-step physics on the GPU**, device-agnostically
(CUDA / MPS / CPU). See [dev/decisions/](dev/decisions/) for why.

## Implemented & tested

- **Core backward LPDM** — RK2 advection + Ornstein–Uhlenbeck / Langevin turbulence,
  surface reflection, mass-conserving. ([docs/LPDM_physics_spec.md](docs/LPDM_physics_spec.md))
- **Turbulence** — Hanna (1982) BL scheme + free-troposphere Richardson closure +
  Thomson well-mixed drift, behind a swappable `TurbulenceScheme` interface.
  ([docs/turbulence.md](docs/turbulence.md))
- **Convection** — reduced Emanuel mass-flux (non-divergent), once per met window.
  ([docs/convection.md](docs/convection.md))
- **Terrain-following vertical coordinate** — pressure-level met resampled onto a
  fixed AGL grid per window; fixes zero surface footprint over high terrain.
  ([dev/decisions/0003-terrain-following-agl-coordinate.md](dev/decisions/0003-terrain-following-agl-coordinate.md))
- **Multi-site simultaneous releases** (`multi_point_periodic`) sharing met windows —
  the efficient way to grow a run across a network.
- **Streaming output** — `footprints.zarr` (5-D), `endpoint_particles.parquet`,
  `trajectory_diagnostics.parquet`, `run_metadata.json`; written per batch.
- **GPU execution path** — whole per-step body captured as one CUDA graph
  (`torch.compile(mode="reduce-overhead")`), device-gated; eager path is the
  numerical reference. ([docs/architecture.md](docs/architecture.md))
- **Tests** — 255 passing, ~90% coverage; includes analytic dispersion checks
  (Taylor, OU autocorrelation, Gaussian-plume footprint, PDE diffusion limit,
  terrain transport) and CPU guards for the GPU capture.
  ([docs/VALIDATION.md](docs/VALIDATION.md))

## Physics validation status — NOT complete

The transport physics is not yet validated against external references. Comparison
machinery exists (`src/lpdm/comparison.py`, `notebooks/`), and a NAME/FLEXPART/
EDGAR comparison is in progress but not signed off.

- **v2 (2026-07-02)** over-estimated mean CH₄ enhancements, worst at polluted
  low-inlet sites, traced to weak near-surface mixing on stable nights. Response:
  FLEXPART v11 `T_L` floors and regime-formulas-to-ground are now the defaults.
- **Terrain-following coordinate (2026-07-16)** changes footprint magnitudes
  **everywhere**, not just over mountains — so **all** site comparisons (not only
  elevated sites) need re-running before any magnitude is trusted.
- **Next:** a GH200 validation re-run with the current defaults + terrain fix,
  compared against NAME/FLEXPART on identical release setups.

## Performance status

The GPU per-step "launch-bound" problem is solved (whole-step CUDA-graph capture).
The current picture, from a validated GH200 A/B on a representative multi-site run
(2026-07-24):

- **met I/O**: a per-hour processed-met cache cut `met_fetch` ~8× at representative
  scale (−32% wall there; less at full 56-site scale, where met is a smaller share).
- **New frontier (deferred):** with met handled, the standout wall consumer is the
  **untimed per-step "residual"** (wind-mean diagnostic, mask/alive/escape
  bookkeeping, per-batch particle-gen + output writes, Python loop overhead) — this,
  not more CUDA-graph work, is the next lever. First step: sub-time the residual to
  see what's in it.
- **Convection needs re-profiling.** That A/B measured convection at ~250 ms/window
  and flagged it as a co-frontier, but it ran on the *pre*-vectorisation parcel lift.
  The subsequent fix (commit `9238f1b`) removed the per-level loop and ~1100
  host syncs/window, so convection should now be much cheaper — how much is
  unmeasured. Re-profile before treating it as a top consumer again.

## Hardware / deployment

Primary GPU target is **Isambard AI (NVIDIA GH200)** via SLURM; the local dev box
is CPU-only. The codebase stays device-agnostic. Containerised/cloud packaging was
removed and returns only once the architecture settles (see README "Next Steps").

## What's next (roughly ordered)

1. **Physics validation** vs NAME/FLEXPART — the gating item; treat current results
   as indicative, not verified.
2. **Native model-level met (HIGH PRIORITY)** — switch the vertical met source from
   ERA5's 37 pressure levels to its native 137 hybrid model levels, available
   analysis-ready and already on the 0.25° grid at
   `gs://gcp-public-data-arco-era5/ar/model-level-1h-0p25deg.zarr-v1`. Model levels
   are terrain-following by construction (no below-ground levels; the Finding-7
   problem never arises) and far finer in the boundary layer — lowest level ~10 m
   AGL, ~20 levels in the lowest ~1.5 km, vs the pressure grid's ~0/300/600 m
   (1000/975/950 hPa). This is the main lever left on near-surface / BL accuracy,
   the regime footprints care about most, and it lets us **retire** most of the
   pressure→AGL resample rather than extend it (it's what FLEXPART does natively).
   **Plumbing is now in place** — what remains is the validation re-run:
   - Download: `scripts/download_sample_cube.py --levels model` (merges the
     model-level 3D fields with surface fields from the pressure/surface store, tags
     the cube `glide_vertical_coordinate=model_level`).
   - Reader: `met_reader` auto-detects model mode from that attr, reads heights from
     `geopotential` directly (`(z − z_sfc)/g`, unchanged), and **reconstructs
     per-level pressure hydrostatically** from geopotential + surface pressure +
     virtual temperature (`model_level_pressure_pa`). Pressure is needed for
     omega→w, air density, and convection; we do *not* use hybrid a/b coefficients
     because ARCO does not ship them and a third-party table cannot be trusted to
     match the data (see [dev/decisions/0010](dev/decisions/0010-model-level-met-reader.md)).
     A run just points `io.zarr_store` at a model-level cube — no config change.
   - Still to do: the GH200 validation re-run on model-level met (~3.7× the vertical
     data; stream only the lowest ~60–90 levels), and a look at whether the
     hydrostatic-pressure approximation is tight enough for convection.
   **Sequencing:** like the terrain fix, this shifts footprint magnitudes
   *everywhere*, so it should land **before** the big validation re-run (#1) —
   otherwise we validate a configuration we're about to change.
3. **Performance:** sub-time and attack the `residual` phase. Convection's headline
   perf issue — the parcel-lift host-sync storm — is already fixed (`9238f1b`);
   any further convection perf work is gated on a re-profile to confirm it's still
   a top consumer.
4. **Column releases** (tall-tower / aircraft profiles via importance sampling).
5. **Satellite-style releases** (many irregular soundings per overpass; the flat
   `release` axis already accommodates the geometry).
6. **Particle aggregation** (far-field merging for compute savings; mass/moment
   preserving, no-merge zones in the BL).
7. **Convection refinements** — full Emanuel quasi-equilibrium closure and
   per-column (vs bbox-mean) profiles, if the validation shows under-convection.

## Open follow-ups from reviews

- Physics (2026-07-02 review): per-column vertical interpolation / convection
  profiles (currently bbox-mean); per-substep σ re-evaluation; convection's
  hardcoded 3600 s interval.
- Tests (2026-07-16 review): forward/backward reciprocity test (T6) deferred;
  otherwise the analytic-test work order is complete
  ([docs/VALIDATION.md](docs/VALIDATION.md)).

The full review documents were one-shot work orders; they live in git history
(`dev/`, removed 2026-07-25), not the working tree.
