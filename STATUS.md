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
- **New frontier (deferred):** with met handled, the top wall consumers are now the
  **untimed per-step "residual"** (wind-mean diagnostic, mask/alive/escape
  bookkeeping, per-batch particle-gen + output writes, Python loop overhead) and
  **convection** (~250 ms/window). These, not more CUDA-graph work, are the next
  levers. First step: sub-time the residual to see what's in it.

## Hardware / deployment

Primary GPU target is **Isambard AI (NVIDIA GH200)** via SLURM; the local dev box
is CPU-only. The codebase stays device-agnostic. Containerised/cloud packaging was
removed and returns only once the architecture settles (see README "Next Steps").

## What's next (roughly ordered)

1. **Physics validation** vs NAME/FLEXPART — the gating item; treat current results
   as indicative, not verified.
2. **Performance:** sub-time and attack the `residual` and `convection` phases.
3. **Column releases** (tall-tower / aircraft profiles via importance sampling).
4. **Satellite-style releases** (many irregular soundings per overpass; the flat
   `release` axis already accommodates the geometry).
5. **Particle aggregation** (far-field merging for compute savings; mass/moment
   preserving, no-merge zones in the BL).
6. **Convection refinements** — full Emanuel quasi-equilibrium closure and
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
