# Project status

A snapshot of what works, what is not yet trustworthy, and what comes next. For
what GLIDE is, see [README.md](README.md); for the physics and engineering, see
[docs/](docs/); for why the major choices were made, see
[dev/decisions/](dev/decisions/). Narrative history lives in git.

_Last updated: 2026-08-14._

> **Research code under active development. The physics has not been validated
> against other models or observations. Not for production use.**

---

## What works

| Area | State |
| --- | --- |
| **Core backward LPDM** | RK2 advection, exact-OU Langevin turbulence with the full Thomson well-mixed drift and Stohl–Thomson density correction, ground reflection, per-particle adaptive sub-stepping. Mass-conserving. ([docs/physics.md](docs/physics.md)) |
| **Turbulence** | Hanna (1982) in three regimes, aligned to FLEXPART v11; gradient-Richardson free-troposphere closure; Maryon meander. Behind a swappable `TurbulenceScheme` interface. ([docs/turbulence.md](docs/turbulence.md)) |
| **Convection** | Reduced Emanuel mass-flux scheme, non-divergent matrix, once per meteorology window. ([docs/convection.md](docs/convection.md)) |
| **Terrain-following vertical coordinate** | Meteorology resampled onto a fixed AGL grid per window, with slope-corrected vertical velocity. Fixed the zero-surface-footprint problem over high terrain. |
| **Model-level meteorology** | Reader auto-detects hybrid model levels and reconstructs per-level pressure hydrostatically. Download path implemented (`--levels model`). **Plumbing complete; not yet validated.** |
| **Multi-site releases** | `multi_point_periodic` — sites share meteorology windows, so per-window costs amortise across the whole network. |
| **Streaming output** | `footprints.zarr` (5-D), `endpoint_particles.parquet`, `trajectory_diagnostics.parquet`, `run_metadata.json`, written per batch. |
| **GPU execution** | Whole per-step body captured as one CUDA graph, device-gated; the eager CPU path is the numerical reference. ([docs/architecture.md](docs/architecture.md)) |
| **Tests** | 298 passing, 90% statement coverage, ~135 s, no network. Includes analytic dispersion checks and well-mixed gates. ([docs/VALIDATION.md](docs/VALIDATION.md)) |

---

## What is not validated

**This is the gating item for everything else.** The transport physics has not
been compared against external references. The comparison machinery exists
(`src/lpdm/comparison.py`, the notebooks under `notebooks/`), and a
NAME/FLEXPART/EDGAR comparison is in progress, but nothing is signed off.

Three things a reader should know about the comparison history:

- An earlier round over-estimated mean CH₄ enhancements, worst at polluted
  low-inlet sites, and this was traced to weak near-surface mixing on stable
  nights — the vertical diffusivity $K = \sigma_w^2 T_{Lw}$ was collapsing as
  $z \to 0$. FLEXPART's Lagrangian-timescale floors and running the regime
  formulas to the ground are now the defaults, which fixes it. See
  [docs/turbulence.md §5](docs/turbulence.md#5-the-lagrangian-timescale-floors).
- The terrain-following coordinate changes footprint magnitudes **everywhere**,
  not only over mountains. So **all** site comparisons predating it need
  re-running before any magnitude is trusted — not just the elevated sites.
- The stable-regime $T_{Lv}$ was 35% too small until 2026-08-17 (it divided by
  $\sigma_u$ rather than $\sigma_v$). The effect is confined to meridional
  turbulent spread under stable stratification — roughly 20% narrow in the
  crosswind direction on stable nights, with no effect on vertical mixing or on
  footprint magnitude. Found by comparing against an independent implementation
  of the same Hanna equations.

The same "changes magnitudes everywhere" argument applies to the switch to native
model levels, which is why it is sequenced *before* the big validation run below
rather than after.

---

## Performance

The per-step GPU path is no longer the bottleneck. Whole-step CUDA-graph capture
took the GH200 from ~30 ms to ~5 ms per step (launches ~1,250 → ~106, GPU busy
17% → 37%), and per-hour meteorology caching cut `met_fetch` roughly 8× at
representative scale (−32% wall there; less at full 56-site scale, where
meteorology is a smaller share).

On a representative multi-site run the per-step phase is now only ~18–23% of
wall. What remains:

- **The untimed per-step "residual"** is the standout consumer — the wind-mean
  diagnostic, mask/liveness bookkeeping, per-batch particle generation and output
  writes, and Python loop overhead. This, not more graph work, is the next lever.
  First step is to sub-time it to see what is actually in it.
- **Convection needs re-profiling.** The A/B that flagged it as a co-frontier at
  ~250 ms/window ran *before* the parcel lift was vectorised; that change removed
  a per-level loop and ~1,100 host synchronisations per window, so convection
  should now be far cheaper. How much is unmeasured. Re-profile before treating
  it as a top consumer again.

---

## Hardware

Primary GPU target is **Isambard AI (NVIDIA GH200)** via SLURM; the local
development box is CPU-only. The codebase stays device-agnostic and the eager CPU
path remains the numerical reference. Containerised/cloud packaging was removed
and returns only once the architecture settles.

---

## What is next, roughly in order

1. **Validate the model-level meteorology path.** The reader, the hydrostatic
   pressure reconstruction and the download are all implemented. What remains is
   a GH200 run on a model-level cube (~3.7× the vertical data — stream only the
   lowest ~60–90 levels) and a check on whether the hydrostatic pressure
   approximation is tight enough for convection. This lands *before* item 2,
   because it shifts footprint magnitudes everywhere.

2. **Physics validation against NAME and FLEXPART** on identical release setups,
   with the current defaults. Everything downstream depends on this.

3. **Performance: sub-time and attack the residual phase.** Any further
   convection work is gated on the re-profile above.

4. **Column releases** — tower inlets and aircraft profiles via importance
   sampling over a pressure-weighted vertical PDF.

5. **Satellite-style releases** — many irregular soundings per overpass, each
   with its own averaging kernel. The flat `release` axis already accommodates
   the geometry; the generator and weighting remain.

6. **Particle aggregation** — far-field merging for compute savings, preserving
   mass and moments, with no-merge zones in the boundary layer.

7. **Convection refinements** — the full Emanuel quasi-equilibrium closure, and
   per-column rather than bounding-box-mean profiles, if validation shows
   under-convective transport.

---

## Known follow-ups

These are documented approximations rather than bugs; each is stated in the
relevant docs page with its consequence.

- **Per-column vertical interpolation and convection profiles.** Both currently
  use the bounding-box mean. Same 3-D refactor for both.
- **Per-sub-step $\sigma$ re-evaluation.** FLEXPART re-evaluates $\sigma$ and
  $T_L$ every sub-step; GLIDE holds them at the outer-step value (the
  velocity-dependent part of the drift *is* re-evaluated).
- **Convection interval hardcoded at 3600 s.** Correct for hourly ERA5 only.
- **Sub-hourly meteorology cadence not supported.** GLIDE brackets on whole
  hours, so a 3-hourly source must be interpolated up to hourly first.
- **Forward/backward reciprocity test deferred.** It would test the backward
  formulation directly, which nothing else does.
