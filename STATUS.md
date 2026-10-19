# Project status

A snapshot of what works, what is not yet trustworthy, and what comes next. For
what GLIDE is, see [README.md](README.md); for the physics and engineering, see
[docs/](docs/); for why the major choices were made, see
[dev/decisions/](dev/decisions/). Narrative history lives in git.

_Last updated: 2026-10-19._

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
| **Model-level meteorology** | Reader auto-detects hybrid model levels and reconstructs per-level pressure hydrostatically. Download path implemented (`--levels model`). **Plumbing complete; not yet validated.** Becomes the default in task A01 ([decision 0017](dev/decisions/0017-model-levels-default.md)). |
| **Multi-site releases** | `multi_point_periodic` — sites share meteorology windows, so per-window costs amortise across the whole network. |
| **Streaming output** | `footprints.zarr` (5-D), `endpoint_particles.parquet`, `trajectory_diagnostics.parquet`, `run_metadata.json`, written per batch. |
| **GPU execution** | Whole per-step body captured as one CUDA graph, device-gated; the eager CPU path is the numerical reference. ([docs/architecture.md](docs/architecture.md)) |
| **Tests** | 319 passing, 90% statement coverage, about 2.5 minutes, no network. Includes analytic dispersion checks and well-mixed gates. ([docs/VALIDATION.md](docs/VALIDATION.md)) |

---

## What is not validated

**This is the gating item for everything else.** The transport physics has not
been compared against external references in a way that can be signed off. The
comparison machinery exists (`src/lpdm/comparison.py`, the notebooks under
`notebooks/`), and earlier NAME and FLEXPART comparisons were exploratory. The
validation that counts is the pre-registered ladder in
[docs/validation-plan.md](docs/validation-plan.md), run once the physics is
frozen.

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
model levels, which is why it is sequenced before the physics freeze and the
validation (task A01).

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
  First step is to sub-time it to see what is actually in it (task A1).
- **Convection needs re-profiling.** The A/B that flagged it as a co-frontier at
  ~250 ms/window ran *before* the parcel lift was vectorised; that change removed
  a per-level loop and ~1,100 host synchronisations per window, so convection
  should now be far cheaper. How much is unmeasured. Re-profile before treating
  it as a top consumer again.

---

## Hardware

Primary GPU target is **Isambard AI (NVIDIA GH200)** via SLURM; the local
development box is CPU-only. The codebase stays device-agnostic and the eager CPU
path remains the numerical reference. An earlier container setup was removed;
packaging, including an Apptainer container for Isambard, returns for the v1
release (task B8).

---

## What is next

The plan is [dev/roadmap.md](dev/roadmap.md): every item with an ID, the v1
scope and release gate, and the sequence. The tasks themselves are in
[dev/agent/QUEUE.md](dev/agent/QUEUE.md), and the validation protocol is
[docs/validation-plan.md](docs/validation-plan.md). In outline:

1. **Prepare** (human): environment, data inventory, FLEXPART v11 reference runs
   on ERA5, tracer-experiment data.
2. **Before the physics freeze**, in order: fix the footprint time-bin labels
   (A00); validate model-level meteorology and make it the default (A01); golden
   fixtures and benchmarks (A0); sub-time the residual phase (A1); optional
   kernel deposition (A2); per-particle time steps (A3); local coordinates and
   half-precision meteorology (A4); sorting particles for memory locality
   (A4b); convergence studies that set the defaults
   (A5); the reciprocity test (A6); in-batch parameter ensembles (A7).
   In parallel: meteorology from [met-bazarr](https://github.com/mrghg/met-bazarr)
   (B1, [decision 0018](dev/decisions/0018-met-bazarr.md)), smaller outputs,
   column and satellite releases, the high-resolution near-receptor footprint,
   figure tooling and packaging.
3. **Freeze** the physics (`v0-physics`) with an external audit (A8).
4. **Pre-register** the validation on OSF, with tolerances computed from
   reference data only (A8a, A8b).
5. **Validate**: Tier 1 component comparison with FLEXPART, Tier 2 whole-model
   comparison, Tier 3 tracer experiments (A9 to A11).
6. **Release v1**, then the model paper.

Particle aggregation (roadmap 6b) and convection refinements (6c) are after v1.

---

## Known follow-ups

These are documented approximations rather than bugs; each is stated in the
relevant docs page with its consequence.

- **Per-column vertical interpolation and convection profiles.** Both currently
  use the bounding-box mean. Same 3-D refactor for both.
- **Per-sub-step $\sigma$ re-evaluation.** FLEXPART re-evaluates $\sigma$ and
  $T_L$ every sub-step; GLIDE holds them at the outer-step value (the
  velocity-dependent part of the drift *is* re-evaluated).
- **Meander timescale is not tied to the met cadence.** It defaults to 1800 s,
  half the hourly ERA5 interval, and stays there on a coarser source. Left as an
  explicit knob because changing it moves results.
- **Forward/backward reciprocity test.** It tests the backward formulation
  directly, which nothing else does. Planned as task A6.
