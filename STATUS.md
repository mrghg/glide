# Project status

A snapshot of what works, what is not yet trustworthy, and what comes next. For
what GLIDE is, see [README.md](README.md); for the physics and engineering, see
[docs/](docs/); for why the major choices were made, see
[dev/decisions/](dev/decisions/). Narrative history lives in git.

_Last updated: 2026-11-25._

> **Research code under active development. The physics has not been validated
> against other models or observations. Not for production use.**

---

## What works

| Area | State |
| --- | --- |
| **Core backward LPDM** | RK2 advection, exact-OU Langevin turbulence with the full Thomson well-mixed drift and Stohl–Thomson density correction, ground reflection, turbulence sub-stepping with one fixed count for every particle (16 at the 60 s step), every coefficient re-evaluated each sub-step (decision 0027); the boundary-layer top treated as a Thomson–Physick–Maryon interface (decision 0028). Mass-conserving. ([docs/physics.md](docs/physics.md)) |
| **Turbulence** | Hanna (1982) in three regimes, aligned to FLEXPART v11; gradient-Richardson free-troposphere closure; Maryon meander. Behind a swappable `TurbulenceScheme` interface. ([docs/turbulence.md](docs/turbulence.md)) |
| **Convection** | Reduced Emanuel mass-flux scheme, non-divergent matrix, in each meteorology column, once per meteorology interval. ([docs/convection.md](docs/convection.md)) |
| **Terrain-following vertical coordinate** | Meteorology resampled onto a fixed AGL grid per window, with slope-corrected vertical velocity. Fixed the zero-surface-footprint problem over high terrain. |
| **Model-level meteorology** | Reader auto-detects hybrid model levels and reconstructs per-level pressure hydrostatically. **Validated and the default** (task A01, [decision 0017](dev/decisions/0017-model-levels-default.md)): runs end to end on a GH200 with graph capture intact; the hydrostatic pressure matches ECMWF's L137 a/b to under 1 Pa near the surface where the surface geopotential is consistent (not in steep terrain on ARCO stores: see A01's progress file); with the same surface fields, 0–40 m sensitivity changes by +0.5 to +3.5% against pressure levels. Built-in grid: 40 stretched levels. |
| **Meteorology input** | Met format 2.0 ([docs/met_schema.md](docs/met_schema.md); decisions [0019](dev/decisions/0019-met-format-versioning.md), [0020](dev/decisions/0020-time-meaning-of-period-fields.md)): provenance attributes, a fill rule for isolated missing source values, and per-variable labels for means and accumulations, which the reader holds through the window they average (as NAME does). Where no grid point lies on a `met_domain` bound, the reader also reads the first point beyond it. u\* comes from the surface stresses. Stores are prepared by [met-bazarr](https://github.com/mrghg/met-bazarr) `v0.2.0` ([decision 0018](dev/decisions/0018-met-bazarr.md)): ERA5 from the ARCO copy (bit-identical to GLIDE's own downloader apart from u\*) and the Met Office UM analyses NAME runs on. ERA5 straight from ECMWF is a separate, later source. |
| **Multi-site releases** | `multi_point_periodic` — sites share meteorology windows, so per-window costs amortise across the whole network. |
| **Streaming output** | `footprints.zarr` (5-D), `endpoint_particles.parquet`, `trajectory_diagnostics.parquet`, `run_metadata.json`, written per batch. |
| **GPU execution** | Whole per-step body captured as one CUDA graph, device-gated; the eager CPU path is the numerical reference. ([docs/architecture.md](docs/architecture.md)) The venv must be built on uv's own Python, which ships `Python.h`; without it `torch.compile` silently falls back to eager, and `scripts/gpu_env.sh` warns and borrows the `cray-python` headers. |
| **Tests** | 332 passing, about 2.5 minutes, no network. Includes analytic dispersion checks and well-mixed gates. ([docs/VALIDATION.md](docs/VALIDATION.md)) |

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

- Two input conventions were corrected in November 2026, both before the
  freeze ([decision 0020](dev/decisions/0020-time-meaning-of-period-fields.md)):
  - ERA5's heat flux, an hourly accumulation, had been used as a snapshot, so
    30 minutes late. It is now held through the hour it averages. On the Mace
    Head smoke test this keeps the total sensitivity and moves cells by about
    2% in sum.
  - GLIDE took u\* from ERA5's `zust`, which leaves out the drag of sub-grid
    hills. FLEXPART derives u\* from the surface stresses, which include it. So
    in earlier comparisons GLIDE's u\* over land was about half FLEXPART's (1.9
    times smaller, median; 3.6 over mountains). u\* now comes from the
    stresses.

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

- **The residual is now named** (task A1; [VALIDATION.md §8](docs/VALIDATION.md#phase-breakdown-with-sub-phases-task-a1)).
  On `ab_multisite_perf` (GH200, GPU synchronised at each phase, 1,154 s,
  0.125 footprints/s) the two largest consumers are **meteorology fetch,
  833 s (72.7%)**, and **the gridder, 91 s (7.9%)**, ahead of the step (74 s,
  6.5%). Of what used to be the residual, the largest are **`gc.collect` every
  50 steps, 48 s (4.2%; 138 ms a call, 271 ms without synchronisation)** and
  **mask and liveness bookkeeping, 44 s (3.8%)**; the wind-mean diagnostic,
  per-step diagnostics, logging, output writes and Python loop overhead are
  each under 1%, and the untimed rest is under 0.1%.
- **Convection is not a top consumer.** Re-profiled in task A1 on the same
  run: 16.9 s (1.5% of wall), 58.6 ms per met window, of which the vectorised
  parcel lift is 38.7 ms. (Task A02 measured 63–78 ms per window on the A01
  comparison config.)

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
   on ERA5, tracer-experiment data. As of 2026-11-20:
   - **Done:** the NAME reference runs (all of `docs/reference-runs.md` §9);
     release times for standard cases 3–5; EDGAR CH₄ (2019, 2024) on the EUR
     and NAM grids; ETEX and DATEM data; ERA5 model-level cubes for every
     validation window and for A01 (downloaded by GLIDE's own script, for
     development); the GPU environment, with `torch.compile` working.
   - **In progress:** the Met Office UM stores, rebuilt from met-bazarr
     `v0.2.0`; the ERA5 validation stores from met-bazarr `v0.2.0`; the B14
     reader changes (PR #33).
   - **Not started:** the FLEXPART reference runs (waiting for MARS access,
     then flex_extract and task B13); the ERA5 ensemble columns; CH₄ and radon
     observations; CI and branch protection on GitHub; an OSF account.
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

**Fixed: longitudes on 0..360 NAM stores (found 2026-11-24 by task B12, fixed
by PR #56).** The reader used to keep an ascending 0..360 store that does not
cross Greenwich (every NAM store) in 0..360, and the engine clamped a particle
at a negative longitude to the grid's western edge, so a configuration written
in −180..180 advected every particle with the winds at the domain's western
edge, with no error. The reader now relabels such stores −180..180
(`_ensure_monotonic_longitude`), and the preflight refuses a met domain outside
the grid (`_validate_meteorology_space_coverage`). Give longitudes on NAM
stores in −180..180; the tracer benchmark takes the extent from the reader
(`get_lon_lat_coverage`). Anything run on a NAM store before #56 with negative
longitudes is invalid; B12's branch-local OKC80 numbers (with a 0..360
workaround) are superseded by the rerun on the merged code.

The rest are documented approximations rather than bugs; each is stated in the
relevant docs page with its consequence.

- **Per-column vertical interpolation.** Uses the bounding-box mean. Convection
  now works column by column (task A02, decision 0023).
- **Near-surface well-mixed error at the 60 s step: fixed (decision 0027).**
  Holding the turbulence coefficients over the sub-steps and choosing a
  sub-step count per particle put 1.66–1.79 times the well-mixed share of
  particles in the lowest 5% of neutral and stable boundary layers, in every
  footprint made so far, the golden fixtures included. One fixed count for
  every particle, 16 at the 60 s step, with every coefficient re-evaluated each
  sub-step, brings all three regimes within 0.04 of 1 in the lowest 5% and in
  0–40 m ([docs/physics.md](docs/physics.md) §6). The golden fixtures are not
  yet regenerated (planned once, after the A2 residence-counting decision).
- **The boundary-layer top: fixed (decision 0028).** $\sigma_w$ jumps at $h$
  between the boundary-layer profile and the free-troposphere closure. The
  drift's ±1 m finite difference straddled the jump and a crossing particle kept
  its $w'$, so a column started uniform did not stay uniform at any sub-step
  count: in a convective column the 50 m above $h$ held 0.4–11.5 times its
  share depending on the count, and main was worse (found by an independent
  review of PR #69). $h$ is now an interface (Thomson, Physick & Maryon 1997):
  one-sided drift stencil, transmission or reflection with velocity rescaling
  inside every sub-step, and velocities carried as $u'/\sigma$ across steps so
  that a moving $h$ rescales them. The column stays uniform near $h$, and a
  boundary layer loses particles to the free troposphere at the diffusion-limit
  rate, at every count tested ([docs/physics.md](docs/physics.md) §5, §6). On
  the reference multi-site run the 0–40 m footprint is +48% against the
  fixtures (median over releases, 3 seeds, seed spread 0.2%; total −4.2%) and
  now converged in the sub-step count (8 against 16 sub-steps: 0.4%); the
  Mace Head smoke run's 0–40 m footprint nearly doubles (+94%). The golden
  fixtures are not yet regenerated.
- **Forward/backward reciprocity test** (task A6, `tests/test_reciprocity.py`;
  open). The backward *sign construction* is confirmed exactly: in a steady
  column the backward step is the forward step with w′ → −w′, bit for bit.
  Beyond that the test checks the shared scheme's transfer symmetry, and at
  16 sub-steps the plan's criteria fail near the ground (the near-surface
  well-mixed error, ~2%) and near a stable boundary-layer top (the drift
  defect at the σ_w floor, decision 0030 / PR #75; with #75 merged in, three
  of five cases still fail, one narrowly). The slow cases are a
  release check; a moving-h case is still needed
  ([docs/VALIDATION.md](docs/VALIDATION.md) §3).
- **The source of u\*.** From the surface stresses for v1; ERA5's `zust` is
  the alternative, about half as large over land. To compare against the
  tracer experiments and radon after v1 (roadmap §5).
