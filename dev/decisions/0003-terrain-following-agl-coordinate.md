# 0003 — Internal geometric-metres AGL coordinate, terrain-following

**Context.** Particle physics (σ_w, T_L, the BL scaling in `z/h`, the footprint's
surface layer) is intrinsically in **height above ground**. But GLIDE streams ERA5
**pressure levels** ([0001](0001-streaming-arco-zarr-io.md)), which are
quasi-horizontal and slice through mountains — below-ground levels exist by
construction. An early version mapped particles through a **bbox-mean AGL profile**,
which over an ocean-dominated domain is effectively height-above-*sea-level*: it
left the surface footprint **zero over all high terrain** and released elevated
tower sites up to ~2 km underground.

**Decision.** Particles carry **geometric metres AGL** internally (convert once, so
the hot loop never touches pressure coordinates). The reader **resamples the
pressure-level met onto a fixed terrain-following AGL grid once per met window**
(FLEXPART `verttransform`-style): per-column interpolation excluding sub-surface
levels, plus a slope correction of the vertical velocity
`w_agl = w − taper·(u·∂h/∂x + v·∂h/∂y)`.

**Rationale.** On a terrain-following grid the single 1-D level array is *exact*
per column (not a bbox approximation), the vertical mapping is a run constant, and
sub-surface levels are excluded cleanly once. Verified: a particle holds its AGL to
~3 m crossing an 800 m hill (vs ~800 m without the correction), and the
surface-footprint terrain holes collapse from 86.6% to 0.5% of high-terrain cells.

**Rejected alternatives.**
- Per-particle terrain offset + slope term in the hot path — adds lookups per step
  and a non-orthogonal Jacobian to the Langevin term.
- Track particles in ASL internally — exact advection, but changes the meaning of
  `particles[:, 2]` across the release generator, reflection, the gridder, and the
  `endpoint_particles.parquet` contract, while every physics term still needs AGL.

**Status.** In force (`terrain_following=True` default; legacy pressure-grid path
retained for A/B). Kernels in `src/lpdm/vertical_grid.py`. **Changes footprint
magnitudes everywhere, so all validation must be re-run** (see
[STATUS.md](../../STATUS.md)). See [docs/architecture.md](../../docs/architecture.md).
