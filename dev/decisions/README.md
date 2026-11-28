# Design & physics decisions

Short records of the **major, still-relevant** decisions behind GLIDE — the *why*,
with the alternatives that were rejected. The *how* lives in [docs/](../../docs/);
current status in [STATUS.md](../../STATUS.md); full narrative in git history.

Add a record when a decision is architectural, physics-defining, or reverses an
earlier one. Keep it short. Format: **Context → Decision → Rationale → Rejected
alternatives → Status**, with a pointer to the relevant `docs/` page.

| # | Decision |
|---|---|
| [0001](0001-streaming-arco-zarr-io.md) | Stream ARCO ERA5 from Zarr, not whole NetCDF/GRIB files |
| [0002](0002-pure-pytorch-device-agnostic.md) | Pure PyTorch, device-agnostic; the eager path is the numerical reference |
| [0003](0003-terrain-following-agl-coordinate.md) | Internal geometric-metres AGL coordinate, terrain-following |
| [0004](0004-cuda-graph-static-step-path.md) | Whole per-step body captured as one static-shape CUDA graph |
| [0005](0005-swappable-scheme-interfaces.md) | Turbulence & convection as swappable schemes; physics stays interrogable |
| [0006](0006-hanna-turbulence-flexpart-aligned.md) | Hanna (1982) turbulence, aligned to FLEXPART v11 |
| [0007](0007-reduced-emanuel-convection.md) | Reduced Emanuel deep convection, once per met window |
| [0008](0008-multi-site-shared-met-batching.md) | Multi-site shared-met batching + per-hour/per-window met caches |
| [0009](0009-ruff-gitleaks-pre-commit.md) | Ruff + gitleaks pre-commit hooks; standardise on PEP 8 spaces |
| [0010](0010-model-level-met-reader.md) | Model-level met: reconstruct pressure hydrostatically, not from a/b coefficients |
| [0011](0011-multi-rate-time-stepping.md) | Multi-rate time stepping: a per-particle outer step (proposed) |
| [0012](0012-kernel-deposition.md) | Optional kernel deposition in the footprint gridder (proposed) |
| [0013](0013-composite-met-source.md) | A composite met source, not a nesting feature (proposed, v1.x) |
| [0014](0014-v1-scope-and-release-gate.md) | v1 scope: validated, fast footprint model without UQ |
| [0015](0015-agent-driven-development-process.md) | Agent-driven development: one task per session, PR per task, human merges |
| [0016](0016-time-ago-bin-width.md) | Time-ago bins of a configured width, covering the run unless dropping old ages is asked for |
| [0017](0017-model-levels-default.md) | ERA5 model levels as the default meteorology |
| [0018](0018-met-bazarr.md) | Meteorology preparation lives in met-bazarr (proposed; GATES may join) |
| [0019](0019-met-format-versioning.md) | A versioned `glide` met format, with provenance and rules for validation inputs (proposed) |
| [0020](0020-time-meaning-of-period-fields.md) | Means and accumulations held through the interval they average, as NAME does; u* from stresses |
| [0021](0021-seed-reproducible-runs.md) | A run's seed seeds torch's default generators, so the whole run is reproducible |
| [0022](0022-density-weighting-and-dry-air.md) | Each particle divided by the dry-air density where it is counted; NAME and FLEXPART corrected to dry air |
| [0023](0023-convection-per-column.md) | Deep convection in each meteorology column, batched over columns |
| [0024](0024-boundary-layer-depth-limits-and-meander-timescale.md) | Boundary-layer depth limits as configuration (defaults 50 m, none); meander timescale half the store's met interval |
| [0025](0025-hanna-column-pressure.md) | Hanna density and potential temperature from each column's own pressure, not the area mean per level |
| [0026](0026-step-mean-deposition-position.md) | Optional deposition at the mean of each step's start and end positions, replacing the specified RK2-predictor midpoint (superseded by 0029: removed) |
| [0027](0027-fixed-substep-count.md) | Hanna turbulence: one fixed sub-step count for every particle (default 16), coefficients re-evaluated every sub-step |
| [0028](0028-boundary-layer-top-interface.md) | Hanna turbulence: the boundary-layer top as a Thomson, Physick & Maryon (1997) interface; one-sided drift stencil; velocities carried as u'/σ across steps |
| [0029](0029-trapezoid-residence-counting.md) | Trapezoid residence counting (half of each step at each end), scored against fine-time residence; proposed as the default; removes step_mean |
| [0030](0030-drift-gradient-at-the-sigma-floor.md) | Hanna turbulence: analytic drift gradient where the stencil is cut at a stable h or meets the σ_w floor (amends 0028 item 1); floor kept at 10⁻³ (28 Nov 2026) |
| [0031](0031-cape-bounded-at-the-cloud-top.md) | Convection: the trigger's CAPE is the diagnosed cloud's, LCL to cloud top, not to the top of the column; the 5 m/s updraft cap, binding in every firing column, left open |
| [0032](0032-wind-aligned-horizontal-turbulence.md) | Hanna turbulence: horizontal turbulence along and across the mean wind (`turbulence.horizontal_frame: wind`), carried east/north between steps; proposed as the default, code default still `grid` (proposed) |
