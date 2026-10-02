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
