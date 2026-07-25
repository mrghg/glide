# 0008 — Multi-site shared-met batching + per-hour/per-window met caches

**Context.** The per-window fixed costs — met fetch, the convection matrix, the 3D
support-field build — dominate the wall at scale. Running one site at a time pays
them per footprint. And a run's compute must map onto three axes of parallelism:
sequential time steps, data-parallel particles, task-parallel releases.

**Decision.**
- **Multi-site simultaneous releases** (`multi_point_periodic`): all sites share one
  release-time grid, hence one set of met windows — so each met fetch, convection
  matrix, and field build is amortised over *all* sites' particles at once. This is
  the intended way to grow a run across a network.
- **Batch = the active window** (`ceil(length/period) × n_sites`): the releases whose
  backward windows overlap the cursor at peak. Bigger batches step ever-more inactive
  particles; smaller ones trade device memory for a larger host met cache.
  `make_multisite_config.py` auto-sizes the batch (capped by GPU memory) and the met
  cache from the geometry.
- **Met caching, three layers:** the runtime's window-level LRU
  (`met_cache_max_hours`, host RAM); a **per-hour processed-met cache** in the reader
  so adjacent windows share their common boundary hour (each physical hour is read,
  ω→w-converted, and terrain-regridded **once**); and **per-window field freezing** —
  the σ/ρ/free-trop support stacks are built once at the window midpoint (`t_alpha=0.5`)
  and reused for every step, a met-cadence approximation (<1%/hr drift) matching how
  FLEXPART-class models refresh turbulence fields.

**Rationale.** Shared met is the amortisation lever; the per-hour cache removed the
~2× redundant boundary-hour work and cut `met_fetch` ~8× on a representative GH200
run (−32% wall there); per-window field freezing cut ~18% of an earlier CPU profile.

**Rejected alternatives.**
- Single-sweep footprint streaming over all releases — bounds footprint memory but
  not particle memory, and on the static GPU path steps ~6× the particles; batching
  is the memory lever precisely because the graph strategy fixes the shape.
- Per-window (not per-hour) cache — duplicated every shared boundary hour.

**Status.** In force. **Gotcha for A/B testing:** an under-sized `met_cache_max_hours`
triggers ~6× cross-batch LRU re-fetch thrash (the run warns); and SLURM + an editable
venv means A/B'ing branches needs one git worktree + venv each, or both jobs load the
same code. See [STATUS.md](../../STATUS.md) and `configs/ab_multisite_perf.yaml`.
