# 0024 — Boundary-layer depth limits as configuration; meander timescale from the met interval

**Context.** The Hanna scheme floored the boundary-layer depth $h$ from the
meteorology at a hard-coded 50 m (`BLH_MIN_M`) and had no ceiling. The reference
runs of FLEXPART and NAME hold $h$ between 100 and 4,500 m (FLEXPART's
`hmixmin`/`hmixmax` defaults; NAME's `Min B L Depth`/`Max B L Depth`), and the
validation plan (Tier 2, "Settings that all three models share") requires GLIDE to
do the same in those runs, so the limits are a shared setting rather than an
expected difference. Separately, the meander timescale defaulted to a fixed
1,800 s, half the hourly ERA5 interval, and stayed there on coarser meteorology
such as the UM's 3-hourly fields, which the validation plan also uses.
Decisions of 16 November 2026 (task A03).

**Decision.**
- Two options, `turbulence.boundary_layer_depth_min_m` (default 50) and
  `turbulence.boundary_layer_depth_max_m` (default none), replace the constant.
  The scheme replaces $h$ by $\min(\max(h, h_{\min}), h_{\max})$ as soon as it is
  interpolated to the particle, on both step paths, so every use sees the limited
  value: regime selection ($h/L$), $w_\ast$, the profiles and timescales, and the
  free-troposphere test $z > h$. `convective_velocity` takes the minimum as an
  argument instead of the constant. The limits are Python constants fixed when
  the scheme is built, so the compiled step specialises on them once and no
  per-step Python scalar is introduced (decision 0004).
- An unset `turbulence.meander.timescale_seconds` is half the interval the
  reader measures from the store's time coordinate. The runtime binds it once,
  after the reader is built and before the first step
  (`TurbulenceScheme.bind_met_interval`, a no-op for other schemes), and logs
  it. A configured value is never changed.
- The validation configurations, when they are written (none exists yet), set
  100 and 4,500 m. The shipped example configurations keep the defaults.

**Measured.** Shipped configurations are unchanged: shortened CPU runs of the
five ERA5 configurations (3 h, 500 particles per release, two releases) give
bit-identical footprints and end records before and after (SHA-256 of the
arrays). On hourly ERA5 the timescale is 1,800 s as before. Tests:
`tests/test_hanna.py` (limits on both paths, `convective_velocity`, the
timescale), `tests/test_met_reader.py` (hourly and 3-hourly stores),
`tests/test_main_runtime.py` (binding in a run), `tests/test_main_config.py`.

**Rationale.** Making the limits configuration, with today's values as the
defaults, lets the validation runs share the reference models' settings without
changing any other run. Applying the limits once, at the point where $h$ enters
the scheme, is the only way to be sure every use sees the same value. Tying the
timescale to the store keeps it right on UM fields without asking the user to
remember it; on hourly ERA5 nothing changes.

**Rejected alternatives.**
- *Change the defaults to 100 and 4,500 m:* moves every existing result, outside
  this task's remit; the defaults are a separate decision (roadmap 7b settles v1
  defaults).
- *Clamp $h$ inside each profile function:* easy to miss one use, and the two
  step paths could drift apart.
- *Resolve the timescale from each met window as it arrives:* a per-window value
  would be a changing Python scalar to the compiled step, and the interval is a
  property of the store, known before the first step.
- *Add a drift correction to the meander term for its spatially varying
  $\sigma_m$:* out of scope; the A03 Tier 0 test measures how much it matters
  (docs/turbulence.md §7).

**Status.** Adopted (task A03). Docs: [turbulence.md](../../docs/turbulence.md)
§2, §7, §9; [met_schema.md](../../docs/met_schema.md).
