# 0007 — Reduced Emanuel deep convection, once per met window

**Context.** The biggest dispersion gap vs FLEXPART was deep cumulus convection —
it lofts surface air through the whole troposphere in minutes-to-hours, which
resolved advection + BL turbulence alone cannot represent.

**Decision.** Implement a **reduced port of the FLEXPART Emanuel & Živković-Rothman
scheme** (`emanuel_reduced`; Stohl 2005 §4.6, Forster et al. 2007), behind the
`ConvectionScheme` interface. It builds a per-column mass-flux **matrix** and
redistributes particles by sampling it. Two deliberate simplifications:
- Runs **once per met window**, not per step — the mass-flux matrix depends only on
  the (T, q) profile, which is constant within the hourly window.
- Uses the **bbox-mean column** for the parcel lift (mirrors the vertical-interp
  approximation), not per-(lon, lat) profiles.

**Rationale.** The Emanuel scheme is FLEXPART's, so it keeps GLIDE comparable. The
per-window / bbox-mean simplifications make it cheap and vectorisable while
capturing the first-order effect. The mass-flux matrix is **non-divergent**
(updraft + compensating subsidence) with FLEXPART's backward-transpose sampling and
mass-weighted BL entrainment — gated by deterministic well-mixed (`mᵀP = mᵀ`) tests
so it conserves mass exactly.

**Rejected / deferred.**
- Full Emanuel quasi-equilibrium closure — our port caps buoyancy velocity at 5 m/s
  as a closure simplification; revisit if validation shows under-convective transport.
- Per-column profiles — same 3D `grid_sample` refactor as the per-column vertical-
  interpolation follow-up.
- The convection interval is hardcoded at 3600 s (documented follow-up).

**Status.** In force. See [docs/convection.md](../../docs/convection.md). A negative-
control test asserts it does **not** fire on a stable winter column.
