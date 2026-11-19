# 0020 — Means and accumulations are held through the interval they average

**Context.** GLIDE interpolates every meteorology field linearly in time
between the two ends of each window, treating each stored value as a snapshot
at its timestamp. Most fields are snapshots. Some are not:

- ERA5's surface sensible heat flux and surface stresses are accumulated over
  the hour ending at their timestamp. GLIDE de-accumulates the heat flux and
  then treats it as a snapshot, so it is used about 30 minutes late.
- In the Met Office UM store (task B14, met-bazarr decision 0003), the heat
  flux and the two surface stresses are means over the 3 hours ending at the
  timestamp, and the friction velocity is derived from the stresses. Treated as
  snapshots, both are used about 1.5 hours late. Around sunrise and sunset, when
  the heat flux changes sign within a window, that shifts the turbulence regime
  by up to one window.

GLIDE uses the heat flux and u\* in the Obukhov length and the Hanna turbulence
scheme (`src/lpdm/turbulence/hanna.py`), interpolated in time at each step
exactly as the winds are.

**What the reference models do** (18 Nov 2026). Neither has the lag.

- *NAME*, on the UM's 3-hour means, holds the heat flux and stresses constant
  through each interval at the values stamped at the interval's end, which are
  the means over that interval. Its met definition for this series is set up
  for time-averaged fluxes. NAME derives u\* from the stresses itself, so u\*
  is held the same way. Sources: the NAME 8.7.1 documentation, and NAME
  8.7.1's behaviour tested in a met-only run with the reference-run settings
  and meteorology (two points, 12 hours).
- *FLEXPART*, on ERA5, gets its fluxes from flex_extract, which de-accumulates
  the hourly heat flux and stresses into interval means and then converts them
  to values at the met times with an integral-preserving cubic,
  (−m₁ + 7m₂ + 7m₃ − m₄)/12 for the time between intervals 2 and 3
  (`disaggregation.dapoly`, applied in `EcFlexpart.deacc_fluxes`). FLEXPART then
  treats them as snapshots: it derives u\* from the stresses
  (`getfields_mod.f90`, `scalev`) and interpolates u\*, w\* and the inverse
  Obukhov length linearly in time (`interpol_mod.f90`).

**Decision.**

- **Each field says what its timestamp means.** A variable that is a mean or an
  accumulation over a period ending at its timestamp carries the CF attribute
  `cell_methods = "time: mean"` or `"time: sum"` and
  `glide_period_seconds = <period>`. A variable with neither is a snapshot. This
  is `glide` format 2.0 (`docs/met_schema.md`, "What a timestamp means").
- **GLIDE holds a period variable constant through each window** at its value at
  the window's end, which is the mean over exactly that window, when the period
  equals the store's time step. This is NAME's treatment. Snapshots are
  interpolated as before. GLIDE refuses a period that differs from the time
  step.
- **u\* comes from the surface stresses in every source.** met-bazarr derives
  it from the mean stresses and labels it as a period variable: the UM's
  (10,800 s, already done) and, for ERA5, the hourly accumulated stresses
  `ewss` and `nsss` (3,600 s), in place of ERA5's own `zust`. The reasons are
  below, under "u\* from the stresses, not ERA5's `zust`".
- **Stores of format 1.0** still read. Their heat flux is a period variable when
  the store has `glide_flux_accumulation_seconds` or a non-zero
  `glide_flux_averaging_seconds`; everything else, including `zust`, is a
  snapshot. Existing development stores need no rewriting.
- **Timing.** This changes GLIDE's results everywhere, slightly for ERA5. It
  lands before the golden fixtures (task A0), with the reader's edge-strip fix
  (decision 0019), as part of the GLIDE side of B14.

**Consequences for the validation.** On the UM, GLIDE and NAME then treat the
surface forcing identically. On ERA5, GLIDE holds the hourly mean through the
hour where FLEXPART interpolates between reconstructed point values. Both are
centred and preserve the hourly total; they differ only within each hour. This
goes on the pre-registered list of expected differences (validation-plan §3),
expected to be small.

**Rationale.** Each field is placed at the time it represents: a snapshot at its
timestamp, a mean over the interval it averages. Treating a mean as a snapshot
at the end of its period is what creates the lag. Holding the value is exact for
the window mean, needs no extra data, matches NAME exactly, and stays within the
hour of FLEXPART's treatment.

**u\* from the stresses, not ERA5's `zust`** (agreed with Matt, 19 Nov 2026).

- *What the two are.* ERA5's surface stresses are the momentum the model
  removes from the atmosphere at the surface: the drag of the local surface
  plus the drag of hills too small for its grid (turbulent orographic form
  drag). u\* derived from them is an effective, grid-box value. `zust` is the
  u\* of the surface-layer scheme with the local roughness only, without the
  hill drag. Both are ERA5's own fields; neither is more direct.
- *How much they differ* (met-bazarr decision 0006, NAM, 6 hours): the same over
  the sea (median ratio 1.00); over land above 100 m, u\* from the stresses is
  1.9 times `zust` (median), and 3.6 times over mountains above 1,500 m.
- *Why the stresses.*
  - **The same definition on both meteorologies.** The UM meteorology NAME
    runs on supplies only the stresses, so u\* there can only come from them.
    With `zust` for ERA5, GLIDE on the two meteorologies (Tier 2) would mix two
    definitions of u\* into the comparison meant to isolate the meteorology.
  - **The quantity GLIDE needs.** GLIDE uses u\* for turbulence averaged over a
    grid cell. Over hilly terrain, flow over hills smaller than the grid makes
    the boundary layer more turbulent on average, which the hill drag
    represents and `zust` leaves out. Over flat terrain the two agree.
  - Both reference models use the stresses (FLEXPART through flex_extract,
    NAME from the UM), so this adds no difference from them. This is a
    consequence, not the reason: GLIDE aims to be independent of them.
- *Why FLEXPART uses them*, as far as is known: availability. FLEXPART's u\*
  routine (1994) predates ERA5, the accumulated stresses have long been
  standard ECMWF fields, and `zust` was not offered in ERA-Interim or earlier.
- *Not settled for good.* Neither choice is clearly right at a 30 km grid
  spacing. GLIDE's earlier over-estimates of near-surface CH₄ at low inlets
  (STATUS, traced to weak near-surface mixing) were made with `zust`, which may
  have contributed. Revisit after v1, by comparing the two against the tracer
  experiments and radon (roadmap §5).

**Rejected alternatives.**
- *u\* from ERA5's `zust`.* See above: it would give GLIDE different
  definitions of u\* on ERA5 and on the UM, and leaves out the hill drag that
  the grid-cell average turbulence includes.
- *Choose between the two in the pre-registered fit on ETEX-1 and CAPTEX.*
  Decides it on data, but adds a dimension to the fit and needs a store format
  that can carry both. Deferred to after v1.
- *Keep treating means as snapshots.* A lag of 1.5 hours in the surface forcing
  on the UM and 30 minutes on ERA5, which neither reference model has.
- *Convert means to point values in met-bazarr, as flex_extract does.* GLIDE
  would need no change and would match FLEXPART exactly on ERA5, but on the UM
  it would differ from NAME by the difference between a smooth curve and a
  3-hour step. NAME is the reference on the UM, and 3 hours is the larger of the
  two intervals, so matching NAME matters more. Each store would also need an
  extra time step at each end.
- *Use ERA5's instantaneous fields* (`ishf`, `iews`, `inss` in ARCO). True
  snapshots, with no lag and no conversion, but each is a single model time step
  rather than an hourly mean, and neither reference model uses them.
- *Shift the timestamps in the store.* All fields share one time coordinate
  (format 1.0, "One grid for every field"), so a period variable cannot carry
  different times from the snapshots.

**Implementation.** GLIDE's reader change is part of the GLIDE side of task
B14 (`dev/agent/QUEUE.md`), made before A0. met-bazarr adds the variable
attributes and the ERA5 u\* from stresses.

**Status.** Adopted 2026-11-18. GLIDE's reader does not implement it yet (task B14). Docs: [docs/met_schema.md](../../docs/met_schema.md), "What a timestamp means".
