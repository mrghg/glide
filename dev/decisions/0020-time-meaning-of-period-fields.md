# 0020 — Means and accumulations are placed at the middle of their period

**Context.** GLIDE interpolates every meteorology field linearly in time
between the two ends of each window, treating each stored value as a snapshot
at its timestamp. Most fields are snapshots. Some are not:

- ERA5's surface sensible heat flux is accumulated over the hour ending at its
  timestamp. GLIDE de-accumulates it and then treats it as a snapshot, so it is
  used about 30 minutes late.
- In the Met Office UM store (task B14, met-bazarr decision 0003), the heat
  flux and the two surface stresses are means over the 3 hours ending at the
  timestamp. The friction velocity is derived from the stresses. Treated as
  snapshots, both are used about 1.5 hours late. Around sunrise and sunset, when
  the heat flux changes sign within a window, that shifts the turbulence regime
  by up to one window.

GLIDE uses the heat flux and u\* in the Obukhov length and the Hanna turbulence
scheme (`src/lpdm/turbulence/hanna.py`), interpolated in time at each step
exactly as the winds are.

ERA5's friction velocity (`zust`) is believed to be a snapshot; this is to be
confirmed from ECMWF's ERA5 documentation before the change is made.

**Decision (proposed).**

- **Each field says what its timestamp means.** A variable that is a mean or an
  accumulation over a period ending at its timestamp carries the CF attribute
  `cell_methods = "time: mean"` or `"time: sum"` and
  `glide_period_seconds = <period>`. A variable with neither is a snapshot. This
  is `glide` format 2.0 (`docs/met_schema.md`, "What a timestamp means").
- **GLIDE places a period variable at the middle of its period.** When the
  period equals the store's time step, the variable is held constant through
  each window at its value at the window's end, which is the mean over exactly
  that window. Snapshots are interpolated as before. GLIDE refuses a period that
  differs from the time step.
- **Stores of format 1.0** still read. Their heat flux is a period variable when
  the store has `glide_flux_accumulation_seconds` or a non-zero
  `glide_flux_averaging_seconds`; everything else is a snapshot. So existing
  ERA5 stores need no rewriting.
- **met-bazarr** writes the attributes: on the UM heat flux and friction velocity
  (10,800 s, `time: mean`) and on the ERA5 heat flux (3,600 s, `time: sum`).
- **Timing.** This changes GLIDE's results everywhere, slightly for ERA5. It
  lands before the golden fixtures (task A0), with the reader's edge-strip fix
  (decision 0019), as part of the GLIDE side of B14.

**Rationale.** It is not a special case for one variable. Each field is placed at
the time it represents: a snapshot at its timestamp, a mean at the middle of its
period. Treating a mean as a snapshot at the end of its period is what creates
the lag. Holding the value constant through the window is exact for the window
mean and needs no extra data.

**Rejected alternatives.**
- *Keep treating means as snapshots.* A lag of 1.5 hours in the surface forcing
  on the UM meteorology, and 30 minutes on ERA5, is a known bias that would
  carry into every footprint.
- *Interpolate linearly between period midpoints.* Smoother, with no step at
  window boundaries, but each window then needs the surface fields of a third
  time, which changes the reader's window loading and the per-hour cache.
  Reconsider if the steps at window boundaries prove to matter.
- *Shift the timestamps in the store.* All fields share one time coordinate
  (format 1.0, "One grid for every field"), so a period variable cannot carry
  different times from the snapshots.

**Before adoption.**
- Confirm that ERA5's `zust` is a snapshot.
- Find out how NAME uses the UM's 3-hour mean heat flux and stresses in time,
  and how FLEXPART uses ERA5's de-accumulated fluxes (flex_extract; task B13).
  If a reference model treats means as snapshots, GLIDE's correct treatment
  gives a systematic difference from it. That difference goes on the
  pre-registered list of expected differences (validation-plan §3), with its
  predicted direction, rather than GLIDE copying the lag.

**Status.** Proposed 2026-11-18, not yet adopted. Docs: [docs/met_schema.md](../../docs/met_schema.md), "What a timestamp means".
