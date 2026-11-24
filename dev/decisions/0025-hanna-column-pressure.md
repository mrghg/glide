# 0025 — Hanna density and potential temperature from each column's own pressure

**Context.** The Hanna scheme builds two grid fields from pressure once per met
window: the air density $\rho = p/(R_d T)$ and its vertical gradient, for the
Stohl-Thomson density drift $\sigma_w^2\,\partial_z \ln\rho$ of the well-mixed
vertical Langevin equation, and the potential temperature
$\theta = T (p_0/p)^\kappa$ of the free-troposphere Richardson closure. Since the
scheme was written both took $p$ from `metadata.pressure_level_hpa`, one value
per grid level. On a pressure-level grid without terrain following that is
each column's pressure on the level, which is correct. On the terrain-following
AGL grid (the default, and the only grid of model-level stores) the reader
computes it as the met domain's **area mean** of the per-column pressure on each
AGL level, so every column got the domain-average pressure at its height above
ground. Measured in PR #61 on the EUROPE model-level store: over terrain above
1,000 m, $\rho$ too high by a median 24–31% (up to 56%), and
$\partial_z\ln\rho$, so the drift term, off by a median 3–12% (up to 24% at
40 m). FLEXPART uses the density of the particle's own column. Matt's decision
of 24 Nov 2026: fix it before the physics freeze and regenerate the golden
fixtures in the same change.

**Decision.**
- Both fields take each column's own pressure at each level: the window's
  per-column pressure `HourlyMetTensors.pressure_hpa` (task A02), averaged over
  the window's two hours to the window midpoint, the same time at which $T$ is
  taken (`HannaScheme._column_pressure_pa_mid`). On model levels this is the
  hydrostatic reconstruction (decision 0010) regridded to the AGL levels; on
  pressure levels it is the level pressure regridded to the AGL levels, so each
  column gets its own pressure at each height above its own ground.
- Without per-column pressure (pressure-level grid without terrain following,
  synthetic readers) `HourlyMetTensors.column_pressure_hpa()` broadcasts the
  level pressures, which there are each column's pressure; the result is bit
  for bit what it was.
- The field is built inside the per-window cache builder, so it is computed and
  moved to the device once per window (it used to be rebuilt from NumPy on
  every step, outside the cache) and lands in the same static buffer of fixed
  shape `[2, Z, Y, X]`; no per-step Python scalar, no new shape (decision 0004).
- `metadata.pressure_level_hpa` (the area mean) is now metadata only: nothing
  in the physics reads it on the AGL grid.

**Measured.** Low-lying sites barely move: the smoke run (Mace Head, 1 day)
changes its release total by +2.2e-5 (CPU) and +6.2e-5 (CUDA) relative and its
0–40 m total by +0.03% (CPU). Over the Alps the change is large and real: one
release at the Jungfraujoch site (46.5475° N, 7.9851° E, 1,000 m above ground,
1 Jan 2024 00 UTC, 5 days back, 21,600 particles, CUDA) loses 17.9% of its
0–40 m sensitivity (weighted mean |log10 ratio| of the 0–40 m map 0.108),
against seed-to-seed differences of 0.5% and 0.060 (seed 42 vs 43, both with
this change). Most of it is the free-troposphere closure, not the density
drift: the drift's $\partial_z\ln\rho$ at the Jungfraujoch column changes by
4–6%, but $N^2 = (g/\theta)\,\partial_z\theta$ is a small difference of two
large terms ($\partial_z \ln T$ and $\kappa\,\partial_z \ln p$), and with the
area-mean $p$ it was 5–55% too low in that column between 0.5 and 10 km
(median 4%, 95th percentile 19% over the Alps, 43–48.5° N, 5–16.5° E,
500 m–15 km; window 31 Dec 2023 12 UTC), so the free
troposphere there was mixed too strongly. The fixtures were regenerated with
this change (docs/VALIDATION.md §8). Unit tests:
`tests/test_hanna_column_pressure.py` (closed-form hydrostatic columns, flat
and elevated; independence from the other columns; window midpoint; bit
identity for horizontally uniform met; θ's pressure).

**Rationale.** The per-column pressure is already in every window (deep
convection lifts parcels on it), on the grid where $\rho$ and $\theta$ are
built, so the fix is a change of input with no new machinery, no new per-step
cost and no change to the cached, graph-captured step. The drift needs
$\partial_z\ln\rho$ on the same grid as $\sigma_w$'s other inputs; taking it
from a gridded $\rho$ keeps the existing finite-difference machinery.

**Rejected alternatives.**
- *Per-particle pressure from the surface pressure (`lpdm.air_density`, as the
  footprint's dry-air density does, decision 0022):* exact at the particle, but
  the drift needs $\partial_z\ln\rho$, which would have to be differentiated per
  particle and per sub-step, and $\theta$'s gradient is a grid quantity; two
  pressure sources for one scheme with no gain at grid resolution.
- *Keep the area mean and document it:* the error is small in the drift term
  but large in $\rho$, grows with terrain height, and the correct input was
  already to hand.
- *Use the hour-start pressure (what the area mean used):* $T$ is taken at the
  window midpoint; mixing times in $p/T$ is inconsistent.

**Status.** Adopted (24 Nov 2026). Docs: [physics.md](../../docs/physics.md) §3,
[turbulence.md](../../docs/turbulence.md) §6 and the density note in §5.
