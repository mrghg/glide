# 0022 — Each particle is divided by the dry-air density where it is counted

**Context.** The footprint of a release (`docs/reference-runs.md` §6) puts the
air density inside the integral, per particle and per moment:

$$
S = \frac{M_\mathrm{air}}{h\ N} \sum_{p=1}^{N} \int \frac{I_p(t)}{\rho_p(t)}\ \mathrm{d}t
$$

GLIDE's gridder counts plain residence seconds. The comparison helper
(`src/lpdm/comparison.py`, `to_stilt_surface_footprint`) then divides the
summed seconds by one density per cell. The two agree only if a particle's
time in the 40 m layer and the density at that time are unrelated. Near the
surface they are related: particles spend most time in the layer at night, in
stable, cold, dense air. So dividing afterwards is biased, in a consistent
direction. The validation review of 21 Nov 2026 (finding 3) showed the
mechanism with densities of 1.0 and 1.2 kg m⁻³ in one cell: −7.6% and +7.0%
for residence split 90/10 and 10/90 s. Real near-surface swings are smaller,
so the real error is probably a few percent, varying with stability and season.
It has not been measured.

§6 also says only "the model's air density", without saying whether the air
is moist or dry. That choice moves the footprint by up to about 2.5% in humid
air.

**What the reference models do** (23 Nov 2026).

- *NAME 8.7.1*, as requested for the reference runs (output quantity Mixing
  Ratio, time-integrated, in ppm s, sampled every 300 s, one 0–40 m level).
  This comes from the NAME agent's reading of the NAME documentation and code,
  and the reference-run inputs and outputs; the code reading is not yet tested.
  - NAME divides each cell's air concentration by a density at every sample,
    then integrates. So it captures the night-time pairing.
  - The density is one value per output cell and sample, not one per particle,
    taken at the cell's centre, 20 m above the ground there.
  - It is p / (R_d T), from the UM's pressure and temperature, interpolated
    linearly between 3-hourly fields. It has no humidity term. In moles,
    NAME's mixing ratio is therefore relative to moist air.
  - The ACRG's older processing requested air concentration and divided by one
    density afterwards. It was dropped for the reason the review gives.
- *FLEXPART v11* (local checkout, v11.1 plus 16 commits), with the reference
  settings `IND_SOURCE = 1`, `IND_RECEPTOR = 2`, `LOUTSAMPLE = 300`.
  - A backward run with `IND_SOURCE = 1` sets `ind_samp = −1`
    (`readoptions_mod.f90`, `readcommand`). At every sample each particle's
    mass is divided by the density interpolated to that particle
    (`output_mod.f90`, `conccalc`; `interpol_mod.f90`, `interpol_density`).
    So FLEXPART weights per particle, as §6 does.
  - The density is taken from one of the two met fields in memory, without
    time interpolation (the code's comment calls this accurate enough).
  - The density is moist-air density, p / (R_d T_v), with virtual temperature
    T_v = T(1 + 0.608 q). At the surface it uses 2 m temperature and dew point
    (`verttransform_mod.f90`, `verttransform_ecmwf_heights`). FLEXPART also
    computes a dry-air density, but only for its optional dry-air output, not
    for the footprint.

How each model's footprint compares with one taken relative to dry air
(q = specific humidity, e = vapour pressure, e/p ≈ 1.61 q):

| | Footprint relative to the dry-air one | Humid summer surface air, q = 15 g/kg | Winter, q = 3 g/kg |
| --- | --- | --- | --- |
| NAME | 1 − e/p | −2.4% | −0.5% |
| FLEXPART | (1 + 0.608 q)(1 − e/p) ≈ 1 − q | −1.5% | −0.3% |

So NAME and FLEXPART also differ from each other by about 0.6 q, about 1% in
humid summer air. In mole terms, neither is quite the moist-air mole fraction:
NAME is exactly it, and FLEXPART sits between moist and dry, because it
divides moist-air mass by the dry-air molar mass.

**Decision.**

- **GLIDE divides each particle's count by the dry-air density at the particle**
  as it is counted: ρ_d = (p − e) / (R_d T), at the particle's position and the
  step's time, using the same met window and time weight as the step
  (`_step_window_time`, PR #41). The raw footprint's units become s m³ kg⁻¹
  of dry air. M_air / h then gives m² s mol⁻¹ of dry air directly, with no
  density applied afterwards.
- **§6 means the dry-air mole fraction.** ρ_p in §6 is the dry-air density.
  That is how greenhouse-gas mole fractions are reported, and it is exact:
  M_air / ρ_d is the inverse of the dry-air molar density.
- **NAME's and FLEXPART's footprints are corrected to dry air afterwards**, cell
  by cell, with the factors in the table, from each model's own meteorology
  averaged over the footprint's period: × p/(p − e) for NAME, and
  × p / ((p − e)(1 + 0.608 q)) for FLEXPART. This is a correction after the
  fact, but it is safe where dividing by density afterwards is not: the
  correction is itself at most about 2.5%, so leaving out how it varies in time
  costs a small fraction of that. The native outputs are kept, and each
  conversion is recorded.
- **Residence seconds stay available.** An output setting chooses the
  weighting: dry-air density (the default, for the validation) or none (plain
  seconds, as now). The choice and the units go into the footprint's
  attributes. The comparison helper refuses to apply a density to a footprint
  that is already weighted.

**Expected differences that remain** (validation plan §3).

- *NAME takes the density at the cell centre, GLIDE and FLEXPART at the
  particle.* Over 40 m, the centre at 20 m balances the vertical change to
  first order. Horizontally, it matters where ground height varies a lot
  within a 0.25° cell: a few percent at Jungfraujoch.
- *How density follows time.* NAME interpolates 3-hourly UM fields; FLEXPART
  takes one hourly ERA5 field without interpolating; GLIDE interpolates hourly
  ERA5 within the step's window. These belong to each model's meteorology and
  are not corrected.

**Rationale.** Dividing while counting is what §6 defines, and both reference
models already do it, NAME per cell and FLEXPART per particle. GLIDE's
after-the-fact helper is the outlier. Dividing afterwards could be made exact
only by keeping hourly time bins for all 240 hours, about 240 times the
footprint's memory over 4,704 releases. Weighting while counting costs one
density per counted particle per step, from fields GLIDE already reads.
Defining §6 as dry air gives the quantity measurements are reported in, and
keeps the humidity differences between the models out of the transport
comparison.

**Rejected alternatives.**
- *Keep dividing afterwards, with a mean density per cell.* Biased in a
  consistent direction by the night-time pairing (above). The review's
  recommendation, if kept, would be to measure the error and declare it; there
  is no reason to keep it once the fix is cheap.
- *Divide afterwards using hourly time bins.* Exact to an hour, but needs
  hourly bins over 240 hours for every release.
- *Copy NAME's per-cell density.* Would match NAME, but GLIDE would have to
  find each cell's centre ground height and evaluate density there every step,
  to reproduce a simplification. It would then differ from FLEXPART instead.
- *Moist-air density, as NAME has it.* Matches NAME exactly, but not FLEXPART,
  and not the dry-air mole fractions the measurements use. The humidity term
  would then sit inside the Tier 3 comparison with observations.
- *Leave the humidity differences uncorrected and declare them.* Simple, but
  up to 2.5% in July at Park Falls and Lamont, larger than several effects the
  Tier 2 comparison is meant to resolve, and correctable at almost no cost.

**Open before registration.**
- *The tracer observations* (settled 23 Nov 2026, from the sources below; no
  primary source states a dry or moist basis, so part of this is inference).
  All four experiments measured volume ratios in fL/L (parts per 10¹⁵). For
  ETEX and ANATEX the sampled volume was normalised by a background
  perfluorocarbon (ptPDCH) taken as constant, which makes the values ratios to
  a long-lived background gas, in effect dry-air mole fractions. For CAPTEX
  and OKC80 the volumes were pumped ambient air, with no correction stated;
  humidity would change them by 0.3–2.5%, below the measurement uncertainty.
  The pg m⁻³ and ng m⁻³ values in DATEM and the JRC files are the fL/L values
  times one fixed factor, not an ambient density: for CAPTEX and ANATEX
  pg m⁻³ = fL/L × M / 22.40 L mol⁻¹ (15.624 for PMCH), and for ETEX
  ng m⁻³ = fL/L × 0.015. **So GLIDE's dry-air mole fraction is compared
  directly, converted with the same fixed factors** (B10, B12), never with
  the model's air density, which would add a spurious 5–15%. No humidity
  correction. Sources: NOAA ARL READY tutorial on CAPTEX units
  (ready2.arl.noaa.gov, "chem_ratio"); ANATEX Vol. IV (EML 1991, OSTI 6100709,
  pp. 9, 15–17); the JRC ETEX pages (remon.jrc.ec.europa.eu, ETEX site pages
  24, 54, 55); EPA-600/7-81-006 for OKC80. Not read: the CAPTEX final report
  (Ferber et al. 1986) and the ANATEX archive report (ARL-167).
- *Test NAME's density treatment.* The per-cell, per-sample division comes
  from reading NAME's code. A short NAME run against a hand calculation would
  confirm it (NAME agent).
- *The NAME conversion as a tested function.* The NAME runner gives
  ppm s × 10⁻⁶ × cell area ÷ 3,600 s, with the source divided over the particles
  released. It goes into the comparison code with a test, with the B12 fixture.

**Implementation.** GLIDE: per-particle dry-air density in the gridder's
weight, the output setting, the footprint attributes and the helper's refusal.
Tests: the review's two cases (1.0 and 1.2 kg m⁻³; 90/10 and 10/90 s) against
`0.02897 / 40 × (s₁/ρ₁ + s₂/ρ₂)`; a constant-density case that equals the
unweighted footprint divided by ρ_d; then old against new on one case-library
release, to measure the real size of the error. The comparison code gains
the NAME and FLEXPART dry-air corrections. §6 of `docs/reference-runs.md` is
updated with this record.

**Status.** Adopted 2026-11-23 (GLIDE PR #42), from finding 3 of the
validation review of 21 Nov 2026 (`VALIDATION-HANDOFF.md`). Implemented by
queue task A04 (2026-11-24): `output_grid.density_weighting: dry_air | none`,
default `dry_air`; the pressure at the particle is the surface pressure carried
up hypsometrically with the virtual temperature at half the particle's height.
Measured on standard cases 3 and 4 against the old recipe (plain seconds
divided afterwards by the 240-hour mean surface density): −3.4% (CBW-20,
winter night) and +6.4% (LEF, summer afternoon) in total 0–40 m sensitivity.
Against the same recipe with a dry-air mean density, which leaves the pairing
of residence and density, the changes are −4.2% and +4.4%; the moist-to-dry
step is +0.8% and +2.0% (`dev/agent/progress/A04.md`). Docs:
[docs/reference-runs.md](../../docs/reference-runs.md) §6,
[docs/physics.md](../../docs/physics.md) §7.
