# GLIDE deep-convection scheme

Specification for the reduced port of FLEXPART's Emanuel & Živković-Rothman
(1999) deep-convection scheme (Forster, Stohl & Seibert 2007). Source-of-truth
for what `lpdm.convection.EmanuelReducedConvection` implements and why; the
audit deliverable for the "biggest non-audit dispersion gap" called out in the
2026-05-30 / 2026-05-31 CHECKPOINT entries.

## 1. Why convection

The Hanna 1982 boundary-layer turbulence scheme handles small-eddy mixing
inside the BL (`lpdm.turbulence.hanna`). The gradient-Richardson free-troposphere
closure handles slow shear-driven mixing above the BL. **Neither captures deep
moist convection** — cumulus updrafts that loft surface air through the entire
troposphere in minutes-to-hours. For mid-latitude continental sites in summer
and tropical sites year-round, this is a major missing transport mechanism:
particles released in the BL ought to occasionally end up at 8-12 km within
a few hours, where the resolved horizontal winds are much faster and more
variable.

This is also the dominant remaining physics gap GLIDE has vs FLEXPART (the
audit `physics-audit-may30` + `physics-audit-followup` PRs closed all the
WMC-violating gaps Wilson & Flesch 1993 / Stohl & Thomson 1999 documented,
but convection wasn't in the audit).

## 2. Architecture

The convection scheme is a separate runtime stage from turbulence:

```
src/lpdm/convection/
    __init__.py
    base.py            # ConvectionScheme ABC + registry
    no_convection.py   # NoConvection placeholder (default)
    emanuel.py         # EmanuelReducedConvection
```

The runtime calls `scheme.maybe_convect(...)` **once per met-update interval**
(typically hourly for ARCO ERA5), NOT every integration timestep:

- Convection redistributes particles non-locally in the vertical (a particle
  can jump from the surface to the tropopause in one event), whereas
  turbulence is small Brownian steps each ~dt.
- The convective mass-flux matrix depends only on the per-column met
  (temperature + humidity profile), so it doesn't change inside a met
  window — computing it per timestep would be wasteful.
- This matches FLEXPART's design (`convmix.f` is called every synctime, not
  every internal timestep; Stohl 2005 §4.6).

The runtime tracks the met bracket's start time and fires convection
exactly once whenever the cursor crosses into a new bracket.

### 2.1 Met-input contract

Beyond the baseline (`u/v/w/blh/sp/t`) the scheme requires:

| Logical key | ARCO ERA5 variable | Units | Type |
| --- | --- | --- | --- |
| `q` | `specific_humidity` | kg/kg | 3D (level, lat, lon) |

This is the only convection-specific dependency. The reader's
`DEFAULT_VARIABLE_MAP` already includes `q`; the runtime adds it to the
required channels when the convection scheme declares it.

### 2.2 Config

```yaml
convection:
  scheme: emanuel_reduced       # or "none" (default, bit-equivalent to no convection)
  emanuel:                      # only consulted when scheme == "emanuel_reduced"
    closure_c: 0.03             # cloud-base mass-flux closure constant
    trigger_dtv_k: 0.9          # Forster 2007 Eq 34 buoyancy threshold
    min_cape_j_kg: 50.0         # CAPE floor below which convection never fires
    min_cloud_depth_m: 500.0    # skip shallow convection
```

Default `scheme: "none"` means YAMLs without a `convection:` block produce
the same output as before (the bit-equivalence gate for the schema change).

## 3. Algorithm — `EmanuelReducedConvection`

Implements the FLEXPART/Forster (2007) scheme at a higher level of abstraction
than the FORTRAN `convect43c.f` (~3000 lines). Five steps per met update:

### 3.1 Parcel lift

Lift a surface parcel `(T_sfc, q_sfc, p_sfc)` through the column:

- **LCL** — Bolton (1980) Eq 22 gives the LCL temperature; Poisson's equation
  the LCL pressure (`lcl_temperature_bolton`, `lcl_pressure_poisson`).
- **Dry adiabat** below LCL: `T(p) = θ_sfc · (p/p_0)^κ`, with `κ = R_d/c_p`.
- **Moist adiabat** above LCL: `θ_e` is conserved. We solve `θ_e(T, q_sat(T,p), p) = θ_e_const`
  for T at each level via **bisection** (`lift_parcel_moist_pseudo_adiabatic`).
  Bisection is used because fixed-point iteration on `θ_e` fails to converge
  under Clausius-Clapeyron's strong nonlinearity (`dθ_e/dT` amplifies fast
  with T near deep-convection temperatures).

### 3.2 LCL / LNB / CAPE

`compute_lcl_lnb_cape` returns:

- `i_lcl` — first level above the LCL.
- `i_lnb` — Level of Neutral Buoyancy (cloud top); first level above LCL
  where buoyancy `(T_v_parcel − T_v_env)` crosses zero from positive.
- `cape` — `∫_LFC^LNB g · (T_v_p − T_v_env)/T_v_env · dz` (positive part of
  buoyancy times hypsometric layer thickness).
- `dtv_lcl_plus_1` — `T_v_parcel − T_v_env` at LCL+1, the Forster 2007 Eq 34
  trigger diagnostic.

### 3.3 Trigger

The Forster 2007 Eq 34 trigger fires when ALL of:

- `i_lcl ≥ 0` (parcel reaches saturation in the column)
- `i_lnb > i_lcl` (positive cloud depth)
- `CAPE ≥ min_cape_j_kg` (sanity floor)
- `dtv_lcl_plus_1 ≥ trigger_dtv_k` (Eq 34 with `T_threshold = 0.9 K`)
- cloud depth `≥ min_cloud_depth_m` (DEEP convection only — shallow cumulus
  is handled by the BL turbulence)

If any check fails, the scheme is a no-op for this met update.

### 3.4 Cloud-base mass flux

The closure (Emanuel 1991 §4; simplified):

```
w_buoy = min(√(2 · CAPE), UPDRAFT_W_MAX_M_S = 5 m/s)
M_b = closure_c · ρ_LCL · w_buoy
```

The cap on `w_buoy` is deliberate — without it CAPE > 500 J/kg gives
unphysically large M_b (much greater than FLEXPART's 0.05–0.5 kg/m²/s
realistic range). The 5 m/s cap represents a typical updraft-cell peak
rather than the rare 15 m/s extreme.

### 3.5 Mass-flux matrix `fmass[i, j]` (non-divergent)

`compute_mass_flux_matrix` builds a **non-divergent** off-diagonal flux matrix —
a coherent BL→cloud updraft plus compensating environmental subsidence — so that
the total mass leaving every layer equals the total entering it
(`Σ_j fmass[i,j] == Σ_j fmass[j,i]`). That property is what makes the
redistribution well-mixed-preserving (§3.7); the pre-2026-07-02 matrix had
updraft only and was divergent (Finding 2 of the physics review).

Construction (levels ascending, surface at 0; cloud = [LCL, LNB]):

- **Entrainment** `e[i]` — the boundary layer `[0, LCL)` feeds the updraft,
  shared across BL layers **by air mass** so the total entrained equals `M_b`.
  (Fixes Finding 3: previously each BL layer sourced the full `M_b`, over-venting
  the BL by a factor of the layer count.)
- **Detrainment** `d[j]` — the updraft deposits mass across the cloud with a
  linear-decay profile, total `M_b`.
- **Direct updraft** `U[i,j] = e[i]·d[j]/M_b` — BL source → cloud, non-local
  (the coherent deep-lofting transport).
- **Compensating subsidence** — the net upward flux across the interface below
  layer `k+1` is `Φ[k] = Σ_{i≤k} e[i] − Σ_{j≤k} d[j]` (≥ 0); the environment
  sinks at the same rate, `fmass[k+1, k] += Φ[k]`.

A single-scalar CFL cap scales the matrix so no layer sheds more than 90 % of its
mass per event (preserves non-divergence). Zero when there is no cloud.

### 3.6 Particle redistribution

Following FLEXPART `calcmatrix`/`redist`: apply the **diagonal closure** so each
row of `fmassfrac` sums to the layer air mass `m_i` (the diagonal is the "stay"
mass), then the per-particle destination distribution is

```
forward  (ldirect=+1):  P[host → j] = fmassfrac[host, j] / m_host   (matrix row)
backward (ldirect=−1):  P[host → j] = fmassfrac[j, host] / m_host   (matrix column)
```

GLIDE runs backward, so it samples the **column**. Sample a uniform `u ∈ [0, 1]`,
find the destination via cumulative-sum search (the distribution includes the
stay-diagonal), and place movers at a uniformly random height within the
destination layer (sub-bin placement). Non-divergence makes both the row- and
column-normalised transitions valid probability distributions, so no ad-hoc
move-probability clamp is needed.

### 3.7 Backward mode

Backward sampling uses the transposed (column) transition — the adjoint of the
forward updraft: forward convection lofts surface air to the cloud top, so the
backward (footprint) operator traces a particle that is aloft *now* down to the
BL air it came from. Because `fmass` is non-divergent, the layer-mass vector is a
stationary distribution of BOTH the forward (row) and backward (column)
transitions: an initially well-mixed (mass-proportional) ensemble stays
well-mixed after convection in either time direction (Forster 2007 §3 + Fig 2).
This is verified deterministically by
`tests/test_convection.py::test_convection_transition_preserves_mass_distribution_both_directions`
(mᵀP = mᵀ) and behaviourally by
`test_emanuel_backward_connects_aloft_particles_to_the_surface`.

## 4. Documented departures from full Emanuel

1. **Bbox-mean column**, not per-(lon,lat) columns. The full Emanuel scheme
   processes each ECMWF grid column independently; we use the bbox-mean
   profile for the parcel lift. This means convection fires uniformly across
   the bbox (one decision for the whole met domain). For small domains this
   is a fine approximation; for very large domains (e.g. continental Europe)
   it can over- or under-trigger compared to FLEXPART. Mirrors the F9
   approximation used in advection (see CHECKPOINT 2026-05-31 entry).

2. **Linear detrainment profile**, not the Emanuel buoyancy-sorting spectrum
   (Forster 2007 Eq 35-36). The updraft detrains with `d(z) ∝ (1 − (z−LCL)/
   (LNB−LCL))`. This sets only WHERE the updraft deposits mass, not the
   mass-conservation structure — non-divergence (hence the well-mixed property)
   is guaranteed by the compensating subsidence regardless of the profile shape.
   The earlier code used the buoyancy-sorting fractions in a divergent matrix
   that violated the well-mixed criterion; that was replaced 2026-07-02.

3. **No explicit saturated-downdraft branch**. Emanuel 1991 §4.b includes a
   separate downdraft mass-flux matrix; the compensating environmental
   subsidence carries the return flux instead. Few-percent effect.

4. **Capped buoyancy velocity** at 5 m/s in the closure. The full scheme
   has a quasi-equilibrium closure (Emanuel's mass-flux balances large-scale
   destabilisation); we cap `√(2·CAPE)` to keep M_b in the realistic range.

5. **Compensating subsidence is adjacent-layer**. The environmental return flux
   moves mass one layer down per event (`fmass[k+1, k] = Φ[k]`), whereas the
   updraft is non-local (BL → any cloud level in one event). Physically apt
   (fast coherent updraft, slow broad subsidence), but the subsidence descent of
   *non-displaced* environmental air is a random walk rather than a prescribed
   velocity. This is now modelled explicitly (it was absent before 2026-07-02),
   which is what makes the scheme well-mixed-preserving.

## 5. Tests (tests/test_convection.py)

- **Free-function physics**: q ↔ e round-trip, T_v, LCL temperature / pressure,
  potential temperature, equivalent potential temperature, saturation specific
  humidity, moist-adiabat lift (θ_e conserved through lift to a higher level).
- **CAPE / LCL / LNB**: synthetic moist-unstable column produces CAPE > 100
  J/kg and meaningful LCL/LNB indices; dry-stable column produces CAPE ≈ 0.
- **Mass-flux matrix**: zero when no cloud (`i_lcl < 0`); no transport above the
  cloud top; **non-divergent** (row sum = column sum per layer, i.e. compensating
  subsidence present); and the derived transition matrix leaves the layer-mass
  vector stationary (`mᵀP = mᵀ`) for BOTH forward and backward — the well-mixed
  criterion, proven deterministically.
- **Scheme-level**: `NoConvection` pass-through; end-to-end **backward** transport
  connecting aloft particles to the boundary-layer source (net downward, mass
  conserved); no displacement on a stable sounding; constructor validation.

## 6. Acceptance / next steps

The integration acceptance is the FLEXPART comparison re-run. With the
convection scheme enabled in `configs/example_mhd_january_periodic.yaml`,
expect:
- More particles in the free troposphere at long backward times
- Surface footprint becomes more dispersed (convective lofting → faster
  long-range transport)
- Mid-latitude January 2024 may show modest changes (winter, weak
  convection); the test will be more meaningful in summer.

If the FLEXPART gap is still meaningful after enabling convection, the most
likely remaining levers are: per-column (not bbox-mean) profiles for the
parcel lift, the full Emanuel quasi-equilibrium closure replacing our capped
buoyancy velocity, and the saturated-downdraft branch.

## 7. References

- Emanuel, K. A. (1991). A scheme for representing cumulus convection in
  large-scale models. *J. Atmos. Sci.* 48, 2313–2335.
- Emanuel, K. A., and Živković-Rothman, M. (1999). Development and evaluation
  of a convection scheme for use in climate models. *J. Atmos. Sci.* 56, 1766–1782.
- Forster, C., Stohl, A., and Seibert, P. (2007). Parameterization of convective
  transport in a Lagrangian particle dispersion model and its evaluation.
  *J. Appl. Meteor. Climatol.* 46, 403–422.
- Stohl, A., Forster, C., Frank, A., Seibert, P., and Wotawa, G. (2005). Technical
  note: The Lagrangian particle dispersion model FLEXPART version 6.2.
  *Atmos. Chem. Phys.* 5, 2461–2474, §4.6 (Moist convection).
- Bolton, D. (1980). The computation of equivalent potential temperature.
  *Mon. Wea. Rev.* 108, 1046–1053.
- Telford, J. W. (1975). Turbulence, entrainment, and mixing in cloud dynamics.
  *Pure Appl. Geophys.* 113, 1067–1084.
