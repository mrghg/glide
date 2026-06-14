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

### 3.5 Mass-flux matrix `MA[i, j]`

`compute_mass_flux_matrix` builds the [Z, Z] matrix with three groups of
source rows:

- **Sub-LCL (i < LCL)** — surface-updraft path. The BL acts as a single
  source feeding the cloud base; `MA[i, j] = M_b · profile(j)` for
  `j ∈ [LCL, LNB]`, normalised so row sum equals `M_b` exactly.
- **In-cloud (i ∈ [LCL, LNB])** — Emanuel buoyancy-sorting per Forster 2007
  Eq 35-36. Mixing fraction
  `ε_{i,j} = (θ_j − θ_lp_{i,j}) / (θ_l_{i,j} − θ_lp_{i,j})`;
  `MA[i, j]` from Eq 35 normalised so the row sums to `M_b · profile(i)`
  (with profile linearly decaying from M_b at LCL to 0 at LNB).
- **Above LNB (i > LNB)** — zero. Particles above the cloud top stay put.

### 3.6 Particle redistribution

For each eligible particle (host level ≤ LNB):

```
prob(stay) = 1 − Σ_j MA[host, j] / m_host
prob(move to j) = MA[host, j] / m_host
```

`m_host` is the layer mass per area (`Δp/g`). Sample a uniform `u ∈ [0, 1]`;
if `u < total_move`, find destination via cumulative-sum search; place at a
uniformly random height within the destination layer (sub-bin placement).

Total move probability is clamped at 0.99 (numerical safety; rare for
realistic closures).

### 3.7 Backward mode

The same matrix is applied in backward integration. The mass-flux matrix is
constructed to preserve the well-mixed criterion under either time direction
(Forster 2007 §3 + Fig 2 in the paper): an initially well-mixed tracer remains
well-mixed after convection in BOTH forward and backward runs. The redistribution
is symmetric under time reversal because mixing fractions sum to mass-conserving
matrices.

## 4. Documented departures from full Emanuel

1. **Bbox-mean column**, not per-(lon,lat) columns. The full Emanuel scheme
   processes each ECMWF grid column independently; we use the bbox-mean
   profile for the parcel lift. This means convection fires uniformly across
   the bbox (one decision for the whole met domain). For small domains this
   is a fine approximation; for very large domains (e.g. continental Europe)
   it can over- or under-trigger compared to FLEXPART. Mirrors the F9
   approximation used in advection (see CHECKPOINT 2026-05-31 entry).

2. **Buoyancy-sorting matrix simplified**. Forster 2007 Eq 36's mixing
   fraction uses liquid-water potential temperature with full microphysics;
   we approximate `θ_l ≈ θ` (latent-heat reservoir implicit in `q_sat`).
   Few-percent effect on the redistribution profile.

3. **No saturated-downdraft branch**. Emanuel 1991 §4.b includes a separate
   downdraft mass-flux matrix; we fold this into the mass conservation
   residual rather than modelling explicitly. Few-percent effect.

4. **Linear M-profile**, not the buoyancy-driven profile from the full
   scheme. `M(z) = M_b · (1 − (z−LCL)/(LNB−LCL))` is a simple closure that
   matches the order of magnitude of the full Emanuel profile.

5. **Capped buoyancy velocity** at 5 m/s in the closure. The full scheme
   has a quasi-equilibrium closure (Emanuel's mass-flux balances large-scale
   destabilisation); we cap `√(2·CAPE)` to keep M_b in the realistic range.

6. **Forster 2007's "subsidence velocity" not modelled**. The compensating
   subsidence in convectively undisturbed cells is implicit in our scheme
   (mass conservation handles it). FLEXPART models it as a downward velocity
   for non-displaced particles. Effect: slight bias in the vertical
   distribution of *non-displaced* particles in convective columns.

## 5. Tests (tests/test_convection.py)

- **Free-function physics**: q ↔ e round-trip, T_v, LCL temperature / pressure,
  potential temperature, equivalent potential temperature, saturation specific
  humidity, moist-adiabat lift (θ_e conserved through lift to a higher level).
- **CAPE / LCL / LNB**: synthetic moist-unstable column produces CAPE > 100
  J/kg and meaningful LCL/LNB indices; dry-stable column produces CAPE ≈ 0.
- **Mass-flux matrix**: zero when no cloud (`i_lcl < 0`); non-zero entries
  confined to source rows ≤ LNB and destination columns ∈ [LCL, LNB];
  in-cloud row sums equal the M-profile; BL rows sum to M_b exactly.
- **Scheme-level**: `NoConvection` pass-through; end-to-end lofting of BL
  particles to mid-troposphere in a convective column; no displacement on a
  stable sounding; constructor validation.

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
