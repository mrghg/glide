# GLIDE turbulence parameterization

Specification for the M1 turbulence rewrite: scope, modular architecture, scheme math, met-input contract, and validation plan. This document is the source of truth for the turbulence design; implementation should match it.

## 1. Goals and scope

- Replace the M0 placeholder constant-OU vertical scheme with met-driven Hanna 1982 turbulence (the FLEXPART formulation), in three stability regimes.
- Add horizontal stochastic diffusion (currently missing entirely).
- Make scheme selection modular — alternative schemes (Wilson-Sawford, hybrid Hanna-Degrazia, future ML schemes) should land as plug-in subclasses without touching the runtime loop.
- Above-BL turbulence is in scope from day 1 (now a gradient-Richardson closure — see §3.2.3; the original constant-diffusivity placeholder was superseded 2026-05-29).
- Backward integration carries the full Thomson (1987) well-mixed drift with the Stohl-Thomson (1999) density correction, sign-flipped for the backward Langevin (see §3.2.5). (An earlier version used a piecewise-homogeneous no-drift treatment; that under-dispersed the surface footprint and was replaced.)
- Keep the existing constant-OU behaviour available as a registered scheme (`PlaceholderConstantOU`) so the M0 baselines remain reproducible and regression-pinnable.

## 2. Architecture

### 2.1 Module layout

```
src/lpdm/turbulence/
    __init__.py         # exports the base class, registry, and convenience get_scheme(name)
    base.py             # TurbulenceScheme ABC, TurbulenceState container
    placeholder.py      # PlaceholderConstantOU (M0 behaviour, regression baseline)
    hanna.py            # HannaScheme (M1 primary)
```

Schemes register themselves with a name (e.g. `"placeholder_constant_ou"`, `"hanna_1982"`). The runtime selects via `lpdm.turbulence.get_scheme(name)` or by passing an instance directly — both paths are first-class.

### 2.2 Base class

```python
class TurbulenceScheme(ABC):
    name: ClassVar[str]

    @abstractmethod
    def required_met_keys(self) -> tuple[str, ...]:
        """Logical met-variable keys this scheme reads from HourlyMetTensors.

        Names must match keys in ArcoEra5ZarrReader.DEFAULT_VARIABLE_MAP. The runtime
        cross-checks these at startup so missing fields fail loud rather than silently
        defaulting.
        """

    @abstractmethod
    def initialize_state(
        self, n_particles: int, *, device: torch.device, dtype: torch.dtype
    ) -> TurbulenceState:
        """Allocate per-particle state (e.g. perturbation velocities)."""

    @abstractmethod
    def step(
        self,
        particles: torch.Tensor,                # (N, 4) [lon, lat, alt, weight]
        state: TurbulenceState,
        met_window: HourlyMetTensors,
        t_alpha: float,                         # temporal interp weight in [0, 1]
        dt_seconds: float,
        active_mask: torch.Tensor,              # (N,) bool
        engine: GPUEngine,
    ) -> tuple[torch.Tensor, TurbulenceState]:
        """Apply one turbulence step. Returns (updated particles, updated state)."""
```

`TurbulenceState` is a small dataclass-like container (`dict[str, torch.Tensor]` under the hood). Schemes choose what to track: `PlaceholderConstantOU` stores only `{"w_prime"}`; `HannaScheme` stores `{"w_prime", "u_prime", "v_prime"}`.

### 2.3 Engine primitives

`GPUEngine` already has `update_langevin_velocity(w_prime, t_lagrangian, sigma_w2, dt_seconds)` and `apply_vertical_turbulence(particles, w_prime, dt_seconds)`. Both already accept either scalars or per-particle tensors for `t_lagrangian` and `sigma_w2`, so per-particle Hanna parameters work via broadcasting — no API change needed.

M1 adds one new primitive:

```python
def apply_horizontal_turbulence(
    self,
    particles: torch.Tensor,
    u_prime: torch.Tensor,
    v_prime: torch.Tensor,
    dt_seconds: float,
    *,
    backward: bool = True,
) -> torch.Tensor:
    """Apply horizontal displacement from turbulent velocity fluctuations.

    Mirrors apply_vertical_turbulence but in lon/lat with the cos-lat correction.
    """
```

The split is deliberate: schemes own the math of computing per-particle `T_L` and `σ²` from met fields; the engine owns the OU update math and the geometric application of perturbation velocities.

### 2.4 Met-input contract

Schemes declare their required met keys via `required_met_keys()`. The runtime loop:

1. Asks the active scheme for its required keys.
2. Unions them with the engine's baseline (`u`, `v`, `w`, `z`, `z_sfc`, `t`, `sp`, `blh`).
3. Configures the reader's variable map and runs the dataset's variable check at startup (`_select_variables`).
4. Slices and interpolates each requested field at the particle position when called by the scheme.

New keys for `HannaScheme`:

| Logical key | ARCO ERA5 variable | Units | Type |
| --- | --- | --- | --- |
| `ustar` | `friction_velocity` | m/s | single-level |
| `shf` | `surface_sensible_heat_flux` (or `instantaneous_surface_sensible_heat_flux` if available) | W/m² | single-level |

`ushf`/`shf` is preferred as instantaneous; if only the accumulated form is available we de-accumulate at fetch time. `download_sample_cube.py` will be updated to fetch both new variables in the same Zarr crop.

The reader currently slices off all but the first three channels (`met_window.hour_start[:3]`) at the call site in `main.py`. M1 removes that slice and passes the full `HourlyMetTensors` to the scheme; the scheme picks out what it needs by channel index (or, ideally, by named accessor — see §2.6).

### 2.5 Scheme selection

Two paths, both first-class:

- YAML config: `turbulence.scheme: hanna_1982` (the `turbulence:` block owns this
  surface; resolved via `lpdm.turbulence.get_scheme(name)`).
- Programmatic: pass a `TurbulenceScheme` instance directly to `_run(cfg, *, reader, scheme=...)`. This is what tests use.

### 2.6 Channel access (small refactor)

Today the channel order in `HourlyMetTensors.hour_start` is implicit (`[u, v, w, blh, sp]`). With more variables, this gets fragile. M1 adds a small mapping field on `HourlyMetTensors` (e.g. `channel_index: dict[str, int]`) so schemes write `met.channel("ustar")` rather than `met.hour_start[5]`. This is a low-risk localized change; the existing `[:3]` use-site in `_advect_active_particles` becomes `met.channel("u")`/`channel("v")`/`channel("w")`.

## 3. Schemes

### 3.1 `PlaceholderConstantOU` (regression baseline)

Reproduces M0 behaviour exactly: vertical OU only, hard-coded `T_L = 300 s`, `σ_w² = 1.0 m²/s²`, no horizontal turbulence.

- State: `{"w_prime": (N,)}`.
- Required met keys: `()` (parameters are constants).
- Step: forwards to `engine.update_langevin_velocity(...)` and `engine.apply_vertical_turbulence(...)` with the fixed parameters.

Stays in the codebase indefinitely as the M0 placeholder reference.

### 3.2 `HannaScheme` (M1 primary)

Implements Hanna 1982 as adopted by FLEXPART (Stohl et al. 1998; Stohl et al. 2005). Three stability regimes within the boundary layer; constant-diffusivity above.

#### 3.2.1 Stability classification

Compute Obukhov length per particle:

```
L = -u*³ T_v ρ c_p / (κ g H)
```

where `u*` is friction velocity (interpolated to the particle column), `T_v` is virtual temperature (approximated as `T` at lowest model level — humidity correction is small, deferred), `ρ` is surface air density derived from `sp` and `T_v`, `c_p ≈ 1005 J/(kg·K)`, `κ ≈ 0.4` (von Kármán), `g = 9.80665 m/s²`, `H` is sensible heat flux (W/m², positive upward).

Convective velocity scale (used in unstable regime):

```
w* = ((g h H) / (T ρ c_p))^(1/3)
```

where `h` is BLH and the temperature is at the top of the surface layer.

Regime selection by `h / L`:

- `h/L > 1`     → **stable**
- `-1 ≤ h/L ≤ 1` → **neutral**
- `h/L < -1`    → **unstable / convective**

(Thresholds match FLEXPART defaults; small adjustments are possible during implementation calibration.)

#### 3.2.2 In-BL formulae (z < h)

Equations from Hanna (1982), Caughey (1982) and Ryall & Maryon (1998) (attributed
per formula below). GLIDE adopts the same boundary-layer parameterisation FLEXPART
uses (`src/turbulence_mod.f90` `subroutine hanna`) so the two are directly
comparable; the coefficients were checked against that reference on 2026-07-02
(Finding 5 of the physics review). ζ = z/h. Note `σ_v` tracks `σ_w` (both 1.3 u*),
not `σ_u`, and the neutral/stable T_L components differ per component.

Stable BL (Hanna 1982 Eqs 7.19–7.24):
```
σ_u = 2.0 u* (1 - ζ)
σ_v = σ_w = 1.3 u* (1 - ζ)          # LINEAR in (1-ζ), not (1-ζ)^¾
T_Lu = 0.15 h / σ_u · √ζ
T_Lv = 0.467 T_Lu
T_Lw = 0.10 h / σ_w · ζ^0.8
```

Neutral BL (Hanna 1982 Eqs 7.25–7.27; per-particle |f|):
```
σ_u = 2.0 u* exp(-3 |f| z / u*)     # note 3|f| for σ_u
σ_v = σ_w = 1.3 u* exp(-2 |f| z / u*)
T_Lu = T_Lv = T_Lw = 0.5 z / σ_w / (1 + 15 |f| z / u*)
```
FLEXPART hardcodes f = 1e-4 s⁻¹; GLIDE uses the per-latitude Coriolis but takes
`|f|` (with a ~1e-5 floor) so the Ekman-decay exponents don't flip sign in the
Southern Hemisphere (Finding 4).

Unstable / convective BL (Caughey 1982 Eq 4.15; Ryall & Maryon 1998; Hanna 1982
Eq 7.17):
```
σ_u = σ_v = u* (12 - 0.5 h/L)^(1/3)
σ_w = √(1.2 w*² (1 - 0.9 ζ) ζ^(2/3) + (1.8 - 1.4 ζ) u*²)
T_Lu = T_Lv = 0.15 h / σ_u
T_Lw = 0.10 z / (σ_w (0.55 - 0.38 |z/L|))   for |z/L| < 1     (near-surface)
     = 0.59 z / σ_w                          for ζ < 0.1       (shallow)
     = 0.15 h / σ_w (1 - exp(-5 ζ))          otherwise         (bulk)
```

**T_L floors (`turbulence.flexpart_tl_floors`, default `true`):** FLEXPART's
Lagrangian-timescale floors (T_Lu, T_Lv ≥ 10 s; T_Lw ≥ 30 s) are applied to the
BL profile (before the free-troposphere override, whose (σ_w, T_Lw) pair defines
K and must not be re-floored). Without them the T_Lw formulas vanish as z → 0,
collapsing the near-surface K = σ_w²·T_Lw and trapping particles in the lowest
metres — the GLIDE ≫ NAME/FLEXPART enhancement over-estimation at polluted
low-inlet sites (v2 validation, 2026-07-02). A convenient side-effect: with
T_Lw ≥ 30 s the adaptive-substep cap (§3.2.7) rarely binds. Set `false` only for
legacy A/B comparisons.

#### 3.2.3 Above-BL (z > h) — gradient-Richardson closure

**Superseded the day-1 constant-diffusivity placeholder (2026-05-29).** The old
`σ = 0.1 m/s, T_L = 100 s` constant was a one-way trap: once a particle left the
BL it was effectively frozen (`K ≈ 1 m²/s`) and could not mix back to the
surface, starving the surface footprint. The free troposphere now uses a
first-order gradient-Richardson closure computed from model-level fields:

```
θ   = T (p0/p)^κ                         (κ = R_d/c_p)
N²  = (g/θ) ∂θ/∂z                        (Brunt–Väisälä)
S²  = (∂u/∂z)² + (∂v/∂z)²                (wind shear)
Ri  = N² / S²
l   = κ_vk z / (1 + κ_vk z / λ)          (Blackadar mixing length, λ = 100 m)
f(Ri) = (1 - Ri/Ri_c)²   for 0 ≤ Ri < Ri_c (Ri_c = 0.25)
      = √(1 - 16 Ri)     for Ri < 0
      = 0                for Ri ≥ Ri_c
K_z = clamp(l² |∂U/∂z| f(Ri),  K_floor=0.1,  K_ceil=50 m²/s)
T_Lw = clamp(0.5/N, T_L_min, 1000 s)     (buoyancy timescale; fallback where N²≤0)
σ_w  = √(K_z / T_Lw)
```

The vertical gradients `∂θ/∂z`, `∂u/∂z`, `∂v/∂z` are central differences over the
model levels using the **true 3-D geopotential heights** (`HourlyMetTensors.height_agl_m`,
exposed by the met reader), not the bbox-averaged `metadata.level`. The resulting
σ/T_L fields are built once per step on the met grid and interpolated trilinearly
at each above-BL particle (reusing the advection's coordinate normalisation). The
`K_floor` guarantees the FT is never fully frozen. Horizontal FT turbulence is
treated as isotropic (`σ_u = σ_v = σ_w`); the unresolved-mesoscale "meander"
horizontal term (which adds horizontal spread at all altitudes, BL and FT alike)
is handled by a separate process — see §3.2.8. Implemented as free functions
`potential_temperature`, `brunt_vaisala_squared`, `gradient_richardson`,
`free_trop_diffusivity`, `free_trop_sigma_TL` in `hanna.py`.

#### 3.2.4 Surface-layer treatment

**Default: no surface-layer override** (`turbulence.surface_layer_override:
false`). FLEXPART v11 `subroutine hanna` has no separate surface-layer treatment
— the §3.2.2 regime formulas run to the ground, bounded by the T_L floors — and
GLIDE now follows it. The previous GLIDE-only MO override both undercut the
floors near the surface (its `T_L = κz/σ` → seconds as z → 0) and put a K
discontinuity at the `0.1 h` seam.

When enabled (legacy / A/B comparisons only), the override applies below
`z_sl = 0.1 h`:

```
σ_w = 1.3 u* (1 - 3 z/L)^(1/3)   (unstable; Flesch et al. 1995 App. B)
σ_w = 1.3 u*                     (stable / neutral — CONSTANT in height)
σ_u = σ_v as in §3.2.2 with z → max(z, z_sl)
T_L  = κ z / σ                    (then floored per §3.2.2 if floors are on)
```

The stable σ_w is height-independent (σ_w/u* ≈ 1.3 in the stable surface layer).
An earlier `1.3 u*(1 + 5 z/L)` form GREW with stability — that is the φ_m
momentum-gradient function, not a σ_w scaling — and over-mixed the nocturnal
near-surface layer (Finding 6 of the 2026-07-02 review).

**Reflection (Wilson & Flesch 1993 §6).** Smooth-wall reflection at `z = 0` is the
joint mapping `(z, w) → (2·z_surf − z, −w)` — **both** position and vertical
perturbation velocity must reverse. Reflecting only `z` (the pre-audit behaviour,
F1 in the 2026-05-30 physics audit) leaves the reflected particle pointing
downward into the boundary for ~τ_L worth of steps, biasing near-surface
residence and inflating the surface footprint. `engine.reflect_surface` returns
`(particles, w_prime)` to make the joint reversal hard to omit. W&F §7b show
even correctly-implemented smooth-wall reflection is only WMC-exact for
*homogeneous* Gaussian turbulence; in the inhomogeneous NSL it is approximate and
the bias scales with `Δt/τ` (see §3.2.7).

**Unresolved basal layer (W&F §7b, "constant-σ basal layer").** To make
smooth-wall reflection WMC-exact in the basal layer, the Hanna scheme holds
σ_w / T_L / drift / density-gradient CONSTANT for any particle below
`z_ubl_m` (default 2 m) by clamping the *sampling* height: `z_eval = max(z, z_ubl_m)`.
The particle *position* is not clamped — only the σ/T_L evaluation. The default
2 m is thin enough to only intercept post-reflection bounces and matches the
FLEXPART/NAME convention of treating the very-near-surface layer as a UBL.
Configurable via the `HannaScheme(z_ubl_m=…)` constructor argument; set to 0 to
disable. F5/F15 in the 2026-05-30 physics audit.

#### 3.2.5 Drift handling — Thomson well-mixed term + Stohl-Thomson density correction

**Added 2026-05-29** (well-mixed drift) and **2026-05-30** (density correction).
The original piecewise-homogeneous no-drift treatment under-dispersed the
surface footprint badly — particles drifted up the σ_w gradient, accumulated
above the BL, and stopped recycling to the surface.

The vertical OU update carries the full Thomson (1987) / Stohl-Thomson (1999)
WMC drift for a ρ-weighted Gaussian `g_a`:

```
a_drift = ½ (1 + w'²/σ_w²) ∂σ_w²/∂z   +   (σ_w² / ρ) · ∂ρ/∂z
          └─── σ-gradient term ───┘       └── density term ──┘
                (Thomson 1987)              (Stohl & Thomson 1999 Eq 3)
```

- `∂σ_w²/∂z` is a central finite difference of the *full* column σ_w profile
  (in-BL → surface-layer → free-troposphere), so it spans the regime transitions.
- `ρ = p/(R_d·T)` and `∂ρ/∂z` are computed once per step on the met grid
  (`_density_fields`, mirroring the FT-closure machinery) using model-level
  pressure, temperature, and the per-column geopotential height; trilinearly
  sampled per particle.
- The density term is the WMC-required correction for the fact that air density
  falls with height: without it, an initially ρ-weighted distribution relaxes
  toward flat, biasing surface concentrations downward (Stohl & Thomson 1999;
  their CAPTEX runs measured +5.5% mean / +1–15% range on surface concentrations).
- `engine.update_langevin_velocity` applies both pieces as a forward-Euler
  `drift·dt` increment on top of the exact-OU homogeneous part. The increment is
  capped at one σ_w per step so the sharp σ_w kink at the BL top can't blow the
  velocity up between substeps (a tracking artefact that goes away if `dt` is
  small enough per §3.2.7).

**Backward-Langevin sign.** GLIDE runs backward in time; the displacement
negates w'·dt. The random forcing is symmetric so that's harmless, but the drift
is *deterministic* and must enter with reversed sign for the adjoint/backward
Langevin. From Flesch, Wilson & Yee (1995) the backward drift is
`a_b = −a_f + b² · ∂ln g_a/∂w = −a_f − 2w/τ`, which for symmetric Gaussian `g_a`
flips **both** the σ-gradient and density inhomogeneity terms in lockstep while
leaving the `−w/τ` relaxation unchanged. So `drift_w = −drift_w` after assembling
the forward formula gives the correct backward drift for both terms.

Getting this sign wrong inverts the σ-gradient correction into a one-way upward
pump — empirically it lofted the entire MHD population to ~2 km and held it
there. Covered by `test_v1_well_mixed_hanna_backward_path` (constant ρ, flat
distribution preserved) and `test_v1_density_weighted_well_mixed_with_F2`
(varying ρ, ρ-weighted distribution preserved).

Horizontal components (u', v') do not carry a drift — the inhomogeneity is in z,
and FLEXPART applies the well-mixed correction to the vertical only.

#### 3.2.6 State

`{"w_prime": (N,), "u_prime": (N,), "v_prime": (N,)}`, plus
`{"u_meander": (N,), "v_meander": (N,)}` when meander is enabled (§3.2.8).

Initial state: zeros. A strictly correct Thomson formulation would initialise from the local σ distribution; this is a deliberate simplification because particles equilibrate to the local σ within ~`T_L` (typically 100 s) and the release window itself is usually longer.

#### 3.2.7 Time-step constraint

The exact-OU update is unconditionally stable (the homogeneous part preserves
the stationary variance for any `dt`), but accuracy of the inhomogeneous-drift
increment degrades when `dt > T_L / 5` (Wilson & Flesch 1993 Appendix derive
an explicit "Δt-bias velocity" `wB/σ_w ≈ −α·β·(Δt/τ)` for the NSL; α ≈ β ≈ 0.5).
Stohl & Thomson (1999) use the stricter `Δt ≤ 0.05·T_L`. The runtime uses
`dt = 60–300 s` in current configs; near-surface near-floor `T_Lw` can drop to
1–30 s.

**F4 Tier 2 (audit 2026-05-30) — per-particle adaptive substepping.**
`HannaScheme.step` integrates the OU + displacement + reflection in
`k_i = ceil(dt / (substep_c · T_Lw_i))` per-particle substeps (default
`substep_c = 0.5`, capped at `max_substeps = 50`). The substepping is
vectorised via masking — only particles still owing substeps are touched in
each loop iteration. σ², T_L, density gradient, and the gradient piece of the
drift are held FIXED at their outer-step values (a deliberate simplification;
FLEXPART re-evaluates per substep, but the cost of per-substep
`_column_turbulence` is significant for moderate σ_w gradients across one
outer dt). The `(1 + w'²/σ_w²)` velocity factor inside the σ-gradient drift IS
recomputed per substep using the current `w'` — holding it fixed would
systematically under-drift newly-released particles (caught by
`test_hanna_well_mixed_no_runaway_lofting`).

Also inside the substep loop:
- F1 reflection is applied at every substep so a particle that crosses z=0 in
  a substep is reflected before the next substep (rather than only at the
  outer-step end).
- The OU velocity is clipped at `|w'/σ_w| ≤ 4` (W_PRIME_SIGMA_RATIO_MAX). This
  is a rogue-trajectory safeguard for particles where the substep cap binds
  and σ_w is near its floor (e.g. at the BL top); without it, the
  `(1+w'²/σ_w²)` factor can snowball drift and produce NaNs. FLEXPART has the
  equivalent clip in its OU implementation.

**F3 (audit 2026-05-30) — drift cap removed.** The legacy cap `|drift| ≤ σ_w/Δt`
was an ad-hoc safeguard for sharp σ_w kinks under too-large Δt; it silently
violated the WMC formula at the SL / BL-top seams. Replaced by F4 Tier 2 +
the rogue-trajectory clip, which address the root cause.

**Cap-saturation warning.** When the per-particle `k_i` hits the `max_substeps`
cap for any active particle, `HannaScheme` emits a `WARNING` (once per scheme
instance) noting `sub_dt/T_Lw` is still above the target and pointing to
`simulation.dt_seconds` / `max_substeps` as the remedies.

The reflection-related part of this same numerical concern (W&F §7b's
reflection-with-finite-Δt/τ bias) is bounded by the same Δt rule and further
mitigated by the F5/F15 constant-σ UBL (§3.2.4) which makes reflection WMC-exact.

#### 3.2.8 Meander — unresolved-mesoscale horizontal turbulence

**Added 2026-05-29.** The Hanna/FT scheme above parameterizes 3-D turbulence
(small eddies) and grid-scale advection (the resolved wind), but leaves a gap:
quasi-2-D mesoscale motions *larger* than turbulent eddies yet *smaller* than the
met grid can resolve. Neglecting them under-disperses the plume horizontally — the
residual seen in the GLIDE↔FLEXPART comparison after the FT/drift fixes.

GLIDE follows the **Maryon (1998) "meandering" scheme** that NAME originated and
FLEXPART adopted (Stohl et al. 2005, §4.5): an *independent* horizontal Langevin
(OU) process added on top of the Hanna `u'/v'` turbulence, applied at **all
altitudes**:

```
σ_meander,i = C_meander · stddev_local(U_i)        i ∈ {u, v}
τ_meander   = ½ · (met field interval)             (≈ 1800 s for hourly ERA5)
```

- `stddev_local(U_i)` is the standard deviation of the grid-scale wind component
  over the `(2r+1)²` horizontal neighbourhood of the particle (`r =
  meander.stencil_radius`, default 1 → 3×3), computed per model level and
  interpolated trilinearly to the particle. The assumption (Maryon; Stohl et al.
  1995) is that the *grid-scale* wind variability carries information about the
  *sub-grid* variability.
- `C_meander` is FLEXPART's `turbmesoscale` constant, default **0.16**
  (`meander.coefficient`).
- The timescale is half the input-field interval, on the reasoning that linear
  interpolation between fields already recovers about half the sub-interval
  variability.

**Resolution dependence** falls out for free: a finer met grid has smaller wind
differences between neighbouring cells → smaller `stddev_local` → smaller meander,
because more of the mesoscale is already resolved. This is the behaviour expected
of NAME's resolution-dependent meander, without a hand-tuned per-resolution
constant.

The process has **no drift** (the well-mixed inhomogeneity is vertical) and uses
symmetric random forcing, so the backward displacement needs no sign flip (unlike
the vertical well-mixed term, §3.2.5). State: `{u_meander, v_meander}`, zeros at
release. Built once per step on the met grid (`_meander_sigma_fields`,
`_windowed_std`) and sampled per-particle.

Config (`turbulence.meander`): `enabled` (default **off** so existing baselines
are bit-identical), `coefficient`, `stencil_radius`, `timescale_seconds` (null →
the half-interval default). Enabled in the shipped `example_mhd_january*.yaml`.
Hanna-only; ignored by other schemes. Covered by
`test_hanna_meander_increases_horizontal_spread` and the `_windowed_std` unit
tests.

## 4. Implementation plan

1. **Subpackage scaffold** — create `src/lpdm/turbulence/{__init__.py,base.py}` with the abstract base and the registry helper.
2. **Move placeholder out of `main.py`** — extract the constant-OU step into `placeholder.py` as `PlaceholderConstantOU`. Wire the runtime loop to the new dispatch. Confirm M0 tests still pass against this scheme bit-for-bit.
3. **Add `HourlyMetTensors.channel(name)` accessor** so schemes index by name. Migrate the existing `[:3]` slice in `_advect_active_columns` to use the accessor.
4. **Extend `met_reader.py`** to fetch `ustar` and `shf` (with units validation, optional de-accumulation for accumulated SHF).
5. **Update `download_sample_cube.py`** to include the new variables in the local cube.
6. **Implement `HannaScheme`** in stages, each with its own tests:
   - 6a. Stability classification helper (Obukhov length, w*, regime selection).
   - 6b. In-BL formulae per regime (returns per-particle σ_u, σ_v, σ_w, T_Lu, T_Lv, T_Lw).
   - 6c. Above-BL constant-K branch.
   - 6d. Surface-layer override.
   - 6e. Stitch together in `step()`: interpolate met → classify → compute params → call `engine.update_langevin_velocity` for w', u', v' → call `engine.apply_vertical_turbulence` and `apply_horizontal_turbulence` → surface reflection.
7. **Add `engine.apply_horizontal_turbulence`** primitive.
8. **CLI surface** — `--turbulence-scheme` flag wired through `_build_config`, with `LPDM_TURBULENCE_SCHEME` env var mirror. Default stays `placeholder_constant_ou` until validation passes; flip default to `hanna_1982` and update README at the end of M1.
9. **Validation per §5**, then update `VALIDATION.md` placeholder list.

## 5. Validation plan

### 5.1 Unit (no met dependency)

- `PlaceholderConstantOU` reproduces `tests/test_physics.py::test_zero_wind_diffusion_langevin_gaussian_spread` bit-for-bit (regression).
- Hanna formula tests: pin σ_u, σ_v, σ_w, T_Lu, T_Lv, T_Lw outputs against literature reference values for a few `(z/h, h/L, u*, w*)` triples in each regime.
- Stability classification: pin regime boundaries for sweep across `h/L`.
- New horizontal turbulence well-mixed test analogous to `test_well_mixed_uniformity_in_periodic_turbulence`.

### 5.2 End-to-end (synthetic met)

- Extend `AnalyticMetReader` to emit `ustar` and `shf` (constant per run, with stability-controllable inputs).
- Constant-stability runs: verify mean trajectory still matches analytic backward transport (advection unchanged).
- Vertical spread: re-baseline the placeholder dispersion metrics from `VALIDATION.md` and document the new tolerances.
- Convective regime: verify particles released near the surface eventually mix throughout the BL within a few `T_Lw`.
- Stable regime: verify near-surface particles stay confined.

### 5.3 External reference

- One canonical FLEXPART comparison case: stable BL point release, identical met (shared ECMWF source, not ARCO ERA5 — pick a date with a published FLEXPART intercomparison or generate our own reference). Targets are mean-position match (within advection-only tolerance) and plume-spread agreement at the σ ±15% level (analogous to the existing OU test tolerance).

`VALIDATION.md` placeholder list shrinks accordingly: endpoint spread, time-height structure, footprint extent, and FLEXPART/STILT comparison all migrate from "placeholder" to "first-class baseline" sections.

## 6. Open questions / known limitations

- Above-BL turbulence uses a gradient-Richardson closure (§3.2.3), not a constant-K placeholder.
- The legacy MO surface-layer override (`surface_layer_override: true`) and the pre-floor 1 s T_L minimum (`flexpart_tl_floors: false`) are retained only for A/B comparisons (§3.2.2/§3.2.4); the defaults follow FLEXPART v11. Remove the legacy paths once the v2-vs-v3 validation confirms the defaults.
- The in-BL σ/T_L coefficients and T_L floors follow FLEXPART v11 (§3.2.2). Strong σ-gradients (e.g. through the BL top) are carried by the well-mixed drift (§3.2.5) with σ re-evaluated each step.
- Coriolis `f` is computed from each particle's instantaneous latitude (using `|f|`); we don't average across the trajectory.
- `T_v` is approximated as `T` for the Obukhov-length calculation; humidity correction is small and deferred.
- Initial perturbation velocities are zero rather than sampled from the local σ; particles equilibrate within `T_L`.

## 7. References

- Hanna, S. R. (1982). Applications in air pollution modeling. In *Atmospheric Turbulence and Air Pollution Modelling*, Reidel.
- Stohl, A., Hittenberger, M., Wotawa, G. (1998). Validation of the Lagrangian particle dispersion model FLEXPART. *Atmos. Environ.* 32, 4245–4264.
- Stohl, A., Forster, C., Frank, A., Seibert, P., Wotawa, G. (2005). Technical note: The Lagrangian particle dispersion model FLEXPART version 6.2. *Atmos. Chem. Phys.* 5, 2461–2474.
- Lin, J. C., Gerbig, C., Wofsy, S. C., Andrews, A. E., Daube, B. C., Davis, K. J., Grainger, C. A. (2003). A near-field tool for simulating the upstream influence of atmospheric observations: STILT. *J. Geophys. Res.* 108, 4493.
- Thomson, D. J. (1987). Criteria for the selection of stochastic models of particle trajectories in turbulent flows. *J. Fluid Mech.* 180, 529–556.
- Wilson, J. D., Sawford, B. L. (1996). Review of Lagrangian stochastic models for trajectories in the turbulent atmosphere. *Boundary-Layer Meteorology* 78, 191–210.
