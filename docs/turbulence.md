# GLIDE turbulence parameterization

Specification for the M1 turbulence rewrite: scope, modular architecture, scheme math, met-input contract, and validation plan. This document is the source of truth for the turbulence design; implementation should match it.

## 1. Goals and scope

- Replace the M0 placeholder constant-OU vertical scheme with met-driven Hanna 1982 turbulence (the FLEXPART formulation), in three stability regimes.
- Add horizontal stochastic diffusion (currently missing entirely).
- Make scheme selection modular — alternative schemes (Wilson-Sawford, hybrid Hanna-Degrazia, future ML schemes) should land as plug-in subclasses without touching the runtime loop.
- Above-BL turbulence is in scope from day 1 (simple constant-diffusivity placeholder; future refinement noted as M1.x).
- Backward integration uses FLEXPART-style piecewise-homogeneous treatment (no explicit Thomson 1987 drift term).
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

- CLI flag `--turbulence-scheme=<name>` (default during M1 transition: `placeholder_constant_ou`; switch default to `hanna_1982` once validated).
- Programmatic: pass a `TurbulenceScheme` instance directly to `_run(cfg, *, reader, scheme=...)`. This is what tests use.

Environment variable `LPDM_TURBULENCE_SCHEME` mirrors the CLI flag. Future M4 YAML config will own this surface; for M1 the CLI flag is sufficient.

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

Stable BL:
```
σ_w = 1.3 u* (1 - z/h)^(3/4)
σ_u = σ_v = 2.0 u* (1 - z/h)^(3/4)
T_Lw = 0.10 h / σ_w · (z/h)^(0.8)
T_Lu = T_Lv = 0.15 h / σ_u · (z/h)^(0.5)
```

Neutral BL (with Coriolis `f` at the particle's latitude):
```
σ_w = 1.3 u* exp(-2 f z / u*)
σ_u = σ_v = 2.0 u* exp(-2 f z / u*)
T_Lw = 0.5 z / σ_w / (1 + 15 f z / u*)
T_Lu = T_Lv = T_Lw / 1.5
```

Unstable / convective BL:
```
σ_w² = 1.5 u*² (1 - 0.95 z/h)^(2/3) + 1.6 w*² (z/h)^(2/3) (1 - z/h)^2
σ_u² = σ_v² = (12 - 0.5 h/L)^(2/3) u*²
T_Lw = 0.15 h / σ_w
T_Lu = T_Lv = 0.15 h / σ_u
```

Constants reproduced from Hanna 1982 (Table 1 / Eqs. 11–24) and FLEXPART's manual sections 4.3.1–4.3.3. **Implementation MUST cross-check exact constants against the FLEXPART source tree before merging**, since secondary references occasionally diverge by ~10% on minor coefficients.

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
treated as isotropic (`σ_u = σ_v = σ_w`) for now; the unresolved-mesoscale
"meander" horizontal term (NAME-style, resolution-dependent) is a separate,
deferred item. Implemented as free functions `potential_temperature`,
`brunt_vaisala_squared`, `gradient_richardson`, `free_trop_diffusivity`,
`free_trop_sigma_TL` in `hanna.py`.

#### 3.2.4 Surface-layer treatment

When `z < z_sl` (surface-layer top, taken as `0.1 h` in FLEXPART), use Monin-Obukhov surface-layer scaling rather than the full in-BL formulae:

```
σ_w = 1.3 u* (1 - 2 z/L)^(1/3)   (unstable)
σ_w = 1.3 u* (1 + 5 z/L)         (stable, capped to 1.3 u* · 6 in very stable)
σ_u = σ_v as in §3.2.2 with z → max(z, z_sl)
T_L  = κ z / σ
```

Implementation will match FLEXPART `mod_par_var_pbl.f90` / `hanna.f90` (or equivalent) so the surface layer behaves consistently with the reference.

#### 3.2.5 Drift handling — Thomson well-mixed term

**Added 2026-05-29** (the original piecewise-homogeneous no-drift treatment
under-dispersed the surface footprint badly — particles drifted up the σ_w
gradient, accumulated above the BL, and stopped recycling to the surface).

The vertical OU update now carries the Thomson (1987) well-mixed drift:

```
a_drift = ½ (1 + w'²/σ_w²) ∂σ_w²/∂z
```

`∂σ_w²/∂z` is a central finite difference of the *full* column σ_w profile
(in-BL → surface-layer → free-troposphere), so it spans the regime transitions.
`engine.update_langevin_velocity` takes a `drift` argument and applies it as a
forward-Euler increment on top of the exact-OU homogeneous part (which still
preserves the stationary variance σ²). The increment is capped at one σ_w per
step so the sharp σ_w kink at the BL top can't blow the velocity up.

**Backward-Langevin sign.** GLIDE runs backward in time; the displacement
negates w'·dt. The random forcing is symmetric so that's harmless, but the drift
is *deterministic* and must enter with reversed sign for the adjoint/backward
Langevin (Thomson 1987 §5 reciprocity; Flesch, Wilson & Yee 1995) so the
*physical* position drift still points down-gradient (toward the BL). Getting
this sign wrong inverts the correction into a one-way upward pump — empirically
it lofted the entire MHD population to ~2 km and held it there. The scheme
pre-negates the drift before the backward displacement; covered by
`test_hanna_well_mixed_no_runaway_lofting` and the engine-level
`test_well_mixed_condition_drift_keeps_uniform_distribution`.

Horizontal components (u', v') do not carry a drift — the inhomogeneity is in z,
and FLEXPART applies the well-mixed correction to the vertical only.

#### 3.2.6 State

`{"w_prime": (N,), "u_prime": (N,), "v_prime": (N,)}`.

Initial state: zeros. A strictly correct Thomson formulation would initialise from the local σ distribution; this is a deliberate simplification because particles equilibrate to the local σ within ~`T_L` (typically 100 s) and the release window itself is usually longer.

#### 3.2.7 Time-step constraint

The exact-OU update is unconditionally stable, but accuracy degrades when `dt > T_L / 5` (standard rule-of-thumb). The runtime currently uses `dt = 300 s`; near-surface unstable `T_Lw` can drop to 10–30 s. M1 adds a runtime warning when `min(T_L) < 5 dt`, with a recommendation to reduce `--dt-seconds`. We do not auto-substep.

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

- Above-BL constant-K is a placeholder; M1.x can refine using N² and Ri.
- Surface-layer transition altitude `z_sl = 0.1 h` is a heuristic and may need tuning per regime.
- We use FLEXPART's piecewise-homogeneous treatment, not Thomson 1987 explicit drift. Strong σ-gradients (e.g. through the BL top) are captured only by re-evaluating σ each step.
- Coriolis `f` is computed from each particle's instantaneous latitude; we don't average across the trajectory.
- `T_v` is approximated as `T` for the Obukhov-length calculation; humidity correction is small and deferred.
- Initial perturbation velocities are zero rather than sampled from the local σ; particles equilibrate within `T_L`.

## 7. References

- Hanna, S. R. (1982). Applications in air pollution modeling. In *Atmospheric Turbulence and Air Pollution Modelling*, Reidel.
- Stohl, A., Hittenberger, M., Wotawa, G. (1998). Validation of the Lagrangian particle dispersion model FLEXPART. *Atmos. Environ.* 32, 4245–4264.
- Stohl, A., Forster, C., Frank, A., Seibert, P., Wotawa, G. (2005). Technical note: The Lagrangian particle dispersion model FLEXPART version 6.2. *Atmos. Chem. Phys.* 5, 2461–2474.
- Lin, J. C., Gerbig, C., Wofsy, S. C., Andrews, A. E., Daube, B. C., Davis, K. J., Grainger, C. A. (2003). A near-field tool for simulating the upstream influence of atmospheric observations: STILT. *J. Geophys. Res.* 108, 4493.
- Thomson, D. J. (1987). Criteria for the selection of stochastic models of particle trajectories in turbulent flows. *J. Fluid Mech.* 180, 529–556.
- Wilson, J. D., Sawford, B. L. (1996). Review of Lagrangian stochastic models for trajectories in the turbulent atmosphere. *Boundary-Layer Meteorology* 78, 191–210.
