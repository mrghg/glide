"""Hanna 1982 / FLEXPART turbulence scheme.

Implementation per `docs/turbulence.md`. Three stability regimes (stable,
neutral, unstable) within the boundary layer, carrying the full Thomson (1987)
well-mixed drift + Stohl-Thomson (1999) density correction (§3.2.5), a
Monin-Obukhov surface-layer override for `z < 0.1 h` (§3.2.4), and a
gradient-Richardson closure above the BL (§3.2.3).

The in-BL σ/T_L parameterisation implements the equations of Hanna (1982),
Caughey (1982) and Ryall & Maryon (1998) — see the per-regime functions for
the specific equation attributions. This is the same set of parameterisations
FLEXPART uses in `src/turbulence_mod.f90` `subroutine hanna`; GLIDE adopts them
(coefficients checked against that reference on 2026-07-02, Finding 5 of the
physics review) so its boundary-layer turbulence is directly comparable to
FLEXPART's. By default GLIDE also applies FLEXPART's Lagrangian-timescale floors
(10/10/30 s; ``flexpart_tl_floors``) and, like FLEXPART, runs the regime formulas
to the ground with no surface-layer override (``surface_layer_override=False``);
the legacy behaviour is retained behind those two flags for A/B comparisons.
See `docs/turbulence.md` §4 (BL profiles) / §8 (surface-layer override).
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from datetime import datetime
from typing import ClassVar

import numpy as np
import torch

from lpdm.gpu_engine import (
    GPUEngine,
    GridInterpolationBounds,
    frame_grid_bounds,
    trilinear_sample,
    use_static_step_path,
)
from lpdm.local_frame import HorizontalFrame
from lpdm.met_reader import HourlyMetTensors
from lpdm.turbulence.base import TurbulenceScheme, TurbulenceState, register_scheme

# Escape hatch for the perf-#1 static-address marking (GH200 diagnosis
# 2026-07-17: marking collapsed the DtoD staging copies as intended, but the
# smoke profile showed per-step cudagraph RE-RECORDING — set
# GLIDE_MARK_STATIC_INPUTS=0 to keep the reusable buffers but skip
# `mark_static_address`, isolating which half causes the re-record).
_MARK_STATIC_INPUTS = os.environ.get("GLIDE_MARK_STATIC_INPUTS", "1") != "0"

LOGGER = logging.getLogger(__name__)


# Physical constants
KARMAN = 0.4
GRAVITY_M_S2 = 9.80665
R_DRY_AIR_J_KG_K = 287.05
C_P_DRY_AIR_J_KG_K = 1005.0
EARTH_ROTATION_RATE_S = 7.2921e-5  # sidereal angular velocity (rad/s)

# Stability regime thresholds on h/L
H_OVER_L_STABLE_THRESHOLD = 1.0
H_OVER_L_UNSTABLE_THRESHOLD = -1.0

# Above-BL placeholder constants (retained for reference / fallback).
ABOVE_BL_SIGMA_M_S = 0.1
ABOVE_BL_T_L_S = 100.0

# Surface-layer fraction of BLH (FLEXPART default)
SURFACE_LAYER_FRACTION = 0.1
# Logged once per run, and recorded in run_metadata.json, when the opt-in
# surface-layer override is on (decision 0028, "Other boundaries").
SURFACE_LAYER_OVERRIDE_WARNING = (
    "turbulence.surface_layer_override is on: this legacy Monin-Obukhov treatment below "
    "0.1 h is kept for A/B comparisons only. sigma_w jumps at the 0.1 h seam and the "
    "well-mixed drift's finite difference straddles it, the defect decision 0028 removed "
    "at h, so the model is not well-mixed consistent at 0.1 h. Do not use it for "
    "production footprints (docs/turbulence.md section 8)."
)

# Numerical floors to avoid degenerate values at z=0 / very low ustar / etc.
SIGMA_MIN_M_S = 1e-3
T_L_MIN_S = 1.0
USTAR_MIN_M_S = 1e-3
# Default limits on the boundary-layer depth h (docs/turbulence.md §2; decision 0024).
# The scheme holds h between `blh_min_m` and `blh_max_m`, set from the
# configuration (`turbulence.boundary_layer_depth_min_m` / `_max_m`) at
# construction. The defaults keep GLIDE's long-standing 50 m floor and no ceiling;
# the validation runs set FLEXPART's and NAME's 100 m and 4,500 m.
BLH_MIN_M_DEFAULT = 50.0
BLH_MAX_M_DEFAULT: float | None = None
Z_MIN_M = 1.0  # height floor (metres) for near-surface z in gradient/timescale lookups

# FLEXPART v11 Lagrangian-timescale floors (turbulence_mod.f90 subroutine hanna:
# tlu=max(10.,tlu); tlv=max(10.,tlv); tlw=max(30.,tlw)). These keep the
# near-surface vertical diffusivity K = σ_w²·T_Lw from collapsing as z → 0
# (T_Lw formulas vanish with z), which otherwise traps particles in the lowest
# metres and inflates near-field surface footprints — the GLIDE-vs-NAME/FLEXPART
# over-estimation seen at polluted low-inlet sites (2026-07-02 v2 validation).
# Applied when HannaScheme(flexpart_tl_floors=True), the default.
T_LU_FLOOR_FLEXPART_S = 10.0
T_LV_FLOOR_FLEXPART_S = 10.0
T_LW_FLOOR_FLEXPART_S = 30.0

# Free-troposphere (above-BL) gradient-Richardson closure constants (docs/turbulence.md §6).
P0_PA = 100000.0  # reference pressure for potential temperature
KAPPA_POISSON = R_DRY_AIR_J_KG_K / C_P_DRY_AIR_J_KG_K  # ~0.2854
FT_MIXING_LENGTH_ASYMPTOTE_M = 100.0  # Blackadar asymptotic mixing length (free troposphere)
FT_RICHARDSON_CRIT = 0.25  # critical gradient Richardson number
FT_KZ_FLOOR_M2_S = 0.1  # background diffusivity so FT particles are never fully frozen
FT_KZ_CEIL_M2_S = 50.0  # cap on shear-driven FT diffusivity
FT_SHEAR_SQ_FLOOR_S2 = 1e-8  # floor on |dU/dz|^2 to avoid div-by-zero in Ri
FT_N2_FLOOR_S2 = 1e-6  # floor on N^2 for the buoyancy timescale
FT_T_L_DEFAULT_S = 100.0  # fallback Lagrangian timescale where N^2 <= 0
FT_T_L_MAX_S = 1000.0  # cap on the buoyancy-derived FT timescale

# Meander (unresolved-mesoscale) horizontal turbulence — Maryon (1998)
# "meandering" as adopted by FLEXPART (Stohl et al. 2005 §4.5). σ_meander is the
# local grid-wind variability times a coefficient; the timescale is ~half the met
# field interval. See docs/turbulence.md §7.
MEANDER_COEFFICIENT_DEFAULT = 0.16  # FLEXPART `turbmesoscale`
MEANDER_STENCIL_RADIUS_DEFAULT = 1  # 3x3 neighbourhood
MEANDER_TIMESCALE_S_DEFAULT = 1800.0  # half of the hourly ERA5 field interval

# Sub-stepping (decision 0027, docs/physics.md §6). Every particle takes the same
# fixed number of equal sub-steps per outer step, and every coefficient (σ, T_L,
# ∂σ_w²/∂z, density drift) is re-evaluated at the particle's current height at
# the start of each sub-step. A sub-step length chosen per particle from its T_Lw
# (the old adaptive schedule) is itself a spurious drift towards where steps are
# short — the ground — so the count is a configured constant, never data-dependent.
N_SUBSTEPS_DEFAULT = 16
# Reporting threshold only: a particle-sub-step with sub_dt > EXCEEDANCE_C · T_Lw
# (T_Lw at that sub-step's height) is counted in run_metadata.json
# (`substep_exceedance`), split boundary layer / free troposphere, together with
# those with sub_dt > T_Lw. It does not change the integration. 0.5 is the old
# adaptive scheme's target ratio.
EXCEEDANCE_C_DEFAULT = 0.5
# The order of the counts the sub-step loop returns (`_integrate_vertical_substeps`).
EXCEEDANCE_COUNT_KEYS = (
    "substeps_bl",
    "exceeded_bl",
    "over_T_L_bl",
    "substeps_ft",
    "exceeded_ft",
    "over_T_L_ft",
)

# F5/F15 — "unresolved basal layer" (W&F 1993 §7b). Hold σ_w, T_L (and the
# density-gradient piece) CONSTANT for any particle below this height by
# evaluating the turbulence profile at z=max(z_particle, Z_UBL_DEFAULT). The
# constant-σ basal layer makes smooth-wall reflection WMC-exact (W&F §7b) and
# also caps the |∂σ²/∂z| spike at the very-near-surface that drives the
# F4 Tier 2 substep cap to bind for some particles. Set to a small value
# (default 2 m) so the layer is thinner than any practical particle release
# altitude — it only intercepts the post-reflection bounce. Particle position
# is NOT clamped; only the σ/T_L *sampling* uses the clamped z.
Z_UBL_DEFAULT_M = 2.0


# Rogue-trajectory safeguard for the F4 Tier 2 substep loop. Spec §T:
# "Watch for unbounded velocities when Δt is too large relative to rapidly
# varying statistics. Check for velocity caps." When the substep cap binds
# (sub_dt > T_L) and a particle's σ_w drops near the floor (e.g. at the BL
# top), the (1 + w'²/σ_w²) drift factor can blow up. We clip |w'/σ| at a
# physically generous 4× (≈ 6e-5 probability for a true Gaussian) at the end
# of each substep OU update. FLEXPART has the same kind of clip in its OU
# implementation (see flexpart.f90 in their turbulence routines).
W_PRIME_SIGMA_RATIO_MAX = 4.0

# Orientation of the horizontal turbulent velocities (decision 0032,
# docs/physics.md §3.4). Hanna's σ_u, T_Lu are along the mean wind and σ_v, T_Lv
# across it. "grid" applies u' eastward and v' northward (GLIDE before 0032);
# "wind" runs the OU updates on along-wind and cross-wind components and rotates
# them to east/north for the displacement, as FLEXPART's `windalign` does.
HORIZONTAL_FRAMES = ("grid", "wind")
HORIZONTAL_FRAME_DEFAULT = "grid"
# Calm limit for the "wind" frame: a particle whose mean horizontal wind speed is
# at most this multiple of its along-wind σ_u keeps the grid axes for the step.
# Below it the turbulent fluctuations exceed the mean wind and the mean wind
# direction does not define an along-wind axis (decision 0032, "Calm winds").
WIND_FRAME_CALM_RATIO = 1.0

# Boundary-layer top interface (decision 0028, docs/physics.md §6). A particle
# transmitted into, or reflected back into, the free troposphere is placed at
# least this far above h, so that rounding never leaves it at z = h, which
# counts as boundary layer. 1 cm is far below any resolved scale and well above
# the float32 spacing at h ≤ 80 km.
INTERFACE_SIDE_EPS_M = 1e-2


# ---------------------------------------------------------------------------
# Free physics functions (testable in isolation)
# ---------------------------------------------------------------------------


def coriolis_parameter(latitude_deg: torch.Tensor) -> torch.Tensor:
    """Coriolis parameter `f = 2 Ω sin(lat)` per particle. Sign-correct in both hemispheres."""

    return 2.0 * EARTH_ROTATION_RATE_S * torch.sin(torch.deg2rad(latitude_deg))


def air_density(sp_pa: torch.Tensor, t_kelvin: torch.Tensor) -> torch.Tensor:
    """Dry-air density `ρ = sp / (R_d T)`."""

    return sp_pa / (R_DRY_AIR_J_KG_K * t_kelvin)


def obukhov_length(
    ustar: torch.Tensor,
    shf: torch.Tensor,
    t_surface: torch.Tensor,
    sp: torch.Tensor,
) -> torch.Tensor:
    """Obukhov length `L = -u*³ T_v ρ c_p / (κ g H)` per particle.

    Sign convention: `H` is the upward sensible heat flux (W/m²). Sign of L:
    `L > 0` stable, `L < 0` unstable, `|L| → ∞` neutral. Returns `+∞` where
    `|H|` is below the numerical threshold.
    """

    rho = air_density(sp, t_surface)
    denom = KARMAN * GRAVITY_M_S2 * shf
    num = -ustar.clamp(min=USTAR_MIN_M_S).pow(3) * t_surface * rho * C_P_DRY_AIR_J_KG_K

    finite = denom.abs() > 1e-10
    safe_denom = torch.where(finite, denom, torch.ones_like(denom))
    return torch.where(finite, num / safe_denom, torch.full_like(num, float("inf")))


def convective_velocity(
    blh: torch.Tensor,
    shf: torch.Tensor,
    t_surface: torch.Tensor,
    sp: torch.Tensor,
    *,
    blh_min_m: float = BLH_MIN_M_DEFAULT,
) -> torch.Tensor:
    """Convective velocity scale `w* = ((g h H) / (T ρ c_p))^(1/3)`.

    Returns 0 where `H ≤ 0` (stable / neutral), since `w*` is undefined there.
    `blh_min_m` floors `h`; the scheme passes its own configured minimum (its `h`
    is already limited, so the floor is then a no-op).
    """

    rho = air_density(sp, t_surface)
    cube = (GRAVITY_M_S2 * blh.clamp(min=blh_min_m) * shf) / (t_surface * rho * C_P_DRY_AIR_J_KG_K)
    return torch.where(cube > 0, cube.clamp(min=0.0).pow(1.0 / 3.0), torch.zeros_like(cube))


# σ_w of each regime on its own. The full regime functions below use these same
# expressions, so the σ_w-only evaluation (`in_bl_sigma_w`, used for the drift's
# finite-difference probes) is bit-identical to σ_w from `in_bl_sigma_TL`.


def _stable_sigma_w(z_over_h: torch.Tensor, ustar: torch.Tensor) -> torch.Tensor:
    """Hanna (1982) Eq 7.19: σ_w = 1.3 u*(1−ζ)."""
    one_minus = (1.0 - z_over_h).clamp(min=0.0)
    return 1.3 * ustar * one_minus


def _neutral_sigma_w(
    z: torch.Tensor, ustar: torch.Tensor, f_abs: torch.Tensor, ustar_safe: torch.Tensor
) -> torch.Tensor:
    """Hanna (1982) Eq 7.26: σ_w = 1.3 u* exp(−2 f z/u*)."""
    return 1.3 * ustar * torch.exp(-2.0 * f_abs * z / ustar_safe)


def _unstable_sigma_w_sq(
    z_over_h: torch.Tensor, ustar: torch.Tensor, w_star: torch.Tensor
) -> torch.Tensor:
    """Ryall & Maryon (1998) σ_w² before the floor: 1.2 w*²(1−0.9ζ)ζ^(2/3) + (1.8−1.4ζ)u*²."""
    return 1.2 * w_star.pow(2) * (1.0 - 0.9 * z_over_h).clamp(min=0.0) * z_over_h.pow(2.0 / 3.0) + (
        1.8 - 1.4 * z_over_h
    ) * ustar.pow(2)


def _unstable_sigma_w(
    z_over_h: torch.Tensor, ustar: torch.Tensor, w_star: torch.Tensor
) -> torch.Tensor:
    """Ryall & Maryon (1998): σ_w = √(1.2 w*²(1−0.9ζ)ζ^(2/3) + (1.8−1.4ζ)u*²)."""
    return _unstable_sigma_w_sq(z_over_h, ustar, w_star).clamp(min=SIGMA_MIN_M_S**2).sqrt()


def _in_bl_stable(
    z: torch.Tensor,
    blh: torch.Tensor,
    ustar: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """(σ_u, σ_v, σ_w, T_Lu, T_Lv, T_Lw) under stable BL.

    Equations from Hanna (1982) Eqs 7.19-7.24. This is the same stable-regime
    parameterisation FLEXPART uses (`turbulence_mod.f90` `subroutine hanna`),
    adopted so GLIDE is directly comparable; the coefficients trace to Hanna."""

    z_over_h = (z / blh).clamp(min=0.0, max=1.0)
    one_minus = (1.0 - z_over_h).clamp(min=0.0)

    # Eq 7.20 / 7.19: σ_u = 2.0 u*(1−ζ); σ_v = σ_w = 1.3 u*(1−ζ). LINEAR in (1−ζ),
    # not the (1−ζ)^¾ of the previous secondary-reference form. σ_v tracks σ_w (1.3),
    # not σ_u (2.0).
    sigma_u = 2.0 * ustar * one_minus
    sigma_w = _stable_sigma_w(z_over_h, ustar)
    sigma_v = sigma_w
    # Small-z floor so ζ^0.8 / √ζ → 0 doesn't force T_L = 0 exactly at the surface.
    z_over_h_floor = z_over_h.clamp(min=1e-6)
    # Eq 7.22 T_Lu = 0.15 h/σ_u·√ζ; Eq 7.23 T_Lv = 0.07 h/σ_v·√ζ; Eq 7.24 T_Lw = 0.10 h/σ_w·ζ^0.8.
    # Each component divides by its OWN σ. Writing T_Lv = 0.467·T_Lu reproduces the
    # coefficient ratio (0.07/0.15 = 0.467) but leaves it divided by σ_u instead of σ_v,
    # and here σ_u = 2.0 u*(1−ζ) while σ_v = 1.3 u*(1−ζ) — so that form came out short by
    # σ_v/σ_u = 0.65 at every height (fixed 2026-08-17).
    T_Lu = 0.15 * blh / sigma_u.clamp(min=SIGMA_MIN_M_S) * z_over_h_floor.pow(0.5)
    T_Lv = 0.07 * blh / sigma_v.clamp(min=SIGMA_MIN_M_S) * z_over_h_floor.pow(0.5)
    T_Lw = 0.10 * blh / sigma_w.clamp(min=SIGMA_MIN_M_S) * z_over_h_floor.pow(0.8)
    return sigma_u, sigma_v, sigma_w, T_Lu, T_Lv, T_Lw


def _in_bl_neutral(
    z: torch.Tensor,
    ustar: torch.Tensor,
    f_cor: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """(σ_u, σ_v, σ_w, T_Lu, T_Lv, T_Lw) under neutral BL.

    Equations from Hanna (1982) Eqs 7.25-7.27 (the same neutral-regime forms
    FLEXPART uses in `subroutine hanna`; coefficients trace to Hanna).

    Uses the per-particle |f| (Finding 4): FLEXPART hardcodes f = 1e-4 s⁻¹, but a
    real per-latitude Coriolis is strictly better provided we take the ABSOLUTE
    value — otherwise the Ekman-decay exponents flip sign in the Southern
    Hemisphere and σ grows (rather than decays) with height.
    """

    ustar_safe = ustar.clamp(min=USTAR_MIN_M_S)
    f_abs = f_cor.abs().clamp(min=1e-5)  # |f|, floored so the equator doesn't degenerate
    # Eq 7.25 σ_u = 2.0 u* exp(−3 f z/u*); Eq 7.26 σ_v = σ_w = 1.3 u* exp(−2 f z/u*).
    sigma_u = 2.0 * ustar * torch.exp(-3.0 * f_abs * z / ustar_safe)
    sigma_w = _neutral_sigma_w(z, ustar, f_abs, ustar_safe)
    sigma_v = sigma_w
    # Eq 7.27 T_L = 0.5 z/σ_w/(1 + 15 f z/u*), same for all three components.
    denom = 1.0 + 15.0 * f_abs * z / ustar_safe
    T_Lw = 0.5 * z / sigma_w.clamp(min=SIGMA_MIN_M_S) / denom.clamp(min=1.0)
    T_Lu = T_Lw
    T_Lv = T_Lw
    return sigma_u, sigma_v, sigma_w, T_Lu, T_Lv, T_Lw


def _in_bl_unstable(
    z: torch.Tensor,
    blh: torch.Tensor,
    ustar: torch.Tensor,
    w_star: torch.Tensor,
    h_over_L: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """(σ_u, σ_v, σ_w, T_Lu, T_Lv, T_Lw) under unstable / convective BL.

    Equations: σ_u from Caughey (1982) Eq 4.15; σ_w from Ryall & Maryon (1998);
    T_Lu from Hanna (1982) Eq 7.17; the piecewise T_Lw follows the convective
    surface-layer / mixed-layer scalings in Hanna (1982). This matches the
    unstable-regime parameterisation FLEXPART uses (`subroutine hanna`), adopted
    for comparability."""

    z_over_h = (z / blh).clamp(min=0.0, max=1.0)

    # σ_u = σ_v = u*(12 − 0.5 h/L)^(1/3)  [Caughey 1982 Eq 4.15].
    h_over_L_capped = h_over_L.clamp(min=-1000.0, max=0.0)
    sigma_u = (12.0 - 0.5 * h_over_L_capped).clamp(min=1e-3).pow(1.0 / 3.0) * ustar
    sigma_v = sigma_u

    # σ_w = √(1.2 w*²(1−0.9ζ)ζ^(2/3) + (1.8−1.4ζ)u*²)  [Ryall & Maryon 1998].
    sigma_w = _unstable_sigma_w(z_over_h, ustar, w_star)

    # T_Lu = T_Lv = 0.15 h/σ_u  [Eq 7.17].
    T_Lu = 0.15 * blh / sigma_u.clamp(min=SIGMA_MIN_M_S)
    T_Lv = T_Lu
    # T_Lw is piecewise in z/L (FLEXPART): near-surface, shallow, and bulk branches.
    # |z/L| = ζ·|h/L| (h_over_L < 0 here).
    abs_z_over_L = (z_over_h * (-h_over_L_capped)).clamp(min=0.0)
    sw = sigma_w.clamp(min=SIGMA_MIN_M_S)
    t_near = 0.1 * z / (sw * (0.55 - 0.38 * abs_z_over_L).clamp(min=1e-2))
    t_shallow = 0.59 * z / sw
    t_bulk = 0.15 * blh / sw * (1.0 - torch.exp(-5.0 * z_over_h))
    T_Lw = torch.where(
        abs_z_over_L < 1.0,
        t_near,
        torch.where(z_over_h < 0.1, t_shallow, t_bulk),
    )
    return sigma_u, sigma_v, sigma_w, T_Lu, T_Lv, T_Lw


def in_bl_sigma_TL(
    z: torch.Tensor,
    blh: torch.Tensor,
    ustar: torch.Tensor,
    w_star: torch.Tensor,
    h_over_L: torch.Tensor,
    latitude_deg: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-particle (σ_u, σ_v, σ_w, T_Lu, T_Lv, T_Lw) inside the BL.

    Combines the three stability regimes (FLEXPART v11 `subroutine hanna`) via
    `torch.where`. σ_u ≠ σ_v in general (σ_v tracks σ_w), and the three T_L
    components differ per regime. All inputs must be per-particle tensors
    broadcasting to a common shape.
    """

    f_cor = coriolis_parameter(latitude_deg)

    u_st, v_st, w_st, tu_st, tv_st, tw_st = _in_bl_stable(z, blh, ustar)
    u_nu, v_nu, w_nu, tu_nu, tv_nu, tw_nu = _in_bl_neutral(z, ustar, f_cor)
    u_un, v_un, w_un, tu_un, tv_un, tw_un = _in_bl_unstable(z, blh, ustar, w_star, h_over_L)

    in_stable = h_over_L > H_OVER_L_STABLE_THRESHOLD
    in_unstable = h_over_L < H_OVER_L_UNSTABLE_THRESHOLD

    def _select(
        stable_v: torch.Tensor, neutral_v: torch.Tensor, unstable_v: torch.Tensor
    ) -> torch.Tensor:
        return torch.where(in_stable, stable_v, torch.where(in_unstable, unstable_v, neutral_v))

    sigma_u = _select(u_st, u_nu, u_un)
    sigma_v = _select(v_st, v_nu, v_un)
    sigma_w = _select(w_st, w_nu, w_un)
    T_Lu = _select(tu_st, tu_nu, tu_un)
    T_Lv = _select(tv_st, tv_nu, tv_un)
    T_Lw = _select(tw_st, tw_nu, tw_un)
    return sigma_u, sigma_v, sigma_w, T_Lu, T_Lv, T_Lw


def in_bl_sigma_w(
    z: torch.Tensor,
    blh: torch.Tensor,
    ustar: torch.Tensor,
    w_star: torch.Tensor,
    h_over_L: torch.Tensor,
    latitude_deg: torch.Tensor,
) -> torch.Tensor:
    """σ_w alone inside the BL: bit-identical to ``in_bl_sigma_TL(...)[2]`` (the
    regime functions share the σ_w expressions) at a fraction of the cost. Used
    for the well-mixed drift's finite-difference probes."""

    f_abs = coriolis_parameter(latitude_deg).abs().clamp(min=1e-5)
    ustar_safe = ustar.clamp(min=USTAR_MIN_M_S)
    z_over_h = (z / blh).clamp(min=0.0, max=1.0)
    w_st = _stable_sigma_w(z_over_h, ustar)
    w_nu = _neutral_sigma_w(z, ustar, f_abs, ustar_safe)
    w_un = _unstable_sigma_w(z_over_h, ustar, w_star)
    in_stable = h_over_L > H_OVER_L_STABLE_THRESHOLD
    in_unstable = h_over_L < H_OVER_L_UNSTABLE_THRESHOLD
    return torch.where(in_stable, w_st, torch.where(in_unstable, w_un, w_nu))


def in_bl_dsigw2_dz(
    z: torch.Tensor,
    blh: torch.Tensor,
    ustar: torch.Tensor,
    w_star: torch.Tensor,
    h_over_L: torch.Tensor,
    latitude_deg: torch.Tensor,
) -> torch.Tensor:
    """Analytic ∂σ_w²/∂z of the floored boundary-layer profile
    σ_w = max(σ_H, SIGMA_MIN_M_S) of ``in_bl_sigma_w``, for 0 < z ≤ h: the
    derivative of the regime's σ_H² where σ_H is above the floor, and zero where the
    floor binds (σ_w is flat there). Decision 0030.

    stable  σ_H = 1.3u*(1−ζ):          ∂σ²/∂z = −2·1.3u*·σ_H/h
    neutral σ_H = 1.3u*exp(−2|f|z/u*): ∂σ²/∂z = −4|f|σ_H²/u*
    unstable σ_H² = 1.2w*²(1−0.9ζ)ζ^⅔ + (1.8−1.4ζ)u*²:
        ∂σ²/∂z = [1.2w*²(−0.9ζ^⅔ + ⅔(1−0.9ζ)ζ^−⅓) − 1.4u*²]/h
    (the unstable form holds for ζ ≤ 1/0.9, i.e. everywhere in the boundary layer).
    SIGMA_MIN_M_S is read at call time."""

    f_abs = coriolis_parameter(latitude_deg).abs().clamp(min=1e-5)
    ustar_safe = ustar.clamp(min=USTAR_MIN_M_S)
    z_over_h = (z / blh).clamp(min=1e-6, max=1.0)
    w_st = _stable_sigma_w(z_over_h, ustar)
    g_st = -2.0 * 1.3 * ustar * w_st / blh
    w_nu = _neutral_sigma_w(z, ustar, f_abs, ustar_safe)
    g_nu = -4.0 * f_abs * w_nu.pow(2) / ustar_safe
    w2_un = _unstable_sigma_w_sq(z_over_h, ustar, w_star)
    g_un = (
        1.2
        * w_star.pow(2)
        * (
            -0.9 * z_over_h.pow(2.0 / 3.0)
            + (2.0 / 3.0) * (1.0 - 0.9 * z_over_h) * z_over_h.pow(-1.0 / 3.0)
        )
        - 1.4 * ustar.pow(2)
    ) / blh
    in_stable = h_over_L > H_OVER_L_STABLE_THRESHOLD
    in_unstable = h_over_L < H_OVER_L_UNSTABLE_THRESHOLD
    grad = torch.where(in_stable, g_st, torch.where(in_unstable, g_un, g_nu))
    sigma_sq = torch.where(in_stable, w_st.pow(2), torch.where(in_unstable, w2_un, w_nu.pow(2)))
    return torch.where(sigma_sq > SIGMA_MIN_M_S**2, grad, torch.zeros_like(grad))


def potential_temperature(t_kelvin: torch.Tensor, pressure_pa: torch.Tensor) -> torch.Tensor:
    """Potential temperature `θ = T (p0/p)^κ`, κ = R_d/c_p."""

    return t_kelvin * (P0_PA / pressure_pa).pow(KAPPA_POISSON)


def brunt_vaisala_squared(theta: torch.Tensor, dtheta_dz: torch.Tensor) -> torch.Tensor:
    """Brunt-Väisälä frequency squared `N² = (g/θ) ∂θ/∂z`. Sign follows ∂θ/∂z
    (positive = stable stratification)."""

    return (GRAVITY_M_S2 / theta.clamp(min=1.0)) * dtheta_dz


def gradient_richardson(n2: torch.Tensor, shear_sq: torch.Tensor) -> torch.Tensor:
    """Gradient Richardson number `Ri = N² / |∂U/∂z|²` with a shear floor."""

    return n2 / shear_sq.clamp(min=FT_SHEAR_SQ_FLOOR_S2)


def free_trop_diffusivity(
    z: torch.Tensor,
    shear_mag: torch.Tensor,
    ri: torch.Tensor,
) -> torch.Tensor:
    """Free-troposphere vertical diffusivity via a first-order gradient-Richardson
    closure: `K_z = l² |∂U/∂z| f(Ri)`.

    * `l = κz / (1 + κz/λ)` — Blackadar mixing length, asymptote λ.
    * `f(Ri)` — stability function: `(1 - Ri/Ri_c)²` for `0 ≤ Ri < Ri_c`,
      `√(1 - 16 Ri)` for `Ri < 0` (unstable, rare in the FT), and 0 for
      `Ri ≥ Ri_c` (sub-critical → laminar). A small background floor keeps `K_z`
      positive everywhere so lofted particles are never fully frozen (the failure
      mode of the old σ=0.1 placeholder).
    """

    mixing_length = (KARMAN * z) / (1.0 + KARMAN * z / FT_MIXING_LENGTH_ASYMPTOTE_M)

    f_stable = (1.0 - ri / FT_RICHARDSON_CRIT).clamp(min=0.0).pow(2)
    f_unstable = (1.0 - 16.0 * ri).clamp(min=0.0).sqrt()
    f_ri = torch.where(ri < 0.0, f_unstable, f_stable)

    k_z = mixing_length.pow(2) * shear_mag * f_ri
    return k_z.clamp(min=FT_KZ_FLOOR_M2_S, max=FT_KZ_CEIL_M2_S)


def free_trop_sigma_TL(
    k_z: torch.Tensor,
    n2: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split a free-troposphere diffusivity into `(σ_w, T_Lw)` with `K_z = σ_w² T_Lw`.

    The Lagrangian timescale is taken as the buoyancy timescale `~1/N` (bounded),
    since stratification sets the eddy decorrelation in the stable FT; where
    `N² ≤ 0` a fixed fallback timescale is used. `σ_w = √(K_z / T_Lw)`.
    """

    t_lw = torch.where(
        n2 > FT_N2_FLOOR_S2,
        (0.5 / n2.clamp(min=FT_N2_FLOOR_S2).sqrt()).clamp(min=T_L_MIN_S, max=FT_T_L_MAX_S),
        torch.full_like(n2, FT_T_L_DEFAULT_S),
    )
    sigma_w = (k_z / t_lw).clamp(min=SIGMA_MIN_M_S**2).sqrt()
    return sigma_w, t_lw


def surface_layer_sigma_w(z: torch.Tensor, ustar: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
    """σ_w in the surface layer, Monin-Obukhov scaling (docs/turbulence.md §8).

    Stable:    `1.3 u*` (constant). σ_w/u* ≈ 1.3 is roughly height-independent in
               the stable surface layer (Flesch, Wilson & Yee 1995 App. B). The
               previous `1.3 u*(1 + 5 z/L)` form GREW with stability (up to a 6×
               cap = 7.8 u*, larger than convective values) — that is the φ_m
               momentum-gradient function, NOT a σ_w scaling, and inflated
               nocturnal near-surface mixing (Finding 6 of the 2026-07-02 review).
    Unstable:  `1.3 u* (1 - 3 z/L)^(1/3)`  (Flesch et al. 1995 App. B; z/L < 0).
    Neutral:   `1.3 u*` (z/L → 0).
    """

    eps = 1e-6
    finite_L = L.abs() > eps
    L_safe = torch.where(finite_L, L, torch.ones_like(L))
    z_over_L = torch.where(finite_L, (z / L_safe).clamp(-50.0, 50.0), torch.zeros_like(L))

    sigma_unstable = 1.3 * ustar * (1.0 - 3.0 * z_over_L).clamp(min=eps).pow(1.0 / 3.0)
    sigma_stable_or_neutral = 1.3 * ustar

    return torch.where(z_over_L < 0.0, sigma_unstable, sigma_stable_or_neutral)


# ---------------------------------------------------------------------------
# Scheme class
# ---------------------------------------------------------------------------


@register_scheme
class HannaScheme(TurbulenceScheme):
    """Hanna 1982 / FLEXPART turbulence scheme. See `docs/turbulence.md`."""

    name: ClassVar[str] = "hanna_1982"

    def __init__(
        self,
        *,
        meander_enabled: bool = False,
        meander_coefficient: float = MEANDER_COEFFICIENT_DEFAULT,
        meander_stencil_radius: int = MEANDER_STENCIL_RADIUS_DEFAULT,
        meander_timescale_seconds: float | None = None,
        blh_min_m: float = BLH_MIN_M_DEFAULT,
        blh_max_m: float | None = BLH_MAX_M_DEFAULT,
        n_substeps: int = N_SUBSTEPS_DEFAULT,
        exceedance_c: float = EXCEEDANCE_C_DEFAULT,
        z_ubl_m: float = Z_UBL_DEFAULT_M,
        basal_zero_gradient: bool = False,
        static_substeps: bool | None = None,
        flexpart_tl_floors: bool = True,
        surface_layer_override: bool = False,
        horizontal_frame: str = HORIZONTAL_FRAME_DEFAULT,
    ) -> None:
        """Construct the scheme.

        ``horizontal_frame`` (decision 0032): ``"grid"`` (default) applies u' to
        east and v' to north, with σ_u, T_Lu on u' and σ_v, T_Lv on v'. ``"wind"``
        treats u' and v' as the along-wind and cross-wind components inside each
        outer step and rotates them to east/north by the mean wind direction at the
        particle for every horizontal displacement; the carried state stays east/north.

        Meander (unresolved-mesoscale horizontal turbulence) is off by default so
        existing runs stay bit-identical; enable it via the YAML ``turbulence.meander``
        block. See ``docs/turbulence.md`` §7.

        ``n_substeps`` is the number of equal sub-steps of the OU + displacement +
        reflection that EVERY particle takes in each outer step (decision 0027,
        docs/physics.md §6); all coefficients are re-evaluated at the particle's
        current height at the start of each sub-step after the first. The count is a
        constant of the run, never chosen per particle or per batch: a sub-step
        length that depends on the particle's height biases the well-mixed state
        towards the ground. ``exceedance_c`` is a reporting threshold only: sub-steps
        with ``sub_dt > exceedance_c · T_Lw`` are counted (``take_substep_exceedance``).
        Meander has τ ≈ 1800 s so always runs at the outer dt.

        ``basal_zero_gradient``: below ``z_ubl_m`` every coefficient is sampled at
        ``z_ubl_m``, so the profile the particle sees is flat there. When True, the
        well-mixed drift's ``∂σ_w²/∂z`` is zero strictly below ``z_ubl_m``,
        consistent with that flat profile; when False, the finite-difference
        stencil, clamped to ``[z_ubl, z_ubl + δ]``, returns the one-sided gradient
        just above the clamp there. The density drift is the same either way.

        ``flexpart_tl_floors`` (default True) applies FLEXPART v11's Lagrangian-
        timescale floors (T_Lu, T_Lv ≥ 10 s; T_Lw ≥ 30 s) after the profile
        assembly. Without them the near-surface T_Lw → 0 with z, collapsing the
        vertical diffusivity K = σ_w²·T_Lw in the lowest metres and inflating
        near-field surface footprints (the GLIDE ≫ NAME/FLEXPART bias at polluted
        low-inlet sites, 2026-07-02 v2 validation). Set False for the legacy 1 s
        floor (A/B comparisons).

        ``surface_layer_override`` (default False) enables the GLIDE-only
        Monin-Obukhov surface-layer treatment below 0.1 h (σ_w from MO scaling,
        T_L = κz/σ). FLEXPART v11 has no such override — its regime formulas plus
        the T_L floors run to the ground — and the override both undercuts the
        floors (κz/σ → seconds near the surface) and introduces a K discontinuity
        at the 0.1 h seam. Kept as an opt-in for A/B comparisons. Floors are
        applied AFTER the override, so enabling both still bounds T_L.

        ``static_substeps`` selects the substep-loop implementation (docs/architecture.md
        §5). ``None`` (default) auto-selects by device: the **static-shape** variant
        on CUDA (every substep processes the full particle buffer with ``sub_dt=0``
        for inactive particles — constant kernel shapes, the prerequisite for
        ``torch.compile`` fusion and CUDA-graph capture on a launch-bound GPU), and
        the **dynamic masked** variant on CPU/MPS (touches only the active
        particles — cheaper where launches are free). ``True``/``False`` force one
        path (used by the equivalence tests). Env override: ``GLIDE_STATIC_SUBSTEPS``.
        The two paths run the same sub-step loop and are bit-identical when every
        particle is active (with inactive particles present the random draws differ).
        """

        if meander_coefficient <= 0:
            raise ValueError("meander_coefficient must be > 0")
        if meander_stencil_radius < 1:
            raise ValueError("meander_stencil_radius must be >= 1")
        if meander_timescale_seconds is not None and meander_timescale_seconds <= 0:
            raise ValueError("meander_timescale_seconds must be > 0")
        if not blh_min_m > 0:
            raise ValueError("blh_min_m must be > 0")
        if blh_max_m is not None and not blh_max_m > blh_min_m:
            raise ValueError("blh_max_m must be > blh_min_m")
        if int(n_substeps) != n_substeps or n_substeps < 1:
            raise ValueError("n_substeps must be an integer >= 1")
        if not exceedance_c > 0:
            raise ValueError("exceedance_c must be > 0")
        if z_ubl_m < 0:
            raise ValueError("z_ubl_m must be >= 0")
        if horizontal_frame not in HORIZONTAL_FRAMES:
            raise ValueError(f"horizontal_frame must be one of {HORIZONTAL_FRAMES}")
        # A construction-time constant: the compiled step specialises on it once.
        self.horizontal_frame = str(horizontal_frame)
        self.z_ubl_m = float(z_ubl_m)
        self.meander_enabled = bool(meander_enabled)
        self.meander_coefficient = float(meander_coefficient)
        self.meander_stencil_radius = int(meander_stencil_radius)
        # Unset (None) => half the meteorology interval, bound by the runtime from
        # the store before the first step (`bind_met_interval`); until then the
        # hourly-ERA5 value. Explicit values are never overridden. Docs §7.
        self._meander_timescale_explicit = meander_timescale_seconds is not None
        self.meander_timescale_seconds = float(
            meander_timescale_seconds
            if meander_timescale_seconds is not None
            else MEANDER_TIMESCALE_S_DEFAULT
        )
        # Boundary-layer depth limits: construction-time Python constants, so the
        # compiled step specialises on them once (decision 0004), never per step.
        self.blh_min_m = float(blh_min_m)
        self.blh_max_m = None if blh_max_m is None else float(blh_max_m)
        # A construction-time constant: the compiled step's loop trip count
        # (decision 0004), and the same for every particle and step (decision 0027).
        self.n_substeps = int(n_substeps)
        self.exceedance_c = float(exceedance_c)
        # Zero ∂σ_w²/∂z strictly below the basal clamp (`_vertical_coefficients`).
        self.basal_zero_gradient = bool(basal_zero_gradient)
        self.flexpart_tl_floors = bool(flexpart_tl_floors)
        self.surface_layer_override = bool(surface_layer_override)
        self._static_substeps = static_substeps
        # Device-side running totals of the sub-step exceedance counter (decision
        # 0027): active particle-sub-steps in the order of `EXCEEDANCE_COUNT_KEYS`
        # (int64) and the worst sub_dt/T_Lw `[BL, FT]`. Accumulated inside the step
        # with no host sync, on both paths; the runtime reads and resets them once
        # per met window (`take_substep_exceedance`). Allocated on first use.
        self._exceed_counts: torch.Tensor | None = None
        self._exceed_worst: torch.Tensor | None = None
        # Time direction. GLIDE runs backward in time; True integrates the same
        # equations FORWARD in time (the drift of docs/physics.md §3.3 unnegated,
        # displacements and advection +U·dt; reflection and the boundary-layer-top
        # interface unchanged). It exists only for the forward–backward
        # reciprocity test (tests/test_reciprocity.py, task A6) and is deliberately
        # not a constructor argument, so no configuration can set it.
        self._forward_in_time = False

        # CUDA-graph capture (phase 4): lazily-built `torch.compile(mode="reduce-overhead")`
        # wrapper of the WHOLE per-step core (`_step_core`). Built on first use on the
        # static+compile path; `_graph_compile_state` is none/ok/failed.
        self._compiled_graph_core: Callable[..., tuple] | None = None
        self._graph_compile_state: str = "none"

        # Per-met-window cache for the grid-wide derived fields (meander σ, density
        # + ∂ρ/∂z, free-trop σ/T_L). These depend only on the hourly met window,
        # but `step` is called every dt (~60 steps/window), so recomputing the
        # avg_pool2d / vertical-gradient stacks per step was ~18% of runtime
        # (profiled 2026-06-18). We instead build each stack at the window's two
        # time endpoints once per window and time-interpolate per step. See
        # `_cached_window_field`. One entry per field key (the current window).
        self._window_field_cache: dict[str, tuple[datetime, torch.Tensor]] = {}

        # Per-window cache of the RAW met already moved to the device: the surface
        # hour-start/end slices, the 3D u/v/w wind, and the level array. These are
        # constant within a met window, so caching them removes a redundant
        # host->device copy (~100 MB of 3D wind + the surface slices) AND a device
        # allocation EVERY step (~60/window) — the latter was a driver of the
        # caching-allocator fragmentation (dev_reserved >> dev_alloc). `_gather_static_inputs`
        # then only does the cheap per-step time interpolation of the surface fields.
        self._raw_met_cache: tuple[datetime, dict] | None = None
        # Persistent, address-stable buffers for the compiled-core inputs (perf
        # review 2026-07-16 #1). Cudagraph replay copies every input whose ADDRESS
        # changed since capture into the graph's staging buffers — and the fresh
        # `.to(device)` allocations each met window made the big window-constant
        # tensors (winds, ft/density/meander stacks) recopy EVERY STEP (~49% of
        # GPU time in the 2026-06-26 profile). These buffers are allocated once,
        # marked `torch._dynamo.mark_static_address` on CUDA, and refilled in
        # place (per window for window-constant inputs; per step for the small
        # surface lerps and alpha) — all outside the captured region.
        self._static_input_buffers: dict[str, torch.Tensor] = {}

    def required_met_keys(self) -> tuple[str, ...]:
        # Baseline (u, v, w, blh, sp) is added by the runtime; declare scheme-specific extras.
        return ("t", "ustar", "shf")

    def bind_met_interval(self, met_interval_seconds: float) -> None:
        """Set the meander timescale to half the met interval, unless configured.

        Called by the runtime once, after the reader has measured the store's time
        spacing and before the first step, so the value is a constant of the run
        (and of the compiled step). On hourly ERA5 it gives the long-standing 1800 s.
        """

        if met_interval_seconds <= 0:
            raise ValueError("met_interval_seconds must be > 0")
        if not self._meander_timescale_explicit:
            self.meander_timescale_seconds = 0.5 * float(met_interval_seconds)

    def _limit_blh(self, blh: torch.Tensor) -> torch.Tensor:
        """Hold the boundary-layer depth within the configured limits (docs §2)."""

        return blh.clamp(min=self.blh_min_m, max=self.blh_max_m)

    def initialize_state(
        self,
        n_particles: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> TurbulenceState:
        state = {
            "u_prime": torch.zeros(n_particles, device=device, dtype=dtype),
            "v_prime": torch.zeros(n_particles, device=device, dtype=dtype),
            "w_prime": torch.zeros(n_particles, device=device, dtype=dtype),
            # (σ_u, σ_v, σ_w) at the particle's position at the end of its last
            # step, with that step's meteorology; 0 before its first step
            # (decision 0028, `_carry_velocities`).
            "sigma_last": torch.zeros(n_particles, 3, device=device, dtype=dtype),
        }
        if self.meander_enabled:
            state["u_meander"] = torch.zeros(n_particles, device=device, dtype=dtype)
            state["v_meander"] = torch.zeros(n_particles, device=device, dtype=dtype)
        return state

    # Finite-difference step (m) for the well-mixed drift's ∂σ_w²/∂z.
    DRIFT_FD_DELTA_M: ClassVar[float] = 1.0
    # PRIVATE, for A/B tests only (no configuration sets it): True restores the
    # ±δ stencil for every particle, i.e. the drift gradient before decision 0030.
    _fd_drift_gradient_only: bool = False
    # PRIVATE, for A/B tests only (no configuration sets it): with the "wind" frame,
    # True carries u', v' from step to step as along-wind and cross-wind components,
    # unrotated when the wind turns, as FLEXPART does; False (production) carries
    # them as east/north components. Neither is exact everywhere (decision 0032,
    # "A turning wind"): the wind-relative carry keeps the wind-frame distribution
    # through a turn in time, but where the direction varies in space a particle's
    # own turbulent motion turns its velocity, which makes calm points sources of
    # order 2π·K·c even in isotropic turbulence (measured: a convective vortex
    # gained 7,000 particles at its centre in an hour, against 0 for the east/north
    # carry and for "grid"); the east/north carry is exact for isotropic turbulence
    # and leaves a covariance (σ_u² − σ_v²) sinΔ cosΔ after a turn Δ in time,
    # relaxing over T_L.
    _carry_wind_relative: bool = False

    def step(
        self,
        particles: torch.Tensor,
        state: TurbulenceState,
        met_window: HourlyMetTensors,
        t_alpha: float,
        dt_seconds: float,
        active_mask: torch.Tensor,
        engine: GPUEngine,
    ) -> tuple[torch.Tensor, TurbulenceState]:
        device = engine.device
        dtype = engine.dtype

        # Two execution paths (docs/architecture.md S5), chosen by device gate.
        if self._use_static_substeps(engine):
            # STATIC / compilable path. The whole per-step tensor pipeline lives in
            # `_step_core`; only the met-field *access* (Python) is hoisted out into
            # `_gather_static_inputs`. The core is ONE torch.compile target -> on CUDA
            # it captures as a single graph, collapsing the ~1000+ eager kernel launches
            # per step that left the GPU launch-bound (profile 2026-06-26). Inactive
            # particles are frozen by the mask (sub_dt=0 in the loop + a where at the end).
            inputs = self._gather_static_inputs(
                met_window,
                t_alpha,
                device,
                dtype,
                frame=engine.frame,
                wind_dtype=engine.wind_dtype,
            )
            graph_core = self._maybe_graph_compile(engine)
            core_fn = graph_core if graph_core is not None else self._step_core
            if graph_core is not None:
                # Required before each replay of a CUDAGraph-captured function: tells
                # the framework that a new invocation is beginning so it doesn't treat
                # the previous run's output buffers as live inputs for the current step.
                torch.compiler.cudagraph_mark_step_begin()
            particles, u_p, v_p, w_p, s_last, u_m, v_m, exceed_counts, exceed_worst = core_fn(
                particles=particles,
                u_prime=state["u_prime"],
                v_prime=state["v_prime"],
                w_prime=state["w_prime"],
                sigma_last=state["sigma_last"],
                u_meander=state.get("u_meander"),
                v_meander=state.get("v_meander"),
                active_mask=active_mask,
                dt_seconds=dt_seconds,
                engine=engine,
                **inputs,
            )
            # Read into the scheme's own accumulators now, before the next replay
            # reuses the graph's output buffers (a device-side add, no host sync).
            self._accumulate_exceedance(exceed_counts, exceed_worst)
            # Clone graph output buffers before storing them back as state / returning
            # particles. Without this, the CUDAGraph output memory is aliased with the
            # inputs of the next invocation, triggering "overwritten by a subsequent
            # run" errors even after cudagraph_mark_step_begin().
            state["u_prime"] = u_p.clone()
            state["v_prime"] = v_p.clone()
            state["w_prime"] = w_p.clone()
            state["sigma_last"] = s_last.clone()
            if self.meander_enabled:
                state["u_meander"] = u_m.clone()
                state["v_meander"] = v_m.clone()
            return particles.clone(), state

        # DYNAMIC / boolean-indexed path (CPU): touch only the active subset; the empty
        # guard (bool(torch.any(...)), a host sync) is cheap here. Byte-identical to the
        # pre-phase-4 dynamic path.
        if not bool(torch.any(active_mask)):
            return particles, state

        active_xyz = particles[active_mask]
        u_prime_in = state["u_prime"][active_mask]
        v_prime_in = state["v_prime"][active_mask]
        w_prime_in = state["w_prime"][active_mask]
        # Horizontal position in the engine's frame (`lpdm.local_frame`); latitude
        # in degrees for the Coriolis parameter.
        frame = engine.frame
        x = active_xyz[:, 0]
        y = active_xyz[:, 1]
        lat_deg = frame.lat_from_y(y)
        z_agl = active_xyz[:, 2]

        def surface(name: str) -> torch.Tensor:
            return self._interp_surface_field(
                met_window, name, t_alpha, x, y, device, dtype, frame=frame
            )

        blh = self._limit_blh(surface("blh"))
        sp = surface("sp")
        ustar = surface("ustar").clamp(min=USTAR_MIN_M_S)
        shf = surface("shf")
        t_surface = self._interp_t_at_lowest_level(
            met_window, t_alpha, x, y, device, dtype, frame=frame
        )

        L = obukhov_length(ustar, shf, t_surface, sp)
        h_over_L = blh / L
        w_star = convective_velocity(blh, shf, t_surface, sp, blh_min_m=self.blh_min_m)

        ft_fields, grid_bounds = self._free_trop_fields(met_window, device, dtype, frame)
        density_fields = self._density_fields(met_window, device, dtype)
        col_kwargs = dict(
            blh=blh,
            ustar=ustar,
            w_star=w_star,
            h_over_L=h_over_L,
            L=L,
            lat=lat_deg,
            x=x,
            y=y,
            ft_fields=ft_fields,
            grid_bounds=grid_bounds,
            engine=engine,
        )
        z_eval = z_agl.clamp(min=self.z_ubl_m)

        def coeffs_at(z: torch.Tensor) -> tuple[torch.Tensor, ...]:
            return self._vertical_coefficients(z, density_fields, **col_kwargs)

        (
            sigma_u,
            sigma_v,
            sigma_w,
            sigma_w_sq,
            T_Lu,
            T_Lv,
            T_Lw,
            half_dsig2_dz_backward,
            density_drift_backward,
        ) = coeffs_at(z_agl)
        wind_axes = None
        if self.horizontal_frame == "wind":
            # The u, v stacks in the wind-gather dtype, as the static path holds them.
            uv_start, uv_end = met_window.channels("u", "v")
            wind_axes = self._wind_axes(
                uv_start.to(device=device, dtype=engine.wind_dtype),
                uv_end.to(device=device, dtype=engine.wind_dtype),
                t_alpha,
                sigma_u,
                x=x,
                y=y,
                z=z_eval,
                grid_bounds=grid_bounds,
                engine=engine,
            )
        u_prime_in, v_prime_in = self._grid_to_wind(
            u_prime_in, v_prime_in, self._state_axes(wind_axes)
        )
        u_prime_in, v_prime_in, w_prime_in = self._carry_velocities(
            u_prime_in,
            v_prime_in,
            w_prime_in,
            state["sigma_last"][active_mask],
            sigma_u,
            sigma_v,
            sigma_w,
        )

        (
            moved,
            u_prime_active,
            v_prime_active,
            w_prime_active,
            exceed_counts,
            exceed_worst,
        ) = self._integrate_vertical_substeps(
            active_xyz=active_xyz,
            u_prime_in=u_prime_in,
            v_prime_in=v_prime_in,
            w_prime_in=w_prime_in,
            sigma_u=sigma_u,
            sigma_v=sigma_v,
            sigma_w=sigma_w,
            sigma_w_sq=sigma_w_sq,
            T_Lu=T_Lu,
            T_Lv=T_Lv,
            T_Lw=T_Lw,
            half_dsig2_dz_backward=half_dsig2_dz_backward,
            density_drift_backward=density_drift_backward,
            coeffs_at=coeffs_at,
            blh=blh,
            dt_seconds=dt_seconds,
            engine=engine,
            interface_sigmas=self._interface_sigmas(**col_kwargs),
            wind_axes=wind_axes,
        )
        u_prime_active, v_prime_active = self._wind_to_grid(
            u_prime_active, v_prime_active, self._state_axes(wind_axes)
        )
        self._accumulate_exceedance(exceed_counts, exceed_worst)
        sigma_last_active = self._sigmas_at(moved[:, 2], **col_kwargs)

        if self.meander_enabled:
            meander_fields, meander_bounds = self._meander_sigma_fields(
                met_window, device, dtype, frame
            )
            sm = self._interp_3d_field(meander_fields, meander_bounds, x, y, z_eval, engine)
            u_meander_active = engine.update_langevin_velocity(
                state["u_meander"][active_mask],
                t_lagrangian=self.meander_timescale_seconds,
                sigma_w2=sm[0].pow(2),
                dt_seconds=dt_seconds,
            )
            v_meander_active = engine.update_langevin_velocity(
                state["v_meander"][active_mask],
                t_lagrangian=self.meander_timescale_seconds,
                sigma_w2=sm[1].pow(2),
                dt_seconds=dt_seconds,
            )
            moved = engine.apply_horizontal_turbulence(
                moved,
                u_meander_active,
                v_meander_active,
                dt_seconds=dt_seconds,
                backward=not self._forward_in_time,
            )

        particles[active_mask] = moved
        state["u_prime"][active_mask] = u_prime_active
        state["v_prime"][active_mask] = v_prime_active
        state["w_prime"][active_mask] = w_prime_active
        state["sigma_last"][active_mask] = sigma_last_active
        if self.meander_enabled:
            state["u_meander"][active_mask] = u_meander_active
            state["v_meander"][active_mask] = v_meander_active
        return particles, state

    def _vertical_coefficients(
        self,
        z_agl: torch.Tensor,
        density_fields: torch.Tensor,
        **col_kwargs,
    ) -> tuple[torch.Tensor, ...]:
        """All turbulence coefficients at height ``z_agl`` (sampled at the basal-layer
        clamp ``max(z, z_ubl)``): ``(σ_u, σ_v, σ_w, σ_w², T_Lu, T_Lv, T_Lw,
        −½∂σ_w²/∂z, −(σ_w²/ρ)∂ρ/∂z)``, the last two pre-negated for the backward
        drift. Called once at the start of the outer step and again at the start of
        every later sub-step at the particle's current height (docs/physics.md §6)."""

        z_eval = z_agl.clamp(min=self.z_ubl_m)
        sigma_u, sigma_v, sigma_w, T_Lu, T_Lv, T_Lw = self._column_turbulence(z_eval, **col_kwargs)

        # ∂σ_w²/∂z by finite difference; the probes need σ_w only. The stencil
        # stays on the particle's own side of the boundary-layer top h (decision
        # 0028): a boundary-layer particle's probes lie in [z_ubl, h], a
        # free-troposphere particle's in [h, ∞), each evaluated with its own
        # side's profile (at z = h too). A stencil that straddled h would
        # difference the jump between the boundary-layer profile and the
        # free-troposphere closure into a spurious drift of order
        # (σ_BL² − σ_FT²)/(2δ); the jump is handled by the interface rule in the
        # sub-step loop instead. More than δ from h this is the centred stencil.
        blh = col_kwargs["blh"]
        side_ft = z_eval > blh
        z_hi = z_eval + self.DRIFT_FD_DELTA_M
        z_lo = (z_eval - self.DRIFT_FD_DELTA_M).clamp(min=self.z_ubl_m)
        z_hi = torch.where(side_ft, z_hi, torch.minimum(z_hi, blh))
        z_lo = torch.where(side_ft, torch.maximum(z_lo, blh), z_lo)
        sw_hi = self._column_sigma_w(z_hi, side_ft=side_ft, **col_kwargs)
        sw_lo = self._column_sigma_w(z_lo, side_ft=side_ft, **col_kwargs)
        dsig2_dz = (sw_hi.pow(2) - sw_lo.pow(2)) / (z_hi - z_lo).clamp(min=self.DRIFT_FD_DELTA_M)
        if not (self.surface_layer_override or self._fd_drift_gradient_only):
            # Decision 0030: the stencil is wrong for a boundary-layer particle
            # whose stencil (i) is cut at h, where it is one-sided and returns the
            # gradient up to δ/2 below the particle (an O(1) error where a stable
            # σ_w → 0 linearly at h), or (ii) touches the floor SIGMA_MIN_M_S,
            # where it differences across the kink of max(σ_H, σ_min) and puts a
            # drift inside the floor layer, where σ_w is flat. Those particles take
            # the analytic gradient of the floored profile (zero where the floor
            # binds); every other particle keeps the stencil, bit for bit. A
            # data-dependent select, no host sync, the same on both step paths.
            # With the surface-layer override the stencil is used everywhere (the
            # analytic form does not include the override's profile).
            g_bl = in_bl_dsigw2_dz(
                z_eval,
                blh,
                col_kwargs["ustar"],
                col_kwargs["w_star"],
                col_kwargs["h_over_L"],
                col_kwargs["lat"],
            )
            use_analytic = (~side_ft) & (
                (z_eval + self.DRIFT_FD_DELTA_M > blh)
                | (sw_hi <= SIGMA_MIN_M_S)
                | (sw_lo <= SIGMA_MIN_M_S)
                | (sigma_w <= SIGMA_MIN_M_S)
            )
            dsig2_dz = torch.where(use_analytic, g_bl, dsig2_dz)
        if self.basal_zero_gradient:
            # Below the basal clamp σ_w is sampled at z_ubl, i.e. flat in z, so its
            # gradient there is zero; the stencil would otherwise look one-sidedly
            # upward from the clamp. The density drift is unchanged.
            dsig2_dz = torch.where(z_agl < self.z_ubl_m, torch.zeros_like(dsig2_dz), dsig2_dz)
        sigma_w_sq = sigma_w.pow(2).clamp(min=SIGMA_MIN_M_S**2)

        density_at_p = self._interp_3d_field(
            density_fields,
            col_kwargs["grid_bounds"],
            col_kwargs["x"],
            col_kwargs["y"],
            z_eval,
            col_kwargs["engine"],
            level_arr=col_kwargs.get("level_arr"),
            ascending=col_kwargs.get("ascending"),
        )
        rho_at_p = density_at_p[0].clamp(min=1e-3)
        density_drift_backward = -(sigma_w_sq * density_at_p[1] / rho_at_p)
        half_dsig2_dz_backward = -0.5 * dsig2_dz
        if self._forward_in_time:
            # Forward time (reciprocity test only): the forward drift, unnegated.
            density_drift_backward = -density_drift_backward
            half_dsig2_dz_backward = -half_dsig2_dz_backward
        return (
            sigma_u,
            sigma_v,
            sigma_w,
            sigma_w_sq,
            T_Lu,
            T_Lv,
            T_Lw,
            half_dsig2_dz_backward,
            density_drift_backward,
        )

    def _cached_raw_device_met(
        self, met_window, device, dtype, wind_dtype: torch.dtype | None = None
    ) -> dict:
        """Raw met moved to the device ONCE per window (keyed by window start time).

        Returns the surface hour-start/end slices (as (start, end) pairs, for the
        per-step time interpolation), the raw 3D u/v/w wind (hour-start/end), the
        per-level AGL ``level_arr`` and the ``ascending`` flag. All constant within a
        met window, so this replaces the per-step host->device copy + allocation that
        ``_gather_static_inputs`` used to do every step. See ``_raw_met_cache``.

        The wind stacks are held in ``wind_dtype`` (``simulation.wind_gather_dtype``,
        task A4): float32, or bfloat16 for the 16-bit trilinear gather of
        ``_advect_rk2``.
        """

        wind_dtype = dtype if wind_dtype is None else wind_dtype
        ts = met_window.metadata.time_start
        key = (ts, wind_dtype)
        entry = self._raw_met_cache
        if entry is not None and entry[0] == key:
            return entry[1]

        def _surf_pair(name, lev):
            cs, ce = met_window.channel(name)
            return (cs[lev].to(device=device, dtype=dtype), ce[lev].to(device=device, dtype=dtype))

        lowest_t = int(np.argmin(met_window.metadata.level))
        m_start_uvw, m_end_uvw = met_window.channels("u", "v", "w")
        level = met_window.metadata.level
        bundle = dict(
            # The surface (start, end) pairs feed the per-step lerp OUTSIDE the
            # graph, so their addresses don't matter — no buffers needed.
            blh=_surf_pair("blh", 0),
            sp=_surf_pair("sp", 0),
            ustar=_surf_pair("ustar", 0),
            shf=_surf_pair("shf", 0),
            t=_surf_pair("t", lowest_t),
            # The wind stacks and level array enter the compiled core directly:
            # static buffers, refilled once per window (perf #1).
            m_start_uvw=self._to_static_buffer(
                "met.m_start_uvw", m_start_uvw.to(device=device, dtype=wind_dtype)
            ),
            m_end_uvw=self._to_static_buffer(
                "met.m_end_uvw", m_end_uvw.to(device=device, dtype=wind_dtype)
            ),
            # per-level AGL heights — the F9 vertical-lookup axis; a dynamic
            # TENSOR so dynamo doesn't specialise the graph on the per-window values.
            # (Under the terrain-following coordinate the values are a run constant,
            # but the legacy pressure-grid path still varies per window.)
            level_arr=self._to_static_buffer(
                "met.level_arr", torch.as_tensor(np.asarray(level), device=device, dtype=dtype)
            ),
            # pressure-level ORDER is genuinely constant, so it stays a plain bool.
            ascending=bool(level[-1] > level[0]),
        )
        self._raw_met_cache = (key, bundle)
        return bundle

    def _gather_static_inputs(
        self,
        met_window,
        t_alpha,
        device,
        dtype,
        *,
        frame: HorizontalFrame,
        wind_dtype: torch.dtype | None = None,
    ):
        """Assemble the per-step pure-tensor inputs for `_step_core` (the torch.compile
        target). The raw met + derived stacks are cached per window; only the cheap
        per-step time interpolation of the surface fields and the ``alpha`` scalar are
        rebuilt each step. Behaviour-preserving vs the pre-cache path (identical math).
        ``frame`` is the engine's horizontal frame (the grid bounds are expressed in
        it) and ``wind_dtype`` the dtype of the wind stacks (task A4)."""
        raw = self._cached_raw_device_met(met_window, device, dtype, wind_dtype)

        def _interp(name, pair):
            # Per-step surface lerp, written INTO a stable buffer (perf #1): the
            # result is a compiled-core input, and a fresh tensor each step would
            # be restaged by cudagraph on every replay. `torch.lerp(..., out=)`
            # runs in eager, outside the captured region.
            # A mean/accumulation channel (decision 0020) is held at its window-end
            # value, the mean over exactly this window: weight 1.0, not alpha.
            s, e = pair
            buf = self._to_static_buffer(f"surf.{name}", s, copy_src=False)
            held = name in getattr(met_window, "period_channels", ())
            torch.lerp(s, e, 1.0 if held else float(t_alpha), out=buf)
            return buf

        ft_fields, grid_bounds = self._free_trop_fields(met_window, device, dtype, frame)
        meander_fields = None
        if self.meander_enabled:
            meander_fields, _ = self._meander_sigma_fields(met_window, device, dtype, frame)

        # alpha (the RK2-midpoint time-interp weight) changes EVERY step. Pass it as
        # a 0-d TENSOR, not a Python float: dynamo specializes the compiled graph on
        # float values, so a per-step float would recompile every step, blow the
        # recompile limit, and fall back to eager (GH200 profile 2026-06-26 — the run
        # went 4× slower than before, GPU-busy 0.7%). A tensor is a dynamic input —
        # held in a stable 0-d buffer and refilled per step (perf #1).
        alpha_buf = self._static_input_buffers.get("alpha")
        if (
            alpha_buf is None
            or not self._device_matches(alpha_buf.device, device)
            or alpha_buf.dtype != dtype
        ):
            alpha_buf = torch.zeros((), device=device, dtype=dtype)
            if device.type == "cuda" and _MARK_STATIC_INPUTS:
                torch._dynamo.mark_static_address(alpha_buf)
            self._static_input_buffers["alpha"] = alpha_buf
        alpha_buf.fill_(float(t_alpha))

        return dict(
            blh_field=_interp("blh", raw["blh"]),
            sp_field=_interp("sp", raw["sp"]),
            ustar_field=_interp("ustar", raw["ustar"]),
            shf_field=_interp("shf", raw["shf"]),
            t_field=_interp("t", raw["t"]),
            ft_fields=ft_fields,
            density_fields=self._density_fields(met_window, device, dtype),
            meander_fields=meander_fields,
            grid_bounds=grid_bounds,
            m_start_uvw=raw["m_start_uvw"],
            m_end_uvw=raw["m_end_uvw"],
            alpha=alpha_buf,
            level_arr=raw["level_arr"],
            ascending=raw["ascending"],
        )

    def _advect_rk2(
        self,
        particles,
        m_start_uvw,
        m_end_uvw,
        alpha,
        grid_bounds,
        dt_seconds,
        engine,
        level_arr=None,
        ascending=None,
    ):
        """RK2 backward advection on the full particle buffer (folded into `_step_core`
        so the whole step captures as one graph). Bit-identical to the pre-fold
        `main._advect_active_particles` wind_fn + `engine.rk2_advect_backward`.
        Positions are in the engine's frame (`lpdm.local_frame`); the wind stacks
        are float32 (``grid_sample``) or 16-bit (``trilinear_sample``, task A4)."""
        frame = engine.frame
        m_start = m_start_uvw.unsqueeze(0)
        m_end = m_end_uvw.unsqueeze(0)
        sample_16bit = m_start_uvw.dtype.itemsize == 2

        def wind(xyz: torch.Tensor) -> torch.Tensor:
            xyz_norm = engine.normalize_particle_coordinates(
                xyz, grid_bounds, level_arr=level_arr, ascending=ascending
            )
            if sample_16bit:
                s, e = trilinear_sample((m_start_uvw, m_end_uvw), xyz_norm)
                v_start, v_end = s.t(), e.t()
            else:
                grid = xyz_norm.view(1, 1, 1, -1, 3)
                v_start = (
                    torch.nn.functional.grid_sample(m_start, grid, align_corners=True)
                    .view(3, -1)
                    .t()
                )
                v_end = (
                    torch.nn.functional.grid_sample(m_end, grid, align_corners=True).view(3, -1).t()
                )
            v = v_start * (1.0 - alpha) + v_end * alpha
            lat = frame.lat_from_y(xyz[:, 1])
            out = torch.empty_like(xyz)
            out[:, 0] = v[:, 0] / frame.metres_per_x(lat)
            out[:, 1] = v[:, 1] / frame.metres_per_y
            out[:, 2] = v[:, 2]
            return out

        xyz = particles[:, :3]
        v1 = wind(xyz)
        out = particles.clone()
        if self._forward_in_time:
            # Forward time (reciprocity test only).
            v2 = wind(xyz + 0.5 * dt_seconds * v1)
            out[:, :3] = xyz + dt_seconds * v2
            return out
        v2 = wind(xyz - 0.5 * dt_seconds * v1)
        out[:, :3] = xyz - dt_seconds * v2
        return out

    def _step_core(
        self,
        *,
        particles,
        u_prime,
        v_prime,
        w_prime,
        sigma_last,
        u_meander,
        v_meander,
        active_mask,
        blh_field,
        sp_field,
        ustar_field,
        shf_field,
        t_field,
        ft_fields,
        density_fields,
        meander_fields,
        grid_bounds,
        m_start_uvw,
        m_end_uvw,
        alpha,
        level_arr,
        ascending,
        dt_seconds,
        engine,
    ):
        """Pure-tensor WHOLE per-step pipeline for the static path -- the torch.compile /
        CUDA-graph target (docs/architecture.md S5): RK2 advection + interp + column turbulence +
        drift + substep loop + meander + mask-gated write-back. All met *tensors* + bounds
        are passed in (gathered by `_gather_static_inputs`) so nothing here touches the met
        window. Operates on the FULL particle buffer; inactive particles are gated back to
        their ORIGINAL (pre-advection) state at the end. Returns the full state (inactive
        unchanged; ``sigma_last`` is σ at each particle's end-of-step position, decision
        0028), plus this step's sub-step exceedance counts (see
        `_integrate_vertical_substeps`). The sub-step loop has the fixed trip count
        `self.n_substeps`, so the same code runs eager and captured."""
        # RK2 advection first (full set). Keep the pre-advection particles for the gate so
        # inactive particles are reverted to their original position, not the advected one.
        orig_particles = particles
        particles = self._advect_rk2(
            particles,
            m_start_uvw,
            m_end_uvw,
            alpha,
            grid_bounds,
            dt_seconds,
            engine,
            level_arr=level_arr,
            ascending=ascending,
        )

        # Horizontal position in the engine's frame; `grid_bounds` are in it too.
        # Latitude in degrees only for the Coriolis parameter.
        x = particles[:, 0]
        y = particles[:, 1]
        lat_deg = engine.frame.lat_from_y(y)
        z_agl = particles[:, 2]
        x0, x1 = grid_bounds.lon_first, grid_bounds.lon_last
        y0, y1 = grid_bounds.lat_first, grid_bounds.lat_last

        blh = self._limit_blh(self._bilinear_at(blh_field, x0, x1, y0, y1, x, y))
        sp = self._bilinear_at(sp_field, x0, x1, y0, y1, x, y)
        ustar = self._bilinear_at(ustar_field, x0, x1, y0, y1, x, y).clamp(min=USTAR_MIN_M_S)
        shf = self._bilinear_at(shf_field, x0, x1, y0, y1, x, y)
        t_surface = self._bilinear_at(t_field, x0, x1, y0, y1, x, y)

        L = obukhov_length(ustar, shf, t_surface, sp)
        h_over_L = blh / L
        w_star = convective_velocity(blh, shf, t_surface, sp, blh_min_m=self.blh_min_m)

        col_kwargs = dict(
            blh=blh,
            ustar=ustar,
            w_star=w_star,
            h_over_L=h_over_L,
            L=L,
            lat=lat_deg,
            x=x,
            y=y,
            ft_fields=ft_fields,
            grid_bounds=grid_bounds,
            engine=engine,
            level_arr=level_arr,
            ascending=ascending,
        )
        z_eval = z_agl.clamp(min=self.z_ubl_m)

        def coeffs_at(z: torch.Tensor) -> tuple[torch.Tensor, ...]:
            return self._vertical_coefficients(z, density_fields, **col_kwargs)

        (
            sigma_u,
            sigma_v,
            sigma_w,
            sigma_w_sq,
            T_Lu,
            T_Lv,
            T_Lw,
            half_dsig2_dz_backward,
            density_drift_backward,
        ) = coeffs_at(z_agl)
        wind_axes = None
        if self.horizontal_frame == "wind":
            wind_axes = self._wind_axes(
                m_start_uvw[:2],
                m_end_uvw[:2],
                alpha,
                sigma_u,
                x=x,
                y=y,
                z=z_eval,
                grid_bounds=grid_bounds,
                engine=engine,
                level_arr=level_arr,
                ascending=ascending,
            )
        u_prime_in, v_prime_in = self._grid_to_wind(u_prime, v_prime, self._state_axes(wind_axes))
        u_prime_in, v_prime_in, w_prime_in = self._carry_velocities(
            u_prime_in, v_prime_in, w_prime, sigma_last, sigma_u, sigma_v, sigma_w
        )

        (
            moved,
            u_prime_new,
            v_prime_new,
            w_prime_new,
            exceed_counts,
            exceed_worst,
        ) = self._integrate_vertical_substeps(
            active_xyz=particles,
            u_prime_in=u_prime_in,
            v_prime_in=v_prime_in,
            w_prime_in=w_prime_in,
            sigma_u=sigma_u,
            sigma_v=sigma_v,
            sigma_w=sigma_w,
            sigma_w_sq=sigma_w_sq,
            T_Lu=T_Lu,
            T_Lv=T_Lv,
            T_Lw=T_Lw,
            half_dsig2_dz_backward=half_dsig2_dz_backward,
            density_drift_backward=density_drift_backward,
            coeffs_at=coeffs_at,
            blh=blh,
            dt_seconds=dt_seconds,
            engine=engine,
            active_mask=active_mask,
            interface_sigmas=self._interface_sigmas(**col_kwargs),
            wind_axes=wind_axes,
        )
        u_prime_new, v_prime_new = self._wind_to_grid(
            u_prime_new, v_prime_new, self._state_axes(wind_axes)
        )
        sigma_last_new = self._sigmas_at(moved[:, 2], **col_kwargs)

        u_meander_new, v_meander_new = u_meander, v_meander
        if self.meander_enabled:
            sm = self._interp_3d_field(
                meander_fields,
                grid_bounds,
                x,
                y,
                z_eval,
                engine,
                level_arr=level_arr,
                ascending=ascending,
            )
            u_meander_new = engine.update_langevin_velocity(
                u_meander,
                t_lagrangian=self.meander_timescale_seconds,
                sigma_w2=sm[0].pow(2),
                dt_seconds=dt_seconds,
            )
            v_meander_new = engine.update_langevin_velocity(
                v_meander,
                t_lagrangian=self.meander_timescale_seconds,
                sigma_w2=sm[1].pow(2),
                dt_seconds=dt_seconds,
            )
            moved = engine.apply_horizontal_turbulence(
                moved,
                u_meander_new,
                v_meander_new,
                dt_seconds=dt_seconds,
                backward=not self._forward_in_time,
            )

        # Mask-gated write-back (authoritative gate: covers advection + substep + meander).
        # Inactive particles revert to `orig_particles` — their ORIGINAL pre-advection
        # state — so they are frozen exactly (advection ran on the full set but is undone
        # here for them, same as the pre-fold `where(active, advected, particles)`).
        am = active_mask.unsqueeze(1)
        particles_new = torch.where(am, moved, orig_particles)
        u_prime_new = torch.where(active_mask, u_prime_new, u_prime)
        v_prime_new = torch.where(active_mask, v_prime_new, v_prime)
        w_prime_new = torch.where(active_mask, w_prime_new, w_prime)
        sigma_last_new = torch.where(am, sigma_last_new, sigma_last)
        if self.meander_enabled:
            u_meander_new = torch.where(active_mask, u_meander_new, u_meander)
            v_meander_new = torch.where(active_mask, v_meander_new, v_meander)
        return (
            particles_new,
            u_prime_new,
            v_prime_new,
            w_prime_new,
            sigma_last_new,
            u_meander_new,
            v_meander_new,
            exceed_counts,
            exceed_worst,
        )

    @staticmethod
    def _bilinear_at(field_2d, lon0, lon1, lat0, lat1, particle_lon, particle_lat):
        """Bilinear sample of a [Y, X] field at particle (lon, lat) from float grid bounds
        -- the compile-clean variant of `_interp_2d_bilinear` (no numpy-array access)."""
        if lon1 == lon0 or lat1 == lat0:
            raise ValueError("Met grid has zero span in lon or lat -- cannot interpolate.")
        lon_norm = (2.0 * (particle_lon - lon0) / (lon1 - lon0) - 1.0).clamp(-1.0, 1.0)
        lat_norm = (2.0 * (particle_lat - lat0) / (lat1 - lat0) - 1.0).clamp(-1.0, 1.0)
        grid = (
            torch.stack([lon_norm, lat_norm], dim=-1)
            .view(1, 1, -1, 2)
            .to(device=field_2d.device, dtype=field_2d.dtype)
        )
        field = field_2d.unsqueeze(0).unsqueeze(0)
        sampled = torch.nn.functional.grid_sample(field, grid, align_corners=True, mode="bilinear")
        return sampled.view(-1)

    def _use_static_substeps(self, engine: GPUEngine) -> bool:
        """Choose the static-shape vs dynamic-masked per-step path (docs/architecture.md §5).

        Delegates to the shared `use_static_step_path` so the scheme and the `main`
        runtime loop always agree on the device gate; the per-instance
        ``static_substeps`` ctor flag is the override (used by the equivalence tests).
        """

        return use_static_step_path(engine.device, self._static_substeps)

    def step_includes_advection(self, engine: GPUEngine) -> bool:
        """On the static path, RK2 advection is folded into `_step_core` (so the whole
        step is one captured graph), and the runtime must not advect separately. The
        dynamic path leaves advection to the runtime (returns False)."""

        return self._use_static_substeps(engine)

    def _maybe_graph_compile(self, engine: GPUEngine) -> Callable[..., tuple] | None:
        """Return a CUDA-graph-capturing compiled per-step core (`_step_core`), or None.

        Enabled only on the static path when compilation was requested
        (``engine._compile_requested``, i.e. GLIDE_COMPILE). Compiles the **whole**
        per-step tensor core once with ``mode="reduce-overhead"`` (Inductor cudagraph
        trees): on CUDA the entire step (interp + column turbulence + drift + substep
        loop + meander + write-back) captures as **one graph**, replayed per step with a
        single host call. This is phase 4 — collapsing the ~1000+ eager kernel launches
        per step that the profile (2026-06-26) showed were the launch-bound wall (the
        earlier phase-3 capture of just the substep loop left them). Eager fallback on
        any failure, so it can never hard-fail a run; the FIRST step pays a one-time
        compile cost. On CPU there are no CUDA graphs — it just compiles via Inductor
        (used by the smoke test to confirm the core is traceable).
        """

        if not (self._use_static_substeps(engine) and getattr(engine, "_compile_requested", False)):
            return None
        if self._graph_compile_state == "failed":
            return None
        if self._compiled_graph_core is not None:
            return self._compiled_graph_core
        try:
            import torch._dynamo as dynamo

            dynamo.config.suppress_errors = True  # per-graph eager fallback
            self._compiled_graph_core = torch.compile(
                self._step_core,
                mode="reduce-overhead",
                dynamic=False,
                fullgraph=False,
            )
            self._graph_compile_state = "ok"
            LOGGER.info(
                "HannaScheme: torch.compile(mode='reduce-overhead') enabled for the "
                "whole per-step core (CUDA-graph capture; eager fallback on)."
            )
            return self._compiled_graph_core
        except Exception as exc:  # pragma: no cover - defensive; compile is opt-in
            LOGGER.warning(
                "HannaScheme: per-step-core graph compile setup failed (%s); running eager", exc
            )
            self._graph_compile_state = "failed"
            return None

    def _integrate_vertical_substeps(
        self,
        *,
        active_xyz: torch.Tensor,
        u_prime_in: torch.Tensor,
        v_prime_in: torch.Tensor,
        w_prime_in: torch.Tensor,
        sigma_u: torch.Tensor,
        sigma_v: torch.Tensor,
        sigma_w: torch.Tensor,
        sigma_w_sq: torch.Tensor,
        T_Lu: torch.Tensor,
        T_Lv: torch.Tensor,
        T_Lw: torch.Tensor,
        half_dsig2_dz_backward: torch.Tensor,
        density_drift_backward: torch.Tensor,
        dt_seconds: float,
        engine: GPUEngine,
        coeffs_at: Callable[[torch.Tensor], tuple[torch.Tensor, ...]] | None = None,
        blh: torch.Tensor | None = None,
        active_mask: torch.Tensor | None = None,
        interface_sigmas: torch.Tensor | None = None,
        wind_axes: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, ...]:
        """The sub-step loop: OU velocity update + displacement + ground reflection,
        ``self.n_substeps`` equal sub-steps of ``dt / n_substeps`` for every particle
        (decision 0027, docs/physics.md §6).

        **Coefficients.** The coefficients passed in are those at the start of the
        outer step and drive the first sub-step. When ``coeffs_at`` is given (as it
        is from ``step``), σ, T_L, ``∂σ_w²/∂z`` and the density drift are
        re-evaluated at each particle's current height at the start of every later
        sub-step. Holding them at the outer-step value acts like a random walk with
        the diffusivity of the starting height and no compensating drift, which
        piles particles up where T_L is short — at the ground. Without ``coeffs_at``
        they stay at the values passed in (the synthetic-input unit tests).

        **One count for everyone.** The count is the configured constant, never
        chosen per particle: a sub-step length that depends on the particle's height
        (the old ``k_i = ceil(dt / (c·T_Lw,i))``, or a count re-chosen every sub-step)
        is a spurious drift towards where the steps are short, and leaves the lowest
        5% of a neutral boundary layer ~30% over-occupied at dt = 60 s even with the
        coefficients re-evaluated. The loop has a compile-time-constant trip count
        and no host sync, so it captures in the CUDA graph (decision 0004).

        **Both paths.** The dynamic (CPU) path passes the active subset with
        ``active_mask=None``. The static path passes the full particle buffer and
        ``active_mask``: inactive particles get ``sub_dt = 0``, a mathematical
        no-op (``a = exp(0) = 1``, no noise, no displacement, and reflecting a
        non-negative ``z`` is the identity), so shapes are constant across
        iterations. With every particle active the two are bit-identical (same
        operations, same random draws); with inactive particles present the
        full-set draw differs from the subset draw, so they agree statistically.

        The well-mixed drift's ``(1 + w'²/σ_w²)`` factor is recomputed every
        substep from the current ``w'``.

        **Exceedance counter.** Every sub-step is checked against
        ``sub_dt > exceedance_c · T_Lw`` and ``sub_dt > T_Lw``, with T_Lw at the
        sub-step's own height, split by whether that height is in the boundary
        layer (``z ≤ h``, needs ``blh``) or the free troposphere. Returns, after the
        state, ``counts`` (active particle-sub-steps, int64, in the order of
        ``EXCEEDANCE_COUNT_KEYS``) and ``worst = [max sub_dt/T_Lw in BL, in FT]``,
        as device tensors (no host sync). Without ``blh`` everything counts as BL.

        **Boundary-layer top.** With ``blh`` and ``interface_sigmas`` (σ at h on
        each side, ``_interface_sigmas``; ``step`` passes both), a particle whose
        sub-step displacement crosses h is transmitted or reflected by the
        Thomson, Physick & Maryon (1997) rule (``_cross_interface``, decision
        0028) before the ground reflection. Without them h is not treated (the
        synthetic-input unit tests).

        Ground reflection (W&F §6) is applied inside the loop, so a particle that
        crosses z=0 within a substep is reflected before the next substep.

        **Horizontal frame.** ``u_prime_in`` / ``v_prime_in`` and the returned
        u', v' are in whatever frame σ_u / σ_v belong to. With ``wind_axes`` (the
        ``[2, N]`` unit vector of the mean wind, ``_wind_axes``) they are the
        along-wind and cross-wind components and are rotated to east/north for each
        displacement only; without it they are east/north already (decision 0032).
        """

        n = self.n_substeps
        xyz = active_xyz.clone()
        u_p = u_prime_in.clone()
        v_p = v_prime_in.clone()
        w_p = w_prime_in.clone()

        sub_dt = torch.full_like(T_Lw, float(dt_seconds) / n)
        if active_mask is not None:
            # Inactive particles: sub_dt = 0, a no-op every iteration.
            sub_dt = torch.where(active_mask, sub_dt, torch.zeros_like(sub_dt))

        # Exceedance counter, per particle across the sub-steps, reduced once at the end.
        n_ft = torch.zeros(T_Lw.shape, dtype=torch.int32, device=T_Lw.device)
        ex_bl = torch.zeros_like(n_ft)
        ex_ft = torch.zeros_like(n_ft)
        over_bl = torch.zeros_like(n_ft)
        over_ft = torch.zeros_like(n_ft)
        worst_bl = torch.zeros_like(sub_dt)
        worst_ft = torch.zeros_like(sub_dt)
        c_exceed = self.exceedance_c

        for i in range(n):
            if coeffs_at is not None and i > 0:
                # Re-evaluate σ, T_L, ∂σ_w²/∂z and the density drift at the
                # particle's CURRENT height (docs/physics.md §6).
                (
                    sigma_u,
                    sigma_v,
                    sigma_w,
                    sigma_w_sq,
                    T_Lu,
                    T_Lv,
                    T_Lw,
                    half_dsig2_dz_backward,
                    density_drift_backward,
                ) = coeffs_at(xyz[:, 2])

            # Exceedance of this sub-step (inactive particles: sub_dt = 0, never).
            ratio = sub_dt / T_Lw
            if blh is None:
                in_ft = torch.zeros(T_Lw.shape, dtype=torch.bool, device=T_Lw.device)
            else:
                # The same test that selects the free-troposphere coefficients
                # (`_column_turbulence` on the clamped height).
                in_ft = xyz[:, 2].clamp(min=self.z_ubl_m) > blh
            in_bl = ~in_ft
            exceeded = ratio > c_exceed
            over = ratio > 1.0
            n_ft += in_ft
            ex_bl += exceeded & in_bl
            ex_ft += exceeded & in_ft
            over_bl += over & in_bl
            over_ft += over & in_ft
            worst_bl = torch.maximum(worst_bl, torch.where(in_ft, 0.0, ratio))
            worst_ft = torch.maximum(worst_ft, torch.where(in_ft, ratio, 0.0))

            # Backward drift with the CURRENT w':
            #   drift_b = -½(1 + w'²/σ_w²)·∂σ²/∂z  -  (σ_w²/ρ)·∂ρ/∂z
            # Pieces are pre-negated for the backward sign.
            factor = 1.0 + w_p.pow(2) / sigma_w_sq
            drift_w = factor * half_dsig2_dz_backward + density_drift_backward

            u_new = engine.update_langevin_velocity(
                u_p, t_lagrangian=T_Lu, sigma_w2=sigma_u.pow(2), dt_seconds=sub_dt
            )
            v_new = engine.update_langevin_velocity(
                v_p, t_lagrangian=T_Lv, sigma_w2=sigma_v.pow(2), dt_seconds=sub_dt
            )
            w_new = engine.update_langevin_velocity(
                w_p,
                t_lagrangian=T_Lw,
                sigma_w2=sigma_w.pow(2),
                dt_seconds=sub_dt,
                drift=drift_w,
            )
            # Rogue-trajectory safeguard (spec §T): clip |w'/σ| at 4× to prevent the
            # (1+w'²/σ²) factor from snowballing the drift when σ_w hits its floor
            # (e.g. at the BL top). FLEXPART has the equivalent clip in its OU
            # implementation.
            w_max = W_PRIME_SIGMA_RATIO_MAX * sigma_w
            u_max = W_PRIME_SIGMA_RATIO_MAX * sigma_u
            v_max = W_PRIME_SIGMA_RATIO_MAX * sigma_v
            w_new = torch.clamp(w_new, min=-w_max, max=w_max)
            u_new = torch.clamp(u_new, min=-u_max, max=u_max)
            v_new = torch.clamp(v_new, min=-v_max, max=v_max)

            z_from = xyz[:, 2]
            backward = not self._forward_in_time
            xyz = engine.apply_vertical_turbulence(xyz, w_new, dt_seconds=sub_dt, backward=backward)
            u_east, v_north = self._wind_to_grid(u_new, v_new, wind_axes)
            xyz = engine.apply_horizontal_turbulence(
                xyz, u_east, v_north, dt_seconds=sub_dt, backward=backward
            )
            if interface_sigmas is not None and blh is not None:
                # Boundary-layer top (decision 0028): transmit or reflect the
                # particles whose displacement crossed h in this sub-step.
                xyz, w_new, u_new, v_new = self._cross_interface(
                    z_from, xyz, w_new, u_new, v_new, blh, interface_sigmas
                )
            # W&F §6 smooth-wall reflection: joint (z, w') flip on those that
            # crossed the surface within this substep.
            xyz, w_new = engine.reflect_surface(xyz, w_new, z_surface=0.0)
            u_p, v_p, w_p = u_new, v_new, w_new

        if active_mask is not None:
            n_ft = n_ft * active_mask
            n_active = active_mask.sum()
        else:
            n_active = torch.tensor(T_Lw.numel(), device=T_Lw.device)
        n_ft_total = n_ft.sum(dtype=torch.int64)
        counts = torch.stack(
            [
                n_active.to(torch.int64) * n - n_ft_total,
                ex_bl.sum(dtype=torch.int64),
                over_bl.sum(dtype=torch.int64),
                n_ft_total,
                ex_ft.sum(dtype=torch.int64),
                over_ft.sum(dtype=torch.int64),
            ]
        )
        if worst_bl.numel():
            worst = torch.stack([worst_bl.amax(), worst_ft.amax()])
        else:
            worst = torch.zeros(2, dtype=sub_dt.dtype, device=sub_dt.device)
        return xyz, u_p, v_p, w_p, counts, worst

    def _accumulate_exceedance(self, counts: torch.Tensor, worst: torch.Tensor) -> None:
        """Add one step's exceedance counts to the running device-side totals
        (no host sync)."""

        if self._exceed_counts is None or self._exceed_counts.device != counts.device:
            self._exceed_counts = torch.zeros_like(counts)
            self._exceed_worst = torch.zeros_like(worst)
        self._exceed_counts += counts
        torch.maximum(self._exceed_worst, worst, out=self._exceed_worst)

    def take_substep_exceedance(self, window_start: object) -> dict[str, float] | None:
        """Read and reset the sub-step exceedance totals (one host sync).

        Called by the runtime once per met window (and at the end of each batch).
        Returns, since the last call, the active particle-sub-steps in the boundary
        layer and the free troposphere (``substeps_*``), how many of each had
        ``sub_dt > exceedance_c · T_Lw`` (``exceeded_*``) and ``sub_dt > T_Lw``
        (``over_T_L_*``), and the worst ``sub_dt / T_Lw`` in each (``worst_*``).
        """

        del window_start
        if self._exceed_counts is None:
            out: dict[str, float] = dict.fromkeys(EXCEEDANCE_COUNT_KEYS, 0)
            out.update(worst_bl=0.0, worst_ft=0.0)
            return out
        values = torch.cat(
            [self._exceed_counts.to(torch.float64), self._exceed_worst.to(torch.float64)]
        ).tolist()
        self._exceed_counts.zero_()
        self._exceed_worst.zero_()
        n = len(EXCEEDANCE_COUNT_KEYS)
        out = {k: int(v) for k, v in zip(EXCEEDANCE_COUNT_KEYS, values[:n], strict=True)}
        out.update(worst_bl=float(values[n]), worst_ft=float(values[n + 1]))
        return out

    def _column_turbulence(
        self,
        z_query: torch.Tensor,
        *,
        blh: torch.Tensor,
        ustar: torch.Tensor,
        w_star: torch.Tensor,
        h_over_L: torch.Tensor,
        L: torch.Tensor,
        lat: torch.Tensor,
        x: torch.Tensor,
        y: torch.Tensor,
        ft_fields: torch.Tensor,
        grid_bounds: GridInterpolationBounds,
        engine: GPUEngine,
        level_arr: torch.Tensor | None = None,
        ascending: bool | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Per-particle (σ_u, σ_v, σ_w, T_Lu, T_Lv, T_Lw) at height ``z_query``.

        Assembles the regimes — in-BL (Hanna, run to the ground by default per
        FLEXPART v11), the optional Monin-Obukhov surface-layer override
        (``surface_layer_override``, z < 0.1 h), and the free-troposphere override
        (z > h, Richardson closure interpolated from ``ft_fields``) — selected per
        particle by where ``z_query`` falls. T_L floors: FLEXPART's 10/10/30 s when
        ``flexpart_tl_floors`` (default), else the legacy 1 s. Callable at
        arbitrary z so the well-mixed drift can finite-difference σ_w through the
        regime transitions.
        """

        sigma_u, sigma_v, sigma_w, T_Lu, T_Lv, T_Lw = in_bl_sigma_TL(
            z_query,
            blh,
            ustar,
            w_star,
            h_over_L,
            lat,
        )

        # OPTIONAL Monin-Obukhov surface-layer override for z < 0.1 h (GLIDE
        # addition; off by default — FLEXPART v11 runs the regime formulas to the
        # ground). The T_L = κz/σ scaling undercuts the FLEXPART floors near the
        # surface and puts a K discontinuity at the 0.1 h seam; keep only for A/B.
        if self.surface_layer_override:
            z_sl = SURFACE_LAYER_FRACTION * blh
            in_sl = z_query < z_sl
            sigma_w_sl = surface_layer_sigma_w(z_query, ustar, L)
            z_clamped = torch.maximum(z_query, z_sl)
            su_cl, sv_cl, _, _, _, _ = in_bl_sigma_TL(z_clamped, blh, ustar, w_star, h_over_L, lat)
            sigma_u = torch.where(in_sl, su_cl, sigma_u)
            sigma_v = torch.where(in_sl, sv_cl, sigma_v)
            sigma_w = torch.where(in_sl, sigma_w_sl, sigma_w)
            z_for_TL = z_query.clamp(min=Z_MIN_M)
            T_Lu = torch.where(in_sl, KARMAN * z_for_TL / sigma_u.clamp(min=SIGMA_MIN_M_S), T_Lu)
            T_Lv = torch.where(in_sl, KARMAN * z_for_TL / sigma_v.clamp(min=SIGMA_MIN_M_S), T_Lv)
            T_Lw = torch.where(in_sl, KARMAN * z_for_TL / sigma_w.clamp(min=SIGMA_MIN_M_S), T_Lw)

        # FLEXPART v11 T_L floors (tlu/tlv >= 10 s, tlw >= 30 s) on the BL profile.
        # Applied BEFORE the free-troposphere override: the FT closure derives
        # (σ_w, T_Lw) as a pair satisfying K = σ_w²·T_Lw, so flooring its T_L after
        # the fact would silently change K. FLEXPART likewise floors only in hanna.
        if self.flexpart_tl_floors:
            T_Lu = T_Lu.clamp(min=T_LU_FLOOR_FLEXPART_S)
            T_Lv = T_Lv.clamp(min=T_LV_FLOOR_FLEXPART_S)
            T_Lw = T_Lw.clamp(min=T_LW_FLOOR_FLEXPART_S)

        # Free-troposphere override for z > h. Applied unconditionally (the
        # `torch.where` is a no-op where `above_bl` is false) rather than guarded
        # by `if bool(torch.any(above_bl))` — that guard is a host sync, and this
        # runs 3× per step (σ_w plus the two drift finite-difference probes). The
        # only cost when nothing is above the BL is one extra cached-field
        # grid_sample, which is far cheaper on CUDA than the sync it removes.
        above_bl = z_query > blh
        ft = self._interp_3d_field(
            ft_fields,
            grid_bounds,
            x,
            y,
            z_query,
            engine,
            level_arr=level_arr,
            ascending=ascending,
        )
        ft_sw, ft_suv, ft_tlw, ft_tluv = ft[0], ft[1], ft[2], ft[3]
        sigma_u = torch.where(above_bl, ft_suv, sigma_u)
        sigma_v = torch.where(above_bl, ft_suv, sigma_v)
        sigma_w = torch.where(above_bl, ft_sw, sigma_w)
        T_Lu = torch.where(above_bl, ft_tluv, T_Lu)
        T_Lv = torch.where(above_bl, ft_tluv, T_Lv)
        T_Lw = torch.where(above_bl, ft_tlw, T_Lw)

        sigma_u = sigma_u.clamp(min=SIGMA_MIN_M_S)
        sigma_v = sigma_v.clamp(min=SIGMA_MIN_M_S)
        sigma_w = sigma_w.clamp(min=SIGMA_MIN_M_S)
        T_Lu = T_Lu.clamp(min=T_L_MIN_S)
        T_Lv = T_Lv.clamp(min=T_L_MIN_S)
        T_Lw = T_Lw.clamp(min=T_L_MIN_S)
        return sigma_u, sigma_v, sigma_w, T_Lu, T_Lv, T_Lw

    def _column_sigma_w(
        self,
        z_query: torch.Tensor,
        *,
        blh: torch.Tensor,
        ustar: torch.Tensor,
        w_star: torch.Tensor,
        h_over_L: torch.Tensor,
        L: torch.Tensor,
        lat: torch.Tensor,
        x: torch.Tensor,
        y: torch.Tensor,
        ft_fields: torch.Tensor,
        grid_bounds: GridInterpolationBounds,
        engine: GPUEngine,
        level_arr: torch.Tensor | None = None,
        ascending: bool | None = None,
        side_ft: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """σ_w alone at ``z_query``: bit-identical to ``_column_turbulence(...)[2]``
        (same regime selection, surface-layer override and free-troposphere
        override; the T_L floors never touch σ), but it evaluates only the σ_w
        expressions and samples only the σ_w channel of the free-troposphere stack.
        Used for the two finite-difference probes of ∂σ_w²/∂z and for σ_w on
        either side of h at the interface.

        ``side_ft`` (bool per particle) forces the side: True takes the
        free-troposphere closure, False the boundary-layer profile, whatever
        ``z_query`` is (so a probe at exactly ``z = h`` can be evaluated on
        either side). ``None`` selects by ``z_query > h``, as the profile does."""

        sigma_w = in_bl_sigma_w(z_query, blh, ustar, w_star, h_over_L, lat)
        if self.surface_layer_override:
            in_sl = z_query < SURFACE_LAYER_FRACTION * blh
            sigma_w = torch.where(in_sl, surface_layer_sigma_w(z_query, ustar, L), sigma_w)
        above_bl = z_query > blh if side_ft is None else side_ft
        ft_sw = self._interp_3d_field(
            ft_fields[:1],
            grid_bounds,
            x,
            y,
            z_query,
            engine,
            level_arr=level_arr,
            ascending=ascending,
        )[0]
        sigma_w = torch.where(above_bl, ft_sw, sigma_w)
        return sigma_w.clamp(min=SIGMA_MIN_M_S)

    def _sigmas_at(self, z_agl: torch.Tensor, **col_kwargs) -> torch.Tensor:
        """(σ_u, σ_v, σ_w) at height ``z_agl`` (basal clamp applied), ``[N, 3]``:
        the same values ``_vertical_coefficients`` would give there."""

        su, sv, sw, *_ = self._column_turbulence(z_agl.clamp(min=self.z_ubl_m), **col_kwargs)
        return torch.stack([su, sv, sw], dim=1)

    def _wind_axes(
        self,
        uv_start: torch.Tensor,
        uv_end: torch.Tensor,
        alpha: torch.Tensor | float,
        sigma_u: torch.Tensor,
        *,
        x: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
        grid_bounds: GridInterpolationBounds,
        engine: GPUEngine,
        level_arr: torch.Tensor | None = None,
        ascending: bool | None = None,
    ) -> torch.Tensor:
        """Unit vector ``[cos φ, sin φ]`` (``[2, N]``, east and north components) of
        the mean horizontal wind at each particle, for the ``"wind"`` frame
        (decision 0032). ``uv_start`` / ``uv_end`` are the ``[2, Z, Y, X]`` u and v
        stacks (m/s) at the window's ends, sampled as the advection samples the wind
        (``grid_sample``, or ``trilinear_sample`` for a 16-bit wind-gather dtype) and
        interpolated in time with ``alpha``; ``(x, y)`` is the position in the frame
        of ``grid_bounds`` and ``z`` the basally clamped height the coefficients use.
        The direction does not depend on the position frame: u and v are east and
        north in every frame.

        Where the speed is at most ``WIND_FRAME_CALM_RATIO · σ_u`` the axes are the
        grid's (``[1, 0]``), so u' stays east and v' north for that particle and
        step. A ``torch.where`` on the device: no host sync, the same on both step
        paths."""

        a = torch.as_tensor(alpha, device=x.device, dtype=x.dtype)
        xyz_norm = engine.normalize_particle_coordinates(
            torch.stack([x, y, z], dim=1), grid_bounds, level_arr=level_arr, ascending=ascending
        )
        if uv_start.dtype.itemsize == 2:
            s, e = trilinear_sample((uv_start, uv_end), xyz_norm)
        else:
            grid = xyz_norm.view(1, 1, 1, -1, 3)
            sample = torch.nn.functional.grid_sample
            s = sample(uv_start.unsqueeze(0), grid, align_corners=True).view(2, -1)
            e = sample(uv_end.unsqueeze(0), grid, align_corners=True).view(2, -1)
        uv = s.to(x.dtype) * (1.0 - a) + e.to(x.dtype) * a
        speed = torch.hypot(uv[0], uv[1])
        windy = speed > WIND_FRAME_CALM_RATIO * sigma_u
        inv = 1.0 / speed.clamp(min=SIGMA_MIN_M_S)
        cos_phi = torch.where(windy, uv[0] * inv, torch.ones_like(speed))
        sin_phi = torch.where(windy, uv[1] * inv, torch.zeros_like(speed))
        return torch.stack([cos_phi, sin_phi])

    def _state_axes(self, wind_axes: torch.Tensor | None) -> torch.Tensor | None:
        """The axes that rotate the carried u', v' into and out of the wind frame at
        the start and end of an outer step: ``wind_axes``, or None (no rotation, the
        state stays wind-relative) under the private ``_carry_wind_relative``."""

        return None if self._carry_wind_relative else wind_axes

    @staticmethod
    def _grid_to_wind(
        u_east: torch.Tensor, v_north: torch.Tensor, wind_axes: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """East/north components to (along-wind, cross-wind); the identity (the same
        tensors) when ``wind_axes`` is None. Cross-wind is 90° to the left of the
        wind, as in FLEXPART's ``windalign``."""

        if wind_axes is None:
            return u_east, v_north
        c, s = wind_axes[0], wind_axes[1]
        return c * u_east + s * v_north, c * v_north - s * u_east

    @staticmethod
    def _wind_to_grid(
        along: torch.Tensor, cross: torch.Tensor, wind_axes: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """(along-wind, cross-wind) to east/north; the inverse of ``_grid_to_wind``
        and likewise the identity when ``wind_axes`` is None."""

        if wind_axes is None:
            return along, cross
        c, s = wind_axes[0], wind_axes[1]
        return c * along - s * cross, s * along + c * cross

    @staticmethod
    def _carry_velocities(
        u_p: torch.Tensor,
        v_p: torch.Tensor,
        w_p: torch.Tensor,
        sigma_last: torch.Tensor,
        sigma_u: torch.Tensor,
        sigma_v: torch.Tensor,
        sigma_w: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Carry u'/σ_u, v'/σ_v and w'/σ_w across the outer step (decision 0028).

        Between the end of one outer step and the start of the next, σ at a
        particle changes because the meteorology was interpolated to a new time
        and the particle was advected by the mean wind; within a step σ changes
        only along the turbulent displacement, which the drift accounts for.
        Scaling each velocity by σ(now)/σ(end of last step) is the exact
        integral of the nonstationary term (w/2σ²)·∂σ²/∂t of the Gaussian
        well-mixed drift (Thomson 1987) over that change, and it is the
        treatment of step changes in time of Thomson, Physick & Maryon (1997,
        §4: keep w/σ_w fixed across a time discontinuity; §5 extends it to u_H). It matters at h: when h moves past a particle,
        σ_w at it jumps between the boundary-layer and free-troposphere values.
        ``sigma_last`` is 0 before a particle's first step (no scaling). In a
        steady, motionless column the ratio is 1 to within the change of σ over
        the particle's horizontal turbulent displacement (< 1e-4,
        ``test_carry_is_one_in_a_steady_column_and_tracks_a_moving_h``)."""

        now = torch.stack([sigma_u, sigma_v, sigma_w], dim=1)
        ratio = torch.where(
            sigma_last > 0.0, now / sigma_last.clamp(min=SIGMA_MIN_M_S), torch.ones_like(now)
        )
        return u_p * ratio[:, 0], v_p * ratio[:, 1], w_p * ratio[:, 2]

    def _interface_sigmas(
        self,
        *,
        blh: torch.Tensor,
        ustar: torch.Tensor,
        w_star: torch.Tensor,
        h_over_L: torch.Tensor,
        lat: torch.Tensor,
        x: torch.Tensor,
        y: torch.Tensor,
        ft_fields: torch.Tensor,
        grid_bounds: GridInterpolationBounds,
        engine: GPUEngine,
        level_arr: torch.Tensor | None = None,
        ascending: bool | None = None,
        **_unused: object,
    ) -> torch.Tensor:
        """(σ_u, σ_v, σ_w) at the boundary-layer top on each side, ``[2, 3, N]``:
        index 0 the boundary-layer profile at z = h (the value ``_column_turbulence``
        gives at h), index 1 the free-troposphere closure at z = h (the limit of
        its value from above), both floored as the profile is. They depend on the
        column only (h, u*, w*, h/L at the particle's outer-step position and
        time), so they are evaluated once per outer step (decision 0028)."""

        su, sv, sw, *_ = in_bl_sigma_TL(blh, blh, ustar, w_star, h_over_L, lat)
        ft = self._interp_3d_field(
            ft_fields[:2],
            grid_bounds,
            x,
            y,
            blh,
            engine,
            level_arr=level_arr,
            ascending=ascending,
        )
        bl_side = torch.stack([su, sv, sw])
        ft_side = torch.stack([ft[1], ft[1], ft[0]])
        return torch.stack([bl_side, ft_side]).clamp(min=SIGMA_MIN_M_S)

    def _cross_interface(
        self,
        z_from: torch.Tensor,
        xyz: torch.Tensor,
        w_p: torch.Tensor,
        u_p: torch.Tensor,
        v_p: torch.Tensor,
        blh: torch.Tensor,
        interface_sigmas: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Transmit or reflect the particles whose sub-step displacement crossed h
        (Thomson, Physick & Maryon 1997; decision 0028, docs/physics.md §6).

        A particle moving from side A to side B is transmitted with probability
        ``min(1, σ_wB/σ_wA)`` (σ at h on each side; the TPM factor ρ_B/ρ_A is 1
        because the density is continuous at h), its velocity scaled by σ_B/σ_A
        (w' by the σ_w ratio, u' and v' by the σ_u and σ_v ratios) and the part of
        the displacement beyond h scaled by the σ_w ratio (the rest of the
        sub-step flown at the new speed). Otherwise it is reflected: z → 2h − z,
        w' → −w', u' and v' unchanged. The rule uses only the direction of the
        displacement and |w'|, so it is the same in backward time, where the
        displacement is −w'·dt. Particles that did not cross are untouched.
        """

        z_to = xyz[:, 2]
        from_ft = z_from.clamp(min=self.z_ubl_m) > blh
        to_ft = z_to.clamp(min=self.z_ubl_m) > blh
        cross = from_ft != to_ft
        sig_a = torch.where(from_ft, interface_sigmas[1], interface_sigmas[0])
        sig_b = torch.where(from_ft, interface_sigmas[0], interface_sigmas[1])
        ratio = sig_b / sig_a  # [3, N]: u, v, w
        r_w = ratio[2]
        transmit = torch.rand_like(r_w) < r_w
        z_new = torch.where(transmit, blh + (z_to - blh) * r_w, 2.0 * blh - z_to)
        # Land strictly on the intended side (rounding could leave it at h).
        end_ft = torch.where(transmit, to_ft, from_ft)
        z_new = torch.where(
            end_ft,
            torch.maximum(z_new, blh + INTERFACE_SIDE_EPS_M),
            torch.minimum(z_new, blh),
        )
        through = cross & transmit
        out = xyz.clone()
        out[:, 2] = torch.where(cross, z_new, z_to)
        w_out = torch.where(cross, torch.where(transmit, w_p * r_w, -w_p), w_p)
        u_out = torch.where(through, u_p * ratio[0], u_p)
        v_out = torch.where(through, v_p * ratio[1], v_p)
        return out, w_out, u_out, v_out

    def _cached_window_field(
        self,
        met_window: HourlyMetTensors,
        key: str,
        build: Callable[[], torch.Tensor],
    ) -> torch.Tensor:
        """Build a grid field stack once per met window and reuse it all window.

        ``build()`` computes the stack ``[C, Z, Y, X]`` from the window met
        evaluated at its MIDPOINT (``t_alpha = 0.5``); we cache it keyed by
        ``metadata.time_start`` and return the same tensor for every step in that
        window (~60 at dt=60). Because ``step`` walks the met windows
        monotonically, a single cached entry per ``key`` suffices — a new window
        overwrites it.

        This replaces the previous per-step recompute of the grid-wide
        ``avg_pool2d`` / vertical-gradient stacks (~18% of runtime, profiled
        2026-06-18). The met-derived σ/ρ/FT fields evolve slowly — sub-percent
        over one met interval — so freezing them at the window midpoint is a
        standard met-cadence approximation (FLEXPART-class models refresh
        turbulence fields at the met step, not the particle step). The previously
        exact per-step *time* interpolation of these support fields is dropped;
        the per-particle interpolation through them is unchanged. Documented
        change (per-window field caching, perf 2026-06-18).
        """

        ts = met_window.metadata.time_start
        entry = self._window_field_cache.get(key)
        if entry is None or entry[0] != ts:
            # Route the freshly-built stack through its static buffer so the
            # compiled core sees a stable address across windows (perf #1); the
            # copy happens once per window, not per step.
            stack = self._to_static_buffer(f"winfield.{key}", build())
            self._window_field_cache[key] = (ts, stack)
            return stack
        return entry[1]

    @staticmethod
    def _device_matches(buf_device: torch.device, requested: torch.device) -> bool:
        """True if a buffer on ``buf_device`` satisfies ``requested``.

        ``torch.device("cuda") != torch.device("cuda:0")`` in PyTorch, and a
        tensor's ``.device`` is always index-qualified — so a naive ``!=``
        against the runtime's unindexed device reallocated (and re-marked) the
        alpha buffer EVERY STEP on CUDA. Each step then presented a brand-new
        marked-static input, forcing cudagraph trees to re-record the whole
        graph per step (~6 ms/step + capture syncs): the 2026-07-17 GH200
        profile regression (17 ms/step vs the 5.0 ms baseline). Invisible on
        CPU, where the comparison is benign — hence a dedicated unit test.
        """
        if buf_device.type != requested.type:
            return False
        return requested.index is None or buf_device.index == requested.index

    def _to_static_buffer(
        self, name: str, src: torch.Tensor, *, copy_src: bool = True
    ) -> torch.Tensor:
        """Copy ``src`` into a persistent buffer with a stable storage address.

        Allocated once per (shape, dtype, device) and marked as a cudagraph
        static address on CUDA, so graph replays read it in place instead of
        staging a copy of the input every step. Callers refill it in place
        (``copy_``) when the underlying value changes — always OUTSIDE the
        captured region. A shape/device change (new run geometry) reallocates,
        which simply triggers a fresh capture. ``copy_src=False`` returns the
        (possibly stale) buffer without copying, for callers that overwrite it
        entirely themselves (e.g. an ``out=`` lerp)."""

        buf = self._static_input_buffers.get(name)
        fresh = (
            buf is None
            or buf.shape != src.shape
            or buf.dtype != src.dtype
            or buf.device != src.device
        )
        if fresh:
            buf = torch.empty_like(src)
            if src.device.type == "cuda" and _MARK_STATIC_INPUTS:
                torch._dynamo.mark_static_address(buf)
            self._static_input_buffers[name] = buf
        if copy_src or fresh:  # a fresh buffer must be initialised regardless
            buf.copy_(src)
        return buf

    @staticmethod
    def _window_mid(
        met_window: HourlyMetTensors,
        name: str,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Window-midpoint (``t_alpha=0.5``) value of a met channel ``[Z, Y, X]``."""

        start, end = met_window.channel(name)
        return 0.5 * (start.to(device=device, dtype=dtype) + end.to(device=device, dtype=dtype))

    @staticmethod
    def _column_pressure_pa_mid(
        met_window: HourlyMetTensors,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Each column's own pressure on the window's levels, Pa, ``[Z, Y, X]``,
        at the window midpoint (the same ``t_alpha = 0.5`` as ``_window_mid``).

        Terrain-following windows carry it per column
        (``HourlyMetTensors.pressure_hpa``: the hydrostatic reconstruction on
        model levels, the level pressure regridded onto the AGL grid on pressure
        levels). Without it (pressure-level grid without terrain following, and
        synthetic readers) every column's pressure on a level is that level's
        pressure, and ``column_pressure_hpa`` broadcasts
        ``metadata.pressure_level_hpa``. Decision 0025: the density and θ used
        to take the met domain's area-mean pressure per AGL level instead.
        """

        p_start, p_end = met_window.column_pressure_hpa()
        p_start = p_start.to(device=device, dtype=dtype)
        p_end = p_end.to(device=device, dtype=dtype)
        return 0.5 * (p_start + p_end) * 100.0

    def _free_trop_fields(
        self,
        met_window: HourlyMetTensors,
        device: torch.device,
        dtype: torch.dtype,
        frame: HorizontalFrame,
    ) -> tuple[torch.Tensor, GridInterpolationBounds]:
        """Build free-troposphere (σ_w, σ_uv, T_Lw, T_Luv) on the met grid [4, Z, Y, X].

        Uses the gradient-Richardson closure on potential-temperature and wind
        shear computed from true per-column geopotential heights (``height_agl_m``).
        Returns the stacked fields plus the interpolation bounds for sampling them
        per-particle.
        """

        if met_window.height_agl_m is None:
            raise ValueError(
                "HannaScheme free-troposphere turbulence requires HourlyMetTensors.height_agl_m "
                "(3D geopotential height). The met reader provides it; synthetic readers must too."
            )

        height = met_window.height_agl_m.to(device=device, dtype=dtype)  # [Z, Y, X]

        def _build() -> torch.Tensor:
            # Grid-wide FT σ/T_L stack at the window midpoint; cached per window.
            # θ from each column's own pressure (decision 0025).
            pressure_pa = self._column_pressure_pa_mid(met_window, device, dtype)
            t_field = self._window_mid(met_window, "t", device, dtype)
            u_field = self._window_mid(met_window, "u", device, dtype)
            v_field = self._window_mid(met_window, "v", device, dtype)

            theta = potential_temperature(t_field, pressure_pa)
            dtheta_dz = self._vertical_gradient(theta, height)
            n2 = brunt_vaisala_squared(theta, dtheta_dz)

            du_dz = self._vertical_gradient(u_field, height)
            dv_dz = self._vertical_gradient(v_field, height)
            shear_sq = du_dz.pow(2) + dv_dz.pow(2)
            shear_mag = shear_sq.clamp(min=FT_SHEAR_SQ_FLOOR_S2).sqrt()

            ri = gradient_richardson(n2, shear_sq)
            k_z = free_trop_diffusivity(height.clamp(min=Z_MIN_M), shear_mag, ri)
            sigma_w_ft, t_lw_ft = free_trop_sigma_TL(k_z, n2)
            # FT shear turbulence treated as isotropic (documented simplification);
            # the unresolved-mesoscale "meander" horizontal term is separate.
            return torch.stack([sigma_w_ft, sigma_w_ft, t_lw_ft, t_lw_ft], dim=0)

        fields = self._cached_window_field(met_window, "freetrop", _build)
        return fields, self._grid_bounds(met_window, frame)

    def _density_fields(
        self,
        met_window: HourlyMetTensors,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Stack `[ρ, ∂ρ/∂z]` on the met grid `[2, Z, Y, X]` for the
        Stohl-Thomson 1999 density-correction drift term (F2, audit 2026-05-30).

        ρ = p / (R_d · T) on each grid level, with p the column's OWN pressure
        at that level (`_column_pressure_pa_mid`, decision 0025) and T local, both
        at the window midpoint. ∂ρ/∂z is a central finite difference of ρ
        using the true per-column geopotential heights — same machinery as the
        FT closure's wind/θ gradients. Returns the stack ready for trilinear
        sampling per-particle (`_interp_3d_field`) using the shared `_grid_bounds`.
        """

        if met_window.height_agl_m is None:
            raise ValueError(
                "HannaScheme density-correction drift requires HourlyMetTensors.height_agl_m "
                "(3D geopotential height). The met reader provides it; synthetic readers must too."
            )

        height = met_window.height_agl_m.to(device=device, dtype=dtype)  # [Z, Y, X]

        def _build() -> torch.Tensor:
            pressure_pa = self._column_pressure_pa_mid(met_window, device, dtype)
            t_field = self._window_mid(met_window, "t", device, dtype)
            rho = pressure_pa / (R_DRY_AIR_J_KG_K * t_field.clamp(min=1.0))
            drho_dz = self._vertical_gradient(rho, height)
            return torch.stack([rho, drho_dz], dim=0)

        return self._cached_window_field(met_window, "density", _build)

    def _meander_sigma_fields(
        self,
        met_window: HourlyMetTensors,
        device: torch.device,
        dtype: torch.dtype,
        frame: HorizontalFrame,
    ) -> tuple[torch.Tensor, GridInterpolationBounds]:
        """Meander (σ_u, σ_v) on the met grid [2, Z, Y, X] from local wind variability.

        Maryon (1998) "meandering" / FLEXPART (Stohl et al. 2005, §4.5): the
        unresolved-mesoscale wind std-dev is the grid-scale wind variability in the
        neighbourhood of each point times ``meander_coefficient``. Resolution-dependent
        by construction — a finer met grid has smaller local wind variance, so less
        mesoscale energy is left to parameterize.
        """

        r = self.meander_stencil_radius

        def _build() -> torch.Tensor:
            u_field = self._window_mid(met_window, "u", device, dtype)
            v_field = self._window_mid(met_window, "v", device, dtype)
            sigma_u = self.meander_coefficient * self._windowed_std(u_field, r)
            sigma_v = self.meander_coefficient * self._windowed_std(v_field, r)
            return torch.stack([sigma_u, sigma_v], dim=0)

        fields = self._cached_window_field(met_window, "meander", _build)
        return fields, self._grid_bounds(met_window, frame)

    @staticmethod
    def _windowed_std(field: torch.Tensor, radius: int) -> torch.Tensor:
        """Local horizontal std-dev over a (2r+1)² stencil, per level.

        ``field`` is [Z, Y, X]; returns [Z, Y, X]. Edge cells average only the valid
        (non-padded) neighbours, so the variance estimate is well-defined at the grid
        boundary."""

        k = 2 * radius + 1
        x = field.unsqueeze(0)  # [1, Z, Y, X] -> treat Z as the channel axis
        mean = torch.nn.functional.avg_pool2d(
            x, k, stride=1, padding=radius, count_include_pad=False
        )
        mean_sq = torch.nn.functional.avg_pool2d(
            x * x, k, stride=1, padding=radius, count_include_pad=False
        )
        var = (mean_sq - mean.pow(2)).clamp(min=0.0)
        return var.sqrt().squeeze(0)

    @staticmethod
    def _grid_bounds(
        met_window: HourlyMetTensors, frame: HorizontalFrame
    ) -> GridInterpolationBounds:
        """Interpolation bounds for sampling met-grid fields per-particle, with the
        horizontal end points in ``frame`` (``gpu_engine.frame_grid_bounds``).

        Passes the per-level AGL array (F9 fix, audit 2026-05-30) so the
        vertical normalisation done by ``GPUEngine.normalize_particle_coordinates``
        uses a proper piecewise-linear level-index lookup rather than the
        linear-in-AGL approximation that was biased for pressure-level data.
        """

        return frame_grid_bounds(met_window.metadata, frame)

    @staticmethod
    def _vertical_gradient(field: torch.Tensor, height: torch.Tensor) -> torch.Tensor:
        """∂field/∂z over the level axis (axis 0) of a [Z, Y, X] field, using the
        per-column heights. Central differences in the interior, one-sided at the
        top/bottom levels. Robust to the top-to-bottom level ordering since the
        actual heights carry the sign."""

        def _safe(den: torch.Tensor) -> torch.Tensor:
            return torch.where(den.abs() < 1e-6, torch.full_like(den, 1e-6), den)

        grad = torch.empty_like(field)
        grad[1:-1] = (field[2:] - field[:-2]) / _safe(height[2:] - height[:-2])
        grad[0] = (field[1] - field[0]) / _safe(height[1] - height[0])
        grad[-1] = (field[-1] - field[-2]) / _safe(height[-1] - height[-2])
        return grad

    @staticmethod
    def _interp_3d_field(
        fields_stacked: torch.Tensor,
        grid_bounds: GridInterpolationBounds,
        x: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
        engine: GPUEngine,
        level_arr: torch.Tensor | None = None,
        ascending: bool | None = None,
    ) -> torch.Tensor:
        """Trilinear interp of a [C, Z, Y, X] field stack at particle (x, y, z), the
        horizontal position in the frame ``grid_bounds`` is expressed in.
        Returns [C, N]. Reuses the engine's coordinate normalisation so the vertical
        mapping matches the wind advection. ``level_arr`` (dynamic tensor) + ``ascending``
        (constant bool) are threaded to ``normalize_particle_coordinates`` on the compiled
        path so the per-window level VALUES don't trigger recompiles (see that method)."""

        xyz = torch.stack([x, y, z], dim=1)
        xyz_norm = engine.normalize_particle_coordinates(
            xyz, grid_bounds, level_arr=level_arr, ascending=ascending
        )
        grid = xyz_norm.view(1, 1, 1, -1, 3)
        vol = fields_stacked.unsqueeze(0)  # [1, C, Z, Y, X]
        sampled = torch.nn.functional.grid_sample(vol, grid, align_corners=True)
        return sampled.view(fields_stacked.shape[0], -1)

    # ---- Interpolation helpers ----

    @staticmethod
    def _interp_2d_bilinear(
        field_2d: torch.Tensor,
        lon_centers: np.ndarray,
        lat_centers: np.ndarray,
        particle_lon: torch.Tensor,
        particle_lat: torch.Tensor,
    ) -> torch.Tensor:
        """Bilinear interp of `[Y, X]` field at particle (lon, lat) positions."""

        device = field_2d.device
        dtype = field_2d.dtype

        lon0, lon1 = float(lon_centers[0]), float(lon_centers[-1])
        lat0, lat1 = float(lat_centers[0]), float(lat_centers[-1])
        if lon1 == lon0 or lat1 == lat0:
            raise ValueError("Met grid has zero span in lon or lat — cannot interpolate.")

        lon_norm = (2.0 * (particle_lon - lon0) / (lon1 - lon0) - 1.0).clamp(-1.0, 1.0)
        lat_norm = (2.0 * (particle_lat - lat0) / (lat1 - lat0) - 1.0).clamp(-1.0, 1.0)
        grid = (
            torch.stack([lon_norm, lat_norm], dim=-1)
            .view(1, 1, -1, 2)
            .to(device=device, dtype=dtype)
        )

        field = field_2d.unsqueeze(0).unsqueeze(0)
        sampled = torch.nn.functional.grid_sample(field, grid, align_corners=True, mode="bilinear")
        return sampled.view(-1)

    @classmethod
    def _interp_surface_field(
        cls,
        met_window: HourlyMetTensors,
        channel_name: str,
        t_alpha: float,
        x: torch.Tensor,
        y: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
        *,
        frame: HorizontalFrame,
    ) -> torch.Tensor:
        """Interpolate a surface (z-broadcast) channel at particle (x, y) in ``frame``.

        In time: linear for a snapshot; a mean/accumulation channel (decision
        0020) is held at its window-end value, the mean over this window.
        """

        chan_start, chan_end = met_window.channel(channel_name)  # [Z, Y, X] each
        field_start = chan_start[0].to(device=device, dtype=dtype)
        field_end = chan_end[0].to(device=device, dtype=dtype)
        if channel_name in getattr(met_window, "period_channels", ()):
            field_2d = field_end
        else:
            field_2d = field_start * (1.0 - t_alpha) + field_end * t_alpha
        return cls._interp_2d_bilinear(
            field_2d,
            frame.x_from_lon(met_window.metadata.lon),
            frame.y_from_lat(met_window.metadata.lat),
            x,
            y,
        )

    @classmethod
    def _interp_t_at_lowest_level(
        cls,
        met_window: HourlyMetTensors,
        t_alpha: float,
        x: torch.Tensor,
        y: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
        *,
        frame: HorizontalFrame,
    ) -> torch.Tensor:
        """Interpolate temperature at the lowest model level (surface proxy)."""

        t_start, t_end = met_window.channel("t")
        level_agl = met_window.metadata.level
        lowest_idx = int(np.argmin(level_agl))
        field_start = t_start[lowest_idx].to(device=device, dtype=dtype)
        field_end = t_end[lowest_idx].to(device=device, dtype=dtype)
        field_2d = field_start * (1.0 - t_alpha) + field_end * t_alpha
        return cls._interp_2d_bilinear(
            field_2d,
            frame.x_from_lon(met_window.metadata.lon),
            frame.y_from_lat(met_window.metadata.lat),
            x,
            y,
        )
