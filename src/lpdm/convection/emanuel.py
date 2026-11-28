"""Reduced port of FLEXPART's Emanuel & Živković-Rothman (1999) convection.

This is a SIMPLIFIED faithful version of the FLEXPART scheme described in
Forster, Stohl & Seibert (2007, J. Appl. Meteorol. Climatol. 46, 403-422) and
Stohl et al. (2005, ACP 5, 2461-2474) §4.6. The full Emanuel scheme in
``convect43c.f`` is ~3000 FORTRAN lines; this version captures the key physics
in pure torch at a higher level of abstraction:

1. **Trigger.** Surface parcel lifted to LCL + 1 layer; convection fires when
   ``T_v_parcel(LCL+1) ≥ T_v_env(LCL+1) + T_threshold`` with
   ``T_threshold = 0.9 K`` (Forster 2007 Eq 34).

2. **Cloud extent.** Surface parcel continues moist-adiabatically until it
   becomes neutrally buoyant — Level of Neutral Buoyancy (LNB). The cloud
   layer is [LCL, LNB].

3. **Mass-flux matrix `fmass[i, j]`.** A NON-DIVERGENT circulation: a coherent
   boundary-layer → cloud updraft plus compensating environmental subsidence,
   so mass leaving every layer equals mass entering it. The updraft entrains
   from the BL (mass-weighted, total ``M_b``) and detrains across the cloud
   [LCL, LNB] with a linear profile; subsidence returns the same interface flux
   downward. The cloud-base mass flux ``M_b`` closes the scheme via the dilution
   closure (Emanuel 1991): ``M_b ∝ (T_v_parcel(LCL+1) − T_v_env(LCL+1)) · ρ_LCL``.
   See ``compute_mass_flux_matrix``.

4. **Particle redistribution.** FLEXPART ``calcmatrix``/``redist``: apply the
   diagonal closure (each row of ``fmassfrac`` sums to the layer air mass), then
   sample a destination level per particle from the transition matrix — the row
   for forward runs, the transposed COLUMN for backward runs. Drawn via
   cumulative sum + uniform random. Because ``fmass`` is non-divergent, both are
   valid probability distributions and leave a mass-proportional ensemble
   invariant (the well-mixed criterion). See ``_redistribute_particles``.

5. **Backward.** GLIDE runs backward, so the destination is sampled from the
   matrix COLUMN (FLEXPART ``redist`` ``ldirect=-1``) — the adjoint of the
   forward updraft: a particle that is aloft *now* is traced back to the BL air
   it came from. Non-divergence guarantees the well-mixed criterion holds in
   this direction too (Forster 2007 §3).

Documented departures from full Emanuel:
- **Detrainment profile is linear** (``_detrainment_weights``), not the Emanuel
  buoyancy-sorting spectrum (Forster 2007 Eq 35-36). This sets only WHERE the
  updraft deposits mass, not the mass-conservation structure; non-divergence is
  guaranteed by the compensating subsidence regardless of the profile shape.
- **No explicit saturated-downdraft branch** (Emanuel 1991 §4.b); the
  environmental subsidence carries the compensating return flux. Few-percent
  effect on the redistribution profile for deep cumulus.
- The mass-flux closure is a simplified buoyancy-times-density relation
  rather than Emanuel's full quasi-equilibrium closure. Calibrated against
  Forster (2007)'s order-of-magnitude values for typical deep-convection
  cases (M_b ≈ 0.01–0.1 kg/m²/s).

Columns (task A02, decision 0023): every (lon, lat) column of the met window is
diagnosed at once (`compute_lcl_lnb_cape_columns`, `compute_mass_flux_matrices`,
[columns, levels] tensors), and each particle is moved with the matrix of its
nearest column. The single-column functions (`compute_lcl_lnb_cape`,
`compute_mass_flux_matrix`) are the reference the batched ones are tested against.
"""

from __future__ import annotations

import logging
from typing import ClassVar, NamedTuple

import numpy as np
import torch

from lpdm.convection.base import (
    ConvectionScheme,
    ConvectionState,
    register_scheme,
)
from lpdm.gpu_engine import GPUEngine
from lpdm.met_reader import HourlyMetTensors
from lpdm.phase_timers import phase

LOGGER = logging.getLogger(__name__)


# Physical constants (mirroring lpdm.turbulence.hanna to avoid drift).
GRAVITY_M_S2 = 9.80665
R_DRY_AIR_J_KG_K = 287.05
R_VAPOR_J_KG_K = 461.5
EPSILON_RD_RV = R_DRY_AIR_J_KG_K / R_VAPOR_J_KG_K  # ≈ 0.622
C_P_DRY_AIR_J_KG_K = 1005.0
L_V_J_KG = 2.5e6  # latent heat of vapourisation at 0 °C
P0_PA = 100000.0
KAPPA_POISSON = R_DRY_AIR_J_KG_K / C_P_DRY_AIR_J_KG_K  # ≈ 0.2854

# Scheme thresholds (Forster 2007 / typical Emanuel defaults).
TRIGGER_DELTA_TV_K = 0.9  # T_threshold in Forster 2007 Eq 34
MIN_CAPE_J_KG = 50.0  # floor on CAPE before we even consider triggering
CLOUD_TOP_MIN_DEPTH_M = 500.0  # require at least 500 m of cloud depth (i.e. shallow → skip)

# Mass-flux closure (Emanuel 1991, simplified). M_b = CLOSURE_C · ρ_LCL · w_buoy
# where w_buoy = √(2 · CAPE_partial). The Emanuel paper gives much more
# elaborate quasi-equilibrium closures; this is the order-of-magnitude
# approximation that gives M_b ~ 0.01–0.1 kg/m²/s for typical CAPE values.
MASS_FLUX_CLOSURE_C = 0.03
# Cap on the effective updraft velocity used in the closure (m/s). Without a
# cap, CAPE > ~500 J/kg gives w_buoy > 30 m/s and an unrealistic M_b ≫ FLEXPART's
# 0.05–0.5 kg/m²/s. Real updraft peaks are 5–15 m/s. We cap at 5 m/s (a
# representative MEAN-cell peak rather than the rare ~15 m/s extreme).
UPDRAFT_W_MAX_M_S = 5.0

# Numerical floors / ceilings.
P_MIN_PA = 1.0
T_MIN_K = 150.0
Q_MIN_KG_KG = 0.0

# CFL-style cap on the convective mass-flux matrix: no layer may shed more than
# this fraction of its air mass in a single convection event. Keeps the
# diagonal (stay) probability of the redistribution non-negative. Because the
# whole flux matrix is scaled by a single factor, non-divergence (hence the
# well-mixed property) is preserved. FLEXPART enforces the equivalent limit on
# the cloud-base mass flux inside convect43c.f.
CFL_MAX_OUTFLUX_FRAC = 0.9


# ---------------------------------------------------------------------------
# Free-function physics (testable in isolation)
# ---------------------------------------------------------------------------


def saturation_vapor_pressure(t_kelvin: torch.Tensor) -> torch.Tensor:
    """Saturation water-vapour pressure (Pa) using Bolton (1980) Eq. 10.

    Valid roughly between -35 and +35 °C; sufficient for parcel-lift CAPE in
    the lower / mid troposphere where convection initiates.
    """

    t_c = t_kelvin - 273.15
    return 611.2 * torch.exp(17.67 * t_c / (t_c + 243.5))


def specific_humidity_to_vapor_pressure(q: torch.Tensor, p_pa: torch.Tensor) -> torch.Tensor:
    """Convert specific humidity (kg/kg) to vapour pressure (Pa).

    ``e = q·p / (ε + (1−ε)·q)`` with ``ε = R_d/R_v ≈ 0.622``. Inverted from
    the standard ``q = ε·e / (p − (1−ε)·e)``.
    """

    return q * p_pa / (EPSILON_RD_RV + (1.0 - EPSILON_RD_RV) * q)


def vapor_pressure_to_specific_humidity(e_pa: torch.Tensor, p_pa: torch.Tensor) -> torch.Tensor:
    """Convert vapour pressure (Pa) to specific humidity (kg/kg)."""

    return EPSILON_RD_RV * e_pa / (p_pa.clamp(min=P_MIN_PA) - (1.0 - EPSILON_RD_RV) * e_pa)


def virtual_temperature(t_kelvin: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Virtual temperature ``T_v = T · (1 + (1/ε − 1)·q)`` (Wallace & Hobbs Eq 3.59)."""

    return t_kelvin * (1.0 + (1.0 / EPSILON_RD_RV - 1.0) * q)


def lcl_temperature_bolton(
    t_kelvin: torch.Tensor,
    q: torch.Tensor,
    p_pa: torch.Tensor,
) -> torch.Tensor:
    """Lifting condensation level temperature, Bolton (1980) Eq. 22.

    ``T_LCL = 2840 / (3.5·ln T − ln e − 4.805) + 55``, ``e`` in hPa.
    """

    e_pa = specific_humidity_to_vapor_pressure(q, p_pa).clamp(min=1.0)
    e_hpa = e_pa / 100.0
    denom = 3.5 * torch.log(t_kelvin) - torch.log(e_hpa) - 4.805
    return 2840.0 / denom.clamp(min=1e-3) + 55.0


def lcl_pressure_poisson(
    t_kelvin: torch.Tensor,
    t_lcl: torch.Tensor,
    p_pa: torch.Tensor,
) -> torch.Tensor:
    """Pressure at the LCL via Poisson: ``p_LCL = p · (T_LCL / T)^(c_p/R_d)``."""

    exponent = C_P_DRY_AIR_J_KG_K / R_DRY_AIR_J_KG_K
    return p_pa * (t_lcl / t_kelvin.clamp(min=T_MIN_K)).pow(exponent)


def potential_temperature(t_kelvin: torch.Tensor, p_pa: torch.Tensor) -> torch.Tensor:
    """Dry potential temperature ``θ = T·(p0/p)^κ``."""

    return t_kelvin * (P0_PA / p_pa.clamp(min=P_MIN_PA)).pow(KAPPA_POISSON)


def equivalent_potential_temperature(
    t_kelvin: torch.Tensor,
    q: torch.Tensor,
    p_pa: torch.Tensor,
) -> torch.Tensor:
    """Equivalent potential temperature ``θ_e ≈ θ · exp(L_v · q_sat / (c_p · T_LCL))``.

    Approximation: q at the parcel level equals q_sat at the LCL temperature
    (valid for a parcel lifted dry-adiabatically to its LCL). Sufficient for
    the parcel-lift CAPE used by the trigger and matrix here; the full Bolton
    (1980) Eq 43 adds higher-order corrections we omit (~1 K error).
    """

    t_lcl = lcl_temperature_bolton(t_kelvin, q, p_pa)
    theta = potential_temperature(t_kelvin, p_pa)
    return theta * torch.exp(L_V_J_KG * q / (C_P_DRY_AIR_J_KG_K * t_lcl.clamp(min=T_MIN_K)))


def saturation_specific_humidity(t_kelvin: torch.Tensor, p_pa: torch.Tensor) -> torch.Tensor:
    """Specific humidity at saturation: ``q_sat = ε·e_sat(T) / (p − (1−ε)·e_sat(T))``."""

    e_sat = saturation_vapor_pressure(t_kelvin)
    return vapor_pressure_to_specific_humidity(e_sat, p_pa)


def lift_parcel_moist_pseudo_adiabatic(
    theta_e_const: torch.Tensor,
    p_target_pa: torch.Tensor,
    *,
    max_iter: int = 30,
    tol_k: float = 0.01,
    t_min_k: float = 180.0,
    t_max_k: float = 350.0,
) -> torch.Tensor:
    """Solve for the temperature at ``p_target`` along the moist (pseudo-)adiabat
    with conserved θ_e.

    θ_e of a parcel at (T, p) is preserved along its moist adiabat: find ``T``
    such that ``θ_e(T, q_sat(T, p), p) == theta_e_const``. Bisection on T in
    [t_min_k, t_max_k] — robust under Clausius-Clapeyron's strong nonlinearity
    (fixed-point and Newton both fail to converge near deep-convection
    temperatures because dθ_e/dT amplifies rapidly with T).

    The default bracket [180, 350] K covers everything from polar-tropopause
    to tropical-surface; 30 bisection halvings shrink it to 170/2^30 ≈ 1.6e-7 K,
    far below `tol_k`. `theta_e_const` broadcasts against `p_target_pa`, so this
    lifts a WHOLE column (or any batch) in one call.

    The loop runs a FIXED `max_iter` iterations with NO per-iteration convergence
    check: that check was a device->host sync every iteration, and called per level
    inside `compute_lcl_lnb_cape` it dominated the convection cost on GPU (~n_lev x
    max_iter syncs per call). A fixed count is sync-free and, at 30 halvings, more
    than converged. `tol_k` is retained only for API compatibility.
    """

    del tol_k  # no early-exit convergence check (it was a per-iteration host sync)
    t_lo = torch.full_like(p_target_pa, t_min_k)
    t_hi = torch.full_like(p_target_pa, t_max_k)
    for _ in range(max_iter):
        t_mid = 0.5 * (t_lo + t_hi)
        q_sat = saturation_specific_humidity(t_mid, p_target_pa)
        theta_e_mid = equivalent_potential_temperature(t_mid, q_sat, p_target_pa)
        too_high = theta_e_mid > theta_e_const
        t_hi = torch.where(too_high, t_mid, t_hi)
        t_lo = torch.where(too_high, t_lo, t_mid)
    return 0.5 * (t_lo + t_hi)


# ---------------------------------------------------------------------------
# Mass-flux matrix (Forster 2007 Eq 35-36)
# ---------------------------------------------------------------------------


def _layer_masses_per_area(
    pressure_pa: torch.Tensor,
) -> torch.Tensor:
    """Mass per unit area for each pressure layer: m_i = (p_bot − p_top) / g.

    ``pressure_pa`` is the per-level pressure [..., Z] (one column [Z], or a
    batch of columns [C, Z]). Layer i lies between levels i and i+1 (or i-1 and
    i for the descending-pressure convention). We use a half-bracket: layer mass
    at level i is `|p_{i+1} - p_{i-1}|·0.5 / g` for interior, and the half-layer
    for the top/bottom. Returns the shape of ``pressure_pa``.
    """

    p = pressure_pa
    dp = torch.zeros_like(p)
    dp[..., 1:-1] = 0.5 * (p[..., :-2] - p[..., 2:]).abs()
    dp[..., 0] = (p[..., 0] - p[..., 1]).abs() * 0.5
    dp[..., -1] = (p[..., -2] - p[..., -1]).abs() * 0.5
    return dp / GRAVITY_M_S2


def compute_lcl_lnb_cape(
    t_env: torch.Tensor,
    q_env: torch.Tensor,
    pressure_pa: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Identify LCL and LNB indices for a surface-parcel lift, plus CAPE.

    Inputs are 1-D column profiles [Z]. Index 0 is assumed to be the SURFACE
    level (the parcel's starting state). The function returns:

    - ``i_lcl``: level index above which the parcel is saturated. ``-1`` if
      the parcel never reaches saturation within the column.
    - ``i_lnb``: level index at which the buoyant parcel becomes neutrally
      buoyant again (i.e. the cloud top, also called LFC / LNB depending on
      convention). ``-1`` if no neutral level found.
    - ``cape``: integrated buoyancy from LFC to LNB, in J/kg.
    - ``parcel_minus_env_tv_at_lcl_plus_1``: ``T_v_parcel − T_v_env`` at level
      LCL+1. Drives the Forster (2007) Eq 34 trigger.

    Uses θ_e conservation along the moist adiabat: at every level above LCL
    we solve `lift_parcel_moist_pseudo_adiabatic` for the parcel temperature.
    """

    device, dtype = t_env.device, t_env.dtype
    n_lev = t_env.numel()

    # Parcel state at the surface (level 0).
    t_sfc = t_env[0]
    q_sfc = q_env[0]
    p_sfc = pressure_pa[0]

    t_lcl = lcl_temperature_bolton(
        t_sfc.unsqueeze(0), q_sfc.unsqueeze(0), p_sfc.unsqueeze(0)
    ).squeeze(0)
    p_lcl = lcl_pressure_poisson(
        t_sfc.unsqueeze(0), t_lcl.unsqueeze(0), p_sfc.unsqueeze(0)
    ).squeeze(0)

    # θ_e of the parcel (conserved along the moist adiabat).
    theta_e_parcel = equivalent_potential_temperature(
        t_sfc.unsqueeze(0),
        q_sfc.unsqueeze(0),
        p_sfc.unsqueeze(0),
    ).squeeze(0)

    # Lift the parcel through every level, VECTORISED (no per-level Python loop /
    # per-iteration host sync — that was the dominant convection cost on GPU).
    # Dry adiabat at/below LCL (θ_parcel = θ_sfc); moist adiabat above (θ_e
    # conserved). The moist lift is evaluated for the whole column at once
    # (`theta_e_parcel` broadcasts against `pressure_pa`) and discarded below the
    # LCL by the mask — cheap vs the ~n_lev × bisection-iter syncs it replaces.
    theta_sfc = potential_temperature(t_sfc.unsqueeze(0), p_sfc.unsqueeze(0)).squeeze(0)
    dry_t = theta_sfc * (pressure_pa / P0_PA).pow(KAPPA_POISSON)
    moist_t = lift_parcel_moist_pseudo_adiabatic(theta_e_parcel, pressure_pa)
    t_parcel = torch.where(pressure_pa >= p_lcl, dry_t, moist_t)

    # Virtual temperatures (parcel uses its own q_sat above LCL; environment
    # uses its actual q).
    tv_env = virtual_temperature(t_env, q_env)
    q_parcel = torch.where(
        pressure_pa >= p_lcl,
        torch.full_like(t_parcel, float(q_sfc)),
        saturation_specific_humidity(t_parcel, pressure_pa),
    )
    tv_parcel = virtual_temperature(t_parcel, q_parcel)

    buoyancy = (tv_parcel - tv_env) / tv_env.clamp(min=T_MIN_K)

    # LCL index: smallest k with p_k < p_lcl.
    below_lcl_mask = pressure_pa >= p_lcl
    if not bool((~below_lcl_mask).any()):
        i_lcl = torch.tensor(-1, dtype=torch.long, device=device)
    else:
        i_lcl = int((~below_lcl_mask).nonzero()[0].item())
        i_lcl = torch.tensor(i_lcl, dtype=torch.long, device=device)

    # Trigger diagnostic: T_v_parcel − T_v_env at level LCL+1.
    if int(i_lcl) >= 0 and int(i_lcl) + 1 < n_lev:
        i_test = int(i_lcl) + 1
        dtv_lcl_plus_1 = tv_parcel[i_test] - tv_env[i_test]
    else:
        dtv_lcl_plus_1 = torch.tensor(-1.0, device=device, dtype=dtype)

    # LNB: first level above LCL where buoyancy crosses zero from positive.
    if int(i_lcl) < 0:
        i_lnb = torch.tensor(-1, dtype=torch.long, device=device)
        cape = torch.tensor(0.0, device=device, dtype=dtype)
    else:
        i_lcl_int = int(i_lcl)
        # CAPE integration: ∫_LFC^LNB g · buoyancy dz, approximated as a sum
        # over layers above LCL where buoyancy > 0.
        # Use AGL height differences for dz; we don't have them here, so use
        # the hypsometric estimate: dz ≈ R_d·T̄·d(ln p)/g.
        dz = torch.zeros(n_lev, device=device, dtype=dtype)
        dz[1:] = (
            R_DRY_AIR_J_KG_K
            * 0.5
            * (t_env[:-1] + t_env[1:])
            * (
                torch.log(pressure_pa[:-1].clamp(min=P_MIN_PA))
                - torch.log(pressure_pa[1:].clamp(min=P_MIN_PA))
            )
            / GRAVITY_M_S2
        )
        # Positive-buoyancy band above LCL.
        positive = buoyancy.clamp(min=0.0)
        positive[:i_lcl_int] = 0.0
        cape = (GRAVITY_M_S2 * positive * dz).sum()
        # LNB = first level above LCL where buoyancy goes ≤ 0.
        above_lcl_neg = (buoyancy <= 0.0) & (torch.arange(n_lev, device=device) > i_lcl_int)
        if bool(above_lcl_neg.any()):
            i_lnb = int(above_lcl_neg.nonzero()[0].item())
            i_lnb = torch.tensor(i_lnb, dtype=torch.long, device=device)
        else:
            # Parcel remains buoyant through the top of the column; cap at top.
            i_lnb = torch.tensor(n_lev - 1, dtype=torch.long, device=device)

    return i_lcl, i_lnb, cape, dtv_lcl_plus_1


def _detrainment_weights(
    i_lcl: int,
    i_lnb: int,
    n_lev: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Relative detrainment weight per cloud level [LCL, LNB], linear decay from
    the cloud base to the cloud top (1 at LCL → 0 at LNB).

    The full Emanuel scheme detrains according to a buoyancy-sorting spectrum
    (Forster 2007 Eq 35-36); we use the documented linear ``M``-profile
    simplification (departure #4). It sets only WHERE the coherent updraft
    deposits mass, not the mass-conservation structure — non-divergence (hence
    the well-mixed property) is guaranteed by the compensating subsidence added
    in ``compute_mass_flux_matrix`` regardless of this profile's shape.
    """

    w = torch.zeros(n_lev, device=device, dtype=dtype)
    depth = i_lnb - i_lcl
    if depth <= 0:
        return w
    for k in range(i_lcl, i_lnb + 1):
        w[k] = 1.0 - (k - i_lcl) / depth
    return w


def compute_mass_flux_matrix(
    t_env: torch.Tensor,
    q_env: torch.Tensor,
    pressure_pa: torch.Tensor,
    i_lcl: int,
    i_lnb: int,
    m_b: float,
) -> torch.Tensor:
    """Non-divergent convective mass-flux matrix ``fmass[i, j]`` (off-diagonal
    mass transported from layer ``i`` to layer ``j`` per convection event).

    Built as a coherent boundary-layer → cloud updraft plus **compensating
    environmental subsidence**, so that the total mass leaving every layer equals
    the total entering it (``Σ_j fmass[i,j] == Σ_j fmass[j,i]``). That
    non-divergence is what makes the particle redistribution preserve a
    mass-proportional (well-mixed) distribution in BOTH time directions — the
    pre-2026-07-02 matrix had updraft only (no subsidence) and violated it
    (Finding 2 of the physics review).

    Construction (levels ascending, surface at index 0; cloud = [LCL, LNB]):

    - **Entrainment** ``e[i]``: the boundary layer ``[0, LCL)`` feeds the updraft,
      shared across BL layers by air mass so the TOTAL entrained = ``m_b`` (the
      cloud-base flux). Fixes Finding 3 — previously each BL layer sourced the
      full ``m_b``, over-venting the BL by a factor of the layer count.
    - **Detrainment** ``d[j]``: the updraft deposits mass across the cloud with a
      linear-decay profile (``_detrainment_weights``), TOTAL = ``m_b``.
    - **Direct updraft** ``U[i,j] = e[i]·d[j]/m_b`` (BL source → cloud, non-local:
      this is the coherent deep-lofting transport).
    - **Compensating subsidence**: the net upward flux across the interface below
      layer ``k+1`` is ``Φ[k] = Σ_{i≤k} e[i] − Σ_{j≤k} d[j]`` (≥ 0); the
      environment sinks at the same rate, moving mass from layer ``k+1`` down to
      ``k`` (``fmass[k+1, k] += Φ[k]``).

    ``m_b`` is the cloud-base mass flux already multiplied by the event interval
    (units kg/m²). The whole matrix is finally scaled down if any layer would
    shed more than ``CFL_MAX_OUTFLUX_FRAC`` of its mass (a single scalar → keeps
    non-divergence). Returns zeros (no transport) when there is no cloud.

    The caller (``_redistribute_particles``) applies the FLEXPART diagonal
    closure and samples destinations from the matrix row (forward) or column
    (backward) — GLIDE runs backward.
    """

    device, dtype = t_env.device, t_env.dtype
    n_lev = t_env.numel()
    fmass = torch.zeros((n_lev, n_lev), device=device, dtype=dtype)

    if i_lcl < 0 or i_lnb <= i_lcl or m_b <= 0.0:
        return fmass

    m = _layer_masses_per_area(pressure_pa)  # [Z] layer air mass per unit area

    # Entrainment: BL [0, LCL) mass-weighted so Σ e = m_b (Finding 3).
    e = torch.zeros(n_lev, device=device, dtype=dtype)
    bl_total = m[:i_lcl].sum().clamp(min=1e-12)
    e[:i_lcl] = m_b * m[:i_lcl] / bl_total

    # Detrainment: cloud [LCL, LNB] linear profile so Σ d = m_b.
    w = _detrainment_weights(i_lcl, i_lnb, n_lev, device, dtype)
    w_total = w.sum().clamp(min=1e-12)
    d = m_b * w / w_total

    # Coherent updraft transfers (BL source -> cloud detrainment). row_i sum = e[i],
    # col_j sum = d[j].
    fmass = fmass + torch.outer(e, d) / m_b

    # Compensating subsidence on the sub-diagonal: Φ[k] from layer k+1 down to k.
    phi = (torch.cumsum(e, dim=0) - torch.cumsum(d, dim=0))[:-1].clamp(min=0.0)
    fmass = fmass + torch.diag(phi, -1)

    # CFL: cap the per-layer outflux fraction (single scalar scale -> preserves
    # non-divergence and keeps the redistribution's stay-probability >= 0).
    max_out_frac = (fmass.sum(dim=1) / m.clamp(min=1e-12)).max()
    if float(max_out_frac) > CFL_MAX_OUTFLUX_FRAC:
        fmass = fmass * (CFL_MAX_OUTFLUX_FRAC / max_out_frac)

    return fmass


# ---------------------------------------------------------------------------
# Every column at once (task A02)
# ---------------------------------------------------------------------------
#
# The functions below are the column-batched forms of `compute_lcl_lnb_cape`,
# the closure in `EmanuelReducedConvection` and `compute_mass_flux_matrix`.
# They take profiles of shape [C, Z] (C columns, Z levels ascending from the
# surface) and do the same arithmetic, element by element, with no Python loop
# over columns or levels. The single-column functions above stay as the
# reference: the tests check that every column of the batched result equals
# them.


def compute_lcl_lnb_cape_columns(
    t_env: torch.Tensor,
    q_env: torch.Tensor,
    pressure_pa: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """`compute_lcl_lnb_cape` for C columns at once.

    Inputs are [C, Z] profiles with the surface at level index 0. Returns
    ``(i_lcl, i_lnb, cape, dtv_lcl_plus_1)``, each of shape [C], with the same
    meaning and the same ``-1`` conventions as the single-column function.
    """

    n_col, n_lev = t_env.shape
    device = t_env.device
    k = torch.arange(n_lev, device=device)

    t_sfc = t_env[:, 0]
    q_sfc = q_env[:, 0]
    p_sfc = pressure_pa[:, 0]

    t_lcl = lcl_temperature_bolton(t_sfc, q_sfc, p_sfc)
    p_lcl = lcl_pressure_poisson(t_sfc, t_lcl, p_sfc)
    theta_e_parcel = equivalent_potential_temperature(t_sfc, q_sfc, p_sfc)

    # Dry adiabat at/below the LCL, moist (θ_e conserved) above; the moist lift
    # is solved for every level of every column in one bisection.
    theta_sfc = potential_temperature(t_sfc, p_sfc)
    dry_t = theta_sfc.unsqueeze(1) * (pressure_pa / P0_PA).pow(KAPPA_POISSON)
    moist_t = lift_parcel_moist_pseudo_adiabatic(theta_e_parcel.unsqueeze(1), pressure_pa)
    below_lcl_mask = pressure_pa >= p_lcl.unsqueeze(1)
    t_parcel = torch.where(below_lcl_mask, dry_t, moist_t)

    tv_env = virtual_temperature(t_env, q_env)
    q_parcel = torch.where(
        below_lcl_mask,
        q_sfc.unsqueeze(1).expand_as(t_parcel),
        saturation_specific_humidity(t_parcel, pressure_pa),
    )
    tv_parcel = virtual_temperature(t_parcel, q_parcel)
    buoyancy = (tv_parcel - tv_env) / tv_env.clamp(min=T_MIN_K)

    # LCL index: first level with p < p_lcl (argmax returns the first maximum).
    above_lcl = ~below_lcl_mask
    has_lcl = above_lcl.any(dim=1)
    i_lcl = torch.where(
        has_lcl,
        above_lcl.to(torch.int8).argmax(dim=1),
        torch.full((n_col,), -1, dtype=torch.long, device=device),
    )

    # Trigger diagnostic at LCL+1 (-1 K when it does not exist).
    i_test = (i_lcl + 1).clamp(min=0, max=n_lev - 1).unsqueeze(1)
    dtv_at_test = (tv_parcel.gather(1, i_test) - tv_env.gather(1, i_test)).squeeze(1)
    has_test = (i_lcl >= 0) & (i_lcl + 1 < n_lev)
    dtv_lcl_plus_1 = torch.where(has_test, dtv_at_test, torch.full_like(dtv_at_test, -1.0))

    # CAPE: Σ g·max(B, 0)·dz over levels at and above the LCL, with the
    # hypsometric layer thickness (same expression as the single column).
    dz = torch.zeros_like(t_env)
    dz[:, 1:] = (
        R_DRY_AIR_J_KG_K
        * 0.5
        * (t_env[:, :-1] + t_env[:, 1:])
        * (
            torch.log(pressure_pa[:, :-1].clamp(min=P_MIN_PA))
            - torch.log(pressure_pa[:, 1:].clamp(min=P_MIN_PA))
        )
        / GRAVITY_M_S2
    )
    positive = buoyancy.clamp(min=0.0)
    positive = torch.where(k.unsqueeze(0) < i_lcl.unsqueeze(1), 0.0, positive)
    cape = (GRAVITY_M_S2 * positive * dz).sum(dim=1)
    cape = torch.where(has_lcl, cape, torch.zeros_like(cape))

    # LNB: first level above the LCL with B <= 0; the column top if none.
    above_lcl_neg = (buoyancy <= 0.0) & (k.unsqueeze(0) > i_lcl.unsqueeze(1))
    i_lnb = torch.where(
        above_lcl_neg.any(dim=1),
        above_lcl_neg.to(torch.int8).argmax(dim=1),
        torch.full((n_col,), n_lev - 1, dtype=torch.long, device=device),
    )
    i_lnb = torch.where(has_lcl, i_lnb, torch.full_like(i_lnb, -1))
    return i_lcl, i_lnb, cape, dtv_lcl_plus_1


def cloud_base_mass_flux_columns(
    t_env: torch.Tensor,
    pressure_pa: torch.Tensor,
    i_lcl: torch.Tensor,
    cape: torch.Tensor,
    closure_c: float,
) -> torch.Tensor:
    """Cloud-base mass flux ``M_b = c·ρ_LCL·min(√(2·CAPE), w_max)`` per column,
    kg m⁻² s⁻¹, shape [C], in float64 (the closure was evaluated in Python
    floats before A02; float64 keeps the batched result identical).

    ``ρ_LCL`` uses the environment's temperature and pressure at the LCL level.
    Columns without an LCL get the value at level 0; the caller masks them.
    """

    idx = i_lcl.clamp(min=0).unsqueeze(1)
    t_lcl = t_env.gather(1, idx).squeeze(1)
    p_lcl = pressure_pa.gather(1, idx).squeeze(1)
    rho_lcl = p_lcl / (R_DRY_AIR_J_KG_K * t_lcl.clamp(min=T_MIN_K))
    w_buoy = torch.sqrt(2.0 * cape.double().clamp(min=0.0)).clamp(max=UPDRAFT_W_MAX_M_S)
    return closure_c * rho_lcl.double() * w_buoy


def compute_mass_flux_matrices(
    pressure_pa: torch.Tensor,
    i_lcl: torch.Tensor,
    i_lnb: torch.Tensor,
    m_b: torch.Tensor,
) -> torch.Tensor:
    """`compute_mass_flux_matrix` for C columns at once: returns [C, Z, Z].

    ``pressure_pa`` is [C, Z] (ascending from the surface); ``i_lcl``, ``i_lnb``
    are [C] level indices and ``m_b`` [C] the cloud-base mass flux already
    multiplied by the event interval (kg/m²). A column with no cloud
    (``i_lcl < 0`` or ``i_lnb <= i_lcl``) or ``m_b <= 0`` gets a zero matrix.
    Same construction as the single column: mass-weighted boundary-layer
    entrainment, linear detrainment over [LCL, LNB], their outer product, the
    compensating subsidence on the sub-diagonal, and one CFL scale per column.
    """

    n_col, n_lev = pressure_pa.shape
    device, dtype = pressure_pa.device, pressure_pa.dtype
    k = torch.arange(n_lev, device=device)

    m_b = m_b.to(dtype)
    valid = (i_lcl >= 0) & (i_lnb > i_lcl) & (m_b > 0.0)
    lcl = i_lcl.clamp(min=0).unsqueeze(1)
    lnb = i_lnb.unsqueeze(1)
    mb = torch.where(valid, m_b, torch.ones_like(m_b)).unsqueeze(1)  # [C, 1]

    m = _layer_masses_per_area(pressure_pa)  # [C, Z]

    # Entrainment: BL [0, LCL), mass-weighted so Σ e = m_b.
    in_bl = (k.unsqueeze(0) < lcl) & valid.unsqueeze(1)
    m_bl = torch.where(in_bl, m, torch.zeros_like(m))
    bl_total = m_bl.sum(dim=1, keepdim=True).clamp(min=1e-12)
    e = mb * m_bl / bl_total

    # Detrainment: linear 1 → 0 over [LCL, LNB], Σ d = m_b. The weight is
    # formed in float64, as the single-column loop formed it in Python floats.
    depth = (lnb - lcl).clamp(min=1).double()
    w64 = 1.0 - (k.unsqueeze(0) - lcl).double() / depth
    in_cloud = (k.unsqueeze(0) >= lcl) & (k.unsqueeze(0) <= lnb) & valid.unsqueeze(1)
    w = torch.where(in_cloud, w64.to(dtype), torch.zeros((), dtype=dtype, device=device))
    w_total = w.sum(dim=1, keepdim=True).clamp(min=1e-12)
    d = mb * w / w_total

    # Coherent updraft (BL source → cloud) and compensating subsidence.
    fmass = e.unsqueeze(2) * d.unsqueeze(1) / mb.unsqueeze(2)
    phi = (torch.cumsum(e, dim=1) - torch.cumsum(d, dim=1))[:, :-1].clamp(min=0.0)
    fmass = fmass + torch.diag_embed(phi, offset=-1)

    # CFL: one scale per column (keeps each column non-divergent).
    max_out_frac = (fmass.sum(dim=2) / m.clamp(min=1e-12)).max(dim=1).values
    scale = torch.where(
        max_out_frac.double() > CFL_MAX_OUTFLUX_FRAC,
        max_out_frac.reciprocal() * CFL_MAX_OUTFLUX_FRAC,
        torch.ones_like(max_out_frac),
    )
    scale = torch.where(valid, scale, torch.zeros_like(scale))
    return fmass * scale.view(n_col, 1, 1)


def backward_transition_matrices(fmass: torch.Tensor, layer_mass: torch.Tensor) -> torch.Tensor:
    """Per-column transition probabilities for GLIDE's backward runs, [C, Z, Z].

    FLEXPART ``calcmatrix``/``redist`` with ``ldirect=-1``: the diagonal closure
    makes each row of ``fmassfrac`` sum to the layer mass, and a particle in
    layer i moves to layer j with probability ``fmassfrac[j, i] / m_i`` (the
    matrix column). ``layer_mass`` is [C, Z].
    """

    fmassfrac = fmass + torch.diag_embed((layer_mass - fmass.sum(dim=2)).clamp(min=0.0))
    trans = fmassfrac.transpose(1, 2)
    return (trans / layer_mass.clamp(min=1e-12).unsqueeze(2)).clamp(min=0.0)


def nearest_column_index(
    lon: torch.Tensor,
    lat: torch.Tensor,
    grid_lon: np.ndarray,
    grid_lat: np.ndarray,
) -> torch.Tensor:
    """Flat index ``iy * n_lon + ix`` of the meteorology column nearest each
    particle, on the regular grid the advection interpolates on (either axis may
    be descending). Particles beyond the edge get the edge column."""

    def nearest(vals: torch.Tensor, grid: np.ndarray) -> torch.Tensor:
        n = int(grid.size)
        if n == 1:
            return torch.zeros(vals.shape, dtype=torch.long, device=vals.device)
        first, last = float(grid[0]), float(grid[-1])
        frac = (vals.double() - first) / (last - first) * (n - 1)
        return torch.round(frac).clamp(0, n - 1).to(torch.long)

    ix = nearest(lon, np.asarray(grid_lon))
    iy = nearest(lat, np.asarray(grid_lat))
    return iy * int(np.asarray(grid_lon).size) + ix


class ColumnConvection(NamedTuple):
    """Per-column convection diagnosis for one met window (task A02).

    Profiles are [C, Z] with the surface at level 0; per-column values are [C].
    ``m_b`` is the cloud-base mass flux in kg m⁻² s⁻¹ (float64, zero where the
    column does not fire); one event moves ``m_b · interval_s`` per unit area.
    """

    t: torch.Tensor
    q: torch.Tensor
    pressure_pa: torch.Tensor
    height_m: torch.Tensor
    ascending: bool
    i_lcl: torch.Tensor
    i_lnb: torch.Tensor
    cape: torch.Tensor
    dtv_lcl_plus_1: torch.Tensor
    cloud_depth_m: torch.Tensor
    fires: torch.Tensor
    m_b: torch.Tensor
    interval_s: float


def convection_interval_seconds(met_window: HourlyMetTensors) -> float:
    """Seconds between convection events: the met window's width.

    The runtime fires convection once per met window, and the reader makes
    every window one source time step wide (``reader.met_interval_seconds``:
    3,600 s on hourly ERA5, 10,800 s on a 3-hourly store), so this follows the
    store's cadence rather than assuming an hour.
    """

    interval_s = (met_window.metadata.time_end - met_window.metadata.time_start).total_seconds()
    if interval_s <= 0:
        raise ValueError(
            "Met window has non-positive duration "
            f"({met_window.metadata.time_start} -> {met_window.metadata.time_end}); "
            "cannot scale the convective mass flux."
        )
    return float(interval_s)


# ---------------------------------------------------------------------------
# Scheme class
# ---------------------------------------------------------------------------


@register_scheme
class EmanuelReducedConvection(ConvectionScheme):
    """Reduced port of FLEXPART's Emanuel & Živković-Rothman convection scheme.

    See module docstring for the algorithm and documented departures from the
    full Emanuel implementation. Configure via the ``HannaScheme``-style
    constructor kwargs (`closure_c`, `trigger_dtv_k`, etc.) wired through
    ``ConvectionConfig``.
    """

    name: ClassVar[str] = "emanuel_reduced"

    def __init__(
        self,
        *,
        closure_c: float = MASS_FLUX_CLOSURE_C,
        trigger_dtv_k: float = TRIGGER_DELTA_TV_K,
        min_cape_j_kg: float = MIN_CAPE_J_KG,
        min_cloud_depth_m: float = CLOUD_TOP_MIN_DEPTH_M,
    ) -> None:
        """Construct the scheme.

        Args:
                closure_c: cloud-base mass-flux closure constant. Higher values →
                        more aggressive convective redistribution. Default 0.03 gives
                        `M_b ~ 0.01–0.1 kg/m²/s` for typical CAPE.
                trigger_dtv_k: buoyancy excess (K) at LCL+1 required to trigger
                        convection (Forster 2007 Eq 34). Default 0.9 K matches FLEXPART.
                min_cape_j_kg: minimum CAPE (J/kg) below which convection never
                        fires regardless of the LCL+1 buoyancy check. Default 50 J/kg.
                min_cloud_depth_m: skip shallow convection (cloud depth < this).
                        Default 500 m — the scheme is for DEEP convection only.
        """

        if closure_c <= 0:
            raise ValueError("closure_c must be > 0")
        if trigger_dtv_k < 0:
            raise ValueError("trigger_dtv_k must be >= 0")
        if min_cape_j_kg < 0:
            raise ValueError("min_cape_j_kg must be >= 0")
        if min_cloud_depth_m < 0:
            raise ValueError("min_cloud_depth_m must be >= 0")
        self.closure_c = float(closure_c)
        self.trigger_dtv_k = float(trigger_dtv_k)
        self.min_cape_j_kg = float(min_cape_j_kg)
        self.min_cloud_depth_m = float(min_cloud_depth_m)
        # Per-instance state for logging (avoid spamming the log).
        self._last_log_t_alpha: float | None = None

    def required_met_keys(self) -> tuple[str, ...]:
        # Specific humidity is the only key not in the baseline already (t/sp
        # are baseline and z/z_sfc come from the reader's derivation set).
        return ("t", "q")

    # ----- Public API -----

    def column_state(
        self,
        met_window: HourlyMetTensors,
        *,
        t_alpha: float,
        device: torch.device,
        dtype: torch.dtype,
    ) -> ColumnConvection:
        """Parcel lift, trigger and closure for every (lon, lat) column of the window.

        Profiles are the window's own columns, time-interpolated at ``t_alpha``
        (temperature, humidity and, when the reader supplies it, pressure),
        reordered so level 0 is the surface. No Python loop over columns.
        """

        if met_window.height_agl_m is None:
            raise ValueError(
                "EmanuelReducedConvection requires HourlyMetTensors.height_agl_m "
                "(3D geopotential height). The met reader provides it; synthetic "
                "readers must too."
            )
        interval_s = convection_interval_seconds(met_window)

        def columns(field: torch.Tensor) -> torch.Tensor:
            # [Z, Y, X] -> [C, Z], column index c = iy * n_lon + ix.
            n_lev = field.shape[0]
            return field.reshape(n_lev, -1).transpose(0, 1)

        t_start, t_end = met_window.channel("t")
        q_start, q_end = met_window.channel("q")
        t_field = (
            t_start.to(device=device, dtype=dtype) * (1.0 - t_alpha)
            + t_end.to(device=device, dtype=dtype) * t_alpha
        )
        q_field = (
            q_start.to(device=device, dtype=dtype) * (1.0 - t_alpha)
            + q_end.to(device=device, dtype=dtype) * t_alpha
        )
        p_start, p_end = met_window.column_pressure_hpa()
        if met_window.pressure_hpa is None:
            # Level pressures, the same in every column and both hours.
            p_field = p_start.to(device=device, dtype=dtype)
        else:
            p_field = (
                p_start.to(device=device, dtype=dtype) * (1.0 - t_alpha)
                + p_end.to(device=device, dtype=dtype) * t_alpha
            )
        height_field = met_window.height_agl_m.to(device=device, dtype=dtype)

        t_col = columns(t_field)
        q_col = columns(q_field).clamp(min=Q_MIN_KG_KG)
        pressure_pa = columns(p_field) * 100.0
        height_col = columns(height_field)

        # Surface at index 0. The level order is the same in every column.
        ascending = bool(height_col[:, -1].mean() > height_col[:, 0].mean())
        if not ascending:
            t_col = t_col.flip(1)
            q_col = q_col.flip(1)
            pressure_pa = pressure_pa.flip(1)
            height_col = height_col.flip(1)
        t_col = t_col.contiguous()
        q_col = q_col.contiguous()
        pressure_pa = pressure_pa.contiguous()
        height_col = height_col.contiguous()

        with phase("convection.lift"):
            i_lcl, i_lnb, cape, dtv = compute_lcl_lnb_cape_columns(t_col, q_col, pressure_pa)

        # Trigger (Forster 2007 Eq 34, CAPE floor, deep cloud only). Thresholds
        # are compared in float64, as the single-column code compared Python floats.
        n_lev = t_col.shape[1]
        cloud_depth_m = (
            height_col.gather(1, i_lnb.clamp(min=0, max=n_lev - 1).unsqueeze(1))
            - height_col.gather(1, i_lcl.clamp(min=0, max=n_lev - 1).unsqueeze(1))
        ).squeeze(1)
        fires = (
            (i_lcl >= 0)
            & (i_lnb > i_lcl)
            & (cape.double() >= self.min_cape_j_kg)
            & (dtv.double() >= self.trigger_dtv_k)
            & (cloud_depth_m.double() >= self.min_cloud_depth_m)
        )
        m_b = cloud_base_mass_flux_columns(t_col, pressure_pa, i_lcl, cape, self.closure_c)
        m_b = torch.where(fires, m_b, torch.zeros_like(m_b))
        return ColumnConvection(
            t=t_col,
            q=q_col,
            pressure_pa=pressure_pa,
            height_m=height_col,
            ascending=ascending,
            i_lcl=i_lcl,
            i_lnb=i_lnb,
            cape=cape,
            dtv_lcl_plus_1=dtv,
            cloud_depth_m=cloud_depth_m,
            fires=fires,
            m_b=m_b,
            interval_s=interval_s,
        )

    def column_mass_flux_matrices(self, cols: ColumnConvection) -> torch.Tensor:
        """The mass-flux matrix of every column, [C, Z, Z] (zeros where no
        convection fires). For diagnostics and tests; the runtime builds only
        the columns that fire and hold active particles."""

        return compute_mass_flux_matrices(
            cols.pressure_pa,
            cols.i_lcl,
            cols.i_lnb,
            (cols.m_b * cols.interval_s).to(cols.pressure_pa.dtype),
        )

    def maybe_convect(
        self,
        particles: torch.Tensor,
        state: ConvectionState,
        met_window: HourlyMetTensors,
        *,
        t_alpha: float,
        dt_seconds: float,
        active_mask: torch.Tensor,
        engine: GPUEngine,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, ConvectionState]:
        if not bool(torch.any(active_mask)):
            return particles, state
        device, dtype = engine.device, engine.dtype

        # 1-3) Parcel lift, trigger and closure in every column. The `convection.*`
        # sub-phases are charged to the run's phase timers when those are on.
        with phase("convection.columns"):
            cols = self.column_state(met_window, t_alpha=t_alpha, device=device, dtype=dtype)
            fire_idx = cols.fires.nonzero().squeeze(1)
            n_fire = int(fire_idx.numel())
        if n_fire == 0:
            return particles, state

        # 4) The columns the active particles are in. Matrices are built only for
        # columns that fire and hold at least one active particle: a matrix is
        # [Z, Z] (137 x 137 on ERA5 model levels), so building one for every
        # firing column of a summer EUROPE window (~150,000 columns) would take
        # tens of GB for columns no particle uses. Each particle's matrix is the
        # same as when every firing column's was built (the construction is
        # column by column).
        with phase("convection.matrices"):
            n_col = cols.fires.numel()
            # Particle and grid positions in the engine's horizontal frame (task A4;
            # affine in lon and lat, so the nearest column is the same column).
            frame = engine.frame
            particle_col = nearest_column_index(
                particles[:, 0],
                particles[:, 1],
                frame.x_from_lon(np.asarray(met_window.metadata.lon, dtype=np.float64)),
                frame.y_from_lat(np.asarray(met_window.metadata.lat, dtype=np.float64)),
            )
            occupied = torch.zeros(n_col, dtype=torch.bool, device=device)
            occupied[particle_col[active_mask]] = True
            use_idx = (cols.fires & occupied).nonzero().squeeze(1)
            n_use = int(use_idx.numel())

            # 5) Mass-flux matrices of the occupied firing columns (kg/m² per event:
            # the flux times the met interval, which follows the store's cadence).
            if n_use > 0:
                fmass = compute_mass_flux_matrices(
                    cols.pressure_pa[use_idx],
                    cols.i_lcl[use_idx],
                    cols.i_lnb[use_idx],
                    (cols.m_b[use_idx] * cols.interval_s).to(dtype),
                )
                layer_mass = _layer_masses_per_area(cols.pressure_pa[use_idx]).clamp(min=1e-6)
                prob = backward_transition_matrices(fmass, layer_mass)  # [U, Z, Z]
                del fmass, layer_mass
            else:
                # No active particle under convection: nothing moves, and (as when
                # no particle is eligible) no random number is drawn.
                prob = torch.zeros((0, 0, 0), dtype=dtype, device=device)

        with phase("convection.redistribute"):
            # 6) Each particle is moved with the matrix of the column it is in.
            col_slot = torch.full((n_col,), -1, dtype=torch.long, device=device)
            col_slot[use_idx] = torch.arange(n_use, device=device)
            particle_slot = col_slot[particle_col]
            particles_out, n_displaced = self._redistribute_particles(
                particles=particles,
                eligible=active_mask & (particle_slot >= 0),
                particle_col=particle_col,
                particle_slot=particle_slot,
                height_col=cols.height_m,
                prob=prob,
                generator=generator,
                device=device,
                dtype=dtype,
            )

        with phase("convection.log"):
            if LOGGER.isEnabledFor(logging.INFO):
                fired_cape = cols.cape[fire_idx]
                LOGGER.info(
                    "Emanuel convection at %s: fires in %d / %d columns "
                    "(CAPE median %.0f, max %.0f J/kg; M_b max %.3f kg/m²/s; interval %.0f s); "
                    "%d hold active particles. Displaced %d / %d active particles.",
                    met_window.metadata.time_start,
                    n_fire,
                    n_col,
                    float(fired_cape.median()),
                    float(fired_cape.max()),
                    float(cols.m_b.max()),
                    cols.interval_s,
                    n_use,
                    n_displaced,
                    int(active_mask.sum().item()),
                )
        return particles_out, state

    # ----- Particle redistribution -----

    @staticmethod
    def _redistribute_particles(
        *,
        particles: torch.Tensor,
        eligible: torch.Tensor,
        particle_col: torch.Tensor,
        particle_slot: torch.Tensor,
        height_col: torch.Tensor,
        prob: torch.Tensor,
        generator: torch.Generator | None,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, int]:
        """Redistribute eligible particles, each with its own column's transition matrix.

        Follows FLEXPART ``calcmatrix`` / ``redist`` (backward, ``ldirect=-1``):
        ``prob[s, i, j]`` is the probability that a particle in layer ``i`` of
        firing column ``s`` moves to layer ``j`` (`backward_transition_matrices`).
        A particle's host layer is its column's level nearest its height; it
        draws a destination from that row by cumulative sum and a uniform number.
        Movers are placed at a uniform random height within the destination layer;
        particles that draw their own layer keep their position. Only altitude
        changes; lon/lat/weight are untouched.

        ``eligible`` marks active particles in a firing column; ``particle_slot``
        maps each particle to its column's index in ``prob`` (-1: no matrix);
        ``height_col`` is [C, Z], ascending.
        """

        n = particles.shape[0]
        n_lev = height_col.shape[1]
        out = particles.clone()

        sel = eligible.nonzero().squeeze(1)
        if int(sel.numel()) == 0:
            return out, 0

        # Host level: nearest level of the particle's own column.
        heights = height_col[particle_col[sel]]  # [S, Z]
        z_particles = out[sel, 2].contiguous()
        idx_upper = (
            torch.searchsorted(heights, z_particles.unsqueeze(1))
            .squeeze(1)
            .clamp(min=1, max=n_lev - 1)
        )
        idx_lower = idx_upper - 1
        z_lo = heights.gather(1, idx_lower.unsqueeze(1)).squeeze(1)
        z_hi = heights.gather(1, idx_upper.unsqueeze(1)).squeeze(1)
        host = torch.where((z_particles - z_lo) < (z_hi - z_particles), idx_lower, idx_upper)

        # Destination distribution (includes the stay-diagonal).
        probs = prob[particle_slot[sel], host]  # [S, Z]
        probs = probs / probs.sum(dim=1, keepdim=True).clamp(min=1e-12)
        cum = probs.cumsum(dim=1)

        # One uniform per particle (all N, so the random stream does not depend
        # on which columns fire).
        u = (
            torch.rand(n, generator=generator, device=device, dtype=dtype)
            if generator is not None
            else torch.rand(n, device=device, dtype=dtype)
        )
        dest = torch.searchsorted(cum, u[sel].unsqueeze(1)).squeeze(1).clamp(max=n_lev - 1)

        moved = dest != host
        n_moved = int(moved.sum().item())
        if n_moved > 0:
            z_lo_d = heights.gather(1, (dest - 1).clamp(min=0).unsqueeze(1)).squeeze(1)
            z_hi_d = heights.gather(1, dest.unsqueeze(1)).squeeze(1)
            u_within = (
                torch.rand(n, generator=generator, device=device, dtype=dtype)
                if generator is not None
                else torch.rand(n, device=device, dtype=dtype)
            )
            z_new = z_lo_d + u_within[sel] * (z_hi_d - z_lo_d)
            out[sel, 2] = torch.where(moved, z_new, z_particles)

        return out, n_moved
