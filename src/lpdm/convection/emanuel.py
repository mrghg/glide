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
- We use the bbox-mean met column (`metadata.level`) for the parcel lift,
  rather than per-(lon,lat) columns. Convection then triggers/doesn't fire
  uniformly across the bbox — a strong approximation for large domains. See
  the F9 follow-up in dev/CHECKPOINT.md for the same approximation in advection.
"""

from __future__ import annotations

import logging
import math
from typing import ClassVar

import numpy as np
import torch

from lpdm.convection.base import (
	ConvectionScheme,
	ConvectionState,
	register_scheme,
)
from lpdm.gpu_engine import GPUEngine
from lpdm.met_reader import HourlyMetTensors


LOGGER = logging.getLogger(__name__)


# Physical constants (mirroring lpdm.turbulence.hanna to avoid drift).
GRAVITY_M_S2 = 9.80665
R_DRY_AIR_J_KG_K = 287.05
R_VAPOR_J_KG_K = 461.5
EPSILON_RD_RV = R_DRY_AIR_J_KG_K / R_VAPOR_J_KG_K  # ≈ 0.622
C_P_DRY_AIR_J_KG_K = 1005.0
L_V_J_KG = 2.5e6           # latent heat of vapourisation at 0 °C
P0_PA = 100000.0
KAPPA_POISSON = R_DRY_AIR_J_KG_K / C_P_DRY_AIR_J_KG_K  # ≈ 0.2854

# Scheme thresholds (Forster 2007 / typical Emanuel defaults).
TRIGGER_DELTA_TV_K = 0.9   # T_threshold in Forster 2007 Eq 34
MIN_CAPE_J_KG = 50.0       # floor on CAPE before we even consider triggering
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

	``pressure_pa`` is the per-level pressure [Z]. Layer i lies between levels i
	and i+1 (or i-1 and i for the descending-pressure convention). We use a
	half-bracket: layer mass at level i is `|p_{i+1} - p_{i-1}|·0.5 / g` for
	interior, and the half-layer for the top/bottom. Returns shape [Z].
	"""

	p = pressure_pa
	dp = torch.zeros_like(p)
	dp[1:-1] = 0.5 * (p[:-2] - p[2:]).abs()
	dp[0] = (p[0] - p[1]).abs() * 0.5
	dp[-1] = (p[-2] - p[-1]).abs() * 0.5
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

	t_lcl = lcl_temperature_bolton(t_sfc.unsqueeze(0), q_sfc.unsqueeze(0), p_sfc.unsqueeze(0)).squeeze(0)
	p_lcl = lcl_pressure_poisson(t_sfc.unsqueeze(0), t_lcl.unsqueeze(0), p_sfc.unsqueeze(0)).squeeze(0)

	# θ_e of the parcel (conserved along the moist adiabat).
	theta_e_parcel = equivalent_potential_temperature(
		t_sfc.unsqueeze(0), q_sfc.unsqueeze(0), p_sfc.unsqueeze(0),
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
		dz[1:] = R_DRY_AIR_J_KG_K * 0.5 * (t_env[:-1] + t_env[1:]) * (
			torch.log(pressure_pa[:-1].clamp(min=P_MIN_PA))
			- torch.log(pressure_pa[1:].clamp(min=P_MIN_PA))
		) / GRAVITY_M_S2
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
		if met_window.height_agl_m is None:
			raise ValueError(
				"EmanuelReducedConvection requires HourlyMetTensors.height_agl_m "
				"(3D geopotential height). The met reader provides it; synthetic "
				"readers must too."
			)

		device, dtype = engine.device, engine.dtype

		# Time-interpolated bbox-mean column profiles (Z,). The full Emanuel
		# scheme would compute per-column profiles; we use the bbox-mean and
		# document it as a simplification (same approximation used by F9 in
		# the audit-followup PR).
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
		t_col = t_field.mean(dim=(1, 2))  # bbox-mean → [Z]
		q_col = q_field.mean(dim=(1, 2)).clamp(min=Q_MIN_KG_KG)
		height_col = met_window.height_agl_m.to(device=device, dtype=dtype).mean(dim=(1, 2))

		# Pressure profile from metadata.pressure_level_hpa.
		pressure_pa = torch.as_tensor(
			np.array(met_window.metadata.pressure_level_hpa), device=device, dtype=dtype,
		) * 100.0
		# Reorder all column quantities to ascending z (Emanuel expects surface
		# at index 0). ARCO ERA5 has level coord descending altitude.
		ascending = bool(height_col[-1] > height_col[0])
		if not ascending:
			t_col = t_col.flip(0)
			q_col = q_col.flip(0)
			pressure_pa = pressure_pa.flip(0)
			height_col = height_col.flip(0)

		# 1) Identify LCL / LNB / CAPE for the bbox-mean column.
		i_lcl_t, i_lnb_t, cape, dtv_lcl_plus_1 = compute_lcl_lnb_cape(
			t_col, q_col, pressure_pa,
		)
		i_lcl, i_lnb = int(i_lcl_t), int(i_lnb_t)

		# 2) Trigger checks (Forster 2007 Eq 34 + a CAPE/depth floor).
		fires = (
			(i_lcl >= 0)
			and (i_lnb > i_lcl)
			and (float(cape) >= self.min_cape_j_kg)
			and (float(dtv_lcl_plus_1) >= self.trigger_dtv_k)
		)
		if fires:
			cloud_depth_m = float(height_col[i_lnb] - height_col[i_lcl])
			if cloud_depth_m < self.min_cloud_depth_m:
				fires = False

		if not fires:
			return particles, state

		# 3) Cloud-base mass flux (closure). M_b ∝ ρ_LCL · w_buoy, with
		# w_buoy = min(√(2·CAPE), UPDRAFT_W_MAX_M_S). The cap keeps M_b in the
		# realistic FLEXPART range (0.05–0.5 kg/m²/s); without it, large CAPE
		# (>500 J/kg) gives unphysically large M_b.
		t_lcl = t_col[i_lcl]
		p_lcl = pressure_pa[i_lcl]
		rho_lcl = p_lcl / (R_DRY_AIR_J_KG_K * t_lcl.clamp(min=T_MIN_K))
		w_buoy = min(math.sqrt(2.0 * max(float(cape), 0.0)), UPDRAFT_W_MAX_M_S)
		m_b = self.closure_c * float(rho_lcl) * w_buoy

		# 4) Mass-flux matrix (kg/m² / met-update). Multiply by dt_seconds to
		# get the per-call redistribution mass — but our scheme is called only
		# at met-update boundaries (~hourly), so use the met interval. We
		# approximate the met interval as 3600 s (hourly ERA5); the runtime
		# could pass it explicitly in a future refinement.
		convection_call_interval_s = 3600.0
		fmass = compute_mass_flux_matrix(
			t_col, q_col, pressure_pa,
			i_lcl=i_lcl, i_lnb=i_lnb,
			m_b=m_b * convection_call_interval_s,
		)

		# Per-level mass per area: m_i = (Δp / g) — used for the diagonal closure
		# and to convert the flux matrix into redistribution probabilities.
		layer_mass = _layer_masses_per_area(pressure_pa).clamp(min=1e-6)

		# 5) Redistribute particles through the (non-divergent) transition matrix.
		# GLIDE runs backward, so the destination is sampled from the matrix COLUMN
		# (FLEXPART redist ldirect=-1); the well-mixed criterion holds either way.
		particles_out, n_displaced = self._redistribute_particles(
			particles=particles,
			active_mask=active_mask,
			height_col=height_col,
			fmass=fmass,
			layer_mass=layer_mass,
			i_lcl=i_lcl, i_lnb=i_lnb,
			generator=generator,
			device=device, dtype=dtype,
			ascending=ascending,
			backward=True,
		)

		# Log once per met update (the t_alpha changes when met advances).
		if self._last_log_t_alpha != t_alpha:
			LOGGER.info(
				"Emanuel convection: fires at t_alpha=%.2f. "
				"CAPE=%.0f J/kg, LCL=%.0f m, LNB=%.0f m, depth=%.0f m, M_b=%.3f kg/m²/s. "
				"Displaced %d / %d active particles.",
				t_alpha, float(cape),
				float(height_col[i_lcl]), float(height_col[i_lnb]),
				float(height_col[i_lnb] - height_col[i_lcl]),
				m_b, n_displaced, int(active_mask.sum().item()),
			)
			self._last_log_t_alpha = t_alpha

		return particles_out, state

	# ----- Particle redistribution -----

	@staticmethod
	def _redistribute_particles(
		*,
		particles: torch.Tensor,
		active_mask: torch.Tensor,
		height_col: torch.Tensor,
		fmass: torch.Tensor,
		layer_mass: torch.Tensor,
		i_lcl: int,
		i_lnb: int,
		generator: torch.Generator | None,
		device: torch.device,
		dtype: torch.dtype,
		ascending: bool,
		backward: bool = True,
	) -> tuple[torch.Tensor, int]:
		"""Redistribute active particles through the convective transition matrix.

		Follows FLEXPART ``calcmatrix`` / ``redist``: apply the diagonal closure so
		each row of ``fmassfrac`` sums to the layer air mass (the diagonal is the
		"stay" mass), then sample a destination level per particle from the
		transition matrix — the matrix ROW for forward runs, the matrix COLUMN for
		backward runs (``ldirect=-1``). GLIDE runs backward, so ``backward=True``
		uses the column (transpose). Because ``fmass`` is non-divergent, both the
		row- and column-normalised transitions are valid probability distributions
		and leave a mass-proportional ensemble invariant (the well-mixed criterion).

		Sampled movers are placed at a uniform random height within the destination
		layer; particles that sample their own layer (the stay-diagonal) keep their
		position. Only altitude changes; lon/lat/weight are untouched.
		"""

		n = particles.shape[0]
		n_lev = height_col.numel()
		out = particles.clone()

		# Diagonal closure: fmassfrac[i,i] = m_i - Σ_j fmass[i,j]; rows sum to m_i.
		# (CFL cap in compute_mass_flux_matrix keeps the diagonal non-negative.)
		fmassfrac = fmass + torch.diag((layer_mass - fmass.sum(dim=1)).clamp(min=0.0))
		# Transition matrix: destination distribution for a particle in layer i is
		# row i (forward) or column i (backward) divided by m_i. Non-divergence ⇒
		# both are normalised to 1.
		trans = fmassfrac.t() if backward else fmassfrac
		prob = (trans / layer_mass.clamp(min=1e-12).unsqueeze(1)).clamp(min=0.0)  # [Z, Z]

		# Per-particle host level (nearest level by height); height_col ascending.
		z_particles = out[:, 2].contiguous()
		idx_upper = torch.searchsorted(height_col, z_particles).clamp(min=1, max=n_lev - 1)
		idx_lower = idx_upper - 1
		z_lo = height_col[idx_lower]
		z_hi = height_col[idx_upper]
		host = torch.where((z_particles - z_lo) < (z_hi - z_particles), idx_lower, idx_upper)

		eligible = active_mask
		if int(eligible.sum().item()) == 0:
			return out, 0

		# Per-particle destination distribution (includes the stay-diagonal).
		probs = prob[host]  # [N, Z]
		probs = probs / probs.sum(dim=1, keepdim=True).clamp(min=1e-12)
		cum = probs.cumsum(dim=1)

		u = (
			torch.rand(n, generator=generator, device=device, dtype=dtype)
			if generator is not None
			else torch.rand(n, device=device, dtype=dtype)
		)
		dest = torch.searchsorted(cum, u.unsqueeze(1)).squeeze(1).clamp(max=n_lev - 1)

		# A particle is "displaced" only if its sampled destination differs from
		# its host level; stayers (dest == host, the diagonal) keep their z.
		moved = eligible & (dest != host)
		n_moved = int(moved.sum().item())

		if n_moved > 0:
			z_lo_d = height_col[(dest - 1).clamp(min=0)]
			z_hi_d = height_col[dest]
			u_within = (
				torch.rand(n, generator=generator, device=device, dtype=dtype)
				if generator is not None
				else torch.rand(n, device=device, dtype=dtype)
			)
			z_new = z_lo_d + u_within * (z_hi_d - z_lo_d)
			out[:, 2] = torch.where(moved, z_new, out[:, 2])

		return out, n_moved
