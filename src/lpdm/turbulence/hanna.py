"""Hanna 1982 / FLEXPART turbulence scheme.

Implementation per `docs/turbulence.md` §3.2. Three stability regimes (stable,
neutral, unstable) within the boundary layer with FLEXPART piecewise-homogeneous
treatment (no explicit Thomson 1987 drift). Surface-layer override for `z < 0.1 h`.
Constant-diffusivity placeholder above the BL.

Constants below come from Hanna 1982 / FLEXPART manual sections 4.3.x. They MUST
be cross-checked against the FLEXPART source tree before any external comparison
run; secondary references occasionally diverge by ~10% on minor coefficients.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np
import torch

from lpdm.gpu_engine import GPUEngine
from lpdm.met_reader import HourlyMetTensors
from lpdm.turbulence.base import TurbulenceScheme, TurbulenceState, register_scheme


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

# Numerical floors to avoid degenerate values at z=0 / very low ustar / etc.
SIGMA_MIN_M_S = 1e-3
T_L_MIN_S = 1.0
USTAR_MIN_M_S = 1e-3
BLH_MIN_M = 50.0

# Free-troposphere (above-BL) gradient-Richardson closure constants (docs/turbulence.md §3.2.3).
P0_PA = 100000.0                  # reference pressure for potential temperature
KAPPA_POISSON = R_DRY_AIR_J_KG_K / C_P_DRY_AIR_J_KG_K  # ~0.2854
FT_MIXING_LENGTH_ASYMPTOTE_M = 100.0  # Blackadar asymptotic mixing length (free troposphere)
FT_RICHARDSON_CRIT = 0.25         # critical gradient Richardson number
FT_KZ_FLOOR_M2_S = 0.1            # background diffusivity so FT particles are never fully frozen
FT_KZ_CEIL_M2_S = 50.0            # cap on shear-driven FT diffusivity
FT_SHEAR_SQ_FLOOR_S2 = 1e-8       # floor on |dU/dz|^2 to avoid div-by-zero in Ri
FT_N2_FLOOR_S2 = 1e-6             # floor on N^2 for the buoyancy timescale
FT_T_L_DEFAULT_S = 100.0          # fallback Lagrangian timescale where N^2 <= 0
FT_T_L_MAX_S = 1000.0             # cap on the buoyancy-derived FT timescale


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
) -> torch.Tensor:
	"""Convective velocity scale `w* = ((g h H) / (T ρ c_p))^(1/3)`.

	Returns 0 where `H ≤ 0` (stable / neutral), since `w*` is undefined there.
	"""

	rho = air_density(sp, t_surface)
	cube = (GRAVITY_M_S2 * blh.clamp(min=BLH_MIN_M) * shf) / (t_surface * rho * C_P_DRY_AIR_J_KG_K)
	return torch.where(cube > 0, cube.clamp(min=0.0).pow(1.0 / 3.0), torch.zeros_like(cube))


def _in_bl_stable(
	z: torch.Tensor,
	blh: torch.Tensor,
	ustar: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
	"""(σ_uv, σ_w, T_Luv, T_Lw) under stable BL (z ≤ h)."""

	z_over_h = (z / blh).clamp(min=0.0, max=1.0)
	one_minus = (1.0 - z_over_h).clamp(min=0.0)

	sigma_w = 1.3 * ustar * one_minus.pow(0.75)
	sigma_uv = 2.0 * ustar * one_minus.pow(0.75)
	# Use small-z floor to avoid (z/h)^0.8 → 0 producing T_L = 0 at the surface.
	z_over_h_floor = z_over_h.clamp(min=1e-6)
	T_Lw = 0.10 * blh / sigma_w.clamp(min=SIGMA_MIN_M_S) * z_over_h_floor.pow(0.8)
	T_Luv = 0.15 * blh / sigma_uv.clamp(min=SIGMA_MIN_M_S) * z_over_h_floor.pow(0.5)
	return sigma_uv, sigma_w, T_Luv, T_Lw


def _in_bl_neutral(
	z: torch.Tensor,
	ustar: torch.Tensor,
	f_cor: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
	"""(σ_uv, σ_w, T_Luv, T_Lw) under neutral BL."""

	ustar_safe = ustar.clamp(min=USTAR_MIN_M_S)
	expfact = torch.exp(-2.0 * f_cor * z / ustar_safe)
	sigma_w = 1.3 * ustar * expfact
	sigma_uv = 2.0 * ustar * expfact
	denom = 1.0 + 15.0 * f_cor * z / ustar_safe
	T_Lw = 0.5 * z / sigma_w.clamp(min=SIGMA_MIN_M_S) / denom.clamp(min=1.0)
	T_Luv = T_Lw / 1.5
	return sigma_uv, sigma_w, T_Luv, T_Lw


def _in_bl_unstable(
	z: torch.Tensor,
	blh: torch.Tensor,
	ustar: torch.Tensor,
	w_star: torch.Tensor,
	h_over_L: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
	"""(σ_uv, σ_w, T_Luv, T_Lw) under unstable / convective BL."""

	z_over_h = (z / blh).clamp(min=0.0, max=1.0)
	one_minus_095 = (1.0 - 0.95 * z_over_h).clamp(min=0.0)

	sigma_w_sq = (
		1.5 * ustar.pow(2) * one_minus_095.pow(2.0 / 3.0)
		+ 1.6 * w_star.pow(2) * z_over_h.pow(2.0 / 3.0) * (1.0 - z_over_h).pow(2)
	)
	sigma_w = sigma_w_sq.clamp(min=SIGMA_MIN_M_S ** 2).sqrt()

	# Cap h/L within a sensible range to avoid pathological values blowing up the bracket.
	h_over_L_capped = h_over_L.clamp(min=-1000.0, max=0.0)
	sigma_uv_sq = (12.0 - 0.5 * h_over_L_capped).clamp(min=1e-3).pow(2.0 / 3.0) * ustar.pow(2)
	sigma_uv = sigma_uv_sq.clamp(min=SIGMA_MIN_M_S ** 2).sqrt()

	T_Lw = 0.15 * blh / sigma_w.clamp(min=SIGMA_MIN_M_S)
	T_Luv = 0.15 * blh / sigma_uv.clamp(min=SIGMA_MIN_M_S)
	return sigma_uv, sigma_w, T_Luv, T_Lw


def in_bl_sigma_TL(
	z: torch.Tensor,
	blh: torch.Tensor,
	ustar: torch.Tensor,
	w_star: torch.Tensor,
	h_over_L: torch.Tensor,
	latitude_deg: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
	"""Per-particle (σ_u, σ_v, σ_w, T_Lu, T_Lv, T_Lw) inside the BL.

	Combines the three stability regimes via `torch.where`. All inputs must be
	per-particle tensors broadcasting to a common shape.
	"""

	f_cor = coriolis_parameter(latitude_deg)

	uv_st, w_st, t_uv_st, t_w_st = _in_bl_stable(z, blh, ustar)
	uv_nu, w_nu, t_uv_nu, t_w_nu = _in_bl_neutral(z, ustar, f_cor)
	uv_un, w_un, t_uv_un, t_w_un = _in_bl_unstable(z, blh, ustar, w_star, h_over_L)

	in_stable = h_over_L > H_OVER_L_STABLE_THRESHOLD
	in_unstable = h_over_L < H_OVER_L_UNSTABLE_THRESHOLD

	def _select(stable_v: torch.Tensor, neutral_v: torch.Tensor, unstable_v: torch.Tensor) -> torch.Tensor:
		return torch.where(in_stable, stable_v, torch.where(in_unstable, unstable_v, neutral_v))

	sigma_uv = _select(uv_st, uv_nu, uv_un)
	sigma_w = _select(w_st, w_nu, w_un)
	T_Luv = _select(t_uv_st, t_uv_nu, t_uv_un)
	T_Lw = _select(t_w_st, t_w_nu, t_w_un)
	# Hanna takes σ_u = σ_v (Eq. 11–24); same for T_Lu = T_Lv.
	return sigma_uv, sigma_uv, sigma_w, T_Luv, T_Luv, T_Lw


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

	l = (KARMAN * z) / (1.0 + KARMAN * z / FT_MIXING_LENGTH_ASYMPTOTE_M)

	f_stable = (1.0 - ri / FT_RICHARDSON_CRIT).clamp(min=0.0).pow(2)
	f_unstable = (1.0 - 16.0 * ri).clamp(min=0.0).sqrt()
	f_ri = torch.where(ri < 0.0, f_unstable, f_stable)

	k_z = l.pow(2) * shear_mag * f_ri
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
	sigma_w = (k_z / t_lw).clamp(min=SIGMA_MIN_M_S ** 2).sqrt()
	return sigma_w, t_lw


def surface_layer_sigma_w(z: torch.Tensor, ustar: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
	"""σ_w in the surface layer, regime-dependent (docs/turbulence.md §3.2.4).

	Stable:    `1.3 u* (1 + 5 z/L)`, capped at `1.3 u* · 6` in very-stable.
	Unstable:  `1.3 u* (1 - 2 z/L)^(1/3)`.
	Neutral:   `1.3 u*` (z/L → 0).
	"""

	eps = 1e-6
	finite_L = L.abs() > eps
	L_safe = torch.where(finite_L, L, torch.ones_like(L))
	z_over_L = torch.where(finite_L, (z / L_safe).clamp(-50.0, 50.0), torch.zeros_like(L))

	sigma_unstable = 1.3 * ustar * (1.0 - 2.0 * z_over_L).clamp(min=eps).pow(1.0 / 3.0)
	cap = 1.3 * ustar * 6.0
	sigma_stable = (1.3 * ustar * (1.0 + 5.0 * z_over_L)).clamp(max=cap)
	sigma_neutral = 1.3 * ustar

	return torch.where(
		z_over_L < 0.0,
		sigma_unstable,
		torch.where(z_over_L > 0.0, sigma_stable, sigma_neutral),
	)


# ---------------------------------------------------------------------------
# Scheme class
# ---------------------------------------------------------------------------


@register_scheme
class HannaScheme(TurbulenceScheme):
	"""Hanna 1982 / FLEXPART turbulence scheme. See `docs/turbulence.md` §3.2."""

	name: ClassVar[str] = "hanna_1982"

	def required_met_keys(self) -> tuple[str, ...]:
		# Baseline (u, v, w, blh, sp) is added by the runtime; declare scheme-specific extras.
		return ("t", "ustar", "shf")

	def initialize_state(
		self,
		n_particles: int,
		*,
		device: torch.device,
		dtype: torch.dtype,
	) -> TurbulenceState:
		return {
			"u_prime": torch.zeros(n_particles, device=device, dtype=dtype),
			"v_prime": torch.zeros(n_particles, device=device, dtype=dtype),
			"w_prime": torch.zeros(n_particles, device=device, dtype=dtype),
		}

	# Finite-difference step (m) for the well-mixed drift's ∂σ_w²/∂z.
	DRIFT_FD_DELTA_M: ClassVar[float] = 1.0

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
		if not bool(torch.any(active_mask)):
			return particles, state

		device = engine.device
		dtype = engine.dtype

		active_xyz = particles[active_mask]
		lon_deg = active_xyz[:, 0]
		lat_deg = active_xyz[:, 1]
		z_agl = active_xyz[:, 2]

		# Interpolate scalar surface fields at particle positions.
		blh = self._interp_surface_field(met_window, "blh", t_alpha, lon_deg, lat_deg, device, dtype)
		sp = self._interp_surface_field(met_window, "sp", t_alpha, lon_deg, lat_deg, device, dtype)
		ustar = self._interp_surface_field(met_window, "ustar", t_alpha, lon_deg, lat_deg, device, dtype)
		shf = self._interp_surface_field(met_window, "shf", t_alpha, lon_deg, lat_deg, device, dtype)
		t_surface = self._interp_t_at_lowest_level(met_window, t_alpha, lon_deg, lat_deg, device, dtype)

		blh = blh.clamp(min=BLH_MIN_M)
		ustar = ustar.clamp(min=USTAR_MIN_M_S)

		# Stability + convective velocity per particle.
		L = obukhov_length(ustar, shf, t_surface, sp)
		h_over_L = blh / L
		w_star = convective_velocity(blh, shf, t_surface, sp)

		# Free-troposphere σ/T_L fields on the met grid (gradient-Richardson
		# closure); interpolated per-particle inside _column_turbulence for the
		# above-BL regime. Computed once per step (depends only on the met window).
		ft_fields, grid_bounds = self._free_trop_fields(met_window, t_alpha, device, dtype)

		col_kwargs = dict(
			blh=blh, ustar=ustar, w_star=w_star, h_over_L=h_over_L, L=L,
			lat=lat_deg, lon=lon_deg, ft_fields=ft_fields, grid_bounds=grid_bounds,
			engine=engine,
		)

		# σ/T_L at the particle height (in-BL + surface-layer + free-troposphere).
		sigma_u, sigma_v, sigma_w, T_Lu, T_Lv, T_Lw = self._column_turbulence(z_agl, **col_kwargs)

		# Thomson (1987) well-mixed drift for the vertical component:
		# a = ½(1 + w'²/σ_w²) ∂σ_w²/∂z. ∂σ_w²/∂z is a central finite difference of
		# the *full* column profile (so it spans the in-BL → free-trop transition),
		# evaluated holding the per-column scalars (blh, u*, …) fixed.
		w_prev = state["w_prime"][active_mask]
		z_hi = z_agl + self.DRIFT_FD_DELTA_M
		z_lo = (z_agl - self.DRIFT_FD_DELTA_M).clamp(min=0.0)
		sw_hi = self._column_turbulence(z_hi, **col_kwargs)[2]
		sw_lo = self._column_turbulence(z_lo, **col_kwargs)[2]
		dsig2_dz = (sw_hi.pow(2) - sw_lo.pow(2)) / (z_hi - z_lo).clamp(min=self.DRIFT_FD_DELTA_M)
		sigma_w_sq = sigma_w.pow(2).clamp(min=SIGMA_MIN_M_S ** 2)
		drift_w = 0.5 * (1.0 + w_prev.pow(2) / sigma_w_sq) * dsig2_dz
		# Backward-Langevin sign. The vertical displacement is applied
		# backward-in-time (apply_vertical_turbulence below uses backward=True,
		# which negates w'·dt). The random forcing is symmetric so that negation
		# is harmless, but the well-mixed drift is a *deterministic* correction:
		# for the adjoint/backward Langevin it must enter with reversed sign
		# (Thomson 1987 §5 reciprocity; Flesch, Wilson & Yee 1995) so that the
		# *physical* position drift still points down the σ_w² gradient (toward
		# the BL) rather than lofting particles out of it. Pre-negate here so the
		# subsequent backward displacement restores the correct physical direction.
		drift_w = -drift_w
		# Safeguard: cap the drift increment at one σ_w per step so sharp σ_w kinks
		# (surface-layer / BL top) can't blow the velocity up between substeps.
		max_accel = sigma_w / float(dt_seconds)
		drift_w = torch.maximum(torch.minimum(drift_w, max_accel), -max_accel)

		# OU velocity update (per-particle T_L and σ²). Drift on the vertical only.
		u_prime_active = engine.update_langevin_velocity(
			state["u_prime"][active_mask],
			t_lagrangian=T_Lu, sigma_w2=sigma_u.pow(2), dt_seconds=dt_seconds,
		)
		v_prime_active = engine.update_langevin_velocity(
			state["v_prime"][active_mask],
			t_lagrangian=T_Lv, sigma_w2=sigma_v.pow(2), dt_seconds=dt_seconds,
		)
		w_prime_active = engine.update_langevin_velocity(
			state["w_prime"][active_mask],
			t_lagrangian=T_Lw, sigma_w2=sigma_w.pow(2), dt_seconds=dt_seconds, drift=drift_w,
		)

		moved = engine.apply_vertical_turbulence(
			active_xyz, w_prime_active, dt_seconds=dt_seconds, backward=True,
		)
		moved = engine.apply_horizontal_turbulence(
			moved, u_prime_active, v_prime_active, dt_seconds=dt_seconds, backward=True,
		)
		moved = engine.reflect_surface(moved, z_surface=0.0)

		particles[active_mask] = moved
		state["u_prime"][active_mask] = u_prime_active
		state["v_prime"][active_mask] = v_prime_active
		state["w_prime"][active_mask] = w_prime_active
		return particles, state

	# ---- Turbulence profile assembly ----

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
		lon: torch.Tensor,
		ft_fields: torch.Tensor,
		grid_bounds: "GridInterpolationBounds",
		engine: GPUEngine,
	) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
		"""Per-particle (σ_u, σ_v, σ_w, T_Lu, T_Lv, T_Lw) at height ``z_query``.

		Assembles the three regimes — in-BL (Hanna), surface-layer override
		(z < 0.1 h), and free-troposphere override (z > h, Richardson closure
		interpolated from ``ft_fields``) — selected per particle by where
		``z_query`` falls. Floors applied. Callable at arbitrary z so the
		well-mixed drift can finite-difference σ_w through the regime transitions.
		"""

		sigma_u, sigma_v, sigma_w, T_Lu, T_Lv, T_Lw = in_bl_sigma_TL(
			z_query, blh, ustar, w_star, h_over_L, lat,
		)

		# Surface-layer override for z < 0.1 h.
		z_sl = SURFACE_LAYER_FRACTION * blh
		in_sl = z_query < z_sl
		sigma_w_sl = surface_layer_sigma_w(z_query, ustar, L)
		z_clamped = torch.maximum(z_query, z_sl)
		su_cl, sv_cl, _, _, _, _ = in_bl_sigma_TL(z_clamped, blh, ustar, w_star, h_over_L, lat)
		sigma_u = torch.where(in_sl, su_cl, sigma_u)
		sigma_v = torch.where(in_sl, sv_cl, sigma_v)
		sigma_w = torch.where(in_sl, sigma_w_sl, sigma_w)
		z_for_TL = z_query.clamp(min=T_L_MIN_S)
		T_Lu = torch.where(in_sl, KARMAN * z_for_TL / sigma_u.clamp(min=SIGMA_MIN_M_S), T_Lu)
		T_Lv = torch.where(in_sl, KARMAN * z_for_TL / sigma_v.clamp(min=SIGMA_MIN_M_S), T_Lv)
		T_Lw = torch.where(in_sl, KARMAN * z_for_TL / sigma_w.clamp(min=SIGMA_MIN_M_S), T_Lw)

		# Free-troposphere override for z > h.
		above_bl = z_query > blh
		if bool(torch.any(above_bl)):
			ft = self._interp_3d_field(ft_fields, grid_bounds, lon, lat, z_query, engine)
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

	def _free_trop_fields(
		self,
		met_window: HourlyMetTensors,
		t_alpha: float,
		device: torch.device,
		dtype: torch.dtype,
	) -> tuple[torch.Tensor, "GridInterpolationBounds"]:
		"""Build free-troposphere (σ_w, σ_uv, T_Lw, T_Luv) on the met grid [4, Z, Y, X].

		Uses the gradient-Richardson closure on potential-temperature and wind
		shear computed from true per-column geopotential heights (``height_agl_m``).
		Returns the stacked fields plus the interpolation bounds for sampling them
		per-particle.
		"""

		from lpdm.gpu_engine import GridInterpolationBounds

		if met_window.height_agl_m is None:
			raise ValueError(
				"HannaScheme free-troposphere turbulence requires HourlyMetTensors.height_agl_m "
				"(3D geopotential height). The met reader provides it; synthetic readers must too."
			)

		def _time_interp(name: str) -> torch.Tensor:
			start, end = met_window.channel(name)  # [Z, Y, X] each
			return (start.to(device=device, dtype=dtype) * (1.0 - t_alpha)
			        + end.to(device=device, dtype=dtype) * t_alpha)

		t_field = _time_interp("t")
		u_field = _time_interp("u")
		v_field = _time_interp("v")
		height = met_window.height_agl_m.to(device=device, dtype=dtype)  # [Z, Y, X]

		pressure_pa = torch.as_tensor(
			np.array(met_window.metadata.pressure_level_hpa), device=device, dtype=dtype
		).view(-1, 1, 1) * 100.0

		theta = potential_temperature(t_field, pressure_pa)
		dtheta_dz = self._vertical_gradient(theta, height)
		n2 = brunt_vaisala_squared(theta, dtheta_dz)

		du_dz = self._vertical_gradient(u_field, height)
		dv_dz = self._vertical_gradient(v_field, height)
		shear_sq = du_dz.pow(2) + dv_dz.pow(2)
		shear_mag = shear_sq.clamp(min=FT_SHEAR_SQ_FLOOR_S2).sqrt()

		ri = gradient_richardson(n2, shear_sq)
		k_z = free_trop_diffusivity(height.clamp(min=T_L_MIN_S), shear_mag, ri)
		sigma_w_ft, t_lw_ft = free_trop_sigma_TL(k_z, n2)
		# FT shear turbulence treated as isotropic (documented simplification);
		# the unresolved-mesoscale "meander" horizontal term is a separate item.
		sigma_uv_ft = sigma_w_ft
		t_luv_ft = t_lw_ft

		fields = torch.stack([sigma_w_ft, sigma_uv_ft, t_lw_ft, t_luv_ft], dim=0)

		lon_arr = met_window.metadata.lon
		lat_arr = met_window.metadata.lat
		level_arr = met_window.metadata.level
		grid_bounds = GridInterpolationBounds(
			lon_first=float(lon_arr[0]),
			lon_last=float(lon_arr[-1]),
			lat_first=float(lat_arr[0]),
			lat_last=float(lat_arr[-1]),
			alt_first=float(level_arr[0]),
			alt_last=float(level_arr[-1]),
		)
		return fields, grid_bounds

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
		grid_bounds: "GridInterpolationBounds",
		lon: torch.Tensor,
		lat: torch.Tensor,
		z: torch.Tensor,
		engine: GPUEngine,
	) -> torch.Tensor:
		"""Trilinear interp of a [C, Z, Y, X] field stack at particle (lon, lat, z).
		Returns [C, N]. Reuses the engine's coordinate normalisation so the vertical
		mapping matches the wind advection."""

		xyz = torch.stack([lon, lat, z], dim=1)
		xyz_norm = engine.normalize_particle_coordinates(xyz, grid_bounds)
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
		grid = torch.stack([lon_norm, lat_norm], dim=-1).view(1, 1, -1, 2).to(device=device, dtype=dtype)

		field = field_2d.unsqueeze(0).unsqueeze(0)
		sampled = torch.nn.functional.grid_sample(field, grid, align_corners=True, mode="bilinear")
		return sampled.view(-1)

	@classmethod
	def _interp_surface_field(
		cls,
		met_window: HourlyMetTensors,
		channel_name: str,
		t_alpha: float,
		lon: torch.Tensor,
		lat: torch.Tensor,
		device: torch.device,
		dtype: torch.dtype,
	) -> torch.Tensor:
		"""Interpolate a surface (z-broadcast) channel at particle (lon, lat)."""

		chan_start, chan_end = met_window.channel(channel_name)  # [Z, Y, X] each
		field_start = chan_start[0].to(device=device, dtype=dtype)
		field_end = chan_end[0].to(device=device, dtype=dtype)
		field_2d = field_start * (1.0 - t_alpha) + field_end * t_alpha
		return cls._interp_2d_bilinear(
			field_2d, met_window.metadata.lon, met_window.metadata.lat, lon, lat,
		)

	@classmethod
	def _interp_t_at_lowest_level(
		cls,
		met_window: HourlyMetTensors,
		t_alpha: float,
		lon: torch.Tensor,
		lat: torch.Tensor,
		device: torch.device,
		dtype: torch.dtype,
	) -> torch.Tensor:
		"""Interpolate temperature at the lowest model level (surface proxy)."""

		t_start, t_end = met_window.channel("t")
		level_agl = met_window.metadata.level
		lowest_idx = int(np.argmin(level_agl))
		field_start = t_start[lowest_idx].to(device=device, dtype=dtype)
		field_end = t_end[lowest_idx].to(device=device, dtype=dtype)
		field_2d = field_start * (1.0 - t_alpha) + field_end * t_alpha
		return cls._interp_2d_bilinear(
			field_2d, met_window.metadata.lon, met_window.metadata.lat, lon, lat,
		)
