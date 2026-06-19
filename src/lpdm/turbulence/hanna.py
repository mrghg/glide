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

import logging
import os
from datetime import datetime
from typing import Callable, ClassVar

import numpy as np
import torch

from lpdm.gpu_engine import GPUEngine
from lpdm.met_reader import HourlyMetTensors
from lpdm.turbulence.base import TurbulenceScheme, TurbulenceState, register_scheme


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

# Meander (unresolved-mesoscale) horizontal turbulence — Maryon (1998)
# "meandering" as adopted by FLEXPART (Stohl et al. 2005 §4.5). σ_meander is the
# local grid-wind variability times a coefficient; the timescale is ~half the met
# field interval. See docs/turbulence.md §3.2.8.
MEANDER_COEFFICIENT_DEFAULT = 0.16    # FLEXPART `turbmesoscale`
MEANDER_STENCIL_RADIUS_DEFAULT = 1    # 3x3 neighbourhood
MEANDER_TIMESCALE_S_DEFAULT = 1800.0  # half of the hourly ERA5 field interval

# F4 Tier 2 — per-particle substepping. When the outer dt is large vs T_Lw, the
# discrete OU integration accumulates a Δt/τ bias (Wilson & Flesch 1993 App.,
# Stohl & Thomson 1999). We substep so each particle's effective sub-dt satisfies
# `sub_dt < SUBSTEP_C · T_Lw`. SUBSTEP_C defaults to 0.5 (≈25% bias bound per
# W&F's linear bias formula) rather than S&T's c=0.05 because we cap the substep
# count at MAX_SUBSTEPS_DEFAULT to keep the per-step cost bounded — the cap is
# the binding constraint for the near-surface particles where T_Lw can drop to a
# few seconds.
SUBSTEP_C_DEFAULT = 0.5
MAX_SUBSTEPS_DEFAULT = 50

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

	def __init__(
		self,
		*,
		meander_enabled: bool = False,
		meander_coefficient: float = MEANDER_COEFFICIENT_DEFAULT,
		meander_stencil_radius: int = MEANDER_STENCIL_RADIUS_DEFAULT,
		meander_timescale_seconds: float = MEANDER_TIMESCALE_S_DEFAULT,
		substep_c: float = SUBSTEP_C_DEFAULT,
		max_substeps: int = MAX_SUBSTEPS_DEFAULT,
		z_ubl_m: float = Z_UBL_DEFAULT_M,
		static_substeps: bool | None = None,
	) -> None:
		"""Construct the scheme.

		Meander (unresolved-mesoscale horizontal turbulence) is off by default so
		existing runs stay bit-identical; enable it via the YAML ``turbulence.meander``
		block. See ``docs/turbulence.md`` §3.2.8.

		``substep_c`` and ``max_substeps`` control the F4 Tier 2 adaptive substepping
		(audit 2026-05-30). Each particle's OU + displacement + reflection is
		substepped so its effective sub-dt satisfies ``sub_dt < substep_c · T_Lw``,
		capped at ``max_substeps`` per outer step to bound the cost for very-near-
		surface particles. Meander has τ ≈ 1800 s so always runs at the outer dt.

		``static_substeps`` selects the substep-loop implementation (architecture.md
		§5). ``None`` (default) auto-selects by device: the **static-shape** variant
		on CUDA (every substep processes the full active set with ``sub_dt=0`` no-ops
		for finished particles — constant kernel shapes, the prerequisite for
		``torch.compile`` fusion and CUDA-graph capture on a launch-bound GPU), and
		the **dynamic masked** variant on CPU/MPS (touches only particles still owing
		a substep — cheaper where launches are free). ``True``/``False`` force one
		path (used by the equivalence tests). Env override: ``GLIDE_STATIC_SUBSTEPS``.
		The two paths are physics-equivalent (statistically; bit-identical only when
		every particle needs the same substep count).
		"""

		if meander_coefficient <= 0:
			raise ValueError("meander_coefficient must be > 0")
		if meander_stencil_radius < 1:
			raise ValueError("meander_stencil_radius must be >= 1")
		if meander_timescale_seconds <= 0:
			raise ValueError("meander_timescale_seconds must be > 0")
		if substep_c <= 0:
			raise ValueError("substep_c must be > 0")
		if max_substeps < 1:
			raise ValueError("max_substeps must be >= 1")
		if z_ubl_m < 0:
			raise ValueError("z_ubl_m must be >= 0")
		self.z_ubl_m = float(z_ubl_m)
		self.meander_enabled = bool(meander_enabled)
		self.meander_coefficient = float(meander_coefficient)
		self.meander_stencil_radius = int(meander_stencil_radius)
		self.meander_timescale_seconds = float(meander_timescale_seconds)
		self.substep_c = float(substep_c)
		self.max_substeps = int(max_substeps)
		self._static_substeps = static_substeps
		# F4 Tier 1 (audit 2026-05-30): once-per-instance warning when dt is large
		# vs the smallest active T_L. After F4 Tier 2 substepping, this warning
		# now fires when the *effective* sub-dt is still large vs T_L — i.e. when
		# `max_substeps` is the binding constraint (the substep cap saturates).
		self._warned_substep_cap: bool = False

		# Per-met-window cache for the grid-wide derived fields (meander σ, density
		# + ∂ρ/∂z, free-trop σ/T_L). These depend only on the hourly met window,
		# but `step` is called every dt (~60 steps/window), so recomputing the
		# avg_pool2d / vertical-gradient stacks per step was ~18% of runtime
		# (profiled 2026-06-18). We instead build each stack at the window's two
		# time endpoints once per window and time-interpolate per step. See
		# `_cached_window_field`. One entry per field key (the current window).
		self._window_field_cache: dict[str, tuple[datetime, torch.Tensor]] = {}

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
		state = {
			"u_prime": torch.zeros(n_particles, device=device, dtype=dtype),
			"v_prime": torch.zeros(n_particles, device=device, dtype=dtype),
			"w_prime": torch.zeros(n_particles, device=device, dtype=dtype),
		}
		if self.meander_enabled:
			state["u_meander"] = torch.zeros(n_particles, device=device, dtype=dtype)
			state["v_meander"] = torch.zeros(n_particles, device=device, dtype=dtype)
		return state

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
		# above-BL regime. Built once per met window and cached (perf 2026-06-18).
		ft_fields, grid_bounds = self._free_trop_fields(met_window, device, dtype)

		# Density and ∂ρ/∂z on the met grid for the Stohl-Thomson (1999) density
		# correction term in the well-mixed drift (F2, audit 2026-05-30). Built
		# once per met window and cached; sampled per-particle below.
		density_fields = self._density_fields(met_window, device, dtype)

		col_kwargs = dict(
			blh=blh, ustar=ustar, w_star=w_star, h_over_L=h_over_L, L=L,
			lat=lat_deg, lon=lon_deg, ft_fields=ft_fields, grid_bounds=grid_bounds,
			engine=engine,
		)

		# F5/F15 (audit 2026-05-30): "unresolved basal layer" per W&F §7b. For
		# the σ/T_L/drift/density sampling we clamp z at z_ubl_m (default 2 m),
		# so anything below the UBL gets the same turbulence parameters as the
		# UBL top. This makes smooth-wall reflection WMC-exact in the basal
		# layer (W&F prove perfect reflection requires constant σ over the
		# largest distance a particle can traverse in one step). Particle
		# *position* is NOT clamped — only the SAMPLING height for σ/T_L.
		z_eval = z_agl.clamp(min=self.z_ubl_m)

		# σ/T_L at the particle height (in-BL + surface-layer + free-troposphere).
		sigma_u, sigma_v, sigma_w, T_Lu, T_Lv, T_Lw = self._column_turbulence(z_eval, **col_kwargs)

		# Thomson (1987) well-mixed drift for the vertical component:
		# a = ½(1 + w'²/σ_w²) ∂σ_w²/∂z. ∂σ_w²/∂z is a central finite difference of
		# the *full* column profile (so it spans the in-BL → free-trop transition),
		# evaluated holding the per-column scalars (blh, u*, …) fixed. The
		# finite-difference probes also use z_eval so the UBL clamping gives
		# ∂σ²/∂z = 0 inside the basal layer (consistent with constant σ).
		w_prev = state["w_prime"][active_mask]
		z_hi = z_eval + self.DRIFT_FD_DELTA_M
		z_lo = (z_eval - self.DRIFT_FD_DELTA_M).clamp(min=self.z_ubl_m)
		sw_hi = self._column_turbulence(z_hi, **col_kwargs)[2]
		sw_lo = self._column_turbulence(z_lo, **col_kwargs)[2]
		dsig2_dz = (sw_hi.pow(2) - sw_lo.pow(2)) / (z_hi - z_lo).clamp(min=self.DRIFT_FD_DELTA_M)
		sigma_w_sq = sigma_w.pow(2).clamp(min=SIGMA_MIN_M_S ** 2)

		# Density-correction term (Stohl & Thomson 1999 Eq 3, raw-w form):
		# `+ (σ_w²/ρ)·∂ρ/∂z`. Required for WMC in ρ-weighted coordinates — air
		# density drops 20%+ across a deep BL and the term suppresses spurious
		# accumulation of particles aloft (S&T's CAPTEX runs: +5.5% mean on
		# surface concentrations, +1–15% range). Sampled per-particle from the
		# stack built in `_density_fields` above.
		density_at_p = self._interp_3d_field(
			density_fields, grid_bounds, lon_deg, lat_deg, z_eval, engine,
		)
		rho_at_p = density_at_p[0].clamp(min=1e-3)
		drho_dz_at_p = density_at_p[1]
		density_drift_forward = sigma_w_sq * drho_dz_at_p / rho_at_p

		# Backward-Langevin sign pieces (Flesch et al. 1995: a_b = -a_f - 2w/τ
		# flips both inhomogeneity terms in lockstep for symmetric Gaussian g_a;
		# the homogeneous -w/τ is handled inside the OU exp(-dt/τ) factor).
		# Pre-negate here so the subsequent backward displacement gives the
		# correct physical direction. The (1 + w'²/σ²) factor is RECOMPUTED
		# inside _integrate_vertical_substeps per substep using the current w',
		# so we pass the w-independent pieces (`half_dsig2_dz_backward` and
		# `density_drift_backward`) and the σ² needed to form the factor.
		half_dsig2_dz_backward = -0.5 * dsig2_dz
		density_drift_backward = -density_drift_forward

		# F3 (audit 2026-05-30): drift cap removed. The cap (`±σ_w/Δt`) was an
		# ad-hoc safeguard for sharp σ_w kinks under too-large Δt; F4 Tier 2
		# substepping below addresses the root cause (Δt/τ bias) without
		# silently violating the WMC formula at the BL-top / SL seam.

		# F4 Tier 2 (audit 2026-05-30): per-particle substepping. Each particle's
		# OU + displacement + reflection runs `k_i = ceil(dt / (substep_c · T_Lw_i))`
		# internal steps so its effective sub-dt satisfies the Δt/τ bias bound.
		# Capped at `max_substeps`; warning fires (once per instance) if the cap
		# saturates. Meander uses τ ≈ 1800 s and stays at the outer dt.
		integrate = (
			self._integrate_vertical_substeps_static
			if self._use_static_substeps(engine)
			else self._integrate_vertical_substeps
		)
		moved, u_prime_active, v_prime_active, w_prime_active = integrate(
			active_xyz=active_xyz,
			u_prime_in=state["u_prime"][active_mask],
			v_prime_in=state["v_prime"][active_mask],
			w_prime_in=state["w_prime"][active_mask],
			sigma_u=sigma_u, sigma_v=sigma_v, sigma_w=sigma_w, sigma_w_sq=sigma_w_sq,
			T_Lu=T_Lu, T_Lv=T_Lv, T_Lw=T_Lw,
			half_dsig2_dz_backward=half_dsig2_dz_backward,
			density_drift_backward=density_drift_backward,
			dt_seconds=dt_seconds,
			engine=engine,
		)

		# Meander (unresolved-mesoscale) horizontal turbulence: an independent
		# horizontal OU process whose σ is the local grid-wind variability (Maryon
		# 1998 / FLEXPART §4.5). Applied at all altitudes. No drift (the
		# inhomogeneity captured by the well-mixed term is vertical). Symmetric
		# random forcing, so the backward displacement needs no sign change.
		u_meander_active = v_meander_active = None
		if self.meander_enabled:
			meander_fields, meander_bounds = self._meander_sigma_fields(
				met_window, device, dtype,
			)
			sm = self._interp_3d_field(
				meander_fields, meander_bounds, lon_deg, lat_deg, z_eval, engine,
			)
			sigma_mu, sigma_mv = sm[0], sm[1]
			u_meander_active = engine.update_langevin_velocity(
				state["u_meander"][active_mask],
				t_lagrangian=self.meander_timescale_seconds,
				sigma_w2=sigma_mu.pow(2), dt_seconds=dt_seconds,
			)
			v_meander_active = engine.update_langevin_velocity(
				state["v_meander"][active_mask],
				t_lagrangian=self.meander_timescale_seconds,
				sigma_w2=sigma_mv.pow(2), dt_seconds=dt_seconds,
			)
			moved = engine.apply_horizontal_turbulence(
				moved, u_meander_active, v_meander_active, dt_seconds=dt_seconds, backward=True,
			)

		# Reflection (with w'-flip) is now applied inside the substep loop above;
		# no additional reflection here. Meander has no boundary semantics
		# (horizontal-only displacement) so it doesn't need an extra reflection.

		particles[active_mask] = moved
		state["u_prime"][active_mask] = u_prime_active
		state["v_prime"][active_mask] = v_prime_active
		state["w_prime"][active_mask] = w_prime_active
		if self.meander_enabled:
			state["u_meander"][active_mask] = u_meander_active
			state["v_meander"][active_mask] = v_meander_active
		return particles, state

	def _warn_if_dt_too_large(
		self,
		T_Lu: torch.Tensor,
		T_Lv: torch.Tensor,
		T_Lw: torch.Tensor,
		*,
		dt_seconds: float,
	) -> None:
		"""Reserved for back-compat — no-op now that F4 Tier 2 substepping
		(audit 2026-05-30) handles the Δt/τ bias automatically. The cap-saturation
		warning has moved into ``_integrate_vertical_substeps``."""
		del T_Lu, T_Lv, T_Lw, dt_seconds  # unused

	def _use_static_substeps(self, engine: GPUEngine) -> bool:
		"""Choose the static-shape vs dynamic-masked substep loop (architecture.md §5).

		Explicit constructor flag wins; then the ``GLIDE_STATIC_SUBSTEPS`` env var;
		otherwise auto-select — static on CUDA (constant kernel shapes beat the
		shrinking masked subsets on a launch-bound GPU, and are the prerequisite for
		CUDA-graph capture), dynamic everywhere else (cheaper where launches are free).
		"""

		if self._static_substeps is not None:
			return self._static_substeps
		env = os.environ.get("GLIDE_STATIC_SUBSTEPS", "")
		if env in ("1", "true", "True"):
			return True
		if env in ("0", "false", "False"):
			return False
		return engine.device.type == "cuda"

	def _substep_schedule(
		self, T_Lw: torch.Tensor, dt_seconds: float, dtype: torch.dtype,
	) -> tuple[torch.Tensor, torch.Tensor, int]:
		"""Per-particle substep count ``k_i``, sub-dt, and ``max_k`` (one host sync).

		Shared by both substep-loop variants. ``k_i = ceil(dt / (substep_c · T_Lw_i))``
		clamped to ``[1, max_substeps]``; ``sub_dt_i = dt / k_i``. Also fires the
		once-per-instance substep-cap warning (reusing the ``max_k`` sync). The
		``max_k`` sync that bounds the loop is the last per-step host sync in the
		static path; phase 3 (CUDA-graph capture) replaces it with a fixed loop count.
		"""

		c = self.substep_c
		k_cap = self.max_substeps
		k_required = torch.ceil(dt_seconds / (c * T_Lw.clamp(min=1e-3))).long().clamp(min=1, max=k_cap)
		sub_dt = float(dt_seconds) / k_required.to(dtype=dtype)
		max_k = int(k_required.max().item())

		if not self._warned_substep_cap and max_k >= k_cap:
			at_cap = k_required >= k_cap
			ratio_at_cap = (sub_dt[at_cap] / T_Lw[at_cap]).max().item()
			LOGGER.warning(
				"Hanna turbulence: %d active particles hit max_substeps=%d "
				"(worst sub_dt/T_Lw=%.2f, target < %.2f). The Δt/τ bias is "
				"only partially controlled for these particles. Reduce "
				"simulation.dt_seconds or raise max_substeps.",
				int(at_cap.sum().item()), k_cap, float(ratio_at_cap), float(c),
			)
			self._warned_substep_cap = True

		return k_required, sub_dt, max_k

	def _integrate_vertical_substeps_static(
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
	) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
		"""Static-shape variant of the substep loop (architecture.md §5; device-gated).

		Identical physics to ``_integrate_vertical_substeps`` but every substep
		processes the **full** active set (constant tensor shapes across iterations)
		rather than a shrinking masked subset. A particle that has completed its
		``k_i`` substeps is advanced with ``sub_dt=0``, which is a mathematical no-op:
		``a = exp(0) = 1`` (velocity unchanged), ``variance = σ²(1−a²) = 0`` (no
		noise term), ``displacement = w'·0 = 0`` (position unchanged), and reflection
		of an already-non-negative ``z`` is the identity. So each particle still
		integrates exactly ``k_i`` real substeps; the rest are no-ops.

		This trades extra elementwise FLOPs (finished particles keep being touched)
		for shape-consistency — one kernel shape per op across all iterations, which
		``torch.compile`` fuses cleanly and which CUDA graphs require. It is a win on
		a launch-bound GPU and a loss on CPU, hence the device gate in
		``_use_static_substeps`` (the dynamic masked variant stays the CPU path).

		**Not bit-identical** to the dynamic variant when ``k_i`` varies across
		particles: ``randn`` is drawn for the full set every iteration (vs only the
		owing subset), so the random realisation differs. The two are statistically
		equivalent (and bit-identical when every particle needs the same ``k``).
		"""

		k_required, sub_dt, max_k = self._substep_schedule(T_Lw, dt_seconds, sigma_w.dtype)

		# Full-set working tensors (no per-iteration indexing). Clone so the input
		# subsets aren't mutated.
		xyz = active_xyz.clone()
		u_p = u_prime_in.clone()
		v_p = v_prime_in.clone()
		w_p = w_prime_in.clone()

		sigma_u_var = sigma_u.pow(2)
		sigma_v_var = sigma_v.pow(2)
		sigma_w_var = sigma_w.pow(2)
		# Rogue-trajectory clip bounds (|w'/σ| ≤ 4), precomputed once — constant
		# across substeps because σ is held fixed at the outer-step value.
		u_max = W_PRIME_SIGMA_RATIO_MAX * sigma_u
		v_max = W_PRIME_SIGMA_RATIO_MAX * sigma_v
		w_max = W_PRIME_SIGMA_RATIO_MAX * sigma_w

		for i in range(max_k):
			# Particles still owing a substep get their real sub_dt; finished ones
			# get 0 (no-op). Multiply-mask, never index → constant shape every pass.
			sub_dt_i = torch.where(
				k_required > i, sub_dt, torch.zeros_like(sub_dt),
			)

			# Backward drift with the CURRENT w' (the (1+w'²/σ²) factor must be
			# recomputed per substep — see the dynamic variant's docstring). For a
			# finished particle the factor is computed but multiplied by sub_dt=0.
			factor = 1.0 + w_p.pow(2) / sigma_w_sq
			drift_w = factor * half_dsig2_dz_backward + density_drift_backward

			u_new = engine.update_langevin_velocity(
				u_p, t_lagrangian=T_Lu, sigma_w2=sigma_u_var, dt_seconds=sub_dt_i,
			)
			v_new = engine.update_langevin_velocity(
				v_p, t_lagrangian=T_Lv, sigma_w2=sigma_v_var, dt_seconds=sub_dt_i,
			)
			w_new = engine.update_langevin_velocity(
				w_p, t_lagrangian=T_Lw, sigma_w2=sigma_w_var, dt_seconds=sub_dt_i,
				drift=drift_w,
			)
			w_new = torch.clamp(w_new, min=-w_max, max=w_max)
			u_new = torch.clamp(u_new, min=-u_max, max=u_max)
			v_new = torch.clamp(v_new, min=-v_max, max=v_max)

			xyz = engine.apply_vertical_turbulence(xyz, w_new, dt_seconds=sub_dt_i, backward=True)
			xyz = engine.apply_horizontal_turbulence(xyz, u_new, v_new, dt_seconds=sub_dt_i, backward=True)
			# W&F §6 smooth-wall reflection (joint z, w' flip). A no-op for finished
			# particles (z ≥ 0 from their own final substep's reflection).
			xyz, w_new = engine.reflect_surface(xyz, w_new, z_surface=0.0)

			u_p, v_p, w_p = u_new, v_new, w_new

		return xyz, u_p, v_p, w_p

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
	) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
		"""Per-particle adaptive substepping of the OU + displacement + reflection.

		Each particle gets ``k_i = ceil(dt / (substep_c · T_Lw_i))`` substeps
		(capped at ``self.max_substeps``); inside the loop only particles still
		owing substeps are touched (vectorised via boolean masking). σ², T_L,
		``∂σ²/∂z`` and the density-gradient piece remain fixed at their
		outer-step values — a deliberate simplification (FLEXPART re-evaluates
		per substep, but the cost of per-substep ``_column_turbulence`` calls is
		significant and the σ change across one outer dt is moderate). Documented
		as a known approximation.

		BUT the well-mixed drift's ``(1 + w'²/σ_w²)`` factor IS recomputed each
		substep using the current ``w'`` — holding it fixed at the outer-step
		``w'`` (which is often ≈ 0 for newly-released particles) systematically
		under-drifts particles, biasing them upward (this was caught by
		``test_hanna_well_mixed_no_runaway_lofting`` during F4 Tier 2 development).

		Returns ``(xyz, u', v', w')`` after all substeps; reflection (with
		w'-flip per W&F §6) is applied at the end of each substep so a particle
		that crosses z=0 within a substep is reflected before the next substep.
		"""

		# Per-particle substep count + sub-dt + loop bound (shared with the static
		# variant; also fires the once-per-instance substep-cap warning). The single
		# `max_k` host sync bounds the loop — looping to the static `k_cap` instead
		# would run up to ~max_substeps empty-mask iterations every step.
		k_required, sub_dt, max_k = self._substep_schedule(T_Lw, dt_seconds, sigma_w.dtype)

		# Working tensors. Clone so we don't mutate the input subsets.
		xyz = active_xyz.clone()
		u_p = u_prime_in.clone()
		v_p = v_prime_in.clone()
		w_p = w_prime_in.clone()

		# `mask` is non-empty for every i < max_k by construction (max_k is the
		# max of k_required), so no per-iteration emptiness check / break is
		# needed — and skipping it removes up to ~max_k host syncs per step,
		# which on CUDA were serialising the substep loop.
		for i in range(max_k):
			mask = k_required > i
			sub_dt_m = sub_dt[mask]
			T_Lu_m, T_Lv_m, T_Lw_m = T_Lu[mask], T_Lv[mask], T_Lw[mask]
			sigma_u_m, sigma_v_m, sigma_w_m = sigma_u[mask], sigma_v[mask], sigma_w[mask]
			sigma_w_sq_m = sigma_w_sq[mask]

			# Re-assemble backward drift each substep with the CURRENT w':
			#   drift_b = -½(1 + w'²/σ_w²)·∂σ²/∂z  -  (σ_w²/ρ)·∂ρ/∂z
			# Pieces are pre-negated for the backward sign; the only thing that
			# updates per substep is the (1 + w'²/σ_w²) velocity factor.
			factor = 1.0 + w_p[mask].pow(2) / sigma_w_sq_m
			drift_w_m = factor * half_dsig2_dz_backward[mask] + density_drift_backward[mask]

			u_new = engine.update_langevin_velocity(
				u_p[mask], t_lagrangian=T_Lu_m, sigma_w2=sigma_u_m.pow(2),
				dt_seconds=sub_dt_m,
			)
			v_new = engine.update_langevin_velocity(
				v_p[mask], t_lagrangian=T_Lv_m, sigma_w2=sigma_v_m.pow(2),
				dt_seconds=sub_dt_m,
			)
			w_new = engine.update_langevin_velocity(
				w_p[mask], t_lagrangian=T_Lw_m, sigma_w2=sigma_w_m.pow(2),
				dt_seconds=sub_dt_m, drift=drift_w_m,
			)
			# Rogue-trajectory safeguard (spec §T): clip |w'/σ| at 4× to prevent
			# the (1+w'²/σ²) factor from snowballing the drift when σ_w hits its
			# numerical floor (e.g. at the BL top where stable σ_w → 0) and the
			# substep cap is binding. Without this, particles can NaN out at the
			# BL-top seam under the F4 Tier 2 substepping. FLEXPART has the
			# equivalent clip in its OU implementation.
			w_max = W_PRIME_SIGMA_RATIO_MAX * sigma_w_m
			u_max = W_PRIME_SIGMA_RATIO_MAX * sigma_u_m
			v_max = W_PRIME_SIGMA_RATIO_MAX * sigma_v_m
			w_new = torch.clamp(w_new, min=-w_max, max=w_max)
			u_new = torch.clamp(u_new, min=-u_max, max=u_max)
			v_new = torch.clamp(v_new, min=-v_max, max=v_max)

			xyz_m = xyz[mask]
			xyz_m = engine.apply_vertical_turbulence(
				xyz_m, w_new, dt_seconds=sub_dt_m, backward=True,
			)
			xyz_m = engine.apply_horizontal_turbulence(
				xyz_m, u_new, v_new, dt_seconds=sub_dt_m, backward=True,
			)
			# W&F §6 smooth-wall reflection: joint (z, w') flip on those that
			# crossed the surface within this substep.
			xyz_m, w_new = engine.reflect_surface(xyz_m, w_new, z_surface=0.0)

			xyz[mask] = xyz_m
			u_p[mask] = u_new
			v_p[mask] = v_new
			w_p[mask] = w_new

		return xyz, u_p, v_p, w_p

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

		# Free-troposphere override for z > h. Applied unconditionally (the
		# `torch.where` is a no-op where `above_bl` is false) rather than guarded
		# by `if bool(torch.any(above_bl))` — that guard is a host sync, and this
		# runs 3× per step (σ_w plus the two drift finite-difference probes). The
		# only cost when nothing is above the BL is one extra cached-field
		# grid_sample, which is far cheaper on CUDA than the sync it removes.
		above_bl = z_query > blh
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
			stack = build()
			self._window_field_cache[key] = (ts, stack)
			return stack
		return entry[1]

	@staticmethod
	def _window_mid(
		met_window: HourlyMetTensors, name: str, device: torch.device, dtype: torch.dtype,
	) -> torch.Tensor:
		"""Window-midpoint (``t_alpha=0.5``) value of a met channel ``[Z, Y, X]``."""

		start, end = met_window.channel(name)
		return 0.5 * (start.to(device=device, dtype=dtype) + end.to(device=device, dtype=dtype))

	def _free_trop_fields(
		self,
		met_window: HourlyMetTensors,
		device: torch.device,
		dtype: torch.dtype,
	) -> tuple[torch.Tensor, "GridInterpolationBounds"]:
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
		pressure_pa = torch.as_tensor(
			np.array(met_window.metadata.pressure_level_hpa), device=device, dtype=dtype
		).view(-1, 1, 1) * 100.0

		def _build() -> torch.Tensor:
			# Grid-wide FT σ/T_L stack at the window midpoint; cached per window.
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
			k_z = free_trop_diffusivity(height.clamp(min=T_L_MIN_S), shear_mag, ri)
			sigma_w_ft, t_lw_ft = free_trop_sigma_TL(k_z, n2)
			# FT shear turbulence treated as isotropic (documented simplification);
			# the unresolved-mesoscale "meander" horizontal term is separate.
			return torch.stack([sigma_w_ft, sigma_w_ft, t_lw_ft, t_lw_ft], dim=0)

		fields = self._cached_window_field(met_window, "freetrop", _build)
		return fields, self._grid_bounds(met_window)

	def _density_fields(
		self,
		met_window: HourlyMetTensors,
		device: torch.device,
		dtype: torch.dtype,
	) -> torch.Tensor:
		"""Stack `[ρ, ∂ρ/∂z]` on the met grid `[2, Z, Y, X]` for the
		Stohl-Thomson 1999 density-correction drift term (F2, audit 2026-05-30).

		ρ = p / (R_d · T) per model level (the layer pressure varies with level
		but is horizontally constant per ERA5's pressure-level convention; T
		varies in all three dims). ∂ρ/∂z is a central finite difference of ρ
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
		pressure_pa = torch.as_tensor(
			np.array(met_window.metadata.pressure_level_hpa), device=device, dtype=dtype
		).view(-1, 1, 1) * 100.0

		def _build() -> torch.Tensor:
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
	) -> tuple[torch.Tensor, "GridInterpolationBounds"]:
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
		return fields, self._grid_bounds(met_window)

	@staticmethod
	def _windowed_std(field: torch.Tensor, radius: int) -> torch.Tensor:
		"""Local horizontal std-dev over a (2r+1)² stencil, per level.

		``field`` is [Z, Y, X]; returns [Z, Y, X]. Edge cells average only the valid
		(non-padded) neighbours, so the variance estimate is well-defined at the grid
		boundary."""

		k = 2 * radius + 1
		x = field.unsqueeze(0)  # [1, Z, Y, X] -> treat Z as the channel axis
		mean = torch.nn.functional.avg_pool2d(x, k, stride=1, padding=radius, count_include_pad=False)
		mean_sq = torch.nn.functional.avg_pool2d(x * x, k, stride=1, padding=radius, count_include_pad=False)
		var = (mean_sq - mean.pow(2)).clamp(min=0.0)
		return var.sqrt().squeeze(0)

	@staticmethod
	def _grid_bounds(met_window: HourlyMetTensors) -> "GridInterpolationBounds":
		"""Interpolation bounds for sampling met-grid fields per-particle.

		Passes the per-level AGL array (F9 fix, audit 2026-05-30) so the
		vertical normalisation done by ``GPUEngine.normalize_particle_coordinates``
		uses a proper piecewise-linear level-index lookup rather than the
		linear-in-AGL approximation that was biased for pressure-level data.
		"""

		from lpdm.gpu_engine import GridInterpolationBounds

		lon_arr = met_window.metadata.lon
		lat_arr = met_window.metadata.lat
		level_arr = met_window.metadata.level
		return GridInterpolationBounds(
			lon_first=float(lon_arr[0]),
			lon_last=float(lon_arr[-1]),
			lat_first=float(lat_arr[0]),
			lat_last=float(lat_arr[-1]),
			alt_first=float(level_arr[0]),
			alt_last=float(level_arr[-1]),
			level_agl_m=tuple(float(v) for v in level_arr),
		)

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
