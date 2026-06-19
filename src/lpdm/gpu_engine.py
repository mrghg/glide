"""GPU physics and advection engine.

Implementation TODO:
- Coordinate normalization for grid_sample
- 3D spatial + 1D temporal interpolation
- RK2 backward advection
- Hanna turbulence parameterization
- Surface reflection
- Particle merge optimization via scatter-reduce
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Callable

import torch

from lpdm.runtime import DEVICE

LOGGER = logging.getLogger(__name__)


# Per-call *value* validation in the hot-path engine methods
# (update_langevin_velocity, apply_vertical/horizontal_turbulence) uses
# `torch.any(... <= 0)` checks. Each one is a device→host sync, and the Hanna
# substep loop calls these hundreds of times per integration step — on CUDA the
# syncs serialise the pipeline and dominate runtime (the GPU stalls waiting for
# the host to read a bool). They guard only against negative dt / σ² / T_L, which
# are positive by construction in the integrator, so they are OFF by default.
# Cheap host-side *shape* checks (ndim/shape) are unaffected and always run.
# Set GLIDE_VALIDATE_ENGINE=1 to re-enable the value checks (tests / debugging).
VALIDATE_ENGINE_INPUTS = os.environ.get("GLIDE_VALIDATE_ENGINE", "") not in ("", "0", "false", "False")


@dataclass(frozen=True)
class CoordinateBounds:
	"""Physical coordinate bounds used for periodic wrapping."""

	lon_min: float
	lon_max: float
	lat_min: float
	lat_max: float
	alt_min: float
	alt_max: float


@dataclass(frozen=True)
class GridInterpolationBounds:
	"""Physical bounds mapping grid index 0 to 1st element, index N-1 to 2nd element.
	Values can be descending (e.g. lat_first > lat_last).

	``level_agl_m`` is the per-level AGL height array (bbox-mean for the met
	window) and is the source-of-truth for the vertical fractional-index
	lookup used by ``normalize_particle_coordinates`` (F9 fix, audit 2026-05-30).
	If ``None``, the vertical mapping falls back to a linear-in-AGL mapping
	between ``alt_first`` and ``alt_last`` — historically used but biased for
	pressure-level data (which is roughly log-linear in altitude). Always pass
	the level array when it's available; the optional fallback is only kept
	for in-place tests of the geometry helper itself.
	"""

	lon_first: float
	lon_last: float
	lat_first: float
	lat_last: float
	alt_first: float
	alt_last: float
	level_agl_m: tuple[float, ...] | None = None


class GPUEngine:
	"""LPDM GPU compute helper class.

	This class currently provides device-safe tensor allocation and core utility
	methods used by upcoming advection and turbulence implementations.
	"""

	def __init__(
		self,
		*,
		device: torch.device | str = DEVICE,
		dtype: torch.dtype = torch.float32,
		compile_hot_paths: bool | None = None,
	) -> None:
		self.device = torch.device(device)
		self.dtype = dtype
		# `torch.compile` the elementwise hot-path methods (OU velocity update +
		# displacement + reflection). Each fuses its internal ops into far fewer
		# CUDA kernels, which helps when the run is launch-bound (the regime after
		# the host-sync removal — see CHECKPOINT 2026-06-18b). Off by default;
		# opt in with GLIDE_COMPILE=1 (or compile_hot_paths=True). Primarily a CUDA
		# win (Inductor/Triton); on CPU Inductor may help or be neutral. NOTE:
		# compiled RNG is statistically equivalent but not bit-identical to eager,
		# and the FIRST call pays a one-time compile cost (seconds).
		if compile_hot_paths is None:
			compile_hot_paths = os.environ.get("GLIDE_COMPILE", "") not in ("", "0", "false", "False")
		self._compile_hot_paths = bool(compile_hot_paths)
		if self._compile_hot_paths:
			self._enable_compiled_hot_paths()

	def _enable_compiled_hot_paths(self) -> None:
		"""Wrap the elementwise hot-path methods with ``torch.compile``.

		Uses ``dynamic=True`` because the Hanna substep loop calls these on
		masked subsets whose size changes each substep — a static compile would
		recompile per shape. ``torch._dynamo`` is set to fall back to eager on any
		graph it can't capture, so enabling this can never hard-fail a run; worst
		case it's a no-op with a warning. Reassigning the bound methods to their
		compiled wrappers is intentional (instance attr shadows the class method).
		"""

		try:
			import torch._dynamo as dynamo

			dynamo.config.suppress_errors = True  # graceful per-graph fallback to eager
			kw = dict(dynamic=True, fullgraph=False)
			self.update_langevin_velocity = torch.compile(self.update_langevin_velocity, **kw)  # type: ignore[method-assign]
			self.apply_vertical_turbulence = torch.compile(self.apply_vertical_turbulence, **kw)  # type: ignore[method-assign]
			self.apply_horizontal_turbulence = torch.compile(self.apply_horizontal_turbulence, **kw)  # type: ignore[method-assign]
			self.reflect_surface = torch.compile(self.reflect_surface, **kw)  # type: ignore[method-assign]
			LOGGER.info(
				"GPUEngine: torch.compile enabled for hot-path methods (dynamic, eager fallback on)"
			)
		except Exception as exc:  # pragma: no cover - defensive; compile is opt-in
			LOGGER.warning("GPUEngine: torch.compile setup failed (%s); running eager", exc)
			self._compile_hot_paths = False

	def to_device(self, tensor: torch.Tensor) -> torch.Tensor:
		"""Move any tensor to the configured compute device/dtype."""

		return tensor.to(device=self.device, dtype=self.dtype)

	def initialize_turbulence_velocity(self, n_particles: int) -> torch.Tensor:
		"""Initialize vertical turbulent velocity state w' for all particles."""

		if n_particles <= 0:
			raise ValueError("n_particles must be > 0")
		return torch.zeros((n_particles,), device=self.device, dtype=self.dtype)

	def normalize_particle_coordinates(
		self,
		particle_xyz: torch.Tensor,
		bounds: GridInterpolationBounds,
	) -> torch.Tensor:
		"""Normalize particle coordinates into grid_sample's [-1, 1] space.

		Args:
			particle_xyz: Tensor shaped (N, 3) with [lon, lat, alt].
			bounds: Exact boundary coordinates matching grid index endpoints.
				When ``bounds.level_agl_m`` is provided, the vertical mapping is
				a piecewise-linear lookup into the per-level AGL array (F9 fix,
				audit 2026-05-30); otherwise it falls back to the legacy
				linear-in-AGL mapping between ``alt_first`` and ``alt_last``.
				The lookup matters because ERA5 pressure-level data is roughly
				log-linear in altitude — the linear-in-metres approximation
				silently warps the vertical wind shear (and σ_w / T_L sampling
				for the Hanna scheme).
		"""

		if particle_xyz.ndim != 2 or particle_xyz.shape[1] != 3:
			raise ValueError("particle_xyz must have shape (N, 3)")

		pts = particle_xyz.to(device=self.device, dtype=self.dtype)

		lon = pts[:, 0]
		lat = pts[:, 1]
		alt = pts[:, 2]

		def scale(vals: torch.Tensor, v0: float, v1: float) -> torch.Tensor:
			denom = float(v1 - v0)
			if denom == 0.0:
				raise ValueError("Grid dimension has 0 span (first == last)")
			return 2.0 * ((vals - v0) / denom) - 1.0

		x = scale(lon, bounds.lon_first, bounds.lon_last)
		y = scale(lat, bounds.lat_first, bounds.lat_last)

		if bounds.level_agl_m is not None:
			# F9: piecewise-linear fractional-level lookup against the per-level
			# AGL array. This handles non-uniform-in-z pressure-level spacing
			# correctly (the levels are roughly log-linear in altitude).
			level_arr = torch.as_tensor(
				bounds.level_agl_m, device=self.device, dtype=self.dtype,
			)
			n_levels = level_arr.numel()
			if n_levels < 2:
				raise ValueError("level_agl_m must have at least 2 entries for vertical interpolation")
			# Determine sorting direction (ARCO ERA5 level arrays may come in
			# descending altitude — pressure-coordinate convention).
			ascending = bool(level_arr[-1] > level_arr[0])
			if not ascending:
				level_arr_sorted = level_arr.flip(0)
			else:
				level_arr_sorted = level_arr
			# torch.searchsorted: returns insertion index i s.t. level[i-1] ≤ alt < level[i].
			# alt is a column slice and may be non-contiguous; force contiguous to
			# avoid the searchsorted perf warning and the implicit internal copy.
			idx_upper = torch.searchsorted(
				level_arr_sorted, alt.contiguous()
			).clamp(min=1, max=n_levels - 1)
			idx_lower = idx_upper - 1
			L_lo = level_arr_sorted[idx_lower]
			L_hi = level_arr_sorted[idx_upper]
			frac_within = (alt - L_lo) / (L_hi - L_lo).clamp(min=1e-6)
			frac_idx_ascending = idx_lower.to(dtype=self.dtype) + frac_within.clamp(0.0, 1.0)
			# Map back to the original level ordering (the grid we sample lives
			# in the original orientation).
			if ascending:
				frac_idx = frac_idx_ascending
			else:
				frac_idx = (n_levels - 1) - frac_idx_ascending
			# Normalised to [-1, 1] for grid_sample.
			z = 2.0 * frac_idx / float(n_levels - 1) - 1.0
		else:
			z = scale(alt, bounds.alt_first, bounds.alt_last)

		# Clamp protects against small numerical drifts beyond interpolation range.
		return torch.stack((x, y, z), dim=1).clamp(-1.0, 1.0)

	def allocate_particle_buffer(self, n_particles: int) -> torch.Tensor:
		"""Allocate an empty particle state buffer shaped (N, 4)."""

		if n_particles <= 0:
			raise ValueError("n_particles must be > 0")
		return torch.empty((n_particles, 4), device=self.device, dtype=self.dtype)

	def rk2_advect_backward(
		self,
		particles: torch.Tensor,
		dt_seconds: float,
		wind_fn: Callable[[torch.Tensor], torch.Tensor],
	) -> torch.Tensor:
		"""Backward-integrate particle positions with 2nd-order Runge-Kutta.

		Args:
			particles: Tensor shaped (N, 4) with [x, y, z, weight].
			dt_seconds: Positive timestep length in seconds.
			wind_fn: Callable receiving xyz tensor (N, 3) and returning velocity
				tensor (N, 3) with [u, v, w].
		"""

		if particles.ndim != 2 or particles.shape[1] != 4:
			raise ValueError("particles must have shape (N, 4)")
		if dt_seconds <= 0:
			raise ValueError("dt_seconds must be > 0")

		state = particles.to(device=self.device, dtype=self.dtype)
		xyz = state[:, :3]

		v1 = wind_fn(xyz)
		if v1.shape != xyz.shape:
			raise ValueError("wind_fn must return velocity tensor shaped (N, 3)")

		x_temp = xyz - 0.5 * float(dt_seconds) * v1
		v2 = wind_fn(x_temp)
		if v2.shape != xyz.shape:
			raise ValueError("wind_fn must return velocity tensor shaped (N, 3)")

		xyz_new = xyz - float(dt_seconds) * v2
		out = state.clone()
		out[:, :3] = xyz_new
		return out

	def update_langevin_velocity(
		self,
		w_prime: torch.Tensor,
		t_lagrangian: torch.Tensor | float,
		sigma_w2: torch.Tensor | float,
		dt_seconds: torch.Tensor | float,
		noise: torch.Tensor | None = None,
		drift: torch.Tensor | float = 0.0,
	) -> torch.Tensor:
		"""Update turbulent velocity using a discrete OU/Langevin step.

		`drift` is an optional deterministic acceleration (m/s²) added as
		`drift * dt`. For the vertical component it carries the Thomson (1987)
		well-mixed correction `½(1 + w²/σ²)·∂σ²/∂z`, which keeps an initially
		well-mixed tracer well-mixed in inhomogeneous turbulence (σ varying with
		height). Omitting it (the default `drift=0`) lets particles spuriously
		accumulate in low-turbulence regions — the cause of GLIDE's surface
		footprint under-dispersion. The homogeneous part stays the exact-OU form
		(stationary variance σ²); the drift is applied with a forward-Euler
		increment, matching the FLEXPART discretization.

		`dt_seconds` may be a scalar (uniform timestep) OR a per-particle tensor
		(adaptive substepping, F4 Tier 2 / audit 2026-05-30). In the tensor case
		each particle integrates its own sub-step length, which is needed when
		particles in the active set have very different Lagrangian timescales
		(near-surface T_L can drop to a few seconds while aloft T_L is hundreds).
		"""

		dt = torch.as_tensor(dt_seconds, device=self.device, dtype=self.dtype)
		w_prev = w_prime.to(device=self.device, dtype=self.dtype)
		tl = torch.as_tensor(t_lagrangian, device=self.device, dtype=self.dtype)
		sigma2 = torch.as_tensor(sigma_w2, device=self.device, dtype=self.dtype)

		if VALIDATE_ENGINE_INPUTS:
			if torch.any(dt <= 0):
				raise ValueError("dt_seconds must be > 0")
			if torch.any(tl <= 0):
				raise ValueError("t_lagrangian must be strictly positive")
			if torch.any(sigma2 < 0):
				raise ValueError("sigma_w2 must be non-negative")

		if noise is None:
			eta = torch.randn_like(w_prev)
		else:
			eta = noise.to(device=self.device, dtype=self.dtype)
			if eta.shape != w_prev.shape:
				raise ValueError("noise must have same shape as w_prime")

		a = torch.exp(-dt / tl)
		# For the OU process dw = -(w/T_L)dt + sqrt(2*sigma_w^2/T_L)dW,
		# this exact discrete form preserves the stationary variance sigma_w^2.
		# The altitude test uses the integrated OU variance:
		# Var[z(t)] = 2*sigma_w^2*T_L*(t - T_L*(1 - exp(-t/T_L))).
		variance = torch.clamp(sigma2 * (1.0 - a * a), min=0.0)
		std = torch.sqrt(variance)
		drift_tensor = torch.as_tensor(drift, device=self.device, dtype=self.dtype)
		drift_term = drift_tensor * dt
		return a * w_prev + drift_term + std * eta

	def apply_vertical_turbulence(
		self,
		particles: torch.Tensor,
		w_prime: torch.Tensor,
		dt_seconds: torch.Tensor | float,
		*,
		backward: bool = True,
	) -> torch.Tensor:
		"""Apply vertical displacement from turbulent velocity fluctuations.

		`dt_seconds` may be a scalar or a per-particle tensor (F4 Tier 2
		substepping). When per-particle, each particle is displaced by its own
		`w_prime * sub_dt`."""

		if particles.ndim != 2 or particles.shape[1] != 4:
			raise ValueError("particles must have shape (N, 4)")

		dt = torch.as_tensor(dt_seconds, device=self.device, dtype=self.dtype)
		if VALIDATE_ENGINE_INPUTS and torch.any(dt <= 0):
			raise ValueError("dt_seconds must be > 0")

		state = particles.to(device=self.device, dtype=self.dtype)
		wp = w_prime.to(device=self.device, dtype=self.dtype)
		if wp.ndim != 1 or wp.shape[0] != state.shape[0]:
			raise ValueError("w_prime must have shape (N,)")

		direction = -1.0 if backward else 1.0
		out = state.clone()
		out[:, 2] = out[:, 2] + direction * wp * dt
		return out

	def apply_horizontal_turbulence(
		self,
		particles: torch.Tensor,
		u_prime: torch.Tensor,
		v_prime: torch.Tensor,
		dt_seconds: torch.Tensor | float,
		*,
		backward: bool = True,
	) -> torch.Tensor:
		"""Apply horizontal displacement from turbulent velocity fluctuations.

		Mirrors `apply_vertical_turbulence` but in lon/lat with the cos-lat correction
		on lon. `u_prime` is the zonal perturbation in m/s, `v_prime` is meridional.
		`dt_seconds` may be a scalar or a per-particle tensor (F4 Tier 2 substepping).
		"""

		if particles.ndim != 2 or particles.shape[1] != 4:
			raise ValueError("particles must have shape (N, 4)")

		dt = torch.as_tensor(dt_seconds, device=self.device, dtype=self.dtype)
		if VALIDATE_ENGINE_INPUTS and torch.any(dt <= 0):
			raise ValueError("dt_seconds must be > 0")

		state = particles.to(device=self.device, dtype=self.dtype)
		up = u_prime.to(device=self.device, dtype=self.dtype)
		vp = v_prime.to(device=self.device, dtype=self.dtype)
		if up.ndim != 1 or up.shape[0] != state.shape[0]:
			raise ValueError("u_prime must have shape (N,)")
		if vp.ndim != 1 or vp.shape[0] != state.shape[0]:
			raise ValueError("v_prime must have shape (N,)")

		direction = -1.0 if backward else 1.0
		out = state.clone()

		lat_deg = state[:, 1]
		meters_per_deg_lat = 110540.0
		meters_per_deg_lon = 111320.0 * torch.cos(torch.deg2rad(lat_deg)).abs().clamp(min=0.05)

		out[:, 0] = out[:, 0] + direction * up * dt / meters_per_deg_lon
		out[:, 1] = out[:, 1] + direction * vp * dt / meters_per_deg_lat
		return out

	def reflect_surface(
		self,
		particles: torch.Tensor,
		w_prime: torch.Tensor,
		*,
		z_surface: float = 0.0,
	) -> tuple[torch.Tensor, torch.Tensor]:
		"""Reflect particles crossing below the lower boundary back into domain.

		Per Wilson & Flesch (1993) §6, smooth-wall reflection is the joint mapping
		``(z, w) → (2·z_surface − z, −w)`` — both position AND vertical perturbation
		velocity must be reversed. Reflecting only `z` (the older behaviour) leaves
		each reflected particle pointing downward into the boundary for ~τ_L worth
		of steps, biasing near-surface residence time and inflating the surface
		footprint. See `docs/turbulence.md` §3.2.4 and `docs/LPDM_physics_spec.md`
		§B for the derivation and the WMC consequence.

		Returns ``(particles, w_prime)`` with the same shapes as inputs; entries
		that did not reflect are unchanged.
		"""

		if particles.ndim != 2 or particles.shape[1] != 4:
			raise ValueError("particles must have shape (N, 4)")
		if w_prime.ndim != 1 or w_prime.shape[0] != particles.shape[0]:
			raise ValueError("w_prime must have shape (N,) matching particles")

		state = particles.to(device=self.device, dtype=self.dtype)
		wp = w_prime.to(device=self.device, dtype=self.dtype)

		below = state[:, 2] < float(z_surface)
		reflected_z = 2.0 * float(z_surface) - state[:, 2]
		out = state.clone()
		out[:, 2] = torch.where(below, reflected_z, state[:, 2])
		wp_out = torch.where(below, -wp, wp)
		return out, wp_out

	def diffuse_positions_periodic(
		self,
		particles: torch.Tensor,
		diffusivity: float,
		dt_seconds: float,
		bounds: CoordinateBounds,
	) -> torch.Tensor:
		"""Apply isotropic diffusion and periodic wrapping in all 3 dimensions."""

		if particles.ndim != 2 or particles.shape[1] != 4:
			raise ValueError("particles must have shape (N, 4)")
		if diffusivity < 0:
			raise ValueError("diffusivity must be >= 0")
		if dt_seconds <= 0:
			raise ValueError("dt_seconds must be > 0")

		state = particles.to(device=self.device, dtype=self.dtype)
		out = state.clone()

		std = (2.0 * float(diffusivity) * float(dt_seconds)) ** 0.5
		dxyz = torch.randn((state.shape[0], 3), device=self.device, dtype=self.dtype) * std
		out[:, :3] = out[:, :3] + dxyz

		def wrap(vals: torch.Tensor, vmin: float, vmax: float) -> torch.Tensor:
			span = float(vmax - vmin)
			if span <= 0.0:
				raise ValueError("Invalid periodic bounds: max must be > min")
			return torch.remainder(vals - vmin, span) + vmin

		out[:, 0] = wrap(out[:, 0], bounds.lon_min, bounds.lon_max)
		out[:, 1] = wrap(out[:, 1], bounds.lat_min, bounds.lat_max)
		out[:, 2] = wrap(out[:, 2], bounds.alt_min, bounds.alt_max)
		return out
