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

from dataclasses import dataclass
from typing import Callable

import torch

from lpdm.runtime import DEVICE


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
	Values can be descending (e.g. lat_first > lat_last)."""

	lon_first: float
	lon_last: float
	lat_first: float
	lat_last: float
	alt_first: float
	alt_last: float


class GPUEngine:
	"""LPDM GPU compute helper class.

	This class currently provides device-safe tensor allocation and core utility
	methods used by upcoming advection and turbulence implementations.
	"""

	def __init__(self, *, device: torch.device | str = DEVICE, dtype: torch.dtype = torch.float32) -> None:
		self.device = torch.device(device)
		self.dtype = dtype

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
		dt_seconds: float,
		noise: torch.Tensor | None = None,
	) -> torch.Tensor:
		"""Update turbulent vertical velocity using a discrete OU/Langevin step."""

		if dt_seconds <= 0:
			raise ValueError("dt_seconds must be > 0")

		w_prev = w_prime.to(device=self.device, dtype=self.dtype)
		tl = torch.as_tensor(t_lagrangian, device=self.device, dtype=self.dtype)
		sigma2 = torch.as_tensor(sigma_w2, device=self.device, dtype=self.dtype)

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

		a = torch.exp(-float(dt_seconds) / tl)
		# For the OU process dw = -(w/T_L)dt + sqrt(2*sigma_w^2/T_L)dW,
		# this exact discrete form preserves the stationary variance sigma_w^2.
		# The altitude test uses the integrated OU variance:
		# Var[z(t)] = 2*sigma_w^2*T_L*(t - T_L*(1 - exp(-t/T_L))).
		variance = torch.clamp(sigma2 * (1.0 - a * a), min=0.0)
		std = torch.sqrt(variance)
		return a * w_prev + std * eta

	def apply_vertical_turbulence(
		self,
		particles: torch.Tensor,
		w_prime: torch.Tensor,
		dt_seconds: float,
		*,
		backward: bool = True,
	) -> torch.Tensor:
		"""Apply vertical displacement from turbulent velocity fluctuations."""

		if particles.ndim != 2 or particles.shape[1] != 4:
			raise ValueError("particles must have shape (N, 4)")
		if dt_seconds <= 0:
			raise ValueError("dt_seconds must be > 0")

		state = particles.to(device=self.device, dtype=self.dtype)
		wp = w_prime.to(device=self.device, dtype=self.dtype)
		if wp.ndim != 1 or wp.shape[0] != state.shape[0]:
			raise ValueError("w_prime must have shape (N,)")

		direction = -1.0 if backward else 1.0
		out = state.clone()
		out[:, 2] = out[:, 2] + direction * wp * float(dt_seconds)
		return out

	def reflect_surface(self, particles: torch.Tensor, *, z_surface: float = 0.0) -> torch.Tensor:
		"""Reflect particles crossing below the lower boundary back into domain."""

		if particles.ndim != 2 or particles.shape[1] != 4:
			raise ValueError("particles must have shape (N, 4)")

		state = particles.to(device=self.device, dtype=self.dtype)
		out = state.clone()
		below = out[:, 2] < float(z_surface)
		out[below, 2] = 2.0 * float(z_surface) - out[below, 2]
		return out

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
