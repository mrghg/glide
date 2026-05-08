"""Particle release initialization module.

All release classes return a standardized tensor shaped (N, 4):
	[lon, lat, alt, weight]

Tensor allocations default to the shared LPDM runtime device so laptop
development uses MPS when available and cloud runs use CUDA.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch

from lpdm.runtime import DEVICE


def _as_1d_tensor(values: Sequence[float] | np.ndarray | torch.Tensor) -> torch.Tensor:
	"""Convert values to a 1D float tensor on CPU for safe preprocessing."""

	if isinstance(values, torch.Tensor):
		return values.detach().to(device="cpu", dtype=torch.float32).flatten()
	return torch.as_tensor(values, dtype=torch.float32, device="cpu").flatten()


@dataclass
class Release(ABC):
	"""Base release abstraction for generating initial particle states."""

	n_particles: int
	device: torch.device | str = DEVICE
	dtype: torch.dtype = torch.float32

	def __post_init__(self) -> None:
		if self.n_particles <= 0:
			raise ValueError("n_particles must be > 0")
		self.device = torch.device(self.device)

	@abstractmethod
	def generate(self) -> torch.Tensor:
		"""Generate particle tensor with shape (N, 4)."""

	def _stack_particle_columns(
		self,
		lon: torch.Tensor,
		lat: torch.Tensor,
		alt: torch.Tensor,
		weight: torch.Tensor,
	) -> torch.Tensor:
		"""Validate and combine columns into standardized particle state tensor."""

		if not (lon.shape == lat.shape == alt.shape == weight.shape):
			raise ValueError("All particle columns must have identical shape")
		return torch.stack((lon, lat, alt, weight), dim=1).to(device=self.device, dtype=self.dtype)


@dataclass
class PointRelease(Release):
	"""Release particles from a single point."""

	lon: float = 0.0
	lat: float = 0.0
	alt: float = 10.0

	def generate(self) -> torch.Tensor:
		lon = torch.full((self.n_particles,), self.lon, dtype=self.dtype, device=self.device)
		lat = torch.full((self.n_particles,), self.lat, dtype=self.dtype, device=self.device)
		alt = torch.full((self.n_particles,), self.alt, dtype=self.dtype, device=self.device)
		weight = torch.full((self.n_particles,), 1.0 / self.n_particles, dtype=self.dtype, device=self.device)
		return self._stack_particle_columns(lon, lat, alt, weight)


@dataclass
class VolumeRelease(Release):
	"""Release particles uniformly throughout a rectangular volume."""

	lon_min: float = -1.0
	lon_max: float = 1.0
	lat_min: float = -1.0
	lat_max: float = 1.0
	alt_min: float = 0.0
	alt_max: float = 100.0

	def generate(self) -> torch.Tensor:
		lon = torch.empty(self.n_particles, dtype=self.dtype, device=self.device).uniform_(self.lon_min, self.lon_max)
		lat = torch.empty(self.n_particles, dtype=self.dtype, device=self.device).uniform_(self.lat_min, self.lat_max)
		alt = torch.empty(self.n_particles, dtype=self.dtype, device=self.device).uniform_(self.alt_min, self.alt_max)
		weight = torch.full((self.n_particles,), 1.0 / self.n_particles, dtype=self.dtype, device=self.device)
		return self._stack_particle_columns(lon, lat, alt, weight)


@dataclass
class FlightTrackRelease(Release):
	"""Release particles along a sequence of flight track coordinates."""

	lons: Sequence[float] = ()
	lats: Sequence[float] = ()
	alts: Sequence[float] = ()

	def generate(self) -> torch.Tensor:
		lon_track = _as_1d_tensor(self.lons)
		lat_track = _as_1d_tensor(self.lats)
		alt_track = _as_1d_tensor(self.alts)

		if not (lon_track.numel() == lat_track.numel() == alt_track.numel()):
			raise ValueError("Flight track lon/lat/alt arrays must have equal length")
		if lon_track.numel() == 0:
			raise ValueError("Flight track must contain at least one coordinate")

		idx = torch.linspace(0, lon_track.numel() - 1, steps=self.n_particles, dtype=torch.float32)
		idx = torch.round(idx).to(torch.long)

		lon = lon_track[idx].to(device=self.device, dtype=self.dtype)
		lat = lat_track[idx].to(device=self.device, dtype=self.dtype)
		alt = alt_track[idx].to(device=self.device, dtype=self.dtype)
		weight = torch.full((self.n_particles,), 1.0 / self.n_particles, dtype=self.dtype, device=self.device)
		return self._stack_particle_columns(lon, lat, alt, weight)


@dataclass
class ColumnRelease(Release):
	"""Vertical column release at a fixed (lon, lat) over a set of altitude levels.

	Particles are placed at user-supplied altitudes (m AGL). When an averaging kernel
	is provided, levels are sampled with probability proportional to the kernel;
	otherwise sampling is uniform across the supplied altitudes. Pressure-mass
	weighting (when wanted, e.g. for satellite retrieval applications) should be
	pre-computed by the caller and passed via `averaging_kernel`.
	"""

	lon: float = 0.0
	lat: float = 0.0
	altitudes_agl_m: Sequence[float] = ()
	averaging_kernel: Sequence[float] | None = None

	def _compute_pdf(self) -> torch.Tensor:
		altitudes = _as_1d_tensor(self.altitudes_agl_m)
		if altitudes.numel() == 0:
			raise ValueError("altitudes_agl_m must contain at least one value")
		if torch.any(altitudes < 0):
			raise ValueError("altitudes_agl_m must be non-negative")

		if self.averaging_kernel is None:
			weights = torch.ones_like(altitudes)
		else:
			weights = _as_1d_tensor(self.averaging_kernel)
			if weights.numel() != altitudes.numel():
				raise ValueError("averaging_kernel length must match altitudes_agl_m length")
			if torch.any(weights < 0):
				raise ValueError("averaging_kernel values must be non-negative")

		total = torch.sum(weights)
		if not torch.isfinite(total) or float(total) <= 0.0:
			raise ValueError("Column sampling weights sum to zero; check averaging_kernel values")
		return weights / total

	def generate(self) -> torch.Tensor:
		altitudes = _as_1d_tensor(self.altitudes_agl_m)
		pdf = self._compute_pdf()

		sampled_idx = torch.multinomial(pdf, num_samples=self.n_particles, replacement=True)
		alt = altitudes[sampled_idx].to(device=self.device, dtype=self.dtype)

		lon = torch.full((self.n_particles,), self.lon, dtype=self.dtype, device=self.device)
		lat = torch.full((self.n_particles,), self.lat, dtype=self.dtype, device=self.device)
		weight = torch.full((self.n_particles,), 1.0 / self.n_particles, dtype=self.dtype, device=self.device)
		return self._stack_particle_columns(lon, lat, alt, weight)
