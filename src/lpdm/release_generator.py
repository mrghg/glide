"""Particle release initialization module.

All release classes return a standardized tensor shaped (N, 4):
	[lon, lat, alt, weight]

Tensor allocations default to the shared LPDM runtime device so laptop
development uses MPS when available and cloud runs use CUDA.

For multi-release schedules, the batch-aware :func:`generate_batch_particles`
consumes a :class:`~lpdm.config.ReleaseBatch` and produces a
:class:`ParticleBatch` carrying particles plus per-particle sidecar tensors
(``release_idx``, per-particle release time relative to the batch start). The
runtime cursor loop uses those sidecars to determine when each particle
activates. For a single-release batch the result is bit-equivalent to the
legacy ``PointRelease`` + manual release-offset draw, so existing single-release
runs are unchanged.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Sequence

import numpy as np
import torch

from lpdm.runtime import DEVICE

if TYPE_CHECKING:
	from lpdm.config import ReleaseBatch


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


@dataclass(frozen=True)
class ParticleBatch:
	"""Particles plus per-particle release metadata for one :class:`ReleaseBatch`.

	All sidecar tensors are aligned with the leading dimension of ``particles``:
	``release_idx[i]`` is the originating :class:`~lpdm.config.ConcreteRelease`'s
	``release_idx`` for particle ``i``; ``release_time_offsets_s[i]`` is that
	particle's absolute release time expressed as seconds after
	``batch_start_time``; ``release_window_end_offsets_s[i]`` is the end of the
	release window that emitted particle ``i`` (also as an offset from
	``batch_start_time``) — used by the runtime to compute per-particle
	time-ago bins for footprint accumulation. Storing offsets rather than
	absolute Unix timestamps keeps the values in a float32-safe range (the
	float-precision bug fixed in the 2026-05-07 M0 work was the same issue at
	the single-release scale).

	The active-mask check in the runtime cursor loop is a per-particle pair of
	comparisons:

	* upper bound (cursor has stepped back to the particle's release time):
	  ``release_time_offsets_s >= cursor_offset_s``
	* lower bound (cursor still inside the particle's backward window):
	  ``release_time_offsets_s - sim.length_seconds <= cursor_offset_s``

	where ``cursor_offset_s = t_cursor.timestamp() - batch_start_time.timestamp()``.

	The per-particle time-ago bin for the footprint is
	``floor((release_window_end_offsets_s - cursor_offset_s) / 3600)`` clamped to
	the gridder's time dimension.
	"""

	particles: torch.Tensor                      # (N, 4) [lon, lat, alt, weight]
	release_idx: torch.Tensor                    # (N,) int64
	release_time_offsets_s: torch.Tensor         # (N,) float32, seconds after batch_start_time
	release_window_end_offsets_s: torch.Tensor   # (N,) float32, seconds after batch_start_time
	batch_start_time: datetime                   # earliest release_time in the batch (UTC)


def generate_batch_particles(
	batch: "ReleaseBatch",
	*,
	device: torch.device | str = DEVICE,
	dtype: torch.dtype = torch.float32,
) -> ParticleBatch:
	"""Build a :class:`ParticleBatch` for a (possibly multi-release) batch.

	Per-release semantics:

	* Each release contributes ``rel.n_particles`` particles at its constant
	  ``(lon, lat, alt_agl_m)`` with weight ``1 / rel.n_particles`` — i.e. the
	  same point cloud :class:`PointRelease` produces today.
	* Within-release release-time offsets are drawn uniformly in
	  ``[0, rel.duration_seconds)`` using a CPU :class:`torch.Generator` seeded
	  with ``rel.seed`` when non-null, then moved to ``device``. The two-step
	  CPU-seed-then-move pattern mirrors the legacy code in ``main.py`` so a
	  single-release batch is bit-equivalent to today's path.
	* Each particle's absolute release time is ``release.release_time + within``,
	  stored as seconds after ``batch_start_time`` (the earliest release in the
	  batch).

	For a one-release batch with ``batch_start_time == release.release_time``,
	``release_time_offsets_s`` equals the within-release draws — bit-identical
	to the legacy ``release_offsets_s`` tensor.
	"""

	if not batch.releases:
		raise ValueError("ReleaseBatch must contain at least one release")

	device_t = torch.device(device)
	batch_start = min(r.release_time for r in batch.releases)

	particle_chunks: list[torch.Tensor] = []
	release_idx_chunks: list[torch.Tensor] = []
	offset_chunks: list[torch.Tensor] = []
	window_end_offset_chunks: list[torch.Tensor] = []

	for rel in batch.releases:
		n = rel.n_particles
		if n <= 0:
			raise ValueError(
				f"ConcreteRelease(release_idx={rel.release_idx}) has n_particles={n}; must be > 0"
			)

		lon = torch.full((n,), rel.lon, dtype=dtype, device=device_t)
		lat = torch.full((n,), rel.lat, dtype=dtype, device=device_t)
		alt = torch.full((n,), rel.alt_agl_m, dtype=dtype, device=device_t)
		weight = torch.full((n,), 1.0 / n, dtype=dtype, device=device_t)
		particle_chunks.append(torch.stack((lon, lat, alt, weight), dim=1))

		release_idx_chunks.append(
			torch.full((n,), rel.release_idx, dtype=torch.int64, device=device_t)
		)

		duration_s = float(rel.duration_seconds)
		if rel.seed is not None:
			rng = torch.Generator(device="cpu")
			rng.manual_seed(int(rel.seed))
			within = torch.empty(n, device="cpu", dtype=torch.float32).uniform_(
				0.0, duration_s, generator=rng
			).to(device=device_t)
		else:
			within = torch.empty(n, device=device_t, dtype=torch.float32).uniform_(
				0.0, duration_s
			)

		rel_offset_s = float((rel.release_time - batch_start).total_seconds())
		offset_chunks.append(within + rel_offset_s)
		# All particles from this release share the same window-end offset
		# (= release start offset + release duration). Used by the runtime to
		# compute per-particle time-ago bins for footprint accumulation.
		window_end_offset_chunks.append(
			torch.full((n,), rel_offset_s + duration_s, dtype=torch.float32, device=device_t)
		)

	return ParticleBatch(
		particles=torch.cat(particle_chunks, dim=0),
		release_idx=torch.cat(release_idx_chunks, dim=0),
		release_time_offsets_s=torch.cat(offset_chunks, dim=0),
		release_window_end_offsets_s=torch.cat(window_end_offset_chunks, dim=0),
		batch_start_time=batch_start,
	)
