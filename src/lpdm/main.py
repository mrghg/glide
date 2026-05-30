"""Top-level LPDM orchestration entry point.

This module provides a minimal end-to-end backward trajectory run mode suitable
for early cloud testing (including Vertex AI user-managed notebooks).

The CLI surface is intentionally small: a YAML ``--config`` plus a few overrides
(``--device``, ``--output-uri``, ``--start-time``). Schema is defined in
:mod:`lpdm.config`.
"""

from __future__ import annotations

import argparse
import gc
import logging
import os
import resource
import sys
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Sequence

import numpy as np
import torch

from lpdm.config import ConcreteRelease, RunConfig, _release_point
from lpdm.footprint_gridder import FootprintGridder
from lpdm.gpu_engine import GPUEngine, GridInterpolationBounds
from lpdm.met_reader import (
	ArcoEra5ZarrReader,
	BoundingBoxRequest,
	HourlyMetTensors,
	MetReader,
	SpatialBounds,
	TimeBounds,
)
from lpdm.output_writer import OutputWriter
from lpdm.release_generator import generate_batch_particles
from lpdm.runtime import DEVICE
from lpdm.turbulence import TurbulenceScheme, get_scheme, list_schemes


LOGGER = logging.getLogger(__name__)


FOOTPRINT_UNITS_DOC = {
	"value": "Σ over active particles in cell of (weight_i × dt_step_seconds)",
	"value_dimensionality": "(dimensionless mass fraction) × seconds — residence time per unit released mass",
	"particle_weight_convention": "1/n_particles for uniform releases; each particle represents 1/N of the total released mass",
	"notes": (
		"Raw residence-time accumulator; downstream code is responsible for converting to a physical "
		"sensitivity (e.g. ppm per emission flux), which requires cell volume, mixed-layer thickness, "
		"air density, and molar-mass conversions. See Lin et al. 2003 (STILT) or Seibert & Frank 2004 "
		"for standard recipes."
	),
}


class PreflightValidationError(ValueError):
	"""Raised when run inputs are invalid before stepping begins."""


def _resolve_device(name: str) -> str:
	"""Resolve a config device string ('auto', 'cpu', 'cuda', 'mps', 'cuda:N') to a concrete torch device."""

	if name == "auto":
		return str(DEVICE)
	return name


def _current_rss_bytes() -> int | None:
	"""Return process memory bytes when available."""

	try:
		with open("/proc/self/status", "r", encoding="utf-8") as f:
			for line in f:
				if line.startswith("VmRSS:"):
					parts = line.split()
					if len(parts) >= 2:
						return int(parts[1]) * 1024
	except OSError:
		pass

	try:
		ru_maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
		if sys.platform == "darwin":
			return int(ru_maxrss)
		return int(ru_maxrss) * 1024
	except OSError:
		return None


def _device_memory_bytes(device: torch.device) -> tuple[int | None, int | None]:
	if device.type == "cuda" and torch.cuda.is_available():
		idx = device.index if device.index is not None else torch.cuda.current_device()
		return int(torch.cuda.memory_allocated(idx)), int(torch.cuda.memory_reserved(idx))

	if device.type == "mps" and hasattr(torch, "mps"):
		allocated_fn = getattr(torch.mps, "current_allocated_memory", None)
		driver_fn = getattr(torch.mps, "driver_allocated_memory", None)
		allocated = int(allocated_fn()) if callable(allocated_fn) else None
		reserved = int(driver_fn()) if callable(driver_fn) else None
		return allocated, reserved

	return None, None


def _format_gib(num_bytes: int | None) -> str:
	if num_bytes is None:
		return "n/a"
	return f"{num_bytes / (1024 ** 3):.3f} GiB"


def _hour_floor(dt: datetime) -> datetime:
	return dt.astimezone(timezone.utc).replace(minute=0, second=0, microsecond=0)


def _footprint_time_bin_index(release_end: datetime, t_cursor: datetime, n_time_bins: int) -> int:
	"""Map the current backward integration cursor to a 0-based time_ago bin."""

	if n_time_bins <= 0:
		raise ValueError("n_time_bins must be > 0")

	elapsed_seconds = max(0.0, release_end.timestamp() - t_cursor.timestamp())
	return min(n_time_bins - 1, int(elapsed_seconds / 3600.0))


def _build_footprint_dataset_metadata(
	cfg: RunConfig,
	gridder: FootprintGridder,
	releases: Sequence[ConcreteRelease],
) -> tuple[dict[str, object], dict[str, float]]:
	"""Build coordinate metadata for the 5D footprint Zarr dataset.

	The leading ``release_time`` axis is populated from the full expanded
	schedule — one timestamp per :class:`ConcreteRelease` in execution order.
	For single-release runs this is a length-1 axis carrying the release's
	start time. Release attrs (lon/lat/alt) come from the schedule's common
	release point (currently all variants use a single location).
	"""

	lon_edges = np.linspace(gridder.lon_min, gridder.lon_max, gridder.n_x + 1, dtype=np.float64)
	lat_edges = np.linspace(gridder.lat_min, gridder.lat_max, gridder.n_y + 1, dtype=np.float64)
	z_edges_m = gridder.z_edges.detach().cpu().numpy().astype(np.float64)
	hours_per_bin = max(
		1.0, float(cfg.simulation.length_seconds) / 3600.0 / max(1, gridder.n_t)
	)
	time_bin_start_h = np.arange(gridder.n_t, dtype=np.float64) * hours_per_bin
	time_bin_end_h = time_bin_start_h + hours_per_bin

	release_times = np.array(
		[np.datetime64(rel.release_time.replace(tzinfo=None), "ns") for rel in releases]
	)
	release_durations_s = np.array(
		[float(rel.duration_seconds) for rel in releases], dtype=np.float64
	)

	coords: dict[str, object] = {
		"release_time": release_times,
		"release_duration_seconds": ("release_time", release_durations_s),
		"time_ago": np.arange(gridder.n_t, dtype=np.int64),
		"time_ago_start_hours": ("time_ago", time_bin_start_h),
		"time_ago_end_hours": ("time_ago", time_bin_end_h),
		"z_bin": 0.5 * (z_edges_m[:-1] + z_edges_m[1:]),
		"z_bottom_m": ("z_bin", z_edges_m[:-1]),
		"z_top_m": ("z_bin", z_edges_m[1:]),
		"latitude": 0.5 * (lat_edges[:-1] + lat_edges[1:]),
		"longitude": 0.5 * (lon_edges[:-1] + lon_edges[1:]),
		"latitude_edge": ("latitude_edge", lat_edges),
		"longitude_edge": ("longitude_edge", lon_edges),
	}
	lon, lat, alt_agl_m = _release_point(cfg.release)
	attrs = {
		"release_lon": float(lon),
		"release_lat": float(lat),
		"release_alt_agl_m": float(alt_agl_m),
	}
	return coords, attrs


def _validate_meteorology_time_coverage(
	reader: MetReader,
	cfg: RunConfig,
	release_end: datetime,
	sim_start: datetime,
) -> None:
	"""Fail early when the configured run requires hours outside dataset coverage."""

	available_start, available_end = reader.get_time_coverage()
	required_start = _hour_floor(sim_start)
	required_end = _hour_floor(release_end) + timedelta(hours=1)

	if required_start < available_start or required_end > available_end:
		raise PreflightValidationError(
			"Meteorological dataset does not cover the requested simulation window. "
			f"Need hourly data from {required_start.isoformat()} through {required_end.isoformat()}, "
			f"but dataset only covers {available_start.isoformat()} through {available_end.isoformat()}. "
			"Reduce simulation.length_seconds, choose a different start_time, or use a dataset with wider time coverage."
		)


@dataclass(frozen=True)
class OutputPaths:
	endpoint_particles: str
	trajectory_diagnostics: str
	footprints: str
	metadata: str


@dataclass
class MemoryStats:
	peak_rss_bytes: int = 0
	peak_device_allocated_bytes: int = 0
	peak_device_reserved_bytes: int = 0

	def observe(self, device: torch.device) -> tuple[int | None, int | None, int | None]:
		rss_bytes = _current_rss_bytes()
		allocated, reserved = _device_memory_bytes(device)

		if rss_bytes is not None:
			self.peak_rss_bytes = max(self.peak_rss_bytes, rss_bytes)
		if allocated is not None:
			self.peak_device_allocated_bytes = max(self.peak_device_allocated_bytes, allocated)
		if reserved is not None:
			self.peak_device_reserved_bytes = max(self.peak_device_reserved_bytes, reserved)

		return rss_bytes, allocated, reserved


def _build_arg_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Run a backward LPDM trajectory simulation from a YAML config")
	parser.add_argument("--config", required=True, help="Path to the run YAML config (see configs/example_mhd_january.yaml)")
	parser.add_argument("--device", default=None, help="Override simulation.device")
	parser.add_argument("--output-uri", default=None, help="Override io.output_uri")
	parser.add_argument("--start-time", default=None, help="Override simulation.start_time (UTC ISO)")
	return parser


def _build_output_paths(output_uri: str) -> OutputPaths:
	base_uri = output_uri.rstrip("/")
	return OutputPaths(
		endpoint_particles=f"{base_uri}/endpoint_particles.parquet",
		trajectory_diagnostics=f"{base_uri}/trajectory_diagnostics.parquet",
		footprints=f"{base_uri}/footprints.zarr",
		metadata=f"{base_uri}/run_metadata.json",
	)


def _config_metadata(cfg: RunConfig, release_end: datetime, sim_start: datetime) -> dict[str, object]:
	# mode="json" serializes datetimes / tuples to JSON-friendly types.
	return {
		**cfg.model_dump(mode="json"),
		"release_end_time": release_end.isoformat(),
		"simulation_start_time": sim_start.isoformat(),
	}


def _runtime_metadata(
	cfg: RunConfig,
	step_count: int,
	hour_windows: int,
	memory_stats: MemoryStats,
	*,
	status: str,
	**extra: object,
) -> dict[str, object]:
	runtime = {
		"device": _resolve_device(cfg.simulation.device),
		"status": status,
		"steps": step_count,
		"hour_windows": hour_windows,
		"met_cache_max_hours": cfg.memory.met_cache_max_hours,
		"peak_rss_bytes": memory_stats.peak_rss_bytes,
		"peak_device_allocated_bytes": memory_stats.peak_device_allocated_bytes,
		"peak_device_reserved_bytes": memory_stats.peak_device_reserved_bytes,
	}
	runtime.update(extra)
	return runtime


def _memory_guard_enabled(cfg: RunConfig) -> bool:
	mem = cfg.memory
	return any(
		limit is not None
		for limit in (
			mem.guard_max_rss_gib,
			mem.guard_max_device_allocated_gib,
			mem.guard_max_device_reserved_gib,
		)
	)


def _write_memory_guard_metadata(
	writer: OutputWriter,
	metadata_path: str,
	cfg: RunConfig,
	release_end: datetime,
	sim_start: datetime,
	step_count: int,
	hour_windows: int,
	outputs: OutputPaths,
	memory_stats: MemoryStats,
	*,
	reason: str,
	**runtime_extra: object,
) -> None:
	writer.write_metadata_json(
		metadata_path,
		{
			"status": "aborted_memory_guard",
			"reason": reason,
			"config": _config_metadata(cfg, release_end, sim_start),
			"runtime": _runtime_metadata(
				cfg,
				step_count,
				hour_windows,
				memory_stats,
				status="aborted_memory_guard",
				**runtime_extra,
			),
			"outputs": {
				"endpoint_particles": outputs.endpoint_particles,
				"trajectory_diagnostics": outputs.trajectory_diagnostics,
				"footprints": outputs.footprints,
				"metadata": outputs.metadata,
			},
			"footprint_units": FOOTPRINT_UNITS_DOC,
		},
	)


def _raise_if_memory_guard_exceeded(
	writer: OutputWriter,
	metadata_path: str,
	cfg: RunConfig,
	release_end: datetime,
	sim_start: datetime,
	step_count: int,
	hour_windows: int,
	outputs: OutputPaths,
	memory_stats: MemoryStats,
	rss_bytes: int | None,
	allocated: int | None,
	reserved: int | None,
) -> None:
	mem = cfg.memory
	guard_checks = (
		(
			mem.guard_max_rss_gib,
			rss_bytes,
			"RSS exceeded memory guard threshold",
			"RSS",
			"guard_limit_rss_bytes",
			"guard_observed_rss_bytes",
		),
		(
			mem.guard_max_device_allocated_gib,
			allocated,
			"Device allocated memory exceeded memory guard threshold",
			"device allocated memory",
			"guard_limit_device_allocated_bytes",
			"guard_observed_device_allocated_bytes",
		),
		(
			mem.guard_max_device_reserved_gib,
			reserved,
			"Device reserved memory exceeded memory guard threshold",
			"device reserved memory",
			"guard_limit_device_reserved_bytes",
			"guard_observed_device_reserved_bytes",
		),
	)

	for limit_gib, observed_bytes, reason, label, limit_key, observed_key in guard_checks:
		if limit_gib is None or observed_bytes is None:
			continue

		limit_bytes = int(limit_gib * (1024 ** 3))
		if observed_bytes <= limit_bytes:
			continue

		_write_memory_guard_metadata(
			writer,
			metadata_path,
			cfg,
			release_end,
			sim_start,
			step_count,
			hour_windows,
			outputs,
			memory_stats,
			reason=reason,
			**{limit_key: limit_bytes, observed_key: observed_bytes},
		)
		raise MemoryError(
			f"Memory safety guard triggered: {label} reached "
			f"{_format_gib(observed_bytes)} which is above {limit_gib:.3f} GiB"
		)


def _met_domain_bounds(cfg: RunConfig) -> SpatialBounds:
	md = cfg.met_domain
	return SpatialBounds(
		lon_min=md.lon_bounds[0],
		lon_max=md.lon_bounds[1],
		lat_min=md.lat_bounds[0],
		lat_max=md.lat_bounds[1],
		z_min=0.0,
		z_max=md.alt_max_m,
	)


def _within_met_domain(particles: torch.Tensor, cfg: RunConfig) -> torch.Tensor:
	"""Boolean [N] mask: True where a particle is inside ``met_domain``.

	Outside the lon/lat bbox or above ``alt_max_m`` there is no valid met — the
	engine's coordinate normalisation clamps to the grid edge, so such a particle
	would otherwise keep being pushed by edge winds for the rest of the
	integration (wasted compute; it can't contribute to the footprint either,
	being outside ``output_grid``). The runtime kills these particles. The z=0
	floor is handled by surface reflection, so there is no lower-altitude kill.
	"""

	md = cfg.met_domain
	lon = particles[:, 0]
	lat = particles[:, 1]
	alt = particles[:, 2]
	return (
		(lon >= md.lon_bounds[0])
		& (lon <= md.lon_bounds[1])
		& (lat >= md.lat_bounds[0])
		& (lat <= md.lat_bounds[1])
		& (alt <= md.alt_max_m)
	)


def _get_hourly_met_window(
	reader: MetReader,
	cfg: RunConfig,
	t_cursor: datetime,
	met_cache: OrderedDict[datetime, HourlyMetTensors],
) -> tuple[HourlyMetTensors, int]:
	hour_key = _hour_floor(t_cursor)
	if cfg.memory.met_cache_max_hours > 0 and hour_key in met_cache:
		met_cache.move_to_end(hour_key)
		return met_cache[hour_key], 0

	met_window = reader.fetch_hourly_window(
		BoundingBoxRequest(
			spatial=_met_domain_bounds(cfg),
			time=TimeBounds(start=hour_key, end=hour_key + timedelta(hours=1)),
		)
	)

	if cfg.memory.met_cache_max_hours > 0:
		met_cache[hour_key] = met_window
		while len(met_cache) > cfg.memory.met_cache_max_hours:
			_, evicted = met_cache.popitem(last=False)
			del evicted

	return met_window, 1


def _advect_active_particles(
	engine: GPUEngine,
	device: torch.device,
	active_particles: torch.Tensor,
	met_window: HourlyMetTensors,
	t_cursor: datetime,
	delta_s: float,
	dtype: torch.dtype,
) -> tuple[torch.Tensor, float, tuple[float, float, float]]:
	"""RK2 backward advection on the active particle subset."""

	t_start_s = met_window.metadata.time_start.timestamp()
	t_end_s = met_window.metadata.time_end.timestamp()

	t_eval_s = t_cursor.timestamp() - 0.5 * delta_s
	alpha = (t_eval_s - t_start_s) / max(1.0, float(t_end_s - t_start_s))
	alpha = max(0.0, min(1.0, alpha))

	grid_bounds = GridInterpolationBounds(
		lon_first=float(met_window.metadata.lon[0]),
		lon_last=float(met_window.metadata.lon[-1]),
		lat_first=float(met_window.metadata.lat[0]),
		lat_last=float(met_window.metadata.lat[-1]),
		alt_first=float(met_window.metadata.level[0]),
		alt_last=float(met_window.metadata.level[-1]),
	)

	m_start_uvw, m_end_uvw = met_window.channels("u", "v", "w")
	m_start = m_start_uvw.unsqueeze(0).to(device=device, dtype=dtype)
	m_end = m_end_uvw.unsqueeze(0).to(device=device, dtype=dtype)

	def wind_fn(xyz: torch.Tensor) -> torch.Tensor:
		xyz_norm = engine.normalize_particle_coordinates(xyz, grid_bounds)
		grid = xyz_norm.view(1, 1, 1, -1, 3)

		v_start = torch.nn.functional.grid_sample(m_start, grid, align_corners=True).view(3, -1).t()
		v_end = torch.nn.functional.grid_sample(m_end, grid, align_corners=True).view(3, -1).t()
		v_interp = v_start * (1.0 - alpha) + v_end * alpha

		lat_deg = xyz[:, 1]
		meters_per_deg_lat = 110540.0
		meters_per_deg_lon = 111320.0 * torch.cos(torch.deg2rad(lat_deg)).abs().clamp(min=0.05)

		out = torch.empty_like(xyz)
		out[:, 0] = v_interp[:, 0] / meters_per_deg_lon
		out[:, 1] = v_interp[:, 1] / meters_per_deg_lat
		out[:, 2] = v_interp[:, 2]
		return out

	advected_active = engine.rk2_advect_backward(active_particles, dt_seconds=delta_s, wind_fn=wind_fn)

	diag_means = (
		float(torch.mean(m_start[0, 0] * (1 - alpha) + m_end[0, 0] * alpha).item()),
		float(torch.mean(m_start[0, 1] * (1 - alpha) + m_end[0, 1] * alpha).item()),
		float(torch.mean(m_start[0, 2] * (1 - alpha) + m_end[0, 2] * alpha).item()),
	)

	del m_start
	del m_end

	return advected_active, alpha, diag_means


def _scheme_kwargs(cfg: RunConfig) -> dict[str, object]:
	"""Constructor kwargs for the configured turbulence scheme.

	Meander config is Hanna-specific; other schemes take no kwargs and would
	reject it, so it is only forwarded for ``hanna_1982``.
	"""

	if cfg.turbulence.scheme != "hanna_1982":
		return {}
	m = cfg.turbulence.meander
	kwargs: dict[str, object] = {
		"meander_enabled": m.enabled,
		"meander_coefficient": m.coefficient,
		"meander_stencil_radius": m.stencil_radius,
	}
	if m.timescale_seconds is not None:
		kwargs["meander_timescale_seconds"] = m.timescale_seconds
	return kwargs


def _run(
	cfg: RunConfig,
	*,
	reader: MetReader | None = None,
	scheme: TurbulenceScheme | None = None,
) -> dict[str, object]:
	sim = cfg.simulation
	mem = cfg.memory
	out_grid = cfg.output_grid
	device_str = _resolve_device(sim.device)

	if scheme is None:
		scheme = get_scheme(cfg.turbulence.scheme, **_scheme_kwargs(cfg))
	if reader is None:
		# Channels = baseline (advection needs u/v/w; runtime telemetry uses blh/sp)
		# unioned with whatever the chosen turbulence scheme declares it needs.
		required_channels = tuple(
			dict.fromkeys(("u", "v", "w", "blh", "sp", *scheme.required_met_keys()))
		)
		reader = ArcoEra5ZarrReader(
			zarr_store=cfg.io.zarr_store,
			channel_names=required_channels,
			device=device_str,
		)
	engine = GPUEngine(device=device_str)
	writer = OutputWriter()
	device = torch.device(device_str)

	# M5 stage 5: expand the schedule into batches and validate met coverage over
	# the full schedule. Each batch is integrated independently and its footprint
	# slice is streamed to the output Zarr region (see the batch loop below), so
	# only one batch's footprint tensor is resident at a time.
	batches = cfg.expand_to_batches()
	all_releases: list[ConcreteRelease] = [r for b in batches for r in b.releases]
	total_releases = len(all_releases)
	all_window_ends = [
		rel.release_time + timedelta(seconds=rel.duration_seconds) for rel in all_releases
	]
	schedule_release_end = max(all_window_ends)
	schedule_sim_start = min(all_window_ends) - timedelta(seconds=sim.length_seconds)
	_validate_meteorology_time_coverage(reader, cfg, schedule_release_end, schedule_sim_start)

	LOGGER.info(
		"schedule expanded: %d release(s) across %d batch(es), max %d per batch",
		total_releases,
		len(batches),
		cfg.batch.max_releases_per_batch,
	)

	# Bookkeeping shared across batches.
	diag_rows: list[dict[str, float | int | str]] = []
	step_count = 0
	hour_windows = 0
	met_cache: OrderedDict[datetime, HourlyMetTensors] = OrderedDict()
	memory_stats = MemoryStats()
	# Endpoint particles + their release_idx accumulated across batches, concatenated
	# once at the end so we emit one parquet covering the whole schedule.
	endpoint_particles_chunks: list[torch.Tensor] = []
	endpoint_release_idx_chunks: list[torch.Tensor] = []

	outputs = _build_output_paths(cfg.io.output_uri)
	sim_length_s = float(sim.length_seconds)
	# Total particles killed for leaving met_domain across the whole run.
	run_escaped_total = 0

	# Streaming footprint output: instead of holding the full
	# (total_releases, T, Z, Y, X) tensor in memory for the whole run, each batch
	# accumulates into a gridder sized for that batch alone, writes its slice to
	# the output Zarr region, then frees the gridder. Peak footprint memory is
	# therefore one batch's worth, not the whole schedule's. The store is created
	# lazily on the first batch (we need a gridder's geometry to build coords).
	footprint_store_created = False

	for batch in batches:
		particle_batch = generate_batch_particles(batch, device=device_str)
		particles = particle_batch.particles
		# release_idx from the batch is global; remap to batch-local for the
		# per-batch gridder, then offset back to global for the region write.
		batch_release_offset = batch.releases[0].release_idx
		release_idx_local = particle_batch.release_idx - batch_release_offset
		release_time_offsets_s = particle_batch.release_time_offsets_s
		release_window_end_offsets_s = particle_batch.release_window_end_offsets_s
		batch_start_ts = particle_batch.batch_start_time.timestamp()

		gridder = FootprintGridder(
			lon_bounds=tuple(out_grid.lon_bounds),
			lat_bounds=tuple(out_grid.lat_bounds),
			z_edges_m=out_grid.z_edges_m,
			n_time_bins=out_grid.n_time_bins,
			n_y=out_grid.n_y,
			n_x=out_grid.n_x,
			n_releases=len(batch.releases),
			device=device,
			dtype=torch.float32,
		)

		if not footprint_store_created:
			# Geometry is batch-independent, so use this batch's gridder to build
			# the full-schedule coords/attrs, then create the empty store sized for
			# all releases. Subsequent batches just write their region.
			footprint_coords, footprint_attrs = _build_footprint_dataset_metadata(
				cfg, gridder, all_releases
			)
			writer.create_footprint_store(
				outputs.footprints,
				shape=(total_releases, gridder.n_t, gridder.n_z, gridder.n_y, gridder.n_x),
				coords=footprint_coords,
				attrs=footprint_attrs,
			)
			footprint_store_created = True

		# Per-batch cursor bounds: walk from the latest window-end down to the
		# earliest window-end minus sim.length_seconds. This gives every release
		# in the batch its own full backward integration window. For a
		# single-release batch this collapses to today's [release_end, sim_start]
		# bounds — preserving the stage 4 bit-equivalence guarantee.
		batch_window_ends = [
			rel.release_time + timedelta(seconds=rel.duration_seconds) for rel in batch.releases
		]
		batch_cursor_start = max(batch_window_ends)
		batch_cursor_terminus = min(batch_window_ends) - timedelta(seconds=sim_length_s)

		# Turbulence state is re-initialised per batch because the particle count
		# varies between batches (last batch may be smaller than the others).
		turbulence_state = scheme.initialize_state(
			particles.shape[0], device=device, dtype=particles.dtype
		)

		# Persistent per-batch liveness mask. A particle is killed (drop-and-count)
		# the step it leaves met_domain — it can never re-enter with valid met, so
		# stepping it further wastes compute. `active_mask` ANDs this with the
		# release-time window, so the active set shrinks as particles escape.
		alive = torch.ones(particles.shape[0], dtype=torch.bool, device=device)
		batch_escaped_total = 0

		t_cursor = batch_cursor_start
		while t_cursor > batch_cursor_terminus:
			delta_s = min(
				float(sim.dt_seconds),
				(t_cursor - batch_cursor_terminus).total_seconds(),
			)
			t_prev = t_cursor - timedelta(seconds=delta_s)

			# Per-particle active mask uses both bounds (upper: cursor reached the
			# release time; lower: cursor still inside the backward window) and the
			# liveness mask. For a single-release batch with no escapes this is
			# identical to the pre-kill release-time check (stage 3 entry in
			# CHECKPOINT.md).
			cursor_offset_s = t_cursor.timestamp() - batch_start_ts
			active_mask = (
				(release_time_offsets_s >= cursor_offset_s)
				& (release_time_offsets_s - sim_length_s <= cursor_offset_s)
				& alive
			)
			active_count = int(torch.count_nonzero(active_mask).item())

			if active_count > 0:
				active_particles = particles[active_mask]
				met_window, fetched_windows = _get_hourly_met_window(
					reader, cfg, t_cursor, met_cache
				)
				hour_windows += fetched_windows
				advected_active, t_alpha, (u_m_s, v_m_s, w_m_s) = _advect_active_particles(
					engine,
					device,
					active_particles,
					met_window,
					t_cursor,
					delta_s,
					particles.dtype,
				)
				particles[active_mask] = advected_active

				particles, turbulence_state = scheme.step(
					particles=particles,
					state=turbulence_state,
					met_window=met_window,
					t_alpha=t_alpha,
					dt_seconds=delta_s,
					active_mask=active_mask,
					engine=engine,
				)

				del active_particles
				del advected_active
			else:
				u_m_s, v_m_s, w_m_s = 0.0, 0.0, 0.0

			if active_count > 0:
				# Per-particle time-ago bin: each particle measures elapsed time
				# from *its own* release window's end (stage 3 semantics).
				elapsed_s = (release_window_end_offsets_s - cursor_offset_s).clamp_(min=0.0)
				t_idx = (elapsed_s / 3600.0).floor().to(torch.int64).clamp_(max=gridder.n_t - 1)
				gridder.accumulate(
					particles=particles[:, :3],
					active_mask=active_mask,
					weights=particles[:, 3],
					t_idx=t_idx,
					release_idx=release_idx_local,
					dt_seconds=delta_s,
				)

			# Kill particles that left met_domain this step (drop-and-count). They
			# accumulated 0 this step already (outside output_grid → dropped by the
			# gridder's valid_mask), and `alive` excludes them from all future
			# steps. Escaped particles are out-of-domain, so the cheap count is
			# `n_particles - escaped_total` (no extra device sync needed).
			escaped_this_step = 0
			if active_count > 0:
				newly_escaped = active_mask & ~_within_met_domain(particles, cfg)
				escaped_this_step = int(torch.count_nonzero(newly_escaped).item())
				if escaped_this_step:
					alive = alive & ~newly_escaped
					batch_escaped_total += escaped_this_step

			step_count += 1

			diag_rows.append(
				{
					"step": step_count,
					"batch_idx": batch.batch_idx,
					"time_hour_end": t_cursor.isoformat(),
					"mean_lon": float(torch.mean(particles[:, 0]).item()),
					"mean_lat": float(torch.mean(particles[:, 1]).item()),
					"mean_alt_agl_m": float(torch.mean(particles[:, 2]).item()),
					"u_mean_m_s": u_m_s,
					"v_mean_m_s": v_m_s,
					"w_mean_m_s": w_m_s,
					"active_particles": active_count,
					"escaped_this_step": escaped_this_step,
					"alive_particles": int(particles.shape[0]) - batch_escaped_total,
				}
			)

			if mem.gc_every_steps > 0 and step_count % mem.gc_every_steps == 0:
				gc.collect()

			if mem.log_every_steps > 0 and step_count % mem.log_every_steps == 0:
				rss_bytes, allocated, reserved = memory_stats.observe(device)
				LOGGER.info(
					"batch=%d step=%d active=%d cache_hours=%d rss=%s dev_alloc=%s dev_reserved=%s",
					batch.batch_idx,
					step_count,
					active_count,
					len(met_cache),
					_format_gib(rss_bytes),
					_format_gib(allocated),
					_format_gib(reserved),
				)

			if _memory_guard_enabled(cfg) and step_count % mem.guard_check_every_steps == 0:
				rss_bytes, allocated, reserved = memory_stats.observe(device)
				_raise_if_memory_guard_exceeded(
					writer,
					outputs.metadata,
					cfg,
					schedule_release_end,
					schedule_sim_start,
					step_count,
					hour_windows,
					outputs,
					memory_stats,
					rss_bytes,
					allocated,
					reserved,
				)

			t_cursor = t_prev

		# End of batch: stream this batch's footprint slice to disk, then drop the
		# gridder and per-batch tensors so peak footprint memory is one batch's
		# worth, not the whole schedule's. (met_cache + bookkeeping persist.)
		writer.write_footprint_region(
			outputs.footprints,
			gridder.tensor,
			release_start=batch_release_offset,
			release_stop=batch_release_offset + len(batch.releases),
		)
		endpoint_particles_chunks.append(particles.detach().cpu().clone())
		endpoint_release_idx_chunks.append(particle_batch.release_idx.detach().cpu().clone())
		del gridder
		del particles
		del release_idx_local
		del release_time_offsets_s
		del release_window_end_offsets_s
		del turbulence_state
		del particle_batch
		del alive
		run_escaped_total += batch_escaped_total

	for cached in met_cache.values():
		del cached
	met_cache.clear()

	endpoint_particles_all = torch.cat(endpoint_particles_chunks, dim=0)
	endpoint_release_idx_all = torch.cat(endpoint_release_idx_chunks, dim=0)
	writer.write_particles_parquet(
		outputs.endpoint_particles,
		endpoint_particles_all,
		release_idx=endpoint_release_idx_all,
	)
	writer.write_trajectory_parquet(
		outputs.trajectory_diagnostics, step_seconds=sim.dt_seconds, rows=diag_rows
	)

	metadata = {
		"config": _config_metadata(cfg, schedule_release_end, schedule_sim_start),
		"runtime": _runtime_metadata(cfg, step_count, hour_windows, memory_stats, status="completed"),
		"schedule": {
			"n_batches": len(batches),
			"n_releases": total_releases,
			"max_releases_per_batch": cfg.batch.max_releases_per_batch,
			"release_times": [rel.release_time.isoformat() for rel in all_releases],
			"escaped_met_domain_total": run_escaped_total,
		},
		"outputs": {
			"endpoint_particles": outputs.endpoint_particles,
			"trajectory_diagnostics": outputs.trajectory_diagnostics,
			"footprints": outputs.footprints,
			"metadata": outputs.metadata,
		},
		"footprint_units": FOOTPRINT_UNITS_DOC,
	}
	writer.write_metadata_json(outputs.metadata, metadata)

	return metadata


def main(argv: Sequence[str] | None = None) -> None:
	logging.basicConfig(
		level=logging.INFO,
		format="%(asctime)s %(levelname)s %(name)s: %(message)s",
	)

	parser = _build_arg_parser()
	args = parser.parse_args(argv)
	cfg = RunConfig.from_yaml(args.config).with_overrides(
		device=args.device,
		output_uri=args.output_uri,
		start_time=args.start_time,
	)

	if cfg.turbulence.scheme not in list_schemes():
		parser.exit(2, f"Error: unknown turbulence scheme {cfg.turbulence.scheme!r}. Registered: {', '.join(list_schemes())}\n")

	try:
		metadata = _run(cfg)
	except PreflightValidationError as exc:
		parser.exit(2, f"Error: {exc}\n")
	print("LPDM run complete")
	print(metadata["outputs"])


if __name__ == "__main__":
	main()
