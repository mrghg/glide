"""Top-level LPDM orchestration entry point.

This module provides a minimal end-to-end backward trajectory run mode suitable
for early cloud testing (including Vertex AI user-managed notebooks).
"""

from __future__ import annotations

import argparse
import gc
import logging
import os
import resource
import sys
from collections import OrderedDict
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from typing import Sequence

import torch

from lpdm.footprint_gridder import FootprintGridder
from lpdm.gpu_engine import CoordinateBounds, GPUEngine, GridInterpolationBounds
from lpdm.met_reader import ArcoEra5ZarrReader, BoundingBoxRequest, HourlyMetTensors, SpatialBounds, TimeBounds
from lpdm.output_writer import OutputWriter
from lpdm.release_generator import PointRelease
from lpdm.runtime import DEVICE


LOGGER = logging.getLogger(__name__)


def _env_str(name: str, default: str | None = None) -> str | None:
	value = os.environ.get(name)
	if value is None or value == "":
		return default
	return value


def _env_int(name: str, default: int) -> int:
	value = os.environ.get(name)
	return default if value is None or value == "" else int(value)


def _env_optional_int(name: str) -> int | None:
	value = os.environ.get(name)
	if value is None or value == "":
		return None
	return int(value)


def _env_float(name: str, default: float) -> float:
	value = os.environ.get(name)
	return default if value is None or value == "" else float(value)


def _env_optional_float(name: str) -> float | None:
	value = os.environ.get(name)
	if value is None or value == "":
		return None
	return float(value)


def _current_rss_bytes() -> int | None:
	"""Return process memory bytes when available.

	On Linux, this reports current RSS from /proc/self/status. On other
	platforms, it falls back to getrusage() peak RSS.
	"""

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
	"""Return allocated/reserved bytes for current torch device when supported."""

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


def _parse_datetime_utc(value: str) -> datetime:
	"""Parse ISO datetime string and normalize to UTC."""

	dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
	if dt.tzinfo is None:
		dt = dt.replace(tzinfo=timezone.utc)
	return dt.astimezone(timezone.utc)


def _hour_floor(dt: datetime) -> datetime:
	return dt.astimezone(timezone.utc).replace(minute=0, second=0, microsecond=0)


def _footprint_time_bin_index(release_end: datetime, t_cursor: datetime, n_time_bins: int) -> int:
	"""Map the current backward integration cursor to a 0-based time_ago bin."""

	if n_time_bins <= 0:
		raise ValueError("n_time_bins must be > 0")

	elapsed_seconds = max(0.0, release_end.timestamp() - t_cursor.timestamp())
	return min(n_time_bins - 1, int(elapsed_seconds / 3600.0))


@dataclass(frozen=True)
class RunConfig:
	zarr_store: str
	start_time: datetime
	release_duration_seconds: int
	simulation_length_seconds: int
	release_seed: int | None
	n_particles: int
	release_lon: float
	release_lat: float
	release_alt_agl_m: float
	dt_seconds: int
	bbox_pad_lon_deg: float
	bbox_pad_lat_deg: float
	bbox_pad_alt_m: float
	output_uri: str
	device: str
	met_cache_max_hours: int
	memory_log_every_steps: int
	gc_every_steps: int
	memory_guard_max_rss_gib: float | None
	memory_guard_max_device_allocated_gib: float | None
	memory_guard_max_device_reserved_gib: float | None
	memory_guard_check_every_steps: int


def _build_arg_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Run a minimal backward LPDM trajectory simulation")

	parser.add_argument("--zarr-store", default=_env_str("LPDM_ZARR_STORE"), help="ERA5 ARCO Zarr store URI")
	parser.add_argument(
		"--start-time",
		default=_env_str("LPDM_START_TIME"),
		help="UTC ISO timestamp for start of particle release window (e.g. 2024-01-01T00:00:00Z)",
	)
	parser.add_argument(
		"--release-duration-seconds",
		type=int,
		default=_env_int("LPDM_RELEASE_DURATION_SECONDS", 3600),
		help="Total duration of the particle release window in seconds",
	)
	parser.add_argument(
		"--simulation-length-seconds",
		type=int,
		default=_env_int("LPDM_SIMULATION_LENGTH_SECONDS", 10800),
		help="Total backward tracking length in seconds from end of release window",
	)
	parser.add_argument(
		"--release-seed",
		type=int,
		default=_env_optional_int("LPDM_RELEASE_SEED"),
		help="Optional RNG seed for deterministic temporal release sampling",
	)

	parser.add_argument("--n-particles", type=int, default=_env_int("LPDM_N_PARTICLES", 1024))
	parser.add_argument("--release-lon", type=float, default=_env_float("LPDM_RELEASE_LON", 0.0))
	parser.add_argument("--release-lat", type=float, default=_env_float("LPDM_RELEASE_LAT", 0.0))
	parser.add_argument("--release-alt-agl-m", type=float, default=_env_float("LPDM_RELEASE_ALT_AGL_M", 500.0))

	parser.add_argument("--dt-seconds", type=int, default=_env_int("LPDM_DT_SECONDS", 300))
	parser.add_argument("--bbox-pad-lon-deg", type=float, default=_env_float("LPDM_BBOX_PAD_LON_DEG", 2.0))
	parser.add_argument("--bbox-pad-lat-deg", type=float, default=_env_float("LPDM_BBOX_PAD_LAT_DEG", 2.0))
	parser.add_argument("--bbox-pad-alt-m", type=float, default=_env_float("LPDM_BBOX_PAD_ALT_M", 3000.0))

	default_output_uri = _env_str("LPDM_OUTPUT_URI", f"outputs/run-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}")
	parser.add_argument("--output-uri", default=default_output_uri)
	parser.add_argument("--device", default=_env_str("LPDM_DEVICE", str(DEVICE)))
	parser.add_argument(
		"--met-cache-max-hours",
		type=int,
		default=_env_int("LPDM_MET_CACHE_MAX_HOURS", 2),
		help="Maximum hourly met tensors kept in memory; 0 disables cache",
	)
	parser.add_argument(
		"--memory-log-every-steps",
		type=int,
		default=_env_int("LPDM_MEMORY_LOG_EVERY_STEPS", 10),
		help="Log memory usage every N integration steps; 0 disables logs",
	)
	parser.add_argument(
		"--gc-every-steps",
		type=int,
		default=_env_int("LPDM_GC_EVERY_STEPS", 50),
		help="Run gc.collect() every N steps to reduce delayed retention; 0 disables",
	)
	parser.add_argument(
		"--memory-guard-max-rss-gib",
		type=float,
		default=_env_optional_float("LPDM_MEMORY_GUARD_MAX_RSS_GIB"),
		help="Abort early if process RSS exceeds this GiB threshold; unset disables guard",
	)
	parser.add_argument(
		"--memory-guard-max-device-allocated-gib",
		type=float,
		default=_env_optional_float("LPDM_MEMORY_GUARD_MAX_DEVICE_ALLOCATED_GIB"),
		help="Abort early if device allocated memory exceeds this GiB threshold; unset disables guard",
	)
	parser.add_argument(
		"--memory-guard-max-device-reserved-gib",
		type=float,
		default=_env_optional_float("LPDM_MEMORY_GUARD_MAX_DEVICE_RESERVED_GIB"),
		help="Abort early if device reserved/driver memory exceeds this GiB threshold; unset disables guard",
	)
	parser.add_argument(
		"--memory-guard-check-every-steps",
		type=int,
		default=_env_int("LPDM_MEMORY_GUARD_CHECK_EVERY_STEPS", 1),
		help="Check memory guard thresholds every N integration steps",
	)
	return parser


def _build_config(args: argparse.Namespace) -> RunConfig:
	if not args.zarr_store:
		raise ValueError("Missing --zarr-store (or LPDM_ZARR_STORE)")
	if not args.start_time:
		raise ValueError("Missing --start-time (or LPDM_START_TIME)")

	start_time = _parse_datetime_utc(args.start_time)
	release_duration_seconds = int(args.release_duration_seconds)
	simulation_length_seconds = int(args.simulation_length_seconds)
	release_seed = None if args.release_seed is None else int(args.release_seed)

	if release_duration_seconds <= 0:
		raise ValueError("release-duration-seconds must be > 0")
	if simulation_length_seconds <= 0:
		raise ValueError("simulation-length-seconds must be > 0")
	if simulation_length_seconds <= release_duration_seconds:
		raise ValueError("simulation-length-seconds must be > release-duration-seconds")
	if release_seed is not None and release_seed < 0:
		raise ValueError("release-seed must be >= 0")
	if args.dt_seconds <= 0:
		raise ValueError("dt-seconds must be > 0")
	if args.met_cache_max_hours < 0:
		raise ValueError("met-cache-max-hours must be >= 0")
	if args.memory_log_every_steps < 0:
		raise ValueError("memory-log-every-steps must be >= 0")
	if args.gc_every_steps < 0:
		raise ValueError("gc-every-steps must be >= 0")
	if args.memory_guard_max_rss_gib is not None and args.memory_guard_max_rss_gib <= 0:
		raise ValueError("memory-guard-max-rss-gib must be > 0 when set")
	if (
		args.memory_guard_max_device_allocated_gib is not None
		and args.memory_guard_max_device_allocated_gib <= 0
	):
		raise ValueError("memory-guard-max-device-allocated-gib must be > 0 when set")
	if (
		args.memory_guard_max_device_reserved_gib is not None
		and args.memory_guard_max_device_reserved_gib <= 0
	):
		raise ValueError("memory-guard-max-device-reserved-gib must be > 0 when set")
	if args.memory_guard_check_every_steps <= 0:
		raise ValueError("memory-guard-check-every-steps must be > 0")

	return RunConfig(
		zarr_store=args.zarr_store,
		start_time=start_time,
		release_duration_seconds=release_duration_seconds,
		simulation_length_seconds=simulation_length_seconds,
		release_seed=release_seed,
		n_particles=args.n_particles,
		release_lon=args.release_lon,
		release_lat=args.release_lat,
		release_alt_agl_m=args.release_alt_agl_m,
		dt_seconds=args.dt_seconds,
		bbox_pad_lon_deg=args.bbox_pad_lon_deg,
		bbox_pad_lat_deg=args.bbox_pad_lat_deg,
		bbox_pad_alt_m=args.bbox_pad_alt_m,
		output_uri=args.output_uri,
		device=args.device,
		met_cache_max_hours=args.met_cache_max_hours,
		memory_log_every_steps=args.memory_log_every_steps,
		gc_every_steps=args.gc_every_steps,
		memory_guard_max_rss_gib=args.memory_guard_max_rss_gib,
		memory_guard_max_device_allocated_gib=args.memory_guard_max_device_allocated_gib,
		memory_guard_max_device_reserved_gib=args.memory_guard_max_device_reserved_gib,
		memory_guard_check_every_steps=args.memory_guard_check_every_steps,
	)


def _build_bbox(particles: torch.Tensor, cfg: RunConfig) -> SpatialBounds:
	lon = particles[:, 0]
	lat = particles[:, 1]
	alt = particles[:, 2]

	return SpatialBounds(
		lon_min=float(torch.min(lon).item()) - cfg.bbox_pad_lon_deg,
		lon_max=float(torch.max(lon).item()) + cfg.bbox_pad_lon_deg,
		lat_min=float(torch.min(lat).item()) - cfg.bbox_pad_lat_deg,
		lat_max=float(torch.max(lat).item()) + cfg.bbox_pad_lat_deg,
		z_min=max(0.0, float(torch.min(alt).item()) - cfg.bbox_pad_alt_m),
		z_max=float(torch.max(alt).item()) + cfg.bbox_pad_alt_m,
	)


def _mean_wind(met: torch.Tensor) -> tuple[float, float, float]:
	"""Return domain-mean wind components from [channels, z, y, x] tensor."""

	u = float(torch.mean(met[0]).item())
	v = float(torch.mean(met[1]).item())
	w = float(torch.mean(met[2]).item())
	return u, v, w


def _run(cfg: RunConfig) -> dict[str, object]:
	reader = ArcoEra5ZarrReader(zarr_store=cfg.zarr_store, device=cfg.device)
	engine = GPUEngine(device=cfg.device)
	writer = OutputWriter()
	device = torch.device(cfg.device)

	particles = PointRelease(
		n_particles=cfg.n_particles,
		lon=cfg.release_lon,
		lat=cfg.release_lat,
		alt=cfg.release_alt_agl_m,
		device=cfg.device,
	).generate()

	diag_rows: list[dict[str, float | int | str]] = []
	release_start = cfg.start_time
	release_end = release_start + timedelta(seconds=cfg.release_duration_seconds)
	sim_start = release_end - timedelta(seconds=cfg.simulation_length_seconds)

	# Uniformly distribute per-particle release times across [release_start, release_end].
	release_start_ts = release_start.timestamp()
	release_end_ts = release_end.timestamp()
	if cfg.release_seed is not None:
		release_rng = torch.Generator(device="cpu")
		release_rng.manual_seed(cfg.release_seed)
		release_times_ts = torch.empty(cfg.n_particles, device="cpu", dtype=torch.float32).uniform_(
			release_start_ts,
			release_end_ts,
			generator=release_rng,
		)
		release_times_ts = release_times_ts.to(device=particles.device)
	else:
		release_times_ts = torch.empty(cfg.n_particles, device=particles.device, dtype=torch.float32).uniform_(
			release_start_ts,
			release_end_ts,
		)

	# 1-hour temporal bins, 0.25 deg spatial bins
	n_hours = int(cfg.simulation_length_seconds / 3600)
	n_y = int(2.0 * cfg.bbox_pad_lat_deg / 0.25)
	n_x = int(2.0 * cfg.bbox_pad_lon_deg / 0.25)
	gridder = FootprintGridder(
		lon_bounds=(cfg.release_lon - cfg.bbox_pad_lon_deg, cfg.release_lon + cfg.bbox_pad_lon_deg),
		lat_bounds=(cfg.release_lat - cfg.bbox_pad_lat_deg, cfg.release_lat + cfg.bbox_pad_lat_deg),
		z_bounds=(0.0, 5000.0),
		n_time_bins=max(1, n_hours),
		n_y=max(1, n_y),
		n_x=max(1, n_x),
		n_z_bins=5,
		device=device,
		dtype=particles.dtype,
	)

	step_count = 0
	hour_windows = 0
	met_cache: OrderedDict[datetime, HourlyMetTensors] = OrderedDict()
	
	# Keep vertical turbulence state alongside particles [N,]
	w_prime = engine.initialize_turbulence_velocity(cfg.n_particles)
	
	peak_rss_bytes = 0
	peak_device_allocated = 0
	peak_device_reserved = 0
	memory_guard_triggered = False

	endpoint_path = f"{cfg.output_uri.rstrip('/')}/endpoint_particles.parquet"
	trajectory_path = f"{cfg.output_uri.rstrip('/')}/trajectory_diagnostics.parquet"
	metadata_path = f"{cfg.output_uri.rstrip('/')}/run_metadata.json"

	t_cursor = release_end
	while t_cursor > sim_start:
		delta_s = min(float(cfg.dt_seconds), (t_cursor - sim_start).total_seconds())
		t_prev = t_cursor - timedelta(seconds=delta_s)

		active_mask = release_times_ts >= t_cursor.timestamp()
		active_count = int(torch.count_nonzero(active_mask).item())

		if active_count > 0:
			active_particles = particles[active_mask]
			hour_key = _hour_floor(t_cursor)

			if cfg.met_cache_max_hours > 0 and hour_key in met_cache:
				met_cache.move_to_end(hour_key)
				met_window = met_cache[hour_key]
			else:
				bbox = _build_bbox(active_particles, cfg)
				met_window = reader.fetch_hourly_window(
					BoundingBoxRequest(
						spatial=bbox,
						time=TimeBounds(start=hour_key, end=hour_key + timedelta(hours=1)),
					)
				)
				hour_windows += 1

				if cfg.met_cache_max_hours > 0:
					met_cache[hour_key] = met_window
					while len(met_cache) > cfg.met_cache_max_hours:
						_, evicted = met_cache.popitem(last=False)
						del evicted

			# 1. Prepare interpolation state and constants
			t_start_s = met_window.metadata.time_start.timestamp()
			t_end_s = met_window.metadata.time_end.timestamp()
			
			# We use the midpoint of the timestep for interpolation weight
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

			m_start = met_window.hour_start[:3].unsqueeze(0).to(device=device, dtype=particles.dtype)
			m_end = met_window.hour_end[:3].unsqueeze(0).to(device=device, dtype=particles.dtype)

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
			
			# 2. Apply random vertical turbulence dispersion 
			w_prime_active = engine.update_langevin_velocity(
				w_prime[active_mask],
				t_lagrangian=300.0,
				sigma_w2=1.0,
				dt_seconds=delta_s,
			)
			advected_active = engine.apply_vertical_turbulence(
				advected_active,
				w_prime_active,
				dt_seconds=delta_s,
				backward=True,
			)
			w_prime[active_mask] = w_prime_active

			advected_active = engine.reflect_surface(advected_active, z_surface=0.0)
			particles[active_mask] = advected_active
			
			# Domain-mean values strictly for the diagnostic log tracking 
			u_m_s = float(torch.mean(m_start[0, 0] * (1-alpha) + m_end[0, 0] * alpha).item())
			v_m_s = float(torch.mean(m_start[0, 1] * (1-alpha) + m_end[0, 1] * alpha).item())
			w_m_s = float(torch.mean(m_start[0, 2] * (1-alpha) + m_end[0, 2] * alpha).item())
			
			del active_particles
			del advected_active
			del m_start
			del m_end
		else:
			u_m_s, v_m_s, w_m_s = 0.0, 0.0, 0.0

		# Accumulate footprint
		if active_count > 0:
			t_idx = _footprint_time_bin_index(release_end, t_cursor, gridder.n_t)
			gridder.accumulate(
				particles=particles[:, :3],
				active_mask=active_mask,
				weights=particles[:, 3],
				t_idx=t_idx,
				dt_seconds=delta_s,
			)

		step_count += 1

		diag_rows.append(
			{
				"step": step_count,
				"time_hour_end": t_cursor.isoformat(),
				"mean_lon": float(torch.mean(particles[:, 0]).item()),
				"mean_lat": float(torch.mean(particles[:, 1]).item()),
				"mean_alt_agl_m": float(torch.mean(particles[:, 2]).item()),
				"u_mean_m_s": u_m_s,
				"v_mean_m_s": v_m_s,
				"w_mean_m_s": w_m_s,
				"active_particles": active_count,
			}
		)

		if cfg.gc_every_steps > 0 and step_count % cfg.gc_every_steps == 0:
			gc.collect()

		if cfg.memory_log_every_steps > 0 and step_count % cfg.memory_log_every_steps == 0:
			rss_bytes = _current_rss_bytes()
			if rss_bytes is not None:
				peak_rss_bytes = max(peak_rss_bytes, rss_bytes)
			allocated, reserved = _device_memory_bytes(device)
			if allocated is not None:
				peak_device_allocated = max(peak_device_allocated, allocated)
			if reserved is not None:
				peak_device_reserved = max(peak_device_reserved, reserved)
			LOGGER.info(
				"step=%d active=%d cache_hours=%d rss=%s dev_alloc=%s dev_reserved=%s",
				step_count,
				active_count,
				len(met_cache),
				_format_gib(rss_bytes),
				_format_gib(allocated),
				_format_gib(reserved),
			)

		if (
			(
				cfg.memory_guard_max_rss_gib is not None
				or cfg.memory_guard_max_device_allocated_gib is not None
				or cfg.memory_guard_max_device_reserved_gib is not None
			)
			and step_count % cfg.memory_guard_check_every_steps == 0
		):
			rss_bytes = _current_rss_bytes()
			allocated, reserved = _device_memory_bytes(device)
			if rss_bytes is not None:
				peak_rss_bytes = max(peak_rss_bytes, rss_bytes)
			if allocated is not None:
				peak_device_allocated = max(peak_device_allocated, allocated)
			if reserved is not None:
				peak_device_reserved = max(peak_device_reserved, reserved)

			if cfg.memory_guard_max_rss_gib is not None and rss_bytes is not None:
				guard_limit_bytes = int(cfg.memory_guard_max_rss_gib * (1024 ** 3))
				if rss_bytes > guard_limit_bytes:
					memory_guard_triggered = True
					guard_metadata = {
						"status": "aborted_memory_guard",
						"reason": "RSS exceeded memory guard threshold",
						"config": {
							**asdict(cfg),
							"start_time": cfg.start_time.isoformat(),
							"release_end_time": release_end.isoformat(),
							"simulation_start_time": sim_start.isoformat(),
						},
						"runtime": {
							"device": str(cfg.device),
							"steps": step_count,
							"hour_windows": hour_windows,
							"met_cache_max_hours": cfg.met_cache_max_hours,
							"peak_rss_bytes": peak_rss_bytes,
							"peak_device_allocated_bytes": peak_device_allocated,
							"peak_device_reserved_bytes": peak_device_reserved,
							"guard_limit_rss_bytes": guard_limit_bytes,
							"guard_observed_rss_bytes": rss_bytes,
						},
						"outputs": {
							"endpoint_particles": endpoint_path,
							"trajectory_diagnostics": trajectory_path,
							"metadata": metadata_path,
						},
					}
					writer.write_metadata_json(metadata_path, guard_metadata)
					raise MemoryError(
						"Memory safety guard triggered: RSS reached "
						f"{_format_gib(rss_bytes)} which is above "
						f"{cfg.memory_guard_max_rss_gib:.3f} GiB"
					)

			if cfg.memory_guard_max_device_allocated_gib is not None and allocated is not None:
				guard_limit_allocated_bytes = int(cfg.memory_guard_max_device_allocated_gib * (1024 ** 3))
				if allocated > guard_limit_allocated_bytes:
					memory_guard_triggered = True
					guard_metadata = {
						"status": "aborted_memory_guard",
						"reason": "Device allocated memory exceeded memory guard threshold",
						"config": {
							**asdict(cfg),
							"start_time": cfg.start_time.isoformat(),
							"release_end_time": release_end.isoformat(),
							"simulation_start_time": sim_start.isoformat(),
						},
						"runtime": {
							"device": str(cfg.device),
							"steps": step_count,
							"hour_windows": hour_windows,
							"met_cache_max_hours": cfg.met_cache_max_hours,
							"peak_rss_bytes": peak_rss_bytes,
							"peak_device_allocated_bytes": peak_device_allocated,
							"peak_device_reserved_bytes": peak_device_reserved,
							"guard_limit_device_allocated_bytes": guard_limit_allocated_bytes,
							"guard_observed_device_allocated_bytes": allocated,
						},
						"outputs": {
							"endpoint_particles": endpoint_path,
							"trajectory_diagnostics": trajectory_path,
							"metadata": metadata_path,
						},
					}
					writer.write_metadata_json(metadata_path, guard_metadata)
					raise MemoryError(
						"Memory safety guard triggered: device allocated memory reached "
						f"{_format_gib(allocated)} which is above "
						f"{cfg.memory_guard_max_device_allocated_gib:.3f} GiB"
					)

			if cfg.memory_guard_max_device_reserved_gib is not None and reserved is not None:
				guard_limit_reserved_bytes = int(cfg.memory_guard_max_device_reserved_gib * (1024 ** 3))
				if reserved > guard_limit_reserved_bytes:
					memory_guard_triggered = True
					guard_metadata = {
						"status": "aborted_memory_guard",
						"reason": "Device reserved memory exceeded memory guard threshold",
						"config": {
							**asdict(cfg),
							"start_time": cfg.start_time.isoformat(),
							"release_end_time": release_end.isoformat(),
							"simulation_start_time": sim_start.isoformat(),
						},
						"runtime": {
							"device": str(cfg.device),
							"steps": step_count,
							"hour_windows": hour_windows,
							"met_cache_max_hours": cfg.met_cache_max_hours,
							"peak_rss_bytes": peak_rss_bytes,
							"peak_device_allocated_bytes": peak_device_allocated,
							"peak_device_reserved_bytes": peak_device_reserved,
							"guard_limit_device_reserved_bytes": guard_limit_reserved_bytes,
							"guard_observed_device_reserved_bytes": reserved,
						},
						"outputs": {
							"endpoint_particles": endpoint_path,
							"trajectory_diagnostics": trajectory_path,
							"metadata": metadata_path,
						},
					}
					writer.write_metadata_json(metadata_path, guard_metadata)
					raise MemoryError(
						"Memory safety guard triggered: device reserved memory reached "
						f"{_format_gib(reserved)} which is above "
						f"{cfg.memory_guard_max_device_reserved_gib:.3f} GiB"
					)

		t_cursor = t_prev

	for cached in met_cache.values():
		del cached
	met_cache.clear()

	writer.write_particles_parquet(endpoint_path, particles)
	writer.write_trajectory_parquet(trajectory_path, step_seconds=cfg.dt_seconds, rows=diag_rows)
	writer.write_footprint_zarr(endpoint_path.replace("endpoint_particles.parquet", "footprints.zarr"), gridder.tensor)

	metadata = {
		"config": {
			**asdict(cfg),
			"start_time": cfg.start_time.isoformat(),
			"release_end_time": release_end.isoformat(),
			"simulation_start_time": sim_start.isoformat(),
		},
		"runtime": {
			"device": str(cfg.device),
			"status": "completed" if not memory_guard_triggered else "aborted_memory_guard",
			"steps": step_count,
			"hour_windows": hour_windows,
			"met_cache_max_hours": cfg.met_cache_max_hours,
			"peak_rss_bytes": peak_rss_bytes,
			"peak_device_allocated_bytes": peak_device_allocated,
			"peak_device_reserved_bytes": peak_device_reserved,
		},
		                "outputs": {
                        "endpoint_particles": endpoint_path,
                        "trajectory_diagnostics": trajectory_path,
                        "footprints": endpoint_path.replace('endpoint_particles.parquet', 'footprints.zarr'),
                },
	}
	writer.write_metadata_json(metadata_path, metadata)

	return metadata


def main(argv: Sequence[str] | None = None) -> None:
	logging.basicConfig(
		level=logging.INFO,
		format="%(asctime)s %(levelname)s %(name)s: %(message)s",
	)

	parser = _build_arg_parser()
	args = parser.parse_args(argv)
	cfg = _build_config(args)

	metadata = _run(cfg)
	print("LPDM run complete")
	print(metadata["outputs"])


if __name__ == "__main__":
	main()
