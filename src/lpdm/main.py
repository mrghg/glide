"""Top-level LPDM orchestration entry point.

This module provides a minimal end-to-end backward trajectory run mode suitable
for early cloud testing (including Vertex AI user-managed notebooks).
"""

from __future__ import annotations

import argparse
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from typing import Sequence

import torch

from lpdm.gpu_engine import GPUEngine
from lpdm.met_reader import ArcoEra5ZarrReader, BoundingBoxRequest, SpatialBounds, TimeBounds
from lpdm.output_writer import OutputWriter
from lpdm.release_generator import PointRelease
from lpdm.runtime import DEVICE


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


def _parse_datetime_utc(value: str) -> datetime:
	"""Parse ISO datetime string and normalize to UTC."""

	dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
	if dt.tzinfo is None:
		dt = dt.replace(tzinfo=timezone.utc)
	return dt.astimezone(timezone.utc)


def _hour_floor(dt: datetime) -> datetime:
	return dt.astimezone(timezone.utc).replace(minute=0, second=0, microsecond=0)


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

	step_count = 0
	hour_windows = 0
	met_cache: dict[datetime, torch.Tensor] = {}

	t_cursor = release_end
	while t_cursor > sim_start:
		delta_s = min(float(cfg.dt_seconds), (t_cursor - sim_start).total_seconds())
		t_prev = t_cursor - timedelta(seconds=delta_s)

		active_mask = release_times_ts >= t_cursor.timestamp()
		active_count = int(torch.count_nonzero(active_mask).item())

		if active_count > 0:
			active_particles = particles[active_mask]
			hour_key = _hour_floor(t_cursor)

			if hour_key not in met_cache:
				bbox = _build_bbox(active_particles, cfg)
				met_window = reader.fetch_hourly_window(
					BoundingBoxRequest(
						spatial=bbox,
						time=TimeBounds(start=hour_key, end=hour_key + timedelta(hours=1)),
					)
				)
				met_cache[hour_key] = 0.5 * (met_window.hour_start + met_window.hour_end)
				hour_windows += 1

			met_avg = met_cache[hour_key]
			u_m_s, v_m_s, w_m_s = _mean_wind(met_avg)

			# Convert horizontal m/s to degrees/s using local spherical approximations.
			lat_mean_deg = float(torch.mean(active_particles[:, 1]).item())
			meters_per_deg_lat = 110540.0
			meters_per_deg_lon = max(
				1e-6,
				111320.0 * max(0.05, abs(torch.cos(torch.deg2rad(torch.tensor(lat_mean_deg))).item())),
			)
			u_deg_s = u_m_s / meters_per_deg_lon
			v_deg_s = v_m_s / meters_per_deg_lat

			def wind_fn(xyz: torch.Tensor) -> torch.Tensor:
				out = torch.empty_like(xyz)
				out[:, 0] = u_deg_s
				out[:, 1] = v_deg_s
				out[:, 2] = w_m_s
				return out

			advected_active = engine.rk2_advect_backward(active_particles, dt_seconds=delta_s, wind_fn=wind_fn)
			advected_active = engine.reflect_surface(advected_active, z_surface=0.0)
			particles[active_mask] = advected_active
		else:
			u_m_s, v_m_s, w_m_s = 0.0, 0.0, 0.0

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

		t_cursor = t_prev

	endpoint_path = f"{cfg.output_uri.rstrip('/')}/endpoint_particles.parquet"
	trajectory_path = f"{cfg.output_uri.rstrip('/')}/trajectory_diagnostics.parquet"
	metadata_path = f"{cfg.output_uri.rstrip('/')}/run_metadata.json"

	writer.write_particles_parquet(endpoint_path, particles)
	writer.write_trajectory_parquet(trajectory_path, step_seconds=cfg.dt_seconds, rows=diag_rows)

	metadata = {
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
		},
		"outputs": {
			"endpoint_particles": endpoint_path,
			"trajectory_diagnostics": trajectory_path,
		},
	}
	writer.write_metadata_json(metadata_path, metadata)

	return metadata


def main(argv: Sequence[str] | None = None) -> None:
	parser = _build_arg_parser()
	args = parser.parse_args(argv)
	cfg = _build_config(args)

	metadata = _run(cfg)
	print("LPDM run complete")
	print(metadata["outputs"])


if __name__ == "__main__":
	main()
