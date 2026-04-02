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
	end_time: datetime
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
		help="UTC ISO timestamp for oldest trajectory time (e.g. 2024-01-01T00:00:00Z)",
	)
	parser.add_argument(
		"--end-time",
		default=_env_str("LPDM_END_TIME"),
		help="UTC ISO timestamp for release time (e.g. 2024-01-01T06:00:00Z)",
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
	if not args.start_time or not args.end_time:
		raise ValueError("Missing --start-time/--end-time (or LPDM_START_TIME/LPDM_END_TIME)")

	start_time = _parse_datetime_utc(args.start_time)
	end_time = _parse_datetime_utc(args.end_time)

	if end_time <= start_time:
		raise ValueError("end-time must be later than start-time")
	if args.dt_seconds <= 0:
		raise ValueError("dt-seconds must be > 0")

	return RunConfig(
		zarr_store=args.zarr_store,
		start_time=start_time,
		end_time=end_time,
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

	t_cursor = _hour_floor(cfg.end_time)
	t_stop = _hour_floor(cfg.start_time)

	step_count = 0
	hour_windows = 0

	while t_cursor > t_stop:
		t_prev = t_cursor - timedelta(hours=1)
		bbox = _build_bbox(particles, cfg)

		met_window = reader.fetch_hourly_window(
			BoundingBoxRequest(
				spatial=bbox,
				time=TimeBounds(start=t_prev, end=t_cursor),
			)
		)

		hour_windows += 1
		met_avg = 0.5 * (met_window.hour_start + met_window.hour_end)
		u_m_s, v_m_s, w_m_s = _mean_wind(met_avg)

		# Convert horizontal m/s to degrees/s using local spherical approximations.
		lat_mean_deg = float(torch.mean(particles[:, 1]).item())
		meters_per_deg_lat = 110540.0
		meters_per_deg_lon = max(1e-6, 111320.0 * max(0.05, abs(torch.cos(torch.deg2rad(torch.tensor(lat_mean_deg))).item())))
		u_deg_s = u_m_s / meters_per_deg_lon
		v_deg_s = v_m_s / meters_per_deg_lat

		def wind_fn(xyz: torch.Tensor) -> torch.Tensor:
			out = torch.empty_like(xyz)
			out[:, 0] = u_deg_s
			out[:, 1] = v_deg_s
			out[:, 2] = w_m_s
			return out

		substeps = max(1, int(round(3600 / cfg.dt_seconds)))
		for _ in range(substeps):
			particles = engine.rk2_advect_backward(particles, dt_seconds=float(cfg.dt_seconds), wind_fn=wind_fn)
			particles = engine.reflect_surface(particles, z_surface=0.0)
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
			"end_time": cfg.end_time.isoformat(),
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
