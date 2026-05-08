"""End-to-end runtime tests using a synthetic analytic meteorology reader.

These tests exercise the full main runtime loop (advection, met fetching, footprint
accumulation, output writing) without depending on remote ERA5 data.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import pytest
import torch
import xarray as xr

from lpdm.main import PreflightValidationError, RunConfig, _run
from lpdm.met_reader import (
    BoundingBoxRequest,
    HourlyMetTensors,
    MetFieldMetadata,
    MetReader,
)


@dataclass
class AnalyticMetReader(MetReader):
    """Synthetic met reader returning spatially-uniform analytic wind fields.

    Each fetch evaluates the user-provided wind callable at hour-start and hour-end
    and broadcasts the result across a small regular grid. Spatial uniformity keeps
    test geometry simple while still exercising temporal interpolation.
    """

    coverage_start: datetime
    coverage_end: datetime
    wind_fn: Callable[[datetime], tuple[float, float, float]]
    lon_bounds: tuple[float, float] = (-10.0, 10.0)
    lat_bounds: tuple[float, float] = (-10.0, 10.0)
    z_bounds_agl_m: tuple[float, float] = (0.0, 10000.0)
    n_lon: int = 8
    n_lat: int = 8
    n_lev: int = 4
    blh_m: float = 1500.0
    sp_pa: float = 101325.0
    device: torch.device | str = "cpu"
    dtype: torch.dtype = torch.float32

    def get_time_coverage(self) -> tuple[datetime, datetime]:
        return self.coverage_start, self.coverage_end

    def fetch_hourly_window(self, request: BoundingBoxRequest) -> HourlyMetTensors:
        t0 = request.time.start
        t1 = request.time.end
        u0, v0, w0 = self.wind_fn(t0)
        u1, v1, w1 = self.wind_fn(t1)

        shape = (self.n_lev, self.n_lat, self.n_lon)

        def _build(u: float, v: float, w: float) -> torch.Tensor:
            return torch.stack(
                [
                    torch.full(shape, float(u), dtype=self.dtype, device=self.device),
                    torch.full(shape, float(v), dtype=self.dtype, device=self.device),
                    torch.full(shape, float(w), dtype=self.dtype, device=self.device),
                    torch.full(shape, self.blh_m, dtype=self.dtype, device=self.device),
                    torch.full(shape, self.sp_pa, dtype=self.dtype, device=self.device),
                ],
                dim=0,
            )

        metadata = MetFieldMetadata(
            lon=np.linspace(self.lon_bounds[0], self.lon_bounds[1], self.n_lon),
            lat=np.linspace(self.lat_bounds[0], self.lat_bounds[1], self.n_lat),
            level=np.linspace(self.z_bounds_agl_m[0], self.z_bounds_agl_m[1], self.n_lev),
            pressure_level_hpa=np.linspace(1000.0, 200.0, self.n_lev),
            time_start=t0,
            time_end=t1,
            variable_units={
                "u": "m/s",
                "v": "m/s",
                "w": "m/s",
                "blh": "m",
                "sp": "Pa",
                "t": "K",
                "z": "m**2s**-2",
                "z_sfc": "m**2s**-2",
            },
        )

        return HourlyMetTensors(
            hour_start=_build(u0, v0, w0),
            hour_end=_build(u1, v1, w1),
            metadata=metadata,
        )


def _make_run_config(**overrides: object) -> RunConfig:
    """Construct a RunConfig with sensible test defaults; override what each test needs."""

    defaults: dict[str, object] = dict(
        zarr_store="fake://placeholder",
        start_time=datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
        release_duration_seconds=60,
        simulation_length_seconds=10800,
        release_seed=42,
        n_particles=512,
        release_lon=0.0,
        release_lat=0.0,
        release_alt_agl_m=500.0,
        dt_seconds=300,
        bbox_pad_lon_deg=2.0,
        bbox_pad_lat_deg=2.0,
        bbox_pad_alt_m=3000.0,
        output_uri="outputs/test",
        device="cpu",
        met_cache_max_hours=2,
        memory_log_every_steps=0,
        gc_every_steps=0,
        memory_guard_max_rss_gib=None,
        memory_guard_max_device_allocated_gib=None,
        memory_guard_max_device_reserved_gib=None,
        memory_guard_check_every_steps=1,
    )
    defaults.update(overrides)
    return RunConfig(**defaults)  # type: ignore[arg-type]


def _make_reader(
    cfg: RunConfig,
    wind_fn: Callable[[datetime], tuple[float, float, float]],
) -> AnalyticMetReader:
    return AnalyticMetReader(
        coverage_start=cfg.start_time - timedelta(seconds=cfg.simulation_length_seconds + 7200),
        coverage_end=cfg.start_time + timedelta(seconds=cfg.release_duration_seconds + 3600),
        wind_fn=wind_fn,
    )


def test_run_completes_with_synthetic_met(tmp_path: Path) -> None:
    """Full runtime loop should complete and produce all expected output artifacts."""

    cfg = _make_run_config(output_uri=str(tmp_path / "out"))
    reader = _make_reader(cfg, wind_fn=lambda _: (5.0, 0.0, 0.0))

    metadata = _run(cfg, reader=reader)

    assert metadata["runtime"]["status"] == "completed"
    out = tmp_path / "out"
    assert (out / "endpoint_particles.parquet").exists()
    assert (out / "trajectory_diagnostics.parquet").exists()
    assert (out / "footprints.zarr").exists()
    assert (out / "run_metadata.json").exists()


def test_preflight_rejects_insufficient_met_coverage(tmp_path: Path) -> None:
    """Coverage that doesn't extend back through sim_start should fail preflight."""

    cfg = _make_run_config(output_uri=str(tmp_path / "out"))
    reader = AnalyticMetReader(
        coverage_start=cfg.start_time,
        coverage_end=cfg.start_time + timedelta(hours=2),
        wind_fn=lambda _: (0.0, 0.0, 0.0),
    )

    with pytest.raises(PreflightValidationError):
        _run(cfg, reader=reader)


def test_constant_wind_advection_trajectory(tmp_path: Path) -> None:
    """Mean particle position under constant wind should match analytic backward transport.

    With u=5 m/s eastward and a release window shorter than dt, iter 1 has 0 active
    particles (release_times are sampled half-open) and the remaining (n_steps - 1)
    iterations transport every particle by -u*dt. Final mean lon should match
    release_lon - u*(n_steps - 1)*dt / m_per_deg_lon.
    """

    cfg = _make_run_config(
        release_duration_seconds=60,
        simulation_length_seconds=10800,
        dt_seconds=300,
        release_lon=0.0,
        release_lat=0.0,
        n_particles=512,
        release_seed=42,
        output_uri=str(tmp_path / "out"),
    )
    u_const = 5.0
    reader = _make_reader(cfg, wind_fn=lambda _: (u_const, 0.0, 0.0))

    _run(cfg, reader=reader)

    traj = pd.read_parquet(tmp_path / "out" / "trajectory_diagnostics.parquet")
    final_lon = float(traj.iloc[-1]["mean_lon"])

    n_steps = cfg.simulation_length_seconds // cfg.dt_seconds
    expected_disp_m = -u_const * (n_steps - 1) * cfg.dt_seconds
    m_per_deg_lon = 111320.0
    expected_final_lon = cfg.release_lon + expected_disp_m / m_per_deg_lon

    assert abs(final_lon - expected_final_lon) < 5e-3


def test_footprint_total_matches_active_particle_time(tmp_path: Path) -> None:
    """Footprint sum should equal sum(active_count) * dt / n_particles for in-domain runs.

    Each accumulate call adds (active_weight * dt). With uniform per-particle weight
    1/n_particles, the total accumulated value equals the trajectory's sum of
    (active_count * dt) divided by n_particles, provided no particle leaves the
    gridder domain. Run length is kept short so vertical turbulence does not push
    particles above the hardcoded z_max=5000 m gridder upper bound.
    """

    cfg = _make_run_config(
        release_duration_seconds=60,
        simulation_length_seconds=3600,
        dt_seconds=300,
        n_particles=512,
        release_seed=42,
        output_uri=str(tmp_path / "out"),
    )
    reader = _make_reader(cfg, wind_fn=lambda _: (5.0, 0.0, 0.0))

    _run(cfg, reader=reader)

    traj = pd.read_parquet(tmp_path / "out" / "trajectory_diagnostics.parquet")
    expected_total = float(
        (traj["active_particles"].astype(float) * cfg.dt_seconds).sum()
    ) / cfg.n_particles

    fp = xr.open_zarr(tmp_path / "out" / "footprints.zarr")
    actual_total = float(fp["footprint"].sum())

    assert abs(actual_total - expected_total) / expected_total < 1e-5
