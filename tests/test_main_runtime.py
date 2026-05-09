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

from lpdm.config import RunConfig
from lpdm.main import PreflightValidationError, _run
from lpdm.met_reader import (
    BoundingBoxRequest,
    HourlyMetTensors,
    MetFieldMetadata,
    MetReader,
)


@dataclass
class AnalyticMetReader(MetReader):
    """Synthetic met reader returning spatially-uniform analytic wind fields."""

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
    t_kelvin: float = 280.0
    ustar_m_s: float = 0.4
    shf_w_m2: float = 0.0
    channel_names: tuple[str, ...] = ("u", "v", "w", "blh", "sp")
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
            available = {
                "u": torch.full(shape, float(u), dtype=self.dtype, device=self.device),
                "v": torch.full(shape, float(v), dtype=self.dtype, device=self.device),
                "w": torch.full(shape, float(w), dtype=self.dtype, device=self.device),
                "blh": torch.full(shape, self.blh_m, dtype=self.dtype, device=self.device),
                "sp": torch.full(shape, self.sp_pa, dtype=self.dtype, device=self.device),
                "t": torch.full(shape, self.t_kelvin, dtype=self.dtype, device=self.device),
                "ustar": torch.full(shape, self.ustar_m_s, dtype=self.dtype, device=self.device),
                "shf": torch.full(shape, self.shf_w_m2, dtype=self.dtype, device=self.device),
            }
            missing = [k for k in self.channel_names if k not in available]
            if missing:
                raise KeyError(f"AnalyticMetReader cannot supply channels: {missing}")
            return torch.stack([available[name] for name in self.channel_names], dim=0)

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
                "ustar": "m/s",
                "shf": "W m**-2",
            },
        )

        return HourlyMetTensors(
            hour_start=_build(u0, v0, w0),
            hour_end=_build(u1, v1, w1),
            metadata=metadata,
            channel_names=tuple(self.channel_names),
        )


def _make_run_config(**flat_overrides: object) -> RunConfig:
    """Build a RunConfig with sensible test defaults from a flat keyword surface.

    Hides the nested schema from tests that just need to flip a knob or two.
    """

    flat: dict[str, object] = dict(
        # IO
        zarr_store="fake://placeholder",
        output_uri="outputs/test",
        # Simulation
        start_time=datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
        simulation_length_seconds=10800,
        dt_seconds=300,
        device="cpu",
        # Release
        release_kind="point",
        release_lon=0.0,
        release_lat=0.0,
        release_alt_agl_m=500.0,
        release_duration_seconds=60,
        n_particles=512,
        release_seed=42,
        # Turbulence
        turbulence_scheme="placeholder_constant_ou",
        # Output grid
        output_lon_bounds=(-2.0, 2.0),
        output_lat_bounds=(-2.0, 2.0),
        output_n_x=16,
        output_n_y=16,
        z_edges_m=(0.0, 1000.0, 2000.0, 3000.0, 4000.0, 5000.0),
        # Use 3 hourly time bins by default; tests with different sim lengths can override.
        n_time_bins=3,
        # Met domain
        met_lon_bounds=(-3.0, 3.0),
        met_lat_bounds=(-3.0, 3.0),
        met_alt_max_m=10000.0,
        # Memory
        met_cache_max_hours=2,
        memory_log_every_steps=0,
        gc_every_steps=0,
        memory_guard_max_rss_gib=None,
        memory_guard_max_device_allocated_gib=None,
        memory_guard_max_device_reserved_gib=None,
        memory_guard_check_every_steps=1,
    )
    flat.update(flat_overrides)

    return RunConfig.model_validate(
        {
            "io": {
                "zarr_store": flat["zarr_store"],
                "output_uri": flat["output_uri"],
            },
            "simulation": {
                "start_time": flat["start_time"],
                "length_seconds": flat["simulation_length_seconds"],
                "dt_seconds": flat["dt_seconds"],
                "device": flat["device"],
            },
            "release": {
                "kind": flat["release_kind"],
                "lon": flat["release_lon"],
                "lat": flat["release_lat"],
                "alt_agl_m": flat["release_alt_agl_m"],
                "duration_seconds": flat["release_duration_seconds"],
                "n_particles": flat["n_particles"],
                "seed": flat["release_seed"],
            },
            "turbulence": {"scheme": flat["turbulence_scheme"]},
            "output_grid": {
                "lon_bounds": flat["output_lon_bounds"],
                "lat_bounds": flat["output_lat_bounds"],
                "n_x": flat["output_n_x"],
                "n_y": flat["output_n_y"],
                "z_edges_m": flat["z_edges_m"],
                "n_time_bins": flat["n_time_bins"],
            },
            "met_domain": {
                "lon_bounds": flat["met_lon_bounds"],
                "lat_bounds": flat["met_lat_bounds"],
                "alt_max_m": flat["met_alt_max_m"],
            },
            "memory": {
                "met_cache_max_hours": flat["met_cache_max_hours"],
                "log_every_steps": flat["memory_log_every_steps"],
                "gc_every_steps": flat["gc_every_steps"],
                "guard_max_rss_gib": flat["memory_guard_max_rss_gib"],
                "guard_max_device_allocated_gib": flat["memory_guard_max_device_allocated_gib"],
                "guard_max_device_reserved_gib": flat["memory_guard_max_device_reserved_gib"],
                "guard_check_every_steps": flat["memory_guard_check_every_steps"],
            },
        }
    )


def _make_reader(
    cfg: RunConfig,
    wind_fn: Callable[[datetime], tuple[float, float, float]],
) -> AnalyticMetReader:
    return AnalyticMetReader(
        coverage_start=cfg.simulation.start_time - timedelta(seconds=cfg.simulation.length_seconds + 7200),
        coverage_end=cfg.simulation.start_time + timedelta(seconds=cfg.release.duration_seconds + 3600),
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
        coverage_start=cfg.simulation.start_time,
        coverage_end=cfg.simulation.start_time + timedelta(hours=2),
        wind_fn=lambda _: (0.0, 0.0, 0.0),
    )

    with pytest.raises(PreflightValidationError):
        _run(cfg, reader=reader)


def test_constant_wind_advection_trajectory(tmp_path: Path) -> None:
    """Mean particle position under constant wind should match analytic backward transport."""

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

    n_steps = cfg.simulation.length_seconds // cfg.simulation.dt_seconds
    expected_disp_m = -u_const * (n_steps - 1) * cfg.simulation.dt_seconds
    m_per_deg_lon = 111320.0
    expected_final_lon = cfg.release.lon + expected_disp_m / m_per_deg_lon

    assert abs(final_lon - expected_final_lon) < 5e-3


def _make_hanna_reader(
    cfg: RunConfig,
    wind_fn: Callable[[datetime], tuple[float, float, float]],
    *,
    ustar_m_s: float = 0.4,
    shf_w_m2: float = 0.0,
    t_kelvin: float = 280.0,
) -> AnalyticMetReader:
    """AnalyticMetReader configured to supply the channels Hanna needs."""

    return AnalyticMetReader(
        coverage_start=cfg.simulation.start_time - timedelta(seconds=cfg.simulation.length_seconds + 7200),
        coverage_end=cfg.simulation.start_time + timedelta(seconds=cfg.release.duration_seconds + 3600),
        wind_fn=wind_fn,
        channel_names=("u", "v", "w", "blh", "sp", "t", "ustar", "shf"),
        ustar_m_s=ustar_m_s,
        shf_w_m2=shf_w_m2,
        t_kelvin=t_kelvin,
    )


def test_hanna_run_completes_with_synthetic_met(tmp_path: Path) -> None:
    """Hanna scheme should run end-to-end and produce all expected artifacts."""

    cfg = _make_run_config(
        output_uri=str(tmp_path / "out"),
        turbulence_scheme="hanna_1982",
    )
    reader = _make_hanna_reader(cfg, wind_fn=lambda _: (5.0, 0.0, 0.0))

    metadata = _run(cfg, reader=reader)

    assert metadata["runtime"]["status"] == "completed"
    out = tmp_path / "out"
    assert (out / "endpoint_particles.parquet").exists()
    assert (out / "trajectory_diagnostics.parquet").exists()
    assert (out / "footprints.zarr").exists()
    assert (out / "run_metadata.json").exists()


def test_hanna_constant_wind_preserves_mean_trajectory(tmp_path: Path) -> None:
    """Hanna's zero-mean perturbations shouldn't bias the ensemble mean position."""

    cfg = _make_run_config(
        release_duration_seconds=60,
        simulation_length_seconds=10800,
        dt_seconds=300,
        release_lon=0.0,
        release_lat=0.0,
        n_particles=512,
        release_seed=42,
        output_uri=str(tmp_path / "out"),
        turbulence_scheme="hanna_1982",
    )
    u_const = 5.0
    reader = _make_hanna_reader(cfg, wind_fn=lambda _: (u_const, 0.0, 0.0))

    _run(cfg, reader=reader)

    traj = pd.read_parquet(tmp_path / "out" / "trajectory_diagnostics.parquet")
    final_lon = float(traj.iloc[-1]["mean_lon"])

    n_steps = cfg.simulation.length_seconds // cfg.simulation.dt_seconds
    expected_disp_m = -u_const * (n_steps - 1) * cfg.simulation.dt_seconds
    m_per_deg_lon = 111320.0
    expected_final_lon = cfg.release.lon + expected_disp_m / m_per_deg_lon

    # Looser tolerance than the placeholder test to accommodate ensemble noise from
    # the new horizontal stochastic component.
    assert abs(final_lon - expected_final_lon) < 1e-2


def test_hanna_produces_nontrivial_vertical_spread(tmp_path: Path) -> None:
    """Hanna with positive sensible heat flux should produce substantial vertical spread."""

    cfg = _make_run_config(
        release_duration_seconds=60,
        simulation_length_seconds=3600,
        dt_seconds=120,
        release_alt_agl_m=500.0,
        n_particles=512,
        release_seed=42,
        output_uri=str(tmp_path / "out"),
        turbulence_scheme="hanna_1982",
        n_time_bins=1,
    )
    # Convective conditions: positive SHF, w* > 0, particles should mix vertically.
    reader = _make_hanna_reader(
        cfg, wind_fn=lambda _: (0.0, 0.0, 0.0), ustar_m_s=0.5, shf_w_m2=200.0,
    )

    _run(cfg, reader=reader)

    endpoints = pd.read_parquet(tmp_path / "out" / "endpoint_particles.parquet")
    alt_std = float(endpoints["alt"].std())
    # Empirically, an unstable run with sigma_w ~0.7 m/s for ~1 hour should produce
    # vertical std of order 100m+. Loose threshold to avoid flakiness.
    assert alt_std > 50.0
    # Surface reflection should keep all particles non-negative.
    assert float(endpoints["alt"].min()) >= 0.0


def test_footprint_total_matches_active_particle_time(tmp_path: Path) -> None:
    """Footprint sum should equal sum(active_count) * dt / n_particles for in-domain runs."""

    cfg = _make_run_config(
        release_duration_seconds=60,
        simulation_length_seconds=3600,
        dt_seconds=300,
        n_particles=512,
        release_seed=42,
        output_uri=str(tmp_path / "out"),
        n_time_bins=1,
    )
    reader = _make_reader(cfg, wind_fn=lambda _: (5.0, 0.0, 0.0))

    _run(cfg, reader=reader)

    traj = pd.read_parquet(tmp_path / "out" / "trajectory_diagnostics.parquet")
    expected_total = float(
        (traj["active_particles"].astype(float) * cfg.simulation.dt_seconds).sum()
    ) / cfg.release.n_particles

    fp = xr.open_zarr(tmp_path / "out" / "footprints.zarr")
    actual_total = float(fp["footprint"].sum())

    assert abs(actual_total - expected_total) / expected_total < 1e-5
