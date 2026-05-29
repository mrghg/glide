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
from lpdm.main import PreflightValidationError, _run, _within_met_domain
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

        # 3D AGL height: per-level height broadcast over the (lat, lon) grid.
        level_agl = torch.linspace(
            self.z_bounds_agl_m[0], self.z_bounds_agl_m[1], self.n_lev,
            dtype=self.dtype, device=self.device,
        )
        height_agl_m = level_agl.view(self.n_lev, 1, 1).expand(self.n_lev, self.n_lat, self.n_lon).contiguous()

        return HourlyMetTensors(
            hour_start=_build(u0, v0, w0),
            hour_end=_build(u1, v1, w1),
            metadata=metadata,
            channel_names=tuple(self.channel_names),
            height_agl_m=height_agl_m,
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


def test_hanna_well_mixed_no_runaway_lofting(tmp_path: Path) -> None:
    """Regression for the backward-Langevin drift sign. From a mid-BL release the
    well-mixed drift must keep particles cycling through the BL — a good fraction
    should end up *below* the release height. The earlier wrong-sign drift lofted
    essentially the whole population upward (one-way valve out of the BL), which
    this test would catch (fraction-below → ~0, mean → above BLH)."""

    blh = 1200.0
    release_alt = 600.0  # mid-BL
    cfg = _make_run_config(
        release_duration_seconds=60,
        simulation_length_seconds=21600,  # 6 h backward, long enough to equilibrate
        dt_seconds=120,
        release_alt_agl_m=release_alt,
        n_particles=2000,
        release_seed=7,
        output_uri=str(tmp_path / "out"),
        turbulence_scheme="hanna_1982",
        n_time_bins=1,
        met_alt_max_m=10000.0,
    )
    # Near-neutral BL (no convection), so σ_w is set by u* and decays toward the
    # BL top — the inhomogeneity the well-mixed drift must handle.
    reader = _make_hanna_reader(
        cfg, wind_fn=lambda _: (0.0, 0.0, 0.0), ustar_m_s=0.4, shf_w_m2=0.0,
    )
    # Override the synthetic BLH to a defined value for a clean expectation.
    reader.blh_m = blh

    _run(cfg, reader=reader)

    alt = pd.read_parquet(tmp_path / "out" / "endpoint_particles.parquet")["alt"].values
    mean_alt = float(alt.mean())
    frac_below_blh = float((alt < blh).mean())

    # The population stays BL-confined on average rather than running away upward
    # (the wrong-sign drift drove the whole population above the BLH: mean ≈ 2000 m,
    # p10 ≈ 1660 m for this setup).
    assert mean_alt < blh
    # A meaningful fraction recycles down into / below the BL, and particles reach
    # the near-surface layer — both ≈ impossible under the one-way upward lofting.
    assert frac_below_blh > 0.1
    assert float(alt.min()) < 0.5 * release_alt
    # No spurious escape to the top of the domain.
    assert float(alt.max()) < 0.9 * cfg.met_domain.alt_max_m


def test_footprint_total_matches_active_particle_time(tmp_path: Path) -> None:
    """Footprint sum should equal sum(active_count) * dt / n_particles for in-domain runs.

    Tolerance 1e-3 accounts for float32 accumulation noise across the 5D
    scatter_add. The exact value of the noise depends on the order torch's
    global RNG draws fall in, which differs depending on what other tests run
    first; the bound is loose enough to be robust to that.
    """

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

    assert abs(actual_total - expected_total) / expected_total < 1e-3


# ---- M5 stage 5: multi-release runtime tests ---------------------------------


def _make_periodic_config(
    *,
    output_uri: str,
    n_releases: int = 3,
    period_seconds: int = 3600,
    duration_seconds: int = 300,
    n_particles_per_release: int = 128,
    simulation_length_seconds: int = 3600,
    max_releases_per_batch: int = 24,
    start_time: datetime | None = None,
    seed: int | None = 42,
) -> RunConfig:
    # Note: duration_seconds defaults to dt_seconds=300 so every release
    # window-end lands exactly on the cursor grid. With a duration not a
    # multiple of dt, the cursor skips the upper-boundary visit for the
    # earliest release in each batch and that release loses one step's
    # worth of mass — a real but small discrete-time effect documented
    # in the M5 stage 5 entry in CHECKPOINT.md.
    start_time = start_time or datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    return RunConfig.model_validate(
        {
            "io": {"zarr_store": "fake://x", "output_uri": output_uri},
            "simulation": {
                "start_time": start_time,
                "length_seconds": simulation_length_seconds,
                "dt_seconds": 300,
                "device": "cpu",
            },
            "release": {
                "kind": "periodic_point",
                "point": {"lon": 0.0, "lat": 0.0, "alt_agl_m": 500.0},
                "start_time": start_time,
                "period_seconds": period_seconds,
                "n_releases": n_releases,
                "duration_seconds": duration_seconds,
                "n_particles_per_release": n_particles_per_release,
                "seed": seed,
            },
            "turbulence": {"scheme": "placeholder_constant_ou"},
            "output_grid": {
                "lon_bounds": (-2.0, 2.0),
                "lat_bounds": (-2.0, 2.0),
                "n_x": 16,
                "n_y": 16,
                "z_edges_m": (0.0, 1000.0, 5000.0),
                "n_time_bins": 3,
            },
            "met_domain": {
                "lon_bounds": (-3.0, 3.0),
                "lat_bounds": (-3.0, 3.0),
                "alt_max_m": 10000.0,
            },
            "memory": {
                "met_cache_max_hours": 4,
                "log_every_steps": 0,
                "gc_every_steps": 0,
                "guard_check_every_steps": 1,
            },
            "batch": {"max_releases_per_batch": max_releases_per_batch},
        }
    )


def _make_multi_release_reader(
    cfg: RunConfig, wind_fn: Callable[[datetime], tuple[float, float, float]]
) -> AnalyticMetReader:
    """Met coverage spanning the whole expanded schedule."""

    all_releases = [r for b in cfg.expand_to_batches() for r in b.releases]
    earliest = min(r.release_time for r in all_releases)
    latest_end = max(
        r.release_time + timedelta(seconds=r.duration_seconds) for r in all_releases
    )
    return AnalyticMetReader(
        coverage_start=earliest - timedelta(seconds=cfg.simulation.length_seconds + 7200),
        coverage_end=latest_end + timedelta(hours=1),
        wind_fn=wind_fn,
    )


def test_periodic_release_run_completes_and_emits_5d_footprint(tmp_path: Path) -> None:
    """End-to-end periodic_point run: outputs land, shape is 5D with the right release_time coord."""

    cfg = _make_periodic_config(
        output_uri=str(tmp_path / "out"),
        n_releases=3,
        period_seconds=3600,
        duration_seconds=60,
    )
    reader = _make_multi_release_reader(cfg, wind_fn=lambda _: (5.0, 0.0, 0.0))

    metadata = _run(cfg, reader=reader)

    assert metadata["runtime"]["status"] == "completed"
    assert metadata["schedule"]["n_releases"] == 3
    assert metadata["schedule"]["n_batches"] == 1  # 3 releases ≤ default 24/batch

    out = tmp_path / "out"
    fp = xr.open_zarr(out / "footprints.zarr")
    assert fp["footprint"].dims == (
        "release_time",
        "time_ago",
        "z_bin",
        "latitude",
        "longitude",
    )
    assert fp["footprint"].sizes["release_time"] == 3

    expected_times = [
        cfg.simulation.start_time + timedelta(seconds=i * 3600) for i in range(3)
    ]
    actual_times = pd.to_datetime(fp["release_time"].values).tz_localize("UTC")
    for actual, expected in zip(actual_times, expected_times):
        assert actual == expected


def test_periodic_release_endpoint_parquet_carries_release_idx_column(tmp_path: Path) -> None:
    cfg = _make_periodic_config(
        output_uri=str(tmp_path / "out"),
        n_releases=2,
        n_particles_per_release=64,
    )
    reader = _make_multi_release_reader(cfg, wind_fn=lambda _: (5.0, 0.0, 0.0))

    _run(cfg, reader=reader)

    df = pd.read_parquet(tmp_path / "out" / "endpoint_particles.parquet")
    assert list(df.columns) == ["lon", "lat", "alt", "weight", "release_idx"]
    assert len(df) == 2 * 64
    counts = df["release_idx"].value_counts().sort_index()
    assert counts.tolist() == [64, 64]


def test_periodic_release_trajectory_carries_batch_idx_column(tmp_path: Path) -> None:
    cfg = _make_periodic_config(
        output_uri=str(tmp_path / "out"),
        n_releases=5,
        max_releases_per_batch=2,
    )
    reader = _make_multi_release_reader(cfg, wind_fn=lambda _: (5.0, 0.0, 0.0))

    _run(cfg, reader=reader)

    traj = pd.read_parquet(tmp_path / "out" / "trajectory_diagnostics.parquet")
    assert "batch_idx" in traj.columns
    # 5 releases at max_releases_per_batch=2 → batches indexed 0, 1, 2.
    assert sorted(traj["batch_idx"].unique().tolist()) == [0, 1, 2]


def test_periodic_release_footprint_mass_matches_active_particle_time(tmp_path: Path) -> None:
    """Total footprint mass = sum(active_count * dt) / n_particles_per_release.

    Multi-release analog of the M0 conservation test. The trajectory diagnostic
    rows already record the total active-particle count per step (summed across
    all releases in the batch), so the global mass conservation check works the
    same way as for single-release runs.
    """

    cfg = _make_periodic_config(
        output_uri=str(tmp_path / "out"),
        n_releases=3,
        period_seconds=3600,
        duration_seconds=300,
        n_particles_per_release=256,
        simulation_length_seconds=3600,
    )
    reader = _make_multi_release_reader(cfg, wind_fn=lambda _: (5.0, 0.0, 0.0))

    _run(cfg, reader=reader)

    traj = pd.read_parquet(tmp_path / "out" / "trajectory_diagnostics.parquet")
    expected_total = float(
        (traj["active_particles"].astype(float) * cfg.simulation.dt_seconds).sum()
    ) / cfg.release.n_particles_per_release

    fp = xr.open_zarr(tmp_path / "out" / "footprints.zarr")["footprint"]
    actual_total = float(fp.sum())

    # 1e-3 (not 1e-5 as the single-release test) because the 5D float32
    # scatter_add over more particles accumulates more rounding error.
    assert abs(actual_total - expected_total) / expected_total < 1e-3


def test_periodic_release_per_release_mass_near_expected(tmp_path: Path) -> None:
    """Constant wind ⇒ each release accumulates ≈ sim_length × dt per particle.

    Per-release totals are not exactly equal because the cursor loop excludes
    its terminus value: the earliest release in each batch's reach-back loses
    one cursor step's worth of mass at the lower boundary. With
    sim_length=10800 s and dt=300 s that's ≈ 36 steps/release, so a 1-step
    asymmetry is ~3%. The tolerance below is 5%.
    """

    cfg = _make_periodic_config(
        output_uri=str(tmp_path / "out"),
        n_releases=3,
        period_seconds=3600,
        duration_seconds=300,
        n_particles_per_release=256,
        simulation_length_seconds=10800,
        max_releases_per_batch=24,  # single batch so only the earliest release pays the cost
    )
    reader = _make_multi_release_reader(cfg, wind_fn=lambda _: (5.0, 0.0, 0.0))

    _run(cfg, reader=reader)

    fp = xr.open_zarr(tmp_path / "out" / "footprints.zarr")["footprint"]
    per_release_total = fp.sum(dim=("time_ago", "z_bin", "latitude", "longitude")).values
    expected_per_release = cfg.simulation.length_seconds  # = n_steps * dt = mass per release
    for total in per_release_total:
        rel_error = abs(float(total) - expected_per_release) / expected_per_release
        assert rel_error < 5e-2


def test_periodic_release_batch_chunking_preserves_shape_and_conserves_per_chunking(
    tmp_path: Path,
) -> None:
    """Multi-batch runs must produce the same 5D shape as single-batch runs, and
    mass conservation (footprint_total == sum(active_count*dt) / n_particles)
    must hold *within each chunking*.

    Per-release totals do depend on chunking because the earliest release in
    each batch pays a one-cursor-step boundary cost — so a 2-batch run of 4
    releases has two boundary losses while a single-batch run has one. The
    invariant is conservation against the diagnostic, not bit-equivalence
    across chunkings.
    """

    def _run_and_check(out_path: Path, max_per_batch: int) -> tuple[tuple[int, ...], float]:
        cfg = _make_periodic_config(
            output_uri=str(out_path),
            n_releases=4,
            max_releases_per_batch=max_per_batch,
            n_particles_per_release=128,
        )
        reader = _make_multi_release_reader(cfg, wind_fn=lambda _: (5.0, 0.0, 0.0))
        _run(cfg, reader=reader)

        traj = pd.read_parquet(out_path / "trajectory_diagnostics.parquet")
        expected = float(
            (traj["active_particles"].astype(float) * cfg.simulation.dt_seconds).sum()
        ) / cfg.release.n_particles_per_release

        fp = xr.open_zarr(out_path / "footprints.zarr")["footprint"]
        actual = float(fp.sum())
        # 5e-3 to allow for float32 accumulation noise across the 5D scatter,
        # which scales with cursor_steps × n_particles. With 4 releases this is
        # ~48 steps × 512 particles ≈ 25k ops, so float32 epsilon × ops ≈ 3e-3.
        assert abs(actual - expected) / expected < 5e-3
        return tuple(fp.shape), actual

    shape_one, total_one = _run_and_check(tmp_path / "one", max_per_batch=24)
    shape_two, total_two = _run_and_check(tmp_path / "two", max_per_batch=2)

    # Shapes must agree (same n_releases, same grid).
    assert shape_one == shape_two
    # Totals can differ by one boundary-step per extra batch; just sanity-bound
    # the difference to a few percent.
    assert abs(total_one - total_two) / total_one < 5e-2


# ---- M4: drop-and-count out-of-domain particle handling ----------------------


def test_within_met_domain_flags_out_of_bounds_particles() -> None:
    """Unit check of the domain predicate: lon/lat bbox + alt_max, no lower kill."""

    cfg = _make_run_config(
        met_lon_bounds=(-3.0, 3.0),
        met_lat_bounds=(-2.0, 2.0),
        met_alt_max_m=5000.0,
    )
    # [lon, lat, alt, weight]; only the first 3 columns are read.
    particles = torch.tensor(
        [
            [0.0, 0.0, 100.0, 1.0],     # inside
            [-3.0, 0.0, 100.0, 1.0],    # on lon edge -> inside (inclusive)
            [-3.01, 0.0, 100.0, 1.0],   # just past lon_min -> outside
            [3.01, 0.0, 100.0, 1.0],    # past lon_max -> outside
            [0.0, 2.5, 100.0, 1.0],     # past lat_max -> outside
            [0.0, 0.0, 5000.1, 1.0],    # above alt_max -> outside
            [0.0, 0.0, 0.0, 1.0],       # at the surface -> inside (no lower kill)
        ],
        dtype=torch.float64,
    )
    within = _within_met_domain(particles, cfg)
    assert within.tolist() == [True, True, False, False, False, False, True]


def test_particles_killed_on_met_domain_exit(tmp_path: Path) -> None:
    """A strong wind pushes particles out of a small met_domain; they must be
    killed (drop-and-count): escaped counts appear in diagnostics, the alive set
    monotonically shrinks, and the active set collapses to zero by the end."""

    cfg = _make_run_config(
        output_uri=str(tmp_path / "out"),
        simulation_length_seconds=10800,   # 3 h backward
        dt_seconds=300,
        n_particles=256,
        release_seed=42,
        n_time_bins=1,
        met_lon_bounds=(-3.0, 3.0),
        met_lat_bounds=(-3.0, 3.0),
        output_lon_bounds=(-2.5, 2.5),
        output_lat_bounds=(-2.5, 2.5),
    )
    # u=+50 m/s: backward advection drives particles west, out through lon_min
    # (-3 deg ≈ 333 km) after ~1.85 h — well inside the 3 h window.
    reader = _make_reader(cfg, wind_fn=lambda _: (50.0, 0.0, 0.0))

    metadata = _run(cfg, reader=reader)

    traj = pd.read_parquet(tmp_path / "out" / "trajectory_diagnostics.parquet")
    assert "escaped_this_step" in traj.columns
    assert "alive_particles" in traj.columns

    # Some particles escaped.
    assert int(traj["escaped_this_step"].sum()) > 0
    assert metadata["schedule"]["escaped_met_domain_total"] == int(traj["escaped_this_step"].sum())

    # alive_particles is monotonically non-increasing and ends below the start.
    alive = traj["alive_particles"].tolist()
    assert all(alive[i] >= alive[i + 1] for i in range(len(alive) - 1))
    assert alive[-1] < alive[0]

    # The active set shrinks as particles escape (efficiency win): the final
    # step has far fewer active particles than the peak.
    assert int(traj["active_particles"].iloc[-1]) < int(traj["active_particles"].max())


def test_no_escapes_leaves_alive_count_full(tmp_path: Path) -> None:
    """A run that stays in-domain must kill nothing: alive stays at n_particles
    and the escaped total is zero (guards bit-equivalence with pre-kill runs)."""

    cfg = _make_run_config(
        output_uri=str(tmp_path / "out"),
        simulation_length_seconds=3600,
        dt_seconds=300,
        n_particles=128,
        release_seed=42,
        n_time_bins=1,
        met_lon_bounds=(-5.0, 5.0),
        met_lat_bounds=(-5.0, 5.0),
    )
    reader = _make_reader(cfg, wind_fn=lambda _: (5.0, 0.0, 0.0))  # gentle: ~0.16 deg drift

    metadata = _run(cfg, reader=reader)

    traj = pd.read_parquet(tmp_path / "out" / "trajectory_diagnostics.parquet")
    assert int(traj["escaped_this_step"].sum()) == 0
    assert (traj["alive_particles"] == 128).all()
    assert metadata["schedule"]["escaped_met_domain_total"] == 0
