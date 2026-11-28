"""End-to-end runtime tests using a synthetic analytic meteorology reader.

These tests exercise the full main runtime loop (advection, met fetching, footprint
accumulation, output writing) without depending on remote ERA5 data.
"""

from __future__ import annotations

import dataclasses
import json
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path

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
from lpdm.turbulence.placeholder import PlaceholderConstantOU


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
    q_kg_kg: float = 0.0
    channel_names: tuple[str, ...] = ("u", "v", "w", "blh", "sp")
    device: torch.device | str = "cpu"
    dtype: torch.dtype = torch.float32
    # Source cadence this synthetic store pretends to have. The base class exposes
    # `met_interval_seconds` as a read-only property, so readers declare their
    # cadence by overriding it (as ArcoEra5ZarrReader does from the time coord).
    interval_seconds: int = 3600

    @property
    def met_interval_seconds(self) -> int:
        return self.interval_seconds

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
                "q": torch.full(shape, self.q_kg_kg, dtype=self.dtype, device=self.device),
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
            self.z_bounds_agl_m[0],
            self.z_bounds_agl_m[1],
            self.n_lev,
            dtype=self.dtype,
            device=self.device,
        )
        height_agl_m = (
            level_agl.view(self.n_lev, 1, 1).expand(self.n_lev, self.n_lat, self.n_lon).contiguous()
        )

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
        start_time=datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC),
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
        met_cache_on_host=True,
        met_prefetch=True,
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
                # Plain residence seconds, as before decision 0022; the
                # density-weighting tests set "dry_air".
                "density_weighting": flat.get("density_weighting", "none"),
            },
            "met_domain": {
                "lon_bounds": flat["met_lon_bounds"],
                "lat_bounds": flat["met_lat_bounds"],
                "alt_max_m": flat["met_alt_max_m"],
            },
            "memory": {
                "met_cache_max_hours": flat["met_cache_max_hours"],
                "met_cache_on_host": flat["met_cache_on_host"],
                "met_prefetch": flat["met_prefetch"],
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
        coverage_start=cfg.simulation.start_time
        - timedelta(seconds=cfg.simulation.length_seconds + 7200),
        coverage_end=cfg.simulation.start_time
        + timedelta(seconds=cfg.release.duration_seconds + 3600),
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

    # Every particle is followed for the full simulation length from its release.
    expected_disp_m = -u_const * cfg.simulation.length_seconds
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
        coverage_start=cfg.simulation.start_time
        - timedelta(seconds=cfg.simulation.length_seconds + 7200),
        coverage_end=cfg.simulation.start_time
        + timedelta(seconds=cfg.release.duration_seconds + 3600),
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


@pytest.mark.parametrize("static", [False, True])
def test_hanna_constant_wind_preserves_mean_trajectory(
    tmp_path: Path, monkeypatch, static: bool
) -> None:
    """Hanna's zero-mean perturbations shouldn't bias the ensemble mean position.

    Parametrized over the dynamic path (the runtime advects) and the static path
    (RK2 advection is folded into `_step_core`'s captured graph). Both must reproduce
    the analytic backward trajectory — this is the advection-correctness check for the
    fold (conservation alone is position-independent and wouldn't catch broken advection)."""

    monkeypatch.setenv("GLIDE_STATIC_SUBSTEPS", "1" if static else "0")
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

    # Every particle is followed for the full simulation length from its release.
    expected_disp_m = -u_const * cfg.simulation.length_seconds
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
        cfg,
        wind_fn=lambda _: (0.0, 0.0, 0.0),
        ustar_m_s=0.5,
        shf_w_m2=200.0,
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
        cfg,
        wind_fn=lambda _: (0.0, 0.0, 0.0),
        ustar_m_s=0.4,
        shf_w_m2=0.0,
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


def _varying_wind_met_window(
    *,
    u_amplitude: float = 10.0,
    n_lev: int = 4,
    n_lat: int = 8,
    n_lon: int = 8,
    blh_m: float = 2000.0,
) -> HourlyMetTensors:
    """Met window with u varying linearly across longitude (so the local wind
    variability the meander reads from is non-zero) and v = w = 0."""

    shape = (n_lev, n_lat, n_lon)
    i = torch.arange(n_lon, dtype=torch.float32)
    u_profile = u_amplitude * (i / (n_lon - 1) - 0.5)  # [-A/2, A/2] across lon
    u = u_profile.view(1, 1, n_lon).expand(shape).contiguous()
    zeros = torch.zeros(shape, dtype=torch.float32)

    def _build() -> torch.Tensor:
        chans = {
            "u": u,
            "v": zeros,
            "w": zeros,
            "blh": torch.full(shape, blh_m),
            "sp": torch.full(shape, 101325.0),
            "t": torch.full(shape, 280.0),
            "ustar": torch.full(shape, 0.4),
            "shf": torch.zeros(shape),
        }
        names = ("u", "v", "w", "blh", "sp", "t", "ustar", "shf")
        return torch.stack([chans[n] for n in names], dim=0)

    t0 = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)
    level = np.linspace(0.0, 4000.0, n_lev)
    metadata = MetFieldMetadata(
        lon=np.linspace(-2.0, 2.0, n_lon),
        lat=np.linspace(-2.0, 2.0, n_lat),
        level=level,
        pressure_level_hpa=np.linspace(1000.0, 600.0, n_lev),
        time_start=t0,
        time_end=t0 + timedelta(hours=1),
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
    height = (
        torch.as_tensor(level, dtype=torch.float32).view(n_lev, 1, 1).expand(shape).contiguous()
    )
    fields = _build()
    return HourlyMetTensors(
        hour_start=fields,
        hour_end=fields,
        metadata=metadata,
        channel_names=("u", "v", "w", "blh", "sp", "t", "ustar", "shf"),
        height_agl_m=height,
    )


def _horizontal_spread_after_steps(*, meander_enabled: bool, n_steps: int = 40) -> float:
    """Std-dev of particle longitude after stepping a cloud through the
    varying-wind met window with the given meander setting (fixed seed)."""

    from lpdm.gpu_engine import GPUEngine
    from lpdm.turbulence import HannaScheme

    torch.manual_seed(1234)
    engine = GPUEngine(device="cpu")
    scheme = HannaScheme(
        meander_enabled=meander_enabled,
        meander_coefficient=1.0,
        meander_timescale_seconds=1800.0,
    )
    met = _varying_wind_met_window()

    n = 4000
    particles = torch.zeros(n, 4, dtype=torch.float32)
    particles[:, 2] = 500.0  # release altitude (AGL), mid-BL
    particles[:, 3] = 1.0
    state = scheme.initialize_state(n, device=torch.device("cpu"), dtype=torch.float32)
    active = torch.ones(n, dtype=torch.bool)

    for _ in range(n_steps):
        particles, state = scheme.step(
            particles,
            state,
            met,
            t_alpha=0.5,
            dt_seconds=300.0,
            active_mask=active,
            engine=engine,
        )
    return float(particles[:, 0].std())


def test_hanna_meander_increases_horizontal_spread() -> None:
    """Enabling meander adds an independent horizontal random walk whose σ comes
    from the local grid-wind variability, so the cloud spreads markedly more in
    the horizontal than with the in-BL turbulence alone (Maryon 1998 / FLEXPART
    §4.5)."""

    spread_off = _horizontal_spread_after_steps(meander_enabled=False)
    spread_on = _horizontal_spread_after_steps(meander_enabled=True)

    assert spread_off > 0.0  # in-BL horizontal turbulence still acts
    assert spread_on > 1.5 * spread_off


def _wmc_met_window(
    *,
    blh_m: float = 3000.0,
    ustar_m_s: float = 0.4,
    n_lev: int = 6,
    n_lat: int = 4,
    n_lon: int = 4,
) -> HourlyMetTensors:
    """Met window for the V1 well-mixed test: spatially uniform, neutral BL,
    so σ_w(z) inside the BL has the smooth Hanna gradient (1.3·u*·exp(−2fz/u*))
    that the well-mixed drift must compensate. No advection (u=v=w=0), no
    surface heat flux. Constant T, sp, level layout — everything horizontally
    uniform so any z-distribution drift is attributable to the drift physics."""

    shape = (n_lev, n_lat, n_lon)
    zeros = torch.zeros(shape, dtype=torch.float32)
    chans = {
        "u": zeros,
        "v": zeros,
        "w": zeros,
        "blh": torch.full(shape, blh_m),
        "sp": torch.full(shape, 101325.0),
        "t": torch.full(shape, 280.0),
        "ustar": torch.full(shape, ustar_m_s),
        "shf": torch.zeros(shape),
    }
    names = ("u", "v", "w", "blh", "sp", "t", "ustar", "shf")
    fields = torch.stack([chans[n] for n in names], dim=0)

    t0 = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)
    level = np.linspace(0.0, max(4000.0, 1.5 * blh_m), n_lev)
    metadata = MetFieldMetadata(
        lon=np.linspace(-2.0, 2.0, n_lon),
        lat=np.linspace(40.0, 50.0, n_lat),  # ~45° lat → nonzero Coriolis
        level=level,
        pressure_level_hpa=np.linspace(1000.0, 600.0, n_lev),
        time_start=t0,
        time_end=t0 + timedelta(hours=1),
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
    height = (
        torch.as_tensor(level, dtype=torch.float32).view(n_lev, 1, 1).expand(shape).contiguous()
    )
    return HourlyMetTensors(
        hour_start=fields,
        hour_end=fields,
        metadata=metadata,
        channel_names=names,
        height_agl_m=height,
    )


@pytest.mark.parametrize("static_substeps", [False, True])
def test_v1_well_mixed_hanna_backward_path(static_substeps: bool) -> None:
    """V1 well-mixed test against the production HannaScheme backward code path.

    Thomson (1987) WMC: a well-mixed distribution of particles in z must remain
    well-mixed under the (drift + reflection + OU) integration. In the neutral BL
    σ_w varies smoothly with z (`1.3·u*·exp(−2fz/u*)`); without the well-mixed
    drift OR with a broken reflection (the bug F1 fixed in this PR), particles
    would either accumulate in the low-σ region near the BL top or hang near the
    surface after reflecting. With both physics correct, the steady-state binned
    occupancy stays flat to within finite-sample noise.

    This is the strongest single empirical check on the integrated scheme — it
    fails if EITHER the drift sign/magnitude is wrong OR reflection doesn't flip
    w' OR dt is so large the OU integration is biased. It is therefore the
    acceptance gate for F1, F2, F3, F4 jointly. After the F2 density-term lands,
    the assertion becomes that the distribution matches ρ(z), not flat — see the
    companion `test_v1_density_weighted_well_mixed` test."""

    from lpdm.gpu_engine import GPUEngine
    from lpdm.turbulence import HannaScheme

    torch.manual_seed(202605301)

    blh = 3000.0
    engine = GPUEngine(device="cpu")
    # WMC test is for in-BL Hanna only. Run it against BOTH substep-loop variants
    # so the static-shape (CUDA) path is held to the same well-mixed criterion as
    # the dynamic (CPU) path (docs/architecture.md §5).
    scheme = HannaScheme(meander_enabled=False, static_substeps=static_substeps)
    met = _wmc_met_window(blh_m=blh, ustar_m_s=0.4)

    n = 6000
    particles = torch.zeros(n, 4, dtype=torch.float32)
    particles[:, 1] = 45.0  # mid-latitude → nonzero Coriolis
    # Uniform initial in [0, blh] — the well-mixed reference distribution.
    particles[:, 2] = torch.rand(n, dtype=torch.float32) * blh
    particles[:, 3] = 1.0 / n
    state = scheme.initialize_state(n, device=torch.device("cpu"), dtype=torch.float32)
    active = torch.ones(n, dtype=torch.bool)

    dt = 15.0
    n_steps = 1500  # ≈ 100·T_L_typical → fully equilibrated if WMC holds
    # Manual reflection at z=blh (z=0 is handled inside scheme.step). The scheme
    # doesn't natively reflect at the BL top — for this synthetic test we add it
    # so particles can't escape into the (different) FT closure during the run.
    # Same physics as the W&F §6 smooth-wall reflection: flip both z and w'.
    for _ in range(n_steps):
        particles, state = scheme.step(
            particles,
            state,
            met,
            t_alpha=0.5,
            dt_seconds=dt,
            active_mask=active,
            engine=engine,
        )
        above = particles[:, 2] > blh
        if bool(above.any()):
            particles[above, 2] = 2.0 * blh - particles[above, 2]
            state["w_prime"][above] = -state["w_prime"][above]

    # Histogram the full BL into 10 bins of 300 m. Assert flatness on the
    # *interior* (bins 2..7, i.e. z ∈ [600, 2400] m) — the surface-layer bin
    # and the BL-top bin are subject to:
    #   - the W&F §7b dt-bias for inhomogeneous Gaussian turbulence (linear in
    #     Δt/τ, biggest where τ is smallest → near the surface);
    #   - the surface-layer formula switchover at z = 0.1·blh = 300 m;
    #   - reflection-induced piling that W&F document is acceptable but not
    #     WMC-exact in inhomogeneous turbulence.
    # The interior is the part the well-mixed drift is responsible for keeping
    # flat, so it's the right WMC acceptance criterion here.
    z_final = particles[:, 2].numpy()
    in_bl = (z_final >= 0.0) & (z_final <= blh)
    assert in_bl.mean() > 0.99, (
        f"only {in_bl.mean():.1%} of particles remain in the BL; reflection is "
        "leaking particles outside [0, blh]"
    )

    n_bins = 10
    edges = np.linspace(0.0, blh, n_bins + 1)
    counts, _ = np.histogram(z_final[in_bl], bins=edges)
    interior = counts[2:8]  # z ∈ [600, 2400] m
    expected = interior.mean()
    rel_rms = float(np.sqrt(np.mean(((interior - expected) / expected) ** 2)))

    # 15% relative-RMS tolerance. Shot noise at N_interior/bin ≈ 600 is ~4%; the
    # rest covers residual drift-OU coupling at our chosen Δt/τ.
    assert rel_rms < 0.15, (
        f"V1 well-mixed test failed: interior relative-RMS deviation from "
        f"uniform = {rel_rms:.3f} > 0.15. All bin counts (0..blh): {counts.tolist()}, "
        f"interior bins (600..2400 m): {interior.tolist()}, expected ≈ "
        f"{expected:.0f}. This indicates a WMC violation in the integrated "
        f"HannaScheme backward path (drift, reflection, or dt-bias)."
    )


def _wmc_density_met_window(
    *,
    blh_m: float = 5000.0,
    ustar_m_s: float = 0.4,
    p_top_hpa: float = 500.0,
    n_lev: int = 6,
    n_lat: int = 4,
    n_lon: int = 4,
) -> HourlyMetTensors:
    """Met window for the F2 density-weighted V1 test. Same as `_wmc_met_window`
    but with a steeper pressure profile (1000 → ``p_top_hpa`` over the height
    range), so ρ(z) varies enough that the ρ-weighted equilibrium is visibly
    different from a flat distribution."""

    shape = (n_lev, n_lat, n_lon)
    zeros = torch.zeros(shape, dtype=torch.float32)
    chans = {
        "u": zeros,
        "v": zeros,
        "w": zeros,
        "blh": torch.full(shape, blh_m),
        "sp": torch.full(shape, 101325.0),
        "t": torch.full(shape, 280.0),
        "ustar": torch.full(shape, ustar_m_s),
        "shf": torch.zeros(shape),
    }
    names = ("u", "v", "w", "blh", "sp", "t", "ustar", "shf")
    fields = torch.stack([chans[n] for n in names], dim=0)

    t0 = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)
    z_top = max(6000.0, 1.2 * blh_m)
    level = np.linspace(0.0, z_top, n_lev)
    metadata = MetFieldMetadata(
        lon=np.linspace(-2.0, 2.0, n_lon),
        lat=np.linspace(40.0, 50.0, n_lat),
        level=level,
        pressure_level_hpa=np.linspace(1000.0, p_top_hpa, n_lev),
        time_start=t0,
        time_end=t0 + timedelta(hours=1),
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
    height = (
        torch.as_tensor(level, dtype=torch.float32).view(n_lev, 1, 1).expand(shape).contiguous()
    )
    return HourlyMetTensors(
        hour_start=fields,
        hour_end=fields,
        metadata=metadata,
        channel_names=names,
        height_agl_m=height,
    )


def _density_at(
    z_m: np.ndarray, *, p_bot_hpa: float, p_top_hpa: float, z_top_m: float, t_kelvin: float
) -> np.ndarray:
    """Air density at height z [m], matching the linear-in-z pressure profile of
    `_wmc_density_met_window` (constant T → ρ ∝ p)."""

    p_hpa = p_bot_hpa + (p_top_hpa - p_bot_hpa) * (z_m / z_top_m)
    return (p_hpa * 100.0) / (287.05 * t_kelvin)


def test_v1_density_weighted_well_mixed_with_F2() -> None:
    """V1 well-mixed test with the F2 density-correction drift. Following the
    same logic as the constant-ρ V1, but with the steeper ρ(z) profile of
    `_wmc_density_met_window` (1000 → 500 hPa over 6000 m, ratio ≈ 0.5) so the
    ρ-weighted equilibrium distinguishes visibly from a flat distribution.

    The WMC predicts the stationary z-distribution is ρ(z)-weighted (Stohl &
    Thomson 1999 Fig 2; their CAPTEX runs show this correction is ~5–15% on
    surface concentrations). Initialise particles ρ(z)-weighted; assert they
    stay ρ-weighted. This test fails if the density term is missing — without
    it, the stationary distribution is flat, so a ρ-weighted initial state
    relaxes AWAY from the assertion."""

    from lpdm.gpu_engine import GPUEngine
    from lpdm.turbulence import HannaScheme

    rng = np.random.default_rng(202605302)
    torch.manual_seed(202605302)

    blh = 5000.0
    z_top_grid = 6000.0
    p_bot, p_top, T = 1000.0, 500.0, 280.0
    engine = GPUEngine(device="cpu")
    scheme = HannaScheme(meander_enabled=False)
    met = _wmc_density_met_window(blh_m=blh, ustar_m_s=0.4, p_top_hpa=p_top, n_lev=6)

    # Rejection-sample N particle altitudes from ρ(z) on [0, blh].
    n = 6000
    z_init: list[float] = []
    rho_max = float(
        _density_at(
            np.array([0.0]), p_bot_hpa=p_bot, p_top_hpa=p_top, z_top_m=z_top_grid, t_kelvin=T
        )[0]
    )
    while len(z_init) < n:
        z_try = rng.uniform(0.0, blh, size=n - len(z_init))
        rho_try = _density_at(
            z_try, p_bot_hpa=p_bot, p_top_hpa=p_top, z_top_m=z_top_grid, t_kelvin=T
        )
        accepted = z_try[rng.uniform(0.0, rho_max, size=z_try.shape) < rho_try]
        z_init.extend(accepted.tolist())
    z_init_arr = np.array(z_init[:n])

    particles = torch.zeros(n, 4, dtype=torch.float32)
    particles[:, 1] = 45.0
    particles[:, 2] = torch.from_numpy(z_init_arr.astype(np.float32))
    particles[:, 3] = 1.0 / n
    state = scheme.initialize_state(n, device=torch.device("cpu"), dtype=torch.float32)
    active = torch.ones(n, dtype=torch.bool)

    dt = 15.0
    n_steps = 1500
    for _ in range(n_steps):
        particles, state = scheme.step(
            particles,
            state,
            met,
            t_alpha=0.5,
            dt_seconds=dt,
            active_mask=active,
            engine=engine,
        )
        above = particles[:, 2] > blh
        if bool(above.any()):
            particles[above, 2] = 2.0 * blh - particles[above, 2]
            state["w_prime"][above] = -state["w_prime"][above]

    z_final = particles[:, 2].numpy()
    in_bl = (z_final >= 0.0) & (z_final <= blh)
    assert in_bl.mean() > 0.99

    n_bins = 10
    edges = np.linspace(0.0, blh, n_bins + 1)
    counts, _ = np.histogram(z_final[in_bl], bins=edges)
    bin_centres = 0.5 * (edges[:-1] + edges[1:])
    rho_at_centres = _density_at(
        bin_centres,
        p_bot_hpa=p_bot,
        p_top_hpa=p_top,
        z_top_m=z_top_grid,
        t_kelvin=T,
    )
    expected = counts.sum() * rho_at_centres / rho_at_centres.sum()
    # Compare interior bins (z = 1500..3500 m, indices 3..6) so surface-layer
    # and reflection-bias regions don't dominate the metric. The ρ-weighting
    # difference vs flat in this range is ~10% — well above shot noise.
    interior_counts = counts[3:7]
    interior_expected = expected[3:7]
    rel_dev = (interior_counts - interior_expected) / interior_expected
    rel_rms = float(np.sqrt(np.mean(rel_dev**2)))

    # Separately confirm a flat-distribution null assertion would have failed:
    # the same data vs a constant expected (i.e. assuming no density correction)
    # should be visibly worse than the ρ-weighted assertion, demonstrating the
    # test is actually sensitive to the F2 term.
    flat_expected = interior_counts.mean()
    flat_rel_rms = float(np.sqrt(np.mean(((interior_counts - flat_expected) / flat_expected) ** 2)))

    assert rel_rms < 0.15, (
        f"V1 density-weighted test failed: interior relative-RMS vs ρ(z) "
        f"expectation = {rel_rms:.3f} > 0.15. Bins: {counts.tolist()}, "
        f"expected ρ-weighted: {expected.round().astype(int).tolist()}. "
        f"Density-correction drift may be missing/wrong."
    )
    # Sanity: the test must be sensitive to ρ-weighting — flat-distribution
    # interpretation should have noticeably worse RMS than ρ-weighted.
    # (If they're equal, the test isn't actually probing F2.)
    assert flat_rel_rms > 0.5 * rel_rms or flat_rel_rms < 0.05, (
        "F2 sensitivity check failed: flat-distribution RMS is suspiciously "
        f"close to ρ-weighted RMS ({flat_rel_rms:.3f} vs {rel_rms:.3f}). The "
        "ρ gradient is too weak to make this test discriminating; widen the "
        "pressure range or deepen the BL."
    )


def test_met_cache_thrash_warning_fires_when_undersized(caplog) -> None:
    """Guard must warn when met_cache_max_hours is below the cross-batch reuse threshold
    (≈ batch met span + advance) — the LRU-thrash footgun (set cache == batch span → zero
    benefit) that cost a GH200 run — and stay quiet when the cache is large enough."""

    from types import SimpleNamespace

    from lpdm.main import _warn_if_met_cache_thrashes

    base = datetime(2024, 1, 1, tzinfo=UTC)

    def _rel(secs: int) -> SimpleNamespace:
        return SimpleNamespace(release_time=base + timedelta(seconds=secs), duration_seconds=3600)

    # Two consecutive days of hourly releases (24/batch), 5-day backward window:
    # batch span ≈ 143 h, advance ≈ 24 h → reuse threshold ≈ 167 h.
    b0 = SimpleNamespace(releases=[_rel(h * 3600) for h in range(24)])
    b1 = SimpleNamespace(releases=[_rel(86400 + h * 3600) for h in range(24)])
    sim_len = 5 * 86400.0

    caplog.clear()
    with caplog.at_level(logging.WARNING):
        _warn_if_met_cache_thrashes(_make_run_config(met_cache_max_hours=144), [b0, b1], sim_len)
    assert "below the cross-batch reuse threshold" in caplog.text  # 144 < 167 → warn

    caplog.clear()
    with caplog.at_level(logging.WARNING):
        _warn_if_met_cache_thrashes(_make_run_config(met_cache_max_hours=200), [b0, b1], sim_len)
    assert "cross-batch reuse threshold" not in caplog.text  # 200 ≥ 167 → quiet


def test_met_prefetch_matches_synchronous_footprints(tmp_path: Path) -> None:
    """Background met prefetch must be bit-identical to the synchronous fetch path.

    Prefetch only changes WHEN met I/O happens (a background thread loads the next backward
    hour while the current window is stepped) — never the fetched data, the cache contents,
    the compute order, or the RNG draws. So the footprints must match exactly. The run spans
    several met-hour boundaries (3 h backward at dt=300 s, cache=2 h so it both evicts and
    re-fetches) to actually exercise the prefetch schedule / pending-reuse / eviction paths.
    """

    def run(prefetch: bool, sub: str):
        cfg = _make_run_config(
            simulation_length_seconds=10800,
            dt_seconds=300,
            n_particles=512,
            release_seed=42,
            met_cache_max_hours=2,
            met_prefetch=prefetch,
            n_time_bins=3,
            output_uri=str(tmp_path / sub),
        )
        reader = _make_reader(cfg, wind_fn=lambda _: (6.0, -3.0, 0.0))
        torch.manual_seed(0)
        _run(cfg, reader=reader)
        return xr.open_zarr(tmp_path / sub / "footprints.zarr")["footprint"].values

    fp_sync = run(False, "sync")
    fp_async = run(True, "async")
    assert np.array_equal(fp_sync, fp_async)


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
    expected_total = (
        float((traj["active_particles"].astype(float) * cfg.simulation.dt_seconds).sum())
        / cfg.release.n_particles
    )

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
    start_time = start_time or datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)
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
                "density_weighting": "none",
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
    latest_end = max(r.release_time + timedelta(seconds=r.duration_seconds) for r in all_releases)
    return AnalyticMetReader(
        coverage_start=earliest - timedelta(seconds=cfg.simulation.length_seconds + 7200),
        coverage_end=latest_end + timedelta(hours=1),
        wind_fn=wind_fn,
    )


def _make_multi_site_config(
    *,
    output_uri: str,
    n_releases: int = 2,
    period_seconds: int = 3600,
    duration_seconds: int = 300,
    n_particles_per_release: int = 128,
    simulation_length_seconds: int = 3600,
    max_releases_per_batch: int = 24,
    start_time: datetime | None = None,
    seed: int | None = 42,
) -> RunConfig:
    start_time = start_time or datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)
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
                "kind": "multi_point_periodic",
                "sites": [
                    {"name": "MHD", "lon": -1.0, "lat": 0.0, "alt_agl_m": 100.0},
                    {"name": "RGL", "lon": 1.0, "lat": 0.5, "alt_agl_m": 300.0},
                ],
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
                "density_weighting": "none",
            },
            "met_domain": {
                "lon_bounds": (-3.0, 3.0),
                "lat_bounds": (-3.0, 3.0),
                "alt_max_m": 10000.0,
            },
            "memory": {"met_cache_max_hours": 4, "log_every_steps": 0, "gc_every_steps": 0},
            "batch": {"max_releases_per_batch": max_releases_per_batch},
        }
    )


def test_multi_site_expand_is_time_major_with_per_site_locations() -> None:
    """multi_point_periodic expands to one release per (time, site), TIME-MAJOR (all sites
    per time), each carrying its own location + label and a distinct seed."""

    cfg = _make_multi_site_config(output_uri="x", n_releases=2)
    rels = [r for b in cfg.expand_to_batches() for r in b.releases]

    assert len(rels) == 4  # 2 times × 2 sites
    assert [r.release_idx for r in rels] == [0, 1, 2, 3]
    assert [r.label for r in rels] == ["MHD", "RGL", "MHD", "RGL"]  # time-major
    assert [r.lon for r in rels] == [-1.0, 1.0, -1.0, 1.0]
    assert [r.alt_agl_m for r in rels] == [100.0, 300.0, 100.0, 300.0]
    assert rels[0].release_time == rels[1].release_time  # both sites share t0
    assert rels[2].release_time == rels[0].release_time + timedelta(seconds=3600)
    assert len({r.seed for r in rels}) == 4  # distinct per (site,time)


def test_multi_site_run_emits_per_site_footprint_coords(tmp_path: Path) -> None:
    """End-to-end multi-site run: the footprint's `release` axis carries per-site location
    + `site` coords, both sites accumulate mass, and it unstacks to a (site, release_time)
    cube."""

    cfg = _make_multi_site_config(
        output_uri=str(tmp_path / "out"), n_releases=2, n_particles_per_release=128
    )
    reader = _make_multi_release_reader(cfg, wind_fn=lambda _: (3.0, 0.0, 0.0))
    metadata = _run(cfg, reader=reader)

    assert metadata["schedule"]["n_releases"] == 4
    fp = xr.open_zarr(tmp_path / "out" / "footprints.zarr").load()
    assert fp["footprint"].sizes["release"] == 4
    assert list(fp["site"].values) == ["MHD", "RGL", "MHD", "RGL"]
    assert np.allclose(fp["release_lon"].values, [-1.0, 1.0, -1.0, 1.0])
    assert np.allclose(fp["release_lat"].values, [0.0, 0.5, 0.0, 0.5])

    total_per_release = fp["footprint"].sum(["time_ago", "z_bin", "latitude", "longitude"]).values
    site_vals = fp["site"].values
    assert total_per_release[site_vals == "MHD"].sum() > 0.0
    assert total_per_release[site_vals == "RGL"].sum() > 0.0

    cube = fp["footprint"].set_index(release=["site", "release_time"]).unstack("release")
    assert set(cube["site"].values) == {"MHD", "RGL"}
    assert cube.sizes["release_time"] == 2


def test_multi_site_footprint_rows_stay_with_their_named_receptor(tmp_path: Path) -> None:
    """Each written footprint row belongs to its labelled site (2026-11-25 test review).

    Still air and no turbulence: every particle stays at its release point, so all of
    a release's residence lies in the output cell containing its site. The sites sit
    at the centres of different 0.25° cells, so a permutation of footprint rows under
    the original labels (e.g. a reversed batch-local release index in the gridder)
    puts a site's residence in the other site's cell.
    """

    sites = {
        "MHD": {"lon": -0.875, "lat": 0.125, "alt_agl_m": 100.0},
        "RGL": {"lon": 1.125, "lat": -0.625, "alt_agl_m": 300.0},
    }
    raw = _make_multi_site_config(
        output_uri=str(tmp_path / "out"), n_releases=2, n_particles_per_release=64
    ).model_dump(mode="json")
    raw["release"]["sites"] = [{"name": k, **v} for k, v in sites.items()]
    cfg = RunConfig.model_validate(raw)
    reader = _make_multi_release_reader(cfg, wind_fn=lambda _: (0.0, 0.0, 0.0))
    scheme = PlaceholderConstantOU()
    scheme.sigma_w2 = 0.0  # no vertical turbulence; the placeholder has no horizontal
    _run(cfg, reader=reader, scheme=scheme)

    fp = xr.open_zarr(tmp_path / "out" / "footprints.zarr").load()
    assert list(fp["site"].values) == ["MHD", "RGL", "MHD", "RGL"]
    lat_c, lon_c = fp["latitude"].values, fp["longitude"].values
    per_cell = (
        fp["footprint"].sum(["time_ago", "z_bin"]).transpose("release", "latitude", "longitude")
    )
    for r, name in enumerate(fp["site"].values):
        iy = int(np.argmin(np.abs(lat_c - sites[name]["lat"])))
        ix = int(np.argmin(np.abs(lon_c - sites[name]["lon"])))
        assert abs(lat_c[iy] - sites[name]["lat"]) < 1e-9  # the site is a cell centre
        assert abs(lon_c[ix] - sites[name]["lon"]) < 1e-9
        cells = per_cell.isel(release=r).values
        own = float(cells[iy, ix])
        assert own > 0.0, f"release {r} ({name}): no residence in its own cell"
        assert own == pytest.approx(float(cells.sum()), rel=1e-12), (
            f"release {r} ({name}): residence outside its own cell"
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
        "release",
        "time_ago",
        "z_bin",
        "latitude",
        "longitude",
    )
    assert fp["footprint"].sizes["release"] == 3
    # release_time is now a coord along `release` (so a timestamp can repeat across sites)
    assert fp["release_time"].dims == ("release",)

    expected_times = [cfg.simulation.start_time + timedelta(seconds=i * 3600) for i in range(3)]
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
    expected_total = (
        float((traj["active_particles"].astype(float) * cfg.simulation.dt_seconds).sum())
        / cfg.release.n_particles_per_release
    )

    fp = xr.open_zarr(tmp_path / "out" / "footprints.zarr")["footprint"]
    actual_total = float(fp.sum())

    # 2e-3 — loosened from 1e-3 after F4 Tier 2 substepping (audit 2026-05-30):
    # the per-particle substep loop introduces additional float32 arithmetic
    # (multiple OU draws + displacement accumulations per outer step), so the
    # 5D scatter_add now sees marginally more rounding noise. The looser bound
    # still catches real conservation bugs (~5×float32 effective resolution at
    # 10500 particles) and is robust to the test-order RNG bleed already
    # documented for the single-release equivalent.
    assert abs(actual_total - expected_total) / expected_total < 2e-3


def test_periodic_release_per_release_mass_near_expected(tmp_path: Path) -> None:
    """Constant wind ⇒ each release accumulates ≈ sim_length per particle.

    Totals fall a little short because the placeholder scheme's σw carries some
    particles above the footprint grid's top; the 5% tolerance allows for that.
    The exact still-air check is `test_every_particle_followed_for_full_simulation_length`.
    """

    cfg = _make_periodic_config(
        output_uri=str(tmp_path / "out"),
        n_releases=3,
        period_seconds=3600,
        duration_seconds=300,
        n_particles_per_release=256,
        simulation_length_seconds=10800,
        max_releases_per_batch=24,
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

    Totals are not bit-identical across chunkings, because particles in different
    batches draw different turbulence, so the invariant is conservation against
    the diagnostic. That every release is followed for the full length whatever
    its batch is checked in `test_every_particle_followed_for_full_simulation_length`.
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
        expected = (
            float((traj["active_particles"].astype(float) * cfg.simulation.dt_seconds).sum())
            / cfg.release.n_particles_per_release
        )

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
    # Totals differ only through the turbulence draws; sanity-bound to a few percent.
    assert abs(total_one - total_two) / total_one < 5e-2


# ---- VALIDATION-HANDOFF findings 1 and 2: release lifetime and the last window ----


@pytest.mark.parametrize("static", [False, True], ids=["dynamic", "static"])
@pytest.mark.parametrize("n_time_bins", [1, 3])
@pytest.mark.parametrize("max_per_batch", [1, 2])
def test_every_particle_followed_for_full_simulation_length(
    tmp_path: Path, max_per_batch: int, n_time_bins: int, static: bool, monkeypatch
) -> None:
    """Each particle is followed for simulation.length_seconds from its own release
    time, so with hour-long release windows every release accumulates the full
    length per particle, whatever releases share its batch.

    Before the fix a batch stopped one length before its earliest window's END,
    so particles released early in that window lost up to an hour: 8,735 s of
    10,800 s for the first release, and the second's total depended on batching.

    With hourly time-ago bins the axis must reach the length plus the release
    window, as ages count from the window's end: 3 bins for 3 h become 4, and the
    hourly totals sum to the same 10,800 s.
    """

    cfg = _make_periodic_config(
        output_uri=str(tmp_path / "out"),
        n_releases=2,
        period_seconds=3600,
        duration_seconds=3600,
        n_particles_per_release=128,
        simulation_length_seconds=10800,
        max_releases_per_batch=max_per_batch,
    )
    # Still air, as in the review's probe: no particle leaves the footprint grid, so
    # a release's total is exactly the time its particles were followed, and every
    # resolved age stays on the time_ago axis.
    monkeypatch.setattr(PlaceholderConstantOU, "sigma_w2", 0.0)
    monkeypatch.setenv("GLIDE_STATIC_SUBSTEPS", "1" if static else "0")
    cfg = cfg.model_copy(
        update={"output_grid": cfg.output_grid.model_copy(update={"n_time_bins": n_time_bins})}
    )
    reader = _make_multi_release_reader(cfg, wind_fn=lambda _: (0.0, 0.0, 0.0))

    _run(cfg, reader=reader)

    fp = xr.open_zarr(tmp_path / "out" / "footprints.zarr")["footprint"]
    per_release = fp.sum(dim=("time_ago", "z_bin", "latitude", "longitude")).values
    np.testing.assert_allclose(per_release, cfg.simulation.length_seconds, rtol=1e-3)
    assert fp.sizes["time_ago"] == (1 if n_time_bins == 1 else 4)


@dataclass
class _CoverageCheckingReader(AnalyticMetReader):
    """Refuses, as a real store does, a window that reaches beyond its coverage."""

    fetched_starts: list[datetime] = field(default_factory=list)

    def fetch_hourly_window(self, request: BoundingBoxRequest) -> HourlyMetTensors:
        if request.time.start < self.coverage_start or request.time.end > self.coverage_end:
            raise KeyError(f"window {request.time.start}..{request.time.end} outside the store")
        self.fetched_starts.append(request.time.start)
        return super().fetch_hourly_window(request)


@pytest.mark.parametrize("static", [False, True], ids=["dynamic", "static"])
@pytest.mark.parametrize("interval_seconds", [3600, 10800])
def test_met_ending_exactly_at_release_end_is_enough(
    tmp_path: Path, interval_seconds: int, static: bool, monkeypatch
) -> None:
    """A store whose last time is the release end covers the run: the last backward
    step ends there and lies in the window before it. Before the fix the preflight
    asked for one more field, and the step ending on a met time read the window
    starting there (held means and convection from the following period)."""

    cfg = _make_run_config(
        release_duration_seconds=interval_seconds,
        simulation_length_seconds=21600,
        n_time_bins=1,
        dt_seconds=300,
        n_particles=64,
        release_seed=42,
        output_uri=str(tmp_path / "out"),
    )
    monkeypatch.setenv("GLIDE_STATIC_SUBSTEPS", "1" if static else "0")
    release_end = cfg.simulation.start_time + timedelta(seconds=interval_seconds)
    reader = _CoverageCheckingReader(
        coverage_start=cfg.simulation.start_time
        - timedelta(seconds=cfg.simulation.length_seconds + interval_seconds),
        coverage_end=release_end,
        wind_fn=lambda _: (5.0, 0.0, 0.0),
        interval_seconds=interval_seconds,
    )

    metadata = _run(cfg, reader=reader)

    assert metadata["runtime"]["status"] == "completed"
    assert max(reader.fetched_starts) == release_end - timedelta(seconds=interval_seconds)


def test_met_ending_before_release_end_still_rejected(tmp_path: Path) -> None:
    cfg = _make_run_config(
        release_duration_seconds=3600,
        simulation_length_seconds=10800,
        output_uri=str(tmp_path / "out"),
    )
    reader = AnalyticMetReader(
        coverage_start=cfg.simulation.start_time - timedelta(hours=5),
        coverage_end=cfg.simulation.start_time + timedelta(minutes=30),
        wind_fn=lambda _: (0.0, 0.0, 0.0),
    )
    with pytest.raises(PreflightValidationError):
        _run(cfg, reader=reader)


def test_step_window_time_is_the_step_midpoint() -> None:
    from lpdm.main import _step_window_time

    t = datetime(2024, 1, 1, 13, 0, tzinfo=UTC)
    assert _step_window_time(t, 300.0) == datetime(2024, 1, 1, 12, 57, 30, tzinfo=UTC)


def test_backward_step_takes_the_short_step_first() -> None:
    from lpdm.main import _backward_step_seconds

    assert _backward_step_seconds(3660.0, 300.0) == pytest.approx(60.0)
    assert _backward_step_seconds(3600.0, 300.0) == 300.0
    assert _backward_step_seconds(120.0, 300.0) == pytest.approx(120.0)
    assert _backward_step_seconds(300.0000001, 300.0) == 300.0


def test_time_ago_axis_covers_release_window_in_hourly_bins(tmp_path: Path) -> None:
    """n_time_bins hours of travel plus the longest release window, labelled in
    hours; a single bin holds every age and says so."""

    for n_bins, want_len, want_end in ((3, 5, 5.0), (2, 4, 4.0), (1, 1, 3.0 + 1.5)):
        cfg = _make_periodic_config(
            output_uri=str(tmp_path / f"out{n_bins}"),
            n_releases=2,
            duration_seconds=5400,
            period_seconds=7200,
            n_particles_per_release=16,
            simulation_length_seconds=10800,
        )
        cfg = cfg.model_copy(
            update={"output_grid": cfg.output_grid.model_copy(update={"n_time_bins": n_bins})}
        )
        _run(cfg, reader=_make_multi_release_reader(cfg, wind_fn=lambda _: (0.0, 0.0, 0.0)))
        fp = xr.open_zarr(tmp_path / f"out{n_bins}" / "footprints.zarr")
        assert fp.sizes["time_ago"] == want_len
        assert float(fp["time_ago_start_hours"][0]) == 0.0
        assert float(fp["time_ago_end_hours"][-1]) == want_end
        if n_bins > 1:
            widths = fp["time_ago_end_hours"].values - fp["time_ago_start_hours"].values
            np.testing.assert_array_equal(widths, 1.0)


_HELD_T0 = datetime(2024, 1, 1, 0, 0, tzinfo=UTC)


@dataclass
class _HeldUstarReader(AnalyticMetReader):
    """u* as a period mean (decision 0020) whose value in each window is the hour
    its averaging period ends, counted from `_HELD_T0`, so the value a step uses
    says which period it took its mean from."""

    channel_names: tuple[str, ...] = ("u", "v", "w", "blh", "sp", "ustar")

    def fetch_hourly_window(self, request: BoundingBoxRequest) -> HourlyMetTensors:
        window = super().fetch_hourly_window(request)
        k = window.channel_names.index("ustar")
        start, end = window.hour_start.clone(), window.hour_end.clone()
        start[k] = (request.time.start - _HELD_T0).total_seconds() / 3600.0
        end[k] = (request.time.end - _HELD_T0).total_seconds() / 3600.0
        return dataclasses.replace(
            window, hour_start=start, hour_end=end, period_channels=frozenset({"ustar"})
        )


@pytest.mark.parametrize("static", [False, True], ids=["dynamic", "static"])
@pytest.mark.parametrize("interval_seconds", [3600, 10800])
def test_held_mean_used_by_a_step_is_the_mean_over_that_step(
    tmp_path: Path, interval_seconds: int, static: bool, monkeypatch
) -> None:
    """Every step must use the period mean whose averaging period contains it.

    Before the fix a step ending exactly on a met time read the window starting
    there, and so held the FOLLOWING period's mean for that step (one step in
    twelve at hourly met, 300 s steps).
    """

    import lpdm.main as glide_main

    monkeypatch.setenv("GLIDE_STATIC_SUBSTEPS", "1" if static else "0")
    current_step: dict[str, datetime] = {}
    used: list[tuple[datetime, datetime, float]] = []

    real_window_time = glide_main._step_window_time

    def window_time(t_cursor: datetime, delta_s: float) -> datetime:
        current_step["end"] = t_cursor
        current_step["start"] = t_cursor - timedelta(seconds=delta_s)
        return real_window_time(t_cursor, delta_s)

    real_step = PlaceholderConstantOU.step

    def step(self, particles, state, met_window, t_alpha, dt_seconds, active_mask, engine):
        s, e = met_window.channel("ustar")
        w = met_window.time_weight("ustar", float(t_alpha))
        held = float(s[0, 0, 0] * (1.0 - w) + e[0, 0, 0] * w)
        used.append((current_step["start"], current_step["end"], held))
        return real_step(
            self, particles, state, met_window, t_alpha, dt_seconds, active_mask, engine
        )

    monkeypatch.setattr(glide_main, "_step_window_time", window_time)
    monkeypatch.setattr(PlaceholderConstantOU, "step", step)

    cfg = _make_run_config(
        release_duration_seconds=interval_seconds,
        simulation_length_seconds=21600,
        n_time_bins=1,
        dt_seconds=300,
        n_particles=32,
        release_seed=42,
        output_uri=str(tmp_path / "out"),
    )
    reader = _HeldUstarReader(
        coverage_start=cfg.simulation.start_time - timedelta(hours=12),
        coverage_end=cfg.simulation.start_time + timedelta(hours=6),
        wind_fn=lambda _: (0.0, 0.0, 0.0),
        interval_seconds=interval_seconds,
    )

    _run(cfg, reader=reader)

    assert len(used) >= 21600 // 300
    interval = timedelta(seconds=interval_seconds)
    for step_start, step_end, held in used:
        period_end = _HELD_T0 + timedelta(hours=held)
        assert period_end - interval <= step_start and step_end <= period_end, (
            f"step {step_start:%H:%M}-{step_end:%H:%M} used the mean over the period "
            f"ending {period_end:%H:%M}"
        )


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
            [0.0, 0.0, 100.0, 1.0],  # inside
            [-3.0, 0.0, 100.0, 1.0],  # on lon edge -> inside (inclusive)
            [-3.01, 0.0, 100.0, 1.0],  # just past lon_min -> outside
            [3.01, 0.0, 100.0, 1.0],  # past lon_max -> outside
            [0.0, 2.5, 100.0, 1.0],  # past lat_max -> outside
            [0.0, 0.0, 5000.1, 1.0],  # above alt_max -> outside
            [0.0, 0.0, 0.0, 1.0],  # at the surface -> inside (no lower kill)
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
        simulation_length_seconds=10800,  # 3 h backward
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


# ---- Phase 2: static-shape (full-set, mask-gated) per-step path ----------------
# (docs/architecture.md §5). The dynamic path stays the CPU default; these tests force
# the static path (scheme ctor flag / GLIDE_STATIC_SUBSTEPS env) and check that it
# (a) freezes inactive particles exactly and (b) conserves footprint mass.


def test_static_step_freezes_inactive_particles() -> None:
    """The static (full-set) HannaScheme.step must leave inactive particles'
    positions AND turbulence/meander state byte-for-byte unchanged, advancing only
    the active ones (the `torch.where` write-back + `sub_dt=0` substep gating)."""

    from lpdm.gpu_engine import GPUEngine
    from lpdm.turbulence import HannaScheme

    torch.manual_seed(7)
    engine = GPUEngine(device="cpu")
    # meander on so the meander state-gating is exercised too.
    scheme = HannaScheme(meander_enabled=True, static_substeps=True)
    met = _wmc_met_window(blh_m=2000.0, ustar_m_s=0.4)

    n = 200
    particles = torch.zeros(n, 4, dtype=torch.float32)
    particles[:, 1] = 45.0
    particles[:, 2] = torch.rand(n) * 1500.0 + 100.0
    particles[:, 3] = 1.0
    state = scheme.initialize_state(n, device=torch.device("cpu"), dtype=torch.float32)
    # Seed nonzero state so "unchanged" is a real assertion (not 0 == 0).
    for k in state:
        state[k] = torch.randn(state[k].shape) * 0.1
    state["sigma_last"] = state["sigma_last"].abs() + 0.1  # a σ: positive

    active = torch.zeros(n, dtype=torch.bool)
    active[: n // 2] = True  # first half active, second half frozen

    p0 = particles.clone()
    s0 = {k: v.clone() for k, v in state.items()}

    particles, state = scheme.step(
        particles,
        state,
        met,
        t_alpha=0.5,
        dt_seconds=60.0,
        active_mask=active,
        engine=engine,
    )

    inactive = ~active
    assert torch.equal(particles[inactive], p0[inactive]), "inactive positions changed"
    for k in s0:
        assert torch.equal(state[k][inactive], s0[k][inactive]), f"inactive state {k} changed"
    # Active particles actually moved vertically (turbulence applied).
    assert not torch.equal(particles[active, 2], p0[active, 2])
    assert torch.isfinite(particles).all()


def test_static_path_footprint_conservation(tmp_path: Path, monkeypatch) -> None:
    """End-to-end conservation on the STATIC per-step runtime path
    (GLIDE_STATIC_SUBSTEPS=1): footprint sum = Σ(active_count)·dt / n_particles —
    the same identity as the dynamic path. Validates full-set advection gating, the
    `active_count` desync (active_particles diagnostic via `active_mask.sum()`), and
    mask-gated accumulation in `main._run`, plus HannaScheme.step's full-set path."""

    monkeypatch.setenv("GLIDE_STATIC_SUBSTEPS", "1")
    cfg = _make_run_config(
        release_duration_seconds=60,
        simulation_length_seconds=3600,
        dt_seconds=300,
        n_particles=512,
        release_seed=42,
        output_uri=str(tmp_path / "out"),
        n_time_bins=1,
        turbulence_scheme="hanna_1982",
        # Exercise the per-step memory-log block on the static path (it references
        # the active count, which only the dynamic branch binds as `active_count`).
        memory_log_every_steps=1,
    )
    reader = _make_hanna_reader(cfg, wind_fn=lambda _: (5.0, 0.0, 0.0))

    _run(cfg, reader=reader)

    traj = pd.read_parquet(tmp_path / "out" / "trajectory_diagnostics.parquet")
    expected_total = (
        float((traj["active_particles"].astype(float) * cfg.simulation.dt_seconds).sum())
        / cfg.release.n_particles
    )

    fp = xr.open_zarr(tmp_path / "out" / "footprints.zarr")
    actual_total = float(fp["footprint"].sum())

    assert abs(actual_total - expected_total) / expected_total < 1e-3


def test_graph_compile_substep_path_runs_on_cpu(monkeypatch) -> None:
    """Phase 3 smoke: force the static + graph-compile path (GLIDE_STATIC_SUBSTEPS=1,
    GLIDE_COMPILE=1) and confirm the fixed-count substep loop actually compiles +
    runs + stays finite, with the per-method engine compile suppressed (no nesting).
    CUDA graphs are CUDA-only, so on CPU this checks the loop is *traceable* — the
    sm% / capture payoff is validated on the GH200, not here. n_substeps=2 keeps
    the unrolled loop small so the one-time compile is quick."""

    monkeypatch.setenv("GLIDE_STATIC_SUBSTEPS", "1")
    monkeypatch.setenv("GLIDE_COMPILE", "1")

    from lpdm.gpu_engine import GPUEngine
    from lpdm.turbulence import HannaScheme

    torch.manual_seed(3)
    engine = GPUEngine(device="cpu")
    assert engine._compile_hot_paths is False  # per-method compile off on the graph path
    scheme = HannaScheme(n_substeps=2)
    met = _wmc_met_window(blh_m=1500.0, ustar_m_s=0.4)

    n = 32
    particles = torch.zeros(n, 4, dtype=torch.float32)
    particles[:, 1] = 45.0
    particles[:, 2] = torch.rand(n) * 1000.0 + 100.0
    particles[:, 3] = 1.0
    state = scheme.initialize_state(n, device=torch.device("cpu"), dtype=torch.float32)
    active = torch.ones(n, dtype=torch.bool)

    particles, state = scheme.step(
        particles,
        state,
        met,
        t_alpha=0.5,
        dt_seconds=60.0,
        active_mask=active,
        engine=engine,
    )

    assert scheme._graph_compile_state == "ok"  # the graph-compile path engaged
    assert torch.isfinite(particles).all()
    assert (particles[:, 2] >= 0.0).all()  # reflection kept z >= 0


def test_step_profiler_captures_window_and_exits(tmp_path: Path, monkeypatch) -> None:
    """GLIDE_PROFILE captures a window of cursor-loop steps, writes a Chrome trace, and
    exits the run early (a profiling pass doesn't need the full output)."""

    trace = tmp_path / "prof.json"
    monkeypatch.setenv("GLIDE_PROFILE", "1")
    monkeypatch.setenv("GLIDE_PROFILE_WARMUP", "1")
    monkeypatch.setenv("GLIDE_PROFILE_STEPS", "2")
    monkeypatch.setenv("GLIDE_PROFILE_TRACE", str(trace))

    cfg = _make_run_config(turbulence_scheme="hanna_1982", output_uri=str(tmp_path / "out"))
    reader = _make_hanna_reader(cfg, wind_fn=lambda _: (5.0, 0.0, 0.0))

    with pytest.raises(SystemExit):
        _run(cfg, reader=reader)

    assert trace.exists() and trace.stat().st_size > 0  # trace was written before exit


_CAPTURE_CASES = pytest.mark.parametrize(
    "frame_kind, wind_dtype",
    [("degrees", torch.float32), ("local_metric", torch.float32), ("local_metric", torch.bfloat16)],
    ids=["degrees", "metric", "metric-bf16"],
)


def _capture_engine(frame_kind: str, wind_dtype: torch.dtype):
    """Engine for the capture guards: the degrees frame and float32 winds (the
    pre-A4 arithmetic), and the runtime's local metric frame with float32 or
    bfloat16 winds (task A4), so the guards hold in every mode."""

    from lpdm.gpu_engine import GPUEngine
    from lpdm.local_frame import DEGREES, HorizontalFrame

    frame = (
        DEGREES
        if frame_kind == "degrees"
        else HorizontalFrame.local_metric((-98.0, 39.5), (10.75, 79.0))
    )
    return GPUEngine(device="cpu", frame=frame, wind_gather_dtype=wind_dtype)


@_CAPTURE_CASES
def test_step_core_traces_as_one_graph_no_breaks(frame_kind: str, wind_dtype: torch.dtype) -> None:
    """The whole per-step core must trace as ONE graph (no graph breaks) so it captures
    as a single CUDA graph on GPU — otherwise the eager glue between sub-graphs keeps the
    run launch-bound (the phase-4 failure mode). `fullgraph=True` + `backend="eager"`
    surfaces any data-dependent break (a tensor `.item()`/`bool()`/Python control flow on
    tensor data) as a hard error here on CPU — no GPU and no C++ codegen needed. Guards
    against regressions like the F9 `ascending = bool(level_arr[...])` break (fixed
    2026-06-26)."""

    import torch._dynamo

    from lpdm.turbulence import HannaScheme

    torch._dynamo.reset()
    prev = torch._dynamo.config.suppress_errors
    torch._dynamo.config.suppress_errors = False  # make breaks raise, don't swallow
    try:
        engine = _capture_engine(frame_kind, wind_dtype)
        scheme = HannaScheme(meander_enabled=True, n_substeps=4)  # meander on → trace that path too
        met = _wmc_met_window(blh_m=2000.0, ustar_m_s=0.4)

        n = 48
        particles = torch.zeros(n, 4, dtype=torch.float32)
        particles[:, 1] = 45.0
        particles[:, 2] = torch.rand(n) * 1500.0 + 100.0
        particles[:, 3] = 1.0
        particles = engine.frame.to_frame(particles)
        state = scheme.initialize_state(n, device=torch.device("cpu"), dtype=torch.float32)
        active = torch.ones(n, dtype=torch.bool)
        inputs = scheme._gather_static_inputs(
            met,
            0.5,
            torch.device("cpu"),
            torch.float32,
            frame=engine.frame,
            wind_dtype=engine.wind_dtype,
        )

        compiled = torch.compile(scheme._step_core, fullgraph=True, backend="eager")
        out = compiled(
            particles=particles,
            u_prime=state["u_prime"],
            v_prime=state["v_prime"],
            w_prime=state["w_prime"],
            sigma_last=state["sigma_last"],
            u_meander=state["u_meander"],
            v_meander=state["v_meander"],
            active_mask=active,
            dt_seconds=60.0,
            engine=engine,
            **inputs,
        )
        # State (7, with sigma_last) + this step's sub-step exceedance counts and worst ratios.
        assert len(out) == 9 and all(torch.isfinite(o).all() for o in out)
    finally:
        torch._dynamo.config.suppress_errors = prev
        torch._dynamo.reset()


@_CAPTURE_CASES
def test_step_core_does_not_recompile_per_step(frame_kind: str, wind_dtype: torch.dtype) -> None:
    """The compiled core must NOT recompile as `alpha` and the met-field VALUES change
    step to step (only shapes/grid are constant). Passing `alpha` (the RK2-midpoint
    weight, different every step) as a Python float would specialize the graph per value
    → recompile every step → blow the recompile limit → eager fallback (GH200 2026-06-26:
    4× *slower*, GPU-busy 0.7%). `error_on_recompile` turns a recompile into a hard error
    here on CPU — no GPU needed. The fix is passing `alpha` as a 0-d tensor."""

    import torch._dynamo

    from lpdm.turbulence import HannaScheme

    torch._dynamo.reset()
    prev_sup = torch._dynamo.config.suppress_errors
    prev_rec = torch._dynamo.config.error_on_recompile
    torch._dynamo.config.suppress_errors = False
    torch._dynamo.config.error_on_recompile = True
    try:
        engine = _capture_engine(frame_kind, wind_dtype)
        scheme = HannaScheme(meander_enabled=True, n_substeps=4)
        n = 48
        compiled = torch.compile(scheme._step_core, fullgraph=True, backend="eager")

        def call(met, t_alpha, seed):
            torch.manual_seed(seed)
            particles = torch.zeros(n, 4, dtype=torch.float32)
            particles[:, 1] = 45.0
            particles[:, 2] = torch.rand(n) * 1500.0 + 100.0
            particles[:, 3] = 1.0
            particles = engine.frame.to_frame(particles)
            state = scheme.initialize_state(n, device=torch.device("cpu"), dtype=torch.float32)
            active = torch.ones(n, dtype=torch.bool)
            inputs = scheme._gather_static_inputs(
                met,
                t_alpha,
                torch.device("cpu"),
                torch.float32,
                frame=engine.frame,
                wind_dtype=engine.wind_dtype,
            )
            compiled(
                particles=particles,
                u_prime=state["u_prime"],
                v_prime=state["v_prime"],
                w_prime=state["w_prime"],
                sigma_last=state["sigma_last"],
                u_meander=state["u_meander"],
                v_meander=state["v_meander"],
                active_mask=active,
                dt_seconds=60.0,
                engine=engine,
                **inputs,
            )

        # Same grid SHAPE (so any recompile is a genuine value-specialisation bug), but
        # different alpha, different met-field values, AND a different per-level AGL array
        # (`_wmc_met_window` sets level = linspace(0, max(4000, 1.5·blh_m), n_lev), so
        # blh_m=4000 → 6000 m top vs 2000 → 4000 m). The level VALUES change every met
        # window in production (weather-dependent geopotential); they must be a dynamic
        # tensor, not a tuple dynamo specialises on (GH200 month run 2026-06-26).
        met_a = _wmc_met_window(blh_m=2000.0, ustar_m_s=0.4)  # level top 4000 m
        met_b = _wmc_met_window(blh_m=4000.0, ustar_m_s=0.6)  # level top 6000 m (different F9 axis)
        call(met_a, 0.3, 1)  # first call: compiles once (not a recompile)
        call(met_a, 0.7, 2)  # different alpha → must NOT recompile
        call(met_b, 0.5, 3)  # different alpha + met values + LEVEL array → must NOT recompile
    finally:
        torch._dynamo.config.suppress_errors = prev_sup
        torch._dynamo.config.error_on_recompile = prev_rec
        torch._dynamo.reset()


def test_wind_mean_cache_matches_direct_and_invalidates_across_windows() -> None:
    """The per-window wind-mean cache (perf review 2026-07-16 #2) must be
    invisible: cached, cache-free, and direct-formula values agree exactly
    (mean and time-lerp commute — both linear), and a new met window invalidates
    the single-slot cache."""

    import dataclasses
    from datetime import timedelta

    from lpdm.main import _advection_alpha_wind_mean

    win1 = _varying_wind_met_window()
    device, dtp = torch.device("cpu"), torch.float64
    t_cur = win1.metadata.time_start + timedelta(minutes=20)

    cache: dict = {}
    a1, wm1 = _advection_alpha_wind_mean(win1, t_cur, 60.0, device, dtp, cache)
    _, wm2 = _advection_alpha_wind_mean(win1, t_cur, 60.0, device, dtp, cache)  # cache hit
    _, wm3 = _advection_alpha_wind_mean(win1, t_cur, 60.0, device, dtp, None)  # cache-free
    assert torch.equal(wm1, wm2)
    assert torch.allclose(wm1, wm3)

    # Reference: the pre-cache formula (interp full grid, then mean). The cache
    # path reduces in the met's source dtype (float32) before casting, so agree
    # only to float32 reduction noise -- physically irrelevant for a diagnostic.
    m_start, m_end = win1.channels("u", "v", "w")
    ref = (m_start.to(dtp) * (1.0 - a1) + m_end.to(dtp) * a1).reshape(3, -1).mean(dim=1)
    assert torch.allclose(wm1, ref, atol=1e-5)

    # A different window (new time_start, doubled fields) must invalidate the slot.
    meta2 = dataclasses.replace(
        win1.metadata,
        time_start=win1.metadata.time_start + timedelta(hours=1),
        time_end=win1.metadata.time_end + timedelta(hours=1),
    )
    win2 = dataclasses.replace(
        win1, metadata=meta2, hour_start=win1.hour_start * 2.0, hour_end=win1.hour_end * 2.0
    )
    _, wm4 = _advection_alpha_wind_mean(
        win2, meta2.time_start + timedelta(minutes=20), 60.0, device, dtp, cache
    )
    assert cache["ts"] == meta2.time_start
    assert torch.allclose(wm4, 2.0 * ref, atol=2e-5)


def test_memory_guard_trip_aborts_and_writes_diagnostics(tmp_path: Path) -> None:
    """A tripped memory guard must abort with MemoryError AND leave the guard
    diagnostics in run_metadata.json (CLAUDE.md: preserve diagnostic metadata on
    guard-triggered aborts). Uses an absurdly small RSS limit so the first check
    trips."""

    import json

    cfg = _make_run_config(
        output_uri=str(tmp_path / "out"),
        memory_guard_max_rss_gib=0.001,  # ~1 MiB: any live process exceeds this
        memory_guard_check_every_steps=1,
    )
    reader = _make_reader(cfg, wind_fn=lambda _: (5.0, 0.0, 0.0))

    with pytest.raises(MemoryError, match="Memory safety guard"):
        _run(cfg, reader=reader)

    meta_path = tmp_path / "out" / "run_metadata.json"
    assert meta_path.exists(), "guard abort must still write run_metadata.json"
    text = meta_path.read_text()
    meta = json.loads(text)
    assert meta["status"] == "aborted_memory_guard"
    assert "RSS" in meta["reason"]
    assert "guard_limit_rss_bytes" in text
    assert "guard_observed_rss_bytes" in text


# --- meteorology cadence ------------------------------------------------------


def _cadence_run(tmp_path: Path, interval_seconds: int) -> dict:
    """Identical constant-wind run, driven by met at the given source cadence."""

    cfg = _make_run_config(
        release_duration_seconds=60,
        simulation_length_seconds=10800,
        dt_seconds=300,
        release_lon=0.0,
        release_lat=0.0,
        n_particles=256,
        release_seed=42,
        output_uri=str(tmp_path / f"out_{interval_seconds}"),
    )
    # Coverage is padded by a whole window at each end: a coarser source has to
    # cover the window ENCLOSING the run, not just its endpoints.
    reader = AnalyticMetReader(
        coverage_start=cfg.simulation.start_time
        - timedelta(seconds=cfg.simulation.length_seconds + 2 * interval_seconds),
        coverage_end=cfg.simulation.start_time
        + timedelta(seconds=cfg.release.duration_seconds + 2 * interval_seconds),
        wind_fn=lambda _: (5.0, 0.0, 0.0),
        interval_seconds=interval_seconds,
    )
    return _run(cfg, reader=reader)


def test_three_hourly_met_run_completes_and_fetches_fewer_windows(tmp_path: Path) -> None:
    """A 3-hourly source must drive the run end to end, bracketing on 3 h rather than
    failing on a timestamp it does not have — and it should read a third as many
    windows over the same simulated period."""

    hourly = _cadence_run(tmp_path, 3600)
    three_hourly = _cadence_run(tmp_path, 10800)

    assert hourly["runtime"]["status"] == "completed"
    assert three_hourly["runtime"]["status"] == "completed"
    assert three_hourly["runtime"]["hour_windows"] < hourly["runtime"]["hour_windows"]


def test_met_cadence_does_not_change_constant_wind_transport(tmp_path: Path) -> None:
    """Under a constant wind the interpolation is exact at any cadence, so the mean
    trajectory must be identical — this isolates bracketing changes from physics."""

    _cadence_run(tmp_path, 3600)
    _cadence_run(tmp_path, 10800)

    hourly = pd.read_parquet(tmp_path / "out_3600" / "trajectory_diagnostics.parquet")
    three_hourly = pd.read_parquet(tmp_path / "out_10800" / "trajectory_diagnostics.parquet")

    assert abs(float(hourly.iloc[-1]["mean_lon"]) - float(three_hourly.iloc[-1]["mean_lon"])) < 1e-9


def test_met_cache_thrash_threshold_counts_windows_not_hours(caplog) -> None:
    """The cache counts WINDOWS. At 3-hourly met a 143 h batch span is only ~48 entries,
    so a 144-entry cache is ample — comparing hours against the limit would warn anyway
    and push the user into a 3x larger host-RAM cache than the run needs."""

    from types import SimpleNamespace

    from lpdm.main import _warn_if_met_cache_thrashes

    base = datetime(2024, 1, 1, tzinfo=UTC)

    def _rel(secs: int) -> SimpleNamespace:
        return SimpleNamespace(release_time=base + timedelta(seconds=secs), duration_seconds=3600)

    b0 = SimpleNamespace(releases=[_rel(h * 3600) for h in range(24)])
    b1 = SimpleNamespace(releases=[_rel(86400 + h * 3600) for h in range(24)])
    sim_len = 5 * 86400.0
    cfg = _make_run_config(met_cache_max_hours=144)

    caplog.clear()
    with caplog.at_level(logging.WARNING):
        _warn_if_met_cache_thrashes(cfg, [b0, b1], sim_len, 3600)
    assert "below the cross-batch reuse threshold" in caplog.text  # 144 < 167 windows

    caplog.clear()
    with caplog.at_level(logging.WARNING):
        _warn_if_met_cache_thrashes(cfg, [b0, b1], sim_len, 10800)
    assert "cross-batch reuse threshold" not in caplog.text  # 144 >= ~56 windows


def test_seeded_runs_are_reproducible(tmp_path: Path) -> None:
    """Decision 0021: the release seed seeds torch's default generators, so the
    turbulence noise (not just the release positions) repeats exactly. Disturbing
    the global generator between runs must not matter; another seed must."""

    import xarray as xr

    def footprint(name: str, seed: int) -> np.ndarray:
        cfg = _make_run_config(output_uri=str(tmp_path / name), release_seed=seed)
        meta = _run(cfg, reader=_make_reader(cfg, wind_fn=lambda _: (5.0, 0.0, 0.0)))
        assert meta["runtime"]["rng_seed"] == seed
        return xr.open_zarr(tmp_path / name / "footprints.zarr")["footprint"].values

    a = footprint("a", 42)
    torch.manual_seed(7)  # disturb the global state: the run must re-seed it
    torch.rand(1000)
    b = footprint("b", 42)
    c = footprint("c", 43)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_run_metadata_lists_met_stores(tmp_path: Path) -> None:
    """Task B1: run_metadata.json records each meteorology store's provenance;
    a reader with no stores (the analytic one) gives an empty list."""

    cfg = _make_run_config(output_uri=str(tmp_path / "out"))
    metadata = _run(cfg, reader=_make_reader(cfg, wind_fn=lambda _: (5.0, 0.0, 0.0)))
    assert metadata["runtime"]["met_stores"] == []


# ---- A00: time-ago bin width (decision 0016) ------------------------------------


def _binned_config(tmp_path: Path, name: str, length_s: int, n_bins: int, width_s: int, **og):
    base = _make_periodic_config(
        output_uri=str(tmp_path / name),
        n_releases=1,
        duration_seconds=300,
        n_particles_per_release=16,
        simulation_length_seconds=3600,  # valid with the fixture's 3 bins; set below
    ).model_dump()
    base["simulation"]["length_seconds"] = length_s
    base["output_grid"].update(n_time_bins=n_bins, time_bin_seconds=width_s, **og)
    return RunConfig.model_validate(base)


@pytest.mark.parametrize(
    ("length_s", "n_bins", "width_s"),
    [(10800, 1, 3600), (10800, 3, 3600), (21600, 3, 7200), (172800, 2, 86400)],
    ids=["one-bin", "hourly", "two-hourly", "daily-on-two-days"],
)
def test_time_bin_labels_match_indexing(
    tmp_path: Path, length_s: int, n_bins: int, width_s: int, monkeypatch
) -> None:
    """Each bin holds the residence its labels say: in still air every particle
    spends one bin width in each full bin, the axis carries one more bin for the
    release window, and the whole run is kept."""

    monkeypatch.setattr(PlaceholderConstantOU, "sigma_w2", 0.0)
    cfg = _binned_config(tmp_path, "out", length_s, n_bins, width_s)
    _run(cfg, reader=_make_multi_release_reader(cfg, wind_fn=lambda _: (0.0, 0.0, 0.0)))

    fp = xr.open_zarr(tmp_path / "out" / "footprints.zarr")
    per_bin = fp["footprint"].sum(dim=("release", "z_bin", "latitude", "longitude")).values
    assert float(per_bin.sum()) == pytest.approx(length_s, rel=1e-3)
    if n_bins == 1:
        assert per_bin.shape == (1,)
        return
    w_h = width_s / 3600.0
    assert per_bin.shape == (n_bins + 1,)  # one extra bin for the 300 s release window
    np.testing.assert_allclose(fp["time_ago_start_hours"].values, np.arange(n_bins + 1) * w_h)
    np.testing.assert_allclose(fp["time_ago_end_hours"].values, (np.arange(n_bins + 1) + 1) * w_h)
    # Ages count from the window's end, so the first bin gives up to one window
    # (300 s) to the extra last bin; the middle bins are full.
    assert width_s - 300 - 1 <= per_bin[0] <= width_s + 1
    np.testing.assert_allclose(per_bin[1:n_bins], width_s, rtol=1e-3)
    assert 0 <= per_bin[n_bins] <= 300 + 1


def test_time_bins_must_cover_the_simulation(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="drop_older_ages"):
        _binned_config(tmp_path, "a", 10800, 2, 3600)
    cfg = _binned_config(tmp_path, "b", 10800, 2, 3600, drop_older_ages=True)
    assert cfg.output_grid.drop_older_ages
    _binned_config(tmp_path, "c", 10800, 1, 3600)  # one bin: time-integrated, always fine
    _binned_config(tmp_path, "d", 10800, 2, 5400)  # 2 x 1.5 h covers 3 h


# ---- A03: meander timescale binding, the model top, and stopping by age ----


@pytest.mark.parametrize(
    ("interval_seconds", "configured", "expected"),
    [(3600, None, 1800.0), (10800, None, 5400.0), (10800, 900.0, 900.0)],
)
def test_meander_timescale_bound_from_the_met_interval(
    tmp_path: Path, interval_seconds: int, configured, expected: float
) -> None:
    """Decision 0024: an unset meander timescale becomes half the interval the
    reader reports, fixed once before the first step; a configured one is kept."""

    from lpdm.main import _scheme_kwargs
    from lpdm.turbulence import get_scheme

    cfg = _make_run_config(
        output_uri=str(tmp_path / "out"),
        turbulence_scheme="hanna_1982",
        simulation_length_seconds=10800,
        n_particles=32,
        n_time_bins=1,
    )
    meander = {"enabled": True, "timescale_seconds": configured}
    cfg = cfg.model_copy(
        update={
            "turbulence": cfg.turbulence.model_copy(
                update={"meander": cfg.turbulence.meander.model_copy(update=meander)}
            )
        }
    )
    reader = AnalyticMetReader(
        coverage_start=cfg.simulation.start_time
        - timedelta(seconds=cfg.simulation.length_seconds + interval_seconds),
        coverage_end=cfg.simulation.start_time + timedelta(seconds=interval_seconds),
        wind_fn=lambda _: (5.0, 0.0, 0.0),
        channel_names=("u", "v", "w", "blh", "sp", "t", "ustar", "shf"),
        interval_seconds=interval_seconds,
    )
    scheme = get_scheme("hanna_1982", **_scheme_kwargs(cfg))

    metadata = _run(cfg, reader=reader, scheme=scheme)

    assert metadata["runtime"]["status"] == "completed"
    assert scheme.meander_timescale_seconds == expected


@pytest.mark.parametrize("static", [False, True], ids=["dynamic", "static"])
def test_particle_reaching_the_model_top_is_removed(
    tmp_path: Path, static: bool, monkeypatch
) -> None:
    """docs/physics.md §5: a particle that rises above met_domain.alt_max_m (the top
    of the vertical grid) is removed at the end of the step in which it crosses
    (`_within_met_domain`, applied in `main._run` after the footprint is counted).
    It is neither reflected nor held at the top. It counts for that last step at
    the height it reached, then for nothing; its end record keeps that position.

    Still air but for a uniform updraft: backward in time a particle moves by
    -w dt, so w = -1 m/s lifts it 300 m per 300 s step. From 450 m it is at
    1,950 m after five steps (inside a 2,000 m top) and 2,250 m after six.
    """

    monkeypatch.setattr(PlaceholderConstantOU, "sigma_w2", 0.0)
    monkeypatch.setenv("GLIDE_STATIC_SUBSTEPS", "1" if static else "0")
    n = 16
    cfg = _make_run_config(
        output_uri=str(tmp_path / "out"),
        simulation_length_seconds=10800,
        dt_seconds=300,
        release_alt_agl_m=450.0,
        release_duration_seconds=300,
        n_particles=n,
        n_time_bins=1,
        met_alt_max_m=2000.0,
        z_edges_m=(0.0, 1000.0, 2000.0, 3000.0),
    )
    reader = _make_reader(cfg, wind_fn=lambda _: (0.0, 0.0, -1.0))

    metadata = _run(cfg, reader=reader)

    traj = pd.read_parquet(tmp_path / "out" / "trajectory_diagnostics.parquet")
    escaped = traj["escaped_this_step"].to_numpy()
    active = traj["active_particles"].to_numpy()
    assert int(escaped.sum()) == n == metadata["schedule"]["escaped_met_domain_total"]
    # All removed together, at the end of their sixth step (450 + 6 x 300 = 2,250 m),
    # none earlier, and nothing active afterwards.
    first_active = int(np.nonzero(active)[0][0])
    (step_idx,) = np.nonzero(escaped)
    assert step_idx.tolist() == [first_active + 5]
    assert int(active[step_idx[0] + 1 :].max()) == 0

    # End records keep the position at removal, 250 m above the top.
    ends = pd.read_parquet(tmp_path / "out" / "endpoint_particles.parquet")
    np.testing.assert_allclose(ends["alt"].to_numpy(), 2250.0, atol=1e-3)

    # The footprint counts every step up to and including the crossing one, that one
    # at 2,250 m in the 2-3 km layer above the top, and nothing after it: one step
    # at 750 m, four between 1,050 and 1,950 m, one at 2,250 m.
    fp = xr.open_zarr(tmp_path / "out" / "footprints.zarr")["footprint"]
    per_layer = fp.sum(dim=("release", "time_ago", "latitude", "longitude")).values
    np.testing.assert_allclose(per_layer, [300.0, 1200.0, 300.0], rtol=1e-5)


@pytest.mark.parametrize("static", [False, True], ids=["dynamic", "static"])
def test_each_particle_stops_its_own_length_after_its_release(
    tmp_path: Path, static: bool, monkeypatch
) -> None:
    """Stopping by age (docs/physics.md §4): every particle is followed for
    simulation.length_seconds back from its own release time and then stops,
    whatever the other releases in its batch do; there is no single end time for
    the run's particles. (`active_mask` in `main._run`; the footprint totals are
    checked in test_every_particle_followed_for_full_simulation_length.)

    Two releases 2 h apart, each within one 300 s step, followed 3 h, in one
    batch. Backward from the later release: 2 h with only its particles, 1 h with
    both, then 2 h with only the earlier release's, which stops 3 h before its
    own release time, not at the end of the later release's 3 h.
    """

    monkeypatch.setattr(PlaceholderConstantOU, "sigma_w2", 0.0)
    monkeypatch.setenv("GLIDE_STATIC_SUBSTEPS", "1" if static else "0")
    n = 8
    cfg = _make_periodic_config(
        output_uri=str(tmp_path / "out"),
        n_releases=2,
        period_seconds=7200,
        duration_seconds=300,
        n_particles_per_release=n,
        simulation_length_seconds=10800,
        max_releases_per_batch=2,
    )
    reader = _make_multi_release_reader(cfg, wind_fn=lambda _: (0.0, 0.0, 0.0))

    _run(cfg, reader=reader)

    traj = pd.read_parquet(tmp_path / "out" / "trajectory_diagnostics.parquet")
    active = traj["active_particles"].to_numpy()
    steps_per_hour = 3600 // cfg.simulation.dt_seconds
    expected = [n] * (2 * steps_per_hour) + [2 * n] * steps_per_hour + [n] * (2 * steps_per_hour)
    assert active[active > 0].tolist() == expected


# --- Sub-step exceedance counter (decision 0027; replaced A1's cap counter) ---------


@pytest.mark.parametrize("path", ["dynamic", "static", "graph"])
def test_substep_exceedance_counted_on_every_path(
    tmp_path: Path, monkeypatch, caplog, path: str
) -> None:
    """Every active particle-sub-step is counted once, in the boundary layer or the
    free troposphere, on the dynamic path, the static eager path and the static
    graph-compiled path (on CPU the compiled core is traced and run, without CUDA
    graphs), and the totals reach run_metadata.json (``runtime.substep_exceedance``)
    and the end-of-run log. A reporting threshold of 1e-6 makes every sub-step
    exceed it."""

    import logging

    from lpdm.turbulence import HannaScheme

    monkeypatch.setenv("GLIDE_STATIC_SUBSTEPS", "0" if path == "dynamic" else "1")
    monkeypatch.setenv("GLIDE_COMPILE", "1" if path == "graph" else "0")
    cfg = _make_run_config(
        output_uri=str(tmp_path / "out"),
        turbulence_scheme="hanna_1982",
        simulation_length_seconds=7200,
        n_particles=64,
    )
    reader = _make_hanna_reader(cfg, wind_fn=lambda _: (5.0, 0.0, 0.0))
    scheme = HannaScheme(n_substeps=3, exceedance_c=1e-6)
    with caplog.at_level(logging.INFO):
        metadata = _run(cfg, reader=reader, scheme=scheme)
    if path == "graph":
        assert scheme._graph_compile_state == "ok"

    exc = metadata["runtime"]["substep_exceedance"]
    traj = pd.read_parquet(tmp_path / "out" / "trajectory_diagnostics.parquet")
    active_steps = int(traj["active_particles"].sum())
    assert active_steps > 0
    assert exc["n_substeps"] == 3 and exc["exceedance_c"] == 1e-6
    # n_substeps is a count: the sub-step length follows dt_seconds, is recorded,
    # and a sub-step longer than the validated 60 s / 16 is warned about.
    assert exc["substep_seconds"] == pytest.approx(cfg.simulation.dt_seconds / 3)
    assert cfg.simulation.dt_seconds / 3 > 3.75
    assert [
        r
        for r in caplog.records
        if r.levelno == logging.WARNING and "sub-step (dt_seconds" in r.getMessage()
    ]
    bl, ft = exc["boundary_layer"], exc["free_troposphere"]
    assert bl["substeps"] + ft["substeps"] == 3 * active_steps
    assert bl["exceeded"] == bl["substeps"] and ft["exceeded"] == ft["substeps"]
    assert bl["over_T_L"] <= bl["exceeded"] and ft["over_T_L"] <= ft["exceeded"]
    assert exc["met_windows_with_exceedance"] >= 2  # 2 h back: at least two windows
    assert bl["worst_sub_dt_over_T_Lw"] > 0.0
    assert [r for r in caplog.records if r.getMessage().startswith("sub-step exceedance")]
    on_disk = json.loads((tmp_path / "out" / "run_metadata.json").read_text())
    assert on_disk["runtime"]["substep_exceedance"] == exc


@pytest.mark.parametrize("static", [False, True])
def test_substep_exceedance_zero_in_the_boundary_layer_with_short_substeps(
    tmp_path: Path, monkeypatch, static: bool
) -> None:
    """With the FLEXPART floor (T_Lw ≥ 30 s in the boundary layer) and 7.5 s
    sub-steps (dt 300 s, 40 sub-steps), no boundary-layer sub-step exceeds 0.5 T_Lw;
    the denominator is still recorded."""

    from lpdm.turbulence import HannaScheme

    monkeypatch.setenv("GLIDE_STATIC_SUBSTEPS", "1" if static else "0")
    cfg = _make_run_config(
        output_uri=str(tmp_path / "out"),
        turbulence_scheme="hanna_1982",
        simulation_length_seconds=7200,
        n_particles=64,
    )
    reader = _make_hanna_reader(cfg, wind_fn=lambda _: (5.0, 0.0, 0.0))
    metadata = _run(cfg, reader=reader, scheme=HannaScheme(n_substeps=40))
    bl = metadata["runtime"]["substep_exceedance"]["boundary_layer"]
    assert bl["substeps"] > 0
    assert bl["exceeded"] == bl["over_T_L"] == 0
    assert 0.0 < bl["worst_sub_dt_over_T_Lw"] <= 7.5 / 30.0 + 1e-6


def test_substep_exceedance_absent_for_scheme_without_substeps(tmp_path: Path) -> None:
    """A scheme with no sub-steps records None (the key is still present)."""

    cfg = _make_run_config(output_uri=str(tmp_path / "out"))
    metadata = _run(cfg, reader=_make_reader(cfg, wind_fn=lambda _: (5.0, 0.0, 0.0)))
    assert metadata["runtime"]["substep_exceedance"] is None


@pytest.mark.parametrize("override", [False, True])
def test_surface_layer_override_warns_once_and_is_recorded(
    tmp_path: Path, caplog, override: bool
) -> None:
    """The opt-in surface-layer override is not well-mixed consistent at 0.1 h
    (decision 0028, "Other boundaries"): a run built from a config with it on logs
    one warning saying so and records it in run_metadata.json
    (``runtime.physics_warnings``). The default logs nothing and records []."""

    import logging

    from lpdm.turbulence.hanna import SURFACE_LAYER_OVERRIDE_WARNING

    cfg = _make_run_config(
        output_uri=str(tmp_path / "out"),
        turbulence_scheme="hanna_1982",
        simulation_length_seconds=7200,
        n_particles=64,
    )
    turbulence = cfg.turbulence.model_copy(update={"surface_layer_override": override})
    cfg = cfg.model_copy(update={"turbulence": turbulence})
    reader = _make_hanna_reader(cfg, wind_fn=lambda _: (5.0, 0.0, 0.0))
    with caplog.at_level(logging.WARNING):
        metadata = _run(cfg, reader=reader)  # the scheme is built from cfg
    warned = [r for r in caplog.records if "surface_layer_override" in r.getMessage()]
    on_disk = json.loads((tmp_path / "out" / "run_metadata.json").read_text())
    if override:
        assert len(warned) == 1 and warned[0].levelno == logging.WARNING
        message = warned[0].getMessage()
        assert message == SURFACE_LAYER_OVERRIDE_WARNING
        assert "A/B comparisons only" in message
        assert "not well-mixed consistent at 0.1 h" in message
        assert metadata["runtime"]["physics_warnings"] == [SURFACE_LAYER_OVERRIDE_WARNING]
    else:
        assert not warned
        assert metadata["runtime"]["physics_warnings"] == []
    assert on_disk["runtime"]["physics_warnings"] == metadata["runtime"]["physics_warnings"]


def test_static_and_dynamic_exceedance_counts_agree_with_inactive_particles() -> None:
    """With one sub-step the counts depend only on T_Lw at the start of the step, so
    one step from the same state counts the same particle-sub-steps on both paths,
    with a quarter of the particles inactive (never counted on either)."""

    from lpdm.gpu_engine import GPUEngine
    from lpdm.turbulence import HannaScheme

    met = _wmc_met_window(blh_m=1500.0, ustar_m_s=0.4)
    n = 64
    torch.manual_seed(5)
    z0 = torch.rand(n) * 2400.0 + 1.0
    counts = {}
    for static in (False, True):
        engine = GPUEngine(device="cpu")
        # Legacy near-surface treatment: T_Lw falls to ~1 s near the ground, so with
        # 300 s sub-steps the low particles exceed T_Lw and the high ones do not.
        scheme = HannaScheme(
            n_substeps=1,
            flexpart_tl_floors=False,
            surface_layer_override=True,
            static_substeps=static,
        )
        particles = torch.zeros(n, 4, dtype=torch.float32)
        particles[:, 1] = 45.0
        particles[:, 2] = z0
        particles[:, 3] = 1.0
        state = scheme.initialize_state(n, device=torch.device("cpu"), dtype=torch.float32)
        active = torch.ones(n, dtype=torch.bool)
        active[: n // 4] = False  # inactive particles are never counted
        scheme.step(
            particles, state, met, t_alpha=0.5, dt_seconds=300.0, active_mask=active,
            engine=engine,
        )  # fmt: skip
        counts[static] = scheme.take_substep_exceedance("w")
        # Read-and-reset: a second read is zero.
        assert scheme.take_substep_exceedance("w")["substeps_bl"] == 0
    dyn, stat = counts[False], counts[True]
    assert dyn["substeps_bl"] + dyn["substeps_ft"] == 3 * n // 4
    assert 0 < dyn["over_T_L_bl"] < dyn["substeps_bl"]
    for key in (
        "substeps_bl",
        "exceeded_bl",
        "over_T_L_bl",
        "substeps_ft",
        "exceeded_ft",
        "over_T_L_ft",
    ):
        assert dyn[key] == stat[key]
    assert dyn["worst_bl"] == pytest.approx(stat["worst_bl"])
