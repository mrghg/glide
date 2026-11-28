"""Task A4: particle positions in a local metric frame, and the 16-bit wind gather.

The local metric frame is an affine map of (lon, lat), so the model's motion is
the same in it as in degrees; these tests check the conversions, that the
degrees frame keeps the old arithmetic to the bit, that the metric frame agrees
with it to float32 rounding on both step paths and through the runtime, and
that the bfloat16 wind gather is a trilinear interpolation with float32
accumulation whose error is bounded by the rounding of the field.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import xarray as xr
from test_main_runtime import _make_hanna_reader, _make_run_config

from lpdm.gpu_engine import GPUEngine, GridInterpolationBounds, frame_grid_bounds, trilinear_sample
from lpdm.local_frame import (
    DEGREES,
    M_PER_DEG_LAT,
    M_PER_DEG_LON_EQ,
    HorizontalFrame,
)
from lpdm.main import _run, _within_met_domain
from lpdm.met_reader import HourlyMetTensors, MetFieldMetadata
from lpdm.turbulence import HannaScheme

METRIC = HorizontalFrame.local_metric((-98.0, 39.5), (10.75, 79.0))  # the EUROPE domain


# --------------------------------------------------------------------------- frame


def test_local_metric_frame_is_metres_from_the_domain_centre() -> None:
    f = METRIC
    assert f.lon0 == pytest.approx(-29.25) and f.lat0 == pytest.approx(44.875)
    assert f.ky == M_PER_DEG_LAT
    assert f.kx == pytest.approx(M_PER_DEG_LON_EQ * math.cos(math.radians(44.875)))
    assert f.x_from_lon(f.lon0) == 0.0 and f.y_from_lat(f.lat0) == 0.0
    # One degree of latitude is 110540 y-metres; one of longitude kx x-metres.
    assert f.y_from_lat(f.lat0 + 1.0) == pytest.approx(M_PER_DEG_LAT)
    assert f.x_from_lon(f.lon0 + 1.0) == pytest.approx(f.kx)
    # The metric: a y unit is a metre; an x unit is a metre at lat0, cos(lat)/cos(lat0) elsewhere.
    lat = torch.tensor([f.lat0, 60.0, 79.0], dtype=torch.float64)
    expected = torch.cos(torch.deg2rad(lat)) / math.cos(math.radians(f.lat0))
    torch.testing.assert_close(f.metres_per_x(lat), expected, rtol=1e-12, atol=0)
    assert f.metres_per_y == 1.0


def test_round_trip_is_float32_rounding() -> None:
    torch.manual_seed(0)
    lonlat = torch.stack(
        [torch.empty(10000).uniform_(-98.0, 39.5), torch.empty(10000).uniform_(10.75, 79.0)], 1
    )
    p = torch.cat([lonlat, torch.rand(10000, 2)], 1).to(torch.float32)
    back = METRIC.to_lonlat(METRIC.to_frame(p))
    # Two float32 roundings of an affine map: a few ulp of the larger operand.
    assert (back[:, 0] - p[:, 0]).abs().max() < 4e-5  # degrees (~4 m)
    assert (back[:, 1] - p[:, 1]).abs().max() < 2e-5
    assert torch.equal(back[:, 2:], p[:, 2:])
    # The degrees frame is the identity, the same tensor.
    assert DEGREES.to_frame(p) is p and DEGREES.to_lonlat(p) is p


def test_degrees_frame_metric_is_the_old_expression_bit_for_bit() -> None:
    lat = torch.linspace(-89.9, 89.9, 4001, dtype=torch.float32)
    old = 111320.0 * torch.cos(torch.deg2rad(lat)).abs().clamp(min=0.05)
    assert torch.equal(DEGREES.metres_per_x(lat), old)
    assert DEGREES.metres_per_y == 110540.0


def test_frame_grid_bounds_degrees_equals_the_old_bounds() -> None:
    md = _met(n_lon=7, n_lat=5).metadata
    old = GridInterpolationBounds(
        lon_first=float(md.lon[0]),
        lon_last=float(md.lon[-1]),
        lat_first=float(md.lat[0]),
        lat_last=float(md.lat[-1]),
        alt_first=float(md.level[0]),
        alt_last=float(md.level[-1]),
        level_agl_m=tuple(float(v) for v in md.level),
    )
    assert frame_grid_bounds(md, DEGREES) == old
    b = frame_grid_bounds(md, METRIC)
    assert b.lon_first == pytest.approx(METRIC.x_from_lon(float(md.lon[0])))
    assert b.lat_last == pytest.approx(METRIC.y_from_lat(float(md.lat[-1])))
    assert b.level_agl_m == old.level_agl_m


def test_met_domain_check_is_the_same_box_in_either_frame() -> None:
    cfg = _make_run_config()
    frame = HorizontalFrame.local_metric(cfg.met_domain.lon_bounds, cfg.met_domain.lat_bounds)
    lon = torch.tensor([0.0, 2.9, 3.1, -3.1, 0.0, 0.0, 0.0])
    lat = torch.tensor([0.0, 0.0, 0.0, 0.0, 2.9, -3.1, 0.0])
    alt = torch.tensor([100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 1e9])
    p = torch.stack([lon, lat, alt, torch.ones(7)], 1)
    inside_deg = _within_met_domain(p, cfg, frame=DEGREES)
    inside_m = _within_met_domain(frame.to_frame(p), cfg, frame)
    assert torch.equal(inside_deg, inside_m)


# --------------------------------------------------------------------------- met


def _met(
    *,
    n_lev: int = 8,
    n_lat: int = 9,
    n_lon: int = 11,
    lat0: float = 50.0,
    time_varying: bool = True,
) -> HourlyMetTensors:
    """A window with winds varying in all three directions and in time, a BL
    whose depth varies horizontally, and a surface heat flux, so every
    interpolation and the latitude metric matter."""

    shape = (n_lev, n_lat, n_lon)
    lev = torch.linspace(0.0, 1.0, n_lev).view(-1, 1, 1)
    la = torch.linspace(0.0, 1.0, n_lat).view(1, -1, 1)
    lo = torch.linspace(0.0, 1.0, n_lon).view(1, 1, -1)

    def fields(phase: float) -> torch.Tensor:
        u = 8.0 + 6.0 * torch.sin(3.0 * lo + 2.0 * la + phase) * (1.0 + lev)
        v = -3.0 + 4.0 * torch.cos(2.0 * lo - 3.0 * la + phase) + 2.0 * lev
        w = 0.05 * torch.sin(5.0 * lo + 4.0 * la + 2.0 * lev + phase)
        blh = 600.0 + 500.0 * la * lo + 0.0 * lev
        ones = torch.ones(shape)
        chans = [
            u.expand(shape),
            v.expand(shape),
            w.expand(shape),
            blh.expand(shape),
            101325.0 * ones,
            (285.0 - 8.0 * lev).expand(shape),
            (0.25 + 0.2 * lo + 0.0 * la).expand(shape),
            (40.0 + 80.0 * la + 0.0 * lo).expand(shape),
        ]
        return torch.stack([c.contiguous() for c in chans]).to(torch.float32)

    t0 = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)
    level = np.linspace(0.0, 4000.0, n_lev)
    metadata = MetFieldMetadata(
        lon=np.linspace(-4.0, 4.0, n_lon),
        lat=np.linspace(lat0 - 3.0, lat0 + 3.0, n_lat),
        level=level,
        pressure_level_hpa=np.linspace(1000.0, 600.0, n_lev),
        time_start=t0,
        # 4096 s: every RK2 time weight is a float32 number (see
        # test_static_and_dynamic_agree_bit_for_bit).
        time_end=t0 + timedelta(seconds=4096),
        variable_units={},
    )
    height = torch.as_tensor(level, dtype=torch.float32).view(-1, 1, 1).expand(shape).contiguous()
    return HourlyMetTensors(
        hour_start=fields(0.0),
        hour_end=fields(0.7 if time_varying else 0.0),
        metadata=metadata,
        channel_names=("u", "v", "w", "blh", "sp", "t", "ustar", "shf"),
        height_agl_m=height,
    )


def _particles(n: int, seed: int = 3) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    p = torch.empty(n, 4, dtype=torch.float32)
    p[:, 0] = torch.empty(n).uniform_(-2.5, 2.5, generator=g)
    p[:, 1] = torch.empty(n).uniform_(48.0, 52.0, generator=g)
    p[:, 2] = torch.empty(n).uniform_(5.0, 1500.0, generator=g)
    p[:, 3] = 1.0
    return p


def _steps(
    frame: HorizontalFrame,
    *,
    static: bool,
    n_steps: int = 5,
    wind_dtype: torch.dtype = torch.float32,
    n: int = 3000,
    time_varying: bool = True,
) -> torch.Tensor:
    """``n_steps`` Hanna steps (advection folded in on the static path, done here
    with the runtime's dynamic-path advection otherwise) from the same start and
    the same random numbers; returns the end positions in (lon, lat, z)."""

    from lpdm.main import _advect_active_particles, _advection_alpha_wind_mean

    engine = GPUEngine(device="cpu", frame=frame, wind_gather_dtype=wind_dtype)
    scheme = HannaScheme(meander_enabled=True, static_substeps=static, n_substeps=4)
    met = _met(time_varying=time_varying)
    p = frame.to_frame(_particles(n))
    state = scheme.initialize_state(n, device=torch.device("cpu"), dtype=torch.float32)
    active = torch.ones(n, dtype=torch.bool)
    torch.manual_seed(11)
    t_cursor = met.metadata.time_end
    for _ in range(n_steps):
        alpha, _ = _advection_alpha_wind_mean(met, t_cursor, 60.0, engine.device, torch.float32)
        if not scheme.step_includes_advection(engine):
            p, _, _ = _advect_active_particles(
                engine, engine.device, p, met, t_cursor, 60.0, torch.float32
            )
        p, state = scheme.step(p, state, met, alpha, 60.0, active, engine)
        t_cursor -= timedelta(seconds=60)
    return frame.to_lonlat(p)


def _horizontal_m(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a, b = a.double(), b.double()
    dx = (a[:, 0] - b[:, 0]) * 111320.0 * torch.cos(torch.deg2rad(b[:, 1]))
    dy = (a[:, 1] - b[:, 1]) * 110540.0
    return torch.hypot(dx, dy)


@pytest.mark.parametrize("static", [False, True])
def test_metric_frame_agrees_with_degrees_to_rounding(static: bool) -> None:
    """Five 60 s steps with advection, turbulence, meander and reflection: the
    metric frame moves particles as the degrees frame does, up to float32
    rounding of the positions. One float32 ulp of a latitude near 50 degrees is
    0.42 m (of a y near 5e5 m, 0.06 m), so the median difference is about one
    ulp of the degrees representation; the maximum allows the rounding to grow
    through five steps of four sub-steps."""

    deg = _steps(DEGREES, static=static)
    met = _steps(METRIC, static=static)
    dh = _horizontal_m(met, deg)
    dz = (met[:, 2] - deg[:, 2]).abs()
    assert torch.isfinite(met).all()
    assert float(dh.median()) < 1.5 and float(dh.max()) < 20.0, (dh.median(), dh.max())
    assert float(dz.median()) < 1e-2 and float(dz.max()) < 1.0, (dz.median(), dz.max())


@pytest.mark.parametrize(
    "frame, wind_dtype",
    [(DEGREES, torch.float32), (METRIC, torch.float32), (METRIC, torch.bfloat16)],
    ids=["degrees", "metric", "metric-bf16"],
)
def test_static_and_dynamic_agree_bit_for_bit(frame, wind_dtype) -> None:
    """With every particle active the two step paths do the same arithmetic
    (docs/architecture.md §5) in either frame, and with the 16-bit wind gather,
    which both share. The window is 4096 s long so that every RK2 time weight is
    a float32 number: the dynamic path uses it as a Python float and the static
    path as a float32 tensor, and with a weight float32 cannot hold (an hourly
    window) the two agree only to rounding, on main as here (69% of particles
    bit-identical in this setup, the rest within one ulp)."""

    dyn = _steps(frame, static=False, wind_dtype=wind_dtype)
    sta = _steps(frame, static=True, wind_dtype=wind_dtype)
    assert torch.equal(dyn, sta)


def test_bf16_winds_move_particles_by_their_rounding() -> None:
    """bfloat16 holds the wind to 2^-8 relative (8 significant bits): over five
    60 s steps at ~10 m/s that is up to ~12 m of 3 km of transport (less on
    average, the roundings being of both signs), plus what the changed positions
    do to the turbulence."""

    f32 = _steps(METRIC, static=True)
    b16 = _steps(METRIC, static=True, wind_dtype=torch.bfloat16)
    dh = _horizontal_m(b16, f32)
    assert float(dh.median()) < 10.0 and float(dh.max()) < 200.0, (dh.median(), dh.max())
    assert float(dh.median()) > 0.0  # it is a different calculation


# --------------------------------------------------------------------------- gather


def test_trilinear_sample_float32_matches_grid_sample() -> None:
    torch.manual_seed(5)
    vol = torch.randn(3, 6, 7, 9)
    grid = torch.rand(4000, 3) * 2.0 - 1.0
    grid[:7] = torch.tensor(
        [[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0], [1.0, -1.0, 0.3]] + [[0.0] * 3] * 4
    )
    ref = torch.nn.functional.grid_sample(
        vol.unsqueeze(0), grid.view(1, 1, 1, -1, 3), align_corners=True
    ).view(3, -1)
    (got,) = trilinear_sample([vol], grid)
    torch.testing.assert_close(got, ref, rtol=1e-5, atol=1e-5)


def test_trilinear_sample_bf16_error_is_the_field_rounding() -> None:
    """Weights and the sum are float32, so the error of a bf16 gather is the
    bf16 rounding of the eight corner values (≤ 2^-8 relative each), not more."""

    torch.manual_seed(6)
    vol = 10.0 + 5.0 * torch.randn(3, 6, 7, 9)
    grid = torch.rand(4000, 3) * 2.0 - 1.0
    (exact,) = trilinear_sample([vol], grid)
    (half,) = trilinear_sample([vol.to(torch.bfloat16)], grid)
    assert half.dtype == torch.float32
    (from_rounded,) = trilinear_sample([vol.to(torch.bfloat16).float()], grid)
    torch.testing.assert_close(half, from_rounded, rtol=1e-6, atol=1e-6)
    bound = 2.0**-8 * vol.abs().amax() * 1.0001
    assert float((half - exact).abs().max()) <= float(bound)


def test_gather_dtype_must_be_supported() -> None:
    with pytest.raises(ValueError):
        GPUEngine(device="cpu", wind_gather_dtype=torch.float64)


# --------------------------------------------------------------------------- runtime


def _hanna_run(
    tmp_path: Path, name: str, monkeypatch, frame: str, position: str = "endpoint", **over
) -> Path:
    out = tmp_path / name
    cfg = _make_run_config(
        output_uri=str(out),
        turbulence_scheme="hanna_1982",
        release_lon=0.5,
        release_lat=0.5,
        n_particles=400,
        simulation_length_seconds=7200,
        dt_seconds=120,
        **over,
    )
    og = cfg.output_grid
    dep = og.deposition.model_copy(update={"position": position})
    cfg = cfg.model_copy(update={"output_grid": og.model_copy(update={"deposition": dep})})
    reader = _make_hanna_reader(cfg, wind_fn=lambda _: (6.0, -2.0, 0.0), shf_w_m2=50.0)
    monkeypatch.setenv("GLIDE_POSITION_FRAME", frame)
    meta = _run(cfg, reader=reader)
    assert meta["runtime"]["position_frame"]["kind"] == frame
    return out


@pytest.mark.parametrize("position", ["endpoint", "trapezoid"])
@pytest.mark.parametrize("static", ["0", "1"])
def test_runtime_metric_frame_matches_degrees(
    tmp_path: Path, monkeypatch, static: str, position: str
) -> None:
    """The whole runtime (release, step, escape check, gridder, endpoint and
    diagnostic outputs, which convert back to degrees) in the two frames:
    endpoints within float32 rounding of positions, footprints the same up to
    the particles that rounding moves across a cell edge. The trapezoid case
    (three 1 h time-ago bins, so half steps are split by age) also deposits at
    each step's start positions, which pass through the same conversion."""

    monkeypatch.setenv("GLIDE_STATIC_SUBSTEPS", static)
    a = _hanna_run(tmp_path, "deg", monkeypatch, "degrees", position)
    b = _hanna_run(tmp_path, "met", monkeypatch, "local_metric", position)
    ea = pd.read_parquet(a / "endpoint_particles.parquet")
    eb = pd.read_parquet(b / "endpoint_particles.parquet")
    pa = torch.tensor(ea[["lon", "lat", "alt"]].to_numpy())
    pb = torch.tensor(eb[["lon", "lat", "alt"]].to_numpy())
    dh = _horizontal_m(pb, pa)
    assert float(dh.max()) < 1.0, float(dh.max())
    assert float((pb[:, 2] - pa[:, 2]).abs().max()) < 1e-2
    fa = xr.open_zarr(a / "footprints.zarr")["footprint"].values
    fb = xr.open_zarr(b / "footprints.zarr")["footprint"].values
    assert fa.sum() == pytest.approx(fb.sum(), rel=1e-6)
    ta = pd.read_parquet(a / "trajectory_diagnostics.parquet")
    tb = pd.read_parquet(b / "trajectory_diagnostics.parquet")
    np.testing.assert_allclose(tb["mean_lon"], ta["mean_lon"], atol=1e-5)
    np.testing.assert_allclose(tb["mean_lat"], ta["mean_lat"], atol=1e-5)


# --------------------------------------------------------------------------- deposition


def _density_met() -> HourlyMetTensors:
    """``_met``'s grid with the dry-air density channels (t, q, sp) varying in x, y
    and z, so a density looked up at the wrong horizontal position is wrong."""

    met = _met()
    shape = tuple(met.hour_start.shape[1:])
    n_lev, n_lat, n_lon = shape
    lev = torch.linspace(0.0, 1.0, n_lev).view(-1, 1, 1)
    la = torch.linspace(0.0, 1.0, n_lat).view(1, -1, 1)
    lo = torch.linspace(0.0, 1.0, n_lon).view(1, 1, -1)

    def fields(phase: float) -> torch.Tensor:
        t = 285.0 - 8.0 * lev + 6.0 * la - 4.0 * lo + phase
        q = 0.008 * (1.0 - 0.5 * lev) + 0.002 * lo * la
        sp = 101325.0 - 3000.0 * la + 1500.0 * lo + 0.0 * lev - 50.0 * phase
        uvw = [met.channel(c)[0 if phase == 0.0 else 1] for c in ("u", "v", "w")]
        chans = [*uvw, t.expand(shape), q.expand(shape), sp.expand(shape)]
        return torch.stack([c.contiguous() for c in chans]).to(torch.float32)

    return HourlyMetTensors(
        hour_start=fields(0.0),
        hour_end=fields(1.0),
        metadata=met.metadata,
        channel_names=("u", "v", "w", "t", "q", "sp"),
        height_agl_m=met.height_agl_m,
    )


@pytest.mark.parametrize("position", ["endpoint", "trapezoid"])
def test_deposit_step_metric_frame_matches_degrees(position: str) -> None:
    """One step's deposit (`main._deposit_step`), dry-air weighted, three time-ago
    bins: in the metric frame (positions in metres, gridder and density in that
    frame) it equals the degrees frame's to float32 rounding. For the trapezoid
    this covers both halves: the start positions and the end positions each
    convert to degrees at the gridder and look up the density in the frame. A
    gridder built in another frame than the engine's is refused."""

    from lpdm.air_density import ParticleDryAirDensity
    from lpdm.footprint_gridder import FootprintGridder
    from lpdm.main import _deposit_step, _time_ago_index, _time_weight

    met = _density_met()
    md = met.metadata
    metric = HorizontalFrame.local_metric((md.lon[0], md.lon[-1]), (md.lat[0], md.lat[-1]))
    n, delta_s, bin_s = 4000, 600.0, 3600.0
    g = torch.Generator().manual_seed(17)
    start = _particles(n, seed=5)
    end = start.clone()
    end[:, 0] += torch.empty(n).uniform_(-0.3, 0.3, generator=g)
    end[:, 1] += torch.empty(n).uniform_(-0.2, 0.2, generator=g)
    end[:, 2] = torch.empty(n).uniform_(5.0, 1500.0, generator=g)
    # Ages on and off the bin edges, so some half steps straddle one.
    elapsed = torch.empty(n).uniform_(0.0, 3.0 * bin_s - delta_s, generator=g)
    t_cursor = md.time_end - timedelta(seconds=600)
    active = torch.ones(n, dtype=torch.bool)
    active[::7] = False

    def gridder(frame: HorizontalFrame) -> FootprintGridder:
        return FootprintGridder(
            (-3.0, 3.0),
            (47.5, 52.5),
            (0.0, 100.0, 500.0, 1000.0, 2000.0),
            3,
            25,
            30,
            device="cpu",
            frame=frame,
        )

    out = {}
    for frame in (DEGREES, metric):
        engine = GPUEngine(device="cpu", frame=frame)
        grid = gridder(frame)
        _deposit_step(
            grid,
            position=position,
            start_xyz=frame.to_frame(start)[:, :3].clone(),
            end_xyz=frame.to_frame(end)[:, :3].clone(),
            active_mask=active,
            weights=start[:, 3],
            elapsed_s=elapsed,
            t_idx=_time_ago_index(elapsed, bin_s, 3),
            time_bin_s=bin_s,
            release_idx=torch.zeros(n, dtype=torch.int64),
            delta_s=delta_s,
            t_cursor=t_cursor,
            particle_density=ParticleDryAirDensity("cpu"),
            met_window=met,
            t_alpha=_time_weight(met, t_cursor.timestamp() - delta_s),
            engine=engine,
        )
        out[frame.kind] = grid.tensor.clone()

    deg, met_fp = out["degrees"], out["local_metric"]
    assert float(deg.sum()) > 0.0
    assert (deg > 0).sum() > 100
    assert float(met_fp.sum()) == pytest.approx(float(deg.sum()), rel=1e-5)
    np.testing.assert_allclose(met_fp, deg, rtol=1e-5, atol=1e-6 * float(deg.max()))

    with pytest.raises(ValueError, match="gridder frame"):
        _deposit_step(
            gridder(DEGREES),
            position=position,
            start_xyz=metric.to_frame(start)[:, :3],
            end_xyz=metric.to_frame(end)[:, :3],
            active_mask=active,
            weights=start[:, 3],
            elapsed_s=elapsed,
            t_idx=_time_ago_index(elapsed, bin_s, 3),
            time_bin_s=bin_s,
            release_idx=torch.zeros(n, dtype=torch.int64),
            delta_s=delta_s,
            t_cursor=t_cursor,
            particle_density=None,
            met_window=met,
            t_alpha=0.5,
            engine=GPUEngine(device="cpu", frame=metric),
        )
