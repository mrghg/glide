"""Dry-air density weighting of the footprint (decision 0022, queue task A04).

Each counted particle-second is divided by the dry-air density at the particle,
(p - e) / (R_d T), in the step's met window. These tests drive the full runtime
with synthetic meteorology, on both step paths, and check the comparison
helpers that convert footprints to m2 s mol-1.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
import torch
import xarray as xr

from lpdm.air_density import (
    GRAVITY_M_S2,
    R_DRY_AIR_J_KG_K,
    TV_COEFF,
    ParticleDryAirDensity,
    dry_air_density_kg_m3,
)
from lpdm.comparison import (
    flexpart_dry_air_factor,
    flexpart_to_dry_air_footprint,
    name_dry_air_factor,
    name_to_dry_air_footprint,
    to_stilt_surface_footprint,
)
from lpdm.config import RunConfig
from lpdm.footprint_gridder import FootprintGridder
from lpdm.gpu_engine import GPUEngine
from lpdm.main import PreflightValidationError, _run
from lpdm.met_reader import BoundingBoxRequest, HourlyMetTensors, MetFieldMetadata, MetReader
from lpdm.turbulence.base import TurbulenceScheme

M_AIR = 0.02897
H_LAYER = 40.0
Z_RELEASE = 20.0
T_K = 280.0
# The window boundary the review's cases straddle; the release is 10 s after it.
BOUNDARY = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)


def _sp_for_density(rho: float, t_k: float, q: float, z: float) -> float:
    """Surface pressure giving dry-air density ``rho`` at height ``z`` (isothermal)."""

    tv = t_k * (1.0 + TV_COEFF * q)
    # rho = (p - e) / (R_d T), e = q p / (0.622 + 0.378 q)  =>  p = rho R_d T / (1 - e/p)
    p = rho * R_DRY_AIR_J_KG_K * t_k / (1.0 - q / (0.622 + 0.378 * q))
    return p * math.exp(GRAVITY_M_S2 * z / (R_DRY_AIR_J_KG_K * tv))


@dataclass
class _WindowConstantReader(MetReader):
    """Still, horizontally uniform met; T, q and p_s constant within each window.

    ``fields(t0)`` gives (T, q, p_s) for the window starting at ``t0``, so the
    density can change from one window to the next (a step in time at the
    window boundary).
    """

    coverage_start: datetime
    coverage_end: datetime
    fields: object
    channel_names: tuple[str, ...] = ("u", "v", "w", "blh", "sp", "t", "q")
    n_side: int = 4
    n_lev: int = 4

    @property
    def met_interval_seconds(self) -> int:
        return 3600

    def get_time_coverage(self) -> tuple[datetime, datetime]:
        return self.coverage_start, self.coverage_end

    def fetch_hourly_window(self, request: BoundingBoxRequest) -> HourlyMetTensors:
        t0, t1 = request.time.start, request.time.end
        t_k, q, sp = self.fields(t0)
        shape = (self.n_lev, self.n_side, self.n_side)
        values = {"u": 0.0, "v": 0.0, "w": 0.0, "blh": 1000.0, "sp": sp, "t": t_k, "q": q}
        stack = torch.stack(
            [torch.full(shape, float(values[k]), dtype=torch.float32) for k in self.channel_names]
        )
        level = np.linspace(0.0, 10000.0, self.n_lev)
        metadata = MetFieldMetadata(
            lon=np.linspace(-3.0, 3.0, self.n_side),
            lat=np.linspace(-3.0, 3.0, self.n_side),
            level=level,
            pressure_level_hpa=np.linspace(1000.0, 200.0, self.n_lev),
            time_start=t0,
            time_end=t1,
            variable_units={"u": "m/s", "v": "m/s", "w": "m/s", "sp": "Pa", "t": "K", "q": "1"},
        )
        height = torch.as_tensor(level, dtype=torch.float32).view(-1, 1, 1).expand(shape)
        return HourlyMetTensors(
            hour_start=stack,
            hour_end=stack.clone(),
            metadata=metadata,
            channel_names=self.channel_names,
            height_agl_m=height.contiguous(),
        )


class _FrozenTurbulence(TurbulenceScheme):
    """No turbulence: with still air every particle stays where it was released."""

    name = "frozen_for_tests"

    def required_met_keys(self) -> tuple[str, ...]:
        return ()

    def initialize_state(self, n_particles, *, device, dtype):
        return {}

    def step(self, particles, state, met_window, t_alpha, dt_seconds, active_mask, engine):
        return particles, state


def _config(out: Path, weighting: str, *, length_s: int = 100, dt_s: int = 10) -> RunConfig:
    release_time = BOUNDARY + timedelta(seconds=10)
    return RunConfig.model_validate(
        {
            "io": {"zarr_store": "fake://x", "output_uri": str(out)},
            "simulation": {
                "start_time": release_time,
                "length_seconds": length_s,
                "dt_seconds": dt_s,
                "device": "cpu",
            },
            "release": {
                "kind": "point",
                "lon": 0.0,
                "lat": 0.0,
                "alt_agl_m": Z_RELEASE,
                # One step long, so every particle is counted for exactly
                # length_s / dt_s steps (the release window ends on a step).
                "duration_seconds": dt_s,
                "n_particles": 64,
                "seed": 7,
            },
            "turbulence": {"scheme": "placeholder_constant_ou"},
            "output_grid": {
                # One cell, one 0-40 m layer.
                "lon_bounds": (-1.0, 1.0),
                "lat_bounds": (-1.0, 1.0),
                "n_x": 1,
                "n_y": 1,
                "z_edges_m": (0.0, H_LAYER),
                "n_time_bins": 1,
                "density_weighting": weighting,
            },
            "met_domain": {"lon_bounds": (-3.0, 3.0), "lat_bounds": (-3.0, 3.0), "alt_max_m": 1e4},
            "memory": {"met_cache_max_hours": 4, "log_every_steps": 0, "gc_every_steps": 0},
        }
    )


def _run_case(tmp_path: Path, name: str, weighting: str, fields) -> xr.DataArray:
    cfg = _config(tmp_path / name, weighting)
    reader = _WindowConstantReader(
        coverage_start=BOUNDARY - timedelta(hours=2),
        coverage_end=BOUNDARY + timedelta(hours=2),
        fields=fields,
    )
    _run(cfg, reader=reader, scheme=_FrozenTurbulence())
    return xr.open_zarr(tmp_path / name / "footprints.zarr")["footprint"].isel(release=0).load()


def _set_path(monkeypatch, static: bool) -> None:
    monkeypatch.setenv("GLIDE_STATIC_SUBSTEPS", "1" if static else "0")


# ---------------------------------------------------------------------------
# The review's two cases (validation review 21 Nov 2026, finding 3)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("static", [False, True], ids=["dynamic", "static"])
@pytest.mark.parametrize(
    ("rho_before", "rho_after", "expected"),
    [
        # 90 s at 1.0 kg m-3 and 10 s at 1.2: 0.02897 / 40 x (90/1.0 + 10/1.2)
        (1.0, 1.2, 0.071218),
        # 10 s at 1.0 and 90 s at 1.2: 0.02897 / 40 x (10/1.0 + 90/1.2)
        (1.2, 1.0, 0.061561),
    ],
    ids=["90-10", "10-90"],
)
def test_review_cases_through_gridder_and_conversion(
    tmp_path, monkeypatch, static, rho_before, rho_after, expected
) -> None:
    """A particle at 20 m in one cell for 100 s: the last 10 s before the release
    (one 10 s step, in the window after the boundary) at one density, the 90 s
    before that (nine steps, the window before) at the other. Through the runtime,
    the gridder and `to_stilt_surface_footprint`."""

    _set_path(monkeypatch, static)

    def fields(t0: datetime):
        rho = rho_after if t0 >= BOUNDARY else rho_before
        return T_K, 0.0, _sp_for_density(rho, T_K, 0.0, Z_RELEASE)

    raw = _run_case(tmp_path, "w", "dry_air", fields)
    assert raw.attrs["density_weighting"] == "dry_air"
    assert raw.attrs["units"] == "s m3 kg-1"
    sens = to_stilt_surface_footprint(raw, surface_layer_depth_m=H_LAYER)
    s_after, s_before = 10.0, 90.0
    exact = M_AIR / H_LAYER * (s_before / rho_before + s_after / rho_after)
    assert float(sens.sum()) == pytest.approx(exact, rel=2e-6)
    assert round(float(sens.sum()), 6) == expected

    # The same particles unweighted hold plain seconds: 100 s per particle.
    plain = _run_case(tmp_path, "p", "none", fields)
    assert float(plain.sum()) == pytest.approx(100.0, rel=1e-6)


@pytest.mark.parametrize("static", [False, True], ids=["dynamic", "static"])
def test_uniform_met_weighted_equals_unweighted_over_density(tmp_path, monkeypatch, static):
    """Uniform T, q, p_s: weighted = unweighted / rho_d at the particle, to float32 rounding."""

    _set_path(monkeypatch, static)
    q, sp = 0.008, 98000.0

    def fields(_t0):
        return T_K, q, sp

    weighted = _run_case(tmp_path, "w", "dry_air", fields)
    plain = _run_case(tmp_path, "p", "none", fields)
    tv = T_K * (1.0 + TV_COEFF * q)
    p = sp * math.exp(-GRAVITY_M_S2 * Z_RELEASE / (R_DRY_AIR_J_KG_K * tv))
    rho_d = dry_air_density_kg_m3(p, T_K, q)
    np.testing.assert_allclose(weighted.values, plain.values / rho_d, rtol=2e-6)
    assert float(plain.sum()) > 0.0


# ---------------------------------------------------------------------------
# The density at the particle
# ---------------------------------------------------------------------------


def _window(t_start: float, t_end: float, q_start: float, q_end: float, sp: float):
    names = ("t", "q", "sp")
    shape = (4, 3, 3)

    def stack(t, q):
        return torch.stack([torch.full(shape, v, dtype=torch.float64) for v in (t, q, sp)], dim=0)

    level = np.array([0.0, 100.0, 1000.0, 5000.0])
    return HourlyMetTensors(
        hour_start=stack(t_start, q_start),
        hour_end=stack(t_end, q_end),
        metadata=MetFieldMetadata(
            lon=np.linspace(-1.0, 1.0, 3),
            lat=np.linspace(-1.0, 1.0, 3),
            level=level,
            pressure_level_hpa=np.linspace(1000.0, 500.0, 4),
            time_start=BOUNDARY,
            time_end=BOUNDARY + timedelta(hours=1),
            variable_units={},
        ),
        channel_names=names,
    )


def _rho_at(window, xyz, alpha) -> torch.Tensor:
    engine = GPUEngine(device="cpu", dtype=torch.float64, compile_hot_paths=False)
    dens = ParticleDryAirDensity("cpu", dtype=torch.float64)
    return 1.0 / dens.inverse_density(
        torch.as_tensor(xyz, dtype=torch.float64), window, alpha, engine
    )


def test_humidity_lowers_dry_air_density_by_e_over_p() -> None:
    """With q > 0, rho_d = p / (R_d T) x (p - e) / p at the same pressure."""

    q, sp = 0.015, 100000.0
    xyz = [[0.0, 0.0, 0.0], [0.3, -0.2, 0.0]]
    rho = _rho_at(_window(T_K, T_K, q, q, sp), xyz, 0.5)
    e = q * sp / (0.622 + 0.378 * q)
    moist_ideal = sp / (R_DRY_AIR_J_KG_K * T_K)
    np.testing.assert_allclose(rho.numpy(), moist_ideal * (sp - e) / sp, rtol=1e-12)
    assert (rho.numpy() < moist_ideal).all()


def test_pressure_at_the_particle_is_hypsometric_from_the_surface() -> None:
    """Isothermal, uniform q: p(z) = p_s exp(-g z / (R_d Tv)) exactly."""

    q, sp = 0.005, 95000.0
    z = np.array([0.0, 20.0, 40.0, 800.0])
    xyz = np.stack([np.zeros_like(z), np.zeros_like(z), z], axis=1)
    rho = _rho_at(_window(T_K, T_K, q, q, sp), xyz, 0.25).numpy()
    tv = T_K * (1.0 + TV_COEFF * q)
    p = sp * np.exp(-GRAVITY_M_S2 * z / (R_DRY_AIR_J_KG_K * tv))
    np.testing.assert_allclose(rho, dry_air_density_kg_m3(p, T_K, q), rtol=1e-12)


def test_density_uses_the_step_time_weight() -> None:
    """T and q are interpolated in time with the step's weight alpha."""

    sp, alpha = 100000.0, 0.3
    xyz = [[0.0, 0.0, 0.0]]
    rho = float(_rho_at(_window(270.0, 290.0, 0.002, 0.006, sp), xyz, alpha)[0])
    t = 270.0 + alpha * 20.0
    q = 0.002 + alpha * 0.004
    assert rho == pytest.approx(dry_air_density_kg_m3(sp, t, q), rel=1e-12)


def test_dry_air_needs_temperature_and_humidity(tmp_path) -> None:
    """A reader without q cannot weight by dry-air density: fail before stepping."""

    cfg = _config(tmp_path / "x", "dry_air")
    reader = _WindowConstantReader(
        coverage_start=BOUNDARY - timedelta(hours=2),
        coverage_end=BOUNDARY + timedelta(hours=2),
        fields=lambda _t0: (T_K, 0.0, 100000.0),
        channel_names=("u", "v", "w", "blh", "sp", "t"),
    )
    with pytest.raises(PreflightValidationError, match="density_weighting"):
        _run(cfg, reader=reader, scheme=_FrozenTurbulence())


# ---------------------------------------------------------------------------
# Output contract and the comparison helper
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("weighting", ["dry_air", "none"])
def test_store_and_metadata_declare_weighting_and_units(tmp_path, weighting) -> None:
    out = tmp_path / weighting
    cfg = _config(out, weighting)
    reader = _WindowConstantReader(
        coverage_start=BOUNDARY - timedelta(hours=2),
        coverage_end=BOUNDARY + timedelta(hours=2),
        fields=lambda _t0: (T_K, 0.004, 100000.0),
    )
    _run(cfg, reader=reader, scheme=_FrozenTurbulence())
    ds = xr.open_zarr(out / "footprints.zarr")
    units = {"dry_air": "s m3 kg-1", "none": "s"}[weighting]
    for attrs in (ds.attrs, ds["footprint"].attrs):
        assert attrs["density_weighting"] == weighting
        assert attrs["units"] == units
        assert "density_formula" in attrs
    meta = json.loads((out / "run_metadata.json").read_text())
    assert meta["footprint_units"]["density_weighting"] == weighting
    assert meta["footprint_units"]["units"] == units
    assert meta["config"]["output_grid"]["density_weighting"] == weighting


def _raw(weighting: str | None) -> xr.DataArray:
    raw = xr.DataArray(
        np.full((1, 1, 2, 2), 10.0),
        dims=("time_ago", "z_bin", "latitude", "longitude"),
        coords={"z_bottom_m": ("z_bin", [0.0]), "z_top_m": ("z_bin", [40.0])},
    )
    if weighting is not None:
        raw.attrs["density_weighting"] = weighting
    return raw


def test_helper_refuses_to_divide_a_weighted_footprint_again() -> None:
    with pytest.raises(ValueError, match="divided twice"):
        to_stilt_surface_footprint(
            _raw("dry_air"), surface_layer_depth_m=40.0, air_density_kg_m3=1.2
        )
    out = to_stilt_surface_footprint(_raw("dry_air"), surface_layer_depth_m=40.0)
    np.testing.assert_allclose(out.values, 10.0 * M_AIR / 40.0)


def test_helper_needs_a_density_for_an_unweighted_footprint() -> None:
    for weighting in ("none", None):  # None: a store written before decision 0022
        with pytest.raises(ValueError, match="air_density_kg_m3 is required"):
            to_stilt_surface_footprint(_raw(weighting), surface_layer_depth_m=40.0)
        out = to_stilt_surface_footprint(
            _raw(weighting), surface_layer_depth_m=40.0, air_density_kg_m3=1.25
        )
        np.testing.assert_allclose(out.values, 10.0 * M_AIR / (40.0 * 1.25))


def test_helper_warns_when_the_weighting_is_not_declared() -> None:
    """A footprint whose attrs were dropped (e.g. `.sum()` without keep_attrs) is
    treated as plain seconds, with a warning that it may be divided twice."""

    with pytest.warns(UserWarning, match="keep_attrs"):
        to_stilt_surface_footprint(_raw(None), surface_layer_depth_m=40.0, air_density_kg_m3=1.2)


# ---------------------------------------------------------------------------
# Reference models to dry air
# ---------------------------------------------------------------------------


def test_flexpart_conversion_known_answer() -> None:
    """Two output times of native s m3 kg-1, summed, x M_air / h x the dry-air factor."""

    p = xr.DataArray([[100000.0, 101000.0], [99000.0, 100500.0]], dims=("time", "x"))
    q = xr.DataArray([[0.010, 0.015], [0.012, 0.0]], dims=("time", "x"))
    factor = flexpart_dry_air_factor(p, q)
    e = q * p / (0.622 + 0.378 * q)
    expected_factor = (p / ((p - e) * (1.0 + 0.608 * q))).mean("time")
    np.testing.assert_allclose(factor.values, expected_factor.values, rtol=1e-12)
    # q = 0 needs no correction. FLEXPART's footprint is about (1 - q) times the
    # dry-air one (decision 0022), so the factor is about 1 + q.
    assert flexpart_dry_air_factor(100000.0, 0.0, time_dim=None) == pytest.approx(1.0)
    assert flexpart_dry_air_factor(100000.0, 0.015, time_dim=None) == pytest.approx(
        1.0 + 0.015, abs=5e-4
    )

    native = xr.DataArray([[3.0, 5.0], [1.0, 2.0]], dims=("time", "x"))
    s = flexpart_to_dry_air_footprint(native, dry_air_factor=factor, surface_layer_depth_m=40.0)
    np.testing.assert_allclose(
        s.values, np.array([4.0, 7.0]) * M_AIR / 40.0 * expected_factor.values, rtol=1e-12
    )
    assert s.attrs["units"] == "m**2 s mol**-1"
    assert s.attrs["density_weighting"] == "dry_air"


def test_name_conversion_known_answer() -> None:
    """ppm s x 1e-6 x cell area / 3,600 s x p / (p - e)."""

    p, q = 100000.0, 0.015
    factor = name_dry_air_factor(p, q, time_dim=None)
    e = q * p / (0.622 + 0.378 * q)
    assert factor == pytest.approx(p / (p - e), rel=1e-12)
    assert factor - 1.0 == pytest.approx(0.0241, abs=5e-4)  # about 2.4% in humid air (0022)

    ppm_s = xr.DataArray([360.0, 720.0], dims=("x",))
    area = xr.DataArray([7.5e8, 5.0e8], dims=("x",))
    s = name_to_dry_air_footprint(ppm_s, cell_area_m2=area, dry_air_factor=factor)
    np.testing.assert_allclose(
        s.values, np.array([360.0 * 7.5e8, 720.0 * 5.0e8]) * 1e-6 / 3600.0 * factor, rtol=1e-12
    )
    assert s.attrs["units"] == "m**2 s mol**-1"


def _fp_factor(p: float, q: float) -> float:
    e = q * p / (0.622 + 0.378 * q)
    return p / ((p - e) * (1.0 + 0.608 * q))


def test_flexpart_time_resolved_factor_corrects_each_interval() -> None:
    """Two hourly intervals, unequal residence and humidity: sum_t F_t x factor_t.

    Cell 0 has 3 s m3 kg-1 in a dry hour (q = 2 g/kg) and 1 in a humid hour
    (q = 18 g/kg); cell 1 the reverse. The exact answer weights each hour's
    factor by its own residence; the mean factor does not.
    """

    times = np.array(["2019-07-13T19:00", "2019-07-13T20:00"], dtype="datetime64[ns]")
    p = xr.DataArray(
        [[100000.0, 100000.0], [99000.0, 99000.0]], dims=("time", "x"), coords={"time": times}
    )
    q = xr.DataArray([[0.002, 0.002], [0.018, 0.018]], dims=("time", "x"), coords={"time": times})
    native = xr.DataArray([[3.0, 1.0], [1.0, 3.0]], dims=("time", "x"), coords={"time": times})
    f_dry, f_humid = _fp_factor(100000.0, 0.002), _fp_factor(99000.0, 0.018)

    factor_t = flexpart_dry_air_factor(p, q, time_dim=None)
    assert factor_t.dims == ("time", "x")
    s = flexpart_to_dry_air_footprint(native, dry_air_factor=factor_t)
    assert s.dims == ("x",)  # no spurious time dimension
    expected = np.array([3.0 * f_dry + 1.0 * f_humid, 1.0 * f_dry + 3.0 * f_humid]) * M_AIR / 40.0
    np.testing.assert_allclose(s.values, expected, rtol=1e-12)
    assert "exact" in s.attrs["dry_air_factor_form"]

    # The time-mean path is unchanged: (sum F) x mean factor, the same for both
    # cells here, and different from the exact answer where residence and
    # humidity covary.
    s_mean = flexpart_to_dry_air_footprint(native, dry_air_factor=flexpart_dry_air_factor(p, q))
    expected_mean = np.array([4.0, 4.0]) * M_AIR / 40.0 * (f_dry + f_humid) / 2.0
    np.testing.assert_allclose(s_mean.values, expected_mean, rtol=1e-12)
    assert "approximate" in s_mean.attrs["dry_air_factor_form"]
    assert s.values[0] < s_mean.values[0]
    assert s_mean.values[1] < s.values[1]
    # A scalar factor on a time-resolved footprint: sum, then multiply.
    s_scalar = flexpart_to_dry_air_footprint(native, dry_air_factor=1.01)
    np.testing.assert_allclose(s_scalar.values, np.array([4.0, 4.0]) * M_AIR / 40.0 * 1.01)


def test_flexpart_conversion_refuses_mismatched_factor() -> None:
    """A factor that would broadcast against the footprint is an error, not a result."""

    times = np.array(["2019-07-13T19:00", "2019-07-13T20:00"], dtype="datetime64[ns]")
    native_t = xr.DataArray(np.ones((2, 3)), dims=("time", "x"), coords={"time": times})
    native_int = native_t.sum("time")
    factor_t = xr.DataArray(np.full((2, 3), 1.01), dims=("time", "x"), coords={"time": times})

    # Time-resolved factor, time-integrated footprint.
    with pytest.raises(ValueError, match="time dimension"):
        flexpart_to_dry_air_footprint(native_int, dry_air_factor=factor_t)
    # Wrong number of output times.
    with pytest.raises(ValueError, match="1 'time' values"):
        flexpart_to_dry_air_footprint(native_t, dry_air_factor=factor_t.isel(time=[0]))
    # Same length, different times.
    shifted = factor_t.assign_coords(time=times + np.timedelta64(1, "h"))
    with pytest.raises(ValueError, match="coordinate differs"):
        flexpart_to_dry_air_footprint(native_t, dry_air_factor=shifted)
    # Only one side has time coordinates: cannot check alignment.
    with pytest.raises(ValueError, match="only one of"):
        flexpart_to_dry_air_footprint(native_t, dry_air_factor=factor_t.drop_vars("time"))
    # A factor dimension the footprint lacks, on either path.
    extra = xr.DataArray(np.full((2, 3, 2), 1.01), dims=("time", "x", "member"))
    with pytest.raises(ValueError, match="member"):
        flexpart_to_dry_air_footprint(native_t.drop_vars("time"), dry_air_factor=extra)
    with pytest.raises(ValueError, match="member"):
        flexpart_to_dry_air_footprint(native_t, dry_air_factor=extra.isel(time=0))


def test_name_conversion_refuses_time_resolved_factor() -> None:
    """NAME output is time-integrated: an unreduced factor is refused, a mean accepted."""

    p = xr.DataArray([[100000.0, 100000.0], [99000.0, 99000.0]], dims=("time", "x"))
    q = xr.DataArray([[0.002, 0.002], [0.018, 0.018]], dims=("time", "x"))
    ppm_s = xr.DataArray([360.0, 720.0], dims=("x",))
    with pytest.raises(ValueError, match="time"):
        name_to_dry_air_footprint(
            ppm_s, cell_area_m2=1.0e8, dry_air_factor=name_dry_air_factor(p, q, time_dim=None)
        )
    factor = name_dry_air_factor(p, q)
    s = name_to_dry_air_footprint(ppm_s, cell_area_m2=1.0e8, dry_air_factor=factor)
    assert s.dims == ("x",)
    np.testing.assert_allclose(
        s.values, ppm_s.values * 1e-6 * 1.0e8 / 3600.0 * factor.values, rtol=1e-12
    )


def test_mean_factor_approximation_bound() -> None:
    """The bound stated in docs/reference-runs.md §6 and the conversion docstrings.

    Hourly output over 240 h at 1013 hPa; diurnal q = 12 +- 2 g/kg (peak at
    15 h); residence only at night (20-06 h) or only by day. The mean-factor
    result differs from the exact one by at most 0.12% (FLEXPART), and always
    lies within [min, max] of factor / mean factor.
    """

    hours = np.arange(240)
    hod = hours % 24
    times = np.datetime64("2019-07-01T00:00", "ns") + hours * np.timedelta64(1, "h")
    q = 0.012 + 0.002 * np.cos(2.0 * np.pi * (hod - 15) / 24.0)
    night = ((hod >= 20) | (hod < 6)).astype(float)
    qa = xr.DataArray(q, dims=("time",), coords={"time": times})
    pa = xr.full_like(qa, 101325.0)
    factor_t = flexpart_dry_air_factor(pa, qa, time_dim=None)
    ratio_range = (factor_t.min() / factor_t.mean(), factor_t.max() / factor_t.mean())
    for residence in (night, 1.0 - night):
        native = xr.DataArray(residence, dims=("time",), coords={"time": times})
        exact = float(flexpart_to_dry_air_footprint(native, dry_air_factor=factor_t))
        approx = float(flexpart_to_dry_air_footprint(native, dry_air_factor=factor_t.mean("time")))
        rel = approx / exact - 1.0
        assert 0.0005 < abs(rel) < 0.0012  # measurable, and within the stated bound
        assert float(ratio_range[0]) <= exact / approx <= float(ratio_range[1])


def test_gridder_without_density_keeps_plain_seconds() -> None:
    """inv_density=None is the pre-0022 arithmetic; ones give the same bits."""

    g1 = FootprintGridder((-1.0, 1.0), (-1.0, 1.0), (0.0, 40.0), 1, 2, 2, device="cpu")
    g2 = FootprintGridder((-1.0, 1.0), (-1.0, 1.0), (0.0, 40.0), 1, 2, 2, device="cpu")
    torch.manual_seed(0)
    n = 50
    xyz = torch.stack([torch.rand(n) * 2 - 1, torch.rand(n) * 2 - 1, torch.rand(n) * 60], 1)
    w = torch.rand(n)
    active = torch.rand(n) > 0.3
    zeros = torch.zeros(n, dtype=torch.int64)
    g1.accumulate(xyz, active, w, zeros, zeros, 7.0)
    g2.accumulate(xyz, active, w, zeros, zeros, 7.0, inv_density=torch.ones(n))
    assert torch.equal(g1.tensor, g2.tensor)
    expected = torch.zeros(4)
    valid = active & (xyz[:, 2] > 0) & (xyz[:, 2] <= 40.0)
    idx = ((xyz[:, 1] + 1) / 2 * 2).floor().long() * 2 + ((xyz[:, 0] + 1) / 2 * 2).floor().long()
    expected.index_add_(0, idx[valid], (w * 7.0)[valid])
    torch.testing.assert_close(g1.tensor.view(-1), expected)


def test_density_core_traces_as_one_graph_and_does_not_recompile() -> None:
    """The density core is compiled on CUDA with GLIDE_COMPILE (decision 0004): it must
    trace without graph breaks and must not recompile as the time weight, the met values
    or the per-window level heights change. Checked on CPU with the eager backend."""

    import torch._dynamo

    torch._dynamo.reset()
    prev_sup = torch._dynamo.config.suppress_errors
    prev_rec = torch._dynamo.config.error_on_recompile
    torch._dynamo.config.suppress_errors = False
    torch._dynamo.config.error_on_recompile = True
    try:
        engine = GPUEngine(device="cpu", compile_hot_paths=False)
        dens = ParticleDryAirDensity("cpu")
        dens._core = torch.compile(dens.inverse_density_from_stack, fullgraph=True, backend="eager")
        xyz = torch.stack([torch.zeros(32), torch.zeros(32), torch.rand(32) * 900.0], dim=1)
        win_a = _window(280.0, 282.0, 0.004, 0.005, 100000.0)
        md = win_a.metadata
        win_b = HourlyMetTensors(
            hour_start=win_a.hour_start.float() * 1.01,
            hour_end=win_a.hour_end.float(),
            metadata=MetFieldMetadata(
                lon=md.lon,
                lat=md.lat,
                level=np.array([0.0, 120.0, 1100.0, 5200.0]),
                pressure_level_hpa=md.pressure_level_hpa,
                time_start=md.time_end,
                time_end=md.time_end + timedelta(hours=1),
                variable_units={},
            ),
            channel_names=win_a.channel_names,
        )
        out = [
            dens.inverse_density(xyz, win_a, 0.2, engine),
            dens.inverse_density(xyz, win_a, 0.7, engine),
            dens.inverse_density(xyz, win_b, 0.4, engine),
        ]
        assert all(torch.isfinite(o).all() for o in out)
    finally:
        torch._dynamo.config.suppress_errors = prev_sup
        torch._dynamo.config.error_on_recompile = prev_rec
        torch._dynamo.reset()


def test_conversions_refuse_misaligned_grids() -> None:
    """External review of PR #57: xarray keeps only shared coordinate values, so
    a footprint at longitudes [0, 1] times a field at [1, 2] silently became one
    cell at [1]. Every conversion now refuses fields whose shared coordinates
    differ, and accepts identical ones."""

    from lpdm.comparison import (
        flexpart_to_dry_air_footprint,
        name_to_dry_air_footprint,
        to_stilt_surface_footprint,
    )

    lon_a, lon_b = [0.0, 1.0], [1.0, 2.0]
    fp = xr.DataArray(
        np.ones((2, 2)), dims=("time", "longitude"), coords={"time": [0, 1], "longitude": lon_a}
    )
    good = xr.DataArray(np.full(2, 1.01), dims=("longitude",), coords={"longitude": lon_a})
    bad = xr.DataArray(np.full(2, 1.01), dims=("longitude",), coords={"longitude": lon_b})

    out = flexpart_to_dry_air_footprint(fp, dry_air_factor=good)
    assert list(out.longitude.values) == lon_a
    with pytest.raises(ValueError, match="shared coordinates differ"):
        flexpart_to_dry_air_footprint(fp, dry_air_factor=bad)
    fp_t = fp.expand_dims(lat=[50.0]).transpose("time", "lat", "longitude")
    bad_t = xr.DataArray(
        np.full((2, 1, 2), 1.01),
        dims=("time", "lat", "longitude"),
        coords={"time": [0, 1], "lat": [50.0], "longitude": lon_b},
    )
    with pytest.raises(ValueError, match="shared coordinates differ"):
        flexpart_to_dry_air_footprint(fp_t, dry_air_factor=bad_t)

    ppm_s = fp.isel(time=0, drop=True)
    area = xr.DataArray(np.full(2, 7.0e8), dims=("longitude",), coords={"longitude": lon_a})
    name_to_dry_air_footprint(ppm_s, cell_area_m2=area, dry_air_factor=good)
    with pytest.raises(ValueError, match="shared coordinates differ"):
        name_to_dry_air_footprint(ppm_s, cell_area_m2=area, dry_air_factor=bad)

    raw = xr.DataArray(
        np.ones((1, 1, 1, 2)),
        dims=("time_ago", "z_bin", "latitude", "longitude"),
        coords={
            "z_bin": [20.0],
            "z_bottom_m": ("z_bin", [0.0]),
            "z_top_m": ("z_bin", [40.0]),
            "longitude": lon_a,
        },
        attrs={"density_weighting": "none"},
    )
    rho_bad = xr.DataArray(np.full(2, 1.2), dims=("longitude",), coords={"longitude": lon_b})
    with pytest.raises(ValueError, match="shared coordinates differ"):
        to_stilt_surface_footprint(raw, surface_layer_depth_m=40.0, air_density_kg_m3=rho_bad)
