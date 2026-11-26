"""Deposition position, `output_grid.deposition.position` (task A2, part 1; [PR-F4]).

`endpoint` (the default) counts each step's residence at the particle's position after
the step; `step_mean` counts it at the mean of the particle's positions at the start
and at the end of the step (decision 0026); `trapezoid` counts half the step at each
end (decision 0029). The analytic-plume comparison is in `test_plume_footprint.py`;
the fine-time residence scoring of all three is
`scripts/validation/score_residence_counting.py`.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pydantic
import pytest
import torch
import xarray as xr
from test_main_runtime import _make_hanna_reader, _make_reader, _make_run_config

from lpdm.footprint_gridder import FootprintGridder
from lpdm.main import _deposit_step, _run, _step_mean_position, _time_ago_index, _time_weight
from lpdm.turbulence.placeholder import PlaceholderConstantOU

M_PER_DEG_LON = 111320.0


def _with_position(cfg, position: str):
    og = cfg.output_grid
    dep = og.deposition.model_copy(update={"position": position})
    return cfg.model_copy(update={"output_grid": og.model_copy(update={"deposition": dep})})


def test_default_position_is_endpoint() -> None:
    cfg = _make_run_config()
    assert cfg.output_grid.deposition.position == "endpoint"


@pytest.mark.parametrize("position", ["start", "midpoint"])
def test_unknown_position_is_rejected(position: str) -> None:
    # `midpoint` (the RK2 predictor) was replaced by `step_mean` (decision 0026).
    cfg = _make_run_config()
    data = cfg.model_dump(mode="json")
    data["output_grid"]["deposition"] = {"position": position}
    with pytest.raises(pydantic.ValidationError):
        type(cfg).model_validate(data)


def test_step_mean_is_the_mean_of_start_and_end() -> None:
    start = torch.tensor([[1.0, 50.0, 10.0], [-3.0, 40.0, 0.0]], dtype=torch.float64)
    end = torch.tensor([[2.0, 52.0, 30.0], [-4.0, 39.0, 5.0]], dtype=torch.float64)
    out = _step_mean_position(start, end)
    torch.testing.assert_close(out, 0.5 * (start + end), rtol=0, atol=1e-12)


def test_step_mean_height_stays_above_ground() -> None:
    # Both ends reflected (z >= 0), so the mean is too, including a step that ends
    # on the ground.
    start = torch.tensor([[0.0, 0.0, 3.0], [0.0, 0.0, 0.0]])
    end = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 7.0]])
    z = _step_mean_position(start, end)[:, 2]
    assert torch.equal(z, torch.tensor([1.5, 3.5]))


@pytest.mark.parametrize(
    ("lon_start", "lon_end", "expected"),
    [
        (179.9, -179.9, 180.0),  # across the dateline, eastward
        (-179.9, 179.9, -180.0),  # across the dateline, westward
        (359.9, 0.1, 360.0),  # across the prime meridian in 0..360
        (0.1, 359.9, 0.0),
        (-10.0, -12.0, -11.0),  # no crossing: the plain mean
    ],
)
def test_step_mean_longitude_takes_the_shorter_arc(
    lon_start: float, lon_end: float, expected: float
) -> None:
    start = torch.tensor([[lon_start, 10.0, 100.0]], dtype=torch.float64)
    end = torch.tensor([[lon_end, 12.0, 300.0]], dtype=torch.float64)
    out = _step_mean_position(start, end)
    assert out[0, 0].item() == pytest.approx(expected, abs=1e-9)
    assert out[0, 1].item() == pytest.approx(11.0)
    assert out[0, 2].item() == pytest.approx(200.0)


def _lon_centroid_and_total(path: Path) -> tuple[float, float, np.ndarray]:
    fp = xr.open_zarr(path / "footprints.zarr")["footprint"]
    by_lon = fp.sum(dim=("release", "time_ago", "z_bin", "latitude")).values
    lon = fp["longitude"].values
    return float((by_lon * lon).sum() / by_lon.sum()), float(by_lon.sum()), fp.values


@pytest.mark.parametrize("other", ["step_mean", "trapezoid"])
@pytest.mark.parametrize("static", [False, True], ids=["dynamic", "static"])
def test_deposit_moves_half_a_step_downwind(
    tmp_path: Path, monkeypatch, static: bool, other: str
) -> None:
    """With uniform wind and no turbulence, every particle-step is counted half a step
    (U dt / 2) less far upwind on average with `step_mean` or `trapezoid` than with
    `endpoint`, and the totals are the same. The placeholder scheme does not fold
    advection, so on the static path this exercises the runtime's own full-set
    advection."""

    monkeypatch.setattr(PlaceholderConstantOU, "sigma_w2", 0.0)
    monkeypatch.setenv("GLIDE_STATIC_SUBSTEPS", "1" if static else "0")
    u, dt = 5.0, 300
    centroids, totals = {}, {}
    for position in ("endpoint", other):
        cfg = _make_run_config(
            output_uri=str(tmp_path / position),
            dt_seconds=dt,
            simulation_length_seconds=3600,
            n_time_bins=1,
            output_n_x=4000,  # 0.001 deg ~ 110 m cells, a tenth of U dt / 2 = 750 m
        )
        cfg = _with_position(cfg, position)
        _run(cfg, reader=_make_reader(cfg, wind_fn=lambda _: (u, 0.0, 0.0)))
        centroids[position], totals[position], _ = _lon_centroid_and_total(tmp_path / position)

    assert totals[other] == pytest.approx(totals["endpoint"], rel=1e-6)
    shift_deg = 0.5 * u * dt / M_PER_DEG_LON
    # Cell centres quantise each deposit by up to half a cell, so the centroid
    # difference is within one cell (0.001 deg) of the half step (0.0067 deg). The
    # tolerance, 0.3 of the shift, fails a zero shift and a whole-step (double) one.
    shift = centroids[other] - centroids["endpoint"]
    assert shift == pytest.approx(shift_deg, abs=0.3 * shift_deg)
    assert abs(shift) > 0.5 * shift_deg and abs(shift) < 1.5 * shift_deg


@pytest.mark.parametrize("position", ["step_mean", "trapezoid"])
def test_static_and_dynamic_paths_agree(tmp_path: Path, monkeypatch, position: str) -> None:
    """With a deterministic step (no turbulence), the static (full-buffer) and dynamic
    (active-subset) paths deposit at the same positions."""

    monkeypatch.setattr(PlaceholderConstantOU, "sigma_w2", 0.0)
    fps = {}
    for static in (False, True):
        monkeypatch.setenv("GLIDE_STATIC_SUBSTEPS", "1" if static else "0")
        out = tmp_path / f"static{int(static)}"
        cfg = _make_run_config(
            output_uri=str(out), dt_seconds=300, simulation_length_seconds=3600, n_time_bins=1
        )
        cfg = _with_position(cfg, position)
        _run(cfg, reader=_make_reader(cfg, wind_fn=lambda _: (5.0, 2.0, 0.0)))
        fps[static] = _lon_centroid_and_total(out)[2]
    assert fps[True].sum() > 0
    np.testing.assert_allclose(fps[True], fps[False], rtol=1e-6, atol=1e-6 * fps[False].max())


@pytest.mark.parametrize("other", ["step_mean", "trapezoid"])
@pytest.mark.parametrize("static", [False, True], ids=["dynamic", "static"])
def test_hanna_run_conserves_residence(tmp_path: Path, monkeypatch, static, other) -> None:
    """End to end with Hanna (on the static path, advection folded into the scheme's
    captured step): a step-mean or trapezoid run completes, and with the grid covering
    the plume the total is the same as the endpoint run's (the position moves
    residence, never removes it)."""

    monkeypatch.setenv("GLIDE_STATIC_SUBSTEPS", "1" if static else "0")
    totals = {}
    for position in ("endpoint", other):
        cfg = _make_run_config(
            output_uri=str(tmp_path / position),
            turbulence_scheme="hanna_1982",
            simulation_length_seconds=3600,
            dt_seconds=300,
            n_time_bins=1,
            release_alt_agl_m=20.0,
        )
        cfg = _with_position(cfg, position)
        reader = _make_hanna_reader(cfg, wind_fn=lambda _: (5.0, 0.0, 0.0))
        meta = _run(cfg, reader=reader)
        assert meta["runtime"]["status"] == "completed"
        fp = xr.open_zarr(tmp_path / position / "footprints.zarr")["footprint"]
        totals[position] = float(fp.sum())
    assert np.isfinite(totals[other]) and totals[other] > 0
    assert totals[other] == pytest.approx(totals["endpoint"], rel=1e-5)


@pytest.mark.parametrize("position", ["endpoint", "step_mean", "trapezoid"])
def test_store_and_metadata_record_the_position(tmp_path: Path, position: str) -> None:
    out = tmp_path / position
    cfg = _with_position(
        _make_run_config(output_uri=str(out), simulation_length_seconds=3600), position
    )
    _run(cfg, reader=_make_reader(cfg, wind_fn=lambda _: (5.0, 0.0, 0.0)))
    ds = xr.open_zarr(out / "footprints.zarr")
    for attrs in (ds.attrs, ds["footprint"].attrs):
        assert attrs["deposition_position"] == position
        assert attrs["deposition_position_description"]
    meta = json.loads((out / "run_metadata.json").read_text())
    assert meta["footprint_units"]["deposition_position"] == position
    assert meta["config"]["output_grid"]["deposition"]["position"] == position


# ---------------------------------------------------------------------------
# Trapezoid (decision 0029): half of each step at its start, half at its end
# ---------------------------------------------------------------------------


class _RecordingDensity:
    """Stands in for `ParticleDryAirDensity`: 1/rho = 1 + z/100 + 10 alpha, and records
    the time weights it was asked for."""

    def __init__(self) -> None:
        self.alphas: list[float] = []

    def inverse_density(self, xyz, met_window, t_alpha, engine):
        self.alphas.append(float(t_alpha))
        return 1.0 + xyz[:, 2] / 100.0 + 10.0 * float(t_alpha)


def test_trapezoid_deposits_each_half_with_its_own_bin_and_density() -> None:
    """The literal rule on a multi-bin, multi-release, density-weighted grid: dt/2 at
    the start (bin of the age at the step's start, density at the start's position
    and time) and dt/2 at the end (bin of the age half a step later, density at the
    end's position and time). Covers a step that straddles a bin edge, a particle
    whose end leaves the grid (it keeps its start half), an inactive particle and a
    second release."""

    gridder = FootprintGridder(
        lon_bounds=(0.0, 4.0),
        lat_bounds=(0.0, 1.0),
        z_edges_m=(0.0, 40.0, 1000.0),
        n_time_bins=3,
        n_y=1,
        n_x=4,
        n_releases=2,
        device="cpu",
        dtype=torch.float64,
    )
    t0 = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)
    met_window = SimpleNamespace(
        metadata=SimpleNamespace(time_start=t0, time_end=t0 + timedelta(hours=1))
    )
    t_cursor = t0 + timedelta(seconds=1800)  # weight 0.5 at the start, 1740/3600 at the end
    dt, bin_s = 60.0, 100.0
    # lon, lat, z at the start and at the end of the step
    start = torch.tensor(
        [[0.5, 0.5, 10.0], [1.5, 0.5, 50.0], [3.5, 0.5, 20.0], [0.5, 0.5, 10.0], [2.5, 0.5, 5.0]],
        dtype=torch.float64,
    )
    end = torch.tensor(
        [[1.5, 0.5, 30.0], [2.5, 0.5, 20.0], [4.5, 0.5, 20.0], [1.5, 0.5, 10.0], [2.5, 0.5, 60.0]],
        dtype=torch.float64,
    )
    active = torch.tensor([True, True, True, False, True])
    weights = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float64)
    elapsed = torch.tensor([0.0, 80.0, 0.0, 0.0, 150.0], dtype=torch.float64)
    release_idx = torch.tensor([0, 0, 0, 0, 1])
    density = _RecordingDensity()
    _deposit_step(
        gridder,
        position="trapezoid",
        start_xyz=start,
        end_xyz=end,
        active_mask=active,
        weights=weights,
        elapsed_s=elapsed,
        t_idx=_time_ago_index(elapsed, bin_s, gridder.n_t),
        time_bin_s=bin_s,
        release_idx=release_idx,
        delta_s=dt,
        t_cursor=t_cursor,
        particle_density=density,
        met_window=met_window,
        t_alpha=0.123,  # the step's midpoint weight: the trapezoid must not use it
        engine=None,
    )
    a_start = _time_weight(met_window, t_cursor.timestamp())
    a_end = _time_weight(met_window, t_cursor.timestamp() - dt)
    assert density.alphas == [a_start, a_end]
    assert a_start == pytest.approx(0.5) and a_end == pytest.approx(1740.0 / 3600.0)

    def inv(z, a):
        return 1.0 + z / 100.0 + 10.0 * a

    h = 0.5 * dt
    expected = torch.zeros_like(gridder.tensor)
    # particle 0: age 0, both halves in bin 0; start in x cell 0, end in x cell 1
    expected[0, 0, 0, 0, 0] += 1.0 * h * inv(10.0, a_start)
    expected[0, 0, 0, 0, 1] += 1.0 * h * inv(30.0, a_end)
    # particle 1: age 80, start half in bin 0 (80 s), end half in bin 1 (110 s)
    expected[0, 0, 1, 0, 1] += 2.0 * h * inv(50.0, a_start)
    expected[0, 1, 0, 0, 2] += 2.0 * h * inv(20.0, a_end)
    # particle 2: its end leaves the grid, so only the start half is counted
    expected[0, 0, 0, 0, 3] += 3.0 * h * inv(20.0, a_start)
    # particle 3: inactive, nothing
    # particle 4: release 1, age 150, both halves in bin 1 (150 s, 180 s)
    expected[1, 1, 0, 0, 2] += 5.0 * h * inv(5.0, a_start)
    expected[1, 1, 1, 0, 2] += 5.0 * h * inv(60.0, a_end)
    torch.testing.assert_close(gridder.tensor, expected, rtol=1e-12, atol=0.0)


def test_time_ago_index_single_bin_lumps_every_age() -> None:
    """A time-integrated footprint (one bin) keeps both halves at any age; with
    several bins the index is the floor of age / width."""

    assert _time_ago_index(torch.tensor([7170.0, 7200.0, 7230.0]), 3600.0, 1).tolist() == [0, 0, 0]
    assert _time_ago_index(torch.tensor([3599.0, 3600.0]), 3600.0, 3).tolist() == [0, 1]


@pytest.mark.parametrize("static", [False, True], ids=["dynamic", "static"])
def test_trapezoid_time_ago_bins_match_endpoint_totals(
    tmp_path: Path, monkeypatch, static: bool
) -> None:
    """End to end with time-ago bins and dry-air weighting: with steps aligned to the
    bins, each step's two halves fall in its own bin, so the per-bin totals (summed
    over space) match the endpoint's up to the density's change along a step; and the
    first steps' start halves are counted in the release column, which endpoint never
    counts."""

    monkeypatch.setattr(PlaceholderConstantOU, "sigma_w2", 0.0)
    monkeypatch.setenv("GLIDE_STATIC_SUBSTEPS", "1" if static else "0")
    fps = {}
    for position in ("endpoint", "trapezoid"):
        cfg = _make_run_config(
            output_uri=str(tmp_path / position),
            dt_seconds=300,
            simulation_length_seconds=3 * 3600,
            n_time_bins=3,
            density_weighting="dry_air",
            output_n_x=400,  # 0.01 deg cells; the release sits inside cell 200
            release_lon=0.0025,
        )
        cfg = _with_position(cfg, position)
        reader = _make_reader(cfg, wind_fn=lambda _: (5.0, 0.0, 0.0))
        reader.channel_names = ("u", "v", "w", "blh", "sp", "t", "q")
        reader.q_kg_kg = 0.004
        _run(cfg, reader=reader)
        fps[position] = xr.open_zarr(tmp_path / position / "footprints.zarr")["footprint"]
    per_bin = {
        p: fp.sum(dim=("release", "z_bin", "latitude", "longitude")).values for p, fp in fps.items()
    }
    assert per_bin["endpoint"][:3].min() > 0
    np.testing.assert_allclose(per_bin["trapezoid"], per_bin["endpoint"], rtol=1e-3)
    by_lon = {
        p: fp.isel(time_ago=0).sum(dim=("release", "z_bin", "latitude")).values
        for p, fp in fps.items()
    }
    release_col = 200
    assert by_lon["endpoint"][release_col] == 0.0
    assert by_lon["trapezoid"][release_col] > 0.0
