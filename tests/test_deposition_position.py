"""Deposition position, `output_grid.deposition.position` (task A2, part 1; [PR-F4]).

`endpoint` (the default) counts each step's residence at the particle's position after
the step; `trapezoid` counts half the step at each end (decision 0029). `step_mean`
(decision 0026) was removed by decision 0029. The analytic-plume comparison is in `test_plume_footprint.py`;
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
from lpdm.gpu_engine import GPUEngine
from lpdm.local_frame import DEGREES
from lpdm.main import _deposit_step, _run, _time_ago_index, _time_weight
from lpdm.turbulence.placeholder import PlaceholderConstantOU

M_PER_DEG_LON = 111320.0


def _with_position(cfg, position: str):
    og = cfg.output_grid
    dep = og.deposition.model_copy(update={"position": position})
    return cfg.model_copy(update={"output_grid": og.model_copy(update={"deposition": dep})})


def test_default_position_is_endpoint() -> None:
    cfg = _make_run_config()
    assert cfg.output_grid.deposition.position == "endpoint"


@pytest.mark.parametrize("position", ["start", "midpoint", "step_mean"])
def test_unknown_or_removed_position_is_rejected(position: str) -> None:
    # `midpoint` (the RK2 predictor, decision 0026) and `step_mean` (decision 0029)
    # were removed; the message says why.
    cfg = _make_run_config()
    data = cfg.model_dump(mode="json")
    data["output_grid"]["deposition"] = {"position": position}
    with pytest.raises(pydantic.ValidationError) as err:
        type(cfg).model_validate(data)
    if position != "start":
        assert "was removed" in str(err.value)


def _lon_centroid_and_total(path: Path) -> tuple[float, float, np.ndarray]:
    fp = xr.open_zarr(path / "footprints.zarr")["footprint"]
    by_lon = fp.sum(dim=("release", "time_ago", "z_bin", "latitude")).values
    lon = fp["longitude"].values
    return float((by_lon * lon).sum() / by_lon.sum()), float(by_lon.sum()), fp.values


@pytest.mark.parametrize("other", ["trapezoid"])
@pytest.mark.parametrize("static", [False, True], ids=["dynamic", "static"])
def test_deposit_moves_half_a_step_downwind(
    tmp_path: Path, monkeypatch, static: bool, other: str
) -> None:
    """With uniform wind and no turbulence, every particle-step is counted half a step
    (U dt / 2) less far upwind on average with `trapezoid` than with `endpoint`, and the totals are the same. The placeholder scheme does not fold
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


@pytest.mark.parametrize("position", ["trapezoid"])
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


@pytest.mark.parametrize("other", ["trapezoid"])
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


@pytest.mark.parametrize("position", ["endpoint", "trapezoid"])
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
    the start (density at the start's position and time) standing for the ages
    [e, e + dt/2), and dt/2 at the end (density at the end's position and time) for
    [e + dt/2, e + dt), each half split between age bins by its overlap with them.
    Covers a half that straddles a bin edge (particle 1, ages 80-140 s against an
    edge at 100 s: 20 + 40 s, not the 30 + 30 s of binning each half by one age),
    a particle whose end leaves the grid (it keeps its start half), an inactive
    particle, and a second release whose end half straddles an edge."""

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
        frame=DEGREES,
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
        engine=GPUEngine(device="cpu", frame=DEGREES),
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
    # particle 1: start half ages 80-110 s, 20 s in bin 0 and 10 s in bin 1; end half
    # ages 110-140 s, all in bin 1
    expected[0, 0, 1, 0, 1] += 2.0 * 20.0 * inv(50.0, a_start)
    expected[0, 1, 1, 0, 1] += 2.0 * 10.0 * inv(50.0, a_start)
    expected[0, 1, 0, 0, 2] += 2.0 * h * inv(20.0, a_end)
    # particle 2: its end leaves the grid, so only the start half is counted
    expected[0, 0, 0, 0, 3] += 3.0 * h * inv(20.0, a_start)
    # particle 3: inactive, nothing
    # particle 4: release 1, start half ages 150-180 s in bin 1; end half 180-210 s,
    # 20 s in bin 1 and 10 s in bin 2
    expected[1, 1, 0, 0, 2] += 5.0 * h * inv(5.0, a_start)
    expected[1, 1, 1, 0, 2] += 5.0 * 20.0 * inv(60.0, a_end)
    expected[1, 2, 1, 0, 2] += 5.0 * 10.0 * inv(60.0, a_end)
    torch.testing.assert_close(gridder.tensor, expected, rtol=1e-12, atol=0.0)


def test_time_ago_index_single_bin_lumps_every_age() -> None:
    """A time-integrated footprint (one bin) keeps both halves at any age; with
    several bins the index is the floor of age / width."""

    assert _time_ago_index(torch.tensor([7170.0, 7200.0, 7230.0]), 3600.0, 1).tolist() == [0, 0, 0]
    assert _time_ago_index(torch.tensor([3599.0, 3600.0]), 3600.0, 3).tolist() == [0, 1]


def _deposit_one_step(gridder, position, start, end, age_s=0.0, dt=60.0, bin_s=3600.0) -> None:
    """`_deposit_step` for unit-weight particles of age ``age_s`` (at the step's
    start) in release 0, without density."""

    t0 = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)
    met_window = SimpleNamespace(
        metadata=SimpleNamespace(time_start=t0, time_end=t0 + timedelta(hours=1))
    )
    n = end.shape[0]
    elapsed = torch.full((n,), float(age_s), dtype=torch.float32)
    _deposit_step(
        gridder,
        position=position,
        start_xyz=start,
        end_xyz=end,
        active_mask=torch.ones(n, dtype=torch.bool),
        weights=torch.ones(n, dtype=end.dtype),
        elapsed_s=elapsed,
        t_idx=_time_ago_index(elapsed, bin_s, gridder.n_t),
        time_bin_s=bin_s,
        release_idx=torch.zeros(n, dtype=torch.int64),
        delta_s=dt,
        t_cursor=t0 + timedelta(seconds=1800),
        particle_density=None,
        met_window=met_window,
        t_alpha=0.5,
        engine=GPUEngine(device="cpu", frame=DEGREES),
    )


def _single_cell_gridder(n_time_bins: int = 1) -> FootprintGridder:
    return FootprintGridder(
        lon_bounds=(0.0, 1.0),
        lat_bounds=(0.0, 1.0),
        z_edges_m=(0.0, 40.0, 1000.0),
        n_time_bins=n_time_bins,
        n_y=1,
        n_x=1,
        device="cpu",
        dtype=torch.float64,
        frame=DEGREES,
    )


@pytest.mark.parametrize("position", ["endpoint", "trapezoid"])
def test_a_deposit_on_the_ground_is_counted(position: str) -> None:
    """The review of PR #66: a unit-weight particle moving from 0 to 1 m in 60 s is in
    the 0-40 m layer for the whole step, so the layer records 60 s. The gridder's
    layers were (lo, hi], which dropped the deposit at z = 0 (trapezoid recorded
    30 s). For endpoint the particle moves from 1 m to the ground."""

    gridder = _single_cell_gridder()
    ground = torch.tensor([[0.5, 0.5, 0.0]], dtype=torch.float64)
    above = torch.tensor([[0.5, 0.5, 1.0]], dtype=torch.float64)
    start, end = (ground, above) if position == "trapezoid" else (above, ground)
    _deposit_one_step(gridder, position, start, end)
    assert gridder.tensor[0, 0, 0, 0, 0].item() == 60.0
    assert gridder.tensor.sum().item() == 60.0


_XYZ = torch.tensor([[0.5, 0.5, 10.0]], dtype=torch.float64)


@pytest.mark.parametrize(
    ("age_s", "bin_s", "n_bins", "trapezoid", "endpoint"),
    [
        # The reviewer's refined case: ages 80-140 s against 120 s bins (40 + 20).
        (80.0, 120.0, 2, [40.0, 20.0], [60.0, 0.0]),
        # The first review: an edge at 100 s inside the step (20 + 40).
        (80.0, 100.0, 2, [20.0, 40.0], [60.0, 0.0]),
        # Bins narrower than a half step: six 10 s bins over ages 0-60 s.
        (0.0, 10.0, 6, [10.0] * 6, [60.0] + [0.0] * 5),
        # A step on the grid of the bins: all in one bin, either way.
        (60.0, 120.0, 2, [60.0, 0.0], [60.0, 0.0]),
    ],
)
def test_one_step_is_split_by_age_bin(age_s, bin_s, n_bins, trapezoid, endpoint) -> None:
    """A stationary unit-weight particle, one 60 s step: `trapezoid` splits each half
    step by its overlap with the age bins, so the seconds per bin are exact;
    `endpoint` keeps counting the whole step in the bin of its starting age."""

    for position, expected in (("trapezoid", trapezoid), ("endpoint", endpoint)):
        gridder = _single_cell_gridder(n_time_bins=n_bins)
        _deposit_one_step(gridder, position, _XYZ, _XYZ, age_s=age_s, bin_s=bin_s)
        assert gridder.tensor[0, :, 0, 0, 0].tolist() == expected, position


@pytest.mark.parametrize("offset_s", [0.0, 20.0, 45.0])
@pytest.mark.parametrize(("dt", "bin_s"), [(60.0, 60.0), (60.0, 180.0), (60.0, 25.0)])
def test_stationary_particle_fills_each_age_bin_with_its_width(
    offset_s: float, dt: float, bin_s: float
) -> None:
    """A stationary unit-weight particle followed for many steps, its ages offset
    from the step grid by ``offset_s`` (a release window ending between steps):
    with `trapezoid` every bin the particle spans completely holds exactly its
    width, whatever the alignment, and the first and last bins hold the rest."""

    n_bins = 8
    n_steps = 20
    gridder = _single_cell_gridder(n_time_bins=n_bins)
    for k in range(n_steps):
        _deposit_one_step(
            gridder, "trapezoid", _XYZ, _XYZ, age_s=offset_s + k * dt, dt=dt, bin_s=bin_s
        )
    lo, hi = offset_s, offset_s + n_steps * dt
    edges = np.arange(n_bins + 1) * bin_s
    exact = np.clip(np.minimum(hi, edges[1:]) - np.maximum(lo, edges[:-1]), 0.0, None)
    np.testing.assert_allclose(gridder.tensor[0, :, 0, 0, 0].numpy(), exact, rtol=0, atol=1e-9)


def test_time_integrated_footprint_takes_any_age() -> None:
    """One bin (time-integrated): no splitting, every age counts in full."""

    gridder = _single_cell_gridder(n_time_bins=1)
    _deposit_one_step(gridder, "trapezoid", _XYZ, _XYZ, age_s=1e6 + 17.0, bin_s=7.0)
    assert gridder.tensor.sum().item() == 60.0


@pytest.mark.parametrize("static", [False, True], ids=["dynamic", "static"])
def test_trapezoid_time_ago_bins_match_endpoint_totals(
    tmp_path: Path, monkeypatch, static: bool
) -> None:
    """End to end with time-ago bins and dry-air weighting: with steps aligned to the
    bins (a 300 s release window and 300 s steps), each step lies in one bin, so the
    per-bin totals (summed over space) match the endpoint's up to the density's change
    along a step; and the first steps' start halves are counted in the release column,
    which endpoint never counts. (The helper's default 60 s window is not aligned with
    300 s steps; misaligned windows are tested in `test_tracer_benchmark.py`.)"""

    monkeypatch.setattr(PlaceholderConstantOU, "sigma_w2", 0.0)
    monkeypatch.setenv("GLIDE_STATIC_SUBSTEPS", "1" if static else "0")
    fps = {}
    for position in ("endpoint", "trapezoid"):
        cfg = _make_run_config(
            output_uri=str(tmp_path / position),
            dt_seconds=300,
            release_duration_seconds=300,
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


def test_float64_accumulation_matches_float32_and_writes_float32(tmp_path: Path) -> None:
    """`output_grid.accumulation_dtype: float64` (prototype, off by default) sums in
    double precision and rounds once at write: the store stays float32, the default
    is float32, and on a short run the two agree to float32 rounding."""

    assert _make_run_config().output_grid.accumulation_dtype == "float32"
    fps = {}
    for dtype in ("float32", "float64"):
        cfg = _make_run_config(output_uri=str(tmp_path / dtype), simulation_length_seconds=3600)
        og = cfg.output_grid.model_copy(update={"accumulation_dtype": dtype})
        cfg = _with_position(cfg.model_copy(update={"output_grid": og}), "trapezoid")
        _run(cfg, reader=_make_reader(cfg, wind_fn=lambda _: (5.0, 0.0, 0.0)))
        fp = xr.open_zarr(tmp_path / dtype / "footprints.zarr")["footprint"]
        assert fp.dtype == np.float32
        fps[dtype] = fp.values.astype(np.float64)
        meta = json.loads((tmp_path / dtype / "run_metadata.json").read_text())
        assert meta["footprint_units"]["accumulation_dtype"] == dtype
    assert fps["float32"].sum() > 0
    np.testing.assert_allclose(
        fps["float64"], fps["float32"], rtol=1e-5, atol=1e-6 * fps["float32"].max()
    )
