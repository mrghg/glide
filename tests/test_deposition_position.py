"""Deposition position, `output_grid.deposition.position` (task A2, part 1; [PR-F4]).

`endpoint` (the default) counts each step's residence at the particle's position after
the step; `step_mean` counts it at the mean of the particle's positions at the start
and at the end of the step (decision 0026). The analytic-plume comparison of the two
is in `test_plume_footprint.py`.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pydantic
import pytest
import torch
import xarray as xr
from test_main_runtime import _make_hanna_reader, _make_reader, _make_run_config

from lpdm.main import _run, _step_mean_position
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


@pytest.mark.parametrize("static", [False, True], ids=["dynamic", "static"])
def test_step_mean_moves_each_deposit_half_a_step_downwind(
    tmp_path: Path, monkeypatch, static: bool
) -> None:
    """With uniform wind and no turbulence, every particle-step is counted half a step
    (U dt / 2) less far upwind with `step_mean` than with `endpoint`, and the totals
    are the same. The placeholder scheme does not fold advection, so on the static
    path this exercises the runtime's own full-set advection."""

    monkeypatch.setattr(PlaceholderConstantOU, "sigma_w2", 0.0)
    monkeypatch.setenv("GLIDE_STATIC_SUBSTEPS", "1" if static else "0")
    u, dt = 5.0, 300
    centroids, totals = {}, {}
    for position in ("endpoint", "step_mean"):
        cfg = _make_run_config(
            output_uri=str(tmp_path / position),
            dt_seconds=dt,
            simulation_length_seconds=3600,
            n_time_bins=1,
            output_n_x=400,  # 0.01 deg ~ 1.1 km cells, finer than U dt / 2 = 750 m
        )
        cfg = _with_position(cfg, position)
        _run(cfg, reader=_make_reader(cfg, wind_fn=lambda _: (u, 0.0, 0.0)))
        centroids[position], totals[position], _ = _lon_centroid_and_total(tmp_path / position)

    assert totals["step_mean"] == pytest.approx(totals["endpoint"], rel=1e-6)
    shift_deg = 0.5 * u * dt / M_PER_DEG_LON
    # Cell centres quantise each deposit by up to half a cell; the shift is the same
    # for every particle-step, so the centroid difference is within one cell of it.
    assert centroids["step_mean"] - centroids["endpoint"] == pytest.approx(shift_deg, abs=0.01)


def test_step_mean_static_and_dynamic_paths_agree(tmp_path: Path, monkeypatch) -> None:
    """With a deterministic step (no turbulence), the static (full-buffer) and dynamic
    (active-subset) paths deposit at the same step-mean positions."""

    monkeypatch.setattr(PlaceholderConstantOU, "sigma_w2", 0.0)
    fps = {}
    for static in (False, True):
        monkeypatch.setenv("GLIDE_STATIC_SUBSTEPS", "1" if static else "0")
        out = tmp_path / f"static{int(static)}"
        cfg = _make_run_config(
            output_uri=str(out), dt_seconds=300, simulation_length_seconds=3600, n_time_bins=1
        )
        cfg = _with_position(cfg, "step_mean")
        _run(cfg, reader=_make_reader(cfg, wind_fn=lambda _: (5.0, 2.0, 0.0)))
        fps[static] = _lon_centroid_and_total(out)[2]
    assert fps[True].sum() > 0
    np.testing.assert_allclose(fps[True], fps[False], rtol=1e-6, atol=1e-6 * fps[False].max())


@pytest.mark.parametrize("static", [False, True], ids=["dynamic", "static"])
def test_hanna_step_mean_run_conserves_residence(tmp_path: Path, monkeypatch, static) -> None:
    """End to end with Hanna (on the static path, advection folded into the scheme's
    captured step): a step-mean run completes, and with the grid covering the plume
    the total is the same as the endpoint run's (the position moves residence, never
    removes it)."""

    monkeypatch.setenv("GLIDE_STATIC_SUBSTEPS", "1" if static else "0")
    totals = {}
    for position in ("endpoint", "step_mean"):
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
    assert np.isfinite(totals["step_mean"]) and totals["step_mean"] > 0
    assert totals["step_mean"] == pytest.approx(totals["endpoint"], rel=1e-5)


@pytest.mark.parametrize("position", ["endpoint", "step_mean"])
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
