"""Deposition position, `output_grid.deposition.position` (task A2, part 1; [PR-F4]).

`endpoint` (the default) counts each step's residence at the particle's position after
the step; `midpoint` counts it at the RK2 advection midpoint x - dt/2 u(x). The
analytic-plume comparison of the two is in `test_plume_footprint.py`.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pydantic
import pytest
import torch
import xarray as xr
from test_main_runtime import (
    _make_hanna_reader,
    _make_reader,
    _make_run_config,
    _wmc_met_window,
)

from lpdm.gpu_engine import GPUEngine
from lpdm.main import _advect_active_particles, _midpoint_deposition_position, _run
from lpdm.turbulence import HannaScheme
from lpdm.turbulence.placeholder import PlaceholderConstantOU

M_PER_DEG_LON = 111320.0


def _with_position(cfg, position: str):
    og = cfg.output_grid
    dep = og.deposition.model_copy(update={"position": position})
    return cfg.model_copy(update={"output_grid": og.model_copy(update={"deposition": dep})})


def test_default_position_is_endpoint() -> None:
    cfg = _make_run_config()
    assert cfg.output_grid.deposition.position == "endpoint"


def test_unknown_position_is_rejected() -> None:
    cfg = _make_run_config()
    data = cfg.model_dump(mode="json")
    data["output_grid"]["deposition"] = {"position": "start"}
    with pytest.raises(pydantic.ValidationError):
        type(cfg).model_validate(data)


def test_midpoint_reflects_heights_below_ground() -> None:
    x_mid = torch.tensor([[1.0, 2.0, -3.0], [4.0, 5.0, 6.0]])
    out = _midpoint_deposition_position(x_mid)
    assert torch.equal(out, torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))


@pytest.mark.parametrize("static", [False, True], ids=["dynamic", "static"])
def test_midpoint_moves_each_deposit_half_a_step_downwind(
    tmp_path: Path, monkeypatch, static: bool
) -> None:
    """With uniform wind and no turbulence, every particle-step is counted half a step
    (U dt / 2) less far upwind with `midpoint` than with `endpoint`, and the totals are
    the same. The placeholder scheme does not fold advection, so on the static path
    this exercises the runtime's own full-set advection."""

    monkeypatch.setattr(PlaceholderConstantOU, "sigma_w2", 0.0)
    monkeypatch.setenv("GLIDE_STATIC_SUBSTEPS", "1" if static else "0")
    u, dt = 5.0, 300
    centroids, totals = {}, {}
    for position in ("endpoint", "midpoint"):
        cfg = _make_run_config(
            output_uri=str(tmp_path / position),
            dt_seconds=dt,
            simulation_length_seconds=3600,
            n_time_bins=1,
            output_n_x=400,  # 0.01 deg ~ 1.1 km cells, finer than U dt / 2 = 750 m
        )
        cfg = _with_position(cfg, position)
        _run(cfg, reader=_make_reader(cfg, wind_fn=lambda _: (u, 0.0, 0.0)))
        fp = xr.open_zarr(tmp_path / position / "footprints.zarr")["footprint"]
        by_lon = fp.sum(dim=("release", "time_ago", "z_bin", "latitude")).values
        lon = fp["longitude"].values
        totals[position] = float(by_lon.sum())
        centroids[position] = float((by_lon * lon).sum() / by_lon.sum())

    assert totals["midpoint"] == pytest.approx(totals["endpoint"], rel=1e-6)
    shift_deg = 0.5 * u * dt / M_PER_DEG_LON
    # Cell centres quantise each deposit by up to half a cell; the shift is the same
    # for every particle-step, so the centroid difference is within one cell of it.
    assert centroids["midpoint"] - centroids["endpoint"] == pytest.approx(shift_deg, abs=0.01)


def _sheared_window():
    """`_wmc_met_window` with a wind that varies in x, y and z, so the RK2 midpoint
    is not a plain translation."""

    met = _wmc_met_window(blh_m=2000.0, ustar_m_s=0.4)
    fields = met.hour_start.clone()
    n_lev, n_lat, n_lon = fields.shape[1:]
    z = torch.linspace(0.0, 1.0, n_lev).view(-1, 1, 1)
    y = torch.linspace(0.0, 1.0, n_lat).view(1, -1, 1)
    x = torch.linspace(0.0, 1.0, n_lon).view(1, 1, -1)
    fields[0] = 5.0 + 10.0 * z + 3.0 * y  # u
    fields[1] = -2.0 + 4.0 * x  # v
    fields[2] = 0.05 * (x - 0.5)  # w
    end = fields.clone()
    end[0] += 2.0
    return dataclasses.replace(met, hour_start=fields, hour_end=end)


def test_hanna_static_midpoint_matches_runtime_advection_midpoint() -> None:
    """The midpoint `HannaScheme` returns from its folded (static-path) advection is the
    same RK2 predictor the runtime's dynamic-path advection computes."""

    from datetime import timedelta

    torch.manual_seed(3)
    engine = GPUEngine(device="cpu")
    met = _sheared_window()
    n = 64
    particles = torch.zeros(n, 4, dtype=torch.float32)
    particles[:, 0] = torch.rand(n) * 2.0 - 1.0
    particles[:, 1] = 42.0 + torch.rand(n) * 6.0
    particles[:, 2] = torch.rand(n) * 2000.0 + 10.0
    particles[:, 3] = 1.0
    dt = 300.0
    t_cursor = met.metadata.time_start + timedelta(minutes=40)

    mids: list[torch.Tensor] = []
    _, t_alpha, _ = _advect_active_particles(
        engine, engine.device, particles.clone(), met, t_cursor, dt, torch.float32, None, mids
    )

    scheme = HannaScheme(meander_enabled=True, static_substeps=True)
    scheme.record_advection_midpoint = True
    state = scheme.initialize_state(n, device=torch.device("cpu"), dtype=torch.float32)
    scheme.step(
        particles.clone(),
        state,
        met,
        t_alpha=t_alpha,
        dt_seconds=dt,
        active_mask=torch.ones(n, dtype=torch.bool),
        engine=engine,
    )
    assert scheme.advection_midpoint is not None
    torch.testing.assert_close(scheme.advection_midpoint, mids[0], rtol=1e-6, atol=1e-5)
    # Not the trivial answer: the predictor moved the particles.
    assert not torch.allclose(mids[0], particles[:, :3])


@pytest.mark.parametrize("static", [False, True], ids=["dynamic", "static"])
def test_hanna_midpoint_run_conserves_residence(tmp_path: Path, monkeypatch, static) -> None:
    """End to end with Hanna (on the static path, advection folded into the scheme):
    a midpoint run completes, and with the grid covering the plume the total is the
    same as the endpoint run's (the position moves residence, never removes it)."""

    monkeypatch.setenv("GLIDE_STATIC_SUBSTEPS", "1" if static else "0")
    totals = {}
    for position in ("endpoint", "midpoint"):
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
    assert np.isfinite(totals["midpoint"]) and totals["midpoint"] > 0
    assert totals["midpoint"] == pytest.approx(totals["endpoint"], rel=1e-5)


@pytest.mark.parametrize("position", ["endpoint", "midpoint"])
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
