"""Tests for the tracer benchmark (scripts/validation/tracers, task B12).

The footprint-to-concentration step is checked against hand calculations and,
end to end, against the exact forward solution for uniform wind with no
turbulence, where GLIDE runs backward from a sample on synthetic meteorology.
Tests that read DATEM's OKC80 files skip when ``$GLIDE_DATEM`` is absent.
GLIDE is never run on a Tier 3 experiment.
"""

from __future__ import annotations

import json
import math
import os
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from test_main_runtime import _make_reader, _make_run_config

from lpdm.config import RunConfig
from lpdm.main import _run
from lpdm.turbulence.placeholder import PlaceholderConstantOU

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "validation"))

from tracers import benchmark, concentration  # noqa: E402
from tracers.concentration import Emission  # noqa: E402


def _datem_dir() -> Path | None:
    d = os.environ.get("GLIDE_DATEM")
    p = Path(d) if d else Path(__file__).resolve().parents[1] / "data" / "DATEM"
    return p if (p / "exp_data" / "okc80" / "meas-t1.txt").exists() else None


needs_datem = pytest.mark.skipif(_datem_dir() is None, reason="DATEM data not available")

T = pd.Timestamp


# ---------------------------------------------------------------------------
# pieces with hand-computed answers
# ---------------------------------------------------------------------------


def test_cell_area_matches_the_model_metric() -> None:
    # A 1° x 1° cell at the equator: 110540 m x 111320 m to within the curvature.
    a = concentration.cell_area_m2(np.array([-0.5, 0.5]), np.array([0.0, 1.0]))
    assert a.shape == (1, 1)
    assert a[0, 0] == pytest.approx(110540.0 * 111320.0, rel=1e-4)
    # At 60° N the zonal extent halves.
    b = concentration.cell_area_m2(np.array([59.5, 60.5]), np.array([0.0, 1.0]))
    assert b[0, 0] / a[0, 0] == pytest.approx(0.5, rel=1e-3)


def test_source_box_weights_cover_partial_cells() -> None:
    edges = np.array([-0.375, -0.125, 0.125, 0.375])
    w = concentration.source_box_weights(edges, edges, 0.0, 0.0, 0.25)
    np.testing.assert_allclose(w, [[0, 0, 0], [0, 1, 0], [0, 0, 0]])
    w1 = concentration.source_box_weights(edges, edges, 0.0, 0.0, 0.5)
    # 0.5° box: the centre cell, and half of each neighbour in each direction.
    np.testing.assert_allclose(w1[1], [0.5, 1.0, 0.5])
    np.testing.assert_allclose(w1[0], [0.25, 0.5, 0.25])
    with pytest.raises(ValueError, match="does not overlap"):
        concentration.source_box_weights(edges, edges, 5.0, 5.0, 0.25)


def test_emission_overlap_follows_ages_back_from_the_window_end() -> None:
    # Window ends 12:00; 30-minute bins: bin k covers (12:00 - (k+1)30 min, 12:00 - k 30 min].
    end = np.array(["2000-01-01T12:00"], dtype="datetime64[ns]")
    edges = np.arange(0, 5) * 1800.0
    f = concentration.emission_overlap(
        end, edges[:-1], edges[1:], T("2000-01-01 10:45"), T("2000-01-01 11:30")
    )
    # Bins: 11:30-12:00 none; 11:00-11:30 all; 10:30-11:00 half (10:45-11:00); 10:00-10:30 none.
    np.testing.assert_allclose(f, [[0.0, 1.0, 0.5, 0.0]])


def _grid() -> dict:
    edges = np.array([-0.375, -0.125, 0.125, 0.375])
    return dict(lat_edges=edges, lon_edges=edges, layer_depth_m=40.0)


def test_predict_by_hand() -> None:
    """Two releases, residence in the centre cell only, on a 15-minute age axis."""

    n_t = 8
    fp = np.zeros((2, n_t, 3, 3))
    fp[0, :, 1, 1] = 100.0  # 100 s per bin in the source cell
    fp[1, :, 1, 1] = np.arange(n_t)  # k s in bin k
    fp[:, :, 0, 0] = 1e6  # outside the 0.25° box: ignored
    edges = np.arange(n_t + 1) * 900.0
    ends = np.array(["2000-01-01T12:00", "2000-01-01T13:00"], dtype="datetime64[ns]")
    e = Emission(T("2000-01-01 10:30"), T("2000-01-01 11:00"), mass=3.6, lat=0.0, lon=0.0)
    c = concentration.predict(
        fp,
        window_end=ends,
        age_start_s=edges[:-1],
        age_end_s=edges[1:],
        emissions=[e],
        source_deg=0.25,
        basis="ambient",
        **_grid(),
    )
    volume = concentration.cell_area_m2([-0.125, 0.125], [-0.125, 0.125])[0, 0] * 40.0
    rate = 3.6 / 1800.0
    # Release 0 (ends 12:00): 10:30-11:00 is ages 60-90 min, bins 4 and 5.
    # Release 1 (ends 13:00): ages 120-150 min, bins 8 and 9, beyond the axis: zero.
    np.testing.assert_allclose(c, [rate * 200.0 / volume, 0.0], rtol=1e-12)

    # Bins 4 and 5 of release 1 for an emission an hour later.
    e2 = Emission(T("2000-01-01 11:30"), T("2000-01-01 12:00"), mass=3.6, lat=0.0, lon=0.0)
    c2 = concentration.predict(
        fp,
        window_end=ends,
        age_start_s=edges[:-1],
        age_end_s=edges[1:],
        emissions=[e, e2],
        source_deg=0.25,
        basis="ambient",
        **_grid(),
    )
    np.testing.assert_allclose(c2[1], rate * (4.0 + 5.0) / volume, rtol=1e-12)

    # Basis "stp": divided by the source density, times M_air / V_m.
    rho = 1.1
    c3 = concentration.predict(
        fp,
        window_end=ends,
        age_start_s=edges[:-1],
        age_end_s=edges[1:],
        emissions=[e],
        source_deg=0.25,
        basis="stp",
        density=lambda t: np.full(len(t), rho),
        **_grid(),
    )
    np.testing.assert_allclose(c3, c * (0.02897 / 0.0224) / rho, rtol=1e-12)
    # A footprint already divided by the density (task A04) needs no meteorology.
    c4 = concentration.predict(
        fp / rho,
        window_end=ends,
        age_start_s=edges[:-1],
        age_end_s=edges[1:],
        emissions=[e],
        source_deg=0.25,
        basis="stp",
        density_weighted=True,
        **_grid(),
    )
    np.testing.assert_allclose(c4, c3, rtol=1e-12)


def test_predict_refuses_what_it_cannot_do() -> None:
    fp = np.ones((1, 1, 3, 3))
    e = Emission(T("2000-01-01 10:30"), T("2000-01-01 11:00"), 1.0, 0.0, 0.0)
    end = np.array(["2000-01-01T12:00"], dtype="datetime64[ns]")
    kw = dict(window_end=end, emissions=[e], source_deg=0.25, **_grid())
    with pytest.raises(ValueError, match="one time bin"):
        concentration.predict(fp, age_start_s=[0.0], age_end_s=[1e5], basis="ambient", **kw)
    fp2 = np.ones((1, 2, 3, 3))
    with pytest.raises(ValueError, match="needs the source density"):
        concentration.predict(fp2, age_start_s=[0, 1], age_end_s=[1, 2], basis="stp", **kw)


def test_dry_air_density() -> None:
    assert concentration.dry_air_density(100000.0, 300.0, 0.0) == pytest.approx(
        100000.0 / (287.05 * 300.0)
    )
    # q = 15 g/kg removes about 2.4% of the pressure (decision 0022's table).
    ratio = concentration.dry_air_density(1e5, 300.0, 0.015) / concentration.dry_air_density(
        1e5, 300.0, 0.0
    )
    assert ratio == pytest.approx(1 - 0.024, abs=0.001)


# ---------------------------------------------------------------------------
# end to end: GLIDE backward against the exact forward answer
# ---------------------------------------------------------------------------

U = 10.0  # m/s, eastward
LON_R = 0.5  # receptor longitude; the source is at 0, 0


def _forward_answer(t0: datetime, dur_s: float, e0: datetime, e1: datetime, box_deg: float):
    """Exact sample-mean ambient concentration per unit emission rate, times the box
    volume: (1/T_r) ∫ |{s in [E0,E1] : x_r - U (t - s) in box}| dt (s)."""

    m_per_deg = 111320.0
    d_a = (LON_R + box_deg / 2) * m_per_deg / U  # oldest age at which air was in the box
    d_b = (LON_R - box_deg / 2) * m_per_deg / U
    ts = t0.timestamp() + np.linspace(0.0, dur_s, 200001)
    lo = np.maximum(ts - d_a, e0.timestamp())
    hi = np.minimum(ts - d_b, e1.timestamp())
    return float(np.mean(np.clip(hi - lo, 0.0, None)))


def _uniform_wind_footprint(tmp_path: Path, box_deg: float, dt_s: int = 60) -> xr.Dataset:
    t0 = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)
    base = _make_run_config(
        output_uri=str(tmp_path / f"out{box_deg}"),
        start_time=t0,
        simulation_length_seconds=4 * 3600,
        dt_seconds=dt_s,
        output_lon_bounds=(-1.5 * box_deg, 1.5 * box_deg),
        output_lat_bounds=(-1.5 * box_deg, 1.5 * box_deg),
        output_n_x=3,
        output_n_y=3,
        z_edges_m=(0.0, 40.0),
        n_time_bins=48,
    ).model_dump(mode="json")
    base["output_grid"]["time_bin_seconds"] = 300
    base["release"] = {
        "kind": "samples",
        "samples": [
            {
                "name": f"S{i}",
                "lon": LON_R,
                "lat": 0.0,
                "start_time": (t0 + timedelta(minutes=30 * i)).isoformat(),
                "duration_seconds": 3600,
                "alt_min_agl_m": 0.0,
                "alt_max_agl_m": 40.0,
            }
            for i in range(3)
        ],
        "n_particles_per_release": 20000,
        "seed": 1,
    }
    cfg = RunConfig.model_validate(base)
    reader = _make_reader(cfg, wind_fn=lambda _: (U, 0.0, 0.0))
    reader.coverage_end = t0 + timedelta(hours=3)
    # No turbulence: every particle keeps its height inside the 0-40 m layer.
    scheme = PlaceholderConstantOU()
    scheme.sigma_w2 = 0.0
    _run(cfg, reader=reader, scheme=scheme)
    return xr.open_zarr(Path(cfg.io.output_uri) / "footprints.zarr").load()


@pytest.mark.parametrize(("box_deg", "dt_s"), [(0.25, 10), (1.0, 60)])
def test_backward_prediction_matches_the_forward_solution(
    tmp_path: Path, box_deg: float, dt_s: int
) -> None:
    """Uniform wind, no turbulence, a sample 0.5° downwind: GLIDE's backward footprint,
    cut to the emission window, gives the forward model's sample-mean concentration.

    Three one-hour samples starting 12:00, 12:30 and 13:00; the source emits
    10:00-10:45, so the first sample sees part of the puff, the others less or
    none, and the timing is tested where it matters."""

    ds = _uniform_wind_footprint(tmp_path, box_deg, dt_s)
    e = Emission(T("2024-01-01 10:00"), T("2024-01-01 10:45"), mass=2700.0, lat=0.0, lon=0.0)
    c = concentration.predict_from_dataset(ds, [e], source_deg=box_deg, basis="ambient")
    lat_e = ds["latitude_edge"].values
    lon_e = ds["longitude_edge"].values
    w = concentration.source_box_weights(lat_e, lon_e, 0.0, 0.0, box_deg)
    volume = float(np.sum(w * concentration.cell_area_m2(lat_e, lon_e))) * 40.0
    t0 = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)
    e0, e1 = datetime(2024, 1, 1, 10, 0, tzinfo=UTC), datetime(2024, 1, 1, 10, 45, tzinfo=UTC)
    expected = np.array(
        [
            _forward_answer(t0 + timedelta(minutes=30 * i), 3600.0, e0, e1, box_deg)
            * e.rate
            / volume
            for i in range(3)
        ]
    )
    assert expected[0] > 0
    # Observed: within 0.13% (seed 1, 20,000 particles per sample). GLIDE credits a
    # step's residence, at the step's older position, to the age of its younger end,
    # a timing offset of dt/2: with 60 s steps the 0.25° case (a 46-minute transit
    # cut by the puff's edges) comes out 2.3% and 7% low, and 1.0% and 2.8% with
    # 30 s; hence 10 s there. For hours-long tracer releases the offset is < 0.5%.
    np.testing.assert_allclose(c, expected, rtol=0.005)


# ---------------------------------------------------------------------------
# the benchmark's plumbing
# ---------------------------------------------------------------------------


def _nam_like_store(path: Path, attrs: dict | None = None) -> Path:
    """A small ascending 0..360 store west of Greenwich, laid out like met-bazarr's
    NAM stores (240..305 E), with what ``ArcoEra5ZarrReader`` needs to open it,
    and the global ``attrs`` given."""

    lon = np.arange(240.0, 305.01, 1.0)
    lat = np.arange(58.0, 21.99, -1.0)
    time = pd.date_range("1980-07-08", periods=3, freq="h")
    shape = (len(time), len(lat), len(lon))
    ds = xr.Dataset(
        {
            "surface_pressure": (("time", "latitude", "longitude"), np.full(shape, 1e5, "f4")),
            "boundary_layer_height": (("time", "latitude", "longitude"), np.ones(shape, "f4")),
        },
        coords={"time": time, "latitude": lat, "longitude": lon},
        attrs=attrs or {},
    )
    ds.to_zarr(path)
    return path


def test_lon_in_extent_follows_the_reader() -> None:
    assert benchmark.lon_in_extent(-97.46, (-120.0, -55.0)) == pytest.approx(-97.46)
    assert benchmark.lon_in_extent(262.54, (-120.0, -55.0)) == pytest.approx(-97.46)
    # A store straddling 180 E stays 0..360 in the reader.
    assert benchmark.lon_in_extent(-170.0, (150.0, 230.0)) == pytest.approx(190.0)
    # Outside the grid: left as given for the preflight to refuse.
    assert benchmark.lon_in_extent(10.0, (-120.0, -55.0)) == 10.0
    # The source box finds the source whichever convention the footprint grid uses.
    edges = np.array([262.165, 262.415, 262.665, 262.915])
    lat = np.array([34.865, 35.115, 35.365, 35.615])
    w = concentration.source_box_weights(lat, edges, 35.24, -97.46, 0.25)
    assert w[1, 1] == pytest.approx(1.0) and w.sum() == pytest.approx(1.0)


def test_config_from_a_0_360_store_passes_the_runtime_preflight(tmp_path: Path) -> None:
    """Review item 1: on an ascending 0..360 NAM-like store, the configuration the
    benchmark generates uses the reader's -180..180 longitudes and passes
    ``_validate_meteorology_space_coverage``; the old 0..360 configuration fails it."""

    from lpdm.main import PreflightValidationError, _validate_meteorology_space_coverage
    from lpdm.met_reader import ArcoEra5ZarrReader

    store = str(_nam_like_store(tmp_path / "nam.zarr"))
    box = benchmark.met_extent(store)
    assert box["lon"] == (-120.0, -55.0)
    assert box["lat"] == (22.0, 58.0)
    rec = pd.DataFrame(
        {
            "receptor_id": ["R0", "R1"],
            "lat": [35.6, 36.1],
            "lon": [-97.0, 262.9],  # either convention in, the reader's out
            "start": [T("1980-07-08 01:00"), T("1980-07-08 01:30")],
            "end": [T("1980-07-08 02:00"), T("1980-07-08 02:00")],
        }
    )
    site = benchmark.SitePlan("OKC", 35.24, -97.46, ["X"], rec, T("1980-07-08 00:00"))
    chunk = benchmark.Chunk("OKC/run00", rec, 7200)
    plan = benchmark.Plan(
        samples=pd.DataFrame(),
        receptors=rec,
        emissions={},
        sites=[site],
        time_bin_s=300,
        layer=(0.0, 100.0),
    )
    cfg = benchmark.glide_config(
        benchmark._load_base(None),
        site,
        chunk,
        plan,
        met=store,
        met_box=box,
        output_uri=str(tmp_path / "glide"),
        particles=10,
        seed=1,
        source_deg=0.25,
        source_layer_m=40.0,
        overrides={},
    )
    rc = RunConfig.model_validate(cfg)
    assert rc.met_domain.lon_bounds == (-119.5, -55.5)
    assert rc.output_grid.lon_bounds == pytest.approx((-97.46 - 0.375, -97.46 + 0.375))
    assert [s.lon for s in rc.release.samples] == pytest.approx([-97.0, -97.1])
    reader = ArcoEra5ZarrReader(zarr_store=store, device="cpu")
    try:
        _validate_meteorology_space_coverage(reader, rc)  # passes
        old = json.loads(json.dumps(cfg))
        # B12's former configuration: everything in 0..360.
        old["met_domain"]["lon_bounds"] = [240.5, 304.5]
        old["output_grid"]["lon_bounds"] = [x % 360.0 for x in old["output_grid"]["lon_bounds"]]
        for smp in old["release"]["samples"]:
            smp["lon"] %= 360.0
        with pytest.raises(PreflightValidationError, match="not inside the meteorology grid"):
            _validate_meteorology_space_coverage(reader, RunConfig.model_validate(old))
    finally:
        reader.close()


REPAIR = {
    "repair": "era5-surface-geopotential",
    "description": "geopotential_at_surface replaced (met-bazarr decision 0009)",
    "date": "2026-11-24T06:36:37Z",
    "producer_version": "0.4.0",
    "producer_commit": "a77db3294aff8e64046cbfda73896f3aaa59314b",
    "change_m": {"median": 0.3, "p99": 16.52, "max": 123.77},
}


def test_scorecard_and_compare_show_the_met_repairs(tmp_path: Path) -> None:
    """A store repaired after it was built: the scorecard (Markdown and JSON) and
    ``compare`` show the repair's producer version beside the store's own."""

    base = {"producer": "met-bazarr", "producer_version": "0.3.0", "producer_commit": "abc"}
    repaired = _nam_like_store(
        tmp_path / "repaired.zarr", dict(base, glide_repairs=json.dumps([REPAIR]))
    )
    plain = _nam_like_store(tmp_path / "plain.zarr", base)
    score = dict(
        experiment="OKC80",
        n_pairs=1,
        rank=1.0,
        r2_term=0.1,
        fb_term=0.2,
        fms_term=0.3,
        ks_term=0.4,
        rank_plain_r=1.0,
        mean_measured_pg_m3=1.0,
        mean_calculated_pg_m3=1.0,
    )
    runs = []
    for store in (repaired, plain):
        box = benchmark.met_extent(str(store))
        meta = benchmark.provenance({}, str(store), box, dict(receptor_layer_m=[0, 100]))
        out = tmp_path / store.stem
        out.mkdir()
        benchmark.build_scorecard(out, [score], [], meta, ["OKC80"])
        pd.DataFrame({"experiment": ["OKC80"], "sample_id": [1], "pred_raw": [1.0]}).to_csv(
            out / "predictions.csv", index=False
        )
        runs.append(out)

    summary = "met-bazarr 0.3.0, repaired 0.4.0 (era5-surface-geopotential)"
    md = (runs[0] / "scorecard.md").read_text()
    assert f"| met | {summary} |" in md
    assert "| met repair 1 | `era5-surface-geopotential`: met-bazarr 0.4.0 (`a77db3294aff`)" in md
    assert "change median 0.3 m, p99 16.52 m, max 123.77 m" in md
    card = json.loads((runs[0] / "scorecard.json").read_text())
    assert card["meta"]["met_summary"] == summary
    assert card["meta"]["met_repairs"] == [REPAIR]
    md_plain = (runs[1] / "scorecard.md").read_text()
    assert "| met | met-bazarr 0.3.0, no repairs recorded |" in md_plain
    assert "| met repairs | none recorded |" in md_plain
    assert json.loads((runs[1] / "scorecard.json").read_text())["meta"]["met_repairs"] == []

    text = benchmark.compare_runs(runs)
    assert f"| run 1 | `{repaired}` | {summary} |" in text
    assert f"| run 2 | `{plain}` | met-bazarr 0.3.0, no repairs recorded |" in text

    # Several repairs group by version; an unreadable attribute is shown, not dropped.
    second = dict(REPAIR, repair="other-fix", producer_version="0.5.0")
    third = dict(REPAIR, repair="third-fix")
    prov = dict(base, glide_repairs=json.dumps([REPAIR, second, third]))
    assert benchmark.met_summary(prov) == (
        "met-bazarr 0.3.0, repaired 0.4.0 (era5-surface-geopotential, third-fix); 0.5.0 (other-fix)"
    )
    bad = benchmark.met_repairs(dict(base, glide_repairs="not json"))
    assert bad == [{"repair": "unreadable", "raw": "not json"}]


def test_choose_time_bin() -> None:
    t = np.array([0, 2700, 4500, 3600 * 5])  # 45-min and 75-min marks
    assert benchmark.choose_time_bin(t) == 900
    assert benchmark.choose_time_bin(np.array([0, 3600, 7200 * 4])) == 3600
    assert benchmark.choose_time_bin(np.array([0, 300 * 7])) == 300


def test_chunks_keep_meteorology_inside_the_store() -> None:
    first = T("2000-01-01 19:00")
    starts = pd.date_range("2000-01-01 21:00", periods=60, freq="90min")
    rec = pd.DataFrame(
        {"receptor_id": [f"R{i}" for i in range(60)], "lat": 0.0, "lon": 0.0, "start": starts}
    )
    rec["end"] = rec["start"] + pd.Timedelta(hours=3)
    met_start = T("2000-01-01 00:00")
    parts = benchmark.chunk_receptors(rec, first, 300, met_start=met_start, max_span_h=24)
    assert sum(len(p) for p, _ in parts) == 60
    for part, length in parts:
        assert part["start"].min() - pd.Timedelta(seconds=length) >= met_start
        assert length >= (part["end"].max() - first).total_seconds()
        assert length % 300 == 0
    with pytest.raises(SystemExit, match="too late"):
        benchmark.chunk_receptors(rec, first, 300, met_start=T("2000-01-01 20:00"))


def test_independence_guard(tmp_path: Path) -> None:
    benchmark.check_independence(["OKC80", "OKC80-PDCH"], None)
    for name in ("CAPTEX-1", "ANATEX-STC"):
        with pytest.raises(SystemExit, match="refusing to run GLIDE"):
            benchmark.check_independence([name], None)
    rec = tmp_path / "0099-preregistration.md"
    rec.write_text("# Record\n\n**Registration.** pending\n")
    with pytest.raises(SystemExit):
        benchmark.check_independence(["CAPTEX-1"], rec)
    rec.write_text(
        "# Record\n\n**Registration.** https://osf.io/abcde, doi:10.17605/X, 2027-01-01\n"
    )
    benchmark.check_independence(["CAPTEX-1"], rec)
    # The command line refuses before reading data or writing anything.
    with pytest.raises(SystemExit, match="refusing"):
        benchmark.main(
            [
                "run",
                "--experiments",
                "CAPTEX-1",
                "--met",
                "x",
                "--out",
                str(tmp_path / "o"),
                "--no-run",
            ]
        )
    assert not (tmp_path / "o").exists()


@needs_datem
def test_okc80_plan(monkeypatch) -> None:
    monkeypatch.setenv("GLIDE_DATEM", str(_datem_dir()))
    plan = benchmark.build_plan(
        ["OKC80", "OKC80-PDCH"], met_start=T("1980-07-08 00:00"), max_span_h=24
    )
    assert plan.samples.groupby("experiment").size().to_dict() == {"OKC80": 697, "OKC80-PDCH": 683}
    assert len(plan.receptors) == 700  # the PDCH samples share windows with PMCH
    assert plan.time_bin_s == 300  # sample times are on a 5-minute grid
    (site,) = plan.sites
    assert (site.lat, site.lon) == (35.24, -97.46)
    assert sorted(r for ch in site.chunks for r in ch.receptors["receptor_id"]) == sorted(
        plan.receptors["receptor_id"]
    )
    e = plan.emissions["OKC80"].iloc[0]
    assert (e["start"], e["end"], e["mass"]) == (
        T("1980-07-08 19:00"),
        T("1980-07-08 22:00"),
        192.0,
    )


@needs_datem
def test_okc80_config_and_scorecard_from_a_synthetic_footprint(tmp_path: Path, monkeypatch) -> None:
    """Without running GLIDE: the configuration a run would use, then predictions,
    DATEM files, scores and scorecard from a synthetic footprint that puts the
    first sample window's particles in the source cell during the release."""

    monkeypatch.setenv("GLIDE_DATEM", str(_datem_dir()))
    plan = benchmark.build_plan(["OKC80"], met_start=T("1980-07-08 00:00"))
    (site,) = plan.sites
    ch = site.chunks[0]
    # The OKC80 store's extent as the reader presents it (-180..180 since PR #56).
    box = dict(lat=(22.0, 58.0), lon=(-120.0, -55.0), provenance={})
    cfg = benchmark.glide_config(
        benchmark._load_base(None),
        site,
        ch,
        plan,
        met="store.zarr",
        met_box=box,
        output_uri=str(tmp_path / "glide"),
        particles=10,
        seed=3,
        source_deg=0.25,
        source_layer_m=40.0,
        overrides={"turbulence.meander.coefficient": 0.3},
    )
    rc = RunConfig.model_validate(cfg)
    assert rc.release.kind == "samples" and len(rc.release.samples) == len(ch.receptors)
    assert rc.release.samples[0].alt_max_agl_m == 100.0
    assert -120.0 < rc.release.samples[0].lon < -55.0
    assert rc.output_grid.lon_bounds == pytest.approx((-97.46 - 0.375, -97.46 + 0.375))
    assert rc.met_domain.lon_bounds == (-119.5, -55.5)
    assert rc.output_grid.density_weighting == "dry_air"
    assert benchmark.check_basis(cfg, "stp") == "dry_air"
    with pytest.raises(SystemExit, match="density_weighting=none"):
        benchmark.check_basis(cfg, "ambient")
    assert rc.output_grid.time_bin_seconds == 300
    assert rc.output_grid.z_edges_m == (0.0, 40.0)
    assert rc.turbulence.meander.coefficient == 0.3
    assert rc.output_grid.n_time_bins * 300 >= rc.simulation.length_seconds
    # The physics hash ignores the releases and paths but sees the override.
    h = benchmark.config_hash(cfg)
    cfg2 = dict(cfg, io={"zarr_store": "other", "output_uri": "elsewhere"})
    assert benchmark.config_hash(cfg2) == h
    cfg3 = json.loads(json.dumps(cfg))
    cfg3["turbulence"]["meander"]["coefficient"] = 0.16
    assert benchmark.config_hash(cfg3) != h

    # Review item 3: an explicit weighting survives the derived output grid, and
    # the hash sees it.
    cfg_none = benchmark.glide_config(
        benchmark._load_base(None),
        site,
        ch,
        plan,
        met="store.zarr",
        met_box=box,
        output_uri=str(tmp_path / "glide"),
        particles=10,
        seed=3,
        source_deg=0.25,
        source_layer_m=40.0,
        overrides={
            "turbulence.meander.coefficient": 0.3,
            "output_grid.density_weighting": "none",
        },
    )
    assert RunConfig.model_validate(cfg_none).output_grid.density_weighting == "none"
    assert cfg_none["output_grid"]["n_x"] == 3
    assert benchmark.check_basis(cfg_none, "ambient") == "none"
    assert benchmark.config_hash(cfg_none) != h

    # A synthetic footprint for every receptor: 10 s per 5-minute bin in the
    # source cell, for the bins inside the release (19:00-22:00).
    rec = plan.receptors
    n_t = rc.output_grid.n_time_bins + 36
    fp = np.zeros((len(rec), n_t, 1, 3, 3), dtype=np.float32)
    ends = rec["end"].to_numpy()
    for i, end in enumerate(ends):
        for k in range(n_t):
            mid = pd.Timestamp(end) - pd.Timedelta(seconds=300 * k + 150)
            if T("1980-07-08 19:00") <= mid < T("1980-07-08 22:00"):
                fp[i, k, 0, 1, 1] = 10.0
    edges_lat = np.linspace(35.24 - 0.375, 35.24 + 0.375, 4)
    edges_lon = np.linspace(-97.46 - 0.375, -97.46 + 0.375, 4)
    ds = xr.Dataset(
        {"footprint": (("release", "time_ago", "z_bin", "latitude", "longitude"), fp)},
        coords={
            "release_time": ("release", rec["start"].to_numpy()),
            "release_duration_seconds": (
                "release",
                (rec["end"] - rec["start"]).dt.total_seconds().to_numpy(),
            ),
            "site": ("release", rec["receptor_id"].to_numpy()),
            "time_ago_start_hours": ("time_ago", np.arange(n_t) * 300 / 3600.0),
            "time_ago_end_hours": ("time_ago", (np.arange(n_t) + 1) * 300 / 3600.0),
            "z_bottom_m": ("z_bin", [0.0]),
            "z_top_m": ("z_bin", [40.0]),
            "latitude_edge": ("latitude_edge", edges_lat),
            "longitude_edge": ("longitude_edge", edges_lon),
        },
        attrs={"density_weighting": "none", "units": "s"},
    )
    out = tmp_path / "run"
    store = out / "glide" / site.key / "run00" / "footprints.zarr"
    store.parent.mkdir(parents=True)
    ds.to_zarr(store)
    meta = dict(
        source_deg=0.25,
        bootstrap_seed=0,
        experiments=["OKC80"],
        receptor_layer_m=[0, 100],
        met_store="store.zarr",
        time_bin_seconds=300,
        basis="ambient",
    )
    scores = benchmark._score_run(out, plan, meta, met=None, basis="ambient")
    (s,) = scores
    assert s["experiment"] == "OKC80" and s["n_pairs"] == 697
    # Every window that overlaps the release gets a known value: rate x 10 s x bins / V.
    volume = concentration.cell_area_m2(edges_lat[1:3], edges_lon[1:3])[0, 0] * 40.0
    pred = pd.read_csv(out / "predictions.csv")
    first = pred.iloc[0]
    n_bins = (
        min(T(first["end"]), T("1980-07-08 22:00")) - T("1980-07-08 19:00")
    ).total_seconds() / 300
    assert first["pred_raw"] == pytest.approx(
        192.0 / 10800 * 10 * n_bins / volume / 1e-15, rel=1e-5
    )
    meta.update(created="x", physics_tag="none", glide_commit="c", config_hash=h)
    benchmark.build_scorecard(out, scores, [], meta, ["OKC80"])
    md = (out / "scorecard.md").read_text()
    assert "Exploratory" in md and "| OKC80 | 697 |" in md
    card = json.loads((out / "scorecard.json").read_text())
    assert card["scores"][0]["rank"] == pytest.approx(s["rank"])
    # The DATEM-format file pairs with every measurement.
    lines = (out / "predictions" / "OKC80.txt").read_text().splitlines()
    assert len(lines) == 2 + 697
    # compare: a run against itself shows no differences.
    text = benchmark.compare_runs([out, out])
    assert "identical predictions" in text and "+0.0000" in text
    assert math.isfinite(s["rank"])


# ---------------------------------------------------------------------------
# footprint weighting and the concentration basis (review item 3)
# ---------------------------------------------------------------------------

RHO = 1.15  # kg m-3, the dry-air density the synthetic weighted store divides by
EDGES = np.array([-0.375, -0.125, 0.125, 0.375])


def _store(weighting: str | None, value: float) -> xr.Dataset:
    """One release ending 12:00, eight 15-minute age bins, ``value`` in the source
    (centre) cell in every bin, a 0-40 m layer, with A04's attributes."""

    n_t = 8
    fp = np.zeros((1, n_t, 1, 3, 3))
    fp[0, :, 0, 1, 1] = value
    attrs = {}
    if weighting is not None:
        attrs = {
            "density_weighting": weighting,
            "units": {"dry_air": "s m3 kg-1", "none": "s"}[weighting],
        }
    return xr.Dataset(
        {"footprint": (("release", "time_ago", "z_bin", "latitude", "longitude"), fp)},
        coords={
            "release_time": ("release", np.array(["2000-01-01T11:00"], dtype="datetime64[ns]")),
            "release_duration_seconds": ("release", [3600.0]),
            "site": ("release", ["R0"]),
            "time_ago_start_hours": ("time_ago", np.arange(n_t) * 0.25),
            "time_ago_end_hours": ("time_ago", (np.arange(n_t) + 1) * 0.25),
            "z_bottom_m": ("z_bin", [0.0]),
            "z_top_m": ("z_bin", [40.0]),
            "latitude_edge": ("latitude_edge", EDGES),
            "longitude_edge": ("longitude_edge", EDGES),
        },
        attrs=attrs,
    )


def test_concentration_on_each_kind_of_store_by_hand() -> None:
    """(a) a dry-air-weighted store on the "stp" basis is not divided by the density
    again; (b) a plain-seconds store on the "ambient" basis is the plain mass
    concentration; (c) a weighted store on the ambient basis is refused."""

    # 3.6 kg over 10:30-11:00 = ages 60-90 min of a release ending 12:00: bins 4 and 5.
    e = Emission(T("2000-01-01 10:30"), T("2000-01-01 11:00"), mass=3.6, lat=0.0, lon=0.0)
    rate = 3.6 / 1800.0  # kg s-1
    # The centre cell, 0.25 deg square at the equator, in the model's metric, x 40 m.
    area = 110540.0 * 111320.0 * (180.0 / math.pi) * 2.0 * math.sin(math.radians(0.125)) * 0.25
    volume = area * 40.0
    tau = 2 * 100.0  # s in the source cell during the emission

    # (b) ambient, plain seconds: Q tau / V.
    ambient = concentration.predict_from_dataset(
        _store("none", 100.0), [e], source_deg=0.25, basis="ambient"
    )
    np.testing.assert_allclose(ambient, [rate * tau / volume], rtol=1e-12)
    assert ambient[0] == pytest.approx(2.0e-3 * 200.0 / volume)

    # (a) stp, weighted store: Q / V x (M_air / V_m) x sum(tau / rho), the store already
    # holding tau / rho; no density function is given or needed.
    stp = concentration.predict_from_dataset(
        _store("dry_air", 100.0 / RHO), [e], source_deg=0.25, basis="stp"
    )
    expected = rate / volume * (0.02897 / 0.0224) * tau / RHO
    np.testing.assert_allclose(stp, [expected], rtol=1e-12)
    # Dividing again would have given expected / RHO; make sure it did not.
    assert stp[0] != pytest.approx(expected / RHO, rel=1e-3)
    # The same answer from the plain store with the density applied afterwards.
    stp_after = concentration.predict_from_dataset(
        _store("none", 100.0),
        [e],
        source_deg=0.25,
        basis="stp",
        density=lambda t: np.full(len(t), RHO),
    )
    np.testing.assert_allclose(stp_after, stp, rtol=1e-12)

    # (c) the mismatched combination.
    with pytest.raises(ValueError, match="ambient concentration needs an unweighted footprint"):
        concentration.predict_from_dataset(
            _store("dry_air", 100.0 / RHO), [e], source_deg=0.25, basis="ambient"
        )
    # Inconsistent attributes are refused rather than guessed.
    bad = _store("dry_air", 1.0)
    bad.attrs["units"] = "s"
    with pytest.raises(ValueError, match="units"):
        concentration.is_density_weighted(bad)
    # A store from before A04 (no attribute) is read as plain seconds, with a warning.
    with pytest.warns(UserWarning, match="declares no density_weighting"):
        assert not concentration.is_density_weighted(_store(None, 1.0))


# ---------------------------------------------------------------------------
# rescoring leaves the record alone unless it succeeds (review item 4)
# ---------------------------------------------------------------------------


def _scored_run(tmp_path: Path, monkeypatch, weighting: str) -> tuple[Path, dict]:
    """A finished run directory (weighted footprint store, metadata, scorecard) whose
    plan is a single sample of a single site, without DATEM files."""

    run = tmp_path / "run"
    store = run / "glide" / "SITE" / "run00" / "footprints.zarr"
    store.parent.mkdir(parents=True)
    _store(weighting, 100.0 / RHO if weighting == "dry_air" else 100.0).to_zarr(store)
    rec = pd.DataFrame(
        {
            "receptor_id": ["R0"],
            "lat": [0.0],
            "lon": [0.0],
            "start": [T("2000-01-01 11:00")],
            "end": [T("2000-01-01 12:00")],
        }
    )
    site = benchmark.SitePlan("SITE", 0.0, 0.0, ["OKC80"], rec, T("2000-01-01 10:30"))
    emis = pd.DataFrame(
        {
            "start": [T("2000-01-01 10:30")],
            "end": [T("2000-01-01 11:00")],
            "mass": [3.6],
            "lat": [0.0],
            "lon": [0.0],
        }
    )
    plan = benchmark.Plan(
        samples=pd.DataFrame(),
        emissions={"OKC80": emis},
        receptors=rec,
        sites=[site],
        time_bin_s=900,
        layer=(0.0, 100.0),
    )
    monkeypatch.setattr(benchmark, "build_plan", lambda *a, **k: plan)
    monkeypatch.setattr(benchmark, "reference_scores", lambda *a, **k: [])
    meta = dict(
        experiments=["OKC80"],
        receptor_layer_m=[0, 100],
        max_span_h=24.0,
        source_deg=0.25,
        bootstrap_seed=0,
        met_store="unused.zarr",
        basis="stp",
    )
    (run / "run_meta.json").write_text(json.dumps(meta, indent=1))
    for name in ("scorecard.md", "scorecard.json", "scorecard.csv", "predictions.csv"):
        (run / name).write_text(f"original {name}\n")
    (run / "predictions").mkdir()
    (run / "predictions" / "OKC80.txt").write_text("original predictions\n")
    return run, meta


def _snapshot(d: Path) -> dict[str, bytes]:
    return {
        str(p.relative_to(d)): p.read_bytes()
        for p in sorted(d.rglob("*"))
        if p.is_file() and "glide" not in p.relative_to(d).parts
    }


def test_failed_rescoring_leaves_the_record_byte_identical(tmp_path: Path, monkeypatch) -> None:
    run, _ = _scored_run(tmp_path, monkeypatch, "dry_air")
    before = _snapshot(run)
    # The weighted store cannot give an ambient concentration: scoring fails...
    with pytest.raises(ValueError, match="unweighted footprint"):
        benchmark.main(["score", str(run), "--basis", "ambient"])
    # ...and neither the metadata nor the scorecard has changed, nor anything been left.
    assert _snapshot(run) == before
    assert sorted(p.name for p in tmp_path.iterdir()) == ["run"]

    # A failure late in scoring (after the scorecard is half written) does the same,
    # into another destination too.
    def broken(out, *a, **k):
        (out / "scorecard.csv").write_text("half\n")
        raise RuntimeError("disk full")

    monkeypatch.setattr(benchmark, "build_scorecard", broken)
    monkeypatch.setattr(
        benchmark, "_score_run", lambda out, *a, **k: (out / "predictions.csv").write_text("x")
    )
    with pytest.raises(RuntimeError, match="disk full"):
        benchmark.main(["score", str(run), "--bootstrap-seed", "5"])
    with pytest.raises(RuntimeError, match="disk full"):
        benchmark.main(["score", str(run), "--bootstrap-seed", "5", "--out", str(tmp_path / "b")])
    assert _snapshot(run) == before
    assert sorted(p.name for p in tmp_path.iterdir()) == ["run"]


def test_rescoring_publishes_metadata_and_scorecard_together(tmp_path: Path, monkeypatch) -> None:
    run, meta = _scored_run(tmp_path, monkeypatch, "none")
    before = _snapshot(run)

    def card(out, scores, refs, m, names):
        (out / "scorecard.md").write_text(f"basis {m['basis']}\n")

    monkeypatch.setattr(benchmark, "build_scorecard", card)
    monkeypatch.setattr(benchmark, "score_experiment", lambda *a, **k: {"experiment": "OKC80"})
    monkeypatch.setattr(benchmark, "write_predictions", _fake_write_predictions)
    other = tmp_path / "ambient"
    benchmark.main(["score", str(run), "--basis", "ambient", "--out", str(other)])
    assert _snapshot(run) == before  # the original run is untouched
    assert json.loads((other / "run_meta.json").read_text())["basis"] == "ambient"
    assert (other / "scorecard.md").read_text() == "basis ambient\n"
    assert (other / "glide").resolve() == (run / "glide").resolve()
    pred = pd.read_csv(other / "predictions.csv")
    area = 110540.0 * 111320.0 * (180.0 / math.pi) * 2.0 * math.sin(math.radians(0.125)) * 0.25
    assert pred["pred"].iloc[0] == pytest.approx(3.6 / 1800.0 * 200.0 / (area * 40.0))
    # In place: the metadata and the scorecard change together.
    benchmark.main(["score", str(run), "--basis", "ambient"])
    assert json.loads((run / "run_meta.json").read_text())["basis"] == "ambient"
    assert (run / "scorecard.md").read_text() == "basis ambient\n"
    assert not (run / "predictions" / "OKC80.txt").exists()  # replaced as a whole

    # An existing destination must hold the requested run's footprints.
    run2 = tmp_path / "run2"
    (run2 / "glide").mkdir(parents=True)
    (run2 / "run_meta.json").write_text(json.dumps(meta))
    with pytest.raises(SystemExit, match="already holds footprints"):
        benchmark.main(["score", str(run2), "--out", str(other)])


def _fake_write_predictions(plan, pred, out):
    """``write_predictions`` without DATEM sample files: the predictions table only."""

    (out / "predictions").mkdir(parents=True, exist_ok=True)
    pred.to_csv(out / "predictions.csv", index=False)
    return pred.assign(experiment="OKC80")
