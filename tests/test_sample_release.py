"""The `samples` release kind (queue task B12): one backward release per measured
sample, each with its own location, window and height range."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
import torch
import xarray as xr
from pydantic import ValidationError
from test_density_weighting import _config as _density_config
from test_density_weighting import _FrozenTurbulence, _WindowConstantReader
from test_main_runtime import _make_reader, _make_run_config

from lpdm.config import ConcreteRelease, ReleaseBatch, RunConfig
from lpdm.main import _run
from lpdm.release_generator import generate_batch_particles

T0 = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)


def _samples_config(samples: list[dict], **release: object) -> RunConfig:
    base = _make_run_config().model_dump(mode="json")
    base["release"] = {
        "kind": "samples",
        "samples": samples,
        "n_particles_per_release": release.get("n", 64),
        "seed": release.get("seed", 7),
    }
    return RunConfig.model_validate(base)


def _sample(name: str, *, lon=0.0, lat=0.0, minutes=0, dur=600, lo=0.0, hi=100.0) -> dict:
    return {
        "name": name,
        "lon": lon,
        "lat": lat,
        "start_time": (T0 + timedelta(minutes=minutes)).isoformat(),
        "duration_seconds": dur,
        "alt_min_agl_m": lo,
        "alt_max_agl_m": hi,
    }


def test_samples_expand_one_release_per_sample_in_order() -> None:
    cfg = _samples_config(
        [_sample("a", lon=0.5, dur=2700), _sample("b", lat=-0.5, minutes=30, lo=10.0, hi=10.0)]
    )
    (batch,) = cfg.expand_to_batches()
    a, b = batch.releases
    assert (a.release_idx, a.label, a.lon, a.duration_seconds) == (0, "a", 0.5, 2700)
    assert (a.alt_agl_m, a.alt_top_agl_m, a.seed) == (0.0, 100.0, 7)
    assert b.release_time == T0 + timedelta(minutes=30)
    # Equal bottom and top is a point release, the same path as every other kind.
    assert (b.alt_agl_m, b.alt_top_agl_m, b.seed) == (10.0, None, 8)
    assert cfg.release.duration_seconds == 2700


def test_samples_batches_follow_max_releases_per_batch() -> None:
    cfg = _samples_config([_sample(f"s{i}", minutes=10 * i) for i in range(5)])
    cfg = cfg.model_copy(
        update={"batch": cfg.batch.model_copy(update={"max_releases_per_batch": 2})}
    )
    assert [len(b.releases) for b in cfg.expand_to_batches()] == [2, 2, 1]


@pytest.mark.parametrize(
    ("samples", "match"),
    [
        ([_sample("a", lo=50.0, hi=10.0)], "alt_max_agl_m must be >= alt_min_agl_m"),
        ([_sample("a"), _sample("a", minutes=5)], "sample names must be unique"),
        ([_sample("a", lon=5.0)], "outside met_domain"),
        ([_sample("a", hi=20000.0)], "above met_domain.alt_max_m"),
        ([_sample("a", dur=20000)], "length_seconds must be > release.duration_seconds"),
        ([], "at least 1"),
    ],
)
def test_samples_validation(samples: list[dict], match: str) -> None:
    with pytest.raises(ValidationError, match=match):
        _samples_config(samples)


def _batch(*releases: ConcreteRelease) -> ReleaseBatch:
    return ReleaseBatch(batch_idx=0, releases=tuple(releases))


def _rel(idx: int, *, seed: int | None, top: float | None) -> ConcreteRelease:
    return ConcreteRelease(
        release_idx=idx,
        release_time=T0,
        lon=1.0,
        lat=2.0,
        alt_agl_m=20.0,
        duration_seconds=1800,
        n_particles=4000,
        seed=seed,
        alt_top_agl_m=top,
    )


def test_height_range_is_uniform_and_times_match_point_release() -> None:
    pb = generate_batch_particles(_batch(_rel(0, seed=3, top=120.0)), device="cpu")
    alt = pb.particles[:, 2].numpy()
    assert alt.min() >= 20.0 and alt.max() < 120.0
    # Uniform: each 20 m slab holds a fifth of the particles (4000: sd about 25).
    counts, _ = np.histogram(alt, bins=5, range=(20.0, 120.0))
    assert np.all(np.abs(counts - 800) < 120)
    # The release times are the draws a point release with the same seed makes.
    point = generate_batch_particles(_batch(_rel(0, seed=3, top=None)), device="cpu")
    assert torch.equal(pb.release_time_offsets_s, point.release_time_offsets_s)
    assert torch.equal(pb.particles[:, [0, 1, 3]], point.particles[:, [0, 1, 3]])
    assert torch.all(point.particles[:, 2] == 20.0)


def test_height_range_draw_is_reproducible() -> None:
    a = generate_batch_particles(_batch(_rel(0, seed=11, top=100.0)), device="cpu")
    b = generate_batch_particles(_batch(_rel(0, seed=11, top=100.0)), device="cpu")
    c = generate_batch_particles(_batch(_rel(0, seed=12, top=100.0)), device="cpu")
    assert torch.equal(a.particles, b.particles)
    assert not torch.equal(a.particles, c.particles)


def _footprint(cfg: RunConfig) -> xr.Dataset:
    reader = _make_reader(cfg, wind_fn=lambda _: (5.0, 1.0, 0.0))
    _run(cfg, reader=reader)
    return xr.open_zarr(Path(cfg.io.output_uri) / "footprints.zarr").load()


def test_point_samples_give_the_same_footprint_as_a_point_schedule(tmp_path: Path) -> None:
    """Bit-identity on CPU: samples with no height range reproduce an existing kind.

    `point_schedule` with the same point, times, durations, particles and seed
    runs the unchanged code path; the new kind must give the same footprint bit
    for bit (and so leaves that path alone)."""

    times = [T0, T0 + timedelta(minutes=20)]
    common = dict(simulation_length_seconds=7200, release_alt_agl_m=30.0, n_time_bins=3)
    sched = _make_run_config(output_uri=str(tmp_path / "sched"), **common).model_dump(mode="json")
    sched["release"] = {
        "kind": "point_schedule",
        "point": {"lon": 0.0, "lat": 0.0, "alt_agl_m": 30.0},
        "times": [t.isoformat() for t in times],
        "duration_seconds": 600,
        "n_particles_per_release": 256,
        "seed": 5,
    }
    samp = _make_run_config(output_uri=str(tmp_path / "samp"), **common).model_dump(mode="json")
    samp["release"] = {
        "kind": "samples",
        "samples": [_sample(f"s{i}", minutes=20 * i, lo=30.0, hi=30.0) for i in range(len(times))],
        "n_particles_per_release": 256,
        "seed": 5,
    }
    a = _footprint(RunConfig.model_validate(sched))
    b = _footprint(RunConfig.model_validate(samp))
    assert float(a["footprint"].sum()) > 0
    np.testing.assert_array_equal(a["footprint"].values, b["footprint"].values)
    assert "release_alt_top_agl_m" not in a.coords and "release_alt_top_agl_m" not in b.coords
    assert list(b["site"].values) == ["s0", "s1"]


def test_samples_run_records_the_height_range(tmp_path: Path) -> None:
    cfg = _samples_config(
        [_sample("a", dur=900), _sample("b", minutes=15, dur=1800, lo=5.0, hi=5.0)], n=128
    )
    cfg = cfg.with_overrides(output_uri=str(tmp_path / "out"))
    ds = _footprint(cfg)
    assert list(ds["site"].values) == ["a", "b"]
    np.testing.assert_array_equal(ds["release_alt_agl_m"].values, [0.0, 5.0])
    np.testing.assert_array_equal(ds["release_alt_top_agl_m"].values, [100.0, 5.0])
    np.testing.assert_array_equal(ds["release_duration_seconds"].values, [900.0, 1800.0])
    assert np.all(ds["footprint"].sum(dim=["time_ago", "z_bin", "latitude", "longitude"]) > 0)


@pytest.mark.parametrize(("weighting", "units"), [("dry_air", "s m3 kg-1"), ("none", "s")])
def test_samples_store_keeps_the_height_range_and_the_weighting(
    tmp_path: Path, weighting: str, units: str
) -> None:
    """The merged metadata keeps both B12's top-height coordinate and A04's attributes.

    The tracer benchmark reads ``density_weighting`` to decide whether to divide by
    the density; losing it would divide a weighted store a second time."""

    data = _density_config(tmp_path / "out", weighting).model_dump(mode="json")
    start = datetime.fromisoformat(data["simulation"]["start_time"])
    data["release"] = {
        "kind": "samples",
        "samples": [
            {
                "name": "a",
                "lon": 0.0,
                "lat": 0.0,
                "start_time": start.isoformat(),
                "duration_seconds": 10,
                "alt_min_agl_m": 0.0,
                "alt_max_agl_m": 30.0,
            }
        ],
        "n_particles_per_release": 64,
        "seed": 7,
    }
    cfg = RunConfig.model_validate(data)
    reader = _WindowConstantReader(
        coverage_start=start - timedelta(hours=2),
        coverage_end=start + timedelta(hours=2),
        fields=lambda _t0: (290.0, 0.01, 100000.0),
    )
    _run(cfg, reader=reader, scheme=_FrozenTurbulence())
    ds = xr.open_zarr(tmp_path / "out" / "footprints.zarr").load()
    np.testing.assert_array_equal(ds["release_alt_top_agl_m"].values, [30.0])
    attrs = {**ds.attrs, **ds["footprint"].attrs}
    assert attrs["density_weighting"] == weighting
    assert attrs["units"] == units
