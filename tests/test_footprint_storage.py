"""Task B2: footprint compression (bit-rounding, zstd) and the sparse COO layout."""

from __future__ import annotations

import json
import sys
import tracemalloc
from pathlib import Path

import numpy as np
import pytest
import torch
import xarray as xr
import zarr

from lpdm.comparison import load_sparse_footprint
from lpdm.output_writer import OutputWriter, footprint_encoding
from lpdm.sparse_footprint import DIMS, write_sparse_footprints

SHAPE = (3, 2, 3, 20, 30)


def _dense(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = rng.lognormal(size=SHAPE).astype(np.float32)
    x[rng.random(SHAPE) < 0.9] = 0.0  # mostly empty, as footprints are
    return x


def _write(path: Path, x: np.ndarray, **enc) -> None:
    coords = {
        "release_time": (
            "release",
            np.arange(SHAPE[0]).astype("datetime64[h]").astype("datetime64[ns]"),
        ),
        "latitude": np.linspace(40, 60, SHAPE[3]),
        "longitude": np.linspace(-10, 20, SHAPE[4]),
    }
    w = OutputWriter()
    w.create_footprint_store(str(path), shape=SHAPE, coords=coords, attrs={"note": "test"}, **enc)
    w.write_footprint_region(str(path), torch.from_numpy(x[:2]), release_start=0, release_stop=2)
    w.write_footprint_region(str(path), torch.from_numpy(x[2:]), release_start=2, release_stop=3)


def test_default_encoding_is_unchanged() -> None:
    assert footprint_encoding(None, None) is None


def test_zstd_alone_is_lossless(tmp_path: Path) -> None:
    x = _dense()
    _write(tmp_path / "z.zarr", x, zstd_level=5)
    np.testing.assert_array_equal(xr.open_zarr(tmp_path / "z.zarr")["footprint"].values, x)


@pytest.mark.parametrize("bits", [7, 10, 16])
def test_bit_rounding_error_is_bounded(tmp_path: Path, bits: int) -> None:
    x = _dense()
    _write(tmp_path / "b.zarr", x, keep_mantissa_bits=bits, zstd_level=3)
    y = xr.open_zarr(tmp_path / "b.zarr")["footprint"].values
    nz = x != 0
    assert np.all(y[~nz] == 0)
    assert np.max(np.abs(y[nz] - x[nz]) / x[nz]) <= 2.0 ** -(bits + 1)


def test_sparse_round_trip(tmp_path: Path) -> None:
    x = _dense()
    _write(tmp_path / "d.zarr", x)
    counts = write_sparse_footprints(tmp_path / "d.zarr", tmp_path / "s.zarr")
    assert counts == {"cells": int(np.prod(SHAPE)), "nonzero": int(np.count_nonzero(x))}
    full = load_sparse_footprint(tmp_path / "s.zarr")
    np.testing.assert_array_equal(full.values, x)
    assert full.attrs["note"] == "test"
    part = load_sparse_footprint(tmp_path / "s.zarr", releases=[2, 0])
    np.testing.assert_array_equal(part.values, x[[2, 0]])
    dense = xr.open_zarr(tmp_path / "d.zarr")
    np.testing.assert_array_equal(part["release_time"].values, dense["release_time"].values[[2, 0]])


def test_run_writes_compressed_and_sparse_footprints(tmp_path: Path) -> None:
    sys.path.insert(0, str(Path(__file__).parent))
    from test_main_runtime import _make_multi_release_reader, _make_periodic_config, _run

    from lpdm.config import RunConfig

    base = _make_periodic_config(output_uri=str(tmp_path / "out"), n_releases=2).model_dump()
    base["output_grid"].update(
        footprint_keep_mantissa_bits=10, footprint_zstd_level=5, footprint_sparse=True
    )
    cfg = RunConfig.model_validate(base)
    meta = _run(cfg, reader=_make_multi_release_reader(cfg, wind_fn=lambda _: (5.0, 0.0, 0.0)))
    assert meta["outputs"]["footprints_sparse"].endswith("footprints_sparse.zarr")
    dense = xr.open_zarr(tmp_path / "out" / "footprints.zarr")["footprint"].values
    np.testing.assert_array_equal(
        load_sparse_footprint(tmp_path / "out" / "footprints_sparse.zarr").values, dense
    )


def _write_sparse_in_memory(dense_path: Path, sparse_path: Path) -> None:
    """The first B2 writer (all entries held, then concatenated), kept as the reference."""

    dense = xr.open_zarr(dense_path)
    fp = dense["footprint"]
    shape = tuple(int(fp.sizes[d]) for d in DIMS)
    names = ("t", "z", "y", "x")
    dtypes = [np.int16 if n < 32768 else np.int32 for n in shape[1:]]
    parts: dict[str, list[np.ndarray]] = {k: [] for k in ("value", *names)}
    offsets = [0]
    for r in range(shape[0]):
        block = np.asarray(fp.isel(release=r).values)
        idx = np.nonzero(block)
        parts["value"].append(block[idx].astype(np.float32))
        for name, i, dt in zip(names, idx, dtypes, strict=True):
            parts[name].append(i.astype(dt))
        offsets.append(offsets[-1] + int(idx[0].size))
    data = {k: ("entry", np.concatenate(v)) for k, v in parts.items()}
    data["release_offset"] = ("release_edge", np.asarray(offsets, dtype=np.int64))
    out = xr.Dataset(data, coords=dict(dense.coords.items()))
    out.attrs.update(dense.attrs)
    out.attrs["glide_sparse_shape"] = json.dumps(shape)
    out.attrs["glide_sparse_layout"] = "COO per release (task B2): see lpdm.sparse_footprint"
    out.to_zarr(sparse_path, mode="w")


def _dense_store(path: Path, n_releases: int, cells: tuple[int, ...], frac: float) -> np.ndarray:
    rng = np.random.default_rng(0)
    shape = (n_releases, *cells)
    x = rng.lognormal(size=shape).astype(np.float32)
    x[rng.random(shape) >= frac] = 0.0
    if n_releases > 5:
        x[5] = 0.0  # an empty release
    coords = {
        "release_time": (
            "release",
            np.arange(n_releases).astype("datetime64[h]").astype("datetime64[ns]"),
        ),
        "latitude": np.linspace(40, 60, shape[3]),
        "longitude": np.linspace(-10, 20, shape[4]),
    }
    w = OutputWriter()
    w.create_footprint_store(str(path), shape=shape, coords=coords, attrs={"note": "test"})
    w.write_footprint_region(
        str(path), torch.from_numpy(x), release_start=0, release_stop=n_releases
    )
    return x


def test_sparse_streaming_matches_in_memory_writer(tmp_path: Path) -> None:
    x = _dense_store(tmp_path / "d.zarr", 12, (2, 3, 20, 30), 0.1)
    _write_sparse_in_memory(tmp_path / "d.zarr", tmp_path / "old.zarr")
    write_sparse_footprints(tmp_path / "d.zarr", tmp_path / "new.zarr")
    old, new = xr.open_zarr(tmp_path / "old.zarr"), xr.open_zarr(tmp_path / "new.zarr")
    xr.testing.assert_identical(old.load(), new.load())
    g_old = zarr.open_group(tmp_path / "old.zarr", mode="r")
    g_new = zarr.open_group(tmp_path / "new.zarr", mode="r")
    assert dict(g_old.attrs) == dict(g_new.attrs)
    assert sorted(g_old.array_keys()) == sorted(g_new.array_keys())
    for k in g_old.array_keys():
        a, b = g_old[k], g_new[k]
        assert (a.dtype, a.shape, dict(a.attrs)) == (b.dtype, b.shape, dict(b.attrs)), k
        np.testing.assert_array_equal(a[...], b[...])
    np.testing.assert_array_equal(load_sparse_footprint(tmp_path / "new.zarr").values, x)


def test_sparse_all_zero_store(tmp_path: Path) -> None:
    x = _dense_store(tmp_path / "d.zarr", 3, (1, 2, 5, 6), 0.0)
    assert write_sparse_footprints(tmp_path / "d.zarr", tmp_path / "s.zarr")["nonzero"] == 0
    np.testing.assert_array_equal(load_sparse_footprint(tmp_path / "s.zarr").values, x)


def test_sparse_writer_memory_does_not_grow_with_releases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Peak Python-side memory stays at about one release, for 10 or 100 releases."""

    # The same entry chunk for both stores, so only the number of releases differs.
    monkeypatch.setattr("lpdm.sparse_footprint.ENTRY_CHUNK", 1 << 14)
    cells = (2, 3, 60, 80)  # 28,800 cells (115 kB) per release, half non-zero
    peaks = {}
    for n in (10, 100):
        _dense_store(tmp_path / f"d{n}.zarr", n, cells, 0.5)
        write_sparse_footprints(tmp_path / f"d{n}.zarr", tmp_path / "warm.zarr")  # lazy imports
        tracemalloc.start()
        try:
            write_sparse_footprints(tmp_path / f"d{n}.zarr", tmp_path / f"s{n}.zarr")
            peaks[n] = tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()
    # Holding every release's entries (12 bytes each) would add ~16 MB at 100 releases.
    held_100 = 100 * int(np.prod(cells)) // 2 * 12
    assert peaks[100] < 2 * peaks[10], peaks
    assert peaks[100] - peaks[10] < held_100 / 10, peaks
