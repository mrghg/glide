"""Task B2: footprint compression (bit-rounding, zstd) and the sparse COO layout."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import xarray as xr

from lpdm.comparison import load_sparse_footprint
from lpdm.output_writer import OutputWriter, footprint_encoding
from lpdm.sparse_footprint import write_sparse_footprints

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
