"""Sparse (COO per release) footprint storage (task B2).

Most cells of a footprint are zero: a release's particles visit a small part of
the grid. ``write_sparse_footprints`` converts a dense ``footprints.zarr`` into a
store holding, for every release, the indices and values of its non-zero cells
only. It counts, then writes, one release at a time, so memory stays at one
release's footprint whatever the number of releases.
``load_sparse_footprint`` rebuilds the dense ``footprint`` DataArray, for all
releases or a selection, with the dense store's coordinates and attributes.

Layout (a Zarr store readable with xarray): ``value`` (float32) and the
indices ``t``, ``z``, ``y``, ``x`` (int16 when the axis has fewer than 32,768
entries, else int32; separate indices compress far better than one flat index)
along dimension ``entry``, release after release; ``release_offset`` (int64,
length R + 1) gives each release's slice of ``entry``. Every coordinate of the
dense store is copied, and the attribute ``glide_sparse_shape`` records the
dense shape.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import xarray as xr

DIMS = ("release", "time_ago", "z_bin", "latitude", "longitude")


ENTRY_CHUNK = 1 << 18  # entries per Zarr chunk along ``entry`` (1 MB of float32 values)


def write_sparse_footprints(dense_path: str | Path, sparse_path: str | Path) -> dict[str, int]:
    """Convert a dense footprint store to the sparse layout; return size counts.

    Memory stays at about one release's dense footprint plus one entry chunk,
    whatever the number of releases: a first pass counts each release's non-zero
    cells to fix ``release_offset``, the store is created at its final size, and
    a second pass buffers entries release by release and writes them out a whole
    number of chunks at a time (no read-modify-write of partial chunks).
    """

    import dask.array as da
    import zarr

    dense = xr.open_zarr(dense_path)
    fp = dense["footprint"]
    shape = tuple(int(fp.sizes[d]) for d in DIMS)
    names = ("t", "z", "y", "x")
    keys = ("value", *names)
    dtypes = [np.int16 if n < 32768 else np.int32 for n in shape[1:]]

    def release(r: int) -> np.ndarray:
        return np.asarray(fp.isel(release=r).values)

    offsets = np.zeros(shape[0] + 1, dtype=np.int64)
    for r in range(shape[0]):
        offsets[r + 1] = offsets[r] + np.count_nonzero(release(r))
    n = int(offsets[-1])

    # Create the store at its final size, coordinates and attributes written by
    # xarray as before. The entry variables are lazy zeros, computed one chunk
    # at a time; the second pass overwrites every chunk.
    chunk = max(1, min(ENTRY_CHUNK, n))
    data = {
        k: ("entry", da.zeros(n, dtype=dt, chunks=chunk))
        for k, dt in zip(keys, (np.float32, *dtypes), strict=True)
    }
    data["release_offset"] = ("release_edge", offsets)
    out = xr.Dataset(data, coords=dict(dense.coords.items()))
    out.attrs.update(dense.attrs)
    out.attrs["glide_sparse_shape"] = json.dumps(shape)
    out.attrs["glide_sparse_layout"] = "COO per release (task B2): see lpdm.sparse_footprint"
    out.to_zarr(sparse_path, mode="w", compute=False).compute(scheduler="synchronous")

    group = zarr.open_group(str(sparse_path), mode="r+")
    buf: dict[str, list[np.ndarray]] = {k: [] for k in keys}
    start = 0  # first entry not yet written
    held = 0  # entries buffered

    def flush(final: bool) -> None:
        nonlocal start, held
        m = held if final else held // chunk * chunk
        if m == 0:
            return
        for k in keys:
            arr = np.concatenate(buf[k])
            group[k][start : start + m] = arr[:m]
            buf[k] = [arr[m:]]
        start += m
        held -= m

    for r in range(shape[0]):
        if offsets[r + 1] == offsets[r]:
            continue
        block = release(r)
        idx = np.nonzero(block)
        buf["value"].append(block[idx].astype(np.float32))
        for name, i, dt in zip(names, idx, dtypes, strict=True):
            buf[name].append(i.astype(dt))
        held += int(idx[0].size)
        if held >= chunk:
            flush(final=False)
    flush(final=True)
    if start != n:
        raise RuntimeError(f"sparse footprint: wrote {start} entries, counted {n}")
    return {"cells": int(np.prod(shape)), "nonzero": n}


def load_sparse_footprint(
    path: str | Path, releases: slice | list[int] | None = None
) -> xr.DataArray:
    """Rebuild the dense ``footprint`` (release, time_ago, z_bin, latitude, longitude)."""

    ds = xr.open_zarr(path)
    shape = tuple(json.loads(ds.attrs["glide_sparse_shape"]))
    all_r = np.arange(shape[0])
    sel = all_r if releases is None else all_r[releases]
    off = ds["release_offset"].values
    out = np.zeros((len(sel),) + shape[1:], dtype=np.float32)
    for j, r in enumerate(sel):
        a, b = int(off[r]), int(off[r + 1])
        if b > a:
            e = ds.isel(entry=slice(a, b))
            out[j, e["t"].values, e["z"].values, e["y"].values, e["x"].values] = e["value"].values
    coords = {
        k: (v.isel(release=sel) if "release" in v.dims else v)
        for k, v in ds.coords.items()
        if set(v.dims) <= set(DIMS)
    }
    da = xr.DataArray(out, dims=DIMS, coords=coords, name="footprint")
    attrs = {k: v for k, v in ds.attrs.items() if not k.startswith("glide_sparse")}
    da.attrs.update(attrs)
    return da
