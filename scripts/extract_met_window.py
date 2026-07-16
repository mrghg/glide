#!/usr/bin/env python
"""Extract a single hour of meteorology from the EUROPE ERA5 store.

Pulls one timestep of the fields needed to diagnose the terrain / AGL handling
(geopotential + geopotential_at_surface give orography and per-column AGL;
temperature and the winds let the Hanna free-troposphere closure be reproduced
off-line), writes them to a Zarr store and tars it into a single file to copy.

Uses only GLIDE's core dependencies (xarray + zarr + dask) — no `[viz]` extra,
no netCDF backend — so it runs in a bare compute-node venv.

Run on Isambard from the repo root:

    .venv/bin/python scripts/extract_met_window.py --time 2024-01-15T12:00

then copy the printed .tar.gz back. Defaults match the icos-202401 run's
`io.zarr_store`. Add `--full-domain` for the whole stored grid (~4x bigger).
"""

from __future__ import annotations

import argparse
import glob
import shutil
import sys
import tarfile
from pathlib import Path

import numpy as np
import xarray as xr

# Logical -> store variable names (mirrors MetReader.DEFAULT_VARIABLE_MAP).
DEFAULT_VARS = (
    "geopotential",              # 3D: per-column height -> AGL
    "geopotential_at_surface",   # 2D: orography (the terrain we're testing against)
    "temperature",               # 3D: potential temperature -> N^2
    "u_component_of_wind",       # 3D: shear
    "v_component_of_wind",       # 3D: shear
    "boundary_layer_height",     # 2D: BL depth
)

# Default crop: the Europe view where the terrain holes are (Iberia, Alps,
# Scandes) plus enough Atlantic to have a sea-level control.
DEFAULT_EXTENT = (-30.0, 30.0, 35.0, 72.0)  # lon_min, lon_max, lat_min, lat_max


def open_store_for_time(pattern: str, when: np.datetime64) -> xr.Dataset:
    """Open whichever monthly EUROPE store contains `when`."""
    expanded = str(Path(pattern).expanduser())
    stores = sorted(glob.glob(expanded)) if any(c in expanded for c in "*?[") else [expanded]
    if not stores:
        raise SystemExit(f"no stores matched {pattern!r}")

    for store in stores:
        ds = xr.open_zarr(store, consolidated=True, chunks="auto")
        t = ds["time"].values
        if t.min() <= when <= t.max():
            print(f"found {when} in {store}")
            return ds
        ds.close()
    raise SystemExit(
        f"{when} not covered by any of:\n  " + "\n  ".join(stores) +
        "\nPick a --time inside one of these stores."
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--zarr-store", default="~/data/arco-era5/EUROPE_*.zarr",
                   help="store or glob (default: the icos-202401 run's io.zarr_store)")
    p.add_argument("--time", default="2024-01-15T12:00", help="timestep to extract (UTC)")
    p.add_argument("--out-dir", type=Path, default=Path("."), help="where to write the archive")
    p.add_argument("--extent", type=float, nargs=4, default=list(DEFAULT_EXTENT),
                   metavar=("LON_MIN", "LON_MAX", "LAT_MIN", "LAT_MAX"))
    p.add_argument("--full-domain", action="store_true", help="no spatial crop (~4x larger)")
    p.add_argument("--vars", nargs="+", default=list(DEFAULT_VARS))
    p.add_argument("--keep-zarr", action="store_true", help="don't delete the .zarr after tarring")
    args = p.parse_args(argv)

    when = np.datetime64(args.time)
    ds = open_store_for_time(args.zarr_store, when)

    missing = [v for v in args.vars if v not in ds.variables]
    if missing:
        raise SystemExit(f"store is missing {missing}\navailable: {sorted(ds.data_vars)}")

    snap = ds[list(args.vars)].sel(time=when, method="nearest")
    actual = np.datetime_as_string(snap["time"].values, unit="h")

    if not args.full_domain:
        lon_min, lon_max, lat_min, lat_max = args.extent
        lat = ds["latitude"].values
        # ERA5 latitude is commonly stored descending; slice() needs matching order.
        lat_slice = slice(lat_max, lat_min) if lat[0] > lat[-1] else slice(lat_min, lat_max)
        snap = snap.sel(longitude=slice(lon_min, lon_max), latitude=lat_slice)

    snap = snap.astype("float32")
    for name in ("latitude", "longitude"):
        print(f"  {name}: {snap.sizes[name]} points "
              f"({float(snap[name].min()):.2f} .. {float(snap[name].max()):.2f})")
    if "level" in snap.sizes:
        print(f"  level: {snap.sizes['level']} pressure levels")

    nbytes = sum(v.nbytes for v in snap.data_vars.values())
    print(f"  extracting {len(args.vars)} vars, ~{nbytes / 2**20:.0f} MiB uncompressed")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"glide_met_{actual.replace(':', '')}"
    zarr_path = args.out_dir / f"{stem}.zarr"
    tar_path = args.out_dir / f"{stem}.tar.gz"
    if zarr_path.exists():
        shutil.rmtree(zarr_path)

    # Inherited encodings can carry codecs the writer rejects; drop them.
    for v in snap.variables:
        snap[v].encoding = {}
    snap.load().to_zarr(zarr_path, mode="w", consolidated=True)

    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(zarr_path, arcname=zarr_path.name)
    if not args.keep_zarr:
        shutil.rmtree(zarr_path)

    print(f"\nwrote {tar_path}  ({tar_path.stat().st_size / 2**20:.1f} MiB)")
    print("\nCopy it back with:")
    print(f"  scp <isambard>:{Path(tar_path).resolve()} .")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
