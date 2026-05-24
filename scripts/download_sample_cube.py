"""Download a localized ERA5 datacube for local LPDM testing.

Two invocation modes:

1. **Named domain + month** (preferred for the FLEXPART comparison work):

       python scripts/download_sample_cube.py --domain EUROPE --year-month 202401

   Writes to ``data/era5/<DOMAIN>_<YYYYMM>.zarr``. Bounding box and full pressure
   level set are looked up from :data:`DOMAINS`. Each month is its own Zarr store
   so multi-month archives are resumable and self-documenting on disk.

2. **Ad-hoc subset** (legacy SF-area smoke tests, custom one-off windows):

       python scripts/download_sample_cube.py --out-path data/sample_met.zarr \
           --time-start 2023-12-29T18:00:00 --time-end 2024-01-01T06:00:00 \
           --lon-min -127.0 --lon-max -117.0 --lat-min 33.0 --lat-max 43.0

   You must provide ``--out-path``, ``--time-*`` and all four lon/lat bounds.
"""

from __future__ import annotations

import argparse
import calendar
import os
import shutil
from pathlib import Path

import numpy as np
import xarray as xr


# Logical to ERA5 physical mapping required by the runtime.
# friction_velocity and surface_sensible_heat_flux are required by the Hanna
# turbulence scheme (see docs/turbulence.md). They are surface fields so adding
# them is cheap; downloading them keeps the local cube ready for either
# placeholder or Hanna runs.
REQUIRED_VARS = [
    "u_component_of_wind",
    "v_component_of_wind",
    "vertical_velocity",
    "boundary_layer_height",
    "surface_pressure",
    "temperature",
    "geopotential",
    "geopotential_at_surface",
    "friction_velocity",
    "surface_sensible_heat_flux",
]


# Registry of named meteorological-archive domains. Add new domains here rather
# than hard-coding bboxes at call sites. lon/lat bounds are inclusive cell-centre
# extents that fully cover the target region; the script handles the negative-to-
# 0..360 conversion for ERA5 internally.
DOMAINS: dict[str, dict[str, float | str]] = {
    "EUROPE": {
        "lon_min": -98.0,
        "lon_max": 39.5,
        "lat_min": 10.6,
        "lat_max": 79.2,
        "description": (
            "Mace Head-centred FLEXPART-EUROPE comparison domain. Matches the "
            "extents of data/FLEXPART/FLEXPART_MHD_test_202401.nc."
        ),
    },
}


def _prepare_for_zarr_write(ds: xr.Dataset, zarr_version: int) -> xr.Dataset:
    """Remove source encodings that are incompatible with target Zarr format."""

    if zarr_version != 3:
        return ds

    # ARCO source metadata may contain v2-style numcodecs objects (e.g. Blosc)
    # in .encoding, which are rejected by Zarr v3's codec API.
    ds_out = ds.copy(deep=False)
    for var_name in ds_out.variables:
        ds_out[var_name].encoding = {}
    return ds_out


def _replace_store_atomically(tmp_path: str, out_path: str) -> None:
    out_store = Path(out_path)
    tmp_store = Path(tmp_path)
    backup_store = out_store.with_name(f"{out_store.name}.bak-replace")

    if backup_store.exists():
        shutil.rmtree(backup_store)

    try:
        if out_store.exists():
            os.replace(out_store, backup_store)
        os.replace(tmp_store, out_store)
    except Exception:
        if out_store.exists():
            shutil.rmtree(out_store)
        if backup_store.exists():
            os.replace(backup_store, out_store)
        raise
    else:
        if backup_store.exists():
            shutil.rmtree(backup_store)


def _validate_written_store(out_path: str) -> None:
    """Stream-check the written store for non-finite values without loading whole vars into RAM.

    Uses xarray's dask-backed reductions: ``isfinite().all()`` lazily evaluates
    per-chunk and only the scalar result is materialised. Required for the
    multi-GB EUROPE archives, where loading a full variable via ``.values``
    would OOM.
    """

    ds = xr.open_zarr(out_path, consolidated=True)
    try:
        for var_name in REQUIRED_VARS:
            all_finite = bool(np.isfinite(ds[var_name]).all().compute().item())
            if not all_finite:
                raise ValueError(
                    f"Downloaded store validation failed: variable {var_name!r} contains non-finite values. "
                    "The local sample cube is incomplete or corrupted, so it was not installed."
                )
    finally:
        close_fn = getattr(ds, "close", None)
        if callable(close_fn):
            close_fn()


def _resolve_year_month_window(year_month: str) -> tuple[str, str]:
    """Convert 'YYYYMM' to (time_start_iso, time_end_iso) covering the full month inclusive."""

    if len(year_month) != 6 or not year_month.isdigit():
        raise ValueError(f"--year-month must be YYYYMM (e.g. 202401), got {year_month!r}")
    year = int(year_month[:4])
    month = int(year_month[4:])
    if not (1 <= month <= 12):
        raise ValueError(f"Invalid month in --year-month {year_month!r}")
    days = calendar.monthrange(year, month)[1]
    t_start = f"{year:04d}-{month:02d}-01T00:00:00"
    t_end = f"{year:04d}-{month:02d}-{days:02d}T23:00:00"
    return t_start, t_end


def _resolve_domain_bbox(domain: str) -> dict[str, float]:
    if domain not in DOMAINS:
        known = ", ".join(sorted(DOMAINS)) or "<none>"
        raise ValueError(f"Unknown domain {domain!r}. Registered domains: {known}")
    spec = DOMAINS[domain]
    return {
        "lon_min": float(spec["lon_min"]),
        "lon_max": float(spec["lon_max"]),
        "lat_min": float(spec["lat_min"]),
        "lat_max": float(spec["lat_max"]),
    }


def download_sample_cube(
    out_path: str,
    store_uri: str,
    time_start: str,
    time_end: str,
    lon_min: float,
    lon_max: float,
    lat_min: float,
    lat_max: float,
    zarr_version: int,
    archive_attrs: dict[str, str] | None = None,
):
    print(f"Opening remote Zarr store at {store_uri}...")
    # Public ARCO bucket access should use anonymous GCS token via gcsfs.
    ds = xr.open_zarr(store_uri, consolidated=True, storage_options={"token": "anon"})

    missing = [v for v in REQUIRED_VARS if v not in ds.variables]
    if missing:
        raise ValueError(f"Missing required variables in dataset: {missing}")

    ds = ds[REQUIRED_VARS]

    # ERA5 uses 0..360 for longitude. Convert negative requested longitudes.
    req_lon_min = lon_min + 360.0 if lon_min < 0 else lon_min
    req_lon_max = lon_max + 360.0 if lon_max < 0 else lon_max

    print(f"Slicing time: {time_start} to {time_end}")
    print(f"Slicing spatial: Lat [{lat_min}, {lat_max}], Lon [{lon_min}, {lon_max}] (ERA5 lon coords [{req_lon_min}, {req_lon_max}])")

    # ERA5 latitudes are typically stored descending (90 to -90).
    lat_values = ds["latitude"].values
    if lat_values[0] > lat_values[-1]:
        lat_slice = slice(lat_max, lat_min)
    else:
        lat_slice = slice(lat_min, lat_max)

    ds_subset = ds.sel({
        "time": slice(time_start, time_end),
        "latitude": lat_slice,
    })

    # Handle longitudes bridging the 0/360 wrap.
    if req_lon_min <= req_lon_max:
        ds_subset = ds_subset.sel({"longitude": slice(req_lon_min, req_lon_max)})
    else:
        ds1 = ds_subset.sel({"longitude": slice(req_lon_min, None)})
        ds2 = ds_subset.sel({"longitude": slice(None, req_lon_max)})
        ds_subset = xr.concat([ds1, ds2], dim="longitude")

    if archive_attrs:
        # Persist provenance into the local store's attrs so we can answer
        # "what is this?" by opening the Zarr alone.
        ds_subset.attrs = {**ds_subset.attrs, **archive_attrs}

    print(f"Subset computed. Estimated size in memory (uncompressed): {ds_subset.nbytes / (1024 ** 3):.2f} GB")

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    tmp_out_path = f"{out_path}.tmp-download"
    if os.path.exists(tmp_out_path):
        shutil.rmtree(tmp_out_path)

    print(f"Downloading and saving data to temporary local Zarr: {tmp_out_path}...")
    ds_to_write = _prepare_for_zarr_write(ds_subset, zarr_version=zarr_version)

    with xr.set_options(keep_attrs=True):
        ds_to_write.to_zarr(
            tmp_out_path,
            mode="w",
            consolidated=True,
            zarr_format=zarr_version,
        )

    _validate_written_store(tmp_out_path)
    _replace_store_atomically(tmp_out_path, out_path)

    print(f"Download and local store setup complete: {out_path}")


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download a localized ERA5 datacube for local LPDM testing.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Domains registered in DOMAINS dict at the top of this script:\n"
            + "\n".join(f"  {name}: {spec['description']}" for name, spec in DOMAINS.items())
        ),
    )
    parser.add_argument(
        "--store-uri",
        default="gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3",
        help="ARCO ERA5 Zarr store URI on GCS.",
    )
    parser.add_argument(
        "--zarr-version",
        type=int,
        choices=[2, 3],
        default=2,
        help="Output Zarr format version (2 is safest; 3 clears inherited v2 codecs).",
    )

    # Named domain + month path (preferred for the FLEXPART comparison archive).
    named = parser.add_argument_group("named-domain mode")
    named.add_argument("--domain", help=f"One of: {', '.join(sorted(DOMAINS))}")
    named.add_argument("--year-month", help="YYYYMM, e.g. 202401")
    named.add_argument(
        "--out-dir",
        default="data/era5",
        help="Parent directory for named-domain stores. Each store is <DOMAIN>_<YYYYMM>.zarr.",
    )

    # Ad-hoc subset path (legacy, kept for SF-area smoke tests and custom one-offs).
    adhoc = parser.add_argument_group("ad-hoc subset mode")
    adhoc.add_argument("--out-path", help="Full output path for ad-hoc subsets.")
    adhoc.add_argument("--time-start", help="ISO datetime, e.g. 2023-12-29T18:00:00.")
    adhoc.add_argument("--time-end", help="ISO datetime.")
    adhoc.add_argument("--lon-min", type=float)
    adhoc.add_argument("--lon-max", type=float)
    adhoc.add_argument("--lat-min", type=float)
    adhoc.add_argument("--lat-max", type=float)

    return parser


def _dispatch(args: argparse.Namespace) -> None:
    using_named = args.domain is not None or args.year_month is not None
    using_adhoc = any(
        v is not None
        for v in (args.out_path, args.time_start, args.time_end, args.lon_min, args.lon_max, args.lat_min, args.lat_max)
    )

    if using_named and using_adhoc:
        raise SystemExit(
            "Cannot mix named-domain mode (--domain/--year-month) with ad-hoc flags "
            "(--out-path/--time-*/--lon-*/--lat-*). Pick one."
        )

    if using_named:
        if args.domain is None or args.year_month is None:
            raise SystemExit("--domain and --year-month must be given together.")
        bbox = _resolve_domain_bbox(args.domain)
        t_start, t_end = _resolve_year_month_window(args.year_month)
        out_path = os.path.join(args.out_dir, f"{args.domain}_{args.year_month}.zarr")
        archive_attrs = {
            "glide_domain": args.domain,
            "glide_year_month": args.year_month,
            "glide_source_store": args.store_uri,
            "glide_domain_description": str(DOMAINS[args.domain]["description"]),
        }
        download_sample_cube(
            out_path=out_path,
            store_uri=args.store_uri,
            time_start=t_start,
            time_end=t_end,
            lon_min=bbox["lon_min"],
            lon_max=bbox["lon_max"],
            lat_min=bbox["lat_min"],
            lat_max=bbox["lat_max"],
            zarr_version=args.zarr_version,
            archive_attrs=archive_attrs,
        )
        return

    required_adhoc = (
        args.out_path,
        args.time_start,
        args.time_end,
        args.lon_min,
        args.lon_max,
        args.lat_min,
        args.lat_max,
    )
    if any(v is None for v in required_adhoc):
        raise SystemExit(
            "Ad-hoc mode requires all of: --out-path, --time-start, --time-end, "
            "--lon-min, --lon-max, --lat-min, --lat-max."
        )
    download_sample_cube(
        out_path=args.out_path,
        store_uri=args.store_uri,
        time_start=args.time_start,
        time_end=args.time_end,
        lon_min=args.lon_min,
        lon_max=args.lon_max,
        lat_min=args.lat_min,
        lat_max=args.lat_max,
        zarr_version=args.zarr_version,
    )


if __name__ == "__main__":
    parser = _build_arg_parser()
    args = parser.parse_args()
    _dispatch(args)
