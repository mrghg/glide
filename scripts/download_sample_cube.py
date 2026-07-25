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

Vertical coordinate (either mode): ``--levels pressure`` (default) downloads the
37 pressure levels from the unified store; ``--levels model`` downloads ERA5's
137 native hybrid model levels (finer near-surface, terrain-following) and merges
in the surface fields from the pressure/surface store, since the model-level store
carries none. Named-domain model cubes are written with an ``_ml`` suffix
(``<DOMAIN>_<YYYYMM>_ml.zarr``) and every cube records its type in the
``glide_vertical_coordinate`` attr, so the two met types are distinguishable on
disk. Model-level stores use a ``hybrid`` vertical coord (vs ``level``), so the
reader must be pointed at it with ``level_name="hybrid"``.

       python scripts/download_sample_cube.py --domain EUROPE --year-month 202401 --levels model
"""

from __future__ import annotations

import argparse
import calendar
import os
import shutil
from pathlib import Path

import numpy as np
import xarray as xr

# ARCO ERA5 analysis-ready source stores (0.25 deg, hourly, on GCS).
#
# - Pressure levels: ONE "unified" store carrying both the 37-pressure-level 3D
#   fields and the surface fields.
# - Model levels: the 3D fields live on ERA5's 137 native hybrid levels in a
#   SEPARATE store that has NO surface fields. The surface fields must therefore
#   be merged in from the pressure/surface store. Confirmed against the store
#   metadata 2026-07-25: the model-level store provides `geopotential` directly
#   on the hybrid levels, so the geometric-height/AGL conversion is unchanged
#   (no hybrid a/b coefficients or hydrostatic integration needed).
PRESSURE_LEVEL_STORE = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
MODEL_LEVEL_STORE = "gs://gcp-public-data-arco-era5/ar/model-level-1h-0p25deg.zarr-v1"

# 3D fields, carried on whichever vertical coordinate the chosen store uses
# (`level` for pressure levels, `hybrid` for model levels).
# - u/v/vertical_velocity: advection (vertical_velocity is omega, Pa/s).
# - temperature: needed for the omega->w conversion.
# - geopotential: geometric-height / AGL conversion.
# - specific_humidity: Emanuel deep convection (docs/convection.md). The example
#   periodic config enables convection, so a cube without it cannot run it.
THREE_D_VARS = [
    "u_component_of_wind",
    "v_component_of_wind",
    "vertical_velocity",
    "temperature",
    "geopotential",
    "specific_humidity",
]

# Surface / single-level fields (2D: time, lat, lon), always from the
# pressure/surface store. friction_velocity and surface_sensible_heat_flux drive
# the Hanna turbulence scheme (docs/turbulence.md); geopotential_at_surface is
# the orography used by the terrain-following AGL conversion.
SURFACE_VARS = [
    "boundary_layer_height",
    "surface_pressure",
    "geopotential_at_surface",
    "friction_velocity",
    "surface_sensible_heat_flux",
]

# The full unified pressure-level list (3D + surface) lives in one store.
REQUIRED_VARS = THREE_D_VARS + SURFACE_VARS


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


def _normalise_longitude_to_ascending(ds: xr.Dataset) -> xr.Dataset:
    """Force a strictly ascending longitude coord on the dataset.

    ARCO ERA5 uses 0..360 longitude. A bbox that wraps Greenwich is built by
    concatenating two halves (e.g. ``[262..359.75]`` then ``[0..39.5]``),
    which leaves a non-monotonic 1D index. Downstream consumers using
    ``.sel(slice(...))`` would either error (KeyError on bound lookup) or
    quietly skip dask-graph pruning, forcing the full domain into memory.

    Remapping ``lon >= 180`` to ``lon - 360`` restores monotonicity in the
    common single-wrap case via a lazy ``assign_coords`` (no chunk writes).
    A fallback ``isel`` re-sort handles unusual layouts. The reader applies
    the same transform on read; doing it at write time means future tools
    (xarray, ncview, anything else) also see a clean coordinate.
    """

    lon = np.asarray(ds["longitude"].values, dtype=np.float64)
    if lon.size <= 1 or np.all(np.diff(lon) > 0):
        return ds

    new_lon = np.where(lon >= 180.0, lon - 360.0, lon)
    if np.all(np.diff(new_lon) > 0):
        return ds.assign_coords(longitude=new_lon)

    order = np.argsort(new_lon)
    return ds.assign_coords(longitude=new_lon).isel(longitude=order)


def _prepare_for_zarr_write(ds: xr.Dataset, zarr_version: int) -> xr.Dataset:
    """Strip source encodings that don't fit a spatially-subset write.

    Two distinct issues are handled here:

    1. **Chunk-shape mismatch (both v2 and v3).** ARCO ERA5 ships with globe-shaped
       chunks (e.g. ``(1, 721, 1440)`` for surface fields). After ``.sel`` to a
       smaller bbox the dask chunks are smaller and don't fill a full source chunk
       anymore. Writing them into the inherited zarr chunk shape would either be
       parallel-unsafe (multiple dask chunks land in one zarr chunk) or pad with
       junk. We drop ``chunks`` and ``preferred_chunks`` so xarray derives the
       output chunk shape from dask instead.

    2. **Codec incompatibility (v3 only).** v2-style numcodecs objects (Blosc,
       etc.) are rejected by Zarr v3's codec API. For v3 we clear the full
       encoding; v2 keeps compressor/dtype/etc.
    """

    ds_out = ds.copy(deep=False)
    for var_name in ds_out.variables:
        if zarr_version == 3:
            ds_out[var_name].encoding = {}
        else:
            enc = dict(ds_out[var_name].encoding)
            enc.pop("chunks", None)
            enc.pop("preferred_chunks", None)
            ds_out[var_name].encoding = enc
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


def _validate_written_store(out_path: str, expected_vars: list[str]) -> None:
    """Stream-check the written store for non-finite values without loading whole vars into RAM.

    Uses xarray's dask-backed reductions: ``isfinite().all()`` lazily evaluates
    per-chunk and only the scalar result is materialised. Required for the
    multi-GB EUROPE archives, where loading a full variable via ``.values``
    would OOM.
    """

    ds = xr.open_zarr(out_path, consolidated=True)
    try:
        for var_name in expected_vars:
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


def _require_vars(ds: xr.Dataset, required: list[str], store_uri: str) -> None:
    missing = [v for v in required if v not in ds.variables]
    if missing:
        raise ValueError(f"Missing required variables in {store_uri!r}: {missing}")


def _slice_domain(
    ds: xr.Dataset,
    time_start: str,
    time_end: str,
    lon_min: float,
    lon_max: float,
    lat_min: float,
    lat_max: float,
) -> xr.Dataset:
    """Slice a global ARCO dataset to the requested time/lat/lon box.

    Handles ERA5's 0..360 longitude (incl. Greenwich-crossing boxes) and
    descending latitude, and returns a store with a strictly ascending
    -180..180 longitude coordinate. Purely lazy — no chunk data is read.
    """

    # ERA5 uses 0..360 for longitude. Convert negative requested longitudes.
    req_lon_min = lon_min + 360.0 if lon_min < 0 else lon_min
    req_lon_max = lon_max + 360.0 if lon_max < 0 else lon_max

    # ERA5 latitudes are typically stored descending (90 to -90).
    lat_values = ds["latitude"].values
    if lat_values[0] > lat_values[-1]:
        lat_slice = slice(lat_max, lat_min)
    else:
        lat_slice = slice(lat_min, lat_max)

    ds_subset = ds.sel({"time": slice(time_start, time_end), "latitude": lat_slice})

    # Handle longitudes bridging the 0/360 wrap.
    if req_lon_min <= req_lon_max:
        ds_subset = ds_subset.sel({"longitude": slice(req_lon_min, req_lon_max)})
    else:
        ds1 = ds_subset.sel({"longitude": slice(req_lon_min, None)})
        ds2 = ds_subset.sel({"longitude": slice(None, req_lon_max)})
        ds_subset = xr.concat([ds1, ds2], dim="longitude")

    # Normalise the stored longitude to a strictly ascending -180..180 coord.
    # Two halves of a Greenwich-crossing bbox were concatenated above (e.g.
    # [262..359.75] then [0..39.5]); without this step the on-disk index is
    # non-monotonic, which breaks .sel(slice(...)) lookups in any downstream
    # consumer. The conversion only relabels the coord, so it's a lazy op
    # that does not touch chunk data.
    return _normalise_longitude_to_ascending(ds_subset)


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
    *,
    levels: str = "pressure",
    surface_store_uri: str = PRESSURE_LEVEL_STORE,
):
    """Download a localised ERA5 cube on pressure levels or native model levels.

    ``levels="pressure"`` (default): all fields come from the single unified
    pressure/surface store ``store_uri`` (the 3D fields sit on a ``level`` coord).

    ``levels="model"``: the 3D fields come from the model-level store
    ``store_uri`` (on a 137-deep ``hybrid`` coord) and the surface fields are
    merged in from ``surface_store_uri`` — the model-level store has none.
    """

    print(f"Slicing time: {time_start} to {time_end}")
    print(f"Slicing spatial: Lat [{lat_min}, {lat_max}], Lon [{lon_min}, {lon_max}]")
    box = (time_start, time_end, lon_min, lon_max, lat_min, lat_max)
    # Public ARCO bucket access should use anonymous GCS token via gcsfs.
    anon = {"token": "anon"}

    if levels == "pressure":
        print(f"Opening unified pressure/surface store at {store_uri}...")
        ds = xr.open_zarr(store_uri, consolidated=True, storage_options=anon)
        _require_vars(ds, REQUIRED_VARS, store_uri)
        ds_subset = _slice_domain(ds[REQUIRED_VARS], *box)
        written_vars = REQUIRED_VARS
    elif levels == "model":
        print(f"Opening model-level 3D store at {store_uri}...")
        ds_ml = xr.open_zarr(store_uri, consolidated=True, storage_options=anon)
        _require_vars(ds_ml, THREE_D_VARS, store_uri)
        print(f"Opening surface store at {surface_store_uri}...")
        ds_sfc = xr.open_zarr(surface_store_uri, consolidated=True, storage_options=anon)
        _require_vars(ds_sfc, SURFACE_VARS, surface_store_uri)

        ds_3d = _slice_domain(ds_ml[THREE_D_VARS], *box)
        ds_surface = _slice_domain(ds_sfc[SURFACE_VARS], *box)
        # Both are 0.25 deg ar products on identical time/lat/lon grids, so an
        # exact-join merge aligns them; any mismatch fails loudly rather than
        # silently dropping timestamps. The 3D store carries the `hybrid` coord;
        # the surface fields have no vertical dim, so there is no name collision.
        print("Merging model-level 3D fields with surface fields...")
        ds_subset = xr.merge([ds_3d, ds_surface], join="exact", combine_attrs="drop_conflicts")
        written_vars = THREE_D_VARS + SURFACE_VARS
    else:
        raise ValueError(f"Unknown levels={levels!r}; expected 'pressure' or 'model'.")

    # Persist provenance (incl. the vertical coordinate type) into the store's
    # attrs so we can answer "what is this?" by opening the Zarr alone, and so
    # consumers can tell pressure- from model-level cubes apart.
    attrs = dict(archive_attrs or {})
    attrs["glide_vertical_coordinate"] = "model_level" if levels == "model" else "pressure_level"
    ds_subset.attrs = {**ds_subset.attrs, **attrs}

    print(
        f"Subset computed. Estimated size in memory (uncompressed): {ds_subset.nbytes / (1024**3):.2f} GB"
    )

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

    _validate_written_store(tmp_out_path, written_vars)
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
        "--levels",
        choices=["pressure", "model"],
        default="pressure",
        help=(
            "Vertical coordinate to download. 'pressure' (default): 37 pressure "
            "levels from the unified store. 'model': ERA5's 137 native hybrid "
            "levels (finer near-surface, terrain-following) merged with surface "
            "fields from the pressure/surface store. Named-domain model cubes get "
            "an '_ml' filename suffix."
        ),
    )
    parser.add_argument(
        "--store-uri",
        default=None,
        help=(
            "ARCO ERA5 3D-field Zarr store on GCS. Defaults to the store matching "
            f"--levels: pressure -> {PRESSURE_LEVEL_STORE}; model -> {MODEL_LEVEL_STORE}."
        ),
    )
    parser.add_argument(
        "--surface-store-uri",
        default=PRESSURE_LEVEL_STORE,
        help=(
            "Store to source the 2D surface fields from when --levels model (the "
            "model-level store has none). Ignored for --levels pressure."
        ),
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
        help=(
            "Parent directory for named-domain stores. Filename is always auto-generated "
            "as <DOMAIN>_<YYYYMM>.zarr inside this directory — point it at an external "
            "drive or mounted volume to write elsewhere (e.g. --out-dir /Volumes/external/met)."
        ),
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
    # Resolve the 3D-field store from --levels unless explicitly overridden.
    store_uri = args.store_uri or (
        MODEL_LEVEL_STORE if args.levels == "model" else PRESSURE_LEVEL_STORE
    )
    level_suffix = "_ml" if args.levels == "model" else ""

    using_named = args.domain is not None or args.year_month is not None
    using_adhoc = any(
        v is not None
        for v in (
            args.out_path,
            args.time_start,
            args.time_end,
            args.lon_min,
            args.lon_max,
            args.lat_min,
            args.lat_max,
        )
    )

    if using_named and using_adhoc:
        raise SystemExit(
            "Cannot mix named-domain mode (--domain/--year-month) with ad-hoc flags "
            "(--out-path/--time-*/--lon-*/--lat-*). Pick one. To write a named-domain "
            "download to a custom location, set --out-dir (the filename is always auto-generated)."
        )

    if using_named:
        if args.domain is None or args.year_month is None:
            raise SystemExit("--domain and --year-month must be given together.")
        bbox = _resolve_domain_bbox(args.domain)
        t_start, t_end = _resolve_year_month_window(args.year_month)
        out_path = os.path.join(args.out_dir, f"{args.domain}_{args.year_month}{level_suffix}.zarr")
        archive_attrs = {
            "glide_domain": args.domain,
            "glide_year_month": args.year_month,
            "glide_source_store": store_uri,
            "glide_domain_description": str(DOMAINS[args.domain]["description"]),
        }
        if args.levels == "model":
            archive_attrs["glide_surface_store"] = args.surface_store_uri
        download_sample_cube(
            out_path=out_path,
            store_uri=store_uri,
            time_start=t_start,
            time_end=t_end,
            lon_min=bbox["lon_min"],
            lon_max=bbox["lon_max"],
            lat_min=bbox["lat_min"],
            lat_max=bbox["lat_max"],
            zarr_version=args.zarr_version,
            archive_attrs=archive_attrs,
            levels=args.levels,
            surface_store_uri=args.surface_store_uri,
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
        store_uri=store_uri,
        time_start=args.time_start,
        time_end=args.time_end,
        lon_min=args.lon_min,
        lon_max=args.lon_max,
        lat_min=args.lat_min,
        lat_max=args.lat_max,
        zarr_version=args.zarr_version,
        levels=args.levels,
        surface_store_uri=args.surface_store_uri,
    )


if __name__ == "__main__":
    parser = _build_arg_parser()
    args = parser.parse_args()
    _dispatch(args)
