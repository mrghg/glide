import os
import argparse
import xarray as xr

# Logical to ERA5 physical mapping required by ARCO Zarr
REQUIRED_VARS = [
    "u_component_of_wind",
    "v_component_of_wind",
    "vertical_velocity",
    "boundary_layer_height",
    "surface_pressure",
    "temperature",
    "geopotential",
    "geopotential_at_surface"
]


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
):
    print(f"Opening remote Zarr store at {store_uri}...")
    # Public ARCO bucket access should use anonymous GCS token via gcsfs.
    ds = xr.open_zarr(store_uri, consolidated=True, storage_options={"token": "anon"})
    
    # Select only the variables we need for LPDM
    missing = [v for v in REQUIRED_VARS if v not in ds.variables]
    if missing:
        raise ValueError(f"Missing required variables in dataset: {missing}")
    
    ds = ds[REQUIRED_VARS]

    # ERA5 generally uses 0-360 for longitude. Convert negative longitudes if needed.
    if lon_min < 0:
        lon_min += 360.0
    if lon_max < 0:
        lon_max += 360.0
        
    print(f"Slicing time: {time_start} to {time_end}")
    print(f"Slicing spatial: Lat [{lat_min}, {lat_max}], Lon [{lon_min}, {lon_max}]")
    
    # Time and latitude slicing
    # NOTE: ERA5 latitudes are typically stored descending (90 to -90)
    lat_values = ds["latitude"].values
    if lat_values[0] > lat_values[-1]:
        lat_slice = slice(lat_max, lat_min)
    else:
        lat_slice = slice(lat_min, lat_max)
        
    ds_subset = ds.sel({
        "time": slice(time_start, time_end),
        "latitude": lat_slice,
    })
    
    # Handle longitudes bridging the 0/360 point safely if needed
    if lon_min <= lon_max:
        ds_subset = ds_subset.sel({"longitude": slice(lon_min, lon_max)})
    else:
        ds1 = ds_subset.sel({"longitude": slice(lon_min, None)})
        ds2 = ds_subset.sel({"longitude": slice(None, lon_max)})
        ds_subset = xr.concat([ds1, ds2], dim="longitude")
        
    print(f"Subset computed. Estimated size in memory: {ds_subset.nbytes / (1024 ** 2):.2f} MB")
    
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        
    print(f"Downloading and saving data to local Zarr: {out_path}...")
    ds_to_write = _prepare_for_zarr_write(ds_subset, zarr_version=zarr_version)

    # Writing the filtered subset
    with xr.set_options(keep_attrs=True):
        ds_to_write.to_zarr(
            out_path,
            mode="w",
            consolidated=True,
            zarr_format=zarr_version,
        )
        
    print("Download and local store setup complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download a localized datacube for local LPDM testing.")
    parser.add_argument("--out-path", default="data/sample_met.zarr", help="Path to write the local zarr store")
    parser.add_argument("--store-uri", default="gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3")
    
    # Using defaults that match your launch.json run config for SF tests
    parser.add_argument("--time-start", default="2023-12-31T18:00:00", help="Start time to slice (gives padding for backwards run)")
    parser.add_argument("--time-end", default="2024-01-01T06:00:00", help="End time to slice")
    
    parser.add_argument("--lon-min", type=float, default=-125.0)
    parser.add_argument("--lon-max", type=float, default=-119.0)
    parser.add_argument("--lat-min", type=float, default=35.0)
    parser.add_argument("--lat-max", type=float, default=41.0)
    parser.add_argument(
        "--zarr-version",
        type=int,
        choices=[2, 3],
        default=2,
        help="Output Zarr format version (2 is safest; 3 clears inherited v2 codecs).",
    )
    
    args = parser.parse_args()
    
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
