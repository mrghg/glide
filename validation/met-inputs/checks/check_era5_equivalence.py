"""Compare a met-bazarr ERA5 ARCO store with one written by GLIDE's own
downloader (``scripts/download_sample_cube.py --levels model`` in GLIDE).

Every variable except ``friction_velocity`` must be bit-identical at every time
the two stores share: met-bazarr copies them unchanged (decision 0006). u* is
derived from the surface stresses here, where GLIDE's downloader took ERA5's
``zust``, so it is compared, not required to match.

Usage::

    $METBAZARR_REPO/.venv/bin/python validation/met-inputs/checks/check_era5_equivalence.py NEW.zarr GLIDE_STORE.zarr
"""

from __future__ import annotations

import sys

import numpy as np
import xarray as xr

COPIED = (
    "u_component_of_wind",
    "v_component_of_wind",
    "vertical_velocity",
    "temperature",
    "specific_humidity",
    "geopotential",
    "boundary_layer_height",
    "surface_pressure",
    "geopotential_at_surface",
    "surface_sensible_heat_flux",
)


def main(new_path: str, old_path: str) -> int:
    new, old = xr.open_zarr(new_path), xr.open_zarr(old_path)
    times = np.intersect1d(new.time.values, old.time.values)
    if times.size == 0:
        print("FAIL: the stores share no times")
        return 1
    new, old = new.sel(time=times), old.sel(time=times)
    ok = True
    for coord in ("latitude", "longitude", "hybrid"):
        same = np.array_equal(new[coord].values, old[coord].values)
        ok &= same
        print(f"{coord:28s} {'identical' if same else 'DIFFERENT'}")
    for name in COPIED:
        a, b = new[name].values, old[name].values
        same = a.dtype == b.dtype and np.array_equal(a, b, equal_nan=True)
        ok &= same
        print(
            f"{name:28s} {'bit-identical' if same else 'DIFFERENT'}  ({a.dtype}, {a.size} values)"
        )
        if not same:
            print(f"    max |diff| {np.nanmax(np.abs(a.astype(float) - b)):.3g}")
    a = new["friction_velocity"].values.astype(np.float64)
    b = old["friction_velocity"].values.astype(np.float64)
    ratio = a / np.where(b > 0, b, np.nan)
    print(
        f"{'friction_velocity':28s} from stresses vs ERA5 zust: median ratio "
        f"{np.nanmedian(ratio):.3f}, 5-95% {np.nanpercentile(ratio, 5):.3f}-"
        f"{np.nanpercentile(ratio, 95):.3f}; medians {np.median(a):.3f} vs "
        f"{np.median(b):.3f} m s-1"
    )
    print(f"{times.size} shared times. {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:3]))
