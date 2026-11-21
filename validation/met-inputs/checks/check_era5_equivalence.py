"""Compare a met-bazarr ERA5 ARCO store with one written by GLIDE's own
downloader (``scripts/download_sample_cube.py --levels model`` in GLIDE).

Every variable except ``friction_velocity`` must be bit-identical at every time
the two stores share: met-bazarr copies them unchanged (decision 0006). u* is
derived from the surface stresses here, where GLIDE's downloader took ERA5's
``zust``, so it is compared, not required to match.

Usage::

    $METBAZARR_REPO/.venv/bin/python validation/met-inputs/checks/check_era5_equivalence.py NEW.zarr GLIDE_STORE.zarr [STRIDE]

Times are compared one at a time, so stores of any size fit in memory;
STRIDE (default 1) compares every STRIDE-th shared time.
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


def main(new_path: str, old_path: str, stride: int = 1) -> int:
    """Compare one time at a time, so a store of any size fits in memory.

    ``stride`` compares every ``stride``-th shared time (1: all of them).
    """
    new, old = xr.open_zarr(new_path), xr.open_zarr(old_path)
    times = np.intersect1d(new.time.values, old.time.values)[::stride]
    if times.size == 0:
        print("FAIL: the stores share no times")
        return 1
    ok = True
    for coord in ("latitude", "longitude", "hybrid"):
        same = np.array_equal(new[coord].values, old[coord].values)
        ok &= same
        print(f"{coord:28s} {'identical' if same else 'DIFFERENT'}")
    differ = dict.fromkeys(COPIED, 0)
    max_diff = dict.fromkeys(COPIED, 0.0)
    ratios = []
    n_values = dict.fromkeys(COPIED, 0)
    for t in times:
        a_t, b_t = new.sel(time=t).load(), old.sel(time=t).load()
        for name in COPIED:
            a, b = a_t[name].values, b_t[name].values
            n_values[name] += a.size
            if not (a.dtype == b.dtype and np.array_equal(a, b, equal_nan=True)):
                differ[name] += 1
                max_diff[name] = max(max_diff[name], float(np.nanmax(np.abs(a.astype(float) - b))))
        ua = a_t["friction_velocity"].values.astype(np.float64)
        ub = b_t["friction_velocity"].values.astype(np.float64)
        ratios.append((ua / np.where(ub > 0, ub, np.nan)).ravel()[::97])
    for name in COPIED:
        same = differ[name] == 0
        ok &= same
        print(
            f"{name:28s} {'bit-identical' if same else 'DIFFERENT'}  "
            f"({n_values[name]} values"
            + ("" if same else f"; {differ[name]} times differ, max |diff| {max_diff[name]:.3g}")
            + ")"
        )
    ratio = np.concatenate(ratios)
    print(
        f"{'friction_velocity':28s} from stresses vs ERA5 zust (sampled): median ratio "
        f"{np.nanmedian(ratio):.3f}, 5-95% {np.nanpercentile(ratio, 5):.3f}-"
        f"{np.nanpercentile(ratio, 95):.3f}"
    )
    print(f"{times.size} times compared (stride {stride}). {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    args = sys.argv[1:]
    stride = int(args[2]) if len(args) > 2 else 1
    sys.exit(main(args[0], args[1], stride))
