"""Task A01: GLIDE's hydrostatic model-level pressure against ECMWF's L137 a/b.

Decision 0010 reconstructs pressure on model levels from geopotential, surface
pressure and virtual temperature, keeping hybrid a/b out of production. This
compares that reconstruction with the IFS definition on real ERA5 columns:
full-level pressure = (p_half(k-1) + p_half(k)) / 2, p_half = a + b * p_s.

    .venv/bin/python scripts/validation/a01/ab_check.py data/era5-arco/EUROPE_*_ERA5-ARCO.zarr
"""

import sys
from pathlib import Path

import numpy as np
import xarray as xr

from lpdm.vertical_grid import model_level_pressure_pa

TABLE = Path(__file__).resolve().parents[3] / "tests" / "data" / "ecmwf_l137_ab.csv"


def ab_full_level_pressure(levels: np.ndarray, ps: np.ndarray) -> np.ndarray:
    tab = np.loadtxt(TABLE, delimiter=",", comments=("#", "n"))
    a, b = tab[:, 1], tab[:, 2]
    k = levels.astype(int)
    half_lo = a[k - 1][:, None, None] + b[k - 1][:, None, None] * ps[None]
    half_hi = a[k][:, None, None] + b[k][:, None, None] * ps[None]
    return 0.5 * (half_lo + half_hi)


def compare(ds: xr.Dataset) -> dict[str, np.ndarray]:
    ps = ds["surface_pressure"].values.astype(np.float64)
    p_glide = model_level_pressure_pa(
        ds["geopotential"].values,
        ds["geopotential_at_surface"].values,
        ps,
        ds["temperature"].values,
        ds["specific_humidity"].values,
    )
    p_ab = ab_full_level_pressure(ds["hybrid"].values, ps)
    return {"diff": p_glide - p_ab, "rel": (p_glide - p_ab) / p_ab, "p_ab": p_ab}


def main(paths: list[str]) -> None:
    crops = {
        "Alps": {"latitude": slice(48.5, 44.5), "longitude": slice(5.0, 12.0)},
        "Atlantic": {"latitude": slice(55.0, 50.0), "longitude": slice(-30.0, -20.0)},
        "EUROPE/8": None,
    }
    for path in paths:
        store = xr.open_zarr(path)
        for t in (0, store.sizes["time"] // 2):
            for name, sel in crops.items():
                ds = store.isel(time=t)
                ds = (
                    ds.sel(**sel)
                    if sel
                    else ds.isel(latitude=slice(None, None, 8), longitude=slice(None, None, 8))
                )
                r = compare(ds.load())
                rel, diff, p = r["rel"], r["diff"], r["p_ab"]
                low = p > 85000.0  # lowest ~1.5 km: the boundary layer and cloud base
                print(
                    f"{Path(path).name} t={str(ds.time.values)[:16]} {name:9s} "
                    f"|rel| median {np.median(np.abs(rel)):.1e} p99 {np.percentile(np.abs(rel), 99):.1e} max {np.abs(rel).max():.1e}; "
                    f"below 850 hPa: |diff| median {np.median(np.abs(diff[low])):.1f} Pa max {np.abs(diff[low]).max():.1f} Pa; "
                    f"top level rel {np.abs(rel[0]).max():.1e}"
                )


if __name__ == "__main__":
    main(sys.argv[1:])
