"""Effect of met-bazarr decision 0009's surface-geopotential repair on the
0-40 m footprints of standard cases 2 (JFJ) and 3 (CBW-20).

    .venv/bin/python scripts/validation/zs-repair/compare.py [noise-tag]

Compares outputs/zs-repair-<case>-before with -after: total 0-40 m sensitivity,
the log ratio over cells above and below 1,000 m terrain, and correlation of
the log footprints. With a noise tag (e.g. after-repeat), the same comparison
between two runs on the same repaired store gives the run-to-run noise.
"""

import sys

import numpy as np
import xarray as xr

STORES = {
    "case2-jfj": "data/era5-arco/EUR_20190621-20190715_ERA5-ARCO.zarr",
    "case3-cbw20": "data/era5-arco/EUR_20181222-20190115_ERA5-ARCO.zarr",
}


def surface(tag: str, case: str) -> xr.DataArray:
    fp = xr.open_zarr(f"outputs/zs-repair-{case}-{tag}/footprints.zarr")["footprint"]
    return fp.isel(release=0, z_bin=0).sum("time_ago").load()


def compare(a: xr.DataArray, b: xr.DataArray, terrain_m: np.ndarray) -> dict:
    both = (a.values > 0) & (b.values > 0)
    lr = np.log10(b.values[both] / a.values[both])
    w = a.values[both]
    high = terrain_m[both] > 1000.0
    out = {
        "total_ratio": float(b.sum() / a.sum()),
        "log_corr": float(np.corrcoef(np.log10(a.values[both]), np.log10(b.values[both]))[0, 1]),
        "weighted_mean_abs_log10_ratio": float(np.sum(w * np.abs(lr)) / np.sum(w)),
    }
    for name, m in (("terrain_above_1000m", high), ("terrain_below_1000m", ~high)):
        out[name] = {
            "share_of_sensitivity": float(w[m].sum() / w.sum()),
            "sum_ratio": float(b.values[both][m].sum() / a.values[both][m].sum())
            if m.any()
            else None,
            "weighted_mean_abs_log10_ratio": float(np.sum(w[m] * np.abs(lr[m])) / np.sum(w[m]))
            if m.any()
            else None,
        }
    return out


def main(noise_tag: str | None) -> None:
    for case, store in STORES.items():
        zs = xr.open_zarr(store)["geopotential_at_surface"].isel(time=0)
        before, after = surface("before", case), surface("after", case)
        terrain = zs.interp(latitude=before.latitude, longitude=before.longitude).values / 9.80665
        print(case, "repair:", compare(before, after, terrain))
        if noise_tag:
            print(case, "noise: ", compare(after, surface(noise_tag, case), terrain))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
