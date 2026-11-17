#!/usr/bin/env python
"""Pick the release times of standard cases 3-5 by the rules in validation-plan §4.

The rules use ERA5 fields only, never model output, read here from the ARCO
ERA5 pressure-level/surface store at the ERA5 grid point nearest each receptor:

- Case 3, stable night (CBW-20, EUR-Jan): of the fourteen nights 1-14 Jan 2019,
  the one with the lowest mean boundary-layer height over the hourly values
  00:00-05:00 UTC inclusive. Release at 02:00 that night. Ties: earlier night.
- Case 4, convective afternoon (LEF, NAM-Jul): of the fourteen days 1-14 Jul
  2019, the one with the highest mean boundary-layer height over the hourly
  values 18:00-22:00 UTC inclusive. Release at 20:00 that day. Ties: earlier day.
- Case 5, frontal passage (TAC, EUR-Jan): the release hour t in EUR-Jan
  (1 Jan 00:00 - 14 Jan 23:00) for which T850(t-3h) - T850(t) is largest. The
  hour t is the end of the fall and the release time. Ties: earlier hour.

Usage:  python scripts/select_case_times.py [--json out.json]
"""

from __future__ import annotations

import argparse
import json

import dask
import numpy as np
import pandas as pd
import xarray as xr

STORE = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"

# Receptor coordinates, docs/reference-runs.md §3.
CBW = (51.9703, 4.9264)
LEF = (45.9451, -90.2732)
TAC = (52.5176, 1.1387)


def nearest(ds: xr.Dataset, lat: float, lon: float) -> dict[str, float]:
    """ERA5 grid point nearest (lat, lon); ARCO longitudes run 0..360."""
    lon360 = lon % 360
    ilat = int(np.abs(ds.latitude.values - lat).argmin())
    ilon = int(np.abs(ds.longitude.values - lon360).argmin())
    return {
        "latitude": float(ds.latitude.values[ilat]),
        "longitude": float(ds.longitude.values[ilon]),
    }


def point_series(da: xr.DataArray, pt: dict[str, float], t0: str, t1: str) -> pd.Series:
    s = da.sel(latitude=pt["latitude"], longitude=pt["longitude"]).sel(time=slice(t0, t1))
    return s.compute().to_series()


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--json", help="also write the result to this file")
    args = ap.parse_args()

    dask.config.set(num_workers=8)  # each temperature chunk is one global hour, 37 levels
    ds = xr.open_zarr(STORE, consolidated=True, storage_options={"token": "anon"})
    out: dict[str, dict] = {"source_store": STORE}

    # Case 3
    pt = nearest(ds, *CBW)
    blh = point_series(ds["boundary_layer_height"], pt, "2019-01-01T00:00", "2019-01-14T05:00")
    night = blh[blh.index.hour <= 5]
    means = night.groupby(night.index.date).mean()
    assert len(means) == 14 and night.groupby(night.index.date).size().eq(6).all()
    day = means.idxmin()  # first occurrence of the minimum -> earlier night on a tie
    out["case3"] = {
        "receptor": "CBW-20",
        "grid_point": pt,
        "release_start_utc": f"{day} 02:00",
        "mean_blh_00_05_m": round(float(means.min()), 2),
        "all_nights_mean_blh_m": {str(k): round(float(v), 2) for k, v in means.items()},
    }

    # Case 4
    pt = nearest(ds, *LEF)
    blh = point_series(ds["boundary_layer_height"], pt, "2019-07-01T18:00", "2019-07-14T22:00")
    aft = blh[(blh.index.hour >= 18) & (blh.index.hour <= 22)]
    means = aft.groupby(aft.index.date).mean()
    assert len(means) == 14 and aft.groupby(aft.index.date).size().eq(5).all()
    day = means.idxmax()
    out["case4"] = {
        "receptor": "LEF",
        "grid_point": pt,
        "release_start_utc": f"{day} 20:00",
        "mean_blh_18_22_m": round(float(means.max()), 2),
        "all_days_mean_blh_m": {str(k): round(float(v), 2) for k, v in means.items()},
    }

    # Case 5
    pt = nearest(ds, *TAC)
    t850 = point_series(
        ds["temperature"].sel(level=850), pt, "2018-12-31T21:00", "2019-01-14T23:00"
    )
    assert len(t850) == 14 * 24 + 3
    fall = (t850.shift(3) - t850).loc["2019-01-01T00:00":]
    assert len(fall) == 336 and fall.notna().all()
    t = fall.idxmax()
    top = fall.sort_values(ascending=False).head(5)
    out["case5"] = {
        "receptor": "TAC",
        "grid_point": pt,
        "release_start_utc": t.strftime("%Y-%m-%d %H:%M"),
        "t850_fall_3h_K": round(float(fall.max()), 3),
        "t850_K_at_t_minus_3h_and_t": [
            round(float(t850[t - pd.Timedelta(hours=3)]), 3),
            round(float(t850[t]), 3),
        ],
        "next_largest_falls_K": {
            k.strftime("%Y-%m-%d %H:%M"): round(float(v), 3) for k, v in top.iloc[1:].items()
        },
    }

    print(json.dumps(out, indent=2))
    if args.json:
        with open(args.json, "w") as f:
            json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
