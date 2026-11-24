"""Task A04: the size of the error from dividing by density after counting.

    .venv/bin/python scripts/validation/a04/compare.py [outputs-dir] [case=run ...]

For standard cases 3 (CBW-20) and 4 (LEF), compares the 0-40 m footprint
weighted by the dry-air density while counting (`density_weighting: dry_air`,
converted with M_air / h) against the same run in plain seconds
(`density_weighting: none`) converted afterwards with
`to_stilt_surface_footprint` and a mean surface density per cell from
`surface_air_density_from_met`, the pre-A04 recipe. Both runs use the same seed
and store, so their particles are identical and the difference is the
weighting alone, with no particle noise. GLIDE against GLIDE only.

Two "old" conversions are reported:
- old: the helper's density sp / (R_d T_lowest level) averaged over the 240 h
  (moist, the pre-A04 recipe; the full change a user sees);
- old_dry: the same with the dry-air density (sp - e) / (R_d T), which leaves
  the covariance of residence and density (and the particle-height and
  time-matching differences), without the humidity term.
Writes a summary JSON and a ratio map per case next to the outputs.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import xarray as xr

from lpdm.air_density import dry_air_density_kg_m3
from lpdm.comparison import (
    DRY_AIR_M_KG_MOL,
    surface_air_density_from_met,
    to_stilt_surface_footprint,
)

CASES = {
    "case3-cbw20": {
        "store": "data/era5-arco/EUR_20181222-20190115_ERA5-ARCO.zarr",
        "receptor": (4.9264, 51.9703),
    },
    "case4-lef": {
        "store": "data/era5-arco/NAM_20190621-20190715_ERA5-ARCO.zarr",
        "receptor": (-90.2732, 45.9451),
    },
}
H = 40.0


def _raw(out: Path, case: str, weighting: str) -> xr.DataArray:
    ds = xr.open_zarr(out / f"a04-{case}-{weighting}" / "footprints.zarr")
    fp = ds["footprint"].isel(release=0).load()
    assert fp.attrs.get("density_weighting") == weighting, fp.attrs
    return fp


def _period_densities(store: str, fp: xr.DataArray) -> tuple[xr.DataArray, xr.DataArray]:
    """Mean surface density over the footprint's 240 h, moist (the helper) and dry."""

    rt = np.datetime64(str(fp["release_time"].values)[:19])
    end = rt + np.timedelta64(1, "h")
    start = end - np.timedelta64(int(fp["time_ago_end_hours"].values.max()), "h")
    met = xr.open_zarr(store).sel(time=slice(start, end))
    lowest = int(met["hybrid"].argmax())
    met = met[["surface_pressure", "temperature", "specific_humidity"]].isel(hybrid=[lowest])
    met = met.rename({"hybrid": "level"})
    # The store may hold longitudes in 0-360; select on its convention, then
    # label with the footprint's.
    lon = fp["longitude"].values
    if float(met["longitude"].max()) > 180.0:
        lon = lon % 360.0
    near = dict(latitude=fp["latitude"].values, longitude=lon, method="nearest")

    def on_grid(field: xr.DataArray) -> xr.DataArray:
        out = field.sel(**near).load()
        return out.assign_coords(latitude=fp["latitude"].values, longitude=fp["longitude"].values)

    rho_moist = on_grid(surface_air_density_from_met(met))
    t = met["temperature"].isel(level=0)
    q = met["specific_humidity"].isel(level=0)
    rho_dry = on_grid(dry_air_density_kg_m3(met["surface_pressure"], t, q).mean("time"))
    return rho_moist, rho_dry


def _compare(new: np.ndarray, old: np.ndarray, lon, lat, receptor) -> dict:
    both = (new > 0) & (old > 0)
    r = new[both] / old[both]
    w = new[both] / new[both].sum()
    order = np.argsort(r)
    cw = np.cumsum(w[order])

    def wq(p):
        return float(r[order][np.searchsorted(cw, p)])

    lon2, lat2 = np.meshgrid(lon, lat)
    d = np.hypot((lon2 - receptor[0]) * np.cos(np.deg2rad(receptor[1])), lat2 - receptor[1])
    out = {
        "total_new_m2_s_mol": float(new.sum()),
        "total_old_m2_s_mol": float(old.sum()),
        "total_ratio_new_over_old": float(new.sum() / old.sum()),
        "sensitivity_weighted_ratio_p10_p50_p90": [wq(0.1), wq(0.5), wq(0.9)],
        "weighted_mean_abs_log10_ratio": float(np.sum(w * np.abs(np.log10(r)))),
    }
    for name, m in (("within_1deg", d <= 1.0), ("beyond_1deg", d > 1.0)):
        out[f"ratio_{name}"] = float(new[m].sum() / old[m].sum()) if old[m].sum() > 0 else None
        out[f"share_of_new_{name}"] = float(new[m].sum() / new.sum())
    return out


def _map(path: Path, ratio: xr.DataArray, case: str, receptor) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5.5))
    pct = 100.0 * (ratio - 1.0)
    im = ax.pcolormesh(
        ratio["longitude"], ratio["latitude"], pct, cmap="RdBu_r", vmin=-6, vmax=6, shading="auto"
    )
    ax.plot(*receptor, "k*", ms=10)
    ax.set_title(f"A04 {case}: weighted while counting / divided afterwards, 0-40 m (%)")
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    fig.colorbar(im, ax=ax, label="change (%)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main(out: Path, runs: dict[str, str]) -> None:
    summary = {}
    for case, info in CASES.items():
        run = runs.get(case, case)
        weighted = _raw(out, run, "dry_air")
        plain = _raw(out, run, "none")
        rho_moist, rho_dry = _period_densities(info["store"], plain)
        new = to_stilt_surface_footprint(weighted, surface_layer_depth_m=H)
        old = to_stilt_surface_footprint(
            plain, surface_layer_depth_m=H, air_density_kg_m3=rho_moist
        )
        old_dry = to_stilt_surface_footprint(
            plain, surface_layer_depth_m=H, air_density_kg_m3=rho_dry
        )
        lon, lat = new["longitude"].values, new["latitude"].values
        rec = info["receptor"]
        if float(lon.max()) > 180.0:  # a run configured in 0-360 longitudes
            rec = (rec[0] % 360.0, rec[1])
        summary[case] = {
            "new_vs_old": _compare(new.values, old.values, lon, lat, rec),
            "new_vs_old_dry": _compare(new.values, old_dry.values, lon, lat, rec),
            "old_dry_vs_old (humidity alone)": float(old_dry.values.sum() / old.values.sum()),
            "plain_seconds_total_0_40m": float(plain.isel(z_bin=0).sum()),
            "m_air_kg_mol": DRY_AIR_M_KG_MOL,
        }
        ratio = (new / old).where((new > 0) & (old > 0))
        _map(out / f"a04-{case}-ratio.png", ratio, case, rec)
        print(case, json.dumps(summary[case], indent=1))
    (out / "a04-summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    # Optional CASE=RUN arguments name a differently tagged run for a case, e.g.
    # case4-lef=case4-lef360 for a NAM run configured in 0-360 longitudes.
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("outputs")
    main(out_dir, dict(a.split("=", 1) for a in sys.argv[2:]))
