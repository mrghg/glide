"""Task A01: compare the model-level and pressure-level comparison runs.

    .venv/bin/python scripts/validation/a01/compare.py outputs/a01-model outputs/a01-pressure23 outputs/a01-compare

Prints per-release integrated sensitivity (0-40 m layer and all layers), the
near-surface share, and writes a figure: the mean 0-40 m footprint of each run
and the log ratio, plus per-release totals.
"""

import json
import sys
from pathlib import Path

import numpy as np
import xarray as xr


def load(run: Path) -> xr.DataArray:
    return xr.open_zarr(run / "footprints.zarr")["footprint"].sum("time_ago").load()


def main(model_dir: str, pressure_dir: str, out_dir: str) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    m, p = load(Path(model_dir)), load(Path(pressure_dir))
    surf_m, surf_p = m.isel(z_bin=0), p.isel(z_bin=0)
    tot_m = surf_m.sum(("latitude", "longitude")).values
    tot_p = surf_p.sum(("latitude", "longitude")).values
    all_m = m.sum(("z_bin", "latitude", "longitude")).values
    all_p = p.sum(("z_bin", "latitude", "longitude")).values
    ratio = tot_m / tot_p
    sites = m["site"].values if "site" in m.coords else np.array([""] * m.sizes["release"])
    summary = {"releases": int(m.sizes["release"])}
    for s in sorted(set(sites)):
        sel = sites == s
        summary[str(s)] = {
            "surface_ratio_median": float(np.median(ratio[sel])),
            "surface_ratio_p10_p90": [float(x) for x in np.percentile(ratio[sel], [10, 90])],
            "surface_share_model": float(np.median(tot_m[sel] / all_m[sel])),
            "surface_share_pressure": float(np.median(tot_p[sel] / all_p[sel])),
            "total_ratio_median": float(np.median(all_m[sel] / all_p[sel])),
        }
    mean_m, mean_p = surf_m.mean("release"), surf_p.mean("release")
    both = (mean_m > 0) & (mean_p > 0)
    lr = np.log10(mean_m.where(both) / mean_p.where(both))
    # Log ratio near the sites (within 3 degrees) and far from them.
    lat, lon = np.meshgrid(m.latitude.values, m.longitude.values, indexing="ij")
    near = np.zeros_like(lat, dtype=bool)
    for la, lo in zip(m["release_lat"].values, m["release_lon"].values, strict=True):
        near |= np.hypot(lat - la, (lon - lo) * np.cos(np.radians(la))) < 3.0
    v = lr.values
    summary["log10_ratio_mean_footprint"] = {
        "near_sites_median": float(np.nanmedian(v[near])),
        "far_median": float(np.nanmedian(v[~near])),
        "weighted_by_pressure_run": float(
            np.nansum(v * mean_p.values) / np.nansum(mean_p.values * np.isfinite(v))
        ),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    box = dict(latitude=slice(40, 65), longitude=slice(-30, 20))
    fig, ax = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    vmax = float(mean_p.sel(**box).max())
    for a, field, title in (
        (ax[0, 0], mean_m, "model levels, 40-level grid"),
        (ax[0, 1], mean_p, "pressure levels, 23-level grid"),
    ):
        f = field.sel(**box)
        im = a.pcolormesh(
            f.longitude, f.latitude, f.where(f > 0), norm=LogNorm(vmax * 1e-4, vmax), shading="auto"
        )
        a.set_title(f"mean 0-40 m footprint (s), {title}")
        fig.colorbar(im, ax=a)
    f = lr.sel(**box)
    im = ax[1, 0].pcolormesh(
        f.longitude, f.latitude, f, cmap="RdBu_r", vmin=-0.5, vmax=0.5, shading="auto"
    )
    ax[1, 0].set_title("log10(model / pressure), mean 0-40 m footprint")
    fig.colorbar(im, ax=ax[1, 0])
    for s in sorted(set(sites)):
        sel = sites == s
        ax[1, 1].plot(np.arange(sel.sum()), ratio[sel], label=str(s))
    ax[1, 1].axhline(1.0, color="k", lw=0.8)
    ax[1, 1].set_xlabel("release (hour of 1 Jan 2024)")
    ax[1, 1].set_ylabel("0-40 m sensitivity, model / pressure")
    ax[1, 1].legend()
    fig.savefig(out / "a01_model_vs_pressure.png", dpi=110)


if __name__ == "__main__":
    main(*sys.argv[1:4])
