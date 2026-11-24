"""Task A02: how per-column convection moves footprints on the A01 comparison config.

    .venv/bin/python scripts/validation/a02/compare.py \
        outputs/a02-after outputs/a02-before outputs/a02-noise outputs/a02-compare

`a02-after` is the per-column scheme, `a02-before` the bbox-mean scheme, and
`a02-noise` the bbox-mean scheme with a different seed: the after/before
differences are read against before/noise, the particle noise of the config
(convection draws random numbers, so every particle's path changes even where
no column fires under it). Prints per-site medians of the per-release 0-40 m and
all-layer totals, and writes summary.json and a figure.
"""

import json
import sys
from pathlib import Path

import numpy as np
import xarray as xr


def load(run: Path) -> xr.DataArray:
    return xr.open_zarr(run / "footprints.zarr")["footprint"].sum("time_ago").load()


def ratios(a: xr.DataArray, b: xr.DataArray) -> dict:
    surf_a = a.isel(z_bin=0).sum(("latitude", "longitude")).values
    surf_b = b.isel(z_bin=0).sum(("latitude", "longitude")).values
    all_a = a.sum(("z_bin", "latitude", "longitude")).values
    all_b = b.sum(("z_bin", "latitude", "longitude")).values
    free_a = a.isel(z_bin=slice(2, None)).sum(("z_bin", "latitude", "longitude")).values
    free_b = b.isel(z_bin=slice(2, None)).sum(("z_bin", "latitude", "longitude")).values
    sites = a["site"].values if "site" in a.coords else np.array([""] * a.sizes["release"])
    out = {}
    for s in sorted(set(sites)):
        sel = sites == s
        r = surf_a[sel] / surf_b[sel]
        out[str(s)] = {
            "surface_ratio_median": float(np.median(r)),
            "surface_ratio_p10_p90": [float(x) for x in np.percentile(r, [10, 90])],
            "all_layer_ratio_median": float(np.median(all_a[sel] / all_b[sel])),
            "above_1km_ratio_median": float(np.median(free_a[sel] / free_b[sel])),
        }
    # Map-level: log ratio of the release-mean 0-40 m footprint where both are set.
    ma = a.isel(z_bin=0).mean("release").values
    mb = b.isel(z_bin=0).mean("release").values
    both = (ma > 0) & (mb > 0)
    lr = np.log10(ma[both] / mb[both])
    out["map_log10_ratio_abs_median"] = float(np.median(np.abs(lr)))
    out["map_log10_ratio_p5_p95"] = [float(x) for x in np.percentile(lr, [5, 95])]
    return out


def main(after: str, before: str, noise: str, out_dir: str) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    a, b = load(Path(after)), load(Path(before))
    summary = {"after_vs_before": ratios(a, b)}
    if Path(noise).exists():
        summary["noise_vs_before"] = ratios(load(Path(noise)), b)
    print(json.dumps(summary, indent=1))
    (out / "summary.json").write_text(json.dumps(summary, indent=1))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ma = a.isel(z_bin=0).mean("release")
    mb = b.isel(z_bin=0).mean("release")
    with np.errstate(divide="ignore", invalid="ignore"):
        lr = np.log10(ma / mb).where((ma > 0) & (mb > 0))
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    np.log10(mb.where(mb > 0)).plot(ax=axes[0], cmap="viridis")
    axes[0].set_title("bbox-mean scheme: mean 0-40 m footprint (log10)")
    lr.plot(ax=axes[1], cmap="RdBu_r", vmin=-1, vmax=1)
    axes[1].set_title("log10(per-column / bbox-mean), 0-40 m")
    fig.savefig(out / "a02_per_column_vs_bbox_mean.png", dpi=110)


if __name__ == "__main__":
    main(*sys.argv[1:5])
