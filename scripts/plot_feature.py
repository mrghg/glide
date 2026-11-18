#!/usr/bin/env python
"""Render a still "hero" map of a GLIDE multi-site run's combined footprint.

The README feature image. It reuses the machinery in `animate_footprints.py`
(STILT conversion, Web Mercator cell edges) but is styled as
a still: Natural Earth land and coastlines drawn locally with cartopy (no tile
service), no axes, and a log-scaled magma raster whose opacity also rises with
sensitivity so faint far-field tails fade into the background instead of ending
in a hard speckled edge.

Each `--time` renders one PNG: the footprints of every site released at that
hour, summed. Use `{time}` in `--out` to render several candidates in one go:

    .venv/bin/python scripts/plot_feature.py \
        --run-dir ~/data/glide/outputs/icos-model-levels \
        --time 2024-01-10T12 2024-01-20T12 \
        --out outputs/feature-{time}.png

`--mean` instead averages the combined footprint over every release hour in
the run (slow: reads the whole store).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from animate_footprints import (  # noqa: E402
    DEFAULT_AIR_DENSITY_KG_M3,
    DEFAULT_SURFACE_LAYER_DEPTH_M,
    _site_markers,
    compute_frames,
    lat_to_y,
    lon_to_x,
)

# East edge sits on the ICOS run's domain boundary (39.56E) so no raster edge shows.
DEFAULT_EXTENT = (-50.0, 39.5, 32.0, 69.0)  # lon_min, lon_max, lat_min, lat_max

# Basemap palettes: (ocean, land, coastline, ink, muted ink, site fill).
THEMES = {
    "dark": ("#06080d", "#1b2029", "#8a93a6", "#ececec", "#9aa0aa", "#ffffff"),
    "light": ("#f4f5f7", "#e3e5e9", "#5b6270", "#1d1f24", "#5f6570", "#00b8d4"),
}


def mean_footprint(footprint: xr.DataArray, args) -> np.ndarray:
    """Combined surface sensitivity averaged over every release hour in the run."""
    from lpdm.comparison import to_stilt_surface_footprint

    total = to_stilt_surface_footprint(
        footprint.sum("release"),
        surface_layer_depth_m=args.surface_layer_depth,
        air_density_kg_m3=args.air_density,
        integrate_time=True,
    ).compute(scheduler="threads")
    n_hours = len(np.unique(footprint["release_time"].values))
    return (total.values / n_hours).astype("float32")


def _to_uniform_mercator(rgba: np.ndarray, ds: xr.Dataset, rows: int) -> tuple[np.ndarray, list]:
    """Resample an RGBA (lat, lon, 4) raster onto rows evenly spaced in Mercator y.

    The model grid is regular in latitude, so its rows are unevenly spaced in
    Mercator. Blending linearly between neighbouring rows lets the raster go
    through `imshow` (which then smooths in x) without stair-stepping in y.
    """
    y_edges = lat_to_y(ds["latitude_edge"].values)
    x_edges = lon_to_x(ds["longitude_edge"].values)
    y_centres = 0.5 * (y_edges[:-1] + y_edges[1:])
    y_new = np.linspace(y_edges[0], y_edges[-1], rows + 1)
    pos = np.interp(0.5 * (y_new[:-1] + y_new[1:]), y_centres, np.arange(len(y_centres)))
    lo = np.floor(pos).astype(int)
    hi = np.minimum(lo + 1, len(y_centres) - 1)
    w = (pos - lo)[:, None, None]
    return (1.0 - w) * rgba[lo] + w * rgba[hi], [x_edges[0], x_edges[-1], y_new[0], y_new[-1]]


def render(args, field: np.ndarray, ds: xr.Dataset, out: Path, caption: str) -> None:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    ocean, land, coast, ink, muted, site_fill = THEMES[args.theme]
    merc = ccrs.Mercator.GOOGLE

    lon_min, lon_max, lat_min, lat_max = args.extent
    view = (lon_to_x(lon_min), lon_to_x(lon_max), lat_to_y(lat_min), lat_to_y(lat_max))
    aspect = (view[3] - view[2]) / (view[1] - view[0])

    positive = field[field > 0]
    vmax = args.vmax or float(np.percentile(positive, 99.95))
    vmin = vmax / 10**args.decades
    norm = LogNorm(vmin=vmin, vmax=vmax, clip=True)
    cmap = plt.get_cmap(args.cmap)

    # RGBA raster: colour by log value, and fade opacity in over the bottom of the
    # scale so the tail dissolves into the basemap. Empty cells are fully clear.
    level = np.ma.filled(norm(np.ma.masked_less_equal(field, 0.0)), 0.0)
    rgba = cmap(level)
    alpha = np.clip(level / args.fade, 0.0, 1.0) ** args.fade_gamma
    rgba[..., 3] = np.where(field > 0, alpha * args.alpha, 0.0)
    rgba, extent = _to_uniform_mercator(rgba, ds, rows=4 * field.shape[0])

    fig_w = args.width / args.dpi
    fig = plt.figure(figsize=(fig_w, fig_w * aspect), dpi=args.dpi, facecolor=ocean)
    ax = fig.add_axes([0, 0, 1, 1], projection=merc)
    ax.set_axis_off()
    ax.set_extent([view[0], view[1], view[2], view[3]], crs=merc)
    ax.set_facecolor(ocean)

    ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor=land, edgecolor="none", zorder=0)
    ax.imshow(
        rgba,
        extent=extent,
        origin="lower",
        transform=merc,
        interpolation="bilinear",
        interpolation_stage="rgba",
        zorder=1,
    )
    # Coastlines on top of the raster, faintly, so the geography reads through it.
    ax.add_feature(
        cfeature.COASTLINE.with_scale("50m"),
        edgecolor=coast,
        linewidth=0.5,
        alpha=0.55,
        zorder=2,
    )

    site_x, site_y = _site_markers(ds)
    ax.scatter(
        site_x,
        site_y,
        s=args.marker_size,
        c=site_fill,
        edgecolors="#111111",
        linewidths=0.8,
        transform=merc,
        zorder=3,
    )

    # Slim horizontal colour bar with caption, bottom-left, on a translucent plate
    # so the labels stay legible when the plume runs underneath them.
    from matplotlib.patches import FancyBboxPatch

    fig.patches.append(
        FancyBboxPatch(
            (0.012, 0.018),
            0.39,
            0.135,
            boxstyle="round,pad=0,rounding_size=0.008",
            transform=fig.transFigure,
            facecolor=ocean,
            edgecolor="none",
            alpha=0.72,
            zorder=1,
        )
    )
    cax = fig.add_axes([0.025, 0.075, 0.22, 0.016], zorder=2)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(labelsize=8, colors=muted, length=2)
    cbar.ax.minorticks_off()
    cbar.set_label("surface sensitivity  (mol mol⁻¹) / (mol m⁻² s⁻¹)", fontsize=8, color=muted)
    fig.text(0.025, 0.11, caption, fontsize=11, color=ink, ha="left", va="bottom")
    fig.text(0.992, 0.01, "Coastlines: Natural Earth", fontsize=6.5, color=muted, ha="right")

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=args.dpi, facecolor=ocean)
    plt.close(fig)
    print(f"wrote {out}  (colour scale {vmin:.3g} .. {vmax:.3g})")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Render a still map of a GLIDE run's combined footprint.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--run-dir", type=Path, default=Path("~/data/glide/outputs/icos-model-levels"))
    p.add_argument(
        "--time",
        nargs="+",
        default=None,
        help="release hour(s), e.g. 2024-01-20T12 (default: middle of the run)",
    )
    p.add_argument("--mean", action="store_true", help="average over every release hour")
    p.add_argument(
        "--out",
        type=Path,
        default=Path("docs/img/glide_feature.png"),
        help="output PNG; '{time}' is replaced by the release hour",
    )
    p.add_argument("--width", type=int, default=2400, help="output width in pixels")
    p.add_argument("--dpi", type=int, default=200)
    p.add_argument("--theme", choices=sorted(THEMES), default="dark")
    p.add_argument("--cmap", default="magma")
    p.add_argument("--decades", type=float, default=5.0, help="log decades below vmax")
    p.add_argument("--vmax", type=float, default=None, help="colour ceiling (default 99.95th pct)")
    p.add_argument(
        "--fade", type=float, default=0.6, help="fraction of the scale over which opacity ramps"
    )
    p.add_argument("--fade-gamma", type=float, default=1.6, help="shape of the opacity ramp")
    p.add_argument("--alpha", type=float, default=0.92, help="maximum raster opacity")
    p.add_argument("--marker-size", type=float, default=22.0)
    p.add_argument(
        "--extent",
        type=float,
        nargs=4,
        default=list(DEFAULT_EXTENT),
        metavar=("LON_MIN", "LON_MAX", "LAT_MIN", "LAT_MAX"),
    )
    p.add_argument("--surface-layer-depth", type=float, default=DEFAULT_SURFACE_LAYER_DEPTH_M)
    p.add_argument("--air-density", type=float, default=DEFAULT_AIR_DENSITY_KG_M3)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.run_dir = args.run_dir.expanduser()
    args.extent = tuple(args.extent)

    ds = xr.open_zarr(args.run_dir / "footprints.zarr")
    footprint = ds["footprint"]
    release_times = ds["release_time"].values
    unique_times = np.unique(release_times)
    n_sites = len(set(ds["site"].values.tolist()))

    if args.mean:
        field = mean_footprint(footprint, args)
        start = np.datetime_as_string(unique_times[0], unit="D")
        end = np.datetime_as_string(unique_times[-1], unit="D")
        caption = f"GLIDE · {n_sites} sites · mean 5-day footprint, {start} to {end}"
        render(args, field, ds, Path(str(args.out).replace("{time}", "mean")), caption)
        return 0

    if args.time:
        wanted = np.array([np.datetime64(t, "ns") for t in args.time])
    else:
        wanted = unique_times[[len(unique_times) // 2]]
    missing = [t for t in wanted if t not in set(unique_times)]
    if missing:
        raise SystemExit(
            f"no releases at {missing}; run covers {unique_times[0]}..{unique_times[-1]}"
        )

    frames = compute_frames(
        footprint,
        release_times,
        wanted,
        surface_layer_depth_m=args.surface_layer_depth,
        air_density_kg_m3=args.air_density,
        batch=4,
    )
    for when, field in zip(wanted, frames, strict=True):
        stamp = np.datetime_as_string(when, unit="h")
        caption = f"GLIDE · {n_sites} sites · 5-day footprint · {stamp.replace('T', ' ')}:00 UTC"
        render(args, field, ds, Path(str(args.out).replace("{time}", stamp)), caption)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
