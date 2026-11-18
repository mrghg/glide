"""Shared map machinery for GLIDE footprint figures.

Used by `plot_feature.py` (the README still) and `animate_footprints.py` (the
MP4), so both have the same look:

- footprints converted to STILT-style surface sensitivity
  (`lpdm.comparison.to_stilt_surface_footprint`), summed over the sites
  released at each hour;
- Web Mercator, with Natural Earth land and coastlines drawn by cartopy, so no
  tile service is needed;
- a log-scaled raster whose opacity rises with sensitivity, so the faint far
  field fades into the background instead of ending in a speckled edge;
- no axes, and a caption and slim colour bar on a translucent plate, bottom left.

The figure is always `FIG_WIDTH_IN` wide and the pixel width is set through
the dpi, so text and the plate are the same size relative to the map at any
output resolution.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

import dask
import numpy as np
import xarray as xr

EARTH_RADIUS_M = 6378137.0
MERCATOR_LAT_LIMIT = 85.05112878

# Matches the FLEXPART/NAME surface layer and the run's bottom z-bin (both 0-40 m),
# so the STILT conversion is exact rather than overlap-approximated.
DEFAULT_SURFACE_LAYER_DEPTH_M = 40.0
DEFAULT_AIR_DENSITY_KG_M3 = 1.2

# The Europe/Atlantic region the ICOS network is sensitive to. The east edge sits
# on the ICOS runs' domain boundary (39.56E) so no raster edge shows.
DEFAULT_EXTENT = (-50.0, 39.5, 32.0, 69.0)  # lon_min, lon_max, lat_min, lat_max

FIG_WIDTH_IN = 12.0

# Palettes: (ocean, land, coastline, ink, muted ink, site fill).
THEMES = {
    "dark": ("#06080d", "#1b2029", "#8a93a6", "#ececec", "#9aa0aa", "#ffffff"),
    "light": ("#f4f5f7", "#e3e5e9", "#5b6270", "#1d1f24", "#5f6570", "#00b8d4"),
}

COLORBAR_LABEL = "surface sensitivity  (mol mol⁻¹) / (mol m⁻² s⁻¹)"


# --------------------------------------------------------------------------- #
# Web Mercator
# --------------------------------------------------------------------------- #


def lon_to_x(lon: np.ndarray | float) -> np.ndarray:
    return np.radians(np.asarray(lon, dtype="float64")) * EARTH_RADIUS_M


def lat_to_y(lat: np.ndarray | float) -> np.ndarray:
    clipped = np.clip(np.asarray(lat, dtype="float64"), -MERCATOR_LAT_LIMIT, MERCATOR_LAT_LIMIT)
    return EARTH_RADIUS_M * np.log(np.tan(np.pi / 4.0 + np.radians(clipped) / 2.0))


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #


def compute_frames(
    footprint: xr.DataArray,
    release_times: np.ndarray,
    frame_times: np.ndarray,
    *,
    surface_layer_depth_m: float,
    air_density_kg_m3: float,
    batch: int,
) -> np.ndarray:
    """Combined surface sensitivity (lat, lon) for each time in `frame_times`."""
    from lpdm.comparison import to_stilt_surface_footprint

    frames: list[np.ndarray] = []
    for start in range(0, len(frame_times), batch):
        pending = []
        for stamp in frame_times[start : start + batch]:
            # release_times is the materialised coord; build integer indices rather
            # than masking the dask-backed store coord.
            at_t = footprint.isel(release=np.nonzero(release_times == stamp)[0])
            combined = at_t.sum("release")  # (time_ago, z_bin, latitude, longitude)
            pending.append(
                to_stilt_surface_footprint(
                    combined,
                    surface_layer_depth_m=surface_layer_depth_m,
                    air_density_kg_m3=air_density_kg_m3,
                    integrate_time=True,
                ).data
            )
        frames.extend(dask.compute(*pending, scheduler="threads"))
        done = min(start + batch, len(frame_times))
        print(f"\r  frames {done}/{len(frame_times)}", end="", flush=True)
    print()
    return np.stack(frames).astype("float32")


def site_markers(ds: xr.Dataset) -> tuple[np.ndarray, np.ndarray]:
    """One (x, y) per unique site, in Mercator."""
    sites = ds["site"].values
    lons = ds["release_lon"].values
    lats = ds["release_lat"].values
    _, first = np.unique(sites, return_index=True)
    return lon_to_x(lons[first]), lat_to_y(lats[first])


def n_sites(ds: xr.Dataset) -> int:
    return len(set(ds["site"].values.tolist()))


def hour_stamp(when: np.datetime64) -> str:
    return np.datetime_as_string(when, unit="h").replace("T", " ") + ":00 UTC"


# --------------------------------------------------------------------------- #
# Styling
# --------------------------------------------------------------------------- #


def add_style_args(p: argparse.ArgumentParser, *, width: int) -> None:
    """CLI options shared by every footprint map."""
    p.add_argument("--width", type=int, default=width, help="output width in pixels")
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
        help="map view; default is the Europe/Atlantic region",
    )
    p.add_argument("--surface-layer-depth", type=float, default=DEFAULT_SURFACE_LAYER_DEPTH_M)
    p.add_argument("--air-density", type=float, default=DEFAULT_AIR_DENSITY_KG_M3)


def colour_ceiling(fields: np.ndarray, vmax: float | None) -> float:
    """`vmax` if given, else the 99.95th percentile of the positive values."""
    if vmax:
        return vmax
    positive = fields[fields > 0]
    if positive.size == 0:
        raise SystemExit("every field is empty — nothing to plot")
    return float(np.percentile(positive, 99.95))


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


@dataclass
class FootprintMap:
    """A styled map figure whose raster and caption can be swapped per frame."""

    fig: object
    image: object
    caption: object
    norm: object
    cmap: object
    ds: xr.Dataset
    args: argparse.Namespace

    @property
    def size_px(self) -> tuple[int, int]:
        w, h = self.fig.get_size_inches() * self.fig.dpi
        return int(round(w)), int(round(h))

    def rgba(self, field: np.ndarray) -> tuple[np.ndarray, list]:
        """Colour by log value; fade opacity in over the bottom of the scale."""
        args = self.args
        level = np.ma.filled(self.norm(np.ma.masked_less_equal(field, 0.0)), 0.0)
        rgba = self.cmap(level)
        alpha = np.clip(level / args.fade, 0.0, 1.0) ** args.fade_gamma
        rgba[..., 3] = np.where(field > 0, alpha * args.alpha, 0.0)
        return _to_uniform_mercator(rgba, self.ds, rows=4 * field.shape[0])

    def show(self, field: np.ndarray, caption: str) -> None:
        rgba, _ = self.rgba(field)
        self.image.set_data(rgba)
        self.caption.set_text(caption)


def build_map(args: argparse.Namespace, ds: xr.Dataset, vmax: float) -> FootprintMap:
    """Draw everything static (land, coastlines, sites, colour bar, plate).

    The raster starts empty; call `FootprintMap.show` to fill it. Pixel size is
    `args.width` wide, with the height rounded to an even number for H.264.
    """
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from matplotlib.patches import FancyBboxPatch

    ocean, land, coast, ink, muted, site_fill = THEMES[args.theme]
    merc = ccrs.Mercator.GOOGLE

    lon_min, lon_max, lat_min, lat_max = args.extent
    view = (lon_to_x(lon_min), lon_to_x(lon_max), lat_to_y(lat_min), lat_to_y(lat_max))
    aspect = (view[3] - view[2]) / (view[1] - view[0])

    width_px = args.width - (args.width % 2)
    dpi = width_px / FIG_WIDTH_IN
    height_px = int(round(width_px * aspect))
    height_px += height_px % 2
    fig = plt.figure(figsize=(FIG_WIDTH_IN, height_px / dpi), dpi=dpi, facecolor=ocean)

    ax = fig.add_axes([0, 0, 1, 1], projection=merc)
    ax.set_axis_off()
    ax.set_extent(list(view), crs=merc)
    ax.set_facecolor(ocean)
    ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor=land, edgecolor="none", zorder=0)

    norm = LogNorm(vmin=vmax / 10**args.decades, vmax=vmax, clip=True)
    cmap = plt.get_cmap(args.cmap)
    blank = np.zeros((4 * ds.sizes["latitude"], ds.sizes["longitude"], 4))
    _, extent = _to_uniform_mercator(blank[: ds.sizes["latitude"]], ds, rows=blank.shape[0])
    image = ax.imshow(
        blank,
        extent=extent,
        origin="lower",
        transform=merc,
        interpolation="bilinear",
        interpolation_stage="rgba",
        zorder=1,
    )
    # Coastlines on top of the raster, faintly, so the geography reads through it.
    ax.add_feature(
        cfeature.COASTLINE.with_scale("50m"), edgecolor=coast, linewidth=0.5, alpha=0.55, zorder=2
    )
    site_x, site_y = site_markers(ds)
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

    # Caption and colour bar on a translucent plate, so they stay legible when
    # the plume runs underneath them.
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
    cbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal"
    )
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(labelsize=8, colors=muted, length=2)
    cbar.ax.minorticks_off()
    cbar.set_label(COLORBAR_LABEL, fontsize=8, color=muted)
    caption = fig.text(0.025, 0.11, "", fontsize=11, color=ink, ha="left", va="bottom")
    fig.text(0.992, 0.01, "Coastlines: Natural Earth", fontsize=6.5, color=muted, ha="right")

    return FootprintMap(fig, image, caption, norm, cmap, ds, args)


def full_domain_extent(ds: xr.Dataset) -> tuple[float, float, float, float]:
    return (
        float(ds["longitude_edge"].min()),
        float(ds["longitude_edge"].max()),
        max(float(ds["latitude_edge"].min()), -MERCATOR_LAT_LIMIT),
        min(float(ds["latitude_edge"].max()), math.floor(MERCATOR_LAT_LIMIT)),
    )
