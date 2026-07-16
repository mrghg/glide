#!/usr/bin/env python
"""Animate a GLIDE multi-site run's combined footprints as an MP4.

For each release hour the footprints of every site released at that instant are
summed and converted to STILT-style surface sensitivity
(`lpdm.comparison.to_stilt_surface_footprint`) — the same physics bridge the
validation notebooks use. Frames are drawn in the style of
`notebooks/multisite_validation.ipynb`: a magma raster on a log colour scale over
a CartoLight basemap, with the release sites as cyan markers.

Data are plotted in Web Mercator so the raster lines up with the tile basemap
exactly. Only the *cell edges* are transformed (a 1-D operation on latitude), so
no per-frame reprojection of the raster is needed.

Reading the store dominates runtime, so computed frames are cached to a `.npz`
alongside the output; re-running to tweak styling reuses the cache.

Typical use:

    .venv/bin/python scripts/animate_footprints.py \
        --run-dir ~/data/glide/outputs/icos-202401 \
        --out outputs/icos-202401-footprints.mp4

    # quick look: every 6th hour
    .venv/bin/python scripts/animate_footprints.py --stride 6 --fps 8
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import io
import math
import sys
import urllib.request
from pathlib import Path

import dask
import numpy as np
import xarray as xr

# Web Mercator (EPSG:3857).
EARTH_RADIUS_M = 6378137.0
WORLD_EDGE_M = math.pi * EARTH_RADIUS_M
MERCATOR_LAT_LIMIT = 85.05112878

TILE_URL = "https://basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png"
TILE_PX = 256
USER_AGENT = "glide-animate-footprints/1.0 (research; contact via project repo)"

# Matches the FLEXPART/NAME surface layer and the run's bottom z-bin (both 0-40 m),
# so the STILT conversion is exact rather than overlap-approximated.
DEFAULT_SURFACE_LAYER_DEPTH_M = 40.0
DEFAULT_AIR_DENSITY_KG_M3 = 1.2

# Default view: the Europe/Atlantic region the ICOS network is sensitive to. The
# stored grid is much wider (out to ~98degW) and mostly empty at these scales.
DEFAULT_EXTENT = (-60.0, 40.0, 25.0, 75.0)  # lon_min, lon_max, lat_min, lat_max


# --------------------------------------------------------------------------- #
# Web Mercator helpers
# --------------------------------------------------------------------------- #


def lon_to_x(lon: np.ndarray | float) -> np.ndarray:
    return np.radians(np.asarray(lon, dtype="float64")) * EARTH_RADIUS_M


def lat_to_y(lat: np.ndarray | float) -> np.ndarray:
    clipped = np.clip(np.asarray(lat, dtype="float64"), -MERCATOR_LAT_LIMIT, MERCATOR_LAT_LIMIT)
    return EARTH_RADIUS_M * np.log(np.tan(np.pi / 4.0 + np.radians(clipped) / 2.0))


def _lonlat_to_tile(lon: float, lat: float, zoom: int) -> tuple[float, float]:
    n = 2**zoom
    x = (lon + 180.0) / 360.0 * n
    lat_r = math.radians(min(max(lat, -MERCATOR_LAT_LIMIT), MERCATOR_LAT_LIMIT))
    y = (1.0 - math.log(math.tan(lat_r) + 1.0 / math.cos(lat_r)) / math.pi) / 2.0 * n
    return x, y


def auto_zoom(lon_min: float, lon_max: float, target_px: int) -> int:
    """Smallest tile zoom whose stitched width is at least `target_px`."""
    span = max(lon_max - lon_min, 1e-6)
    zoom = math.ceil(math.log2(target_px * 360.0 / (TILE_PX * span)))
    return int(min(max(zoom, 2), 8))


# --------------------------------------------------------------------------- #
# Basemap
# --------------------------------------------------------------------------- #


def _fetch_tile(zoom: int, x: int, y: int, cache_dir: Path):
    from PIL import Image

    cached = cache_dir / str(zoom) / str(x) / f"{y}.png"
    if cached.exists():
        return Image.open(cached).convert("RGB")

    url = TILE_URL.format(z=zoom, x=x, y=y)
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    payload = urllib.request.urlopen(request, timeout=30).read()
    cached.parent.mkdir(parents=True, exist_ok=True)
    cached.write_bytes(payload)
    return Image.open(io.BytesIO(payload)).convert("RGB")


def fetch_basemap(
    extent: tuple[float, float, float, float],
    zoom: int,
    cache_dir: Path,
    workers: int = 8,
) -> tuple[np.ndarray, list[float]]:
    """Stitch CartoLight tiles covering `extent`; return (RGB array, Mercator extent).

    Missing tiles are left as the CartoLight background grey rather than failing the
    run — an incomplete basemap is better than no animation.
    """
    from PIL import Image

    lon_min, lon_max, lat_min, lat_max = extent
    n = 2**zoom
    x0f, y0f = _lonlat_to_tile(lon_min, lat_max, zoom)
    x1f, y1f = _lonlat_to_tile(lon_max, lat_min, zoom)
    x0, x1 = max(0, int(math.floor(x0f))), min(n - 1, int(math.floor(x1f)))
    y0, y1 = max(0, int(math.floor(y0f))), min(n - 1, int(math.floor(y1f)))

    cols, rows = x1 - x0 + 1, y1 - y0 + 1
    canvas = Image.new("RGB", (cols * TILE_PX, rows * TILE_PX), (242, 242, 240))

    jobs = [(x, y) for y in range(y0, y1 + 1) for x in range(x0, x1 + 1)]
    failures = 0
    with cf.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_fetch_tile, zoom, x, y, cache_dir): (x, y) for x, y in jobs}
        for future in cf.as_completed(futures):
            x, y = futures[future]
            try:
                canvas.paste(future.result(), ((x - x0) * TILE_PX, (y - y0) * TILE_PX))
            except Exception:
                failures += 1
    if failures:
        print(f"  warning: {failures}/{len(jobs)} basemap tiles failed to load", file=sys.stderr)

    tile_span = 2.0 * WORLD_EDGE_M / n
    mercator_extent = [
        -WORLD_EDGE_M + x0 * tile_span,
        -WORLD_EDGE_M + (x1 + 1) * tile_span,
        WORLD_EDGE_M - (y1 + 1) * tile_span,
        WORLD_EDGE_M - y0 * tile_span,
    ]
    return np.asarray(canvas), mercator_extent


# --------------------------------------------------------------------------- #
# Frame computation
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


def load_or_compute_frames(args, footprint, release_times, frame_times) -> np.ndarray:
    cache = args.cache
    if cache and cache.exists() and not args.refresh_cache:
        with np.load(cache) as data:
            frames, cached_times = data["frames"], data["times"]
        if len(cached_times) == len(frame_times) and (cached_times == frame_times).all():
            print(f"Reusing cached frames: {cache}")
            return frames
        print(f"Cache {cache} does not match requested times — recomputing")

    print(f"Computing {len(frame_times)} frames (reads the footprint store; this is the slow part)")
    frames = compute_frames(
        footprint,
        release_times,
        frame_times,
        surface_layer_depth_m=args.surface_layer_depth,
        air_density_kg_m3=args.air_density,
        batch=args.batch,
    )
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, frames=frames, times=frame_times)
        print(f"Cached frames -> {cache} ({frames.nbytes / 2**20:.0f} MiB)")
    return frames


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #


def _degree_ticks(lo: float, hi: float, to_mercator, suffix: str) -> tuple[list[float], list[str]]:
    span = hi - lo
    step = next((s for s in (1, 2, 5, 10, 15, 20, 30) if span / s <= 8), 30)
    start = math.ceil(lo / step) * step
    values = np.arange(start, hi + 1e-9, step)
    return list(to_mercator(values)), [f"{v:g}°{suffix}" for v in values]


def render(args, frames: np.ndarray, frame_times: np.ndarray, ds: xr.Dataset) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter
    from matplotlib.colors import LogNorm

    import imageio_ffmpeg

    matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()

    lon_min, lon_max, lat_min, lat_max = args.extent

    # Colour limits fixed across frames, else the animation flickers.
    positive = frames[frames > 0]
    if positive.size == 0:
        raise SystemExit("every frame is empty — nothing to animate")
    vmax = args.vmax if args.vmax else float(np.percentile(positive, 99.9))
    vmin = args.vmin if args.vmin else vmax / 10**args.decades
    print(f"colour scale: {vmin:.3g} .. {vmax:.3g} (log, {args.decades} decades)")

    zoom = args.zoom or auto_zoom(lon_min, lon_max, args.width)
    print(f"fetching basemap tiles (zoom {zoom})")
    basemap, basemap_extent = fetch_basemap(args.extent, zoom, args.tile_cache)

    # Cell edges -> Mercator. Regular in lon; non-uniform in lat, which pcolormesh
    # handles exactly, so the raster needs no reprojection.
    x_edges = lon_to_x(ds["longitude_edge"].values)
    y_edges = lat_to_y(ds["latitude_edge"].values)

    view = (lon_to_x(lon_min), lon_to_x(lon_max), lat_to_y(lat_min), lat_to_y(lat_max))
    aspect = (view[3] - view[2]) / (view[1] - view[0])

    # The map axes is drawn at equal aspect, so size the figure to the map rather
    # than letterboxing it: axes_h/axes_w must equal the Mercator aspect. H.264
    # needs even pixel dimensions.
    dpi = args.dpi
    map_l, map_b, map_w, map_h = 0.055, 0.05, 0.85, 0.895
    fig_w_px = args.width - (args.width % 2)
    fig_w = fig_w_px / dpi
    fig_h_px = int(round(fig_w * (map_w / map_h) * aspect * dpi))
    fig_h_px += fig_h_px % 2
    fig_h = fig_h_px / dpi

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([map_l, map_b, map_w, map_h])
    cax = fig.add_axes([map_l + map_w + 0.012, map_b, 0.015, map_h])
    print(f"figure: {fig_w_px}x{fig_h_px} px")

    ax.imshow(basemap, extent=basemap_extent, origin="upper", interpolation="bilinear", zorder=0)

    cmap = plt.get_cmap(args.cmap).copy()
    cmap.set_bad(alpha=0.0)  # zero/empty cells stay transparent

    first = np.ma.masked_less_equal(frames[0], 0.0)
    mesh = ax.pcolormesh(
        x_edges, y_edges, first,
        cmap=cmap, norm=LogNorm(vmin=vmin, vmax=vmax),
        alpha=0.85, shading="flat", zorder=2,
    )

    site_x, site_y = _site_markers(ds)
    ax.scatter(site_x, site_y, s=16, c="cyan", edgecolors="black", linewidths=0.5, zorder=3)

    ax.set_xlim(view[0], view[1])
    ax.set_ylim(view[2], view[3])
    xticks, xlabels = _degree_ticks(lon_min, lon_max, lon_to_x, "")
    yticks, ylabels = _degree_ticks(lat_min, lat_max, lat_to_y, "N")
    ax.set_xticks(xticks); ax.set_xticklabels(xlabels, fontsize=8)
    ax.set_yticks(yticks); ax.set_yticklabels(ylabels, fontsize=8)
    ax.tick_params(length=2, colors="#666666")
    for spine in ax.spines.values():
        spine.set_edgecolor("#cccccc")

    cbar = fig.colorbar(mesh, cax=cax)
    cbar.set_label("combined surface sensitivity (mol/mol)/(mol/m²/s)", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    n_sites = len(set(ds["site"].values.tolist()))
    title = ax.set_title("", fontsize=11, loc="left")

    def stamp(when: np.datetime64) -> str:
        return np.datetime_as_string(when, unit="m").replace("T", " ") + " UTC"

    out = args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = FFMpegWriter(
        fps=args.fps,
        bitrate=args.bitrate,
        metadata={"title": "GLIDE combined network footprints", "artist": "GLIDE"},
    )
    print(f"writing {len(frames)} frames -> {out}")
    with writer.saving(fig, str(out), dpi):
        for i, when in enumerate(frame_times):
            mesh.set_array(np.ma.masked_less_equal(frames[i], 0.0).ravel())
            title.set_text(f"GLIDE — {n_sites} sites, 5-day back-trajectory footprint  ·  {stamp(when)}")
            writer.grab_frame()
            if (i + 1) % 25 == 0 or i == len(frames) - 1:
                print(f"\r  rendered {i + 1}/{len(frames)}", end="", flush=True)
    print()
    plt.close(fig)
    print(f"done: {out}  ({out.stat().st_size / 2**20:.1f} MiB, {len(frames) / args.fps:.0f} s)")


def _site_markers(ds: xr.Dataset) -> tuple[np.ndarray, np.ndarray]:
    """One (x, y) per unique site, in Mercator."""
    sites = ds["site"].values
    lons = ds["release_lon"].values
    lats = ds["release_lat"].values
    _, first = np.unique(sites, return_index=True)
    return lon_to_x(lons[first]), lat_to_y(lats[first])


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Animate GLIDE combined multi-site footprints as an MP4.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--run-dir", type=Path, default=Path("~/data/glide/outputs/icos-202401"),
                   help="GLIDE output directory containing footprints.zarr")
    p.add_argument("--out", type=Path, default=None,
                   help="output .mp4 (default: <run-dir name>-footprints.mp4 in outputs/)")
    p.add_argument("--stride", type=int, default=1, help="use every Nth release time")
    p.add_argument("--max-frames", type=int, default=None, help="stop after N frames")
    p.add_argument("--fps", type=int, default=12, help="frames per second")
    p.add_argument("--width", type=int, default=1600, help="output width in pixels")
    p.add_argument("--dpi", type=int, default=100)
    p.add_argument("--bitrate", type=int, default=6000, help="kbit/s")
    p.add_argument("--cmap", default="magma")
    p.add_argument("--extent", type=float, nargs=4, default=list(DEFAULT_EXTENT),
                   metavar=("LON_MIN", "LON_MAX", "LAT_MIN", "LAT_MAX"),
                   help="map view; default is the Europe/Atlantic region")
    p.add_argument("--full-domain", action="store_true", help="view the whole stored grid instead")
    p.add_argument("--zoom", type=int, default=None, help="basemap tile zoom (default: auto)")
    p.add_argument("--vmin", type=float, default=None, help="colour scale floor (default: vmax/10^decades)")
    p.add_argument("--vmax", type=float, default=None, help="colour scale ceiling (default: 99.9th pct)")
    p.add_argument("--decades", type=float, default=6.0, help="log decades below vmax")
    p.add_argument("--surface-layer-depth", type=float, default=DEFAULT_SURFACE_LAYER_DEPTH_M)
    p.add_argument("--air-density", type=float, default=DEFAULT_AIR_DENSITY_KG_M3)
    p.add_argument("--batch", type=int, default=8, help="release-times computed per dask pass")
    p.add_argument("--cache", type=Path, default=None, help="frame cache .npz (default: alongside --out)")
    p.add_argument("--refresh-cache", action="store_true", help="recompute frames, ignoring any cache")
    p.add_argument("--tile-cache", type=Path, default=Path("~/.cache/glide/tiles"),
                   help="on-disk basemap tile cache")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.run_dir = args.run_dir.expanduser()
    args.tile_cache = args.tile_cache.expanduser()
    if args.out is None:
        args.out = Path("outputs") / f"{args.run_dir.name}-footprints.mp4"
    args.out = args.out.expanduser()
    if args.cache is None:
        args.cache = args.out.with_suffix(".frames.npz")

    store = args.run_dir / "footprints.zarr"
    if not store.exists():
        raise SystemExit(f"no footprints.zarr under {args.run_dir}")

    ds = xr.open_zarr(store)
    footprint = ds["footprint"]
    if "release" not in footprint.dims:
        raise SystemExit("this store has no `release` axis — not a multi-site/periodic run")

    release_times = ds["release_time"].values
    frame_times = np.unique(release_times)[:: args.stride]
    if args.max_frames:
        frame_times = frame_times[: args.max_frames]

    if args.full_domain:
        args.extent = (
            float(ds["longitude_edge"].min()), float(ds["longitude_edge"].max()),
            float(ds["latitude_edge"].min()), float(ds["latitude_edge"].max()),
        )
    else:
        args.extent = tuple(args.extent)

    n_sites = len(set(ds["site"].values.tolist()))
    print(f"store:  {store}")
    print(f"        {n_sites} sites x {len(np.unique(release_times))} release times "
          f"= {footprint.sizes['release']} footprints")
    print(f"frames: {len(frame_times)} (stride {args.stride}) "
          f"{np.datetime_as_string(frame_times[0], unit='h')} .. "
          f"{np.datetime_as_string(frame_times[-1], unit='h')}")
    print(f"view:   lon {args.extent[0]}..{args.extent[1]}, lat {args.extent[2]}..{args.extent[3]}")

    frames = load_or_compute_frames(args, footprint, release_times, frame_times)
    render(args, frames, frame_times, ds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
