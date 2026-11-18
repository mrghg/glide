#!/usr/bin/env python
"""Animate a GLIDE multi-site run's combined footprints as an MP4.

For each release hour the footprints of every site released at that instant are
summed and converted to STILT-style surface sensitivity
(`lpdm.comparison.to_stilt_surface_footprint`) — the same physics bridge the
validation notebooks use. Frames use the README feature image's style, from
`footprint_maps.py`: Natural Earth land and coastlines on a dark Web Mercator
map, and a log-scaled magma raster that fades out over its faint tail. The
colour scale is fixed across frames so the animation does not flicker.

Reading the store dominates runtime, so computed frames are cached to a `.npz`
alongside the output; re-running to tweak styling reuses the cache.

Typical use:

    .venv/bin/python scripts/animate_footprints.py \
        --run-dir ~/data/glide/outputs/icos-model-levels \
        --out outputs/icos-model-levels-footprints.mp4

    # quick look: every 6th hour
    .venv/bin/python scripts/animate_footprints.py --stride 6 --fps 8
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from footprint_maps import (  # noqa: E402
    add_style_args,
    build_map,
    colour_ceiling,
    compute_frames,
    full_domain_extent,
    hour_stamp,
    n_sites,
)


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


def render(args, frames: np.ndarray, frame_times: np.ndarray, ds: xr.Dataset) -> None:
    import imageio_ffmpeg
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter

    matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()

    vmax = colour_ceiling(frames, args.vmax)
    fmap = build_map(args, ds, vmax)
    width_px, height_px = fmap.size_px
    print(f"colour scale: {fmap.norm.vmin:.3g} .. {vmax:.3g} (log, {args.decades:g} decades)")
    print(f"figure: {width_px}x{height_px} px")

    out = args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = FFMpegWriter(
        fps=args.fps,
        bitrate=args.bitrate,
        metadata={"title": "GLIDE combined network footprints", "artist": "GLIDE"},
    )
    prefix = f"GLIDE · {n_sites(ds)} sites · 5-day footprint · "
    print(f"writing {len(frames)} frames -> {out}")
    with writer.saving(fmap.fig, str(out), fmap.fig.dpi):
        for i, when in enumerate(frame_times):
            fmap.show(frames[i], prefix + hour_stamp(when))
            writer.grab_frame(facecolor=fmap.fig.get_facecolor())
            if (i + 1) % 25 == 0 or i == len(frames) - 1:
                print(f"\r  rendered {i + 1}/{len(frames)}", end="", flush=True)
    print()
    plt.close(fmap.fig)
    print(f"done: {out}  ({out.stat().st_size / 2**20:.1f} MiB, {len(frames) / args.fps:.0f} s)")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Animate GLIDE combined multi-site footprints as an MP4.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--run-dir",
        type=Path,
        default=Path("~/data/glide/outputs/icos-model-levels"),
        help="GLIDE output directory containing footprints.zarr",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output .mp4 (default: <run-dir name>-footprints.mp4 in outputs/)",
    )
    p.add_argument("--stride", type=int, default=1, help="use every Nth release time")
    p.add_argument("--max-frames", type=int, default=None, help="stop after N frames")
    p.add_argument("--fps", type=int, default=12, help="frames per second")
    p.add_argument("--bitrate", type=int, default=6000, help="kbit/s")
    p.add_argument("--full-domain", action="store_true", help="view the whole stored grid instead")
    p.add_argument("--batch", type=int, default=8, help="release-times computed per dask pass")
    p.add_argument(
        "--cache", type=Path, default=None, help="frame cache .npz (default: alongside --out)"
    )
    p.add_argument(
        "--refresh-cache", action="store_true", help="recompute frames, ignoring any cache"
    )
    add_style_args(p, width=1600)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.run_dir = args.run_dir.expanduser()
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

    args.extent = full_domain_extent(ds) if args.full_domain else tuple(args.extent)

    print(f"store:  {store}")
    print(
        f"        {n_sites(ds)} sites x {len(np.unique(release_times))} release times "
        f"= {footprint.sizes['release']} footprints"
    )
    print(
        f"frames: {len(frame_times)} (stride {args.stride}) "
        f"{np.datetime_as_string(frame_times[0], unit='h')} .. "
        f"{np.datetime_as_string(frame_times[-1], unit='h')}"
    )
    print(f"view:   lon {args.extent[0]}..{args.extent[1]}, lat {args.extent[2]}..{args.extent[3]}")

    frames = load_or_compute_frames(args, footprint, release_times, frame_times)
    render(args, frames, frame_times, ds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
