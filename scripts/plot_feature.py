#!/usr/bin/env python
"""Render a still map of a GLIDE multi-site run's combined footprint.

This makes the README feature image. The styling lives in `footprint_maps.py`
and is shared with `animate_footprints.py`, so the still and the video match.

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
from footprint_maps import (  # noqa: E402
    add_style_args,
    build_map,
    colour_ceiling,
    compute_frames,
    hour_stamp,
    n_sites,
)


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


def save(args, ds: xr.Dataset, field: np.ndarray, caption: str, out: Path) -> None:
    import matplotlib.pyplot as plt

    vmax = colour_ceiling(field, args.vmax)
    fmap = build_map(args, ds, vmax)
    fmap.show(field, caption)
    out.parent.mkdir(parents=True, exist_ok=True)
    fmap.fig.savefig(out, dpi=fmap.fig.dpi, facecolor=fmap.fig.get_facecolor())
    plt.close(fmap.fig)
    print(f"wrote {out}  (colour scale {fmap.norm.vmin:.3g} .. {vmax:.3g})")


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
    add_style_args(p, width=2400)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.run_dir = args.run_dir.expanduser()
    args.extent = tuple(args.extent)

    ds = xr.open_zarr(args.run_dir / "footprints.zarr")
    footprint = ds["footprint"]
    release_times = ds["release_time"].values
    unique_times = np.unique(release_times)

    if args.mean:
        field = mean_footprint(footprint, args)
        start = np.datetime_as_string(unique_times[0], unit="D")
        end = np.datetime_as_string(unique_times[-1], unit="D")
        caption = f"GLIDE · {n_sites(ds)} sites · mean 5-day footprint, {start} to {end}"
        save(args, ds, field, caption, Path(str(args.out).replace("{time}", "mean")))
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
        caption = f"GLIDE · {n_sites(ds)} sites · 5-day footprint · {hour_stamp(when)}"
        stamp = np.datetime_as_string(when, unit="h")
        save(args, ds, field, caption, Path(str(args.out).replace("{time}", stamp)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
