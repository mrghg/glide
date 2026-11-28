#!/usr/bin/env python
"""Worst-cell relative difference between two footprint runs, per release (task A2).

For each release and vertical bin, among the cells holding more than ``--frac`` of
that layer's maximum (in run B) and outside the 3 x 3 cells around the receptor,
take the cell with the largest |A/B - 1| and record its signed value. Prints, per
site and layer, the median and 90th percentile of |worst| over releases, the
maximum, and the fraction negative; and the per-layer total relative difference.
This is the metric of the PR #66 review's float32 scan.

Usage: compare_footprint_cells.py <run A> <run B> [--frac 0.25] [--out file.json]
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import xarray as xr


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--frac", type=float, default=0.25)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    da = xr.open_zarr(f"{args.a}/footprints.zarr")
    db = xr.open_zarr(f"{args.b}/footprints.zarr")
    A = da["footprint"].sum("time_ago").values.astype(np.float64)
    B = db["footprint"].sum("time_ago").values.astype(np.float64)
    lon_e = db["longitude_edge"].values
    lat_e = db["latitude_edge"].values
    rlon = db["release_lon"].values
    rlat = db["release_lat"].values
    sites = db["site"].values if "site" in db else np.array(["site"] * len(rlon))
    n_r, n_z = B.shape[0], B.shape[1]
    worst = np.full((n_r, n_z), np.nan)
    tot = np.full((n_r, n_z), np.nan)
    for r in range(n_r):
        ix = int(np.searchsorted(lon_e, rlon[r], side="right") - 1)
        iy = int(np.searchsorted(lat_e, rlat[r], side="right") - 1)
        for z in range(n_z):
            a, b = A[r, z], B[r, z]
            if b.max() <= 0:
                continue
            m = b > args.frac * b.max()
            m[max(iy - 1, 0) : iy + 2, max(ix - 1, 0) : ix + 2] = False
            tot[r, z] = a.sum() / b.sum() - 1.0
            if not m.any():
                continue
            rel = a[m] / b[m] - 1.0
            worst[r, z] = rel[np.argmax(np.abs(rel))]
    out = {}
    print(f"A={args.a}\nB={args.b}\nworst-cell signed rel diff % (A/B-1), cells > {args.frac} of layer max, outside receptor 3x3")
    for site in sorted(set(sites)):
        sel = sites == site
        for z in range(n_z):
            w = worst[sel, z]
            w = w[np.isfinite(w)] * 100
            t = tot[sel, z]
            t = t[np.isfinite(t)] * 100
            if w.size == 0:
                continue
            row = {
                "median_abs": float(np.median(np.abs(w))),
                "p90_abs": float(np.percentile(np.abs(w), 90)),
                "max_abs": float(np.abs(w).max()),
                "frac_negative": float((w < 0).mean()),
                "layer_total_median_pct": float(np.median(t)),
                "layer_total_max_abs_pct": float(np.abs(t).max()),
            }
            out[f"{site} z{z}"] = row
            print(
                f"{site} z{z} median|.| {row['median_abs']:.4f}  p90|.| {row['p90_abs']:.4f}  "
                f"max|.| {row['max_abs']:.4f}  frac negative {row['frac_negative']:.2f}  "
                f"layer total: median {row['layer_total_median_pct']:+.5f}  max|.| {row['layer_total_max_abs_pct']:.5f}"
            )
    if args.out:
        with open(args.out, "w") as f:
            json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
