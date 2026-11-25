#!/usr/bin/env python
"""Task A4: compare two GLIDE runs of the same config and seed (run A vs run B).

    python scripts/validation/a4/compare_runs.py <run A> <run B> [--json out.json] [--label txt]

Run directories hold ``footprints.zarr`` and ``endpoint_particles.parquet``.
Reports, with B as the reference:

- bit-identity of the footprint and the endpoint particles;
- per release: relative difference of the total (all cells) and of the 0-40 m
  layer total (z bin 0), and the largest cell difference over the largest cell;
- the 0-40 m totals summed by site and by day/night (release start 09-15 UTC
  is "day", every other hour "night"; the reference sites are within 10 degrees
  of Greenwich, so UTC is within 40 minutes of local solar time);
- endpoint particles: horizontal distance (metres, on the model's sphere) and
  height difference, median / 99th percentile / max, and the fraction
  bit-identical.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def read_footprint(run: Path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    import zarr

    g = zarr.open_group(str(run / "footprints.zarr"), mode="r")
    fp = np.asarray(g["footprint"][...], dtype=np.float64)
    coords = {"release_time": np.asarray(g["release_time"][...])}
    if "site" in g:
        coords["site"] = np.asarray(g["site"][...]).astype(str)
    return fp, coords


def rel(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(b != 0, (a - b) / b, np.where(a == b, 0.0, np.inf))


def summarise(x: np.ndarray) -> dict[str, float]:
    ax = np.abs(x[np.isfinite(x)])
    if ax.size == 0:
        return {"median_abs": float("nan"), "max_abs": float("nan")}
    return {"median_abs": float(np.median(ax)), "max_abs": float(np.max(ax))}


def compare(run_a: Path, run_b: Path) -> dict:
    fa, ca = read_footprint(run_a)
    fb, cb = read_footprint(run_b)
    if fa.shape != fb.shape:
        raise SystemExit(f"footprint shapes differ: {fa.shape} vs {fb.shape}")
    n_rel = fa.shape[0]
    out: dict = {"run_a": str(run_a), "run_b": str(run_b), "n_releases": n_rel}
    out["footprint_bit_identical"] = bool(np.array_equal(fa, fb))
    out["cells_differing"] = int(np.count_nonzero(fa != fb))
    out["cells_nonzero_b"] = int(np.count_nonzero(fb))

    tot_a = fa.reshape(n_rel, -1).sum(1)
    tot_b = fb.reshape(n_rel, -1).sum(1)
    low_a = fa[:, :, 0].reshape(n_rel, -1).sum(1)
    low_b = fb[:, :, 0].reshape(n_rel, -1).sum(1)
    cell = np.abs(fa - fb).reshape(n_rel, -1).max(1) / np.maximum(
        np.abs(fb).reshape(n_rel, -1).max(1), 1e-300
    )
    out["release_total_rel"] = summarise(rel(tot_a, tot_b))
    out["release_0_40m_rel"] = summarise(rel(low_a, low_b))
    out["release_max_cell_over_max"] = summarise(cell)
    out["per_release"] = [
        {
            "release": i,
            "site": str(cb["site"][i]) if "site" in cb else "",
            "release_time": str(cb["release_time"][i]),
            "total_rel": float(rel(tot_a[i : i + 1], tot_b[i : i + 1])[0]),
            "low_rel": float(rel(low_a[i : i + 1], low_b[i : i + 1])[0]),
            "max_cell_over_max": float(cell[i]),
        }
        for i in range(n_rel)
    ]

    hours = pd.DatetimeIndex(cb["release_time"]).hour
    day = (hours >= 9) & (hours <= 15)
    sites = cb.get("site", np.array([""] * n_rel))
    groups = {}
    for s in sorted(set(sites)):
        for name, sel in (("day", day), ("night", ~day), ("all", np.ones(n_rel, bool))):
            m = (sites == s) & sel
            if m.any():
                a, b = float(low_a[m].sum()), float(low_b[m].sum())
                groups[f"{s or 'all'}/{name}"] = {
                    "n": int(m.sum()),
                    "low_sum_a": a,
                    "low_sum_b": b,
                    "rel": (a - b) / b if b else float("nan"),
                    "per_release_low_rel_median_abs": float(
                        np.median(np.abs(rel(low_a[m], low_b[m])))
                    ),
                }
    out["low_by_site_daynight"] = groups
    out["release_total_sum_rel"] = float((tot_a.sum() - tot_b.sum()) / tot_b.sum())

    ea = pd.read_parquet(run_a / "endpoint_particles.parquet")
    eb = pd.read_parquet(run_b / "endpoint_particles.parquet")
    if len(ea) == len(eb):
        lon_a, lat_a, z_a = (ea[c].to_numpy(np.float64) for c in ("lon", "lat", "alt"))
        lon_b, lat_b, z_b = (eb[c].to_numpy(np.float64) for c in ("lon", "lat", "alt"))
        dx = (lon_a - lon_b) * 111320.0 * np.cos(np.deg2rad(lat_b))
        dy = (lat_a - lat_b) * 110540.0
        dh = np.hypot(dx, dy)
        dz = np.abs(z_a - z_b)
        same = (
            (ea["lon"].to_numpy() == eb["lon"].to_numpy())
            & (ea["lat"].to_numpy() == eb["lat"].to_numpy())
            & (ea["alt"].to_numpy() == eb["alt"].to_numpy())
        )

        def q(v: np.ndarray) -> dict[str, float]:
            return {
                "median": float(np.median(v)),
                "p90": float(np.quantile(v, 0.90)),
                "p99": float(np.quantile(v, 0.99)),
                "max": float(v.max()),
            }

        out["endpoints"] = {
            "n": len(ea),
            "fraction_bit_identical": float(same.mean()),
            "horizontal_m": q(dh),
            "vertical_m": q(dz),
        }
    return out


def text(r: dict, label: str) -> str:
    lines = [f"## {label}", f"A = {r['run_a']}", f"B = {r['run_b']}"]
    lines.append(
        f"footprint bit-identical: {r['footprint_bit_identical']} "
        f"({r['cells_differing']} of {r['cells_nonzero_b']} non-zero cells differ)"
    )
    for k in ("release_total_rel", "release_0_40m_rel", "release_max_cell_over_max"):
        v = r[k]
        lines.append(f"{k}: median |.| {v['median_abs']:.3e}, max |.| {v['max_abs']:.3e}")
    lines.append(f"sum of all release totals, rel: {r['release_total_sum_rel']:+.3e}")
    lines.append("0-40 m totals by site and day/night (A vs B):")
    for g, v in r["low_by_site_daynight"].items():
        lines.append(
            f"  {g:12s} n={v['n']:3d}  sum rel {v['rel']:+.3e}  "
            f"per-release median |rel| {v['per_release_low_rel_median_abs']:.3e}"
        )
    e = r.get("endpoints")
    if e:
        lines.append(
            f"endpoints: {e['n']} particles, {e['fraction_bit_identical']:.4f} bit-identical; "
            f"horizontal m median {e['horizontal_m']['median']:.3g} p90 "
            f"{e['horizontal_m']['p90']:.3g} p99 {e['horizontal_m']['p99']:.3g} max "
            f"{e['horizontal_m']['max']:.3g}; vertical m median {e['vertical_m']['median']:.3g} "
            f"p99 {e['vertical_m']['p99']:.3g} max {e['vertical_m']['max']:.3g}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("run_a", type=Path)
    p.add_argument("run_b", type=Path)
    p.add_argument("--json", type=Path)
    p.add_argument("--label", default="A vs B")
    args = p.parse_args(argv)
    r = compare(args.run_a, args.run_b)
    print(text(r, args.label))
    if args.json:
        args.json.write_text(json.dumps(r, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
