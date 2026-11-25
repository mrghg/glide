#!/usr/bin/env python
"""Compare production runs made with each counting rule against endpoint (task A2).

Each run is ``<rule>:<seed>=<run dir>``; runs of one seed share their trajectories
(deposition draws no random numbers), so the rule-minus-endpoint change of a seed
is paired. For every release: the 0-40 m total, the 0-40 m receptor cell, the
0-40 m by Chebyshev ring of cells around it, and the 40-1000 m total. Releases are
grouped by site and by day / night / transition from the local solar time
(UTC + lon/15 h) of the middle of the release window: day 09-16 h, night 18-07 h.
Reports, per group, the mean over releases of (rule / endpoint - 1) for each seed,
the mean and sample standard deviation of that over seeds, and, for scale, the
spread of endpoint itself between seeds (the standard deviation over seeds of the
group's mean 0-40 m total, relative to its mean).

Usage: compare_counting_runs.py --out <stem> <rule>:<seed>=<dir> ...
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict

import numpy as np
import pandas as pd
import xarray as xr

RINGS = ((0, 0), (1, 1), (2, 3), (4, 10), (11, 100_000))
RING_NAMES = ("cell", "ring1", "ring2-3", "ring4-10", "ring11+")


def release_table(run: str) -> pd.DataFrame:
    ds = xr.open_zarr(f"{run}/footprints.zarr")
    fp = ds["footprint"].sum("time_ago")
    lon_e = ds["longitude_edge"].values
    lat_e = ds["latitude_edge"].values
    z0 = fp.isel(z_bin=0).values  # (release, y, x)
    z1 = fp.isel(z_bin=1).values.sum(axis=(1, 2))
    rows = []
    rel_lon = ds["release_lon"].values
    rel_lat = ds["release_lat"].values
    rel_time = pd.to_datetime(ds["release_time"].values)
    dur = ds["release_duration_seconds"].values
    sites = ds["site"].values if "site" in ds else np.array(["site"] * len(rel_lon))
    jj, ii = np.indices(z0.shape[1:])
    for r in range(z0.shape[0]):
        ix = int(np.searchsorted(lon_e, rel_lon[r], side="right") - 1)
        iy = int(np.searchsorted(lat_e, rel_lat[r], side="right") - 1)
        ring = np.maximum(abs(jj - iy), abs(ii - ix))
        mid = rel_time[r] + pd.Timedelta(seconds=float(dur[r]) / 2)
        lst = (mid.hour + mid.minute / 60 + rel_lon[r] / 15.0) % 24
        period = "day" if 9 <= lst < 16 else ("night" if (lst >= 18 or lst < 7) else "transition")
        row = {
            "release": r,
            "site": str(sites[r]),
            "time": str(rel_time[r]),
            "period": period,
            "z0": float(z0[r].sum()),
            "z1": float(z1[r]),
        }
        for (lo, hi), nm in zip(RINGS, RING_NAMES):
            row[nm] = float(z0[r][(ring >= lo) & (ring <= hi)].sum())
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("runs", nargs="+")
    args = ap.parse_args()
    tables: dict[tuple[str, int], pd.DataFrame] = {}
    for spec in args.runs:
        key, run = spec.split("=", 1)
        rule, seed = key.split(":")
        tables[(rule, int(seed))] = release_table(run)
    seeds = sorted({s for (_, s) in tables})
    rules = sorted({r for (r, _) in tables if r != "endpoint"})
    metrics = ["z0", *RING_NAMES, "z1"]
    groups = [("all", None, None)]
    t0 = tables[("endpoint", seeds[0])]
    for site in sorted(t0["site"].unique()):
        groups.append((site, site, None))
        for period in ("day", "night", "transition"):
            if ((t0["site"] == site) & (t0["period"] == period)).any():
                groups.append((f"{site} {period}", site, period))
    for period in ("day", "night"):
        groups.append((f"all {period}", None, period))

    def sel(df, site, period):
        m = np.ones(len(df), dtype=bool)
        if site is not None:
            m &= (df["site"] == site).values
        if period is not None:
            m &= (df["period"] == period).values
        return df[m]

    result: dict = defaultdict(dict)
    lines = []
    for rule in rules:
        lines.append(f"\n### {rule} vs endpoint (mean over releases of rule/endpoint - 1, %)\n")
        hdr = (
            "| group | n | "
            + " | ".join(f"{m}" for m in metrics)
            + " | endpoint 0-40 m seed spread |"
        )
        lines.append(hdr)
        lines.append("|" + "---|" * (len(metrics) + 3))
        for gname, site, period in groups:
            per_seed = {m: [] for m in metrics}
            ep_tot = []
            n = 0
            for s in seeds:
                if (rule, s) not in tables or ("endpoint", s) not in tables:
                    continue
                a = sel(tables[("endpoint", s)], site, period)
                b = sel(tables[(rule, s)], site, period)
                n = len(a)
                ep_tot.append(a["z0"].mean())
                for m in metrics:
                    ok = a[m].values > 0
                    per_seed[m].append(float(np.mean(b[m].values[ok] / a[m].values[ok] - 1.0)))
            cells = []
            res = {"n_releases": n}
            for m in metrics:
                v = np.array(per_seed[m]) * 100
                sd = float(v.std(ddof=1)) if len(v) > 1 else float("nan")
                res[m] = {"per_seed_pct": v.tolist(), "mean_pct": float(v.mean()), "sd_pct": sd}
                cells.append(f"{v.mean():+.2f} ± {sd:.2f}" if len(v) > 1 else f"{v.mean():+.2f}")
            spread = (
                float(np.std(ep_tot, ddof=1) / np.mean(ep_tot) * 100)
                if len(ep_tot) > 1
                else float("nan")
            )
            res["endpoint_z0_seed_spread_pct"] = spread
            result[rule][gname] = res
            lines.append(f"| {gname} | {n} | " + " | ".join(cells) + f" | {spread:.2f}% |")
    lines.insert(0, f"seeds: {seeds}; '±' is the sample s.d. over seeds of the per-seed mean")
    text = "\n".join(lines)
    print(text)
    with open(args.out + ".md", "w") as f:
        f.write(text + "\n")
    with open(args.out + ".json", "w") as f:
        json.dump(result, f, indent=1)


if __name__ == "__main__":
    main()
