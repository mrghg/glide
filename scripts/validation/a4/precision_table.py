#!/usr/bin/env python
"""Task A4: float32 resolution of a particle position, degrees vs the local metric frame.

    python scripts/validation/a4/precision_table.py [--markdown]

For each met domain in ``configs/*.yaml`` and a few larger illustrative domains
(global; the Arctic; a polar cap reaching the model's cos(lat) clamp), samples a
dense lon/lat grid over the domain and reports the spacing between adjacent
float32 values of the position (one ulp), in true metres on the model's sphere:

- degrees: x = lon, y = lat; east ulp(lon) * 111320 cos(lat), north ulp(lat) * 110540;
- local metric (``lpdm.local_frame``): x, y metres from the domain centre; east
  ulp(x) * cos(lat)/cos(lat0), north ulp(y).

Also the stretch of the metric frame's x axis, cos(lat0)/cos(lat): an x metre
is a true metre only at lat0. The motion is the same in either frame (the
frame is affine in lon and lat and the metric factor is applied exactly), so
the stretch is not an error; it only sets how the float32 rounding of x maps
to true metres.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))

from lpdm.local_frame import COS_LAT_MIN, HorizontalFrame  # noqa: E402

EXTRA = {
    "global (illustrative)": ((-180.0, 180.0), (-90.0, 90.0)),
    "Arctic 50-85N (illustrative)": ((-180.0, 180.0), (50.0, 85.0)),
    "North America NAM stores (illustrative)": ((-140.0, -50.0), (15.0, 65.0)),
}


def ulp(v: np.ndarray) -> np.ndarray:
    return np.spacing(np.abs(v).astype(np.float32)).astype(np.float64)


def domain_row(name: str, lon_b, lat_b) -> dict:
    f = HorizontalFrame.local_metric(lon_b, lat_b)
    lon = np.linspace(lon_b[0], lon_b[1], 721)
    lat = np.linspace(lat_b[0], lat_b[1], 361)
    lo, la = np.meshgrid(lon, lat)
    cos_la = np.maximum(np.abs(np.cos(np.radians(la))), COS_LAT_MIN)
    cos0 = max(abs(math.cos(math.radians(f.lat0))), COS_LAT_MIN)
    deg_e = ulp(lo) * 111320.0 * cos_la
    deg_n = ulp(la) * 110540.0
    x = (lo - f.lon0) * f.kx
    y = (la - f.lat0) * f.ky
    met_e = ulp(x) * cos_la / cos0
    met_n = ulp(y)
    deg = np.maximum(deg_e, deg_n)
    met = np.maximum(met_e, met_n)
    stretch = cos0 / cos_la
    return {
        "domain": name,
        "lon": lon_b,
        "lat": lat_b,
        "lon0": f.lon0,
        "lat0": f.lat0,
        "deg_max": float(deg.max()),
        "deg_median": float(np.median(deg)),
        "met_max": float(met.max()),
        "met_median": float(np.median(met)),
        "metric_better_fraction": float((met < deg).mean()),
        "stretch_min": float(stretch.min()),
        "stretch_max": float(stretch.max()),
        "x_max_m": float(np.abs(x).max()),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--markdown", action="store_true")
    args = p.parse_args()
    domains: dict[str, tuple] = {}
    for cfg in sorted((REPO / "configs").glob("*.yaml")):
        md = (yaml.safe_load(cfg.read_text()) or {}).get("met_domain")
        if md:
            key = (tuple(md["lon_bounds"]), tuple(md["lat_bounds"]))
            domains.setdefault(key, []).append(cfg.name)
    rows = [domain_row(", ".join(v), *k) for k, v in domains.items()]
    rows += [domain_row(n, *b) for n, b in EXTRA.items()]
    hdr = (
        "| met domain | lon, lat | centre (lon0, lat0) | position ulp, degrees: median / max (m) "
        "| local metric: median / max (m) | metric finer at | x stretch cos(lat0)/cos(lat) |"
    )
    print(hdr)
    print("|" + "---|" * 7)
    for r in rows:
        print(
            f"| {r['domain']} | {r['lon'][0]:g}..{r['lon'][1]:g}, {r['lat'][0]:g}..{r['lat'][1]:g} "
            f"| {r['lon0']:.2f}, {r['lat0']:.2f} | {r['deg_median']:.3f} / {r['deg_max']:.3f} "
            f"| {r['met_median']:.3f} / {r['met_max']:.3f} | {100 * r['metric_better_fraction']:.0f}% "
            f"of the domain | {r['stretch_min']:.2f} to {r['stretch_max']:.2f} |"
        )


if __name__ == "__main__":
    main()
