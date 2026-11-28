"""Task CONV: compare footprint runs against a reference run.

    .venv/bin/python scripts/validation/conv/compare_footprints.py <ref dir> <other dir> ...

For each run: the footprint summed over everything but the z bin (per z bin and
total), its ratio to the reference, the log-ratio spread over cells holding
1e-3 of the reference 0-40 m maximum, whether the endpoints are bit-identical,
and, from the run's log (``<dir>.log``), the convection events and the number of
particles they displaced.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


def load(d: str) -> np.ndarray:
    fp = xr.open_zarr(Path(d) / "footprints.zarr")["footprint"].values.astype(np.float64)
    return fp.sum(axis=(0, 1))  # [z, y, x]


def log_stats(d: str) -> str:
    log = Path(str(d).rstrip("/") + ".log")
    if not log.exists():
        return "no log"
    txt = log.read_text()
    events = re.findall(
        r"fires in (\d+) / \d+ columns.*?(\d+) hold active particles\. Displaced (\d+) / (\d+)", txt
    )
    if not events:
        return "no convection events logged"
    held = sum(int(e[1]) > 0 for e in events)
    disp = sum(int(e[2]) for e in events)
    act = sum(int(e[3]) for e in events)
    return f"{len(events)} events, {held} with occupied firing columns, displaced {disp} of {act} particle-events ({disp / max(act, 1):.3%})"


def main() -> None:
    ref_dir, *others = sys.argv[1:]
    ref = load(ref_dir)
    zb = xr.open_zarr(Path(ref_dir) / "footprints.zarr")
    edges = list(zip(zb["z_bottom_m"].values, zb["z_top_m"].values, strict=True))
    print(f"reference {ref_dir}: per z bin {[f'{a:.0f}-{b:.0f} m' for a, b in edges]}")
    print(
        f"   sums {np.array2string(ref.sum(axis=(1, 2)), precision=6)} total {ref.sum():.6g}; {log_stats(ref_dir)}"
    )
    ep_ref = pd.read_parquet(Path(ref_dir) / "endpoint_particles.parquet")
    for d in others:
        fp = load(d)
        rel = fp.sum(axis=(1, 2)) / ref.sum(axis=(1, 2)) - 1.0
        tot = fp.sum() / ref.sum() - 1.0
        mask = ref[0] > 1e-3 * ref[0].max()
        lr = np.log10(np.maximum(fp[0][mask], 1e-30) / ref[0][mask])
        ep = pd.read_parquet(Path(d) / "endpoint_particles.parquet")
        same = ep.equals(ep_ref)
        print(f"{d}:")
        print(
            f"   per z bin rel. change {np.array2string(rel, precision=4)}, total {tot:+.4%}; bit-identical footprint {np.array_equal(fp, ref)}, endpoints {same}"
        )
        print(
            f"   0-40 m log10 ratio over {mask.sum()} cells: p5/p50/p95 {np.percentile(lr, [5, 50, 95]).round(4)}"
        )
        print(f"   {log_stats(d)}")


if __name__ == "__main__":
    main()
