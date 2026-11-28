#!/usr/bin/env python
"""Check the residence scorer's offline rule tallies against production runs (task A2).

The scorer (`score_residence_counting.py config`) hooks an endpoint run and counts
every rule offline from the same trajectories. Deposition draws no random numbers,
so a production run of the same config and seed with another
`output_grid.deposition.position` follows the same trajectories, and its footprint
should equal the scorer's tally for that rule (to float32 rounding of the
production grid). Also checks that the hooked run's own footprint is bit-identical
to an unhooked endpoint run (the hooks do not perturb the model).

Usage: check_scorer_against_runs.py <scores.npz> <hooked run dir> <run dir per rule: rule=dir ...>
"""

from __future__ import annotations

import sys

import numpy as np
import xarray as xr


def load(run: str) -> xr.DataArray:
    return xr.open_zarr(f"{run}/footprints.zarr")["footprint"].load()


def main() -> None:
    npz = np.load(sys.argv[1])
    hooked = load(sys.argv[2])
    weighting = hooked.attrs.get("density_weighting", "dry_air")
    wkey = "w" if weighting == "dry_air" else "u"
    for arg in sys.argv[3:]:
        rule, run = arg.split("=", 1)
        fp = load(run)
        if rule == "endpoint":
            same = bool(np.array_equal(fp.values, hooked.values))
            print(f"hooked run vs unhooked endpoint run: bit-identical={same}")
        prod = fp.sum(dim=("release", "time_ago")).values.astype(np.float64)  # (z, y, x)
        tally = npz[f"{wkey}|{rule}|grid"]
        diff = np.abs(prod - tally)
        scale = np.abs(tally).max()
        print(
            f"{rule:10s} production total {prod.sum():.6e} scorer {tally.sum():.6e} "
            f"rel {(prod.sum() - tally.sum()) / tally.sum():+.2e}; max cell |diff| / max cell "
            f"{diff.max() / scale:.2e}; z0 rel {(prod[0].sum() - tally[0].sum()) / tally[0].sum():+.2e}"
        )
        if fp.sizes["time_ago"] > 1:
            # Production bins are the run's time_bin_seconds; compare when they are hourly.
            per_bin = fp.sum(dim=("release", "z_bin", "latitude", "longitude")).values
            tz = npz[f"{wkey}|{rule}|tz"].sum(axis=1)
            n = min(len(per_bin), len(tz))
            rel = (per_bin[:n] - tz[:n]) / np.where(tz[:n] > 0, tz[:n], 1)
            print(f"{'':10s} per time-ago bin rel diff: " + " ".join(f"{r:+.1e}" for r in rel))


if __name__ == "__main__":
    main()
