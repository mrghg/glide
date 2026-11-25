"""Footprint change of a run against a golden fixture, per release and height bin.

    python scripts/validation/wm_near_surface/golden_change.py <fixture dir> <run dir> [...]

For each release prints the total footprint (all cells, times and heights) and
the total in each output height bin (the first is 0-40 m on the shipped
configs), for the fixture and the run, and their relative differences; then the
median and range over releases. Used to report how decision 0027 moves the
fixtures without regenerating them.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "agent"))
import golden_common as gc  # noqa: E402


def totals(fp: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per release: total, and total per height bin ``[release, z]``."""
    f = np.nan_to_num(fp.astype(np.float64))
    return f.sum(axis=(1, 2, 3, 4)), f.sum(axis=(1, 3, 4))


def release_groups(run: Path) -> dict[str, np.ndarray]:
    """Release indices by site and by day (09-15 UTC) / night (19-05 UTC)."""
    store = run / "footprints.zarr"
    if not store.exists():
        return {}
    import xarray as xr

    ds = xr.open_zarr(store)
    if "site" not in ds.coords:
        return {}
    site = np.asarray(ds["site"].values).astype(str)
    hour = ds["release_time"].dt.hour.values
    day, night = (hour >= 9) & (hour <= 15), (hour >= 19) | (hour <= 5)
    out = {}
    for s in sorted(set(site)):
        for label, m in (("all", np.ones_like(day)), ("day", day), ("night", night)):
            idx = np.flatnonzero((site == s) & m)
            if idx.size:
                out[f"{s} {label}"] = idx
    return out


def main() -> None:
    fixture, *runs = (Path(a) for a in sys.argv[1:])
    manifest = json.loads((fixture / gc.MANIFEST_FILE).read_text())
    ref, _ = gc.read_fixture_footprint(gc.fixture_data_dir(fixture, manifest))
    ref_tot, ref_z = totals(ref)
    for run in runs:
        new, _ = gc.read_run_footprint(run)
        tot, z = totals(new)
        d_tot = tot / ref_tot - 1.0
        d_z = z / np.where(ref_z > 0, ref_z, np.nan) - 1.0
        print(f"## {run} vs {fixture}")
        print(f"releases: {len(tot)}; height bins: {z.shape[1]}")
        if len(tot) <= 4:
            for r in range(len(tot)):
                zs = ", ".join(f"bin{k} {d_z[r, k]:+.4f}" for k in range(z.shape[1]))
                print(f"release {r}: total {d_tot[r]:+.4f}; {zs}")
        for name, v in [("total", d_tot)] + [(f"bin{k}", d_z[:, k]) for k in range(z.shape[1])]:
            v = v[np.isfinite(v)]
            print(
                f"{name}: median {np.median(v):+.4f}, mean {v.mean():+.4f}, "
                f"min {v.min():+.4f}, max {v.max():+.4f}"
            )
        groups = release_groups(run)
        if groups:
            print("bin0 (0-40 m) and total by site and time of day (release hour UTC):")
            for label, idx in groups.items():
                v0, vt = d_z[idx, 0], d_tot[idx]
                print(
                    f"  {label:24s} n={idx.size:3d}  bin0 median {np.median(v0):+.4f} "
                    f"[{v0.min():+.4f}, {v0.max():+.4f}]  total median {np.median(vt):+.4f}"
                )


if __name__ == "__main__":
    main()
