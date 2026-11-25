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


if __name__ == "__main__":
    main()
