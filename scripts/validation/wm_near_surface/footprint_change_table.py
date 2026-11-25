"""Footprint change against a golden fixture, over seeds, by site and day/night.

    python footprint_change_table.py <fixture dir> <label>=<run dir>[,<run dir>...] ...

Each ``<label>`` (e.g. ``n16``) names a set of runs that differ only in their
seed. For every run and release: the relative change of the 0-40 m footprint
(output height bin 0) and of the total against the fixture. Per group (all,
and each site's all/day/night releases, as ``golden_change.release_groups``) the
median over releases is taken for each run; the table gives the mean of those
medians over the seeds and their standard deviation (seed sd). When two labels
are given, the same-seed difference between the first two (paired by position
in their lists) is also tabulated: the median over releases of
``fp_b/fp_a - 1`` per seed, mean and sd over seeds. Markdown on stdout.
"""

from __future__ import annotations

import statistics as st
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "agent"))
import golden_change as gch  # noqa: E402
import golden_common as gc  # noqa: E402


def fmt(vals: list[float]) -> str:
    if not vals:
        return "–"
    m = st.mean(vals)
    sd = st.stdev(vals) if len(vals) > 1 else float("nan")
    return f"{100 * m:+.1f}% (sd {100 * sd:.1f})"


def main() -> None:
    fixture = Path(sys.argv[1])
    sets = []
    for arg in sys.argv[2:]:
        label, runs = arg.split("=", 1)
        sets.append((label, [Path(r) for r in runs.split(",")]))
    import json

    manifest = json.loads((fixture / gc.MANIFEST_FILE).read_text())
    ref, _ = gc.read_fixture_footprint(gc.fixture_data_dir(fixture, manifest))
    ref_tot, ref_z = gch.totals(ref)

    per_run = {}
    groups = None
    for _label, runs in sets:
        for run in runs:
            new, _ = gc.read_run_footprint(run)
            tot, z = gch.totals(new)
            per_run[run] = (tot, z)
            if groups is None:
                groups = gch.release_groups(run) or {"all": np.arange(len(tot))}
    if len(groups) > 1 or "all" not in groups:
        groups = {"all releases": np.arange(len(ref_tot)), **groups}

    print(f"Fixture: `{fixture}`; runs: " + "; ".join(f"{lab}: {len(r)} seeds" for lab, r in sets))
    print()
    head = "| group | n releases | " + " | ".join(f"{lab} 0–40 m | {lab} total" for lab, _ in sets)
    print(head + " |")
    print("|---|---|" + "---|---|" * len(sets))
    for gname, idx in groups.items():
        cells = []
        for _label, runs in sets:
            d0, dt = [], []
            for run in runs:
                tot, z = per_run[run]
                d0.append(float(np.nanmedian(z[idx, 0] / ref_z[idx, 0] - 1.0)))
                dt.append(float(np.median(tot[idx] / ref_tot[idx] - 1.0)))
            cells += [fmt(d0), fmt(dt)]
        print(f"| {gname} | {len(idx)} | " + " | ".join(cells) + " |")
    print()
    print(
        "Each cell: median over the group's releases of (run / fixture − 1), mean over "
        "seeds, seed standard deviation in brackets."
    )
    if len(sets) >= 2:
        (la, ra), (lb, rb) = sets[0], sets[1]
        print()
        print(f"Same-seed change {lb} against {la} (median over releases of {lb}/{la} − 1):")
        print()
        print("| group | 0–40 m | 40–1000 m | 1–5 km | total |")
        print("|---|---|---|---|---|")
        for gname, idx in groups.items():
            cols = [[], [], [], []]
            for a, b in zip(ra, rb, strict=True):
                ta, za = per_run[a]
                tb, zb = per_run[b]
                for k in range(3):
                    cols[k].append(float(np.nanmedian(zb[idx, k] / za[idx, k] - 1.0)))
                cols[3].append(float(np.median(tb[idx] / ta[idx] - 1.0)))
            print(f"| {gname} | " + " | ".join(fmt(c) for c in cols) + " |")


if __name__ == "__main__":
    main()
