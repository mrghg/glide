#!/usr/bin/env python
"""Tables of counting-rule error against the fine-time reference, over seeds (task A2).

Reads the JSON files written by ``score_residence_counting.py``; files are grouped
by their name with the trailing ``_s<seed>`` removed. For each group and each
weighting (``w`` density-weighted, ``u`` plain seconds) it prints the mean ± sample
s.d. over seeds of (estimator - reference) / reference, in %, for each region,
and the cell-wise L1 error of the 0-40 m grid. Plume files also get the
crosswind-integrated column scores against the reference and the closed form.

Usage: summarise_residence_scores.py --out <stem> <json> ...
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

REGIONS = [
    "z0_ring0",
    "z0_ring1",
    "z0_ring2-3",
    "z0_ring4-10",
    "z0_ring11+",
    "z0_total",
    "z1_total",
    "z0_bin0",
    "z0_bin1",
    "z0_bin2",
    "z0_bin6+",
    "total",
]
RULE_ROWS = ["endpoint", "step_mean", "trapezoid"]


def fmt(v: list[float], scale: float = 100.0, digits: int = 2) -> str:
    a = np.array(v, dtype=float) * scale
    a = a[np.isfinite(a)]
    if a.size == 0:
        return "–"
    if a.size == 1:
        return f"{a[0]:+.{digits}f}"
    return f"{a.mean():+.{digits}f} ± {a.std(ddof=1):.{digits}f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--weighting", default="w")
    ap.add_argument("files", nargs="+")
    args = ap.parse_args()
    groups: dict[str, list[dict]] = defaultdict(list)
    for f in args.files:
        name = re.sub(r"_s\d+$", "", Path(f).stem)
        groups[name].append(json.loads(Path(f).read_text()))
    lines = []
    summary: dict = {}
    for gname, runs in sorted(groups.items()):
        wk = args.weighting
        ref = runs[0]["reference"]
        names = list(runs[0]["scores"][wk]["rel"].keys())
        regions = [r for r in REGIONS if r in runs[0]["scores"][wk]["rel"][names[0]]]
        lines.append(f"\n### {gname} ({len(runs)} seeds; reference {ref}; weighting {wk})\n")
        lines.append("Relative error against the reference, % (mean ± s.d. over seeds):\n")
        lines.append("| estimator | " + " | ".join(regions) + " | L1 0-40 m |")
        lines.append("|" + "---|" * (len(regions) + 2))
        summary[gname] = {}
        rows = RULE_ROWS + [n for n in names if n not in RULE_ROWS]
        for n in rows:
            if n not in names:
                continue
            vals = {r: [run["scores"][wk]["rel"][n][r] for run in runs] for r in regions}
            l1 = [run["scores"][wk]["l1_z0"][n] for run in runs]
            summary[gname][n] = {r: vals[r] for r in regions} | {"l1_z0": l1}
            lines.append(
                f"| {n} | " + " | ".join(fmt(vals[r]) for r in regions) + f" | {fmt(l1)} |"
            )
        ref_tot = [run["scores"][wk]["ref"] for run in runs]
        lines.append(
            "\nReference totals (seed mean): "
            + ", ".join(f"{r} {np.mean([t[r] for t in ref_tot]):.4g}" for r in regions)
        )
        if "plume_scores" in runs[0]:
            keys = list(runs[0]["plume_scores"]["endpoint"].keys())
            lines.append(
                "\nPlume, crosswind-integrated 0-40 m columns (mean ± s.d. over seeds, %):\n"
            )
            lines.append("| estimator | " + " | ".join(keys) + " |")
            lines.append("|" + "---|" * (len(keys) + 1))
            for n in rows:
                if n not in runs[0]["plume_scores"]:
                    continue
                lines.append(
                    f"| {n} | "
                    + " | ".join(fmt([run["plume_scores"][n][k] for run in runs]) for k in keys)
                    + " |"
                )
                summary[gname][n]["plume"] = {
                    k: [run["plume_scores"][n][k] for run in runs] for k in keys
                }
    text = "\n".join(lines)
    print(text)
    Path(args.out + ".md").write_text(text + "\n")
    Path(args.out + ".json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
