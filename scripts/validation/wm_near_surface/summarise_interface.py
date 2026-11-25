"""Tables (markdown) from interface_study.py results.

    python summarise_interface.py <study dir> > tables.md

Every cell is the mean over seeds ± its standard error, with the seed
standard deviation (sd) alongside; each seed's value is itself a mean over the
2-6 h records (``openbl``: also the 6-h value).
"""

from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

REG = ("stable", "neutral", "convective")


def load(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def stat(vals: list[float]) -> tuple[float, float, float, int]:
    k = len(vals)
    m = sum(vals) / k
    sd = math.sqrt(sum((v - m) ** 2 for v in vals) / (k - 1)) if k > 1 else float("nan")
    return m, sd / math.sqrt(k) if k > 1 else float("nan"), sd, k


def cell(vals: list[float], ref: float | None = None) -> str:
    if not vals:
        return "–"
    m, se, sd, k = stat(vals)
    s = f"{m:.3f} ± {se:.3f} (sd {sd:.3f})"
    if ref is not None:
        s += f" [{m - ref:+.3f}]"
    return s


def table(rows, keys, labels, group=("variant", "n"), title="", refs=None):
    by = defaultdict(lambda: defaultdict(list))
    for r in rows:
        g = tuple(r.get(x) for x in group)
        by[(r["regime"], g)]["_seeds"].append(r["seed"])
        for k in keys:
            if k in r["metrics"]:
                by[(r["regime"], g)][k].append(r["metrics"][k])
    out = [f"### {title}", ""]
    out.append("| regime | " + " | ".join(group) + " | seeds | " + " | ".join(labels) + " |")
    out.append("|---" * (2 + len(group) + len(labels)) + "|")

    def order(item):
        (regime, g), _ = item
        return (REG.index(regime), [(-1 if v is None else v) if v != "x" else 0 for v in g])

    for (regime, g), d in sorted(by.items(), key=lambda it: (REG.index(it[0][0]), str(it[0][1]))):
        ref = (refs or {}).get(regime, {})
        cells = [cell(d[k], ref.get(k)) for k in keys]
        gs = " | ".join("–" if v is None else str(v) for v in g)
        out.append(f"| {regime} | {gs} | {len(d['_seeds'])} | " + " | ".join(cells) + " |")
    out.append("")
    return "\n".join(out)


def sort_n(rows):
    return sorted(rows, key=lambda r: (REG.index(r["regime"]), r.get("variant", ""), r["n"] or 0))


def main() -> None:
    d = Path(sys.argv[1])
    print("# Boundary-layer-top interface study (decision 0028)\n")
    print(
        "Mean ± standard error over seeds, seed standard deviation in brackets; "
        "20,000 particles per seed, 60 s outer step, records every 15 min from 2 to 6 h. "
        "1 = the exact answer for a uniform column. Source of each run in its JSON line "
        "(`src`).\n"
    )
    o2 = sort_n(load(d / "open2h.jsonl"))
    keys = ["below200", "below50", "below10", "above10", "above50", "above200", "frac_below_h"]
    labels = [
        "h−200…h−50",
        "h−50…h",
        "h−10…h",
        "h…h+10",
        "h…h+50",
        "h+50…h+200",
        "share ≤ h (÷0.5)",
    ]
    print(table(o2, keys, labels, title="Open column, uniform in [0, 2h], lid at 2h: this branch"))
    print(
        table(
            o2,
            ["lo5", "z40"],
            ["lowest 5% of h", "0–40 m"],
            title="Open column, uniform in [0, 2h]: near-surface occupancy (÷ uniform share)",
        )
    )
    for name, title in (
        ("verify_pr_open2h.jsonl", "Open column, uniform in [0, 2h]: PR #69 head 0de5f94"),
        ("verify_main_open2h.jsonl", "Open column, uniform in [0, 2h]: main 20db902"),
        ("mech_open2h.jsonl", "Open column, uniform in [0, 2h]: one half of the interface"),
    ):
        rows = sort_n(load(d / name))
        if rows:
            print(table(rows, keys, labels, title=title))
    dl = d / "diffusion_limit.json"
    refs = {}
    if dl.exists():
        lim = json.loads(dl.read_text())
        refs = {r: {"frac_bl": v["frac_bl_2_6h"], "frac_bl_6h": v["frac_bl_6h"]} for r, v in lim.items()}
    ob = sort_n(load(d / "openbl.jsonl"))
    print(
        table(
            ob,
            ["frac_bl", "frac_bl_6h", "lo5_bl", "z40_bl"],
            ["BL share 2–6 h", "BL share at 6 h", "lowest 5% (in BL)", "0–40 m (in BL)"],
            title="Open top, start uniform in [0, h]: BL retention [difference from the diffusion limit]",
            refs=refs,
        )
    )
    if refs:
        print("Diffusion limit (`interface_diffusion_limit.py`, `diffusion_limit.json`):\n")
        for r, v in refs.items():
            print(f"- {r}: 2–6 h mean {v['frac_bl']:.4f}, 6 h {v['frac_bl_6h']:.4f}")
        print()
    cl = sort_n(load(d / "closed.jsonl"))
    print(
        table(
            cl,
            ["lo5", "z40"],
            ["lowest 5% of h", "0–40 m"],
            title="Closed column (lid at h every sub-step, crossing rule off), start uniform in [0, h]",
        )
    )
    hv = sort_n(load(d / "hvar.jsonl"))
    if hv:
        print(
            table(
                hv,
                keys + ["lo5", "z40"],
                labels + ["lowest 5% of h0", "0–40 m"],
                title="Moving h (0.75–1.25 h0, 2-h period), uniform in [0, 2h0], bins relative to h(t)",
            )
        )


if __name__ == "__main__":
    main()
