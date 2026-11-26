"""Candidate table from the rowmat JSONs: per (variant, floor, n, particles), the
seed-mean of the pooled down/up ratio (row 10 with rows 6-9), of 10->9, of row
11 with rows 6-9, and of row 9 with rows 6-8 (both below h), each with the mean
per-seed standard error / sqrt(n_seeds); the per-seed pooled values; and the
exchange rate (residence in rows 6-9 of particles started in row 10).
Markdown on stdout. Usage: table.py ROWMAT_DIR"""

import glob
import json
import re
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from asym import ratios  # noqa: E402

groups = defaultdict(list)
for p in sorted(glob.glob(sys.argv[1] + "/fwd_n*.json")):
    with open(p) as f:
        d = json.load(f)
    v = d["variant"]
    if "mode=" not in v:
        v = "mode=fd," + v
    v = re.sub(r",?floor=[0-9.e-]+", "", v).strip(",")
    groups[(v, d["sigma_min"], d["nsub"], d["nper"])].append(ratios(d))


def agg(rs, k):
    x = np.array([r[k][0] for r in rs if k in r])
    e = np.array([r[k][1] for r in rs if k in r])
    if len(x) < len(rs):
        return "n/a"
    return f"{x.mean():.3f} ± {np.sqrt((e**2).mean() / len(x)):.3f}"


print(
    "| variant | floor | n | particles/row | seeds | pooled 10↔BL | 10↔9 | 11↔BL | 9↔6-8 "
    "| per-seed pooled | exchange |"
)
print("|---|---|---|---|---|---|---|---|---|---|---|")
for (v, fl, n, npr), rs in sorted(groups.items(), key=lambda k: (k[0][1], k[0][0], k[0][2])):
    px = ", ".join(f"{r['pooled'][0]:.3f}" for r in rs)
    down = np.mean([r["down_s"][0] for r in rs])
    print(
        f"| {v} | {fl:g} | {n} | {npr} | {len(rs)} | {agg(rs, 'pooled')} | {agg(rs, '10->9')} "
        f"| {agg(rs, '11pooled')} | {agg(rs, 'below')} | {px} | {down:.0f} |"
    )
