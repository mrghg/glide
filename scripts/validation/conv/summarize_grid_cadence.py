"""Task CONV: summarise `closure_grid_cadence.py` output (one or more .npz files).

Per grid: firing columns and trigger-check pass rates. Per grid and variant:
percentiles (0/10/50/90/100) over firing columns of M_b, the Courant number,
the transport per unit time E/Δt, the fractions of sub-cloud air above 3 and
6 km after Δt, and the cadence ratios; and the domain total Σ (E/Δt)·cos φ per
hour (kg m-2 s-1 summed over columns, area-weighted), whose ratio between grids
is the grid dependence of the transport including the trigger.
"""

from __future__ import annotations

import sys
from collections import defaultdict

import numpy as np


def pct(x):
    x = np.asarray(x, float)
    if x.size == 0:
        return "n/a"
    return " / ".join(f"{v:.3g}" for v in np.percentile(x, [0, 10, 50, 90, 100]))


def main() -> None:
    data = {}
    for f in sys.argv[1:]:
        z = np.load(f)
        data.update({k: z[k] for k in z.files})
    grids = sorted({k.split("/")[0] for k in data})
    hours = sorted({k.split("/")[1] for k in data})
    variants = sorted({k.split("/")[2] for k in data if k.count("/") == 3})
    totals = defaultdict(dict)  # (variant) -> grid -> list per hour
    print(f"hours: {' '.join(hours)}\n")
    for g in grids:
        nfire = ncol = 0
        chk = defaultdict(int)
        nlcl = 0
        dz = []
        dtv = []
        for h in hours:
            key = f"{g}/{h}"
            if f"{key}/fires" not in data:
                continue
            fires = data[f"{key}/fires"]
            ncol += fires.size
            nfire += int(fires.sum())
            lcl = data[f"{key}/chk_has_lcl"]
            nlcl += int(lcl.sum())
            for c in ("depth_pos", "cape", "dtv", "cdepth"):
                chk[c] += int((data[f"{key}/chk_{c}"] & lcl).sum())
            # columns passing every check except dtv: how often the dtv test decides
            others = (
                lcl
                & data[f"{key}/chk_depth_pos"]
                & data[f"{key}/chk_cape"]
                & data[f"{key}/chk_cdepth"]
            )
            chk["others_pass"] += int(others.sum())
            chk["others_pass_dtv_fail"] += int((others & ~data[f"{key}/chk_dtv"]).sum())
            dz.append(data[f"{key}/dz_test"][fires])
            dtv.append(data[f"{key}/dtv"][others])
        print(f"== {g}: {nfire} firing / {ncol} column-hours ({nfire / max(ncol, 1):.4%})")
        print(
            f"   pass among columns with an LCL ({nlcl}): "
            + ", ".join(
                f"{c} {chk[c] / max(nlcl, 1):.3f}" for c in ("depth_pos", "cape", "dtv", "cdepth")
            )
        )
        print(
            f"   pass all but dtv: {chk['others_pass']}, of which dtv fails: {chk['others_pass_dtv_fail']}"
            f" ({chk['others_pass_dtv_fail'] / max(chk['others_pass'], 1):.3f})"
        )
        print(f"   LCL+1 above LCL level (firing), m: {pct(np.concatenate(dz))}")
        dzl = [
            data[f"{g}/{h}/dz_test_lcl"][
                data[f"{g}/{h}/chk_has_lcl"]
                & data[f"{g}/{h}/chk_cape"]
                & data[f"{g}/{h}/chk_cdepth"]
            ]
            for h in hours
            if f"{g}/{h}/dz_test_lcl" in data
        ]
        if dzl:
            print(
                f"   LCL+1 above the LCL itself (columns passing CAPE and depth), m: {pct(np.concatenate(dzl))}"
            )
        print(f"   dtv(LCL+1) where the other checks pass, K: {pct(np.concatenate(dtv))}")
        for v in variants:
            acc = defaultdict(list)
            tot = []
            for h in hours:
                key = f"{g}/{h}"
                if f"{key}/{v}/E" not in data:
                    continue
                dt = float(data[f"{key}/interval_s"])
                cos = data[f"{key}/cosphi"][data[f"{key}/idx"]]
                for name in ("mb", "courant", "E", "f3", "f6", "E3x1", "E3"):
                    acc[name].append(data[f"{key}/{v}/{name}"])
                acc["dt"].append(np.full(cos.size, dt))
                tot.append(float((data[f"{key}/{v}/E"] / dt * cos).sum()))
            if not acc:
                continue
            a = {k: np.concatenate(x) for k, x in acc.items()}
            rate1 = a["E"] / a["dt"]
            rate3 = a["E3"] / (3 * a["dt"])
            totals[v][g] = tot
            print(f"   -- {v}")
            print(f"      M_b kg m-2 s-1:            {pct(a['mb'])}")
            print(f"      Courant (unscaled M_b dt): {pct(a['courant'])}")
            print(f"      E/dt kg m-2 s-1 (1 h):     {pct(rate1)}")
            print(
                f"      E/dt kg m-2 s-1 (3 h evt): {pct(rate3)}  ratio 3h/1h: {pct(rate3 / rate1)}"
            )
            print(f"      E3(1 event 3h)/E3(3 x 1h): {pct(a['E3'] / a['E3x1'])}")
            print(f"      frac >3 km, >6 km (1 h):   {pct(a['f3'])} | {pct(a['f6'])}")
            print(f"      domain sum (E/dt) cos, per hour: {' '.join(f'{t:.3g}' for t in tot)}")
        print()
    if len(grids) > 1:
        for ref in [g for g in grids if g.startswith("L40")]:
            print(
                f"== grid dependence: domain sum of (E/dt)·cos φ over all hours, relative to {ref}"
            )
            same = [g for g in grids if g[3:] == ref[3:]]
            for v in variants:
                row = []
                for g in same:
                    if g in totals[v] and ref in totals[v]:
                        row.append(f"{g} {sum(totals[v][g]) / max(sum(totals[v][ref]), 1e-30):.3f}")
                print(f"   {v:12s} " + ", ".join(row))


if __name__ == "__main__":
    main()
