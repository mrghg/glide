"""Decision 6 study: summarise trigger_columns.py output against FLEXPART's CONVECT.

    .venv/bin/python scripts/validation/conv/trigger/summarize_columns.py <jan|jul> <columns.npz>

FLEXPART: task CONV's driver outputs (outputs/agents/CONV/flexpart/*.npz, FLEXPART 11
`convect` on the stores' native L137 columns, 12 calls of 900 s carrying CBMF; the
last 4 calls = 1 h give E, f3, f6) and the review's 1-s baselines
(outputs/agents/CONV/review/fp_base_*.npz), on the same horizontal columns.
"FLEXPART, no memory" is its FIRST call (cold start, CBMF = 0 before it), which is
exactly the memory-less trigger; "FLEXPART, steady" is the last call.

All sums are cos(lat)-weighted. "lifted above 3 km" is the sub-cloud air above 3 km
AGL after 1 h minus what was there at the start (LCL above 3 km), as a fraction of
the domain's sub-cloud air.
"""

from __future__ import annotations

import sys

import numpy as np

B = "/home/u6za/chxmr.u6za/code/glide/outputs/agents/CONV/"
FP = {
    "jan": (
        ["flexpart/jan_jan.npz", "flexpart/jan_dec.npz"],
        ["review/fp_base_jan_jan.npz", "review/fp_base_jan_dec.npz"],
    ),
    "jul": (["flexpart/jul.npz"], ["review/fp_base_jul.npz"]),
}
TRIG = ("glide", "fpcb", "fpcb0")
LABEL = {
    "glide": "GLIDE trigger (default)",
    "fpcb": "FLEXPART cloud base, depth >= 500 m",
    "fpcb0": "FLEXPART cloud base, no depth check",
    "fp1": "FLEXPART 11, no memory (1st call)",
    "fpS": "FLEXPART 11, steady (12th call)",
}


def load(ps):
    d = {}
    for p in ps:
        z = np.load(B + p)
        d.update({k: z[k] for k in z.files})
    return d


def main() -> None:
    season, path = sys.argv[1], sys.argv[2]
    g = np.load(path)
    f, fb = load(FP[season][0]), load(FP[season][1])
    levels = sorted({int(k.split("/")[0][1:]) for k in g.files})
    hours = sorted({k.split("/")[1] for k in g.files if k.startswith("L40/")})
    print(f"=== {season}: {len(hours)} hours; levels {levels}; file {path}")
    ident = [bool(g[k]) for k in g.files if k.endswith("glide_identical")]
    print(
        f"glide trigger identical to the earlier CONV run (grid/*_lcl1.npz) in {sum(ident)} of {len(ident)} level-hours"
    )
    T: dict = {}  # (nlev, trig) -> dict of sums

    def acc(key, name, val):
        T.setdefault(key, {}).setdefault(name, 0.0)
        T[key][name] += float(val)

    med: dict = {}
    for L in levels:
        for h in hours:
            k = f"L{L}/{h}"
            cos = g[f"{k}/cosphi"]
            src = g[f"{k}/src"]
            idx = g[f"{k}/idx"]
            dt = float(g[f"{k}/interval_s"])
            n = cos.size
            E = np.zeros(n)
            d3 = np.zeros(n)
            d6 = np.zeros(n)
            E[idx] = g[f"{k}/E"]
            d3[idx] = (g[f"{k}/f3"] - g[f"{k}/f3_0"]) * src[idx]
            d6[idx] = (g[f"{k}/f6"] - g[f"{k}/f6_0"]) * src[idx]
            mb = g[f"{k}/mb_all"]
            depth = g[f"{k}/depth"]
            cb1 = f[f"{h}/900:12:8/cbmf"][:, 0].astype(float)
            cbS = f[f"{h}/900:12:8/cbmf"][:, -1].astype(float)
            assert cb1.size == n, (cb1.size, n)
            for t in TRIG:
                m = g[f"{k}/fires_{t}"]
                key = (L, t)
                acc(key, "n", n)
                acc(key, "fires", m.sum())
                acc(key, "cos", cos.sum())
                acc(key, "fires_cos", cos[m].sum())
                acc(key, "mb", (mb * cos)[m].sum())
                acc(key, "E", (E * cos)[m].sum() / dt * 3600.0)
                acc(key, "M3", (d3 * cos)[m].sum())
                acc(key, "M6", (d6 * cos)[m].sum())
                acc(key, "src", (src * cos).sum())
                acc(key, "and_fp1", (m & (cb1 > 0)).sum())
                acc(key, "and_fpS", (m & (cbS > 0)).sum())
                acc(key, "shallow", (m & (depth < 500.0)).sum())
                acc(key, "E_shallow", (E * cos)[m & (depth < 500.0)].sum() / dt * 3600.0)
                if L == 40:
                    sel = np.isin(idx, np.nonzero(m)[0])
                    med.setdefault((t, "mb"), []).append(mb[m])
                    med.setdefault((t, "E/h"), []).append(g[f"{k}/E"][sel] / dt * 3600.0)
                    med.setdefault((t, "f3"), []).append((g[f"{k}/f3"] - g[f"{k}/f3_0"])[sel])
                    med.setdefault((t, "f6"), []).append((g[f"{k}/f6"] - g[f"{k}/f6_0"])[sel])
                    med.setdefault((t, "depth"), []).append(depth[m])
            if L == 40:
                fsrc = f[f"{h}/900:12:8/src"]
                for t, cb in (("fp1", cb1), ("fpS", cbS)):
                    m = cb > 0
                    key = (40, t)
                    acc(key, "n", n)
                    acc(key, "fires", m.sum())
                    acc(key, "cos", cos.sum())
                    acc(key, "fires_cos", cos[m].sum())
                    acc(key, "mb", (cb * cos)[m].sum())
                    if t == "fpS":
                        e = f[f"{h}/900:12:8/E"] - fb[f"{h}/1:1:0/E"]
                        f3 = f[f"{h}/900:12:8/f3"] - fb[f"{h}/1:1:0/f3"]
                        f6 = f[f"{h}/900:12:8/f6"] - fb[f"{h}/1:1:0/f6"]
                        acc(key, "E", (e * cos).sum())
                        acc(key, "M3", (f3 * fsrc * cos).sum())
                        acc(key, "M6", (f6 * fsrc * cos).sum())
                        acc(key, "src", (fsrc * cos).sum())
                        med.setdefault((t, "mb"), []).append(cb[m])
                        med.setdefault((t, "E/h"), []).append(e[m])
                        med.setdefault((t, "f3"), []).append(f3[m])
                        med.setdefault((t, "f6"), []).append(f6[m])
                    else:
                        med.setdefault((t, "mb"), []).append(cb[m])
                    acc(key, "and_fp1", (m & (cb1 > 0)).sum())
                    acc(key, "and_fpS", (m & (cbS > 0)).sum())

    fp1n = T[(40, "fp1")]["fires"]
    print(
        "\n-- Firing and transport, 40 levels (FLEXPART on L137, same columns). Fractions of column-hours."
    )
    print(
        "   E/h: sub-cloud air moved into the cloud layers in 1 h; lift>3/6 km: fraction of the domain's sub-cloud air"
    )
    print(
        "   newly above 3 / 6 km after 1 h; overlap: share of this trigger's firing column-hours where FLEXPART"
    )
    print(
        "   fires (1st call / steady), and share of FLEXPART's (1st call) that this trigger catches."
    )
    hdr = f"   {'trigger':42s} {'fires':>7s} {'cos-wt':>7s} {'sum Mb cos':>11s} {'sum E/h cos':>12s} {'lift>3km':>9s} {'lift>6km':>9s} {'in FP1':>7s} {'in FPS':>7s} {'of FP1':>7s}"
    print(hdr)
    for t in (*TRIG, "fp1", "fpS"):
        s = T[(40, t)]
        e = f"{s['E']:12.4g}" if "E" in s else f"{'':>12s}"
        m3 = f"{s['M3'] / s['src']:9.2e}" if "M3" in s else f"{'':>9s}"
        m6 = f"{s['M6'] / s['src']:9.2e}" if "M6" in s else f"{'':>9s}"
        print(
            f"   {LABEL[t]:42s} {s['fires'] / s['n']:7.2%} {s['fires_cos'] / s['cos']:7.2%} {s['mb']:11.4g} {e} {m3} {m6}"
            f" {s['and_fp1'] / max(s['fires'], 1):7.1%} {s['and_fpS'] / max(s['fires'], 1):7.1%} {s['and_fp1'] / fp1n:7.1%}"
        )
    print("\n-- Medians (p10 / p50 / p90) in firing column-hours, 40 levels")
    for (t, q), v in sorted(med.items(), key=lambda x: (list(LABEL).index(x[0][0]), x[0][1])):
        v = np.concatenate(v)
        if v.size:
            p = np.percentile(v, [10, 50, 90])
            print(f"   {LABEL[t]:42s} {q:6s} {p[0]:10.4g} {p[1]:10.4g} {p[2]:10.4g}   (n={v.size})")
    print(
        "\n-- Shallow clouds (GLIDE first-LNB depth < 500 m) among firing column-hours, and their share of E/h"
    )
    for L in levels:
        for t in TRIG:
            s = T[(L, t)]
            print(
                f"   L{L:<3d} {LABEL[t]:42s} {s['shallow'] / max(s['fires'], 1):7.1%} of fires, {s['E_shallow'] / max(s['E'], 1e-30):7.1%} of E"
            )
    print("\n-- Grid dependence: domain totals on each grid relative to 40 levels")
    print(
        f"   {'trigger':42s} {'L':>4s} {'fires':>8s} {'fires/L40':>9s} {'Mb/L40':>7s} {'E/L40':>7s} {'M3/L40':>7s} {'M6/L40':>7s} {'fires %':>8s}"
    )
    for t in TRIG:
        r = T[(40, t)]
        for L in levels:
            s = T[(L, t)]
            print(
                f"   {LABEL[t]:42s} {L:4d} {int(s['fires']):8d} {s['fires'] / r['fires']:9.3f} {s['mb'] / r['mb']:7.3f}"
                f" {s['E'] / r['E']:7.3f} {s['M3'] / r['M3']:7.3f} {s['M6'] / max(r['M6'], 1e-30):7.3f} {s['fires'] / s['n']:8.2%}"
            )
    print("\n-- Domain totals per trigger and grid (cos-weighted sums over all hours)")
    for L in levels:
        for t in TRIG:
            s = T[(L, t)]
            print(
                f"   L{L:<3d} {LABEL[t]:42s} sum Mb {s['mb']:10.4g}  sum E/h {s['E']:10.4g}  lift>3km {s['M3'] / s['src']:9.2e}  lift>6km {s['M6'] / s['src']:9.2e}"
            )


if __name__ == "__main__":
    main()
