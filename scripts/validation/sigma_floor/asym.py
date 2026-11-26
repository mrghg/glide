"""Transfer symmetry of a row matrix (rowmat.py): M[a, b] / M[b, a] for a above h
(rows 10, 11) and b in the boundary layer (rows 6-9); 1 under detailed balance.
Also the pooled ratio sum_b M[10, b] / sum_b M[b, 10] over b = 6..9 and the
exchange rate itself (sum_b M[10, b], seconds per 2 h).
Usage: asym.py [--json OUT] FILE... (files of one variant are also pooled over
seeds by inverse-variance weighting when given with --pool TAG)."""

import argparse
import json

import numpy as np


def ratios(d):
    M, S = np.array(d["M"]), np.array(d["SE"])
    rows = d["rows"]
    ix = {r: i for i, r in enumerate(rows)}
    out = {}
    for a in (10, 11):
        if a not in ix:
            continue
        for b in (6, 7, 8, 9):
            if b not in ix:
                continue
            x, y = M[ix[a], b], M[ix[b], a]
            r = x / y
            out[f"{a}->{b}"] = (r, r * np.hypot(S[ix[a], b] / x, S[ix[b], a] / y))
    # Pooled pairs from M alone (the SE ignores the correlation between entries of
    # one start row, so it is approximate): "below" = row 9 (h-40..h) with rows
    # 6-8, both below h; "11pooled" = row 11 (h+40..h+80) with rows 6-9.
    for name, a_row, bs in (("below", 9, (6, 7, 8)), ("11pooled", 11, (6, 7, 8, 9))):
        if a_row in ix and all(b in ix for b in bs):
            x = sum(M[ix[a_row], b] for b in bs)
            y = sum(M[ix[b], a_row] for b in bs)
            ex = np.sqrt(sum(S[ix[a_row], b] ** 2 for b in bs))
            ey = np.sqrt(sum(S[ix[b], a_row] ** 2 for b in bs))
            if x > 0 and y > 0:
                out[name] = (x / y, x / y * np.hypot(ex / x, ey / y))
    if "BL4" in d and 10 in ix:
        down = d["BL4"][ix[10]]
        down_se = d["BL4_SE"][ix[10]]
        up = sum(M[ix[b], 10] for b in (6, 7, 8, 9))
        up_se = np.sqrt(sum(S[ix[b], 10] ** 2 for b in (6, 7, 8, 9)))
        r = down / up
        out["pooled"] = (r, r * np.hypot(down_se / down, up_se / up))
        out["down_s"] = (down, down_se)
        out["up_s"] = (up, up_se)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--json")
    a = ap.parse_args()
    res = {}
    for p in a.files:
        try:
            with open(p) as f:
                d = json.load(f)
        except FileNotFoundError:
            print(p, "missing")
            continue
        r = ratios(d)
        res[p] = {k: [float(v[0]), float(v[1])] for k, v in r.items()}
        keys = ["10->6", "10->7", "10->8", "10->9", "pooled"]
        print(
            f"{p.split('/')[-1]:44s} n={d['nsub']:3d} floor={d.get('sigma_min', 1e-3):g} "
            + " ".join(f"{k}={r[k][0]:.3f}±{r[k][1]:.3f}" for k in keys if k in r)
            + (f" | down={r['down_s'][0]:.1f}±{r['down_s'][1]:.1f}s" if "down_s" in r else "")
        )
    if a.json:
        with open(a.json, "w") as f:
            json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
