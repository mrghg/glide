"""Decision 0031: compare two outputs of ``trigger_counts.py`` (before and after).

    .venv/bin/python scripts/validation/cape/compare_trigger_counts.py before.npz after.npz

Per window and in total: columns, columns firing before and after, columns that
stop firing and that start firing, the cloud-base mass flux summed over columns
with a cos(latitude) area weight, before and after, and the largest M_b change
among columns that fire under both. Also for a sub-box around the UK and
Ireland release sites (lon -15..5, lat 48..60), where the reference config's
particles spend their first hours.
"""

from __future__ import annotations

import sys

import numpy as np

SUB_BOX = (-15.0, 5.0, 48.0, 60.0)


def summarise(before: np.lib.npyio.NpzFile, after: np.lib.npyio.NpzFile, box) -> dict:
    keys = sorted({k.split("/")[0] for k in before.files})
    rows = []
    for key in keys:
        lat, lon = before[f"{key}/lat"], before[f"{key}/lon"]
        assert np.array_equal(lat, after[f"{key}/lat"])
        sel = np.ones(lat.shape, dtype=bool)
        if box is not None:
            sel = (lon >= box[0]) & (lon <= box[1]) & (lat >= box[2]) & (lat <= box[3])
        fb, fa = before[f"{key}/fires"][sel], after[f"{key}/fires"][sel]
        mb, ma = before[f"{key}/m_b"][sel], after[f"{key}/m_b"][sel]
        w = np.cos(np.deg2rad(lat[sel]))
        both = fb & fa
        for name in ("i_lcl", "i_lnb", "dtv", "depth"):
            assert np.array_equal(before[f"{key}/{name}"], after[f"{key}/{name}"]), name
        rows.append(
            {
                "window": key,
                "columns": int(sel.sum()),
                "fire_before": int(fb.sum()),
                "fire_after": int(fa.sum()),
                "stop": int((fb & ~fa).sum()),
                "start": int((~fb & fa).sum()),
                "mb_before": float((mb * w).sum()),
                "mb_after": float((ma * w).sum()),
                "max_dmb_both": float(np.abs(ma[both] - mb[both]).max()) if both.any() else 0.0,
                "cape_before_stop": before[f"{key}/cape"][sel][fb & ~fa],
                "cape_after_stop": after[f"{key}/cape"][sel][fb & ~fa],
                "cape_changed": int(
                    (before[f"{key}/cape"][sel] != after[f"{key}/cape"][sel]).sum()
                ),
            }
        )
    return rows


def show(rows: list[dict], title: str) -> None:
    print(f"\n## {title}\n")
    print(
        "| window | columns | CAPE changed | fire before | fire after | stop | start "
        "| ΣM_b·cosφ before | after | change | max ΔM_b (fire both) |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    tot = dict.fromkeys(("columns", "fire_before", "fire_after", "stop", "start", "cape_changed"), 0)
    mb_b = mb_a = 0.0
    max_d = 0.0
    cape_b, cape_a = [], []
    for r in rows:
        rel = (r["mb_after"] / r["mb_before"] - 1.0) * 100 if r["mb_before"] > 0 else float("nan")
        print(
            f"| {r['window']} | {r['columns']} | {r['cape_changed']} | {r['fire_before']} "
            f"| {r['fire_after']} | {r['stop']} | {r['start']} | {r['mb_before']:.4g} "
            f"| {r['mb_after']:.4g} | {rel:+.2f}% | {r['max_dmb_both']:.3g} |"
        )
        for k in tot:
            tot[k] += r[k]
        mb_b += r["mb_before"]
        mb_a += r["mb_after"]
        max_d = max(max_d, r["max_dmb_both"])
        cape_b.append(r["cape_before_stop"])
        cape_a.append(r["cape_after_stop"])
    rel = (mb_a / mb_b - 1.0) * 100 if mb_b > 0 else float("nan")
    print(
        f"| **all** | {tot['columns']} | {tot['cape_changed']} | {tot['fire_before']} "
        f"| {tot['fire_after']} | {tot['stop']} | {tot['start']} | {mb_b:.4g} | {mb_a:.4g} "
        f"| {rel:+.2f}% | {max_d:.3g} |"
    )
    cb, ca = np.concatenate(cape_b), np.concatenate(cape_a)
    if cb.size:
        print(
            f"\nColumns that stop firing: {cb.size} "
            f"({100 * cb.size / max(tot['fire_before'], 1):.2f}% of firing columns). "
            f"CAPE before: median {np.median(cb):.0f}, range {cb.min():.0f}-{cb.max():.0f} J/kg; "
            f"after: median {np.median(ca):.0f}, range {ca.min():.0f}-{ca.max():.0f} J/kg."
        )


def main(before_path: str, after_path: str) -> None:
    before, after = np.load(before_path), np.load(after_path)
    show(summarise(before, after, None), "Whole met domain")
    show(summarise(before, after, SUB_BOX), f"Sub-box lon {SUB_BOX[:2]}, lat {SUB_BOX[2:]}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
