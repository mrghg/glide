"""Decision 6 study: paired-seed footprint comparison of the two triggers for one case.

    .venv/bin/python scripts/validation/conv/trigger/compare_trigger_footprints.py <case dir>

<case dir> holds glide_s<seed>/ and fpcb_s<seed>/ (footprints.zarr), their .log and
.exposure.json (run_case.sh). For each seed the FLEXPART-trigger run is compared
with the GLIDE-trigger run of the same seed (paired); the seed spread is the
spread of the paired changes, and the Monte Carlo noise is the spread of the
GLIDE runs between seeds. Reported:

- exposure: particle-events (active particle x convection event) in columns
  that fire under each trigger, and in columns that fire under FLEXPART's but
  not GLIDE's ("new"), from the exposure log of every run;
- particle-events displaced by convection (the scheme's own log), over all
  particle-events (the log omits events with no firing column);
- 0-40 m and total footprint, relative change per seed, mean and range;
- 0-40 m footprint by age bin (time_ago), relative change per seed;
- footprint per height bin (density-weighted residence), relative change, and
  the share of the total in each bin;
- cell-level 0-40 m log10 ratio spread (cells above 1e-3 of the max).
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import xarray as xr


def load(d: Path) -> tuple[np.ndarray, xr.Dataset]:
    ds = xr.open_zarr(d / "footprints.zarr")
    fp = ds["footprint"].values.astype(np.float64).sum(axis=0)  # [age, z, y, x]
    return fp, ds


def displaced(log: Path) -> tuple[int, int]:
    ev = re.findall(r"Displaced (\d+) / (\d+)", log.read_text())
    return sum(int(a) for a, _ in ev), sum(int(b) for _, b in ev)


def pct(a) -> str:
    a = np.asarray(a) * 100.0
    return f"{a.mean():+7.2f}% [{a.min():+7.2f}, {a.max():+7.2f}]"


def main() -> None:
    case = Path(sys.argv[1])
    seeds = sorted(
        int(p.name.split("_s")[1])
        for p in case.glob("glide_s*")
        if p.is_dir() and (case / f"fpcb_s{p.name.split('_s')[1]}" / "footprints.zarr").exists()
    )
    print(f"=== {case}: paired seeds {seeds}")
    G, F = {}, {}
    for s in seeds:
        G[s], ds = load(case / f"glide_s{s}")
        F[s], _ = load(case / f"fpcb_s{s}")
    zb = list(zip(ds["z_bottom_m"].values, ds["z_top_m"].values, strict=True))
    ages = ds["time_ago"].values
    print(f"z bins {[f'{a:.0f}-{b:.0f}' for a, b in zb]}; {len(ages)} age bins")

    print("\n-- Exposure (particle-events = active particles x convection events)")
    for tr in ("glide", "fpcb"):
        for s in seeds:
            ex = json.loads((case / f"{tr}_s{s}.exposure.json").read_text())["totals"]
            d, a = displaced(case / f"{tr}_s{s}.log")
            print(
                f"   {tr:5s} s{s}: {ex['active']} particle-events; in GLIDE-firing columns {ex['in_glide']} "
                f"({ex['in_glide'] / ex['active']:.2%}), in FLEXPART-trigger columns {ex['in_fp']} "
                f"({ex['in_fp'] / ex['active']:.2%}), in NEW columns {ex['in_new']} ({ex['in_new'] / ex['active']:.2%}; "
                f"below cloud top {ex['in_new_below_top']}), GLIDE-only {ex['in_glide_not_fp']}; "
                f"displaced by convection {d} ({d / ex['active']:.2%} of all particle-events)"
            )

    def rel(f, g, sel):
        return f[sel].sum() / g[sel].sum() - 1.0

    s0 = seeds[0]
    noise = [G[s].sum() for s in seeds]
    print(
        "\n-- Footprint, paired change FLEXPART-trigger vs GLIDE-trigger: mean [min, max] over seeds"
    )
    z0 = (slice(None), 0)
    print(f"   0-40 m : {pct([rel(F[s], G[s], z0) for s in seeds])}")
    print(f"   total  : {pct([rel(F[s], G[s], slice(None)) for s in seeds])}")
    print(
        "   seed noise (GLIDE trigger, each seed vs seed "
        f"{s0}): 0-40 m {pct([rel(G[s], G[s0], z0) for s in seeds[1:]])}, total "
        f"{pct([G[s].sum() / G[s0].sum() - 1 for s in seeds[1:]])}"
    )
    print(
        f"   seed noise (FLEXPART trigger): 0-40 m {pct([rel(F[s], F[s0], z0) for s in seeds[1:]])}"
    )
    del noise
    print(
        "\n-- 0-40 m footprint by age (time_ago bin), paired change; share of the GLIDE 0-40 m total"
    )
    for i, a in enumerate(ages):
        sel = (i, 0)
        share = np.mean([G[s][sel].sum() / G[s][:, 0].sum() for s in seeds])
        noise_a = [rel(G[s], G[s0], sel) for s in seeds[1:]]
        print(
            f"   age bin {int(a):2d}: share {share:6.1%}  change {pct([rel(F[s], G[s], sel) for s in seeds])}"
            f"   (seed noise {pct(noise_a)})"
        )
    print(
        "\n-- Footprint by height bin (all ages), paired change; share of the total (GLIDE / FLEXPART trigger)"
    )
    for j, (a, b) in enumerate(zb):
        sel = (slice(None), j)
        sg = np.mean([G[s][sel].sum() / G[s].sum() for s in seeds])
        sf = np.mean([F[s][sel].sum() / F[s].sum() for s in seeds])
        print(
            f"   {a:6.0f}-{b:6.0f} m: share {sg:7.3%} / {sf:7.3%}  change {pct([rel(F[s], G[s], sel) for s in seeds])}"
            f"   (seed noise {pct([rel(G[s], G[s0], sel) for s in seeds[1:]])})"
        )
    print("\n-- Cell-level 0-40 m log10 ratio (cells > 1e-3 of the max), p5 / p50 / p95")
    for s in seeds:
        g, f = G[s][:, 0].sum(0), F[s][:, 0].sum(0)
        m = g > 1e-3 * g.max()
        lr = np.log10(np.maximum(f[m], 1e-30) / g[m])
        print(f"   s{s} trigger: {np.percentile(lr, [5, 50, 95]).round(3)} over {m.sum()} cells")
    for s in seeds[1:]:
        g, f = G[s0][:, 0].sum(0), G[s][:, 0].sum(0)
        m = g > 1e-3 * g.max()
        lr = np.log10(np.maximum(f[m], 1e-30) / g[m])
        print(f"   seed noise s{s} vs s{s0}: {np.percentile(lr, [5, 50, 95]).round(3)}")


if __name__ == "__main__":
    main()
