"""Compare bitid.py outputs against a reference. A particle is 'away from the
floor' if its height after every outer step stayed below z_k - MARGIN (z_k: where
the stable floor starts to bind, h - 1.23 m); convective has no floor layer.
Usage: bitid_compare.py DIR"""

import sys

import numpy as np

d = sys.argv[1]
MARGIN = 20.0
for regime in ("stable", "convective"):
    ref = np.load(f"{d}/{regime}_a6head.npz")
    zr = ref["z"]
    h = float(ref["h"])
    zk = h - 1.231 if regime == "stable" else np.inf
    away = zr.max(0) < zk - MARGIN
    for m in ("fd", "kink", "analytic", "hybrid"):
        x = np.load(f"{d}/{regime}_{m}.npz")
        diff = (x["z"] != zr).any(0) | (x["w"] != ref["w"])
        print(
            f"{regime:10s} {m:9s} particles differing from A6 head: {int(diff.sum()):6d}/{diff.size}"
            f"; of those away from the floor ({int(away.sum())}): {int((diff & away).sum())}"
            f"; max |dz| {float(np.abs(x['z'] - zr).max()):.3g} m"
        )
