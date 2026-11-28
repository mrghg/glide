"""Calibrate the transfer-symmetry gate of tests/test_sigma_floor_gradient.py
(decision 0030): the downward/upward exchange ratio across a stable h in the
thin-floor column (h 400 m, u* 0.45 m s⁻¹, H −60 W m⁻², floor layer 0.68 m),
for production and for the stencil-only gradient before decision 0030
(``_fd_drift_gradient_only``, the code path of main d19755d), over seeds.

Usage (CPU):
  PYTHONPATH=src python scripts/validation/sigma_floor/transfer_gate_calibration.py \\
      --seeds 0-9 --n-substeps 16 --n-per-row 10000 --variant production --out gate.jsonl
Appends one JSON line per seed: variant, seed, n_substeps, n_per_row, floor,
ratio, se, seconds, lpdm path."""

import argparse
import json
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "tests"))

from test_sigma_floor_gradient import transfer_ratio  # noqa: E402

import lpdm  # noqa: E402
import lpdm.turbulence.hanna as H  # noqa: E402
from lpdm.turbulence import HannaScheme  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0-9")
    ap.add_argument("--n-substeps", type=int, default=16)
    ap.add_argument("--n-per-row", type=int, default=10_000)
    ap.add_argument("--variant", choices=["production", "fd_only"], default="production")
    ap.add_argument("--floor", type=float, default=None)
    ap.add_argument("--column", default="stable_thin_floor")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    if a.floor is not None:
        H.SIGMA_MIN_M_S = a.floor
    lo, _, hi = a.seeds.partition("-")
    for seed in range(int(lo), int(hi or lo) + 1):
        scheme = HannaScheme(n_substeps=a.n_substeps, meander_enabled=False, static_substeps=False)
        scheme._fd_drift_gradient_only = a.variant == "fd_only"
        t0 = time.time()
        r, se = transfer_ratio(scheme, a.n_per_row, seed, a.column)
        rec = dict(variant=a.variant, seed=seed, n_substeps=a.n_substeps, n_per_row=a.n_per_row,
                   floor=H.SIGMA_MIN_M_S, column=a.column, ratio=r, se=se,
                   seconds=round(time.time() - t0, 1), lpdm=lpdm.__file__)  # fmt: skip
        with open(a.out, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
