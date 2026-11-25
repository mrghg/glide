"""Calibrate tests/test_bl_top_well_mixed.py: run its exact procedure for several
seeds with the HannaScheme on the PYTHONPATH (this branch, PR #69's head or main)
and append the four band occupancies per run to ``--out`` as JSON lines.

    PYTHONPATH=<src>:tests python scripts/validation/wm_near_surface/bl_top_gate_calibration.py \
        --regime convective --seed 1 --label branch --out gate.jsonl [--n 8] [--particles 20000]

``--particles`` defaults to the test's ``N_PARTICLES``. Each line also records the
jump across h, (h..h+50) - (h-50..h), and the record schedule.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import test_bl_top_well_mixed as gate
import torch

from lpdm.turbulence import HannaScheme
from lpdm.turbulence import hanna as hanna_mod


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--n", type=int, default=None, help="n_substeps (default: the scheme's)")
    ap.add_argument("--particles", type=int, default=gate.N_PARTICLES)
    ap.add_argument("--label", required=True)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    kw = dict(meander_enabled=False, static_substeps=False)
    if a.n is not None:
        kw["n_substeps"] = a.n
    t0 = time.time()
    bands = gate.run_column(a.regime, HannaScheme(**kw), seed=a.seed, n=a.particles)
    jump = bands["h..h+50"] - bands["h-50..h"]
    rec = dict(
        label=a.label, regime=a.regime, seed=a.seed, n=a.n,
        src=os.path.dirname(hanna_mod.__file__), wall_s=round(time.time() - t0, 1),
        bands=bands, jump=jump, start="stratified", particles=a.particles,
        records=dict(from_step=gate.RECORD_FROM_STEP, every=gate.RECORD_EVERY),
    )  # fmt: skip
    print(json.dumps(rec))
    with open(a.out, "a") as f:
        f.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
