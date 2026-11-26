"""Calibrate tests/test_bl_top_well_mixed.py: run its exact procedure for several
seeds with the HannaScheme on the PYTHONPATH (this branch, PR #69's head or main)
and append the four band occupancies per run to ``--out`` as JSON lines.

    PYTHONPATH=<src>:tests python scripts/validation/wm_near_surface/bl_top_gate_calibration.py \
        --regime convective --seed 1 --label branch --out gate.jsonl \
        [--tier fast|full] [--n 8] [--particles 20000]

``--tier`` picks the test's design: ``fast`` is the default-suite gate (per-regime
particles, run length, first record and tolerances, ``FAST``), ``full`` the slow
full-strength one (``FULL``, the default). ``--particles`` overrides the tier's
count. Each line records the jump across h, (h..h+50) - (h-50..h), the design,
and the checks the run fails at the tier's tolerances.
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
    ap.add_argument("--tier", choices=("fast", "full"), default="full")
    ap.add_argument("--n", type=int, default=None, help="n_substeps (default: the scheme's)")
    ap.add_argument("--particles", type=int, default=None, help="default: the tier's")
    ap.add_argument("--label", required=True)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    design = gate.FAST[a.regime] if a.tier == "fast" else gate.FULL
    if a.particles is not None:
        design = design._replace(particles=a.particles)
    kw = dict(meander_enabled=False, static_substeps=False)
    if a.n is not None:
        kw["n_substeps"] = a.n
    t0 = time.time()
    bands = gate.run_column(a.regime, HannaScheme(**kw), design, seed=a.seed)
    jump = bands["h..h+50"] - bands["h-50..h"]
    rec = dict(
        label=a.label, regime=a.regime, seed=a.seed, n=a.n, tier=a.tier,
        src=os.path.dirname(hanna_mod.__file__), wall_s=round(time.time() - t0, 1),
        bands=bands, jump=jump, start="stratified", particles=design.particles,
        design=design._asdict(), records=dict(from_step=design.record_from_step, every=gate.RECORD_EVERY),
        fails=gate.failures(bands, design),
    )  # fmt: skip
    print(json.dumps(rec))
    with open(a.out, "a") as f:
        f.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
