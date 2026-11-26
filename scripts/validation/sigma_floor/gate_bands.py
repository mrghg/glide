"""Occupancy numbers behind the two well-mixed gates (the tests assert on them
but print them only on failure): tests/test_bl_top_well_mixed.py run_column
(bands h-100..h+100) and tests/test_near_surface_well_mixed.py (lowest 5% and
0-40 m), with the tests' own schemes, seeds and counts. Env GLIDE_WT = tree.
Usage: gate_bands.py REGIME [REGIME ...]"""

import json
import os
import sys

WT = os.environ["GLIDE_WT"]
sys.path.insert(0, WT + "/tests")
import lpdm  # noqa: E402

assert lpdm.__file__.startswith(WT), lpdm.__file__
import test_bl_top_well_mixed as B  # noqa: E402

from lpdm.turbulence import HannaScheme  # noqa: E402

for regime in sys.argv[1:]:
    bands = B.run_column(regime, HannaScheme(meander_enabled=False, static_substeps=False))
    bands["jump"] = bands["h..h+50"] - bands["h-50..h"]
    print(json.dumps({"wt": WT, "gate": "bl_top", "regime": regime, **bands}), flush=True)
