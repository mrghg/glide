"""CPU cost of _vertical_coefficients, production vs the stencil-only switch,
for 200,000 particles spread over [0, 2h] in a stable and a convective column.
Single thread; median of 15 calls after 3 warm-up calls. Env: PYTHONPATH=<tree>/src."""

import sys
import time

import torch

tree = sys.argv[1]
sys.path.insert(0, tree + "/tests")
import lpdm  # noqa: E402

assert lpdm.__file__.startswith(tree), lpdm.__file__
from test_bl_top_well_mixed import _met  # noqa: E402
from test_sigma_floor_gradient import _col  # noqa: E402

from lpdm.turbulence import HannaScheme  # noqa: E402

torch.set_num_threads(1)
cols = {"stable": (400.0, 0.45, -60.0), "convective": (1500.0, 0.3, 200.0)}
for name, (blh, us, shf) in cols.items():
    met = _met(blh, us, shf)
    z = torch.rand(200_000) * 2 * blh
    for fd in (True, False, True, False):
        s = HannaScheme(static_substeps=False)
        s._fd_drift_gradient_only = fd
        col = _col(s, met, z.numel())
        dens = s._density_fields(met, torch.device("cpu"), torch.float32)
        ts = []
        for i in range(18):
            t0 = time.perf_counter()
            s._vertical_coefficients(z, dens, **col)
            if i >= 3:
                ts.append(time.perf_counter() - t0)
        ts.sort()
        tag = "stencil-only" if fd else "production"
        print(f"{name:10s} {tag:12s}: median {1e3 * ts[len(ts) // 2]:.2f} ms per call", flush=True)
