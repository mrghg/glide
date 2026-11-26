"""Backward trajectories through the production step for the bit-identity check:
uniform start in [0, 1.3 h], Eulerian velocities, 60 steps of 60 s, no advection,
heights after every step saved to NPZ. Env GLIDE_WT = the source tree.
Usage: bitid.py REGIME MODE FORM STATIC(0/1) OUT.npz"""

import os
import sys

import numpy as np
import torch

WT = os.environ["GLIDE_WT"]
sys.path.insert(0, WT + "/tests")
import lpdm  # noqa: E402

assert lpdm.__file__.startswith(WT), lpdm.__file__
import test_reciprocity as R  # noqa: E402

from lpdm.gpu_engine import GPUEngine  # noqa: E402

regime, mode, form, static, out = sys.argv[1:6]
torch.set_num_threads(1)
met = R.reciprocity_met(*R.REGIMES[regime])
h = R.REGIMES[regime][0]
n = 20000
g = torch.Generator().manual_seed(5)
z0 = torch.rand(n, generator=g) * 1.3 * h
eng = GPUEngine(device="cpu")
s = R.HannaScheme(n_substeps=16, static_substeps=bool(int(static)))
if hasattr(s, "DRIFT_GRADIENT_MODE"):
    s.DRIFT_GRADIENT_MODE = mode
    s.SIGMA_W_FLOOR_FORM = form
p = torch.zeros(n, 4)
p[:, 1] = R.LAT0
p[:, 2] = z0
p[:, 3] = 1 / n
st = s.initialize_state(n, device=torch.device("cpu"), dtype=torch.float32)
sig = R.eulerian_sigmas(s, met, p[:, :3], eng)
e = torch.randn(n, 3, generator=torch.Generator().manual_seed(7))
st["u_prime"], st["v_prime"], st["w_prime"] = (sig[:, i] * e[:, i] for i in range(3))
st["sigma_last"] = sig.clone()
torch.manual_seed(11)
act = torch.ones(n, dtype=torch.bool)
zs = []
for _ in range(60):
    p, st = s.step(p, st, met, 0.5, 60.0, act, eng)
    zs.append(p[:, 2].clone())
np.savez(out, z=torch.stack(zs).numpy(), w=st["w_prime"].numpy(), h=h)
print(out, "ok")
