"""Row transfer matrices in the steady stable column of tests/test_reciprocity.py
(the A6 review's rowmat.py, with the prototype's switches).

M[a, b] = mean over particles started uniformly (Eulerian velocities) in row a
of their residence sum(dt / rho_d) in row b over 2 h. In a steady, horizontally
uniform column the backward chain is the forward chain with w' -> -w', so
reciprocity there is detailed balance: M[a, b] = M[b, a] for rows of equal depth.

--variant is a comma list of key=value (or bare flags):
  mode=fd|kink|analytic         HannaScheme.DRIFT_GRADIENT_MODE
  form=clamp|quadrature|offset  HannaScheme.SIGMA_W_FLOOR_FORM
  floor=<m/s>                   hanna.SIGMA_MIN_M_S (every use, as the review did)
  fd=<m>                        HannaScheme.DRIFT_FD_DELTA_M
  tlcap=<s>                     cap T_Lu, T_Lv, T_Lw at this value
  nocarry                       the outer-step velocity carry disabled
  noclip                        the 4 sigma clip disabled
Env GLIDE_WT = the source tree."""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

WT = os.environ["GLIDE_WT"]
sys.path.insert(0, WT + "/tests")
import lpdm  # noqa: E402

assert lpdm.__file__.startswith(WT), lpdm.__file__
import test_reciprocity as R  # noqa: E402

import lpdm.turbulence.hanna as H  # noqa: E402


def scheme_class(variant: str):
    opts = {}
    for tok in filter(None, variant.split(",")):
        k, _, v = tok.partition("=")
        opts[k] = v
    if "floor" in opts:
        H.SIGMA_MIN_M_S = float(opts["floor"])
    if "noclip" in opts:
        H.W_PRIME_SIGMA_RATIO_MAX = 1e9
    attrs = {}
    if "mode" in opts:
        attrs["DRIFT_GRADIENT_MODE"] = opts["mode"]
    if "form" in opts:
        attrs["SIGMA_W_FLOOR_FORM"] = opts["form"]
    if "fd" in opts:
        attrs["DRIFT_FD_DELTA_M"] = float(opts["fd"])
    if "tlcap" in opts:
        cap = float(opts["tlcap"])

        def _column_turbulence(self, z, **ck):
            su, sv, sw, tu, tv, tw = R.HannaScheme._column_turbulence(self, z, **ck)
            return su, sv, sw, tu.clamp(max=cap), tv.clamp(max=cap), tw.clamp(max=cap)

        attrs["_column_turbulence"] = _column_turbulence
    if "nocarry" in opts:
        attrs["_carry_velocities"] = staticmethod(lambda u, v, w, *a: (u, v, w))
    return type("Variant", (R.HannaScheme,), attrs), opts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", default="stable")
    ap.add_argument("--nsub", type=int, default=16)
    ap.add_argument("--rows", default="6-11")
    ap.add_argument("--nper", type=int, default=40000)
    ap.add_argument("--dir", choices=["fwd", "bwd"], default="fwd")
    ap.add_argument("--variant", default="")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dt", type=float, default=60.0)
    ap.add_argument("--nsteps", type=int, default=120)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    cls, opts = scheme_class(a.variant)

    lo_r, hi_r = map(int, a.rows.split("-"))
    rows = range(lo_r, hi_r + 1)
    h = R.REGIMES[a.regime][0]
    dz = h / 10
    edges = tuple(float(x) for x in np.arange(0.0, 1.3 * h + 1e-6, dz))
    grid = R.Grid(edges, 1.0e5, -1, 1)
    met = R.reciprocity_met(*R.REGIMES[a.regime])
    gen = torch.Generator().manual_seed(9000 + a.seed)
    z0 = R._release_heights(edges, rows, a.nper, gen)
    t0 = time.time()
    contrib, _ = R.run_particles(
        cls(n_substeps=a.nsub, static_substeps=False),
        met,
        z0,
        forward=(a.dir == "fwd"),
        grid=grid,
        n_steps=a.nsteps,
        dt=a.dt,
        seed=100 + a.seed,
    )
    c = contrib.sum(2).view(len(rows), a.nper, grid.n_rows)  # [start row, particle, res row]
    out = dict(
        regime=a.regime, nsub=a.nsub, dir=a.dir, variant=a.variant, opts=opts, rows=list(rows),
        nper=a.nper, seed=a.seed, dt=a.dt, nsteps=a.nsteps, seconds=time.time() - t0,
        sigma_min=H.SIGMA_MIN_M_S, wt=WT,
        M=c.mean(1).numpy().tolist(), SE=(c.std(1) / a.nper**0.5).numpy().tolist(),
        NZ=(c > 0).sum(1).numpy().tolist(),
        # residence summed over the boundary-layer rows 6-9 (h - 160 m ... h) per start row
        BL4=c[:, :, 6:10].sum(2).mean(1).numpy().tolist(),
        BL4_SE=(c[:, :, 6:10].sum(2).std(1) / a.nper**0.5).numpy().tolist(),
    )  # fmt: skip
    with open(a.out, "w") as f:
        json.dump(out, f)
    print(a.out, "done in", round(time.time() - t0), "s", flush=True)


if __name__ == "__main__":
    main()
