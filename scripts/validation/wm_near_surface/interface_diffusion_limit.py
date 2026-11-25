"""Diffusion-limit reference for the ``openbl`` harness of interface_study.py.

For times long compared with T_L, a well-mixed-consistent Langevin model
diffuses with K = σ_w² T_Lw and keeps the concentration continuous across a
jump in σ_w (no net flux through h in a well-mixed state). This solves

    ∂c/∂t = ∂/∂z (K ∂c/∂z),  c(z, 0) = 1/h on [0, h], 0 above,

with no-flux boundaries at the ground and at 2.5 h, on a 0.5 m finite-volume
grid (implicit Euler, 5 s steps; face diffusivity = harmonic mean of the two
cell values, which carries the flux across the jump at h exactly), with K
from the production profile of the scheme on the PYTHONPATH
(``_column_turbulence``: boundary-layer profile at and below h, the
free-troposphere closure above), in the same column as the harness. Prints,
per regime, the share of the mass at or below h at each 15-min record, its
2-6 h mean, and the semi-infinite estimate 1 − (2/h)√(K_FT t/π) (well-mixed
boundary layer, uniform K_FT above h).

    PYTHONPATH=src:scripts/validation/wm_near_surface \
        python scripts/validation/wm_near_surface/interface_diffusion_limit.py [--json out]

The diffusion limit needs t ≫ T_L at every height; where T_Lw is long (the
top of the stable boundary layer, where σ_w → 0 and T_Lw grows without bound)
it is only indicative.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime

import numpy as np
import torch
from fixcount_study import REGIMES
from interface_study import column_sigmas, met_window
from scipy.linalg import solve_banded

from lpdm.gpu_engine import GPUEngine
from lpdm.turbulence import HannaScheme
from lpdm.turbulence import hanna as hanna_mod


def k_profile(regime: str, z: np.ndarray, ft_n: float) -> tuple[np.ndarray, float]:
    h, ustar, shf = REGIMES[regime]
    scheme = HannaScheme(meander_enabled=False, static_substeps=False)
    t0 = datetime(2024, 1, 1, 12, tzinfo=UTC)
    met = met_window(h, h, ustar, shf, ft_n=ft_n, strat_from=h, t0=t0, top=max(4000.0, 2.5 * h))
    n = z.size
    full = lambda v: torch.full((n,), float(v))  # noqa: E731
    sp, tk, blh, us = full(101325.0), full(280.0), full(h), full(ustar)
    L = hanna_mod.obukhov_length(us, full(shf), tk, sp)
    w_star = hanna_mod.convective_velocity(blh, full(shf), tk, sp, blh_min_m=scheme.blh_min_m)
    ft_fields, gb = scheme._free_trop_fields(met, torch.device("cpu"), torch.float32)
    col = dict(
        blh=blh, ustar=us, w_star=w_star, h_over_L=blh / L, L=L, lat=full(45.0),
        lon=full(0.0), ft_fields=ft_fields, grid_bounds=gb, engine=GPUEngine(device="cpu"),
    )  # fmt: skip
    zq = torch.as_tensor(z, dtype=torch.float32).clamp(min=scheme.z_ubl_m)
    _, _, sw, _, _, tlw = scheme._column_turbulence(zq, **col)
    k = (sw.double().pow(2) * tlw.double()).numpy()
    sig = column_sigmas(scheme, met, h, ustar, shf, 1)
    k_ft = float(sig[1, 2, 0]) ** 2 * float(tlw[z > h][0])
    return k, k_ft


def solve(regime: str, ft_n: float, hours: float = 6.0, dz: float = 0.5, dt: float = 5.0):
    h = REGIMES[regime][0]
    top = 2.5 * h
    edges = np.arange(0.0, top + dz / 2, dz)
    zc = 0.5 * (edges[:-1] + edges[1:])
    k, k_ft = k_profile(regime, zc, ft_n)
    kf = 2.0 * k[:-1] * k[1:] / np.maximum(k[:-1] + k[1:], 1e-30)  # interior faces
    a = kf * dt / dz**2
    m = zc.size
    ab = np.zeros((3, m))
    ab[0, 1:] = -a
    ab[2, :-1] = -a
    diag = np.ones(m)
    diag[:-1] += a
    diag[1:] += a
    ab[1] = diag
    c = np.where(zc <= h, 1.0 / h, 0.0)
    below = zc <= h
    out = []
    t = 0.0
    for t_rec in np.arange(900.0, hours * 3600.0 + 1.0, 900.0):
        while t < t_rec - 1e-9:
            c = solve_banded((1, 1), ab, c)
            t += dt
        out.append((t, float(c[below].sum() * dz)))
    est = [(tt, 1.0 - 2.0 * np.sqrt(k_ft * tt / np.pi) / h) for tt, _ in out]
    return out, est, k_ft


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ft-n", type=float, default=0.015)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    res = {}
    for regime in ("stable", "neutral", "convective"):
        series, est, k_ft = solve(regime, a.ft_n)
        late = [v for t, v in series if t >= 7200.0 - 1e-6]
        late_est = [v for t, v in est if t >= 7200.0 - 1e-6]
        res[regime] = dict(
            k_ft=k_ft,
            frac_bl_series=series,
            frac_bl_2_6h=float(np.mean(late)),
            frac_bl_6h=series[-1][1],
            semi_infinite_2_6h=float(np.mean(late_est)),
            semi_infinite_6h=est[-1][1],
        )
        print(
            f"{regime}: K_FT {k_ft:.3f} m2/s; BL share 2-6 h mean {np.mean(late):.4f}, "
            f"6 h {series[-1][1]:.4f}; semi-infinite estimate {np.mean(late_est):.4f} / "
            f"{est[-1][1]:.4f}"
        )
    if a.json:
        with open(a.json, "w") as f:
            json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
