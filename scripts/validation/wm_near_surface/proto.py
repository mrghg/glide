"""WM near-surface investigation: prototype Hanna sub-step loop with switches.

Usage: proto.py <regime> <dt> <variant> [substep_c|default] [seed] [n]

Variants (which coefficients are re-evaluated at the particle's CURRENT height
at the start of every sub-step; the rest stay at the outer-step value):
  prod      the HannaScheme in this checkout (dynamic/CPU path; reference)
  frozen    prototype loop, everything frozen (must reproduce the main-branch scheme)
  TL        re-evaluate T_L only
  sigma     re-evaluate sigma_u/v/w only (and the clip bounds)
  drift     re-evaluate dsigma_w^2/dz and density drift only
  sig_drift sigma + drift (everything except T_L)
  all       re-evaluate everything (FLEXPART-like)

Diagnostic copied from scripts/validation/figures/tier0.py on
origin/agent/B7-figure-infrastructure (_hanna_met_window, _hanna_well_mixed,
_near_surface_error): occupancy of the lowest 5% of the BL relative to uniform,
averaged over 2-6 h, with a 10-subset standard error. Results are appended as
JSON lines to $WM_OUT/$WM_LOG (default outputs/agents/WM/results.jsonl).
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import UTC, datetime, timedelta

import numpy as np
import torch

torch.set_num_threads(int(os.environ.get("WM_THREADS", "4")))

from lpdm.gpu_engine import GPUEngine  # noqa: E402
from lpdm.met_reader import HourlyMetTensors, MetFieldMetadata  # noqa: E402
from lpdm.turbulence import HannaScheme  # noqa: E402
from lpdm.turbulence.hanna import (  # noqa: E402
    SIGMA_MIN_M_S,
    USTAR_MIN_M_S,
    W_PRIME_SIGMA_RATIO_MAX,
    convective_velocity,
    obukhov_length,
)

OUT = os.environ.get("WM_OUT", "outputs/agents/WM")

WM_REGIMES = {
    "stable": (400.0, 0.25, -20.0),
    "neutral": (1500.0, 0.4, 0.0),
    "convective": (1500.0, 0.3, 200.0),
}


def _hanna_met_window(*, blh_m, ustar, shf):
    n_lev, n_lat, n_lon = 8, 4, 4
    shape = (n_lev, n_lat, n_lon)
    zeros = torch.zeros(shape, dtype=torch.float32)
    chans = {
        "u": zeros, "v": zeros, "w": zeros,
        "blh": torch.full(shape, blh_m), "sp": torch.full(shape, 101325.0),
        "t": torch.full(shape, 280.0), "ustar": torch.full(shape, ustar),
        "shf": torch.full(shape, shf),
    }  # fmt: skip
    names = ("u", "v", "w", "blh", "sp", "t", "ustar", "shf")
    fields = torch.stack([chans[k] for k in names], dim=0)
    t0 = datetime(2024, 1, 1, 12, tzinfo=UTC)
    level = np.linspace(0.0, max(4000.0, 1.5 * blh_m), n_lev)
    md = MetFieldMetadata(
        lon=np.linspace(-2.0, 2.0, n_lon),
        lat=np.linspace(40.0, 50.0, n_lat),
        level=level,
        pressure_level_hpa=np.linspace(1000.0, 999.0, n_lev),
        time_start=t0,
        time_end=t0 + timedelta(hours=1),
        variable_units={
            "u": "m/s", "v": "m/s", "w": "m/s", "blh": "m", "sp": "Pa", "t": "K",
            "z": "m**2s**-2", "z_sfc": "m**2s**-2", "ustar": "m/s", "shf": "W m**-2",
        },
    )  # fmt: skip
    height = torch.as_tensor(level, dtype=torch.float32).view(n_lev, 1, 1).expand(shape)
    return HourlyMetTensors(
        hour_start=fields, hour_end=fields, metadata=md, channel_names=names,
        height_agl_m=height.contiguous(),
    )  # fmt: skip


class ProtoHanna(HannaScheme):
    """Dynamic (CPU) step with a switchable per-sub-step re-evaluation."""

    def __init__(self, reeval=(), **kw):
        super().__init__(**kw)
        self.reeval = set(reeval)

    def _coeffs(self, z, col_kwargs, density_fields, grid_bounds, lon, lat, engine):
        z_eval = z.clamp(min=self.z_ubl_m)
        su, sv, sw, tlu, tlv, tlw = self._column_turbulence(z_eval, **col_kwargs)
        z_hi = z_eval + self.DRIFT_FD_DELTA_M
        z_lo = (z_eval - self.DRIFT_FD_DELTA_M).clamp(min=self.z_ubl_m)
        sw_hi = self._column_turbulence(z_hi, **col_kwargs)[2]
        sw_lo = self._column_turbulence(z_lo, **col_kwargs)[2]
        dsig2 = (sw_hi.pow(2) - sw_lo.pow(2)) / (z_hi - z_lo).clamp(min=self.DRIFT_FD_DELTA_M)
        sw2 = sw.pow(2).clamp(min=SIGMA_MIN_M_S**2)
        dens = self._interp_3d_field(density_fields, grid_bounds, lon, lat, z_eval, engine)
        rho = dens[0].clamp(min=1e-3)
        return dict(
            su=su, sv=sv, sw=sw, sw2=sw2, tlu=tlu, tlv=tlv, tlw=tlw,
            half_b=-0.5 * dsig2, dens_b=-(sw2 * dens[1] / rho),
        )  # fmt: skip

    def step(self, particles, state, met_window, t_alpha, dt_seconds, active_mask, engine):
        device, dtype = engine.device, engine.dtype
        xyz = particles[active_mask]
        u_p = state["u_prime"][active_mask].clone()
        v_p = state["v_prime"][active_mask].clone()
        w_p = state["w_prime"][active_mask].clone()
        lon, lat = xyz[:, 0], xyz[:, 1]
        sf = self._interp_surface_field
        blh = self._limit_blh(sf(met_window, "blh", t_alpha, lon, lat, device, dtype))
        sp = sf(met_window, "sp", t_alpha, lon, lat, device, dtype)
        ustar = sf(met_window, "ustar", t_alpha, lon, lat, device, dtype).clamp(min=USTAR_MIN_M_S)
        shf = sf(met_window, "shf", t_alpha, lon, lat, device, dtype)
        t_s = self._interp_t_at_lowest_level(met_window, t_alpha, lon, lat, device, dtype)
        L = obukhov_length(ustar, shf, t_s, sp)
        w_star = convective_velocity(blh, shf, t_s, sp, blh_min_m=self.blh_min_m)
        ft_fields, grid_bounds = self._free_trop_fields(met_window, device, dtype)
        density_fields = self._density_fields(met_window, device, dtype)
        col = dict(blh=blh, ustar=ustar, w_star=w_star, h_over_L=blh / L, L=L, lat=lat, lon=lon,
                   ft_fields=ft_fields, grid_bounds=grid_bounds, engine=engine)  # fmt: skip
        args = (col, density_fields, grid_bounds, lon, lat, engine)
        c0 = self._coeffs(xyz[:, 2], *args)
        # Per-particle count, as GLIDE did before the fix (computed here so the
        # prototype does not depend on which HannaScheme is on the path).
        k_req = (
            torch.ceil(dt_seconds / (self.substep_c * c0["tlw"].clamp(min=1e-3)))
            .long()
            .clamp(min=1, max=self.max_substeps)
        )
        sub_dt = float(dt_seconds) / k_req.to(dtype)
        max_k = int(k_req.max())
        if os.environ.get("WM_KUNIFORM", "0") == "1":
            # Same sub-step count for every particle (the near-surface maximum).
            k_req = torch.full_like(k_req, max_k)
            sub_dt = float(dt_seconds) / k_req.to(dtype)
        xyz = xyz.clone()
        groups = {
            "TL": ("tlu", "tlv", "tlw"),
            "sigma": ("su", "sv", "sw", "sw2"),
            "drift": ("half_b", "dens_b"),
        }
        kadapt = os.environ.get("WM_KADAPT", "0") == "1"
        if kadapt:
            # FLEXPART-like: each sub-step's length is set from the T_Lw at the
            # particle's CURRENT height, until the outer step is used up; the
            # last allowed iteration takes whatever remains.
            remaining = torch.full_like(c0["tlw"], float(dt_seconds))
            max_k = self.max_substeps
        for i in range(max_k):
            c = dict(c0)
            if self.reeval and i > 0:
                ci = self._coeffs(xyz[:, 2], *args)
                for g in self.reeval:
                    for key in groups[g]:
                        c[key] = ci[key]
            if kadapt:
                m = remaining > 1e-6
                if not bool(m.any()):
                    break
                n_left = torch.ceil(remaining / (self.substep_c * c["tlw"])).clamp(min=1)
                step_i = remaining / n_left
                if i == max_k - 1:
                    step_i = remaining
                dtm = step_i[m]
                remaining = torch.where(m, remaining - step_i, remaining)
            else:
                m = k_req > i
                dtm = sub_dt[m]
            factor = 1.0 + w_p[m].pow(2) / c["sw2"][m]
            drift = factor * c["half_b"][m] + c["dens_b"][m]
            un = engine.update_langevin_velocity(u_p[m], c["tlu"][m], c["su"][m].pow(2), dtm)
            vn = engine.update_langevin_velocity(v_p[m], c["tlv"][m], c["sv"][m].pow(2), dtm)
            wn = engine.update_langevin_velocity(
                w_p[m], c["tlw"][m], c["sw"][m].pow(2), dtm, drift=drift
            )
            R = W_PRIME_SIGMA_RATIO_MAX
            wn = torch.clamp(wn, min=-R * c["sw"][m], max=R * c["sw"][m])
            un = torch.clamp(un, min=-R * c["su"][m], max=R * c["su"][m])
            vn = torch.clamp(vn, min=-R * c["sv"][m], max=R * c["sv"][m])
            xm = engine.apply_vertical_turbulence(xyz[m], wn, dtm, backward=True)
            xm = engine.apply_horizontal_turbulence(xm, un, vn, dtm, backward=True)
            xm, wn = engine.reflect_surface(xm, wn, z_surface=0.0)
            if os.environ.get("WM_TOPSUB", "0") == "1":
                # BL-top reflection inside the sub-step loop (harness option): keeps
                # particles that cross h mid-step from sitting in the (near-still)
                # free troposphere until the outer-step reflection.
                h = blh[m]
                over = xm[:, 2] > h
                xm[:, 2] = torch.where(over, 2.0 * h - xm[:, 2], xm[:, 2])
                wn = torch.where(over, -wn, wn)
            xyz[m] = xm
            u_p[m], v_p[m], w_p[m] = un, vn, wn
        particles[active_mask] = xyz
        state["u_prime"][active_mask] = u_p
        state["v_prime"][active_mask] = v_p
        state["w_prime"][active_mask] = w_p
        return particles, state


VARIANTS = {
    "frozen": (),
    "TL": ("TL",),
    "sigma": ("sigma",),
    "drift": ("drift",),
    "sig_drift": ("sigma", "drift"),
    "TL_sigma": ("TL", "sigma"),
    "all": ("TL", "sigma", "drift"),
}


def run(regime, dt, variant, substep_c=None, seed=11, n=10_000, static=False):
    blh, ustar, shf = WM_REGIMES[regime]
    torch.manual_seed(seed)
    engine = GPUEngine(device="cpu")
    kw = {} if substep_c is None else {"substep_c": substep_c}
    if variant == "prod":
        scheme = HannaScheme(meander_enabled=False, static_substeps=static, **kw)
    else:
        scheme = ProtoHanna(reeval=VARIANTS[variant], meander_enabled=False,
                            static_substeps=False, **kw)  # fmt: skip
    met = _hanna_met_window(blh_m=blh, ustar=ustar, shf=shf)
    p = torch.zeros(n, 4, dtype=torch.float32)
    p[:, 1] = 45.0
    p[:, 2] = torch.rand(n, dtype=torch.float32) * blh
    p[:, 3] = 1.0 / n
    state = scheme.initialize_state(n, device=torch.device("cpu"), dtype=torch.float32)
    active = torch.ones(n, dtype=torch.bool)
    rec = list(np.arange(7200.0, 21600.0 + 1, 900.0))
    zs = []
    t = 0.0
    for t_rec in rec:
        while t < t_rec - 1e-9:
            p, state = scheme.step(p, state, met, t_alpha=0.5, dt_seconds=dt,
                                   active_mask=active, engine=engine)  # fmt: skip
            above = p[:, 2] > blh
            if bool(above.any()):
                p[above, 2] = 2.0 * blh - p[above, 2]
                state["w_prime"][above] = -state["w_prime"][above]
            t += dt
        zs.append(p[:, 2].numpy().copy())
    zs = np.stack(zs)
    occ = (zs < 0.05 * blh).mean() / 0.05
    rng = np.random.default_rng(0)
    perm = rng.permutation(n)
    vals = [(zs[:, idx] < 0.05 * blh).mean() / 0.05 for idx in np.array_split(perm, 10)]
    se = float(np.std(vals, ddof=1) / np.sqrt(10))
    prof = np.histogram(zs.ravel(), bins=10, range=(0, blh))[0] / zs.size * 10
    return float(occ), se, prof.tolist()


if __name__ == "__main__":
    regime, dt, variant = sys.argv[1], float(sys.argv[2]), sys.argv[3]
    sc = None if len(sys.argv) < 5 or sys.argv[4] == "default" else float(sys.argv[4])
    seed = int(sys.argv[5]) if len(sys.argv) > 5 else 11
    n = int(sys.argv[6]) if len(sys.argv) > 6 else 10_000
    static = os.environ.get("WM_STATIC", "0") == "1"
    t0 = time.time()
    occ, se, prof = run(regime, dt, variant, sc, seed, n, static)
    rec = dict(regime=regime, dt=dt, variant=variant, substep_c=sc, seed=seed, n=n,
               static=static, tag=os.environ.get("WM_TAG", ""),
               occ=round(occ, 4), se=round(se, 4), profile10=[round(x, 3) for x in prof],
               wall_s=round(time.time() - t0, 1))  # fmt: skip
    line = json.dumps(rec)
    print(line)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, os.environ.get("WM_LOG", "results.jsonl")), "a") as f:
        f.write(line + "\n")
