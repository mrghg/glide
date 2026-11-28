"""Boundary-layer-top interface study (decision 0028).

One run = one regime, harness, sub-step count, variant and seed; appends one
JSON line to ``--out``. Uses the production ``HannaScheme`` on the PYTHONPATH
(dynamic CPU path, backward), so the same script measures this branch, the
PR #69 head and main (``--variant old`` for main's adaptive schedule).

    PYTHONPATH=src:scripts/validation/wm_near_surface python \
        scripts/validation/wm_near_surface/interface_study.py \
        --regime convective --harness open2h --n 16 --seed 1 --out results.jsonl

Column (``fixcount_study.met_window``): horizontally uniform, motionless; the
free troposphere above h is stratified with N = ``--ft-n`` (K_z at the closure's
0.1 m² s⁻¹ floor, T_Lw = 0.5/N), the boundary layer has uniform density.

Harnesses:

* ``closed``: start uniform in [0, h], reflecting lid at h in every sub-step
  (the CI near-surface gate); the lid replaces the interface, so the crossing
  rule is off. Exact answer: uniform in [0, h].
* ``open2h``: start uniform in [0, 2h], reflecting lid at 2h in every
  sub-step, the free-troposphere coefficients between h and 2h. Exact answer:
  uniform in [0, 2h] (the density varies by ~3% over [h, 2h], but its drift
  moves particles by < 1 m in 6 h, so the 6-h answer is uniform).
* ``openbl``: start uniform in [0, h], no lid. The share still in the BL is
  compared with the diffusion limit (``interface_diffusion_limit.py``).
* ``hvar``: as ``open2h``, but h moves: a triangle wave between 0.75 h0 and
  1.25 h0 with a 2-h period (each met window ramps h linearly, as the reader's
  time interpolation does), stratification above 0.75 h0. Exact answer for a
  model that keeps a well-mixed state well mixed: uniform in [0, 2 h0].

Variants (``--variant``):

* ``prod``: ``HannaScheme(n_substeps=n)`` as on the PYTHONPATH.
* ``old``: ``HannaScheme()`` with its own defaults (main's adaptive schedule).
* ``nocross``: this branch with the crossing rule switched off (one-sided
  stencil only).
* ``nostencil``: this branch with the drift stencil allowed to straddle h, as
  before decision 0028 (crossing rule kept).
* ``rescale`` (``hvar`` only): ``prod`` plus, before each step, w', u' and v'
  of a particle whose side of h changed because h moved are scaled by the
  ratio of σ at h on its new side to its old side (no reflection: the
  interface moved, the particle did not cross it).
* ``carry`` (``hvar`` only): ``prod`` plus, before each step, every particle's
  u', v', w' scaled by σ at its height with this step's h over σ at its height
  with the previous step's h (w'/σ carried across steps, the treatment of step
  changes in time that STILT attributes to Thomson et al. 1997).
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import UTC, datetime, timedelta

import numpy as np
import torch
from fixcount_study import GRAVITY, REGIMES, LidEngine

from lpdm.gpu_engine import GPUEngine
from lpdm.local_frame import DEGREES
from lpdm.met_reader import HourlyMetTensors, MetFieldMetadata
from lpdm.turbulence import HannaScheme
from lpdm.turbulence import hanna as hanna_mod


def met_window(blh_a, blh_b, ustar, shf, *, ft_n, strat_from, t0, top):
    """``fixcount_study.met_window`` with h ramping from ``blh_a`` (window start)
    to ``blh_b`` (window end) and θ rising with N = ``ft_n`` above ``strat_from``."""
    n_lev, n_lat, n_lon = 41, 4, 4
    shape = (n_lev, n_lat, n_lon)
    level = np.linspace(0.0, top, n_lev)
    dtheta_dz = ft_n**2 * 280.0 / GRAVITY
    above = torch.as_tensor(np.clip(level - strat_from, 0.0, None), dtype=torch.float32)
    t = torch.full(shape, 280.0) + (dtheta_dz * above).view(n_lev, 1, 1)
    zeros = torch.zeros(shape, dtype=torch.float32)
    names = ("u", "v", "w", "blh", "sp", "t", "ustar", "shf")

    def fields(blh):
        chans = {
            "u": zeros, "v": zeros, "w": zeros,
            "blh": torch.full(shape, float(blh)), "sp": torch.full(shape, 101325.0),
            "t": t, "ustar": torch.full(shape, ustar), "shf": torch.full(shape, shf),
        }  # fmt: skip
        return torch.stack([chans[k] for k in names], dim=0)

    md = MetFieldMetadata(
        lon=np.linspace(-2.0, 2.0, n_lon),
        lat=np.linspace(40.0, 50.0, n_lat),
        level=level,
        pressure_level_hpa=np.linspace(1000.0, 999.0, n_lev),
        time_start=t0,
        time_end=t0 + timedelta(hours=1),
        variable_units=dict.fromkeys(names, "1"),
    )
    height = torch.as_tensor(level, dtype=torch.float32).view(n_lev, 1, 1).expand(shape)
    return HourlyMetTensors(
        hour_start=fields(blh_a),
        hour_end=fields(blh_b),
        metadata=md,
        channel_names=names,
        height_agl_m=height.contiguous(),
    )


class NoCross(HannaScheme):
    """One-sided stencil only: the crossing rule is the identity."""

    def _cross_interface(self, z_from, xyz, w_p, u_p, v_p, blh, interface_sigmas):
        return xyz, w_p, u_p, v_p


class NoStencil(HannaScheme):
    """Crossing rule only: the drift stencil may straddle h (pre-0028 stencil)."""

    def _vertical_coefficients(self, z_agl, density_fields, **ck):
        out = list(super()._vertical_coefficients(z_agl, density_fields, **ck))
        z_eval = z_agl.clamp(min=self.z_ubl_m)
        z_hi = z_eval + self.DRIFT_FD_DELTA_M
        z_lo = (z_eval - self.DRIFT_FD_DELTA_M).clamp(min=self.z_ubl_m)
        sw_hi = self._column_sigma_w(z_hi, **ck)
        sw_lo = self._column_sigma_w(z_lo, **ck)
        g = (sw_hi.pow(2) - sw_lo.pow(2)) / (z_hi - z_lo).clamp(min=self.DRIFT_FD_DELTA_M)
        if self.basal_zero_gradient:
            g = torch.where(z_agl < self.z_ubl_m, torch.zeros_like(g), g)
        out[7] = -0.5 * g
        return tuple(out)


def build_scheme(variant: str, n: int) -> HannaScheme:
    kw = dict(meander_enabled=False, static_substeps=False)
    if variant == "old":
        return HannaScheme(**kw)
    cls = {"prod": HannaScheme, "rescale": HannaScheme, "carry": HannaScheme, "nocross": NoCross,
           "nostencil": NoStencil}[variant]  # fmt: skip
    return cls(n_substeps=n, **kw)


def column_kwargs(scheme, met, h, ustar, shf, n):
    """``col_kwargs`` of the uniform column with boundary-layer depth ``h``."""
    dev, dt_ = torch.device("cpu"), torch.float32
    full = lambda v: torch.full((n,), float(v))  # noqa: E731
    sp, tk = full(101325.0), full(280.0)
    blh = full(h)
    us = full(ustar)
    L = hanna_mod.obukhov_length(us, full(shf), tk, sp)
    w_star = hanna_mod.convective_velocity(blh, full(shf), tk, sp, blh_min_m=scheme.blh_min_m)
    ft_fields, gb = scheme._free_trop_fields(met, dev, dt_, frame=DEGREES)
    return dict(
        blh=blh, ustar=us, w_star=w_star, h_over_L=blh / L, L=L, lat=full(45.0),
        x=full(0.0), y=full(45.0), ft_fields=ft_fields, grid_bounds=gb, engine=GPUEngine(device="cpu"),
    )  # fmt: skip


def column_sigmas(scheme, met, h, ustar, shf, n):
    """σ (u, v, w) at h on the BL and FT side for the uniform column, ``[2, 3, n]``."""
    return scheme._interface_sigmas(**column_kwargs(scheme, met, h, ustar, shf, n))


def sigmas_at(scheme, met, h, ustar, shf, z):
    """σ (u, v, w) at heights ``z`` in the uniform column with depth ``h``, ``[3, n]``."""
    col = column_kwargs(scheme, met, h, ustar, shf, z.numel())
    su, sv, sw, *_ = scheme._column_turbulence(z.clamp(min=scheme.z_ubl_m), **col)
    return torch.stack([su, sv, sw])


def h_of_t(t: float, h0: float) -> float:
    """Triangle wave: 0.75 h0 at t = 0, 1.25 h0 at 1 h, 0.75 h0 at 2 h, ..."""
    phase = (t / 3600.0) % 2.0
    frac = phase if phase <= 1.0 else 2.0 - phase
    return h0 * (0.75 + 0.5 * frac)


def band(z, lo, hi):
    """Share of particles in [lo, hi) per unit share of height (uniform = 1)."""
    return float(((z >= lo) & (z < hi)).mean())


def run(a) -> dict:
    h0, ustar, shf = REGIMES[a.regime]
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    if a.sigma_floor is not None:
        # Probe: the σ floor of the profiles (module constant, read at call time).
        hanna_mod.SIGMA_MIN_M_S = float(a.sigma_floor)
    scheme = build_scheme(a.variant, a.n)
    if a.fd_delta is not None:
        scheme.DRIFT_FD_DELTA_M = float(a.fd_delta)
    if a.harness == "closed":
        # The lid at h replaces the interface: no crossing rule (a crossing
        # particle is reflected by the lid, as in the CI near-surface gate).
        scheme._cross_interface = lambda z_from, xyz, w, u, v, blh, s: (xyz, w, u, v)
    t0 = datetime(2024, 1, 1, 12, tzinfo=UTC)
    top = max(4000.0, 2.5 * h0)
    hvar = a.harness == "hvar"
    ft_n = None if a.harness == "closed" else a.ft_n
    if a.harness == "closed":
        from fixcount_study import met_window as closed_met

        static_met = closed_met(h0, ustar, shf, ft_n=None)
    elif not hvar:
        static_met = met_window(h0, h0, ustar, shf, ft_n=ft_n, strat_from=h0, t0=t0, top=top)
    lid = {"closed": h0, "open2h": 2.0 * h0, "hvar": 2.0 * h0, "openbl": None}[a.harness]
    engine = GPUEngine(device="cpu") if lid is None else LidEngine(lid, device="cpu")
    start_top = h0 if a.harness in ("closed", "openbl") else 2.0 * h0

    n = a.particles
    p = torch.zeros(n, 4, dtype=torch.float32)
    p[:, 1] = 45.0
    p[:, 2] = torch.rand(n, dtype=torch.float32) * start_top
    p[:, 3] = 1.0 / n
    state = scheme.initialize_state(n, device=torch.device("cpu"), dtype=torch.float32)
    if a.init_sigma:
        # Probe: start u', v', w' from N(0, σ²) at each particle's height and side
        # (the production start is 0).
        init_met = static_met if not hvar else met_window(
            h_of_t(0.0, h0), h_of_t(3600.0, h0), ustar, shf, ft_n=ft_n,
            strat_from=0.75 * h0, t0=t0, top=top,
        )  # fmt: skip
        sig = sigmas_at(scheme, init_met, h_of_t(0.0, h0) if hvar else h0, ustar, shf, p[:, 2])
        for k, key in enumerate(("u_prime", "v_prime", "w_prime")):
            state[key] = sig[k] * torch.randn(n)
    active = torch.ones(n, dtype=torch.bool)

    rec_every = 900.0
    t_end = a.hours * 3600.0
    first_rec = 900.0 if a.harness == "openbl" else 7200.0
    rec_times = list(np.arange(first_rec, t_end + 1.0, rec_every))
    records = []
    t, steps, t_wall = 0.0, 0, time.time()
    met, met_hour = None, None
    h_prev = None
    for t_rec in rec_times:
        while t < t_rec - 1e-9:
            if hvar:
                hour = int(t // 3600)
                if hour != met_hour:
                    met = met_window(
                        h_of_t(hour * 3600.0, h0), h_of_t((hour + 1) * 3600.0, h0), ustar, shf,
                        ft_n=ft_n, strat_from=0.75 * h0, t0=t0 + timedelta(hours=hour), top=top,
                    )  # fmt: skip
                    met_hour = hour
                alpha = (t - hour * 3600.0) / 3600.0
                h_now = (
                    h_of_t(hour * 3600.0, h0) * (1 - alpha)
                    + h_of_t((hour + 1) * 3600.0, h0) * alpha
                )
                if a.variant == "rescale" and h_prev is not None:
                    z = p[:, 2]
                    was_ft, now_ft = z > h_prev, z > h_now
                    flip = was_ft != now_ft
                    sig = column_sigmas(scheme, met, h_now, ustar, shf, n)
                    s_new = torch.where(now_ft, sig[1], sig[0])
                    s_old = torch.where(now_ft, sig[0], sig[1])
                    r = s_new / s_old
                    for k, key in enumerate(("u_prime", "v_prime", "w_prime")):
                        state[key] = torch.where(flip, state[key] * r[k], state[key])
                if a.variant == "carry" and h_prev is not None:
                    # w'/σ carried across the step: σ at the particle's height
                    # with this step's h over σ there with the previous step's.
                    z = p[:, 2]
                    r = sigmas_at(scheme, met, h_now, ustar, shf, z) / sigmas_at(
                        scheme, met, h_prev, ustar, shf, z
                    )
                    for k, key in enumerate(("u_prime", "v_prime", "w_prime")):
                        state[key] = state[key] * r[k]
                h_prev = h_now
            else:
                met, alpha = static_met, 0.5
            p, state = scheme.step(p, state, met, t_alpha=alpha, dt_seconds=a.dt,
                                   active_mask=active, engine=engine)  # fmt: skip
            t += a.dt
            steps += 1
        z = p[:, 2].numpy().astype(np.float64)
        h_t = h_now if hvar else h0
        records.append(dict(t=t, h=h_t, z=z))

    out = dict(
        regime=a.regime, harness=a.harness, dt=a.dt, n=None if a.variant == "old" else a.n,
        variant=a.variant, seed=a.seed, particles=n, steps=steps, ft_n=ft_n, tag=a.tag,
        fd_delta=a.fd_delta, src=os.path.dirname(hanna_mod.__file__), h0=h0,
        init_sigma=a.init_sigma, sigma_floor=a.sigma_floor,
    )  # fmt: skip
    m = {}
    if a.harness == "closed":
        zs = np.stack([r["z"] for r in records])
        m["lo5"] = float((zs < 0.05 * h0).mean() / 0.05)
        m["z40"] = float((zs < 40.0).mean() / (40.0 / h0))
    elif a.harness in ("open2h", "hvar"):
        span = 2.0 * h0
        dens = {
            "lo5": (0.0, 0.05 * h0), "z40": (0.0, 40.0),
            "below200": (-200.0, -50.0), "below50": (-50.0, 0.0), "below10": (-10.0, 0.0),
            "above10": (0.0, 10.0), "above50": (0.0, 50.0), "above200": (50.0, 200.0),
        }  # fmt: skip
        acc = {k: [] for k in dens}
        acc["frac_below_h"] = []
        for r in records:
            z, h = r["z"], r["h"]
            for k, (lo, hi) in dens.items():
                rel = k not in ("lo5", "z40")
                a_lo, a_hi = (h + lo, h + hi) if rel else (lo, hi)
                acc[k].append(band(z, a_lo, a_hi) / ((a_hi - a_lo) / span))
            acc["frac_below_h"].append(float((z <= h).mean()) / (h / span))
        m = {k: float(np.mean(v)) for k, v in acc.items()}
        # 10-m profile from h - 200 to h + 200 m, relative to uniform.
        edges = np.arange(-200.0, 201.0, 10.0)
        prof = np.zeros(len(edges) - 1)
        for r in records:
            prof += np.histogram(r["z"] - r["h"], bins=edges)[0] / n / (10.0 / span)
        out["profile10_edges_rel_h"] = edges.tolist()
        out["profile10"] = [round(float(x), 4) for x in prof / len(records)]
        full_edges = np.linspace(0.0, span, 41)
        fp = np.zeros(40)
        for r in records:
            fp += np.histogram(r["z"], bins=full_edges)[0] / n * 40
        out["profile40"] = [round(float(x), 4) for x in fp / len(records)]
    elif a.harness == "openbl":
        out["frac_bl_series"] = [[r["t"], round(float((r["z"] <= h0).mean()), 6)] for r in records]
        late = [r for r in records if r["t"] >= 7200.0 - 1e-6]
        zs = np.stack([r["z"] for r in late])
        m["frac_bl"] = float((zs <= h0).mean())
        m["frac_bl_6h"] = float((records[-1]["z"] <= h0).mean())
        in_bl = zs <= h0
        m["lo5_bl"] = float(((zs < 0.05 * h0) & in_bl).sum() / in_bl.sum() / 0.05)
        m["z40_bl"] = float(((zs < 40.0) & in_bl).sum() / in_bl.sum() / (40.0 / h0))
    out["metrics"] = m
    out["wall_s"] = round(time.time() - t_wall, 1)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", choices=sorted(REGIMES), required=True)
    ap.add_argument("--harness", choices=["closed", "open2h", "openbl", "hvar"], required=True)
    ap.add_argument("--dt", type=float, default=60.0)
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--variant", default="prod")
    ap.add_argument("--ft-n", type=float, default=0.015)
    ap.add_argument("--fd-delta", type=float, default=None)
    ap.add_argument("--hours", type=float, default=6.0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--particles", type=int, default=20_000)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--tag", default="")
    ap.add_argument("--init-sigma", action="store_true", help="probe: w' etc. start ~ N(0, σ²)")
    ap.add_argument("--sigma-floor", type=float, default=None, help="probe: σ floor (m/s)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rec = run(a)
    m = rec["metrics"]
    head = {k: rec[k] for k in ("regime", "harness", "n", "variant", "seed")}
    print(json.dumps(head), " ".join(f"{k}={v:.3f}" for k, v in m.items()), f"{rec['wall_s']}s")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "a") as f:
        f.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
