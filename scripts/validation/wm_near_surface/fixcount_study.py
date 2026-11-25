"""Near-surface well-mixed study for the fixed sub-step count (decision 0027).

One run = one regime, harness, outer step, sub-step count, variant and seed.
Appends one JSON line per run to ``--out``.

    PYTHONPATH=src python scripts/validation/wm_near_surface/fixcount_study.py \
        --regime neutral --harness closed --dt 60 --n 8 --seed 1 --out results.jsonl

Set-up (B7's ``tier0._near_surface_error``, PR #67): the production
``HannaScheme`` (dynamic CPU path, backward), horizontally uniform and
motionless meteorology, uniform density in the boundary layer, ``--particles``
particles released uniformly in [0, h], recorded every 15 min from 2 to 6 h.

Harnesses:

* ``closed``: the boundary-layer top is a reflecting lid applied inside every
  sub-step, next to the ground reflection (an engine whose ``reflect_surface``
  also reflects at h). Asks: does the integrator keep a well-mixed column well
  mixed? Exact answer: uniform in [0, h].
* ``open``: no lid. Particles cross h into the free-troposphere coefficients,
  as in production. The free troposphere is stably stratified,
  N = ``--ft-n`` s⁻¹ (θ rising above h; no wind, so K_z is the closure's
  background floor and T_Lw = 0.5/N). No exact answer: compare with the
  fine-step reference.
* ``b7``: B7's harness as written, lid applied once after each outer step.

Variants (``--variant``):

* ``fixed``: production, ``n`` sub-steps for every particle.
* ``clamp0``: production with ``basal_zero_gradient=True`` (∂σ_w²/∂z = 0
  strictly below the basal clamp z_ubl).
* ``adaptive``: per-particle k_i = max(n, ceil(dt / (c · T_Lw,i))), T_Lw at the
  start of the outer step, ``c`` = ``--adapt-c``; refinement only ever adds
  sub-steps. Coefficients re-evaluated every sub-step, as in ``fixed``.
* ``old``: the scheme on the PYTHONPATH with its own defaults (use with the
  pre-0027 source to measure the old code; ``--max-substeps`` sets the cap).

Metrics, averaged over the 17 records (2-6 h), each with a 10-subset standard
error: occupancy of the lowest 5% of the BL and of 0-40 m, 0-10 m and 0-2 m,
each divided by its uniform-in-[0, h] share (1 = well mixed), and the fraction
of particles in the BL. For ``open`` the ``_bl`` variants are conditional on
being in the BL.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from datetime import UTC, datetime, timedelta

import numpy as np
import torch

from lpdm.gpu_engine import GPUEngine
from lpdm.met_reader import HourlyMetTensors, MetFieldMetadata
from lpdm.turbulence import HannaScheme
from lpdm.turbulence import hanna as hanna_mod

REGIMES = {
    # name: (blh m, u* m/s, sensible heat flux W m-2), as B7's WM_REGIMES
    "stable": (400.0, 0.25, -20.0),
    "neutral": (1500.0, 0.4, 0.0),
    "convective": (1500.0, 0.3, 200.0),
}
LAYERS = {"lo5": None, "z40": 40.0, "z10": 10.0, "z2": 2.0}
RECORD_S = list(np.arange(7200.0, 21600.0 + 1, 900.0))
GRAVITY = 9.80665


def met_window(blh_m: float, ustar: float, shf: float, *, ft_n: float | None) -> HourlyMetTensors:
    """B7's ``_hanna_met_window``; with ``ft_n``, T rises above h so that the
    free troposphere has buoyancy frequency ``ft_n`` (pressure is held almost
    constant, so θ ∝ T), on 41 levels to 4 km for a sharper BL top."""
    n_lev = 8 if ft_n is None else 41
    n_lat, n_lon = 4, 4
    shape = (n_lev, n_lat, n_lon)
    top = max(4000.0, 1.5 * blh_m)
    level = np.linspace(0.0, top, n_lev)
    t = torch.full(shape, 280.0)
    if ft_n is not None:
        dtheta_dz = ft_n**2 * 280.0 / GRAVITY
        above = torch.as_tensor(np.clip(level - blh_m, 0.0, None), dtype=torch.float32)
        t = t + (dtheta_dz * above).view(n_lev, 1, 1)
    zeros = torch.zeros(shape, dtype=torch.float32)
    chans = {
        "u": zeros, "v": zeros, "w": zeros,
        "blh": torch.full(shape, blh_m), "sp": torch.full(shape, 101325.0),
        "t": t, "ustar": torch.full(shape, ustar), "shf": torch.full(shape, shf),
    }  # fmt: skip
    names = ("u", "v", "w", "blh", "sp", "t", "ustar", "shf")
    fields = torch.stack([chans[k] for k in names], dim=0)
    t0 = datetime(2024, 1, 1, 12, tzinfo=UTC)
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
        hour_start=fields,
        hour_end=fields,
        metadata=md,
        channel_names=names,
        height_agl_m=height.contiguous(),
    )


class LidEngine(GPUEngine):
    """Ground reflection plus a reflecting lid at ``lid_m`` in every sub-step."""

    def __init__(self, lid_m: float, **kw) -> None:
        super().__init__(**kw)
        self.lid_m = float(lid_m)

    def reflect_surface(self, particles, w_prime, *, z_surface: float = 0.0):
        p, w = super().reflect_surface(particles, w_prime, z_surface=z_surface)
        over = p[:, 2] > self.lid_m
        p = p.clone()
        p[:, 2] = torch.where(over, 2.0 * self.lid_m - p[:, 2], p[:, 2])
        return p, torch.where(over, -w, w)


class AdaptiveHanna(HannaScheme):
    """Per-particle k_i = max(n, ceil(dt / (c · T_Lw,i))) with T_Lw at the start
    of the outer step; every coefficient re-evaluated each sub-step at the current
    height, as in production. Particles that have taken their k_i sub-steps do
    sub_dt = 0 no-ops until the batch maximum."""

    def __init__(self, adapt_c: float, **kw) -> None:
        super().__init__(**kw)
        self.adapt_c = float(adapt_c)
        self.k_hist: list[np.ndarray] = []

    def _integrate_vertical_substeps(self, *, active_xyz, u_prime_in, v_prime_in, w_prime_in,
                                     sigma_u, sigma_v, sigma_w, sigma_w_sq, T_Lu, T_Lv, T_Lw,
                                     half_dsig2_dz_backward, density_drift_backward, dt_seconds,
                                     engine, coeffs_at=None, blh=None, active_mask=None):  # fmt: skip
        assert active_mask is None, "dynamic path only"
        k = torch.ceil(dt_seconds / (self.adapt_c * T_Lw)).clamp(min=self.n_substeps).long()
        self.k_hist.append(np.bincount(k.numpy(), minlength=65)[:65])
        sub_dt = float(dt_seconds) / k.to(T_Lw.dtype)
        xyz, u_p, v_p, w_p = (t.clone() for t in (active_xyz, u_prime_in, v_prime_in, w_prime_in))
        R = hanna_mod.W_PRIME_SIGMA_RATIO_MAX
        for i in range(int(k.max())):
            if i > 0:
                (sigma_u, sigma_v, sigma_w, sigma_w_sq, T_Lu, T_Lv, T_Lw,
                 half_dsig2_dz_backward, density_drift_backward) = coeffs_at(xyz[:, 2])  # fmt: skip
            dt_i = torch.where(k > i, sub_dt, torch.zeros_like(sub_dt))
            drift = (
                1.0 + w_p.pow(2) / sigma_w_sq
            ) * half_dsig2_dz_backward + density_drift_backward
            un = engine.update_langevin_velocity(u_p, t_lagrangian=T_Lu, sigma_w2=sigma_u.pow(2),
                                                 dt_seconds=dt_i)  # fmt: skip
            vn = engine.update_langevin_velocity(v_p, t_lagrangian=T_Lv, sigma_w2=sigma_v.pow(2),
                                                 dt_seconds=dt_i)  # fmt: skip
            wn = engine.update_langevin_velocity(w_p, t_lagrangian=T_Lw, sigma_w2=sigma_w.pow(2),
                                                 dt_seconds=dt_i, drift=drift)  # fmt: skip
            wn = torch.clamp(wn, min=-R * sigma_w, max=R * sigma_w)
            un = torch.clamp(un, min=-R * sigma_u, max=R * sigma_u)
            vn = torch.clamp(vn, min=-R * sigma_v, max=R * sigma_v)
            xyz = engine.apply_vertical_turbulence(xyz, wn, dt_seconds=dt_i, backward=True)
            xyz = engine.apply_horizontal_turbulence(xyz, un, vn, dt_seconds=dt_i, backward=True)
            xyz, wn = engine.reflect_surface(xyz, wn, z_surface=0.0)
            u_p, v_p, w_p = un, vn, wn
        counts = torch.zeros(len(hanna_mod.EXCEEDANCE_COUNT_KEYS), dtype=torch.int64)
        return xyz, u_p, v_p, w_p, counts, torch.zeros(2)


def build_scheme(args):
    kw = dict(meander_enabled=False, static_substeps=False)
    if args.variant == "old":
        if args.max_substeps is not None:
            kw["max_substeps"] = args.max_substeps
        return HannaScheme(**kw)
    kw["n_substeps"] = args.n
    if args.variant == "fixed":
        return HannaScheme(**kw)
    if args.variant == "clamp0":
        return HannaScheme(basal_zero_gradient=True, **kw)
    if args.variant == "fixed_clampoff":
        return HannaScheme(basal_zero_gradient=False, **kw)
    if args.variant == "adaptive":
        return AdaptiveHanna(args.adapt_c, **kw)
    raise SystemExit(f"unknown variant {args.variant}")


def metrics(zs: np.ndarray, blh: float, idx=None) -> dict[str, float]:
    z = zs if idx is None else zs[:, idx]
    out = {"frac_bl": float((z <= blh).mean())}
    in_bl = z <= blh
    for name, top in LAYERS.items():
        top = 0.05 * blh if top is None else top
        share = top / blh
        below = z < top
        out[name] = float(below.mean() / share)
        out[name + "_bl"] = float(below.sum() / max(in_bl.sum(), 1) / share)
    return out


def run(args) -> dict:
    blh, ustar, shf = REGIMES[args.regime]
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    met = met_window(blh, ustar, shf, ft_n=args.ft_n if args.harness == "open" else None)
    engine = LidEngine(blh, device="cpu") if args.harness == "closed" else GPUEngine(device="cpu")
    scheme = build_scheme(args)
    n = args.particles
    p = torch.zeros(n, 4, dtype=torch.float32)
    p[:, 1] = 45.0
    p[:, 2] = torch.rand(n, dtype=torch.float32) * blh
    p[:, 3] = 1.0 / n
    state = scheme.initialize_state(n, device=torch.device("cpu"), dtype=torch.float32)
    active = torch.ones(n, dtype=torch.bool)
    zs = []
    t, steps = 0.0, 0
    t0 = time.time()
    for t_rec in RECORD_S:
        while t < t_rec - 1e-9:
            p, state = scheme.step(p, state, met, t_alpha=0.5, dt_seconds=args.dt,
                                   active_mask=active, engine=engine)  # fmt: skip
            if args.harness == "b7":
                above = p[:, 2] > blh
                p[above, 2] = 2.0 * blh - p[above, 2]
                state["w_prime"][above] = -state["w_prime"][above]
            t += args.dt
            steps += 1
        zs.append(p[:, 2].numpy().copy())
    zs = np.stack(zs)  # [record, particle]
    m = metrics(zs, blh)
    perm = np.random.default_rng(0).permutation(n)
    subsets = [metrics(zs, blh, idx) for idx in np.array_split(perm, 10)]
    se = {k: float(np.std([s[k] for s in subsets], ddof=1) / math.sqrt(10)) for k in m}
    hi = 1.5 * blh if args.harness == "open" else blh
    prof = np.histogram(zs.ravel(), bins=30, range=(0.0, hi))[0] / zs.size * 30 * hi / blh
    taken = getattr(scheme, "take_substep_exceedance", lambda _w: None)(None)
    rec = dict(
        regime=args.regime, harness=args.harness, dt=args.dt,
        n=None if args.variant == "old" else args.n, variant=args.variant,
        adapt_c=args.adapt_c if args.variant == "adaptive" else None,
        ft_n=args.ft_n if args.harness == "open" else None,
        seed=args.seed, particles=n, steps=steps, tag=args.tag,
        src=os.path.dirname(hanna_mod.__file__),
        metrics=m, se=se, profile30=[round(float(x), 4) for x in prof], profile_top_m=hi,
        exceedance=taken, wall_s=round(time.time() - t0, 1),
    )  # fmt: skip
    if isinstance(scheme, AdaptiveHanna):
        rec["adaptive_k_hist"] = np.sum(scheme.k_hist, axis=0).tolist()
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", choices=sorted(REGIMES), required=True)
    ap.add_argument("--harness", choices=["closed", "open", "b7"], default="closed")
    ap.add_argument("--dt", type=float, default=60.0)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--variant", default="fixed")
    ap.add_argument("--adapt-c", type=float, default=0.125)
    ap.add_argument("--max-substeps", type=int, default=None)
    ap.add_argument("--ft-n", type=float, default=0.015)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--particles", type=int, default=20_000)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rec = run(args)
    line = json.dumps(rec)
    head = json.dumps({k: rec[k] for k in ("regime", "harness", "dt", "n", "variant", "seed")})
    m = rec["metrics"]
    print(f"{head} lo5={m['lo5']:.3f} z40={m['z40']:.3f} wall={rec['wall_s']}s")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "a") as f:
        f.write(line + "\n")


if __name__ == "__main__":
    main()
