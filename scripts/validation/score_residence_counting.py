#!/usr/bin/env python
"""Fine-time residence scoring of the footprint counting rules (task A2, decision 0029).

The footprint approximates, for each particle and step, the time integral
``∫ 1[x(t) ∈ cell] w / ρ(x(t), t) dt`` over the step from a few positions. This
script measures the error of each counting rule (``endpoint``, ``step_mean``,
``trapezoid``; ``main._deposit_step``) separately from the trajectory error: it
scores every rule and a fine-time reference on **the same trajectories**.

**Trajectories.** One outer step of the production model is: RK2 advection over the
whole step, then ``n`` turbulent sub-steps (``HannaScheme._integrate_vertical_substeps``,
n = 16 by default), then the meander displacement and, once per met window,
convection. The scorer records the start ``x0``, the advected position ``xa``, the
position after each sub-step ``s_i`` and the scheme's output ``xm`` (after meander),
and reconstructs the path at the sub-step times t_i = t0 - i Δ/n as

    P_i = x0 + (i/n) (xa - x0) + (s_i - xa) + (i/n) (xm - s_n),     i = 0..n,

that is, the mean-wind advection and the meander spread uniformly over the step and
the turbulent displacement at its sub-step resolution (the operator splitting
applies them in sequence; their sum is the same at the step's ends: P_0 = x0,
P_n = xm). Heights are clamped at 0. Convection is a jump at the end of the step
(the rules see the post-convection end ``x1``; the path ends at ``xm``).

**Reference.** Between the nodes the path is linear. The reference is the midpoint
rule on that path with ``m`` samples per sub-step (time step Δ/(n m)); each sample
is binned by its own position and age, and weighted by 1/ρ interpolated linearly
between the nodes (1/ρ evaluated with the production density code at each node's
position and time). The convergence check reports the same tallies for
m = 1, 2, 4 against the finest m, and the composite trapezoid and right-rectangle
rules on the nodes at spacings of k = 1, 2, 4, 8 sub-steps (k = n is the trapezoid
and endpoint rules themselves).

**Rules**, exactly as `main._deposit_step` counts them: endpoint (Δ at x1, density
at x1 and the step's mid-time, age bin of the step's start), step_mean (Δ at the
mean of x0 and x1, density there at mid-time), trapezoid (Δ/2 at x0, density at the
step's start time, age bin of the start; Δ/2 at x1, density at the step's end time,
age bin of the age half a step later).

**Tallies.** Each estimator accumulates, density-weighted and not, (a) the
time-integrated (z, y, x) grid and (b) per hourly time-ago bin, the totals by
z-bin and by distance class from the receptor's cell (Chebyshev rings 0, 1, 2-3,
4-10, > 10 cells). ``summarise`` turns them into relative errors against the
reference.

Subcommands:

* ``config``: hook a production run (CPU, dynamic path) of a GLIDE config.
* ``column``: the production Hanna step in a horizontally uniform synthetic
  boundary layer, a point release with a uniform wind (plume) or an equilibrated
  column (well-mixed).
* ``plume``: the analytic Gaussian plume (homogeneous OU, as tests/test_plume_footprint.py)
  with n fine OU steps per outer step; also scores every estimator against the
  closed form.

Outputs ``<out>.json`` (summary) and ``<out>.npz`` (tallies).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

RULES = ("endpoint", "step_mean", "trapezoid")
RING_EDGES = ((0, 0), (1, 1), (2, 3), (4, 10), (11, 1_000_000))
RING_NAMES = ("ring0", "ring1", "ring2-3", "ring4-10", "ring11+")


# --------------------------------------------------------------------------- tallies


@dataclass
class Grid:
    lon_bounds: tuple[float, float]
    lat_bounds: tuple[float, float]
    n_x: int
    n_y: int
    z_edges: tuple[float, ...]
    receptor_lon: float
    receptor_lat: float
    n_t: int = 25
    bin_s: float = 3600.0

    def __post_init__(self) -> None:
        self.n_z = len(self.z_edges) - 1
        self.ix0, self.iy0 = self.cell_of(self.receptor_lon, self.receptor_lat)

    def cell_of(self, lon: float, lat: float) -> tuple[int, int]:
        ix = math.floor(
            (lon - self.lon_bounds[0]) / (self.lon_bounds[1] - self.lon_bounds[0]) * self.n_x
        )
        iy = math.floor(
            (lat - self.lat_bounds[0]) / (self.lat_bounds[1] - self.lat_bounds[0]) * self.n_y
        )
        return ix, iy


@dataclass
class Tallies:
    grid: Grid
    names: list[str]
    weighted: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        g = self.grid
        self.z_edges_t = torch.tensor(g.z_edges, dtype=torch.float64)
        ring_of = np.zeros(2_000_001, dtype=np.int64)
        for k, (lo, hi) in enumerate(RING_EDGES):
            ring_of[lo : min(hi, 2_000_000) + 1] = k
        self.ring_lut = torch.as_tensor(ring_of)
        self.n_ring = len(RING_EDGES)
        self.data = {}
        for wkey in ("w", "u"):  # density-weighted, unweighted
            for name in self.names:
                self.data[(wkey, name, "grid")] = torch.zeros(
                    (g.n_z, g.n_y, g.n_x), dtype=torch.float64
                )
                self.data[(wkey, name, "tz")] = torch.zeros((g.n_t, g.n_z), dtype=torch.float64)
                self.data[(wkey, name, "tr")] = torch.zeros(
                    (g.n_t, g.n_z, self.n_ring), dtype=torch.float64
                )

    def add(
        self,
        name: str,
        xyz: torch.Tensor,
        w_plain: torch.Tensor,
        inv_rho: torch.Tensor | None,
        age_s: torch.Tensor,
    ) -> None:
        """Add samples: positions (K, 3), plain weights w*dt (K), 1/rho (K) or None, age (K)."""

        g = self.grid
        xyz = xyz.to(torch.float64)
        ix = torch.floor(
            (xyz[:, 0] - g.lon_bounds[0]) / (g.lon_bounds[1] - g.lon_bounds[0]) * g.n_x
        ).long()
        iy = torch.floor(
            (xyz[:, 1] - g.lat_bounds[0]) / (g.lat_bounds[1] - g.lat_bounds[0]) * g.n_y
        ).long()
        iz = torch.bucketize(xyz[:, 2].contiguous(), self.z_edges_t, right=False) - 1
        it = torch.floor(age_s.to(torch.float64) / g.bin_s).long()
        ok = (
            (ix >= 0)
            & (ix < g.n_x)
            & (iy >= 0)
            & (iy < g.n_y)
            & (iz >= 0)
            & (iz < g.n_z)
            & (it >= 0)
            & (it < g.n_t)
        )
        ix, iy, iz, it = ix[ok], iy[ok], iz[ok], it[ok]
        ring = torch.maximum((ix - g.ix0).abs(), (iy - g.iy0).abs()).clamp(max=2_000_000)
        ir = self.ring_lut[ring]
        flat_g = (iz * g.n_y + iy) * g.n_x + ix
        flat_tz = it * g.n_z + iz
        flat_tr = (it * g.n_z + iz) * self.n_ring + ir
        wp = w_plain.to(torch.float64)[ok]
        weights = {"u": wp}
        weights["w"] = wp * inv_rho.to(torch.float64)[ok] if inv_rho is not None else wp
        for wkey, w in weights.items():
            self.data[(wkey, name, "grid")].view(-1).index_add_(0, flat_g, w)
            self.data[(wkey, name, "tz")].view(-1).index_add_(0, flat_tz, w)
            self.data[(wkey, name, "tr")].view(-1).index_add_(0, flat_tr, w)

    def save(self, path: Path) -> None:
        arrays = {f"{w}|{n}|{k}": v.numpy() for (w, n, k), v in self.data.items()}
        arrays["receptor_cell"] = np.array([self.grid.ix0, self.grid.iy0])
        np.savez_compressed(path, **arrays)


def estimator_names(m_levels: tuple[int, ...], k_levels: tuple[int, ...]) -> list[str]:
    names = list(RULES)
    names += [f"ref_m{m}" for m in m_levels]
    names += [f"trap_k{k}" for k in k_levels] + [f"rect_k{k}" for k in k_levels]
    return names


def score_step(
    T: Tallies,
    *,
    P: torch.Tensor,
    x1: torch.Tensor,
    w: torch.Tensor,
    age0: torch.Tensor,
    delta_s: float,
    inv_rho_at,
    t0_s: float,
    m_levels: tuple[int, ...],
    k_levels: tuple[int, ...],
) -> None:
    """Score one outer step for one set of particles.

    P: (n+1, N, 3) path nodes at t0 - iΔ/n (float64); x1: (N, 3) end after
    convection; w: (N,) particle weights; age0: (N,) age at the step's start (s);
    inv_rho_at(xyz, t_s) -> (N,) 1/rho, or None for no density.
    """

    n = P.shape[0] - 1
    x0 = P[0]
    half = 0.5 * delta_s
    t_mid = t0_s - half

    def inv(xyz, t_s):
        return None if inv_rho_at is None else inv_rho_at(xyz, t_s)

    # The three rules, as main._deposit_step.
    T.add("endpoint", x1, w * delta_s, inv(x1, t_mid), age0)
    d = x1 - x0
    d_lon = torch.remainder(d[:, 0:1] + 180.0, 360.0) - 180.0
    xm = x0 + 0.5 * torch.cat((d_lon, d[:, 1:]), dim=1)
    T.add("step_mean", xm, w * delta_s, inv(xm, t_mid), age0)
    T.add("trapezoid", x0, w * half, inv(x0, t0_s), age0)
    T.add("trapezoid", x1, w * half, inv(x1, t0_s - delta_s), age0 + half)

    # 1/rho at every node, at its own time.
    if inv_rho_at is None:
        inv_nodes = None
    else:
        inv_nodes = torch.stack([inv_rho_at(P[i], t0_s - i * delta_s / n) for i in range(n + 1)])
    sub = delta_s / n

    # Reference family: midpoint rule with m samples per sub-step on the linear path.
    for m in m_levels:
        fr = (torch.arange(m, dtype=torch.float64) + 0.5) / m  # (m,)
        a = P[:-1].unsqueeze(1)  # (n, 1, N, 3)
        b = P[1:].unsqueeze(1)
        f = fr.view(1, m, 1, 1)
        pos = (a + f * (b - a)).reshape(-1, 3)
        i_idx = torch.arange(n, dtype=torch.float64).view(n, 1, 1)
        age = (age0.view(1, 1, -1) + (i_idx + fr.view(1, m, 1)) * sub).reshape(-1)
        ww = (w.view(1, 1, -1) * (sub / m)).expand(n, m, -1).reshape(-1)
        if inv_nodes is None:
            ir = None
        else:
            ia = inv_nodes[:-1].unsqueeze(1)
            ib = inv_nodes[1:].unsqueeze(1)
            ir = (ia + fr.view(1, m, 1) * (ib - ia)).reshape(-1)
        T.add(f"ref_m{m}", pos, ww, ir, age)

    # Node rules at spacing k sub-steps (k = n would be the rules themselves).
    for k in k_levels:
        idx = list(range(0, n + 1, k))
        for a_i, b_i in zip(idx[:-1], idx[1:]):
            age_a = age0 + a_i * sub
            ia = None if inv_nodes is None else inv_nodes[a_i]
            ib = None if inv_nodes is None else inv_nodes[b_i]
            T.add(f"trap_k{k}", P[a_i], w * (0.5 * k * sub), ia, age_a)
            T.add(f"trap_k{k}", P[b_i], w * (0.5 * k * sub), ib, age_a + 0.5 * k * sub)
            T.add(f"rect_k{k}", P[b_i], w * (k * sub), ib, age_a)


def build_path(x0, xa, subs, xm) -> torch.Tensor:
    """P_i = x0 + (i/n)(xa - x0) + (s_i - xa) + (i/n)(xm - s_n), heights clamped at 0."""

    n = len(subs)
    x0 = x0.to(torch.float64)
    xa = xa.to(torch.float64)
    xm = xm.to(torch.float64)
    s = torch.stack([xa] + [t.to(torch.float64) for t in subs])  # (n+1, N, 3)
    frac = (torch.arange(n + 1, dtype=torch.float64) / n).view(-1, 1, 1)
    P = (
        x0.unsqueeze(0)
        + frac * (xa - x0).unsqueeze(0)
        + (s - xa.unsqueeze(0))
        + frac * (xm - s[-1]).unsqueeze(0)
    )
    P[..., 2] = P[..., 2].clamp(min=0.0)
    return P


# --------------------------------------------------------------------------- summary


def _regions(T: Tallies, wkey: str, name: str) -> tuple[dict, torch.Tensor]:
    """Region totals of one estimator: whole grid, 0-40 m and 40-1000 m, 0-40 m by
    distance class from the receptor's cell, 0-40 m by hourly time-ago bin."""

    g = T.grid
    gr, tz, tr = (T.data[(wkey, name, k)] for k in ("grid", "tz", "tr"))
    reg = {
        "total": float(gr.sum()),
        "z0_total": float(gr[0].sum()),
        "z1_total": float(gr[1].sum()) if g.n_z > 1 else float("nan"),
    }
    for k, rn in enumerate(RING_NAMES):
        reg[f"z0_{rn}"] = float(tr[:, 0, k].sum())
    for t in range(min(g.n_t, 6)):
        reg[f"z0_bin{t}"] = float(tz[t, 0])
    reg["z0_bin6+"] = float(tz[6:, 0].sum())
    reg["z1_bin0"] = float(tz[0, 1]) if g.n_z > 1 else float("nan")
    return reg, gr


def summarise(T: Tallies, ref: str) -> dict:
    """Relative errors (estimator - ref) / ref for each region, both weightings, and
    the cell-wise L1 error of the 0-40 m grid relative to its total."""

    out: dict = {}
    for wkey in ("w", "u"):
        ref_reg, ref_grid = _regions(T, wkey, ref)
        block = {"ref": ref_reg, "rel": {}, "l1_z0": {}}
        for name in T.names:
            reg, gr = _regions(T, wkey, name)
            block["rel"][name] = {
                k: (reg[k] - ref_reg[k]) / ref_reg[k] if ref_reg[k] else float("nan") for k in reg
            }
            block["l1_z0"][name] = float((gr[0] - ref_grid[0]).abs().sum() / ref_grid[0].sum())
        out[wkey] = block
    return out


# --------------------------------------------------------------------------- config (production run)


def run_config(args) -> None:
    import logging

    import lpdm.main as M
    from lpdm.air_density import ParticleDryAirDensity
    from lpdm.config import RunConfig
    from lpdm.footprint_gridder import FootprintGridder
    from lpdm.turbulence.hanna import HannaScheme

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    cfg = RunConfig.from_yaml(args.config).with_overrides(device="cpu", output_uri=args.output_uri)
    if cfg.output_grid.deposition.position != "endpoint":
        raise SystemExit("run the scorer on an endpoint config (the rules are scored offline)")
    og = cfg.output_grid
    rel = cfg.release
    pt = getattr(rel, "point", None)
    rlon, rlat = (pt.lon, pt.lat) if pt is not None else (rel.lon, rel.lat)
    grid = Grid(
        lon_bounds=tuple(og.lon_bounds),
        lat_bounds=tuple(og.lat_bounds),
        n_x=og.n_x,
        n_y=og.n_y,
        z_edges=tuple(og.z_edges_m),
        receptor_lon=rlon,
        receptor_lat=rlat,
        n_t=args.n_t,
        bin_s=3600.0,
    )
    m_levels = tuple(args.m_levels)
    k_levels = tuple(args.k_levels)
    T = Tallies(grid, estimator_names(m_levels, k_levels))
    density = (
        ParticleDryAirDensity("cpu", torch.float32) if og.density_weighting == "dry_air" else None
    )

    rec: dict = {}
    orig_gen = M.generate_batch_particles

    def gen_hook(batch, **kw):
        pb = orig_gen(batch, **kw)
        rec["batch"] = pb
        return pb

    orig_adv = M._advect_active_particles

    def adv_hook(
        engine, device, active_particles, met_window, t_cursor, delta_s, dtype, wind_mean_cache=None
    ):
        out = orig_adv(
            engine,
            device,
            active_particles,
            met_window,
            t_cursor,
            delta_s,
            dtype,
            wind_mean_cache=wind_mean_cache,
        )
        rec.update(
            x0=active_particles[:, :3].clone(),
            xa=out[0][:, :3].clone(),
            t_cursor=t_cursor,
            delta_s=float(delta_s),
            met_window=met_window,
            engine=engine,
        )
        return out

    orig_int = HannaScheme._integrate_vertical_substeps

    def int_hook(self, **kw):
        engine = kw["engine"]
        subs: list[torch.Tensor] = []
        orig_reflect = engine.reflect_surface

        def reflect_rec(xyz, w_prime, *, z_surface=0.0):
            out = orig_reflect(xyz, w_prime, z_surface=z_surface)
            subs.append(out[0][:, :3].clone())
            return out

        engine.reflect_surface = reflect_rec
        try:
            res = orig_int(self, **kw)
        finally:
            del engine.reflect_surface
        rec["subs"] = subs
        return res

    orig_step = HannaScheme.step

    def step_hook(self, particles, state, met_window, t_alpha, dt_seconds, active_mask, engine):
        out = orig_step(
            self, particles, state, met_window, t_alpha, dt_seconds, active_mask, engine
        )
        rec["xm"] = out[0][active_mask][:, :3].clone()
        return out

    orig_acc = FootprintGridder.accumulate
    stats = {"steps": 0, "particle_steps": 0, "t_score": 0.0}

    def acc_hook(
        self, particles, active_mask, weights, t_idx, release_idx, dt_seconds, inv_density=None
    ):
        orig_acc(self, particles, active_mask, weights, t_idx, release_idx, dt_seconds, inv_density)
        t1 = time.perf_counter()
        pb = rec["batch"]
        x1 = particles[active_mask]
        n_act = x1.shape[0]
        if n_act == 0:
            return
        assert rec["x0"].shape[0] == n_act and rec["xm"].shape[0] == n_act
        P = build_path(rec["x0"], rec["xa"], rec["subs"], rec["xm"])
        t0_s = rec["t_cursor"].timestamp()
        cursor_off = t0_s - pb.batch_start_time.timestamp()
        age0 = (
            (pb.release_window_end_offsets_s - cursor_off)
            .clamp(min=0.0)[active_mask]
            .to(torch.float64)
        )
        w = weights[active_mask].to(torch.float64)
        mw, engine = rec["met_window"], rec["engine"]

        if density is None:
            inv_rho_at = None
        else:

            def inv_rho_at(xyz, t_s):
                a = M._time_weight(mw, t_s)
                return density.inverse_density(xyz.to(torch.float32), mw, a, engine).to(
                    torch.float64
                )

        score_step(
            T,
            P=P,
            x1=x1.to(torch.float64),
            w=w,
            age0=age0,
            delta_s=float(dt_seconds),
            inv_rho_at=inv_rho_at,
            t0_s=t0_s,
            m_levels=m_levels,
            k_levels=k_levels,
        )
        stats["steps"] += 1
        stats["particle_steps"] += n_act
        stats["t_score"] += time.perf_counter() - t1
        for k in ("x0", "xa", "subs", "xm"):
            rec.pop(k, None)

    M.generate_batch_particles = gen_hook
    M._advect_active_particles = adv_hook
    HannaScheme._integrate_vertical_substeps = int_hook
    HannaScheme.step = step_hook
    FootprintGridder.accumulate = acc_hook
    t_start = time.perf_counter()
    meta = M._run(cfg)
    stats["wall_s"] = time.perf_counter() - t_start
    finish(
        T,
        args,
        {
            "kind": "config",
            "config": str(args.config),
            "stats": stats,
            "status": meta["runtime"]["status"],
        },
    )


# --------------------------------------------------------------------------- synthetic


M_PER_DEG_LON0 = 111320.0
M_PER_DEG_LAT = 110540.0


def column_met_window(blh_m: float, ustar: float, shf: float, n_lev: int = 13):
    from datetime import UTC, datetime, timedelta

    from lpdm.met_reader import HourlyMetTensors, MetFieldMetadata

    shape = (n_lev, 4, 4)
    zeros = torch.zeros(shape, dtype=torch.float32)
    chans = {
        "u": zeros,
        "v": zeros,
        "w": zeros,
        "blh": torch.full(shape, blh_m),
        "sp": torch.full(shape, 101325.0),
        "t": torch.full(shape, 280.0),
        "q": torch.full(shape, 0.004),
        "ustar": torch.full(shape, ustar),
        "shf": torch.full(shape, shf),
    }
    names = tuple(chans)
    fields = torch.stack([chans[k] for k in names], dim=0)
    t0 = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)
    level = np.concatenate(
        [
            [0.0, 10.0, 25.0, 50.0, 100.0, 200.0],
            np.linspace(400.0, max(6000.0, 2 * blh_m), n_lev - 6),
        ]
    )
    metadata = MetFieldMetadata(
        lon=np.linspace(-20.0, 20.0, 4),
        lat=np.linspace(35.0, 55.0, 4),
        level=level,
        pressure_level_hpa=np.linspace(1000.0, 450.0, n_lev),
        time_start=t0,
        time_end=t0 + timedelta(hours=1),
        variable_units={
            "u": "m/s",
            "v": "m/s",
            "w": "m/s",
            "blh": "m",
            "sp": "Pa",
            "t": "K",
            "q": "kg kg**-1",
            "z": "m**2s**-2",
            "z_sfc": "m**2s**-2",
            "ustar": "m/s",
            "shf": "W m**-2",
        },
    )
    height = (
        torch.as_tensor(level, dtype=torch.float32).view(n_lev, 1, 1).expand(shape).contiguous()
    )
    return HourlyMetTensors(
        hour_start=fields,
        hour_end=fields,
        metadata=metadata,
        channel_names=names,
        height_agl_m=height,
    )


def run_column(args) -> None:
    from lpdm.air_density import ParticleDryAirDensity
    from lpdm.gpu_engine import GPUEngine
    from lpdm.main import _time_weight
    from lpdm.turbulence.hanna import HannaScheme

    torch.manual_seed(args.seed)
    engine = GPUEngine(device="cpu")
    scheme = HannaScheme(meander_enabled=False, static_substeps=False, n_substeps=args.n_sub)
    met = column_met_window(args.blh, args.ustar, args.shf)
    t0_s = met.metadata.time_end.timestamp()
    density = ParticleDryAirDensity("cpu", torch.float32)
    lat0 = 45.0
    m_per_deg_lon = M_PER_DEG_LON0 * math.cos(math.radians(lat0))
    dx = args.cell_m / m_per_deg_lon
    dy = args.cell_m / M_PER_DEG_LAT
    nx, ny = args.n_cells, args.n_cells
    # Receptor at the centre of the central cell.
    grid = Grid(
        lon_bounds=(-dx * nx / 2, dx * nx / 2),
        lat_bounds=(lat0 - dy * ny / 2, lat0 + dy * ny / 2),
        n_x=nx,
        n_y=ny,
        z_edges=(0.0, 40.0, 1000.0, 5000.0),
        receptor_lon=0.0,
        receptor_lat=lat0,
        n_t=args.n_t,
    )
    m_levels = tuple(args.m_levels)
    k_levels = tuple(args.k_levels)
    T = Tallies(grid, estimator_names(m_levels, k_levels))
    n = args.n_particles
    p = torch.zeros(n, 4)
    p[:, 1] = lat0
    p[:, 3] = 1.0 / n
    if args.mode == "plume":
        p[:, 2] = args.release_z
        spin_steps = 0
    else:
        p[:, 2] = torch.rand(n) * args.blh
        spin_steps = int(round(args.spin_s / args.dt))
    state = scheme.initialize_state(n, device=torch.device("cpu"), dtype=torch.float32)
    active = torch.ones(n, dtype=torch.bool)
    du_deg = args.wind * args.dt / m_per_deg_lon  # backward: upwind is -x for a wind towards +x

    rec: dict = {}
    orig_int = HannaScheme._integrate_vertical_substeps

    def int_hook(self, **kw):
        subs: list[torch.Tensor] = []
        orig_reflect = engine.reflect_surface

        def reflect_rec(xyz, w_prime, *, z_surface=0.0):
            out = orig_reflect(xyz, w_prime, z_surface=z_surface)
            subs.append(out[0][:, :3].clone())
            return out

        engine.reflect_surface = reflect_rec
        try:
            res = orig_int(self, **kw)
        finally:
            del engine.reflect_surface
        rec["subs"] = subs
        return res

    HannaScheme._integrate_vertical_substeps = int_hook
    n_steps = int(round(args.length_s / args.dt))
    t_wall = time.perf_counter()
    for step in range(spin_steps + n_steps):
        x0 = p[:, :3].clone()
        p[:, 0] -= du_deg  # mean-wind advection (uniform, exact)
        xa = p[:, :3].clone()
        p, state = scheme.step(
            p, state, met, t_alpha=0.5, dt_seconds=args.dt, active_mask=active, engine=engine
        )
        if step < spin_steps:
            continue
        if args.mode == "column":
            # Horizontally uniform: keep every particle in the receptor's column.
            pass
        k = step - spin_steps
        t_cur = t0_s - (k * args.dt) % 3600.0  # stays within the window (fields are constant)

        def inv_rho_at(xyz, t_s):
            a = _time_weight(met, max(t_s, t0_s - 3600.0))
            return density.inverse_density(xyz.to(torch.float32), met, a, engine).to(torch.float64)

        P = build_path(x0, xa, rec["subs"], p[:, :3])
        if args.mode == "column":
            P[..., 0] = 0.0
            P[..., 1] = lat0
        x1 = P[-1].clone()
        score_step(
            T,
            P=P,
            x1=x1,
            w=p[:, 3].to(torch.float64),
            age0=torch.full((n,), k * args.dt, dtype=torch.float64),
            delta_s=args.dt,
            inv_rho_at=inv_rho_at,
            t0_s=t_cur,
            m_levels=m_levels,
            k_levels=k_levels,
        )
    finish(T, args, {"kind": "column", "mode": args.mode, "wall_s": time.perf_counter() - t_wall})


def run_plume(args) -> None:
    """Analytic plume: homogeneous OU (sigma_v = sigma_w, T_L), uniform wind U, release
    at z_r, ground reflection, u' = 0 (as tests/test_plume_footprint.py), n fine OU
    steps of Δ/n per outer step."""

    from scipy.special import ndtr

    from lpdm.gpu_engine import GPUEngine

    U, sig2, TL, zr, h = 5.0, 0.25, 100.0, 50.0, 40.0
    torch.manual_seed(args.seed)
    engine = GPUEngine(device="cpu", dtype=torch.float64)
    DX, DY, NX, NY = 500.0, 50.0, 41, 100
    # The receptor's cell is [-f DX, (1 - f) DX] with f = --receptor-frac. f = 1 is the
    # geometry of tests/test_plume_footprint.py (receptor on the cell's east edge,
    # every cell edge on an outer-step time, since DX / U = 100 s); the default 0.63
    # puts no edge on a node time (63 s + 100 s j is not a multiple of any sub-step).
    f = args.receptor_frac
    x_lo = -(NX - 1 + f) * DX
    grid = Grid(
        lon_bounds=(x_lo / M_PER_DEG_LON0, (x_lo + NX * DX) / M_PER_DEG_LON0),
        lat_bounds=(-NY * DY / 2 / M_PER_DEG_LAT, NY * DY / 2 / M_PER_DEG_LAT),
        n_x=NX,
        n_y=NY,
        z_edges=(0.0, h, 10000.0),
        receptor_lon=0.0,
        receptor_lat=0.0,
        n_t=1,
        bin_s=1e9,
    )
    m_levels = tuple(args.m_levels)
    k_levels = tuple(args.k_levels)
    T = Tallies(grid, estimator_names(m_levels, k_levels))
    n = args.n_particles
    p = torch.zeros((n, 4), dtype=torch.float64)
    p[:, 2] = zr
    p[:, 3] = 1.0 / n
    sigma = math.sqrt(sig2)
    v = torch.randn(n, dtype=torch.float64) * sigma
    wv = torch.randn(n, dtype=torch.float64) * sigma
    u0 = torch.zeros(n, dtype=torch.float64)
    dt, nsub = args.dt, args.n_sub
    sub = dt / nsub
    length = args.length_s
    for k in range(int(round(length / dt))):
        nodes = [p[:, :3].clone()]
        for _ in range(nsub):
            p[:, 0] -= U * sub / M_PER_DEG_LON0
            v = engine.update_langevin_velocity(v, t_lagrangian=TL, sigma_w2=sig2, dt_seconds=sub)
            wv = engine.update_langevin_velocity(wv, t_lagrangian=TL, sigma_w2=sig2, dt_seconds=sub)
            p = engine.apply_horizontal_turbulence(p, u0, v, dt_seconds=sub, backward=True)
            p = engine.apply_vertical_turbulence(p, wv, dt_seconds=sub, backward=True)
            p, wv = engine.reflect_surface(p, wv, z_surface=0.0)
            nodes.append(p[:, :3].clone())
        P = torch.stack(nodes)
        score_step(
            T,
            P=P,
            x1=P[-1].clone(),
            w=p[:, 3],
            age0=torch.full((n,), k * dt, dtype=torch.float64),
            delta_s=dt,
            inv_rho_at=None,
            t0_s=0.0,
            m_levels=m_levels,
            k_levels=k_levels,
        )

    # Closed form: crosswind-integrated surface residence per along-wind cell.
    def taylor(t):
        return math.sqrt(2.0 * sig2 * TL * (t - TL * (1.0 - math.exp(-t / TL))))

    edges = x_lo + np.arange(NX + 1) * DX
    analytic = np.zeros(NX)
    for j in range(NX):
        lo, hi = edges[j], min(edges[j + 1], 0.0)
        if lo >= 0.0 or hi <= -U * length:
            continue
        lo = max(lo, -U * length)
        xs = np.linspace(lo, hi, 2001)
        sz = np.maximum(np.array([taylor(max(-x / U, 1e-6)) for x in xs]), 1e-9)
        pz = (ndtr((h - zr) / sz) - ndtr(-zr / sz)) + (ndtr((h + zr) / sz) - ndtr(zr / sz))
        analytic[j] = np.trapezoid(pz, xs) / U
    cols = {}
    for (wk, name, kind), arr in T.data.items():
        if wk == "u" and kind == "grid":
            cols[name] = arr[0].sum(dim=0).numpy()  # crosswind-integrated, 0-h layer
    # Columns fully covered: travel time to the cell's far edge within the run.
    lower = edges[:-1]
    ok = (analytic > 0) & (lower >= -U * length)
    near = ok & (edges[1:] > -2 * TL * U)  # within 2 T_L of the receptor
    far = ok & ~near
    rc = int(np.flatnonzero(ok)[-1])  # the receptor's (for f = 1, the last covered) column
    vs = {}
    for name, c in cols.items():
        vs[name] = {
            "mre_vs_analytic_all": float(np.mean(np.abs(c[ok] - analytic[ok]) / analytic[ok])),
            "mre_vs_analytic_near": float(
                np.mean(np.abs(c[near] - analytic[near]) / analytic[near])
            ),
            "mre_vs_analytic_far": float(np.mean(np.abs(c[far] - analytic[far]) / analytic[far])),
            "receptor_col_vs_analytic": float((c[rc] - analytic[rc]) / analytic[rc]),
        }
    ref = f"ref_m{max(m_levels)}"
    for name, c in cols.items():
        r = cols[ref]
        vs[name]["mre_vs_ref_all"] = float(np.mean(np.abs(c[ok] - r[ok]) / r[ok]))
        vs[name]["mre_vs_ref_near"] = float(np.mean(np.abs(c[near] - r[near]) / r[near]))
        vs[name]["mre_vs_ref_far"] = float(np.mean(np.abs(c[far] - r[far]) / r[far]))
        vs[name]["receptor_col_vs_ref"] = float((c[rc] - r[rc]) / r[rc])
    finish(
        T,
        args,
        {
            "kind": "plume",
            "columns": {k: v.tolist() for k, v in cols.items()},
            "analytic": analytic.tolist(),
            "plume_scores": vs,
        },
    )


# --------------------------------------------------------------------------- main


def finish(T: Tallies, args, extra: dict) -> None:
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    ref = f"ref_m{max(args.m_levels)}"
    summary = {
        "args": {
            k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items() if k != "func"
        },
        "reference": ref,
        **extra,
    }
    summary["scores"] = summarise(T, ref)
    T.save(Path(f"{out}.npz"))
    Path(f"{out}.json").write_text(json.dumps(summary, indent=1))
    s = summary["scores"]["w"]["rel"]
    keys = (
        "z0_ring0",
        "z0_ring1",
        "z0_ring2-3",
        "z0_ring4-10",
        "z0_ring11+",
        "z0_total",
        "z1_total",
        "total",
    )
    print(f"reference {ref}; relative error vs reference (density-weighted):")
    print("estimator      " + " ".join(f"{k:>11s}" for k in keys))
    for name, rel in s.items():
        print(f"{name:14s} " + " ".join(f"{rel[k]:+11.4f}" for k in keys))
    print(json.dumps({k: v for k, v in extra.items() if k in ("stats", "wall_s")}))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sp = ap.add_subparsers(required=True)

    def common(p):
        p.add_argument("--out", required=True, help="output path stem (.json, .npz)")
        p.add_argument("--m-levels", type=int, nargs="+", default=[1, 2, 4, 8])
        p.add_argument("--k-levels", type=int, nargs="+", default=[1, 2, 4, 8])
        p.add_argument("--n-t", type=int, default=25, help="hourly time-ago bins scored")

    p = sp.add_parser("config", help="hook a production run (CPU) of a GLIDE config")
    p.add_argument("config", type=Path)
    p.add_argument("--output-uri", default=None)
    common(p)
    p.set_defaults(func=run_config)

    p = sp.add_parser("column", help="production Hanna step, synthetic uniform boundary layer")
    p.add_argument("--mode", choices=["plume", "column"], default="plume")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--blh", type=float, default=1000.0)
    p.add_argument("--ustar", type=float, default=0.4)
    p.add_argument("--shf", type=float, default=0.0)
    p.add_argument("--wind", type=float, default=8.0)
    p.add_argument("--release-z", type=float, default=10.0)
    p.add_argument("--cell-m", type=float, default=25000.0)
    p.add_argument("--n-cells", type=int, default=61)
    p.add_argument("--dt", type=float, default=60.0)
    p.add_argument("--n-sub", type=int, default=16)
    p.add_argument("--length-s", type=float, default=6 * 3600.0)
    p.add_argument("--spin-s", type=float, default=1800.0)
    p.add_argument("--n-particles", type=int, default=20000)
    common(p)
    p.set_defaults(func=run_column)

    p = sp.add_parser("plume", help="analytic Gaussian plume, homogeneous OU")
    p.add_argument("--seed", type=int, default=9021)
    p.add_argument("--dt", type=float, default=20.0)
    p.add_argument("--n-sub", type=int, default=16)
    p.add_argument("--length-s", type=float, default=4000.0)
    p.add_argument("--n-particles", type=int, default=50000)
    p.add_argument("--receptor-frac", type=float, default=0.63)
    common(p)
    p.set_defaults(func=run_plume)

    args = ap.parse_args(argv)
    torch.set_num_threads(int(__import__("os").environ.get("OMP_NUM_THREADS", "4")))
    args.func(args)


if __name__ == "__main__":
    sys.exit(main())
