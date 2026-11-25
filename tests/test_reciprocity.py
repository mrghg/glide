"""Forward–backward reciprocity (Tier 0; docs/validation-plan.md §3, task A6).

The backward footprint of a receptor must equal the concentration a forward run
puts at that receptor from a unit source in each cell (Flesch, Wilson & Yee
1995; docs/physics.md §4). GLIDE runs backward; this file adds the forward run
through the production scheme with a private flag (``HannaScheme._forward_in_time``:
the drift unnegated, displacements and advection +dt; reflection, the
boundary-layer-top interface of decision 0028 and the velocity carry
unchanged), which no configuration can set.

**Set-up.** A steady, horizontally uniform column (``reciprocity_met``): the
Hanna profile below h (σ_w strongly height-dependent, so the drift terms
matter), the free-troposphere closure above it (N = 0.015 s⁻¹), a hydrostatic
air density (about 10% per km, so the density drift matters too) and a
westerly wind sheared from 3 m s⁻¹ at the ground to 6 m s⁻¹ at h. Cells are
rows h/10 deep up to 1.3 h and columns 3 km long along the wind; a receptor
is one cell of the section (integrated across the wind). Particles start
uniformly in their cell with velocities drawn from the Eulerian Gaussian
(the theorem's premise; production starts at w' = 0), and both directions
count ``dt / ρ_d`` (``ParticleDryAirDensity``, as the gridder) for 2 h at the
shipped 60 s step and 16 sub-steps.

**Procedure** (validation plan §3, Tier 0): a separate backward run chooses
the cells at least 50 different particles pass through; each map is
bootstrapped over its own particles; the test passes if at least 90% of the
tested cells agree within their 95% noise range and the sum over the tested
cells does too. Counts are per case in ``CASES``.

**Results at the defaults** (seed 0, ``scripts/validation/reciprocity/run_cases.py``;
outputs/agents/A6/cases/; docs/VALIDATION.md §3): convective surface receptor
0.876 of the tested cells agree, sum 1.016 (fails); convective, just above h,
0.927 and 0.998 (passes); stable surface 0.776 and 0.978, just below h 0.861
and 1.011, just above h 0.673 and 1.007 (all fail). Every difference shrinks
from 16 to 64 sub-steps; near the ground it matches the forward model's own
well-mixed error. The companions fail as they must: without the backward sign
flip 0.077 agree, without the interface rule 0.175 (sum 0.24), with the drift
stencil straddling h 0.148 (sum 1.13). The slow tests therefore fail at the
defaults for four of the five cases; what to accept is an open question for
the validation plan (dev/agent/progress/A6.md), not something this file tunes.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest
import torch

from lpdm.air_density import ParticleDryAirDensity
from lpdm.gpu_engine import GPUEngine
from lpdm.main import _advect_active_particles
from lpdm.met_reader import HourlyMetTensors, MetFieldMetadata
from lpdm.turbulence import HannaScheme
from lpdm.turbulence.hanna import (
    N_SUBSTEPS_DEFAULT,
    USTAR_MIN_M_S,
    convective_velocity,
    obukhov_length,
)

GRAVITY = 9.80665
R_D = 287.05
CP = 1004.6
KAPPA = R_D / CP
P0_PA = 100_000.0
THETA0_K = 285.0
FT_N = 0.015  # free-troposphere buoyancy frequency, s-1 (as tests/test_bl_top_well_mixed.py)
LAT0 = 45.0
LON0 = 0.0
M_PER_DEG_LON = 111_320.0 * math.cos(math.radians(LAT0))
T0 = datetime(2024, 1, 1, 12, tzinfo=UTC)

REGIMES = {
    # name: (h m, u* m/s, sensible heat flux W m-2), as the well-mixed gates
    "stable": (400.0, 0.25, -20.0),
    "convective": (1500.0, 0.3, 200.0),
}


def reciprocity_met(
    blh_m: float, ustar: float, shf: float, *, u_surface: float = 3.0, u_top: float = 6.0
) -> HourlyMetTensors:
    """Steady, horizontally uniform window.

    θ is constant (THETA0_K) in the boundary layer and rises with N = FT_N above
    h; pressure and temperature are hydrostatic (Exner function), so the air
    density falls by about 10% per km and its drift term is exercised. The wind
    is westerly, rising linearly from ``u_surface`` at the ground to ``u_top``
    at h and constant above."""

    top = max(4000.0, 3.0 * blh_m)
    level = np.unique(
        np.concatenate([np.linspace(0.0, 1.5 * blh_m, 61), np.linspace(1.5 * blh_m, top, 21)])
    )
    n_lev, n_lat, n_lon = level.size, 4, 5
    dtheta_dz = FT_N**2 * THETA0_K / GRAVITY
    theta = THETA0_K + dtheta_dz * np.clip(level - blh_m, 0.0, None)
    # Exner: dπ/dz = -g / (cp θ), π(0) = (sp / P0)^κ; trapezoid on the fine grid.
    zf = np.linspace(0.0, top, 20001)
    th_f = THETA0_K + dtheta_dz * np.clip(zf - blh_m, 0.0, None)
    integ = np.concatenate([[0.0], np.cumsum(0.5 * (1 / th_f[1:] + 1 / th_f[:-1]) * np.diff(zf))])
    pi = 1.0 - GRAVITY / CP * np.interp(level, zf, integ)
    t_k = theta * pi
    p_hpa = P0_PA * pi ** (1.0 / KAPPA) / 100.0
    u = u_surface + (u_top - u_surface) * np.clip(level / blh_m, 0.0, 1.0)

    shape = (n_lev, n_lat, n_lon)

    def col(v):
        return torch.as_tensor(v, dtype=torch.float32).view(n_lev, 1, 1).expand(shape).contiguous()

    zeros = torch.zeros(shape, dtype=torch.float32)
    chans = {
        "u": col(u),
        "v": zeros,
        "w": zeros,
        "blh": torch.full(shape, blh_m),
        "sp": torch.full(shape, P0_PA),
        "t": col(t_k),
        "q": zeros,
        "ustar": torch.full(shape, ustar),
        "shf": torch.full(shape, shf),
    }
    names = tuple(chans)
    fields = torch.stack([chans[k] for k in names], dim=0)
    md = MetFieldMetadata(
        lon=np.linspace(-3.0, 3.0, n_lon),
        lat=np.linspace(LAT0 - 2.0, LAT0 + 2.0, n_lat),
        level=level,
        pressure_level_hpa=p_hpa,
        time_start=T0,
        time_end=T0 + timedelta(hours=1),
        variable_units=dict.fromkeys(names, "1"),
    )
    return HourlyMetTensors(
        hour_start=fields,
        hour_end=fields,
        metadata=md,
        channel_names=names,
        height_agl_m=col(level),
    )


def _negated_wind(met: HourlyMetTensors) -> HourlyMetTensors:
    """The same window with (u, v, w) negated: the backward RK2 step of
    ``main._advect_active_particles`` on it is the forward RK2 step on ``met``."""

    idx = [met.channel_names.index(c) for c in ("u", "v", "w")]
    start = met.hour_start.clone()
    end = met.hour_end.clone()
    start[idx] = -start[idx]
    end[idx] = -end[idx]
    return replace(met, hour_start=start, hour_end=end)


def eulerian_sigmas(
    scheme: HannaScheme, met: HourlyMetTensors, xyz: torch.Tensor, engine: GPUEngine
) -> torch.Tensor:
    """(σ_u, σ_v, σ_w) of the scheme's profile at each particle, [N, 3], with the
    column inputs gathered as ``HannaScheme.step`` gathers them (dynamic path)."""

    device, dtype = engine.device, engine.dtype
    lon, lat = xyz[:, 0], xyz[:, 1]

    def surf(name):
        return scheme._interp_surface_field(met, name, 0.5, lon, lat, device, dtype)

    blh = scheme._limit_blh(surf("blh"))
    sp = surf("sp")
    ustar = surf("ustar").clamp(min=USTAR_MIN_M_S)
    shf = surf("shf")
    t_surface = scheme._interp_t_at_lowest_level(met, 0.5, lon, lat, device, dtype)
    L = obukhov_length(ustar, shf, t_surface, sp)
    w_star = convective_velocity(blh, shf, t_surface, sp, blh_min_m=scheme.blh_min_m)
    ft_fields, grid_bounds = scheme._free_trop_fields(met, device, dtype)
    return scheme._sigmas_at(
        xyz[:, 2],
        blh=blh,
        ustar=ustar,
        w_star=w_star,
        h_over_L=blh / L,
        L=L,
        lat=lat,
        lon=lon,
        ft_fields=ft_fields,
        grid_bounds=grid_bounds,
        engine=engine,
    )


@dataclass(frozen=True)
class Grid:
    """Cells of the (x, z) section: rows ``z_edges`` (m) and columns of width
    ``dx_m`` along the wind, indexed by the separation k (receptor cell minus
    source cell, in columns): k > 0 is a source upwind of the receptor."""

    z_edges: tuple[float, ...]
    dx_m: float
    k_min: int
    k_max: int

    @property
    def n_rows(self) -> int:
        return len(self.z_edges) - 1

    @property
    def n_k(self) -> int:
        return self.k_max - self.k_min + 1

    @property
    def dz(self) -> np.ndarray:
        return np.diff(np.asarray(self.z_edges))


def run_particles(
    scheme: HannaScheme,
    met: HourlyMetTensors,
    z0: torch.Tensor,
    *,
    forward: bool,
    grid: Grid,
    n_steps: int,
    dt: float,
    seed: int,
    eulerian_start: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Integrate particles released uniformly over the column x in [0, dx] at
    heights ``z0``, forward or backward in time, through the production step
    (``main._advect_active_particles`` then ``HannaScheme.step``, the dynamic
    path's order), and count each one's residence ``dt / ρ_d`` (ρ_d from
    ``ParticleDryAirDensity``, as the gridder) in every cell after every step.

    Returns ``(contrib [N, n_rows, n_k] float64, visited [N, n_rows, n_k] bool)``.
    A cell's column index is the separation k: ``floor(x / dx)`` forward (the
    particle is a source's tracer, the cell a receptor) and ``-floor(x / dx)``
    backward (the cell is a source of the receptor's particle)."""

    torch.manual_seed(seed)
    engine = GPUEngine(device="cpu", dtype=torch.float32)
    scheme._forward_in_time = forward
    n = z0.numel()
    dlon = grid.dx_m / M_PER_DEG_LON
    particles = torch.zeros(n, 4, dtype=torch.float32)
    particles[:, 0] = LON0 + torch.rand(n) * dlon
    particles[:, 1] = LAT0
    particles[:, 2] = z0
    particles[:, 3] = 1.0 / n
    state = scheme.initialize_state(n, device=torch.device("cpu"), dtype=torch.float32)
    if eulerian_start:
        sig = eulerian_sigmas(scheme, met, particles[:, :3], engine)
        state["u_prime"] = sig[:, 0] * torch.randn(n)
        state["v_prime"] = sig[:, 1] * torch.randn(n)
        state["w_prime"] = sig[:, 2] * torch.randn(n)
        state["sigma_last"] = sig.clone()
    active = torch.ones(n, dtype=torch.bool)
    advect_met = _negated_wind(met) if forward else met
    density = ParticleDryAirDensity("cpu")
    z_edges = torch.as_tensor(grid.z_edges, dtype=torch.float32)
    contrib = torch.zeros(n * grid.n_rows * grid.n_k, dtype=torch.float64)
    visited = torch.zeros(n * grid.n_rows * grid.n_k, dtype=torch.bool)
    pid = torch.arange(n)
    t_cursor = T0 + timedelta(minutes=30)
    for _ in range(n_steps):
        assert not scheme.step_includes_advection(engine)
        adv, t_alpha, _ = _advect_active_particles(
            engine, engine.device, particles, advect_met, t_cursor, dt, torch.float32
        )
        particles = adv
        particles, state = scheme.step(
            particles=particles,
            state=state,
            met_window=met,
            t_alpha=t_alpha,
            dt_seconds=dt,
            active_mask=active,
            engine=engine,
        )
        inv_rho = density.inverse_density(particles[:, :3], met, t_alpha, engine)
        z = particles[:, 2].contiguous()
        row = torch.bucketize(z, z_edges, right=True) - 1
        col = torch.floor((particles[:, 0] - LON0) / dlon).to(torch.int64)
        k = col if forward else -col
        ok = (row >= 0) & (row < grid.n_rows) & (k >= grid.k_min) & (k <= grid.k_max)
        flat = (pid[ok] * grid.n_rows + row[ok]) * grid.n_k + (k[ok] - grid.k_min)
        contrib.index_add_(0, flat, (dt * inv_rho[ok]).to(torch.float64))
        visited[flat] = True
    scheme._forward_in_time = False
    shape = (n, grid.n_rows, grid.n_k)
    return contrib.view(shape), visited.view(shape)


# ---------------------------------------------------------------------------
# mutations for the teeth
# ---------------------------------------------------------------------------


class NoBackwardSignFlip(HannaScheme):
    """Backward drift with the forward sign: the sign flip of docs/physics.md §4
    removed (forward runs are unaffected)."""

    def _vertical_coefficients(self, z_agl, density_fields, **ck):
        out = list(super()._vertical_coefficients(z_agl, density_fields, **ck))
        if not self._forward_in_time:
            out[7], out[8] = -out[7], -out[8]
        return tuple(out)


class NoInterface(HannaScheme):
    """The boundary-layer-top crossing rule disabled: a particle crossing h keeps
    its velocity and position (decision 0028 reverted in part)."""

    def _cross_interface(self, z_from, xyz, w_p, u_p, v_p, blh, interface_sigmas):
        return xyz, w_p, u_p, v_p


class StraddlingStencil(HannaScheme):
    """The drift's ±1 m stencil allowed to straddle h again (the pre-0028
    stencil; ``NoStencil`` of scripts/validation/wm_near_surface/interface_study.py),
    with the sign of the current time direction."""

    def _vertical_coefficients(self, z_agl, density_fields, **ck):
        out = list(super()._vertical_coefficients(z_agl, density_fields, **ck))
        z_eval = z_agl.clamp(min=self.z_ubl_m)
        z_hi = z_eval + self.DRIFT_FD_DELTA_M
        z_lo = (z_eval - self.DRIFT_FD_DELTA_M).clamp(min=self.z_ubl_m)
        sw_hi = self._column_sigma_w(z_hi, **ck)
        sw_lo = self._column_sigma_w(z_lo, **ck)
        g = (sw_hi.pow(2) - sw_lo.pow(2)) / (z_hi - z_lo).clamp(min=self.DRIFT_FD_DELTA_M)
        out[7] = (0.5 if self._forward_in_time else -0.5) * g
        return tuple(out)


VARIANTS: dict[str, type[HannaScheme]] = {
    "production": HannaScheme,
    "no_backward_sign_flip": NoBackwardSignFlip,
    "no_interface": NoInterface,
    "straddling_stencil": StraddlingStencil,
}


# ---------------------------------------------------------------------------
# the comparison (docs/validation-plan.md §3, Tier 0, reciprocity)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Case:
    """One receptor in one regime. Rows are ``h / rows_per_h`` deep up to 1.3 h;
    the receptor is row ``receptor_row``; columns ``dx_m`` wide."""

    regime: str
    receptor_row: int
    n_back: int
    n_fwd: int  # per source row
    n_steps: int = 120
    dt: float = 60.0
    n_substeps: int = N_SUBSTEPS_DEFAULT
    rows_per_h: int = 10
    dx_m: float = 3000.0
    k_min: int = -2
    k_max: int = 20
    seed: int = 0

    @property
    def grid(self) -> Grid:
        h = REGIMES[self.regime][0]
        dz = h / self.rows_per_h
        edges = tuple(float(v) for v in np.arange(0.0, 1.3 * h + 1e-6, dz))
        return Grid(edges, self.dx_m, self.k_min, self.k_max)


@dataclass
class Result:
    case: Case
    backward: np.ndarray  # [rows, k]: footprint of source cell (row, k) at the receptor
    forward: np.ndarray  # [rows, k]: forward concentration, scaled by dz_S / dz_R
    tested: np.ndarray  # [rows, k] bool, from the selection run
    lo: np.ndarray  # 95% range of the difference under particle noise alone
    hi: np.ndarray
    sum_lo: float
    sum_hi: float
    n_boot: int

    @property
    def diff(self) -> np.ndarray:
        return self.backward - self.forward

    @property
    def agree(self) -> np.ndarray:
        return (self.diff >= self.lo) & (self.diff <= self.hi)

    @property
    def frac_agree(self) -> float:
        return float(self.agree[self.tested].mean())

    @property
    def sum_diff(self) -> float:
        return float(self.diff[self.tested].sum())

    @property
    def sum_ok(self) -> bool:
        return self.sum_lo <= self.sum_diff <= self.sum_hi

    @property
    def sum_ratio(self) -> float:
        t = self.tested
        return float(self.backward[t].sum() / self.forward[t].sum())

    def summary(self) -> dict:
        t = self.tested
        f = self.forward[t].sum()
        return dict(
            regime=self.case.regime,
            receptor_z_m=self.case.grid.z_edges[
                self.case.receptor_row : self.case.receptor_row + 2
            ],
            n_tested=int(t.sum()),
            frac_agree=round(self.frac_agree, 4),
            sum_ratio=round(self.sum_ratio, 5),
            sum_range_rel=(round(self.sum_lo / f, 5), round(self.sum_hi / f, 5)),
            sum_ok=self.sum_ok,
        )


def _release_heights(edges: tuple[float, ...], rows: range, n: int, gen: torch.Generator):
    return torch.cat(
        [edges[r] + (edges[r + 1] - edges[r]) * torch.rand(n, generator=gen) for r in rows]
    )


_FORWARD_CACHE: dict[tuple, torch.Tensor] = {}


def forward_contributions(case: Case, variant: str) -> torch.Tensor:
    """Forward runs from a unit source in every row (``n_fwd`` particles each,
    uniform over column 0 of the row), as ``[source row, particle, receptor row,
    k]`` float32. One forward run per row serves every column of that row: the
    meteorology is horizontally uniform, so the concentration in receptor column
    k from a source in column 0 is that in column 0 from a source k columns
    upwind. Cached per regime, grid, counts and variant (every receptor of a
    regime shares it)."""

    grid = case.grid
    key = (case.regime, grid, case.n_fwd, case.n_steps, case.dt, case.n_substeps, case.seed,
           variant)  # fmt: skip
    if key not in _FORWARD_CACHE:
        gen = torch.Generator().manual_seed(1000 + case.seed)
        z0 = _release_heights(grid.z_edges, range(grid.n_rows), case.n_fwd, gen)
        contrib, _ = run_particles(
            VARIANTS[variant](n_substeps=case.n_substeps, static_substeps=False),
            reciprocity_met(*REGIMES[case.regime]),
            z0,
            forward=True,
            grid=grid,
            n_steps=case.n_steps,
            dt=case.dt,
            seed=11 + case.seed,
        )
        _FORWARD_CACHE[key] = contrib.view(grid.n_rows, case.n_fwd, grid.n_rows, grid.n_k).float()
    return _FORWARD_CACHE[key]


def _bootstrap_deviation(contrib: torch.Tensor, n_boot: int, rng: np.random.Generator):
    """Bootstrap resamples of a particle mean minus the mean itself, ``[B, ...]``:
    ``contrib`` is ``[N, ...]``, one row per particle."""

    n = contrib.shape[0]
    flat = contrib.reshape(n, -1).float()
    w = torch.as_tensor(rng.multinomial(n, np.full(n, 1.0 / n), size=n_boot), dtype=torch.float32)
    boot = (w @ flat) / n - flat.mean(0)
    return boot.double().numpy().reshape(n_boot, *contrib.shape[1:])


def reciprocity(case: Case, *, backward_variant: str = "production",
                forward_variant: str | None = None, n_boot: int = 400) -> Result:  # fmt: skip
    """The four steps of docs/validation-plan.md §3 (Tier 0, reciprocity) for one
    receptor: (1) a selection run, backward with its own seed, picks the cells at
    least 50 different particles pass through; (2) each map is bootstrapped over
    its own particles, giving each tested cell's 95% range of the difference
    under particle noise alone; (3) and (4) are read off the returned ``Result``
    (``frac_agree``, ``sum_ok``).

    The backward map is the footprint of each source cell at the receptor row,
    released uniformly over column 0; the forward map the receptor's
    concentration from a unit source spread uniformly over each source cell.
    Reciprocity (Flesch, Wilson & Yee 1995, with the velocities started from the
    Eulerian distribution at both ends) makes ``backward · dz_R`` equal
    ``forward · dz_S`` when both count ``dt / ρ``."""

    forward_variant = forward_variant or (
        "production" if backward_variant == "no_backward_sign_flip" else backward_variant
    )
    key = (case, backward_variant, forward_variant, n_boot)
    if key not in _RESULT_CACHE:
        _RESULT_CACHE[key] = _reciprocity(case, backward_variant, forward_variant, n_boot)
    return _RESULT_CACHE[key]


_RESULT_CACHE: dict[tuple, Result] = {}


def _reciprocity(case: Case, backward_variant: str, forward_variant: str, n_boot: int) -> Result:
    grid = case.grid
    met = reciprocity_met(*REGIMES[case.regime])
    rr = case.receptor_row
    cf = forward_contributions(case, forward_variant)[:, :, rr, :]  # [S row, particle, k]

    def backward_run(seed: int):
        gen = torch.Generator().manual_seed(2000 + seed)
        z0 = _release_heights(grid.z_edges, range(rr, rr + 1), case.n_back, gen)
        return run_particles(
            VARIANTS[backward_variant](n_substeps=case.n_substeps, static_substeps=False),
            met, z0, forward=False, grid=grid, n_steps=case.n_steps, dt=case.dt, seed=seed,
        )  # fmt: skip

    cb, _ = backward_run(21 + case.seed)
    _, visited = backward_run(31 + case.seed)  # the selection run
    tested = (visited.sum(0) >= 50).numpy()
    del visited

    scale = (grid.dz / grid.dz[rr])[:, None]  # dz_S / dz_R
    backward = cb.mean(0).numpy()
    forward = cf.double().mean(1).numpy() * scale

    rng = np.random.default_rng(7 + case.seed)
    dev = _bootstrap_deviation(cb, n_boot, rng)
    for r in range(grid.n_rows):
        dev[:, r] -= _bootstrap_deviation(cf[r], n_boot, rng) * scale[r]
    lo, hi = np.percentile(dev, [2.5, 97.5], axis=0)
    sum_lo, sum_hi = np.percentile(dev[:, tested].sum(axis=1), [2.5, 97.5])
    return Result(case, backward, forward, tested, lo, hi, float(sum_lo), float(sum_hi), n_boot)


# ---------------------------------------------------------------------------
# the forward mode itself
# ---------------------------------------------------------------------------


def test_forward_mode_is_private_and_off_by_default() -> None:
    """No configuration can reach the forward mode: it is not a constructor
    argument (``main._scheme_kwargs`` builds the scheme from the constructor's
    keywords), and a new scheme runs backward."""

    import inspect

    assert "forward" not in " ".join(inspect.signature(HannaScheme).parameters)
    assert HannaScheme()._forward_in_time is False


def test_forward_step_static_and_dynamic_paths_agree() -> None:
    """Forward in time, the static path (advection folded into the step, the
    CUDA-graph code) and the dynamic path (``main._advect_active_particles`` on
    the wind-negated window, then the step) give bit-identical heights,
    latitudes and velocities, with a sheared wind and particles on both sides of
    h. Longitudes differ in the last bits (at most 9e-10 degrees here): the two
    paths' RK2 round differently, backward as much (1.9e-9 degrees on the same
    case; the backward identity test in test_hanna.py has no wind)."""

    met = reciprocity_met(*REGIMES["convective"])
    engine = GPUEngine(device="cpu", dtype=torch.float32)
    results = []
    for static in (False, True):
        torch.manual_seed(3)
        scheme = HannaScheme(n_substeps=4, static_substeps=static)
        scheme._forward_in_time = True
        n = 2000
        particles = torch.zeros(n, 4)
        particles[:, 1] = LAT0
        particles[:, 2] = torch.rand(n) * 2500.0
        particles[:, 3] = 1.0 / n
        state = scheme.initialize_state(n, device=torch.device("cpu"), dtype=torch.float32)
        active = torch.ones(n, dtype=torch.bool)
        for _ in range(4):
            if not scheme.step_includes_advection(engine):
                particles = _advect_active_particles(
                    engine, engine.device, particles, _negated_wind(met),
                    T0 + timedelta(minutes=30), 60.0, torch.float32,
                )[0]  # fmt: skip
            particles, state = scheme.step(
                particles, state, met, t_alpha=0.5, dt_seconds=60.0, active_mask=active,
                engine=engine,
            )  # fmt: skip
        results.append((particles, state))
    (p_dyn, s_dyn), (p_stat, s_stat) = results
    assert torch.equal(p_dyn[:, 1:], p_stat[:, 1:])
    torch.testing.assert_close(p_dyn[:, 0], p_stat[:, 0], rtol=0.0, atol=1e-8)
    for key in ("u_prime", "v_prime", "w_prime", "sigma_last"):
        assert torch.equal(s_dyn[key], s_stat[key])
    # The wind carried them east (forward), by 4 min at 3-6 m/s.
    assert float(p_dyn[:, 0].mean()) * M_PER_DEG_LON > 500.0


def test_forward_mode_flips_the_drift_and_the_displacement_only() -> None:
    """The coefficients forward are the backward ones with the two drift pieces
    negated and nothing else changed (docs/physics.md §4)."""

    met = reciprocity_met(*REGIMES["convective"])
    engine = GPUEngine(device="cpu", dtype=torch.float32)
    scheme = HannaScheme()
    n = 500
    xyz = torch.zeros(n, 3)
    xyz[:, 1] = LAT0
    xyz[:, 2] = torch.linspace(0.0, 2500.0, n)
    captured = {}
    orig = scheme._vertical_coefficients

    def spy(z, d, **ck):
        out = orig(z, d, **ck)
        captured.setdefault(scheme._forward_in_time, out)
        return out

    scheme._vertical_coefficients = spy  # type: ignore[method-assign]
    for fwd in (False, True):
        scheme._forward_in_time = fwd
        p = torch.cat([xyz, torch.full((n, 1), 1.0 / n)], dim=1)
        st = scheme.initialize_state(n, device=torch.device("cpu"), dtype=torch.float32)
        scheme.step(p, st, met, 0.5, 60.0, torch.ones(n, dtype=torch.bool), engine)
    back, fwd = captured[False], captured[True]
    for i in range(7):
        assert torch.equal(back[i], fwd[i])
    assert torch.equal(back[7], -fwd[7]) and torch.equal(back[8], -fwd[8])
    assert float(back[7].abs().max()) > 0 and float(back[8].abs().max()) > 0


# ---------------------------------------------------------------------------
# reciprocity: the plan's criteria (slow; GLIDE_SLOW_TESTS=1)
# ---------------------------------------------------------------------------

SLOW = os.environ.get("GLIDE_SLOW_TESTS", "") not in ("", "0", "false", "False")
slow = pytest.mark.skipif(
    not SLOW, reason="minutes on CPU: set GLIDE_SLOW_TESTS=1 (docs/VALIDATION.md §3)"
)

N_BACK = 40_000
N_FWD = 10_000  # per source row; 13 rows
CASES: dict[str, Case] = {
    "convective_surface": Case("convective", 0, n_back=N_BACK, n_fwd=N_FWD),
    "convective_above_h": Case("convective", 10, n_back=N_BACK, n_fwd=N_FWD),
    "stable_surface": Case("stable", 0, n_back=N_BACK, n_fwd=N_FWD),
    "stable_below_h": Case("stable", 9, n_back=N_BACK, n_fwd=N_FWD),
    "stable_above_h": Case("stable", 10, n_back=N_BACK, n_fwd=N_FWD),
}
SIGN_TEETH_CASE = "convective_surface"
TEETH_CASES = {
    "no_backward_sign_flip": "convective_surface",
    "no_interface": "convective_above_h",
    "straddling_stencil": "convective_above_h",
}


def _passes(res: Result) -> bool:
    return res.frac_agree >= 0.90 and res.sum_ok


QUICK_CASE = Case("convective", 0, n_back=10_000, n_fwd=2_500, n_steps=60)


def test_removing_the_backward_sign_flip_breaks_reciprocity_quick() -> None:
    """A quick companion for the default suite (about 40 s on 16 CPU threads):
    a surface receptor in a convective boundary layer, 1 h, 10,000 backward and
    2,500 forward particles per source row. With the backward drift's sign flip
    removed, the per-cell agreement collapses; the production construction keeps
    most cells. Measured at seeds 0, 1, 2: production 0.89, 0.93, 0.90 of the
    tested cells agree, the mutant 0.09, 0.10, 0.10
    (outputs/agents/A6/explore/quick_config.log). The bounds below (0.8 and 0.3)
    are this smoke check's own, not the plan's acceptance criterion, which the
    slow tests apply; the production sum is not asserted here because at the
    default 16 sub-steps it carries the forward model's near-surface well-mixed
    error (docs/VALIDATION.md §3)."""

    prod = reciprocity(QUICK_CASE)
    mutant = reciprocity(QUICK_CASE, backward_variant="no_backward_sign_flip")
    assert prod.tested.sum() >= 50 and mutant.tested.sum() >= 50
    assert prod.frac_agree >= 0.8, prod.summary()
    assert mutant.frac_agree <= 0.3, mutant.summary()


@slow
@pytest.mark.parametrize("name", list(CASES))
def test_backward_footprint_equals_forward_concentration(name: str) -> None:
    """Validation plan §3, Tier 0: at least 90% of the tested cells agree
    within their 95% noise range, and so does the sum over them."""

    res = reciprocity(CASES[name])
    print(name, res.summary())
    assert res.frac_agree >= 0.90, res.summary()
    assert res.sum_ok, res.summary()


@slow
@pytest.mark.parametrize("variant", list(TEETH_CASES))
def test_reciprocity_fails_when_the_physics_is_broken(variant: str) -> None:
    """The companions: the same procedure fails with the backward drift's sign
    flip removed (validation plan §3), with the boundary-layer-top crossing
    rule disabled, and with the drift stencil straddling h (decision 0028)."""

    res = reciprocity(CASES[TEETH_CASES[variant]], backward_variant=variant)
    print(variant, res.summary())
    assert not _passes(res), res.summary()
