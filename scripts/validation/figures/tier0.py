"""Tier 0 figures: the mathematical checks (docs/validation-plan.md §2 and §5).

Each figure recomputes the quantity an existing analytic test asserts, with the
same parameters and seed, through the same GLIDE primitives, and draws it
against its exact or numerical reference. Parameters, reference solutions and
helper builders are imported from the tests themselves wherever the tests
expose them, so a figure cannot silently drift from what CI checks.

| figure | test it draws |
| --- | --- |
| ``taylor_dispersion`` | ``tests/test_dispersion_analytic.py::test_taylor_dispersion_curve_ballistic_to_diffusive`` |
| ``well_mixed`` | ``tests/test_main_runtime.py::test_v1_well_mixed_hanna_backward_path`` (extended to three stability regimes) and ``tests/test_physics.py::test_well_mixed_condition_drift_keeps_uniform_distribution`` |
| ``analytic_plume`` | ``tests/test_plume_footprint.py`` |
| ``diffusion_limit`` | ``tests/test_diffusion_pde.py`` |
| ``convergence`` | ``tests/test_dispersion_analytic.py`` (rotation, Euler bias with dt), particle count, Hanna ``substep_c`` |
| ``terrain`` | ``tests/test_terrain_transport.py`` |
| ``reciprocity`` | pending: needs the forward-in-time mode (validation plan §2, Tier 0) |

All cases run on CPU in float64 (float32 for the production Hanna path, as in
its test); the whole set takes about eight minutes on 8 threads and under 0.5 GB.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import torch
import xarray as xr

from . import core, style
from .style import plt

TESTS = core.REPO_ROOT / "tests"
if str(TESTS) not in sys.path:
    sys.path.insert(0, str(TESTS))

TIER = 0


@dataclass(frozen=True)
class FigureSpec:
    figure_id: str
    title: str
    plan_item: str  # the bullet in docs/validation-plan.md §5
    status: str = "ready"  # or "pending: <reason>"


FIGURES: list[FigureSpec] = [
    FigureSpec("taylor_dispersion", "Taylor dispersion", "Taylor dispersion"),
    FigureSpec("well_mixed", "Well-mixed condition", "Well-mixed test"),
    FigureSpec("analytic_plume", "Analytic plume footprint", "Analytic plume"),
    FigureSpec("diffusion_limit", "Diffusion limit", "Diffusion limit"),
    FigureSpec(
        "reciprocity",
        "Forward–backward reciprocity",
        "Reciprocity",
        status="pending: needs the forward-in-time mode (validation plan §2, Tier 0)",
    ),
    FigureSpec("convergence", "Convergence", "Convergence"),
    FigureSpec("terrain", "Terrain-following transport", "Terrain"),
]


def _engine():
    from lpdm.gpu_engine import GPUEngine

    return GPUEngine(device="cpu", dtype=torch.float64)


def _taylor_var(sigma2: float, t_l: float, t):
    t = np.asarray(t, dtype=float)
    return 2.0 * sigma2 * t_l * (t - t_l * (1.0 - np.exp(-t / t_l)))


def _finish(fig, out_dir, spec: FigureSpec, config, caption, metrics, source):
    prov = core.provenance(config)
    core.stamp(fig, spec.title, prov, f"tier0/{spec.figure_id}")
    return core.save(
        fig,
        out_dir,
        figure_id=spec.figure_id,
        tier=TIER,
        title=spec.title,
        caption=caption,
        config=config,
        prov=prov,
        metrics=metrics,
        source=source,
    )


def _spec(fid: str) -> FigureSpec:
    return next(s for s in FIGURES if s.figure_id == fid)


# --------------------------------------------------------------------------- Taylor


def taylor_dispersion(out_dir: Path) -> Path:
    cfg = dict(n=100_000, sigma_w2=1.0, t_l=100.0, dt=1.0, t_end_tl=30.0, seed=2201)
    torch.manual_seed(cfg["seed"])
    engine = _engine()
    n, s2, t_l, dt = cfg["n"], cfg["sigma_w2"], cfg["t_l"], cfg["dt"]
    p = torch.zeros((n, 4), dtype=torch.float64)
    p[:, 3] = 1.0 / n
    w = torch.randn(n, dtype=torch.float64) * math.sqrt(s2)
    n_steps = int(cfg["t_end_tl"] * t_l / dt)
    rec_steps = np.unique(np.round(np.logspace(0, np.log10(n_steps), 40)).astype(int))
    z_rec = np.empty((rec_steps.size, n))
    k = 0
    for step in range(1, n_steps + 1):
        w = engine.update_langevin_velocity(w, t_lagrangian=t_l, sigma_w2=s2, dt_seconds=dt)
        p = engine.apply_vertical_turbulence(p, w, dt_seconds=dt, backward=False)
        if k < rec_steps.size and step == rec_steps[k]:
            z_rec[k] = p[:, 2].numpy()
            k += 1
    t = rec_steps * dt
    var = z_rec.var(axis=1, ddof=1)
    se = core.subset_spread(lambda idx: z_rec[:, idx].var(axis=1, ddof=1), n)
    ref = _taylor_var(s2, t_l, t)

    spec = _spec("taylor_dispersion")
    fig, axes = core.new_figure(2, width=4.6, height=3.6)
    ax, axr = axes[0]
    x = t / t_l
    xb, xd = x[x <= 3.0], x[x >= 0.3]
    ax.loglog(xb, s2 * (xb * t_l) ** 2, ls=":", color=style.MUTED, lw=1.4)
    ax.loglog(xd, 2 * s2 * t_l * xd * t_l, ls=":", color=style.MUTED, lw=1.4)
    ax.annotate("ballistic  σ_w² t²", (3.0, s2 * (3.0 * t_l) ** 2), xytext=(-4, 0),
                textcoords="offset points", color=style.INK2, fontsize=8, ha="right")  # fmt: skip
    ax.annotate("diffusive  2 σ_w² T_L t", (0.3, 2 * s2 * t_l * 0.3 * t_l), xytext=(0, 8),
                textcoords="offset points", color=style.INK2, fontsize=8, ha="left")  # fmt: skip
    core.log_band(ax, x, var, se)
    ax.loglog(x, var, color=style.GLIDE, label="GLIDE (100,000 particles)")
    ax.loglog(x, ref, color=style.EXACT, ls="--", lw=1.4, label="Taylor (1921)")
    ax.set_xlabel("time / T_L")
    ax.set_ylabel("vertical spread σ_z²  (m²)")
    ax.set_title("Spread against time")
    ax.legend(loc="lower right")
    core.recessive_grid(ax, "both")

    lr = np.log10(var / ref)
    lr_se = se / var / np.log(10)
    core.band(axr, x, lr, lr_se)
    axr.semilogx(x, lr, color=style.GLIDE, label="GLIDE / Taylor")
    axr.axhline(0, color=style.EXACT, ls="--", lw=1.2)
    axr.set_ylim(-0.03, 0.03)
    axr.set_xlabel("time / T_L")
    axr.set_ylabel("log₁₀(GLIDE / Taylor)")
    axr.set_title("Log ratio")
    axr.legend(loc="lower right")
    core.recessive_grid(axr)
    fig.subplots_adjust(wspace=0.32)
    rel = np.abs(np.sqrt(var / ref) - 1)
    caption = (
        "Vertical spread of an Ornstein–Uhlenbeck particle ensemble (σ_w² = 1 m² s⁻², "
        "T_L = 100 s, Δt = 1 s) through GLIDE's engine primitives, against Taylor's (1921) "
        "curve, with the ballistic and diffusive limits dotted. Band: particle noise from "
        "10 disjoint subsets."
    )
    return _finish(
        fig, out_dir, spec, cfg, caption,
        {"max_rel_err_sigma_z": float(rel.max()), "sigma_z_rel_err_at_30TL": float(rel[-1])},
        "tests/test_dispersion_analytic.py::test_taylor_dispersion_curve_ballistic_to_diffusive",
    )  # fmt: skip


# --------------------------------------------------------------------------- well-mixed


def _hanna_met_window(*, blh_m: float, ustar: float, shf: float):
    """Horizontally uniform, motionless met window (as ``_wmc_met_window`` in
    tests/test_main_runtime.py) with a chosen surface heat flux, and an almost
    constant pressure so the density is uniform and the well-mixed state is
    flat (the density-weighted state is tested separately)."""
    from lpdm.met_reader import HourlyMetTensors, MetFieldMetadata

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
        hour_start=fields,
        hour_end=fields,
        metadata=md,
        channel_names=names,
        height_agl_m=height.contiguous(),
    )


WM_REGIMES = {
    # name: (blh m, u* m/s, sensible heat flux W m-2)
    "stable": (400.0, 0.25, -20.0),
    "neutral": (1500.0, 0.4, 0.0),
    "convective": (1500.0, 0.3, 200.0),
}


def _hanna_well_mixed(blh, ustar, shf, *, n, dt, record_s, seed, substep_c=None):
    from lpdm.gpu_engine import GPUEngine
    from lpdm.turbulence import HannaScheme

    torch.manual_seed(seed)
    engine = GPUEngine(device="cpu")
    kw = {} if substep_c is None else {"substep_c": substep_c}
    scheme = HannaScheme(meander_enabled=False, static_substeps=False, **kw)
    met = _hanna_met_window(blh_m=blh, ustar=ustar, shf=shf)
    p = torch.zeros(n, 4, dtype=torch.float32)
    p[:, 1] = 45.0
    p[:, 2] = torch.rand(n, dtype=torch.float32) * blh
    p[:, 3] = 1.0 / n
    state = scheme.initialize_state(n, device=torch.device("cpu"), dtype=torch.float32)
    active = torch.ones(n, dtype=torch.bool)
    out = {}
    rec = sorted(record_s)
    t = 0.0
    for t_rec in rec:
        while t < t_rec - 1e-9:
            p, state = scheme.step(p, state, met, t_alpha=0.5, dt_seconds=dt,
                                   active_mask=active, engine=engine)  # fmt: skip
            above = p[:, 2] > blh  # manual BL-top reflection, as in the test
            if bool(above.any()):
                p[above, 2] = 2.0 * blh - p[above, 2]
                state["w_prime"][above] = -state["w_prime"][above]
            t += dt
        out[t_rec] = p[:, 2].numpy().copy()
    return out


def _engine_linear_sigma(use_drift: bool, *, n=60_000, steps=4000, seed=7):
    """``test_well_mixed_condition_drift_keeps_uniform_distribution``: linear
    σ_w² from 0.5 to 2.0 m² s⁻² over 1 km, with and without the drift."""
    torch.manual_seed(seed)
    engine = _engine()
    H, tl, dt = 1000.0, 100.0, 1.0
    s_lo, s_hi = 0.5, 2.0
    ds2 = (s_hi - s_lo) / H
    z = torch.rand(n, dtype=torch.float64) * H
    w = torch.zeros(n, dtype=torch.float64)
    for _ in range(steps):
        sig2 = s_lo + (s_hi - s_lo) * (z / H)
        drift = 0.5 * (1.0 + w.pow(2) / sig2) * ds2 if use_drift else 0.0
        w = engine.update_langevin_velocity(w, t_lagrangian=tl, sigma_w2=sig2, dt_seconds=dt,
                                            drift=drift)  # fmt: skip
        z = z + w * dt
        below, above = z < 0.0, z > H
        z = torch.where(below, -z, z)
        z = torch.where(above, 2.0 * H - z, z)
        w = torch.where(below | above, -w, w)
    return z.numpy() / H


def _profile(zn: np.ndarray, edges: np.ndarray, idx=None) -> np.ndarray:
    """Particle density normalised by the uniform value (1 = well mixed)."""
    z = zn if idx is None else zn[idx]
    h, _ = np.histogram(z, bins=edges)
    return h / (z.size * np.diff(edges))


def well_mixed(out_dir: Path) -> Path:
    cfg = dict(
        regimes=WM_REGIMES, n=20_000, dt=15.0, record_s=[1800.0, 7200.0, 21600.0],
        seed=202605301, bins=15, hanna="HannaScheme(meander off, dynamic substeps, defaults)",
        companion=dict(test="test_well_mixed_condition_drift_keeps_uniform_distribution",
                       n=60_000, steps=4000, seed=7),
    )  # fmt: skip
    edges = np.linspace(0.0, 1.0, cfg["bins"] + 1)
    zc = 0.5 * (edges[1:] + edges[:-1])
    spec = _spec("well_mixed")
    fig, axes = core.new_figure(4, width=3.0, height=3.9, sharey=True)
    axes = axes[0]
    shades = ["#9ec5f4", "#5598e7", style.GLIDE]
    metrics = {}
    for ax, (name, (blh, ustar, shf)) in zip(axes[:3], WM_REGIMES.items()):
        res = _hanna_well_mixed(blh, ustar, shf, n=cfg["n"], dt=cfg["dt"],
                                record_s=cfg["record_s"], seed=cfg["seed"])  # fmt: skip
        for (t_rec, z), col in zip(res.items(), shades):
            zn = np.clip(z / blh, 0.0, 1.0 - 1e-12)
            prof = _profile(zn, edges)
            if col == style.GLIDE:
                se = core.subset_spread(lambda idx, zn=zn: _profile(zn, edges, idx), zn.size)
                ax.fill_betweenx(zc, prof - 2 * se, prof + 2 * se, color=col,
                                 alpha=style.BAND_ALPHA, lw=0, step="mid")  # fmt: skip
                metrics[f"{name}_max_abs_dev"] = float(np.abs(prof - 1).max())
            ax.step(prof, zc, where="mid", color=col, lw=1.8, label=f"{t_rec / 3600:g} h")
        ax.axvline(1.0, color=style.EXACT, ls="--", lw=1.3, label="well mixed")
        ax.set_title(f"{name}\nh = {blh:.0f} m, H = {shf:+.0f} W m⁻²", fontsize=9.5)
        ax.set_xlim(0.4, 1.6)
        ax.set_xlabel("density / uniform")
        core.recessive_grid(ax, "x")
    axes[0].set_ylabel("height / boundary-layer depth")
    axes[0].legend(loc="upper left", fontsize=7.5)

    ax = axes[3]
    for use_drift, col, lab in [(True, style.GLIDE, "drift on"), (False, style.ALT, "drift off")]:
        zn = np.clip(_engine_linear_sigma(use_drift), 0.0, 1.0 - 1e-12)
        prof = _profile(zn, edges)
        se = core.subset_spread(lambda idx, zn=zn: _profile(zn, edges, idx), zn.size)
        ax.fill_betweenx(zc, prof - 2 * se, prof + 2 * se, color=col, alpha=style.BAND_ALPHA,
                         lw=0, step="mid")  # fmt: skip
        ax.step(prof, zc, where="mid", color=col, lw=1.8, label=lab)
        metrics[f"companion_bottom_third_{lab.replace(' ', '_')}"] = float((zn < 1 / 3).mean())
    ax.axvline(1.0, color=style.EXACT, ls="--", lw=1.3)
    ax.set_title("drift correction switched off\nσ_w² 0.5 → 2 m² s⁻², 4000 s", fontsize=9.5)
    ax.set_xlim(0.4, 1.6)
    ax.set_xlabel("density / uniform")
    ax.legend(loc="upper right", fontsize=7.5)
    core.recessive_grid(ax, "x")
    fig.subplots_adjust(wspace=0.12, top=0.78)
    caption = (
        "Particles released uniformly through the boundary layer, stepped by the production "
        "Hanna scheme (backward, dynamic sub-steps, meander off, uniform density, reflection "
        "at the ground and the boundary-layer top, outer step 15 s as in the test), against "
        "the uniform well-mixed state at 0.5, 2 and 6 h in three stability regimes. The CI "
        "test asserts only the neutral case's interior. Right: the engine-level companion "
        "test, where switching off the well-mixed drift piles particles into the weakly "
        "turbulent lower layer. Bands: particle noise (±2 s.e., 10 subsets) at the last time."
    )
    return _finish(fig, out_dir, spec, cfg, caption, metrics,
                   "tests/test_main_runtime.py::test_v1_well_mixed_hanna_backward_path; "
                   "tests/test_physics.py::test_well_mixed_condition_drift_keeps_uniform_distribution")  # fmt: skip


# --------------------------------------------------------------------------- plume


def analytic_plume(out_dir: Path) -> Path:
    import test_plume_footprint as tp
    from scipy.special import ndtr

    from lpdm.comparison import to_stilt_surface_footprint
    from lpdm.footprint_gridder import FootprintGridder

    K = core.N_SUBSETS
    cfg = dict(
        u_ms=tp.U_MS, sigma2=tp.SIGMA2, t_l=tp.T_L, z_r=tp.Z_R, h_surf=tp.H_SURF, dt=tp.DT,
        n_steps=tp.N_STEPS, n=tp.N_PART, dx=tp.DX_M, dy=tp.DY_M, nx=tp.NX, ny=tp.NY,
        seed=9021, rho=1.2, m_air=0.02897, subsets=K,
    )  # fmt: skip
    torch.manual_seed(cfg["seed"])
    engine = _engine()
    gridder = FootprintGridder(
        lon_bounds=(tp.X0_M / tp.M_PER_DEG_LON, (tp.X0_M + tp.NX * tp.DX_M) / tp.M_PER_DEG_LON),
        lat_bounds=(tp.Y0_M / tp.M_PER_DEG_LAT, (tp.Y0_M + tp.NY * tp.DY_M) / tp.M_PER_DEG_LAT),
        z_edges_m=(0.0, tp.H_SURF, 10000.0),
        n_time_bins=1, n_y=tp.NY, n_x=tp.NX, n_releases=K, device="cpu", dtype=torch.float64,
    )  # fmt: skip
    n = tp.N_PART
    p = torch.zeros((n, 4), dtype=torch.float64)
    p[:, 2] = tp.Z_R
    p[:, 3] = 1.0 / n
    sig = math.sqrt(tp.SIGMA2)
    v = torch.randn(n, dtype=torch.float64) * sig
    w = torch.randn(n, dtype=torch.float64) * sig
    u0 = torch.zeros(n, dtype=torch.float64)
    t_idx = torch.zeros(n, dtype=torch.int64)
    # Subset label as the release index: the gridder keeps one footprint per subset.
    rel_idx = torch.as_tensor(np.random.default_rng(0).permutation(n) % K, dtype=torch.int64)
    active = torch.ones(n, dtype=torch.bool)

    def wind_fn(xyz):
        out = torch.zeros_like(xyz)
        out[:, 0] = tp.U_MS / tp.M_PER_DEG_LON
        return out

    for _ in range(tp.N_STEPS):
        p = engine.rk2_advect_backward(p, dt_seconds=tp.DT, wind_fn=wind_fn)
        v = engine.update_langevin_velocity(v, t_lagrangian=tp.T_L, sigma_w2=tp.SIGMA2,
                                            dt_seconds=tp.DT)  # fmt: skip
        w = engine.update_langevin_velocity(w, t_lagrangian=tp.T_L, sigma_w2=tp.SIGMA2,
                                            dt_seconds=tp.DT)  # fmt: skip
        p = engine.apply_horizontal_turbulence(p, u0, v, dt_seconds=tp.DT, backward=True)
        p = engine.apply_vertical_turbulence(p, w, dt_seconds=tp.DT, backward=True)
        p, w = engine.reflect_surface(p, w, z_surface=0.0)
        gridder.accumulate(particles=p[:, :3], active_mask=active, weights=p[:, 3],
                           t_idx=t_idx, release_idx=rel_idx, dt_seconds=tp.DT)  # fmt: skip
    sub = gridder.tensor[:, 0, 0].numpy()  # [K, NY, NX] raw seconds
    raw = sub.sum(axis=0)
    xc = (np.arange(tp.NX) + 0.5) * tp.DX_M + tp.X0_M
    yc = (np.arange(tp.NY) + 0.5) * tp.DY_M + tp.Y0_M
    ye = np.arange(tp.NY + 1) * tp.DY_M + tp.Y0_M
    ana = np.zeros_like(raw)
    for j in range(tp.NX):
        if xc[j] >= 0.0:
            continue
        t = -xc[j] / tp.U_MS
        s = tp._taylor_sigma(t)
        p_y = ndtr(ye[1:] / s) - ndtr(ye[:-1] / s)
        p_z = (ndtr((tp.H_SURF - tp.Z_R) / s) - ndtr(-tp.Z_R / s)) + (
            ndtr((tp.H_SURF + tp.Z_R) / s) - ndtr(tp.Z_R / s)
        )
        ana[:, j] = (tp.DX_M / tp.U_MS) * p_y * p_z

    # STILT units through GLIDE's own conversion (an exact scale factor; the test
    # checks it), applied to both fields.
    da = xr.DataArray(
        raw[None, None], dims=("time_ago", "z_bin", "latitude", "longitude"),
        coords=dict(z_bottom_m=("z_bin", [0.0]), z_top_m=("z_bin", [tp.H_SURF]),
                    latitude=yc, longitude=xc),
        attrs={"density_weighting": "none"},  # plain residence seconds
    )  # fmt: skip
    stilt = to_stilt_surface_footprint(
        da, surface_layer_depth_m=tp.H_SURF, air_density_kg_m3=cfg["rho"],
        m_air_kg_per_mol=cfg["m_air"], integrate_time=True,
    ).values  # fmt: skip
    scale = float(stilt.sum() / raw.sum())
    f_glide, f_ana, f_sub = raw * scale, ana * scale, sub * scale * K

    # Upwind columns the particles fully crossed (the last one is only partly
    # reached at the end of the run).
    up = (xc < 0) & (-xc + tp.DX_M / 2 <= tp.U_MS * tp.N_STEPS * tp.DT)
    keep = up | (xc >= 0)
    f_glide, f_ana = np.where(keep, f_glide, np.nan), np.where(keep, f_ana, np.nan)
    xk, yk = xc / 1000.0, yc / 1000.0
    spec = _spec("analytic_plume")
    fig = plt.figure(figsize=(13.2, 7.6))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1], hspace=0.42, wspace=0.28,
                          top=0.86, bottom=0.08, left=0.06, right=0.98)  # fmt: skip
    norm = core.footprint_norm(f_glide, f_ana, decades=4)
    a1, a2, a3 = (fig.add_subplot(gs[0, i]) for i in range(3))
    core.footprint_panel(a1, xk, yk, f_glide, norm, title="GLIDE")
    core.footprint_panel(a2, xk, yk, f_ana, norm, title="Exact plume solution")
    # Ratio only where the exact solution carries non-negligible signal.
    sig_mask = f_ana > norm.vmin
    core.ratio_panel(a3, xk, yk, np.where(sig_mask, core.log_ratio(f_glide, f_ana), np.nan),
                     half_range=0.3, title="log₁₀(GLIDE / exact)")  # fmt: skip
    for a in (a1, a2, a3):
        a.set_xlabel("distance downwind of the receptor (km)")
        a.set_ylim(-2.0, 2.0)
        a.plot(0, 0, "o", ms=7, mfc=style.INK, mec=style.SURFACE, mew=1.5)
    a1.set_ylabel("crosswind (km)")

    b1 = fig.add_subplot(gs[1, :2])
    b2 = fig.add_subplot(gs[1, 2])
    col_g = f_glide.sum(axis=0) * tp.DY_M
    col_a = f_ana.sum(axis=0) * tp.DY_M
    col_sub = f_sub.sum(axis=1) * tp.DY_M  # [K, NX]
    se = col_sub.std(axis=0, ddof=1) / np.sqrt(K)
    d = -xk[up]
    core.band(b1, d, col_g[up], se[up])
    b1.plot(d, col_g[up], color=style.GLIDE, label="GLIDE")
    b1.plot(d, col_a[up], color=style.EXACT, ls="--", lw=1.4, label="exact")
    b1.set_yscale("log")
    b1.set_xlabel("distance upwind of the receptor (km)")
    b1.set_ylabel(f"crosswind-integrated\nfootprint ({FU_M})")
    b1.set_title("Crosswind-integrated footprint against distance")
    b1.legend(loc="upper right")
    core.recessive_grid(b1)
    lr = core.log_ratio(col_g[up], col_a[up])
    lr_se = se[up] / col_g[up] / np.log(10)
    core.band(b2, d, lr, lr_se, label=None)
    b2.plot(d, lr, color=style.GLIDE)
    b2.axhline(0, color=style.EXACT, ls="--", lw=1.2)
    b2.axvline(2 * tp.T_L * tp.U_MS / 1000, color=style.MUTED, lw=1, ls=":")
    b2.annotate("2 T_L", (2 * tp.T_L * tp.U_MS / 1000, 0.08), xytext=(3, 0),
                textcoords="offset points", fontsize=8, color=style.INK2)  # fmt: skip
    b2.set_ylim(-0.1, 0.1)
    b2.set_xlabel("distance upwind (km)")
    b2.set_ylabel("log₁₀(GLIDE / exact)")
    b2.set_title("Log ratio")
    core.recessive_grid(b2)
    far = ((-xc / tp.U_MS) >= 2 * tp.T_L) & up
    tot_ratio = float(raw[:, far].sum() / ana[:, far].sum())
    corr = float(np.corrcoef(raw[:, far].ravel(), ana[:, far].ravel())[0, 1])
    caption = (
        "Backward release at 50 m in a uniform 5 m s⁻¹ wind with homogeneous turbulence "
        "(σ_v = σ_w = 0.5 m s⁻¹, T_L = 100 s, ground reflection, 200,000 particles): GLIDE's "
        "0–40 m footprint against the exact cell-integrated reflected-Gaussian solution, "
        "in STILT units through GLIDE's conversion (ρ = 1.2 kg m⁻³). The ratio map is "
        "masked where the exact footprint is below the colour scale. The three one-column "
        "spikes in the profile are not particle noise: along-wind motion is deterministic, "
        "and particle positions fall on cell edges, so those columns receive 9 or 11 time "
        "samples instead of 10 (±10%). Band: particle noise (±2 s.e., 10 subsets)."
    )
    return _finish(fig, out_dir, spec, cfg, caption,
                   {"total_ratio_beyond_2TL": tot_ratio, "corr_beyond_2TL": corr},
                   "tests/test_plume_footprint.py")  # fmt: skip


FU_M = "m³ s mol⁻¹"  # footprint (m² s mol⁻¹) integrated across wind (m)


# --------------------------------------------------------------------------- diffusion


def diffusion_limit(out_dir: Path) -> Path:
    import test_diffusion_pde as td

    cfg = dict(
        k0=td.K0, a_kz=td.A_KZ, z_cap=td.Z_CAP, h_top=td.H_TOP, t_l=td.T_L, dt=td.DT,
        n=td.N_PART, release=(td.Z0_LO, td.Z0_HI), checkpoints=td.CHECKPOINTS,
        bin_w=td.BIN_W, pde_dz=td.PDE_DZ, seed=3301,
    )  # fmt: skip
    torch.manual_seed(cfg["seed"])
    engine = _engine()

    def sigma2(z):
        return (td.K0 + td.A_KZ * torch.clamp(z, max=td.Z_CAP)) / td.T_L

    dsig_below = torch.tensor(td.A_KZ / td.T_L, dtype=torch.float64)
    zero = torch.tensor(0.0, dtype=torch.float64)
    n = td.N_PART
    z = torch.rand(n, dtype=torch.float64) * (td.Z0_HI - td.Z0_LO) + td.Z0_LO
    w = torch.randn(n, dtype=torch.float64) * torch.sqrt(sigma2(z))
    p = torch.zeros((n, 4), dtype=torch.float64)
    p[:, 2] = z
    p[:, 3] = 1.0 / n
    zn = np.arange(0.0, td.H_TOP + td.PDE_DZ / 2, td.PDE_DZ)
    zf = 0.5 * (zn[:-1] + zn[1:])
    steppers = {k: td._make_cn_stepper(td._k_faces(zf, f), zn.size)
                for k, f in [("true", 1.0), ("half", 0.5)]}  # fmt: skip
    c0 = np.where((zn >= td.Z0_LO) & (zn <= td.Z0_HI), 1.0, 0.0)
    c0 /= c0.sum()
    pde = {k: c0.copy() for k in steppers}
    edges = np.arange(0.0, td.H_TOP + td.BIN_W / 2, td.BIN_W)
    node_bin = np.minimum((zn // td.BIN_W).astype(int), edges.size - 2)

    def bin_pde(c):
        h = np.bincount(node_bin, weights=c, minlength=edges.size - 1)
        return h / h.sum() / td.BIN_W

    res = {}
    t = 0.0
    for tc in td.CHECKPOINTS:
        for _ in range(int(round((tc - t) / td.DT))):
            zc = p[:, 2]
            s2 = sigma2(zc)
            drift = 0.5 * (1.0 + w * w / s2) * torch.where(zc < td.Z_CAP, dsig_below, zero)
            w = engine.update_langevin_velocity(w, t_lagrangian=td.T_L, sigma_w2=s2,
                                                dt_seconds=td.DT, drift=drift)  # fmt: skip
            p = engine.apply_vertical_turbulence(p, w, dt_seconds=td.DT, backward=False)
            p, w = engine.reflect_surface(p, w, z_surface=0.0)
            above = p[:, 2] > td.H_TOP
            p[:, 2] = torch.where(above, 2.0 * td.H_TOP - p[:, 2], p[:, 2])
            w = torch.where(above, -w, w)
            for k in pde:
                pde[k] = steppers[k](pde[k])
        t = tc
        zp = p[:, 2].numpy().copy()
        res[tc] = dict(z=zp, **{k: bin_pde(c) for k, c in pde.items()})

    def dens(zp, idx=None):
        zz = zp if idx is None else zp[idx]
        h, _ = np.histogram(zz, bins=edges)
        return h / zz.size / td.BIN_W

    spec = _spec("diffusion_limit")
    fig, axes = core.new_figure(3, width=3.6, height=4.0, sharey=True)
    axes = axes[0]
    zc = 0.5 * (edges[1:] + edges[:-1])
    metrics = {}
    for ax, tc in zip(axes, td.CHECKPOINTS):
        r = res[tc]
        g = dens(r["z"])
        se = core.subset_spread(lambda idx, r=r: dens(r["z"], idx), n)
        ax.fill_betweenx(zc, (g - 2 * se) * 1e3, (g + 2 * se) * 1e3, step="mid",
                         color=style.GLIDE, alpha=style.BAND_ALPHA, lw=0)  # fmt: skip
        ax.step(g * 1e3, zc, where="mid", color=style.GLIDE, label="GLIDE particles")
        ax.step(r["true"] * 1e3, zc, where="mid", color=style.EXACT, ls="--", lw=1.4,
                label="diffusion equation, K(z)")  # fmt: skip
        ax.step(r["half"] * 1e3, zc, where="mid", color=style.MUTED, ls=":", lw=1.4,
                label="same, K halved below 50 m")  # fmt: skip
        ax.set_title(f"t = {tc / 60:g} min")
        ax.set_xlabel("density (per km)")
        core.recessive_grid(ax, "x")
        metrics[f"L1_true_t{int(tc)}"] = float(np.abs(g - r["true"]).sum() * td.BIN_W)
        metrics[f"L1_half_t{int(tc)}"] = float(np.abs(g - r["half"]).sum() * td.BIN_W)
    axes[0].set_ylabel("height (m)")
    axes[-1].legend(loc="upper right", fontsize=7.5)
    fig.subplots_adjust(wspace=0.1, top=0.82)
    caption = (
        "Particles released at 5–15 m, stepped with the production OU update, Thomson drift "
        "and reflection in K(z) = 0.02 + 0.12 min(z, 200 m) m² s⁻¹ (T_L = 10 s, Δt = 1 s), "
        "against a Crank–Nicolson solution of ∂c/∂t = ∂_z(K ∂_z c). Dotted: the PDE with "
        "K halved below 50 m, the error class this test must catch. Band: particle noise "
        "(±2 s.e., 10 subsets)."
    )
    return _finish(fig, out_dir, spec, cfg, caption, metrics, "tests/test_diffusion_pde.py")


# --------------------------------------------------------------------------- convergence


def _rotation_errors(dts):
    import test_dispersion_analytic as tda

    engine = _engine()
    omega, period = 2.0 * math.pi / 3600.0, 3600.0
    start = torch.tensor([[100_000.0, 0.0, 500.0, 0.25], [0.0, 50_000.0, 500.0, 0.25],
                          [-70_000.0, 40_000.0, 500.0, 0.25], [30_000.0, -80_000.0, 500.0, 0.25]],
                         dtype=torch.float64)  # fmt: skip
    errs = []
    for dt in dts:
        f = tda._run_rotation(engine, start, omega, dt, period)
        errs.append(float(torch.linalg.norm(f[:, :2] - start[:, :2], dim=1).max()))
    return np.array(errs) / 100_000.0


def _euler_bias(dts, *, n=400_000, seed=71):
    """Relative error of σ_z at 5 T_L against Taylor, per time step, with s.e."""
    t_l, s2 = 100.0, 1.0
    total = 5 * t_l
    exact = math.sqrt(float(_taylor_var(s2, t_l, total)))
    engine = _engine()
    errs, ses = [], []
    for dt in dts:
        torch.manual_seed(seed)
        p = torch.zeros((n, 4), dtype=torch.float64)
        w = torch.randn(n, dtype=torch.float64)
        for _ in range(int(round(total / dt))):
            w = engine.update_langevin_velocity(w, t_lagrangian=t_l, sigma_w2=s2, dt_seconds=dt)
            p = engine.apply_vertical_turbulence(p, w, dt_seconds=dt, backward=False)
        z = p[:, 2].numpy()
        errs.append(z.std(ddof=1) / exact - 1)
        ses.append(core.subset_spread(lambda idx, z=z: z[idx].std(ddof=1), n) / exact)
    return np.array(errs), np.array(ses)


def _particle_count_errors(ns, *, pool=819_200, dt=0.5, seed=5):
    """RMS relative error of σ_z² at 5 T_L over disjoint groups of N particles."""
    t_l, s2 = 100.0, 1.0
    total = 5 * t_l
    torch.manual_seed(seed)
    engine = _engine()
    p = torch.zeros((pool, 4), dtype=torch.float64)
    w = torch.randn(pool, dtype=torch.float64)
    for _ in range(int(round(total / dt))):
        w = engine.update_langevin_velocity(w, t_lagrangian=t_l, sigma_w2=s2, dt_seconds=dt)
        p = engine.apply_vertical_turbulence(p, w, dt_seconds=dt, backward=False)
    z = p[:, 2].numpy()
    exact = float(_taylor_var(s2, t_l, total))
    rms, se = [], []
    for nn in ns:
        g = z[: (pool // nn) * nn].reshape(-1, nn)
        e2 = (g.var(axis=1, ddof=1) / exact - 1) ** 2
        rms.append(math.sqrt(e2.mean()))
        se.append(e2.std(ddof=1) / math.sqrt(e2.size) / (2 * rms[-1]))
    return np.array(rms), np.array(se)


def _near_surface_error(regime: str, *, dt: float, substep_c=None, n=10_000, seed=11):
    """Near-surface well-mixed error of the production Hanna scheme.

    Wilson & Flesch (1993) §7b: the well-mixed error in inhomogeneous turbulence
    is first order in the step over T_L, and largest near the ground, where T_L
    is shortest. Measured as the occupancy of the lowest 5% of the boundary
    layer relative to uniform, averaged over 2–6 h, minus 1 (signed), with its
    particle-noise standard error.
    """
    blh, ustar, shf = WM_REGIMES[regime]
    rec = list(np.arange(7200.0, 21600.0 + 1, 900.0))
    res = _hanna_well_mixed(blh, ustar, shf, n=n, dt=dt, record_s=rec, seed=seed,
                            substep_c=substep_c)  # fmt: skip
    zs = np.stack(list(res.values()))  # [time, particle]

    def occ(idx):
        return (zs[:, idx] < 0.05 * blh).mean() / 0.05

    return occ(slice(None)) - 1.0, float(core.subset_spread(occ, zs.shape[1]))


def _value_ticks(ax, values) -> None:
    """Label a log x axis at the sampled values only."""
    from matplotlib.ticker import FixedLocator, NullLocator

    ax.xaxis.set_major_locator(FixedLocator(list(values)))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xticklabels([f"{v:g}" for v in values])


def convergence(out_dir: Path) -> Path:
    cfg = dict(
        rotation_dts=[240.0, 120.0, 60.0, 30.0, 15.0],
        euler_dts=[160.0, 80.0, 40.0, 20.0, 10.0],
        euler_n=400_000,
        outer_dts=[60.0, 30.0, 15.0, 7.5],
        substep_cs={"60": [2.0, 1.0, 0.5, 0.25, 0.125], "15": [2.0, 0.5, 0.125]},
        hanna_case=dict(n=10_000, seed=11, measure="lowest 5% of the BL, 2-6 h"),
        particle_counts=[400, 1600, 6400, 25600, 102400],
    )
    rot = _rotation_errors(cfg["rotation_dts"])
    eb, eb_se = _euler_bias(cfg["euler_dts"], n=cfg["euler_n"])
    outer = {
        reg: np.array([_near_surface_error(reg, dt=dt) for dt in cfg["outer_dts"]])
        for reg in WM_REGIMES
    }
    sub = {
        dt: np.array([_near_surface_error("neutral", dt=float(dt), substep_c=c) for c in cs])
        for dt, cs in cfg["substep_cs"].items()
    }
    pc, pc_se = _particle_count_errors(cfg["particle_counts"])

    spec = _spec("convergence")
    fig, axes = core.new_figure(4, width=3.7, height=3.7)
    a1, a2, a3, a4 = axes[0]

    def slope_line(ax, x, y_anchor, slope, label):
        x = np.asarray(x, dtype=float)
        y = y_anchor * (x / x[0]) ** slope
        ax.plot(x, y, color=style.EXACT, ls="--", lw=1.2)
        ax.annotate(label, (x[-1], y[-1]), xytext=(4, -2), textcoords="offset points",
                    fontsize=8, color=style.INK2)  # fmt: skip

    def noise_floor(ax, level, label="particle-noise floor (2 s.e.)"):
        ax.axhspan(1e-9, level, color=style.MUTED, alpha=0.15, lw=0, label=label)

    def signed(ax, x, e, color, marker, label):
        """|error| on log axes; open markers where the error is negative."""
        ax.loglog(x, np.abs(e), "-", color=color, label=label)
        neg = e < 0
        ax.plot(x[~neg], np.abs(e[~neg]), marker, color=color, ls="none")
        ax.plot(x[neg], np.abs(e[neg]), marker, mfc=style.SURFACE, mec=color, mew=1.6,
                ls="none")  # fmt: skip

    # (a) time step, engine primitives
    x1 = np.array(cfg["rotation_dts"])
    a1.loglog(x1, rot, "o-", color=style.GLIDE, label="RK2 advection: rotation return")
    slope_line(a1, x1, rot[0] * 2.5, 2, "slope 2")
    x2 = np.array(cfg["euler_dts"])
    noise_floor(a1, 2 * float(eb_se.mean()))
    signed(a1, x2, eb, style.REFERENCE_MODEL, "s", "OU + displacement: σ_z bias")
    slope_line(a1, x2, abs(eb[0]) * 2.0, 1, "slope 1")
    a1.set_ylim(1e-4, 1.0)
    a1.set_xlabel("time step Δt (s)")
    a1.set_ylabel("relative error")
    a1.set_title("Time step: engine")
    a1.legend(loc="lower right", fontsize=7)
    core.recessive_grid(a1, "both")

    # (b) outer step, production Hanna, three regimes
    x3 = np.array(cfg["outer_dts"])
    floor = 0.0
    for (reg, arr), col, mk in zip(outer.items(), style.CATEGORICAL[:3], "os^"):
        signed(a2, x3, arr[:, 0], col, mk, reg)
        floor = max(floor, 2 * float(arr[:, 1].mean()))
    noise_floor(a2, floor)
    slope_line(a2, x3, abs(outer["neutral"][0, 0]) * 1.3, 1, "slope 1")
    a2.axvline(60.0, color=style.MUTED, lw=1, ls=":")
    a2.set_ylim(3e-3, 3.0)
    _value_ticks(a2, x3)
    a2.set_xlabel("outer step Δt (s); shipped 60 s")
    a2.set_ylabel("near-surface occupancy / uniform − 1")
    a2.set_title("Time step: Hanna scheme")
    a2.legend(loc="lower right", fontsize=7)
    core.recessive_grid(a2, "both")

    # (c) sub-step target, neutral
    floor = 0.0
    for (dt, arr), col, mk in zip(sub.items(), [style.GLIDE, style.ALT], "os"):
        signed(a3, np.array(cfg["substep_cs"][dt]), arr[:, 0], col, mk, f"outer step {dt} s")
        floor = max(floor, 2 * float(arr[:, 1].mean()))
    noise_floor(a3, floor)
    x60 = np.array(cfg["substep_cs"]["60"])
    slope_line(a3, x60, abs(sub["60"][0, 0]) * 1.3, 1, "slope 1")
    a3.axvline(0.5, color=style.MUTED, lw=1, ls=":")
    a3.set_ylim(3e-3, 3.0)
    _value_ticks(a3, x60)
    a3.set_xlabel("sub-step target substep_c; default 0.5")
    a3.set_ylabel("near-surface occupancy / uniform − 1")
    a3.set_title("Sub-step target: Hanna, neutral")
    a3.legend(loc="lower right", fontsize=7)
    core.recessive_grid(a3, "both")

    # (d) particle count
    x4 = np.array(cfg["particle_counts"])
    core.log_band(a4, x4, pc, pc_se)
    a4.loglog(x4, pc, "o-", color=style.GLIDE, label="RMS error of σ_z²")
    slope_line(a4, x4, pc[0] * 1.6, -0.5, "slope −½")
    a4.set_xlabel("particle count N")
    a4.set_ylabel("relative error")
    a4.set_title("Particle count")
    a4.legend(loc="lower left", fontsize=7)
    core.recessive_grid(a4, "both")
    fig.subplots_adjust(wspace=0.42, top=0.8)
    caption = (
        "Error against each numerical setting, on log axes, with the slope theory predicts "
        "(dashed); open markers are negative errors. (a) RK2 return error after one "
        "solid-body rotation (second order) and the bias of σ_z at 5 T_L against Taylor "
        "(first order), through the engine primitives. (b) The production Hanna scheme's "
        "near-surface well-mixed error (occupancy of the lowest 5% of the boundary layer "
        "relative to uniform, 2–6 h, the setup of the well-mixed figure) against the outer "
        "step, default sub-stepping. (c) The same, neutral, against the sub-step target "
        "substep_c at outer steps of 60 s and 15 s. Wilson & Flesch (1993) expect first "
        "order in the step. (d) Sampling error of σ_z² against particle count. Grey: the "
        "particle-noise floor (2 s.e., 10 particle subsets). A preview of the Tier 0 "
        "convergence figure; the full study is task A5."
    )
    metrics = {
        "rotation_order": float(np.polyfit(np.log(x1), np.log(rot), 1)[0]),
        "euler_order_3_coarsest": float(
            np.polyfit(np.log(x2[:3]), np.log(np.abs(eb[:3])), 1)[0]
        ),
        "euler_rel_err": [float(v) for v in eb],
        "particle_order": float(np.polyfit(np.log(x4), np.log(pc), 1)[0]),
        **{f"near_surface_err_{reg}_by_outer_dt": [float(v) for v in arr[:, 0]]
           for reg, arr in outer.items()},
        **{f"near_surface_err_neutral_dt{dt}_by_substep_c": [float(v) for v in arr[:, 0]]
           for dt, arr in sub.items()},
        "near_surface_err_se_max": float(
            max(arr[:, 1].max() for arr in [*outer.values(), *sub.values()])
        ),
    }  # fmt: skip
    return _finish(fig, out_dir, spec, cfg, caption, metrics,
                   "tests/test_dispersion_analytic.py (rotation, Euler bias); new: Hanna "
                   "outer step and substep_c, particle count")  # fmt: skip


# --------------------------------------------------------------------------- terrain


def _terrain_track(terrain_following: bool):
    import test_terrain_transport as tt

    from lpdm.main import _advect_active_particles
    from lpdm.met_reader import BoundingBoxRequest, SpatialBounds, TimeBounds

    ds = tt._hill_store(u_ms=15.0, hill_height_m=800.0, halfwidth_deg=0.5)
    reader = tt._InMemoryReader(ds, terrain_following=terrain_following)
    t0 = datetime(2024, 1, 1, tzinfo=UTC)
    window = reader.fetch_hourly_window(
        BoundingBoxRequest(
            spatial=SpatialBounds(lon_min=-3.5, lon_max=3.5, lat_min=44.7, lat_max=45.3,
                                  z_min=0.0, z_max=15000.0),
            time=TimeBounds(start=t0, end=t0 + timedelta(hours=1)),
        )
    )  # fmt: skip
    engine = _engine()
    p = torch.tensor([[2.5, 45.0, 50.0, 1.0]], dtype=torch.float64)
    lon, agl = [2.5], [50.0]
    for _ in range(400):
        p, _, _ = _advect_active_particles(engine, torch.device("cpu"), p, window,
                                           t0 + timedelta(minutes=30), 60.0, torch.float64)  # fmt: skip
        lon.append(float(p[0, 0]))
        agl.append(float(p[0, 2]))
        if lon[-1] < -2.0:
            break
    return np.array(lon), np.array(agl), ds


def terrain(out_dir: Path) -> Path:
    cfg = dict(u_ms=15.0, hill_height_m=800.0, halfwidth_deg=0.5, z0_agl=50.0, dt=60.0,
               reader="ArcoEra5ZarrReader (in-memory synthetic pressure-level store)")  # fmt: skip
    lon_t, agl_t, ds = _terrain_track(True)
    lon_f, agl_f, _ = _terrain_track(False)
    lon_g = ds["longitude"].values
    h = ds["geopotential_at_surface"].isel(time=0, latitude=1).values / 9.80665

    def terr(x):
        return np.interp(x, lon_g, h)

    spec = _spec("terrain")
    fig, axes = core.new_figure(2, width=5.4, height=3.6)
    a1, a2 = axes[0]
    xs = np.linspace(-2.2, 2.6, 400)
    a1.fill_between(xs, 0, terr(xs), color="#c9c7c0", lw=0, label="terrain")
    a1.plot(lon_t, terr(lon_t) + agl_t, color=style.GLIDE, label="terrain-following (GLIDE)")
    a1.plot(lon_f, terr(lon_f) + agl_f, color=style.ALT, ls="-", lw=1.6,
            label="no slope correction")  # fmt: skip
    a1.set_xlim(2.6, -2.2)  # backward in time: the particle moves west (upwind)
    a1.set_ylim(0, 1700)
    a1.set_xlabel("longitude (°), particle moves right to left (backward in time)")
    a1.set_ylabel("height above sea level (m)")
    a1.set_title("Trajectory over an 800 m hill")
    a1.legend(loc="upper left", fontsize=7.5)
    core.recessive_grid(a1)
    a2.axhline(50.0, color=style.EXACT, ls="--", lw=1.3, label="release height, 50 m")
    a2.plot(lon_t, agl_t, color=style.GLIDE, label="terrain-following (GLIDE)")
    a2.plot(lon_f, agl_f, color=style.ALT, lw=1.6, label="no slope correction")
    a2.set_xlim(2.6, -2.2)
    a2.set_ylim(0, 900)
    a2.set_xlabel("longitude (°)")
    a2.set_ylabel("height above ground (m)")
    a2.set_title("Height above ground")
    a2.legend(loc="upper left", fontsize=7.5)
    core.recessive_grid(a2)
    fig.subplots_adjust(wspace=0.3, top=0.8)
    hill = np.abs(lon_t) < 1.5
    hill_f = np.abs(lon_f) < 1.5
    caption = (
        "A particle released 50 m above ground in a 15 m s⁻¹ wind, run backward across a "
        "Gaussian hill through the real ERA5 reader resample, on a synthetic store whose "
        "vertical velocity is the terrain-following flow. With the slope correction it holds "
        "its height above ground; without it, it rides up with the terrain."
    )
    metrics = {
        "max_agl_excursion_m": float(np.abs(agl_t[hill] - 50).max()),
        "max_agl_excursion_no_correction_m": float(np.abs(agl_f[hill_f] - 50).max()),
    }
    return _finish(fig, out_dir, spec, cfg, caption, metrics, "tests/test_terrain_transport.py")


BUILDERS = {
    "taylor_dispersion": taylor_dispersion,
    "well_mixed": well_mixed,
    "analytic_plume": analytic_plume,
    "diffusion_limit": diffusion_limit,
    "convergence": convergence,
    "terrain": terrain,
}
