"""Analytic Gaussian-plume footprint — the flagship end-to-end check (docs/VALIDATION.md §3).

Backward release from height z_r in uniform wind U with homogeneous OU turbulence
(sigma_v = sigma_w, T_L) and ground reflection. With deterministic along-wind
motion (u' = 0, so x(t) = -U t exactly) and independent OU crosswind/vertical
spread, the expected RAW residence time in a surface-layer cell is exact:

    R(cell) = (dx/U) * [Phi(y_hi/sy) - Phi(y_lo/sy)]
                     * [Phi((h-z_r)/sz) - Phi(-z_r/sz) + Phi((h+z_r)/sz) - Phi(z_r/sz)]

with sy(t), sz(t) from Taylor (1921) at travel time t = |x|/U — cell-integrated
(erf), so the analytic reference carries no discretisation approximations. This is
the only test that verifies the whole chain — advection + OU + reflection +
gridder binning — in footprint SHAPE and ABSOLUTE MAGNITUDE against a closed form;
the STILT unit conversion is then checked as an exact scale factor.

One simulation (module fixture, ~10 s, n=200k) feeds all assertions. Observed
agreement at seed 9021: total magnitude ratio 0.992; crosswind-integrated column
error max 0.4% (2-5 T_L band), worst-case ~10% in far-field low-occupancy columns
(statistical); correlation >= 0.9985; sigma_y within 1.1% of Taylor. Tolerances
are set at ~2-3x observed.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
import xarray as xr
from scipy.special import ndtr

from lpdm.comparison import to_stilt_surface_footprint
from lpdm.footprint_gridder import FootprintGridder
from lpdm.gpu_engine import GPUEngine

U_MS = 5.0
SIGMA2 = 0.25  # sigma_v = sigma_w = 0.5 m/s
T_L = 100.0
Z_R = 50.0
H_SURF = 40.0
DT = 5.0
N_STEPS = 800  # 4000 s backward -> plume to x = -20 km
N_PART = 200_000
M_PER_DEG_LON = 111320.0  # engine constants at lat ~ 0
M_PER_DEG_LAT = 110540.0
DX_M, DY_M = 250.0, 50.0  # DX/U = 50 s = 10 samples/cell, edges aligned with dt
NX, NY = 82, 100
X0_M, Y0_M = -20250.0, -2500.0  # grid origin (west/south edges)


def _taylor_sigma(t: float) -> float:
    return math.sqrt(2.0 * SIGMA2 * T_L * (t - T_L * (1.0 - math.exp(-t / T_L))))


@pytest.fixture(scope="module")
def plume() -> dict:
    """Run the backward plume once; return raw surface footprint + analytic field."""
    torch.manual_seed(9021)
    engine = GPUEngine(device="cpu", dtype=torch.float64)

    gridder = FootprintGridder(
        lon_bounds=(X0_M / M_PER_DEG_LON, (X0_M + NX * DX_M) / M_PER_DEG_LON),
        lat_bounds=(Y0_M / M_PER_DEG_LAT, (Y0_M + NY * DY_M) / M_PER_DEG_LAT),
        z_edges_m=(0.0, H_SURF, 10000.0),
        n_time_bins=1,
        n_y=NY,
        n_x=NX,
        device="cpu",
        dtype=torch.float64,
    )

    particles = torch.zeros((N_PART, 4), dtype=torch.float64)
    particles[:, 2] = Z_R
    particles[:, 3] = 1.0 / N_PART
    sigma = math.sqrt(SIGMA2)
    # Stationary initial velocities — the correct receptor-release initialisation
    # (and what Taylor's formula assumes).
    v_prime = torch.randn(N_PART, dtype=torch.float64) * sigma
    w_prime = torch.randn(N_PART, dtype=torch.float64) * sigma
    u_zero = torch.zeros(N_PART, dtype=torch.float64)
    t_idx = torch.zeros(N_PART, dtype=torch.int64)
    release_idx = torch.zeros(N_PART, dtype=torch.int64)
    active = torch.ones(N_PART, dtype=torch.bool)

    def wind_fn(xyz: torch.Tensor) -> torch.Tensor:
        out = torch.zeros_like(xyz)
        out[:, 0] = U_MS / M_PER_DEG_LON  # uniform eastward wind, deg/s
        return out

    for _ in range(N_STEPS):
        # Mirror the production step order: advect, OU updates, displacements,
        # reflect, accumulate.
        particles = engine.rk2_advect_backward(particles, dt_seconds=DT, wind_fn=wind_fn)
        v_prime = engine.update_langevin_velocity(
            v_prime, t_lagrangian=T_L, sigma_w2=SIGMA2, dt_seconds=DT
        )
        w_prime = engine.update_langevin_velocity(
            w_prime, t_lagrangian=T_L, sigma_w2=SIGMA2, dt_seconds=DT
        )
        particles = engine.apply_horizontal_turbulence(
            particles, u_zero, v_prime, dt_seconds=DT, backward=True
        )
        particles = engine.apply_vertical_turbulence(
            particles, w_prime, dt_seconds=DT, backward=True
        )
        particles, w_prime = engine.reflect_surface(particles, w_prime, z_surface=0.0)
        gridder.accumulate(
            particles=particles[:, :3],
            active_mask=active,
            weights=particles[:, 3],
            t_idx=t_idx,
            release_idx=release_idx,
            dt_seconds=DT,
        )

    raw = gridder.tensor[0, 0, 0].numpy()  # surface bin, [NY, NX], seconds
    x_centres = (np.arange(NX) + 0.5) * DX_M + X0_M
    y_centres = (np.arange(NY) + 0.5) * DY_M + Y0_M
    y_edges = np.arange(NY + 1) * DY_M + Y0_M

    analytic = np.zeros_like(raw)
    for j in range(NX):
        if x_centres[j] >= 0.0:
            continue  # at/downwind of the receptor
        t = -x_centres[j] / U_MS
        s_y, s_z = _taylor_sigma(t), _taylor_sigma(t)
        p_y = ndtr(y_edges[1:] / s_y) - ndtr(y_edges[:-1] / s_y)
        p_z = (ndtr((H_SURF - Z_R) / s_z) - ndtr(-Z_R / s_z)) + (
            ndtr((H_SURF + Z_R) / s_z) - ndtr(Z_R / s_z)
        )
        analytic[:, j] = (DX_M / U_MS) * p_y * p_z

    travel_t = -x_centres / U_MS
    return dict(
        raw=raw, analytic=analytic, x_centres=x_centres, y_centres=y_centres, travel_t=travel_t
    )


def _band(plume: dict, t_lo: float, t_hi: float) -> np.ndarray:
    return np.where((plume["travel_t"] >= t_lo) & (plume["travel_t"] <= t_hi))[0]


def test_footprint_matches_analytic_gaussian_plume(plume: dict) -> None:
    """Crosswind-integrated footprint tracks the plume solution in every band
    beyond the near field, and the 2-D pattern correlates > 0.995."""
    for t_lo, t_hi, tol_max, tol_mean in [
        (2 * T_L, 5 * T_L, 0.05, 0.02),  # obs max 0.004
        (5 * T_L, 10 * T_L, 0.15, 0.05),  # obs max 0.102 (far-field statistics)
        (10 * T_L, N_STEPS * DT, 0.15, 0.05),
    ]:
        cols = _band(plume, t_lo, t_hi)
        got = plume["raw"][:, cols].sum(axis=0)
        want = plume["analytic"][:, cols].sum(axis=0)
        rel = np.abs(got - want) / want
        assert rel.max() < tol_max, f"band {t_lo}-{t_hi}s: max col err {rel.max():.3f}"
        assert rel.mean() < tol_mean, f"band {t_lo}-{t_hi}s: mean col err {rel.mean():.3f}"

    cols = _band(plume, 2 * T_L, N_STEPS * DT)
    g, a = plume["raw"][:, cols].ravel(), plume["analytic"][:, cols].ravel()
    corr = float(np.corrcoef(g, a)[0, 1])
    assert corr > 0.995, f"2-D pattern correlation {corr:.4f}"


def test_footprint_absolute_magnitude_matches_plume(plume: dict) -> None:
    """Total surface residence within 3% of the analytic plume (obs ratio 0.992)
    — the absolute-units check nothing else in the suite provides."""
    cols = _band(plume, 2 * T_L, N_STEPS * DT)
    ratio = plume["raw"][:, cols].sum() / plume["analytic"][:, cols].sum()
    assert 0.97 < ratio < 1.03, f"total surface residence ratio {ratio:.4f}"


def test_footprint_crosswind_width_matches_taylor(plume: dict) -> None:
    """Moment-based sigma_y (Sheppard-corrected) within 5% of Taylor at several
    downwind distances (obs <= 1.1%)."""
    yc = plume["y_centres"]
    cols = _band(plume, 2 * T_L, N_STEPS * DT)
    for c in cols[:: max(1, len(cols) // 6)]:
        col = plume["raw"][:, c]
        assert col.sum() > 0
        mu = float((col * yc).sum() / col.sum())
        var = float((col * (yc - mu) ** 2).sum() / col.sum()) - DY_M**2 / 12.0
        got = math.sqrt(max(var, 1e-9))
        want = _taylor_sigma(plume["travel_t"][c])
        assert abs(got - want) / want < 0.05, (
            f"sigma_y at t={plume['travel_t'][c]:.0f}s: {got:.1f} vs Taylor {want:.1f}"
        )


def test_stilt_conversion_scales_raw_footprint_exactly(plume: dict) -> None:
    """The STILT conversion of this footprint is exactly raw * m_air/(h*rho) —
    wiring check that the analytic-magnitude result carries into STILT units."""
    m_air, rho = 0.02897, 1.2
    da = xr.DataArray(
        plume["raw"][None, None],  # (time_ago, z_bin, lat, lon)
        dims=("time_ago", "z_bin", "latitude", "longitude"),
        coords=dict(
            z_bottom_m=("z_bin", [0.0]),
            z_top_m=("z_bin", [H_SURF]),
            latitude=plume["y_centres"],
            longitude=plume["x_centres"],
        ),
    )
    stilt = to_stilt_surface_footprint(
        da,
        surface_layer_depth_m=H_SURF,
        air_density_kg_m3=rho,
        m_air_kg_per_mol=m_air,
        integrate_time=True,
    )
    expected = plume["raw"] * m_air / (H_SURF * rho)
    assert np.allclose(stilt.values, expected, rtol=1e-12, atol=0.0)


# ---------------------------------------------------------------------------
# Deposition position (task A2, [PR-F4], decision 0026): endpoint vs step mean
# ---------------------------------------------------------------------------

POS_N_PART = 50_000
POS_DX_M = 500.0  # 100 s of travel: a whole number of steps at DT and at 4 DT
POS_NX = 41
POS_X0_M = -POS_NX * POS_DX_M
POS_T_END = N_STEPS * DT


def _surface_columns_by_position(dt: float, seed: int) -> dict[str, np.ndarray]:
    """One backward plume at step `dt`, deposited at both positions from the SAME
    particles (paired, so the comparison is not swamped by particle noise). Returns
    the crosswind-integrated surface-layer residence per along-wind cell."""

    from lpdm.main import _step_mean_position

    torch.manual_seed(seed)
    engine = GPUEngine(device="cpu", dtype=torch.float64)
    gridders = {
        position: FootprintGridder(
            lon_bounds=(POS_X0_M / M_PER_DEG_LON, (POS_X0_M + POS_NX * POS_DX_M) / M_PER_DEG_LON),
            lat_bounds=(Y0_M / M_PER_DEG_LAT, (Y0_M + NY * DY_M) / M_PER_DEG_LAT),
            z_edges_m=(0.0, H_SURF, 10000.0),
            n_time_bins=1,
            n_y=NY,
            n_x=POS_NX,
            device="cpu",
            dtype=torch.float64,
        )
        for position in ("endpoint", "step_mean")
    }
    n = POS_N_PART
    particles = torch.zeros((n, 4), dtype=torch.float64)
    particles[:, 2] = Z_R
    particles[:, 3] = 1.0 / n
    sigma = math.sqrt(SIGMA2)
    v_prime = torch.randn(n, dtype=torch.float64) * sigma
    w_prime = torch.randn(n, dtype=torch.float64) * sigma
    u_zero = torch.zeros(n, dtype=torch.float64)
    zeros_idx = torch.zeros(n, dtype=torch.int64)
    active = torch.ones(n, dtype=torch.bool)

    def wind_fn(xyz: torch.Tensor) -> torch.Tensor:
        out = torch.zeros_like(xyz)
        out[:, 0] = U_MS / M_PER_DEG_LON
        return out

    for _ in range(int(round(POS_T_END / dt))):
        start_xyz = particles[:, :3].clone()
        particles = engine.rk2_advect_backward(particles, dt_seconds=dt, wind_fn=wind_fn)
        v_prime = engine.update_langevin_velocity(
            v_prime, t_lagrangian=T_L, sigma_w2=SIGMA2, dt_seconds=dt
        )
        w_prime = engine.update_langevin_velocity(
            w_prime, t_lagrangian=T_L, sigma_w2=SIGMA2, dt_seconds=dt
        )
        particles = engine.apply_horizontal_turbulence(
            particles, u_zero, v_prime, dt_seconds=dt, backward=True
        )
        particles = engine.apply_vertical_turbulence(
            particles, w_prime, dt_seconds=dt, backward=True
        )
        particles, w_prime = engine.reflect_surface(particles, w_prime, z_surface=0.0)
        positions = {
            "endpoint": particles[:, :3],
            "step_mean": _step_mean_position(start_xyz, particles[:, :3]),
        }
        for position, gridder in gridders.items():
            gridder.accumulate(
                particles=positions[position],
                active_mask=active,
                weights=particles[:, 3],
                t_idx=zeros_idx,
                release_idx=zeros_idx,
                dt_seconds=dt,
            )
    return {p: g.tensor[0, 0, 0].numpy().sum(axis=0) for p, g in gridders.items()}


def _analytic_surface_columns() -> np.ndarray:
    """Crosswind-integrated surface residence per along-wind cell, integrated over the
    cell in x (travel time) by a fine trapezoid, so it carries no quadrature error of
    its own at this resolution."""

    edges = np.arange(POS_NX + 1) * POS_DX_M + POS_X0_M
    out = np.zeros(POS_NX)
    for j in range(POS_NX):
        lo, hi = edges[j], min(edges[j + 1], 0.0)
        if lo >= 0.0:
            continue
        xs = np.linspace(lo, hi, 2001)
        s_z = np.array([_taylor_sigma(max(-x / U_MS, 1e-6)) for x in xs])
        s_z = np.maximum(s_z, 1e-9)
        p_z = (ndtr((H_SURF - Z_R) / s_z) - ndtr(-Z_R / s_z)) + (
            ndtr((H_SURF + Z_R) / s_z) - ndtr(Z_R / s_z)
        )
        out[j] = np.trapezoid(p_z, xs) / U_MS
    return out


def _position_test_columns(analytic: np.ndarray) -> np.ndarray:
    """Cells upwind of the receptor that the simulation covers completely (the
    farthest cell's upwind edge lies beyond the distance travelled, U x T)."""

    lower_edges = np.arange(POS_NX) * POS_DX_M + POS_X0_M
    return np.where((analytic > 0) & (lower_edges >= -U_MS * POS_T_END))[0]


def test_step_mean_deposition_no_worse_at_dt_and_better_at_4dt() -> None:
    """A2 acceptance ([PR-F4], with `step_mean` in place of the spec's `midpoint`,
    decision 0026): the mean relative error of the crosswind-integrated surface
    footprint, against the cell-integrated analytic plume, is no larger with
    `step_mean` deposition than with `endpoint` at the reference step DT, and smaller
    at 4 DT. Observed values for three seeds are in docs/physics.md section 7."""

    analytic = _analytic_surface_columns()
    cols = _position_test_columns(analytic)
    errors = {}
    for factor in (1, 4):
        got = _surface_columns_by_position(DT * factor, seed=9021)
        errors[factor] = {
            p: float(np.mean(np.abs(v[cols] - analytic[cols]) / analytic[cols]))
            for p, v in got.items()
        }
    assert errors[1]["step_mean"] <= errors[1]["endpoint"], errors
    assert errors[4]["step_mean"] < 0.75 * errors[4]["endpoint"], errors
