"""Analytic dispersion verification (dev/TEST_REVIEW_2026-07-16.md).

Quantitative checks of the stochastic transport against closed-form results, at
the engine-OU level with prescribed constant (sigma_w, T_L):

- T5a: OU autocorrelation R(tau)=exp(-tau/T_L) and stationary variance.
- T5b: solid-body-rotation advection (circle closure + RK2 second order).
- T1:  Taylor (1921) dispersion curve sigma_z^2(t) across ballistic -> diffusive.

Why engine-level, not through `HannaScheme.step`: Hanna has NO homogeneous regime
by construction (T_L is intrinsically height-dependent — that inhomogeneity is
exactly what the Thomson well-mixed drift exists to correct), so Taylor/OU
statistics have no closed form through the assembled scheme. The scheme's
integration is covered instead by the well-mixed tests (inhomogeneous, through the
full step) and the static/dynamic substep-equivalence tests; these fill the
missing homogeneous-OU quantitative baseline.
"""

from __future__ import annotations

import math

import torch

from lpdm.gpu_engine import GPUEngine


def _taylor_var(sigma_w2: float, t_l: float, t: float) -> float:
    """Taylor (1921) displacement variance for an OU velocity process."""
    return 2.0 * sigma_w2 * t_l * (t - t_l * (1.0 - math.exp(-t / t_l)))


# --------------------------------------------------------------------------- #
# T5a — OU autocorrelation & stationarity
# --------------------------------------------------------------------------- #


def test_ou_autocorrelation_and_stationarity() -> None:
    """The turbulent-velocity OU process must be stationary at Var=sigma_w^2 with
    autocorrelation R(tau)=exp(-tau/T_L). Nothing else pins the *correlation*
    structure (the diffusion tests only see the integrated spread), yet T_L is
    what sets dispersion timescales and is retuned via substep_c/max_substeps."""

    torch.manual_seed(4111)
    engine = GPUEngine(device="cpu", dtype=torch.float64)

    n = 100_000
    sigma_w2 = 0.7
    t_l = 100.0
    dt = 5.0
    sigma = math.sqrt(sigma_w2)

    # Start from the stationary distribution and burn in ~2 T_L to shed any bias.
    w = torch.randn(n, dtype=torch.float64) * sigma
    for _ in range(int(2 * t_l / dt)):
        w = engine.update_langevin_velocity(w, t_lagrangian=t_l, sigma_w2=sigma_w2, dt_seconds=dt)

    # Record frames spanning tau up to ~3 T_L; ensemble (n) + time-origin averaging.
    n_frames = int(3 * t_l / dt) + 1  # 61 frames
    frames = torch.empty((n_frames, n), dtype=torch.float64)
    for k in range(n_frames):
        frames[k] = w
        w = engine.update_langevin_velocity(w, t_lagrangian=t_l, sigma_w2=sigma_w2, dt_seconds=dt)

    var = frames.var(unbiased=True).item()
    assert 0.97 * sigma_w2 <= var <= 1.03 * sigma_w2, f"stationary Var(w')={var:.4f} != {sigma_w2}"

    centered = frames - frames.mean()
    for tau_over_tl in (0.5, 1.0, 2.0):
        lag = int(round(tau_over_tl * t_l / dt))
        r = (centered[:-lag] * centered[lag:]).mean().item() / var
        expected = math.exp(-tau_over_tl)
        assert abs(r - expected) < 0.02, (
            f"autocorrelation R(tau={tau_over_tl} T_L)={r:.4f}, expected {expected:.4f}"
        )


# --------------------------------------------------------------------------- #
# T5b — solid-body-rotation advection
# --------------------------------------------------------------------------- #


def _run_rotation(engine: GPUEngine, r0: torch.Tensor, omega: float, dt: float, period: float) -> torch.Tensor:
    """Backward-advect through a solid-body rotation field v = omega x (x - c),
    centred at the origin, for one full period. Works in a local metres frame."""

    def wind_fn(xyz: torch.Tensor) -> torch.Tensor:
        out = torch.zeros_like(xyz)
        out[:, 0] = -omega * xyz[:, 1]
        out[:, 1] = omega * xyz[:, 0]
        return out  # d(xyz)/dt, m/s

    p = r0.clone()
    for _ in range(int(round(period / dt))):
        p = engine.rk2_advect_backward(p, dt_seconds=dt, wind_fn=wind_fn)
    return p


def test_solid_body_rotation_advection_returns_to_start() -> None:
    """In a rigid rotation the trajectory is a circle: after one full period a
    particle returns to its start, and RK2's return error is second-order in dt.
    A spatially-varying-wind check with curvature, beyond the constant/linear
    wind tests, that also exercises the backward integration sign."""

    engine = GPUEngine(device="cpu", dtype=torch.float64)
    omega = 2.0 * math.pi / 3600.0  # period 3600 s
    period = 3600.0

    # Particles at several radii/angles (columns [x, y, z, weight], local metres).
    start = torch.tensor(
        [
            [100_000.0, 0.0, 500.0, 0.25],
            [0.0, 50_000.0, 500.0, 0.25],
            [-70_000.0, 40_000.0, 500.0, 0.25],
            [30_000.0, -80_000.0, 500.0, 0.25],
        ],
        dtype=torch.float64,
    )

    dts = [60.0, 30.0, 15.0]
    errors: list[float] = []
    for dt in dts:
        final = _run_rotation(engine, start, omega, dt, period)
        errors.append(torch.max(torch.linalg.norm(final[:, :2] - start[:, :2], dim=1)).item())

    # Second order: halving dt cuts the return error ~4x.
    for i in range(len(errors) - 1):
        ratio = errors[i] / max(errors[i + 1], 1e-12)
        assert ratio > 3.5, f"RK2 rotation not second-order: ratio {ratio:.2f} at dt {dts[i]}->{dts[i+1]}"

    # Absolute closure at the finest dt: tiny vs the ~10^5 m orbit radius.
    r_max = torch.max(torch.linalg.norm(start[:, :2], dim=1)).item()
    assert errors[-1] < 1e-3 * r_max, f"return error {errors[-1]:.3g} m too large vs radius {r_max:.3g}"

    # z and weight untouched by horizontal advection.
    final = _run_rotation(engine, start, omega, dts[-1], period)
    assert torch.allclose(final[:, 2], start[:, 2])
    assert torch.allclose(final[:, 3], start[:, 3])


# --------------------------------------------------------------------------- #
# T1 — Taylor (1921) dispersion curve
# --------------------------------------------------------------------------- #


def test_taylor_dispersion_curve_ballistic_to_diffusive() -> None:
    """Vertical spread of an OU-driven ensemble follows Taylor (1921),
    sigma_z^2(t) = 2 sigma_w^2 T_L [t - T_L(1 - e^{-t/T_L})], across the
    ballistic (sigma_z ~ sigma_w t, t << T_L) and diffusive (sigma_z^2 ~ 2Kt,
    K = sigma_w^2 T_L, t >> T_L) limits. Upgrades the single-point spread check
    to the full curve through the engine's OU + displacement primitives."""

    torch.manual_seed(2201)
    engine = GPUEngine(device="cpu", dtype=torch.float64)

    n = 100_000
    sigma_w2 = 1.0
    t_l = 100.0
    dt = 1.0  # dt/T_L = 0.01 -> forward-Euler position bias ~1%
    sigma = math.sqrt(sigma_w2)

    particles = torch.zeros((n, 4), dtype=torch.float64)
    particles[:, 3] = 1.0 / n
    w = torch.randn(n, dtype=torch.float64) * sigma  # start velocity stationary

    checkpoints_tl = [0.1, 0.3, 1.0, 3.0, 10.0, 30.0]
    checkpoint_steps = [int(round(c * t_l / dt)) for c in checkpoints_tl]

    sigma_z: dict[float, float] = {}
    step = 0
    for c_tl, target in zip(checkpoints_tl, checkpoint_steps):
        while step < target:
            w = engine.update_langevin_velocity(w, t_lagrangian=t_l, sigma_w2=sigma_w2, dt_seconds=dt)
            particles = engine.apply_vertical_turbulence(particles, w, dt_seconds=dt, backward=False)
            step += 1
        sigma_z[c_tl] = torch.std(particles[:, 2], unbiased=True).item()

    # Full curve tight to Taylor at every checkpoint (observed ~0.1%; 5% bound
    # leaves ample CI/seed margin while still catching any %-level physics bug).
    for c_tl in checkpoints_tl:
        t = c_tl * t_l
        expected = math.sqrt(_taylor_var(sigma_w2, t_l, t))
        got = sigma_z[c_tl]
        assert abs(got - expected) / expected < 0.05, (
            f"sigma_z at t={c_tl} T_L = {got:.3f}, Taylor {expected:.3f} "
            f"(rel err {abs(got-expected)/expected:.3f})"
        )

    # Ballistic limit: sigma_z ~ sigma_w * t for t << T_L.
    t_bal = 0.1 * t_l
    assert abs(sigma_z[0.1] - sigma * t_bal) / (sigma * t_bal) < 0.05

    # Diffusive limit: sigma_z^2 ~ 2 K t for t >> T_L, K = sigma_w^2 T_L.
    k = sigma_w2 * t_l
    t_dif = 30.0 * t_l
    assert abs(sigma_z[30.0] ** 2 - 2.0 * k * t_dif) / (2.0 * k * t_dif) < 0.08

    # Mass conservation.
    assert torch.allclose(particles[:, 3].sum(), torch.tensor(1.0, dtype=torch.float64))


def test_taylor_dispersion_position_integration_bias_with_dt() -> None:
    """The forward-Euler position update biases sigma_z at large dt/T_L. Pin the
    documented direction/size: at dt/T_L=0.01 the spread matches Taylor within
    ~2%; at dt/T_L=0.2 it is biased but bounded (< ~15%). Guards against a silent
    regression that widens the production-dt error."""

    t_l = 100.0
    sigma_w2 = 1.0
    total_t = 5.0 * t_l
    expected = math.sqrt(_taylor_var(sigma_w2, t_l, total_t))

    def spread_at_dt(dt: float, seed: int) -> float:
        torch.manual_seed(seed)
        engine = GPUEngine(device="cpu", dtype=torch.float64)
        n = 100_000
        p = torch.zeros((n, 4), dtype=torch.float64)
        w = torch.randn(n, dtype=torch.float64) * math.sqrt(sigma_w2)
        for _ in range(int(round(total_t / dt))):
            w = engine.update_langevin_velocity(w, t_lagrangian=t_l, sigma_w2=sigma_w2, dt_seconds=dt)
            p = engine.apply_vertical_turbulence(p, w, dt_seconds=dt, backward=False)
        return torch.std(p[:, 2], unbiased=True).item()

    fine = spread_at_dt(1.0, 71)
    coarse = spread_at_dt(20.0, 71)
    assert abs(fine - expected) / expected < 0.02, f"fine-dt spread off Taylor by {abs(fine-expected)/expected:.3f}"
    assert abs(coarse - expected) / expected < 0.15, f"coarse-dt spread off Taylor by {abs(coarse-expected)/expected:.3f}"
