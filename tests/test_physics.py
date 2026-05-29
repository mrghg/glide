"""Physics regression tests for GPU LPDM core behavior.

These tests use analytical/mock wind fields and stochastic updates to validate:
1) RK2 advection precision in a uniform flow.
2) Gaussian spread under Langevin turbulence with zero mean wind.
3) Well-mixed behavior in a periodic turbulent domain.

Mass conservation is checked in each test by verifying the particle weight sum.
"""

from __future__ import annotations

import math

import torch

from lpdm.gpu_engine import CoordinateBounds, GPUEngine


def _make_particles(n: int, *, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    """Create deterministic particles with normalized total mass."""

    x = torch.linspace(-500.0, 500.0, steps=n, dtype=dtype)
    y = torch.linspace(1000.0, 2000.0, steps=n, dtype=dtype)
    z = torch.linspace(200.0, 400.0, steps=n, dtype=dtype)
    w = torch.full((n,), 1.0 / n, dtype=dtype)
    return torch.stack((x, y, z, w), dim=1)


def test_uniform_wind_advection_rk2_precision() -> None:
    """RK2 should be exact for constant wind and conserve weight."""

    torch.manual_seed(11)

    engine = GPUEngine(device="cpu", dtype=torch.float64)
    particles = _make_particles(2048)
    initial_mass = particles[:, 3].sum()

    wind = torch.tensor([2.5, -1.25, 0.3], dtype=torch.float64)

    def uniform_wind(xyz: torch.Tensor) -> torch.Tensor:
        return wind.view(1, 3).expand_as(xyz)

    dt = 10.0
    n_steps = 300
    for _ in range(n_steps):
        particles = engine.rk2_advect_backward(particles, dt_seconds=dt, wind_fn=uniform_wind)

    total_time = dt * n_steps
    initial_xyz = _make_particles(2048)[:, :3]
    expected_xyz = initial_xyz - wind * total_time

    max_abs_err = torch.max(torch.abs(particles[:, :3] - expected_xyz)).item()
    assert max_abs_err < 1e-9

    final_mass = particles[:, 3].sum()
    assert torch.allclose(initial_mass, final_mass, atol=0.0, rtol=0.0)


def test_rk2_advection_second_order_in_dt() -> None:
    """RK2 backward advection should converge at order 2 under a linear wind.

    With v(x) = -a*x the backward integrator advances x as exp(a*t). RK2 captures
    only the first three terms of that expansion, so global error scales as dt^2
    and halving dt should reduce it by ~4x. This distinguishes a true second-order
    scheme from first-order Euler, which the constant-wind test cannot do.
    """

    engine = GPUEngine(device="cpu", dtype=torch.float64)

    a = 1e-3  # 1/s
    total_time = 200.0
    initial = torch.tensor([[100.0, 0.0, 0.0, 1.0]], dtype=torch.float64)
    expected_x = 100.0 * math.exp(a * total_time)

    def linear_wind(xyz: torch.Tensor) -> torch.Tensor:
        out = torch.zeros_like(xyz)
        out[:, 0] = -a * xyz[:, 0]
        return out

    dts = [10.0, 5.0, 2.5, 1.25]
    errors: list[float] = []
    for dt in dts:
        n_steps = int(round(total_time / dt))
        p = initial.clone()
        for _ in range(n_steps):
            p = engine.rk2_advect_backward(p, dt_seconds=dt, wind_fn=linear_wind)
        errors.append(abs(p[0, 0].item() - expected_x))

    for i in range(len(errors) - 1):
        ratio = errors[i] / max(errors[i + 1], 1e-15)
        assert ratio > 3.5, (
            f"RK2 not second-order in dt: error ratio {ratio:.2f} at "
            f"dt={dts[i]}->{dts[i + 1]} (expected ~4x reduction)"
        )

    p_final = initial.clone()
    for _ in range(int(round(total_time / dts[-1]))):
        p_final = engine.rk2_advect_backward(p_final, dt_seconds=dts[-1], wind_fn=linear_wind)
    assert torch.allclose(p_final[:, 3].sum(), torch.tensor(1.0, dtype=torch.float64))


def test_reflect_surface_handles_boundary_cases() -> None:
    """Surface reflection should mirror below-surface particles around z_surface."""

    engine = GPUEngine(device="cpu", dtype=torch.float64)

    particles = torch.tensor(
        [
            [0.0, 0.0, 200.0, 0.25],
            [0.0, 0.0, 0.0, 0.25],
            [0.0, 0.0, -100.0, 0.25],
            [0.0, 0.0, -1500.0, 0.25],
        ],
        dtype=torch.float64,
    )
    initial_mass = particles[:, 3].sum()

    reflected = engine.reflect_surface(particles, z_surface=0.0)

    expected_z = torch.tensor([200.0, 0.0, 100.0, 1500.0], dtype=torch.float64)
    assert torch.allclose(reflected[:, 2], expected_z)
    assert torch.allclose(initial_mass, reflected[:, 3].sum())


def test_reflect_surface_nonzero_z() -> None:
    """Reflection must use z_surface as the mirror plane, not z=0."""

    engine = GPUEngine(device="cpu", dtype=torch.float64)

    z_surf = 100.0
    particles = torch.tensor(
        [
            [0.0, 0.0, 50.0, 1.0],
            [0.0, 0.0, 90.0, 1.0],
            [0.0, 0.0, 100.0, 1.0],
            [0.0, 0.0, 200.0, 1.0],
        ],
        dtype=torch.float64,
    )

    reflected = engine.reflect_surface(particles, z_surface=z_surf)

    expected_z = torch.tensor([150.0, 110.0, 100.0, 200.0], dtype=torch.float64)
    assert torch.allclose(reflected[:, 2], expected_z)


def test_apply_horizontal_turbulence_displaces_with_cos_lat_correction() -> None:
    """u_prime moves longitude (with cos-lat scaling); v_prime moves latitude."""

    engine = GPUEngine(device="cpu", dtype=torch.float64)

    # Two particles at different latitudes; same u_prime, v_prime.
    particles = torch.tensor(
        [
            [10.0, 0.0, 500.0, 0.5],
            [10.0, 60.0, 500.0, 0.5],
        ],
        dtype=torch.float64,
    )
    u_prime = torch.tensor([1.0, 1.0], dtype=torch.float64)  # 1 m/s eastward
    v_prime = torch.tensor([0.5, 0.5], dtype=torch.float64)  # 0.5 m/s northward
    dt = 100.0

    moved_back = engine.apply_horizontal_turbulence(particles, u_prime, v_prime, dt, backward=True)

    # Backward integration: positions go opposite direction (west / south).
    expected_d_lon_lat0 = -1.0 * dt / 111320.0
    expected_d_lon_lat60 = -1.0 * dt / (111320.0 * math.cos(math.radians(60.0)))
    expected_d_lat = -0.5 * dt / 110540.0

    assert abs(moved_back[0, 0].item() - (10.0 + expected_d_lon_lat0)) < 1e-12
    assert abs(moved_back[1, 0].item() - (10.0 + expected_d_lon_lat60)) < 1e-12
    # |Δlon at lat=60| should be ~2x |Δlon at lat=0|.
    assert abs(moved_back[1, 0] - 10.0).item() > 1.9 * abs(moved_back[0, 0] - 10.0).item()

    assert abs(moved_back[0, 1].item() - (0.0 + expected_d_lat)) < 1e-12
    assert abs(moved_back[1, 1].item() - (60.0 + expected_d_lat)) < 1e-12

    # Forward direction flips the sign.
    moved_fwd = engine.apply_horizontal_turbulence(particles, u_prime, v_prime, dt, backward=False)
    assert moved_fwd[0, 0].item() > particles[0, 0].item()
    assert moved_fwd[0, 1].item() > particles[0, 1].item()

    # Mass and altitude untouched.
    assert torch.allclose(moved_back[:, 2], particles[:, 2])
    assert torch.allclose(moved_back[:, 3], particles[:, 3])


def test_zero_wind_diffusion_langevin_gaussian_spread() -> None:
    """Zero-mean Langevin turbulence should produce near-Gaussian vertical spread."""

    torch.manual_seed(22)

    n = 30000
    engine = GPUEngine(device="cpu", dtype=torch.float64)

    particles = torch.zeros((n, 4), dtype=torch.float64)
    particles[:, 2] = 5000.0
    particles[:, 3] = 1.0 / n

    initial_mass = particles[:, 3].sum()

    w_prime = engine.initialize_turbulence_velocity(n).to(dtype=torch.float64)

    dt = 1.0
    t_lagrangian = 120.0
    sigma_w2 = 2.0
    n_steps = 600
    total_time = dt * n_steps

    for _ in range(n_steps):
        w_prime = engine.update_langevin_velocity(
            w_prime,
            t_lagrangian=t_lagrangian,
            sigma_w2=sigma_w2,
            dt_seconds=dt,
        )
        particles = engine.apply_vertical_turbulence(
            particles,
            w_prime,
            dt_seconds=dt,
            backward=False,
        )

    dz = particles[:, 2] - 5000.0
    mean = torch.mean(dz)
    std = torch.std(dz, unbiased=True)

    centered = dz - mean
    skew = torch.mean((centered / std) ** 3)
    excess_kurt = torch.mean((centered / std) ** 4) - 3.0

    # Mean displacement should be close to zero relative to sampling uncertainty.
    stderr = std / torch.sqrt(torch.tensor(float(n), dtype=std.dtype))

    # Integrated OU process variance for z(t) with stationary velocity variance sigma_w2.
    expected_var = 2.0 * sigma_w2 * t_lagrangian * (
        total_time - t_lagrangian * (1.0 - torch.exp(torch.tensor(-total_time / t_lagrangian)))
    )
    expected_std = torch.sqrt(expected_var)

    assert abs(mean.item()) < 4.0 * stderr.item()
    assert 0.85 * expected_std.item() <= std.item() <= 1.15 * expected_std.item()
    assert abs(skew.item()) < 0.2
    assert abs(excess_kurt.item()) < 0.35

    final_mass = particles[:, 3].sum()
    assert torch.allclose(initial_mass, final_mass, atol=0.0, rtol=0.0)


def test_langevin_drift_term_is_deterministic_increment() -> None:
    """With zero noise, the OU update is a*w_prev + drift*dt; drift=0 is the
    legacy behaviour (guards backward compatibility)."""

    engine = GPUEngine(device="cpu", dtype=torch.float64)
    w_prev = torch.tensor([0.2, -0.1, 0.0], dtype=torch.float64)
    zero_noise = torch.zeros_like(w_prev)
    tl, sig2, dt = 100.0, 1.0, 10.0
    a = math.exp(-dt / tl)

    no_drift = engine.update_langevin_velocity(
        w_prev, t_lagrangian=tl, sigma_w2=sig2, dt_seconds=dt, noise=zero_noise,
    )
    assert torch.allclose(no_drift, a * w_prev, atol=1e-12)

    drift = torch.tensor([0.01, 0.02, -0.03], dtype=torch.float64)
    with_drift = engine.update_langevin_velocity(
        w_prev, t_lagrangian=tl, sigma_w2=sig2, dt_seconds=dt, noise=zero_noise, drift=drift,
    )
    assert torch.allclose(with_drift, a * w_prev + drift * dt, atol=1e-12)


def test_well_mixed_condition_drift_keeps_uniform_distribution() -> None:
    """Thomson well-mixed test: in inhomogeneous turbulence (σ_w² varying with
    height) an initially-uniform tracer must stay uniform WITH the drift term,
    and must spuriously accumulate in the low-σ region WITHOUT it."""

    import math as _math

    torch.manual_seed(7)
    engine = GPUEngine(device="cpu", dtype=torch.float64)

    H = 1000.0
    n = 60000
    tl = 100.0
    dt = 1.0
    n_steps = 4000
    # σ_w²(z) linear: 0.5 at surface → 2.0 at top. ∂σ²/∂z constant.
    sig2_base, sig2_top = 0.5, 2.0
    dsig2_dz = (sig2_top - sig2_base) / H

    def evolve(use_drift: bool) -> torch.Tensor:
        z = torch.rand(n, dtype=torch.float64) * H  # uniform initial (well-mixed)
        w = torch.zeros(n, dtype=torch.float64)
        for _ in range(n_steps):
            sig2 = sig2_base + (sig2_top - sig2_base) * (z / H)
            drift = (
                0.5 * (1.0 + w.pow(2) / sig2) * dsig2_dz if use_drift else 0.0
            )
            w = engine.update_langevin_velocity(
                w, t_lagrangian=tl, sigma_w2=sig2, dt_seconds=dt, drift=drift,
            )
            z = z + w * dt
            # Reflect at both boundaries (displacement << H each step).
            z = torch.where(z < 0.0, -z, z)
            z = torch.where(z > H, 2.0 * H - z, z)
        return z

    # Fraction of particles in the low-σ bottom third of the column.
    z_drift = evolve(use_drift=True)
    z_nodrift = evolve(use_drift=False)
    bottom_drift = float((z_drift < H / 3.0).float().mean())
    bottom_nodrift = float((z_nodrift < H / 3.0).float().mean())

    # With drift: stays ~uniform → ~1/3 in the bottom third.
    assert abs(bottom_drift - 1.0 / 3.0) < 0.05
    # Without drift: spurious accumulation in the low-σ bottom third.
    assert bottom_nodrift > bottom_drift + 0.08


def test_well_mixed_uniformity_in_periodic_turbulence() -> None:
    """Uniform particles in a periodic domain should remain close to uniform."""

    torch.manual_seed(33)

    n = 50000
    engine = GPUEngine(device="cpu", dtype=torch.float64)

    bounds = CoordinateBounds(
        lon_min=0.0,
        lon_max=1.0,
        lat_min=0.0,
        lat_max=1.0,
        alt_min=0.0,
        alt_max=1.0,
    )

    particles = torch.empty((n, 4), dtype=torch.float64)
    particles[:, 0] = torch.rand(n, dtype=torch.float64)
    particles[:, 1] = torch.rand(n, dtype=torch.float64)
    particles[:, 2] = torch.rand(n, dtype=torch.float64)
    particles[:, 3] = 1.0 / n

    initial_mass = particles[:, 3].sum()

    dt = 1.0
    diffusivity = 0.001
    for _ in range(250):
        particles = engine.diffuse_positions_periodic(
            particles,
            diffusivity=diffusivity,
            dt_seconds=dt,
            bounds=bounds,
        )

    n_bins = 8
    idx_x = torch.clamp((particles[:, 0] * n_bins).long(), min=0, max=n_bins - 1)
    idx_y = torch.clamp((particles[:, 1] * n_bins).long(), min=0, max=n_bins - 1)
    idx_z = torch.clamp((particles[:, 2] * n_bins).long(), min=0, max=n_bins - 1)
    flat_idx = idx_x + n_bins * idx_y + (n_bins * n_bins) * idx_z

    counts = torch.bincount(flat_idx, minlength=n_bins**3).to(torch.float64)
    expected = float(n) / float(n_bins**3)

    # Relative RMS deviation of occupancy from ideal uniform occupancy.
    rel_rms = torch.sqrt(torch.mean(((counts - expected) / expected) ** 2)).item()
    assert rel_rms < 0.12

    final_mass = particles[:, 3].sum()
    assert torch.allclose(initial_mass, final_mass, atol=0.0, rtol=0.0)
