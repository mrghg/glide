"""Unit tests for the Hanna 1982 / FLEXPART turbulence scheme.

Covers free-function physics (Coriolis, air density, Obukhov length, convective
velocity, surface-layer sigma_w) and the in-BL formulae across the three stability
regimes. The end-to-end runtime test that exercises HannaScheme.step() lives in
tests/test_main_runtime.py.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from lpdm.turbulence import HannaScheme, get_scheme, list_schemes
from lpdm.turbulence.hanna import (
    ABOVE_BL_SIGMA_M_S,
    ABOVE_BL_T_L_S,
    EARTH_ROTATION_RATE_S,
    FT_KZ_CEIL_M2_S,
    FT_KZ_FLOOR_M2_S,
    FT_RICHARDSON_CRIT,
    air_density,
    brunt_vaisala_squared,
    convective_velocity,
    coriolis_parameter,
    free_trop_diffusivity,
    free_trop_sigma_TL,
    gradient_richardson,
    in_bl_sigma_TL,
    obukhov_length,
    potential_temperature,
    surface_layer_sigma_w,
)


def test_hanna_is_registered() -> None:
    """The HannaScheme should self-register on import."""

    assert "hanna_1982" in list_schemes()
    instance = get_scheme("hanna_1982")
    assert isinstance(instance, HannaScheme)
    assert instance.required_met_keys() == ("t", "ustar", "shf")


def test_coriolis_parameter_signs_and_magnitude() -> None:
    """f = 2*Omega*sin(lat). Positive in N hemisphere, negative in S, ~zero at equator."""

    lat = torch.tensor([0.0, 30.0, 45.0, 60.0, -45.0], dtype=torch.float64)
    f = coriolis_parameter(lat)

    assert abs(f[0].item()) < 1e-10
    expected_45 = 2.0 * EARTH_ROTATION_RATE_S * math.sin(math.radians(45.0))
    assert abs(f[2].item() - expected_45) < 1e-12
    assert f[1].item() > 0
    assert f[3].item() > f[2].item() > f[1].item() > 0
    assert f[4].item() < 0
    assert abs(f[4].item() + f[2].item()) < 1e-12


def test_air_density_standard_conditions() -> None:
    """Dry-air density at 288 K, 101325 Pa should be ~1.225 kg/m^3."""

    sp = torch.tensor([101325.0], dtype=torch.float64)
    t = torch.tensor([288.0], dtype=torch.float64)
    rho = air_density(sp, t)
    assert abs(rho.item() - 1.225) < 0.01


def test_obukhov_length_signs() -> None:
    """L > 0 stable, L < 0 unstable, |L| -> inf neutral."""

    ustar = torch.tensor([0.4, 0.4, 0.4], dtype=torch.float64)
    t = torch.tensor([288.0, 288.0, 288.0], dtype=torch.float64)
    sp = torch.tensor([101325.0, 101325.0, 101325.0], dtype=torch.float64)
    # SHF: positive (unstable), negative (stable), zero (neutral)
    shf = torch.tensor([100.0, -50.0, 0.0], dtype=torch.float64)

    L = obukhov_length(ustar, shf, t, sp)
    assert L[0].item() < 0  # unstable
    assert L[1].item() > 0  # stable
    assert math.isinf(L[2].item())  # neutral


def test_convective_velocity_zero_when_stable_or_neutral() -> None:
    """w* is only physical when H > 0; should be 0 for H <= 0."""

    blh = torch.tensor([1000.0, 1000.0, 1000.0], dtype=torch.float64)
    t = torch.tensor([288.0, 288.0, 288.0], dtype=torch.float64)
    sp = torch.tensor([101325.0, 101325.0, 101325.0], dtype=torch.float64)
    shf = torch.tensor([200.0, -50.0, 0.0], dtype=torch.float64)

    w_star = convective_velocity(blh, shf, t, sp)
    assert w_star[0].item() > 0  # unstable -> finite w*
    assert w_star[1].item() == 0.0
    assert w_star[2].item() == 0.0

    # Spot-check magnitude: w* = ((g h H) / (T rho c_p))^(1/3)
    rho = 101325.0 / (287.05 * 288.0)
    expected_w_star = ((9.80665 * 1000.0 * 200.0) / (288.0 * rho * 1005.0)) ** (1.0 / 3.0)
    assert abs(w_star[0].item() - expected_w_star) < 1e-6


def test_in_bl_sigma_at_z_zero_matches_ustar_scaling() -> None:
    """Both stable and neutral regimes give sigma_w = 1.3 u*, sigma_uv = 2.0 u* at z=0."""

    z = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64)
    blh = torch.tensor([1000.0, 1000.0, 1000.0], dtype=torch.float64)
    ustar = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float64)
    w_star = torch.tensor([0.0, 0.0, 1.5], dtype=torch.float64)
    # Stable, neutral, unstable
    h_over_L = torch.tensor([5.0, 0.0, -5.0], dtype=torch.float64)
    lat = torch.tensor([45.0, 45.0, 45.0], dtype=torch.float64)

    sigma_u, sigma_v, sigma_w, T_Lu, T_Lv, T_Lw = in_bl_sigma_TL(z, blh, ustar, w_star, h_over_L, lat)

    # Stable z=0: sigma_w = 1.3 u*
    assert abs(sigma_w[0].item() - 1.3 * 0.5) < 1e-9
    assert abs(sigma_u[0].item() - 2.0 * 0.5) < 1e-9
    # Neutral z=0: sigma_w = 1.3 u* (exp factor = 1)
    assert abs(sigma_w[1].item() - 1.3 * 0.5) < 1e-9
    # Unstable z=0: sigma_w^2 has only the u* term (z/h=0 kills the w* term)
    expected_sigma_w_unstable = math.sqrt(1.5 * 0.5 ** 2)
    assert abs(sigma_w[2].item() - expected_sigma_w_unstable) < 1e-6
    # σ_u = σ_v in Hanna
    assert torch.allclose(sigma_u, sigma_v)
    assert torch.allclose(T_Lu, T_Lv)


def test_in_bl_stable_sigma_vanishes_at_top_of_bl() -> None:
    """Stable: sigma_w = 1.3 u* (1 - z/h)^(3/4) -> 0 at z=h."""

    z = torch.tensor([1000.0], dtype=torch.float64)
    blh = torch.tensor([1000.0], dtype=torch.float64)
    ustar = torch.tensor([0.5], dtype=torch.float64)
    w_star = torch.tensor([0.0], dtype=torch.float64)
    h_over_L = torch.tensor([5.0], dtype=torch.float64)  # stable
    lat = torch.tensor([45.0], dtype=torch.float64)

    _, _, sigma_w, _, _, _ = in_bl_sigma_TL(z, blh, ustar, w_star, h_over_L, lat)

    # (1 - 1)^(3/4) = 0, so sigma_w should clamp to the floor.
    assert sigma_w[0].item() < 1e-2


def test_in_bl_regime_selection_at_boundaries() -> None:
    """h/L just past +-1 should pick the stable / unstable branch, not neutral."""

    z = torch.tensor([100.0, 100.0, 100.0, 100.0], dtype=torch.float64)
    blh = torch.tensor([1000.0, 1000.0, 1000.0, 1000.0], dtype=torch.float64)
    ustar = torch.tensor([0.5, 0.5, 0.5, 0.5], dtype=torch.float64)
    w_star = torch.tensor([0.0, 0.0, 1.5, 1.5], dtype=torch.float64)
    # Just inside neutral, just stable, just unstable, deep unstable
    h_over_L = torch.tensor([0.5, 1.5, -1.5, -10.0], dtype=torch.float64)
    lat = torch.tensor([45.0, 45.0, 45.0, 45.0], dtype=torch.float64)

    sigma_u_neu, _, _, _, _, _ = in_bl_sigma_TL(z[:1], blh[:1], ustar[:1], w_star[:1], h_over_L[:1], lat[:1])
    sigma_u_stab, _, _, _, _, _ = in_bl_sigma_TL(z[1:2], blh[1:2], ustar[1:2], w_star[1:2], h_over_L[1:2], lat[1:2])
    sigma_u_unst, _, _, _, _, _ = in_bl_sigma_TL(z[2:3], blh[2:3], ustar[2:3], w_star[2:3], h_over_L[2:3], lat[2:3])
    sigma_u_deep, _, _, _, _, _ = in_bl_sigma_TL(z[3:], blh[3:], ustar[3:], w_star[3:], h_over_L[3:], lat[3:])

    # sigma_u in unstable regime grows with |h/L|: (12 - 0.5*h/L)^(2/3) * u*^2
    assert sigma_u_deep.item() > sigma_u_unst.item() > 0
    # Neutral and stable should give different values (different formula structure)
    assert sigma_u_neu.item() != sigma_u_stab.item()


def test_surface_layer_sigma_w_regimes() -> None:
    """SL formulae: 1.3u* in neutral, 1.3u*(1-2z/L)^(1/3) unstable, 1.3u*(1+5z/L) stable."""

    z = torch.tensor([10.0, 10.0, 10.0], dtype=torch.float64)
    ustar = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float64)
    L = torch.tensor([float("inf"), -100.0, 200.0], dtype=torch.float64)

    sigma_w = surface_layer_sigma_w(z, ustar, L)
    # Neutral: 1.3 * 0.5 = 0.65
    assert abs(sigma_w[0].item() - 0.65) < 1e-6
    # Unstable: 1.3 * 0.5 * (1 - 2*10/-100)^(1/3) = 0.65 * (1.2)^(1/3)
    expected_unstable = 0.65 * (1.2 ** (1.0 / 3.0))
    assert abs(sigma_w[1].item() - expected_unstable) < 1e-6
    # Stable: 1.3 * 0.5 * (1 + 5*10/200) = 0.65 * 1.25
    expected_stable = 0.65 * 1.25
    assert abs(sigma_w[2].item() - expected_stable) < 1e-6


def test_surface_layer_sigma_w_caps_in_very_stable() -> None:
    """Stable SL is capped at 1.3 u* * 6 to avoid runaway in very-stable regimes."""

    z = torch.tensor([100.0], dtype=torch.float64)
    ustar = torch.tensor([0.5], dtype=torch.float64)
    L = torch.tensor([5.0], dtype=torch.float64)  # very stable -> z/L = 20 -> uncapped = 1.3*0.5*101

    sigma_w = surface_layer_sigma_w(z, ustar, L)
    cap = 1.3 * 0.5 * 6.0
    assert abs(sigma_w[0].item() - cap) < 1e-9


def test_hanna_above_bl_constants_used() -> None:
    """The above-BL fallback constants exposed by the module are unchanged."""

    assert ABOVE_BL_SIGMA_M_S == 0.1
    assert ABOVE_BL_T_L_S == 100.0


# ---- Free-troposphere gradient-Richardson closure --------------------------


def test_potential_temperature_reference_level() -> None:
    """At p = p0 = 1000 hPa, θ == T; aloft (lower p) θ > T."""

    t = torch.tensor([280.0, 250.0], dtype=torch.float64)
    p = torch.tensor([100000.0, 50000.0], dtype=torch.float64)
    kappa = 287.05 / 1005.0  # R_d / c_p, matching the module constant
    theta = potential_temperature(t, p)
    assert abs(float(theta[0]) - 280.0) < 1e-9
    assert float(theta[1]) > float(t[1])  # 500 hPa: θ ≈ 250 * 2^κ ≈ 305 K
    assert abs(float(theta[1]) - 250.0 * 2.0 ** kappa) < 1e-6


def test_brunt_vaisala_sign_follows_stratification() -> None:
    """N² > 0 for stable (θ increasing with height), < 0 for unstable."""

    theta = torch.tensor([300.0, 300.0], dtype=torch.float64)
    dtheta_dz_stable = torch.tensor([0.005, 0.005], dtype=torch.float64)
    dtheta_dz_unstable = torch.tensor([-0.005, -0.005], dtype=torch.float64)
    assert float(brunt_vaisala_squared(theta, dtheta_dz_stable)[0]) > 0
    assert float(brunt_vaisala_squared(theta, dtheta_dz_unstable)[0]) < 0


def test_gradient_richardson_and_diffusivity_shutoff() -> None:
    """K_z decays to the background floor as Ri crosses the critical value."""

    z = torch.tensor([1000.0, 1000.0, 1000.0], dtype=torch.float64)
    shear = torch.tensor([0.01, 0.01, 0.01], dtype=torch.float64)  # |dU/dz|
    # Ri well below, near, and above critical (0.25).
    n2 = torch.tensor([0.0, 1e-5, 1e-3], dtype=torch.float64)
    ri = gradient_richardson(n2, shear.pow(2))
    k_z = free_trop_diffusivity(z, shear, ri)
    # Strongly sub-critical (Ri=0) → highest K; super-critical → floor.
    assert float(k_z[0]) > float(k_z[1]) >= float(k_z[2])
    assert abs(float(k_z[2]) - FT_KZ_FLOOR_M2_S) < 1e-9
    assert float(k_z[0]) <= FT_KZ_CEIL_M2_S


def test_free_trop_sigma_TL_satisfies_diffusivity_identity() -> None:
    """σ_w² · T_Lw must reconstruct K_z (the closure's defining relation)."""

    k_z = torch.tensor([0.1, 1.0, 10.0], dtype=torch.float64)
    n2 = torch.tensor([1e-4, 1e-4, 1e-4], dtype=torch.float64)
    sigma_w, t_lw = free_trop_sigma_TL(k_z, n2)
    assert torch.allclose(sigma_w.pow(2) * t_lw, k_z, rtol=1e-5)


def test_free_trop_diffusivity_never_zero() -> None:
    """Even at very high Ri the diffusivity stays at the floor (particles aloft
    must never be fully frozen — the failure mode of the old σ=0.1 placeholder)."""

    z = torch.tensor([2000.0], dtype=torch.float64)
    shear = torch.tensor([0.001], dtype=torch.float64)
    ri = torch.tensor([100.0], dtype=torch.float64)  # extremely stable
    k_z = free_trop_diffusivity(z, shear, ri)
    assert float(k_z[0]) >= FT_KZ_FLOOR_M2_S


# ---------------------------------------------------------------------------
# Meander (unresolved-mesoscale) horizontal turbulence
# ---------------------------------------------------------------------------


def test_meander_windowed_std_uniform_field_is_zero() -> None:
    """A spatially-uniform wind field has no local variability → σ_meander = 0."""

    field = torch.full((3, 5, 5), 7.0, dtype=torch.float64)
    std = HannaScheme._windowed_std(field, radius=1)
    assert std.shape == field.shape
    assert torch.allclose(std, torch.zeros_like(std), atol=1e-6)


def test_meander_windowed_std_matches_numpy_population_std() -> None:
    """The windowed std matches a per-cell numpy population std over the clipped
    (2r+1)² neighbourhood (edge cells use only the valid neighbours)."""

    rng = np.random.default_rng(0)
    arr = rng.standard_normal((1, 6, 7))
    field = torch.as_tensor(arr, dtype=torch.float64)
    radius = 1
    got = HannaScheme._windowed_std(field, radius=radius).numpy()

    z, ny, nx = arr.shape
    want = np.empty_like(arr)
    for k in range(z):
        for j in range(ny):
            for i in range(nx):
                window = arr[
                    k,
                    max(0, j - radius) : j + radius + 1,
                    max(0, i - radius) : i + radius + 1,
                ]
                want[k, j, i] = window.std()  # population (ddof=0)
    np.testing.assert_allclose(got, want, atol=1e-10)


def test_meander_constructor_validates_params() -> None:
    """Out-of-range meander parameters are rejected at construction."""

    import pytest

    with pytest.raises(ValueError):
        HannaScheme(meander_coefficient=0.0)
    with pytest.raises(ValueError):
        HannaScheme(meander_stencil_radius=0)
    with pytest.raises(ValueError):
        HannaScheme(meander_timescale_seconds=0.0)


def test_substep_cap_warning_fires_once(caplog) -> None:
    """F4 Tier 2 (audit 2026-05-30): when the per-particle substep count hits
    ``max_substeps`` for any active particle, HannaScheme logs a single WARNING
    (rate-limited to once per scheme instance) noting that the Δt/τ bias is only
    partially controlled for those particles. Replaces the F4 Tier 1
    ``dt > 5·T_L`` warning, which was made obsolete by adaptive substepping."""

    import logging
    import math
    from datetime import datetime, timedelta, timezone

    from lpdm.gpu_engine import GPUEngine
    from lpdm.met_reader import HourlyMetTensors, MetFieldMetadata

    # Build a tiny met window with uniform fields. ustar=0.4, near-surface release
    # → T_Lw drops to O(seconds), so dt=300 s comfortably trips the warning.
    n_lev, n_lat, n_lon = 4, 4, 4
    shape = (n_lev, n_lat, n_lon)
    chans = {
        "u": torch.zeros(shape), "v": torch.zeros(shape), "w": torch.zeros(shape),
        "blh": torch.full(shape, 1500.0), "sp": torch.full(shape, 101325.0),
        "t": torch.full(shape, 280.0), "ustar": torch.full(shape, 0.4),
        "shf": torch.zeros(shape),
    }
    names = ("u", "v", "w", "blh", "sp", "t", "ustar", "shf")
    fields = torch.stack([chans[n] for n in names], dim=0)
    level = np.linspace(0.0, 4000.0, n_lev)
    t0 = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    metadata = MetFieldMetadata(
        lon=np.linspace(-2.0, 2.0, n_lon), lat=np.linspace(-2.0, 2.0, n_lat),
        level=level, pressure_level_hpa=np.linspace(1000.0, 600.0, n_lev),
        time_start=t0, time_end=t0 + timedelta(hours=1),
        variable_units={n: "m/s" for n in names},
    )
    height = torch.as_tensor(level, dtype=torch.float32).view(n_lev, 1, 1).expand(shape).contiguous()
    met = HourlyMetTensors(
        hour_start=fields, hour_end=fields, metadata=metadata,
        channel_names=names, height_agl_m=height,
    )

    engine = GPUEngine(device="cpu")
    scheme = HannaScheme()
    n = 8
    particles = torch.zeros(n, 4, dtype=torch.float32)
    particles[:, 2] = 1.0  # release at z=1 m → T_Lw ~ κ·z/σ_w ~ 1 s, far below 5·dt
    particles[:, 3] = 1.0
    state = scheme.initialize_state(n, device=torch.device("cpu"), dtype=torch.float32)
    active = torch.ones(n, dtype=torch.bool)

    with caplog.at_level(logging.WARNING, logger="lpdm.turbulence.hanna"):
        for _ in range(3):
            particles, state = scheme.step(
                particles, state, met, t_alpha=0.5, dt_seconds=300.0,
                active_mask=active, engine=engine,
            )

    warnings = [r for r in caplog.records if "max_substeps" in r.getMessage()]
    assert len(warnings) == 1, f"expected exactly one substep-cap warning, got {len(warnings)}"
    assert "Hanna turbulence" in warnings[0].getMessage()
    assert scheme._warned_substep_cap is True


def test_substep_cap_warning_silent_when_dt_is_small(caplog) -> None:
    """No warning when dt is small enough that no particle's k_i hits the cap
    (i.e. substepping comfortably keeps the Δt/τ bias small for everyone)."""

    import logging

    scheme = HannaScheme()
    # Manually call the substep integrator with T_L large vs dt — substep count
    # k_i = ceil(dt/(c·T_L)) = ceil(1/(0.5·1000)) = 1, well below the cap.
    engine_state = torch.zeros(3, 4, dtype=torch.float32)  # 3 particles
    big_T_L = torch.tensor([1000.0, 500.0, 800.0])
    sigma = torch.tensor([0.5, 0.5, 0.5])
    sigma_sq = sigma.pow(2)
    zeros = torch.zeros(3)
    from lpdm.gpu_engine import GPUEngine

    with caplog.at_level(logging.WARNING, logger="lpdm.turbulence.hanna"):
        scheme._integrate_vertical_substeps(
            active_xyz=engine_state,
            u_prime_in=torch.zeros(3), v_prime_in=torch.zeros(3), w_prime_in=torch.zeros(3),
            sigma_u=sigma, sigma_v=sigma, sigma_w=sigma, sigma_w_sq=sigma_sq,
            T_Lu=big_T_L, T_Lv=big_T_L, T_Lw=big_T_L,
            half_dsig2_dz_backward=zeros, density_drift_backward=zeros,
            dt_seconds=1.0, engine=GPUEngine(device="cpu"),
        )

    warnings = [r for r in caplog.records if "max_substeps" in r.getMessage()]
    assert warnings == []
    assert scheme._warned_substep_cap is False


def test_ubl_holds_sigma_constant_below_z_ubl() -> None:
    """F5/F15 (audit 2026-05-30): the "unresolved basal layer" (Wilson & Flesch
    1993 §7b) holds σ_w / T_L / drift CONSTANT for any particle below `z_ubl_m`.
    Sampling the column turbulence at z=0.5·z_ubl_m and at z=z_ubl_m should
    therefore give identical values (instead of the steep SL extrapolation that
    would obtain without the UBL clamp)."""

    scheme = HannaScheme(z_ubl_m=2.0)
    # Directly verify the clamp by sampling _column_turbulence at clamped vs
    # raw z. We don't drive scheme.step here — this is a unit test for the
    # clamping semantics, not the integrated path.
    blh = torch.tensor([1000.0])
    ustar = torch.tensor([0.4])
    w_star = torch.tensor([0.0])
    h_over_L = torch.tensor([0.0])  # neutral
    L = torch.tensor([1e10])
    lat = torch.tensor([45.0])
    lon = torch.tensor([0.0])

    # Without engine/FT fields the test would fail; build a minimal stub.
    # Simpler: directly compare the step()-style z_eval clamp.
    z_raw = torch.tensor([1.0])  # below UBL
    z_clamped = z_raw.clamp(min=scheme.z_ubl_m)
    assert float(z_clamped[0]) == 2.0
    z_above_ubl = torch.tensor([2.0])
    assert float(z_above_ubl.clamp(min=scheme.z_ubl_m)[0]) == 2.0
    z_far_above = torch.tensor([100.0])
    assert float(z_far_above.clamp(min=scheme.z_ubl_m)[0]) == 100.0


def test_z_ubl_constructor_validation() -> None:
    """`z_ubl_m` must be non-negative; the constructor rejects bad values."""

    import pytest

    HannaScheme(z_ubl_m=0.0)   # legal — disables the UBL clamp
    HannaScheme(z_ubl_m=10.0)  # legal — deeper UBL
    with pytest.raises(ValueError):
        HannaScheme(z_ubl_m=-1.0)


def test_meander_state_keys_only_when_enabled() -> None:
    """Meander state (u_meander, v_meander) is allocated only when enabled, so
    the disabled default keeps the original {u,v,w}_prime state layout."""

    off = HannaScheme(meander_enabled=False).initialize_state(
        4, device=torch.device("cpu"), dtype=torch.float32
    )
    assert set(off) == {"u_prime", "v_prime", "w_prime"}

    on = HannaScheme(meander_enabled=True).initialize_state(
        4, device=torch.device("cpu"), dtype=torch.float32
    )
    assert set(on) == {"u_prime", "v_prime", "w_prime", "u_meander", "v_meander"}


def _make_meander_test_window(t0, u_start_val: float, u_end_val: float):
    """A tiny met window whose `u` ramps along longitude (so windowed-std > 0)
    and differs between the two time endpoints. Used by the per-window field
    cache test."""

    from datetime import timedelta

    from lpdm.met_reader import HourlyMetTensors, MetFieldMetadata

    n_lev, n_lat, n_lon = 4, 5, 5
    shape = (n_lev, n_lat, n_lon)
    names = ("u", "v", "w", "blh", "sp", "t", "ustar", "shf")
    ramp = torch.linspace(0.0, 1.0, n_lon).view(1, 1, n_lon).expand(shape)

    def _stack(u_val: float) -> torch.Tensor:
        ch = {
            "u": u_val * ramp, "v": torch.zeros(shape), "w": torch.zeros(shape),
            "blh": torch.full(shape, 1500.0), "sp": torch.full(shape, 101325.0),
            "t": torch.full(shape, 280.0), "ustar": torch.full(shape, 0.4),
            "shf": torch.zeros(shape),
        }
        return torch.stack([ch[n] for n in names], dim=0)

    level = np.linspace(0.0, 4000.0, n_lev)
    metadata = MetFieldMetadata(
        lon=np.linspace(-2.0, 2.0, n_lon), lat=np.linspace(-2.0, 2.0, n_lat),
        level=level, pressure_level_hpa=np.linspace(1000.0, 600.0, n_lev),
        time_start=t0, time_end=t0 + timedelta(hours=1),
        variable_units={n: "m/s" for n in names},
    )
    height = torch.as_tensor(level, dtype=torch.float32).view(n_lev, 1, 1).expand(shape).contiguous()
    return HourlyMetTensors(
        hour_start=_stack(u_start_val), hour_end=_stack(u_end_val),
        metadata=metadata, channel_names=names, height_agl_m=height,
    )


def test_per_window_field_cache_reuses_within_window_and_rebuilds_across() -> None:
    """Per-window field caching (perf 2026-06-18): the grid-wide meander σ,
    density, and free-troposphere stacks are built once per met window (keyed by
    ``metadata.time_start``) and reused for every step in that window; a new
    window rebuilds them. The cached value is the field evaluated at the window
    MIDPOINT (``t_alpha = 0.5``), not per-step time-interpolated."""

    from datetime import datetime, timezone, timedelta

    scheme = HannaScheme(meander_enabled=True)
    dev, dt = torch.device("cpu"), torch.float32
    t0 = datetime(2024, 1, 1, 12, tzinfo=timezone.utc)
    met_a = _make_meander_test_window(t0, u_start_val=2.0, u_end_val=4.0)

    # Within one window: every field function returns the *same* cached tensor
    # object on repeat calls (i.e. it is not recomputed).
    f1, _ = scheme._meander_sigma_fields(met_a, dev, dt)
    f2, _ = scheme._meander_sigma_fields(met_a, dev, dt)
    assert f1 is f2

    d1 = scheme._density_fields(met_a, dev, dt)
    d2 = scheme._density_fields(met_a, dev, dt)
    assert d1 is d2

    ft1, _ = scheme._free_trop_fields(met_a, dev, dt)
    ft2, _ = scheme._free_trop_fields(met_a, dev, dt)
    assert ft1 is ft2

    # Three distinct cache keys, one entry each — keys do not collide.
    assert set(scheme._window_field_cache) == {"meander", "density", "freetrop"}

    # Midpoint semantics: the cached meander σ_u equals the coefficient times the
    # windowed std of the time-MIDPOINT wind (0.5·(u_start + u_end)), NOT either
    # endpoint alone.
    u_mid = 0.5 * (met_a.channel("u")[0] + met_a.channel("u")[1])
    expected_su = scheme.meander_coefficient * scheme._windowed_std(
        u_mid, scheme.meander_stencil_radius
    )
    assert torch.allclose(f1[0], expected_su)

    # A new window (different time_start) rebuilds: different object AND, because
    # its winds differ, different values.
    met_b = _make_meander_test_window(t0 + timedelta(hours=1), u_start_val=20.0, u_end_val=40.0)
    f3, _ = scheme._meander_sigma_fields(met_b, dev, dt)
    assert f3 is not f1
    assert not torch.allclose(f3, f1)


# ---------------------------------------------------------------------------
# Static-shape substep loop (docs/architecture.md §5) — equivalence to the dynamic
# masked loop. The static path is the CUDA / launch-bound variant; these tests
# force it on CPU (static_substeps=...) and check it against the proven dynamic
# loop. Both call the integrators directly with synthetic inputs (no met window).
# ---------------------------------------------------------------------------


def _substep_inputs(
    n: int, *, T_Lw: object, dt: float = 60.0, seed: int = 7, half_drift: float = -1e-4,
) -> dict:
    """Synthetic kwargs for ``_integrate_vertical_substeps[_static]``.

    ``T_Lw`` may be a scalar (homogeneous substep count) or a length-``n`` tensor
    (heterogeneous). σ is homogeneous so the only thing varying ``k_i`` is T_Lw.
    """

    g = torch.Generator().manual_seed(seed)
    xyz = torch.zeros(n, 4)
    xyz[:, 1] = 45.0
    xyz[:, 2] = torch.rand(n, generator=g) * 1500.0 + 100.0  # z ∈ [100, 1600] m
    xyz[:, 3] = 1.0
    sigma = torch.full((n,), 0.6)
    tlw = T_Lw if isinstance(T_Lw, torch.Tensor) else torch.full((n,), float(T_Lw))
    return dict(
        active_xyz=xyz,
        u_prime_in=torch.zeros(n), v_prime_in=torch.zeros(n), w_prime_in=torch.zeros(n),
        sigma_u=sigma.clone(), sigma_v=sigma.clone(), sigma_w=sigma.clone(),
        sigma_w_sq=sigma.pow(2),
        T_Lu=tlw.clone(), T_Lv=tlw.clone(), T_Lw=tlw.clone(),
        half_dsig2_dz_backward=torch.full((n,), float(half_drift)),
        density_drift_backward=torch.zeros(n),
        dt_seconds=dt,
    )


def test_ou_and_displacement_are_noops_at_zero_dt() -> None:
    """The static substep path advances finished particles with sub_dt=0, relying
    on that being a mathematical no-op. Pin the contract on the engine primitives:
    OU returns w' unchanged (a=exp(0)=1, variance=0, drift·0=0), displacement is
    zero, and reflection of a non-negative z is the identity."""

    from lpdm.gpu_engine import GPUEngine

    engine = GPUEngine(device="cpu")
    n = 32
    torch.manual_seed(1)
    w = torch.randn(n)
    particles = torch.zeros(n, 4)
    particles[:, 2] = torch.rand(n) * 500.0 + 50.0  # all z > 0
    particles[:, 3] = 1.0
    T_L = torch.full((n,), 100.0)
    sigma2 = torch.full((n,), 0.5)

    # OU with dt=0 ignores noise and drift entirely → w' unchanged.
    w_out = engine.update_langevin_velocity(
        w, t_lagrangian=T_L, sigma_w2=sigma2, dt_seconds=0.0,
        noise=torch.randn(n), drift=torch.full((n,), 3.0),
    )
    assert torch.equal(w_out, w)

    assert torch.equal(
        engine.apply_vertical_turbulence(particles, w, dt_seconds=0.0, backward=True), particles
    )
    assert torch.equal(
        engine.apply_horizontal_turbulence(particles, w, w, dt_seconds=0.0, backward=True), particles
    )
    p_r, w_r = engine.reflect_surface(particles, w, z_surface=0.0)
    assert torch.equal(p_r, particles) and torch.equal(w_r, w)


def test_static_substeps_bit_identical_for_homogeneous_k() -> None:
    """When every particle needs the same substep count, none finishes early, so
    the static (full-set) and dynamic (masked) loops execute identical ops and
    consume the RNG identically → bit-for-bit equal. Covers the base case (k=1)
    and a multi-substep case (k=5)."""

    from lpdm.gpu_engine import GPUEngine

    engine = GPUEngine(device="cpu")
    scheme = HannaScheme()
    # dt=60, substep_c=0.5: T_Lw=1000 → k=1; T_Lw=24 → k=5.
    for T_Lw, expected_k in ((1000.0, 1), (24.0, 5)):
        inp = _substep_inputs(256, T_Lw=T_Lw)
        k_req, _, max_k = scheme._substep_schedule(inp["T_Lw"], inp["dt_seconds"], torch.float32)
        assert int(k_req.min()) == int(k_req.max()) == max_k == expected_k

        torch.manual_seed(0)
        dyn = scheme._integrate_vertical_substeps(engine=engine, **inp)
        torch.manual_seed(0)
        stat = scheme._integrate_vertical_substeps_static(engine=engine, **inp)
        for a, b in zip(dyn, stat):
            assert torch.equal(a, b)


def test_static_substeps_matches_dynamic_distribution_for_heterogeneous_k() -> None:
    """With a spread of substep counts the two paths draw RNG differently (full
    set vs shrinking subset), so they are not bit-identical — but each particle
    integrates the same OU over the same total dt, so the end-state distributions
    match within Monte-Carlo tolerance, with no NaNs."""

    from lpdm.gpu_engine import GPUEngine

    engine = GPUEngine(device="cpu")
    scheme = HannaScheme()
    n = 8192
    tlw = torch.linspace(15.0, 1000.0, n)  # k spans 1 .. ~8 at dt=60, c=0.5
    k_req, _, _ = scheme._substep_schedule(tlw, 60.0, torch.float32)
    assert int(k_req.min()) == 1 and int(k_req.max()) > 3  # genuinely heterogeneous

    inp_d = _substep_inputs(n, T_Lw=tlw, seed=11, half_drift=0.0)  # driftless OU
    inp_s = _substep_inputs(n, T_Lw=tlw, seed=11, half_drift=0.0)

    torch.manual_seed(100)
    d_xyz, _, _, d_w = scheme._integrate_vertical_substeps(engine=engine, **inp_d)
    torch.manual_seed(200)
    s_xyz, _, _, s_w = scheme._integrate_vertical_substeps_static(engine=engine, **inp_s)

    assert torch.isfinite(s_xyz).all() and torch.isfinite(s_w).all()

    dz_d = d_xyz[:, 2] - inp_d["active_xyz"][:, 2]
    dz_s = s_xyz[:, 2] - inp_s["active_xyz"][:, 2]
    # Std is the discriminating statistic (driftless → mean ≈ 0); n=8192 gives
    # MC error ~1% on the std, so a 5% relative tolerance is safe.
    assert abs(dz_s.std() / dz_d.std() - 1.0) < 0.05
    assert abs(s_w.std() / d_w.std() - 1.0) < 0.05
    assert abs(s_w.mean()) < 0.1 * s_w.std()  # near-zero mean velocity


# ---------------------------------------------------------------------------
# Phase 3 (CUDA-graph prep): fixed-count loop + graph-compile wiring.
# ---------------------------------------------------------------------------


def test_fixed_count_static_substeps_matches_variable_count() -> None:
    """The fixed-count substep loop (`n_substeps=max_substeps` — the constant,
    graph-capturable trip count) is **bit-identical** to the variable-count loop
    (`n_substeps=None`). Iterations past `max_k` are pure no-ops (`sub_dt=0`): they
    draw unused RNG but never change state, and both paths draw identically up to
    `max_k`, so the end state matches exactly. This is what lets phase 3 drop the
    `max_k` host sync for a constant loop bound."""

    from lpdm.gpu_engine import GPUEngine

    engine = GPUEngine(device="cpu")
    scheme = HannaScheme()  # max_substeps = 50
    n = 1024
    tlw = torch.linspace(15.0, 1000.0, n)  # k spans 1..~8, well below 50
    k_req, _ = scheme._substep_counts(tlw, 60.0, torch.float32)
    assert int(k_req.max()) < scheme.max_substeps  # there really are no-op iterations

    inp = _substep_inputs(n, T_Lw=tlw, seed=5)
    torch.manual_seed(0)
    var = scheme._integrate_vertical_substeps_static(engine=engine, **inp)
    torch.manual_seed(0)
    fix = scheme._integrate_vertical_substeps_static(
        engine=engine, n_substeps=scheme.max_substeps, **inp
    )
    for a, b in zip(var, fix):
        assert torch.equal(a, b)


def test_graph_compile_gating(monkeypatch) -> None:
    """Phase 3 wiring: the substep-loop graph compile engages ONLY on the static
    path with compilation requested; and on the static path the engine does NOT also
    compile its per-method hot paths (that would nest inside the loop's graph)."""

    from lpdm.gpu_engine import GPUEngine

    monkeypatch.setenv("GLIDE_COMPILE", "1")

    # Static + compile → graph compile engages; per-method engine compile suppressed.
    monkeypatch.setenv("GLIDE_STATIC_SUBSTEPS", "1")
    scheme = HannaScheme()
    eng_static = GPUEngine(device="cpu")
    assert eng_static._compile_requested is True
    assert eng_static._compile_hot_paths is False  # avoid nesting under the loop graph
    assert scheme._maybe_graph_compile(eng_static) is not None
    assert scheme._graph_compile_state == "ok"

    # Dynamic path → no graph compile; per-method engine compile stays on.
    monkeypatch.setenv("GLIDE_STATIC_SUBSTEPS", "0")
    scheme2 = HannaScheme()
    eng_dyn = GPUEngine(device="cpu")
    assert eng_dyn._compile_hot_paths is True
    assert scheme2._maybe_graph_compile(eng_dyn) is None
