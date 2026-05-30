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


def test_dt_too_large_warning_fires_once(caplog) -> None:
    """F4 Tier 1: HannaScheme.step logs a single WARNING when min(T_L) < 5·Δt
    (Stohl & Thomson 1999 use c=0.05; Wilson & Flesch 1993 App. derive the linear
    Δt/T_L bias). The warning is rate-limited to once per scheme instance so a
    multi-month run doesn't drown its log in repeats."""

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

    warnings = [r for r in caplog.records if "dt" in r.getMessage() and "T_L" in r.getMessage()]
    assert len(warnings) == 1, f"expected exactly one dt-vs-T_L warning, got {len(warnings)}"
    assert "Hanna turbulence" in warnings[0].getMessage()


def test_dt_too_large_warning_silent_when_dt_is_small() -> None:
    """No warning when dt is comfortably smaller than min(T_L) (i.e. the
    integration is already in the recommended regime)."""

    scheme = HannaScheme()
    # Large T_L (1000 s) vs tiny dt (1 s) → no warning.
    T_L = torch.tensor([1000.0, 500.0, 800.0])
    scheme._warn_if_dt_too_large(T_L, T_L, T_L, dt_seconds=1.0)
    assert scheme._warned_dt is False


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
