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
    air_density,
    convective_velocity,
    coriolis_parameter,
    in_bl_sigma_TL,
    obukhov_length,
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
    """The above-BL constants exposed by the module match the spec."""

    assert ABOVE_BL_SIGMA_M_S == 0.1
    assert ABOVE_BL_T_L_S == 100.0
