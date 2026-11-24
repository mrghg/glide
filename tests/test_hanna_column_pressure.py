"""Hanna density and free-troposphere θ from each column's own pressure.

Decision 0025: `HannaScheme._density_fields` (the Stohl-Thomson density drift)
and `_free_trop_fields` (θ for N²) used the met domain's area-mean pressure per
level (`metadata.pressure_level_hpa`). They now use each column's own pressure
(`HourlyMetTensors.pressure_hpa`, falling back to the level pressure broadcast
when the reader supplies none). These tests use a synthetic terrain-following
box with flat and elevated columns in a constant-lapse-rate atmosphere, whose
hydrostatic pressure and density are known in closed form.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import torch

from lpdm.met_reader import HourlyMetTensors, MetFieldMetadata
from lpdm.turbulence import HannaScheme
from lpdm.turbulence.hanna import GRAVITY_M_S2, R_DRY_AIR_J_KG_K

G = GRAVITY_M_S2
R = R_DRY_AIR_J_KG_K
GAMMA = 0.0065  # K/m lapse rate
LEVELS_AGL = np.array([0.0, 40.0, 250.0, 500.0, 750.0, 1000.0, 1500.0, 2000.0, 3000.0])
NAMES = ("u", "v", "w", "blh", "sp", "t", "ustar", "shf")


def _column_profiles(ps_pa: np.ndarray, t_sfc: np.ndarray):
    """Closed-form hydrostatic profiles for a constant lapse rate, per column.

    T(z) = T_s - Γz;  p(z) = p_s (T(z)/T_s)^(g/(R Γ));  ρ = p/(R T);
    ∂ ln ρ/∂z = -g/(R T) + Γ/T.  Shapes `[Z, Y, X]`, float64.
    """

    z = LEVELS_AGL[:, None, None]
    t = t_sfc[None] - GAMMA * z
    p = ps_pa[None] * (t / t_sfc[None]) ** (G / (R * GAMMA))
    rho = p / (R * t)
    dlnrho_dz = -G / (R * t) + GAMMA / t
    return t, p, rho, dlnrho_dz


def _window(ps_pa, t_sfc, *, with_column_pressure=True, dtype=torch.float64, p_end_scale=1.0):
    """A terrain-following window: columns differ in surface pressure (terrain
    height) and surface temperature; the grid levels are AGL heights."""

    t, p, _, _ = _column_profiles(ps_pa, t_sfc)
    shape = t.shape
    chans = {
        "u": np.full(shape, 5.0),
        "v": np.zeros(shape),
        "w": np.zeros(shape),
        "blh": np.full(shape, 800.0),
        "sp": np.broadcast_to(ps_pa, shape),
        "t": t,
        "ustar": np.full(shape, 0.3),
        "shf": np.zeros(shape),
    }
    fields = torch.as_tensor(np.stack([chans[n] for n in NAMES]), dtype=dtype)
    t0 = datetime(2024, 1, 1, 12, tzinfo=UTC)
    p_hpa = p / 100.0
    metadata = MetFieldMetadata(
        lon=np.linspace(0.0, 1.0, shape[2]),
        lat=np.linspace(45.0, 46.0, shape[1]),
        level=LEVELS_AGL,
        pressure_level_hpa=p_hpa.mean(axis=(1, 2)),  # the old area mean
        time_start=t0,
        time_end=t0 + timedelta(hours=1),
        variable_units=dict.fromkeys(NAMES, "1"),
    )
    height = torch.as_tensor(np.broadcast_to(LEVELS_AGL[:, None, None], shape).copy(), dtype=dtype)
    pressure = None
    if with_column_pressure:
        pressure = (
            torch.as_tensor(p_hpa, dtype=dtype),
            torch.as_tensor(p_hpa * p_end_scale, dtype=dtype),
        )
    return HourlyMetTensors(
        hour_start=fields,
        hour_end=fields,
        metadata=metadata,
        channel_names=NAMES,
        height_agl_m=height,
        pressure_hpa=pressure,
    )


# Two rows, three columns: sea level, 1,500 m and 3,000 m terrain (approximate
# surface pressures), with different surface temperatures.
PS_PA = np.array([[101325.0, 84500.0, 70100.0], [101000.0, 85000.0, 101200.0]])
T_SFC = np.array([[283.0, 275.0, 265.0], [281.0, 274.0, 288.0]])
DEV = torch.device("cpu")


def _density(met, dtype=torch.float64):
    return HannaScheme()._density_fields(met, DEV, dtype)


def test_density_matches_each_columns_hydrostatic_values() -> None:
    met = _window(PS_PA, T_SFC)
    rho_exp = _column_profiles(PS_PA, T_SFC)[2]
    dlnrho_exp = _column_profiles(PS_PA, T_SFC)[3]
    rho, drho_dz = _density(met).numpy()

    np.testing.assert_allclose(rho, rho_exp, rtol=1e-12)
    # ∂ρ/∂z is the central difference of the column's own ρ on its own heights
    # (one-sided at the ends) ...
    z = LEVELS_AGL[:, None, None]
    fd = np.empty_like(rho_exp)
    fd[1:-1] = (rho_exp[2:] - rho_exp[:-2]) / (z[2:] - z[:-2])
    fd[0] = (rho_exp[1] - rho_exp[0]) / (z[1] - z[0])
    fd[-1] = (rho_exp[-1] - rho_exp[-2]) / (z[-1] - z[-2])
    np.testing.assert_allclose(drho_dz, fd, rtol=1e-10)
    # ... so ∂ ln ρ/∂z matches the analytic -g/(R T) + Γ/T where the stencil is
    # centred (500 m and 750 m, spacing 250 m): truncation error only.
    for k in (3, 4):
        np.testing.assert_allclose(drho_dz[k] / rho[k], dlnrho_exp[k], rtol=1e-4)

    # The old area-mean pressure was far off over the elevated columns: the
    # 3,000 m column's density was too high by more than 20%.
    t = _column_profiles(PS_PA, T_SFC)[0]
    rho_old = met.metadata.pressure_level_hpa[:, None, None] * 100.0 / (R * t)
    assert (rho_old[:, 0, 2] / rho_exp[:, 0, 2] - 1.0).min() > 0.2


def test_density_does_not_depend_on_other_columns() -> None:
    base = _density(_window(PS_PA, T_SFC))
    ps2, t2 = PS_PA.copy(), T_SFC.copy()
    ps2[0, 1:] = 60000.0  # raise the other columns of row 0 to ~4 km
    ps2[1, :] = 80000.0
    t2[1, :] = 260.0
    other = _density(_window(ps2, t2))
    # Column (0, 0) is unchanged: bit-identical.
    assert torch.equal(base[:, :, 0, 0], other[:, :, 0, 0])


def test_density_uses_window_midpoint_pressure() -> None:
    met = _window(PS_PA, T_SFC, p_end_scale=1.02)
    t, p, _, _ = _column_profiles(PS_PA, T_SFC)
    rho = _density(met)[0].numpy()
    np.testing.assert_allclose(rho, 1.01 * p / (R * t), rtol=1e-12)


def test_horizontally_uniform_met_is_bit_identical_to_area_mean() -> None:
    """Horizontally uniform columns: per-column pressure, the level-pressure
    fallback (no `pressure_hpa`) and the old formula all agree bit for bit in
    float32, the production dtype."""

    ps = np.full((2, 3), 98000.0)
    ts = np.full((2, 3), 280.0)
    f32 = torch.float32
    with_p = _window(ps, ts, dtype=f32)
    without_p = _window(ps, ts, dtype=f32, with_column_pressure=False)
    d_with = _density(with_p, f32)
    d_without = _density(without_p, f32)
    assert torch.equal(d_with, d_without)

    # The pre-0025 formula, verbatim.
    p_old = torch.as_tensor(np.array(with_p.metadata.pressure_level_hpa), dtype=f32).view(-1, 1, 1)
    p_old = p_old * 100.0
    t_mid = 0.5 * (with_p.channel("t")[0] + with_p.channel("t")[1])
    rho_old = p_old / (R * t_mid.clamp(min=1.0))
    assert torch.equal(d_with[0], rho_old)

    ft_with = HannaScheme()._free_trop_fields(with_p, DEV, f32)[0]
    ft_without = HannaScheme()._free_trop_fields(without_p, DEV, f32)[0]
    assert torch.equal(ft_with, ft_without)


def test_free_troposphere_theta_from_own_column() -> None:
    """θ (and so N², Ri, K_z, σ_w, T_L) of a column depends only on that column."""

    base = HannaScheme()._free_trop_fields(_window(PS_PA, T_SFC), DEV, torch.float64)[0]
    ps2 = PS_PA.copy()
    ps2[0, 1:] = 60000.0
    other = HannaScheme()._free_trop_fields(_window(ps2, T_SFC), DEV, torch.float64)[0]
    assert torch.equal(base[:, :, 0, 0], other[:, :, 0, 0])


def test_free_troposphere_theta_uses_column_pressure(monkeypatch) -> None:
    """The pressure handed to θ is each column's own (window-midpoint) pressure."""

    import lpdm.turbulence.hanna as hanna

    seen = {}
    real = hanna.potential_temperature

    def spy(t_kelvin, pressure_pa):
        seen["p"] = pressure_pa
        return real(t_kelvin, pressure_pa)

    monkeypatch.setattr(hanna, "potential_temperature", spy)
    HannaScheme()._free_trop_fields(_window(PS_PA, T_SFC, p_end_scale=1.02), DEV, torch.float64)
    p = _column_profiles(PS_PA, T_SFC)[1]
    np.testing.assert_allclose(seen["p"].numpy(), 1.01 * p, rtol=1e-12)
