"""Decision 6 study: the `trigger` option of the reduced Emanuel scheme.

`glide` (the default) must reproduce the trigger before the option existed bit
for bit; `flexpart_cloud_base` applies FLEXPART 11's tests without memory
(`flexpart_cloud_base_trigger_columns`): the parcel not DTMAX = 0.9 K colder than
the environment at the first level above the LCL, DTMIN > 0, and FLEXPART's
other skip conditions; no CAPE floor.
"""

from __future__ import annotations

import pytest
import torch

from lpdm.convection import emanuel as em
from lpdm.convection.emanuel import EmanuelReducedConvection
from tests.test_convection_closure import _convective_column, _window

CPU = torch.device("cpu")


def _state(scheme: EmanuelReducedConvection, window, dtype=torch.float32):
    return scheme.column_state(window, t_alpha=0.0, device=CPU, dtype=dtype)


def _pre_option_fires(scheme: EmanuelReducedConvection, cols) -> torch.Tensor:
    """The trigger expression before the option existed (branch agent/CONV-closure, 0ee9296)."""

    return (
        (cols.i_lcl >= 0)
        & (cols.i_lnb > cols.i_lcl)
        & (cols.cape.double() >= scheme.min_cape_j_kg)
        & (cols.dtv_lcl_plus_1.double() >= scheme.trigger_dtv_k)
        & (cols.cloud_depth_m.double() >= scheme.min_cloud_depth_m)
    )


@pytest.mark.parametrize("n_lev", [12, 23, 40, 80])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_glide_trigger_is_the_default_and_bit_identical(n_lev: int, dtype: torch.dtype) -> None:
    window = _window(n_lev)
    default = EmanuelReducedConvection()
    explicit = EmanuelReducedConvection(trigger="glide")
    assert default.trigger == "glide"
    a, b = _state(default, window, dtype), _state(explicit, window, dtype)
    for name in a._fields:
        va, vb = getattr(a, name), getattr(b, name)
        if isinstance(va, torch.Tensor):
            assert torch.equal(va, vb), name
    assert torch.equal(a.fires, _pre_option_fires(default, a))
    m_b = default.buoyancy_mass_flux_per_k * a.dtmin_k.double().clamp(min=0.0)
    assert torch.equal(a.m_b, torch.where(a.fires, m_b, torch.zeros_like(m_b)))


def test_config_default_and_validation() -> None:
    from lpdm.config import EmanuelConvectionConfig  # noqa: PLC0415

    assert EmanuelConvectionConfig().trigger == "glide"
    assert EmanuelConvectionConfig(trigger="flexpart_cloud_base").trigger == "flexpart_cloud_base"
    for bad in (
        {"trigger": "x"},
        {"trigger": "flexpart_cloud_base", "closure": "cape_velocity"},
        {"trigger": "flexpart_cloud_base", "trigger_height_above_lcl_m": 200.0},
    ):
        with pytest.raises(ValueError):
            EmanuelConvectionConfig(**bad)
        with pytest.raises(ValueError):
            EmanuelReducedConvection(**bad)


def _profiles(n_col: int, n_lev: int = 12):
    """Simple columns (surface first): T, q, heights, pressures."""

    z = torch.linspace(0.0, 11000.0, n_lev, dtype=torch.float64)
    t = (300.0 - 6.5e-3 * z).expand(n_col, n_lev).clone()
    q = (0.012 * torch.exp(-z / 2500.0)).expand(n_col, n_lev).clone()
    h = z.expand(n_col, n_lev).clone()
    return t, q, h


def test_flexpart_trigger_tests_one_by_one() -> None:
    """Each FLEXPART skip condition switches off exactly its column."""

    n_col, n_lev = 7, 12
    t, q, h = _profiles(n_col, n_lev)
    p_lcl = torch.full((n_col,), 90000.0, dtype=torch.float64)
    i_lcl = torch.full((n_col,), 2, dtype=torch.long)
    dtv = torch.full((n_col, n_lev), 0.2, dtype=torch.float64)
    dtmin = torch.full((n_col,), 1.5, dtype=torch.float64)
    # 0: passes. 1: parcel 0.9 K colder at ICB. 2: just warmer than -0.9 K, passes.
    dtv[1, 2] = -0.9
    dtv[2, 2] = -0.89
    # 3: DTMIN <= 0. 4: parcel level colder than 250 K. 5: PLCL < 200 hPa.
    dtmin[3] = 0.0
    t[4, 0] = 249.0
    p_lcl[5] = 19000.0
    # 6: moist static energy increasing monotonically (no minimum below the top).
    q[6] = 0.0
    t[6] = 300.0
    ok, dtv_icb = em.flexpart_cloud_base_trigger_columns(t, q, h, p_lcl, i_lcl, dtv, dtmin)
    assert ok.tolist() == [True, False, True, False, False, False, False]
    assert torch.equal(dtv_icb, dtv[:, 2])
    # ICB >= NL - 1 (GLIDE's i_lcl >= Z - 3) and no LCL.
    i_lcl[0], i_lcl[2] = n_lev - 3, -1
    ok, _ = em.flexpart_cloud_base_trigger_columns(t, q, h, p_lcl, i_lcl, dtv, dtmin)
    assert not bool(ok[0]) and not bool(ok[2])


def test_flexpart_trigger_on_synthetic_soundings() -> None:
    """On the convective sounding both triggers fire on 23 levels with the same
    M_b; on 40 levels ΔT_v at LCL+1 is 0.887 K and only FLEXPART's test fires
    (the grid dependence of GLIDE's). A parcel 0.3 K warmer than the environment
    at ICB in a stable layer fires FLEXPART's test without a CAPE floor, but its
    first-LNB cloud is one level deep, so only without the depth check."""

    glide = EmanuelReducedConvection()
    fp = EmanuelReducedConvection(trigger="flexpart_cloud_base")
    a, b = _state(glide, _window(23)), _state(fp, _window(23))
    assert bool(a.fires.all()) and bool(b.fires.all())
    assert torch.equal(a.m_b, b.m_b)
    a, b = _state(glide, _window(40)), _state(fp, _window(40))
    assert not bool(a.fires.any()) and bool(b.fires.all())
    assert float(a.dtv_lcl_plus_1[0]) == pytest.approx(0.887, abs=0.01)

    t, q_, p = _convective_column(40)
    i_lcl = int(b.i_lcl[0])
    dtv = em.compute_lcl_lnb_cape_columns(
        t.unsqueeze(0), q_.unsqueeze(0), p.unsqueeze(0), return_extra=True
    )[4]["dtv"][0]
    t2 = t.clone()
    t2[i_lcl + 1 :] = t[i_lcl + 1 :] + dtv[i_lcl + 1 :].clamp(min=0.0) + 0.5
    t2[i_lcl] = t[i_lcl] + float(dtv[i_lcl]) - 0.3  # parcel 0.3 K warmer at ICB
    from tests.test_convection import _build_convective_met_window  # noqa: PLC0415

    w2 = _build_convective_met_window(column=(t2, q_, p))
    a2, b2 = _state(glide, w2, torch.float64), _state(fp, w2, torch.float64)
    fp0 = EmanuelReducedConvection(trigger="flexpart_cloud_base", min_cloud_depth_m=0.0)
    c2 = _state(fp0, w2, torch.float64)
    assert not bool(a2.fires.any()) and not bool(b2.fires.any())
    assert bool((c2.dtmin_k > 0).all()) and bool(c2.fires.all())
    assert bool((c2.i_lnb == c2.i_lcl + 1).all()) and bool((c2.cloud_depth_m < 500.0).all())
    assert float(c2.dtv_lcl_plus_1[0]) == pytest.approx(0.3, abs=0.05)
    assert bool((c2.cape < 50.0).all())
