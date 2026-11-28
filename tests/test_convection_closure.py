"""Task CONV (prototype): rate-based convective closure applied exactly in time.

The matrix exponential of the rate generator (`generator_exponential`) must be a
non-negative stochastic matrix equal to the true exponential; the backward
transition must keep the layer-mass vector stationary and be the adjoint of the
forward one; a 3-hour interval must equal three 1-hour intervals (cadence
independence); and, on a column refined from 40 to 79 levels, the mass a rate
closure moves per hour must barely change, while the CFL-scaled single transfer
halves (grid dependence of the default). The defaults stay as before CONV.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from lpdm.convection import emanuel as em
from lpdm.convection.emanuel import EmanuelReducedConvection


def _columns(n_lev: int = 40, n_col: int = 4) -> tuple[torch.Tensor, ...]:
    """Pressure profiles [C, Z] (Pa, ascending height) and cloud indices."""

    p = torch.linspace(101000.0, 15000.0, n_lev, dtype=torch.float64).expand(n_col, n_lev).clone()
    lcl = torch.tensor([8, 10, 12, 15][:n_col])
    lnb = torch.tensor([30, 25, 35, 20][:n_col])
    return p, lcl, lnb


def _random_generators(n_col: int, n_lev: int, scale: float, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    off = torch.rand(n_col, n_lev, n_lev, generator=g, dtype=torch.float64) * scale
    off = off * (torch.rand(n_col, n_lev, n_lev, generator=g, dtype=torch.float64) < 0.2)
    off = off * (1.0 - torch.eye(n_lev, dtype=torch.float64))
    return off - torch.diag_embed(off.sum(dim=2))


@pytest.mark.parametrize("scale", [1e-6, 1e-3, 5e-2])
def test_generator_exponential_matches_matrix_exp_and_is_stochastic(scale: float) -> None:
    """Up to Courant numbers of several hundred (scale·Z·dt), the uniformised
    scaling-and-squaring exponential equals torch.linalg.matrix_exp to 1e-12 and
    is exactly non-negative with rows summing to one."""

    q = _random_generators(3, 30, scale, seed=1)
    dt = 3600.0
    got = em.generator_exponential(q, dt)
    ref = torch.linalg.matrix_exp(q * dt)
    assert torch.allclose(got, ref, atol=1e-11, rtol=0.0)
    assert float(got.min()) >= 0.0
    assert torch.allclose(got.sum(dim=2), torch.ones(3, 30, dtype=torch.float64), atol=1e-13)


def test_zero_generator_gives_identity() -> None:
    q = torch.zeros(2, 5, 5, dtype=torch.float64)
    assert torch.equal(em.generator_exponential(q, 3600.0), torch.eye(5).expand(2, 5, 5).double())


@pytest.mark.parametrize("m_b", [0.01, 0.18])
def test_exponential_backward_is_stationary_and_adjoint(m_b: float) -> None:
    """mᵀP = mᵀ (well-mixed criterion) in both directions, and the backward
    transition is D⁻¹ P_fwdᵀ D, for a realistic and for the nominal (CFL-
    violating) cloud-base flux."""

    p, lcl, lnb = _columns()
    rate = em.compute_mass_flux_matrices(
        p, lcl, lnb, torch.full((4,), m_b, dtype=torch.float64), cfl_max_outflux_frac=None
    )
    m = em._layer_masses_per_area(p)
    pb = em.backward_transition_matrices_exponential(rate, m, 3600.0)
    pf = em.generator_exponential(em.rate_generator(rate, m, backward=False), 3600.0)
    for prob in (pb, pf):
        stat = (m.unsqueeze(1) @ prob).squeeze(1)
        assert torch.allclose(stat, m, rtol=1e-12, atol=0.0)
        assert float(prob.min()) >= 0.0
    d = torch.diag_embed(m)
    d_inv = torch.diag_embed(1.0 / m)
    assert torch.allclose(pb, d_inv @ pf.transpose(1, 2) @ d, atol=1e-12)


def test_exponential_is_cadence_independent() -> None:
    """One 3-hour interval equals three 1-hour intervals (constant meteorology);
    the CFL-scaled transfer does not have this property (it moves the same mass
    per event whatever the interval once the scale binds)."""

    p, lcl, lnb = _columns()
    mb = torch.full((4,), 0.02, dtype=torch.float64)
    rate = em.compute_mass_flux_matrices(p, lcl, lnb, mb, cfl_max_outflux_frac=None)
    m = em._layer_masses_per_area(p)
    p1 = em.backward_transition_matrices_exponential(rate, m, 3600.0)
    p3 = em.backward_transition_matrices_exponential(rate, m, 10800.0)
    assert torch.allclose(p1 @ p1 @ p1, p3, atol=1e-12)

    # Legacy: once the CFL scale binds, a 3-h event is the same transfer as a 1-h one.
    big = torch.full((4,), 0.18, dtype=torch.float64)
    f1 = em.compute_mass_flux_matrices(p, lcl, lnb, big * 3600.0)
    f3 = em.compute_mass_flux_matrices(p, lcl, lnb, big * 10800.0)
    assert torch.allclose(f1, f3)


def _sub_cloud_to_cloud_mass(prob_fwd: torch.Tensor, m: torch.Tensor, lcl: torch.Tensor):
    k = torch.arange(m.shape[1])
    src = (k.unsqueeze(0) < lcl.unsqueeze(1)).double()
    moved = torch.einsum("ci,cij->cj", src * m, prob_fwd.double())
    return (moved * (1.0 - src)).sum(dim=1)


def _refined(p: torch.Tensor) -> torch.Tensor:
    """Insert the log-pressure midpoint between every pair of levels."""

    lp = torch.log(p)
    mid = 0.5 * (lp[:, :-1] + lp[:, 1:])
    out = torch.empty(p.shape[0], 2 * p.shape[1] - 1, dtype=p.dtype)
    out[:, 0::2] = lp
    out[:, 1::2] = mid
    return torch.exp(out)


def test_rate_closure_is_grid_independent_and_cfl_scaled_is_not() -> None:
    """The same column on 40 and 79 levels (every layer split in two), the same
    cloud base and top pressure, and the same cloud-base flux: the exponential
    application moves nearly the same sub-cloud air into the cloud per hour
    (within 10%: only the adjacent-layer subsidence and the index-linear
    detrainment see the grid), whereas the CFL-scaled transfer is limited by
    the mass of the layer below cloud base and roughly halves."""

    p40, lcl40, lnb40 = _columns()
    p79 = _refined(p40)
    lcl79, lnb79 = 2 * lcl40, 2 * lnb40
    mb = torch.full((4,), 0.18, dtype=torch.float64)
    out = {}
    for name, p, lcl, lnb in (("40", p40, lcl40, lnb40), ("79", p79, lcl79, lnb79)):
        m = em._layer_masses_per_area(p)
        rate = em.compute_mass_flux_matrices(p, lcl, lnb, mb * 0.1, cfl_max_outflux_frac=None)
        pf = em.generator_exponential(em.rate_generator(rate, m, backward=False), 3600.0)
        fmass = em.compute_mass_flux_matrices(p, lcl, lnb, mb * 3600.0)
        legacy = (fmass + torch.diag_embed(m - fmass.sum(dim=2))) / m.unsqueeze(2)
        out[name] = (
            _sub_cloud_to_cloud_mass(pf, m, lcl),
            _sub_cloud_to_cloud_mass(legacy, m, lcl),
        )
    exp_ratio = out["79"][0] / out["40"][0]
    cfl_ratio = out["79"][1] / out["40"][1]
    assert torch.all((exp_ratio - 1.0).abs() < 0.1), exp_ratio
    assert torch.all(cfl_ratio < 0.7), cfl_ratio


def test_buoyancy_closure_is_flexpart_steady_state() -> None:
    """M_b = 0.1·ALPHA/DAMPS·DTMIN with FLEXPART's ALPHA = 0.025, DAMPS = 0.3."""

    assert pytest.approx(0.1 * 0.025 / 0.3) == em.BUOYANCY_MASS_FLUX_PER_K
    # DTMIN of a parcel exactly neutral everywhere is DTMAX.
    c, z = 2, 10
    p = torch.linspace(100000.0, 50000.0, z, dtype=torch.float64).expand(c, z)
    tv = torch.full((c, z), 290.0, dtype=torch.float64)
    dtmin = em.cloud_base_buoyancy_excess_columns(
        tv, tv, p, torch.tensor([92000.0, 80000.0], dtype=torch.float64), torch.tensor([2, 4])
    )
    assert torch.allclose(dtmin, torch.full((c,), 0.9, dtype=torch.float64))
    # +1 K everywhere adds 1 K at the LCL and 1 K in the sub-cloud mean.
    dtmin = em.cloud_base_buoyancy_excess_columns(
        tv + 1.0, tv, p, torch.tensor([92000.0, 80000.0], dtype=torch.float64), torch.tensor([2, 4])
    )
    assert torch.allclose(dtmin, torch.full((c,), 2.9, dtype=torch.float64))


def test_trigger_at_height_above_lcl_interpolates() -> None:
    """ΔT_v at a fixed height above the LCL: linear in height between levels."""

    h = torch.tensor([[0.0, 100.0, 200.0, 300.0, 400.0]], dtype=torch.float64)
    p = torch.tensor([[1000.0, 990.0, 980.0, 970.0, 960.0]], dtype=torch.float64) * 100.0
    dtv = torch.tensor([[0.0, 0.0, 1.0, 2.0, 3.0]], dtype=torch.float64)
    # LCL exactly at level 1 (p_lcl = p[1]); i_lcl is the first level above it.
    out = em.trigger_dtv_at_height_above_lcl(dtv, h, p, p[:, 1], torch.tensor([2]), 150.0)
    assert float(out[0]) == pytest.approx(1.5)
    # Target above the column top: no trigger.
    out = em.trigger_dtv_at_height_above_lcl(dtv, h, p, p[:, 1], torch.tensor([2]), 1000.0)
    assert float(out[0]) == -1.0


def test_defaults_are_unchanged_and_options_validate() -> None:
    s = EmanuelReducedConvection()
    assert (s.closure, s.application, s.trigger_height_above_lcl_m) == (
        "cape_velocity",
        "cfl_scaled",
        None,
    )
    for bad in (
        {"closure": "x"},
        {"application": "x"},
        {"n_substeps": 0},
        {"buoyancy_mass_flux_per_k": 0.0},
        {"updraft_w_max_m_s": 0.0},
        {"trigger_height_above_lcl_m": 0.0},
    ):
        with pytest.raises(ValueError):
            EmanuelReducedConvection(**bad)


def test_default_transition_is_the_pre_conv_matrix() -> None:
    """`transition_matrices` at the defaults reproduces exactly the construction
    `maybe_convect` used before CONV (one CFL-scaled transfer, backward)."""

    from tests.test_convection import _build_convective_met_window  # noqa: PLC0415

    met = _build_convective_met_window()
    scheme = EmanuelReducedConvection(closure_c=0.1, trigger_dtv_k=0.1)
    cols = scheme.column_state(met, t_alpha=0.5, device=torch.device("cpu"), dtype=torch.float32)
    idx = cols.fires.nonzero().squeeze(1)
    assert idx.numel() > 0
    fmass = em.compute_mass_flux_matrices(
        cols.pressure_pa[idx],
        cols.i_lcl[idx],
        cols.i_lnb[idx],
        (cols.m_b[idx] * cols.interval_s).float(),
    )
    lm = em._layer_masses_per_area(cols.pressure_pa[idx]).clamp(min=1e-6)
    ref = em.backward_transition_matrices(fmass, lm)
    assert torch.equal(scheme.transition_matrices(cols, idx), ref)


@pytest.mark.parametrize("application", ["exponential", "substeps"])
def test_options_preserve_well_mixed_in_a_real_scheme_column(application: str) -> None:
    from tests.test_convection import _build_convective_met_window  # noqa: PLC0415

    met = _build_convective_met_window()
    scheme = EmanuelReducedConvection(trigger_dtv_k=0.1, closure_c=0.1, application=application)
    cols = scheme.column_state(met, t_alpha=0.5, device=torch.device("cpu"), dtype=torch.float32)
    idx = cols.fires.nonzero().squeeze(1)
    assert idx.numel() > 0 and bool((cols.m_b[idx] > 0).all())
    prob = scheme.transition_matrices(cols, idx).double()
    m = em._layer_masses_per_area(cols.pressure_pa[idx].double()).clamp(min=1e-6)
    stat = (m.unsqueeze(1) @ prob).squeeze(1)
    assert np.allclose(stat.numpy(), m.numpy(), rtol=1e-5)
    assert float(prob.min()) >= 0.0
