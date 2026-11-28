"""Decision 0033: the buoyancy closure applied exactly in time (the defaults).

The matrix exponential of the rate generator (`generator_exponential`) must be a
non-negative stochastic matrix equal to the true exponential; the backward
transition must keep the layer-mass vector stationary, be the adjoint of the
forward one and agree with `backward_transition_matrices` (the Euler step) in
the small-step limit; a 3-hour interval must equal three 1-hour intervals
(cadence independence); and the mass the scheme moves per hour must barely
change when the vertical grid is refined, while the CFL-scaled single transfer
of the old closure halves. Selecting the old options reproduces the old matrix.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from lpdm.convection import emanuel as em
from lpdm.convection.emanuel import EmanuelReducedConvection

LEGACY = {"closure": "cape_velocity", "application": "cfl_scaled", "trigger": "glide"}
CPU = torch.device("cpu")


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


def _convective_column(n_lev: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """A convective sounding on a stretched AGL grid of ``n_lev`` levels (surface
    first): dry-adiabatic, well-mixed boundary layer to 1 km, 6 K/km above, and a
    slightly superadiabatic surface layer. Its DTMIN is about the same on any
    grid, while ΔT_v at the level LCL+1 depends on the grid."""

    z = np.concatenate([[0.0], np.geomspace(10.0, 15000.0, n_lev - 1)])
    t = np.where(z < 1000.0, 301.0 - 9.8e-3 * z, 301.0 - 9.8 - 6.0e-3 * (z - 1000.0))
    t = np.maximum(t, 210.0) + 1.5 * np.exp(-z / 50.0)
    p = 100000.0 * np.exp(-z / (287.05 * 270.0 / 9.80665))
    q = np.where(z < 1000.0, 0.0145, 0.0145 * np.exp(-(z - 1000.0) / 2500.0))
    return tuple(torch.as_tensor(a, dtype=torch.float64) for a in (t, q, p))


def _window(n_lev: int, window_hours: float = 1.0):
    from tests.test_convection import _build_convective_met_window  # noqa: PLC0415

    return _build_convective_met_window(window_hours=window_hours, column=_convective_column(n_lev))


# ---------------------------------------------------------------------------
# The exponential of a rate generator
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("scale", [1e-6, 1e-3, 5e-2])
def test_generator_exponential_matches_matrix_exp_and_is_stochastic(scale: float) -> None:
    """Up to Courant numbers of several hundred (scale·Z·dt), the uniformised
    scaling-and-squaring exponential equals torch.linalg.matrix_exp to 1e-11 and
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


def test_rate_generator_rows_sum_to_zero_and_conserve_mass() -> None:
    p, lcl, lnb = _columns()
    rate = em.compute_mass_flux_matrices(
        p, lcl, lnb, torch.full((4,), 0.02, dtype=torch.float64), cfl_max_outflux_frac=None
    )
    m = em._layer_masses_per_area(p)
    for backward in (True, False):
        q = em.rate_generator(rate, m, backward=backward)
        off = q - torch.diag_embed(torch.diagonal(q, dim1=1, dim2=2))
        assert float(off.min()) >= 0.0
        assert float(q.sum(dim=2).abs().max()) < 1e-15
        assert float((m.unsqueeze(1) @ q).abs().max()) < 1e-15


@pytest.mark.parametrize("m_b", [0.01, 0.18])
def test_exponential_backward_is_stationary_and_adjoint(m_b: float) -> None:
    """mᵀP = mᵀ (well-mixed criterion) in both directions, and the backward
    transition is D⁻¹ P_fwdᵀ D, for a realistic and for the old nominal
    (CFL-violating) cloud-base flux."""

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


def test_backward_exponential_matches_the_euler_backward_matrix_for_small_steps() -> None:
    """The three-layer cycle of `test_backward_transition_points_from_current_layer_
    to_source_layer` (task T1) as a rate: for a short interval the exponential's
    backward matrix equals `backward_transition_matrices` of the same transfer
    to second order, and for a long one it still points a particle in layer 0
    to its source layer 2 rather than to layer 1 (where layer 0's air goes)."""

    fmass = torch.tensor([[[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]], dtype=torch.float64)
    m = torch.tensor([[2.0, 3.0, 4.0]], dtype=torch.float64)
    eps = 1e-6
    exp_b = em.backward_transition_matrices_exponential(fmass, m, eps)
    euler_b = em.backward_transition_matrices(fmass * eps, m)
    assert float((exp_b - euler_b).abs().max()) < 1e-11
    long_b = em.backward_transition_matrices_exponential(fmass, m, 1.0)
    assert float(long_b[0, 0, 2]) > float(long_b[0, 0, 1])
    stat = (m.unsqueeze(1) @ long_b).squeeze(1)
    torch.testing.assert_close(stat, m, rtol=1e-13, atol=0.0)


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


def test_rate_matrix_is_grid_independent_and_cfl_scaled_is_not() -> None:
    """The same column on 40 and 79 levels (every layer split in two), the same
    cloud base and top pressure, and the same cloud-base flux: the exponential
    moves nearly the same sub-cloud air into the cloud per hour (within 10%:
    only the adjacent-layer subsidence and the index-linear detrainment see the
    grid), whereas the CFL-scaled transfer is limited by the mass of the layer
    below cloud base and roughly halves."""

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


# ---------------------------------------------------------------------------
# Closure and options
# ---------------------------------------------------------------------------


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
    """ΔT_v at a fixed height above the LCL (an option, off by default)."""

    h = torch.tensor([[0.0, 100.0, 200.0, 300.0, 400.0]], dtype=torch.float64)
    p = torch.tensor([[1000.0, 990.0, 980.0, 970.0, 960.0]], dtype=torch.float64) * 100.0
    dtv = torch.tensor([[0.0, 0.0, 1.0, 2.0, 3.0]], dtype=torch.float64)
    # LCL exactly at level 1 (p_lcl = p[1]); i_lcl is the first level above it.
    out = em.trigger_dtv_at_height_above_lcl(dtv, h, p, p[:, 1], torch.tensor([2]), 150.0)
    assert float(out[0]) == pytest.approx(1.5)
    out = em.trigger_dtv_at_height_above_lcl(dtv, h, p, p[:, 1], torch.tensor([2]), 1000.0)
    assert float(out[0]) == -1.0


def test_production_defaults_are_decision_0033_and_options_validate() -> None:
    """Defaults: buoyancy closure, exact exponential; the trigger is unchanged
    (level LCL+1, +0.9 K, 50 J/kg, 500 m)."""

    from lpdm.config import EmanuelConvectionConfig  # noqa: PLC0415

    s = EmanuelReducedConvection()
    assert (s.closure, s.application, s.trigger_height_above_lcl_m) == (
        "buoyancy",
        "exponential",
        None,
    )
    assert (s.trigger_dtv_k, s.min_cape_j_kg, s.min_cloud_depth_m) == (0.9, 50.0, 500.0)
    assert pytest.approx(0.1 * 0.025 / 0.3) == s.buoyancy_mass_flux_per_k
    c = EmanuelConvectionConfig()
    assert (c.closure, c.application, c.trigger_height_above_lcl_m) == (
        "buoyancy",
        "exponential",
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


def test_legacy_options_reproduce_the_pre_0033_matrix() -> None:
    """Selecting closure cape_velocity + application cfl_scaled gives exactly the
    M_b and the construction `maybe_convect` used before decision 0033."""

    from tests.test_convection import _build_convective_met_window  # noqa: PLC0415

    met = _build_convective_met_window()
    scheme = EmanuelReducedConvection(**LEGACY, closure_c=0.1, trigger_dtv_k=0.1)
    cols = scheme.column_state(met, t_alpha=0.5, device=CPU, dtype=torch.float32)
    idx = cols.fires.nonzero().squeeze(1)
    assert idx.numel() > 0
    ref_mb = em.cloud_base_mass_flux_columns(cols.t, cols.pressure_pa, cols.i_lcl, cols.cape, 0.1)
    assert torch.equal(cols.m_b[idx], ref_mb[idx])
    fmass = em.compute_mass_flux_matrices(
        cols.pressure_pa[idx],
        cols.i_lcl[idx],
        cols.i_lnb[idx],
        (cols.m_b[idx] * cols.interval_s).float(),
    )
    lm = em._layer_masses_per_area(cols.pressure_pa[idx]).clamp(min=1e-6)
    ref = em.backward_transition_matrices(fmass, lm)
    assert torch.equal(scheme.transition_matrices(cols, idx), ref)


# ---------------------------------------------------------------------------
# The production defaults through the scheme
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("application", ["exponential", "substeps"])
def test_production_transition_is_well_mixed_and_non_negative(application: str) -> None:
    """In a scheme column (buoyancy closure, M_b > 0) the backward transition keeps
    the layer masses stationary, is row-stochastic and has no negative entry."""

    met = _window(40)
    scheme = EmanuelReducedConvection(trigger="glide", trigger_dtv_k=0.1, application=application)
    cols = scheme.column_state(met, t_alpha=0.5, device=CPU, dtype=torch.float32)
    idx = cols.fires.nonzero().squeeze(1)
    assert idx.numel() > 0 and bool((cols.m_b[idx] > 0).all())
    prob = scheme.transition_matrices(cols, idx).double()
    m = em._layer_masses_per_area(cols.pressure_pa[idx].double()).clamp(min=1e-6)
    stat = (m.unsqueeze(1) @ prob).squeeze(1)
    assert np.allclose(stat.numpy(), m.numpy(), rtol=1e-5)
    assert float(prob.min()) >= 0.0
    assert torch.allclose(prob.sum(dim=2), torch.ones_like(prob.sum(dim=2)), atol=1e-5)


def _hourly_transport(scheme: EmanuelReducedConvection, n_lev: int) -> tuple[float, float]:
    """(M_b, sub-cloud air moved into the cloud in one interval, kg m-2)."""

    cols = scheme.column_state(_window(n_lev), t_alpha=0.5, device=CPU, dtype=torch.float64)
    assert bool(cols.fires.all())
    idx = torch.tensor([0])
    pf = scheme.transition_matrices(cols, idx, backward=False).double()
    m = em._layer_masses_per_area(cols.pressure_pa[idx].double())
    return float(cols.m_b[0]), float(_sub_cloud_to_cloud_mass(pf, m, cols.i_lcl[idx])[0])


def test_scheme_transport_is_grid_independent_on_a_synthetic_column() -> None:
    """The same sounding on 23, 40 and 79 stretched levels: with the production
    defaults M_b and the hourly transport stay within 15% of the 40-level value;
    the old closure's CFL-scaled transfer changes by more than 1.5x each step.
    (Trigger relaxed to 0.1 K: ΔT_v at LCL+1 itself depends on the grid, which
    decision 0033 leaves to a separate study.)"""

    new = {
        n: _hourly_transport(EmanuelReducedConvection(trigger="glide", trigger_dtv_k=0.1), n)
        for n in (23, 40, 79)
    }
    old = {
        n: _hourly_transport(EmanuelReducedConvection(**LEGACY, trigger_dtv_k=0.1), n)
        for n in (23, 40, 79)
    }
    for n in (23, 79):
        assert new[n][0] == pytest.approx(new[40][0], rel=0.15), new
        assert new[n][1] == pytest.approx(new[40][1], rel=0.15), new
    assert old[23][1] / old[40][1] > 1.5, old
    assert old[40][1] / old[79][1] > 1.5, old


def test_scheme_is_cadence_independent() -> None:
    """A 3-hour window moves exactly what three 1-hour windows do (same met)."""

    scheme = EmanuelReducedConvection(trigger="glide", trigger_dtv_k=0.1)
    c1 = scheme.column_state(_window(40, 1.0), t_alpha=0.5, device=CPU, dtype=torch.float64)
    c3 = scheme.column_state(_window(40, 3.0), t_alpha=0.5, device=CPU, dtype=torch.float64)
    idx = torch.tensor([0])
    p1 = scheme.transition_matrices(c1, idx)
    p3 = scheme.transition_matrices(c3, idx)
    torch.testing.assert_close(p1 @ p1 @ p1, p3, rtol=0.0, atol=1e-12)


def test_run_diagnostics_count_courant_numbers_beyond_the_squaring_cap() -> None:
    scheme = EmanuelReducedConvection(trigger="glide", trigger_dtv_k=0.1)
    assert scheme.run_diagnostics()["closure"] == "buoyancy"
    cols = scheme.column_state(_window(40), t_alpha=0.5, device=CPU, dtype=torch.float64)
    idx = torch.tensor([0])
    scheme.transition_matrices(cols, idx)
    d = scheme.run_diagnostics()
    assert d["exponential_events"] == 1 and d["exponential_matrices"] == 1
    assert 0.0 < d["courant_max"] < em.EXPM_COURANT_MAX
    assert d["columns_beyond_courant_cap"] == 0
    # A thousand-year interval pushes the Courant number past the cap: counted.
    scheme.transition_matrices(cols, idx, interval_s=3.15e10)
    d = scheme.run_diagnostics()
    assert d["columns_beyond_courant_cap"] == 1 and d["courant_max"] > em.EXPM_COURANT_MAX


def test_production_defaults_move_aloft_particles_back_to_the_boundary_layer() -> None:
    """End to end through `maybe_convect` with the defaults (trigger relaxed to
    0.1 K): particles placed in the cloud are, backward in time, partly traced to
    the sub-cloud layer; none moves horizontally; weights are untouched."""

    from lpdm.gpu_engine import GPUEngine  # noqa: PLC0415

    torch.manual_seed(20261128)
    met = _window(40)
    scheme = EmanuelReducedConvection(trigger="glide", trigger_dtv_k=0.1)
    cols = scheme.column_state(met, t_alpha=0.5, device=CPU, dtype=torch.float32)
    z_base = float(cols.height_m[0, int(cols.i_lcl[0])])
    z_top = float(cols.height_m[0, int(cols.i_lnb[0])])
    n = 4000
    particles = torch.zeros(n, 4, dtype=torch.float32)
    particles[:, 2] = torch.empty(n).uniform_(z_base, z_top)
    particles[:, 3] = 1.0 / n
    out, _ = scheme.maybe_convect(
        particles,
        scheme.initialize_state(n, device=CPU, dtype=torch.float32),
        met,
        t_alpha=0.5,
        dt_seconds=60.0,
        active_mask=torch.ones(n, dtype=torch.bool),
        engine=GPUEngine(device="cpu"),
    )
    assert torch.equal(out[:, :2], particles[:, :2])
    assert torch.equal(out[:, 3], particles[:, 3])
    to_bl = int((out[:, 2] < z_base).sum())
    assert to_bl > 20, to_bl


# ---------------------------------------------------------------------------
# The production path and the wiring (review of PR #78)
# ---------------------------------------------------------------------------


def test_production_transition_is_the_backward_exponential_of_the_rate() -> None:
    """Under the defaults, `transition_matrices` (what `maybe_convect` uses) equals
    `backward_transition_matrices_exponential` of the column's rate matrix over the
    met interval: the runtime's exponential branch runs backward, not forward."""

    scheme = EmanuelReducedConvection(trigger="glide", trigger_dtv_k=0.1)
    cols = scheme.column_state(_window(40), t_alpha=0.5, device=CPU, dtype=torch.float64)
    idx = cols.fires.nonzero().squeeze(1)
    assert idx.numel() > 0
    p = cols.pressure_pa[idx]
    rate = em.compute_mass_flux_matrices(
        p, cols.i_lcl[idx], cols.i_lnb[idx], cols.m_b[idx], cfl_max_outflux_frac=None
    )
    m = em._layer_masses_per_area(p).clamp(min=1e-6)
    expected = em.backward_transition_matrices_exponential(rate, m, cols.interval_s)
    torch.testing.assert_close(scheme.transition_matrices(cols, idx), expected, rtol=0.0, atol=0.0)


def test_backward_rows_reach_the_sub_cloud_layer_and_forward_rows_do_not() -> None:
    """Physical direction: air now high in the cloud (levels LCL+3 to LNB) came,
    backward in time, partly from below cloud base (the updraft), so the backward
    rows put probability on the sub-cloud layers; forward, that air only subsides
    a layer or so in an hour and essentially never reaches them."""

    scheme = EmanuelReducedConvection(trigger="glide", trigger_dtv_k=0.1)
    cols = scheme.column_state(_window(40), t_alpha=0.5, device=CPU, dtype=torch.float64)
    idx = torch.tensor([0])
    lcl, lnb = int(cols.i_lcl[0]), int(cols.i_lnb[0])
    assert lnb > lcl + 3
    back = scheme.transition_matrices(cols, idx, backward=True)[0]
    fwd = scheme.transition_matrices(cols, idx, backward=False)[0]
    rows = slice(lcl + 3, lnb + 1)
    to_sub_cloud_back = back[rows, :lcl].sum(dim=1)
    to_sub_cloud_fwd = fwd[rows, :lcl].sum(dim=1)
    assert float(to_sub_cloud_back.mean()) > 1e-3, to_sub_cloud_back
    assert float(to_sub_cloud_fwd.max()) < 1e-2 * float(to_sub_cloud_back.mean()), to_sub_cloud_fwd


def test_substeps_apply_the_interval_over_n_per_substep() -> None:
    """`substeps` is n CFL-capped transfers of M_b·Δt/n each, not of M_b·Δt."""

    n = 4
    scheme = EmanuelReducedConvection(
        trigger="glide", trigger_dtv_k=0.1, application="substeps", n_substeps=n
    )
    cols = scheme.column_state(_window(40), t_alpha=0.5, device=CPU, dtype=torch.float64)
    idx = torch.tensor([0])
    p = cols.pressure_pa[idx]
    lm = em._layer_masses_per_area(p).clamp(min=1e-6)
    fmass = em.compute_mass_flux_matrices(
        p, cols.i_lcl[idx], cols.i_lnb[idx], cols.m_b[idx] * (cols.interval_s / n)
    )
    expected = torch.linalg.matrix_power(em.backward_transition_matrices(fmass, lm), n)
    torch.testing.assert_close(scheme.transition_matrices(cols, idx), expected)


def test_config_default_flux_per_kelvin_is_the_scheme_constant() -> None:
    from lpdm.config import EmanuelConvectionConfig  # noqa: PLC0415

    assert EmanuelConvectionConfig().buoyancy_mass_flux_per_k == em.BUOYANCY_MASS_FLUX_PER_K


def _run_config(emanuel: dict | None):
    from pathlib import Path  # noqa: PLC0415

    import yaml  # noqa: PLC0415

    from lpdm.config import RunConfig  # noqa: PLC0415

    path = Path(__file__).resolve().parents[1] / "configs/smoke_mhd_single_release.yaml"
    data = yaml.safe_load(path.read_text())
    data["convection"] = {"scheme": "emanuel_reduced"}
    if emanuel is not None:
        data["convection"]["emanuel"] = emanuel
    return RunConfig.model_validate(data)


@pytest.mark.parametrize(
    "emanuel",
    [
        None,
        {**LEGACY, "closure_c": 0.05},
        {"closure": "buoyancy", "application": "substeps", "n_substeps": 3},
    ],
)
def test_main_passes_the_convection_options_to_the_scheme(emanuel: dict | None) -> None:
    """`main._convection_scheme_kwargs` carries closure, application and their
    parameters, the defaults and the legacy values, through to the scheme."""

    from lpdm.convection import get_scheme  # noqa: PLC0415
    from lpdm.main import _convection_scheme_kwargs  # noqa: PLC0415

    cfg = _run_config(emanuel)
    scheme = get_scheme("emanuel_reduced", **_convection_scheme_kwargs(cfg))
    e = cfg.convection.emanuel
    assert (scheme.closure, scheme.application) == (e.closure, e.application)
    assert scheme.trigger == e.trigger
    assert scheme.closure_c == e.closure_c
    assert scheme.n_substeps == e.n_substeps
    assert scheme.buoyancy_mass_flux_per_k == e.buoyancy_mass_flux_per_k
    if emanuel is None:
        assert (scheme.closure, scheme.application) == ("buoyancy", "exponential")
    else:
        assert (scheme.closure, scheme.application) == (emanuel["closure"], emanuel["application"])


def test_options_the_selected_closure_ignores_are_warned() -> None:
    """closure_c / updraft_w_max_m_s without closure cape_velocity, and n_substeps
    without application substeps, are warned (not refused) for
    runtime.physics_warnings; nothing is warned when they apply or are unset."""

    from lpdm.main import _convection_option_warnings  # noqa: PLC0415

    assert _convection_option_warnings(_run_config(None)) == []
    msgs = _convection_option_warnings(
        _run_config({"closure_c": 0.03, "updraft_w_max_m_s": 5.0, "n_substeps": 4})
    )
    assert len(msgs) == 3
    assert any("closure_c" in m for m in msgs) and any("n_substeps" in m for m in msgs)
    legacy = {"closure": "cape_velocity", "application": "substeps", "closure_c": 0.03}
    legacy["trigger"] = "glide"
    assert _convection_option_warnings(_run_config({**legacy, "n_substeps": 2})) == []
