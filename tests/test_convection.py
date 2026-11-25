"""Unit tests for the convection package.

Covers:
- Free-function physics: q ↔ e conversions, virtual T, LCL temperature/pressure,
  potential T, equivalent potential T, saturation specific humidity, moist-adiabat
  pseudo-adiabatic lift.
- LCL/LNB/CAPE column analysis on a constructed convective sounding.
- Mass-flux matrix shape, sparsity, and per-row normalisation.
- Scheme-level: NoConvection pass-through, EmanuelReducedConvection trigger,
  particle mass conservation (no particles gained/lost), constructor validation.
"""

from __future__ import annotations

import math
from datetime import UTC

import numpy as np
import pytest
import torch

from lpdm.convection import (
    EmanuelReducedConvection,
    NoConvection,
    get_scheme,
    list_schemes,
)
from lpdm.convection import emanuel as emanuel_module
from lpdm.convection.emanuel import (
    _layer_masses_per_area,
    compute_lcl_lnb_cape,
    compute_mass_flux_matrix,
    equivalent_potential_temperature,
    lcl_pressure_poisson,
    lcl_temperature_bolton,
    lift_parcel_moist_pseudo_adiabatic,
    potential_temperature,
    saturation_specific_humidity,
    saturation_vapor_pressure,
    specific_humidity_to_vapor_pressure,
    vapor_pressure_to_specific_humidity,
    virtual_temperature,
)
from lpdm.met_reader import HourlyMetTensors

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_registry_lists_both_schemes() -> None:
    names = list_schemes()
    assert "none" in names
    assert "emanuel_reduced" in names
    assert isinstance(get_scheme("none"), NoConvection)
    assert isinstance(get_scheme("emanuel_reduced"), EmanuelReducedConvection)


# ---------------------------------------------------------------------------
# Free-function physics
# ---------------------------------------------------------------------------


def test_saturation_vapor_pressure_bolton_reference_values() -> None:
    """e_sat from Bolton (1980) against tabulated values from standard textbooks
    (Wallace & Hobbs, Curry & Webster). 1% tolerance, well within Bolton's
    stated accuracy."""

    t = torch.tensor([273.15, 283.15, 293.15, 303.15], dtype=torch.float64)
    e_sat = saturation_vapor_pressure(t)
    # Reference: 6.11, 12.27, 23.37, 42.43 hPa
    ref_hpa = torch.tensor([6.11, 12.27, 23.37, 42.43], dtype=torch.float64)
    rel_err = (e_sat / 100.0 - ref_hpa).abs() / ref_hpa
    assert rel_err.max() < 0.01, f"max rel error {rel_err.max()} > 1%; got {e_sat / 100.0}"


def test_q_to_e_round_trip() -> None:
    """q → e → q must be the identity."""

    q = torch.tensor([1e-3, 5e-3, 1e-2, 2e-2], dtype=torch.float64)
    p = torch.full_like(q, 100000.0)
    e = specific_humidity_to_vapor_pressure(q, p)
    q_back = vapor_pressure_to_specific_humidity(e, p)
    assert torch.allclose(q, q_back, atol=1e-12)


def test_virtual_temperature_increases_with_humidity() -> None:
    """T_v > T for any q > 0; T_v ≈ T·(1 + 0.61q) at low q."""

    t = torch.tensor([280.0], dtype=torch.float64)
    q_dry = torch.tensor([0.0], dtype=torch.float64)
    q_wet = torch.tensor([0.01], dtype=torch.float64)
    tv_dry = virtual_temperature(t, q_dry)
    tv_wet = virtual_temperature(t, q_wet)
    assert float(tv_dry) == float(t)
    assert float(tv_wet) > float(t)
    # Small-q approximation: T_v ≈ T·(1 + 0.61·q).
    expected = float(t) * (1.0 + 0.61 * float(q_wet))
    assert abs(float(tv_wet) - expected) / expected < 0.001


def test_lcl_pressure_below_surface_pressure() -> None:
    """A parcel at the surface always condenses ABOVE its starting altitude
    (p_LCL < p_surface) unless it's already saturated. Sanity check for the
    Bolton + Poisson combo."""

    t_sfc = torch.tensor([293.15], dtype=torch.float64)
    q_sfc = torch.tensor([0.008], dtype=torch.float64)  # 8 g/kg (moderate)
    p_sfc = torch.tensor([101325.0], dtype=torch.float64)
    t_lcl = lcl_temperature_bolton(t_sfc, q_sfc, p_sfc)
    p_lcl = lcl_pressure_poisson(t_sfc, t_lcl, p_sfc)
    assert float(t_lcl) < float(t_sfc)
    assert float(p_lcl) < float(p_sfc)
    # Roughly: 20°C surface with 8 g/kg → LCL at ~900 m → p_LCL ~ 91 kPa
    assert 80000.0 < float(p_lcl) < 100000.0


def test_potential_temperature_at_reference_pressure_is_temperature() -> None:
    """θ(T, p=p0) == T by construction."""

    t = torch.tensor([288.0], dtype=torch.float64)
    theta = potential_temperature(t, torch.tensor([100000.0], dtype=torch.float64))
    assert abs(float(theta) - float(t)) < 1e-9


def test_equivalent_potential_temperature_exceeds_potential_temperature() -> None:
    """θ_e ≥ θ for any q ≥ 0; the latent-heat factor adds a positive contribution."""

    t = torch.tensor([288.0], dtype=torch.float64)
    q = torch.tensor([0.005], dtype=torch.float64)
    p = torch.tensor([95000.0], dtype=torch.float64)
    theta = potential_temperature(t, p)
    theta_e = equivalent_potential_temperature(t, q, p)
    assert float(theta_e) > float(theta)


def test_moist_adiabat_lift_conserves_theta_e() -> None:
    """A parcel lifted from (T0, p0, q0) along its moist adiabat must conserve
    θ_e. Forward + inverse: lift to a higher level, solve for T there, then
    compute θ_e at the new state and compare."""

    t0 = torch.tensor([290.0], dtype=torch.float64)
    q0 = torch.tensor([0.010], dtype=torch.float64)
    p0 = torch.tensor([95000.0], dtype=torch.float64)
    theta_e0 = equivalent_potential_temperature(t0, q0, p0)

    p_target = torch.tensor([70000.0], dtype=torch.float64)
    t_at_target = lift_parcel_moist_pseudo_adiabatic(theta_e0, p_target)
    q_at_target = saturation_specific_humidity(t_at_target, p_target)
    theta_e_at_target = equivalent_potential_temperature(t_at_target, q_at_target, p_target)

    # θ_e conserved to within the iterative solver's tolerance.
    rel_err = float((theta_e_at_target - theta_e0).abs() / theta_e0)
    assert rel_err < 0.01, f"θ_e changed by {rel_err:.4%}; expected < 1%"


# ---------------------------------------------------------------------------
# CAPE / LCL / LNB on synthetic columns
# ---------------------------------------------------------------------------


def _moist_unstable_column(n_lev: int = 12) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build a synthetic surface-up sounding with a deep convectively-unstable
    profile: warm, moist surface (300 K, 14 g/kg, 1000 hPa) over a roughly-
    dry-adiabatic boundary layer + slowly-cooling free troposphere. CAPE > 0,
    LCL near 950 hPa, LNB around 200 hPa for this profile.
    """

    # Pressure: 1000 → 100 hPa over n_lev levels (linear in log p).
    p_hpa = np.exp(np.linspace(np.log(1000.0), np.log(100.0), n_lev))
    p_pa = torch.as_tensor(p_hpa * 100.0, dtype=torch.float64)
    # Temperature: 300 K at surface, lapse 6.5 K / 1 km up to tropopause at
    # ~12 km, then 200 K above. Approximated via pressure → altitude (hyp).
    z_km = -8.0 * np.log(p_hpa / 1000.0)  # crude scale height 8 km
    t_k = 300.0 - 6.5 * z_km
    t_k = np.maximum(t_k, 200.0)
    t = torch.as_tensor(t_k, dtype=torch.float64)
    # Specific humidity: ~14 g/kg at surface, dropping exponentially with p.
    q_kg = 0.014 * np.exp(-(1000.0 - p_hpa) / 400.0)
    q = torch.as_tensor(q_kg, dtype=torch.float64).clamp(min=1e-6)
    return t, q, p_pa


def _dry_stable_column(n_lev: int = 8) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Strongly stable, dry profile (no CAPE) — convection should NOT trigger."""

    p_hpa = np.linspace(1000.0, 200.0, n_lev)
    p_pa = torch.as_tensor(p_hpa * 100.0, dtype=torch.float64)
    # Strong inversion: T increases with height for first few levels then cools.
    z_km = -8.0 * np.log(p_hpa / 1000.0)
    t_k = 280.0 + 2.0 * z_km  # rises 2 K/km → very stable
    t_k = np.minimum(t_k, 290.0)
    t = torch.as_tensor(t_k, dtype=torch.float64)
    # Bone dry.
    q = torch.full((n_lev,), 1e-5, dtype=torch.float64)
    return t, q, p_pa


def test_lcl_lnb_cape_on_unstable_column_triggers() -> None:
    """On a deep moist-unstable sounding, the parcel-lift CAPE calculation
    should return a positive CAPE and a meaningful LCL/LNB pair separated by
    several km."""

    t, q, p = _moist_unstable_column()
    i_lcl, i_lnb, cape, dtv_lcl_plus_1 = compute_lcl_lnb_cape(t, q, p)
    assert int(i_lcl) > 0, f"LCL not found in unstable column (got {int(i_lcl)})"
    assert int(i_lnb) > int(i_lcl), f"LNB ({int(i_lnb)}) not above LCL ({int(i_lcl)})"
    assert float(cape) > 100.0, f"expected CAPE > 100 J/kg, got {float(cape):.1f}"
    # The trigger diagnostic should be positive (parcel buoyant at LCL+1).
    assert float(dtv_lcl_plus_1) > 0, (
        f"parcel-env T_v difference at LCL+1 = {float(dtv_lcl_plus_1):.3f} K (expected > 0 for an unstable column)"
    )


def test_lcl_lnb_cape_on_stable_column_does_not_trigger() -> None:
    """On a dry-stable sounding, either no LCL exists OR the LCL+1 buoyancy
    check fails and CAPE is ~0."""

    t, q, p = _dry_stable_column()
    i_lcl, i_lnb, cape, dtv_lcl_plus_1 = compute_lcl_lnb_cape(t, q, p)
    # CAPE essentially zero in either case (negative buoyancy above LCL).
    assert float(cape) < 50.0, f"expected CAPE ~ 0 J/kg in stable column, got {float(cape):.1f}"


# ---------------------------------------------------------------------------
# Mass-flux matrix
# ---------------------------------------------------------------------------


def test_mass_flux_matrix_zero_when_no_cloud() -> None:
    """If i_lcl < 0 or i_lnb ≤ i_lcl, MA must be all zeros (no convection)."""

    t, q, p = _dry_stable_column()
    ma = compute_mass_flux_matrix(t, q, p, i_lcl=-1, i_lnb=-1, m_b=1.0)
    assert torch.all(ma == 0.0)
    ma2 = compute_mass_flux_matrix(t, q, p, i_lcl=3, i_lnb=3, m_b=1.0)
    assert torch.all(ma2 == 0.0)


def test_mass_flux_matrix_no_transport_above_cloud_top() -> None:
    """Convection touches nothing above the cloud top: every entry with a source
    row OR destination column strictly above LNB must be zero (updraft detrains
    within [LCL, LNB]; compensating subsidence acts only within the circulation
    below LNB). Entries in BL destination columns (< LCL) ARE expected now — that
    is the compensating subsidence, absent from the pre-2026-07-02 matrix."""

    t, q, p = _moist_unstable_column()
    i_lcl, i_lnb, _cape, _ = compute_lcl_lnb_cape(t, q, p)
    fmass = compute_mass_flux_matrix(t, q, p, i_lcl=int(i_lcl), i_lnb=int(i_lnb), m_b=0.05)
    n = fmass.shape[0]
    for r in range(int(i_lnb) + 1, n):
        assert torch.all(fmass[r, :] == 0.0), f"above-LNB source row {r} should be all zero"
        assert torch.all(fmass[:, r] == 0.0), f"above-LNB destination column {r} should be all zero"
    # Compensating subsidence must actually be present (some BL-column entry > 0).
    assert float(fmass[:, : int(i_lcl)].sum()) > 0.0, (
        "expected compensating subsidence into BL layers"
    )


def test_mass_flux_matrix_is_non_divergent() -> None:
    """Well-mixed guarantee (Finding 2, 2026-07-02 review): the convective flux
    matrix must be NON-DIVERGENT — every layer's row sum equals its column sum
    (mass leaving a layer = mass entering it). The diagonal closure makes each row
    sum to the layer air mass m_i; non-divergence then forces each COLUMN to sum
    to m_i too, which is exactly what makes the redistribution preserve a
    mass-proportional (well-mixed) distribution in BOTH time directions.
    Compensating subsidence is what closes the circulation. The pre-fix matrix
    (updraft only, no subsidence) was divergent and failed this."""

    t, q, p = _moist_unstable_column()
    i_lcl_t, i_lnb_t, _cape, _ = compute_lcl_lnb_cape(t, q, p)
    i_lcl, i_lnb = int(i_lcl_t), int(i_lnb_t)
    # compute_mass_flux_matrix returns the OFF-DIAGONAL flux matrix fmass[i,j]
    # (mass from layer i to j); the redistribution adds the stay-diagonal.
    fmass = compute_mass_flux_matrix(t, q, p, i_lcl=i_lcl, i_lnb=i_lnb, m_b=0.05)

    assert torch.allclose(fmass.diagonal(), torch.zeros_like(fmass.diagonal())), (
        "matrix holds off-diagonal fluxes only"
    )
    flux_out = fmass.sum(dim=1)  # mass leaving each layer
    flux_in = fmass.sum(dim=0)  # mass entering each layer
    assert torch.allclose(flux_out, flux_in, rtol=1e-5, atol=1e-9), (
        "divergent matrix (no compensating subsidence) → not well-mixed"
    )


def test_convection_transition_preserves_mass_distribution_both_directions() -> None:
    """V1 well-mixed test, proven exactly at the transition-matrix level (Finding 2
    acceptance): the layer-mass vector m must be a stationary distribution of the
    redistribution Markov operator, for BOTH the forward (row-sampled) and backward
    (column-sampled, GLIDE's mode) transitions. i.e. mᵀP = mᵀ. This is the V1
    condition (a mass-proportional ensemble stays mass-proportional) with the
    Monte-Carlo noise removed."""

    t, q, p = _moist_unstable_column()
    i_lcl_t, i_lnb_t, _cape, _ = compute_lcl_lnb_cape(t, q, p)
    i_lcl, i_lnb = int(i_lcl_t), int(i_lnb_t)
    m = _layer_masses_per_area(p)
    fmass = compute_mass_flux_matrix(t, q, p, i_lcl=i_lcl, i_lnb=i_lnb, m_b=0.05)

    # Diagonal closure (FLEXPART calcmatrix): each row sums to the layer mass.
    fmassfrac = fmass.clone()
    fmassfrac += torch.diag(m - fmass.sum(dim=1))

    # Forward transition P[i→j] = fmassfrac[i,j]/m_i (rows are the source).
    p_fwd = fmassfrac / m.unsqueeze(1)
    assert torch.allclose(p_fwd.sum(dim=1), torch.ones_like(m), atol=1e-6), (
        "forward rows must be probability distributions"
    )
    assert (p_fwd >= -1e-9).all(), "forward transition probabilities must be non-negative (CFL)"
    assert torch.allclose(m @ p_fwd, m, rtol=1e-5, atol=1e-9), (
        "m must be stationary under the forward transition"
    )

    # Backward transition P[i→j] = fmassfrac[j,i]/m_i (transpose; FLEXPART redist,
    # ldirect=-1) — this is the mode GLIDE actually runs.
    p_bwd = fmassfrac.t() / m.unsqueeze(1)
    assert torch.allclose(p_bwd.sum(dim=1), torch.ones_like(m), atol=1e-6), (
        "backward rows must be probability distributions"
    )
    assert (p_bwd >= -1e-9).all(), "backward transition probabilities must be non-negative (CFL)"
    assert torch.allclose(m @ p_bwd, m, rtol=1e-5, atol=1e-9), (
        "m must be stationary under the backward transition"
    )


# ---------------------------------------------------------------------------
# Scheme-level integration
# ---------------------------------------------------------------------------


def test_no_convection_is_pass_through() -> None:
    """NoConvection must not move any particle, regardless of met state."""

    from datetime import datetime, timedelta

    from lpdm.gpu_engine import GPUEngine
    from lpdm.met_reader import HourlyMetTensors, MetFieldMetadata

    # Minimal met window (NoConvection ignores it).
    t0 = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)
    shape = (3, 2, 2)
    fields = torch.zeros((1, *shape), dtype=torch.float32)
    level = np.array([0.0, 1000.0, 4000.0])
    metadata = MetFieldMetadata(
        lon=np.array([-1.0, 1.0]),
        lat=np.array([-1.0, 1.0]),
        level=level,
        pressure_level_hpa=np.array([1000.0, 850.0, 500.0]),
        time_start=t0,
        time_end=t0 + timedelta(hours=1),
        variable_units={"placeholder": ""},
    )
    met = HourlyMetTensors(
        hour_start=fields,
        hour_end=fields,
        metadata=metadata,
        channel_names=("placeholder",),
        height_agl_m=torch.as_tensor(level, dtype=torch.float32)
        .view(3, 1, 1)
        .expand(shape)
        .contiguous(),
    )

    engine = GPUEngine(device="cpu")
    scheme = NoConvection()
    state = scheme.initialize_state(5, device=torch.device("cpu"), dtype=torch.float32)
    particles = torch.tensor(
        [[0.0, 0.0, 100.0, 0.2]] * 5,
        dtype=torch.float32,
    )
    active = torch.ones(5, dtype=torch.bool)

    out, _ = scheme.maybe_convect(
        particles,
        state,
        met,
        t_alpha=0.5,
        dt_seconds=300.0,
        active_mask=active,
        engine=engine,
    )
    assert torch.equal(particles, out)


def _build_convective_met_window(
    blh_m: float = 1500.0, window_hours: float = 1.0
) -> HourlyMetTensors:
    """Build a met window where Emanuel's parcel lift WILL trigger deep
    convection. Spatially uniform; columns built from a moist-unstable sounding.

    `window_hours` sets the bracket width, which is also the interval the scheme
    fires on and therefore the timescale it scales the mass flux by."""

    from datetime import datetime, timedelta

    from lpdm.met_reader import HourlyMetTensors, MetFieldMetadata

    t_col, q_col, p_col = _moist_unstable_column(n_lev=12)
    # Spatially broadcast.
    n_lev, n_lat, n_lon = t_col.numel(), 3, 3
    shape = (n_lev, n_lat, n_lon)
    t_3d = t_col.to(torch.float32).view(n_lev, 1, 1).expand(shape).contiguous()
    q_3d = q_col.to(torch.float32).view(n_lev, 1, 1).expand(shape).contiguous()

    chans = {
        "u": torch.zeros(shape),
        "v": torch.zeros(shape),
        "w": torch.zeros(shape),
        "blh": torch.full(shape, blh_m),
        "sp": torch.full(shape, float(p_col[0])),
        "t": t_3d,
        "q": q_3d,
    }
    names = ("u", "v", "w", "blh", "sp", "t", "q")
    fields = torch.stack([chans[n] for n in names], dim=0)

    # Convert pressure to AGL height via hypsometric (matches what the met
    # reader does — bbox-mean level array).
    height = np.zeros(n_lev)
    for k in range(1, n_lev):
        t_mean = 0.5 * float(t_col[k] + t_col[k - 1])
        dz = 287.05 * t_mean * (math.log(float(p_col[k - 1])) - math.log(float(p_col[k]))) / 9.80665
        height[k] = height[k - 1] + dz

    t0 = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)
    metadata = MetFieldMetadata(
        lon=np.linspace(-1.0, 1.0, n_lon),
        lat=np.linspace(-1.0, 1.0, n_lat),
        level=height,  # ascending altitude
        pressure_level_hpa=(np.asarray(p_col, dtype=float) / 100.0),
        time_start=t0,
        time_end=t0 + timedelta(hours=window_hours),
        variable_units=dict.fromkeys(names, ""),
    )
    height_3d = (
        torch.as_tensor(height, dtype=torch.float32).view(n_lev, 1, 1).expand(shape).contiguous()
    )
    return HourlyMetTensors(
        hour_start=fields,
        hour_end=fields,
        metadata=metadata,
        channel_names=names,
        height_agl_m=height_3d,
    )


def test_emanuel_backward_connects_aloft_particles_to_the_surface() -> None:
    """End-to-end backward convection: build a convectively unstable column, place
    particles in the cloud/free troposphere, call ``maybe_convect`` (which runs in
    GLIDE's backward mode), and assert (a) some are displaced, (b) mass is
    conserved, (c) the coherent jumps are DOWNWARD — aloft particles are traced
    back to the boundary-layer source.

    This is the adjoint of forward updraft: forward convection lofts surface air to
    the cloud top, so the backward (footprint) operator must move a particle that
    is aloft *now* down to the surface it came from. Getting this direction wrong
    (applying the forward row instead of the transposed column) would instead push
    aloft particles further up and starve the surface footprint."""

    from lpdm.gpu_engine import GPUEngine

    torch.manual_seed(20260601)
    met = _build_convective_met_window()
    engine = GPUEngine(device="cpu")
    # Relaxed trigger: the synthetic 12-level column underestimates the LCL+1
    # buoyancy that the default 0.9 K threshold is tuned for on 60-level ECMWF
    # columns. This test validates redistribution once fired, not the threshold
    # (covered by test_emanuel_does_not_fire_on_stable_column).
    scheme = EmanuelReducedConvection(closure_c=0.1, trigger_dtv_k=0.1)

    # Cloud extent for this column, so we can place particles inside it.
    t_col, q_col, p_col = _moist_unstable_column(n_lev=12)
    i_lcl_t, i_lnb_t, _cape, _ = compute_lcl_lnb_cape(t_col, q_col, p_col)
    i_lcl, i_lnb = int(i_lcl_t), int(i_lnb_t)
    height = torch.as_tensor(met.metadata.level, dtype=torch.float32)
    z_cloud_base = float(height[i_lcl])
    z_cloud_top = float(height[i_lnb])

    n = 2000
    particles = torch.zeros(n, 4, dtype=torch.float32)
    # Place particles up in the cloud layer (aloft).
    particles[:, 2] = torch.empty(n).uniform_(z_cloud_base, z_cloud_top)
    particles[:, 3] = 1.0 / n
    z_initial = particles[:, 2].clone()
    state = scheme.initialize_state(n, device=torch.device("cpu"), dtype=torch.float32)
    active = torch.ones(n, dtype=torch.bool)

    particles_out, _ = scheme.maybe_convect(
        particles,
        state,
        met,
        t_alpha=0.5,
        dt_seconds=300.0,
        active_mask=active,
        engine=engine,
    )

    displacement = particles_out[:, 2] - z_initial
    displaced = displacement.abs() > 0.1
    n_displaced = int(displaced.sum().item())
    assert n_displaced > 0, (
        "Expected at least some particles to be redistributed by deep convection"
    )

    # Particle count / mass is conserved (no creation/loss).
    assert torch.allclose(particles_out[:, 3].sum(), particles[:, 3].sum(), atol=1e-12)

    # Net transport is DOWNWARD: the coherent reverse-updraft (big cloud->BL jumps)
    # dominates the small one-level reverse-subsidence moves, so the ensemble-mean
    # displacement is strongly negative. (Median is noisy here — roughly half the
    # displaced particles take small upward subsidence-reversal steps.)
    mean_disp = float(displacement.mean())
    assert mean_disp < -200.0, (
        f"mean displacement is {mean_disp:.0f} m; expected net downward transport "
        "(< -200 m) for backward deep convection"
    )
    # A meaningful fraction of displaced particles is traced all the way back to
    # the boundary-layer source (below cloud base) — the footprint-relevant path.
    frac_to_bl = float((particles_out[displaced, 2] < z_cloud_base).float().mean())
    assert frac_to_bl > 0.25, (
        f"only {frac_to_bl:.0%} of displaced particles reached the boundary layer; "
        "backward convection should trace aloft air to the surface source"
    )


def test_emanuel_does_not_fire_on_stable_column() -> None:
    """Negative control: with a dry-stable sounding (no CAPE) the scheme must
    NOT redistribute any particle. This pins the Forster (2007) trigger check."""

    from datetime import datetime, timedelta

    from lpdm.gpu_engine import GPUEngine
    from lpdm.met_reader import HourlyMetTensors, MetFieldMetadata

    torch.manual_seed(20260602)
    t_col, q_col, p_col = _dry_stable_column(n_lev=8)
    n_lev, n_lat, n_lon = t_col.numel(), 3, 3
    shape = (n_lev, n_lat, n_lon)
    t_3d = t_col.to(torch.float32).view(n_lev, 1, 1).expand(shape).contiguous()
    q_3d = q_col.to(torch.float32).view(n_lev, 1, 1).expand(shape).contiguous()
    chans = {
        "u": torch.zeros(shape),
        "v": torch.zeros(shape),
        "w": torch.zeros(shape),
        "blh": torch.full(shape, 500.0),
        "sp": torch.full(shape, float(p_col[0])),
        "t": t_3d,
        "q": q_3d,
    }
    names = ("u", "v", "w", "blh", "sp", "t", "q")
    fields = torch.stack([chans[n] for n in names], dim=0)
    height = np.linspace(0.0, 10000.0, n_lev)
    t0 = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)
    metadata = MetFieldMetadata(
        lon=np.linspace(-1.0, 1.0, n_lon),
        lat=np.linspace(-1.0, 1.0, n_lat),
        level=height,
        pressure_level_hpa=(np.asarray(p_col, dtype=float) / 100.0),
        time_start=t0,
        time_end=t0 + timedelta(hours=1),
        variable_units=dict.fromkeys(names, ""),
    )
    height_3d = (
        torch.as_tensor(height, dtype=torch.float32).view(n_lev, 1, 1).expand(shape).contiguous()
    )
    met = HourlyMetTensors(
        hour_start=fields,
        hour_end=fields,
        metadata=metadata,
        channel_names=names,
        height_agl_m=height_3d,
    )

    engine = GPUEngine(device="cpu")
    scheme = EmanuelReducedConvection()
    n = 200
    particles = torch.zeros(n, 4, dtype=torch.float32)
    particles[:, 2] = torch.empty(n).uniform_(50.0, 1500.0)
    particles[:, 3] = 1.0 / n
    z_initial = particles[:, 2].clone()
    state = scheme.initialize_state(n, device=torch.device("cpu"), dtype=torch.float32)
    active = torch.ones(n, dtype=torch.bool)

    particles_out, _ = scheme.maybe_convect(
        particles,
        state,
        met,
        t_alpha=0.5,
        dt_seconds=300.0,
        active_mask=active,
        engine=engine,
    )
    # No particle should have been moved — the trigger fails on this sounding.
    assert torch.equal(particles_out[:, 2], z_initial), (
        "convection should NOT fire on a dry-stable sounding (no CAPE)"
    )


def test_emanuel_constructor_validates_params() -> None:
    """Out-of-range constructor args raise ValueError."""

    with pytest.raises(ValueError, match="closure_c"):
        EmanuelReducedConvection(closure_c=0.0)
    with pytest.raises(ValueError, match="trigger_dtv_k"):
        EmanuelReducedConvection(trigger_dtv_k=-1.0)
    with pytest.raises(ValueError, match="min_cape_j_kg"):
        EmanuelReducedConvection(min_cape_j_kg=-10.0)
    with pytest.raises(ValueError, match="min_cloud_depth_m"):
        EmanuelReducedConvection(min_cloud_depth_m=-5.0)


def _winter_inversion_column(n_lev: int = 10) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """January-like maritime column: cold surface (272 K) under a low-level
    inversion, moist near the surface but conditionally stable throughout — the
    bbox-mean regime the winter validation runs feed the scheme."""

    p_hpa = np.linspace(1000.0, 200.0, n_lev)
    z_km = -8.0 * np.log(p_hpa / 1000.0)
    t_k = np.where(z_km < 1.0, 272.0 + 3.0 * z_km, 275.0 - 6.5 * (z_km - 1.0))
    t_k = np.maximum(t_k, 210.0)
    q = 3e-3 * np.exp(-z_km / 2.0)
    return (
        torch.as_tensor(t_k, dtype=torch.float64),
        torch.as_tensor(q, dtype=torch.float64),
        torch.as_tensor(p_hpa * 100.0, dtype=torch.float64),
    )


def test_emanuel_does_not_fire_on_winter_inversion_column() -> None:
    """Runtime guard for the January validation runs (a test instead of a log
    grep for spurious Emanuel fires): a cold moist column capped by a surface
    inversion must NOT trigger; every particle passes through untouched."""

    from datetime import datetime, timedelta

    from lpdm.gpu_engine import GPUEngine
    from lpdm.met_reader import HourlyMetTensors, MetFieldMetadata

    torch.manual_seed(20260717)
    t_col, q_col, p_col = _winter_inversion_column()
    n_lev, n_lat, n_lon = t_col.numel(), 3, 3
    shape = (n_lev, n_lat, n_lon)
    t_3d = t_col.to(torch.float32).view(n_lev, 1, 1).expand(shape).contiguous()
    q_3d = q_col.to(torch.float32).view(n_lev, 1, 1).expand(shape).contiguous()
    chans = {
        "u": torch.zeros(shape),
        "v": torch.zeros(shape),
        "w": torch.zeros(shape),
        "blh": torch.full(shape, 300.0),
        "sp": torch.full(shape, float(p_col[0])),
        "t": t_3d,
        "q": q_3d,
    }
    names = ("u", "v", "w", "blh", "sp", "t", "q")
    fields = torch.stack([chans[n] for n in names], dim=0)
    height = np.asarray(-8000.0 * np.log(np.linspace(1000.0, 200.0, n_lev) / 1000.0))
    t0 = datetime(2024, 1, 15, 12, 0, 0, tzinfo=UTC)
    metadata = MetFieldMetadata(
        lon=np.linspace(-1.0, 1.0, n_lon),
        lat=np.linspace(-1.0, 1.0, n_lat),
        level=height,
        pressure_level_hpa=(np.asarray(p_col, dtype=float) / 100.0),
        time_start=t0,
        time_end=t0 + timedelta(hours=1),
        variable_units=dict.fromkeys(names, ""),
    )
    height_3d = (
        torch.as_tensor(height, dtype=torch.float32).view(n_lev, 1, 1).expand(shape).contiguous()
    )
    met = HourlyMetTensors(
        hour_start=fields,
        hour_end=fields,
        metadata=metadata,
        channel_names=names,
        height_agl_m=height_3d,
    )

    engine = GPUEngine(device="cpu")
    scheme = EmanuelReducedConvection()
    n = 200
    particles = torch.zeros(n, 4, dtype=torch.float32)
    particles[:, 2] = torch.empty(n).uniform_(20.0, 3000.0)
    particles[:, 3] = 1.0 / n
    z_initial = particles[:, 2].clone()
    state = scheme.initialize_state(n, device=torch.device("cpu"), dtype=torch.float32)
    active = torch.ones(n, dtype=torch.bool)

    particles_out, _ = scheme.maybe_convect(
        particles,
        state,
        met,
        t_alpha=0.5,
        dt_seconds=300.0,
        active_mask=active,
        engine=engine,
    )
    assert torch.equal(particles_out[:, 2], z_initial), (
        "Emanuel convection fired on a stable winter-inversion column"
    )


def _event_m_b_for_window(window_hours: float) -> float:
    """The per-event cloud-base mass (kg/m², flux x interval) the scheme uses for
    a window of this width, in one column of a uniform convective box.

    Read from the column diagnosis rather than by counting displaced particles:
    the redistribution probability saturates well before 3-hourly at realistic
    closure settings, so a particle count cannot distinguish the two.
    """

    met = _build_convective_met_window(window_hours=window_hours)
    scheme = EmanuelReducedConvection(closure_c=0.1, trigger_dtv_k=0.1)
    cols = scheme.column_state(met, t_alpha=0.5, device=torch.device("cpu"), dtype=torch.float32)
    assert bool(cols.fires.all())
    return float(cols.m_b[0]) * cols.interval_s


def test_convective_mass_flux_follows_the_met_window_width() -> None:
    """The scheme fires once per met window and redistributes the mass the cloud-base
    flux moves over that window, so the mass it redistributes must scale with the
    window. The interval was hardcoded at 3600 s, which under-convected a 3-hourly
    source by exactly this factor of 3."""

    hourly = _event_m_b_for_window(1.0)
    three_hourly = _event_m_b_for_window(3.0)

    assert hourly > 0.0
    assert three_hourly == pytest.approx(3.0 * hourly, rel=1e-12)


# ---------------------------------------------------------------------------
# Per-column convection (task A02)
# ---------------------------------------------------------------------------


def _old_bbox_mean_matrix(scheme, met, t_alpha: float) -> tuple[torch.Tensor, torch.Tensor]:
    """The pre-A02 scheme's matrix: one parcel lift in the bbox-mean column,
    using the single-column reference functions and the closure exactly as
    `maybe_convect` evaluated them before A02. Returns (fmass, height_col)."""

    dtype = torch.float32
    t_start, t_end = met.channel("t")
    q_start, q_end = met.channel("q")
    t_col = (t_start.to(dtype) * (1.0 - t_alpha) + t_end.to(dtype) * t_alpha).mean(dim=(1, 2))
    q_col = (
        (q_start.to(dtype) * (1.0 - t_alpha) + q_end.to(dtype) * t_alpha)
        .mean(dim=(1, 2))
        .clamp(min=0.0)
    )
    height_col = met.height_agl_m.to(dtype).mean(dim=(1, 2))
    p_col = torch.as_tensor(np.array(met.metadata.pressure_level_hpa), dtype=dtype) * 100.0
    if not bool(height_col[-1] > height_col[0]):
        t_col, q_col, p_col, height_col = (x.flip(0) for x in (t_col, q_col, p_col, height_col))
    i_lcl_t, i_lnb_t, cape, dtv = compute_lcl_lnb_cape(t_col, q_col, p_col)
    i_lcl, i_lnb = int(i_lcl_t), int(i_lnb_t)
    fires = (
        i_lcl >= 0
        and i_lnb > i_lcl
        and float(cape) >= scheme.min_cape_j_kg
        and float(dtv) >= scheme.trigger_dtv_k
        and float(height_col[i_lnb] - height_col[i_lcl]) >= scheme.min_cloud_depth_m
    )
    assert fires, "reference column must fire for this comparison"
    rho_lcl = p_col[i_lcl] / (emanuel_module.R_DRY_AIR_J_KG_K * t_col[i_lcl].clamp(min=150.0))
    w_buoy = min(math.sqrt(2.0 * max(float(cape), 0.0)), emanuel_module.UPDRAFT_W_MAX_M_S)
    m_b = scheme.closure_c * float(rho_lcl) * w_buoy
    interval_s = (met.metadata.time_end - met.metadata.time_start).total_seconds()
    fmass = compute_mass_flux_matrix(
        t_col, q_col, p_col, i_lcl=i_lcl, i_lnb=i_lnb, m_b=m_b * interval_s
    )
    return fmass, height_col


def test_uniform_box_gives_the_old_matrix_in_every_column() -> None:
    """Acceptance (A02): on meteorology that is uniform in the horizontal, every
    column's matrix equals the one the pre-A02 scheme built from the bbox-mean
    column. Checked bit for bit (float32 on CPU)."""

    met = _build_convective_met_window()
    scheme = EmanuelReducedConvection(closure_c=0.1, trigger_dtv_k=0.1)
    cols = scheme.column_state(met, t_alpha=0.5, device=torch.device("cpu"), dtype=torch.float32)
    fmass_cols = scheme.column_mass_flux_matrices(cols)
    fmass_old, height_old = _old_bbox_mean_matrix(scheme, met, t_alpha=0.5)

    n_col = 3 * 3
    assert fmass_cols.shape == (n_col, *fmass_old.shape)
    assert bool(cols.fires.all())
    assert float(fmass_old.abs().sum()) > 0.0
    for c in range(n_col):
        assert torch.equal(fmass_cols[c], fmass_old), f"column {c} differs from the old matrix"
        # The old scheme averaged nine equal float32 heights, which rounds in
        # the last place; the column keeps its own.
        assert torch.allclose(cols.height_m[c], height_old, rtol=1e-6, atol=0.0)


def _mixed_column_window() -> HourlyMetTensors:
    """A 3 x 3 box whose columns differ: the convective sounding warmed or cooled
    at the surface and moistened or dried by column, so LCL, LNB and M_b vary."""

    met = _build_convective_met_window()
    n_lev, n_lat, n_lon = met.hour_start.shape[1:]
    t_idx, q_idx = met.channel_names.index("t"), met.channel_names.index("q")
    fields = met.hour_start.clone()
    dt = torch.tensor([[-1.0, 0.0, 1.5], [2.5, 0.5, -0.5], [1.0, -1.5, 0.25]])
    dq = torch.tensor([[0.9, 1.0, 1.1], [1.15, 0.95, 1.05], [1.0, 0.85, 1.2]])
    decay = torch.exp(-torch.arange(n_lev, dtype=torch.float32) / 3.0).view(n_lev, 1, 1)
    fields[t_idx] = fields[t_idx] + dt * decay
    fields[q_idx] = fields[q_idx] * (1.0 + (dq - 1.0) * decay)
    return HourlyMetTensors(
        hour_start=fields,
        hour_end=fields,
        metadata=met.metadata,
        channel_names=met.channel_names,
        height_agl_m=met.height_agl_m,
    )


def test_mass_is_conserved_in_every_column() -> None:
    """Acceptance (A02), the single-column tests extended to every column: each
    column's matrix is non-divergent, its backward transition rows are probability
    distributions, and its layer masses are stationary (mᵀP = mᵀ)."""

    met = _mixed_column_window()
    scheme = EmanuelReducedConvection(closure_c=0.1, trigger_dtv_k=0.1)
    cols = scheme.column_state(met, t_alpha=0.5, device=torch.device("cpu"), dtype=torch.float64)
    fmass = scheme.column_mass_flux_matrices(cols)
    assert int(cols.fires.sum()) >= 3, "most columns should fire"
    assert len({float(x) for x in cols.m_b[cols.fires]}) > 1, "columns should differ"

    m = _layer_masses_per_area(cols.pressure_pa).clamp(min=1e-6)
    prob = emanuel_module.backward_transition_matrices(fmass, m)
    for c in range(fmass.shape[0]):
        assert torch.allclose(fmass[c].sum(dim=1), fmass[c].sum(dim=0), rtol=1e-9, atol=1e-12)
        assert torch.allclose(prob[c].sum(dim=1), torch.ones_like(m[c]), atol=1e-12)
        assert bool((prob[c] >= 0.0).all())
        assert torch.allclose(m[c] @ prob[c], m[c], rtol=1e-9, atol=1e-12)


def test_matrices_match_the_single_column_reference_in_every_column() -> None:
    """The batched lift, trigger and matrix agree with the single-column reference
    functions column by column on a box whose columns all differ."""

    met = _mixed_column_window()
    scheme = EmanuelReducedConvection(closure_c=0.1, trigger_dtv_k=0.1)
    cols = scheme.column_state(met, t_alpha=0.5, device=torch.device("cpu"), dtype=torch.float64)
    fmass = scheme.column_mass_flux_matrices(cols)
    for c in range(fmass.shape[0]):
        i_lcl, i_lnb, cape, dtv = compute_lcl_lnb_cape(cols.t[c], cols.q[c], cols.pressure_pa[c])
        assert int(i_lcl) == int(cols.i_lcl[c])
        assert int(i_lnb) == int(cols.i_lnb[c])
        assert float(cape) == pytest.approx(float(cols.cape[c]), rel=1e-12, abs=1e-9)
        assert float(dtv) == pytest.approx(float(cols.dtv_lcl_plus_1[c]), rel=1e-12)
        if bool(cols.fires[c]):
            ref = compute_mass_flux_matrix(
                cols.t[c],
                cols.q[c],
                cols.pressure_pa[c],
                i_lcl=int(i_lcl),
                i_lnb=int(i_lnb),
                m_b=float(cols.m_b[c]) * cols.interval_s,
            )
            assert torch.allclose(fmass[c], ref, rtol=1e-12, atol=0.0)
        else:
            assert torch.all(fmass[c] == 0.0)


def _two_column_window() -> HourlyMetTensors:
    """1 x 2 box: the convective sounding in the western column, the dry stable
    sounding (same levels) in the eastern one."""

    from datetime import datetime, timedelta

    from lpdm.met_reader import MetFieldMetadata

    convective = _build_convective_met_window()
    n_lev = convective.hour_start.shape[1]
    t_unstable, q_unstable, p_col = _moist_unstable_column(n_lev=n_lev)
    p_hpa = p_col.numpy() / 100.0
    z_km = -8.0 * np.log(p_hpa / 1000.0)
    t_stable = torch.as_tensor(np.minimum(280.0 + 2.0 * z_km, 290.0), dtype=torch.float32)
    q_stable = torch.full((n_lev,), 1e-5)

    names = ("u", "v", "w", "blh", "sp", "t", "q")
    shape = (n_lev, 1, 2)
    t3 = torch.stack((t_unstable.to(torch.float32), t_stable), dim=1).view(shape)
    q3 = torch.stack((q_unstable.to(torch.float32), q_stable), dim=1).view(shape)
    chans = {
        "u": torch.zeros(shape),
        "v": torch.zeros(shape),
        "w": torch.zeros(shape),
        "blh": torch.full(shape, 1500.0),
        "sp": torch.full(shape, float(p_col[0])),
        "t": t3,
        "q": q3,
    }
    fields = torch.stack([chans[n] for n in names], dim=0)
    height = convective.metadata.level
    t0 = datetime(2024, 7, 1, 12, 0, 0, tzinfo=UTC)
    metadata = MetFieldMetadata(
        lon=np.array([0.0, 1.0]),
        lat=np.array([45.0]),
        level=height,
        pressure_level_hpa=p_hpa,
        time_start=t0,
        time_end=t0 + timedelta(hours=1),
        variable_units=dict.fromkeys(names, ""),
    )
    height_3d = torch.as_tensor(height, dtype=torch.float32).view(n_lev, 1, 1).expand(shape)
    return HourlyMetTensors(
        hour_start=fields,
        hour_end=fields,
        metadata=metadata,
        channel_names=names,
        height_agl_m=height_3d.contiguous(),
    )


def test_convection_fires_in_the_unstable_column_only() -> None:
    """Acceptance (A02): two columns, one unstable and one stable. Convection
    fires in the first only, and only particles in the first move."""

    from lpdm.gpu_engine import GPUEngine

    torch.manual_seed(20261124)
    met = _two_column_window()
    scheme = EmanuelReducedConvection(closure_c=0.1, trigger_dtv_k=0.1)
    cols = scheme.column_state(met, t_alpha=0.5, device=torch.device("cpu"), dtype=torch.float32)
    assert cols.fires.tolist() == [True, False]
    fmass = scheme.column_mass_flux_matrices(cols)
    assert float(fmass[0].abs().sum()) > 0.0
    assert torch.all(fmass[1] == 0.0)

    # Particles in the cloud layer of each column; the eastern ones sit just
    # west of the eastern grid point, still nearer to it than to the western one.
    i_lcl, i_lnb = int(cols.i_lcl[0]), int(cols.i_lnb[0])
    z_base, z_top = float(cols.height_m[0, i_lcl]), float(cols.height_m[0, i_lnb])
    n = 2000
    particles = torch.zeros(n, 4)
    particles[: n // 2, 0] = 0.3  # western (unstable) column
    particles[n // 2 :, 0] = 0.7  # eastern (stable) column
    particles[:, 1] = 45.0
    particles[:, 2] = torch.empty(n).uniform_(z_base, z_top)
    particles[:, 3] = 1.0 / n

    out, _ = scheme.maybe_convect(
        particles,
        scheme.initialize_state(n, device=torch.device("cpu"), dtype=torch.float32),
        met,
        t_alpha=0.5,
        dt_seconds=300.0,
        active_mask=torch.ones(n, dtype=torch.bool),
        engine=GPUEngine(device="cpu"),
    )
    moved = out[:, 2] != particles[:, 2]
    assert int(moved[: n // 2].sum()) > 100, "particles in the unstable column must move"
    assert not bool(moved[n // 2 :].any()), "no particle in the stable column may move"
    assert torch.equal(out[:, [0, 1, 3]], particles[:, [0, 1, 3]])
    # Backward: the unstable column's particles are traced down towards the surface.
    assert float((out[: n // 2, 2] - particles[: n // 2, 2]).mean()) < -200.0


def _wide_firing_box(n_lat: int = 6, n_lon: int = 10, seed: int = 7) -> HourlyMetTensors:
    """An n_lat x n_lon box of the convective sounding, each column warmed or
    cooled and moistened or dried near the surface by its own random amount, so
    most columns fire, with different LCL, LNB and M_b."""

    from dataclasses import replace

    base = _build_convective_met_window()
    n_ch, n_lev = base.hour_start.shape[:2]
    gen = torch.Generator().manual_seed(seed)
    column = base.hour_start[:, :, :1, :1]
    fields = column.expand(n_ch, n_lev, n_lat, n_lon).clone()
    t_idx, q_idx = base.channel_names.index("t"), base.channel_names.index("q")
    decay = torch.exp(-torch.arange(n_lev, dtype=torch.float32) / 3.0).view(n_lev, 1, 1)
    dt = torch.empty(n_lat, n_lon).uniform_(-0.5, 2.5, generator=gen)
    dq = torch.empty(n_lat, n_lon).uniform_(0.95, 1.2, generator=gen)
    fields[t_idx] = fields[t_idx] + dt * decay
    fields[q_idx] = fields[q_idx] * (1.0 + (dq - 1.0) * decay)
    metadata = replace(
        base.metadata,
        lon=np.linspace(0.0, n_lon - 1.0, n_lon),
        lat=np.linspace(40.0, 40.0 + n_lat - 1.0, n_lat),
    )
    height = base.height_agl_m[:, :1, :1].expand(n_lev, n_lat, n_lon).contiguous()
    return HourlyMetTensors(
        hour_start=fields,
        hour_end=fields,
        metadata=metadata,
        channel_names=base.channel_names,
        height_agl_m=height,
    )


def _convect_with_every_firing_column(scheme, particles, met, active, generator):
    """The step as first written in A02: a matrix for every firing column, then
    each particle's column looked up. The reference for the occupied-column path."""

    dtype = torch.float32
    cols = scheme.column_state(met, t_alpha=0.5, device=torch.device("cpu"), dtype=dtype)
    fire_idx = cols.fires.nonzero().squeeze(1)
    fmass = emanuel_module.compute_mass_flux_matrices(
        cols.pressure_pa[fire_idx],
        cols.i_lcl[fire_idx],
        cols.i_lnb[fire_idx],
        (cols.m_b[fire_idx] * cols.interval_s).to(dtype),
    )
    layer_mass = _layer_masses_per_area(cols.pressure_pa[fire_idx]).clamp(min=1e-6)
    prob = emanuel_module.backward_transition_matrices(fmass, layer_mass)
    col_slot = torch.full((cols.fires.numel(),), -1, dtype=torch.long)
    col_slot[fire_idx] = torch.arange(int(fire_idx.numel()))
    particle_col = emanuel_module.nearest_column_index(
        particles[:, 0], particles[:, 1], met.metadata.lon, met.metadata.lat
    )
    particle_slot = col_slot[particle_col]
    out, _ = scheme._redistribute_particles(
        particles=particles,
        eligible=active & (particle_slot >= 0),
        particle_col=particle_col,
        particle_slot=particle_slot,
        height_col=cols.height_m,
        prob=prob,
        generator=generator,
        device=torch.device("cpu"),
        dtype=dtype,
    )
    return out, cols


def test_matrices_only_for_occupied_columns_give_identical_positions() -> None:
    """PR #54 review: matrices are built only for firing columns that hold an
    active particle. On a box where most columns fire but particles occupy few
    (and inactive particles sit in other firing columns), every particle ends at
    the same position, bit for bit, as when every firing column's matrix is built."""

    from lpdm.gpu_engine import GPUEngine

    met = _wide_firing_box()
    scheme = EmanuelReducedConvection(closure_c=0.1, trigger_dtv_k=0.1)
    n = 3000
    gen = torch.Generator().manual_seed(11)
    particles = torch.zeros(n, 4)
    # Three occupied columns among 60; inactive particles in four other columns.
    lon_occ = torch.tensor([1.0, 4.0, 8.0])[torch.randint(0, 3, (n,), generator=gen)]
    lat_occ = torch.tensor([41.0, 43.0, 45.0])[torch.randint(0, 3, (n,), generator=gen)]
    particles[:, 0] = lon_occ + torch.empty(n).uniform_(-0.4, 0.4, generator=gen)
    particles[:, 1] = lat_occ + torch.empty(n).uniform_(-0.4, 0.4, generator=gen)
    particles[:, 2] = torch.empty(n).uniform_(0.0, 9000.0, generator=gen)
    particles[:, 3] = 1.0 / n
    active = torch.ones(n, dtype=torch.bool)
    active[-200:] = False
    particles[-200:, 0] = torch.tensor([0.0, 2.0, 6.0, 9.0]).repeat(50)
    particles[-200:, 1] = 44.0

    ref, cols = _convect_with_every_firing_column(
        scheme, particles, met, active, torch.Generator().manual_seed(99)
    )
    n_fire = int(cols.fires.sum())
    assert n_fire >= 40, f"most of the 60 columns should fire, got {n_fire}"

    built: list[int] = []
    real = emanuel_module.compute_mass_flux_matrices

    def spy(pressure_pa, *args):
        built.append(int(pressure_pa.shape[0]))
        return real(pressure_pa, *args)

    emanuel_module.compute_mass_flux_matrices = spy
    try:
        out, _ = scheme.maybe_convect(
            particles,
            scheme.initialize_state(n, device=torch.device("cpu"), dtype=torch.float32),
            met,
            t_alpha=0.5,
            dt_seconds=300.0,
            active_mask=active,
            engine=GPUEngine(device="cpu"),
            generator=torch.Generator().manual_seed(99),
        )
    finally:
        emanuel_module.compute_mass_flux_matrices = real

    particle_col = emanuel_module.nearest_column_index(
        particles[:, 0], particles[:, 1], met.metadata.lon, met.metadata.lat
    )
    n_occupied_firing = int(cols.fires[particle_col[active].unique()].sum())
    assert built == [n_occupied_firing]
    assert 0 < n_occupied_firing <= 9 < n_fire
    assert int((out[:, 2] != particles[:, 2]).sum()) > 100, "convection must move particles"
    assert torch.equal(out, ref)
    assert torch.equal(out[~active], particles[~active])


def test_backward_transition_points_from_current_layer_to_source_layer() -> None:
    """The backward matrix is the upstream source distribution of the air now in a
    layer, not the forward destination distribution (2026-11-25 test review).

    Three layers with masses 2, 3, 4 and one unit of forward mass flux on each link
    of the cycle 0→1→2→0, so fmass[i, j] (mass from i to j) is 1 for (0,1), (1,2),
    (2,0). The retained mass closes each row: 1, 2, 3. After the event layer 0
    holds its own retained unit and the unit that came up from layer 2, so a
    particle in layer 0 came from 0 or 2 with probability 1/2 each; layer 1 holds
    1 unit from 0 and 2 of its own; layer 2 holds 1 unit from 1 and 3 of its own.
    The forward matrix (rows: where layer i's air goes) differs in six entries, so
    comparing all nine values distinguishes the two operators.
    """

    fmass = torch.tensor([[[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]], dtype=torch.float64)
    layer_mass = torch.tensor([[2.0, 3.0, 4.0]], dtype=torch.float64)
    prob = emanuel_module.backward_transition_matrices(fmass, layer_mass)

    # Rows: the particle's current layer; columns: the layer its air came from.
    expected = torch.tensor(
        [[[1 / 2, 0.0, 1 / 2], [1 / 3, 2 / 3, 0.0], [0.0, 1 / 4, 3 / 4]]], dtype=torch.float64
    )
    assert prob.shape == (1, 3, 3)
    torch.testing.assert_close(prob, expected, rtol=0.0, atol=1e-15)


def test_nearest_column_index_handles_descending_latitude() -> None:
    lon = np.array([-10.0, -9.75, -9.5])
    lat = np.array([54.0, 53.75])  # descending, as ERA5 stores it
    idx = emanuel_module.nearest_column_index(
        torch.tensor([-10.0, -9.6, -9.74, -20.0]),
        torch.tensor([54.0, 53.8, 53.7, 60.0]),
        lon,
        lat,
    )
    assert idx.tolist() == [0, 1 * 3 + 2, 1 * 3 + 1, 0]


def test_non_positive_met_window_is_refused_by_convection() -> None:
    """A zero-width window would silently zero the convective mass flux."""

    from lpdm.gpu_engine import GPUEngine

    met = _build_convective_met_window(window_hours=0.0)
    scheme = EmanuelReducedConvection(closure_c=0.1, trigger_dtv_k=0.1)
    n = 8
    particles = torch.zeros(n, 4, dtype=torch.float32)
    particles[:, 2] = 3000.0
    particles[:, 3] = 1.0 / n

    with pytest.raises(ValueError, match="non-positive duration"):
        scheme.maybe_convect(
            particles,
            scheme.initialize_state(n, device=torch.device("cpu"), dtype=torch.float32),
            met,
            t_alpha=0.5,
            dt_seconds=300.0,
            active_mask=torch.ones(n, dtype=torch.bool),
            engine=GPUEngine(device="cpu"),
        )
