"""The drift gradient where the boundary-layer σ_w meets its floor and at h
(decision 0030, amending decision 0028 item 1).

In the stable regime σ_w = 1.3 u*(1 − z/h) falls to SIGMA_MIN_M_S (1 mm s⁻¹) a
little below h: σ_min·h/(1.3 u*) below it, 0.68 m in the thin-floor column here
(h 400 m, u* 0.45 m s⁻¹). The ±1 m difference of ∂σ_w²/∂z had two defects there:
it straddled the kink of max(σ_H, σ_min) and put a drift inside the floor layer,
where σ_w is flat; and within 1 m of h decision 0028's stencil is cut at h, so
it is one-sided and returns the gradient up to 0.5 m below the particle, an O(1)
error where σ_H → 0 linearly. Production now takes the analytic derivative of
the floored profile for exactly those boundary-layer particles (stencil cut at h
or touching the floor) and the stencil, bit for bit, everywhere else.

Checks: the analytic gradient against fine differences in all three regimes; bit
identity with the stencil away from h and the floor; static and dynamic step
paths bit-identical; and a transfer-symmetry gate across a stable h.
"""

from __future__ import annotations

import itertools

import pytest
import torch
from test_bl_top_well_mixed import _met

import lpdm.turbulence.hanna as H
from lpdm.gpu_engine import GPUEngine
from lpdm.turbulence import HannaScheme
from lpdm.turbulence.hanna import (
    convective_velocity,
    in_bl_dsigw2_dz,
    in_bl_sigma_w,
    obukhov_length,
)

# Columns (h m, u* m s⁻¹, H W m⁻²): every regime, thick and thin floor layers.
COLUMNS = {
    "stable_thick_floor": (400.0, 0.25, -20.0),  # floor layer 1.23 m (the gates' column)
    "stable_thin_floor": (400.0, 0.45, -60.0),  # floor layer 0.68 m (the gate below)
    "stable_shallow": (100.0, 0.10, -10.0),  # 0.77 m
    "stable_weak": (1000.0, 0.05, -2.0),  # 15.4 m
    "neutral": (1500.0, 0.4, 0.0),
    "neutral_floor": (1500.0, 0.03, 0.0),  # σ_w reaches the floor at about 530 m
    "convective": (1500.0, 0.3, 200.0),
    "convective_weak": (600.0, 0.05, 30.0),
}


def _col(scheme: HannaScheme, met, n: int) -> dict:
    """Column inputs as ``HannaScheme.step`` gathers them (dynamic path)."""

    eng = GPUEngine(device="cpu")
    lon, lat = torch.zeros(n), torch.full((n,), 45.0)

    def surf(k):
        return scheme._interp_surface_field(met, k, 0.5, lon, lat, eng.device, eng.dtype)

    h = scheme._limit_blh(surf("blh"))
    sp = surf("sp")
    us = surf("ustar").clamp(min=H.USTAR_MIN_M_S)
    ts = scheme._interp_t_at_lowest_level(met, 0.5, lon, lat, eng.device, eng.dtype)
    L = obukhov_length(us, surf("shf"), ts, sp)
    ws = convective_velocity(h, surf("shf"), ts, sp, blh_min_m=scheme.blh_min_m)
    ft, gb = scheme._free_trop_fields(met, eng.device, eng.dtype)
    return dict(blh=h, ustar=us, w_star=ws, h_over_L=h / L, L=L, lat=lat, lon=lon,
                ft_fields=ft, grid_bounds=gb, engine=eng)  # fmt: skip


def _dsig2_dz(scheme: HannaScheme, met, z: torch.Tensor) -> torch.Tensor:
    col = _col(scheme, met, z.numel())
    density = scheme._density_fields(met, torch.device("cpu"), torch.float32)
    return -2.0 * scheme._vertical_coefficients(z, density, **col)[7]


def _fd_only() -> HannaScheme:
    scheme = HannaScheme(static_substeps=False)
    scheme._fd_drift_gradient_only = True
    return scheme


def _profile_sq(z: torch.Tensor, args: tuple, floored: bool) -> torch.Tensor:
    """σ_w² of the boundary-layer profile in float64, floored or not. The unstable
    branch is floored inside ``in_bl_sigma_w``; with the floor at ~0 it is not."""

    if floored:
        return in_bl_sigma_w(z, *args).pow(2).clamp(min=H.SIGMA_MIN_M_S**2)
    floor = H.SIGMA_MIN_M_S
    H.SIGMA_MIN_M_S = 1e-30
    try:
        return in_bl_sigma_w(z, *args).pow(2)
    finally:
        H.SIGMA_MIN_M_S = floor


def _away_from_h_and_floor(z: torch.Tensor, col: dict, margin: float = 1e-3) -> torch.Tensor:
    """Boundary-layer heights whose ±1 m stencil is neither cut at h nor touches the
    floor, with ``margin`` (m) to spare: there production must be the stencil."""

    args = tuple(col[k] for k in ("blh", "ustar", "w_star", "h_over_L", "lat"))
    reach = 1.0 + margin
    ok = z + reach < col["blh"]
    for zz in (z - reach, z, z + reach):
        ok &= in_bl_sigma_w(zz.clamp(min=0.0), *args) > H.SIGMA_MIN_M_S * (1.0 + 1e-3)
    return ok


# ---------------------------------------------------------------------------
# The analytic gradient
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("floor", [1e-4, 1e-3, 1e-2])
def test_analytic_gradient_matches_fine_differences_in_every_regime(monkeypatch, floor) -> None:
    """``in_bl_dsigw2_dz`` in float64 against a one-sided 1 mm difference of the
    floored profile max(σ_H, σ_min)² taken on the particle's side of the kink (the
    derivative of the floored profile: zero where the floor binds), from z_ubl to
    h, over a grid of h and u* (and w*, latitude) in each regime."""

    monkeypatch.setattr(H, "SIGMA_MIN_M_S", floor)
    d = torch.float64
    eps = 1e-3
    cases = (
        [("stable", h, us, 0.0, 5.0, 45.0)
         for h, us in itertools.product([50.0, 100.0, 400.0, 1500.0], [0.01, 0.05, 0.25, 0.6])]
        + [("neutral", h, us, 0.0, 0.0, lat)
           for h, us, lat in itertools.product([200.0, 800.0, 2000.0], [0.005, 0.04, 0.3, 0.8],
                                               [5.0, 45.0, -60.0])]
        + [("convective", h, us, ws, -10.0, 45.0)
           for h, us, ws in itertools.product([300.0, 1500.0, 3000.0], [0.001, 0.1, 0.4],
                                              [0.05, 0.5, 2.5])]
    )  # fmt: skip
    floored_somewhere = dict.fromkeys(("stable", "neutral", "convective"), 0)
    for regime, h, us, ws, hl, lat in cases:
        z = torch.cat(
            [torch.linspace(2.0, h, 2000, dtype=d), h - torch.logspace(-2.5, 1.0, 200, dtype=d)]
        )
        z = z[(z >= 2.0) & (z <= h)]
        args = tuple(torch.full_like(z, v) for v in (h, us, ws, hl, lat))
        g = in_bl_dsigw2_dz(z, *args)
        free = _profile_sq(z, args, floored=False) > floor**2
        floored_somewhere[regime] += int((~free).any())

        def prof(zz, args=args):
            return _profile_sq(zz, args, floored=True)

        # Difference within the particle's own branch (above or at the floor).
        up_same = (_profile_sq(z + eps, args, floored=False) > floor**2) == free
        z_up = torch.minimum(z + eps, torch.full_like(z, h))
        forward = (prof(z_up) - prof(z)) / (z_up - z).clamp(min=1e-12)
        backward = (prof(z) - prof(z - eps)) / eps
        fd = torch.where(up_same & (z_up > z), forward, backward)
        scale = float(g.abs().max())
        if scale == 0.0:  # the floor binds everywhere in this column
            assert not bool(free.any())
            continue
        err = float((g - fd).abs().max()) / scale
        assert err < 1e-3, (regime, h, us, ws, lat, err)
        assert torch.all(g[~free] == 0.0)
    # Every regime's grid reaches the floor somewhere at the 1e-2 floor, and
    # the stable and neutral ones at every floor.
    assert floored_somewhere["stable"] > 0 and floored_somewhere["neutral"] > 0
    if floor >= 1e-2:
        assert floored_somewhere["convective"] > 0


@pytest.mark.parametrize("name", sorted(COLUMNS))
@pytest.mark.parametrize("floor", [1e-3, 1e-4])
def test_production_gradient_near_h_is_the_derivative_of_the_floored_profile(
    monkeypatch, name: str, floor: float
) -> None:
    """At scheme level, float32: the production ∂σ_w²/∂z (the 8th coefficient)
    equals the float64 analytic derivative of the floored profile wherever it
    departs from the ±1 m stencil (within 1 m of h, or where a probe touches the
    floor), and the stencil is what it was elsewhere, bit for bit."""

    monkeypatch.setattr(H, "SIGMA_MIN_M_S", floor)
    blh, ustar, shf = COLUMNS[name]
    met = _met(blh, ustar, shf)
    z = torch.cat([torch.linspace(0.0, blh, 4001), blh - torch.logspace(-3.0, 0.7, 300)])
    z = z[(z >= 0.0) & (z <= blh)]
    prod = _dsig2_dz(HannaScheme(static_substeps=False), met, z)
    fd = _dsig2_dz(_fd_only(), met, z)

    col = _col(HannaScheme(), met, z.numel())
    args = tuple(col[k].double() for k in ("blh", "ustar", "w_star", "h_over_L", "lat"))
    z_eval = z.double().clamp(min=HannaScheme().z_ubl_m)
    exact = in_bl_dsigw2_dz(z_eval, *args)
    scale = float(exact.abs().max())

    changed = prod != fd
    assert bool(changed.any())  # every column has particles within 1 m of h
    # The analytic gradient is used within 1 m of h and at the floor only.
    assert not bool((changed & _away_from_h_and_floor(z, col)).any())
    err = float((prod.double() - exact)[changed].abs().max()) / scale
    assert err < 1e-4, err


@pytest.mark.parametrize("floor", [1e-3, 1e-4])
def test_stencil_defects_at_a_stable_top(monkeypatch, floor: float) -> None:
    """What decision 0030 removes, in the thin-floor column (floor layer 0.68 m at
    1 mm s⁻¹, 0.07 m at 0.1 mm s⁻¹): the stencil puts a drift inside the floor
    layer, where σ_w is flat, and mis-states it just below (cut at h, one-sided,
    or straddling the kink). Production has the exact derivative,
    −2·1.3u*·σ_H/h above the floor and zero in it."""

    monkeypatch.setattr(H, "SIGMA_MIN_M_S", floor)
    blh, ustar, shf = COLUMNS["stable_thin_floor"]
    met = _met(blh, ustar, shf)
    z = blh - torch.tensor([6.0, 3.0, 2.0, 1.5, 1.2, 1.0, 0.8, 0.6, 0.5, 0.3, 0.1, 0.05, 0.02, 0.0])
    sigma_h = 1.3 * ustar * (blh - z.double()) / blh
    in_floor = sigma_h <= floor
    exact = torch.where(in_floor, torch.zeros_like(sigma_h), -2.0 * 1.3 * ustar / blh * sigma_h)
    prod = _dsig2_dz(HannaScheme(static_substeps=False), met, z)
    # rtol: float32 rounding of 1 − z/h within 0.1 m of h
    torch.testing.assert_close(prod.double(), exact, rtol=1e-3, atol=1e-12)
    assert bool(in_floor.any()) and torch.all(prod[in_floor] == 0.0)
    fd = _dsig2_dz(_fd_only(), met, z).double()
    scale = float(exact.abs().max())
    assert float(fd[in_floor].abs().max()) > 0.03 * scale  # a drift where σ_w is flat
    if floor < 1e-3:
        # Floor layer much thinner than δ: the one-sided stencil cut at h returns
        # the gradient up to 0.5 m lower down, several times too large.
        near = (blh - z < 1.0) & ~in_floor
        assert float((fd[near] / exact[near]).max()) > 3.0


@pytest.mark.parametrize("name", sorted(COLUMNS))
def test_production_is_the_stencil_away_from_h_and_the_floor(name: str) -> None:
    """Every coefficient, bit for bit, at every boundary-layer height whose ±1 m
    stencil is neither cut at h nor touches the floor, and the gradient above h."""

    blh, ustar, shf = COLUMNS[name]
    met = _met(blh, ustar, shf)
    z = torch.linspace(0.0, blh, 3000)
    z = z[_away_from_h_and_floor(z, _col(HannaScheme(), met, z.numel()))]
    assert z.numel() > 1000
    density = HannaScheme()._density_fields(met, torch.device("cpu"), torch.float32)
    col = _col(HannaScheme(), met, z.numel())
    a = HannaScheme(static_substeps=False)._vertical_coefficients(z, density, **col)
    b = _fd_only()._vertical_coefficients(z, density, **col)
    for x, y in zip(a, b, strict=True):
        assert torch.equal(x, y)
    above = torch.linspace(blh + 1e-2, 2.0 * blh, 500)
    assert torch.equal(
        _dsig2_dz(HannaScheme(static_substeps=False), met, above), _dsig2_dz(_fd_only(), met, above)
    )


def test_surface_layer_override_keeps_the_stencil() -> None:
    """With the opt-in surface-layer override the analytic form (which does not
    include the override's profile) is not used: the stencil everywhere."""

    blh, ustar, shf = COLUMNS["stable_thin_floor"]
    met = _met(blh, ustar, shf)
    z = torch.cat([torch.linspace(0.0, blh, 801), blh - torch.logspace(-3.0, 0.5, 50)])
    a = HannaScheme(static_substeps=False, surface_layer_override=True)
    b = HannaScheme(static_substeps=False, surface_layer_override=True)
    b._fd_drift_gradient_only = True
    assert torch.equal(_dsig2_dz(a, met, z), _dsig2_dz(b, met, z))


# ---------------------------------------------------------------------------
# Static and dynamic paths
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["stable_thick_floor", "stable_thin_floor"])
def test_static_and_dynamic_steps_bit_identical_near_a_stable_top(name: str) -> None:
    """Decision 0004: with every particle active and no mean wind, the static
    path (the CUDA-graph code) and the dynamic path give bit-identical particles,
    velocities and exceedance counts, with half the particles starting within 5 m
    of a stable h (so the analytic branch is exercised) and the rest in [0, 2h]."""

    blh, ustar, shf = COLUMNS[name]
    met = _met(blh, ustar, shf)
    engine = GPUEngine(device="cpu")
    results = []
    for static in (False, True):
        torch.manual_seed(5)
        scheme = HannaScheme(n_substeps=16, static_substeps=static)
        n = 2000
        p = torch.zeros(n, 4)
        p[:, 1] = 45.0
        p[: n // 2, 2] = blh - 5.0 * torch.rand(n // 2)
        p[n // 2 :, 2] = 2.0 * blh * torch.rand(n - n // 2)
        p[:, 3] = 1.0
        state = scheme.initialize_state(n, device=torch.device("cpu"), dtype=torch.float32)
        active = torch.ones(n, dtype=torch.bool)
        for _ in range(5):
            p, state = scheme.step(
                p, state, met, t_alpha=0.5, dt_seconds=60.0, active_mask=active, engine=engine
            )
        results.append((p, state, scheme.take_substep_exceedance(None)))
    (pd, sd, xd), (ps, ss, xs) = results
    assert torch.equal(pd, ps)
    for key in ("u_prime", "v_prime", "w_prime", "sigma_last"):
        assert torch.equal(sd[key], ss[key])
    assert xd == xs
    assert int(((pd[:, 2] > blh - 1.0) & (pd[:, 2] <= blh)).sum()) > 0


# ---------------------------------------------------------------------------
# Transfer symmetry across a stable boundary-layer top
# ---------------------------------------------------------------------------

ROW_M = 40.0  # rows h − 160 … h (four, in the boundary layer) and h … h + 40
GATE_HOURS = 2.0


def transfer_ratio(
    scheme: HannaScheme, n_per_row: int, seed: int, column: str = "stable_thin_floor"
) -> tuple[float, float]:
    """Downward/upward exchange ratio across h in a steady, motionless column.

    Particles start uniform in each of five 40 m rows (four below h, one above)
    with velocities drawn from the Eulerian Gaussian, and run backward through
    the production step for 2 h at 60 s. With M[a, b] the mean time a particle
    started in row a spends in row b, the ratio is
    Σ_b M[above, b] / Σ_b M[b, above] over the four boundary-layer rows b. In a
    steady column the chain is reversible about the well-mixed state (uniform in
    z here: the air density varies by under 0.1% over these rows), so M is
    symmetric and the ratio is 1 up to noise and sub-step error. Returns the
    ratio and its standard error (independent particles, delta method)."""

    blh, ustar, shf = COLUMNS[column]
    met = _met(blh, ustar, shf)
    engine = GPUEngine(device="cpu")
    gen = torch.Generator().manual_seed(7000 + seed)
    n_rows = 5
    lows = torch.tensor([blh - 4 * ROW_M, blh - 3 * ROW_M, blh - 2 * ROW_M, blh - ROW_M, blh])
    start_row = torch.arange(n_rows).repeat_interleave(n_per_row)
    n = start_row.numel()
    p = torch.zeros(n, 4)
    p[:, 1] = 45.0
    p[:, 2] = lows[start_row] + ROW_M * torch.rand(n, generator=gen)
    p[:, 3] = 1.0
    state = scheme.initialize_state(n, device=torch.device("cpu"), dtype=torch.float32)
    col = _col(scheme, met, n)
    sig = scheme._sigmas_at(p[:, 2], **col)
    state["u_prime"] = sig[:, 0] * torch.randn(n, generator=gen)
    state["v_prime"] = sig[:, 1] * torch.randn(n, generator=gen)
    state["w_prime"] = sig[:, 2] * torch.randn(n, generator=gen)
    state["sigma_last"] = sig.clone()
    active = torch.ones(n, dtype=torch.bool)
    torch.manual_seed(100 + seed)
    down = torch.zeros(n)  # per particle: steps spent in the four BL rows
    up = torch.zeros(n)  # per particle: steps spent in h … h + 40
    for _ in range(int(GATE_HOURS * 60)):
        p, state = scheme.step(
            p, state, met, t_alpha=0.5, dt_seconds=60.0, active_mask=active, engine=engine
        )
        z = p[:, 2]
        down += ((z >= blh - 4 * ROW_M) & (z < blh)).float()
        up += ((z >= blh) & (z < blh + ROW_M)).float()
    from_above = start_row == n_rows - 1
    num = down[from_above].double()
    den = up[~from_above].double()  # four rows of n_per_row: Σ_b M[b, above] = 4·mean
    r = float(num.mean() / (4.0 * den.mean()))
    rel = (
        float(num.var() / n_per_row) / float(num.mean()) ** 2
        + float(den.var() / den.numel()) / float(den.mean()) ** 2
    ) ** 0.5
    return r, r * rel


# Calibration (scripts/validation/sigma_floor/transfer_gate_calibration.py;
# outputs/agents/SF/gate/n16_p10000.jsonl), this procedure at the default 16
# sub-steps, 10,000 particles per row, seeds 0-9:
#   this code:                 1.03-1.27, mean 1.18, seed sd 0.08 (each seed's
#                              standard error 0.085);
#   the stencil only (main):   3.49-4.70, mean 3.80, seed sd 0.34.
# At 64 sub-steps (n64_p10000.jsonl, seeds 0-2) this code gives 1.04-1.10 and
# the stencil 3.02-3.32: this code's residual shrinks with the sub-step (first
# order, decision 0027), the stencil's does not. A tolerance of 0.5 is 3.0 seed sd above this
# code's worst seed and fails main's best by 2.0. About 30 s on 6 CPU threads.
GATE_N_PER_ROW = 10_000
GATE_TOLERANCE = 0.5  # on |ratio − 1|


def test_exchange_across_a_stable_top_is_symmetric() -> None:
    """The gate: downward/upward exchange across a stable h with a floor layer
    thinner than the stencil (0.68 m) is 1 within GATE_TOLERANCE at the default
    sub-step count. It was about 3.8 before decision 0030."""

    scheme = HannaScheme(meander_enabled=False, static_substeps=False)  # default 16 sub-steps
    r, se = transfer_ratio(scheme, GATE_N_PER_ROW, seed=0)
    assert abs(r - 1.0) < GATE_TOLERANCE, f"downward/upward exchange across h {r:.3f} ± {se:.3f}"
