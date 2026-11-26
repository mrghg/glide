"""The drift gradient where the boundary-layer σ_w meets its floor (prototype,
task WM-sigma-floor; outputs/agents/WM/sigma_floor/SUMMARY.md, main checkout).

In the stable regime σ_w = 1.3 u*(1 − z/h) falls to SIGMA_MIN_M_S a little below
h (1.23 m in this column at the 1 mm/s floor). The production ±1 m difference
("fd") straddles the kink where the floor starts to bind and returns a drift
inside the floor layer, where σ_w is flat; within δ of h the stencil is cut at h
(decision 0028), and when the floor layer is thinner than δ that one-sided
difference over-states the gradient. "hybrid" uses the analytic derivative of
the floored profile for exactly those particles and the production stencil,
bit for bit, everywhere else."""

from __future__ import annotations

import pytest
import torch
from test_bl_top_well_mixed import REGIMES, _met

import lpdm.turbulence.hanna as H
from lpdm.gpu_engine import GPUEngine
from lpdm.turbulence import HannaScheme
from lpdm.turbulence.hanna import convective_velocity, obukhov_length


def _coeffs(regime: str, z: torch.Tensor, mode: str) -> tuple[torch.Tensor, ...]:
    blh, ustar, shf = REGIMES[regime]
    met = _met(blh, ustar, shf)
    eng = GPUEngine(device="cpu")
    s = HannaScheme(static_substeps=False)
    s.DRIFT_GRADIENT_MODE = mode
    n = z.numel()
    lon, lat = torch.zeros(n), torch.full((n,), 45.0)

    def surf(k):
        return s._interp_surface_field(met, k, 0.5, lon, lat, eng.device, eng.dtype)

    h = s._limit_blh(surf("blh"))
    sp = surf("sp")
    us = surf("ustar").clamp(min=H.USTAR_MIN_M_S)
    hf = surf("shf")
    ts = s._interp_t_at_lowest_level(met, 0.5, lon, lat, eng.device, eng.dtype)
    L = obukhov_length(us, hf, ts, sp)
    ws = convective_velocity(h, hf, ts, sp, blh_min_m=s.blh_min_m)
    ft, gb = s._free_trop_fields(met, eng.device, eng.dtype)
    ck = dict(blh=h, ustar=us, w_star=ws, h_over_L=h / L, L=L, lat=lat, lon=lon,
              ft_fields=ft, grid_bounds=gb, engine=eng)  # fmt: skip
    return s._vertical_coefficients(z, s._density_fields(met, eng.device, eng.dtype), **ck)


@pytest.mark.parametrize("floor", [1e-3, 1e-4])
def test_hybrid_gradient_is_the_derivative_of_the_floored_profile(monkeypatch, floor) -> None:
    monkeypatch.setattr(H, "SIGMA_MIN_M_S", floor)
    h, ustar, _ = REGIMES["stable"]
    depth = torch.tensor([6.0, 3.0, 2.0, 1.5, 1.3, 1.2, 1.0, 0.5, 0.2, 0.1, 0.05, 0.0])
    z = h - depth
    sigma_h = 1.3 * ustar * depth / h
    exact = torch.where(sigma_h > floor, -1.3 * ustar / h * sigma_h, torch.zeros_like(z))
    # The 8th coefficient is −½∂σ_w²/∂z (pre-negated for the backward drift).
    hybrid = -_coeffs("stable", z, "hybrid")[7]
    torch.testing.assert_close(hybrid, exact, rtol=1e-4, atol=1e-11)
    fd = -_coeffs("stable", z, "fd")[7]
    # The defect this replaces: a drift inside the floor layer (or, with a floor
    # layer thinner than δ, a too-large one just below it).
    assert float((fd - exact).abs().max()) > 1e-7


@pytest.mark.parametrize("regime", sorted(REGIMES))
def test_hybrid_gradient_is_the_production_stencil_away_from_h(regime: str) -> None:
    h = REGIMES[regime][0]
    # Stable: the floor binds above h - 1.23 m, so probes reach it from h - 2.23 m.
    z = torch.linspace(0.0, h - 2.5, 400)
    full_fd = _coeffs(regime, z, "fd")
    full_hy = _coeffs(regime, z, "hybrid")
    for a, b in zip(full_fd, full_hy, strict=True):
        assert torch.equal(a, b)
    above = torch.linspace(h + 0.01, 2 * h, 200)
    assert torch.equal(_coeffs(regime, above, "fd")[7], _coeffs(regime, above, "hybrid")[7])


def test_default_mode() -> None:
    assert HannaScheme.DRIFT_GRADIENT_MODE == "hybrid"
    assert HannaScheme.SIGMA_W_FLOOR_FORM == "clamp"
