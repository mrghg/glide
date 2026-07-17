"""T3 — Langevin diffusion limit vs Crank-Nicolson PDE (dev/TEST_REVIEW_2026-07-16.md).

For t >> T_L the Langevin model must converge to the diffusion equation
dc/dt = d/dz(K(z) dc/dz) with K = sigma_w^2 T_L. Well-mixed tests cannot see a
wrong K (uniform stays uniform under ANY K); only a quantitative profile-evolution
test can — this is the class that would have caught the 2026-07 v2-validation
near-surface bias (K 3-8x low in the lowest ~15 m inflating surface residence).

Design note (measured, not assumed): a mid-column release is INSENSITIVE to
near-surface K — quartering K below 50 m moved the solution by only L1 ~ 0.02,
below any usable tolerance. The release therefore sits INSIDE the low-K
near-surface layer (5-15 m), where that exchange is rate-limiting: there a merely
HALVED near-surface K moves the profile by L1 = 0.18-0.33, 7-12x the true-K
agreement (L1 <= 0.044), which the companion teeth test asserts.

Particles use the production primitives (OU update with the Thomson well-mixed
drift, vertical displacement, surface reflection) with K(z) = K0 + a*min(z, z_cap)
and constant T_L, dt/T_L = 0.1. Reference: conservative flux-form Crank-Nicolson
with zero-flux boundaries (mass-exact), scipy tridiagonal solve.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from scipy.linalg import solve_banded

from lpdm.gpu_engine import GPUEngine

K0 = 0.02  # m^2/s at the ground
A_KZ = 0.12  # dK/dz below the cap
Z_CAP = 200.0
H_TOP = 400.0
T_L = 10.0
DT = 1.0  # dt/T_L = 0.1
N_PART = 100_000
Z0_LO, Z0_HI = 5.0, 15.0  # release slab inside the low-K layer
CHECKPOINTS = (300.0, 900.0, 1800.0)
BIN_W = 20.0
PDE_DZ = 2.0


def _k_faces(z_faces: np.ndarray, near_surface_factor: float = 1.0) -> np.ndarray:
    k = K0 + A_KZ * np.minimum(z_faces, Z_CAP)
    k = k.copy()
    k[z_faces < 50.0] *= near_surface_factor
    return k


def _make_cn_stepper(k_faces: np.ndarray, n_nodes: int):
    """Conservative flux-form Crank-Nicolson step with zero-flux boundaries."""
    main = np.zeros(n_nodes)
    main[0] = k_faces[0]
    main[1:-1] = k_faces[:-1] + k_faces[1:]
    main[-1] = k_faces[-1]
    lam = DT / PDE_DZ**2
    ab = np.zeros((3, n_nodes))
    ab[0, 1:] = -0.5 * lam * k_faces
    ab[1, :] = 1.0 + 0.5 * lam * main
    ab[2, :-1] = -0.5 * lam * k_faces

    def step(c: np.ndarray) -> np.ndarray:
        rhs = c - 0.5 * lam * (
            main * c - np.r_[0.0, k_faces * c[:-1]] - np.r_[k_faces * c[1:], 0.0]
        )
        return solve_banded((1, 1), ab, rhs)

    return step


@pytest.fixture(scope="module")
def diffusion() -> dict:
    """One Langevin evolution + three CN references (true / half / quarter
    near-surface K), binned at the checkpoints."""
    torch.manual_seed(3301)
    engine = GPUEngine(device="cpu", dtype=torch.float64)

    def sigma2(z: torch.Tensor) -> torch.Tensor:
        return (K0 + A_KZ * torch.clamp(z, max=Z_CAP)) / T_L

    dsig2_below = torch.tensor(A_KZ / T_L, dtype=torch.float64)
    zero = torch.tensor(0.0, dtype=torch.float64)

    z = torch.rand(N_PART, dtype=torch.float64) * (Z0_HI - Z0_LO) + Z0_LO
    w = torch.randn(N_PART, dtype=torch.float64) * torch.sqrt(sigma2(z))
    particles = torch.zeros((N_PART, 4), dtype=torch.float64)
    particles[:, 2] = z
    particles[:, 3] = 1.0 / N_PART

    z_nodes = np.arange(0.0, H_TOP + PDE_DZ / 2, PDE_DZ)
    z_faces = (z_nodes[:-1] + z_nodes[1:]) / 2
    steppers = {
        name: _make_cn_stepper(_k_faces(z_faces, f), len(z_nodes))
        for name, f in [("true", 1.0), ("half", 0.5), ("quarter", 0.25)]
    }
    c0 = np.where((z_nodes >= Z0_LO) & (z_nodes <= Z0_HI), 1.0, 0.0)
    c0 /= c0.sum()
    pde = {name: c0.copy() for name in steppers}

    edges = np.arange(0.0, H_TOP + BIN_W / 2, BIN_W)
    node_bin = np.minimum((z_nodes // BIN_W).astype(int), len(edges) - 2)

    def bin_pde(c: np.ndarray) -> np.ndarray:
        h = np.bincount(node_bin, weights=c, minlength=len(edges) - 1)
        return h / h.sum()

    out: dict[float, dict] = {}
    t = 0.0
    for t_check in CHECKPOINTS:
        for _ in range(int(round((t_check - t) / DT))):
            zc = particles[:, 2]
            s2 = sigma2(zc)
            # Thomson well-mixed drift; dsigma^2/dz vanishes above the cap.
            drift = 0.5 * (1.0 + w * w / s2) * torch.where(zc < Z_CAP, dsig2_below, zero)
            w = engine.update_langevin_velocity(
                w, t_lagrangian=T_L, sigma_w2=s2, dt_seconds=DT, drift=drift
            )
            particles = engine.apply_vertical_turbulence(particles, w, dt_seconds=DT, backward=False)
            particles, w = engine.reflect_surface(particles, w, z_surface=0.0)
            above = particles[:, 2] > H_TOP
            particles[:, 2] = torch.where(above, 2.0 * H_TOP - particles[:, 2], particles[:, 2])
            w = torch.where(above, -w, w)
            for name in pde:
                pde[name] = steppers[name](pde[name])
        t = t_check
        hist, _ = np.histogram(particles[:, 2].numpy(), bins=edges)
        out[t_check] = dict(
            particles=hist / hist.sum(),
            **{name: bin_pde(c) for name, c in pde.items()},
        )

    assert torch.allclose(particles[:, 3].sum(), torch.tensor(1.0, dtype=torch.float64))
    return out


def test_langevin_diffusion_limit_matches_pde(diffusion: dict) -> None:
    """Binned particle density tracks the true-K Crank-Nicolson solution at every
    checkpoint (obs L1 <= 0.044 vs 0.08 bound), including the near-surface
    occupancy that the v2 bias corrupted (obs <= 7.9% vs 12%)."""
    for t_check in CHECKPOINTS:
        got = diffusion[t_check]["particles"]
        want = diffusion[t_check]["true"]
        l1 = np.abs(got - want).sum()
        assert l1 < 0.08, f"t={t_check}s: L1(particles, PDE)={l1:.3f}"
        ns_got, ns_want = got[:3].sum(), want[:3].sum()  # lowest 3 bins = 0-60 m
        rel = abs(ns_got - ns_want) / ns_want
        assert rel < 0.12, (
            f"t={t_check}s: near-surface occupancy {ns_got:.3f} vs PDE {ns_want:.3f} (rel {rel:.3f})"
        )


def test_diffusion_pde_discriminates_near_surface_k_errors(diffusion: dict) -> None:
    """Teeth: the same particle solution is FAR from PDE references whose
    near-surface K is halved/quartered (the v2-bias class), so a wrong K in the
    model cannot slip under the tolerance above. Observed L1: half 0.18-0.33,
    quarter 0.46-0.66 — 7-15x the true-K agreement."""
    for t_check, min_half, min_quarter in [
        (300.0, 0.25, 0.45),
        (900.0, 0.20, 0.45),
        (1800.0, 0.12, 0.30),
    ]:
        got = diffusion[t_check]["particles"]
        l1_half = np.abs(got - diffusion[t_check]["half"]).sum()
        l1_quarter = np.abs(got - diffusion[t_check]["quarter"]).sum()
        assert l1_half > min_half, f"t={t_check}s: half-K L1 {l1_half:.3f} too close"
        assert l1_quarter > min_quarter, f"t={t_check}s: quarter-K L1 {l1_quarter:.3f} too close"
