"""Boundary-layer-top well-mixed gate at the shipped 60 s outer step (decision 0028).

A horizontally uniform, motionless column with an open boundary-layer top: the
boundary-layer profile below h, the free-troposphere closure above it (stably
stratified, N = 0.015 s⁻¹, no wind, so K_z is the closure's 0.1 m² s⁻¹ floor),
and a reflecting lid at 2h applied inside every sub-step. Particles start
uniform in [0, 2h]. A well-mixed-consistent model keeps them uniform, including
just below and just above h where σ_w jumps between the two profiles (the air
density varies by ~3% over [h, 2h], but its drift moves particles by less than
1 m in the test's 6 h, so the answer is uniform). Three regimes, as the
near-surface gate (tests/test_near_surface_well_mixed.py): stable (h 400 m),
neutral (h 1,500 m, u* 0.4 m s⁻¹), convective (h 1,500 m, H +200 W m⁻²).

Two checks, both from a 6-h run (records every 15 min from 2 h): every one of
the four 50 m bands from h - 100 to h + 100 m holds its uniform share to within
TOLERANCE, and the occupancy is continuous across h (the 50 m above minus the
50 m below, within JUMP_TOLERANCE). A well-mixed-consistent interface keeps the
concentration continuous across h whatever the integration error elsewhere
(decision 0028), so the second check isolates the interface from the
first-order error of the stable boundary layer near its top, which the first
check also sees.

Before decision 0028 the drift's ±1 m finite difference straddled h, and a
particle crossing h kept its w'. Calibration with this test's procedure
(bl_top_gate_calibration.py; outputs/agents/WM/interface/gate/gate.jsonl):

- this branch, default 16 sub-steps, 6 seeds: worst band deviation 0.056
  (stable; neutral 0.038, convective 0.044), worst jump 0.043; at 8 sub-steps
  0.067 and 0.080;
- PR #69 head 0de5f94, 16 sub-steps, 3 seeds: jump +0.11 to +0.17 (stable),
  -0.76 to -0.85 (neutral), +5.1 (convective); worst band 0.07-0.09, 0.51-0.57
  and 5.0;
- main 20db902, 3 seeds: jump -0.14 to -0.19, +4.0, +9.7; worst band 0.48-0.50,
  3.4-3.6, 9.0.

Both tolerances, 0.08, leave this branch's worst seed a margin of 0.024 and
0.037 and fail PR #69's head and main in every regime (by the jump in the
stable one).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest
import torch

from lpdm.gpu_engine import GPUEngine
from lpdm.met_reader import HourlyMetTensors, MetFieldMetadata
from lpdm.turbulence import HannaScheme

REGIMES = {
    # name: (h m, u* m/s, sensible heat flux W m-2)
    "stable": (400.0, 0.25, -20.0),
    "neutral": (1500.0, 0.4, 0.0),
    "convective": (1500.0, 0.3, 200.0),
}
FT_N = 0.015  # free-troposphere buoyancy frequency, s-1
GRAVITY = 9.80665
TOLERANCE = 0.08  # on each 50 m band's occupancy relative to uniform
JUMP_TOLERANCE = 0.08  # on (h..h+50) - (h-50..h)


def _met(blh_m: float, ustar: float, shf: float) -> HourlyMetTensors:
    """Uniform, motionless window; θ rises with N = FT_N above h (41 levels)."""
    n_lev, n_lat, n_lon = 41, 4, 4
    shape = (n_lev, n_lat, n_lon)
    level = np.linspace(0.0, max(4000.0, 2.5 * blh_m), n_lev)
    dtheta_dz = FT_N**2 * 280.0 / GRAVITY
    above = torch.as_tensor(np.clip(level - blh_m, 0.0, None), dtype=torch.float32)
    zeros = torch.zeros(shape, dtype=torch.float32)
    chans = {
        "u": zeros,
        "v": zeros,
        "w": zeros,
        "blh": torch.full(shape, blh_m),
        "sp": torch.full(shape, 101325.0),
        "t": torch.full(shape, 280.0) + (dtheta_dz * above).view(n_lev, 1, 1),
        "ustar": torch.full(shape, ustar),
        "shf": torch.full(shape, shf),
    }
    names = tuple(chans)
    fields = torch.stack([chans[k] for k in names], dim=0)
    t0 = datetime(2024, 1, 1, 12, tzinfo=UTC)
    md = MetFieldMetadata(
        lon=np.linspace(-2.0, 2.0, n_lon),
        lat=np.linspace(40.0, 50.0, n_lat),
        level=level,
        pressure_level_hpa=np.linspace(1000.0, 999.0, n_lev),
        time_start=t0,
        time_end=t0 + timedelta(hours=1),
        variable_units=dict.fromkeys(names, "1"),
    )
    height = torch.as_tensor(level, dtype=torch.float32).view(n_lev, 1, 1).expand(shape)
    return HourlyMetTensors(
        hour_start=fields,
        hour_end=fields,
        metadata=md,
        channel_names=names,
        height_agl_m=height.contiguous(),
    )


class _LidEngine(GPUEngine):
    """Ground reflection plus a reflecting lid at ``lid_m``, in every sub-step."""

    def __init__(self, lid_m: float, **kw) -> None:
        super().__init__(**kw)
        self.lid_m = float(lid_m)

    def reflect_surface(self, particles, w_prime, *, z_surface: float = 0.0):
        p, w = super().reflect_surface(particles, w_prime, z_surface=z_surface)
        over = p[:, 2] > self.lid_m
        p = p.clone()
        p[:, 2] = torch.where(over, 2.0 * self.lid_m - p[:, 2], p[:, 2])
        return p, torch.where(over, -w, w)


def run_column(regime: str, scheme: HannaScheme, seed: int = 20261125) -> dict[str, float]:
    """Occupancy of the four 50 m bands around h, relative to uniform, averaged
    over records every 15 min from 2 h to 6 h. The start is stratified (one
    particle per equal slot of [0, 2h], jittered within it): above h particles
    move little in 6 h, so the binomial noise of an independent uniform start
    would stay frozen in the free-troposphere bands."""

    blh, ustar, shf = REGIMES[regime]
    torch.manual_seed(seed)
    engine = _LidEngine(2.0 * blh, device="cpu")
    met = _met(blh, ustar, shf)
    n = 20_000
    span = 2.0 * blh
    particles = torch.zeros(n, 4, dtype=torch.float32)
    particles[:, 1] = 45.0
    particles[:, 2] = (torch.arange(n, dtype=torch.float32) + torch.rand(n)) * (span / n)
    particles[:, 3] = 1.0 / n
    state = scheme.initialize_state(n, device=torch.device("cpu"), dtype=torch.float32)
    active = torch.ones(n, dtype=torch.bool)

    samples = []
    for step in range(1, 361):  # 6 h at 60 s; record every 15 min from 2 h
        particles, state = scheme.step(
            particles, state, met, t_alpha=0.5, dt_seconds=60.0, active_mask=active, engine=engine
        )
        if step >= 120 and step % 15 == 0:
            samples.append(particles[:, 2].clone())
    z = torch.stack(samples)
    assert float(z.max()) <= span and float(z.min()) >= 0.0

    def share(lo: float, hi: float) -> float:
        """Occupancy of [lo, hi) relative to its uniform share."""
        return float(((z >= lo) & (z < hi)).float().mean()) / ((hi - lo) / span)

    return {
        "h-100..h-50": share(blh - 100.0, blh - 50.0),
        "h-50..h": share(blh - 50.0, blh),
        "h..h+50": share(blh, blh + 50.0),
        "h+50..h+100": share(blh + 50.0, blh + 100.0),
    }


@pytest.mark.parametrize("regime", sorted(REGIMES))
def test_uniform_column_stays_uniform_across_the_boundary_layer_top(regime: str) -> None:
    scheme = HannaScheme(meander_enabled=False, static_substeps=False)  # default n_substeps
    bands = run_column(regime, scheme)
    bad = {k: round(v, 3) for k, v in bands.items() if abs(v - 1.0) >= TOLERANCE}
    assert not bad, f"{regime}: occupancy near h relative to uniform {bad} (all: {bands})"
    jump = bands["h..h+50"] - bands["h-50..h"]
    assert abs(jump) < JUMP_TOLERANCE, (
        f"{regime}: occupancy jumps by {jump:+.3f} across h (all: {bands})"
    )
