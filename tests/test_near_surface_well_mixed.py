"""Near-surface well-mixed gate at the shipped 60 s outer step (decision 0027).

A horizontally uniform, motionless boundary layer of uniform density, closed by
a reflecting lid at its top that is applied inside every turbulence sub-step
(an engine whose ``reflect_surface`` also reflects at h), must stay well mixed:
the lowest 5% of the column and the 0-40 m layer, where footprints are counted,
must each hold their uniform share of the particles. Three regimes, as B7's
``tier0._near_surface_error`` (PR #67): stable (h 400 m, H -20 W m⁻²), neutral
(h 1,500 m, u* 0.4 m s⁻¹) and convective (h 1,500 m, H +200 W m⁻²).

Before decision 0027 the coefficients were held at their outer-step values over
the sub-steps and each particle chose its own sub-step count from its T_Lw. On
this harness that code gives the lowest 5% 1.79 (stable) and 1.66 (neutral)
times its share, and the convective 0-40 m 0.90 (5 seeds of 20,000 particles;
this file fails on it in all three regimes). At the default 16 sub-steps the
5-seed means are 1.035 / 1.035 (stable, lowest 5% / 0-40 m), 1.006 / 1.008
(neutral) and 0.994 / 0.987 (convective), seed-to-seed standard deviation at
most 0.016; 8 sub-steps give 1.074 / 1.064 and 0.970 / 0.940
and 4 give 1.156 / 1.141 and 0.882 / 0.815
(outputs/agents/WM/fixcount/study/tables.md, docs/physics.md §6). The
tolerance, ±0.09, is the largest remaining error at the default (0.035) plus
three seed standard deviations (0.048), rounded up; it fails the old code and
4 sub-steps, not 8.
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
TOLERANCE = 0.09


def _met(blh_m: float, ustar: float, shf: float) -> HourlyMetTensors:
    """Uniform, motionless window with an almost constant pressure (uniform density)."""
    n_lev, n_lat, n_lon = 8, 4, 4
    shape = (n_lev, n_lat, n_lon)
    zeros = torch.zeros(shape, dtype=torch.float32)
    chans = {
        "u": zeros,
        "v": zeros,
        "w": zeros,
        "blh": torch.full(shape, blh_m),
        "sp": torch.full(shape, 101325.0),
        "t": torch.full(shape, 280.0),
        "ustar": torch.full(shape, ustar),
        "shf": torch.full(shape, shf),
    }
    names = tuple(chans)
    fields = torch.stack([chans[k] for k in names], dim=0)
    t0 = datetime(2024, 1, 1, 12, tzinfo=UTC)
    level = np.linspace(0.0, max(4000.0, 1.5 * blh_m), n_lev)
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


@pytest.mark.parametrize("regime", sorted(REGIMES))
def test_near_surface_well_mixed_at_60_s(regime: str) -> None:
    blh, ustar, shf = REGIMES[regime]
    torch.manual_seed(20261125)
    engine = _LidEngine(blh, device="cpu")
    scheme = HannaScheme(meander_enabled=False, static_substeps=False)  # default n_substeps
    met = _met(blh, ustar, shf)

    n = 20_000
    particles = torch.zeros(n, 4, dtype=torch.float32)
    particles[:, 1] = 45.0
    particles[:, 2] = torch.rand(n, dtype=torch.float32) * blh
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
    assert float(z.max()) <= blh and float(z.min()) >= 0.0
    lowest_5pc = float((z < 0.05 * blh).float().mean()) / 0.05
    lowest_40m = float((z < 40.0).float().mean()) / (40.0 / blh)
    assert abs(lowest_5pc - 1.0) < TOLERANCE, (
        f"{regime}: lowest 5% of the BL holds {lowest_5pc:.3f}x its share at dt = 60 s"
    )
    assert abs(lowest_40m - 1.0) < TOLERANCE, (
        f"{regime}: 0-40 m holds {lowest_40m:.3f}x its share at dt = 60 s"
    )
