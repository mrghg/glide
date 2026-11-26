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

Two checks, both on occupancies averaged over records every 5 min: every one of
the four 50 m bands from h - 100 to h + 100 m holds its uniform share to within
a tolerance, and the occupancy is continuous across h (the 50 m above minus the
50 m below, within a jump tolerance). A well-mixed-consistent interface keeps the
concentration continuous across h whatever the integration error elsewhere
(decision 0028), so the second check isolates the interface from the
first-order error of the stable boundary layer near its top, which the first
check also sees.

Two tiers (``Gate``):

- default suite (``FAST``, ~2.7 min at 8 threads): stable 60,000 particles,
  6 h, records from 1 h, tolerances 0.08; neutral and convective 40,000
  particles, 2 h, records from 15 min, tolerances 0.20;
- full strength (``FULL``, GLIDE_SLOW_TESTS=1, ~8.5 min): 150,000 particles,
  6 h, records from 1 h, tolerances 0.08, all three regimes.

The stable regime keeps the 0.08 jump tolerance and the 6-h run because the
pre-0028 defect there is small (+0.12 on the jump at 6 h, +0.09 at 2 h). The
neutral and convective defects are large (jump -0.59 and +3.2 already at 2 h),
so the default suite checks them with a shorter run, fewer particles and a wider
tolerance; the full tier holds them to 0.08.

Particle count. The seed-to-seed spread is set by the bands just above h, where
particles barely move (K_z 0.1 m² s⁻¹) and the counting noise stays frozen, so
more records do not reduce it; only more particles per metre do. The deep
neutral and convective columns (2h = 3 km) need the most: with 20,000 particles
the jump's spread there was 0.03-0.04, and 1 of 12 fresh seeds failed the 0.08
tolerance on correct code (jump -0.106, neutral). 100 m bands at h would halve
the noise but dilute the pre-0028 stable failure to about the tolerance (+0.09),
so the bands stay 50 m. Design study and per-seed values:
outputs/agents/T2/calib/designs_summary.txt (main checkout).

Before decision 0028 the drift's ±1 m finite difference straddled h, and a
particle crossing h kept its w'. Calibration, mean ± sd over seeds, default 16
sub-steps, margin = (tolerance - |mean|) / sd:

- default suite (outputs/agents/T2/fast/pooled_summary.txt; calib.jsonl seeds
  401-412, confirm.jsonl 501-512 through this file's run_column): current code
  0 failures; stable, 12 seeds, worst band h..h+50 -0.033 ± 0.009 (5.2 sd),
  jump -0.014 ± 0.011 (5.8 sd); neutral and convective, 24 seeds each, worst
  the jump, -0.012 ± 0.026 and -0.008 ± 0.027 (7.3 and 7.2 sd). PR #69 head
  0de5f94 fails every seed: stable 8/8 by the jump only, +0.124 ± 0.009
  (+0.110 to +0.138); neutral 10/10, jump -0.59 ± 0.02 and the two bands at h;
  convective 10/10, jump +3.17 ± 0.05 and the two bands above h;
- full strength (outputs/agents/T2/gate/gate.jsonl and summary.txt; seeds
  201-212): current code 0 failures, closest the convective h..h+50 band,
  -0.005 ± 0.014, and jump, -0.001 ± 0.014 (5.4 and 5.6 sd); neutral 6.3 and
  6.5 sd; stable worst band -0.032 ± 0.006 (8.0 sd). 0de5f94, 5 seeds, 5/5
  failures in each regime: stable only by the jump, +0.121 ± 0.010; neutral
  jump -0.80 ± 0.01 and every band; convective jump +4.86 ± 0.02 and three or
  four bands.

An earlier calibration at 20,000 particles and 15-min records from 2 h
(outputs/agents/WM/interface/gate/gate.jsonl) also failed main 20db902 in every
regime (jump -0.14 to -0.19, +4.0, +9.7).
"""

from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta
from typing import NamedTuple

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
RECORD_EVERY = 5  # steps (5 min) between records


class Gate(NamedTuple):
    particles: int
    steps: int  # 60 s outer steps; 360 = 6 h
    record_from_step: int  # first record (then every RECORD_EVERY steps)
    tolerance: float  # on each 50 m band's occupancy relative to uniform
    jump_tolerance: float  # on (h..h+50) - (h-50..h)


# Default suite: per-regime cost and tolerance (calibration in the docstring).
FAST = {
    "stable": Gate(60_000, 360, 60, 0.08, 0.08),
    "neutral": Gate(40_000, 120, 15, 0.20, 0.20),
    "convective": Gate(40_000, 120, 15, 0.20, 0.20),
}
# Full strength (GLIDE_SLOW_TESTS=1): one design, the tightest tolerances.
FULL = Gate(150_000, 360, 60, 0.08, 0.08)

SLOW = os.environ.get("GLIDE_SLOW_TESTS", "") not in ("", "0", "false", "False")
slow = pytest.mark.skipif(
    not SLOW, reason="~8.5 min on CPU: set GLIDE_SLOW_TESTS=1 (docs/VALIDATION.md §4)"
)


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


def run_column(
    regime: str, scheme: HannaScheme, gate: Gate = FULL, seed: int = 20261125
) -> dict[str, float]:
    """Occupancy of the four 50 m bands around h, relative to uniform, averaged
    over records every 5 min from ``gate.record_from_step`` to the end of a run
    of ``gate.steps`` 60 s steps with ``gate.particles`` particles. The start is
    stratified (one particle per equal slot of [0, 2h], jittered within it):
    above h particles move little in 6 h, so the binomial noise of an
    independent uniform start would stay frozen in the free-troposphere bands."""

    blh, ustar, shf = REGIMES[regime]
    n = gate.particles
    torch.manual_seed(seed)
    engine = _LidEngine(2.0 * blh, device="cpu")
    met = _met(blh, ustar, shf)
    span = 2.0 * blh
    particles = torch.zeros(n, 4, dtype=torch.float32)
    particles[:, 1] = 45.0
    particles[:, 2] = (torch.arange(n, dtype=torch.float32) + torch.rand(n)) * (span / n)
    particles[:, 3] = 1.0 / n
    state = scheme.initialize_state(n, device=torch.device("cpu"), dtype=torch.float32)
    active = torch.ones(n, dtype=torch.bool)

    samples = []
    for step in range(1, gate.steps + 1):
        particles, state = scheme.step(
            particles, state, met, t_alpha=0.5, dt_seconds=60.0, active_mask=active, engine=engine
        )
        if step >= gate.record_from_step and step % RECORD_EVERY == 0:
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


def failures(bands: dict[str, float], gate: Gate) -> list[str]:
    """The checks a run fails: bands off uniform by >= gate.tolerance, and the
    jump across h, (h..h+50) - (h-50..h), by >= gate.jump_tolerance."""

    bad = [k for k, v in bands.items() if abs(v - 1.0) >= gate.tolerance]
    if abs(bands["h..h+50"] - bands["h-50..h"]) >= gate.jump_tolerance:
        bad.append("jump")
    return bad


def _check(regime: str, gate: Gate) -> None:
    scheme = HannaScheme(meander_enabled=False, static_substeps=False)  # default n_substeps
    bands = run_column(regime, scheme, gate)
    jump = bands["h..h+50"] - bands["h-50..h"]
    bad = failures(bands, gate)
    assert not bad, (
        f"{regime}: fails {bad} (tolerance {gate.tolerance}, jump {gate.jump_tolerance}); "
        f"jump {jump:+.3f}, occupancy relative to uniform {bands}"
    )


@pytest.mark.parametrize("regime", sorted(REGIMES))
def test_uniform_column_stays_uniform_across_the_boundary_layer_top(regime: str) -> None:
    _check(regime, FAST[regime])


@slow
@pytest.mark.parametrize("regime", sorted(REGIMES))
def test_uniform_column_stays_uniform_across_the_boundary_layer_top_full(regime: str) -> None:
    _check(regime, FULL)
