"""Unit tests for FootprintGridder accumulation correctness.

Covers binning, mass conservation, active-mask handling, out-of-bounds rejection,
and time-bin behavior. These tests run entirely on CPU with float64 for
determinism and have no dependency on real meteorology or the main runtime loop.
"""

from __future__ import annotations

import pytest
import torch

from lpdm.footprint_gridder import FootprintGridder


def _make_gridder() -> FootprintGridder:
    """Default 3x2x4x4 (t,z,y,x) gridder over [-1,1] lon/lat and [0,1000] z.

    Cell widths: 0.5 deg in lon/lat, 500 m in z, 1 hour bins.
    """

    return FootprintGridder(
        lon_bounds=(-1.0, 1.0),
        lat_bounds=(-1.0, 1.0),
        z_edges_m=(0.0, 500.0, 1000.0),
        n_time_bins=3,
        n_y=4,
        n_x=4,
        device="cpu",
        dtype=torch.float64,
    )


def test_single_particle_lands_in_expected_cell() -> None:
    """A particle at a known position should add weight*dt to one specific cell."""

    g = _make_gridder()

    # lon=-0.75 -> bin 0; lat=-0.75 -> bin 0; z=250 -> bin 0; t_idx=1
    particles = torch.tensor([[-0.75, -0.75, 250.0]], dtype=torch.float64)
    weights = torch.tensor([0.5], dtype=torch.float64)
    active = torch.tensor([True])

    g.accumulate(particles, active, weights, t_idx=1, dt_seconds=10.0)

    assert g.tensor[1, 0, 0, 0].item() == 0.5 * 10.0
    assert g.tensor.sum().item() == 0.5 * 10.0


def test_total_mass_conservation_in_bounds() -> None:
    """For all-in-bounds particles, total accumulated equals sum(weights) * dt."""

    g = _make_gridder()

    n = 100
    torch.manual_seed(42)
    particles = torch.empty((n, 3), dtype=torch.float64)
    particles[:, 0] = torch.empty(n, dtype=torch.float64).uniform_(-0.99, 0.99)
    particles[:, 1] = torch.empty(n, dtype=torch.float64).uniform_(-0.99, 0.99)
    particles[:, 2] = torch.empty(n, dtype=torch.float64).uniform_(0.0, 999.0)
    weights = torch.full((n,), 1.0 / n, dtype=torch.float64)
    active = torch.ones(n, dtype=torch.bool)

    dt = 5.0
    g.accumulate(particles, active, weights, t_idx=0, dt_seconds=dt)

    expected = weights.sum().item() * dt
    assert abs(g.tensor.sum().item() - expected) < 1e-12


def test_inactive_particles_excluded() -> None:
    """Particles with active_mask=False should not contribute to any bin."""

    g = _make_gridder()

    particles = torch.tensor(
        [
            [-0.5, -0.5, 100.0],
            [0.5, 0.5, 600.0],
            [0.0, 0.0, 250.0],
        ],
        dtype=torch.float64,
    )
    weights = torch.tensor([0.5, 0.3, 0.2], dtype=torch.float64)
    active = torch.tensor([True, False, True])

    g.accumulate(particles, active, weights, t_idx=0, dt_seconds=2.0)

    # Active weights sum = 0.5 + 0.2 = 0.7; * dt=2.0 = 1.4.
    assert abs(g.tensor.sum().item() - 1.4) < 1e-12


def test_out_of_bounds_particles_dropped() -> None:
    """Particles outside the gridder domain should not accumulate anywhere."""

    g = _make_gridder()

    particles = torch.tensor(
        [
            [-2.0, 0.0, 100.0],
            [2.0, 0.0, 100.0],
            [0.0, -2.0, 100.0],
            [0.0, 2.0, 100.0],
            [0.0, 0.0, -100.0],
            [0.0, 0.0, 2000.0],
            [0.0, 0.0, 100.0],
        ],
        dtype=torch.float64,
    )
    weights = torch.full((7,), 1.0, dtype=torch.float64)
    active = torch.ones(7, dtype=torch.bool)

    g.accumulate(particles, active, weights, t_idx=0, dt_seconds=1.0)

    # Only the last (in-bounds) particle should contribute.
    assert abs(g.tensor.sum().item() - 1.0) < 1e-12


def test_repeat_accumulate_sums_into_same_bin() -> None:
    """Multiple accumulate calls into the same cell should sum, not overwrite."""

    g = _make_gridder()

    particles = torch.tensor([[-0.75, -0.75, 250.0]], dtype=torch.float64)
    weights = torch.tensor([1.0], dtype=torch.float64)
    active = torch.tensor([True])

    g.accumulate(particles, active, weights, t_idx=0, dt_seconds=2.0)
    g.accumulate(particles, active, weights, t_idx=0, dt_seconds=3.0)

    assert g.tensor[0, 0, 0, 0].item() == 5.0
    assert g.tensor.sum().item() == 5.0


def test_invalid_time_bin_silently_dropped() -> None:
    """Out-of-range t_idx is currently a silent no-op; lock that contract in."""

    g = _make_gridder()

    particles = torch.tensor([[0.0, 0.0, 100.0]], dtype=torch.float64)
    weights = torch.tensor([1.0], dtype=torch.float64)
    active = torch.tensor([True])

    g.accumulate(particles, active, weights, t_idx=5, dt_seconds=1.0)
    g.accumulate(particles, active, weights, t_idx=-1, dt_seconds=1.0)

    assert g.tensor.sum().item() == 0.0


def test_empty_active_mask_is_noop() -> None:
    """If no particles are active, the tensor should be untouched."""

    g = _make_gridder()

    particles = torch.tensor(
        [[-0.5, -0.5, 100.0], [0.5, 0.5, 600.0]],
        dtype=torch.float64,
    )
    weights = torch.tensor([1.0, 1.0], dtype=torch.float64)
    active = torch.tensor([False, False])

    g.accumulate(particles, active, weights, t_idx=0, dt_seconds=1.0)

    assert g.tensor.sum().item() == 0.0


def test_non_uniform_z_edges_assign_correct_bins() -> None:
    """Non-uniform z edges should bin particles by interval, not fraction."""

    g = FootprintGridder(
        lon_bounds=(-1.0, 1.0),
        lat_bounds=(-1.0, 1.0),
        z_edges_m=(0.0, 40.0, 1000.0, 5000.0),
        n_time_bins=1,
        n_y=2,
        n_x=2,
        device="cpu",
        dtype=torch.float64,
    )

    # One particle in each of the three asymmetric vertical bins.
    particles = torch.tensor(
        [
            [-0.5, -0.5, 20.0],     # surface bin (0-40 m)
            [-0.5, -0.5, 500.0],    # mixed-layer bin (40-1000 m)
            [-0.5, -0.5, 3000.0],   # free-trop bin (1000-5000 m)
        ],
        dtype=torch.float64,
    )
    weights = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64)
    active = torch.ones(3, dtype=torch.bool)

    g.accumulate(particles, active, weights, t_idx=0, dt_seconds=1.0)

    # Each particle should land in a different z bin, all at the same horizontal cell.
    assert g.tensor[0, 0].sum().item() == 1.0
    assert g.tensor[0, 1].sum().item() == 1.0
    assert g.tensor[0, 2].sum().item() == 1.0
    assert g.tensor.sum().item() == 3.0


def test_z_edges_validation_rejects_non_ascending() -> None:
    with pytest.raises(ValueError, match="ascending"):
        FootprintGridder(
            lon_bounds=(-1.0, 1.0),
            lat_bounds=(-1.0, 1.0),
            z_edges_m=(0.0, 1000.0, 500.0),
            n_time_bins=1,
            n_y=2,
            n_x=2,
        )


def test_z_edges_validation_rejects_too_few_values() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        FootprintGridder(
            lon_bounds=(-1.0, 1.0),
            lat_bounds=(-1.0, 1.0),
            z_edges_m=(40.0,),
            n_time_bins=1,
            n_y=2,
            n_x=2,
        )
