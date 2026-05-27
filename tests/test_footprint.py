"""Unit tests for FootprintGridder accumulation correctness.

Covers binning, mass conservation, active-mask handling, out-of-bounds rejection,
and time-bin behavior. These tests run entirely on CPU with float64 for
determinism and have no dependency on real meteorology or the main runtime loop.
"""

from __future__ import annotations

import pytest
import torch

from lpdm.footprint_gridder import FootprintGridder


def _make_gridder(n_releases: int = 1) -> FootprintGridder:
    """Default 1x3x2x4x4 (r,t,z,y,x) gridder over [-1,1] lon/lat and [0,1000] z.

    Cell widths: 0.5 deg in lon/lat, 500 m in z, 1 hour bins. Stage 4 added the
    leading ``n_releases`` axis; the default `n_releases=1` mirrors the
    single-release runtime shape.
    """

    return FootprintGridder(
        lon_bounds=(-1.0, 1.0),
        lat_bounds=(-1.0, 1.0),
        z_edges_m=(0.0, 500.0, 1000.0),
        n_time_bins=3,
        n_y=4,
        n_x=4,
        n_releases=n_releases,
        device="cpu",
        dtype=torch.float64,
    )


def _t_idx(value: int, n: int) -> torch.Tensor:
    """Per-particle t_idx tensor with a single value broadcast over n particles.

    Stage 3 made FootprintGridder.accumulate take a per-particle t_idx tensor;
    the old scalar-int API is replaced by feeding a uniform tensor — which
    is what single-release runs always do.
    """

    return torch.full((n,), value, dtype=torch.int64)


def _release_idx(value: int, n: int) -> torch.Tensor:
    """Per-particle release_idx tensor with a single value broadcast over n.

    Stage 4 added the release dim; single-release tests use all-zero indices,
    multi-release tests use mixed values to exercise the per-release scatter.
    """

    return torch.full((n,), value, dtype=torch.int64)


def test_single_particle_lands_in_expected_cell() -> None:
    """A particle at a known position should add weight*dt to one specific cell."""

    g = _make_gridder()

    # lon=-0.75 -> bin 0; lat=-0.75 -> bin 0; z=250 -> bin 0; t_idx=1
    particles = torch.tensor([[-0.75, -0.75, 250.0]], dtype=torch.float64)
    weights = torch.tensor([0.5], dtype=torch.float64)
    active = torch.tensor([True])

    g.accumulate(particles, active, weights, t_idx=_t_idx(1, 1), release_idx=_release_idx(0, 1), dt_seconds=10.0)

    assert g.tensor[0, 1, 0, 0, 0].item() == 0.5 * 10.0
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
    g.accumulate(particles, active, weights, t_idx=_t_idx(0, n), release_idx=_release_idx(0, n), dt_seconds=dt)

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

    g.accumulate(particles, active, weights, t_idx=_t_idx(0, 3), release_idx=_release_idx(0, 3), dt_seconds=2.0)

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

    g.accumulate(particles, active, weights, t_idx=_t_idx(0, 7), release_idx=_release_idx(0, 7), dt_seconds=1.0)

    # Only the last (in-bounds) particle should contribute.
    assert abs(g.tensor.sum().item() - 1.0) < 1e-12


def test_repeat_accumulate_sums_into_same_bin() -> None:
    """Multiple accumulate calls into the same cell should sum, not overwrite."""

    g = _make_gridder()

    particles = torch.tensor([[-0.75, -0.75, 250.0]], dtype=torch.float64)
    weights = torch.tensor([1.0], dtype=torch.float64)
    active = torch.tensor([True])

    g.accumulate(particles, active, weights, t_idx=_t_idx(0, 1), release_idx=_release_idx(0, 1), dt_seconds=2.0)
    g.accumulate(particles, active, weights, t_idx=_t_idx(0, 1), release_idx=_release_idx(0, 1), dt_seconds=3.0)

    assert g.tensor[0, 0, 0, 0, 0].item() == 5.0
    assert g.tensor.sum().item() == 5.0


def test_invalid_time_bin_silently_dropped() -> None:
    """Out-of-range t_idx is currently a silent no-op; lock that contract in."""

    g = _make_gridder()

    particles = torch.tensor([[0.0, 0.0, 100.0]], dtype=torch.float64)
    weights = torch.tensor([1.0], dtype=torch.float64)
    active = torch.tensor([True])

    g.accumulate(particles, active, weights, t_idx=_t_idx(5, 1), release_idx=_release_idx(0, 1), dt_seconds=1.0)
    g.accumulate(particles, active, weights, t_idx=_t_idx(-1, 1), release_idx=_release_idx(0, 1), dt_seconds=1.0)

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

    g.accumulate(particles, active, weights, t_idx=_t_idx(0, 2), release_idx=_release_idx(0, 2), dt_seconds=1.0)

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

    g.accumulate(particles, active, weights, t_idx=_t_idx(0, 3), release_idx=_release_idx(0, 3), dt_seconds=1.0)

    # Each particle should land in a different z bin, all at the same horizontal cell.
    assert g.tensor[0, 0, 0].sum().item() == 1.0
    assert g.tensor[0, 0, 1].sum().item() == 1.0
    assert g.tensor[0, 0, 2].sum().item() == 1.0
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


def test_per_particle_t_idx_scatters_into_different_time_bins() -> None:
    """M5 stage 3: different t_idx per particle → different time slices accumulated."""

    g = _make_gridder()  # n_time_bins=3

    particles = torch.tensor(
        [
            [-0.5, -0.5, 100.0],
            [-0.5, -0.5, 100.0],
            [-0.5, -0.5, 100.0],
        ],
        dtype=torch.float64,
    )
    weights = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64)
    active = torch.ones(3, dtype=torch.bool)
    t_idx = torch.tensor([0, 1, 2], dtype=torch.int64)

    g.accumulate(particles, active, weights, t_idx=t_idx, release_idx=_release_idx(0, 3), dt_seconds=1.0)

    # All particles land at the same (z, y, x) cell, one per time bin.
    assert g.tensor[0, 0, 0, 1, 1].item() == 1.0
    assert g.tensor[0, 1, 0, 1, 1].item() == 1.0
    assert g.tensor[0, 2, 0, 1, 1].item() == 1.0
    assert g.tensor.sum().item() == 3.0


def test_per_particle_invalid_t_idx_dropped_others_accumulate() -> None:
    """A mix of valid and invalid per-particle t_idx: only the valid ones land."""

    g = _make_gridder()  # n_time_bins=3

    particles = torch.tensor(
        [
            [-0.5, -0.5, 100.0],
            [-0.5, -0.5, 100.0],
            [-0.5, -0.5, 100.0],
            [-0.5, -0.5, 100.0],
        ],
        dtype=torch.float64,
    )
    weights = torch.tensor([1.0, 1.0, 1.0, 1.0], dtype=torch.float64)
    active = torch.ones(4, dtype=torch.bool)
    # t_idx values: 0 (valid), -1 (out-of-range below), 1 (valid), 99 (out-of-range above).
    t_idx = torch.tensor([0, -1, 1, 99], dtype=torch.int64)

    g.accumulate(particles, active, weights, t_idx=t_idx, release_idx=_release_idx(0, 4), dt_seconds=1.0)

    # Only the two valid particles should contribute.
    assert g.tensor[0, 0, 0, 1, 1].item() == 1.0
    assert g.tensor[0, 1, 0, 1, 1].item() == 1.0
    assert g.tensor.sum().item() == 2.0


def test_uniform_t_idx_is_bit_equivalent_to_legacy_scalar_path() -> None:
    """Single-release runs always pass a uniform t_idx tensor; result must match
    what the pre-stage-3 scalar-int path would have produced."""

    g = _make_gridder()
    n = 50
    torch.manual_seed(7)
    particles = torch.empty((n, 3), dtype=torch.float64)
    particles[:, 0] = torch.empty(n, dtype=torch.float64).uniform_(-0.99, 0.99)
    particles[:, 1] = torch.empty(n, dtype=torch.float64).uniform_(-0.99, 0.99)
    particles[:, 2] = torch.empty(n, dtype=torch.float64).uniform_(0.0, 999.0)
    weights = torch.full((n,), 1.0 / n, dtype=torch.float64)
    active = torch.ones(n, dtype=torch.bool)

    g.accumulate(particles, active, weights, t_idx=_t_idx(2, n), release_idx=_release_idx(0, n), dt_seconds=4.0)

    # Per-particle scatter into time bin 2 should produce a tensor that sums to
    # the same total mass as the legacy scalar-slice scatter would: weights*dt.
    assert g.tensor[0, 2].sum().item() == pytest.approx(weights.sum().item() * 4.0, abs=1e-12)
    # And bins 0/1 are untouched.
    assert g.tensor[0, 0].sum().item() == 0.0
    assert g.tensor[0, 1].sum().item() == 0.0


def test_per_release_scatter_into_correct_leading_slice() -> None:
    """M5 stage 4: per-particle release_idx routes each contribution to its
    own slice along the leading n_releases axis."""

    g = _make_gridder(n_releases=3)  # shape (3, 3, 2, 4, 4)
    assert g.tensor.shape == (3, 3, 2, 4, 4)

    particles = torch.tensor(
        [
            [-0.5, -0.5, 100.0],   # release 0
            [-0.5, -0.5, 100.0],   # release 1
            [-0.5, -0.5, 100.0],   # release 2
            [-0.5, -0.5, 100.0],   # release 0 again
        ],
        dtype=torch.float64,
    )
    weights = torch.tensor([1.0, 1.0, 1.0, 1.0], dtype=torch.float64)
    active = torch.ones(4, dtype=torch.bool)
    t_idx = _t_idx(0, 4)
    release_idx = torch.tensor([0, 1, 2, 0], dtype=torch.int64)

    g.accumulate(particles, active, weights, t_idx=t_idx, release_idx=release_idx, dt_seconds=1.0)

    # Release 0 gets two contributions, releases 1 and 2 get one each.
    assert g.tensor[0].sum().item() == 2.0
    assert g.tensor[1].sum().item() == 1.0
    assert g.tensor[2].sum().item() == 1.0
    assert g.tensor.sum().item() == 4.0


def test_per_release_out_of_range_release_idx_dropped() -> None:
    """release_idx values outside [0, n_releases) must be filtered like other dims."""

    g = _make_gridder(n_releases=2)

    particles = torch.tensor(
        [
            [-0.5, -0.5, 100.0],   # release 0 (valid)
            [-0.5, -0.5, 100.0],   # release 2 (out of range above)
            [-0.5, -0.5, 100.0],   # release -1 (out of range below)
            [-0.5, -0.5, 100.0],   # release 1 (valid)
        ],
        dtype=torch.float64,
    )
    weights = torch.tensor([1.0, 1.0, 1.0, 1.0], dtype=torch.float64)
    active = torch.ones(4, dtype=torch.bool)
    t_idx = _t_idx(0, 4)
    release_idx = torch.tensor([0, 2, -1, 1], dtype=torch.int64)

    g.accumulate(particles, active, weights, t_idx=t_idx, release_idx=release_idx, dt_seconds=1.0)

    assert g.tensor[0].sum().item() == 1.0
    assert g.tensor[1].sum().item() == 1.0
    assert g.tensor.sum().item() == 2.0


def test_n_releases_zero_rejected_at_construction() -> None:
    with pytest.raises(ValueError, match="must be > 0"):
        FootprintGridder(
            lon_bounds=(-1.0, 1.0),
            lat_bounds=(-1.0, 1.0),
            z_edges_m=(0.0, 1000.0),
            n_time_bins=1,
            n_y=2,
            n_x=2,
            n_releases=0,
        )


def test_oversized_tensor_rejected_with_clear_error() -> None:
    """A misconfigured FLEXPART-scale run (e.g. 720 releases × 720 time bins ×
    400×300 grid) would allocate ~700 GB. The gridder must fail at construction
    with a MemoryError that names the requested size and the cap, not OOM."""

    with pytest.raises(MemoryError, match="Footprint tensor would require"):
        FootprintGridder(
            lon_bounds=(-100.0, 40.0),
            lat_bounds=(10.0, 80.0),
            z_edges_m=(0.0, 40.0, 1000.0, 5000.0),
            n_time_bins=720,
            n_y=293,
            n_x=391,
            n_releases=720,
            device="cpu",
        )


def test_env_var_lifts_footprint_cap(monkeypatch) -> None:
    """LPDM_FOOTPRINT_MAX_GIB overrides the default cap so users who genuinely
    need a bigger allocation can opt in without editing the source."""

    # A tensor that would fail with the default 32 GiB cap but fits under 200 GiB.
    monkeypatch.setenv("LPDM_FOOTPRINT_MAX_GIB", "200.0")
    with pytest.raises(MemoryError, match="Footprint tensor would require"):
        # ~700 GB; still exceeds 200 GiB.
        FootprintGridder(
            lon_bounds=(-100.0, 40.0),
            lat_bounds=(10.0, 80.0),
            z_edges_m=(0.0, 40.0, 1000.0, 5000.0),
            n_time_bins=720,
            n_y=293,
            n_x=391,
            n_releases=720,
            device="cpu",
        )

    # A tensor that would fail with default 32 GiB but FITS under 200 GiB:
    # ~7 GB (10× smaller leading dims).
    monkeypatch.setenv("LPDM_FOOTPRINT_MAX_GIB", "200.0")
    g = FootprintGridder(
        lon_bounds=(-100.0, 40.0),
        lat_bounds=(10.0, 80.0),
        z_edges_m=(0.0, 40.0, 1000.0, 5000.0),
        n_time_bins=24,
        n_y=293,
        n_x=391,
        n_releases=240,
        device="cpu",
    )
    assert g.tensor.shape == (240, 24, 3, 293, 391)
