"""Unit tests for the release-generator classes.

Focus is on the rewritten ColumnRelease: it now takes altitudes (m AGL) directly
and uses the averaging kernel as the only sampling weight, so the API and
weighting semantics need explicit coverage. PointRelease/VolumeRelease are
exercised end-to-end elsewhere.
"""

from __future__ import annotations

import pytest
import torch

from lpdm.release_generator import ColumnRelease


def test_column_release_uniform_sampling_covers_all_levels() -> None:
    """Without an averaging kernel, every supplied altitude should be sampled."""

    torch.manual_seed(101)
    altitudes = [100.0, 500.0, 1500.0, 3000.0]
    release = ColumnRelease(
        n_particles=2000,
        lon=10.0,
        lat=20.0,
        altitudes_agl_m=altitudes,
        device="cpu",
        dtype=torch.float64,
    )
    particles = release.generate()

    assert particles.shape == (2000, 4)
    assert torch.allclose(particles[:, 0], torch.full((2000,), 10.0, dtype=torch.float64))
    assert torch.allclose(particles[:, 1], torch.full((2000,), 20.0, dtype=torch.float64))

    sampled = particles[:, 2]
    unique = torch.unique(sampled)
    assert sorted(unique.tolist()) == altitudes

    counts = torch.tensor([(sampled == a).sum().item() for a in altitudes], dtype=torch.float64)
    fractions = counts / 2000.0
    expected = 0.25
    assert torch.all((fractions - expected).abs() < 0.05)

    assert torch.allclose(particles[:, 3].sum(), torch.tensor(1.0, dtype=torch.float64))


def test_column_release_averaging_kernel_biases_sampling() -> None:
    """Averaging-kernel weights should bias sampling toward heavier levels."""

    torch.manual_seed(202)
    altitudes = [100.0, 1000.0, 5000.0]
    kernel = [1.0, 4.0, 1.0]
    release = ColumnRelease(
        n_particles=4000,
        altitudes_agl_m=altitudes,
        averaging_kernel=kernel,
        device="cpu",
        dtype=torch.float64,
    )
    particles = release.generate()

    sampled = particles[:, 2]
    counts = torch.tensor([(sampled == a).sum().item() for a in altitudes], dtype=torch.float64)
    fractions = counts / 4000.0
    expected = torch.tensor([1.0, 4.0, 1.0], dtype=torch.float64)
    expected = expected / expected.sum()
    assert torch.all((fractions - expected).abs() < 0.03)


def test_column_release_rejects_negative_altitudes() -> None:
    release = ColumnRelease(
        n_particles=64,
        altitudes_agl_m=[100.0, -50.0, 200.0],
        device="cpu",
    )
    with pytest.raises(ValueError, match="non-negative"):
        release.generate()


def test_column_release_rejects_kernel_length_mismatch() -> None:
    release = ColumnRelease(
        n_particles=64,
        altitudes_agl_m=[100.0, 500.0],
        averaging_kernel=[1.0, 1.0, 1.0],
        device="cpu",
    )
    with pytest.raises(ValueError, match="length must match"):
        release.generate()


def test_column_release_rejects_zero_total_weight() -> None:
    release = ColumnRelease(
        n_particles=64,
        altitudes_agl_m=[100.0, 500.0],
        averaging_kernel=[0.0, 0.0],
        device="cpu",
    )
    with pytest.raises(ValueError, match="weights sum to zero"):
        release.generate()
