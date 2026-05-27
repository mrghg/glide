"""Unit tests for the release-generator classes.

Focus is on the rewritten ColumnRelease: it now takes altitudes (m AGL) directly
and uses the averaging kernel as the only sampling weight, so the API and
weighting semantics need explicit coverage. PointRelease/VolumeRelease are
exercised end-to-end elsewhere.

The batch-aware ``generate_batch_particles`` (added in M5 stage 2) is covered
at the end of this module: single-release bit-equivalence with the legacy
PointRelease path, plus per-particle release-time sidecar correctness for
multi-release batches.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
import torch

from lpdm.config import ConcreteRelease, ReleaseBatch
from lpdm.release_generator import (
    ColumnRelease,
    PointRelease,
    ParticleBatch,
    generate_batch_particles,
)


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


# ---- generate_batch_particles --------------------------------------------------


def _single_release_batch(
    *,
    seed: int | None = 42,
    n_particles: int = 100,
    duration_seconds: int = 1800,
    release_time: datetime | None = None,
) -> ReleaseBatch:
    release_time = release_time or datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)
    concrete = ConcreteRelease(
        release_idx=0,
        release_time=release_time,
        lon=1.0,
        lat=2.0,
        alt_agl_m=400.0,
        duration_seconds=duration_seconds,
        n_particles=n_particles,
        seed=seed,
    )
    return ReleaseBatch(batch_idx=0, releases=(concrete,))


def test_generate_batch_particles_single_release_matches_pointrelease() -> None:
    """For a one-release batch, the particle tensor must equal PointRelease's output."""

    batch = _single_release_batch(seed=42, n_particles=100)
    pb = generate_batch_particles(batch, device="cpu")

    legacy = PointRelease(n_particles=100, lon=1.0, lat=2.0, alt=400.0, device="cpu").generate()

    assert isinstance(pb, ParticleBatch)
    assert torch.equal(pb.particles, legacy)
    assert pb.particles.shape == (100, 4)
    assert pb.release_idx.shape == (100,)
    assert torch.all(pb.release_idx == 0)


def test_generate_batch_particles_single_release_seeded_offsets_match_legacy() -> None:
    """The release-offset draw must reproduce main.py's legacy seeded draw exactly."""

    seed = 42
    n = 100
    duration_s = 1800.0

    batch = _single_release_batch(seed=seed, n_particles=n, duration_seconds=int(duration_s))
    pb = generate_batch_particles(batch, device="cpu")

    # Re-create the legacy draw verbatim. Same Generator on CPU, same uniform_,
    # same move-to-device pattern. batch_start equals release_time for N=1, so
    # release_time_offsets_s should equal the within-release draw.
    legacy_rng = torch.Generator(device="cpu")
    legacy_rng.manual_seed(seed)
    legacy_offsets = torch.empty(n, device="cpu", dtype=torch.float32).uniform_(
        0.0, duration_s, generator=legacy_rng
    )

    assert torch.equal(pb.release_time_offsets_s, legacy_offsets)
    assert pb.batch_start_time == batch.releases[0].release_time


def test_generate_batch_particles_unseeded_offsets_are_in_range() -> None:
    batch = _single_release_batch(seed=None, n_particles=500, duration_seconds=600)
    pb = generate_batch_particles(batch, device="cpu")

    assert torch.all(pb.release_time_offsets_s >= 0.0)
    assert torch.all(pb.release_time_offsets_s < 600.0)


def test_generate_batch_particles_multi_release_concatenates_and_tags() -> None:
    """Two releases at different times: particles, indices and offsets line up."""

    t0 = datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)
    t1 = t0 + timedelta(hours=1)
    rel0 = ConcreteRelease(
        release_idx=0,
        release_time=t0,
        lon=-9.9046,
        lat=53.3267,
        alt_agl_m=10.0,
        duration_seconds=3600,
        n_particles=200,
        seed=100,
    )
    rel1 = ConcreteRelease(
        release_idx=1,
        release_time=t1,
        lon=-9.9046,
        lat=53.3267,
        alt_agl_m=10.0,
        duration_seconds=3600,
        n_particles=300,
        seed=101,
    )
    batch = ReleaseBatch(batch_idx=0, releases=(rel0, rel1))

    pb = generate_batch_particles(batch, device="cpu")

    assert pb.particles.shape == (500, 4)
    assert pb.release_idx.tolist()[:200] == [0] * 200
    assert pb.release_idx.tolist()[200:] == [1] * 300
    assert pb.batch_start_time == t0

    # rel0's offsets stay in [0, 3600); rel1's are shifted by +3600 and stay in [3600, 7200).
    rel0_offsets = pb.release_time_offsets_s[:200]
    rel1_offsets = pb.release_time_offsets_s[200:]
    assert float(rel0_offsets.min()) >= 0.0
    assert float(rel0_offsets.max()) < 3600.0
    assert float(rel1_offsets.min()) >= 3600.0
    assert float(rel1_offsets.max()) < 7200.0

    # Weights are per-release (1/n_particles), so they differ between rel0 and rel1.
    assert torch.allclose(pb.particles[:200, 3], torch.full((200,), 1.0 / 200))
    assert torch.allclose(pb.particles[200:, 3], torch.full((300,), 1.0 / 300))


def test_generate_batch_particles_per_release_seeding_is_independent() -> None:
    """Different release seeds must produce different within-release offset draws."""

    t0 = datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)
    common_kwargs = dict(
        release_time=t0,
        lon=0.0,
        lat=0.0,
        alt_agl_m=10.0,
        duration_seconds=3600,
        n_particles=64,
    )
    rel_a = ConcreteRelease(release_idx=0, seed=1, **common_kwargs)
    rel_b = ConcreteRelease(release_idx=1, seed=2, **common_kwargs)
    pb = generate_batch_particles(
        ReleaseBatch(batch_idx=0, releases=(rel_a, rel_b)),
        device="cpu",
    )

    # With identical release_time and duration, batch_start_time == t0 so both
    # offset ranges are [0, 3600). Different seeds → different draws.
    a_offsets = pb.release_time_offsets_s[:64]
    b_offsets = pb.release_time_offsets_s[64:]
    assert not torch.equal(a_offsets, b_offsets)


def test_generate_batch_particles_seeded_draw_is_reproducible() -> None:
    """Same batch built twice must yield identical sidecar tensors."""

    batch = _single_release_batch(seed=7, n_particles=50)
    pb_a = generate_batch_particles(batch, device="cpu")
    pb_b = generate_batch_particles(batch, device="cpu")

    assert torch.equal(pb_a.particles, pb_b.particles)
    assert torch.equal(pb_a.release_time_offsets_s, pb_b.release_time_offsets_s)
    assert torch.equal(pb_a.release_idx, pb_b.release_idx)


def test_generate_batch_particles_rejects_empty_batch() -> None:
    with pytest.raises(ValueError, match="at least one release"):
        generate_batch_particles(ReleaseBatch(batch_idx=0, releases=()), device="cpu")


def test_generate_batch_particles_emits_release_window_end_offsets() -> None:
    """M5 stage 3: per-particle window-end offsets per release, in batch-start frame."""

    t0 = datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)
    t1 = t0 + timedelta(hours=2)
    rel0 = ConcreteRelease(
        release_idx=0, release_time=t0, lon=0.0, lat=0.0, alt_agl_m=10.0,
        duration_seconds=1800, n_particles=10, seed=1,
    )
    rel1 = ConcreteRelease(
        release_idx=1, release_time=t1, lon=0.0, lat=0.0, alt_agl_m=10.0,
        duration_seconds=3600, n_particles=20, seed=2,
    )
    pb = generate_batch_particles(
        ReleaseBatch(batch_idx=0, releases=(rel0, rel1)),
        device="cpu",
    )

    # batch_start = t0. rel0 window ends at t0 + 1800s → offset 1800.
    # rel1 starts at t0 + 7200s and lasts 3600s → window ends at t0 + 10800s → offset 10800.
    rel0_window_ends = pb.release_window_end_offsets_s[:10]
    rel1_window_ends = pb.release_window_end_offsets_s[10:]
    assert torch.all(rel0_window_ends == 1800.0)
    assert torch.all(rel1_window_ends == 10800.0)
    assert pb.release_window_end_offsets_s.shape == (30,)


def test_single_release_window_end_offset_equals_duration() -> None:
    """For a single release the window-end is exactly the duration_seconds."""

    batch = _single_release_batch(seed=42, n_particles=100, duration_seconds=1800)
    pb = generate_batch_particles(batch, device="cpu")

    assert pb.release_window_end_offsets_s.shape == (100,)
    assert torch.all(pb.release_window_end_offsets_s == 1800.0)


def test_generate_batch_particles_rejects_zero_n_particles() -> None:
    batch = _single_release_batch()
    bad = ConcreteRelease(
        release_idx=0,
        release_time=batch.releases[0].release_time,
        lon=0.0,
        lat=0.0,
        alt_agl_m=0.0,
        duration_seconds=600,
        n_particles=0,
        seed=None,
    )
    with pytest.raises(ValueError, match="n_particles=0"):
        generate_batch_particles(
            ReleaseBatch(batch_idx=0, releases=(bad,)),
            device="cpu",
        )
