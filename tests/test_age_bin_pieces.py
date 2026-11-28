"""`_age_bin_pieces` without float64 on the simulation device (PR #66 review).

Apple's MPS backend cannot allocate float64, so the trapezoid's split of each half
step by the time-ago bins is done in int64 ticks of 2**-20 s. These tests check
the seconds per bin, the agreement with the float64 arithmetic it replaced, that
no float64 tensor is created, and that a float64 accumulation grid is refused on
MPS.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import numpy as np
import pydantic
import pytest
import torch
from test_main_runtime import _make_run_config
from torch.utils._python_dispatch import TorchDispatchMode

import lpdm.main as lpdm_main
from lpdm.config import MPS_FLOAT64_ACCUMULATION_MESSAGE
from lpdm.footprint_gridder import FootprintGridder
from lpdm.gpu_engine import GPUEngine
from lpdm.local_frame import DEGREES
from lpdm.main import (
    PreflightValidationError,
    _age_bin_pieces,
    _check_device_dtypes,
    _deposit_step,
    _time_ago_index,
)


def _float64_age_bin_pieces(age_s, length_s, time_bin_s, dtype):
    """The float64 arithmetic of 6edcb96, kept here as the reference."""

    lo = age_s.to(torch.float64)
    hi = lo + length_s
    first = torch.floor(lo / time_bin_s)
    for j in range(math.ceil(length_s / time_bin_s) + 1):
        bin_lo = (first + j) * time_bin_s
        overlap = (torch.minimum(hi, bin_lo + time_bin_s) - torch.maximum(lo, bin_lo)).clamp_(
            min=0.0
        )
        yield (first + j).to(torch.int64), overlap.to(dtype)


def _one_cell_gridder(n_time_bins: int, dtype: torch.dtype) -> FootprintGridder:
    return FootprintGridder(
        lon_bounds=(0.0, 1.0),
        lat_bounds=(0.0, 1.0),
        z_edges_m=(0.0, 40.0, 1000.0),
        n_time_bins=n_time_bins,
        n_y=1,
        n_x=1,
        device="cpu",
        dtype=dtype,
        frame=DEGREES,
    )


def _trapezoid_step(gridder, age_s: float, dt: float, bin_s: float) -> None:
    """One trapezoid step of a stationary unit-weight particle, no density. The age
    is float32, as the runtime's ``elapsed_s`` is."""

    t0 = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)
    met_window = SimpleNamespace(
        metadata=SimpleNamespace(time_start=t0, time_end=t0 + timedelta(hours=1))
    )
    xyz = torch.tensor([[0.5, 0.5, 10.0]], dtype=torch.float32)
    elapsed = torch.tensor([age_s], dtype=torch.float32)
    _deposit_step(
        gridder,
        position="trapezoid",
        start_xyz=xyz,
        end_xyz=xyz,
        active_mask=torch.ones(1, dtype=torch.bool),
        weights=torch.ones(1, dtype=torch.float32),
        elapsed_s=elapsed,
        t_idx=_time_ago_index(elapsed, bin_s, gridder.n_t),
        time_bin_s=bin_s,
        release_idx=torch.zeros(1, dtype=torch.int64),
        delta_s=dt,
        t_cursor=t0 + timedelta(seconds=1800),
        particle_density=None,
        met_window=met_window,
        t_alpha=0.5,
        engine=GPUEngine(device="cpu", frame=DEGREES),
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64], ids=["f32", "f64"])
@pytest.mark.parametrize(
    ("age_s", "dt", "bin_s", "expected"),
    [
        # Young: a 300 s step from age 3500 s across the 3600 s edge. Halves
        # [3500, 3650) and [3650, 3800): 100 s in bin 0, 50 + 150 s in bin 1.
        (3500.0, 300.0, 3600.0, {0: 100.0, 1: 200.0}),
        # ~30 days: edge 722 h = 2,599,200 s inside the first half.
        (2_599_100.0, 300.0, 3600.0, {721: 100.0, 722: 200.0}),
        # ~30 days, a quarter second (one float32 spacing here) below the edge.
        (2_599_199.75, 300.0, 3600.0, {721: 0.25, 722: 299.75}),
        # ~30 days, odd dt (half steps of 150.5 s), the edge in the second half.
        (2_599_000.0, 301.0, 3600.0, {721: 200.0, 722: 101.0}),
        # ~30 days, bins narrower than a half step: 100 s bins, 26,000 + ...
        (2_599_950.0, 300.0, 100.0, {25999: 50.0, 26000: 100.0, 26001: 100.0, 26002: 50.0}),
    ],
)
def test_stationary_step_across_a_bin_edge_seconds_per_bin(
    age_s: float, dt: float, bin_s: float, expected: dict[int, float], dtype: torch.dtype
) -> None:
    """A stationary particle whose trapezoid step straddles a time-ago edge holds
    exactly the overlap in EACH bin, and the bins sum to the step, in a float32 and
    a float64 grid, young and at ~30 days (where float32 resolves only 0.25 s)."""

    n_bins = max(expected) + 2
    gridder = _one_cell_gridder(n_bins, dtype)
    _trapezoid_step(gridder, age_s, dt, bin_s)
    per_bin = gridder.tensor[0, :, 0, 0, 0]
    got = {int(k): float(per_bin[k]) for k in torch.nonzero(per_bin).flatten()}
    assert got == expected
    assert float(per_bin.sum()) == dt


def _float32_ages(rng: np.random.Generator, lo: float, hi: float, n: int) -> torch.Tensor:
    """Float32 ages: random values (every float32 fraction the range allows),
    whole and half seconds, and values on and one spacing either side of bin edges."""

    rand = rng.uniform(lo, hi, n).astype(np.float32)
    whole = np.floor(rand).astype(np.float32)
    half = (whole + np.float32(0.5)).astype(np.float32)
    edges = np.array([3600.0 * k for k in range(int(lo // 3600) + 1, int(hi // 3600))])
    edges = edges.astype(np.float32)
    near = np.concatenate(
        [edges, np.nextafter(edges, np.float32(0)), np.nextafter(edges, np.float32(1e9))]
    )
    return torch.from_numpy(np.concatenate([rand, whole, half, near]).astype(np.float32))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64], ids=["f32", "f64"])
@pytest.mark.parametrize("length_s", [150.0, 150.5, 30.0, 1800.0])
@pytest.mark.parametrize("bin_s", [3600.0, 300.0, 100.0, 7.0])
def test_long_ages_match_the_float64_arithmetic_bit_for_bit(
    bin_s: float, length_s: float, dtype: torch.dtype
) -> None:
    """Ages of 25-35 days (2.2e6-3.0e6 s): every float32 age there is a whole number
    of 2**-20 s ticks, and every overlap is exact in float64, so the int64 split
    gives the same bins and the same seconds, to the bit, as the float64 split it
    replaced, in either grid dtype; and its pieces sum to the length exactly."""

    rng = np.random.default_rng(20261128)
    ages = _float32_ages(rng, 2.2e6, 3.0e6, 20_000)
    new = list(_age_bin_pieces(ages, length_s, bin_s, 10**6, dtype))
    old = list(_float64_age_bin_pieces(ages, length_s, bin_s, dtype))
    assert len(new) == len(old)
    for (t_new, s_new), (t_old, s_old) in zip(new, old, strict=True):
        assert t_new.dtype == torch.int64 and s_new.dtype == dtype
        assert torch.equal(t_new, t_old)
        assert torch.equal(s_new, s_old)
    total = torch.stack([s.to(torch.float64) for _, s in new]).sum(dim=0)
    assert torch.all(total == length_s)


def test_young_ages_within_half_a_tick_of_the_float64_arithmetic() -> None:
    """Below 8 s a float32 age can be finer than a tick (2**-20 s); there the split
    is within 2**-21 s per bin boundary of the float64 one (stated precision)."""

    rng = np.random.default_rng(7)
    ages = torch.from_numpy(rng.uniform(0.0, 8.0, 20_000).astype(np.float32))
    new = list(_age_bin_pieces(ages, 0.75, 0.5, 10**6, torch.float64))
    old = list(_float64_age_bin_pieces(ages, 0.75, 0.5, torch.float64))
    worst = max(float((s_new - s_old).abs().max()) for (_, s_new), (_, s_old) in zip(new, old))
    assert worst <= 2.0**-21 * 2  # each piece has two boundaries


class _Float64Detector(TorchDispatchMode):
    """Records every aten op that returns a float64 tensor."""

    def __init__(self) -> None:
        super().__init__()
        self.float64_ops: list[str] = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        outs = out if isinstance(out, (tuple, list)) else (out,)
        if any(isinstance(o, torch.Tensor) and o.dtype == torch.float64 for o in outs):
            self.float64_ops.append(str(func))
        return out


@pytest.mark.parametrize("device", ["cpu", "meta"])
@pytest.mark.parametrize("length_s", [150.0, 150.5])
def test_age_bin_pieces_creates_no_float64_tensor(device: str, length_s: float) -> None:
    """With a float32 grid, no aten op inside `_age_bin_pieces` returns float64,
    including intermediates. `meta` is a non-CPU device that dispatches the same
    dtype rules without data. This shows the split asks for no float64; it does not
    run on MPS, so it cannot show that MPS implements each int64 op used (floor
    division, round, minimum/maximum, clamp), which PyTorch documents as
    supported. The detector is checked against the float64 reference."""

    ages = torch.tensor([3500.0, 2_599_199.75, 0.0], dtype=torch.float32, device=device)
    with _Float64Detector() as detector:
        for t_idx, seconds in _age_bin_pieces(ages, length_s, 3600.0, 800, torch.float32):
            assert t_idx.dtype == torch.int64 and seconds.dtype == torch.float32
    assert detector.float64_ops == []

    with _Float64Detector() as detector:
        list(_float64_age_bin_pieces(ages, length_s, 3600.0, torch.float32))
    assert detector.float64_ops  # the old arithmetic is caught


def test_float64_accumulation_is_refused_on_mps_in_config() -> None:
    cfg = _make_run_config()
    data = cfg.model_dump(mode="json")
    data["simulation"]["device"] = "mps"
    type(cfg).model_validate(data)  # float32 (default) is fine on MPS
    data["output_grid"]["accumulation_dtype"] = "float64"
    with pytest.raises(pydantic.ValidationError, match="no float64"):
        type(cfg).model_validate(data)
    data["simulation"]["device"] = "cuda"
    type(cfg).model_validate(data)


def test_float64_accumulation_is_refused_when_auto_resolves_to_mps(monkeypatch) -> None:
    """`device: auto` (or a CLI `--device mps`, applied without revalidation) is
    checked once the device is resolved, before any met or particle work."""

    cfg = _make_run_config()
    og = cfg.output_grid.model_copy(update={"accumulation_dtype": "float64"})
    cfg = cfg.model_copy(update={"output_grid": og})
    with pytest.raises(PreflightValidationError, match="no float64"):
        _check_device_dtypes(cfg, "mps")
    _check_device_dtypes(cfg, "cpu")
    _check_device_dtypes(cfg, "cuda:0")
    og32 = cfg.output_grid.model_copy(update={"accumulation_dtype": "float32"})
    _check_device_dtypes(cfg.model_copy(update={"output_grid": og32}), "mps")

    # `_run` stops there: the device resolves to "mps" and nothing else runs.
    monkeypatch.setattr(lpdm_main, "_resolve_device", lambda _name: "mps")

    with pytest.raises(PreflightValidationError) as err:
        lpdm_main._run(cfg)
    assert str(err.value) == MPS_FLOAT64_ACCUMULATION_MESSAGE
