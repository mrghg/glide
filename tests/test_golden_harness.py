"""Tests for the golden-fixture comparison and benchmark parser (scripts/agent, task A0).

Synthetic arrays and log text only: no meteorology, no model run.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "agent"))

import benchmark  # noqa: E402
import compare_golden as cg  # noqa: E402

CUDA_TOL = cg.Tolerance(exact=False, **cg.CUDA_DEFAULTS)


def _fixture(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    b = rng.random((3, 1, 2, 8, 9)).astype(np.float32) * 100.0
    b[:, :, :, :2, :] = 0.0  # empty cells, as in a real footprint
    return b


def test_identical_footprints_pass_exact_and_cuda():
    b = _fixture()
    for tol in (cg.Tolerance(exact=True), CUDA_TOL):
        r = cg.compare_footprints(b.copy(), b, tol)
        assert r["pass"]
        assert r["worst"]["max_abs"] == 0.0
        assert r["worst"]["releases_bitwise_equal"] == 3


def test_one_ulp_fails_exact_passes_cuda():
    b = _fixture()
    a = b.copy()
    a[1, 0, 1, 5, 5] = np.nextafter(a[1, 0, 1, 5, 5], np.float32(np.inf))
    assert not cg.compare_footprints(a, b, cg.Tolerance(exact=True))["pass"]
    r = cg.compare_footprints(a, b, CUDA_TOL)
    assert r["pass"]
    assert r["releases"][1]["cells_differing"] == 1
    assert r["releases"][0]["bitwise_equal"] and not r["releases"][1]["bitwise_equal"]


def test_cell_bound_is_relative_plus_floor_of_release_max():
    b = _fixture()
    maxb = float(np.abs(b[0]).max())
    a = b.copy()
    # An empty cell may take up to 1e-7 * max|b| ...
    a[0, 0, 0, 0, 0] = 0.9e-7 * maxb
    r = cg.compare_footprints(a, b, CUDA_TOL)
    assert r["releases"][0]["cell_ratio"] == pytest.approx(0.9, rel=1e-3)
    assert r["releases"][0]["cells_new_nonzero"] == 1
    # ... but not more.
    a[0, 0, 0, 0, 0] = 1.1e-7 * maxb
    r = cg.compare_footprints(a, b, CUDA_TOL)
    assert r["releases"][0]["cell_ratio"] > 1.0
    assert not r["pass"]


def test_relative_cell_error_above_limit_fails():
    b = _fixture()
    a = b.copy()
    i = (2, 0, 1, 6, 3)
    a[i] = b[i] * (1 + 5e-5)
    r = cg.compare_footprints(a, b, CUDA_TOL)
    assert not r["releases"][2]["pass"]
    assert r["worst"]["max_rel"] == pytest.approx(5e-5, rel=1e-2)


def test_release_total_criterion():
    b = _fixture()
    a = b.copy()
    # Every cell 4e-6 high: inside the per-cell bound (1e-5), outside the total (1e-6).
    a[0] = (b[0].astype(np.float64) * (1 + 4e-6)).astype(np.float32)
    r = cg.compare_footprints(a, b, CUDA_TOL)
    assert r["releases"][0]["cell_ratio"] < 1.0
    assert r["releases"][0]["total_rel"] == pytest.approx(4e-6, rel=0.05)
    assert not r["pass"]
    loose = cg.Tolerance(exact=False, rtol=1e-5, atol_frac=1e-7, total_rtol=1e-5)
    assert cg.compare_footprints(a, b, loose)["pass"]


def test_shape_mismatch_fails():
    b = _fixture()
    r = cg.compare_footprints(b[:2], b, CUDA_TOL)
    assert not r["pass"] and "shape" in r["error"]


LOG = """\
2026-11-24 12:42:22,473 INFO __main__: something
GLIDE phase timers — total wall 1202.8s  (met fetches: 168)  [no CUDA sync: GPU phases under-counted]
------------------------------------------------------------------------------
  phase                      total_s   % wall      calls      ms/call
  met_fetch                   943.31    78.4%      17280       54.590
  gridder                      75.69     6.3%      17280        4.380
  convection                   14.61     1.2%        288       50.728
  (residual/untimed)          103.60     8.6%
==============================================================================
LPDM run complete
\tMaximum resident set size (kbytes): 53300000
\tElapsed (wall clock) time (h:mm:ss or m:ss): 20:17.70
"""


def test_parse_phase_table_and_time_v(tmp_path):
    t = benchmark.parse_phase_table(LOG)
    assert t["total_wall_s"] == 1202.8 and t["met_fetches"] == 168
    assert not t["gpu_synced"]
    names = [p["phase"] for p in t["phases"]]
    assert names == ["met_fetch", "gridder", "convection", "(residual/untimed)"]
    assert t["phases"][0]["calls"] == 17280 and t["phases"][0]["ms_per_call"] == 54.59
    assert t["phases"][-1]["calls"] is None
    log = tmp_path / "run.log"
    log.write_text(LOG)
    s = benchmark.summarise(log, None, None)
    assert s["process_wall_s"] == pytest.approx(1217.7)
    assert s["max_rss_gb"] == pytest.approx(53.3)
    md = benchmark.to_markdown(s)
    assert "| met_fetch | 943.31 | 78.4 | 17280 | 54.590 |" in md


def test_parse_phase_table_absent():
    assert benchmark.parse_phase_table("no timers here\n") is None
