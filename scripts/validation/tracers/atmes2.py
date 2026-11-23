"""ATMES-II statistics for ETEX-1 (Mosca et al., 1998).

Summary of the method in ``docs/published-tracer-scores.md`` §6. Inputs
are the JRC files (the filter needs the ``-.99`` and ``-.88`` codes, which
DATEM's copy drops). Concentrations are in ng m-3 throughout.

Choices this module makes where the paper is not explicit (each is an
argument, so the pre-registration can fix it):

- **Zero filter.** A station's cloud arrival and departure are its first
  and last interval with a positive measurement; a measured zero is kept
  when it lies no more than ``margin`` intervals (default 2, 6 h) before the
  arrival or after the departure. ``-.88`` (detected, not quantifiable) and
  ``-.99`` are lost samples. Stations with no positive measurement keep no
  zeros.
- **Logarithmic indices** (MG, VG, PCC) set zeros to ``log_floor``
  (0.01 ng m-3). MG is ``exp(<ln P> - <ln M>)``: the published signs show it
  (model 117 under-predicts, FOEX -41, and has MG 0.315; model 101
  over-predicts, FOEX +12, and has MG 1.258).
- **FA2 and FA5** use the same floor, so a pair of zeros counts as within a
  factor of 2.
- **KS.** The published values (72–365) exceed 100, so they are not the
  percentage DATEM uses. :func:`atmes2_statistics` returns the maximum
  difference between the two empirical cumulative distributions as a
  fraction (``ks_d``), and that difference times the number of pairs
  (``ks_count``), which on 1,423 pairs gives D = 0.05–0.26 for the published
  range. ``ks_count`` is a hypothesis to be checked against the JRC report
  (EUR 17756 EN); until then compare ``ks_d`` only between GLIDE runs.
- **FMS in space** interpolates log concentrations linearly on a Delaunay
  triangulation of the stations, in a Lambert azimuthal equal-area
  projection, and measures areas above the threshold on a regular grid.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sps

LOST = (-0.99, -0.88)


def zero_filter(samples: pd.DataFrame, *, margin: int = 2, last_interval: int = 30) -> pd.Series:
    """Boolean mask of the JRC samples ATMES-II keeps (see module docstring).

    ``samples`` is :func:`readers.read_etex_samples` output (``site``,
    ``interval``, ``conc_raw``).
    """
    raw = samples["conc_raw"].to_numpy()
    interval = samples["interval"].to_numpy()
    valid = (raw >= 0) & (interval <= last_interval)
    pos = valid & (raw > 0)
    g = pd.DataFrame({"site": samples["site"].to_numpy(), "i": np.where(pos, interval, np.nan)})
    first = g.groupby("site")["i"].transform("min").to_numpy()
    last = g.groupby("site")["i"].transform("max").to_numpy()
    with np.errstate(invalid="ignore"):
        near = (interval >= first - margin) & (interval <= last + margin)
    keep = pos | (valid & (raw == 0) & near)
    return pd.Series(keep, index=samples.index, name="atmes2_keep")


def _ci95_mean(x: np.ndarray) -> tuple[float, float]:
    n = len(x)
    if n < 2:
        return float("nan"), float("nan")
    half = sps.t.ppf(0.975, n - 1) * x.std(ddof=1) / np.sqrt(n)
    return float(x.mean() - half), float(x.mean() + half)


def ks_statistic(meas: np.ndarray, pred: np.ndarray) -> float:
    """Maximum absolute difference between the two empirical CDFs (0–1)."""
    grid = np.union1d(meas, pred)
    fm = np.searchsorted(np.sort(meas), grid, side="right") / len(meas)
    fp = np.searchsorted(np.sort(pred), grid, side="right") / len(pred)
    return float(np.max(np.abs(fm - fp)))


def atmes2_statistics(meas, pred, *, log_floor: float = 0.01) -> dict:
    """Global ATMES-II statistics on filtered pairs (ng m-3)."""
    m = np.asarray(meas, dtype=float)
    p = np.asarray(pred, dtype=float)
    if m.shape != p.shape or np.any(m < 0) or np.any(p < 0):
        raise ValueError("paired, non-negative concentrations are required")
    n = len(m)
    d = p - m
    lo, hi = _ci95_mean(d)
    lm, lp = np.log(np.maximum(m, log_floor)), np.log(np.maximum(p, log_floor))
    ratio = np.maximum(p, log_floor) / np.maximum(m, log_floor)
    ks_d = ks_statistic(m, p)
    return dict(
        n_pairs=n,
        pct_measured_zero=100.0 * float((m == 0).mean()),
        bias=float(d.mean()),
        bias_ci95_low=lo,
        bias_ci95_high=hi,
        nmse=float((d**2).mean() / (m.mean() * p.mean())),
        mg=float(np.exp(lp.mean() - lm.mean())),
        vg=float(np.exp(((lp - lm) ** 2).mean())),
        pcc_log=float(np.corrcoef(lm, lp)[0, 1]),
        fa2=100.0 * float(((ratio >= 0.5) & (ratio <= 2.0)).mean()),
        fa5=100.0 * float(((ratio >= 0.2) & (ratio <= 5.0)).mean()),
        foex=100.0 * (float((p > m).mean()) - 0.5),
        ks_d=ks_d,
        ks_count=ks_d * n,
    )


# ---------------------------------------------------------------------------
# figure of merit in space from interpolated contours
# ---------------------------------------------------------------------------


def _laea(lat, lon, lat0=50.0, lon0=10.0):
    """Lambert azimuthal equal-area projection (unit sphere radius 6371 km)."""
    R = 6371.0
    φ, λ = np.radians(lat), np.radians(lon)
    φ0, λ0 = np.radians(lat0), np.radians(lon0)
    k = np.sqrt(2.0 / (1.0 + np.sin(φ0) * np.sin(φ) + np.cos(φ0) * np.cos(φ) * np.cos(λ - λ0)))
    x = R * k * np.cos(φ) * np.sin(λ - λ0)
    y = R * k * (np.cos(φ0) * np.sin(φ) - np.sin(φ0) * np.cos(φ) * np.cos(λ - λ0))
    return x, y


def fms_space(
    lat, lon, meas, pred, *, threshold: float = 0.1, log_floor: float = 0.01, cell_km: float = 10.0
) -> float:
    """FMS (%) from log-linearly interpolated fields at one time (or for dose).

    Interpolates ``log(max(c, log_floor))`` linearly on the stations'
    Delaunay triangulation in an equal-area projection; areas are the grid
    cells (``cell_km``) inside the convex hull where the field exceeds
    ``threshold``. Stations with a missing value are left out by the caller.
    """
    from scipy.interpolate import LinearNDInterpolator

    x, y = _laea(np.asarray(lat), np.asarray(lon))
    pts = np.column_stack([x, y])
    gx = np.arange(x.min(), x.max() + cell_km, cell_km)
    gy = np.arange(y.min(), y.max() + cell_km, cell_km)
    X, Y = np.meshgrid(gx, gy)
    lt = np.log(threshold)
    areas = []
    for c in (meas, pred):
        f = LinearNDInterpolator(pts, np.log(np.maximum(np.asarray(c, float), log_floor)))
        areas.append(f(X, Y) > lt)
    a_m, a_p = areas
    union = np.sum(a_m | a_p)
    return float(100.0 * np.sum(a_m & a_p) / union) if union else float("nan")
