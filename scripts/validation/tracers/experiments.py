"""Split DATEM samples into the published per-release experiments.

Hegarty et al. (2013) and Selvaratnam et al. (2023) score CAPTEX releases
1, 2, 3, 4, 5 and 7 separately, and ANATEX tracers 1 (Glasgow, GGW) and 2
(St Cloud, STC) over the first five releases of each (5–16 January 1987)
(``docs/published-tracer-scores.md`` §2). The papers do not say which
samples belong to which release, so several plausible rules are offered;
:func:`rule_table` gives each rule's pair counts beside Hegarty et al.'s.

Uses the measurement and emission files only, never model output.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .readers import datem_root, read_datem_emit, read_datem_meas

#: Pair counts in Hegarty et al. (2013) §6.
HEGARTY_PAIRS = {
    "CAPTEX-1": 354,
    "CAPTEX-2": 365,
    "CAPTEX-3": 346,
    "CAPTEX-4": 299,
    "CAPTEX-5": 306,
    "CAPTEX-7": 183,
    "ANATEX-GGW": 562,
    "ANATEX-STC": 556,
}


@dataclass(frozen=True)
class Experiment:
    name: str
    datem_dir: str
    tracer: int
    releases: tuple[int, ...]  # 1-based rows of emit-t<tracer>.txt


PUBLISHED_EXPERIMENTS = (
    *(Experiment(f"CAPTEX-{r}", "captex", 1, (r,)) for r in (1, 2, 3, 4, 5, 7)),
    Experiment("ANATEX-GGW", "anatex", 1, (1, 2, 3, 4, 5)),
    Experiment("ANATEX-STC", "anatex", 2, (1, 2, 3, 4, 5)),
)


@dataclass(frozen=True)
class AssignmentRule:
    """How a sample is assigned to a release.

    ``durations``: ``"all"``, or ``"6h_or_24h"`` to keep only samples of 6 or
    24 hours: CAPTEX's 6-hour samples (its 3-hour ones are dropped) and
    ANATEX's samples, which are all 24 h.
    ``assign``: ``"recent"`` — the latest release that started at or before
    the sample start; ``"window_start"`` — the latest release whose
    ``window_h``-hour window from its start contains the sample start;
    ``"window_end"`` — as ``window_start`` but the whole sample must end
    inside the window.
    ``anatex_last_start``: if set (``"YYYY-MM-DD"``), ANATEX samples are
    instead every sample starting on or after the first release's date and
    on or before this date, all credited to the experiment as a whole
    (release 0 = unassigned within it).
    ``anatex_wrf_domain``: if true, ANATEX samples at sites outside the
    10 km WRF domain that drove Hegarty et al.'s runs
    (:func:`inside_anatex_wrf_domain`) are dropped.
    """

    durations: str = "all"
    assign: str = "recent"
    window_h: float = 60.0
    anatex_last_start: str | None = None
    anatex_wrf_domain: bool = False

    @property
    def label(self) -> str:
        a = self.assign if self.assign == "recent" else f"{self.assign}{self.window_h:g}h"
        lab = f"{self.durations}/{a}"
        if self.anatex_last_start:
            lab += f"; ANATEX starts to {self.anatex_last_start}"
        if self.anatex_wrf_domain:
            lab += "; ANATEX sites inside WRF d02"
        return lab


CANDIDATE_RULES = (
    *(
        AssignmentRule(d, a, w)
        for d in ("6h_or_24h", "all")
        for a in ("recent", "window_start", "window_end")
        for w in ((60.0,) if a == "recent" else (48.0, 60.0, 72.0))
    ),
    *(
        AssignmentRule("6h_or_24h", "recent", anatex_last_start=f"1987-01-{d}", anatex_wrf_domain=w)
        for d in (12, 13, 14, 15)
        for w in (False, True)
    ),
)

#: The rule adopted: closest to Hegarty et al.'s counts (data/tracer-experiments.md §5).
ADOPTED_RULE = AssignmentRule("6h_or_24h", "recent", anatex_last_start="1987-01-13")


#: The 10 km WRF domain (d02) of Hegarty et al.'s ANATEX runs, from the
#: header of the ARL-format file they used, posted by NOAA ARL with DATEM:
#: https://www.ready.noaa.gov/data/archives/datem/anatex/aer_anatex_PBL1A_010600.d02
#: (first index record, read 2026-11-24): Lambert conformal, true latitudes
#: 33 and 45 N, reference longitude 100 W, 10 km grid, 454 x 334 points,
#: grid point (227, 167) at 41.2 N, 98.02 W. ARL's mapping routines use a
#: sphere of radius 6371.2 km.
ANATEX_WRF_D02 = dict(
    lat1=33.0,
    lat2=45.0,
    lon0=-100.0,
    dx_km=10.0,
    nx=454,
    ny=334,
    sync_x=227.0,
    sync_y=167.0,
    sync_lat=41.2,
    sync_lon=-98.02,
    radius_km=6371.2,
)


def wrf_grid_xy(lat, lon, grid: dict = ANATEX_WRF_D02) -> tuple[np.ndarray, np.ndarray]:
    """1-based grid coordinates of (lat, lon) on a spherical Lambert
    conformal grid with two true latitudes (Snyder, 1987, eqs. 15-1 to 15-3)."""
    rad = np.pi / 180.0
    p1, p2 = grid["lat1"] * rad, grid["lat2"] * rad
    n = np.log(np.cos(p1) / np.cos(p2)) / np.log(
        np.tan(np.pi / 4 + p2 / 2) / np.tan(np.pi / 4 + p1 / 2)
    )
    f = np.cos(p1) * np.tan(np.pi / 4 + p1 / 2) ** n / n

    def xy(la, lo):
        la, lo = np.asarray(la, float) * rad, np.asarray(lo, float)
        rho = grid["radius_km"] * f / np.tan(np.pi / 4 + la / 2) ** n
        th = n * (lo - grid["lon0"]) * rad
        return rho * np.sin(th), -rho * np.cos(th)

    x0, y0 = xy(grid["sync_lat"], grid["sync_lon"])
    x, y = xy(lat, lon)
    return grid["sync_x"] + (x - x0) / grid["dx_km"], grid["sync_y"] + (y - y0) / grid["dx_km"]


def inside_anatex_wrf_domain(lat, lon, grid: dict = ANATEX_WRF_D02) -> np.ndarray:
    """True where (lat, lon) lies inside the WRF grid (between its first and
    last points)."""
    gx, gy = wrf_grid_xy(lat, lon, grid)
    return (gx >= 1) & (gx <= grid["nx"]) & (gy >= 1) & (gy <= grid["ny"])


def assign_releases(
    samples: pd.DataFrame, releases: pd.DataFrame, rule: AssignmentRule
) -> pd.Series:
    """Release number (1-based) for each sample, or 0 when none.

    ``releases`` must be in time order (as in DATEM's emit files): the
    ``"recent"`` rule searches the start times and ``"window_*"`` let later
    releases win.
    """
    rs = releases["start"].to_numpy()
    if len(rs) > 1 and not (np.diff(rs) >= np.timedelta64(0, "s")).all():
        raise ValueError("releases must be in time order of their start")
    s_start = samples["start"].to_numpy()
    s_end = samples["end"].to_numpy()
    out = np.zeros(len(samples), dtype=int)
    if rule.assign == "recent":
        idx = np.searchsorted(rs, s_start, side="right") - 1
        out = np.where(idx >= 0, releases["release"].to_numpy()[np.maximum(idx, 0)], 0)
    else:
        w = np.timedelta64(int(rule.window_h * 3600), "s")
        for rel, r0 in zip(releases["release"], rs, strict=True):  # later releases win
            t = s_start if rule.assign == "window_start" else s_end
            inside = (s_start >= r0) & (
                (t < r0 + w) if rule.assign == "window_start" else (t <= r0 + w)
            )
            out = np.where(inside, rel, out)
    if rule.durations == "6h_or_24h":
        dur = samples["duration_s"].to_numpy()
        out = np.where((dur == 6 * 3600) | (dur == 24 * 3600), out, 0)
    return pd.Series(out, index=samples.index, name="release")


def split_samples(
    rule: AssignmentRule = ADOPTED_RULE, root: str | Path | None = None
) -> pd.DataFrame:
    """All samples of the published experiments under ``rule``.

    Columns: ``experiment``, ``release`` and the sample table of
    :func:`readers.read_datem_meas`. Samples with a missing measurement are
    dropped (none in the posted files).
    """
    root = Path(root) if root else datem_root()
    out = []
    for exp in PUBLISHED_EXPERIMENTS:
        d = root / "exp_data" / exp.datem_dir
        rel = read_datem_emit(d / f"emit-t{exp.tracer}.txt")
        meas = read_datem_meas(d / f"meas-t{exp.tracer}.txt")
        r = assign_releases(meas, rel, rule)
        keep = r.isin(exp.releases)
        if exp.datem_dir == "anatex" and rule.anatex_last_start:
            first = rel.loc[rel["release"] == exp.releases[0], "start"].iloc[0].normalize()
            last = pd.Timestamp(rule.anatex_last_start) + pd.Timedelta(days=1)
            keep = (meas["start"] >= first) & (meas["start"] < last)
            r = r.where(r.isin(exp.releases), 0)
        if exp.datem_dir == "anatex" and rule.anatex_wrf_domain:
            keep = keep & inside_anatex_wrf_domain(meas["lat"], meas["lon"])
        sel = meas[keep & (meas["conc_raw"] >= 0)].copy()
        sel.insert(0, "release", r[sel.index])
        sel.insert(0, "experiment", exp.name)
        sel["tracer"] = exp.tracer
        out.append(sel)
    return pd.concat(out, ignore_index=True)


def rule_table(root: str | Path | None = None, rules=CANDIDATE_RULES) -> pd.DataFrame:
    """Pair counts per experiment for each rule, beside Hegarty et al.'s."""
    rows = {}
    for rule in rules:
        counts = split_samples(rule, root).groupby("experiment").size()
        rows[rule.label] = counts
    t = pd.DataFrame(rows).T.reindex(columns=list(HEGARTY_PAIRS)).fillna(0).astype(int)
    hz = pd.Series(HEGARTY_PAIRS)
    t["sum_abs_diff"] = (t[list(HEGARTY_PAIRS)] - hz).abs().sum(axis=1)
    t.loc["Hegarty et al. (2013)"] = [*hz.to_list(), 0]
    return t


def sample_list(rule: AssignmentRule = ADOPTED_RULE, root: str | Path | None = None):
    """Sample list for A11, and the unique receptor windows to run.

    Returns ``(samples, receptors)``. ``samples``: experiment, release,
    sample ID, site, latitude, longitude, UTC start and end.
    ``receptors``: one row per unique (site, location, start, end); both
    ANATEX tracers share sites and times, so one backward run serves both.
    """
    s = split_samples(rule, root)
    samples = s[
        ["experiment", "release", "tracer", "sample_id", "site", "lat", "lon", "start", "end"]
    ].reset_index(drop=True)
    receptors = (
        samples.groupby(["site", "lat", "lon", "start", "end"], as_index=False)
        .agg(experiments=("experiment", lambda x: ";".join(sorted(set(x)))))
        .sort_values(["start", "site"])
        .reset_index(drop=True)
    )
    receptors.insert(0, "receptor_id", [f"R{i:05d}" for i in range(len(receptors))])
    return samples, receptors
