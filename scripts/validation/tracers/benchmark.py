"""The tracer benchmark: run GLIDE backward from every sample, score, write a scorecard.

Queue task B12. Run from ``scripts/validation``::

    python -m tracers.benchmark plan  --experiments OKC80 OKC80-PDCH
    python -m tracers.benchmark run   --experiments OKC80 OKC80-PDCH \\
        --met $GLIDE_DATA/era5-arco/NAM_19800708-19800712_ERA5-ARCO.zarr --out RUN_DIR \\
        [--config BASE.yaml] [--set turbulence.meander.coefficient=0.3] [--particles 2000]
    python -m tracers.benchmark score RUN_DIR [--basis ambient] [--out OTHER_DIR]
    python -m tracers.benchmark compare RUN_A RUN_B [RUN_C ...]

``run`` builds one backward release per unique sample window (kind
``samples``: particles spread uniformly over the sample period and, by
default, between the ground and 100 m), runs GLIDE once per source site
with a footprint grid of 3 × 3 cells of ``--source-deg`` centred on the
source and one layer (the lowest 40 m by default), converts the footprints
into predicted concentrations (``concentration.py``), writes them in DATEM
format and scores each experiment with B10's ``datem_stats`` under the
published protocol. ``scorecard.md``, ``scorecard.csv`` and
``scorecard.json`` record the scores beside the published values and the
provenance needed to compare two runs.

**Independence.** Before the pre-registration GLIDE may run only on OKC80
(validation-plan Tier 3). ``run`` refuses any other experiment unless
``--registration-record`` names a pre-registration record whose
``**Registration.**`` line is filled in; ``plan`` prints sizes for any
experiment but writes nothing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from functools import reduce
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from . import concentration, datem_stats, experiments
from .readers import REPO_ROOT, datem_root, read_datem_emit, read_datem_meas

BASE_CONFIG = Path(__file__).with_name("benchmark_base.yaml")
PUBLISHED_CSV = REPO_ROOT / "docs" / "published-tracer-scores.csv"
SCORECARD_VERSION = 1
#: The output-grid settings the benchmark derives for every site (`glide_config`).
OUTPUT_GEOMETRY = frozenset(
    ("lon_bounds", "lat_bounds", "n_x", "n_y", "z_edges_m", "n_time_bins", "time_bin_seconds")
)


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BenchmarkExperiment:
    """One scored experiment: which samples, and which releases they are predicted from."""

    name: str
    datem_dir: str
    tracer: int
    releases: tuple[int, ...]  # 1-based rows of emit-t<tracer>.txt summed into each prediction
    graded: bool  # a Tier 3 experiment: GLIDE may not run it before the registration
    #: Molar volume behind the file's fL/L -> pg m-3 conversion, and where it comes from.
    molar_volume_m3_mol: float = concentration.DATEM_MOLAR_VOLUME_M3_MOL
    molar_volume_source: str = "decision 0022: pg m-3 = fL/L x M / 22.40 L mol-1"


EXPERIMENTS: dict[str, BenchmarkExperiment] = {
    "OKC80": BenchmarkExperiment(
        "OKC80",
        "okc80",
        1,
        (1,),
        graded=False,
        molar_volume_source=(
            "assumed 22.40 L mol-1 as for CAPTEX and ANATEX; DATEM's readme gives pg m-3 only "
            "and the OKC80 report (EPA-600/7-81-006) was not checked for the factor"
        ),
    ),
    "OKC80-PDCH": BenchmarkExperiment(
        "OKC80-PDCH",
        "okc80",
        2,
        (1,),
        graded=False,
        molar_volume_source=("assumed 22.40 L mol-1 as for CAPTEX and ANATEX (see OKC80)"),
    ),
    **{
        e.name: BenchmarkExperiment(e.name, e.datem_dir, e.tracer, e.releases, graded=True)
        for e in experiments.PUBLISHED_EXPERIMENTS
    },
}
#: The eight experiments of the published per-release comparisons.
PUBLISHED_SET = tuple(e.name for e in experiments.PUBLISHED_EXPERIMENTS)


def load_experiment(name: str, root: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Samples and emissions of one experiment.

    Samples: the B10 sample table (DATEM keys kept) plus ``experiment``.
    Published experiments use B10's adopted assignment rule; OKC80 uses every
    measured sample. Emissions: the rows of the release table summed into
    each prediction.
    """

    exp = EXPERIMENTS[name]
    root = Path(root) if root else datem_root()
    d = root / "exp_data" / exp.datem_dir
    emit = read_datem_emit(d / f"emit-t{exp.tracer}.txt")
    emit = emit[emit["release"].isin(exp.releases)].reset_index(drop=True)
    if exp.graded:
        s = experiments.split_samples(experiments.ADOPTED_RULE, root)
        s = s[s["experiment"] == name].drop(columns=["release", "tracer"])
    else:
        s = read_datem_meas(d / f"meas-t{exp.tracer}.txt")
        s = s[s["conc_raw"] >= 0].copy()
        s.insert(0, "experiment", name)
    return s.reset_index(drop=True), emit


# ---------------------------------------------------------------------------
# Independence guard
# ---------------------------------------------------------------------------


def registration_filled(record: Path | None) -> bool:
    """True when the record's ``**Registration.**`` line names a registration."""

    if record is None or not Path(record).is_file():
        return False
    for line in Path(record).read_text().splitlines():
        m = re.match(r"\s*\*\*Registration\.\*\*\s*(.*)$", line)
        if m:
            rest = m.group(1).strip()
            return bool(rest) and "pending" not in rest.lower()
    return False


def check_independence(names, record: Path | None) -> None:
    graded = [n for n in names if EXPERIMENTS[n].graded]
    if graded and not registration_filled(record):
        raise SystemExit(
            f"refusing to run GLIDE on {', '.join(graded)}: Tier 3 experiments stay independent "
            "until the pre-registration is registered (validation-plan Tier 3). Pass "
            "--registration-record with a record whose **Registration.** line is filled in."
        )


# ---------------------------------------------------------------------------
# Plan: receptors, source sites, timing
# ---------------------------------------------------------------------------


@dataclass
class Chunk:
    """One GLIDE run: consecutive sample windows of one site, with one run length."""

    key: str
    receptors: pd.DataFrame  # receptor_id, lat, lon, start, end
    length_s: int


@dataclass
class SitePlan:
    """The receptors that need one source site's footprint, split into GLIDE runs."""

    key: str
    lat: float
    lon: float
    experiments: list[str]
    receptors: pd.DataFrame
    earliest_emission: pd.Timestamp
    chunks: list[Chunk] = field(default_factory=list)


@dataclass
class Plan:
    samples: pd.DataFrame  # every scored sample, with receptor_id and site_key
    emissions: dict[str, pd.DataFrame]  # experiment -> emission rows
    receptors: pd.DataFrame
    sites: list[SitePlan]
    time_bin_s: int
    layer: tuple[float, float]


def _epoch_s(t) -> np.ndarray:
    return (np.asarray(t, dtype="datetime64[ns]").astype(np.int64) // 10**9).astype(np.int64)


def choose_time_bin(times_s: np.ndarray, cap_s: int = 3600) -> int:
    """The widest bin (at most ``cap_s``) that every sample and emission time is a
    multiple of, so that emission windows fall on bin edges and the timing is exact."""

    g = reduce(math.gcd, (int(t) for t in np.unique(times_s)), 0)
    return math.gcd(g, cap_s) if g else cap_s


def _length_needed(rec: pd.DataFrame, first_emission: pd.Timestamp, bin_s: int) -> int:
    """Run length for a set of windows: every particle is followed for the same length
    from its own release, so it must reach from the latest sample end back to the
    first emission (and exceed the longest window, which GLIDE requires)."""

    need = (rec["end"].max() - first_emission).total_seconds()
    longest = (rec["end"] - rec["start"]).dt.total_seconds().max()
    need = max(need, longest + bin_s)
    return int(math.ceil(need / bin_s) * bin_s)


def chunk_receptors(
    rec: pd.DataFrame,
    first_emission: pd.Timestamp,
    bin_s: int,
    *,
    met_start: pd.Timestamp | None = None,
    max_span_h: float = 24.0,
) -> list[tuple[pd.DataFrame, int]]:
    """Split time-ordered windows into GLIDE runs.

    One length serves a whole run, and the run needs meteorology from its
    earliest window start minus that length. Grouping windows that start
    within ``max_span_h`` of each other keeps each run's length close to what
    its own windows need, and keeps the start of the meteorology needed after
    ``met_start`` (a few hours before the first emission).
    """

    rec = rec.sort_values(["start", "end"]).reset_index(drop=True)
    out, i = [], 0
    while i < len(rec):
        j = i + 1
        while j < len(rec):
            cand = rec.iloc[i : j + 1]
            span_ok = (
                cand["start"].max() - cand["start"].min()
            ).total_seconds() <= max_span_h * 3600
            length = _length_needed(cand, first_emission, bin_s)
            cover_ok = met_start is None or (
                cand["start"].min() - pd.Timedelta(seconds=length) >= met_start
            )
            if not (span_ok and cover_ok):
                break
            j += 1
        part = rec.iloc[i:j].reset_index(drop=True)
        length = _length_needed(part, first_emission, bin_s)
        if met_start is not None and part["start"].min() - pd.Timedelta(seconds=length) < met_start:
            raise SystemExit(
                f"the meteorology starts at {met_start}, too late for the windows starting "
                f"{part['start'].min()} (they need {length / 3600:.1f} h back)"
            )
        out.append((part, length))
        i = j
    return out


def build_plan(
    names: list[str],
    *,
    layer: tuple[float, float] = (0.0, 100.0),
    root: Path | None = None,
    met_start: pd.Timestamp | None = None,
    max_span_h: float = 24.0,
) -> Plan:
    samples, emissions = [], {}
    for n in names:
        s, e = load_experiment(n, root)
        samples.append(s)
        emissions[n] = e
    samples = pd.concat(samples, ignore_index=True)
    key = ["lat", "lon", "start", "end"]
    receptors = (
        samples[key]
        .drop_duplicates()
        .sort_values(["start", "end", "lat", "lon"])
        .reset_index(drop=True)
    )
    receptors.insert(0, "receptor_id", [f"R{i:05d}" for i in range(len(receptors))])
    samples = samples.merge(receptors, on=key, how="left")

    times = np.concatenate(
        [_epoch_s(receptors["start"]), _epoch_s(receptors["end"])]
        + [_epoch_s(e["start"]) for e in emissions.values()]
        + [_epoch_s(e["end"]) for e in emissions.values()]
    )
    bin_s = choose_time_bin(times)

    # One footprint grid per source site.
    site_of = {}
    for n in names:
        locs = emissions[n][["lat", "lon"]].drop_duplicates()
        if len(locs) != 1:
            raise ValueError(f"{n}: releases from more than one site are not supported")
        site_of[n] = (float(locs["lat"].iloc[0]), float(locs["lon"].iloc[0]))
    sites: list[SitePlan] = []
    for lat, lon in sorted(set(site_of.values())):
        exps = [n for n in names if site_of[n] == (lat, lon)]
        key_s = f"site_{lat:+.2f}_{lon:+.2f}".replace("+", "p").replace("-", "m").replace(".", "")
        rid = samples.loc[samples["experiment"].isin(exps), "receptor_id"].unique()
        rec = receptors[receptors["receptor_id"].isin(rid)].reset_index(drop=True)
        first = pd.Timestamp(min(emissions[n]["start"].min() for n in exps))
        site = SitePlan(key_s, lat, lon, exps, rec, first)
        for k, (part, length) in enumerate(
            chunk_receptors(rec, first, bin_s, met_start=met_start, max_span_h=max_span_h)
        ):
            site.chunks.append(Chunk(f"{key_s}/run{k:02d}", part, length))
        sites.append(site)
    samples["site_key"] = samples["experiment"].map(
        {n: s.key for s in sites for n in s.experiments}
    )
    return Plan(samples, emissions, receptors, sites, bin_s, layer)


# ---------------------------------------------------------------------------
# GLIDE configuration
# ---------------------------------------------------------------------------


def _set_dotted(cfg: dict, dotted: str, value) -> None:
    node = cfg
    parts = dotted.split(".")
    for p in parts[:-1]:
        node = node.setdefault(p, {})
    node[parts[-1]] = value


def parse_overrides(items: list[str]) -> dict[str, object]:
    out = {}
    for it in items or []:
        k, sep, v = it.partition("=")
        if not sep:
            raise SystemExit(f"--set expects key.path=value, got {it!r}")
        out[k.strip()] = yaml.safe_load(v)
    return out


def met_extent(store: str) -> dict:
    """Horizontal extent of a meteorology store as GLIDE's reader presents it, and
    its met-bazarr provenance attributes.

    The longitudes come from ``ArcoEra5ZarrReader.get_lon_lat_coverage``, after
    the reader's own normalisation (an ascending 0..360 store west of Greenwich,
    such as met-bazarr's NAM stores, is relabelled -180..180; PR #56). The
    runtime's preflight checks the met domain against that same extent, so the
    benchmark keeps no longitude convention rule of its own.
    """

    import xarray as xr

    from lpdm.met_reader import ArcoEra5ZarrReader

    reader = ArcoEra5ZarrReader(zarr_store=store, device="cpu")
    try:
        lon0, lon1, lat0, lat1 = reader.get_lon_lat_coverage()
    finally:
        reader.close()
    ds = xr.open_zarr(store)
    keys = (
        "producer",
        "producer_version",
        "producer_commit",
        "compression",
        "source_dataset",
        "glide_format_version",
        "glide_vertical_coordinate",
        "recipe",
        "glide_repairs",
    )
    prov = {k: str(ds.attrs[k]) for k in keys if k in ds.attrs}
    t = ds["time"].values
    return dict(
        lat=(lat0, lat1),
        lon=(lon0, lon1),
        time=(pd.Timestamp(t[0]), pd.Timestamp(t[-1])),
        provenance=prov,
    )


def lon_in_extent(lon: float, extent: tuple[float, float]) -> float:
    """``lon`` written the way it falls inside the reader's longitude ``extent``
    (as given, or shifted by 360 degrees); a point outside it is returned as
    given, for the runtime's preflight to refuse."""

    lo, hi = extent
    for cand in (float(lon), float(lon) - 360.0, float(lon) + 360.0):
        if lo <= cand <= hi:
            return cand
    return float(lon)


def glide_config(
    base: dict,
    site: SitePlan,
    chunk: Chunk,
    plan: Plan,
    *,
    met: str,
    met_box: dict,
    output_uri: str,
    particles: int,
    seed: int,
    source_deg: float,
    source_layer_m: float,
    overrides: dict[str, object],
) -> dict:
    """The GLIDE configuration for one site: the base physics, the sample releases,
    a 3 × 3 footprint grid of ``source_deg`` cells centred on the source, one layer."""

    cfg = json.loads(json.dumps(base))  # deep copy
    for k, v in overrides.items():
        _set_dotted(cfg, k, v)
    rec = chunk.receptors
    lo, hi = plan.layer
    cfg["io"] = {"zarr_store": met, "output_uri": output_uri}
    sim = cfg.setdefault("simulation", {})
    sim["start_time"] = rec["start"].min().strftime("%Y-%m-%dT%H:%M:%SZ")
    sim["length_seconds"] = int(chunk.length_s)
    cfg["release"] = {
        "kind": "samples",
        "n_particles_per_release": int(particles),
        "seed": int(seed),
        "samples": [
            {
                "name": r.receptor_id,
                "lon": round(lon_in_extent(r.lon, met_box["lon"]), 6),
                "lat": round(float(r.lat), 6),
                "start_time": r.start.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "duration_seconds": int((r.end - r.start).total_seconds()),
                "alt_min_agl_m": float(lo),
                "alt_max_agl_m": float(hi),
            }
            for r in rec.itertuples()
        ],
    }
    h = 1.5 * source_deg
    n_bins = int(math.ceil(chunk.length_s / plan.time_bin_s))
    site_lon = lon_in_extent(site.lon, met_box["lon"])
    # Update, not replace: the geometry is the benchmark's, but any other output
    # setting given in the base or with --set (e.g. output_grid.density_weighting)
    # must survive.
    cfg.setdefault("output_grid", {}).update(
        {
            "lon_bounds": [site_lon - h, site_lon + h],
            "lat_bounds": [site.lat - h, site.lat + h],
            "n_x": 3,
            "n_y": 3,
            "z_edges_m": [0.0, float(source_layer_m)],
            "n_time_bins": n_bins,
            "time_bin_seconds": int(plan.time_bin_s),
        }
    )
    md = cfg.setdefault("met_domain", {})
    if "lon_bounds" not in md or md.get("lon_bounds") == "store":
        md["lon_bounds"] = [met_box["lon"][0] + 0.5, met_box["lon"][1] - 0.5]
    if "lat_bounds" not in md or md.get("lat_bounds") == "store":
        md["lat_bounds"] = [met_box["lat"][0] + 0.5, met_box["lat"][1] - 0.5]
    return cfg


def check_basis(cfg: dict, basis: str) -> str:
    """The footprint weighting ``cfg`` will produce, refused if ``basis`` cannot use it.

    An ambient concentration needs plain residence seconds
    (``output_grid.density_weighting: none``); the default ``dry_air`` store
    serves the "stp" basis. Checked before GLIDE runs, so a mismatch costs
    nothing."""

    from lpdm.config import RunConfig

    weighting = RunConfig.model_validate(cfg).output_grid.density_weighting
    if basis == "ambient" and weighting != "none":
        raise SystemExit(
            f"basis 'ambient' needs an unweighted footprint, but output_grid.density_weighting "
            f"is {weighting!r}; add --set output_grid.density_weighting=none"
        )
    return weighting


def config_hash(cfg: dict) -> str:
    """SHA-256 of the configuration's physics and numerics: everything except the
    paths, the release list and the output grid's geometry (which the benchmark
    derives). Other output settings, such as ``density_weighting``, count."""

    core = {k: v for k, v in cfg.items() if k not in ("io", "release", "output_grid")}
    og = {k: v for k, v in cfg.get("output_grid", {}).items() if k not in OUTPUT_GEOMETRY}
    if og:
        core["output_grid"] = og
    core = json.loads(json.dumps(core, sort_keys=True, default=str))
    core.get("simulation", {}).pop("start_time", None)
    core.get("simulation", {}).pop("length_seconds", None)
    return hashlib.sha256(json.dumps(core, sort_keys=True).encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Footprints -> predictions
# ---------------------------------------------------------------------------


def site_predictions(
    fp_store: Path,
    site: SitePlan,
    plan: Plan,
    *,
    source_deg: float,
    basis: str,
    met: str | None,
) -> pd.DataFrame:
    """Predicted concentration (kg m-3) for every (experiment, receptor) of one site."""

    import xarray as xr

    ds = xr.open_zarr(fp_store)
    density = None
    if basis == "stp" and not concentration.is_density_weighted(ds):
        density = concentration.met_source_density(met, site.lat, site.lon)
    names = [str(s) for s in ds["site"].values]
    rows = []
    for n in site.experiments:
        ems = [
            concentration.Emission(r.start, r.end, r.mass, r.lat, r.lon)
            for r in plan.emissions[n].itertuples()
        ]
        c = concentration.predict_from_dataset(
            ds,
            ems,
            source_deg=source_deg,
            basis=basis,
            density=density,
            molar_volume_m3_mol=EXPERIMENTS[n].molar_volume_m3_mol,
        )
        rows.append(pd.DataFrame({"experiment": n, "receptor_id": names, "pred": c}))
    return pd.concat(rows, ignore_index=True)


def write_predictions(plan: Plan, pred: pd.DataFrame, out: Path) -> pd.DataFrame:
    """Attach predictions to samples and write one DATEM-format file per experiment."""

    s = plan.samples.merge(pred, on=["experiment", "receptor_id"], how="left")
    if s["pred"].isna().any():
        raise RuntimeError("some samples have no prediction")
    s["pred_raw"] = s["pred"] / 1.0e-15  # kg m-3 -> pg m-3, DATEM's unit
    (out / "predictions").mkdir(parents=True, exist_ok=True)
    from .readers import write_datem_predictions

    for n, g in s.groupby("experiment", sort=False):
        write_datem_predictions(
            g, g["pred_raw"].to_numpy(), out / "predictions" / f"{n}.txt", title=f"GLIDE {n}"
        )
    cols = ["experiment", "sample_id", "receptor_id", "site", "lat", "lon", "start", "end"]
    s[cols + ["conc_raw", "pred_raw"]].to_csv(out / "predictions.csv", index=False)
    return s


# ---------------------------------------------------------------------------
# Scoring and scorecard
# ---------------------------------------------------------------------------


#: The published protocol (published-tracer-scores.md §2; B10 handoff). Provisional
#: until the pre-registration (A8b) fixes it.
def protocol_options(seed: int, bootstrap: bool = True) -> datem_stats.StatOptions:
    return datem_stats.StatOptions(
        zero_percentile=0.0,
        exclude_zero_pairs=False,
        averaging=None,
        bootstrap=bootstrap,
        seed=seed if bootstrap else None,
    )


def score_experiment(samples: pd.DataFrame, pred_file: Path, seed: int) -> dict:
    exp = samples["experiment"].iloc[0]
    meas = samples.drop(columns=[c for c in samples.columns if c in ("pred", "pred_raw")])
    pred = read_datem_meas(pred_file, experiment=EXPERIMENTS[exp].datem_dir)
    st = datem_stats.score(meas, pred, protocol_options(seed, True))
    plain = datem_stats.score(meas, pred, protocol_options(seed, False))
    t = st.rank_terms
    return dict(
        experiment=exp,
        n_pairs=st.n_pairs,
        rank=st.rank,
        r2_term=t["r2"],
        fb_term=t["fb"],
        fms_term=t["fms"],
        ks_term=t["ks"],
        rank_plain_r=plain.rank,
        r_plain=plain.correlation,
        fb=st.fractional_bias,
        fms=st.fms,
        ksp=st.ksp,
        mean_measured_pg_m3=st.mean_measured,
        mean_calculated_pg_m3=st.mean_calculated,
    )


def reference_scores(names: list[str], seed: int, root: Path | None = None) -> list[dict]:
    """DATEM's own HYSPLIT example (``mdl_data/<exp>/ct<t>_000.txt``) scored on the
    same samples and options, where it exists: a reference computed, not transcribed."""

    root = Path(root) if root else datem_root()
    out = []
    for n in names:
        exp = EXPERIMENTS[n]
        f = root / "mdl_data" / exp.datem_dir / f"ct{exp.tracer}_000.txt"
        if not f.is_file():
            continue
        s, _ = load_experiment(n, root)
        r = score_experiment(s, f, seed)
        r.update(
            source="DATEM archive mdl_data (HYSPLIT run 007), scored here",
            model="HYSPLIT",
            mode="forward",
            meteorology="not documented",
            met_class="DATEM example",
        )
        out.append(r)
    return out


def published_rows(names: list[str]) -> pd.DataFrame:
    if not PUBLISHED_CSV.is_file():
        return pd.DataFrame()
    p = pd.read_csv(PUBLISHED_CSV)
    return p[p["experiment"].isin(names)].reset_index(drop=True)


def _git(*args: str, cwd: Path = REPO_ROOT) -> str:
    try:
        return subprocess.run(
            ["git", *args], cwd=cwd, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def provenance(cfg: dict, met: str, met_box: dict, args_dict: dict) -> dict:
    import lpdm

    lpdm_root = Path(lpdm.__file__).resolve().parents[2]
    tag = _git("describe", "--tags", "--match", "v*-physics", "--abbrev=0", cwd=lpdm_root)
    # Only the code that makes the numbers counts: the model source and the scoring package.
    dirty = _git(
        "status",
        "--porcelain",
        "--untracked-files=no",
        "--",
        "src",
        "scripts/validation/tracers",
        cwd=lpdm_root,
    )
    return dict(
        physics_tag=tag if tag != "unknown" else "none (pre-freeze)",
        glide_commit=_git("rev-parse", "HEAD", cwd=lpdm_root),
        glide_tree_dirty=bool(dirty and dirty != "unknown"),
        glide_source=str(lpdm_root),
        scoring_commit=_git("log", "-1", "--format=%H", "--", "scripts/validation/tracers"),
        config_hash=config_hash(cfg),
        met_store=str(met),
        met_provenance=met_box["provenance"],
        **args_dict,
    )


def build_scorecard(
    out: Path, scores: list[dict], refs: list[dict], meta: dict, names: list[str]
) -> None:
    glide = pd.DataFrame(scores)
    glide.insert(0, "source", "GLIDE (this run)")
    glide.insert(1, "model", "GLIDE")
    glide.insert(
        2, "mode", f"backward, {meta['receptor_layer_m'][0]:g}-{meta['receptor_layer_m'][1]:g} m"
    )
    glide.insert(3, "meteorology", Path(meta["met_store"]).name)
    glide.insert(4, "met_class", "this run")
    pub = published_rows(names)
    ref = pd.DataFrame(refs)
    table = pd.concat([glide, ref, pub], ignore_index=True)
    cols = [
        "source",
        "model",
        "mode",
        "meteorology",
        "met_class",
        "experiment",
        "n_pairs",
        "rank",
        "r2_term",
        "fb_term",
        "fms_term",
        "ks_term",
        "rank_plain_r",
        "r_plain",
        "fb",
        "fms",
        "ksp",
        "mean_measured_pg_m3",
        "mean_calculated_pg_m3",
    ]
    table = table.reindex(columns=cols)

    # Mean over the published set, when the run scored all eight experiments.
    means = []
    if set(PUBLISHED_SET) <= set(glide["experiment"]):
        g = glide[glide["experiment"].isin(PUBLISHED_SET)]
        means.append(
            dict(source="GLIDE (this run)", met_class="this run", mean_rank=g["rank"].mean())
        )
        if not pub.empty:
            for (_src, model, metn, cls), grp in pub.groupby(
                ["source", "model", "meteorology", "met_class"]
            ):
                if set(PUBLISHED_SET) <= set(grp["experiment"]):
                    means.append(
                        dict(
                            source=f"{model}, {metn}",
                            met_class=cls,
                            mean_rank=grp[grp["experiment"].isin(PUBLISHED_SET)]["rank"].mean(),
                        )
                    )
    table.to_csv(out / "scorecard.csv", index=False)
    (out / "scorecard.json").write_text(
        json.dumps(
            dict(version=SCORECARD_VERSION, meta=meta, scores=scores, references=refs, means=means),
            indent=1,
            default=str,
        )
    )

    L = [f"# Tracer benchmark scorecard: {', '.join(names)}", ""]
    if not any(EXPERIMENTS[n].graded for n in names):
        L += [
            "_Exploratory: run before the pre-registration on experiments that are not graded "
            "(validation-plan §3, development history)._",
            "",
        ]
    L += ["| Item | Value |", "| --- | --- |"]
    for k in (
        "created",
        "physics_tag",
        "glide_commit",
        "glide_tree_dirty",
        "scoring_commit",
        "config_hash",
        "config_overrides",
        "met_store",
        "release_seed",
        "bootstrap_seed",
        "particles_per_sample",
        "receptor_layer_m",
        "source_deg",
        "source_layer_m",
        "time_bin_seconds",
        "basis",
        "glide_wall_seconds",
    ):
        L.append(f"| {k} | `{meta.get(k)}` |")
    for k, v in meta.get("met_provenance", {}).items():
        if k in ("producer", "producer_version", "producer_commit", "compression", "recipe"):
            L.append(f"| met {k} | `{v}` |")
    L.append(
        f"| met repairs | `{'yes' if 'glide_repairs' in meta.get('met_provenance', {}) else 'none recorded'}` |"
    )
    L += [
        "",
        "## Scores",
        "",
        "Protocol: `-z0`, zero–zero pairs kept, unaveraged, R² by DATEM's resampling (`-b`, "
        "seeded). `rank (plain R)` uses the plain correlation.",
        "",
    ]
    L += [
        "| Experiment | Pairs | Rank | R² | 1−\\|FB/2\\| | FMS/100 | 1−KS/100 | Rank (plain R) | Mean meas. | Mean calc. (pg m⁻³) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in scores:
        L.append(
            f"| {r['experiment']} | {r['n_pairs']} | {r['rank']:.3f} | {r['r2_term']:.3f} | "
            f"{r['fb_term']:.3f} | {r['fms_term']:.3f} | {r['ks_term']:.3f} | {r['rank_plain_r']:.3f} | "
            f"{r['mean_measured_pg_m3']:.4g} | {r['mean_calculated_pg_m3']:.4g} |"
        )
    L += ["", "## Beside other models", ""]
    if refs:
        L += [
            "DATEM's HYSPLIT example, scored here on the same samples and options "
            "(whole experiment; not a per-release published value):",
            "",
            "| Experiment | Rank | R² | 1−\\|FB/2\\| | FMS/100 | 1−KS/100 |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
        for r in refs:
            L.append(
                f"| {r['experiment']} | {r['rank']:.3f} | {r['r2_term']:.3f} | {r['fb_term']:.3f} | {r['fms_term']:.3f} | {r['ks_term']:.3f} |"
            )
        L.append("")
    if pub.empty:
        L.append(
            "No published per-release values for these experiments "
            "(`docs/published-tracer-scores.csv`)."
        )
    else:
        L += [
            "Published per-release ranks by class of meteorology "
            "(`docs/published-tracer-scores.csv`):",
            "",
            "| Experiment | GLIDE | Class | Runs | Min | Max | Mean |",
            "| --- | ---: | --- | ---: | ---: | ---: | ---: |",
        ]
        g_rank = {r["experiment"]: r["rank"] for r in scores}
        for (e, cls), grp in pub.groupby(["experiment", "met_class"], sort=False):
            L.append(
                f"| {e} | {g_rank.get(e, float('nan')):.2f} | {cls} | {len(grp)} | "
                f"{grp['rank'].min():.2f} | {grp['rank'].max():.2f} | {grp['rank'].mean():.2f} |"
            )
    if means:
        L += [
            "",
            "Mean over the eight published experiments:",
            "",
            "| Run | Class | Mean rank |",
            "| --- | --- | ---: |",
        ]
        for m in means:
            L.append(f"| {m['source']} | {m['met_class']} | {m['mean_rank']:.2f} |")
    (out / "scorecard.md").write_text("\n".join(L) + "\n")


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def _load_base(path: Path | None) -> dict:
    with open(path or BASE_CONFIG) as fh:
        return yaml.safe_load(fh)


def cmd_plan(args) -> int:
    plan = build_plan(
        args.experiments, layer=tuple(args.receptor_layer), max_span_h=args.max_span_h
    )
    print(plan.samples.groupby("experiment").size().to_string())
    print(f"{len(plan.receptors)} unique sample windows; age bins of {plan.time_bin_s} s")
    for s in plan.sites:
        print(f"{s.key}: {', '.join(s.experiments)}; {len(s.receptors)} releases")
        for ch in s.chunks:
            print(
                f"  {ch.key}: {len(ch.receptors)} releases, {ch.receptors['start'].min()} to "
                f"{ch.receptors['end'].max()}, run length {ch.length_s / 3600:.2f} h"
            )
    return 0


def _score_run(out: Path, plan: Plan, meta: dict, *, met: str, basis: str) -> list[dict]:
    """Predictions from every GLIDE run under ``out/glide``, then the scores."""

    preds = []
    for s in plan.sites:
        stores = sorted((out / "glide" / s.key).glob("run*/footprints.zarr"))
        if not stores:
            raise SystemExit(f"no footprints for {s.key} under {out / 'glide'}")
        for store in stores:
            preds.append(
                site_predictions(
                    store, s, plan, source_deg=meta["source_deg"], basis=basis, met=met
                )
            )
    pred = pd.concat(preds, ignore_index=True)
    if pred.duplicated(["experiment", "receptor_id"]).any():
        raise RuntimeError("a sample window appears in more than one GLIDE run")
    s_all = write_predictions(plan, pred, out)
    scores = []
    for n, g in s_all.groupby("experiment", sort=False):
        scores.append(score_experiment(g, out / "predictions" / f"{n}.txt", meta["bootstrap_seed"]))
    return scores


#: What scoring writes into a run directory; published together or not at all.
SCORE_OUTPUTS = (
    "run_meta.json",
    "predictions",
    "predictions.csv",
    "scorecard.csv",
    "scorecard.json",
    "scorecard.md",
)


def score_and_publish(out: Path, glide_dir: Path, plan: Plan, meta: dict, names: list[str]) -> None:
    """Score ``glide_dir``'s footprints with ``meta`` and publish to ``out``.

    Everything is written first to a temporary directory beside ``out`` (whose
    ``glide`` links to ``glide_dir``); only when scoring has succeeded are the
    metadata, predictions and scorecard moved into ``out``, together. A failure
    leaves ``out`` exactly as it was, so a scorecard never sits beside metadata
    that describes another scoring.
    """

    import shutil
    import tempfile

    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=f".{out.name}.scoring-", dir=out.parent))
    try:
        (tmp / "glide").symlink_to(glide_dir.resolve(), target_is_directory=True)
        (tmp / "run_meta.json").write_text(json.dumps(meta, indent=1, default=str))
        scores = _score_run(tmp, plan, meta, met=meta["met_store"], basis=meta["basis"])
        build_scorecard(tmp, scores, reference_scores(names, meta["bootstrap_seed"]), meta, names)
        out.mkdir(parents=True, exist_ok=True)
        for name in SCORE_OUTPUTS:
            src, dst = tmp / name, out / name
            if not src.exists():
                continue
            if dst.is_dir() and not dst.is_symlink():
                old = tmp / f"{name}.old"
                dst.rename(old)
            os.replace(src, dst)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _check_glide_link(out: Path, run: Path) -> None:
    """``out/glide``, if present, must be the requested run's footprints."""

    link = out / "glide"
    if (link.exists() or link.is_symlink()) and link.resolve() != (run / "glide").resolve():
        raise SystemExit(
            f"{out} already holds footprints from {link.resolve()}, not from {run / 'glide'}; "
            "choose another --out"
        )


def cmd_run(args) -> int:
    names = list(args.experiments)
    check_independence(names, args.registration_record)
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    base = _load_base(args.config)
    overrides = parse_overrides(args.set)
    met = str(args.met)
    box = met_extent(met)
    plan = build_plan(
        names,
        layer=tuple(args.receptor_layer),
        met_start=box["time"][0],
        max_span_h=args.max_span_h,
    )
    for s in plan.sites:
        if s.receptors["end"].max() > box["time"][1]:
            raise SystemExit(f"{met} ends before {s.key}'s last sample")
    plan.samples.to_csv(out / "samples.csv", index=False)
    plan.receptors.to_csv(out / "receptors.csv", index=False)

    wall = 0.0
    cfgs = {}
    for s in plan.sites:
        for ch in s.chunks:
            cfg = glide_config(
                base,
                s,
                ch,
                plan,
                met=met,
                met_box=box,
                output_uri=str(out / "glide" / ch.key),
                particles=args.particles,
                seed=args.seed,
                source_deg=args.source_deg,
                source_layer_m=args.source_layer_m,
                overrides=overrides,
            )
            check_basis(cfg, args.basis)
            cfgs[ch.key] = cfg
            tag = ch.key.replace("/", "_")
            path = out / "configs" / f"{tag}.yaml"
            path.parent.mkdir(exist_ok=True)
            path.write_text(yaml.safe_dump(cfg, sort_keys=False))
            if args.no_run:
                continue
            (out / "logs").mkdir(exist_ok=True)
            t0 = time.time()
            with open(out / "logs" / f"{tag}.log", "w") as log:
                r = subprocess.run(
                    [sys.executable, "-m", "lpdm.main", "--config", str(path)],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env=dict(os.environ),
                )
            dt = time.time() - t0
            wall += dt
            print(f"{ch.key}: GLIDE exit {r.returncode} after {dt:.0f} s", flush=True)
            if r.returncode != 0:
                raise SystemExit(f"GLIDE failed for {ch.key}; see {out / 'logs' / f'{tag}.log'}")
    if args.no_run:
        print(f"configurations written to {out / 'configs'}; GLIDE not run")
        return 0

    first = next(iter(cfgs.values()))
    meta = provenance(
        first,
        met,
        box,
        dict(
            created=datetime.now(UTC).isoformat(timespec="seconds"),
            experiments=names,
            config_base=str(args.config or BASE_CONFIG),
            config_overrides=overrides,
            config_hashes_all_runs=sorted({config_hash(c) for c in cfgs.values()}),
            release_seed=args.seed,
            bootstrap_seed=args.bootstrap_seed,
            particles_per_sample=args.particles,
            receptor_layer_m=list(args.receptor_layer),
            source_deg=args.source_deg,
            source_layer_m=args.source_layer_m,
            time_bin_seconds=plan.time_bin_s,
            met_lon_extent=list(box["lon"]),
            max_span_h=args.max_span_h,
            basis=args.basis,
            glide_wall_seconds=round(wall, 1),
            glide_runs={
                ch.key: dict(releases=len(ch.receptors), length_s=ch.length_s)
                for s in plan.sites
                for ch in s.chunks
            },
        ),
    )
    score_and_publish(out, out / "glide", plan, meta, names)
    print((out / "scorecard.md").read_text())
    return 0


def cmd_score(args) -> int:
    """Re-score an existing run (e.g. on the other concentration basis)."""

    run = Path(args.run).resolve()
    meta = json.loads((run / "run_meta.json").read_text())
    names = meta["experiments"]
    plan = build_plan(names, layer=tuple(meta["receptor_layer_m"]), max_span_h=meta["max_span_h"])
    out = Path(args.out).resolve() if args.out else run
    if out != run:
        _check_glide_link(out, run)
    meta = dict(meta, basis=args.basis or meta["basis"])
    if args.bootstrap_seed is not None:
        meta["bootstrap_seed"] = args.bootstrap_seed
    # Nothing in `out` changes unless scoring succeeds (score_and_publish).
    score_and_publish(out, run / "glide", plan, meta, names)
    if out != run and not (out / "glide").is_symlink() and not (out / "glide").exists():
        (out / "glide").symlink_to(run / "glide")
    print((out / "scorecard.md").read_text())
    return 0


def compare_runs(runs: list[Path]) -> str:
    """Two or more scorecards side by side: scores, differences from the first run,
    the provenance that differs, and whether the predictions are identical."""

    cards = [json.loads((Path(r) / "scorecard.json").read_text()) for r in runs]
    labels = [f"run {i + 1}" for i in range(len(runs))]
    L = ["# Tracer benchmark comparison", ""]
    L += ["| Run | Directory |", "| --- | --- |"]
    L += [f"| {lab} | `{r}` |" for lab, r in zip(labels, runs, strict=True)]
    keys = sorted(
        {k for c in cards for k in c["meta"]} - {"created", "glide_wall_seconds", "sites"}
    )
    diff = [
        k
        for k in keys
        if len({json.dumps(c["meta"].get(k), sort_keys=True, default=str) for c in cards}) > 1
    ]
    L += ["", "## Settings that differ", ""]
    if diff:
        L += ["| Setting | " + " | ".join(labels) + " |", "| --- |" + " --- |" * len(cards)]
        for k in diff:
            L.append(f"| {k} | " + " | ".join(f"`{c['meta'].get(k)}`" for c in cards) + " |")
    else:
        L.append("None (apart from creation time and wall time).")
    L += ["", "## Scores", ""]
    terms = ["rank", "r2_term", "fb_term", "fms_term", "ks_term", "rank_plain_r"]
    L += [
        "| Experiment | Quantity | "
        + " | ".join(labels)
        + " | "
        + " | ".join(f"Δ {lab}" for lab in labels[1:])
        + " |",
        "| --- | --- |" + " ---: |" * (2 * len(cards) - 1),
    ]
    exps = [s["experiment"] for s in cards[0]["scores"]]
    for e in exps:
        rows = [next((s for s in c["scores"] if s["experiment"] == e), None) for c in cards]
        for t in terms:
            vals = [r[t] if r else float("nan") for r in rows]
            d = [v - vals[0] for v in vals[1:]]
            L.append(
                f"| {e} | {t} | "
                + " | ".join(f"{v:.4f}" for v in vals)
                + " | "
                + " | ".join(f"{x:+.4f}" for x in d)
                + " |"
            )
    L += ["", "## Predictions", ""]
    p0 = pd.read_csv(Path(runs[0]) / "predictions.csv")
    for lab, r in zip(labels[1:], runs[1:], strict=True):
        p = pd.read_csv(Path(r) / "predictions.csv")
        m = p0.merge(p, on=["experiment", "sample_id"], suffixes=("_a", "_b"))
        same = bool((m["pred_raw_a"] == m["pred_raw_b"]).all()) and len(m) == len(p0) == len(p)
        rel = (m["pred_raw_b"] - m["pred_raw_a"]).abs().max()
        L.append(
            f"- {lab} against run 1: {len(m)} samples matched; "
            f"{'identical predictions' if same else f'max abs difference {rel:.4g} pg m-3'}."
        )
    return "\n".join(L) + "\n"


def cmd_compare(args) -> int:
    text = compare_runs([Path(r) for r in args.runs])
    if args.out:
        Path(args.out).write_text(text)
    print(text)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="tracers.benchmark", description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--experiments", nargs="+", default=["OKC80"], choices=sorted(EXPERIMENTS))
        p.add_argument("--config", type=Path, default=None, help="base GLIDE YAML (physics)")
        p.add_argument(
            "--receptor-layer",
            nargs=2,
            type=float,
            default=[0.0, 100.0],
            metavar=("BOTTOM", "TOP"),
            help="release layer in m above ground",
        )
        p.add_argument(
            "--max-span-h",
            type=float,
            default=24.0,
            help="longest span of sample starts in one GLIDE run",
        )

    p = sub.add_parser("plan")
    common(p)
    r = sub.add_parser("run")
    common(r)
    r.add_argument("--met", required=True, help="GLIDE meteorology store")
    r.add_argument("--out", required=True)
    r.add_argument("--set", action="append", default=[], help="override, key.path=value")
    r.add_argument("--particles", type=int, default=2000, help="particles per sample")
    r.add_argument("--seed", type=int, default=42, help="release seed (per sample: seed + index)")
    r.add_argument("--bootstrap-seed", type=int, default=0, help="seed for the -b resampling")
    r.add_argument("--source-deg", type=float, default=0.25, help="source box size (degrees)")
    r.add_argument("--source-layer-m", type=float, default=40.0, help="footprint layer depth")
    r.add_argument("--basis", choices=concentration.BASES, default="stp")
    r.add_argument("--registration-record", type=Path, default=None)
    r.add_argument("--no-run", action="store_true", help="write the configurations only")
    s = sub.add_parser("score")
    s.add_argument("run")
    s.add_argument("--basis", choices=concentration.BASES, default=None)
    s.add_argument("--bootstrap-seed", type=int, default=None)
    s.add_argument("--out", default=None)
    c = sub.add_parser("compare")
    c.add_argument("runs", nargs="+")
    c.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    return {"plan": cmd_plan, "run": cmd_run, "score": cmd_score, "compare": cmd_compare}[args.cmd](
        args
    )


if __name__ == "__main__":
    sys.exit(main())
