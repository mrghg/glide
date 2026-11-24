"""Readers for the DATEM and JRC ETEX tracer files, and a DATEM-format writer.

Every reader returns a :class:`pandas.DataFrame` in one of two common layouts.

Releases (one row per release):
    ``release`` (1-based, file order), ``start`` and ``end`` (UTC, naive
    ``datetime64``), ``duration_s``, ``lat``, ``lon`` (degrees, east
    positive), ``height_m`` (NaN when the file does not give it), ``mass``
    (SI: kg, or Bq for Kr-85), ``mass_unit``, ``mass_raw`` (file units).

Samples (one row per measurement):
    ``sample_id`` (``<exp>-t<tracer>-<row>`` for DATEM, ``<code>-<interval>``
    for JRC), ``site`` (string), ``start``, ``end``, ``duration_s``, ``lat``,
    ``lon``, ``height_m``, ``conc`` (SI: kg m-3 or Bq m-3; NaN when not
    measured), ``conc_unit``, ``conc_raw`` (file units), ``quality_flag``
    (JRC flag; NaN for DATEM), and, for DATEM, the raw key fields
    ``yr, mo, dy, hhmm, dur_hhmm, site_num`` that ``statmain`` matches on.

Units: DATEM gives emissions in g (Ci for ACURATE) and concentrations in
pg m-3 (pCi m-3); JRC gives ng m-3.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

#: Repository root (scripts/validation/tracers/readers.py -> repo).
REPO_ROOT = Path(__file__).resolve().parents[3]

PG = 1.0e-15  # kg per pg
NG = 1.0e-12  # kg per ng
CURIE = 3.7e10  # Bq per Ci

#: Experiments whose amounts are radioactivity rather than mass.
_RADIOACTIVE = {"acurate"}

#: Which background each ACURATE measurement file uses (exp_data/acurate/readme.txt).
ACURATE_BACKGROUNDS = {
    0: "original background analysis (arl-080)",
    1: "variable background from a global model (Draxler, 2007)",
}


def datem_root() -> Path:
    """``$GLIDE_DATEM``, or ``data/DATEM`` in the checkout."""
    return Path(os.environ.get("GLIDE_DATEM") or REPO_ROOT / "data" / "DATEM")


def etex_root() -> Path:
    """``$GLIDE_ETEX``, or ``data/ETEX`` in the checkout."""
    return Path(os.environ.get("GLIDE_ETEX") or REPO_ROOT / "data" / "ETEX")


# ---------------------------------------------------------------------------
# DATEM
# ---------------------------------------------------------------------------


def _hhmm_to_seconds(hhmm: np.ndarray) -> np.ndarray:
    hhmm = np.asarray(hhmm, dtype=np.int64)
    return (hhmm // 100) * 3600 + (hhmm % 100) * 60


def _read_datem_records(path: Path, ncol: int) -> pd.DataFrame:
    """Read a DATEM text file: two header lines, then whitespace-separated
    records (Fortran list-directed, as ``statmain`` reads them)."""
    rows = []
    with open(path) as fh:
        lines = fh.read().splitlines()
    for line in lines[2:]:
        parts = line.split()
        if not parts:
            continue
        if len(parts) < ncol:
            raise ValueError(f"{path}: short record {line!r}")
        rows.append(parts[:ncol])
    return rows


def _datem_times(df: pd.DataFrame) -> pd.DataFrame:
    base = pd.to_datetime(
        dict(year=df["yr"], month=df["mo"], day=df["dy"]), errors="raise"
    ) + pd.to_timedelta(_hhmm_to_seconds(df["hhmm"].to_numpy()), unit="s")
    df["start"] = base
    df["duration_s"] = _hhmm_to_seconds(df["dur_hhmm"].to_numpy()).astype(float)
    df["end"] = base + pd.to_timedelta(df["duration_s"], unit="s")
    return df


def read_datem_emit(path: str | Path, *, experiment: str | None = None) -> pd.DataFrame:
    """Read a DATEM ``emit-t?.txt`` into the release table (SI units)."""
    path = Path(path)
    experiment = (experiment or path.parent.name).lower()
    rows = _read_datem_records(path, 8)
    df = pd.DataFrame(rows, columns=["yr", "mo", "dy", "hhmm", "dur_hhmm", "lat", "lon", "raw"])
    for c in ("yr", "mo", "dy", "hhmm", "dur_hhmm"):
        df[c] = df[c].astype(int)
    for c in ("lat", "lon", "raw"):
        df[c] = df[c].astype(float)
    df = _datem_times(df)
    df.insert(0, "release", np.arange(1, len(df) + 1))
    df["height_m"] = np.nan
    if experiment in _RADIOACTIVE:
        df["mass"] = df["raw"] * CURIE
        df["mass_unit"] = "Bq"
    else:
        df["mass"] = df["raw"] * 1.0e-3  # g -> kg
        df["mass_unit"] = "kg"
    df = df.rename(columns={"raw": "mass_raw"})
    return df[
        [
            "release",
            "start",
            "end",
            "duration_s",
            "lat",
            "lon",
            "height_m",
            "mass",
            "mass_unit",
            "mass_raw",
        ]
    ]


def read_datem_meas(path: str | Path, *, experiment: str | None = None) -> pd.DataFrame:
    """Read a DATEM ``meas-t?.txt`` or ``ct?_*.txt`` into the sample table.

    ``conc_raw`` is in file units (pg m-3, or pCi m-3 for ACURATE) and
    ``conc`` in SI. The raw key fields used by ``statmain`` to pair
    predictions with measurements are kept.
    """
    path = Path(path)
    experiment = (experiment or path.parent.name).lower()
    rows = _read_datem_records(path, 9)
    df = pd.DataFrame(
        rows, columns=["yr", "mo", "dy", "hhmm", "dur_hhmm", "lat", "lon", "raw", "site"]
    )
    for c in ("yr", "mo", "dy", "hhmm", "dur_hhmm"):
        df[c] = df[c].astype(int)
    for c in ("lat", "lon", "raw"):
        df[c] = df[c].astype(float)
    df["site_num"] = df["site"].astype(int)
    df = _datem_times(df)
    stem = path.stem
    df.insert(0, "sample_id", [f"{experiment}-{stem}-{i}" for i in range(len(df))])
    df["height_m"] = np.nan
    df["conc_raw"] = df["raw"]
    if experiment in _RADIOACTIVE:
        df["conc"] = df["raw"] * 1.0e-12 * CURIE
        df["conc_unit"] = "Bq m-3"
    else:
        df["conc"] = df["raw"] * PG
        df["conc_unit"] = "kg m-3"
    df.loc[df["raw"] < 0, "conc"] = np.nan
    df["quality_flag"] = np.nan
    return df[
        [
            "sample_id",
            "site",
            "start",
            "end",
            "duration_s",
            "lat",
            "lon",
            "height_m",
            "conc",
            "conc_unit",
            "conc_raw",
            "quality_flag",
            "yr",
            "mo",
            "dy",
            "hhmm",
            "dur_hhmm",
            "site_num",
        ]
    ]


def read_acurate(root: str | Path | None = None) -> pd.DataFrame:
    """Read both ACURATE measurement files, tagged with their background.

    Returns the sample table with ``background_file`` (0 or 1) and
    ``background`` (the description from DATEM's readme).
    """
    d = Path(root) if root else datem_root() / "exp_data" / "acurate"
    out = []
    for k in (0, 1):
        df = read_datem_meas(d / f"meas-t{k}.txt", experiment="acurate")
        df["background_file"] = k
        df["background"] = ACURATE_BACKGROUNDS[k]
        out.append(df)
    return pd.concat(out, ignore_index=True)


def write_datem_predictions(
    samples: pd.DataFrame,
    predicted_raw,
    path: str | Path,
    *,
    title: str = "GLIDE predictions",
    species: str = "conc",
) -> None:
    """Write predictions in DATEM measurement format, one record per sample.

    ``samples`` is a table from :func:`read_datem_meas` (its raw key fields
    are written back unchanged so ``statmain`` pairs every record);
    ``predicted_raw`` is in the measurement file's units (pg m-3). The
    concentration is written with 6 significant figures rather than
    DATEM's ``F8.1``, so small values are not truncated; ``statmain`` reads
    the records list-directed, so either works.
    """
    pred = np.asarray(predicted_raw, dtype=float)
    if pred.shape != (len(samples),):
        raise ValueError("one prediction per sample is required")
    lines = [title, f"year mn dy  shr  dur   lat     lon    {species:>6s}  stn"]
    for (_, r), c in zip(samples.iterrows(), pred, strict=True):
        lines.append(
            f"{r.yr:4d} {r.mo:02d} {r.dy:02d} {r.hhmm:04d} {r.dur_hhmm:04d} "
            f"{r.lat:6.2f} {r.lon:8.2f} {c:12.6g} {r.site:>6s}"
        )
    Path(path).write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# JRC ETEX
# ---------------------------------------------------------------------------

#: Interval 1 start for each release (pmch.readme, pmcp.readmev1.2).
ETEX_INTERVAL_ORIGIN = {
    1: pd.Timestamp("1994-10-23 15:00"),
    2: pd.Timestamp("1994-11-14 15:00"),
}
#: Release table from ETEX_release?/release?.txt. Source coordinates are not
#: in the JRC files; they must come from the experiment documentation.
ETEX_RELEASES = {
    1: dict(
        start=pd.Timestamp("1994-10-23 16:00"),
        end=pd.Timestamp("1994-10-24 03:50"),
        mass=340.0,
        tracer="PMCH",
        height_m=8.0,
    ),
    2: dict(
        start=pd.Timestamp("1994-11-14 15:00"),
        end=pd.Timestamp("1994-11-15 02:45"),
        mass=490.0,
        tracer="PMCP",
        height_m=8.0,
    ),
}
_ETEX_FILES = {1: ("pmch.dat", "pmch.cod"), 2: ("pmcp.datv1.2", "pmcp.codv1.2")}

#: JRC missing-value codes (ng m-3).
ETEX_NOT_SAMPLED = -0.99
ETEX_NOT_QUANTIFIABLE = -0.88


def _deg_min(text: str) -> float:
    """JRC station coordinates are degrees.minutes (``-04.25`` is 4°25' W)."""
    s = text.strip()
    neg = s.startswith("-")
    deg, _, minutes = s.lstrip("+-").partition(".")
    val = int(deg) + int((minutes + "00")[:2]) / 60.0
    return -val if neg else val


def read_etex_stations(path: str | Path) -> pd.DataFrame:
    """Read ``stationlist.950130`` (code, name, lat, lon in degrees.minutes)."""
    rows = []
    for line in Path(path).read_text(errors="replace").splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 5 or parts[0] in ("cc",) or not parts[0][:1].isalpha():
            continue
        try:
            lat, lon = _deg_min(parts[2]), _deg_min(parts[3])
            alt = float(parts[4])
        except ValueError:
            continue
        rows.append(dict(site=parts[0], name=parts[1], lat=lat, lon=lon, alt_m=alt))
    return pd.DataFrame(rows)


def _read_etex_grid(path: Path, *, integer: bool) -> tuple[list[str], np.ndarray]:
    lines = path.read_text().splitlines()
    codes, values = [], []
    for line in lines[2:]:
        if not line.strip():
            continue
        code = line[6:10].strip()
        body = line[10:]
        if integer:
            vals = [int(body[6 * i : 6 * i + 6]) for i in range(30)]
        else:
            vals = [float(body[6 * i : 6 * i + 6]) for i in range(30)]
        codes.append(code)
        values.append(vals)
    return codes, np.asarray(values)


def read_etex_release(release: int = 1, root: str | Path | None = None) -> pd.DataFrame:
    """Read one JRC ETEX release table (one row; coordinates NaN, see module)."""
    r = ETEX_RELEASES[release]
    dur = (r["end"] - r["start"]).total_seconds()
    return pd.DataFrame(
        [
            dict(
                release=release,
                start=r["start"],
                end=r["end"],
                duration_s=dur,
                lat=np.nan,
                lon=np.nan,
                height_m=r["height_m"],
                mass=r["mass"],
                mass_unit="kg",
                mass_raw=r["mass"],
            )
        ]
    )


def read_etex_samples(release: int = 1, root: str | Path | None = None) -> pd.DataFrame:
    """Read the JRC ETEX concentrations and flags into the sample table.

    One row per station and 3-hour interval (``interval`` 1–30).
    ``conc`` is NaN for ``-.99`` (not sampled) and ``-.88`` (detected, not
    quantifiable); ``conc_raw`` keeps the code. ``quality_flag`` is the
    integer flag from the ``.cod`` file.
    """
    d = Path(root) if root else etex_root() / f"ETEX_release{release}"
    dat, cod = _ETEX_FILES[release]
    codes, conc = _read_etex_grid(d / dat, integer=False)
    codes2, flags = _read_etex_grid(d / cod, integer=True)
    if codes != codes2:
        raise ValueError("concentration and flag files list different stations")
    stations = read_etex_stations(d / "stationlist.950130").set_index("site")
    origin = ETEX_INTERVAL_ORIGIN[release]
    n_site, n_int = conc.shape
    site = np.repeat(codes, n_int)
    interval = np.tile(np.arange(1, n_int + 1), n_site)
    start = origin + pd.to_timedelta((interval - 1) * 3, unit="h")
    raw = conc.ravel()
    df = pd.DataFrame(
        dict(
            sample_id=[f"{s}-{i}" for s, i in zip(site, interval, strict=True)],
            site=site,
            interval=interval,
            start=start,
            end=start + pd.Timedelta(hours=3),
            duration_s=10800.0,
            lat=stations["lat"].reindex(site).to_numpy(),
            lon=stations["lon"].reindex(site).to_numpy(),
            height_m=np.nan,
            conc=np.where(raw >= 0, raw * NG, np.nan),
            conc_unit="kg m-3",
            conc_raw=raw,
            quality_flag=flags.ravel(),
        )
    )
    return df
