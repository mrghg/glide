"""ETEX-1: the JRC and DATEM versions matched sample by sample.

DATEM's ``exp_data/etex/meas-t1.txt`` is ETEX release 1 reformatted from
the ATMES data (``mea95t1.txt``), with numeric station codes and pg m-3.
:func:`match_jrc_datem` pairs every DATEM record with the JRC record of the
same station and interval, and classifies each difference.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from .readers import datem_root, read_datem_meas, read_etex_samples


def datem_to_jrc_code(site_num: int, prefixes: list[str]) -> str:
    """DATEM ETEX station number to JRC code.

    DATEM numbers the JRC country prefixes in alphabetical order: ``1xxx``
    to ``8xxx`` are A, B, BG, CH, CR, D, DK, F; ``9000``–``9099`` is H, and
    ``91xx`` to ``98xx`` are N, NL, PL, R, S, SF, SR, UK. The last two
    digits are the JRC station number (``6013`` is D13, ``9203`` NL03).
    """
    if site_num < 9000:
        idx, num = site_num // 1000, site_num % 1000
    else:
        idx, num = 9 + (site_num - 9000) // 100, site_num % 100
    return f"{prefixes[idx - 1]}{num:02d}"


def _code_class(v: float) -> str:
    if np.isnan(v):
        return "absent"
    if v == -0.99:
        return "not sampled (-.99)"
    if v == -0.88:
        return "not quantifiable (-.88)"
    return "zero" if v == 0 else "positive"


def match_jrc_datem(
    etex_dir: str | Path | None = None, datem_dir: str | Path | None = None
) -> pd.DataFrame:
    """One row per (station, interval) present in either file.

    Columns: ``jrc_site``, ``datem_site``, ``start``, ``jrc_ng`` (JRC value
    or code, ng m-3), ``jrc_flag``, ``datem_ng`` (DATEM value / 1000),
    ``jrc_class``, ``in_datem``, ``category`` (the explanation), and the
    station offsets ``dlat``, ``dlon`` (DATEM minus JRC, degrees).
    """
    j = read_etex_samples(1, etex_dir)
    dpath = Path(datem_dir) if datem_dir else datem_root() / "exp_data" / "etex"
    d = read_datem_meas(dpath / "meas-t1.txt", experiment="etex")
    prefixes = sorted({re.match(r"[A-Z]+", s).group() for s in j["site"].unique()})
    d["jrc_site"] = [datem_to_jrc_code(s, prefixes) for s in d["site_num"]]
    left = d[["jrc_site", "site_num", "start", "conc_raw", "lat", "lon"]].rename(
        columns={"site_num": "datem_site", "conc_raw": "datem_pg", "lat": "dlat_", "lon": "dlon_"}
    )
    right = j[["site", "start", "conc_raw", "quality_flag", "lat", "lon"]].rename(
        columns={"site": "jrc_site", "conc_raw": "jrc_ng", "quality_flag": "jrc_flag"}
    )
    m = right.merge(left, on=["jrc_site", "start"], how="outer")
    m["datem_ng"] = m.pop("datem_pg") / 1000.0
    m["dlat"] = m.pop("dlat_") - m["lat"]
    m["dlon"] = m.pop("dlon_") - m["lon"]
    m["in_datem"] = m["datem_ng"].notna()
    m["jrc_class"] = [_code_class(v) for v in m["jrc_ng"]]
    cat = np.full(len(m), "", dtype=object)
    jv = m["jrc_ng"].to_numpy()
    dv = m["datem_ng"].to_numpy()
    valid = m["jrc_class"].isin(["zero", "positive"]).to_numpy()
    ind = m["in_datem"].to_numpy()
    same = valid & ind & (np.abs(dv - jv) < 0.0005)
    cat[same] = "identical"
    cat[valid & ind & ~same] = "value differs"
    cat[valid & ~ind] = "valid in JRC, absent from DATEM"
    cat[~valid & ~ind & (m["jrc_class"] == "not sampled (-.99)")] = "not sampled; absent from DATEM"
    cat[~valid & ~ind & (m["jrc_class"] == "not quantifiable (-.88)")] = (
        "not quantifiable (-.88); absent from DATEM"
    )
    cat[~valid & ind & (m["jrc_class"] != "absent")] = "JRC code, value in DATEM"
    cat[(m["jrc_class"] == "absent").to_numpy()] = "DATEM only"
    m["category"] = cat
    return m.sort_values(["jrc_site", "start"]).reset_index(drop=True)


def summarise_match(m: pd.DataFrame) -> pd.DataFrame:
    return m.groupby("category").size().rename("n").reset_index()
