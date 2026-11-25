"""Tests for the tracer readers and scoring (scripts/validation/tracers, task B10).

Synthetic files only, apart from the tests marked ``needs_datem`` /
``needs_etex``, which reproduce DATEM's posted statistics and skip cleanly
when ``$GLIDE_DATEM`` or ``$GLIDE_ETEX`` is absent.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "validation"))

from tracers import atmes2, datem_stats, etex, experiments, readers  # noqa: E402
from tracers.datem_stats import StatOptions  # noqa: E402


def _datem_dir() -> Path | None:
    d = os.environ.get("GLIDE_DATEM")
    p = Path(d) if d else readers.REPO_ROOT / "data" / "DATEM"
    return p if (p / "mdl_stat" / "captex" / "statA1_000.txt").exists() else None


def _etex_dir() -> Path | None:
    d = os.environ.get("GLIDE_ETEX")
    p = Path(d) if d else readers.REPO_ROOT / "data" / "ETEX"
    return p if (p / "ETEX_release1" / "pmch.dat").exists() else None


needs_datem = pytest.mark.skipif(_datem_dir() is None, reason="DATEM data not available")
needs_etex = pytest.mark.skipif(_etex_dir() is None, reason="JRC ETEX data not available")

# ---------------------------------------------------------------------------
# synthetic files
# ---------------------------------------------------------------------------

EMIT = """\\datem\\exp_data\\test\\emit-t1.txt
year mn dy  shr  dur   lat     lon    pmch
1983 09 18 1700 0300 39.80  -84.05  208000
1983 09 20 0530 0030 46.62  -80.78   32000
"""

MEAS = """\\datem\\exp_data\\test\\meas-t1.txt
year mn dy  shr  dur   lat     lon    pmch  stn
1983 09 18 1800 0600 39.65  -80.42      .0  302
1983 09 18 1800 0600 40.38  -80.63    10.0  306
1983 09 19 0000 0600 39.65  -80.42    20.0  302
1983 09 19 0000 0300 40.38  -80.63    40.0  306
1983 09 20 0600 0600 39.65  -80.42     5.0  302
"""


@pytest.fixture
def datem_files(tmp_path):
    d = tmp_path / "test"
    d.mkdir()
    (d / "emit-t1.txt").write_text(EMIT)
    (d / "meas-t1.txt").write_text(MEAS)
    return d


def test_read_datem_emit_units_and_times(datem_files):
    e = readers.read_datem_emit(datem_files / "emit-t1.txt")
    assert list(e["release"]) == [1, 2]
    assert e["start"].iloc[0] == pd.Timestamp("1983-09-18 17:00")
    assert e["end"].iloc[1] == pd.Timestamp("1983-09-20 06:00")
    assert e["duration_s"].iloc[1] == 1800
    np.testing.assert_allclose(e["mass"], [208.0, 32.0])  # g -> kg
    assert (e["mass_unit"] == "kg").all()


def test_read_datem_meas_units_and_keys(datem_files):
    m = readers.read_datem_meas(datem_files / "meas-t1.txt")
    assert len(m) == 5
    assert m["conc"].iloc[1] == pytest.approx(10.0e-15)  # pg m-3 -> kg m-3
    assert m["site_num"].tolist() == [302, 306, 302, 306, 302]
    assert m["end"].iloc[3] == pd.Timestamp("1983-09-19 03:00")
    assert m["hhmm"].iloc[2] == 0


def test_acurate_units(tmp_path):
    d = tmp_path / "acurate"
    d.mkdir()
    (d / "emit-t1.txt").write_text(EMIT)
    e = readers.read_datem_emit(d / "emit-t1.txt")
    assert e["mass_unit"].iloc[0] == "Bq"
    assert e["mass"].iloc[0] == pytest.approx(208000 * 3.7e10)
    for k in (0, 1):
        (d / f"meas-t{k}.txt").write_text(MEAS)
    a = readers.read_acurate(d)
    assert set(a["background_file"]) == {0, 1}
    assert a["conc"].iloc[1] == pytest.approx(10.0e-12 * 3.7e10)
    assert a["background"].str.contains("global model").sum() == 5


def test_prediction_writer_round_trip_and_pairing(datem_files, tmp_path):
    m = readers.read_datem_meas(datem_files / "meas-t1.txt")
    pred = np.array([0.0, 12.5, 1.0e-3, 80.0, 0.0])
    out = tmp_path / "ct1_test.txt"
    readers.write_datem_predictions(m, pred, out)
    p = readers.read_datem_meas(out, experiment="test")
    paired = datem_stats.pair_predictions(m, p)
    np.testing.assert_allclose(paired["calc"], pred, rtol=1e-5)
    # an unmatched measurement is missing (-1); the last duplicate wins
    p2 = pd.concat([p.iloc[[0, 1, 1]], p.iloc[[2]]], ignore_index=True)
    p2.loc[2, "conc_raw"] = 99.0
    paired = datem_stats.pair_predictions(m, p2)
    assert paired["calc"].tolist()[1] == pytest.approx(99.0)
    assert paired["calc"].tolist()[3:] == [-1.0, -1.0]


# ---------------------------------------------------------------------------
# DATEM statistics
# ---------------------------------------------------------------------------


def test_statistics_hand_computed():
    meas = np.array([0.0, 0.0, 1.0, 2.0, 4.0, 8.0])
    calc = np.array([0.0, 3.0, 0.0, 2.0, 2.0, 32.0])
    s = datem_stats.datem_statistics(meas, calc)
    assert s.n_pairs == 6
    assert s.mean_measured == pytest.approx(15 / 6)
    assert s.mean_calculated == pytest.approx(39 / 6)
    assert s.bias == pytest.approx(24 / 6)
    assert s.fractional_bias == pytest.approx(2 * 4 / (15 / 6 + 39 / 6))
    # union of non-zero: pairs 2..6 (5); intersection: pairs 4..6 (3)
    assert s.fms == pytest.approx(60.0)
    assert s.correlation == pytest.approx(np.corrcoef(meas, calc)[0, 1], rel=1e-5)
    # FOEX: C > M in pairs 2 and 6 -> 2/6 - 0.5
    assert s.foex == pytest.approx(100 * 2 / 6 - 50, abs=1e-4)
    # factor of 2: pairs 4 (1.0) and 5 (0.5); factor of 5 adds nothing; 6 is 4.0 -> FA5
    assert s.fa2 == pytest.approx(100 * 2 / 6, abs=1e-4)
    assert s.fa5 == pytest.approx(100 * 3 / 6, abs=1e-4)
    assert s.rank == pytest.approx(
        s.correlation**2 + 1 - abs(s.fractional_bias) / 2 + s.fms / 100 + 1 - s.ksp / 100,
        abs=1e-5,
    )


def test_identical_fields_score_four():
    rng = np.random.default_rng(1)
    m = np.where(rng.random(200) < 0.4, 0.0, rng.lognormal(3, 1, 200))
    s = datem_stats.datem_statistics(m, m.copy())
    assert s.ksp == 0.0
    assert s.fractional_bias == 0.0
    assert s.rank == pytest.approx(4.0, abs=1e-5)


def test_zero_pair_options_and_missing():
    meas = np.array([0.0, 0.0, 1.0, 2.0, -1.0])
    calc = np.array([0.0, 1.0, 1.0, 0.0, 5.0])
    assert datem_stats.datem_statistics(meas, calc).n_pairs == 4  # missing dropped
    s = datem_stats.datem_statistics(meas, calc, StatOptions(exclude_zero_pairs=True))
    assert s.n_pairs == 3
    s = datem_stats.datem_statistics(meas, calc, StatOptions(plume_only=True))
    assert s.n_pairs == 0  # one plume pair, fewer than three -> statmain reports 0


def test_ks_quirk_copied_from_statmain():
    # statmain's KSPARM counts a measured percentile only when some modelled
    # percentile reaches it, so far too low a model scores KS = 0 (term 1),
    # while the same error upwards is penalised. Copied on purpose: scores
    # must follow the published protocol (B10b; published-tracer-scores.md).
    rng = np.random.default_rng(0)
    m = rng.lognormal(3.0, 1.5, 400).astype(np.float32)
    m[:100] = 0.0
    low = datem_stats.datem_statistics(m, m * 1e-3)
    high = datem_stats.datem_statistics(m, m * 1e3)
    assert low.ksp == 0.0 and low.rank_terms["ks"] == pytest.approx(1.0)
    assert high.ksp > 50.0


def test_zero_percentile():
    meas = np.arange(0.0, 101.0)
    calc = meas.copy()
    s = datem_stats.datem_statistics(meas, calc, StatOptions(zero_percentile=10))
    assert s.zero_value == 10.0  # sorted[int(10*100/100) - 1] of 1..100


def test_time_averaging(datem_files):
    m = readers.read_datem_meas(datem_files / "meas-t1.txt")
    p = m.copy()
    paired = datem_stats.pair_predictions(m, p)
    a_m, a_c = datem_stats.average_pairs(paired, "time")
    # two sites, padded to the record count with missing values
    np.testing.assert_allclose(a_m[:2], [25 / 3, 25.0], rtol=1e-6)
    assert (a_m[2:] == -1).all()


def test_time_averaging_keeps_measured_and_calculated_apart(datem_files):
    """Per-site means of the measured and calculated values, each from its own
    column (2026-11-25 test review): predictions differ from the measurements, so
    a routine that summed one column into both means would fail.

    Records in file order: site 302 measures 0, 20, 5; site 306 measures 10, 40.
    Predictions 3, 7, 6, 11, 9 in the same order. Sites appear in order of first
    record (302, 306); the arrays are padded with -1 to the five records.
    """

    m = readers.read_datem_meas(datem_files / "meas-t1.txt")
    assert list(m["site_num"]) == [302, 306, 302, 306, 302]
    p = m.copy()
    p["conc_raw"] = [3.0, 7.0, 6.0, 11.0, 9.0]
    paired = datem_stats.pair_predictions(m, p)
    a_m, a_c = datem_stats.average_pairs(paired, "time")
    assert a_m.dtype == np.float32 and a_c.dtype == np.float32
    assert len(a_m) == len(a_c) == 5
    np.testing.assert_allclose(a_m, [25 / 3, 25.0, -1, -1, -1], rtol=1e-6)
    np.testing.assert_allclose(a_c, [6.0, 9.0, -1, -1, -1], rtol=1e-6)

    # A missing prediction drops the whole pair from both means: without the last
    # record, site 302 averages measured (0 + 20)/2 and calculated (3 + 6)/2.
    paired = datem_stats.pair_predictions(m, p.iloc[:4])
    assert paired["calc"].iloc[4] == -1
    a_m, a_c = datem_stats.average_pairs(paired, "time")
    np.testing.assert_allclose(a_m, [10.0, 25.0, -1, -1, -1], rtol=1e-6)
    np.testing.assert_allclose(a_c, [4.5, 9.0, -1, -1, -1], rtol=1e-6)


def test_bootstrap_needs_seed_and_is_reproducible():
    with pytest.raises(ValueError):
        StatOptions(bootstrap=True)
    rng = np.random.default_rng(0)
    m = rng.lognormal(0, 1, 300)
    c = m * rng.lognormal(0, 0.5, 300)
    a = datem_stats.datem_statistics(m, c, StatOptions(bootstrap=True, seed=3)).correlation
    b = datem_stats.datem_statistics(m, c, StatOptions(bootstrap=True, seed=3)).correlation
    full = datem_stats.datem_statistics(m, c).correlation
    assert a == b
    assert a == pytest.approx(full, abs=0.05)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_prediction_writer_refuses_non_finite(datem_files, tmp_path, bad):
    # Written as "nan", pairing would read the record as missing and drop it.
    m = readers.read_datem_meas(datem_files / "meas-t1.txt")
    pred = np.array([0.0, 12.5, bad, 80.0, 0.0])
    out = tmp_path / "ct1_test.txt"
    with pytest.raises(ValueError, match="non-finite"):
        readers.write_datem_predictions(m, pred, out)
    assert not out.exists()


# ---------------------------------------------------------------------------
# experiments split
# ---------------------------------------------------------------------------


def test_assign_releases(datem_files):
    e = readers.read_datem_emit(datem_files / "emit-t1.txt")
    m = readers.read_datem_meas(datem_files / "meas-t1.txt")
    r = experiments.assign_releases(m, e, experiments.AssignmentRule("all", "recent"))
    assert r.tolist() == [1, 1, 1, 1, 2]
    r = experiments.assign_releases(m, e, experiments.AssignmentRule("6h_or_24h", "recent"))
    assert r.tolist() == [1, 1, 1, 0, 2]
    r = experiments.assign_releases(m, e, experiments.AssignmentRule("all", "window_end", 7.0))
    assert r.tolist() == [1, 1, 0, 0, 2]  # 7 h windows: samples ending inside them


def test_assign_releases_needs_time_order(datem_files):
    e = readers.read_datem_emit(datem_files / "emit-t1.txt")
    m = readers.read_datem_meas(datem_files / "meas-t1.txt")
    with pytest.raises(ValueError, match="time order"):
        experiments.assign_releases(m, e.iloc[::-1], experiments.AssignmentRule("all", "recent"))


def test_duration_rule_keeps_6h_and_24h_samples():
    m = pd.DataFrame(
        {
            "start": pd.to_datetime(["1983-09-18 18:00"] * 3),
            "end": pd.to_datetime(["1983-09-18 21:00", "1983-09-19 00:00", "1983-09-19 18:00"]),
            "duration_s": [3 * 3600.0, 6 * 3600.0, 24 * 3600.0],
        }
    )
    e = pd.DataFrame({"release": [1], "start": pd.to_datetime(["1983-09-18 17:00"])})
    r = experiments.assign_releases(m, e, experiments.AssignmentRule("6h_or_24h", "recent"))
    assert r.tolist() == [0, 1, 1]


def test_anatex_wrf_grid():
    g = experiments.ANATEX_WRF_D02
    x, y = experiments.wrf_grid_xy([g["sync_lat"]], [g["sync_lon"]])
    assert (x[0], y[0]) == pytest.approx((g["sync_x"], g["sync_y"]))
    # Along the reference longitude, x does not change; 1 degree of latitude
    # at a true latitude is 111.2 km on the 6371.2 km sphere, 11.12 cells.
    x1, y1 = experiments.wrf_grid_xy([33.0, 34.0], [-100.0, -100.0])
    assert x1[0] == pytest.approx(x1[1])
    assert y1[1] - y1[0] == pytest.approx(6371.2 * np.pi / 180 / 10.0, rel=2e-3)
    inside = experiments.inside_anatex_wrf_domain(
        [41.2, 25.90, 53.30, 41.67], [-98.02, -97.43, -60.37, -69.97]
    )
    assert inside.tolist() == [True, False, False, False]


# ---------------------------------------------------------------------------
# JRC ETEX and ATMES-II
# ---------------------------------------------------------------------------


def _etex_line(num, code, values, fmt):
    return f" {num:3d}  {code:4s}" + "".join(fmt(v) for v in values)


@pytest.fixture
def etex_files(tmp_path):
    d = tmp_path / "ETEX_release1"
    d.mkdir()
    head2 = "         " + "".join(f"{i:6d}" for i in range(1, 31))
    v1 = [-0.99] + [0.0] * 4 + [0.5, 1.2, -0.88, 0.3] + [0.0] * 20 + [-0.99]
    v2 = [0.0] * 30
    f1 = [0] + [1] * 4 + [11, 11, 41, 11] + [1] * 20 + [0]
    num = lambda v: f"{v:6.2f}"  # noqa: E731
    dat = ["header", head2, _etex_line(1, "A01", v1, num), _etex_line(2, "F10", v2, num)]
    cod = ["header", head2]
    cod += [_etex_line(1, "A01", f1, lambda v: f"{v:6d}")]
    cod += [_etex_line(2, "F10", [1] * 30, lambda v: f"{v:6d}")]
    (d / "pmch.dat").write_text("\n".join(dat) + "\n")
    (d / "pmch.cod").write_text("\n".join(cod) + "\n")
    (d / "stationlist.950130").write_text(
        "ETEX list\nLast update\nLatitude, Longitude given in degree.minutes!\n\n"
        "cc, Station name, Lat, Long, Alt, WMOCode, Remarks\n"
        "A01, Bregenz, 47.30, 09.44, 424, 11101, \n"
        "F10, Cherbourg , 49.31, -01.30, 61, 07039, \n"
    )
    return d


def test_etex_reader(etex_files):
    j = readers.read_etex_samples(1, etex_files)
    assert len(j) == 60
    a = j[j["site"] == "A01"].set_index("interval")
    assert a.loc[1, "start"] == pd.Timestamp("1994-10-23 15:00")
    assert a.loc[6, "conc"] == pytest.approx(0.5e-12)
    assert np.isnan(a.loc[8, "conc"]) and a.loc[8, "conc_raw"] == -0.88
    assert a.loc[8, "quality_flag"] == 41
    assert a.loc[1, "lat"] == pytest.approx(47.5)  # 47°30'
    f = j[j["site"] == "F10"].iloc[0]
    assert f["lon"] == pytest.approx(-1.5)  # -01.30 is 1°30' W


def test_etex_release_source():
    r = readers.read_etex_release(1).iloc[0]
    # Monterfil, 2 00'30" W, 48 03'30" N, 90 m asl, release at 8 m agl (JRC)
    assert r["site"] == "Monterfil"
    assert r["lat"] == pytest.approx(48.058333, abs=1e-6)
    assert r["lon"] == pytest.approx(-2.008333, abs=1e-6)
    assert r["site_alt_m"] == 90.0 and r["height_m"] == 8.0
    assert r["mass"] == 340.0 and r["start"] == pd.Timestamp("1994-10-23 16:00")
    # ETEX-2: the same site (JRC ETEX overview page)
    r2 = readers.read_etex_release(2).iloc[0]
    assert r2["site"] == "Monterfil"
    assert (r2["lat"], r2["lon"], r2["site_alt_m"]) == (r["lat"], r["lon"], r["site_alt_m"])
    assert r2["mass"] == 490.0 and r2["start"] == pd.Timestamp("1994-11-14 15:00")


def test_atmes2_results_state_unverified_definitions():
    m = np.array([0.0, 0.05, 0.2, 1.0])
    s = atmes2.atmes2_statistics(m, m)
    assert s["definitions"] == atmes2.DEFINITIONS_NOTE
    assert "EUR 17756 EN" in s["definitions"] and "not checked" in s["definitions"]


def test_zero_filter(etex_files):
    j = readers.read_etex_samples(1, etex_files)
    k = atmes2.zero_filter(j)
    a = j[(j["site"] == "A01") & k]
    # positives at 6, 7, 9; zeros kept from 4 to 11; -.88 at 8 lost
    assert a["interval"].tolist() == [4, 5, 6, 7, 9, 10, 11]
    assert not k[j["site"] == "F10"].any()  # no tracer: no zeros kept


def test_atmes2_perfect_and_biased():
    m = np.array([0.0, 0.05, 0.2, 1.0, 3.0, 0.0])
    s = atmes2.atmes2_statistics(m, m, fa_zeros="floor")
    assert s["mg"] == pytest.approx(1.0)
    assert s["vg"] == pytest.approx(1.0)
    assert s["nmse"] == pytest.approx(0.0)
    assert s["fa2"] == 100.0 and s["ks_d"] == 0.0 and s["foex"] == -50.0
    assert atmes2.atmes2_statistics(m, m)["fa2"] == pytest.approx(100 * 4 / 6)
    s = atmes2.atmes2_statistics(m, 2 * m)
    assert s["mg"] > 1.0 and s["bias"] > 0
    assert s["ks_count"] == pytest.approx(s["ks_d"] * len(m))


def test_fms_space():
    rng = np.random.default_rng(0)
    lat = rng.uniform(43, 58, 80)
    lon = rng.uniform(-5, 20, 80)
    c = np.exp(-((lon - 5) ** 2 + (lat - 50) ** 2) / 20.0)
    assert atmes2.fms_space(lat, lon, c, c, cell_km=25) == pytest.approx(100.0)
    shifted = np.exp(-((lon - 12) ** 2 + (lat - 50) ** 2) / 20.0)
    assert atmes2.fms_space(lat, lon, c, shifted, cell_km=25) < 60.0


def test_datem_to_jrc_code():
    pre = sorted(
        [
            "A",
            "B",
            "BG",
            "CH",
            "CR",
            "D",
            "DK",
            "F",
            "H",
            "N",
            "NL",
            "PL",
            "R",
            "S",
            "SF",
            "SR",
            "UK",
        ],
    )
    assert etex.datem_to_jrc_code(1001, pre) == "A01"
    assert etex.datem_to_jrc_code(6013, pre) == "D13"
    assert etex.datem_to_jrc_code(9003, pre) == "H03"
    assert etex.datem_to_jrc_code(9203, pre) == "NL03"
    assert etex.datem_to_jrc_code(9807, pre) == "UK07"


# ---------------------------------------------------------------------------
# against the real archives (skipped when absent)
# ---------------------------------------------------------------------------


@needs_datem
def test_reproduces_posted_datem_statistics():
    from tracers.__main__ import verify_datem

    results = verify_datem(_datem_dir())
    assert len(results) == 12
    for exp, tr, kind, _rank, diffs in results:
        assert diffs == [], f"{exp} t{tr} stat{kind}: {diffs}"


#: Range of the rank from 20 runs of DATEM's statmain -b (gfortran 7.5,
#: printed to 5 decimals), 2026-11-23.
STATMAIN_B_RANGE = {
    ("captex", 1): (2.24240, 2.24759),
    ("anatex", 1): (2.07872, 2.08328),
    ("anatex", 2): (1.73225, 1.73235),
    ("etex", 1): (2.33404, 2.33928),
    ("acurate", 1): (1.96272, 1.96443),
}


@needs_datem
@pytest.mark.parametrize("case", sorted(STATMAIN_B_RANGE))
def test_bootstrap_mean_within_statmain_range(case):
    root = _datem_dir()
    exp, tr = case
    m = readers.read_datem_meas(root / "exp_data" / exp / f"meas-t{tr}.txt")
    p = readers.read_datem_meas(root / "mdl_data" / exp / f"ct{tr}_000.txt", experiment=exp)
    ranks = [datem_stats.score(m, p, StatOptions(bootstrap=True, seed=s)).rank for s in range(20)]
    lo, hi = STATMAIN_B_RANGE[case]
    assert lo <= np.mean(ranks) <= hi


@needs_datem
def test_adopted_split_counts():
    s = experiments.split_samples(root=_datem_dir())
    counts = s.groupby("experiment").size().to_dict()
    assert counts == {
        "CAPTEX-1": 347,
        "CAPTEX-2": 352,
        "CAPTEX-3": 356,
        "CAPTEX-4": 281,
        "CAPTEX-5": 318,
        "CAPTEX-7": 177,
        "ANATEX-GGW": 577,
        "ANATEX-STC": 570,
    }


@needs_datem
def test_anatex_wrf_domain_cut_counts():
    # B10b: the cut drops three sites (2601, 3007, 3052) and 26 pairs per
    # tracer, more than the surplus over Hegarty et al. (15 and 14), so it is
    # recorded but not adopted (data/tracer-experiments.md section 5).
    rule = experiments.AssignmentRule(
        "6h_or_24h", "recent", anatex_last_start="1987-01-13", anatex_wrf_domain=True
    )
    s = experiments.split_samples(rule, root=_datem_dir())
    counts = s.groupby("experiment").size()
    assert counts["ANATEX-GGW"] == 551 and counts["ANATEX-STC"] == 544
    adopted = experiments.split_samples(root=_datem_dir())
    a = adopted[adopted["experiment"].str.startswith("ANATEX")]
    dropped = set(a["site"]) - set(s.loc[s["experiment"].str.startswith("ANATEX"), "site"])
    assert dropped == {"2601", "3007", "3052"}


@needs_etex
def test_atmes2_zero_filter_count():
    j = readers.read_etex_samples(1, _etex_dir() / "ETEX_release1")
    assert int(atmes2.zero_filter(j).sum()) == 1415


@needs_etex
@needs_datem
def test_jrc_datem_match():
    m = etex.match_jrc_datem(_etex_dir() / "ETEX_release1", _datem_dir() / "exp_data" / "etex")
    n = m.groupby("category").size().to_dict()
    assert n["identical"] == 3103
    assert n["value differs"] == 1  # D28 interval 3
    assert n["JRC code, value in DATEM"] == 1  # D10 interval 2, changed in v1.1
    assert "DATEM only" not in n
