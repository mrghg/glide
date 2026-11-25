"""Tests for the validation figure library (scripts/validation/figures, task B7).

The library's statistics (log ratio, particle-noise standard error), the
provenance stamp, the gallery page and the static footprint explorer, on small
synthetic inputs. The Tier 0 figures themselves are not rendered here (minutes
of CPU); ``make_all.py`` does that.
"""

from __future__ import annotations

import base64
import json
import re
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "validation"))

from figures import core, explorer, gallery, style, tier0  # noqa: E402


def test_log_ratio_is_log10_and_blank_without_information() -> None:
    model = np.array([10.0, 1.0, 0.0, 5.0])
    ref = np.array([1.0, 10.0, 3.0, 0.0])
    out = core.log_ratio(model, ref)
    assert out[0] == pytest.approx(1.0)
    assert out[1] == pytest.approx(-1.0)
    assert np.isnan(out[2]) and np.isnan(out[3])


def test_subset_spread_estimates_the_standard_error_of_the_mean() -> None:
    """For the mean of n unit-variance draws the standard error is 1/sqrt(n);
    the subset estimate must recover it (averaged over seeds, within 10%)."""
    rng = np.random.default_rng(1)
    n = 20_000
    est = []
    for seed in range(40):
        x = rng.standard_normal(n)
        est.append(core.subset_spread(lambda idx, x=x: x[idx].mean(), n, k=10, seed=seed))
    assert np.mean(est) == pytest.approx(1 / np.sqrt(n), rel=0.1)


def test_subset_indices_partition_the_particles() -> None:
    parts = core.subset_indices(1003, k=10, seed=3)
    allidx = np.sort(np.concatenate(parts))
    assert np.array_equal(allidx, np.arange(1003))
    assert {p.size for p in parts} <= {100, 101}


def test_config_hash_is_order_independent_and_sensitive() -> None:
    a = core.config_hash({"dt": 60, "n": 1000, "nested": {"x": 1, "y": 2}})
    b = core.config_hash({"nested": {"y": 2, "x": 1}, "n": 1000, "dt": 60})
    c = core.config_hash({"dt": 30, "n": 1000, "nested": {"x": 1, "y": 2}})
    assert a == b != c
    assert re.fullmatch(r"[0-9a-f]{10}", a)


def test_physics_tag_names_a_version() -> None:
    tag = core.physics_tag()
    assert tag == "unknown" or "physics" in tag or tag.startswith("pre-freeze-")


def _fake_figure(root: Path, tag: str, fid: str, created: str) -> None:
    fig, axes = core.new_figure(1)
    axes[0, 0].plot([0, 1], [0, 1], color=style.GLIDE)
    prov = {"physics_tag": tag, "config_hash": "abcdef0123", "git_rev": "0" * 40,
            "created": created}  # fmt: skip
    core.stamp(fig, "A figure", prov, f"tier0/{fid}")
    core.save(fig, root / "figures" / tag / "tier0", figure_id=fid, tier=0, title="A figure",
              caption="Caption text.", config={"n": 1}, prov=prov,
              metrics={"err": np.float64(0.0123456789)}, source="tests/x.py")  # fmt: skip


def test_save_writes_png_and_sidecar(tmp_path: Path) -> None:
    _fake_figure(tmp_path, "v0-physics", "taylor_dispersion", "2026-01-01T00:00:00Z")
    out = tmp_path / "figures" / "v0-physics" / "tier0"
    assert (out / "taylor_dispersion.png").stat().st_size > 1000
    side = json.loads((out / "taylor_dispersion.json").read_text())
    assert side["physics_tag"] == "v0-physics" and side["config_hash"] == "abcdef0123"
    assert side["config"] == {"n": 1}
    assert side["metrics"]["err"] == pytest.approx(0.0123457)


def test_gallery_is_keyed_by_physics_tag_and_lists_pending(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Every Tier 0 figure is ready since task A6 drew reciprocity; register a
    # pending one so the page's pending list is still exercised.
    pending = tier0.FigureSpec("pending_probe", "Pending probe", "none", status="pending: a test")
    monkeypatch.setattr(tier0, "FIGURES", [*tier0.FIGURES, pending])
    _fake_figure(tmp_path, "v0-physics", "taylor_dispersion", "2026-01-01T00:00:00Z")
    _fake_figure(tmp_path, "v1-physics", "taylor_dispersion", "2026-06-01T00:00:00Z")
    _fake_figure(tmp_path, "v1-physics", "terrain", "2026-06-01T00:00:00Z")
    page = gallery.write(tmp_path).read_text()
    # Newest tag first, both as columns of the index and as sections.
    assert page.index("## Physics `v1-physics`") < page.index("## Physics `v0-physics`")
    assert "| tier | figure | `v1-physics` | `v0-physics` |" in page
    # Every image link resolves relative to the page.
    for link in re.findall(r"!\[[^\]]*\]\(([^)]+)\)", page):
        assert (tmp_path / link).exists(), link
    # Planned but missing figures are visible.
    assert "Forward–backward reciprocity" in page
    assert "- Tier 0, Pending probe: a test" in page
    planned = {s.figure_id for s in tier0.FIGURES}
    assert {"taylor_dispersion", "well_mixed", "analytic_plume", "diffusion_limit",
            "reciprocity", "convergence", "terrain"} <= planned  # fmt: skip
    ready = {s.figure_id for s in tier0.FIGURES if s.status == "ready"}
    assert set(tier0.BUILDERS) == ready and "reciprocity" in ready


def _store(path: Path, field: np.ndarray, weighting: str | None) -> None:
    n_rel, ny, nx = field.shape
    da = xr.DataArray(
        field[:, None, None].repeat(2, axis=2).astype("float32"),  # two z bins
        dims=("release", "time_ago", "z_bin", "latitude", "longitude"),
        coords=dict(
            latitude=np.linspace(50.0, 52.0, ny),
            longitude=np.linspace(-3.0, 0.0, nx),
            z_bottom_m=("z_bin", [0.0, 40.0]),
            z_top_m=("z_bin", [40.0, 1000.0]),
            release_lon=("release", np.full(n_rel, -1.5)),
            release_lat=("release", np.full(n_rel, 51.0)),
            release_time=("release", pd.date_range("2024-01-01", periods=n_rel, freq="h")),
            site=("release", ["XYZ"] * n_rel),
        ),
        attrs={} if weighting is None else {"density_weighting": weighting},
    )
    path.mkdir(parents=True)
    da.to_dataset(name="footprint").to_zarr(path / "footprints.zarr")


def _decode(b64: str, vmax: float) -> np.ndarray:
    q = np.frombuffer(zlib.decompress(base64.b64decode(b64)), dtype=np.uint8).astype(float)
    lo = np.log10(vmax) - explorer.DECADES
    return np.where(q > 0, 10 ** (lo + (q - 1) / (explorer.LEVELS - 1) * explorer.DECADES), 0.0)


def test_explorer_embeds_stilt_footprints_that_decode_to_within_quantisation(
    tmp_path: Path,
) -> None:
    rng = np.random.default_rng(0)
    g = rng.lognormal(-8, 1.5, size=(2, 6, 8))
    g[:, 0, 0] = 0.0
    _store(tmp_path / "glide", g, "dry_air")
    _store(tmp_path / "ref", g * 2.0, "dry_air")
    page = explorer.build(tmp_path / "glide", tmp_path / "ref", label_reference="REF")
    data = json.loads(re.search(r"const D = (\{.*?\});\n", page).group(1))
    assert data["nx"] == 8 and data["ny"] == 6 and len(data["releases"]) == 2
    r0 = data["releases"][0]
    assert r0["label"].startswith("XYZ, 2024-01-01 00:00")
    # STILT units for a dry_air store: m_air / h, summed over time, lowest 40 m only.
    want = g[0] * 0.02897 / 40.0
    got = _decode(r0["glide"], r0["vmax"]).reshape(6, 8)
    shown = want >= r0["vmax"] * 10**-explorer.DECADES
    assert got[0, 0] == 0.0
    np.testing.assert_allclose(got[shown], want[shown], rtol=0.025)  # 5 decades / 254 levels
    assert r0["total_ref"] / r0["total_glide"] == pytest.approx(2.0)
    assert "<script>" in page and "http" not in page.split("<script>")[1]  # no network


def test_explorer_needs_a_density_for_unweighted_stores(tmp_path: Path) -> None:
    g = np.ones((1, 3, 4))
    _store(tmp_path / "old", g, None)  # a store from before decision 0022
    fp = explorer.surface_footprint(tmp_path / "old", air_density=1.2)
    np.testing.assert_allclose(fp.values[0], 0.02897 / (40.0 * 1.2), rtol=1e-6)
