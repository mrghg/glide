from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest


def _load_download_sample_cube_module():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "download_sample_cube.py"
    spec = importlib.util.spec_from_file_location("download_sample_cube", module_path)
    if spec is None or spec.loader is None:
        raise AssertionError("Failed to load download_sample_cube module")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_resolve_year_month_window_covers_full_month_inclusive() -> None:
    module = _load_download_sample_cube_module()

    t_start, t_end = module._resolve_year_month_window("202401")
    assert t_start == "2024-01-01T00:00:00"
    assert t_end == "2024-01-31T23:00:00"

    # February 2024 is a leap year, 29 days.
    t_start, t_end = module._resolve_year_month_window("202402")
    assert t_end == "2024-02-29T23:00:00"


@pytest.mark.parametrize("bad", ["20240", "abcdef", "202413", "202400", ""])
def test_resolve_year_month_window_rejects_bad_input(bad: str) -> None:
    module = _load_download_sample_cube_module()
    with pytest.raises(ValueError):
        module._resolve_year_month_window(bad)


def test_resolve_domain_bbox_returns_registered_extents() -> None:
    module = _load_download_sample_cube_module()
    bbox = module._resolve_domain_bbox("EUROPE")
    assert set(bbox) == {"lon_min", "lon_max", "lat_min", "lat_max"}
    assert bbox["lon_min"] < bbox["lon_max"]
    assert bbox["lat_min"] < bbox["lat_max"]


def test_resolve_domain_bbox_rejects_unknown_domain() -> None:
    module = _load_download_sample_cube_module()
    with pytest.raises(ValueError, match="Unknown domain"):
        module._resolve_domain_bbox("ATLANTIS")


def test_dispatch_rejects_mixing_named_and_adhoc_modes() -> None:
    from argparse import Namespace

    module = _load_download_sample_cube_module()
    args = Namespace(
        store_uri="gs://x",
        zarr_version=2,
        domain="EUROPE",
        year_month="202401",
        out_dir="data/era5",
        out_path="data/sample.zarr",   # also ad-hoc → conflict
        time_start=None,
        time_end=None,
        lon_min=None,
        lon_max=None,
        lat_min=None,
        lat_max=None,
    )
    with pytest.raises(SystemExit, match="Cannot mix"):
        module._dispatch(args)


def test_dispatch_requires_both_domain_and_year_month_together() -> None:
    from argparse import Namespace

    module = _load_download_sample_cube_module()
    args = Namespace(
        store_uri="gs://x",
        zarr_version=2,
        domain="EUROPE",
        year_month=None,
        out_dir="data/era5",
        out_path=None,
        time_start=None,
        time_end=None,
        lon_min=None,
        lon_max=None,
        lat_min=None,
        lat_max=None,
    )
    with pytest.raises(SystemExit, match="must be given together"):
        module._dispatch(args)


def test_replace_store_atomically_restores_existing_store_on_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_download_sample_cube_module()

    out_store = tmp_path / "sample_met.zarr"
    tmp_store = tmp_path / "sample_met.zarr.tmp-download"
    out_store.mkdir()
    tmp_store.mkdir()
    (out_store / "marker.txt").write_text("old", encoding="utf-8")
    (tmp_store / "marker.txt").write_text("new", encoding="utf-8")

    real_replace = os.replace
    replace_calls = {"count": 0}

    def flaky_replace(src: str | os.PathLike[str], dst: str | os.PathLike[str]) -> None:
        replace_calls["count"] += 1
        if replace_calls["count"] == 2:
            raise OSError("simulated replace failure")
        real_replace(src, dst)

    monkeypatch.setattr(module.os, "replace", flaky_replace)

    with pytest.raises(OSError, match="simulated replace failure"):
        module._replace_store_atomically(str(tmp_store), str(out_store))

    assert out_store.exists()
    assert (out_store / "marker.txt").read_text(encoding="utf-8") == "old"
    assert not (tmp_path / "sample_met.zarr.bak-replace").exists()