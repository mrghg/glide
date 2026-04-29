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