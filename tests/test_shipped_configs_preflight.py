"""Every shipped and validation config passes the spatial preflight against the
store it reads, when that store is present (skipped otherwise).

The preflight (`main._validate_meteorology_space_coverage`, PR #56) refuses a
met_domain outside the meteorology grid. The shipped EUROPE configs declared
100 W-40 E, 10-80 N over stores covering 98 W-39.5 E, 10.75-79 N, so particles
in the overhang had been moved by the grid edge's winds; the unit tests, on
synthetic readers, could not see it.
"""

import glob
from pathlib import Path

import pytest

from lpdm.config import RunConfig
from lpdm.main import _validate_meteorology_space_coverage
from lpdm.met_reader import ArcoEra5ZarrReader

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = sorted(
    p
    for pattern in ("configs/*.yaml", "scripts/validation/*/*.yaml")
    for p in glob.glob(str(ROOT / pattern))
)


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: str(Path(p).relative_to(ROOT)))
def test_config_met_domain_inside_its_store(path: str, monkeypatch) -> None:
    monkeypatch.chdir(ROOT)  # configs give store paths relative to the repository
    try:
        cfg = RunConfig.from_yaml(path)
    except Exception as exc:  # not a run config (e.g. a met-bazarr recipe)
        pytest.skip(f"not a GLIDE run config: {type(exc).__name__}")
    stores = (
        sorted(glob.glob(cfg.io.zarr_store)) if "*" in cfg.io.zarr_store else [cfg.io.zarr_store]
    )
    stores = [s for s in stores if Path(s).exists() and any(Path(s).iterdir())]
    if not stores:
        pytest.skip(f"store not present: {cfg.io.zarr_store}")
    reader = ArcoEra5ZarrReader(zarr_store=cfg.io.zarr_store, channel_names=("u", "v", "w"))
    _validate_meteorology_space_coverage(reader, cfg)
