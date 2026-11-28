"""Task CONV: one CPU footprint run with convection overrides.

    PYTHONPATH=<worktree>/src .venv/bin/python scripts/validation/conv/run_footprint.py \
        configs/smoke_mhd_single_release.yaml outputs/agents/CONV/smoke/buoy_exp \
        --set convection.emanuel.closure=buoyancy convection.emanuel.application=exponential

Writes the effective config next to the output, then runs ``lpdm.main`` (on the
CPU unless ``--device``) in this process (the code on PYTHONPATH; the run refuses to start if lpdm is
imported from anywhere else). ``--set`` takes dotted keys; values are parsed as
YAML scalars.
"""

from __future__ import annotations

import argparse
import os
import runpy
import sys
from pathlib import Path

import yaml


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("out")
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    for item in args.set:
        key, value = item.split("=", 1)
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(value)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg_path = out.with_suffix(".yaml")
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))

    import lpdm

    src = os.environ.get("PYTHONPATH", "").split(os.pathsep)[0]
    assert src and lpdm.__file__.startswith(src), (lpdm.__file__, src)
    print(f"lpdm from {lpdm.__file__}; config {cfg_path}", flush=True)
    sys.argv = [
        "lpdm.main",
        "--config",
        str(cfg_path),
        "--device",
        args.device,
        "--output-uri",
        str(out),
    ]
    runpy.run_module("lpdm.main", run_name="__main__")


if __name__ == "__main__":
    main()
