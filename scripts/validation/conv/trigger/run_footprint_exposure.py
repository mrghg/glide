"""Decision 6 study: one footprint run, logging particle exposure to both triggers.

    PYTHONPATH=<worktree>/src .venv/bin/python scripts/validation/conv/trigger/run_footprint_exposure.py \
        <config.yaml> <out dir> [--device cuda] --set convection.emanuel.trigger=flexpart_cloud_base ...

Like ``scripts/validation/conv/run_footprint.py`` (effective config next to the
output, ``lpdm.main`` in this process, refuses an lpdm imported from outside
PYTHONPATH), plus an exposure log. Before each convection event it diagnoses the
window's columns with BOTH triggers (`glide` and `flexpart_cloud_base`, all other
options as the run's scheme) and counts the active particles (particle-events)
standing in columns that fire under each, under FLEXPART's but not GLIDE's
("new"), and of those the ones below the cloud top (the layers the matrix acts
on). This only reads the state; the run itself is unchanged (no random numbers,
no particle edits). Written to ``<out dir>.exposure.json``.
"""

from __future__ import annotations

import argparse
import json
import os
import runpy
import sys
from pathlib import Path

import numpy as np
import torch
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
    from lpdm.convection import emanuel as em

    src = os.environ.get("PYTHONPATH", "").split(os.pathsep)[0]
    assert src and lpdm.__file__.startswith(src), (lpdm.__file__, src)
    print(f"lpdm from {lpdm.__file__}; config {cfg_path}", flush=True)

    events: list[dict] = []
    shadows: dict[int, dict] = {}
    orig = em.EmanuelReducedConvection.maybe_convect

    def wrapped(
        self,
        particles,
        state,
        met_window,
        *,
        t_alpha,
        dt_seconds,
        active_mask,
        engine,
        generator=None,
    ):
        if id(self) not in shadows:
            common = dict(
                closure=self.closure,
                application=self.application,
                min_cloud_depth_m=self.min_cloud_depth_m,
                buoyancy_mass_flux_per_k=self.buoyancy_mass_flux_per_k,
            )
            shadows[id(self)] = {
                "glide": em.EmanuelReducedConvection(**common, trigger="glide"),
                "fp": em.EmanuelReducedConvection(**common, trigger="flexpart_cloud_base"),
            }
        sh = shadows[id(self)]
        dev, dt = engine.device, engine.dtype
        if bool(torch.any(active_mask)):
            cg = sh["glide"].column_state(met_window, t_alpha=t_alpha, device=dev, dtype=dt)
            cf = sh["fp"].column_state(met_window, t_alpha=t_alpha, device=dev, dtype=dt)
            frame = engine.frame
            pc = em.nearest_column_index(
                particles[:, 0],
                particles[:, 1],
                frame.x_from_lon(np.asarray(met_window.metadata.lon, dtype=np.float64)),
                frame.y_from_lat(np.asarray(met_window.metadata.lat, dtype=np.float64)),
            )
            act = active_mask
            fg, ff = cg.fires[pc] & act, cf.fires[pc] & act
            new = ff & ~fg
            top = cf.height_m[pc].gather(1, cf.i_lnb.clamp(min=0)[pc].unsqueeze(1)).squeeze(1)
            below_top = particles[:, 2] <= top
            occ_new = torch.zeros_like(cf.fires)
            occ_new[pc[new]] = True
            events.append(
                {
                    "time": str(met_window.metadata.time_start),
                    "t_alpha": float(t_alpha),
                    "active": int(act.sum()),
                    "in_glide": int(fg.sum()),
                    "in_fp": int(ff.sum()),
                    "in_new": int(new.sum()),
                    "in_new_below_top": int((new & below_top).sum()),
                    "in_glide_not_fp": int((fg & ~ff).sum()),
                    "cols_glide": int(cg.fires.sum()),
                    "cols_fp": int(cf.fires.sum()),
                    "occupied_new_cols": int(occ_new.sum()),
                    "n_col": int(cf.fires.numel()),
                }
            )
        return orig(
            self,
            particles,
            state,
            met_window,
            t_alpha=t_alpha,
            dt_seconds=dt_seconds,
            active_mask=active_mask,
            engine=engine,
            generator=generator,
        )

    em.EmanuelReducedConvection.maybe_convect = wrapped
    sys.argv = [
        "lpdm.main",
        "--config",
        str(cfg_path),
        "--device",
        args.device,
        "--output-uri",
        str(out),
    ]
    try:
        runpy.run_module("lpdm.main", run_name="__main__")
    finally:
        tot = {
            k: sum(e[k] for e in events)
            for k in (
                "active",
                "in_glide",
                "in_fp",
                "in_new",
                "in_new_below_top",
                "in_glide_not_fp",
            )
        }
        Path(str(out) + ".exposure.json").write_text(
            json.dumps({"totals": tot, "events": events}, indent=1)
        )
        print(f"exposure totals: {tot}", flush=True)


if __name__ == "__main__":
    main()
