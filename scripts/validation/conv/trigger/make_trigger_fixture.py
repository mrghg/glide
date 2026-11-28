"""Decision 0034: a fixture of real columns with the Decision 6 study's firing masks.

    cd <main checkout>   # so data/ resolves
    PYTHONPATH=<worktree>/src .venv/bin/python \
        <worktree>/scripts/validation/conv/trigger/make_trigger_fixture.py \
        outputs/agents/CONV/trigger/columns/jul.npz <worktree>/tests/data/trigger_columns_l40.npz

Reads one hour of the study's July sample (8 Jul 2019 12 UTC, EUR store, 40
levels, the study's met domain, as `run_columns.sh`), diagnoses every column
with the code on PYTHONPATH under both triggers, and checks the two masks
against the study's (`fires_fpcb0`, `fires_glide` in the study's npz, written by
`trigger_columns.py` at d9f09f0). It then keeps a stratified subset of columns:
those where the FLEXPART trigger is close to its threshold (ΔT_v at ICB within
0.3 K of −0.9 K, or DTMIN within 0.3 K of 0), those where the glide trigger
fires, and random firing and non-firing columns. It writes their profiles
(the window's hour-start T, q, pressure and height, [Z, C], level order as
read) and the study's masks. The profiles are modified Copernicus Climate
Change Service information (ERA5, 2019).
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from datetime import UTC, datetime

import numpy as np
import torch

import lpdm
from lpdm.config import RunConfig
from lpdm.convection import emanuel as em
from lpdm.main import _met_domain_bounds, _met_interval
from lpdm.met_reader import ArcoEra5ZarrReader, BoundingBoxRequest, TimeBounds
from lpdm.turbulence import get_scheme
from lpdm.vertical_grid import stretched_agl_levels

STAMP = "2019-07-08T12"
STORE = "data/era5-arco/EUR_20190621-20190715_ERA5-ARCO.zarr"
DOMAIN = (-30.0, 40.0, 32.0, 72.0)
CONFIG = "configs/ab_multisite_perf.yaml"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("study_npz")
    ap.add_argument("out")
    ap.add_argument("--config", default=CONFIG)
    ap.add_argument("--n-random", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    print(f"lpdm: {lpdm.__file__}", flush=True)
    torch.set_num_threads(4)

    cfg = RunConfig.from_yaml(args.config)
    turb = get_scheme(cfg.turbulence.scheme)
    channels = tuple(
        dict.fromkeys(("u", "v", "w", "blh", "sp", *turb.required_met_keys(), "t", "q"))
    )
    levels = stretched_agl_levels(
        40, cfg.met_domain.alt_max_m, first_layer_m=cfg.met_domain.first_layer_m
    )
    reader = ArcoEra5ZarrReader(
        zarr_store=STORE, channel_names=channels, device="cpu", agl_levels_m=levels
    )
    bounds = _met_domain_bounds(cfg)
    lon0, lon1, lat0, lat1 = DOMAIN
    bounds = replace(bounds, lon_min=lon0, lon_max=lon1, lat_min=lat0, lat_max=lat1)
    t0 = datetime.fromisoformat(STAMP).replace(tzinfo=UTC)
    met = reader.fetch_hourly_window(
        BoundingBoxRequest(
            spatial=bounds, time=TimeBounds(start=t0, end=t0 + _met_interval(reader))
        )
    )
    cpu = torch.device("cpu")
    cols = {
        trig: em.EmanuelReducedConvection(trigger=trig).column_state(
            met, t_alpha=0.0, device=cpu, dtype=torch.float32
        )
        for trig in ("flexpart_cloud_base", "glide")
    }
    study = np.load(args.study_npz)
    key = f"L40/{t0.strftime('%Y%m%dT%H')}"
    fires_fp = cols["flexpart_cloud_base"].fires.numpy()
    fires_gl = cols["glide"].fires.numpy()
    same_fp = np.array_equal(study[f"{key}/fires_fpcb0"], fires_fp)
    same_gl = np.array_equal(study[f"{key}/fires_glide"], fires_gl)
    print(
        f"{key}: {fires_fp.size} columns; flexpart_cloud_base fires {int(fires_fp.sum())}, "
        f"identical to the study: {same_fp}; glide fires {int(fires_gl.sum())}, "
        f"identical to the study: {same_gl}",
        flush=True,
    )
    if not (same_fp and same_gl):
        sys.exit(1)

    c = cols["flexpart_cloud_base"]
    dtv_icb = c.dtv_lcl_plus_1.double().numpy()
    dtmin = c.dtmin_k.double().numpy()
    rng = np.random.default_rng(args.seed)
    near_dtv = np.flatnonzero(np.abs(dtv_icb + 0.9) < 0.3)
    near_dtmin = np.flatnonzero(np.abs(dtmin) < 0.3)
    pick = [
        rng.choice(near_dtv, min(40, near_dtv.size), replace=False),
        rng.choice(near_dtmin, min(40, near_dtmin.size), replace=False),
        np.flatnonzero(fires_gl)[:: max(1, int(fires_gl.sum()) // 30)][:30],
        rng.choice(np.flatnonzero(fires_fp), args.n_random, replace=False),
        rng.choice(np.flatnonzero(~fires_fp), args.n_random, replace=False),
    ]
    idx = np.unique(np.concatenate(pick))

    def flat(field: torch.Tensor) -> np.ndarray:  # [Z, Y, X] -> [Z, C]
        return field.reshape(field.shape[0], -1)[:, idx].numpy().astype(np.float32)

    p_start, _ = met.column_pressure_hpa()
    out = {
        "t": flat(met.channel("t")[0]),
        "q": flat(met.channel("q")[0]),
        "pressure_hpa": flat(p_start),
        "height_agl_m": flat(met.height_agl_m),
        "fires_flexpart_cloud_base": study[f"{key}/fires_fpcb0"][idx],
        "fires_glide": study[f"{key}/fires_glide"][idx],
        "column_index": idx,
        "interval_s": np.asarray(_met_interval(reader).total_seconds()),
    }
    np.savez_compressed(args.out, **out)
    print(
        f"wrote {args.out}: {idx.size} columns ({near_dtv.size} near ΔT_v(ICB) = -0.9 K and "
        f"{near_dtmin.size} near DTMIN = 0 in the hour); flexpart_cloud_base fires "
        f"{int(out['fires_flexpart_cloud_base'].sum())}, glide {int(out['fires_glide'].sum())}",
        flush=True,
    )


if __name__ == "__main__":
    main()
