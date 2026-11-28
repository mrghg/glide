"""Decision 0031: per-column convection diagnosis on real meteorology, one code version.

Run once with the code before the change and once after (``PYTHONPATH`` picks
the version), then compare the two outputs with ``compare_trigger_counts.py``:

    CUDA_VISIBLE_DEVICES= PYTHONPATH=<checkout>/src .venv/bin/python \
        scripts/validation/cape/trigger_counts.py configs/ab_multisite_perf.yaml \
        <out.npz> --times 2023-12-27T12 2023-12-29T03 ...

For each met window (hour) it builds the reader as `lpdm.main` does for the
config (channels, AGL levels, met domain), calls the configured scheme's
``column_state`` at the window start (``t_alpha`` 0), and saves per column:
whether convection fires, CAPE, M_b, the LCL and cloud-top indices and the
cloud depth, plus the columns' latitudes. Nothing is modelled here; the
scheme is the code on ``PYTHONPATH``.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

import lpdm
from lpdm.config import RunConfig
from lpdm.convection import get_scheme as get_convection_scheme
from lpdm.main import _convection_scheme_kwargs, _met_domain_bounds, _met_interval
from lpdm.met_reader import ArcoEra5ZarrReader, BoundingBoxRequest, TimeBounds
from lpdm.turbulence import get_scheme


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("out")
    ap.add_argument("--times", nargs="+", required=True, help="window starts, UTC")
    ap.add_argument("--zarr-store", default=None, help="override io.zarr_store")
    ap.add_argument(
        "--met-domain",
        nargs=4,
        type=float,
        default=None,
        metavar=("LON0", "LON1", "LAT0", "LAT1"),
        help="override met_domain lon/lat bounds (for a store smaller than the config's)",
    )
    args = ap.parse_args()

    print(f"lpdm: {lpdm.__file__}", flush=True)
    torch.set_num_threads(8)
    cfg = RunConfig.from_yaml(args.config)
    store = args.zarr_store or cfg.io.zarr_store
    conv = get_convection_scheme(cfg.convection.scheme, **_convection_scheme_kwargs(cfg))
    turb = get_scheme(cfg.turbulence.scheme)
    channels = tuple(
        dict.fromkeys(
            ("u", "v", "w", "blh", "sp", *turb.required_met_keys(), *conv.required_met_keys())
        )
    )
    reader = ArcoEra5ZarrReader(
        zarr_store=store,
        channel_names=channels,
        device="cpu",
        agl_levels_m=cfg.met_domain.resolve_agl_levels(),
    )
    bounds = _met_domain_bounds(cfg)
    if args.met_domain is not None:
        lon0, lon1, lat0, lat1 = args.met_domain
        bounds = replace(bounds, lon_min=lon0, lon_max=lon1, lat_min=lat0, lat_max=lat1)
    out: dict[str, np.ndarray] = {}
    for stamp in args.times:
        t0 = datetime.fromisoformat(stamp).replace(tzinfo=UTC)
        tic = time.perf_counter()
        met = reader.fetch_hourly_window(
            BoundingBoxRequest(
                spatial=bounds,
                time=TimeBounds(start=t0, end=t0 + _met_interval(reader)),
            )
        )
        cols = conv.column_state(met, t_alpha=0.0, device=torch.device("cpu"), dtype=torch.float32)
        lat = np.asarray(met.metadata.lat, dtype=np.float64)
        lon = np.asarray(met.metadata.lon, dtype=np.float64)
        lat_c = np.repeat(lat, lon.size)  # column c = iy * n_lon + ix
        lon_c = np.tile(lon, lat.size)
        key = t0.strftime("%Y%m%dT%H")
        out[f"{key}/fires"] = cols.fires.numpy()
        out[f"{key}/cape"] = cols.cape.numpy()
        out[f"{key}/m_b"] = cols.m_b.numpy()
        out[f"{key}/i_lcl"] = cols.i_lcl.numpy()
        out[f"{key}/i_lnb"] = cols.i_lnb.numpy()
        out[f"{key}/dtv"] = cols.dtv_lcl_plus_1.numpy()
        out[f"{key}/depth"] = cols.cloud_depth_m.numpy()
        out[f"{key}/lat"] = lat_c
        out[f"{key}/lon"] = lon_c
        print(
            f"{key}: {cols.fires.numel()} columns x {cols.t.shape[1]} levels, "
            f"{int(cols.fires.sum())} fire, {time.perf_counter() - tic:.1f} s",
            flush=True,
        )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **out)


if __name__ == "__main__":
    main()
