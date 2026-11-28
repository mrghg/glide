"""Decision 6 study: cost of the column diagnosis (`column_state`) with each trigger.

    PYTHONPATH=<worktree>/src .venv/bin/python scripts/validation/conv/trigger/bench_columns.py

EUR store, 8 Jul 2019 12 UTC, 40 levels, CPU (3 threads), median of 5 calls.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime, timedelta

import torch

import lpdm
from lpdm.convection import emanuel as em
from lpdm.met_reader import ArcoEra5ZarrReader, BoundingBoxRequest, SpatialBounds, TimeBounds
from lpdm.vertical_grid import stretched_agl_levels

print(f"lpdm: {lpdm.__file__}", flush=True)
torch.set_num_threads(3)
reader = ArcoEra5ZarrReader(
    zarr_store="data/era5-arco/EUR_20190621-20190715_ERA5-ARCO.zarr",
    channel_names=("u", "v", "w", "blh", "sp", "t", "q"),
    device="cpu",
    agl_levels_m=stretched_agl_levels(40, 15000.0, first_layer_m=10.0),
)
t0 = datetime(2019, 7, 8, 12, tzinfo=UTC)
met = reader.fetch_hourly_window(
    BoundingBoxRequest(
        spatial=SpatialBounds(-30.0, 40.0, 32.0, 72.0, 0.0, 15000.0),
        time=TimeBounds(start=t0, end=t0 + timedelta(hours=1)),
    )
)
for tr in ("glide", "flexpart_cloud_base", "glide", "flexpart_cloud_base"):
    s = em.EmanuelReducedConvection(trigger=tr)
    ts = []
    for _ in range(5):
        tic = time.perf_counter()
        c = s.column_state(met, t_alpha=0.0, device=torch.device("cpu"), dtype=torch.float32)
        ts.append(time.perf_counter() - tic)
    ts.sort()
    print(
        f"{tr:22s} column_state median {ts[2] * 1e3:7.1f} ms over {c.fires.numel()} columns; "
        f"fires {int(c.fires.sum())}"
    )
