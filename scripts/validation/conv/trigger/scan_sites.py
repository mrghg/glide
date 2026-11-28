"""Decision 6 study: choose the summer low-inlet case (b).

    PYTHONPATH=<worktree>/src .venv/bin/python scripts/validation/conv/trigger/scan_sites.py

On the EUR 2019 store (40 levels, defaults), every 3 h from 24 Jun to 15 Jul 2019,
counts the columns within ±3° of each candidate site that fire under GLIDE's
trigger and under FLEXPART's cloud-base trigger. Prints, per site and release day
(12 UTC), the column-hour totals over the 3 days before the release.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import torch

import lpdm
from lpdm.convection import emanuel as em
from lpdm.main import _met_interval
from lpdm.met_reader import ArcoEra5ZarrReader, BoundingBoxRequest, SpatialBounds, TimeBounds
from lpdm.vertical_grid import stretched_agl_levels

SITES = {
    "HUN (Hegyhatsal)": (16.652, 46.956),
    "KIT (Karlsruhe)": (8.425, 49.091),
    "IPR (Ispra)": (8.636, 45.815),
    "KRE (Kresin)": (15.080, 49.572),
    "TRN (Trainou)": (2.112, 47.965),
    "CBW (Cabauw)": (4.927, 51.971),
}
STORE = "data/era5-arco/EUR_20190621-20190715_ERA5-ARCO.zarr"


def main() -> None:
    print(f"lpdm: {lpdm.__file__}", flush=True)
    torch.set_num_threads(3)
    reader = ArcoEra5ZarrReader(
        zarr_store=STORE,
        channel_names=("u", "v", "w", "blh", "sp", "t", "q"),
        device="cpu",
        agl_levels_m=stretched_agl_levels(40, 15000.0, first_layer_m=10.0),
    )
    glide = em.EmanuelReducedConvection(trigger="glide")
    fp = em.EmanuelReducedConvection(trigger="flexpart_cloud_base")
    spatial = SpatialBounds(
        lon_min=-2.0, lon_max=21.0, lat_min=42.0, lat_max=55.0, z_min=0.0, z_max=15000.0
    )
    t = datetime(2019, 6, 24, tzinfo=UTC)
    rows = []
    while t < datetime(2019, 7, 15, 0, tzinfo=UTC):
        met = reader.fetch_hourly_window(
            BoundingBoxRequest(
                spatial=spatial, time=TimeBounds(start=t, end=t + _met_interval(reader))
            )
        )
        cg = glide.column_state(met, t_alpha=0.0, device=torch.device("cpu"), dtype=torch.float32)
        cf = fp.column_state(met, t_alpha=0.0, device=torch.device("cpu"), dtype=torch.float32)
        lon = np.asarray(met.metadata.lon, dtype=np.float64)
        lat = np.asarray(met.metadata.lat, dtype=np.float64)
        lo, la = np.meshgrid(lon, lat)
        lo, la = lo.ravel(), la.ravel()
        for s, (x, y) in SITES.items():
            box = (np.abs(lo - x) <= 3.0) & (np.abs(la - y) <= 3.0)
            rows.append(
                (
                    s,
                    t,
                    int(cg.fires.numpy()[box].sum()),
                    int(cf.fires.numpy()[box].sum()),
                    int(box.sum()),
                )
            )
        t += timedelta(hours=3)
    for s in SITES:
        print(
            f"\n{s}: release 12 UTC; column-hours (3-hourly samples) within ±3° over the 3 days before"
        )
        for d in range(27, 31 + 15):
            rel = datetime(2019, 6, 1, 12, tzinfo=UTC) + timedelta(days=d - 1)
            if rel > datetime(2019, 7, 14, 21, tzinfo=UTC):
                break
            sel = [r for r in rows if r[0] == s and rel - timedelta(days=3) <= r[1] <= rel]
            g = sum(r[2] for r in sel)
            f = sum(r[3] for r in sel)
            n = sum(r[4] for r in sel)
            print(f"  {rel:%Y-%m-%d}: glide {g:6d} ({g / n:6.2%})  fpcb {f:6d} ({f / n:6.2%})")


if __name__ == "__main__":
    main()
