"""ERA5 boundary-layer conditions at a site, for choosing and describing test
cases (decision 0030): h, u*, the upward sensible heat flux H (GLIDE's sign) and
the stable floor-layer depth sigma_min·h/(1.3 u*), hourly over a window.

Usage: era5_site_conditions.py STORE LON LAT START END"""

import sys

import numpy as np
import xarray as xr

store, lon, lat, start, end = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), *sys.argv[4:6]
ds = xr.open_zarr(store)
acc = float(ds.attrs.get("glide_flux_accumulation_seconds", 3600))
p = ds.sel(longitude=lon, latitude=lat, method="nearest").sel(time=slice(start, end))
p = p[["boundary_layer_height", "friction_velocity", "surface_sensible_heat_flux"]].load()
h = p.boundary_layer_height.values
us = p.friction_velocity.values
upward_h = -p.surface_sensible_heat_flux.values / acc  # ECMWF: downward positive, J m-2
d = 1e-3 * h / (1.3 * np.maximum(us, 1e-3))
print(f"{store.rsplit('/', 1)[-1]} at lon {lon}, lat {lat} (nearest grid point)")
for t, a, b, c, e in zip(p.time.values, h, us, upward_h, d, strict=True):
    print(
        f"{str(t)[:16]}  h {a:7.1f} m  u* {b:5.3f} m/s  H {c:7.1f} W/m2  floor layer if stable {e:5.2f} m"
    )
