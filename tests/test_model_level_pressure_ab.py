"""Task A01: GLIDE's hydrostatic model-level pressure against ECMWF's L137 a/b.

Decision 0010 keeps hybrid a/b out of production and reconstructs pressure from
geopotential. These tests check that reconstruction against the IFS definition,
full-level p = (p_half(k-1) + p_half(k)) / 2 with p_half = a + b p_s, using
ECMWF's published table (tests/data/ecmwf_l137_ab.csv, source in its header).
"""

import glob
import os
from pathlib import Path

import numpy as np
import pytest

from lpdm.vertical_grid import model_level_pressure_pa

TABLE = Path(__file__).parent / "data" / "ecmwf_l137_ab.csv"
R_D, G = 287.06, 9.80665


def _ab() -> tuple[np.ndarray, np.ndarray]:
    tab = np.loadtxt(TABLE, delimiter=",", comments=("#", "n"))
    assert tab.shape == (138, 3)
    return tab[:, 1], tab[:, 2]


def _ifs_column(ps: float, zs_m: float):
    """An IFS-consistent column: full-level geopotential from a/b and T, q by the
    IFS hydrostatic discretisation (Simmons and Burridge 1981), as ERA5 does."""

    a, b = _ab()
    ph = a + b * ps  # half levels 0..137, top to surface
    pf = 0.5 * (ph[:-1] + ph[1:])  # full levels 1..137
    z_m = 7000.0 * np.log(ps / pf) + zs_m  # rough height, only for the T profile
    t = np.maximum(288.0 - 0.0065 * (z_m - 0.0), 216.65)
    q = 0.008 * np.exp(-(z_m - zs_m) / 2000.0)
    tv = t * (1.0 + 0.6078 * q)
    phi_half = np.empty(138)
    phi_half[137] = G * zs_m
    phi_full = np.empty(137)
    for k in range(136, -1, -1):  # full level k+1 between half levels k and k+1
        lnr = np.log(ph[k + 1] / ph[k]) if ph[k] > 0 else 0.0
        alpha = np.log(2.0) if k == 0 else 1.0 - ph[k] / (ph[k + 1] - ph[k]) * lnr
        phi_full[k] = phi_half[k + 1] + alpha * R_D * tv[k]
        phi_half[k] = phi_half[k + 1] + R_D * tv[k] * lnr
    col = lambda v: v[:, None, None]  # noqa: E731
    return col(phi_full), np.full((1, 1), G * zs_m), np.full((1, 1), ps), col(t), col(q), pf


@pytest.mark.parametrize(
    ("ps", "zs_m"), [(101325.0, 0.0), (85000.0, 1500.0)], ids=["sea", "mountain"]
)
def test_hydrostatic_pressure_matches_l137_ab_on_an_ifs_column(ps: float, zs_m: float) -> None:
    phi, phi_s, p_s, t, q, pf = _ifs_column(ps, zs_m)
    p = model_level_pressure_pa(phi, phi_s, p_s, t, q)[:, 0, 0]
    rel = np.abs(p - pf) / pf
    # Lowest 1.5 km (where the boundary layer, cloud base and footprint layer are):
    low = pf > 0.85 * ps
    assert rel[low].max() < 1e-4  # measured ~1e-5 (under 1 Pa)
    # Whole column, apart from the top few levels (p < 1 hPa), where the
    # discretisations differ most and nothing GLIDE uses lives:
    assert rel[pf > 100.0].max() < 5e-3


ARCO = sorted(
    glob.glob(
        os.path.join(
            os.environ.get("GLIDE_DATA", "/nonexistent"), "era5-arco", "EUROPE_*_ERA5-ARCO.zarr"
        )
    )
)


@pytest.mark.skipif(not ARCO, reason="needs $GLIDE_DATA/era5-arco/EUROPE_* model-level stores")
def test_hydrostatic_pressure_matches_l137_ab_on_era5_over_sea() -> None:
    """On real ERA5 over the Atlantic, where the store's surface geopotential is
    consistent with its model levels (task A01 found it is not in steep terrain)."""

    import xarray as xr

    ds = (
        xr.open_zarr(ARCO[0])
        .isel(time=0)
        .sel(latitude=slice(55.0, 50.0), longitude=slice(-30.0, -20.0))
        .load()
    )
    a, b = _ab()
    k = ds["hybrid"].values.astype(int)
    ps = ds["surface_pressure"].values.astype(np.float64)
    pf = 0.5 * ((a[k - 1] + b[k - 1] * ps[..., None]) + (a[k] + b[k] * ps[..., None]))
    pf = np.moveaxis(pf, -1, 0)
    p = model_level_pressure_pa(
        ds["geopotential"].values,
        ds["geopotential_at_surface"].values,
        ps,
        ds["temperature"].values,
        ds["specific_humidity"].values,
    )
    rel = np.abs(p - pf) / pf
    assert np.abs(p - pf)[pf > 85000.0].max() < 50.0  # Pa, lowest ~1.5 km
    assert np.median(rel) < 1e-3
