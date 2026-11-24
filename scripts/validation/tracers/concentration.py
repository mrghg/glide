"""From a backward footprint and a known source to a sample's concentration.

Queue task B12. Derivation (``data/tracer-experiments.md`` §5, step 5):

GLIDE's raw footprint ``f[r, k, z, y, x]`` (``footprints.zarr``, variable
``footprint``) is the residence time, in seconds (divided by the dry-air density
for a ``dry_air``-weighted store, see below), of release ``r``'s
particles in cell ``(z, y, x)`` during age bin ``k``, per unit released
weight (the weights of one release sum to 1; ``docs/physics.md`` §7). Ages
count back from the END of the release window in bins of
``output_grid.time_bin_seconds`` (decision 0016), so bin ``k`` of release
``r`` covers the absolute interval ``(T_r - (k+1)Δ, T_r - kΔ]`` with
``T_r`` the window end.

Release ``r`` puts its particles uniformly over the sample's volume ``V_r``
and period ``[t_0, t_1]``. By the adjoint relation (Seibert and Frank,
2004), for a source ``S(x, t)`` (kg m⁻³ s⁻¹) the sample's mean mixing
ratio is the residence-weighted source divided by the air in the source
volume. For a source that emits mass ``M`` uniformly over a box of volume
``V_s`` (area ``A_s`` × layer depth ``h``) and over ``[E_0, E_1]``, at the
rate ``Q = M / (E_1 - E_0)``:

    mass mixing ratio   q = Q / V_s × Σ_k φ_k τ_k / ρ_k
    τ_k = Σ_{cells in box} f[r, k, layer, y, x]      (s)
    φ_k = |bin k ∩ [E_0, E_1]| / Δ                   (fraction of the bin that emits)

with ``ρ_k`` the dry-air density in the source box during bin ``k``. The
emission is assumed uniform within each bin; when every time involved is a
multiple of Δ, ``φ_k`` is 0 or 1 and the timing is exact (the benchmark
chooses Δ so).

**What the observations are.** DATEM's (and JRC's) mass concentrations are
measured volume ratios multiplied by one fixed factor, not by the ambient
air density (decision 0022, "The tracer observations"): for CAPTEX and
ANATEX pg m⁻³ = fL L⁻¹ × M / 22.40 L mol⁻¹. So the comparable model value
is the dry-air mole fraction ``χ = q M_air / M`` times ``M / V_m``:

    c_obs-basis = χ M / V_m = Q / V_s × (M_air / V_m) × Σ_k φ_k τ_k / ρ_k    (basis "stp")

The tracer's molar mass cancels. ``M_air / V_m`` is 1.2933 kg m⁻³ for
``V_m`` = 22.40 L mol⁻¹. The alternative, the ambient mass concentration,

    c_ambient = Q / V_s × Σ_k φ_k τ_k                                        (basis "ambient")

differs from it by ``ρ_k / (M_air / V_m)``, 10–15% in a warm boundary
layer, which is the spurious factor decision 0022 warns about.

Which basis a store supports follows from its ``density_weighting``
attribute (A04, decision 0022). GLIDE's default, ``dry_air``, stores
``τ/ρ`` per particle (s m³ kg⁻¹): the "stp" basis uses it as it is and reads
no meteorology, and the "ambient" basis is refused (it would need the density
multiplied back in). A ``none`` store (``--set output_grid.density_weighting=none``)
holds plain seconds: "ambient" uses it as it is, and "stp" divides each age
bin's residence by the dry-air density of the meteorology at the source,
interpolated in time to the bin's centre, which captures how residence and
density vary together through the day to the resolution of the bins but not
within the box or the 40 m layer (decision 0022 estimates the error of dividing
afterwards at a few percent). A store with no attribute predates A04 and is
read as plain seconds, with a warning.

Areas use the model's own metric (``GPUEngine.horizontal_turbulent_displacement``:
110,540 m per degree of latitude and 111,320 cos φ m per degree of
longitude), so that the volume and the residence time are measured on the
same Earth.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

#: Metres per degree in GLIDE's advection (src/lpdm/gpu_engine.py).
M_PER_DEG_LAT = 110540.0
M_PER_DEG_LON_EQUATOR = 111320.0
#: Molar mass of dry air (src/lpdm/comparison.py DRY_AIR_M_KG_MOL).
M_AIR_KG_MOL = 0.02897
#: Molar volume behind DATEM's conversion of fL/L to pg m-3 (decision 0022).
DATEM_MOLAR_VOLUME_M3_MOL = 0.02240
R_DRY_AIR_J_KG_K = 287.05

BASES = ("stp", "ambient")


@dataclass(frozen=True)
class Emission:
    """One tracer release: ``mass`` (kg, or Bq) emitted uniformly over [start, end)."""

    start: pd.Timestamp
    end: pd.Timestamp
    mass: float
    lat: float
    lon: float

    @property
    def rate(self) -> float:
        return self.mass / (self.end - self.start).total_seconds()


def cell_area_m2(lat_edges: np.ndarray, lon_edges: np.ndarray) -> np.ndarray:
    """Cell areas (ny, nx) in the model's metric.

    ∫∫ 110540 × 111320 cos φ dφ dλ over the cell, with φ and λ in degrees.
    """

    lat_edges = np.asarray(lat_edges, dtype=np.float64)
    lon_edges = np.asarray(lon_edges, dtype=np.float64)
    band = (
        M_PER_DEG_LAT
        * M_PER_DEG_LON_EQUATOR
        * (180.0 / math.pi)
        * np.abs(np.diff(np.sin(np.deg2rad(lat_edges))))
    )
    return band[:, None] * np.abs(np.diff(lon_edges))[None, :]


def box_overlap(edges: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Fraction of each 1-D cell ``[edges[i], edges[i+1]]`` inside ``[lo, hi]``."""

    edges = np.asarray(edges, dtype=np.float64)
    a, b = np.minimum(edges[:-1], edges[1:]), np.maximum(edges[:-1], edges[1:])
    return np.clip(np.minimum(b, hi) - np.maximum(a, lo), 0.0, None) / (b - a)


def source_box_weights(
    lat_edges: np.ndarray, lon_edges: np.ndarray, lat: float, lon: float, size_deg: float
) -> np.ndarray:
    """Fraction of each footprint cell inside the ``size_deg`` box centred on the source."""

    h = 0.5 * size_deg
    # Put the source in the grid's longitude convention (0..360 or -180..180).
    if float(np.max(lon_edges)) > 180.0 and lon < 0.0:
        lon += 360.0
    elif float(np.min(lon_edges)) < 0.0 and lon > 180.0:
        lon -= 360.0
    wy = box_overlap(lat_edges, lat - h, lat + h)
    wx = box_overlap(lon_edges, lon - h, lon + h)
    w = wy[:, None] * wx[None, :]
    if not np.any(w > 0):
        raise ValueError(
            f"the {size_deg}° source box at ({lat}, {lon}) does not overlap the footprint grid"
        )
    return w


def emission_overlap(
    window_end: np.ndarray,
    age_start_s: np.ndarray,
    age_end_s: np.ndarray,
    emis_start: pd.Timestamp,
    emis_end: pd.Timestamp,
) -> np.ndarray:
    """Fraction of each (release, age bin) interval during which the source emits.

    ``window_end`` (n_release,) are the release windows' ends as
    ``datetime64``; ``age_start_s``/``age_end_s`` (n_t,) the bins' age edges
    in seconds. Bin k of release r covers ``(T_r - age_end, T_r - age_start]``.
    Returns (n_release, n_t).
    """

    t_end = (np.asarray(window_end, dtype="datetime64[ns]").astype(np.int64) / 1e9)[:, None]
    lo = t_end - np.asarray(age_end_s, dtype=np.float64)[None, :]
    hi = t_end - np.asarray(age_start_s, dtype=np.float64)[None, :]
    e0 = pd.Timestamp(emis_start).value / 1e9
    e1 = pd.Timestamp(emis_end).value / 1e9
    width = hi - lo
    return np.clip(np.minimum(hi, e1) - np.maximum(lo, e0), 0.0, None) / width


def bin_centre_times(window_end: np.ndarray, age_start_s, age_end_s) -> np.ndarray:
    """Absolute centre time of every (release, age bin), as ``datetime64[ns]``."""

    t_end = np.asarray(window_end, dtype="datetime64[ns]")[:, None]
    mid = 0.5 * (np.asarray(age_start_s, float) + np.asarray(age_end_s, float))
    return t_end - (mid * 1e9).astype("timedelta64[ns]")[None, :]


#: A callable giving the dry-air density (kg m-3) at the source for an array of times.
DensityFn = Callable[[np.ndarray], np.ndarray]


def predict(
    footprint: np.ndarray,
    *,
    window_end: np.ndarray,
    age_start_s: np.ndarray,
    age_end_s: np.ndarray,
    lat_edges: np.ndarray,
    lon_edges: np.ndarray,
    layer_depth_m: float,
    emissions: Sequence[Emission],
    source_deg: float,
    basis: str = "stp",
    density: DensityFn | None = None,
    density_weighted: bool = False,
    molar_volume_m3_mol: float = DATEM_MOLAR_VOLUME_M3_MOL,
) -> np.ndarray:
    """Predicted concentration (kg m-3 on the chosen basis) for every release.

    ``footprint`` is the source layer of the raw footprint, shape
    (n_release, n_t, ny, nx), in s (or s m³ kg⁻¹ when ``density_weighted``).
    ``emissions`` all come from the same site, at the centre of the source
    box. See the module docstring for the formulae and the two bases.
    """

    if basis not in BASES:
        raise ValueError(f"basis must be one of {BASES}")
    fp = np.asarray(footprint, dtype=np.float64)
    if fp.ndim != 4:
        raise ValueError("footprint must be (release, time_ago, lat, lon)")
    n_rel, n_t = fp.shape[:2]
    if n_t == 1 and len(age_start_s) == 1:
        # A time-integrated footprint cannot be cut to the emission window.
        raise ValueError("the footprint has one time bin; the emission window cannot be applied")
    if density_weighted and basis == "ambient":
        raise ValueError("an ambient concentration needs an unweighted footprint")
    if basis == "stp" and not density_weighted and density is None:
        raise ValueError("basis 'stp' with an unweighted footprint needs the source density")

    out = np.zeros(n_rel, dtype=np.float64)
    for e in emissions:
        w = source_box_weights(lat_edges, lon_edges, e.lat, e.lon, source_deg)
        volume = float(np.sum(w * cell_area_m2(lat_edges, lon_edges))) * layer_depth_m
        tau = np.einsum("rkyx,yx->rk", fp, w)  # residence in the box per age bin
        phi = emission_overlap(window_end, age_start_s, age_end_s, e.start, e.end)
        weight = phi * tau
        if basis == "stp":
            factor = M_AIR_KG_MOL / molar_volume_m3_mol
            if not density_weighted:
                used = weight > 0
                rho = np.ones_like(weight)
                if np.any(used):
                    times = bin_centre_times(window_end, age_start_s, age_end_s)[used]
                    rho[used] = density(times)
                weight = weight / rho
            weight = weight * factor
        out += e.rate / volume * weight.sum(axis=1)
    return out


def dry_air_density(p_pa, t_k, q_kg_kg) -> np.ndarray:
    """ρ_d = (p − e) / (R_d T), e = q p / (0.622 + 0.378 q) (decision 0022)."""

    p = np.asarray(p_pa, dtype=np.float64)
    q = np.asarray(q_kg_kg, dtype=np.float64)
    e = q * p / (0.622 + 0.378 * q)
    return (p - e) / (R_DRY_AIR_J_KG_K * np.asarray(t_k, dtype=np.float64))


def met_source_density(store: str, lat: float, lon: float) -> DensityFn:
    """Dry-air density at the source from a GLIDE meteorology store.

    Surface pressure, and temperature and specific humidity on the lowest
    model level (about 10 m above the ground), at the grid point nearest the
    source, interpolated linearly in time. Reads one column per hour only.
    """

    import xarray as xr

    ds = xr.open_zarr(store)
    lon_c = lon % 360.0 if float(ds["longitude"].max()) > 180.0 else lon
    pt = ds.sel(latitude=lat, longitude=lon_c, method="nearest")
    level_dim = next(
        d for d in ds["temperature"].dims if d not in ("time", "latitude", "longitude")
    )
    # Model levels count down to 137 at the ground; pressure levels end at the largest
    # pressure. Either way the lowest level has the largest coordinate value.
    lowest = int(np.argmax(ds[level_dim].values))
    t = pt["temperature"].isel({level_dim: lowest}).values
    q = pt["specific_humidity"].isel({level_dim: lowest}).values
    p = pt["surface_pressure"].values
    rho = dry_air_density(p, t, q)
    tsec = ds["time"].values.astype("datetime64[ns]").astype(np.int64) / 1e9

    def density(times: np.ndarray) -> np.ndarray:
        x = np.asarray(times, dtype="datetime64[ns]").astype(np.int64) / 1e9
        if np.any(x < tsec[0]) or np.any(x > tsec[-1]):
            raise ValueError("source density requested outside the meteorology's period")
        return np.interp(x, tsec, rho)

    return density


def is_density_weighted(ds) -> bool:
    """Whether a footprint store is already divided by the dry-air density (task A04)."""

    attrs = {**ds.attrs, **ds["footprint"].attrs}
    weighting = attrs.get("density_weighting")
    if weighting is None:
        # Stores written before A04 carry no attribute and hold plain seconds.
        warnings.warn(
            "the footprint store declares no density_weighting; treating it as plain "
            "residence seconds (a store written before decision 0022)",
            stacklevel=2,
        )
        return False
    expected_units = {"dry_air": "s m3 kg-1", "none": "s"}
    if weighting not in expected_units:
        raise ValueError(f"unknown density_weighting {weighting!r} on the footprint store")
    units = attrs.get("units")
    if units is not None and units != expected_units[weighting]:
        raise ValueError(
            f"footprint store says density_weighting={weighting!r} but units={units!r}"
        )
    return weighting == "dry_air"


def predict_from_dataset(
    ds,
    emissions: Sequence[Emission],
    *,
    source_deg: float,
    basis: str = "stp",
    density: DensityFn | None = None,
    molar_volume_m3_mol: float = DATEM_MOLAR_VOLUME_M3_MOL,
) -> np.ndarray:
    """:func:`predict` on a GLIDE footprint store (``footprints.zarr`` opened with
    xarray): the lowest layer, the release windows' ends and the age-bin edges are
    read from its coordinates. One value per release, in the store's order."""

    src = (
        ds["footprint"].isel(z_bin=0).transpose("release", "time_ago", "latitude", "longitude")
    ).values
    window_end = ds["release_time"].values + (
        np.round(ds["release_duration_seconds"].values).astype(np.int64) * 10**9
    ).astype("timedelta64[ns]")
    # Edges are stored in hours; round back to the millisecond so that bins on
    # whole seconds overlap emission windows exactly.
    age_start = np.round(ds["time_ago_start_hours"].values * 3600.0, 3)
    age_end = np.round(ds["time_ago_end_hours"].values * 3600.0, 3)
    return predict(
        src,
        window_end=window_end,
        age_start_s=age_start,
        age_end_s=age_end,
        lat_edges=ds["latitude_edge"].values,
        lon_edges=ds["longitude_edge"].values,
        layer_depth_m=float(ds["z_top_m"].values[0] - ds["z_bottom_m"].values[0]),
        emissions=emissions,
        source_deg=source_deg,
        basis=basis,
        density=density,
        density_weighted=is_density_weighted(ds),
        molar_volume_m3_mol=molar_volume_m3_mol,
    )
