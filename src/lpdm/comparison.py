"""Footprint comparison utilities.

Post-processing helpers for putting GLIDE footprints onto common ground with
external references (FLEXPART, NAME, STILT). Three pieces:

- `to_stilt_surface_footprint`: convert a raw GLIDE footprint to STILT-style
  surface sensitivity in `m**2 s mol**-1`, equivalent to `(mol/mol)/(mol/m**2/s)`,
  per Lin et al. 2003 Eq. 5. It reads the footprint's declared
  `density_weighting` (decision 0022): a `dry_air` footprint (s m3 kg-1, the
  default) needs M_air / h only; a `none` footprint (plain seconds) needs a
  density, applied afterwards.

- `surface_air_density_from_met`: build a spatially-varying surface air-density
  field `ρ(lat, lon)` from a met store (`sp` and `t`) for use as the
  `air_density_kg_m3` argument above, for **unweighted footprints only**.

- `flexpart_to_dry_air_footprint`, `name_to_dry_air_footprint`: the reference
  models' footprints in the same dry-air units (decision 0022,
  `docs/reference-runs.md` §6), with the per-cell humidity corrections
  `flexpart_dry_air_factor` and `name_dry_air_factor`.

- `regrid_conservative`: area-weighted mass-conservative regridding between
  rectangular lat/lon grids using `(sin(lat_top) - sin(lat_bottom)) × dlon`
  spherical-cell areas. Pure NumPy; no `xesmf` / `ESMPy` dependency. Limited to
  rectangular grids — sufficient for FLEXPART and NAME outputs.

None of these run in the main runtime; all are intended for comparison
notebooks and validation scripts.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

# Standard dry-air mean molar mass (kg/mol).
DRY_AIR_M_KG_MOL = 0.02897
# Dry-air gas constant (J/(kg·K)) — for the ideal-gas density helper.
R_DRY_AIR_J_KG_K = 287.05
# FLEXPART's virtual-temperature coefficient (Tv = T (1 + 0.608 q)).
FLEXPART_TV_COEFF = 0.608


def _density_weighting(raw_footprint: xr.DataArray) -> str:
    """The footprint's declared density weighting; stores written before
    decision 0022 carry none and hold plain seconds."""

    return str(raw_footprint.attrs.get("density_weighting", "none"))


# ---------------------------------------------------------------------------
# (a) STILT-style unit conversion
# ---------------------------------------------------------------------------


def to_stilt_surface_footprint(
    raw_footprint: xr.DataArray,
    *,
    surface_layer_depth_m: float,
    air_density_kg_m3: float | xr.DataArray | None = None,
    m_air_kg_per_mol: float = DRY_AIR_M_KG_MOL,
    integrate_time: bool = True,
) -> xr.DataArray:
    """Convert raw GLIDE residence-time footprint to STILT-style surface sensitivity.

    The footprint declares its weighting in ``attrs["density_weighting"]``
    (decision 0022):

    - ``dry_air`` (GLIDE's default): each particle was divided by the dry-air
      density as it was counted, so `F` is in s m3 kg-1 and

          f_STILT(y, x) = m_air / h * sum_{t, z in surface_layer} F[t, z, y, x]

      Passing ``air_density_kg_m3`` for such a footprint raises: it would divide
      by the density twice.
    - ``none`` (or no attribute: stores written before decision 0022): `F` is
      plain residence seconds, and per Lin et al. 2003 Eq. 5

          f_STILT(y, x) = m_air / (h * rho_bar) * sum_{t, z in surface_layer} F[t, z, y, x]

      with ``rho_bar`` the given air density. Dividing afterwards is biased
      where residence and density co-vary (decision 0022).

    `h` is the surface-layer depth used for the integration (typical FLEXPART /
    NAME convention: 0-40 m).

    Bins partially overlapping the chosen surface layer are credited by their
    overlap fraction (depth-weighted). For exact comparison, configure the runtime
    so the footprint has a z-bin whose extent matches the surface layer; otherwise
    the partial-overlap weighting introduces a small approximation error from
    assuming uniform residence-time density within each bin.

    Args:
            raw_footprint: 4D `xarray.DataArray` with dims `(time_ago, z_bin, latitude,
                    longitude)`. Must carry `z_bottom_m` and `z_top_m` coords on `z_bin`
                    (set by `FootprintGridder` via `main._build_footprint_dataset_metadata`).
                    Multi-release runs produce 5D stores `(release_time, time_ago, z_bin,
                    latitude, longitude)` — select one release with `.isel(release_time=i)`
                    before passing in.
            surface_layer_depth_m: Depth of the surface layer for sensitivity
                    integration (m). Match the reference dataset's convention.
            air_density_kg_m3: For an unweighted (``none``) footprint only, and then
                    required: surface air density, scalar or 2D `DataArray` on
                    `(latitude, longitude)`. Must be None for a ``dry_air`` footprint.
            m_air_kg_per_mol: Dry-air mean molar mass; default 0.02897 kg/mol.
            integrate_time: If True (default), sum across `time_ago` to return
                    `[latitude, longitude]`. Otherwise return `[time_ago, latitude, longitude]`.

    Returns:
            `xarray.DataArray` with units `m**2 s mol**-1` (equivalently
            `(mol/mol)/(mol/m**2/s)`). Attrs include the conversion parameters used.
    """

    expected_dims = {"time_ago", "z_bin", "latitude", "longitude"}
    if set(raw_footprint.dims) != expected_dims:
        raise ValueError(
            f"raw_footprint must have dims {expected_dims}; got {tuple(raw_footprint.dims)}"
        )
    for name in ("z_bottom_m", "z_top_m"):
        if name not in raw_footprint.coords:
            raise ValueError(
                f"raw_footprint is missing coord {name!r}; the FootprintGridder writer "
                "populates these — re-run with the current main.py if this is an old store."
            )

    if surface_layer_depth_m <= 0:
        raise ValueError("surface_layer_depth_m must be > 0")

    weighting = _density_weighting(raw_footprint)
    if weighting == "dry_air":
        if air_density_kg_m3 is not None:
            raise ValueError(
                "raw_footprint is already divided by the dry-air density at each particle "
                "(attrs density_weighting='dry_air', decision 0022); pass no "
                "air_density_kg_m3, or it is divided twice."
            )
    elif weighting == "none":
        if air_density_kg_m3 is None:
            raise ValueError(
                "raw_footprint holds plain residence seconds (density_weighting='none'); "
                "air_density_kg_m3 is required."
            )
    else:
        raise ValueError(f"unknown density_weighting {weighting!r} on raw_footprint")

    # Per-bin overlap fraction with [0, surface_layer_depth_m].
    z_bot = raw_footprint.coords["z_bottom_m"]
    z_top = raw_footprint.coords["z_top_m"]
    overlap_top = xr.where(z_top < surface_layer_depth_m, z_top, surface_layer_depth_m)
    overlap_bot = xr.where(z_bot > 0.0, z_bot, 0.0)
    overlap_depth = (overlap_top - overlap_bot).clip(min=0.0)
    bin_depth = (z_top - z_bot).clip(min=1e-9)
    z_weights = overlap_depth / bin_depth  # along z_bin, in [0, 1]

    surface_integrated = (raw_footprint * z_weights).sum(dim="z_bin")
    if integrate_time:
        surface_integrated = surface_integrated.sum(dim="time_ago")

    if weighting == "dry_air":
        conversion = m_air_kg_per_mol / float(surface_layer_depth_m)
    else:
        conversion = m_air_kg_per_mol / (float(surface_layer_depth_m) * air_density_kg_m3)
    result = surface_integrated * conversion

    result.attrs = dict(raw_footprint.attrs)
    result.attrs["units"] = "m**2 s mol**-1"
    result.attrs["stilt_surface_layer_depth_m"] = float(surface_layer_depth_m)
    result.attrs["stilt_m_air_kg_per_mol"] = float(m_air_kg_per_mol)
    if weighting == "dry_air":
        result.attrs["stilt_air_density_kg_m3"] = "per particle, while counting (dry air)"
    else:
        result.attrs["stilt_air_density_kg_m3"] = (
            float(air_density_kg_m3) if isinstance(air_density_kg_m3, (int, float)) else "DataArray"
        )
    result.attrs["conversion_reference"] = "Lin et al. 2003 J. Geophys. Res. Eq. 5"
    result.name = "footprint_stilt"
    return result


# ---------------------------------------------------------------------------
# (b) Surface air-density field from met
# ---------------------------------------------------------------------------


def surface_air_density_from_met(
    met_ds: xr.Dataset,
    *,
    sp_var: str = "surface_pressure",
    t_var: str = "temperature",
    t_lowest_level_kwargs: dict | None = None,
    time_reduce: str | None = "mean",
) -> xr.DataArray:
    """Build a spatially-varying surface air-density field `ρ(lat, lon)` from a
    met store, for use as ``air_density_kg_m3`` in :func:`to_stilt_surface_footprint`.

    **For unweighted footprints only** (``density_weighting: none``). A GLIDE
    footprint weighted while counting (``dry_air``, the default, decision 0022)
    takes no density afterwards; dividing afterwards is biased where residence
    and density co-vary.

    Implements ``ρ = sp / (R_d · T_surface)`` using the ideal gas law. Temperature
    is sampled at the lowest model level (or by the `t_lowest_level_kwargs`
    override) as a surface proxy. The result is a 2D field on the met's
    `(latitude, longitude)` grid; if the met has a time dim it's reduced by
    `time_reduce` (default `"mean"`), since the footprint conversion is normally
    a single-time operation. Set `time_reduce=None` to preserve time.

    Per Seibert & Frank (2004) Eq. 8 the source-receptor footprint is
    density-weighted at the source cell; using a scalar ρ in
    :func:`to_stilt_surface_footprint` biases the result by a few percent for
    deep-PBL receptors. Pass the output of this helper instead. (F10 in the
    2026-05-30 physics audit.)

    Args:
            met_ds: xarray Dataset opened from an ARCO ERA5 Zarr store (or any
                    store with surface-pressure and 3D temperature variables on
                    `(latitude, longitude)` and optionally `(level, time)`).
            sp_var: surface-pressure variable name. Default matches ARCO ERA5.
            t_var: temperature variable name (must have a `level` dim).
            t_lowest_level_kwargs: optional `xr.DataArray.isel(...)` kwargs to pick
                    a specific temperature level (e.g. `{"level": -1}`). Default selects
                    the lowest level present in the dataset.
            time_reduce: `"mean"` (default) reduces the `time` dim by averaging;
                    `None` preserves it.

    Returns:
            `xarray.DataArray` named ``air_density_kg_m3`` with dims
            `(latitude, longitude)` (and `time` if `time_reduce` is None).
    """

    if sp_var not in met_ds.variables:
        raise KeyError(f"met_ds missing surface-pressure variable {sp_var!r}")
    if t_var not in met_ds.variables:
        raise KeyError(f"met_ds missing temperature variable {t_var!r}")

    sp = met_ds[sp_var]  # Pa (or kPa — let units validation happen at use)
    t_field = met_ds[t_var]
    if "level" not in t_field.dims:
        raise ValueError(f"{t_var!r} must have a 'level' dim for the lowest-level selection")
    if t_lowest_level_kwargs is None:
        # Pick the lowest level (largest pressure ≈ surface). In ARCO ERA5 the
        # level coord is in hPa and stored in descending altitude order (so
        # the lowest altitude is at the *end* of the level array).
        t_surface = t_field.isel(level=int(t_field["level"].argmax()))
    else:
        t_surface = t_field.isel(**t_lowest_level_kwargs)

    rho = sp / (R_DRY_AIR_J_KG_K * t_surface)
    rho.name = "air_density_kg_m3"
    rho.attrs["units"] = "kg m**-3"
    rho.attrs["formula"] = "sp / (R_d · T_lowest_level)"
    rho.attrs["R_d_J_kg_K"] = float(R_DRY_AIR_J_KG_K)
    if time_reduce == "mean" and "time" in rho.dims:
        rho = rho.mean(dim="time")
        rho.attrs["time_reduce"] = "mean"
    elif time_reduce is not None and time_reduce != "mean":
        raise ValueError(f"time_reduce must be 'mean' or None; got {time_reduce!r}")
    return rho


# ---------------------------------------------------------------------------
# (b2) Reference models' footprints in dry-air units (decision 0022)
# ---------------------------------------------------------------------------


def _vapour_pressure(p, q):
    return q * p / (0.622 + 0.378 * q)


def _mean_over(field, time_dim: str | None):
    if time_dim is not None and isinstance(field, xr.DataArray) and time_dim in field.dims:
        return field.mean(dim=time_dim)
    return field


def name_dry_air_factor(pressure_pa, specific_humidity, *, time_dim: str | None = "time"):
    """NAME's moist-air footprint to dry air: p / (p - e), per cell.

    NAME divides by p / (R_d T), with no humidity term, so its footprint is
    relative to moist air (decision 0022). Pass the surface-layer pressure and
    specific humidity of NAME's own (UM) meteorology; with a ``time_dim`` the
    factor is computed at each time and averaged over the footprint's period.
    """

    factor = pressure_pa / (pressure_pa - _vapour_pressure(pressure_pa, specific_humidity))
    return _mean_over(factor, time_dim)


def flexpart_dry_air_factor(pressure_pa, specific_humidity, *, time_dim: str | None = "time"):
    """FLEXPART's footprint to dry air: p / ((p - e)(1 + 0.608 q)), per cell.

    FLEXPART divides each particle by the moist-air density p / (R_d T_v),
    T_v = T (1 + 0.608 q) (decision 0022). Pass FLEXPART's own meteorology;
    with a ``time_dim`` the factor is computed at each time and averaged over
    the footprint's period.
    """

    q = specific_humidity
    factor = pressure_pa / (
        (pressure_pa - _vapour_pressure(pressure_pa, q)) * (1.0 + FLEXPART_TV_COEFF * q)
    )
    return _mean_over(factor, time_dim)


def flexpart_to_dry_air_footprint(
    native_s_m3_kg: xr.DataArray,
    *,
    dry_air_factor: float | xr.DataArray,
    surface_layer_depth_m: float = 40.0,
    m_air_kg_per_mol: float = DRY_AIR_M_KG_MOL,
    time_dim: str | None = "time",
) -> xr.DataArray:
    """FLEXPART backward footprint to m2 s mol-1 of dry air.

    Input: native output with ``IND_SOURCE = 1``, ``IND_RECEPTOR = 2`` in
    s m3 kg-1 for the 0-h layer, per output time. Output:

        S = (sum over output times) x M_air / h x p / ((p - e)(1 + 0.608 q))

    ``dry_air_factor`` is :func:`flexpart_dry_air_factor` from FLEXPART's own
    meteorology over the footprint's period (scalar or a per-cell field). The
    function takes fields; it never reads restricted data itself.
    """

    if surface_layer_depth_m <= 0:
        raise ValueError("surface_layer_depth_m must be > 0")
    summed = native_s_m3_kg
    if time_dim is not None and time_dim in summed.dims:
        summed = summed.sum(dim=time_dim)
    result = summed * (m_air_kg_per_mol / float(surface_layer_depth_m)) * dry_air_factor
    result.attrs = dict(native_s_m3_kg.attrs)
    result.attrs.update(
        units="m**2 s mol**-1",
        density_weighting="dry_air",
        conversion="FLEXPART s m3 kg-1 (moist air) x M_air / h x p / ((p - e)(1 + 0.608 q))",
        surface_layer_depth_m=float(surface_layer_depth_m),
        m_air_kg_per_mol=float(m_air_kg_per_mol),
    )
    result.name = "footprint_dry_air"
    return result


def name_to_dry_air_footprint(
    mixing_ratio_ppm_s: xr.DataArray,
    *,
    cell_area_m2: float | xr.DataArray,
    dry_air_factor: float | xr.DataArray,
    release_seconds: float = 3600.0,
) -> xr.DataArray:
    """NAME's footprint to m2 s mol-1 of dry air (`docs/reference-runs.md` §6).

    Input: time-integrated Mixing Ratio in ppm s for the 0-40 m layer, per cell.
    Output:

        S = ppm s x 1e-6 x cell area / release_seconds x p / (p - e)

    ``release_seconds`` is the NAME runner's source normalisation (3,600 s,
    the release hour). ``dry_air_factor`` is :func:`name_dry_air_factor` from
    the UM fields NAME used, over the footprint's period.
    """

    if release_seconds <= 0:
        raise ValueError("release_seconds must be > 0")
    result = mixing_ratio_ppm_s * (1.0e-6 / float(release_seconds)) * cell_area_m2
    result = result * dry_air_factor
    result.attrs = dict(mixing_ratio_ppm_s.attrs)
    result.attrs.update(
        units="m**2 s mol**-1",
        density_weighting="dry_air",
        conversion="NAME ppm s (moist air) x 1e-6 x cell area / release_seconds x p / (p - e)",
        release_seconds=float(release_seconds),
    )
    result.name = "footprint_dry_air"
    return result


# ---------------------------------------------------------------------------
# (c) Conservative rectangular-grid regridder
# ---------------------------------------------------------------------------


def regrid_conservative(
    src: xr.DataArray,
    *,
    target_latitude: np.ndarray,
    target_longitude: np.ndarray,
    src_lat_dim: str = "latitude",
    src_lon_dim: str = "longitude",
) -> xr.DataArray:
    """Area-weighted, mass-conservative regridding between rectangular lat/lon grids.

    Each source cell value is redistributed to overlapping target cells in
    proportion to spherical-cell-area overlap. Total integrated value over the
    source-target intersection region is preserved exactly; values outside the
    target extent are dropped (and inversely, target cells outside the source
    extent get zero contribution).

    Cell areas use the standard spherical approximation
    `A ∝ (sin(lat_top) - sin(lat_bottom)) × (lon_east - lon_west)`. The factorisation
    holds for any rectangular lat/lon grid (uniform or non-uniform spacing); for
    curvilinear or unstructured grids use a dedicated tool (e.g. xesmf).

    Treats input as a per-cell quantity. For a per-area quantity, multiply by
    source-cell area before regridding and divide by target-cell area afterwards.

    Args:
            src: source `xarray.DataArray` with at least the named lat/lon dims.
                    Other dimensions (e.g. `time_ago`, `z_bin`) are preserved.
            target_latitude: target latitude cell centres, 1D ascending.
            target_longitude: target longitude cell centres, 1D ascending.
            src_lat_dim, src_lon_dim: name of the lat/lon dims in `src`.

    Returns:
            Regridded `xarray.DataArray` with `(latitude, longitude)` replaced by the
            target grid; other dims and coords preserved.
    """

    if src_lat_dim not in src.dims:
        raise ValueError(f"src is missing lat dim {src_lat_dim!r}")
    if src_lon_dim not in src.dims:
        raise ValueError(f"src is missing lon dim {src_lon_dim!r}")

    src_lat_centres = np.asarray(src.coords[src_lat_dim].values, dtype=float)
    src_lon_centres = np.asarray(src.coords[src_lon_dim].values, dtype=float)
    tgt_lat_centres = np.asarray(target_latitude, dtype=float)
    tgt_lon_centres = np.asarray(target_longitude, dtype=float)

    src_lat_edges = _edges_from_centres(src_lat_centres)
    src_lon_edges = _edges_from_centres(src_lon_centres)
    tgt_lat_edges = _edges_from_centres(tgt_lat_centres)
    tgt_lon_edges = _edges_from_centres(tgt_lon_centres)

    W_lat = _conservative_weights_1d(src_lat_edges, tgt_lat_edges, axis_kind="lat")
    W_lon = _conservative_weights_1d(src_lon_edges, tgt_lon_edges, axis_kind="lon")

    # Move lat/lon to the trailing axes for broadcasting-friendly matmul.
    other_dims = [d for d in src.dims if d not in (src_lat_dim, src_lon_dim)]
    src_t = src.transpose(*other_dims, src_lat_dim, src_lon_dim)
    src_arr = np.asarray(src_t.values, dtype=float)

    # First regrid lon: (..., M_src, N_src) @ (N_src, N_tgt) -> (..., M_src, N_tgt)
    intermediate = src_arr @ W_lon
    # Then regrid lat: contract i over (..., M_src, N_tgt) and (M_src, M_tgt).
    # Output shape: (..., M_tgt, N_tgt).
    result = np.einsum("...ij,ik->...kj", intermediate, W_lat)

    new_coords: dict[str, np.ndarray] = {}
    for dim in other_dims:
        if dim in src.coords:
            new_coords[dim] = src.coords[dim].values
    new_coords[src_lat_dim] = tgt_lat_centres
    new_coords[src_lon_dim] = tgt_lon_centres

    return xr.DataArray(
        result,
        dims=tuple(other_dims) + (src_lat_dim, src_lon_dim),
        coords=new_coords,
        name=src.name,
        attrs=dict(src.attrs),
    )


def _edges_from_centres(centres: np.ndarray) -> np.ndarray:
    """Infer cell edges from centres as midpoints; outer edges extrapolate symmetrically."""

    centres = np.asarray(centres, dtype=float)
    if centres.size == 0:
        raise ValueError("centres must be non-empty")
    if centres.size == 1:
        return np.array([centres[0] - 0.5, centres[0] + 0.5], dtype=float)
    if not np.all(np.diff(centres) > 0):
        raise ValueError("centres must be strictly ascending")

    edges = np.empty(centres.size + 1, dtype=float)
    edges[1:-1] = 0.5 * (centres[:-1] + centres[1:])
    edges[0] = centres[0] - 0.5 * (centres[1] - centres[0])
    edges[-1] = centres[-1] + 0.5 * (centres[-1] - centres[-2])
    return edges


def _conservative_weights_1d(
    src_edges: np.ndarray,
    tgt_edges: np.ndarray,
    *,
    axis_kind: str,
) -> np.ndarray:
    """Build (n_src, n_tgt) weight matrix for area-weighted 1D regridding.

    `axis_kind="lat"` uses `sin(top) - sin(bot)` as the cell-area factor;
    `axis_kind="lon"` uses `(east - west)` in radians. Each entry is the source
    cell's overlap fraction with the target cell, suitable for redistributing a
    per-cell value: `target[j] = sum_i src[i] * W[i, j]`.
    """

    if axis_kind == "lat":
        src_factors = np.sin(np.deg2rad(src_edges[1:])) - np.sin(np.deg2rad(src_edges[:-1]))
    elif axis_kind == "lon":
        src_factors = np.deg2rad(src_edges[1:] - src_edges[:-1])
    else:
        raise ValueError(f"axis_kind must be 'lat' or 'lon'; got {axis_kind!r}")

    n_src = src_edges.size - 1
    n_tgt = tgt_edges.size - 1
    W = np.zeros((n_src, n_tgt), dtype=float)

    for i in range(n_src):
        s_low, s_high = src_edges[i], src_edges[i + 1]
        factor = src_factors[i]
        if factor <= 0:
            continue
        for j in range(n_tgt):
            t_low, t_high = tgt_edges[j], tgt_edges[j + 1]
            o_low = max(s_low, t_low)
            o_high = min(s_high, t_high)
            if o_high <= o_low:
                continue
            if axis_kind == "lat":
                overlap = np.sin(np.deg2rad(o_high)) - np.sin(np.deg2rad(o_low))
            else:
                overlap = np.deg2rad(o_high - o_low)
            W[i, j] = overlap / factor

    return W


__all__ = [
    "DRY_AIR_M_KG_MOL",
    "flexpart_dry_air_factor",
    "flexpart_to_dry_air_footprint",
    "name_dry_air_factor",
    "name_to_dry_air_footprint",
    "regrid_conservative",
    "surface_air_density_from_met",
    "to_stilt_surface_footprint",
]
