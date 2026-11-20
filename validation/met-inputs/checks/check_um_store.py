"""Check a UM-derived ``glide`` store against the source files (GLIDE task B14).

For the chosen times, reads the source fields again and reports, as summary
statistics only (no field values are written anywhere):

1. Fields stored unchanged (T, q, w, surface pressure, BL depth) are
   bit-identical to the source, and the heat flux is exactly its negative.
2. Every destaggered wind lies within the range of the four source values it
   was interpolated from (two horizontal neighbours x two rho levels).
3. Pressure as GLIDE rebuilds it from the store (geopotential, surface
   pressure, T, q) against the UM's own pressure on theta levels (STASH 0/408).
4. Hydrostatic consistency of the geopotential: the geopotential thickness of
   each layer divided by R_d T_v ln(p_below / p_above), with the UM's pressure.
5. Ranges of boundary-layer depth and friction velocity.
6. Values at the EUR / NAM receptors (heights above ground of the lowest
   levels, ground height), to compare with what NAME reports.

Usage:
    $METBAZARR_REPO/.venv-jaspy/bin/python validation/met-inputs/checks/check_um_store.py \
        validation/met-inputs/um/recipes/EUR-day-20190108.yaml --times 0 4
"""

from __future__ import annotations

import argparse

import numpy as np
import xarray as xr
from metbazarr import build
from metbazarr.formats import glide
from metbazarr.recipe import load_recipe
from metbazarr.sources import um

RECEPTORS = {  # GLIDE docs/reference-runs.md §3
    "MHD": (53.3267, -9.9046),
    "JFJ": (46.5475, 7.9851),
    "CBW": (51.9703, 4.9264),
    "TAC": (52.5176, 1.1387),
    "LEF": (45.9451, -90.2732),
    "SGP": (36.6040, -97.4860),
}


def glide_model_level_pressure_pa(phi, phi_s, ps, t, q):
    """numpy copy of GLIDE's ``model_level_pressure_pa``
    (src/lpdm/vertical_grid.py at commit glide.GLIDE_SCHEMA_COMMIT), which
    cannot be imported without torch. Levels ordered surface to top."""
    tv = t * (1.0 + glide.TV_COEFF * q)
    dphi = np.empty_like(phi)
    dphi[0] = phi[0] - phi_s
    dphi[1:] = phi[1:] - phi[:-1]
    tvl = np.empty_like(tv)
    tvl[0] = tv[0]
    tvl[1:] = 0.5 * (tv[1:] + tv[:-1])
    return np.exp(np.log(ps)[None] - np.cumsum(dphi / (glide.R_DRY_AIR_J_KG_K * tvl), axis=0))


def pct(x, qs=(0, 1, 50, 99, 100)):
    return " / ".join(f"{v:.4g}" for v in np.nanpercentile(np.asarray(x), qs))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("recipe")
    ap.add_argument("--times", type=int, nargs="+", default=[0])
    args = ap.parse_args()

    recipe = load_recipe(args.recipe)
    p = build.plan(recipe)
    ds = xr.open_zarr(recipe.output, consolidated=True)
    phi = ds[glide.GEOPOTENTIAL].values.astype(np.float64)
    phi_s = ds[glide.SURFACE_GEOPOTENTIAL].values.astype(np.float64)
    orog = p.archive.orography(p.crop, p.parts)
    lat, lon = ds.latitude.values, ds.longitude.values

    print(f"# Checks of {recipe.output.name} against {recipe.source.get('series')}\n")
    print(f"Grid {lat.size} x {lon.size}, {ds.sizes['model_level']} levels; times {args.times}\n")

    for i in args.times:
        when = p.times[i]
        fields = (
            "u",
            "v",
            "w",
            "t",
            "q",
            "p_theta",
            "ps",
            "blh",
            "shf",
            "taux",
            "tauy",
        )
        src, levels = p.archive.read_time(when, p.crop, p.parts, fields=fields)
        st = ds.isel(time=i).load()
        print(f"## {when:%Y-%m-%d %H:%M}\n")

        # 1. unchanged fields
        print("| Field | store vs source |\n| --- | --- |")
        for name, key, sign in (
            ("temperature", "t", 1),
            ("specific_humidity", "q", 1),
            ("vertical_velocity", "w", 1),
            ("surface_pressure", "ps", 1),
            ("boundary_layer_height", "blh", 1),
            ("surface_sensible_heat_flux", "shf", -1),
        ):
            same = np.array_equal(st[name].values, (sign * src[key]).astype(np.float32))
            print(
                f"| {name} | {'bit-identical' if same else 'DIFFERENT'}{' (negated)' if sign < 0 else ''} |"
            )

        # 2. winds within their four source values
        for name, key, hslice in (
            ("u_component_of_wind", "u", lambda a: (a[..., :-1], a[..., 1:])),
            ("v_component_of_wind", "v", lambda a: (a[..., :-1, :], a[..., 1:, :])),
        ):
            a, b = hslice(src[key])
            lo4 = np.minimum.reduce([a[:-1], a[1:], b[:-1], b[1:]])
            hi4 = np.maximum.reduce([a[:-1], a[1:], b[:-1], b[1:]])
            out = st[name].values[:-1].astype(np.float64)
            tol = 1e-5 * np.maximum(1.0, np.abs(out))
            outside = int(np.sum((out < lo4 - tol) | (out > hi4 + tol)))
            top_ok = np.allclose(st[name].values[-1], 0.5 * (a[-1] + b[-1]), rtol=1e-6, atol=1e-6)
            near = out - 0.5 * (a[:-1] + b[:-1])  # change from the rho level below
            print(
                f"| {name} | {outside} of {out.size} values outside their 4 source values; "
                f"top level = top rho level: {top_ok}; change from rho level k: "
                f"|mean| {np.mean(np.abs(near)):.3f}, max {np.max(np.abs(near)):.2f} m/s |"
            )
        print()

        # 3. pressure as GLIDE rebuilds it
        t, q = (
            st["temperature"].values.astype(np.float64),
            st["specific_humidity"].values.astype(np.float64),
        )
        ps = st["surface_pressure"].values.astype(np.float64)
        p_glide = glide_model_level_pressure_pa(phi, phi_s, ps, t, q)
        p_um = src["p_theta"]
        rel = 100.0 * (p_glide / p_um - 1.0)
        print(
            "Pressure rebuilt as GLIDE does vs the UM's theta-level pressure (%, min / 1 / 50 / 99 / max):\n"
        )
        print(
            "| Level | a (m) | median p_UM (hPa) | rebuilt / UM - 1 (%) |\n| --- | --- | --- | --- |"
        )
        for k in (0, 4, 9, 19, 29, 39, 49, 58):
            print(
                f"| {k + 1} | {levels.a_theta[k]:.0f} | {np.median(p_um[k]) / 100:.1f} | {pct(rel[k])} |"
            )
        print(f"\nLargest |difference| anywhere: {np.max(np.abs(p_glide - p_um)) / 100:.3f} hPa\n")

        # 4. hydrostatic consistency with the UM's pressure
        tv = t * (1 + glide.TV_COEFF * q)
        ratio = (phi[1:] - phi[:-1]) / (
            glide.R_DRY_AIR_J_KG_K * 0.5 * (tv[1:] + tv[:-1]) * np.log(p_um[:-1] / p_um[1:])
        )
        print(f"Hydrostatic ratio dPhi / (R_d Tv dln p), all layers: {pct(ratio)}")
        print(f"  below 15 km: {pct(ratio[: np.searchsorted(levels.a_theta, 15000)])}\n")

        # 5. ranges
        rho = um.air_density(ps, t[0], q[0])
        print(f"Boundary-layer depth (m): {pct(st['boundary_layer_height'])}")
        print(f"Friction velocity (m/s): {pct(st['friction_velocity'])}")
        print(f"Air density at theta 1 (kg/m3): {pct(rho)}")
        print(
            f"Sensible heat flux, stored positive-down (W/m2): {pct(st['surface_sensible_heat_flux'])}\n"
        )

    # 6. receptors (grid-point nearest; heights above ground)
    print("## Receptors (nearest grid point)\n")
    print(
        "| Site | lat, lon of point | ground (m) | lowest theta levels AGL (m) |\n| --- | --- | --- | --- |"
    )
    for site, (la, lo) in RECEPTORS.items():
        if not (lat.min() <= la <= lat.max() and lon.min() <= lo <= lon.max()):
            continue
        j, k = int(np.argmin(abs(lat - la))), int(np.argmin(abs(lon - lo)))
        agl = (phi[:3, j, k] - phi_s[j, k]) / glide.GRAVITY_M_S2
        print(
            f"| {site} | {lat[j]:.4f}, {lon[k]:.4f} | {orog[j, k]:.1f} | "
            f"{', '.join(f'{z:.1f}' for z in agl)} |"
        )


if __name__ == "__main__":
    main()
