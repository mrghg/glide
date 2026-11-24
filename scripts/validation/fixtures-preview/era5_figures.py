"""ERA5 behind the golden fixtures: the EUROPE stores (met-bazarr v0.3.0, repaired
v0.4.0), Dec 2023 and Jan 2024, around the reference config's releases.

    .venv/bin/python scripts/validation/fixtures-preview/era5_figures.py OUTDIR
"""

import sys
from pathlib import Path

import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).parent))
from style import GREY, INK2, SEQ, SITE_COORDS, SITES, mark_sites, plt  # noqa: E402

G = 9.80665
DEC = "data/era5-arco/EUROPE_20231201-20231231_ERA5-ARCO.zarr"
JAN = "data/era5-arco/EUROPE_20240101-20240131_ERA5-ARCO.zarr"


def domain(out: Path) -> None:
    ds = xr.open_zarr(JAN)
    zs = ds["geopotential_at_surface"].isel(time=0).load() / G
    fig, ax = plt.subplots(figsize=(9, 4.8), constrained_layout=True)
    im = ax.pcolormesh(
        ds.longitude, ds.latitude, zs.where(zs > 0), cmap=GREY, vmin=0, vmax=3000, shading="auto"
    )
    ax.contour(ds.longitude, ds.latitude, zs, levels=[2.0], colors=[INK2], linewidths=0.4)
    ax.add_patch(
        plt.Rectangle((-98.0, 10.75), 137.5, 68.25, fill=False, ec="#2a78d6", lw=1.5, ls="--")
    )
    ax.text(-95, 14, "met_domain = the stores' extent", color="#2a78d6", fontsize=9)
    mark_sites(ax, label=False)
    ax.annotate(
        "MHD · RGL · TAC",
        (-4, 53),
        xytext=(8, 64),
        fontsize=9,
        arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8),
    )
    ax.set_xlabel("longitude (°E)")
    ax.set_ylabel("latitude (°N)")
    ax.set_aspect(1.3)
    fig.colorbar(im, ax=ax, shrink=0.8, label="surface height (m)")
    fig.savefig(out / "domain.png", dpi=130)
    plt.close(fig)


def fields(out: Path) -> None:
    ds = xr.open_zarr(JAN).sel(time="2024-01-01T00:00")
    box = dict(latitude=slice(65, 42), longitude=slice(-25, 15))
    blh = ds["boundary_layer_height"].sel(**box).load()
    low = ds.isel(hybrid=-1)
    u = low["u_component_of_wind"].sel(**box).load()
    v = low["v_component_of_wind"].sel(**box).load()
    speed = np.hypot(u, v)
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.6), constrained_layout=True)
    im = axs[0].pcolormesh(
        blh.longitude, blh.latitude, blh, cmap=SEQ, vmin=0, vmax=2000, shading="auto"
    )
    fig.colorbar(im, ax=axs[0], shrink=0.85, label="boundary-layer depth (m)")
    axs[0].set_title("Boundary-layer depth, 1 Jan 2024 00:00 UTC")
    im = axs[1].pcolormesh(
        speed.longitude, speed.latitude, speed, cmap=SEQ, vmin=0, vmax=25, shading="auto"
    )
    s = (slice(None, None, 8), slice(None, None, 8))
    axs[1].quiver(
        u.longitude[s[1]],
        u.latitude[s[0]],
        u.values[s],
        v.values[s],
        color=INK2,
        scale=500,
        width=0.0025,
    )
    zs = ds["geopotential_at_surface"].sel(**box).load() / G
    for ax in axs:
        ax.contour(zs.longitude, zs.latitude, zs, levels=[2.0], colors=[INK2], linewidths=0.6)
    fig.colorbar(
        im, ax=axs[1], shrink=0.85, label="wind speed at the lowest model level, ~10 m (m s⁻¹)"
    )
    axs[1].set_title("Wind at the lowest model level, 1 Jan 2024 00:00 UTC")
    for ax in axs:
        mark_sites(ax)
        ax.set_aspect(1.6)
        ax.set_xlabel("longitude (°E)")
    axs[0].set_ylabel("latitude (°N)")
    fig.savefig(out / "fields.png", dpi=130)
    plt.close(fig)


def series(out: Path) -> None:
    ds = xr.concat([xr.open_zarr(DEC), xr.open_zarr(JAN)], dim="time").sel(
        time=slice("2023-12-27T00", "2024-01-02T00")
    )
    fig, axs = plt.subplots(2, 1, figsize=(10, 5.2), sharex=True, constrained_layout=True)
    for name, (lon, lat) in SITE_COORDS.items():
        pt = ds.sel(longitude=lon, latitude=lat, method="nearest")
        axs[0].plot(pt.time, pt["boundary_layer_height"].load(), color=SITES[name], label=name)
        low = pt.isel(hybrid=-1)
        axs[1].plot(
            pt.time,
            np.hypot(low["u_component_of_wind"].load(), low["v_component_of_wind"].load()),
            color=SITES[name],
            label=name,
        )
    axs[0].set_ylabel("BL depth (m)")
    axs[1].set_ylabel("wind ~10 m (m s⁻¹)")
    for ax in axs:
        ax.grid(axis="y")
        ax.axvspan(
            np.datetime64("2024-01-01T00"),
            np.datetime64("2024-01-02T00"),
            color="#e4e3df",
            alpha=0.6,
            lw=0,
        )
    axs[0].legend(ncol=3, loc="upper left")
    axs[0].text(
        np.datetime64("2024-01-01T01"),
        axs[0].get_ylim()[1] * 0.92,
        "releases",
        color=INK2,
        fontsize=9,
    )
    axs[0].set_title("At the three sites, over the 5-day backward window (nearest grid point)")
    fig.savefig(out / "series.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    domain(out)
    fields(out)
    series(out)
    print("done")
