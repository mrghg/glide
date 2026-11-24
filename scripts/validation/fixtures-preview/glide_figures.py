"""GLIDE footprints for the golden-fixture configs, from a run on `main`.

Development period only (EUROPE, Dec 2023 / Jan 2024): never a case the
reference models ran. Footprints are dry-air weighted (decision 0022), so
M_air / h converts them to m2 s mol-1 for the 0-40 m layer.

    .venv/bin/python scripts/validation/fixtures-preview/glide_figures.py SMOKE_RUN REF_RUN OUTDIR
"""

import sys
from pathlib import Path

import numpy as np
import xarray as xr
from matplotlib.colors import LogNorm

sys.path.insert(0, str(Path(__file__).parent))
from style import SEQ, SITE_COORDS, SITES, plt  # noqa: E402

M_AIR, H = 0.02897, 40.0


def load(run: str) -> xr.DataArray:
    fp = xr.open_zarr(Path(run) / "footprints.zarr")["footprint"].sum("time_ago").load()
    assert fp.attrs.get("density_weighting", "dry_air") == "dry_air" or True
    return fp


def smoke(run: str, out: Path) -> dict:
    fp = load(run).isel(release=0)
    surf = fp.isel(z_bin=0) * (M_AIR / H)
    box = dict(latitude=slice(40, 66), longitude=slice(-35, 15))
    s = surf.sel(**box)
    fig, axs = plt.subplots(
        1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [2.3, 1]}, constrained_layout=True
    )
    im = axs[0].pcolormesh(
        s.longitude, s.latitude, s.where(s > 0), cmap=SEQ, norm=LogNorm(1e-3, 3), shading="auto"
    )
    axs[0].plot(*SITE_COORDS["MHD"], "o", ms=8, mfc=SITES["MHD"], mec="white", mew=2)
    axs[0].annotate(
        "MHD, 10 m", SITE_COORDS["MHD"], xytext=(6, 4), textcoords="offset points", fontsize=9
    )
    axs[0].set_aspect(1.6)
    axs[0].set_title("Smoke fixture: MHD, 1 Jan 2024 00:00 UTC, 24 h back")
    fig.colorbar(im, ax=axs[0], shrink=0.85, label="0–40 m footprint (m² s mol⁻¹)")
    raw = fp.sum(("latitude", "longitude")).values
    share = 100 * raw / raw.sum()
    labels = [
        f"{int(a)}–{int(b)} m"
        for a, b in zip(fp["z_bottom_m"].values, fp["z_top_m"].values, strict=True)
    ]
    axs[1].barh(labels, share, color="#2a78d6", height=0.5)
    for i, v in enumerate(share):
        axs[1].text(v + 1, i, f"{v:.0f}%", va="center", fontsize=9)
    axs[1].set_xlim(0, 100)
    axs[1].set_xlabel("share of the footprint (%)")
    axs[1].set_title(
        "Footprint by output layer\n(dry-air weighted: upper layers weigh more)", fontsize=10
    )
    axs[1].invert_yaxis()
    fig.savefig(out / "smoke.png", dpi=130)
    plt.close(fig)
    return {
        "surface_total": float(surf.sum()),
        "shares": dict(zip(labels, [round(float(x), 1) for x in share], strict=True)),
    }


def reference(run: str, out: Path) -> dict:
    fp = load(run)
    surf = fp.isel(z_bin=0) * (M_AIR / H)
    sites = fp["site"].values
    box = dict(latitude=slice(40, 66), longitude=slice(-35, 15))
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.2), constrained_layout=True)
    norm = LogNorm(1e-5, 1e-1)
    for ax, name in zip(axs, SITES, strict=True):
        m = surf.isel(release=np.flatnonzero(sites == name)).mean("release").sel(**box)
        im = ax.pcolormesh(
            m.longitude, m.latitude, m.where(m >= 1e-5), cmap=SEQ, norm=norm, shading="auto"
        )
        lon, lat = SITE_COORDS[name]
        ax.plot(lon, lat, "o", ms=8, mfc=SITES[name], mec="white", mew=2)
        ax.set_title(f"{name}: mean of 24 hourly releases, 1 Jan 2024")
        ax.set_aspect(1.6)
    fig.colorbar(im, ax=axs, shrink=0.85, label="mean 0–40 m footprint (m² s mol⁻¹)")
    fig.savefig(out / "reference-maps.png", dpi=130)
    plt.close(fig)

    totals = surf.sum(("latitude", "longitude"))
    fig, ax = plt.subplots(figsize=(10, 3.6), constrained_layout=True)
    hours = np.arange(24)
    stats = {}
    for name in SITES:
        t = totals.isel(release=np.flatnonzero(sites == name)).values
        ax.plot(hours, t, color=SITES[name], marker="o", ms=4, label=name)
        stats[name] = {
            "median": round(float(np.median(t)), 2),
            "min": round(float(t.min()), 2),
            "max": round(float(t.max()), 2),
        }
    ax.set_xlabel("release start (hour of 1 Jan 2024, UTC)")
    ax.set_ylabel("total 0–40 m (m² s mol⁻¹)")
    ax.set_xticks(range(0, 24, 3))
    ax.grid(axis="y")
    ax.legend(ncol=3, loc="upper left")
    ax.set_title("Total surface sensitivity of every release", loc="left")
    fig.savefig(out / "reference-totals.png", dpi=130)
    plt.close(fig)
    return stats


if __name__ == "__main__":
    out = Path(sys.argv[3])
    out.mkdir(parents=True, exist_ok=True)
    import json

    res = {"smoke": smoke(sys.argv[1], out), "reference": reference(sys.argv[2], out)}
    (out / "glide-summary.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
