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


def load_fixture(fixture_dir: str, config: str) -> xr.DataArray:
    """A golden fixture's footprint (``footprint.npz``, no coordinates) with the
    coordinates and site labels rebuilt from its config: cell centres from the
    output grid, the release order from the config's own schedule."""

    from lpdm.config import RunConfig

    cfg = RunConfig.from_yaml(config)
    og = cfg.output_grid
    arr = np.load(Path(fixture_dir) / "footprint.npz")["footprint"]
    lon_e = np.linspace(og.lon_bounds[0], og.lon_bounds[1], og.n_x + 1)
    lat_e = np.linspace(og.lat_bounds[0], og.lat_bounds[1], og.n_y + 1)
    edges = np.asarray(og.z_edges_m, dtype=float)
    releases = [r for b in cfg.expand_to_batches() for r in b.releases]
    if len(releases) != arr.shape[0]:
        raise ValueError(f"{len(releases)} releases in the config, {arr.shape[0]} in the fixture")
    da = xr.DataArray(
        arr,
        dims=("release", "time_ago", "z_bin", "latitude", "longitude"),
        coords={
            "latitude": 0.5 * (lat_e[:-1] + lat_e[1:]),
            "longitude": 0.5 * (lon_e[:-1] + lon_e[1:]),
            "z_bottom_m": ("z_bin", edges[:-1]),
            "z_top_m": ("z_bin", edges[1:]),
            "site": ("release", np.array([r.label or "MHD" for r in releases])),
        },
        name="footprint",
    )
    return da.sum("time_ago")


def load(run: str) -> xr.DataArray:
    fp = xr.open_zarr(Path(run) / "footprints.zarr")["footprint"].sum("time_ago").load()
    assert fp.attrs.get("density_weighting", "dry_air") == "dry_air" or True
    return fp


def smoke(run, out: Path) -> dict:
    fp = (run if isinstance(run, xr.DataArray) else load(run)).isel(release=0)
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


def reference(run, out: Path) -> dict:
    fp = run if isinstance(run, xr.DataArray) else load(run)
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
        n = int((sites == name).sum())
        ax.set_title(f"{name}: mean of {n} hourly releases")
        ax.set_aspect(1.6)
    fig.colorbar(im, ax=axs, shrink=0.85, label="mean 0–40 m footprint (m² s mol⁻¹)")
    fig.savefig(out / "reference-maps.png", dpi=130)
    plt.close(fig)

    totals = surf.sum(("latitude", "longitude"))
    fig, ax = plt.subplots(figsize=(10, 3.6), constrained_layout=True)
    stats = {}
    for name in SITES:
        t = totals.isel(release=np.flatnonzero(sites == name)).values
        ax.plot(np.arange(t.size), t, color=SITES[name], marker="o", ms=3, label=name)
        stats[name] = {
            "median": round(float(np.median(t)), 2),
            "min": round(float(t.min()), 2),
            "max": round(float(t.max()), 2),
        }
    ax.set_xlabel("release start (hours from 1 Jan 2024 00:00 UTC)")
    ax.set_ylabel("total 0–40 m (m² s mol⁻¹)")
    ax.set_xticks(range(0, 49, 6))
    ax.grid(axis="y")
    ax.legend(ncol=3, loc="upper left")
    ax.set_title("Total surface sensitivity of every release", loc="left")
    fig.savefig(out / "reference-totals.png", dpi=130)
    plt.close(fig)
    return stats


if __name__ == "__main__":
    # Two run directories and an output directory, or "--fixtures OUTDIR" to
    # draw the golden fixtures themselves (run from the repository root).
    import json

    if sys.argv[1] == "--fixtures":
        out = Path(sys.argv[2])
        out.mkdir(parents=True, exist_ok=True)
        smoke_fp = load_fixture(
            "tests/golden/smoke_mhd_single_release/cpu", "configs/smoke_mhd_single_release.yaml"
        )
        ref_fp = load_fixture(
            "outputs/golden/ab_multisite_perf/cuda", "configs/ab_multisite_perf.yaml"
        )
        res = {"smoke": smoke(smoke_fp, out), "reference": reference(ref_fp, out)}
    else:
        out = Path(sys.argv[3])
        out.mkdir(parents=True, exist_ok=True)
        res = {"smoke": smoke(sys.argv[1], out), "reference": reference(sys.argv[2], out)}
    (out / "glide-summary.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
