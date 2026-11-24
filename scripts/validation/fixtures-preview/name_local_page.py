"""NAME reference runs, described on their own: a LOCAL page, never published.

NAME output is Met Office-derived and restricted (each file's `licence`
attribute); this page stays on the machine that holds the data. It shows the
reference data only, never beside GLIDE output (pre-registration: no GLIDE vs
reference comparison on graded cases before registration).

    .venv/bin/python scripts/validation/fixtures-preview/name_local_page.py OUTDIR
"""

import base64
import glob
import io
import sys
from pathlib import Path

import numpy as np
import xarray as xr
from matplotlib.colors import LogNorm

sys.path.insert(0, str(Path(__file__).parent))
from style import INK2, SEQ, plt  # noqa: E402

NAME = Path("/projects/u6za/glide-data/name-footprints")
CASES = [
    (1, "Coastal", "MHD", "EUR_201901"),
    (2, "Mountain", "JFJ", "EUR_201907"),
    (3, "Stable night", "CBW-20", "EUR_201901"),
    (5, "Frontal passage", "TAC", "EUR_201901"),
    (4, "Convective afternoon", "LEF", "NAM_201907"),
    (6, "Continental winter", "LEF", "NAM_201901"),
    (7, "Column", "SGP", "NAM_201907"),
]


def png(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def cases_fig() -> str:
    fig, axs = plt.subplots(2, 4, figsize=(15, 6.6), constrained_layout=True)
    norm = LogNorm(1e-3, 10.0)
    for ax, (n, label, site, block) in zip(axs.flat, CASES, strict=False):
        ds = xr.open_dataset(NAME / f"{site}_NAME_UMG-Mk10_{block}_case{n}.nc")
        f = ds["srr"].isel(time=0)
        im = ax.pcolormesh(
            f.longitude, f.latitude, f.where(f > 0), cmap=SEQ, norm=norm, shading="auto"
        )
        ax.plot(
            float(ds.release_lon[0]),
            float(ds.release_lat[0]),
            "o",
            ms=7,
            mfc="#eb6834",
            mec="white",
            mew=1.5,
        )
        ax.set_title(
            f"Case {n}: {label}, {site}\n{str(ds.time.values[0])[:16].replace('T', ' ')} UTC",
            fontsize=9.5,
        )
        ax.set_aspect(1.4)
        ax.tick_params(labelsize=8)
    axs.flat[-1].axis("off")
    fig.colorbar(im, ax=axs, shrink=0.7, label="0–40 m footprint (m² s mol⁻¹, NAME, moist air)")
    return png(fig)


def totals_fig() -> str:
    files = sorted(glob.glob(str(NAME / "*_main.nc")))
    blocks = sorted({Path(f).name.split("_")[3] + "_" + Path(f).name.split("_")[4] for f in files})
    fig, axs = plt.subplots(
        len(blocks), 1, figsize=(12, 2.4 * len(blocks)), constrained_layout=True
    )
    hues = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
    for ax, block in zip(np.atleast_1d(axs), blocks, strict=True):
        for i, f in enumerate(sorted(x for x in files if block in x)):
            ds = xr.open_dataset(f)
            tot = ds["srr"].sum(("latitude", "longitude"))
            ax.plot(
                ds.time,
                tot,
                color=hues[i % len(hues)],
                lw=1.6,
                label=ds.attrs.get("receptor", Path(f).name.split("_")[0]),
            )
        ax.set_title(block.replace("_", " "), loc="left")
        ax.set_ylabel("total (m² s mol⁻¹)")
        ax.grid(axis="y")
        ax.legend(ncol=5, fontsize=8, loc="upper right")
    return png(fig)


def seeds_fig() -> str:
    fig, ax = plt.subplots(figsize=(8, 3.6), constrained_layout=True)
    for i, site in enumerate(["MHD", "JFJ", "CBW-20"]):
        tots = np.array(
            [
                xr.open_dataset(NAME / f"{site}_NAME_UMG-Mk10_EUR_201901_seed{k}.nc")["srr"]
                .sum(("latitude", "longitude"))
                .values
                for k in range(1, 6)
            ]
        )
        cv = 100 * tots.std(axis=0, ddof=1) / tots.mean(axis=0)
        ax.hist(
            cv,
            bins=np.linspace(0, 10, 41),
            histtype="step",
            lw=2,
            color=["#2a78d6", "#eb6834", "#1baf7a"][i],
            label=f"{site} (median {np.median(cv):.1f}%)",
        )
    ax.set_xlabel(
        "spread of the total footprint across the five seed runs (coefficient of variation, %)"
    )
    ax.set_ylabel("releases")
    ax.legend()
    ax.set_title("Particle noise in NAME's totals: 120 releases, 1–5 Jan 2019", loc="left")
    return png(fig)


def run_matrix() -> str:
    rows = {}
    for f in sorted(NAME.glob("*.nc")):
        parts = f.stem.split("_")
        site, block, run = parts[0], f"{parts[3]} {parts[4]}", "_".join(parts[5:])
        if run.endswith("_end"):
            continue
        rows.setdefault((site, block), []).append(run)
    out = [
        "<table><thead><tr><th>Receptor</th><th>Block</th><th>Runs on disk</th></tr></thead><tbody>"
    ]
    for (site, block), runs in sorted(rows.items()):
        out.append(f"<tr><td>{site}</td><td>{block}</td><td>{', '.join(sorted(runs))}</td></tr>")
    out.append("</tbody></table>")
    return "".join(out)


def main(outdir: str) -> None:
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    html = f"""<!doctype html><meta charset="utf-8"><title>NAME reference runs (local)</title>
<style>
body{{font:15px/1.55 system-ui,sans-serif;max-width:1180px;margin:0 auto;padding:24px 16px;color:#1d1c1a;background:#fcfcfb}}
h1{{font-size:1.6rem;margin:.2em 0}} h2{{font-size:1.15rem;margin-top:2em}}
.warn{{border:1px solid #d95926;background:#fdf1eb;padding:10px 14px;border-radius:6px}}
img{{max-width:100%;border:1px solid #e4e3df;border-radius:4px}} table{{border-collapse:collapse;font-size:.9rem}}
td,th{{border-bottom:1px solid #e4e3df;padding:4px 10px;text-align:left}} p{{max-width:72ch}}
</style>
<h1>NAME reference runs</h1>
<p class="warn"><b>Local page, restricted data.</b> NAME output is Met Office-derived (licence in each file). Keep this
file on Isambard; do not upload, publish or commit it. It describes the reference data only and deliberately shows no
GLIDE output (pre-registration).</p>
<p>NAME 8.7.1 (ACRG-patched fate records) on Met Office UM global analyses (UMG_Mk10PT, 3-hourly, 59 levels), run as
specified in GLIDE <code>docs/reference-runs.md</code>. Footprints are the 0–40 m layer in m² s mol⁻¹ relative to moist
air (decision 0022 gives the dry-air correction).</p>
<h2>The seven standard cases</h2>
<img src="data:image/png;base64,{cases_fig()}" alt="NAME footprints for the seven standard cases">
<h2>Total footprint of every release, by block</h2>
<img src="data:image/png;base64,{totals_fig()}" alt="Total footprint per release, per receptor and block">
<h2>Particle noise from the repeat-seed runs</h2>
<img src="data:image/png;base64,{seeds_fig()}" alt="Spread of totals across five seed runs">
<p>The July LEF and SGP repeat-seed sets (added 23 Nov 2026) are not on disk yet.</p>
<h2>Run matrix on disk</h2>
{run_matrix()}
<p style="color:{INK2}">Generated by <code>scripts/validation/fixtures-preview/name_local_page.py</code>.</p>
"""
    (out / "name-reference-runs.html").write_text(html)
    print(out / "name-reference-runs.html")


if __name__ == "__main__":
    main(sys.argv[1])
