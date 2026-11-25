"""Static footprint explorer: one self-contained HTML page per run.

The exportable version of ``notebooks/footprint_explorer.ipynb`` for the paper's
supplementary material (docs/validation-plan.md §6). Choose a release (site and
time) and see GLIDE's time-integrated surface footprint, in STILT units on the
shared logarithmic scale, beside a reference footprint and their log10 ratio,
with the noise of neither hidden: hovering reads the values out.

The page needs no server, no network and no libraries: fields are quantised to
256 logarithmic levels, deflate-compressed and embedded as base64, and drawn on
canvases with the same colour scales as the static figures (``style.SEQ`` and
``style.DIV``).

    .venv/bin/python scripts/validation/figures/explorer.py GLIDE_RUN_DIR \
        [--reference REF_RUN_DIR] [--label-reference FLEXPART] [--out page.html]

Each run directory holds a ``footprints.zarr`` in GLIDE's layout (a reference
model's footprints converted to it, for example by the Tier 2 comparison
tools). The reference must be on the same grid and have the same releases.
"""

from __future__ import annotations

import argparse
import base64
import html
import json
import sys
import zlib
from pathlib import Path

import numpy as np
import xarray as xr

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "figures"

from . import core, style  # noqa: E402

DECADES = 5.0
LEVELS = 255  # 0 = no particles; 1..255 = log10 levels


def surface_footprint(run_dir: Path, air_density: float, surface_m: float = 40.0) -> xr.DataArray:
    """Time-integrated footprint of the lowest ``surface_m`` metres in STILT units,
    through GLIDE's conversion (``lpdm.comparison.to_stilt_surface_footprint``)."""
    from lpdm.comparison import to_stilt_surface_footprint

    fp = xr.open_zarr(Path(run_dir) / "footprints.zarr")["footprint"]
    attrs = dict(fp.attrs)
    if "density_weighting" not in attrs:
        print(f"{run_dir}: no density_weighting attribute (a store from before decision "
              f"0022); treating it as plain seconds with ρ = {air_density} kg m-3")  # fmt: skip
        attrs["density_weighting"] = "none"
    keep = np.flatnonzero(fp["z_top_m"].values <= surface_m + 1e-6)
    if keep.size == 0:
        raise SystemExit(f"{run_dir}: no z bin with its top at or below {surface_m} m")
    surf = fp.isel(z_bin=keep)
    depth = float(surf["z_top_m"].max())
    weighted = attrs.get("density_weighting", "none") != "none"
    fields = []
    for i in range(surf.sizes["release"]):
        one = surf.isel(release=i).load()
        one.attrs = attrs  # the conversion reads the declared weighting
        fields.append(
            to_stilt_surface_footprint(
                one,
                surface_layer_depth_m=depth,
                air_density_kg_m3=None if weighted else air_density,
                integrate_time=True,
            ).values
        )
    rel_coords = {k: v for k, v in fp.coords.items() if v.dims == ("release",)}
    return xr.DataArray(
        np.stack(fields),
        dims=("release", "latitude", "longitude"),
        coords={**rel_coords, "latitude": fp["latitude"], "longitude": fp["longitude"]},
        attrs={"units": core.FOOTPRINT_UNITS, "density_weighting": attrs.get("density_weighting")},
    )


def _encode(field: np.ndarray, vmax: float) -> str:
    lo = np.log10(vmax) - DECADES
    q = np.zeros(field.shape, dtype=np.uint8)
    pos = np.isfinite(field) & (field > 0)
    lv = (np.log10(np.where(pos, field, 1.0)) - lo) / DECADES
    q[pos] = np.clip(np.round(lv[pos] * (LEVELS - 1)) + 1, 1, LEVELS).astype(np.uint8)
    return base64.b64encode(zlib.compress(q.tobytes(), 9)).decode()


def _run_info(run_dir: Path) -> dict:
    meta_path = Path(run_dir) / "run_metadata.json"
    if not meta_path.exists():
        return {"config_hash": "unknown"}
    meta = json.loads(meta_path.read_text())
    cfg = json.loads(json.dumps(meta.get("config", {})))
    cfg.pop("io", None)
    cfg.get("simulation", {}).pop("device", None)
    return {"config_hash": core.config_hash(cfg)}


def build(
    glide_dir: Path,
    reference_dir: Path | None = None,
    *,
    label_reference: str = "reference",
    air_density: float = 1.2,
    max_releases: int = 200,
    title: str | None = None,
) -> str:
    g = surface_footprint(glide_dir, air_density)
    r = surface_footprint(reference_dir, air_density) if reference_dir else None
    if r is not None and (g.shape != r.shape or not np.allclose(g.latitude, r.latitude)
                          or not np.allclose(g.longitude, r.longitude)):  # fmt: skip
        raise SystemExit("reference footprint is not on GLIDE's grid / releases")
    if g["latitude"].values[0] > g["latitude"].values[-1]:  # north at the top of the canvas
        g = g.isel(latitude=slice(None, None, -1))
        r = r.isel(latitude=slice(None, None, -1)) if r is not None else None
    n_rel = min(g.sizes["release"], max_releases)
    lat, lon = g["latitude"].values, g["longitude"].values
    releases = []
    for i in range(n_rel):
        gi = g.isel(release=i).values
        ri = r.isel(release=i).values if r is not None else None
        vmax = float(np.nanmax(gi if ri is None else np.fmax(gi, ri)))
        vmax = vmax if vmax > 0 else 1.0
        rel = g.isel(release=i)
        site = str(rel["site"].values) if "site" in rel.coords else f"release {i}"
        when = (
            str(np.datetime_as_string(rel["release_time"].values, unit="m")).replace("T", " ")
            if "release_time" in rel.coords
            else ""
        )
        releases.append({
            "label": f"{site}, {when}",
            "lon": float(rel["release_lon"]) if "release_lon" in rel.coords else float("nan"),
            "lat": float(rel["release_lat"]) if "release_lat" in rel.coords else float("nan"),
            "vmax": vmax, "glide": _encode(gi, vmax),
            "total_glide": float(np.nansum(gi)),
            **({"ref": _encode(ri, vmax), "total_ref": float(np.nansum(ri))}
               if ri is not None else {}),
        })  # fmt: skip
    info = _run_info(glide_dir)
    prov = {
        "physics_tag": core.physics_tag(),
        "config_hash": info["config_hash"],
        "glide_run": Path(glide_dir).name,
        "reference_run": Path(reference_dir).name if reference_dir else None,
    }
    data = {
        "nx": int(lon.size), "ny": int(lat.size),
        "lon": [float(lon[0]), float(lon[-1])], "lat": [float(lat[0]), float(lat[-1])],
        "decades": DECADES, "levels": LEVELS, "units": core.FOOTPRINT_UNITS,
        "seq": _ramp(style.SEQ), "div": _ramp(style.DIV), "ratio_half_range": 1.0,
        "label_ref": label_reference, "releases": releases, "prov": prov,
        "truncated": g.sizes["release"] > n_rel,
    }  # fmt: skip
    page_title = title or f"GLIDE footprint explorer: {Path(glide_dir).name}"
    return (
        TEMPLATE.replace("__TITLE__", html.escape(page_title))
        .replace("__DATA__", json.dumps(data, separators=(",", ":")))
        .replace("__SURFACE__", style.SURFACE)
        .replace("__INK__", style.INK)
        .replace("__INK2__", style.INK2)
        .replace("__GRID__", style.GRID)
    )


def _ramp(cmap, n: int = 256) -> list[str]:
    rgba = cmap(np.linspace(0, 1, n))
    return ["#" + "".join(f"{int(round(c * 255)):02x}" for c in px[:3]) for px in rgba]


TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
:root{--surface:__SURFACE__;--ink:__INK__;--ink2:__INK2__;--grid:__GRID__;color-scheme:light}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--surface:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--grid:#383835;color-scheme:dark}}
:root[data-theme="dark"]{--surface:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--grid:#383835;color-scheme:dark}
body{margin:0;background:var(--surface);color:var(--ink);font:14px/1.45 system-ui,-apple-system,Segoe UI,sans-serif}
main{max-width:1400px;margin:0 auto;padding:16px}
h1{font-size:18px;margin:0 0 2px}.sub{color:var(--ink2);font-size:12px;margin-bottom:12px}
.controls{display:flex;gap:12px;flex-wrap:wrap;align-items:center;margin-bottom:12px}
select{font:inherit;padding:4px 6px;max-width:100%}
.panels{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:16px}
figure{margin:0}figcaption{font-weight:600;margin-bottom:4px}
canvas.map{width:100%;height:auto;image-rendering:pixelated;border:1px solid var(--grid);background:var(--surface)}
.bar{display:flex;align-items:center;gap:6px;font-size:11px;color:var(--ink2);margin-top:4px}
.bar canvas{flex:1;height:10px;min-width:60px}
#readout{font-variant-numeric:tabular-nums;color:var(--ink2);min-height:1.5em;margin-top:10px}
.note{color:var(--ink2);font-size:12px;margin-top:16px}
</style></head><body><main>
<h1>__TITLE__</h1>
<div class="sub" id="prov"></div>
<div class="controls"><label>Release <select id="rel"></select></label><span id="totals" class="sub" style="margin:0"></span></div>
<div class="panels" id="panels"></div>
<div id="readout">Hover a map to read values.</div>
<p class="note">Time-integrated footprint of the lowest model layer in STILT units (mole fraction per unit surface flux), on a logarithmic scale spanning five decades below each release's maximum; blank cells received no particles. The ratio is log<sub>10</sub>(GLIDE / reference), shown only where both are non-zero. Values are quantised to 256 logarithmic levels (a factor of 1.05).</p>
</main>
<script>
const D = __DATA__;
const $ = s => document.querySelector(s);
const hexToRgb = h => [1,3,5].map(i => parseInt(h.slice(i,i+2),16));
const SEQ = D.seq.map(hexToRgb), DIV = D.div.map(hexToRgb);
async function inflate(b64){
  const bytes = Uint8Array.from(atob(b64), c => c.charCodeAt(0));
  const ds = new DecompressionStream("deflate");
  const buf = await new Response(new Blob([bytes]).stream().pipeThrough(ds)).arrayBuffer();
  return new Uint8Array(buf);
}
const toVal = (q, vmax) => q === 0 ? 0 : Math.pow(10, Math.log10(vmax) - D.decades + (q-1)/(D.levels-1)*D.decades);
const fmt = v => v === 0 ? "0" : v.toExponential(2);
const panels = [];
function makePanel(title, kind){
  const f = document.createElement("figure");
  f.innerHTML = `<figcaption>${title}</figcaption>`;
  const c = document.createElement("canvas"); c.className = "map"; c.width = D.nx; c.height = D.ny;
  c.setAttribute("role","img"); c.setAttribute("aria-label", title);
  f.appendChild(c);
  const bar = document.createElement("div"); bar.className = "bar";
  const lo = document.createElement("span"), hi = document.createElement("span");
  const bc = document.createElement("canvas"); bc.width = 256; bc.height = 1;
  const bctx = bc.getContext("2d"), img = bctx.createImageData(256,1);
  const ramp = kind === "ratio" ? DIV : SEQ;
  for (let i=0;i<256;i++){ const [r,g,b] = ramp[i]; img.data.set([r,g,b,255], i*4); }
  bctx.putImageData(img,0,0);
  bar.append(lo, bc, hi); f.appendChild(bar);
  $("#panels").appendChild(f);
  const p = {canvas:c, kind, lo, hi, title, q:null};
  c.addEventListener("mousemove", e => readout(e, p));
  c.addEventListener("mouseleave", () => $("#readout").textContent = "Hover a map to read values.");
  panels.push(p); return p;
}
let cur = null;
function readout(e, p){
  if (!cur) return;
  const r = p.canvas.getBoundingClientRect();
  const i = Math.floor((e.clientX - r.left) / r.width * D.nx);
  const jc = Math.floor((e.clientY - r.top) / r.height * D.ny);
  if (i<0||i>=D.nx||jc<0||jc>=D.ny) return;
  const j = D.ny - 1 - jc, k = j*D.nx + i;
  const lon = D.lon[0] + (D.lon[1]-D.lon[0]) * i/(D.nx-1), lat = D.lat[0] + (D.lat[1]-D.lat[0]) * j/(D.ny-1);
  const g = toVal(cur.g[k], cur.vmax);
  let s = `${lon.toFixed(2)}°, ${lat.toFixed(2)}°   GLIDE ${fmt(g)} ${D.units}`;
  if (cur.r){ const v = toVal(cur.r[k], cur.vmax); s += `   ${D.label_ref} ${fmt(v)}`;
    if (g>0 && v>0) s += `   log10 ratio ${(Math.log10(g/v)).toFixed(2)}`; }
  $("#readout").textContent = s;
}
function paint(p, q, vmax, q2){
  const ctx = p.canvas.getContext("2d"), img = ctx.createImageData(D.nx, D.ny);
  const H = D.ratio_half_range;
  for (let j=0;j<D.ny;j++) for (let i=0;i<D.nx;i++){
    const k = j*D.nx + i, o = ((D.ny-1-j)*D.nx + i)*4;
    let rgb = null;
    if (p.kind === "ratio"){
      if (q[k] && q2[k]){ const lr = (q[k]-q2[k])/(D.levels-1)*D.decades;
        const t = Math.max(0, Math.min(255, Math.round((lr + H)/(2*H)*255))); rgb = DIV[t]; }
    } else if (q[k]) rgb = SEQ[Math.round((q[k]-1)/(D.levels-1)*255)];
    if (rgb) img.data.set([rgb[0],rgb[1],rgb[2],255], o);
  }
  ctx.putImageData(img,0,0);
  if (p.kind === "ratio"){ p.lo.textContent = `−${H}`; p.hi.textContent = `+${H} log10`; }
  else { p.lo.textContent = fmt(vmax*Math.pow(10,-D.decades)); p.hi.textContent = `${fmt(vmax)} ${D.units}`; }
}
async function show(idx){
  const R = D.releases[idx];
  const g = await inflate(R.glide), r = R.ref ? await inflate(R.ref) : null;
  cur = {g, r, vmax:R.vmax};
  paint(panels[0], g, R.vmax);
  if (r){ paint(panels[1], r, R.vmax); paint(panels[2], g, R.vmax, r); }
  let t = `total GLIDE ${fmt(R.total_glide)}`;
  if (R.ref) t += `, ${D.label_ref} ${fmt(R.total_ref)} (ratio ${(R.total_glide/R.total_ref).toFixed(3)})`;
  $("#totals").textContent = t + ` · release at ${R.lon.toFixed(3)}°, ${R.lat.toFixed(3)}°`;
}
(function init(){
  const P = D.prov;
  $("#prov").textContent = `physics ${P.physics_tag} · config ${P.config_hash} · GLIDE run ${P.glide_run}` + (P.reference_run ? ` · ${D.label_ref} run ${P.reference_run}` : "") + (D.truncated ? " · releases truncated" : "");
  makePanel("GLIDE", "seq");
  if (D.releases[0].ref){ makePanel(D.label_ref, "seq"); makePanel(`log10(GLIDE / ${D.label_ref})`, "ratio"); }
  const sel = $("#rel");
  D.releases.forEach((R,i) => { const o = document.createElement("option"); o.value = i; o.textContent = R.label; sel.appendChild(o); });
  sel.addEventListener("change", () => show(+sel.value));
  show(0);
})();
</script></body></html>
"""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("glide_run", type=Path)
    ap.add_argument("--reference", type=Path, default=None)
    ap.add_argument("--label-reference", default="reference")
    ap.add_argument("--air-density", type=float, default=1.2,
                    help="kg m-3, only for footprints without density weighting")  # fmt: skip
    ap.add_argument("--max-releases", type=int, default=200)
    ap.add_argument("--title", default=None)
    ap.add_argument("--out", type=Path, default=Path("footprint_explorer.html"))
    a = ap.parse_args(argv)
    page = build(a.glide_run, a.reference, label_reference=a.label_reference,
                 air_density=a.air_density, max_releases=a.max_releases, title=a.title)  # fmt: skip
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(page)
    print(f"wrote {a.out} ({a.out.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
