#!/usr/bin/env python
"""Draw a schematic of a meteorology Zarr store as SVG.

Used in the lightning talk. The full version, with the pipeline strip, is
met-bazarr's README feature image (drawn there by scripts/draw_zarr_schematic.py,
a copy of this script). The layout follows Zarr v3 (the format zarr-python 3
writes) and the chunking GLIDE's docs/met_schema.md recommends: one hour per
chunk, 128 x 128 horizontal tiles, levels split around 24, int16 + zstd.

    .venv/bin/python slides/tools/zarr_store_schematic.py --no-pipeline slides/img/zarr-store-slide.svg

`--no-pipeline` drops the source -> met-bazarr -> GLIDE strip along the bottom,
which leaves a wider image that fits under a slide title.
"""

from __future__ import annotations

import sys
from pathlib import Path

ARGS = [a for a in sys.argv[1:] if not a.startswith("--")]
PIPELINE = "--no-pipeline" not in sys.argv
W, H = 1500, (840 if PIPELINE else 710)

BG = "#06080d"
TEXT = "#e8eef6"
MUTED = "#9db0c7"
FAINT = "#5d6f88"
AMBER = "#ffb454"
TEAL = "#5fd0d6"
FRONT = "#16263d"
TOP = "#1f3554"
SIDE = "#112038"
TILE = "#33507a"
SANS = "Helvetica Neue, Segoe UI, system-ui, Arial, sans-serif"
MONO = "SF Mono, Menlo, Consolas, DejaVu Sans Mono, monospace"

out: list[str] = []


def add(s: str) -> None:
    out.append(s)


def text(x, y, s, *, size=16, fill=TEXT, weight=400, anchor="start", family=SANS, extra=""):
    add(
        f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{fill}" '
        f'font-weight="{weight}" text-anchor="{anchor}" font-family="{family}" {extra}>{s}</text>'
    )


def lines(x, y, rows, *, size=16, fill=TEXT, lh=1.4, **kw):
    for i, r in enumerate(rows):
        text(x, y + i * size * lh, r, size=size, fill=fill, **kw)


def poly(pts, *, fill="none", stroke="none", sw=1.0, extra=""):
    p = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
    add(f'<polygon points="{p}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}" '
        f'stroke-linejoin="round" {extra}/>')


def line(x1, y1, x2, y2, *, stroke=TILE, sw=1.0, extra=""):
    add(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
        f'stroke="{stroke}" stroke-width="{sw}" {extra}/>')


def block(x0, y0, w, h, dx, dy, *, n_lon, n_lat, n_lev, fills=(FRONT, TOP, SIDE),
          outline=MUTED, grid=TILE, opacity=1.0, grid_sw=1.2):
    """Oblique box. (x0, y0) is the front-bottom-left corner; (dx, dy) the depth vector."""
    g = f'<g opacity="{opacity}">'
    add(g)
    front = [(x0, y0), (x0 + w, y0), (x0 + w, y0 - h), (x0, y0 - h)]
    top = [(x0, y0 - h), (x0 + w, y0 - h), (x0 + w + dx, y0 - h + dy), (x0 + dx, y0 - h + dy)]
    side = [(x0 + w, y0), (x0 + w + dx, y0 + dy), (x0 + w + dx, y0 - h + dy), (x0 + w, y0 - h)]
    poly(front, fill=fills[0])
    poly(top, fill=fills[1])
    poly(side, fill=fills[2])
    if grid:
        for i in range(1, n_lon):
            x = x0 + i * w / n_lon
            line(x, y0, x, y0 - h, stroke=grid, sw=grid_sw)
            line(x, y0 - h, x + dx, y0 - h + dy, stroke=grid, sw=grid_sw)
        for k in range(1, n_lev):
            y = y0 - k * h / n_lev
            line(x0, y, x0 + w, y, stroke=grid, sw=grid_sw)
            line(x0 + w, y, x0 + w + dx, y + dy, stroke=grid, sw=grid_sw)
        for j in range(1, n_lat):
            ox, oy = j * dx / n_lat, j * dy / n_lat
            line(x0 + ox, y0 - h + oy, x0 + w + ox, y0 - h + oy, stroke=grid, sw=grid_sw)
            line(x0 + w + ox, y0 + oy, x0 + w + ox, y0 - h + oy, stroke=grid, sw=grid_sw)
    for face in (front, top, side):
        poly(face, stroke=outline, sw=1.6)
    add("</g>")


def badge(x, y, n):
    add(f'<circle cx="{x}" cy="{y}" r="15" fill="none" stroke="{AMBER}" stroke-width="2"/>')
    text(x, y + 6, str(n), size=17, fill=AMBER, weight=700, anchor="middle")


def arrow(x1, y1, x2, y2, *, stroke=MUTED, sw=1.6, dash="", marker="arr"):
    d = f'stroke-dasharray="{dash}"' if dash else ""
    add(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{stroke}" '
        f'stroke-width="{sw}" {d} marker-end="url(#{marker})"/>')


# --------------------------------------------------------------------------- canvas
add(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}">')
add("<title>A meteorology Zarr store</title>")
add("<defs>")
for mid, col in (("arr", MUTED), ("arrAmber", AMBER)):
    add(f'<marker id="{mid}" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" '
        f'markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="{col}"/></marker>')
add("</defs>")
add(f'<rect width="{W}" height="{H}" rx="18" fill="{BG}"/>')

# --------------------------------------------------------------------------- panel 1: folder
PX1 = 50
badge(PX1 + 15, 62, 1)
text(PX1 + 42, 69, "It is a folder", size=24, weight=700)
lines(PX1, 104, ["Small JSON files describe the data;", "the data sit in many chunk files."],
      size=16, fill=MUTED)

tree = [
    ("era5_europe.zarr/", TEXT, 700, ""),
    ("├─ zarr.json", TEAL, 400, "dimensions, units"),
    ("├─ temperature/", TEXT, 400, ""),
    ("│  ├─ zarr.json", TEAL, 400, "shape, chunks, codec"),
    ("│  └─ c/", TEXT, 400, ""),
    ("│     ├─ 0/0/0/0", MUTED, 400, ""),
    ("│     ├─ 0/0/0/1", MUTED, 400, ""),
    ("│     ├─ 0/1/0/2", AMBER, 700, "one chunk"),
    ("│     └─ …", MUTED, 400, ""),
    ("├─ u_component_of_wind/", TEXT, 400, ""),
    ("├─ v_component_of_wind/", TEXT, 400, ""),
    ("├─ geopotential/", TEXT, 400, ""),
    ("├─ …", MUTED, 400, ""),
    ("└─ surface_pressure/", TEXT, 400, ""),
]
TY0, TLH = 178, 29
for i, (s, col, wt, note) in enumerate(tree):
    y = TY0 + i * TLH
    text(PX1, y, s, size=17, fill=col, weight=wt, family=MONO, extra='xml:space="preserve"')
    if note:
        text(PX1 + 222, y, "← " + note, size=14, fill=col if col != TEXT else MUTED, family=SANS)
CHUNK_FILE_Y = TY0 + 7 * TLH

# --------------------------------------------------------------------------- panel 2: array
PX2 = 520
badge(PX2 + 15, 62, 2)
text(PX2 + 42, 69, "Each variable is a 4-D array, cut into chunks", size=24, weight=700)
lines(PX2, 104, ["temperature(time, level, latitude, longitude)"], size=16, fill=MUTED,
      family=MONO)

BW, BH = 330, 210          # front face: longitude x level
DX, DY = 105, -78          # depth: latitude
NLON, NLAT, NLEV = 4, 3, 2
BX, BY = 560, 600          # front-bottom-left corner of the 00:00 block
TDX, TDY = 46, -40         # offset per hour

for t in (2, 1):
    block(BX + t * TDX, BY + t * TDY, BW, BH, DX, DY, n_lon=NLON, n_lat=NLAT, n_lev=NLEV,
          outline=FAINT, grid=None, opacity=0.9 if t == 1 else 0.75,
          fills=("#0f1a2b", "#14223a", "#0c1626"))
block(BX, BY, BW, BH, DX, DY, n_lon=NLON, n_lat=NLAT, n_lev=NLEV)

# Highlighted chunk: hour 0, upper level chunk (1), front latitude row (0), longitude tile 2.
cw, ch = BW / NLON, BH / NLEV
cdx, cdy = DX / NLAT, DY / NLAT
hx = BX + 2 * cw
hy = BY - BH
poly([(hx, hy), (hx + cw, hy), (hx + cw, hy + ch), (hx, hy + ch)],
     fill="rgba(255,180,84,0.30)", stroke=AMBER, sw=2.4)
poly([(hx, hy), (hx + cw, hy), (hx + cw + cdx, hy + cdy), (hx + cdx, hy + cdy)],
     fill="rgba(255,180,84,0.42)", stroke=AMBER, sw=2.4)

# Axis labels.
text(BX + BW / 2, BY + 32, "longitude", size=16, fill=MUTED, anchor="middle")
add(f'<text x="{BX - 18}" y="{BY - BH / 2}" font-size="16" fill="{MUTED}" font-family="{SANS}" '
    f'text-anchor="middle" transform="rotate(-90 {BX - 18} {BY - BH / 2})">level</text>')
lx, ly = BX + BW + DX / 2 + 14, BY + DY / 2 + 14
add(f'<text x="{lx:.1f}" y="{ly:.1f}" font-size="16" fill="{MUTED}" font-family="{SANS}" '
    f'transform="rotate(-36.6 {lx:.1f} {ly:.1f})">latitude</text>')
tx0, ty0 = BX + DX - 6, BY - BH + DY - 14
arrow(tx0, ty0, tx0 + 2 * TDX + 4, ty0 + 2 * TDY + 4, stroke=MUTED)
text(tx0 + 2 * TDX + 14, ty0 + 2 * TDY + 2, "time: one block per hour", size=15, fill=MUTED)
text(BX, BY + 70, "Chunk: 1 hour × 24 levels × 128 × 128 grid points",
     size=16, fill=TEXT)
text(BX, BY + 96, "Its file name is its position in the grid of chunks: c/time/level/lat/lon",
     size=15, fill=MUTED)

# --------------------------------------------------------------------------- panel 3: chunk
PX3 = 1140
badge(PX3 + 15, 62, 3)
text(PX3 + 42, 69, "Chunks are compressed", size=24, weight=700)
text(PX3 + 42, 97, "one by one", size=24, weight=700)

CX, CY, CW, CH = PX3 + 30, 380, 170, 170
CDX, CDY = 72, -54
block(CX, CY, CW, CH, CDX, CDY, n_lon=8, n_lat=4, n_lev=6,
      fills=("rgba(255,180,84,0.22)", "rgba(255,180,84,0.34)", "rgba(255,180,84,0.14)"),
      outline=AMBER, grid="rgba(255,180,84,0.35)", grid_sw=0.8)
text(CX + CW / 2, CY + 30, "temperature/c/0/1/0/2", size=15, fill=AMBER, anchor="middle",
     family=MONO, weight=700)

# Leader from the highlighted chunk to its enlargement, and to its file in the tree.
arrow(hx + cw + cdx + 6, hy + cdy + 4, CX - 10, CY - CH + 10, stroke=AMBER, dash="5 5",
      marker="arrAmber")
arrow(hx - 6, hy + ch / 2, PX1 + 345, CHUNK_FILE_Y - 6, stroke=AMBER, dash="5 5",
      marker="arrAmber")

sy = CY + 78
lines(PX3, sy, ["float32 → int16 → zstd"], size=17, fill=TEXT, weight=700, family=MONO)
lines(PX3, sy + 32, [
    "About half the size of compressed",
    "float32. The rounding errors are far",
    "below what the physics resolves.",
], size=15, fill=MUTED)
lines(PX3, sy + 124, [
    "A reader works out from the JSON",
    "which chunks hold the hours, levels",
    "and region it wants, and opens",
    "only those: on a laptop disk or in",
    "cloud object storage.",
], size=15, fill=TEXT)

# --------------------------------------------------------------------------- pipeline strip
SY = 774
if PIPELINE:
    line(50, SY - 52, W - 50, SY - 52, stroke="#1c2738", sw=1.2)
    boxes = [
        ("ERA5, Met Office UM, …", "GRIB, NetCDF, PP files", MUTED, "6 4"),
        ("met-bazarr", "subset · fix units and signs · compress", AMBER, ""),
        ("Zarr store", "like the one above", TEAL, ""),
        ("GLIDE", "reads one hour, one region at a time", MUTED, ""),
    ]
    bw, gap = 300, 52
    x = (W - (4 * bw + 3 * gap)) / 2
    for i, (title, sub, col, dash) in enumerate(boxes):
        d = f'stroke-dasharray="{dash}"' if dash else ""
        add(f'<rect x="{x:.1f}" y="{SY - 30}" width="{bw}" height="60" rx="8" fill="#0d1522" '
            f'stroke="{col}" stroke-width="1.6" {d}/>')
        text(x + bw / 2, SY - 6, title, size=17, fill=TEXT, weight=700, anchor="middle")
        text(x + bw / 2, SY + 16, sub, size=13, fill=MUTED, anchor="middle")
        if i < 3:
            arrow(x + bw + 6, SY, x + bw + gap - 6, SY)
        x += bw + gap

add("</svg>")

dest = Path(ARGS[0] if ARGS else "zarr-store.svg")
dest.write_text("\n".join(out) + "\n")
print(f"wrote {dest}")
