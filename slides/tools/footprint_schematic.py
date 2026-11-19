#!/usr/bin/env python
"""Draw a schematic of a backward Lagrangian particle model and its footprint.

Particles leave a measurement tower and are run backwards in time on a toy
wind with random turbulence, inside a boundary layer whose depth changes over
the day. Wherever they pass close to the ground, the time they spend there is
counted on the map below: that map is the footprint. The particle motion is
a cartoon, not GLIDE output; it only has to look right.

    .venv/bin/python slides/tools/footprint_schematic.py slides/img/footprint-concept.svg
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from matplotlib import colormaps
from matplotlib.colors import to_hex

W, H = 1000, 560

BG = "#06080d"
TEXT = "#e8eef6"
MUTED = "#9db0c7"
AMBER = "#ffb454"
TEAL = "#5fd0d6"
GROUND = "#111a28"
GRID = "#1d2a3d"
SANS = "Helvetica Neue, Segoe UI, system-ui, Arial, sans-serif"

# Ground plane, in model units: x east 0..10, y north 0..6, z height.
NX, NY = 10.0, 6.0
S = 70.0  # px per x unit
DXY = (40.0, -32.0)  # px per y unit (oblique depth)
SZ = 95.0  # px per z unit
X0, Y0 = 26.0, 392.0  # screen position of model (0, 0, 0)

TOWER = (8.9, 0.9)
INLET_Z = 0.55
SURFACE_Z = 0.28  # "touching the surface" for the footprint count

out: list[str] = []
add = out.append


def P(x, y, z=0.0):
    return X0 + x * S + y * DXY[0], Y0 + y * DXY[1] - z * SZ


def text(x, y, s, *, size=16, fill=TEXT, weight=400, anchor="start", extra=""):
    add(
        f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{fill}" '
        f'font-weight="{weight}" text-anchor="{anchor}" font-family="{SANS}" {extra}>{s}</text>'
    )


def poly(pts, fill, *, stroke="none", sw=1.0, opacity=1.0):
    p = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
    add(f'<polygon points="{p}" fill="{fill}" fill-opacity="{opacity:.3f}" '
        f'stroke="{stroke}" stroke-width="{sw}" stroke-linejoin="round"/>')


# --------------------------------------------------------------------------- toy model
rng = np.random.default_rng(7)
N, STEPS = 600, 72  # particles, hourly steps back over three days
x = np.full(N, TOWER[0]) + rng.normal(0, 0.02, N)
y = np.full(N, TOWER[1]) + rng.normal(0, 0.02, N)
z = np.full(N, INLET_Z)
lifted = np.zeros(N, bool)

track = np.empty((STEPS + 1, N, 3))
track[0] = np.c_[x, y, z]
for k in range(1, STEPS + 1):
    hour = (12 - k) % 24
    bl = 0.45 + 0.75 * max(0.0, np.sin(np.pi * (hour - 6) / 12))  # deep by day, shallow at night
    u = -0.105 - 0.02 * np.tanh((k - 30) / 10)  # backwards: air came from the west...
    v = 0.01 + 0.07 * (k / STEPS)  # ...and, further back, from the south-west
    x += u + rng.normal(0, 0.035 + 0.0012 * k, N)
    y += v + rng.normal(0, 0.035 + 0.0016 * k, N)
    # Some particles leave the boundary layer and stop touching the ground.
    lifted |= rng.random(N) < 0.006
    z += rng.normal(0, 0.16, N)
    z = np.abs(z)
    top = np.where(lifted, 1.7, bl)
    over = z > top
    z[over] = 2 * top[over] - z[over]
    z = np.where(lifted, np.maximum(z, bl + 0.15), z)
    track[k] = np.c_[x, y, z]

# Footprint: time spent close to the ground, counted on a 0.5-unit grid.
CELL = 0.5
nx, ny = int(NX / CELL), int(NY / CELL)
fp = np.zeros((ny, nx))
for k in range(1, STEPS + 1):
    px, py, pz = track[k].T
    near = (pz < SURFACE_Z) & (px >= 0) & (px < NX) & (py >= 0) & (py < NY)
    np.add.at(fp, ((py[near] / CELL).astype(int), (px[near] / CELL).astype(int)), 1)

# --------------------------------------------------------------------------- canvas
add(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}">')
add("<title>How a backward particle model makes a footprint</title>")
add("<defs>")
for mid, col in (("arr", MUTED), ("arrAmber", AMBER)):
    add(f'<marker id="{mid}" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" '
        f'markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="{col}"/></marker>')
add("</defs>")
add(f'<rect width="{W}" height="{H}" fill="{BG}"/>')

# Ground and footprint.
poly([P(0, 0), P(NX, 0), P(NX, NY), P(0, NY)], GROUND)
cmap = colormaps["magma"]
vmax = fp.max()
lo = vmax / 10**3
for j in range(ny):
    for i in range(nx):
        val = fp[j, i]
        if val <= 0:
            continue
        level = np.clip(np.log10(val / lo) / np.log10(vmax / lo), 0, 1)
        if level <= 0:
            continue
        x0, y0 = i * CELL, j * CELL
        poly([P(x0, y0), P(x0 + CELL, y0), P(x0 + CELL, y0 + CELL), P(x0, y0 + CELL)],
             to_hex(cmap(0.08 + 0.92 * level)), opacity=min(1.0, level / 0.6) ** 1.6 * 0.95)
for i in range(1, nx):
    add('<line x1="{:.1f}" y1="{:.1f}" x2="{:.1f}" y2="{:.1f}" stroke="{}" stroke-width="1" '
        'stroke-opacity="0.7"/>'.format(*P(i * CELL, 0), *P(i * CELL, NY), GRID))
for j in range(1, ny):
    add('<line x1="{:.1f}" y1="{:.1f}" x2="{:.1f}" y2="{:.1f}" stroke="{}" stroke-width="1" '
        'stroke-opacity="0.7"/>'.format(*P(0, j * CELL), *P(NX, j * CELL), GRID))
poly([P(0, 0), P(NX, 0), P(NX, NY), P(0, NY)], "none", stroke="#3a4a63", sw=1.5)

# A few particle tracks, then particle positions at a few times back.
kern = np.ones(5) / 5
for n in rng.choice(N, 10, replace=False):
    sm = np.stack([np.convolve(np.pad(track[:, n, c], 2, mode="edge"), kern, "valid") for c in range(3)], 1)
    pts = [P(*sm[k]) for k in range(STEPS + 1) if 0 <= sm[k, 0] < NX]
    d = "M" + " L".join(f"{a:.1f},{b:.1f}" for a, b in pts)
    add(f'<path d="{d}" fill="none" stroke="{TEAL}" stroke-width="1.2" stroke-opacity="0.45"/>')

snapshots = [(6, "6 hours ago"), (24, "1 day ago"), (48, "2 days ago"), (72, "3 days ago")]
for k, label in snapshots:
    sel = (track[k, :, 0] >= 0) & (track[k, :, 0] < NX)
    for px, py, pz in track[k, sel][::3]:
        sx, sy = P(px, py, pz)
        add(f'<circle cx="{sx:.1f}" cy="{sy:.1f}" r="2.2" fill="{TEXT}" fill-opacity="0.85"/>')
    cx = np.median(track[k, sel, 0])
    lx, ly = P(cx, 0)
    add(f'<line x1="{lx:.1f}" y1="{ly + 4:.1f}" x2="{lx:.1f}" y2="{ly + 12:.1f}" '
        f'stroke="{MUTED}" stroke-width="1.5"/>')
    text(lx, ly + 30, label, size=14, fill=MUTED, anchor="middle")

# Tower with its inlet.
bx, by = P(*TOWER)
ix, iy = P(*TOWER, INLET_Z + 0.08)
add(f'<line x1="{bx:.1f}" y1="{by:.1f}" x2="{ix:.1f}" y2="{iy - 6:.1f}" stroke="{TEXT}" '
    'stroke-width="3" stroke-linecap="round"/>')
add(f'<circle cx="{ix:.1f}" cy="{iy - 8:.1f}" r="5.5" fill="{AMBER}"/>')
text(ix + 14, iy - 30, "measurement", size=16, fill=AMBER, weight=700)

# Numbered notes.
def badge(cx, cy, n):
    add(f'<circle cx="{cx}" cy="{cy}" r="13" fill="none" stroke="{AMBER}" stroke-width="2"/>')
    text(cx, cy + 5, str(n), size=15, fill=AMBER, weight=700, anchor="middle")


notes = [
    ("Release particles at the", "measurement"),
    ("Run the winds and turbulence", "backwards in time"),
    ("Count the time particles spend", "near the ground: the footprint"),
]
NOTE_Y = 482
for i, (a, b) in enumerate(notes):
    nx0 = 40 + i * 320
    badge(nx0 + 13, NOTE_Y - 5, i + 1)
    text(nx0 + 36, NOTE_Y, a, size=17)
    text(nx0 + 36, NOTE_Y + 23, b, size=17)

# Direction of the run.
ax0, ay0 = P(8.6, 5.6, 1.6)
ax1, ay1 = P(5.6, 5.6, 1.6)
add(f'<line x1="{ax0:.1f}" y1="{ay0:.1f}" x2="{ax1:.1f}" y2="{ay1:.1f}" stroke="{MUTED}" '
    'stroke-width="1.8" marker-end="url(#arr)"/>')
text((ax0 + ax1) / 2, ay0 - 12, "backwards in time", size=15, fill=MUTED, anchor="middle")

add("</svg>")

dest = Path(sys.argv[1] if len(sys.argv) > 1 else "footprint-concept.svg")
dest.write_text("\n".join(out) + "\n")
print(f"wrote {dest}")
