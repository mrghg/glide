#!/usr/bin/env python
"""Draw a schematic of GLIDE's GPU step: particles as tensor rows, and the CUDA graph.

Left: the particle state is a tensor, one row per particle; each operation acts
on every row at once, and finished particles stay in with a zero timestep so
every step has the same shape. Right: one step before and after recording it as
a CUDA graph, on a common millisecond axis. The step times, instruction counts
and GPU work are the GH200 measurements in docs/architecture.md §5;
the individual marks are schematic.

    .venv/bin/python slides/tools/gpu_step_schematic.py slides/img/gpu-step.svg
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

W, H = 1500, 690

BG = "#06080d"
TEXT = "#e8eef6"
MUTED = "#9db0c7"
FAINT = "#4c5d75"
AMBER = "#ffb454"
TEAL = "#5fd0d6"
PANEL = "#0d1522"
CELL = "#16263d"
SANS = "Helvetica Neue, Segoe UI, system-ui, Arial, sans-serif"
MONO = "SF Mono, Menlo, Consolas, DejaVu Sans Mono, monospace"

out: list[str] = []
add = out.append


def text(x, y, s, *, size=16, fill=TEXT, weight=400, anchor="start", family=SANS, extra=""):
    add(
        f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{fill}" '
        f'font-weight="{weight}" text-anchor="{anchor}" font-family="{family}" {extra}>{s}</text>'
    )


def rect(x, y, w, h, fill, *, stroke="none", sw=1.0, rx=0.0, opacity=1.0, extra=""):
    add(
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.2f}" height="{h:.1f}" rx="{rx}" '
        f'fill="{fill}" fill-opacity="{opacity}" stroke="{stroke}" stroke-width="{sw}" {extra}/>'
    )


def badge(cx, cy, n):
    add(f'<circle cx="{cx}" cy="{cy}" r="15" fill="none" stroke="{AMBER}" stroke-width="2"/>')
    text(cx, cy + 6, str(n), size=17, fill=AMBER, weight=700, anchor="middle")


add(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}">')
add("<title>How GLIDE uses the GPU</title>")
add("<defs>")
for mid, col in (("arr", MUTED), ("arrAmber", AMBER), ("arrTeal", TEAL)):
    add(f'<marker id="{mid}" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" '
        f'markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="{col}"/></marker>')
add("</defs>")
add(f'<rect width="{W}" height="{H}" fill="{BG}"/>')

# --------------------------------------------------------------------------- panel 1: tensor
PX = 40
badge(PX + 15, 40, 1)
text(PX + 42, 47, "Every particle steps at once", size=24, weight=700)
text(PX, 84, "The particles are one tensor: a row each, up to ~500,000 rows.", size=16,
     fill=MUTED)

cols = ["x", "y", "z", "u′", "v′", "w′", "Δt"]
CW, RH = 56, 27
TX, TY = PX + 34, 150
rows = 13
finished = {4, 9}  # rows drawn as finished particles
for j, c in enumerate(cols):
    text(TX + j * CW + CW / 2, TY - 12, c, size=17, fill=MUTED, anchor="middle", family=MONO)
rng = np.random.default_rng(3)
for i in range(rows):
    y = TY + i * RH
    done = i in finished
    text(TX - 12, y + 19, "⋮" if i == rows - 1 else str(i), size=13, fill=FAINT, anchor="end",
         family=MONO)
    for j, c in enumerate(cols):
        x = TX + j * CW
        if i == rows - 1:
            text(x + CW / 2, y + 19, "⋮", size=16, fill=FAINT, anchor="middle")
            continue
        fill = CELL if not done else "#0f1622"
        rect(x + 1, y + 1, CW - 2, RH - 2, fill, rx=3)
        if c == "Δt":
            val = "0" if done else "τ"
            col = AMBER if done else MUTED
            text(x + CW / 2, y + 19, val, size=14, fill=col, anchor="middle", family=MONO,
                 weight=700 if done else 400)
        else:
            shade = 0.25 + 0.5 * rng.random()
            rect(x + 10, y + 11, (CW - 20) * shade, 5, TEAL if not done else FAINT, rx=2,
                 opacity=0.75)

# The operation sweeping down the z column, all rows at once.
zx = TX + 2 * CW
rect(zx - 3, TY - 2, CW + 6, (rows - 1) * RH + 4, "none", stroke=TEAL, sw=2.2, rx=5)
oy = TY + rows * RH + 22
add(f'<line x1="{zx + CW / 2}" y1="{oy + 6}" x2="{zx + CW / 2}" y2="{TY + (rows - 1) * RH + 8}" '
    f'stroke="{TEAL}" stroke-width="2" marker-end="url(#arrTeal)"/>')
text(zx - 30, oy + 30, "z ← z + w′ Δt", size=18, fill=TEAL, family=MONO, weight=700)
text(zx + 150, oy + 30, "one operation, every row at once", size=15, fill=MUTED)

ny = oy + 74
lines = [
    ("Finished particles stay in with ", "Δt = 0", ","),
    ("so every step has exactly the same shape:", "", ""),
    ("the condition for recording it.", "", ""),
]
for k, (a, b, c) in enumerate(lines):
    y = ny + k * 24
    add(f'<text x="{PX}" y="{y}" font-size="16" fill="{TEXT}" font-family="{SANS}">{a}'
        f'<tspan fill="{AMBER}" font-family="{MONO}" font-weight="700">{b}</tspan>{c}</text>')

# --------------------------------------------------------------------------- panel 2: timelines
QX = 600
badge(QX + 15, 40, 2)
text(QX + 42, 47, "Record one step, replay it with one launch", size=24, weight=700)
text(QX, 84, "One time step on the GH200, before and after recording it as a CUDA graph.",
     size=16, fill=MUTED)

AX0 = QX + 120  # x of t = 0
PXMS = 23.5  # px per ms; 30 ms -> 705 px
LANE = 30


def lane_label(y, label, col):
    text(QX + 100, y + 20, label, size=15, fill=col, anchor="end", weight=700)
    rect(AX0, y, 30 * PXMS, LANE, PANEL, rx=4)


def axis(y):
    add(f'<line x1="{AX0}" y1="{y}" x2="{AX0 + 30 * PXMS + 10}" y2="{y}" stroke="{FAINT}" '
        'stroke-width="1.2"/>')
    for ms in range(0, 31, 5):
        x = AX0 + ms * PXMS
        add(f'<line x1="{x}" y1="{y}" x2="{x}" y2="{y + 6}" stroke="{FAINT}" stroke-width="1.2"/>')
        text(x, y + 22, f"{ms}", size=13, fill=MUTED, anchor="middle")
    text(AX0 + 30 * PXMS + 18, y + 22, "ms", size=13, fill=MUTED)


# Before: many small launches, GPU mostly idle.
BY = 150
text(QX, BY - 16, "Before: the CPU issues each operation separately", size=17, weight=700)
lane_label(BY, "CPU", AMBER)
lane_label(BY + LANE + 8, "GPU", TEAL)
n_marks = 62  # one mark per ~20 instructions; ~1,250 in all
t = np.linspace(0.15, 29.7, n_marks) + rng.normal(0, 0.04, n_marks)
for tm in t:
    x = AX0 + tm * PXMS
    rect(x, BY + 5, 1.6, LANE - 10, AMBER, opacity=0.9)
    rect(x + 2.4, BY + LANE + 8 + 6, 2.5 / n_marks * PXMS, LANE - 12, TEAL)
axis(BY + 2 * LANE + 18)
text(AX0, BY + 2 * LANE + 66, "~1,250 instructions per step · only ~2.5 ms of GPU work",
     size=15, fill=MUTED)
text(AX0 + 30 * PXMS, BY + 2 * LANE + 66, "30 ms", size=20, fill=TEXT, weight=700,
     anchor="end")

# After: one replay, GPU work packed together.
AY = 352
text(QX, AY - 16, "After: the step is recorded once, then replayed with one command",
     size=17, weight=700)
lane_label(AY, "CPU", AMBER)
lane_label(AY + LANE + 8, "GPU", TEAL)
rect(AX0, AY + 5, 0.35 * PXMS, LANE - 10, AMBER, rx=2)
text(AX0 + 0.35 * PXMS + 8, AY + 20, "replay", size=14, fill=AMBER)
x = AX0 + 0.3 * PXMS
end = AX0 + 4.85 * PXMS
widths = 0.6 + 2.4 * rng.random(40)
widths *= 0.6 * (end - x) / widths.sum()
gap = 0.4 * (end - x) / len(widths)
for wdt in widths:
    rect(x, AY + LANE + 8 + 6, wdt, LANE - 12, TEAL)
    x += wdt + gap
axis(AY + 2 * LANE + 18)
text(AX0 + 5 * PXMS + 14, AY + LANE + 30, "5 ms", size=20, fill=TEXT, weight=700)
text(AX0, AY + 2 * LANE + 66, "~106 instructions, replayed from the recording",
     size=15, fill=MUTED)

# The payoff.
add(f'<rect x="{QX}" y="{AY + 2 * LANE + 96}" width="{W - QX - 40}" height="64" rx="8" '
    f'fill="{PANEL}" stroke="{AMBER}" stroke-width="1.5"/>')
text(QX + 24, AY + 2 * LANE + 136, "6× faster per step,", size=22, fill=AMBER, weight=700)
text(QX + 236, AY + 2 * LANE + 136,
     "with more arithmetic than before: launches were the bottleneck, not maths",
     size=16, fill=TEXT)

add("</svg>")

dest = Path(sys.argv[1] if len(sys.argv) > 1 else "gpu-step.svg")
dest.write_text("\n".join(out) + "\n")
print(f"wrote {dest}")
