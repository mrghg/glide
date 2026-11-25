"""One figure style for every validation tier (docs/validation-plan.md §6).

Generalises ``scripts/validation/fixtures-preview/style.py``: the validated
reference palette of the project's visualisation guidance (the dataviz skill),
recessive axes, one chart surface. Every validation figure imports its colours
and colour scales from here, so the tiers read as one set:

- ``GLIDE`` is categorical slot 1 (blue), ``REFERENCE_MODEL`` slot 2 (orange)
  for a second model (FLEXPART, NAME); an exact or numerical reference solution
  is drawn in ink (``EXACT``), dashed, so it never competes with a model colour.
- ``ALT`` (slot 3, aqua) is a third series, for example a "teeth" variant
  with a piece of physics switched off.
- Footprints use ``SEQ`` (one hue, light to dark) on a logarithmic scale.
- Log ratios use ``DIV`` (blue for GLIDE low, red for GLIDE high, grey at 0).
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#8a8880"
GRID = "#e4e3df"

# Categorical slots, in the palette's fixed order (never cycled).
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]
GLIDE = CATEGORICAL[0]
REFERENCE_MODEL = CATEGORICAL[1]
ALT = CATEGORICAL[2]
EXACT = INK

# Particle-noise bands: the series colour at this alpha.
BAND_ALPHA = 0.22

SEQ = LinearSegmentedColormap.from_list(
    "glide_seq", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
).with_extremes(bad=SURFACE, under=SURFACE)  # empty and below-scale cells read as blank
# Diverging: equal steps per arm, neutral grey midpoint (log ratio 0).
DIV = LinearSegmentedColormap.from_list(
    "glide_div", ["#184f95", "#3987e5", "#9ec5f4", "#f0efec", "#f2a9a8", "#e34948", "#a8292a"]
).with_extremes(bad=SURFACE)
TERRAIN = LinearSegmentedColormap.from_list(
    "terrain_grey", ["#f4f3f0", "#c9c7c0", "#8a8880", "#4a4945"]
)

RC = {
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.edgecolor": INK2,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "axes.grid": False,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "font.size": 9.5,
    "axes.titlesize": 10.5,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "lines.linewidth": 2.0,
    "lines.markersize": 6.0,
    "legend.frameon": False,
    "legend.fontsize": 8.5,
    "savefig.dpi": 130,
    "figure.dpi": 100,
}
plt.rcParams.update(RC)
