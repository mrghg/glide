"""Shared figure style for the fixtures preview: the validated reference palette
(dataviz skill), recessive axes, one chart surface."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3df"
SITES = {"MHD": "#2a78d6", "RGL": "#eb6834", "TAC": "#1baf7a"}
SEQ = LinearSegmentedColormap.from_list(
    "blue_seq", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
)
DIV = LinearSegmentedColormap.from_list(
    "blue_orange", ["#256abf", "#86b6ef", "#f0efec", "#f3a582", "#d95926"]
)
GREY = LinearSegmentedColormap.from_list(
    "terrain_grey", ["#f4f3f0", "#c9c7c0", "#8a8880", "#4a4945"]
)

plt.rcParams.update(
    {
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
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "semibold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "lines.linewidth": 2.0,
        "legend.frameon": False,
    }
)

SITE_COORDS = {"MHD": (-9.9046, 53.3267), "RGL": (-2.5400, 51.9970), "TAC": (1.1387, 52.5176)}


def mark_sites(ax, label=True):
    for name, (lon, lat) in SITE_COORDS.items():
        ax.plot(lon, lat, "o", ms=8, mfc=SITES[name], mec=SURFACE, mew=2, zorder=5)
        if label:
            ax.annotate(
                name,
                (lon, lat),
                xytext=(6, 4),
                textcoords="offset points",
                fontsize=9,
                color=INK,
                zorder=6,
            )
