"""Figure library for the validation ladder (docs/validation-plan.md §5–6).

The design rules are implemented once, here, and every tier's figure script
uses them:

- **Provenance on every figure.** :func:`provenance` gives the physics tag and a
  short hash of the exact configuration; :func:`stamp` puts both in the figure
  title; :func:`save` writes the PNG with a JSON sidecar (tag, hash, git
  revision, the configuration itself, headline metrics), which the gallery
  generator reads.
- **Log ratio for every difference** (:func:`log_ratio`, :func:`ratio_panel`),
  on one symmetric diverging scale.
- **Particle-noise bands** (:func:`subset_spread`, :func:`band`): the statistic
  is recomputed on ``k`` random disjoint subsets of the particles, and the
  standard error of the full-sample value is estimated as the spread of the
  subset values divided by sqrt(k). Bands are drawn at ±2 standard errors.
- **Footprints on a logarithmic scale in STILT units**
  (:data:`FOOTPRINT_UNITS`, :func:`footprint_panel`).
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import subprocess
import sys
from collections.abc import Callable, Mapping
from pathlib import Path

import numpy as np
from matplotlib.colors import LogNorm, Normalize

from . import style
from .style import plt

REPO_ROOT = Path(__file__).resolve().parents[3]

#: Footprint units used in every footprint panel (STILT: mole fraction per unit
#: surface flux, mol mol-1 / (mol m-2 s-1)).
FOOTPRINT_UNITS = "m² s mol⁻¹"
#: Default half-range of the log10-ratio colour scale (a factor of 10^0.5 ≈ 3.2).
RATIO_HALF_RANGE = 0.5
#: Number of particle subsets for noise bands.
N_SUBSETS = 10


# --------------------------------------------------------------------------- provenance


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()


def physics_tag() -> str:
    """The physics version the figures describe.

    The nearest tag matching ``*physics*`` (``v0-physics``, ``v1-physics`` …) as
    ``git describe`` gives it, so commits after a tag read ``v0-physics-3-gabc1234``.
    Before the freeze tag exists: ``pre-freeze-<short rev>``. A ``+dirty`` suffix
    marks uncommitted changes under ``src/``.
    """
    try:
        try:
            tag = _git("describe", "--tags", "--match", "*physics*")
        except subprocess.CalledProcessError:
            tag = f"pre-freeze-{_git('rev-parse', '--short=8', 'HEAD')}"
        if _git("status", "--porcelain", "--", "src"):
            tag += "+dirty"
        return tag
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def config_hash(config: Mapping) -> str:
    """First 10 hex digits of the SHA-256 of the configuration as canonical JSON.

    For a GLIDE run, pass the resolved configuration (as in
    ``scripts/agent/golden_common.config_hashes``); for an analytic Tier 0 case,
    the case parameters (seed, particle count, time step, profiles …).
    """
    blob = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:10]


def provenance(config: Mapping) -> dict:
    try:
        rev = _git("rev-parse", "HEAD")
    except (subprocess.CalledProcessError, FileNotFoundError):
        rev = "unknown"
    return {
        "physics_tag": physics_tag(),
        "config_hash": config_hash(config),
        "git_rev": rev,
        "created": _dt.datetime.now(_dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def stamp(fig, title: str, prov: Mapping, case_id: str) -> None:
    """Figure title with the physics tag and the configuration hash beneath it."""
    h = fig.get_figheight()
    fig.text(0.01, 1 - 0.12 / h, title, ha="left", va="top", fontsize=12, fontweight="bold")
    fig.text(
        0.01,
        1 - 0.38 / h,
        f"{case_id}  ·  physics {prov['physics_tag']}  ·  config {prov['config_hash']}",
        ha="left",
        va="top",
        fontsize=8.5,
        color=style.INK2,
    )


def save(
    fig,
    out_dir: Path,
    *,
    figure_id: str,
    tier: int,
    title: str,
    caption: str,
    config: Mapping,
    prov: Mapping,
    metrics: Mapping | None = None,
    source: str | None = None,
) -> Path:
    """Write ``<out_dir>/<figure_id>.png`` and its JSON sidecar; close the figure."""
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / f"{figure_id}.png"
    fig.savefig(png, bbox_inches="tight")
    plt.close(fig)
    side = {
        "figure_id": figure_id,
        "tier": tier,
        "title": title,
        "caption": caption,
        "image": png.name,
        **prov,
        "config": json.loads(json.dumps(config, default=str)),
        "metrics": {k: _jsonable(v) for k, v in (metrics or {}).items()},
        "source": source or "",
        "command": " ".join(Path(a).name if i == 0 else a for i, a in enumerate(sys.argv)),
    }
    png.with_suffix(".json").write_text(json.dumps(side, indent=2) + "\n")
    return png


def _jsonable(v):
    if isinstance(v, np.generic):
        v = v.item()
    if isinstance(v, list | tuple):
        return [_jsonable(x) for x in v]
    if isinstance(v, float):
        return float(f"{v:.6g}")
    return v


# --------------------------------------------------------------------------- statistics


def log_ratio(model: np.ndarray, reference: np.ndarray, floor: float = 0.0) -> np.ndarray:
    """log10(model / reference); NaN where either is <= ``floor`` (no information)."""
    model = np.asarray(model, dtype=float)
    reference = np.asarray(reference, dtype=float)
    ok = (model > floor) & (reference > floor)
    out = np.full(np.broadcast(model, reference).shape, np.nan)
    np.log10(model, out=out, where=ok)
    out[ok] -= np.log10(np.broadcast_to(reference, out.shape)[ok])
    return out


def subset_indices(n: int, k: int = N_SUBSETS, seed: int = 0) -> list[np.ndarray]:
    """``k`` random disjoint subsets of ``range(n)`` of (nearly) equal size."""
    perm = np.random.default_rng(seed).permutation(n)
    return np.array_split(perm, k)


def subset_spread(
    statistic: Callable[[np.ndarray], np.ndarray], n: int, k: int = N_SUBSETS, seed: int = 0
) -> np.ndarray:
    """Standard error of ``statistic`` over all ``n`` particles, from ``k`` subsets.

    ``statistic(idx)`` computes the quantity (scalar or array) from the
    particles ``idx``, normalised so that its expectation does not depend on the
    subset size (a density, a variance, a mean). The subset values scatter with
    variance ``k`` times that of the full-sample value, so the standard error is
    ``std(subset values) / sqrt(k)``.
    """
    vals = np.stack([np.asarray(statistic(idx), dtype=float) for idx in subset_indices(n, k, seed)])
    return vals.std(axis=0, ddof=1) / np.sqrt(k)


def band(ax, x, centre, stderr, color=style.GLIDE, label: str | None = "particle noise (±2 s.e.)"):
    """Particle-noise band at ±2 standard errors around ``centre``."""
    centre, stderr = np.asarray(centre), np.asarray(stderr)
    return ax.fill_between(
        x, centre - 2 * stderr, centre + 2 * stderr, color=color, alpha=style.BAND_ALPHA,
        lw=0, label=label,
    )  # fmt: skip


def log_band(
    ax, x, centre, stderr, color=style.GLIDE, label: str | None = "particle noise (±2 s.e.)"
):
    """Noise band for a log-axis quantity, clipped above zero."""
    centre, stderr = np.asarray(centre), np.asarray(stderr)
    lo = np.clip(centre - 2 * stderr, centre * 1e-3, None)
    return ax.fill_between(
        x, lo, centre + 2 * stderr, color=color, alpha=style.BAND_ALPHA, lw=0, label=label
    )


# --------------------------------------------------------------------------- panels


def footprint_norm(*fields: np.ndarray, decades: float = 4.0) -> LogNorm:
    """One logarithmic scale for several footprints: top at the largest value,
    ``decades`` below it at the bottom."""
    vmax = max(float(np.nanmax(f)) for f in fields)
    return LogNorm(vmin=vmax * 10.0**-decades, vmax=vmax)


def footprint_panel(ax, x, y, field, norm, *, title: str = "", units: str = FOOTPRINT_UNITS):
    """A footprint on the shared logarithmic scale; zero cells left blank."""
    f = np.where(np.asarray(field) > 0, field, np.nan)
    m = ax.pcolormesh(x, y, f, cmap=style.SEQ, norm=norm, shading="nearest", rasterized=True)
    if title:
        ax.set_title(title)
    cb = ax.figure.colorbar(m, ax=ax, shrink=0.9, pad=0.02)
    cb.set_label(units)
    cb.outline.set_visible(False)
    return m


def ratio_panel(ax, x, y, ratio, *, title: str = "log₁₀(GLIDE / reference)",
                half_range: float = RATIO_HALF_RANGE):  # fmt: skip
    """A log10 ratio on the shared symmetric diverging scale."""
    m = ax.pcolormesh(
        x, y, ratio, cmap=style.DIV, norm=Normalize(-half_range, half_range),
        shading="nearest", rasterized=True,
    )  # fmt: skip
    ax.set_title(title)
    cb = ax.figure.colorbar(m, ax=ax, shrink=0.9, pad=0.02, extend="both")
    cb.set_label("log₁₀ ratio")
    cb.outline.set_visible(False)
    return m


def recessive_grid(ax, axis: str = "y") -> None:
    ax.grid(True, axis=axis, color=style.GRID, lw=0.8)
    ax.set_axisbelow(True)


def new_figure(ncols: int, nrows: int = 1, *, width: float = 4.2, height: float = 3.4, **kw):
    """Figure with room above the axes for :func:`stamp`."""
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(width * ncols, height * nrows + 0.6), squeeze=False, **kw
    )
    fig.subplots_adjust(top=1 - 0.95 / (height * nrows + 0.6))
    return fig, axes
