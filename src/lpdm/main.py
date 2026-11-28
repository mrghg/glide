"""Top-level LPDM orchestration entry point.

This module provides a minimal end-to-end backward trajectory run mode suitable
for early cloud testing (including Vertex AI user-managed notebooks).

The CLI surface is intentionally small: a YAML ``--config`` plus a few overrides
(``--device``, ``--output-uri``, ``--start-time``). Schema is defined in
:mod:`lpdm.config`.
"""

from __future__ import annotations

import argparse
import gc
import logging
import math
import os
import resource
import sys
import time
from collections import OrderedDict
from collections.abc import Iterator, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch

from lpdm.air_density import DENSITY_WEIGHTING_ATTRS, ParticleDryAirDensity
from lpdm.config import (
    MPS_FLOAT64_ACCUMULATION_MESSAGE,
    N_SUBSTEPS_DEFAULT,
    ConcreteRelease,
    RunConfig,
)
from lpdm.convection import (
    ConvectionScheme,
)
from lpdm.convection import (
    get_scheme as get_convection_scheme,
)
from lpdm.convection import (
    list_schemes as list_convection_schemes,
)
from lpdm.footprint_gridder import FootprintGridder
from lpdm.gpu_engine import (
    GPUEngine,
    frame_grid_bounds,
    trilinear_sample,
    use_static_step_path,
)
from lpdm.local_frame import DEGREES, HorizontalFrame
from lpdm.met_reader import (
    ArcoEra5ZarrReader,
    BoundingBoxRequest,
    HourlyMetTensors,
    MetReader,
    SpatialBounds,
    TimeBounds,
)
from lpdm.output_writer import OutputWriter
from lpdm.phase_timers import install as install_phase_timer
from lpdm.phase_timers import make_phase_timer
from lpdm.release_generator import generate_batch_particles
from lpdm.runtime import DEVICE
from lpdm.turbulence import TurbulenceScheme, get_scheme, list_schemes
from lpdm.turbulence.hanna import SURFACE_LAYER_OVERRIDE_WARNING
from lpdm.vertical_grid import default_agl_levels

LOGGER = logging.getLogger(__name__)


def _emit_profile_summary(prof, device: torch.device, trace_path: Path, wall_s: float) -> None:
    """Write a Chrome trace and print a CPU/GPU summary for a profiled step window.

    The summary is built to localise *why the GPU idles*: the GPU-busy fraction, any
    host-sync ops (``cudaStreamSynchronize`` ⇒ a stall), and the top ops by GPU and
    CPU self-time (many tiny GPU ops ⇒ launch-bound; large CPU spans with little GPU
    ⇒ Python/met-bound). See docs/architecture.md §5.
    """

    try:
        prof.export_chrome_trace(str(trace_path))
        LOGGER.info("profiler: chrome trace -> %s (open in https://ui.perfetto.dev)", trace_path)
    except Exception as exc:  # pragma: no cover - defensive
        LOGGER.warning("profiler: chrome trace export failed: %s", exc)

    ka = prof.key_averages()

    def _dev_us(evt) -> float:
        # PyTorch renamed self_cuda_time_total -> self_device_time_total across versions.
        for attr in ("self_device_time_total", "self_cuda_time_total"):
            val = getattr(evt, attr, 0) or 0
            if val:
                return float(val)
        return 0.0

    gpu_us = sum(_dev_us(e) for e in ka)
    busy = 100.0 * (gpu_us / 1e6) / wall_s if wall_s > 0 else 0.0

    print("\n" + "=" * 74)
    print(f"GLIDE profiler — {device.type} — window wall {wall_s:.3f}s — GPU-busy ~{busy:.1f}%")
    print("=" * 74)

    syncs = [e for e in ka if "synchronize" in e.key.lower()]
    if syncs:
        print("host syncs in window (each one stalls the GPU on the CPU):")
        for e in syncs:
            print(
                f"  {e.key}: count={e.count}  cpu_total={getattr(e, 'cpu_time_total', 0) / 1000:.1f}ms"
            )
    else:
        print("host syncs in window: none detected by name")

    def _safe_table(sort_by: str):
        try:
            return ka.table(sort_by=sort_by, row_limit=20)
        except Exception:
            return None

    gpu_tbl = _safe_table("self_cuda_time_total") or _safe_table("self_device_time_total")
    if gpu_tbl and device.type == "cuda":
        print("\n--- top ops by GPU (device) self-time ---")
        print(gpu_tbl)
    cpu_tbl = _safe_table("self_cpu_time_total")
    if cpu_tbl:
        print("\n--- top ops by CPU self-time (look for many calls = launch-bound) ---")
        print(cpu_tbl)
    print("=" * 74 + "\n")


class _StepProfiler:
    """Profiles a contiguous window of cursor-loop steps when ``GLIDE_PROFILE`` is set.

    Skips ``warmup`` steps (one-time ``torch.compile`` + settle), profiles the next
    ``active`` steps, writes a trace + summary, then (by default) exits the process —
    a profiling run doesn't need the full output. Disabled = a no-op object isn't even
    created, so steady-state runs pay nothing. Env knobs:

    * ``GLIDE_PROFILE``           — enable (truthy)
    * ``GLIDE_PROFILE_WARMUP``    — steps to skip first (default 5; lets the CUDA-graph
                                    trees warm up so the window is steady-state replay)
    * ``GLIDE_PROFILE_STEPS``     — steps to profile (default 20)
    * ``GLIDE_PROFILE_TRACE``     — trace output path (default ./glide_profile_trace.json)
    * ``GLIDE_PROFILE_CONTINUE``  — finish the run instead of exiting after the window
    """

    def __init__(
        self, *, device: torch.device, warmup: int, active: int, trace_path: Path, exit_after: bool
    ) -> None:
        self.device = device
        self.warmup = max(0, warmup)
        self.active = max(1, active)
        self.trace_path = trace_path
        self.exit_after = exit_after
        self.n = 0
        self.prof = None
        self.t0 = 0.0
        self.done = False

    def before_step(self) -> None:
        """Call at the TOP of the cursor-loop body: starts capture once warmup is done."""
        if self.done or self.prof is not None or self.n < self.warmup:
            return
        acts = [torch.profiler.ProfilerActivity.CPU]
        if self.device.type == "cuda":
            acts.append(torch.profiler.ProfilerActivity.CUDA)
        self.prof = torch.profiler.profile(activities=acts)
        self.prof.__enter__()
        self.t0 = time.perf_counter()
        LOGGER.info("profiler: capturing %d steps (after %d warmup)…", self.active, self.warmup)

    def after_step(self) -> None:
        """Call at the BOTTOM of the cursor-loop body: stops + reports once the window ends."""
        if self.done:
            return
        self.n += 1
        if self.prof is not None and self.n >= self.warmup + self.active:
            wall = time.perf_counter() - self.t0
            self.prof.__exit__(None, None, None)
            _emit_profile_summary(self.prof, self.device, self.trace_path, wall)
            self.prof = None
            self.done = True
            if self.exit_after:
                LOGGER.info(
                    "profiler: window complete; exiting (GLIDE_PROFILE_CONTINUE=1 to finish the run)"
                )
                raise SystemExit(0)


def _make_step_profiler(device: torch.device) -> _StepProfiler | None:
    """Build the step profiler from env, or None when ``GLIDE_PROFILE`` is unset."""
    if os.environ.get("GLIDE_PROFILE", "") in ("", "0", "false", "False"):
        return None
    return _StepProfiler(
        device=device,
        warmup=int(os.environ.get("GLIDE_PROFILE_WARMUP", "10")),
        active=int(os.environ.get("GLIDE_PROFILE_STEPS", "20")),
        trace_path=Path(os.environ.get("GLIDE_PROFILE_TRACE", "glide_profile_trace.json")),
        exit_after=os.environ.get("GLIDE_PROFILE_CONTINUE", "") in ("", "0", "false", "False"),
    )


FOOTPRINT_UNITS_DOC = {
    "value": "Σ over active particles in cell of (weight_i × dt_step_seconds)",
    "value_dimensionality": "(dimensionless mass fraction) × seconds — residence time per unit released mass",
    "particle_weight_convention": "1/n_particles for uniform releases; each particle represents 1/N of the total released mass",
    "notes": (
        "Raw residence-time accumulator; downstream code is responsible for converting to a physical "
        "sensitivity (e.g. ppm per emission flux), which requires cell volume, mixed-layer thickness, "
        "air density, and molar-mass conversions. See Lin et al. 2003 (STILT) or Seibert & Frank 2004 "
        "for standard recipes."
    ),
}


class PreflightValidationError(ValueError):
    """Raised when run inputs are invalid before stepping begins."""


def _density_weighting_met_keys(cfg: RunConfig) -> tuple[str, ...]:
    """Met channels the footprint's density weighting reads (decision 0022)."""

    if cfg.output_grid.density_weighting == "dry_air":
        return ParticleDryAirDensity.CHANNELS
    return ()


DEPOSITION_POSITION_DOC = {
    "endpoint": (
        "each step's residence is counted at the particle's position at the end of the "
        "step (after advection, turbulence and convection)"
    ),
    "trapezoid": (
        "each step's residence is split in two halves: dt/2 counted at the particle's "
        "position at the start of the step and dt/2 at its position at the end (after "
        "advection, turbulence and convection); each half is split between time-ago bins "
        "by its overlap with them and, with dry-air weighting, uses the density at its own "
        "position and time"
    ),
}


def _deposition_attrs(cfg: RunConfig) -> dict[str, str]:
    """Deposition settings for the footprint store attributes (task A2)."""

    position = cfg.output_grid.deposition.position
    return {
        "deposition_position": position,
        "deposition_position_description": DEPOSITION_POSITION_DOC[position],
    }


def _footprint_units_doc(cfg: RunConfig) -> dict[str, object]:
    """`footprint_units` block of run_metadata.json for the run's weighting."""

    weighting = cfg.output_grid.density_weighting
    doc: dict[str, object] = dict(FOOTPRINT_UNITS_DOC)
    doc.update(DENSITY_WEIGHTING_ATTRS[weighting])
    doc.update(_deposition_attrs(cfg))
    doc["accumulation_dtype"] = cfg.output_grid.accumulation_dtype
    if weighting == "dry_air":
        doc["value"] = (
            "Σ over active particles in cell of (weight_i × dt_step_seconds / rho_d,i), "
            "rho_d,i the dry-air density at particle i as it is counted"
        )
        doc["value_dimensionality"] = (
            "(dimensionless mass fraction) × s m3 kg-1 — residence time per unit released "
            "mass, divided by the dry-air density"
        )
    return doc


def _resolve_device(name: str) -> str:
    """Resolve a config device string ('auto', 'cpu', 'cuda', 'mps', 'cuda:N') to a concrete torch device."""

    if name == "auto":
        return str(DEVICE)
    # Fail fast with a clear message if an explicit CUDA/MPS device was requested
    # but this torch build can't provide it — otherwise the failure surfaces deep
    # in the first tensor allocation. Matters on Isambard AI, where the wrong
    # (CPU-only or x86) torch wheel in the venv is an easy mistake.
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise PreflightValidationError(
            f"device={name!r} requested but torch.cuda.is_available() is False. "
            "Check the venv has a CUDA (and, on Isambard AI, ARM64/aarch64) torch "
            "build and that the CUDA module is loaded (e.g. `module load cuda/12.6`)."
        )
    if name == "mps" and not torch.backends.mps.is_available():
        raise PreflightValidationError(
            f"device={name!r} requested but torch.backends.mps.is_available() is False."
        )
    return name


def _check_device_dtypes(cfg: RunConfig, device_str: str) -> None:
    """Refuse a configuration whose dtypes the resolved device cannot allocate.

    The config validator catches an explicit ``device: mps``; this catches
    ``device: auto`` resolving to MPS, which has no float64 tensors, before any
    met or particle work is done.
    """

    if torch.device(device_str).type == "mps" and cfg.output_grid.accumulation_dtype == "float64":
        raise PreflightValidationError(MPS_FLOAT64_ACCUMULATION_MESSAGE)


def _current_rss_bytes() -> int | None:
    """Return process memory bytes when available."""

    try:
        with open("/proc/self/status", encoding="utf-8") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        return int(parts[1]) * 1024
    except OSError:
        pass

    try:
        ru_maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform == "darwin":
            return int(ru_maxrss)
        return int(ru_maxrss) * 1024
    except OSError:
        return None


def _device_memory_bytes(device: torch.device) -> tuple[int | None, int | None]:
    if device.type == "cuda" and torch.cuda.is_available():
        idx = device.index if device.index is not None else torch.cuda.current_device()
        return int(torch.cuda.memory_allocated(idx)), int(torch.cuda.memory_reserved(idx))

    if device.type == "mps" and hasattr(torch, "mps"):
        allocated_fn = getattr(torch.mps, "current_allocated_memory", None)
        driver_fn = getattr(torch.mps, "driver_allocated_memory", None)
        allocated = int(allocated_fn()) if callable(allocated_fn) else None
        reserved = int(driver_fn()) if callable(driver_fn) else None
        return allocated, reserved

    return None, None


def _format_gib(num_bytes: int | None) -> str:
    if num_bytes is None:
        return "n/a"
    return f"{num_bytes / (1024**3):.3f} GiB"


def _met_window_start(reader: MetReader, dt: datetime) -> datetime:
    """Start of the met window containing `dt`, on the reader's own sample grid.

    This is the met-cache key and the identity of the current bracket, so it has to
    agree exactly with what the reader will fetch. Asking the reader keeps the two
    in step for any source cadence.
    """

    return reader.floor_to_window_start(dt)


def _step_window_time(t_cursor: datetime, delta_s: float) -> datetime:
    """The time that picks a backward step's met window: the step's midpoint.

    A step covers (t_cursor - delta_s, t_cursor]. Taking the window that contains
    `t_cursor` would give a step ending exactly on a met timestamp the window that
    starts there, which the step never enters: it would need one field beyond the
    end of a store, and would hold the next window's mean for that step (decision
    0020). The midpoint is also where the time weight is evaluated
    (`_advection_alpha_wind_mean`), so the two always agree.
    """

    return t_cursor - timedelta(seconds=0.5 * delta_s)


def _backward_step_seconds(remaining_s: float, dt_s: float) -> float:
    """Length of the next backward step when `remaining_s` is left to the terminus.

    Full steps of `dt_s` end exactly on the terminus; any remainder is taken by the
    first step. Remainders within 1 ms of 0 or `dt_s` count as whole steps.
    """

    remainder = remaining_s % dt_s
    if remainder < 1e-3 or dt_s - remainder < 1e-3:
        return min(dt_s, remaining_s)
    return remainder


def _met_interval(reader: MetReader) -> timedelta:
    return timedelta(seconds=reader.met_interval_seconds)


def _footprint_time_bin_index(
    release_end: datetime, t_cursor: datetime, n_time_bins: int, time_bin_seconds: float = 3600.0
) -> int:
    """Map the current backward integration cursor to a 0-based time_ago bin."""

    if n_time_bins <= 0:
        raise ValueError("n_time_bins must be > 0")

    elapsed_seconds = max(0.0, release_end.timestamp() - t_cursor.timestamp())
    return min(n_time_bins - 1, int(elapsed_seconds / time_bin_seconds))


def _footprint_time_bin_count(cfg: RunConfig, releases: Sequence[ConcreteRelease]) -> int:
    """Length of the footprint's ``time_ago`` axis.

    Ages are measured from each release window's END in bins of
    ``output_grid.time_bin_seconds`` (decision 0016), and a particle released at
    the window's start is followed for ``simulation.length_seconds`` from there,
    so it reaches an age of the length plus the release duration.
    ``output_grid.n_time_bins`` counts the bins of travel to resolve; the axis
    adds enough bins for the longest release window, so no particle's resolved
    travel falls off the end. One bin stays one bin: the time-integrated
    footprint.

    With `trapezoid` deposition the axis also covers the counted ages exactly. A
    particle is counted from the step boundary at or below its release, for the
    run length, so when a release window starts above the batch's step grid its
    oldest counted ages run past the length plus the duration by that offset
    (less than one step); trapezoid splits the step's ages exactly
    (`_age_bin_pieces`), so the axis adds the largest such offset. `endpoint` bins
    each step by its starting age, which stays inside the axis, so its axis is
    unchanged.
    """

    n_time_bins = cfg.output_grid.n_time_bins
    if n_time_bins == 1:
        return 1
    longest_window_s = max(float(r.duration_seconds) for r in releases)
    if cfg.output_grid.deposition.position == "trapezoid":
        longest_window_s += _largest_start_offset_from_step_grid(cfg)
    return n_time_bins + math.ceil(longest_window_s / cfg.output_grid.time_bin_seconds)


def _largest_start_offset_from_step_grid(cfg: RunConfig) -> float:
    """Largest distance (s) of a release window's start above the step grid of its
    batch. The runtime's steps end on the batch terminus (its earliest release start
    minus the run length) plus whole steps; 0 when every start is on that grid."""

    dt_s = float(cfg.simulation.dt_seconds)
    largest = 0.0
    for batch in cfg.expand_to_batches():
        terminus = min(r.release_time for r in batch.releases) - timedelta(
            seconds=cfg.simulation.length_seconds
        )
        for rel in batch.releases:
            off = (rel.release_time - terminus).total_seconds() % dt_s
            largest = max(largest, 0.0 if min(off, dt_s - off) < 1e-3 else off)
    return largest


def _build_footprint_dataset_metadata(
    cfg: RunConfig,
    gridder: FootprintGridder,
    releases: Sequence[ConcreteRelease],
) -> tuple[dict[str, object], dict[str, float]]:
    """Build coordinate metadata for the 5D footprint Zarr dataset.

    The leading ``release_time`` axis is populated from the full expanded
    schedule — one timestamp per :class:`ConcreteRelease` in execution order.
    For single-release runs this is a length-1 axis carrying the release's
    start time. Release attrs (lon/lat/alt) come from the schedule's common
    release point (currently all variants use a single location).
    """

    lon_edges = np.linspace(gridder.lon_min, gridder.lon_max, gridder.n_x + 1, dtype=np.float64)
    lat_edges = np.linspace(gridder.lat_min, gridder.lat_max, gridder.n_y + 1, dtype=np.float64)
    z_edges_m = gridder.z_edges.detach().cpu().numpy().astype(np.float64)
    # Bins are output_grid.time_bin_seconds wide (the runtime's `t_idx`), counted
    # back from each release window's end. A single bin holds every age: up to
    # the simulation length plus the longest release window.
    if gridder.n_t == 1:
        longest_window_s = max(float(r.duration_seconds) for r in releases)
        time_bin_start_h = np.array([0.0])
        time_bin_end_h = np.array([(cfg.simulation.length_seconds + longest_window_s) / 3600.0])
    else:
        bin_h = cfg.output_grid.time_bin_seconds / 3600.0
        time_bin_start_h = np.arange(gridder.n_t, dtype=np.float64) * bin_h
        time_bin_end_h = time_bin_start_h + bin_h

    # Per-release coords ride along the `release` axis: a single timestamp may repeat across
    # sites (multi-site) or every release may be a distinct (location, time) (satellite).
    release_times = np.array(
        [np.datetime64(rel.release_time.replace(tzinfo=None), "ns") for rel in releases]
    )
    release_durations_s = np.array([float(r.duration_seconds) for r in releases], dtype=np.float64)
    release_lon = np.array([float(r.lon) for r in releases], dtype=np.float64)
    release_lat = np.array([float(r.lat) for r in releases], dtype=np.float64)
    release_alt = np.array([float(r.alt_agl_m) for r in releases], dtype=np.float64)

    coords: dict[str, object] = {
        "release_time": ("release", release_times),
        "release_duration_seconds": ("release", release_durations_s),
        "release_lon": ("release", release_lon),
        "release_lat": ("release", release_lat),
        "release_alt_agl_m": ("release", release_alt),
        "time_ago": np.arange(gridder.n_t, dtype=np.int64),
        "time_ago_start_hours": ("time_ago", time_bin_start_h),
        "time_ago_end_hours": ("time_ago", time_bin_end_h),
        "z_bin": 0.5 * (z_edges_m[:-1] + z_edges_m[1:]),
        "z_bottom_m": ("z_bin", z_edges_m[:-1]),
        "z_top_m": ("z_bin", z_edges_m[1:]),
        "latitude": 0.5 * (lat_edges[:-1] + lat_edges[1:]),
        "longitude": 0.5 * (lon_edges[:-1] + lon_edges[1:]),
        "latitude_edge": ("latitude_edge", lat_edges),
        "longitude_edge": ("longitude_edge", lon_edges),
    }
    # `site` label only when present (multi-site); single-location runs omit it.
    if any(r.label is not None for r in releases):
        coords["site"] = ("release", np.array([r.label or "" for r in releases]))
    # Top of each release's height range, only when some release has one (`kind: samples`);
    # `release_alt_agl_m` is then the bottom. Other runs omit it, so their output is unchanged.
    if any(r.alt_top_agl_m is not None for r in releases):
        coords["release_alt_top_agl_m"] = (
            "release",
            np.array(
                [r.alt_top_agl_m if r.alt_top_agl_m is not None else r.alt_agl_m for r in releases],
                dtype=np.float64,
            ),
        )
    # The weighting and units of the values (decision 0022), for downstream code
    # (e.g. `comparison.to_stilt_surface_footprint`) to read, not assume.
    return coords, {
        **DENSITY_WEIGHTING_ATTRS[cfg.output_grid.density_weighting],
        **_deposition_attrs(cfg),
    }


def _validate_meteorology_space_coverage(reader: MetReader, cfg: RunConfig) -> None:
    """Refuse a met domain that the meteorology grid does not cover.

    Particles outside the grid are not an error in the interpolation: they are
    held at its edge. A met domain in one longitude convention over a grid in
    the other (-120..-55 against 240..305) therefore ran silently with the
    winds of the grid's edge (task B12, 24 Nov 2026). The reader normalises
    longitudes; this check makes any remaining mismatch an error.
    """

    coverage = reader.get_lon_lat_coverage()
    if coverage is None:
        return
    g_lon0, g_lon1, g_lat0, g_lat1 = coverage
    md = cfg.met_domain
    tol = 1e-3
    lon_ok = md.lon_bounds[0] >= g_lon0 - tol and md.lon_bounds[1] <= g_lon1 + tol
    lat_ok = md.lat_bounds[0] >= g_lat0 - tol and md.lat_bounds[1] <= g_lat1 + tol
    if not (lon_ok and lat_ok):
        raise PreflightValidationError(
            "met_domain is not inside the meteorology grid: "
            f"met_domain lon {tuple(md.lon_bounds)}, lat {tuple(md.lat_bounds)}; "
            f"grid lon ({g_lon0}, {g_lon1}), lat ({g_lat0}, {g_lat1}). "
            "Check the longitude convention (-180..180 vs 0..360) and the bounds."
        )


def _validate_meteorology_time_coverage(
    reader: MetReader,
    cfg: RunConfig,
    release_end: datetime,
    sim_start: datetime,
) -> None:
    """Fail early when the configured run requires times outside dataset coverage."""

    available_start, available_end = reader.get_time_coverage()
    # Backward steps take the met window containing each step's midpoint
    # (`_step_window_time`), so the latest window needed is the one ending at or
    # after `release_end`: no further field when `release_end` lies on the met grid.
    required_start = _met_window_start(reader, sim_start)
    end_window = _met_window_start(reader, release_end)
    required_end = end_window if end_window == release_end else end_window + _met_interval(reader)

    if required_start < available_start or required_end > available_end:
        cadence_h = reader.met_interval_seconds / 3600.0
        raise PreflightValidationError(
            "Meteorological dataset does not cover the requested simulation window. "
            f"Need {cadence_h:g}-hourly data from {required_start.isoformat()} through "
            f"{required_end.isoformat()}, "
            f"but dataset only covers {available_start.isoformat()} through {available_end.isoformat()}. "
            "Reduce simulation.length_seconds, choose a different start_time, or use a dataset with wider time coverage."
        )


@dataclass(frozen=True)
class OutputPaths:
    endpoint_particles: str
    trajectory_diagnostics: str
    footprints: str
    metadata: str


@dataclass
class MemoryStats:
    peak_rss_bytes: int = 0
    peak_device_allocated_bytes: int = 0
    peak_device_reserved_bytes: int = 0

    def observe(self, device: torch.device) -> tuple[int | None, int | None, int | None]:
        rss_bytes = _current_rss_bytes()
        allocated, reserved = _device_memory_bytes(device)

        if rss_bytes is not None:
            self.peak_rss_bytes = max(self.peak_rss_bytes, rss_bytes)
        if allocated is not None:
            self.peak_device_allocated_bytes = max(self.peak_device_allocated_bytes, allocated)
        if reserved is not None:
            self.peak_device_reserved_bytes = max(self.peak_device_reserved_bytes, reserved)

        return rss_bytes, allocated, reserved


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a backward LPDM trajectory simulation from a YAML config"
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the run YAML config (see configs/example_mhd_january.yaml)",
    )
    parser.add_argument("--device", default=None, help="Override simulation.device")
    parser.add_argument("--output-uri", default=None, help="Override io.output_uri")
    parser.add_argument(
        "--start-time", default=None, help="Override simulation.start_time (UTC ISO)"
    )
    return parser


def _build_output_paths(output_uri: str) -> OutputPaths:
    base_uri = output_uri.rstrip("/")
    return OutputPaths(
        endpoint_particles=f"{base_uri}/endpoint_particles.parquet",
        trajectory_diagnostics=f"{base_uri}/trajectory_diagnostics.parquet",
        footprints=f"{base_uri}/footprints.zarr",
        metadata=f"{base_uri}/run_metadata.json",
    )


def _config_metadata(
    cfg: RunConfig, release_end: datetime, sim_start: datetime
) -> dict[str, object]:
    # mode="json" serializes datetimes / tuples to JSON-friendly types.
    return {
        **cfg.model_dump(mode="json"),
        "release_end_time": release_end.isoformat(),
        "simulation_start_time": sim_start.isoformat(),
    }


def _runtime_metadata(
    cfg: RunConfig,
    step_count: int,
    hour_windows: int,
    memory_stats: MemoryStats,
    *,
    status: str,
    **extra: object,
) -> dict[str, object]:
    runtime = {
        "device": _resolve_device(cfg.simulation.device),
        "status": status,
        "steps": step_count,
        "hour_windows": hour_windows,
        "met_cache_max_hours": cfg.memory.met_cache_max_hours,
        "peak_rss_bytes": memory_stats.peak_rss_bytes,
        "peak_device_allocated_bytes": memory_stats.peak_device_allocated_bytes,
        "peak_device_reserved_bytes": memory_stats.peak_device_reserved_bytes,
    }
    runtime.update(extra)
    return runtime


def _memory_guard_enabled(cfg: RunConfig) -> bool:
    mem = cfg.memory
    return any(
        limit is not None
        for limit in (
            mem.guard_max_rss_gib,
            mem.guard_max_device_allocated_gib,
            mem.guard_max_device_reserved_gib,
        )
    )


def _write_memory_guard_metadata(
    writer: OutputWriter,
    metadata_path: str,
    cfg: RunConfig,
    release_end: datetime,
    sim_start: datetime,
    step_count: int,
    hour_windows: int,
    outputs: OutputPaths,
    memory_stats: MemoryStats,
    *,
    reason: str,
    **runtime_extra: object,
) -> None:
    writer.write_metadata_json(
        metadata_path,
        {
            "status": "aborted_memory_guard",
            "reason": reason,
            "config": _config_metadata(cfg, release_end, sim_start),
            "runtime": _runtime_metadata(
                cfg,
                step_count,
                hour_windows,
                memory_stats,
                status="aborted_memory_guard",
                **runtime_extra,
            ),
            "outputs": {
                "endpoint_particles": outputs.endpoint_particles,
                "trajectory_diagnostics": outputs.trajectory_diagnostics,
                "footprints": outputs.footprints,
                "metadata": outputs.metadata,
            },
            "footprint_units": _footprint_units_doc(cfg),
        },
    )


def _raise_if_memory_guard_exceeded(
    writer: OutputWriter,
    metadata_path: str,
    cfg: RunConfig,
    release_end: datetime,
    sim_start: datetime,
    step_count: int,
    hour_windows: int,
    outputs: OutputPaths,
    memory_stats: MemoryStats,
    rss_bytes: int | None,
    allocated: int | None,
    reserved: int | None,
) -> None:
    mem = cfg.memory
    guard_checks = (
        (
            mem.guard_max_rss_gib,
            rss_bytes,
            "RSS exceeded memory guard threshold",
            "RSS",
            "guard_limit_rss_bytes",
            "guard_observed_rss_bytes",
        ),
        (
            mem.guard_max_device_allocated_gib,
            allocated,
            "Device allocated memory exceeded memory guard threshold",
            "device allocated memory",
            "guard_limit_device_allocated_bytes",
            "guard_observed_device_allocated_bytes",
        ),
        (
            mem.guard_max_device_reserved_gib,
            reserved,
            "Device reserved memory exceeded memory guard threshold",
            "device reserved memory",
            "guard_limit_device_reserved_bytes",
            "guard_observed_device_reserved_bytes",
        ),
    )

    for limit_gib, observed_bytes, reason, label, limit_key, observed_key in guard_checks:
        if limit_gib is None or observed_bytes is None:
            continue

        limit_bytes = int(limit_gib * (1024**3))
        if observed_bytes <= limit_bytes:
            continue

        _write_memory_guard_metadata(
            writer,
            metadata_path,
            cfg,
            release_end,
            sim_start,
            step_count,
            hour_windows,
            outputs,
            memory_stats,
            reason=reason,
            **{limit_key: limit_bytes, observed_key: observed_bytes},
        )
        raise MemoryError(
            f"Memory safety guard triggered: {label} reached "
            f"{_format_gib(observed_bytes)} which is above {limit_gib:.3f} GiB"
        )


def _met_domain_bounds(cfg: RunConfig) -> SpatialBounds:
    md = cfg.met_domain
    return SpatialBounds(
        lon_min=md.lon_bounds[0],
        lon_max=md.lon_bounds[1],
        lat_min=md.lat_bounds[0],
        lat_max=md.lat_bounds[1],
        z_min=0.0,
        z_max=md.alt_max_m,
    )


def _within_met_domain(
    particles: torch.Tensor, cfg: RunConfig, frame: HorizontalFrame
) -> torch.Tensor:
    """Boolean [N] mask: True where a particle is inside ``met_domain``.

    ``particles`` hold horizontal positions in ``frame`` (`lpdm.local_frame`); the
    lon/lat bounds are converted to it (the frame is affine and increasing in
    both, so the box is the same box).

    Outside the lon/lat bbox or above ``alt_max_m`` there is no valid met — the
    engine's coordinate normalisation clamps to the grid edge, so such a particle
    would otherwise keep being pushed by edge winds for the rest of the
    integration (wasted compute; it can't contribute to the footprint either,
    being outside ``output_grid``). The runtime kills these particles. The z=0
    floor is handled by surface reflection, so there is no lower-altitude kill.
    """

    md = cfg.met_domain
    x = particles[:, 0]
    y = particles[:, 1]
    alt = particles[:, 2]
    return (
        (x >= frame.x_from_lon(md.lon_bounds[0]))
        & (x <= frame.x_from_lon(md.lon_bounds[1]))
        & (y >= frame.y_from_lat(md.lat_bounds[0]))
        & (y <= frame.y_from_lat(md.lat_bounds[1]))
        & (alt <= md.alt_max_m)
    )


def _position_frame(cfg: RunConfig) -> HorizontalFrame:
    """The run's horizontal frame for particle positions (task A4).

    Metres from the met domain's centre (`HorizontalFrame.local_metric`).
    ``GLIDE_POSITION_FRAME=degrees`` keeps (lon, lat) in degrees, the arithmetic
    of the code before the frame existed, for A/B checks (docs/architecture.md §8).
    """

    choice = os.environ.get("GLIDE_POSITION_FRAME", "local_metric")
    if choice == "degrees":
        return DEGREES
    if choice != "local_metric":
        raise ValueError(f"GLIDE_POSITION_FRAME must be local_metric or degrees, not {choice!r}")
    md = cfg.met_domain
    return HorizontalFrame.local_metric(md.lon_bounds, md.lat_bounds)


def _get_hourly_met_window(
    reader: MetReader,
    cfg: RunConfig,
    t_cursor: datetime,
    met_cache: OrderedDict[datetime, HourlyMetTensors],
) -> tuple[HourlyMetTensors, int]:
    hour_key = _met_window_start(reader, t_cursor)
    if cfg.memory.met_cache_max_hours > 0 and hour_key in met_cache:
        met_cache.move_to_end(hour_key)
        return met_cache[hour_key], 0

    met_window = reader.fetch_hourly_window(
        BoundingBoxRequest(
            spatial=_met_domain_bounds(cfg),
            time=TimeBounds(start=hour_key, end=hour_key + _met_interval(reader)),
        )
    )

    if cfg.memory.met_cache_max_hours > 0:
        met_cache[hour_key] = met_window
        while len(met_cache) > cfg.memory.met_cache_max_hours:
            _, evicted = met_cache.popitem(last=False)
            del evicted

    return met_window, 1


def _warn_if_met_cache_thrashes(
    cfg: RunConfig,
    batches: list,
    sim_length_s: float,
    met_interval_seconds: int = 3600,
) -> None:
    """Warn when `met_cache_max_hours` is too small to retain the met OVERLAP between
    consecutive backward batches, so each batch RE-FETCHES it (LRU thrash — GH200 2026-06-29).

    The cursor walks latest→earliest, so at a batch's end the LRU-oldest entry is the *top*
    window — exactly what the next batch (shifted later by one batch-advance) needs first. With
    the cache barely large enough, the next batch's new top-window fetches evict the overlap
    right before it would be reused → ~zero reuse. Full reuse needs roughly one batch's met
    span PLUS one batch's advance. Setting the cache to *exactly* the span gives NO benefit;
    this guard catches that silent footgun at startup.

    The cache counts WINDOWS, not hours, so the batch spans are converted at the source
    cadence before the comparison: at 3-hourly met a 240 h span is 80 cache entries, and
    comparing 240 against the limit would demand a 3x larger cache than the run needs."""

    max_windows = cfg.memory.met_cache_max_hours
    if max_windows <= 0 or len(batches) < 2:
        return

    def _start(b) -> datetime:
        return max(r.release_time + timedelta(seconds=r.duration_seconds) for r in b.releases)

    def _terminus(b) -> datetime:
        earliest_end = min(
            r.release_time + timedelta(seconds=r.duration_seconds) for r in b.releases
        )
        return earliest_end - timedelta(seconds=sim_length_s)

    window_s = float(met_interval_seconds)
    span_windows = (_start(batches[0]) - _terminus(batches[0])).total_seconds() / window_s
    advance_windows = abs((_start(batches[1]) - _start(batches[0])).total_seconds()) / window_s
    overlap_windows = span_windows - advance_windows
    if overlap_windows <= 0:
        return  # batches don't overlap in met time → caching can't help, no thrash to warn about
    threshold = math.ceil(span_windows + advance_windows)
    if max_windows >= threshold:
        return
    LOGGER.warning(
        "met_cache_max_hours=%d (a count of met WINDOWS, %.0f h each) is below the cross-batch "
        "reuse threshold ~%d windows (batch met span ~%.0f + advance ~%.0f): consecutive "
        "batches will RE-FETCH the ~%.0f-window overlap (LRU thrash, ~%.1fx redundant met I/O). "
        "Raise met_cache_max_hours to >= %d to reuse it%s.",
        max_windows,
        window_s / 3600.0,
        threshold,
        span_windows,
        advance_windows,
        overlap_windows,
        span_windows / max(advance_windows, 1e-6),
        threshold,
        " (cached in host RAM since met_cache_on_host=True)"
        if cfg.memory.met_cache_on_host
        else "",
    )


class MetPrefetcher:
    """Serves hourly met windows, optionally prefetching the NEXT (backward) hour on a
    background thread so its I/O overlaps the current window's compute.

    The run is met-I/O-bound (~1.6 s/fetch, GPU otherwise idle), so while the caller works
    through hour H we prefetch H−1 (the cursor walks backward). When the caller reaches H−1
    the fetch is already done → no stall.

    Safety:
    * a SINGLE worker thread performs EVERY reader access, so the reader is never touched
      concurrently (no assumption about reader thread-safety);
    * it produces HOST tensors only — enabling prefetch requires ``met_cache_on_host`` so the
      worker never issues CUDA calls off the main thread;
    * the met cache is mutated only by the caller (main thread) inside ``get`` → no cache lock;
    * a fetch exception re-raises on the main thread via ``Future.result()``.

    Bit-identical to the synchronous path: same fetch for the same hour, same cache contents.
    Disabled (``enabled=False``) collapses to the old synchronous fetch on the main thread.
    """

    def __init__(
        self,
        reader: MetReader,
        cfg: RunConfig,
        met_cache: OrderedDict[datetime, HourlyMetTensors],
        *,
        enabled: bool,
    ) -> None:
        self._reader = reader
        self._cfg = cfg
        self._cache = met_cache
        self._max_hours = cfg.memory.met_cache_max_hours
        self._executor = (
            ThreadPoolExecutor(max_workers=1, thread_name_prefix="glide-met-prefetch")
            if enabled
            else None
        )
        self._pending_key: datetime | None = None
        self._pending_future: Future[HourlyMetTensors] | None = None

    def _fetch(self, hour_key: datetime) -> HourlyMetTensors:
        return self._reader.fetch_hourly_window(
            BoundingBoxRequest(
                spatial=_met_domain_bounds(self._cfg),
                time=TimeBounds(start=hour_key, end=hour_key + _met_interval(self._reader)),
            )
        )

    def _cache_put(self, hour_key: datetime, window: HourlyMetTensors) -> None:
        if self._max_hours > 0:
            self._cache[hour_key] = window
            while len(self._cache) > self._max_hours:
                _, evicted = self._cache.popitem(last=False)
                del evicted

    def _cancel_pending(self) -> None:
        if self._pending_future is not None:
            self._pending_future.cancel()
        self._pending_key = None
        self._pending_future = None

    def get(self, t_cursor: datetime) -> tuple[HourlyMetTensors, int]:
        """Return (window, fetched) for ``t_cursor``'s hour; ``fetched`` is 1 on a real read
        (cache miss), 0 on a cache hit — same accounting as the synchronous path."""

        hour_key = _met_window_start(self._reader, t_cursor)
        if self._max_hours > 0 and hour_key in self._cache:
            self._cache.move_to_end(hour_key)
            window, fetched = self._cache[hour_key], 0
        elif self._executor is None:
            window, fetched = self._fetch(hour_key), 1
            self._cache_put(hour_key, window)
        else:
            if self._pending_key == hour_key and self._pending_future is not None:
                fut, self._pending_key, self._pending_future = self._pending_future, None, None
            else:
                self._cancel_pending()  # stale (cursor jumped, e.g. batch boundary)
                fut = self._executor.submit(self._fetch, hour_key)
            window = fut.result()  # blocks iff not done yet; re-raises fetch errors
            self._cache_put(hour_key, window)
            fetched = 1
        self._schedule_prefetch(hour_key)
        return window, fetched

    def _schedule_prefetch(self, hour_key: datetime) -> None:
        if self._executor is None:
            return
        nxt = hour_key - _met_interval(self._reader)
        if (self._max_hours > 0 and nxt in self._cache) or self._pending_key == nxt:
            return
        self._cancel_pending()
        self._pending_key = nxt
        self._pending_future = self._executor.submit(self._fetch, nxt)

    def shutdown(self) -> None:
        self._cancel_pending()
        if self._executor is not None:
            self._executor.shutdown(wait=False, cancel_futures=True)
            self._executor = None


def _advection_alpha_wind_mean(
    met_window: HourlyMetTensors,
    t_cursor: datetime,
    delta_s: float,
    device: torch.device,
    dtype: torch.dtype,
    wind_mean_cache: dict | None = None,
) -> tuple[float, torch.Tensor]:
    """RK2-midpoint time-interpolation weight ``alpha`` + the grid-mean (u, v, w) wind.

    Shared by the dynamic path (`_advect_active_particles`) and the static path, where
    advection is folded into `HannaScheme._step_core` so `main._run` only needs these two
    scalars/diagnostics. ``wind_mean`` is a (3,) device tensor (no host sync — materialised
    once per batch for the trajectory diagnostics).

    The grid means are computed ONCE per met window and cached in ``wind_mean_cache``
    (single-slot, keyed by window start): the mean and the time interpolation are both
    linear, so ``mean(lerp(start, end)) == lerp(mean(start), mean(end))`` exactly. The
    per-step cost is a 3-element lerp on device. Without the cache this did a full
    [3, Z, Y, X] host->device copy + grid interpolation + reduction EVERY step, just
    for a 3-number diagnostic (perf review 2026-07-16 #2) — on the static path the
    scheme never needs main's device copy of the winds at all.
    """

    t_start_s = met_window.metadata.time_start.timestamp()
    t_end_s = met_window.metadata.time_end.timestamp()
    t_eval_s = t_cursor.timestamp() - 0.5 * delta_s
    alpha = max(0.0, min(1.0, (t_eval_s - t_start_s) / max(1.0, float(t_end_s - t_start_s))))

    ts = met_window.metadata.time_start
    cached = (
        wind_mean_cache if wind_mean_cache is not None and wind_mean_cache.get("ts") == ts else None
    )
    if cached is None:
        m_start_uvw, m_end_uvw = met_window.channels("u", "v", "w")
        # Reduce where the met lives (host when met_cache_on_host); ship only (3,).
        mean_start = m_start_uvw.reshape(3, -1).mean(dim=1).to(device=device, dtype=dtype)
        mean_end = m_end_uvw.reshape(3, -1).mean(dim=1).to(device=device, dtype=dtype)
        if wind_mean_cache is not None:
            wind_mean_cache.clear()
            wind_mean_cache["ts"] = ts
            wind_mean_cache["means"] = (mean_start, mean_end)
    else:
        mean_start, mean_end = cached["means"]

    wind_mean = mean_start * (1.0 - alpha) + mean_end * alpha
    return alpha, wind_mean


def _time_ago_index(elapsed_s: torch.Tensor, time_bin_s: float, n_time_bins: int) -> torch.Tensor:
    """Per-particle time-ago bin of an age (seconds since its release window's end).

    Single-bin footprint = time-integrated: lump ALL residence ages into bin 0.
    Multi-bin: leave out-of-range ages for the gridder to DROP (clamping them into
    the last bin would contradict time_ago_end_hours).
    """

    t_idx = (elapsed_s / time_bin_s).floor().to(torch.int64)
    if n_time_bins == 1:
        t_idx.clamp_(max=0)
    return t_idx


def _time_weight(met_window: HourlyMetTensors, t_eval_s: float) -> float:
    """Time-interpolation weight of the window end at ``t_eval_s`` (Unix seconds).

    The same formula as ``_advection_alpha_wind_mean``, which evaluates it at the
    step's midpoint; the trapezoid deposit evaluates it at each end of the step.
    """

    t_start_s = met_window.metadata.time_start.timestamp()
    t_end_s = met_window.metadata.time_end.timestamp()
    return max(0.0, min(1.0, (t_eval_s - t_start_s) / max(1.0, float(t_end_s - t_start_s))))


# Ticks per second of the exact age arithmetic in `_age_bin_pieces`: 2**20 (about
# 0.95 us). A power of two, so a float32 age converts to ticks by an exact scaling,
# and every float32 age >= 8 s (whose spacing is >= 2**-20 s) is a whole number of
# ticks. int64 ticks reach 2**43 s, far beyond any run.
_AGE_TICKS_PER_S = 1 << 20


def _seconds_to_ticks(seconds: float) -> int:
    """A host-side length (s) in `_AGE_TICKS_PER_S` ticks, to the nearest tick."""

    return round(seconds * _AGE_TICKS_PER_S)


def _age_bin_pieces(
    age_s: torch.Tensor,
    length_s: float,
    time_bin_s: float,
    n_time_bins: int,
    dtype: torch.dtype = torch.float32,
) -> Iterator[tuple[torch.Tensor, torch.Tensor | float]]:
    """Split each particle's age interval [age, age + length) by the time-ago bins.

    Returns ``(t_idx, seconds)`` pieces whose seconds sum to ``length_s`` for every
    particle: piece j is bin floor(age / width) + j, with the overlap of the
    interval with that bin. The number of pieces, ceil(length / width) + 1 (two
    whenever a bin is at least as wide as the interval), depends only on the step
    length and the configured width, never on the data, so the shapes are static
    and there is no host sync. Pieces that do not overlap carry 0 s (an exact
    no-op in the scatter). With one bin (time-integrated) every age is bin 0 and
    the interval is one piece. Pieces are yielded one at a time, so only one
    piece's temporaries are alive while it is scattered.

    The overlaps are computed exactly in int64 ticks of 2**-20 s
    (`_AGE_TICKS_PER_S`), never in float64, which Apple's MPS backend cannot
    allocate: the float32 age is scaled by 2**20 (exact) and rounded to int64,
    which is exact for every age >= 8 s and within 2**-21 s below; ``length_s``
    and ``time_bin_s`` are host floats rounded to the nearest tick (exact for the
    whole and half seconds the config produces). Each piece's seconds are its
    ticks converted to ``dtype`` (the grid's) and scaled by 2**-20, which is one
    correctly rounded conversion of the exact overlap, so they equal the float64
    computation rounded to ``dtype`` whenever that computation was exact.
    """

    if n_time_bins == 1:
        yield _time_ago_index(age_s, time_bin_s, 1), length_s
        return
    width = _seconds_to_ticks(time_bin_s)
    length = _seconds_to_ticks(length_s)
    lo = torch.round(age_s * float(_AGE_TICKS_PER_S)).to(torch.int64)
    hi = lo + length
    first = torch.div(lo, width, rounding_mode="floor")
    for j in range(-(-length // width) + 1):
        bin_lo = (first + j) * width
        overlap = (torch.minimum(hi, bin_lo + width) - torch.maximum(lo, bin_lo)).clamp_(min=0)
        yield first + j, overlap.to(dtype) * (1.0 / _AGE_TICKS_PER_S)


def _deposit_step(
    gridder: FootprintGridder,
    *,
    position: str,
    start_xyz: torch.Tensor | None,
    end_xyz: torch.Tensor,
    active_mask: torch.Tensor,
    weights: torch.Tensor,
    elapsed_s: torch.Tensor,
    t_idx: torch.Tensor,
    time_bin_s: float,
    release_idx: torch.Tensor,
    delta_s: float,
    t_cursor: datetime,
    particle_density: ParticleDryAirDensity | None,
    met_window: HourlyMetTensors,
    t_alpha: float,
    engine: GPUEngine,
) -> None:
    """Count one backward step's residence, (t_cursor - delta_s, t_cursor], in the grid
    (task A2, `output_grid.deposition.position`; docs/physics.md §7).

    ``elapsed_s`` is each particle's age e at the step's start (``t_cursor``), counted
    from its own release window's end, and ``t_idx`` its time-ago bin. The step
    covers the ages [e, e + delta_s). Inactive rows are never counted (the gridder
    masks them, and escaped particles are no longer active).

    * ``endpoint``: all of ``delta_s`` at ``end_xyz``, in bin ``t_idx``, with the
      density at ``end_xyz`` and the step's time weight ``t_alpha``. A step that
      straddles a bin edge is counted wholly in the bin of its starting age (the
      long-standing behaviour, kept).
    * ``trapezoid``: ``delta_s / 2`` at ``start_xyz``, standing for the ages
      [e, e + delta_s/2), and ``delta_s / 2`` at ``end_xyz``, for
      [e + delta_s/2, e + delta_s). Each half is split by its overlap with the
      time-ago bins (`_age_bin_pieces`), each piece keeping that half's position and
      density, so a stationary particle's residence per bin is exact whatever the
      alignment of release windows, steps and bins. With dry-air weighting each
      half divides by the density at its own position and time (the start at
      ``t_cursor``, the end at ``t_cursor - delta_s``), in the step's met window. A
      particle's first step is counted half at its release position; a particle
      that leaves the met domain in a step keeps the start half (the end half is
      dropped by the gridder if outside the output grid, as endpoint's is).

    Positions (both halves) are in ``engine.frame``: the gridder converts them to
    degrees at entry and the density is sampled with grid bounds in the same frame,
    so the gridder must have been built with the engine's frame.
    """

    if gridder.frame != engine.frame:
        raise ValueError(f"gridder frame {gridder.frame} differs from the engine's {engine.frame}")
    if position == "trapezoid":
        half_s = 0.5 * delta_s
        t_start_s = t_cursor.timestamp()
        halves = (
            (start_xyz, elapsed_s, t_start_s),
            (end_xyz, elapsed_s + half_s, t_start_s - delta_s),
        )
        for xyz, age_s, t_eval_s in halves:
            inv_density = (
                None
                if particle_density is None
                else particle_density.inverse_density(
                    xyz, met_window, _time_weight(met_window, t_eval_s), engine
                )
            )
            pieces = _age_bin_pieces(age_s, half_s, time_bin_s, gridder.n_t, gridder.dtype)
            for piece_idx, piece_s in pieces:
                gridder.accumulate(
                    particles=xyz,
                    active_mask=active_mask,
                    weights=weights,
                    t_idx=piece_idx,
                    release_idx=release_idx,
                    dt_seconds=piece_s,
                    inv_density=inv_density,
                )
        return

    deposit_xyz = end_xyz
    inv_density = (
        None
        if particle_density is None
        else particle_density.inverse_density(deposit_xyz, met_window, t_alpha, engine)
    )
    gridder.accumulate(
        particles=deposit_xyz,
        active_mask=active_mask,
        weights=weights,
        t_idx=t_idx,
        release_idx=release_idx,
        dt_seconds=delta_s,
        inv_density=inv_density,
    )


def _advect_active_particles(
    engine: GPUEngine,
    device: torch.device,
    active_particles: torch.Tensor,
    met_window: HourlyMetTensors,
    t_cursor: datetime,
    delta_s: float,
    dtype: torch.dtype,
    wind_mean_cache: dict | None = None,
) -> tuple[torch.Tensor, float, torch.Tensor]:
    """RK2 backward advection on the active particle subset (dynamic / CPU path).

    The third return is the grid-mean (u, v, w) wind as a device tensor — NOT
    materialised to floats here, so the trajectory diagnostics can be transferred
    to the host once per batch rather than syncing every step (matters on CUDA).
    """

    alpha, diag_wind_mean = _advection_alpha_wind_mean(
        met_window, t_cursor, delta_s, device, dtype, wind_mean_cache
    )

    # Grid bounds in the engine's horizontal frame (`lpdm.local_frame`), with the
    # per-level AGL array for the piecewise-linear vertical lookup (F9).
    frame = engine.frame
    grid_bounds = frame_grid_bounds(met_window.metadata, frame)

    # The wind stacks in `simulation.wind_gather_dtype` (task A4): float32 is
    # sampled with grid_sample as always, a 16-bit type with `trilinear_sample`.
    wind_dtype = engine.wind_dtype
    sample_16bit = wind_dtype.itemsize == 2
    m_start_uvw, m_end_uvw = met_window.channels("u", "v", "w")
    m_start = m_start_uvw.unsqueeze(0).to(device=device, dtype=wind_dtype)
    m_end = m_end_uvw.unsqueeze(0).to(device=device, dtype=wind_dtype)

    def wind_fn(xyz: torch.Tensor) -> torch.Tensor:
        xyz_norm = engine.normalize_particle_coordinates(xyz, grid_bounds)

        # m_start/m_end are closed over from the enclosing scope; the `del` below
        # (a memory optimisation) confuses ruff's unbound analysis -> noqa: F821.
        if sample_16bit:
            s, e = trilinear_sample((m_start[0], m_end[0]), xyz_norm)  # noqa: F821
            v_start, v_end = s.t(), e.t()
        else:
            grid = xyz_norm.view(1, 1, 1, -1, 3)
            sampled = [
                torch.nn.functional.grid_sample(m, grid, align_corners=True).view(3, -1).t()
                for m in (m_start, m_end)  # noqa: F821
            ]
            v_start, v_end = sampled
        v_interp = v_start * (1.0 - alpha) + v_end * alpha

        # Metres per second to frame units per second at the particle's latitude.
        lat_deg = frame.lat_from_y(xyz[:, 1])
        out = torch.empty_like(xyz)
        out[:, 0] = v_interp[:, 0] / frame.metres_per_x(lat_deg)
        out[:, 1] = v_interp[:, 1] / frame.metres_per_y
        out[:, 2] = v_interp[:, 2]
        return out

    advected_active = engine.rk2_advect_backward(
        active_particles, dt_seconds=delta_s, wind_fn=wind_fn
    )

    del m_start
    del m_end

    return advected_active, alpha, diag_wind_mean


def _scheme_kwargs(cfg: RunConfig) -> dict[str, object]:
    """Constructor kwargs for the configured turbulence scheme.

    Meander config is Hanna-specific; other schemes take no kwargs and would
    reject it, so it is only forwarded for ``hanna_1982``.
    """

    if cfg.turbulence.scheme != "hanna_1982":
        return {}
    m = cfg.turbulence.meander
    kwargs: dict[str, object] = {
        "meander_enabled": m.enabled,
        "meander_coefficient": m.coefficient,
        "meander_stencil_radius": m.stencil_radius,
        "n_substeps": cfg.turbulence.n_substeps,
        "flexpart_tl_floors": cfg.turbulence.flexpart_tl_floors,
        "surface_layer_override": cfg.turbulence.surface_layer_override,
        "horizontal_frame": cfg.turbulence.horizontal_frame,
        "blh_min_m": cfg.turbulence.boundary_layer_depth_min_m,
        "blh_max_m": cfg.turbulence.boundary_layer_depth_max_m,
    }
    # Unset => the scheme takes half the met interval, bound in `_run` once the
    # reader has measured the store (decision 0024).
    if m.timescale_seconds is not None:
        kwargs["meander_timescale_seconds"] = m.timescale_seconds
    return kwargs


def _convection_scheme_kwargs(cfg: RunConfig) -> dict[str, object]:
    """Constructor kwargs for the configured convection scheme.

    The Emanuel tuning block is scheme-specific; other schemes take no kwargs
    (``NoConvection`` is parameter-less).
    """

    if cfg.convection.scheme != "emanuel_reduced":
        return {}
    e = cfg.convection.emanuel
    return {
        "closure_c": e.closure_c,
        "trigger_dtv_k": e.trigger_dtv_k,
        "min_cape_j_kg": e.min_cape_j_kg,
        "min_cloud_depth_m": e.min_cloud_depth_m,
    }


def _compile_count() -> int:
    """Number of torch.compile graph compilations so far (-1 if the counter is unavailable).

    Grows by one per (re)compilation. If it climbs at every batch boundary, the compiled
    core is being RE-compiled per batch — each recompile records a NEW CUDA graph whose
    memory the caching allocator's ``empty_cache`` cannot reclaim, so it accumulates toward
    OOM. The usual culprit is a per-batch-varying Python scalar reaching the core (the
    last-step clamped ``dt_seconds`` — same class as the ``alpha``/``level`` fixes)."""

    try:
        from torch._dynamo.utils import counters

        return int(counters.get("frames", {}).get("ok", 0))
    except Exception:
        return -1


def _log_device_memory(label: str, device: torch.device) -> None:
    """Log CUDA allocated/reserved/peak memory + compile count at a coarse boundary (no-op
    off CUDA).

    Used at batch boundaries to localise the per-batch GPU-memory growth that risks OOM-ing
    a long multi-batch run (2026-06-29). `allocated` is live tensors; `reserved` is the
    caching allocator's pool; `peak` is the in-batch high-water mark (reset each batch);
    `compiles` is the cumulative torch.compile count (climbing per batch = recompilation)."""

    if device.type != "cuda":
        return
    gib = 2**30
    LOGGER.info(
        "device mem [%s]: alloc=%.2f GiB reserved=%.2f GiB peak=%.2f GiB compiles=%d",
        label,
        torch.cuda.memory_allocated(device) / gib,
        torch.cuda.memory_reserved(device) / gib,
        torch.cuda.max_memory_allocated(device) / gib,
        _compile_count(),
    )


def _start_mem_history(device: torch.device) -> bool:
    """Begin recording the CUDA allocation history when ``GLIDE_MEM_SNAPSHOT`` is set, so we
    can localise the per-batch device-memory growth to its exact allocation call stacks
    (2026-06-29). Returns whether recording is active. Off by default (it has overhead)."""

    if device.type != "cuda" or os.environ.get("GLIDE_MEM_SNAPSHOT", "") in (
        "",
        "0",
        "false",
        "False",
    ):
        return False
    try:
        torch.cuda.memory._record_memory_history(max_entries=200_000)
        LOGGER.info(
            "mem snapshot: recording allocation history; will dump after batch %s",
            os.environ.get("GLIDE_MEM_SNAPSHOT_BATCH", "1"),
        )
        return True
    except Exception as exc:  # pragma: no cover - diagnostic only
        LOGGER.warning("mem snapshot: could not start recording: %s", exc)
        return False


def _maybe_dump_mem_snapshot(batch_idx: int, device: torch.device) -> None:
    """After the configured batch, dump a CUDA allocation snapshot (open at
    https://pytorch.org/memory_viz — it shows each live block's allocation stack) plus a
    pasteable allocator summary, then stop recording. No-op unless armed for this batch."""

    if device.type != "cuda" or os.environ.get("GLIDE_MEM_SNAPSHOT", "") in (
        "",
        "0",
        "false",
        "False",
    ):
        return
    if batch_idx != int(os.environ.get("GLIDE_MEM_SNAPSHOT_BATCH", "1")):
        return
    path = os.environ.get("GLIDE_MEM_SNAPSHOT_PATH", "glide_mem_snapshot.pickle")
    try:
        torch.cuda.memory._dump_snapshot(path)
        LOGGER.info("mem snapshot: wrote %s — open at https://pytorch.org/memory_viz", path)
        LOGGER.info(
            "mem summary after batch %d:\n%s",
            batch_idx,
            torch.cuda.memory_summary(device, abbreviated=True),
        )
        torch.cuda.memory._record_memory_history(enabled=None)
    except Exception as exc:  # pragma: no cover - diagnostic only
        LOGGER.warning("mem snapshot: dump failed: %s", exc)


@torch.no_grad()
def _seed_global_generators(cfg: RunConfig) -> int | None:
    """Seed torch's default generators (CPU and every CUDA device) from the
    release seed, so a run with a seed is reproducible (decision 0021).

    The release seed already fixes the particles' starting positions, through
    its own generator. The turbulence and convection noise is drawn from torch's
    default generators, inside the captured CUDA graph, where only the default
    generators are safe to use; seeding them here, once per run before any
    particle moves, makes that noise reproducible too. With no seed, nothing is
    seeded and runs differ, as before.
    """

    seed = getattr(cfg.release, "seed", None)
    if seed is None:
        LOGGER.info("rng: no release seed; turbulence noise is not reproducible")
        return None
    torch.manual_seed(int(seed))  # seeds the CPU and all CUDA default generators
    LOGGER.info("rng: torch default generators seeded with %d", int(seed))
    return int(seed)


# The turbulence sub-step length at which the well-mixed tests are run: the default
# count at the shipped 60 s outer step (decisions 0027 and 0028).
VALIDATED_SUBSTEP_SECONDS = 60.0 / N_SUBSTEPS_DEFAULT


def _log_substep_length(scheme: TurbulenceScheme, dt_seconds: float) -> float | None:
    """Log the turbulence sub-step length, ``dt_seconds / n_substeps``.

    ``turbulence.n_substeps`` is a count, so the sub-step length follows the
    outer step: doubling ``dt_seconds`` doubles it. Warn when it is longer than
    the sub-step the well-mixed tests are run at. Returns the length, or None for
    a scheme without sub-steps.
    """

    n = getattr(scheme, "n_substeps", None)
    if not n:
        return None
    sub = float(dt_seconds) / int(n)
    LOGGER.info(
        "turbulence: %d sub-steps of %.4g s per %.4g s step (turbulence.n_substeps is a "
        "count; the sub-step length follows simulation.dt_seconds)",
        int(n),
        sub,
        float(dt_seconds),
    )
    if sub > VALIDATED_SUBSTEP_SECONDS * (1.0 + 1e-9):
        LOGGER.warning(
            "turbulence: the %.4g s sub-step (dt_seconds %.4g / n_substeps %d) is longer than "
            "the %.4g s at which the near-surface and boundary-layer-top well-mixed tests are "
            "run (decisions 0027, 0028); raise turbulence.n_substeps to at least %d to keep it.",
            sub,
            float(dt_seconds),
            int(n),
            VALIDATED_SUBSTEP_SECONDS,
            math.ceil(float(dt_seconds) / VALIDATED_SUBSTEP_SECONDS - 1e-9),
        )
    return sub


def _physics_warnings(scheme: TurbulenceScheme) -> list[str]:
    """Log, once per run, the configured physics options that are not well-mixed
    consistent, and return their messages for ``runtime.physics_warnings`` in
    run_metadata.json (empty when there are none).

    The opt-in surface-layer override leaves a sigma_w jump at 0.1 h that the drift
    straddles (decision 0028, "Other boundaries"; docs/turbulence.md §8).
    """

    warnings: list[str] = []
    if getattr(scheme, "surface_layer_override", False):
        warnings.append(SURFACE_LAYER_OVERRIDE_WARNING)
    for message in warnings:
        LOGGER.warning(message)
    return warnings


class _ExceedanceTally:
    """Run totals of the turbulence sub-steps that are long compared with T_Lw.

    The scheme counts, on the device and inside its step (on the graph path inside
    the captured step, with no host sync), the active particle-sub-steps, those
    with ``sub_dt > exceedance_c · T_Lw`` and those with ``sub_dt > T_Lw``, split
    boundary layer / free troposphere (decision 0027). ``poll`` reads them once per
    met window, when the cursor enters a new window, and ``flush`` at the end of
    each batch, so the run pays one host sync per window. A scheme without
    sub-steps returns None and nothing is recorded.
    """

    _LAYERS = (("boundary_layer", "bl"), ("free_troposphere", "ft"))
    _COUNTS = ("substeps", "exceeded", "over_T_L")

    def __init__(self, scheme: TurbulenceScheme, dt_seconds: float | None = None) -> None:
        self.scheme = scheme
        self.dt_seconds = dt_seconds
        self.supported = True
        self.window: datetime | None = None
        self.totals = {f"{c}_{s}": 0 for _, s in self._LAYERS for c in self._COUNTS}
        self.worst = {s: 0.0 for _, s in self._LAYERS}
        self.windows_exceeded = 0

    def poll(self, window_start: datetime) -> None:
        """Read the counts of the window just left, if the cursor has left one."""
        if not self.supported or window_start == self.window:
            return
        previous, self.window = self.window, window_start
        if previous is not None:
            self._take(previous)

    def flush(self) -> None:
        """Read the counts of the current window (end of a batch)."""
        if self.supported and self.window is not None:
            self._take(self.window)
        self.window = None

    def _take(self, window_start: datetime) -> None:
        taken = self.scheme.take_substep_exceedance(window_start)
        if taken is None:
            self.supported = False
            return
        for k in self.totals:
            self.totals[k] += int(taken[k])
        for _, s in self._LAYERS:
            self.worst[s] = max(self.worst[s], float(taken[f"worst_{s}"]))
        if taken["exceeded_bl"] or taken["exceeded_ft"]:
            self.windows_exceeded += 1

    def metadata(self) -> dict[str, object] | None:
        if not self.supported:
            return None
        t = self.totals

        def frac(num: int, den: int) -> float:
            return num / den if den else 0.0

        meta: dict[str, object] = {
            "n_substeps": getattr(self.scheme, "n_substeps", None),
            "substep_seconds": (
                self.dt_seconds / self.scheme.n_substeps
                if self.dt_seconds and getattr(self.scheme, "n_substeps", None)
                else None
            ),
            "exceedance_c": getattr(self.scheme, "exceedance_c", None),
            "criterion": (
                "exceeded: sub_dt > exceedance_c * T_Lw; over_T_L: sub_dt > T_Lw; "
                "T_Lw at the sub-step's starting height; boundary layer z <= h"
            ),
            "unit": "active particle-sub-steps",
        }
        for name, s in self._LAYERS:
            n = t[f"substeps_{s}"]
            meta[name] = {
                "substeps": n,
                "exceeded": t[f"exceeded_{s}"],
                "exceeded_fraction": frac(t[f"exceeded_{s}"], n),
                "over_T_L": t[f"over_T_L_{s}"],
                "over_T_L_fraction": frac(t[f"over_T_L_{s}"], n),
                "worst_sub_dt_over_T_Lw": self.worst[s],
            }
        meta["met_windows_with_exceedance"] = self.windows_exceeded
        return meta

    def log_run_total(self) -> None:
        meta = self.metadata()
        if meta is None:
            return
        bl, ft = meta["boundary_layer"], meta["free_troposphere"]
        LOGGER.info(
            "sub-step exceedance (sub_dt > %s T_Lw, n_substeps=%s): boundary layer %d of %d "
            "(%.3g%%, worst %.2f); free troposphere %d of %d (%.3g%%, worst %.2f; "
            "sub_dt > T_Lw in %.3g%%); %d met window(s).",
            meta["exceedance_c"],
            meta["n_substeps"],
            bl["exceeded"],
            bl["substeps"],
            100.0 * bl["exceeded_fraction"],
            bl["worst_sub_dt_over_T_Lw"],
            ft["exceeded"],
            ft["substeps"],
            100.0 * ft["exceeded_fraction"],
            ft["worst_sub_dt_over_T_Lw"],
            100.0 * ft["over_T_L_fraction"],
            meta["met_windows_with_exceedance"],
        )


def _run(
    cfg: RunConfig,
    *,
    reader: MetReader | None = None,
    scheme: TurbulenceScheme | None = None,
    convection_scheme: ConvectionScheme | None = None,
) -> dict[str, object]:
    # Coarse whole-run phase accountant (GLIDE_PHASE_TIMERS; lpdm.phase_timers). It spans
    # every window and batch, so it catches per-window met-fetch idle, and it is installed
    # for the schemes to time their own parts. A shared no-op when disabled.
    phase_timer = make_phase_timer()
    install_phase_timer(phase_timer)
    t_setup_start = time.perf_counter()

    sim = cfg.simulation
    mem = cfg.memory
    out_grid = cfg.output_grid
    device_str = _resolve_device(sim.device)
    _check_device_dtypes(cfg, device_str)

    if scheme is None:
        scheme = get_scheme(cfg.turbulence.scheme, **_scheme_kwargs(cfg))
    _log_substep_length(scheme, sim.dt_seconds)
    physics_warnings = _physics_warnings(scheme)
    if convection_scheme is None:
        convection_scheme = get_convection_scheme(
            cfg.convection.scheme,
            **_convection_scheme_kwargs(cfg),
        )
    if reader is None:
        # Channels = baseline (advection needs u/v/w; runtime telemetry uses blh/sp)
        # unioned with whatever the turbulence + convection schemes declare they
        # need (e.g. Hanna adds ustar/shf/t; Emanuel adds q).
        required_channels = tuple(
            dict.fromkeys(
                (
                    "u",
                    "v",
                    "w",
                    "blh",
                    "sp",
                    *scheme.required_met_keys(),
                    *convection_scheme.required_met_keys(),
                    *_density_weighting_met_keys(cfg),
                )
            )
        )
        # Cache met windows on the HOST when requested (default): each window is a
        # [C,Z,Y,X] stack (~hundreds of MiB), so a large met cache on the GPU consumes
        # GiBs of HBM (snapshot 2026-06-29: 192 h ≈ 50 GiB). The per-step physics moves the
        # active window to the device itself (its `.to(device)` calls), so the reader can
        # stage in host RAM and keep the device free for compute. CPU runs: host == device.
        reader_device = "cpu" if cfg.memory.met_cache_on_host else device_str
        reader = ArcoEra5ZarrReader(
            zarr_store=cfg.io.zarr_store,
            channel_names=required_channels,
            device=reader_device,
            # Unset met_domain.vertical_levels => 40 stretched levels (decision 0017).
            agl_levels_m=cfg.met_domain.resolve_agl_levels(),
        )
        # The AGL grid sets the vertical resolution the physics actually sees (and the
        # met cache size), so record it — it is not otherwise recoverable from the log.
        _agl = reader.agl_levels_m
        if _agl is None:
            _agl = default_agl_levels(cfg.met_domain.alt_max_m)
            _origin = "built-in default"
        else:
            _origin = "met_domain.vertical_levels"
        LOGGER.info(
            "met vertical grid: %d AGL levels to %.0f m (%s); %d below 1.5 km, lowest %.1f m",
            _agl.size,
            _agl[-1],
            _origin,
            int((_agl <= 1500.0).sum()),
            float(_agl[1]) if _agl.size > 1 else 0.0,
        )
    run_seed = _seed_global_generators(cfg)
    # Particle positions in metres from the met domain's centre (task A4,
    # `lpdm.local_frame`), and the dtype of the wind stacks in the gather.
    position_frame = _position_frame(cfg)
    engine = GPUEngine(
        device=device_str,
        frame=position_frame,
        wind_gather_dtype=getattr(torch, cfg.simulation.wind_gather_dtype),
    )
    LOGGER.info(
        "particle positions: %s frame (lon0 %.4f, lat0 %.4f, %.3f x-units and %.3f "
        "y-units per degree); wind gather dtype %s",
        position_frame.kind,
        position_frame.lon0,
        position_frame.lat0,
        position_frame.kx,
        position_frame.ky,
        cfg.simulation.wind_gather_dtype,
    )
    writer = OutputWriter()
    device = torch.device(device_str)

    # Per-step execution path (docs/architecture.md §5). On the static path the cursor
    # loop processes the FULL particle buffer every step and gates inactive
    # particles with `torch.where` (no boolean indexing, no per-step host sync) —
    # the shapes/control-flow CUDA-graph capture needs, and a win on a launch-bound
    # GPU. On the dynamic path it boolean-indexes the active subset (cheaper on
    # CPU). Auto: static on CUDA, dynamic elsewhere; `GLIDE_STATIC_SUBSTEPS`
    # overrides (used to exercise the static path on CPU in tests).
    use_static = use_static_step_path(device)
    # Dry-air density at each counted particle (decision 0022), or None for
    # plain residence seconds.
    particle_density: ParticleDryAirDensity | None = None
    if cfg.output_grid.density_weighting == "dry_air":
        reader_channels = getattr(reader, "channel_names", None)
        if reader_channels is not None:
            missing = [k for k in ParticleDryAirDensity.CHANNELS if k not in reader_channels]
            if missing:
                raise PreflightValidationError(
                    "output_grid.density_weighting 'dry_air' needs met channels "
                    f"{list(ParticleDryAirDensity.CHANNELS)}; the reader lacks {missing}. "
                    "Use density_weighting: none for plain residence seconds."
                )
        particle_density = ParticleDryAirDensity(
            device,
            dtype=torch.float32,
            # Same switch as the scheme's whole-step compile (GLIDE_COMPILE).
            compile_core=use_static and bool(getattr(engine, "_compile_requested", False)),
        )
    LOGGER.info("footprint density weighting: %s", cfg.output_grid.density_weighting)
    # Where each step's residence is counted (task A2, decision 0029): the endpoint
    # (default) or half at each end of the step (`_deposit_step`). The runtime keeps
    # the start, outside the captured step, so the step's graph and its outputs are
    # the same for both.
    deposition_position = cfg.output_grid.deposition.position
    accumulation_dtype = (
        torch.float64 if cfg.output_grid.accumulation_dtype == "float64" else torch.float32
    )
    LOGGER.info("footprint accumulation dtype: %s", cfg.output_grid.accumulation_dtype)
    keep_step_start = deposition_position != "endpoint"
    LOGGER.info("footprint deposition position: %s", cfg.output_grid.deposition.position)
    if use_static:
        LOGGER.info("runtime: static-shape per-step path (full-set, mask-gated)")
    _start_mem_history(device)

    # Optional per-step profiler (GLIDE_PROFILE): captures a window of cursor-loop
    # steps to localise where the per-step time goes (GPU kernels vs host syncs vs
    # CPU/met). None unless GLIDE_PROFILE is set, so normal runs pay nothing.
    step_profiler = _make_step_profiler(device)

    # M5 stage 5: expand the schedule into batches and validate met coverage over
    # the full schedule. Each batch is integrated independently and its footprint
    # slice is streamed to the output Zarr region (see the batch loop below), so
    # only one batch's footprint tensor is resident at a time.
    batches = cfg.expand_to_batches()
    all_releases: list[ConcreteRelease] = [r for b in batches for r in b.releases]
    total_releases = len(all_releases)
    all_window_ends = [
        rel.release_time + timedelta(seconds=rel.duration_seconds) for rel in all_releases
    ]
    schedule_release_end = max(all_window_ends)
    # Every particle is followed for sim.length_seconds from its OWN release time,
    # and the earliest is released at the start of the earliest release window.
    schedule_sim_start = min(rel.release_time for rel in all_releases) - timedelta(
        seconds=sim.length_seconds
    )
    _validate_meteorology_time_coverage(reader, cfg, schedule_release_end, schedule_sim_start)
    _validate_meteorology_space_coverage(reader, cfg)

    # Source cadence drives the met bracketing, the cache key and the convection
    # cadence, so log what was actually measured: it is not recoverable from the
    # output otherwise, and it changes how much wall time a run spends on met I/O.
    met_interval_s = reader.met_interval_seconds
    LOGGER.info("met cadence: %.0f s (%.2f h) per window", met_interval_s, met_interval_s / 3600.0)
    # Settings that follow the met interval are fixed here, once, before any step
    # (the meander timescale, when unset; decision 0024).
    scheme.bind_met_interval(met_interval_s)
    if getattr(scheme, "meander_enabled", False):
        LOGGER.info(
            "meander: timescale %.0f s (%s)",
            scheme.meander_timescale_seconds,
            "configured"
            if cfg.turbulence.meander.timescale_seconds is not None
            else "half the met interval",
        )
    if met_interval_s > 3600:
        LOGGER.warning(
            "met cadence is %.1f h. GLIDE interpolates the wind linearly across a window, so "
            "sub-window features (the diurnal PBL cycle, frontal passages) are smoothed out; "
            "hourly met is what the physics is tuned against.",
            met_interval_s / 3600.0,
        )

    LOGGER.info(
        "schedule expanded: %d release(s) across %d batch(es), max %d per batch",
        total_releases,
        len(batches),
        cfg.batch.max_releases_per_batch,
    )
    _warn_if_met_cache_thrashes(cfg, batches, float(sim.length_seconds), met_interval_s)

    # Bookkeeping shared across batches.
    diag_rows: list[dict[str, float | int | str]] = []
    step_count = 0
    hour_windows = 0
    met_cache: OrderedDict[datetime, HourlyMetTensors] = OrderedDict()
    # Background prefetch of the next backward met hour (overlaps fetch I/O with compute).
    # Requires a HOST-resident reader (the worker must not issue CUDA calls); disabled with a
    # warning otherwise. No-op-equivalent to the synchronous path when disabled.
    prefetch_enabled = cfg.memory.met_prefetch and cfg.memory.met_cache_on_host
    if cfg.memory.met_prefetch and not cfg.memory.met_cache_on_host:
        LOGGER.warning("met prefetch disabled: requires memory.met_cache_on_host (CPU reader)")
    if prefetch_enabled:
        LOGGER.info("met prefetch: enabled (1 background worker, next backward hour)")
    prefetcher = MetPrefetcher(reader, cfg, met_cache, enabled=prefetch_enabled)
    memory_stats = MemoryStats()
    # Endpoint particles + their release_idx accumulated across batches, concatenated
    # once at the end so we emit one parquet covering the whole schedule.
    endpoint_particles_chunks: list[torch.Tensor] = []
    endpoint_release_idx_chunks: list[torch.Tensor] = []

    outputs = _build_output_paths(cfg.io.output_uri)
    sim_length_s = float(sim.length_seconds)
    # Time-ago bin width (decision 0016): fixed for the run, so a constant to the
    # compiled step, never a per-step value (decision 0004).
    time_bin_s = float(cfg.output_grid.time_bin_seconds)
    # Total particles killed for leaving met_domain across the whole run.
    run_escaped_total = 0

    # Streaming footprint output: instead of holding the full
    # (total_releases, T, Z, Y, X) tensor in memory for the whole run, each batch
    # accumulates into a gridder sized for that batch alone, writes its slice to
    # the output Zarr region, then frees the gridder. Peak footprint memory is
    # therefore one batch's worth, not the whole schedule's. The store is created
    # lazily on the first batch (we need a gridder's geometry to build coords).
    footprint_store_created = False

    # Single-slot per-window cache for the grid-mean wind diagnostic (perf review
    # 2026-07-16 #2): without it every step paid a full [3, Z, Y, X] host->device
    # copy + reduction just to log a (3,) mean.
    wind_mean_cache: dict = {}
    # Sub-step exceedance counter (decision 0027): counted on the device inside the
    # step, read here once per met window, recorded in run_metadata.json.
    exceedance = _ExceedanceTally(scheme, sim.dt_seconds)

    phase_timer.add("setup", time.perf_counter() - t_setup_start)
    for batch in batches:
        batch_setup_phase = phase_timer.start("batch_setup")
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        _log_device_memory(f"batch {batch.batch_idx} start", device)
        with phase_timer.phase("particle_generation"):
            particle_batch = generate_batch_particles(batch, device=device_str)
        # Released at (lon, lat); held in the engine's frame from here on.
        particles = engine.frame.to_frame(particle_batch.particles)
        # release_idx from the batch is global; remap to batch-local for the
        # per-batch gridder, then offset back to global for the region write.
        batch_release_offset = batch.releases[0].release_idx
        release_idx_local = particle_batch.release_idx - batch_release_offset
        release_time_offsets_s = particle_batch.release_time_offsets_s
        release_window_end_offsets_s = particle_batch.release_window_end_offsets_s
        batch_start_ts = particle_batch.batch_start_time.timestamp()

        gridder = FootprintGridder(
            lon_bounds=tuple(out_grid.lon_bounds),
            lat_bounds=tuple(out_grid.lat_bounds),
            z_edges_m=out_grid.z_edges_m,
            n_time_bins=_footprint_time_bin_count(cfg, all_releases),
            n_y=out_grid.n_y,
            n_x=out_grid.n_x,
            n_releases=len(batch.releases),
            device=device,
            dtype=accumulation_dtype,
            frame=engine.frame,
        )

        if not footprint_store_created:
            # Geometry is batch-independent, so use this batch's gridder to build
            # the full-schedule coords/attrs, then create the empty store sized for
            # all releases. Subsequent batches just write their region.
            footprint_coords, footprint_attrs = _build_footprint_dataset_metadata(
                cfg, gridder, all_releases
            )
            with phase_timer.phase("output_write"):
                writer.create_footprint_store(
                    outputs.footprints,
                    shape=(total_releases, gridder.n_t, gridder.n_z, gridder.n_y, gridder.n_x),
                    coords=footprint_coords,
                    attrs=footprint_attrs,
                    keep_mantissa_bits=out_grid.footprint_keep_mantissa_bits,
                    zstd_level=out_grid.footprint_zstd_level,
                )
            footprint_store_created = True

        # Per-batch cursor bounds: walk from the latest window-end down to the
        # earliest particle release time minus sim.length_seconds. Each particle is
        # active for sim.length_seconds from its own release time (`active_mask`), and
        # the earliest is released at the START of the earliest release window, so
        # stopping one length before a window's END cut short the particles released
        # early in it, by up to the release duration, and made a release's lifetime
        # depend on which releases shared its batch (VALIDATION-HANDOFF finding 1).
        batch_window_ends = [
            rel.release_time + timedelta(seconds=rel.duration_seconds) for rel in batch.releases
        ]
        batch_cursor_start = max(batch_window_ends)
        batch_cursor_terminus = min(rel.release_time for rel in batch.releases) - timedelta(
            seconds=sim_length_s
        )

        # Turbulence state is re-initialised per batch because the particle count
        # varies between batches (last batch may be smaller than the others).
        turbulence_state = scheme.initialize_state(
            particles.shape[0], device=device, dtype=particles.dtype
        )
        convection_state = convection_scheme.initialize_state(
            particles.shape[0], device=device, dtype=particles.dtype
        )
        # Track the met bracket start time so we fire convection exactly once
        # per met update (i.e. when the cursor crosses an hour boundary into a
        # new bracket). `None` = nothing fired yet.
        last_convection_bracket_start: datetime | None = None

        # Persistent per-batch liveness mask. A particle is killed (drop-and-count)
        # the step it leaves met_domain — it can never re-enter with valid met, so
        # stepping it further wastes compute. `active_mask` ANDs this with the
        # release-time window, so the active set shrinks as particles escape.
        alive = torch.ones(particles.shape[0], dtype=torch.bool, device=device)
        batch_escaped_total = 0

        # Per-step trajectory diagnostics are accumulated as device tensors here
        # and transferred to the host ONCE at batch end (`_materialize_*` below),
        # instead of a `.item()` sync every step. On CUDA those per-step syncs
        # serialise the pipeline; on CPU it's equivalent. The host-side bookkeeping
        # (step index, time, active count) is collected alongside.
        diag_pos_means: list[torch.Tensor] = []  # each (3,): mean x/y/alt over all particles
        diag_wind_means: list[
            torch.Tensor
        ] = []  # each (3,): grid-mean u/v/w (0 when no active set)
        diag_escaped: list[torch.Tensor] = []  # each (): newly-escaped count this step
        diag_active: list[
            torch.Tensor
        ] = []  # each (): active-particle count this step (device, no sync)
        diag_host: list[tuple[int, int, str]] = []  # (step, batch_idx, iso_time)
        zero_wind = torch.zeros(3, device=device, dtype=particles.dtype)
        zero_escaped = torch.zeros((), device=device, dtype=torch.long)

        phase_timer.stop(batch_setup_phase)

        t_cursor = batch_cursor_start
        while t_cursor > batch_cursor_terminus:
            # Every named phase of this step opens inside `loop_overhead`, so its own
            # (exclusive) time is the Python overhead of the loop between them.
            loop_phase = phase_timer.start("loop_overhead")
            if step_profiler is not None:
                step_profiler.before_step()
            # A span that is not a whole number of steps takes its short step FIRST,
            # at the top, where nothing has been released yet. A short LAST step would
            # end each particle's final step early (a particle counts for a step when
            # active at its upper end), costing it up to one dt of its lifetime.
            delta_s = _backward_step_seconds(
                (t_cursor - batch_cursor_terminus).total_seconds(), float(sim.dt_seconds)
            )
            t_prev = t_cursor - timedelta(seconds=delta_s)

            # Per-particle active mask uses both bounds (upper: cursor reached the
            # release time; lower: cursor still inside the backward window) and the
            # liveness mask. For a single-release batch with no escapes this is
            # identical to the pre-kill release-time check.
            cursor_offset_s = t_cursor.timestamp() - batch_start_ts
            with phase_timer.phase("mask_liveness"):
                active_mask = (
                    (release_time_offsets_s >= cursor_offset_s)
                    & (release_time_offsets_s - sim_length_s <= cursor_offset_s)
                    & alive
                )
            # Per-step body has two device-gated paths (docs/architecture.md S5):
            #   static  - full buffer every step, inactive frozen via torch.where; no
            #             per-step host sync; constant shapes for CUDA-graph capture.
            #   dynamic - boolean-index the active subset (cheaper on CPU). Unchanged.
            if use_static:
                with phase_timer.phase("met_fetch"):
                    met_window, fetched_windows = prefetcher.get(
                        _step_window_time(t_cursor, delta_s)
                    )
                hour_windows += fetched_windows
                if fetched_windows:
                    phase_timer.tick_fetch()
                with phase_timer.phase("exceedance_poll"):
                    exceedance.poll(met_window.metadata.time_start)
                # On the static path HannaScheme folds RK2 advection into its captured
                # graph (one graph for advect + turbulence), so the runtime must NOT
                # advect separately. A scheme that does NOT fold advection (e.g. the
                # placeholder, or any non-Hanna scheme on CUDA) gets the runtime's own
                # full-set advection + where-gate here. Either way we need `t_alpha`
                # (the RK2-midpoint weight, also used for the surface fields) and the
                # grid-mean wind diagnostic.
                with phase_timer.phase("wind_mean"):
                    t_alpha, wind_mean = _advection_alpha_wind_mean(
                        met_window, t_cursor, delta_s, device, particles.dtype, wind_mean_cache
                    )
                # Start-of-step positions (a copy, so nothing below can alias them).
                start_xyz = particles[:, :3].clone() if keep_step_start else None
                if not scheme.step_includes_advection(engine):
                    with phase_timer.phase("advect"):
                        advected_full = _advect_active_particles(
                            engine,
                            device,
                            particles,
                            met_window,
                            t_cursor,
                            delta_s,
                            particles.dtype,
                            wind_mean_cache=wind_mean_cache,
                        )[0]
                        particles = torch.where(active_mask.unsqueeze(1), advected_full, particles)
                with phase_timer.phase("step"):
                    particles, turbulence_state = scheme.step(
                        particles=particles,
                        state=turbulence_state,
                        met_window=met_window,
                        t_alpha=t_alpha,
                        dt_seconds=delta_s,
                        active_mask=active_mask,
                        engine=engine,
                    )
                # Convection: once per met bracket (Emanuel matrix is constant in a window).
                bracket_start = met_window.metadata.time_start
                if bracket_start != last_convection_bracket_start:
                    with phase_timer.phase("convection"):
                        particles, convection_state = convection_scheme.maybe_convect(
                            particles=particles,
                            state=convection_state,
                            met_window=met_window,
                            t_alpha=t_alpha,
                            dt_seconds=delta_s,
                            active_mask=active_mask,
                            engine=engine,
                        )
                    last_convection_bracket_start = bracket_start
                with phase_timer.phase("mask_liveness"):
                    elapsed_s = (release_window_end_offsets_s - cursor_offset_s).clamp_(min=0.0)
                    t_idx = _time_ago_index(elapsed_s, time_bin_s, gridder.n_t)
                with phase_timer.phase("gridder"):
                    _deposit_step(
                        gridder,
                        position=deposition_position,
                        start_xyz=start_xyz,
                        end_xyz=particles[:, :3],
                        active_mask=active_mask,
                        weights=particles[:, 3],
                        elapsed_s=elapsed_s,
                        t_idx=t_idx,
                        time_bin_s=time_bin_s,
                        release_idx=release_idx_local,
                        delta_s=delta_s,
                        t_cursor=t_cursor,
                        particle_density=particle_density,
                        met_window=met_window,
                        t_alpha=t_alpha,
                        engine=engine,
                    )
                with phase_timer.phase("mask_liveness"):
                    newly_escaped = active_mask & ~_within_met_domain(particles, cfg, engine.frame)
                    alive = alive & ~newly_escaped
                    escaped_count = torch.count_nonzero(newly_escaped)
            else:
                with phase_timer.phase("mask_liveness"):
                    active_count = int(torch.count_nonzero(active_mask).item())

                if active_count > 0:
                    active_particles = particles[active_mask]
                    with phase_timer.phase("met_fetch"):
                        met_window, fetched_windows = prefetcher.get(
                            _step_window_time(t_cursor, delta_s)
                        )
                    hour_windows += fetched_windows
                    if fetched_windows:
                        phase_timer.tick_fetch()
                    with phase_timer.phase("exceedance_poll"):
                        exceedance.poll(met_window.metadata.time_start)
                    # Start-of-step positions; `particles` is updated in place below.
                    start_xyz = particles[:, :3].clone() if keep_step_start else None
                    with phase_timer.phase("advect"):
                        advected_active, t_alpha, wind_mean = _advect_active_particles(
                            engine,
                            device,
                            active_particles,
                            met_window,
                            t_cursor,
                            delta_s,
                            particles.dtype,
                            wind_mean_cache=wind_mean_cache,
                        )
                        particles[active_mask] = advected_active

                    with phase_timer.phase("step"):
                        particles, turbulence_state = scheme.step(
                            particles=particles,
                            state=turbulence_state,
                            met_window=met_window,
                            t_alpha=t_alpha,
                            dt_seconds=delta_s,
                            active_mask=active_mask,
                            engine=engine,
                        )

                    # Deep convection - fired ONCE per met-update interval (when the
                    # cursor crosses into a new hourly bracket), not per timestep.
                    bracket_start = met_window.metadata.time_start
                    if bracket_start != last_convection_bracket_start:
                        with phase_timer.phase("convection"):
                            particles, convection_state = convection_scheme.maybe_convect(
                                particles=particles,
                                state=convection_state,
                                met_window=met_window,
                                t_alpha=t_alpha,
                                dt_seconds=delta_s,
                                active_mask=active_mask,
                                engine=engine,
                            )
                        last_convection_bracket_start = bracket_start

                    del active_particles
                    del advected_active
                else:
                    wind_mean = zero_wind

                if active_count > 0:
                    with phase_timer.phase("mask_liveness"):
                        # Per-particle time-ago bin: each particle measures elapsed time
                        # from *its own* release window's end (stage 3 semantics).
                        elapsed_s = (release_window_end_offsets_s - cursor_offset_s).clamp_(min=0.0)
                        t_idx = _time_ago_index(elapsed_s, time_bin_s, gridder.n_t)
                    with phase_timer.phase("gridder"):
                        _deposit_step(
                            gridder,
                            position=deposition_position,
                            start_xyz=start_xyz,
                            end_xyz=particles[:, :3],
                            active_mask=active_mask,
                            weights=particles[:, 3],
                            elapsed_s=elapsed_s,
                            t_idx=t_idx,
                            time_bin_s=time_bin_s,
                            release_idx=release_idx_local,
                            delta_s=delta_s,
                            t_cursor=t_cursor,
                            particle_density=particle_density,
                            met_window=met_window,
                            t_alpha=t_alpha,
                            engine=engine,
                        )

                # Kill particles that left met_domain this step (drop-and-count). They
                # accumulated 0 already (outside output_grid -> dropped by valid_mask);
                # alive excludes them from future steps. Escaped count is a device
                # tensor summed at batch end.
                if active_count > 0:
                    with phase_timer.phase("mask_liveness"):
                        newly_escaped = active_mask & ~_within_met_domain(
                            particles, cfg, engine.frame
                        )
                        alive = alive & ~newly_escaped
                        escaped_count = torch.count_nonzero(newly_escaped)
                else:
                    escaped_count = zero_escaped

            step_count += 1

            # Accumulate diagnostics as device tensors (no host sync this step);
            # materialised once per batch below. active_mask.sum() replaces the
            # former per-step active_count .item() in the diagnostics.
            with phase_timer.phase("diagnostics"):
                diag_pos_means.append(particles[:, :3].mean(dim=0))
                diag_wind_means.append(wind_mean)
                diag_escaped.append(escaped_count)
                diag_active.append(active_mask.sum())
                diag_host.append((step_count, batch.batch_idx, t_cursor.isoformat()))

            if mem.gc_every_steps > 0 and step_count % mem.gc_every_steps == 0:
                with phase_timer.phase("gc"):
                    gc.collect()

            log_guard_phase = phase_timer.start("log_guard")
            if mem.log_every_steps > 0 and step_count % mem.log_every_steps == 0:
                rss_bytes, allocated, reserved = memory_stats.observe(device)
                # Materialise the active count just for this log line (runs only every
                # `log_every_steps`, so the sync is off the hot path). Works on both
                # the static and dynamic paths; `active_count` itself only exists in
                # the dynamic branch.
                LOGGER.info(
                    "batch=%d step=%d active=%d cache_hours=%d rss=%s dev_alloc=%s dev_reserved=%s",
                    batch.batch_idx,
                    step_count,
                    int(active_mask.sum().item()),
                    len(met_cache),
                    _format_gib(rss_bytes),
                    _format_gib(allocated),
                    _format_gib(reserved),
                )

            if _memory_guard_enabled(cfg) and step_count % mem.guard_check_every_steps == 0:
                rss_bytes, allocated, reserved = memory_stats.observe(device)
                _raise_if_memory_guard_exceeded(
                    writer,
                    outputs.metadata,
                    cfg,
                    schedule_release_end,
                    schedule_sim_start,
                    step_count,
                    hour_windows,
                    outputs,
                    memory_stats,
                    rss_bytes,
                    allocated,
                    reserved,
                )

            phase_timer.stop(log_guard_phase)

            if step_profiler is not None:
                step_profiler.after_step()

            t_cursor = t_prev
            phase_timer.stop(loop_phase)

        batch_end_phase = phase_timer.start("batch_end")
        # Sub-step exceedance counts of this batch's last met window.
        with phase_timer.phase("exceedance_poll"):
            exceedance.flush()

        # Materialise this batch's per-step diagnostics in ONE host transfer
        # (instead of a `.item()` sync every step). `alive_particles` is the
        # running survivor count = N − cumulative escaped, recovered from the
        # escaped-per-step tensor via cumsum.
        if diag_host:
            diag_phase = phase_timer.start("batch_diagnostics")
            pos = torch.stack(diag_pos_means).to("cpu").tolist()
            wind = torch.stack(diag_wind_means).to("cpu").tolist()
            escaped_dev = torch.stack(diag_escaped).to("cpu")
            batch_escaped_total = int(escaped_dev.sum().item())
            escaped_each = escaped_dev.tolist()
            active_each = torch.stack(diag_active).to("cpu").tolist()
            alive_each = (particles.shape[0] - torch.cumsum(escaped_dev, dim=0)).tolist()
            for (st, bidx, iso), ac, pm, wm, esc, al in zip(
                diag_host, active_each, pos, wind, escaped_each, alive_each
            ):
                diag_rows.append(
                    {
                        "step": st,
                        "batch_idx": bidx,
                        "time_hour_end": iso,
                        # Mean position, from the frame back to degrees (affine,
                        # so the mean converts exactly up to rounding).
                        "mean_lon": float(engine.frame.lon_from_x(pm[0])),
                        "mean_lat": float(engine.frame.lat_from_y(pm[1])),
                        "mean_alt_agl_m": pm[2],
                        "u_mean_m_s": wm[0],
                        "v_mean_m_s": wm[1],
                        "w_mean_m_s": wm[2],
                        "active_particles": ac,
                        "escaped_this_step": esc,
                        "alive_particles": al,
                    }
                )
            phase_timer.stop(diag_phase)

        # End of batch: stream this batch's footprint slice to disk, then drop the
        # gridder and per-batch tensors so peak footprint memory is one batch's
        # worth, not the whole schedule's. (met_cache + bookkeeping persist.)
        with phase_timer.phase("output_write"):
            writer.write_footprint_region(
                outputs.footprints,
                # The store is float32; a float64 accumulator is rounded once here.
                gridder.tensor.to(torch.float32),
                release_start=batch_release_offset,
                release_stop=batch_release_offset + len(batch.releases),
            )
            # Endpoints are written in (lon, lat): out of the frame here.
            endpoint_particles_chunks.append(
                engine.frame.to_lonlat(particles.detach()).cpu().clone()
            )
            endpoint_release_idx_chunks.append(particle_batch.release_idx.detach().cpu().clone())
        _log_device_memory(f"batch {batch.batch_idx} end (pre-reclaim)", device)
        del gridder
        del particles
        del release_idx_local
        del release_time_offsets_s
        del release_window_end_offsets_s
        del turbulence_state
        del convection_state
        del particle_batch
        del alive
        # Per-batch reclaim: break reference cycles (closures / compiled-fn caches retain
        # device tensors that plain `del` + refcounting won't free) and return the freed
        # blocks to the driver, so peak device memory stays ~one batch's worth instead of
        # growing per batch toward OOM on a long run (2026-06-29 dev_alloc investigation).
        if device.type == "cuda":
            gc.collect()
            torch.cuda.empty_cache()
        _log_device_memory(f"batch {batch.batch_idx} end (post-reclaim)", device)
        _maybe_dump_mem_snapshot(batch.batch_idx, device)
        run_escaped_total += batch_escaped_total
        phase_timer.stop(batch_end_phase)

    final_phase = phase_timer.start("output_write")
    prefetcher.shutdown()
    for cached in met_cache.values():
        del cached
    met_cache.clear()

    endpoint_particles_all = torch.cat(endpoint_particles_chunks, dim=0)
    endpoint_release_idx_all = torch.cat(endpoint_release_idx_chunks, dim=0)
    writer.write_particles_parquet(
        outputs.endpoint_particles,
        endpoint_particles_all,
        release_idx=endpoint_release_idx_all,
    )
    writer.write_trajectory_parquet(
        outputs.trajectory_diagnostics, step_seconds=sim.dt_seconds, rows=diag_rows
    )

    # Task B2: optional sparse copy of the footprint, converted release by release.
    sparse_path = None
    if cfg.output_grid.footprint_sparse:
        from lpdm.sparse_footprint import write_sparse_footprints

        sparse_path = outputs.footprints.replace("footprints.zarr", "footprints_sparse.zarr")
        counts = write_sparse_footprints(outputs.footprints, sparse_path)
        LOGGER.info(
            "sparse footprint: %d of %d cells non-zero -> %s",
            counts["nonzero"],
            counts["cells"],
            sparse_path,
        )

    metadata = {
        "config": _config_metadata(cfg, schedule_release_end, schedule_sim_start),
        "runtime": _runtime_metadata(
            cfg,
            step_count,
            hour_windows,
            memory_stats,
            status="completed",
            # The period actually used to de-accumulate surface fluxes, not the one
            # configured: it is normally resolved from the store attr, and a run whose
            # heat fluxes look wrong is diagnosed by reading it back here rather than
            # re-deriving it from the cube.
            flux_accumulation_seconds=getattr(reader, "accumulation_seconds", None),
            # Channels held through each window as means/accumulations (decision
            # 0020), with their cell_methods and period: how the surface forcing
            # was placed in time is part of what a run used.
            met_period_channels=dict(getattr(reader, "period_details", {}) or {}),
            # The seed of torch's default generators (decision 0021), or None.
            rng_seed=run_seed,
            # Each meteorology store's provenance (format version, producer,
            # source, recipe...): which inputs a run used (task B1).
            met_stores=(reader.store_provenance() if hasattr(reader, "store_provenance") else []),
            # Particle-sub-steps long compared with T_Lw, boundary layer and free
            # troposphere (decision 0027; replaced A1's substep_saturation), or None
            # for a scheme without sub-steps.
            substep_exceedance=exceedance.metadata(),
            # Horizontal frame the particle positions were held in (task A4).
            position_frame={
                "kind": position_frame.kind,
                "lon0_deg": position_frame.lon0,
                "lat0_deg": position_frame.lat0,
                "x_units_per_deg_lon": position_frame.kx,
                "y_units_per_deg_lat": position_frame.ky,
            },
            # Configured physics options that are not well-mixed consistent, as
            # logged at the start of the run; empty for the defaults.
            physics_warnings=physics_warnings,
        ),
        "schedule": {
            "n_batches": len(batches),
            "n_releases": total_releases,
            "max_releases_per_batch": cfg.batch.max_releases_per_batch,
            "release_times": [rel.release_time.isoformat() for rel in all_releases],
            "escaped_met_domain_total": run_escaped_total,
        },
        "outputs": {
            "endpoint_particles": outputs.endpoint_particles,
            "trajectory_diagnostics": outputs.trajectory_diagnostics,
            "footprints": outputs.footprints,
            "metadata": outputs.metadata,
            **({"footprints_sparse": sparse_path} if sparse_path else {}),
        },
        "footprint_units": _footprint_units_doc(cfg),
    }
    writer.write_metadata_json(outputs.metadata, metadata)
    phase_timer.stop(final_phase)

    exceedance.log_run_total()
    phase_timer.summary(hour_windows=hour_windows)
    install_phase_timer(None)
    return metadata


def _configure_cpu_threads() -> None:
    """Set torch's intra-op thread pool from ``GLIDE_NUM_THREADS`` if present.

    The per-step tensors here are modest (10^4–10^5 particles), so torch's CPU
    parallel sections hit lock-contention overhead well before they saturate a
    big core count: benchmarking the real engine on a 48-core node showed ~16
    threads is ~25% faster than 48 at both 20k and 200k particles (the SLURM
    default of one-thread-per-core is the *slowest* option). We therefore honour
    an explicit ``GLIDE_NUM_THREADS`` override and otherwise leave torch's
    default untouched. CPU-only knob — no effect on CUDA/MPS runs.
    """

    raw = os.environ.get("GLIDE_NUM_THREADS")
    if not raw:
        return
    try:
        n = int(raw)
    except ValueError:
        LOGGER.warning("Ignoring non-integer GLIDE_NUM_THREADS=%r", raw)
        return
    if n < 1:
        LOGGER.warning("Ignoring GLIDE_NUM_THREADS=%d (must be >= 1)", n)
        return
    torch.set_num_threads(n)
    LOGGER.info("torch intra-op threads set to %d (GLIDE_NUM_THREADS)", n)


def main(argv: Sequence[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    _configure_cpu_threads()

    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    cfg = RunConfig.from_yaml(args.config).with_overrides(
        device=args.device,
        output_uri=args.output_uri,
        start_time=args.start_time,
    )

    if cfg.turbulence.scheme not in list_schemes():
        parser.exit(
            2,
            f"Error: unknown turbulence scheme {cfg.turbulence.scheme!r}. Registered: {', '.join(list_schemes())}\n",
        )
    if cfg.convection.scheme not in list_convection_schemes():
        parser.exit(
            2,
            f"Error: unknown convection scheme {cfg.convection.scheme!r}. Registered: {', '.join(list_convection_schemes())}\n",
        )

    try:
        metadata = _run(cfg)
    except PreflightValidationError as exc:
        parser.exit(2, f"Error: {exc}\n")
    print("LPDM run complete")
    print(metadata["outputs"])


if __name__ == "__main__":
    main()
