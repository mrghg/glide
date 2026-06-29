"""Top-level LPDM orchestration entry point.

This module provides a minimal end-to-end backward trajectory run mode suitable
for early cloud testing (including Vertex AI user-managed notebooks).

The CLI surface is intentionally small: a YAML ``--config`` plus a few overrides
(``--device``, ``--output-uri``, ``--start-time``). Schema is defined in
:mod:`lpdm.config`.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import logging
import os
import resource
import sys
import time
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from lpdm.config import ConcreteRelease, RunConfig, _release_point
from lpdm.footprint_gridder import FootprintGridder
from lpdm.gpu_engine import GPUEngine, GridInterpolationBounds, use_static_step_path
from lpdm.met_reader import (
	ArcoEra5ZarrReader,
	BoundingBoxRequest,
	HourlyMetTensors,
	MetReader,
	SpatialBounds,
	TimeBounds,
)
from lpdm.output_writer import OutputWriter
from lpdm.release_generator import generate_batch_particles
from lpdm.runtime import DEVICE
from lpdm.turbulence import TurbulenceScheme, get_scheme, list_schemes
from lpdm.convection import (
	ConvectionScheme,
	get_scheme as get_convection_scheme,
	list_schemes as list_convection_schemes,
)


LOGGER = logging.getLogger(__name__)


def _emit_profile_summary(prof, device: torch.device, trace_path: Path, wall_s: float) -> None:
	"""Write a Chrome trace and print a CPU/GPU summary for a profiled step window.

	The summary is built to localise *why the GPU idles*: the GPU-busy fraction, any
	host-sync ops (``cudaStreamSynchronize`` ⇒ a stall), and the top ops by GPU and
	CPU self-time (many tiny GPU ops ⇒ launch-bound; large CPU spans with little GPU
	⇒ Python/met-bound). See architecture.md §5.
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
			print(f"  {e.key}: count={e.count}  cpu_total={getattr(e, 'cpu_time_total', 0) / 1000:.1f}ms")
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

	def __init__(self, *, device: torch.device, warmup: int, active: int, trace_path: Path, exit_after: bool) -> None:
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
				LOGGER.info("profiler: window complete; exiting (GLIDE_PROFILE_CONTINUE=1 to finish the run)")
				raise SystemExit(0)


def _make_step_profiler(device: torch.device) -> "_StepProfiler | None":
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


class _PhaseTimer:
	"""Coarse whole-run wall-clock accountant, enabled by ``GLIDE_PHASE_TIMERS``.

	Where ``_StepProfiler`` samples ~20 steps inside a SINGLE met window (so it's blind to
	per-window met I/O and batch boundaries — the very places the GPU goes idle), this
	accumulates wall time across the WHOLE run bucketed by phase and prints a breakdown. It
	answers "where does the wall go over many windows/batches" — chiefly: *is the GPU idling
	on synchronous met fetches?* The ``met_fetch`` bucket is exact (synchronous CPU/IO, GPU
	uninvolved), so ``met_fetch / wall`` is a faithful idle-fraction even with no CUDA sync.

	It also logs a running breakdown every ``log_every`` met fetches, so even a job that
	*times out* (no final summary, no manifest) leaves the split in the log. NOTE: the
	``step`` bucket includes the one-time ``torch.compile`` (~tens of s) charged to the first
	step — negligible on a long run, but run long enough that it washes out (or compile-off).

	Env:
	* ``GLIDE_PHASE_TIMERS``        — enable (truthy)
	* ``GLIDE_PHASE_TIMERS_SYNC``   — ``torch.cuda.synchronize()`` at each phase boundary so
	                                  GPU phases (step/gridder) get real device time. This
	                                  SERIALISES the pipeline (perturbs absolute numbers but
	                                  cleans the step-vs-gridder split); off by default.
	* ``GLIDE_PHASE_TIMERS_EVERY``  — log running breakdown every N met fetches (default 25)
	"""

	_NULLCTX = contextlib.nullcontext()

	def __init__(self, *, enabled: bool, sync_cuda: bool, log_every: int) -> None:
		self.enabled = enabled
		self.sync_cuda = sync_cuda
		self.log_every = max(0, log_every)
		self.totals: dict[str, float] = defaultdict(float)
		self.counts: dict[str, int] = defaultdict(int)
		self.t_start = time.perf_counter()
		self._fetches = 0

	def phase(self, name: str):
		"""Context manager timing a phase; a cheap shared no-op when disabled."""
		if not self.enabled:
			return self._NULLCTX
		return self._timed(name)

	@contextlib.contextmanager
	def _timed(self, name: str):
		if self.sync_cuda:
			torch.cuda.synchronize()
		t0 = time.perf_counter()
		try:
			yield
		finally:
			if self.sync_cuda:
				torch.cuda.synchronize()
			self.totals[name] += time.perf_counter() - t0
			self.counts[name] += 1

	def tick_fetch(self) -> None:
		"""Call after each REAL met fetch (cache miss); logs a running breakdown periodically."""
		if not self.enabled or self.log_every == 0:
			return
		self._fetches += 1
		if self._fetches % self.log_every == 0:
			wall = time.perf_counter() - self.t_start
			parts = ", ".join(
				f"{n}={t:.1f}s({100 * t / wall:.0f}%)"
				for n, t in sorted(self.totals.items(), key=lambda kv: kv[1], reverse=True)
			)
			LOGGER.info("phase timers @ %d fetches — wall %.1fs: %s", self._fetches, wall, parts)

	def summary(self, *, hour_windows: int) -> None:
		if not self.enabled:
			return
		wall = time.perf_counter() - self.t_start
		accounted = sum(self.totals.values())
		note = "  [CUDA-synced]" if self.sync_cuda else "  [no CUDA sync: GPU phases under-counted]"
		lines = [
			"=" * 78,
			f"GLIDE phase timers — total wall {wall:.1f}s  (met fetches: {hour_windows}){note}",
			"-" * 78,
			f"  {'phase':<22}{'total_s':>12}{'% wall':>9}{'calls':>11}{'ms/call':>13}",
		]
		for name, total in sorted(self.totals.items(), key=lambda kv: kv[1], reverse=True):
			c = self.counts[name]
			pct = 100.0 * total / wall if wall else 0.0
			per = 1000.0 * total / c if c else 0.0
			lines.append(f"  {name:<22}{total:>12.2f}{pct:>8.1f}%{c:>11}{per:>13.3f}")
		resid = wall - accounted
		lines.append(f"  {'(residual/untimed)':<22}{resid:>12.2f}{(100.0 * resid / wall if wall else 0.0):>8.1f}%")
		lines.append("=" * 78)
		print("\n".join(lines))


def _make_phase_timer() -> "_PhaseTimer":
	"""Build the phase timer from env; always returns an object (a no-op when disabled, so
	call sites need no None-checks)."""
	def _truthy(name: str) -> bool:
		return os.environ.get(name, "") not in ("", "0", "false", "False")

	return _PhaseTimer(
		enabled=_truthy("GLIDE_PHASE_TIMERS"),
		sync_cuda=_truthy("GLIDE_PHASE_TIMERS_SYNC") and torch.cuda.is_available(),
		log_every=int(os.environ.get("GLIDE_PHASE_TIMERS_EVERY", "25")),
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


def _current_rss_bytes() -> int | None:
	"""Return process memory bytes when available."""

	try:
		with open("/proc/self/status", "r", encoding="utf-8") as f:
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
	return f"{num_bytes / (1024 ** 3):.3f} GiB"


def _hour_floor(dt: datetime) -> datetime:
	return dt.astimezone(timezone.utc).replace(minute=0, second=0, microsecond=0)


def _footprint_time_bin_index(release_end: datetime, t_cursor: datetime, n_time_bins: int) -> int:
	"""Map the current backward integration cursor to a 0-based time_ago bin."""

	if n_time_bins <= 0:
		raise ValueError("n_time_bins must be > 0")

	elapsed_seconds = max(0.0, release_end.timestamp() - t_cursor.timestamp())
	return min(n_time_bins - 1, int(elapsed_seconds / 3600.0))


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
	hours_per_bin = max(
		1.0, float(cfg.simulation.length_seconds) / 3600.0 / max(1, gridder.n_t)
	)
	time_bin_start_h = np.arange(gridder.n_t, dtype=np.float64) * hours_per_bin
	time_bin_end_h = time_bin_start_h + hours_per_bin

	release_times = np.array(
		[np.datetime64(rel.release_time.replace(tzinfo=None), "ns") for rel in releases]
	)
	release_durations_s = np.array(
		[float(rel.duration_seconds) for rel in releases], dtype=np.float64
	)

	coords: dict[str, object] = {
		"release_time": release_times,
		"release_duration_seconds": ("release_time", release_durations_s),
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
	lon, lat, alt_agl_m = _release_point(cfg.release)
	attrs = {
		"release_lon": float(lon),
		"release_lat": float(lat),
		"release_alt_agl_m": float(alt_agl_m),
	}
	return coords, attrs


def _validate_meteorology_time_coverage(
	reader: MetReader,
	cfg: RunConfig,
	release_end: datetime,
	sim_start: datetime,
) -> None:
	"""Fail early when the configured run requires hours outside dataset coverage."""

	available_start, available_end = reader.get_time_coverage()
	required_start = _hour_floor(sim_start)
	required_end = _hour_floor(release_end) + timedelta(hours=1)

	if required_start < available_start or required_end > available_end:
		raise PreflightValidationError(
			"Meteorological dataset does not cover the requested simulation window. "
			f"Need hourly data from {required_start.isoformat()} through {required_end.isoformat()}, "
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
	parser = argparse.ArgumentParser(description="Run a backward LPDM trajectory simulation from a YAML config")
	parser.add_argument("--config", required=True, help="Path to the run YAML config (see configs/example_mhd_january.yaml)")
	parser.add_argument("--device", default=None, help="Override simulation.device")
	parser.add_argument("--output-uri", default=None, help="Override io.output_uri")
	parser.add_argument("--start-time", default=None, help="Override simulation.start_time (UTC ISO)")
	return parser


def _build_output_paths(output_uri: str) -> OutputPaths:
	base_uri = output_uri.rstrip("/")
	return OutputPaths(
		endpoint_particles=f"{base_uri}/endpoint_particles.parquet",
		trajectory_diagnostics=f"{base_uri}/trajectory_diagnostics.parquet",
		footprints=f"{base_uri}/footprints.zarr",
		metadata=f"{base_uri}/run_metadata.json",
	)


def _config_metadata(cfg: RunConfig, release_end: datetime, sim_start: datetime) -> dict[str, object]:
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
			"footprint_units": FOOTPRINT_UNITS_DOC,
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

		limit_bytes = int(limit_gib * (1024 ** 3))
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


def _within_met_domain(particles: torch.Tensor, cfg: RunConfig) -> torch.Tensor:
	"""Boolean [N] mask: True where a particle is inside ``met_domain``.

	Outside the lon/lat bbox or above ``alt_max_m`` there is no valid met — the
	engine's coordinate normalisation clamps to the grid edge, so such a particle
	would otherwise keep being pushed by edge winds for the rest of the
	integration (wasted compute; it can't contribute to the footprint either,
	being outside ``output_grid``). The runtime kills these particles. The z=0
	floor is handled by surface reflection, so there is no lower-altitude kill.
	"""

	md = cfg.met_domain
	lon = particles[:, 0]
	lat = particles[:, 1]
	alt = particles[:, 2]
	return (
		(lon >= md.lon_bounds[0])
		& (lon <= md.lon_bounds[1])
		& (lat >= md.lat_bounds[0])
		& (lat <= md.lat_bounds[1])
		& (alt <= md.alt_max_m)
	)


def _get_hourly_met_window(
	reader: MetReader,
	cfg: RunConfig,
	t_cursor: datetime,
	met_cache: OrderedDict[datetime, HourlyMetTensors],
) -> tuple[HourlyMetTensors, int]:
	hour_key = _hour_floor(t_cursor)
	if cfg.memory.met_cache_max_hours > 0 and hour_key in met_cache:
		met_cache.move_to_end(hour_key)
		return met_cache[hour_key], 0

	met_window = reader.fetch_hourly_window(
		BoundingBoxRequest(
			spatial=_met_domain_bounds(cfg),
			time=TimeBounds(start=hour_key, end=hour_key + timedelta(hours=1)),
		)
	)

	if cfg.memory.met_cache_max_hours > 0:
		met_cache[hour_key] = met_window
		while len(met_cache) > cfg.memory.met_cache_max_hours:
			_, evicted = met_cache.popitem(last=False)
			del evicted

	return met_window, 1


def _advection_alpha_wind_mean(
	met_window: HourlyMetTensors,
	t_cursor: datetime,
	delta_s: float,
	device: torch.device,
	dtype: torch.dtype,
) -> tuple[float, torch.Tensor]:
	"""RK2-midpoint time-interpolation weight ``alpha`` + the grid-mean (u, v, w) wind.

	Shared by the dynamic path (`_advect_active_particles`) and the static path, where
	advection is folded into `HannaScheme._step_core` so `main._run` only needs these two
	scalars/diagnostics. ``wind_mean`` is a (3,) device tensor (no host sync — materialised
	once per batch for the trajectory diagnostics).
	"""

	t_start_s = met_window.metadata.time_start.timestamp()
	t_end_s = met_window.metadata.time_end.timestamp()
	t_eval_s = t_cursor.timestamp() - 0.5 * delta_s
	alpha = max(0.0, min(1.0, (t_eval_s - t_start_s) / max(1.0, float(t_end_s - t_start_s))))

	m_start_uvw, m_end_uvw = met_window.channels("u", "v", "w")
	interp_wind = (
		m_start_uvw.to(device=device, dtype=dtype) * (1.0 - alpha)
		+ m_end_uvw.to(device=device, dtype=dtype) * alpha
	)  # [3, Z, Y, X]
	wind_mean = interp_wind.reshape(3, -1).mean(dim=1)
	return alpha, wind_mean


def _advect_active_particles(
	engine: GPUEngine,
	device: torch.device,
	active_particles: torch.Tensor,
	met_window: HourlyMetTensors,
	t_cursor: datetime,
	delta_s: float,
	dtype: torch.dtype,
) -> tuple[torch.Tensor, float, torch.Tensor]:
	"""RK2 backward advection on the active particle subset (dynamic / CPU path).

	The third return is the grid-mean (u, v, w) wind as a device tensor — NOT
	materialised to floats here, so the trajectory diagnostics can be transferred
	to the host once per batch rather than syncing every step (matters on CUDA).
	"""

	alpha, diag_wind_mean = _advection_alpha_wind_mean(met_window, t_cursor, delta_s, device, dtype)

	grid_bounds = GridInterpolationBounds(
		lon_first=float(met_window.metadata.lon[0]),
		lon_last=float(met_window.metadata.lon[-1]),
		lat_first=float(met_window.metadata.lat[0]),
		lat_last=float(met_window.metadata.lat[-1]),
		alt_first=float(met_window.metadata.level[0]),
		alt_last=float(met_window.metadata.level[-1]),
		# F9 (audit 2026-05-30): pass the per-level AGL array so the vertical
		# normalisation uses a piecewise-linear lookup rather than a linear-in-z
		# approximation. Matters because pressure levels are log-linear in z.
		level_agl_m=tuple(float(v) for v in met_window.metadata.level),
	)

	m_start_uvw, m_end_uvw = met_window.channels("u", "v", "w")
	m_start = m_start_uvw.unsqueeze(0).to(device=device, dtype=dtype)
	m_end = m_end_uvw.unsqueeze(0).to(device=device, dtype=dtype)

	def wind_fn(xyz: torch.Tensor) -> torch.Tensor:
		xyz_norm = engine.normalize_particle_coordinates(xyz, grid_bounds)
		grid = xyz_norm.view(1, 1, 1, -1, 3)

		v_start = torch.nn.functional.grid_sample(m_start, grid, align_corners=True).view(3, -1).t()
		v_end = torch.nn.functional.grid_sample(m_end, grid, align_corners=True).view(3, -1).t()
		v_interp = v_start * (1.0 - alpha) + v_end * alpha

		lat_deg = xyz[:, 1]
		meters_per_deg_lat = 110540.0
		meters_per_deg_lon = 111320.0 * torch.cos(torch.deg2rad(lat_deg)).abs().clamp(min=0.05)

		out = torch.empty_like(xyz)
		out[:, 0] = v_interp[:, 0] / meters_per_deg_lon
		out[:, 1] = v_interp[:, 1] / meters_per_deg_lat
		out[:, 2] = v_interp[:, 2]
		return out

	advected_active = engine.rk2_advect_backward(active_particles, dt_seconds=delta_s, wind_fn=wind_fn)

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
		"substep_c": cfg.turbulence.substep_c,
		"max_substeps": cfg.turbulence.max_substeps,
	}
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
	gib = 2 ** 30
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

	if device.type != "cuda" or os.environ.get("GLIDE_MEM_SNAPSHOT", "") in ("", "0", "false", "False"):
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

	if device.type != "cuda" or os.environ.get("GLIDE_MEM_SNAPSHOT", "") in ("", "0", "false", "False"):
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
def _run(
	cfg: RunConfig,
	*,
	reader: MetReader | None = None,
	scheme: TurbulenceScheme | None = None,
	convection_scheme: ConvectionScheme | None = None,
) -> dict[str, object]:
	sim = cfg.simulation
	mem = cfg.memory
	out_grid = cfg.output_grid
	device_str = _resolve_device(sim.device)

	if scheme is None:
		scheme = get_scheme(cfg.turbulence.scheme, **_scheme_kwargs(cfg))
	if convection_scheme is None:
		convection_scheme = get_convection_scheme(
			cfg.convection.scheme, **_convection_scheme_kwargs(cfg),
		)
	if reader is None:
		# Channels = baseline (advection needs u/v/w; runtime telemetry uses blh/sp)
		# unioned with whatever the turbulence + convection schemes declare they
		# need (e.g. Hanna adds ustar/shf/t; Emanuel adds q).
		required_channels = tuple(
			dict.fromkeys((
				"u", "v", "w", "blh", "sp",
				*scheme.required_met_keys(),
				*convection_scheme.required_met_keys(),
			))
		)
		reader = ArcoEra5ZarrReader(
			zarr_store=cfg.io.zarr_store,
			channel_names=required_channels,
			device=device_str,
		)
	engine = GPUEngine(device=device_str)
	writer = OutputWriter()
	device = torch.device(device_str)

	# Per-step execution path (architecture.md §5). On the static path the cursor
	# loop processes the FULL particle buffer every step and gates inactive
	# particles with `torch.where` (no boolean indexing, no per-step host sync) —
	# the shapes/control-flow CUDA-graph capture needs, and a win on a launch-bound
	# GPU. On the dynamic path it boolean-indexes the active subset (cheaper on
	# CPU). Auto: static on CUDA, dynamic elsewhere; `GLIDE_STATIC_SUBSTEPS`
	# overrides (used to exercise the static path on CPU in tests).
	use_static = use_static_step_path(device)
	if use_static:
		LOGGER.info("runtime: static-shape per-step path (full-set, mask-gated)")
	_start_mem_history(device)

	# Optional per-step profiler (GLIDE_PROFILE): captures a window of cursor-loop
	# steps to localise where the per-step time goes (GPU kernels vs host syncs vs
	# CPU/met). None unless GLIDE_PROFILE is set, so normal runs pay nothing.
	step_profiler = _make_step_profiler(device)
	# Coarse whole-run phase accountant (GLIDE_PHASE_TIMERS). Unlike the step profiler it
	# spans every window/batch, so it catches per-window met-fetch idle. No-op when disabled.
	phase_timer = _make_phase_timer()

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
	schedule_sim_start = min(all_window_ends) - timedelta(seconds=sim.length_seconds)
	_validate_meteorology_time_coverage(reader, cfg, schedule_release_end, schedule_sim_start)

	LOGGER.info(
		"schedule expanded: %d release(s) across %d batch(es), max %d per batch",
		total_releases,
		len(batches),
		cfg.batch.max_releases_per_batch,
	)

	# Bookkeeping shared across batches.
	diag_rows: list[dict[str, float | int | str]] = []
	step_count = 0
	hour_windows = 0
	met_cache: OrderedDict[datetime, HourlyMetTensors] = OrderedDict()
	memory_stats = MemoryStats()
	# Endpoint particles + their release_idx accumulated across batches, concatenated
	# once at the end so we emit one parquet covering the whole schedule.
	endpoint_particles_chunks: list[torch.Tensor] = []
	endpoint_release_idx_chunks: list[torch.Tensor] = []

	outputs = _build_output_paths(cfg.io.output_uri)
	sim_length_s = float(sim.length_seconds)
	# Total particles killed for leaving met_domain across the whole run.
	run_escaped_total = 0

	# Streaming footprint output: instead of holding the full
	# (total_releases, T, Z, Y, X) tensor in memory for the whole run, each batch
	# accumulates into a gridder sized for that batch alone, writes its slice to
	# the output Zarr region, then frees the gridder. Peak footprint memory is
	# therefore one batch's worth, not the whole schedule's. The store is created
	# lazily on the first batch (we need a gridder's geometry to build coords).
	footprint_store_created = False

	for batch in batches:
		if device.type == "cuda":
			torch.cuda.reset_peak_memory_stats(device)
		_log_device_memory(f"batch {batch.batch_idx} start", device)
		particle_batch = generate_batch_particles(batch, device=device_str)
		particles = particle_batch.particles
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
			n_time_bins=out_grid.n_time_bins,
			n_y=out_grid.n_y,
			n_x=out_grid.n_x,
			n_releases=len(batch.releases),
			device=device,
			dtype=torch.float32,
		)

		if not footprint_store_created:
			# Geometry is batch-independent, so use this batch's gridder to build
			# the full-schedule coords/attrs, then create the empty store sized for
			# all releases. Subsequent batches just write their region.
			footprint_coords, footprint_attrs = _build_footprint_dataset_metadata(
				cfg, gridder, all_releases
			)
			writer.create_footprint_store(
				outputs.footprints,
				shape=(total_releases, gridder.n_t, gridder.n_z, gridder.n_y, gridder.n_x),
				coords=footprint_coords,
				attrs=footprint_attrs,
			)
			footprint_store_created = True

		# Per-batch cursor bounds: walk from the latest window-end down to the
		# earliest window-end minus sim.length_seconds. This gives every release
		# in the batch its own full backward integration window. For a
		# single-release batch this collapses to today's [release_end, sim_start]
		# bounds — preserving the stage 4 bit-equivalence guarantee.
		batch_window_ends = [
			rel.release_time + timedelta(seconds=rel.duration_seconds) for rel in batch.releases
		]
		batch_cursor_start = max(batch_window_ends)
		batch_cursor_terminus = min(batch_window_ends) - timedelta(seconds=sim_length_s)

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
		diag_pos_means: list[torch.Tensor] = []   # each (3,): mean lon/lat/alt over all particles
		diag_wind_means: list[torch.Tensor] = []  # each (3,): grid-mean u/v/w (0 when no active set)
		diag_escaped: list[torch.Tensor] = []     # each (): newly-escaped count this step
		diag_active: list[torch.Tensor] = []      # each (): active-particle count this step (device, no sync)
		diag_host: list[tuple[int, int, str]] = []  # (step, batch_idx, iso_time)
		zero_wind = torch.zeros(3, device=device, dtype=particles.dtype)
		zero_escaped = torch.zeros((), device=device, dtype=torch.long)

		t_cursor = batch_cursor_start
		while t_cursor > batch_cursor_terminus:
			if step_profiler is not None:
				step_profiler.before_step()
			delta_s = min(
				float(sim.dt_seconds),
				(t_cursor - batch_cursor_terminus).total_seconds(),
			)
			t_prev = t_cursor - timedelta(seconds=delta_s)

			# Per-particle active mask uses both bounds (upper: cursor reached the
			# release time; lower: cursor still inside the backward window) and the
			# liveness mask. For a single-release batch with no escapes this is
			# identical to the pre-kill release-time check (stage 3 entry in
			# CHECKPOINT.md).
			cursor_offset_s = t_cursor.timestamp() - batch_start_ts
			active_mask = (
				(release_time_offsets_s >= cursor_offset_s)
				& (release_time_offsets_s - sim_length_s <= cursor_offset_s)
				& alive
			)
			# Per-step body has two device-gated paths (architecture.md S5):
			#   static  - full buffer every step, inactive frozen via torch.where; no
			#             per-step host sync; constant shapes for CUDA-graph capture.
			#   dynamic - boolean-index the active subset (cheaper on CPU). Unchanged.
			if use_static:
				with phase_timer.phase("met_fetch"):
					met_window, fetched_windows = _get_hourly_met_window(
						reader, cfg, t_cursor, met_cache
					)
				hour_windows += fetched_windows
				if fetched_windows:
					phase_timer.tick_fetch()
				# On the static path HannaScheme folds RK2 advection into its captured
				# graph (one graph for advect + turbulence), so the runtime must NOT
				# advect separately. A scheme that does NOT fold advection (e.g. the
				# placeholder, or any non-Hanna scheme on CUDA) gets the runtime's own
				# full-set advection + where-gate here. Either way we need `t_alpha`
				# (the RK2-midpoint weight, also used for the surface fields) and the
				# grid-mean wind diagnostic.
				t_alpha, wind_mean = _advection_alpha_wind_mean(
					met_window, t_cursor, delta_s, device, particles.dtype
				)
				if not scheme.step_includes_advection(engine):
					with phase_timer.phase("advect"):
						advected_full = _advect_active_particles(
							engine, device, particles, met_window, t_cursor, delta_s, particles.dtype,
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
				elapsed_s = (release_window_end_offsets_s - cursor_offset_s).clamp_(min=0.0)
				t_idx = (elapsed_s / 3600.0).floor().to(torch.int64).clamp_(max=gridder.n_t - 1)
				with phase_timer.phase("gridder"):
					gridder.accumulate(
						particles=particles[:, :3],
						active_mask=active_mask,
						weights=particles[:, 3],
						t_idx=t_idx,
						release_idx=release_idx_local,
						dt_seconds=delta_s,
					)
				newly_escaped = active_mask & ~_within_met_domain(particles, cfg)
				alive = alive & ~newly_escaped
				escaped_count = torch.count_nonzero(newly_escaped)
			else:
				active_count = int(torch.count_nonzero(active_mask).item())

				if active_count > 0:
					active_particles = particles[active_mask]
					with phase_timer.phase("met_fetch"):
						met_window, fetched_windows = _get_hourly_met_window(
							reader, cfg, t_cursor, met_cache
						)
					hour_windows += fetched_windows
					if fetched_windows:
						phase_timer.tick_fetch()
					advected_active, t_alpha, wind_mean = _advect_active_particles(
						engine,
						device,
						active_particles,
						met_window,
						t_cursor,
						delta_s,
						particles.dtype,
					)
					particles[active_mask] = advected_active

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
					# Per-particle time-ago bin: each particle measures elapsed time
					# from *its own* release window's end (stage 3 semantics).
					elapsed_s = (release_window_end_offsets_s - cursor_offset_s).clamp_(min=0.0)
					t_idx = (elapsed_s / 3600.0).floor().to(torch.int64).clamp_(max=gridder.n_t - 1)
					gridder.accumulate(
						particles=particles[:, :3],
						active_mask=active_mask,
						weights=particles[:, 3],
						t_idx=t_idx,
						release_idx=release_idx_local,
						dt_seconds=delta_s,
					)

				# Kill particles that left met_domain this step (drop-and-count). They
				# accumulated 0 already (outside output_grid -> dropped by valid_mask);
				# alive excludes them from future steps. Escaped count is a device
				# tensor summed at batch end.
				if active_count > 0:
					newly_escaped = active_mask & ~_within_met_domain(particles, cfg)
					alive = alive & ~newly_escaped
					escaped_count = torch.count_nonzero(newly_escaped)
				else:
					escaped_count = zero_escaped

			step_count += 1

			# Accumulate diagnostics as device tensors (no host sync this step);
			# materialised once per batch below. active_mask.sum() replaces the
			# former per-step active_count .item() in the diagnostics.
			diag_pos_means.append(particles[:, :3].mean(dim=0))
			diag_wind_means.append(wind_mean)
			diag_escaped.append(escaped_count)
			diag_active.append(active_mask.sum())
			diag_host.append((step_count, batch.batch_idx, t_cursor.isoformat()))

			if mem.gc_every_steps > 0 and step_count % mem.gc_every_steps == 0:
				gc.collect()

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

			if step_profiler is not None:
				step_profiler.after_step()

			t_cursor = t_prev

		# Materialise this batch's per-step diagnostics in ONE host transfer
		# (instead of a `.item()` sync every step). `alive_particles` is the
		# running survivor count = N − cumulative escaped, recovered from the
		# escaped-per-step tensor via cumsum.
		if diag_host:
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
						"mean_lon": pm[0],
						"mean_lat": pm[1],
						"mean_alt_agl_m": pm[2],
						"u_mean_m_s": wm[0],
						"v_mean_m_s": wm[1],
						"w_mean_m_s": wm[2],
						"active_particles": ac,
						"escaped_this_step": esc,
						"alive_particles": al,
					}
				)

		# End of batch: stream this batch's footprint slice to disk, then drop the
		# gridder and per-batch tensors so peak footprint memory is one batch's
		# worth, not the whole schedule's. (met_cache + bookkeeping persist.)
		writer.write_footprint_region(
			outputs.footprints,
			gridder.tensor,
			release_start=batch_release_offset,
			release_stop=batch_release_offset + len(batch.releases),
		)
		endpoint_particles_chunks.append(particles.detach().cpu().clone())
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

	metadata = {
		"config": _config_metadata(cfg, schedule_release_end, schedule_sim_start),
		"runtime": _runtime_metadata(cfg, step_count, hour_windows, memory_stats, status="completed"),
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
		},
		"footprint_units": FOOTPRINT_UNITS_DOC,
	}
	writer.write_metadata_json(outputs.metadata, metadata)

	phase_timer.summary(hour_windows=hour_windows)
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
		parser.exit(2, f"Error: unknown turbulence scheme {cfg.turbulence.scheme!r}. Registered: {', '.join(list_schemes())}\n")
	if cfg.convection.scheme not in list_convection_schemes():
		parser.exit(2, f"Error: unknown convection scheme {cfg.convection.scheme!r}. Registered: {', '.join(list_convection_schemes())}\n")

	try:
		metadata = _run(cfg)
	except PreflightValidationError as exc:
		parser.exit(2, f"Error: {exc}\n")
	print("LPDM run complete")
	print(metadata["outputs"])


if __name__ == "__main__":
	main()
