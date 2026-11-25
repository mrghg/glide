"""Whole-run wall-clock accountant, enabled by ``GLIDE_PHASE_TIMERS``.

Where the step profiler (``GLIDE_PROFILE``) samples ~20 steps inside a SINGLE met
window (so it is blind to per-window met I/O and batch boundaries, the very places
the GPU goes idle), this accumulates wall time across the WHOLE run bucketed by
phase and prints a breakdown. The ``met_fetch`` bucket is exact (synchronous
CPU/IO, GPU uninvolved), so ``met_fetch / wall`` is a faithful idle fraction even
with no CUDA sync.

**Phases nest, and each is charged its exclusive time.** A phase opened inside
another is a sub-phase: its time is charged to it and subtracted from its parent,
so the rows of the table add up to the wall time (less the small untimed
residual) with nothing counted twice. A parent that wraps a whole region
therefore keeps only what none of its sub-phases covers: the runtime wraps each
iteration of the step loop in ``loop_overhead``, whose own time is the Python
overhead between the named phases. Inclusive totals of the phases that have
sub-phases are printed under the table. Schemes time their own parts through the
module-level :func:`phase` (for example ``convection.columns``), which is a
shared no-op unless the runtime has installed an enabled timer.

It also logs a running breakdown every ``log_every`` met fetches, so even a job
that *times out* (no final summary) leaves the split in the log. The ``step``
bucket includes the one-time ``torch.compile`` charged to the first step.

Env:
* ``GLIDE_PHASE_TIMERS``        — enable (truthy)
* ``GLIDE_PHASE_TIMERS_SYNC``   — ``torch.cuda.synchronize()`` at each phase boundary
                                  so GPU work is charged to the phase that issued it.
                                  This SERIALISES the pipeline (perturbs absolute
                                  numbers, cleans the attribution); off by default.
                                  Without it, queued GPU work is charged to whichever
                                  phase next waits for the device.
* ``GLIDE_PHASE_TIMERS_EVERY``  — log running breakdown every N met fetches (default 25)

The timers read the clock and, with ``_SYNC``, wait for the device; they never
touch model tensors, so they cannot change the output.
"""

from __future__ import annotations

import contextlib
import logging
import os
import time
from collections import defaultdict

import torch

LOGGER = logging.getLogger(__name__)

RESIDUAL_LABEL = "(residual/untimed)"


class PhaseTimer:
    """Nested, exclusive-time phase accountant (see the module docstring)."""

    _NULLCTX = contextlib.nullcontext()

    def __init__(self, *, enabled: bool, sync_cuda: bool, log_every: int) -> None:
        self.enabled = enabled
        self.sync_cuda = sync_cuda
        self.log_every = max(0, log_every)
        # Exclusive time per phase (sub-phases subtracted), inclusive time, call count.
        self.totals: dict[str, float] = defaultdict(float)
        self.inclusive: dict[str, float] = defaultdict(float)
        self.counts: dict[str, int] = defaultdict(int)
        # Phases that had at least one sub-phase open inside them.
        self.parents: set[str] = set()
        self.t_start = time.perf_counter()
        self._fetches = 0
        # One entry per open phase: [name, time charged to its sub-phases, start time].
        self._stack: list[list] = []

    def phase(self, name: str):
        """Context manager timing a phase; a cheap shared no-op when disabled."""
        if not self.enabled:
            return self._NULLCTX
        return self._timed(name)

    @contextlib.contextmanager
    def _timed(self, name: str):
        frame = self.start(name)
        try:
            yield
        finally:
            self.stop(frame)

    def start(self, name: str) -> list | None:
        """Open a phase without a ``with`` block; pass the result to :meth:`stop`.

        For a phase that spans a long loop body, where a ``with`` block would
        re-indent the whole body. Returns None (and ``stop`` ignores it) when disabled.
        """
        if not self.enabled:
            return None
        if self.sync_cuda:
            torch.cuda.synchronize()
        frame = [name, 0.0, time.perf_counter()]
        self._stack.append(frame)
        return frame

    def stop(self, frame: list | None) -> None:
        """Close the phase :meth:`start` opened (phases close innermost first)."""
        if frame is None:
            return
        if self.sync_cuda:
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - frame[2]
        if not self._stack or self._stack[-1] is not frame:
            raise RuntimeError(f"phase {frame[0]!r} closed out of order")
        self._stack.pop()
        name = frame[0]
        self.totals[name] += elapsed - frame[1]
        self.inclusive[name] += elapsed
        self.counts[name] += 1
        if self._stack:
            parent = self._stack[-1]
            parent[1] += elapsed
            self.parents.add(parent[0])

    def add(self, name: str, seconds: float, *, calls: int = 1) -> None:
        """Charge ``seconds`` measured by the caller to a top-level phase."""
        if not self.enabled:
            return
        self.totals[name] += seconds
        self.inclusive[name] += seconds
        self.counts[name] += calls

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

    def table(self, *, hour_windows: int = 0) -> str:
        """The phase table (the format ``scripts/agent/benchmark.py`` parses)."""
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
        lines.append(
            f"  {RESIDUAL_LABEL:<22}{resid:>12.2f}{(100.0 * resid / wall if wall else 0.0):>8.1f}%"
        )
        lines.append("=" * 78)
        if self.parents:
            lines.append("Inclusive totals (the phase with its sub-phases):")
            for name in sorted(self.parents, key=lambda n: self.inclusive[n], reverse=True):
                c = self.counts[name]
                incl = self.inclusive[name]
                pct = 100.0 * incl / wall if wall else 0.0
                per = 1000.0 * incl / c if c else 0.0
                lines.append(
                    f"  {name + ' (incl.)':<22}{incl:>12.2f}{pct:>8.1f}%{c:>11}{per:>13.3f}"
                )
        return "\n".join(lines)

    def summary(self, *, hour_windows: int) -> None:
        if not self.enabled:
            return
        print(self.table(hour_windows=hour_windows))


def _truthy(name: str) -> bool:
    return os.environ.get(name, "") not in ("", "0", "false", "False")


def make_phase_timer() -> PhaseTimer:
    """Build the phase timer from env; always returns an object (a no-op when disabled, so
    call sites need no None-checks)."""

    return PhaseTimer(
        enabled=_truthy("GLIDE_PHASE_TIMERS"),
        sync_cuda=_truthy("GLIDE_PHASE_TIMERS_SYNC") and torch.cuda.is_available(),
        log_every=int(os.environ.get("GLIDE_PHASE_TIMERS_EVERY", "25")),
    )


_DISABLED = PhaseTimer(enabled=False, sync_cuda=False, log_every=0)
_current: PhaseTimer = _DISABLED


def install(timer: PhaseTimer | None) -> None:
    """Make ``timer`` the one :func:`phase` reports to (``None``: back to the no-op)."""
    global _current
    _current = timer if timer is not None else _DISABLED


def phase(name: str):
    """Time a phase on the installed timer (a shared no-op when none is enabled)."""
    return _current.phase(name)
