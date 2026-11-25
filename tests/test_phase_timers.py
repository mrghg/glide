"""The whole-run phase accountant (`lpdm.phase_timers`, task A1)."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

from lpdm import phase_timers
from lpdm.phase_timers import PhaseTimer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "agent"))
import benchmark  # noqa: E402


def _timer() -> PhaseTimer:
    return PhaseTimer(enabled=True, sync_cuda=False, log_every=0)


def test_nested_phases_are_charged_exclusive_time() -> None:
    """A sub-phase's time is charged to it and taken off its parent, so the rows add
    up to the time spent with nothing counted twice; the parent's inclusive total
    keeps the whole."""

    t = _timer()
    with t.phase("outer"):
        time.sleep(0.02)
        with t.phase("inner"):
            time.sleep(0.03)
        frame = t.start("inner")
        time.sleep(0.01)
        t.stop(frame)
    assert t.counts["inner"] == 2 and t.counts["outer"] == 1
    assert t.totals["inner"] == pytest.approx(0.04, abs=0.02)
    assert t.totals["outer"] == pytest.approx(0.02, abs=0.015)
    assert t.inclusive["outer"] == pytest.approx(t.totals["outer"] + t.totals["inner"])
    assert t.parents == {"outer"}


def test_phases_must_close_innermost_first() -> None:
    t = _timer()
    a = t.start("a")
    t.start("b")
    with pytest.raises(RuntimeError, match="out of order"):
        t.stop(a)


def test_disabled_timer_records_nothing() -> None:
    t = PhaseTimer(enabled=False, sync_cuda=False, log_every=0)
    with t.phase("x"):
        pass
    t.stop(t.start("y"))
    t.add("z", 1.0)
    assert not t.totals and not t.counts


def test_module_phase_reports_to_the_installed_timer() -> None:
    """Schemes time their parts through `phase_timers.phase`, a no-op until the runtime
    installs an enabled timer."""

    with phase_timers.phase("orphan"):
        pass
    t = _timer()
    phase_timers.install(t)
    try:
        with t.phase("convection"), phase_timers.phase("convection.lift"):
            pass
    finally:
        phase_timers.install(None)
    with phase_timers.phase("after"):
        pass
    assert set(t.counts) == {"convection", "convection.lift"}
    assert t.parents == {"convection"}


def test_table_parses_and_rows_add_up_to_wall() -> None:
    """The printed table parses with the benchmark script, the sub-phase rows are
    rows of their own, and rows plus residual equal the wall time."""

    t = _timer()
    t0 = time.perf_counter()
    time.sleep(0.01)
    t.add("setup", time.perf_counter() - t0)
    with t.phase("loop_overhead"), t.phase("step"):
        time.sleep(0.01)
    text = t.table(hour_windows=3)
    parsed = benchmark.parse_phase_table(text)
    assert parsed["met_fetches"] == 3
    names = [p["phase"] for p in parsed["phases"]]
    assert set(names) == {"setup", "step", "loop_overhead", "(residual/untimed)"}
    total = sum(p["total_s"] for p in parsed["phases"])
    assert total == pytest.approx(parsed["total_wall_s"], abs=0.02)
    # Inclusive totals are printed after the table, not parsed as rows.
    assert "loop_overhead (incl.)" in text
