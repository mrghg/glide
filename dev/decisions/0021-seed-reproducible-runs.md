# 0021 — A run's seed makes the whole run reproducible

**Context.** Each release type has a `seed`, and per-release seeds are derived
from it (`seed + release index`). It seeded only the particles' starting
positions, through its own generator (`release_generator.py`). The turbulence
noise (`gpu_engine.py`) and Emanuel convection's random draws, when no generator
is passed, use torch's **default** generators, which nothing seeded. So two runs
with the same seed gave different footprints, on CPU (even single-threaded) and
on GPU. Found during task B1 (20 Nov 2026), where an equivalence test between two
meteorology stores could not be bit-for-bit without it: repeated runs on one
store differed cell by cell by up to tens of percent of a cell's value.

Golden fixtures (A0), parity tests, bisecting a regression, and GLIDE's own
repeat-seed noise floor all need a seed to mean a reproducible run.

**Decision.** When the release has a seed, the runtime seeds torch's default
generators with it (`torch.manual_seed`, CPU and every CUDA device) once per
run, before any particle moves (`main._seed_global_generators`). The seed is
logged and recorded in `run_metadata.json` as `runtime.rng_seed`. With no seed,
nothing is seeded and runs differ, as before.

**Measured** (20 Nov 2026): two seeded runs are bit-identical on CPU and on a
GH200 (Mace Head smoke test, 6 h, compiled once); without the fix the same pair
differs. A test pins this (`test_seeded_runs_are_reproducible`). The random
sequence differs from what earlier runs drew, so results move by sampling noise
only.

**Rationale.** The default generators are the ones that are safe inside the
captured CUDA graph (the per-step physics, decision 0004), so seeding them is the
smallest change that covers every random draw without touching the compiled
code.

**Rejected alternatives.**
- *A dedicated `torch.Generator` passed to the engine and convection:* cleaner
  ownership, but a custom generator inside the captured graph needs explicit
  registration with the graph and risks breaking capture.
- *Seed per batch:* not needed; batches run in a fixed order, and one seed per
  run is easier to record.

**Not covered.** Bit-for-bit equality across different hardware, thread counts or
library versions; GPU reductions with atomic additions may in general depend on
scheduling, although the smoke test repeated exactly.

**Status.** Adopted 2026-11-20 (Matt Rigby), before A0.
