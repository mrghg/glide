# GLIDE architecture & systems-performance notes

This document records *how the code is constructed to get the most out of the
hardware it runs on*. It is engineering/architecture reference (for the model
write-up and for future contributors), complementary to the physics docs
(`docs/turbulence.md`, `docs/convection.md`, `docs/LPDM_physics_spec.md`) and to
`CHECKPOINT.md` (chronological project history). Where this doc and CHECKPOINT
overlap, CHECKPOINT is the dated narrative; this doc is the current-state
synthesis.

---

## 1. The computational shape of the problem

GLIDE is a backward-in-time Lagrangian particle dispersion model in pure
PyTorch. Its workload has three distinct axes of parallelism, and getting the
hardware mapping right means treating each differently:

| Axis | Extent (production) | Character | Maps best to |
|------|--------------------|-----------|--------------|
| **Time steps** (outer loop) | 7,200 steps/batch (5-day window, `dt=60 s`) | **Strictly sequential** — step *n+1* depends on *n* | nothing; this is the irreducible serial depth |
| **Particles** (inner) | 480,000 / batch (24 releases × 20k) | **Data-parallel** (SIMD) — identical OU/advection math per particle | GPU SIMT lanes; CPU SIMD + threads |
| **Releases / batches** | 720 releases, 30 batches | **Task-parallel** (embarrassingly) — independent backward integrations | multi-core / multi-GPU / multi-node fan-out |

The key consequence: **the per-batch problem is "narrow and deep"** — a modest
data-parallel width (hundreds of thousands of particles) driven through an
enormous serial depth (thousands of steps × up to tens of turbulence substeps).
GPUs want "wide and shallow." 480k particles *is* enough width to use a GPU
well; the difficulty is that the serial depth means we pay per-step CPU
orchestration overhead thousands of times, and if the GPU idles between steps
that overhead — not the math — sets the wall-clock.

This framing drives every optimization below.

---

## 2. Hardware targets

| Target | Role | Notes |
|--------|------|-------|
| **48-core x86_64 (el8)** | Local dev, parameter sweeps, CPU production | Thread-count tuned (§4.2); coarse batch-parallelism is the natural fan-out |
| **Isambard AI — NVIDIA GH200 (Grace-Hopper)** | GPU production runs | aarch64; unified CPU+GPU memory; `module load cuda/12.6` (+ `cudatoolkit/24.11_12.6`, `gcc-native/14.2` for `torch.compile`, §6) |
| **Isambard 3 — NVIDIA Grace (2×72-core)** | Large CPU fan-out | aarch64; good for task-parallel batch arrays |
| **NVIDIA L4 on GCP Cloud Run** | Production deployment target | per-request footprints; same launch-overhead profile as any small-batch GPU run — the §5 work benefits this directly |

Device-agnosticism is a core principle: the same code runs on `cuda → mps → cpu`
(dynamic fallback in `lpdm.runtime`). All optimizations must preserve a working,
**numerically-reference** CPU/eager path (see §7).

---

## 3. Module map (compute-relevant)

- **`met_reader.py`** — streams ARCO ERA5 Zarr subsets; geopotential→AGL; omega→w.
  Returns `hour_start`/`hour_end` 3D tensors for per-step temporal interpolation.
- **`gpu_engine.py`** — device-safe primitives: RK2 backward advection, OU/Langevin
  velocity update (`_ou_step_kernel`), horizontal/vertical turbulent displacement,
  surface reflection. The elementwise hot paths live here.
- **`turbulence/`** — `TurbulenceScheme` ABC + registry; `hanna.py` is production
  (free-function physics for `σ_w`, `T_L`, drift, Richardson closure, meander;
  per-particle adaptive substepping). **Physics math is kept as standalone,
  unit-testable free functions** — this is a hard design constraint (§7).
- **`convection/`** — `ConvectionScheme` ABC + registry; Emanuel reduced port.
  Runs **once per met-window**, not per step (per-column mass-flux matrix depends
  only on the T/q profile, constant within a window).
- **`footprint_gridder.py`** — on-the-fly Eulerian accumulation via `scatter_add_`
  on fractional indices (`bucketize` for non-uniform z-edges); device-resident.
- **`main.py`** — the runtime: batch loop → per-step cursor loop → advect → scheme
  step → reflect → accumulate → escaped-particle drop-and-count.

---

## 4. Performance ladder (implemented)

Ordered roughly by when each became the binding constraint.

### 4.1 Vectorization & coordinate hygiene
- All runtime-critical paths are vectorized tensor ops — **no per-particle Python**.
- Internal physics operates in **geometric metres (AGL)**, converting once, so the
  hot loop never touches pressure coordinates.
- **F9 vertical interpolation**: `grid_sample`'s vertical axis uses a piecewise-
  linear fractional-level lookup against the per-level AGL array (pressure levels
  are ~log-linear in altitude). *This closes the M3 roadmap "vertical-interpolation
  approximation" item.*

### 4.2 CPU thread tuning
- Per-step tensors (10⁴–10⁵ particles) are too small to feed 48 cores — lock
  contention dominates first. **Measured optimum ≈ 16 torch threads**; 48 was the
  *slowest* (~25% worse). `main._configure_cpu_threads` reads `GLIDE_NUM_THREADS`;
  SLURM scripts default to 16 with `--cpus-per-task=16`.
- Corollary: on CPU, **throughput comes from running many batches concurrently**
  (one ~16-thread process each), not from one wide run — the task-parallel axis.

### 4.3 Per-window field caching
- `_meander_sigma_fields`, `_density_fields`, `_free_trop_fields` depend only on
  the hourly met window (~60 steps), not the step. Built **once per window at the
  window midpoint** (`t_alpha=0.5`), cached by `metadata.time_start`. `avg_pool2d`
  calls dropped 796→32 over 200 steps (~20% faster at 20k particles). Deliberate,
  documented met-cadence approximation (<1%/hr drift).

### 4.4 Host-sync minimization (the CUDA enabler)
On CUDA, every `.item()` / `bool(tensor)` / `torch.any(...).item()` is a
device→host sync that stalls the pipeline. The Hanna substep loop was firing
*hundreds* per step. Reduced to **3 per step**:
- Engine value-validation gated behind `VALIDATE_ENGINE_INPUTS` (off by default).
- Dead-code per-substep `bool(mask.any()): break` removed.
- FT-override `bool(torch.any(above_bl))` replaced by unconditional `torch.where`.
- Per-step diagnostics accumulated as device tensors, materialized **once per
  batch**.
- **Remaining 3 syncs:** `max_k` (bounds substep loop), `scheme.step` empty-active
  guard, `active_count` (control flow in `_run`). These are the ones the CUDA-graph
  work (§5) must eliminate.

### 4.5 Active-set shrinking (CPU-favourable, GPU-graph-hostile)
- **F4T2 adaptive substepping**: each particle integrates `k_i = ceil(dt /
  (substep_c·T_Lw_i))` substeps (capped at `max_substeps`); masking touches only
  particles still owing substeps. Exposed as `turbulence.substep_c` /
  `turbulence.max_substeps` (also a perf/accuracy knob and a launch-bound diagnostic).
- **M4 drop-and-count**: particles leaving `met_domain` clear an `alive` bit and
  are never advected/accumulated again — the active set shrinks over the 30-day
  window.
- Both rely on **dynamic shapes**. They are wins on CPU (cheap masking, less work)
  but are in direct tension with CUDA graphs (§5), which require static shapes.

### 4.6 `torch.compile` of hot paths (opt-in)
- `GPUEngine(compile_hot_paths=…)` / `GLIDE_COMPILE=1` wraps the elementwise hot
  paths with `torch.compile(dynamic=True)` (dynamic to tolerate the changing
  substep-subset size). `suppress_errors=True` → per-graph eager fallback, so it
  can never hard-fail a run. Inductor/Triton codegen needs a CUDA toolkit **and a
  C++20 host compiler** (§6).
- **Diagnosis status (2026-06-19): launch/orchestration-bound.** With sync removal
  but no graph capture, GH200 `sm%` sits at 0–37%. Cutting `max_substeps` 50→5
  *halved wall time* while `sm%` stayed flat — the signature of launch-bound:
  cutting *launches* (CPU work) cut wall-clock, cutting GPU *math* would not have.
  480k particles is adequate width, so the idle is per-step CPU orchestration
  bubbles, not insufficient parallelism. `dynamic=True` compile fuses *within* each
  method call but cannot fuse *across* the Python substep loop — hence §5.

---

## 5. Planned: CUDA-graph capture of the per-step body (Milestone 3, in progress)

**Goal:** collapse per-step CPU→GPU launch/orchestration overhead so the GH200
runs near compute-bound (target `sm%` 60–80%, ~roughly halving per-batch time),
and the same fix carries to the L4 production target.

**Why graphs, not more `torch.compile`:** the bubbles are *between* kernels
(per-step Python orchestration, the 3 residual host syncs, sequential launch
latency), not inside them. CUDA graphs record the whole per-step launch sequence
once and replay it with a single host call. `torch.compile(mode="reduce-overhead")`
emits CUDA-graph trees via Inductor.

**Hard requirements (and what each forces):**
1. **Static shapes** → run the full particle set every substep with `sub_dt=0`
   (a math no-op: `a=exp(0)=1`, `variance=0`, `drift·0=0` ⇒ state unchanged) for
   particles past their `k_i`. Mask by *multiply*, never by *index*.
2. **Fixed loop count** → run a fixed `max_substeps` iterations every step (not a
   data-dependent `max_k`); this *removes* the `max_k` sync naturally.
3. **No host syncs in the captured region** → eliminate the remaining 3 (the
   `scheme.step` empty-guard and `active_count` move outside capture or become
   mask-multiplies).
4. **Graph-safe RNG** → register the generator with the graph so `torch.randn`
   replay stays statistically valid and seed-controllable.
5. **Capture only the per-step transport** → met fetch, convection (once/window),
   and per-window field rebuilds (§4.3) stay *outside* the graph. The existing
   per-window/per-step separation makes this clean.

**The central tradeoff (decision required — see §5.1):** requirements 1–2
discard the §4.5 active-set-shrinking optimizations. The captured path does
*more total FLOPs* (every particle runs to `max_substeps`; escaped particles keep
being processed) in exchange for far fewer launches. Net win **iff** launch
overhead > the added compute. For a 30-day backward run where a large fraction of
particles escape early, the escaped-particle waste is not negligible.

### 5.1 Open decision: how to reconcile graphs with the escaped-particle drop
Candidate strategies (to be chosen by measurement on the single-release smoke
config + one full batch):
- **(A) Accept the waste.** Simplest. Capture once; process all 480k every step
  for the whole window. Best if launch overhead dominates and escape fraction is
  modest within the 5-day batch window.
- **(B) Periodic recapture / compaction.** Every *K* met-windows, drop escaped
  particles outside the graph and re-capture at the new (smaller) static size.
  Recovers most of the M4 savings; modest complexity (graph lifecycle management).
- **(C) Bucketed sizes.** Maintain a few pre-captured graphs at power-of-two
  particle counts; step down as the active set shrinks. More machinery; smoother
  than (B).
- **(D) Device-gated.** Keep the dynamic-shape CPU path untouched (it's optimal
  there) and only build the static/graph path for CUDA. Orthogonal to A–C and
  almost certainly required regardless, to avoid slowing CPU sweeps.

**Default recommendation:** implement **(D) + (A)** first (device-gated static path,
accept the waste), measure against the eager-dynamic baseline, and only escalate
to **(B)** if the escaped-particle waste shows up in the profile. Measure before
adding graph-lifecycle complexity.

### 5.2 Phasing
1. **✅ DONE (2026-06-19) — Static-shape substep loop, device-gated** (CPU keeps
   dynamic masking). Full set + `sub_dt=0` no-ops for finished particles. Validated
   against the dynamic path (bit-identical for homogeneous `k`; WMC gate parametrized
   over both paths). *No graphs yet.*
2. **✅ DONE (2026-06-19) — Full-set per-step path, device-gated (= strategy (D)+(A)).**
   `HannaScheme.step` and the `main._run` cursor loop now process the **full**
   particle buffer on the static path and gate inactive particles with `torch.where`
   (no boolean indexing, no `active_count`/empty-guard `.item()`). The shared
   `use_static_step_path(device, override)` (in `gpu_engine.py`) keys the gate; the
   dynamic CPU branch is byte-for-byte the pre-phase-2 runtime. The `active_count`
   diagnostic is now a per-step device tensor (`active_mask.sum()`), materialized per
   batch. *Remaining per-step sync on the static path:* `max_k` only (phase 3).
   Convection stays outside the per-step path (once per met-window) by design.
3. **✅ DONE (2026-06-19) — `mode="reduce-overhead"` graph capture wired.** On the
   static path with `GLIDE_COMPILE`, the **fixed-count** substep loop
   (`n_substeps=max_substeps`, a compile-time-constant trip count → removes the last
   per-step sync, `max_k`) is wrapped with `torch.compile(mode="reduce-overhead",
   dynamic=False)` so it captures as one CUDA graph (`HannaScheme._maybe_graph_compile`).
   Per-method engine compile is suppressed on this path to avoid nesting
   (`GPUEngine._compile_requested` vs `_compile_hot_paths`). Graph-safe RNG is handled
   by Inductor's cudagraph trees. Capture boundary is the substep loop — met fetch,
   convection, and per-window field rebuilds are computed in `step` *outside* it and
   passed in as tensors, so they never enter the graph. **CPU-verified**: fixed-count
   == variable-count (bit-identical), the loop compiles + runs (smoke test); **CUDA
   graph capture + sm% is pending the GH200 run** (CUDA-only — see §6 / the SLURM
   script's "WHAT TO CHECK").
4. **Measure** `sm%` + per-batch time on the GH200; decide on §5.1 (B) only if needed.
5. **Document** the benchmark + dominant remaining bottleneck (M3 exit criterion).

**Status:** phases 1–3 landed (strategy **(D)+(A)** implemented and the graph-capture
path wired). The remaining work is **phase 4 — measurement on the GH200** (the sm%
payoff, and whether the §5.1 (B) escalation is needed). On the graph path
`max_substeps` is the fixed per-step iteration count, so it should be tuned down
(~15–25) from the default 50.

---

## 6. Build / toolchain notes (Isambard AI)

`torch.compile` (Inductor/Triton) needs **two** toolchain pieces; loading one
without the other makes every kernel silently fall back to eager:
- `cudatoolkit/24.11_12.6` — `nvcc`/`ptxas` for the Triton GPU backend.
- `gcc-native/14.2` — a **C++20 host compiler** for Inductor's C++ codegen.
  Inductor compiles a C++ glue layer + precompiled header with the host compiler
  *even on the GPU path*. The system GCC 8.x rejects `-std=c++20` (knows only
  `-std=c++2a`), so the precompiled-header build fails and compilation never
  happens. Point Inductor at the new compiler: `export CC=gcc; export CXX=g++`.
- **Confirmation it worked:** the `torch._dynamo` "WON'T CONVERT" warnings
  disappear (compilation ran rather than falling back).
- **Diagnosis tip:** `suppress_errors=True` hides the real exception. Reproduce
  with it off (`torch._dynamo.config.suppress_errors = False`) and call the
  compiled fn directly to see the true error at the bottom of the traceback.

`scripts/run_periodic_cuda.slurm` wires this up (active when `GLIDE_COMPILE != 0`)
and runs `nvidia-smi dmon -s um` telemetry to a sidecar log with an end-of-run
mean/max `sm%` summary.

---

## 7. Invariant: physics stays interrogable

Performance work must not make the physics opaque. Non-negotiable constraints:
- **Free-function physics** (`hanna.py`'s `σ_w`/`T_L`/drift/Richardson/meander,
  the convection thermodynamics) stay as standalone, unit-testable functions.
  Graph capture wraps the *assembled* per-step call; it does **not** rewrite or
  inline-away the physics functions.
- **The eager path is the numerical reference.** Compiled/graph paths are opt-in
  accelerations validated *against* eager (existing
  `test_compiled_hot_paths_match_eager_*` + the WMC/no-runaway tests are the
  acceptance gate). Eager must remain runnable for debugging and on CPU/MPS.
- **`TurbulenceScheme` / `ConvectionScheme` ABC + registry** stay intact — schemes
  remain swappable, and a new scheme inherits the runtime's performance machinery
  without bespoke plumbing.
- **Knobs stay exposed in config** (`substep_c`, `max_substeps`, …) so physics can
  be tuned and evaluated without code edits.

---

## 8. Quick reference — what to reach for

- **Run faster on CPU now:** `GLIDE_NUM_THREADS=16`, fan out batches as a SLURM
  array (one process per batch).
- **Run on GH200:** `scripts/run_periodic_cuda.slurm` (compile on by default).
- **Diagnose GPU under-utilization:** read the `*.gpu.log` `sm%` summary; low and
  bursty ⇒ launch-bound ⇒ §5. Cross-check by lowering `max_substeps` — if wall
  time drops but `sm%` doesn't, it's launch-bound, not compute-bound.
- **Confirm `torch.compile` actually engaged:** no "WON'T CONVERT" in the `.err`
  log; first step noticeably slower (one-time compile).
