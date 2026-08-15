# Architecture

GLIDE exists to test a proposition: that a backward LPDM can be made *scalable*
and *flexible* by rethinking its two traditional bottlenecks — file-based
meteorology I/O, and single-threaded CPU physics. This page describes how the
code is put together to do that, and what the shape of the problem forced.

The physics it implements is in [physics.md](physics.md),
[turbulence.md](turbulence.md) and [convection.md](convection.md).

**Contents**

1. [The shape of the problem](#1-the-shape-of-the-problem)
2. [Module map](#2-module-map)
3. [Anatomy of a run](#3-anatomy-of-a-run)
4. [The meteorology pipeline](#4-the-meteorology-pipeline)
5. [The per-step path, and CUDA graphs](#5-the-per-step-path-and-cuda-graphs)
6. [Memory](#6-memory)
7. [Invariants](#7-invariants)
8. [Diagnostics and knobs](#8-diagnostics-and-knobs)

---

## 1. The shape of the problem

A backward LPDM run has three axes of parallelism, and they want completely
different hardware treatment:

| Axis | Typical extent | Character | Maps to |
| --- | --- | --- | --- |
| **Time steps** | 7,200 per batch (5-day window at $\Delta t = 60$ s) | strictly sequential — step $n{+}1$ depends on $n$ | nothing; irreducible serial depth |
| **Particles** | ~5×10⁵ per batch | data-parallel; identical arithmetic per particle | GPU SIMT lanes; CPU SIMD |
| **Releases / batches** | hundreds to thousands | embarrassingly parallel; independent integrations | multi-core / multi-GPU / multi-node |

The consequence is that **the per-batch problem is narrow and deep**: a modest
data-parallel width driven through an enormous serial depth. GPUs want wide and
shallow. Half a million particles *is* enough width to use a GPU well — the
difficulty is that the serial depth means per-step CPU orchestration is paid
thousands of times, and if the GPU idles between steps then that overhead, not
the arithmetic, sets the wall-clock.

That framing explains most of the design decisions below, in particular §5.

---

## 2. Module map

| Module | Responsibility |
| --- | --- |
| `main.py` | Orchestration: config → schedule → batch loop → cursor loop → outputs. Also the CLI. |
| `config.py` | Pydantic run-config schema, validation, and schedule/batch expansion. |
| `met_reader.py` | Streams ERA5 (or any conforming store) from Zarr; unit checks, ω→w, geopotential→AGL, terrain-following resample, per-hour cache. |
| `vertical_grid.py` | The terrain-following resample kernels: AGL ladder construction, per-column regrid weights, terrain slope, $w$ transform, hydrostatic pressure on model levels. |
| `gpu_engine.py` | Device-safe primitives: coordinate normalisation, RK2 advection, the OU/Langevin kernel, turbulent displacement, surface reflection. |
| `turbulence/` | `TurbulenceScheme` ABC + registry; `hanna.py` is production. |
| `convection/` | `ConvectionScheme` ABC + registry; `emanuel.py` is production. |
| `release_generator.py` | Particle generation: point, volume, column, flight-track; per-batch assembly with release indices and time offsets. |
| `footprint_gridder.py` | On-the-fly Eulerian accumulation onto the 5-D output grid via `scatter_add_`. |
| `output_writer.py` | Streaming Zarr footprint store + Parquet endpoint/diagnostic writers. |
| `comparison.py` | STILT-unit conversion and conservative regridding, for comparison against other models. |
| `runtime.py` | Device selection with CUDA → MPS → CPU fallback. |

Everything is **pure PyTorch and device-agnostic**. There are no custom CUDA
kernels; the same code runs on a laptop CPU and on a GH200.

---

## 3. Anatomy of a run

```
RunConfig (YAML)
   │
   ├─ expand_to_batches()          releases → batches of ≤ max_releases_per_batch
   ├─ preflight                    met time coverage spans the whole schedule?
   │                               every release inside met_domain?
   │
   └─ for each batch:
         generate particles        (n_releases × n_particles, one buffer)
         allocate gridder          (n_releases, T, Z, Y, X) on device
         │
         └─ cursor loop, backward from max(window_end) to min(window_end) − length:
               fetch met window    (LRU + background prefetch)
               RK2 advection       ─┐
               turbulence step      ├─ one fused call on the GPU path
               (reflection)        ─┘
               convection          once per met hour
               accumulate          scatter_add into the gridder
               drop escapees       clear liveness bit outside met_domain
         │
         write footprint region → footprints.zarr, free the gridder
```

**Particles are generated up front for the whole batch** and held in one
contiguous `(N, 4)` buffer. Each particle knows its release index and its release
time (drawn uniformly within its release window), so a single sweep integrates
every release in the batch correctly: the active mask is
`released_yet AND still_inside_backward_window AND alive`.

**Releases are the outermost axis** and it is flat — one entry per (site × time).
That is deliberate: it accommodates future release geometries (column releases,
satellite soundings with per-sounding averaging kernels) without changing the
output contract. A per-site cube is recovered downstream with
`.set_index(release=["site", "release_time"]).unstack("release")`.

**Multi-site releases share meteorology.** With `multi_point_periodic`, every
site releases on the same schedule, so all sites' particles occupy the same
meteorology windows. Every per-window fixed cost — the met fetch, the convection
matrix, the support-field build, the per-step Python overhead — is then paid once
for the whole network instead of once per site. This is the intended way to scale
a run across a station network, and it is why the reference run in the README
does 56 sites × 48 hours in a single process.

Outputs land under `io.output_uri`:

| File | Contents |
| --- | --- |
| `footprints.zarr` | the 5-D footprint store, written region-by-region as each batch finishes |
| `endpoint_particles.parquet` | final particle states |
| `trajectory_diagnostics.parquet` | per-step ensemble diagnostics |
| `run_metadata.json` | provenance, config echo, timings, escape counts, any memory-guard report |

---

## 4. The meteorology pipeline

This is the half of the project that addresses the I/O bottleneck. Rather than
reading whole NetCDF/GRIB files, GLIDE treats the archive as something to sip
from: it asks for **one hour, over the particle cloud's bounding box, up to its
vertical ceiling**, and the chunked Zarr layout means only the chunks that
intersect that box are decompressed.

Per hour, the reader:

1. selects the bounding box and the vertical levels that intersect the requested
   AGL range;
2. validates units on every variable (a missing or unrecognised `units` attribute
   is a hard error — GLIDE will not guess);
3. converts ω → geometric $w$ where the source supplies a pressure tendency,
   $w = -(R_d T)/(g p)\,\omega$;
4. de-accumulates surface fluxes if they arrive as J m⁻², and flips the ECMWF
   heat-flux sign convention;
5. derives height above ground from geopotential, $(z - z_{sfc})/g$;
6. **resamples the whole hour onto the fixed terrain-following AGL ladder**
   (`vertical_grid`), excluding sub-surface levels, and slope-corrects $w$ into
   that frame;
7. packs the result into a `[C, Z, Y, X]` tensor with a name→channel index.

On model (hybrid) levels the level coordinate is an index rather than a pressure,
so per-level pressure is reconstructed hydrostatically from the archive's own
geopotential, surface pressure, temperature and humidity — no hybrid $a/b$
coefficient table required, deliberately, since ARCO does not ship one. See
[dev/decisions/0010](../dev/decisions/0010-model-level-met-reader.md).

The full input contract is [met_schema.md](met_schema.md).

### Three layers of caching

Step 6 is the expensive one — profiling a window fetch put `_resample_hour_to_agl`
at ~74% of it (and `compute_agl_regrid_weights` alone at ~50%), against only ~17%
for the actual dask read. Three caches sit on top of that:

| Layer | Keyed by | Why |
| --- | --- | --- |
| **Per-hour processed cache** (reader, 6 entries) | met hour | Adjacent windows share a boundary hour. Without this, every physical hour is read, converted and regridded **twice**. |
| **Window LRU** (runtime, `memory.met_cache_max_hours`) | window start | Consecutive batches re-walk overlapping backward windows. Under-size it and you get repeated re-fetch thrash; the run warns at startup if it is too small. |
| **Per-window derived fields** (scheme) | window start | The density, free-troposphere and meander support stacks are built **once at the window midpoint** ($\alpha = 0.5$) and reused for every step in that hour. |

The third is a physics approximation, and a deliberate one: these fields drift by
under a percent per hour, and FLEXPART-class models likewise refresh turbulence
fields at the meteorology step rather than the particle step. It removed roughly
18% of an earlier CPU profile.

A **background prefetch thread** fetches the next backward hour while the GPU
computes the current one, so the I/O is hidden behind compute. It requires a
host-resident reader (the worker must not issue CUDA calls) and is disabled with
a warning if `met_cache_on_host` is false.

The regrid weights themselves are also split out and computed once per window
rather than once per tensor per level — the bracketing depends only on the level
heights and the target grid, not on the field being regridded. That alone cut the
per-window regrid CPU cost by about an order of magnitude, which mattered because
at full-domain scale the unsplit version could no longer hide behind the GPU.

---

## 5. The per-step path, and CUDA graphs

Recall §1: the per-batch problem is narrow and deep, so the risk is that per-step
CPU orchestration, not arithmetic, sets the wall-clock. On the GH200 that risk
was realised — profiling showed **~1,250 kernel launches per step, GPU busy ~17%,
~2.5 ms of GPU work inside a ~30 ms step**. The GPU was idling in the gaps
between kernels. The diagnostic signature is characteristic: cutting
`max_substeps` from 50 to 5 halved the wall time while `sm%` stayed flat. Cutting
*launches* helped; cutting *arithmetic* would not have.

`torch.compile` alone cannot fix this. It fuses within a call but cannot fuse
across a Python loop. The fix is to record the whole launch sequence once and
replay it with a single host call — a **CUDA graph**.

### Two execution paths

GLIDE therefore carries two per-step implementations, selected by device
(`use_static_step_path`, shared by the scheme and the runtime loop so they always
agree):

| | **Dynamic** (CPU / MPS default) | **Static** (CUDA default) |
| --- | --- | --- |
| Active particles | boolean-indexed subset | full buffer, every step |
| Inactive particles | not touched | gated by `torch.where`; sub-steps run with `sub_dt = 0` |
| Sub-step loop | to `max_k` (one host sync) | fixed `max_substeps` iterations |
| Host syncs per step | a few; cheap here | none inside the captured region |
| Role | **the numerical reference** | the throughput path |

`sub_dt = 0` is a mathematical no-op, which is what makes the static path
correct: $a = e^0 = 1$ leaves the velocity unchanged, $\sigma^2(1-a^2) = 0$ kills
the noise, $w'\cdot 0 = 0$ leaves the position unchanged, and reflecting an
already-non-negative $z$ is the identity. Every particle still integrates exactly
its own $k_i$ real sub-steps; the remainder are no-ops. Masking is by
**multiplication, never by indexing** — that is what keeps the tensor shapes
constant.

The trade is explicit: the static path does *more* arithmetic (finished and
escaped particles keep being processed) in exchange for far fewer launches. It is
a net win precisely because launch overhead dominated. On CPU, where launches are
free, the trade inverts — hence the device gate.

### What gets captured

On the static path with `GLIDE_COMPILE=1`, the whole per-step body is a single
`torch.compile(mode="reduce-overhead")` target: RK2 advection, met interpolation,
column turbulence, the drift, the sub-step loop, meander, and the mask-gated
write-back. Met fetch, convection, and the per-window field rebuilds stay
*outside* it — the existing per-window/per-step separation makes that boundary
clean.

Measured on the GH200: per-step **30 → 5 ms**, launches **~1,250 → ~106**, GPU
busy **17% → 37%**.

Four constraints make the capture actually hold, and all four were learned by
breaking them:

1. **No graph breaks.** Any data-dependent Python control flow inside the core
   splits the graph. A single `bool(level_arr[-1] > level_arr[0])` — reached five
   times per step — was enough to leave performance unchanged after the first
   capture attempt.
2. **No recompiles.** Dynamo specialises on Python scalars, so any value that
   changes between steps or between meteorology windows must be passed as a
   *tensor*, not a float or tuple. The per-step time-interpolation weight
   $\alpha$ and the per-window AGL level array both had to become tensors. Miss
   this and you blow the recompile limit and silently fall back to eager — 4×
   *slower* than before.
3. **Stable input addresses.** Graph replay copies any input whose storage
   address changed since capture into the graph's staging buffers. A fresh
   `.to(device)` per meteorology window therefore re-staged the large
   window-constant tensors on *every step* — ~49% of GPU time. Every tensor input
   now lives in a persistent buffer marked `mark_static_address` and is refilled
   in place, outside the captured region. (Watch the subtlety that
   `torch.device("cuda") != torch.device("cuda:0")`; a naive comparison
   reallocated a buffer every step and forced a per-step re-record.)
4. **Clone outputs that outlive the call**, and call
   `cudagraph_mark_step_begin()` before each invocation — graph outputs alias
   buffers that the next replay overwrites.

The first two are caught **on CPU, in CI**, with no GPU needed:
`test_step_core_traces_as_one_graph_no_breaks` compiles with `fullgraph=True,
backend="eager"` and raises on any break;
`test_step_core_does_not_recompile_per_step` sets
`torch._dynamo.config.error_on_recompile` and steps through changing $\alpha$,
met values, and level arrays. Both regressions destroy the capture silently, so
they are worth the guard.

### Where the time goes now

With the per-step path handled, it is no longer the wall. On a representative
multi-site GH200 run the per-step phase is roughly 18–23% of total; the
remaining large consumers are meteorology I/O, convection, and an untimed
per-step "residual" (diagnostics, mask bookkeeping, per-batch particle generation
and output writes, Python loop overhead). Further step-side graph work is
low-priority against those — see [../STATUS.md](../STATUS.md).

---

## 6. Memory

A batch is the unit of both memory and wasted work. It holds all its releases'
particles in one buffer and one dense footprint tensor:

$$
\text{peak} \;\approx\; n_{\text{releases/batch}} \times \Big( n_{\text{particles}} \times 112\ \mathrm{B} \;+\; \text{footprint bytes per release} \Big)
$$

**Footprints are streamed.** The Zarr store is created once, sized for all
releases, and each batch writes its own region and frees its gridder — so only
one batch's tensor is ever resident. A single batch containing every release
defeats this entirely (40,320 releases at one z-integrated store is 51.6 GiB of
footprint plus ~29 GiB of particles).

**The compute-optimal batch is the "active window"** — the set of releases whose
backward windows overlap the cursor at peak,
$\lceil \text{length}/\text{period} \rceil \times n_{\text{sites}}$. Larger
batches step ever more inactive
particles; smaller batches trade device memory for a larger *host* met cache,
since consecutive batches re-read the overlapping backward meteorology.
`scripts/make_multisite_config.py` does this arithmetic, sizing the batch to the
active window (capped by a GPU memory budget) and the met cache from the run
geometry.

**The met cache lives in host RAM by default** (`met_cache_on_host: true`). Each
window is a `[C, Z, Y, X]` stack of hundreds of MiB, so a 192-hour cache is
around 50 GiB — on a GH200 that is LPDDR5X rather than HBM, leaving the device
free for compute. Note the cache scales *linearly with the AGL level count*: on
the EUROPE domain at 192 hours, 23 levels is ≈52 GiB but 40 levels is ≈90 GiB.

**Guards are fail-fast.** `guard_max_rss_gib`,
`guard_max_device_allocated_gib` and `guard_max_device_reserved_gib` are optional
hard limits checked every `guard_check_every_steps`; a tripped guard raises
`MemoryError` and writes diagnostics into `run_metadata.json` rather than dying
in the allocator. The footprint gridder separately refuses to allocate a tensor
above 32 GiB (`LPDM_FOOTPRINT_MAX_GIB`) with a message explaining which dimension
to cut.

**Rejected: single-sweep footprint streaming.** Keeping one sweep over all
releases and flushing each footprint as its window closes would bound *footprint*
memory but not *particle* memory, and on the static GPU path would step roughly
6× the particles (only about a sixth are active at once). Bounding particle
memory too would require a dynamic, shrinking particle buffer — which breaks the
static-shape capture the GPU throughput depends on. Batching is the memory lever
precisely because the graph strategy fixes the shape.

---

## 7. Invariants

Performance work must not make the physics opaque. These are treated as hard
constraints, not preferences:

- **The physics stays in free functions.** $\sigma_w$, $T_L$, the drift, the
  Richardson closure, the meander stencil, the convection thermodynamics — each
  is a standalone, unit-testable function. Graph capture wraps the *assembled*
  step; it never rewrites or inlines these away.
- **The eager path is the numerical reference.** Compiled and graph paths are
  opt-in accelerations validated against eager. Eager must stay runnable for
  debugging and on CPU/MPS.
- **Schemes stay swappable.** The `TurbulenceScheme` / `ConvectionScheme` ABCs
  and registries stay intact, so a new scheme inherits the runtime's performance
  machinery without bespoke plumbing.
- **Knobs stay in config**, not in code, so physics can be tuned and compared
  without edits.
- **Memory guards stay fail-fast**, and long-running loops get no unbounded
  caches.

---

## 8. Diagnostics and knobs

Environment variables, all optional:

| Variable | Effect |
| --- | --- |
| `GLIDE_COMPILE=0/1` | Disable / enable `torch.compile` + graph capture. Off skips Triton compile cost — handy for debugging. |
| `GLIDE_STATIC_SUBSTEPS=0/1` | Force the static or dynamic per-step path, overriding the device gate. |
| `GLIDE_PROFILE=1` | Capture ~20 cursor-loop steps with `torch.profiler`, print GPU-busy %, host-sync ops and top ops with call counts, write a Chrome trace, then exit. Tune with `GLIDE_PROFILE_STEPS` / `_WARMUP` / `_TRACE` / `_CONTINUE`. |
| `GLIDE_PHASE_TIMERS=1` | Whole-run wall breakdown by phase (met_fetch / advect / step / convection / gridder). `_EVERY`, `_SYNC` tune it. |
| `GLIDE_NUM_THREADS=N` | torch intra-op threads (CPU only). ~16 measured optimal on a 48-core node; 48 was the *slowest* option, ~25% worse — the per-step tensors are too small to outrun lock contention. |
| `GLIDE_VALIDATE_ENGINE=1` | Re-enable per-call value validation in the engine hot paths. Off by default: each check is a device→host sync, and the sub-step loop calls them hundreds of times per step. |
| `GLIDE_MEM_SNAPSHOT=1` | Dump a CUDA memory snapshot (`_BATCH`, `_PATH` tune it). |
| `LPDM_FOOTPRINT_MAX_GIB` | Raise the footprint-tensor allocation cap above 32 GiB. |

**Reading a GPU run.** Low and bursty `sm%` in the `*.gpu.log` summary means
launch-bound. Cross-check by lowering `max_substeps`: if wall time drops but
`sm%` does not, it is launches, not arithmetic. In a `GLIDE_PROFILE` trace, many
small GPU ops means launch-bound; `cudaStreamSynchronize` means a residual host
sync; long CPU spans mean Python- or met-bound; and CPU dominated by
`dynamo_timed`/`fx_codegen_and_compile` with `aten::*` running thousands of times
means recompile thrashing — a per-step Python scalar has reached the compiled
core (constraint 2 of §5).

**Confirming compilation engaged.** No `WON'T CONVERT` warnings in the error log,
and a noticeably slower first step (the one-time compile).

**Toolchain, Isambard AI.** Inductor needs *two* pieces, and loading one without
the other makes every kernel silently fall back to eager: `cudatoolkit/24.11_12.6`
for `nvcc`/`ptxas`, and `gcc-native/14.2` for a C++20 host compiler (Inductor
compiles a C++ glue layer even on the GPU path; the system GCC 8.x rejects
`-std=c++20`). Export `CC=gcc; CXX=g++` so Inductor picks up the right one.
`scripts/run_periodic_cuda.slurm` wires all of this up.
