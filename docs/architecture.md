# Architecture

GLIDE exists to test a proposition: that a backward LPDM can be made *scalable*
and *flexible* by rethinking its two traditional bottlenecks — reading
meteorology from large files, and running the physics on a single CPU core. This
page describes how the code is put together to do that, and what the shape of the
problem forced.

The physics it implements is in [physics.md](physics.md),
[turbulence.md](turbulence.md) and [convection.md](convection.md).

**Contents**

- [A little vocabulary](#a-little-vocabulary)
1. [The shape of the problem](#1-the-shape-of-the-problem)
2. [Module map](#2-module-map)
3. [Anatomy of a run](#3-anatomy-of-a-run)
4. [The meteorology pipeline](#4-the-meteorology-pipeline)
5. [Making the GPU work: one step, one launch](#5-making-the-gpu-work-one-step-one-launch)
6. [Memory](#6-memory)
7. [Invariants](#7-invariants)
8. [Diagnostics and knobs](#8-diagnostics-and-knobs)

---

## A little vocabulary

A handful of computing terms recur below. None is complicated, but they are the
kind of thing that is assumed rather than explained, so they are collected here.

| Term | What it means |
| --- | --- |
| **host** and **device** | the CPU and its ordinary memory (host), versus the GPU and its own memory (device). They are separate machines that talk over a link, and most performance questions come down to who is waiting for whom. |
| **kernel** | one operation the GPU performs on a whole array at once — an addition, an exponential, an interpolation. A single line of PyTorch usually becomes one kernel. |
| **launch** | the CPU telling the GPU to run one kernel. Launching costs the CPU a few microseconds *whether or not the kernel does much work*, so a step made of a thousand tiny kernels can spend all its time launching. |
| **synchronisation** ("sync") | the CPU stopping to wait for a number back from the GPU — for instance to decide an `if`. Everything queued behind it stalls, so syncs inside a loop are expensive. |
| **eager** | running operations one at a time, exactly as written, with no ahead-of-time optimisation. The straightforward way, and GLIDE's reference behaviour — the code and tests use the word, so it is worth knowing. |
| **compiled** | letting PyTorch inspect a block of code ahead of time and emit a fused, optimised version. Faster, but only if it can see the whole block (see §5). |
| **buffer** | a block of memory holding an array. "Reusing a buffer" means writing new numbers into the same memory rather than allocating fresh memory. |
| **cache** | keeping a previously computed result so it need not be recomputed. An **LRU cache** ("least recently used") holds a fixed number of entries and, when full, discards whichever has gone longest untouched. |
| **wall-clock time** | elapsed real time, as opposed to how much work was done. The thing you actually wait for. |
| **thread** | an independent strand of execution within the same program, so two things can happen at once — used here to read the next hour of meteorology while the current one is being computed. |

---

## 1. The shape of the problem

A backward LPDM run contains three kinds of repetition, and they suit completely
different hardware:

| Repetition | Typical extent | Can the pieces run at the same time? |
| --- | --- | --- |
| **Time steps** | 7,200 per batch (a 5-day window at $\Delta t = 60$ s) | **No.** Step $n{+}1$ needs the particle positions from step $n$. This is irreducibly one-after-another. |
| **Particles** | ~5×10⁵ per batch | **Yes, perfectly.** Every particle does identical arithmetic on its own numbers, so all of them can be advanced simultaneously. This is what a GPU is built for; a CPU exploits it more modestly across its cores. |
| **Releases / batches** | hundreds to thousands | **Yes, completely independently.** Each release is its own backward integration and shares nothing with the others, so they can be spread across cores, GPUs or whole nodes. |

The awkward consequence is the middle row combined with the first. Each batch
gives the GPU a **moderate amount of work to do at once, repeated an enormous
number of times in strict order**. GPUs are happiest with the opposite: a huge
amount of work at once, repeated few times.

Half a million particles *is* enough work per step to keep a GPU busy. The
difficulty is the ordering. Because the steps cannot overlap, the CPU has to set
up each one individually — thousands of times per batch — and if the GPU sits
idle during that setup, it is the setup, not the physics, that determines how
long the run takes.

That single observation explains most of the design decisions below, and §5
entirely.

---

## 2. Module map

| Module | Responsibility |
| --- | --- |
| `main.py` | Overall control flow: config → schedule → batch loop → step loop → outputs. Also the command-line entry point. |
| `config.py` | The run-configuration schema and its validation (using Pydantic, a library that checks a YAML file against a declared structure and rejects it with a clear message if it does not fit), plus expansion of a schedule into batches. |
| `met_reader.py` | Reads ERA5 (or any conforming store) from Zarr: unit checks, ω→w conversion, geopotential→AGL, terrain-following resampling, per-hour caching. |
| `vertical_grid.py` | The terrain-following resampling itself: building the AGL ladder, computing per-column interpolation weights, terrain slope, the $w$ transform, and hydrostatic pressure on model levels. |
| `gpu_engine.py` | The elementary operations, written so they run on any device: coordinate normalisation, RK2 advection, the Langevin velocity update, turbulent displacement, surface reflection. |
| `turbulence/` | A common interface every turbulence scheme implements, plus a name-keyed lookup so one can be chosen from the config. `hanna.py` is the production scheme. |
| `convection/` | The same arrangement for convection; `emanuel.py` is the production scheme. |
| `release_generator.py` | Particle generation — point, volume, column, flight-track — and assembly of a batch, tagging each particle with which release it belongs to and when it is released. |
| `footprint_gridder.py` | Adds each particle's residence time into its grid cell, on the fly, directly on the GPU. |
| `output_writer.py` | Writes the footprint store (Zarr) and the particle/diagnostic tables (Parquet). |
| `comparison.py` | Conversion to STILT units and mass-conserving regridding, for comparison against other models. |
| `runtime.py` | Picks the device: NVIDIA GPU if present, else Apple GPU, else CPU. |

Everything is **plain PyTorch**, and everything runs on any of those devices.
There is no hand-written GPU code: the same source runs on a laptop CPU and on a
GH200 supercomputer node.

---

## 3. Anatomy of a run

```
RunConfig (YAML)
   │
   ├─ expand the schedule       releases → batches of ≤ max_releases_per_batch
   ├─ preflight checks          does the meteorology cover the whole schedule?
   │                            is every release inside met_domain?
   │
   └─ for each batch:
         generate particles     (n_releases × n_particles, in one array)
         allocate the footprint grid    (n_releases, T, Z, Y, X) on the GPU
         │
         └─ walk the clock backwards, from the latest release-window end
            to the earliest one minus the run length:
               fetch the meteorology hour   (from cache; next hour fetched in background)
               advection                 ─┐
               turbulence                 ├─ one combined operation on the GPU
               ground reflection         ─┘
               convection                   once per meteorology hour
               accumulate the footprint
               retire particles that have left the domain
         │
         write this batch's footprints to disk, then free the grid
```

**Particles are generated up front for the whole batch** and held in one array.
Each particle records which release it belongs to and its own release time (drawn
at random within that release's window), so a single backward sweep integrates
every release in the batch correctly. At each step a particle is active only if
it has been released yet, is still inside its own backward window, and has not
left the domain.

**Releases are the outermost dimension of the output, and the list is flat** —
one entry per site per time, rather than a site axis crossed with a time axis.
That is deliberate: it accommodates future release geometries (vertical column
releases, or satellite soundings each with its own averaging kernel) without
changing the output format. A conventional per-site cube is easy to recover
afterwards:

```python
fp.set_index(release=["site", "release_time"]).unstack("release")
```

**Sites released together share their meteorology.** With `multi_point_periodic`,
every site releases on the same schedule, so all their particles need the same
meteorology hours at the same time. Everything that costs the same regardless of
how many particles are involved — fetching and regridding the meteorology,
building the convection matrix, constructing the turbulence support fields, the
per-step bookkeeping — is then paid **once for the whole network** instead of
once per site. This is the intended way to scale a run across a station network,
and it is why the reference run in the README does 56 sites × 48 hours in a
single process.

Outputs land under `io.output_uri`:

| File | Contents |
| --- | --- |
| `footprints.zarr` | the 5-D footprint store, written one batch at a time as each finishes |
| `endpoint_particles.parquet` | final particle states |
| `trajectory_diagnostics.parquet` | per-step ensemble diagnostics |
| `run_metadata.json` | provenance, a copy of the config, timings, escape counts, any memory-guard report |

---

## 4. The meteorology pipeline

This is the half of the project that addresses the reading bottleneck. Rather
than opening whole NetCDF or GRIB files, GLIDE treats the archive as something to
sip from: it asks for **one hour, over the box currently containing the particle
cloud, up to its vertical ceiling**.

Zarr stores an array as many small compressed blocks ("chunks") rather than one
continuous file, and each can be fetched independently. So a request for a small
region only decompresses the handful of blocks that overlap it, instead of
reading through the whole archive. That is what makes a regional, time-bounded
run cheap against a global dataset.

Per hour, the reader:

1. selects the bounding box and the vertical levels that intersect the requested
   AGL range;
2. checks units on every variable (a missing or unrecognised `units` attribute is
   a hard error — GLIDE will not guess);
3. converts ω to a geometric vertical velocity where the source supplies a
   pressure tendency, $w = -(R_d T)/(g p)\ \omega$;
4. de-accumulates surface fluxes if they arrive as J m⁻², and flips the ECMWF
   heat-flux sign convention;
5. derives height above ground from geopotential, $(z - z_{sfc})/g$;
6. **resamples the whole hour onto the fixed terrain-following AGL ladder**
   (`vertical_grid`), excluding sub-surface levels, and corrects $w$ for terrain
   slope into that frame;
7. packs the result into a single array indexed by (variable, level, latitude,
   longitude).

On model (hybrid) levels the vertical coordinate is a level index rather than a
pressure, so pressure at each level is reconstructed hydrostatically from the
archive's own geopotential, surface pressure, temperature and humidity — no
hybrid coefficient table required, deliberately, since the ARCO archive does not
ship one. See
[dev/decisions/0010](../dev/decisions/0010-model-level-met-reader.md).

The full input contract is [met_schema.md](met_schema.md).

### Three layers of caching

Step 6 is the expensive one. Timing a single hour's fetch: the resampling takes
about 74% of it (and just working out the interpolation weights is about 50%),
while actually reading the data off disk is only about 17%. Three caches sit on
top of that:

| Layer | What it remembers | Why |
| --- | --- | --- |
| **Processed hours** (in the reader; 6 kept) | one fully processed meteorology hour | Consecutive steps interpolate between two bracketing hours, and neighbouring windows share one of them. Without this cache, every hour is read, converted and regridded **twice**. |
| **Meteorology windows** (in the run loop; `memory.met_cache_max_hours`) | a whole window, ready to use | Consecutive batches walk back over overlapping periods. Set this too small and the run repeatedly discards hours it is about to need again and re-fetches them; the run warns at startup if the setting looks too low. |
| **Derived turbulence fields** (in the scheme) | air density, free-troposphere σ and $T_L$, meander σ, on the meteorology grid | These are built **once at the middle of each meteorology hour** ($\alpha = 0.5$) and reused for every step within it. |

The third is a physics approximation, and a deliberate one: those fields change
by well under a percent across an hour, and FLEXPART-class models likewise
refresh their turbulence fields once per meteorology step rather than once per
particle step. It removed roughly 18% of an earlier CPU profile.

While the GPU works on the current hour, a **background thread fetches the next
one**, so the waiting for data happens underneath the computation instead of
between steps. This requires the meteorology to be held in ordinary CPU memory
(the background thread must not talk to the GPU), and is disabled with a warning
if `met_cache_on_host` is turned off.

The interpolation weights are also computed once per window and shared, rather
than recomputed for every field and every level: which source levels bracket a
given target height depends only on the level heights and the target grid, not on
*which* variable is being interpolated. That change alone cut the resampling cost
by about a factor of ten, which mattered because at full domain size the slower
version could no longer be hidden behind the GPU's work.

---

## 5. Making the GPU work: one step, one launch

Recall §1: each step gives the GPU a moderate amount of work, and there are
thousands of steps in strict order. The risk is that setting up each step costs
more than performing it. On the GH200 that risk was realised. Measuring one step:

- the CPU issued about **1,250 separate instructions to the GPU** (one per
  elementary operation);
- the GPU was **busy only ~17% of the time**;
- there was about **2.5 ms of actual GPU work inside a step that took ~30 ms**.

The GPU was spending most of the step waiting for the next instruction.

There is a neat way to confirm this rather than assume it. Cutting the maximum
sub-step count from 50 to 5 *halved* the run time, while GPU utilisation stayed
flat. Cutting the amount of arithmetic would have raised utilisation and saved
little; cutting the *number of instructions* saved a lot. That is the signature of
a run limited by instruction overhead rather than by computation.

Compiling alone does not fix it. PyTorch's compiler can fuse the operations
within one function call, but it cannot see across a Python loop, and the loop is
where the cost is. The fix is to **record the entire sequence of instructions for
one step, once, and then replay the whole recording with a single command** on
every subsequent step. NVIDIA calls such a recording a **CUDA graph**.

### Two ways of taking a step

Recording only works if every step issues *exactly* the same sequence of
instructions on *exactly* the same-shaped arrays. That conflicts with the obvious
way to save work, which is to select only the particles that are still active and
operate on just those — because the number selected changes from step to step.

GLIDE therefore carries two implementations, chosen automatically by device:

| | **Selective** (CPU default) | **Uniform** (GPU default) |
| --- | --- | --- |
| Which particles are processed | only the currently active ones, selected out of the array | all of them, every step |
| Inactive particles | skipped | processed, but with their timestep set to zero, so nothing about them changes |
| Sub-step loop | runs as many times as the busiest particle needs | always runs the maximum number of times |
| Waiting on the GPU for a number | a few times per step; cheap on a CPU | never, inside the recorded region |
| Role | **the reference behaviour** | the fast path |

Setting a particle's timestep to zero is exactly equivalent to leaving it alone,
which is what makes the uniform path give the same answer: the memory factor
becomes $r = e^0 = 1$ so the velocity is unchanged, the noise amplitude
$\sigma^2(1-r^2) = 0$ so no randomness is added, the displacement $w'\cdot 0 = 0$
so the position is unchanged, and reflecting a particle already above ground does
nothing. Every particle still takes exactly its own $k_p$ real sub-steps; the
extra iterations do nothing at all. Inactive particles are held still by
*multiplying* by zero rather than by being excluded from the array — that is what
keeps the array sizes fixed.

The trade-off is explicit: the uniform path performs *more* arithmetic (finished
and departed particles are still processed) in exchange for far fewer
instructions. It wins because instruction overhead was the problem. On a CPU,
where issuing work is free, the trade reverses — hence choosing by device.

### What gets recorded

On the GPU path with `GLIDE_COMPILE=1`, the whole body of a step is compiled as
one unit: advection, meteorology interpolation, the turbulence profile, the
drift, the sub-step loop, meander, and writing the results back. Fetching
meteorology, convection, and rebuilding the per-hour support fields all stay
*outside* it, which the existing once-per-hour / once-per-step separation makes
straightforward.

Measured on the GH200: **30 → 5 ms per step**, instructions issued **~1,250 →
~106**, GPU busy **17% → 37%**.

Four conditions have to hold for the recording to work, and each was learned by
violating it:

1. **The step must contain no decisions that depend on the data.** An `if` whose
   answer is a number computed on the GPU forces the recording to stop and
   restart there. One such test — comparing two level heights, reached five times
   per step — was enough that the first attempt at recording changed nothing at
   all.
2. **Nothing that varies from step to step may be passed in as a plain Python
   number.** The compiler treats a Python number as a fixed constant and rebuilds
   the whole recording when it changes. The time-interpolation weight $\alpha$
   (different every step) and the level heights (different every meteorology
   hour) both had to be passed as arrays instead. Get this wrong and it rebuilds
   continually, gives up, and falls back to running uninstrumented — four times
   *slower* than before, with no error message.
3. **The input arrays must stay at the same memory addresses.** A replay copies
   in any input that has moved since the recording was made. Loading each
   meteorology window into freshly allocated memory therefore caused the large,
   hour-constant arrays to be copied again on *every* step — about 49% of all GPU
   time. They now live in permanent, reused buffers that are refilled in place,
   outside the recorded region. (With one subtlety: PyTorch treats the device
   names `cuda` and `cuda:0` as different, so a naive comparison reallocated a
   buffer every step and forced the recording to be remade every step.)
4. **Results that outlive the step must be copied out.** The recording writes its
   outputs into fixed memory that the next replay overwrites, so anything kept
   must be duplicated first.

The first two are caught **on a CPU, in the automated test suite**, with no GPU
needed: `test_step_core_traces_as_one_graph_no_breaks` asks the compiler to
insist on a single unbroken unit and fails if anything splits it, and
`test_step_core_does_not_recompile_per_step` fails if anything triggers a rebuild
while $\alpha$, the meteorology values and the level heights all change. Both
faults are silent in production, which is what makes the tests worth having.

### Where the time goes now

With the per-step path handled, it is no longer the dominant cost. On a
representative multi-site GH200 run it is roughly 18–23% of the total. What
remains is reading meteorology, convection, and an untimed remainder —
diagnostics, bookkeeping about which particles are active, generating particles
and writing output at batch boundaries, and general Python overhead. Further work
on the step itself is low priority against those; see
[../STATUS.md](../STATUS.md).

---

## 6. Memory

A batch is the unit of both memory use and wasted work. It holds all its
releases' particles in one array, plus one footprint grid:

$$
\text{peak} \ \approx\  n_{\text{releases/batch}} \times \Big( n_{\text{particles}} \times 112\ \mathrm{B} \ +\  \text{footprint bytes per release} \Big)
$$

**Footprints are written out as the run proceeds.** The output file is created
once, sized for every release, and each batch writes its own slice and then frees
its grid — so only one batch's worth is ever in memory at a time. Putting every
release in a single batch defeats this entirely: 40,320 releases with one
vertical layer would need 51.6 GiB for footprints plus about 29 GiB for
particles.

**The most efficient batch size is the "active window"** — the number of releases
whose backward windows overlap at the busiest moment, which is
$\lceil \text{length}/\text{period} \rceil \times n_{\text{sites}}$. Larger
batches spend their time stepping particles that are not yet released or already
finished. Smaller batches use less GPU memory but need a larger meteorology
cache, because consecutive batches re-read overlapping periods.
`scripts/make_multisite_config.py` does this arithmetic for you, sizing the batch
to the active window (subject to a GPU memory budget) and the cache to the run
geometry.

**The meteorology cache lives in ordinary CPU memory by default**
(`met_cache_on_host: true`). Each window is a large array — hundreds of
megabytes — so a 192-hour cache is around 50 GiB. Keeping it in the node's main
memory rather than the GPU's own (smaller, faster) memory leaves the latter free
for computation. Note that the cache grows *in proportion to the number of AGL
levels*: on the EUROPE domain at 192 hours, 23 levels needs ≈52 GiB but 40 levels
needs ≈90 GiB.

**Memory limits stop the run rather than letting it die messily.**
`guard_max_rss_gib`, `guard_max_device_allocated_gib` and
`guard_max_device_reserved_gib` are optional ceilings, checked every
`guard_check_every_steps`. Exceeding one raises a clear error and writes
diagnostics into `run_metadata.json`, instead of failing deep inside a memory
allocation. Separately, the footprint grid refuses to allocate more than 32 GiB
(`LPDM_FOOTPRINT_MAX_GIB`), with a message saying which dimension to reduce.

**An alternative that was rejected: one continuous sweep.** Keeping a single
sweep over all releases and writing out each footprint as its window closed would
bound the *footprint* memory but not the *particle* memory, since all particles
would stay resident. Worse, on the GPU path it would step roughly six times as
many particles, because only about a sixth are active at any moment. Bounding
particle memory too would require an array that shrinks during the run, which is
precisely what the fixed-size recording of §5 forbids. Batching is the memory
lever available *because* the recording strategy fixes the array size.

---

## 7. Invariants

Performance work must not make the physics opaque. These are treated as hard
constraints, not preferences:

- **The physics stays in ordinary standalone functions.** $\sigma_w$, $T_L$, the
  drift, the Richardson closure, the meander stencil, the convection
  thermodynamics — each is a plain function that can be called and tested on its
  own. The compilation of §5 wraps the *assembled* step; it never rewrites or
  absorbs these.
- **The straightforward path is the reference.** Compiled and recorded paths are
  optional accelerations, validated against it. It must remain runnable, for
  debugging and on machines without a GPU.
- **Schemes stay interchangeable.** The turbulence and convection interfaces and
  their lookup tables stay intact, so a new scheme inherits all of the above
  machinery without any bespoke plumbing.
- **Adjustable quantities stay in the configuration file**, not in the code, so
  the physics can be tuned and compared without editing anything.
- **Memory limits stay strict**, and no cache in a long-running loop is allowed
  to grow without bound.

---

## 8. Diagnostics and knobs

All of these are environment variables, and all are optional.

| Variable | Effect |
| --- | --- |
| `GLIDE_COMPILE=0/1` | Turn compilation and step recording off or on. Off avoids the one-off compilation cost at startup — useful when debugging. |
| `GLIDE_STATIC_SUBSTEPS=0/1` | Force the uniform or selective step path (§5), overriding the automatic choice by device. |
| `GLIDE_PROFILE=1` | Record about 20 steps in detail, print how busy the GPU was, where it waited, and which operations ran most often, write a timeline file viewable in a browser, then exit. `GLIDE_PROFILE_STEPS` / `_WARMUP` / `_TRACE` / `_CONTINUE` adjust it. |
| `GLIDE_PHASE_TIMERS=1` | Break the whole run's elapsed time down by phase (meteorology fetch / advection / step / convection / gridding). `_EVERY` and `_SYNC` adjust it. |
| `GLIDE_NUM_THREADS=N` | How many CPU cores PyTorch uses within a single operation (CPU runs only). About 16 measured best on a 48-core node; using all 48 was the *slowest* option, ~25% worse, because the arrays are small enough that the cores spend more time coordinating than working. |
| `GLIDE_VALIDATE_ENGINE=1` | Re-enable per-call checking of input values in the innermost routines. Off by default, because each check makes the CPU wait for an answer from the GPU, and the sub-step loop calls them hundreds of times per step. |
| `GLIDE_MEM_SNAPSHOT=1` | Dump a detailed GPU memory snapshot (`_BATCH`, `_PATH` adjust it). |
| `LPDM_FOOTPRINT_MAX_GIB` | Raise the 32 GiB ceiling on the footprint grid. |

**Reading a GPU run.** The `*.gpu.log` sidecar reports GPU utilisation. Low and
bursty means the run is limited by instruction overhead rather than computation.
Confirm it the way §5 did: lower `max_substeps` and see whether the elapsed time
falls while utilisation does not. In a `GLIDE_PROFILE` timeline, many tiny GPU
operations point the same way; a lot of time in `cudaStreamSynchronize` means the
CPU is still waiting on the GPU somewhere; long stretches of CPU-only activity
mean the bottleneck is Python or meteorology reading. If the CPU time is
dominated by the compiler's own machinery (`dynamo_timed`,
`fx_codegen_and_compile`) while ordinary operations run thousands of times, the
recording is being rebuilt continually — some per-step Python number has reached
the compiled region, which is condition 2 of §5.

**Confirming compilation actually engaged.** There should be no `WON'T CONVERT`
warnings in the error log, and the first step should be noticeably slower than
the rest — that is the one-off compilation.

**Toolchain on Isambard AI.** PyTorch's compiler needs *two* things present, and
loading one without the other makes it silently fall back to running
uncompiled: `cudatoolkit/24.11_12.6` supplies the GPU compiler, and
`gcc-native/14.2` supplies a C++20-capable host compiler (a small amount of C++
is generated even for GPU work, and the system's default GCC 8.x is too old to
accept it). Set `CC=gcc` and `CXX=g++` so the newer one is picked up.
`scripts/run_periodic_cuda.slurm` arranges all of this.
