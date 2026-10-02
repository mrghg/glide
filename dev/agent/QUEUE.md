# Task queues

_Task specifications for coding-agent sessions. Read
[INSTRUCTIONS.md](INSTRUCTIONS.md) first. Roadmap IDs refer to
[dev/roadmap.md](../roadmap.md). Task status is derived from the repository (see INSTRUCTIONS.md §1);
per-task records live in [progress/](progress/). Written 2026-09-16; the human operator
reorders and edits this file weekly._

Two queues. **Queue A** is sequential: numerics-touching work before the
physics freeze, then the freeze, then validation. **Queue B** is infrastructure
and can run in parallel in a second allocation. Each task states its model
class, size, dependencies, scope, acceptance criteria and verification. Expected
runtimes are for a single GH200 unless stated.

Every spec inherits the verification standard in INSTRUCTIONS.md §3; the
"Verification" lines below are the task-specific additions.

Findings from the 2026-08-22 physics specification review are folded into the
tasks below and tagged **[PR-F1]** to **[PR-F8]**. The review itself is not in
the repository; each spec states the finding it addresses in full.

**Contents**

- [Queue A: numerics and validation](#queue-a-numerics-and-validation)
- [Queue B: infrastructure](#queue-b-infrastructure)
- [Final task: v1 release gate](#final-task-v1-release-gate)
- [Later queues (not yet specified)](#later-queues-not-yet-specified)

---

## Queue A: numerics and validation

Model: Fable, high effort. Run strictly in order unless a task is BLOCKED.

### A00 — Consistent time-ago bin width [PR-F1]

- **Roadmap:** prerequisite for A0. **Size:** S. **Depends on:** nothing.
  **Decision record:** required (0016).
- **Finding.** Per-particle time-ago bins are indexed hourly
  (`floor(elapsed_s / 3600)` in `src/lpdm/main.py`, both the eager and the
  static paths, and `_footprint_time_bin_index`), but the output metadata
  labels each bin as `simulation.length_seconds / 3600 / n_time_bins` hours
  wide (`time_ago_start_hours` / `time_ago_end_hours`). When
  `1 < n_time_bins < length_hours` the labels are wrong and residence older
  than `n_time_bins` hours is dropped by the gridder. Single-bin runs are
  special-cased (all ages clamped into bin 0) and are correct; the shipped
  multi-bin configs use one bin per hour and are unaffected. The defect is
  latent but will be hit as soon as coarser bins are used (e.g. daily bins on a
  30-day run), which the storage items invite.
- **Scope.** Add `output_grid.time_bin_seconds` (default 3600). Index bins as
  `floor(elapsed_s / time_bin_seconds)` on both runtime paths and in
  `_footprint_time_bin_index`, and derive the metadata from the same value.
  Keep the single-bin special case (time-integrated). Validate at config load:
  if `n_time_bins > 1` and `n_time_bins * time_bin_seconds < length_seconds`,
  raise unless a new `output_grid.drop_older_ages: true` is set, so dropping
  old residence is always an explicit choice. The value must reach the
  compiled step as a tensor or a construction-time constant, never a
  per-step Python scalar (decision 0004).
- **Output contract.** The change to `time_ago_*_hours` values and the new
  config fields are authorised by this spec; document them in README "Outputs"
  and in `docs/physics.md` footprint section.
- **Acceptance.** Tests: labels match indexing for (bins, width, length)
  combinations including 1 bin, bins = hours, and daily bins on a multi-day
  run; residence conservation when bins cover the run; the validation error
  when they do not. Existing `test_footprint_time_bin_index_advances_each_hour`
  still passes unchanged. Decision record 0016 states the choice.
- **Verification.** All shipped configs produce bit-identical footprints on CPU
  (they use 1 bin, or bins = hours with the 1 h default); graph capture intact.

### A0 — Golden fixtures, comparison and benchmark harness

- **Roadmap:** enabler for everything. **Size:** S. **Depends on:** A00.
- **Scope.** Create `scripts/agent/make_golden.py`, `scripts/agent/compare_golden.py`
  and `scripts/agent/benchmark.py`.
  - `make_golden.py` runs a named config with a fixed seed and stores the
    footprint tensor, endpoint particles and run metadata under
    `tests/golden/<config-name>/` (small configs) or under the outputs
    directory (large), with a JSON manifest (git rev, config hash, device,
    seed, torch version).
  - `compare_golden.py` compares a fresh run to a stored fixture and prints
    max absolute, max relative and integrated-sensitivity differences per
    release, plus pass/fail at a tolerance given on the command line.
  - `benchmark.py` runs a config with `GLIDE_PHASE_TIMERS=1`, parses the phase
    breakdown, and prints footprints per second and the phase table in
    Markdown.
- **Fixtures to generate:** smoke config on CPU eager (committed, small);
  smoke config on CUDA static path; reference multi-site config on CUDA (kept
  in outputs, manifest committed).
- **Out of scope.** Any change to `src/lpdm/`.
- **Acceptance.** Fixtures exist and are documented in `docs/VALIDATION.md`
  (new subsection "Golden fixtures"); `compare_golden.py` on the same rev passes
  bit-identically on CPU and within the existing parity tolerance on CUDA;
  `benchmark.py` output for the reference config is recorded as the baseline
  in the new `docs/VALIDATION.md` subsection and in the task's progress file.
- **Runtime.** Reference config on GH200: ~25 min.

### A1 — Sub-time the residual phase (6a)

- **Size:** S. **Depends on:** A0.
- **Scope.** Add named timers inside the per-step residual (wind-mean
  diagnostic, mask and liveness bookkeeping, particle generation, output
  writes, Python loop overhead, convection) so the phase table attributes all
  wall time. Re-profile convection after the vectorised parcel lift. Do not
  optimise anything in this task; measure.
- **Also [PR-F2]: sub-step saturation on the graph path.** The once-per-run
  warning when particles hit `max_substeps` fires only on the dynamic path
  (`HannaScheme`, around the `hit max_substeps` log); the static/graph path
  skips it, yet saturation is exactly where the drift and reflection accuracy
  target is violated. Add a device-side counter of saturated particle-steps,
  accumulated inside the captured step with no host sync, read once per met
  window, logged with the same wording as the dynamic path, and recorded in
  `run_metadata.json` (additive). Update the `docs/physics.md` sub-stepping
  section to say the warning now covers both paths.
- **Acceptance.** Saturation count reported on both paths for a config
  forced to saturate (small `max_substeps`), and zero on the smoke config; phase
  table for the reference config with no "residual"
  bucket above 5% of wall; convection re-profiled; the two largest consumers
  named in STATUS.md "Performance" with numbers. List follow-up optimisation tasks under `proposed follow-ups` in the
  task's progress file.
- **Verification.** Golden bit-identical on CPU (timers must not change
  numerics); graph capture intact.

### A2 — Kernel deposition in the gridder (2b)

- **Size:** S–M. **Depends on:** A0. **Decision record:** 0012.
- **Scope.** Add `output_grid.deposition` with `kind: delta | gaussian`
  (default `delta`, so behaviour is unchanged by default), `bandwidth_mode:
  fixed | time_since_release` and parameters. Gaussian deposition spreads each
  particle's residence weight over a compact stencil (3×3 or 5×5 cells)
  computed from the particle's sub-cell position and bandwidth; the stencil is
  a fixed size so shapes stay static. Mass is conserved exactly (weights
  normalised over the in-grid stencil). Works with the release axis and the
  nested grid (B6) later.
- **Also [PR-F4]: deposition position.** Residence is currently deposited at
  the particle's position after advection, turbulence and convection, i.e.
  endpoint quadrature: a particle crossing cell or vertical-bin boundaries
  during a step contributes nothing to the cells it traversed. Add
  `output_grid.deposition.position: endpoint | midpoint` (default `endpoint`,
  bit-identical). `midpoint` deposits at the RK2 midpoint position already
  computed for advection, which is consistent with the second-order scheme.
  List endpoint quadrature under known approximations in `docs/physics.md`
  regardless of the default.
- **Acceptance.** New tests: conservation with the kernel; convergence to the
  delta result as bandwidth → 0; the analytic plume test passes with the
  kernel at a stated bandwidth with equal or lower error; a noise test showing
  variance reduction at fixed particle count; the analytic plume error with
  `midpoint` no worse than `endpoint` at the reference Δt and better at 4Δt.
  Figure: plume footprint delta vs kernel vs analytic (Tier 0 set). `docs/physics.md` footprint section updated.
- **Verification.** With `kind: delta` golden bit-identical on CPU. Graph
  capture intact with `kind: gaussian` on CUDA.
- **Runtime.** Tests only; GPU smoke run.

### A3 — Multi-rate time stepping (7a)

- **Size:** M. **Depends on:** A1, A2. **Decision record:** 0011.
- **Scope.** Give each particle its own outer step Δt_i ∈ {Δt, 2Δt, 4Δt, …, Δt_max}
  chosen from local criteria (height above the boundary layer, |w|, σ_w, T_L,
  and a CFL-type bound on the met grid), re-evaluated once per met window. The
  loop keeps a fixed trip count: a particle advances only on the iterations
  that are multiples of its rate, gated by multiply, never by index. Residence
  accumulation uses the particle's own Δt_i and the `midpoint` deposition
  position from A2 [PR-F4], because a long far-field step makes endpoint
  quadrature coarser in proportion to the step. Reflection, the OU sub-stepping and
  the drift are integrated over Δt_i. Time interpolation of the met remains
  correct across the window.
- **Out of scope.** Particle aggregation; changes to the turbulence
  parameterisation.
- **Acceptance.** Well-mixed gates pass; analytic plume and Taylor tests pass
  (they run in the boundary layer, so Δt_i = Δt there); a new test on a
  two-layer synthetic met shows far-field particles taking the long step and
  the footprint agreeing with the single-rate run within a declared tolerance;
  reference-config golden difference within the tolerance declared in the PR
  (numerics task); benchmark before/after showing the step count and wall-time
  reduction. Default remains single-rate until A5 sets defaults.
- **Verification.** Parity static/dynamic; capture intact; benchmark.
- **Runtime.** Reference config twice: ~50 min.

### A4 — Local metric coordinates and half-precision met (7f, 7e)

- **Size:** S. **Depends on:** A3.
- **Scope.** (i) Store particle positions in metres relative to the met-domain
  origin (local equal-area or equirectangular with cos(lat) scaling), converting
  to degrees only at the reader and gridder boundaries. Exact to fp32 rounding.
  (ii) Behind a flag, hold the per-window met channel tensor in bf16 for the
  trilinear gather, with fp32 accumulation. Off by default.
- **Acceptance.** (i) Golden differences at fp32 rounding level on CPU and
  documented; all tests pass. (ii) A/B on the reference config: parity
  difference reported; benchmark before/after; recommendation recorded in the task's progress file whether to enable by default (decision record required if yes).
- **Verification.** Capture intact in both modes.
- **Runtime.** ~1 h.

### A5 — Convergence studies and defaults (1b, 7b)

- **Size:** M. **Depends on:** A2, A3, A4.
- **Scope.** Sweeps on the analytic plume and on one real-met case from the
  case library: Δt and Δt_max (multi-rate), sub-step accuracy target, particle
  count, vertical ladder, kernel bandwidth, and deposition position (endpoint
  vs midpoint) [PR-F4]. Report sub-step saturation counts from A1 [PR-F2] for
  every sweep point. Produce error-against-parameter
  figures with expected slopes (Tier 0 figure set) and a table. Propose the v1
  defaults with the converged values.
- **Acceptance.** Figures and table in `docs/validation/`; a decision record
  proposing the new defaults (the human accepts by merging it); no default is
  changed in this task.
- **Runtime.** Sweeps: budget 6 GPU-hours; stop and report if exceeded.

### A6 — Forward–backward reciprocity test (1b)

- **Size:** S–M. **Depends on:** A5.
- **Scope.** A forward integration mode usable only from tests (release a unit
  source, accumulate concentration at the receptor cell), and a test that the
  backward footprint from the receptor equals the forward concentration field
  at every cell, on synthetic met with turbulence and reflection on. Figure:
  forward field beside backward footprint, 1:1 scatter.
- **Acceptance.** Test passes at a stated tolerance with a stated particle
  count; a "teeth" companion shows the test fails when the backward sign flip
  of the drift is removed. `docs/physics.md` backward-construction section
  cites the test.

### A7 — In-batch parameter ensembles (2a, plumbing)

- **Size:** S. **Depends on:** A5.
- **Scope.** Allow the turbulence and convection schemes to take per-particle
  parameter tensors (broadcast from a per-group table) instead of scalars, and
  the release generator to assign particles to groups with drawn parameters.
  Also the particle-subset bootstrap: tag particles with a subset index and
  write per-subset footprints when requested. All off by default.
- **Acceptance.** With one group at the default parameters the golden
  comparison is bit-identical on CPU; capture intact with groups on CUDA; a
  test that two groups with different σ floors produce different footprints;
  output contract extended additively (documented).

### A8 — Physics freeze preparation and audit table (1a, 1c)

- **Size:** S. **Depends on:** A0–A7 merged; model-level met path validated
  (existing item 1).
- **Scope.** Write `docs/physics-audit.md`: a table with one row per equation
  in `docs/physics.md`, `turbulence.md`, `convection.md`, giving the literature
  source, the code location and the test that exercises it, with gaps marked.
  Prepare the tag `v0-physics` (the human creates it). Update STATUS.md.
- **Also, specification precision [PR-F5 to PR-F8]**, to be fixed in
  `docs/physics.md` before the external reviewer reads it:
  - **F5.** The OU step is written with forward-time indexing ($u'_{n+1}$, "one
    step later") while backward advection uses $x_n \to x_{n-1}$ and the
    displacement uses $z_{n+1} = z_n - w'\Delta t$. Use neutral old/new
    notation, or define a positive simulation-age coordinate $\tau = t_r - t$
    and use it throughout.
  - **F6.** The Taylor-dispersion curve is given for $\sigma_z^2$ but its
    asymptotes are stated as $\sigma_w t$ and $\sqrt{2Kt}$, which are
    statements about $\sigma_z$. Write $\sigma_z^2 \sim \sigma_w^2 t^2$ and
    $\sigma_z^2 \sim 2Kt$, or switch to standard deviation explicitly.
  - **F7.** Flooring $\lvert\cos\phi\rvert$ at 0.05 (about 87.1°) caps zonal
    angular displacement poleward of that latitude; label it a numerical
    approximation and state the supported latitude range.
  - **F8.** Qualify claims stronger than the scheme supports: "GLIDE
    accumulates exactly that", "conversion exact" (true only for the
    arithmetic conversion when bins align), and "the state … is the position at
    step $n-1$". The accumulator uses endpoint (or midpoint) quadrature, split
    operators, finite cells, interpolated meteorology and coefficients frozen
    over sub-steps.
  The review confirmed these elements of the formulation against code, which
  the audit table should record as evidence: the exact homogeneous OU update
  with Euler inhomogeneous drift (`gpu_engine.py`); the forward well-mixed drift
  and backward sign reversal (`turbulence/hanna.py`); joint reflection of
  position and $w'$ (`gpu_engine.py`); per-particle sub-stepping with fixed
  outer-step coefficients (`turbulence/hanna.py`); spherical displacement
  factors (`gpu_engine.py`); particle weights of $1/N$ (`release_generator.py`).
- **Acceptance.** Every equation has a row; every gap has a proposed test under `proposed follow-ups` in the task's
  progress file. The human and the external reviewer sign the
  table by merging.

### A9 — Tier 1 component parity (validation)

- **Size:** M. **Depends on:** A8; FLEXPART v11 source available.
- **Scope.** Scripts under `scripts/validation/tier1/` that call the FLEXPART
  Hanna and convection routines (via f2py or a transcribed reference marked as
  such) and GLIDE's on identical columns; figures per the plan; tolerances
  from the pre-registration record.
- **Stop condition specific to this task:** if `$GLIDE_FLEXPART_SRC` is unset
  or does not contain the FLEXPART v11 source, stop.

### A10 — Tier 2 twin analysis tooling (validation)

- **Size:** M. **Depends on:** A8; FLEXPART reference outputs available.
- **Scope.** Readers for FLEXPART particle and gridded output; particle
  statistics against time; footprint metrics; systematic-difference maps;
  mole-fraction convolution; the Tier 2 figure set including the cloud
  animation; a report page in `docs/validation/`.
- **Acceptance.** All Tier 2 figures for the case library, generated by
  scripts, with pre-registered metrics reported pass or fail.

### A11 — Tier 3 tracer-experiment runs (validation)

- **Size:** M. **Depends on:** A10, B10; the ERA5 cubes for ETEX, CAPTEX and
  ANATEX listed in the data inventory.
- **Scope.** Receptor-oriented backward runs, one per measured sample, as set out
  in `data/tracer-experiments.md` §5: ETEX-1, ETEX-2, CAPTEX, and ANATEX tracers
  1 and 2 (one footprint per sample serves both ANATEX tracers). Predicted
  concentrations written in DATEM format; scored with the B10 code using the
  pre-registered options; the Tier 3 figure set.
- **Acceptance.** DATEM ranking scores and their four terms for every
  experiment, and ATMES-II statistics for ETEX, reported against the published
  values for NAME, FLEXPART, HYSPLIT and STILT; pass or fail against the
  pre-registered thresholds; figures in the gallery; `data/tracer-experiments.md`
  §5 updated with what was run.
- **Runtime.** About 16,000 backward releases; estimate from the A0 benchmark
  before launching and stop if a batch exceeds twice the estimate.

---

## Queue B: infrastructure

Model: Sonnet unless a task says otherwise. Tasks are independent unless a
dependency is listed; run in the listed order by default.

### B1 — Met inventory and GLIDE-ready met cache (7d, 3a)

- **Size:** M. **Depends on:** nothing.
- **Scope.** `scripts/build_met_cache.py`: from local cubes, subset to the
  required variables and the levels below `alt_max_m`, optionally pre-resample
  onto the AGL ladder, store as float16 or bit-rounded float32 with zstd, in a
  Zarr layout the reader can open through the existing met contract (add a
  `glide_cache` attribute set). A reader path that uses the cache when present.
  Report storage before/after and met-fetch time before/after on the reference
  config.
- **Acceptance.** Golden comparison on the reference config within a declared
  tolerance for the float16 variant and bit-identical for the lossless
  variant; `docs/met_schema.md` documents the cache; benchmark shows the
  met-fetch phase reduced.

### B2 — Sparse and bit-rounded footprint output (7g)

- **Size:** S. **Depends on:** nothing.
- **Scope.** Zarr codec configuration for the footprint store (bit-rounding
  keeping a configurable number of mantissa bits, zstd), and an optional sparse
  (COO per release) writer with a loader helper in `comparison.py`. Additive
  to the output contract, documented.
- **Acceptance.** Storage numbers on the reference output; round-trip test;
  loader test.

### B3 — Release-time batching (7h)

- **Size:** S. **Depends on:** nothing.
- **Scope.** Batch expansion groups releases so that the sweep length per
  batch is minimised (cluster by release time); report the live-particle
  fraction per batch in run metadata.
- **Acceptance.** Golden bit-identical per release on CPU; live fraction
  reported; benchmark on a staggered-release config.

### B4 — Adaptive particle count per release (4e)

- **Size:** S. **Depends on:** A7 (bootstrap subsets) for the convergence
  criterion; can be built first with a fixed rule.
- **Scope.** Config option to set particles per release from a rule (fixed,
  by region class, or by a target relative noise estimated from bootstrap
  subsets in a pilot batch).
- **Acceptance.** Fixed rule bit-identical to today; documented.

### B5 — Column and satellite releases (4, 4a, 7c) — model: Opus or Fable

- **Size:** M. **Depends on:** nothing for column; A2 recommended for satellite.
- **Scope.** Column releases by importance sampling over a pressure-weighted
  vertical PDF; satellite releases as many irregular soundings per overpass,
  each with its own averaging kernel and pressure weighting, with endpoint
  particles written per sounding; a short-integration-plus-endpoints mode
  documented as the satellite recommendation (7c). Flat release axis unchanged.
- **Acceptance.** Tests for weighting and kernel application; a GOSAT-style
  example config; docs in README "Release geometries"; output contract
  extended additively.

### B6 — Nested footprint output (4b)

- **Size:** S. **Depends on:** A2 recommended.
- **Scope.** `output_grid.nested` with half-width, resolution, and alignment to
  coarse cell edges enforced by validation; per-release inner grid with origin
  tensors; second store `footprints_nested.zarr` with per-release coordinate
  arrays; works with delta and kernel deposition.
- **Also [PR-F3]: spatial meaning of footprint cells.** State in
  `docs/physics.md`, the README "Outputs" section and the `comparison.py`
  STILT-conversion docstring that each cell value is a **cell-integrated**
  horizontal sensitivity: multiplying it by a flux that is uniform over the
  cell gives the concentration enhancement, it is not a density per square
  metre, and conservative regridding must preserve the cell integral. The
  fine-sums-to-coarse property below depends on exactly this, and so does the
  GATES training target.
- **Acceptance.** Test that the coarse value over the covered area equals the
  sum of the fine cells; documented in physics and README.

### B7 — Validation figure infrastructure (Tier 0 set)

- **Size:** M. **Depends on:** nothing.
- **Scope.** `scripts/validation/` with a small figure library implementing the
  design rules in `docs/validation-plan.md` §6, the Tier 0 figures generated
  from the existing test quantities, a gallery page generator writing
  `docs/validation/gallery.md` keyed by physics tag, and an exportable static
  version of the footprint explorer. Follow the project's visualisation
  guidance.
- **Acceptance.** Gallery page renders on GitHub with all Tier 0 figures;
  regeneration is one command.

### B8 — Packaging and reproducibility

- **Size:** M. **Depends on:** A8 (freeze) for the final version, but start
  early.
- **Scope.** CLI polish; `pip install glide-lpdm` (name to be confirmed by the
  human); an Apptainer definition for Isambard and a Dockerfile; a pinned lock
  file; a nightly GPU test sbatch that runs the parity tests and posts the result as a comment on a pinned
  GitHub issue; documentation site configuration; DOI metadata
  (`CITATION.cff`, Zenodo).
- **Acceptance.** A clean machine can install and run the smoke test from the
  published instructions; the container runs the smoke test on Isambard.

### B10 — Tracer-experiment readers and scoring (1e)

- **Size:** S–M. **Depends on:** nothing in the queue; `$GLIDE_ETEX` and
  `$GLIDE_DATEM` populated per `data/tracer-experiments.md`. No model runs and no
  physics, so it can run before the freeze. Model: Sonnet, or Opus.
- **Scope.** Under `scripts/validation/tracers/`: readers for the ETEX files and
  the DATEM `emit`/`meas` files into common release and sample tables with SI
  units; a writer for DATEM-format predictions; the DATEM statistics and ranking
  score; the ATMES-II statistics for ETEX. All options of DATEM's `statmain`
  that the pre-registration must fix (zero threshold, zero–zero pairs,
  averaging) exposed as arguments.
- **Acceptance.** Running the scoring on DATEM's HYSPLIT predictions
  (`mdl_data/<exp>/ct?_001.txt`) reproduces DATEM's `mdl_stat` files for CAPTEX
  and both ANATEX tracers to their printed precision, as a test that skips
  cleanly when `$GLIDE_DATEM` is unset. Reader tests on small synthetic files in
  the repository. `data/tracer-experiments.md` §5 updated. No data files
  committed.

### B9 — Documentation upkeep (continuous)

- **Size:** S per pass. Run whenever the human asks, or after every four
  merged tasks.
- **Scope.** Reconcile README, STATUS, docs pages and decision records with
  what is merged; fix drift; keep `docs/README.md` index current.

---

## Final task: v1 release gate

Model: Fable. Runs when every v1 item is MERGED. Checks each line of the gate
in `dev/roadmap.md` §3 against the repository, writes `docs/validation/v1-report.md`
summarising the evidence for each line with links, and lists anything unmet.
The human decides the release.

---

## Later queues (not yet specified)

To be written when v1 is tagged. Roadmap IDs: v1.x — 3b, 3c, 6b, 6c; v2 — 2c,
2d, 2e, 1f, 4c, 4d, 5a, 5b; research — 2f, 3d, 3e, 3f, 5c, 5d, 5e.
