# GLIDE roadmap

_Written 2026-09-16. Owner: Matt Rigby. Revised as items are completed. The
"what is next" pointer in [STATUS.md](../STATUS.md) links here._

This page is the development plan. It covers:
- every proposed piece of work, each with a short ID such as "2b";
- what the first public release (v1) will contain, and the conditions it must
  meet before release;
- the order in which the work will be done.

The IDs are used in three other places:
- [dev/agent/QUEUE.md](agent/QUEUE.md), the detailed task specifications for
  the AI coding agents;
- [dev/decisions/](decisions/), short dated notes recording why each major
  choice was made;
- [docs/validation-plan.md](../docs/validation-plan.md), how the model will be
  validated.

**Contents**

1. [Where GLIDE is going](#1-where-glide-is-going)
2. [All proposed items](#2-all-proposed-items)
3. [v1: scope and release gate](#3-v1-scope-and-release-gate)
4. [Sequence](#4-sequence)
5. [After v1](#5-after-v1)
6. [Effort](#6-effort)

---

## 1. Where GLIDE is going

GLIDE is a backward-in-time Lagrangian particle dispersion model. Its aim is to
make footprint calculations fast and flexible enough for much larger problems
than current models can handle. Its two innovations so far:
- it reads meteorology from Zarr, a cloud-friendly format, either streamed or
  from a local copy;
- it runs on GPUs.

The long-term goal is greenhouse-gas inversion using very large satellite
datasets, with a fuller account of uncertainty than inversions include today.

Three lines of work ("tracks") lead there.

- **Track A: transport uncertainty.** The first step is to run many copies of
  a simulation with different parameter values cheaply, inside one GPU run.
  The next is to compute how footprints change with each parameter, by
  automatic differentiation through the model code. Those derivatives give an
  estimate of transport error that an inversion can use. Finally, the
  parameters are calibrated against observations.
- **Track B: how meteorology is stored and supplied.**
  - compressed copies of the meteorology, prepared by
    [met-bazarr](https://github.com/mrghg/met-bazarr), a separate repository
    that wraps each centre's download client and writes Zarr stores in
    GLIDE's format (GATES's meteorology processing may also move there, to be
    agreed with the team and the GATES maintainer)
    ([decision 0018](decisions/0018-met-bazarr.md));
  - a cache prepared specifically for GLIDE;
  - the ability to combine several meteorological sources: a high-resolution
    regional model inside a global one, ensemble members, or the output of
    machine-learning weather models;
  - later, machine-learned compression, trained so that footprints computed
    from the compressed fields stay accurate.
- **Track C: scale.**
  - releases that represent satellite soundings;
  - compact storage for footprints;
  - joint development with GATES, the group's graph-neural-network emulator of
    footprints. GLIDE produces the large sets of footprints that GATES is
    trained on. GATES then becomes a very fast stand-in for transport, from
    which an inversion can draw many samples.

None of this is credible until the physics has been validated, to a standard
the community will accept from a model written mostly by AI agents. That is
item 1, and it has its own plan:
[docs/validation-plan.md](../docs/validation-plan.md).

Related, in the GATES repository: `docs/code_review_2026-09-16.md` is the
equivalent task list for the emulator.

---

## 2. All proposed items

How to read the tables:
- **Size:** **S** is days to a week, **M** is several weeks, **L** is a research
  project of several months.
- **Needs** lists the items that must be finished first.
- **Changes output?** marks items that change the model's numerical results.
  These must be done before the physics is frozen (item 1a). After the freeze,
  any such change needs a decision record and a new physics version.
- **Release** is the version the item is planned for. v1.x means a later minor
  release, and "research" means an open-ended project with no release
  commitment.

Explanations of each item follow each table.

### Validation and calibration (item 1)

| ID | Item | Size | Needs | Changes output? | Release |
| --- | --- | --- | --- | --- | --- |
| 1a | Freeze the physics | S | 6d | — | v1 |
| 1b | Reciprocity test and convergence studies | S | none | sets defaults | v1 |
| 1c | Physics audit table, signed off externally | S | none | — | v1 |
| 1d | Comparison with FLEXPART on the same meteorology | M | FLEXPART runs (human) | — | v1 |
| 1e | Tracer experiments and radon | M | 1d | — | v1 |
| 1f | Parameter sensitivity and calibration | M | 1e, 2a | yes | v2 |
| 1g | UK CH₄ inversion compared with NAME-based results | M | 1a (v1) or 1f (v2) | — | v1 paper |
| 1h | Model paper, public benchmark, reproducibility package | M | 1b–1e | — | v1 |

- **1a.** Fix the model physics and give it a version label (`v0-physics`) in
  git, so that all validation refers to one version. This waits until ERA5
  model levels are validated and made the default (6d), because that changes
  footprint magnitudes everywhere.
- **1b.** Two checks on the numerics:
  - *Reciprocity:* a backward footprint should equal the result of a forward
    run with unit sources, so the backward formulation is tested directly.
  - *Convergence studies:* each numerical setting (time step, sub-steps,
    particle count, number of vertical levels) is refined until the answer
    stops changing. The results set the defaults.
- **1c.** A table listing every equation in the physics documentation, its
  literature source, where it is in the code, and which test checks it. An
  external scientist reviews and signs it.
- **1d.** GLIDE and FLEXPART v11 are both run on ERA5, and the particle clouds
  and footprints are compared (Tiers 1 and 2 of the validation plan).
- **1e.** GLIDE is compared with observations from controlled tracer releases
  (ETEX, CAPTEX, ANATEX) and with radon-222 (Tier 3). The metrics, tolerances
  and cases for Tiers 1–3 are fixed in advance and registered on OSF, which
  gives the registration an independent timestamp and a DOI.
- **1f.** Find which physical parameters matter most, then calibrate those
  against the tracer experiments with Bayesian methods. The result is
  published as a table of each parameter's prior and fitted values (a
  "parameter card").
- **1g.** Run the group's UK methane inversion with GLIDE footprints in place
  of NAME footprints, and compare the results. This is the paper's
  application.
- **1h.** The model description paper (for GMD), a public benchmark that
  others can run, and everything needed to reproduce the results.

### Transport uncertainty (Track A)

| ID | Item | Size | Needs | Changes output? | Release |
| --- | --- | --- | --- | --- | --- |
| 2a | Parameter ensembles in one run; particle-resampling noise estimates | S | none | no (infrastructure only) | v1 (not advertised) |
| 2b | Optional smoothed deposition of particles onto the grid | S | none | yes | v1 |
| 2c | Footprint derivatives with respect to parameters | M | 2b | yes | v2 |
| 2d | Smooth the switches between turbulence regimes | S | 2c | yes | v2 |
| 2e | Transport-error estimates for inversions, from the derivatives | M | 2c | — | v2 |
| 2f | Calibration using derivatives; learned turbulence corrections | L | 2c | yes | research |

- **2a.** Lets one GPU run carry many groups of particles, each with different
  parameter values. Parameter sensitivity studies then cost little more than a
  single run. Also lets particles be split into random subsets, so the
  sampling noise in any result can be estimated. Off by default; not
  advertised in v1.
- **2b.** Normally each particle's contribution goes entirely into the grid
  cell that contains it. This option spreads it over neighbouring cells with a
  small Gaussian, which reduces noise for the same number of particles. It is
  also needed for 2c, because the derivative of "which cell is the particle
  in" is zero almost everywhere. The width of the spreading is a numerical
  setting, chosen by a convergence study and never tuned. Validation always
  uses the unspread version
  ([decision 0012](decisions/0012-kernel-deposition.md)).
- **2c.** Uses PyTorch's automatic differentiation to compute, alongside each
  footprint, how it would change if each parameter changed (a "tangent"
  footprint). Particle trajectories are chaotic, so these derivatives may stop
  being meaningful over long runs. The item includes a test of that on a
  24-hour case.
- **2d.** The Hanna turbulence scheme switches abruptly between its stable,
  neutral and convective formulas. The derivatives in 2c are undefined at a
  switch, so the transitions are made smooth.
- **2e.** Combines the derivatives from 2c with the uncertainty in each
  parameter, to estimate how uncertain each footprint is and how those errors
  are correlated. This is in a form an inversion can use directly.
- **2f.** Uses the derivatives to calibrate parameters by gradient-based
  optimisation, and possibly to learn small data-driven corrections to the
  turbulence profiles.

### Meteorology (Track B)

| ID | Item | Size | Needs | Changes output? | Release |
| --- | --- | --- | --- | --- | --- |
| 3a | Compressed meteorology stores; a store for the ERA5 ensemble (in met-bazarr) | S | none | no | v1 |
| 3b | Combining several meteorological sources | L | none | yes | v1.x |
| 3c | Met Office UKV nested inside the global model (converter in met-bazarr) | M | 3b | yes | v1.x |
| 3d | Machine-learned compression of meteorology | L | 2c, 3b | — | research |
| 3e | Generated meteorological perturbations | L | 3d | — | research |
| 3f | Meteorology stored as a neural network | L | 3d | — | research |

- **3a.** Built in met-bazarr: compressed Zarr stores of the meteorology, in
  GLIDE's format, with provenance recorded in each store. Lossless
  compression by default; reducing precision to what is physically meaningful
  ("bit rounding") makes files much smaller, but it changes GLIDE's inputs, so
  it is enabled only after a comparison shows the effect is negligible
  ([decision 0018](decisions/0018-met-bazarr.md)). Also a store for the ERA5
  Ensemble of Data Assimilations (EDA), a 10-member ensemble that samples
  uncertainty in the meteorology. GLIDE adopts these stores in task B1.
- **3b.** Lets GLIDE read several meteorological sources at once, each with
  its own area, time resolution and vertical levels. For each particle, GLIDE
  uses the finest source that covers it. The same mechanism handles nested
  regional models, ensemble members and machine-learning weather models
  ([decision 0013](decisions/0013-composite-met-source.md)).
- **3c.** The first use of 3b: the Met Office's 1.5 km UK model (UKV) nested
  inside its global model (UMG). The converter for Met Office files belongs in
  met-bazarr; the nesting, and physics settings that can differ between the
  two domains, belong in GLIDE. It also
  allows a direct comparison with NAME on the same meteorology.
- **3d.** A neural network (for example a variational autoencoder) that
  compresses meteorology much further than 3a. It is trained so that
  footprints computed from the compressed fields match those from the
  originals, not just so that the fields look similar. The training needs
  GLIDE's footprint derivatives, so it lives with GLIDE; storing and serving
  the compressed form would be a met-bazarr output format.
- **3e.** A generative model, trained on the spread of the ERA5 ensemble, that
  produces realistic perturbations to the meteorology. This would sample
  meteorological uncertainty more cheaply than running a full ensemble.
- **3f.** Represent the meteorology as a neural network that returns the wind
  at any point and time, instead of as a grid.

### Releases and outputs

| ID | Item | Size | Needs | Changes output? | Release |
| --- | --- | --- | --- | --- | --- |
| 4  | Releases from vertical columns (multi-height towers, aircraft profiles) | M | none | no | v1 |
| 4a | Satellite releases | M | 4 | no | v1 |
| 4b | High-resolution footprint near the receptor | S | none | no | v1 |
| 4c | Ensemble members and derivatives in the footprint files | S | 2a, 2c | no | v2 |
| 4d | Compressed footprint storage | M | 4a | no | v2 |
| 4e | Particle count chosen per release | S | none | only if configured | v1 |

- **4.** Release particles through a vertical column, not from a single point,
  weighted by height. This covers tower inlets at several heights and aircraft
  profiles.
- **4a.** Releases that represent satellite column measurements. Each
  sounding gets its own vertical weighting from the retrieval's averaging
  kernel. The particle positions at the end of the run are saved, so that
  concentrations from a coarse global model can be added for the air
  arriving from outside the domain (the "boundary" or "endpoint"
  contribution).
- **4b.** A second, finer footprint grid around each receptor, where the
  sensitivity is concentrated and the coarse grid blurs it. Its cells line up
  exactly with the coarse grid, so the fine values sum to the coarse ones.
- **4c.** Store the outputs of 2a and 2c in the footprint files, labelled by
  ensemble member or parameter.
- **4d.** Store footprints as combinations of a smaller set of common spatial
  patterns, to cut storage for very large satellite datasets.
- **4e.** Let the number of particles differ between releases, for example
  more where footprints are noisy. The default is a fixed count, which gives
  the same output as now.

### Scale (Track C)

| ID | Item | Size | Needs | Changes output? | Release |
| --- | --- | --- | --- | --- | --- |
| 5a | Train GATES on large GLIDE footprint sets | M | 2a, 2b, 4a, 4b | — | v2 |
| 5b | Run GLIDE where GATES is least reliable | M | 5a | — | v2 |
| 5c | Inversions without storing footprints | L | 4a | — | research |
| 5d | Machine-learning weather models feeding GLIDE directly | L | 3b | yes | research |
| 5e | Simulation-based inference with GATES | L | 5a | — | research |

- **5a.** Produce GATES's training data with GLIDE. The training footprints
  are smoother and less noisy than single runs: averaged over particle subsets
  (2a) or spread onto the grid (2b). They include satellite releases (4a) and
  the fine near-receptor grid (4b).
- **5b.** Use disagreement between GATES models, or GATES's own uncertainty,
  to decide where new GLIDE runs would most improve GATES. This is "active
  learning".
- **5c.** An inversion multiplies footprints by emissions (and the reverse).
  At satellite scale the footprints are too large to store. This item would
  compute those products when needed, by re-running or replaying the particle
  transport, instead of keeping every footprint.
- **5d.** Run a machine-learning weather model (for example GraphCast or
  Aurora) on the same GPU as GLIDE, and pass its winds straight to the
  particles without writing files.
- **5e.** Inversion methods that learn the relationship between emissions and
  observations from many simulations, with GATES as the fast simulator. This
  is an alternative to MCMC, which does not scale to satellite datasets.

### Earlier roadmap items, kept

| ID | Item | Size | Needs | Changes output? | Release |
| --- | --- | --- | --- | --- | --- |
| 6a | Measure and reduce unexplained run time | S | none | no | v1 |
| 6b | Merge particles far from the receptor | M | 7a first | yes | v1.x |
| 6c | Improvements to the convection scheme | M | 1d | yes | v1.x |
| 6d | Validate ERA5 model levels and make them the default (task A01) | S | none | yes | v1, second |

- **6a.** GLIDE's timing breaks each time step into named phases (meteorology,
  turbulence, convection, and so on). A large share of the time is not yet
  assigned to any phase. This item measures where it goes, then speeds up the
  largest parts.
- **6b.** Far from the receptor, particles spread out and contribute little
  individually. Merging them into fewer particles, conserving mass and spread,
  saves computation. This is deferred until 7a, which gives similar savings
  more simply.
- **6c.** Fuller versions of parts of the Emanuel convection scheme, if
  validation shows that GLIDE under-represents convective transport.
- **6d.** ERA5 is available on 37 pressure levels or on its 137 native model
  levels. Model levels follow the terrain and are about four times denser in
  the boundary layer, and FLEXPART uses them too. GLIDE can already read them,
  but the reader has not been tested end to end, and downloads, example
  configurations and GLIDE's internal vertical grid all still default to
  pressure levels. This item validates the model-level path and makes it the
  default for all runs, including validation
  ([decision 0017](decisions/0017-model-levels-default.md)).

### Physics specification review (2026-08-22)

An independent review of the physics documentation against the code found
the following. None changes the results of the configurations in use today.
All are addressed in v1, and 8e before the physics audit (1c).

| ID | Item | Size | Needs | Changes output? | Release |
| --- | --- | --- | --- | --- | --- |
| 8a | Make the time bins of the footprint match their labels (task A00) | S | none | output labels | v1, first |
| 8b | Warn when particles reach the sub-step limit, on the GPU path too (in task A1) | S | none | no | v1 |
| 8c | Option to record particle residence at the middle of each step (in tasks A2, A3, A5) | S | none | yes | v1 |
| 8d | State that footprint cells are cell totals, not densities (in task B6) | S | none | no | v1 |
| 8e | Fix notation and overstated claims in physics.md (in task A8) | S | none | no | v1, before audit |

- **8a.** Footprints can be split by how long ago the air was at the surface
  ("time-ago" bins). The code always uses one-hour bins, but the file labels
  assume the bins divide the run length evenly. The two agree for the
  configurations in use. They would disagree, and data would be dropped, if
  coarser bins (for example daily) were used.
- **8b.** Particles in strong, rapidly varying turbulence take several
  sub-steps per time step, up to a fixed maximum. Hitting the maximum means
  the accuracy target is not met. The warning for this currently appears only
  on the slower CPU path.
- **8c.** Each particle's time near the surface is currently credited to the
  cell where it ends the time step. A particle that crosses cells during the
  step is credited only to the last one. Crediting it at the step's midpoint
  is more accurate and matches the time-stepping scheme.
- **8d.** Each footprint value is the sensitivity integrated over its grid
  cell. It is not a density per square metre. Regridding and GATES training
  both depend on this, so it should be stated in the documentation.
- **8e.** Inconsistent time indexing in some equations, and a few statements
  that are stronger than the method supports (for example "exact"). Also
  documents the limit on accuracy within about 3° of the poles.

### Compute and storage efficiency

| ID | Item | Size | Needs | Changes output? | Release |
| --- | --- | --- | --- | --- | --- |
| 7a | Longer time steps for particles in smooth flow | M | 6a | yes | v1 |
| 7b | Particle count and time step defaults from the convergence studies | S | 1b, 2b | yes | v1 |
| 7c | Shorter runs for satellite footprints | S | 4a | design choice | v1 |
| 7d | A meteorology cache prepared for GLIDE | M | none | no | v1 |
| 7e | Half-precision meteorology for interpolation | S | 7f | yes | v1 |
| 7f | Particle positions in metres, not degrees | S | none | only at rounding level | v1 |
| 7g | Smaller footprint files | S | none | no | v1 |
| 7h | Group releases by start time | S | none | no | v1 |

- **7a.** At present every particle uses the same 60-second time step. That is
  needed in the boundary layer but wasteful in the free troposphere, where most
  particle-hours of a multi-day backward run are spent. Each particle gets its
  own step (60 s, 120 s, 240 s and so on), chosen from local conditions. This
  could cut total computation by 3–5 times
  ([decision 0011](decisions/0011-multi-rate-time-stepping.md)).
- **7b.** Set the v1 defaults for the number of particles and the time steps
  from the convergence studies in 1b.
- **7c.** For satellite footprints, run particles backward for a shorter
  period and hand over to a global model's concentrations at the end (see
  4a), instead of running for many days.
- **7d.** Built in GLIDE from met-bazarr's stores: a local copy of the
  meteorology with only the variables and levels GLIDE uses. It is already interpolated onto GLIDE's terrain-following
  vertical levels and stored at reduced precision, so it is faster to read.
- **7e.** Interpolating the meteorology to each particle position is a large
  part of the run time. Holding the fields in 16-bit rather than 32-bit
  precision halves the data moved. Off by default unless tests show the
  effect is negligible.
- **7f.** Storing positions in metres from a local origin gives better
  precision in single-precision arithmetic than latitude and longitude. It
  also removes repeated unit conversions.
- **7g.** Footprint files with precision reduced to what is meaningful, and an
  option to store only the non-zero cells.
- **7h.** When releases start at different times, a batch of releases runs
  until the latest has finished, and particles from early releases sit idle.
  Grouping releases by start time reduces this idle time.

---

## 3. v1: scope and release gate

**What v1 will offer.**
- A validated backward Lagrangian particle dispersion model whose footprints
  are comparable with FLEXPART's and NAME's.
- At least ten times faster than the established CPU models, running on a
  GPU.
- Meteorology streamed from, or cached as, Zarr.
- Can be installed, and its results reproduced, by people outside the group.

v1 does not include uncertainty quantification.

| In v1 | Not in v1 |
| --- | --- |
| ERA5 model levels as the default meteorology (6d) | Footprint derivatives, transport-error estimates and derivative-based calibration (2c–2f) |
| Efficiency changes that alter numerical results (6a, 2b, 7a, 7e, 7f, 1b, 7b) | Calibrated parameters and the parameter card (1f) |
| Physics frozen with published default parameter values matching FLEXPART v11 (1a) | The UK inversion (1g), which is the paper's application, not a release condition |
| Validation Tiers 0–3 (1b–1e) | Combining meteorological sources, and nesting (3b, 3c) |
| Point, tower, column and satellite releases, with particle end positions saved (4, 4a) | Machine-learned compression and the other research items |
| High-resolution footprint near the receptor (4b) | Inversions without stored footprints, machine-learning weather input, simulation-based inference |
| Meteorology from met-bazarr stores (3a) and the GLIDE-ready cache (7d), smaller footprint files (7g), releases grouped by start time (7h), particle count per release (4e) | |
| Parameter ensembles, as infrastructure only (2a) | |
| Packaging: installable with pip, a container, a command-line interface, a tutorial, a stable output format and a DOI | |

**Release gate.** v1 is released only when every condition below is met. The
same checklist is the final task in [dev/agent/QUEUE.md](agent/QUEUE.md).

- The physics is frozen and tagged. Any later physics change increases the
  minor version number and has a decision record.
- All tests pass, on CPU in the automatic GitHub checks and on GPU on
  Isambard AI. This includes the reciprocity, well-mixed and convergence
  tests.
- The physics audit table is complete and signed by an external reviewer.
- The validation was pre-registered on OSF before any Tier 1–3 run, and the
  validation report cites the registration's DOI and any later updates to it.
  The run logs show every Tier 1–3 run, each one is in the report, and every
  repeated run has a logged reason.
- A validation report in `docs/` shows that the pre-registered criteria are
  met:
  - agreement with FLEXPART at the particle and footprint levels;
  - DATEM ranking scores for ETEX, CAPTEX and ANATEX, and the ETEX ATMES-II
    statistics, within the range published for NAME, FLEXPART, HYSPLIT and
    STILT.
- A performance table is published: footprints per second on a GH200 GPU and
  on CPU for a standard configuration, with the time spent in each phase.
- The results are reproducible: fixed software versions, a container, fixed
  random seeds, archived inputs for the validation cases, and a DOI. The
  met-bazarr version that prepared those inputs is pinned and archived too.
- The documentation covers:
  - installation and a quick start;
  - the required meteorology format and the output format;
  - the physics and the validation;
  - a worked example from footprint to inversion.
- The output format is declared stable, with a note on migration if it has
  changed.

---

## 4. Sequence

The principle: **efficiency changes that alter results are made before the
physics freeze. Changes that do not alter results can run in parallel, and
should start early, because every later run benefits.** Validation then
happens once, on the version of the model that will actually be used at
scale.

The work is done by AI coding agents in two parallel queues of tasks, defined
in [dev/agent/QUEUE.md](agent/QUEUE.md):
- **Queue A** covers anything that changes results, then the freeze, then
  validation. It runs one task at a time, in order, with the most capable
  model.
- **Queue B** is infrastructure that does not change results. It runs
  alongside Queue A, with a cheaper model.

A human reviews and merges every task.

| Phase | What | Who | Queue |
| --- | --- | --- | --- |
| **0. Prepare** | Background material, computing environment, list of available data (including a model-level ERA5 cube for A01), reference outputs for regression checks, access credentials (see [agent-operations.md](agent-operations.md) §2) | Human | — |
| **1. Model levels, then efficiency, before the freeze** | Fix time-bin labels (A00) → model levels as default (6d, A01) → 6a → 2b → 7a → 7f, 7e → 1b, 7b → 2a | Agent, most capable model, one task at a time | A |
| **1′. Infrastructure, in parallel** | 3a and 7d, 7g, 7h, 4e, 4, 4a, 4b, tools for validation figures, early packaging | Agent, cheaper model | B |
| **2. Freeze** | 1a; the physics audit table (1c) prepared for the external reviewer; tag `v0-physics` | Agent prepares, human signs off | A |
| **2′. Pre-register** | A8a computes the tolerances from reference data by the rules in the validation plan, without seeing any GLIDE result; A8b drafts the pre-registration record; the human checks both, merges, registers on OSF and records the DOI ([validation plan §3](../docs/validation-plan.md#3-pre-registration-protocol)) | Agent computes and drafts, human checks and registers | A |
| **3. Validate** | New Tier 0 tests (reciprocity, convergence figures) → Tier 1 component comparison → Tier 2 whole-model comparison → Tier 3 tracer experiments and radon | Agent does the analysis; human produces the FLEXPART reference runs | A |
| **4. Release v1** | Packaging, reproducibility bundle, documentation, release-gate checklist | Agent | B |
| **5. Paper** | 1g and 1h | Human, with agent support | — |

Two dependencies need a person and take time, so they should start early:
- the FLEXPART v11 reference runs on ERA5 (1d);
- the tracer-experiment data and the ERA5 data for those periods (1e). How to
  download them is described in
  [data/tracer-experiments.md](../data/tracer-experiments.md).

The FLEXPART runs can start now, with GLIDE's current physics. The final
comparison is repeated once the physics is frozen.

The GATES task list (`docs/code_review_2026-09-16.md` in the GATES repository)
is independent and can be worked on at any time, as a third queue.

met-bazarr has its own repository, instructions and task list
([`docs/roadmap.md`](https://github.com/mrghg/met-bazarr/blob/main/docs/roadmap.md)
there). GLIDE needs from it, in order:
- the `glide` format and ERA5 from the ARCO store (its tasks M1 and M2), for
  task B1;
- ERA5 ensemble columns for the standard cases (M3), for task A8a;
- ERA5 for the tracer experiments (M5) and a released, citable version (M6),
  for the pre-registration (A8b).

Until B1, GLIDE's existing download scripts stay in use.

---

## 5. After v1

- **v1.x:**
  - combining meteorological sources (3b);
  - UKV nesting (3c);
  - merging far-field particles (6b);
  - convection improvements, if validation shows they are needed (6c).
- **v2:**
  - Track A in full: footprint derivatives and transport-error estimates
    (2c–2e);
  - calibration and the parameter card (1f);
  - ensemble members and derivatives in the footprint files (4c);
  - the GATES training campaign and active learning (5a, 5b);
  - compressed footprint storage (4d).
- **Research projects,** each of which can be dropped without affecting the
  rest:
  - machine-learned meteorology compression (3d–3f);
  - inversions without stored footprints (5c);
  - machine-learning weather input (5d);
  - simulation-based inference (5e);
  - learned turbulence corrections (2f).

---

## 6. Effort

This assumes one person directing the agents, plus someone able to run
FLEXPART.

| Phase | Effort |
| --- | --- |
| Efficiency and convergence, before the freeze | 4–6 weeks |
| Freeze, audit, reciprocity, pre-registration | 2–3 weeks |
| FLEXPART comparison and tracer experiments | 8–12 weeks, mostly waiting for the reference runs |
| Releases, high-resolution output and packaging, in parallel | no extra time |
| **v1** | **roughly 4–5 months from 2026-09-16** |

The estimate is the least reliable thing on this page. Use it to notice when
work is slipping, not as a commitment.
