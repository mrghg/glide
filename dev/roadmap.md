# GLIDE roadmap

_Written 2026-09-16. Owner: Matt Rigby. Revised as items land; the authoritative
"what is next" pointer in [STATUS.md](../STATUS.md) links here._

This page holds the development plan: every proposed item with an ID, the scope
of the v1 release and its gate, and the sequence in which work happens. The IDs
are used throughout [dev/agent/QUEUE.md](agent/QUEUE.md) (task specifications
for coding agents), [dev/decisions/](decisions/) (why the major choices were
made) and [docs/validation-plan.md](../docs/validation-plan.md) (how the model
will be validated).

**Contents**

1. [Where GLIDE is going](#1-where-glide-is-going)
2. [All proposed items](#2-all-proposed-items)
3. [v1: scope and release gate](#3-v1-scope-and-release-gate)
4. [Sequence](#4-sequence)
5. [After v1](#5-after-v1)
6. [Effort](#6-effort)

---

## 1. Where GLIDE is going

GLIDE exists to make backward Lagrangian transport scalable and flexible. Its
first two innovations are streaming Zarr meteorology and GPU execution. The
long-term application is greenhouse-gas inversion from very large satellite
datasets with a fuller treatment of uncertainty than inversions carry today.

Three tracks lead there:

- **Track A, transport uncertainty.** Cheap parameter ensembles inside one
  batch, then differentiable transport (tangent footprints) giving a linearised
  transport-error covariance that an inversion can consume, then calibration.
- **Track B, meteorology representation.** Compressed and cached met stores, a
  composite met source (nesting, ensembles, ML weather models), and later
  learned compression trained with a footprint-aware loss.
- **Track C, scale.** Satellite releases, compressed footprint storage, and
  co-development with GATES, the group's graph-neural-network footprint emulator
  (GLIDE generates GATES's training ensembles; GATES becomes the fast,
  sampleable representation of transport the inversion uses).

Before any of that is credible, the physics has to be validated to a standard
the community will accept from a largely agent-written model. That is item one,
and it has its own plan in [docs/validation-plan.md](../docs/validation-plan.md).

Related, in the GATES repository: `docs/code_review_2026-09-16.md` is the
equivalent task list for the emulator.

---

## 2. All proposed items

Sizes: **S** days to a week, **M** weeks, **L** a multi-month research line.
"Numerics" marks items that change model output and therefore must land before
the physics freeze or after it with a decision record.

### Validation and calibration (item 1)

| ID | Item | Size | Depends on | Numerics | Version |
| --- | --- | --- | --- | --- | --- |
| 1a | Freeze physics v0 after the model-level met path lands | S | existing item | — | v1 |
| 1b | Forward–backward reciprocity test; convergence studies in dt, sub-steps, particles, ladder | S | none | sets defaults | v1 |
| 1c | Physics audit table (equation → source → code → test) with external human sign-off | S | none | — | v1 |
| 1d | FLEXPART v11 twins on ERA5, particle level then footprint level | M | FLEXPART runs (human) | — | v1 |
| 1e | Tracer experiments (ETEX, CAPTEX, ANATEX) and radon, pre-registered protocol | M | 1d | — | v1 |
| 1f | Parameter sensitivity screen and Bayesian calibration; parameter card | M | 1e, 2a | yes | v2 |
| 1g | UK CH₄ inversion against NAME-based results | M | 1a (v1) or 1f (v2) | — | v1 paper |
| 1h | Model paper, public benchmark, reproducibility package | M | 1b–1e | — | v1 |

### Transport uncertainty (Track A)

| ID | Item | Size | Depends on | Numerics | Version |
| --- | --- | --- | --- | --- | --- |
| 2a | In-batch parameter ensembles and particle bootstrap | S | none | plumbing only | v1 (unadvertised) |
| 2b | Soft kernel deposition in the gridder | S | none | yes | v1 |
| 2c | Forward-mode tangent footprints; chaos test on a 24 h case | M | 2b | yes | v2 |
| 2d | Smooth the Hanna regime switches | S | 2c | yes | v2 |
| 2e | Tangent-based transport-error covariance as an inversion input | M | 2c | — | v2 |
| 2f | Gradient calibration and learned Hanna corrections | L | 2c | yes | research |

### Meteorology (Track B)

| ID | Item | Size | Depends on | Numerics | Version |
| --- | --- | --- | --- | --- | --- |
| 3a | Bit-rounded, zstd-compressed met stores; an EDA ensemble store | S | none | no | v1 |
| 3b | Composite met source: multiple readers, priorities, cadences | L | none | yes | v1.x |
| 3c | UKV-in-UMG nesting, UM converter, per-domain physics config | M | 3b | yes | v1.x |
| 3d | Learned compressor with footprint-aware loss | L | 2c, 3b | — | research |
| 3e | Generative met perturbations trained on EDA spread | L | 3d | — | research |
| 3f | Neural-field met representation | L | 3d | — | research |

### Releases and outputs

| ID | Item | Size | Depends on | Numerics | Version |
| --- | --- | --- | --- | --- | --- |
| 4  | Column releases (tower inlets, aircraft profiles) | M | existing item | no | v1 |
| 4a | Satellite-style releases with averaging kernels and endpoints | M | 4 | no | v1 |
| 4b | Release-centred nested footprint output, aligned to coarse edges | S | none | no | v1 |
| 4c | Tagged ensemble members and tangents in the footprint store | S | 2a, 2c | no | v2 |
| 4d | Compressed footprint basis for storage | M | 4a | no | v2 |
| 4e | Adaptive particle count per release | S | none | config-driven | v1 |

### Scale (Track C)

| ID | Item | Size | Depends on | Numerics | Version |
| --- | --- | --- | --- | --- | --- |
| 5a | GATES v2 training campaign on GLIDE ensembles, denoised targets | M | 2a, 2b, 4a, 4b | — | v2 |
| 5b | Active-learning loop: run GLIDE where GATES disagrees | M | 5a | — | v2 |
| 5c | Matrix-free H and Hᵀ products by replay | L | 4a | — | research |
| 5d | ML weather models as a device-resident met source | L | 3b | yes | research |
| 5e | Simulation-based inference with GATES as the simulator | L | 5a | — | research |

### Existing roadmap items retained

| ID | Item | Size | Depends on | Numerics | Version |
| --- | --- | --- | --- | --- | --- |
| 6a | Performance: sub-time and attack the residual phase | S | none | no | v1 |
| 6b | Particle aggregation in the far field | M | 7a first | yes | v1.x |
| 6c | Convection refinements | M | 1d | yes | v1.x |

### Physics specification review (2026-08-22)

| ID | Item | Size | Depends on | Numerics | Version |
| --- | --- | --- | --- | --- | --- |
| 8a | Consistent time-ago bin width in indexing and metadata (task A00) | S | none | output metadata | v1, first |
| 8b | Sub-step saturation diagnostic on the graph path (in task A1) | S | none | no | v1 |
| 8c | Midpoint deposition option; disclose endpoint quadrature (in tasks A2, A3, A5) | S | none | yes | v1 |
| 8d | State that footprint cells are cell-integrated sensitivities (in task B6) | S | none | no | v1 |
| 8e | Notation, asymptotes, polar cap, and "exact" claims in physics.md (in task A8) | S | none | no | v1, before audit |

### Compute and storage efficiency

| ID | Item | Size | Depends on | Numerics | Version |
| --- | --- | --- | --- | --- | --- |
| 7a | Multi-rate time stepping: per-particle outer step | M | 6a | yes | v1 |
| 7b | Particle count and step defaults set by convergence study | S | 1b, 2b | yes | v1 |
| 7c | Short integration plus endpoints for satellites, instead of long footprints | S | 4a | design | v1 |
| 7d | GLIDE-ready met cache: subset, lowest levels, AGL ladder, float16, bit-rounded | M | none | no | v1 |
| 7e | Half-precision met tensors for the trilinear gather | S | 7f | yes | v1 |
| 7f | Local metric coordinates for particle positions | S | none | exact to fp32 | v1 |
| 7g | Sparse or bit-rounded footprint storage | S | none | no | v1 |
| 7h | Batch releases by release time | S | none | no | v1 |

---

## 3. v1: scope and release gate

**The claim v1 makes.** A validated backward LPDM that produces footprints
comparable to FLEXPART and NAME, runs an order of magnitude or more faster on a
GPU, streams or caches meteorology from Zarr, and can be installed and
reproduced by someone outside the group. No uncertainty quantification.

| In v1 | Out of v1 |
| --- | --- |
| Efficiency items that touch numerics (6a, 2b, 7a, 7e, 7f, 1b, 7b) | Tangents, transport-error covariance, gradient calibration (2c–2f) |
| Physics freeze with literature defaults aligned to FLEXPART v11 (1a) | Calibrated parameter posterior and parameter card (1f) |
| Validation Tiers 0–3 (1b–1e) | UK inversion (1g) as the paper's application, not a gate |
| Point, tower, column and satellite releases with endpoints (4, 4a) | Composite met source and nesting (3b, 3c) |
| Nested footprint output (4b) | Learned compression and other research lines |
| Met cache and compressed stores (3a, 7d), sparse output (7g), release-time batching (7h), adaptive particle count (4e) | Matrix-free inversion, ML weather input, SBI |
| In-batch ensembles as plumbing (2a) | |
| Packaging: pip install, container, CLI, tutorial, stable output contract, DOI | |

**Release gate.** v1 ships when every line holds; the checklist is reproduced in
[dev/agent/QUEUE.md](agent/QUEUE.md) as the final task.

- Physics frozen and tagged; any later physics change bumps the minor version
  and carries a decision record.
- Test suite green on CPU in CI and on GPU on Isambard AI, including
  reciprocity, well-mixed and convergence tests.
- Physics audit table complete and signed by an external reviewer.
- Validation report in `docs/` with pre-registered metrics met: FLEXPART twin
  agreement at particle and footprint level; DATEM ranking scores for ETEX,
  CAPTEX and ANATEX, and the ETEX ATMES-II statistics, within the range
  published for the comparison models (NAME, FLEXPART, HYSPLIT, STILT).
- Performance table published: footprints per second on GH200 and on CPU for
  the reference configuration, with a phase profile.
- Reproducibility: pinned environment, container, seeds, archived inputs for
  the validation cases, DOI.
- Documentation: install, quick start, met contract, output contract, physics,
  validation, a worked footprint-to-inversion example.
- Output contract declared stable, with a migration note if it changed.

---

## 4. Sequence

The principle: **efficiency work that alters results goes before the physics
freeze; work that does not can run in parallel and should start early because
it pays back every run after it.** Validation then happens once, on the model
that will actually be run at scale.

| Phase | What | Who | Queue |
| --- | --- | --- | --- |
| **0. Prepare** | Background material, environment, data inventory, golden fixtures, credentials (see [agent-operations.md](agent-operations.md) §2) | human | — |
| **1. Efficiency before the freeze** | time-bin fix (A00) → 6a → 2b → 7a → 7f, 7e → 1b, 7b → 2a | agent, strongest model, sequential | A |
| **1′. Infrastructure, in parallel** | 3a + 7d, 7g, 7h, 4e, 4, 4a, 4b, validation figure tooling, packaging groundwork | agent, cheaper model | B |
| **2. Freeze** | 1a; physics audit table (1c) prepared for the external reviewer; tag `v0-physics` | agent prepares, human signs | A |
| **3. Validate** | Tier 0 additions (reciprocity, convergence figures) → Tier 1 parity → Tier 2 twins → Tier 3 tracer experiments and radon | agent analysis, human reference runs | A |
| **4. Release v1** | Packaging, reproducibility bundle, docs, gate checklist | agent | B |
| **5. Paper** | 1g and 1h | human with agent support | — |

Hard dependencies to start early because they are on the critical path and
need a person: the FLEXPART v11 reference runs on ERA5 (1d), and the
tracer-experiment data and their ERA5 cubes (1e). The download recipe is in
[data/tracer-experiments.md](../data/tracer-experiments.md). Start the FLEXPART runs on the current physics; the
final comparison is repeated on the frozen version.

The GATES queue (`docs/code_review_2026-09-16.md` in the GATES repository) is
independent and can run at any time as a third queue.

---

## 5. After v1

- **v1.x** — composite met source (3b), UKV nesting (3c), far-field particle
  aggregation (6b), convection refinements if validation asks for them (6c).
- **v2** — Track A in full (2c–2e), calibration and the parameter card (1f),
  tagged ensembles in the store (4c), the GATES v2 campaign (5a, 5b),
  compressed footprint basis (4d).
- **Research lines**, each droppable without affecting the rest: learned met
  compression (3d–3f), matrix-free inversion (5c), ML weather input (5d),
  simulation-based inference (5e), learned Hanna corrections (2f).

---

## 6. Effort

Assuming one person directing agents, plus someone able to run FLEXPART:

| Phase | Effort |
| --- | --- |
| Efficiency and convergence, before the freeze | 4–6 weeks |
| Freeze, audit, reciprocity | 2 weeks |
| FLEXPART twins and tracer experiments | 8–12 weeks, dominated by the reference runs |
| Releases, nested output, packaging, in parallel | absorbed |
| **v1** | **roughly 4–5 months from 2026-09-16** |

The estimate is the least reliable thing on this page. Use it to notice
slippage, not as a commitment.
