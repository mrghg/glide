# Validation plan

_The plan and protocol for validating GLIDE's transport physics. What has
**already** been verified, and how to run the existing suite, is in
[VALIDATION.md](VALIDATION.md); this page is what remains to be done, in what
order, against what, and how the results will be shown._

**Contents**

1. [Why a ladder, and why it is pre-registered](#1-why-a-ladder-and-why-it-is-pre-registered)
2. [The tiers](#2-the-tiers)
3. [Pre-registration protocol](#3-pre-registration-protocol)
4. [The case library](#4-the-case-library)
5. [The figure set](#5-the-figure-set)
6. [Figure infrastructure and design rules](#6-figure-infrastructure-and-design-rules)
7. [Parameter calibration (v2)](#7-parameter-calibration-v2)
8. [Data and reference runs required](#8-data-and-reference-runs-required)

---

## 1. Why a ladder, and why it is pre-registered

Most of GLIDE was written with coding agents. The community's question will
not be "is the code wrong" but "where is it wrong, and how would anyone know".
The answer is a validation ladder in which each rung is independent of the
last, the metrics and tolerances are written down before the runs are made,
every result carries its reference in the same figure, and everything is
reproducible by a sceptic from a public benchmark.

Two rules follow. **Nothing is tuned to another model.** FLEXPART and NAME are
references for consistency, never truth; the only calibration targets are
observations with a known source. **Expected differences are declared in
advance.** GLIDE differs from FLEXPART by design (terrain-following AGL
coordinate, second-order advection, exact OU integration, the timescale
floors); a difference predicted before the comparison is evidence, one
explained afterwards is not.

---

## 2. The tiers

| Tier | Question it answers | Reference | Status |
| --- | --- | --- | --- |
| 0 | Does the engine solve the equations it claims to? | closed-form solutions, PDE references, invariants | largely done; reciprocity and convergence figures missing |
| 1 | Are the parameterisations the same numbers as the reference implementation? | FLEXPART v11 routines on the same column | not started |
| 2 | Does the assembled model transport particles like the incumbents on identical meteorology? | FLEXPART v11 on ERA5; NAME secondary | tooling exists, not run on frozen physics |
| 3 | Does it reproduce a real tracer with a known source? | ETEX, radon-222 | not started |
| 4 | Does an inversion the group trusts give the same answer with GLIDE footprints? | UK CH₄ with NAME footprints | not started; paper application |

### Tier 0 — analytic and internal

Exists in the test suite (see [VALIDATION.md §3–4](VALIDATION.md)). Two additions:

- **Forward–backward reciprocity.** A backward footprint from a receptor must
  equal the forward concentration from a unit source at every cell. This tests
  the backward formulation directly, which nothing else does. Implemented as a
  forward integration mode used only by the test (GLIDE remains a backward
  model in production).
- **Convergence studies as published numbers.** Error against timestep,
  sub-step accuracy target, particle count and vertical ladder, on the analytic
  plume and on a real-met case, with the expected slopes. These set the v1
  defaults (roadmap 7b) and demonstrate that numerical parameters are converged
  rather than tuned.

### Tier 1 — component parity

Compute on the same column, from the same inputs, with FLEXPART v11's own
routines and with GLIDE's:

- Hanna σ_u, σ_v, σ_w and T_L profiles against z/h in the stable, neutral and
  convective regimes, including the timescale floors.
- The Emanuel mass-flux displacement matrix and its column conservation.

Tolerance: relative difference under a stated percentage on every level, with
the residual explained by documented design differences.

### Tier 2 — model twins on identical meteorology

Primary reference: **FLEXPART v11 driven by ERA5** via flex_extract, so both
models read the same fields and differences are transport physics. Compare in
this order, because each level is stricter than the next:

1. **Particle statistics** from FLEXPART's particle output: centroid
   displacement, horizontal spread, vertical percentiles, boundary-layer
   fraction and domain-exit fraction against time, with particle-noise bands.
2. **Gridded footprints** on the case library: overlap, log-space correlation,
   integrated sensitivity, distance-weighted sensitivity, and the systematic
   log-ratio map over a month.
3. **Mole fractions** after convolution with a flux map at the sites.

NAME is secondary because it runs on Met Office fields; a GLIDE-on-ERA5 versus
NAME-on-UM comparison confounds meteorology with model and is reported as such.
Once the composite met source exists (roadmap 3b, 3c) a GLIDE-on-UM twin
becomes possible and is the clean NAME comparison.

### Tier 3 — observations with a known source

- **ETEX** (1994). Two releases, ERA5 covers the period, and the published
  ATMES-II figure-of-merit scores for FLEXPART and NAME give a scale. GLIDE is a
  backward model, so ETEX is run receptor-oriented: one backward release per
  station per sampling interval, concentration = footprint × source strength.
  This also exercises reciprocity on a real case.
- **Radon-222** at a few European sites, as a continuous check on
  boundary-layer mixing, with the flux-map uncertainty acknowledged.

Release one of ETEX is the calibration set (v2); release two and radon are
held out for testing.

### Tier 4 — inversion

UK CH₄ with the DECC network: GLIDE footprints substituted for NAME footprints
in the same inversion, posterior fluxes and country totals compared. This is
the paper's application section rather than a release gate.

---

## 3. Pre-registration protocol

Before any Tier 1–3 run is made, a decision record in
[dev/decisions/](../dev/decisions/) fixes:

- the frozen physics tag the runs use;
- the case library (§4) and the full release sets;
- the metrics, exactly as implemented in `scripts/validation/`;
- the tolerances and the published scores they are judged against;
- the expected-difference list for the twins;
- the data splits (ETEX-1 fit, ETEX-2 and radon test).

Any change after that point is logged as a further decision record with its
reason. Results are reported against the pre-registered numbers whether or not
they pass. Tolerances are never loosened to pass; a failed tolerance is a
finding.

---

## 4. The case library

Six canonical situations, chosen once and used in every tier and every figure,
so that a reader learns the cases and can follow them up the ladder. Selection
criteria: each isolates one regime, each has FLEXPART and NAME reference output
available, and each has an observational site where possible.

| Case | Regime it isolates | Candidate |
| --- | --- | --- |
| Stable night, low inlet | near-surface timescale floors | a UK or ICOS tower at night, winter |
| Convective summer afternoon | convective-regime Hanna, boundary-layer depth | continental site, July |
| Frontal passage | resolved transport, time interpolation | a tower during a documented front |
| Mountain site | terrain-following coordinate | Jungfraujoch or similar |
| Coastal site | land–sea transition, sub-grid representation | Mace Head |
| Satellite column | column release geometry, endpoints | a GOSAT or TROPOMI sounding over Brazil |

The exact dates are fixed in the pre-registration record.

---

## 5. The figure set

Three principles: every figure carries its reference; every comparison shows
the difference (as a log-ratio) and the particle-sampling noise (as a band from
bootstrapped particle subsets); the case library is used throughout.

**Tier 0**
- Taylor dispersion σ²(t) against the analytic curve with both asymptotes.
- Well-mixed vertical profiles at several times against the uniform line, one
  panel per stability regime, with a companion "drift off" panel showing failure.
- Analytic plume: 2-D footprint, analytic footprint, log-ratio, crosswind-
  integrated profile against distance.
- Diffusion limit: particle density at three times against the PDE solution.
- Reciprocity: forward concentration beside the backward footprint; scatter on
  a 1:1 line.
- Convergence: error against dt, sub-step target and particle count on log axes
  with expected slopes.
- Terrain: a trajectory crossing the hill with terrain shaded and AGL flat.

**Tier 1**
- Hanna profiles for the three regimes, GLIDE and FLEXPART overlaid, percentage
  difference underneath.
- Convection displacement matrix heat maps for both codes, column-sum
  conservation as a line.

**Tier 2**
- Particle cloud snapshots at 6, 24, 72, 120 h, map and vertical cross-section,
  GLIDE beside FLEXPART, plus an animation of the same.
- Cloud statistics against time (centroid, spreads, vertical percentiles,
  boundary-layer fraction, exit fraction), two lines and a noise band per panel.
- Footprint triptychs per case: GLIDE, FLEXPART, log-ratio, shared log colour
  scale in physical units.
- Systematic-difference map: mean log-ratio over a month.
- Population statistics over thousands of footprints: scatter of integrated
  sensitivity, metric distributions, per-site box plots.
- Mole-fraction time series at sites (GLIDE, FLEXPART, NAME), scatter, diurnal
  composites.

**Tier 3**
- ETEX concentration maps at sampling times with station observations as points
  on the same scale; arrival-time map; station time series; figure-of-merit bar
  beside the published FLEXPART and NAME values.
- Radon time series and diurnal composites per site.

**Tier 4**
- Posterior flux maps (GLIDE-based, NAME-based), country totals with uncertainty
  bars, posterior fit to observations.

**Calibration (v2)**
- Sensitivity ranking bars; prior against posterior per parameter; maps of
  footprint ensemble spread; posterior predictive check on the held-out release.

---

## 6. Figure infrastructure and design rules

- Figures are produced by **scripts, not notebooks**, under
  `scripts/validation/`, from the run outputs, and are regenerated automatically
  whenever the physics tag changes.
- They are collected on a **gallery page** in `docs/validation/` versioned by
  physics tag, so a reader can see the same figure at successive versions.
- Each validation task in the agent queue lists its figures as acceptance
  criteria, so the gallery accumulates as work proceeds.
- An **interactive explorer** for browsing cases (site, time, GLIDE against
  reference) is built from the existing footprint-explorer notebook and exported
  as static HTML for the paper supplement.
- **Design rules:** identical layouts, colour scales and units across tiers; log
  colour scales for footprints in physical (STILT) units; difference panels
  always log-ratio; noise bands always present where particles are involved;
  configuration hash and physics tag in every title. Follow the project's
  visualisation guidance so the set reads as one system.

---

## 7. Parameter calibration (v2)

Out of scope for v1, which ships literature defaults aligned to FLEXPART v11 so
that the first comparison the community makes is like for like. Recorded here
because Tier 3 is designed around it.

1. **Split the parameters.** Numerical parameters (timestep, sub-step target,
   particle count, ladder, kernel bandwidth) are converged in Tier 0, never
   tuned. Physical parameters (Hanna coefficients, timescale floors, meander
   timescale and amplitude, gradient-Richardson constants, basal-layer depth,
   Emanuel constants) carry literature priors and are calibrated.
2. **Screen** with in-batch parameter ensembles (roadmap 2a): a Morris or Sobol
   design over the physical parameters with footprint and mole-fraction
   metrics as outputs, to find the few that matter.
3. **Calibrate** the influential few against ETEX release one, Bayesian rather
   than point estimation, using forward-mode tangents (roadmap 2c) where the
   chaos test allows and an ensemble emulator where it does not. Test on ETEX
   release two and radon.
4. **Ship** the posterior mean as the v2 default, a parameter card in the docs
   (prior, posterior, sensitivity index per parameter), and the covariance as
   the linearised transport-error term (roadmap 2e).
5. **Keep structural uncertainty separate.** Scheme swaps are a structural
   ensemble, reported distinctly from the parametric posterior.

Identifiability is the risk with tower data alone; ETEX removes the flux
unknown, which is why it anchors the calibration.

---

## 8. Data and reference runs required

| Need | Tier | Who provides | Notes |
| --- | --- | --- | --- |
| Local ERA5 cubes for the case library (pressure and model level) | 0–2 | human, existing download tooling | inventory in `dev/agent/PROGRESS.md` |
| FLEXPART v11 built with ERA5 via flex_extract; particle and gridded output for the release sets | 1, 2 | human | start on current physics; repeat on the frozen tag |
| FLEXPART v11 source for the parity scripts | 1 | human | licence permitting |
| NAME footprints for the same sites and dates | 2 | group archive (restricted) | secondary comparison |
| ETEX release data and station concentrations; ERA5 for Oct–Nov 1994 | 3 | human (public data), download tooling | |
| Radon-222 concentrations and a flux map | 3 | human | |
| EDGAR or equivalent flux maps for the mole-fraction level | 2 | group archive (restricted) | |
| UK DECC network data and the group's inversion setup | 4 | group | paper stage |

Restricted data never enters the repository or the tests
([data/README.md](../data/README.md)); the public benchmark (roadmap 1h) uses
only the ERA5 subsets, the FLEXPART reference outputs and ETEX.
