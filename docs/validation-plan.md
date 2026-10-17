# Validation plan

_This page sets out how GLIDE's transport will be validated. It covers what
remains to be done, in what order, against which references, and how the
results will be shown. What has **already** been verified, and how to run the
existing tests, is in [VALIDATION.md](VALIDATION.md)._

**Contents**

1. [Why a ladder, and why it is pre-registered](#1-why-a-ladder-and-why-it-is-pre-registered)
2. [The tiers](#2-the-tiers)
3. [Pre-registration protocol](#3-pre-registration-protocol)
4. [The case library](#4-the-case-library)
5. [The figure set](#5-the-figure-set)
6. [Figure infrastructure and design rules](#6-figure-infrastructure-and-design-rules)
7. [Parameter calibration (v2)](#7-parameter-calibration-v2)
8. [Data and reference runs required](#8-data-and-reference-runs-required)

**Terms used throughout**

- **Footprint.** The output of a backward run: how sensitive a measurement is to
  a surface emission in each grid cell, in units of concentration per unit
  flux. GLIDE runs backward in time from the measurement point, the
  *receptor*.
- **FLEXPART and NAME.** The established Lagrangian models GLIDE is compared
  with. FLEXPART v11 is the main reference because GLIDE's turbulence scheme
  follows it and both can be run on the same ERA5 meteorology. NAME is the Met
  Office model that the group's UK inversions currently use.
- **The physics freeze.** The point at which the model physics is fixed and
  given a version label in git (a *physics tag*, for example `v0-physics`). All
  validation runs use one tagged version, so every result refers to the same
  model. Any later change to the physics produces a new tag.
- **Decision record.** A short dated note in [dev/decisions/](../dev/decisions/)
  that records a choice and the reason for it.
- **Roadmap items** (for example "roadmap 2a") are listed in
  [dev/roadmap.md](../dev/roadmap.md). Each one is described in words here
  where it is mentioned.

---

## 1. Why a ladder, and why it is pre-registered

Most of GLIDE's code was written by AI coding agents. Other scientists will
assume there are errors. Their question will be where the errors are, and how
anyone would find them. Our answer is a *validation ladder*: a sequence of
tests, from simple to demanding, where:

- each step tests something the previous step could not;
- the metrics and pass/fail tolerances are written down **before** the runs
  are made (this is what "pre-registered" means);
- every figure shows GLIDE and its reference together;
- anyone can reproduce the results from a public benchmark.

Two rules follow.

**GLIDE is never tuned to match another model.** FLEXPART and NAME are used to
check consistency, not as truth. GLIDE's parameters are adjusted only to fit
observations of tracers with known emissions (§2, Tier 3).

**Expected differences are stated in advance.** Some of GLIDE's design choices
differ from FLEXPART's on purpose, and they will produce differences in the
comparisons. Examples include:

- how the vertical grid is built: metres above ground on a fixed set of
  terrain-following levels, interpolated once from ERA5's native model levels
  (the default; ERA5 pressure levels are also supported);
- the vertical velocity: GLIDE converts ERA5's pressure vertical velocity,
  while FLEXPART uses a vertical velocity that flex_extract derives for its
  own coordinate;
- a more accurate time-stepping scheme for the resolved wind;
- an exact, rather than approximate, update for the random turbulent velocity;
- where the minimum turbulence timescales are applied;
- the boundary-layer inputs to the turbulence scheme: GLIDE takes the friction
  velocity and boundary-layer depth directly from ERA5's fields, while
  FLEXPART computes its own, the friction velocity from the surface stress and
  the boundary-layer depth by a critical Richardson-number method
  (Vogelezang and Holtslag, 1996). This is likely to be one of the larger
  sources of near-surface differences in Tier 2.

If a difference was predicted before the comparison was run, it counts as
evidence that the model is understood. If it is explained only after it is
seen, it does not count. The full list is fixed in the pre-registration record
(§3).

---

## 2. The tiers

| Tier | Question it answers | Reference | Status |
| --- | --- | --- | --- |
| 0 | Does the code solve the equations it claims to solve? | Exact mathematical solutions, numerical solutions of the diffusion equation, and conservation checks | Largely done. The reciprocity test and the convergence figures are still missing |
| 1 | Do GLIDE's turbulence and convection schemes give the same numbers as FLEXPART's code? | FLEXPART v11's own routines, run on the same input profile | Not started |
| 2 | Does the full model move particles like the established models when both use the same meteorology? | FLEXPART v11 on ERA5; NAME as a secondary check | Comparison tools exist, but have not been run on the frozen physics |
| 3 | Does GLIDE reproduce measured concentrations from tracer releases with known emissions? | ETEX, CAPTEX and ANATEX observations; radon-222 | Data located; not started |
| 4 | Does an inversion the group already trusts give the same answer with GLIDE footprints? | UK CH₄ inversion with NAME footprints | Not started; this is the paper's application |

### Tier 0: mathematical checks

These tests are already in the test suite (see
[VALIDATION.md §3–4](VALIDATION.md)). Two more are needed.

- **Forward–backward reciprocity.** A backward footprint from a receptor
  should equal the concentration at that receptor from a forward run with a
  unit source in each grid cell. No other test checks the backward formulation
  this directly. The test needs a forward-in-time mode, which will be added for
  this test only. GLIDE stays a backward model in normal use.
- **Convergence studies, with published numbers.** The error against a
  reference solution is measured as each numerical setting is refined. The
  settings are:
  - the time step;
  - the accuracy target that sets how many sub-steps a particle takes within
    one time step (particles in strong, fast-changing turbulence take more);
  - the number of particles;
  - the number of vertical levels the meteorology is interpolated onto.

  The studies use the analytic plume test and one real-meteorology case. For
  each setting, the error should fall at the rate theory predicts. The results
  set the v1 default values (roadmap 7b). They also show that these settings
  were chosen because the answer stopped changing, not to improve agreement
  with observations.

### Tier 1: component-by-component comparison with FLEXPART

The same vertical profile of meteorological inputs is fed to FLEXPART v11's
own subroutines and to GLIDE's equivalents, and the outputs are compared. No
FLEXPART model run is involved. The agent compiles the two FLEXPART routines
from source, calls them through a small Fortran driver program, and feeds both
codes the same inputs: ERA5 columns from the standard cases (§4) and synthetic
stable, neutral and convective profiles. **The human's role** is to provide
the FLEXPART v11 source on Isambard (`GLIDE_FLEXPART_SRC`) and to set the
tolerances in the pre-registration record.

The FLEXPART outputs must come from FLEXPART's compiled code. A re-typed
("transcribed") version of FLEXPART's equations is not acceptable: GLIDE's own
Hanna code was itself transcribed from FLEXPART, so comparing two
transcriptions would only show that they agree with each other.

- **Boundary-layer turbulence** (the Hanna, 1982, scheme). This gives profiles
  of the velocity standard deviations σ_u, σ_v and σ_w and the Lagrangian
  timescales T_L. They are compared against height divided by boundary-layer
  depth (z/h) in stable, neutral and convective conditions. The comparison
  includes the minimum timescales (10 s horizontal and 30 s vertical) that
  FLEXPART imposes near the ground.
- **Deep convection** (Emanuel's mass-flux scheme). This produces a matrix
  saying what fraction of the air at each level is moved to each other level.
  Each column of the matrix should sum to one, so that no mass is created or
  lost.

The tolerances follow the Tier 1 rules in §3: rounding level where the
formulas are the same, and the spread of FLEXPART's own output across the ERA5
ensemble where they are not. Every difference beyond tolerance is classified
as one of three things:

- **a GLIDE bug**, handled by the registered failure rule (§3);
- **a declared design difference**, already on the expected-difference list;
- **a suspected problem in FLEXPART.** FLEXPART is a reference, not truth.
  Where FLEXPART's code appears to disagree with its own documentation or with
  the published scheme (Hanna, 1982; Ryall and Maryon, 1998; Emanuel and
  Živković-Rothman, 1999), the finding is written up with the file and line,
  the published equation, and the size of the effect. GLIDE is not changed to
  match FLEXPART, nor FLEXPART's behaviour adopted, without a decision record.
  Where GLIDE copied the questionable behaviour, that is stated too.
  Confirmed problems are reported to the FLEXPART developers.

### Tier 2: the whole model against FLEXPART on the same meteorology

The main reference is **FLEXPART v11 driven by ERA5**. Both models use ERA5 on
its native model levels: FLEXPART through its preparation tool, flex_extract,
and GLIDE from the same fields in the ARCO ERA5 archive. Differences then come
from the transport, not the meteorology, apart from the declared differences
in how each model derives its vertical velocity (§1). The
comparison works in the order below. Each level is stricter than the one after
it, because each later level averages over more detail.

1. **The particle clouds themselves,** from snapshots of FLEXPART's particle
   positions at 6, 24, 72 and 120 hours for the standard cases. Plotted
   against time:
   - the movement of the cloud's centre;
   - its horizontal spread;
   - the heights below which 10%, 50% and 90% of the particles lie;
   - the fraction of particles inside the boundary layer;
   - the fraction that have left the model domain.

   Each is plotted with a band showing how much the statistic varies from
   particle sampling alone.
2. **Gridded footprints** for the standard cases (§4). The comparison metrics
   are:
   - spatial overlap between the two footprints;
   - correlation of the logarithms of the cell values;
   - the total sensitivity summed over all cells;
   - the sensitivity weighted by distance from the receptor, which measures
     how far upwind the influence extends;
   - a map of the average ratio between the two models over a month, which
     shows any systematic geographic difference.
3. **Concentrations at measurement sites,** after each model's footprints are
   multiplied by the same emissions map.

NAME is a secondary comparison because it is normally driven by Met Office
meteorology, not ERA5. Comparing GLIDE on ERA5 with NAME on Met Office fields
mixes up differences in meteorology with differences in the models, and the
results will be reported with that caveat. A clean comparison with NAME will
become possible once GLIDE can read Met Office fields (roadmap 3b and 3c).

### Tier 3: tracer experiments with known emissions

These are controlled releases of an inert tracer, at a known place, time and
rate, measured at many downwind sites. All three are covered by ERA5. How to
download the data, the file formats and the scoring are described in
[data/tracer-experiments.md](../data/tracer-experiments.md).

- **ETEX** (Europe, 1994). Two 12-hour releases from north-west France,
  measured at about 168 stations in 3-hour samples. The ATMES-II model
  intercomparison (Mosca et al., 1998) published scores for the models of the
  time, including FLEXPART and NAME.
- **CAPTEX** (eastern North America, 1983). Six short releases, measured at 84
  sites 300–800 km away in 3- and 6-hour samples. The data come from NOAA's
  DATEM archive (Data Archive of Tracer Experiments and Meteorology).
- **ANATEX** (North America, 1987). 66 releases from two sites over three
  months, measured at 75 sites in 24-hour samples, at distances of up to
  several thousand kilometres. Also from DATEM.
- **Radon-222** at a few European sites. Radon is emitted from soils at a
  roughly known rate and decays with a half-life of 3.8 days. It is a
  continuous check on boundary-layer mixing, although the uncertainty in its
  emission map limits what it can show.

CAPTEX and ANATEX are especially useful because the same score has already
been published for four established models on them:
- NAME (Selvaratnam, Thomson and Webster, 2023, doi:10.1175/JAMC-D-23-0021.1);
- HYSPLIT, STILT and FLEXPART (Hegarty et al., 2013).

GLIDE can then be ranked among four models rather than against a single
study. Selvaratnam et al. also tuned NAME's representation of unresolved
mesoscale motions: horizontal wind variations too small for the meteorological
model to resolve but larger than boundary-layer turbulence. GLIDE represents
the same process with its *meander* term. Their tuned values will be the
starting estimate (the prior) for GLIDE's meander parameters in the v2
calibration (§7).

**Main metric: the DATEM ranking score.** This is the sum of four terms, each
scaled from 0 (worst) to 1 (best), so a perfect score is 4:

- the squared correlation between measured and modelled concentrations;
- a term based on the fractional bias (the overall over- or under-prediction);
- the figure of merit in space: the overlap between the measured and modelled
  areas where the tracer was detected, as a fraction of their combined area;
- a term based on the Kolmogorov–Smirnov parameter: the largest difference
  between the measured and modelled distributions of concentration values.

For ETEX, the statistics used in ATMES-II are reported as well. Before any
GLIDE result is scored, GLIDE's scoring code must reproduce the scores that
DATEM's own software gives for the example HYSPLIT output in the DATEM
archive. This checks the scoring code itself.

**How GLIDE is run for these experiments.** Each measured sample gets its own
backward run. Particles are released from the sampler's location and height,
spread over the sample's collection period, and run backward in time. The
resulting footprint gives the sample's sensitivity to emissions at each
location. The predicted concentration is that sensitivity at the release site
multiplied by the known release rate. The numbers of samples, and so of
backward runs, are:
- about 3,100 for ETEX;
- 2,281 for CAPTEX;
- about 5,500 per tracer for ANATEX.

Two of ANATEX's tracers are scored, one from each release site, and both were
measured at the same sites and times. One footprint per sample therefore
predicts both. The
published scores for the other models come from forward runs, from the source
to the samplers. Our backward runs should give the same answer. This is
declared in advance as a possible source of difference, and agreement is a
test of GLIDE's backward formulation on real cases.

**Which experiments are used for fitting and which for testing (v2
calibration only).** GLIDE's parameters are fitted to ETEX-1 and CAPTEX. The
fitted model is then tested, without further adjustment, on ANATEX and radon.
ETEX-2 is scored and reported but is not one of the test cases. Most models
reproduced it poorly, for reasons that are not well understood, so it would be
a weak test of whether the calibration has improved the model.

### Tier 4: an inversion

The group's UK CH₄ inversion uses data from the DECC tall-tower network. It is
run twice, once with NAME footprints and once with GLIDE footprints, and the
resulting emission maps and national totals are compared. This is the
application section of the paper, not a requirement for release.

---

## 3. Pre-registration protocol

Pre-registration means fixing, in public and with an independent timestamp,
what will be run and how it will be judged, before any results are seen. It
happens once, after the physics freeze and after the scoring code has been
verified (task B10), and before any Tier 1, 2 or 3 run with GLIDE.

### What is fixed

A decision record in [dev/decisions/](../dev/decisions/) fixes the following:

- **The model version:** the physics tag that all runs will use, and its
  commit hash.
- **The scoring code:** the commit hash of the verified scoring code in
  `scripts/validation/`.
- **The cases:** the standard cases (§4) with their dates, and the full list
  of releases.
- **The metrics**, exactly as coded. This includes three options in the DATEM
  scoring that can change scores a lot:
  - the concentration below which a measurement is treated as zero;
  - whether pairs where both measurement and model are zero are counted;
  - whether samples are compared one by one, or averaged over time or space
    first.
- **The tolerances,** computed by the rules below, and the published scores
  of other models that GLIDE will be judged against.
- **How particles are counted onto the grid.** In normal use, each particle's
  contribution goes entirely into the grid cell that contains it ("delta"
  deposition). An optional alternative spreads it over neighbouring cells to
  reduce noise ("kernel" deposition). All Tier 1–3 runs, model comparisons and
  the v2 calibration use delta deposition. The spreading slightly widens
  plumes, which would be hard to tell apart from the effect of the meander term
  (see [decision 0012](../dev/decisions/0012-kernel-deposition.md)).
- **The expected differences** between GLIDE and FLEXPART in the Tier 2
  comparison (§1).
- **The fitting and test sets:** fit on ETEX-1 and CAPTEX; test on ANATEX and
  radon; report ETEX-2.
- **The development history.** Listing earlier comparisons is not enough. The
  record must also say where earlier results shaped the model. It has three
  parts:
  - *Comparisons already made.* GLIDE was compared with FLEXPART and NAME
    before the freeze, and more comparisons will be run on the pre-freeze
    physics while the reference runs are being set up. Each is listed with
    its date, the reference it used and what it showed, and labelled
    exploratory.
  - *Changes made because of results.* Some physics changes were made after
    seeing results. For example, GLIDE's modelled CH₄ enhancements at polluted
    low-inlet sites came out about twice those from NAME and FLEXPART (about
    120 ppb against under 60). The cause was near-surface mixing 3–8 times
    weaker than FLEXPART's. FLEXPART's minimum turbulence timescales were then
    switched on, and a GLIDE-only surface-layer scheme switched off (commit
    2808a76; [turbulence.md §5](turbulence.md#5-the-lagrangian-timescale-floors)).
    Each such change is listed with its commit and the result that prompted
    it. Changes that came from code review against the literature or
    FLEXPART's source, such as adopting FLEXPART v11's Hanna coefficients
    (commit 01610bd), are listed separately, because no result prompted them.
  - *Data that are therefore not independent.* The UK sites and periods used
    in those development comparisons, which matters for Tier 4, and the
    FLEXPART and NAME output already compared. The tracer experiments and the
    radon data have not been used, which is why Tier 3 is the independent
    test.

  Without this section, a reviewer could reasonably ask whether the model or
  the tolerances were adjusted after seeing the agreement.
- **The rules for running and reporting** (below), so that they are part of
  what is registered.

### How tolerances are set

The tolerances are not chosen by judgement just before registration. By then,
everyone involved will have seen GLIDE's earlier comparisons with FLEXPART,
and numbers chosen then could be shaped by them, even unintentionally.
Instead:

- **The rules are fixed here, now,** before any run on the frozen physics.
- **The numbers are computed later by a script** (task A8a), following these
  rules, from **reference data only**: FLEXPART's output, NAME's output,
  published scores, the ERA5 ensemble, and analytic calculations. No GLIDE
  output of any kind is used.
- **The human checks the calculation, not the numbers.** If a rule turns out
  to give a number that makes no sense, the rule is changed, the reason is
  recorded, and the change is part of what is registered.

**Three outcomes, not pass or fail.** Each comparison is graded:

| Outcome | Meaning |
| --- | --- |
| **Indistinguishable** | The difference from the reference is within the reference's own noise. |
| **Consistent** | Larger than that, but within the spread among established models or within the meteorological uncertainty. |
| **Outside** | Beyond that spread. |

v1 requires "consistent" for every graded metric. "Indistinguishable" is
reported wherever it holds.

**Many comparisons mean some chance failures.** Tier 2 has thousands of
footprints and several metrics. With a 95% noise band, about 5% would fall
outside it even if GLIDE matched FLEXPART exactly. So the criteria apply to
distributions, not to single cases: a metric is graded by the fraction of
cases that meet it, and any failures must show no pattern by region, site or
standard case.

**The rules, tier by tier.** The constants in **bold** are proposals for the
human to confirm or change before this plan is merged.

- **Tier 0.** The existing test tolerances ([VALIDATION.md](VALIDATION.md))
  stand. For the new tests:
  - *Reciprocity:* forward and backward results agree within the combined
    95% particle-noise band (from bootstrap resamples of both runs) in at
    least **95%** of the grid cells that carry non-negligible sensitivity.
  - *Convergence:* the fitted rate at which the error falls is within **25%**
    of the theoretical rate, over the range where the error is above the
    particle noise.
- **Tier 1.** The routines are deterministic, so the tolerances can be tight.
  - Where GLIDE and FLEXPART use the same formula: agreement to within
    single-precision rounding, a relative difference of at most **10⁻⁵**.
  - Where a declared design difference applies and its effect can be
    calculated: GLIDE must match FLEXPART's output with that calculated
    difference applied, to the same rounding level.
  - Where the effect cannot be calculated exactly (mainly GLIDE's reduced
    convection scheme, whose departures from the full scheme are listed in
    [convection.md §7](convection.md)): the difference must be smaller than
    the spread of FLEXPART's own output when it is run on the members of the
    ERA5 ensemble (the EDA) for the same column. In plain terms, GLIDE may
    differ from FLEXPART by no more than FLEXPART differs from itself under
    realistic uncertainty in the meteorology. The EDA is coarser than the main
    ERA5 product and is known to underestimate the true uncertainty, so this
    tolerance errs on the strict side. For convection, the compared
    quantities are the fraction of boundary-layer air moved above 5 km, the
    level of maximum outflow and the column-sum conservation (rounding
    level).
- **Tier 2.** Two levels, both computed from reference data:
  - *Noise floor* (for "indistinguishable"): the 95% band of each metric
    across repeated FLEXPART runs that differ only in their random seed. A
    subset of cases is run five times: one week of hourly releases at three
    sites. The resulting noise, expressed against particle count and footprint
    size, is applied to the full set.
  - *Ceiling* (for "consistent"): the same metric computed between NAME and
    FLEXPART for the same sites and dates. NAME runs on different
    meteorology, so this is the disagreement between established models as
    they are actually used.
  - *Footprints:* for each metric, at least **90%** of footprints within the
    ceiling, and the median within the ceiling for every site and every
    standard case.
  - *Concentrations at sites:* the root-mean-square difference between GLIDE
    and FLEXPART at each site is at most **half** the model-error term the
    group's UK inversion already assumes for that site, and no larger than
    the NAME–FLEXPART difference.
  - *Particle statistics* are diagnostic, graded only against the noise
    floor. They show where differences arise, rather than deciding the
    outcome.
  - *Declared expected differences* (§1) may fall outside the noise floor,
    but must stay within the ceiling, and should have the predicted sign.
- **Tier 3.**
  - *CAPTEX and ANATEX:* "consistent" means GLIDE's DATEM ranking score is at
    least the lowest score published for NAME, HYSPLIT, STILT and FLEXPART on
    that experiment. Scores are computed with the same scoring options as
    each published study. If the studies used different options, GLIDE is
    scored separately with each study's options and compared with that
    study's values.
  - *ETEX-1:* "consistent" means each ATMES-II statistic lies within the
    range of the models published by Mosca et al. (1998).
  - Each score is reported with its particle-noise interval, from bootstrap
    resamples, and with FLEXPART's published score alongside.
  - ETEX-2 is reported without a grade.
- **Radon and Tier 4: reported, not graded.** The uncertainty in the radon
  emission map is too large to support a threshold, so radon results are
  compared descriptively with published radon evaluations. Tier 4 differences
  are expressed relative to the posterior uncertainty (for example, whether
  national totals agree within one standard deviation), and Tier 4 is not a
  release condition.

### The steps

1. **Compute the tolerances.** Agent task A8a applies the rules above to the
   reference data, in a session that is not allowed to open any GLIDE output.
   The human checks that the rules were applied correctly and merges it.
2. **Draft and review.** Agent task A8b drafts the record from this plan, with
   A8a's tolerances unchanged and the development history. The human checks
   it and merges it.
3. **Register.** The human submits the merged record on
   [OSF Registries](https://osf.io/registries), using the open-ended
   registration template. The submission includes the record itself, the two
   commit hashes, and the release lists. OSF timestamps the registration,
   makes it unchangeable and gives it a DOI. It can be embargoed (kept private
   for up to four years) if the tolerances should not be public before the
   paper.
4. **Record.** The human adds the registration's URL, DOI and date to the
   record's "Registration" line, and to "Current state" in
   `dev/agent/PROGRESS.md`.
5. **Run.** Only then do the Tier 1–3 tasks (A9, A10, A11) start. Each one
   stops if the Registration line is empty.

Git history alone is not enough evidence. Commit dates are set by whoever makes
the commit, and history can be rewritten. The decision record stays in the
repository because the agents read their settings from it. The OSF
registration is the evidence of when those settings were fixed, and the paper
cites its DOI.

### Rules for running and reporting

GPU runs are cheap and the agents have a large allocation. That makes it easy
to try many variants and report only the ones that work, and hard for anyone
to see that it happened. Vaccaro (2026, arXiv:2606.11217) makes this argument
for experiments with AI agents. These rules are registered with the record.

1. **Every run is logged and reported.** Each Tier 1–3 task keeps a run log in
   `docs/validation/runs/<tier>.csv`. Each row gives the date, the
   configuration file and its hash, the physics tag, the output location, the
   outcome, and the agent model that launched it. A run may be repeated only
   for a documented failure: a crash, corrupt or missing input data, or a
   configuration that differs from the registered one. The reason goes in
   the log. Both the failed run and the repeat stay in the log, and the
   report includes every run.
2. **What happens when a test fails** is decided now, not after the failure:
   - If the cause is a bug in GLIDE, the bug is fixed under a new physics tag,
     the fix gets a decision record, and the change is filed as an OSF update.
     **Every** tier that the fix could affect is then rerun, not only the
     case that failed. The report gives the results before and after the fix.
   - If the cause is a bug in the scoring code, the same applies, with the
     scoring-code commit in place of the physics tag.
   - If no bug is found, the failure is reported as a finding. Nothing is
     changed to make the test pass.
3. **Pass or fail is computed, not judged.** The registered scripts compute
   every metric and compare it with its tolerance. The agent running the task
   does not decide whether a result passes. It may add interpretation, which
   is labelled as such and kept separate from the computed results.
4. **The agent is recorded.** Each validation task's progress file gives the
   exact model identifier of every agent session that worked on it, and the
   run log gives the model for each run. The agents write and run the
   validation code, so they are part of the measurement process and are
   reported like any other instrument.

### After registration

Any change is recorded in a new decision record that gives the reason, and is
filed as an update to the OSF registration, linked to the original. Results are
reported against the registered numbers whether or not they pass. Tolerances
are never loosened to make a test pass; a failed tolerance is reported as a
finding.

---

## 4. The case library

Six standard cases are chosen once and used in every tier and every figure.
Readers then get to know a few situations well and can follow each one up the
ladder. Each case is chosen to:

- test one meteorological situation or model feature;
- have FLEXPART and NAME output available for comparison;
- where possible, include a site with observations.

| Case | What it tests | Candidate |
| --- | --- | --- |
| Stable night, low inlet | Weak night-time mixing near the ground, where the minimum turbulence timescales matter | A UK or ICOS tower, winter night |
| Convective summer afternoon | Daytime convective turbulence and boundary-layer depth | A continental site in July |
| Frontal passage | Transport by the resolved winds, and interpolation between hourly meteorology | A tower during a well-documented front |
| Mountain site | The terrain-following vertical coordinate | Jungfraujoch or similar |
| Coastal site | The land–sea transition, and features smaller than a grid cell | Mace Head |
| Satellite column | Releasing particles through a whole atmospheric column, as for a satellite retrieval, and how runs are ended | A GOSAT or TROPOMI sounding over Brazil |

The exact dates are fixed in the pre-registration record.

---

## 5. The figure set

Three rules apply to every figure:
- the reference is shown with GLIDE;
- every comparison shows the difference as a log ratio (the logarithm of
  GLIDE's value divided by the reference value);
- every comparison shows the particle-sampling noise, as a band estimated by
  repeating the calculation on random subsets of the particles.

The standard cases (§4) are used throughout.

**Tier 0**
- **Taylor dispersion.** Particle spread (variance σ²) against time, with
  Taylor's (1921) analytic curve. The spread grows as t² at short times and as
  t at long times, and both limits are shown.
- **Well-mixed test.** Particles that start evenly mixed in the vertical should
  stay evenly mixed, even where turbulence varies with height. Vertical
  profiles are shown at several times against a uniform line, one panel per
  stability regime. A companion panel switches off the drift correction that
  makes this work, and shows particles wrongly piling up in the weakly
  turbulent layers.
- **Analytic plume.** GLIDE's 2-D footprint and the exact solution for a plume
  in uniform wind and turbulence, side by side, with their log ratio. A further
  panel shows the crosswind-integrated profile against distance.
- **Diffusion limit.** Particle density at three times against a numerical
  solution of the diffusion equation with the same height-varying
  diffusivity.
- **Reciprocity.** The forward-run concentration beside the backward
  footprint, and a scatter plot of one against the other around the 1:1 line.
- **Convergence.** Error against time step, sub-step accuracy target and
  particle count, on log axes, with the slope theory predicts.
- **Terrain.** A particle trajectory crossing a hill, with the terrain shaded,
  showing that the height above ground stays constant.

**Tier 1**
- Turbulence profiles for the three stability regimes, GLIDE and FLEXPART
  overlaid, with the percentage difference underneath.
- Heat maps of the convection redistribution matrix from both codes, with each
  column's sum plotted as a line to show mass conservation.

**Tier 2**
- Snapshots of the particle cloud at 6, 24, 72 and 120 hours, as a map and a
  vertical cross-section, GLIDE beside FLEXPART, plus an animation.
- The cloud statistics from §2 against time, with one line per model and a
  noise band in each panel.
- For each case, three panels: GLIDE's footprint, FLEXPART's footprint and the
  log ratio. Both footprints use the same logarithmic colour scale, in
  physical units.
- A map of the log ratio averaged over a month, to show systematic
  differences.
- Statistics over thousands of footprints:
  - a scatter plot of total sensitivity;
  - distributions of each comparison metric;
  - box plots by site.
- Concentration time series at sites from GLIDE, FLEXPART and NAME, with
  scatter plots and average daily cycles.

**Tier 3**
- **ETEX:**
  - maps of modelled concentration at each sampling time, with the station
    measurements as coloured points on the same scale;
  - a map of tracer arrival times;
  - time series at individual stations;
  - GLIDE's figure of merit in space as a bar beside the published values for
    FLEXPART and NAME.
- **CAPTEX and ANATEX:**
  - maps of measured and modelled concentration for each release;
  - scatter plots of measured against modelled values for each sample;
  - the DATEM ranking score and its four terms as bars, beside the published
    values for NAME, HYSPLIT, STILT and FLEXPART.
- **Radon:** time series and average daily cycles at each site.

**Tier 4**
- Maps of the inferred emissions from the GLIDE-based and NAME-based
  inversions.
- National totals with uncertainty bars.
- The fit of each inversion's modelled concentrations to the observations.

**Calibration (v2)**
- Bar charts ranking parameters by how much they affect the results.
- For each parameter, the starting distribution (prior) against the fitted
  distribution (posterior).
- Maps of how much footprints vary across the range of fitted parameters.
- A check that the calibrated model predicts the held-out test experiments.

---

## 6. Figure infrastructure and design rules

- Figures are produced by **scripts, not notebooks**, under
  `scripts/validation/`, directly from the run outputs. Scripts can be rerun
  automatically, so every figure is regenerated whenever the physics tag
  changes.
- All figures are collected on a **gallery page** in `docs/validation/`,
  organised by physics tag, so a reader can compare the same figure across
  versions.
- Each validation task in the agent work queue lists the figures it must
  produce as part of being complete, so the gallery fills up as the work
  proceeds.
- An **interactive viewer** for browsing the cases (choose a site and time,
  see GLIDE beside the reference) is built from the existing footprint-explorer
  notebook. It is exported as a standalone web page for the paper's
  supplementary material.
- **Design rules:**
  - the same layouts, colour scales and units in every tier;
  - footprints on logarithmic colour scales, in the physical units used by
    STILT (concentration per unit surface flux);
  - difference panels always show the log ratio;
  - particle-noise bands wherever particles are involved;
  - every figure title gives the physics tag and a short identifier of the
    exact run configuration (a hash of the configuration file).

  Follow the project's visualisation guidance so that the figures read as one
  consistent set.

---

## 7. Parameter calibration (v2)

This is not part of v1. v1 uses published default values matching FLEXPART
v11, so that the community's first comparisons are like for like. The plan is
recorded here because Tier 3 is designed around it.

1. **Separate numerical settings from physical parameters.**
   - Numerical settings are fixed in Tier 0 by refining them until the answer
     stops changing, never by fitting to data. They are the time step,
     sub-step accuracy target, particle count, vertical levels and kernel
     width.
   - Physical parameters have published starting values with uncertainties,
     and these are what is calibrated. They are:
     - the Hanna coefficients;
     - the minimum turbulence timescales;
     - the timescale and strength of the meander term;
     - the constants in the scheme for turbulence above the boundary layer,
       which depends on the gradient Richardson number;
     - the depth of a thin layer next to the ground in which turbulence
       statistics are held constant;
     - the convection scheme's constants.
2. **Screen the parameters.** GLIDE can run many copies of a release, each
   with different parameter values, in a single GPU run (roadmap 2a). This is
   used for a standard sensitivity analysis (the Morris or Sobol method) over
   the physical parameters. The outputs are the footprint and concentration
   metrics. The aim is to find the few parameters that matter.
3. **Calibrate those few parameters.** They are fitted against ETEX-1 and
   CAPTEX using Bayesian inference, so that each parameter gets a probability
   distribution rather than a single best value. The fit needs the derivative
   of the outputs with respect to each parameter. Where possible, this comes
   directly from GLIDE by automatic differentiation (roadmap 2c). Particle
   trajectories are chaotic, so these derivatives may not be meaningful over
   long runs; roadmap 2c includes a test of this on a 24-hour case. Where they
   are not meaningful, a fast statistical approximation fitted to a set of
   GLIDE runs (an emulator) is used instead. The calibrated model is tested on
   ANATEX and radon. Selvaratnam et al. (2023) provide the starting values for
   the meander parameters.
4. **Release the result.**
   - The posterior mean values become the v2 defaults.
   - A table in the documentation gives each parameter's prior, posterior and
     sensitivity.
   - The posterior covariance of the parameters is used to estimate how
     uncertain each footprint is, as an input for inversions (roadmap 2e).
5. **Keep structural uncertainty separate.** The uncertainty from choosing one
   scheme over another (for example, a different turbulence scheme) is
   estimated by running each alternative. It is reported separately from the
   uncertainty in parameter values.

Tower measurements of greenhouse gases are a weak basis for calibrating
transport on their own, because the emissions are also uncertain: errors in
transport and in emissions can compensate for each other. In the tracer
experiments the emissions are known, which is why they anchor the
calibration.

---

## 8. Data and reference runs required

| Need | Tier | Who provides it | Notes |
| --- | --- | --- | --- |
| ERA5 data downloaded locally for the standard cases, on model levels (the default); pressure levels only for comparison | 0–2 | Human, with the existing download scripts | Listed in `dev/agent/PROGRESS.md` |
| FLEXPART v11 on ERA5 via flex_extract: gridded output for each release; particle-position snapshots at 6, 24, 72 and 120 h for the six standard cases; five repeat-seed runs of one week of hourly releases at three sites | 2 | Human | FLEXPART's output does not depend on GLIDE, so these runs can be done once, early. The repeat-seed runs give the Tier 2 noise floor (§3). Run each standard case as its own FLEXPART run with a single release: FLEXPART's particle-position files do not record which release a particle came from |
| ERA5 ensemble (EDA) members for the columns of the standard cases | 1 | Human, from the Copernicus Climate Data Store ("ensemble members" product type; not in the Google Cloud ERA5 copy) | Small: single columns, pressure levels. For the Tier 1 tolerance rule (§3) |
| The model-error term the group's UK inversion assumes at each site | 2 | Group | For the Tier 2 concentration rule (§3) |
| FLEXPART v11 source code on Isambard, for the Tier 1 component comparison (no FLEXPART run is needed for Tier 1) | 1 | Human; set `GLIDE_FLEXPART_SRC` | GPL: use locally, never copy into the repository; `gfortran` on the compute node |
| NAME footprints for the same sites and dates | 2 | Group archive (restricted) | Secondary comparison, and the ceiling in the Tier 2 tolerance rules (§3) |
| ETEX observations (JRC) | 3 | Human, using a download script | Instructions in [data/tracer-experiments.md](../data/tracer-experiments.md) |
| CAPTEX and ANATEX observations, DATEM's scoring software and its HYSPLIT example output (NOAA DATEM) | 3 | Human, downloaded through a web browser | Same page |
| ERA5 for ETEX (Oct–Nov 1994, Europe), CAPTEX (Sep–Oct 1983) and ANATEX (Jan–Mar 1987), both North America | 3 | Human, with the existing download scripts | On model levels. ANATEX is the largest download; estimate its size first |
| Radon-222 concentrations and an emission map | 3 | Human | |
| EDGAR or a similar emissions map, for the concentration comparison | 2 | Group archive (restricted) | |
| UK DECC network data and the group's inversion setup | 4 | Group | Needed at the paper stage |

Restricted data never goes into the repository or the tests (see
[data/README.md](../data/README.md)). The public benchmark (roadmap 1h) uses
only the ERA5 subsets, the FLEXPART reference outputs and the tracer
experiment data. These are downloaded from their original archives, not
redistributed.
