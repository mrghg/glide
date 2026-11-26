# 0029 — Trapezoid residence counting, scored against fine-time residence (proposed default)

**Status: proposed.** `trapezoid` is implemented as an option; the default stays
`endpoint` until Matt decides. This record proposes making `trapezoid` the
default, with the evidence below.

**Plain-language explainer** (what the rules are, what each gets wrong, and the
age-bin split, with a figure): [docs/physics.md §7.1](../../docs/physics.md#71-explainer-counting-a-particles-residence-within-a-step).

**Context.** The footprint approximates, for each particle and step, the time
integral $\int 1[\mathbf{x}(t)\in\text{cell}]\,w/\rho_d\,dt$ from one or two
positions. Decision 0026 added `step_mean` (the mean of the step's start and end
positions) next to the default `endpoint`; PR #66 then found that `step_mean`
moved 0–40 m residence away from the ground when a step's vertical displacement
is comparable to the 40 m layer. All of those numbers compared one rule with the
other, on the turbulence before decisions 0027 (16 fixed sub-steps) and 0028 (the
boundary-layer-top interface). The external review (`SUBSTEPPING-REVIEW.md` §4)
pointed out that a difference from `endpoint` cannot fix the sign of an error,
and that the trapezoid rule's smooth-integrand argument does not hold for a cell
indicator: score the rules against fine-time residence on the same trajectories.

**Decision (proposed).**
- Add `output_grid.deposition.position: trapezoid`: each step deposits
  $\Delta t/2$ at the particle's start position and $\Delta t/2$ at its end
  position, as two literal deposits (no "endpoint plus corrections" shortcut).
  The start half stands for the ages $[e, e + \Delta t/2)$ and the end half for
  $[e + \Delta t/2, e + \Delta t)$, and each is split by its overlap with the
  time-ago bins (see "Residence counting fixes" below; the first version binned
  each half by a single age). With
  dry-air weighting each half has its own density, at its own position
  and time (the start at $t$, the end at $t - \Delta t$, in the step's met
  window). The first step of a particle is counted half at its release position;
  a particle that leaves the met domain in a step keeps its start half; a
  multi-release batch needs nothing special (each particle carries its own age
  and release index). The start positions are copied outside the captured step,
  as for `step_mean` (decision 0004).
- **Propose `trapezoid` as the default.**
- **Remove `step_mean`** (in its own commit, so it can be dropped if Matt
  disagrees). Reasons: it is measured biased low in the 0–40 m layer (below);
  no run uses it; and it is the only rule that averages positions, so the only
  one that depends on the horizontal frame (its longitude wrap would corrupt
  steps in the metric frame proposed in PR #74). Configuration validation
  rejects it, and `midpoint`, with a message giving the reason.
- `endpoint` stays available and bit-identical.

**Evidence** (outputs in `$GLIDE_AGENT_OUTPUTS/agents/A2/counting/`; scorer
`scripts/validation/score_residence_counting.py`). Labels: *measured* (a file),
*derived* (shown), *argued*.

- *Measured, scorer design.* The production step (60 s, 16 sub-steps) is hooked to
  record each particle's start, advected position, the 16 sub-step positions and
  the end; the path is rebuilt at the sub-step times with advection and meander
  spread uniformly over the step. The reference is the midpoint rule on that
  path with m samples per sub-step (m = 8, 0.47 s), each sample with its own
  cell, age and 1/ρ. The three rules are counted offline from the same
  trajectories, exactly as `_deposit_step` counts them; on the smoke run their
  totals match the production runs of each rule (same seed) to float32
  accumulation (`scores/check_s4*.txt`), and the hooked run is bit-identical to
  an unhooked one.
- *Measured, convergence.* 1, 2 and 4 samples per sub-step differ from 8 by
  ≤ 0.025% in every region of the smoke run and the synthetic columns; the
  composite trapezoid on the sub-step positions by ≤ 0.1%
  (`tables/reference_convergence.txt`).
- *Measured, smoke* (CPU, Mace Head 10 m inlet, 1 day, seeds 42/43/44; error
  against the reference, 0–40 m, mean over seeds, s.d. ≤ 0.27 points;
  `tables/scores_smoke_w.md`):

  | rule | receptor cell | ring 1 | rings 2–3 | rings 4–10 | beyond 10 | 0–40 m total | 40–1000 m | first hour | cell-wise L1 |
  |---|---|---|---|---|---|---|---|---|---|
  | `endpoint` | −10.77% | +0.17% | −0.12% | +0.03% | +0.07% | −1.25% | +0.04% | −7.59% | 1.62% |
  | `step_mean` | +0.74% | −2.87% | −3.85% | −4.91% | −4.02% | −3.46% | +0.21% | +0.09% | 3.64% |
  | `trapezoid` | −0.44% | +0.17% | −0.12% | +0.03% | +0.02% | −0.03% | +0.00% | −0.34% | 0.35% |

  Plain residence seconds (no density) give the same numbers to 0.04 points
  (`tables/scores_smoke_u.md`).
  The same release at 12 UTC (daytime, 3 seeds, `tables/scores_smokeday_w.md`):
  receptor cell −7.69% / +0.59% / −0.28%, 0–40 m total −0.60% / −0.48% / −0.02%,
  rings within ±0.06% for `trapezoid`, −0.1 to −1.4% for `step_mean`.
- *Measured, synthetic Hanna plumes* (production step, 10 m release, 25 km
  cells, 8 m s⁻¹, 6 h, 3 seeds; `tables/scores_synth_w.md`): receptor cell
  `endpoint` −2.7% (stable) to −11.0% (convective), `step_mean` −9.8% to +0.7%,
  `trapezoid` −0.31% to +0.10%; 0–40 m total `endpoint` −0.4 to −3.1%,
  `step_mean` +0.5 to −14.0%, `trapezoid` within ±0.07%. With 1 km cells and a
  3 m s⁻¹ wind (5.5 steps per cell), single crossings show for every rule:
  receptor cell −23% / +3.6% / −2.0% and ring 1 +1.9% / −0.9% / +1.9%
  (`endpoint` / `step_mean` / `trapezoid`), cell-wise L1 3.25% / 0.94% / 0.72%.
- *Measured, equilibrated columns* (horizontally uniform, no receptor):
  `step_mean` −14.6% in 0–40 m at $u_*$ = 1 m s⁻¹ ($h$ = 300 and 1,000 m),
  −6.4% convective, −0.9% at $u_*$ = 0.4 m s⁻¹; `endpoint` and `trapezoid`
  within 0.1%.
- *Measured, analytic plume* (u′ = 0, release at 50 m above the 0–40 m layer):
  `trapezoid` = `endpoint` in 0–40 m, as derived below. The ranking of
  `endpoint` and `step_mean` flips with where the cell edges fall relative to
  the step times (near-field column error at 5 s: 2.0% / 0.0% with the test's
  edges on step times, 0.4% / 2.0% with the edges moved; at 60 s every rule
  aliases, 26% in the far field), so this case cannot rank the rules.
- *Measured, production runs against `endpoint`* (same seeds, paired;
  `tables/production_*.md`): smoke (3 seeds) `trapezoid` +11.66 ± 0.03% in the
  receptor's 0–40 m cell, +1.25% on the 0–40 m total, ≤ 0.05% in every ring;
  `step_mean` +12.9% in the cell, −3.0 to −4.9% in the rings, −2.24% in total.
  CUDA smoke (static path, graph capture) agrees: `trapezoid` +11.68% / +1.24%.
  Reference config (GPU, 144 releases, 5 days back, seeds 42/43/44,
  `tables/production_ref.md`): `trapezoid` changes only the 10 m inlet — Mace
  Head receptor cell +7.49 ± 0.01%, 0–40 m total +0.37% (day +0.26%, night
  +0.45%), rings ≤ 0.05%; Ridge Hill (90 m) and Tacolneston (185 m) ≤ 0.04%
  everywhere, since their release half is above the layer. `step_mean`: 0–40 m
  total −2.0 / −3.2 / −3.5% (MHD / RGL / TAC), down to −16% in Ridge Hill's
  rings 2–3 at night and −13% in Tacolneston's receptor cell at night. Endpoint's
  own per-release seed spread of the 0–40 m total: 0.26–0.44% (median).

**Rationale.**
- *Derived.* Summed over consecutive steps, `trapezoid` equals `endpoint` plus
  $\Delta t/2$ at each particle's first position, minus $\Delta t/2$ at its last,
  with density evaluated at the node's own time. So away
  from the release it inherits `endpoint`'s interior sampling, which the scorer
  shows to be unbiased at 60 s (rings within ±0.2% on the smoke run), and near
  the release it supplies the half step `endpoint` never counts.
- *Measured, mechanism isolated.* `endpoint`'s error is the first step: the
  right-rectangle rule on the sub-step nodes (`rect_k`) has a receptor-cell
  error proportional to its spacing (smoke: −0.66, −1.32, −2.6, −5.2% at 3.75,
  7.5, 15, 30 s, −10.8% at 60 s), while every other region is within ±0.2%.
- *Measured, mechanism isolated.* `step_mean`'s far-field deficit is the mean of
  two heights: in the horizontally uniform equilibrated column (no receptor, no
  horizontal structure) it counts 15% too little in 0–40 m at $u_*$ = 1 m s⁻¹
  and 6% in the convective column, against ≤ 0.1% for the other two rules.
- *Measured, with the mechanism argued.* The review's concern (one crossing can
  be off by up to Δt/2) is real per particle-step. Where crossings are locked in
  phase to the step times — cells only a few steps wide in a steady wind, or the
  analytic plume's deterministic along-wind motion — every rule shows per-cell
  errors of a few per cent (±2% for `trapezoid` with 1 km cells; aliasing in the
  plume). Where turbulence and varying winds spread the crossing times (the
  smoke run, 25 km cells), they average out: `trapezoid`'s cell-wise L1 error is
  0.35% of the 0–40 m total.

**Rejected alternatives.**
- *Keep `endpoint`:* biased low in the receptor's cell by about one step's
  release residence (−11% on the smoke run, −2.7 to −11% in the synthetic
  plumes), which is where the footprint is largest for a surface inlet.
- *`step_mean`:* measured biased low in the 0–40 m far field whenever a step's
  vertical displacement is comparable to the layer (−3 to −5% on the smoke run,
  up to −15% in strong turbulence).
- *An "endpoint plus release half minus final half" shortcut:* equal only when
  densities and bins coincide at shared nodes; the literal rule is cheap enough
  and has no special cases.
- *Finer outer steps:* `endpoint`'s receptor error falls with Δt, but the cost is
  the whole step, for an error that one extra deposit removes.

**Amends decision 0022.** 0022 evaluates the dry-air density "at the
particle's position and the step's time, using the same met window and time
weight as the step" (the step's mid-time). With `trapezoid` each half uses the
density at its own time instead: the start half at the step's start ($t$), the end
half at its end ($t - \Delta t$), still in the step's met window. `endpoint` keeps
0022's mid-time weight.

**The analytic-plume harness drops `trapezoid`'s release half** (*derived*,
consistent with the independent review's measurement). In
`tests/test_plume_footprint.py`'s geometry, and in the scorer's `plume` mode with
`--receptor-frac 1`, the release point sits on the grid's east edge, where the
gridder's floor puts it outside the grid. Every particle's first start half is
therefore dropped, so the plume's all-layer total is low by exactly Δt/2 per
particle over its lifetime, and `trapezoid` ≡ `endpoint` there. Real
configurations always have the receptor inside the grid, so this is a property
of that harness, not of the rule; it is one more reason the plume cannot rank
the rules.

**Residence counting fixes** (external review of PR #66, 26 Nov 2026; asked for
by Matt the same day). Both concern where deposits land, not the rule itself.

1. *The ground was outside the grid.* The gridder binned heights with
   `torch.bucketize(z, edges, right=False) - 1`, i.e. layers $(\text{lo},
   \text{hi}]$, so a deposit at exactly $z = 0$ got index −1 and was dropped
   (its own comment described half-open layers). A unit-weight particle moving
   from 0 to 1 m in 60 s was counted 30 s by `trapezoid` and, ending on the
   ground, 0 s by `endpoint`, instead of 60 s (*measured*, the review's example,
   now `tests/test_deposition_position.py::test_a_deposit_on_the_ground_is_counted`;
   it fails on the old binning). The bug is on `main` too. **Decision:** layers
   are $[\text{lo}, \text{hi})$ (`right=True`), the convention of the horizontal
   cells (`floor`) and the time-ago bins, so one rule covers all axes and needs no
   special case. Consequences: the ground is in the first layer; a height exactly
   on an interior edge moves to the layer above; a height exactly at the top edge
   is now outside the grid, as a longitude exactly at `lon_max` already was.
   *Rejected:* keeping $(\text{lo}, \text{hi}]$ and including $z = 0$
   explicitly. It changes nothing for $z > 0$, but leaves the vertical axis with
   a different convention from the other four and a special case at the ground,
   for no physical gain: an output edge other than the ground has no physical
   meaning, and which side a height exactly on it falls is a convention either
   way. *What moves* (*measured* on the CPU smoke run, `endpoint` and
   `trapezoid`, by a read-only hook on every deposit,
   `$GLIDE_AGENT_OUTPUTS/agents/A2/fixes/edge_counts_*.json`): no deposit at
   exactly 0, 40, 1,000 or 5,000 m among 28.8 million (`endpoint`) and 57.6
   million (`trapezoid`) particle deposits, and the `endpoint` smoke run is
   bit-identical to `main` (footprint, endpoint particles, trajectory
   diagnostics; `bitcmp_smoke_main_vs_endpoint_a630b27.txt`). In transported
   particles a height exactly on an edge needs exact float cancellation
   (*argued*). The one deterministic case is `trapezoid`'s first
   start half at the release height: a release at 0 m is now counted in the
   first layer, and a release exactly on an interior edge (the JFJ case,
   `scripts/validation/zs-repair/case2-jfj.yaml`, 1,000 m with an edge at
   1,000 m) puts that half step in the layer above instead of below (*derived*;
   neither is the 0–40 m layer).
2. *Half steps were binned by one age.* Each half stands for an interval of
   ages but was counted in the bin of one age, so a half that straddled a
   time-ago edge put residence in the wrong bin. The first review gave ages
   80–140 s against an edge at 100 s: 30/30 instead of 20/40, and bins narrower
   than a step were worse. The reviewer then refined it: the particles of a
   batch share one step clock, but each counts its age from its own release
   window's end, so a window ending off the step grid makes steps straddle bin
   edges even when the bin width is a multiple of $\Delta t$ (ages 80–140 s
   against 120 s bins: 60/0 instead of 40/20). Planned column and satellite
   releases have observation times not aligned to $\Delta t$, so arbitrary
   windows must work. **Decision:** split each half step by its actual overlap
   with the age bins, each piece keeping its half's position and density
   (`lpdm.main._age_bin_pieces`). A half spans at most
   ⌈(Δt/2) / width⌉ + 1 bins (two for any bin at least half a step wide); the
   count depends only on the step length and the configured width, so shapes are
   static, there is no host sync, and the captured step is untouched (the
   deposit is outside it). Pieces that do not overlap add an exact 0, so a run
   whose steps never straddle an edge is unchanged (*derived*). One
   time-integrated bin is one piece, unrestricted. No configuration check is
   needed. *Measured* (`tests/test_deposition_position.py`,
   `tests/test_tracer_benchmark.py::test_misaligned_release_residence_and_one_bin_emission_are_exact`):
   the reviewer's three cases split exactly; a stationary particle at age
   offsets 0, 20 and 45 s from the step grid fills every bin with its exact
   seconds (widths 25, 60, 180 s, 60 s steps); end to end on both step paths,
   two releases in one batch, one ending 20 s off the step grid (so the runtime
   takes a 20 s first step), give the exact seconds per bin, including the partly
   covered bins at each end of the lifetime, and the exact predicted enhancement
   of an emission active in one partly covered bin. Every one of these fails on
   the code before this fix except the aligned cases. *Rejected:* validating that
   bin widths, run lengths and release windows are multiples of $\Delta t$ (it
   was implemented first, then dropped at the coordinator's request): it would
   refuse the misaligned windows that column and satellite releases need, and
   the split costs only one more scatter per half.
   **`endpoint` has the same issue and is not changed** (Matt's instruction): it
   counts a whole step in the bin of its starting age. Where windows end on the
   step grid and the width is a multiple of $\Delta t$ (every shipped
   configuration, *measured*: `$GLIDE_AGENT_OUTPUTS/agents/A2/fixes/shipped_configs.txt`)
   it is exact per bin; otherwise it moves up to a step of residence into the
   neighbouring bin, which cancels in fully covered bins but not at the ends of
   a release's lifetime (up to 4.4 s of a 120 s bin in the test above,
   *measured*), and mistimes an emission active in only some bins by up to a
   step. A related property (*derived*): a particle is counted from the step
   boundary at or below its release time, so its counted ages are shifted older
   by up to one step. With a window starting off the step grid, splitting would
   send up to that much of a particle's oldest residence past the end of the age
   axis (*measured* by the fresh-context review: 3598.1 of 3600 s for the test's
   second release); so for `trapezoid` the axis adds the largest offset of a
   release start above its batch's step grid (`_footprint_time_bin_count`), one
   more bin in that test, and each release keeps all 3600 s (*measured*, same
   test). `endpoint`'s axis is unchanged.

**Consequences.** Changing the default changes every surface-inlet footprint
near the receptor (smoke: +11.6% in the receptor's cell, +1.25% on the 0–40 m
total) and so needs the golden fixtures regenerated, which is planned once the
defaults settle. Two scatters and two density evaluations per step instead of
one. Cost on the reference config (GPU, shared node, *measured*, single runs per
seed): wall time 20:51 / 21:47 / 22:18 for `trapezoid` against 21:54 / 25:31 /
20:36 for `endpoint`, so no cost is resolved at this noise (±2–5 min). On the smoke run, the float32 grid's largest
cells (40–1000 m) differ from a float64 tally by up to 0.2% with `trapezoid`
(0.02% with `endpoint`); the same run with a float64 grid matches the scorer to
2e-9 on the total (`scores/check_s42_f64grid.txt`).

**Float32 accumulation** (outputs in `$GLIDE_AGENT_OUTPUTS/agents/A2/counting/f64/`;
reference config, GPU, seed 42; metric: per release and layer, the cell with the
largest relative difference among cells above 25% of the layer's maximum,
outside the receptor's 3 × 3, as in the PR #66 review; `compare_footprint_cells.py`).
The float32 grid loses precision as many small contributions are added to large
cell totals. With the prototype `output_grid.accumulation_dtype: float64` (sum in
float64, round once at write; default float32, unchanged):
- *Measured:* `endpoint`'s own float32 error against a float64 grid (same
  trajectories: endpoint particles identical): worst cell p90 0.11–0.14% and max
  0.29% in 40–1000 m, ≤ 0.07% in 0–40 m, ≤ 0.02% in 1–5 km; layer totals
  ≤ 0.07%.
- *Measured:* `trapezoid`'s own float32 error is about five times larger in
  40–1000 m (p90 0.51–0.76%, max 1.21%), because it adds twice as many,
  half-sized contributions (*argued*).
- *Measured:* with float64 grids, `trapezoid` against `endpoint` drops from
  p90 0.53–0.71% / max 1.33% to p90 ≤ 0.0054% / max 0.026% in 40–1000 m and to
  ≤ 0.008% in 0–40 m (outside the receptor's 3 × 3). What remains in 1–5 km
  (p90 0.14–0.16%, max 1.0%, almost all negative) is not rounding: it is the
  half step `trapezoid` does not count at each particle's final position. The
  predicted map, −Δt/2 × weight / ρ at the final positions in
  `endpoint_particles.parquet` (ρ from a scale height), matches the measured
  difference with correlation 0.9986 and slope 0.96 (*measured, isolated*;
  `f64/final_half_z2.txt`).
- *Measured, cost:* device peak 1.51 GiB against 1.42 GiB (one batch's grid in
  float64), host peak RSS unchanged (53.9 GB), wall time 20:52 and 20:51 against
  21:54 and 20:51 for float32: no time cost resolved.
So `trapezoid` should go with a float64 grid; the review recommends the two
together. The default stays float32 here.

**Proposed with it:** `trapezoid` together with `output_grid.accumulation_dtype:
float64` (prototype here, off by default), decided together with the fixture
regeneration.
Docs: [docs/physics.md](../../docs/physics.md) §7 and §8.
