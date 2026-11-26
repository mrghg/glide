# 0029 — Trapezoid residence counting, scored against fine-time residence (proposed default)

**Status: proposed.** `trapezoid` is implemented as an option; the default stays
`endpoint` until Matt decides. This record proposes making `trapezoid` the
default, with the evidence below.

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
  Each half has its own time-ago bin (the start half the age at the step's
  start, the end half the age half a step later, as `endpoint` bins the whole
  step by its starting age; with steps aligned to the bins both fall in the
  step's bin) and, with dry-air weighting, its own density, at its own position
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
  with half-step shifts only where a step straddles a time-ago bin edge (none when
  steps align with bins) and density evaluated at the node's own time. So away
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

**Consequences.** Changing the default changes every surface-inlet footprint
near the receptor (smoke: +11.6% in the receptor's cell, +1.25% on the 0–40 m
total) and so needs the golden fixtures regenerated, which is planned once the
defaults settle. Two scatters and two density evaluations per step instead of
one. Cost on the reference config (GPU, shared node, *measured*, single runs per
seed): wall time 20:51 / 21:47 / 22:18 for `trapezoid` against 21:54 / 25:31 /
20:36 for `endpoint`, so no cost is resolved at this noise (±2–5 min). The float32 footprint grid accumulates twice as many,
half-sized contributions; the largest smoke cells (40–1000 m) then differ from a
float64 tally by up to 0.2% (0.02% with `endpoint`; 0–40 m total 7e-5). *Measured,
isolated:* the same run with a float64 grid matches the scorer to 2e-9 on the
total (`scores/check_s42_f64grid.txt`). A float64 (or compensated) grid is a
separate question, proposed as a follow-up.
Docs: [docs/physics.md](../../docs/physics.md) §7 and §8.
