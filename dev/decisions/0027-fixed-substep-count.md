# 0027 — Hanna turbulence: one fixed sub-step count for every particle, coefficients re-evaluated every sub-step

**Context.** The Hanna scheme sub-steps the turbulence inside each 60 s outer
step. Until this change each particle chose its own count,
$k_p = \lceil \Delta t / (c\,T_{Lw,p}) \rceil$ with $c = 0.5$, capped at
`max_substeps` (6 in every shipped config), from $T_{Lw}$ at its starting
height, and held $\sigma$, $T_L$, $\partial\sigma_w^2/\partial z$ and the
density drift at their starting-height values across the sub-steps. B7 (PR #67)
found that a horizontally uniform neutral boundary layer then holds 1.67 times
its well-mixed share of particles in its lowest 5% at $\Delta t = 60$ s (stable
1.78), and that more sub-steps made it worse. The WM investigation
(`outputs/agents/WM/`, branch `agent/WM-near-surface`) isolated two causes, each
by switching it alone in a prototype that reproduced the production scheme bit
for bit: the frozen coefficients (mostly $T_L$), and a sub-step length that
depends on the particle's height. It affects every GLIDE footprint made so far,
including the golden fixtures. Matt approved the fix before the physics freeze
on 25 Nov 2026 (coordinator's note `outputs/agents/WM/SUBSTEPPING-SUMMARY.md`
§9 and §12, and the review `SUBSTEPPING-REVIEW.md`).

**Decision.**
- Every particle takes the same number of equal sub-steps,
  `turbulence.n_substeps`, in every outer step. The count is a constant of the
  run: never chosen per particle, per step or per batch. It is the compiled
  loop's trip count, so the CUDA-graph step keeps static shapes and no host
  synchronisation (decision 0004), and with every particle active the dynamic
  CPU path and the static path are bit-identical
  (`test_static_and_dynamic_steps_bit_identical_when_all_active`).
- At the start of every sub-step after the first, $\sigma_{u,v,w}$,
  $T_{Lu,v,w}$, $\partial\sigma_w^2/\partial z$ and the density drift are
  re-evaluated at the particle's current height. The gradient's two
  finite-difference probes evaluate $\sigma_w$ only, bit-identical to taking it
  from the full profile.
- **Default $n = 16$**: a 3.75 s sub-step at the shipped 60 s outer step.
  Chosen from the accuracy study (physics.md §6): with the boundary-layer top a
  reflecting lid inside every sub-step, 5 seeds of 20,000 particles, the
  lowest 5% and the 0–40 m layer hold 1.035 / 1.035 (stable), 1.006 / 1.008
  (neutral) and 0.994 / 0.987 (convective) times their well-mixed share, the
  only count tested that is within the ±0.05 target in all three regimes;
  $n = 8$ leaves +0.074 ± 0.005 (stable lowest 5%) and −0.060 ± 0.006
  (convective 0–40 m), $n = 4$ +0.156 and −0.185. Fine-step references
  (1.875 s and 0.9375 s) agree with 1 to within 0.025. On the reference GPU benchmark the step costs 11.7 ms
  against 4.5 ms before, about 11% of the run's wall time (5.2 ms at $n = 4$,
  9.7 ms at $n = 8$).
- `turbulence.substep_c` and `turbulence.max_substeps` are removed; a config
  that sets either is refused with a message naming the replacement. The
  shipped and validation configs no longer set a count and take the default.
- Every active particle-sub-step is counted against
  $\Delta t_{\mathrm{sub}} > 0.5\,T_{Lw}$ and $\Delta t_{\mathrm{sub}} > T_{Lw}$,
  boundary layer and free troposphere separately, with the denominators, inside
  the step and without host synchronisation; the runtime reads the counts once
  per met window into `run_metadata.json` `runtime.substep_exceedance`. This
  replaces task A1's count of particle-steps reaching the old cap
  (`runtime.substep_saturation`, removed). On the reference run at $n = 16$: no boundary-layer sub-step
  and 0.00005% of free-troposphere sub-steps exceed $0.5\,T_{Lw}$ (at $n = 8$,
  0.0039%; A1's cap count had been 2.6% of particle-steps).
- The basal-clamp gradient is left as it was: below $z_{\mathrm{ubl}}$ the
  stencil returns the gradient just above the clamp. Zeroing it there
  (`basal_zero_gradient`, kept as a constructor switch for comparisons) changed
  the lowest 5% and 0–40 m occupancy by at most 0.003 in every regime (same
  seeds), within particle noise.

**Measured, not decided here.** (1) A per-particle count that only *adds*
sub-steps near the ground, $k_p = \max(n, \lceil 60/(0.125\,T_{Lw,p}) \rceil)$,
still moves occupancy towards the ground: neutral lowest 5% +0.052 ± 0.006
($n = 4$) and +0.017 ± 0.007 ($n = 8$) against the fixed count, above the fixed
$n = 16$ it refines to. So adaptivity itself biases, as argued for A3 (decision
0011). (2) With an open top into a quiescent free troposphere, too many
particles leave a convective boundary layer at practical sub-steps (25% after
2–6 h at $n = 16$, 6% at a 1.875 s step, none at 0.9375 s); widening the
drift's finite-difference step from 1 m to 10 m alone removes it, which locates
it at the $\sigma_w$ jump at $h$. That is a boundary-layer-top question for a
follow-up, not part of this change. (3) On the reference config the 0–40 m
footprint moves by −28%, −16% and +15% (median over releases) at $n$ = 4, 8 and
16 against the old code, with the share of time above the boundary layer
falling as $n$ grows: real footprints are not yet converged in $n$, and the
boundary-layer top is the likely reason (argued from the open-top study, not
isolated on real meteorology). The default of 16 is therefore the best of the
counts tested, not a converged value; the boundary-layer top should be settled
before the freeze and the fixtures regenerated after it.

**Rejected alternatives.**
- *The largest per-particle count over the batch* (the WM prototype, commit
  7c756ae). It removed the bias as well, but a release's footprint then depends
  on which other releases share its batch, and on the reference config a single
  free-troposphere particle in an inversion would set the count for everyone
  almost every step (A1: 2.6% of particle-steps reached the old cap of 6).
- *Keeping the per-particle adaptive count with the coefficients re-evaluated*,
  or a FLEXPART-like count re-chosen every sub-step. Both left the neutral
  lowest 5% at 1.31-1.37 at $c = 0.5$ (WM `exp2`, `exp3`), falling only in
  proportion to $c$.
- *Reinterpreting `max_substeps` as the fixed count.* Every shipped config sets
  it to 6 and the code default was 50; reinterpreting it would silently change
  what those numbers mean. A new key, with the old keys refused by name, makes
  a config that set them choose again.
- *A configurable reporting threshold.* The exceedance threshold is a
  diagnostic, not physics; it is a scheme constant (0.5, the old target), and
  the run records it next to the counts.
- *Changing the free-troposphere integration now* (a $T_{Lw}$ floor above the
  boundary layer, or the exact joint OU velocity-displacement update of
  Gillespie 1996). Out of scope here; the exceedance counts are the measurement
  that decides it.

**Status.** Accepted, 25 Nov 2026 (task WM). The golden fixtures and the
benchmark baseline are not regenerated here; they are regenerated once, after
the residence-counting decision (A2), as planned. Docs:
[physics.md](../../docs/physics.md) §6 and §8,
[turbulence.md](../../docs/turbulence.md) §5 and §9.
