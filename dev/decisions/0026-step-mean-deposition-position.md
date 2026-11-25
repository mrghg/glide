# 0026 — Deposition at the step-mean position, not the RK2 advection midpoint

**Context.** Task A2 ([PR-F4], QUEUE.md) specified an optional deposition
position `output_grid.deposition.position: endpoint | midpoint`, with `midpoint`
"the RK2 midpoint position already computed for advection, which is consistent
with the second-order scheme". PR #66 implemented it as specified: the RK2
predictor $\mathbf{x} - \tfrac{\Delta t}{2}\mathbf{u}(\mathbf{x})$, reflected
below ground. Its review found that the predictor leaves out the step's
turbulent and convective displacement, so in the vertical it is the height at
the **start** of the step (a left-rectangle rule, not a midpoint). Every
particle's first step after a surface release is then counted at the release
height. On the smoke run (Mace Head, 10 m inlet, Δt = 60 s, 1 day) it added
+4.99% (CPU) / +4.88% (CUDA) to the 0–40 m total, +45.9 of the +49.7 units in
the receptor's own cell, about one full step of surface residence. A prototype
in the same PR showed that depositing at the mean of each step's start and end
positions was about ten times more accurate near the receptor on the analytic
plume. Matt's decision of 25 Nov 2026: revise #66 before merging.

**Decision.**
- The non-default option is `step_mean`: each step's residence (and the
  dry-air density it is divided by, decision 0022) is taken at
  $\tfrac12(\mathbf{x}_\text{start} + \mathbf{x}_\text{end})$, the mean of the
  particle's position at the start of the outer step and at its end, after
  advection, turbulence, surface reflection and convection. `midpoint` is
  removed rather than kept as a known-biased alternative.
- `endpoint` stays the default and bit-identical (golden CPU fixture).
- Ground and top: both ends are particle positions as the step leaves them, so
  the mean height is $\ge 0$ without further handling. A particle whose step
  ends above the met domain is killed after the step as before; its mean
  position is counted if it lies inside the output grid.
- Longitude is averaged along the shorter arc: $\lambda_\text{start} +
  \tfrac12\,\mathrm{wrap}(\lambda_\text{end} - \lambda_\text{start})$ with the
  difference in $[-180°, 180°)$. The runtime does not wrap particle longitudes
  today (particles leaving the met domain's longitude box are killed), so this
  changes nothing for current runs; it keeps the option correct if periodic
  (global) domains are added.
- The runtime copies the start positions (N × 3) before the step, outside the
  captured graph. The Hanna step returns the same six outputs for both
  positions, so the static (CUDA) path's graph, shapes and capture are
  unchanged (decision 0004), and no scheme needs to know the option exists.

**Measured** (outputs in `$GLIDE_AGENT_OUTPUTS/agents/A2/step-mean/`).
Analytic plume, mean relative error of the crosswind-integrated surface
footprint (50,000 particles, seeds 9021 / 1234 / 7): Δt = 5 s endpoint
0.0099 / 0.0133 / 0.0117, step_mean 0.0056 / 0.0081 / 0.0076; 4Δt endpoint
0.0262 / 0.0271 / 0.0233, step_mean 0.0113 / 0.0104 / 0.0069. Within two
Lagrangian timescales of the receptor: endpoint 0.015–0.021 (Δt) and
0.082–0.091 (4Δt); step_mean 0.0014–0.011 and 0.0024–0.011. Smoke run
(step_mean against endpoint, same seed): 0–40 m total −2.19% (CPU) / −2.33%
(CUDA), all layers −5.4e-5; receptor cell +11.7% / +12.1% (+26.3 / +26.4, the
largest cell change); every ring of cells around it −3.6% (adjacent) to −8.0%
(beyond 10 cells). The RK2 predictor (the spec's `midpoint`) gave +4.99% /
+4.88% on the 0–40 m total, +20.5% in the receptor cell and +0.1–1.3% elsewhere.

**Found after the decision, before merge: a surface-layer bias of its own.** The
far-field loss on the smoke run is not a quadrature correction. In a
horizontally uniform neutral Hanna boundary layer, equilibrated
(`wellmixed_*` in the outputs folder), `step_mean` and `endpoint` count the same
fraction of particle-steps below 40 m to 0.3–1% at u* = 0.4 m/s, but at Δt =
60 s and u* = 1.0 m/s, where the rms vertical step of particles ending below
40 m is 52–63 m, `step_mean` counts 18% fewer (0.0526 vs 0.0640 for a 1,000 m
BL; 0.134 vs 0.165 for 300 m). The mean of two heights a layer-depth apart is
pulled away from the ground; the endpoint of a stationary chain is not. The
Δt = 5 s reference (0.0439, 0.148) shows `endpoint` itself 46% / 11% high at
60 s (the near-surface well-mixed excess under separate investigation), so
which position is closer to the truth there depends on that fix. The analytic
plume (T_L = 100 s, Δt ≤ 20 s) does not reach this regime. **Open for Matt:**
whether `step_mean` should stay as specified here, or be restricted / replaced
(e.g. a trapezoid split, which keeps each end's stationary distribution) once
the well-mixed excess is understood.

**Rationale.** The step mean is the trapezoid-in-position rule: for the mean
motion it is the midpoint in time (with uniform wind it moves each deposit
exactly $U\Delta t/2$ downwind of the endpoint's), and it includes the
turbulent displacement that dominates vertical motion near the surface, where
the footprint is most sensitive. Both positions are already in hand on both
step paths, so it costs one (N, 3) copy per step and no change to the
captured step. The RK2 predictor gets the along-wind half step right but not
the height, which is what matters for a surface footprint.

**Rejected alternatives.**
- *Keep `midpoint` as specified, alongside `step_mean`:* a documented option
  known to bias the near-receptor footprint by +5% at Δt = 60 s invites use in
  the A5 sweep and A3 for no benefit.
- *Trapezoid split (half the weight at each end):* also includes all of the
  motion, but deposits twice per step, doubling the gridder's scatter work,
  and credits the start position, which the previous step already credited as
  its end; not tested beyond the prototype.
- *Return the predictor (or the turbulent displacement) from the Hanna step:*
  an extra output of the captured graph, scheme-specific, and unnecessary when
  the start position is already in the runtime.

**Consequences.** A3 (multi-rate stepping) and A5 (deposition-position sweep)
use `step_mean` where the spec said `midpoint`; QUEUE.md is updated. Still a
delta deposit at one position per step, so a particle crossing several cells
in one step credits one of them; kernel deposition (decision 0012) remains
after v1.

**Update (decision 0029).** Scored against fine-time residence on the same
trajectories, after decisions 0027 and 0028, `step_mean` undercounts the 0–40 m
layer outside the receptor's cell by 3–5% on the smoke run and by up to 15% in
an equilibrated column with $u_*$ = 1 m s⁻¹; the surface-layer bias above is
real, not an artefact of comparing with a biased `endpoint`. Decision 0029 adds
`trapezoid` and proposes it as the default.
