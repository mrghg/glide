# 0030 — Hanna turbulence: an analytic drift gradient at a stable boundary-layer top and the σ_w floor (amends 0028 item 1)

Claims are labelled *measured* (with the file, under `outputs/agents/` in the
main checkout unless a test is named), *derived* (follows from the equations)
or *argued*. The investigation is `WM/sigma_floor/SUMMARY.md` (task
WM-sigma-floor, branch `agent/WM-sigma-floor` ce07cb6) and its independent
review `WM/sigma_floor/review/`; this record's own runs are under `SF/`.

**Context.** In the stable regime the Hanna profile
$\sigma_H = 1.3\,u_\ast(1 - z/h)$ falls to zero at $h$, and GLIDE floors it,
$\sigma_w = \max(\sigma_H, \sigma_{\min})$ with
$\sigma_{\min}$ = `SIGMA_MIN_M_S` = $10^{-3}$ m s⁻¹. The floor binds in a
*floor layer* of depth $d = \sigma_{\min} h/(1.3\,u_\ast)$ just below $h$
(*derived*). The drift's $\partial\sigma_w^2/\partial z$ was a ±δ (1 m) finite
difference of the floored profile, kept on the particle's side of $h$ by
decision 0028 item 1. That had two defects there.

1. **The stencil straddles the floor's kink.** A particle within δ of the
   floor layer, or in it, differences across the corner of
   $\max(\sigma_H, \sigma_{\min})$: it gets a drift inside the floor layer,
   where $\sigma_w$ is flat, and a wrong one just below it (*measured*:
   at $h - 1$ m in the $u_\ast$ = 0.25 m s⁻¹ column, ½∂σ_w²/∂z = −4.1×10⁻⁷ m s⁻²
   where the exact value is 0; `WM/sigma_floor/logs/drift_profile_floor0.001.log`).
2. **Decision 0028's one-sided stencil at $h$ over-states the gradient when
   $d < \delta$.** Within δ of $h$ a boundary-layer particle's stencil is cut
   at $h$, so it returns the gradient at the stencil's midpoint, up to δ/2 below
   the particle. Where $\sigma_H \to 0$ linearly, $\partial\sigma_H^2/\partial z
   \propto (h - z)$ and that is an O(1) relative error (*derived*). At a
   $10^{-4}$ floor ($d$ = 0.07 m) in the thin-floor column below the stencil
   gives 5.5 times the exact gradient 0.1 m below $h$
   (*measured*, `test_stencil_defects_at_a_stable_top`).

**How common.** $d < 1$ m whenever $u_\ast > 7.7\times10^{-4}\,h$ at the
$10^{-3}$ floor (*derived*). In ERA5 over Europe, 44% of stable columns (by
area, every third hour) have $d < 1$ m: 0.439 in 22 Dec 2018 – 15 Jan 2019 and
0.447 in 21 Jun – 15 Jul 2019; the median $d$ is 1.1 m (*measured*,
`WM/sigma_floor/review/logs/era5_eur_{dec,jul}.log`). Stable columns are
23–27% of all columns there.

**How bad.** The test is transfer symmetry across $h$. In a steady column
the chain is reversible about the well-mixed state, so with $M[a,b]$ the time
a particle started uniformly in 40 m row $a$ spends in row $b$, the
downward/upward ratio $\sum_b M[\text{above}, b] / \sum_b M[b, \text{above}]$
over the four boundary-layer rows below $h$ must be 1 (*derived*;
`transfer_ratio` in `tests/test_sigma_floor_gradient.py`).

| column | gradient | 16 sub-steps | 64 sub-steps | source (*measured*) |
| --- | --- | --- | --- | --- |
| thin floor: h 400 m, u* 0.45, H −60 W m⁻², d 0.68 m | before (stencil) | 3.37 ± 0.19, 3.20 ± 0.17 | 3.05 ± 0.16 | `WM/sigma_floor/review/logs/rowmat_table.log` (forward, 40,000 per row, 2 seeds) |
| | this decision | 1.11 ± 0.04, 1.10 ± 0.04 | 1.03 ± 0.03, 0.97 ± 0.03 | same |
| | before (stencil) | GATE_FD_N16 | GATE_FD_N64 | `SF/gate/` (backward, 10,000 per row, GATE_NSEEDS) |
| | this decision | GATE_PROD_N16 | GATE_PROD_N64 | same |
| thick floor: h 400 m, u* 0.25, H −20 W m⁻², d 1.23 m | before | 1.215 ± 0.030 | 1.189 ± 0.029 | `WM/sigma_floor/logs/table.md` (2 seeds pooled) |
| | this decision | 1.025 ± 0.024 | 1.004 ± 0.023 | same |
| thick-floor column at a 10⁻⁴ floor (d 0.12 m) | before | 52 ± 25 | — | same |
| | this decision | 1.12 ± 0.05 | 0.98 ± 0.04 | same |

Before the fix the asymmetry does not converge with the sub-step count; with it
the residual at 16 sub-steps is first order in the sub-step (it halves or
better from 16 to 64), decision 0027's ordinary error (*measured*, above; the
attribution is *argued*). Below $h$ (row $h$−40…$h$ with the three rows under
it) every variant gives 1.00–1.03, so the defect is the exchange across $h$
(*measured*, same files). The velocity carry, the 4σ clip and a cap on
$T_{Lw}$ change the pooled ratio by at most 0.04 once the gradient is right
(*measured*, `WM/sigma_floor/SUMMARY.md` §1b).

**Decision.** Decision 0028 item 1 is amended. A boundary-layer particle
($z \le h$) whose ±δ stencil

- is cut at $h$ ($z + \delta > h$), or
- touches the floor (σ_w at either probe or at the particle is at the floor)

takes the analytic derivative of the floored boundary-layer profile,
`in_bl_dsigw2_dz`: $\partial\sigma_H^2/\partial z$ from the regime formulas
(stable, neutral and unstable), and zero where $\sigma_H \le \sigma_{\min}$.
Every other particle, including every free-troposphere particle, keeps the
stencil, bit for bit. The selection is a `torch.where` on data already
computed (no new parameter, static shapes, no host synchronisation), so the
static/graph and dynamic paths stay bit-identical (decision 0004). The floor
value and form are unchanged.

- The stencil-only behaviour remains as a private attribute,
  `HannaScheme._fd_drift_gradient_only`, for A/B tests. No configuration
  sets it; with it set, a run is bit-identical to main d19755d (*measured*,
  `SF/bitid/compare.log`).
- **With the opt-in surface-layer override** (`surface_layer_override=True`,
  off by default) the stencil is used everywhere, including at $h$ and the
  floor. The analytic form does not include the override's profile below
  $0.1h$, and the override already has an undifferentiated seam
  (docs/turbulence.md §8). The defects above therefore remain in runs with the
  override (`test_surface_layer_override_keeps_the_stencil`).

**Checks.** `tests/test_sigma_floor_gradient.py`:
- `in_bl_dsigw2_dz` matches 1 mm one-sided differences of the floored profile
  to 10⁻³ of its maximum, in all three regimes, over grids of $h$, $u_\ast$,
  $w_\ast$ and latitude, at floors 10⁻⁴, 10⁻³ and 10⁻² (the review's
  `grad_check.py` check C, in float64);
- in eight columns (three regimes; thin and thick floor layers; a neutral
  column whose σ_w reaches the floor mid-layer) the production gradient
  equals the float64 analytic one wherever it departs from the stencil,
  departs only within δ of $h$ or of the floor, and every coefficient is
  bit-identical to the stencil's elsewhere;
- static and dynamic steps are bit-identical with half the particles within
  5 m of a stable $h$;
- **the transfer-symmetry gate**: the thin-floor column at the default 16
  sub-steps, 10,000 particles per row, $|r - 1| <$ GATE_TOL. Calibration
  in `SF/gate/` (GATE_CALIB). It fails on main.

The two well-mixed gates pass unchanged (`tests/test_near_surface_well_mixed.py`,
`tests/test_bl_top_well_mixed.py`; GATE_EXISTING).

**Production effect** (*measured*, `SF/smoke/`; CPU smoke,
`configs/smoke_mhd_single_release.yaml`, Mace Head, one 1-day release in
January, 20,000 particles): SMOKE_EFFECT A winter release at Mace Head barely
sees a stable boundary-layer top, so this is no measure of the fix's effect
where it matters: a night-time tall-tower or elevated receptor above a shallow
stable $h$ (*argued*).

## The floor is an exchange parameter

Once the gradient is right, $\sigma_{\min}$ is what sets the turbulent exchange
across a stable $h$ (the Hanna profile itself has none).

**Exchange velocity** (*derived*). In a well-mixed state of concentration $c$,
the one-way flux through a level is $c\,\overline{w^+} = c\,\sigma_w/\sqrt{2\pi}$.
Just below a stable $h$, $\sigma_w = \sigma_{\min}$, and TPM's rule (decision
0028) transmits from the free troposphere with probability
$\sigma_{\min}/\sigma_{FT}$, so the flux through $h$ is the same both ways,
$v_h = \sigma_{\min}/\sqrt{2\pi}$: 0.4 mm s⁻¹ at $10^{-3}$.

**Resistances in series** (*derived* here, following the review's argument; an order-of-magnitude estimate, since the exchange measured below is a 2 h residence, not a steady flux). Exchange between the
free troposphere and the well-mixed interior of the boundary layer passes
three resistances:
- the interface, $r_h = 1/v_h = \sqrt{2\pi}/\sigma_{\min}$: 2,500 s m⁻¹ at
  $10^{-3}$;
- the upper boundary layer, where $K = \sigma_w^2 T_{Lw} \approx 0.13\,u_\ast h\,(1 - \zeta)$
  falls linearly to $h$:
  $r_{BL} = \int dz/K \approx \ln(1.3u_\ast/\sigma_{\min})/(0.13\,u_\ast)$, cut
  off by the floor layer; 110 s m⁻¹ at $u_\ast$ = 0.45 m s⁻¹, 180 s m⁻¹ at 0.25
  (logarithmic in $\sigma_{\min}$);
- the free troposphere, $r_{FT} \approx \ell/K_{FT}$ over the depth $\ell$ the
  exchange is measured over: 400 s m⁻¹ for 40 m at the closure's
  0.1 m² s⁻¹ floor.

The floor layer itself ($K = 0.1\,h\,\sigma_{\min}$, depth $d$) adds
$d/K = 1/(0.13\,u_\ast)$, which does not depend on $\sigma_{\min}$: tens of
s m⁻¹. With $r_h$ dominant the exchange is proportional to $\sigma_{\min}$;
it saturates once $r_h$ falls to the others, at $\sigma_{\min}$ of order
$10^{-2}$ (*derived*).

**Measured dependence** (`WM/sigma_floor/SUMMARY.md` §4; thick-floor column,
this decision's gradient, 16 sub-steps, 2 seeds, exchange from $h$…$h$+40 m
into the 160 m below over 2 h):

| σ_min (m s⁻¹) | floor layer | exchange relative to 10⁻³ |
| --- | --- | --- |
| 10⁻⁴ | 0.12 m | 0.11 |
| 10⁻³ | 1.23 m | 1 |
| 10⁻² | 12.3 m | 6.0 |

Proportional below 10⁻³, as derived; the six-fold rise per decade above it is
faster than the series estimate suggests (it would give about 4), which the
derivation does not explain (the floor layer's own $T_{Lw}$, 0.1h/σ_min, also
changes; capping it at 1,000 s cut the exchange at 10⁻³ by 20%, *measured*,
same file).

**FLEXPART** (sources in `WM/sigma_floor/refs/`, lines verified by the review):
- default path (`hanna1`, `CTL < 0`): stable
  `sigw = max(1.3*ust*(1-zeta), 1.e-6)` with the analytic
  `dsigw2dz = 3.38*ust*ust*(zeta-1.)/h` (v10.4 `hanna1.f90`), i.e. a 10⁻⁶ floor
  and the gradient this decision adopts;
- at $h$ it reflects, inside the boundary-layer loop:
  `else if (delz.gt.(h-zt)) then ! reflection at h` (v10.4 `advance.f90`
  lines 485–487);
- above $h$ there is no vertical turbulence (`wp = 0.`, v10.4 `advance.f90`
  650–656; v10.4 adds a mesoscale term, v11 turns it off by default).

So in FLEXPART no turbulent exchange crosses $h$, and its floor sets nothing;
GLIDE's interface (decision 0028) makes the floor matter.

**Options** (not a choice; *argued* unless marked):
1. **Keep 10⁻³.** No change beyond this fix. The exchange
   (0.4 mm s⁻¹; $K \approx 0.04$ m² s⁻¹ in the floor layer at $h$ = 400 m) is set
   by a number chosen for numerical safety; it scales with neither $u_\ast$ nor
   the inversion.
2. **10⁻⁴ or FLEXPART's 10⁻⁶.** Approaches Hanna's no-exchange limit and
   FLEXPART's behaviour; exchange falls about tenfold per decade (*measured*
   at 10⁻⁴). Costs: a larger first-order residual at 16 sub-steps (1.12 at
   10⁻⁴, *measured*), $T_{Lw}$ up to $4\times10^5$ s in the floor layer, and
   sub-millimetre motions near $h$ that float32 resolves coarsely when $h$ is a
   few km.
3. **10⁻²** (or FLEXPART `hanna`'s additive 1 cm s⁻¹). Six times the
   exchange; the stencil defect is diluted even without this fix
   (*measured*, 0.999 before the fix). FLEXPART pairs that σ with reflection
   at $h$, so it is no precedent for exchange at that rate.
4. **Matched to the free troposphere's diffusivity**, $\sigma_{\min} = K_{FT}(h)/(0.1h)$:
   $K$ continuous across $h$, so the closures on each side, not a constant,
   set the exchange; 2.5×10⁻³ m s⁻¹ at $h$ = 400 m with the 0.1 m² s⁻¹ closure
   floor. TPM's maximum-transmission assumption (decision 0028) is most
   natural when the two sides' diffusivities are comparable. Not run.
5. **A $u_\ast$-scaled floor**, $\sigma_{\min} = c\,u_\ast$: the floor layer
   becomes $c\,h/1.3$, a fixed fraction of $h$, and the exchange scales with the
   surface turbulence. Needs a value of $c$ and a reason for it. Not run.
6. **Reflection at a stable $h$**, as FLEXPART. Decision 0028 rejected
   reflection for well-mixed consistency; restricting it to stable columns
   would need a new decision.

**The floor stays at 10⁻³ in this decision.** Its value is Matt's decision,
recorded here as an open question.

**Rationale.** The analytic gradient is exact where the stencil is wrong and
is not used anywhere else, so it removes both defects at every floor tested
without changing a single trajectory that never comes within δ of $h$ or of the
floor. It adds no parameter. FLEXPART's default path uses the same analytic
gradient.

**Rejected alternatives** (*measured* in `WM/sigma_floor/SUMMARY.md` §2
unless marked).
- *Difference the unfloored branch, zero where floored* (`kink`).
  Bit-identical elsewhere and right when $d > \delta$, but it keeps defect 2:
  2.62 ± 0.16 at a 10⁻⁴ floor, 1.15 in the thin-floor column at 10⁻³
  (`WM/sigma_floor/review/logs/rowmat_table.log`).
- *The analytic gradient for every boundary-layer particle.* Equally right,
  but it changes every trajectory (rounding) and the convective gradient at
  the basal clamp; no accuracy gain away from $h$.
- *A smooth floor*, $\sqrt{\sigma_H^2 + \sigma_{\min}^2}$ or
  $\sigma_H + \sigma_{\min}$ (FLEXPART `hanna`). Removes the kink but changes
  σ_w everywhere and, with it, the exchange (−6% for the additive form): a
  floor-form choice, not a fix, so it goes with the open question above.
- *A smaller stencil* (δ = 0.01 m). Changes every gradient in the boundary
  layer; 1.04 at 10⁻³.
- *Leaving the cut stencil at $h$ and fixing only the kink.* That is `kink`.

**Status.** Proposed, 26 Nov 2026 (task SF, draft PR). Amends decision 0028
item 1. The floor value is open (Matt). Docs:
[physics.md §3.3, §6](../../docs/physics.md),
[turbulence.md §2, §6](../../docs/turbulence.md).
