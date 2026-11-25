# 0028 — Hanna turbulence: the boundary-layer top as an interface (Thomson, Physick & Maryon 1997)

**Context.** In the Hanna scheme $\sigma_w$ (and $\sigma_{u,v}$) jump at the
boundary-layer top $h$, from the boundary-layer profile to the
free-troposphere Richardson closure. In the study's columns (*measured*,
`_interface_sigmas`) $\sigma_w$ just below / just above $h$ is 0.73 / 0.055
m s⁻¹ (convective, a factor of 178 in variance), 0.24 / 0.055 (neutral) and
0.001 / 0.046 (stable, where the boundary-layer profile falls to its floor at
$h$). Decision 0027 (PR #69) left the boundary-layer
top as it was. An independent review of PR #69
(`outputs/agents/WM/review69/`), verified here with 3 seeds
(`outputs/agents/WM/interface/study/tables.md`), found that a column started
uniform in $[0, 2h]$, with an open top at $h$ and a lid at $2h$, does not stay
uniform at any sub-step count:

- the drift's $\partial\sigma_w^2/\partial z$, a ±1 m central difference of the
  assembled profile, straddles $h$ for particles within 1 m of it and
  differences the jump into a drift of order
  $(\sigma_{BL}^2 - \sigma_{FT}^2)/(2\,\delta)$;
- a particle that crosses $h$ keeps its $w'$, so above $h$ the
  $(1 + w'^2/\sigma_w^2)$ factor is inflated by up to $(\sigma_{BL}/\sigma_{FT})^2$;
- the result depends on the sub-step: in the convective column the 50 m above
  $h$ held 11.5, 9.0, 6.0, 1.8 and 0.44 times its uniform share at
  $n$ = 4, 8, 16, 32 and 64 (PR #69 head 0de5f94; trapped above $h$ at long
  sub-steps, pumped into the boundary layer at short ones); neutral 3.3, 1.5,
  0.48, 0.39, 0.42; stable 0.96–1.12. Main (20db902) was worse (convective
  9.9, neutral 4.6, stable 0.53).

The PR's fine-step reference for the open-top column (boundary-layer share
1.00 at 0.9375 s) was therefore itself biased, and its 10 m finite-difference
probe reversed the bias rather than removing it (TBD numbers). Matt chose, on
25 Nov 2026, a well-mixed-consistent crossing of $h$ (option a), not
FLEXPART-style reflection.

**Decision.** Three changes in `HannaScheme`, on both the dynamic and the
static/graph path (one shared sub-step loop; static shapes, no host
synchronisation, bit-identical between the paths with every particle active):

1. **One-sided drift stencil.** A particle's $\partial\sigma_w^2/\partial z$ is
   differenced on its own side of $h$: probes in $[z_{\mathrm{ubl}}, h]$ with
   the boundary-layer profile for $z \le h$, in $[h, \infty)$ with the
   free-troposphere closure for $z > h$, each evaluated on its own side even at
   $z = h$ (`_column_sigma_w(..., side_ft=)`). More than $\delta$ from $h$ it is
   the centred stencil, bit for bit.
2. **Crossing rule at $h$** (`_cross_interface`), inside every sub-step, after
   the displacement and before the ground reflection. With $A$ the side the
   particle starts the sub-step on, $B$ the other, $\sigma_A, \sigma_B$ the
   one-sided limits of $\sigma_w$ at $h$ (`_interface_sigmas`: the
   boundary-layer profile at $h$, the closure at $h$; once per outer step):
   - transmitted with probability $\alpha_{AB} = \min(1, \sigma_B/\sigma_A)$:
     $w' \to w'\,\sigma_B/\sigma_A$, $u', v'$ scaled by their own
     $\sigma_B/\sigma_A$, and the overshoot beyond $h$ scaled by
     $\sigma_B/\sigma_A$ ($z \to h + (z - h)\,\sigma_B/\sigma_A$: the rest of the
     sub-step flown at the new speed);
   - otherwise reflected: $z \to 2h - z$, $w' \to -w'$, $u', v'$ unchanged.
3. **$w'/\sigma$ carried across outer steps** (`_carry_velocities`). Each
   particle stores $\sigma_{u,v,w}$ at its end-of-step position with that
   step's meteorology (`state["sigma_last"]`); at the start of the next step
   $u', v', w'$ are scaled by $\sigma(\text{now})/\sigma_{\mathrm{last}}$. This
   handles $h$ (and the regime, $u_\ast$, $w_\ast$) changing between steps:
   time interpolation within and across met windows, horizontal advection into
   another column, mean vertical motion, convective redistribution.

In the closed-column harnesses (the near-surface CI gate, the study's
`closed`) the lid at $h$ replaces the interface, so the crossing rule is off
there (`_LidHanna` in `tests/test_near_surface_well_mixed.py`).

**What is implemented, against TPM 1997.** The paper itself could not be
retrieved (journal page refused automated access). The scheme as described
here is from STILT's implementation, which labels it "Thomson [1997]
reflection/transmission scheme" (`pardsp.f` of the STILT/HYSPLIT merged code,
Lin et al. 2003; read for this decision): transmission probability
$\rho_B\sigma_B/(\rho_A\sigma_A)$ (always transmitted if ≥ 1), $w'$ scaled by
$\sigma_B/\sigma_A$ on transmission, $w' \to -w'$ on reflection, and $w'/\sigma_w$
stored between outer steps "to have proper treatment of step changes in
turbulence statistics from one timestep to the next (Thomson et al. [1997])".
A human check against the paper is still owed. Departures, each deliberate:

| TPM / STILT | GLIDE | why |
| --- | --- | --- |
| piecewise-constant $\sigma$ in layers, no drift; every layer boundary an interface | smooth profiles with the Thomson drift on each side; only $h$ is an interface | the boundary-layer and free-troposphere profiles are smooth; only $h$ jumps |
| particle moved to the interface, the internal step ends there, the rest of the time taken by later internal steps (variable number) | the sub-step is completed ballistically: transmitted overshoot scaled by $\sigma_B/\sigma_A$, reflected overshoot mirrored | keeps the fixed sub-step count (decision 0027, static shapes); exactly consistent, below |
| factor $\rho_B/\rho_A$ | omitted | $\rho$ is continuous at $h$, so the factor is 1 |
| $w'$ only | $u', v'$ also scaled on transmission | the horizontal $\sigma$ jump too; consistent for independent Gaussian components, below |
| $w'/\sigma_w$ stored between calls | $u'/\sigma_u$, $v'/\sigma_v$, $w'/\sigma_w$ carried, $\sigma_{\mathrm{last}}$ at the end-of-step position | the same treatment for all three components |

**Why it keeps a well-mixed state well mixed** (*derived*). GLIDE's backward
model ($z_{n+1} = z_n - w'\Delta t$, drift negated, §4 of physics.md) is the
forward model written for $(z, \tilde w = -w')$: substituting $\tilde w$ turns
$a^b$ into the forward Thomson–Stohl drift and the displacement into
$+\tilde w\Delta t$, with the same noise. The crossing rule depends only on the
direction of the displacement and on $|w'|$, and commutes with $w' \to -w'$,
so it suffices to show the forward case.

Take $\sigma$ piecewise constant near $h$ ($\sigma_A$ below, $\sigma_B$ above,
Gaussian velocities) and the well-mixed density $f = c\,P_A(w)$ for $z < h$,
$c\,P_B(w)$ for $z > h$ ($\rho$ continuous). A sub-step is an exact OU update
at fixed $z$, which preserves $P$ at each height for any $\Delta t$, followed
by the displacement $z \to z + w\Delta t$ and the interface map. Look at the
density after the displacement and map at a point $(z', w')$ on side $B$ with
$0 < z' - h < w'\Delta t$, $w' > 0$ (moving away from $h$, within one
displacement of it); every other point is reached by free flight within one
side, which preserves $f$. Two kinds of particle arrive there:

- transmitted $A$-particles, from $(z_0, w)$ with $w = w'/r$,
  $z_0 = h + (z' - h)/r - w'\Delta t/r$, $r = \sigma_B/\sigma_A$; the map
  $(z_0, w) \to (z', w')$ has Jacobian $r^2$, and for Gaussians
  $P_A(w'/r) = r\,P_B(w')$, so they contribute
  $\alpha_{AB}\,c\,P_A(w'/r)/r^2 = (\alpha_{AB}/r)\,c\,P_B(w')$;
- reflected $B$-particles, from $(2h - z' + w'\Delta t, -w')$ with Jacobian 1:
  $(1 - \alpha_{BA})\,c\,P_B(w')$.

The sum is $c\,P_B(w')$ if and only if $\alpha_{AB}/r = \alpha_{BA}$, i.e.
$\alpha_{AB}\,\sigma_A = \alpha_{BA}\,\sigma_B$: the fluxes $\alpha\,\sigma\,c/\sqrt{2\pi}$
through $h$ balance. $\alpha_{AB} = \min(1, \sigma_B/\sigma_A)$,
$\alpha_{BA} = \min(1, \sigma_A/\sigma_B)$ satisfy it, with the fewest
reflections. The result holds for any sub-step length: with piecewise-constant
$\sigma$ the rule is exactly well-mixed-consistent; with smooth profiles on
each side the one-sided drift supplies Thomson's term for each side's profile,
and the remaining error is the ordinary $O(\Delta t_{\mathrm{sub}})$ error of
the drift (decision 0027). For $u'$ (and $v'$), independent of $w'$ on each
side, the transmitted particles carry $P_A(u)$ (the crossing selects on $w$
only) and the scaling maps it to $P_B(u)$; reflection leaves it unchanged.

The carry is the exact integral of the nonstationary term
$(w'/2\sigma_w^2)\,\partial\sigma_w^2/\partial t$ of the Gaussian well-mixed
drift (Thomson 1987): alone, $dw'/dt = w'\,\partial_t \ln\sigma_w$ gives
$w' \propto \sigma_w$. It maps $N(0, \sigma_{\mathrm{last}}^2)$ onto
$N(0, \sigma_{\mathrm{now}}^2)$ and moves no particle, so a uniform
concentration stays uniform when $h$ moves past it (*argued* for the
horizontal advection part: the same term along the mean flow).

**Other boundaries** (*argued* unless marked):
- *Ground reflection and the near-surface fix* (decision 0027): unchanged code;
  the closed-column gate passes (*measured*, TBD).
- *Basal clamp* ($z < z_{\mathrm{ubl}}$): $\sigma$ is continuous there (flat
  continuation), only its gradient has a kink; no interface needed. Measured in
  0027 (zeroing the gradient below the clamp: ≤ 0.003).
- *Surface-layer override* (opt-in, off by default): $\sigma_w$ jumps at
  $0.1h$ between the Monin–Obukhov and Hanna forms, and the stencil straddles
  it as the old stencil did at $h$. Not treated: the override is kept only for
  A/B comparisons. Anyone enabling it should treat $0.1h$ the same way.
- *$T_L$ discontinuities* (the convective $T_{Lw}$ branches at $|z/L| = 1$ and
  $z/h = 0.1$, the floors, the jump at $h$): $T_L$ does not enter the
  stationary velocity distribution or the well-mixed drift, and the exact OU
  update preserves $P(w; \sigma)$ for any $T_L$, so no treatment (*derived*).
- *Regime switch* ($h/L$ crossing ±1 in time or along a trajectory): a jump of
  $\sigma$ in time, handled by the carry.
- *4σ clip*: on transmission $|w'|/\sigma$ is preserved, so the clip does not
  bind on entry any more than elsewhere.
- *$h$ varying*: within an outer step $h$ is fixed per particle (the column and
  time of the step's start), so the interface is a fixed plane for the
  sub-steps. Between steps $h$ follows the met interpolation (linear within a
  window, continuous across windows) and the particle's column; a particle
  whose side changes between steps is not transmitted or reflected (it did
  not cross, the interface moved), and the carry rescales its velocities.

**Measured** (`outputs/agents/WM/interface/study/tables.md`; TBD).

**Rejected alternatives.**
- *Reflection at $h$, no turbulent exchange* (as FLEXPART reflects inside its
  boundary-layer loop). Not well-mixed-consistent: a uniform column would
  keep its free-troposphere and boundary-layer parts apart forever, which is
  harmless only if no particle ever needs to cross. Matt's decision of
  25 Nov 2026.
- *A wider finite-difference step* (10 m, PR #69's probe). Still straddles $h$
  and differences the jump, only over a wider band; it reverses the bias
  rather than removing it (TBD).
- *A smooth transition layer* between the profiles. Adds a thickness parameter
  and a strong gradient that the sub-step must resolve; the jump is a property
  of the parameterisation, and TPM handle it exactly.
- *TPM with the sub-step truncated at $h$* (as STILT). Needs a variable number
  of sub-steps, which the fixed-count, graph-captured loop does not allow; the
  ballistic completion is exactly consistent instead.
- *Rescaling only when the side of $h$ changes between steps*, with $\sigma$ at
  $h$ (measured: fixes the neutral and convective moving-$h$ column but not the
  stable one, where $\sigma_w$ at $h$ is the 1 mm s⁻¹ floor). The carry, with
  $\sigma$ at the particle, fixes all three.

**Status.** Accepted, 25 Nov 2026 (task WM, PR #69). Golden fixtures not
regenerated. Docs: [physics.md](../../docs/physics.md) §3.3, §6, §8;
[turbulence.md](../../docs/turbulence.md).
