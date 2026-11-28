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
probe reversed the bias rather than removing it (convective, $n = 16$: the 50 m
above $h$ at 0.33 and the 50 m below at 1.34 of their uniform shares, 3 seeds;
its boundary-layer retention of 0.998 came from pumping particles down). Matt chose, on
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
   *Amended by [decision 0030](0030-drift-gradient-at-the-sigma-floor.md):* a
   boundary-layer particle whose stencil is cut at $h$, or touches the
   $\sigma_w$ floor, takes the analytic gradient of the floored profile
   instead; the cut stencil over-stated the gradient at a stable $h$.
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

**What is implemented, against TPM 1997.** Checked against the paper
(Thomson, Physick & Maryon 1997, *J. Appl. Meteor.* 36, 1284–1295; section
numbers below are the paper's) by the coordinating session on 25 Nov 2026,
after the scheme had been written from STILT's implementation of it
(`pardsp.f`, Lin et al. 2003), because the paper was not available to the
implementing agent. The paper does not prescribe a single scheme: it derives
the condition any consistent scheme must meet (§2: if the particles
reaching the interface are well mixed, those leaving it must have the
well-mixed flux-weighted velocity distribution of the side they enter) and
offers several schemes that meet it. GLIDE uses its §3 "alternative
approach" for Gaussian turbulence. Point by point:

| Element | TPM 1997 | GLIDE | Relation and reason |
| --- | --- | --- | --- |
| Choice of scheme | §2–3 offer (i) a deterministic mapping of the incident velocity, Eq. (6), with reflection below a critical velocity, and (ii) the "alternative approach": transmission with probability $\sigma_w(z_{i+})/\sigma_w(z_{i-})$ (or 1), velocity scaled by that ratio, otherwise perfect reflection | (ii) | **Choice within the paper.** Simpler (no critical velocity, no logarithm), and more robust at long steps: in TPM's §7 tests method (i) over-fills the layer just above $h$ by about 4% and 8% at $\Delta t = 0.05\tau$ and $0.1\tau$, while (ii) shows "no significant problems" at either. It is a single uniform draw and two `where`s, so it is graph-safe. Like (i), it gives the maximum transmission the consistency condition allows. |
| Physics assumed inside the interface | §2: no unique model; maximum transmission is appropriate when exchange is limited by diffusion on each side, and fails only "if the diffusivity within the interface is substantially smaller than the diffusivities occurring both above and below" | the same assumption, stated here | **Follows the paper; an assumption to cite.** A capping inversion whose own turbulence is much weaker than the free troposphere's would make GLIDE's exchange too fast. The Hanna/Richardson parameterisation has no separate inversion layer to represent this. |
| Profiles on each side | §2 takes a continuous-statistics random walk model (Thomson 1987) as "truth" on each side; only the interface is discontinuous | smooth Hanna profile below $h$, Richardson closure above, each with Thomson's drift; the drift stencil kept on the particle's own side | **Follows the paper.** (STILT uses piecewise-constant layers without drift and treats every layer boundary as an interface; GLIDE departs from STILT, not from TPM.) |
| Completing the step after a crossing | §2: ideally the velocity changes "at the instant the particle reaches the interface, with $z(t+\Delta t)$ calculated in a way that accounts for the change in velocity in the middle of the time step"; §6: "continuing at this new velocity until the end of the time step", also after reflection | the transmitted overshoot scaled by $\sigma_B/\sigma_A$, the reflected one mirrored | **Follows the paper.** (STILT moves the particle to the interface and spends the remaining time in further, variable-count internal steps; GLIDE departs from STILT, which the fixed-count graph-captured loop of decision 0027 could not do anyway.) |
| Density | §2 is written for a general well-mixed density $p_a(z,w)$, which includes the air density; §3's Gaussian form assumes it uniform | no density factor | **Equivalent here.** The air density is continuous at $h$, so the general form's density ratio is 1. |
| Horizontal velocities at $z = h$ | §5: assume $u_H$ and $w$ independent at a fixed point (adopted "in many practical dispersion models"); treat $u_H$ at the interface by conserving $\int p_a\,du_H$, i.e. by keeping $u'/\sigma_u$ | $u', v'$ scaled by their own $\sigma$ ratios on transmission, unchanged on reflection | **Follows the paper**, including its independence assumption, which holds in Hanna's scheme (no $u'w'$ covariance). Would need revisiting if a covariance is added (cf. Wilson & Flesch 1993, who also reverse $u'$ at a wall when $u'$ and $w'$ are correlated). (STILT scales $w'$ only.) |
| $h$ moving | §2: a moving interface can be treated in a frame fixed in it; §4: where the model's $h$ moves in jumps, treat the jump as a discontinuity in time and keep $w/\sigma_w$ fixed (or redraw $w$ from the new distribution) | $h$ fixed during an outer step; between steps $u'/\sigma$, $v'/\sigma$, $w'/\sigma$ carried (`sigma_last`) | **Follows §4.** GLIDE's $h$ is constant within an outer step and changes between steps, which is the situation of TPM's Fig. 4. The carry is the "keep $w/\sigma_w$ fixed" option, applied to all three components (§5 extends it to $u_H$). |
| $h$ varying horizontally | §5: for $w$, treat as a time discontinuity (as above); for $u_H$, in principle conserve the flux-weighted $\int \mathbf{u}\cdot\mathbf{n}\,p_a\,d\mathbf{u}$ across the sloping interface, but "the horizontal advection across the discontinuity will probably be dominated by the mean flow, and so details of the procedure adopted are probably of less importance" | the carry ($\int p_a\,du$ conserved) for all three components | **Departure, for $u_H$ only, justified by the paper's own remark.** A particle advected into a column with a different $h$ is mostly carried by the mean wind, not by $u'$. |
| Mean vertical velocity | §2–3 assume none relative to the interface, or a split step: "the interface can be treated as if there is no mean flow in the turbulence step, and ignored in the mean flow step" | split: advection (mean $w$) and turbulence are separate steps; the crossing rule acts in the turbulence sub-steps only | **Follows the paper.** |
| Order within a step | §7: $w$ incremented with $\sigma$ at the last $z$, then $z$ moved with the new $w$ | the same | **Follows the paper.** |
| Time step on the two sides | §6: the scheme "will not work if $\Delta t$ changes discontinuously at the interface" (Fig. 8); §7 (closing warning): a time step that depends on the particle's random number "hopelessly biases the random numbers" | one sub-step length for every particle (decision 0027) | **Follows the paper**, and independently supports decision 0027's finding that a height-dependent sub-step biases the well-mixed state. |
| Step length | §7 tests method (ii) at $\Delta t$ up to $0.1\tau$ | sub-steps of 3.75 s ($n = 16$, $\Delta t = 60$ s) | **At or slightly beyond TPM's tested range.** Below $h$, $T_{Lw} \ge 30$ s (the FLEXPART floor), so $\Delta t_{\mathrm{sub}} \le 0.125\,T_{Lw}$, against TPM's largest tested $0.1\tau$; near the top of a convective or neutral boundary layer $T_{Lw}$ is usually well above the floor (*argued*). Above $h$, $T_{Lw}$ can be shorter; the exceedance counter records $\Delta t_{\mathrm{sub}}/T_{Lw}$ there (decision 0027). The open-column tests (*Measured*, below) are the direct check at GLIDE's step. |
| Velocity clip | not in TPM | GLIDE's $4\sigma$ rogue-trajectory clip, applied before the crossing | **GLIDE addition**, not part of the scheme. Transmission preserves $|w'|/\sigma$, so the clip binds no more after a crossing than elsewhere. |
| Two boundaries in one sub-step | not discussed | not treated: the crossing rule is applied once, before the ground reflection | **Limitation.** A displacement crossing both $h$ and the ground needs $h$ below about $4\sigma_w\,\Delta t_{\mathrm{sub}} \approx 30$ m, under the 50 m minimum $h$ at the defaults (*derived*, review 2, `outputs/agents/WM/review69b/`). The long-sub-step warning covers non-default settings. |

TPM also show (their Fig. 9) that a diffusion model with a smooth transition
layer converges to the discontinuous treatment as the layer is made
thinner, at about ten times the cost. That is the justification for modelling
$h$ as a discontinuity at all (see *Rejected alternatives*).

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
  the closed-column gate (`tests/test_near_surface_well_mixed.py`) passes
  (*measured*). The closed-column numbers themselves moved, because the old
  stencil also straddled the lid there: see decision 0027, "Revised".
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

**Measured** (`outputs/agents/WM/interface/study/tables.md`; 20,000 particles
per seed, 60 s outer step, records every 15 min from 2 to 6 h; mean ± standard
error over seeds, seed sd in the tables; the steady columns ran at 83c2d1e,
before the carry, which changes their same-seed results by at most 0.0009):

| | stable | neutral | convective |
| --- | --- | --- | --- |
| uniform $[0, 2h]$, $h - 50$ … $h$ / $h$ … $h + 50$, $n = 8$ (5 seeds) | 0.944 / 0.968 | 0.983 / 0.992 | 0.985 / 0.981 |
| same, $n = 16$ | 0.967 / 0.977 | 0.980 / 0.985 | 0.979 / 0.995 |
| same, $n = 32$ | 0.984 / 0.973 | 0.982 / 1.006 | 0.992 / 0.997 |
| same, PR #69 head, $n = 16$ (3 seeds) | 0.922 / 1.102 | 1.261 / 0.484 | 0.893 / 5.969 |
| same, main (3 seeds) | 0.680 / 0.531 | 0.450 / 4.592 | 0.275 / 9.871 |
| start $[0, h]$, open top: BL share 2–6 h, $n$ = 4 … 32 (5 seeds) | 0.992–0.995 | 0.975–0.976 | 0.974–0.975 |
| diffusion limit, same | 0.929 (not applicable: $T_{Lw} \to \infty$ at $h$) | 0.973 | 0.972 |
| $h$ moving, max deviation of the near-$h$ bands, no carry / carry, $n = 16$ (3 seeds) | 0.135 / 0.016 | 0.200 / 0.014 | 0.235 / 0.016 |

Seed standard deviations: 0.003–0.025 for the 50 m bands of this branch, 0.001
for the BL shares. Mechanism (*measured* by switching one part at a time, 3
seeds, $n = 64$): with the one-sided stencil but no crossing rule the 50 m
above $h$ hold 0.42 (stable), 5.8 (neutral) and 14.8 (convective) of their
share, so the crossing rule is necessary; with the crossing rule but the
straddling stencil the 10 m above $h$ hold 0.10–0.12 (neutral, convective) and
the stable 50 m above $h$ 1.11, at $n$ = 16 and 64, so the one-sided stencil is
necessary too. The stable residual
below $h$ falls as the sub-step shrinks ($-0.056$, $-0.033$, $-0.016$ at
$n$ = 8, 16, 32), the same first-order error that over-fills the bottom of the
stable column; the 10 m just above $h$ follow it (0.940 ± 0.011), as a
continuous concentration across a consistent interface requires (*argued*;
starting the velocities from $N(0, \sigma^2)$ or lowering the $\sigma$ floor to
0.1 mm s⁻¹ moves that band to 0.960–0.981, within 1–2 standard errors of the
band below).

On real meteorology (*measured*): CPU smoke (Mace Head, one release, 1 day,
3 seeds): 0–40 m footprint +92.0% / +94.4% / +95.8% (seed sd 0.4–0.6) against
the fixture at $n$ = 8, 16, 32, total −6.6%; $n = 8$ against 16, same seed,
−1.2 ± 0.2% in 0–40 m. Reference config
(`configs/ab_multisite_perf.yaml`, 144 releases, GPU; 3 seeds; median over
releases against the CUDA fixture; `outputs/agents/WM/interface/bench/`):
0–40 m +47.9% (seed sd 0.2%), 40–1,000 m +119%, 1–5 km −31.0%, total −4.2% at
$n = 16$; $n = 8$ against 16, same seed, +0.4% in 0–40 m (at most 1.2% in any
site's day or night releases), and the share of sub-steps above the boundary
layer 52.6% against 52.7%. With decision 0027 alone the 0–40 m change was −16%
and +15% at $n$ = 8 and 16 (one seed), with 72% and 61% of sub-steps above the
boundary layer: the boundary-layer top was what kept footprints from
converging in $n$ (*measured* change; the attribution is *argued* from the
columns).

**Rejected alternatives.**
- *Reflection at $h$, no turbulent exchange* (as FLEXPART reflects inside its
  boundary-layer loop). Not well-mixed-consistent: a uniform column would
  keep its free-troposphere and boundary-layer parts apart forever, which is
  harmless only if no particle ever needs to cross. Matt's decision of
  25 Nov 2026.
- *A wider finite-difference step* (10 m, PR #69's probe). Still straddles $h$
  and differences the jump, only over a wider band; it reverses the bias
  rather than removing it (convective h … h + 50 m at 0.33 of uniform, above).
- *A smooth transition layer* between the profiles. Adds a thickness parameter
  and a strong gradient that the sub-step must resolve; the jump is a property
  of the parameterisation, and TPM handle it exactly. TPM's Fig. 9 shows the
  smooth-layer results converging to the interface treatment as the layer
  thins, at about ten times the computing cost.
- *Truncating the sub-step at $h$* (as STILT does). Needs a variable number
  of sub-steps, which the fixed-count, graph-captured loop does not allow; TPM
  themselves complete the step at the new velocity (§2, §6), which is what
  GLIDE does, and it is exactly consistent (above).
- *TPM's deterministic scheme, Eq. (6)* (method (i) of §3). Consistent too, but
  TPM found it over-fills the layer above the interface by 4–8% at
  $\Delta t = 0.05$–$0.1	au$, where the scheme adopted showed no such error.
- *Rescaling only when the side of $h$ changes between steps*, with $\sigma$ at
  $h$ (measured: fixes the neutral and convective moving-$h$ column but not the
  stable one, where $\sigma_w$ at $h$ is the 1 mm s⁻¹ floor). The carry, with
  $\sigma$ at the particle, fixes all three.

**Status.** Accepted, 25 Nov 2026 (task WM, PR #69); checked against TPM 1997
the same day (table above). Golden fixtures not
regenerated. Docs: [physics.md](../../docs/physics.md) §3.3, §6, §8;
[turbulence.md](../../docs/turbulence.md).
