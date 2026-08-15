# The GLIDE model

What GLIDE computes, and the equations it integrates to get there. This page is
the model formulation: the stochastic differential equation, the coordinate
system, the backward-in-time construction, the boundary conditions, and the
definition of the footprint it produces.

The two parameterisations that supply coefficients to those equations get their
own pages — [turbulence.md](turbulence.md) for $\sigma$ and $T_L$, and
[convection.md](convection.md) for deep convective transport. How the code is
organised to run this at scale is [architecture.md](architecture.md).

**Contents**

1. [What a footprint is](#1-what-a-footprint-is)
2. [Coordinates and particle state](#2-coordinates-and-particle-state)
3. [The governing equations](#3-the-governing-equations)
4. [Running backward in time](#4-running-backward-in-time)
5. [Boundary conditions](#5-boundary-conditions)
6. [Time stepping](#6-time-stepping)
7. [Footprint accumulation](#7-footprint-accumulation)
8. [Known approximations](#8-known-approximations)
9. [References](#9-references)

---

## 1. What a footprint is

An atmospheric measurement at a receptor — a tall-tower inlet, say — responds to
surface fluxes upwind of it. The **footprint** is the linear sensitivity that
connects the two:

$$
\Delta c(\mathbf{x}_r, t_r) \;=\; \int \! \int f(\mathbf{x}_r, t_r \,|\, \mathbf{x}_s, t_s)\; F(\mathbf{x}_s, t_s)\; \mathrm{d}\mathbf{x}_s \, \mathrm{d}t_s
$$

where $F$ is the surface flux and $f$ is the footprint. Computing $f$ for one
receptor by running a forward model from every possible source is hopeless;
running **one backward simulation from the receptor** gives the whole field at
once. That adjoint relationship is why LPDMs used for flux inversion run
backward, and it is what GLIDE is built to do.

Following Seibert & Frank (2004), the footprint of a backward run is the
**residence time** the released particle ensemble spends in each source volume,
per unit released mass. GLIDE accumulates exactly that (§7); converting it to a
concentration-per-flux sensitivity in STILT units is a post-processing step
(`lpdm.comparison.to_stilt_surface_footprint`).

---

## 2. Coordinates and particle state

Each particle carries four numbers:

| Symbol | Meaning | Units |
| --- | --- | --- |
| $\lambda$ | longitude | degrees east |
| $\phi$ | latitude | degrees north |
| $z$ | height **above ground level** | metres |
| $w_p$ | mass weight | dimensionless |

plus turbulent velocity state $(u', v', w')$, and $(u_m, v_m)$ when the meander
process is enabled.

Two choices in that table do real work.

**Height is geometric metres above ground, not pressure.** Every piece of the
physics is naturally posed in AGL: $\sigma_w$ and $T_L$ scale with $z/h$ where
$h$ is boundary-layer depth, surface reflection happens at $z = 0$, and the
footprint's surface layer is "0–40 m above the ground". Carrying pressure
internally would mean converting at every one of those points. GLIDE converts
once, in the meteorology reader, and the integration loop never sees a pressure
coordinate.

**The vertical grid follows the terrain.** ERA5 pressure levels are
quasi-horizontal — they cut through mountains, so levels below the local surface
exist by construction and carry ERA5's fictitious below-ground extrapolation. The
reader therefore resamples each met hour onto a **fixed terrain-following AGL
ladder** shared by all columns, excluding sub-surface levels, in the manner of
FLEXPART's `verttransform`. The vertical velocity is transformed into that frame:

$$
w_{\mathrm{AGL}} \;=\; w \;-\; \tau(z)\left(u\,\frac{\partial h_s}{\partial x} + v\,\frac{\partial h_s}{\partial y}\right),
\qquad
\tau(z) = \max\!\left(0,\, 1 - \frac{z}{z_{\mathrm{top}}}\right)
$$

$h_s$ is the surface elevation. The bracketed term is the vertical velocity a
particle riding the horizontal wind needs simply to hold its height above sloping
ground; subtracting it leaves the motion *relative to the terrain*, which is what
an AGL coordinate should evolve. The taper $\tau$ relaxes the correction to zero
at the model top so orography does not perturb the stratosphere.

Without this, a near-surface particle crossing an 800 m hill rides the terrain
upward by roughly the full hill height; with it, it holds its AGL to a few metres
(`tests/test_terrain_transport.py`). The consequence for footprints is not
subtle: before the fix, 86.6% of high-terrain cells had *zero* surface footprint.

The default ladder is 23 levels from 0 to `met_domain.alt_max_m`, 13 of them
below 1.5 km; `met_domain.vertical_levels` replaces it with a geometrically
stretched grid of any count, or an explicit list. **This grid, not the source's
level count, sets the vertical resolution the physics sees** — see
[met_schema.md](met_schema.md).

Horizontal displacements are computed in metres and converted to degrees with a
local spherical metric,

$$
\frac{\partial \lambda}{\partial x} = \frac{1}{111320 \, \lvert\cos\phi\rvert}\ \mathrm{deg\ m^{-1}},
\qquad
\frac{\partial \phi}{\partial y} = \frac{1}{110540}\ \mathrm{deg\ m^{-1}}
$$

with $\lvert\cos\phi\rvert$ floored at 0.05 (about 87°) so the polar singularity
cannot produce an infinite zonal step.

---

## 3. The governing equations

GLIDE is a **first-order Lagrangian stochastic model** in the sense of Thomson
(1987): turbulent velocity is a state variable with memory, not a random
displacement applied to position. This matters for footprints because the
near-field — the first few $T_L$ after release, right next to the receptor and
right next to the ground — is exactly the regime a zeroth-order random-walk model
gets wrong, and it is where the footprint is largest.

The trajectory of one particle obeys

$$
\mathrm{d}u_i' = a_i(\mathbf{x}, \mathbf{u}', t)\,\mathrm{d}t \;+\; b_{ij}\,\mathrm{d}\xi_j,
\qquad
\mathrm{d}x_i = \left(U_i + u_i'\right)\mathrm{d}t
$$

with $U_i$ the resolved (meteorology) wind, $u_i'$ the turbulent fluctuation,
and $\mathrm{d}\xi$ a Wiener increment of variance $\mathrm{d}t$.

GLIDE integrates this by **operator splitting**: within one step, the resolved
advection is applied first, then the turbulent velocity is updated and its
displacement applied, then (once per meteorology window) convection. Each piece
is described below.

### 3.1 Resolved advection

$$
\frac{\mathrm{d}\mathbf{x}}{\mathrm{d}t} = \mathbf{U}(\mathbf{x}, t)
$$

integrated with a second-order Runge–Kutta midpoint step. Backward in time:

$$
\mathbf{x}^{*} = \mathbf{x}_n - \tfrac{1}{2}\Delta t\; \mathbf{U}\!\left(\mathbf{x}_n, t_{n-1/2}\right),
\qquad
\mathbf{x}_{n-1} = \mathbf{x}_n - \Delta t\; \mathbf{U}\!\left(\mathbf{x}^{*}, t_{n-1/2}\right)
$$

Both stages evaluate the wind at the **same** midpoint time
$t_{n-1/2} = t_n - \Delta t/2$, which is what makes this the midpoint rule
rather than a half-and-half hybrid. $\mathbf{U}$ comes from trilinear interpolation
(`grid_sample`) of the two bracketing meteorology hours, linearly weighted in
time by $\alpha = (t_{n-1/2} - t_{\mathrm{hour}}) / 3600$.

The vertical index used by that interpolation is a piecewise-linear lookup into
the AGL level array, not a linear-in-metres mapping — the ladder is stretched, so
treating it as uniform would systematically warp the vertical wind shear.

Convergence is verified at second order in $\Delta t$
(`test_rk2_advection_second_order_in_dt`) and against an analytic solid-body
rotation (`test_solid_body_rotation_advection_returns_to_start`).

### 3.2 The turbulent velocity: an exact Ornstein–Uhlenbeck step

For each component, GLIDE uses the **exact** solution of the homogeneous OU
process over $\Delta t$, with the inhomogeneous drift added as a forward-Euler
increment:

$$
\boxed{\;
u'_{n+1} \;=\; a\,u'_n \;+\; a_{\mathrm{drift}}\,\Delta t \;+\; \sigma\sqrt{1 - a^2}\;\eta,
\qquad a = e^{-\Delta t / T_L},\quad \eta \sim \mathcal{N}(0,1)
\;}
$$

Two properties are worth stating explicitly.

*It is unconditionally stationary.* Set $a_{\mathrm{drift}} = 0$: if
$\mathrm{Var}(u'_n) = \sigma^2$ then
$\mathrm{Var}(u'_{n+1}) = a^2\sigma^2 + \sigma^2(1-a^2) = \sigma^2$, for
**any** $\Delta t$. A naive Euler
discretisation of $\mathrm{d}u' = -u'/T_L\,\mathrm{d}t + b\,\mathrm{d}\xi$ loses
this and blows up once $\Delta t > 2T_L$. The accuracy limit on $\Delta t$ in
GLIDE therefore comes from the *drift* term and from position integration, not
from stability (§6).

*It has the right inertial-subrange limit.* As $\Delta t / T_L \to 0$,
$\sigma\sqrt{1-a^2} \to \sigma\sqrt{2\Delta t/T_L}$, i.e. $b\sqrt{\Delta t}$ with

$$
b^2 = \frac{2\sigma^2}{T_L}
$$

which is the standard Thomson (1987) diffusion coefficient, tied to
$C_0\varepsilon$ through the same identification. $b$ is diagonal — GLIDE carries
no $u$–$w$ cross-correlation, in common with FLEXPART and other regional models
(Stohl & Thomson 1999).

The autocorrelation $R(\tau) = e^{-\tau/T_L}$ and the stationary variance are
both verified numerically (`test_ou_autocorrelation_and_stationarity`), as is the
resulting Taylor dispersion curve across the ballistic-to-diffusive transition
(`test_taylor_dispersion_curve_ballistic_to_diffusive`):

$$
\sigma_z^2(t) = 2\sigma_w^2 T_L\left[t - T_L\left(1 - e^{-t/T_L}\right)\right]
$$

### 3.3 The drift term, and why it is not just $-w'/T_L$

In *homogeneous* turbulence the drift is pure relaxation, $a = -w'/T_{Lw}$. The
real boundary layer is nothing of the sort: $\sigma_w$ varies strongly with
height, and air density falls with height. The **well-mixed condition** (Thomson
1987) — an ensemble distributed according to the air's density-weighted velocity
PDF must stay so distributed forever, absent sources — then *determines* the
extra drift terms. They are not optional corrections; omitting them makes the
model wrong in a specific, measurable way.

For a density-weighted Gaussian velocity PDF, the forward vertical drift is

$$
a_w \;=\; \underbrace{-\frac{w'}{T_{Lw}}}_{\text{relaxation}}
\;+\; \underbrace{\frac{1}{2}\left(1 + \frac{w'^2}{\sigma_w^2}\right)\frac{\partial \sigma_w^2}{\partial z}}_{\sigma\text{-gradient (Thomson 1987)}}
\;+\; \underbrace{\frac{\sigma_w^2}{\rho}\frac{\partial \rho}{\partial z}}_{\text{density (Stohl and Thomson 1999)}}
$$

**The $\sigma$-gradient term** pushes particles down the gradient of turbulence
intensity. Without it, particles accumulate in low-turbulence regions — they
drift *up* the $\sigma_w$ gradient, pile up above the boundary layer, and stop
recycling to the surface. In GLIDE's own history this was not a theoretical
concern: removing it under-dispersed the surface footprint badly. It is the
single most important term in this equation after the relaxation.

The $(1 + w'^2/\sigma_w^2)$ factor is velocity-dependent and is re-evaluated
every sub-step with the current $w'$. Freezing it at the step-start value
systematically under-drifts freshly released particles, whose $w'$ is still near
zero — a real bug caught by `test_hanna_well_mixed_no_runaway_lofting`.

$\partial\sigma_w^2/\partial z$ is a central finite difference (±1 m) of the
**fully assembled** $\sigma_w$ profile, so it spans the seams between regimes —
boundary layer, optional surface layer, free troposphere — rather than
differentiating one branch in isolation. This is what keeps the well-mixed
condition intact across a regime boundary.

**The density term** exists because the well-mixed condition is enforced in
Cartesian $z$, but the equilibrium distribution of air is $\propto \rho(z)$. Over
a deep boundary layer $\rho$ at the top can be 20% below the surface value.
Without the term, an initially $\rho$-weighted ensemble relaxes toward uniform,
biasing surface concentrations — and in a backward run that bias propagates
straight into the inferred flux. Stohl & Thomson measured +5.5% mean (range
+1–15%) on surface concentrations in their CAPTEX runs.

$\rho = p/(R_d T)$ and $\partial\rho/\partial z$ are built on the meteorology
grid once per window and sampled trilinearly per particle.

**Horizontal components carry no drift.** The inhomogeneity that the well-mixed
correction addresses is vertical; FLEXPART likewise applies it to the vertical
only.

Both terms are gated by well-mixed tests that run in CI: a uniform ensemble stays
uniform under constant $\rho$
(`test_v1_well_mixed_hanna_backward_path`), and a $\rho$-weighted ensemble stays
$\rho$-weighted under varying $\rho$
(`test_v1_density_weighted_well_mixed_with_F2`).

---

## 4. Running backward in time

Backward integration is **not** the same as running forward with the wind
reversed. Reversing the mean advection is correct; reversing the whole drift is
not, and Smith's reciprocal theorem — that the backward model equals the
$U$-reversed forward model — is false in inhomogeneous turbulence.

From Flesch, Wilson & Yee (1995), the backward drift is

$$
a^{b} \;=\; -a^{f} \;+\; b^2 \frac{\partial \ln g_a}{\partial w'}
\;=\; -a^{f} - \frac{2w'}{T_{Lw}}
$$

for a symmetric Gaussian $g_a$. Substituting the forward drift from §3.3, the
$-w'/T_{Lw}$ relaxation comes back **unchanged**, while the two inhomogeneity
terms flip sign together:

$$
a^{b}_w \;=\; -\frac{w'}{T_{Lw}}
\;-\; \frac{1}{2}\left(1 + \frac{w'^2}{\sigma_w^2}\right)\frac{\partial \sigma_w^2}{\partial z}
\;-\; \frac{\sigma_w^2}{\rho}\frac{\partial \rho}{\partial z}
$$

In code this is exactly what happens: the forward drift is assembled and then
negated, with the relaxation supplied separately by the exact-OU factor $a$.

Getting this sign wrong is not a small error. It converts the $\sigma$-gradient
correction into a one-way upward pump; when it was wrong, the entire Mace Head
particle population lofted to ~2 km and stayed there.

The random forcing is symmetric, so the noise term needs no sign treatment. The
displacement of position simply runs the other way: $z_{n+1} = z_n - w'\Delta t$.

**Mass bookkeeping.** Particles carry a weight $w_p = 1/N$ set at release and
never modified by transport; convection redistributes particles between levels
but conserves their number and weight. Total weight is checked in the physics
tests as a conservation invariant.

---

## 5. Boundary conditions

The footprint is dominated by near-ground residence time, so the ground boundary
is where correctness matters most.

**Ground: smooth-wall reflection.** A particle that crosses $z = 0$ within a
sub-step is reflected by the *joint* mapping

$$
(z,\; w') \;\longmapsto\; (-z,\; -w')
$$

Both must reverse. Reflecting only the position — a real bug in GLIDE's history —
leaves the reflected particle still pointing downward into the boundary for
roughly a $T_L$ worth of steps, which biases near-surface residence time and
inflates the surface footprint. The engine's `reflect_surface` returns both
values as a pair specifically so the joint reversal is hard to omit.

**The unresolved basal layer.** Wilson & Flesch (1993, §7b) show that smooth-wall
reflection is exactly well-mixed-preserving only where the velocity PDF is
homogeneous and symmetric over the step — which it is not in the near-surface
layer, where $\sigma_w$ has its strongest gradient. The standard device is to
declare a thin basal layer over which the statistics are held constant. GLIDE
does this by clamping the *sampling* height:

$$
z_{\mathrm{eval}} = \max(z,\; z_{\mathrm{ubl}}), \qquad z_{\mathrm{ubl}} = 2\ \mathrm{m\ (default)}
$$

$\sigma$, $T_L$, the drift, and the density gradient are all evaluated at
$z_{\mathrm{eval}}$. **The particle's position is not clamped** — only the
profile lookup. At 2 m the layer is thinner than any practical release altitude,
so it intercepts nothing but the post-reflection bounce. Note this is the
opposite of the "artificial unattainability" schemes that clamp particles to
$z = 0$ or restrict $\Delta t$ to prevent crossings; those always violate the
well-mixed condition, and GLIDE contains none of them.

**Domain edges: drop and count.** A particle leaving the horizontal `met_domain`
bounding box, or rising above `met_domain.alt_max_m`, has no valid meteorology —
the coordinate normalisation would clamp it to the grid edge and it would be
pushed by edge winds for the rest of the run. Such particles have their liveness
bit cleared and are never advected or accumulated again. The count is reported in
`run_metadata.json`. There is no lower kill: $z = 0$ is handled by reflection.

---

## 6. Time stepping

The outer step $\Delta t$ is `simulation.dt_seconds` (60 s in the shipped
configs). Because the OU update is unconditionally stationary (§3.2), the
constraint on $\Delta t$ comes from two other places:

- the **forward-Euler drift increment**, whose error grows once
  $\Delta t \gtrsim T_L/5$. Wilson & Flesch (1993) derive an explicit bias velocity
  $w_B/\sigma_w \approx -\alpha\beta(\Delta t/T_L)$ for the near-surface layer;
  Stohl & Thomson (1999) recommend the stricter $\Delta t \le 0.05\,T_L$.
- the **reflection bias**, whose magnitude also scales with $\Delta t/T_L$.

Near-surface $T_{Lw}$ can fall to tens of seconds, so a single global $\Delta t$
small enough for the worst particle would be ruinously expensive for the rest.
GLIDE instead **sub-steps per particle**:

$$
k_i = \left\lceil \frac{\Delta t}{c\, T_{Lw,i}} \right\rceil,
\qquad
\Delta t_{\mathrm{sub},i} = \frac{\Delta t}{k_i},
\qquad
c = 0.5,\quad k_i \le k_{\max}
$$

Inside the sub-step loop, the OU update, the displacement, and the ground
reflection all run at $\Delta t_{\mathrm{sub}}$, so a particle that crosses the
ground mid-step is reflected before the next sub-step rather than at the end of
the outer step. $\sigma$, $T_L$, $\partial\sigma_w^2/\partial z$ and
$\partial\rho/\partial z$ are held at their outer-step values (§8); the
velocity-dependent $(1 + w'^2/\sigma_w^2)$ factor is not.

$c = 0.5$ rather than Stohl & Thomson's 0.05 because the sub-step count is capped
at `max_substeps` to bound the per-step cost; the cap, not $c$, is the binding
constraint for the most demanding particles. A once-per-run warning fires if the
cap saturates. With the FLEXPART $T_{Lw} \ge 30$ s floor in force (the default,
see [turbulence.md](turbulence.md)) it rarely does — at $\Delta t = 60$ s the
required count is $\lceil 60/(0.5 \times 30)\rceil = 4$, which is why the shipped
configs set `max_substeps: 6`.

**Rogue-trajectory clip.** At the end of each sub-step's velocity update,

$$
\lvert u_i' \rvert \le 4\,\sigma_i
$$

A true Gaussian exceeds $4\sigma$ with probability $\sim 6\times10^{-5}$, so this
is a safety net, not a physics change. It exists because where the sub-step cap
binds *and* $\sigma_w$ sits near its numerical floor — the stable boundary-layer
top, where $\sigma_w \to 0$ — the $(1 + w'^2/\sigma_w^2)$ factor can snowball the
drift into a NaN. FLEXPART has the equivalent clip.

---

## 7. Footprint accumulation

At every step, every live particle inside the output grid adds

$$
\Delta f = w_p \, \Delta t
$$

to the cell it occupies, in its own time-ago bin, in its own release's slice. The
grid is 5-dimensional:

$$
f\big[\,\text{release},\ \text{time\_ago},\ z,\ \text{lat},\ \text{lon}\,\big]
$$

Raw units are **seconds** (weight is dimensionless and sums to 1 per release), so
the raw value is residence time per unit released mass. The vertical bins are set
by `output_grid.z_edges_m`, an arbitrary strictly-ascending edge list — e.g.
`[0, 40, 1000, 5000]` gives a 0–40 m surface bin matching the FLEXPART/NAME
convention, a mixed-layer bin, and a free-troposphere bin. Accumulating *directly*
into a bin that matches the reference's surface layer makes the downstream unit
conversion exact rather than a depth-weighted approximation.

Time-ago bins are one hour wide and are measured from each particle's **own**
release window end, so a multi-release batch produces correctly-aged footprints
for every release from one sweep. With `n_time_bins: 1` all ages collapse into one bin, giving the
time-integrated footprint the FLEXPART comparison uses; with more bins, ages past
the last bin are dropped rather than clamped, so the axis label stays honest.

Particles outside the grid horizontally, vertically, in time-ago, or in release
index contribute nothing.

**Conversion to STILT units.** Lin et al. (2003) Eq. 5:

$$
f_{\mathrm{STILT}}(y,x) \;=\; \frac{m_{\mathrm{air}}}{h\,\bar{\rho}} \sum_{t}\sum_{z \in \text{surface layer}} f\big[t,z,y,x\big]
$$

giving $\mathrm{m^2\,s\,mol^{-1}}$, equivalently
$(\mathrm{mol/mol})/(\mathrm{mol\,m^{-2}\,s^{-1}})$. $h$ is the surface-layer
depth, $\bar{\rho}$ the surface air density (a scalar, or a 2-D field derived from
the meteorology by `surface_air_density_from_met`), and
$m_{\mathrm{air}} = 0.02897\ \mathrm{kg\,mol^{-1}}$. Bins that only partially
overlap the chosen surface layer are credited by their overlap fraction.

The whole chain — advection, turbulence, reflection, gridding, and the STILT
conversion — is validated end-to-end against an analytic reflected-Gaussian plume
in `tests/test_plume_footprint.py`, matching absolute magnitude to within 1%.

---

## 8. Known approximations

Stated plainly, because they bound what the model can be trusted for.

| Approximation | Consequence |
| --- | --- |
| Gaussian vertical velocity PDF everywhere — no skewed convective-boundary-layer PDF | Under-represents the updraft/downdraft asymmetry of a strongly convective BL. Thomson (1987, §5.2) found ground-level concentration relatively insensitive to skewness even where dispersion aloft is sensitive, so this is defensible for a surface-flux footprint model — but it is a modelling choice, not a free lunch. |
| $\sigma$, $T_L$ and the profile gradients held fixed across the sub-steps of one outer step | FLEXPART re-evaluates per sub-step. The change in $\sigma$ across one 60 s outer step is moderate; the velocity-dependent part of the drift *is* re-evaluated. A documented follow-up. |
| Turbulence support fields ($\rho$, free-troposphere $\sigma/T_L$, meander $\sigma$) built once per meteorology hour at the window midpoint | A met-cadence approximation (<1%/hr drift), matching how FLEXPART-class models refresh turbulence fields. Per-particle interpolation through those fields is unchanged. |
| Deep convection uses one bounding-box-mean column for the whole domain | Convection fires or does not fire uniformly across the met domain. Fine for a small domain; for continental Europe it can over- or under-trigger. See [convection.md](convection.md) §4. |
| Convection interval hardcoded at 3600 s | Correct for hourly ERA5; wrong for any other cadence. |
| Initial $u', v', w' = 0$ rather than sampled from the local $\sigma$ | Particles equilibrate within one $T_L$ (typically ~100 s), which is shorter than the release window. |
| Virtual temperature approximated as $T$ in the Obukhov length | Humidity correction is small; deferred. |
| Free-troposphere horizontal turbulence treated as isotropic with the vertical | The unresolved-mesoscale horizontal spread is carried separately by the meander process. |

Beyond these, the transport physics as a whole **has not been validated against
external references**. The analytic tests below verify GLIDE against closed-form
solutions; they do not establish that the parameterisations are right for the
real atmosphere. See [VALIDATION.md](VALIDATION.md) and
[../STATUS.md](../STATUS.md).

---

## 9. References

- Flesch, T. K., Wilson, J. D., Yee, E. (1995). Backward-time Lagrangian
  stochastic dispersion models and their application to estimate gaseous
  emissions. *J. Appl. Meteor.* 34, 1320–1332.
- Lin, J. C., Gerbig, C., Wofsy, S. C., Andrews, A. E., Daube, B. C., Davis,
  K. J., Grainger, C. A. (2003). A near-field tool for simulating the upstream
  influence of atmospheric observations: STILT. *J. Geophys. Res.* 108, 4493.
- Seibert, P., Frank, A. (2004). Source–receptor matrix calculation with a
  Lagrangian particle dispersion model in backward mode. *Atmos. Chem. Phys.* 4,
  51–63.
- Stohl, A., Thomson, D. J. (1999). A density correction for Lagrangian particle
  dispersion models. *Boundary-Layer Meteorol.* 90, 155–167.
- Taylor, G. I. (1921). Diffusion by continuous movements. *Proc. London Math.
  Soc.* 20, 196–212.
- Thomson, D. J. (1987). Criteria for the selection of stochastic models of
  particle trajectories in turbulent flows. *J. Fluid Mech.* 180, 529–556.
- Wilson, J. D., Flesch, T. K. (1993). Flow boundaries in random-flight
  dispersion models: enforcing the well-mixed condition. *J. Appl. Meteor.* 32,
  1695–1707.
- Wilson, J. D., Legg, B. J., Thomson, D. J. (1983). Calculation of particle
  trajectories in the presence of a gradient in turbulent-velocity variance.
  *Boundary-Layer Meteorol.* 27, 163–169.
