# The GLIDE model

What GLIDE computes, and the equations it integrates to get there. This page is
the model formulation: the stochastic differential equation, the coordinate
system, the backward-in-time construction, the boundary conditions, and the
definition of the footprint it produces.

The two parameterisations that supply coefficients to those equations get their
own pages — [turbulence.md](turbulence.md) for the turbulence strength $\sigma$
and memory timescale $T_L$ (both defined in §2), and
[convection.md](convection.md) for deep convective transport. How the code is
organised to run this at scale is [architecture.md](architecture.md).

**Contents**

1. [What a footprint is](#1-what-a-footprint-is)
2. [Coordinates, state, and notation](#2-coordinates-state-and-notation)
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
\Delta c(\mathbf{x}_r, t_r) \ =\  \int  \int f(\mathbf{x}_r, t_r \ |\  \mathbf{x}_s, t_s)\  F(\mathbf{x}_s, t_s)\  \mathrm{d}\mathbf{x}_s \  \mathrm{d}t_s
$$

Reading that from the inside out:

- $\mathbf{x}_r, t_r$ — where and when the **receptor** measures.
- $\mathbf{x}_s, t_s$ — where and when a **source** emits. The integral runs over
  every source location and every earlier time.
- $F$ — the surface flux at that source (e.g. mol m⁻² s⁻¹).
- $f$ — the **footprint**: how much the receptor's reading responds to a unit
  flux at that source. Large where the air arriving at the receptor recently
  touched the ground; zero where it never went.
- $\Delta c$ — the resulting enhancement in measured concentration above
  background.

Computing $f$ for one receptor by running a forward model from every possible
source is hopeless; running **one backward simulation from the receptor** gives
the whole field at once. That adjoint relationship is why Lagrangian particle
dispersion models (LPDMs) used for flux inversion run backward, and it is what
GLIDE is built to do.

Following Seibert & Frank (2004), the footprint of a backward run is the
**residence time** the released particle ensemble spends in each source volume,
per unit released mass. GLIDE accumulates that, divided by the dry-air density at
each particle as it is counted (§7, decision 0022); converting it to a
concentration-per-flux sensitivity in STILT units is a post-processing step
(`lpdm.comparison.to_stilt_surface_footprint`).

---

## 2. Coordinates, state, and notation

Each particle carries four numbers:

| Symbol | Meaning | Units |
| --- | --- | --- |
| $\lambda$ | longitude | degrees east |
| $\phi$ | latitude | degrees north |
| $z$ | height **above ground level** (AGL) | metres |
| $m_p$ | mass weight — the share of the release this particle represents | dimensionless |

plus the turbulent velocity state $(u', v', w')$ described below, and
$(u_m, v_m)$ when the *meander* process is enabled — an extra, slower horizontal
wobble standing in for eddies too large for the turbulence scheme but too small
for the meteorology grid to resolve (see [turbulence.md §7](turbulence.md#7-meander-unresolved-mesoscale-motion)).

### Notation

These recur throughout, and are worth having in one place.

| Symbol | What it means |
| --- | --- |
| $u, v, w$ | wind components: eastward, northward, upward (m s⁻¹) |
| $\mathbf{U}$ | the **resolved** wind — what the meteorology actually carries, interpolated to the particle |
| $u', v', w'$ | the **turbulent fluctuation** each particle carries: its own departure from the resolved wind. This is the model's memory of the eddy it is riding, and it is a state variable, not a random number redrawn each step. |
| $\sigma_u, \sigma_v, \sigma_w$ | how vigorous the turbulence is in each direction — the standard deviation of the corresponding fluctuation (m s⁻¹) |
| $T_{Lu}, T_{Lv}, T_{Lw}$ | **Lagrangian timescales**: how long a particle remembers its fluctuation before it decorrelates (s). Written $T_L$ where the component does not matter. |
| $K = \sigma_w^2 T_{Lw}$ | the eddy diffusivity these two imply — a convenient single number for "how fast does this layer mix" (m² s⁻¹) |
| $h$ | boundary-layer depth (m) |
| $\rho$ | air density, $\rho = p/(R_d T)$, with $p$ pressure, $T$ temperature and $R_d = 287.05$ J kg⁻¹ K⁻¹ the gas constant for dry air |
| $\Delta t$ | the integration step (`simulation.dt_seconds`; 60 s in the shipped configs) |
| $N$ | the number of particles in one release |
| $g_a$ | the density-weighted velocity PDF — the distribution of vertical velocities the *air itself* has at a given height. The well-mixed condition is a statement about this. |
| $a$, $b$ | the **drift** (deterministic acceleration) and **diffusion** (random forcing) coefficients of the Langevin equation, §3 |
| $C_0$, $\varepsilon$ | the Kolmogorov constant for the Lagrangian structure function, and the turbulent kinetic energy dissipation rate — the inertial-subrange quantities $b$ is tied to |

Two conventions worth stating, because both are easy to misread:

- **A prime always means a turbulent fluctuation, never a derivative.** $w'$ is
  the particle's vertical velocity departure from the resolved wind.
- **Subscripts $u$, $v$, $w$ on $\sigma$ and $T_L$ select the component**, not a
  derivative or a location. $T_{Lw}$ is the vertical Lagrangian timescale.

Two choices in the state table do real work.

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
w_{\mathrm{AGL}} \ =\  w \ -\  s(z)\left(u\ \frac{\partial h_s}{\partial x} + v\ \frac{\partial h_s}{\partial y}\right),
\qquad
s(z) = \max\left(0,\  1 - \frac{z}{z_{\mathrm{top}}}\right)
$$

Here $h_s$ is the surface elevation (m), $x$ and $y$ are eastward and northward
distance (m), so $\partial h_s/\partial x$ and $\partial h_s/\partial y$ are the
terrain slopes; $z_{\mathrm{top}}$ is the top of the AGL grid
(`met_domain.alt_max_m`), and $s(z)$ is a taper running from 1 at the ground to 0
there.

The bracketed term is the vertical velocity a particle riding the horizontal wind
needs simply to hold its height above sloping ground; subtracting it leaves the
motion *relative to the terrain*, which is what an AGL coordinate should evolve.
The taper relaxes the correction to zero at the model top so orography does not
perturb the stratosphere.

Without this, a near-surface particle crossing an 800 m hill rides the terrain
upward by roughly the full hill height; with it, it holds its AGL to a few metres
(`tests/test_terrain_transport.py`). The consequence for footprints is not
subtle: before the fix, 86.6% of high-terrain cells had *zero* surface footprint.

The default ladder is 40 levels from 0 to `met_domain.alt_max_m`, stretched
geometrically from a `first_layer_m` (10 m) lowest layer (decision 0017;
interim until task A5). `met_domain.vertical_levels` replaces it with a
stretched grid of any count, or an explicit list; the pre-A01 default was a
fixed 23-level ladder. **This grid, not the source's
level count, sets the vertical resolution the physics sees** — see
[met_schema.md](met_schema.md).

Horizontal displacements are computed in metres and converted to degrees with a
local spherical metric,

$$
\frac{\partial \lambda}{\partial x} = \frac{1}{111320 \  \lvert\cos\phi\rvert}\ \mathrm{deg\ m^{-1}},
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
\mathrm{d}u_i' = a_i(\mathbf{x}, \mathbf{u}', t)\ \mathrm{d}t \ +\  b_{ij}\ \mathrm{d}\xi_j,
\qquad
\mathrm{d}x_i = \left(U_i + u_i'\right)\mathrm{d}t
$$

The index $i$ runs over the three directions (east, north, up), so this is three
coupled equations. Term by term:

- $x_i$, $U_i$, $u_i'$ — position, resolved wind and turbulent fluctuation in
  direction $i$. The second equation just says the particle moves at the total
  wind, resolved plus turbulent.
- $a_i$ — the **drift**: the deterministic part of how the fluctuation evolves
  (units m s⁻²). Mostly it pulls $u_i'$ back toward zero over a time $T_L$, but
  not only that — §3.3 is entirely about the rest of it.
- $b_{ij}$ — the **diffusion coefficient**: how hard the random kicks are, in
  $\mathrm{m\ s^{-3/2}}$. GLIDE's is diagonal, so each component is kicked
  independently.
- $\mathrm{d}\xi_j$ — a Wiener increment: Gaussian white noise with mean zero and
  variance $\mathrm{d}t$. This is the only stochastic ingredient.

That the *velocity* gets the random kick, and the position merely integrates the
velocity, is what makes this a first-order model.

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
\mathbf{x}^{\ast} = \mathbf{x}_n - \tfrac{1}{2}\Delta t\  \mathbf{U}\left(\mathbf{x}_n, t_{n-1/2}\right),
\qquad
\mathbf{x}_{n-1} = \mathbf{x}_n - \Delta t\  \mathbf{U}\left(\mathbf{x}^{\ast}, t_{n-1/2}\right)
$$

Here $\mathbf{x}_n$ is the particle position at step $n$; the starred quantity is
a throwaway half-step probe, used only to get a better wind estimate. The state
that survives is the position at step $n-1$, one step further back in time.

Both stages evaluate the wind at the **same** midpoint time
$t_{n-1/2} = t_n - \Delta t/2$, which is what makes this the midpoint rule
rather than a half-and-half hybrid. $\mathbf{U}$ comes from trilinear interpolation
(`grid_sample`) of the two bracketing meteorology hours, linearly weighted in
time by $\alpha = (t_{n-1/2} - t_{\mathrm{hour}}) / 3600$, where
$t_{\mathrm{hour}}$ is the start of the bracketing hour — so $\alpha$ runs from 0
at the start of that hour to 1 at the end.

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
u_{n+1}' = r\ u_n' + a_{\mathrm{drift}}\ \Delta t + \sigma\sqrt{1 - r^2}\ \eta,
\qquad r = e^{-\Delta t / T_L},\quad \eta \sim \mathcal{N}(0,1)
$$

with

- $u_n'$ — the fluctuation at step $n$; $u_{n+1}'$ is the same particle one step
  later. The formula is applied to each of $u'$, $v'$, $w'$ with that component's
  own $\sigma$ and $T_L$.
- $r$ — how much of the old fluctuation is retained over one step. It is exactly
  the autocorrelation at lag $\Delta t$: $r \to 1$ for a step much shorter than
  $T_L$ (perfect memory), $r \to 0$ for a step much longer (memory lost).
- $a_{\mathrm{drift}}$ — the drift $a$ from the equation above, minus the
  relaxation part that $r$ already accounts for. In GLIDE this is the well-mixed
  correction of §3.3, and it is zero for the horizontal components.
- $\sigma$ — the target standard deviation for this component, so
  $\sigma\sqrt{1-r^2}$ is exactly the amount of fresh randomness needed to keep
  the spread at $\sigma$.
- $\eta$ — a fresh standard normal draw (mean 0, variance 1) per particle per
  step.

Two properties are worth stating explicitly.

*It is unconditionally stationary.* Set $a_{\mathrm{drift}} = 0$: if
$\mathrm{Var}(u_n') = \sigma^2$ then
$\mathrm{Var}(u_{n+1}') = r^2\sigma^2 + \sigma^2(1-r^2) = \sigma^2$, for
**any** $\Delta t$. A naive Euler
discretisation of $\mathrm{d}u' = -u'/T_L\ \mathrm{d}t + b\ \mathrm{d}\xi$ loses
this and blows up once $\Delta t > 2T_L$. The accuracy limit on $\Delta t$ in
GLIDE therefore comes from the *drift* term and from position integration, not
from stability (§6).

*It has the right inertial-subrange limit.* As $\Delta t / T_L \to 0$,
$\sigma\sqrt{1-r^2} \to \sigma\sqrt{2\Delta t/T_L}$, i.e. $b\sqrt{\Delta t}$ with

$$
b^2 = \frac{2\sigma^2}{T_L}
$$

which is the standard Thomson (1987) diffusion coefficient. It is what ties the
model to real turbulence: in the inertial subrange the theory fixes
$b^2 = C_0\varepsilon$, with $C_0$ the Kolmogorov constant for the Lagrangian
structure function and $\varepsilon$ the rate at which turbulent kinetic energy
dissipates. Choosing $\sigma$ and $T_L$ is therefore equivalent to choosing
$C_0 \varepsilon$. $b$ is diagonal — GLIDE carries no cross-correlation between
$u$ and $w$, in common with FLEXPART and other regional models
(Stohl & Thomson 1999).

Two consequences are checked numerically. The **autocorrelation**, i.e. how
strongly a particle's vertical velocity now resembles its velocity a lag $\tau$
ago, should decay as $R(\tau) = e^{-\tau/T_L}$; that and the stationary variance
are verified by `test_ou_autocorrelation_and_stationarity`. And the **Taylor
dispersion curve** — the growth of $\sigma_z^2$, the variance of particle height
about the ensemble mean, as travel time $t$ increases — should follow
(`test_taylor_dispersion_curve_ballistic_to_diffusive`):

$$
\sigma_z^2(t) = 2\sigma_w^2 T_L\left[t - T_L\left(1 - e^{-t/T_L}\right)\right]
$$

which is $\sigma_w t$ at short times (**ballistic** — particles still fly
straight) crossing over to $\sqrt{2Kt}$ at long times (**diffusive** — memory
lost, ordinary random walk).

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
a_w \ =\  \underbrace{-\frac{w'}{T_{Lw}}}_{\text{relaxation}}
\ +\  \underbrace{\frac{1}{2}\left(1 + \frac{w'^2}{\sigma_w^2}\right)\frac{\partial \sigma_w^2}{\partial z}}_{\sigma\text{-gradient (Thomson 1987)}}
\ +\  \underbrace{\frac{\sigma_w^2}{\rho}\frac{\partial \rho}{\partial z}}_{\text{density (Stohl and Thomson 1999)}}
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
assembled $\sigma_w$ profile **on the particle's own side of the
boundary-layer top** $h$: below $h$ the boundary-layer profile (with the
optional surface layer), above it the free-troposphere closure. Within 1 m of
$h$ the stencil stops at $h$ and evaluates that side's profile there
(decision 0028). $\sigma_w$ jumps at $h$ (in a convective boundary layer from
about 0.7 to 0.05 m s⁻¹), and the drift must not difference the jump: a
stencil straddling it turns the jump into a drift of order
$(\sigma_{BL}^2 - \sigma_{FT}^2)/(2\delta)$ on the particles within $\delta$ of
$h$. Until decision 0028 the stencil did straddle it, and in a column started
uniform the 50 m above $h$ ended up holding from 0.4 to 11.5 times its share,
depending on the sub-step. The jump is handled instead by the interface rule
of §5. (The optional surface layer's seam at $0.1h$ is still differenced
across; it is off by default.)

**The density term** exists because the well-mixed condition is enforced in
Cartesian $z$, but the equilibrium distribution of air is $\propto \rho(z)$. Over
a deep boundary layer $\rho$ at the top can be 20% below the surface value.
Without the term, an initially $\rho$-weighted ensemble relaxes toward uniform,
biasing surface concentrations — and in a backward run that bias propagates
straight into the inferred flux. Stohl & Thomson measured +5.5% mean (range
+1–15%) on surface concentrations in their CAPTEX runs.

$\rho = p/(R_d T)$ and $\partial\rho/\partial z$ are built on the meteorology
grid once per window and sampled trilinearly per particle. $p$ is each
column's own pressure at each level and $T$ the local temperature, both at the
window midpoint, and $\partial\rho/\partial z$ is differenced on the column's
own geopotential heights, so the density is the density of the particle's own
column, as in FLEXPART (decision 0025). The column pressure is the one deep
convection uses (`HourlyMetTensors.pressure_hpa`): on model levels the
hydrostatic reconstruction from the surface pressure (decision 0010), on
pressure levels the level pressure, both regridded with the column onto the
terrain-following levels. Until decision 0025 the scheme used the met domain's
area-mean pressure of each terrain-following level instead; over terrain above
1,000 m that made $\rho$ too high by a median 24–31% (up to 56%) and the drift
term wrong by a median 3–12% (up to 24% at 40 m; measured in PR #61).

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
a^{b} \ =\  -a^{f} \ +\  b^2 \frac{\partial \ln g_a}{\partial w'}
\ =\  -a^{f} - \frac{2w'}{T_{Lw}}
$$

where $a^{f}$ and $a^{b}$ are the forward and backward drifts, and $g_a$ is the
density-weighted velocity PDF — the distribution of vertical velocities the air
itself holds at that height. The $\partial \ln g_a / \partial w'$ term is what
makes the reversal non-trivial: it knows about the *shape* of that distribution,
so the answer depends on the turbulence, not just on the sign of time.

For a symmetric Gaussian $g_a$ it evaluates to $-2w'/T_{Lw}$, as above.
Substituting the forward drift from §3.3, the $-w'/T_{Lw}$ relaxation comes back
**unchanged**, while the two inhomogeneity terms flip sign together:

$$
a^{b}_w \ =\  -\frac{w'}{T_{Lw}}
\ -\  \frac{1}{2}\left(1 + \frac{w'^2}{\sigma_w^2}\right)\frac{\partial \sigma_w^2}{\partial z}
\ -\  \frac{\sigma_w^2}{\rho}\frac{\partial \rho}{\partial z}
$$

In code this is exactly what happens: the forward drift is assembled and then
negated, with the relaxation supplied separately by the exact-OU factor $r$.

Getting this sign wrong is not a small error. It converts the $\sigma$-gradient
correction into a one-way upward pump; when it was wrong, the entire Mace Head
particle population lofted to ~2 km and stayed there.

The random forcing is symmetric, so the noise term needs no sign treatment. The
displacement of position simply runs the other way: $z_{n+1} = z_n - w'\Delta t$.

**Mass bookkeeping.** Each particle carries a weight $m_p = 1/N$, where $N$ is
the number of particles in that release — so one particle represents $1/N$ of the
released mass and the weights of a release sum to 1. It is set at release and
never modified by transport; convection redistributes particles between levels
but conserves their number and weight. Total weight is checked in the physics
tests as a conservation invariant.

**When a particle stops: by its own age.** Every particle is followed for
`simulation.length_seconds` back from its *own* release time, then stops. The
run has no single end time for all particles: a release made two hours after
another stops two hours (of model time) later, whatever else shares its batch.
In the code this is the per-particle `active_mask` in `main._run`: a particle is
active while the backward cursor lies between its release time and that time
minus the length; each batch's cursor runs from its latest release-window end
to its earliest release time minus the length. A particle counts for a step
when it is active at the step's upper (later) end, so its last step ends
exactly one length before its release, to within one step where the length is
not a whole number of steps. This is the rule the reference runs use
([reference-runs.md](reference-runs.md) §1: 240 h from each particle's own
release). Tests: `test_each_particle_stops_its_own_length_after_its_release` and
`test_every_particle_followed_for_full_simulation_length`
(`tests/test_main_runtime.py`).

---

## 5. Boundary conditions

The footprint is dominated by near-ground residence time, so the ground boundary
is where correctness matters most.

**Ground: smooth-wall reflection.** A particle that crosses $z = 0$ within a
sub-step is reflected by the *joint* mapping

$$
(z,\  w') \ \longmapsto\  (-z,\  -w')
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
z_{\mathrm{eval}} = \max(z,\  z_{\mathrm{ubl}}), \qquad z_{\mathrm{ubl}} = 2\ \mathrm{m\ (default)}
$$

$\sigma$, $T_L$, the drift, and the density gradient are all evaluated at
$z_{\mathrm{eval}}$. **The particle's position is not clamped** — only the
profile lookup. At 2 m the layer is thinner than any practical release altitude,
so it intercepts nothing but the post-reflection bounce. Note this is the
opposite of the "artificial unattainability" schemes that clamp particles to
$z = 0$ or restrict $\Delta t$ to prevent crossings; those always violate the
well-mixed condition, and GLIDE contains none of them.

**The boundary-layer top: an interface** (decision 0028). At $h$ the
turbulence switches from the boundary-layer profile to the free-troposphere
closure, and $\sigma_w$, $\sigma_{u,v}$ and $T_L$ jump. GLIDE treats $h$ as an
interface after Thomson, Physick & Maryon (1997). Inside every sub-step, after
the displacement and before the ground reflection, a particle whose
displacement crossed $h$ from side $A$ to side $B$ is

- **transmitted** with probability $\min(1, \sigma_{wB}/\sigma_{wA})$, with
  $w'$ scaled by $\sigma_{wB}/\sigma_{wA}$, $u'$ and $v'$ by their own ratios,
  and the part of the displacement beyond $h$ by $\sigma_{wB}/\sigma_{wA}$ (the
  rest of the sub-step flown at the new speed);
- otherwise **reflected**: $z \to 2h - z$, $w' \to -w'$.

$\sigma_A$ and $\sigma_B$ are the one-sided limits at $h$: the boundary-layer
profile at $h$ and the closure at $h$, evaluated once per outer step from the
particle's column. The transmission probabilities balance the flux through $h$
in each direction in a well-mixed state ($\alpha_{AB}\sigma_A = \alpha_{BA}\sigma_B$),
and the scaling maps the flux-weighted velocity distribution of one side onto
the other's, so a uniform column stays uniform for any sub-step when $\sigma$
is constant on each side; the derivation is in decision 0028. The rule depends
only on the direction of the displacement and on $|w'|$, so the backward model
(which is the forward model written for $-w'$, §4) uses it unchanged. Air
density is continuous at $h$, so TPM's density factor is 1. This is the
"alternative approach" of TPM's §3, the one of their two Gaussian schemes that
stays accurate at long steps; it assumes, as TPM state, that exchange across
$h$ is limited by diffusion on either side rather than by weaker turbulence
within the inversion itself. Decision 0028 compares every element of GLIDE's
implementation with the paper and records the reasons for each choice.

Between outer steps $h$ moves (time interpolation of the meteorology, advection
into another column) and a particle can find itself on the other side without
having crossed. It is neither transmitted nor reflected. Instead every
particle's $u'$, $v'$, $w'$ are scaled at the start of each step by $\sigma$ at
its position now over $\sigma$ at its position at the end of its last step,
with that step's meteorology (`sigma_last` in the scheme state). This is the
exact integral of the nonstationary term $(w'/2\sigma_w^2)\,\partial_t\sigma_w^2$
of the Gaussian well-mixed drift (Thomson 1987) over the change between steps;
it also covers changes of regime, $u_\ast$ or $w_\ast$ and convective
redistribution. In a steady column the ratio is 1 to within $10^{-4}$.

**Domain edges: drop and count.** A particle leaving the horizontal `met_domain`
bounding box, or rising above `met_domain.alt_max_m`, has no valid meteorology —
the coordinate normalisation would clamp it to the grid edge and it would be
pushed by edge winds for the rest of the run. Such particles have their liveness
bit cleared and are never advected or accumulated again. The count is reported in
`run_metadata.json`. There is no lower kill: $z = 0$ is handled by reflection.

**The top of the model.** `met_domain.alt_max_m` is the top of the vertical
grid (the AGL ladder ends there). A particle that rises above it — by the
resolved wind, turbulence or convection — is **removed**, not reflected and
not held below the top:
- within the step that takes it there, its meteorology is that of the top
  level (the coordinate normalisation clamps $z$ to the grid);
- the footprint is counted for that step at the height the particle reached,
  then the particle's liveness bit is cleared (`_within_met_domain`, applied in
  `main._run` after the gridder on both step paths: `alt <= alt_max_m` is
  inside), so it counts for nothing afterwards;
- its end record in `endpoint_particles.parquet` keeps that position, above the
  top, and it is counted in `escaped_this_step` and `escaped_met_domain_total`.

The convection scheme places particles within the grid's layers, so it takes a
particle above `alt_max_m` only when a configured list of levels
(`met_domain.vertical_levels`) ends above it; with the default ladder, removal
at the top comes from the resolved and turbulent vertical motion. Test:
`test_particle_reaching_the_model_top_is_removed` (`tests/test_main_runtime.py`).
The reference models differ: FLEXPART holds a particle just below its top level
(near 80 km), NAME removes it at its domain top
([reference-runs.md](reference-runs.md) §2, which requires that no particle be
removed below 20 km). GLIDE's shipped configurations set `alt_max_m` to
15,000 m, so a validation configuration must raise it to at least 20,000 m to
meet that rule.

---

## 6. Time stepping

The outer step $\Delta t$ is `simulation.dt_seconds` (60 s in the shipped
configs). Because the OU update is unconditionally stationary (§3.2), the
constraint on $\Delta t$ comes from two other places:

- the **forward-Euler drift increment**, whose error grows once
  $\Delta t \gtrsim T_L/5$. Wilson & Flesch (1993) derive an explicit spurious
  "bias velocity" $w_B$ for the near-surface layer, growing in proportion to
  $\Delta t/T_L$ — that is, the integration itself invents a slow drift that is
  not in the physics. Stohl & Thomson (1999) recommend the stricter
  $\Delta t \le 0.05\ T_L$.
- the **reflection bias**, whose magnitude also scales with $\Delta t/T_L$.

Near-surface $T_{Lw}$ falls to tens of seconds, so the turbulence is
**sub-stepped** within the outer step (advection is not). Every particle takes
the same number of equal sub-steps in every outer step (decision 0027):

$$
\Delta t_{\mathrm{sub}} = \frac{\Delta t}{n},
\qquad n = \texttt{turbulence.n\_substeps}\ (\text{default } 16)
$$

Inside the sub-step loop, the OU update, the displacement, and the ground
reflection all run at $\Delta t_{\mathrm{sub}}$, so a particle that crosses the
ground mid-step is reflected before the next sub-step rather than at the end of
the outer step. At the start of every sub-step after the first, $\sigma$, $T_L$,
$\partial\sigma_w^2/\partial z$ and the density term are re-evaluated at the
particle's current height (its horizontal position, and so the column, are those
of the start of the outer step), as in FLEXPART, and the
$(1 + w'^2/\sigma_w^2)$ factor uses the current $w'$.

**Why one fixed count, re-evaluated every sub-step.** Both parts are needed for
the well-mixed condition at the shipped $\Delta t = 60$ s. Measured by the
occupancy of the lowest 5% of a horizontally uniform boundary layer relative to
its well-mixed share (WM investigation, `outputs/agents/WM/`, 2026-11-25):

- **Coefficients held at their outer-step values** (GLIDE until decision 0027)
  make the sub-steps a random walk with the diffusivity $\sigma_w^2 T_{Lw}$ of
  the *starting* height and no compensating drift, which collects particles
  where $T_{Lw}$ is short — at the ground. Neutral boundary layer: 1.67 at
  $\Delta t = 60$ s, getting *worse* as the sub-steps shrank, against 1.03 at
  $\Delta t = 15$ s. Re-evaluating $T_L$ alone brought it to 1.33.
- **A sub-step length that depends on the particle's height** (each particle
  its own count $\lceil \Delta t/(c\,T_{Lw,p}) \rceil$ from its starting
  $T_{Lw}$, as GLIDE did, or a count re-chosen every sub-step, as FLEXPART does)
  gives 1.31 even with the coefficients re-evaluated, falling only in
  proportion to $c$. A count that only *adds* sub-steps where
  $T_{Lw}$ is short biases too: with
  $k_p = \max(n, \lceil \Delta t/(0.125\,T_{Lw,p}) \rceil)$ (16 at the 30 s
  floor) the neutral lowest 5% is 1.090 for $n = 4$ and 1.035 for $n = 8$, above
  both the fixed count it starts from (1.038, 1.018) and the fixed $n = 16$ it
  refines to near the ground (1.006); same-seed differences from the fixed
  count +0.052 ± 0.006 and +0.017 ± 0.007 (5 seeds; convective +0.136 and
  +0.033, stable −0.044 and −0.001, where under-resolution at the ground
  dominates).
- A count shared by all particles but taken as the batch maximum (the WM
  prototype) fixed the bias, but made a release's footprint depend on which
  other releases share its batch; a fixed count is deterministic per release,
  needs no data-dependent loop bound, and is what the CUDA-graph path runs
  anyway (decision 0004).

**Choosing $n$.** The target is the well-mixed share within ±0.05 in the
lowest 5% of the boundary layer and in the 0–40 m layer where footprints are
counted, after particle noise. Measured on B7's three regimes with the
boundary-layer top a reflecting lid applied inside every sub-step (the lid
replaces the interface of §5 there), $\Delta t = 60$ s, 20,000 particles,
5 seeds, records every 15 min from 2 to 6 h
(`scripts/validation/wm_near_surface/interface_study.py`, harness `closed`;
`outputs/agents/WM/interface/study/tables.md`, code of decision 0028). Mean over
seeds; standard errors 0.002–0.008, seed standard deviation 0.002–0.017:

| | stable ($h$ 400 m, $H$ −20 W m⁻²) | neutral ($h$ 1,500 m, $u_\ast$ 0.4 m s⁻¹) | convective ($h$ 1,500 m, $H$ +200 W m⁻²) |
| --- | --- | --- | --- |
| before decision 0027 (cap 6) | 1.787 / 1.750 | 1.660 / 1.788 | 0.965 / 0.905 |
| $n = 4$ (15 s) | 1.160 / 1.145 | 1.024 / 1.028 | 0.850 / 0.787 |
| $n = 8$ (7.5 s) | 1.078 / 1.068 | 1.011 / 1.009 | 0.938 / 0.910 |
| **$n = 16$ (3.75 s)** | **1.039 / 1.040** | **1.004 / 1.007** | **0.971 / 0.964** |
| $n = 32$ (1.875 s) | 1.021 / 1.022 | 1.006 / 1.009 | 0.987 / 0.983 |

(lowest 5% / 0–40 m.) $n = 16$ is the smallest count that meets the target in
every regime: $n = 8$ leaves the stable lowest 5% at +0.078 ± 0.005 and the
convective 0–40 m at −0.090 ± 0.007. The error is first order in
$\Delta t_{\mathrm{sub}}$ (it roughly halves from $n = 16$ to 32), positive where
$\sigma_w$ falls with height (stable) and negative where it rises near the
ground (convective), as Wilson & Flesch (1993) predict. Decision 0027 first
reported smaller convective errors (0.994 / 0.987 at $n = 16$): those runs used
the drift stencil that straddled $h$, which in this closed column acts at the
lid. With only the stencil reverted to the straddling one, this code gives
0.994 / 0.987 (convective, $n = 16$) and 0.971 / 0.940 ($n = 8$) again (5 seeds,
`study/closed_nostencil.jsonl`), so the stencil alone accounts for the change.
That it does so by adding a spurious drift at the lid that depletes the top of
the column, and with it raises every other level, is argued, not measured.

**The boundary-layer top** (decision 0028; harnesses `open2h`, `openbl`,
`hvar`, same column, above $h$ a stably stratified free troposphere,
$N$ = 0.015 s⁻¹, no wind, so $K_z$ is the closure's 0.1 m² s⁻¹ floor):

- *Uniform in $[0, 2h]$, lid at $2h$.* The exact answer is uniform. With the
  interface, each of the bands $h - 200$ … $h - 50$, $h - 50$ … $h$,
  $h$ … $h + 50$ and $h + 50$ … $h + 200$ m holds 0.94–1.03 of its share at
  $n$ = 8, 16 and 32 in all three regimes (5 seeds; the stable
  residual below and just above $h$, −0.056, −0.033, −0.016 in $h - 50$ … $h$
  at $n$ = 8, 16, 32, falls with the sub-step like the near-surface error, and
  the 10 m just above $h$ follow the concentration just below, as a consistent
  interface should). Before (PR #69 head) the 50 m above $h$ held 11.5, 9.0,
  6.0, 1.8 and 0.44 times its share in the convective column at $n$ = 4, 8,
  16, 32 and 64 (trapped above $h$ at long sub-steps, pumped down at short
  ones), 3.3 to 0.39 in the neutral column, 0.96–1.12 in the stable one; main
  was worse (convective 9.9, neutral 4.6, stable 0.53).
- *Uniform in $[0, h]$, open top.* After 2–6 h, 0.974–0.976 of the particles
  are still in a convective or neutral boundary layer at every $n$ from 4 to
  32, within 0.003 of the diffusion limit (0.973, from
  $\partial_t c = \partial_z(K\partial_z c)$ with the scheme's own
  $K = \sigma_w^2 T_{Lw}$, `interface_diffusion_limit.py`). The stable boundary
  layer keeps 0.992–0.995, more than the diffusion limit's 0.93, which does not
  apply there ($T_{Lw}$ grows without bound as $\sigma_w \to 0$ at $h$).
- *$h$ moving* (a triangle wave between 0.75 and 1.25 $h_0$, 2-h period).
  Without carrying $u'/\sigma$ across steps the 10 m below $h$ held 1.20
  (neutral), 1.15 (convective) and 0.87 (stable) of their share; with it, all
  bands are within 0.016 of 1 (3 seeds, $n = 16$).

The PR #69 code's own "fine-step reference" for the open top (all particles
still in the boundary layer at a 0.9375 s step) was therefore biased, and its
10 m finite-difference probe reversed the bias rather than removing it (in
the uniform column the 50 m above $h$ then held 0.33 of its share, 3 seeds).

**The basal clamp.** Below $z_{\mathrm{ubl}}$ (§5) every coefficient is
sampled at $z_{\mathrm{ubl}}$, so the profile the particle sees is flat there.
Below $z_{\mathrm{ubl}}$ the finite-difference
stencil, clamped to $[z_{\mathrm{ubl}}, z_{\mathrm{ubl}} + \delta]$, nevertheless
returns the one-sided gradient just above the clamp, a drift where the profile is
flat. `HannaScheme(basal_zero_gradient=True)` sets $\partial\sigma_w^2/\partial z$
to zero strictly below the clamp, leaving the density drift unchanged
(`test_basal_zero_gradient_only_below_the_clamp`). Same-seed differences from the
default in the closed column, at $n = 8$ and at $\Delta t$ = 1.875 s, are at most
0.003 in the lowest 5% and the 0–40 m layer in all three regimes (5 seeds; in
0–2 m, within ±0.025 and its noise), so the default keeps the stencil as it was
and the switch stays for comparisons.

**Sub-steps long compared with $T_{Lw}$.** In the boundary layer the FLEXPART
floor $T_{Lw} \ge 30$ s bounds $\Delta t_{\mathrm{sub}}/T_{Lw}$ by
$\Delta t/(30\,n)$, 0.125 at the defaults. Above the boundary layer there is no
floor: the free-troposphere closure gives $T_{Lw} = 0.5/N$, a few seconds in
strong inversions. Every sub-step of every active particle is checked against
$\Delta t_{\mathrm{sub}} > 0.5\,T_{Lw}$ and $\Delta t_{\mathrm{sub}} > T_{Lw}$,
with $T_{Lw}$ at the sub-step's own height, and counted as boundary layer
($z \le h$) or free troposphere. On the CUDA-graph path the counts are taken
inside the captured step and kept on the device (no host synchronisation); the
runtime reads them once per met window and `run_metadata.json` records the run
totals under `runtime.substep_exceedance` (sub-steps, exceedances of each
threshold and their fractions, the worst ratio, per layer, and the number of met
windows with an exceedance). The integration does not change where the ratio is
large: the OU velocity update stays exact, but the displacement
$w'\,\Delta t_{\mathrm{sub}}$ over-states the variance of the exact integrated
OU displacement, $2\sigma_w^2T_L^2(r - 1 + e^{-r})$ with
$r = \Delta t_{\mathrm{sub}}/T_L$, by the factor $r^2/[2(r-1+e^{-r})]$ (1.59 at
$r = 1.58$). On the reference multi-site run
(`configs/ab_multisite_perf.yaml`, January, 144 releases; GH200, 25 Nov 2026)
no boundary-layer sub-step exceeded $0.5\,T_{Lw}$ at $n$ = 4, 8 or 16 (worst
ratio $60/(30n)$, as the floor implies), and in the free troposphere 11.6%,
0.0039% and 0.00005% of sub-steps did at $n$ = 4, 8 and 16 (worst ratios 2.28,
1.16 and 0.83; $\Delta t_{\mathrm{sub}} > T_{Lw}$ in 0.0027%, 0.000012% and
none; code of decision 0027 alone, one seed). With decision 0028 (3 seeds) the
free troposphere holds 52.7% of the sub-steps, and 0.052% (n = 8) and
0.00026% (n = 16) of them exceed $0.5\,T_{Lw}$ (worst ratios 1.65–1.68 and
0.81–0.83; $\Delta t_{\mathrm{sub}} > T_{Lw}$ in 0.00026% and none); still no
boundary-layer sub-step does. The boundary-layer half of the counter is
structurally zero at the defaults: with $T_{Lw} \ge 30$ s it needs
$\Delta t/n > 15$ s. It is kept for runs without the floors, with a longer
step or fewer sub-steps. This count replaced task A1's count of
particle-steps that reached the old cap of 6 (2.6% on the same run, worst ratio
1.58), which could not say how many exceeded the target.

**Cost.** The re-evaluation is one full profile evaluation, two
$\sigma_w$-only evaluations for the finite-difference gradient (bit-identical to
taking $\sigma_w$ from the full profile,
`test_sigma_w_probe_is_bit_identical_to_the_full_profile`) and one density
lookup per sub-step. On the same run, GPU-synchronised phase
timers, the step took 4.49 ms (code before this decision, 6 iterations without
re-evaluation), 5.18, 9.66 and 11.72 ms at $n$ = 4, 8 and 16 with decision
0027 alone. The boundary-layer-top interface of decision 0028 (two σ
evaluations per outer step, a uniform random number and a few element-wise
operations per sub-step) adds 1.6 ms at $n = 16$: 11.92 against 13.50 ms run
back to back on a quiet node, and 13.29–13.50 ms over three runs (8.67–8.69 ms
at $n = 8$; `outputs/agents/WM/interface/bench/timing_table.md`). At $n = 16$
the step costs about 155 s more over the run than before decision 0027, about
13% of its wall time (1,233–1,340 s, against 1,157 s), and about 80 s more
than $n = 8$. Reading meteorology still dominates (52–62% of wall time).

**Effect on footprints.** The reference run (`configs/ab_multisite_perf.yaml`,
144 releases at three sites, 5 days back) against its golden fixture, made with
the code before decision 0027 (median over releases; 3 seeds, seed standard
deviation in brackets; `outputs/agents/WM/interface/bench/`): with decisions
0027 and 0028 at $n = 16$ the 0–40 m footprint is +47.9% (0.2%), the
40–1,000 m layer +119% (0.9%), the 1–5 km layer −31.0% (0.05%) and the total
−4.2% (0.02%); by site, 0–40 m +44.7% (Mace Head), +55.9% (Ridge Hill) and
+45.4% (Tacolneston), +34.6% to +77.2% across their day and night releases. At
$n = 8$ the same seed gives a 0–40 m footprint within 0.4% (median; at most
1.2% in any site's day or night releases), and the share of sub-steps spent
above the boundary layer is 52.6% against 52.7%: real footprints are converged
in $n$ between 8 and 16. With decision 0027 alone they were not (0–40 m −16%
and +15% at $n$ = 8 and 16, 72% and 61% of sub-steps above the boundary layer),
because the boundary-layer top was not well mixed. The smoke run (Mace Head,
one release, one day, 3 seeds) changes by +92.0%, +94.4% and +95.8% in 0–40 m
at $n$ = 8, 16 and 32 (seed standard deviation 0.4–0.6%) and −6.6% in total.

**Rogue-trajectory clip.** At the end of each sub-step's velocity update,

$$
\lvert u_i' \rvert \le 4\ \sigma_i
$$

A true Gaussian exceeds $4\sigma$ with probability $\sim 6\times10^{-5}$, so this
is a safety net, not a physics change. It exists because where a sub-step is
long compared with $T_{Lw}$ *and* $\sigma_w$ sits near its numerical floor — the stable boundary-layer
top, where $\sigma_w \to 0$ — the $(1 + w'^2/\sigma_w^2)$ factor can snowball the
drift into a NaN. FLEXPART has the equivalent clip.

---

## 7. Footprint accumulation

At every step, every live particle inside the output grid adds

$$
\Delta f = \frac{m_p \  \Delta t}{\rho_d}
$$

to the cell it occupies, in its own time-ago bin, in its own release's slice,
where $\rho_d$ is the **dry-air density at the particle** as it is counted
(`output_grid.density_weighting: dry_air`, the default; decision 0022). With
`density_weighting: none` the density is left out and $\Delta f = m_p\,\Delta t$.
The grid is 5-dimensional:

$$
f\big[\ \text{release},\ \text{time ago},\ z,\ \text{lat},\ \text{lon}\ \big]
$$

Raw units are **s m³ kg⁻¹** with `dry_air` (residence time per unit released
mass, divided by the dry-air density) and **seconds** with `none` (weight is
dimensionless and sums to 1 per release). The footprint store declares which, in
the attributes `density_weighting` and `units` of the dataset and of its
`footprint` variable, and `run_metadata.json` repeats them under
`footprint_units`. The vertical bins are set
by `output_grid.z_edges_m`, an arbitrary strictly-ascending edge list — e.g.
`[0, 40, 1000, 5000]` gives a 0–40 m surface bin matching the FLEXPART/NAME
convention, a mixed-layer bin, and a free-troposphere bin. Accumulating *directly*
into a bin that matches the reference's surface layer makes the downstream unit
conversion exact rather than a depth-weighted approximation.

Time-ago bins are `output_grid.time_bin_seconds` wide (default 3,600 s, one
hour; decision 0016) and are measured from each particle's **own** release
window end, so a multi-release batch produces correctly-aged footprints for
every release from one sweep, and bin *k* of a release covers the same absolute
interval for all its particles. Every particle is followed for
`simulation.length_seconds` from its own release time, so one released at the
start of its window reaches an age of the length plus the window's duration.
`n_time_bins` counts the bins of travel to resolve, and the `time_ago` axis adds
one bin per started bin width of the longest release window, so that travel
stays on the axis. `time_ago_start_hours` and `time_ago_end_hours` give each
bin's edges. With `n_time_bins: 1` all ages collapse into one bin, giving the
time-integrated footprint the FLEXPART comparison uses. With more bins,
`n_time_bins × time_bin_seconds` must cover the simulation length; a
configuration that does not is refused unless it sets
`output_grid.drop_older_ages: true`, and then ages past the last bin are
dropped rather than clamped, so the axis labels stay honest.

Particles outside the grid horizontally, vertically, in time-ago, or in release
index contribute nothing.

**Where in the step a particle is counted.** Each step's $\Delta f$ stands for
the time integral of the cell indicator over the step,
$\int 1[\mathbf{x}(t)\in\text{cell}]\,m_p/\rho_d\,dt$, which the gridder
approximates from the positions it has. `output_grid.deposition.position`
(task A2; decisions 0026 and 0029) chooses how:

- `endpoint` (the default): all of $\Delta t$ at the particle's position at the
  end of the step, after advection, turbulence, reflection and convection, with
  $\rho_d$ there at the step's mid-time: a right-rectangle rule in time. A
  particle's first step is counted where it ends, never at its release point.
- `step_mean`: all of $\Delta t$ at the mean of the particle's start and end
  positions, $\tfrac12(\mathbf{x}_n + \mathbf{x}_{n-1})$, with $\rho_d$ there at
  the step's mid-time (decision 0026). Longitude is averaged along the shorter
  arc (the difference taken into [−180°, 180°)); the runtime does not wrap
  particle longitudes today, so this only guards future periodic domains.
- `trapezoid`: $\Delta t/2$ at the start position and $\Delta t/2$ at the end
  position, as two deposits (decision 0029). Each half has its own time-ago bin
  (the start half the age at the start of the step, the end half the age half a
  step later, just as `endpoint` bins the whole step by its starting age; with
  steps aligned to the bins both halves fall in the step's bin) and its own
  $\rho_d$, at its own position and time (the start at $t$, the end at
  $t-\Delta t$, in the step's met window). A particle's first step is counted
  half at its release point; a particle that leaves the met domain keeps the
  start half of its last step, and the end half is dropped, like `endpoint`'s,
  if it lies outside the output grid.

For `step_mean` and `trapezoid` the runtime copies the start positions before the
step, outside the captured graph, so the static (CUDA) step's graph, shapes and
outputs are the same for all three (decision 0004). Summed over consecutive
steps, `trapezoid` is `endpoint` plus $\Delta t/2$ at each particle's first
position minus $\Delta t/2$ at its last, because each interior position is the
end of one step and the start of the next (*derived*); the only other
differences are that each half carries the density at its own time and, where a
step straddles a time-ago edge, its own bin. In space, then, `endpoint` and
`trapezoid` sample the same interior positions with the same weights; what
`endpoint` lacks is the first half step at the release point. Kernel deposition
(decision 0012) is deferred to after v1; deposition is a delta at the chosen
position(s).

**Scored against fine-time residence** (decision 0029;
`scripts/validation/score_residence_counting.py`, outputs in
`$GLIDE_AGENT_OUTPUTS/agents/A2/counting/`). The scorer measures each rule's
counting error apart from the trajectory error. It records the production
step's sub-step positions (60 s outer step, 16 sub-steps of 3.75 s), rebuilds
each particle's path at the sub-step times (advection and meander spread
uniformly over the step), and takes as reference the midpoint rule on that path
with 8 samples per sub-step (0.47 s), each sample binned by its own cell and age
and divided by its own $\rho_d$. The three rules are counted from the same
trajectories. The reference is converged: 1, 2 and 4 samples per sub-step differ
from 8 by ≤ 0.025% in every region of the smoke run and the synthetic columns, and the composite trapezoid on the sub-step positions by ≤ 0.1%.
The offline tallies of each rule match production runs of that rule, same seed,
to the float32 accumulation of the production grid (exactly, with a float64
grid), and the hooked run is bit-identical to an unhooked one.

Error against the reference on the smoke run (Mace Head, 10 m inlet, 1 day, CPU,
three seeds, 0–40 m unless stated, mean over seeds; the s.d. over seeds is
≤ 0.27 percentage points; `tables/scores_smoke_w.md`):

| rule | receptor cell | ring 1 | rings 2–3 | rings 4–10 | beyond 10 | 0–40 m total | 40–1000 m | first hour | cell-wise L1 |
|---|---|---|---|---|---|---|---|---|---|
| `endpoint` | −10.77% | +0.17% | −0.12% | +0.03% | +0.07% | −1.25% | +0.04% | −7.59% | 1.62% |
| `step_mean` | +0.74% | −2.87% | −3.85% | −4.91% | −4.02% | −3.46% | +0.21% | +0.09% | 3.64% |
| `trapezoid` | −0.44% | +0.17% | −0.12% | +0.03% | +0.02% | −0.03% | +0.00% | −0.34% | 0.35% |

(Rings are Chebyshev distances in cells from the receptor's cell; "first hour" is
the first hourly time-ago bin; the L1 error is Σ|rule − reference| over the
0–40 m cells divided by the reference's 0–40 m total.) The same release at 12 UTC
(daytime; `tables/scores_smokeday_w.md`, three seeds): `endpoint` −7.69% in the
receptor's cell and −0.60% in total, `step_mean` +0.59% and −0.1 to −1.4% in the
rings, `trapezoid` −0.28% and within ±0.06% elsewhere (L1 0.88 / 0.80 / 0.28%).

- `endpoint` misses about one step of release residence in the receptor's cell
  and is unbiased elsewhere. The right-rectangle rule on the sub-step positions
  has a receptor-cell error proportional to its spacing (−0.66, −1.32, −2.6,
  −5.2% at 3.75, 7.5, 15 and 30 s), which locates the error at the first step.
- `step_mean` gets the first step about right but undercounts the surface layer
  everywhere else: the mean of two heights a layer-depth apart lies above the
  layer the particle spent its time in. In horizontally uniform, equilibrated
  Hanna columns at Δt = 60 s (no receptor, so nothing but this mechanism) it
  undercounts 0–40 m by 14.6% ($u_*$ = 1 m s⁻¹, $h$ = 300 m and 1,000 m), 6.4%
  (convective, $h$ = 1,500 m) and 0.9% ($u_*$ = 0.4 m s⁻¹), while `endpoint`
  and `trapezoid` are within 0.1%.
- `trapezoid` is within 0.5% of the reference in every region of the smoke run
  and of the synthetic plumes from a 10 m release (neutral, convective, stable;
  25 km cells), with cell-wise L1 errors of 0.05–0.35%.
- Where cells are only a few steps wide, single crossings show: with 1 km cells
  and a 3 m s⁻¹ wind (5.5 steps per cell) `trapezoid` is −2.0% in the receptor's
  cell and +1.9% in the next ring, `step_mean` +3.6% and −0.9%, `endpoint`
  −23% and +1.9%; the 0–40 m totals of `trapezoid` and `step_mean` are within
  0.01%, and the L1 errors are 0.72%, 0.94% and 3.25%.
- The analytic plume (`tests/test_plume_footprint.py`: homogeneous OU, $u' = 0$,
  release at 50 m, 500 m cells = 100 s of travel; run here with 16 OU sub-steps
  per outer step) cannot rank the rules. Its release is above the 0–40 m layer,
  so `trapezoid` and `endpoint` count the same 0–40 m residence (*derived* above;
  *measured* equal). With deterministic along-wind motion, the cell indicator is
  sampled at fixed phases, so the column errors are aliasing, and which of
  `endpoint` and `step_mean` wins depends on where the cell edges fall: near the
  receptor (< 2 $T_L$), mean relative column error against the reference,
  `endpoint` / `step_mean`, three seeds, is 2.0% / 0.0% at Δt = 5 s with the
  test's geometry (edges on step times) and 0.4% / 2.0% with the receptor 0.63
  of a cell from the edge; 8.4% / 0.8% and 7.6% / 4.0% at 20 s; 21% / 38% and
  41% / 21% at 60 s, where every rule is 26% off in the far field (one or two
  samples per 100 s column). The reference itself is within 1–2% of the closed
  form at every Δt (`tables/scores_synth_u.md`). The A2 acceptance test ("`step_mean`
  no worse at Δt, better at 4Δt") passes partly because of that alignment: with
  the edges moved, `step_mean` is still better at 20 s but worse at 5 s.

Against `endpoint` on production runs (same seeds, so the changes are paired;
`tables/production_*.md`): on the smoke run `trapezoid` adds +11.66% (s.d. over
seeds 0.03) to the receptor's 0–40 m cell and +1.25% to the 0–40 m total, and
changes every ring by ≤ 0.05%; `step_mean` gives +12.9% in the cell, −3.0 to
−4.9% in the rings and −2.24% in total. On the reference configuration (`configs/ab_multisite_perf.yaml`, GPU, 144
releases from Mace Head 10 m, Ridge Hill 90 m and Tacolneston 185 m, 5 days back,
seeds 42/43/44; `tables/production_ref.md`), `trapezoid` changes only the 10 m
inlet: Mace Head +7.49% in the receptor's cell and +0.37% on the 0–40 m total
(day +0.26%, night +0.45%), every ring ≤ 0.05%; at the two elevated inlets the
release half lies above the 0–40 m layer and the change is ≤ 0.04% everywhere.
`step_mean` lowers the 0–40 m total at every site (−2.0% Mace Head, −3.2% Ridge
Hill, −3.5% Tacolneston), most near elevated inlets at night (Ridge Hill rings
2–3 −16%, Tacolneston receptor cell −13%). The s.d. over seeds of these paired
changes is ≤ 0.01 points for `trapezoid` and ≤ 0.08 for `step_mean` except its receptor cells (≤ 2.1);
for scale, endpoint's own 0–40 m total varies between seeds by 0.26–0.44% per
release (median).

**The dry-air density at the particle.** $\rho_d = (p - e)/(R_d T)$ with
$R_d = 287.05$ J kg⁻¹ K⁻¹ and the vapour pressure $e = q p / (0.622 + 0.378 q)$
from the specific humidity $q$. $T$, $q$ and the surface pressure $p_s$ are
interpolated to the particle's position (trilinearly, with the same vertical
lookup as the wind) in the step's met window and with the step's time weight,
as the step itself uses them. The met window holds no 3-D pressure, so the
pressure at the particle's height $z$ above the ground is the surface pressure
carried up hydrostatically,

$$
p(z) = p_s \exp\!\left(-\frac{g\,z}{R_d\,\bar T_v}\right), \qquad T_v = T\,(1 + 0.6077\,q),
$$

with the layer-mean virtual temperature $\bar T_v$ taken at $z/2$. This is the
hypsometric relation GLIDE already uses to reconstruct model-level pressure
from the surface pressure (decision 0010), applied from the ground to the
particle; the midpoint rule's error is below 10⁻⁷ in $p$ over the 40 m
layer. The fields move to the device once per met window; each step is tensor
operations on the whole particle buffer, with no host synchronisation
(decision 0004). Dividing while counting captures the pairing of residence and
density (more time near the ground at night, in denser air) that dividing the
summed residence by one density per cell misses (decision 0022).

**Conversion to STILT units.** For a `dry_air` footprint,

$$
f_{\mathrm{STILT}}(y,x) \ =\  \frac{m_{\mathrm{air}}}{h_{\mathrm{sfc}}} \sum_{t}\sum_{z \in \text{surface layer}} f\big[t,z,y,x\big]
$$

giving $\mathrm{m^2\ s\ mol^{-1}}$, equivalently
$(\mathrm{mol/mol})/(\mathrm{mol\ m^{-2}\ s^{-1}})$, for the dry-air mole
fraction (`docs/reference-runs.md` §6). Here $h_{\mathrm{sfc}}$ is the depth of
the surface layer being integrated over — 40 m by convention, and **not** the
boundary-layer depth $h$ of §2 — and $m_{\mathrm{air}} = 0.02897\ \mathrm{kg\ mol^{-1}}$
is the molar mass of dry air. For a `none` footprint (plain seconds) the density
is applied afterwards, as in Lin et al. (2003) Eq. 5, dividing by a surface air
density $\bar\rho$ (a scalar, or a 2-D field from `surface_air_density_from_met`, on pressure- or model-level stores);
that is biased where residence and density vary together (decision 0022).
`to_stilt_surface_footprint` reads the footprint's `density_weighting`
attribute, applies the matching formula, and refuses a density for a `dry_air`
footprint, which would divide twice. Bins that only partially overlap the chosen
surface layer are credited by their overlap fraction.

The whole chain — advection, turbulence, reflection, gridding, and the STILT
conversion — is validated end-to-end against an analytic reflected-Gaussian plume
in `tests/test_plume_footprint.py`, matching absolute magnitude to within 1%.

---

## 8. Known approximations

Stated plainly, because they bound what the model can be trusted for.

| Approximation | Consequence |
| --- | --- |
| Gaussian vertical velocity PDF everywhere — no skewed convective-boundary-layer PDF | Under-represents the updraft/downdraft asymmetry of a strongly convective BL. Thomson (1987, §5.2) found ground-level concentration relatively insensitive to skewness even where dispersion aloft is sensitive, so this is defensible for a surface-flux footprint model — but it is a modelling choice, not a free lunch. |
| A fixed sub-step count, $n = 16$ at $\Delta t = 60$ s (3.75 s) | Near-surface well-mixed error, closed boundary layer: +3.9% / +4.0% (stable, lowest 5% / 0–40 m), +0.4% / +0.7% (neutral), −2.9% / −3.6% (convective), particle noise about ±1% (§6); the same first-order error leaves the top 50 m of a stable boundary layer 3% short. In the free troposphere $T_{Lw}$ can fall below the sub-step; how often is recorded per run (`runtime.substep_exceedance`). |
| The boundary-layer top is a jump of $\sigma_w$ and $T_{Lw}$ to the free-troposphere closure, treated as a Thomson–Physick–Maryon interface (§5, decision 0028) | Well-mixed-consistent for any sub-step if $\sigma$ were constant on each side; the interface does not smooth the jump, which is a property of the two parameterisations, not of the atmosphere. $h$ is fixed per particle within an outer step. Horizontal velocities are scaled on transmission like $w'$. The optional surface layer's seam at $0.1h$ is not treated as an interface. |
| Turbulence support fields ($\rho$, free-troposphere $\sigma/T_L$, meander $\sigma$) built once per meteorology hour at the window midpoint | A met-cadence approximation (<1%/hr drift), matching how FLEXPART-class models refresh turbulence fields. Per-particle interpolation through those fields is unchanged. |
| Deep convection assigns each particle to the nearest meteorology column | Each (lon, lat) column has its own parcel lift, trigger and matrix (task A02), fired once per meteorology interval read from the store. A particle near a column boundary takes the nearest column's matrix rather than a blend. See [convection.md](convection.md) §1 and §7. |
| Each step's residence is counted from at most two positions: the step's end (`output_grid.deposition.position: endpoint`, the default), the mean of its start and end (`step_mean`), or half at each end (`trapezoid`; decision 0029) | The cell indicator is sampled at 60 s, so a particle crossing a cell or bin edge during a step is credited by where its sample points fall, not by when it crossed. Scored against fine-time residence on the same trajectories (§7): `endpoint` misses the first half step at the release point (−10.8% in the receptor's 0–40 m cell and −1.25% on the 0–40 m total of the smoke run) and is otherwise within ±0.2%; `step_mean` undercounts the 0–40 m layer wherever a step's vertical displacement is comparable to it (−3 to −5% outside the receptor's cell on the smoke run, up to −15% in strong turbulence); `trapezoid` is within 0.5% in every region scored. Where cells are only a few steps wide, single crossings leave errors of a few per cent per cell for every rule (±2% with 1 km cells at 3 m s⁻¹), and with deterministic along-wind motion (the analytic plume) they alias. |
| Initial $u', v', w' = 0$ rather than sampled from the local $\sigma$ | Particles equilibrate within one $T_L$ (typically ~100 s), which is shorter than the release window. |
| Virtual temperature approximated as $T$ in the Obukhov length | Humidity correction is small; deferred. |
| Free-troposphere horizontal turbulence treated as isotropic with the vertical | The unresolved-mesoscale horizontal spread is carried separately by the meander process. |
| Meander process has no drift correction for its spatially varying $\sigma_m$ | Particles collect slowly where $\sigma_m$ is small: after 10 days, 0.8–1.9% systematic unevenness in 0.25° cells and 0.35–0.75% in 1° cells, below particle noise at validation particle counts (task A03; [turbulence.md §7](turbulence.md#7-meander-unresolved-mesoscale-motion)). Left uncorrected in v1 (decision of 24 Nov 2026). |

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
- Thomson, D. J., Physick, W. L., Maryon, R. H. (1997). Treatment of interfaces
  in random walk dispersion models. *J. Appl. Meteor.* 36, 1284–1295. (GLIDE
  uses the §3 "alternative approach", with §4's treatment of jumps of $h$ in
  time; checked against the paper, decision 0028.)
- Wilson, J. D., Flesch, T. K. (1993). Flow boundaries in random-flight
  dispersion models: enforcing the well-mixed condition. *J. Appl. Meteor.* 32,
  1695–1707.
- Wilson, J. D., Legg, B. J., Thomson, D. J. (1983). Calculation of particle
  trajectories in the presence of a gradient in turbulent-velocity variance.
  *Boundary-Layer Meteorol.* 27, 163–169.
