# 0032 — Hanna turbulence: horizontal turbulence along and across the mean wind (proposed)

> **The default is Matt's decision.** This record proposes `horizontal_frame:
> wind` as the default. The code default is still `grid`, which leaves every run
> bit-identical to GLIDE before this record. Nothing changes until the default is
> switched or a configuration sets `wind`.

**Context.** Hanna (1982) gives the horizontal turbulence as an **along-wind**
component ($\sigma_u$, $T_{Lu}$) and a **cross-wind** component ($\sigma_v$,
$T_{Lv}$). In stable conditions both differ: $\sigma_u = 2.0u_*(1-\zeta)$ against
$\sigma_v = 1.3u_*(1-\zeta)$, and $T_{Lu} = 0.15h\sqrt\zeta/\sigma_u$ against
$T_{Lv} = 0.07h\sqrt\zeta/\sigma_v$, so $K_u/K_v = \sigma_u^2T_{Lu}/\sigma_v^2T_{Lv}
\approx 3.3$. In neutral conditions $\sigma_u \neq \sigma_v$ and $T_{Lu} = T_{Lv}$.
In convective conditions and the free troposphere the two components are the
same. GLIDE applied $u'$ to longitude and $v'$ to latitude
(`HannaScheme._integrate_vertical_substeps` → `GPUEngine.apply_horizontal_turbulence`),
so a stable plume's spread depended on the wind direction relative to the grid.
An external review raised it ("GLIDE turbulence physics discussion notes",
"Horizontal spread and receptor setup"). Measured
(`tests/test_horizontal_frame.py`, the review's experiment: uniform stable
meteorology, a point source at 100 m, 10 min, meander off, 20,000 particles), the
ratio of along-wind to cross-wind plume width was 1.745 with the wind towards
east, 1.004 towards north-east and 0.573 towards north.

**What FLEXPART does.** FLEXPART rotates. Read in v11 (local source,
`flexversion_major = '11'`, commit d4e0a868) and v10.4 (gitlab.phaidra.org
`flexpart/flexpart` at ref `v10.4`, `src/advance.f90`, `src/windalign.f90`;
same code, no line numbers verified there):

- `turbulence_mod.f90` 66–76 (v11): the OU updates of `turbvel%u` with `sigu`,
  `tlu` and of `turbvel%v` with `sigv`, `tlv`; `hanna` (283–284) documents `tlu`
  as "the along wind component" and `tlv` as "the cross wind component";
  `advance_mod.f90` 108–109: "up,vp,wp random velocities due to turbulence (along
  wind, cross wind, vertical wind".
- `advance_mod.f90` 609–612: inside the boundary-layer time loop the mean-wind
  displacement accumulates in `dxsave`, `dysave` and the turbulent one in
  `dawsave` (along) and `dcwsave` (cross).
- `advance_mod.f90` 273–276: `call windalign(dxsave,dysave,dawsave,dcwsave,ux,vy)`,
  then `dxsave=dxsave+ux`, `dysave=dysave+vy`, once per synchronisation
  interval, after the mesoscale ("meander") displacement has been added to
  `dxsave`, `dysave` (`turbulence_mod.f90` 249–250).
- `turbulence_mod.f90` 612–663, `windalign`: `ffinv=1./max(sqrt(u*u+v*v),eps)`
  with `eps=1.e-30`, `sinphi=v*ffinv`, `cosphi=u*ffinv`, along-wind to
  $(\cos\phi, \sin\phi)$, cross-wind to $(-\sin\phi, \cos\phi)$.
- `turbvel%u`, `%v` are never rotated: they are carried from step to step as
  along-wind and cross-wind components, whatever the wind did in between.
- Above the boundary layer `ux`, `vy` are white noise added in the grid frame
  (`advance_mod.f90` 431–432), isotropic (`turbulence_above_pbl`).
- The meander (`turbulence_mesoscale`, `turbulence_mod.f90` 247–250) is added
  in the grid frame with the standard deviations of the grid u and v.

Not verified: v10.4 line numbers; v10.4's `hanna_mod`; whether either version
documents the frame anywhere but in code comments. So GLIDE's "aligned to
FLEXPART v11" (decision 0006) did not hold for the orientation of horizontal
turbulence, which affects the Tier 1 FLEXPART comparisons.

**Decision.** A switch `turbulence.horizontal_frame: grid | wind` (Hanna only).

- `grid` (code default): unchanged; the new code is not on its path.
- `wind`: inside each outer step $u'$ and $v'$ are the along-wind and
  cross-wind components. The OU updates, the $\sigma$ carry (0028), the
  boundary-layer-top interface rule (0028; $u'$, $v'$ scaled by their own
  $\sigma$ ratios) and the $4\sigma$ clip all act on them with $\sigma_u, T_{Lu}$
  and $\sigma_v, T_{Lv}$. Before each horizontal displacement they are rotated to
  east/north. The displacement functions are unchanged, so the change is
  independent of the position frame (and of PR #74's metric frame).
- **Axes.** The unit vector of the mean horizontal wind at the particle:
  interpolated in space at the post-advection position (basal-clamped height, as
  the coefficients are), and in time with the advection's weight. One direction
  per particle per outer step; the sub-steps' turbulent displacements (tens of
  metres) do not change it materially.
- **Carried state.** $u'$, $v'$ are rotated into the wind frame at the start of
  each outer step and back to east/north at its end. Between steps the state is
  east/north, so its meaning and the output contract are unchanged (see "A
  turning wind").
- **Calm winds.** A particle whose mean wind speed is no more than its
  along-wind $\sigma_u$ keeps the grid axes for that step (see "Calm winds").
- Static shapes, `torch.where` only, no host sync; both step paths run the same
  code and are bit-identical (test). The meander is unchanged: its
  $\sigma$s are defined on the grid's u and v, as in FLEXPART.

**Proposed default: `wind`.** It is the parameterisation as published, it
removes a dependence on the grid's orientation, and it is what FLEXPART does,
so it *improves* comparability with the FLEXPART reference runs. What it costs:
every Tier 1–3 result produced with `grid` (and the golden fixtures, already stale
since 0027) would change in stable and neutral conditions; see "Effect".

**A turning wind.** When the wind direction at a particle changes between steps,
the carried velocity can either keep its wind-relative components (FLEXPART) or
keep its east/north components (this record). Write $V = R(\theta)DR(\theta)^T$
for the Eulerian covariance of $(u', v')$ in the fixed frame,
$D = \mathrm{diag}(\sigma_u^2, \sigma_v^2)$. Thomson's (1987) well-mixed condition
for a Gaussian PDF has the drift term
$\tfrac12 (\mathrm{d}V/\mathrm{d}t)\,V^{-1}u'$ along the particle path (plus
$\tfrac12\partial_l V_{il}$ for spatial variation, below). Integrated over a
change $\Delta\theta$ with $D$ fixed, it rotates the *normalised* wind-frame
velocity $(a/\sigma_u, c/\sigma_v)$ by
$-\tfrac12(\sigma_u/\sigma_v + \sigma_v/\sigma_u)\Delta\theta$ (derived). Then:

| carry | normalised rotation | Eulerian PDF after a turn | isotropic case ($\sigma_u = \sigma_v$) |
| --- | --- | --- | --- |
| Thomson (1987), exact | $-\tfrac12(k^{1/2} + k^{-1/2})\Delta\theta$ | kept | east/north velocity unchanged |
| east/north (this record) | a rotation by $-\Delta\theta$ in the unnormalised components | distorted, relaxing over $T_L$ | east/north velocity unchanged |
| wind-relative (FLEXPART; private `_carry_wind_relative`) | 0 | kept | velocity turned with the wind |

($k = \sigma_u^2/\sigma_v^2 \le 2.37$ in the Hanna stable regime.) Neither
practical option is Thomson's: the east/north carry gets the isotropic limit
exactly (so the convective boundary layer and the free troposphere behave
exactly as in `grid`, in distribution) and distorts the anisotropic covariance
for about $T_L$ after a turn; the wind-relative carry keeps the anisotropic PDF
exactly but turns an isotropic velocity with the wind. In the diffusion limit
($T_L \to 0$) the two are the same. Measured
(`test_wind_turning_through_calm_keeps_velocity_statistics_and_adds_no_drift`, a
wind turning through 180° via calm, half of it in ~3 min): the wind-relative
carry kept both normalised variances within 0.05 of 1; the east/north carry
departed by up to ~0.6 (cross-wind) for a few minutes at the turn, within the
derived range $[1, \tfrac12(k + 1/k)]$ for the mean of the two, and was back
within 0.05 once the turn ended. The exact form needs the previous direction per
particle as a new state tensor. It is not implemented: wind directions change by
a degree or less in a boundary-layer $T_L$ except at calm points. The east/north
carry is chosen for the exact isotropic limit (the review expects "no change in
the free troposphere") and because a calm-zone frame switch then leaves the
velocity continuous. The wind-relative carry stays as a private A/B switch for
FLEXPART-like runs. Smoke effect of the choice: see "Effect".

**Calm winds.** Below some speed the mean wind no longer defines an along-wind
axis. The threshold is $|U| \le \sigma_u$, per particle, on the step's own
$\sigma_u$ (`WIND_FRAME_CALM_RATIO = 1`). At $|U| = \sigma_u$ the wind-direction
fluctuation $\sigma_\theta \approx \sigma_v/|U| = \sigma_v/\sigma_u = 0.65$ rad
(37°) in the stable regime; below it the turbulent velocity exceeds the mean
wind in both components (argued). The fallback is the grid axes: any fallback
axis is arbitrary when there is no wind, the grid one needs no new $\sigma$ or
$T_L$ definition, and it is bit-identical to `grid` for those particles (test). An
isotropic fallback ($\tfrac12(\sigma_u^2+\sigma_v^2)$ for both) is closer to what
low-wind measurements show, but needs a $T_L$ choice. FLEXPART has no calm
treatment: it uses any non-zero accumulated displacement for the direction, and
at exactly zero it drops the horizontal turbulent displacement (`windalign`).
Fraction of particle-steps under the threshold in the smoke run: see "Effect".

**What the rotation does not fix: the horizontal drift.** GLIDE has no horizontal
well-mixed drift ($\nabla\cdot K$ in the diffusion limit), with either frame, and
neither has FLEXPART. Without it the horizontal transport is
$\partial c/\partial t = \partial_i\partial_j(K_{ij}c)$, which does not keep a
uniform $c$ uniform where $K$ varies. With `grid`, $K = \mathrm{diag}(K_u, K_v)$
varies only with $u_*$, $h$ and stability. With `wind` its orientation also varies,
$K = K_v I + (K_u - K_v)\,t\,t^T$ ($t$ the unit vector along the wind).
For a vortex, $\partial_i\partial_j(t_it_j)$ vanishes everywhere except at the
calm centre, where it integrates to $-2\pi$, whatever the calm treatment
(derived). So a calm centre is a point sink of
$2\pi(K_u - K_v)c$, and nothing else changes. The test
(`test_vortex_with_a_calm_centre_loses_only_the_derived_neglected_drift_term`:
a neutral column, $h$ = 300 m, $u_*$ = 0.4 m s⁻¹, a vortex with a 0.9 km calm
region, 100,000 particles over ±4 km, 1 h, the expectation from the along-wind
and cross-wind displacement variances of a companion run in a uniform wind)
measured VORTEX_RESULT. In a stable column ($T_{Lu}$ up to ~30 min near $h$, not
short against the hour) the measured deficit was about 0.75 of the expectation
(−509 against −693 particles, one seed;
`outputs/agents/ROT/logs/vortex_calib_wind.log`, an earlier test design): the
diffusion limit does not hold there, and the frame turns within $T_L$ along a
particle's own turbulent path, both of which reduce the effective anisotropy
(argued). In a smoothly turning wind the source rate is of order
$(K_u - K_v)/L^2$ for a turning length $L$: $10^{-9}$ s⁻¹ at $L$ = 100 km (derived).
A calm point loses $2\pi(K_u - K_v)\cdot 3600$ s of tracer area per hour:
$1.5\times10^5$ m² (a 220 m disk) for the stable column above
($\langle K_u - K_v\rangle$ = 6.7 m² s⁻¹) and about $9\times10^5$ m² (a 540 m disk)
for the neutral one (41 m² s⁻¹, from the test's companion run), against $\sim5\times10^8$ m² in a 0.25° cell
(derived). This is the same class of error as the one `grid` already makes where
$u_*$ or $h$ vary horizontally.

**Effect.** EFFECT_PARAGRAPH

**Rejected alternatives.**
- *Rotate the displacement functions* (`apply_horizontal_turbulence` taking a
  direction): ties the fix to the position frame that PR #74 changes; the
  velocity-space rotation is frame-independent.
- *The wind-relative carry as production* (FLEXPART): see "A turning wind";
  kept as a private A/B switch.
- *Thomson's exact rotation of the carried state*: a new per-particle state
  tensor through both step paths for an effect confined to near-calm turns.
- *Isotropic calm fallback*: needs a $T_L$ choice; arbitrary either way.
- *No calm treatment* (FLEXPART): the direction of a 0.01 m/s wind then sets an
  anisotropy of 3.3 in $K$.
- *The direction of the step's RK2 displacement* (FLEXPART uses the accumulated
  displacement): the dynamic step path does not see the advection.

**Status.** Proposed; implemented behind `turbulence.horizontal_frame`, default
`grid`. Matt decides the default. Docs: [physics.md](../../docs/physics.md) §3.4,
[turbulence.md](../../docs/turbulence.md) §4 and §9.
