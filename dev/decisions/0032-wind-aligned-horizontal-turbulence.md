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
  `dxsave`, `dysave` (`turbulence_mod.f90` 249–250). So the axis is that of the
  mean-plus-mesoscale displacement, and for a particle that leaves the boundary
  layer within the interval it also includes the above-boundary-layer
  displacement, turbulent part included (`advance_mod.f90` 431–432).
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
  the coefficients are) with the same sampler as the advection (`grid_sample`,
  or `trilinear_sample` for a 16-bit `wind_gather_dtype`, task A4), and in time
  with the advection's weight. u and v are east and north in either position
  frame, so the direction does not depend on the frame. One direction
  per particle per outer step; the sub-steps' turbulent displacements (tens of
  metres) do not change it materially.
- **Carried state.** $u'$, $v'$ are rotated into the wind frame at the start of
  each outer step and back to east/north at its end, so between steps the state
  is east/north (see "A turning wind" for why, and for FLEXPART's wind-relative
  alternative).
- **Calm winds.** A particle whose mean wind speed is no more than its
  along-wind $\sigma_u$ keeps the grid axes for that step (see "Calm winds").
- Static shapes, `torch.where` only, no host sync; both step paths run the same
  code and are bit-identical (test). The meander is unchanged: its
  $\sigma$s are defined on the grid's u and v, as in FLEXPART.

**Proposed default: `wind`, with the east/north carry.** It is the
parameterisation as published, it removes a dependence on the grid's
orientation, and FLEXPART rotates too, so it *improves* comparability with the
FLEXPART reference runs. The independent review of PR #77 recommended `wind` as
the default together with the wind-relative carry; the second part is not taken
here, because the wind-relative carry collects tracer at calm points (measured,
"A turning wind"); Matt decides both. What it costs:
every Tier 1–3 result produced with `grid` (and the golden fixtures, already stale
since 0027) would change in stable and neutral conditions; see "Effect".

**A turning wind.** When the wind direction at a particle changes between steps,
the carried velocity can keep its wind-relative components (FLEXPART) or its
east/north components (this record). Write $V = R(\theta)DR(\theta)^T$ for the
Eulerian covariance of $(u', v')$ in the fixed frame,
$D = \mathrm{diag}(\sigma_u^2, \sigma_v^2)$, $k = \sigma_u^2/\sigma_v^2 \le 2.37$
in the Hanna stable regime. In two dimensions the well-mixed solution is not
unique: any map that sends $N(0, V_{\rm old})$ to $N(0, V_{\rm new})$ keeps a
well-mixed state well-mixed (Thomson 1987's drift is one choice; integrated over a
turn $\Delta$ with $D$ fixed it rotates the normalised wind-frame velocity by
$-\tfrac12(k^{1/2} + k^{-1/2})\Delta$, derived). Two cases differ:

- *The wind turns in time* (the direction at a fixed place changes). The
  wind-relative carry maps $N(0, D)$ in the old wind frame to $N(0, D)$ in the new
  one, so it is exact in every regime (with $\sigma_u = \sigma_v$ any rotation
  leaves the isotropic Gaussian unchanged). The east/north carry is exact only
  when $\sigma_u = \sigma_v$; otherwise a turn $\Delta$ leaves a spurious
  along/cross covariance $(\sigma_u^2 - \sigma_v^2)\sin\Delta\cos\Delta$ and
  variances off by up to a factor $k$, relaxing over $T_L$ (PR #77 review,
  derived). Measured
  (`test_wind_turning_through_calm_keeps_velocity_statistics_and_adds_no_drift`,
  a wind turning through 180° via calm, half of it in ~3 min, particles below
  $0.75h$, `outputs/agents/ROT/logs/turning_series.log`): wind-relative, both
  normalised variances within 0.02 of 1 throughout; east/north, 0.89 (along) and
  1.17 (cross) for about five minutes at the turn, back within 0.02 a minute after.
- *The direction varies in space* (a particle crosses a turning or calm region by
  its own turbulent motion). Here the wind-relative carry turns the particle's
  velocity with the direction it moves through, a term $(u'\cdot\nabla\theta)$ in
  the velocity equation that does not vanish for isotropic turbulence and that
  the well-mixed condition does not allow (derived: with $p = c\,N(w; 0, D)$ in
  wind-frame velocity $w$, stationarity of a uniform $c$ needs
  $\nabla_x\cdot(R(\theta(x))w) = 0$ for every $w$, false wherever $\nabla\theta \ne 0$).
  Measured in the calm-centre vortex (`outputs/agents/ROT/logs/vortex_carry.log`,
  1 h, two seeds): in a convective, isotropic column, where the exact answer is
  no change and `grid` and the east/north carry give −116/−193 and −167/−31
  particles in the disk round the centre, the wind-relative carry *gained*
  +7,048/+7,195 (z ≈ +45); in the neutral column it gave +728/+716 against the
  −1,444 of the derived $\nabla\cdot K$ sink (z ≈ +23). The east/north carry is
  exact for isotropic turbulence whatever the direction field, so it adds
  nothing beyond the neglected $\nabla\cdot K$ term below.

So neither practical carry is exact everywhere. The east/north carry's error is a
velocity-covariance transient that lasts about $T_L$ after a fast turn in time and
moves no tracer in a horizontally uniform field; the wind-relative carry's error
is a systematic drift that collects tracer at calm points, in every regime. **The
east/north carry is production**, guarded by
`test_isotropic_vortex_has_no_calm_point_source_with_the_east_north_carry`; the
wind-relative carry stays as the private A/B switch `_carry_wind_relative`. The
carried $u'$, $v'$ are not part of any output, so the choice is physics only. An
exact treatment of both cases would rotate the carried velocity by Thomson's
normalised angle for the along-path change of direction, which needs the previous
direction per particle as a new state tensor (follow-up). Smoke effect of the
choice: see "Effect".

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

**What the rotation does not fix: the horizontal drift (a known error,
characterised).** GLIDE has no horizontal
well-mixed drift ($\nabla\cdot K$ in the diffusion limit), with either frame, and
neither has FLEXPART. Without it the horizontal transport is
$\partial c/\partial t = \partial_i\partial_j(K_{ij}c)$, which does not keep a
uniform $c$ uniform where $K$ varies. With `grid`, $K = \mathrm{diag}(K_u, K_v)$
varies only with $u_*$, $h$ and stability. With `wind` its orientation also varies,
$K = K_v I + (K_u - K_v)\,t\,t^T$ ($t$ the unit vector along the wind).
For a vortex, $\partial_i\partial_j(t_it_j)$ vanishes everywhere except at the
calm centre, where it integrates to $-2\pi$, whatever the calm treatment
(derived). So a calm centre is a point sink of
$2\pi(K_u - K_v)c$, and nothing else changes. The vortex test characterises this
known error; it does not show its absence
(`test_vortex_with_a_calm_centre_loses_only_the_derived_neglected_drift_term`:
a neutral column, $h$ = 300 m, $u_*$ = 0.4 m s⁻¹, a vortex with a 0.9 km calm
region, 156,250 particles over ±5 km, 1 h, the expectation from the along-wind
and cross-wind displacement variances of a companion run in a uniform wind)
measured, over 10 seeds, a loss of 1,391 ± 100 particles (mean ± sd) from the
2.4 km disk round the centre against 1,436 expected (ratio 0.97; each seed
within 1.9 of its standard deviation), and no significant change in a ring
outside the calm region (within 2.3 standard deviations); with `grid` both
were zero within 2.0 standard deviations (3 seeds;
`outputs/agents/ROT/logs/vortex_calib_*_final.log`). The test asserts 4
standard deviations. In a stable column ($T_{Lu}$ up to ~30 min near $h$, not
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

**Effect.** Measured on CPU, 20,000 particles, 24 h backward, after merging
PR #74 (`outputs/agents/ROT/logs/compare_smoke.log`):

- `grid` is bit-identical to PR #74's head ea8bb16 (footprint, endpoint
  particles, trajectory diagnostics; Mace Head smoke, seed 42), and before the
  merge to main 7b6c5eb.
- Mace Head smoke (`configs/smoke_mhd_single_release.yaml`, 10 m, 1 Jan 2024),
  `wind` against `grid`, same seed: 0–40 m +0.36% (seed 42) and −0.11% (seed
  43); total −0.055% and +0.013%. Seed to seed with `grid`: 0–40 m +0.42%, total
  −0.02%. So no change beyond particle noise. Plume widths (footprint-weighted
  principal-axis standard deviations) changed by ≤ 1%: 0–40 m cross-plume
  38.5 → 38.2 km, along-plume 371.4 → 371.7 km. The flow was near-westerly, so
  along-wind was close to east and the two frames nearly coincide here; this
  case cannot show the effect's size in a cross-grid wind.
- Tacolneston 185 m, stable night (31 Jan 2024 03 UTC, seed 42): 0–40 m −0.52%,
  total +0.02% (seed spread of 0–40 m for this case ~0.7%, PR #75); widths
  within 1%.
- Calm fallback: no particle-step in either run had $|U| \le \sigma_u$, nor
  $|U| \le 1$ m s⁻¹ (28.8 million each; read-only counter
  `outputs/agents/ROT/scripts/calmcount.py`).
- Carried state wind-relative (FLEXPART) against east/north, Mace Head seed 42:
  0–40 m −0.15%, total +0.02%, within noise.
- Cost: one extra wind sample and two rotations per particle per outer step, and one
  rotation per sub-step; smoke wall time 6:21 (`wind`, seed 43) against 6:00 (`grid`) on a
  shared, heavily loaded CPU node, not a benchmark; no GPU measurement.

The idealised test (stable, 10 min, 90° rotation) shows what the switch is for:
a plume's along-wind and cross-wind widths differ by 1.75:1 and with `grid` that
ratio follows the grid, not the wind. A cross-grid case (PR #77 review,
`outputs/agents/ROT/review/compare_tac28.log`; Tacolneston 185 m, 28 Jan 2024
03 UTC, southerly wind, 6 h, seeds 42 and 43, meander on and off): footprints
within particle noise (0–40 m totals −0.11% to +0.02% against a seed spread of
0.4–0.7%; cell-by-cell L1 0.0014–0.0027 against 0.0067–0.0087 seed to seed), and
the cross-wind (east–west) spread of the endpoints 0.2–0.9% narrower with
`wind`. That is small because at hours the horizontal spread comes from the
resolved wind (shear, meander), and Hanna's horizontal $K$ adds only about 1% to
it (review). So the switch matters most where the boundary-layer turbulence
itself sets the spread: near the receptor, in the first tens of minutes of
travel, in stable or neutral air with the wind across the grid axes, and in the
Tier 1 comparisons with FLEXPART, which rotates. Not run: a calm-prone case;
the compiled GPU smoke with `wind` (the GPU run was not permitted in this
session; the CPU capture guards pass for both frames).

**Rejected alternatives.**
- *Rotate the displacement functions* (`apply_horizontal_turbulence` taking a
  direction): ties the fix to the position frame that PR #74 changes; the
  velocity-space rotation is frame-independent.
- *The wind-relative carry as production* (FLEXPART; recommended by the PR #77
  review): exact for a turn in time, but a source of tracer at calm points where
  the direction varies in space, in every regime (see "A turning wind"); kept as
  a private A/B switch.
- *Thomson's exact rotation of the carried state* (along-path, normalised): exact
  in both cases apart from the ∇·K term, but needs the previous direction per particle as a new state
  tensor through both step paths; a follow-up.
- *Isotropic calm fallback*: needs a $T_L$ choice; arbitrary either way.
- *No calm treatment* (FLEXPART): the direction of a 0.01 m/s wind then sets an
  anisotropy of 3.3 in $K$.
- *The direction of the step's RK2 displacement* (FLEXPART uses the accumulated
  displacement): the dynamic step path does not see the advection.

**Related, not in this record.** $u'$, $v'$ carry no correction for the
vertical gradient of $\sigma_u$, $\sigma_v$ (found by this record's tests; both
frames; FLEXPART the same). A cheap exact fix is to rescale them at every
sub-step by $\sigma(z_{\rm new})/\sigma(z_{\rm old})$, $u' \to
u'\,\sigma_u(z_{\rm new})/\sigma_u(z_{\rm old})$ and likewise $v'$, the same
treatment the $\sigma$ carry (0028) gives a change in time; not part of this PR.

**Status.** Proposed; implemented behind `turbulence.horizontal_frame`, default
`grid`. Matt decides the default. Docs: [physics.md](../../docs/physics.md) §3.4,
[turbulence.md](../../docs/turbulence.md) §4 and §9.
