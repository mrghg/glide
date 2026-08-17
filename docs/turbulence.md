# Turbulence

The turbulence scheme's job is narrow and well defined: given a particle's
position and the local meteorology, return the velocity variance $\sigma_i^2$ and
Lagrangian timescale $T_{Li}$ for each component. Everything else — the
Ornstein–Uhlenbeck update, the well-mixed drift, ground reflection, the backward
sign convention — is the model core, described in [physics.md](physics.md).

GLIDE's production scheme is `hanna_1982`: Hanna (1982) boundary-layer profiles
in three stability regimes, a gradient-Richardson closure above the boundary
layer, and an optional unresolved-mesoscale "meander" process. Coefficients and
conventions follow **FLEXPART v11**, so that GLIDE's dispersion is directly
comparable to a reference the field already trusts. The equations were
reimplemented from the primary literature; no FLEXPART source is included (see
[NOTICE](../NOTICE)).

**Contents**

1. [The scheme interface](#1-the-scheme-interface)
2. [How a profile is assembled](#2-how-a-profile-is-assembled)
3. [Stability classification](#3-stability-classification)
4. [Boundary-layer profiles](#4-boundary-layer-profiles)
5. [The Lagrangian-timescale floors](#5-the-lagrangian-timescale-floors)
6. [Above the boundary layer](#6-above-the-boundary-layer)
7. [Meander: unresolved mesoscale motion](#7-meander-unresolved-mesoscale-motion)
8. [The legacy surface-layer override](#8-the-legacy-surface-layer-override)
9. [Configuration](#9-configuration)
10. [References](#10-references)

---

## 1. The scheme interface

`TurbulenceScheme` (`src/lpdm/turbulence/base.py`) is a small ABC with a
name-keyed registry, so a scheme is selected by string in the run YAML and
inherits the runtime's machinery — sub-stepping, per-window field caching, GPU
graph capture — with no bespoke plumbing:

```python
class TurbulenceScheme(ABC):
    name: ClassVar[str]

    def required_met_keys(self) -> tuple[str, ...]: ...
    def initialize_state(self, n_particles, *, device, dtype) -> TurbulenceState: ...
    def step(self, particles, state, met_window, t_alpha,
             dt_seconds, active_mask, engine) -> tuple[Tensor, TurbulenceState]: ...
```

`required_met_keys` is cross-checked against the meteorology reader at startup,
so a scheme asking for a field the store does not have fails loudly rather than
silently defaulting.

Two schemes are registered:

| Name | Role |
| --- | --- |
| `hanna_1982` | Production. State `{u', v', w'}`, plus `{u_m, v_m}` with meander enabled. Requires `t`, `ustar`, `shf` beyond the baseline. |
| `placeholder_constant_ou` | Regression pin only. Vertical OU with fixed $T_L = 300$ s, $\sigma_w^2 = 1\ \mathrm{m^2\ s^{-2}}$, no horizontal turbulence, no drift. Requires no meteorology. Used to isolate runtime plumbing from physics in a few end-to-end tests. |

The physics itself lives in **standalone free functions** —
`in_bl_sigma_TL`, `obukhov_length`, `free_trop_diffusivity`,
`brunt_vaisala_squared` and so on — deliberately, so each can be unit-tested
against literature values in isolation. Graph capture wraps the *assembled*
per-step call; it never inlines these away.

---

## 2. How a profile is assembled

`_column_turbulence(z)` returns
$(\sigma_u, \sigma_v, \sigma_w, T_{Lu}, T_{Lv}, T_{Lw})$ at any height, for
every particle at once. It is callable at arbitrary
$z$ because the well-mixed drift needs to finite-difference $\sigma_w$ through
the profile, including across regime seams. The order of operations matters:

```
1. Boundary-layer profile        in_bl_sigma_TL(z, h, u*, w*, h/L, lat)
                                 → stable / neutral / unstable, selected per particle
2. Surface-layer override        z < 0.1h        [OFF by default]
3. FLEXPART T_L floors           T_Lu,T_Lv ≥ 10 s ; T_Lw ≥ 30 s
4. Free-troposphere override     z > h           Richardson closure
5. Numerical floors              σ ≥ 1e-3 m/s ; T_L ≥ 1 s
```

Step 3 comes **before** step 4 deliberately. The free-troposphere closure derives
$(\sigma_w, T_{Lw})$ as a *pair* satisfying $K_z = \sigma_w^2 T_{Lw}$; flooring
its $T_L$ afterwards would silently change the diffusivity it was constructed to
represent. FLEXPART likewise floors only inside its boundary-layer routine.

All three regime branches are evaluated for every particle and combined with
`torch.where` rather than branching. On a GPU this is faster than the data-
dependent control flow it replaces, and it is a prerequisite for graph capture.

The sampling height is $z_{\mathrm{eval}} = \max(z, 2\ \mathrm{m})$ — the
unresolved basal layer described in [physics.md §5](physics.md#5-boundary-conditions).

---

## 3. Stability classification

The regime follows the ratio of boundary-layer depth to Obukhov length.

$$
L \ =\  -\ \frac{u_\ast^3\  T_v\  \rho\  c_p}{\kappa\  g\  H}
\qquad
w_\ast \ =\  \left(\frac{g\  h\  H}{T\  \rho\  c_p}\right)^{1/3}
$$

with $\kappa = 0.4$, $c_p = 1005\ \mathrm{J\ kg^{-1}K^{-1}}$,
$g = 9.80665\ \mathrm{m\ s^{-2}}$, $\rho = p_s/(R_d T)$, and $h$ the
boundary-layer depth. $T$ is the temperature at the lowest model level, used as a
surface proxy; $T_v$ is approximated by $T$ (the humidity correction is small and
deferred). $w_\ast$ is set to zero wherever $H \le 0$, where it is undefined.

$H$ is the **upward** sensible heat flux. This is worth stating loudly, because
ERA5 stores the opposite: ECMWF's convention is positive *downward*, so a daytime
upward flux is stored negative. GLIDE negates on read. Getting this wrong inverts
the stability classification on every real met field — which is exactly what
happened before the 2026-07-02 audit, and it is the single easiest thing to get
wrong when preparing meteorology from a non-ERA5 source (see
[met_schema.md](met_schema.md)).

$$
\begin{aligned}
h/L > 1 &\quad\Rightarrow\quad \text{stable} \cr
-1 \le h/L \le 1 &\quad\Rightarrow\quad \text{neutral} \cr
h/L < -1 &\quad\Rightarrow\quad \text{unstable / convective}
\end{aligned}
$$

$L \to \pm\infty$ (hence $h/L \to 0$, neutral) wherever $|H|$ falls below a
numerical threshold. Thresholds match FLEXPART's defaults.

---

## 4. Boundary-layer profiles

Throughout, $\zeta = z/h$ and $u_\ast$ is friction velocity. Note two things that
are easy to misread: $\sigma_v$ tracks $\sigma_w$ (both $1.3u_\ast$), **not**
$\sigma_u$; and the three $T_L$ components differ from one another in the stable
regime but not the neutral one.

### Stable — Hanna (1982) Eqs. 7.19–7.24

$$
\sigma_u = 2.0\ u_\ast(1-\zeta),
\qquad
\sigma_v = \sigma_w = 1.3\ u_\ast(1-\zeta)
$$

$$
T_{Lu} = 0.15\ \frac{h}{\sigma_u}\ \zeta^{1/2},
\qquad
T_{Lv} = 0.467\ T_{Lu},
\qquad
T_{Lw} = 0.10\ \frac{h}{\sigma_w}\ \zeta^{0.8}
$$

The $\sigma$ profiles are **linear** in $(1-\zeta)$, not $(1-\zeta)^{3/4}$ — a
form that appears in some secondary references and was corrected here against
FLEXPART v11.

### Neutral — Hanna (1982) Eqs. 7.25–7.27

$$
\sigma_u = 2.0\ u_\ast\ e^{-3|f|z/u_\ast},
\qquad
\sigma_v = \sigma_w = 1.3\ u_\ast\ e^{-2|f|z/u_\ast}
$$

$$
T_{Lu} = T_{Lv} = T_{Lw} = \frac{0.5\ z}{\sigma_w\left(1 + 15|f|z/u_\ast\right)}
$$

Note the differing Ekman-decay exponents, $3|f|$ for $\sigma_u$ against $2|f|$ for
the other two.

$f = 2\Omega\sin\phi$ is the true per-latitude Coriolis parameter (FLEXPART
hardcodes $10^{-4}\ \mathrm{s^{-1}}$), but taken in **absolute value**, floored at
$10^{-5}$. Without the absolute value the decay exponents change sign in the
Southern Hemisphere and $\sigma$ *grows* with height; without the floor the
equator degenerates.

### Unstable / convective — Caughey (1982) Eq. 4.15; Ryall & Maryon (1998); Hanna (1982) Eq. 7.17

$$
\sigma_u = \sigma_v = u_\ast\left(12 - 0.5\ \frac{h}{L}\right)^{1/3}
$$

$$
\sigma_w = \sqrt{\ 1.2\ w_\ast^2\ (1 - 0.9\zeta)\ \zeta^{2/3} \ +\  (1.8 - 1.4\zeta)\ u_\ast^2\ }
$$

$$
T_{Lu} = T_{Lv} = 0.15\ \frac{h}{\sigma_u}
$$

$T_{Lw}$ is piecewise, tested in the order shown:

$$
T_{Lw} =
\begin{cases}
\dfrac{0.10\ z}{\sigma_w\left(0.55 - 0.38\ |z/L|\right)} & |z/L| < 1 \quad \text{(near-surface)} \cr[2.2ex]
\dfrac{0.59\ z}{\sigma_w} & \zeta < 0.1 \quad \text{(shallow)} \cr[2.2ex]
0.15\ \dfrac{h}{\sigma_w}\left(1 - e^{-5\zeta}\right) & \text{otherwise (bulk mixed layer)}
\end{cases}
$$

The two $\sigma_w$ contributions are the convective ($w_\ast$) and shear ($u_\ast$)
scalings: near the surface the second dominates, in the mid mixed layer the
first.

---

## 5. The Lagrangian-timescale floors

$$
T_{Lu},\ T_{Lv} \ge 10\ \mathrm{s}, \qquad T_{Lw} \ge 30\ \mathrm{s}
$$

These are FLEXPART v11's floors, and they are on by default
(`turbulence.flexpart_tl_floors`). They are not cosmetic.

Every $T_{Lw}$ formula above vanishes as $z \to 0$. The vertical diffusivity is
$K = \sigma_w^2 T_{Lw}$, so an unfloored $T_{Lw}$ collapses $K$ in the lowest
metres and **traps particles there**. In a backward run, trapped particles
accumulate residence time next to the receptor, which inflates the near-field
surface footprint — which in turn over-estimates the inferred enhancement.

This was measured, not theorised. GLIDE's v2 validation over-estimated mean CH₄
enhancements, worst at polluted low-inlet sites, on stable nights: near-surface
$K$ was running 3–8× below FLEXPART's. Turning the floors on was the fix.

A convenient side effect: with $T_{Lw} \ge 30$ s, the per-particle sub-step count
at $\Delta t = 60$ s is $\lceil 60/(0.5 \times 30) \rceil = 4$, so the
`max_substeps` cap rarely binds.

Set `flexpart_tl_floors: false` only to reproduce legacy A/B comparisons; it
reverts to a 1 s floor.

---

## 6. Above the boundary layer

For $z > h$, the boundary-layer scalings do not apply and GLIDE uses a
first-order **gradient-Richardson closure** built from the meteorology's own
vertical structure:

$$
\theta = T\left(\frac{p_0}{p}\right)^{\kappa},\quad \kappa = R_d/c_p
\qquad
N^2 = \frac{g}{\theta}\frac{\partial\theta}{\partial z}
\qquad
S^2 = \left(\frac{\partial u}{\partial z}\right)^2 + \left(\frac{\partial v}{\partial z}\right)^2
$$

$$
\mathrm{Ri} = \frac{N^2}{S^2},
\qquad
\ell = \frac{\kappa z}{1 + \kappa z/\lambda},\quad \lambda = 100\ \mathrm{m}
$$

$$
F(\mathrm{Ri}) =
\begin{cases}
\sqrt{1 - 16\ \mathrm{Ri}} & \mathrm{Ri} < 0 \quad\text{(unstable; rare aloft)}\cr[1ex]
\left(1 - \mathrm{Ri}/\mathrm{Ri}_c\right)^2 & 0 \le \mathrm{Ri} < \mathrm{Ri}_c \cr[1ex]
0 & \mathrm{Ri} \ge \mathrm{Ri}_c = 0.25 \quad\text{(sub-critical, laminar)}
\end{cases}
$$

$$
K_z = \mathrm{clamp}\left(\ell^2 \left|\frac{\partial \mathbf{U}}{\partial z}\right| F(\mathrm{Ri}),\ \ 0.1,\ \ 50\right)\ \mathrm{m^2\ s^{-1}}
$$

which is then split into the $(\sigma_w, T_{Lw})$ pair the Langevin equation
needs, using the buoyancy timescale:

$$
T_{Lw} = \mathrm{clamp}\left(\frac{0.5}{N},\ 1,\ 1000\right)\ \mathrm{s}
\quad\text{(100 s where } N^2 \le 10^{-6}\ \mathrm{s^{-2}}\text{)},
\qquad
\sigma_w = \sqrt{K_z / T_{Lw}}
$$

Horizontal free-troposphere turbulence is treated as isotropic with the vertical,
$\sigma_u = \sigma_v = \sigma_w$; the *horizontal* spread that actually matters
aloft comes from the meander process (§7), which runs at all altitudes.

**The $K_z$ floor of $0.1\ \mathrm{m^2\ s^{-1}}$ is load-bearing.** The scheme
this replaced used fixed constants ($\sigma = 0.1\ \mathrm{m\ s^{-1}}$,
$T_L = 100$ s, so $K \approx 1\ \mathrm{m^2\ s^{-1}}$), and that was a one-way
trap: a particle that left the boundary layer was effectively frozen and could
never mix back down, starving the surface footprint. A background floor
guarantees the free troposphere is never fully still.

The vertical gradients are central differences over the model levels using the
**true 3-D geopotential heights**, not the bounding-box-mean level array — so
they are correct per column even where the terrain varies sharply. The resulting
$\sigma/T_L$ fields are built once per meteorology hour on the grid, then
interpolated trilinearly to each particle above the boundary layer.

---

## 7. Meander: unresolved mesoscale motion

The scheme above parameterises three-dimensional turbulence (small eddies) and
the resolved wind carries grid-scale advection. Between them is a gap: quasi-2-D
mesoscale motions *larger* than turbulent eddies but *smaller* than the
meteorology grid resolves. Neglecting them under-disperses the plume
horizontally.

GLIDE follows the Maryon (1998) "meandering" scheme that NAME originated and
FLEXPART adopted — an **independent** horizontal OU process layered on top of the
Hanna $u'/v'$ turbulence, applied at all altitudes:

$$
\sigma_{m,i} = C_m \cdot \mathrm{std}_{\text{local}}(U_i), \quad i \in \lbrace u, v\rbrace
\qquad
\tau_m = \tfrac{1}{2}\ \Delta t_{\text{met}} \approx 1800\ \mathrm{s}
$$

$\mathrm{std}_{\text{local}}$ is the standard deviation of the grid-scale wind
component over the $(2r+1)^2$ horizontal neighbourhood of the particle (default
$r=1$, a 3×3 stencil), computed per level and interpolated to the particle. The
assumption is that grid-scale wind variability carries information about the
sub-grid variability. $C_m$ is FLEXPART's `turbmesoscale`, default 0.16. The
timescale is half the input-field interval, on the reasoning that linear
interpolation between fields already recovers about half of the sub-interval
variability.

Resolution dependence falls out for free: a finer meteorology grid has smaller
wind differences between neighbouring cells, hence a smaller $\sigma_m$, because
more of the mesoscale is already resolved. That is the behaviour NAME's
resolution-dependent meander is designed to have, without a hand-tuned
per-resolution constant.

The process has **no drift** — the well-mixed inhomogeneity is vertical — and
symmetric forcing, so unlike the vertical drift it needs no backward sign flip.
State $(u_m, v_m)$ starts at zero.

Meander is **off by default** in code so existing baselines stay bit-identical,
and **on** in every shipped example config.

---

## 8. The legacy surface-layer override

`turbulence.surface_layer_override`, default **false**, and best left that way.

FLEXPART v11 has no separate surface-layer treatment — the regime formulas of §4
run to the ground, bounded by the $T_L$ floors of §5 — and GLIDE now follows it.
When enabled, a Monin–Obukhov treatment replaces the profile below
$z_{sl} = 0.1h$:

$$
\sigma_w =
\begin{cases}
1.3\ u_\ast\left(1 - 3z/L\right)^{1/3} & \text{unstable } (z/L < 0)\cr
1.3\ u_\ast & \text{stable / neutral}
\end{cases}
\qquad
T_L = \frac{\kappa z}{\sigma}
$$

with $\sigma_u, \sigma_v$ taken from §4 evaluated at $\max(z, z_{sl})$. The floors
of §5 are applied afterwards, so enabling both still bounds $T_L$.

Two reasons it is off. First, $T_L = \kappa z/\sigma$ falls to seconds as
$z \to 0$, undercutting the floors that §5 exists to impose. Second, it puts a
discontinuity in $K$ at the $0.1h$ seam.

The stable branch being **constant in height** is itself a correction. An earlier
form, $1.3u_\ast(1 + 5z/L)$, *grew* with stability — that expression is the $\phi_m$
momentum-gradient function, not a $\sigma_w$ scaling — and it over-mixed the
nocturnal near-surface layer.

---

## 9. Configuration

```yaml
turbulence:
  scheme: hanna_1982          # required; or placeholder_constant_ou
  substep_c: 0.5              # sub_dt target: sub_dt < substep_c · T_Lw
  max_substeps: 6             # cap on per-particle sub-steps per outer step
  flexpart_tl_floors: true    # §5 — leave on
  surface_layer_override: false   # §8 — leave off
  meander:
    enabled: true             # §7
    coefficient: 0.16         # C_m, FLEXPART `turbmesoscale`
    stencil_radius: 1         # 3×3 neighbourhood
    timescale_seconds: null   # null → half the met interval (1800 s)
```

`max_substeps` is a genuine performance/accuracy knob. On the GPU graph path it
is a **fixed** per-step iteration count rather than a cap, so every particle pays
it — the default of 50 in code is far more than the $T_L$ floors require, and the
shipped configs set 6. Raise it only if you lower `simulation.dt_seconds` or
`substep_c`, or turn the floors off.

**Extra meteorology required by `hanna_1982`:**

| Key | ERA5 variable | Units | Used for |
| --- | --- | --- | --- |
| `ustar` | `friction_velocity` | m s⁻¹ | all $\sigma$, $T_L$ |
| `shf` | `surface_sensible_heat_flux` | W m⁻² (or J m⁻², de-accumulated) | $L$, $w_\ast$ |
| `t` | `temperature` | K | $L$, $w_\ast$, $\rho$, $\theta$, $N^2$ |

---

## 10. References

- Caughey, S. J. (1982). Observed characteristics of the atmospheric boundary
  layer. In *Atmospheric Turbulence and Air Pollution Modelling*, Reidel.
- Hanna, S. R. (1982). Applications in air pollution modeling. In *Atmospheric
  Turbulence and Air Pollution Modelling*, Reidel.
- Maryon, R. H. (1998). Determining cross-wind variance for low-frequency wind
  meander. *Atmos. Environ.* 32, 115–121.
- Ryall, D. B., Maryon, R. H. (1998). Validation of the UK Met Office NAME model
  against the ETEX dataset. *Atmos. Environ.* 32, 4265–4276.
- Stohl, A., Forster, C., Frank, A., Seibert, P., Wotawa, G. (2005). Technical
  note: The Lagrangian particle dispersion model FLEXPART version 6.2. *Atmos.
  Chem. Phys.* 5, 2461–2474. (§4.5 meander.)
- Stohl, A., Hittenberger, M., Wotawa, G. (1998). Validation of the Lagrangian
  particle dispersion model FLEXPART. *Atmos. Environ.* 32, 4245–4264.
- Wilson, J. D., Sawford, B. L. (1996). Review of Lagrangian stochastic models
  for trajectories in the turbulent atmosphere. *Boundary-Layer Meteorol.* 78,
  191–210.

Plus the core-model references in [physics.md §9](physics.md#9-references) —
Thomson (1987), Wilson & Flesch (1993), Stohl & Thomson (1999), Flesch et al.
(1995).
