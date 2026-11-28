# Deep convection

Boundary-layer turbulence mixes air within the boundary layer; the
gradient-Richardson closure handles slow shear-driven mixing above it. Neither
represents **deep moist convection** — cumulus updrafts that loft surface air
through the whole troposphere in minutes to hours. For mid-latitude continental
sites in summer, and tropical sites year-round, that is a major missing transport
mechanism: particles released in the boundary layer ought sometimes to reach
8–12 km within a few hours, where the resolved winds are far faster and more
variable.

GLIDE's `emanuel_reduced` scheme is a reduced port of the Emanuel &
Živković-Rothman (1999) mass-flux scheme as FLEXPART implements it (Forster,
Stohl & Seibert 2007), written at a higher level of abstraction than the ~3000
lines of `convect43c.f`. It is off by default (`convection.scheme: none`) and
enabled in the shipped example configs. Like FLEXPART's, it works column by
column: every (longitude, latitude) column of the meteorology gets its own parcel
lift, trigger and mass-flux matrix, and each particle is moved with the matrix of
the column it is in (since task A02; before that one bounding-box-mean column
stood for the whole domain).

**Contents**

1. [Where it runs in the timestep](#1-where-it-runs-in-the-timestep)
2. [The parcel lift](#2-the-parcel-lift)
3. [The trigger](#3-the-trigger)
4. [Cloud-base mass flux](#4-cloud-base-mass-flux)
5. [The mass-flux matrix](#5-the-mass-flux-matrix)
6. [Particle redistribution, and the backward transpose](#6-particle-redistribution-and-the-backward-transpose)
7. [Departures from full Emanuel](#7-departures-from-full-emanuel)
8. [Configuration](#8-configuration)
9. [References](#9-references)

---

## 1. Where it runs in the timestep

Convection is a separate runtime stage from turbulence, and it fires **once per
meteorology window** — typically once an hour — not every integration step. Three
reasons:

- It redistributes particles **non-locally**. A particle can jump from the
  surface to the tropopause in one event, whereas turbulence is a sequence of
  small correlated increments.
- The mass-flux matrix depends only on the column's $(T, q)$ profile, which is
  constant within a meteorology window. Recomputing it per step would be pure
  waste.
- It matches FLEXPART, whose `convmix` is called every sync-time rather than
  every internal timestep (Stohl et al. 2005 §4.6).

The runtime tracks the current meteorology bracket's start time and fires
convection exactly once whenever the cursor crosses into a new bracket. A
bracket is one time step of the meteorology store (`reader.met_interval_seconds`,
measured from the store's time coordinate): 3,600 s on hourly ERA5, 10,800 s on a
3-hourly source. The mass flux is scaled by that interval (§4), so the scheme
follows the store's cadence.

**Columns.** Each event works on every column of the meteorology window at once:
the profiles are tensors of shape [columns, levels], interpolated in time to the
event, and the parcel lift, the trigger and the closure are evaluated for all
columns together, with no Python loop over columns or levels. A column's profile
is its own temperature, humidity, height above ground and pressure on the
terrain-following levels (the reader supplies the per-column pressure; synthetic
windows without it use the level-mean pressure). Matrices ([Z, Z] each) are built
only for the columns that fire **and** hold at least one active particle, so the
matrix memory scales with the occupied columns, not with the firing area. On a
synthetic EUROPE window with 90% of columns firing and particles in 1,808 of them,
the event's GPU peak fell from 3.6 to 0.5 GB on the default 40-level grid and from
41.2 to 1.7 GB on 137 levels (PR #54 review).
A particle belongs to the column of the nearest meteorology
grid point, and its host layer is that column's level nearest its height.

The only meteorology this scheme needs beyond the baseline is specific humidity:

| Key | ERA5 variable | Units | Dims |
| --- | --- | --- | --- |
| `q` | `specific_humidity` | kg kg⁻¹ | 3-D |

---

## 2. The parcel lift

A surface parcel $(T_s, q_s, p_s)$ is lifted through the column. Below its
lifting condensation level it follows a dry adiabat; above, a moist
pseudo-adiabat.

**LCL**, from Bolton (1980) Eq. 22 for the temperature and Poisson's equation for
the pressure:

$$
T_{\mathrm{LCL}} = \frac{2840}{3.5\ln T - \ln e - 4.805} + 55,
\qquad
p_{\mathrm{LCL}} = p\left(\frac{T_{\mathrm{LCL}}}{T}\right)^{c_p/R_d}
$$

with vapour pressure $e$ in hPa, obtained from specific humidity by
$e = qp/(\epsilon + (1-\epsilon)q)$, $\epsilon = R_d/R_v \approx 0.622$.

**Below the LCL**, dry adiabatic:

$$
T(p) = \theta_s \left(\frac{p}{p_0}\right)^{\kappa}, \qquad \kappa = R_d/c_p
$$

**Above the LCL**, equivalent potential temperature is conserved:

$$
\theta_e \approx \theta \  \exp\left(\frac{L_v\ q}{c_p\ T_{\mathrm{LCL}}}\right)
$$

and the parcel temperature at each level is found by solving
$\theta_e\big(T, q_{\mathrm{sat}}(T,p), p\big) = \theta_{e,\text{parcel}}$ for
$T$.

That solve is done by **bisection**, deliberately. Fixed-point iteration and
Newton both fail to converge near deep-convection temperatures, because
$\mathrm{d}\theta_e/\mathrm{d}T$ amplifies rapidly with $T$ through
Clausius–Clapeyron. The bracket $[180, 350]$ K covers polar tropopause to
tropical surface, and 30 halvings shrink it to $\sim 1.6\times10^{-7}$ K — well
past converged. The loop runs a fixed count with no convergence check, since the
check would be a device-to-host synchronisation on every iteration of every
level.

**Buoyancy** is evaluated in virtual temperature, with the parcel saturated above
its LCL:

$$
B = \frac{T_{v,\text{parcel}} - T_{v,\text{env}}}{T_{v,\text{env}}},
\qquad
T_v = T\left(1 + \left(\tfrac{1}{\epsilon}-1\right)q\right)
$$

giving the level of neutral buoyancy (cloud top: the first level above the LCL
where $B$ crosses zero from positive) and

$$
\mathrm{CAPE} = \int_{\mathrm{LFC}}^{\mathrm{LNB}} g\ B\ \mathrm{d}z
\ \approx \sum_{k = i_{\mathrm{LCL}}}^{i_{\mathrm{LNB}}} g\ \max(B_k, 0)\ \mathrm{d}z_k
$$

with layer thicknesses from the hypsometric relation,
$\mathrm{d}z \approx R_d \bar{T}\ \mathrm{d}(\ln p)/g$ ($\mathrm{d}z_k$ is the
layer below level $k$), and $i_{\mathrm{LCL}}$ the first level above the LCL
(FLEXPART's `ICB`). The sum is the **cloud's** buoyant energy
([decision 0031](../dev/decisions/0031-cape-bounded-at-the-cloud-top.md)):

- **Upper bound: the diagnosed cloud top.** Positive buoyancy in a layer above
  the LNB, separated from the cloud by stable air, does not count. Until
  November 2026 it did: the sum ran to the top of the column, so a second
  buoyant layer aloft could lift a lower cloud over the CAPE floor of §3.
- **Lower bound: the LFC, within a level.** $B > 0$ at every level strictly
  between $i_{\mathrm{LCL}}$ and $i_{\mathrm{LNB}}$ (that is how the LNB is
  found), so the only level in the sum that can be negatively buoyant is
  $i_{\mathrm{LCL}}$ itself. The LFC is therefore $i_{\mathrm{LCL}}$ or
  $i_{\mathrm{LCL}} + 1$, and summing $\max(B, 0)$ from $i_{\mathrm{LCL}}$ is the
  sum from the LFC.
- **Convective inhibition is left out, not subtracted.** Negative buoyancy below
  the LCL (the dry ascent) and at $i_{\mathrm{LCL}}$ does not reduce CAPE. A
  column whose parcel is negatively buoyant at $i_{\mathrm{LCL}} + 1$ never fires,
  whatever lies above: its cloud top is then $i_{\mathrm{LCL}} + 1$ and the
  $\Delta T_v$ check of §3 fails there. GLIDE has no elevated LFC.

The scalar reference (`compute_lcl_lnb_cape`) and the batched form
(`compute_lcl_lnb_cape_columns`) apply the same bounds
(`tests/test_cape_cloud_top.py`).

---

## 3. The trigger

Convection fires only when **all** of the following hold:

| Condition | Meaning |
| --- | --- |
| $i_{\mathrm{LCL}} \ge 0$ | the parcel reaches saturation somewhere in the column |
| $i_{\mathrm{LNB}} > i_{\mathrm{LCL}}$ | positive cloud depth |
| $\mathrm{CAPE} \ge 50\ \mathrm{J\ kg^{-1}}$ | sanity floor (`min_cape_j_kg`), on the cloud's CAPE, LCL to LNB (§2) |
| $\Delta T_v\big\vert_{\mathrm{LCL}+1} \ge 0.9\ \mathrm{K}$ | Forster (2007) Eq. 34 buoyancy threshold (`trigger_dtv_k`) |
| cloud depth $\ge 500\ \mathrm{m}$ | deep convection only (`min_cloud_depth_m`) — shallow cumulus is the boundary-layer scheme's job |

The checks are made in each column separately. A column where any check fails
has no convection in this meteorology update, and its particles are untouched. A negative
control test asserts it does **not** fire on a stable winter sounding
(`test_emanuel_does_not_fire_on_winter_inversion_column`).

---

## 4. Cloud-base mass flux

$$
w_{\text{buoy}} = \min\left(\sqrt{2\ \mathrm{CAPE}},\ 5\ \mathrm{m\ s^{-1}}\right),
\qquad
M_b = c_{\text{closure}} \  \rho_{\mathrm{LCL}} \  w_{\text{buoy}}
$$

with $c_{\text{closure}} = 0.03$ by default, evaluated in each column with that
column's CAPE and its environmental temperature and pressure at the LCL level. One
event moves $M_b\,\Delta t_{\mathrm{met}}$ per unit area, where
$\Delta t_{\mathrm{met}}$ is the meteorology interval (§1).

**At the defaults the cap binds in every column that fires.** It is reached at
$\mathrm{CAPE} = 12.5\ \mathrm{J\ kg^{-1}}$ and the trigger needs
$50\ \mathrm{J\ kg^{-1}}$, so a firing column always has
$M_b = 5\,c_{\text{closure}}\,\rho_{\mathrm{LCL}}$: CAPE decides only *whether*
a column convects, and $M_b$ varies between columns only through
$\rho_{\mathrm{LCL}}$.

**And the CFL cap of §5 binds as well, so neither sets the transport.** On ERA5
over Europe it binds in more than 99% of firing columns, with a median scale of
0.13 (January) and 0.25 (summer). The layer that limits is always the one just
below the LCL, $i_{\mathrm{LCL}} - 1$: it passes on the whole updraft flux
(its own entrainment plus the subsidence from the boundary layer below it), so
it sheds $M_b\,\Delta t_{\mathrm{met}}$, and the scaled matrix entrains exactly
$0.9\,m_{i_{\mathrm{LCL}}-1}$ per event. The mass a firing column moves in one
event therefore depends on the vertical grid (the thickness of that layer) and
on the meteorology cadence (one event per window), not on CAPE, $M_b$ or
$c_{\text{closure}}$. On 80 levels instead of 40 the median mass per event
falls from 79 to 37 kg m⁻², and about half as many columns fire. These are
open questions about the closure, to be settled before the physics freeze; an
external review raised them in November 2026, and they are recorded with what
FLEXPART does in
[decision 0031](../dev/decisions/0031-cape-bounded-at-the-cloud-top.md). The
closure is not changed.

The cap on $w_{\text{buoy}}$ is not decoration. Without it,
$\mathrm{CAPE} > 500\ \mathrm{J\ kg^{-1}}$ gives
$w_{\text{buoy}} > 30\ \mathrm{m\ s^{-1}}$ and an $M_b$ far outside FLEXPART's
realistic $0.05\text{–}0.5\ \mathrm{kg\ m^{-2}\ s^{-1}}$ range.
5 m s⁻¹ represents a typical updraft-cell peak rather than the rare 15 m s⁻¹
extreme. This stands in for the full Emanuel quasi-equilibrium closure (§7).

---

## 5. The mass-flux matrix

$\mathrm{fmass}[i,j]$ is the mass moved from layer $i$ to layer $j$ per
convection event. It is built to be **non-divergent**:

$$
\sum_j \mathrm{fmass}[i,j] \ =\  \sum_j \mathrm{fmass}[j,i] \qquad \text{for every } i
$$

— everything leaving a layer is matched by something entering it. That property
is the whole game: it is what makes the redistribution preserve a mass-weighted
(well-mixed) ensemble, in either time direction.

With levels ascending, surface at index 0, cloud spanning
$[\mathrm{LCL}, \mathrm{LNB}]$, and layer air masses $m_i = \Delta p_i / g$:

**Entrainment.** The boundary layer $[0, \mathrm{LCL})$ feeds the updraft, shared
across its layers *by air mass* so that the total entrained equals $M_b$:

$$
e_i = M_b \frac{m_i}{\sum_{k < \mathrm{LCL}} m_k}
$$

(Sharing by mass matters. An earlier version gave each boundary-layer layer the
full $M_b$, over-venting the boundary layer by a factor of the layer count.)

**Detrainment.** The updraft deposits mass across the cloud with a linear-decay
profile — 1 at cloud base falling to 0 at cloud top — normalised so the total is
again $M_b$:

$$
d_j = M_b \frac{\omega_j}{\sum_k \omega_k},
\qquad
\omega_j = 1 - \frac{j - \mathrm{LCL}}{\mathrm{LNB} - \mathrm{LCL}}
$$

**Direct updraft.** The outer product of the two — boundary-layer source to cloud
destination, non-local in one event. This is the coherent deep-lofting transport:

$$
U[i,j] = \frac{e_i\  d_j}{M_b}
$$

**Compensating subsidence.** Mass carried up by the updraft has to come back
down. The net upward flux across the interface below layer $k+1$ is

$$
\Phi_k = \sum_{i \le k} e_i \ -\  \sum_{j \le k} d_j \ \ (\ge 0)
$$

and the environment sinks at the same rate, adding $\Phi_k$ to
$\mathrm{fmass}[k+1, k]$ — a sub-diagonal term.

**CFL cap.** Finally the whole matrix is scaled by a **single scalar** if any
layer would shed more than 90% of its mass in one event. One scalar, so
non-divergence survives, and so the "stay" diagonal of §6 stays non-negative.
At the defaults this cap binds in almost every firing column, at the layer just
below the LCL, so it sets how much one event moves (§4). FLEXPART has no
counterpart: its `convect` flags a CFL violation (`IFLAG = 4`), but `calcmatrix`
uses the matrix anyway, and it is called every `LSYNCTIME` (900 s by default)
rather than once per met window.

The matrix is identically zero when there is no cloud.

---

## 6. Particle redistribution, and the backward transpose

Following FLEXPART's `calcmatrix`/`redist`, the diagonal closure makes each row
sum to the layer's air mass — the diagonal is the mass that *stays*:

$$
\mathrm{fmassfrac}[i,i] = m_i - \sum_j \mathrm{fmass}[i,j]
$$

The destination distribution for a particle hosted in layer $i$ is then a row or
a column, depending on time direction:

$$
P[i \to j] =
\begin{cases}
\mathrm{fmassfrac}[i,j] \ /\  m_i & \text{forward } (\text{ldirect}=+1) \cr[1ex]
\mathrm{fmassfrac}[j,i] \ /\  m_i & \text{backward } (\text{ldirect}=-1)
\end{cases}
$$

**GLIDE runs backward, so it samples the column.** This is the adjoint of the
forward updraft: forward convection lofts boundary-layer air to cloud top, so the
backward (footprint) operator traces a particle that is aloft *now* down to the
boundary-layer air it came from.

Each particle uses $P$ of its own column (§1). Sampling is a uniform draw and a
cumulative-sum search over the row of $P$; a particle whose sampled destination is its own layer keeps its position, and a
mover is placed at a uniformly random height within the destination layer. Only
altitude changes — longitude, latitude and weight are untouched, so the particle
count and total mass are exactly conserved.

**Why non-divergence matters, concretely.** Because $\mathrm{fmass}$ is
non-divergent, the layer-mass vector $\mathbf{m}$ is a stationary distribution of
*both* the row- and column-normalised transitions:

$$
\mathbf{m}^{\mathsf{T}} P = \mathbf{m}^{\mathsf{T}}
$$

An initially well-mixed (mass-proportional) ensemble therefore stays well-mixed
after a convection event, in either time direction. This is asserted
deterministically in the tests, not just checked statistically
(`test_convection_transition_preserves_mass_distribution_both_directions`, and
for every column of a box whose columns differ,
`test_mass_is_conserved_in_every_column`). The
matrix that preceded it had an updraft but no subsidence, was divergent, and
violated the well-mixed criterion; no ad-hoc move-probability clamp is needed
once the matrix is built correctly.

---

## 7. Departures from full Emanuel

Each of these is a deliberate simplification, listed with what it costs.

1. **Resolved (task A02): each column is now treated on its own**, as in the
   full scheme. Until A02 GLIDE lifted one parcel in the bounding-box-mean
   profile, so convection fired everywhere or nowhere: over a continental domain
   the mean column almost never fired in winter, while hundreds of tropical and
   warm-sector columns would have. What remains different from FLEXPART is the
   column assignment: GLIDE uses the nearest grid column, on the
   terrain-following levels the rest of the model uses, rather than FLEXPART's
   own convection grid. Decision
   [0023](../dev/decisions/0023-convection-per-column.md).

2. **Linear detrainment profile**, not Emanuel's buoyancy-sorting spectrum
   (Forster 2007 Eqs. 35–36). This sets only *where* the updraft deposits mass —
   the mass-conservation structure is guaranteed by the compensating subsidence
   regardless of the profile's shape.

3. **No explicit saturated-downdraft branch.** Emanuel (1991 §4b) carries a
   separate downdraft mass-flux matrix; here the compensating environmental
   subsidence carries the return flux instead. A few-percent effect.

4. **Capped buoyancy velocity** in place of the quasi-equilibrium closure (§4).
   The full scheme balances the mass flux against large-scale destabilisation;
   GLIDE caps $\sqrt{2\ \mathrm{CAPE}}$ to keep $M_b$ realistic. Revisit if
   validation shows under-convective transport.

5. **Adjacent-layer subsidence.** The environmental return flux moves mass one
   layer down per event, while the updraft is non-local. Physically apt — a fast
   coherent updraft against slow broad subsidence — but it means the descent of
   non-displaced environmental air is a random walk rather than a prescribed
   velocity.

6. **Convection fires once per met window**, and scales the mass flux by that
   window's own duration, which the reader sets to the store's time step
   (`reader.met_interval_seconds`). It therefore follows whatever cadence the
   meteorology has rather than assuming an hour (since September 2026; the
   acceptance tests of A02 pin 3,600 s for hourly and 10,800 s for a 3-hourly
   store).

---

## 8. Configuration

```yaml
convection:
  scheme: emanuel_reduced     # or "none" (default; bit-equivalent to no convection)
  emanuel:
    closure_c: 0.03           # cloud-base mass-flux closure constant
    trigger_dtv_k: 0.9        # Forster 2007 Eq 34 buoyancy threshold, K
    min_cape_j_kg: 50.0       # CAPE floor below which convection never fires
    min_cloud_depth_m: 500.0  # skip shallow convection
```

A YAML with no `convection:` block produces output identical to a run with no
convection at all.

**What to expect when you turn it on:** more particles in the free troposphere at
long backward times, and a more dispersed surface footprint (convective lofting
feeds faster long-range transport). Mid-latitude January is a weak test — the
scheme will show much more in summer.

---

## 9. References

- Bolton, D. (1980). The computation of equivalent potential temperature. *Mon.
  Wea. Rev.* 108, 1046–1053.
- Emanuel, K. A. (1991). A scheme for representing cumulus convection in
  large-scale models. *J. Atmos. Sci.* 48, 2313–2335.
- Emanuel, K. A., Živković-Rothman, M. (1999). Development and evaluation of a
  convection scheme for use in climate models. *J. Atmos. Sci.* 56, 1766–1782.
- Forster, C., Stohl, A., Seibert, P. (2007). Parameterization of convective
  transport in a Lagrangian particle dispersion model and its evaluation. *J.
  Appl. Meteor. Climatol.* 46, 403–422.
- Stohl, A., Forster, C., Frank, A., Seibert, P., Wotawa, G. (2005). Technical
  note: The Lagrangian particle dispersion model FLEXPART version 6.2. *Atmos.
  Chem. Phys.* 5, 2461–2474. (§4.6, moist convection.)
