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
3-hourly source. The transition each event applies is the exact exponential of
the convective rate over that interval (§6, decision
[0033](../dev/decisions/0033-buoyancy-closure-exact-exponential.md)), so one
3-hourly event moves what three hourly ones would on the same meteorology.

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
  buoyant layer aloft could lift a lower cloud over the CAPE floor of the
  trigger `glide` (§3).
- **Lower bound: the LFC, within a level.** $B > 0$ at every level strictly
  between $i_{\mathrm{LCL}}$ and $i_{\mathrm{LNB}}$ (that is how the LNB is
  found), so the only level in the sum that can be negatively buoyant is
  $i_{\mathrm{LCL}}$ itself. The LFC is therefore $i_{\mathrm{LCL}}$ or
  $i_{\mathrm{LCL}} + 1$, and summing $\max(B, 0)$ from $i_{\mathrm{LCL}}$ is the
  sum from the LFC.
- **Convective inhibition is left out, not subtracted.** Negative buoyancy below
  the LCL (the dry ascent) and at $i_{\mathrm{LCL}}$ does not reduce CAPE. A
  column whose parcel is negatively buoyant at $i_{\mathrm{LCL}} + 1$ has its
  cloud top there, whatever lies above. Under trigger `glide` it never fires
  (the $\Delta T_v$ check fails at that level); under the default trigger (§3)
  it can, with a one-level cloud. GLIDE has no elevated LFC.

The scalar reference (`compute_lcl_lnb_cape`) and the batched form
(`compute_lcl_lnb_cape_columns`) apply the same bounds
(`tests/test_cape_cloud_top.py`).

---

## 3. The trigger

Since decision [0034](../dev/decisions/0034-flexpart-cloud-base-trigger.md)
(28 Nov 2026) the trigger is FLEXPART 11's (`convect` in `conv_mod.f90`),
without its memory (`trigger: flexpart_cloud_base`,
`flexpart_cloud_base_trigger_columns`). ICB is the first level above the LCL,
$i_{\mathrm{LCL}}$. A column convects unless one of these holds:

| Skip condition | FLEXPART |
| --- | --- |
| the parcel is 0.9 K or more **colder** than the environment at ICB: $\Delta T_v\big\vert_{\mathrm{ICB}} \le -0.9\ \mathrm{K}$ | `TVP(ICB) <= TV(ICB) - DTMAX`, with CBMF = 0 (no memory) |
| $\mathrm{DTMIN} \le 0$, so the closure's $M_b$ (§4) would be zero | `CBMF > 0` |
| the parcel level has $T < 250$ K or $q \le 0$ (GLIDE's parcel is the surface level; FLEXPART's is NK) | `T(NK) < 250`, `Q(NK) <= 0` |
| the moist static energy has no minimum below the top two levels | `IHMIN >= NL - 1` |
| the LCL is outside 200–2000 hPa, there is none, or ICB is one of the top two levels | `PLCL`, `ICB >= NL - 1` |

and the cloud must have at least one level ($i_{\mathrm{LNB}} > i_{\mathrm{LCL}}$;
FLEXPART's INB is at least ICB + 1). There is **no CAPE floor and no
cloud-depth check**: FLEXPART has neither. CAPE now only enters the CAPE
diagnostic; the cloud, the closure and the matrix are as in §2, §4 and §5.
The test needs `closure: buoyancy`, since its DTMIN test is the closure's.

The checks are made in each column separately. A column that is skipped has no
convection in this meteorology update, and its particles are untouched. A
negative control test asserts it does **not** fire on a stable winter sounding
(`test_emanuel_does_not_fire_on_winter_inversion_column`).

**How often it fires** (decision 0034, measured on the Decision 6 study's
columns, 40 levels): in 49% (January) and 60% (July) of column-hours, against
FLEXPART's 56% and 61% on the same columns; 95–98% of its firing is where
FLEXPART fires, and it catches 87–93% of FLEXPART's. The firing count changes
by at most 3.5% between 23, 40 and 80 levels.

**The trigger before 0034** (`trigger: glide`, kept for A/B and bit-identical
to the earlier output) fired only when all of these held:

| Condition | Option |
| --- | --- |
| $i_{\mathrm{LCL}} \ge 0$ and $i_{\mathrm{LNB}} > i_{\mathrm{LCL}}$ | |
| $\mathrm{CAPE} \ge 50\ \mathrm{J\ kg^{-1}}$, the cloud's CAPE (§2) | `min_cape_j_kg` |
| the parcel 0.9 K or more **warmer** at the level LCL+1: $\Delta T_v\big\vert_{\mathrm{LCL}+1} \ge 0.9\ \mathrm{K}$ | `trigger_dtv_k` (or at `trigger_height_above_lcl_m` above the LCL) |
| cloud depth $\ge 500\ \mathrm{m}$ | `min_cloud_depth_m` |

Its ΔT_v test has the opposite sign and level to FLEXPART's, and FLEXPART has
neither the CAPE floor nor the depth check. Earlier versions of this page and
the code attributed it to Forster et al. (2007) Eq. 34; that could not be
checked, and it is not what FLEXPART does. It fired in 0.5–1.3% of
column-hours, and its firing count halved from 40 to 80 levels, because the
height of the level LCL+1 above the LCL depends on the grid.

---

## 4. Cloud-base mass flux

Since decision [0033](../dev/decisions/0033-buoyancy-closure-exact-exponential.md)
(28 Nov 2026) the cloud-base mass flux is FLEXPART's, in its steady state:

$$
M_b = k\ \max(\mathrm{DTMIN},\ 0),
\qquad
k = \frac{0.1\ \alpha}{\mathrm{DAMPS}} = \frac{0.1 \times 0.025}{0.3}
= 0.00833\ \mathrm{kg\ m^{-2}\ s^{-1}\ K^{-1}}
$$

$$
\mathrm{DTMIN} = \Delta T_v(\mathrm{LCL}) + 0.9\ \mathrm{K} + \overline{\Delta T_v}_{\text{sub-cloud}}
$$

where $\Delta T_v$ is the parcel's virtual-temperature excess over its
environment.
- $\Delta T_v(\mathrm{LCL})$ is interpolated linearly in pressure to the LCL, between the levels either side of it.
- $\overline{\Delta T_v}_{\text{sub-cloud}}$ is its layer-mass-weighted mean over the levels below the LCL.

This is the quantity Emanuel's scheme as FLEXPART 11 runs it relaxes towards (`convect` in
`conv_mod.f90`: `CBMF = (1 − DAMPS)·CBMF + 0.1·ALPHA·DTMA`, with ALPHA = 0.025,
and DAMPS = 0.3 whatever the time step). GLIDE computes DTMIN from its own surface
parcel (`cloud_base_buoyancy_excess_columns`).

**How it compares with FLEXPART.** In GLIDE's firing columns, FLEXPART's own
`convect`, run on the same ERA5 hours, gives a median CBMF of 0.0184 (January) and
0.0197 kg m⁻² s⁻¹ (July 2019); GLIDE gives 0.0179 and 0.0188. Over every column
where both are positive the correlation is 0.95 and the median ratio is 1.01–1.02.

**No memory.** FLEXPART remembers CBMF per column between calls. GLIDE
uses the steady state, a function of the meteorology alone, because a
backward model visits the hours in reverse (decision 0033).

The transport per unit time is $M_b$. Per firing column it no longer depends on the
vertical grid, nor (on the same meteorology) on the met cadence, as it did under
the closure below (§6).

**The closure before decision 0033** (`closure: cape_velocity`, kept for A/B) was

$$
M_b = c_{\text{closure}} \  \rho_{\mathrm{LCL}} \  \min\left(\sqrt{2\ \mathrm{CAPE}},\ 5\ \mathrm{m\ s^{-1}}\right),
\qquad c_{\text{closure}} = 0.03,
$$

applied as one transfer of $M_b\,\Delta t_{\mathrm{met}}$ with the CFL scale of §5
(`application: cfl_scaled`).
- The cap binds in every firing column, so $M_b = 0.15\,\rho_{\mathrm{LCL}} \approx 0.17$ kg m⁻² s⁻¹, about ten times FLEXPART's.
- The CFL scale then binds in more than 99% of firing columns: an event moves $0.9\,m_{i_{\mathrm{LCL}}-1}$, set by the thickness of the layer below cloud base and by the cadence (decisions 0031 and 0033).

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

**CFL cap (only with `application: cfl_scaled`, before decision 0033).** With the
default exponential application (§6) the matrix is a rate and is not scaled.
In the single-transfer application the whole matrix is scaled by a **single scalar** if any
layer would shed more than 90% of its mass in one event. One scalar, so
non-divergence survives, and so the "stay" diagonal of §6 stays non-negative.
With the closure before 0033 this cap bound in almost every firing column, at the
layer just below the LCL, so it set how much one event moved (§4). FLEXPART has no
counterpart: its `convect` flags a CFL violation (`IFLAG = 4`), but `calcmatrix`
uses the matrix anyway, and it is called every `LSYNCTIME` (900 s by default)
rather than once per met window.

The matrix is identically zero when there is no cloud.

---

## 6. Particle redistribution, and the backward transpose

**The transition over an interval (decision 0033).** With the default
`application: exponential`, $\mathrm{fmass}$ is a rate (kg m⁻² s⁻¹, built from
$M_b$ itself), and its generator is

$$
Q_{ij} = \frac{\mathrm{fmass}[j,i]}{m_i}\ (i \ne j), \qquad
Q_{ii} = -\sum_{j \ne i} Q_{ij}
$$

for GLIDE's backward runs: the transposed column, weighted by layer mass, as
below. The forward generator is $\mathrm{fmass}[i,j]/m_i$.

The transition over the met interval is the exact exponential
$P = \exp(\Delta t_{\mathrm{met}}\,Q)$ (`generator_exponential`):
- It uses uniformised scaling and squaring in float64, with static shapes and no host synchronisation.
- It is non-negative and row-stochastic by construction.
- Because $\mathbf{m}^{\mathsf T}Q = 0$, the layer masses are stationary.
- The backward $P$ equals $D^{-1}P_{\mathrm{fwd}}^{\mathsf T}D$ with $D = \mathrm{diag}(m)$: the time reversal of the forward transition.
- For a short interval it reduces to the one-step matrices below.

Accuracy against `scipy.linalg.expm` is ≤ 10⁻¹¹ up to a Courant number (Δt × the largest
fractional exchange rate of a layer) of about 1,700 on 23–137 levels. Columns
above the squaring cap of 2,048 are counted in `runtime.convection` of
run_metadata.json and logged; the buoyancy closure gives at most 5.6 on ERA5
over Europe. Because $\exp(3\Delta t\,Q) = \exp(\Delta t\,Q)^3$, on the same
meteorology the transport does not depend on the met cadence (a coarser cadence
still samples a changing atmosphere less often). Because no layer's mass caps it, it does not
depend on the vertical grid either: per firing column the median changes by
less than 20% between 23, 40 and 80 levels, against ×2 per doubling before.

**The one-step matrices** (`application: cfl_scaled` and `substeps`, and
FLEXPART's `calcmatrix`/`redist`):

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

4. **Steady-state buoyancy closure without memory** (decision 0033). This is
   FLEXPART's relaxed CBMF at steady state, from GLIDE's surface parcel. Two
   differences remain:
   - FLEXPART's parcel starts at the level of maximum moist static energy (NK), and its updraft draws only from that level, with turbulence refilling it. GLIDE entrains from the whole sub-cloud layer by mass.
   - FLEXPART's onset and decay lag by its relaxation time, about 40 min at its 900 s call.
   The closure before 0033 (capped $\sqrt{2\,\mathrm{CAPE}}$) remains selectable.

5. **Adjacent-layer subsidence.** The environmental return flux moves mass one
   layer down per event, while the updraft is non-local. Physically apt — a fast
   coherent updraft against slow broad subsidence — but it means the descent of
   non-displaced environmental air is a random walk rather than a prescribed
   velocity.

6. **Convection fires once per met window**, over that window's own duration,
   which the reader sets to the store's time step
   (`reader.met_interval_seconds`). Since decision 0033 the transition over the
   window is exact in time (§6), so the cadence changes only how often the
   meteorology is sampled. FLEXPART is different: it calls every 900 s, and its
   relaxed CBMF depends on the call interval.

7. **Deferred, deliberately (decisions 0033 and 0034).** These differences from
   FLEXPART's full scheme are known, and each is a study of its own:
   - **Detrainment profile.** In January GLIDE lifts about 10× less sub-cloud air above 3 km than FLEXPART in the same columns.
   - **Downdrafts** (item 3).
   - **Cloud top.** GLIDE uses the first LNB; FLEXPART uses the highest.
   - **Parcel origin** (NK). The trigger's parcel is GLIDE's surface parcel, not FLEXPART's NK.
   - **Trigger memory.** FLEXPART skips the ΔT_v test of §3 in a column that was convecting at its previous call; GLIDE has no memory (decision 0033).

   With FLEXPART's trigger (0034) GLIDE fires about as often as FLEXPART, and its domain-total cloud-base flux is 91–96% of FLEXPART's, but it lifts only 7% (January) and 40% (July) as much sub-cloud air above 3 km. That gap lies in the cloud top and the detrainment profile.

---

## 8. Configuration

```yaml
convection:
  scheme: emanuel_reduced     # or "none" (default; bit-equivalent to no convection)
  emanuel:
    closure: buoyancy         # M_b = buoyancy_mass_flux_per_k · max(DTMIN, 0) (decision 0033)
    buoyancy_mass_flux_per_k: 0.00833   # 0.1·ALPHA/DAMPS, FLEXPART's 0.025 and 0.3
    application: exponential  # exact exp(Δt·Q) over the met interval (§6)
    trigger: flexpart_cloud_base   # FLEXPART's cloud-base test, no memory (decision 0034, §3)
```

A YAML with no `convection:` block produces output identical to a run with no
convection at all.

**The trigger before decision 0034**, for A/B comparison: `trigger: glide`,
with its options (§3):

```yaml
    trigger: glide
    trigger_dtv_k: 0.9        # parcel ΔT_v at the level LCL+1, K
    min_cape_j_kg: 50.0       # CAPE floor
    min_cloud_depth_m: 500.0  # minimum cloud depth
    trigger_height_above_lcl_m: null    # option: test ΔT_v this far above the LCL instead
```

It reproduces the earlier output bit for bit. These four options apply only to
`glide`. Setting one under the default trigger is not an error, but the run
logs a warning and records it in `runtime.physics_warnings` of
run_metadata.json. The shipped configs no longer set them.

**The scheme before decision 0033**, for A/B comparison:
`closure: cape_velocity` (with `closure_c: 0.03`, `updraft_w_max_m_s: 5.0`,
which apply only to it), `application: cfl_scaled` and `trigger: glide` (the
default trigger needs `closure: buoyancy`, and refuses `cape_velocity` with a
message saying so). These reproduce the earlier output bit for bit. `application: substeps` with `n_substeps` applies
n CFL-capped transfers per interval instead; it is approximate and is kept for
comparison. `closure_c` and `updraft_w_max_m_s` apply only to
`cape_velocity`, and `n_substeps` only to `substeps`. Setting one while another
closure or application is selected is not an error, but the run logs a warning
and records it in `runtime.physics_warnings` of run_metadata.json. The shipped
configs no longer set `closure_c`.

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
