# 0031 — Convection: CAPE is the buoyant energy of the diagnosed cloud, LCL to cloud top

Claims are labelled *measured* (with the file, under `outputs/agents/CAPE/` in
the main checkout unless a test is named), *derived* (follows from the code or
the equations) or *argued*. Task CAPE, 28 Nov 2026, from an external review
("GLIDE turbulence physics discussion notes", section "CAPE above the
diagnosed cloud top").

**Context.** The reduced Emanuel scheme ([0007](0007-reduced-emanuel-convection.md),
[0023](0023-convection-per-column.md)) fires in a column only if, among other
checks, its CAPE is at least `min_cape_j_kg` (50 J kg⁻¹). `docs/convection.md`
defined CAPE as $\int_{\mathrm{LFC}}^{\mathrm{LNB}} gB\,\mathrm{d}z$, with the
LNB the cloud top: the first level above the LCL where the parcel's buoyancy
$B$ is no longer positive. Both implementations, the single-column reference
`compute_lcl_lnb_cape` and the batched `compute_lcl_lnb_cape_columns`, summed
$g\max(B, 0)\,\mathrm{d}z$ from the LCL level to the **top of the column**
(*derived*, read from the code at 7b6c5eb). A second buoyant layer above a
stable layer over the cloud therefore counted towards the trigger of the lower
cloud, although the mass-flux matrix only ever spans [LCL, LNB]. The review's
case: 35 J kg⁻¹ in the cloud plus 30 J kg⁻¹ in a layer above its top gave
65 J kg⁻¹ and the column fired; under the documented definition it has
35 J kg⁻¹ and does not (*measured*, `tests/test_cape_cloud_top.py` against
main fails with `[False, True] == [False, False]`,
`logs/new_tests_on_main.log`).

**The lower bound** was already the LFC to within a level (*derived*). The sum
starts at $i_{\mathrm{LCL}}$, the first level above the LCL (FLEXPART's `ICB`).
The LNB is the first level above $i_{\mathrm{LCL}}$ with $B \le 0$, so every
level strictly between them is buoyant and only $i_{\mathrm{LCL}}$ itself can
carry negative buoyancy; the LFC is $i_{\mathrm{LCL}}$ or
$i_{\mathrm{LCL}} + 1$. Convective inhibition (negative $B$ in the dry ascent
and at $i_{\mathrm{LCL}}$) is left out, not subtracted. A parcel negatively
buoyant at $i_{\mathrm{LCL}} + 1$ never fires, since its LNB is then
$i_{\mathrm{LCL}} + 1$ and the $\Delta T_v \ge 0.9$ K check at that level fails;
there is no elevated LFC in GLIDE. Not changed here.

**Decision.**
1. **CAPE is summed over the diagnosed cloud only**: $\sum_{k=i_{\mathrm{LCL}}}^{i_{\mathrm{LNB}}}
   g\max(B_k, 0)\,\mathrm{d}z_k$. Buoyancy above the LNB does not count.
   Within the cloud the integrand is positive except possibly at
   $i_{\mathrm{LCL}}$, so this is the documented
   $\int_{\mathrm{LFC}}^{\mathrm{LNB}} gB\,\mathrm{d}z$ in the model's level
   discretisation.
2. **Identical in both paths.** The single-column function zeroes the levels
   above its LNB; the batched function computes the LNB first and masks with
   `(k >= i_lcl) & (k <= i_lnb)` — a `torch.where` on tensors it already has,
   static shapes, no host synchronisation. Columns with nothing buoyant above
   their LNB get bit-identical CAPE (the summed tensor is unchanged).
3. **Unchanged:** the cloud top (first LNB), the lower bound and the treatment of
   CIN, the other trigger checks, the closure and the matrix.

**Rationale** (*argued*). CAPE in the trigger stands for the energy available to
the cloud the scheme then builds. GLIDE's cloud is [LCL, LNB]: the detrainment
profile, the matrix and the cloud-depth check all stop at the first LNB. Energy
in a detached layer above is not available to a parcel that stops at the LNB, so
counting it makes the trigger describe a different cloud from the one
transported. It also made the trigger depend on the column's top (`alt_max_m`,
the vertical grid), which the matrix does not.

**What FLEXPART does** (*derived*, FLEXPART 11 `src/conv_mod.f90`, `convect`, at
commit d4e0a86 of the local checkout; Forster et al. (2007) itself was not
available to check):
- **The parcel starts at the level of maximum moist static energy** (`NK`,
  searched from `MINORIG` up to the level of minimum moist static energy),
  not at the surface as in GLIDE.
- **No CAPE trigger.** Convection is skipped when: the parcel level has
  $T < 250$ K or $q \le 0$; moist static energy increases monotonically with
  height (no minimum below the top, `IHMIN = NL - 1`); `PLCL` is outside
  200–2000 hPa; the first level above the LCL is too high (`ICB >= NL - 1`);
  or the column was not convecting and the parcel at `ICB` is at least
  `DTMAX` = 0.9 K *colder* than the environment
  (`TVP(ICB) <= TV(ICB) - DTMAX`). Otherwise the cloud-base mass flux is
  relaxed, `CBMF = (1 - DAMPS)·CBMF + 0.1·ALPHA·DTMA`, and the column convects
  while `CBMF > 0`.
- **CAPE sets the cloud top only.** From `ICB + 1` upwards FLEXPART accumulates
  the *signed* buoyancy (negative layers subtract), takes `INB` as the highest
  level where that running sum is positive, `INB1` as one above the highest
  buoyant level, and the cloud top as `max(INB, INB1)`: the **highest** level of
  neutral buoyancy below about 50 hPa, not the first. A second buoyant layer
  therefore deepens a FLEXPART cloud; it never changes whether it convects.

So neither the old nor the new GLIDE CAPE matches FLEXPART, which has no CAPE
floor. This decision makes GLIDE consistent with its own documentation and its
own cloud; it does not move it towards FLEXPART, and it should not (AGENTS.md).

**How often it matters** (*measured*, `trigger/jan_compare.md`,
`trigger/jul_compare.md`; scripts `scripts/validation/cape/`). The configured
scheme's `column_state` on every column of the reference config's met domain
(`configs/ab_multisite_perf.yaml`, EUROPE ERA5, 150,974 columns × 40 levels),
nine hours over its backward period (28 Dec 2023 – 2 Jan 2024), and of the
EUR store in summer (45,241 columns, eight hours 25 Jun – 12 Jul 2019), with
main (7b6c5eb) and this decision's code, float32 on CPU:

| | columns × hours | firing, main | firing, this decision | stop firing | start firing | Σ M_b cos φ |
| --- | --- | --- | --- | --- | --- | --- |
| January, EUROPE domain | 150,974 × 9 | 7,858 | 7,243 | 615 (7.8%) | 0 | −9.7% (−0.7% to −21.9% by hour) |
| January, UK and Ireland box (lon −15…5, lat 48…60) | 3,969 × 9 | 563 | 561 | 2 (0.4%) | 0 | −0.4% |
| summer 2019, EUR domain | 45,241 × 8 | 4,740 | 4,706 | 34 (0.7%) | 0 | −0.8% |
| summer 2019, UK and Ireland box | 3,969 × 8 | 128 | 122 | 6 (4.7%) | 0 | −4.9% |

Σ M_b cos φ is the nominal cloud-base flux summed over columns with an area
weight, before the CFL scale of the matrix (below). No column starts firing
(CAPE can only fall), and in columns that fire under
both $M_b$, the LCL, the cloud top and so the matrix are unchanged (max
$|\Delta M_b|$ = 0 in every hour; the matrix depends only on those and the
pressures, *derived*). The transport therefore changes only in the columns
that stop firing. CAPE changes in 17% (January) and 30% (summer) of all
columns, almost all of which do not fire either way.

**The columns that stop firing have a real stable layer, not a one-level
dip** (*measured*, `trigger/jan_gap.log`, `trigger/jul_gap.log`, January;
summer in brackets). Their cloud is shallow — depth median 670 m (800 m),
just over the 500 m floor — and capped by a stable gap of median 6 (3.5)
levels and 1.4 (1.1) km, with a most negative $T_v$ excess of median −1.5 K
(−0.9 K) and signed energy median −49 J kg⁻¹ (−20); only 6% (0%) have a gap
whose deficit never exceeds 0.1 K. Above the gap they have median 254
(137) J kg⁻¹ of positive energy and buoyant levels up to a median 5.6 (8.6) km.
Under main these columns fired with a cloud reaching only the first LNB
(median top 1.4 km) on the strength of energy in a layer that cloud never
reaches; with this decision they do not fire. FLEXPART would convect them,
with a cloud top at the highest buoyant level (below). Which of the two is
better for these columns is part of the cloud-top question below, not of this
fix.

**The trigger is now sensitive to a weak one-level dip** (*measured*,
`review/edge_cases.log`, and `test_weak_one_level_dip_ends_the_cloud` in
`tests/test_cape_cloud_top.py`). Because the cloud top is the first level with
$B \le 0$, one level that is barely non-buoyant ends the cloud, and now also
ends the CAPE sum. In the review's example a −0.15 K dip ($B$ = −0.0005) at one
level cut CAPE from 82 to 38 J kg⁻¹, which turns a firing column into a
non-firing one at the 50 J kg⁻¹ floor. Before this decision the dip already
ended the cloud (and the matrix) there, but not the CAPE. In the real columns
that stop firing such dips are rare (6% in January, 0% in summer have a gap
whose deficit never exceeds 0.1 K; above), but a tolerance on the cloud top
(or FLEXPART's signed sum) belongs to the cloud-top question.

**Rejected alternatives.**
- *Keep the column-top sum and fix the docs.* The trigger would keep describing
  a cloud the scheme does not build, and depend on the vertical extent of the
  met domain.
- *FLEXPART's cloud top (highest LNB) with signed CAPE.* A larger change: it
  deepens clouds and changes where the matrix detrains, not just the trigger.
  It belongs with the closure question below, not in a bound fix.
- *Subtract CIN (signed integral from $i_{\mathrm{LCL}}$).* Within [LCL, LNB]
  only the $i_{\mathrm{LCL}}$ level can be negative, and that level's layer is
  partly below the LCL; subtracting it would change many columns' CAPE by a
  discretisation artefact for no physical gain.

**Open question for Matt before the physics freeze, not changed: the updraft
cap binds in every firing column, and the CFL scale behind it sets the
transport.** `cloud_base_mass_flux_columns` uses
$w = \min(\sqrt{2\,\mathrm{CAPE}}, 5\ \mathrm{m\,s^{-1}})$. The cap is reached at
12.5 J kg⁻¹ and the trigger needs 50, so every firing column has
$M_b = 0.03 \times 5 \times \rho_{\mathrm{LCL}} = 0.15\,\rho_{\mathrm{LCL}}$
(*derived*); CAPE only switches convection on or off. Consequently this
decision changes $M_b$ only by switching columns off: in columns that fire
under both definitions $M_b$ is unchanged (*derived*; *measured* max
$|\Delta M_b|$ = 0 in every window above).

On the same hours (*measured*, `trigger/jan_closure.log`,
`trigger/jul_closure.log`, this decision's code, all firing columns: 7,243 in
January, 4,706 in summer): $\sqrt{2\,\mathrm{CAPE}}$ is 10–58 m s⁻¹ (median
12.3) in January and 10–80 m s⁻¹ (median 22.6) in summer, so the cap binds in
100% of firing columns, as derived. $M_b$ is 0.150–0.191 kg m⁻² s⁻¹ in
January (0.092–0.179 in summer, where $\rho_{\mathrm{LCL}}$ falls to
0.62 kg m⁻³ for high cloud bases). **The CFL scale of the matrix binds as
well**, in 100% (January) and 99.4% (summer) of firing columns, with median
scale 0.13 (0.25): one event moves an eighth to a quarter of $M_b\,\Delta t$,
and the updraft entrains a median 13% (12%) of the sub-cloud air mass per hourly
event, range 9–30%.

**In the sampled columns the limiting layer was the one just below the LCL, so
the mass per event was exactly $0.9\,m_{i_{\mathrm{LCL}}-1}$.** This holds
under two conditions: the CFL scale binds, and layer $i_{\mathrm{LCL}} - 1$
has the largest fractional outflux of any row. The scale is set by the largest
outflux ratio over *all* rows, so a thin cloud layer can limit the column
instead: on a constructed grid (pressures 100000, 98000, 96000, 95950, 95900,
95850 Pa; LCL index 2, cloud top 5; unscaled event mass 600 kg m⁻²) row 3
limits and the boundary-layer-to-cloud transfer is 9.18 kg m⁻², not the
183.55 kg m⁻² the formula gives (external review, 28 Nov 2026). Layer $i_{\mathrm{LCL}} - 1$
sheds its own entrainment plus the subsidence it passes down,
$e_{i_{\mathrm{LCL}}-1} + \Phi_{i_{\mathrm{LCL}}-2} = \sum_{i<i_{\mathrm{LCL}}} e_i
= M_b\,\Delta t$, the whole updraft (*derived*, §5 of `docs/convection.md`),
so whenever $M_b\,\Delta t > 0.9\,m_{i_{\mathrm{LCL}}-1}$ the scale makes the
entrained mass $0.9\,m_{i_{\mathrm{LCL}}-1}$. The independent review measured
it: in 3,069 firing columns (three January hours, 40 levels) the layer of
largest outflux is $i_{\mathrm{LCL}} - 1$ in every column, and entrained mass
/ $0.9\,m_{i_{\mathrm{LCL}}-1}$ is 1.000 at every percentile; the mass per
event is 32–171 kg m⁻², median 79 (*measured*, `review/cfl_jan40.log`). The
transport per event therefore depends on the **vertical grid** (the mass of
the layer below the LCL) and on the **met cadence** (one event per window),
not on CAPE, $M_b$ or `closure_c`. On 80 levels, same hours, the median mass
per event falls to 37 kg m⁻² (21–77) and firing columns fall from 3,069 to
1,695 (*measured*, `review/cfl_jan80.log`). With `closure_c` divided by 10
the scale still binds in 23% of firing columns at 40 levels and 97% at 80
(same files). This is an open question about the closure, to be settled
before the physics freeze; the closure is not changed here.

`CFL_MAX_OUTFLUX_FRAC`'s code comment said FLEXPART enforces an equivalent
limit; it does not, and the comment is corrected. FLEXPART's `convect` sets
`IFLAG = 4` when its own CFL test fails (`conv_mod.f90` lines 1901, 1929), but
`calcmatrix` accepts `IFLAG` 1 or 4 and updates `fmassfrac` regardless, and
convection is called every `LSYNCTIME` (900 s by default) rather than once per
met window (*derived*, same file and `readoptions_mod.f90`).

FLEXPART's closure does not use CAPE either (*derived*, same file): `CBMF`
relaxes each call towards $0.1\,\alpha\,\Delta T/\mathrm{DAMPS}$ with
$\alpha$ = 0.025, `DAMPS` = 3 × `DAMP` = 0.3 (FLEXPART fixes `DELT0 = DELT/3`),
and $\Delta T$ = `DTMIN` = the parcel–environment $T_v$ difference interpolated
to the LCL + 0.9 K + the mean sub-cloud parcel excess; it is remembered per
column between calls (`cbaseflux`). Its steady state is about
$0.0083\ \mathrm{kg\,m^{-2}\,s^{-1}}$ per kelvin of `DTMIN` (*derived*), so
0.008–0.017 for 1–2 K: an order of magnitude below GLIDE's nominal $M_b$
(0.15–0.19), but of the order of what GLIDE actually moves after the CFL scale
(median $0.13 \times 0.177 \approx 0.02$ in January). This is an
order-of-magnitude comparison only (*argued*): FLEXPART's matrix also carries
the entrainment and detrainment mixing `MENT` beyond `CBMF`, and `DTMIN` was not
computed on these columns. The "0.05–0.5 kg m⁻² s⁻¹" range
`docs/convection.md` attributes to FLEXPART could not be traced to a source.

Options for Matt: (a) keep the cap and the CFL limit as they are: CAPE is an
on/off switch and the transport per event is $0.9\,m_{i_{\mathrm{LCL}}-1}$,
set by the vertical grid and the met cadence (the status quo, now documented);
(b) lower `closure_c` or the cap until the CFL scale rarely binds, so that
$M_b$ (and, with a higher cap, CAPE) sets the transport (a tenfold lower
`closure_c` is not enough on 80 levels, above);
(c) a FLEXPART-like closure on the buoyancy at cloud base with relaxation
memory, a bigger change and a Tier 2 comparison question. The cloud-top
question goes with (c): FLEXPART's highest LNB would make the columns this
decision switches off convect, and deeply. Whether any of it matters for
footprints depends on how often particles meet firing columns. In the CPU smoke
run (`configs/smoke_mhd_single_release.yaml`: Mace Head, 1 Jan 2024, one day
back, seed 42) no particle was ever in a firing column, under main or this
decision, and the footprints and endpoint particles are bit-identical, so 0–40 m
and total change by 0 (*measured*, `smoke/compare.log`, `smoke/*-s42.log`).
That is a January Atlantic site; the summer UK box above (4.7% of firing
columns switched off) is where an effect would first show, and no summer
footprint run was made.

**Status.** In force from task CAPE (PR to be merged by Matt). Amends the CAPE
of [0007](0007-reduced-emanuel-convection.md) and
[0023](0023-convection-per-column.md); see `docs/convection.md` §2–§4. The
trigger's sign and level relative to FLEXPART (GLIDE requires the parcel to be
0.9 K *warmer* at $i_{\mathrm{LCL}} + 1$; FLEXPART lets it be up to 0.9 K
*colder* at $i_{\mathrm{LCL}}$) is a further difference found here, recorded
for Matt and not changed; the code attributes GLIDE's form to Forster (2007)
Eq. 34, which could not be checked.
