# Physics Verification Spec — Backward-in-Time Lagrangian Particle Dispersion Model

## Purpose of this document

This is a specification handed to **Claude Code** to audit an existing Python LPDM
implementation against the established physics of Lagrangian stochastic (LS) dispersion
models. The model's intended use is **backward-in-time simulation of inert tracer
transport to compute regional (10–1000 km) source–receptor "footprints" for greenhouse
gas emissions inference.**

The model already exists: scaffolding, integrator, and a turbulence scheme are in place.
**Your job is to verify, not to rebuild.** Read the existing code first, infer what it
actually does, and compare it against the physics below. Report discrepancies. Propose
fixes only where the code genuinely departs from correct physics, and explain the physical
consequence of each discrepancy (especially: does it bias the footprint systematically, or
only add noise?).

### How to approach the audit (read this first)

1. **Do not assume the equations are the source of bugs.** In this class of model, the
   governing SDE is usually written correctly; the errors hide in (a) discretization,
   (b) boundary handling, (c) the forward→backward transformation, and (d) consistency
   between the turbulence parameterization and the drift term. Audit all four even though
   the drift equation is the stated priority.
2. **The unifying correctness criterion is the well-mixed condition (WMC):** an ensemble
   of particles initially distributed according to the air's density-weighted velocity PDF
   must *remain* so distributed for all time, in the absence of sources. Almost every bug
   worth finding manifests as a WMC violation. Treat "does this preserve the WMC?" as the
   question behind every check.
3. **Determine, don't assume, the turbulence PDF.** First establish whether the vertical
   velocity is modelled as Gaussian everywhere or with a skewed convective-boundary-layer
   (CBL) PDF. The correct form of the drift term depends entirely on this, so resolve it
   before checking the drift.
4. **Report findings as: (location in code) → (what it does) → (what physics says) →
   (consequence for footprints) → (proposed fix, if warranted).**

### Notation used in this document

- `x = (x, y, z)` particle position; `z` vertical.
- `u = (u, v, w)` particle velocity; `w` vertical.
- `U_i` mean (resolved) wind; `u'_i = u_i − U_i` turbulent fluctuation.
- `σ_w`, `σ_u`, `σ_v` velocity standard deviations (turbulence intensities).
- `τ_L` (component-wise `τ_Lw`, etc.) Lagrangian decorrelation timescale.
- `ρ` air density; `g_a` the density-weighted Eulerian velocity PDF.
- `Δt` integration time step; `dξ` a Wiener increment, mean 0, variance `dt`.
- `ε` TKE dissipation rate; `C_0` the Lagrangian-structure-function (Kolmogorov) constant.

---

## PRIMARY CHECKS — the Langevin / well-mixed drift

### P1. The generalized Langevin equation has the correct structure

The model must integrate a first-order LS model of the form (Thomson 1987):

```
du_i = a_i(x, u, t) dt + b_ij(x, u, t) dξ_j
dx_i = u_i dt
```

Check:
- **Velocity is a state variable, not just position.** A *zeroth*-order model (random
  displacement / random walk in position only, `dx_i = ... dξ`) is a different and weaker
  model — it cannot represent the near-source/near-field regime and is only valid at
  travel times ≫ τ_L. Confirm which the code implements. For regional footprints the
  near-field matters near the receptor and near ground, so a first-order (velocity-memory)
  model is expected. Flag if the code is secretly a random-displacement model.
- **The random forcing is Gaussian.** Thomson (1987, Appendix A) proves that a continuous
  Markov process in (x, u) with the right local structure *must* have Gaussian increments.
  Non-Gaussian forcing is either non-existent or implies discontinuous trajectories. If the
  code draws `dξ` from anything other than a Gaussian, flag it.
- **`dξ` has variance `Δt`, not `1` or `Δt²`.** A common bug: scaling the random term by
  `Δt` or `sqrt(dt)` inconsistently. The increment must be `b · sqrt(Δt) · N(0,1)` where
  `N(0,1)` is standard normal. Verify the `sqrt(Δt)` scaling explicitly.

### P2. The diffusion coefficient b is tied to the inertial subrange

The standard, inertial-subrange-consistent choice (Thomson 1987 §4.1; Flesch et al. 1995
Eq. 4) is:

```
b_ij = δ_ij · sqrt(2 σ_w² / τ_L)        (component form, diagonal)
equivalently   b² = 2 σ_w² / τ_L
```

This follows from `2⟨B⟩ = C_0 ε` with the identification that gives `b² = σ²·(2/τ_L)`.
Check:
- `b` is **diagonal** (no off-diagonal forcing) unless the model deliberately carries
  cross-correlated forcing — most regional models neglect cross-correlations (Stohl &
  Thomson 1999 cite Uliasz 1994 that this is minor at large scale). If off-diagonal terms
  exist, check they're intentional and consistent.
- The relationship `b² = 2σ²/τ_L` holds **component-wise** with the *component* σ and τ_L.
  A frequent bug is mixing a vertical σ_w with a horizontal τ_L, or using a single scalar
  σ for all three components.
- If the code uses a finite-`Δt` correction to `b` (e.g. Wilson & Flesch 1993 Eq. 5:
  `b² = (σ²/Δt)[1 − (1−Δt/τ)²]`), confirm it reduces to `2σ²/τ_L` as `Δt/τ_L → 0`. This
  correction matters only when `Δt` is not ≪ τ_L; see T-checks.

### P3. The drift term a satisfies the well-mixed condition

This is the heart of the audit. The drift is **not** free — given the turbulence PDF `g_a`
and the diffusion `b`, the WMC *determines* the drift (uniquely in 1-D; up to a divergence-
free term in higher dimensions). The model must use the WMC-consistent drift, not an ad hoc
"relaxation toward the mean" term.

**Step 1 — establish g_a.** Determine from the code whether the vertical velocity PDF is
Gaussian or skewed. Inspect the turbulence scheme: does it compute only σ_w (Gaussian), or
also a third moment / skewness / bi-Gaussian parameters (skewed CBL)?

**Step 2a — if Gaussian (expected for a regional model):** the WMC-consistent 1-D vertical
drift is (Thomson 1987 Eq. 30–31; the form actually used in FLEXPART-type models):

```
a = −w/τ_Lw  +  (1/2)(∂σ_w²/∂z)(1 + w²/σ_w²)      [+ density term, see S-checks]
```

Equivalently, models often integrate the **`w/σ_w` formulation** (Wilson et al. 1983;
Stohl & Thomson 1999 Eq. 4), which cancels an awkward term and is numerically cleaner:

```
d(w/σ_w) = −(w/σ_w)(dt/τ_Lw) + (∂σ_w/∂z) dt  [+ density term] + sqrt(2/τ_Lw) dξ
```

Check carefully:
- **The drift is more than `−w/τ`.** The `−w/τ` term alone (pure relaxation) is only correct
  in *homogeneous* turbulence. In the real boundary layer σ_w varies with height, and the
  **`∂σ_w²/∂z` ("drift-correction" / gradient) term is mandatory.** Its absence is a classic,
  serious bug: it produces a spurious accumulation of particles in low-turbulence regions
  (Wilson, Legg & Thomson 1983 is the canonical reference for this term). A model missing it
  will systematically misplace particles vertically and bias surface footprints. **This is
  the single highest-value thing to check.**
- The gradient term has a **`(1 + w²/σ_w²)` velocity dependence** in the Gaussian case
  (not just a constant `∂σ_w²/∂z`). Confirm the `w²/σ_w²` factor is present if the code
  uses the raw-`w` form (Eq. 30). If it uses the `w/σ_w` form, the equivalent term is
  `∂σ_w/∂z` (note: σ_w, not σ_w²) — verify the algebra matches one consistent formulation,
  not a mix of the two.
- **Sign of the drift correction.** It should push particles *down the gradient of σ_w*
  (toward lower turbulence) in the deterministic mean — verify the sign by checking that an
  initially uniform-in-z ensemble stays uniform (the WMC test, see V-checks).

**Step 2b — if skewed CBL:** the drift is considerably more complex (Luhar & Britter 1989;
Thomson 1987 §5.2 gives a worked skewed example). Do **not** approximate it as the Gaussian
drift with a skewed PDF bolted on — the WMC-consistent skewed drift must be derived from the
specific `g_a` (e.g. a bi-Gaussian, Baerentsen & Berkowicz 1984). If the code uses a skewed
PDF, verify the drift was derived consistently *from that same PDF* and not copied from a
Gaussian model. **Consistency between the PDF assumed in `g_a` and the PDF implied by the
drift is itself a thing to check** — a model can carry a skewed σ-scheme but a Gaussian drift,
which violates the WMC.

> **Note on scope for regional footprints:** Thomson (1987 §5.2) found that ground-level
> concentration is relatively *insensitive* to skewness (even though the dispersion aloft is
> sensitive). For a surface-flux footprint model, a Gaussian scheme is often a defensible
> simplification. If the code is Gaussian, this is likely fine — flag it as a modelling
> choice to be aware of, not a bug.

### P4. The turbulence parameterization is internally consistent with the drift

The drift term consumes σ_w(z), τ_L(z), and their gradients. Check that:
- `∂σ_w/∂z` (or `∂σ_w²/∂z`) used in the drift is computed **consistently** with the σ_w(z)
  profile used elsewhere — i.e. it is the actual analytic or numerical derivative of the
  *same* profile, not a separately-parameterized gradient. A mismatch here silently breaks
  the WMC.
- τ_L is **positive and bounded away from zero** everywhere (a vanishing τ_L blows up the
  drift and the `b` term). Many schemes impose a floor (e.g. Stohl & Thomson use floors of
  ~10 s on τ_L). Check for such a floor; its absence near the surface or PBL top is a
  rogue-trajectory risk.
- The σ and τ_L formulae are stability-dependent (functions of L, the Obukhov length, and
  u_*, w_*) and switch correctly between stable/neutral and unstable regimes. (Reference
  surface-layer forms: Flesch et al. 1995 Appendix b; Hanna 1982.)

---

## SECONDARY CHECKS — these gate WMC correctness even if the drift equation is right

### S. Density handling (Stohl & Thomson 1999)

Standard models enforce the WMC in *Cartesian* coordinates (uniform in z), but air density
falls with height; in a deep PBL ρ at the top can be >20% below the surface. Neglecting this
biases surface concentrations by ~10% — and for a backward footprint that bias goes straight
into the inferred flux.

Check:
- Is there a **density-correction term** in the vertical drift? In the `w/σ_w` formulation
  it is `+ (σ_w/ρ)(∂ρ/∂z) dt` (Stohl & Thomson 1999 Eq. 4). Its absence is a real,
  quantifiable bias for a regional model spanning deep boundary layers.
- ρ and ∂ρ/∂z must be interpolated to the particle position.
- Note this is *physically the mean pressure-gradient force* — it should not be confused
  with, or double-counted against, the σ_w gradient term.

### B. Boundary conditions (Wilson & Flesch 1993)

The footprint is dominated by near-ground residence time, so the ground boundary is where
correctness matters most.

Check:
- **Reflection scheme.** Perfect/smooth-wall reflection (`z → −z` or `2z_R − z`, and
  `w → −w`) is WMC-exact **only** where the velocity PDF is symmetric and homogeneous over
  one step. Confirm reflection is applied in a near-ground layer where σ_w is taken
  ~constant and skewness ~0 (the standard "unresolved basal layer" device). If reflection
  is applied where σ_w has a strong gradient or the PDF is skewed, it violates the WMC.
- **In a model carrying u–w correlation, BOTH `w` and the correlated `u` fluctuation must be
  reversed on reflection** (Wilson & Flesch 1993 §7a) — reversing only `w` flips the sign of
  ⟨uw⟩ and breaks the WMC. Check this if cross-correlations are present.
- **Reject "artificial unattainability"** — schemes that clamp a particle to `z = 0` or add
  an extra Δt restriction to prevent crossing. Wilson & Flesch show these *always* violate
  the WMC. Look for any such clamping logic.
- PBL-top boundary: same logic. Check what happens to particles reaching the PBL top /
  model lid.

### T. Time-stepping and numerics (Thomson 1987 App. B; Wilson & Flesch 1993 App.; Flesch et al. 1995 §3a)

- **Adaptive Δt tied to local timescales.** Δt must be small relative to τ_L *and* to the
  scale over which σ_w changes. Typical constraint: `Δt = min(c·τ_L, c/|∂σ_w/∂z|·..., c·z_i/|w|)`
  with `c ~ 0.05` (Stohl & Thomson 1999), or `Δt ≈ 0.025 τ_L` (Flesch et al. 1995). Check
  for such a constraint; a fixed Δt that can exceed τ_L near the surface is a bug.
- **The Δt-bias.** Even with correct drift and reflection, a finite *height-varying* Δt
  produces a systematic bias velocity in inhomogeneous turbulence (Wilson & Flesch 1993
  Appendix) that violates the WMC. The cure is small enough Δt; verify the constraint is
  tight enough that this bias is negligible.
- **Rogue trajectories.** Watch for unbounded velocities when Δt is too large relative to
  rapidly varying statistics. Check for velocity caps / `g_a > g_min` clamps and whether
  they're physically defensible or just papering over a too-large Δt.
- **Integration scheme for position (CRITICAL for backward + surface sources).** Flesch et
  al. (1995 §3a, Appendix a) found the **explicit Euler scheme unsatisfactory** for backward
  surface-source prediction — it produces anomalous near-ground particle density and
  overestimates concentration, and this does *not* go away by shrinking Δt. Their fix: an
  **implicit scheme for position** (advance `x` using the *updated* velocity `u_{N+1}`, not
  the old `u_N`). **Check which scheme the position update uses.** If it's explicit Euler and
  the model is backward + surface-source, this is a likely real bug.

### R. Backward-mode reversal (Flesch et al. 1995; Thomson 1987 §3.4; Seibert & Frank 2004)

For inert tracers this is mostly straightforward, but there is one subtle trap:
- **Reversing the mean wind is NOT the same as running backward.** The backward Langevin
  drift `a'_i` differs from the forward `a_i` by a sign change on the mean-advection term
  *only* — the turbulent/inhomogeneity drift (the σ_w-gradient and density terms) does
  **not** simply flip (Flesch et al. 1995 Eq. 7; Thomson 1987 §3.4). For a Gaussian
  symmetric scheme this often reduces to "run the same SDE with the wind reversed and a
  negative time step," but verify the code isn't naively negating the *entire* drift.
- **Smith's reciprocal theorem is false in inhomogeneous turbulence** — the backward model
  is not the U-reversed forward model. Flag any code comment or logic assuming they're
  equivalent.
- **Particles carry mixing ratio, not mass** (Seibert & Frank 2004). The conserved Lagrangian
  quantity is mixing ratio; conversion to/from mass at source/receptor ends is a *per-particle*
  multiply/divide by local ρ (because ρ varies across a volume). Check the units bookkeeping.
- **Footprint = density-weighted mean residence time** in each source cell (Seibert & Frank
  2004 Eq. 8); for a surface source, equivalently the sum of `2/|w_touchdown|` over ground
  touchdowns within the source (Flesch et al. 1995 Eq. 13). Verify the footprint
  accumulation matches one of these forms and that the density weighting is present.

---

## VALIDATION HARNESS — build these tests, don't just inspect

The literature gives two cheap, decisive numerical tests. If the model has no such tests,
**adding them is the most valuable single contribution**, because they catch WMC violations
from *any* source (drift, boundary, Δt, density) at once.

### V1. The well-mixed test (the master test)

Initialize an ensemble of particles distributed according to the air density-weighted PDF
(uniform in z if density-neglected; ∝ρ(z) if density-corrected) with velocities drawn from
`g_a`. Integrate forward with **no sources** in a horizontally-homogeneous, stationary column.
The distribution must **remain unchanged** for arbitrarily long times.

- Diagnostic: bin particles by height, normalize by the expected (uniform or ρ-weighted)
  count, integrate for many τ_L. Any drift away from flat (or away from the ρ(z) profile)
  is a WMC violation. (Stohl & Thomson 1999 Figs. 1–2 show exactly this test, with and
  without the density term.)
- This single test will expose: a missing σ_w-gradient drift term, a missing density term,
  a bad reflection scheme, and a too-large Δt — without needing to know which.
- Run it with the boundaries active (reflection at z=0 and PBL top) AND in an unbounded
  homogeneous column, to localize whether a failure is in the drift or the boundary.

### V2. Forward–backward consistency test (Seibert & Frank 2004 Tests 1–4)

Compute a source–receptor relationship both forward (release at source, sample at receptor)
and backward (release at receptor, integrate back, accumulate residence time at source). For
the same volume pair they must agree.

- With analytic/constant winds they should agree to sampling error. With interpolated NWP
  winds they will differ slightly — Seibert & Frank show the residual is dominated by
  **wind-field interpolation error**, and suggest using the forward–backward difference as an
  estimate of transport error (useful later for your inversion's error budget).
- A *large* forward–backward discrepancy in constant winds indicates a real asymmetry bug
  (likely in the drift reversal R-checks or the boundary B-checks).

### V3. (If feasible) single-step Chapman–Kolmogorov check (Wilson & Flesch 1993)

For a stationary flow, the WMC need only be checked on the **first** time step: if a
well-mixed distribution is preserved after one Δt, it's preserved for all time. This makes a
cheap unit test — apply one step to a well-mixed state and check the position/velocity PDF is
unchanged. Especially useful as a fast regression test on the boundary-reflection logic.

---

## Deliverables requested from Claude Code

1. **Inventory:** what the model actually implements — order (zeroth/first), turbulence PDF
   (Gaussian/skewed), drift form, b form, boundary scheme, Δt rule, position integration
   scheme, density handling, backward-mode mechanics. State this before assessing it.
2. **Findings table:** each discrepancy as (location → behavior → correct physics →
   footprint consequence → severity). Severity = {systematic footprint bias / noise only /
   cosmetic}.
3. **The σ_w-gradient drift term (P3) and the density term (S) are the two highest-value
   checks** — confirm their presence and correctness first.
4. **Proposed fixes** only where physics is genuinely violated, with the WMC test (V1) as the
   acceptance criterion: a fix is validated when V1 passes where it previously failed.
5. **Add the V1 well-mixed test** to the codebase if absent.

## Source references (all digested; equations cited above trace to these)

- Taylor (1921) — diffusion by continuous movements; near-field vs far-field spread.
- Thomson (1987) — well-mixed criterion; drift derivation; Gaussian & skewed examples; App. B numerics.
- Wilson, Legg & Thomson (1983) — the σ_w-gradient drift-correction term.
- Wilson & Flesch (1993) — boundary reflection & WMC; Chapman–Kolmogorov test; Δt-bias.
- Stohl & Thomson (1999) — density correction term; well-mixed test figures.
- Seibert & Frank (2004) — backward source–receptor formalism; mixing-ratio bookkeeping; forward-backward tests.
- Flesch, Wilson & Yee (1995) — backward LS derivation; implicit position scheme; touchdown footprint; surface-layer parameterization (App. b).
- Hanna (1982) — surface-layer σ and τ_L parameterizations (not yet obtained; forms reproduced in Flesch et al. 1995 App. b).
