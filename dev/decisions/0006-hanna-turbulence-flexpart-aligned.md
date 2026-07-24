# 0006 — Hanna (1982) turbulence, aligned to FLEXPART v11

**Context.** The placeholder constant-OU scheme (`T_L=300 s`, `σ²=1`, no horizontal,
no drift) was never physical. A validated boundary-layer scheme was needed, and
GLIDE's dispersion should be directly comparable to FLEXPART's.

**Decision.** Production turbulence is `hanna_1982`:
- Hanna (1982) in-BL σ/T_L formulae, regime-selected from `h/L` (stable / neutral /
  unstable); a **free-troposphere Richardson closure** (`K_z` from `N²`/Ri with a
  Blackadar length) above the BL.
- **Thomson (1987) well-mixed drift** `½(1 + w'²/σ_w²)·∂σ_w²/∂z`, sign-flipped for
  the backward Langevin integration (Flesch et al. 1995), plus a density-gradient
  term — without these an initially well-mixed tracer spuriously accumulates in
  low-turbulence regions.
- Optional Maryon (1998) / FLEXPART **meander** (independent horizontal OU).
- Coefficients and conventions aligned to **FLEXPART v11** after a 2026-07-02 audit:
  ERA5 SHF sign fixed (stability classification was inverted on real met),
  `|f|` Coriolis, constant stable-surface-layer σ_w.

**Rationale.** Hanna/FLEXPART is a recognised, validated scheme; matching v11
coefficients makes GLIDE's dispersion comparable to a reference the field trusts.
The equations were reimplemented independently from the primary literature — no
FLEXPART source is included (see [NOTICE](../../NOTICE)).

**Defaults with history.** `flexpart_tl_floors: true` (10/10/30 s `T_L` floors) and
`surface_layer_override: false` (regime formulas to the ground) are the defaults —
v2 validation showed near-surface `K = σ_w²·T_Lw` was 3–8× below FLEXPART's on
stable nights, over-inflating near-source surface residence. Legacy behaviour is
kept behind the flags for A/B.

**Rejected / deferred.** STILT-style mixing (not chosen — FLEXPART comparability);
per-substep σ re-evaluation and per-column (vs bbox-mean) vertical sampling are
documented follow-ups.

**Status.** In force. See [docs/turbulence.md](../../docs/turbulence.md) for the
assembled formulation and [docs/LPDM_physics_spec.md](../../docs/LPDM_physics_spec.md)
for the audit spec. Physics not yet externally validated ([STATUS.md](../../STATUS.md)).
