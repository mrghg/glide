# 0011 — Multi-rate time stepping: a per-particle outer step

**Context.** The outer step is a fixed 60 s for every particle. That is right in
the boundary layer and wasteful in the free troposphere, where timescales are
long, winds are smooth on the 25 km met grid, and most particle-hours of a
5-day backward run are spent. Turbulence already sub-steps per particle
(adaptive, fixed trip count on the graph path); advection does not.

**Decision.** Each particle carries its own outer step Δt_i ∈ {Δt, 2Δt, 4Δt, …,
Δt_max}, chosen from local criteria (height above the boundary layer, |w|,
σ_w, T_L, a CFL-type bound on the met grid) and re-evaluated once per met
window. The step loop keeps a fixed trip count; a particle advances only on
iterations that are multiples of its rate, gated by multiply so shapes stay
static and the CUDA graph is unaffected. Residence accumulation uses Δt_i.
Single-rate remains the default until the convergence study (roadmap 7b) sets
Δt_max.

**Rationale.** Potentially a 3–5× reduction in total steps at no loss where it
matters, exact in the sense that every particle still integrates the same
equations at a step that is converged for its local conditions. Simpler and
more transparent than far-field particle aggregation, which changes what a
particle is.

**Rejected alternatives.** A globally longer Δt (loses boundary-layer accuracy).
Particle aggregation first (roadmap 6b; deferred to v1.x, may still be worth
doing on top). Dynamic-shape active-set compaction (breaks graph capture,
decision 0004).

**Status.** Proposed 2026-09-16; implemented by task A3. Docs: physics.md time
stepping section once merged.
