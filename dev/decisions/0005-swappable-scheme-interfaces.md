# 0005 — Turbulence & convection as swappable schemes; physics stays interrogable

**Context.** A research LPDM needs to compare parameterisations (and pick the best
per application), while performance work threatens to obscure the physics behind
compiled graphs and abstraction.

**Decision.**
- **Swappable schemes.** `TurbulenceScheme` and `ConvectionScheme` are ABCs with a
  name-keyed registry (`register_scheme` / `get_scheme`). A scheme is selected by
  string in the run YAML; a new scheme inherits the runtime's performance machinery
  (substepping, graph capture) without bespoke plumbing.
- **Interrogable physics.** The physics equations (Hanna σ/T_L/drift/Richardson/
  meander, the convection thermodynamics) stay as **standalone, unit-testable free
  functions**. Graph capture wraps the *assembled* per-step call; it never rewrites
  or inlines away the physics functions. Knobs stay exposed in config
  (`substep_c`, `max_substeps`, `flexpart_tl_floors`, …) so physics can be tuned
  and evaluated without code edits.

**Rationale.** Comparability is a core project goal, and opacity is the failure mode
this guards against. The eager path staying the numerical reference
([0002](0002-pure-pytorch-device-agnostic.md)) depends on the free functions being
callable and testable in isolation.

**Rejected alternatives.** A single monolithic hard-coded scheme (fast to write,
impossible to compare or audit); inlining physics into the compiled kernel for
speed (obscures the model, and the profile shows the physics math isn't the wall).

**Status.** In force and treated as a hard constraint. Production schemes today:
`hanna_1982` ([0006](0006-hanna-turbulence-flexpart-aligned.md)) and
`emanuel_reduced` ([0007](0007-reduced-emanuel-convection.md));
`placeholder_constant_ou` is kept only as a regression pin.
