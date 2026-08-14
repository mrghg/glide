# GLIDE documentation

In-depth reference for the model. Start with the [project README](../README.md)
for what GLIDE is and how to run it; these pages cover the physics and the
engineering.

## Suggested reading order

**If you are here for the science**, read these three in order:

1. **[physics.md](physics.md)** — the model formulation. What a footprint is, the
   coordinate system, the Langevin equation GLIDE integrates, the well-mixed
   drift, the backward-in-time construction, boundary conditions, time stepping,
   and how the footprint is accumulated. Start here.
2. **[turbulence.md](turbulence.md)** — where $\sigma$ and $T_L$ come from: Hanna
   (1982) in three stability regimes, the gradient-Richardson closure above the
   boundary layer, the Lagrangian-timescale floors, and the unresolved-mesoscale
   meander process.
3. **[convection.md](convection.md)** — deep moist convection: the reduced
   Emanuel mass-flux scheme, its trigger, and why the mass-flux matrix has to be
   non-divergent.

**If you are here for the engineering:**

4. **[architecture.md](architecture.md)** — the shape of the problem and what it
   forces: the streaming meteorology pipeline and its three caches, the two
   device-gated per-step execution paths, CUDA-graph capture and the four
   constraints that make it hold, batching and memory, diagnostics.

**Reference:**

5. **[met_schema.md](met_schema.md)** — the meteorology input contract. Variable
   names, units, sign conventions, vertical modes, chunking, storage precision.
   Read this to prepare meteorology from a source that is not ARCO ERA5.
6. **[VALIDATION.md](VALIDATION.md)** — what has been verified and against what,
   what has **not**, and how to run a comparison against FLEXPART or NAME.

## Related

- **[../STATUS.md](../STATUS.md)** — current state: what works, what is not yet
  validated, what is next.
- **[../dev/decisions/](../dev/decisions/)** — the major design and physics
  decisions, each with its rationale and the alternatives that were rejected.
  These record *why*; the pages above record *how*.
- **[../AGENTS.md](../AGENTS.md)** — contributor and coding-agent guide.

## Notes

These are plain Markdown with GitHub-flavoured LaTeX (`$$…$$`), rendered as-is by
github.com. They are structured so a static documentation site can be pointed at
this directory later without reorganisation.
