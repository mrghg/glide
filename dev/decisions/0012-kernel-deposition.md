# 0012 — Kernel deposition in the footprint gridder

**Context.** Residence time is deposited into the cell containing the particle
(a delta). This is exact in expectation but noisy, and it makes the footprint
non-differentiable with respect to particle position, which blocks tangent
footprints (roadmap 2c). Fine near-field grids (roadmap 4b) make the noise
worse.

**Decision.** Add optional kernel deposition: each particle spreads its
residence weight over a fixed compact stencil from its sub-cell position and a
bandwidth, mass-conserving, with bandwidth either fixed or growing with time
since release (as STILT-R v2 does). Delta deposition stays the default and the
reference; the kernel is a config choice validated against the analytic plume.

**Rationale.** One implementation serves three needs: variance reduction at a
given particle count (compute efficiency), usable high-resolution near-field
output, and differentiability. The sub-grid information is in the particle
positions already; no learned downscaling is needed for a particle model.

**Rejected alternatives.** Learned downscaling of gridded footprints (right for
an image-output emulator such as GATES, redundant here). Adaptive quadtree
grids (dynamic shapes, complex; two aligned regular grids suffice).

**Status.** Proposed 2026-09-16; implemented by task A2. Docs: physics.md
footprint accumulation section.
