# 0012 — Kernel deposition in the footprint gridder

**Context.** Residence time is deposited into the cell containing the particle
(a delta). This is exact in expectation but noisy, and it makes the footprint
non-differentiable with respect to particle position, which blocks tangent
footprints (roadmap 2c). Fine near-field grids (roadmap 4b) make the noise
worse. Spreading each deposit over neighbouring cells reduces the noise, but it
also widens the apparent plume, so it has to be shown not to change results.

**Decision.** Add optional kernel deposition, treated as a **numerical
parameter, converged and never tuned**:

- Each particle spreads its residence weight over a fixed compact horizontal
  stencil, using a Gaussian centred on its exact position. Weights are
  normalised over the in-grid stencil, so residence time is conserved.
- The bandwidth is a fixed fraction of the **local spread of the particle
  cloud**: the horizontal standard deviation of the active particles of each
  release, computed per age band, and capped at one cell width. The relative
  bias is then the same everywhere, and the kernel reduces to a delta just
  after release, where the cloud is compact.
- Horizontal only. Vertical bins keep delta deposition, because near-surface
  residence sets footprint magnitude and must not be smeared across bin edges.
- **Delta deposition is the default and the reference.** Validation (Tiers 1 to
  3), comparison with other models, and the v2 parameter calibration all use
  delta deposition. The kernel may be used in production only at or below the
  bandwidth fraction that the convergence study (task A5) shows gives results
  within particle noise of delta.
- The deposition settings are recorded in every run's metadata and output
  attributes.

**Why this is not a change to the physics.** The kernel never moves a particle,
so trajectories, vertical mixing and boundary-layer trapping are unchanged. Its
spread does not compound downstream, because it is applied once per deposit and
has no effect on where a particle goes next. It is symmetric, so it preserves
the plume's centre of mass. And it vanishes in the limit of many particles.
Statistically the footprint is a density estimated from a finite particle
sample; gridding is already a box-kernel estimate, and a Gaussian kernel is a
lower-variance estimator of the same density.

**The bias, stated.** Depositing with a kernel of width h onto cells of width Δ
smooths the true footprint with a total variance of h² plus Δ²/12, the second
term being the smoothing gridding imposes anyway. For a plume of crosswind width
σ, the output width is

$$
\sigma_{\mathrm{out}} = \sqrt{\sigma^2 + h^2 + \Delta^2/12},
$$

so with h = 0.3σ the width is about 4% larger and the peak about 4% lower than
with delta deposition at the same cell size. This is listed among the known
approximations in `docs/physics.md`.

**The risk this decision guards against.** Horizontal kernel smoothing and the
meander term both widen plumes. If the kernel were on during calibration, the
calibrated meander amplitude could shrink to compensate, describing the kernel
rather than the atmosphere. The same confounding would contaminate comparisons
with FLEXPART and NAME. Hence delta deposition for validation and calibration.

**Rationale for having it at all.** One implementation serves three needs:
variance reduction at a given particle count, usable high-resolution near-field
output, and the differentiability tangent footprints require. The sub-grid
information is already in the particle positions; no learned downscaling is
needed for a particle model.

**Rejected alternatives.**
- A fixed bandwidth: the relative bias would vary with plume width, largest
  close to the release where structure is sharpest.
- A bandwidth that grows with time since release (as STILT-R v2 does), as the
  primary rule: only a proxy for cloud spread, and wrong when meteorology makes
  the cloud contract or stretch. Usable as a fallback.
- A vertical kernel: smears near-surface residence, which sets footprint
  magnitude.
- Calibrating with the kernel on: confounds with the meander term.
- Learned downscaling of gridded footprints: right for an image-output emulator
  such as GATES, redundant here.
- Adaptive quadtree grids: dynamic shapes and complexity; two aligned regular
  grids suffice.
- A per-particle bandwidth from the distance to its nearest neighbours, as in
  EASLEE (Brioude et al., 2026, Sect. 2.16; doi:10.5194/egusphere-2026-5993),
  which uses the mean squared distance to the 12 nearest particles. It is the
  closest related method, and more local than ours, so it follows a split
  plume without the split-plume test A2 adds. EASLEE computes it only at
  output times. GLIDE deposits at every step inside the captured CUDA graph,
  where a neighbour search per step would add a spatial hash and a variable
  amount of work to every step, against a few sync-free reductions per
  release and age band for the cloud spread. If the split-plume test or the
  convergence study (A5) shows our bandwidth over-smoothing, a per-step
  neighbour bandwidth is the first alternative to try. EASLEE's ETEX-1 result,
  where changing the output kernel moved the near-source correlation from
  0.66 to 0.33 through a single station, also supports keeping delta
  deposition for validation.

**Status.** Proposed 2026-09-16; revised 2026-10-14 to tie the bandwidth to
cloud spread and to restrict the kernel to production use after convergence.
Implemented by task A2; bandwidth converged by task A5. Docs: physics.md
footprint accumulation section.
