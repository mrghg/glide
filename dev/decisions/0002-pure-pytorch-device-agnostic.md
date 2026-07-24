# 0002 — Pure PyTorch, device-agnostic; the eager path is the numerical reference

**Context.** The LPDM hot path — `grid_sample` interpolation of the wind/turbulence
fields plus the elementwise OU turbulence step — is dense, data-parallel work that
a GPU accelerates well. Legacy LPDMs are single-threaded CPU code.

**Decision.** Implement the whole engine in **pure PyTorch**, **device-agnostic**
across CUDA / MPS / CPU with dynamic fallback (`lpdm.runtime`). The **eager CPU
path is the numerical reference**; compiled/graph paths are opt-in accelerations
validated *against* eager.

**Rationale.** One codebase scales from a CPU laptop to a GH200 and grows onto
larger accelerators. Keeping eager as the reference means physics stays debuggable
and testable without a GPU, and every optimisation has an oracle to check against
(the `test_compiled_hot_paths_match_eager_*` and well-mixed gates).

**Rejected alternatives.**
- CPU-only (the legacy bottleneck).
- Custom CUDA kernels — would abandon device-agnosticism and the eager reference,
  and obscure the physics.

**Status.** Foundational, in force. All runtime-critical paths are vectorised
tensor ops — no per-particle Python. GPU-specific work is device-gated. The
compiled GPU path is [0004](0004-cuda-graph-static-step-path.md).
