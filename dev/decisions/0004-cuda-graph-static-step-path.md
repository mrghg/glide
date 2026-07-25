# 0004 — Whole per-step body captured as one static-shape CUDA graph

**Context.** The per-batch problem is "narrow and deep": ~10⁵ particles (adequate
GPU width) driven through thousands of sequential steps. On the GH200 the profile
was **launch-bound, not compute-bound** — ~1,250 kernel launches/step, GPU busy
~17%, the GPU idling in per-step CPU-orchestration bubbles between kernels.
`torch.compile` alone fuses *within* a method call but not *across* the Python
substep loop.

**Decision.** Capture the **whole per-step body** (advection + interpolation +
column turbulence + drift + substep loop + meander + mask-gated write-back) as a
**single CUDA graph** via `torch.compile(mode="reduce-overhead")`, device-gated
(CUDA only; CPU keeps the dynamic masked path as the reference). This forces:
static shapes (run the full particle set every substep, gate finished/inactive
particles by *multiply*, never by index — `sub_dt=0` is a math no-op); a fixed
substep count; no host syncs in the captured region; and **address-stable inputs**
(persistent buffers marked `mark_static_address`, refilled in place per window, so
replay doesn't re-stage them).

**Rationale.** Recording the launch sequence once and replaying it with one host
call removes the between-kernel bubbles. Measured on the GH200: per-step 30 → 5 ms,
launches ~1,250 → ~106, GPU busy 17 → 37%; the static-input buffers then collapsed
the cudagraph staging copies (~49% → 4.3% of GPU time). Accepting the added FLOPs
(every particle runs to `max_substeps`; escaped particles keep being processed) is
a net win because launch overhead dominated.

**Rejected alternatives.**
- More `torch.compile` without graphs — can't fuse across the Python step loop.
- Keep the dynamic active-set-shrinking optimisations on GPU — they need dynamic
  shapes, which break graph capture. Kept for CPU; **strategy (D)+(A)** = device-gated
  static path that accepts the escaped-particle waste. Periodic recapture/compaction
  (strategy B) deferred; a later profile showed the per-step phase is not the wall.

**Guardrails.** Two CPU-only tests protect the capture without a GPU:
`test_step_core_traces_as_one_graph_no_breaks` (graph break) and
`test_step_core_does_not_recompile_per_step` (recompile). Rules: no `.item()`/
data-dependent control flow; no per-step/per-window value may reach the core as a
Python scalar/tuple (pass as tensors); clone outputs that outlive the call;
`cudagraph_mark_step_begin()` per invocation. See
[docs/architecture.md](../../docs/architecture.md) §5, §8.

**Status.** In force. NB (2026-07-24): the per-step phase is now only ~18–23% of
representative production wall — further step-side graph work (folding the gridder,
particle compaction) is low priority vs met/convection/residual (see
[STATUS.md](../../STATUS.md)).
