# CHECKPOINT.md — retired 2026-07-24

The stream-of-consciousness dev journal that used to live here was retired to cut
bloat and move to an agent-agnostic layout. Its **durable** content moved to:

- **[STATUS.md](../STATUS.md)** — current state: what works, what's pending, latest results.
- **[dev/decisions/](decisions/)** — the major design & physics decisions (why, with
  rejected alternatives).
- **[docs/](../docs/)** — the physics & systems reference (architecture, turbulence,
  convection, validation, LPDM spec).
- **[AGENTS.md](../AGENTS.md)** — contributor / coding-agent guide.

The **full chronological narrative is in git history**:

```
git log --follow -- dev/CHECKPOINT.md
git show <commit>:dev/CHECKPOINT.md    # read the journal at any point in time
```

## Where the frequently-referenced anchors went

Code comments and docs still cite journal anchors; here is the map:

| Journal anchor | Now in |
|---|---|
| Finding 7 (terrain-blind vertical coordinate) | [decisions/0003](decisions/0003-terrain-following-agl-coordinate.md), [docs/architecture.md](../docs/architecture.md) |
| M3 / CUDA-graph phases (2026-06-19, -26) | [decisions/0004](decisions/0004-cuda-graph-static-step-path.md), [docs/architecture.md](../docs/architecture.md) §5, §8 |
| Physics audit findings F1–F10 (2026-05-30/31) | [decisions/0006](decisions/0006-hanna-turbulence-flexpart-aligned.md), [docs/turbulence.md](../docs/turbulence.md), [dev/PHYSICS_REVIEW_2026-07-02.md](PHYSICS_REVIEW_2026-07-02.md) |
| Deep convection (Emanuel, 2026-06-01) | [decisions/0007](decisions/0007-reduced-emanuel-convection.md), [docs/convection.md](../docs/convection.md) |
| Multi-site releases / met caching / "NEW OPTIMISATION FRONTIER" | [decisions/0008](decisions/0008-multi-site-shared-met-batching.md), [STATUS.md](../STATUS.md) |
| Perf #1/#4 A/B, the venv-trap methodology lesson | [decisions/0004](decisions/0004-cuda-graph-static-step-path.md), [decisions/0008](decisions/0008-multi-site-shared-met-batching.md), [STATUS.md](../STATUS.md) |
| M0–M6 milestone roadmap, public-release audit | git history (completed / superseded) |
