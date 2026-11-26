# SF — σ_w floor: drift gradient at a stable boundary-layer top

- status: CLAIMED
- queue: A   model: claude-opus-5-5   branch: agent/SF-sigma-floor-gradient
- started: 2026-11-26   updated: 2026-11-26
- plan: Turn the reviewed WM-sigma-floor prototype (agent/WM-sigma-floor ce07cb6; outputs/agents/WM/sigma_floor/SUMMARY.md and review/) into a PR on main d19755d. Keep only the `hybrid` drift gradient (analytic ∂σ_w²/∂z of the floored Hanna profile for boundary-layer particles whose stencil is cut at h or touches the floor; the production stencil elsewhere, bit for bit); drop the kink/analytic/quadrature/offset modes and the T_Lw rescale; old FD behaviour only as a private attribute for A/B tests. Files: src/lpdm/turbulence/hanna.py, tests/test_hanna.py, tests/test_sigma_floor_gradient.py (new: analytic exactness in three regimes, bit identity away from h and the floor, static/dynamic parity, transfer-symmetry gate in a thin-floor stable column calibrated over ≥10 seeds), dev/decisions/0030 (amends 0028 item 1; floor kept at 1e-3 as an open question for Matt), docs/physics.md, docs/turbulence.md, STATUS.md (one row). CPU only (another agent holds the GPU): CPU smoke before/after, 2 seeds; full CPU suite. Outputs: outputs/agents/SF/. No GPU runs.
- done:
- verified:
- docs:
- handoff:
- proposed follow-ups:
- reference issues: none
- incidents:
- question: none
- handled:
  - in-session: 2026-11-26 coordinator brief: build the PR from the reviewed prototype on current main; hybrid only; tests incl. transfer-symmetry gate (fails on main, passes with the fix); decision record amending 0028 item 1, floor value left to Matt; smoke before/after; draft PR, do not merge; finish by 07:30 UTC.
