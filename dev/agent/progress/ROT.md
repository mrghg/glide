# ROT — Orientation of horizontal turbulence (along-wind / cross-wind)

- status: CLAIMED
- queue: A   model: claude-opus-5-5   branch: agent/ROT-wind-frame-turbulence
- started: 2026-11-28   updated: 2026-11-28
- plan: Requested by Matt on 2026-11-28 (external review, "Horizontal spread and receptor setup"). Hanna's σ_u, T_Lu are along-wind and σ_v, T_Lv cross-wind, but GLIDE applies u′ to longitude and v′ to latitude. Add `turbulence.horizontal_frame: grid | wind` (default `grid`, bit-identical to main). In `wind`, the OU updates, σ carry (0028), interface rule and 4σ clip run on along/cross components, rotated to east/north before each horizontal displacement by the local mean wind direction at the particle (start of each outer step); near-calm particles keep the grid frame. Files: src/lpdm/turbulence/hanna.py, src/lpdm/config.py, src/lpdm/main.py, tests (new tests/test_horizontal_frame.py), dev/decisions/0031 or next free, docs/physics.md, docs/turbulence.md, README. CPU only. Outputs: outputs/agents/ROT/. No GPU runs.
- done:
- verified:
- docs:
- handoff:
- proposed follow-ups:
- reference issues:
- incidents:
- handled:
  - in-session: 2026-11-28 Matt asked for this task (spec in the session prompt); open a PR, do not merge; Matt decides the default.
