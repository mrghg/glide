# WM — Fixed sub-step count with per-sub-step coefficient re-evaluation

- status: CLAIMED
- queue: A   model: claude-opus-5-5   branch: agent/WM-fixed-substeps
- started: 2026-11-25   updated: 2026-11-25
- plan: Pre-freeze physics fix approved by Matt on 25 Nov 2026 (outputs/agents/WM/SUBSTEPPING-SUMMARY.md §9/§12). Brings over agent/WM-near-surface (0032313: all Hanna coefficients re-evaluated at the current height every sub-step) and replaces the batch-max / per-particle adaptive count with one configured count `turbulence.n_substeps` for every particle and step (substep_c / max_substeps removed with a migration error). σ_w-only finite-difference probes. No-sync exceedance counter (δt > c·T_Lw per particle-sub-step, BL/FT, with denominators), read once per met window, in run_metadata.json. Basal-clamp gradient test (fix if it matters). Adaptivity test (k_i = max(n, adaptive k_i) vs fixed n). Accuracy study n = 4/8/16, neutral/stable/convective, ≥3 seeds, closed column (lid per sub-step) and open top, lowest 5% and 0–40 m, against a 1.875 s reference. GPU timing on ab_multisite_perf for main and n = 4/8/16 (one run at a time, ~54 GB host peak). CI near-surface gate at Δt = 60 s. Docs: physics.md §6, turbulence.md, STATUS.md, decision record 0026. Files: src/lpdm/turbulence/hanna.py, base.py, config.py, main.py, configs/*.yaml, tests/test_hanna.py, tests/test_main_runtime.py, tests/test_main_config.py, scripts/validation/wm_near_surface/. Outputs: outputs/agents/WM/fixcount/. Golden fixtures not regenerated.
- done:
- verified:
- docs:
- handoff:
- proposed follow-ups:
- reference issues: none
- incidents:
- question:
- handled:
  - in-session: 2026-11-25 coordinator brief (Matt approved the work on 25 Nov 2026): fixed configured count, exceedance counter, clamp and adaptivity tests, n = 4/8/16 study, GPU timing, CI gate; no fixture regeneration.
