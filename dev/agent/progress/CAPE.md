# CAPE — CAPE bounded at the diagnosed cloud top

- status: CLAIMED
- queue: A   model: claude-opus-5-5   branch: agent/CAPE-cloud-top-bound
- started: 2026-11-28   updated: 2026-11-28
- plan: External review ("GLIDE turbulence physics discussion notes.md", "CAPE above the diagnosed cloud top"): docs/convection.md defines CAPE from the LFC to the diagnosed cloud top (LNB), but `compute_lcl_lnb_cape` and `compute_lcl_lnb_cape_columns` in src/lpdm/convection/emanuel.py sum positive buoyancy from the LCL to the top of the column. Confirm in both paths; check the lower bound (LFC vs LCL, CIN); read FLEXPART conv_mod.f90 (convect43c); decision record 0031; bound CAPE at the cloud top identically in the scalar and batched paths (static shapes, no host sync in the batched path); regression test from the review (fails on main) and scalar–batched agreement; count trigger changes and mass-flux change on the reference config's convection columns over a few met windows (CPU); optional CPU smoke before/after; report (not change) the 5 m/s updraft cap; docs/convection.md; full CPU suite. CPU only, ≤8 threads, no GPU runs. Outputs: outputs/agents/CAPE/.
- done:
- verified:
- docs:
- handoff:
- proposed follow-ups:
- reference issues:
- incidents:
- question: none
- handled:
  - in-session: 2026-11-28 coordinator brief (Matt's request of 28 Nov 2026): fix the CAPE upper bound per the external review; decision record; regression test failing on main; trigger counts on representative meteorology; report the updraft cap as a question for Matt; open a PR, do not merge.
