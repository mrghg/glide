"""Validation figure library and the Tier 0 figure set (docs/validation-plan.md §5–6).

- ``style``: palette and colour scales shared by every tier.
- ``core``: provenance stamp, log ratio, particle-noise bands, panels, save.
- ``tier0``: the Tier 0 figures, recomputed from the analytic tests.
- ``gallery``: writes ``docs/validation/gallery.md`` keyed by physics tag.
- ``explorer``: a static, self-contained HTML footprint explorer.

Regenerate everything with ``.venv/bin/python scripts/validation/figures/make_all.py``.
"""
