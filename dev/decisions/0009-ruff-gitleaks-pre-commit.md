# 0009 — Ruff + gitleaks pre-commit hooks; PEP 8 spaces

**Context.** The repo had no enforced lint, format, or secret-scanning. Style had
drifted: 13 files (the older core — `main.py`, `hanna.py`, `emanuel.py`, …) used
tab indentation, 25 used 4-space, and `gpu_engine.py` mixed both. "Match the
surrounding style" had no single answer, and nothing stopped a credential from
being committed.

**Decision.**
- **Ruff** for both lint and format, enforced via pre-commit and configured in
  `pyproject.toml` (single source of truth). Curated rule set: `E, W, F, I, UP,
  B, C4, SIM`, with the pedantic/style-only members ignored (`C408`, `C401`,
  `SIM108`, `B905`) and `E501` left to the formatter. Line length 100 (matches the
  code's existing p95 ≈ 87).
- **Standardise on 4-space PEP 8 indentation.** The 13 tab files + the mixed one
  were reflowed once (`ruff format`) to a uniform space baseline.
- **Gitleaks** in the same hook (commit and pre-push) for secret scanning.
- Plus standard hygiene hooks (trailing whitespace, EOF, merge-conflict markers,
  private-key detection, large-file guard at 1 MB to keep met/validation data out).

**Rationale.** Automating format ends the tabs-vs-spaces drift and removes layout
from review entirely. Spaces is the PEP 8 default and already the majority (and all
the newer files). Gitleaks is cheap insurance against the one mistake that can't be
undone by a later commit. A curated rule set keeps signal high — the safe autofixes
(153 of them: `datetime.UTC`, import sorting, modernisations) landed in one baseline
commit; the physics core wasn't churned for pure-style rules.

**Rejected alternatives.**
- **Standardise on tabs** — would have reflowed 26 files (the majority) instead of
  14, and diverges from PEP 8 for no benefit.
- **Defer the formatter** (lint + secrets only) — leaves the mixed indentation in
  place and keeps formatting in code review.
- **Black + flake8 + isort** — three tools where ruff does all three, faster.

**Status.** In force. Enable locally with `pre-commit install` (and
`--hook-type pre-push`); see [CONTRIBUTING.md](../../CONTRIBUTING.md). The one-time
reflow was behaviour-preserving (full suite green, 257 passed).
