# Status

- **Maturity:** reference (library; validated on synthetic learners only)
- **Version:** 0.2.0 (pyproject; tag v0.2.0 per org listing)
- **Last git activity on the default branch:** 2026-10-13 (from `git log -1 --format=%cs`; the working clone is shallow)

## What is supported
- Deterministic next-step planner (TEACH/REVIEW/ASSESS/STOP) over a prerequisite concept graph
- Retention and spacing model (v0.2.0); Bayesian-style belief update (`observe`)
- Tests in `tests/` with CI in `.github/workflows/tests.yml`

## What is not supported / not claimed
- No real-learner or pedagogy validation (README: proves planning logic only)
- learnerbot.ai is a private product and is not part of this repo
- Retention figures 0.755 / 0.495 / 0.020 are Illustrative/synthetic (`knowledge_frontier/backtest.py::run_retention_acceptance`)

## Known issues
- No NVIDIA Inception badge here, unlike sibling repos (left as is)

No quantitative claim in this repo has been externally reproduced or field validated.
