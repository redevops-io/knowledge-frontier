# knowledge-frontier

A **domain-neutral** primitive for deciding *what to learn or teach next* to maximise demonstrated
understanding — extracted so multiple products can share one engine (the ReDevOps Agentic Apps stack
and **learnerbot.ai**).

Model understanding as a **prerequisite concept graph** with a per-concept **mastery** belief, then pick
the single best next move by structural importance, prerequisite readiness, expected gain, uncertainty,
misconception risk and retention need:

```
concept graph + mastery state → eligible concepts (prerequisites met) → priority per concept → next_step
                                                                   ↳ TEACH / REVIEW / ASSESS / STOP
```

`ASSESS` is the information-gathering move: high uncertainty about a concept whose belief is *unverified*
is often better probed (cheap) than taught blind.

Deterministic and model-free. Whether the selection *policy* reaches mastery in fewer steps than naive
orderings is decided by `knowledge_frontier.backtest` on controlled synthetic learners — proving the
planning **logic**, not real-world pedagogy.

## Use

```python
from knowledge_frontier import Concept, ConceptGraph, Mastery, FrontierPolicy, next_step

graph = ConceptGraph({c.id: c for c in [
    Concept("basics"), Concept("advanced", prerequisites=("basics",))]})
state = {"basics": Mastery(prob=0.9, exposed=True)}
choice = next_step(graph, state)          # → advance "advanced", now that its prereq is met
```

## Retention & spacing (v0.2.0)

Real learners forget. A concept now carries a **memory stability** (`Mastery.stability`): recall decays as
`retrievability = 2**(-staleness / stability)` (the forgetting curve), and each successful review **grows**
stability (`reinforce`) — so `next_step` schedules `REVIEW` at **expanding, per-concept intervals** (spaced
repetition) instead of a flat clock. `observe(mastery, correct)` is a Bayesian-Knowledge-Tracing belief
update (slip/guess + a learning transition) for grading a response, and marks the belief `assessed`.

```python
from knowledge_frontier import Mastery, reinforce, observe, retrievability
m = reinforce(Mastery(), success=True)          # first study: stability set, belief lifted
m = reinforce(m, success=True)                  # a later review: stability doubles → next review due later
belief = observe(Mastery(prob=0.5, exposed=True), correct=True)   # grade a probe → posterior P(known)
```

Backward-compatible: `stability` defaults to `0`, which keeps the v0.1.0 flat-`staleness` retention
behaviour, so existing callers are unaffected until they populate it.

**Proven on forgetting learners** (`backtest.run_retention_acceptance`, 120 learners): spaced review
retains **0.755** of importance-weighted understanding at the horizon vs **0.495** for fixed-interval review
(with *fewer* reviews) and **0.020** for no review. Spacing beats fixed on every seed — validating the
retention *logic*, not real-world pedagogy.

## Consumers
- **agentic-os** — `agentic_os.knowledge_frontier` re-exports this package (thin shim).
- **learnerbot.ai** — wraps it with pedagogy (learner model, Socratic tutoring, assessment, retention).

Public surface: `Concept`, `ConceptGraph`, `Mastery`, `ConceptState`, `FrontierAction`, `StopReason`,
`FrontierPolicy`, `FrontierChoice`, `next_step`, plus the signal functions (`prerequisites_met`,
`uncertainty`, `expected_gain`, `retention_need`, `concept_state`, `retrievability`, `reinforce`,
`observe`).
