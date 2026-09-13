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

## Consumers
- **agentic-os** — `agentic_os.knowledge_frontier` re-exports this package (thin shim).
- **learnerbot.ai** — wraps it with pedagogy (learner model, Socratic tutoring, assessment, retention).

Public surface: `Concept`, `ConceptGraph`, `Mastery`, `ConceptState`, `FrontierAction`, `StopReason`,
`FrontierPolicy`, `FrontierChoice`, `next_step`, plus the signal functions (`prerequisites_met`,
`uncertainty`, `expected_gain`, `retention_need`, `concept_state`).
