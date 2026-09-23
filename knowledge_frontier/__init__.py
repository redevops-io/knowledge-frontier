"""Knowledge Frontier — decide what to learn/teach next to maximise understanding
(AGENTIC_APPS_PROACTIVE_INTELLIGENCE_PLAN §12, generalised).

'Establish the map before traversing every branch': rather than push a long sequence of disconnected
facts, model understanding as a prerequisite graph of concepts, track each concept's mastery state, and
pick the next concept by structural importance, prerequisites, uncertainty, misconception risk and
retention need — optimising demonstrated mastery, not content consumed.

This is a DOMAIN-NEUTRAL stack primitive, not a tutoring product: a 'learner' is any entity acquiring a
body of concepts — a person being taught, a new hire onboarding, an agent building competence, a team
closing a skill gap. (It is unrelated to, and shares no code with, the separate learnerbot.ai product.)

    concept graph + mastery state → eligible concepts (prerequisites met) → priority per concept
      (importance · expected gain · uncertainty · misconception · retention) → TEACH / REVIEW / ASSESS / STOP

ASSESS is the information-gathering move: a high *uncertainty* about a concept does not always mean
'teach it' — when our belief is also UNVERIFIED, the cheaper, higher-value move is often to probe
whether the entity already knows it, and only then decide to teach or move on. (For non-human domains
this generalises to ACQUIRE_EVIDENCE — inspect docs, run a test, query an app, ask a human — where the
frontier says WHAT is missing and the Mission/Context Runtime decides HOW to acquire it.)

Deterministic and model-free. Whether this selection POLICY reaches mastery with fewer steps than naive
orderings is decided by :mod:`agentic_os.knowledge_frontier_backtest` on controlled learners — not
asserted here. As with the other kernels, that proves the planning LOGIC on a controlled benchmark, not
real-world pedagogy.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Mapping, Optional, Sequence, Set, Tuple


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


class ConceptState(Enum):
    NOT_ENCOUNTERED = "not_encountered"
    EXPOSED = "exposed"
    UNCERTAIN = "uncertain"
    PROBABLY_UNDERSTOOD = "probably_understood"
    MISCONCEPTION = "misconception"
    RETENTION_AT_RISK = "retention_at_risk"
    MASTERED = "mastered"


@dataclass(frozen=True)
class Concept:
    id: str
    prerequisites: Tuple[str, ...] = ()
    difficulty: float = 0.5                       # 0..1, informational (harder concepts cost more)


@dataclass(frozen=True)
class Mastery:
    """What we believe about a learner's grasp of one concept.

    ``prob`` is our BELIEF; ``assessed`` says whether that belief is grounded in an actual observation
    (a probe / demonstration) rather than a prior guess. The two together are what lets the frontier
    tell 'they half-know this, teach it' from 'we don't actually know what they know — go find out'."""
    prob: float = 0.0                             # P(mastered), 0..1 (our belief)
    exposed: bool = False                         # has the concept been encountered at all?
    misconception: bool = False                   # an active wrong belief (worse than not knowing)
    staleness: float = 0.0                        # 'time' since last reinforced (retention decay proxy)
    assessed: bool = False                        # has this belief been verified (vs a prior guess)?
    stability: float = 0.0                        # memory strength: the staleness at which recall halves.
    #   0 ⇒ no forgetting model (v0.1.0 behaviour: retention uses the flat `at_risk_staleness` proxy).
    #   >0 ⇒ retrievability decays as 2**(-staleness/stability); it GROWS with each successful review
    #   (the spacing effect), so reviews are scheduled at expanding, per-concept-correct intervals.


KnowledgeState = Mapping[str, Mastery]


# ── the concept graph + structural importance ────────────────────────────────────────
@dataclass
class ConceptGraph:
    concepts: Mapping[str, Concept]
    _importance: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        self._importance = self._compute_importance()

    def _transitive_dependents(self, cid: str) -> Set[str]:
        """Every concept that (transitively) depends on ``cid`` — its structural reach."""
        dependents: Set[str] = set()
        stack = [c.id for c in self.concepts.values() if cid in c.prerequisites]
        while stack:
            d = stack.pop()
            if d in dependents:
                continue
            dependents.add(d)
            stack.extend(c.id for c in self.concepts.values() if d in c.prerequisites)
        return dependents

    def _compute_importance(self) -> Dict[str, float]:
        n = len(self.concepts)
        if n <= 1:
            return {cid: 1.0 for cid in self.concepts}
        return {cid: len(self._transitive_dependents(cid)) / (n - 1) for cid in self.concepts}

    def importance(self, cid: str) -> float:
        """0..1: how foundational — the share of the body that (transitively) builds on this concept."""
        return self._importance.get(cid, 0.0)


# ── per-concept signals ────────────────────────────────────────────────────────────────
def _mastery(state: KnowledgeState, cid: str) -> Mastery:
    return state.get(cid, Mastery())


def prerequisites_met(graph: ConceptGraph, state: KnowledgeState, cid: str,
                      threshold: float = 0.7) -> bool:
    """True when every prerequisite is understood well enough to build on."""
    c = graph.concepts[cid]
    return all(_mastery(state, p).prob >= threshold and not _mastery(state, p).misconception
               for p in c.prerequisites)


def uncertainty(m: Mastery) -> float:
    """Highest when a concept is half-grasped (p≈0.5); low when clearly known or clearly not."""
    if not m.exposed:
        return 0.0
    return _clamp(1.0 - abs(2.0 * m.prob - 1.0))


def expected_gain(graph: ConceptGraph, cid: str, m: Mastery, mastery_threshold: float) -> float:
    """Room to grow, weighted by how foundational the concept is (advancing a foundation lifts more)."""
    room = _clamp(mastery_threshold - m.prob) / max(mastery_threshold, 1e-9)
    return _clamp(room * (0.4 + 0.6 * graph.importance(cid)))


def retrievability(m: Mastery) -> float:
    """P(recall right now), 0..1. With a stability model it decays as 2**(-staleness/stability)
    (the forgetting curve); without one it falls back to the flat belief `prob`."""
    if m.stability <= 0:
        return _clamp(m.prob)
    return _clamp(2.0 ** (-(m.staleness / m.stability)))


def retention_need(m: Mastery, at_risk_staleness: float, *,
                   target_retrievability: float = 0.9) -> float:
    """How overdue a once-learned concept is for review. 0 while fresh; ≥1 once due.

    With a stability model, 'due' is when retrievability has fallen to `target_retrievability`
    — a per-concept interval that EXPANDS as stability grows (spaced repetition). Without one it
    reduces to the v0.1.0 flat proxy (staleness / at_risk_staleness)."""
    if m.prob < 0.6:
        return 0.0
    if m.stability > 0:
        due = m.stability * math.log2(1.0 / target_retrievability)   # staleness where R hits target
        return (m.staleness / due) if due > 0 else 0.0               # ≥1 ⇒ due; grows while overdue
    if at_risk_staleness <= 0:
        return 0.0
    return _clamp(m.staleness / at_risk_staleness)


def reinforce(m: Mastery, *, success: bool = True, first_stability: float = 1.0,
              spacing_factor: float = 2.0, fail_prob: float = 0.4) -> Mastery:
    """Fold a TEACH/REVIEW outcome into memory strength (the write side of the forgetting model).

    A SUCCESS resets staleness, lifts the belief, and GROWS stability (first exposure → first_stability;
    each later success × spacing_factor) — so the next review is due later, the spacing effect. A FAILURE
    drops the belief and collapses stability back to a short interval (lapses must be re-learned)."""
    if success:
        prob = _clamp(m.prob + 0.5 * (1.0 - m.prob)) if m.prob else 0.9
        stability = first_stability if m.stability <= 0 else m.stability * spacing_factor
        return Mastery(prob=max(prob, 0.85), exposed=True, misconception=False,
                       staleness=0.0, assessed=True, stability=stability)
    return Mastery(prob=min(m.prob, fail_prob), exposed=True, misconception=m.misconception,
                   staleness=0.0, assessed=True, stability=first_stability)


def observe(m: Mastery, correct: bool, *, learn: float = 0.3, slip: float = 0.1,
            guess: float = 0.2) -> Mastery:
    """Bayesian-Knowledge-Tracing belief update from one graded response (the read side).

    Posterior P(known) given the answer (slip = knew-but-wrong, guess = didn't-know-but-right), then a
    learning transition (`learn`). Marks the belief `assessed` (grounded in a real observation) and resets
    staleness. Correctness updates BELIEF; use `reinforce` to also grow retention stability."""
    p = _clamp(m.prob)
    if correct:
        post = (p * (1 - slip)) / max(p * (1 - slip) + (1 - p) * guess, 1e-9)
    else:
        post = (p * slip) / max(p * slip + (1 - p) * (1 - guess), 1e-9)
    p_after = _clamp(post + (1 - post) * learn)
    return Mastery(prob=p_after, exposed=True, misconception=m.misconception,
                   staleness=0.0, assessed=True, stability=m.stability)


def concept_state(graph: ConceptGraph, state: KnowledgeState, cid: str, *,
                  mastery_threshold: float = 0.85, at_risk_staleness: float = 4.0,
                  target_retrievability: float = 0.9) -> ConceptState:
    m = _mastery(state, cid)
    if m.misconception:
        return ConceptState.MISCONCEPTION
    if not m.exposed and m.prob <= 1e-9:
        return ConceptState.NOT_ENCOUNTERED
    if m.prob >= mastery_threshold:
        return (ConceptState.RETENTION_AT_RISK
                if retention_need(m, at_risk_staleness,
                                  target_retrievability=target_retrievability) >= 1.0
                else ConceptState.MASTERED)
    if m.prob >= 0.6:
        return ConceptState.PROBABLY_UNDERSTOOD
    if uncertainty(m) >= 0.6:
        return ConceptState.UNCERTAIN
    return ConceptState.EXPOSED


# ── the frontier selection (plan §12) ─────────────────────────────────────────────────
class FrontierAction(Enum):
    TEACH = "teach"           # advance an eligible, not-yet-mastered concept
    REVIEW = "review"         # reinforce a mastered concept whose retention is at risk
    ASSESS = "assess"         # probe an UNVERIFIED, uncertain belief before investing in teaching
    STOP = "stop"
    # Planned extension for non-human domains (agents): ACQUIRE_EVIDENCE — when the frontier finds a
    # capability gap, the *how* (read docs, run a test, query an app, ask a human) is a governed
    # Mission/Context-Runtime concern, not a teaching step. Not implemented until a producer needs it.


class StopReason(Enum):
    OBJECTIVE_MET = "objective_met"             # all target concepts mastered
    ALL_MASTERED = "all_mastered"               # nothing left to learn
    BLOCKED_ON_PREREQS = "blocked_on_prereqs"   # unmastered concepts remain but all are prereq-blocked


@dataclass(frozen=True)
class FrontierPolicy:
    prereq_threshold: float = 0.7
    mastery_threshold: float = 0.85
    at_risk_staleness: float = 4.0
    target_retrievability: float = 0.9   # review a concept once its recall probability falls to this
                                         # (only used when a concept carries a stability model)
    w_importance: float = 0.4
    w_gain: float = 0.4
    w_uncertainty: float = 0.15
    w_misconception: float = 0.3      # a misconception is worth correcting before new material
    w_retention: float = 0.7          # reviewing an at-risk concept competes with teaching new ones
    w_assess: float = 0.6             # probing an unverified, uncertain belief competes with teaching
    assess_uncertainty: float = 0.4   # only probe when the belief is genuinely ambiguous
    enable_assess: bool = True        # off ⇒ the pre-ASSESS behaviour (teach on belief alone)


@dataclass(frozen=True)
class FrontierChoice:
    action: FrontierAction
    concept_id: Optional[str] = None
    state: Optional[ConceptState] = None
    priority: float = 0.0
    stop_reason: Optional[StopReason] = None
    rationale: str = ""


def _teach_priority(graph, state, cid, m, p: FrontierPolicy) -> float:
    return (p.w_importance * graph.importance(cid)
            + p.w_gain * expected_gain(graph, cid, m, p.mastery_threshold)
            + p.w_uncertainty * uncertainty(m)
            + p.w_misconception * (1.0 if m.misconception else 0.0))


def next_step(graph: ConceptGraph, state: KnowledgeState, *, objective: Optional[Set[str]] = None,
              policy: Optional[FrontierPolicy] = None) -> FrontierChoice:
    """Pick the single best next move: teach an eligible concept, review an at-risk one, or stop.
    Eligibility requires prerequisites met — which is what makes the frontier 'establish the map before
    traversing every branch' rather than waste effort on concepts the learner isn't ready for."""
    p = policy or FrontierPolicy()

    if objective and all(_mastery(state, c).prob >= p.mastery_threshold and not _mastery(state, c).misconception
                         for c in objective):
        return FrontierChoice(FrontierAction.STOP, stop_reason=StopReason.OBJECTIVE_MET,
                              rationale="every target concept is mastered")

    candidates: List[Tuple[float, str, FrontierAction]] = []
    unmastered_exists = blocked_exists = False
    for cid in graph.concepts:
        m = _mastery(state, cid)
        if m.prob >= p.mastery_threshold and not m.misconception:      # mastered
            need = retention_need(m, p.at_risk_staleness, target_retrievability=p.target_retrievability)
            if need >= 1.0:
                candidates.append((p.w_retention * need * (0.5 + 0.5 * graph.importance(cid)),
                                   cid, FrontierAction.REVIEW))
            continue
        unmastered_exists = True
        if prerequisites_met(graph, state, cid, p.prereq_threshold):
            candidates.append((_teach_priority(graph, state, cid, m, p), cid, FrontierAction.TEACH))
            # ASSESS competes with TEACH: when the belief is uncertain AND unverified, it may be worth
            # probing (cheap) to learn whether the entity already knows it before investing in teaching.
            if p.enable_assess and not m.assessed and uncertainty(m) >= p.assess_uncertainty:
                assess_prio = p.w_assess * uncertainty(m) * (0.5 + 0.5 * graph.importance(cid))
                candidates.append((assess_prio, cid, FrontierAction.ASSESS))
        else:
            blocked_exists = True

    if not candidates:
        if unmastered_exists and blocked_exists:
            return FrontierChoice(FrontierAction.STOP, stop_reason=StopReason.BLOCKED_ON_PREREQS,
                                  rationale="unmastered concepts remain but their prerequisites aren't met")
        return FrontierChoice(FrontierAction.STOP, stop_reason=StopReason.ALL_MASTERED,
                              rationale="nothing left to learn")

    prio, cid, action = max(candidates, key=lambda c: c[0])
    st = concept_state(graph, state, cid, mastery_threshold=p.mastery_threshold,
                       at_risk_staleness=p.at_risk_staleness,
                       target_retrievability=p.target_retrievability)
    verb = {FrontierAction.REVIEW: "Review", FrontierAction.ASSESS: "Assess"}.get(action, "Advance")
    return FrontierChoice(action, concept_id=cid, state=st, priority=prio,
                          rationale=(f"{verb} '{cid}' (importance {graph.importance(cid):.2f}, "
                                     f"state {st.value})"))


__all__ = [
    "ConceptState", "Concept", "Mastery", "KnowledgeState", "ConceptGraph",
    "prerequisites_met", "uncertainty", "expected_gain", "retention_need", "concept_state",
    "retrievability", "reinforce", "observe",
    "FrontierAction", "StopReason", "FrontierPolicy", "FrontierChoice", "next_step",
]
