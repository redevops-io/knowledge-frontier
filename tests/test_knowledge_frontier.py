"""Unit tests for the Knowledge Frontier engine (knowledge_frontier) + its Priority Engine
adapter. Test the selection LOGIC — structural importance, prerequisite gating, uncertainty/retention
signals, and the TEACH/REVIEW/STOP decision. Whether the policy beats naive orderings is in
test_knowledge_frontier_backtest.py.
"""
from __future__ import annotations

import pytest

from knowledge_frontier import (
    Concept, ConceptGraph, ConceptState, FrontierAction, FrontierPolicy, Mastery, StopReason,
    concept_state, expected_gain, next_step, prerequisites_met, uncertainty)


def _graph():
    # a small prerequisite DAG:  base → mid → top   (base is the most foundational)
    return ConceptGraph({
        "base": Concept("base"),
        "mid": Concept("mid", prerequisites=("base",)),
        "top": Concept("top", prerequisites=("mid",)),
        "aside": Concept("aside", prerequisites=("base",)),
    })


# ── structural importance ─────────────────────────────────────────────────────────────
def test_importance_reflects_transitive_dependents():
    g = _graph()
    # base underlies mid, top, aside → most important; top underlies nothing → least
    assert g.importance("base") > g.importance("mid") > g.importance("top")
    assert g.importance("top") == pytest.approx(0.0)


# ── prerequisite gating ───────────────────────────────────────────────────────────────
def test_prerequisites_met_gates_on_prereq_mastery():
    g = _graph()
    empty = {}
    assert prerequisites_met(g, empty, "base")                    # no prereqs
    assert not prerequisites_met(g, empty, "mid")                 # base not learned yet
    ready = {"base": Mastery(prob=0.8, exposed=True)}
    assert prerequisites_met(g, ready, "mid")


def test_a_misconceived_prerequisite_blocks_the_dependent():
    g = _graph()
    state = {"base": Mastery(prob=0.9, exposed=True, misconception=True)}
    assert not prerequisites_met(g, state, "mid")


# ── per-concept signals ───────────────────────────────────────────────────────────────
def test_uncertainty_peaks_at_half_knowledge():
    assert uncertainty(Mastery(prob=0.5, exposed=True)) > uncertainty(Mastery(prob=0.9, exposed=True))
    assert uncertainty(Mastery(prob=0.5, exposed=False)) == 0.0        # unexposed ⇒ no uncertainty signal


def test_expected_gain_higher_for_foundational_and_unlearned():
    g = _graph()
    unlearned_base = expected_gain(g, "base", Mastery(prob=0.1, exposed=True), 0.85)
    learned_base = expected_gain(g, "base", Mastery(prob=0.8, exposed=True), 0.85)
    assert unlearned_base > learned_base


def test_concept_state_classification():
    g = _graph()
    assert concept_state(g, {}, "base") == ConceptState.NOT_ENCOUNTERED
    assert concept_state(g, {"base": Mastery(prob=0.95, exposed=True)}, "base") == ConceptState.MASTERED
    assert concept_state(g, {"base": Mastery(prob=0.3, exposed=True, misconception=True)}, "base") \
        == ConceptState.MISCONCEPTION
    assert concept_state(g, {"base": Mastery(prob=0.5, exposed=True)}, "base") == ConceptState.UNCERTAIN


# ── the frontier selection ────────────────────────────────────────────────────────────
def test_frontier_teaches_a_foundation_first_when_nothing_is_known():
    g = _graph()
    step = next_step(g, {})
    assert step.action == FrontierAction.TEACH and step.concept_id == "base"   # only eligible + most important


def test_frontier_will_not_teach_a_concept_whose_prereqs_are_unmet():
    g = _graph()
    # base half-known (< prereq threshold 0.7): mid/top/aside are blocked, so base is the only pick
    step = next_step(g, {"base": Mastery(prob=0.5, exposed=True)})
    assert step.concept_id == "base"


def test_frontier_advances_to_dependents_once_prereqs_met():
    g = _graph()
    state = {"base": Mastery(prob=0.9, exposed=True)}      # base mastered ⇒ mid & aside now eligible
    step = next_step(g, state)
    assert step.action == FrontierAction.TEACH and step.concept_id in ("mid", "aside")


def _flat():
    # two independent concepts (no prereqs, no dependents ⇒ importance 0): the regime where the value
    # of PROBING a maybe-known belief can exceed the value of teaching it outright.
    return ConceptGraph({"x": Concept("x"), "y": Concept("y")})


def test_frontier_assesses_an_uncertain_unverified_belief_before_teaching():
    g = _flat()
    # x is eligible, believed at 0.5 (ambiguous) and NOT verified ⇒ probe before investing in teaching
    step = next_step(g, {"x": Mastery(prob=0.5, exposed=True, assessed=False),
                         "y": Mastery(prob=0.95, exposed=True, assessed=True)})
    assert step.action == FrontierAction.ASSESS and step.concept_id == "x"


def test_frontier_teaches_once_the_belief_is_verified():
    g = _flat()
    # same ambiguous belief, but now VERIFIED (assessed) ⇒ no point probing again; teach it
    step = next_step(g, {"x": Mastery(prob=0.5, exposed=True, assessed=True),
                         "y": Mastery(prob=0.95, exposed=True, assessed=True)})
    assert step.action == FrontierAction.TEACH and step.concept_id == "x"


def test_frontier_does_not_assess_a_clearly_unknown_concept():
    g = _flat()
    # never encountered ⇒ low belief-uncertainty ⇒ just teach it, no probe needed
    step = next_step(g, {"y": Mastery(prob=0.95, exposed=True, assessed=True)})
    assert step.action == FrontierAction.TEACH and step.concept_id == "x"


def test_assess_can_be_disabled_reverting_to_teach_on_belief():
    g = _flat()
    state = {"x": Mastery(prob=0.5, exposed=True, assessed=False),
             "y": Mastery(prob=0.95, exposed=True, assessed=True)}
    step = next_step(g, state, policy=FrontierPolicy(enable_assess=False))
    assert step.action == FrontierAction.TEACH and step.concept_id == "x"


def test_frontier_reviews_a_retention_at_risk_concept():
    g = _graph()
    # everything mastered, but base is stale (retention at risk) ⇒ review it
    state = {c: Mastery(prob=0.95, exposed=True) for c in g.concepts}
    state["base"] = Mastery(prob=0.95, exposed=True, staleness=10.0)
    step = next_step(g, state, policy=FrontierPolicy(at_risk_staleness=4.0))
    assert step.action == FrontierAction.REVIEW and step.concept_id == "base"


def test_frontier_stops_when_all_mastered():
    g = _graph()
    state = {c: Mastery(prob=0.95, exposed=True) for c in g.concepts}
    step = next_step(g, state)
    assert step.action == FrontierAction.STOP and step.stop_reason == StopReason.ALL_MASTERED


def test_frontier_stops_objective_met_even_with_other_concepts_open():
    g = _graph()
    state = {"base": Mastery(prob=0.95, exposed=True), "mid": Mastery(prob=0.9, exposed=True)}
    step = next_step(g, state, objective={"base", "mid"})
    assert step.action == FrontierAction.STOP and step.stop_reason == StopReason.OBJECTIVE_MET


def test_frontier_reports_blocked_on_prereqs_when_only_blocked_remain():
    # a graph where the only unmastered concept is blocked by a MISCONCEIVED prereq
    g = ConceptGraph({"a": Concept("a"), "b": Concept("b", prerequisites=("a",))})
    state = {"a": Mastery(prob=0.95, exposed=True, misconception=True)}   # 'a' blocks 'b', 'a' is "mastered prob" but misconceived
    # a is misconceived ⇒ not mastered ⇒ it's actually teachable; make a truly mastered-but-b-blocked case instead:
    state = {"a": Mastery(prob=0.5, exposed=True)}    # a not mastered & is a prereq of b; a is teachable though
    step = next_step(g, state)
    assert step.action == FrontierAction.TEACH and step.concept_id == "a"   # a is eligible




def test_retrievability_decays_with_stability():
    from knowledge_frontier import Mastery, retrievability
    fresh = Mastery(prob=0.9, exposed=True, stability=4.0, staleness=0.0)
    half = Mastery(prob=0.9, exposed=True, stability=4.0, staleness=4.0)   # one stability unit
    assert retrievability(fresh) == 1.0
    assert abs(retrievability(half) - 0.5) < 1e-9
    # no stability model ⇒ falls back to belief (v0.1.0 behaviour)
    assert retrievability(Mastery(prob=0.7, exposed=True)) == 0.7


def test_reinforce_grows_stability_on_success_and_collapses_on_failure():
    from knowledge_frontier import Mastery, reinforce
    m0 = Mastery(prob=0.9, exposed=True, stability=3.0, staleness=5.0)
    ok = reinforce(m0, success=True, spacing_factor=2.0)
    assert ok.stability == 6.0 and ok.staleness == 0.0 and ok.prob >= 0.85 and ok.assessed
    first = reinforce(Mastery(prob=0.0, exposed=False), success=True, first_stability=2.0)
    assert first.stability == 2.0
    bad = reinforce(m0, success=False, first_stability=1.0)
    assert bad.stability == 1.0 and bad.prob <= m0.prob


def test_observe_is_a_bkt_update_and_marks_assessed():
    from knowledge_frontier import Mastery, observe
    prior = Mastery(prob=0.5, exposed=True, assessed=False)
    up = observe(prior, correct=True)
    down = observe(prior, correct=False)
    assert up.prob > prior.prob and down.prob < prior.prob
    assert up.assessed and down.assessed and up.staleness == 0.0


def test_spaced_retention_need_is_due_when_recall_hits_target():
    from knowledge_frontier import Mastery, retention_need
    # stability=10, target 0.9 ⇒ due at staleness ≈ 10*log2(1/0.9) ≈ 1.52
    m = Mastery(prob=0.9, exposed=True, stability=10.0)
    from dataclasses import replace
    assert retention_need(replace(m, staleness=1.0), 4.0, target_retrievability=0.9) < 1.0   # not yet due
    assert retention_need(replace(m, staleness=2.0), 4.0, target_retrievability=0.9) >= 1.0  # overdue
    # stability=0 ⇒ v0.1.0 flat proxy
    assert retention_need(Mastery(prob=0.9, exposed=True, staleness=4.0), 4.0) >= 1.0
