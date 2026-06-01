"""
F-5: Effective-predicate inference from FFNRule conditions.

Given an FFNRule, returns a predicate over token-position state
characterizing positions where the rule CAN fire. Conservative
over-approximation: the actual firing set is a SUBSET of the
returned predicate's satisfying positions.
"""

from __future__ import annotations

from typing import List

from neural_vm.dim_registry import DimRegistry
from neural_vm.unified_compiler.ir import FFNRule
from neural_vm.unified_compiler.predicates import (
    And,
    Atom,
    Not,
    Or,
    Predicate,
    parse,
    satisfiable,
)


HARD_BLOCKER_THRESHOLD = 1e6   # |weight| >= this is treated as hard NOT-blocker

# Maximum disjunct count permitted in any single contribution to the
# AND-of-positives / AND-of-NOT-blockers. Predicates whose NNF expansion
# would exceed this are dropped (weaker over-approximation, still safe).
_MAX_DISJUNCTS_PER_CONTRIB = 4

# Total DNF size budget: product of all kept contributions' disjunct
# counts must not exceed this. Once the budget is exhausted, remaining
# multi-disjunct contributions are dropped (further weakening).
_TOTAL_DNF_BUDGET = 256


# Sentinel tautology expressed via the DSL: returned when we have no
# useful information to constrain the over-approximation.
_TAUTOLOGY = parse("step_is_fresh OR NOT step_is_fresh")

# Sentinel contradiction expressed via the DSL: returned when the AND of
# positive condition semantics (and NOT-blockers) is unsatisfiable -- i.e.,
# the rule's effective firing set is empty under our over-approximation,
# meaning two condition semantics structurally contradict (e.g., MARK_MEM
# AND CMP+X-with-mark==AX). entails(contradiction, anything) returns True
# vacuously, so callers treating effective_predicate output as the firing
# set under-approximation correctly conclude "rule fires nowhere -> any
# scope claim trivially holds."
_CONTRADICTION = parse("is_byte AND NOT is_byte")


def _nnf_disjunct_count(p: Predicate, negated: bool = False) -> int:
    """Estimate the number of disjuncts that `p` (optionally negated)
    would contribute to a DNF cross-product, in NNF form.

    Multiplicative across And children; additive across Or children.
    Atoms count as 1. Used to bound contributions and avoid exponential
    blowup downstream in `entails()`.
    """
    if isinstance(p, Atom):
        return 1
    if isinstance(p, Not):
        return _nnf_disjunct_count(p.child, not negated)
    if isinstance(p, And):
        if negated:
            # NOT (A AND B) -> NOT A OR NOT B -> additive
            return sum(_nnf_disjunct_count(c, True) for c in p.children)
        # AND -> multiplicative cross-product
        total = 1
        for c in p.children:
            total *= _nnf_disjunct_count(c, False)
            if total > _MAX_DISJUNCTS_PER_CONTRIB * 4:
                return total  # short-circuit
        return total
    if isinstance(p, Or):
        if negated:
            # NOT (A OR B) -> NOT A AND NOT B -> multiplicative
            total = 1
            for c in p.children:
                total *= _nnf_disjunct_count(c, True)
                if total > _MAX_DISJUNCTS_PER_CONTRIB * 4:
                    return total
            return total
        return sum(_nnf_disjunct_count(c, False) for c in p.children)
    return 1


def effective_predicate(rule: FFNRule, registry: DimRegistry) -> Predicate:
    """Return the predicate over token-position state characterizing
    positions where `rule` can fire under `registry`.

    Linear-time conservative over-approximation:
      effective = (AND of every positive-weight condition's semantics)
                  AND (AND of NOT for every hard-blocker condition's
                   semantics)

    Contributions whose NNF expansion would exceed
    ``_MAX_DISJUNCTS_PER_CONTRIB`` disjuncts are dropped to avoid
    exponential blowup in downstream DNF-based `entails()` evaluation.
    Dropping is a STRICTLY WEAKER over-approximation: it admits more
    positions, never fewer. Reflexive entailment and the unit test
    expectations remain intact.

    The result is STRICTLY MORE CONSERVATIVE than the true firing
    predicate -- it admits more positions, never fewer. Suitable for
    "effective |= scope" entailment checking: passing tests in this
    approximation still correspond to true scope satisfaction.

    O(N) in number of conditions.
    """
    # First pass: collect (predicate, disjunct_count) for positives and
    # negated-disjunct_count for hard blockers, dropping any single
    # contribution that exceeds the per-contribution cap.
    pos_candidates: List[tuple] = []   # (pred, disjunct_count)
    blocker_candidates: List[tuple] = []   # (pred, disjunct_count_when_negated)

    for term in rule.conditions:
        sem_str = registry.semantics(term.dim.name)

        if sem_str is None:
            # Conservatively treat as tautology -- don't constrain.
            continue

        sem_pred = parse(sem_str)

        if term.weight > 0:
            dc = _nnf_disjunct_count(sem_pred, negated=False)
            if dc > _MAX_DISJUNCTS_PER_CONTRIB:
                continue
            pos_candidates.append((sem_pred, dc))
        elif term.weight < 0:
            if abs(term.weight) >= HARD_BLOCKER_THRESHOLD:
                dc = _nnf_disjunct_count(sem_pred, negated=True)
                if dc > _MAX_DISJUNCTS_PER_CONTRIB:
                    continue
                blocker_candidates.append((sem_pred, dc))
            # else: soft blocker, dropped in over-approximation
        # weight == 0: skip

    # Second pass: respect total DNF budget. Prefer atomic (dc==1)
    # contributions first; spend budget on multi-disjunct contributions
    # until exhausted.
    parts: List[Predicate] = []
    running_product = 1

    # Cheap (atomic / single-disjunct) contributions first.
    for pred, dc in pos_candidates:
        if dc <= 1:
            parts.append(pred)
    for pred, dc in blocker_candidates:
        if dc <= 1:
            parts.append(Not(pred))

    # Multi-disjunct contributions, only while the budget allows.
    for pred, dc in pos_candidates:
        if dc <= 1:
            continue
        if running_product * dc > _TOTAL_DNF_BUDGET:
            continue
        parts.append(pred)
        running_product *= dc

    for pred, dc in blocker_candidates:
        if dc <= 1:
            continue
        if running_product * dc > _TOTAL_DNF_BUDGET:
            continue
        parts.append(Not(pred))
        running_product *= dc

    if not parts:
        return _TAUTOLOGY

    if len(parts) == 1:
        result = parts[0]
    else:
        result = And(tuple(parts))

    # If the AND of positives + NOT-blockers is unsatisfiable, the
    # rule's effective firing set is structurally empty: two or more
    # positive-weight condition semantics contradict (e.g., MARK_MEM
    # AND a CMP+X-style condition whose semantics force mark == AX).
    # Returning the original AND would expose `entails(effective, scope)`
    # to a non-trivial firing set, manufacturing false-positive
    # scope_violation flags. Collapse to a contradiction sentinel; the
    # patched entails() (in predicates.py) treats internally-contradictory
    # disjuncts as vacuously entailing anything.
    if not satisfiable(result):
        return _CONTRADICTION

    return result
