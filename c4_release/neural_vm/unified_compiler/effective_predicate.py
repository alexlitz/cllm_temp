"""
F-5: Effective-predicate inference from FFNRule conditions.

Given an FFNRule, returns a predicate over token-position state
characterizing positions where the rule CAN fire. Conservative
over-approximation: the actual firing set is a SUBSET of the
returned predicate's satisfying positions.
"""

from __future__ import annotations

from itertools import combinations
from typing import List, Tuple

from neural_vm.dim_registry import DimRegistry
from neural_vm.unified_compiler.ir import FFNRule
from neural_vm.unified_compiler.predicates import (
    And,
    Not,
    Or,
    Predicate,
    parse,
)


HARD_BLOCKER_THRESHOLD = 1e6   # |weight| >= this is treated as hard NOT-blocker


# Sentinel predicates expressed via the DSL: a tautology / contradiction we
# can return when the over-approximation has no useful information.
_TAUTOLOGY = parse("step_is_fresh OR NOT step_is_fresh")
_CONTRADICTION = parse("step_is_fresh AND NOT step_is_fresh")


def _and_of(*children: Predicate) -> Predicate:
    """Build an And of `children`, collapsing singletons and dropping empties."""
    flat: List[Predicate] = []
    for c in children:
        if isinstance(c, And):
            flat.extend(c.children)
        else:
            flat.append(c)
    if not flat:
        return _TAUTOLOGY
    if len(flat) == 1:
        return flat[0]
    return And(tuple(flat))


def _or_of(*children: Predicate) -> Predicate:
    """Build an Or of `children`, collapsing singletons and dropping empties."""
    flat: List[Predicate] = []
    for c in children:
        if isinstance(c, Or):
            flat.extend(c.children)
        else:
            flat.append(c)
    if not flat:
        return _CONTRADICTION
    if len(flat) == 1:
        return flat[0]
    return Or(tuple(flat))


def effective_predicate(rule: FFNRule, registry: DimRegistry) -> Predicate:
    """Return the predicate over token-position state characterizing
    positions where `rule` can fire under `registry`.

    Conservative over-approximation: any position satisfying the
    returned predicate MIGHT be fired by the rule; positions NOT
    satisfying it cannot. Useful for checking whether a rule's
    intended `scope` predicate is consistent with its weights.
    """
    positives: List[Tuple[Predicate, float]] = []
    hard_blockers: List[Predicate] = []
    soft_blockers_ignored: List[str] = []

    for term in rule.conditions:
        sem_str = registry.semantics(term.dim.name)
        if sem_str is None:
            # Conservative: any-position. Don't constrain.
            sem_pred: Predicate = _TAUTOLOGY
        else:
            sem_pred = parse(sem_str)

        if term.weight > 0:
            positives.append((sem_pred, term.weight))
        elif term.weight < 0:
            if abs(term.weight) >= HARD_BLOCKER_THRESHOLD:
                hard_blockers.append(sem_pred)
            else:
                soft_blockers_ignored.append(term.dim.name)
        # weight == 0: skip

    # Find minimal subsets of positives summing >= threshold
    disjuncts = _minimal_firing_subsets(positives, rule.threshold)

    if not disjuncts:
        # Rule can never fire under our over-approximation; return false
        # by convention. Caller should treat as a warning.
        return _CONTRADICTION

    # Build OR over disjuncts, each ANDed with negated hard blockers
    blocker_pred: Predicate
    if hard_blockers:
        blocker_pred = _and_of(*[Not(b) for b in hard_blockers])
    else:
        blocker_pred = _TAUTOLOGY

    full_disjuncts: List[Predicate] = []
    for subset in disjuncts:
        conj = _and_of(*subset)
        if hard_blockers:
            conj = _and_of(conj, blocker_pred)
        full_disjuncts.append(conj)

    return _or_of(*full_disjuncts)


def _minimal_firing_subsets(
    positives: List[Tuple[Predicate, float]],
    threshold: float,
    max_subsets: int = 1024,
) -> List[List[Predicate]]:
    """Return predicates of minimal subsets of positives whose weights
    sum to >= threshold. Truncates if subset count would exceed
    max_subsets (returns coarsest approximation in that case)."""
    n = len(positives)
    if n == 0:
        # No positive evidence: rule cannot drive its sum above threshold
        # (assuming threshold > 0). Treat as "no firing subset".
        if threshold <= 0:
            # Any position fires trivially; signal via empty-conjunction.
            return [[]]
        return []
    if 2 ** n > max_subsets:
        # Coarse fallback: AND of all positives (assumes all must fire)
        return [[p for p, _ in positives]]

    found: List[Tuple[frozenset, List[Predicate]]] = []
    # Enumerate subsets in order of increasing size (so minimal subsets come first)
    for k in range(1, n + 1):
        for indices in combinations(range(n), k):
            subset = [positives[i] for i in indices]
            if sum(w for _, w in subset) >= threshold:
                # Check minimality: this index set is not a strict superset
                # of any previously found minimal subset.
                index_set = frozenset(indices)
                if any(prior_idx <= index_set for prior_idx, _ in found):
                    continue
                found.append((index_set, [p for p, _ in subset]))
    return [preds for _, preds in found]
