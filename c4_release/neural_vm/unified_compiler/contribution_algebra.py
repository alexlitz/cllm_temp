"""
S-2: Contribution algebra for FFNRules.

Compute conservative upper bounds on a rule's pre-softmax contribution
to a specific output dim. Used by strength/margin verification to check
which rule must dominate at each output dim.
"""

from typing import Optional
from neural_vm.unified_compiler.ir import FFNRule


def max_contribution(rule: FFNRule, output_dim_name: str, *, output_offset: Optional[int] = None) -> float:
    """Conservative upper bound on `rule`'s additive contribution to
    the pre-softmax logit at output dim `output_dim_name` (optionally
    at a specific offset within a one-hot family), under any position
    in the rule's effective firing scope.

    Returns 0.0 if the rule doesn't write to that output_dim.

    Algebra:
      max_activation = max(0, sum_positive_condition_weights - threshold)
      contribution = write_weight_at_target_dim * max_activation
    """
    write_weight = _write_weight_for_dim(rule, output_dim_name, output_offset)
    if write_weight == 0.0:
        return 0.0

    positive_sum = sum(t.weight for t in rule.conditions if t.weight > 0)
    max_activation = max(0.0, positive_sum - rule.threshold)
    return write_weight * max_activation


def _write_weight_for_dim(rule: FFNRule, output_dim_name: str, output_offset: Optional[int]) -> float:
    """Find the write weight for the given (name, offset). 0.0 if not written."""
    for wt in rule.writes:
        if wt.dim.name == output_dim_name:
            if output_offset is None or wt.dim.offset == output_offset:
                return wt.weight
    return 0.0


def write_dims(rule: FFNRule) -> list[tuple[str, int, float]]:
    """Return list of (output_dim_name, offset, write_weight) tuples that
    the rule writes to (with nonzero weight)."""
    return [(wt.dim.name, wt.dim.offset, wt.weight) for wt in rule.writes if wt.weight != 0.0]


def is_dominant_writer(
    rule: FFNRule,
    output_dim_name: str,
    competing_rules: list[FFNRule],
    *,
    output_offset: Optional[int] = None,
    backbone_bound: float = 0.0,
    margin: float = 1.0,
) -> tuple[bool, float]:
    """Check if `rule` dominates over `competing_rules` + backbone at
    the given output dim.

    Returns (is_dominant, margin_actual). is_dominant is True iff
      rule.contribution >= max(competing) + backbone + margin
    """
    rule_contrib = max_contribution(rule, output_dim_name, output_offset=output_offset)
    competing_max = max(
        (max_contribution(r, output_dim_name, output_offset=output_offset) for r in competing_rules),
        default=0.0,
    )
    required = competing_max + backbone_bound + margin
    return (rule_contrib >= required, rule_contrib - (competing_max + backbone_bound))
