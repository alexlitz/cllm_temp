"""
S-2: Contribution algebra for FFNRules.

Compute conservative upper bounds on a rule's pre-softmax contribution
to a specific output dim. Used by strength/margin verification to check
which rule must dominate at each output dim.
"""

from typing import Optional
from neural_vm.unified_compiler.ir import FFNRule


def max_contribution(rule: FFNRule, output_dim_name: str, *, output_offset: Optional[int] = None) -> float:
    """Conservative upper bound on `rule`'s signed additive contribution
    to the pre-softmax logit at output dim `output_dim_name` (optionally
    at a specific offset within a one-hot family), under any position
    in the rule's effective firing scope.

    Returns 0.0 if the rule doesn't write to that output_dim.

    Algebra:
      max_activation = max(0, sum_positive_condition_weights - threshold)
      contribution = write_weight_at_target_dim * max_activation

    Sign: positive for override rules (positive write_weight); negative
    for suppressor rules (negative write_weight). Callers that need to
    compare magnitudes across signs should use
    :func:`signed_contribution_bound`.
    """
    write_weight = _write_weight_for_dim(rule, output_dim_name, output_offset)
    if write_weight == 0.0:
        return 0.0

    positive_sum = sum(t.weight for t in rule.conditions if t.weight > 0)
    max_activation = max(0.0, positive_sum - rule.threshold)
    return write_weight * max_activation


def signed_contribution_bound(
    rule: FFNRule,
    output_dim_name: str,
    *,
    output_offset: Optional[int] = None,
) -> tuple[float, str]:
    """Return ``(magnitude, sign)`` for ``rule``'s bound at the given
    output dim, where ``magnitude >= 0`` and ``sign`` is one of
    ``'positive'`` (override rule, wants this dim to win argmax) or
    ``'negative'`` (suppressor rule, wants this dim to lose argmax).

    The sign is decided by the rule's *write_weight* at this dim — not
    by the resulting contribution — so that suppressor sub-writes (e.g.
    a 16-wide one-hot's negative competitors) remain classified as
    suppressors even when the rule's activation upper bound is zero.

    Override rules and suppressor rules pursue different goals at a
    shared output dim and therefore do not directly compete in the
    dominance algebra.
    """
    write_weight = _write_weight_for_dim(rule, output_dim_name, output_offset)
    contrib = max_contribution(rule, output_dim_name, output_offset=output_offset)
    sign = 'negative' if write_weight < 0 else 'positive'
    return (abs(contrib), sign)


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
    the given output dim, comparing only same-sign contributions.

    Override rules (positive write_weight) and suppressor rules
    (negative write_weight) have different goals at a shared dim, so
    cross-sign rules are treated as non-competing. Within a sign,
    dominance compares contribution magnitudes:

    * Override: dominator's positive contribution magnitude must be
      >= max(other positives) + backbone + margin.
    * Suppressor: dominator's negative contribution magnitude must be
      >= max(other negative magnitudes) + backbone + margin.

    Returns (is_dominant, margin_actual). ``margin_actual`` is
    ``rule_magnitude - (competing_max + backbone_bound)``.
    """
    rule_mag, rule_sign = signed_contribution_bound(
        rule, output_dim_name, output_offset=output_offset
    )
    same_sign_mags: list[float] = []
    for r in competing_rules:
        m, s = signed_contribution_bound(
            r, output_dim_name, output_offset=output_offset
        )
        if s == rule_sign:
            same_sign_mags.append(m)
    competing_max = max(same_sign_mags, default=0.0)
    required = competing_max + backbone_bound + margin
    return (rule_mag >= required, rule_mag - (competing_max + backbone_bound))
