"""Building-Blocks DSL — typed FFNRule constructors for the foundational
primitives documented in ``docs/BLOG_SPEC.md`` §504-568.

This module promotes the "Building Blocks" patterns from the blog post
to first-class DSL constructors. Each helper takes typed inputs (dim
names, integer thresholds, scale ``S``) and returns either one
``FFNRule`` or a ``tuple[FFNRule, ...]`` that lowers through
``Primitives.lower_ffn_rules`` unchanged.

**Discrete vs. continuous indicators.** The blog's smooth ``+1/-2/+1``
3-SiLU zero indicator (§510-518) is a *continuous-math* primitive.
Under the IR's hard-threshold symbolic semantics, the same indicator
that's clean neurally diverges from its discrete reading. The
c4-release codebase universally encodes integer values as **one-hot
bands** (e.g., ``ALU_LO+k`` for k∈[0,15]), so the practical primitives
exposed here operate on one-hot binary inputs: ``one_hot_indicator_rule``
is a single FFNRule that fires iff ``band+value`` is set;
``band_range_check_rules`` emits one rule per in-range index. This is
the byte-identical version that matches every existing call site.

Blog primitive → constructor:

| §504 Step function           | ``step_function_rule``       |
| §510 Point/value indicator   | ``one_hot_indicator_rule``   |
| §510 (special) Zero indicator| ``one_hot_indicator_rule(value=0)``  |
| §524 Range check             | ``band_range_check_rules``   |
| §522 Cancel residual         | ``cancel_residual_rule``     |
| §590 N-way balanced AND      | ``multi_way_and_rule``       |
|      N-way OR                | ``multi_way_or_rules``       |
| §520 Lookup table            | ``lookup_table_rules``       |

Wave V2 will add ``magic_floor_rules``, ``bit_range_extract_rules``,
attention helpers (``efficient_exp_attention``,
``memory_load_attention``, ``fetch_byte_attention``), and the MoE
wrapper ``opcode_expert_rules``.
"""

from __future__ import annotations

from typing import Iterable, Mapping, Optional, Sequence, Tuple

from .ir import FFNRule


# ---------------------------------------------------------------------------
# Default write amplitude used across the codebase for lookup-mode writes.
#
# Convention (mirrors ``wide_alu_dsl.bitwise_rules``): write_weight = 2.0/S.
# At scale S=100, this gives output magnitude ≈ 1.0 when the rule fires on
# a binary one-hot condition (silu(S*0.5)/S * 2.0 ≈ 50/100 * 2 = 1.0).
# Callers may override via ``write_value``; the helper passes the value
# through to ``FFNRule.constant_write`` / ``gated_write``.
# ---------------------------------------------------------------------------

_DEFAULT_WRITE = 2.0  # raw numerator; lowered as write_value / S.


def _ww(value: float, S: float) -> float:
    """Normalize ``value`` to a per-rule write_weight at scale ``S``."""

    return value / S


def _build_rule(
    *,
    conditions: Sequence[Tuple[str, float]],
    threshold: float,
    writes: Sequence[Tuple[str, float]],
    gate: Optional[str],
    gate_terms: Sequence[Tuple[str, float]],
    gate_weight: float = 1.0,
    name: Optional[str],
    scope: Optional[str],
    dominates_at: Optional[Mapping[str, str]] = None,
) -> FFNRule:
    """Dispatch to ``FFNRule.constant_write`` or ``gated_write`` depending
    on whether a gate is supplied. Keeps the gate_bias convention
    consistent (1.0 for constant_write, 0.0 for gated_write).
    """
    if gate is None and not gate_terms:
        return FFNRule.constant_write(
            name=name,
            conditions=tuple(conditions),
            threshold=float(threshold),
            writes=tuple(writes),
            scope=scope,
            dominates_at=dominates_at,
        )
    return FFNRule.gated_write(
        name=name,
        conditions=tuple(conditions),
        threshold=float(threshold),
        gate=gate,
        gate_weight=gate_weight,
        gate_terms=tuple(gate_terms),
        gate_bias=0.0,
        writes=tuple(writes),
        scope=scope,
        dominates_at=dominates_at,
    )


# ===========================================================================
# §504 — Step function
# ===========================================================================


def step_function_rule(
    *,
    input_dim: str,
    threshold: float,
    write_dim: str,
    write_value: float = _DEFAULT_WRITE,
    gate: Optional[str] = None,
    gate_terms: Sequence[Tuple[str, float]] = (),
    S: float = 100.0,
    name: Optional[str] = None,
    scope: Optional[str] = None,
) -> FFNRule:
    """Heaviside-like step function: fires when ``input_dim`` >= ``threshold``.

    The blog's smooth ``H_ε(x) = silu(S(x+ε)) - silu(S(x-ε))`` reduces to
    a single hard-thresholded ``FFNRule`` under the IR's symbolic
    semantics. For scalar integer inputs, pick ``threshold = target - 0.5``
    so the rule fires iff input is at or above the target integer. For
    one-hot dims (which are 0 or 1), pick ``threshold = 0.5`` to fire
    iff the dim is set — see ``one_hot_indicator_rule`` for that sugar.

    Output: ``write_value / S`` written to ``write_dim`` if input >=
    threshold else 0.

    Args:
        input_dim: name of the residual dim being thresholded (e.g.
            ``"NIBBLE+5"``).
        threshold: scalar threshold.
        write_dim: name of the output residual dim.
        write_value: numerator of the write weight; lowered as
            ``write_value / S``. Default ``2.0`` gives ≈1.0 output
            magnitude at S=100 for a binary one-hot input.
        gate: optional multiplicative gate dim.
        gate_terms: optional additive gate terms.
        S: SwiGLU scale.
        name: optional rule name (for verifier diagnostics).
        scope: optional predicate-DSL scope.

    Returns:
        One ``FFNRule``.
    """
    return _build_rule(
        conditions=((input_dim, 1.0),),
        threshold=threshold,
        writes=((write_dim, _ww(write_value, S)),),
        gate=gate,
        gate_terms=gate_terms,
        name=name,
        scope=scope,
    )


# ===========================================================================
# §510 — One-hot value indicator (discrete realization of point indicator)
# ===========================================================================


def one_hot_indicator_rule(
    *,
    band: str,
    value: int,
    write_dim: str,
    write_value: float = _DEFAULT_WRITE,
    gate: Optional[str] = None,
    gate_terms: Sequence[Tuple[str, float]] = (),
    S: float = 100.0,
    name: Optional[str] = None,
    scope: Optional[str] = None,
) -> FFNRule:
    """Point indicator over a one-hot ``band``: fires iff ``band+value``
    is the active cell.

    This is the discrete/byte-identical version of the blog's smooth
    ``+1/-2/+1`` zero detector (§510). Since the codebase encodes
    integer values as one-hot 16-band cells (one cell per value), the
    indicator collapses to a single ``FFNRule`` reading the specific
    cell with threshold ``0.5``.

    Args:
        band: base dim of the one-hot band (e.g. ``"ALU_LO"``).
        value: the integer cell to fire on (typically 0..15).
        write_dim: output residual dim.
        write_value: numerator of write weight; lowered as
            ``write_value / S``.
        gate / gate_terms / S / name / scope: as in
            ``step_function_rule``.

    Returns:
        One ``FFNRule``.
    """
    return step_function_rule(
        input_dim=f"{band}+{value}",
        threshold=0.5,
        write_dim=write_dim,
        write_value=write_value,
        gate=gate,
        gate_terms=gate_terms,
        S=S,
        name=name,
        scope=scope,
    )


# ===========================================================================
# §524 — Range check over a one-hot band
# ===========================================================================


def band_range_check_rules(
    *,
    band: str,
    lo: int,
    hi: int,
    write_dim: str,
    write_value: float = _DEFAULT_WRITE,
    gate: Optional[str] = None,
    gate_terms: Sequence[Tuple[str, float]] = (),
    S: float = 100.0,
    name: Optional[str] = None,
    scope: Optional[str] = None,
) -> Tuple[FFNRule, ...]:
    """Range check ``R_{lo,hi}`` over a one-hot ``band``: fires iff the
    active cell index is in ``[lo, hi]`` (inclusive integer range).

    Discrete realization of blog §524. Emits one rule per in-range
    cell index, each writing ``write_value`` to ``write_dim``. For a
    one-hot input, exactly one rule fires (or zero, if the active cell
    is out of range).

    Args:
        band: base dim of the one-hot band.
        lo / hi: inclusive integer bounds (lo <= hi).
        write_dim / write_value / gate / gate_terms / S / name / scope:
            applied to every emitted rule.

    Returns:
        Tuple of ``hi - lo + 1`` ``FFNRule``s.

    Raises:
        ValueError: if ``lo > hi``.
    """
    if lo > hi:
        raise ValueError(
            f"band_range_check_rules: lo ({lo}) > hi ({hi}) — empty range"
        )
    rules: list[FFNRule] = []
    for k in range(lo, hi + 1):
        suffix = f"{name}_at_{k}" if name else None
        rules.append(one_hot_indicator_rule(
            band=band,
            value=k,
            write_dim=write_dim,
            write_value=write_value,
            gate=gate,
            gate_terms=gate_terms,
            S=S,
            name=suffix,
            scope=scope,
        ))
    return tuple(rules)


# ===========================================================================
# §590 — N-way conjunction / disjunction
# ===========================================================================


def multi_way_and_rule(
    *,
    conditions: Sequence[Tuple[str, float]],
    writes: Sequence[Tuple[str, float]],
    gate: Optional[str] = None,
    gate_terms: Sequence[Tuple[str, float]] = (),
    gate_weight: float = 1.0,
    threshold: Optional[float] = None,
    name: Optional[str] = None,
    scope: Optional[str] = None,
    dominates_at: Optional[Mapping[str, str]] = None,
) -> FFNRule:
    """N-way balanced AND across ``conditions``: fires iff ALL conditions
    are active (each a one-hot binary dim).

    The blog's 3-way construction ``marker(40) + a(30) + b(30) > 80``
    generalizes naturally. Pass weighted conditions explicitly + a
    matched ``threshold`` to reproduce hand-derived patterns. When
    ``threshold is None``, the helper derives it as
    ``(total + total - max_w) / 2.0`` so the "missing one" sum is
    below threshold and the "all on" sum is above.

    Args:
        conditions: list of ``(dim_name, weight)`` tuples.
        writes: list of ``(dim_name, weight)`` output writes.
        gate / gate_terms / gate_weight: optional multiplicative gate.
        threshold: explicit threshold; derived if ``None``.
        name / scope / dominates_at: passed through to FFNRule.

    Returns:
        One ``FFNRule``.

    Raises:
        ValueError: empty ``conditions``, or non-positive weights with
            default-derived threshold.
    """
    cond_tuple = tuple(conditions)
    if not cond_tuple:
        raise ValueError("multi_way_and_rule: conditions must be non-empty")
    if threshold is None:
        weights = [w for _, w in cond_tuple]
        if any(w <= 0 for w in weights):
            raise ValueError(
                "multi_way_and_rule: default threshold derivation requires "
                "strictly positive condition weights; got "
                f"{weights!r}. Pass explicit threshold=."
            )
        total = sum(weights)
        max_w = max(weights)
        threshold = (total + (total - max_w)) / 2.0

    return _build_rule(
        conditions=cond_tuple,
        threshold=threshold,
        writes=tuple(writes),
        gate=gate,
        gate_terms=tuple(gate_terms),
        gate_weight=gate_weight,
        name=name,
        scope=scope,
        dominates_at=dominates_at,
    )


def multi_way_or_rules(
    *,
    conditions: Sequence[str],
    writes: Sequence[Tuple[str, float]],
    gate: Optional[str] = None,
    gate_terms: Sequence[Tuple[str, float]] = (),
    S: float = 100.0,
    name: Optional[str] = None,
    scope: Optional[str] = None,
) -> Tuple[FFNRule, ...]:
    """N-way OR across binary ``conditions``: emits one rule per
    condition, each firing iff that condition is active.

    For one-hot conditions exactly one rule fires per state (clean OR).
    For overlapping conditions, writes accumulate — caller must ensure
    exclusivity if pure OR semantics are needed.

    Args:
        conditions: list of dim names (each binary 0/1).
        writes: output writes applied per firing rule.
        gate / gate_terms: optional shared multiplicative gate.
        S: SwiGLU scale (preserved in API for symmetry).
        name: optional name prefix.
        scope: optional shared scope.

    Returns:
        Tuple of ``len(conditions)`` ``FFNRule``s.
    """
    del S
    rules: list[FFNRule] = []
    for cond in conditions:
        suffix = f"{name}_or_{cond.replace('+', '_')}" if name else None
        rules.append(_build_rule(
            conditions=((cond, 1.0),),
            threshold=0.5,
            writes=tuple(writes),
            gate=gate,
            gate_terms=tuple(gate_terms),
            name=suffix,
            scope=scope,
        ))
    return tuple(rules)


# ===========================================================================
# §522 — Cancelling residuals
# ===========================================================================


def cancel_residual_rule(
    *,
    input_dim: str,
    output_dim: Optional[str] = None,
    write_value: float = _DEFAULT_WRITE,
    gate: Optional[str] = None,
    gate_terms: Sequence[Tuple[str, float]] = (),
    S: float = 100.0,
    name: Optional[str] = None,
    scope: Optional[str] = None,
) -> FFNRule:
    """Cancel the residual at ``input_dim`` (write a negation back when
    the input is active).

    Blog §522 describes the bias-``1.27846`` single-unit pattern where
    ``SiLU(b) ≈ 1``, ``up = 1``, ``down = -1``. Under the IR's hard-
    threshold semantics, this lowers to a one-shot rule that fires on
    ``input_dim`` being set and writes ``-write_value / S`` back to
    ``output_dim`` (defaulting to the same dim).

    Args:
        input_dim: the dim whose value to cancel.
        output_dim: target of the cancellation write. Defaults to
            ``input_dim`` (self-cancel).
        write_value: positive magnitude of the cancellation; lowered as
            ``-write_value / S``.
        gate / gate_terms: optional multiplicative gating.
        S: SwiGLU scale.
        name / scope: passed through.

    Returns:
        One ``FFNRule``.
    """
    target = output_dim if output_dim is not None else input_dim
    return _build_rule(
        conditions=((input_dim, 1.0),),
        threshold=0.5,
        writes=((target, -_ww(write_value, S)),),
        gate=gate,
        gate_terms=tuple(gate_terms),
        name=name,
        scope=scope,
    )


# ===========================================================================
# §520 — Lookup tables (one-hot keyed)
# ===========================================================================


def lookup_table_rules(
    *,
    key_band: str,
    key_to_writes: Mapping[int, Sequence[Tuple[str, float]]],
    write_value_scale: float = 1.0,
    gate: Optional[str] = None,
    gate_terms: Sequence[Tuple[str, float]] = (),
    S: float = 100.0,
    name_prefix: Optional[str] = None,
    scope: Optional[str] = None,
) -> Tuple[FFNRule, ...]:
    """N-entry lookup table keyed on a one-hot ``key_band``.

    For each (``key``, ``writes_for_key``) pair, emits one ``FFNRule``
    that fires when ``key_band+key`` is set and applies the per-key
    writes. Each per-key write weight is scaled by
    ``write_value_scale / S``.

    Args:
        key_band: base dim of the one-hot key band (e.g. ``"NIBBLE"``).
        key_to_writes: mapping ``{key_value: [(out_dim, raw_weight), ...]}``.
            ``raw_weight`` is scaled by ``write_value_scale / S``.
        write_value_scale: numerator factor; default 1.0 means the
            caller passes weight = output_magnitude directly.
        gate / gate_terms: optional multiplicative gating applied to
            every key entry.
        S: SwiGLU scale.
        name_prefix: optional rule name prefix.
        scope: optional shared scope predicate.

    Returns:
        Tuple of ``len(key_to_writes)`` ``FFNRule``s.
    """
    rules: list[FFNRule] = []
    for key, writes_for_key in key_to_writes.items():
        scaled_writes = tuple(
            (out_dim, write_value_scale * raw_weight / S)
            for out_dim, raw_weight in writes_for_key
        )
        suffix = (
            f"{name_prefix}_at_{key}" if name_prefix else None
        )
        rules.append(_build_rule(
            conditions=((f"{key_band}+{key}", 1.0),),
            threshold=0.5,
            writes=scaled_writes,
            gate=gate,
            gate_terms=tuple(gate_terms),
            name=suffix,
            scope=scope,
        ))
    return tuple(rules)


__all__ = [
    "step_function_rule",
    "one_hot_indicator_rule",
    "band_range_check_rules",
    "multi_way_and_rule",
    "multi_way_or_rules",
    "cancel_residual_rule",
    "lookup_table_rules",
]
