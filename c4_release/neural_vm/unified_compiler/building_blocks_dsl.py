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
| §561 Efficient exp (softmax1)| ``efficient_exp_attention``  |
| (IR_DSL §3.3) Memory load    | ``memory_load_attention``    |
| (IR_DSL §3.3) Byte fetch     | ``fetch_byte_attention``     |
| §566 MoE opcode-gating       | ``opcode_expert_rules``      |
"""

from __future__ import annotations

import dataclasses
from typing import Iterable, Mapping, Optional, Sequence, Tuple

from .ir import ConditionTerm, DimRef, FFNRule
from .primitives import (
    AO,
    AP,
    AttentionOutputWrite,
    AttentionProjectionWrite,
    DeclarativeAttentionHeadSpec,
)


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
    gate_bias: Optional[float] = None,
    name: Optional[str],
    scope: Optional[str],
    dominates_at: Optional[Mapping[str, str]] = None,
) -> FFNRule:
    """Dispatch to ``FFNRule.constant_write`` or ``gated_write`` depending
    on whether a gate is supplied.

    ``gate_bias`` default behavior (``None``): 1.0 for constant_write
    path (no gate), 0.0 for gated_write path. Pass explicit value to
    override — needed for patterns with non-zero gate bias (e.g.
    ``gate_bias=-2.5`` for L9 BP+8 shift, ``gate_bias=-15.0`` for L9
    addr_b1 cascade).
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
        gate_bias=0.0 if gate_bias is None else float(gate_bias),
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
    gate_bias: Optional[float] = None,
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
        gate_bias: explicit gate bias; if ``None``, defaults to 1.0
            without a gate and 0.0 with one. Non-zero values are
            common in negative-bias gating (e.g. ``-2.5`` for L9
            BP+8 shift, ``-15.0`` for L9 addr_b1 cascade).
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
        gate_bias=gate_bias,
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

# ===========================================================================
# §561 — Efficient exp via softmax1 + ALiBi
# ===========================================================================


def efficient_exp_attention(
    *,
    head_idx: int,
    input_dim: str,
    output_dim: str,
    bos_token_key_dim: str,
    bias: float,
    head_dim: int = 64,
    input_slot: int = 0,
    value_slot: int = 0,
    bias_dim: str = "CONST",
    dim_positions: Mapping[str, int],
) -> DeclarativeAttentionHeadSpec:
    """Single-head softmax1+ALiBi-0 spec that approximates ``e^N`` (blog §561-564).

    The construction: with softmax1 (a ``+1`` term in the denominator
    matching the empty BOS row) and a single auxiliary BOS token row
    carrying ``K = sqrt(d_head)`` and ``V = e^B``, attention against a
    query position whose ``N`` lives in ``input_dim`` produces

        softmax1(Q · Kᵀ / sqrt(d_head)) · V
            = e^{(N-B) · sqrt(d_head) / sqrt(d_head)} / (1 + e^{N-B}) · e^B
            ≈ e^{N-B} · e^B = e^N      (when e^{N-B} ≫ 1)

    The ``B`` bias keeps the denominator exponent ≤ 1 so the softmax1
    "+1" anchor pushes denominator-mass to the BOS row, recovering a
    clean exp output on the V projection.

    Spec layout (one head):
      * Q: slot ``input_slot`` reads ``input_dim`` with weight 1 and
        ``bias_dim`` with weight ``-bias``. ``input_dim`` carries ``N``
        and the per-position constant ``bias_dim`` carries the offset
        ``-B``. Net Q-projection at the query position equals ``N - B``.
      * K: slot ``input_slot`` reads ``bos_token_key_dim`` with weight
        ``sqrt(head_dim)``. ``bos_token_key_dim`` is 1.0 at the BOS row
        and 0 elsewhere, so K = sqrt(head_dim) on that row alone.
      * V: slot ``value_slot`` reads ``bos_token_key_dim`` with weight
        ``e^B``. The V projection is non-zero **only on the BOS row**
        (because ``bos_token_key_dim`` is 1 there and 0 elsewhere),
        so the head's output at the query position is exactly the
        BOS-row attention weight times ``e^B``.
      * O: writes the matched V projection to ``output_dim``.
      * alibi_slope=0.0: the ALiBi-style position bias is disabled on
        this head (the blog construction has the exponent depend on
        ``N`` only, not on token-distance).

    Because ``Q · Kᵀ = (N - B) · sqrt(head_dim)`` for the BOS row, the
    pre-softmax score is ``(N - B) · sqrt(head_dim) / sqrt(head_dim) =
    N - B``. Softmax1 over a vector whose only non-trivial score is
    ``N - B`` (other tokens see K=0 → score=0) and whose "+1" anchor
    represents a virtual all-zero row gives weight
    ``e^{N-B} / (1 + e^{N-B} + (T-1))`` on BOS where T is the number of
    non-BOS tokens whose score is also 0. In the canonical two-token
    [BOS, query] case (T=1), the BOS weight is
    ``e^{N-B} / (2 + e^{N-B})`` and the head's output equals
    ``e^B · e^{N-B} / (2 + e^{N-B})``. With B chosen so ``N < B``
    (i.e. ``e^{N-B} ≪ 1``), this approximates ``e^B · e^{N-B} / 2 =
    e^N / 2``. The blog's exact "``≈ e^N``" identity requires the
    other non-BOS scores to also be suppressed (e.g. via causal
    masking that excludes self-attention) so the denominator
    collapses to ``1 + e^{N-B}``.

    Args:
        head_idx: attention head index (used by the head allocator).
        input_dim: residual dim holding the per-position scalar ``N``.
        output_dim: residual dim that receives ``e^N`` (approximately).
        bos_token_key_dim: residual dim that's 1.0 at the BOS token row
            and 0 elsewhere. Carries the K=sqrt(d_head) signal.
        bias: scalar ``B`` keeping the softmax1 denominator small. Pick
            ``B`` so ``N - B`` ranges over [0, ~5] in practice; the V
            multiplier ``e^B`` then recovers the full exponent magnitude.
        head_dim: per-head slot width ``d_head``. Default 64 (matches
            the standard c4 layer config). The query weight ``sqrt(d)``
            and key weight ``sqrt(d)`` cancel to leave ``N-B`` in the
            score.
        input_slot / value_slot: head-local slot indices for Q/K and V/O.
            Defaults to ``0``. ``input_slot`` must be < ``head_dim``.
        bias_dim: residual dim permanently held at ``1.0`` (used to
            carry the ``-B`` Q-side offset and the ``e^B`` V multiplier).
        dim_positions: dim layout map — required because attention
            specs are constructed with resolved dim ints.

    Returns:
        One ``DeclarativeAttentionHeadSpec`` with ``alibi_slope=0.0``.

    Raises:
        ValueError: if ``input_slot >= head_dim`` or any referenced dim
            is missing from ``dim_positions``.
    """
    import math

    if input_slot >= head_dim:
        raise ValueError(
            f"efficient_exp_attention: input_slot ({input_slot}) must be "
            f"< head_dim ({head_dim})"
        )
    if value_slot >= head_dim:
        raise ValueError(
            f"efficient_exp_attention: value_slot ({value_slot}) must be "
            f"< head_dim ({head_dim})"
        )
    for dim_name in (input_dim, output_dim, bos_token_key_dim, bias_dim):
        if dim_name not in dim_positions:
            raise ValueError(
                f"efficient_exp_attention: dim {dim_name!r} missing from "
                f"dim_positions"
            )

    sqrt_d = math.sqrt(float(head_dim))
    exp_bias = math.exp(float(bias))

    q_writes = (
        AP(input_slot, dim_positions[input_dim], 1.0),
        AP(input_slot, dim_positions[bias_dim], -float(bias)),
    )
    k_writes = (
        AP(input_slot, dim_positions[bos_token_key_dim], sqrt_d),
    )
    v_writes = (
        AP(value_slot, dim_positions[bos_token_key_dim], exp_bias),
    )
    o_writes = (
        AO(dim_positions[output_dim], value_slot, 1.0),
    )
    return DeclarativeAttentionHeadSpec(
        head_idx=int(head_idx),
        q=q_writes,
        k=k_writes,
        v=v_writes,
        o=o_writes,
        alibi_slope=0.0,
    )


# ===========================================================================
# IR_DSL §3.3 — Memory-load attention
# ===========================================================================


def memory_load_attention(
    *,
    head_idx: int,
    addr_query_dims: Sequence[str],
    addr_key_dims: Sequence[str],
    value_dims: Sequence[str],
    output_dims: Sequence[str],
    head_dim: int = 64,
    query_weight: float = 15.0,
    key_weight: float = 15.0,
    dim_positions: Mapping[str, int],
) -> DeclarativeAttentionHeadSpec:
    """Address-keyed memory-load attention head.

    Models the address-based load pattern used by L13/L15 memory heads
    (``Primitives.memory_lookup_attention`` is the imperative parent
    helper). The Q-side reads address nibbles at the query position;
    the K-side reads the same address nibbles stored at the memory-row
    position via ``addr_key_dims``. Where Q and K match, the attention
    weight saturates to 1 and the V projection broadcasts the stored
    byte(s) to ``output_dims``.

    Spec layout:
      * Q: slot ``i`` reads ``addr_query_dims[i]`` with weight
        ``query_weight`` (one slot per address nibble).
      * K: slot ``i`` reads ``addr_key_dims[i]`` with weight
        ``key_weight``. ``addr_key_dims`` carry the stored address one
        memory row at a time (e.g. ``ADDR_KEY+0...15``).
      * V: slot ``j`` reads ``value_dims[j]`` with weight 1.0 (one slot
        per output byte).
      * O: writes V slot ``j`` to ``output_dims[j]``.

    The Q/K slot count must match (``len(addr_query_dims) ==
    len(addr_key_dims)``); the V/O slot count must match. Both groups
    live in the same head's slot block — V/O slots start where Q/K
    slots end. ``head_dim`` must accommodate both: ``addr_width +
    value_width <= head_dim``.

    Args:
        head_idx: attention head index.
        addr_query_dims: residual dims holding the address nibbles at
            the query position (one dim per nibble).
        addr_key_dims: residual dims holding the stored address at the
            target memory row (same length as ``addr_query_dims``).
        value_dims: residual dims holding the stored bytes at the
            target memory row.
        output_dims: residual dims that receive the loaded bytes (same
            length as ``value_dims``).
        head_dim: per-head slot width. Must be >= addr_width + value_width.
        query_weight / key_weight: scale factors for the Q/K dot
            product. The product ``query_weight * key_weight`` controls
            attention sharpness — defaults match L15 ``L=15``.
        dim_positions: dim layout map (resolved at IR-build time).

    Returns:
        One ``DeclarativeAttentionHeadSpec``.

    Raises:
        ValueError: on Q/K width mismatch, V/O width mismatch, or slot
            overflow past ``head_dim``.
    """
    addr_q = tuple(addr_query_dims)
    addr_k = tuple(addr_key_dims)
    vals = tuple(value_dims)
    outs = tuple(output_dims)
    if len(addr_q) != len(addr_k):
        raise ValueError(
            "memory_load_attention: addr_query_dims/addr_key_dims length "
            f"mismatch ({len(addr_q)} vs {len(addr_k)})"
        )
    if len(vals) != len(outs):
        raise ValueError(
            "memory_load_attention: value_dims/output_dims length mismatch "
            f"({len(vals)} vs {len(outs)})"
        )
    if len(addr_q) + len(vals) > head_dim:
        raise ValueError(
            "memory_load_attention: addr_width + value_width > head_dim "
            f"({len(addr_q)} + {len(vals)} > {head_dim})"
        )
    for name in (*addr_q, *addr_k, *vals, *outs):
        if name not in dim_positions:
            raise ValueError(
                f"memory_load_attention: dim {name!r} missing from "
                f"dim_positions"
            )

    addr_w = len(addr_q)

    q_writes = tuple(
        AP(i, dim_positions[dim], query_weight)
        for i, dim in enumerate(addr_q)
    )
    k_writes = tuple(
        AP(i, dim_positions[dim], key_weight)
        for i, dim in enumerate(addr_k)
    )
    v_writes = tuple(
        AP(addr_w + j, dim_positions[dim], 1.0)
        for j, dim in enumerate(vals)
    )
    o_writes = tuple(
        AO(dim_positions[outs[j]], addr_w + j, 1.0)
        for j in range(len(outs))
    )
    return DeclarativeAttentionHeadSpec(
        head_idx=int(head_idx),
        q=q_writes,
        k=k_writes,
        v=v_writes,
        o=o_writes,
    )


# ===========================================================================
# IR_DSL §3.3 — Fetch-byte attention
# ===========================================================================


def fetch_byte_attention(
    *,
    head_idx: int,
    pc_dim_base: str,
    addr_key_dim_base: str,
    value_dim_base: str,
    output_dim_base: str,
    pc_offset: int = 0,
    pc_nibbles: int = 4,
    nibble_bits: int = 4,
    value_width: int = 8,
    head_dim: int = 64,
    query_weight: float = 15.0,
    key_weight: float = 15.0,
    dim_positions: Mapping[str, int],
) -> DeclarativeAttentionHeadSpec:
    """Fetch the byte at ``PC + pc_offset`` into ``output_dim_base...``
    (L4/L5 instruction-fetch pattern).

    Models the c4 fetch heads: PC is one-hot encoded across
    ``pc_nibbles`` banks of ``2 ** nibble_bits`` cells each (typically
    4 banks × 16 cells = 16-bit code address), and the ``addr_key`` at
    each memory row mirrors the same one-hot encoding for the stored
    code address. The Q-side reads the PC one-hot cells **after
    applying the constant ``pc_offset``** by shifting each bank's hot
    cell forward by the per-bank carry. The K-side reads the
    ``addr_key`` cells from each row's stored address. Where Q and K
    one-hot cells match across all banks, the attention score saturates
    and the row's value bytes broadcast to ``output_dim_base``.

    Spec layout (per nibble bank ``b``, per cell ``k``):
      * Q: slot ``b * 2**nibble_bits + k_shifted`` reads
        ``pc_dim_base + b * 2**nibble_bits + k`` with ``query_weight``,
        where ``k_shifted = (k + pc_offset_per_bank_b) mod 2**nibble_bits``.
        The offset shift is encoded by the slot-index permutation: a
        PC cell hot at ``k`` lights up the Q-slot for ``k_shifted``,
        which the K-side reads as the "match" cell.
      * K: slot ``b * 2**nibble_bits + k`` reads
        ``addr_key_dim_base + b * 2**nibble_bits + k`` with
        ``key_weight``.
      * V: slots ``pc_nibbles * 2**nibble_bits + j`` for j ∈
        ``[0, value_width)`` read ``value_dim_base + j`` with weight 1.
      * O: write each value slot to ``output_dim_base + j``.

    For ``pc_offset == 0`` the slot permutation is the identity (Q
    reads ``pc_dim_base + b * 2**nibble_bits + k`` at the matching K
    slot). For positive offsets that fit within a single bank (no
    carry across nibble banks), the shift is local to bank 0. Carries
    across banks aren't materialized here — the caller is expected to
    pre-resolve PC + offset into one-hot cells when the offset would
    span banks. The helper validates this by limiting ``pc_offset`` to
    ``[0, 2**nibble_bits - 1]`` so cell-level shifts are local.

    Args:
        head_idx: attention head index.
        pc_dim_base: base residual dim for the query PC one-hot bands.
            Cell ``b * 2**nibble_bits + k`` is hot when nibble ``b`` of
            PC equals ``k``.
        addr_key_dim_base: base residual dim for the stored-address
            one-hot bands at memory rows. Same layout as ``pc_dim_base``.
        value_dim_base: base residual dim for the per-row stored byte
            values (``value_width`` consecutive dims).
        output_dim_base: base residual dim that receives the fetched
            bytes (``value_width`` consecutive dims).
        pc_offset: scalar offset added to PC before the lookup. Must
            fit in a single nibble (``0 <= pc_offset < 2**nibble_bits``).
        pc_nibbles: number of one-hot nibble banks for the PC. Default
            4 (matches c4's 16-bit code addresses split into 4 nibbles).
        nibble_bits: bits per nibble bank (default 4, so each band has
            16 cells).
        value_width: number of value bytes broadcast (default 8 — a
            full byte's bits or two nibbles).
        head_dim: per-head slot width. Must accommodate
            ``pc_nibbles * 2**nibble_bits + value_width`` slots.
        query_weight / key_weight: sharpness scale.
        dim_positions: dim layout map.

    Returns:
        One ``DeclarativeAttentionHeadSpec``.

    Raises:
        ValueError: on slot overflow, missing dim_positions, or out-of-
            range ``pc_offset``.
    """
    cells_per_bank = 1 << nibble_bits
    if pc_offset < 0 or pc_offset >= cells_per_bank:
        raise ValueError(
            "fetch_byte_attention: pc_offset must be in "
            f"[0, {cells_per_bank}), got {pc_offset}"
        )
    addr_slot_count = pc_nibbles * cells_per_bank
    total_slots = addr_slot_count + value_width
    if total_slots > head_dim:
        raise ValueError(
            "fetch_byte_attention: pc_nibbles * 2**nibble_bits + "
            f"value_width ({addr_slot_count} + {value_width} = "
            f"{total_slots}) > head_dim ({head_dim})"
        )
    for name in (pc_dim_base, addr_key_dim_base, value_dim_base,
                 output_dim_base):
        if name not in dim_positions:
            raise ValueError(
                f"fetch_byte_attention: dim {name!r} missing from "
                f"dim_positions"
            )

    q_writes: list[AttentionProjectionWrite] = []
    k_writes: list[AttentionProjectionWrite] = []
    pc_base = dim_positions[pc_dim_base]
    addr_base = dim_positions[addr_key_dim_base]
    val_base = dim_positions[value_dim_base]
    out_base = dim_positions[output_dim_base]
    for b in range(pc_nibbles):
        # Only bank 0 carries the pc_offset shift; higher banks pass
        # through. (Carries across banks are out of scope — the caller
        # is expected to pre-resolve cross-bank offsets.)
        offset_for_bank = pc_offset if b == 0 else 0
        for k in range(cells_per_bank):
            slot = b * cells_per_bank + k
            # Q at slot reads the PC cell that would shift to k under
            # +pc_offset: i.e. PC's cell at (k - pc_offset) mod 16.
            pc_cell = (k - offset_for_bank) % cells_per_bank
            q_writes.append(AP(
                slot, pc_base + b * cells_per_bank + pc_cell, query_weight,
            ))
            k_writes.append(AP(
                slot, addr_base + b * cells_per_bank + k, key_weight,
            ))

    v_writes = tuple(
        AP(addr_slot_count + j, val_base + j, 1.0)
        for j in range(value_width)
    )
    o_writes = tuple(
        AO(out_base + j, addr_slot_count + j, 1.0)
        for j in range(value_width)
    )
    return DeclarativeAttentionHeadSpec(
        head_idx=int(head_idx),
        q=tuple(q_writes),
        k=tuple(k_writes),
        v=v_writes,
        o=o_writes,
    )


# ===========================================================================
# Per-nibble byte-band factories (reduction map ⑥ — cross-layer dedup)
# ===========================================================================
#
# Three families of 16-cell-per-band FFN loops are re-authored per layer
# across l6/l9/l10/l14 (and mirrored in l11/l13/l15/l16):
#
#   * BYTE-CLEAR    — an N-way AND on marker/opcode conditions writes a
#                     fixed (usually negative) value to every cell of one
#                     or more one-hot bands, scrubbing residue. Examples:
#                     l9 ``_alu_clear_rules`` (ALU_LO/HI clear),
#                     l9 ``_marker_suppress_rules`` (OUTPUT suppress under
#                     NEXT_*), l14 ``_layer14_clear_mem_marker_output``.
#   * BYTE-ROUTE    — an N-way AND, per-cell **gated** on ``SOURCE+k``,
#                     writes ``value`` to ``DEST+k``: it routes a stored
#                     one-hot value forward. Examples: l6
#                     ``_layer6_ax_output_route_rules`` (AX_CARRY->OUTPUT),
#                     l6 ``_layer6_imm_fetch_route_rules`` (FETCH->OUTPUT).
#   * CARRY-RELAY   — an N-way AND, per-cell gated by the difference
#                     ``(EMBED+k, -1) + (CARRY+k, +1)``, writes ``value``
#                     to ``DEST+k``: it emits only where the relayed carry
#                     value differs from the embed residual. Example: l6
#                     ``_layer6_stack_writeback_rules``.
#
# Each factory returns the SAME ``FFNRule`` tuple the hand-authored loop
# produced, so porting a call site is byte-identical. The per-band spec is
# a ``(band_label, ...)`` tuple so a call site can drive lo/hi (or an
# arbitrary set of bands) in one call while preserving rule order.
# ===========================================================================


def byte_clear_rules(
    *,
    bands: Sequence[str],
    conditions: Sequence[Tuple[str, float]],
    threshold: float,
    write_value: float,
    S: float = 100.0,
    lo: int = 0,
    hi: int = 15,
    gate: Optional[str] = None,
    gate_weight: float = 1.0,
    gate_bias: Optional[float] = None,
    gate_terms: Sequence[Tuple[str, float]] = (),
    name_prefix: Optional[str] = None,
    name_by_band: Optional[Mapping[str, str]] = None,
    scope: Optional[str] = None,
    write_scale_by_S: bool = True,
) -> Tuple[FFNRule, ...]:
    """Emit a per-cell BYTE-CLEAR band: one rule per cell writes
    ``write_value`` to ``{band}+k`` for every ``band`` and every ``k`` in
    ``[lo, hi]``, gated by the shared N-way AND ``conditions``.

    This is the discrete "scrub a one-hot band" primitive shared by the
    l9 ALU-clear / marker-suppress bands and the l14 OUTPUT-clear bands.
    Each cell's rule is a :func:`multi_way_and_rule` with the same
    ``conditions`` + ``threshold`` and a single write ``({band}+k,
    write_value / S)`` (or ``write_value`` verbatim when
    ``write_scale_by_S=False``).

    Rule ORDER is band-major then cell-major: for ``bands=("A", "B")`` the
    rules are ``A+lo .. A+hi, B+lo .. B+hi`` — matching the imperative
    cursor walk the legacy helpers used.

    Args:
        bands: ordered band base names to clear (e.g. ``("ALU_LO",
            "ALU_HI")`` or a single ``("OUTPUT_LO",)``).
        conditions: shared N-way AND ``(dim, weight)`` conditions.
        threshold: shared AND threshold.
        write_value: the value written per cell. Typically negative for a
            scrub. Lowered as ``write_value / S`` unless
            ``write_scale_by_S=False``.
        S: SwiGLU scale.
        lo / hi: inclusive cell-index bounds (default full 16-cell band).
        gate / gate_weight / gate_bias / gate_terms: optional shared gate
            threaded to every cell rule (e.g. ``gate="NEXT_PC"`` for the
            marker-suppress band). ``None`` gate + empty terms => a plain
            ``constant_write``.
        name_prefix: rule-name prefix; the per-cell name is
            ``f"{name_prefix}_{k}"``. Ignored for bands covered by
            ``name_by_band``.
        name_by_band: optional ``{band: prefix}`` overriding
            ``name_prefix`` per band (used when the legacy names differ by
            band, e.g. ``alu_lo_clear`` vs ``alu_hi_clear``).
        scope: optional shared predicate-DSL scope.
        write_scale_by_S: divide ``write_value`` by ``S`` (default). Set
            ``False`` when the legacy loop wrote the raw value (e.g. the
            marker-suppress ``-1.0`` cells).

    Returns:
        Tuple of ``len(bands) * (hi - lo + 1)`` ``FFNRule``s.
    """
    if lo > hi:
        raise ValueError(f"byte_clear_rules: lo ({lo}) > hi ({hi})")
    cond = tuple(conditions)
    rules: list[FFNRule] = []
    for band in bands:
        prefix = None
        if name_by_band is not None and band in name_by_band:
            prefix = name_by_band[band]
        elif name_prefix is not None:
            prefix = name_prefix
        for k in range(lo, hi + 1):
            value = _ww(write_value, S) if write_scale_by_S else write_value
            rules.append(multi_way_and_rule(
                name=f"{prefix}_{k}" if prefix else None,
                conditions=cond,
                threshold=threshold,
                gate=gate,
                gate_weight=gate_weight,
                gate_bias=gate_bias,
                gate_terms=tuple(gate_terms),
                writes=((f"{band}+{k}", value),),
                scope=scope,
            ))
    return tuple(rules)


def byte_route_rules(
    *,
    band_specs: Sequence[Tuple[str, str, str]],
    conditions: Sequence[Tuple[str, float]],
    threshold: float,
    write_value: float = _DEFAULT_WRITE,
    S: float = 100.0,
    lo: int = 0,
    hi: int = 15,
    name_prefix: Optional[str] = None,
    scope_by_band: Optional[Mapping[str, str]] = None,
    dominates_at_by_band: Optional[Mapping[str, Mapping[str, str]]] = None,
) -> Tuple[FFNRule, ...]:
    """Emit a per-cell BYTE-ROUTE band: for each ``(band_label,
    source_base, dest_base)`` spec, one rule per cell fires under the
    shared N-way AND ``conditions`` **gated on ``{source_base}+k``** and
    writes ``write_value`` to ``{dest_base}+k``.

    This routes a stored one-hot value forward one cell at a time — the
    shared shape behind l6's opcode AX_CARRY->OUTPUT routes and the IMM
    FETCH->OUTPUT route. The gate is the source cell, so the rule only
    contributes where that source one-hot is set.

    Rule ORDER is spec-major then cell-major, matching the legacy
    ``for band, source, output`` / ``for k`` nesting.

    Args:
        band_specs: ordered ``(band_label, source_base, dest_base)``
            triples. ``band_label`` (e.g. ``"lo"`` / ``"hi"``) is used to
            key per-band ``scope_by_band`` / ``dominates_at_by_band`` and
            to build the rule name.
        conditions: shared N-way AND ``(dim, weight)`` conditions.
        threshold: shared AND threshold.
        write_value: raw write numerator; lowered as ``write_value / S``.
            Default ``2.0`` matches the codebase's route scale.
        S: SwiGLU scale.
        lo / hi: inclusive cell-index bounds.
        name_prefix: per-cell name is ``f"{name_prefix}_{band_label}_{k}"``.
        scope_by_band: optional ``{band_label: scope}`` predicate scope
            applied to that band's cells (e.g. only the HI band annotated).
        dominates_at_by_band: optional ``{band_label: {dim: scope}}``
            dominates_at map applied to that band's cells.

    Returns:
        Tuple of ``len(band_specs) * (hi - lo + 1)`` ``FFNRule``s.
    """
    if lo > hi:
        raise ValueError(f"byte_route_rules: lo ({lo}) > hi ({hi})")
    cond = tuple(conditions)
    write_scale = _ww(write_value, S)
    rules: list[FFNRule] = []
    for band_label, source_base, dest_base in band_specs:
        scope = None
        if scope_by_band is not None:
            scope = scope_by_band.get(band_label)
        dominates_at = None
        if dominates_at_by_band is not None:
            dominates_at = dominates_at_by_band.get(band_label)
        for k in range(lo, hi + 1):
            name = (
                f"{name_prefix}_{band_label}_{k}" if name_prefix else None
            )
            rules.append(multi_way_and_rule(
                name=name,
                conditions=cond,
                threshold=threshold,
                gate=f"{source_base}+{k}",
                writes=((f"{dest_base}+{k}", write_scale),),
                scope=scope,
                dominates_at=dominates_at,
            ))
    return tuple(rules)


def byte_copy_computed_rules(
    *,
    src_lo: str,
    src_hi: str,
    dst_lo: str = "OUTPUT_LO",
    dst_hi: str = "OUTPUT_HI_THIS_STEP",
    dst_lo_offset: int = 0,
    dst_hi_offset: int = 0,
    base_conditions: Sequence[Tuple[str, float]],
    threshold: float,
    strength: float,
    name_for=None,
    gate: Optional[object] = None,
    scope: Optional[str] = None,
    dominates_at: Optional[Mapping[str, str]] = None,
    additive: bool = False,
) -> Tuple[FFNRule, ...]:
    """GENERAL cross-lane COMPUTED byte-copy: ``byte_copy(src_lane, dst_lane)``.

    The COMPUTED (32-unit) counterpart of an ENUMERATED (256-unit per-VALUE
    AND) cross-lane materializer that reads a byte one-hot from a SOURCE lane
    ``(src_lo, src_hi)`` and copies it to a DESTINATION lane ``(dst_lo,
    dst_hi)`` (default the L10 OUTPUT band).  Generalises the M8 same-lane
    per-nibble route (``l10_ops._computed_byte_writeback_route_rules``, which
    hardwired READ==WRITE==OUTPUT) to the SOURCE!=DEST case.

    The subtle point (flagged by agent #389): for a cross-lane copy the firing
    evidence must sum the **SOURCE** lane's other-band one-hot, NOT the
    destination's.  The enumerated per-value rule fires on ``src_lo+lo AND
    src_hi+hi`` (a 2-channel AND on the source byte).  Each route unit for dest
    channel ``k`` therefore fires on:

      * its own source channel ``{src_band}+k`` (weight 1.0), AND
      * the SUM of the OTHER source band's one-hot (the 16 ``{src_other}+j``
        terms, weight 1.0 each — exactly one is on for a valid byte).

    That reconstructs the SAME 2-channel evidence magnitude / ``threshold`` the
    enumerated AND used, so the route fires on exactly the same production
    contexts.  It then WRITES to the DEST band: ``+strength`` to ``{dst_band}+k``
    and ``-strength`` to that dest band's other 15 channels (a one-hot nibble
    write).  When the LO and HI route units for the observed SOURCE byte both
    fire, together they reconstruct that byte in the DEST band — a computed
    copy, not a 256-way lookup.

    Order: dest-LO band then dest-HI band, channel-ascending (matches the
    enumerated ``byte_value_writes`` LO-then-HI convention).

    Args:
        src_lo / src_hi: SOURCE lane low/high nibble one-hot bases (e.g.
            ``"ALU_LO"`` / ``"ALU_HI"`` for an ALU->OUTPUT materializer).
        dst_lo / dst_hi: DESTINATION lane nibble bases (default OUTPUT).
        dst_lo_offset / dst_hi_offset: constant base offset added to each
            dest channel index. Lets a band that PACKS both nibbles into one
            dim (e.g. ADDR_KEY: lo at ``+0..+15``, hi at ``+16..+31``) pass
            ``dst_lo=dst_hi="ADDR_KEY"``, ``dst_hi_offset=16`` — the write key
            stays a single-``+`` form (``ADDR_KEY+16``.. not ``ADDR_KEY+16+..``,
            which ``DimRef.parse`` would mis-split). Default 0 (separate dims).
        base_conditions: shared structural-evidence gate, emitted verbatim
            ahead of the per-channel source one-hot terms.
        threshold: per-rule AND threshold (matches the enumerated bank's).
        strength: one-hot nibble write ``+/-`` magnitude.
        name_for: ``(dst_band, k) -> name`` callable; a default is used if
            ``None``.
        gate / scope / dominates_at: threaded to ``multi_way_and_rule``.
        additive: when ``True``, each route unit writes ONLY ``+strength`` to
            its own dest channel ``k`` (no ``-strength`` competitor suppression
            of the other 15). This exactly reproduces an enumerated bank whose
            per-value unit does an ADDITIVE positive-only write into a CAM /
            one-hot key band (e.g. the L14 ADDR_KEY nibble decode, which feeds
            L15's per-nibble equality-match key: the enumerated form writes only
            ``+2/S`` per matched cell and never suppresses the losers). The
            default (``False``) keeps the L10 one-hot form (``+strength`` /
            ``-strength``) used for an OUTPUT-argmax destination.

    Returns:
        A ``2 * 16`` (== 32) ``FFNRule`` tuple.
    """
    if name_for is None:
        def name_for(dst_band, k):  # noqa: E306
            return f"byte_copy_{dst_band}_{k}"
    base = tuple(base_conditions)
    write_scale = strength
    rules: list[FFNRule] = []
    for src_band, src_other, dst_band, dst_off in (
        (src_lo, src_hi, dst_lo, dst_lo_offset),
        (src_hi, src_lo, dst_hi, dst_hi_offset),
    ):
        other_terms = tuple((f"{src_other}+{j}", 1.0) for j in range(16))
        for k in range(16):
            # ``dst_off`` lets a band that PACKS lo+hi into one dim address the
            # hi nibbles (e.g. ADDR_KEY hi at ``+16..+31``) without a double
            # ``+`` in the write key (``DimRef.parse`` rsplits on the LAST ``+``).
            if additive:
                # ADDITIVE positive-only write into the DEST band: +strength to
                # channel k, nothing to the losers (matches an enumerated CAM /
                # one-hot key materializer that only ever ADDS to matched cells).
                dst_writes = ((f"{dst_band}+{dst_off + k}", write_scale),)
            else:
                # One-hot nibble write into the DEST band: +strength to channel
                # k, -strength to the 15 competitors.
                dst_writes = tuple(
                    (
                        f"{dst_band}+{dst_off + j}",
                        write_scale if j == k else -write_scale,
                    )
                    for j in range(16)
                )
            rules.append(multi_way_and_rule(
                # Name by the ACTUAL in-band channel index (``dst_off + k``) so
                # a lo+hi-packed band (offset 0 lo / offset 16 hi) yields unique
                # names across the two routes; offset-0 callers see ``(band, k)``
                # unchanged.
                name=name_for(dst_band, dst_off + k),
                scope=scope,
                dominates_at=dominates_at,
                conditions=base + ((f"{src_band}+{k}", 1.0),) + other_terms,
                threshold=threshold,
                gate=gate,
                writes=dst_writes,
            ))
    return tuple(rules)


def carry_relay_rules(
    *,
    band_specs: Sequence[Tuple[str, str, str, str]],
    conditions: Sequence[Tuple[str, float]],
    threshold: float,
    write_value: float = _DEFAULT_WRITE,
    S: float = 100.0,
    lo: int = 0,
    hi: int = 15,
    name_prefix: Optional[str] = None,
) -> Tuple[FFNRule, ...]:
    """Emit a per-cell CARRY-RELAY band: for each ``(band_label,
    embed_base, carry_base, dest_base)`` spec, one rule per cell fires
    under the shared N-way AND ``conditions`` with per-cell **gate_terms**
    ``[({embed_base}+k, -1.0), ({carry_base}+k, +1.0)]`` and writes
    ``write_value`` to ``{dest_base}+k``.

    The gate is the signed difference "relayed carry value minus embed
    residual", so the write only fires where the two one-hots disagree —
    the shape of l6's stack-writeback band (emit OUTPUT = relayed
    AX_CARRY only where it differs from the natural EMBED value).

    Rule ORDER is spec-major then cell-major.

    Args:
        band_specs: ordered ``(band_label, embed_base, carry_base,
            dest_base)`` tuples (e.g. ``("lo", "EMBED_LO", "AX_CARRY_LO",
            "OUTPUT_LO")``).
        conditions: shared N-way AND ``(dim, weight)`` conditions.
        threshold: shared AND threshold.
        write_value: raw write numerator; lowered as ``write_value / S``.
        S: SwiGLU scale.
        lo / hi: inclusive cell-index bounds.
        name_prefix: per-cell name is ``f"{name_prefix}_{band_label}_{k}"``.

    Returns:
        Tuple of ``len(band_specs) * (hi - lo + 1)`` ``FFNRule``s.
    """
    if lo > hi:
        raise ValueError(f"carry_relay_rules: lo ({lo}) > hi ({hi})")
    cond = tuple(conditions)
    write_scale = _ww(write_value, S)
    rules: list[FFNRule] = []
    for band_label, embed_base, carry_base, dest_base in band_specs:
        for k in range(lo, hi + 1):
            name = (
                f"{name_prefix}_{band_label}_{k}" if name_prefix else None
            )
            rules.append(multi_way_and_rule(
                name=name,
                conditions=cond,
                threshold=threshold,
                gate_terms=(
                    (f"{embed_base}+{k}", -1.0),
                    (f"{carry_base}+{k}", 1.0),
                ),
                writes=((f"{dest_base}+{k}", write_scale),),
            ))
    return tuple(rules)


# ===========================================================================
# §566 — Mixture-of-Experts wrapper (opcode-gated rule batches)
# ===========================================================================


def opcode_expert_rules(
    opcode_gate: str,
    rules: Sequence[FFNRule],
    *,
    gate_weight: float = 1.0,
) -> Tuple[FFNRule, ...]:
    """Thread an opcode-gate dim through every rule in ``rules``.

    Blog §566 — opcodes are MoE experts: each opcode's rule batch
    should only fire when the matching ``OP_<X>`` flag is hot. The
    legacy pattern is to pass ``gate="OP_X"`` to every ``wide_*_rules``
    call site (alu_ops.py has 6+ such sites). This wrapper lets *any*
    rule batch (typed FFNRule tuple) get opcode-gated centrally:

        gated = opcode_expert_rules("OP_ADD", wide_add_rules(...))

    For each input rule:
      * If ``rule.gate is None and rule.gate_terms == ()``, replace it
        with ``gate=DimRef(opcode_gate, 0)`` (gate_weight defaulting to
        1.0) and flip ``gate_bias`` from 1.0 to 0.0 (matching the
        ``constant_write`` → ``gated_write`` convention used elsewhere
        in this module).
      * Otherwise (the rule already has its own gate or gate_terms),
        fold the opcode condition into ``gate_terms`` so the existing
        gating semantics are preserved and the opcode becomes a
        balanced additive condition on the gate side.

    Args:
        opcode_gate: name of the opcode-flag dim (e.g. ``"OP_ADD"``).
            The gate fires when this dim is hot.
        rules: tuple/sequence of ``FFNRule`` objects to wrap.
        gate_weight: multiplier applied to the opcode gate term.
            Defaults to 1.0, matching the typical ``gate_bias=0``
            convention used by ``gated_write``.

    Returns:
        Tuple of ``FFNRule`` objects of the same length as ``rules``,
        each with the opcode gate threaded through.
    """
    gate_ref = DimRef.parse(opcode_gate)
    wrapped: list[FFNRule] = []
    for rule in rules:
        if rule.gate is None and not rule.gate_terms:
            # Convert constant_write-style to opcode-gated. The
            # gate_bias flips from 1.0 (constant_write default) to 0.0
            # (gated_write default) so the gate value is precisely the
            # opcode flag (no baseline offset).
            wrapped.append(dataclasses.replace(
                rule,
                gate=gate_ref,
                gate_weight=float(gate_weight),
                gate_bias=0.0,
            ))
        else:
            # Already has a gate or gate_terms — fold the opcode in as
            # an additive gate term so the existing structure is
            # preserved. The opcode flag (binary 0/1) adds 0 or
            # ``gate_weight`` to the gate value; downstream consumers
            # see the combined gate magnitude.
            new_gate_terms = rule.gate_terms + (
                ConditionTerm(gate_ref, float(gate_weight)),
            )
            wrapped.append(dataclasses.replace(
                rule,
                gate_terms=new_gate_terms,
            ))
    return tuple(wrapped)


# ===========================================================================
# V2.1 — Binary-encoded address lookup attention (L15 memory_lookup pattern)
# ===========================================================================


def binary_address_lookup_attention(
    *,
    head_idx: int,
    addr_dim_bases: Sequence[str],
    addr_width_bits: int = 4,
    addr_slot_base: int = 4,
    bit_scale: float = 10.0,
    bias_slot: int = 0,
    bias_dim: Optional[str] = "CONST",
    bias_weight: float = 0.0,
    key_bias_dim: Optional[str] = None,
    key_bias_weight: float = 0.0,
    discriminators: Optional[Sequence[Tuple[str, float]]] = None,
    suppressors: Optional[Sequence[Tuple[str, float]]] = None,
    value_slot_base: Optional[int] = None,
    value_dims: Sequence[str] = (),
    output_dims: Sequence[str] = (),
    head_dim: int = 64,
    dim_positions: Mapping[str, int],
) -> DeclarativeAttentionHeadSpec:
    """Binary-encoded address lookup attention head (L15 pattern).

    Unlike :func:`memory_load_attention` (one-hot per nibble), this
    primitive encodes the query/key address with ``±bit_scale`` per bit
    across one-hot nibble bands. For each nibble band ``base + k`` with
    ``k ∈ [0, 2**addr_width_bits)``, the W_q / W_k row at slot
    ``addr_slot_base + bit`` writes ``+bit_scale`` when bit ``bit`` of
    ``k`` is 1 and ``-bit_scale`` when it is 0. This matches the legacy
    L15 ``_set_layer15_memory_lookup_heads_0_3`` body's "dims 4-27 are
    binary 24-bit address encoding" comment block.

    Per nibble band the helper consumes ``addr_width_bits`` slots, so
    the address occupies ``len(addr_dim_bases) * addr_width_bits`` slots
    starting at ``addr_slot_base``.

    Optional Q-side **discriminators** are added to ``bias_slot`` (slot
    0 by default): each ``(dim_name, weight)`` pair contributes
    ``+weight`` to ``W_q[base + bias_slot, dim_positions[dim_name]]``.
    Use these for opcode gates (e.g. ``("OP_LI_RELAY", 2000.0)``) and
    extra match conditions (e.g. ``("CMP+3", 2000.0)`` for the
    pop-group dual-role head 0).

    Optional Q-side **suppressors** are also added to ``bias_slot``:
    each ``(dim_name, weight)`` pair contributes ``+weight`` (typically
    negative) so non-target queries (e.g. at MARK_PC/MARK_SP positions
    during LEV) produce overwhelmingly negative scores. The L15 body
    uses MARK_PC=-25000, MARK_SP=-100000, OP_LEV=-1000, H1[SP]=-50000,
    H1[BP]=-50000.

    Both ``discriminators`` and ``suppressors`` accept band-offset names
    of the form ``"BAND+N"`` (e.g. ``"CMP+3"``, ``"H1+3"``); the helper
    resolves ``dim_positions["BAND"] + N`` exactly the way ``DimRef.parse``
    does for FFN rules.

    Optional V/O block: if ``value_dims`` / ``output_dims`` are supplied,
    the helper also writes V slots (one per ``value_dims`` entry) and
    matching O writes. ``value_slot_base`` defaults to ``addr_slot_base +
    len(addr_dim_bases) * addr_width_bits`` (right after the address
    block). The legacy L15 V/O block at slots 32..63 writes
    ``W_v[base + 32 + k, CLEAN_EMBED_LO + k] = 1`` and
    ``W_o[OUTPUT_LO + k, base + 32 + k] = 1`` for k in [0, 16) and the
    same for HI; pass ``value_slot_base=32`` to match it.

    Args:
        head_idx: attention head index.
        addr_dim_bases: ordered list of nibble-band base dim names.
            Example for L15's 24-bit address:
            ``("ADDR_B0_LO", "ADDR_B0_HI", "ADDR_B1_LO", "ADDR_B1_HI",
              "ADDR_B2_LO", "ADDR_B2_HI")``.
        addr_width_bits: bits per nibble band (default 4 — each band is
            16 one-hot cells covering a 4-bit value).
        addr_slot_base: head-local slot index where the address block
            starts. Default 4 matches the legacy L15 layout
            (slots 0..3 are reserved for bias / store anchor / ZFOD /
            byte selection).
        bit_scale: ``±bit_scale`` per Q/K bit. Default 10.0 (legacy L15
            ``scale = 10.0`` in ``_set_layer15_memory_lookup_heads_0_3``).
        bias_slot: head-local slot for the non-target bias + Q-side
            discriminators + suppressors. Default 0 (legacy L15 Dim 0).
        bias_dim: residual dim driving the per-position non-target bias.
            Default ``"CONST"`` — the legacy ``-2000`` default bias.
            Set to ``None`` to skip the bias write.
        bias_weight: weight for the bias-dim Q-write (e.g. ``-2000.0``
            for the legacy L15 non-target suppression).
        key_bias_dim: residual dim that the K-side reads at ``bias_slot``.
            Default ``None`` (no K-side bias). The legacy L15 writes
            ``W_k[base + 0, CONST] = 10.0`` — pass ``"CONST"`` with
            ``key_bias_weight=10.0`` to reproduce that.
        key_bias_weight: K-side weight at ``bias_slot``.
        discriminators: optional Q-side ``(dim_name, weight)`` writes at
            ``bias_slot``. Used for opcode gates (e.g.
            ``("OP_LI_RELAY", 2000.0)``).
        suppressors: optional Q-side ``(dim_name, weight)`` writes at
            ``bias_slot`` (typically negative). Names may include
            band-offset form ``"BAND+N"``.
        value_slot_base: starting slot for V/O writes. Default
            ``addr_slot_base + len(addr_dim_bases) * addr_width_bits``.
        value_dims: residual dims read by V projection (one slot per dim).
        output_dims: residual dims written by O projection
            (same length as ``value_dims``).
        head_dim: per-head slot width. Validates that all writes
            (address + V/O) fit.
        dim_positions: dim layout map.

    Returns:
        One ``DeclarativeAttentionHeadSpec``.

    Raises:
        ValueError: on dim-name typos, slot overflow, or V/O length
            mismatch.
    """
    addr_bases = tuple(addr_dim_bases)
    discs = tuple(discriminators or ())
    supps = tuple(suppressors or ())
    vals = tuple(value_dims)
    outs = tuple(output_dims)

    if len(vals) != len(outs):
        raise ValueError(
            "binary_address_lookup_attention: value_dims/output_dims "
            f"length mismatch ({len(vals)} vs {len(outs)})"
        )
    if addr_width_bits <= 0:
        raise ValueError(
            "binary_address_lookup_attention: addr_width_bits must be "
            f"positive, got {addr_width_bits}"
        )

    cells_per_band = 1 << addr_width_bits
    addr_slots = len(addr_bases) * addr_width_bits
    if value_slot_base is None:
        value_slot_base = addr_slot_base + addr_slots

    total_slot_extent = max(
        bias_slot + 1,
        addr_slot_base + addr_slots,
        value_slot_base + len(vals),
    )
    if total_slot_extent > head_dim:
        raise ValueError(
            "binary_address_lookup_attention: slot extent "
            f"{total_slot_extent} > head_dim ({head_dim}) "
            f"(addr_slot_base={addr_slot_base}, addr_slots={addr_slots}, "
            f"value_slot_base={value_slot_base}, value_count={len(vals)})"
        )

    # Resolve dim references. Supports "BAND+N" syntax (e.g. "CMP+3")
    # the same way the FFN-rule DimRef.parse does.
    def _resolve(name: str) -> int:
        if "+" in name:
            base, off = name.rsplit("+", 1)
            base = base.strip()
            off = int(off.strip())
            if base not in dim_positions:
                raise ValueError(
                    f"binary_address_lookup_attention: dim {base!r} "
                    f"(from {name!r}) missing from dim_positions"
                )
            return dim_positions[base] + off
        if name not in dim_positions:
            raise ValueError(
                f"binary_address_lookup_attention: dim {name!r} missing "
                f"from dim_positions"
            )
        return dim_positions[name]

    for nibble_base in addr_bases:
        if nibble_base not in dim_positions:
            raise ValueError(
                f"binary_address_lookup_attention: addr nibble base "
                f"{nibble_base!r} missing from dim_positions"
            )
    for name in (*vals, *outs):
        if name not in dim_positions:
            raise ValueError(
                f"binary_address_lookup_attention: dim {name!r} missing "
                f"from dim_positions"
            )

    q_writes: list[AttentionProjectionWrite] = []
    k_writes: list[AttentionProjectionWrite] = []

    # --- bias / discriminator / suppressor row at bias_slot ---
    if bias_dim is not None and bias_weight != 0.0:
        q_writes.append(AP(bias_slot, _resolve(bias_dim), float(bias_weight)))
    if key_bias_dim is not None and key_bias_weight != 0.0:
        k_writes.append(
            AP(bias_slot, _resolve(key_bias_dim), float(key_bias_weight))
        )
    for dim_name, weight in discs:
        q_writes.append(AP(bias_slot, _resolve(dim_name), float(weight)))
    for dim_name, weight in supps:
        q_writes.append(AP(bias_slot, _resolve(dim_name), float(weight)))

    # --- binary-encoded address block at addr_slot_base.. ---
    slot = addr_slot_base
    for nibble_base in addr_bases:
        base_pos = dim_positions[nibble_base]
        for bit in range(addr_width_bits):
            for k in range(cells_per_band):
                bit_val = 2 * ((k >> bit) & 1) - 1
                q_writes.append(AP(slot, base_pos + k, bit_scale * bit_val))
                k_writes.append(AP(slot, base_pos + k, bit_scale * bit_val))
            slot += 1

    # --- V/O block ---
    v_writes: list[AttentionProjectionWrite] = []
    o_writes: list[AttentionOutputWrite] = []
    for j, (val_dim, out_dim) in enumerate(zip(vals, outs)):
        v_slot = value_slot_base + j
        v_writes.append(AP(v_slot, dim_positions[val_dim], 1.0))
        o_writes.append(AO(dim_positions[out_dim], v_slot, 1.0))

    return DeclarativeAttentionHeadSpec(
        head_idx=int(head_idx),
        q=tuple(q_writes),
        k=tuple(k_writes),
        v=tuple(v_writes),
        o=tuple(o_writes),
    )


# ===========================================================================
# V2.1 — Attention head extension (compose multiple spec fragments)
# ===========================================================================


def attention_head_extension(
    base_spec: DeclarativeAttentionHeadSpec,
    *,
    extra_q_writes: Sequence[AttentionProjectionWrite] = (),
    extra_k_writes: Sequence[AttentionProjectionWrite] = (),
    extra_v_writes: Sequence[AttentionProjectionWrite] = (),
    extra_o_writes: Sequence[AttentionOutputWrite] = (),
    alibi_slope: Optional[float] = None,
) -> DeclarativeAttentionHeadSpec:
    """Augment an existing :class:`DeclarativeAttentionHeadSpec` with
    additional Q/K/V/O writes.

    This is the V2.1 composition helper. The L15 lookup heads layer
    bespoke discriminator rows, per-head position gates, marker
    suppressions, and value-lane writes on top of a shared "binary
    address match" base. Rather than encoding the full surface in
    :func:`binary_address_lookup_attention`, the V2.1 design exposes a
    small primitive plus a generic extension wrapper:

        base = binary_address_lookup_attention(...)
        spec = attention_head_extension(
            base,
            extra_q_writes=(AP(28, MARK_AX, 500.0), ...),
            extra_o_writes=(AO(OUTPUT_LO+0, 32, 1.0), ...),
        )

    Writes are *appended*, not merged. ``Primitives.generate_attention_head``
    applies each write in order, so a later write to the same
    ``(slot, dim)`` cell overwrites the earlier one — useful for
    reconstructing the legacy L15 head 9 wipe-then-write pattern.

    ``alibi_slope`` is propagated when supplied; otherwise the base
    spec's slope is preserved.

    Args:
        base_spec: the base spec to extend.
        extra_q_writes / extra_k_writes / extra_v_writes / extra_o_writes:
            additional ``AP`` / ``AO`` writes to append.
        alibi_slope: optional override for the head's ALiBi slope.

    Returns:
        A new :class:`DeclarativeAttentionHeadSpec` with the extra
        writes appended. The ``head_idx``, ``head_dim``, and
        ``group_size`` are inherited from ``base_spec``.
    """
    new_q = tuple(base_spec.q) + tuple(extra_q_writes)
    new_k = tuple(base_spec.k) + tuple(extra_k_writes)
    new_v = tuple(base_spec.v) + tuple(extra_v_writes)
    new_o = tuple(base_spec.o) + tuple(extra_o_writes)
    new_slope = alibi_slope if alibi_slope is not None else base_spec.alibi_slope
    return dataclasses.replace(
        base_spec,
        q=new_q,
        k=new_k,
        v=new_v,
        o=new_o,
        alibi_slope=new_slope,
    )


__all__ = [
    "step_function_rule",
    "one_hot_indicator_rule",
    "band_range_check_rules",
    "multi_way_and_rule",
    "multi_way_or_rules",
    "cancel_residual_rule",
    "byte_clear_rules",
    "byte_route_rules",
    "carry_relay_rules",
    "lookup_table_rules",
    "efficient_exp_attention",
    "memory_load_attention",
    "fetch_byte_attention",
    "binary_address_lookup_attention",
    "attention_head_extension",
    "opcode_expert_rules",
]
