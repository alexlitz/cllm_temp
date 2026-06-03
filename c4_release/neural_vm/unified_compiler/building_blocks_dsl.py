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
| §547 Efficient floor (MAGIC) | ``magic_floor_rules``        |
| §558 Bit range extraction    | ``bit_range_extract_rules``  |
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


# ===========================================================================
# §547 — Efficient floor via fp32 MAGIC trick
# ===========================================================================
#
# The MAGIC trick (blog §547-555): at a sufficiently large scale, fp32's
# unit in last place (ULP) becomes 1.0, so the format can only represent
# integers. ``MAGIC32 = 1.5 * 2**23 = 12_582_912.0`` lies at that
# boundary; adding it to any non-negative value < 2^23 forces rounding
# to the nearest integer, and subtracting it back recovers ``round(x)``.
# Subtract ``(0.5 - eps)`` before adding MAGIC to turn round-to-nearest
# into ``floor``.
#
# DSL encoding (no IR extension): two ``constant_write`` ``FFNRule``s
# evaluated at ``S=1.0``. The rounding happens inside the ``F.linear``
# accumulation of ``W_up @ x + b_up``:
#
#   Unit 1 (the floor unit):
#     conditions = [(input_dim, +1.0), (const_dim, -0.5 + eps)]
#     threshold  = -MAGIC32       → b_up = +MAGIC32
#     gate       = constant 1.0   (constant_write's gate_bias = 1.0)
#     writes     = [(output_dim, +1.0)]
#
#     SwiGLU forward:
#       pre = (1.0 * input + (-0.5+eps) * 1.0) + MAGIC32
#           = MAGIC32 + (input - 0.5 + eps)
#       silu(pre) ≈ pre  (sigmoid(MAGIC32) = 1.0 in fp32)
#       hidden = silu(pre) * 1.0  → ≈ MAGIC32 + floor(input)   (fp32 round)
#       W_down @ hidden  → +(MAGIC32 + floor(input))
#
#   Unit 2 (the cancel unit):
#     conditions = []
#     threshold  = -MAGIC32       → b_up = +MAGIC32
#     gate       = constant 1.0
#     writes     = [(output_dim, -1.0)]
#
#     hidden = silu(MAGIC32) * 1.0 ≈ MAGIC32
#     W_down @ hidden → -MAGIC32
#
#   Net per-residual write: ``floor(input)`` (the two MAGIC terms cancel).
#
# Why S=1.0 is mandatory: ``lower_ffn`` sets ``W_up[unit, dim] = S *
# weight``. At S>1, ``MAGIC32`` would scale beyond 2^24 where fp32 ULP
# > 1 and the trick breaks (the rounding granularity becomes coarser
# than one integer step). The helper hard-pins ``S=1.0``; callers that
# need a larger band step must extract per-nibble and combine via
# ``bit_range_extract_rules``.

_MAGIC32 = 1.5 * float(2 ** 23)  # 12_582_912.0 — fp32 ULP=1 at this scale.


def magic_floor_rules(
    *,
    input_dim: str,
    output_dim: str,
    const_dim: str = "CONST",
    eps: float = 0.001,
    gate: Optional[str] = None,
    gate_terms: Sequence[Tuple[str, float]] = (),
    name: Optional[str] = None,
    scope: Optional[str] = None,
) -> Tuple[FFNRule, ...]:
    """Compute ``floor(input_dim)`` into ``output_dim`` via the fp32 MAGIC
    trick (blog §547-555).

    Emits exactly two ``FFNRule``s evaluated at scale ``S=1.0``: the
    first reads ``input_dim`` plus the constant offset ``-0.5 + eps``
    and adds ``MAGIC32`` via its b_up; the second writes ``-MAGIC32``
    back to ``output_dim`` so the two cancel and the floor value
    survives. The rounding happens inside ``F.linear``'s accumulation of
    ``W_up @ x + b_up`` in fp32.

    **Scale is hard-pinned to ``S=1.0`` at lowering time.** Pass this
    function's rules through ``Primitives.lower_ffn_rules(..., S=1.0)``;
    larger ``S`` blows the ``MAGIC32`` magnitude past 2^24 where fp32
    ULP > 1 and the rounding granularity stops matching integer steps.

    Args:
        input_dim: residual dim holding the non-negative scalar to floor.
            Must lie in ``[0, 2**23)`` for fp32 ULP=1 to apply.
        output_dim: residual dim that receives ``floor(input_dim)``.
        const_dim: residual dim that's permanently held at ``1.0`` (the
            usual ``"CONST"`` injection dim). Used to carry the
            ``-0.5 + eps`` shift into the silu input without an
            explicit bias field.
        eps: small offset that converts round-to-nearest into floor at
            half-integers and integers. Defaults to ``0.001``, matching
            ``alu/ops/common.py:magic_floor32``.
        gate / gate_terms: optional shared multiplicative gate applied
            to both units (e.g. an opcode flag). When ``None`` the rules
            use ``constant_write`` (gate_bias=1.0); otherwise they use
            ``gated_write`` (gate_bias=0.0).
        name: optional prefix for the rule names (``"<name>_floor"`` and
            ``"<name>_cancel"``).
        scope: optional shared predicate-DSL scope.

    Returns:
        Tuple of two ``FFNRule``s.
    """
    floor_name = f"{name}_floor" if name else None
    cancel_name = f"{name}_cancel" if name else None
    floor_unit = _build_rule(
        conditions=(
            (input_dim, 1.0),
            (const_dim, -0.5 + eps),
        ),
        threshold=-_MAGIC32,
        writes=((output_dim, +1.0),),
        gate=gate,
        gate_terms=gate_terms,
        name=floor_name,
        scope=scope,
    )
    cancel_unit = _build_rule(
        conditions=(),
        threshold=-_MAGIC32,
        writes=((output_dim, -1.0),),
        gate=gate,
        gate_terms=gate_terms,
        name=cancel_name,
        scope=scope,
    )
    return (floor_unit, cancel_unit)


# ===========================================================================
# §558 — Bit range extraction
# ===========================================================================


def bit_range_extract_rules(
    *,
    input_dim: str,
    lo_shift_dim: str,
    hi_shift_dim: str,
    lo_bit: int,
    hi_bit: int,
    const_dim: str = "CONST",
    eps: float = 0.001,
    gate: Optional[str] = None,
    gate_terms: Sequence[Tuple[str, float]] = (),
    name: Optional[str] = None,
    scope: Optional[str] = None,
) -> Tuple[FFNRule, ...]:
    """Two-pair bit-shift step toward extracting bits ``[lo_bit, hi_bit)``
    from ``input_dim`` (blog §558 — partial primitive).

    The full identity is

        bits[lo:hi)(x) = floor(x / 2**lo) mod 2**(hi-lo)
                       = floor(x / 2**lo) - 2**w * floor(x / 2**hi)

    where ``w = hi - lo``. The MOD combine step (the subtraction) cannot
    safely happen in the same FFN pass as the two floors: ``F.linear``
    sums all hidden-unit contributions into one residual dim in fp32,
    and the four MAGIC-scale partial products cancel down to a small
    integer through intermediate values around ``2 * 2**24`` where
    fp32 ULP > 1. The lossy add destroys the low bits we're trying to
    recover.

    The fix is to keep the two floors on **separate output dims** in
    this pass — each output dim has its own ``W_down`` sum covering
    exactly the matching MAGIC-pair, which cancels cleanly. A second
    FFN pass (or a residual-stream linear write) then combines

        out = lo_shift_dim - 2**w * hi_shift_dim

    The blog's "do it per nibble … and combine" pattern matches this
    split exactly.

    Implementation: four ``FFNRule``s in one layer. The low-shift floor
    pair writes ``floor(x / 2**lo)`` to ``lo_shift_dim``. The
    high-shift floor pair writes ``floor(x / 2**hi)`` to
    ``hi_shift_dim``. Each pair is independently a ``magic_floor_rules``
    instance over the same input but with a different ``W_up`` scale.

    Args:
        input_dim: residual dim holding the non-negative integer-valued
            scalar to extract bits from. Must lie in ``[0, 2**23)`` for
            fp32 ULP=1 to apply.
        lo_shift_dim: residual dim that receives ``floor(x / 2**lo)``.
        hi_shift_dim: residual dim that receives ``floor(x / 2**hi)``.
        lo_bit: inclusive low bit index (0-based, LSB).
        hi_bit: exclusive high bit index. ``lo_bit < hi_bit`` and
            ``hi_bit <= 23``.
        const_dim: residual dim permanently held at ``1.0`` (carries
            the ``-0.5+eps`` offset for each floor pair).
        eps: floor offset (see ``magic_floor_rules``).
        gate / gate_terms: optional shared multiplicative gate applied
            to all four emitted units.
        name: optional prefix; emitted rules end in ``_lo_floor``,
            ``_lo_cancel``, ``_hi_floor``, ``_hi_cancel``.
        scope: optional shared predicate-DSL scope.

    Returns:
        Tuple of four ``FFNRule``s. The caller must run a subsequent
        FFN pass (or residual write) to compute the mod combine:
        ``lo_shift_dim - 2**(hi_bit - lo_bit) * hi_shift_dim``.

    Raises:
        ValueError: if ``lo_bit < 0``, ``hi_bit <= lo_bit``, or
            ``hi_bit > 23``.
    """
    if lo_bit < 0:
        raise ValueError(
            f"bit_range_extract_rules: lo_bit ({lo_bit}) must be >= 0"
        )
    if hi_bit <= lo_bit:
        raise ValueError(
            f"bit_range_extract_rules: hi_bit ({hi_bit}) must be > "
            f"lo_bit ({lo_bit})"
        )
    if hi_bit > 23:
        raise ValueError(
            f"bit_range_extract_rules: hi_bit ({hi_bit}) must be <= 23 "
            f"(fp32 ULP=1 bound)"
        )

    pow_lo = 1.0 / float(1 << lo_bit)  # 2**(-lo_bit)
    pow_hi = 1.0 / float(1 << hi_bit)  # 2**(-hi_bit)

    lo_floor_name = f"{name}_lo_floor" if name else None
    lo_cancel_name = f"{name}_lo_cancel" if name else None
    hi_floor_name = f"{name}_hi_floor" if name else None
    hi_cancel_name = f"{name}_hi_cancel" if name else None

    # floor(x * 2**(-lo)) → +1 contribution to lo_shift_dim.
    lo_floor = _build_rule(
        conditions=(
            (input_dim, pow_lo),
            (const_dim, -0.5 + eps),
        ),
        threshold=-_MAGIC32,
        writes=((lo_shift_dim, +1.0),),
        gate=gate,
        gate_terms=gate_terms,
        name=lo_floor_name,
        scope=scope,
    )
    lo_cancel = _build_rule(
        conditions=(),
        threshold=-_MAGIC32,
        writes=((lo_shift_dim, -1.0),),
        gate=gate,
        gate_terms=gate_terms,
        name=lo_cancel_name,
        scope=scope,
    )
    # floor(x * 2**(-hi)) → +1 contribution to hi_shift_dim.
    hi_floor = _build_rule(
        conditions=(
            (input_dim, pow_hi),
            (const_dim, -0.5 + eps),
        ),
        threshold=-_MAGIC32,
        writes=((hi_shift_dim, +1.0),),
        gate=gate,
        gate_terms=gate_terms,
        name=hi_floor_name,
        scope=scope,
    )
    hi_cancel = _build_rule(
        conditions=(),
        threshold=-_MAGIC32,
        writes=((hi_shift_dim, -1.0),),
        gate=gate,
        gate_terms=gate_terms,
        name=hi_cancel_name,
        scope=scope,
    )
    return (lo_floor, lo_cancel, hi_floor, hi_cancel)


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


__all__ = [
    "step_function_rule",
    "one_hot_indicator_rule",
    "band_range_check_rules",
    "multi_way_and_rule",
    "multi_way_or_rules",
    "cancel_residual_rule",
    "lookup_table_rules",
    "magic_floor_rules",
    "bit_range_extract_rules",
    "efficient_exp_attention",
    "memory_load_attention",
    "fetch_byte_attention",
    "opcode_expert_rules",
]
