"""Wide-ALU DSL — FFNRule-generating helpers (IR DSL Section 3.2).

Per ``docs/IR_DSL_DESIGN.md`` Sections 3.2-4, the ~3,781 lines of
imperative composites in ``efficient_alu_*.py`` are to be re-expressed as
``FFNRule`` lists. This module is the home for those helpers.

The helpers are **plain Python functions** that take typed inputs (dim
name strings, opcode/marker gate refs, scale ``S``) and return ``tuple[
FFNRule, ...]`` lists. They lower through the existing
``Primitives.lower_ffn_rules`` pipeline; no new IR type is introduced.

Status (Task 81, scaffolding):
  - ``bitwise_rules`` — IMPLEMENTED (Wave W1 POC).
  - ``wide_add_rules`` / ``wide_sub_rules`` — stubs (Wave W3).
  - ``wide_shift_rules`` — stub (Wave W2).
  - ``wide_mul_rules`` — stub (Wave W5, 9-stage pipeline).
  - ``wide_div_rules`` — stub (Wave W4, long-division).

The validation contract per helper is:
  ``rule_lowered_ffn.forward(x)  ==  hand_composite.forward(x)``
bit-for-bit on randomized input (see Section 4 of the design doc and
``tests/test_wide_alu_dsl.py`` for the ``bitwise_rules`` POC).
"""

from __future__ import annotations

import operator
from typing import Callable, Literal, Tuple

from .ir import FFNRule


# ---------------------------------------------------------------------------
# Wave W1: Bitwise (AND/OR/XOR) — POC implementation.
# ---------------------------------------------------------------------------


_BITWISE_OP_FN: dict[str, Callable[[int, int], int]] = {
    "and": operator.and_,
    "or": operator.or_,
    "xor": operator.xor,
}


def bitwise_rules(
    *,
    op: Literal["and", "or", "xor"],
    operand_a_lo: str,
    operand_a_hi: str,
    operand_b_lo: str,
    operand_b_hi: str,
    result_lo: str,
    result_hi: str,
    opcode_gate: str,
    marker_gate: str,
    S: float,
) -> Tuple[FFNRule, ...]:
    """Generate FFNRule list for a per-byte bitwise op (AND/OR/XOR).

    Emits 512 rules per op (256 low-nibble + 256 high-nibble cross-product),
    matching the structure of ``_layer10_alu_bitwise_rules`` in
    ``ops/l10_ops.py`` — the existing rule-driven L10 bitwise bake.

    Each unit is a 3-way AND across (``marker_gate``, ``operand_*[a]``,
    ``operand_*[b]``) gated on ``opcode_gate``. Weights (40, 30, 30) and
    threshold 80 implement the balanced 3-way AND used by the legacy bake:

      * all three present: 40 + 30 + 30 = 100 > 80 → fires
      * any two present:   max(40 + 30) = 70 < 80 → blocked

    The output write is ``2.0 / S`` to ``result_*[op_fn(a, b)]``, matching
    the legacy ``OUTPUT_LO/HI_THIS_STEP`` writes (lookup-mode).

    Args:
        op: ``"and"``, ``"or"``, or ``"xor"``.
        operand_a_lo: residual dim name for operand A's low nibble band
            (e.g. ``"ALU_LO"``). Reads ``operand_a_lo+k`` for k in 0..15.
        operand_a_hi: dim name for operand A's high nibble band.
        operand_b_lo: dim name for operand B's low nibble band.
        operand_b_hi: dim name for operand B's high nibble band.
        result_lo: dim name for the result low nibble band
            (e.g. ``"OUTPUT_LO"``).
        result_hi: dim name for the result high nibble band.
        opcode_gate: dim ref for the opcode flag (e.g. ``"OP_AND"`` or
            the ``dim_ref("opcode_flag", "AND")`` semantic form).
        marker_gate: dim name for the AX-style marker (e.g. ``"MARK_AX"``).
        S: SwiGLU scale (typically 100.0).

    Returns:
        ``tuple[FFNRule, ...]`` of length 512 — same shape as
        ``_layer10_alu_bitwise_rules(S, op_name=op.upper(), op_fn=op_fn)``.

    Raises:
        ValueError: if ``op`` is not one of ``"and"``, ``"or"``, ``"xor"``.
    """
    op_fn = _BITWISE_OP_FN.get(op)
    if op_fn is None:
        raise ValueError(
            f"bitwise_rules: op must be 'and'/'or'/'xor'; got {op!r}"
        )

    rules: list[FFNRule] = []
    nibble_iter = (
        # (label, operand_a_band, operand_b_band, output_band)
        ("lo", operand_a_lo, operand_b_lo, result_lo),
        ("hi", operand_a_hi, operand_b_hi, result_hi),
    )
    for nibble_label, a_band, b_band, out_band in nibble_iter:
        for a in range(16):
            for b in range(16):
                result = op_fn(a, b)
                rules.append(FFNRule.gated_write(
                    name=(
                        f"bitwise_{op}_{nibble_label}_"
                        f"a{a:x}_b{b:x}"
                    ),
                    conditions=(
                        (marker_gate, 40.0),
                        (f"{a_band}+{a}", 30.0),
                        (f"{b_band}+{b}", 30.0),
                    ),
                    threshold=80.0,
                    gate=opcode_gate,
                    gate_weight=1.0,
                    gate_bias=0.0,
                    writes=((f"{out_band}+{result}", 2.0 / S),),
                ))
    return tuple(rules)


# ---------------------------------------------------------------------------
# Wave W3: Wide ADD/SUB — stubs.
# ---------------------------------------------------------------------------


def wide_add_rules(
    *,
    operand_a_base: str,
    operand_b_base: str,
    result_base: str,
    carry_base: str,
    width_bytes: int,
    opcode_gate: str,
    marker_gate: str,
    S: float,
) -> Tuple[FFNRule, ...]:
    """Generate FFNRule list for wide multi-byte ADD with carry propagation.

    Stub for Wave W3 (``AddSub5StageBlock`` migration). Will emit:
      - per-byte sum nibble units (lo + hi)
      - per-byte carry-out units feeding ``carry_base+(k+1)``
      - inter-byte carry-in conditioned units (carry_in=0 vs 1)

    Args:
        operand_a_base: residual dim base for operand A's per-byte bands
            (e.g. ``"AX_LO"`` — helpers append ``+k`` for byte ``k``).
        operand_b_base: dim base for operand B's per-byte bands.
        result_base: dim base for the result per-byte bands
            (e.g. ``"OUTPUT_LO"``).
        carry_base: dim base for the inter-byte carry cascade
            (e.g. ``"CARRY"``).
        width_bytes: number of bytes in the wide operation (e.g. 4 for u32).
        opcode_gate: dim ref for the ``ADD`` opcode flag.
        marker_gate: dim name for the AX-style marker.
        S: SwiGLU scale.

    Returns:
        Empty tuple for now (W3 NOT IMPLEMENTED). The signature is pinned
        so the lowering pipeline and call-sites can be wired up first.
    """
    del (
        operand_a_base, operand_b_base, result_base, carry_base,
        width_bytes, opcode_gate, marker_gate, S,
    )
    raise NotImplementedError(
        "wide_add_rules: Wave W3 not implemented. "
        "Tracked under ``docs/IR_DSL_DESIGN.md`` Section 5 — AddSub migration."
    )


def wide_sub_rules(
    *,
    operand_a_base: str,
    operand_b_base: str,
    result_base: str,
    borrow_base: str,
    width_bytes: int,
    opcode_gate: str,
    marker_gate: str,
    S: float,
) -> Tuple[FFNRule, ...]:
    """Generate FFNRule list for wide multi-byte SUB with borrow.

    Stub for Wave W3 (same composite as ``wide_add_rules`` — they share
    the ``AddSub5StageBlock`` pipeline; only the per-nibble arithmetic and
    the carry-vs-borrow polarity differ).

    Args:
        operand_a_base: dim base for operand A (minuend).
        operand_b_base: dim base for operand B (subtrahend).
        result_base: dim base for the result per-byte bands.
        borrow_base: dim base for the inter-byte borrow cascade.
        width_bytes: number of bytes in the wide operation.
        opcode_gate: dim ref for the ``SUB`` opcode flag.
        marker_gate: dim name for the AX-style marker.
        S: SwiGLU scale.

    Returns:
        Stub raises ``NotImplementedError`` (W3 pending).
    """
    del (
        operand_a_base, operand_b_base, result_base, borrow_base,
        width_bytes, opcode_gate, marker_gate, S,
    )
    raise NotImplementedError(
        "wide_sub_rules: Wave W3 not implemented. "
        "Tracked under ``docs/IR_DSL_DESIGN.md`` Section 5 — AddSub migration."
    )


# ---------------------------------------------------------------------------
# Wave W2: Wide shift — stub.
# ---------------------------------------------------------------------------


def wide_shift_rules(
    *,
    direction: Literal["left", "right"],
    operand_base: str,
    shift_amount_dim: str,
    result_base: str,
    width_bytes: int,
    opcode_gate: str,
    marker_gate: str,
    S: float,
) -> Tuple[FFNRule, ...]:
    """Generate FFNRule list for wide shift (SHL/SHR).

    Stub for Wave W2 (``ALUShiftComposite`` migration). The composite is
    a per-byte select + precompute pipeline; the rules will encode the
    SHL/SHR lookup table currently produced by ``_set_layer13_shifts``
    (already partially migrated — see ``ops/l13_ops.py``).

    Args:
        direction: ``"left"`` (SHL) or ``"right"`` (SHR).
        operand_base: dim base for the operand per-byte bands.
        shift_amount_dim: dim name for the shift amount nibble.
        result_base: dim base for the result per-byte bands.
        width_bytes: number of bytes in the wide operation.
        opcode_gate: dim ref for the ``SHL`` / ``SHR`` opcode flag.
        marker_gate: dim name for the AX-style marker.
        S: SwiGLU scale.

    Returns:
        Stub raises ``NotImplementedError`` (W2 pending).
    """
    del (
        direction, operand_base, shift_amount_dim, result_base,
        width_bytes, opcode_gate, marker_gate, S,
    )
    raise NotImplementedError(
        "wide_shift_rules: Wave W2 not implemented. "
        "Tracked under ``docs/IR_DSL_DESIGN.md`` Section 5 — Shift migration."
    )


# ---------------------------------------------------------------------------
# Wave W5: Wide MUL — 9-stage pipeline stub.
# ---------------------------------------------------------------------------


def wide_mul_rules(
    *,
    operand_a_base: str,
    operand_b_base: str,
    result_base: str,
    carry_base: str,
    width_bytes: int,
    opcode_gate: str,
    marker_gate: str,
    S: float,
) -> Tuple[FFNRule, ...]:
    """Generate FFNRule list for the 9-stage wide MUL pipeline.

    Stub for Wave W5 (``FlattenedALUMul`` migration — the boss fight).
    The 9 stages currently live in ``efficient_alu_neural.py`` and span
    BD↔GE format conversion, schoolbook partial products, carry-pass
    propagation, and a final merge into ``OUTPUT_LO/HI``. The DSL
    migration will subdivide the lowering by stage so each ships
    independently (see Section 5 of the design doc).

    Args:
        operand_a_base: dim base for operand A per-byte bands.
        operand_b_base: dim base for operand B per-byte bands.
        result_base: dim base for the 2*width-byte result.
        carry_base: dim base for inter-stage carry cascades.
        width_bytes: input operand width in bytes.
        opcode_gate: dim ref for the ``MUL`` opcode flag.
        marker_gate: dim name for the AX-style marker.
        S: SwiGLU scale.

    Returns:
        Stub raises ``NotImplementedError`` (W5 pending).
    """
    del (
        operand_a_base, operand_b_base, result_base, carry_base,
        width_bytes, opcode_gate, marker_gate, S,
    )
    raise NotImplementedError(
        "wide_mul_rules: Wave W5 not implemented (9-stage pipeline). "
        "Tracked under ``docs/IR_DSL_DESIGN.md`` Section 5 — MUL migration."
    )


# ---------------------------------------------------------------------------
# Wave W4: Wide DIV/MOD — long-division stub.
# ---------------------------------------------------------------------------


def wide_div_rules(
    *,
    dividend_base: str,
    divisor_base: str,
    quotient_base: str,
    remainder_base: str,
    width_bytes: int,
    opcode_gate: str,
    marker_gate: str,
    S: float,
) -> Tuple[FFNRule, ...]:
    """Generate FFNRule list for the long-division DIV/MOD pipeline.

    Stub for Wave W4 (``FlattenedDivMod`` migration). The current
    composite is a multi-stage long-division pipeline using
    ``wide_sub_rules``-equivalents internally; the DSL migration will
    layer it on top of the W3 ``wide_sub_rules`` helper.

    Args:
        dividend_base: dim base for dividend per-byte bands.
        divisor_base: dim base for divisor per-byte bands.
        quotient_base: dim base for quotient per-byte bands.
        remainder_base: dim base for remainder per-byte bands.
        width_bytes: operand width in bytes.
        opcode_gate: dim ref for the ``DIV`` / ``MOD`` opcode flag.
        marker_gate: dim name for the AX-style marker.
        S: SwiGLU scale.

    Returns:
        Stub raises ``NotImplementedError`` (W4 pending).
    """
    del (
        dividend_base, divisor_base, quotient_base, remainder_base,
        width_bytes, opcode_gate, marker_gate, S,
    )
    raise NotImplementedError(
        "wide_div_rules: Wave W4 not implemented (long-division). "
        "Tracked under ``docs/IR_DSL_DESIGN.md`` Section 5 — DivMod migration."
    )


__all__ = [
    "bitwise_rules",
    "wide_add_rules",
    "wide_sub_rules",
    "wide_shift_rules",
    "wide_mul_rules",
    "wide_div_rules",
]
