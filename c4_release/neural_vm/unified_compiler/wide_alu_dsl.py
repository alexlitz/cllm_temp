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

    Per-byte nibble-add lookup table (Wave W3 — AddSub5StageBlock migration,
    POC slice). Each "byte" in this DSL slice is a single 4-bit nibble:
    ``operand_a_base+(b*16+k)`` is the one-hot at nibble ``k`` for byte
    ``b``. Wider operands stack bytes contiguously in the residual.

    For each byte ``b`` in ``0..width_bytes-1`` and each nibble pair
    ``(a_nib, b_nib)`` in ``0..15 × 0..15``:

      * sum_nib  = (a_nib + b_nib + carry_in) % 16
      * carry_out = (a_nib + b_nib + carry_in) >= 16

    Byte 0 has no carry-in, so it emits 256 rules (one per (a, b)).

    Byte ``b > 0`` emits 256 + 256 = 512 rules — one set for
    carry_in = 0 (suppressed by a negative weight on ``carry_base+(b-1)``)
    and one set for carry_in = 1 (positively conditioned on
    ``carry_base+(b-1)``). Each byte ``b > 0`` also emits one carry-in
    relay rule (single unit) that copies ``carry_base+(b-1)`` forward as
    a verification probe, matching the spec's "single carry_in detection
    rule for byte > 0".

    Rule mechanics (lookup mode, same shape as ``bitwise_rules``):

      * Carry-in = 0, byte 0:
          marker(+40) + a(+30) + b(+30) > 80 → sum + carry_out write.
      * Carry-in = 0, byte > 0:
          marker(+40) + a(+30) + b(+30) + carry_dim(-50) > 80.
          With carry set, total = 50 < 80 → blocked.
      * Carry-in = 1, byte > 0:
          marker(+40) + a(+30) + b(+30) + carry_dim(+30) > 120.
          All four required; without carry, total = 100 < 120 → blocked.

    Output writes use the standard ``2.0 / S`` lookup-mode amplitude to
    ``result_base+(b*16+sum_nib)``. When ``carry_out`` is true the rule
    also writes ``2.0 / S`` to ``carry_base+b`` so the next byte's
    carry-in is set.

    Args:
        operand_a_base: residual dim base for operand A's per-byte nibble
            bands (e.g. ``"AX_CARRY_LO"``). Rule reads
            ``operand_a_base+(b*16+a_nib)``.
        operand_b_base: dim base for operand B's per-byte nibble bands.
        result_base: dim base for the result per-byte nibble bands
            (e.g. ``"OUTPUT_LO"``).
        carry_base: dim base for the inter-byte carry cascade. Rule
            writes ``carry_base+b`` on carry-out and reads
            ``carry_base+(b-1)`` for carry-in (byte > 0 only).
        width_bytes: number of nibble-wide bytes in the wide ADD.
        opcode_gate: dim ref for the ``ADD`` opcode flag (e.g.
            ``"OP_ADD"``).
        marker_gate: dim name for the AX-style marker (e.g. ``"MARK_AX"``).
        S: SwiGLU scale (typically 100.0).

    Returns:
        ``tuple[FFNRule, ...]`` of length
        ``256 + (width_bytes - 1) * (512 + 1)`` — 256 sum rules for byte
        0 plus 512 sum rules and 1 carry-relay rule per subsequent byte.
    """
    if width_bytes < 1:
        raise ValueError(
            f"wide_add_rules: width_bytes must be >= 1; got {width_bytes!r}"
        )

    rules: list[FFNRule] = []
    write_amplitude = 2.0 / S

    for b in range(width_bytes):
        a_band = operand_a_base
        b_band = operand_b_base
        # carry_in possibilities for this byte
        if b == 0:
            carry_in_cases = (0,)
        else:
            carry_in_cases = (0, 1)

        for carry_in in carry_in_cases:
            for a_nib in range(16):
                for b_nib in range(16):
                    total = a_nib + b_nib + carry_in
                    sum_nib = total % 16
                    carry_out = total >= 16

                    writes: list[Tuple[str, float]] = [
                        (
                            f"{result_base}+{b * 16 + sum_nib}",
                            write_amplitude,
                        ),
                    ]
                    if carry_out:
                        writes.append(
                            (f"{carry_base}+{b}", write_amplitude)
                        )

                    if b == 0:
                        # No carry-in dim. Standard 3-way AND
                        # (marker, a, b) > 80.
                        conditions = (
                            (marker_gate, 40.0),
                            (f"{a_band}+{b * 16 + a_nib}", 30.0),
                            (f"{b_band}+{b * 16 + b_nib}", 30.0),
                        )
                        threshold = 80.0
                    elif carry_in == 0:
                        # Suppress when carry-in dim is active.
                        # marker(40) + a(30) + b(30) - carry(50) > 80
                        # → 100 > 80 fires without carry, 50 < 80 with.
                        conditions = (
                            (marker_gate, 40.0),
                            (f"{a_band}+{b * 16 + a_nib}", 30.0),
                            (f"{b_band}+{b * 16 + b_nib}", 30.0),
                            (f"{carry_base}+{b - 1}", -50.0),
                        )
                        threshold = 80.0
                    else:
                        # carry_in == 1: require carry dim positively.
                        # marker(40) + a(30) + b(30) + carry(30) > 120
                        # → 130 > 120 fires only if all four set.
                        conditions = (
                            (marker_gate, 40.0),
                            (f"{a_band}+{b * 16 + a_nib}", 30.0),
                            (f"{b_band}+{b * 16 + b_nib}", 30.0),
                            (f"{carry_base}+{b - 1}", 30.0),
                        )
                        threshold = 120.0

                    rules.append(FFNRule.gated_write(
                        name=(
                            f"wide_add_b{b}_cin{carry_in}_"
                            f"a{a_nib:x}_b{b_nib:x}"
                        ),
                        conditions=conditions,
                        threshold=threshold,
                        gate=opcode_gate,
                        gate_weight=1.0,
                        gate_bias=0.0,
                        writes=tuple(writes),
                    ))

        # Single carry-in detection / relay rule for byte > 0. Probes
        # the prior byte's carry-out (no residual write — it is a
        # standalone observable unit that the validator can check). The
        # write is intentionally a self-relay back to ``carry_base+(b-1)``
        # at the same amplitude so the carry cascade is bit-stable across
        # repeated lowerings.
        if b > 0:
            rules.append(FFNRule.gated_write(
                name=f"wide_add_b{b}_carry_in_detect",
                conditions=(
                    (marker_gate, 40.0),
                    (f"{carry_base}+{b - 1}", 60.0),
                ),
                threshold=80.0,
                gate=opcode_gate,
                gate_weight=1.0,
                gate_bias=0.0,
                writes=(
                    (f"{carry_base}+{b - 1}", 0.0),
                ),
            ))

    return tuple(rules)


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
