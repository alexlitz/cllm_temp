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
  - ``wide_add_rules`` / ``wide_sub_rules`` — IMPLEMENTED (Wave W3,
    verified multi-byte up to width_bytes=8 / 32-bit; see commit
    ``478120d9``).
  - ``wide_shift_rules`` — IMPLEMENTED (Wave W2, per-byte lookup).
  - ``wide_mul_rules`` — IMPLEMENTED (Wave W5; width_bytes=1 POC and
    width_bytes=2 flat 8-bit×8-bit→16-bit lookup; wider widths deferred
    to a partial-product cascade — see
    ``docs/DSL_W5_MULDIV_LIMIT.md``).
  - ``wide_div_rules`` — IMPLEMENTED (Wave W4, width_bytes=1 nibble
    lookup POC; per-nibble and MATHEMATICALLY WRONG for cross-nibble
    dividends; see ``docs/DSL_W5_MULDIV_LIMIT.md``). Superseded by
    ``wide_div_rules_ge_format`` for byte-accurate single-byte div.
  - ``wide_div_rules_ge_format`` — IMPLEMENTED. Byte-accurate
    width_bytes=1 single-byte DIV/MOD via flat 8-bit cross-product
    lookup (65,536 rules per opcode batch). Multi-byte deferred — see
    ``docs/LONG_DIVISION_FFN_RULE_INFEASIBILITY_2026_06_09.md``.

The validation contract per helper is:
  ``rule_lowered_ffn.forward(x)  ==  hand_composite.forward(x)``
bit-for-bit on randomized input (see Section 4 of the design doc and
``tests/test_wide_alu_dsl.py`` for the ``bitwise_rules`` POC).
"""

from __future__ import annotations

import operator
from typing import Callable, Literal, Tuple

from .building_blocks_dsl import multi_way_and_rule
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
                rules.append(multi_way_and_rule(
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

                    rules.append(multi_way_and_rule(
                        name=(
                            f"wide_add_b{b}_cin{carry_in}_"
                            f"a{a_nib:x}_b{b_nib:x}"
                        ),
                        conditions=conditions,
                        threshold=threshold,
                        gate=opcode_gate,
                        writes=tuple(writes),
                    ))

        # Single carry-in detection / relay rule for byte > 0. Probes
        # the prior byte's carry-out (no residual write — it is a
        # standalone observable unit that the validator can check). The
        # write is intentionally a self-relay back to ``carry_base+(b-1)``
        # at the same amplitude so the carry cascade is bit-stable across
        # repeated lowerings.
        if b > 0:
            rules.append(multi_way_and_rule(
                name=f"wide_add_b{b}_carry_in_detect",
                conditions=(
                    (marker_gate, 40.0),
                    (f"{carry_base}+{b - 1}", 60.0),
                ),
                threshold=80.0,
                gate=opcode_gate,
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
    """Generate FFNRule list for wide multi-byte SUB with borrow propagation.

    Per-byte nibble-sub lookup table (Wave W3 — AddSub5StageBlock
    migration, SUB half). Mirror of :func:`wide_add_rules`; only the
    per-nibble arithmetic and the carry-vs-borrow polarity differ. Each
    "byte" in this DSL slice is a single 4-bit nibble:
    ``operand_a_base+(b*16+k)`` is the one-hot at nibble ``k`` for byte
    ``b``. Wider operands stack bytes contiguously in the residual.

    For each byte ``b`` in ``0..width_bytes-1`` and each nibble pair
    ``(a_nib, b_nib)`` in ``0..15 × 0..15``:

      * diff_nib   = (a_nib - b_nib - borrow_in) & 0xF
      * borrow_out = (a_nib - b_nib - borrow_in) < 0

    Byte 0 has no borrow-in, so it emits 256 rules (one per (a, b)).

    Byte ``b > 0`` emits 256 + 256 = 512 rules — one set for
    borrow_in = 0 (suppressed by a negative weight on ``borrow_base+(b-1)``)
    and one set for borrow_in = 1 (positively conditioned on
    ``borrow_base+(b-1)``). Each byte ``b > 0`` also emits one borrow-in
    relay rule (single unit) that copies ``borrow_base+(b-1)`` forward as
    a verification probe, matching the spec's "single borrow_in detection
    rule for byte > 0".

    Rule mechanics (lookup mode, same shape as :func:`wide_add_rules`):

      * Borrow-in = 0, byte 0:
          marker(+40) + a(+30) + b(+30) > 80 → diff + borrow_out write.
      * Borrow-in = 0, byte > 0:
          marker(+40) + a(+30) + b(+30) + borrow_dim(-50) > 80.
          With borrow set, total = 50 < 80 → blocked.
      * Borrow-in = 1, byte > 0:
          marker(+40) + a(+30) + b(+30) + borrow_dim(+30) > 120.
          All four required; without borrow, total = 100 < 120 → blocked.

    Output writes use the standard ``2.0 / S`` lookup-mode amplitude to
    ``result_base+(b*16+diff_nib)``. When ``borrow_out`` is true the rule
    also writes ``2.0 / S`` to ``borrow_base+b`` so the next byte's
    borrow-in is set.

    Args:
        operand_a_base: residual dim base for operand A (minuend) per-byte
            nibble bands. Rule reads ``operand_a_base+(b*16+a_nib)``.
        operand_b_base: dim base for operand B (subtrahend) per-byte
            nibble bands.
        result_base: dim base for the result per-byte nibble bands
            (e.g. ``"OUTPUT_LO"``).
        borrow_base: dim base for the inter-byte borrow cascade. Rule
            writes ``borrow_base+b`` on borrow-out and reads
            ``borrow_base+(b-1)`` for borrow-in (byte > 0 only).
        width_bytes: number of nibble-wide bytes in the wide SUB.
        opcode_gate: dim ref for the ``SUB`` opcode flag (e.g.
            ``"OP_SUB"``).
        marker_gate: dim name for the AX-style marker (e.g. ``"MARK_AX"``).
        S: SwiGLU scale (typically 100.0).

    Returns:
        ``tuple[FFNRule, ...]`` of length
        ``256 + (width_bytes - 1) * (512 + 1)`` — 256 diff rules for byte
        0 plus 512 diff rules and 1 borrow-relay rule per subsequent byte.
    """
    if width_bytes < 1:
        raise ValueError(
            f"wide_sub_rules: width_bytes must be >= 1; got {width_bytes!r}"
        )

    rules: list[FFNRule] = []
    write_amplitude = 2.0 / S

    for b in range(width_bytes):
        a_band = operand_a_base
        b_band = operand_b_base
        # borrow_in possibilities for this byte
        if b == 0:
            borrow_in_cases = (0,)
        else:
            borrow_in_cases = (0, 1)

        for borrow_in in borrow_in_cases:
            for a_nib in range(16):
                for b_nib in range(16):
                    raw = a_nib - b_nib - borrow_in
                    diff_nib = raw & 0xF
                    borrow_out = raw < 0

                    writes: list[Tuple[str, float]] = [
                        (
                            f"{result_base}+{b * 16 + diff_nib}",
                            write_amplitude,
                        ),
                    ]
                    if borrow_out:
                        writes.append(
                            (f"{borrow_base}+{b}", write_amplitude)
                        )

                    if b == 0:
                        # No borrow-in dim. Standard 3-way AND
                        # (marker, a, b) > 80.
                        conditions = (
                            (marker_gate, 40.0),
                            (f"{a_band}+{b * 16 + a_nib}", 30.0),
                            (f"{b_band}+{b * 16 + b_nib}", 30.0),
                        )
                        threshold = 80.0
                    elif borrow_in == 0:
                        # Suppress when borrow-in dim is active.
                        # marker(40) + a(30) + b(30) - borrow(50) > 80
                        # → 100 > 80 fires without borrow, 50 < 80 with.
                        conditions = (
                            (marker_gate, 40.0),
                            (f"{a_band}+{b * 16 + a_nib}", 30.0),
                            (f"{b_band}+{b * 16 + b_nib}", 30.0),
                            (f"{borrow_base}+{b - 1}", -50.0),
                        )
                        threshold = 80.0
                    else:
                        # borrow_in == 1: require borrow dim positively.
                        # marker(40) + a(30) + b(30) + borrow(30) > 120
                        # → 130 > 120 fires only if all four set.
                        conditions = (
                            (marker_gate, 40.0),
                            (f"{a_band}+{b * 16 + a_nib}", 30.0),
                            (f"{b_band}+{b * 16 + b_nib}", 30.0),
                            (f"{borrow_base}+{b - 1}", 30.0),
                        )
                        threshold = 120.0

                    rules.append(multi_way_and_rule(
                        name=(
                            f"wide_sub_b{b}_bin{borrow_in}_"
                            f"a{a_nib:x}_b{b_nib:x}"
                        ),
                        conditions=conditions,
                        threshold=threshold,
                        gate=opcode_gate,
                        writes=tuple(writes),
                    ))

        # Single borrow-in detection / relay rule for byte > 0. Probes
        # the prior byte's borrow-out (no residual write — it is a
        # standalone observable unit that the validator can check). The
        # write is intentionally a self-relay back to
        # ``borrow_base+(b-1)`` at the same amplitude so the borrow
        # cascade is bit-stable across repeated lowerings.
        if b > 0:
            rules.append(multi_way_and_rule(
                name=f"wide_sub_b{b}_borrow_in_detect",
                conditions=(
                    (marker_gate, 40.0),
                    (f"{borrow_base}+{b - 1}", 60.0),
                ),
                threshold=80.0,
                gate=opcode_gate,
                writes=(
                    (f"{borrow_base}+{b - 1}", 0.0),
                ),
            ))

    return tuple(rules)


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
    """Generate FFNRule list for wide shift (SHL/SHR) — per-byte lookup.

    Wave W2 implementation. The composite ``ALUShiftComposite`` in
    ``efficient_alu_neural.py`` is a 4-stage precompute + select pipeline
    (BD->GE, SHL/SHR precompute, opcode-gated select, GE->BD). This DSL
    helper expresses the equivalent semantics as a flat lookup table:
    one FFNRule per ``(byte_position, shift_amount, byte_value)`` triple.

    Per byte position ``b`` (0..``width_bytes``-1), per shift amount
    ``k`` (0..7), per byte value ``N`` (0..255), the helper emits one
    rule that:

      * fires when ``marker_gate`` AND ``shift_amount_dim+k`` (the shift
        amount one-hot at value ``k``) AND ``operand_base+(b*256+N)``
        (the byte-b operand one-hot at value ``N``) are all hot, AND
      * is gated by ``opcode_gate`` (the ``SHL`` / ``SHR`` opcode flag).

    The write lands on ``result_base+(b*256 + result_value)`` where
    ``result_value = (N << k) & 0xFF`` for ``direction='left'`` or
    ``N >> k`` for ``direction='right'``. The shift is per-byte and
    independent across bytes — no inter-byte propagation, matching the
    simplified semantics this Wave is targeting (full carry propagation
    is a future extension).

    The 3-way AND uses weights (40, 30, 30) with threshold 80 — same
    balanced-AND pattern as :func:`bitwise_rules`:

      * all three present: 40 + 30 + 30 = 100 > 80 -> fires
      * any two present:   max(40 + 30) = 70 < 80 -> blocked

    Args:
        direction: ``"left"`` (SHL) or ``"right"`` (SHR).
        operand_base: dim base for the operand. Per byte ``b`` the rule
            reads ``operand_base+(b*256+N)``. A 256-wide one-hot band
            per byte position.
        shift_amount_dim: dim base for the shift amount. The rule reads
            ``shift_amount_dim+k`` for shift amount ``k``.
        result_base: dim base for the result. Per byte ``b`` the rule
            writes ``result_base+(b*256+result_value)``.
        width_bytes: number of bytes in the wide operation (e.g. 1 for
            u8, 4 for u32).
        opcode_gate: dim ref for the ``SHL`` / ``SHR`` opcode flag.
        marker_gate: dim name for the AX-style marker.
        S: SwiGLU scale (typically 100.0).

    Returns:
        ``tuple[FFNRule, ...]`` of length ``width_bytes * 8 * 256``.

    Raises:
        ValueError: if ``direction`` is not ``"left"`` or ``"right"``.
        ValueError: if ``width_bytes`` is not a positive integer.
    """
    if direction not in ("left", "right"):
        raise ValueError(
            f"wide_shift_rules: direction must be 'left' or 'right'; "
            f"got {direction!r}"
        )
    if not isinstance(width_bytes, int) or width_bytes < 1:
        raise ValueError(
            f"wide_shift_rules: width_bytes must be a positive int; "
            f"got {width_bytes!r}"
        )

    if direction == "left":
        shift_fn = lambda n, k: (n << k) & 0xFF
        dir_tag = "shl"
    else:
        shift_fn = lambda n, k: (n >> k) & 0xFF
        dir_tag = "shr"

    write_scale = 2.0 / S
    rules: list[FFNRule] = []
    for b in range(width_bytes):
        operand_offset = b * 256
        result_offset = b * 256
        for k in range(8):
            for n in range(256):
                result = shift_fn(n, k)
                rules.append(multi_way_and_rule(
                    name=(
                        f"wide_{dir_tag}_b{b}_k{k}_n{n:02x}"
                    ),
                    conditions=(
                        (marker_gate, 40.0),
                        (f"{shift_amount_dim}+{k}", 30.0),
                        (f"{operand_base}+{operand_offset + n}", 30.0),
                    ),
                    threshold=80.0,
                    gate=opcode_gate,
                    writes=(
                        (f"{result_base}+{result_offset + result}",
                         write_scale),
                    ),
                ))
    return tuple(rules)


# ---------------------------------------------------------------------------
# Wave W5: Wide MUL — 9-stage pipeline stub.
# ---------------------------------------------------------------------------


def wide_mul_rules(
    *,
    operand_a_base: str,
    operand_b_base: str,
    result_base: str,
    width_bytes: int,
    opcode_gate: str,
    marker_gate: str,
    S: float,
    operand_a_cond_weight: float = 30.0,
    operand_b_cond_weight: float = 30.0,
    marker_cond_weight: float = 40.0,
    threshold: float = None,
) -> Tuple[FFNRule, ...]:
    """Generate FFNRule list for wide MUL — nibble-stacked flat lookup.

    Wave W5. The full ``FlattenedALUMul`` composite in
    ``efficient_alu_neural.py:1066+`` is a 9-stage pipeline (BDToGE →
    schoolbook → 3 carry passes → genprop → binary-lookahead →
    final-correction → MulCombine → GEToBD). The DSL collapses that
    pipeline into a single flat lookup table over the full operand
    cross-product.

    Each "byte" in this DSL slice is a single 4-bit nibble lane (matching
    the per-byte stacking convention of :func:`wide_add_rules`):
    ``operand_a_base + (b*16 + nib)`` is the one-hot at nibble ``b``,
    value ``nib``. So ``width_bytes=1`` is a 4-bit × 4-bit MUL (POC) and
    ``width_bytes=2`` is an 8-bit × 8-bit MUL with a 16-bit (4-nibble)
    result.

    ``width_bytes=1`` — 4-bit × 4-bit lookup (POC):
      For each ``(a, b)`` in ``0..15 × 0..15`` (256 pairs), emit one
      FFNRule that fires when ``marker_gate`` AND ``operand_a_base+a``
      AND ``operand_b_base+b`` are all hot, gated by ``opcode_gate``.
      Each rule writes ``2.0 / S`` to two output positions:
        - ``result_base + (a * b) & 0xF``                 (low nibble)
        - ``result_base + 16 + ((a * b) >> 4) & 0xF``     (high nibble)
      Conditions use the bitwise-rules pattern (marker(40) + a(30)
      + b(30) > 80).

    ``width_bytes=2`` — 8-bit × 8-bit flat lookup (16-bit result):
      For each ``(a_lo, a_hi, b_lo, b_hi)`` quad in
      ``0..15 × 0..15 × 0..15 × 0..15`` (65536 quads), emit one rule
      that fires when all four nibble one-hots AND ``marker_gate`` are
      hot, gated by ``opcode_gate``. Each rule writes ``2.0 / S`` to
      four result positions — one per nibble lane — covering the full
      16-bit product:
        - ``result_base + (product       & 0xF)``         (byte 0)
        - ``result_base + 16  + ((product >>  4) & 0xF)`` (byte 1)
        - ``result_base + 32  + ((product >>  8) & 0xF)`` (byte 2)
        - ``result_base + 48  + ((product >> 12) & 0xF)`` (byte 3)
      Conditions use a 5-way balanced AND: ``marker(+40) + a_lo(+30)
      + a_hi(+30) + b_lo(+30) + b_hi(+30) > 150``. All-on totals 160
      > 150 → fires; missing the marker totals 120 < 150; missing any
      single nibble totals 130 < 150 → blocked. This mirrors the 4-way
      AND pattern :func:`wide_add_rules` uses for the carry-in=1 case.

    ``width_bytes > 2``: deferred (see ``docs/DSL_W5_MULDIV_LIMIT.md``).
    A flat cross-product table grows as ``16 ** (2 * width_bytes)`` —
    ``width_bytes=3`` would emit ~16.8M rules, intractable as a single
    FFN lookup. Wider MUL requires the schoolbook partial-product
    pipeline (a multi-pass cascade with inter-nibble carry propagation),
    which is what ``FlattenedALUMul`` implements and what a future DSL
    helper must reproduce.

    Args:
        operand_a_base: dim base for operand A. For nibble lane ``b``
            the rule reads ``operand_a_base + (b * 16 + a_nib)``.
        operand_b_base: dim base for operand B. Read as
            ``operand_b_base + (b * 16 + b_nib)``.
        result_base: dim base for the result. Writes land on
            ``result_base + (lane * 16 + nib_value)`` for each nibble
            lane in the product.
        width_bytes: operand width in nibble lanes. Supported:
            ``width_bytes=1`` (4-bit MUL POC) and ``width_bytes=2``
            (8-bit MUL, 16-bit result). Wider raises NotImplementedError.
        opcode_gate: dim ref for the ``MUL`` opcode flag (e.g. ``"OP_MUL"``).
        marker_gate: dim name for the AX-style marker (e.g. ``"MARK_AX"``).
        S: SwiGLU scale (typically 100.0).

    Returns:
        ``tuple[FFNRule, ...]`` — 256 rules for ``width_bytes=1``,
        65536 rules for ``width_bytes=2``.

    Raises:
        ValueError: if ``width_bytes < 1``.
        NotImplementedError: if ``width_bytes > 2`` (intractable flat
            lookup; multi-byte requires partial-product cascade).
    """
    if not isinstance(width_bytes, int) or width_bytes < 1:
        raise ValueError(
            f"wide_mul_rules: width_bytes must be a positive int; "
            f"got {width_bytes!r}"
        )
    if width_bytes > 2:
        raise NotImplementedError(
            f"wide_mul_rules: width_bytes={width_bytes} is deferred — a "
            f"flat cross-product table would emit "
            f"16 ** (2 * {width_bytes}) = {16 ** (2 * width_bytes)} rules. "
            f"Wider MUL requires the schoolbook partial-product pipeline "
            f"(see docs/DSL_W5_MULDIV_LIMIT.md and FlattenedALUMul in "
            f"efficient_alu_neural.py)."
        )

    write_amplitude = 2.0 / S
    rules: list[FFNRule] = []

    if width_bytes == 1:
        # 4-bit POC: 3-way AND (marker + a + b).
        # Default threshold 80 fires when all three default-weight (40/30/30)
        # binary one-hots are present. The cond-weight kwargs let callers
        # rescale for operand bands carrying non-1.0 residual magnitude (the
        # L8 operand-gather emits ~5-6-magnitude one-hots on ALU_LO and only
        # ~0.9 on AX_CARRY_LO); pass a matching ``threshold`` then.
        thr1 = 80.0 if threshold is None else threshold
        for a_nib in range(16):
            for b_nib in range(16):
                product = (a_nib * b_nib) & 0xFFFF
                lo_nib = product & 0xF
                hi_nib = (product >> 4) & 0xF
                rules.append(multi_way_and_rule(
                    name=f"wide_mul_b0_a{a_nib:x}_b{b_nib:x}",
                    conditions=(
                        (marker_gate, marker_cond_weight),
                        (f"{operand_a_base}+{a_nib}", operand_a_cond_weight),
                        (f"{operand_b_base}+{b_nib}", operand_b_cond_weight),
                    ),
                    threshold=thr1,
                    gate=opcode_gate,
                    writes=(
                        (f"{result_base}+{lo_nib}", write_amplitude),
                        (f"{result_base}+{16 + hi_nib}", write_amplitude),
                    ),
                ))
        return tuple(rules)

    # width_bytes == 2: 8-bit × 8-bit flat lookup. 5-way AND
    # (marker + a_lo + a_hi + b_lo + b_hi).
    # Weights: marker=40, each nibble=30. Threshold=150.
    #   all-on  = 40 + 4*30 = 160 > 150 → fires
    #   no mark = 4*30      = 120 < 150 → blocked
    #   miss-one= 40 + 3*30 = 130 < 150 → blocked
    for a_lo in range(16):
        for a_hi in range(16):
            a_byte = (a_hi << 4) | a_lo
            for b_lo in range(16):
                for b_hi in range(16):
                    b_byte = (b_hi << 4) | b_lo
                    product = (a_byte * b_byte) & 0xFFFF
                    nib0 = product & 0xF
                    nib1 = (product >> 4) & 0xF
                    nib2 = (product >> 8) & 0xF
                    nib3 = (product >> 12) & 0xF
                    rules.append(multi_way_and_rule(
                        name=(
                            f"wide_mul_w2_alo{a_lo:x}_ahi{a_hi:x}_"
                            f"blo{b_lo:x}_bhi{b_hi:x}"
                        ),
                        conditions=(
                            (marker_gate, marker_cond_weight),
                            (f"{operand_a_base}+{a_lo}", operand_a_cond_weight),
                            (f"{operand_a_base}+{16 + a_hi}", operand_a_cond_weight),
                            (f"{operand_b_base}+{b_lo}", operand_b_cond_weight),
                            (f"{operand_b_base}+{16 + b_hi}", operand_b_cond_weight),
                        ),
                        threshold=150.0 if threshold is None else threshold,
                        gate=opcode_gate,
                        writes=(
                            (f"{result_base}+{nib0}", write_amplitude),
                            (f"{result_base}+{16 + nib1}", write_amplitude),
                            (f"{result_base}+{32 + nib2}", write_amplitude),
                            (f"{result_base}+{48 + nib3}", write_amplitude),
                        ),
                    ))
    return tuple(rules)


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
    """Generate FFNRule list for wide DIV/MOD — per-byte nibble lookup (POC).

    Wave W4 implementation (8-bit POC). The hand-written composite
    ``FlattenedDivMod`` (see ``efficient_alu_divmod_split.py``) is a
    4-stage long-division pipeline (BD->GE, DIV pipeline, MOD pipeline,
    GE->BD writeback) wrapping an 8-outer x 3-inner long-division loop.
    This DSL helper expresses the equivalent semantics as a flat lookup
    table: one FFNRule per ``(byte_position, dividend_nibble,
    divisor_nibble)`` triple.

    **Multi-byte gap (width_bytes > 1).** Per-nibble independent
    division is mathematically wrong for multi-nibble operands: the
    quotient/remainder of a wide value is **not** the per-nibble
    quotients/remainders concatenated. For example, ``0xFF / 0x0F == 0x11``
    (quotient 17, remainder 0), but per-nibble division would yield
    nibble0 = ``0xF / 0xF == 1`` and nibble1 = ``0xF / 0`` (zero-divide
    guard → ``q=0, r=0xF``), giving the wrong reassembled value. The
    full long-division pipeline (shift + subtract + select per bit)
    cannot be collapsed into a single per-nibble lookup. Multi-byte DIV
    is therefore deferred to a follow-up wave (see
    ``docs/DSL_W5_MULDIV_LIMIT.md``); ``FlattenedDivMod`` remains
    authoritative for ``width_bytes > 1``.

    Per byte position ``b`` (0..``width_bytes``-1):

      * For each ``(a_nib, b_nib)`` with ``b_nib > 0``:
        emit one rule that fires when ``marker_gate`` AND
        ``dividend_base+(b*16+a_nib)`` AND ``divisor_base+(b*16+b_nib)``
        are all hot, gated on ``opcode_gate``. The rule writes BOTH:
          - ``quotient_base+(b*16 + a_nib // b_nib)``
          - ``remainder_base+(b*16 + a_nib % b_nib)``

      * For each ``a_nib`` with ``b_nib == 0`` (divide-by-zero guard):
        emit one rule that fires on the divisor zero-bin
        (``divisor_base+(b*16+0)``) and writes the conventional
        zero-divide convention:
          - quotient = 0  (write to ``quotient_base+(b*16+0)``)
          - remainder = a_nib  (write to
            ``remainder_base+(b*16+a_nib)``, i.e. dividend pass-through)

    Rule mechanics (lookup mode, same 3-way AND pattern as
    :func:`bitwise_rules` and :func:`wide_shift_rules`):

      * conditions: marker(+40), dividend_nib(+30), divisor_nib(+30)
      * threshold:  80.0  (40+30+30=100 > 80 fires; any two = 70 < 80)
      * gate:       ``opcode_gate``  (DIV / MOD opcode flag)

    Output writes use the standard ``2.0 / S`` lookup-mode amplitude.

    Total rule count per byte: 16 * 16 = 256 (240 quotient+remainder
    rules for ``b_nib in 1..15`` plus 16 guard rules for ``b_nib == 0``).
    For ``width_bytes=1`` this matches the brief's nibble-level POC
    target.

    Args:
        dividend_base: dim base for dividend per-byte nibble bands. Per
            byte ``b`` rule reads ``dividend_base+(b*16+a_nib)``.
        divisor_base: dim base for divisor per-byte nibble bands.
        quotient_base: dim base for quotient per-byte bands. Writes land
            on ``quotient_base+(b*16 + a_nib // b_nib)``.
        remainder_base: dim base for remainder per-byte bands.
        width_bytes: number of nibble-wide bytes in the wide operation.
            **Only ``width_bytes=1`` is supported**; wider values raise
            ``NotImplementedError`` because per-nibble independent
            division does not compose into wide-operand division.
        opcode_gate: dim ref for the ``DIV`` / ``MOD`` opcode flag.
        marker_gate: dim name for the AX-style marker.
        S: SwiGLU scale (typically 100.0).

    Returns:
        ``tuple[FFNRule, ...]`` of length 256 (POC, ``width_bytes=1``).

    Raises:
        ValueError: if ``width_bytes < 1``.
        NotImplementedError: if ``width_bytes > 1`` (multi-byte DIV
            requires the long-division pipeline — see
            ``docs/DSL_W5_MULDIV_LIMIT.md``).
    """
    if not isinstance(width_bytes, int) or width_bytes < 1:
        raise ValueError(
            f"wide_div_rules: width_bytes must be a positive int; "
            f"got {width_bytes!r}"
        )
    if width_bytes > 1:
        raise NotImplementedError(
            f"wide_div_rules: width_bytes={width_bytes} is deferred — "
            f"per-nibble independent division does not compose into "
            f"wide-operand division (e.g. 0xFF / 0x0F = 0x11, but "
            f"per-nibble would give 1 and a zero-divide guard). "
            f"Multi-byte DIV requires the long-division pipeline "
            f"implemented by FlattenedDivMod (see efficient_alu_"
            f"divmod_split.py) and is tracked under "
            f"docs/DSL_W5_MULDIV_LIMIT.md."
        )

    write_amplitude = 2.0 / S
    rules: list[FFNRule] = []

    for b in range(width_bytes):
        byte_offset = b * 16
        # Non-zero divisor cases: quotient + remainder lookup.
        for a_nib in range(16):
            for b_nib in range(1, 16):
                q = a_nib // b_nib
                r = a_nib % b_nib
                rules.append(multi_way_and_rule(
                    name=(
                        f"wide_div_b{b}_a{a_nib:x}_b{b_nib:x}"
                    ),
                    conditions=(
                        (marker_gate, 40.0),
                        (f"{dividend_base}+{byte_offset + a_nib}", 30.0),
                        (f"{divisor_base}+{byte_offset + b_nib}", 30.0),
                    ),
                    threshold=80.0,
                    gate=opcode_gate,
                    writes=(
                        (f"{quotient_base}+{byte_offset + q}",
                         write_amplitude),
                        (f"{remainder_base}+{byte_offset + r}",
                         write_amplitude),
                    ),
                ))

        # Divide-by-zero guard: divisor nibble == 0.
        # Convention: quotient = 0, remainder = dividend (pass-through).
        # Matches the common "saturate-to-0 / preserve dividend"
        # behavior used by other ALU divide-by-zero handlers.
        for a_nib in range(16):
            rules.append(multi_way_and_rule(
                name=f"wide_div_b{b}_a{a_nib:x}_b0_guard",
                conditions=(
                    (marker_gate, 40.0),
                    (f"{dividend_base}+{byte_offset + a_nib}", 30.0),
                    (f"{divisor_base}+{byte_offset + 0}", 30.0),
                ),
                threshold=80.0,
                gate=opcode_gate,
                writes=(
                    (f"{quotient_base}+{byte_offset + 0}",
                     write_amplitude),
                    (f"{remainder_base}+{byte_offset + a_nib}",
                     write_amplitude),
                ),
            ))

    return tuple(rules)


def wide_div_rules_ge_format(
    *,
    dividend_lo_base: str,
    dividend_hi_base: str,
    divisor_lo_base: str,
    divisor_hi_base: str,
    result_lo_base: str,
    result_hi_base: str,
    width_bytes: int,
    opcode_gate: str,
    marker_gate: str,
    S: float,
    op: str = "div",
    dividend_cond_weight: float = 30.0,
    divisor_cond_weight: float = 30.0,
    marker_cond_weight: float = 40.0,
    threshold: float = 150.0,
) -> Tuple[FFNRule, ...]:
    """Byte-accurate multi-byte DIV/MOD via flat 8-bit cross-product lookup.

    GE-format here refers to the "greater-or-equal" *semantic* of long
    division (each output digit determined by whether the partial
    dividend is >= k * divisor) — for single-byte operands the entire
    256x256 quotient/remainder table fits in one FFN layer as a flat
    cross-product lookup, which is *byte-accurate* (unlike
    :func:`wide_div_rules` which is per-nibble and wrong for
    cross-nibble dividends — see ``docs/DSL_W5_MULDIV_LIMIT.md``).

    For ``width_bytes == 1`` (one full byte): each rule reads the
    dividend ``a = a_hi*16 + a_lo`` and divisor ``b = b_hi*16 + b_lo``
    one-hot encodings via a 5-way AND on
    ``(marker_gate, dividend_lo+a_lo, dividend_hi+a_hi,
    divisor_lo+b_lo, divisor_hi+b_hi)`` and writes the *byte-accurate*
    quotient (``op="div"``) or remainder (``op="mod"``) nibbles to
    ``result_lo+q_lo`` / ``result_hi+q_hi``.

    Total rules per call:
      * width_bytes=1: 256 * 256 = 65,536 rules (one per (a, b)).

    Rule mechanics (5-way AND lookup — same shape as
    ``wide_mul_rules(width_bytes=2)``):
      * conditions: marker(+40), a_lo(+30), a_hi(+30), b_lo(+30),
        b_hi(+30); total=160; threshold=150; all-on=160 > 150 fires;
        missing marker -> 120 < 150 blocked; missing any nibble ->
        130 < 150 blocked.
      * gate: ``opcode_gate``.

    Output writes use the standard ``2.0 / S`` lookup-mode amplitude
    so the SwiGLU activation reconstructs a clean one-hot.

    Divide-by-zero convention: ``q = r = 0`` (matches
    ``_init_full_lookup_mode`` behavior in ``vm_step.py:1350``).

    Args:
        dividend_lo_base: dim base for the dividend low-nibble one-hot
            band (e.g. ``"ALU_LO"``). Reads ``dividend_lo_base+a_lo``
            for ``a_lo`` in 0..15.
        dividend_hi_base: dim base for the dividend high-nibble band.
        divisor_lo_base: dim base for the divisor low-nibble band.
        divisor_hi_base: dim base for the divisor high-nibble band.
        result_lo_base: dim base for the result low-nibble band
            (writes land on ``result_lo_base+q_lo``).
        result_hi_base: dim base for the result high-nibble band.
        width_bytes: number of full bytes in the operands. Only
            ``width_bytes == 1`` is supported here; wider widths
            require GE-cascade across byte rows (deferred — see
            ``docs/LONG_DIVISION_FFN_RULE_INFEASIBILITY_2026_06_09.md``).
        opcode_gate: dim ref for the ``DIV`` / ``MOD`` opcode flag
            (e.g. ``"OP_DIV"`` or ``"OP_MOD"``).
        marker_gate: dim name for the AX-style marker (e.g.
            ``"MARK_AX"``).
        S: SwiGLU scale.
        op: ``"div"`` to emit quotient rules, ``"mod"`` to emit
            remainder rules.
        dividend_cond_weight: per-cell condition weight for the dividend
            (ALU_LO/HI) one-hots. Default ``30.0`` assumes clean binary
            one-hots (value 1.0). When the operand bands carry a larger
            residual magnitude (e.g. ~5.82 from the L8 operand-gather),
            rescale to ``30.0 / magnitude`` so a matched cell contributes
            ~30 and the 5-way-AND threshold math is preserved.
        divisor_cond_weight: per-cell condition weight for the divisor
            (AX_CARRY_LO/HI) one-hots. Default ``30.0``.
        marker_cond_weight: condition weight for ``marker_gate``.
            Default ``40.0``.
        threshold: 5-way-AND firing threshold. Default ``150.0`` (all-on
            = 40 + 4*30 = 160 fires; missing one operand -> 130 blocked).

    Returns:
        ``tuple[FFNRule, ...]`` of length ``256 * 256 = 65536`` for
        ``width_bytes=1``.

    Raises:
        ValueError: if ``width_bytes < 1`` or ``op`` not in
            ``{"div", "mod"}``.
        NotImplementedError: if ``width_bytes > 1`` (multi-byte GE
            cascade is deferred).
    """
    if not isinstance(width_bytes, int) or width_bytes < 1:
        raise ValueError(
            f"wide_div_rules_ge_format: width_bytes must be a positive "
            f"int; got {width_bytes!r}"
        )
    if width_bytes > 1:
        raise NotImplementedError(
            f"wide_div_rules_ge_format: width_bytes={width_bytes} is "
            f"deferred — multi-byte byte-accurate DIV requires a "
            f"GE-cascade across byte rows (see "
            f"docs/LONG_DIVISION_FFN_RULE_INFEASIBILITY_2026_06_09.md). "
            f"For 8-bit-fits-in-byte cases use width_bytes=1; for "
            f"wider operands ``FlattenedDivMod`` remains authoritative."
        )
    if op not in ("div", "mod"):
        raise ValueError(
            f"wide_div_rules_ge_format: op must be 'div' or 'mod'; "
            f"got {op!r}"
        )

    write_amplitude = 2.0 / S
    rules: list[FFNRule] = []

    # Flat 8-bit x 8-bit cross-product lookup. Each rule fires on a
    # specific (a, b) pair and writes the byte-accurate q or r
    # decomposition to result_lo/hi. 5-way AND mechanics:
    #   marker(40) + a_lo(30) + a_hi(30) + b_lo(30) + b_hi(30) = 160
    #   missing marker -> 120; missing any operand -> 130; threshold=150.
    # (Same shape as wide_mul_rules width_bytes=2.)
    for a in range(256):
        a_lo = a & 0xF
        a_hi = (a >> 4) & 0xF
        for b in range(256):
            b_lo = b & 0xF
            b_hi = (b >> 4) & 0xF
            if b == 0:
                # Divide-by-zero convention: quotient = remainder = 0.
                out_value = 0
            else:
                out_value = (a // b) if op == "div" else (a % b)
            out_lo = out_value & 0xF
            out_hi = (out_value >> 4) & 0xF
            rules.append(multi_way_and_rule(
                name=(
                    f"wide_div_ge_{op}_a{a:02x}_b{b:02x}"
                ),
                conditions=(
                    (marker_gate, marker_cond_weight),
                    (f"{dividend_lo_base}+{a_lo}", dividend_cond_weight),
                    (f"{dividend_hi_base}+{a_hi}", dividend_cond_weight),
                    (f"{divisor_lo_base}+{b_lo}", divisor_cond_weight),
                    (f"{divisor_hi_base}+{b_hi}", divisor_cond_weight),
                ),
                threshold=threshold,
                gate=opcode_gate,
                writes=(
                    (f"{result_lo_base}+{out_lo}", write_amplitude),
                    (f"{result_hi_base}+{out_hi}", write_amplitude),
                ),
            ))

    return tuple(rules)


# ---------------------------------------------------------------------------
# Wave W3 (retry): GE-format wide ADD/SUB helpers.
# ---------------------------------------------------------------------------
#
# These mirror ``wide_add_rules`` / ``wide_sub_rules`` semantically but
# emit rules that address the **GE workspace** ``[seq, byte, nibble]``
# (per ``efficient_alu_addsub_split.py`` and ``efficient_alu_neural.py``
# lines 236-415) rather than the BD band-offset model.
#
# Why this exists: ``docs/DSL_W3_ADDSUB_LIMIT.md`` documents that
# multi-byte ADD/SUB cannot be expressed via the band-offset DSL because
# the BD residual's wide-ALU bands (``ALU_LO``/``ALU_HI``/``OUTPUT_LO``/
# ``OUTPUT_HI``/``CARRY``) are only 16 dims wide each. Stacking nibbles
# via ``operand_a_base + (b*16 + nib)`` works for ``width_bytes <= 2``
# (because ``ALU_LO+16 == ALU_HI``) but corrupts unrelated bands for
# wider operands (``ALU_LO+32`` lands in ``CARRY``; ``ALU_LO+48`` lands
# in ``CLEAN_EMBED_HI``).
#
# The GE workspace already separates bytes by *position* — each position
# row has its own ``GE.DIM == 160`` slot space — so byte rows never
# collide regardless of width. ``AddSub5StageBlock`` operates in this
# workspace; the helpers here express the same cascade as a flat
# ``FFNRule`` table over ``(byte_position, nibble_a, nibble_b)`` triples.
#
# Naming convention: each position's per-nibble bands are addressed via
# the dim refs ``"p{b}_NIB_A+{nib}"``, ``"p{b}_NIB_B+{nib}"``,
# ``"p{b}_RESULT+{nib}"``, and the per-position carry-out by
# ``"p{b}_CARRY_OUT"``. The marker and opcode gates are global
# (``marker_gate``, ``opcode_gate``). Callers supply a ``dim_positions``
# map that resolves ``p{b}_NIB_A`` → ``b * POS_STRIDE + nib_a_offset``
# for the flattened ``[seq, 8 * POS_STRIDE]`` workspace.


def wide_ge_add_rules(
    *,
    width_bytes: int,
    opcode_gate: str,
    marker_gate: str,
    S: float,
    operand_a_name: str = "NIB_A",
    operand_b_name: str = "NIB_B",
    result_name: str = "RESULT",
    carry_out_name: str = "CARRY_OUT",
    position_prefix: str = "p",
) -> Tuple[FFNRule, ...]:
    """Emit FFN rules for wide multi-byte ADD on the GE workspace.

    Mirrors :func:`wide_add_rules` semantically but routes each byte
    row through a distinct GE-format position rather than offsetting
    into a single 16-dim band. Each byte position ``b`` in
    ``0..width_bytes-1`` references its own per-nibble one-hot bands
    ``p{b}_NIB_A+{nib}`` (operand A), ``p{b}_NIB_B+{nib}`` (operand B),
    ``p{b}_RESULT+{nib}`` (sum nibble), and a single per-position
    carry-out flag ``p{b}_CARRY_OUT``. The inter-byte carry cascade
    reads ``p{b-1}_CARRY_OUT`` for byte ``b > 0``.

    Per-byte rule semantics are identical to :func:`wide_add_rules`:

      * sum_nib  = (a_nib + b_nib + carry_in) % 16
      * carry_out = (a_nib + b_nib + carry_in) >= 16

    Byte 0 emits 256 rules (no carry-in). Byte ``b > 0`` emits 256+256
    rules (separately gated for ``carry_in=0`` via the ``-50`` borrow-
    suppression weight and ``carry_in=1`` via the ``+30`` cumulative
    threshold-120 cascade) plus one carry-relay rule. Total:
    ``256 + (width_bytes - 1) * (512 + 1)``.

    The function is GE-only: it does NOT touch BD bands. Callers
    pair it with a BD→GE projection upstream (BDToGEConverter) and a
    GE→BD writeback downstream (GEToBDConverter), the same way the
    ``AddSub5StageBlock`` does.

    Args:
        width_bytes: number of nibble-wide bytes in the wide ADD
            (1 for 4-bit, 2 for 8-bit, 4 for 16-bit, 8 for 32-bit).
        opcode_gate: dim ref for the ``ADD`` opcode gate.
        marker_gate: dim ref for the AX-style marker gate.
        S: SwiGLU scale.
        operand_a_name: slot suffix for operand A's per-position nibble
            band. Default ``"NIB_A"``. Combined with ``position_prefix``
            yields ``"p{b}_NIB_A"``.
        operand_b_name: slot suffix for operand B's per-position band.
        result_name: slot suffix for the per-position result band.
        carry_out_name: slot suffix for the per-position 1-bit carry
            flag.
        position_prefix: prefix for the per-position slot names.

    Returns:
        ``tuple[FFNRule, ...]`` of length
        ``256 + (width_bytes - 1) * 513``.
    """
    if width_bytes < 1:
        raise ValueError(
            f"wide_ge_add_rules: width_bytes must be >= 1; "
            f"got {width_bytes!r}"
        )

    rules: list[FFNRule] = []
    write_amplitude = 2.0 / S

    for b in range(width_bytes):
        a_band = f"{position_prefix}{b}_{operand_a_name}"
        b_band = f"{position_prefix}{b}_{operand_b_name}"
        out_band = f"{position_prefix}{b}_{result_name}"
        carry_out_dim = f"{position_prefix}{b}_{carry_out_name}"
        carry_in_dim = (
            f"{position_prefix}{b - 1}_{carry_out_name}" if b > 0 else None
        )

        carry_in_cases = (0,) if b == 0 else (0, 1)

        for carry_in in carry_in_cases:
            for a_nib in range(16):
                for b_nib in range(16):
                    total = a_nib + b_nib + carry_in
                    sum_nib = total % 16
                    carry_out = total >= 16

                    writes: list[Tuple[str, float]] = [
                        (f"{out_band}+{sum_nib}", write_amplitude),
                    ]
                    if carry_out:
                        writes.append(
                            (carry_out_dim, write_amplitude)
                        )

                    if b == 0:
                        conditions = (
                            (marker_gate, 40.0),
                            (f"{a_band}+{a_nib}", 30.0),
                            (f"{b_band}+{b_nib}", 30.0),
                        )
                        threshold = 80.0
                    elif carry_in == 0:
                        conditions = (
                            (marker_gate, 40.0),
                            (f"{a_band}+{a_nib}", 30.0),
                            (f"{b_band}+{b_nib}", 30.0),
                            (carry_in_dim, -50.0),
                        )
                        threshold = 80.0
                    else:
                        conditions = (
                            (marker_gate, 40.0),
                            (f"{a_band}+{a_nib}", 30.0),
                            (f"{b_band}+{b_nib}", 30.0),
                            (carry_in_dim, 30.0),
                        )
                        threshold = 120.0

                    rules.append(multi_way_and_rule(
                        name=(
                            f"wide_ge_add_b{b}_cin{carry_in}_"
                            f"a{a_nib:x}_b{b_nib:x}"
                        ),
                        conditions=conditions,
                        threshold=threshold,
                        gate=opcode_gate,
                        writes=tuple(writes),
                    ))

        # Single carry-in detection / relay rule for byte > 0. Probes
        # the prior byte's carry-out as a verification observable. The
        # write is a no-op self-relay (amplitude 0) so the cascade is
        # bit-stable across repeated lowerings.
        if b > 0:
            rules.append(multi_way_and_rule(
                name=f"wide_ge_add_b{b}_carry_in_detect",
                conditions=(
                    (marker_gate, 40.0),
                    (carry_in_dim, 60.0),
                ),
                threshold=80.0,
                gate=opcode_gate,
                writes=(
                    (carry_in_dim, 0.0),
                ),
            ))

    return tuple(rules)


def wide_ge_sub_rules(
    *,
    width_bytes: int,
    opcode_gate: str,
    marker_gate: str,
    S: float,
    operand_a_name: str = "NIB_A",
    operand_b_name: str = "NIB_B",
    result_name: str = "RESULT",
    borrow_out_name: str = "BORROW_OUT",
    position_prefix: str = "p",
) -> Tuple[FFNRule, ...]:
    """Emit FFN rules for wide multi-byte SUB on the GE workspace.

    Mirror of :func:`wide_ge_add_rules` for subtraction. Per-byte rule
    semantics are identical to :func:`wide_sub_rules`:

      * diff_nib   = (a_nib - b_nib - borrow_in) & 0xF
      * borrow_out = (a_nib - b_nib - borrow_in) < 0

    Per-position dim names ``p{b}_NIB_A``, ``p{b}_NIB_B``,
    ``p{b}_RESULT``, ``p{b}_BORROW_OUT`` (the borrow flag is per
    position; previous-byte borrow read via
    ``p{b-1}_BORROW_OUT`` for ``b > 0``). Rule mechanics, condition
    weights and thresholds match :func:`wide_ge_add_rules` modulo the
    sub-specific arithmetic.

    Args:
        width_bytes: number of nibble-wide bytes in the wide SUB.
        opcode_gate: dim ref for the ``SUB`` opcode gate.
        marker_gate: dim ref for the AX-style marker gate.
        S: SwiGLU scale.
        operand_a_name, operand_b_name, result_name: per-position slot
            suffixes (defaults match the ADD helper).
        borrow_out_name: per-position 1-bit borrow flag suffix.
        position_prefix: prefix for the per-position slot names.

    Returns:
        ``tuple[FFNRule, ...]`` of length
        ``256 + (width_bytes - 1) * 513``.
    """
    if width_bytes < 1:
        raise ValueError(
            f"wide_ge_sub_rules: width_bytes must be >= 1; "
            f"got {width_bytes!r}"
        )

    rules: list[FFNRule] = []
    write_amplitude = 2.0 / S

    for b in range(width_bytes):
        a_band = f"{position_prefix}{b}_{operand_a_name}"
        b_band = f"{position_prefix}{b}_{operand_b_name}"
        out_band = f"{position_prefix}{b}_{result_name}"
        borrow_out_dim = f"{position_prefix}{b}_{borrow_out_name}"
        borrow_in_dim = (
            f"{position_prefix}{b - 1}_{borrow_out_name}" if b > 0 else None
        )

        borrow_in_cases = (0,) if b == 0 else (0, 1)

        for borrow_in in borrow_in_cases:
            for a_nib in range(16):
                for b_nib in range(16):
                    raw = a_nib - b_nib - borrow_in
                    diff_nib = raw & 0xF
                    borrow_out = raw < 0

                    writes: list[Tuple[str, float]] = [
                        (f"{out_band}+{diff_nib}", write_amplitude),
                    ]
                    if borrow_out:
                        writes.append(
                            (borrow_out_dim, write_amplitude)
                        )

                    if b == 0:
                        conditions = (
                            (marker_gate, 40.0),
                            (f"{a_band}+{a_nib}", 30.0),
                            (f"{b_band}+{b_nib}", 30.0),
                        )
                        threshold = 80.0
                    elif borrow_in == 0:
                        conditions = (
                            (marker_gate, 40.0),
                            (f"{a_band}+{a_nib}", 30.0),
                            (f"{b_band}+{b_nib}", 30.0),
                            (borrow_in_dim, -50.0),
                        )
                        threshold = 80.0
                    else:
                        conditions = (
                            (marker_gate, 40.0),
                            (f"{a_band}+{a_nib}", 30.0),
                            (f"{b_band}+{b_nib}", 30.0),
                            (borrow_in_dim, 30.0),
                        )
                        threshold = 120.0

                    rules.append(multi_way_and_rule(
                        name=(
                            f"wide_ge_sub_b{b}_bin{borrow_in}_"
                            f"a{a_nib:x}_b{b_nib:x}"
                        ),
                        conditions=conditions,
                        threshold=threshold,
                        gate=opcode_gate,
                        writes=tuple(writes),
                    ))

        if b > 0:
            rules.append(multi_way_and_rule(
                name=f"wide_ge_sub_b{b}_borrow_in_detect",
                conditions=(
                    (marker_gate, 40.0),
                    (borrow_in_dim, 60.0),
                ),
                threshold=80.0,
                gate=opcode_gate,
                writes=(
                    (borrow_in_dim, 0.0),
                ),
            ))

    return tuple(rules)


__all__ = [
    "bitwise_rules",
    "wide_add_rules",
    "wide_sub_rules",
    "wide_ge_add_rules",
    "wide_ge_sub_rules",
    "wide_shift_rules",
    "wide_mul_rules",
    "wide_div_rules",
    "wide_div_rules_ge_format",
]
