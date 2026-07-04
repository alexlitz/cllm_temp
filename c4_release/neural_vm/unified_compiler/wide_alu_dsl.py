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
from typing import Callable, Literal, Mapping, Optional, Sequence, Tuple

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
    operand_a_cond_weight: float = 30.0,
    operand_b_cond_weight: float = 30.0,
    marker_cond_weight: float = 40.0,
    threshold: float = 80.0,
    result_write_value: float = 2.0,
    name_prefix: str = "bitwise",
    name_suffix: str = "",
    gate_weight: float = 1.0,
    emit_stale_cancel_band: bool = False,
    stale_operand_weight: float = 1000.0,
    stale_residual_factor: float = 1.02,
) -> Tuple[FFNRule, ...]:
    """Generate FFNRule list for a per-byte bitwise op (AND/OR/XOR).

    Emits 512 rules per op (256 low-nibble + 256 high-nibble cross-product),
    matching the structure of ``_layer10_alu_bitwise_rules`` in
    ``ops/l10_ops.py`` — the existing rule-driven L10 bitwise bake. With
    ``emit_stale_cancel_band=True`` it ALSO emits the L10 stale-ALU residue
    cancel band INTERLEAVED per nibble (287 rules per nibble = 574 total),
    so the L10 main-FFN bitwise builder is a pure call into this generator.

    Each main unit is a 3-way AND across (``marker_gate``, ``operand_*[a]``,
    ``operand_*[b]``) gated on ``opcode_gate``. Weights (40, 30, 30) and
    threshold 80 implement the balanced 3-way AND used by the legacy bake:

      * all three present: 40 + 30 + 30 = 100 > 80 → fires
      * any two present:   max(40 + 30) = 70 < 80 → blocked

    The output write is ``2.0 / S`` to ``result_*[op_fn(a, b)]``, matching
    the legacy ``OUTPUT_LO/HI_THIS_STEP`` writes (lookup-mode).

    Stale-ALU residue cancel band (``emit_stale_cancel_band=True``): on
    COLLAPSED IMM+OP steps the ``operand_a_*+0`` and ``operand_b_*+0``
    channels carry ~1.04 stale residual from the prior sub-cycle. A plain
    3-way AND at threshold 80 would fire on stale (1.04 vs the legit 1.0
    one-hot) and write spurious mass. For each (legit-operand, stale-+0)
    pair the generator emits one negative-write cancel unit gated on an
    asymmetric-weight stale detector: weight ``stale_operand_weight`` on the
    ``+0`` dim with threshold ``marker + stale_operand_weight *
    stale_residual_factor + other_operand`` fires only when ``+0 >= 1.04``.
    Two sub-bands per nibble, appended after that nibble's 256 main rules:
      * stale-A0 (b in 0..15): cancels ``op_fn(0, b)``.
      * stale-B0 (a in 1..15): cancels ``op_fn(a, 0)`` (a=0 skipped — the
        stale-A0 b=0 unit already targets ``op_fn(0,0)``).
    See ``_layer10_alu_bitwise_rules`` and
    ``tests/test_l9_collapsed_imm_input_isolated.py``.

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
        operand_a_cond_weight: per-cell condition weight for operand A
            (default 30.0 preserves byte-identity). The L10 MARK_AX install
            reads a non-unit operand one-hot (~5.82 after the operand
            cleanup stage), so it rescales this to ``30/5.82`` — mirroring
            ``wide_div_rules_ge_format``'s ``dividend_cond_weight``.
        operand_b_cond_weight: per-cell condition weight for operand B
            (default 30.0). The cleaned AX_CARRY one-hot is ~0.94, so the
            install passes ``30/0.94``.
        marker_cond_weight: marker condition weight (default 40.0).
        threshold: 3-way AND threshold (default 80.0). With the rescaled
            cond weights a match contributes ~30 each, so the install keeps
            the default-equivalent ``40 + 30 + 30 = 100 > 80`` math.
        result_write_value: numerator of the output write weight
            (default 2.0, lowered as ``result_write_value / S``).
        name_prefix: rule-name prefix (default ``"bitwise"``; the L10
            main-FFN caller passes ``"l10_bitwise"``).
        name_suffix: rule-name suffix appended verbatim (default ``""``;
            the L10 main-FFN caller passes ``"_step_end"``).
        gate_weight: multiplicative gate weight (default 1.0).
        emit_stale_cancel_band: when True, emit the interleaved stale-ALU
            residue cancel band (see above). Default False preserves the
            512-rule lookup-mode post-op shape.
        stale_operand_weight: asymmetric-weight stale detector weight on the
            ``+0`` operand dim (default 1000.0). Only used when
            ``emit_stale_cancel_band`` is True.
        stale_residual_factor: stale residual magnitude the threshold is
            tuned against (default 1.02). Only used when
            ``emit_stale_cancel_band`` is True.

    Returns:
        ``tuple[FFNRule, ...]`` of length 512 (or 574 with the cancel band)
        — same shape as ``_layer10_alu_bitwise_rules(S, op_name=op.upper(),
        op_fn=op_fn)``.

    Raises:
        ValueError: if ``op`` is not one of ``"and"``, ``"or"``, ``"xor"``.
    """
    op_fn = _BITWISE_OP_FN.get(op)
    if op_fn is None:
        raise ValueError(
            f"bitwise_rules: op must be 'and'/'or'/'xor'; got {op!r}"
        )

    write_value = result_write_value / S
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
                        f"{name_prefix}_{op}_{nibble_label}_"
                        f"a{a:x}_b{b:x}{name_suffix}"
                    ),
                    conditions=(
                        (marker_gate, marker_cond_weight),
                        (f"{a_band}+{a}", operand_a_cond_weight),
                        (f"{b_band}+{b}", operand_b_cond_weight),
                    ),
                    threshold=threshold,
                    gate=opcode_gate,
                    gate_weight=gate_weight,
                    writes=((f"{out_band}+{result}", write_value),),
                ))
        if emit_stale_cancel_band:
            # Stale operand-A+0 detector: cancels ``op_fn(0, b)`` when the
            # ``+0`` A channel is stale (>= stale_residual_factor) with a
            # legit operand-B one-hot.
            for b in range(16):
                spurious = op_fn(0, b)
                rules.append(multi_way_and_rule(
                    name=(
                        f"{name_prefix}_{op}_{nibble_label}_"
                        f"cancel_stale_alu0_b{b:x}{name_suffix}"
                    ),
                    conditions=(
                        (marker_gate, marker_cond_weight),
                        (f"{a_band}+0", stale_operand_weight),
                        (f"{b_band}+{b}", operand_b_cond_weight),
                    ),
                    threshold=(
                        marker_cond_weight
                        + stale_operand_weight * stale_residual_factor
                        + operand_b_cond_weight
                    ),
                    gate=opcode_gate,
                    gate_weight=gate_weight,
                    writes=((f"{out_band}+{spurious}", -write_value),),
                ))
            # Stale operand-B+0 detector: mirror for operand B. Skip a=0 to
            # avoid a duplicate cancel at ``op_fn(0, 0)`` (the stale-A0 b=0
            # unit above already targets that cell).
            for a in range(1, 16):
                spurious = op_fn(a, 0)
                rules.append(multi_way_and_rule(
                    name=(
                        f"{name_prefix}_{op}_{nibble_label}_"
                        f"cancel_stale_a{a:x}_carry0{name_suffix}"
                    ),
                    conditions=(
                        (marker_gate, marker_cond_weight),
                        (f"{a_band}+{a}", operand_a_cond_weight),
                        (f"{b_band}+0", stale_operand_weight),
                    ),
                    threshold=(
                        marker_cond_weight
                        + operand_a_cond_weight
                        + stale_operand_weight * stale_residual_factor
                    ),
                    gate=opcode_gate,
                    gate_weight=gate_weight,
                    writes=((f"{out_band}+{spurious}", -write_value),),
                ))
    return tuple(rules)


# ---------------------------------------------------------------------------
# Byte-0 nibble ADD/SUB — the LOOKUP-path L8/L9 compute, from a compact spec.
# ---------------------------------------------------------------------------
#
# The production golden path is ``alu_mode="lookup"`` (see
# ``docs/semantic_spec_ALU.md`` §0). Under it, byte-0 ADD/SUB is computed by a
# TWO-BLOCK cascade of computed nibble lookups:
#
#   * L8 (``l8_ops._layer8_alu_*``): the byte-0 LOW nibble (``ALU_LO`` A,
#     ``AX_CARRY_LO`` B -> ``OUTPUT_LO``) plus the inter-nibble carry/borrow
#     GENERATION into ``CARRY+0``. Marker = ``MARK_SE_ONLY`` (mirrored to
#     ``MARK_AX`` by ``_l8_alu_add_mark_ax_mirror``), ``MARK_PC`` blocker -4.
#   * L9 (``l9_ops._*_hi_nibble_rules`` / ``_*_carry_out_rules``): the byte-0
#     HIGH nibble (``ALU_HI`` A, ``AX_CARRY_HI`` B -> ``OUTPUT_HI_THIS_STEP``)
#     which READS the ``CARRY+0`` carry/borrow-in generated by L8, plus the
#     byte-LEVEL carry/borrow OUT into ``CARRY+1`` (ADD) / ``CARRY+2`` (SUB)
#     the downstream L10 ``CarryPropagationPostOp`` consumes. Marker =
#     ``MARK_AX``, ``MARK_PC`` blocker -2.
#
# This is EXACTLY the `for a: for b: result = f(a,b,cin)` computed-lookup
# derivable regime the ALU spec (§2, §G1/§G2) describes. Every one of the 8
# hand-authored builders is this ONE loop with a different `(op, bands,
# marker, weights, thresholds, carry_rule)` DATA tuple -- no per-value logic.
# :func:`nibble_alu_lane_rules` is that single generator; the L8/L9 op
# factories carry only the compact per-lane spec and call it. Byte-identity
# vs the prior hand-authored builders is proven by
# ``tools/verify_l8l9_addsub_derived.py`` (and the golden hash).
#
# NOTE: this is a SEPARATE generator from :func:`wide_add_rules` /
# :func:`wide_sub_rules`. Those target the ``efficient``-mode
# ``DeclarativeAddSubBlock`` (a single-band ``b*16+k`` nibble-stacked layout
# with 40/30/30 balanced-AND weights); the LOOKUP path uses the distinct
# separate-LO/HI-band, unit-weight, ``MARK_PC``-blocked, two-block-cascade
# shape reproduced here. They are two lowerings of the same arithmetic.


def nibble_alu_lane_rules(
    *,
    op: Literal["add", "sub"],
    emit: Literal["result", "carry_flag"],
    operand_a_band: str,
    operand_b_band: str,
    marker_gate: str,
    marker_weight: float,
    mark_pc_weight: float,
    gate: str,
    threshold_no_carry: float,
    result_band: str | None = None,
    write_scale: float = 0.0,
    carry_flag_dim: str | None = None,
    carry_flag_scale: float = 0.0,
    carry_in_dim: str | None = None,
    carry_in_weight: float = 0.0,
    threshold_with_carry: float | None = None,
    extra_conditions: Tuple[Tuple[str, float], ...] = (),
    name_fn: Callable[[int, int, int], str] | None = None,
) -> Tuple[FFNRule, ...]:
    """Emit the byte-0 nibble ADD/SUB lane rules from a compact spec.

    ONE generator for all 8 hand-authored L8/L9 add/sub builders. Each is a
    ``for carry_in: for a: for b:`` loop over the 16x16 nibble cross-product
    that evaluates ``f(a, b, carry_in)`` at BUILD TIME and emits one
    ``multi_way_and_rule`` per surviving combination. The DATA that
    distinguishes the 8 callers is entirely in the kwargs -- there is zero
    per-value hand-tuning.

    Two emission modes:

      * ``emit="result"``: write the result nibble
        ``f(a,b,cin) % 16`` to ``result_band`` at ``write_scale``. Emits one
        rule for EVERY ``(carry_in, a, b)`` (the L8 lo / L9 hi lanes).
      * ``emit="carry_flag"``: write a fixed ``carry_flag_dim`` at
        ``carry_flag_scale`` ONLY for combinations that produce a carry-out
        (ADD: ``a+b+cin >= 16``) / borrow-out (SUB: ``a-b-cin < 0``). Emits
        one rule per SURVIVING combination (the L8 carry / L9 carry-out
        lanes).

    Carry-IN handling:

      * ``carry_in_dim is None``: byte-0 LOW-nibble lanes (L8). Only
        ``carry_in = 0`` is iterated -- there is no carry into byte 0's low
        nibble. Threshold is ``threshold_no_carry``.
      * ``carry_in_dim`` set: HIGH-nibble lanes (L9) that read the
        L8-generated ``CARRY+0``. Both ``carry_in in (0, 1)`` are iterated;
        ``carry_in=0`` adds ``(carry_in_dim, -carry_in_weight)`` (repel) at
        ``threshold_no_carry``, ``carry_in=1`` adds
        ``(carry_in_dim, +carry_in_weight)`` (require) at
        ``threshold_with_carry``.

    Args:
        op: ``"add"`` (``result=(a+b+cin)%16``, carry when ``a+b+cin>=16``)
            or ``"sub"`` (``result=(a-b-cin)%16``, borrow when
            ``a-b-cin<0``).
        emit: ``"result"`` or ``"carry_flag"`` (see above).
        operand_a_band / operand_b_band: per-nibble one-hot bands for A / B
            (read at ``+a`` / ``+b``).
        marker_gate / marker_weight: the AX-style marker condition.
        mark_pc_weight: the ``MARK_PC`` NOT-blocker weight (negative).
        gate: the opcode-flag gate (``OP_ADD`` / ``OP_SUB``).
        threshold_no_carry: threshold for the carry_in=0 (and low-nibble)
            rules.
        result_band / write_scale: result-nibble write (``emit="result"``).
        carry_flag_dim / carry_flag_scale: fixed carry/borrow flag write
            (``emit="carry_flag"``).
        carry_in_dim / carry_in_weight / threshold_with_carry: high-nibble
            carry-in discrimination (see above).
        extra_conditions: extra fixed condition terms appended verbatim
            (e.g. ``()``; reserved for future blockers).
        name_fn: ``(carry_in, a, b) -> name``. Required (the byte-identity
            rule names encode the sub-stage + indices).

    Returns:
        ``tuple[FFNRule, ...]`` in the SAME order as the hand-authored loops
        (outer ``carry_in``, then ``a``, then ``b``).
    """
    if op not in ("add", "sub"):
        raise ValueError(f"nibble_alu_lane_rules: op must be add/sub; got {op!r}")
    if emit not in ("result", "carry_flag"):
        raise ValueError(
            f"nibble_alu_lane_rules: emit must be result/carry_flag; got {emit!r}"
        )
    if emit == "result" and result_band is None:
        raise ValueError("nibble_alu_lane_rules: emit='result' needs result_band")
    if emit == "carry_flag" and carry_flag_dim is None:
        raise ValueError(
            "nibble_alu_lane_rules: emit='carry_flag' needs carry_flag_dim"
        )
    if name_fn is None:
        raise ValueError("nibble_alu_lane_rules: name_fn is required")

    carry_in_cases = (0,) if carry_in_dim is None else (0, 1)

    rules: list[FFNRule] = []
    for carry_in in carry_in_cases:
        for a in range(16):
            for b in range(16):
                if op == "add":
                    total = a + b + carry_in
                    result = total % 16
                    overflow = total >= 16
                else:
                    raw = a - b - carry_in
                    result = raw % 16
                    overflow = raw < 0

                if emit == "carry_flag" and not overflow:
                    continue

                conditions: list[Tuple[str, float]] = [
                    (marker_gate, marker_weight),
                    ("MARK_PC", mark_pc_weight),
                    (f"{operand_a_band}+{a}", 1.0),
                    (f"{operand_b_band}+{b}", 1.0),
                ]
                conditions.extend(extra_conditions)
                if carry_in_dim is not None:
                    if carry_in == 0:
                        conditions.append((carry_in_dim, -carry_in_weight))
                        threshold = threshold_no_carry
                    else:
                        conditions.append((carry_in_dim, carry_in_weight))
                        threshold = threshold_with_carry
                else:
                    threshold = threshold_no_carry

                if emit == "result":
                    writes = ((f"{result_band}+{result}", write_scale),)
                else:
                    writes = ((carry_flag_dim, carry_flag_scale),)

                rules.append(multi_way_and_rule(
                    name=name_fn(carry_in, a, b),
                    conditions=tuple(conditions),
                    threshold=threshold,
                    gate=gate,
                    writes=writes,
                ))
    return tuple(rules)


# ---------------------------------------------------------------------------
# Amplified nibble adder lane (L9 LEA / ADJ / ENT hi-nibble, live operand-B).
# ---------------------------------------------------------------------------
#
# The L9 LEA / ADJ / ENT hi-nibble bands are the SAME amplified 16x16 nibble
# adder: operand-A is a per-nibble one-hot band (``ALU_HI``) and operand-B is a
# LIVE OPERAND BAND (``FETCH_HI`` — the instruction immediate's high nibble),
# summed with a carry/borrow-IN read from ``CARRY+0`` into
# ``OUTPUT_HI_THIS_STEP`` at ``2.0/S``. Unlike :func:`nibble_alu_lane_rules`
# (the unit-weight ``MARK_PC``-blocked ADD/SUB lane), this shape uses the
# AMPLIFIED marker AND: ``marker(+20)`` + the seven non-AX blocker dims at
# ``-1000`` each (so PC/SP/BP/STACK0/MEM/SE/byte rows never fire even when the
# FETCH gate is high) + ``ALU_HI[a](+1)`` + operand-B ``[b](+20)`` + the
# ``CARRY+0`` carry-in discrimination (``+/-8``). It is THE shared ALU adder the
# CONTROL/MEMORY families reuse for a ``reg + live-operand`` update — the same
# generator LEA (``imm`` into an address), ADJ (``SP += imm``), and ENT
# (``SP -= imm``) all call, differing only by the DATA tuple (opcode gate,
# marker, thresholds, add-vs-subtract, an optional extra blocker, rule names).
#
# ``op="sub"`` mirrors ADD with ``result = (a - b - carry_in) % 16`` (ENT's
# ``SP - imm`` borrow adder). Byte-identity vs the prior hand-authored LEA/ADJ/
# ENT loops is proven by the whole-model golden hash.


def amplified_nibble_adder_rules(
    *,
    op: Literal["add", "sub"],
    operand_a_band: str,
    operand_b_band: str,
    marker_gate: str,
    blocker_dims: Sequence[str],
    blocker_weight: float,
    gate: str,
    carry_in_dim: str,
    carry_in_weight: float,
    threshold_no_carry: float,
    threshold_with_carry: float,
    result_band: str,
    write_scale: float,
    name_fn: Callable[[int, int, int], str],
    marker_weight: float = 20.0,
    operand_a_weight: float = 1.0,
    operand_b_weight: float = 20.0,
    extra_conditions: Tuple[Tuple[str, float], ...] = (),
) -> Tuple[FFNRule, ...]:
    """Emit the L9 amplified ``reg + live-operand`` nibble-adder lane rules.

    ONE generator for the L9 LEA / ADJ / ENT hi-nibble bands. Each is a
    ``for carry_in in (0, 1): for a: for b:`` walk over the 16x16 nibble
    cross-product that evaluates ``f(a, b, carry_in)`` at BUILD TIME and emits
    one ``multi_way_and_rule`` per combination. The distinguishing DATA is
    entirely in the kwargs; there is zero per-value hand-tuning.

    Each emitted rule is the amplified marker AND: ``(marker_gate,
    marker_weight)`` + one ``(dim, -blocker_weight)`` per ``blocker_dims`` entry
    (the non-AX marker/byte blockers) + ``extra_conditions`` +
    ``(operand_a_band+a, operand_a_weight)`` + ``(operand_b_band+b,
    operand_b_weight)``, plus the carry-in discrimination on ``carry_in_dim``:

      * ``carry_in=0``: append ``(carry_in_dim, -carry_in_weight)`` (repel) at
        ``threshold_no_carry``.
      * ``carry_in=1``: append ``(carry_in_dim, +carry_in_weight)`` (require) at
        ``threshold_with_carry``.

    gated on ``gate`` (the opcode flag), writing ``(result_band+result,
    write_scale)`` where ``result = (a + b + carry_in) % 16`` for ``op="add"``
    or ``(a - b - carry_in) % 16`` for ``op="sub"``.

    Args:
        op: ``"add"`` (``result=(a+b+cin)%16``, LEA/ADJ) or ``"sub"``
            (``result=(a-b-cin)%16``, ENT's ``SP - imm`` borrow adder).
        operand_a_band / operand_b_band: per-nibble one-hot bands for A / B.
            operand-B is the LIVE operand (``FETCH_HI``, the immediate).
        marker_gate / marker_weight: the amplified marker condition (``+20``).
        blocker_dims / blocker_weight: the non-AX marker/byte blockers, each
            appended as ``(dim, -blocker_weight)`` (``-1000``).
        gate: the opcode-flag gate (``OP_LEA`` / ``OP_ADJ`` / ``OP_ENT``).
        carry_in_dim / carry_in_weight: the ``CARRY+0`` carry/borrow-in dim +
            its discrimination weight (``+/-8``).
        threshold_no_carry / threshold_with_carry: the carry_in=0 / carry_in=1
            AND thresholds (the explicit values the ``-1000`` blockers require).
        result_band / write_scale: the result-nibble write.
        name_fn: ``(carry_in, a, b) -> name`` (the byte-identity rule names).
        operand_a_weight / operand_b_weight: per-cell operand condition weights
            (default 1.0 / 20.0).
        extra_conditions: extra fixed condition terms appended verbatim, in the
            SAME position the hand-authored loops used (right after the blocker
            band) — e.g. ENT's ``(HAS_SE, -1000)`` first-step blocker.

    Returns:
        ``tuple[FFNRule, ...]`` in the SAME order as the hand-authored loops
        (outer ``carry_in``, then ``a``, then ``b``) — 512 rules.
    """
    if op not in ("add", "sub"):
        raise ValueError(
            f"amplified_nibble_adder_rules: op must be add/sub; got {op!r}"
        )
    rules: list[FFNRule] = []
    for carry_in in (0, 1):
        for a in range(16):
            for b in range(16):
                if op == "add":
                    result = (a + b + carry_in) % 16
                else:
                    result = (a - b - carry_in) % 16
                conditions: list[Tuple[str, float]] = [
                    (marker_gate, marker_weight),
                ]
                conditions.extend(
                    (dim, -blocker_weight) for dim in blocker_dims
                )
                conditions.extend(extra_conditions)
                conditions.append((f"{operand_a_band}+{a}", operand_a_weight))
                conditions.append((f"{operand_b_band}+{b}", operand_b_weight))
                if carry_in == 0:
                    conditions.append((carry_in_dim, -carry_in_weight))
                    threshold = threshold_no_carry
                else:
                    conditions.append((carry_in_dim, carry_in_weight))
                    threshold = threshold_with_carry
                rules.append(multi_way_and_rule(
                    name=name_fn(carry_in, a, b),
                    conditions=tuple(conditions),
                    threshold=threshold,
                    gate=gate,
                    writes=((f"{result_band}+{result}", write_scale),),
                ))
    return tuple(rules)


# ---------------------------------------------------------------------------
# Nibble comparator lane (L9 CMP condition-flag combine).
# ---------------------------------------------------------------------------
#
# The L9 CMP combine is a NIBBLE COMPARATOR: for a pair of one-hot operand
# nibbles ``a`` (operand-A band) and ``b`` (operand-B band), one hidden unit
# per surviving ``(a, b)`` combination fires a marker-gated N-way AND and
# writes a fixed condition-flag dim. Two comparator modes span all four L9
# CMP lanes (hi_eq / lo_eq / hi_lt / lo_lt):
#
#   * ``mode="equal"``    -> one rule per ``a == b`` pair (the 16 diagonal
#     ``a == k AND b == k`` combinations), writing the equality-flag dim.
#   * ``mode="less_than"`` -> one rule per ``a < b`` pair (the 120 upper-
#     triangular combinations), writing the less-than-flag dim.
#
# This is the comparator sibling of :func:`nibble_alu_lane_rules` (which
# covers the ADD/SUB arithmetic lanes). Both share the same lowering path
# (``multi_way_and_rule`` -> ``Primitives.lower_ffn_rules``); the DATA that
# distinguishes each caller is entirely in the kwargs — zero per-value
# hand-tuning. Byte-identity vs the prior hand-authored CMP loops is proven
# by ``tools/verify_l9_cmp_derived.py`` and the whole-model golden hash.
#
# NOTE on the MARK_AX-vs-MARK_SE_ONLY wall: the memory note
# ``project_wave_b_cmp_needs_l9_internal_relay.md`` records that the L9 CMP
# operands live at MARK_AX, but the CMP combine has to gate at
# MARK_SE_ONLY. That wall was already resolved by the
# ``layer9_step_end_operand_relay`` attention head (Wave A v2), which mirrors
# the raw operand bands into the ``SE_``-tagged mirror dims at the STEP_END
# row. This comparator generator therefore reads whatever operand bands the
# caller supplies (the SE_-tagged mirrors in production) and gates at the
# caller's marker; it does NOT re-introduce the wall.


def nibble_compare_lane_rules(
    *,
    mode: Literal["equal", "less_than"],
    operand_a_band: str,
    operand_b_band: str,
    marker_gate: str,
    marker_weight: float,
    mark_pc_weight: float,
    gate: str,
    threshold: float,
    flag_dim: str,
    flag_scale: float,
    name_fn: Callable[[int, int], str],
) -> Tuple[FFNRule, ...]:
    """Emit the L9 CMP nibble-comparator lane rules from a compact spec.

    ONE generator for the four hand-authored L9 CMP builders (hi_eq, lo_eq,
    hi_lt, lo_lt). Each is a ``for a: for b:`` walk over the 16x16 nibble
    cross-product that keeps only the combinations satisfying the comparator
    predicate and emits one ``multi_way_and_rule`` per survivor. The DATA
    that distinguishes the four callers is entirely in the kwargs.

    Each emitted rule is a 4-way AND at the marker row:
    ``(marker_gate, marker_weight)`` + ``(MARK_PC, mark_pc_weight)`` NOT-blocker
    + ``(operand_a_band+a, 1.0)`` + ``(operand_b_band+b, 1.0)``, gated on
    ``gate`` (the CMP-group opcode flag) at the explicit ``threshold``, writing
    ``(flag_dim, flag_scale)``.

    Two comparator modes:

      * ``mode="equal"``: emit one rule per ``a == b`` pair. The a/b bands are
        read at the SAME index ``k`` (the diagonal ``a == k AND b == k``
        detector), so the walk collapses to ``for k in range(16)`` — 16 rules.
      * ``mode="less_than"``: emit one rule per ``a < b`` pair (``b`` ranges
        over ``a+1..15``) — 120 rules.

    Args:
        mode: ``"equal"`` (16 diagonal rules) or ``"less_than"`` (120 upper-
            triangular rules).
        operand_a_band / operand_b_band: per-nibble one-hot bands for A / B.
        marker_gate / marker_weight: the STEP_END-style marker condition.
        mark_pc_weight: the ``MARK_PC`` NOT-blocker weight (negative).
        gate: the CMP-group opcode-flag gate.
        threshold: explicit AND threshold (the ``MARK_PC`` negative blocker
            rules out the default positive-weight derivation).
        flag_dim / flag_scale: the condition-flag dim written per firing.
        name_fn: ``(a, b) -> name``. For ``mode="equal"`` it is called with
            ``(k, k)``; the byte-identity rule names encode the sub-stage +
            index.

    Returns:
        ``tuple[FFNRule, ...]`` in the SAME order as the hand-authored loops.
    """
    if mode not in ("equal", "less_than"):
        raise ValueError(
            f"nibble_compare_lane_rules: mode must be equal/less_than; got {mode!r}"
        )

    if mode == "equal":
        pairs = ((k, k) for k in range(16))
    else:
        pairs = ((a, b) for a in range(16) for b in range(a + 1, 16))

    rules: list[FFNRule] = []
    for a, b in pairs:
        rules.append(multi_way_and_rule(
            name=name_fn(a, b),
            conditions=(
                (marker_gate, marker_weight),
                ("MARK_PC", mark_pc_weight),
                (f"{operand_a_band}+{a}", 1.0),
                (f"{operand_b_band}+{b}", 1.0),
            ),
            threshold=threshold,
            gate=gate,
            writes=((flag_dim, flag_scale),),
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
    operand_a_cond_weight: float = 30.0,
    operand_b_cond_weight: float = 30.0,
    marker_cond_weight: float = 40.0,
    threshold: float = 80.0,
    final_carry_dim: str | None = None,
    operand_a_artifact_blocker_weight: float = 0.0,
    result_write_amplitude: float | None = None,
    carry_signal_weight: float | None = None,
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
    write_amplitude = (
        2.0 / S if result_write_amplitude is None
        else result_write_amplitude / S
    )
    # The inter-byte carry cascade dim is written at a CONTROLLED residual
    # amplitude (~2.0) regardless of the dominant result amplitude, so the
    # next byte's carry-read math sees a known carry signal. (Writing the
    # carry at the dominant result amplitude would make the carry-read term
    # swamp the operand discrimination.)
    carry_write_amplitude = 2.0 / S
    if carry_signal_weight is None:
        # Legacy operand-relative carry discrimination. With defaults
        # (a=b=30, marker=40, thr=80) this reproduces the historical
        # -50 / +30 / thr 120 cascade exactly. Carry written at the result
        # amplitude (back-compat: the historical single-amplitude behavior).
        carry_write_amplitude = write_amplitude
        carry_suppress_w = -(marker_cond_weight + operand_a_cond_weight
                             + operand_b_cond_weight - threshold + 30.0)
        carry_require_w = operand_a_cond_weight
        carry_threshold = threshold + carry_require_w + 10.0
    else:
        # Absolute carry discrimination decoupled from operand weights. The
        # carry cascade dim is written by the prior byte at residual ~6.0
        # (the lowering's saturated output for a 2.0/S write_amplitude AND
        # rule -- measured, lowering-stable). Choose ``carry_signal_weight``
        # so ``csw * 6.0`` equals roughly ONE operand discriminator's worth,
        # so a present carry acts like an extra required operand:
        #   cin=0 fires iff carry absent: M > T (C=0); M - csw*6 < T (C=6).
        #   cin=1 fires iff carry present AND operands match:
        #     C=0 -> M < T+csw*6 (dies); C=6, no-b -> M+csw*6 < T+csw*6
        #     (dies); C=6, full -> M+csw*6 > T+csw*6 (fires).
        csw = carry_signal_weight
        carry_residual = 6.0
        carry_suppress_w = -csw
        carry_require_w = csw
        carry_threshold = threshold + csw * carry_residual
    blk = operand_a_artifact_blocker_weight

    for b in range(width_bytes):
        a_band = operand_a_base
        b_band = operand_b_base
        is_last = (b == width_bytes - 1)
        # Where this byte's carry-out lands. The final byte can be
        # redirected (e.g. SUB borrow -> CARRY+2) via ``final_carry_dim``.
        if is_last and final_carry_dim is not None:
            carry_out_dim = final_carry_dim
        else:
            carry_out_dim = f"{carry_base}+{b}"
        # carry_in possibilities for this byte
        if b == 0:
            carry_in_cases = (0,)
        else:
            carry_in_cases = (0, 1)

        for carry_in in carry_in_cases:
            for a_nib in range(16):
                # Index-0 artifact blocker: negative weight on every OTHER
                # non-zero operand-A nibble cell in this lane. When A's
                # true nibble is non-zero its strong one-hot trips these,
                # suppressing the spurious a_nib=0 rule that the gather's
                # value-proportional index-0 artifact would otherwise fire.
                blocker_terms = ()
                if blk > 0.0:
                    blocker_terms = tuple(
                        (f"{a_band}+{b * 16 + j}", -blk)
                        for j in range(1, 16)
                        if j != a_nib
                    )
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
                        # Carry dims (both the inter-byte cascade and the
                        # final overflow consumed downstream) use the
                        # controlled carry amplitude, not the dominant result
                        # amplitude.
                        writes.append((carry_out_dim, carry_write_amplitude))

                    if b == 0:
                        # No carry-in dim. Standard 3-way AND.
                        conditions = (
                            (marker_gate, marker_cond_weight),
                            (f"{a_band}+{b * 16 + a_nib}",
                             operand_a_cond_weight),
                            (f"{b_band}+{b * 16 + b_nib}",
                             operand_b_cond_weight),
                        ) + blocker_terms
                        rule_threshold = threshold
                    elif carry_in == 0:
                        # Suppress when carry-in dim is active.
                        conditions = (
                            (marker_gate, marker_cond_weight),
                            (f"{a_band}+{b * 16 + a_nib}",
                             operand_a_cond_weight),
                            (f"{b_band}+{b * 16 + b_nib}",
                             operand_b_cond_weight),
                            (f"{carry_base}+{b - 1}", carry_suppress_w),
                        ) + blocker_terms
                        rule_threshold = threshold
                    else:
                        # carry_in == 1: require carry dim positively.
                        conditions = (
                            (marker_gate, marker_cond_weight),
                            (f"{a_band}+{b * 16 + a_nib}",
                             operand_a_cond_weight),
                            (f"{b_band}+{b * 16 + b_nib}",
                             operand_b_cond_weight),
                            (f"{carry_base}+{b - 1}", carry_require_w),
                        ) + blocker_terms
                        rule_threshold = carry_threshold

                    rules.append(multi_way_and_rule(
                        name=(
                            f"wide_add_b{b}_cin{carry_in}_"
                            f"a{a_nib:x}_b{b_nib:x}"
                        ),
                        conditions=conditions,
                        threshold=rule_threshold,
                        gate=opcode_gate,
                        writes=tuple(writes),
                    ))

        # Single carry-in detection / relay rule for byte > 0 (no-op
        # observable; self-relay keeps the cascade bit-stable).
        if b > 0:
            rules.append(multi_way_and_rule(
                name=f"wide_add_b{b}_carry_in_detect",
                conditions=(
                    (marker_gate, marker_cond_weight),
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
    operand_a_cond_weight: float = 30.0,
    operand_b_cond_weight: float = 30.0,
    marker_cond_weight: float = 40.0,
    threshold: float = 80.0,
    final_borrow_dim: str | None = None,
    operand_a_artifact_blocker_weight: float = 0.0,
    result_write_amplitude: float | None = None,
    carry_signal_weight: float | None = None,
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
    write_amplitude = (
        2.0 / S if result_write_amplitude is None
        else result_write_amplitude / S
    )
    # See wide_add_rules: carry/borrow cascade dims use a controlled
    # amplitude so the next byte's borrow-read math sees a known signal.
    borrow_write_amplitude = 2.0 / S
    if carry_signal_weight is None:
        # Legacy operand-relative discrimination (reproduces -50/+30/thr120).
        borrow_write_amplitude = write_amplitude
        borrow_suppress_w = -(marker_cond_weight + operand_a_cond_weight
                              + operand_b_cond_weight - threshold + 30.0)
        borrow_require_w = operand_a_cond_weight
        borrow_threshold = threshold + borrow_require_w + 10.0
    else:
        # Absolute discrimination decoupled from operand weights (see
        # wide_add_rules: carry cascade residual ~6.0, the lowering-stable
        # saturated AND output for a 2.0/S write).
        csw = carry_signal_weight
        borrow_residual = 6.0
        borrow_suppress_w = -csw
        borrow_require_w = csw
        borrow_threshold = threshold + csw * borrow_residual
    blk = operand_a_artifact_blocker_weight

    for b in range(width_bytes):
        a_band = operand_a_base
        b_band = operand_b_base
        is_last = (b == width_bytes - 1)
        # Where this byte's borrow-out lands. The final byte can be
        # redirected (e.g. byte-0 SUB borrow -> CARRY+2) via
        # ``final_borrow_dim``.
        if is_last and final_borrow_dim is not None:
            borrow_out_dim = final_borrow_dim
        else:
            borrow_out_dim = f"{borrow_base}+{b}"
        # borrow_in possibilities for this byte
        if b == 0:
            borrow_in_cases = (0,)
        else:
            borrow_in_cases = (0, 1)

        for borrow_in in borrow_in_cases:
            for a_nib in range(16):
                # Index-0 artifact blocker (see wide_add_rules): negative
                # weight on every OTHER non-zero operand-A cell in this lane.
                blocker_terms = ()
                if blk > 0.0:
                    blocker_terms = tuple(
                        (f"{a_band}+{b * 16 + j}", -blk)
                        for j in range(1, 16)
                        if j != a_nib
                    )
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
                            (borrow_out_dim, borrow_write_amplitude)
                        )

                    if b == 0:
                        # No borrow-in dim. Standard 3-way AND.
                        conditions = (
                            (marker_gate, marker_cond_weight),
                            (f"{a_band}+{b * 16 + a_nib}",
                             operand_a_cond_weight),
                            (f"{b_band}+{b * 16 + b_nib}",
                             operand_b_cond_weight),
                        ) + blocker_terms
                        rule_threshold = threshold
                    elif borrow_in == 0:
                        # Suppress when borrow-in dim is active.
                        conditions = (
                            (marker_gate, marker_cond_weight),
                            (f"{a_band}+{b * 16 + a_nib}",
                             operand_a_cond_weight),
                            (f"{b_band}+{b * 16 + b_nib}",
                             operand_b_cond_weight),
                            (f"{borrow_base}+{b - 1}", borrow_suppress_w),
                        ) + blocker_terms
                        rule_threshold = threshold
                    else:
                        # borrow_in == 1: require borrow dim positively.
                        conditions = (
                            (marker_gate, marker_cond_weight),
                            (f"{a_band}+{b * 16 + a_nib}",
                             operand_a_cond_weight),
                            (f"{b_band}+{b * 16 + b_nib}",
                             operand_b_cond_weight),
                            (f"{borrow_base}+{b - 1}", borrow_require_w),
                        ) + blocker_terms
                        rule_threshold = borrow_threshold

                    rules.append(multi_way_and_rule(
                        name=(
                            f"wide_sub_b{b}_bin{borrow_in}_"
                            f"a{a_nib:x}_b{b_nib:x}"
                        ),
                        conditions=conditions,
                        threshold=rule_threshold,
                        gate=opcode_gate,
                        writes=tuple(writes),
                    ))

        # Single borrow-in detection / relay rule for byte > 0 (no-op
        # observable; self-relay keeps the cascade bit-stable).
        if b > 0:
            rules.append(multi_way_and_rule(
                name=f"wide_sub_b{b}_borrow_in_detect",
                conditions=(
                    (marker_gate, marker_cond_weight),
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
# Nibble fused multiply-accumulate combine (L11/L12 MUL nibble-combine lane).
# ---------------------------------------------------------------------------
#
# The DEFAULT (non-multipass) byte-0 MUL path stages its result nibbles as a
# pair of triple-loop nibble-COMBINE banks — the L11 ``mul_partial`` and the
# L12 ``mul_combine`` FFNs. BOTH are the identical shape: a ``for c: for a:
# for b:`` walk over three 16-wide one-hot bands that evaluates a fused
# multiply-accumulate ``result = (accum(c, a, b) + a * b) % 16`` at BUILD TIME
# and emits one marker-gated 4-way AND per (c, a, b) triple, writing a single
# result-nibble one-hot gated on ``OP_MUL``. There is ZERO per-value
# hand-tuning; the DATA that distinguishes L11 from L12 is entirely the
# ``(bands, accum_fn, weights, threshold, write_scale, marker)`` tuple.
#
# ``nibble_fused_madd_combine_rules`` is that single generator (the
# schoolbook-combine sibling of :func:`nibble_alu_lane_rules`). It lets the
# L11/L12 op factories carry only the compact per-lane spec instead of an
# inline hand-authored triple-loop, so the shape is deduplicated and the
# COVERAGE-first "derive every op" contract is met by a SHARED derivation.
# Byte-identity vs the prior inline loops is proven by the whole-model golden
# hash (``tools/_isa_golden_hash.py``) and the DSL unit test.


def nibble_fused_madd_combine_rules(
    *,
    accum_band: str,
    factor_a_band: str,
    factor_b_band: str,
    result_band: str,
    marker_gate: str,
    gate: str,
    threshold: float,
    write_scale: float,
    name_fn: Callable[[int, int, int], str],
    accum_fn: Callable[[int, int, int], int] = lambda c, a, b: c,
    marker_weight: float = 1.0,
    accum_weight: float = 1.0,
    factor_a_weight: float = 1.0,
    factor_b_weight: float = 1.0,
    outer: Literal["accum", "factor_a"] = "accum",
    scope: Optional[str] = None,
    dominates_at_fn: Optional[Callable[[int], Mapping[str, str]]] = None,
) -> Tuple[FFNRule, ...]:
    """Emit the byte-0 MUL nibble fused-multiply-accumulate combine lane.

    ONE generator for the two DEFAULT-path MUL nibble-combine banks (L11
    ``mul_partial`` and L12 ``mul_combine``). Each is a ``for c: for a: for
    b:`` walk over the ``16 x 16 x 16`` one-hot cross-product that evaluates
    the fused multiply-accumulate ``result = (accum_fn(c, a, b) + a * b) % 16``
    at BUILD TIME and emits one ``multi_way_and_rule`` per triple. The
    distinguishing DATA is entirely in the kwargs — zero per-value hand-tuning.

    Each emitted rule is a marker-gated 4-way AND:
    ``(marker_gate, marker_weight)`` + ``(accum_band+c, accum_weight)`` +
    ``(factor_a_band+a, factor_a_weight)`` + ``(factor_b_band+b,
    factor_b_weight)``, gated on ``gate`` (the ``OP_MUL`` flag) at the explicit
    ``threshold``, writing ``(result_band+result, write_scale)``.

    L12 ``mul_combine`` is the direct instance: ``accum_band=TEMP`` (the
    L11-staged partial),  ``factor_a_band=ALU_HI`` (a_hi), ``factor_b_band=
    AX_CARRY_LO`` (b_lo), ``accum_fn=lambda c, a, b: c`` (the partial is read
    directly), so ``result = (partial + a_hi * b_lo) % 16``.

    L11 ``mul_partial`` is the instance with a DERIVED accumulator:
    ``accum_band=AX_CARRY_LO`` supplies b_lo, ``factor_a_band=ALU_LO`` supplies
    a_lo, ``factor_b_band=AX_CARRY_HI`` supplies b_hi, and ``accum_fn=lambda
    b_lo, a_lo, b_hi: (a_lo * b_lo) // 16`` (the low-nibble carry), so
    ``result = ((a_lo * b_lo) // 16 + a_lo * b_hi) % 16``. Because L11's rule
    ORDER is ``for a_lo: for b_lo: for b_hi`` (a_lo outer, so the lowering
    cursor lands ``a_lo`` slabs contiguously), pass ``outer="factor_a"`` to
    put ``factor_a`` (a_lo) on the outer loop and preserve byte-identity.

    Args:
        accum_band: one-hot band whose index is the ``c`` loop variable (the
            L12 partial read, or the L11 b_lo used to derive the carry).
        factor_a_band / factor_b_band: the two MULTIPLIED one-hot operand
            bands (``a`` and ``b``); the product ``a * b`` is added to the
            accumulator.
        result_band: result-nibble one-hot band; the rule writes
            ``result_band+result``.
        marker_gate / marker_weight: the STEP_END-style marker condition.
        gate: the ``OP_MUL`` opcode-flag gate.
        threshold: explicit 4-way AND threshold (the caller's amplitude
            contract — L11 uses 3.5, L12 uses 7.5).
        write_scale: the result-nibble write weight (already ``value / S``).
        name_fn: ``(c, a, b) -> name`` — the byte-identity rule names encode
            the sub-stage + indices. Called with the loop variables in
            ``(accum_idx, factor_a_idx, factor_b_idx)`` order regardless of
            ``outer``.
        accum_fn: ``(c, a, b) -> accumulator_value`` evaluated at build time.
            Default ``lambda c, a, b: c`` reads the accumulator band directly
            (the L12 case); L11 passes the low-nibble carry derivation.
        accum_weight / factor_a_weight / factor_b_weight: per-cell condition
            weights (default 1.0 — the unit-weight one-hot AND both banks use).
        outer: ``"accum"`` (default; loop nesting ``for c: for a: for b:`` —
            the L12 order) or ``"factor_a"`` (``for a: for c: for b:`` — the
            L11 ``a_lo``-outer order). Controls ONLY the emission order (and
            therefore the lowered hidden-unit indices), not the arithmetic.
        scope: optional shared ``scope`` predicate string threaded verbatim
            to every emitted rule (the L12 ``"MARK_SE_ONLY and OP_MUL"``
            verifier annotation). Weight-neutral (metadata only).
        dominates_at_fn: optional ``result_nibble -> Mapping`` callable that
            builds the per-rule ``dominates_at`` verifier annotation from the
            computed result nibble (the L12 ``{OUTPUT_HI+result_hi: ...}``
            claim). Weight-neutral (metadata only).

    Returns:
        ``tuple[FFNRule, ...]`` of length 4096 (16 x 16 x 16), in the emission
        order the ``outer`` kwarg selects.
    """
    if outer not in ("accum", "factor_a"):
        raise ValueError(
            f"nibble_fused_madd_combine_rules: outer must be "
            f"accum/factor_a; got {outer!r}"
        )

    rules: list[FFNRule] = []

    def emit(c: int, a: int, b: int) -> None:
        result = (accum_fn(c, a, b) + a * b) % 16
        rules.append(multi_way_and_rule(
            name=name_fn(c, a, b),
            conditions=(
                (marker_gate, marker_weight),
                (f"{accum_band}+{c}", accum_weight),
                (f"{factor_a_band}+{a}", factor_a_weight),
                (f"{factor_b_band}+{b}", factor_b_weight),
            ),
            threshold=threshold,
            gate=gate,
            writes=((f"{result_band}+{result}", write_scale),),
            scope=scope,
            dominates_at=(
                dominates_at_fn(result) if dominates_at_fn is not None
                else None
            ),
        ))

    if outer == "accum":
        for c in range(16):
            for a in range(16):
                for b in range(16):
                    emit(c, a, b)
    else:
        for a in range(16):
            for c in range(16):
                for b in range(16):
                    emit(c, a, b)
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
    result_byte1_lo_base: str = None,
    result_byte1_hi_base: str = None,
    operand_a_artifact_blocker_weight: float = 0.0,
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

      Result-HIGH-byte routing (``result_byte1_lo_base`` /
      ``result_byte1_hi_base``): the 16-bit product's two BYTES are
      byte 0 = bits 0..7 (nib0 lo, nib1 hi) and byte 1 = bits 8..15
      (nib2 lo, nib3 hi). By default all four nibble lanes write to a
      single contiguous ``result_base`` (lanes at +0/+16/+32/+48). When
      ``result_byte1_lo_base`` and ``result_byte1_hi_base`` are BOTH
      supplied (width_bytes=2 only), the BYTE-1 lanes are re-routed:
        - byte 0 lo: ``result_base + nib0``
        - byte 0 hi: ``result_base + 16 + nib1``
        - byte 1 lo: ``result_byte1_lo_base + nib2``
        - byte 1 hi: ``result_byte1_hi_base + nib3``
      This is how the L11 install keeps byte 0 in OUTPUT_LO/OUTPUT_HI
      while routing byte 1 to the dedicated MUL_RESULT_HI band (instead
      of ``result_base+32`` = OUTPUT_LO+32 = ADDR_KEY, which would
      corrupt the memory address-key band).

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
        result_byte1_lo_base: optional dim base for the product's BYTE-1
            LOW nibble (nib2). When supplied together with
            ``result_byte1_hi_base`` (width_bytes=2 only), byte 1 is
            routed to a dedicated band instead of ``result_base+32/+48``.
        result_byte1_hi_base: optional dim base for the product's BYTE-1
            HIGH nibble (nib3). See ``result_byte1_lo_base``.
        operand_a_artifact_blocker_weight: width_bytes=2 only. When > 0,
            each rule adds a NEGATIVE condition of weight
            ``-operand_a_artifact_blocker_weight`` on every OTHER non-zero
            operand-A nibble cell (``operand_a_base+j`` for j in 1..15,
            j != a_lo on the low-nibble lane and j != a_hi on the
            high-nibble lane). This disambiguates the value-proportional
            index-0 magnitude artifact the L8 operand-gather emits on
            ``operand_a_base+0`` / ``operand_a_base+16`` (~5.5, NEARLY
            equal to a true ~5.84 nibble): when operand A's true nibble is
            NON-zero, its strong one-hot at the real cell trips this
            blocker and suppresses the spurious ``a_nib=0`` rule that the
            artifact would otherwise fire. When A's true nibble IS zero
            (A < 16, e.g. mul_basic's 6) no other A cell is hot so the
            ``a_nib=0`` rule fires correctly. This is the SAME technique
            ``_layer10_alu_ordering_engine_rules`` uses for the CMP flags
            (see ops/l10_ops.py); it is the documented workaround for the
            ``project_operand_gather_hybrid_encoding`` Wall-1 cell-0
            artifact without an upstream gather fix. Default 0.0 keeps the
            byte-identity (clean-one-hot) form for the DSL unit test.

    Returns:
        ``tuple[FFNRule, ...]`` — 256 rules for ``width_bytes=1``,
        65536 rules for ``width_bytes=2``.

    Raises:
        ValueError: if ``width_bytes < 1``, or if exactly one of
            ``result_byte1_lo_base`` / ``result_byte1_hi_base`` is given
            (both or neither required), or if either is given with
            ``width_bytes != 2``.
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

    route_byte1 = (
        result_byte1_lo_base is not None or result_byte1_hi_base is not None
    )
    if route_byte1:
        if result_byte1_lo_base is None or result_byte1_hi_base is None:
            raise ValueError(
                "wide_mul_rules: result_byte1_lo_base and "
                "result_byte1_hi_base must both be supplied (or both "
                f"omitted); got lo={result_byte1_lo_base!r}, "
                f"hi={result_byte1_hi_base!r}."
            )
        if width_bytes != 2:
            raise ValueError(
                "wide_mul_rules: result_byte1_lo_base/result_byte1_hi_base "
                f"routing requires width_bytes=2; got {width_bytes!r}."
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
                    if route_byte1:
                        # Byte 0 -> result_base (OUTPUT_LO/HI); byte 1 ->
                        # dedicated MUL_RESULT_HI band (avoids result_base+32
                        # = ADDR_KEY collision).
                        writes = (
                            (f"{result_base}+{nib0}", write_amplitude),
                            (f"{result_base}+{16 + nib1}", write_amplitude),
                            (f"{result_byte1_lo_base}+{nib2}", write_amplitude),
                            (f"{result_byte1_hi_base}+{nib3}", write_amplitude),
                        )
                    else:
                        writes = (
                            (f"{result_base}+{nib0}", write_amplitude),
                            (f"{result_base}+{16 + nib1}", write_amplitude),
                            (f"{result_base}+{32 + nib2}", write_amplitude),
                            (f"{result_base}+{48 + nib3}", write_amplitude),
                        )
                    conditions = [
                        (marker_gate, marker_cond_weight),
                        (f"{operand_a_base}+{a_lo}", operand_a_cond_weight),
                        (f"{operand_a_base}+{16 + a_hi}", operand_a_cond_weight),
                        (f"{operand_b_base}+{b_lo}", operand_b_cond_weight),
                        (f"{operand_b_base}+{16 + b_hi}", operand_b_cond_weight),
                    ]
                    if operand_a_artifact_blocker_weight > 0.0:
                        # Suppress the index-0 magnitude artifact: a rule
                        # whose A nibble is k is blocked if ANY OTHER
                        # non-zero A cell is hot (-> the real nibble is
                        # that other cell, and this k is the artifact).
                        bw = -operand_a_artifact_blocker_weight
                        conditions += [
                            (f"{operand_a_base}+{j}", bw)
                            for j in range(1, 16) if j != a_lo
                        ]
                        conditions += [
                            (f"{operand_a_base}+{16 + j}", bw)
                            for j in range(1, 16) if j != a_hi
                        ]
                    rules.append(multi_way_and_rule(
                        name=(
                            f"wide_mul_w2_alo{a_lo:x}_ahi{a_hi:x}_"
                            f"blo{b_lo:x}_bhi{b_hi:x}"
                        ),
                        conditions=tuple(conditions),
                        threshold=150.0 if threshold is None else threshold,
                        gate=opcode_gate,
                        writes=writes,
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
            f"deferred — multi-byte byte-accurate DIV hits TWO walls: "
            f"(a) a general 2-byte-dividend / 1-byte-divisor table is a "
            f"3-input (256^3 = 16.7M-rule) lookup, intractable as a flat "
            f"FFN; the bit-serial multi_pass alternative needs a GE-cascade "
            f"across byte rows. (b) the L7 operand-gather only delivers "
            f"STACK0 byte 0 to ALU_LO/HI at the AX row, so the dividend's "
            f"high byte is never in the residual the lookup reads — a "
            f"multi-byte operand relay is a prerequisite. See "
            f"docs/DIV_MOD_MULTIBYTE_DIVIDEND_BLOCKER_2026_06_12.md and "
            f"docs/LONG_DIVISION_FFN_RULE_INFEASIBILITY_2026_06_09.md. "
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


# ---------------------------------------------------------------------------
# GAP-PRIMITIVE #2 (docs/DSL_W5_MULDIV_LIMIT.md Path 1): multi-pass MUL.
#
# ``wide_mul_rules(width_bytes=2)`` above is a FLAT 65,536-rule cross-product
# lookup — it only exists because 8-bit x 8-bit is small enough to enumerate.
# It does NOT generalize: width_bytes=3 is 16.8M rules, width_bytes=4 is 4.3B.
#
# ``multi_pass_mul_rules`` derives the SAME 16-bit product from a COMPACT
# SCHOOLBOOK spec (per-partial-product accumulate with column carry) staged
# across FFN passes via the ``MultiPassOp`` IR. The cross-pass carry chain
# (a column's carry-out feeds the next column's pass) is exactly what a
# single-forward FFN lookup cannot express. Rule count is O(width^2 * 256)
# — TRACTABLE at every width (no cross-product explosion): width-2 is
# ~2.8k rules vs the flat lookup's 65,536.
# ---------------------------------------------------------------------------


# AMPLITUDE-NORMALIZED cascade convention (GAP-PRIMITIVE #2).
#
# A multi-pass cascade needs a FIXED POINT: every pass must CONSUME one-hots
# at a fixed residual amplitude and PRODUCE one-hots at the SAME amplitude,
# or the SwiGLU magnitude explodes pass-over-pass (silu(up) ~= up scales with
# the input, so weight-30 reads of a residual-30 workspace one-hot run away).
#
# Convention: read EVERY one-hot (marker + operands, all residual == 1.0) at
# weight 1.0. A k-way AND with threshold ``k - 0.5`` gives
# ``up = S * (k - (k-0.5)) = S * 0.5 = 50`` when all k fire, and
# ``up = S * (k-1 - (k-0.5)) = -S*0.5`` when one is missing (silu ~= 0). The
# gate (opcode) contributes 1.0, so ``output = silu(50) * 1.0 * W_down``.
# Writing at ``W_down = 1.0 / 50`` pins EVERY fired one-hot back to residual
# 1.0 — the stable fixed point (verified: all-on -> 1.0, miss-one -> 0.0).
_MP_UP_AT_FIRE = 50.0  # = S * 0.5 with threshold k-0.5 (S=100)


def _mp_write_scale(S: float, target_amplitude: float = 1.0) -> float:
    """Write W_down that pins a fired one-hot to ``target_amplitude``."""
    # up_at_fire = S * 0.5; output = up_at_fire * W_down (gate=1). Solve for
    # W_down. Independent of S beyond the 0.5 margin (S cancels).
    return target_amplitude / (S * 0.5)


def _normalized_and_rule(
    *,
    name: str,
    conditions: Sequence[Tuple[str, float]],
    writes: Sequence[str],
    opcode_gate: str,
    S: float,
    target_amplitude: float = 1.0,
) -> FFNRule:
    """A k-way AND in the amplitude-normalized cascade convention.

    ``conditions`` is a list of one-hot dim names (each read at weight 1.0,
    each at incoming residual == ``target_amplitude``). Threshold is
    ``k - 0.5`` so the AND fires iff ALL k are hot; the write pins each
    output back to ``target_amplitude``.
    """
    k = len(conditions)
    cond_terms = tuple((c, 1.0) for c in conditions)
    ws = _mp_write_scale(S, target_amplitude)
    return multi_way_and_rule(
        name=name,
        conditions=cond_terms,
        threshold=float(k) - 0.5,
        gate=opcode_gate,
        writes=tuple((dim, ws) for dim in writes),
    )


def _nibble_lookup_pass_rules(
    *,
    name: str,
    x_lane: Tuple[str, int],
    y_lane: Tuple[str, int],
    f,
    writes_fn,
    opcode_gate: str,
    marker_gate: str,
    S: float,
    x_lanes: int = 16,
    y_lanes: int = 16,
    target_amplitude: float = 1.0,
) -> Tuple[FFNRule, ...]:
    """One pass = a 2-input nibble VALUE lookup over one-hot workspace lanes.

    For every ``(x, y)`` in ``0..x_lanes-1 x 0..y_lanes-1`` emit ONE rule
    that fires iff ``marker`` AND ``x_lane[x]`` AND ``y_lane[y]`` are hot
    (each residual == ``target_amplitude``, an input one-hot OR a workspace
    one-hot written by a PRIOR pass — verified identical amplitude under
    the normalized convention), gated by ``opcode_gate``. The rule writes
    the one-hot(s) ``writes_fn(f(x, y))`` back at ``target_amplitude``, so
    the NEXT pass reads them at the SAME fixed amplitude — the cascade
    fixed point.

    ``x_lane`` / ``y_lane`` are ``(base_name, start_offset)`` tuples; the
    cell for nibble ``k`` is the SINGLE-offset dim ``"base+(start+k)"``
    (never a nested ``base+start+k``). This is the atom of the multi-pass
    cascade: the partial-product pass and every pairwise-add column pass
    are instances of it. ``f`` is the BUILD-TIME arithmetic (the compact
    spec); ``writes_fn`` maps its result to workspace/result lanes.
    """
    def cell(lane, nib):
        return f"{lane[0]}+{lane[1] + nib}"

    rules: list[FFNRule] = []
    for x in range(x_lanes):
        for y in range(y_lanes):
            res = f(x, y)
            rules.append(_normalized_and_rule(
                name=f"{name}_x{x:x}_y{y:x}",
                conditions=(marker_gate, cell(x_lane, x), cell(y_lane, y)),
                writes=tuple(writes_fn(res)),
                opcode_gate=opcode_gate,
                S=S,
                target_amplitude=target_amplitude,
            ))
    return tuple(rules)


def multi_pass_mul_rules(
    *,
    operand_a_base: str,
    operand_b_base: str,
    result_base: str,
    workspace_base: str,
    opcode_gate: str,
    marker_gate: str,
    S: float,
    width_bytes: int = 2,
    result_lane_bases: Optional[Sequence[str]] = None,
):
    """Derive wide MUL from a COMPACT schoolbook spec as a MultiPassOp.

    GAP-PRIMITIVE #2 pilot. Returns a
    :class:`~c4_release.neural_vm.unified_compiler.ir.MultiPassOp` whose
    staged passes compute ``(A * B) & 0xFFFF`` for 8-bit x 8-bit operands
    (``width_bytes=2``) via schoolbook partial-product accumulation with a
    cross-pass column-carry chain — the construct the flat
    ``wide_mul_rules`` lookup only avoids by brute-force enumeration.

    Operands are read as nibble one-hots: ``operand_a_base+nib`` is a0
    (low nibble), ``operand_a_base+16+nib`` is a1 (high nibble); likewise
    for B. The result is written as four nibble one-hots at
    ``result_base + lane*16 + nib`` (lanes 0..3 = the 16-bit product,
    little-endian). ``workspace_base`` is a scratch band (op-local
    residual band) the passes populate with intermediate value one-hots.

    Workspace lane layout (each a 16-wide value one-hot slot):

      lane 0: pp00_hi      lane 6:  col1_partial (pp00_hi+pp01_lo)%16
      lane 1: pp01_lo      lane 7:  col1_c_a     carry of that add
      lane 2: pp01_hi      lane 8:  col2_ab      (pp01_hi+pp10_hi)%16
      lane 3: pp10_lo      lane 9:  col2_c_ab    carry of that add
      lane 4: pp10_hi      lane 10: carry1       col1 -> col2 carry (0..2)
      lane 5: pp11_lo      lane 11: col2_abc     (+pp11_lo)%16
      lane 12: pp11_hi     lane 13: col2_c_abc   carry
      lane 14: carry2      col2 -> col3 carry (0..2)

    Passes (each a tractable <=1024-rule nibble lookup, NOT a
    cross-product):

      P0  Partial products: 4 * 256 rules. Compute a_i*b_j (0..225) and
          split into (lo, hi). nib0 (=pp00_lo) is FINAL -> result lane 0.
          pp00_hi/pp01_lo/pp10_lo/pp01_hi/pp10_hi/pp11_lo/pp11_hi -> ws.
      P1  Column 1 add #1: (pp00_hi + pp01_lo) -> (col1_partial, col1_c_a).
      P2  Column 1 add #2: (col1_partial + pp10_lo) -> nib1 (result lane 1)
          + col1_c_b; carry1 = col1_c_a + col1_c_b (0..2) -> ws lane 10.
      P3  Column 2 add #1: (pp01_hi + pp10_hi) -> (col2_ab, col2_c_ab).
      P4  Column 2 add #2: (col2_ab + pp11_lo) -> (col2_abc, col2_c_abc).
      P5  Column 2 add #3: (col2_abc + carry1) -> nib2 (result lane 2)
          + col2 final carry contribution; carry2 = col2_c_ab +
          col2_c_abc + this add's carry -> ws lane 14.
      P6  Column 3: (pp11_hi + carry2) -> nib3 (result lane 3).

    The carry1/carry2 SUMMING across the pairwise adds is itself a small
    lookup pass. The whole cascade is ~2.8k rules independent of the flat
    lookup's 65,536, and generalizes to width>2 by adding more
    partial-product + column passes (O(width^2) passes).

    Args:
        operand_a_base / operand_b_base: nibble-one-hot operand bands.
        result_base: 4-lane result band (used only when
            ``result_lane_bases`` is None — the 4 nibble lanes then land at
            ``result_base + lane*16 + nib`` for a single CONTIGUOUS 64-wide
            band).
        workspace_base: scratch band (>= 16*15 = 240 wide).
        opcode_gate: MUL opcode flag dim.
        marker_gate: AX-style marker dim.
        S: SwiGLU scale.
        width_bytes: only 2 supported in the pilot (8-bit x 8-bit).
        result_lane_bases: optional 4-tuple of dim base names, one per
            result nibble lane ``(nib0, nib1, nib2, nib3)``. When given, the
            product's little-endian nibble ``lane`` is written to
            ``result_lane_bases[lane] + nib`` — decoupling the 4 lanes so
            they can route to NON-CONTIGUOUS live bands (the live MUL byte-0
            lands in OUTPUT_LO/OUTPUT_HI, byte-1 in the separate
            MUL_RESULT_HI_LO/HI band). ``result_base`` is ignored when this
            is set. Byte-identity contract with the contiguous form:
            passing ``(result_base+0, result_base+16, result_base+32,
            result_base+48)`` reproduces the default lane routing exactly.

    Returns:
        ``MultiPassOp`` with 7 passes (P0..P6).
    """
    from .ir import MultiPassOp

    if width_bytes != 2:
        raise NotImplementedError(
            "multi_pass_mul_rules: pilot supports width_bytes=2 only; the "
            "schoolbook cascade generalizes to width>2 by adding "
            "partial-product + column-add passes (O(width^2) passes)."
        )

    if result_lane_bases is not None and len(result_lane_bases) != 4:
        raise ValueError(
            "multi_pass_mul_rules: result_lane_bases must be a 4-tuple "
            f"(one dim base per product nibble lane), got "
            f"{len(result_lane_bases)}"
        )

    def split(v):
        return (v & 0xF, (v >> 4) & 0xF)

    # A "lane" is a (base_name, start_offset) pair addressing a 16-wide
    # value one-hot slot. ``cell(lane, nib)`` emits a SINGLE-offset dim key
    # ``"base+(start+nib)"`` — never a nested ``base+start+nib`` (which the
    # DimRef.rsplit-on-"+" parser would mis-read). ``base(lane)`` is the
    # cell at nib 0 (for use as an ``x_base``/``y_base`` the lookup helper
    # then appends the nibble to).
    def cell(lane, nib):
        name, start = lane
        return f"{name}+{start + nib}"

    # Workspace lanes (each a 16-wide value one-hot slot).
    def WS(lane_idx):
        return (workspace_base, lane_idx * 16)

    L_pp00_hi, L_pp01_lo, L_pp01_hi = WS(0), WS(1), WS(2)
    L_pp10_lo, L_pp10_hi, L_pp11_lo = WS(3), WS(4), WS(5)
    L_col1_partial, L_col1_c_a = WS(6), WS(7)
    L_col2_ab, L_col2_c_ab = WS(8), WS(9)
    L_carry1 = WS(10)
    L_col2_abc, L_col2_c_abc = WS(11), WS(12)
    L_pp11_hi = WS(13)
    L_carry2 = WS(14)

    def RES(lane_idx):
        # Each result nibble lane is a 16-wide value one-hot slot. Default
        # (contiguous) form: lanes 0..3 pack into ``result_base`` at
        # ``lane*16``. Per-lane form: each lane routes to its own base at
        # offset 0 (the live MUL byte-0/byte-1 non-contiguous band routing).
        if result_lane_bases is not None:
            return (result_lane_bases[lane_idx], 0)
        return (result_base, lane_idx * 16)

    mp = MultiPassOp(
        name="multi_pass_mul_w2",
        workspace_band=workspace_base,
    )

    a0, a1 = (operand_a_base, 0), (operand_a_base, 16)
    b0, b1 = (operand_b_base, 0), (operand_b_base, 16)

    # ----- P0: partial products (4 nibble-pair lookups) -----------------
    p0 = mp.add_pass("p0_partial_products")
    # pp00 = a0*b0: low nibble is FINAL result lane 0; high -> ws.
    p0.ffn.rules.extend(_nibble_lookup_pass_rules(
        name="pp00", x_lane=a0, y_lane=b0, f=lambda x, y: x * y,
        writes_fn=lambda v: (cell(RES(0), split(v)[0]), cell(L_pp00_hi, split(v)[1])),
        opcode_gate=opcode_gate, marker_gate=marker_gate, S=S,
    ))
    p0.ffn.rules.extend(_nibble_lookup_pass_rules(
        name="pp01", x_lane=a0, y_lane=b1, f=lambda x, y: x * y,
        writes_fn=lambda v: (cell(L_pp01_lo, split(v)[0]), cell(L_pp01_hi, split(v)[1])),
        opcode_gate=opcode_gate, marker_gate=marker_gate, S=S,
    ))
    p0.ffn.rules.extend(_nibble_lookup_pass_rules(
        name="pp10", x_lane=a1, y_lane=b0, f=lambda x, y: x * y,
        writes_fn=lambda v: (cell(L_pp10_lo, split(v)[0]), cell(L_pp10_hi, split(v)[1])),
        opcode_gate=opcode_gate, marker_gate=marker_gate, S=S,
    ))
    p0.ffn.rules.extend(_nibble_lookup_pass_rules(
        name="pp11", x_lane=a1, y_lane=b1, f=lambda x, y: x * y,
        writes_fn=lambda v: (cell(L_pp11_lo, split(v)[0]), cell(L_pp11_hi, split(v)[1])),
        opcode_gate=opcode_gate, marker_gate=marker_gate, S=S,
    ))

    # ----- P1: column 1 add #1: pp00_hi + pp01_lo -----------------------
    p1 = mp.add_pass("p1_col1_add_a")
    p1.ffn.rules.extend(_nibble_lookup_pass_rules(
        name="c1a", x_lane=L_pp00_hi, y_lane=L_pp01_lo, f=lambda x, y: x + y,
        writes_fn=lambda v: (cell(L_col1_partial, v & 0xF), cell(L_col1_c_a, v >> 4)),
        opcode_gate=opcode_gate, marker_gate=marker_gate, S=S,
    ))

    # ----- P2: column 1 add #2: col1_partial + pp10_lo ------------------
    # nib1 is FINAL (result lane 1); carry of THIS add + col1_c_a = carry1.
    p2 = mp.add_pass("p2_col1_add_b")
    p2.ffn.rules.extend(_nibble_lookup_pass_rules(
        name="c1b", x_lane=L_col1_partial, y_lane=L_pp10_lo, f=lambda x, y: x + y,
        writes_fn=lambda v: (cell(RES(1), v & 0xF),),
        opcode_gate=opcode_gate, marker_gate=marker_gate, S=S,
    ))
    # carry1 = col1_c_a + (col1_partial + pp10_lo)//16. col1_c_a in 0..1,
    # this add's carry in 0..1 -> carry1 in 0..2 (proven max_carry1=2). A
    # 4-way AND over (col1_c_a, col1_partial, pp10_lo) reconstructs both
    # the incoming carry and the fresh add-carry in one clean lookup pass
    # (~512 units) -- still O(256), no cross-product blowup.
    for ca in range(2):
        for partial in range(16):
            for plo in range(16):
                c_add = (partial + plo) // 16
                carry1 = ca + c_add
                p2.ffn.rules.append(_normalized_and_rule(
                    name=f"carry1_ca{ca}_p{partial:x}_l{plo:x}",
                    conditions=(
                        marker_gate,
                        cell(L_col1_c_a, ca),
                        cell(L_col1_partial, partial),
                        cell(L_pp10_lo, plo),
                    ),
                    writes=(cell(L_carry1, carry1),),
                    opcode_gate=opcode_gate,
                    S=S,
                ))

    # ----- P3: column 2 add #1: pp01_hi + pp10_hi -----------------------
    p3 = mp.add_pass("p3_col2_add_a")
    p3.ffn.rules.extend(_nibble_lookup_pass_rules(
        name="c2a", x_lane=L_pp01_hi, y_lane=L_pp10_hi, f=lambda x, y: x + y,
        writes_fn=lambda v: (cell(L_col2_ab, v & 0xF), cell(L_col2_c_ab, v >> 4)),
        opcode_gate=opcode_gate, marker_gate=marker_gate, S=S,
    ))

    # ----- P4: column 2 add #2: col2_ab + pp11_lo -----------------------
    p4 = mp.add_pass("p4_col2_add_b")
    p4.ffn.rules.extend(_nibble_lookup_pass_rules(
        name="c2b", x_lane=L_col2_ab, y_lane=L_pp11_lo, f=lambda x, y: x + y,
        writes_fn=lambda v: (cell(L_col2_abc, v & 0xF), cell(L_col2_c_abc, v >> 4)),
        opcode_gate=opcode_gate, marker_gate=marker_gate, S=S,
    ))

    # ----- P5: column 2 add #3: col2_abc + carry1 -----------------------
    # nib2 FINAL (result lane 2); carry2 = col2_c_ab + col2_c_abc + this
    # add's carry. col2_c_ab,col2_c_abc in 0..1; this add carry in 0..1;
    # sum in 0..2 (proven: max_carry2=2).
    p5 = mp.add_pass("p5_col2_add_c")
    p5.ffn.rules.extend(_nibble_lookup_pass_rules(
        name="c2c", x_lane=L_col2_abc, y_lane=L_carry1, f=lambda x, y: x + y,
        writes_fn=lambda v: (cell(RES(2), v & 0xF),),
        opcode_gate=opcode_gate, marker_gate=marker_gate, S=S,
        y_lanes=3,
    ))
    # carry2 = col2_c_ab + col2_c_abc + (col2_abc + carry1)//16.
    for cab in range(2):
        for cabc in range(2):
            for abc in range(16):
                for c1 in range(3):
                    c_add = (abc + c1) // 16
                    carry2 = cab + cabc + c_add
                    p5.ffn.rules.append(_normalized_and_rule(
                        name=f"carry2_ab{cab}_abc{cabc}_v{abc:x}_c{c1}",
                        conditions=(
                            marker_gate,
                            cell(L_col2_c_ab, cab),
                            cell(L_col2_c_abc, cabc),
                            cell(L_col2_abc, abc),
                            cell(L_carry1, c1),
                        ),
                        writes=(cell(L_carry2, carry2),),
                        opcode_gate=opcode_gate,
                        S=S,
                    ))

    # ----- P6: column 3: pp11_hi + carry2 -------------------------------
    # nib3 FINAL (result lane 3). Any col-4 carry drops (product<65536).
    p6 = mp.add_pass("p6_col3")
    p6.ffn.rules.extend(_nibble_lookup_pass_rules(
        name="c3", x_lane=L_pp11_hi, y_lane=L_carry2, f=lambda x, y: x + y,
        writes_fn=lambda v: (cell(RES(3), v & 0xF),),
        opcode_gate=opcode_gate, marker_gate=marker_gate, S=S,
        y_lanes=3,
    ))

    return mp


# ---------------------------------------------------------------------------
# GAP-PRIMITIVE #2 (DIV pilot): multi-pass long-division cascade.
#
# ``wide_div_rules_ge_format(width_bytes=1)`` is a FLAT 256x256 cross-product
# lookup (65,536 rules per opcode batch) — byte-accurate but O(256^width),
# so it does NOT generalize past 8-bit operands (width=2 is 4.3B rules).
#
# ``multi_pass_div_rules`` derives the SAME single-byte quotient/remainder
# from a COMPACT binary long-division spec (bit-serial shift-subtract with a
# cross-pass RUNNING-REMAINDER carry) staged across FFN passes via the
# ``MultiPassOp`` IR. The cross-pass remainder chain (bit i's remainder feeds
# bit i-1's shift) is exactly what a single-forward FFN lookup cannot express.
# Rule count is O(width * 256) per shift-subtract iteration — TRACTABLE at
# every width (no cross-product explosion): the 8-bit cascade is ~7.7k rules
# vs the flat lookup's 65,536 per opcode.
#
# This is the DIV analogue of ``multi_pass_mul_rules`` (schoolbook MUL). It
# uses the SAME amplitude-normalized cascade convention (``_normalized_and_rule``
# / ``_nibble_lookup_pass_rules``: read weight 1.0, threshold k-0.5 -> up=S*0.5,
# write 1/(S*0.5)) so stacked SwiGLU magnitude stays pinned at the residual-1.0
# fixed point pass-over-pass.
#
# The additive FFN cannot ERASE a residual one-hot, so every distinct
# intermediate value uses its OWN fresh workspace lane (never a lane reused
# across bits) — a stale prior-bit one-hot would additively collide with a
# fresh write. The running remainder therefore snapshots into a fresh 2-nibble
# lane pair each bit (R[8]..R[0]); the workspace is bounded (O(width) lanes).
# ---------------------------------------------------------------------------


def _mp_onehot_rule(
    *,
    name: str,
    conditions: Sequence[str],
    writes: Sequence[str],
    opcode_gate: str,
    S: float,
) -> FFNRule:
    """A k-input AND lookup rule in the normalized cascade convention.

    Thin wrapper over :func:`_normalized_and_rule` that takes the condition
    one-hot dim names directly (each read at weight 1.0). Used for the
    1-/2-/3-/4-input nibble lookups the long-division cascade needs (bit
    extract, double, borrow-subtract, mux-select, nibble assembly).
    """
    return _normalized_and_rule(
        name=name,
        conditions=tuple(conditions),
        writes=tuple(writes),
        opcode_gate=opcode_gate,
        S=S,
        target_amplitude=1.0,
    )


def multi_pass_div_rules(
    *,
    dividend_a_base: str,
    divisor_b_base: str,
    quotient_lane_bases: Sequence[str],
    remainder_lane_bases: Sequence[str],
    workspace_base: str,
    opcode_gate: str,
    marker_gate: str,
    S: float,
    width_bytes: int = 1,
):
    """Derive single-byte DIV/MOD from a COMPACT long-division spec (MultiPassOp).

    GAP-PRIMITIVE #2 DIV pilot. Returns a
    :class:`~c4_release.neural_vm.unified_compiler.ir.MultiPassOp` whose staged
    passes compute ``a // b`` (quotient) and ``a % b`` (remainder) for 8-bit
    operands (``width_bytes=1``) via binary long division — bit-serial
    shift-subtract with a cross-pass running-remainder carry, the construct
    the flat ``wide_div_rules_ge_format`` lookup only avoids by brute-force
    enumeration.

    Divide-by-zero convention (matches ``wide_div_rules`` /
    ``FlattenedDivMod``): ``b == 0`` -> ``q = 0, r = a`` (dividend
    pass-through), realized structurally (see the b==0 note below).

    Operands are read as nibble one-hots: ``dividend_a_base+nib`` is a's low
    nibble a0, ``dividend_a_base+16+nib`` is a's high nibble a1; likewise the
    divisor B at ``divisor_b_base`` (b0 = ``+nib``, b1 = ``+16+nib``). The
    quotient / remainder are each written as two nibble one-hots (little-endian
    lo, hi) at ``quotient_lane_bases[k]+nib`` / ``remainder_lane_bases[k]+nib``
    (``k in {0,1}``). ``workspace_base`` is a scratch band (op-local residual
    band) the passes populate with intermediate value one-hots.

    Algorithm (validated bit-exact vs Python ``divmod`` over all 65,536 (a,b)
    pairs — ``tools/_div_algo_proto.py`` + ``run_symbolic`` sweep):

      Maintain a running remainder ``R`` (0..255, two nibbles). Initialise
      R = 0. For bit ``i`` = 7..0 (MSB->LSB):

        R2 = 2*R + a_bit(i)          (<= 9 bits: R2_lo, R2_hi, R2_top)
        if R2 >= b:  R = R2 - b,  q_bit(i) = 1
        else:        R = R2,       q_bit(i) = 0

      The compare ``R2 >= b`` and subtract ``R2 - b`` are one nibble
      borrow-subtract chain (cmp = final borrow == 0). After the last bit,
      R is the remainder; the 8 q_bits assemble into the quotient byte.

    Pass structure (per bit, distinct fresh workspace lanes so no additive
    one-hot collision — the additive FFN cannot erase):

      * P0 seed:      init R[8] = (0,0) one-hots; extract the 8 a_bit one-hots
                      from a's two nibbles (1-input nibble->bit lookups).
      * per bit i (4 passes, strictly sequential — bit i's R feeds bit i-1):
          - double_lo:  R2_lo, carry_lo = split(2*R_lo + a_bit(i))
          - double_hi:  R2_hi, R2_top   = split(2*R_hi + carry_lo)
          - subtract:   d_lo,bl0 = subb(R2_lo,b_lo,0); d_hi,bl1 = subb(R2_hi,
                        b_hi,bl0); cmp = (R2_top >= bl1)  (top borrow == 0)
          - select:     if cmp: R_next=(d_lo,d_hi), q_bit=1 else
                        R_next=(R2_lo,R2_hi), q_bit=0
      * final assemble: q_lo = bits0..3, q_hi = bits4..7 (4-input AND lookups);
                        remainder = final R; route q -> quotient lanes and
                        r -> remainder lanes.

    b==0 handling: with divisor b==0 the compare ``R2 >= b`` is ALWAYS true
    (R2 >= 0 == b), so the cascade result is garbage. A seed-time detector
    fires BZERO (both b nibbles == 0) or BNONZERO. The final ASSEMBLE writes
    (cascade -> result lanes) are gated on BNONZERO, and a b==0 override pass
    gated on BZERO writes the zero-divide convention (q=0, r=a). BZERO and
    BNONZERO are mutually-exclusive one-hots, so each result lane holds EXACTLY
    ONE hot cell for every (a,b) — a clean argmax decode. This matches
    ``wide_div_rules`` / ``FlattenedDivMod``'s explicit zero-divide guard.

    Rule count: ~7.7k units (8 bits x ~950 + seed + assemble) — vs the flat
    ``wide_div_rules_ge_format`` 65,536 per opcode. O(width*256) per bit, so
    it generalizes to wider dividends by adding bit iterations (no
    cross-product blowup).

    Args:
        dividend_a_base: nibble-one-hot dividend band (a0 at +nib, a1 at
            +16+nib).
        divisor_b_base: nibble-one-hot divisor band (b0 at +nib, b1 at +16+nib).
        quotient_lane_bases: 2-tuple of dim base names (q_lo, q_hi lanes).
        remainder_lane_bases: 2-tuple of dim base names (r_lo, r_hi lanes).
        workspace_base: scratch band (>= _DIV_WS_LANES*16 wide).
        opcode_gate: DIV/MOD opcode flag dim (the cascade computes BOTH; the
            live install routes q->OUTPUT for OP_DIV and r->OUTPUT for OP_MOD
            in a final gated pass).
        marker_gate: AX-style marker dim.
        S: SwiGLU scale.
        width_bytes: only 1 supported in the pilot (8-bit dividend/divisor).

    Returns:
        ``MultiPassOp`` with the seed + 32 per-bit + assemble passes.
    """
    from .ir import MultiPassOp

    if width_bytes != 1:
        raise NotImplementedError(
            "multi_pass_div_rules: pilot supports width_bytes=1 only; the "
            "shift-subtract cascade generalizes to wider dividends by adding "
            "bit iterations (O(width) passes)."
        )
    if len(quotient_lane_bases) != 2 or len(remainder_lane_bases) != 2:
        raise ValueError(
            "multi_pass_div_rules: quotient_lane_bases and "
            "remainder_lane_bases must each be a 2-tuple (lo, hi nibble lane "
            "base names)"
        )

    def cell(base, off):
        return f"{base}+{off}"

    # Workspace lane layout. Each lane is a 16-wide value one-hot slot except
    # bit/carry/borrow/cmp lanes which use 2 cells (0/1). We allocate them at
    # 16-cell strides for a uniform address scheme (bit lanes waste 14 cells;
    # cheap vs the collision-free clarity).
    _next = [0]

    def alloc():
        idx = _next[0]
        _next[0] += 1
        return (workspace_base, idx * 16)

    # a_bit one-hot lanes (2-value each), b==0 guard, running-R snapshots.
    A_BIT = [alloc() for _ in range(8)]          # a_bit(0)..a_bit(7)
    BZERO = alloc()                              # b==0 detector (cell 1 hot)
    BNONZERO = alloc()                           # b!=0 detector (cell 1 hot)
    # R snapshots: R[8] = seed (0), R[k] = remainder AFTER processing down to
    # bit k. R[k] is (R_lo, R_hi) lane pair.
    R_LO = [alloc() for _ in range(9)]           # R_LO[8..0]
    R_HI = [alloc() for _ in range(9)]
    # per-bit scratch (fresh per bit)
    R2_LO = [alloc() for _ in range(8)]
    R2_HI = [alloc() for _ in range(8)]
    R2_TOP = [alloc() for _ in range(8)]         # bit 8 (0/1)
    CARRY_LO = [alloc() for _ in range(8)]       # doubling lo->hi carry (0/1)
    D_LO = [alloc() for _ in range(8)]
    D_HI = [alloc() for _ in range(8)]
    BL0 = [alloc() for _ in range(8)]            # borrow after lo subtract
    BL1 = [alloc() for _ in range(8)]            # borrow after hi subtract
    CMP = [alloc() for _ in range(8)]            # R2 >= b (cell 1 hot)
    QBIT = [alloc() for _ in range(8)]           # q bit i (cell 1 hot)

    ws_lanes = _next[0]

    a0, a1 = (dividend_a_base, 0), (dividend_a_base, 16)
    b0, b1 = (divisor_b_base, 0), (divisor_b_base, 16)

    mp = MultiPassOp(name="multi_pass_div_w1", workspace_band=workspace_base)

    # ----- P0 seed: init R[8]=0, extract a_bits, detect b==0 -----------------
    p0 = mp.add_pass("p0_seed")
    # Seed R[8] one-hots to value 0 (marker-only rule -> writes cell 0).
    for lane in (R_LO[8], R_HI[8]):
        p0.ffn.rules.append(_mp_onehot_rule(
            name=f"seed_{lane[0]}_{lane[1]}",
            conditions=(marker_gate,),
            writes=(cell(lane[0], lane[1] + 0),),
            opcode_gate=opcode_gate, S=S,
        ))
    # Extract a_bit(i): bit b of nibble a0 (b=0..3) / a1 (b=4..7). 1-input
    # lookup over the nibble value -> the bit one-hot cell.
    for b_idx in range(8):
        nib_lane = a0 if b_idx < 4 else a1
        bit_in_nib = b_idx % 4
        for nv in range(16):
            bit = (nv >> bit_in_nib) & 1
            p0.ffn.rules.append(_mp_onehot_rule(
                name=f"abit{b_idx}_n{nv:x}",
                conditions=(marker_gate, cell(nib_lane[0], nib_lane[1] + nv)),
                writes=(cell(A_BIT[b_idx][0], A_BIT[b_idx][1] + bit),),
                opcode_gate=opcode_gate, S=S,
            ))
    # b==0 / b!=0 detectors: fire BZERO iff b0==0 AND b1==0; fire BNONZERO iff
    # (b0!=0) OR (b1!=0). Realized as: for each (b0v, b1v) pair, write the
    # right detector cell. 256 rules — but only need which of the two; emit a
    # 2-input AND per pair once.
    for b0v in range(16):
        for b1v in range(16):
            is_zero = (b0v == 0 and b1v == 0)
            det = BZERO if is_zero else BNONZERO
            p0.ffn.rules.append(_mp_onehot_rule(
                name=f"bzero_b0{b0v:x}_b1{b1v:x}",
                conditions=(marker_gate, cell(b0[0], b0[1] + b0v),
                            cell(b1[0], b1[1] + b1v)),
                writes=(cell(det[0], det[1] + 1),),
                opcode_gate=opcode_gate, S=S,
            ))

    b_lo_cell = lambda v: cell(b0[0], b0[1] + v)
    b_hi_cell = lambda v: cell(b1[0], b1[1] + v)

    # ----- per-bit passes (bit = 7..0) --------------------------------------
    for step, i in enumerate(range(7, -1, -1)):
        r_in_lo, r_in_hi = R_LO[i + 1], R_HI[i + 1]   # R from prior bit
        r_out_lo, r_out_hi = R_LO[i], R_HI[i]         # R after this bit
        abit = A_BIT[i]

        # -- double_lo: R2_lo, carry_lo = split(2*R_lo + a_bit) --
        pd_lo = mp.add_pass(f"p_bit{i}_double_lo")
        for rlo in range(16):
            for bit in range(2):
                v = 2 * rlo + bit
                lo, c = v & 0xF, v >> 4
                pd_lo.ffn.rules.append(_mp_onehot_rule(
                    name=f"dbllo_i{i}_r{rlo:x}_b{bit}",
                    conditions=(marker_gate, cell(r_in_lo[0], r_in_lo[1] + rlo),
                                cell(abit[0], abit[1] + bit)),
                    writes=(cell(R2_LO[step][0], R2_LO[step][1] + lo),
                            cell(CARRY_LO[step][0], CARRY_LO[step][1] + c)),
                    opcode_gate=opcode_gate, S=S,
                ))

        # -- double_hi: R2_hi, R2_top = split(2*R_hi + carry_lo) --
        pd_hi = mp.add_pass(f"p_bit{i}_double_hi")
        for rhi in range(16):
            for c in range(2):
                v = 2 * rhi + c
                hi, top = v & 0xF, v >> 4
                pd_hi.ffn.rules.append(_mp_onehot_rule(
                    name=f"dblhi_i{i}_r{rhi:x}_c{c}",
                    conditions=(marker_gate, cell(r_in_hi[0], r_in_hi[1] + rhi),
                                cell(CARRY_LO[step][0], CARRY_LO[step][1] + c)),
                    writes=(cell(R2_HI[step][0], R2_HI[step][1] + hi),
                            cell(R2_TOP[step][0], R2_TOP[step][1] + top)),
                    opcode_gate=opcode_gate, S=S,
                ))

        # -- subtract lo: d_lo, bl0 = subborrow(R2_lo, b_lo, 0) --
        ps_lo = mp.add_pass(f"p_bit{i}_sub_lo")
        for r2lo in range(16):
            for blo in range(16):
                d = r2lo - blo
                diff, borrow = (d + 16, 1) if d < 0 else (d, 0)
                ps_lo.ffn.rules.append(_mp_onehot_rule(
                    name=f"sublo_i{i}_r{r2lo:x}_b{blo:x}",
                    conditions=(marker_gate,
                                cell(R2_LO[step][0], R2_LO[step][1] + r2lo),
                                b_lo_cell(blo)),
                    writes=(cell(D_LO[step][0], D_LO[step][1] + diff),
                            cell(BL0[step][0], BL0[step][1] + borrow)),
                    opcode_gate=opcode_gate, S=S,
                ))

        # -- subtract hi + top-borrow -> cmp: d_hi, bl1 = subborrow(R2_hi,
        #    b_hi, bl0); cmp = (R2_top >= bl1). Fuse the top compare in by
        #    also reading R2_top: 4-input AND over (R2_hi, b_hi, bl0, R2_top).
        ps_hi = mp.add_pass(f"p_bit{i}_sub_hi")
        for r2hi in range(16):
            for bhi in range(16):
                for bl0 in range(2):
                    d = r2hi - bhi - bl0
                    diff, bl1 = (d + 16, 1) if d < 0 else (d, 0)
                    for r2top in range(2):
                        cmp = 1 if r2top >= bl1 else 0
                        ps_hi.ffn.rules.append(_mp_onehot_rule(
                            name=f"subhi_i{i}_r{r2hi:x}_b{bhi:x}_c{bl0}_t{r2top}",
                            conditions=(
                                marker_gate,
                                cell(R2_HI[step][0], R2_HI[step][1] + r2hi),
                                b_hi_cell(bhi),
                                cell(BL0[step][0], BL0[step][1] + bl0),
                                cell(R2_TOP[step][0], R2_TOP[step][1] + r2top),
                            ),
                            writes=(
                                cell(D_HI[step][0], D_HI[step][1] + diff),
                                cell(CMP[step][0], CMP[step][1] + cmp),
                            ),
                            opcode_gate=opcode_gate, S=S,
                        ))

        # -- select: q_bit = cmp; R_next = cmp ? (d_lo,d_hi) : (R2_lo,R2_hi) --
        psel = mp.add_pass(f"p_bit{i}_select")
        # q_bit = cmp (identity copy)
        for c in range(2):
            psel.ffn.rules.append(_mp_onehot_rule(
                name=f"qbit_i{i}_c{c}",
                conditions=(marker_gate, cell(CMP[step][0], CMP[step][1] + c)),
                writes=(cell(QBIT[step][0], QBIT[step][1] + c),),
                opcode_gate=opcode_gate, S=S,
            ))
        # R_next_lo: cmp==1 -> d_lo; cmp==0 -> R2_lo. (2-input AND per branch)
        for v in range(16):
            psel.ffn.rules.append(_mp_onehot_rule(
                name=f"rnlo_i{i}_cmp1_v{v:x}",
                conditions=(marker_gate, cell(CMP[step][0], CMP[step][1] + 1),
                            cell(D_LO[step][0], D_LO[step][1] + v)),
                writes=(cell(r_out_lo[0], r_out_lo[1] + v),),
                opcode_gate=opcode_gate, S=S,
            ))
            psel.ffn.rules.append(_mp_onehot_rule(
                name=f"rnlo_i{i}_cmp0_v{v:x}",
                conditions=(marker_gate, cell(CMP[step][0], CMP[step][1] + 0),
                            cell(R2_LO[step][0], R2_LO[step][1] + v)),
                writes=(cell(r_out_lo[0], r_out_lo[1] + v),),
                opcode_gate=opcode_gate, S=S,
            ))
        for v in range(16):
            psel.ffn.rules.append(_mp_onehot_rule(
                name=f"rnhi_i{i}_cmp1_v{v:x}",
                conditions=(marker_gate, cell(CMP[step][0], CMP[step][1] + 1),
                            cell(D_HI[step][0], D_HI[step][1] + v)),
                writes=(cell(r_out_hi[0], r_out_hi[1] + v),),
                opcode_gate=opcode_gate, S=S,
            ))
            psel.ffn.rules.append(_mp_onehot_rule(
                name=f"rnhi_i{i}_cmp0_v{v:x}",
                conditions=(marker_gate, cell(CMP[step][0], CMP[step][1] + 0),
                            cell(R2_HI[step][0], R2_HI[step][1] + v)),
                writes=(cell(r_out_hi[0], r_out_hi[1] + v),),
                opcode_gate=opcode_gate, S=S,
            ))

    # ----- assemble: quotient nibbles from the 8 q_bits, route results ------
    # q_lo = bit0 + 2*bit1 + 4*bit2 + 8*bit3 ; q_hi = bit4..bit7.
    # QBIT[step] holds bit i where step = 7 - i, so QBIT for bit b is at
    # index (7 - b).
    #
    # Every assemble-from-cascade write is CONDITIONED on BNONZERO (b != 0) so
    # the cascade result NEVER lands on the lanes for a b==0 row (where it is
    # garbage — the compare R2>=0 is always true). The b==0 override pass then
    # writes the zero-divide convention on the SAME lanes gated on BZERO. BZERO
    # and BNONZERO are mutually exclusive one-hots, so each result lane holds
    # EXACTLY ONE hot cell — a clean argmax decode, not a fragile tie.
    def qbit_lane(b):
        return QBIT[7 - b]

    bnz_cell = cell(BNONZERO[0], BNONZERO[1] + 1)

    p_asm = mp.add_pass("p_assemble")
    # q_lo (bits 0..3): 5-input AND over the 4 bit one-hots + BNONZERO.
    for combo in range(16):
        bits = [(combo >> k) & 1 for k in range(4)]
        conds = [marker_gate, bnz_cell]
        for k in range(4):
            lane = qbit_lane(k)
            conds.append(cell(lane[0], lane[1] + bits[k]))
        p_asm.ffn.rules.append(_mp_onehot_rule(
            name=f"qlo_{combo:x}",
            conditions=tuple(conds),
            writes=(cell(quotient_lane_bases[0], combo),),
            opcode_gate=opcode_gate, S=S,
        ))
    # q_hi (bits 4..7)
    for combo in range(16):
        bits = [(combo >> k) & 1 for k in range(4)]
        conds = [marker_gate, bnz_cell]
        for k in range(4):
            lane = qbit_lane(4 + k)
            conds.append(cell(lane[0], lane[1] + bits[k]))
        p_asm.ffn.rules.append(_mp_onehot_rule(
            name=f"qhi_{combo:x}",
            conditions=tuple(conds),
            writes=(cell(quotient_lane_bases[1], combo),),
            opcode_gate=opcode_gate, S=S,
        ))
    # remainder = final R (R[0]); copy R[0] nibbles to the remainder lanes
    # (b != 0 only).
    for v in range(16):
        p_asm.ffn.rules.append(_mp_onehot_rule(
            name=f"rlo_{v:x}",
            conditions=(marker_gate, bnz_cell, cell(R_LO[0][0], R_LO[0][1] + v)),
            writes=(cell(remainder_lane_bases[0], v),),
            opcode_gate=opcode_gate, S=S,
        ))
        p_asm.ffn.rules.append(_mp_onehot_rule(
            name=f"rhi_{v:x}",
            conditions=(marker_gate, bnz_cell, cell(R_HI[0][0], R_HI[0][1] + v)),
            writes=(cell(remainder_lane_bases[1], v),),
            opcode_gate=opcode_gate, S=S,
        ))

    # ----- b==0 override: q=0, r=a (dividend pass-through). ------------------
    # Gated on the BZERO detector (mutually exclusive with the BNONZERO-gated
    # assemble above), so for a b==0 row ONLY these convention cells are hot —
    # the cascade garbage never reaches the result lanes. This is the
    # zero-divide convention wide_div_rules / FlattenedDivMod use (q=0, r=a).
    p_bz = mp.add_pass("p_bzero_override")
    # q_lo = 0, q_hi = 0
    for lane in (quotient_lane_bases[0], quotient_lane_bases[1]):
        p_bz.ffn.rules.append(_mp_onehot_rule(
            name=f"bz_q0_{lane}",
            conditions=(marker_gate, cell(BZERO[0], BZERO[1] + 1)),
            writes=(cell(lane, 0),),
            opcode_gate=opcode_gate, S=S,
        ))
    # r = a: copy a0 -> r_lo, a1 -> r_hi (gated on BZERO)
    for av in range(16):
        p_bz.ffn.rules.append(_mp_onehot_rule(
            name=f"bz_rlo_{av:x}",
            conditions=(marker_gate, cell(BZERO[0], BZERO[1] + 1),
                        cell(a0[0], a0[1] + av)),
            writes=(cell(remainder_lane_bases[0], av),),
            opcode_gate=opcode_gate, S=S,
        ))
        p_bz.ffn.rules.append(_mp_onehot_rule(
            name=f"bz_rhi_{av:x}",
            conditions=(marker_gate, cell(BZERO[0], BZERO[1] + 1),
                        cell(a1[0], a1[1] + av)),
            writes=(cell(remainder_lane_bases[1], av),),
            opcode_gate=opcode_gate, S=S,
        ))

    mp._div_ws_lanes = ws_lanes  # introspection: workspace width in lanes
    return mp


__all__ = [
    "bitwise_rules",
    "nibble_alu_lane_rules",
    "nibble_compare_lane_rules",
    "nibble_fused_madd_combine_rules",
    "wide_add_rules",
    "wide_sub_rules",
    "wide_ge_add_rules",
    "wide_ge_sub_rules",
    "wide_shift_rules",
    "wide_mul_rules",
    "multi_pass_mul_rules",
    "multi_pass_div_rules",
    "wide_div_rules",
    "wide_div_rules_ge_format",
]
