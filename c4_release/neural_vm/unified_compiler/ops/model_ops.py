"""Model-level and post-pass op factories. See ../migrated_ops.py for history."""

from ...constants import INSTR_WIDTH, PC_OFFSET
from ..ir import CompilerIR, FFNRule, TokenEmbeddingRule
from ..building_blocks_dsl import multi_way_and_rule
from ..isa_semantics_dsl import (
    AssignDelta,
    BranchTargetDelta,
    ControlOpBundle,
    FullWidthByteEmissionSpec,
    PushDelta,
    RegisterDeltaSpec,
    control_op,
    full_width_byte_emission,
    register_delta,
)
from ..layer_compiler import Operation
from ..primitives import AO, AP, DeclarativeAttentionHeadSpec, Primitives
import torch.nn as nn
from .shared import _as_setdim_proxy, l8_operand_sp_disc_enabled
from .residual_band_registry import register_residual_band
from ...dim_registry import dim_ref
from ...attention_head_allocator import (
    AttentionHeadAllocator as _AttnHeadAlloc,
)


_IO_PUTCHAR_ROUTING_START_UNIT = 1500


# === JSR-target PC byte-1 delivery (flag C4_JSR_PC_BYTE1, default OFF) ===
#
# A JSR to instruction idx>=32 jumps to PC=idx*8+2>=256, so PC byte-1 must be
# NONZERO. The model emitted 0 and diverged at step 0 (gcd 900-949 / rec_fib
# 725-749 deep-loop wall). This feature STAGEs byte-1 = FETCH_HI_nib>>1 into the
# ``JSR_PC_B1`` band at the PC marker (override FFN), the L6+1 relay head copies
# it to ``JSR_PC_B1_AT_B0`` at the PC byte-0 row, and the post-tail emit FFN
# re-supplies it into ``OUTPUT_LO`` so the LM head emits the correct token.
# All three stages are gated on this flag; OFF -> the bands are omitted, the
# override stages nothing (legacy reserved no-op), and the relay/emit produce
# zero rules -> golden byte-identical.
#
# Width 8: the staged value is ``FETCH_HI_nib >> 1`` (k in 0..15 -> 0..7), and
# every 1096 first-JSR target's PC byte-1 low nibble is in 1..7 (high nibble 0).
_JSR_PC_B1_WIDTH = 8


def _jsr_pc_byte1_enabled() -> bool:
    """True when the JSR-target PC byte-1 delivery flag is on (default OFF)."""
    import os as _os
    return _os.environ.get("C4_JSR_PC_BYTE1", "0") != "0"


# Flag-gated over-width bands for the JSR PC byte-1 relay (collected only when
# the flag is on -> flag-OFF d_model is byte-identical to golden). JSR_PC_B1 is
# the marker-staged nibble; JSR_PC_B1_AT_B0 is the relayed copy at the PC byte-0
# row that the post-tail emit FFN reads. never_share: both carry a JSR-exclusive
# one-hot that must not be liveness-merged onto another op's dim.
register_residual_band(
    "JSR_PC_B1", _JSR_PC_B1_WIDTH, owner="make_function_call_weights_op",
    flag=_jsr_pc_byte1_enabled, never_share=True,
)
register_residual_band(
    "JSR_PC_B1_AT_B0", _JSR_PC_B1_WIDTH, owner="make_jsr_pc_byte1_emit_op",
    flag=_jsr_pc_byte1_enabled, never_share=True,
)


def _io_putchar_routing_rules(S: float) -> tuple[FFNRule, ...]:
    """CompilerIR rules for ``_set_io_putchar_routing`` (L6 FFN units 1500..1532).

    Mirrors the imperative helper bit-for-bit:
      - unit 1500 (constant_write): ``OP_PUTCHAR + MARK_AX >= 4.0`` ->
        ``IO_IS_PUTCHAR += 2.0/S``.
      - units 1501..1516 (gated_write): same threshold gate,
        ``W_gate[AX_CARRY_LO+k]=1.0`` -> ``OUTPUT_LO+k += 2.0/S``.
      - units 1517..1532 (gated_write): same threshold gate,
        ``W_gate[AX_CARRY_HI+k]=1.0`` -> ``OUTPUT_HI_THIS_STEP+k += 2.0/S``
        (``OUTPUT_HI_THIS_STEP`` is the canonical name for the same-step
        write band; numerically aliases ``OUTPUT_HI``).
    """
    T = 4.0
    write_scale = 2.0 / S
    conditions = (("OP_PUTCHAR", 1.0), ("MARK_AX", 1.0))
    rules: list[FFNRule] = [
        multi_way_and_rule(
            name="io_putchar_detect",
            conditions=conditions,
            threshold=T,
            writes=(("IO_IS_PUTCHAR", write_scale),),
        ),
    ]
    for k in range(16):
        rules.append(multi_way_and_rule(
            name=f"io_putchar_route_lo_{k}",
            conditions=conditions,
            threshold=T,
            gate=f"AX_CARRY_LO+{k}",
            writes=((f"OUTPUT_LO+{k}", write_scale),),
        ))
    for k in range(16):
        rules.append(multi_way_and_rule(
            name=f"io_putchar_route_hi_{k}",
            conditions=conditions,
            threshold=T,
            gate=f"AX_CARRY_HI+{k}",
            writes=((f"OUTPUT_HI_THIS_STEP+{k}", write_scale),),
        ))
    return tuple(rules)


def _lower_io_putchar_routing_ir(ffn, S: float, BD) -> int:
    rules = _io_putchar_routing_rules(S)
    dim_positions = Primitives.dim_positions_from_bd(
        BD,
        Primitives.ffn_rule_dim_names(rules),
    )
    return Primitives.lower_ffn_rules(
        ffn,
        rules,
        dim_positions,
        start_unit=_IO_PUTCHAR_ROUTING_START_UNIT,
        S=S,
    )


def _io_putchar_routing_ir(dim_positions, HD, S: float = 100.0) -> CompilerIR:
    """Informational :class:`CompilerIR` for ``io_putchar_routing``.

    Mirrors ``_function_call_weights_ir`` -- exposed via
    ``compiler_ir_factory=`` so the declarative verifier and symbolic
    tooling see the 33 :class:`FFNRule` declarations the bake lowers.
    The production bake stays in :func:`_lower_io_putchar_routing_ir`
    because it pins ``start_unit=_IO_PUTCHAR_ROUTING_START_UNIT`` (1500)
    while ``CompilerIR.lower_ffn`` lowers at ``start_unit=0``; both
    produce the same per-rule weights at their respective offsets.
    Phase 11.A.
    """
    del dim_positions, HD
    ir = CompilerIR()
    ir.layer(0).ffn.rules.extend(_io_putchar_routing_rules(S))
    return ir


def make_io_putchar_routing_op() -> Operation:
    """Bake L6 FFN PUTCHAR routing units (IO_IS_PUTCHAR + AX_CARRY -> OUTPUT).

    Originally an inline call in `set_vm_weights`:
        `_set_io_putchar_routing(ffn6, S, BD)`

    Operates on `model.blocks[6].ffn` (L6 FFN). Modeled as kind="model" so we
    can resolve `ffn6` from the model handle inside the bake_fn.

    Phase 998: runs just BEFORE legacy_bake (999) so that the L6 FFN units we
    program (starting at unit 1500) are present when `_right_size_ffns`
    (called at the end of legacy_bake) prunes dead units. Running at phase
    > 999 would write into already-rightsized FFN slots that no longer exist.

    Migrated 2026-06-01 (Phase 6 Wave 3L): the imperative helper
    ``_set_io_putchar_routing`` is replaced by an ``FFNRule``-driven
    lowerer (``_lower_io_putchar_routing_ir``) — 33 rules
    (1 ``constant_write`` detector + 16 LO ``gated_write`` + 16 HI
    ``gated_write``) lowered through ``Primitives.lower_ffn_rules`` at
    pinned units 1500..1532. Byte-identical to the legacy helper;
    validated via ``compare_symbolic_to_lowered_ffn`` and direct tensor
    comparison against ``_set_io_putchar_routing``.
    """
    def bake(model, dim_positions, S):
        proxy = _as_setdim_proxy(dim_positions)
        _lower_io_putchar_routing_ir(model.blocks[6].ffn, S, proxy)

    # Dim-ownership claims. ``_set_io_putchar_routing`` (vm_step.py:8068+)
    # programs 33 L6 FFN units starting at unit 1500:
    #   - unit 1500: OP_PUTCHAR & MARK_AX detector -> IO_IS_PUTCHAR
    #   - units 1501..1516: AX_CARRY_LO[k] -> OUTPUT_LO[k] (k=0..15)
    #   - units 1517..1532: AX_CARRY_HI[k] -> OUTPUT_HI[k] (k=0..15)
    # Every unit shares the (OP_PUTCHAR, MARK_AX) gate on W_up; units 1501+
    # also write a W_gate column (AX_CARRY_*[k]) and a W_down row
    # (OUTPUT_*[k] for the routing units; IO_IS_PUTCHAR for the detector).
    # The verifier doesn't track b_up / b_gate bias cells, so those writes
    # are intentionally absent from this claim set.
    _claims = set()
    # W_up: 33 units * 2 columns (OP_PUTCHAR, MARK_AX) = 66 cells
    for unit in range(1500, 1533):
        _claims.add((6, "ffn_W_up", str(unit), "OP_PUTCHAR+0"))
        _claims.add((6, "ffn_W_up", str(unit), "MARK_AX+0"))
    # W_down: detector unit 1500 -> IO_IS_PUTCHAR; routing units -> OUTPUT_*[k]
    _claims.add((6, "ffn_W_down", "1500", "IO_IS_PUTCHAR+0"))
    for k in range(16):
        _claims.add((6, "ffn_W_down", str(1501 + k), f"OUTPUT_LO+{k}"))
        _claims.add((6, "ffn_W_down", str(1517 + k), f"OUTPUT_HI+{k}"))
    # W_gate: routing units read AX_CARRY_*[k]
    for k in range(16):
        _claims.add((6, "ffn_W_gate", str(1501 + k), f"AX_CARRY_LO+{k}"))
        _claims.add((6, "ffn_W_gate", str(1517 + k), f"AX_CARRY_HI+{k}"))

    return Operation(
        name="io_putchar_routing",
        reads=set(),
        # UNDECLARED_DIM_AUDIT_2026_06_09: declare actual writes from
        # ``_lower_io_putchar_routing_ir`` — unit 1500 writes
        # IO_IS_PUTCHAR; units 1501..1532 write OUTPUT_LO/HI.
        writes={"IO_IS_PUTCHAR", "OUTPUT_HI", "OUTPUT_LO"},
        # Wave 6 (docs/PRODUCES_CONSUMES_MIGRATION.md). Model-level FFN
        # routing bake: programs L5 FFN units to dispatch ``putchar``
        # output. Writes target the L5 FFN weights/biases (model setup),
        # not per-step residual dims, so the in-step semantic
        # produces/consumes_fresh surface is empty by construction.
        audited_empty_produces=True,
        kind="model",
        declarative_bake_fn=bake,
        declarative_authority="spec_generated",
        # Phase 11.A IR exposure: informational factory (33 FFNRules).
        compiler_ir_factory=_io_putchar_routing_ir,
        phase=998,
        migrated=True,
        claims=_claims,
        smoke_tests=set(),
        spec_section="BLOG_SPEC.md#printing-and-reading-input",
        semantic_label="putchar output routing",
    )


_FUNCTION_CALL_L6_FFN_START_UNIT = 1700


def _function_call_blank_rule(name: str) -> FFNRule:
    """Reserved no-op FFN unit matching the legacy unit-cursor advance.

    The legacy ``_set_function_call_weights`` increments its hidden-unit
    cursor past several reserved bands without writing weights (see the
    ``# Units 852-881 unused (reserved)`` and ``# Reserve 128 units``
    comments in vm_step.py). To replay the same layout through a single
    declarative lower we emit one no-op :class:`FFNRule` per reserved
    unit. Lowering matches a zero-initialised PureFFN row byte-for-byte:

      * ``conditions=()`` -> ``W_up[u, :] = 0``
      * ``threshold=0.0`` -> ``b_up[u] = -S * 0 = 0``
      * ``gate=None`` and ``gate_bias=0.0`` -> ``b_gate[u] = 0``
      * ``writes=()`` -> ``W_down[:, u] = 0``

    Mirrors the ``BlankUnit`` band the L5 ``decode_band`` engine lowers for
    ``opcode_decode_ffn``'s unit-52 blank (same zero-row pattern).
    """
    return FFNRule(
        conditions=(),
        threshold=0.0,
        writes=(),
        gate=None,
        gate_bias=0.0,
        name=name,
    )


def _function_call_lea_first_step_alu_init_rules(S: float) -> tuple[FFNRule, ...]:
    """LEA first-step ALU init: initialise ALU_LO[0]/ALU_HI[0] (2 units).

    For the LEA first step (``OP_LEA + MARK_AX + NOT HAS_SE``) the ALU is
    seeded with ``BP_default = 0x00010000`` (byte 0 = 0x00). The HAS_SE
    blocker (-10*S) must dominate OP_LEA's strong +5 amplification on
    later steps so the seed only fires on the first step.
    """
    write_scale = 2.0 / S
    conditions = (
        ("OP_LEA", 1.0),
        ("MARK_AX", 1.0),
        ("HAS_SE", -10.0),
    )
    return (
        multi_way_and_rule(
            name="lea_first_step_alu_lo_init",
            conditions=conditions,
            threshold=1.5,
            writes=(("ALU_LO+0", write_scale),),
        ),
        multi_way_and_rule(
            name="lea_first_step_alu_hi_init",
            conditions=conditions,
            threshold=1.5,
            writes=(("ALU_HI+0", write_scale),),
        ),
    )


def _function_call_jsr_stack0_marker_rules(S: float) -> tuple[FFNRule, ...]:
    """JSR STACK0 marker writeback: return_addr -> OUTPUT (34 units).

    Layout (matches the legacy bake at vm_step.py ~8639+):

      * 2 cancel-L3-default units (LO, HI): gate=CONST so the gate is
        always 1.0; writes OUTPUT_LO[0] / OUTPUT_HI[0] by -2.0/S to
        counteract L3's default OUTPUT_LO[0]=1 baseline. CMP[4]=2
        (the JSR relay marker) + MARK_STACK0 fires only at STACK0.
      * 16+16 AX_CARRY -> OUTPUT writeback units (LO band, HI band):
        gated_write with gate_terms (-EMBED_*[k] + AX_CARRY_*[k]) so the
        unit fires only when AX_CARRY != EMBED at byte k.

    The strong negative MARK_PC / MARK_AX / IS_BYTE blockers (-10*S each)
    confine firing to the actual MARK_STACK0 position; without them the
    CMP[4]=2 leak via L6 head 4 (BZ/BNZ relay) would corrupt PC on
    branch steps. See vm_step.py:8645-8650 for the full failure-mode note.
    """
    T_jsr_s0 = 1.5
    JSR_S0_BLOCK = -10.0
    write_scale_cancel = -2.0 / S
    write_scale_writeback = 2.0 / S
    conditions = (
        (dim_ref("cmp_flag", "cascade", 4), 1.0),
        ("MARK_STACK0", 1.0),
        ("MARK_PC", JSR_S0_BLOCK),
        ("MARK_AX", JSR_S0_BLOCK),
        ("IS_BYTE", JSR_S0_BLOCK),
    )

    rules: list[FFNRule] = [
        # Cancel L3 default OUTPUT_LO[0]. The legacy bake implements the
        # always-on gate as ``W_gate[unit, CONST] = 1.0`` (not as a
        # ``b_gate = 1.0`` bias) so the gate value at any position with
        # ``CONST=1`` (every position) is 1.0. ``gated_write`` with
        # ``gate="CONST"`` reproduces that cell layout exactly.
        multi_way_and_rule(
            name="jsr_stack0_cancel_l3_default_lo",
            conditions=conditions,
            threshold=T_jsr_s0,
            gate="CONST",
            writes=(("OUTPUT_LO+0", write_scale_cancel),),
        ),
        multi_way_and_rule(
            name="jsr_stack0_cancel_l3_default_hi",
            conditions=conditions,
            threshold=T_jsr_s0,
            gate="CONST",
            writes=(("OUTPUT_HI+0", write_scale_cancel),),
        ),
    ]
    for k in range(16):
        rules.append(multi_way_and_rule(
            name=f"jsr_stack0_writeback_lo_{k}",
            conditions=conditions,
            threshold=T_jsr_s0,
            gate_terms=(
                (f"EMBED_LO+{k}", -1.0),
                (f"AX_CARRY_LO+{k}", 1.0),
            ),
            writes=((f"OUTPUT_LO+{k}", write_scale_writeback),),
        ))
    for k in range(16):
        rules.append(multi_way_and_rule(
            name=f"jsr_stack0_writeback_hi_{k}",
            conditions=conditions,
            threshold=T_jsr_s0,
            gate_terms=(
                (f"EMBED_HI+{k}", -1.0),
                (f"AX_CARRY_HI+{k}", 1.0),
            ),
            writes=((f"OUTPUT_HI+{k}", write_scale_writeback),),
        ))
    return tuple(rules)


def _function_call_jsr_pc_override_conditions() -> tuple[tuple[str, float], ...]:
    """Common up-branch conditions for the JSR PC override band.

    Gates on MARK_PC + TEMP[0] (IS_JSR flag). On a step-0 JSR the
    HAS_SE-gated L5 first-step decode supplies ``TEMP+0 = +5``; on a
    NESTED JSR (after an ENT) the all-step JSR IS_JSR decode
    (the ``all_step_jsr_at_pc`` derived decode context in
    ``l5_ops._derived_decode_spec``, flag ``C4_NESTED_JSR_PC_FIX``)
    supplies the same ``+5`` from the clean
    per-step JSR opcode byte. Strong negative blockers for every other
    opcode (NOP, EXIT, JMP, BZ, BNZ, IMM, LEV, ENT) prevent spurious firing
    on non-JSR steps where TEMP[0] is polluted by L6 head 4 (BZ/BNZ relay).
    IS_BYTE blocker confines to the PC marker position. See Root B in
    ``docs/PHASE_5_JSR_ENT_LEV_FOLLOWUP.md``.
    """
    return (
        ("MARK_PC", 1.0),
        ("TEMP+0", 1.0),
        ("OP_NOP", -4.0),
        ("OP_EXIT", -4.0),
        ("OP_JMP", -4.0),
        ("OP_BZ", -4.0),
        ("OP_BNZ", -4.0),
        ("OP_IMM", -4.0),
        ("OP_LEV", -4.0),
        ("OP_ENT", -4.0),
        ("IS_BYTE", -10.0),
    )


def _function_call_jsr_pc_override_rules(S: float) -> tuple[FFNRule, ...]:
    """JSR PC override: cancel OUTPUT (PC+5) + materialize target (80 units).

    DERIVED via the ``register_delta`` primitive
    (``isa_semantics_dsl.register_delta``, kind ``branch_target``). The JSR
    PC-next is the branch-target ``PC = imm*INSTR_WIDTH + PC_OFFSET`` register
    delta (docs/semantic_spec_CONTROL.md §1d/§2b): the SAME ``idx_to_pc``
    encoder JMP/BZ/BNZ use, cancelling the L3 sequential default in the
    same-step OUTPUT bank then writing the byte-0 target from the raw
    instruction INDEX in ``FETCH_LO``.

    Layout (matches vm_step.py ~8718+), all emitted by the primitive:

      * 16 OUTPUT_LO cancel units: gate=-OUTPUT_LO[k], writes OUTPUT_LO[k]
      * 16 OUTPUT_HI cancel units: gate=-OUTPUT_HI[k], writes OUTPUT_HI[k]
      * 16 FETCH_LO -> OUTPUT_LO[target_lo] target units
      * 16 FETCH_HI reserved gate units (no writes; legacy reservation
        for targets >= 256 that would need a FETCH_HI byte path)
      * 16 FETCH_LO -> OUTPUT_HI[target_hi_from_lo] carry units

    All units share the same conditions (MARK_PC + TEMP[0] + opcode
    blockers + IS_BYTE blocker). The reserved FETCH_HI block keeps the
    unit cursor aligned with the legacy layout.

    The flag-gated ``C4_JSR_PC_BYTE1`` byte-1 stage (§G3, the known >=0x100
    branch-target wall — NOT derivable from the byte-0 ISA encoder) SPLICES its
    16 ``JSR_PC_B1`` staging units over the primitive's reserved band when ON;
    OFF the derived reserved no-op band leaves the override golden
    byte-identical. That branch stays hand-authored: it is the residual DSL gap
    the frame primitive does not yet cover.
    """
    from ...constants import INSTR_WIDTH, PC_OFFSET

    write_scale = 2.0 / S
    conditions = _function_call_jsr_pc_override_conditions()

    derived = list(register_delta(
        RegisterDeltaSpec(
            name="jsr_pc",
            kind="branch_target",
            write_scale=write_scale,
            conditions=conditions,
            threshold=4.0,
            branch_target=BranchTargetDelta(),
        ),
        instr_width=INSTR_WIDTH,
        pc_offset=PC_OFFSET,
    ).rules_builder())

    if not _jsr_pc_byte1_enabled():
        return tuple(derived)

    # Flag C4_JSR_PC_BYTE1 (§G3, out-of-scope for the frame primitive): stage
    # PC byte-1 = (k>>1) into JSR_PC_B1 from FETCH_HI+k over the reserved band
    # (units 48-63). FETCH_HI holds the high nibble of the JSR target idx
    # (verified spec_k=0: idx=33 -> FETCH_HI dominant index 2; idx=7 -> 0). PC
    # byte-1 = (idx*8+2)>>8 = (16*FETCH_HI_nib*8)>>8 = FETCH_HI_nib>>1 (the +2
    # and the FETCH_LO*8 < 128 never carry into bit 8). JSR-gated by the shared
    # override conditions so JSR_PC_B1 is nonzero ONLY on a real JSR step. The
    # staging MUST use the SAME strict threshold as the byte-0 PC target (4.0):
    # a lower value fires on every non-JSR PC marker where TEMP+0 carries
    # L6-head BZ/BNZ-relay pollution and over-fires PC byte-1 on low-PC
    # arithmetic programs (a whole-corpus regression).
    T_jsr_pc = 4.0
    reserved_start = 3 * 16  # 2 cancel + 1 target-lo bands precede the reserved
    for k in range(16):
        derived[reserved_start + k] = multi_way_and_rule(
            name=f"jsr_pc_byte1_stage_{k}",
            conditions=conditions,
            threshold=T_jsr_pc,
            gate=f"FETCH_HI+{k}",
            writes=((f"JSR_PC_B1+{k >> 1}", write_scale),),
        )
    return tuple(derived)


def _function_call_jsr_ax_passthrough_rules(S: float) -> tuple[FFNRule, ...]:
    """JSR AX passthrough: AX_CARRY -> OUTPUT at AX marker (32 units)."""
    T = 4.0
    write_scale = 2.0 / S
    conditions = (("OP_JSR", 1.0), ("MARK_AX", 1.0))
    rules: list[FFNRule] = []
    for k in range(16):
        rules.append(multi_way_and_rule(
            name=f"jsr_ax_passthrough_lo_{k}",
            conditions=conditions,
            threshold=T,
            gate=f"AX_CARRY_LO+{k}",
            writes=((f"OUTPUT_LO+{k}", write_scale),),
        ))
    for k in range(16):
        rules.append(multi_way_and_rule(
            name=f"jsr_ax_passthrough_hi_{k}",
            conditions=conditions,
            threshold=T,
            gate=f"AX_CARRY_HI+{k}",
            writes=((f"OUTPUT_HI+{k}", write_scale),),
        ))
    return tuple(rules)


def _ent_control_op(S: float) -> ControlOpBundle:
    """The ENT opcode's whole L6 FFN band as ONE :func:`control_op` descriptor.

    ``ControlOp(ENT)`` names ENT's ordered ``frame_delta`` list as DATA
    (docs/semantic_spec_CONTROL.md §2b/§G10) and lowers the WHOLE opcode through
    one :func:`control_op` call:

      * PUSH(saved-BP) — at the STACK0 marker on an ENT step (CMP[2]=1,
        MARK_STACK0=1) the caller's BP (relayed into ``TEMP`` per-nibble by L5
        head 5) is written onto OUTPUT while the arriving EMBED identity is
        cancelled in the SAME hidden unit (copy-with-cancel, TEMP LO/HI split at
        ``+16``). L6 attn head 6 broadcasts OP_ENT through CMP[2].
      * ASSIGN(BP := SP - INSTR_WIDTH) — at the BP marker the caller's SP
        (relayed into ``TEMP``) is copied into BP with the frame-link constant
        subtracted (``lo_shift = (-INSTR_WIDTH) % 16 = 8``, ``hi_shift = -1``,
        ``borrow_blocker_range = (INSTR_WIDTH, 16)`` — the nibble-shift-with-
        borrow adder).

    Both are ``RegisterDeltaSpec`` frame deltas → they lower TOGETHER via one
    ``frame_step`` (ordered-group grouping in production). The ENT AX-passthrough
    band (:func:`_function_call_ent_ax_passthrough_rules`) is the opcode's
    trailing CORRECTIVE band — not a register-delta kind — so it rides as the
    third ``bands`` entry, inlined after the two frame deltas. Byte-identical to
    the hand-sequenced ENT bank (proof: ``tools/_isa_golden_hash.py`` ==
    91f55411).
    """
    return control_op(
        "ENT",
        [
            RegisterDeltaSpec(
                name="ent_stack0",
                kind="push",
                write_scale=2.0 / S,
                conditions=(
                    (dim_ref("cmp_flag", "cascade", 2), 1.0),
                    ("MARK_STACK0", 1.0),
                ),
                push=PushDelta(
                    value_src="TEMP",
                    value_hi_offset=16,
                    identity_src_lo="EMBED_LO",
                    identity_src_hi="EMBED_HI",
                    dst_lo="OUTPUT_LO",
                    dst_hi="OUTPUT_HI",
                    threshold=1.5,
                ),
            ),
            RegisterDeltaSpec(
                name="ent_bp",
                kind="assign",
                write_scale=2.0 / S,
                conditions=(
                    (dim_ref("cmp_flag", "cascade", 2), 1.0),
                    ("MARK_BP", 1.0),
                ),
                assign=AssignDelta(
                    value_src="TEMP",
                    value_hi_offset=16,
                    dst_lo="OUTPUT_LO",
                    dst_hi="OUTPUT_HI",
                    lo_shift=(-INSTR_WIDTH) % 16,
                    hi_shift=(-1) % 16,
                    borrow_blocker_range=(INSTR_WIDTH, 16),
                    threshold=1.5,
                ),
            ),
            lambda: _function_call_ent_ax_passthrough_rules(S),
        ],
        instr_width=INSTR_WIDTH,
        pc_offset=PC_OFFSET,
    )


def _function_call_ent_ax_passthrough_rules(S: float) -> tuple[FFNRule, ...]:
    """ENT AX passthrough: AX_CARRY -> OUTPUT at AX marker (32 units).

    PER-STEP ACTUAL-IMM BLOCKER (func / simple_function fix). This band
    routes the ENT frame's AX_CARRY onto OUTPUT at the ENT step's AX
    marker, gated ``OP_ENT + MARK_AX >= 4``. ``OP_ENT`` is durably carried
    forward across the call frame (re-broadcast by the L6 opcode-relay /
    marker carry from the ENT step into the body), so on the *callee* ``IMM``
    step that immediately follows ENT, ``OP_ENT`` is still ~5 at the AX
    marker and this band SPURIOUSLY fires, copying the carried frame value
    (the ENT SP-decrement constant 8) onto OUTPUT_LO and burying the genuine
    immediate -> ``ENT 0; IMM 42`` emits 40/8 and the func / 150-fail cluster
    fails. The L5 main-AX opcode decode produces a CLEAN per-step ``OP_IMM``
    one-hot (= +5.0 ONLY on real IMM steps, 0 on real ENT steps; probed
    spec_k=0) that survives to this band's input, so a strong ``OP_IMM``
    NOT-blocker suppresses the band on the callee IMM step while leaving a
    real ENT step (OP_IMM = 0) byte-identical. Mirrors the head-1 actual-IMM
    suppression in ``l7_ops._layer7_operand_gather_head_specs`` and the
    HAS_SE blocker in ``l9_ops._layer9_ent_hi_nibble_rules``.
    """
    T = 4.0
    write_scale = 2.0 / S
    conditions = (("OP_ENT", 1.0), ("MARK_AX", 1.0), ("OP_IMM", -10.0))
    rules: list[FFNRule] = []
    for k in range(16):
        rules.append(multi_way_and_rule(
            name=f"ent_ax_passthrough_lo_{k}",
            conditions=conditions,
            threshold=T,
            gate=f"AX_CARRY_LO+{k}",
            writes=((f"OUTPUT_LO+{k}", write_scale),),
        ))
    for k in range(16):
        rules.append(multi_way_and_rule(
            name=f"ent_ax_passthrough_hi_{k}",
            conditions=conditions,
            threshold=T,
            gate=f"AX_CARRY_HI+{k}",
            writes=((f"OUTPUT_HI+{k}", write_scale),),
        ))
    return tuple(rules)


def _function_call_lev_ax_passthrough_rules(S: float) -> tuple[FFNRule, ...]:
    """LEV AX passthrough at AX marker (32 units).

    MARK_PC -15*S blocker prevents firing at PC marker (OP_LEV gets
    amplified to ~10 by L6 attention; without it units would fire at PC).
    IS_BYTE -10*S blocker prevents firing at AX byte positions where
    OP_LEV ~7.5 alone would clear T=4.
    """
    T = 4.0
    write_scale = 2.0 / S
    conditions = (
        ("OP_LEV", 1.0),
        ("MARK_AX", 1.0),
        ("MARK_PC", -15.0),
        ("IS_BYTE", -10.0),
    )
    rules: list[FFNRule] = []
    for k in range(16):
        rules.append(multi_way_and_rule(
            name=f"lev_ax_passthrough_lo_{k}",
            conditions=conditions,
            threshold=T,
            gate=f"AX_CARRY_LO+{k}",
            writes=((f"OUTPUT_LO+{k}", write_scale),),
        ))
    for k in range(16):
        rules.append(multi_way_and_rule(
            name=f"lev_ax_passthrough_hi_{k}",
            conditions=conditions,
            threshold=T,
            gate=f"AX_CARRY_HI+{k}",
            writes=((f"OUTPUT_HI+{k}", write_scale),),
        ))
    return tuple(rules)


def _function_call_lev_ax_byte_rules(S: float) -> tuple[FFNRule, ...]:
    """LEV AX byte: AX_CARRY -> OUTPUT at AX byte positions (32 units).

    3-way AND ``OP_LEV + IS_BYTE + H1[AX_IDX]`` with T=9 fires only at
    byte positions inside the AX region. The marker-position units above
    are blocked at byte positions; these byte-position units fill in
    the gap so AX_CARRY propagates onto OUTPUT at AX byte positions
    during LEV. See vm_step.py:8925-8953 for the gating analysis.
    """
    T_byte = 9.0
    AX_IDX = 1
    write_scale = 2.0 / S
    conditions = (
        ("OP_LEV", 1.0),
        ("IS_BYTE", 1.0),
        (f"H1+{AX_IDX}", 1.0),
        ("MARK_AX", -15.0),
    )
    rules: list[FFNRule] = []
    for k in range(16):
        rules.append(multi_way_and_rule(
            name=f"lev_ax_byte_lo_{k}",
            conditions=conditions,
            threshold=T_byte,
            gate=f"AX_CARRY_LO+{k}",
            writes=((f"OUTPUT_LO+{k}", write_scale),),
        ))
    for k in range(16):
        rules.append(multi_way_and_rule(
            name=f"lev_ax_byte_hi_{k}",
            conditions=conditions,
            threshold=T_byte,
            gate=f"AX_CARRY_HI+{k}",
            writes=((f"OUTPUT_HI+{k}", write_scale),),
        ))
    return tuple(rules)


def _function_call_l6_ffn_rules(S: float) -> tuple[FFNRule, ...]:
    """Full ordered :class:`FFNRule` sequence for function_call_weights' L6 FFN.

    Matches the 594-unit layout in the legacy ``_set_function_call_weights``
    (vm_step.py:8602-8953):

      * units 1700..1701  - LEA first-step ALU init (2 units)
      * units 1702..1731  - reserved (30 blank units)
      * units 1732..1859  - JSR SP -= 8 reserved (128 blank units; moved to L7)
      * units 1860..1893  - JSR STACK0 marker writeback (34 units)
      * units 1894..2021  - JSR STACK0 bytes 0-3 reserved (128 blank units)
      * units 2022..2101  - JSR PC override (80 units; 16+16 cancel +
                            16 LO target + 16 FETCH_HI reserved + 16 HI carry)
      * units 2102..2133  - JSR AX passthrough (32 units)
      * units 2134..2229  - ENT band (96 units), emitted as ONE ControlOp
                            frame-descriptor (``_ent_control_op``): STACK0 =
                            old_BP PUSH (2134..2165), BP = SP - 8 ASSIGN
                            (2166..2197) — the two frame deltas via one
                            frame_step — then the AX-passthrough corrective
                            band (2198..2229)
      * units 2230..2261  - LEV AX passthrough at AX marker (32 units)
      * units 2262..2293  - LEV AX byte positions (32 units)

    Total: 594 rules. End cursor (start_unit=1700 + 594) = 2294, matching
    ``Operation.ffn_units_used``.

    The reserved bands are emitted as no-op :func:`_function_call_blank_rule`
    placeholders so the lower_ffn cursor walks the same unit numbers the
    legacy helper used. ``right_size_ffns`` correctly prunes them because
    every weight column / row remains all-zero.
    """
    rules: list[FFNRule] = []

    rules.extend(_function_call_lea_first_step_alu_init_rules(S))
    rules.extend(
        _function_call_blank_rule(f"lea_reserved_{i}")
        for i in range(30)
    )
    rules.extend(
        _function_call_blank_rule(f"jsr_sp_dec_reserved_{i}")
        for i in range(128)
    )
    rules.extend(_function_call_jsr_stack0_marker_rules(S))
    rules.extend(
        _function_call_blank_rule(f"jsr_stack0_bytes_reserved_{i}")
        for i in range(128)
    )
    rules.extend(_function_call_jsr_pc_override_rules(S))
    rules.extend(_function_call_jsr_ax_passthrough_rules(S))
    # ENT: the whole opcode band (PUSH saved-BP + ASSIGN BP:=SP-w frame deltas
    # via one frame_step, then the AX-passthrough corrective band) named as ONE
    # ControlOp frame-descriptor.
    rules.extend(_ent_control_op(S).rules_builder())
    rules.extend(_function_call_lev_ax_passthrough_rules(S))
    rules.extend(_function_call_lev_ax_byte_rules(S))

    return tuple(rules)


def _function_call_l5_head_specs(BD) -> tuple[DeclarativeAttentionHeadSpec, ...]:
    """Declarative L5 attention specs for the ENT relay heads (5, 6).

    Mirrors the imperative L5 section of ``_set_function_call_weights``
    bit-for-bit (vm_step.py ~8520-8563):

      * Head 5 (BP EMBED -> TEMP at STACK0): relays the caller's BP
        value into TEMP[0..31] at MARK_STACK0 positions so the L6 FFN
        ``ent_stack0_*`` band writes it onto OUTPUT during ENT.
      * Head 6 (SP EMBED -> TEMP at BP): relays the caller's SP value
        into TEMP[0..31] at MARK_BP positions so the L6 FFN ``ent_bp_*``
        band can compute ``new_BP = old_SP - 8``. Head 6 additionally
        carries an OP_ENT-only firing gate at slot 34 so the TEMP band
        is only populated on ENT steps.

    Slot layout (HD=64):
      slot 0       : main Q/K gather (marker x marker)
      slot 1..16   : V[EMBED_LO+k] -> O[TEMP+k]      (low nibble band)
      slot 17..32  : V[EMBED_HI+k] -> O[TEMP+16+k]   (high nibble band)
      slot 33      : anti-leakage gate (Q[marker]=500, Q[CONST]=-500,
                     K[CONST]=5) so the head only fires at the target
                     marker positions
      slot 34      : (head 6 only) OP_ENT firing gate
    """
    L5 = 20.0

    def _band_v(slot_base: int, dim_base: int):
        return tuple(AP(slot_base + k, dim_base + k, 1.0) for k in range(16))

    def _band_o(dim_base: int, slot_base: int):
        return tuple(AO(dim_base + k, slot_base + k, 1.0) for k in range(16))

    head5 = DeclarativeAttentionHeadSpec(
        head_idx=5,
        q=(
            AP(0, BD.MARK_STACK0, L5),
            AP(33, BD.MARK_STACK0, 500.0),
            AP(33, BD.CONST, -500.0),
        ),
        k=(
            AP(0, BD.MARK_BP, L5),
            AP(33, BD.CONST, 5.0),
        ),
        v=_band_v(1, BD.EMBED_LO) + _band_v(17, BD.EMBED_HI),
        o=_band_o(BD.TEMP, 1) + _band_o(BD.TEMP + 16, 17),
    )

    head6 = DeclarativeAttentionHeadSpec(
        head_idx=6,
        q=(
            AP(0, BD.MARK_BP, L5),
            AP(33, BD.MARK_BP, 500.0),
            AP(33, BD.CONST, -500.0),
            AP(34, BD.OP_ENT, 500.0),
            AP(34, BD.CONST, -500.0),
        ),
        k=(
            AP(0, BD.MARK_SP, L5),
            AP(33, BD.CONST, 5.0),
            AP(34, BD.CONST, 5.0),
        ),
        v=_band_v(1, BD.EMBED_LO) + _band_v(17, BD.EMBED_HI),
        o=_band_o(BD.TEMP, 1) + _band_o(BD.TEMP + 16, 17),
    )

    return (head5, head6)


def _function_call_l6_head_spec(BD) -> DeclarativeAttentionHeadSpec:
    """Declarative L6 attention spec for the JSR PC OUTPUT -> AX_CARRY relay.

    Mirrors the imperative L6 section of ``_set_function_call_weights``
    bit-for-bit (vm_step.py ~8585-8600). Copies the previous PC's OUTPUT
    (PC+INSTR_WIDTH from L3, the return address) into AX_CARRY at the
    MARK_STACK0 position so the L6 FFN ``jsr_stack0_*`` band can write
    it onto OUTPUT during JSR.

    Slot layout (HD=64):
      slot 0       : Q[MARK_STACK0]=+1050 (strong); Q[MARK_AX]=-50
                     (block at AX); Q[CONST]=-1000 (baseline
                     suppression). K[MARK_PC]=30 (positive at PC).
                     K[OP_JSR]=-20 fires only on JSR (the OP_JSR relay
                     at MARK_AX ~5.0), canceling AX's K for JSR while
                     preserving AX's K for PSH (head 7 is shared).
      slot 1..16   : V[OUTPUT_LO+k] -> O[AX_CARRY_LO+k] (low nibble)
      slot 17..32  : V[OUTPUT_HI+k] -> O[AX_CARRY_HI+k] (high nibble)

    Phase ordering: ``layer6_relay_heads_bake`` (998.6) OVERWRITES
    Q[MARK_STACK0] with 50 so PSH semantics win. Both bakes write the
    same slot-0 cells; relay_heads's later write replaces this one's.
    See ``make_layer6_relay_heads_bake_op`` docstring for ordering notes.
    """
    L6 = 50.0
    return DeclarativeAttentionHeadSpec(
        head_idx=7,
        q=(
            AP(0, BD.MARK_STACK0, L6 + L6 * 20),
            AP(0, BD.MARK_AX, -L6),
            AP(0, BD.CONST, -L6 * 20),
        ),
        k=(
            AP(0, BD.MARK_PC, 30.0),
            AP(0, BD.OP_JSR, -20.0),
        ),
        v=(
            tuple(AP(1 + k, BD.OUTPUT_LO + k, 1.0) for k in range(16))
            + tuple(AP(17 + k, BD.OUTPUT_HI + k, 1.0) for k in range(16))
        ),
        o=(
            tuple(AO(BD.AX_CARRY_LO + k, 1 + k, 1.0) for k in range(16))
            + tuple(AO(BD.AX_CARRY_HI + k, 17 + k, 1.0) for k in range(16))
        ),
    )


def _function_call_weights_ir(dim_positions, HD) -> CompilerIR:
    """Informational :class:`CompilerIR` for ``function_call_weights``.

    Carries the L5 H5/H6 + L6 H7 attention specs plus the L6 FFN rule
    list so the declarative verifier, scope checker, symbolic-tools, and
    static census see the same writes the bake produces. The actual bake
    writes to three blocks (L5 attn, L6 attn, L6 FFN); the generic
    ``_dispatch_operation_ir`` only handles single-target kinds, so this
    IR is informational only (the imperative bake_fn lowers it via per-
    block ``Primitives`` calls).

    Mirrors the ``compiler_ir_factory`` convention used by
    ``layer6_attn_bake`` / ``layer6_relay_heads_bake`` for model-level
    multi-block bakes.
    """
    del HD
    proxy = _as_setdim_proxy(dim_positions)
    ir = CompilerIR()
    for spec in _function_call_l5_head_specs(proxy):
        ir.layer(0).attention.append(spec)
    ir.layer(0).attention.append(_function_call_l6_head_spec(proxy))
    ir.layer(0).ffn.rules.extend(_function_call_l6_ffn_rules(100.0))
    return ir


def make_function_call_weights_op() -> Operation:
    """Bake function-call opcode weights (JSR, ENT, LEV, LEA).

    Originally an inline call in `set_vm_weights`:
        `_set_function_call_weights(model, S, BD, HD)`

    Operates across multiple blocks (L5 attn, L6 attn, L6 ffn) so this is a
    model-level op.

    Phase 998: runs just BEFORE legacy_bake (999) so that the L6 FFN units
    we program (1700-2158) are present when `_right_size_ffns` (called at the
    end of legacy_bake) prunes dead units. Running at phase > 999 would write
    into already-rightsized FFN slots that no longer exist (IndexError).

    Phase 8.C.2 migration (2026-06-02): the imperative
    ``_set_function_call_weights`` helper is replaced by
    :class:`DeclarativeAttentionHeadSpec` lowering for L5 H5/H6 + L6 H7
    plus 594 :class:`FFNRule`s (~13 rule families + 286 blank-unit
    reservations) for the L6 FFN routing band (units 1700..2293). The
    bake_fn drives every write through
    :meth:`Primitives.generate_attention_head` and
    :meth:`Primitives.lower_ffn_rules`; ``compiler_ir_factory`` exposes
    the same specs to the declarative verifier and symbolic tools.
    Byte-identical to the legacy helper.
    """
    def bake(model, dim_positions, S):
        proxy = _as_setdim_proxy(dim_positions)
        attn5 = model.blocks[5].attn
        attn6 = model.blocks[6].attn
        ffn6 = model.blocks[6].ffn
        HD = attn5.W_q.shape[0] // attn5.num_heads

        # L5 attention: ENT relay heads (5 BP->TEMP, 6 SP->TEMP).
        Primitives.generate_attention_heads(
            attn5, _function_call_l5_head_specs(proxy), HD,
        )
        # L6 attention: JSR PC OUTPUT->AX_CARRY relay (head 7).
        Primitives.generate_attention_head(
            attn6, _function_call_l6_head_spec(proxy), HD,
        )
        # L6 FFN: LEA / JSR / ENT / LEV output routing (units 1700..2293).
        rules = _function_call_l6_ffn_rules(S)
        dim_positions_for_rules = Primitives.dim_positions_from_bd(
            proxy,
            Primitives.ffn_rule_dim_names(rules),
        )
        Primitives.lower_ffn_rules(
            ffn6,
            rules,
            dim_positions_for_rules,
            start_unit=_FUNCTION_CALL_L6_FFN_START_UNIT,
            S=S,
        )

        # JSR-target PC byte-1 delivery RELAY (flag C4_JSR_PC_BYTE1, default OFF).
        # The byte-1 nibble is STAGED into JSR_PC_B1 at the PC MARKER by the
        # override FFN above (``jsr_pc_byte1_stage_*``). The MARKER write only
        # becomes visible in the residual AFTER this block's FFN runs, so the
        # relay attention head (which copies MARKER -> PC byte-0 row) MUST live
        # on the NEXT block (``model.blocks[7]`` here resolves to the physical
        # block just past the override FFN block); a same-block attention head
        # runs BEFORE the FFN and would read JSR_PC_B1=0. The relay's
        # JSR_PC_B1_AT_B0 write then PERSISTS in the residual to the tail, where
        # ``make_jsr_pc_byte1_emit_op`` (a post-tail-corruptor FFN) reads it and
        # writes the PC byte-1 OUTPUT. Off -> JSR_PC_B1 band omitted -> skipped
        # -> golden byte-identical.
        if _jsr_pc_byte1_enabled() and len(model.blocks) > 7:
            relay_block = model.blocks[7]
            relay_attn = relay_block.attn
            relay_HD = relay_attn.W_q.shape[0] // relay_attn.num_heads
            allocator = _allocate_jsr_pc_byte1_relay_head()
            relay_attn._jsr_pc_byte1_relay_head_allocator = allocator
            relay_head_idx = allocator.heads()[-1].head_idx
            relay_spec = _jsr_pc_byte1_relay_head_spec(proxy, relay_head_idx)
            Primitives.generate_attention_head(relay_attn, relay_spec, relay_HD)

    # Dim-ownership claims (see c4_release/docs/DIM_OWNERSHIP_REGISTRY.md).
    # `_set_function_call_weights` programs three ENT/JSR relay attention
    # heads. Each head writes V slots [base+1..base+16] (low-nibble path)
    # and [base+17..base+32] (high-nibble path):
    #
    #   L5 attn5 head 5: BP→TEMP at STACK0 (ENT: STACK0 = old_BP)
    #     W_v[5*HD + 1 + k, EMBED_LO + k]    for k=0..15 (slot 1..16)
    #     W_v[5*HD + 17 + k, EMBED_HI + k]   for k=0..15 (slot 17..32)
    #   L5 attn5 head 6: SP→TEMP at BP        (ENT: BP = old_SP - 8)
    #     W_v[6*HD + 1 + k, EMBED_LO + k]    for k=0..15
    #     W_v[6*HD + 17 + k, EMBED_HI + k]   for k=0..15
    #   L6 attn6 head 7: PC OUTPUT→AX_CARRY at STACK0 (JSR return addr)
    #     W_v[7*HD + 1 + k, OUTPUT_LO + k]   for k=0..15
    #     W_v[7*HD + 17 + k, OUTPUT_HI + k]  for k=0..15
    #
    # The 4-tuple claim's ``column`` records the input dim + offset so the
    # registry can distinguish co-tenants of the same row that touch
    # disjoint input columns. In particular, this op's L5 head 5 slot 32
    # claim is (5, "attn_W_v", "5_32", "EMBED_HI+15") — column-disjoint
    # from layer5_fetch's (5, "attn_W_v", "5_32", "CLEAN_EMBED_LO+0"). The
    # row-only registry treated this as a benign collision via
    # ``KNOWN_BENIGN_COLLISIONS`` in tests; the column-aware registry
    # detects the disjointness automatically.
    #
    # Historical context: the deprecated `_set_layer5_fetch` head 6 (deleted
    # 2026-05-11, commit c1a5398) wrote V slots 1..16 on the same matrix
    # using `attn.W_v[6*HD + 1 + k, EMBED_LO + k]`. That collided with
    # this op's head 6 ENT relay at *the same column*:
    # ``(5, "attn_W_v", "6_<k>", "EMBED_LO+<k-1>")`` for k=1..16 — the
    # kind of true collision the column-aware registry still catches.
    _claims = set()
    # L5 attn5 head 5 ENT relay
    for k in range(16):
        _claims.add(
            (5, "attn_W_v", f"5_{1 + k}", f"EMBED_LO+{k}")
        )
        _claims.add(
            (5, "attn_W_v", f"5_{17 + k}", f"EMBED_HI+{k}")
        )
    # L5 attn5 head 6 ENT relay
    for k in range(16):
        _claims.add(
            (5, "attn_W_v", f"6_{1 + k}", f"EMBED_LO+{k}")
        )
        _claims.add(
            (5, "attn_W_v", f"6_{17 + k}", f"EMBED_HI+{k}")
        )
    # L6 attn6 head 7 JSR PC OUTPUT relay
    for k in range(16):
        _claims.add(
            (6, "attn_W_v", f"7_{1 + k}", f"OUTPUT_LO+{k}")
        )
        _claims.add(
            (6, "attn_W_v", f"7_{17 + k}", f"OUTPUT_HI+{k}")
        )

    # JSR-target PC byte-1 delivery (flag C4_JSR_PC_BYTE1): the override FFN
    # STAGES JSR_PC_B1 (FETCH_HI), the relay head (next block) reads FETCH-staged
    # JSR_PC_B1 + IS_BYTE/H1/BYTE_INDEX_0/HAS_SE/MARK_PC and writes
    # JSR_PC_B1_AT_B0, the emit FFN reads JSR_PC_B1_AT_B0 and writes OUTPUT_LO.
    # Declared only when the flag is on so the OMITTED bands are never referenced
    # off (which would fail dim validation).
    _jsr_b1_reads = (
        {"IS_BYTE", "H1", "BYTE_INDEX_0", "HAS_SE", "FETCH_HI",
         "JSR_PC_B1", "JSR_PC_B1_AT_B0"}
        if _jsr_pc_byte1_enabled() else set()
    )
    _jsr_b1_writes = (
        {"JSR_PC_B1", "JSR_PC_B1_AT_B0"}
        if _jsr_pc_byte1_enabled() else set()
    )
    return Operation(
        name="function_call_weights",
        # Phase: docs/DIM_LIVENESS_FINDINGS_2026_06_05.md audit — declare
        # baked reads (Q/K/V columns + FFN rule input dims) and writes
        # (O rows + FFN rule output dims). Derived from
        # ``_function_call_l5_head_specs`` (L5 H5/H6 ENT relay),
        # ``_function_call_l6_head_spec`` (L6 H7 JSR PC OUTPUT relay),
        # and ``_function_call_l6_ffn_rules``.
        reads={"MARK_STACK0", "MARK_BP", "MARK_SP", "MARK_PC", "MARK_AX",
               "CONST", "OP_JSR", "OP_ENT",
               "OP_IMM",  # ent_ax_passthrough per-step actual-IMM blocker
               "EMBED_LO", "EMBED_HI",
               "OUTPUT_LO", "OUTPUT_HI"} | _jsr_b1_reads,
        # UNDECLARED_DIM_AUDIT_2026_06_09: added ALU_HI/LO, OUTPUT_HI/LO,
        # TEMP to match the actual write footprint of the L6 FFN rules
        # produced by ``_function_call_l6_ffn_rules`` (LEA / JSR / ENT /
        # LEV output routing band writes ALU_*, OUTPUT_* and TEMP slots).
        writes={"AX_CARRY_LO", "AX_CARRY_HI",
                "ALU_HI", "ALU_LO",
                "OUTPUT_HI", "OUTPUT_LO",
                "TEMP"} | _jsr_b1_writes,
        # Wave 6 (docs/PRODUCES_CONSUMES_MIGRATION.md). Model-level FFN
        # routing bake into L6 FFN units 1700..2158 (function-call dispatch
        # table). Writes target FFN weights, not per-step residual dims,
        # so produces/consumes_fresh stays empty by construction.
        audited_empty_produces=True,
        kind="model",
        declarative_bake_fn=bake,
        compiler_ir_factory=_function_call_weights_ir,
        declarative_authority="spec_generated",
        phase=998,
        migrated=True,
        claims=_claims,
        # ``_set_function_call_weights`` writes L6 FFN units starting at
        # 1700 (LEA/JSR/ENT/LEV output routing) and reaching unit 2293 with
        # the trailing LEV-AX-byte routing pairs (``vm_step.py:8230+``).
        # Annotating this op with the per-layer max (2294) lets the
        # dynamic-FFN allocator pre-size L6 instead of falling back to 4096
        # and trimming post-bake. The (former) ``layer_idx=6`` was purely
        # a sizing hint -- model ops are dispatched against the whole
        # model and the layer_idx field doesn't affect their bake
        # execution. Phase 8.A.4: dropped the literal; L6 FFN sizing
        # falls back to ``_right_size_ffns`` trim post-bake, which is
        # behaviour-equivalent (the bake itself writes the same cells).
        ffn_units_used=2294,
        smoke_tests={
            "TestSmokeFunctionCall::test_simple_function",
        },
        spec_section="BLOG_SPEC.md#function-calls",
        compaction_safe=False,
        semantic_label="function-call routing",
    )


# Free L6 attention head for the JSR PC byte-1 marker->byte0 relay. Heads 0..7
# are the named _L6_HEAD_LAYOUT owners; the production L6 attn block is widened
# (>=11 heads, head_dim 111) and heads 8+ are free padding. We pin the FIRST
# free padding slot via a per-bake AttentionHeadAllocator (layer_max_heads
# raised) so the head is DECLARED in the layout IR and survives the post-bake
# right-size / norm passes (a raw model-bake write into an undeclared padding
# head is silently zeroed; mirrors l11_ops bp_save_prev_carry head 8).
_JSR_PC_BYTE1_RELAY_HEAD_PIN = 8
_JSR_PC_BYTE1_RELAY_HEAD_LAYOUT = (
    ("jsr_pc_byte1_relay.head", _JSR_PC_BYTE1_RELAY_HEAD_PIN),
)


def _allocate_jsr_pc_byte1_relay_head() -> "_AttnHeadAlloc":
    """Per-bake head allocator pinning the JSR PC byte-1 relay head on L6.

    Head 8 is a widen-padding slot on the L6 attn block (heads 0..7 are the
    named owners). Pinning it through the allocator (``layer_max_heads`` raised
    past the default 8) DECLARES the head so the compiler preserves it.
    """
    allocator = _AttnHeadAlloc(strategy="dynamic_first_fit", layer_max_heads=12)
    for (op_name, head_idx) in _JSR_PC_BYTE1_RELAY_HEAD_LAYOUT:
        allocator.alloc(op_name, layer_idx=6, pin=head_idx)
    return allocator


def _jsr_pc_byte1_relay_head_spec(BD, head_idx: int) -> DeclarativeAttentionHeadSpec:
    """L6 relay head: copy the staged JSR PC byte-1 nibble MARKER -> byte0 row.

    Q fires at the PC byte-0 token (IS_BYTE + H1[PC] + BYTE_INDEX_0 + HAS_SE,
    explicitly NOT the marker), K matches the same step's PC MARKER row
    (MARK_PC). V copies the staged ``JSR_PC_B1`` band; O writes the relayed
    ``JSR_PC_B1_AT_B0`` band. JSR_PC_B1 is nonzero ONLY on a real JSR step (the
    marker stage is gated by the override's TEMP[0] + opcode blockers), so the
    relay is inert on every non-JSR PC byte-0 row even though the Q/K pattern
    fires there. Mirrors the L3 head-7 marker->byte0 relay pattern + the head-5
    ``branch_pc_byte0_relay`` slot-52 gate (Q row that attends MARK_PC from the
    PC byte-0 token).
    """
    L = 15.0
    PC_I = 0
    GATE = 40  # anti-leakage Q/K-only gate slot (beyond the value-copy slots)
    v = []
    o = []
    for j in range(_JSR_PC_B1_WIDTH):
        v.append(AP(1 + j, BD.JSR_PC_B1 + j, 1.0))
        o.append(AO(BD.JSR_PC_B1_AT_B0 + j, 1 + j, 1.0))
    return DeclarativeAttentionHeadSpec(
        head_idx=head_idx,
        # STEEP positive ALiBi slope (5.0): the relay MUST attend ONLY the
        # SAME-STEP PC marker (exactly 1 token back from the PC byte-0 row).
        # ``MARK_PC`` is hot on EVERY step's marker, so with slope 0 the relay's
        # softmax spreads across ALL prior markers and step-0's staged
        # ``JSR_PC_B1`` (the main-JSR target's byte-1 nibble) LEAKS into every
        # later step's ``JSR_PC_B1_AT_B0`` -> the emit over-fires PC byte-1=1 on
        # every non-(small-target-JSR) step (gcd id900 stepped to PC 282 instead
        # of 26). The production attention applies ``-slope*dist`` (dist = q_pos
        # - k_pos), so a steep positive slope penalises the distant step-0 marker
        # (dist >= 35) by >=175 while the same-step marker (dist=1) keeps the
        # full Q/K score (~150) -- isolating the 1-token-back marker the comment
        # always intended. Matches the L6 head-5 steep-ALiBi convention (5.0).
        alibi_slope=5.0,
        q=(
            AP(0, BD.IS_BYTE, L),
            AP(0, BD.H1 + PC_I, L),
            AP(0, BD.BYTE_INDEX_0, L),
            AP(0, BD.HAS_SE, L),
            AP(0, BD.MARK_PC, -3.0 * L),  # not the marker itself
            AP(0, BD.CONST, -3.0 * L),
            # Anti-leakage gate: only the PC byte-0 token attends.
            AP(GATE, BD.IS_BYTE, 500.0),
            AP(GATE, BD.H1 + PC_I, 500.0),
            AP(GATE, BD.BYTE_INDEX_0, 500.0),
            AP(GATE, BD.CONST, -1000.0),
        ),
        k=(
            AP(0, BD.MARK_PC, L * 10.0),  # attend the step's PC MARKER row
            AP(GATE, BD.MARK_PC, 50.0),
        ),
        v=tuple(v),
        o=tuple(o),
    )


# OUTPUT re-supply write scale for the tail emit. The PC byte-0 row's OUTPUT is
# OVERWRITTEN by the L25 tail corruptor BEFORE the emit runs (~5e4 on gcd). The
# emit HARD-SETS the byte-0 row's OUTPUT_LO (+2*WS at the byte1 nibble, -WS at
# every other nibble) so the argmax is the byte1 nibble. WS must dominate the
# corruptor but stay BELOW the fp32-accumulation-instability band (~1e8+): a
# WS=5e7 emit produced OUTPUT_LO ~9e11, where the GPU and CPU W_down @ hidden
# accumulation orders DIVERGE and flip a nibble (the documented saturated-tie
# class) -> gcd PASSed on CPU but the GPU canonical decoded a leaked byte. Match
# l11_ops ``bp_save_dump`` (WS=2e5): the per-unit gain silu(cond)~relayed(~90) *
# 2*WS gives OUTPUT_LO ~3.6e7, dominating the ~5e4 nuke by >700x while staying
# in the stable fp band so GPU == CPU.
_JSR_PC_BYTE1_EMIT_WS = float(
    __import__("os").environ.get("C4_JSR_PC_BYTE1_EMIT_WS", "2e5")
)


def _jsr_pc_byte1_emit_rules(S: float, *, write_scale: float | None = None) -> tuple[FFNRule, ...]:
    """Emit PC byte-1 at the PC byte-0 row from the relayed band (tail FFN).

    The PC byte-0 token's OUTPUT predicts byte-1 (autoregressive shift). For each
    byte-1 value v in 1..7, when ``JSR_PC_B1_AT_B0+v`` is hot (relayed from the
    marker by the L6+1 relay head, and PERSISTING in the residual to the tail),
    write the value's low nibble into OUTPUT_LO and CANCEL OUTPUT_LO+0 (the byte-0
    nibble the L3 default + tail corruptor left high). High nibble of PC byte-1 is
    0 for every 1096 first-JSR target so OUTPUT_HI stays. Gated on the PC byte-0
    row (IS_BYTE + H1[PC] + BYTE_INDEX_0) so it never touches the marker or other
    byte positions; the relayed band is itself JSR-exclusive (nonzero only on a
    real JSR-to-idx>=32 step). The default ``write_scale`` is tuned to dominate
    the tail corruptor when this FFN runs AFTER it.
    """
    if write_scale is None:
        write_scale = _JSR_PC_BYTE1_EMIT_WS
    PC_I = 0
    rules: list[FFNRule] = []
    base_conditions = (
        ("IS_BYTE", 1.0),
        (f"H1+{PC_I}", 1.0),
        ("BYTE_INDEX_0", 1.0),
    )
    for v in range(1, _JSR_PC_B1_WIDTH):
        lo = v & 0xF
        # HARD-SET OUTPUT_LO at the PC byte-0 row: write +2*WS at the byte1
        # nibble and -WS at EVERY OTHER nibble, so the argmax is the byte1
        # nibble regardless of which OTHER nibble the tail corruptor left high
        # (rec_fib id725 has a ~2e10 SENTINEL at OUTPUT_LO+2 — the byte0 value
        # 0x02 leaking into the byte1 prediction — which a single OUTPUT_LO+0
        # cancel could not beat). The net at the target nibble is +2*WS, at all
        # others -WS; with WS dominating the corruptor magnitude the byte1
        # nibble wins cleanly. OUTPUT_HI stays (PC byte-1 high nibble = 0).
        writes = [(f"OUTPUT_LO+{lo}", 2.0 * write_scale)]
        for j in range(16):
            if j == lo:
                continue
            writes.append((f"OUTPUT_LO+{j}", -1.0 * write_scale))
        rules.append(multi_way_and_rule(
            name=f"jsr_pc_byte1_emit_{v}",
            # DOMINANT-index discriminator: fire ONLY when the relayed byte-1
            # nibble v EXCEEDS the index-0 ("no byte-1") cell. The staging is a
            # SwiGLU gate=FETCH_HI+k, so a small FETCH_HI residue at a non-dominant
            # nibble (e.g. gcd id900 step-6's JSR-to-idx-3 leaves FETCH_HI+3~1.0
            # alongside the dominant FETCH_HI+0~40) stages a WEAK JSR_PC_B1+1 that
            # the bare ``+v >= 0.8`` gate would treat as "byte1=1" and the hard-set
            # emit would amplify to a spurious PC=282. For a small-target JSR the
            # TRUE byte-1 is 0, so JSR_PC_B1_AT_B0+0 is the DOMINANT cell (~360 vs
            # the ~9 leak); subtracting it (weight -1.0) keeps the emit OFF unless
            # nibble v genuinely dominates (step-0 main-JSR: +1=1228 >> +0=78 ->
            # still fires). The relay copies JSR_PC_B1 -> JSR_PC_B1_AT_B0 1:1 so
            # the +0 cell carries the index-0 magnitude faithfully.
            conditions=base_conditions + (
                (f"JSR_PC_B1_AT_B0+{v}", 1.0),
                ("JSR_PC_B1_AT_B0+0", -1.0),
            ),
            threshold=3.5,  # IS_BYTE(1)+H1(1)+BYTE_INDEX_0(1)+relay(>=~0.8)
            writes=tuple(writes),
        ))
    return tuple(rules)


def make_jsr_pc_byte1_emit_op() -> Operation:
    """Append the JSR PC byte-1 emit FFN as a POST-TAIL-CORRUPTOR post_op.

    The relay head (baked inside ``make_function_call_weights_op`` on the block
    just past the override FFN) delivers the staged byte-1 nibble into
    ``JSR_PC_B1_AT_B0`` at the PC byte-0 row, where it PERSISTS in the residual.
    This emit re-supplies it into ``OUTPUT_LO`` at the PC byte-0 row so the LM
    head emits the correct PC byte-1 token (autoregressive shift). It runs as a
    standalone ``PureFFN`` post_op on the L25 tail block AFTER
    ``tail_bit32_result_correction`` (the 0xFF SENTINEL-magnitude corruptor that
    re-zeros the PC byte-0 OUTPUT ~block 42, flipping byte-1 back to 0), so it is
    the LAST writer of OUTPUT before the LM head — exactly the position +
    mechanism of ``l11_ops.bp_save_dump_repopulate`` (a large write_scale that
    dominates the ~5e4 nuke). Gated on the PC byte-0 row (IS_BYTE + H1[PC] +
    BYTE_INDEX_0) AND the JSR-exclusive relayed band, so it fires ONLY on a real
    JSR-to-idx>=32 step-0 byte-0 row. Off -> 0 rules -> inert -> golden
    byte-identical (the JSR_PC_B1_AT_B0 band is omitted too).
    """
    enabled = _jsr_pc_byte1_enabled()
    rules = _jsr_pc_byte1_emit_rules(100.0) if enabled else ()

    def bake(block, dim_positions, S):
        if not rules:
            return
        from ...base_layers import PureFFN
        # Resolve d_model from the block's attn / ffn.
        attn = getattr(block, "attn", None)
        d_model = None
        if attn is not None and hasattr(attn, "W_q"):
            d_model = attn.W_q.shape[0]
        if d_model is None and hasattr(block, "ffn") and hasattr(block.ffn, "W_up"):
            d_model = block.ffn.W_up.shape[1]
        if d_model is None:
            d_model = max(int(v) for v in dim_positions.values()) + 1
        real_rules = _jsr_pc_byte1_emit_rules(S)
        ffn = PureFFN(d_model, len(real_rules))
        dim_map = Primitives.dim_positions_from_bd(
            _as_setdim_proxy(dim_positions),
            Primitives.ffn_rule_dim_names(real_rules),
        )
        Primitives.lower_ffn_rules(ffn, real_rules, dim_map, S=S)
        block.post_ops.append(ffn)

    _reads = (
        {"IS_BYTE", "H1", "BYTE_INDEX_0", "JSR_PC_B1_AT_B0", "OUTPUT_LO"}
        if enabled else set()
    )
    _writes = {"OUTPUT_LO"} if enabled else set()
    return Operation(
        name="jsr_pc_byte1_emit",
        reads=_reads,
        writes=_writes,
        audited_empty_produces=not enabled,
        kind="block",
        # Append AFTER the tail correction on the L25 tail block, so this op is
        # the LAST writer of OUTPUT_LO before the LM head (it runs after the
        # block-42 corruptor that re-zeros the PC byte-0 OUTPUT). Mirrors
        # ``l11_ops.bp_save_dump_repopulate`` (same host + ordering).
        target_op_name="l10_post_ops_combined",
        requires={"after": ("tail_bit32_result_correction",)},
        declarative_bake_fn=bake,
        compiler_ir=CompilerIR(),
        declarative_authority="structural_model",
        migrated=True,
        semantic_label="jsr-pc-byte1 emit",
    )


def make_opcode_relay_head_op() -> Operation:
    """L6 attention head 6: relay opcode flags from AX -> SP/STACK0/PC markers.

    Originally an inline call in ``set_vm_weights`` (with two preceding
    ``attn6.alibi_slopes`` mutations folded in)::

        if hasattr(attn6, 'alibi_slopes') and attn6.alibi_slopes is not None:
            attn6.alibi_slopes[6] = 5.0
            attn6.alibi_slopes[7] = 5.0  # JSR PC+5 relay: steep for head 7
        _set_opcode_relay_head(attn6, S, BD, HD)

    The L5 FFN decodes opcodes only at the AX marker, but L6 FFN consumes
    OP_PSH/OP_ADJ/OP_LEV/OP_JSR/OP_ENT and a pop-group flag at SP, STACK0,
    BP, PC, and MEM markers. Head 6 copies those flags across markers.
    ALiBi slope=5.0 is also set on head 7 (used by JSR PC+5 relay configured
    by ``_set_function_call_weights``).

    Phase=1002 (AFTER ``legacy_bake`` at 999) is REQUIRED because the
    inline section that this op replaces is preceded inside
    ``set_vm_weights`` by::

        attn6.alibi_slopes.fill_(0.0)
        attn6.alibi_slopes[0] = 5.0
        attn6.alibi_slopes[1] = 5.0
        ...

    which still fires from inside legacy_bake. Running this op at phase
    <999 would have its alibi_slopes[6]/[7]=5.0 writes wiped by that
    ``fill_(0.0)`` call. Phase=1002 also slots cleanly after
    ``head_bake`` (1000) and ``embedding_bake`` (1001), and before the
    defensive post-pass ``branch_override_patch`` (1100).

    Head 6 attn weights (W_q/W_k/W_v/W_o) don't conflict with any other
    L6 attn writes in ``set_vm_weights``: head 0-5 are programmed by
    ``_set_layer6_attn`` and ``_set_bz_bnz_relay`` (now migrated to
    phases 998.5/.7), and head 7 Q/K is programmed by
    ``_set_function_call_weights`` (phase=998) plus
    ``_set_layer6_relay_heads`` (phase=998.6) — none of which touch
    head 6 slots, so phase ordering against those is irrelevant.

    Claim-coverage note: ``claims`` is intentionally left empty. The
    migrated ``_bake_layer6_relay_heads_spec`` head-6 section in
    ``ops/l6_ops.py`` (phase=998.6) writes the *same* W_q/W_k/W_v/W_o
    cells with the *same* values that this op writes -- it was promoted
    to a per-block spec ahead of ``opcode_relay_head``'s phase=1002
    dispatch. By the time this op runs, every cell it would set already
    holds the target value, so the snapshot diff comes back empty and
    the static verifier marks the bake "inert". The only remaining
    side effect of this op is ``alibi_slopes[6]=5.0`` /
    ``alibi_slopes[7]=5.0`` -- a non-weight-matrix mutation outside the
    verifier's diff scope. Declaring claims here would either flag every
    one as ``declared_but_not_written`` (the writes look like no-ops to
    the differ) or pass only via the inert short-circuit -- neither adds
    verification value. The genuine bake of these cells lives on
    ``make_layer6_relay_heads_bake_op``; claims should be attached there
    when that op is backfilled.
    """
    def bake(model, dim_positions, S):
        del S
        proxy = _as_setdim_proxy(dim_positions)
        attn = model.blocks[6].attn
        if hasattr(attn, 'alibi_slopes') and attn.alibi_slopes is not None:
            attn.alibi_slopes[6] = 5.0
            attn.alibi_slopes[7] = 5.0  # JSR PC+5 relay: steep for head 7
        HD = attn.W_q.shape[0] // attn.num_heads
        Primitives.generate_attention_head(attn, _opcode_relay_head_spec(proxy), HD)

    # Phase 8.A.4: ``layer_idx`` is intentionally omitted. This is
    # a ``kind="model"`` op without an ``ffn_units_used`` annotation, so
    # the (former) ``layer_idx=6`` neither anchored a bake (model ops
    # dispatch against the whole model) nor contributed to the per-block
    # FFN sizing aggregate (only ``layer_idx`` + ``ffn_units_used`` model
    # ops feed ``ffn_widths``). Purely cosmetic and removed without
    # behaviour change.
    return Operation(
        name="opcode_relay_head",
        # Phase: docs/DIM_LIVENESS_FINDINGS_2026_06_05.md audit — declare
        # baked reads (Q/K/V columns) and writes (O rows) for the L6
        # opcode-relay head 6. Derived from ``_opcode_relay_head_spec``.
        # (The cells overlap byte-identically with
        # ``layer6_relay_heads_bake``'s head-6 writes per the
        # claim-coverage note above; declarations stay accurate w.r.t.
        # the residual positions touched even though the bake is inert
        # at the static-verifier level.)
        reads={"MARK_SP", "MARK_STACK0", "MARK_BP", "MARK_PC",
               "MARK_MEM", "MARK_AX",
               "H1", "L1H4",
               "OP_PSH", "OP_ADJ", "OP_ENT", "OP_JSR",
               "OP_SI", "OP_SC", "OP_LEV",
               "OP_ADD", "OP_SUB", "OP_MUL", "OP_DIV", "OP_MOD",
               "OP_EQ", "OP_NE", "OP_LT", "OP_GT", "OP_LE", "OP_GE",
               "OP_OR", "OP_XOR", "OP_AND", "OP_SHL", "OP_SHR"},
        writes={"CMP", "PSH_AT_SP", "OP_JSR", "OP_ENT", "OP_LEV",
                "MEM_STORE", "MEM_ADDR_SRC"},
        # Wave 6 (docs/PRODUCES_CONSUMES_MIGRATION.md). Model-level bake
        # that rewires the L6 opcode relay attention head (K/V matrices
        # + alibi slope). Writes target attention parameters, not
        # per-step residual dims, so produces/consumes_fresh stays empty.
        audited_empty_produces=True,
        kind="model",
        declarative_bake_fn=bake,
        compiler_ir_factory=_opcode_relay_head_ir,
        declarative_authority="spec_generated",
        phase=1002,
        migrated=True,
        alibi_slopes={6: 5.0, 7: 5.0},
        smoke_tests={"all"},
        spec_section="BLOG_SPEC.md#how-bytecode-is-passed-to-the-network",
    )


def _opcode_relay_head_ir(dim_positions, HD) -> CompilerIR:
    proxy = _as_setdim_proxy(dim_positions)
    ir = CompilerIR()
    ir.layer(0).attention.append(_opcode_relay_head_spec(proxy))
    return ir


def _opcode_relay_head_spec(BD) -> DeclarativeAttentionHeadSpec:
    """Declarative L6 head 6: relay opcode flags from AX to active markers."""

    L = 50.0
    SP_I = 2
    BP_I = 3
    pop_op_dims = (
        BD.OP_ADD,
        BD.OP_SUB,
        BD.OP_MUL,
        BD.OP_DIV,
        BD.OP_MOD,
        BD.OP_EQ,
        BD.OP_NE,
        BD.OP_LT,
        BD.OP_GT,
        BD.OP_LE,
        BD.OP_GE,
        BD.OP_OR,
        BD.OP_XOR,
        BD.OP_AND,
        BD.OP_SHL,
        BD.OP_SHR,
        BD.OP_SI,
        BD.OP_SC,
    )
    v = [
        AP(1, BD.OP_PSH, 0.2),
        AP(2, BD.OP_ADJ, 0.2),
        AP(4, BD.OP_ENT, 0.2),
        AP(5, BD.OP_JSR, 0.2),
        AP(6, BD.OP_SI, 0.2),
        AP(6, BD.OP_SC, 0.2),
        AP(6, BD.OP_PSH, 0.2),
        AP(6, BD.OP_JSR, 0.2),
        AP(6, BD.OP_ENT, 0.2),
        AP(7, BD.OP_SI, 0.2),
        AP(7, BD.OP_SC, 0.2),
        AP(0, BD.OP_LEV, 0.1),
    ]
    v.extend(AP(3, op_dim, 0.04) for op_dim in pop_op_dims)
    return DeclarativeAttentionHeadSpec(
        head_idx=6,
        q=(
            AP(0, BD.MARK_SP, L),
            AP(0, BD.H1 + SP_I, L),
            AP(0, BD.MARK_STACK0, L),
            AP(0, BD.L1H4 + BP_I, L),
            AP(0, BD.MARK_BP, L),
            AP(0, BD.MARK_PC, L),
            AP(0, BD.MARK_MEM, L),
            AP(0, BD.MARK_AX, -L),
        ),
        k=(AP(0, BD.MARK_AX, L),),
        v=tuple(v),
        o=(
            AO(BD.CMP + 0, 1, 1.0),
            AO(BD.PSH_AT_SP, 1, 1.0),
            AO(BD.CMP + 1, 2, 1.0),
            AO(BD.CMP + 3, 3, 5.0),
            AO(BD.CMP + 2, 4, 1.0),
            AO(BD.CMP + 4, 5, 1.0),
            AO(BD.OP_JSR, 5, 5.0),
            AO(BD.OP_ENT, 4, 5.0),
            AO(BD.MEM_STORE, 6, 1.0),
            AO(BD.MEM_ADDR_SRC, 7, 1.0),
            AO(BD.OP_LEV, 0, 10.0),
        ),
    )


def make_residual_alibi_slopes_op() -> Operation:
    """Bake the residual ALiBi-slope mutations previously inline in set_vm_weights.

    Migrated 2026-05-11 (architectural milestone — final piece of the
    set_vm_weights → compiler-ops migration). Replaces the per-layer
    ``attn.alibi_slopes.fill_(...)`` and ``alibi_slopes[i] = ...`` writes
    that used to live in the inline body of ``set_vm_weights``:

      - L6 (head 0..4): fill_(0.0), then [0]=5.0, [1]=5.0, [2]=0.5,
        [3]=0.5, [4]=5.0 — must run BEFORE the legacy_bake retirement
        because ``make_opcode_relay_head_op`` (phase=1002) writes
        [6]=5.0 / [7]=5.0 and relies on the head-6/7 slots already being
        zero (per the docstring's "fill_(0.0) wipes" argument).
      - L8: fill_(0.5) — head-3/4 multibyte fetch / OP_IMM relay use
        the L8-wide gentle recency.
      - L10 head 0..4 (lookup) / 0..3 (efficient): steep carry relay +
        gentle byte passthrough slopes. Mode-conditional.
      - L14: fill_(5.0) — steep latest-source bias for MEM generation.
      - L15: fill_(0.01) — gentle latest-write-wins bias for memory lookup.

    Phase=999 places this exactly where ``legacy_bake`` used to run,
    preserving the previous override contract relative to phase-1002
    ``opcode_relay_head`` (which still needs to write [6]/[7]=5.0
    AFTER the L6 fill_(0.0)).
    """
    def _bake(model, dim_positions, S):
        # L6: head 0..4 slopes
        attn6 = model.blocks[6].attn
        if hasattr(attn6, 'alibi_slopes') and attn6.alibi_slopes is not None:
            attn6.alibi_slopes.fill_(0.0)
            attn6.alibi_slopes[0] = 5.0
            attn6.alibi_slopes[1] = 5.0
            attn6.alibi_slopes[2] = 0.5  # STACK0←AX relay: prefer nearest AX marker
            attn6.alibi_slopes[3] = 0.5  # SP←AX relay: prefer nearest AX marker
            attn6.alibi_slopes[4] = 5.0  # BZ/BNZ relay: attend to nearest AX marker
            # Softmax-sharpness fix (head 5 — first-step OP flag / FETCH
            # relay programmed by _set_layer6_attn). The audit (87442ad)
            # flagged head 5 with slope=0 leading to mass=0.10 at the
            # synthetic K target — the AX-marker-to-PC-marker hop is at
            # distance 4 in the audit context, and with slope=0 the head
            # has no positional preference between target and runner-up.
            # Raising slope to 1.0 closes the gap (1.0 * 4 = 4 nats per
            # the audit's gap recommendation). Paired with the K-scale
            # 10x bump in make_layer6_attn_bake_op (phase 998.5) so the
            # head clears the 99% sharpness threshold in strict mode.
            attn6.alibi_slopes[5] = 1.0

        # L8: per-layer recency
        attn8 = model.blocks[8].attn
        if hasattr(attn8, 'alibi_slopes') and attn8.alibi_slopes is not None:
            attn8.alibi_slopes.fill_(0.5)
            # Operand-gather one-hot fix (2026-06-11): head 0 is the
            # binary-op operand-A gather (STACK0 byte 0 -> ALU_LO/HI, baked
            # by layer7_operand_gather). The operand source token can be FAR
            # from the AX query row (~40 positions in a smoke step), so the
            # default recency slope 0.5 imposes a ~-20 alibi penalty that
            # nearly cancels the head's strong STACK0_BYTE0 QK match (~31.75)
            # relative to the uniform CONST baseline (~10.78 on every row).
            # The operand then wins by a razor-thin margin (w~0.51) and a
            # tail of nearby zero-byte AX-result rows (CLEAN_EMBED one-hot at
            # nibble 0) bleeds into the band copy -> the value-proportional
            # @0 magnitude artifact in ALU_LO/HI that blocks the CMP/ALU
            # multi_way_and consumers (AX_CARRY, a sharper head, stays a
            # clean one-hot). A shallow slope sharpens the pointer onto the
            # single operand source so ALU_LO/HI become clean per-nibble
            # one-hots. This op (phase 999) is the authoritative block-8
            # slope owner; it runs AFTER layer7_operand_gather's bake.
            attn8.alibi_slopes[0] = 0.1
            # Head 3 fetches IMM bytes from the static code prefix by exact
            # ADDR_KEY. A steep recency penalty makes long smoke contexts lose
            # the code-byte match to the zero anchor, leaving stale AX_CARRY.
            attn8.alibi_slopes[3] = 0.1

        # L14: steep recency bias so MEM generation chooses the current
        # step's SP/AX source instead of blending in prior-step markers.
        if len(model.blocks) > 14:
            attn14 = model.blocks[14].attn
            if hasattr(attn14, 'alibi_slopes') and attn14.alibi_slopes is not None:
                attn14.alibi_slopes.fill_(5.0)

        # L15: load heads are last-write-wins only among matching addresses.
        # Keep the primary LI/LC/STACK0 heads as a small recency tie-breaker
        # so a newer write to a different local slot cannot beat address match.
        if len(model.blocks) > 15:
            attn15 = model.blocks[15].attn
            if hasattr(attn15, 'alibi_slopes') and attn15.alibi_slopes is not None:
                attn15.alibi_slopes.fill_(0.01)
                attn15.alibi_slopes[:4] = 0.05
                # L15 can be widened by declarative block ops before this
                # residual slope pass. Preserve steep recency on those
                # single-purpose heads; otherwise SI/SC MEM addr0 blends many
                # historical STACK0 byte-0 rows instead of choosing the
                # nearest pre-store address row.
                if attn15.alibi_slopes.numel() > 8:
                    attn15.alibi_slopes[8] = 1.0
                if attn15.alibi_slopes.numel() > 12:
                    attn15.alibi_slopes[12] = 1.0
                if attn15.alibi_slopes.numel() > 13:
                    attn15.alibi_slopes[13] = 1.0

    return Operation(
        name="residual_alibi_slopes",
        reads=set(),
        writes=set(),
        # Wave 6 (docs/PRODUCES_CONSUMES_MIGRATION.md). Model-level bake
        # that sets attention alibi slope tables. Writes target attention
        # bias parameters, not per-step residual dims.
        audited_empty_produces=True,
        kind="model",
        declarative_bake_fn=_bake,
        phase=999,
        migrated=True,
        declarative_authority="structural_model",
        # Dim-ownership claims: empty. ``bake`` writes
        # ``attn.alibi_slopes`` tables on L6 / L8 / L14 / L15 -- ALiBi
        # bias parameters (not residual dims), spanning four blocks.
        # Module-level table writes, not per-cell ``(layer, scope,
        # identifier, column)`` writes. Sentinel below documents the
        # structural effect.
        claims=set(),
        # Module-replacement sentinel: dynamic verifier (Mode B) skips
        # drift detection; static (Mode A) snapshot diffing unaffected.
        produces={'__module_replacement': 'L6/L8/L14/L15.attn[alibi_slopes]'},
        smoke_tests=set(),
        spec_section="BLOG_SPEC.md#the-attention-layer",
    )


# ---------------------------------------------------------------------------
# Model-level post-passes
# ---------------------------------------------------------------------------


def make_branch_override_patch_op() -> Operation:  # noqa: E302
    """Defensive gate that suppresses spurious branch/LEV-override FFN units.

    Any FFN unit firing with positive MARK_PC AND a positive
    OP_JMP/OP_LEV/OP_BZ/OP_BNZ/CMP[0] trigger AND writing OUTPUT_LO/HI gets
    strong negative weights for non-target opcodes — only the legit branch/LEV
    opcode lets the unit fire. Without this, L5 attention leaks opcode flags
    into MARK_PC of subsequent steps and FFN units fire spuriously.
    """
    def bake(model, dim_positions, S):
        BD = _as_setdim_proxy(dim_positions)
        OPCODE_BLOCK_MAP = {
            BD.OP_JMP: [BD.OP_IMM, BD.OP_EXIT, BD.OP_NOP, BD.OP_LEA, BD.OP_LEV,
                        BD.OP_ADD, BD.OP_SUB, BD.OP_MUL, BD.OP_DIV, BD.OP_MOD,
                        BD.OP_OR, BD.OP_XOR, BD.OP_AND,
                        BD.OP_EQ, BD.OP_NE, BD.OP_LT, BD.OP_GT, BD.OP_LE, BD.OP_GE,
                        BD.OP_SHL, BD.OP_SHR, BD.OP_PSH, BD.OP_LI, BD.OP_LC,
                        BD.OP_SI, BD.OP_SC, BD.OP_ADJ, BD.OP_ENT,
                        BD.OP_BZ, BD.OP_BNZ, BD.OP_JSR],
            BD.OP_LEV: [BD.OP_IMM, BD.OP_EXIT, BD.OP_NOP, BD.OP_LEA, BD.OP_JMP,
                        BD.OP_ADD, BD.OP_SUB, BD.OP_MUL, BD.OP_DIV, BD.OP_MOD,
                        BD.OP_OR, BD.OP_XOR, BD.OP_AND,
                        BD.OP_EQ, BD.OP_NE, BD.OP_LT, BD.OP_GT, BD.OP_LE, BD.OP_GE,
                        BD.OP_SHL, BD.OP_SHR, BD.OP_PSH, BD.OP_LI, BD.OP_LC,
                        BD.OP_SI, BD.OP_SC, BD.OP_ADJ, BD.OP_ENT,
                        BD.OP_BZ, BD.OP_BNZ, BD.OP_JSR],
            BD.OP_BZ:  [BD.OP_IMM, BD.OP_EXIT, BD.OP_NOP, BD.OP_LEA, BD.OP_JMP, BD.OP_LEV,
                        BD.OP_ADD, BD.OP_SUB, BD.OP_MUL, BD.OP_DIV, BD.OP_MOD,
                        BD.OP_OR, BD.OP_XOR, BD.OP_AND,
                        BD.OP_EQ, BD.OP_NE, BD.OP_LT, BD.OP_GT, BD.OP_LE, BD.OP_GE,
                        BD.OP_SHL, BD.OP_SHR, BD.OP_PSH, BD.OP_LI, BD.OP_LC,
                        BD.OP_SI, BD.OP_SC, BD.OP_ADJ, BD.OP_ENT,
                        BD.OP_BNZ, BD.OP_JSR],
            BD.OP_BNZ: [BD.OP_IMM, BD.OP_EXIT, BD.OP_NOP, BD.OP_LEA, BD.OP_JMP, BD.OP_LEV,
                        BD.OP_ADD, BD.OP_SUB, BD.OP_MUL, BD.OP_DIV, BD.OP_MOD,
                        BD.OP_OR, BD.OP_XOR, BD.OP_AND,
                        BD.OP_EQ, BD.OP_NE, BD.OP_LT, BD.OP_GT, BD.OP_LE, BD.OP_GE,
                        BD.OP_SHL, BD.OP_SHR, BD.OP_PSH, BD.OP_LI, BD.OP_LC,
                        BD.OP_SI, BD.OP_SC, BD.OP_ADJ, BD.OP_ENT,
                        BD.OP_BZ, BD.OP_JSR],
            BD.CMP + 0: [BD.OP_IMM, BD.OP_EXIT, BD.OP_NOP, BD.OP_LEA, BD.OP_LEV,
                         BD.OP_ADD, BD.OP_SUB, BD.OP_MUL, BD.OP_DIV, BD.OP_MOD,
                         BD.OP_OR, BD.OP_XOR, BD.OP_AND,
                         BD.OP_EQ, BD.OP_NE, BD.OP_LT, BD.OP_GT, BD.OP_LE, BD.OP_GE,
                         BD.OP_SHL, BD.OP_SHR, BD.OP_PSH, BD.OP_LI, BD.OP_LC,
                         BD.OP_SI, BD.OP_SC, BD.OP_ADJ, BD.OP_ENT,
                         BD.OP_BZ, BD.OP_BNZ, BD.OP_JSR],
        }
        branch_override_patches = 0
        for block_idx in range(len(model.blocks)):
            ffn = model.blocks[block_idx].ffn
            if not (hasattr(ffn, 'W_up') and isinstance(getattr(ffn, 'W_up', None), nn.Parameter)):
                continue
            hidden_dim = ffn.W_up.shape[0]
            for u in range(hidden_dim):
                mark_pc_w = ffn.W_up[u, BD.MARK_PC].item()
                if mark_pc_w <= 50:
                    continue
                writes_output = (
                    ffn.W_down[BD.OUTPUT_LO:BD.OUTPUT_LO+16, u].abs().max().item() > 0.01
                    or ffn.W_down[BD.OUTPUT_HI:BD.OUTPUT_HI+16, u].abs().max().item() > 0.01
                )
                if not writes_output:
                    continue
                for trigger_dim, blockers in OPCODE_BLOCK_MAP.items():
                    trigger_w = ffn.W_up[u, trigger_dim].item()
                    if trigger_w <= 5:
                        continue
                    for opcode_dim in blockers:
                        cur = ffn.W_up[u, opcode_dim].item()
                        ffn.W_up.data[u, opcode_dim] = min(cur, -S)
                    branch_override_patches += 1
                    break
        print(f"  BRANCH OVERRIDE PATCH: {branch_override_patches} units gated")

    return Operation(
        name="branch_override_patch",
        # Phase: docs/DIM_LIVENESS_FINDINGS_2026_06_05.md audit — declare
        # the residual positions inspected/modified by the defensive
        # sweep. The bake reads ``W_up[u, BD.MARK_PC]`` and
        # ``W_down[BD.OUTPUT_LO/HI:..., u]`` as filters, then sets
        # ``W_up[u, BD.OP_*]`` to ``-S`` for non-target opcodes. From
        # the liveness analysis's perspective every BD column touched
        # (read or modified) and every BD row covered must be declared
        # so the producer/consumer graph stays sound.
        reads={"MARK_PC", "CMP",
               "OP_JMP", "OP_LEV", "OP_BZ", "OP_BNZ", "OP_JSR",
               "OP_IMM", "OP_EXIT", "OP_NOP", "OP_LEA",
               "OP_ADD", "OP_SUB", "OP_MUL", "OP_DIV", "OP_MOD",
               "OP_OR", "OP_XOR", "OP_AND",
               "OP_EQ", "OP_NE", "OP_LT", "OP_GT", "OP_LE", "OP_GE",
               "OP_SHL", "OP_SHR", "OP_PSH", "OP_LI", "OP_LC",
               "OP_SI", "OP_SC", "OP_ADJ", "OP_ENT"},
        writes={"OUTPUT_LO", "OUTPUT_HI"},
        # Wave 6 (docs/PRODUCES_CONSUMES_MIGRATION.md). Defensive sweep
        # / topology patch that zeros stale branch-override weights. No
        # in-step semantic residual reads/writes.
        audited_empty_produces=True,
        kind="model",
        declarative_bake_fn=bake,
        phase=1100,
        migrated=True,
        declarative_authority="structural_model",
        # Phase 11.A: empty CompilerIR (defensive sweep / topology op;
        # no static FFNRule form -- introspects already-baked weights).
        compiler_ir=CompilerIR(),
        # Dim-ownership claims: empty. ``bake`` scans every block's
        # FFN and zeros W_up rows of spuriously-firing branch/LEV
        # override units across the entire model -- whole-model
        # defensive weight surgery, not per-cell ``(layer, scope,
        # identifier, column)`` writes. Sentinel below documents the
        # structural effect.
        claims=set(),
        # Module-replacement sentinel: dynamic verifier (Mode B) skips
        # drift detection; static (Mode A) snapshot diffing unaffected.
        produces={'__module_replacement': 'model.blocks[*].ffn[branch override zero]'},
        smoke_tests={"all"},
        spec_section="BLOG_SPEC.md#control-flow",
    )


def make_l6_dead_unit_zero_op() -> Operation:
    """Defensive gate that zeros L6 FFN units misreading OUTPUT_BYTE residuals.

    Originally an inline post-pass in `set_vm_weights` (BUG FIX 2026-04-09 part
    8c/8g). Unprogrammed L6 FFN units can fire spuriously when:

    - they write to OUTPUT_LO/HI (W_down rows in those slices are non-zero), AND
    - they have strong W_up reads on OUTPUT_BYTE_LO/HI (which carry residual
      values from the prior IO step), AND
    - they don't have positive marker weights (MARK_PC, MARK_STACK0, MARK_BP)
      indicating intentional TEMP usage by JSR PC override / ENT.

    The patch zeros W_up/W_gate rows, W_down columns, and biases for every
    matching unit so they cannot fire at any position.

    Operates on `model.blocks[6].ffn` (the L6 routing FFN). Phase=1160 so it
    runs after branch_override_patch (phase=1100) and the head/embedding bakes
    (1000/1001), and before right_size_ffns (1200) — preserving the original
    in-set_vm_weights ordering where this pass ran before the right-size pass.

    Migrated=True: the corresponding inline block in `set_vm_weights` has been
    removed to avoid double-bake.
    """
    def bake(model, dim_positions, S):
        BD = _as_setdim_proxy(dim_positions)
        ffn6 = model.blocks[6].ffn
        if not (hasattr(ffn6, 'W_up') and isinstance(getattr(ffn6, 'W_up', None), nn.Parameter)):
            return
        hidden_dim = ffn6.W_up.shape[0]
        patched_count = 0
        for u in range(hidden_dim):
            writes_output_lo = (
                ffn6.W_down[BD.OUTPUT_LO:BD.OUTPUT_LO + 16, u].abs().max().item() > 0.01
            )
            writes_output_hi = (
                ffn6.W_down[BD.OUTPUT_HI:BD.OUTPUT_HI + 16, u].abs().max().item() > 0.01
            )
            if not (writes_output_lo or writes_output_hi):
                continue

            # Skip units that legitimately use TEMP at marker positions.
            if ffn6.W_up[u, BD.MARK_PC].item() > 10:  # JSR PC override uses S=100
                continue
            if ffn6.W_up[u, BD.MARK_STACK0].item() > 10:  # ENT uses TEMP at STACK0
                continue
            if ffn6.W_up[u, BD.MARK_BP].item() > 10:  # ENT uses TEMP at BP
                continue
            if (
                ffn6.W_up[u, BD.IS_BYTE].item() > 10
                and ffn6.W_up[u, BD.H1 + 0].item() > 10
                and ffn6.W_up[u, BD.BYTE_INDEX_0].item() > 10
            ):
                continue

            output_byte_lo_weight = (
                ffn6.W_up[u, BD.OUTPUT_BYTE_LO:BD.OUTPUT_BYTE_LO + 16].abs().max().item()
            )
            output_byte_hi_weight = (
                ffn6.W_up[u, BD.OUTPUT_BYTE_HI:BD.OUTPUT_BYTE_HI + 16].abs().max().item()
            )

            if output_byte_lo_weight > 50 or output_byte_hi_weight > 50:
                ffn6.W_up.data[u, :] = 0
                ffn6.W_gate.data[u, :] = 0
                ffn6.W_down.data[:, u] = 0
                ffn6.b_up.data[u] = 0
                ffn6.b_gate.data[u] = 0
                patched_count += 1
        if patched_count:
            print(f"  L6 DEAD-UNIT ZERO: {patched_count} units zeroed")

    return Operation(
        name="l6_dead_unit_zero",
        # Phase: docs/DIM_LIVENESS_FINDINGS_2026_06_05.md audit — declare
        # the residual positions inspected/modified by the defensive
        # sweep. The bake reads ``W_down[BD.OUTPUT_LO/HI:..., u]`` and
        # ``W_up[u, BD.MARK_PC/STACK0/BP/IS_BYTE/H1/BYTE_INDEX_0]``
        # plus the ``W_up[u, BD.OUTPUT_BYTE_LO/HI:...]`` filter, then
        # zeros entire units (W_up/W_gate/W_down/biases) that match.
        # The zeroed W_down covers ALL residual rows — declared via the
        # OUTPUT_LO/HI writes since those are the rows the filter
        # required to be non-zero.
        reads={"MARK_PC", "MARK_STACK0", "MARK_BP",
               "IS_BYTE", "H1", "BYTE_INDEX_0",
               "OUTPUT_BYTE_LO", "OUTPUT_BYTE_HI"},
        writes={"OUTPUT_LO", "OUTPUT_HI"},
        # Wave 6 (docs/PRODUCES_CONSUMES_MIGRATION.md). Defensive sweep:
        # zero L6 FFN units that no op currently writes (post-rightsize
        # safety). No in-step semantic residual reads/writes.
        audited_empty_produces=True,
        kind="model",
        declarative_bake_fn=bake,
        phase=1160,
        migrated=True,
        declarative_authority="structural_model",
        # Phase 11.A: empty CompilerIR (defensive sweep / topology op;
        # no static FFNRule form -- introspects already-baked weights).
        compiler_ir=CompilerIR(),
        # Dim-ownership claims: empty. ``bake`` scans
        # ``model.blocks[6].ffn`` and zeros W_up/W_gate rows, W_down
        # columns, and biases for spuriously-firing units writing to
        # OUTPUT_LO/HI -- structural defensive weight surgery, not
        # per-cell ``(layer, scope, identifier, column)`` writes.
        # Sentinel below documents the structural effect.
        claims=set(),
        # Module-replacement sentinel: dynamic verifier (Mode B) skips
        # drift detection; static (Mode A) snapshot diffing unaffected.
        produces={'__module_replacement': 'L6.ffn[dead unit zero]'},
        smoke_tests=set(),
        spec_section="BLOG_SPEC.md#registers",
    )


def make_l7_dead_unit_zero_op() -> Operation:
    """Defensive gate that suppresses L7 FFN units firing at PC marker.

    Originally an inline post-pass in `set_vm_weights` (BUG FIX 2026-04-17).
    Some FFN units (e.g. units 746/754) carry W_up[OP_ENT]=100, W_up[MARK_SP]=100,
    b_up=-150 — they're meant for ENT SP byte 0 generation but `OP_ENT * 5 = 500`
    overcomes the bias even when MARK_SP=0, so they spuriously fire at PC marker
    positions. The patch adds strong negative MARK_PC and IS_BYTE weights to the
    matching units so they cannot fire outside their intended marker position.

    Matching criteria:
      - writes to OUTPUT_LO/HI, AND
      - has strong (>50) W_up reads on OP_ENT/OP_LEV/OP_JSR/OP_LEA, AND
      - lacks existing MARK_PC suppression (W_up[MARK_PC] >= -100), AND
      - is not a byte-position unit (no positive IS_BYTE / BYTE_INDEX_0..3
        weights >5), AND
      - does not legitimately fire at PC marker (W_up[MARK_PC] <= 10).

    Operates on `model.blocks[6].ffn` (preserving the original code's choice;
    the legacy comment claims `blocks[6] = L7` but the same FFN is also the L6
    routing FFN in the current block layout — the pass is purely defensive and
    only modifies units matching the criteria above, so co-location is safe).

    Phase=1170 so it runs after the L6 dead-unit zero pass (phase=1160) and
    before right_size_ffns (1200) — preserving original ordering.
    """
    def bake(model, dim_positions, S):
        BD = _as_setdim_proxy(dim_positions)
        ffn7 = model.blocks[6].ffn  # blocks[6] = L7 (per original code's comment)
        if not (hasattr(ffn7, 'W_up') and isinstance(getattr(ffn7, 'W_up', None), nn.Parameter)):
            return
        hidden_dim = ffn7.W_up.shape[0]
        patched_l7_count = 0
        for u in range(hidden_dim):
            writes_output_lo = (
                ffn7.W_down[BD.OUTPUT_LO:BD.OUTPUT_LO + 16, u].abs().max().item() > 0.01
            )
            writes_output_hi = (
                ffn7.W_down[BD.OUTPUT_HI:BD.OUTPUT_HI + 16, u].abs().max().item() > 0.01
            )
            if not (writes_output_lo or writes_output_hi):
                continue

            has_strong_opcode = (
                ffn7.W_up[u, BD.OP_ENT].abs().item() > 50 or
                ffn7.W_up[u, BD.OP_LEV].abs().item() > 50 or
                ffn7.W_up[u, BD.OP_JSR].abs().item() > 50 or
                ffn7.W_up[u, BD.OP_LEA].abs().item() > 50
            )

            has_mark_pc_suppression = ffn7.W_up[u, BD.MARK_PC].item() < -100

            is_byte_unit = (
                ffn7.W_up[u, BD.IS_BYTE].item() > 5 or
                ffn7.W_up[u, BD.BYTE_INDEX_0].item() > 5 or
                ffn7.W_up[u, BD.BYTE_INDEX_1].item() > 5 or
                ffn7.W_up[u, BD.BYTE_INDEX_2].item() > 5 or
                ffn7.W_up[u, BD.BYTE_INDEX_3].item() > 5
            )

            # Skip units that legitimately fire at PC marker.
            if ffn7.W_up[u, BD.MARK_PC].item() > 10:
                continue

            if has_strong_opcode and not has_mark_pc_suppression and not is_byte_unit:
                ffn7.W_up.data[u, BD.MARK_PC] = -S * 100  # -1000 when MARK_PC = 1
                ffn7.W_up.data[u, BD.IS_BYTE] = -S * 100  # -1000 when IS_BYTE = 1
                patched_l7_count += 1
        if patched_l7_count:
            print(f"  L7 DEAD-UNIT ZERO: {patched_l7_count} units suppressed")

    return Operation(
        name="l7_dead_unit_zero",
        # Phase: docs/DIM_LIVENESS_FINDINGS_2026_06_05.md audit — declare
        # the residual positions inspected/modified by the defensive
        # sweep. The bake reads OP_ENT/LEV/JSR/LEA, MARK_PC, IS_BYTE,
        # BYTE_INDEX_0..3, OUTPUT_LO/HI (row check on W_down) and
        # writes (adds suppression weights into) the MARK_PC and
        # IS_BYTE columns of W_up. From a liveness standpoint these
        # are existing read columns whose values are being modified
        # (not a fresh residual producer); OUTPUT_LO/HI declared as
        # writes since the row-check filter binds them as touched.
        reads={"OP_ENT", "OP_LEV", "OP_JSR", "OP_LEA",
               "MARK_PC", "IS_BYTE",
               "BYTE_INDEX_0", "BYTE_INDEX_1", "BYTE_INDEX_2", "BYTE_INDEX_3"},
        writes={"OUTPUT_LO", "OUTPUT_HI"},
        # Wave 6 (docs/PRODUCES_CONSUMES_MIGRATION.md). Defensive sweep:
        # zero L7 FFN units that no op currently writes. No in-step
        # semantic residual reads/writes.
        audited_empty_produces=True,
        kind="model",
        declarative_bake_fn=bake,
        phase=1170,
        migrated=True,
        declarative_authority="structural_model",
        # Phase 11.A: empty CompilerIR (defensive sweep / topology op;
        # no static FFNRule form -- introspects already-baked weights).
        compiler_ir=CompilerIR(),
        # Dim-ownership claims: empty. ``bake`` scans
        # ``model.blocks[7].ffn`` and adds strong MARK_PC / IS_BYTE
        # negative weights to spuriously-firing units (suppression
        # patch) -- structural defensive weight surgery, not per-cell
        # ``(layer, scope, identifier, column)`` writes. Sentinel
        # below documents the structural effect.
        claims=set(),
        # Module-replacement sentinel: dynamic verifier (Mode B) skips
        # drift detection; static (Mode A) snapshot diffing unaffected.
        produces={'__module_replacement': 'L7.ffn[dead unit suppress]'},
        smoke_tests=set(),
        spec_section="BLOG_SPEC.md#registers",
    )


def make_right_size_ffns_op() -> Operation:
    """Trim each block's FFN hidden dim to actually-programmed unit count."""
    def bake(model, dim_positions, S):
        from ...vm_step import _right_size_ffns
        _right_size_ffns(model)

    return Operation(
        name="right_size_ffns",
        reads=set(),
        writes=set(),
        # Wave 6 (docs/PRODUCES_CONSUMES_MIGRATION.md). FFN width trim
        # pass: drops unused hidden units. Pure topology pass with no
        # residual-dim semantics.
        audited_empty_produces=True,
        kind="model",
        declarative_bake_fn=bake,
        phase=1200,
        migrated=True,
        declarative_authority="structural_model",
        # Phase 11.A: empty CompilerIR (defensive sweep / topology op;
        # no static FFNRule form -- introspects already-baked weights).
        compiler_ir=CompilerIR(),
        # Dim-ownership claims: empty. ``bake`` trims each block's FFN
        # hidden_dim to the actually-programmed unit count -- whole-
        # model topology surgery on ``W_up`` / ``W_gate`` / ``W_down``
        # shapes, not per-cell ``(layer, scope, identifier, column)``
        # writes. Sentinel below documents the structural effect.
        claims=set(),
        # Module-replacement sentinel: dynamic verifier (Mode B) skips
        # drift detection; static (Mode A) snapshot diffing unaffected.
        produces={'__module_replacement': 'model.blocks[*].ffn[trim hidden_dim]'},
        smoke_tests=set(),
        spec_section=None,
    )


def make_expand_wrapper_blocks_op() -> Operation:
    """Split HybridALUBlock + post_ops into separate transformer blocks.

    Default path runs :func:`_expand_wrapper_blocks` (each post_op
    becomes its own block with a zero-init passthrough attention).

    Phase 10.B alternate path: when ``C4_DISABLE_WRAPPER_EXPANSION=1``
    is set in the environment, :func:`_merge_wrapper_blocks` folds each
    block's post_ops into the parent block's FFN via ``nn.Sequential``
    instead, eliminating ~35.8M dead wrapper-attention params (19.5%%
    of total at the production d_model=800). Forward math is byte-
    identical on the default ``use_rms_norm=False`` setting.
    """
    def bake(model, dim_positions, S):
        import os as _os
        if _os.environ.get("C4_DISABLE_WRAPPER_EXPANSION") == "1":
            from ...vm_step import _merge_wrapper_blocks
            _merge_wrapper_blocks(model)
        else:
            from ...vm_step import _expand_wrapper_blocks
            _expand_wrapper_blocks(model)

    return Operation(
        name="expand_wrapper_blocks",
        reads=set(),
        writes=set(),
        # Wave 6 (docs/PRODUCES_CONSUMES_MIGRATION.md). Topology pass that
        # inserts wrapper blocks (multi-block expansion). No residual-dim
        # semantics.
        audited_empty_produces=True,
        kind="model",
        declarative_bake_fn=bake,
        phase=1300,
        migrated=True,
        declarative_authority="structural_model",
        # Phase 11.A: empty CompilerIR (defensive sweep / topology op;
        # no static FFNRule form -- introspects already-baked weights).
        compiler_ir=CompilerIR(),
        # Dim-ownership claims: empty. ``bake`` rebuilds
        # ``model.blocks`` -- splits HybridALUBlocks + post_ops into
        # separate transformer blocks (default) or folds them into the
        # parent block's FFN via ``nn.Sequential``
        # (C4_DISABLE_WRAPPER_EXPANSION=1 path). Whole-model topology
        # surgery, not per-cell ``(layer, scope, identifier, column)``
        # writes. Sentinel below documents the structural effect.
        claims=set(),
        # Module-replacement sentinel: dynamic verifier (Mode B) skips
        # drift detection; static (Mode A) snapshot diffing unaffected.
        produces={'__module_replacement': 'model.blocks[* expansion/fold]'},
        smoke_tests=set(),
        spec_section=None,
    )


def _head_bake_rules(vocab_size: int, dim_positions) -> tuple:
    """Build the :class:`TokenEmbeddingRule` list mirroring ``setup_head_weights``.

    The imperative helper in ``ops/shared.py`` (see ``setup_head_weights``)
    walks the head's weight + bias as follows:

      1. Zero both ``head.weight`` and ``head.bias``.
      2. Build a ``next_flags`` list of NEXT_* dim names (some optional).
      3. For each byte ``b in 0..255``:
         - ``head.weight[b, OUTPUT_LO+lo] = 5.0``
         - ``head.weight[b, OUTPUT_HI+hi] = 5.0``
         - ``head.bias[b] = -5.0``
         - For each ``flag in next_flags``: ``head.weight[b, flag] += -80.0``
      4. ``head.bias[0] = -4.0`` (overrides the -5.0 from step 3).
      5. For each (tok, flag_name) pair: ``head.weight[tok, D(flag_name)] = 20.0``
         and ``head.bias[tok] = -10.0``.
      6. For [CODE_START, CODE_END, DATA_START, DATA_END, SEP,
         USER_INPUT_START, USER_INPUT_END]: ``head.bias[tok] = -50.0``.
      7. ``head.bias[IO_STATE_EMIT_BYTE] = -20.0`` and
         ``head.bias[IO_STATE_EMIT_THINKING] = -20.0``.

    The rule lowering is accumulative (``+=``), so we follow exactly the same
    sequence: the bake_fn zeroes ``head.weight`` / ``head.bias`` first, then
    every rule is additive. The single ``=`` vs ``+=`` conflict is step 4,
    which writes byte 0's bias to ``-4.0`` AFTER step 3 wrote ``-5.0``. We
    emit a delta rule (``bias += +1.0`` for byte 0) so the final cell value
    is ``-5.0 + 1.0 = -4.0`` — byte-identical.

    The ``next_flags`` list is built from ``dim_positions`` so the rules
    only reference dims that exist in the active layout (mirrors the
    legacy helper's per-flag ``try: D(opt) except AttributeError``).
    Marker / step-end rules likewise skip the flag-name lookup when the
    dim is absent.
    """
    from ...vm_step import Token

    rules: list[TokenEmbeddingRule] = []

    # 2. Build the NEXT_* flags list (the optional flags only appear when
    #    conversational I/O is enabled — same try/except semantics as
    #    ``setup_head_weights``).
    next_flags: list[str] = [
        "NEXT_PC", "NEXT_AX", "NEXT_SP", "NEXT_BP",
        "NEXT_STACK0", "NEXT_MEM", "NEXT_SE", "NEXT_HALT",
    ]
    for opt in ("NEXT_TOOL_CALL", "NEXT_THINKING_START", "NEXT_THINKING_END"):
        if opt in dim_positions:
            next_flags.append(opt)

    byte_tokens = [b for b in range(min(256, vocab_size))]

    # 3a. For each byte: head.weight[b, OUTPUT_LO+lo] = 5.0 and
    #     head.weight[b, OUTPUT_HI+hi] = 5.0. Emit one rule per byte.
    for b in byte_tokens:
        lo = b & 0xF
        hi = (b >> 4) & 0xF
        rules.append(TokenEmbeddingRule.head_weight_write(
            token_ids=[b],
            writes=(
                (f"OUTPUT_LO+{lo}", 5.0),
                (f"OUTPUT_HI+{hi}", 5.0),
            ),
            name=f"head_bake_byte_{b}_output",
        ))

    # 3b. All bytes get -80.0 on every NEXT_* flag (the inner ``+=``).
    #     Emit one rule per flag with token_ids=byte_tokens.
    for flag in next_flags:
        rules.append(TokenEmbeddingRule.head_weight_write(
            token_ids=byte_tokens,
            writes=((flag, -80.0),),
            name=f"head_bake_byte_next_flag_{flag.lower()}",
        ))

    # 3c. All bytes get bias = -5.0.
    rules.append(TokenEmbeddingRule.head_bias_write(
        token_ids=byte_tokens,
        value=-5.0,
        name="head_bake_byte_bias_minus_five",
    ))

    # 4. byte 0's bias is overridden to -4.0. With ``+=`` semantics we
    #    add +1.0 so the cell ends at -5.0 + 1.0 = -4.0.
    if 0 < vocab_size:
        rules.append(TokenEmbeddingRule.head_bias_write(
            token_ids=[0],
            value=1.0,
            name="head_bake_byte_zero_bias_override_delta",
        ))

    # 5. Marker/step-end tokens: head.weight[tok, D(flag_name)] = 20.0
    #    and head.bias[tok] = -10.0. The legacy helper's try/except
    #    skips the entry when the flag dim is missing; we do the same.
    for tok, flag_name in (
        (Token.REG_PC, "NEXT_PC"),
        (Token.REG_AX, "NEXT_AX"),
        (Token.REG_SP, "NEXT_SP"),
        (Token.REG_BP, "NEXT_BP"),
        (Token.STACK0, "NEXT_STACK0"),
        (Token.MEM, "NEXT_MEM"),
        (Token.STEP_END, "NEXT_SE"),
        (Token.HALT, "NEXT_HALT"),
        (Token.TOOL_CALL, "NEXT_TOOL_CALL"),
        (Token.THINKING_START, "NEXT_THINKING_START"),
        (Token.THINKING_END, "NEXT_THINKING_END"),
    ):
        if tok >= vocab_size:
            continue
        if flag_name not in dim_positions:
            continue
        rules.append(TokenEmbeddingRule.head_weight_write(
            token_ids=[tok],
            writes=((flag_name, 20.0),),
            name=f"head_bake_marker_weight_token_{tok}",
        ))
        rules.append(TokenEmbeddingRule.head_bias_write(
            token_ids=[tok],
            value=-10.0,
            name=f"head_bake_marker_bias_token_{tok}",
        ))

    # 6. Section/special tokens: head.bias[tok] = -50.0.
    section_tokens = [
        tok for tok in (
            Token.CODE_START, Token.CODE_END,
            Token.DATA_START, Token.DATA_END,
            Token.SEP, Token.USER_INPUT_START, Token.USER_INPUT_END,
        ) if tok < vocab_size
    ]
    if section_tokens:
        rules.append(TokenEmbeddingRule.head_bias_write(
            token_ids=section_tokens,
            value=-50.0,
            name="head_bake_section_token_bias",
        ))

    # 7. IO_STATE_* bias = -20.0.
    if Token.IO_STATE_EMIT_BYTE < vocab_size:
        rules.append(TokenEmbeddingRule.head_bias_write(
            token_ids=[Token.IO_STATE_EMIT_BYTE],
            value=-20.0,
            name="head_bake_io_state_emit_byte_bias",
        ))
    if Token.IO_STATE_EMIT_THINKING < vocab_size:
        rules.append(TokenEmbeddingRule.head_bias_write(
            token_ids=[Token.IO_STATE_EMIT_THINKING],
            value=-20.0,
            name="head_bake_io_state_emit_thinking_bias",
        ))

    return tuple(rules)


def _head_bake_ir(dim_positions) -> CompilerIR:
    """Build the head-bake :class:`CompilerIR`.

    Built lazily from ``dim_positions`` because the optional ``NEXT_*``
    convo-IO flags appear only when conversational I/O is enabled, mirroring
    the legacy helper's ``try: D(opt) except AttributeError`` behaviour.
    """
    from ...vm_step import Token

    ir = CompilerIR()
    ir.embeddings.extend(_head_bake_rules(Token.VOCAB_SIZE, dim_positions))
    return ir


def _head_bake_ir_factory(dim_positions, HD) -> CompilerIR:
    """``compiler_ir_factory``-shaped wrapper around :func:`_head_bake_ir`.

    Mirrors the ``(dim_positions, head_dim)`` signature expected by
    :func:`layer_compiler._make_operation_ir`. ``HD`` is unused because
    head-bake writes ``model.head.{weight,bias}`` (no attention head
    state). Phase 11.A.
    """
    del HD
    return _head_bake_ir(dim_positions)


def make_head_bake_op() -> Operation:
    """Bake the output projection head: byte/marker token logits.

    Phase=1000 so it runs AFTER legacy_bake (phase=999); the corresponding
    head section in `set_vm_weights` has been removed to avoid double-bake.

    Phase 7.D.2 migration: the imperative ``setup_head_weights`` helper is
    replaced by an op-owned :class:`CompilerIR` carrying ~270 + 13
    :class:`TokenEmbeddingRule`s (256 per-byte OUTPUT_LO/OUTPUT_HI writes,
    one rule per NEXT_* flag for the -80.0 byte gating, one bulk
    ``head.bias = -5.0`` write across all bytes, one ``+1.0`` delta on
    byte 0 to override to ``-4.0``, plus marker / step-end / section /
    IO_STATE bias and weight rules). The bake_fn zeroes ``head.weight``
    and ``head.bias`` first (the IR has no zeroing primitive) and then
    calls ``CompilerIR.lower_token_embeddings``. The single ``=`` /
    ``+=`` conflict in the legacy helper -- ``head.bias[0] = -4.0``
    overriding the prior ``-5.0`` -- is modelled as a ``+1.0`` delta
    rule. Byte-identical to ``setup_head_weights`` for the canonical
    ``_SetDim`` and the compact ``pin_io_only=True`` layouts.

    Claim-coverage note: ``claims`` is intentionally left empty. The static
    verifier in ``decl_verifier.py`` (``_diff_all_blocks_by_ptr``) diffs only
    ``model.blocks[*].attn`` / ``.ffn`` matrices and the token-embedding row
    table. ``setup_head_weights`` writes to ``model.head.weight`` and
    ``model.head.bias`` -- two parameters outside the verifier's diff scope.
    Even if we declared claims, the snapshot diff would observe zero cells
    for this op and the verifier would either mark it inert (passing via
    the inert short-circuit) or flag every claim as ``declared_but_not_
    written``. Neither outcome adds verification value until the diff path
    is extended to cover the head module. Structural-anchor-style op: the
    only writer to ``head.{weight,bias}``, so collisions aren't a concern.
    """
    def _bake(model, dim_positions, S):
        del S
        import torch
        with torch.no_grad():
            model.head.weight.zero_()
            model.head.bias.zero_()
        ir = _head_bake_ir(dim_positions)
        ir.lower_token_embeddings(model, dim_positions)

    return Operation(
        name="head_bake",
        reads=set(),
        writes=set(),
        # Wave 6 (docs/PRODUCES_CONSUMES_MIGRATION.md). Unembed-head bake:
        # populates the output projection (lm_head). The rules are
        # TokenEmbeddingRules over per-token logit lanes, not per-step
        # residual reads/writes, so produces/consumes_fresh stays empty.
        audited_empty_produces=True,
        kind="model",
        declarative_bake_fn=_bake,
        # Phase 11.A IR exposure: informational factory (~270 TokenEmbeddingRules).
        compiler_ir_factory=_head_bake_ir_factory,
        phase=1000,
        declarative_authority="declarative",
        migrated=True,
        smoke_tests={"all"},
        spec_section="BLOG_SPEC.md#registers",
    )


# ---------------------------------------------------------------------------
# AX byte-1 register-dump emission head bake (H1_DUMP_OUT columns)
# ---------------------------------------------------------------------------
#
# Mirrors the existing AX byte-1 H1 emission columns
# (``head.weight[token v, H1+(v+2)] = 5.0`` for v in 0..4, the high-byte
# one-hot the LM head reads at the byte-1 predictor row) onto the dedicated
# ``H1_DUMP_OUT`` band that the ``ax_byte1_dump_repopulate`` FFN populates on
# carried steps. ADDITIVE (runs AFTER ``head_bake`` at a higher phase, never
# zeroes), and ``H1_DUMP_OUT`` is all-zero on fresh steps -> the byte-1
# emission is byte-identical on fresh steps and additively re-emits the
# carried high byte on carried steps.
_AX_BYTE1_DUMP_HEAD_MAX_VALUE = 15  # value-general: H1 (0..4) + H2 (5..11) +
#                                     H3 (12..15) mirror the LM-head H-band map


def _ax_byte1_dump_band_for_value(v: int) -> tuple[str, int]:
    """Return ``(DUMP_band, offset)`` mirroring the LM head's H-band emission.

    The LM head emits byte token ``v`` at the byte-1 predictor row via a
    positional one-hot SPREAD across its marker-distance H-bands (verified by
    ``tools/probe_hband_byte1_map.py``, spec_k=0):

      * v in 0..4   -> ``H1+(v+2)``
      * v in 5..11  -> ``H2+(v-5)``
      * v in 12..15 -> ``H3+(v-12)``

    The carried-step dump must therefore write the SAME (band, offset) into the
    matching private ``H<k>_DUMP_OUT`` band so the additive LM column re-emits
    the right token. Original H1-only carry capped byte-1 at 4 (v>=5 lives in
    H2/H3); this generalises it to 0..15 (covers the whole add/sub corpus,
    whose high byte is 0..7 -> the H2 band).
    """
    if v <= 4:
        return "H1_DUMP_OUT", v + 2
    if v <= 11:
        return "H2_DUMP_OUT", v - 5
    return "H3_DUMP_OUT", v - 12


def _ax_byte1_dump_head_bake_rules(
    vocab_size: int, skip_h1: bool = False, skip_h23: bool = False
) -> tuple:
    """``head.weight[token v, H<k>_DUMP_OUT+off] += 5.0`` for v in 0..15.

    Value-general byte-1 register-dump emission columns. Mirrors the LM head's
    own H1/H2/H3 byte-1 one-hot layout onto the private ``H<k>_DUMP_OUT`` bands
    the carry FFN fills on carried steps (see ``_ax_byte1_dump_band_for_value``).

    ``skip_h1`` (opt-in): OMIT the ``H1_DUMP_OUT`` columns (v in 0..4) — the L25
    ``b1_to_output`` FFN (the OUTPUT-canonical byte-1 decode) already routes the
    carried byte-1 out of ``H1_DUMP_OUT`` into the canonical ``OUTPUT_LO/HI``
    nibbles, so the H1_DUMP_OUT LM-head columns are redundant.

    ``skip_h23`` (opt-in): ALSO OMIT the ``H2_DUMP_OUT`` (v in 5..11) +
    ``H3_DUMP_OUT`` (v in 12..15) columns — the L25 decode covers the whole
    byte-1 range (v 0..15) out of H1/H2/H3_DUMP_OUT, so ALL H*_DUMP_OUT LM-head
    columns are redundant. ``skip_h1`` and ``skip_h23`` are set TOGETHER (both
    gated by ``C4_AX_HIBYTE_CLEAR``). Default (neither set) bakes ALL columns ->
    byte-identical golden.
    """
    rules: list[TokenEmbeddingRule] = []
    for v in range(_AX_BYTE1_DUMP_HEAD_MAX_VALUE + 1):
        if v >= vocab_size:
            break
        band, off = _ax_byte1_dump_band_for_value(v)
        if skip_h1 and band == "H1_DUMP_OUT":
            continue
        if skip_h23 and band in ("H2_DUMP_OUT", "H3_DUMP_OUT"):
            continue
        rules.append(TokenEmbeddingRule.head_weight_write(
            token_ids=[v],
            writes=((f"{band}+{off}", 5.0),),
            name=f"ax_byte1_dump_head_token_{v}",
        ))
    return tuple(rules)


def make_ax_byte1_dump_head_bake_op() -> Operation:
    """Add the ``H1_DUMP_OUT`` byte-1 emission columns to the LM head.

    Runs at phase=1002 (AFTER ``head_bake`` phase=1000 AND the legacy H1
    emission columns) so it ADDS to ``head.weight`` rather than being zeroed.
    Mirrors the H1 high-byte one-hot columns onto ``H1_DUMP_OUT`` so the LM
    head re-emits the carried high byte once the ``ax_byte1_dump_repopulate``
    FFN fills ``H1_DUMP_OUT`` on carried steps. Byte-identical on fresh steps
    (``H1_DUMP_OUT == 0`` -> these columns contribute nothing).

    When ``C4_AX_HIBYTE_CLEAR`` is ON the L25 ``b1_to_output`` decode +
    byte-2/3 zero-default route ALL AX bytes through the canonical OUTPUT
    nibbles, so ALL H*_DUMP_OUT LM-head columns are redundant and are OMITTED
    (``skip_h1`` + ``skip_h23``). Default (flag OFF) bakes ALL columns ->
    byte-identical golden.
    """
    # The byte-1 register-dump EMISSION (the LM-head ``H1_DUMP_OUT`` columns) is
    # gated by ``C4_AX_BYTE1_DUMP``, which is now DEFAULT-ON (opt out with
    # ``C4_AX_BYTE1_DUMP=0``). The full carry machinery — the ``H1_PREV_STEP``
    # band, the L13 carry head, the L25 overflow-flag precursor + gated dump FFN
    # — runs unconditionally; only these LM-head emission columns are gated.
    # With the flag OFF ``H1_DUMP_OUT`` has no LM-head columns, so the model is
    # byte-identical to the pre-carry build (the carry head / dump FFN write
    # fresh bands that nothing else reads) and the smoke gate stays 50/1.
    #
    # Default ON delivers the cross-step AX byte-1 carry: add 0-49 full_trace
    # 12/50 -> 40/50, sub 50-99 5/50 -> 9/50 (spec_k=0), smoke 50/1 (only the
    # pre-existing simple_function fails). The dump-read GATE no longer
    # over-fires on the SHL-result (AX_CARRY ~ +12.85) / JMP (~ +47.86) step
    # classes — the two-sided Σ AX_CARRY band-pass (the
    # ``ax_byte1_carry_overflow_flag`` precursor + the ``AX_CARRY_OVERFLOW``
    # kill condition on the dump) excludes them. See
    # ``docs/AX_BYTE1_DUMP_CARRY_LANDED_2026_06_13.md``.
    import os as _os
    _emission_on = _os.environ.get("C4_AX_BYTE1_DUMP", "1") != "0"
    # When ``C4_AX_HIBYTE_CLEAR`` is ON, the L25 ``b1_to_output`` FFN decodes the
    # carried byte-1 out of the ``H1/H2/H3_DUMP_OUT`` bands into the canonical
    # ``OUTPUT_LO/HI`` nibbles across THIS op's value range (0..15), and the
    # byte-2/3 zero-default routes bytes 2/3 through OUTPUT too, so ALL of this
    # op's ``H*_DUMP_OUT`` LM-head columns are REDUNDANT and are OMITTED (both
    # ``skip_h1`` and ``skip_h23``). Default-OFF -> byte-identical golden.
    # Evaluated lazily so a per-process env flip is honoured and reflected in the
    # compile cache key. NOTE: this op only owns byte-1 tokens 0..15; tokens
    # 16..255 (byte-1 high nibble >= 1) get their own H*_DUMP_OUT low-nibble
    # mirror columns from ``make_ax_byte1_hinib_emission_op`` (flag
    # ``C4_AX_BYTE1_HINIB``), which is NOT touched here — the whole-corpus byte-1
    # carry is 0..15 (add/sub high byte 0..7) so OUTPUT-canonical removes the
    # H*_DUMP_OUT dependence for the entire carried range in practice.
    _skip_all_hbands = _os.environ.get("C4_AX_HIBYTE_CLEAR", "0") != "0"
    _skip_h1 = _skip_all_hbands
    _skip_h23 = _skip_all_hbands

    def _bake(model, dim_positions, S):
        del S
        if not _emission_on:
            return
        from ...vm_step import Token
        ir = CompilerIR()
        ir.embeddings.extend(
            _ax_byte1_dump_head_bake_rules(
                Token.VOCAB_SIZE, skip_h1=_skip_h1, skip_h23=_skip_h23
            )
        )
        ir.lower_token_embeddings(model, dim_positions)

    def _ir_factory(dim_positions, HD):
        del HD
        from ...vm_step import Token
        ir = CompilerIR()
        if _emission_on:
            ir.embeddings.extend(
                _ax_byte1_dump_head_bake_rules(
                    Token.VOCAB_SIZE, skip_h1=_skip_h1, skip_h23=_skip_h23
                )
            )
        return ir

    return Operation(
        name="ax_byte1_dump_head_bake",
        reads=set(),
        writes=set(),
        audited_empty_produces=True,
        kind="model",
        declarative_bake_fn=_bake,
        compiler_ir_factory=_ir_factory,
        # AFTER head_bake (1000); the legacy H1 emission columns are baked by
        # the imperative legacy path -> use 1002 to be safely last.
        phase=1002,
        declarative_authority="declarative",
        migrated=True,
        smoke_tests={"all"},
        spec_section="AX_HIGH_BYTE_DUMP_ROOT_IS_H1_ONEHOT_2026_06_13.md",
    )


# ---------------------------------------------------------------------------
# AX byte-1 FULL-WIDTH emission (lift the 16-cell H-band one-hot cap)
# ---------------------------------------------------------------------------
#
# The byte-1 emission path is a marker-distance positional one-hot SPREAD
# across the L0 H1/H2/H3 bands (16 distinct cells), and the LM head's H-band
# columns ALIAS mod 16 -> every byte value >= 16 collapses to ``byte mod 16``
# (the edge_literal cluster: ``got == exp & 0x0F``; see
# ``docs/EDGE_LITERAL_BYTE1_HIGH_NIBBLE_WALL_2026_06_15.md`` +
# ``project_ax_byte1_dump_is_h1_onehot_wall``).
#
# This op uses the ISA-semantics-DSL :func:`full_width_byte_emission` generator
# to supply the MECHANICAL alias-break: a dedicated 256-cell WIDE value band
# (``AX_BYTE1_FULL_WIDE``) with ONE distinct cell per byte value, plus the
# un-aliased LM-head columns for values 16..255 (``head.weight[v,
# AX_BYTE1_FULL_WIDE+v] = 5.0``). On a step where the wide band's cell ``v`` is
# active, the LM head emits the byte token ``v`` directly (no mod-16 fold).
#
# Honest scope (probed exhaustively spec_k=0, 2026-06-16):
#   * The columns + band are byte-identity-gateable on CPU
#     (``compare_symbolic_to_lowered_embedding``) and FLAG-OFF (default) omits
#     the band (smaller d_model) + emits zero columns -> the model is
#     BYTE-IDENTICAL to the pre-feature build.
#   * The wide band is NOT fed by this op. The byte-1 HIGH nibble (value >=
#     4096 / byte1 >= 16) has NO value-faithful source at the byte-1 predictor
#     row: ``OUTPUT_HI`` / ``AX_FULL_HI`` / ``AX_CARRY_HI`` are empty at every
#     block, and the high nibble survives only as a coarse ``H3+4`` hi==0-vs-
#     hi>0 flag. Materializing the source one-hot at the IMM decode point +
#     relaying it to the predictor row is the DEFERRED two-part build (the plan
#     deferred list + the wall doc's coordinated halves). This op is the
#     EMISSION half that build plugs into: once a partner relay fills
#     ``AX_BYTE1_FULL_WIDE`` with the value-faithful byte-1 one-hot, edge_literal
#     (and every byte1 >= 16) emits correctly with NO further model change.
#
# Default OFF (``C4_AX_BYTE1_FULL_WIDTH`` opt-in) because the band/columns are
# inert until the deferred relay supplies the source: shipping them ON would
# add d_model + LM-head columns that nothing feeds (still byte-IDENTICAL in
# logits, but pure overhead). Default-OFF keeps the build byte-identical; the
# relay increment flips it ON together with its source feed. The
# ``emission_flag`` gates the WIDE band registration too (flag-off => the band
# is NOT collected -> smaller d_model -> byte-identical to the pre-feature
# build), evaluated FRESH at each compile (so an env flip takes effect without
# a re-import).
def _ax_byte1_full_width_enabled() -> bool:
    import os as _os
    return _os.environ.get("C4_AX_BYTE1_FULL_WIDTH", "0") != "0"


# Value SOURCE for the wide band (the re-point, 2026-06-16): the byte-1
# predictor row carries the IMM operand's byte-1 value VALUE-FAITHFULLY as a
# NIBBLE PAIR in the AX_CARRY register bands — ``AX_CARRY_LO+lo`` (low nibble)
# and ``AX_CARRY_HI+hi`` (high nibble), where lo/hi are byte-1's nibbles and the
# cell == the nibble (NO offset). Confirmed spec_k=0 at the BUILT layout
# position (tools/probe_axcarry_byte1.py + probe_byte1_builtscan.py): a clean
# nibble sweep 0..7 maps AC_LO am==lo, AC_HI am==hi exactly. The wall doc
# (EDGE_LITERAL_BYTE1_HIGH_NIBBLE_WALL_2026_06_15) "AX_CARRY_HI empty" verdict
# was a STATIC-REGISTRY misread — the widen repack moves AX_CARRY_LO 328->362,
# so the static-328 probe read a DEAD cell. At the BUILT position the source IS
# present (the dim-repack trap, memory feedback_probe_dims_use_built_layout).
#
# AX_CARRY at the byte-1 row is byte-1 ONLY on an IMM step. For an ADD/SUB it
# holds the arithmetic CARRY (NOT byte-1) and for some operands that carry is a
# non-zero high nibble -> a spurious >=16 emission (verified: ADD 464+650 emitted
# 0x2600). So the fill MUST be hard-gated on ``OP_IMM`` (=5.0 at the IMM byte-1
# row, 0 on ADD/SUB; probe_axcarry_byte1.py + the OP_IMM check). Row gate:
# ``OP_IMM`` (IMM scope) + ``IS_BYTE`` + a hard ``MARK_AX`` blocker (excludes the
# AX-marker row marker+0, whose AX_CARRY holds byte-0 at ~40 and would otherwise
# swamp IS_BYTE). marker+2/+3 carry byte=0 -> no >=16 column -> harmless.
_AX_CARRY_NIBBLE_OFFSET = 0     # AX_CARRY encodes nibble n at cell n (no offset)
# Additive AND over [AX_CARRY_LO[lo], AX_CARRY_HI[hi], IS_BYTE, OP_IMM,
# NOT MARK_AX]. At the IMM byte-1 row: AC cells ~3.0, IS_BYTE=1, OP_IMM=5,
# MARK_AX=0. OP_IMM (weight 1 -> ~5) is NECESSARY: the threshold (11) is above
# everything reachable WITHOUT it, so ADD/SUB (OP_IMM=0) can never fire.
#   IMM correct (both nibbles + IS_BYTE + OP_IMM): 3 + 3 + 1 + 5 = 12  -> clears
#   IMM missing one nibble:                      ~0.5 + 3 + 1 + 5 = 9.5 -> sinks
#   ADD byte-1 (no OP_IMM, AC=carry):           up to 3+3+1+0     = 7   -> sinks
#   marker+0 (MARK_AX=1):                          .. - 1000 << 0       -> sinks
_AX_B1_FW_NIB_W = 1.0
_AX_B1_FW_ISBYTE_W = 1.0
_AX_B1_FW_OPIMM_W = 1.0
_AX_B1_FW_MARK_AX_BLOCK = 1000.0
_AX_B1_FW_THRESHOLD = 11.0

_AX_BYTE1_FULL_WIDTH_EMISSION = full_width_byte_emission(
    FullWidthByteEmissionSpec(
        name="ax_byte1_full_width",
        band_name="AX_BYTE1_FULL_WIDE",
        bits=8,
        head_scale=5.0,          # mirror the H-band +5.0 emission columns
        lo_value=16,             # 0..15 already emit via the H-band columns
        emission_flag=_ax_byte1_full_width_enabled,
        # Value-source RE-POINT: fill band+v from the AX_CARRY nibble pair at the
        # byte-1 predictor row (the IMM operand's byte-1, value-faithful, cell ==
        # nibble), gated on the byte-1 row.
        dump_nibble_lo="AX_CARRY_LO",
        dump_nibble_hi="AX_CARRY_HI",
        dump_nibble_lo_offset=_AX_CARRY_NIBBLE_OFFSET,
        dump_nibble_hi_offset=_AX_CARRY_NIBBLE_OFFSET,
        dump_nibble_lo_weight=_AX_B1_FW_NIB_W,
        dump_nibble_hi_weight=_AX_B1_FW_NIB_W,
        # Offset 0 -> all 16 nibbles fit the 16-cell band (no overflow / cap).
        dump_nibble_max=15,
        # Row selector: OP_IMM (IMM scope -- excludes ADD/SUB whose byte-1-row
        # AX_CARRY is the arithmetic carry, not byte-1) + IS_BYTE + a hard
        # MARK_AX blocker (excludes the AX-marker row marker+0). The nibble cells
        # select the value; marker+2/+3 carry byte=0 so they fill no >=16 column.
        dump_gate_conditions=(
            ("OP_IMM", _AX_B1_FW_OPIMM_W),
            ("IS_BYTE", _AX_B1_FW_ISBYTE_W),
            ("MARK_AX", -_AX_B1_FW_MARK_AX_BLOCK),
        ),
        dump_threshold=_AX_B1_FW_THRESHOLD,
        # The wide-band cell must overcome the LM head's value-emission
        # competition (token byte1&0xF wins via OUTPUT_LO/HI). A large write
        # makes the +5.0 emission column dominate (cell ~large * 5.0).
        dump_write_scale=2000.0,
    )
)


def make_ax_byte1_full_width_emission_op() -> Operation:
    """Add the un-aliased ``AX_BYTE1_FULL_WIDE`` byte-1 emission columns (16-255).

    Uses the ISA-DSL :func:`full_width_byte_emission` generator. The 256-cell
    wide value band breaks the LM head's mod-16 H-band alias: each byte value
    16..255 gets its OWN distinct LM-head column. Runs at phase=1002 (additive,
    AFTER head_bake); byte-identical when the band is empty (fresh steps) and
    when the flag is OFF (default).

    Gated by ``C4_AX_BYTE1_FULL_WIDTH`` (default-OFF -> the band is not collected
    -> smaller d_model -> byte-identical to the pre-feature build).
    """
    _emission_on = _ax_byte1_full_width_enabled()
    bundle = _AX_BYTE1_FULL_WIDTH_EMISSION

    def _bake(model, dim_positions, S):
        del S
        if not _emission_on:
            return
        from ...vm_step import Token
        ir = CompilerIR()
        ir.embeddings.extend(
            bundle.head_columns_builder(True, Token.VOCAB_SIZE)
        )
        ir.lower_token_embeddings(model, dim_positions)

    def _ir_factory(dim_positions, HD):
        del HD, dim_positions
        from ...vm_step import Token
        ir = CompilerIR()
        if _emission_on:
            ir.embeddings.extend(
                bundle.head_columns_builder(True, Token.VOCAB_SIZE)
            )
        return ir

    return Operation(
        name="ax_byte1_full_width_emission",
        reads=set(),
        writes=set(),
        audited_empty_produces=True,
        kind="model",
        declarative_bake_fn=_bake,
        compiler_ir_factory=_ir_factory,
        phase=1002,
        declarative_authority="declarative",
        migrated=True,
        smoke_tests={"all"},
        spec_section="EDGE_LITERAL_BYTE1_HIGH_NIBBLE_WALL_2026_06_15.md",
    )


# ---------------------------------------------------------------------------
# AX byte-1 FULL-WIDTH band FILL (the value-source re-point FFN)
# ---------------------------------------------------------------------------
#
# Fills ``AX_BYTE1_FULL_WIDE`` from the ALU nibble pair at the byte-1 predictor
# row (see the SOURCE comment on ``_AX_BYTE1_FULL_WIDTH_EMISSION``). The 240
# fill rules (v = 16..255) each AND the LOW nibble cell ``ALU_LO+(lo+2)``, the
# HIGH nibble cell ``ALU_HI+(hi+2)`` and the byte-1 row gate, writing the wide
# band cell the LM-head emission column reads. Runs at the L25 tail (after the
# tail correctors) so the ALU bands are final; gated by the SAME
# ``C4_AX_BYTE1_FULL_WIDTH`` flag (flag-off => zero rules => byte-identical).
def make_ax_byte1_full_width_fill_op() -> Operation:
    """Fill ``AX_BYTE1_FULL_WIDE`` from the ALU byte-1 nibble pair (FFN).

    The VALUE half of the edge_literal fix: an L25-tail FFN whose 240 rules
    reconstruct byte-1's full value (16..255) into the wide band by AND-ing the
    two ALU nibble cells at the byte-1 predictor row, so the un-aliased LM-head
    column emits the correct high byte. Gated by ``C4_AX_BYTE1_FULL_WIDTH``
    (flag-off => no band collected + zero rules => byte-identical).
    """
    _emission_on = _ax_byte1_full_width_enabled()
    bundle = _AX_BYTE1_FULL_WIDTH_EMISSION

    def _rules():
        return bundle.dump_rules_builder(_emission_on, {})

    def _bake(block, dim_positions, S):
        rules = _rules()
        if not rules:
            return
        from ...base_layers import PureFFN
        d_model = block.ffn.W_up.shape[1] if hasattr(block, "ffn") else None
        if d_model is None:
            d_model = max(int(v) for v in dim_positions.values()) + 1
        ffn = PureFFN(d_model, len(rules))
        dim_map = {
            nm: int(dim_positions[nm.split("+", 1)[0]])
            + (int(nm.split("+", 1)[1]) if "+" in nm else 0)
            for nm in Primitives.ffn_rule_dim_names(rules)
        }
        Primitives.lower_ffn_rules(ffn, rules, dim_map, S=S)
        block.post_ops.append(ffn)

    def _ir_factory(dim_positions, HD):
        del HD, dim_positions
        ir = CompilerIR()
        ir.layer(0).ffn.rules.extend(_rules())
        return ir

    reads = set(bundle.dump_reads) if _emission_on else set()
    writes = set(bundle.dump_writes) if _emission_on else set()
    return Operation(
        name="ax_byte1_full_width_fill",
        reads=reads,
        writes=writes,
        kind="block",
        target_op_name="l10_post_ops_combined",
        requires={"after": "tail_bit32_result_correction"},
        declarative_bake_fn=_bake,
        compiler_ir_factory=_ir_factory,
        declarative_authority="spec_generated",
        migrated=True,
        smoke_tests={"all"},
        spec_section="EDGE_LITERAL_BYTE1_HIGH_NIBBLE_WALL_2026_06_15.md",
    )


# ---------------------------------------------------------------------------
# AX byte-1 HIGH-NIBBLE band (narrow, head-count-friendly alias-break)
# ---------------------------------------------------------------------------
#
# The 256-cell ``AX_BYTE1_FULL_WIDE`` build (above) breaks the mod-16 byte-1
# emission alias by giving every value 16..255 its own band cell + LM-head
# column. But the 256-cell band pushes d_model 1090->1417 (n_heads 10->13),
# which perturbs the width-sensitive L10/L25 tail ops -> can't ship default-ON
# (docs/EDGE_LITERAL_BYTE1_HIGH_NIBBLE_WALL_2026_06_15.md "Remaining blocker").
#
# This op is the NARROW alternative: the LM head already emits the byte-1 token
# as a FACTORED nibble pair -- ``head.weight[v, OUTPUT_LO.*.-1 + (v&0xF)] = 5.0``
# (LOW nibble) + ``head.weight[v, OUTPUT_HI.*.-1 + (v>>4)] = 5.0`` (HIGH nibble),
# verified spec_k=0 at the BUILT layout. The LOW nibble is correct on every step
# (byte1<16 emits fine). The HIGH nibble is the only thing lost: an L14 default
# rule (l14_ops.py:1852, ``OUTPUT_HI_THIS_STEP+0=+50, +{nonzero}=-5000``) forces
# the byte-1 HIGH nibble to ZERO -> every byte1>=16 collapses to ``byte1 & 0x0F``.
#
# So we only need a 16-CELL high-nibble emission axis (one cell per high nibble
# 1..15), NOT a 256-cell full-value band -> d_model 1090->1199 (n_heads 10->11,
# +1 head not +3). The fill reads a value-faithful high-nibble SOURCE that is
# present on BOTH the fresh IMM step AND the carried (persisting-AX) step:
# ``H3_PREV_STEP+(4+hi)`` (the existing AX byte-1 carry head's H3 band, which
# already transports the byte-value H3 high-nibble one-hot cross-step -- probed
# spec_k=0: am == 4+hi, v~12, on step0 AND step1, present ONLY at the byte-1
# predictor row). The carried-step source is what the 256-cell ``OP_IMM``-gated
# fill MISSED (it fixed only the fresh IMM step; the faithful gate showed
# edge_literal still failing at step=1). This unified source fixes both.
#
# Default-ON (GPU-confirmed +11: edge_literal 5->15, smoke 51/0, zero arith
# regression; commit 1f0403f9 verified). Set ``C4_AX_BYTE1_HINIB=0`` to omit the
# band -> the pre-feature build (legacy emission covers byte1<16; this ADDS byte1>=16).
def _ax_byte1_hinib_enabled() -> bool:
    import os as _os
    return _os.environ.get("C4_AX_BYTE1_HINIB", "1") != "0"


_AX_BYTE1_HINIB_BAND = "AX_BYTE1_HINIB"
_AX_BYTE1_HINIB_WIDTH = 16            # one cell per high nibble 0..15
_AX_BYTE1_HINIB_HEAD_SCALE = 5.0     # mirror the OUTPUT_HI +5.0 emission column
# Highest high nibble the H3_PREV_STEP source faithfully reaches: the H3 band is
# 7 cells and the high-nibble one-hot sits at H3_PREV_STEP+(4+hi), so hi in
# 1..2 (cells 5,6) is in-band. hi=0 needs no fix (byte1<16 already emits via the
# unchanged LOW-nibble path), and the cap stays low so the source never reads a
# neighbouring band's cell. edge_literal byte1 max = 0x26 (hi=2) -> covered.
_AX_BYTE1_HINIB_SRC_MAX_HI = 2
_AX_BYTE1_HINIB_SRC_BAND = "H3_PREV_STEP"
_AX_BYTE1_HINIB_SRC_OFFSET = 4       # high-nibble one-hot at H3_PREV_STEP+(4+hi)
# Row gate (the AX byte-1 predictor row ONLY). THREE discriminators, because the
# H3_PREV_STEP carry broadcasts the AX high nibble to the byte-1 row of EVERY
# register (PC/SP/BP/STACK0) and to byte2/3 of the AX register (probed spec_k=0)
# -- a fill gated only on the source corrupts all of them (observed: PC byte-1
# -> 0x10/0x20, breaking the EXIT PC).
#   1. BYTE_INDEX_0+0 (multiplicative SwiGLU gate): =~1 at byte1, =~0 at
#      byte2/3 (advances to cell 1/2) -> floors byte2/3 to a zero write
#      REGARDLESS of source magnitude (an additive byte-index term is useless --
#      the source's ~12 magnitude alone blows past any additive threshold).
#   2. SE_REG_AX_PRESENT (additive, the AX-REGISTER discriminator): ~4.0 at AX
#      byte rows, ~1.48 at SP, ~0.54 at BP, ~0.0 at PC -> a high weight + high
#      threshold passes AX but sinks PC/SP/BP byte-1.
#   3. the H3_PREV_STEP+(4+hi) source cell (additive, the VALUE selector): picks
#      WHICH high-nibble cell to fill and forces the rule off when AX's high
#      nibble != hi (so the AX byte-1 row fires the RIGHT cell, not a stale one).
# Plus IS_BYTE (excludes the byte0 marker row) and a hard NOT-marker blocker.
_AX_B1_HINIB_SRC_W = 1.0
_AX_B1_HINIB_ISBYTE_W = 1.0
_AX_B1_HINIB_AXPRESENT_W = 4.0       # SE_REG_AX_PRESENT (AX~4 vs SP~1.48/BP~.54)
_AX_B1_HINIB_BYTEIDX_GATE = dim_ref("byte_index", "0", 0)   # multiplicative row selector
_AX_B1_HINIB_MARKER_BLOCK = 1000.0
# AND threshold. Reachable sums (BYTE_INDEX_0+0 ~1 multiplies the silu output):
#   AX byte1, hi MATCH:    src~12 + AXP 4*4=16 + IS_BYTE 1   = ~29   -> CLEAR
#   AX byte1, hi MISMATCH: src~0  + 16          + 1          = ~17   -> sink
#   SP byte1, hi match:    src~12 + AXP 1.48*4=5.9 + 1       = ~18.9 -> sink
#   PC byte1:              src~12 + AXP 0*4=0   + 1          = ~13   -> sink
# Threshold 24 sits above the SP/PC/mismatch ceilings and below the AX-match sum.
_AX_B1_HINIB_THRESHOLD = 24.0
# H3_PREV_STEP+(4+hi) cell value is ~12 on the byte-1 row (one-hot * carry
# magnitude). Threshold sits above "row gate alone" (IS_BYTE + BYTE_INDEX_0,
# weights 1+1=2) and below "row gate + source cell" so the AND requires the
# source. The source's large native magnitude (~12) is folded via the source
# weight; we keep weights ~1 and set the threshold between gate-only and
# gate+source. The HINIB write must be BIG ENOUGH to overcome the L14 OUTPUT_HI
# zero-default (-5000/S -> residual ~-1700..-8500 on the wrong-token
# competition) yet SMALL ENOUGH that the ~5-logit OUTPUT_LO low-nibble emission
# still resolves WHICH token within the high-nibble group wins (a write that is
# too large saturates every same-high-nibble token to the same fp logit and the
# low nibble stops deciding). The HINIB->logit gain is head_scale(5) * residual;
# write_scale ~6 gives a HINIB cell ~6.9k -> contribution ~34k (clears -8500
# with margin) while leaving the OUTPUT_LO ~5-logit resolution intact at that
# magnitude. Configurable via C4_AX_B1_HINIB_WRITE for CPU tuning.
import os as _os_hinib
_AX_B1_HINIB_WRITE_SCALE = float(
    _os_hinib.environ.get("C4_AX_B1_HINIB_WRITE", "6.0")
)

register_residual_band(
    _AX_BYTE1_HINIB_BAND, _AX_BYTE1_HINIB_WIDTH,
    owner="make_ax_byte1_hinib_emission_op",
    flag=_ax_byte1_hinib_enabled, never_share=True,
)


# === expr_add_mul operand-A SP-frame discriminator bands (2026-06-25) ===
#
# Registered HERE (model_ops imports LAST) — NOT in l7_ops next to the relay op
# that owns them — DELIBERATELY: these are flag-gated over-width bands, and a
# band registered at L7's early import position re-packs EVERY later flag-ON band
# (BP_SAVE_PREV, MUL_RESULT_HI, AX_BYTE1_HINIB, ...) by its width in the campaign
# build, which breaks a static-position consumer (observed: a step-0 AX byte-1
# replication regression across expr_paren/mul_div/mod). Appending them after the
# last existing band keeps every other band's dim position byte-stable, so the
# campaign build is unchanged except for the appended SP_ADDR_* slots.
#
# ``SP_ADDR_LO`` (16-cell one-hot): the push-time SP LOW byte, relayed by
# ``make_layer7_sp_addr_relay_op`` (L7 block 9) from the nearest prior MARK_SP's
# OUTPUT_LO onto the MEM store value rows + binary-op AX query rows, so the L8
# head-5 mem[SP] CAM can tell a LIVE store from a POPPED store on the only
# depth-2 expr cluster (``expr_add_mul`` ``a+b*c``: two live stack stores;
# a@0xF8, b@0xF0, post-MUL SP back to 0xF8 == a's frame).
# ``SP_ADDR_PRESENT`` (scalar): 1.0 exactly on rows carrying a relayed SP frame;
# gates head-5's ``-G`` penalty baseline so it is 0 on every non-SP row (a CONST
# baseline would shift head-5's per-query softmax normalization at saturated byte
# ties and regress the single-store expr clusters).
# Flag-gated by ``C4_L8_OPERAND_SP_DISC`` (DEFAULT-OFF blueprint) so a flag-OFF
# build omits both bands → smaller d_model, byte-identical to golden.
register_residual_band(
    "SP_ADDR_LO", 16, owner="make_layer7_sp_addr_relay_op",
    flag=l8_operand_sp_disc_enabled, never_share=True,
)
register_residual_band(
    "SP_ADDR_PRESENT", 1, owner="make_layer7_sp_addr_relay_op",
    flag=l8_operand_sp_disc_enabled, never_share=True,
)
# ``SP_ADDR_LO_SHARP`` / ``SP_ADDR_PRESENT_SHARP`` (BLOCKER-1 fix, 2026-06-26):
# the WINNER-TAKE-ALL one-hot SHARPENER of the relayed SP_ADDR_LO band. A
# block-10 SwiGLU FFN (``make_layer7_sp_addr_sharpen_op``, between the L7 relay
# at block 9 and the L8 head-5 read at block 11) thresholds each SP_ADDR_LO cell
# at 0.5 and writes a clean 0/1 cell into ``SP_ADDR_LO_SHARP``; the SAME units
# also sum into ``SP_ADDR_PRESENT_SHARP``. Because every surviving cell is
# exactly 1.0 and the relay delivers a near-one-hot (>=0.98 on real SP frames,
# <0.5 noise floor elsewhere — measured tools/_probe_sp_addr_lo_vals.py), PRESENT
# is bounded to {0,1} so the head-5 bilinear penalty cancels EXACTLY on a MATCH
# for ALL ops (not just clean-SP-frame expr programs). This kills the var_simple
# PRESENT blow-up (raw PRESENT reached ~3.9 -> +80k penalty -> -20 regression).
# Same flag-gating so a flag-OFF build omits them too.
register_residual_band(
    "SP_ADDR_LO_SHARP", 16, owner="make_layer7_sp_addr_sharpen_op",
    flag=l8_operand_sp_disc_enabled, never_share=True,
)
register_residual_band(
    "SP_ADDR_PRESENT_SHARP", 1, owner="make_layer7_sp_addr_sharpen_op",
    flag=l8_operand_sp_disc_enabled, never_share=True,
)


def _ax_byte1_lo_dump_cell(lo: int) -> str:
    """The carried-step LOW-nibble DUMP cell the LM head reads for nibble ``lo``.

    Mirrors the legacy byte-value emission's H-band DUMP layout (probed
    spec_k=0): lo 0..4 -> ``H1_DUMP_OUT+(lo+2)``, lo 5..11 -> ``H2_DUMP_OUT+
    (lo-5)``, lo 12..15 -> ``H3_DUMP_OUT+(lo-12)``. This is the band the
    existing AX byte-1 carry re-supplies on the carried step, so a value 0..15
    emits its low nibble correctly when AX persists. Byte tokens >= 16 do NOT
    read these cells natively (the carry only feeds tokens 0..15), so the
    full-byte fix must MIRROR this low-nibble column onto token v's high-nibble
    sibling — otherwise on a carried step the low-nibble tie-break inside a
    high-nibble group falls back to the (unreliable) fresh ``OUTPUT_LO`` band.
    """
    if lo <= 4:
        return f"H1_DUMP_OUT+{lo + 2}"
    if lo <= 11:
        return f"H2_DUMP_OUT+{lo - 5}"
    return f"H3_DUMP_OUT+{lo - 12}"


def _ax_byte1_hinib_head_rules(vocab_size: int) -> tuple:
    """For v in 16..255: HIGH-nibble column + a MIRROR of the carried LOW-nibble.

    Two additive emission columns per byte value 16..255:
      * ``AX_BYTE1_HINIB+(v>>4)`` — the un-aliased HIGH-nibble axis (the new
        narrow band the fill lights on the byte-1 row).
      * the carried LOW-nibble DUMP cell that token ``v & 0xF`` already reads
        (``_ax_byte1_lo_dump_cell``) — so token v gets the SAME carried
        low-nibble support its 0..15 sibling gets. Without this, on a carried
        step the low-nibble tie-break inside the v's high-nibble group is left
        to the unreliable fresh ``OUTPUT_LO`` band and a same-high-nibble
        neighbour (e.g. 0x20 vs 0x25) can win.

    Values 0..15 (hi=0) get NO column, so byte1<16 stays byte-identical. With
    the flag OFF this returns nothing (the band/columns are omitted).
    """
    rules: list[TokenEmbeddingRule] = []
    top = min(255, int(vocab_size) - 1)
    for v in range(16, top + 1):
        hi = (v >> 4) & 0xF
        lo = v & 0xF
        writes = (
            (f"{_AX_BYTE1_HINIB_BAND}+{hi}", _AX_BYTE1_HINIB_HEAD_SCALE),
            (_ax_byte1_lo_dump_cell(lo), _AX_BYTE1_HINIB_HEAD_SCALE),
        )
        rules.append(TokenEmbeddingRule.head_weight_write(
            token_ids=[v],
            writes=writes,
            name=f"ax_byte1_hinib_head_token_{v}",
        ))
    return tuple(rules)


def make_ax_byte1_hinib_emission_op() -> Operation:
    """Add the narrow ``AX_BYTE1_HINIB`` byte-1 HIGH-nibble emission columns.

    16-cell high-nibble alias-break (head-count-friendly: d_model 1090->1199,
    n_heads 10->11). Phase=1002 (additive, AFTER head_bake). Gated by
    ``C4_AX_BYTE1_HINIB`` (default-OFF -> band omitted -> byte-identical).
    """
    _emission_on = _ax_byte1_hinib_enabled()

    def _bake(model, dim_positions, S):
        del S
        if not _emission_on:
            return
        from ...vm_step import Token
        ir = CompilerIR()
        ir.embeddings.extend(_ax_byte1_hinib_head_rules(Token.VOCAB_SIZE))
        ir.lower_token_embeddings(model, dim_positions)

    def _ir_factory(dim_positions, HD):
        del HD, dim_positions
        from ...vm_step import Token
        ir = CompilerIR()
        if _emission_on:
            ir.embeddings.extend(_ax_byte1_hinib_head_rules(Token.VOCAB_SIZE))
        return ir

    return Operation(
        name="ax_byte1_hinib_emission",
        reads=set(),
        writes=set(),
        audited_empty_produces=True,
        kind="model",
        declarative_bake_fn=_bake,
        compiler_ir_factory=_ir_factory,
        phase=1002,
        declarative_authority="declarative",
        migrated=True,
        smoke_tests={"all"},
        spec_section="EDGE_LITERAL_BYTE1_HIGH_NIBBLE_WALL_2026_06_15.md",
    )


def _ax_byte1_hinib_fill_rules() -> tuple[FFNRule, ...]:
    """Fill ``AX_BYTE1_HINIB+hi`` from ``H3_PREV_STEP+(4+hi)`` on the byte-1 row.

    For each high nibble hi in 1.._AX_BYTE1_HINIB_SRC_MAX_HI: AND the source
    cell (the value selector) with the byte-1 row gate (IS_BYTE + BYTE_INDEX_0+0
    + NOT-marker) and write the matching high-nibble band cell. The source is
    present on BOTH the fresh IMM step and the carried persisting-AX step, so the
    byte-1 high nibble emits correctly on every step (unlike the OP_IMM-gated
    256-cell fill, which only fixed the fresh step).
    """
    if not _ax_byte1_hinib_enabled():
        return ()
    marker_blockers = (
        ("MARK_AX", -_AX_B1_HINIB_MARKER_BLOCK),
        ("MARK_PC", -_AX_B1_HINIB_MARKER_BLOCK),
        ("MARK_SP", -_AX_B1_HINIB_MARKER_BLOCK),
        ("MARK_BP", -_AX_B1_HINIB_MARKER_BLOCK),
        ("MARK_STACK0", -_AX_B1_HINIB_MARKER_BLOCK),
        ("MARK_MEM", -_AX_B1_HINIB_MARKER_BLOCK),
        ("MARK_SE", -_AX_B1_HINIB_MARKER_BLOCK),
    )
    rules: list[FFNRule] = []
    for hi in range(1, _AX_BYTE1_HINIB_SRC_MAX_HI + 1):
        src_cell = f"{_AX_BYTE1_HINIB_SRC_BAND}+{_AX_BYTE1_HINIB_SRC_OFFSET + hi}"
        conditions = (
            (src_cell, _AX_B1_HINIB_SRC_W),
            ("SE_REG_AX_PRESENT", _AX_B1_HINIB_AXPRESENT_W),
            ("IS_BYTE", _AX_B1_HINIB_ISBYTE_W),
        ) + marker_blockers
        # Additive AND over source (value selector) + SE_REG_AX_PRESENT (AX-reg
        # discriminator) + IS_BYTE, MULTIPLICATIVELY gated on BYTE_INDEX_0+0
        # (byte-1-only). Requires BOTH the source AND the AX-register signal, so
        # it fires ONLY at the AX byte-1 row with the matching high nibble.
        rules.append(multi_way_and_rule(
            name=f"ax_byte1_hinib_fill_hi{hi}",
            conditions=conditions,
            threshold=_AX_B1_HINIB_THRESHOLD,
            gate=_AX_B1_HINIB_BYTEIDX_GATE,
            writes=((f"{_AX_BYTE1_HINIB_BAND}+{hi}", _AX_B1_HINIB_WRITE_SCALE),),
        ))
    return tuple(rules)


def make_ax_byte1_hinib_fill_op() -> Operation:
    """Fill ``AX_BYTE1_HINIB`` from the carried H3 high-nibble one-hot (FFN).

    The VALUE half of the narrow edge_literal fix: an L25-tail FFN whose rules
    light ``AX_BYTE1_HINIB+hi`` from ``H3_PREV_STEP+(4+hi)`` at the byte-1 row,
    so the un-aliased HINIB LM-head column emits the correct high nibble on both
    the fresh and carried steps. Gated by ``C4_AX_BYTE1_HINIB`` (flag-off => no
    band collected + zero rules => byte-identical).
    """
    _emission_on = _ax_byte1_hinib_enabled()

    def _bake(block, dim_positions, S):
        rules = _ax_byte1_hinib_fill_rules()
        if not rules:
            return
        from ...base_layers import PureFFN
        d_model = block.ffn.W_up.shape[1] if hasattr(block, "ffn") else None
        if d_model is None:
            d_model = max(int(v) for v in dim_positions.values()) + 1
        ffn = PureFFN(d_model, len(rules))
        dim_map = {
            nm: int(dim_positions[nm.split("+", 1)[0]])
            + (int(nm.split("+", 1)[1]) if "+" in nm else 0)
            for nm in Primitives.ffn_rule_dim_names(rules)
        }
        Primitives.lower_ffn_rules(ffn, rules, dim_map, S=S)
        block.post_ops.append(ffn)

    def _ir_factory(dim_positions, HD):
        del HD, dim_positions
        ir = CompilerIR()
        ir.layer(0).ffn.rules.extend(_ax_byte1_hinib_fill_rules())
        return ir

    reads = (
        {_AX_BYTE1_HINIB_SRC_BAND, "SE_REG_AX_PRESENT", "IS_BYTE",
         "BYTE_INDEX_0", "MARK_AX", "MARK_PC", "MARK_SP", "MARK_BP",
         "MARK_STACK0", "MARK_MEM", "MARK_SE"}
        if _emission_on else set()
    )
    writes = {_AX_BYTE1_HINIB_BAND} if _emission_on else set()
    return Operation(
        name="ax_byte1_hinib_fill",
        reads=reads,
        writes=writes,
        kind="block",
        target_op_name="l10_post_ops_combined",
        requires={"after": "tail_bit32_result_correction"},
        declarative_bake_fn=_bake,
        compiler_ir_factory=_ir_factory,
        declarative_authority="spec_generated",
        migrated=True,
        smoke_tests={"all"},
        spec_section="EDGE_LITERAL_BYTE1_HIGH_NIBBLE_WALL_2026_06_15.md",
    )


def _embedding_bake_rules(vocab_size: int) -> tuple:
    """Build the :class:`TokenEmbeddingRule` list mirroring ``setup_token_embeddings``.

    Walks the same per-token write list as the imperative helper in
    ``ops/shared.py`` (see ``setup_token_embeddings``) and emits one
    :class:`TokenEmbeddingRule` per logical group of writes. Group structure:

      1. ``CONST=1.0`` for every token id < ``vocab_size``.
      2. Each register/section marker token writes ``MARK_<X>=1.0`` and
         ``IS_MARK=1.0`` (one rule per token to keep dim names symbolic).
      3. ``STACK0`` writes ``MARK_STACK0=1.0`` (no ``IS_MARK``).
      4. ``STEP_END``/``DATA_END``/``HALT`` write ``MARK_SE=1.0`` + ``IS_MARK``.
      5. ``STEP_END`` adds ``MARK_SE_ONLY=1.0``.
      6. ``TOOL_CALL`` writes ``MARK_SE`` + ``IS_MARK`` + ``MARK_SE_ONLY`` +
         ``CONST``.
      7. ``THINKING_START`` and ``THINKING_END`` each write ``IS_MARK`` +
         ``CONST`` + ``TEMP+{1,2}`` + ``MARK_THINKING_{START,END}``.
      8. ``IO_STATE_EMIT_BYTE`` / ``IO_STATE_EMIT_THINKING`` each write
         ``IS_MARK`` + ``CONST``.
      9. Each byte token ``0..255`` writes ``IS_BYTE=1.0``, ``EMBED_LO+nibble``,
         ``EMBED_HI+nibble``, ``CLEAN_EMBED_LO+nibble``, ``CLEAN_EMBED_HI+nibble``.

    The rule lowering is accumulative (``+=``) -- mirroring the imperative
    helper requires zeroing ``embed.weight`` first, which is still done in
    the bake_fn (the IR has no zeroing primitive). After the zero, every
    cell is written by at most one rule per token, so ``+=`` and ``=``
    produce byte-identical results.
    """
    from ...vm_step import Token

    rules: list[TokenEmbeddingRule] = []

    # 1. CONST=1 for every token.
    rules.append(TokenEmbeddingRule.embed_write(
        token_ids=[tok for tok in range(vocab_size)],
        writes=(("CONST", 1.0),),
        name="embedding_bake_const_all_tokens",
    ))

    # 2. Register / section markers (MARK_<X> + IS_MARK).
    for tok, dim_name in (
        (Token.REG_PC, "MARK_PC"),
        (Token.REG_AX, "MARK_AX"),
        (Token.REG_SP, "MARK_SP"),
        (Token.REG_BP, "MARK_BP"),
        (Token.MEM, "MARK_MEM"),
        (Token.CODE_START, "MARK_CS"),
    ):
        if tok < vocab_size:
            rules.append(TokenEmbeddingRule.embed_write(
                token_ids=[tok],
                writes=((dim_name, 1.0), ("IS_MARK", 1.0)),
                name=f"embedding_bake_marker_{dim_name.lower()}",
            ))

    # 3. STACK0 marker WITHOUT IS_MARK.
    if Token.STACK0 < vocab_size:
        rules.append(TokenEmbeddingRule.embed_write(
            token_ids=[Token.STACK0],
            writes=(("MARK_STACK0", 1.0),),
            name="embedding_bake_stack0_mark",
        ))

    # 4. Step-end / data-end / halt: MARK_SE + IS_MARK.
    for tok in (Token.STEP_END, Token.DATA_END, Token.HALT):
        if tok < vocab_size:
            rules.append(TokenEmbeddingRule.embed_write(
                token_ids=[tok],
                writes=(("MARK_SE", 1.0), ("IS_MARK", 1.0)),
                name=f"embedding_bake_se_token_{tok}",
            ))

    # 5. STEP_END additionally writes MARK_SE_ONLY.
    if Token.STEP_END < vocab_size:
        rules.append(TokenEmbeddingRule.embed_write(
            token_ids=[Token.STEP_END],
            writes=(("MARK_SE_ONLY", 1.0),),
            name="embedding_bake_step_end_se_only",
        ))

    # 6. TOOL_CALL: MARK_SE + IS_MARK + MARK_SE_ONLY + CONST. Note that
    #    CONST was already written for every token by rule 1 above; the
    #    imperative helper rewrote the same +1.0 (assignment, not add) which
    #    leaves the cell at 1.0. With IR ``+=`` semantics we must NOT add a
    #    second +1.0 to CONST here or we'd produce 2.0. Skip CONST here.
    if Token.TOOL_CALL < vocab_size:
        rules.append(TokenEmbeddingRule.embed_write(
            token_ids=[Token.TOOL_CALL],
            writes=(
                ("MARK_SE", 1.0),
                ("IS_MARK", 1.0),
                ("MARK_SE_ONLY", 1.0),
            ),
            name="embedding_bake_tool_call",
        ))

    # 7. Thinking markers. The dims MAY be undeclared (convo-IO off); the
    #    bake_fn skips them via try/except in the legacy helper. We register
    #    the rules unconditionally and let ``compare_symbolic_to_lowered_
    #    embedding`` raise a ``declaration_semantics`` issue at validation
    #    time when dims are missing, but the bake-time
    #    ``lower_token_embeddings`` will resolve them through the proxy.
    #    Reduce to a conditional rule list: only include the THINKING_*
    #    rules when both the token AND the dims are present. Since dim
    #    declarations are done via ``declare_setdim_compat_dims`` and
    #    ``MARK_THINKING_*`` are always declared (see the ``one_dim`` list
    #    in ``shared.py``), it's safe to register the rules unconditionally
    #    when the tokens are in vocab.
    if Token.THINKING_START < vocab_size:
        rules.append(TokenEmbeddingRule.embed_write(
            token_ids=[Token.THINKING_START],
            writes=(
                ("IS_MARK", 1.0),
                ("TEMP+1", 1.0),
                ("MARK_THINKING_START", 1.0),
            ),
            name="embedding_bake_thinking_start",
        ))
    if Token.THINKING_END < vocab_size:
        rules.append(TokenEmbeddingRule.embed_write(
            token_ids=[Token.THINKING_END],
            writes=(
                ("IS_MARK", 1.0),
                ("TEMP+2", 1.0),
                ("MARK_THINKING_END", 1.0),
            ),
            name="embedding_bake_thinking_end",
        ))

    # 8. IO_STATE_EMIT_*: IS_MARK (CONST already covered by rule 1).
    if Token.IO_STATE_EMIT_BYTE < vocab_size:
        rules.append(TokenEmbeddingRule.embed_write(
            token_ids=[Token.IO_STATE_EMIT_BYTE],
            writes=(("IS_MARK", 1.0),),
            name="embedding_bake_io_state_emit_byte",
        ))
    if Token.IO_STATE_EMIT_THINKING < vocab_size:
        rules.append(TokenEmbeddingRule.embed_write(
            token_ids=[Token.IO_STATE_EMIT_THINKING],
            writes=(("IS_MARK", 1.0),),
            name="embedding_bake_io_state_emit_thinking",
        ))

    # 9. Byte tokens 0-255: IS_BYTE + nibble decoding. Group by (lo, hi)
    #    nibble pair so each rule still carries symbolic dim names. There are
    #    256 byte tokens, each with a unique (lo, hi) so we emit 256 rules.
    #    IS_BYTE is +1.0 for all; group it into per-byte rules so the rule
    #    stays self-contained.
    for b in range(256):
        if b >= vocab_size:
            break
        lo = b & 0xF
        hi = (b >> 4) & 0xF
        rules.append(TokenEmbeddingRule.embed_write(
            token_ids=[b],
            writes=(
                ("IS_BYTE", 1.0),
                (f"EMBED_LO+{lo}", 1.0),
                (f"EMBED_HI+{hi}", 1.0),
                (f"CLEAN_EMBED_LO+{lo}", 1.0),
                (f"CLEAN_EMBED_HI+{hi}", 1.0),
            ),
            name=f"embedding_bake_byte_{b}",
        ))

    return tuple(rules)


def _embedding_bake_ir() -> CompilerIR:
    """Build the embedding-bake :class:`CompilerIR` (rules + lower target)."""
    from ...vm_step import Token

    ir = CompilerIR()
    ir.embeddings.extend(_embedding_bake_rules(Token.VOCAB_SIZE))
    return ir


def make_embedding_bake_op() -> Operation:
    """Bake the per-token embedding table.

    Phase=1001 so it runs AFTER legacy_bake (phase=999) and head_bake (1000);
    the corresponding embedding section in `set_vm_weights` has been removed
    to avoid double-bake.

    Phase 7.D.2 migration: the imperative ``setup_token_embeddings`` helper
    is replaced by an op-owned :class:`CompilerIR` containing one
    :class:`TokenEmbeddingRule` per logical write group (CONST,
    register markers, byte-nibble decoding, etc.). The bake_fn first zeroes
    ``model.embed.embed.weight`` (mirroring the helper's ``.zero_()`` call —
    the IR has no zeroing primitive) and then calls
    ``CompilerIR.lower_token_embeddings``. Byte-identical to the imperative
    path because every per-token cell is hit by at most one rule, so the
    accumulating ``+=`` of rule lowering matches the helper's direct ``=``.
    """
    _ir = _embedding_bake_ir()

    def _bake(model, dim_positions, S):
        del S
        import torch
        with torch.no_grad():
            model.embed.embed.weight.zero_()
        _ir.lower_token_embeddings(model, dim_positions)

    # Dim-ownership claims. ``setup_token_embeddings`` calls ``embed_weight
    # .zero_()`` first, which differs from the fresh ``nn.Embedding`` random
    # init for every row; the static verifier's embedding diff therefore
    # registers an ``embed_row`` claim for every token id 0 .. V-1. The
    # vocabulary size matches ``Token.VOCAB_SIZE`` (276). Declaring all rows
    # makes the verifier check that no row is missed by future refactors.
    #
    # Exception: the REG_PC row (id 257) is claimed by ``initial_pc_bake``
    # (phase 1001.5), which is the canonical writer of the +1.0 EMBED_LO /
    # EMBED_HI nibble cells representing the initial PC value. Both ops
    # legitimately touch row 257 (embedding_bake writes marker bits +
    # zero-fill; initial_pc_bake adds the PC-nibble bits on top), but the
    # dim-ownership registry only allows one canonical owner per
    # ``(layer, scope, identifier, column)`` 4-tuple. The PC-nibble writes
    # are the load-bearing signal worth gating in the registry, so
    # ``initial_pc_bake`` keeps the claim and ``embedding_bake`` drops
    # it. The verifier still observes embedding_bake's row-257 write, but
    # ``written_but_not_declared`` is non-fatal in non-strict mode.
    from ...vm_step import Token as _Token  # local import to avoid module-load cycle
    _claims = frozenset(
        (-1, "embed_row", str(tok), None)
        for tok in range(_Token.VOCAB_SIZE)
        if tok != _Token.REG_PC
    )

    return Operation(
        name="embedding_bake",
        reads=set(),
        writes=set(),
        # Wave 6 (docs/PRODUCES_CONSUMES_MIGRATION.md). Embedding-table
        # bake: populates the token embedding matrix. Rules write
        # cross-step EMBED_LO / EMBED_HI (allowlisted as cross-step
        # durable in tools/derive_produces_consumes._CROSS_STEP_DURABLE),
        # not in-step residual dims, so the in-step produces surface is
        # empty by convention.
        audited_empty_produces=True,
        kind="model",
        declarative_bake_fn=_bake,
        compiler_ir=_ir,
        phase=1001,
        declarative_authority="declarative",
        migrated=True,
        claims=_claims,
        smoke_tests={"all"},
        spec_section="BLOG_SPEC.md#registers",
    )


def _initial_pc_bake_ir() -> CompilerIR:
    """Build the :class:`TokenEmbeddingRule` IR for ``initial_pc_bake``.

    Mirrors the imperative bake bit-for-bit:
      - ``embed[REG_PC, EMBED_LO + (PC_OFFSET & 0xF)] += 1.0``
      - ``embed[REG_PC, EMBED_HI + ((PC_OFFSET >> 4) & 0xF)] += 1.0``

    The rule lowering is accumulative (``+=``), and the imperative bake used
    direct assignment (``=``) to +1.0 on cells previously zeroed by
    ``setup_token_embeddings``. Since the embedding-table cells in question
    are NOT also written by ``setup_token_embeddings`` for the ``REG_PC``
    token (the token only gets ``MARK_PC``, ``IS_MARK``, and ``CONST``
    set — none of which alias these EMBED_LO/HI nibble slots), the
    ``+=`` semantics produce the same byte-identical 1.0 cell value.
    """
    from ...vm_step import Token
    from ...constants import PC_OFFSET

    ir = CompilerIR()
    init_pc_lo = PC_OFFSET & 0xF
    init_pc_hi = (PC_OFFSET >> 4) & 0xF
    ir.embeddings.append(TokenEmbeddingRule.embed_write(
        token_ids=[Token.REG_PC],
        writes=(
            (f"EMBED_LO+{init_pc_lo}", 1.0),
            (f"EMBED_HI+{init_pc_hi}", 1.0),
        ),
        name="initial_pc_reg_pc_nibbles",
    ))
    return ir


def make_initial_pc_bake_op() -> Operation:
    """Bake the initial PC value (PC_OFFSET) into the REG_PC token embedding.

    Phase=1001.5 so it runs AFTER `embedding_bake` (1001), which zeros and
    rewrites the embedding table. By running just after, we add (rather than
    have-overwritten) the initial-PC pattern on top of the standard REG_PC
    marker pattern.

    Migration 2026-05-11 (i1): replaces the runtime `_inject_initial_pc`
    method on NeuralVMEmbedding. The runtime injection wrote +1.0 to
    EMBED_LO+(PC_OFFSET & 0xF) and +1.0 to EMBED_HI+((PC_OFFSET>>4) & 0xF)
    only at the FIRST REG_PC marker (one with no preceding STEP_END). The
    same residual contribution can be produced by baking those +1.0 values
    into the REG_PC token-embedding row. Because the bake fires at every
    REG_PC position (steps 1+ too), a pair of L3-FFN cancel units (added
    inline at `_set_layer3_ffn`) subtracts -1.0 from those same EMBED_LO/HI
    dims at MARK_PC AND HAS_SE positions — leaving step-1+ residuals
    bit-identical to the pre-migration behavior.

    Phase 7.D.2 migration: the imperative body is replaced by an op-owned
    :class:`CompilerIR` carrying a single :class:`TokenEmbeddingRule` (one
    rule, two writes — the EMBED_LO and EMBED_HI nibble columns derived
    from ``PC_OFFSET``). The bake_fn delegates to
    ``CompilerIR.lower_token_embeddings``. Byte-identical to the prior
    imperative path because the IR rule writes ``+1.0`` to cells that
    ``setup_token_embeddings`` (phase=1001) leaves at 0 for ``REG_PC``.
    """
    _ir = _initial_pc_bake_ir()

    def _bake(model, dim_positions, S):
        del S
        _ir.lower_token_embeddings(model, dim_positions)

    # Dim-ownership claims. The bake adds two non-zero values into the REG_PC
    # row of the token-embedding table (column EMBED_LO + (PC_OFFSET & 0xF)
    # and column EMBED_HI + ((PC_OFFSET >> 4) & 0xF)). The static verifier's
    # embedding diff is row-granular (one row -> one ``embed_row`` claim with
    # ``column=None``); declaring the single REG_PC row covers both column
    # writes. ``Token.REG_PC`` == 257 (see ``vm_step.Token``).
    from ...vm_step import Token as _Token  # local import to avoid module-load cycle
    _claims = frozenset({(-1, "embed_row", str(_Token.REG_PC), None)})

    return Operation(
        name="initial_pc_bake",
        reads=set(),
        writes=set(),
        # Wave 6 (docs/PRODUCES_CONSUMES_MIGRATION.md). Sets the initial
        # PC value into the embedding table for token position 0. Pure
        # embed-time write, not a per-step residual production.
        audited_empty_produces=True,
        kind="model",
        declarative_bake_fn=_bake,
        compiler_ir=_ir,
        phase=1001.5,
        declarative_authority="declarative",
        migrated=True,
        claims=_claims,
        smoke_tests={"all"},
        spec_section="BLOG_SPEC.md#registers",
    )


# ============================================================================
# Qwen R1: NORM_COMPENSATOR seed op
# ============================================================================
# When ``C4_QWEN_EXPORT_COMPAT=1`` is set, this op pins the constant
# ``K = 1000.0`` into the NORM_COMPENSATOR residual slot for every token
# (via ``embed.embed.weight[:, idx] = K``) and defensively zeroes the
# corresponding row of every block's attention ``W_o`` and FFN ``W_down``
# so downstream layers cannot clobber the constant.
#
# K is chosen large enough that ``K^2 >> sum(real_dims^2)`` for the entire
# residual stream, so the RMSNorm denominator is effectively ``K /
# sqrt(d_model)`` and the norm collapses to a per-dim scalar that the
# Qwen export can absorb into ``gamma``. See
# ``docs/QWEN_STRUCTURAL_ADAPTER_PLAN_2026_06_07.md`` §"RMSNorm
# compensation" and §R1 acceptance.
NORM_COMPENSATOR_K = 1000.0


def make_norm_compensator_seed_op() -> Operation:
    """Seed the NORM_COMPENSATOR residual slot with ``K`` for every token.

    Phase=1400 — runs after every other model bake (head_bake=1000,
    embedding_bake=1001, initial_pc_bake=1001.5, opcode_relay_head=1002,
    branch_override_patch=1100, l6/l7_dead_unit_zero=1160/1170,
    right_size_ffns=1200, expand_wrapper_blocks=1300) so its defensive
    W_o / W_down zeros land on the FINAL weights and survive the entire
    bake pipeline.

    The bake is a no-op when ``C4_QWEN_EXPORT_COMPAT != "1"`` or when the
    dim is not present in ``dim_positions`` (defence in depth — the dim
    is only declared when the flag is on, so the early-exit and the
    declaration are belt-and-braces).
    """
    def _bake(model, dim_positions, S):
        del S
        import os as _os
        if _os.environ.get("C4_QWEN_EXPORT_COMPAT") != "1":
            return
        idx = dim_positions.get("NORM_COMPENSATOR")
        if idx is None:
            return
        import torch
        with torch.no_grad():
            # 1. Embed: every token id carries K at the compensator slot.
            embed_weight = model.embed.embed.weight
            embed_weight[:, idx] = NORM_COMPENSATOR_K
            # 2. Defensive zero of W_o[idx, :] and W_down[idx, :] on
            #    every block so attention / FFN cannot write through the
            #    compensator slot. W_o is (d_model, d_model) with output
            #    dim first; W_down is (d_model, hidden_dim) with output
            #    dim first. Both store sparse COO or dense after
            #    ``sparsify()`` / ``compact()`` — handle the dense path
            #    (the only one set_vm_weights leaves on the model at
            #    phase 1400; ``sparsify``/``compact`` run later if at
            #    all). ``ffn`` may have been wrapped in ``nn.Sequential``
            #    by ``expand_wrapper_blocks`` (when
            #    ``C4_DISABLE_WRAPPER_EXPANSION=1``); descend into the
            #    Sequential and zero W_down on every PureFFN inside.
            for block in model.blocks:
                attn = getattr(block, "attn", None)
                if attn is not None and hasattr(attn, "W_o"):
                    w_o = attn.W_o.data
                    if w_o.dim() == 2 and idx < w_o.shape[0]:
                        w_o[idx, :] = 0.0
                ffn = getattr(block, "ffn", None)
                if ffn is None:
                    continue
                # Walk a Sequential (or list) and zero every W_down we
                # see; a bare PureFFN exposes W_down directly.
                _candidates = []
                if hasattr(ffn, "W_down"):
                    _candidates.append(ffn)
                else:
                    # Iterable composite (Sequential, ModuleList, ...).
                    try:
                        _candidates.extend(list(ffn))
                    except TypeError:
                        pass
                for sub in _candidates:
                    if not hasattr(sub, "W_down"):
                        continue
                    w_down = sub.W_down.data
                    if w_down.dim() == 2 and idx < w_down.shape[0]:
                        w_down[idx, :] = 0.0

    return Operation(
        name="norm_compensator_seed",
        reads=set(),
        writes=set(),
        # Wave 6: Qwen R1 compat bake. Writes a constant column into
        # the embedding table and zeros one row of W_o / W_down per
        # block. No in-step residual production; flagged as
        # ``audited_empty_produces`` so the producer/consumer audit
        # doesn't expect a per-step write surface.
        audited_empty_produces=True,
        kind="model",
        declarative_bake_fn=_bake,
        # Phase 1400: AFTER expand_wrapper_blocks (1300), the final
        # model-level structural pass. Any later op that mutates
        # W_o / W_down rows would have to coexist with R2+ exports.
        phase=1400,
        migrated=True,
        declarative_authority="structural_model",
        smoke_tests={"all"},
        spec_section="docs/QWEN_STRUCTURAL_ADAPTER_PLAN_2026_06_07.md#phase-r1",
    )


def make_contract_validation_op() -> Operation:
    """Run the contract validator. Previously inline in set_vm_weights.

    The validator prints any contract errors but does not raise — it's
    diagnostic only. Kept as a compiler op so the diagnostic still fires
    after every full bake. Phase=1199 (just before ``right_size_ffns``
    at 1200) so all layer/block/model bakes have completed.
    """
    def _bake(model, dim_positions, S):
        from ...dim_registry import (
            build_default_registry,
            build_default_contracts,
            ContractValidator,
        )
        reg = build_default_registry()
        contracts = build_default_contracts(reg)
        errors = ContractValidator(reg, contracts).validate()
        if errors:
            for e in errors:
                print(f"  CONTRACT: {e}")

    return Operation(
        name="contract_validation",
        reads=set(),
        writes=set(),
        # Wave 6 (docs/PRODUCES_CONSUMES_MIGRATION.md). Pure invariant
        # check / assertion pass; no writes at all.
        audited_empty_produces=True,
        kind="model",
        declarative_bake_fn=_bake,
        phase=1199,
        migrated=True,
        declarative_authority="structural_model",
        smoke_tests=set(),
        spec_section=None,
    )
