"""Auto-extracted per-layer factories. See ../migrated_ops.py for history."""

from ..layer_compiler import Operation
from ..ir import FFNRule
from ..primitives import AO, AP, DeclarativeAttentionHeadSpec, Primitives
from .shared import _as_setdim_proxy


L6_ALL_STEP_JMP_PC_OVERRIDE_START_UNIT = 320
L6_ALL_STEP_JMP_PC_OVERRIDE_END_UNIT = 384
L6_IMM_FETCH_ROUTE_START_UNIT = 0
L6_IMM_FETCH_ROUTE_END_UNIT = 32
L6_IMM_CARRY_REFRESH_START_UNIT = 32
L6_IMM_CARRY_REFRESH_END_UNIT = 64
L6_EXIT_AX_ROUTE_START_UNIT = 64
L6_EXIT_AX_ROUTE_END_UNIT = 96
L6_NOP_AX_ROUTE_START_UNIT = 96
L6_NOP_AX_ROUTE_END_UNIT = 128
L6_JSR_AX_ROUTE_START_UNIT = 128
L6_JSR_AX_ROUTE_END_UNIT = 160
L6_JMP_AX_ROUTE_START_UNIT = 160
L6_JMP_AX_ROUTE_END_UNIT = 192
L6_DELAYED_JMP_PC_OVERRIDE_START_UNIT = 192
L6_DELAYED_JMP_PC_OVERRIDE_END_UNIT = 256
L6_FIRST_STEP_JMP_PC_OVERRIDE_START_UNIT = 256
L6_FIRST_STEP_JMP_PC_OVERRIDE_END_UNIT = 320
L6_HALT_DETECT_START_UNIT = 384
L6_HALT_DETECT_END_UNIT = 385
L6_TEMP_CLEANUP_START_UNIT = 385
L6_TEMP_CLEANUP_RULE_START_UNIT = 386
L6_TEMP_CLEANUP_END_UNIT = 417
L6_CMP3_CLEANUP_START_UNIT = 417
L6_CMP3_CLEANUP_END_UNIT = 418
L6_STACK_IDENTITY_START_UNIT = 418
L6_STACK_IDENTITY_END_UNIT = 514
L6_PSH_SP_DECREMENT_START_UNIT = 514
L6_PSH_SP_DECREMENT_END_UNIT = 546
L6_JSR_SP_DECREMENT_START_UNIT = 546
L6_JSR_SP_DECREMENT_END_UNIT = 578
L6_JSR_SP_FIXUP_START_UNIT = 578
L6_JSR_SP_FIXUP_END_UNIT = 580
L6_JSR_SP_BYTES_START_UNIT = 580
L6_JSR_SP_BYTES_END_UNIT = 584
L6_PSH_STACK0_WRITEBACK_START_UNIT = 584
L6_PSH_STACK0_WRITEBACK_END_UNIT = 616
L6_GETCHAR_AX_ROUTE_START_UNIT = 616
L6_GETCHAR_AX_ROUTE_END_UNIT = 648
L6_BZ_AX_ROUTE_START_UNIT = 648
L6_BZ_AX_ROUTE_END_UNIT = 680
L6_BNZ_AX_ROUTE_START_UNIT = 680
L6_BNZ_AX_ROUTE_END_UNIT = 712
L6_PSH_AX_ROUTE_START_UNIT = 712
L6_PSH_AX_ROUTE_END_UNIT = 744
L6_ADJ_AX_ROUTE_START_UNIT = 744
L6_ADJ_AX_ROUTE_END_UNIT = 776
L6_ADJ_SP_WRITEBACK_START_UNIT = 776
L6_ADJ_SP_WRITEBACK_END_UNIT = 808
L6_ENT_SP_WRITEBACK_START_UNIT = 808
L6_ENT_SP_WRITEBACK_END_UNIT = 840
L6_ENT_FIRST_STEP_SP_BYTE0_START_UNIT = 840
L6_ENT_FIRST_STEP_SP_BYTE0_END_UNIT = 872
L6_ENT_FIRST_STEP_SP_BYTES_START_UNIT = 872
L6_ENT_FIRST_STEP_SP_BYTES_END_UNIT = 878
L6_BZ_PC_OVERRIDE_START_UNIT = 878
L6_BZ_PC_OVERRIDE_END_UNIT = 942
L6_BNZ_PC_OVERRIDE_START_UNIT = 942
L6_BNZ_PC_OVERRIDE_END_UNIT = 1070
L6_OPCODE_CONTAMINATION_CLEANUP_START_UNIT = 1070
L6_OPCODE_CONTAMINATION_CLEANUP_END_UNIT = 1102
L6_MEM_LEAKAGE_CLEANUP_START_UNIT = 1102
L6_MEM_LEAKAGE_CLEANUP_END_UNIT = 1104
L6_ALU_CLEAR_START_UNIT = 1104
L6_ALU_CLEAR_END_UNIT = 1136
L6_BRANCH_PC_BYTE1_OVERRIDE_START_UNIT = 1136
L6_BRANCH_PC_BYTE1_OVERRIDE_END_UNIT = 1332
L6_ALL_STEP_JSR_PC_OVERRIDE_START_UNIT = 1410
L6_ALL_STEP_JSR_PC_OVERRIDE_END_UNIT = 1490
L6_BINARY_POP_SP_INCREMENT_START_UNIT = 2294
L6_BINARY_POP_SP_INCREMENT_END_UNIT = 2328
L6_ENT_AFTER_JSR_SP_BYTE0_FIXUP_START_UNIT = 1668
L6_ENT_AFTER_JSR_SP_BYTE0_FIXUP_END_UNIT = 1675


def _clear_ffn_unit_band(ffn, start: int, end: int) -> None:
    """Clear one hidden-unit band before an IR lowering claims it."""

    ffn.W_up.data[start:end, :] = 0
    ffn.b_up.data[start:end] = 0
    ffn.W_gate.data[start:end, :] = 0
    ffn.b_gate.data[start:end] = 0
    ffn.W_down.data[:, start:end] = 0


def _pc_target_lo_from_index(k: int) -> int:
    return (k * 8 + 2) & 0xF


def _pc_target_hi_from_index(k: int) -> int:
    return ((k * 8 + 2) >> 4) & 0xF


def _pc_target_hi_plus_odd_imm_hi_from_index(k: int) -> int:
    return (_pc_target_hi_from_index(k) + 8) & 0xF


def _jsr_opcode_nibble_conditions(
    *, blocker: float = -10.0
) -> tuple[tuple[str, float], ...]:
    """Match opcode byte 0x03 while blocking other residual opcode nibbles."""

    return (
        ("OPCODE_BYTE_LO+3", 1.0),
        ("OPCODE_BYTE_HI+0", 1.0),
    ) + tuple(
        (f"OPCODE_BYTE_LO+{k}", blocker)
        for k in range(16)
        if k != 3
    ) + tuple(
        (f"OPCODE_BYTE_HI+{k}", blocker)
        for k in range(16)
        if k != 0
    )


def _pc_target_byte1_lo_from_imm_hi(k: int) -> int:
    return (k >> 1) & 0xF


def _append_pc_byte0_direct_copy_rules(
    rules: list[FFNRule],
    *,
    name_prefix: str,
    conditions: tuple[tuple[str, float], ...],
    threshold: float,
    lo_source: str,
    hi_source: str,
    write_scale: float,
) -> None:
    """Copy an already-encoded branch target byte into PC byte 0.

    Compiler branch immediates are PC byte addresses, not instruction indexes.
    Recomputing ``imm * 8 + PC_OFFSET`` from the low nibble aliases targets
    whose byte addresses share a low nibble (for example 0x12 and 0x22).
    """

    for k in range(16):
        rules.append(FFNRule.gated_write(
            name=f"{name_prefix}_target_lo_{k}",
            conditions=conditions,
            threshold=threshold,
            gate=f"{lo_source}+{k}",
            writes=((f"OUTPUT_LO+{k}", write_scale),),
        ))
    for k in range(16):
        rules.append(FFNRule.gated_write(
            name=f"{name_prefix}_target_hi_{k}",
            conditions=conditions,
            threshold=threshold,
            gate=f"{hi_source}+{k}",
            writes=((f"OUTPUT_HI+{k}", write_scale),),
        ))


def _layer6_all_step_jmp_pc_override_rules(S: float) -> tuple[FFNRule, ...]:
    """CompilerIR rules for L6 all-step JMP PC override units 320..383."""

    rules = []
    threshold = 4.5
    conditions = (
        ("MARK_PC", 1.0),
        ("OP_JMP", 1.0),
        ("MARK_AX", -10.0),
    )
    write_scale = 2.0 / S

    for band, output_base in (("lo", "OUTPUT_LO"), ("hi", "OUTPUT_HI")):
        for k in range(16):
            rules.append(FFNRule.gated_write(
                name=f"l6_jmp_all_step_cancel_{band}_{k}",
                conditions=conditions,
                threshold=threshold,
                gate=f"{output_base}+{k}",
                gate_weight=-1.0,
                writes=((f"{output_base}+{k}", write_scale),),
            ))

    _append_pc_byte0_direct_copy_rules(
        rules,
        name_prefix="l6_jmp_all_step",
        conditions=conditions,
        threshold=threshold,
        lo_source="FETCH_LO",
        hi_source="FETCH_HI",
        write_scale=write_scale,
    )

    return tuple(rules)


def _layer6_imm_fetch_route_rules(S: float) -> tuple[FFNRule, ...]:
    """CompilerIR rules for L6 IMM FETCH -> OUTPUT units 0..31."""

    rules = []
    conditions = (
        ("OP_IMM", 1.0),
        ("OP_EXIT", -20.0),
        ("OP_JMP", -20.0),
        ("MARK_AX", 1.0),
        ("MARK_PC", -8.0),
        ("IS_BYTE", -10.0),
    )
    write_scale = 2.0 / S
    for band, source_base, output_base in (
        ("lo", "FETCH_LO", "OUTPUT_LO"),
        ("hi", "FETCH_HI", "OUTPUT_HI"),
    ):
        for k in range(16):
            rules.append(FFNRule.gated_write(
                name=f"l6_imm_fetch_to_output_{band}_{k}",
                conditions=conditions,
                threshold=4.0,
                gate=f"{source_base}+{k}",
                writes=((f"{output_base}+{k}", write_scale),),
            ))
    return tuple(rules)


def _layer6_imm_carry_refresh_rules(S: float) -> tuple[FFNRule, ...]:
    """CompilerIR rules for L6 IMM AX_CARRY refresh units 32..63."""

    rules = []
    conditions = (
        ("OP_IMM", 0.2),
        ("MARK_AX", 1.0),
        ("MARK_PC", -8.0),
        ("IS_BYTE", -10.0),
        ("OP_EXIT", -20.0),
        ("OP_JMP", -20.0),
    )
    write_scale = 2.0 / S
    for band, fetch_base, carry_base in (
        ("lo", "FETCH_LO", "AX_CARRY_LO"),
        ("hi", "FETCH_HI", "AX_CARRY_HI"),
    ):
        for k in range(16):
            rules.append(FFNRule.gated_write(
                name=f"l6_imm_carry_refresh_{band}_{k}",
                conditions=conditions,
                threshold=1.5,
                gate_terms=(
                    (f"{fetch_base}+{k}", 1.0),
                    (f"{carry_base}+{k}", -1.0),
                ),
                writes=((f"{carry_base}+{k}", write_scale),),
            ))
    return tuple(rules)


def _layer6_all_step_jsr_pc_override_rules(S: float) -> tuple[FFNRule, ...]:
    """CompilerIR rules for all-step JSR PC override units 1410..1473.

    JSR immediates are instruction indexes, so byte 0 is still derived as
    ``imm * 8 + 2``.  The high-byte correction must not fire when the current
    target has an even high immediate nibble and an odd FETCH_HI lane is only
    stale residual from an earlier target.
    """

    rules = []
    conditions = (
        ("MARK_PC", 20.0),
        *_jsr_opcode_nibble_conditions(),
        ("MARK_AX", -100.0),
        ("MARK_SP", -100.0),
        ("MARK_BP", -100.0),
        ("MARK_STACK0", -100.0),
        ("MARK_MEM", -100.0),
        ("NEXT_SE", -100.0),
        ("IS_BYTE", -100.0),
    )
    threshold = 21.5
    write_scale = 2.0 / S

    for band, output_base in (("lo", "OUTPUT_LO"), ("hi", "OUTPUT_HI")):
        for k in range(16):
            rules.append(FFNRule.gated_write(
                name=f"l6_jsr_all_step_cancel_{band}_{k}",
                conditions=conditions,
                threshold=threshold,
                gate=f"{output_base}+{k}",
                gate_weight=-1.0,
                writes=((f"{output_base}+{k}", write_scale),),
            ))

    for k in range(16):
        rules.append(FFNRule.gated_write(
            name=f"l6_jsr_all_step_target_lo_{k}",
            conditions=conditions,
            threshold=threshold,
            gate=f"FETCH_LO+{k}",
            writes=((f"OUTPUT_LO+{_pc_target_lo_from_index(k)}", write_scale),),
        ))
    for k in range(16):
        rules.append(FFNRule.gated_write(
            name=f"l6_jsr_all_step_target_hi_{k}",
            conditions=conditions,
            threshold=threshold,
            gate=f"FETCH_LO+{k}",
            writes=((f"OUTPUT_HI+{_pc_target_hi_from_index(k)}", write_scale),),
        ))
    odd_imm_hi_gate = tuple(
        (f"FETCH_HI+{k}", 1.0)
        for k in range(1, 16, 2)
    )
    even_imm_hi_blockers = tuple(
        (f"FETCH_HI+{k}", -10.0)
        for k in range(0, 16, 2)
    )
    for k in range(16):
        rules.append(FFNRule.gated_write(
            name=f"l6_jsr_all_step_target_hi_odd_imm_hi_correction_{k}",
            conditions=(
                ("MARK_PC", 20.0),
                *_jsr_opcode_nibble_conditions(blocker=-100.0),
                ("MARK_AX", -100.0),
                ("MARK_SP", -100.0),
                ("MARK_BP", -100.0),
                ("MARK_STACK0", -100.0),
                ("MARK_MEM", -100.0),
                ("NEXT_SE", -100.0),
                ("IS_BYTE", -100.0),
                (f"FETCH_LO+{k}", 1.0),
                *even_imm_hi_blockers,
            ),
            threshold=threshold + 0.5,
            gate_terms=odd_imm_hi_gate,
            writes=(
                (f"OUTPUT_HI+{_pc_target_hi_from_index(k)}", -write_scale),
                (
                    f"OUTPUT_HI+{_pc_target_hi_plus_odd_imm_hi_from_index(k)}",
                    write_scale,
                ),
            ),
        ))

    return tuple(rules)


def _layer6_exit_ax_route_rules(S: float) -> tuple[FFNRule, ...]:
    """CompilerIR rules for L6 EXIT AX_CARRY -> OUTPUT units 64..95."""

    return _layer6_ax_output_route_rules(
        name_prefix="l6_exit_ax_to_output",
        threshold=4.0,
        conditions=(
            ("OP_EXIT", 1.0),
            ("OP_IMM", -20.0),
            ("MARK_AX", 1.0),
            ("MARK_PC", -8.0),
            ("IS_BYTE", -1.0),
        ),
        S=S,
    )


def _layer6_nop_ax_route_rules(S: float) -> tuple[FFNRule, ...]:
    """CompilerIR rules for L6 NOP AX_CARRY -> OUTPUT units 96..127."""

    return _layer6_ax_output_route_rules(
        name_prefix="l6_nop_ax_to_output",
        threshold=4.0,
        conditions=(
            ("OP_NOP", 1.0),
            ("MARK_AX", 1.0),
            ("MARK_PC", -8.0),
            ("IS_BYTE", -10.0),
        ),
        S=S,
    )


def _layer6_jsr_ax_route_rules(S: float) -> tuple[FFNRule, ...]:
    """CompilerIR rules for L6 JSR AX_CARRY -> OUTPUT units 128..159."""

    return _layer6_ax_output_route_rules(
        name_prefix="l6_jsr_ax_to_output",
        threshold=4.0,
        conditions=(
            ("OP_JSR", 1.0),
            ("MARK_AX", 1.0),
            ("MARK_PC", -8.0),
            ("MARK_SP", -8.0),
            ("MARK_BP", -8.0),
            ("MARK_STACK0", -8.0),
            ("MARK_MEM", -8.0),
            ("IS_BYTE", -10.0),
        ),
        S=S,
    )


def _layer6_jmp_ax_route_rules(S: float) -> tuple[FFNRule, ...]:
    """CompilerIR rules for L6 JMP AX_CARRY -> OUTPUT units 160..191."""

    return _layer6_ax_output_route_rules(
        name_prefix="l6_jmp_ax_to_output",
        threshold=6.5,
        conditions=(
            ("OP_JMP", 1.0),
            ("MARK_AX", 1.0),
            ("HAS_SE", 1.0),
            ("MARK_PC", -1.0),
            ("IS_BYTE", -10.0),
        ),
        S=S,
    )


def _layer6_delayed_jmp_pc_override_rules(S: float) -> tuple[FFNRule, ...]:
    """CompilerIR rules for L6 delayed JMP PC override units 192..255."""

    rules = []
    conditions = (
        ("MARK_PC", 1.0),
        ("CMP+0", 1.0),
        ("MARK_AX", -10.0),
        ("CONST", -1000.0),
    )
    write_scale = 2.0 / S
    for band, output_base in (("lo", "OUTPUT_LO"), ("hi", "OUTPUT_HI")):
        for k in range(16):
            rules.append(FFNRule.gated_write(
                name=f"l6_delayed_jmp_cancel_{band}_{k}",
                conditions=conditions,
                threshold=5.5,
                gate=f"{output_base}+{k}",
                gate_weight=-1.0,
                writes=((f"{output_base}+{k}", write_scale),),
            ))
    _append_pc_byte0_direct_copy_rules(
        rules,
        name_prefix="l6_delayed_jmp",
        conditions=conditions,
        threshold=5.5,
        lo_source="AX_CARRY_LO",
        hi_source="AX_CARRY_HI",
        write_scale=write_scale,
    )
    return tuple(rules)


def _layer6_first_step_jmp_pc_override_rules(S: float) -> tuple[FFNRule, ...]:
    """CompilerIR rules for L6 first-step JMP PC override units 256..319."""

    rules = []
    conditions = (
        ("MARK_PC", 1.0),
        ("OP_JMP", 1.0),
        ("HAS_SE", -1.0),
        ("MARK_AX", -10.0),
    )
    write_scale = 2.0 / S
    threshold = 5.0
    for band, output_base in (("lo", "OUTPUT_LO"), ("hi", "OUTPUT_HI")):
        for k in range(16):
            rules.append(FFNRule.gated_write(
                name=f"l6_first_step_jmp_cancel_{band}_{k}",
                conditions=conditions,
                threshold=threshold,
                gate=f"{output_base}+{k}",
                gate_weight=-1.0,
                writes=((f"{output_base}+{k}", write_scale),),
            ))
    _append_pc_byte0_direct_copy_rules(
        rules,
        name_prefix="l6_first_step_jmp",
        conditions=conditions,
        threshold=threshold,
        lo_source="AX_CARRY_LO",
        hi_source="AX_CARRY_HI",
        write_scale=write_scale,
    )
    return tuple(rules)


def _layer6_halt_detect_rules(S: float) -> tuple[FFNRule, ...]:
    """CompilerIR rules for L6 EXIT halt conversion unit 384."""

    write_scale = 2.0 / S
    return (
        FFNRule.constant_write(
            name="l6_exit_halt_detect",
            conditions=(("CMP+1", 1.0), ("NEXT_SE", 1.0)),
            threshold=1.3,
            writes=(
                ("NEXT_HALT", write_scale),
                ("NEXT_SE", -write_scale),
            ),
        ),
    )


def _layer6_temp_cleanup_rules(S: float) -> tuple[FFNRule, ...]:
    """CompilerIR rules for L6 TEMP[1..31] cleanup units 386..416."""

    write_scale = 2.0 / S
    return tuple(
        FFNRule.gated_write(
            name=f"l6_temp_cleanup_{k}",
            conditions=(("MARK_PC", 1.0), ("IS_BYTE", -1.0)),
            threshold=0.5,
            gate=f"TEMP+{k}",
            gate_weight=-1.0,
            writes=((f"TEMP+{k}", write_scale),),
        )
        for k in range(1, 32)
    )


def _layer6_cmp3_cleanup_rules(S: float) -> tuple[FFNRule, ...]:
    """CompilerIR rules for L6 CMP[3] cleanup unit 417."""

    write_scale = 2.0 / S
    return (
        FFNRule.gated_write(
            name="l6_cmp3_cleanup",
            conditions=(("MARK_PC", 1.0), ("IS_BYTE", -1.0)),
            threshold=0.5,
            gate="CMP+3",
            gate_weight=-1.0,
            writes=(("CMP+3", write_scale),),
        ),
    )


def _layer6_stack_identity_rules(S: float) -> tuple[FFNRule, ...]:
    """CompilerIR rules for L6 SP/BP/STACK0 identity units 418..513."""

    rules = []
    write_scale = 2.0 / S
    for marker_name in ("MARK_SP", "MARK_BP", "MARK_STACK0"):
        label = marker_name.removeprefix("MARK_").lower()
        conditions = (
            (marker_name, 1.0),
            ("IS_BYTE", -1.0),
            ("HAS_SE", 1.0),
        )
        for band, source_base, output_base in (
            ("lo", "EMBED_LO", "OUTPUT_LO"),
            ("hi", "EMBED_HI", "OUTPUT_HI"),
        ):
            for k in range(16):
                rules.append(FFNRule.gated_write(
                    name=f"l6_{label}_identity_{band}_{k}",
                    conditions=conditions,
                    threshold=1.5,
                    gate=f"{source_base}+{k}",
                    writes=((f"{output_base}+{k}", write_scale),),
                ))
    return tuple(rules)


def _layer6_psh_sp_decrement_rules(S: float) -> tuple[FFNRule, ...]:
    """CompilerIR rules for L6 PSH SP decrement units 514..545."""

    return _layer6_sp_decrement_rules(
        name_prefix="l6_psh_sp_decrement",
        conditions=(("PSH_AT_SP", 1.0), ("MARK_SP", 1.0)),
        threshold=1.5,
        S=S,
    )


def _layer6_jsr_sp_decrement_rules(S: float) -> tuple[FFNRule, ...]:
    """CompilerIR rules for L6 JSR SP decrement units 546..577."""

    return _layer6_sp_decrement_rules(
        name_prefix="l6_jsr_sp_decrement",
        conditions=(("CMP+4", 1.0), ("MARK_SP", 1.0)),
        threshold=1.5,
        S=S,
    )


def _layer6_sp_decrement_rules(
    *,
    name_prefix: str,
    conditions: tuple[tuple[str, float], ...],
    threshold: float,
    S: float,
) -> tuple[FFNRule, ...]:
    rules = []
    write_scale = 2.0 / S
    for k in range(16):
        new_k = (k - 8) % 16
        rules.append(FFNRule.gated_write(
            name=f"{name_prefix}_lo_{k}",
            conditions=conditions,
            threshold=threshold,
            gate=f"EMBED_LO+{k}",
            writes=(
                (f"OUTPUT_LO+{new_k}", write_scale),
                (f"OUTPUT_LO+{k}", -write_scale),
            ),
        ))
    for k in range(16):
        new_k_borrow = (k - 1) % 16
        rules.append(FFNRule.gated_write(
            name=f"{name_prefix}_hi_{k}",
            conditions=conditions + tuple(
                (f"EMBED_LO+{lo_bit}", -1.0)
                for lo_bit in range(8, 16)
            ),
            threshold=threshold,
            gate=f"EMBED_HI+{k}",
            writes=(
                (f"OUTPUT_HI+{new_k_borrow}", write_scale),
                (f"OUTPUT_HI+{k}", -write_scale),
            ),
        ))
    return tuple(rules)


def _layer6_jsr_sp_fixup_rules(S: float) -> tuple[FFNRule, ...]:
    """CompilerIR rules for L6 JSR SP byte-0 fixup units 578..579."""

    write_scale = 2.0 / S
    return (
        FFNRule.constant_write(
            name="l6_jsr_sp_fixup_lo",
            conditions=(("OP_JSR", 0.2), ("MARK_SP", 1.0), ("HAS_SE", -1.0)),
            threshold=1.5,
            writes=(
                ("OUTPUT_LO+8", write_scale),
                ("OUTPUT_LO+0", -write_scale),
            ),
        ),
        FFNRule.constant_write(
            name="l6_jsr_sp_fixup_hi",
            conditions=(("OP_JSR", 0.2), ("MARK_SP", 1.0), ("HAS_SE", -1.0)),
            threshold=1.5,
            writes=(
                ("OUTPUT_HI+15", write_scale),
                ("OUTPUT_HI+0", -write_scale),
            ),
        ),
    )


def _layer6_jsr_sp_bytes_rules(S: float) -> tuple[FFNRule, ...]:
    """CompilerIR rules for L6 JSR SP byte fixups units 580..583."""

    rules = []
    specs = (
        ("byte1", "BYTE_INDEX_0", 15, 15),
        ("byte2", "BYTE_INDEX_1", 0, 0),
    )
    for label, byte_index, lo, hi in specs:
        conditions = (
            ("CMP+4", 1.0),
            (byte_index, 1.0),
            ("IS_BYTE", 1.0),
            ("H1+2", 1.0),
        )
        rules.append(FFNRule.gated_write(
            name=f"l6_jsr_sp_{label}_lo",
            conditions=conditions,
            threshold=3.5,
            gate="CONST",
            writes=((f"OUTPUT_LO+{lo}", 10.0 / S),),
        ))
        rules.append(FFNRule.gated_write(
            name=f"l6_jsr_sp_{label}_hi",
            conditions=conditions,
            threshold=3.5,
            gate="CONST",
            writes=((f"OUTPUT_HI+{hi}", 10.0 / S),),
        ))
    return tuple(rules)


def _layer6_psh_stack0_writeback_rules(S: float) -> tuple[FFNRule, ...]:
    """CompilerIR rules for L6 PSH STACK0 writeback units 584..615."""

    rules = []
    write_scale = 2.0 / S
    conditions = (("PSH_AT_SP", 1.0), ("MARK_STACK0", 1.0))
    for band, embed_base, alu_base, output_base in (
        ("lo", "EMBED_LO", "ALU_LO", "OUTPUT_LO"),
        ("hi", "EMBED_HI", "ALU_HI", "OUTPUT_HI"),
    ):
        for k in range(16):
            rules.append(FFNRule.gated_write(
                name=f"l6_psh_stack0_writeback_{band}_{k}",
                conditions=conditions,
                threshold=1.5,
                gate_terms=(
                    (f"{embed_base}+{k}", -1.0),
                    (f"{alu_base}+{k}", 1.0),
                ),
                writes=((f"{output_base}+{k}", write_scale),),
            ))
    return tuple(rules)


def _layer6_getchar_ax_route_rules(S: float) -> tuple[FFNRule, ...]:
    """CompilerIR rules for L6 GETCHAR AX passthrough units 616..647."""

    return _layer6_ax_output_route_rules(
        name_prefix="l6_getchar_ax_to_output",
        threshold=4.0,
        conditions=(
            ("OP_GETCHAR", 1.0),
            ("MARK_AX", 1.0),
            ("MARK_PC", -1.0),
        ),
        S=S,
    )


def _layer6_bz_ax_route_rules(S: float) -> tuple[FFNRule, ...]:
    """CompilerIR rules for L6 BZ AX passthrough units 648..679."""

    return _layer6_ax_output_route_rules(
        name_prefix="l6_bz_ax_to_output",
        threshold=4.0,
        conditions=(
            ("OP_BZ", 1.0),
            ("MARK_AX", 1.0),
            ("MARK_PC", -1.0),
        ),
        S=S,
    )


def _layer6_bnz_ax_route_rules(S: float) -> tuple[FFNRule, ...]:
    """CompilerIR rules for L6 BNZ AX passthrough units 680..711."""

    return _layer6_ax_output_route_rules(
        name_prefix="l6_bnz_ax_to_output",
        threshold=4.0,
        conditions=(
            ("OP_BNZ", 1.0),
            ("MARK_AX", 1.0),
            ("MARK_PC", -1.0),
        ),
        S=S,
    )


def _layer6_psh_ax_route_rules(S: float) -> tuple[FFNRule, ...]:
    """CompilerIR rules for L6 PSH AX passthrough units 712..743."""

    return _layer6_ax_output_route_rules(
        name_prefix="l6_psh_ax_to_output",
        threshold=4.0,
        conditions=(
            ("OP_PSH", 1.0),
            ("MARK_AX", 1.0),
            ("MARK_PC", -1.0),
        ),
        S=S,
    )


def _layer6_adj_ax_route_rules(S: float) -> tuple[FFNRule, ...]:
    """CompilerIR rules for L6 ADJ AX passthrough units 744..775."""

    return _layer6_ax_output_route_rules(
        name_prefix="l6_adj_ax_to_output",
        threshold=4.0,
        conditions=(
            ("OP_ADJ", 1.0),
            ("MARK_AX", 1.0),
            ("MARK_PC", -1.0),
        ),
        S=S,
    )


def _layer6_adj_sp_writeback_rules(S: float) -> tuple[FFNRule, ...]:
    """CompilerIR rules for L6 ADJ SP writeback units 776..807."""

    return _layer6_stack_writeback_rules(
        name_prefix="l6_adj_sp_writeback",
        conditions=(("OP_ADJ", 1.0), ("MARK_SP", 1.0)),
        threshold=1.5,
        S=S,
    )


def _layer6_ent_sp_writeback_rules(S: float) -> tuple[FFNRule, ...]:
    """CompilerIR rules for L6 ENT SP writeback units 808..839."""

    return _layer6_stack_writeback_rules(
        name_prefix="l6_ent_sp_writeback",
        conditions=(
            ("OP_ENT", 1.0),
            ("MARK_SP", 1.0),
            ("HAS_SE", 1.0),
        ),
        threshold=2.5,
        S=S,
    )


def _layer6_ent_first_step_sp_byte0_rules(S: float) -> tuple[FFNRule, ...]:
    """CompilerIR rules for L6 ENT first-step SP byte 0 units 840..871."""

    rules = []
    conditions = (
        ("OP_ENT", 1.0),
        ("MARK_SP", 1.0),
        ("HAS_SE", -10.0),
    )
    for imm_lo in range(16):
        result_lo = (-8 - imm_lo) % 16
        rules.append(FFNRule.gated_write(
            name=f"l6_ent_first_step_sp_byte0_lo_{imm_lo}",
            conditions=conditions,
            threshold=1.5,
            gate=f"FETCH_LO+{imm_lo}",
            writes=((f"OUTPUT_LO+{result_lo}", 5.0 / S),),
        ))
    for imm_hi in range(16):
        result_hi = (-1 - imm_hi) % 16
        rules.append(FFNRule.gated_write(
            name=f"l6_ent_first_step_sp_byte0_hi_{imm_hi}",
            conditions=conditions,
            threshold=1.5,
            gate=f"FETCH_HI+{imm_hi}",
            writes=((f"OUTPUT_HI+{result_hi}", 5.0 / S),),
        ))
    return tuple(rules)


def _layer6_ent_first_step_sp_bytes_rules(S: float) -> tuple[FFNRule, ...]:
    """CompilerIR rules for L6 ENT first-step SP bytes 1..3 units 872..877."""

    rules = []
    conditions_by_byte = (
        ("byte1", "BYTE_INDEX_0", 15, 15, 10.0 / S),
        ("byte2", "BYTE_INDEX_1", 0, 0, 5.0 / S),
        ("byte3", "BYTE_INDEX_2", 0, 0, 5.0 / S),
    )
    for label, byte_index, lo, hi, scale in conditions_by_byte:
        conditions = (
            ("OP_ENT", 1.0),
            (byte_index, 1.0),
            ("IS_BYTE", 1.0),
            ("H1+2", 1.0),
            ("HAS_SE", -10.0),
        )
        rules.append(FFNRule.gated_write(
            name=f"l6_ent_first_step_sp_{label}_lo",
            conditions=conditions,
            threshold=4.0,
            gate="CONST",
            writes=((f"OUTPUT_LO+{lo}", scale),),
        ))
        rules.append(FFNRule.gated_write(
            name=f"l6_ent_first_step_sp_{label}_hi",
            conditions=conditions,
            threshold=4.0,
            gate="CONST",
            writes=((f"OUTPUT_HI+{hi}", scale),),
        ))
    return tuple(rules)


def _layer6_ent_after_jsr_sp_byte0_fixup_rules(S: float) -> tuple[FFNRule, ...]:
    """Correct ENT SP byte 0 after a JSR push leaves SP byte 0 at 0xf8."""

    del S
    return (
        FFNRule.constant_write(
            name="l6_ent_after_jsr_sp_byte0_e8",
            conditions=(
                ("OP_ENT", 1.0),
                ("MARK_SP", 1.0),
                ("HAS_SE", 1.0),
                ("EMBED_LO+8", 1.0),
                ("EMBED_HI+15", 1.0),
            ),
            threshold=7.5,
            writes=(
                ("OUTPUT_LO+8", 0.10),
                ("OUTPUT_HI+14", 0.05),
                ("OUTPUT_LO+0", -0.05),
                ("OUTPUT_LO+10", -0.05),
                ("OUTPUT_LO+14", -0.05),
                ("OUTPUT_HI+1", -0.05),
                ("OUTPUT_HI+15", -0.05),
            ),
            # F-9: fires at SP marker rows for ENT-after-JSR (any non-zero
            # ENT immediate whose ones-place is 8 — e.g. SP=0xffe8 for
            # ENT 1..8 family member). EMBED_LO+8/HI+15 immediate-value
            # constraint not expressible at slot granularity (EMBED nibble
            # semantics widen to tautology), so we keep scope loose.
            scope="mark == SP AND opcode_in_step in {ENT}",
            dominates_at={
                "OUTPUT_LO": "mark == SP AND opcode_in_step in {ENT}",
                "OUTPUT_HI": "mark == SP AND opcode_in_step in {ENT}",
            },
        ),
        FFNRule.constant_write(
            name="l6_ent_after_jsr_sp_byte0_f0_when_ent_zero",
            conditions=(
                ("OP_ENT", 1.0),
                ("MARK_SP", 1.0),
                ("HAS_SE", 1.0),
                ("EMBED_LO+0", 1.0),
                ("EMBED_HI+15", 1.0),
                ("EMBED_LO+8", -1.0),
            ),
            threshold=7.5,
            writes=(
                ("OUTPUT_LO+0", 0.10),
                ("OUTPUT_HI+15", 0.10),
                ("OUTPUT_LO+8", -0.05),
                ("OUTPUT_LO+10", -0.05),
                ("OUTPUT_HI+0", -0.05),
                ("OUTPUT_HI+14", -0.05),
            ),
            # F-9: SP marker row for ENT 0 (SP byte0 = 0xf0). ENT-immediate
            # discrimination via EMBED nibble lanes is not visible to the
            # scope DSL; loose scope matches the other SP-byte0 family
            # members.
            scope="mark == SP AND opcode_in_step in {ENT}",
            dominates_at={
                "OUTPUT_LO": "mark == SP AND opcode_in_step in {ENT}",
                "OUTPUT_HI": "mark == SP AND opcode_in_step in {ENT}",
            },
        ),
        FFNRule.constant_write(
            name="l6_ent_after_jsr_bp_byte0_f0",
            conditions=(
                ("OP_ENT", 1.0),
                ("MARK_BP", 1.0),
                ("HAS_SE", 1.0),
            ),
            threshold=6.5,
            writes=(
                ("OUTPUT_LO+0", 0.30),
                ("OUTPUT_HI+15", 0.30),
                ("OUTPUT_LO+8", -0.10),
                ("OUTPUT_HI+1", -0.10),
            ),
            # F-9: BP marker row right after JSR; BP byte 0 -> 0xf0.
            scope="mark == BP AND opcode_in_step in {ENT}",
            dominates_at={
                "OUTPUT_LO": "mark == BP AND opcode_in_step in {ENT}",
                "OUTPUT_HI": "mark == BP AND opcode_in_step in {ENT}",
            },
        ),
        FFNRule.constant_write(
            name="l6_ent_after_jsr_bp_byte1_ff",
            conditions=(
                ("OP_ENT", 1.0),
                ("IS_BYTE", 1.0),
                ("H1+3", 1.0),
                ("BYTE_INDEX_0", 1.0),
                ("HAS_SE", 1.0),
            ),
            threshold=8.5,
            writes=(
                ("OUTPUT_LO+15", 0.10),
                ("OUTPUT_HI+15", 0.10),
                ("OUTPUT_LO+0", -0.10),
                ("OUTPUT_HI+0", -0.10),
            ),
            # F-9: byte-position 0 row carrying the BP byte 1 staging
            # under H1+3 staging. BP-discrimination via H1+3 not modeled
            # in slot semantics (H1+3 widens to a permissive predicate).
            scope="is_byte AND byte_index == 0 AND opcode_in_step in {ENT}",
            dominates_at={
                "OUTPUT_LO": "is_byte AND byte_index == 0 AND opcode_in_step in {ENT}",
                "OUTPUT_HI": "is_byte AND byte_index == 0 AND opcode_in_step in {ENT}",
            },
        ),
        FFNRule.constant_write(
            name="l6_ent_after_jsr_bp_byte2_00",
            conditions=(
                ("OP_ENT", 1.0),
                ("IS_BYTE", 1.0),
                ("H1+3", 1.0),
                ("BYTE_INDEX_1", 1.0),
                ("HAS_SE", 1.0),
            ),
            threshold=8.5,
            writes=(
                ("OUTPUT_LO+0", 0.30),
                ("OUTPUT_HI+0", 0.30),
                ("OUTPUT_LO+1", -0.30),
                ("OUTPUT_LO+15", -0.10),
                ("OUTPUT_HI+15", -0.10),
            ),
            # F-9: byte-position 1 row for BP byte 2 = 0x00.
            scope="is_byte AND byte_index == 1 AND opcode_in_step in {ENT}",
            dominates_at={
                "OUTPUT_LO": "is_byte AND byte_index == 1 AND opcode_in_step in {ENT}",
                "OUTPUT_HI": "is_byte AND byte_index == 1 AND opcode_in_step in {ENT}",
            },
        ),
        FFNRule.constant_write(
            name="l6_ent_after_jsr_bp_byte3_00",
            conditions=(
                ("OP_ENT", 1.0),
                ("IS_BYTE", 1.0),
                ("H1+3", 1.0),
                ("BYTE_INDEX_2", 1.0),
                ("HAS_SE", 1.0),
            ),
            threshold=8.5,
            writes=(
                ("OUTPUT_LO+0", 0.30),
                ("OUTPUT_HI+0", 0.30),
                ("OUTPUT_LO+1", -0.30),
                ("OUTPUT_LO+15", -0.10),
                ("OUTPUT_HI+15", -0.10),
            ),
            # F-9: byte-position 2 row for BP byte 3 = 0x00.
            scope="is_byte AND byte_index == 2 AND opcode_in_step in {ENT}",
            dominates_at={
                "OUTPUT_LO": "is_byte AND byte_index == 2 AND opcode_in_step in {ENT}",
                "OUTPUT_HI": "is_byte AND byte_index == 2 AND opcode_in_step in {ENT}",
            },
        ),
        FFNRule.constant_write(
            name="l6_ent_after_jsr_stack0_byte0_00",
            conditions=(
                ("OP_ENT", 1.0),
                ("MARK_STACK0", 1.0),
                ("HAS_SE", 1.0),
            ),
            threshold=6.5,
            writes=(
                ("OUTPUT_LO+0", 0.30),
                ("OUTPUT_HI+0", 0.30),
                ("OUTPUT_LO+2", -0.10),
                ("OUTPUT_LO+12", -0.10),
                ("OUTPUT_HI+1", -0.10),
                ("OUTPUT_HI+2", -0.10),
            ),
            # F-9: STACK0 marker row right after JSR; byte 0 -> 0x00.
            scope="mark == STACK0 AND opcode_in_step in {ENT}",
            dominates_at={
                "OUTPUT_LO": "mark == STACK0 AND opcode_in_step in {ENT}",
                "OUTPUT_HI": "mark == STACK0 AND opcode_in_step in {ENT}",
            },
        ),
    )


def _layer6_bz_pc_override_rules(S: float) -> tuple[FFNRule, ...]:
    """CompilerIR rules for L6 BZ PC override units 878..941."""

    rules = []
    cancel_conditions = (
        ("MARK_PC", 1.0),
        ("OP_BZ", 0.2),
        ("CMP+4", 1.0),
        ("CMP+5", 1.0),
        ("IS_BYTE", -10.0),
    )
    target_conditions = cancel_conditions + (("MARK_STACK0", -10.0),)
    write_scale = 2.0 / S
    for band, output_base in (("lo", "OUTPUT_LO"), ("hi", "OUTPUT_HI")):
        for k in range(16):
            rules.append(FFNRule.gated_write(
                name=f"l6_bz_cancel_{band}_{k}",
                conditions=cancel_conditions,
                threshold=3.5,
                gate=f"{output_base}+{k}",
                gate_weight=-1.0,
                writes=((f"{output_base}+{k}", write_scale),),
            ))
    _append_pc_byte0_direct_copy_rules(
        rules,
        name_prefix="l6_bz",
        conditions=target_conditions,
        threshold=3.5,
        lo_source="FETCH_LO",
        hi_source="FETCH_HI",
        write_scale=write_scale,
    )
    return tuple(rules)


def _layer6_bnz_pc_override_rules(S: float) -> tuple[FFNRule, ...]:
    """CompilerIR rules for L6 BNZ PC override units 942..1069."""

    rules = []
    write_scale = 2.0 / S
    groups = (
        (
            "lo_nonzero",
            (("MARK_PC", 1.0), ("OP_BNZ", 0.2), ("CMP+4", -1.0)),
            1.5,
        ),
        (
            "hi_nonzero",
            (
                ("MARK_PC", 1.0),
                ("OP_BNZ", 0.2),
                ("CMP+4", 1.0),
                ("CMP+5", -1.0),
            ),
            2.5,
        ),
    )
    for group, conditions, threshold in groups:
        for band, output_base in (("lo", "OUTPUT_LO"), ("hi", "OUTPUT_HI")):
            for k in range(16):
                rules.append(FFNRule.gated_write(
                    name=f"l6_bnz_{group}_cancel_{band}_{k}",
                    conditions=conditions,
                    threshold=threshold,
                    gate=f"{output_base}+{k}",
                    gate_weight=-1.0,
                    writes=((f"{output_base}+{k}", write_scale),),
                ))
        _append_pc_byte0_direct_copy_rules(
            rules,
            name_prefix=f"l6_bnz_{group}",
            conditions=conditions,
            threshold=threshold,
            lo_source="FETCH_LO",
            hi_source="FETCH_HI",
            write_scale=write_scale,
        )
    return tuple(rules)


def _layer6_branch_pc_byte1_override_rules(S: float) -> tuple[FFNRule, ...]:
    """Emit PC byte 1 for taken BZ/BNZ targets above 255.

    The marker-row override emits target byte 0.  The next token row
    (PC byte 0, identified by H1[PC] + BYTE_INDEX_0) must predict byte 1.
    For instruction-index immediates, byte1((imm * 8) + 2) == imm >> 5, which
    is the high immediate nibble shifted right by one.
    """

    rules = []
    write_scale = 2.0 / S
    const_zero_scale = 10.0 / S
    byte0_conditions = (
        ("IS_BYTE", 1.0),
        ("H1+0", 1.0),
        ("BYTE_INDEX_0", 1.0),
    )

    groups = (
        (
            "bz_zero",
            byte0_conditions
            + (
                ("OP_BZ", 0.2),
                ("CMP+4", 1.0),
                ("CMP+5", 1.0),
            ),
            5.5,
        ),
        (
            "bnz_lo_nonzero",
            byte0_conditions
            + (
                ("OP_BNZ", 0.2),
                ("CMP+4", -1.0),
            ),
            3.5,
        ),
        (
            "bnz_hi_nonzero",
            byte0_conditions
            + (
                ("OP_BNZ", 0.2),
                ("CMP+4", 1.0),
                ("CMP+5", -1.0),
            ),
            4.5,
        ),
        (
            "jsr_target",
            byte0_conditions + _jsr_opcode_nibble_conditions(),
            4.5,
        ),
    )

    for group, conditions, threshold in groups:
        for band, output_base in (("lo", "OUTPUT_LO"), ("hi", "OUTPUT_HI")):
            for k in range(16):
                rules.append(FFNRule.gated_write(
                    name=f"l6_branch_pc_byte1_{group}_cancel_{band}_{k}",
                    conditions=conditions,
                    threshold=threshold,
                    gate=f"{output_base}+{k}",
                    gate_weight=-1.0,
                    writes=((f"{output_base}+{k}", write_scale),),
                ))
        for k in range(16):
            rules.append(FFNRule.gated_write(
                name=f"l6_branch_pc_byte1_{group}_target_lo_{k}",
                conditions=conditions,
                threshold=threshold,
                gate=f"FETCH_HI+{k}",
                writes=((
                    f"OUTPUT_LO+{_pc_target_byte1_lo_from_imm_hi(k)}",
                    write_scale,
                ),),
            ))
        rules.append(FFNRule.gated_write(
            name=f"l6_branch_pc_byte1_{group}_target_hi_zero",
            conditions=conditions,
            threshold=threshold,
            gate="CONST",
            writes=(("OUTPUT_HI+0", const_zero_scale),),
        ))

    return tuple(rules)


def _layer6_tail_cleanup_rules(S: float) -> tuple[FFNRule, ...]:
    """CompilerIR rules for L6 opcode/MEM/ALU cleanup units 1070..1135."""

    rules = []
    for k in range(16):
        rules.append(FFNRule.gated_write(
            name=f"l6_opcode_lo_cleanup_{k}",
            conditions=(("MARK_AX", 1.0),),
            threshold=0.5,
            gate=f"OPCODE_BYTE_LO+{k}",
            gate_weight=-1.0,
            writes=((f"ADDR_B0_LO+{k}", 2.0 / S),),
        ))
    for k in range(16):
        rules.append(FFNRule.gated_write(
            name=f"l6_opcode_hi_cleanup_{k}",
            conditions=(("MARK_AX", 1.0),),
            threshold=0.5,
            gate=f"OPCODE_BYTE_HI+{k}",
            gate_weight=-1.0,
            writes=((f"ADDR_B1_LO+{k}", 2.0 / S),),
        ))
    for dim_name in ("MEM_STORE", "MEM_ADDR_SRC"):
        rules.append(FFNRule.gated_write(
            name=f"l6_{dim_name.lower()}_leakage_cleanup",
            conditions=(
                ("MARK_SP", 1.0),
                ("MARK_STACK0", 1.0),
                ("MARK_BP", 1.0),
            ),
            threshold=0.5,
            gate=dim_name,
            gate_weight=-1.0,
            writes=((dim_name, 2.0 / S),),
        ))
    for k in range(16):
        rules.append(FFNRule.constant_write(
            name=f"l6_alu_lo_clear_{k}",
            conditions=(("MARK_AX", 1.0),),
            threshold=0.5,
            writes=((f"ALU_LO+{k}", -10.0 / S),),
        ))
    for k in range(16):
        rules.append(FFNRule.constant_write(
            name=f"l6_alu_hi_clear_{k}",
            conditions=(("MARK_AX", 1.0),),
            threshold=0.5,
            writes=((f"ALU_HI+{k}", -10.0 / S),),
        ))
    return tuple(rules)


def _layer6_stack_writeback_rules(
    *,
    name_prefix: str,
    conditions: tuple[tuple[str, float], ...],
    threshold: float,
    S: float,
) -> tuple[FFNRule, ...]:
    rules = []
    write_scale = 2.0 / S
    for band, embed_base, carry_base, output_base in (
        ("lo", "EMBED_LO", "AX_CARRY_LO", "OUTPUT_LO"),
        ("hi", "EMBED_HI", "AX_CARRY_HI", "OUTPUT_HI"),
    ):
        for k in range(16):
            rules.append(FFNRule.gated_write(
                name=f"{name_prefix}_{band}_{k}",
                conditions=conditions,
                threshold=threshold,
                gate_terms=(
                    (f"{embed_base}+{k}", -1.0),
                    (f"{carry_base}+{k}", 1.0),
                ),
                writes=((f"{output_base}+{k}", write_scale),),
            ))
    return tuple(rules)


def _layer6_ax_output_route_rules(
    *,
    name_prefix: str,
    threshold: float,
    conditions: tuple[tuple[str, float], ...],
    S: float,
) -> tuple[FFNRule, ...]:
    rules = []
    write_scale = 2.0 / S
    for band, source_base, output_base in (
        ("lo", "AX_CARRY_LO", "OUTPUT_LO"),
        ("hi", "AX_CARRY_HI", "OUTPUT_HI"),
    ):
        for k in range(16):
            rules.append(FFNRule.gated_write(
                name=f"{name_prefix}_{band}_{k}",
                conditions=conditions,
                threshold=threshold,
                gate=f"{source_base}+{k}",
                writes=((f"{output_base}+{k}", write_scale),),
            ))
    return tuple(rules)


def _lower_layer6_ffn_rules(
    ffn,
    rules: tuple[FFNRule, ...],
    S: float,
    BD,
    *,
    unit: int,
) -> int:
    dim_positions = Primitives.dim_positions_from_bd(
        BD,
        Primitives.ffn_rule_dim_names(rules),
    )
    return Primitives.lower_ffn_rules(
        ffn,
        rules,
        dim_positions,
        start_unit=unit,
        S=S,
    )


def _lower_layer6_imm_fetch_route_ir(
    ffn,
    S: float,
    BD,
    *,
    unit: int = L6_IMM_FETCH_ROUTE_START_UNIT,
) -> int:
    """Lower the IR-authored IMM FETCH route band into L6 FFN weights."""

    return _lower_layer6_ffn_rules(
        ffn,
        _layer6_imm_fetch_route_rules(S),
        S,
        BD,
        unit=unit,
    )


def _lower_layer6_imm_carry_refresh_ir(
    ffn,
    S: float,
    BD,
    *,
    unit: int = L6_IMM_CARRY_REFRESH_START_UNIT,
) -> int:
    """Lower the IR-authored IMM carry refresh band into L6 FFN weights."""

    return _lower_layer6_ffn_rules(
        ffn,
        _layer6_imm_carry_refresh_rules(S),
        S,
        BD,
        unit=unit,
    )


def _lower_layer6_ax_output_route_ir(ffn, S: float, BD) -> tuple[int, int, int, int]:
    """Lower IR-authored EXIT/NOP/JSR/JMP AX-output route bands."""

    exit_end = _lower_layer6_ffn_rules(
        ffn,
        _layer6_exit_ax_route_rules(S),
        S,
        BD,
        unit=L6_EXIT_AX_ROUTE_START_UNIT,
    )
    nop_end = _lower_layer6_ffn_rules(
        ffn,
        _layer6_nop_ax_route_rules(S),
        S,
        BD,
        unit=L6_NOP_AX_ROUTE_START_UNIT,
    )
    jsr_end = _lower_layer6_ffn_rules(
        ffn,
        _layer6_jsr_ax_route_rules(S),
        S,
        BD,
        unit=L6_JSR_AX_ROUTE_START_UNIT,
    )
    jmp_end = _lower_layer6_ffn_rules(
        ffn,
        _layer6_jmp_ax_route_rules(S),
        S,
        BD,
        unit=L6_JMP_AX_ROUTE_START_UNIT,
    )
    return exit_end, nop_end, jsr_end, jmp_end


def _lower_layer6_delayed_jmp_pc_override_ir(
    ffn,
    S: float,
    BD,
    *,
    unit: int = L6_DELAYED_JMP_PC_OVERRIDE_START_UNIT,
) -> int:
    """Lower the IR-authored delayed JMP override band into L6 FFN weights."""

    return _lower_layer6_ffn_rules(
        ffn,
        _layer6_delayed_jmp_pc_override_rules(S),
        S,
        BD,
        unit=unit,
    )


def _lower_layer6_first_step_jmp_pc_override_ir(
    ffn,
    S: float,
    BD,
    *,
    unit: int = L6_FIRST_STEP_JMP_PC_OVERRIDE_START_UNIT,
) -> int:
    """Lower the IR-authored first-step JMP override band into L6 FFN weights."""

    return _lower_layer6_ffn_rules(
        ffn,
        _layer6_first_step_jmp_pc_override_rules(S),
        S,
        BD,
        unit=unit,
    )


def _lower_layer6_halt_detect_ir(
    ffn,
    S: float,
    BD,
    *,
    unit: int = L6_HALT_DETECT_START_UNIT,
) -> int:
    """Lower the IR-authored HALT conversion unit into L6 FFN weights."""

    return _lower_layer6_ffn_rules(
        ffn,
        _layer6_halt_detect_rules(S),
        S,
        BD,
        unit=unit,
    )


def _lower_layer6_temp_cleanup_ir(
    ffn,
    S: float,
    BD,
    *,
    unit: int = L6_TEMP_CLEANUP_RULE_START_UNIT,
) -> int:
    """Lower the IR-authored TEMP cleanup band into L6 FFN weights."""

    return _lower_layer6_ffn_rules(
        ffn,
        _layer6_temp_cleanup_rules(S),
        S,
        BD,
        unit=unit,
    )


def _lower_layer6_cmp3_cleanup_ir(
    ffn,
    S: float,
    BD,
    *,
    unit: int = L6_CMP3_CLEANUP_START_UNIT,
) -> int:
    """Lower the IR-authored CMP[3] cleanup unit into L6 FFN weights."""

    return _lower_layer6_ffn_rules(
        ffn,
        _layer6_cmp3_cleanup_rules(S),
        S,
        BD,
        unit=unit,
    )


def _lower_layer6_stack_identity_ir(
    ffn,
    S: float,
    BD,
    *,
    unit: int = L6_STACK_IDENTITY_START_UNIT,
) -> int:
    """Lower the IR-authored stack marker identity band into L6 FFN weights."""

    return _lower_layer6_ffn_rules(
        ffn,
        _layer6_stack_identity_rules(S),
        S,
        BD,
        unit=unit,
    )


def _lower_layer6_stack_arithmetic_ir(ffn, S: float, BD) -> tuple[int, int, int, int, int]:
    """Lower IR-authored PSH/JSR stack arithmetic bands."""

    psh_sp_end = _lower_layer6_ffn_rules(
        ffn,
        _layer6_psh_sp_decrement_rules(S),
        S,
        BD,
        unit=L6_PSH_SP_DECREMENT_START_UNIT,
    )
    jsr_sp_end = _lower_layer6_ffn_rules(
        ffn,
        _layer6_jsr_sp_decrement_rules(S),
        S,
        BD,
        unit=L6_JSR_SP_DECREMENT_START_UNIT,
    )
    jsr_fixup_end = _lower_layer6_ffn_rules(
        ffn,
        _layer6_jsr_sp_fixup_rules(S),
        S,
        BD,
        unit=L6_JSR_SP_FIXUP_START_UNIT,
    )
    jsr_bytes_end = _lower_layer6_ffn_rules(
        ffn,
        _layer6_jsr_sp_bytes_rules(S),
        S,
        BD,
        unit=L6_JSR_SP_BYTES_START_UNIT,
    )
    psh_stack0_end = _lower_layer6_ffn_rules(
        ffn,
        _layer6_psh_stack0_writeback_rules(S),
        S,
        BD,
        unit=L6_PSH_STACK0_WRITEBACK_START_UNIT,
    )
    return (
        psh_sp_end,
        jsr_sp_end,
        jsr_fixup_end,
        jsr_bytes_end,
        psh_stack0_end,
    )


def _lower_layer6_late_ax_output_route_ir(ffn, S: float, BD) -> tuple[int, int, int, int, int]:
    """Lower IR-authored GETCHAR/BZ/BNZ/PSH/ADJ AX-output route bands."""

    getchar_end = _lower_layer6_ffn_rules(
        ffn,
        _layer6_getchar_ax_route_rules(S),
        S,
        BD,
        unit=L6_GETCHAR_AX_ROUTE_START_UNIT,
    )
    bz_end = _lower_layer6_ffn_rules(
        ffn,
        _layer6_bz_ax_route_rules(S),
        S,
        BD,
        unit=L6_BZ_AX_ROUTE_START_UNIT,
    )
    bnz_end = _lower_layer6_ffn_rules(
        ffn,
        _layer6_bnz_ax_route_rules(S),
        S,
        BD,
        unit=L6_BNZ_AX_ROUTE_START_UNIT,
    )
    psh_end = _lower_layer6_ffn_rules(
        ffn,
        _layer6_psh_ax_route_rules(S),
        S,
        BD,
        unit=L6_PSH_AX_ROUTE_START_UNIT,
    )
    adj_end = _lower_layer6_ffn_rules(
        ffn,
        _layer6_adj_ax_route_rules(S),
        S,
        BD,
        unit=L6_ADJ_AX_ROUTE_START_UNIT,
    )
    return getchar_end, bz_end, bnz_end, psh_end, adj_end


def _lower_layer6_stack_writeback_ir(ffn, S: float, BD) -> tuple[int, int]:
    """Lower IR-authored ADJ/ENT SP writeback bands."""

    adj_end = _lower_layer6_ffn_rules(
        ffn,
        _layer6_adj_sp_writeback_rules(S),
        S,
        BD,
        unit=L6_ADJ_SP_WRITEBACK_START_UNIT,
    )
    ent_end = _lower_layer6_ffn_rules(
        ffn,
        _layer6_ent_sp_writeback_rules(S),
        S,
        BD,
        unit=L6_ENT_SP_WRITEBACK_START_UNIT,
    )
    return adj_end, ent_end


def _lower_layer6_ent_first_step_ir(ffn, S: float, BD) -> tuple[int, int]:
    """Lower IR-authored ENT first-step SP bands."""

    byte0_end = _lower_layer6_ffn_rules(
        ffn,
        _layer6_ent_first_step_sp_byte0_rules(S),
        S,
        BD,
        unit=L6_ENT_FIRST_STEP_SP_BYTE0_START_UNIT,
    )
    bytes_end = _lower_layer6_ffn_rules(
        ffn,
        _layer6_ent_first_step_sp_bytes_rules(S),
        S,
        BD,
        unit=L6_ENT_FIRST_STEP_SP_BYTES_START_UNIT,
    )
    return byte0_end, bytes_end


def _lower_layer6_ent_after_jsr_sp_byte0_fixup_ir(ffn, S: float, BD) -> int:
    """Lower focused ENT-after-JSR SP byte-0 correction."""

    return _lower_layer6_ffn_rules(
        ffn,
        _layer6_ent_after_jsr_sp_byte0_fixup_rules(S),
        S,
        BD,
        unit=L6_ENT_AFTER_JSR_SP_BYTE0_FIXUP_START_UNIT,
    )


def _lower_layer6_branch_pc_override_ir(ffn, S: float, BD) -> tuple[int, int]:
    """Lower IR-authored BZ/BNZ PC override bands."""

    bz_end = _lower_layer6_ffn_rules(
        ffn,
        _layer6_bz_pc_override_rules(S),
        S,
        BD,
        unit=L6_BZ_PC_OVERRIDE_START_UNIT,
    )
    bnz_end = _lower_layer6_ffn_rules(
        ffn,
        _layer6_bnz_pc_override_rules(S),
        S,
        BD,
        unit=L6_BNZ_PC_OVERRIDE_START_UNIT,
    )
    return bz_end, bnz_end


def _lower_layer6_branch_pc_byte1_override_ir(
    ffn,
    S: float,
    BD,
    *,
    unit: int = L6_BRANCH_PC_BYTE1_OVERRIDE_START_UNIT,
) -> int:
    """Lower taken-branch PC byte-1 override rules into L6 FFN weights."""

    return _lower_layer6_ffn_rules(
        ffn,
        _layer6_branch_pc_byte1_override_rules(S),
        S,
        BD,
        unit=unit,
    )


def _lower_layer6_tail_cleanup_ir(ffn, S: float, BD) -> int:
    """Lower IR-authored L6 tail cleanup bands."""

    return _lower_layer6_ffn_rules(
        ffn,
        _layer6_tail_cleanup_rules(S),
        S,
        BD,
        unit=L6_OPCODE_CONTAMINATION_CLEANUP_START_UNIT,
    )


def _lower_layer6_all_step_jmp_pc_override_ir(
    ffn,
    S: float,
    BD,
    *,
    unit: int = L6_ALL_STEP_JMP_PC_OVERRIDE_START_UNIT,
) -> int:
    """Lower the IR-authored all-step JMP override band into L6 FFN weights."""

    return _lower_layer6_ffn_rules(
        ffn,
        _layer6_all_step_jmp_pc_override_rules(S),
        S,
        BD,
        unit=unit,
    )


def _lower_layer6_all_step_jsr_pc_override_ir(
    ffn,
    S: float,
    BD,
    *,
    unit: int = L6_ALL_STEP_JSR_PC_OVERRIDE_START_UNIT,
) -> int:
    """Lower the IR-authored all-step JSR override band into L6 FFN weights."""

    return _lower_layer6_ffn_rules(
        ffn,
        _layer6_all_step_jsr_pc_override_rules(S),
        S,
        BD,
        unit=unit,
    )


def _bake_layer6_routing_ffn(ffn, S: float, BD) -> None:
    """Bake L6 routing FFN via the smoke-stable legacy wrapper.

    The CompilerIR lowerers below have parity coverage for individual unit
    bands, but strict neural smoke still depends on the full legacy L6 bake
    until the generated specs preserve PSH/ADD autoregressive step structure.
    """

    from ...vm_step import _set_layer6_routing_ffn

    _set_layer6_routing_ffn(ffn, S, BD)
    unit = L6_ALL_STEP_JSR_PC_OVERRIDE_END_UNIT + 2

    # Strict neural PSH needs STACK0 byte 0 to be exactly AX. The legacy
    # writeback cancels EMBED and adds ALU, but a tiny residual OUTPUT_HI[1]
    # can beat OUTPUT_HI[0] at the STACK0 marker and emit 0x1a instead of
    # 0x0a. Do a local marker-only OUTPUT rewrite from the relayed ALU value.
    for output_base, alu_base in (
        (BD.OUTPUT_LO, BD.ALU_LO),
        (BD.OUTPUT_HI, BD.ALU_HI),
    ):
        for k in range(16):
            ffn.W_up.data[unit, BD.PSH_AT_SP] = S
            ffn.W_up.data[unit, BD.MARK_STACK0] = S
            ffn.b_up.data[unit] = -S * 1.5
            ffn.W_gate.data[unit, output_base + k] = -1.0
            ffn.W_down.data[output_base + k, unit] = 2.0 / S
            unit += 1
        for k in range(16):
            ffn.W_up.data[unit, BD.PSH_AT_SP] = S
            ffn.W_up.data[unit, BD.MARK_STACK0] = S
            ffn.b_up.data[unit] = -S * 1.5
            ffn.W_gate.data[unit, alu_base + k] = 1.0
            ffn.W_down.data[output_base + k, unit] = 2.0 / S
            unit += 1
        for k in range(16):
            ffn.W_up.data[unit, BD.PSH_AT_SP] = S
            ffn.W_up.data[unit, BD.MARK_STACK0] = S
            ffn.W_up.data[unit, alu_base + k] = S
            ffn.b_up.data[unit] = -S * 2.5
            ffn.b_gate.data[unit] = 1.0
            ffn.W_down.data[output_base + k, unit] = 3.0 / S
            unit += 1
    _clear_ffn_unit_band(
        ffn,
        L6_BRANCH_PC_BYTE1_OVERRIDE_START_UNIT,
        L6_BRANCH_PC_BYTE1_OVERRIDE_END_UNIT,
    )
    branch_byte1_end = _lower_layer6_branch_pc_byte1_override_ir(ffn, S, BD)
    if branch_byte1_end != L6_BRANCH_PC_BYTE1_OVERRIDE_END_UNIT:
        raise AssertionError(
            "L6 branch PC-byte1 override IR lowered to unexpected unit "
            f"{branch_byte1_end}; expected "
            f"{L6_BRANCH_PC_BYTE1_OVERRIDE_END_UNIT}"
        )
    _clear_ffn_unit_band(
        ffn,
        L6_ALL_STEP_JSR_PC_OVERRIDE_START_UNIT,
        L6_ALL_STEP_JSR_PC_OVERRIDE_END_UNIT,
    )
    jsr_pc_end = _lower_layer6_all_step_jsr_pc_override_ir(ffn, S, BD)
    if jsr_pc_end != L6_ALL_STEP_JSR_PC_OVERRIDE_END_UNIT:
        raise AssertionError(
            "L6 all-step JSR PC override IR lowered to unexpected unit "
            f"{jsr_pc_end}; expected {L6_ALL_STEP_JSR_PC_OVERRIDE_END_UNIT}"
        )
    _clear_ffn_unit_band(
        ffn,
        L6_JSR_SP_DECREMENT_START_UNIT,
        L6_JSR_SP_DECREMENT_END_UNIT,
    )
    jsr_sp_end = _lower_layer6_ffn_rules(
        ffn,
        _layer6_jsr_sp_decrement_rules(S),
        S,
        BD,
        unit=L6_JSR_SP_DECREMENT_START_UNIT,
    )
    if jsr_sp_end != L6_JSR_SP_DECREMENT_END_UNIT:
        raise AssertionError(
            "L6 JSR SP decrement IR lowered to unexpected unit "
            f"{jsr_sp_end}; expected {L6_JSR_SP_DECREMENT_END_UNIT}"
        )
    for start, end in (
        (
            L6_DELAYED_JMP_PC_OVERRIDE_START_UNIT,
            L6_DELAYED_JMP_PC_OVERRIDE_END_UNIT,
        ),
        (
            L6_FIRST_STEP_JMP_PC_OVERRIDE_START_UNIT,
            L6_FIRST_STEP_JMP_PC_OVERRIDE_END_UNIT,
        ),
        (
            L6_ALL_STEP_JMP_PC_OVERRIDE_START_UNIT,
            L6_ALL_STEP_JMP_PC_OVERRIDE_END_UNIT,
        ),
        (L6_BZ_PC_OVERRIDE_START_UNIT, L6_BZ_PC_OVERRIDE_END_UNIT),
        (L6_BNZ_PC_OVERRIDE_START_UNIT, L6_BNZ_PC_OVERRIDE_END_UNIT),
    ):
        _clear_ffn_unit_band(ffn, start, end)
    delayed_end = _lower_layer6_delayed_jmp_pc_override_ir(ffn, S, BD)
    if delayed_end != L6_DELAYED_JMP_PC_OVERRIDE_END_UNIT:
        raise AssertionError(
            "L6 delayed JMP PC override IR lowered to unexpected unit "
            f"{delayed_end}; expected {L6_DELAYED_JMP_PC_OVERRIDE_END_UNIT}"
        )
    first_step_end = _lower_layer6_first_step_jmp_pc_override_ir(ffn, S, BD)
    if first_step_end != L6_FIRST_STEP_JMP_PC_OVERRIDE_END_UNIT:
        raise AssertionError(
            "L6 first-step JMP PC override IR lowered to unexpected unit "
            f"{first_step_end}; expected "
            f"{L6_FIRST_STEP_JMP_PC_OVERRIDE_END_UNIT}"
        )
    all_step_jmp_end = _lower_layer6_all_step_jmp_pc_override_ir(ffn, S, BD)
    if all_step_jmp_end != L6_ALL_STEP_JMP_PC_OVERRIDE_END_UNIT:
        raise AssertionError(
            "L6 all-step JMP PC override IR lowered to unexpected unit "
            f"{all_step_jmp_end}; expected {L6_ALL_STEP_JMP_PC_OVERRIDE_END_UNIT}"
        )
    branch_pc_ends = _lower_layer6_branch_pc_override_ir(ffn, S, BD)
    expected_branch_pc_ends = (
        L6_BZ_PC_OVERRIDE_END_UNIT,
        L6_BNZ_PC_OVERRIDE_END_UNIT,
    )
    if branch_pc_ends != expected_branch_pc_ends:
        raise AssertionError(
            "L6 BZ/BNZ PC override IR lowered to unexpected units "
            f"{branch_pc_ends}; expected {expected_branch_pc_ends}"
        )
    return
    _clear_ffn_unit_band(
        ffn,
        L6_IMM_FETCH_ROUTE_START_UNIT,
        L6_IMM_FETCH_ROUTE_END_UNIT,
    )
    imm_end = _lower_layer6_imm_fetch_route_ir(ffn, S, BD)
    if imm_end != L6_IMM_FETCH_ROUTE_END_UNIT:
        raise AssertionError(
            "L6 IMM fetch route IR lowered to unexpected unit "
            f"{imm_end}; expected {L6_IMM_FETCH_ROUTE_END_UNIT}"
        )
    _clear_ffn_unit_band(
        ffn,
        L6_IMM_CARRY_REFRESH_START_UNIT,
        L6_IMM_CARRY_REFRESH_END_UNIT,
    )
    imm_carry_end = _lower_layer6_imm_carry_refresh_ir(ffn, S, BD)
    if imm_carry_end != L6_IMM_CARRY_REFRESH_END_UNIT:
        raise AssertionError(
            "L6 IMM carry refresh IR lowered to unexpected unit "
            f"{imm_carry_end}; expected {L6_IMM_CARRY_REFRESH_END_UNIT}"
        )
    for start, end in (
        (L6_EXIT_AX_ROUTE_START_UNIT, L6_EXIT_AX_ROUTE_END_UNIT),
        (L6_NOP_AX_ROUTE_START_UNIT, L6_NOP_AX_ROUTE_END_UNIT),
        (L6_JSR_AX_ROUTE_START_UNIT, L6_JSR_AX_ROUTE_END_UNIT),
        (L6_JMP_AX_ROUTE_START_UNIT, L6_JMP_AX_ROUTE_END_UNIT),
    ):
        _clear_ffn_unit_band(ffn, start, end)
    route_ends = _lower_layer6_ax_output_route_ir(ffn, S, BD)
    expected_ends = (
        L6_EXIT_AX_ROUTE_END_UNIT,
        L6_NOP_AX_ROUTE_END_UNIT,
        L6_JSR_AX_ROUTE_END_UNIT,
        L6_JMP_AX_ROUTE_END_UNIT,
    )
    if route_ends != expected_ends:
        raise AssertionError(
            "L6 AX-output route IR lowered to unexpected units "
            f"{route_ends}; expected {expected_ends}"
        )
    for start, end in (
        (
            L6_DELAYED_JMP_PC_OVERRIDE_START_UNIT,
            L6_DELAYED_JMP_PC_OVERRIDE_END_UNIT,
        ),
        (
            L6_FIRST_STEP_JMP_PC_OVERRIDE_START_UNIT,
            L6_FIRST_STEP_JMP_PC_OVERRIDE_END_UNIT,
        ),
    ):
        _clear_ffn_unit_band(ffn, start, end)
    delayed_end = _lower_layer6_delayed_jmp_pc_override_ir(ffn, S, BD)
    if delayed_end != L6_DELAYED_JMP_PC_OVERRIDE_END_UNIT:
        raise AssertionError(
            "L6 delayed JMP PC override IR lowered to unexpected unit "
            f"{delayed_end}; expected {L6_DELAYED_JMP_PC_OVERRIDE_END_UNIT}"
        )
    first_step_end = _lower_layer6_first_step_jmp_pc_override_ir(ffn, S, BD)
    if first_step_end != L6_FIRST_STEP_JMP_PC_OVERRIDE_END_UNIT:
        raise AssertionError(
            "L6 first-step JMP PC override IR lowered to unexpected unit "
            f"{first_step_end}; expected "
            f"{L6_FIRST_STEP_JMP_PC_OVERRIDE_END_UNIT}"
        )
    _clear_ffn_unit_band(
        ffn,
        L6_ALL_STEP_JMP_PC_OVERRIDE_START_UNIT,
        L6_ALL_STEP_JMP_PC_OVERRIDE_END_UNIT,
    )
    end = _lower_layer6_all_step_jmp_pc_override_ir(ffn, S, BD)
    if end != L6_ALL_STEP_JMP_PC_OVERRIDE_END_UNIT:
        raise AssertionError(
            "L6 all-step JMP PC override IR lowered to unexpected unit "
            f"{end}; expected {L6_ALL_STEP_JMP_PC_OVERRIDE_END_UNIT}"
        )
    for start, end in (
        (L6_HALT_DETECT_START_UNIT, L6_HALT_DETECT_END_UNIT),
        (L6_TEMP_CLEANUP_START_UNIT, L6_TEMP_CLEANUP_END_UNIT),
        (L6_CMP3_CLEANUP_START_UNIT, L6_CMP3_CLEANUP_END_UNIT),
        (L6_STACK_IDENTITY_START_UNIT, L6_STACK_IDENTITY_END_UNIT),
    ):
        _clear_ffn_unit_band(ffn, start, end)
    halt_end = _lower_layer6_halt_detect_ir(ffn, S, BD)
    if halt_end != L6_HALT_DETECT_END_UNIT:
        raise AssertionError(
            "L6 HALT detect IR lowered to unexpected unit "
            f"{halt_end}; expected {L6_HALT_DETECT_END_UNIT}"
        )
    temp_end = _lower_layer6_temp_cleanup_ir(ffn, S, BD)
    if temp_end != L6_TEMP_CLEANUP_END_UNIT:
        raise AssertionError(
            "L6 TEMP cleanup IR lowered to unexpected unit "
            f"{temp_end}; expected {L6_TEMP_CLEANUP_END_UNIT}"
        )
    cmp3_end = _lower_layer6_cmp3_cleanup_ir(ffn, S, BD)
    if cmp3_end != L6_CMP3_CLEANUP_END_UNIT:
        raise AssertionError(
            "L6 CMP[3] cleanup IR lowered to unexpected unit "
            f"{cmp3_end}; expected {L6_CMP3_CLEANUP_END_UNIT}"
        )
    stack_identity_end = _lower_layer6_stack_identity_ir(ffn, S, BD)
    if stack_identity_end != L6_STACK_IDENTITY_END_UNIT:
        raise AssertionError(
            "L6 stack identity IR lowered to unexpected unit "
            f"{stack_identity_end}; expected {L6_STACK_IDENTITY_END_UNIT}"
        )
    for start, end in (
        (L6_PSH_SP_DECREMENT_START_UNIT, L6_PSH_SP_DECREMENT_END_UNIT),
        (L6_JSR_SP_DECREMENT_START_UNIT, L6_JSR_SP_DECREMENT_END_UNIT),
        (L6_JSR_SP_FIXUP_START_UNIT, L6_JSR_SP_FIXUP_END_UNIT),
        (L6_JSR_SP_BYTES_START_UNIT, L6_JSR_SP_BYTES_END_UNIT),
        (
            L6_PSH_STACK0_WRITEBACK_START_UNIT,
            L6_PSH_STACK0_WRITEBACK_END_UNIT,
        ),
    ):
        _clear_ffn_unit_band(ffn, start, end)
    stack_arithmetic_ends = _lower_layer6_stack_arithmetic_ir(ffn, S, BD)
    expected_stack_arithmetic_ends = (
        L6_PSH_SP_DECREMENT_END_UNIT,
        L6_JSR_SP_DECREMENT_END_UNIT,
        L6_JSR_SP_FIXUP_END_UNIT,
        L6_JSR_SP_BYTES_END_UNIT,
        L6_PSH_STACK0_WRITEBACK_END_UNIT,
    )
    if stack_arithmetic_ends != expected_stack_arithmetic_ends:
        raise AssertionError(
            "L6 stack arithmetic IR lowered to unexpected units "
            f"{stack_arithmetic_ends}; expected "
            f"{expected_stack_arithmetic_ends}"
        )
    for start, end in (
        (L6_GETCHAR_AX_ROUTE_START_UNIT, L6_GETCHAR_AX_ROUTE_END_UNIT),
        (L6_BZ_AX_ROUTE_START_UNIT, L6_BZ_AX_ROUTE_END_UNIT),
        (L6_BNZ_AX_ROUTE_START_UNIT, L6_BNZ_AX_ROUTE_END_UNIT),
        (L6_PSH_AX_ROUTE_START_UNIT, L6_PSH_AX_ROUTE_END_UNIT),
        (L6_ADJ_AX_ROUTE_START_UNIT, L6_ADJ_AX_ROUTE_END_UNIT),
    ):
        _clear_ffn_unit_band(ffn, start, end)
    late_route_ends = _lower_layer6_late_ax_output_route_ir(ffn, S, BD)
    expected_late_route_ends = (
        L6_GETCHAR_AX_ROUTE_END_UNIT,
        L6_BZ_AX_ROUTE_END_UNIT,
        L6_BNZ_AX_ROUTE_END_UNIT,
        L6_PSH_AX_ROUTE_END_UNIT,
        L6_ADJ_AX_ROUTE_END_UNIT,
    )
    if late_route_ends != expected_late_route_ends:
        raise AssertionError(
            "L6 late AX-output route IR lowered to unexpected units "
            f"{late_route_ends}; expected {expected_late_route_ends}"
        )
    for start, end in (
        (L6_ADJ_SP_WRITEBACK_START_UNIT, L6_ADJ_SP_WRITEBACK_END_UNIT),
        (L6_ENT_SP_WRITEBACK_START_UNIT, L6_ENT_SP_WRITEBACK_END_UNIT),
    ):
        _clear_ffn_unit_band(ffn, start, end)
    stack_writeback_ends = _lower_layer6_stack_writeback_ir(ffn, S, BD)
    expected_stack_writeback_ends = (
        L6_ADJ_SP_WRITEBACK_END_UNIT,
        L6_ENT_SP_WRITEBACK_END_UNIT,
    )
    if stack_writeback_ends != expected_stack_writeback_ends:
        raise AssertionError(
            "L6 stack writeback IR lowered to unexpected units "
            f"{stack_writeback_ends}; expected {expected_stack_writeback_ends}"
        )
    for start, end in (
        (
            L6_ENT_FIRST_STEP_SP_BYTE0_START_UNIT,
            L6_ENT_FIRST_STEP_SP_BYTE0_END_UNIT,
        ),
        (
            L6_ENT_FIRST_STEP_SP_BYTES_START_UNIT,
            L6_ENT_FIRST_STEP_SP_BYTES_END_UNIT,
        ),
    ):
        _clear_ffn_unit_band(ffn, start, end)
    ent_first_step_ends = _lower_layer6_ent_first_step_ir(ffn, S, BD)
    expected_ent_first_step_ends = (
        L6_ENT_FIRST_STEP_SP_BYTE0_END_UNIT,
        L6_ENT_FIRST_STEP_SP_BYTES_END_UNIT,
    )
    if ent_first_step_ends != expected_ent_first_step_ends:
        raise AssertionError(
            "L6 ENT first-step IR lowered to unexpected units "
            f"{ent_first_step_ends}; expected {expected_ent_first_step_ends}"
        )
    for start, end in (
        (L6_BZ_PC_OVERRIDE_START_UNIT, L6_BZ_PC_OVERRIDE_END_UNIT),
        (L6_BNZ_PC_OVERRIDE_START_UNIT, L6_BNZ_PC_OVERRIDE_END_UNIT),
    ):
        _clear_ffn_unit_band(ffn, start, end)
    branch_ends = _lower_layer6_branch_pc_override_ir(ffn, S, BD)
    expected_branch_ends = (
        L6_BZ_PC_OVERRIDE_END_UNIT,
        L6_BNZ_PC_OVERRIDE_END_UNIT,
    )
    if branch_ends != expected_branch_ends:
        raise AssertionError(
            "L6 branch override IR lowered to unexpected units "
            f"{branch_ends}; expected {expected_branch_ends}"
        )
    _clear_ffn_unit_band(
        ffn,
        L6_OPCODE_CONTAMINATION_CLEANUP_START_UNIT,
        L6_ALU_CLEAR_END_UNIT,
    )
    tail_cleanup_end = _lower_layer6_tail_cleanup_ir(ffn, S, BD)
    if tail_cleanup_end != L6_ALU_CLEAR_END_UNIT:
        raise AssertionError(
            "L6 tail cleanup IR lowered to unexpected unit "
            f"{tail_cleanup_end}; expected {L6_ALU_CLEAR_END_UNIT}"
        )
    _clear_ffn_unit_band(
        ffn,
        L6_BRANCH_PC_BYTE1_OVERRIDE_START_UNIT,
        L6_BRANCH_PC_BYTE1_OVERRIDE_END_UNIT,
    )
    branch_byte1_end = _lower_layer6_branch_pc_byte1_override_ir(ffn, S, BD)
    if branch_byte1_end != L6_BRANCH_PC_BYTE1_OVERRIDE_END_UNIT:
        raise AssertionError(
            "L6 branch PC-byte1 override IR lowered to unexpected unit "
            f"{branch_byte1_end}; expected "
            f"{L6_BRANCH_PC_BYTE1_OVERRIDE_END_UNIT}"
        )


def make_layer6_attn_op() -> Operation:
    """L6 attention: relay heads for IS_JMP, IS_EXIT, etc. at PC marker.

    Kept as a migrated ``kind="attn"`` dep anchor: its declared reads/writes
    preserve the LayerCompiler dep-graph topology that places downstream ops
    (e.g. ALU layers L8-L13) at the right model.blocks indices. The actual
    weight bake happens in ``make_layer6_attn_bake_op`` (kind="model",
    phase=998.5, migrated=True).
    """
    def bake(attn, dim_positions, S):
        # No-op: actual bake is in `layer6_attn_bake` model op below.
        return

    return Operation(
        name="layer6_attn",
        phase=6,
        reads={"OP_JMP", "OP_EXIT", "OP_JSR", "MARK_AX", "MARK_PC", "MARK_SP",
               "MARK_STACK0", "NEXT_SE", "FETCH_LO", "FETCH_HI",
               "PSH_AT_SP", "OP_PSH", "OP_ADJ", "OP_ENT", "OP_LEV",
               "AX_CARRY_LO", "AX_CARRY_HI"},
        writes={"CMP", "AX_CARRY_LO", "AX_CARRY_HI"},
        kind="attn",
        layer_idx=6,
        bake_fn=bake,
        migrated=True,
        declarative_authority="topology_anchor",
        smoke_tests={"all"},
        spec_section="BLOG_SPEC.md#registers",
    )


def make_layer6_routing_ffn_op() -> Operation:
    """L6 FFN: per-opcode routing — write FETCH/AX_CARRY → OUTPUT, etc.

    Originally an inline call in `set_vm_weights`:
        `_set_layer6_routing_ffn(ffn6, S, BD)`

    Pinned to ``layer_idx=6`` via ``kind="block"`` so the bake hits the same
    transformer block (block[6].ffn) the legacy path used (mirrors the L4 FFN
    regression fix: dep-graph assignment alone can place a kind="ffn" op on a
    different block, leaving block[6] unbaked). ``migrated=True`` skips the
    legacy_bake path for this op; the inline call in ``set_vm_weights`` has
    been removed to avoid double-bake.

    Phase 6.5: runs AFTER the L6 attention ops (phases 6.0-6.2) and before
    other L6 FFN extension ops migrated to model-level phase=998.

    Claim-coverage note: ``claims`` is intentionally left empty. This op
    delegates almost the entirety of L6 routing to ``_set_layer6_routing_ffn``
    in ``vm_step.py``, plus a chain of IR lowerers that program
    ~1486 FFN hidden units (every opcode's AX_CARRY/FETCH -> OUTPUT relay,
    PSH stack writeback, every branch's PC byte0/byte1 override band, the
    full delayed/first-step/all-step JMP/JSR PC override families, BZ/BNZ
    overrides, JSR SP decrement, and a strict-neural PSH STACK0 rewrite
    block). The static verifier observes ~13.6k weight cells written by a
    single dispatch -- enumerating them at the (unit, column) grain would
    duplicate every line in ``_set_layer6_routing_ffn`` plus the rule
    bodies in ``_layer6_*_rules`` helpers, with no semantic gain over the
    code those helpers are. Until the underlying spec is decomposed into
    smaller per-band ops (which would be claimed individually), declaring
    claims here would be a maintenance-cost pure-noise duplicate; skip with
    the same precedent as model_ops' ``head_bake`` and ``opcode_relay_head``.
    """
    def bake(block, dim_positions, S):
        _bake_layer6_routing_ffn(
            block.ffn,
            S,
            _as_setdim_proxy(dim_positions),
        )

    return Operation(
        name="layer6_routing_ffn",
        phase=6.5,
        reads={"OP_IMM", "OP_EXIT", "OP_JMP", "OP_NOP", "OP_LEA",
               "MARK_AX", "MARK_PC", "MARK_STACK0", "MARK_BP",
               "IS_BYTE", "FETCH_LO", "FETCH_HI",
               "AX_CARRY_LO", "AX_CARRY_HI", "CMP",
               "OUTPUT_LO", "OUTPUT_HI", "HAS_SE",
               "OPCODE_BASE", "OUTPUT_BYTE_LO", "OUTPUT_BYTE_HI",
               "TEMP", "DIV_STAGING"},
        writes={"OUTPUT_LO", "OUTPUT_HI", "AX_CARRY_LO", "AX_CARRY_HI"},
        kind="block",
        bake_fn=bake,
        declarative_bake_fn=bake,
        declarative_authority="spec_generated",
        layer_idx=6,
        migrated=True,
        smoke_tests={
            "TestSmokeBasic::test_imm_exit",
            "TestSmokeControlFlow::test_jmp_forward",
            "TestSmokeFunctionCall::test_simple_function",
            "all",
        },
        spec_section="BLOG_SPEC.md#function-calls",
        produces={
            "OUTPUT_LO": "AX_byte0",
            "OUTPUT_HI": "AX_byte0",
        },
        opcodes={"OP_IMM", "OP_EXIT", "OP_NOP", "OP_JMP", "OP_JSR"},
    )


def make_layer6_ent_after_jsr_sp_byte0_fixup_op() -> Operation:
    """L6 FFN: correct ENT's SP byte 0 after the preceding JSR stack push."""

    def bake(block, dim_positions, S):
        end = _lower_layer6_ent_after_jsr_sp_byte0_fixup_ir(
            block.ffn,
            S,
            _as_setdim_proxy(dim_positions),
        )
        if end != L6_ENT_AFTER_JSR_SP_BYTE0_FIXUP_END_UNIT:
            raise AssertionError(
                "L6 ENT-after-JSR SP byte0 fixup lowered to unexpected unit "
                f"{end}; expected {L6_ENT_AFTER_JSR_SP_BYTE0_FIXUP_END_UNIT}"
            )

    # Dim-ownership claims. ``_layer6_ent_after_jsr_sp_byte0_fixup_rules``
    # lowers seven FFNRules into units 1668..1674 (one per rule). Each rule
    # writes its conditions into ``W_up[unit, dim]`` (per-rule discrimination
    # gating ENT after a JSR) and its constant writes into
    # ``W_down[output_dim, unit]``. Claims enumerate the exact (unit, column)
    # pairs the IR lowerer programs; bias columns are intentionally absent
    # from the verifier's claim grid.
    _claims = frozenset({
        # Unit 1668: l6_ent_after_jsr_sp_byte0_e8 (SP byte0 0xe8)
        (6, "ffn_W_up", "1668", "OP_ENT+0"),
        (6, "ffn_W_up", "1668", "MARK_SP+0"),
        (6, "ffn_W_up", "1668", "HAS_SE+0"),
        (6, "ffn_W_up", "1668", "EMBED_LO+8"),
        (6, "ffn_W_up", "1668", "EMBED_HI+15"),
        (6, "ffn_W_down", "1668", "OUTPUT_LO+8"),
        (6, "ffn_W_down", "1668", "OUTPUT_HI+14"),
        (6, "ffn_W_down", "1668", "OUTPUT_LO+0"),
        (6, "ffn_W_down", "1668", "OUTPUT_LO+10"),
        (6, "ffn_W_down", "1668", "OUTPUT_LO+14"),
        (6, "ffn_W_down", "1668", "OUTPUT_HI+1"),
        (6, "ffn_W_down", "1668", "OUTPUT_HI+15"),
        # Unit 1669: l6_ent_after_jsr_sp_byte0_f0_when_ent_zero
        (6, "ffn_W_up", "1669", "OP_ENT+0"),
        (6, "ffn_W_up", "1669", "MARK_SP+0"),
        (6, "ffn_W_up", "1669", "HAS_SE+0"),
        (6, "ffn_W_up", "1669", "EMBED_LO+0"),
        (6, "ffn_W_up", "1669", "EMBED_HI+15"),
        (6, "ffn_W_up", "1669", "EMBED_LO+8"),
        (6, "ffn_W_down", "1669", "OUTPUT_LO+0"),
        (6, "ffn_W_down", "1669", "OUTPUT_HI+15"),
        (6, "ffn_W_down", "1669", "OUTPUT_LO+8"),
        (6, "ffn_W_down", "1669", "OUTPUT_LO+10"),
        (6, "ffn_W_down", "1669", "OUTPUT_HI+0"),
        (6, "ffn_W_down", "1669", "OUTPUT_HI+14"),
        # Unit 1670: l6_ent_after_jsr_bp_byte0_f0
        (6, "ffn_W_up", "1670", "OP_ENT+0"),
        (6, "ffn_W_up", "1670", "MARK_BP+0"),
        (6, "ffn_W_up", "1670", "HAS_SE+0"),
        (6, "ffn_W_down", "1670", "OUTPUT_LO+0"),
        (6, "ffn_W_down", "1670", "OUTPUT_HI+15"),
        (6, "ffn_W_down", "1670", "OUTPUT_LO+8"),
        (6, "ffn_W_down", "1670", "OUTPUT_HI+1"),
        # Unit 1671: l6_ent_after_jsr_bp_byte1_ff
        (6, "ffn_W_up", "1671", "OP_ENT+0"),
        (6, "ffn_W_up", "1671", "IS_BYTE+0"),
        (6, "ffn_W_up", "1671", "H1+3"),
        (6, "ffn_W_up", "1671", "BYTE_INDEX_0+0"),
        (6, "ffn_W_up", "1671", "HAS_SE+0"),
        (6, "ffn_W_down", "1671", "OUTPUT_LO+15"),
        (6, "ffn_W_down", "1671", "OUTPUT_HI+15"),
        (6, "ffn_W_down", "1671", "OUTPUT_LO+0"),
        (6, "ffn_W_down", "1671", "OUTPUT_HI+0"),
        # Unit 1672: l6_ent_after_jsr_bp_byte2_00
        (6, "ffn_W_up", "1672", "OP_ENT+0"),
        (6, "ffn_W_up", "1672", "IS_BYTE+0"),
        (6, "ffn_W_up", "1672", "H1+3"),
        (6, "ffn_W_up", "1672", "BYTE_INDEX_1+0"),
        (6, "ffn_W_up", "1672", "HAS_SE+0"),
        (6, "ffn_W_down", "1672", "OUTPUT_LO+0"),
        (6, "ffn_W_down", "1672", "OUTPUT_HI+0"),
        (6, "ffn_W_down", "1672", "OUTPUT_LO+1"),
        (6, "ffn_W_down", "1672", "OUTPUT_LO+15"),
        (6, "ffn_W_down", "1672", "OUTPUT_HI+15"),
        # Unit 1673: l6_ent_after_jsr_bp_byte3_00
        (6, "ffn_W_up", "1673", "OP_ENT+0"),
        (6, "ffn_W_up", "1673", "IS_BYTE+0"),
        (6, "ffn_W_up", "1673", "H1+3"),
        (6, "ffn_W_up", "1673", "BYTE_INDEX_2+0"),
        (6, "ffn_W_up", "1673", "HAS_SE+0"),
        (6, "ffn_W_down", "1673", "OUTPUT_LO+0"),
        (6, "ffn_W_down", "1673", "OUTPUT_HI+0"),
        (6, "ffn_W_down", "1673", "OUTPUT_LO+1"),
        (6, "ffn_W_down", "1673", "OUTPUT_LO+15"),
        (6, "ffn_W_down", "1673", "OUTPUT_HI+15"),
        # Unit 1674: l6_ent_after_jsr_stack0_byte0_00
        (6, "ffn_W_up", "1674", "OP_ENT+0"),
        (6, "ffn_W_up", "1674", "MARK_STACK0+0"),
        (6, "ffn_W_up", "1674", "HAS_SE+0"),
        (6, "ffn_W_down", "1674", "OUTPUT_LO+0"),
        (6, "ffn_W_down", "1674", "OUTPUT_HI+0"),
        (6, "ffn_W_down", "1674", "OUTPUT_LO+2"),
        (6, "ffn_W_down", "1674", "OUTPUT_LO+12"),
        (6, "ffn_W_down", "1674", "OUTPUT_HI+1"),
        (6, "ffn_W_down", "1674", "OUTPUT_HI+2"),
    })

    return Operation(
        name="layer6_ent_after_jsr_sp_byte0_fixup",
        phase=6.55,
        reads={"OP_ENT", "MARK_SP", "HAS_SE", "EMBED_LO", "EMBED_HI"},
        writes={"OUTPUT_LO", "OUTPUT_HI"},
        kind="block",
        bake_fn=bake,
        declarative_bake_fn=bake,
        declarative_authority="spec_generated",
        layer_idx=6,
        ffn_units_used=L6_ENT_AFTER_JSR_SP_BYTE0_FIXUP_END_UNIT,
        migrated=True,
        claims=_claims,
        smoke_tests={"TestSmokeFunctionCall::test_simple_function"},
        spec_section="BLOG_SPEC.md#function-calls",
    )


def make_layer6_relay_heads_op() -> Operation:
    """L6 head 6/7: STACK0 ← AX relay for PSH.

    Kept as a migrated ``kind="attn"`` dep anchor: its declared reads/writes
    preserve the LayerCompiler dep-graph topology. The actual weight bake now
    happens in ``make_layer6_relay_heads_bake_op`` (kind="model", phase=998.6,
    migrated=True).
    """
    def bake(attn, dim_positions, S):
        # No-op: actual bake is in `layer6_relay_heads_bake` model op below.
        return

    return Operation(
        name="layer6_relay_heads",
        phase=6.1,
        # Head 7 LEV AX_CARRY refresh (pairs with L16's 3650e01) additionally
        # reads STACK0_BYTE0 / CLEAN_EMBED_LO / CLEAN_EMBED_HI / OP_LEV at the
        # K side and writes AX_CARRY_LO / AX_CARRY_HI at the MARK_AX query
        # position; declare those so the LayerCompiler dep graph routes the
        # producer before downstream consumers.
        reads={"MARK_STACK0", "MARK_AX", "AX_CARRY_LO", "AX_CARRY_HI",
               "STACK0_BYTE0", "CLEAN_EMBED_LO", "CLEAN_EMBED_HI",
               "OP_LEV", "CONST"},
        writes={"ALU_LO", "ALU_HI", "AX_CARRY_LO", "AX_CARRY_HI"},
        kind="attn",
        layer_idx=6,
        bake_fn=bake,
        migrated=True,
        declarative_authority="topology_anchor",
        smoke_tests={"all"},
        spec_section="BLOG_SPEC.md#registers",
    )


def _bake_layer6_attn_spec(attn, BD, HD):
    """Spec writer for L6 heads 0-5.

    This is the declarative replacement for the old `_set_layer6_attn`
    wrapper: each section owns one head and writes only the rows described by
    the L6 relay comments below.
    """
    L = 50.0

    # Head 0: later-step JMP relay, PC marker reads previous AX marker.
    base = 0 * HD
    attn.W_q[base, BD.MARK_PC] = L
    attn.W_q[base, BD.MARK_AX] = -L
    attn.W_q[base, BD.HAS_SE] = L * 20
    attn.W_q[base, BD.CONST] = -L * 20
    attn.W_k[base, BD.MARK_AX] = L
    attn.W_k[base, BD.CONST] = 1.0
    attn.W_v[base + 1, BD.OP_JMP] = 1.0
    for k in range(16):
        attn.W_v[base + 2 + k, BD.FETCH_LO + k] = 1.0
        attn.W_v[base + 18 + k, BD.FETCH_HI + k] = 1.0
        attn.W_o[BD.AX_CARRY_LO + k, base + 2 + k] = 1.0
        attn.W_o[BD.AX_CARRY_HI + k, base + 18 + k] = 1.0
    attn.W_o[BD.CMP + 0, base + 1] = 1.0

    # Head 1: EXIT relay, NEXT_SE reads current AX marker.
    base = 1 * HD
    attn.W_q[base, BD.NEXT_SE] = L
    attn.W_q[base, BD.MARK_AX] = -L
    attn.W_k[base, BD.MARK_AX] = L
    attn.W_v[base + 1, BD.OP_EXIT] = 0.2
    attn.W_o[BD.CMP + 1, base + 1] = 1.0

    # Head 2: first-step JMP relay, PC marker self-attends to fetched target.
    base = 2 * HD
    attn.W_q[base, BD.MARK_PC] = L
    attn.W_q[base, BD.HAS_SE] = -L
    attn.W_q[base, BD.MARK_AX] = -L
    attn.W_q[base, BD.OP_JMP] = L * 20
    attn.W_q[base, BD.CONST] = -L * 20
    attn.W_k[base, BD.MARK_PC] = L
    attn.W_v[base + 1, BD.OP_JMP] = 1.0
    for k in range(16):
        attn.W_v[base + 2 + k, BD.FETCH_LO + k] = 1.0
        attn.W_v[base + 18 + k, BD.FETCH_HI + k] = 1.0
        attn.W_o[BD.AX_CARRY_LO + k, base + 2 + k] = 1.0
        attn.W_o[BD.AX_CARRY_HI + k, base + 18 + k] = 1.0
    attn.W_o[BD.CMP + 0, base + 1] = 1.0

    # Head 3: first-step JSR relay, AX marker to PC marker.
    base = 3 * HD
    attn.W_q[base, BD.MARK_PC] = L
    attn.W_q[base, BD.MARK_AX] = -L
    attn.W_q[base, BD.HAS_SE] = -L
    attn.W_k[base, BD.MARK_AX] = L
    attn.W_v[base + 1, BD.OP_JSR] = 1.0
    attn.W_o[BD.TEMP + 0, base + 1] = 1.0

    # Head 4 is reserved for layer6_bz_bnz_relay_bake.

    # Head 5: first-step FETCH relay, PC marker to AX marker.
    base = 5 * HD
    attn.W_q[base, BD.MARK_AX] = L
    attn.W_q[base, BD.HAS_SE] = -L
    attn.W_k[base, BD.MARK_PC] = L
    for k in range(16):
        attn.W_v[base + k, BD.FETCH_LO + k] = 1.0
        attn.W_v[base + 16 + k, BD.FETCH_HI + k] = 1.0
        attn.W_o[BD.FETCH_LO + k, base + k] = 1.0
        attn.W_o[BD.FETCH_HI + k, base + 16 + k] = 1.0
    # Branch byte-1 relay: at the PC byte-0 row, re-read the PC marker so
    # the L6 FFN can emit byte1(target) for BZ/BNZ/JSR targets above 255.
    attn.W_v[base + 32, BD.OP_BZ] = 1.0
    attn.W_v[base + 33, BD.OP_BNZ] = 1.0
    attn.W_v[base + 34, BD.OP_JSR] = 1.0
    attn.W_o[BD.OP_BZ, base + 32] = 1.0
    attn.W_o[BD.OP_BNZ, base + 33] = 1.0
    attn.W_o[BD.OP_JSR, base + 34] = 1.0
    for k in range(16):
        attn.W_v[base + 35 + k, BD.OPCODE_BYTE_LO + k] = 1.0
        attn.W_v[base + 51 + k, BD.OPCODE_BYTE_HI + k] = 1.0
        attn.W_o[BD.OPCODE_BYTE_LO + k, base + 35 + k] = 1.0
        attn.W_o[BD.OPCODE_BYTE_HI + k, base + 51 + k] = 1.0
    branch_pc_byte0_relay = 52
    attn.W_q[base + branch_pc_byte0_relay, BD.IS_BYTE] = 300.0
    attn.W_q[base + branch_pc_byte0_relay, BD.H1 + 0] = 300.0
    attn.W_q[base + branch_pc_byte0_relay, BD.BYTE_INDEX_0] = 300.0
    attn.W_q[base + branch_pc_byte0_relay, BD.MARK_PC] = -300.0
    attn.W_k[base + branch_pc_byte0_relay, BD.MARK_PC] = 50.0
    fetch_gate = 50
    attn.W_q[base + fetch_gate, BD.MARK_AX] = 500.0
    attn.W_q[base + fetch_gate, BD.CONST] = -500.0
    attn.W_k[base + fetch_gate, BD.CONST] = 5.0
    # AX byte rows also carry MARK_AX, but they need L4's PC+2/3/4 FETCH
    # address, not the first immediate byte relayed from the PC marker.
    ax_byte_fetch_blocker = 53
    attn.W_q[base + ax_byte_fetch_blocker, BD.H1 + 1] = -6500.0
    attn.W_q[base + ax_byte_fetch_blocker, BD.IS_BYTE] = -6500.0
    attn.W_q[base + ax_byte_fetch_blocker, BD.MARK_AX] = 6500.0
    attn.W_q[base + ax_byte_fetch_blocker, BD.H1 + 0] = 6500.0
    attn.W_k[base + ax_byte_fetch_blocker, BD.CONST] = 5.0
    has_se_gate = 49
    attn.W_q[base + has_se_gate, BD.HAS_SE] = -500.0
    attn.W_k[base + has_se_gate, BD.CONST] = 5.0
    # After a JSR into a function prologue, the ENT immediate has been
    # fetched at the AX marker.  Relay it forward to the SP marker so later
    # declarative SP-write rules can handle frame sizes beyond one local slot.
    ent_sp_fetch_gate = 51
    attn.W_q[base + ent_sp_fetch_gate, BD.MARK_SP] = 500.0
    attn.W_q[base + ent_sp_fetch_gate, BD.HAS_SE] = 500.0
    attn.W_q[base + ent_sp_fetch_gate, BD.CONST] = -500.0
    attn.W_k[base + ent_sp_fetch_gate, BD.MARK_AX] = 5.0
    attn.W_k[base + ent_sp_fetch_gate, BD.OP_ENT] = 5.0


def _bake_layer6_relay_heads_spec(attn, BD, HD):
    """Spec writer for L6 PSH relay heads 6-7."""
    L = 50.0
    SP_I = 2
    BP_I = 3

    # Head 6: STACK0 reads AX_CARRY_LO from AX into ALU_LO.
    base = 6 * HD
    # Mirror the opcode relay's marker coverage here so the PSH/JSR/ENT flags
    # are available to the same L6 FFN step even if later model-level relay
    # post-passes are trimmed or reordered.
    attn.W_q[base, BD.MARK_SP] = L
    attn.W_q[base, BD.H1 + SP_I] = L
    attn.W_q[base, BD.MARK_STACK0] = L
    attn.W_q[base, BD.L1H4 + BP_I] = L
    attn.W_q[base, BD.MARK_BP] = L
    attn.W_q[base, BD.MARK_PC] = L
    attn.W_q[base, BD.MARK_MEM] = L
    attn.W_q[base, BD.MARK_AX] = -L
    attn.W_k[base, BD.MARK_AX] = L
    attn.W_v[base + 0, BD.OP_LEV] = 0.1
    attn.W_v[base + 1, BD.OP_PSH] = 0.2
    attn.W_v[base + 2, BD.OP_ADJ] = 0.2
    for op_dim in (
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
    ):
        attn.W_v[base + 3, op_dim] = 0.04
    attn.W_v[base + 4, BD.OP_ENT] = 0.2
    attn.W_v[base + 5, BD.OP_JSR] = 0.2
    attn.W_v[base + 6, BD.OP_SI] = 0.2
    attn.W_v[base + 6, BD.OP_SC] = 0.2
    attn.W_v[base + 6, BD.OP_PSH] = 0.2
    attn.W_v[base + 6, BD.OP_JSR] = 0.2
    attn.W_v[base + 6, BD.OP_ENT] = 0.2
    attn.W_v[base + 7, BD.OP_SI] = 0.2
    attn.W_v[base + 7, BD.OP_SC] = 0.2
    attn.W_o[BD.CMP + 0, base + 1] = 1.0
    attn.W_o[BD.PSH_AT_SP, base + 1] = 1.0
    attn.W_o[BD.CMP + 1, base + 2] = 1.0
    attn.W_o[BD.CMP + 3, base + 3] = 5.0
    attn.W_o[BD.CMP + 2, base + 4] = 1.0
    attn.W_o[BD.CMP + 4, base + 5] = 1.0
    attn.W_o[BD.OP_JSR, base + 5] = 5.0
    attn.W_o[BD.OP_ENT, base + 4] = 5.0
    attn.W_o[BD.MEM_STORE, base + 6] = 1.0
    attn.W_o[BD.MEM_ADDR_SRC, base + 7] = 1.0
    attn.W_o[BD.OP_LEV, base + 0] = 10.0
    for k in range(16):
        attn.W_v[base + 8 + k, BD.AX_CARRY_LO + k] = 1.0
        attn.W_o[BD.ALU_LO + k, base + 8 + k] = 1.0

    # Head 7: STACK0 reads AX_CARRY_HI from AX into ALU_HI.
    base = 7 * HD
    attn.W_q[base, BD.MARK_STACK0] = L + L * 20
    attn.W_q[base, BD.MARK_AX] = -L
    attn.W_q[base, BD.CONST] = -L * 20
    attn.W_k[base, BD.MARK_AX] = L
    for k in range(16):
        attn.W_v[base + 33 + k, BD.AX_CARRY_HI + k] = 1.0
        attn.W_o[BD.ALU_HI + k, base + 33 + k] = 1.0

    # ---- LEV AX_CARRY refresh: post-LEV MARK_AX <- STACK0_byte0 CLEAN_EMBED ----
    #
    # Post-LEV (step 6 of a typical function-call sequence) the L8 ALU contract
    # requires AX_CARRY_LO/HI at MARK_AX to hold the popped return value, which
    # lives on the freed STACK0 saved-AX slot. No earlier layer populates it:
    # L3 head 1 (legacy carry-forward) copies the *previous* AX byte 0 EMBED
    # which during LEV is the callee's local AX, not the value just popped
    # from STACK0. The 2026-06-01 stack/JSR/LEV triage attributes 91 rows of
    # ``step6:AX_byte0`` corruption to this missing producer (see
    # ``.agent-logs/stack_jsr_lev_triage_2026_06_01.md``).
    #
    # The companion L16 op ``layer16_lev_routing`` (commit 3650e01) added
    # ``l16_lev_ax_carry_lo/hi_{k}`` FFN rules that gate on ``AX_CARRY_LO/HI+k``
    # and write OUTPUT_LO/HI at MARK_AX during OP_LEV -- but those gates are
    # silent unless something populates AX_CARRY_LO/HI first. This sub-pattern
    # within head 7 provides that producer.
    #
    # Wiring (mirrors L7 head 0 ``_layer7_operand_gather_head_specs`` which
    # already gathers STACK0_byte0 CLEAN_EMBED -> ALU at MARK_AX for binary
    # ops): an attention sub-pattern within head 7 at unused slots that fires
    # on (MARK_AX AND OP_LEV) at the Q side and matches the STACK0_BYTE0 flag
    # (set by the L1 FFN, true at the byte-0 row of the saved-AX stack slot)
    # at the K side, then routes the attended row's CLEAN_EMBED_LO/HI nibbles
    # into AX_CARRY_LO/HI at the Q (MARK_AX) position.
    #
    # Slot layout (head 7, head_dim 64; existing claims use slot 0 and
    # 33..48):
    #   slot 1               : LEV main gate (Q[MARK_AX]+Q[OP_LEV]-Q[CONST];
    #                          K[STACK0_BYTE0])
    #   slot 2 + k (k=0..15) : V[CLEAN_EMBED_LO+k] -> O[AX_CARRY_LO+k]
    #   slot 18              : V[CLEAN_EMBED_HI+0] -> O[AX_CARRY_HI+0]
    #   slot 49 + k (k=0..14): V[CLEAN_EMBED_HI+(1+k)] -> O[AX_CARRY_HI+(1+k)]
    #
    # The Q at slot 1 uses a sign pattern designed so the slot contribution to
    # the Q*K product is positive only at the MARK_AX query position with
    # OP_LEV simultaneously set (Q[1] = +L only when both bits are on). Head 7's
    # existing slot-0 attention (MARK_STACK0 -> MARK_AX) is undisturbed because
    # slot 1's K only matches STACK0_BYTE0 (not MARK_AX), so slot 1's
    # contribution to the existing routing is zero. At MARK_AX during LEV the
    # combined score for j=STACK0_BYTE0 wins by a wide margin over self-match
    # (slot 0 is deeply negative at the MARK_AX query) and over arbitrary
    # other positions (slot 1 only fires at K=STACK0_BYTE0).
    L_lev = L
    # Slot 1 Q/K gate: positive contribution only at (MARK_AX + OP_LEV).
    LEV_GATE = 1
    attn.W_q[base + LEV_GATE, BD.MARK_AX] = L_lev
    attn.W_q[base + LEV_GATE, BD.OP_LEV] = L_lev
    attn.W_q[base + LEV_GATE, BD.CONST] = -L_lev
    attn.W_k[base + LEV_GATE, BD.STACK0_BYTE0] = L_lev
    # V/O lanes for AX_CARRY_LO band: slots 2..17.
    for k in range(16):
        attn.W_v[base + 2 + k, BD.CLEAN_EMBED_LO + k] = 1.0
        attn.W_o[BD.AX_CARRY_LO + k, base + 2 + k] = 1.0
    # V/O lanes for AX_CARRY_HI band: slot 18 for k=0, slots 49..63 for k=1..15
    # (slots 33..48 are claimed by the existing AX_CARRY_HI -> ALU_HI relay
    # above, so we route the HI band's k=0 to the lowest unused slot 18 and
    # the rest to the high tail of the head).
    attn.W_v[base + 18, BD.CLEAN_EMBED_HI + 0] = 1.0
    attn.W_o[BD.AX_CARRY_HI + 0, base + 18] = 1.0
    for k in range(1, 16):
        attn.W_v[base + 48 + k, BD.CLEAN_EMBED_HI + k] = 1.0
        attn.W_o[BD.AX_CARRY_HI + k, base + 48 + k] = 1.0


def make_layer6_attn_bake_op() -> Operation:
    """Bake the L6 attention head spec into ``model.blocks[6].attn``.

    Originally an inline call in ``set_vm_weights``:
        ``_set_layer6_attn(attn6, S, BD, HD)``

    Migrated as ``kind="model"`` with ``migrated=True`` and now owned by the
    local spec writer above, so this op no longer calls the legacy helper.
    Phase=998.5 so this op runs AFTER ``function_call_weights`` (998) but
    BEFORE ``legacy_bake`` (999), matching the legacy in-set_vm_weights
    ordering where _set_layer6_attn ran after function_call_weights.

    A model-level op (rather than ``kind="block"`` pinned to ``layer_idx=6``)
    is required because compile_full_vm dispatches block ops BEFORE all
    model ops, which would flip the override contract between this op's
    sibling ``layer6_relay_heads_bake`` (head 7 Q[MARK_STACK0]=50) and
    ``function_call_weights`` (head 7 Q[MARK_STACK0]=1050). Operating on
    ``model.blocks[6].attn`` directly sidesteps the dep-graph
    layer-assignment risk that motivates the L4 FFN regression's pinning.
    The dep anchor ``layer6_attn`` (kind="attn") above preserves the
    LayerCompiler topology so downstream ops (L8-L13 ALU) remain placed
    at their legacy block indices.
    """
    def bake(model, dim_positions, S):
        del S
        attn = model.blocks[6].attn
        HD = attn.W_q.shape[0] // attn.num_heads
        _bake_layer6_attn_spec(attn, _as_setdim_proxy(dim_positions), HD)
        # Keep the first-step FETCH relay sharp enough to select the PC marker
        # under strict neural smoke; the base spec stays byte-identical to the
        # legacy helper and this bake owns the production-only scale bump.
        attn.W_k.data[5 * HD] *= 10.0

    # Dim-ownership claims. ``_bake_layer6_attn_spec`` writes heads 0, 1, 2,
    # 3, and 5 (head 4 is owned by ``layer6_bz_bnz_relay_bake``, heads 6/7 by
    # ``layer6_relay_heads_bake``). Each head writes a fixed pattern of Q/K
    # gates at slot 0, plus V/O 16-wide low/high nibble lanes that relay the
    # fetched immediate (heads 0, 2, 5) or single-bit opcode flags (heads 1,
    # 3). Head 5 additionally programs the branch byte-1 / opcode-byte relay
    # at slots 32-66 plus a handful of fetch-blocker discriminators at slots
    # 49-53. See ``_bake_layer6_attn_spec`` for the structure.
    _claims = set()
    # --- Head 0: later-step JMP relay (PC reads previous AX) ---
    for col in ("MARK_PC+0", "MARK_AX+0", "HAS_SE+0", "CONST+0"):
        _claims.add((6, "attn_W_q", "0_0", col))
    for col in ("MARK_AX+0", "CONST+0"):
        _claims.add((6, "attn_W_k", "0_0", col))
    _claims.add((6, "attn_W_v", "0_1", "OP_JMP+0"))
    _claims.add((6, "attn_W_o", "0_1", "CMP+0"))
    for k in range(16):
        _claims.add((6, "attn_W_v", f"0_{2 + k}", f"FETCH_LO+{k}"))
        _claims.add((6, "attn_W_v", f"0_{18 + k}", f"FETCH_HI+{k}"))
        _claims.add((6, "attn_W_o", f"0_{2 + k}", f"AX_CARRY_LO+{k}"))
        _claims.add((6, "attn_W_o", f"0_{18 + k}", f"AX_CARRY_HI+{k}"))
    # --- Head 1: EXIT relay (NEXT_SE reads current AX) ---
    for col in ("NEXT_SE+0", "MARK_AX+0"):
        _claims.add((6, "attn_W_q", "1_0", col))
    _claims.add((6, "attn_W_k", "1_0", "MARK_AX+0"))
    _claims.add((6, "attn_W_v", "1_1", "OP_EXIT+0"))
    _claims.add((6, "attn_W_o", "1_1", "CMP+1"))
    # --- Head 2: first-step JMP relay (PC self-attends to fetched target) ---
    for col in ("MARK_PC+0", "HAS_SE+0", "MARK_AX+0", "OP_JMP+0", "CONST+0"):
        _claims.add((6, "attn_W_q", "2_0", col))
    _claims.add((6, "attn_W_k", "2_0", "MARK_PC+0"))
    _claims.add((6, "attn_W_v", "2_1", "OP_JMP+0"))
    _claims.add((6, "attn_W_o", "2_1", "CMP+0"))
    for k in range(16):
        _claims.add((6, "attn_W_v", f"2_{2 + k}", f"FETCH_LO+{k}"))
        _claims.add((6, "attn_W_v", f"2_{18 + k}", f"FETCH_HI+{k}"))
        _claims.add((6, "attn_W_o", f"2_{2 + k}", f"AX_CARRY_LO+{k}"))
        _claims.add((6, "attn_W_o", f"2_{18 + k}", f"AX_CARRY_HI+{k}"))
    # --- Head 3: first-step JSR relay (AX -> PC marker temp tag) ---
    for col in ("MARK_PC+0", "MARK_AX+0", "HAS_SE+0"):
        _claims.add((6, "attn_W_q", "3_0", col))
    _claims.add((6, "attn_W_k", "3_0", "MARK_AX+0"))
    _claims.add((6, "attn_W_v", "3_1", "OP_JSR+0"))
    _claims.add((6, "attn_W_o", "3_1", "TEMP+0"))
    # --- Head 5: first-step FETCH relay (PC marker self-attends to AX) ---
    for col in ("MARK_AX+0", "HAS_SE+0"):
        _claims.add((6, "attn_W_q", "5_0", col))
    _claims.add((6, "attn_W_k", "5_0", "MARK_PC+0"))
    for k in range(16):
        _claims.add((6, "attn_W_v", f"5_{k}", f"FETCH_LO+{k}"))
        _claims.add((6, "attn_W_v", f"5_{16 + k}", f"FETCH_HI+{k}"))
        _claims.add((6, "attn_W_o", f"5_{k}", f"FETCH_LO+{k}"))
        _claims.add((6, "attn_W_o", f"5_{16 + k}", f"FETCH_HI+{k}"))
    # Branch byte-1 relay V/O (slots 32, 33, 34) and opcode-byte relay
    # (slots 35..66).
    for slot, dim in ((32, "OP_BZ+0"), (33, "OP_BNZ+0"), (34, "OP_JSR+0")):
        _claims.add((6, "attn_W_v", f"5_{slot}", dim))
        _claims.add((6, "attn_W_o", f"5_{slot}", dim))
    for k in range(16):
        _claims.add((6, "attn_W_v", f"5_{35 + k}", f"OPCODE_BYTE_LO+{k}"))
        _claims.add((6, "attn_W_v", f"5_{51 + k}", f"OPCODE_BYTE_HI+{k}"))
        _claims.add((6, "attn_W_o", f"5_{35 + k}", f"OPCODE_BYTE_LO+{k}"))
        _claims.add((6, "attn_W_o", f"5_{51 + k}", f"OPCODE_BYTE_HI+{k}"))
    # Head-5 Q/K discriminator slots: the branch byte-0 relay (slot 52), the
    # AX-byte fetch blocker (slot 53), the fetch gate (slot 50), the HAS_SE
    # gate (slot 49), and the ENT-after-JSR SP fetch gate (slot 51).
    _claims.add((6, "attn_W_q", "5_49", "HAS_SE+0"))
    _claims.add((6, "attn_W_k", "5_49", "CONST+0"))
    _claims.add((6, "attn_W_q", "5_50", "MARK_AX+0"))
    _claims.add((6, "attn_W_q", "5_50", "CONST+0"))
    _claims.add((6, "attn_W_k", "5_50", "CONST+0"))
    _claims.add((6, "attn_W_q", "5_51", "MARK_SP+0"))
    _claims.add((6, "attn_W_q", "5_51", "HAS_SE+0"))
    _claims.add((6, "attn_W_q", "5_51", "CONST+0"))
    _claims.add((6, "attn_W_k", "5_51", "MARK_AX+0"))
    _claims.add((6, "attn_W_k", "5_51", "OP_ENT+0"))
    _claims.add((6, "attn_W_q", "5_52", "IS_BYTE+0"))
    _claims.add((6, "attn_W_q", "5_52", "H1+0"))
    _claims.add((6, "attn_W_q", "5_52", "BYTE_INDEX_0+0"))
    _claims.add((6, "attn_W_q", "5_52", "MARK_PC+0"))
    _claims.add((6, "attn_W_k", "5_52", "MARK_PC+0"))
    _claims.add((6, "attn_W_q", "5_53", "H1+1"))
    _claims.add((6, "attn_W_q", "5_53", "IS_BYTE+0"))
    _claims.add((6, "attn_W_q", "5_53", "MARK_AX+0"))
    _claims.add((6, "attn_W_q", "5_53", "H1+0"))
    _claims.add((6, "attn_W_k", "5_53", "CONST+0"))
    _claims = frozenset(_claims)

    return Operation(
        name="layer6_attn_bake",
        phase=998.5,
        reads=set(),
        writes=set(),
        kind="model",
        bake_fn=bake,
        declarative_bake_fn=bake,
        declarative_authority="spec_generated",
        migrated=True,
        claims=_claims,
        smoke_tests={"all"},
        spec_section="BLOG_SPEC.md#registers",
        produces={
            "ALU_LO": "STACK0",
            "ALU_HI": "STACK0",
        },
    )


def make_layer6_relay_heads_bake_op() -> Operation:
    """Bake the L6 PSH relay head spec into ``model.blocks[6].attn``.

    Originally an inline call in ``set_vm_weights``:
        ``_set_layer6_relay_heads(attn6, S, BD, HD)``

    Migrated as ``kind="model"`` with ``migrated=True`` and now owned by the
    local spec writer above, so this op no longer calls the legacy helper.
    Phase=998.6 so this op runs AFTER
    ``function_call_weights`` (998) AND AFTER ``layer6_attn_bake`` (998.5)
    but BEFORE ``legacy_bake`` (999).

    The ordering matters: this op writes head 7 Q[MARK_STACK0]=50,
    Q[MARK_AX]=-50, K[MARK_AX]=50, which overlap with
    ``function_call_weights``'s head 7 writes (Q[MARK_STACK0]=1050,
    Q[CONST]=-1000, K[MARK_PC]=30, K[OP_JSR]=-20). Legacy
    in-set_vm_weights ordering had relay_heads run after function_call
    weights, so relay_heads's Q[MARK_STACK0]=50 overrides the JSR-side
    1050 (PSH semantics win). The phase ordering 998 < 998.6 < 999
    preserves that override contract; the dep anchor ``layer6_relay_heads``
    above preserves the LayerCompiler topology.
    """
    def bake(model, dim_positions, S):
        del S
        attn = model.blocks[6].attn
        HD = attn.W_q.shape[0] // attn.num_heads
        _bake_layer6_relay_heads_spec(attn, _as_setdim_proxy(dim_positions), HD)

    # Dim-ownership claims. ``_bake_layer6_relay_heads_spec`` programs L6
    # heads 6 and 7 (Q/K/V/O). Head 6 owns the PSH/JSR/ENT/LEV/PSH-group
    # opcode flag relay (slots 0-7 are control flags, slots 8-23 are the
    # AX_CARRY_LO byte-0 lanes). Head 7 owns the AX_CARRY_HI -> ALU_HI relay
    # at slots 33-48. ``opcode_relay_head`` (phase=1002) re-writes the
    # head-6 cells with identical values and therefore declares no claims;
    # ownership is anchored here (see ``model_ops.make_opcode_relay_head_op``
    # docstring for the rationale).
    _claims = frozenset({
        # Head 6 Q at slot 0: gather across markers.
        (6, "attn_W_q", "6_0", "MARK_SP+0"),
        (6, "attn_W_q", "6_0", "H1+2"),
        (6, "attn_W_q", "6_0", "MARK_STACK0+0"),
        (6, "attn_W_q", "6_0", "L1H4+3"),
        (6, "attn_W_q", "6_0", "MARK_BP+0"),
        (6, "attn_W_q", "6_0", "MARK_PC+0"),
        (6, "attn_W_q", "6_0", "MARK_MEM+0"),
        (6, "attn_W_q", "6_0", "MARK_AX+0"),
        # Head 6 K at slot 0.
        (6, "attn_W_k", "6_0", "MARK_AX+0"),
        # Head 6 V opcode lanes (slots 0-7).
        (6, "attn_W_v", "6_0", "OP_LEV+0"),
        (6, "attn_W_v", "6_1", "OP_PSH+0"),
        (6, "attn_W_v", "6_2", "OP_ADJ+0"),
        # Slot 3 receives the binary-pop group (OP_ADD..OP_SC) at +0.04.
        (6, "attn_W_v", "6_3", "OP_ADD+0"),
        (6, "attn_W_v", "6_3", "OP_SUB+0"),
        (6, "attn_W_v", "6_3", "OP_MUL+0"),
        (6, "attn_W_v", "6_3", "OP_DIV+0"),
        (6, "attn_W_v", "6_3", "OP_MOD+0"),
        (6, "attn_W_v", "6_3", "OP_EQ+0"),
        (6, "attn_W_v", "6_3", "OP_NE+0"),
        (6, "attn_W_v", "6_3", "OP_LT+0"),
        (6, "attn_W_v", "6_3", "OP_GT+0"),
        (6, "attn_W_v", "6_3", "OP_LE+0"),
        (6, "attn_W_v", "6_3", "OP_GE+0"),
        (6, "attn_W_v", "6_3", "OP_OR+0"),
        (6, "attn_W_v", "6_3", "OP_XOR+0"),
        (6, "attn_W_v", "6_3", "OP_AND+0"),
        (6, "attn_W_v", "6_3", "OP_SHL+0"),
        (6, "attn_W_v", "6_3", "OP_SHR+0"),
        (6, "attn_W_v", "6_3", "OP_SI+0"),
        (6, "attn_W_v", "6_3", "OP_SC+0"),
        (6, "attn_W_v", "6_4", "OP_ENT+0"),
        (6, "attn_W_v", "6_5", "OP_JSR+0"),
        # Slot 6 is a multi-opcode pop-group flag.
        (6, "attn_W_v", "6_6", "OP_SI+0"),
        (6, "attn_W_v", "6_6", "OP_SC+0"),
        (6, "attn_W_v", "6_6", "OP_PSH+0"),
        (6, "attn_W_v", "6_6", "OP_JSR+0"),
        (6, "attn_W_v", "6_6", "OP_ENT+0"),
        (6, "attn_W_v", "6_7", "OP_SI+0"),
        (6, "attn_W_v", "6_7", "OP_SC+0"),
        # Head 6 O: opcode flag writebacks.
        (6, "attn_W_o", "6_1", "CMP+0"),
        (6, "attn_W_o", "6_1", "PSH_AT_SP+0"),
        (6, "attn_W_o", "6_2", "CMP+1"),
        (6, "attn_W_o", "6_3", "CMP+3"),
        (6, "attn_W_o", "6_4", "CMP+2"),
        (6, "attn_W_o", "6_5", "CMP+4"),
        (6, "attn_W_o", "6_5", "OP_JSR+0"),
        (6, "attn_W_o", "6_4", "OP_ENT+0"),
        (6, "attn_W_o", "6_6", "MEM_STORE+0"),
        (6, "attn_W_o", "6_7", "MEM_ADDR_SRC+0"),
        (6, "attn_W_o", "6_0", "OP_LEV+0"),
    })
    # Head 6 V + O for AX_CARRY_LO byte-0 relay (slots 8..23).
    _claims = _claims | frozenset({
        (6, "attn_W_v", f"6_{8 + k}", f"AX_CARRY_LO+{k}")
        for k in range(16)
    }) | frozenset({
        (6, "attn_W_o", f"6_{8 + k}", f"ALU_LO+{k}")
        for k in range(16)
    })
    # Head 7 Q/K at slot 0 and V/O for AX_CARRY_HI byte-1 relay (slots 33..48).
    _claims = _claims | frozenset({
        (6, "attn_W_k", "7_0", "MARK_AX+0"),
    }) | frozenset({
        (6, "attn_W_v", f"7_{33 + k}", f"AX_CARRY_HI+{k}")
        for k in range(16)
    }) | frozenset({
        (6, "attn_W_o", f"7_{33 + k}", f"ALU_HI+{k}")
        for k in range(16)
    })
    # Head 7 LEV AX_CARRY refresh: (MARK_AX + OP_LEV) -> STACK0_BYTE0 attention
    # at slot 1; CLEAN_EMBED -> AX_CARRY band at slots 2..18 + 49..63. Pairs
    # with the L16 ``l16_lev_ax_carry_lo/hi_{k}`` rules at commit 3650e01.
    _claims = _claims | frozenset({
        (6, "attn_W_q", "7_1", "MARK_AX+0"),
        (6, "attn_W_q", "7_1", "OP_LEV+0"),
        (6, "attn_W_q", "7_1", "CONST+0"),
        (6, "attn_W_k", "7_1", "STACK0_BYTE0+0"),
    }) | frozenset({
        (6, "attn_W_v", f"7_{2 + k}", f"CLEAN_EMBED_LO+{k}")
        for k in range(16)
    }) | frozenset({
        (6, "attn_W_o", f"7_{2 + k}", f"AX_CARRY_LO+{k}")
        for k in range(16)
    }) | frozenset({
        (6, "attn_W_v", "7_18", "CLEAN_EMBED_HI+0"),
        (6, "attn_W_o", "7_18", "AX_CARRY_HI+0"),
    }) | frozenset({
        (6, "attn_W_v", f"7_{48 + k}", f"CLEAN_EMBED_HI+{k}")
        for k in range(1, 16)
    }) | frozenset({
        (6, "attn_W_o", f"7_{48 + k}", f"AX_CARRY_HI+{k}")
        for k in range(1, 16)
    })

    return Operation(
        name="layer6_relay_heads_bake",
        phase=998.6,
        reads=set(),
        writes=set(),
        kind="model",
        bake_fn=bake,
        declarative_bake_fn=bake,
        declarative_authority="spec_generated",
        migrated=True,
        claims=_claims,
        smoke_tests={"all"},
        spec_section="BLOG_SPEC.md#registers",
    )


def make_layer6_bz_bnz_relay_bake_op() -> Operation:
    """Bake ``_set_bz_bnz_relay`` into ``model.blocks[6].attn`` (head 4).

    Originally an inline call in ``set_vm_weights``:
        ``_set_bz_bnz_relay(attn6, S, BD, HD)``

    Migrated as ``kind="model"`` with ``migrated=True``: the inline call
    has been removed. Phase=998.7 so this op runs AFTER
    ``layer6_attn_bake`` (998.5) and ``layer6_relay_heads_bake`` (998.6),
    BEFORE ``legacy_bake`` (999), matching the original in-set_vm_weights
    order. _set_bz_bnz_relay programs head 4's Q/K/V/O slots, left
    intentionally unprogrammed by ``_set_layer6_attn`` (head 4 is reserved
    for BZ/BNZ relay per the comment in _set_layer6_attn).

    No dep anchor needed: the legacy code never declared an op for this
    function, so its absence from the dep graph is the existing baseline.
    """
    def bake(model, dim_positions, S):
        attn = model.blocks[6].attn
        HD = attn.W_q.shape[0] // attn.num_heads
        Primitives.generate_attention_head(
            attn,
            _layer6_bz_bnz_relay_head_spec(_as_setdim_proxy(dim_positions)),
            HD,
        )

    _claims = {
        (6, "attn_W_v", "4_1", "OP_BZ+0"),
        (6, "attn_W_v", "4_2", "OP_BNZ+0"),
        (6, "attn_W_v", "4_3", "EMBED_LO+0"),
        (6, "attn_W_v", "4_4", "EMBED_HI+0"),
        (6, "attn_W_o", "4_1", "CMP+2"),
        (6, "attn_W_o", "4_2", "CMP+3"),
        (6, "attn_W_o", "4_3", "CMP+4"),
        (6, "attn_W_o", "4_4", "CMP+5"),
    }

    return Operation(
        name="layer6_bz_bnz_relay_bake",
        phase=998.7,
        reads=set(),
        writes=set(),
        kind="model",
        bake_fn=bake,
        declarative_bake_fn=bake,
        migrated=True,
        smoke_tests={
            "TestSmokeControlFlow::test_bnz_branch",
            "TestSmokeControlFlow::test_bz_branch",
        },
        spec_section="BLOG_SPEC.md#control-flow",
        claims=_claims,
        produces={
            "CMP": "PC_marker",
        },
        opcodes={"OP_BZ", "OP_BNZ"},
    )


def _layer6_bz_bnz_relay_head_spec(BD) -> DeclarativeAttentionHeadSpec:
    """Declarative L6 head 4: relay BZ/BNZ and AX-byte-zero flags."""

    L = 50.0
    AX_I = 1
    return DeclarativeAttentionHeadSpec(
        head_idx=4,
        q=(
            AP(0, BD.MARK_PC, L),
            AP(0, BD.MARK_AX, -L),
            AP(0, BD.CONST, -L * 1.3),
            AP(0, BD.OP_BZ, L / 5.0),
            AP(0, BD.OP_BNZ, L / 5.0),
            AP(5, BD.IS_BYTE, 60.0),
            AP(5, BD.H1 + 0, 60.0),
            AP(5, BD.BYTE_INDEX_0, 60.0),
            AP(5, BD.MARK_PC, -60.0),
        ),
        k=(
            AP(0, BD.L1H1 + AX_I, L),
            AP(0, BD.L1H0 + AX_I, -L),
            AP(0, BD.CONST, L),
            AP(5, BD.L1H1 + AX_I, L),
            AP(5, BD.L1H0 + AX_I, -L),
        ),
        v=(
            AP(1, BD.OP_BZ, 1.0),
            AP(2, BD.OP_BNZ, 1.0),
            AP(3, BD.EMBED_LO + 0, 1.0),
            AP(4, BD.EMBED_HI + 0, 1.0),
        ),
        o=(
            AO(BD.CMP + 2, 1, 0.2),
            AO(BD.CMP + 3, 2, 0.2),
            AO(BD.CMP + 4, 3, 1.0),
            AO(BD.CMP + 5, 4, 1.0),
        ),
    )


def make_binary_pop_sp_increment_op() -> Operation:
    """L6 FFN extension: SP += 8 for binary-pop ops (ADD/SUB/etc.).

    Originally an inline call in `set_vm_weights`:
        `_set_binary_pop_sp_increment(ffn6, S, BD)`

    Operates on `model.blocks[6].ffn` (L6 FFN). Modeled as kind="model" so we
    can resolve `ffn6` from the model handle inside the bake_fn.

    Phase 998: runs just BEFORE legacy_bake (999) so that the L6 FFN units we
    program after the function-call band are present when `_right_size_ffns`
    (called at the end of legacy_bake) prunes dead units. Running at phase
    > 999 would write into already-rightsized FFN slots that no longer exist.
    """
    def bake(model, dim_positions, S):
        proxy = _as_setdim_proxy(dim_positions)
        _lower_layer6_binary_pop_sp_increment_ir(model.blocks[6].ffn, S, proxy)

    # Dim-ownership claims. ``_layer6_binary_pop_sp_increment_rules`` lowers
    # 34 FFNRules into units 2294..2327 of the L6 FFN:
    #   - units 2294..2309: SP_LO += 1 ladder gated on EMBED_LO+k (k=0..15);
    #     each unit's W_up shares the (MARK_SP, CMP+3, -IS_BYTE/-other-marker)
    #     gate, W_gate selects EMBED_LO+k, W_down writes OUTPUT_LO+(k+8) and
    #     OUTPUT_LO+k (the canceling pair).
    #   - units 2310..2325: SP_HI += 1 ladder gated on EMBED_HI+k (k=0..15)
    #     with an additional 8-wide EMBED_LO blocker bank in W_up; W_gate
    #     selects EMBED_HI+k, W_down writes OUTPUT_HI+(k+1)%16 and OUTPUT_HI+k.
    #   - unit 2326: byte-row pop boundary fixup (BYTE_INDEX_0 + CLEAN_EMBED
    #     gates -> OUTPUT_{LO,HI}+0).
    #   - unit 2327: same idea for BYTE_INDEX_1 (-> OUTPUT_LO+1, OUTPUT_HI+0
    #     plus clean-embed cancels).
    _claims = set()
    # Shared marker / opcode-flag conditions present on every LO/HI unit
    # (rule.conditions[0:8] for the LO band; HI band adds EMBED_LO blockers).
    _lo_hi_cond_cols = (
        "MARK_SP+0", "CMP+3", "IS_BYTE+0", "MARK_PC+0", "MARK_AX+0",
        "MARK_BP+0", "MARK_STACK0+0", "MARK_MEM+0",
    )
    for unit_off in range(16):
        unit = 2294 + unit_off
        for col in _lo_hi_cond_cols:
            _claims.add((6, "ffn_W_up", str(unit), col))
        _claims.add((6, "ffn_W_gate", str(unit), f"EMBED_LO+{unit_off}"))
        new_off = (unit_off + 8) % 16
        _claims.add((6, "ffn_W_down", str(unit), f"OUTPUT_LO+{new_off}"))
        _claims.add((6, "ffn_W_down", str(unit), f"OUTPUT_LO+{unit_off}"))
    for unit_off in range(16):
        unit = 2310 + unit_off
        for col in _lo_hi_cond_cols:
            _claims.add((6, "ffn_W_up", str(unit), col))
        for lo_bit in range(8):
            _claims.add((6, "ffn_W_up", str(unit), f"EMBED_LO+{lo_bit}"))
        _claims.add((6, "ffn_W_gate", str(unit), f"EMBED_HI+{unit_off}"))
        new_carry = (unit_off + 1) % 16
        _claims.add((6, "ffn_W_down", str(unit), f"OUTPUT_HI+{new_carry}"))
        _claims.add((6, "ffn_W_down", str(unit), f"OUTPUT_HI+{unit_off}"))
    # byte_row_conditions: IS_BYTE, H1+2, CMP+3, plus 6 marker blockers
    # (-PC, -AX, -SP, -BP, -STACK0, -MEM).
    _byte_row_cond_cols = (
        "IS_BYTE+0", "H1+2", "CMP+3", "MARK_PC+0", "MARK_AX+0",
        "MARK_SP+0", "MARK_BP+0", "MARK_STACK0+0", "MARK_MEM+0",
    )
    # Unit 2326: l6_binary_pop_sp_byte1_ff_to_00_lo
    for col in _byte_row_cond_cols:
        _claims.add((6, "ffn_W_up", "2326", col))
    _claims.add((6, "ffn_W_up", "2326", "BYTE_INDEX_0+0"))
    _claims.add((6, "ffn_W_up", "2326", "CLEAN_EMBED_LO+0"))
    _claims.add((6, "ffn_W_up", "2326", "CLEAN_EMBED_HI+0"))
    _claims.add((6, "ffn_W_gate", "2326", "CONST+0"))
    _claims.add((6, "ffn_W_down", "2326", "OUTPUT_LO+0"))
    _claims.add((6, "ffn_W_down", "2326", "OUTPUT_HI+0"))
    # Unit 2327: l6_binary_pop_sp_byte2_00_to_01_lo
    for col in _byte_row_cond_cols:
        _claims.add((6, "ffn_W_up", "2327", col))
    _claims.add((6, "ffn_W_up", "2327", "BYTE_INDEX_1+0"))
    _claims.add((6, "ffn_W_up", "2327", "CLEAN_EMBED_LO+0"))
    _claims.add((6, "ffn_W_up", "2327", "CLEAN_EMBED_HI+0"))
    _claims.add((6, "ffn_W_gate", "2327", "CONST+0"))
    _claims.add((6, "ffn_W_down", "2327", "OUTPUT_LO+1"))
    _claims.add((6, "ffn_W_down", "2327", "OUTPUT_HI+0"))
    _claims.add((6, "ffn_W_down", "2327", "CLEAN_EMBED_LO+0"))
    _claims.add((6, "ffn_W_down", "2327", "CLEAN_EMBED_HI+0"))
    _claims = frozenset(_claims)

    return Operation(
        name="binary_pop_sp_increment",
        reads=set(),
        writes=set(),
        kind="model",
        bake_fn=bake,
        declarative_bake_fn=bake,
        declarative_authority="spec_generated",
        phase=998,
        migrated=True,
        layer_idx=6,
        ffn_units_used=L6_BINARY_POP_SP_INCREMENT_END_UNIT,
        claims=_claims,
        smoke_tests={"all"},
        spec_section="BLOG_SPEC.md#registers",
    )


def _layer6_binary_pop_sp_increment_rules(S: float) -> tuple[FFNRule, ...]:
    """CompilerIR rules for SP += 8 after binary pop-style ops."""

    rules = []
    write_scale = 2.0 / S
    conditions = (
        ("MARK_SP", 1.0),
        ("CMP+3", 1.0),
        ("IS_BYTE", -1_000_000.0),
        ("MARK_PC", -1_000_000.0),
        ("MARK_AX", -1_000_000.0),
        ("MARK_BP", -1_000_000.0),
        ("MARK_STACK0", -1_000_000.0),
        ("MARK_MEM", -1_000_000.0),
    )
    for k in range(16):
        new_k = (k + 8) % 16
        rules.append(FFNRule.gated_write(
            name=f"l6_binary_pop_sp_lo_{k}",
            conditions=conditions,
            threshold=1.5,
            gate=f"EMBED_LO+{k}",
            writes=(
                (f"OUTPUT_LO+{new_k}", write_scale),
                (f"OUTPUT_LO+{k}", -write_scale),
            ),
        ))

    for k in range(16):
        new_k_carry = (k + 1) % 16
        rules.append(FFNRule.gated_write(
            name=f"l6_binary_pop_sp_hi_{k}",
            conditions=conditions + tuple(
                (f"EMBED_LO+{lo_bit}", -1.0)
                for lo_bit in range(8)
            ),
            threshold=1.5,
            gate=f"EMBED_HI+{k}",
            writes=(
                (f"OUTPUT_HI+{new_k_carry}", write_scale),
                (f"OUTPUT_HI+{k}", -write_scale),
            ),
        ))

    byte_row_conditions = (
        ("IS_BYTE", 1.0),
        ("H1+2", 1.0),
        ("CMP+3", 1.0),
        ("MARK_PC", -1_000_000.0),
        ("MARK_AX", -1_000_000.0),
        ("MARK_SP", -1_000_000.0),
        ("MARK_BP", -1_000_000.0),
        ("MARK_STACK0", -1_000_000.0),
        ("MARK_MEM", -1_000_000.0),
    )
    clean_zero_byte = (
        ("CLEAN_EMBED_LO+0", 1.0),
        ("CLEAN_EMBED_HI+0", 1.0),
    )
    # The common stack-pop boundary is 0x00fff8 + 8 => 0x010000.  L6
    # already emits byte 0 on the SP marker; materialize the upper-byte
    # carry before later tail layers can reinterpret the same CMP[3] relay.
    rules.append(FFNRule.gated_write(
        name="l6_binary_pop_sp_byte1_ff_to_00_lo",
        conditions=byte_row_conditions + (
            ("BYTE_INDEX_0", 1.0),
        ) + clean_zero_byte,
        threshold=5.5,
        gate="CONST",
        writes=(
            ("OUTPUT_LO+0", 10.0 / S),
            ("OUTPUT_HI+0", 10.0 / S),
        ),
    ))
    rules.append(FFNRule.gated_write(
        name="l6_binary_pop_sp_byte2_00_to_01_lo",
        conditions=byte_row_conditions + (
            ("BYTE_INDEX_1", 1.0),
        ) + clean_zero_byte,
        threshold=5.5,
        gate="CONST",
        writes=(
            ("OUTPUT_LO+1", 10.0 / S),
            ("OUTPUT_HI+0", 10.0 / S),
            ("CLEAN_EMBED_LO+0", -1.5 / S),
            ("CLEAN_EMBED_HI+0", -1.5 / S),
        ),
    ))
    return tuple(rules)


def _lower_layer6_binary_pop_sp_increment_ir(
    ffn,
    S: float,
    BD,
    *,
    unit: int = L6_BINARY_POP_SP_INCREMENT_START_UNIT,
) -> int:
    return _lower_layer6_ffn_rules(
        ffn,
        _layer6_binary_pop_sp_increment_rules(S),
        S,
        BD,
        unit=unit,
    )


def make_putchar_think_protocol_op(
    enable_neural_io_think_protocol: bool = False,
) -> Operation:
    """L6 FFN: PUTCHAR THINK-tag I/O protocol — emit THINKING_END,
    output byte token, THINKING_START at end of PUTCHAR step.

    See ``c4_release/docs/NEURAL_IO_VIA_THINK_PROTOCOL_PLAN.md`` for the
    full design. This op implements the canonical neural-I/O mode from
    BLOG_SPEC.md:851: instead of the runner reading ``REG_AX`` byte 0
    off a just-completed PUTCHAR step (the current Phase-6 default),
    the model itself emits the protocol tokens
    ``THINKING_END, byte, THINKING_START`` between the MEM section and
    the STEP_END of a PUTCHAR step.

    Gated by ``enable_neural_io_think_protocol`` (default False) so the
    existing AX-readoff path remains the production default while the
    new bake is brought up. When False, the bake_fn is a no-op so the
    op stays registered for dep-graph stability — mirroring the
    ``make_tool_call_*_op`` and ``make_convo_io_*_op`` pattern.

    Phase 1 contract (this commit):
      - When enabled, AND ``IO_IS_PUTCHAR`` (set at AX marker by L5/L6
        FFN units 1500) with ``NEXT_SE`` and emit ``NEXT_THINKING_END``
        + suppress ``NEXT_SE`` (so the model writes THINKING_END
        instead of STEP_END at the end of a PUTCHAR step). This
        re-uses the convo-io state-machine pattern from
        ``_set_conversational_io_state_machine`` (setup_helpers.py:1789).
      - The L6 FFN ``_set_io_putchar_routing`` units (1500-1532) already
        write ``AX_CARRY → OUTPUT_LO/HI``; the same routing populates
        the next-byte-token slot so the model emits the actual byte
        right after THINKING_END.
      - Closing ``THINKING_START`` emission and the trailing STEP_END
        are deferred to a follow-up bake (see plan doc § B5).

    Phase 1 stub (this commit, enable_neural_io_think_protocol=False):
      - The bake_fn is a no-op. The op is registered for dep-graph
        stability and to expose the scaffolding so a follow-up worker
        can flip ``enable_neural_io_think_protocol=True`` and fill in
        the weight wiring without restructuring the migration chain.
      - The intended unit allocation (when enabled) is L6 FFN unit
        ~1402, immediately above ``_set_conversational_io_state_machine``
        units 1400-1401. This avoids overlap with the
        routing-FFN range (units 0-1033),
        ``_set_tool_call_detection`` (unit 1300),
        ``_set_conversational_io_state_machine`` (units 1400-1401),
        ``_set_io_putchar_routing`` (units 1500-1532), and the
        function-call-weights units (1700-2158).

    Phase 6.6 (pinned ``layer_idx=6``, ``kind="block"``): runs alongside
    other L6 block ops so the FFN unit writes survive ``_right_size_ffns``
    (phase=1200) trimming. Same phase as
    ``make_convo_io_state_machine_op`` since both extend the same L6
    FFN with adjacent unit ranges and reads/writes are disjoint.
    """
    def bake(block, dim_positions, S):
        # Phase 1 no-op stub. When ``enable_neural_io_think_protocol``
        # flips to True the body will route ``IO_IS_PUTCHAR`` through the
        # same state-machine entry as PRTF/READ (see
        # ``_set_conversational_io_state_machine`` at setup_helpers.py:1789
        # for the CMP[5]/CMP[6] pattern this will mirror) and write a new
        # L6 FFN unit at ~1402. The runner-side collector in run_vm.py is
        # already in place behind the same flag — the follow-up worker
        # only needs to fill in the weight writes here.
        return

    return Operation(
        name="putchar_think_protocol",
        phase=6.6,
        reads={"IO_IS_PUTCHAR", "NEXT_SE", "AX_CARRY_LO", "AX_CARRY_HI"},
        writes={"NEXT_THINKING_END", "NEXT_SE", "IO_STATE",
                "OUTPUT_BYTE_LO", "OUTPUT_BYTE_HI"},
        kind="block",
        bake_fn=bake,
        declarative_bake_fn=bake,
        layer_idx=6,
        migrated=True,
        smoke_tests={"all"},
        spec_section="BLOG_SPEC.md#printing-and-reading-input",
    )


def make_prtf_think_protocol_op(
    enable_neural_io_think_protocol: bool = False,
) -> Operation:
    """L6 FFN: PRTF THINK-tag I/O protocol — multi-byte variable-length
    output via the existing format-string-walk bake chain.

    See ``c4_release/docs/V9_PRTF_NEURAL_PLAN.md`` for the full Phase 2a
    design. This op extends the PUTCHAR Phase 1 THINK-tag protocol from
    single-byte (PUTCHAR) to variable-length (PRTF) output, reusing the
    existing convo-IO bake chain:

      - L5 FFN units 410-411 (``_set_conversational_io_opcode_decode``):
        decode PRTF → IO_IS_PRTF.
      - L6 attn head 4 (``_set_conversational_io_relay_heads``): relay
        IO_IS_PRTF AX → SE.
      - L6 FFN units 1400-1401 (``_set_conversational_io_state_machine``):
        IO_IS_PRTF (CMP[5]) AND NEXT_SE → emit NEXT_THINKING_END.
      - L7 attn head 7 (``_set_format_pointer_extraction``): extract
        FORMAT_PTR from previous step's STACK0.
      - L8 FFN unit 600+ (``_set_format_position_counter``): increment
        IO_FORMAT_POS on each emitted byte.
      - L9 attn head 0 (``_set_format_string_fetch_head``): fetch byte
        at FORMAT_PTR + IO_FORMAT_POS via ADDR_KEY attention.
      - L10 FFN unit 1864 (``_set_null_terminator_detection``):
        OUTPUT_BYTE == 0 → emit NEXT_THINKING_START.
      - L15 FFN unit 1200 (``_set_conversational_io_output_routing``):
        OUTPUT_BYTE → OUTPUT when IO_IN_OUTPUT_MODE.

    All the above bakes are already in place under
    ``enable_conversational_io``. This op's role is to:
      (a) register a stable dep-graph node tying PRTF's THINK protocol
          to the same ``enable_neural_io_think_protocol`` flag as
          PUTCHAR;
      (b) bake any *additional* L6 FFN units that are PRTF-specific and
          not covered by ``_set_conversational_io_state_machine`` (e.g.
          end-of-PRTF cleanup that resets IO_FORMAT_POS for the next
          PRTF call) — deferred to Phase 2b.

    Phase 2a (this commit): bake_fn is a no-op. The full bake chain
    above is already wired (gated by ``enable_conversational_io``). The
    next Phase-2b commit will:
      - Tie ``enable_neural_io_think_protocol`` to imply
        ``enable_conversational_io`` at the compile-full-vm level so
        the existing PRTF bakes fire.
      - Gate the runner-side ``_neural_prtf_emit`` shim off when the
        flag is True so the model's emitted bytes are not double-
        emitted alongside the Python format-walk fallback.
      - Optionally add an L6 FFN unit (~1403) that resets
        ``IO_FORMAT_POS`` to 0 on ``LAST_WAS_THINKING_START`` so a
        second PRTF in the same program starts at position 0 of its
        new format string.

    Gated by ``enable_neural_io_think_protocol`` (default False). When
    False, the bake_fn is a no-op — matching the PUTCHAR
    ``make_putchar_think_protocol_op`` pattern. The op stays registered
    for dep-graph stability regardless of flag state.

    Phase 6.6 (pinned ``layer_idx=6``, ``kind="block"``): runs alongside
    other L6 block ops so any future FFN unit writes survive
    ``_right_size_ffns`` (phase=1200) trimming. Same phase as
    ``make_putchar_think_protocol_op`` and
    ``make_convo_io_state_machine_op``; reads/writes declared below
    are disjoint from those two so dispatch order within phase 6.6 is
    irrelevant.
    """
    def bake(block, dim_positions, S):
        return  # Phase 2a stub; see docstring for the full wiring plan.

    return Operation(
        name="prtf_think_protocol",
        phase=6.6,
        reads=set(),
        writes=set(),
        kind="block",
        bake_fn=bake,
        declarative_bake_fn=bake,
        layer_idx=6,
        migrated=True,
        # B12 backfill: this op is a Phase 2a no-op stub that piggybacks
        # on the existing convo-IO bake chain. The L6-anchored side of
        # the chain (state-machine / pc-sp latch) sits at the same L6
        # FFN as ``layer6_routing_ffn`` (phase 6.5), and the docstring
        # explicitly groups this op with ``make_convo_io_state_machine_op``
        # at phase 6.6. Pin after the base L6 routing FFN via a B10
        # op-name reference so the dynamic scheduler honours the dep
        # edge despite empty reads/writes.
        requires={"after": "layer6_routing_ffn"},
        smoke_tests={"all"},
        spec_section="BLOG_SPEC.md#printing-and-reading-input",
    )


def make_open_clos_tool_call_op(
    enable_tool_calling: bool = False,
) -> Operation:
    """L6 FFN: OPEN/CLOS TOOL_CALL boundary opcode — dep-graph anchor.

    Per BLOG_SPEC.md:853, OPEN and CLOS cross the host boundary (file
    descriptors, ``os.open``/``os.close``) and have no sensible
    "in the transformer" implementation. The canonical design for these
    two opcodes is to emit a ``TOOL_CALL`` token at the end of an
    OPEN/CLOS step, which the runner intercepts to perform the syscall.

    This bake already exists end-to-end under ``enable_tool_calling=True``:

      - L5 FFN units 400-405 (``_set_tool_call_opcode_decode``): decode
        all 6 I/O opcodes (OPEN=30, READ=31, CLOS=32, PRTF=33,
        GETCHAR=64, PUTCHAR=65) at the AX marker → IO_IS_TOOL_CALL.
      - L6 attn head 5 (``_set_tool_call_relay_head``): relay
        IO_IS_TOOL_CALL AX → SE via ALiBi slope=5.0.
      - L6 FFN unit 1300 (``_set_tool_call_detection``): CMP[2] AND
        NEXT_SE → NEXT_TOOL_CALL, clear NEXT_SE. The model emits
        ``Token.TOOL_CALL`` (271) at the end of an I/O-opcode step.

    Each of those three bakes is wrapped in a no-op-when-False factory
    in ``flag_gated_ops.py`` (``make_tool_call_*_op``) and registered
    unconditionally in ``all_core_ops``. When
    ``enable_tool_calling=True`` is passed to ``compile_full_vm``, all
    three fire and OPEN/CLOS steps produce TOOL_CALL.

    **This op is dep-graph documentation, not a new weight write.**
    Its bake_fn is always a no-op (regardless of flag); it exists to
    register the OPEN/CLOS-specific reads/writes in the dep graph for
    discoverability and to anchor a future Phase A bake (e.g. an
    OPEN/CLOS-specific marker dim if we want to distinguish them from
    PRTF/READ at the TOOL_CALL emission position).

    The runner-side shims ``_neural_open_emit`` and ``_neural_clos_emit``
    at ``run_vm.py:1450-1495`` remain as the test-suite fallback for the
    ``pure_neural_runner`` fixture (which sets neither
    ``enable_tool_calling`` nor a syscall handler). Production builds
    that need file I/O set ``enable_tool_calling=True`` and get the
    TOOL_CALL path.

    Phase 6.7 (pinned ``layer_idx=6``, ``kind="block"``): same phase
    range as the other L6 I/O-related dep anchors. No unit writes
    occur, so the phase only matters for graph topology.
    """
    def bake(block, dim_positions, S):
        return  # Dep-graph anchor only; see docstring.

    return Operation(
        name="open_clos_tool_call",
        phase=6.7,
        reads=set(),
        writes=set(),
        kind="block",
        bake_fn=bake,
        declarative_bake_fn=bake,
        layer_idx=6,
        migrated=True,
        # B12 backfill: dep-graph anchor only (bake_fn is always a
        # no-op). Phase 6.7 sits in the same L6 I/O block as
        # ``convo_io_state_machine`` (6.6) and ``convo_io_pc_sp_latch``
        # (6.7); the docstring positions it after the L6 routing-FFN
        # bake. Pin after ``layer6_routing_ffn`` via a B10 op-name
        # reference so the dynamic scheduler honours the dep edge
        # despite empty reads/writes.
        requires={"after": "layer6_routing_ffn"},
        smoke_tests={"all"},
        spec_section="BLOG_SPEC.md#tool-use-mode",
    )
