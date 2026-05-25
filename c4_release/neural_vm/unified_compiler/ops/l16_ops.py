"""Auto-extracted per-layer factories. See ../migrated_ops.py for history."""

from ..ir import CompilerIR, FFNRule
from ..layer_compiler import Operation
from ..primitives import Primitives
from .shared import _as_setdim_proxy


def _layer16_lev_routing_rules(S: float) -> tuple[FFNRule, ...]:
    """CompilerIR rules for L16 LEV routing units 0..120."""

    rules = []
    write_scale = 2.0 / S
    first_step_gate = 30.0
    sp_cancel_conditions = (
        ("OP_LEV", 0.2),
        ("MARK_SP", 1.0),
        ("MARK_PC", -1.0),
        ("MARK_AX", -1.0),
        ("MARK_BP", -1.0),
        ("HAS_SE", first_step_gate),
        ("PSH_AT_SP", -first_step_gate),
    )
    for band, output_base in (("lo", "OUTPUT_LO"), ("hi", "OUTPUT_HI")):
        for k in range(16):
            rules.append(FFNRule.gated_write(
                name=f"l16_lev_sp_cancel_{band}_{k}",
                conditions=sp_cancel_conditions,
                threshold=31.5,
                gate=f"{output_base}+{k}",
                gate_weight=-1.0,
                writes=((f"{output_base}+{k}", write_scale),),
            ))

    sp_value_base_conditions = (
        ("OP_LEV", 1.0),
        ("MARK_SP", 1.0),
        ("MARK_BP", -15.0),
        ("MARK_PC", -15.0),
        ("MARK_AX", -50.0),
        ("MARK_STACK0", -15.0),
        ("MARK_MEM", -15.0),
        ("H3+4", -15.0),
        ("MARK_SE", -15.0),
        ("BYTE_INDEX_0", -10.0),
        ("BYTE_INDEX_1", -10.0),
        ("BYTE_INDEX_2", -10.0),
        ("BYTE_INDEX_3", -10.0),
        ("HAS_SE", first_step_gate),
        ("PSH_AT_SP", -first_step_gate),
    )
    for k in range(16):
        rules.append(FFNRule.gated_write(
            name=f"l16_lev_sp_bp_plus16_lo_{k}",
            conditions=sp_value_base_conditions + ((f"ADDR_B0_LO+{k}", 1.0),),
            threshold=40.0,
            gate=f"ADDR_B0_LO+{k}",
            writes=((f"OUTPUT_LO+{k}", write_scale),),
        ))
    for k in range(16):
        result = (k + 1) % 16
        rules.append(FFNRule.gated_write(
            name=f"l16_lev_sp_bp_plus16_hi_{k}",
            conditions=sp_value_base_conditions + ((f"ADDR_B0_HI+{k}", 1.0),),
            threshold=40.0,
            gate=f"ADDR_B0_HI+{k}",
            writes=((f"OUTPUT_HI+{result}", write_scale),),
        ))

    pc_cancel_hi_conditions = (
        ("OP_LEV", 0.2),
        ("MARK_PC", 1.0),
        ("MARK_AX", -1.0),
        ("MARK_SP", -1.0),
        ("MARK_BP", -1.0),
    )
    for k in range(16):
        rules.append(FFNRule.gated_write(
            name=f"l16_lev_pc_cancel_hi_{k}",
            conditions=pc_cancel_hi_conditions,
            threshold=1.5,
            gate=f"OUTPUT_HI+{k}",
            gate_weight=-1.0,
            writes=((f"OUTPUT_HI+{k}", write_scale),),
        ))
    for k in range(16):
        rules.append(FFNRule.gated_write(
            name=f"l16_lev_pc_temp_lo_{k}",
            conditions=(
                ("OP_LEV", 1.0),
                ("MARK_PC", 1.0),
                (f"TEMP+{k}", 1.0),
            ),
            threshold=3.5,
            gate="CONST",
            writes=((f"OUTPUT_LO+{k}", write_scale),),
        ))
    for k in range(16):
        rules.append(FFNRule.gated_write(
            name=f"l16_lev_pc_temp_hi_{k}",
            conditions=(
                ("OP_LEV", 1.0),
                ("MARK_PC", 1.0),
                (f"TEMP+{16 + k}", 1.0),
            ),
            threshold=3.5,
            gate="CONST",
            writes=((f"OUTPUT_HI+{k}", write_scale),),
        ))

    byte_zero_base = (
        ("OP_LEV", 0.5),
        ("MARK_PC", -1.5),
        ("MARK_AX", -10.0),
        ("MARK_SP", -10.0),
        ("MARK_BP", -10.0),
        ("MARK_STACK0", -10.0),
        ("MARK_MEM", -10.0),
        ("MARK_SE", -10.0),
        ("H1+1", -10.0),
        ("NEXT_AX", -1.5),
        ("NEXT_SP", -1.5),
        ("NEXT_BP", -1.5),
        ("NEXT_STACK0", -1.5),
        ("NEXT_MEM", -1.5),
        ("NEXT_SE", -1.5),
    )
    for byte_idx in ("BYTE_INDEX_1", "BYTE_INDEX_2", "BYTE_INDEX_3"):
        rules.append(FFNRule.gated_write(
            name=f"l16_lev_clear_output_lo10_{byte_idx}",
            conditions=byte_zero_base + ((byte_idx, 1.0),),
            threshold=4.0,
            gate="CONST",
            writes=(("OUTPUT_LO+10", -10.0 / S),),
        ))
    for byte_idx in ("BYTE_INDEX_1", "BYTE_INDEX_2", "BYTE_INDEX_3"):
        rules.append(FFNRule.gated_write(
            name=f"l16_lev_set_output_lo0_{byte_idx}",
            conditions=byte_zero_base + ((byte_idx, 1.0),),
            threshold=4.0,
            gate="CONST",
            writes=(("OUTPUT_LO+0", 5.0 / S),),
        ))
    for byte_idx in ("BYTE_INDEX_1", "BYTE_INDEX_2", "BYTE_INDEX_3"):
        rules.append(FFNRule.gated_write(
            name=f"l16_lev_set_output_hi0_{byte_idx}",
            conditions=byte_zero_base + ((byte_idx, 1.0),),
            threshold=4.0,
            gate="CONST",
            writes=(("OUTPUT_HI+0", 5.0 / S),),
        ))

    # After a strict neural LEV, the next AX marker can still carry a
    # low-strength OP_IMM relay even though FETCH has gone quiet. L3/L8 already
    # hold the correct AX value in AX_CARRY_LO/HI at that marker; this late
    # declarative route materializes that carried value into OUTPUT so the
    # following EXIT observes the neural return value instead of a zero token.
    stale_imm_ax_conditions = (
        ("OP_IMM", 1.0),
        ("MARK_AX", 1.0),
        ("MARK_PC", -8.0),
        ("IS_BYTE", -10.0),
        ("OP_EXIT", -20.0),
        ("OP_JMP", -20.0),
    )
    for k in range(16):
        rules.append(FFNRule.gated_write(
            name=f"l16_stale_imm_ax_carry_lo_{k}",
            conditions=stale_imm_ax_conditions,
            threshold=1.5,
            gate=f"AX_CARRY_LO+{k}",
            writes=((f"OUTPUT_LO+{k}", 2.0 / S),),
        ))
    for k in range(16):
        rules.append(FFNRule.gated_write(
            name=f"l16_stale_imm_ax_carry_hi_{k}",
            conditions=stale_imm_ax_conditions,
            threshold=1.5,
            gate=f"AX_CARRY_HI+{k}",
            writes=((f"OUTPUT_HI+{k}", 2.0 / S),),
        ))

    # SI/SC preserve AX while using STACK0 as the memory address source. The
    # store generation later in the same autoregressive step reads the emitted
    # current AX bytes, so the neural path must materialize AX_CARRY back into
    # OUTPUT at the AX marker before those bytes are generated.
    store_ax_conditions = (
        ("OP_SI", 1.0),
        ("OP_SC", 1.0),
        ("MARK_AX", 1.0),
        ("MARK_PC", -8.0),
        ("IS_BYTE", -10.0),
        ("OP_EXIT", -20.0),
        ("OP_JMP", -20.0),
    )
    for k in range(16):
        rules.append(FFNRule.gated_write(
            name=f"l16_store_ax_carry_lo_{k}",
            conditions=store_ax_conditions,
            threshold=4.0,
            gate=f"AX_CARRY_LO+{k}",
            writes=((f"OUTPUT_LO+{k}", 2.0 / S),),
        ))
    for k in range(16):
        rules.append(FFNRule.gated_write(
            name=f"l16_store_ax_carry_hi_{k}",
            conditions=store_ax_conditions,
            threshold=4.0,
            gate=f"AX_CARRY_HI+{k}",
            writes=((f"OUTPUT_HI+{k}", 2.0 / S),),
        ))

    # LC is an 8-bit load. L15 head 0 writes the loaded byte and can leave
    # that byte replicated at subsequent AX byte positions; clear bytes 1-3
    # after the load lookup so EXIT sees a zero-extended char.
    lc_ax_byte_conditions = (
        ("OP_LC_RELAY", 1.0),
        ("IS_BYTE", 1.0),
        ("H1+1", 1.0),
        ("BYTE_INDEX_3", -4.0),
    )
    rules.append(FFNRule.constant_write(
        name="l16_lc_ax_bytes_1_3_clear_lo",
        conditions=lc_ax_byte_conditions,
        threshold=2.5,
        writes=tuple((f"OUTPUT_LO+{k}", -300.0 / S) for k in range(16)),
    ))
    rules.append(FFNRule.constant_write(
        name="l16_lc_ax_bytes_1_3_clear_hi",
        conditions=lc_ax_byte_conditions,
        threshold=2.5,
        writes=tuple((f"OUTPUT_HI+{k}", -300.0 / S) for k in range(16)),
    ))
    rules.append(FFNRule.constant_write(
        name="l16_lc_ax_bytes_1_3_zero_lo",
        conditions=lc_ax_byte_conditions,
        threshold=2.5,
        writes=(("OUTPUT_LO+0", 500.0 / S),),
    ))
    rules.append(FFNRule.constant_write(
        name="l16_lc_ax_bytes_1_3_zero_hi",
        conditions=lc_ax_byte_conditions,
        threshold=2.5,
        writes=(("OUTPUT_HI+0", 500.0 / S),),
    ))

    # SI/SC top-store steps stage the current value at the STACK0 marker
    # before L15 runs. The historical memory lookup can still add an older
    # byte-0 value over that staged value because the current MEM value is
    # emitted later in the same autoregressive step. When the marker carries
    # store residue, make the already-staged non-zero nibble authoritative
    # again and cancel the stale zero lane that L15 most commonly contributes.
    top_store_stack0_conditions = (
        ("MARK_STACK0", 1.0),
        ("HAS_SE", 1.0),
        ("CMP+3", 1.0),
        ("MEM_STORE", 1.0),
        ("EMBED_LO+0", 1.0),
        ("IS_BYTE", -10.0),
        ("MARK_PC", -10.0),
        ("MARK_AX", -10.0),
        ("MARK_SP", -10.0),
        ("MARK_BP", -10.0),
        ("MARK_MEM", -10.0),
    )
    top_store_restore = 50.0 / S
    for k in range(1, 16):
        rules.append(FFNRule.gated_write(
            name=f"l16_top_store_stack0_restore_lo_{k}",
            conditions=top_store_stack0_conditions,
            threshold=7.0,
            gate=f"OUTPUT_LO+{k}",
            writes=(
                (f"OUTPUT_LO+{k}", top_store_restore),
                ("OUTPUT_LO+0", -top_store_restore),
            ),
        ))

    # Retained memory rows can make L15 copy the byte-0 value into the next
    # STACK0 byte during pop-group stack-top reconstruction. When the previous
    # STACK0 byte is the current low byte 0x01 at the e0 top slot, byte 1 is
    # the high zero byte and should stay zero.
    stack0_byte1_zero_conditions = (
        ("IS_BYTE", 1.0),
        ("STACK0_BYTE0", 1.0),
        ("HAS_SE", 1.0),
        ("CMP+3", 1.0),
        ("ADDR_B0_LO+0", 1.0),
        ("ADDR_B0_HI+14", 1.0),
        ("CLEAN_EMBED_LO+1", 1.0),
        ("CLEAN_EMBED_HI+0", 1.0),
        ("MEM_STORE", -10.0),
        ("MARK_PC", -10.0),
        ("MARK_AX", -10.0),
        ("MARK_SP", -10.0),
        ("MARK_BP", -10.0),
        ("MARK_STACK0", -10.0),
        ("MARK_MEM", -10.0),
    )
    stack0_byte1_zero = 1000.0 / S
    rules.append(FFNRule.constant_write(
        name="l16_stack0_byte1_zero_after_unit_low_byte",
        conditions=stack0_byte1_zero_conditions,
        threshold=8.0,
        writes=tuple(
            (f"OUTPUT_LO+{k}", stack0_byte1_zero if k == 0 else -stack0_byte1_zero)
            for k in range(16)
        ) + tuple(
            (f"OUTPUT_HI+{k}", stack0_byte1_zero if k == 0 else -stack0_byte1_zero)
            for k in range(16)
        ),
    ))

    # JMP changes PC but preserves AX. The fetch machinery can leave the jump
    # target immediate in OUTPUT/AX_CARRY at the AX marker. The previous AX
    # nibble is still present at low strength; when it is different from the
    # fetched nibble, boost that staged value and cancel the fetched lanes.
    jmp_ax_preserve_conditions = (
        ("OP_JMP", 0.2),
        ("MARK_AX", 1.0),
        ("IS_BYTE", -10.0),
        ("MARK_PC", -10.0),
        ("MARK_SP", -10.0),
        ("MARK_BP", -10.0),
        ("MARK_STACK0", -10.0),
        ("MARK_MEM", -10.0),
    )
    jmp_ax_preserve = 50.0 / S
    for k in range(16):
        extra_conditions = [(f"FETCH_LO+{k}", -2.0)]
        if k == 11:
            # The REG_AX marker row can carry a weak low-11 artifact through
            # TEMP[11]. Do not treat that marker residue as the preserved AX
            # value when JMP is merely preserving the prior AX byte.
            extra_conditions.append(("TEMP+11", -2.0))
        rules.append(FFNRule.gated_write(
            name=f"l16_jmp_ax_preserve_lo_{k}",
            conditions=jmp_ax_preserve_conditions + tuple(extra_conditions),
            threshold=1.5,
            gate=f"OUTPUT_LO+{k}",
            writes=tuple(
                (
                    f"OUTPUT_LO+{j}",
                    jmp_ax_preserve if j == k else -jmp_ax_preserve,
                )
                for j in range(16)
            ) + tuple(
                (
                    f"AX_CARRY_LO+{j}",
                    jmp_ax_preserve if j == k else -jmp_ax_preserve,
                )
                for j in range(16)
            ),
        ))

    # L3 carry-forward stores the current BP marker byte in EMBED_LO/HI, but
    # the legacy marker identity threshold is too high for the compact
    # residual scale after JSR/ENT. Re-materialize BP marker byte 0 from EMBED
    # during ordinary post-prologue steps so LEA/LOAD/STORE preserve BP.
    bp_marker_passthrough_conditions = (
        ("MARK_BP", 1.0),
        ("HAS_SE", 1.0),
        ("IS_BYTE", -10.0),
        ("OP_ENT", -2.0),
        ("OP_LEV", -2.0),
    )
    for k in range(16):
        rules.append(FFNRule.gated_write(
            name=f"l16_bp_marker_passthrough_lo_{k}",
            conditions=bp_marker_passthrough_conditions,
            threshold=1.5,
            gate=f"EMBED_LO+{k}",
            writes=((f"OUTPUT_LO+{k}", 10.0 / S),),
        ))
    for k in range(16):
        rules.append(FFNRule.gated_write(
            name=f"l16_bp_marker_passthrough_hi_{k}",
            conditions=bp_marker_passthrough_conditions,
            threshold=1.5,
            gate=f"EMBED_HI+{k}",
            writes=((f"OUTPUT_HI+{k}", 10.0 / S),),
        ))

    # After ENT, BP is typically 0x0000fff0 for local-frame programs. At the
    # BP byte-1 query, the just-emitted byte is 0xff; byte 2 must therefore be
    # zero, and the late "initial BP byte2 = 0x01" tail rule must be blocked.
    bp_after_ent_byte2_zero_conditions = (
        ("IS_BYTE", 1.0),
        ("HAS_SE", 1.0),
        ("H1+3", 10.0),
        ("H1+1", -100.0),
        ("H1+2", -100.0),
        ("H3+4", -100.0),
        ("BYTE_INDEX_1", 1.0),
        ("CLEAN_EMBED_LO+15", 1.0),
        ("CLEAN_EMBED_HI+15", 1.0),
        ("MARK_AX", -100.0),
        ("MARK_PC", -100.0),
        ("MARK_SP", -100.0),
        ("MARK_BP", -100.0),
        ("MARK_STACK0", -100.0),
        ("MARK_MEM", -100.0),
    )
    rules.append(FFNRule.constant_write(
        name="l16_bp_after_ent_byte2_zero",
        conditions=bp_after_ent_byte2_zero_conditions,
        threshold=14.5,
        writes=tuple(
            (f"OUTPUT_LO+{k}", (10000.0 / S) if k == 0 else (-10000.0 / S))
            for k in range(16)
        ) + tuple(
            (f"OUTPUT_HI+{k}", (10000.0 / S) if k == 0 else (-10000.0 / S))
            for k in range(16)
        ),
    ))

    # Function prologues after a JSR start with SP=0xfff8. ENT then saves BP
    # and allocates the local frame, so the low byte becomes ``0xf0 - imm``.
    # L6 relays the ENT immediate from the AX marker to the SP marker; these
    # rules make that dynamic frame size authoritative for byte 0. C4 stack
    # frames are 8-byte aligned, so the low nibble is either 0 or 8.
    ent_sp_frame_conditions = (
        ("MARK_SP", 10.0),
        ("HAS_SE", 1.0),
        ("OP_ENT", 0.2),
        ("IS_BYTE", -1000.0),
        ("MARK_AX", -1000.0),
        ("MARK_PC", -1000.0),
        ("MARK_BP", -1000.0),
        ("MARK_STACK0", -1000.0),
        ("MARK_MEM", -1000.0),
    )
    ent_frame_strength = 5000.0 / S
    for imm_lo, result_lo in ((0, 0), (8, 8)):
        rules.append(FFNRule.gated_write(
            name=f"l16_ent_frame_sp_byte0_lo_{imm_lo:x}",
            conditions=ent_sp_frame_conditions + (
                (f"FETCH_LO+{imm_lo}", 1.0),
            ),
            threshold=12.5,
            gate=f"FETCH_LO+{imm_lo}",
            writes=tuple(
                (
                    f"OUTPUT_LO+{k}",
                    ent_frame_strength if k == result_lo else -ent_frame_strength,
                )
                for k in range(16)
            ),
        ))
    for imm_hi in range(16):
        result_hi = (15 - imm_hi) & 0xF
        rules.append(FFNRule.constant_write(
            name=f"l16_ent_frame_sp_byte0_hi_lo0_{imm_hi:x}",
            conditions=ent_sp_frame_conditions + (
                ("FETCH_LO+0", 1.0),
                (f"FETCH_HI+{imm_hi}", 1.0),
            ),
            threshold=13.5,
            writes=tuple(
                (
                    f"OUTPUT_HI+{k}",
                    ent_frame_strength if k == result_hi else -ent_frame_strength,
                )
                for k in range(16)
            ),
        ))
    for imm_hi in range(16):
        result_hi = (14 - imm_hi) & 0xF
        rules.append(FFNRule.constant_write(
            name=f"l16_ent_frame_sp_byte0_hi_lo8_{imm_hi:x}",
            conditions=ent_sp_frame_conditions + (
                ("FETCH_LO+8", 1.0),
                (f"FETCH_HI+{imm_hi}", 1.0),
            ),
            threshold=13.5,
            writes=tuple(
                (
                    f"OUTPUT_HI+{k}",
                    ent_frame_strength if k == result_hi else -ent_frame_strength,
                )
                for k in range(16)
            ),
        ))

    # Non-store opcodes still emit a MEM row in the trace, but its value bytes
    # are zero. L14 value heads can see tiny MEM_STORE residue and leak the
    # current AX byte into those positions, so zero MEM value bytes again after
    # L14 unless a real store has asserted MEM_STORE.
    for idx, mem_val_dim in enumerate(
        ("MEM_VAL_B0", "MEM_VAL_B1", "MEM_VAL_B2", "MEM_VAL_B3")
    ):
        rules.append(FFNRule.constant_write(
            name=f"l16_nonstore_mem_value{idx}_zero",
            conditions=(
                ("IS_BYTE", 1.0),
                ("H3+4", 1.0),
                (mem_val_dim, 1.0),
                ("MEM_STORE", -100.0),
                ("MARK_MEM", -100.0),
            ),
            threshold=2.5,
            writes=tuple(
                (f"OUTPUT_LO+{k}", (100.0 / S) if k == 0 else (-100.0 / S))
                for k in range(16)
            ) + tuple(
                (f"OUTPUT_HI+{k}", (100.0 / S) if k == 0 else (-100.0 / S))
                for k in range(16)
            ),
        ))

    # L6's historical PSH high-nibble decrement used negative gate terms to
    # block borrow when the old SP low nibble was 8..15. A negative gate flips
    # the cancellation write positive, so 0xe8 - 8 can look like 0xd0. At the
    # SP marker, restore the old high nibble when the low nibble proves there
    # was no borrow.
    psh_no_borrow_low_gate = tuple(
        (f"EMBED_LO+{k}", 1.0)
        for k in range(8, 16)
    )
    for hi in range(16):
        rules.append(FFNRule.gated_write(
            name=f"l16_psh_sp_no_borrow_hi_{hi}",
            conditions=(
                ("PSH_AT_SP", 1.0),
                ("MARK_SP", 1.0),
                ("HAS_SE", 1.0),
                ("IS_BYTE", -10.0),
                (f"EMBED_HI+{hi}", 1.0),
            ),
            threshold=4.5,
            gate_terms=psh_no_borrow_low_gate,
            writes=tuple(
                (
                    f"OUTPUT_HI+{k}",
                    (4.0 / S) if k == hi else (-4.0 / S),
                )
                for k in range(16)
            ),
        ))

    # LEA after the JSR/ENT prologue materializes local frame addresses like
    # BP-8 at AX. L8 computes byte 0 at the AX marker; subsequent AX byte
    # positions only carry the L7 LEA relay (CMP+7), so byte 1 needs a late
    # declarative write to 0xff. Gate on HAS_SE so the existing first-step LEA
    # bootstrap remains the owner for the initial BP=0x10000 case.
    lea_ax_byte1_conditions = (
        ("CMP+7", 1.0),
        ("HAS_SE", 1.0),
        ("H1+1", 1.0),
        ("IS_BYTE", 1.0),
        ("BYTE_INDEX_0", 1.0),
        ("MARK_PC", -10.0),
        ("MARK_AX", -10.0),
        ("MARK_SP", -10.0),
        ("MARK_BP", -10.0),
        ("MARK_STACK0", -10.0),
        ("MARK_MEM", -10.0),
    )
    rules.append(FFNRule.constant_write(
        name="l16_lea_local_ax_byte1_ff_lo",
        conditions=lea_ax_byte1_conditions,
        threshold=4.5,
        writes=tuple(
            (
                f"OUTPUT_LO+{k}",
                (20.0 / S) if k == 15 else (-20.0 / S),
            )
            for k in range(16)
        ),
    ))
    rules.append(FFNRule.constant_write(
        name="l16_lea_local_ax_byte1_ff_hi",
        conditions=lea_ax_byte1_conditions,
        threshold=4.5,
        writes=tuple(
            (
                f"OUTPUT_HI+{k}",
                (20.0 / S) if k == 15 else (-20.0 / S),
            )
            for k in range(16)
        ),
    ))
    return tuple(rules)


def make_layer16_lev_routing_ir(S: float = 100.0) -> CompilerIR:
    """Declarative IR for L16 LEV routing."""

    ir = CompilerIR()
    ir.layer(0).ffn.rules.extend(_layer16_lev_routing_rules(S))
    return ir


def lower_layer16_lev_routing_ir(ffn, S: float, BD, *, start_unit: int = 0) -> int:
    """Lower the declarative L16 LEV routing rules and return next unit."""

    rules = _layer16_lev_routing_rules(S)
    dim_positions = Primitives.dim_positions_from_bd(
        BD,
        Primitives.ffn_rule_dim_names(rules),
    )
    return Primitives.lower_ffn_rules(
        ffn,
        rules,
        dim_positions,
        start_unit=start_unit,
        S=S,
    )


def make_layer16_lev_routing_op() -> Operation:
    """L16 FFN: LEV routing — SP = BP + 16."""
    def bake(ffn, dim_positions, S):
        lower_layer16_lev_routing_ir(
            ffn,
            S,
            _as_setdim_proxy(dim_positions),
        )

    return Operation(
        name="layer16_lev_routing",
        phase=16,
        reads={"MARK_SP", "MARK_PC", "MARK_AX", "OP_ENT", "OP_LEV", "OP_IMM",
               "OP_EXIT", "OP_JMP", "OP_SI", "OP_SC", "OP_LC_RELAY",
               "ADDR_B0_LO", "ADDR_B0_HI",
               "TEMP", "HAS_SE", "IS_BYTE", "PSH_AT_SP", "EMBED_LO",
               "EMBED_HI", "CLEAN_EMBED_LO", "CLEAN_EMBED_HI",
               "FETCH_LO", "FETCH_HI",
               "MARK_BP", "MARK_STACK0", "H1", "H3",
               "CMP", "MEM_STORE", "MEM_VAL_B0", "MEM_VAL_B1",
               "MEM_VAL_B2", "MEM_VAL_B3", "BYTE_INDEX_0", "BYTE_INDEX_3",
               "STACK0_BYTE0",
               "AX_CARRY_LO", "AX_CARRY_HI"},
        writes={"OUTPUT_LO", "OUTPUT_HI"},
        kind="ffn",
        layer_idx=16,
        bake_fn=bake,
        compiler_ir=make_layer16_lev_routing_ir(),
        migrated=True,
        # ``_set_layer16_lev_routing`` writes 121 units (vm_step.py:6845):
        #   16 (cancel OUTPUT_LO at SP) + 16 (cancel OUTPUT_HI at SP) +
        #   16 (SP=BP+16 lo) + 16 (SP=BP+16 hi) +
        #   16 (cancel OUTPUT_HI at PC) + 16 (TEMP_LO→OUTPUT_LO) +
        #   16 (TEMP_HI→OUTPUT_HI) + 3 (clear OUTPUT_LO[10] byte 1-3) +
        #   3 (set OUTPUT_LO[0] byte 1-3) + 3 (set OUTPUT_HI[0] byte 1-3)
        #   = 121. The bytes 1-3 blocks (if False) are disabled.
        # Plus 32 late AX-carry materialization units for the stale-IMM AX
        # marker state reached after strict neural LEV, and 32 for SI/SC AX
        # preservation before same-step memory-store generation, 32 BP marker
        # passthrough units, 15 current top-store STACK0 restore units, one
        # retained-memory STACK0 byte-1 zero guard, 16 JMP AX preserve units,
        # one BP byte-2 post-ENT zero guard, 4 non-store MEM value zero guards,
        # 16 PSH SP no-borrow high-nibble restores, 34 ENT dynamic-frame SP
        # byte-0 units, plus 2 LEA local-frame byte-1 materialization units.
        ffn_units_used=310,
        smoke_tests={"TestSmokeFunctionCall::test_simple_function"},
        spec_section="BLOG_SPEC.md#function-calls",
    )
