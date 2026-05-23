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
                ("OP_LEV", 0.1),
                ("MARK_PC", 1.0),
                (f"TEMP+{k}", 1.0),
            ),
            threshold=2.0,
            gate="CONST",
            writes=((f"OUTPUT_LO+{k}", write_scale),),
        ))
    for k in range(16):
        rules.append(FFNRule.gated_write(
            name=f"l16_lev_pc_temp_hi_{k}",
            conditions=(
                ("OP_LEV", 0.1),
                ("MARK_PC", 1.0),
                (f"TEMP+{16 + k}", 1.0),
            ),
            threshold=2.0,
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
        reads={"MARK_SP", "MARK_PC", "MARK_AX", "OP_LEV", "OP_IMM",
               "OP_EXIT", "OP_JMP", "OP_SI", "OP_SC", "OP_LC_RELAY",
               "ADDR_B0_LO", "ADDR_B0_HI",
               "TEMP", "HAS_SE", "IS_BYTE", "PSH_AT_SP", "EMBED_LO",
               "EMBED_HI", "MARK_BP", "MARK_STACK0", "H1", "H3",
               "BYTE_INDEX_3",
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
        # preservation before same-step memory-store generation.
        ffn_units_used=189,
        smoke_tests={"TestSmokeFunctionCall::test_simple_function"},
        spec_section="BLOG_SPEC.md#function-calls",
    )
