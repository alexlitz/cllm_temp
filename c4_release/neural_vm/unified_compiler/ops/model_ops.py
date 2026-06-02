"""Model-level and post-pass op factories. See ../migrated_ops.py for history."""

from ..ir import CompilerIR, FFNRule, TokenEmbeddingRule
from ..layer_compiler import Operation
from ..primitives import AO, AP, DeclarativeAttentionHeadSpec, Primitives
import torch.nn as nn
from .shared import _as_setdim_proxy, setup_token_embeddings, setup_head_weights


_IO_PUTCHAR_ROUTING_START_UNIT = 1500


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
        FFNRule.constant_write(
            name="io_putchar_detect",
            conditions=conditions,
            threshold=T,
            writes=(("IO_IS_PUTCHAR", write_scale),),
        ),
    ]
    for k in range(16):
        rules.append(FFNRule.gated_write(
            name=f"io_putchar_route_lo_{k}",
            conditions=conditions,
            threshold=T,
            gate=f"AX_CARRY_LO+{k}",
            writes=((f"OUTPUT_LO+{k}", write_scale),),
        ))
    for k in range(16):
        rules.append(FFNRule.gated_write(
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
        writes=set(),
        kind="model",
        declarative_bake_fn=bake,
        declarative_authority="spec_generated",
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

    Mirrors ``l5_ops._opcode_decode_jsr_temp0_blank_rule`` (the same
    pattern is used by ``opcode_decode_ffn`` for its unit-52 blank).
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
        FFNRule.constant_write(
            name="lea_first_step_alu_lo_init",
            conditions=conditions,
            threshold=1.5,
            writes=(("ALU_LO+0", write_scale),),
        ),
        FFNRule.constant_write(
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
        ("CMP+4", 1.0),
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
        FFNRule.gated_write(
            name="jsr_stack0_cancel_l3_default_lo",
            conditions=conditions,
            threshold=T_jsr_s0,
            gate="CONST",
            writes=(("OUTPUT_LO+0", write_scale_cancel),),
        ),
        FFNRule.gated_write(
            name="jsr_stack0_cancel_l3_default_hi",
            conditions=conditions,
            threshold=T_jsr_s0,
            gate="CONST",
            writes=(("OUTPUT_HI+0", write_scale_cancel),),
        ),
    ]
    for k in range(16):
        rules.append(FFNRule.gated_write(
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
        rules.append(FFNRule.gated_write(
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

    Gates on MARK_PC + TEMP[0] (IS_JSR flag relayed by L6 head 3), with
    strong negative blockers for every other opcode (NOP, EXIT, JMP, BZ,
    BNZ, IMM, LEV, ENT) to prevent spurious firing on non-JSR steps
    where TEMP[0] is polluted by L6 head 4 (BZ/BNZ relay). IS_BYTE
    blocker confines to the PC marker position.
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

    Layout (matches vm_step.py ~8718+):

      * 16 OUTPUT_LO cancel units: gate=-OUTPUT_LO[k], writes OUTPUT_LO[k]
      * 16 OUTPUT_HI cancel units: gate=-OUTPUT_HI[k], writes OUTPUT_HI[k]
      * 16 FETCH_LO -> OUTPUT_LO[target_lo] target units
      * 16 FETCH_HI reserved gate units (no writes; legacy reservation
        for targets >= 256 that would need a FETCH_HI byte path)
      * 16 FETCH_LO -> OUTPUT_HI[target_hi_from_lo] carry units

    All units share the same conditions (MARK_PC + TEMP[0] + opcode
    blockers + IS_BYTE blocker). The reserved FETCH_HI block keeps the
    unit cursor aligned with the legacy layout.
    """
    from ...constants import INSTR_WIDTH, PC_OFFSET

    T_jsr_pc = 4.0
    write_scale = 2.0 / S
    conditions = _function_call_jsr_pc_override_conditions()

    rules: list[FFNRule] = []
    for k in range(16):
        rules.append(FFNRule.gated_write(
            name=f"jsr_pc_cancel_output_lo_{k}",
            conditions=conditions,
            threshold=T_jsr_pc,
            gate=f"OUTPUT_LO+{k}",
            gate_weight=-1.0,
            writes=((f"OUTPUT_LO+{k}", write_scale),),
        ))
    for k in range(16):
        rules.append(FFNRule.gated_write(
            name=f"jsr_pc_cancel_output_hi_{k}",
            conditions=conditions,
            threshold=T_jsr_pc,
            gate=f"OUTPUT_HI+{k}",
            gate_weight=-1.0,
            writes=((f"OUTPUT_HI+{k}", write_scale),),
        ))
    for k in range(16):
        target_lo = ((k * INSTR_WIDTH) + PC_OFFSET) & 0xF
        rules.append(FFNRule.gated_write(
            name=f"jsr_pc_target_lo_{k}",
            conditions=conditions,
            threshold=T_jsr_pc,
            gate=f"FETCH_LO+{k}",
            writes=((f"OUTPUT_LO+{target_lo}", write_scale),),
        ))
    # FETCH_HI reserved band: gate is wired but no down write. Legacy
    # bake writes ``ffn6.W_gate[unit, FETCH_HI+k] = 1.0`` and no down
    # assignment because JSR fixtures target instruction indexes < 16.
    for k in range(16):
        rules.append(FFNRule.gated_write(
            name=f"jsr_pc_fetch_hi_reserved_{k}",
            conditions=conditions,
            threshold=T_jsr_pc,
            gate=f"FETCH_HI+{k}",
            writes=(),
        ))
    for k in range(16):
        target_hi_from_lo = ((k * INSTR_WIDTH) + PC_OFFSET) >> 4
        rules.append(FFNRule.gated_write(
            name=f"jsr_pc_target_hi_from_lo_{k}",
            conditions=conditions,
            threshold=T_jsr_pc,
            gate=f"FETCH_LO+{k}",
            writes=((f"OUTPUT_HI+{target_hi_from_lo}", write_scale),),
        ))
    return tuple(rules)


def _function_call_jsr_ax_passthrough_rules(S: float) -> tuple[FFNRule, ...]:
    """JSR AX passthrough: AX_CARRY -> OUTPUT at AX marker (32 units)."""
    T = 4.0
    write_scale = 2.0 / S
    conditions = (("OP_JSR", 1.0), ("MARK_AX", 1.0))
    rules: list[FFNRule] = []
    for k in range(16):
        rules.append(FFNRule.gated_write(
            name=f"jsr_ax_passthrough_lo_{k}",
            conditions=conditions,
            threshold=T,
            gate=f"AX_CARRY_LO+{k}",
            writes=((f"OUTPUT_LO+{k}", write_scale),),
        ))
    for k in range(16):
        rules.append(FFNRule.gated_write(
            name=f"jsr_ax_passthrough_hi_{k}",
            conditions=conditions,
            threshold=T,
            gate=f"AX_CARRY_HI+{k}",
            writes=((f"OUTPUT_HI+{k}", write_scale),),
        ))
    return tuple(rules)


def _function_call_ent_stack0_rules(S: float) -> tuple[FFNRule, ...]:
    """ENT STACK0 = old_BP at STACK0 marker (32 units).

    At STACK0 marker when ENT (CMP[2]=1, MARK_STACK0=1): cancel EMBED
    identity (gate_terms[EMBED_*+k]=-1) and write TEMP (= old BP relayed
    by L5 head 5 above). L6 attn head 6 broadcasts OP_ENT through CMP[2]
    (see ``layer6_relay_heads_bake``), which is what these rules gate on.
    """
    T_ent_s0 = 1.5
    write_scale = 2.0 / S
    conditions = (("CMP+2", 1.0), ("MARK_STACK0", 1.0))
    rules: list[FFNRule] = []
    for k in range(16):
        rules.append(FFNRule.gated_write(
            name=f"ent_stack0_lo_{k}",
            conditions=conditions,
            threshold=T_ent_s0,
            gate_terms=(
                (f"EMBED_LO+{k}", -1.0),
                (f"TEMP+{k}", 1.0),
            ),
            writes=((f"OUTPUT_LO+{k}", write_scale),),
        ))
    for k in range(16):
        rules.append(FFNRule.gated_write(
            name=f"ent_stack0_hi_{k}",
            conditions=conditions,
            threshold=T_ent_s0,
            gate_terms=(
                (f"EMBED_HI+{k}", -1.0),
                (f"TEMP+{16 + k}", 1.0),
            ),
            writes=((f"OUTPUT_HI+{k}", write_scale),),
        ))
    return tuple(rules)


def _function_call_ent_bp_rules(S: float) -> tuple[FFNRule, ...]:
    """ENT BP = SP - 8 at BP marker (32 units).

    Each LO unit writes ``OUTPUT_LO[new_k] += 2/S`` AND
    ``OUTPUT_LO[k] -= 2/S`` (cancels identity at source). new_k =
    (k - 8) % 16. Each HI unit writes the borrowed-down high nibble
    (new_k_borrow = (k - 1) % 16) and cancels identity at k.

    Borrow detection: when ``old SP lo < 8`` (TEMP[8..15] not hot) the
    HI unit must NOT fire on the borrow path. The legacy bake adds
    ``ffn6.W_up[unit, BD.TEMP + lo_bit] = -S`` for lo_bit in 8..15 on
    HI units only; mirrored here via the ``borrow_blockers`` term set.
    """
    T_ent_bp = 1.5
    write_scale = 2.0 / S
    base_conditions = (
        ("CMP+2", 1.0),
        ("MARK_BP", 1.0),
    )
    borrow_blockers = tuple(
        (f"TEMP+{lo_bit}", -1.0) for lo_bit in range(8, 16)
    )

    rules: list[FFNRule] = []
    for k in range(16):
        new_k = (k - 8) % 16
        rules.append(FFNRule.gated_write(
            name=f"ent_bp_lo_{k}",
            conditions=base_conditions,
            threshold=T_ent_bp,
            gate=f"TEMP+{k}",
            writes=(
                (f"OUTPUT_LO+{new_k}", write_scale),
                (f"OUTPUT_LO+{k}", -write_scale),
            ),
        ))
    for k in range(16):
        new_k_borrow = (k - 1) % 16
        rules.append(FFNRule.gated_write(
            name=f"ent_bp_hi_{k}",
            conditions=base_conditions + borrow_blockers,
            threshold=T_ent_bp,
            gate=f"TEMP+{16 + k}",
            writes=(
                (f"OUTPUT_HI+{new_k_borrow}", write_scale),
                (f"OUTPUT_HI+{k}", -write_scale),
            ),
        ))
    return tuple(rules)


def _function_call_ent_ax_passthrough_rules(S: float) -> tuple[FFNRule, ...]:
    """ENT AX passthrough: AX_CARRY -> OUTPUT at AX marker (32 units)."""
    T = 4.0
    write_scale = 2.0 / S
    conditions = (("OP_ENT", 1.0), ("MARK_AX", 1.0))
    rules: list[FFNRule] = []
    for k in range(16):
        rules.append(FFNRule.gated_write(
            name=f"ent_ax_passthrough_lo_{k}",
            conditions=conditions,
            threshold=T,
            gate=f"AX_CARRY_LO+{k}",
            writes=((f"OUTPUT_LO+{k}", write_scale),),
        ))
    for k in range(16):
        rules.append(FFNRule.gated_write(
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
        rules.append(FFNRule.gated_write(
            name=f"lev_ax_passthrough_lo_{k}",
            conditions=conditions,
            threshold=T,
            gate=f"AX_CARRY_LO+{k}",
            writes=((f"OUTPUT_LO+{k}", write_scale),),
        ))
    for k in range(16):
        rules.append(FFNRule.gated_write(
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
        rules.append(FFNRule.gated_write(
            name=f"lev_ax_byte_lo_{k}",
            conditions=conditions,
            threshold=T_byte,
            gate=f"AX_CARRY_LO+{k}",
            writes=((f"OUTPUT_LO+{k}", write_scale),),
        ))
    for k in range(16):
        rules.append(FFNRule.gated_write(
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
      * units 2134..2165  - ENT STACK0 = old_BP (32 units)
      * units 2166..2197  - ENT BP = SP - 8 (32 units)
      * units 2198..2229  - ENT AX passthrough (32 units)
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
    rules.extend(_function_call_ent_stack0_rules(S))
    rules.extend(_function_call_ent_bp_rules(S))
    rules.extend(_function_call_ent_ax_passthrough_rules(S))
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

    return Operation(
        name="function_call_weights",
        reads=set(),
        writes=set(),
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
        reads=set(),
        writes=set(),
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
        kind="model",
        declarative_bake_fn=_bake,
        phase=999,
        migrated=True,
        declarative_authority="structural_model",
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
        reads=set(),
        writes=set(),
        kind="model",
        declarative_bake_fn=bake,
        phase=1100,
        migrated=True,
        declarative_authority="structural_model",
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
        reads=set(),
        writes=set(),
        kind="model",
        declarative_bake_fn=bake,
        phase=1160,
        migrated=True,
        declarative_authority="structural_model",
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
        reads=set(),
        writes=set(),
        kind="model",
        declarative_bake_fn=bake,
        phase=1170,
        migrated=True,
        declarative_authority="structural_model",
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
        kind="model",
        declarative_bake_fn=bake,
        phase=1200,
        migrated=True,
        declarative_authority="structural_model",
        smoke_tests=set(),
        spec_section=None,
    )


def make_expand_wrapper_blocks_op() -> Operation:
    """Split HybridALUBlock + post_ops into separate transformer blocks."""
    def bake(model, dim_positions, S):
        from ...vm_step import _expand_wrapper_blocks
        _expand_wrapper_blocks(model)

    return Operation(
        name="expand_wrapper_blocks",
        reads=set(),
        writes=set(),
        kind="model",
        declarative_bake_fn=bake,
        phase=1300,
        migrated=True,
        declarative_authority="structural_model",
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
        kind="model",
        declarative_bake_fn=_bake,
        phase=1000,
        declarative_authority="declarative",
        migrated=True,
        smoke_tests={"all"},
        spec_section="BLOG_SPEC.md#registers",
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
        kind="model",
        declarative_bake_fn=_bake,
        phase=1199,
        migrated=True,
        declarative_authority="structural_model",
        smoke_tests=set(),
        spec_section=None,
    )
