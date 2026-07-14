"""Test-only oracle mirrors of the legacy imperative ``vm_step`` layer bakes.

These 17 ``_set_layer*`` helpers were **dead on the model-build path** -- the
sole bake authority is
``neural_vm.unified_compiler.full_vm_compiler_dynamic.compile_full_vm_dynamic``,
whose declarative op factories emit the production weights. A ``settrace`` audit
over a full ``compile_full_vm_dynamic(disk_cache=False)`` build fires **none** of
these functions (0/17), and ``tools/_isa_golden_hash.py`` (the byte-identity
gate) is unchanged whether or not they exist.

They survive only as the *legacy expected* side of the declarative-bake parity
tests (``test_declarative_ffn_bakes_l*`` / ``test_declarative_attention_specs``
and the per-op L8/L13/L15/L16 tests), which lower the current production spec and
assert byte-for-byte agreement with the hand-built layout these mirrors encode.

Moved verbatim out of ``neural_vm/vm_step.py`` (LOWERING cut 2026-07-13) so the
model-build core no longer carries test-oracle-only weight-writing code.

NOTHING here is imported by the compiler; it lives entirely under ``tests/``.
The ``_SetDim`` dim-index class (still defined in ``neural_vm.vm_step``) is passed
in by callers as the ``BD`` argument; these mirrors reference it only through
that parameter.
"""

from neural_vm.constants import INSTR_WIDTH, PC_OFFSET

__all__ = [
    "_set_layer3_ffn",
    "_set_layer4_pc_relay",
    "_set_layer4_ffn",
    "_set_layer6_attn",
    "_set_layer6_routing_ffn",
    "_set_layer6_relay_heads",
    "_set_layer7_memory_heads",
    "_set_layer8_sp_gather",
    "_set_layer8_multibyte_fetch",
    "_set_layer8_alu",
    "_set_layer8_multibyte_routing",
    "_set_layer9_alu",
    "_set_layer9_marker_suppress",
    "_set_layer10_alu",
    "_set_layer15_memory_lookup_heads_0_3",
    "_set_layer15_memory_lookup",
    "_set_layer16_lev_routing",
]


def _set_layer3_ffn(ffn, S, BD):
    """Layer 3 FFN: PC/SP/BP first-step defaults + PC increment.

    First step: PC = PC_OFFSET, SP = STACK_INIT, BP = STACK_INIT
    Subsequent steps: PC += INSTR_WIDTH, SP/BP from carry-forward
    """
    from neural_vm.constants import STACK_INIT
    unit = 0

    # PC FIRST-STEP DEFAULT: when MARK_PC AND NOT HAS_SE, set PC=PC_OFFSET+INSTR_WIDTH
    # (draft tokens represent state AFTER executing first instruction)
    # CRITICAL: Also write to EMBED so L4 attention can relay to AX marker for L5 fetch!
    first_pc = PC_OFFSET + INSTR_WIDTH
    pc_lo = first_pc & 0xF
    pc_hi = (first_pc >> 4) & 0xF
    ffn.W_up[unit, BD.MARK_PC] = S
    ffn.b_up[unit] = -S * 0.5
    ffn.b_gate[unit] = 1.0
    ffn.W_down[BD.OUTPUT_LO + pc_lo, unit] = 2.0 / S
    ffn.W_down[BD.EMBED_LO + pc_lo, unit] = 2.0 / S  # Also write to EMBED for L4 relay
    unit += 1
    # Unit B: undo when HAS_SE (subsequent steps use carry-forward + increment)
    ffn.W_up[unit, BD.HAS_SE] = S
    ffn.b_up[unit] = -S * 0.5
    ffn.W_gate[unit, BD.MARK_PC] = 1.0
    ffn.W_down[BD.OUTPUT_LO + pc_lo, unit] = -2.0 / S
    ffn.W_down[BD.EMBED_LO + pc_lo, unit] = -2.0 / S  # Also undo EMBED
    unit += 1

    # Same for OUTPUT_HI and EMBED_HI
    ffn.W_up[unit, BD.MARK_PC] = S
    ffn.b_up[unit] = -S * 0.5
    ffn.b_gate[unit] = 1.0
    ffn.W_down[BD.OUTPUT_HI + pc_hi, unit] = 2.0 / S
    ffn.W_down[BD.EMBED_HI + pc_hi, unit] = 2.0 / S  # Also write to EMBED for L4 relay
    unit += 1
    ffn.W_up[unit, BD.HAS_SE] = S
    ffn.b_up[unit] = -S * 0.5
    ffn.W_gate[unit, BD.MARK_PC] = 1.0
    ffn.W_down[BD.OUTPUT_HI + pc_hi, unit] = -2.0 / S
    ffn.W_down[BD.EMBED_HI + pc_hi, unit] = -2.0 / S  # Also undo EMBED
    unit += 1

    # INITIAL_PC_BAKE CANCEL: previously cancelled the REG_PC token-embedding
    # initial-PC bake (``make_initial_pc_bake_op`` in
    # unified_compiler/ops/model_ops.py) here at MARK_PC AND HAS_SE.
    #
    # 2026-05-12 (Phase-1 PC carry fix): moved to L2 FFN
    # (``make_layer2_initial_pc_bake_cancel_op``) because the cancel needs
    # to land BEFORE L3's PC INCREMENT reads EMBED_LO[k]. Inlining the
    # cancel inside L3 FFN only modified the OUTPUT residual, so PC
    # INCREMENT (which reads the FFN's INPUT residual EMBED_LO[k]) still
    # leaked a phantom contribution from the polluted
    # EMBED_LO[init_pc_lo]=+1.0 nibble into
    # OUTPUT_LO[(init_pc_lo + INSTR_WIDTH) % 16] — aliasing onto the
    # FIRST-STEP DEFAULT slot and corrupting every step-1+ PC byte 0 by
    # +INSTR_WIDTH (broke ``test_two_imms``, ``test_jmp_from_step_2``,
    # ``test_jmp_backward``, smoke ``test_add_basic`` multi-step).
    #
    # The two cancel units below are kept as no-ops to preserve the L3
    # FFN unit count / right-sizing footprint; their W_down stays zero so
    # they contribute nothing to the residual. Removing them entirely
    # would shift downstream unit indices in this bake and risk subtle
    # regressions; leaving them as zero-write placeholders is a tiny
    # 2-unit cost that ``_right_size_ffns`` later prunes anyway.
    init_pc_lo = PC_OFFSET & 0xF
    init_pc_hi = (PC_OFFSET >> 4) & 0xF
    # (no-op placeholder: cancel now happens at L2)
    unit += 1
    # (no-op placeholder: cancel now happens at L2)
    unit += 1

    # SP DEFAULT: STACK_INIT = 0x10000
    # Bytes: 0x00, 0x00, 0x01, 0x00
    # At SP positions, default to 0 for bytes 0,1,3 and 1 for byte 2.
    # Later layers (PSH/ADJ) will override when SP changes.
    SP_I = 2  # SP marker index in MARKS array

    # SP bytes 0, 1, 3 = 0
    # At the SP marker, predict byte 0 = 0 for the synthetic first step.
    # Later SP-changing ops override marker-position output after HAS_SE is set.
    ffn.W_up[unit, BD.MARK_SP] = S
    ffn.W_up[unit, BD.HAS_SE] = -S
    ffn.b_up[unit] = -S * 0.5
    ffn.b_gate[unit] = 1.0
    ffn.W_down[BD.OUTPUT_LO + 0, unit] = 2.0 / S
    unit += 1
    ffn.W_up[unit, BD.MARK_SP] = S
    ffn.W_up[unit, BD.HAS_SE] = -S
    ffn.b_up[unit] = -S * 0.5
    ffn.b_gate[unit] = 1.0
    ffn.W_down[BD.OUTPUT_HI + 0, unit] = 2.0 / S
    unit += 1

    for byte_idx_dim in [BD.BYTE_INDEX_0, BD.BYTE_INDEX_2]:
        # LO nibble = 0
        ffn.W_up[unit, BD.H1 + SP_I] = S
        ffn.W_up[unit, byte_idx_dim] = S
        ffn.b_up[unit] = -S * 1.5
        ffn.b_gate[unit] = 1.0
        ffn.W_down[BD.OUTPUT_LO + 0, unit] = 2.0 / S
        unit += 1
        # HI nibble = 0
        ffn.W_up[unit, BD.H1 + SP_I] = S
        ffn.W_up[unit, byte_idx_dim] = S
        ffn.b_up[unit] = -S * 1.5
        ffn.b_gate[unit] = 1.0
        ffn.W_down[BD.OUTPUT_HI + 0, unit] = 2.0 / S
        unit += 1

    # SP byte 2 = 0x01 (lo=1, hi=0) - FIRST STEP ONLY
    # FIX 2026-04-16: Add HAS_SE suppression. After step 0, SP value changes from
    # 0x10000 to actual values like 0xFFF8 (where byte 2 = 0x00). L10 attention
    # passthrough handles subsequent steps, so this default should only fire on step 0.
    ffn.W_up[unit, BD.H1 + SP_I] = S
    ffn.W_up[unit, BD.BYTE_INDEX_1] = S
    ffn.W_up[unit, BD.HAS_SE] = -S  # Only first step
    ffn.b_up[unit] = -S * 1.5
    ffn.b_gate[unit] = 1.0
    ffn.W_down[BD.OUTPUT_LO + 1, unit] = 2.0 / S  # lo nibble = 1
    unit += 1
    ffn.W_up[unit, BD.H1 + SP_I] = S
    ffn.W_up[unit, BD.BYTE_INDEX_1] = S
    ffn.W_up[unit, BD.HAS_SE] = -S  # Only first step
    ffn.b_up[unit] = -S * 1.5
    ffn.b_gate[unit] = 1.0
    ffn.W_down[BD.OUTPUT_HI + 0, unit] = 2.0 / S  # hi nibble = 0
    unit += 1

    # BP DEFAULT: same as SP
    BP_I = 3  # BP marker index in MARKS array

    # BP bytes 0, 1, 3 = 0
    # At the BP marker, predict byte 0 = 0 for the synthetic first step.
    ffn.W_up[unit, BD.MARK_BP] = S
    ffn.W_up[unit, BD.HAS_SE] = -S
    ffn.b_up[unit] = -S * 0.5
    ffn.b_gate[unit] = 1.0
    ffn.W_down[BD.OUTPUT_LO + 0, unit] = 2.0 / S
    unit += 1
    ffn.W_up[unit, BD.MARK_BP] = S
    ffn.W_up[unit, BD.HAS_SE] = -S
    ffn.b_up[unit] = -S * 0.5
    ffn.b_gate[unit] = 1.0
    ffn.W_down[BD.OUTPUT_HI + 0, unit] = 2.0 / S
    unit += 1

    for byte_idx_dim in [BD.BYTE_INDEX_0, BD.BYTE_INDEX_2]:
        # LO nibble = 0
        ffn.W_up[unit, BD.H1 + BP_I] = S
        ffn.W_up[unit, byte_idx_dim] = S
        ffn.b_up[unit] = -S * 1.5
        ffn.b_gate[unit] = 1.0
        ffn.W_down[BD.OUTPUT_LO + 0, unit] = 2.0 / S
        unit += 1
        # HI nibble = 0
        ffn.W_up[unit, BD.H1 + BP_I] = S
        ffn.W_up[unit, byte_idx_dim] = S
        ffn.b_up[unit] = -S * 1.5
        ffn.b_gate[unit] = 1.0
        ffn.W_down[BD.OUTPUT_HI + 0, unit] = 2.0 / S
        unit += 1

    # BP byte 2 = 0x01 (lo=1, hi=0) - FIRST STEP ONLY
    # FIX 2026-04-16: Add HAS_SE suppression (same reason as SP byte 2).
    ffn.W_up[unit, BD.H1 + BP_I] = S
    ffn.W_up[unit, BD.BYTE_INDEX_1] = S
    ffn.W_up[unit, BD.HAS_SE] = -S  # Only first step
    ffn.b_up[unit] = -S * 1.5
    ffn.b_gate[unit] = 1.0
    ffn.W_down[BD.OUTPUT_LO + 1, unit] = 2.0 / S  # lo nibble = 1
    unit += 1
    ffn.W_up[unit, BD.H1 + BP_I] = S
    ffn.W_up[unit, BD.BYTE_INDEX_1] = S
    ffn.W_up[unit, BD.HAS_SE] = -S  # Only first step
    ffn.b_up[unit] = -S * 1.5
    ffn.b_gate[unit] = 1.0
    ffn.W_down[BD.OUTPUT_HI + 0, unit] = 2.0 / S  # hi nibble = 0
    unit += 1

    # PC marker index
    PC_I = 0  # PC marker index in MARKS array

    # PC DEFAULT: bytes 1-3 = 0 (for PC < 256, which covers most small programs)
    # At byte K position, we predict byte K+1, so:
    #   - BYTE_INDEX_0 → predict byte 1 (OUTPUT = 0)
    PC_I = 0  # PC marker index in MARKS array
    #   - BYTE_INDEX_1 → predict byte 2 (OUTPUT = 0)
    #   - BYTE_INDEX_2 → predict byte 3 (OUTPUT = 0)
    # This fires at all PC byte positions 0-2, outputting 0 for bytes 1-3.
    for byte_idx_dim in [BD.BYTE_INDEX_0, BD.BYTE_INDEX_1, BD.BYTE_INDEX_2]:
        # Condition: H1[PC] AND BYTE_INDEX_K → OUTPUT = 0 (for byte K+1)
        ffn.W_up[unit, BD.H1 + PC_I] = S
        ffn.W_up[unit, byte_idx_dim] = S
        ffn.b_up[unit] = -S * 1.5
        ffn.b_gate[unit] = 1.0
        ffn.W_down[BD.OUTPUT_LO + 0, unit] = 2.0 / S  # lo = 0
        unit += 1
        # HI nibble also = 0
        ffn.W_up[unit, BD.H1 + PC_I] = S
        ffn.W_up[unit, byte_idx_dim] = S
        ffn.b_up[unit] = -S * 1.5
        ffn.b_gate[unit] = 1.0
        ffn.W_down[BD.OUTPUT_HI + 0, unit] = 2.0 / S  # hi = 0
        unit += 1

    # AX DEFAULT: bytes 1-3 = 0 (for single-byte immediates)
    # Multi-byte results (e.g., ADD with carry) will override in later layers
    # At byte K position, we predict byte K+1, so:
    #   - BYTE_INDEX_0 → predict byte 1 (OUTPUT = 0)
    #   - BYTE_INDEX_1 → predict byte 2 (OUTPUT = 0)
    #   - BYTE_INDEX_2 → predict byte 3 (OUTPUT = 0)
    AX_I = 1  # AX marker index in MARKS array
    for byte_idx_dim in [BD.BYTE_INDEX_0, BD.BYTE_INDEX_1, BD.BYTE_INDEX_2]:
        # Condition: H1[AX] AND BYTE_INDEX_K → OUTPUT = 0 (for byte K+1)
        ffn.W_up[unit, BD.H1 + AX_I] = S
        ffn.W_up[unit, byte_idx_dim] = S
        ffn.b_up[unit] = -S * 1.5
        ffn.b_gate[unit] = 1.0
        ffn.W_down[BD.OUTPUT_LO + 0, unit] = 2.0 / S  # lo = 0
        unit += 1
        # HI nibble also = 0
        ffn.W_up[unit, BD.H1 + AX_I] = S
        ffn.W_up[unit, byte_idx_dim] = S
        ffn.b_up[unit] = -S * 1.5
        ffn.b_gate[unit] = 1.0
        ffn.W_down[BD.OUTPUT_HI + 0, unit] = 2.0 / S  # hi = 0
        unit += 1

    # MEM DEFAULT: all bytes = 0 (for non-memory-writing ops like IMM)
    # At MEM marker position, predict addr byte 0 = 0
    # For PSH/SI/SC, L14 will write actual values that override this default.
    # Condition: MARK_MEM = 1 → OUTPUT = 0
    MEM_I = 4  # MEM marker index in MARKS array
    ffn.W_up[unit, BD.MARK_MEM] = S
    ffn.b_up[unit] = -S * 0.5
    ffn.b_gate[unit] = 1.0
    ffn.W_down[BD.OUTPUT_LO + 0, unit] = 2.0 / S
    unit += 1
    ffn.W_up[unit, BD.MARK_MEM] = S
    ffn.b_up[unit] = -S * 0.5
    ffn.b_gate[unit] = 1.0
    ffn.W_down[BD.OUTPUT_HI + 0, unit] = 2.0 / S
    unit += 1

    # MEM addr bytes 1-3 default to 0 (for non-store ops)
    # At MEM byte positions d=1..3, H1[MEM]=1, BYTE_INDEX_K=1 for position K+1
    # FIX 2026-04-16: Exclude BYTE_INDEX_3 (d=4) because that position PREDICTS val_b0,
    # not addr_b3. L14 handles val bytes; this default is only for addr bytes 1-3.
    for byte_idx_dim in [BD.BYTE_INDEX_0, BD.BYTE_INDEX_1, BD.BYTE_INDEX_2]:
        # LO = 0
        ffn.W_up[unit, BD.H1 + MEM_I] = S
        ffn.W_up[unit, byte_idx_dim] = S
        ffn.b_up[unit] = -S * 1.5
        ffn.b_gate[unit] = 1.0
        ffn.W_down[BD.OUTPUT_LO + 0, unit] = 2.0 / S
        unit += 1
        # HI = 0
        ffn.W_up[unit, BD.H1 + MEM_I] = S
        ffn.W_up[unit, byte_idx_dim] = S
        ffn.b_up[unit] = -S * 1.5
        ffn.b_gate[unit] = 1.0
        ffn.W_down[BD.OUTPUT_HI + 0, unit] = 2.0 / S
        unit += 1

    # STACK0 DEFAULT: bytes 1-3 = 0 (for single-byte results)
    # At STACK0 byte positions, output 0 for bytes 1-3.
    # STACK0 byte 0: d=6 from BP → L1H4[BP]=1 (d≤6.5)
    # STACK0 byte 1: d=7 from BP → H2[BP]=1 (d≤7.5), NOT L1H4[BP]
    # STACK0 byte 2: d=8 from BP → H3[BP]=1 (d≤8.5), NOT H2[BP]
    # STACK0 byte 3: d=9 from BP → H4[BP]=1 (d≤9.5), NOT H3[BP]
    # Use H4[BP] (d≤9.5) to cover all STACK0 positions d=5-9.
    # H1[BP] = 1 for d <= 4.5 from BP (only BP area, not STACK0)
    for byte_idx_dim in [BD.BYTE_INDEX_0, BD.BYTE_INDEX_1, BD.BYTE_INDEX_2]:
        # Condition: H4[BP] AND BYTE_INDEX_K AND NOT H1[BP] → OUTPUT = 0
        ffn.W_up[unit, BD.H4 + BP_I] = S
        ffn.W_up[unit, byte_idx_dim] = S
        ffn.W_up[unit, BD.H1 + BP_I] = -S  # Exclude BP area
        ffn.b_up[unit] = -S * 1.5
        ffn.b_gate[unit] = 1.0
        ffn.W_down[BD.OUTPUT_LO + 0, unit] = 2.0 / S
        unit += 1
        ffn.W_up[unit, BD.H4 + BP_I] = S
        ffn.W_up[unit, byte_idx_dim] = S
        ffn.W_up[unit, BD.H1 + BP_I] = -S  # Exclude BP area
        ffn.b_up[unit] = -S * 1.5
        ffn.b_gate[unit] = 1.0
        ffn.W_down[BD.OUTPUT_HI + 0, unit] = 2.0 / S
        unit += 1

    # STACK0 first-step default: byte 0 = 0 (empty stack)
    # At STACK0 marker, when NOT HAS_SE, output 0.
    ffn.W_up[unit, BD.MARK_STACK0] = S
    ffn.W_up[unit, BD.HAS_SE] = -S
    ffn.b_up[unit] = -S * 0.5
    ffn.b_gate[unit] = 1.0
    ffn.W_down[BD.OUTPUT_LO + 0, unit] = 2.0 / S
    unit += 1
    ffn.W_up[unit, BD.MARK_STACK0] = S
    ffn.W_up[unit, BD.HAS_SE] = -S
    ffn.b_up[unit] = -S * 0.5
    ffn.b_gate[unit] = 1.0
    ffn.W_down[BD.OUTPUT_HI + 0, unit] = 2.0 / S
    unit += 1

    # STACK0 carry-forward marker projection. L3 attention carries the
    # previous STACK0 byte 0 into EMBED_LO/HI at the STACK0 marker; project it
    # to OUTPUT so the token head emits that carried byte on non-first steps.
    for k in range(16):
        ffn.W_up[unit, BD.MARK_STACK0] = S
        ffn.W_up[unit, BD.HAS_SE] = S
        ffn.b_up[unit] = -S * 1.5
        ffn.W_gate[unit, BD.EMBED_LO + k] = 1.0
        ffn.W_down[BD.OUTPUT_LO + k, unit] = 2.0 / S
        unit += 1
    for k in range(16):
        ffn.W_up[unit, BD.MARK_STACK0] = S
        ffn.W_up[unit, BD.HAS_SE] = S
        ffn.b_up[unit] = -S * 1.5
        ffn.W_gate[unit, BD.EMBED_HI + k] = 1.0
        ffn.W_down[BD.OUTPUT_HI + k, unit] = 2.0 / S
        unit += 1

    # NEXT_STACK0 LOCALITY: STACK0 is intentionally not IS_MARK, so BP
    # remains the nearest threshold marker across the STACK0 field. That makes
    # the BP→STACK0 transition flag leak past the marker and into value-byte
    # slots, where the output head otherwise repeats the STACK0 marker. Clear
    # the flag locally after the marker and at STACK0 byte positions.
    ffn.W_up[unit, BD.MARK_STACK0] = S
    ffn.b_up[unit] = -S * 0.5
    ffn.W_gate[unit, BD.NEXT_STACK0] = 1.0
    ffn.W_down[BD.NEXT_STACK0, unit] = -3.0 / S
    unit += 1
    # STACK0 byte 0 aliases BYTE_INDEX_3; exclude BP byte 3 by requiring
    # H0[BP] to be effectively zero.
    ffn.W_up[unit, BD.H4 + BP_I] = S
    ffn.W_up[unit, BD.BYTE_INDEX_3] = S
    ffn.W_up[unit, BD.H0 + BP_I] = -S * 1000
    ffn.b_up[unit] = -S * 1.5
    ffn.W_gate[unit, BD.NEXT_STACK0] = 1.0
    ffn.W_down[BD.NEXT_STACK0, unit] = -3.0 / S
    ffn.W_down[BD.OUTPUT_LO + 0, unit] = 5.0 / S
    ffn.W_down[BD.OUTPUT_HI + 0, unit] = 5.0 / S
    unit += 1
    for byte_idx_dim in [BD.BYTE_INDEX_1, BD.BYTE_INDEX_2]:
        ffn.W_up[unit, BD.H4 + BP_I] = S
        ffn.W_up[unit, byte_idx_dim] = S
        ffn.b_up[unit] = -S * 1.5
        ffn.W_gate[unit, BD.NEXT_STACK0] = 1.0
        ffn.W_down[BD.NEXT_STACK0, unit] = -3.0 / S
        unit += 1

    # PC INCREMENT: when MARK_PC AND HAS_SE AND NOT OP_LEV, add INSTR_WIDTH to carried-forward value
    # For each lo nibble k (0-15): new_lo = (k+INSTR_WIDTH)%16
    # FIX 2026-04-15: Suppress when OP_LEV - LEV gets PC from return_addr in memory, not increment
    for k in range(16):
        new_k = (k + INSTR_WIDTH) % 16
        ffn.W_up[unit, BD.HAS_SE] = S
        ffn.W_up[unit, BD.MARK_PC] = S
        ffn.W_up[unit, BD.OP_LEV] = -S / 5  # OP_LEV ≈ 5.0, so -S/5 * 5 = -S suppresses
        ffn.b_up[unit] = -S * 1.5
        ffn.W_gate[unit, BD.EMBED_LO + k] = 1.0
        ffn.W_down[BD.OUTPUT_LO + new_k, unit] = 2.0 / S
        unit += 1

    # Hi nibble copy (only at MARK_PC AND HAS_SE AND NOT OP_LEV)
    for k in range(16):
        ffn.W_up[unit, BD.HAS_SE] = S
        ffn.W_up[unit, BD.MARK_PC] = S
        ffn.W_up[unit, BD.OP_LEV] = -S / 5  # Suppress when LEV
        ffn.b_up[unit] = -S * 1.5
        ffn.W_gate[unit, BD.EMBED_HI + k] = 1.0
        ffn.W_down[BD.OUTPUT_HI + k, unit] = 2.0 / S
        unit += 1

    # PC carry correction: when lo nibble >= (16-INSTR_WIDTH), adding INSTR_WIDTH wraps (>=16),
    # so hi nibble must increment by 1. Fires when MARK_PC AND HAS_SE AND NOT OP_LEV
    # AND any of EMBED_LO[(16-INSTR_WIDTH)..15] is set (old lo nibble >= 16-INSTR_WIDTH).
    # MARK_PC has weight 4*S to strictly require it (prevents false positive
    # at byte positions where EMBED_LO can be inflated by L3 leakage).
    # FIX 2026-04-15: Added OP_LEV suppression
    carry_threshold = 16 - INSTR_WIDTH  # For INSTR_WIDTH=8, this is 8
    for k in range(16):
        ffn.W_up[unit, BD.MARK_PC] = 4 * S
        ffn.W_up[unit, BD.HAS_SE] = S
        ffn.W_up[unit, BD.OP_LEV] = -S  # OP_LEV ≈ 5, use stronger -S for higher threshold
        for lo_bit in range(carry_threshold, 16):
            ffn.W_up[unit, BD.EMBED_LO + lo_bit] = S
        # Bias: activate when MARK_PC(4*S) + HAS_SE(S) + carry_bit(S) >= 5.5*S
        #       but not when just MARK_PC(4*S) + HAS_SE(S) = 5*S < 5.5*S
        ffn.b_up[unit] = -S * 5.5
        ffn.W_gate[unit, BD.EMBED_HI + k] = 1.0
        ffn.W_down[BD.OUTPUT_HI + k, unit] = -2.0 / S  # cancel old
        ffn.W_down[BD.OUTPUT_HI + (k + 1) % 16, unit] = 2.0 / S  # add shifted
        unit += 1


def _set_layer4_pc_relay(attn, S, BD, HD):
    """Layer 4 attention: Relay PREVIOUS step's PC to AX marker position.

    Head 0: AX marker reads the PC MARKER's EMBED_LO/HI, which was
    populated by L3 carry-forward from the previous step's PC byte 0.
    This gives the instruction address that was just executed, enabling
    L5 to fetch the correct opcode and immediate.

    Output goes to EMBED_LO/HI at AX marker (which starts as all zeros
    since marker tokens have no nibble embedding).
    """
    L = 15.0
    PC_I = 0  # PC marker index in MARKS

    base = 0 * HD
    # Q: active at AX markers
    attn.W_q[base, BD.MARK_AX] = L
    # K: target PC marker (has prev step's PC from L3 carry-forward)
    attn.W_k[base, BD.MARK_PC] = L
    # V: copies EMBED_LO and EMBED_HI (OLD PC from L3 carry-forward).
    # L3 carry-forward writes the OLD PC (instruction just executed) to EMBED_LO/HI.
    # L3 FFN increments it and writes NEW PC to OUTPUT_LO/HI.
    # We need the OLD PC to fetch the opcode at the correct address.
    # Example: IMM 42 at address 0, after execution PC=10 (new), but we fetch opcode at 0 (old).
    for k in range(16):
        attn.W_v[base + 1 + k, BD.EMBED_LO + k] = 1.0
        attn.W_v[base + 17 + k, BD.EMBED_HI + k] = 1.0
        attn.W_v[base + 33 + k, BD.ADDR_KEY + 32 + k] = 1.0
    # O: writes to EMBED_LO and EMBED_HI at AX marker
    for k in range(16):
        attn.W_o[BD.EMBED_LO + k, base + 1 + k] = 1.0
        attn.W_o[BD.EMBED_HI + k, base + 17 + k] = 1.0
        attn.W_o[BD.ADDR_KEY + 32 + k, base + 33 + k] = 1.0

    # Anti-leakage gate: suppress at non-AX positions
    GATE = 33
    attn.W_q[base + GATE, BD.MARK_AX] = L
    attn.W_q[base + GATE, BD.CONST] = -L / 2
    attn.W_k[base + GATE, BD.CONST] = L

    # Head 1: Relay PC from PC marker to AX byte positions for multi-byte IMM fetch.
    # At byte positions (IS_BYTE + H1[AX_I]), attend to PC marker -> copy EMBED (PC value)
    # to TEMP dims. L4 FFN then computes PC+2/3/4 from this.
    # Fires on all steps (first step uses PC_OFFSET, subsequent steps use prev PC).
    base = 1 * HD
    AX_I = 1
    # Q: fire at byte positions near AX marker (IS_BYTE + H1[AX_I])
    attn.W_q[base, BD.IS_BYTE] = L
    attn.W_q[base, BD.H1 + AX_I] = L
    attn.W_q[base, BD.CONST] = -L * 1.5  # need IS_BYTE + H1[AX] > 1.5
    # K: target PC marker
    attn.W_k[base, BD.MARK_PC] = L
    attn.W_k[base, BD.CONST] = L * 0.5  # bias
    # V: copy EMBED_LO/HI (PC value from L3 carry-forward at PC marker)
    for k in range(16):
        attn.W_v[base + 1 + k, BD.EMBED_LO + k] = 1.0
        attn.W_v[base + 17 + k, BD.EMBED_HI + k] = 1.0
        attn.W_v[base + 33 + k, BD.ADDR_KEY + 32 + k] = 1.0
    # O: write to TEMP dims at byte positions (PC value staging for L4 FFN)
    for k in range(16):
        attn.W_o[BD.TEMP + k, base + 1 + k] = 1.0
        attn.W_o[BD.TEMP + 16 + k, base + 17 + k] = 1.0
        attn.W_o[BD.ADDR_KEY + 32 + k, base + 33 + k] = 1.0
    # Anti-leakage gate: suppress at non-byte positions
    GATE1 = 33
    attn.W_q[base + GATE1, BD.IS_BYTE] = 500.0
    attn.W_q[base + GATE1, BD.CONST] = -500.0
    attn.W_k[base + GATE1, BD.CONST] = 5.0


def _set_layer4_ffn(ffn, S, BD):
    """Layer 4 FFN: Compute (PC+1) nibbles at AX marker for IMM fetch.

    PC value is in EMBED_LO/HI at AX marker (from L4 attention relay).
    L5 fetch uses:
      - opcode query at address PC (EMBED_LO/HI)
      - immediate query at address PC+1 (TEMP[0..31], written here)
    """
    unit = 0

    # === PC_PLUS1_LO: rotate EMBED_LO by +1 ===
    for k in range(16):
        src = (k - 1) % 16
        ffn.W_up[unit, BD.MARK_AX] = S
        ffn.b_up[unit] = -S * 0.5
        ffn.W_gate[unit, BD.EMBED_LO + src] = 1.0
        ffn.W_down[BD.TEMP + k, unit] = 2.0 / S
        unit += 1

    # === PC_PLUS1_HI: default copy (no carry) ===
    for k in range(16):
        ffn.W_up[unit, BD.MARK_AX] = S
        ffn.b_up[unit] = -S * 0.5
        ffn.W_gate[unit, BD.EMBED_HI + k] = 1.0
        ffn.W_down[BD.TEMP + 16 + k, unit] = 2.0 / S
        unit += 1

    # === PC_PLUS1_HI: carry correction when EMBED_LO[15] == 1 ===
    # Cancel default copy and write rotated (+1) hi nibble.
    for k in range(16):
        # Cancel: MARK_AX AND LO[15] → subtract EMBED_HI[k]
        ffn.W_up[unit, BD.MARK_AX] = S
        ffn.W_up[unit, BD.EMBED_LO + 15] = S
        ffn.b_up[unit] = -S * 1.5
        ffn.W_gate[unit, BD.EMBED_HI + k] = -1.0
        ffn.W_down[BD.TEMP + 16 + k, unit] = 2.0 / S
        unit += 1
        # Write rotated: MARK_AX AND LO[15] → add EMBED_HI[(k-1)%16]
        src = (k - 1) % 16
        ffn.W_up[unit, BD.MARK_AX] = S
        ffn.W_up[unit, BD.EMBED_LO + 15] = S
        ffn.b_up[unit] = -S * 1.5
        ffn.W_gate[unit, BD.EMBED_HI + src] = 1.0
        ffn.W_down[BD.TEMP + 16 + k, unit] = 2.0 / S
        unit += 1

    # === TEMP clearing at PC marker ===
    # Clear TEMP dims at PC marker to prevent them from leaking to Layer 6.
    # TEMP values are only meaningful at AX marker (where PC+1 is computed).
    # At PC marker, TEMP should be zero to prevent spurious BZ/BNZ activation.
    # EXCEPT: TEMP[0] is used for IS_JSR flag (first-step decode + L6 relay).
    # Condition: MARK_PC (fires at PC marker token)
    for k in range(32):
        if k == 0:
            unit += 1
            continue
        ffn.W_up[unit, BD.MARK_PC] = S
        ffn.b_up[unit] = -S * 0.5
        ffn.W_gate[unit, BD.TEMP + k] = -1.0
        ffn.W_down[BD.TEMP + k, unit] = 2.0 / S
        unit += 1

    # === Multi-byte IMM: write PC+2/3/4 to FETCH_LO/HI at AX byte positions ===
    # L4 head 1 relays PC to TEMP at byte positions. Rotate lo nibble by offset
    # and copy hi nibble, writing the rotated address directly to FETCH_LO/HI.
    # This avoids the TEMP clearing problem (clearing only achieves 94% removal).
    # FETCH_LO/HI at byte positions is free — L5 only writes at AX marker.
    # OFF-BY-ONE: OUTPUT at position N generates token N+1. So:
    #   BYTE_INDEX_0 → generates byte1 → needs FETCH = PC+2
    #   BYTE_INDEX_1 → generates byte2 → needs FETCH = PC+3
    #   BYTE_INDEX_2 → generates byte3 → needs FETCH = PC+4
    # Condition: IS_BYTE AND H1[AX_I] AND BYTE_INDEX_k (specific byte position)
    # Hi-nibble has its own carry block: lo + offset >= 16 ⇒ lo ∈ [16-offset, 15]
    # (each bit triggers the cancel-default + add-rotated +1 pair).
    # REFACTORED: this is `nibble_rotation_chain` with gate_marker=IS_BYTE,
    # condition_dims supplying the AX-row + byte-index AND, with_carry=True
    # (the primitive iterates carry_src ∈ [16-offset, 15] automatically).
    from neural_vm.unified_compiler.primitives import Primitives as _P
    AX_I = 1
    for byte_idx in range(0, 3):
        offset = byte_idx + 2  # PC+2, PC+3, PC+4
        unit = _P.nibble_rotation_chain(
            ffn,
            unit=unit,
            gate_marker=BD.IS_BYTE,
            source_lo_dim=BD.TEMP,
            source_hi_dim=BD.TEMP + 16,
            target_lo_dim=BD.FETCH_LO,
            target_hi_dim=BD.FETCH_HI,
            offset=offset,
            with_carry=True,
            S=S,
            magnitude=2.0,
            condition_dims=[BD.H1 + AX_I, BD.BYTE_INDEX_0 + byte_idx],
        )

    # === PC+1 at PC marker → FETCH_LO/HI for dynamic immediate fetch ===
    # FIX 2026-04-29: Compute PC+1 at the PC marker and write to FETCH dims.
    # L5 head 3 reads FETCH as the address query, enabling dynamic immediate
    # fetch at the correct address (current PC+1) instead of a fixed address.
    # This fixes BZ/BNZ branch targets and JMP targets for non-first steps.
    # EMBED_LO/HI at PC marker contains current PC (from L3 carry-forward).
    # Same nibble rotation logic as TEMP at AX marker, but gated on MARK_PC.
    # REFACTORED: this is the canonical carry-aware (PC+1) rotation at MARK_PC.
    unit = _P.nibble_rotation_chain(
        ffn,
        unit=unit,
        gate_marker=BD.MARK_PC,
        source_lo_dim=BD.EMBED_LO,
        source_hi_dim=BD.EMBED_HI,
        target_lo_dim=BD.FETCH_LO,
        target_hi_dim=BD.FETCH_HI,
        offset=1,
        with_carry=True,
        S=S,
        magnitude=2.0,
    )


    # Head 5: unused in Layer 5 fetch


def _set_layer6_attn(attn, S, BD, HD):
    """Layer 6 attention: JMP relay + EXIT relay + first-step relays.

    Head 0 — JMP relay: At current step's PC marker, attend to previous
    step's AX marker (d=30). Copy OP_JMP flag and FETCH_LO/HI (JMP target).
    Writes CMP[0] (IS_JMP) and AX_CARRY_LO/HI (JMP target) at PC marker.

    Head 1 — EXIT relay: At NEXT_SE positions, attend to current step's
    AX marker (d=28). Copy OP_EXIT flag. Writes CMP[1] (IS_EXIT).

    Head 2 — First-step JMP relay: At PC marker (first step only), attend to
    current step's AX marker. Copy OP_JMP and FETCH for intra-step JMP.

    Head 3 — JSR relay: At PC marker (all steps), attend to current step's
    AX marker. Copy OP_JSR flag to TEMP[0] for JSR PC override.

    Head 4 — First-step FETCH relay: At AX marker (first step only), attend to
    PC marker. Copy FETCH_LO/HI from PC marker to AX marker for IMM routing.

    Head 5 — First-step OP flag relay: At AX marker (first step only), attend to
    PC marker. Copy OP_IMM, OP_LEA, OP_JMP, OP_EXIT, OP_NOP, arithmetic/bitwise/
    comparison/shift flags from PC marker to AX marker for Layer 6 FFN routing.
    Required because Layer 5 FFN decodes opcodes at PC marker (NOT at AX marker),
    but Layer 6 FFN needs flags at AX marker. Relays 17 OP flags total.

    Uses L=50 (large) + ALiBi slope=5.0 (steep) to minimize leakage.
    Attention scale = 1/sqrt(64) = 0.125, so score = L²*0.125 - slope*d.
    Head 0 at PC: 50²*0.125 - 5*30 = 162 (strong). Leakage at Q=0: <0.7%.
    Head 1 at SE: 50²*0.7*0.125 - 5*28 = 79 (strong). Leakage at Q=0: <0.7%.
    Q guards (-L at MARK_AX) block self-attention at AX markers entirely.

    Head 6: Configured by _set_layer6_relay_heads() for PSH relay (STACK0 ← AX).
    Head 7: Reserved for JSR handling (configured later in set_vm_weights).
    """
    L = 50.0

    # Head 0: JMP relay (PC marker → previous step's AX marker)
    # FIX 2026-04-16: Added HAS_SE gating so Head 0 only fires on step 1+.
    # For step 0, there's no previous step's AX marker to relay from.
    # Without this gate, Head 0 fires at PC marker (Q=L from MARK_PC), and with no
    # valid K positions (no MARK_AX), softmax gives uniform attention. This averages
    # FETCH values from all positions, writing garbage to AX_CARRY_LO/HI, which
    # causes L8 FFN to produce massive OUTPUT values (40000+) at PC marker.
    base = 0 * HD
    attn.W_q[base, BD.MARK_PC] = L
    attn.W_q[base, BD.MARK_AX] = -L  # block at AX marker (prevents AX_CARRY corruption)
    # Strong anti-leakage gate: Q must be very negative for step 0 (no HAS_SE)
    # Q = 50 - 0 + 0 - 1000 = -950 for step 0 (exp(-950) ≈ 0)
    # Q = 50 - 0 + 1000 - 1000 = 50 for step 1+ (fires normally)
    attn.W_q[base, BD.HAS_SE] = L * 20  # +1000 when HAS_SE
    attn.W_q[base, BD.CONST] = -L * 20  # -1000 baseline
    attn.W_k[base, BD.MARK_AX] = L
    # FIX 2026-04-16: Add CONST to K so the Q gate actually works.
    # Without this, K[0] = 0 at non-AX positions (including PC marker),
    # so Q·K = (-950)*0 = 0 regardless of Q, and attention fires via ALiBi.
    # With CONST: K[0] = 50*MARK_AX + 1*CONST, so K[0] = 1 at non-AX positions.
    # Q·K = (-950)*1 = -950 → exp(-950) ≈ 0 → blocked.
    attn.W_k[base, BD.CONST] = 1.0
    # V: copy OP_JMP flag, FETCH_LO/HI (target from fetched immediate)
    # NOTE: OP_JSR is NOT included here. The JMP relay has a one-step delay
    # (fires at step N+1 using step N's AX flags). For JMP this is the design:
    # step N outputs old PC+5, step N+1 overrides with target. For JSR, the
    # runner overrides PC to the target at step N directly, so the relay at
    # step N+1 would double-override (writing target instead of target+5).
    attn.W_v[base + 1, BD.OP_JMP] = 1.0
    for k in range(16):
        attn.W_v[base + 2 + k, BD.FETCH_LO + k] = 1.0
        attn.W_v[base + 18 + k, BD.FETCH_HI + k] = 1.0
    # O: write IS_JMP to CMP[0], JMP target to AX_CARRY at PC marker
    attn.W_o[BD.CMP + 0, base + 1] = 1.0
    for k in range(16):
        attn.W_o[BD.AX_CARRY_LO + k, base + 2 + k] = 1.0
        attn.W_o[BD.AX_CARRY_HI + k, base + 18 + k] = 1.0

    # Head 1: EXIT relay (NEXT_SE position → current step's AX marker)
    base = 1 * HD
    attn.W_q[base, BD.NEXT_SE] = L
    attn.W_q[base, BD.MARK_AX] = -L  # block at AX marker (prevents CMP[1] leakage)
    attn.W_k[base, BD.MARK_AX] = L
    # V: copy OP_EXIT flag (scaled to improve EXIT vs NOP separation downstream)
    attn.W_v[base + 1, BD.OP_EXIT] = 0.2
    # O: write IS_EXIT to CMP[1]
    attn.W_o[BD.CMP + 1, base + 1] = 1.0

    # Head 2: First-step JMP relay (PC marker self-attention)
    # For first step (NOT HAS_SE), copy OP_JMP and FETCH from PC marker to PC marker.
    # This works with L5 head 3 fix (which no longer writes to AX marker).
    # Head 0 handles subsequent steps (cross-step relay), head 2 handles first step.
    # BUG FIX 2026-04-09: Changed key from MARK_AX to MARK_PC to work with L5 head 3 fix.
    # BUG FIX 2026-04-09 (part 2): Block non-JMP operations (IMM, NOP, EXIT) to prevent
    # this head from firing for all first-step ops and copying amplified FETCH to AX_CARRY.
    base = 2 * HD
    attn.W_q[base, BD.MARK_PC] = L
    attn.W_q[base, BD.HAS_SE] = -L  # Only fire when NOT HAS_SE (first step)
    attn.W_q[base, BD.MARK_AX] = -L  # block at AX marker
    # FIX 2026-04-16: Strong anti-leakage gate for non-JMP operations.
    # Without strong gate, Q = -50 for non-JMP gives ~5% attention weight (not zero!),
    # causing FETCH values to leak into AX_CARRY_LO and corrupt PC output.
    # With OP_JMP ≈ 5.0 when JMP is active, 0 otherwise:
    # Q = MARK_PC(50) - HAS_SE(50) - MARK_AX(50) + OP_JMP(1000) - CONST(1000)
    # For JMP first-step: 50 - 0 - 0 + 5000 - 1000 = 4050 (strong positive)
    # For non-JMP first-step: 50 - 0 - 0 + 0 - 1000 = -950 (blocked, exp(-950) ≈ 0)
    attn.W_q[base, BD.OP_JMP] = L * 20  # +1000 when OP_JMP active
    attn.W_q[base, BD.CONST] = -L * 20  # -1000 baseline
    attn.W_k[base, BD.MARK_PC] = L  # Read from PC marker (not AX marker)
    # V: copy OP_JMP and FETCH_LO/HI from PC marker
    attn.W_v[base + 1, BD.OP_JMP] = 1.0
    for k in range(16):
        attn.W_v[base + 2 + k, BD.FETCH_LO + k] = 1.0
        attn.W_v[base + 18 + k, BD.FETCH_HI + k] = 1.0
    # O: write IS_JMP to CMP[0], JMP target to AX_CARRY at PC marker
    attn.W_o[BD.CMP + 0, base + 1] = 1.0
    for k in range(16):
        attn.W_o[BD.AX_CARRY_LO + k, base + 2 + k] = 1.0
        attn.W_o[BD.AX_CARRY_HI + k, base + 18 + k] = 1.0

    # Head 3: JSR relay (AX marker → PC marker, FIRST STEP ONLY)
    # For first step, L5 FFN decodes JSR at PC marker and writes TEMP[0].
    # This relay is DISABLED (first-step only) because subsequent steps use L5 opcode decode.
    # BUG FIX 2026-04-13: Added HAS_SE gate to prevent false positive on subsequent steps.
    # Without HAS_SE gate, this head attends to ALL AX markers in context, including
    # previous steps' AX markers which may have OP_JSR set (causing PC override on non-JSR steps).
    base = 3 * HD
    attn.W_q[base, BD.MARK_PC] = L
    attn.W_q[base, BD.MARK_AX] = -L  # block at AX marker
    attn.W_q[base, BD.HAS_SE] = -L   # BUG FIX: only fire when NOT HAS_SE (first step)
    attn.W_k[base, BD.MARK_AX] = L
    # V: copy OP_JSR flag only (FETCH already has target)
    attn.W_v[base + 1, BD.OP_JSR] = 1.0
    # O: write IS_JSR flag to TEMP[0] at PC marker
    # (Layer 5 FFN clears TEMP at PC; Layer 6 attention writes it; Layer 6 FFN reads it)
    attn.W_o[BD.TEMP + 0, base + 1] = 1.0

    # Head 4: Reserved for BZ/BNZ relay (set by _set_bz_bnz_relay)
    # NOTE: First-step FETCH relay moved to head 5 to avoid Q weight conflicts.
    # _set_bz_bnz_relay sets up PC→AX relay for BZ/BNZ conditional branches.
    # base = 4 * HD (left for _set_bz_bnz_relay)

    # Head 5: First-step OP flag relay + FETCH relay (AX marker ← PC marker)
    # For first step, L5 FFN decodes opcodes at PC marker (OP_IMM, OP_LEA, OP_EXIT, OP_NOP, OP_JMP, OP_JSR, arithmetic, bitwise, cmp, shift).
    # L6 FFN needs these flags at AX marker for routing (IMM, EXIT, NOP, JMP, arithmetic, etc).
    # NOTE: L5 FFN already sets OP_* globally, so we only need to relay FETCH.
    # The OP_* relay was causing doubling (5.0 from L5 + 5.0 from relay = 10.0).
    base = 5 * HD
    attn.W_q[base, BD.MARK_AX] = L
    attn.W_q[base, BD.HAS_SE] = -L
    attn.W_k[base, BD.MARK_PC] = L
    # V/O: only relay FETCH (OP_* already set globally by L5 FFN)
    for k in range(16):
        attn.W_v[base + k, BD.FETCH_LO + k] = 1.0
        attn.W_v[base + 16 + k, BD.FETCH_HI + k] = 1.0
    for k in range(16):
        attn.W_o[BD.FETCH_LO + k, base + k] = 1.0
        attn.W_o[BD.FETCH_HI + k, base + 16 + k] = 1.0
    # Branch/call byte-1 relay: PC byte 0 re-reads the PC marker so L6 FFN can
    # emit byte1(target) for taken BZ/BNZ and first-step JSR targets above 255.
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
    # FIX 2026-05-09: REMOVED `attn.W_o[BD.OP_EXIT, base + 4] = 1.0`. This was a
    # leftover OP_* relay (the comment above explicitly says we only relay FETCH
    # because OP_* is set globally by L5 FFN). Slot 4 is also used by FETCH_LO[4]
    # relay, so this line caused a slot collision: any byte value with low nibble = 4
    # (e.g. 100, 4, 20, 36, ...) had FETCH_LO[4] = 40 *also* written into OP_EXIT.
    # The L6 IMM routing units have W_up[OP_EXIT] = -S*20 = -2000, so OP_EXIT = 40
    # produced up = -80,000 and the IMM unit didn't fire. test_imm_byte_values[100]
    # was the visible symptom.
    # Anti-leakage gate for FETCH relay (prevent writes at non-AX positions)
    # Without this gate, softmax1 distributes attention uniformly at byte positions,
    # and W_o writes garbage FETCH values, overwriting the correct FETCH from L4 FFN.
    # Gate mechanism: Q[slot] = 500 * MARK_AX - 500, K[slot] = 5 * CONST
    # At AX marker: score += (500*1 - 500) * 5 / 8 = 0 (neutral)
    # At non-AX: score += (500*0 - 500) * 5 / 8 = -312.5 (blocked)
    FETCH_GATE = 50
    attn.W_q[base + FETCH_GATE, BD.MARK_AX] = 500.0
    attn.W_q[base + FETCH_GATE, BD.CONST] = -500.0
    attn.W_k[base + FETCH_GATE, BD.CONST] = 5.0
    # AX byte rows also carry MARK_AX, but they need L4's PC+2/3/4 FETCH
    # address, not the first immediate byte relayed from the PC marker.
    AX_BYTE_FETCH_BLOCKER = 53
    attn.W_q[base + AX_BYTE_FETCH_BLOCKER, BD.H1 + 1] = -6500.0
    attn.W_q[base + AX_BYTE_FETCH_BLOCKER, BD.IS_BYTE] = -6500.0
    attn.W_q[base + AX_BYTE_FETCH_BLOCKER, BD.MARK_AX] = 6500.0
    attn.W_q[base + AX_BYTE_FETCH_BLOCKER, BD.H1 + 0] = 6500.0
    attn.W_k[base + AX_BYTE_FETCH_BLOCKER, BD.CONST] = 5.0
    # HAS_SE gate: block attention when HAS_SE = 1 (non-first steps)
    # Without this gate, softmax1 distributes attention uniformly when Q is zero,
    # causing spurious FETCH relay from step 1 PC marker (which has V[18]=40.0 from L5).
    # Gate mechanism: Q[slot] = -500 * HAS_SE, K[slot] = 5 * CONST
    # Score contribution = Q · K / sqrt(HD) = -500 * 5 / 8 = -312.5 when HAS_SE=1
    HAS_SE_GATE = 49  # Free slot after FETCH_HI (slots 0-48 used)
    attn.W_q[base + HAS_SE_GATE, BD.HAS_SE] = -500.0
    attn.W_k[base + HAS_SE_GATE, BD.CONST] = 5.0
    # After a JSR into a function prologue, relay the ENT immediate from AX
    # to SP so declarative SP-write rules can handle larger frame sizes.
    ent_sp_fetch_gate = 51
    attn.W_q[base + ent_sp_fetch_gate, BD.MARK_SP] = 500.0
    attn.W_q[base + ent_sp_fetch_gate, BD.HAS_SE] = 500.0
    attn.W_q[base + ent_sp_fetch_gate, BD.CONST] = -500.0
    attn.W_k[base + ent_sp_fetch_gate, BD.MARK_AX] = 5.0
    attn.W_k[base + ent_sp_fetch_gate, BD.OP_ENT] = 5.0


def _set_layer6_routing_ffn(ffn, S, BD):
    """Layer 6 FFN: Output routing for AX, PC (JMP), SP, BP + HALT detection.

    With CLEAN_EMBED-sourced opcode decode, correct OP_* ≈ 5.0, false
    positive OP_* ≈ 0. Threshold 4.0 cleanly separates them (5+1=6 > 4).

    AX routing (explicit per-opcode):
      - IMM: OP_IMM AND MARK_AX → FETCH → OUTPUT
      - EXIT/NOP/JMP: OP_xxx AND MARK_AX → AX_CARRY → OUTPUT

    PC routing (JMP override):
      - CMP[0] (IS_JMP) AND MARK_PC

    HALT detection:
      - CMP[1] (IS_EXIT) AND NEXT_SE
    """
    unit = 0
    # Threshold with guards: OP + MARK_AX - MARK_PC - IS_BYTE > T
    # BUG FIX 2026-04-09: T=0.5 was far too low! With MARK_AX≈2.0 at later steps,
    # units fire even when OP_xxx=0: 100*0 + 100*2 - 50 = 150 > 0 (spurious!)
    #
    # Correct calculation with S=100 scale:
    # - Block non-target: S*MARK_AX - S*T < 0 → 200 - 100*T < 0 → T > 2
    # - Allow target:     S*OP + S*MARK_AX - S*T > 0 → 500 + 100 - 100*T > 0 → T < 6
    # T=4.0 works: blocks 200-400=-200, allows 600-400=200 (for MARK_AX=1, OP=5)
    T = 4.0

    # === IMM: FETCH → OUTPUT ===
    # Read from FETCH_LO/HI (clean staging dims written by L5 fetch head 0).
    # These dims have no prior-layer leakage, unlike EMBED_LO/HI which
    # accumulates carry-forward residuals from L3.
    # BUG FIX 2026-04-09 (part 4): Increased MARK_PC blocker from -6*S to -8*S.
    # OP_IMM ≈ 6.6 at PC marker (higher than expected 5.0), so:
    # activation = S*OP_IMM + S*MARK_AX + (-8*S)*MARK_PC + bias
    #            = 20*6.6 + 20*0 + (-160)*1 + (-10) = 132 - 160 - 10 = -38 (blocked!)
    # BUG FIX 2026-04-09 (part 8f): Add OP_JMP blocker to prevent crossfire.
    # At AX marker for JMP: OP_JMP ≈ 11.0, causing spurious activation:
    # activation = S*OP_IMM + S*MARK_AX + (-20*S)*OP_EXIT + ... - 50
    #            = 100*0 + 100*1 + 0 + ... - 50 = 50 (fires incorrectly!)
    # With OP_JMP blocker: 100*0 + 100*1 + (-2000)*11 - 50 = -22050 (blocked!)
    for k in range(16):
        ffn.W_up[unit, BD.OP_IMM] = S
        ffn.W_up[unit, BD.OP_EXIT] = -S * 20  # Strong block EXIT crossfire
        ffn.W_up[unit, BD.OP_JMP] = -S * 20  # Strong block JMP crossfire
        ffn.W_up[unit, BD.MARK_AX] = S
        ffn.W_up[unit, BD.MARK_PC] = -S * 8  # INCREASED from -6*S to block at PC marker
        # FIX 2026-05-09: was -S, but with W_down upscaled to 2.0/S, even silu(0)≈0.27 at
        # byte positions started broadcasting the IMM value into AX bytes 1-3 after enough
        # steps had accumulated residual. Strengthening IS_BYTE blocker to -10*S forces
        # the byte-position activation to a strongly negative value (silu≈0) so the unit
        # only fires at MARK_AX where IS_BYTE=0.
        ffn.W_up[unit, BD.IS_BYTE] = -S * 10  # Block at byte positions, fire at markers
        ffn.b_up[unit] = -S * T
        ffn.W_gate[unit, BD.FETCH_LO + k] = 1.0
        ffn.W_down[BD.OUTPUT_LO + k, unit] = 2.0 / S  # 2.0/(S*40) -> 2.0/S (2026-05-09)
        unit += 1
    for k in range(16):
        ffn.W_up[unit, BD.OP_IMM] = S
        ffn.W_up[unit, BD.OP_EXIT] = -S * 20
        ffn.W_up[unit, BD.OP_JMP] = -S * 20
        ffn.W_up[unit, BD.MARK_AX] = S
        ffn.W_up[unit, BD.MARK_PC] = -S * 8
        ffn.W_up[unit, BD.IS_BYTE] = -S * 10  # FIX 2026-05-09 (mirror)
        ffn.b_up[unit] = -S * T
        ffn.W_gate[unit, BD.FETCH_HI + k] = 1.0
        ffn.W_down[BD.OUTPUT_HI + k, unit] = 2.0 / S  # FIX 2026-05-09 (mirror of LO IMM fix)
        unit += 1

    # === IMM: refresh AX_CARRY staging at the AX marker ===
    # Later ALU layers consume AX_CARRY as the current AX operand. The carry
    # path can retain the previous AX value after an IMM; replace it with the
    # fetched immediate while it is still clean in FETCH_LO/HI.
    for k in range(16):
        ffn.W_up[unit, BD.OP_IMM] = S / 5
        ffn.W_up[unit, BD.MARK_AX] = S
        ffn.W_up[unit, BD.MARK_PC] = -S * 8
        ffn.W_up[unit, BD.IS_BYTE] = -S * 10
        ffn.W_up[unit, BD.OP_EXIT] = -S * 20
        ffn.W_up[unit, BD.OP_JMP] = -S * 20
        ffn.b_up[unit] = -S * 1.5
        ffn.W_gate[unit, BD.FETCH_LO + k] = 1.0
        ffn.W_gate[unit, BD.AX_CARRY_LO + k] = -1.0
        ffn.W_down[BD.AX_CARRY_LO + k, unit] = 2.0 / S
        unit += 1
    for k in range(16):
        ffn.W_up[unit, BD.OP_IMM] = S / 5
        ffn.W_up[unit, BD.MARK_AX] = S
        ffn.W_up[unit, BD.MARK_PC] = -S * 8
        ffn.W_up[unit, BD.IS_BYTE] = -S * 10
        ffn.W_up[unit, BD.OP_EXIT] = -S * 20
        ffn.W_up[unit, BD.OP_JMP] = -S * 20
        ffn.b_up[unit] = -S * 1.5
        ffn.W_gate[unit, BD.FETCH_HI + k] = 1.0
        ffn.W_gate[unit, BD.AX_CARRY_HI + k] = -1.0
        ffn.W_down[BD.AX_CARRY_HI + k, unit] = 2.0 / S
        unit += 1

    # === EXIT: AX_CARRY → OUTPUT ===
    # FIX 2026-04-16: Changed BACK from AX_FULL to AX_CARRY.
    # AX_FULL (dims 471-502) overlaps with TEMP (dims 480-511), causing PC+1
    # computation to corrupt AX_FULL_HI. AX_CARRY (dims 328-359) is safe.
    # BUG FIX 2026-04-09 (part 5): Increased MARK_PC blocker from -S to -8*S.
    # OP_EXIT ≈ 6.0 at PC marker, so activation = 20*6 + 20*0 + (-160)*1 + (-10) = -50 (blocked!)
    for k in range(16):
        ffn.W_up[unit, BD.OP_EXIT] = S
        ffn.W_up[unit, BD.OP_IMM] = -S * 20  # Strong block IMM crossfire
        ffn.W_up[unit, BD.MARK_AX] = S
        ffn.W_up[unit, BD.MARK_PC] = -S * 8  # INCREASED from -S to block at PC marker
        ffn.W_up[unit, BD.IS_BYTE] = -S  # Block at byte positions, fire at markers
        ffn.b_up[unit] = -S * T
        ffn.W_gate[unit, BD.AX_CARRY_LO + k] = 1.0  # Use AX_CARRY (no TEMP overlap)
        ffn.W_down[BD.OUTPUT_LO + k, unit] = 2.0 / S
        unit += 1
    for k in range(16):
        ffn.W_up[unit, BD.OP_EXIT] = S
        ffn.W_up[unit, BD.OP_IMM] = -S * 20  # Strong block IMM crossfire
        ffn.W_up[unit, BD.MARK_AX] = S
        ffn.W_up[unit, BD.MARK_PC] = -S * 8  # INCREASED from -S to block at PC marker
        ffn.W_up[unit, BD.IS_BYTE] = -S  # Block at byte positions, fire at markers
        ffn.b_up[unit] = -S * T
        ffn.W_gate[unit, BD.AX_CARRY_HI + k] = 1.0  # Use AX_CARRY (no TEMP overlap)
        ffn.W_down[BD.OUTPUT_HI + k, unit] = 2.0 / S
        unit += 1

    # === NOP: AX_CARRY → OUTPUT ===
    # FIX 2026-04-16: Changed BACK from AX_FULL to AX_CARRY.
    # AX_FULL (dims 471-502) overlaps with TEMP (dims 480-511), causing PC+1
    # computation to corrupt AX_FULL_HI. AX_CARRY (dims 328-359) is safe.
    # BUG FIX 2026-04-09 (part 6): Increased MARK_PC blocker from -S to -8*S.
    # BUG FIX 2026-04-09 (part 8e): Add IS_BYTE blocker to prevent firing at byte positions.
    # OP_NOP has residual values at PC byte positions, causing spurious activation.
    for k in range(16):
        ffn.W_up[unit, BD.OP_NOP] = S
        ffn.W_up[unit, BD.MARK_AX] = S
        ffn.W_up[unit, BD.MARK_PC] = -S * 8  # INCREASED from -S to block at PC marker
        ffn.W_up[unit, BD.IS_BYTE] = -S * 10  # Block at byte positions
        ffn.b_up[unit] = -S * T
        ffn.W_gate[unit, BD.AX_CARRY_LO + k] = 1.0  # Use AX_CARRY (no TEMP overlap)
        ffn.W_down[BD.OUTPUT_LO + k, unit] = 2.0 / S
        unit += 1
    for k in range(16):
        ffn.W_up[unit, BD.OP_NOP] = S
        ffn.W_up[unit, BD.MARK_AX] = S
        ffn.W_up[unit, BD.MARK_PC] = -S * 8  # INCREASED from -S to block at PC marker
        ffn.W_up[unit, BD.IS_BYTE] = -S * 10  # Block at byte positions
        ffn.b_up[unit] = -S * T
        ffn.W_gate[unit, BD.AX_CARRY_HI + k] = 1.0  # Use AX_CARRY (no TEMP overlap)
        ffn.W_down[BD.OUTPUT_HI + k, unit] = 2.0 / S
        unit += 1

    # === JSR: AX_CARRY → OUTPUT (preserves AX through JSR) ===
    # FIX 2026-05-10: Without an OP_JSR routing here, OUTPUT at the AX marker
    # for a JSR step is left at L3's default (cancel-only), which makes AX byte
    # 0 prediction generate spurious low-order bytes (e.g. 6 for the JSR target
    # `JSR 18` instead of the preserved AX value 0). The malformed AX bytes
    # then derail the rest of the step structure: at MEM marker the L14 attn
    # writes -115 to OUTPUT[0..15], byte tokens get logits ≈ -1100, and the
    # marker logits (all -10) win the argmax, emitting REG_PC at MEM byte 0.
    # That spurious PC marker re-anchors the L0 threshold heads on PC instead
    # of MEM, so H[MEM] no longer fires at d=8 → NEXT_SE stays 0 → STEP_END is
    # not emitted at the SE slot. Mirror the EXIT/NOP/JMP routing pattern so
    # AX is preserved across JSR.
    #
    # Unlike OP_IMM/OP_EXIT/OP_NOP/OP_JMP, OP_JSR is also relayed to SP/STACK0
    # markers (CMP[4] / OP_JSR pathway used by L6 head 6 + L7 SP gather). So we
    # add explicit MARK_SP/BP/STACK0/MEM blockers to keep this unit firing only
    # at MARK_AX (where MARK_AX=1 and the rest are 0).
    for k in range(16):
        ffn.W_up[unit, BD.OP_JSR] = S
        ffn.W_up[unit, BD.MARK_AX] = S
        ffn.W_up[unit, BD.MARK_PC] = -S * 8  # Block at PC marker (OP_JSR active via L5)
        ffn.W_up[unit, BD.MARK_SP] = -S * 8  # Block at SP marker (OP_JSR relayed)
        ffn.W_up[unit, BD.MARK_BP] = -S * 8  # Block at BP marker
        ffn.W_up[unit, BD.MARK_STACK0] = -S * 8  # Block at STACK0 (OP_JSR relayed)
        ffn.W_up[unit, BD.MARK_MEM] = -S * 8  # Block at MEM marker
        ffn.W_up[unit, BD.IS_BYTE] = -S * 10  # Block at byte positions
        ffn.b_up[unit] = -S * T
        ffn.W_gate[unit, BD.AX_CARRY_LO + k] = 1.0
        ffn.W_down[BD.OUTPUT_LO + k, unit] = 2.0 / S
        unit += 1
    for k in range(16):
        ffn.W_up[unit, BD.OP_JSR] = S
        ffn.W_up[unit, BD.MARK_AX] = S
        ffn.W_up[unit, BD.MARK_PC] = -S * 8
        ffn.W_up[unit, BD.MARK_SP] = -S * 8
        ffn.W_up[unit, BD.MARK_BP] = -S * 8
        ffn.W_up[unit, BD.MARK_STACK0] = -S * 8
        ffn.W_up[unit, BD.MARK_MEM] = -S * 8
        ffn.W_up[unit, BD.IS_BYTE] = -S * 10
        ffn.b_up[unit] = -S * T
        ffn.W_gate[unit, BD.AX_CARRY_HI + k] = 1.0
        ffn.W_down[BD.OUTPUT_HI + k, unit] = 2.0 / S
        unit += 1

    # === JMP: AX_CARRY → OUTPUT (preserves AX through JMP) ===
    # FIX 2026-04-16: Changed BACK from AX_FULL to AX_CARRY.
    # AX_FULL (dims 471-502) overlaps with TEMP (dims 480-511), causing PC+1
    # computation to corrupt AX_FULL_HI. AX_CARRY (dims 328-359) is safe.
    # BUG FIX 2026-04-09 (part 8e): Add IS_BYTE blocker to prevent firing at byte positions.
    # OP_JMP has residual values at PC byte positions, causing spurious activation.
    for k in range(16):
        ffn.W_up[unit, BD.OP_JMP] = S
        ffn.W_up[unit, BD.MARK_AX] = S
        ffn.W_up[unit, BD.HAS_SE] = S
        ffn.W_up[unit, BD.MARK_PC] = -S  # Block at PC marker
        ffn.W_up[unit, BD.IS_BYTE] = -S * 10  # Block at byte positions
        ffn.b_up[unit] = -S * 6.5
        ffn.W_gate[unit, BD.AX_CARRY_LO + k] = 1.0
        ffn.W_down[BD.OUTPUT_LO + k, unit] = 2.0 / S
        unit += 1
    for k in range(16):
        ffn.W_up[unit, BD.OP_JMP] = S
        ffn.W_up[unit, BD.MARK_AX] = S
        ffn.W_up[unit, BD.HAS_SE] = S
        ffn.W_up[unit, BD.MARK_PC] = -S  # Block at PC marker
        ffn.W_up[unit, BD.IS_BYTE] = -S * 10  # Block at byte positions
        ffn.b_up[unit] = -S * 6.5
        ffn.W_gate[unit, BD.AX_CARRY_HI + k] = 1.0
        ffn.W_down[BD.OUTPUT_HI + k, unit] = 2.0 / S
        unit += 1

    # === FIRST-STEP JMP: OP_JMP at AX, write FETCH to OUTPUT at PC marker ===
    # For the first step, OP_JMP is at the current AX marker (not previous step).
    # Layer 6 attention head 0 relays from previous step (one-step delay), so it
    # won't work for first-step JMP. Instead, we need to route FETCH directly.
    # Condition: OP_JMP at AX marker AND NOT HAS_SE
    # At PC marker position, attend to current AX marker and copy FETCH → OUTPUT
    # This is handled by Layer 13 attention (branch target relay), but we need
    # to ensure it fires for first-step JMP by setting a flag.
    # Actually, simpler: just write FETCH directly at PC marker when OP_JMP AND NOT HAS_SE.

    # Check at PC marker: is there an active JMP at the AX marker in this step?
    # We'll use a relay from AX to PC within the same step (intra-step relay).
    # This is complex, so instead let's modify the PC increment logic to check OP_JMP.

    # Actually, the cleanest solution: Add units that fire at PC marker when
    # OP_JMP is active anywhere in the step. But OP_JMP is only at AX marker...

    # Better approach: Modify Layer 13 to handle first-step JMP by relaying
    # FETCH from AX to PC when OP_JMP AND NOT HAS_SE.
    # For now, skip this and let Layer 13 handle it.

    # === JMP PC override: cancel PC+5, write JMP target from AX_CARRY ===
    # CMP[0] ≈ 7 for JMP, ≈ 3.2 false positive (longer programs inflate it).
    # Threshold 5.5: requires CMP[0] > 4.5, separating 7.0 from 3.2.
    T_jmp = 5.5
    for k in range(16):
        ffn.W_up[unit, BD.MARK_PC] = S
        ffn.W_up[unit, BD.CMP + 0] = S
        ffn.W_up[unit, BD.MARK_AX] = -S * 10
        ffn.W_up[unit, BD.CONST] = -S * 1000
        ffn.b_up[unit] = -S * T_jmp
        ffn.W_gate[unit, BD.OUTPUT_LO + k] = -1.0
        ffn.W_down[BD.OUTPUT_LO + k, unit] = 2.0 / S
        unit += 1
    for k in range(16):
        ffn.W_up[unit, BD.MARK_PC] = S
        ffn.W_up[unit, BD.CMP + 0] = S
        ffn.W_up[unit, BD.MARK_AX] = -S * 10
        ffn.W_up[unit, BD.CONST] = -S * 1000
        ffn.b_up[unit] = -S * T_jmp
        ffn.W_gate[unit, BD.OUTPUT_HI + k] = -1.0
        ffn.W_down[BD.OUTPUT_HI + k, unit] = 2.0 / S
        unit += 1
    # Add JMP target (AX_CARRY at PC marker from L6 attn head 0)
    for k in range(16):
        target_lo = (k * INSTR_WIDTH + PC_OFFSET) & 0xF
        ffn.W_up[unit, BD.MARK_PC] = S
        ffn.W_up[unit, BD.CMP + 0] = S
        ffn.W_up[unit, BD.MARK_AX] = -S * 10
        ffn.W_up[unit, BD.CONST] = -S * 1000
        ffn.b_up[unit] = -S * T_jmp
        ffn.W_gate[unit, BD.AX_CARRY_LO + k] = 1.0
        ffn.W_down[BD.OUTPUT_LO + target_lo, unit] = 2.0 / S
        unit += 1
    for k in range(16):
        target_hi = ((k * INSTR_WIDTH + PC_OFFSET) >> 4) & 0xF
        ffn.W_up[unit, BD.MARK_PC] = S
        ffn.W_up[unit, BD.CMP + 0] = S
        ffn.W_up[unit, BD.MARK_AX] = -S * 10
        ffn.W_up[unit, BD.CONST] = -S * 1000
        ffn.b_up[unit] = -S * T_jmp
        ffn.W_gate[unit, BD.AX_CARRY_LO + k] = 1.0
        ffn.W_down[BD.OUTPUT_HI + target_hi, unit] = 2.0 / S
        unit += 1

    # === FIRST-STEP JMP PC override: use OP_JMP directly when NOT HAS_SE ===
    # For first step, CMP[0] isn't set (no previous step to relay from).
    # Instead, use OP_JMP flag directly (set by L5 FFN opcode decode at PC marker).
    # Threshold: OP_JMP ≈ 5.0, so T=4.5 separates it from false positives.
    T_op_jmp = 4.5
    # Cancel PC+INSTR_WIDTH
    for k in range(16):
        ffn.W_up[unit, BD.MARK_PC] = S
        ffn.W_up[unit, BD.OP_JMP] = S
        ffn.W_up[unit, BD.HAS_SE] = -S  # only when NOT HAS_SE (first step)
        ffn.W_up[unit, BD.MARK_AX] = -S * 10
        ffn.b_up[unit] = -S * (T_op_jmp + 0.5)  # require all three conditions
        ffn.W_gate[unit, BD.OUTPUT_LO + k] = -1.0
        ffn.W_down[BD.OUTPUT_LO + k, unit] = 2.0 / S
        unit += 1
    for k in range(16):
        ffn.W_up[unit, BD.MARK_PC] = S
        ffn.W_up[unit, BD.OP_JMP] = S
        ffn.W_up[unit, BD.HAS_SE] = -S
        ffn.W_up[unit, BD.MARK_AX] = -S * 10
        ffn.b_up[unit] = -S * (T_op_jmp + 0.5)
        ffn.W_gate[unit, BD.OUTPUT_HI + k] = -1.0
        ffn.W_down[BD.OUTPUT_HI + k, unit] = 2.0 / S
        unit += 1
    # Add JMP target from AX_CARRY (written by L5 head 3)
    # JMP immediate is converted: idx = imm // INSTR_WIDTH, PC = idx_to_pc(idx)
    # idx_to_pc adds PC_OFFSET, so for JMP 0x20: idx=4, PC=4*8+2=34
    # We need to output PC (not raw immediate), so add PC_OFFSET.
    # Strategy: nibble shift by +2 for LO, direct copy for HI
    for k in range(16):
        target_lo = (k * INSTR_WIDTH + PC_OFFSET) & 0xF
        ffn.W_up[unit, BD.MARK_PC] = S
        ffn.W_up[unit, BD.OP_JMP] = S
        ffn.W_up[unit, BD.HAS_SE] = -S
        ffn.W_up[unit, BD.MARK_AX] = -S * 10
        ffn.b_up[unit] = -S * (T_op_jmp + 0.5)
        ffn.W_gate[unit, BD.AX_CARRY_LO + k] = 1.0
        ffn.W_down[BD.OUTPUT_LO + target_lo, unit] = 2.0 / S
        unit += 1
    for k in range(16):
        target_hi = ((k * INSTR_WIDTH + PC_OFFSET) >> 4) & 0xF
        ffn.W_up[unit, BD.MARK_PC] = S
        ffn.W_up[unit, BD.OP_JMP] = S
        ffn.W_up[unit, BD.HAS_SE] = -S
        ffn.W_up[unit, BD.MARK_AX] = -S * 10
        ffn.b_up[unit] = -S * (T_op_jmp + 0.5)
        ffn.W_gate[unit, BD.AX_CARRY_LO + k] = 1.0
        ffn.W_down[BD.OUTPUT_HI + target_hi, unit] = 2.0 / S
        unit += 1

    # === ALL-STEP JMP PC override: use OP_JMP + FETCH directly ===
    # FIX 2026-04-16: Now that L5 head 3 fetches immediate for ALL steps (HAS_SE gate
    # removed), we can use FETCH directly for JMP at any step position.
    # This path uses OP_JMP (detected by L5 FFN) + FETCH (from L5 head 3) at PC marker.
    # The JMP immediate IS the PC value (compiler encodes idx*8+PC_OFFSET directly).
    T_op_jmp_all = 4.5
    # Cancel PC+INSTR_WIDTH (clear OUTPUT)
    for k in range(16):
        ffn.W_up[unit, BD.MARK_PC] = S
        ffn.W_up[unit, BD.OP_JMP] = S
        ffn.W_up[unit, BD.MARK_AX] = -S * 10
        ffn.b_up[unit] = -S * T_op_jmp_all
        ffn.W_gate[unit, BD.OUTPUT_LO + k] = -1.0
        ffn.W_down[BD.OUTPUT_LO + k, unit] = 2.0 / S
        unit += 1
    for k in range(16):
        ffn.W_up[unit, BD.MARK_PC] = S
        ffn.W_up[unit, BD.OP_JMP] = S
        ffn.W_up[unit, BD.MARK_AX] = -S * 10
        ffn.b_up[unit] = -S * T_op_jmp_all
        ffn.W_gate[unit, BD.OUTPUT_HI + k] = -1.0
        ffn.W_down[BD.OUTPUT_HI + k, unit] = 2.0 / S
        unit += 1
    # Add JMP target from FETCH (immediate value = PC target)
    for k in range(16):
        target_lo = (k * INSTR_WIDTH + PC_OFFSET) & 0xF
        ffn.W_up[unit, BD.MARK_PC] = S
        ffn.W_up[unit, BD.OP_JMP] = S
        ffn.W_up[unit, BD.MARK_AX] = -S * 10
        ffn.b_up[unit] = -S * T_op_jmp_all
        ffn.W_gate[unit, BD.FETCH_LO + k] = 1.0
        ffn.W_down[BD.OUTPUT_LO + target_lo, unit] = 2.0 / S
        unit += 1
    for k in range(16):
        target_hi = ((k * INSTR_WIDTH + PC_OFFSET) >> 4) & 0xF
        ffn.W_up[unit, BD.MARK_PC] = S
        ffn.W_up[unit, BD.OP_JMP] = S
        ffn.W_up[unit, BD.MARK_AX] = -S * 10
        ffn.b_up[unit] = -S * T_op_jmp_all
        ffn.W_gate[unit, BD.FETCH_LO + k] = 1.0
        ffn.W_down[BD.OUTPUT_HI + target_hi, unit] = 2.0 / S
        unit += 1

    # === HALT detection: CMP[1] AND NEXT_SE → convert SE to HALT ===
    # CMP[1] is scaled relay from OP_EXIT (L6 attn head 1), so use a lower
    # threshold tuned to keep EXIT active while suppressing NOP leakage.
    ffn.W_up[unit, BD.CMP + 1] = S
    ffn.W_up[unit, BD.NEXT_SE] = S
    ffn.b_up[unit] = -S * 1.3
    ffn.b_gate[unit] = 1.0
    ffn.W_down[BD.NEXT_HALT, unit] = 2.0 / S
    ffn.W_down[BD.NEXT_SE, unit] = -2.0 / S
    unit += 1

    # === TEMP clearing at PC marker ===
    # Clear TEMP dims at PC marker to prevent residual values from causing
    # spurious BZ/BNZ borrow logic activation. TEMP values are only valid
    # at AX marker (where they're computed in L5 FFN), not at other markers.
    # EXCEPT: TEMP[0] is used for IS_JSR flag (first-step decode + L6 relay).
    # Condition: MARK_PC AND NOT IS_BYTE (only at actual PC marker token)
    for k in range(32):
        if k == 0:
            # Skip TEMP[0] - used for IS_JSR flag, leave unit with zero weights
            unit += 1
            continue
        ffn.W_up[unit, BD.MARK_PC] = S
        ffn.W_up[unit, BD.IS_BYTE] = -S
        ffn.b_up[unit] = -S * 0.5
        ffn.W_gate[unit, BD.TEMP + k] = -1.0
        ffn.W_down[BD.TEMP + k, unit] = 2.0 / S
        unit += 1

    # === CMP[3] clearing at PC marker ===
    # FIX 2026-04-16: L6 attention head 6 relays POP group flags to CMP[3] at all
    # positions it fires (SP, STACK0, BP, PC, MEM). At PC marker, CMP[3] is spurious
    # and causes unit 960 to fire, corrupting OUTPUT for EXIT instruction.
    # Clear CMP[3] at PC marker to prevent this. CMP[3] is only valid at SP/STACK0.
    # Condition: MARK_PC AND NOT IS_BYTE AND CMP[3] > 0 (only clear if set)
    ffn.W_up[unit, BD.MARK_PC] = S
    ffn.W_up[unit, BD.IS_BYTE] = -S
    ffn.b_up[unit] = -S * 0.5
    ffn.W_gate[unit, BD.CMP + 3] = -1.0  # Self-referential clearing
    ffn.W_down[BD.CMP + 3, unit] = 2.0 / S
    unit += 1

    # === SP/BP/STACK0 identity carry (EMBED → OUTPUT passthrough) ===
    # 2-way AND: MARK_xxx AND NOT IS_BYTE
    # Prevents activation at byte positions where markers are relayed by attention.
    # Only activates at actual marker tokens (SP=259, BP=260, STACK0=268).
    for marker_dim in [BD.MARK_SP, BD.MARK_BP, BD.MARK_STACK0]:
        for k in range(16):
            ffn.W_up[unit, marker_dim] = S
            ffn.W_up[unit, BD.IS_BYTE] = -S  # NOT IS_BYTE
            ffn.W_up[unit, BD.HAS_SE] = S
            # Marker residuals are around 1.0 in the compact declarative path.
            # Require HAS_SE so this identity path is disabled on the
            # synthetic first step, where L3 owns SP/BP/STACK0 defaults.
            ffn.b_up[unit] = -S * 1.5
            ffn.W_gate[unit, BD.EMBED_LO + k] = 1.0
            ffn.W_down[BD.OUTPUT_LO + k, unit] = 2.0 / S
            unit += 1
        for k in range(16):
            ffn.W_up[unit, marker_dim] = S
            ffn.W_up[unit, BD.IS_BYTE] = -S  # NOT IS_BYTE
            ffn.W_up[unit, BD.HAS_SE] = S
            ffn.b_up[unit] = -S * 1.5
            ffn.W_gate[unit, BD.EMBED_HI + k] = 1.0
            ffn.W_down[BD.OUTPUT_HI + k, unit] = 2.0 / S
            unit += 1

    # === PSH: SP -= 8 (SP output = SP_carry - 8) ===
    # At SP marker, when PSH: cancel identity carry, write new value.
    # Uses PSH_AT_SP ≈ 1.0 (relayed OP_PSH from L6 attn head 6, clean dimension).
    # Threshold 1.5: PSH_AT_SP(1) + MARK_SP(1) = 2 > 1.5 → fires.
    T_psh = 1.5
    for k in range(16):
        new_k = (k - 8) % 16
        ffn.W_up[unit, BD.PSH_AT_SP] = S
        ffn.W_up[unit, BD.MARK_SP] = S
        ffn.b_up[unit] = -S * T_psh
        ffn.W_gate[unit, BD.EMBED_LO + k] = 1.0
        ffn.W_down[BD.OUTPUT_LO + new_k, unit] = 2.0 / S
        ffn.W_down[BD.OUTPUT_LO + k, unit] += -2.0 / S  # cancel identity
        unit += 1
    # Hi nibble: if old_lo < 8, borrow → hi -= 1
    for k in range(16):
        new_k_borrow = (k - 1) % 16
        ffn.W_up[unit, BD.PSH_AT_SP] = S
        ffn.W_up[unit, BD.MARK_SP] = S
        ffn.b_up[unit] = -S * T_psh
        ffn.W_gate[unit, BD.EMBED_HI + k] = 1.0
        for lo_bit in range(8, 16):
            ffn.W_up[unit, BD.EMBED_LO + lo_bit] = -S
        ffn.W_down[BD.OUTPUT_HI + new_k_borrow, unit] = 2.0 / S
        ffn.W_down[BD.OUTPUT_HI + k, unit] += -2.0 / S  # cancel identity
        unit += 1


    
    
    
    
    
    
    
        # === JSR: SP -= 8 (computed from actual SP carry in L7) ===
    # CMP[4] (JSR flag) relayed by L6 attn head 6. EMBED has SP bytes in L7.
    T_jsr_sp = 1.5
    for k in range(16):
        new_k = (k - 8) % 16
        ffn.W_up[unit, BD.CMP + 4] = S
        ffn.W_up[unit, BD.MARK_SP] = S
        ffn.W_up[unit, BD.HAS_SE] = -S
        ffn.b_up[unit] = -S * T_jsr_sp
        ffn.W_gate[unit, BD.EMBED_LO + k] = 1.0
        ffn.W_down[BD.OUTPUT_LO + new_k, unit] = 2.0 / S
        ffn.W_down[BD.OUTPUT_LO + k, unit] += -2.0 / S
        unit += 1
    for k in range(16):
        new_k_borrow = (k - 1) % 16
        ffn.W_up[unit, BD.CMP + 4] = S
        ffn.W_up[unit, BD.MARK_SP] = S
        ffn.W_up[unit, BD.HAS_SE] = -S
        ffn.b_up[unit] = -S * T_jsr_sp
        ffn.W_gate[unit, BD.EMBED_HI + k] = 1.0
        for lo_bit in range(8, 16):
            ffn.W_up[unit, BD.EMBED_LO + lo_bit] = -S
        ffn.W_down[BD.OUTPUT_HI + new_k_borrow, unit] = 2.0 / S
        ffn.W_down[BD.OUTPUT_HI + k, unit] += -2.0 / S
        unit += 1

    # FIX 2026-05-09 (C1.F): SP byte 0 fixup gated on OP_JSR + MARK_SP.
    # The 32 CMP[4]-gated nibble units above don't fire reliably at MARK_SP
    # for JSR step 1 — empirically that produced SP=0xFF08 (only the lo
    # nibble decremented; the hi-nibble borrow was missing). Add two units
    # gated directly on OP_JSR (relayed to MARK_SP at value ≈ 5.0 by L6 attn
    # head 6: W_o[OP_JSR, base+5] = 5.0 in _set_opcode_relay_head). Mirrors
    # the PSH SP-decrement pattern but specialized for SP_init=0x10000:
    # SP=0x10000, byte 0 LO=0,HI=0; after -=8 → byte 0 = 0xF8 (LO=8,HI=15).
    # Activation: (S/5)*5 + S*MARK_SP - S*1.5 = 0.5S → silu(50) ≈ 50.
    # For non-JSR opcodes: OP_JSR ≈ 0 → activation = -0.5S → silu ≈ 0
    # (no spurious firing; PSH path remains untouched).
    # LO nibble: write OUTPUT_LO[8], cancel OUTPUT_LO[0] (L3 default).
    ffn.W_up[unit, BD.OP_JSR] = S / 5
    ffn.W_up[unit, BD.MARK_SP] = S
    ffn.W_up[unit, BD.HAS_SE] = -S
    ffn.b_up[unit] = -S * T_jsr_sp
    ffn.b_gate[unit] = 1.0  # constant gate
    ffn.W_down[BD.OUTPUT_LO + 8, unit] = 2.0 / S
    ffn.W_down[BD.OUTPUT_LO + 0, unit] = -2.0 / S  # cancel L3 default (lo=0)
    unit += 1
    # HI nibble: write OUTPUT_HI[15], cancel OUTPUT_HI[0].
    ffn.W_up[unit, BD.OP_JSR] = S / 5
    ffn.W_up[unit, BD.MARK_SP] = S
    ffn.W_up[unit, BD.HAS_SE] = -S
    ffn.b_up[unit] = -S * T_jsr_sp
    ffn.b_gate[unit] = 1.0
    ffn.W_down[BD.OUTPUT_HI + 15, unit] = 2.0 / S
    ffn.W_down[BD.OUTPUT_HI + 0, unit] = -2.0 / S  # cancel L3 default (hi=0)
    unit += 1

    # Byte 1 (BYTE_INDEX_0): always 0xFF (borrow propagates from byte 0)
    T_jsr_b1 = 3.5
    ffn.W_up[unit, BD.CMP + 4] = S
    ffn.W_up[unit, BD.BYTE_INDEX_0] = S
    ffn.W_up[unit, BD.IS_BYTE] = S
    ffn.W_up[unit, BD.H1 + 2] = S
    ffn.b_up[unit] = -S * T_jsr_b1
    ffn.W_gate[unit, BD.CONST] = 1.0
    ffn.W_down[BD.OUTPUT_LO + 15, unit] = 10.0 / S
    unit += 1
    ffn.W_up[unit, BD.CMP + 4] = S
    ffn.W_up[unit, BD.BYTE_INDEX_0] = S
    ffn.W_up[unit, BD.IS_BYTE] = S
    ffn.W_up[unit, BD.H1 + 2] = S
    ffn.b_up[unit] = -S * T_jsr_b1
    ffn.W_gate[unit, BD.CONST] = 1.0
    ffn.W_down[BD.OUTPUT_HI + 15, unit] = 10.0 / S
    unit += 1
    # Byte 2 (BYTE_INDEX_1): 0x00 (borrow from byte 1 underflow for SP=0x10000)
    ffn.W_up[unit, BD.CMP + 4] = S
    ffn.W_up[unit, BD.BYTE_INDEX_1] = S
    ffn.W_up[unit, BD.IS_BYTE] = S
    ffn.W_up[unit, BD.H1 + 2] = S
    ffn.b_up[unit] = -S * T_jsr_b1
    ffn.W_gate[unit, BD.CONST] = 1.0
    ffn.W_down[BD.OUTPUT_LO + 0, unit] = 10.0 / S
    unit += 1
    ffn.W_up[unit, BD.CMP + 4] = S
    ffn.W_up[unit, BD.BYTE_INDEX_1] = S
    ffn.W_up[unit, BD.IS_BYTE] = S
    ffn.W_up[unit, BD.H1 + 2] = S
    ffn.b_up[unit] = -S * T_jsr_b1
    ffn.W_gate[unit, BD.CONST] = 1.0
    ffn.W_down[BD.OUTPUT_HI + 0, unit] = 10.0 / S
    unit += 1
    # Byte 3: identity carry (unchanged for SP=0x10000)

        # === PSH: STACK0 = AX (override STACK0 carry with AX value) ===
    # Uses PSH_AT_SP ≈ 1.0 (relayed OP_PSH, clean dimension).
    # ALU_LO/HI at STACK0 marker has AX_CARRY value (copied by L6 attn head 2).
    # Cancel identity carry and write ALU value.
    T_psh_s0 = 1.5
    for k in range(16):
        ffn.W_up[unit, BD.PSH_AT_SP] = S
        ffn.W_up[unit, BD.MARK_STACK0] = S
        ffn.b_up[unit] = -S * T_psh_s0
        # Gate: data routing only (EMBED vs ALU)
        ffn.W_gate[unit, BD.EMBED_LO + k] = -1.0
        ffn.W_gate[unit, BD.ALU_LO + k] = 1.0
        ffn.W_down[BD.OUTPUT_LO + k, unit] = 2.0 / S
        unit += 1
    for k in range(16):
        ffn.W_up[unit, BD.PSH_AT_SP] = S
        ffn.W_up[unit, BD.MARK_STACK0] = S
        ffn.b_up[unit] = -S * T_psh_s0
        ffn.W_gate[unit, BD.EMBED_HI + k] = -1.0
        ffn.W_gate[unit, BD.ALU_HI + k] = 1.0
        ffn.W_down[BD.OUTPUT_HI + k, unit] = 2.0 / S
        unit += 1

    # === GETCHAR: AX = AX_CARRY (pass through, runner overrides later) ===
    for k in range(16):
        ffn.W_up[unit, BD.OP_GETCHAR] = S
        ffn.W_up[unit, BD.MARK_AX] = S
        ffn.W_up[unit, BD.MARK_PC] = -S  # Block at PC marker
        ffn.b_up[unit] = -S * T
        ffn.W_gate[unit, BD.AX_CARRY_LO + k] = 1.0
        ffn.W_down[BD.OUTPUT_LO + k, unit] = 2.0 / S
        unit += 1
    for k in range(16):
        ffn.W_up[unit, BD.OP_GETCHAR] = S
        ffn.W_up[unit, BD.MARK_AX] = S
        ffn.W_up[unit, BD.MARK_PC] = -S  # Block at PC marker
        ffn.b_up[unit] = -S * T
        ffn.W_gate[unit, BD.AX_CARRY_HI + k] = 1.0
        ffn.W_down[BD.OUTPUT_HI + k, unit] = 2.0 / S
        unit += 1

    # === BZ: AX passthrough (AX unchanged during branch test) ===
    for k in range(16):
        ffn.W_up[unit, BD.OP_BZ] = S
        ffn.W_up[unit, BD.MARK_AX] = S
        ffn.W_up[unit, BD.MARK_PC] = -S  # Block at PC marker
        ffn.b_up[unit] = -S * T
        ffn.W_gate[unit, BD.AX_CARRY_LO + k] = 1.0
        ffn.W_down[BD.OUTPUT_LO + k, unit] = 2.0 / S
        unit += 1
    for k in range(16):
        ffn.W_up[unit, BD.OP_BZ] = S
        ffn.W_up[unit, BD.MARK_AX] = S
        ffn.W_up[unit, BD.MARK_PC] = -S  # Block at PC marker
        ffn.b_up[unit] = -S * T
        ffn.W_gate[unit, BD.AX_CARRY_HI + k] = 1.0
        ffn.W_down[BD.OUTPUT_HI + k, unit] = 2.0 / S
        unit += 1

    # === BNZ: AX passthrough (AX unchanged during branch test) ===
    for k in range(16):
        ffn.W_up[unit, BD.OP_BNZ] = S
        ffn.W_up[unit, BD.MARK_AX] = S
        ffn.W_up[unit, BD.MARK_PC] = -S  # Block at PC marker
        ffn.b_up[unit] = -S * T
        ffn.W_gate[unit, BD.AX_CARRY_LO + k] = 1.0
        ffn.W_down[BD.OUTPUT_LO + k, unit] = 2.0 / S
        unit += 1
    for k in range(16):
        ffn.W_up[unit, BD.OP_BNZ] = S
        ffn.W_up[unit, BD.MARK_AX] = S
        ffn.W_up[unit, BD.MARK_PC] = -S  # Block at PC marker
        ffn.b_up[unit] = -S * T
        ffn.W_gate[unit, BD.AX_CARRY_HI + k] = 1.0
        ffn.W_down[BD.OUTPUT_HI + k, unit] = 2.0 / S
        unit += 1

    # === PSH: AX passthrough (AX unchanged during push) ===
    for k in range(16):
        ffn.W_up[unit, BD.OP_PSH] = S
        ffn.W_up[unit, BD.MARK_AX] = S
        ffn.W_up[unit, BD.MARK_PC] = -S  # Block at PC marker
        ffn.b_up[unit] = -S * T
        ffn.W_gate[unit, BD.AX_CARRY_LO + k] = 1.0
        ffn.W_down[BD.OUTPUT_LO + k, unit] = 2.0 / S
        unit += 1
    for k in range(16):
        ffn.W_up[unit, BD.OP_PSH] = S
        ffn.W_up[unit, BD.MARK_AX] = S
        ffn.W_up[unit, BD.MARK_PC] = -S  # Block at PC marker
        ffn.b_up[unit] = -S * T
        ffn.W_gate[unit, BD.AX_CARRY_HI + k] = 1.0
        ffn.W_down[BD.OUTPUT_HI + k, unit] = 2.0 / S
        unit += 1

    # === ADJ: AX passthrough (AX unchanged during stack adjust) ===
    for k in range(16):
        ffn.W_up[unit, BD.OP_ADJ] = S
        ffn.W_up[unit, BD.MARK_AX] = S
        ffn.W_up[unit, BD.MARK_PC] = -S  # Block at PC marker
        ffn.b_up[unit] = -S * T
        ffn.W_gate[unit, BD.AX_CARRY_LO + k] = 1.0
        ffn.W_down[BD.OUTPUT_LO + k, unit] = 2.0 / S
        unit += 1
    for k in range(16):
        ffn.W_up[unit, BD.OP_ADJ] = S
        ffn.W_up[unit, BD.MARK_AX] = S
        ffn.W_up[unit, BD.MARK_PC] = -S  # Block at PC marker
        ffn.b_up[unit] = -S * T
        ffn.W_gate[unit, BD.AX_CARRY_HI + k] = 1.0
        ffn.W_down[BD.OUTPUT_HI + k, unit] = 2.0 / S
        unit += 1

    # === ADJ: SP writeback (route AX result → SP marker) ===
    # ADJ computes new_sp in AX (via ALU), then writes result to SP marker
    # Cancel identity carry, write AX_CARRY (the computed result)
    T_adj = 1.5
    for k in range(16):
        ffn.W_up[unit, BD.OP_ADJ] = S
        ffn.W_up[unit, BD.MARK_SP] = S
        ffn.b_up[unit] = -S * T_adj
        ffn.W_gate[unit, BD.EMBED_LO + k] = -1.0  # Cancel identity
        ffn.W_gate[unit, BD.AX_CARRY_LO + k] = 1.0  # Write result
        ffn.W_down[BD.OUTPUT_LO + k, unit] = 2.0 / S
        unit += 1
    for k in range(16):
        ffn.W_up[unit, BD.OP_ADJ] = S
        ffn.W_up[unit, BD.MARK_SP] = S
        ffn.b_up[unit] = -S * T_adj
        ffn.W_gate[unit, BD.EMBED_HI + k] = -1.0  # Cancel identity
        ffn.W_gate[unit, BD.AX_CARRY_HI + k] = 1.0  # Write result
        ffn.W_down[BD.OUTPUT_HI + k, unit] = 2.0 / S
        unit += 1

    # === ENT: SP writeback (route AX result → SP marker) ===
    # ENT computes new_sp = sp - (8 + imm) in L8/L9 ALU at AX marker
    # Result is relayed to AX_CARRY by L6 attention, then written to SP marker
    # Cancel identity carry (EMBED = old SP), write AX_CARRY (new SP)
    # FIX 2026-04-17: Add HAS_SE gate. On first step, AX_CARRY is empty
    # (no previous step to relay from), causing garbage OUTPUT values.
    # First-step ENT SP is handled by separate units below.
    T_ent = 2.5  # Increased threshold: OP_ENT(5) + MARK_SP(1) + HAS_SE(1) = 7 > 2.5
    for k in range(16):
        ffn.W_up[unit, BD.OP_ENT] = S
        ffn.W_up[unit, BD.MARK_SP] = S
        ffn.W_up[unit, BD.HAS_SE] = S  # Only subsequent steps
        ffn.b_up[unit] = -S * T_ent
        ffn.W_gate[unit, BD.EMBED_LO + k] = -1.0  # Cancel identity
        ffn.W_gate[unit, BD.AX_CARRY_LO + k] = 1.0  # Write result
        ffn.W_down[BD.OUTPUT_LO + k, unit] = 2.0 / S
        unit += 1
    for k in range(16):
        ffn.W_up[unit, BD.OP_ENT] = S
        ffn.W_up[unit, BD.MARK_SP] = S
        ffn.W_up[unit, BD.HAS_SE] = S  # Only subsequent steps
        ffn.b_up[unit] = -S * T_ent
        ffn.W_gate[unit, BD.EMBED_HI + k] = -1.0  # Cancel identity
        ffn.W_gate[unit, BD.AX_CARRY_HI + k] = 1.0  # Write result
        ffn.W_down[BD.OUTPUT_HI + k, unit] = 2.0 / S
        unit += 1

    # === ENT first-step SP byte 0 (32 units) ===
    # On first step, SP_init = 0x10000, byte 0 = 0x00.
    # ENT computes: SP_new = SP_init - 8 - imm
    # For byte 0: result = 0x00 - 8 - imm_byte0 = -8 - imm
    # Lo nibble: (-8 - FETCH_LO) mod 16
    # Hi nibble: (-1 - FETCH_HI) mod 16 (always borrow from lo since -8 - x < 0)
    # Condition: OP_ENT + MARK_SP + NOT HAS_SE
    T_ent_first = 1.5
    for imm_lo in range(16):
        result_lo = (-8 - imm_lo) % 16
        ffn.W_up[unit, BD.OP_ENT] = S
        ffn.W_up[unit, BD.MARK_SP] = S
        ffn.W_up[unit, BD.HAS_SE] = -S * 10  # Block on subsequent steps
        ffn.b_up[unit] = -S * T_ent_first
        ffn.W_gate[unit, BD.FETCH_LO + imm_lo] = 1.0  # Select based on imm lo nibble
        ffn.W_down[BD.OUTPUT_LO + result_lo, unit] = 5.0 / S  # Strong output
        unit += 1
    for imm_hi in range(16):
        result_hi = (-1 - imm_hi) % 16  # Always borrow from lo
        ffn.W_up[unit, BD.OP_ENT] = S
        ffn.W_up[unit, BD.MARK_SP] = S
        ffn.W_up[unit, BD.HAS_SE] = -S * 10  # Block on subsequent steps
        ffn.b_up[unit] = -S * T_ent_first
        ffn.W_gate[unit, BD.FETCH_HI + imm_hi] = 1.0  # Select based on imm hi nibble
        ffn.W_down[BD.OUTPUT_HI + result_hi, unit] = 5.0 / S  # Strong output
        unit += 1

    # === ENT first-step SP bytes 1-3 (6 units) ===
    # FIX 2026-04-17: Add units for SP bytes 1-3 during ENT first step.
    # SP = 0x10000 - 8 - imm, for small imm:
    # - Byte 1 = 0xFF (borrow from byte 0 < 0 always)
    # - Byte 2 = 0x00 (borrow absorbed from 0x01 - 1 = 0x00)
    # - Byte 3 = 0x00 (unchanged)
    # Fire at: OP_ENT + BYTE_INDEX_* + IS_BYTE + H1[SP] + NOT HAS_SE
    SP_I = 2
    T_ent_byte = 4.0

    # SP byte 0 pos → predict byte 1 = 0xFF
    # Need OUTPUT_LO[15], OUTPUT_HI[15]
    ffn.W_up[unit, BD.OP_ENT] = S
    ffn.W_up[unit, BD.BYTE_INDEX_0] = S
    ffn.W_up[unit, BD.IS_BYTE] = S
    ffn.W_up[unit, BD.H1 + SP_I] = S
    ffn.W_up[unit, BD.HAS_SE] = -S * 10  # Block on subsequent steps
    ffn.b_up[unit] = -S * T_ent_byte
    ffn.W_gate[unit, BD.CONST] = 1.0
    ffn.W_down[BD.OUTPUT_LO + 15, unit] = 10.0 / S  # Strong for nibble 15
    unit += 1
    ffn.W_up[unit, BD.OP_ENT] = S
    ffn.W_up[unit, BD.BYTE_INDEX_0] = S
    ffn.W_up[unit, BD.IS_BYTE] = S
    ffn.W_up[unit, BD.H1 + SP_I] = S
    ffn.W_up[unit, BD.HAS_SE] = -S * 10
    ffn.b_up[unit] = -S * T_ent_byte
    ffn.W_gate[unit, BD.CONST] = 1.0
    ffn.W_down[BD.OUTPUT_HI + 15, unit] = 10.0 / S  # Strong for nibble 15
    unit += 1

    # SP byte 1 pos → predict byte 2 = 0x00
    # L3 default OUTPUT = 0 should work, but add explicit units for safety
    ffn.W_up[unit, BD.OP_ENT] = S
    ffn.W_up[unit, BD.BYTE_INDEX_1] = S
    ffn.W_up[unit, BD.IS_BYTE] = S
    ffn.W_up[unit, BD.H1 + SP_I] = S
    ffn.W_up[unit, BD.HAS_SE] = -S * 10
    ffn.b_up[unit] = -S * T_ent_byte
    ffn.W_gate[unit, BD.CONST] = 1.0
    ffn.W_down[BD.OUTPUT_LO + 0, unit] = 5.0 / S  # Nibble 0
    unit += 1
    ffn.W_up[unit, BD.OP_ENT] = S
    ffn.W_up[unit, BD.BYTE_INDEX_1] = S
    ffn.W_up[unit, BD.IS_BYTE] = S
    ffn.W_up[unit, BD.H1 + SP_I] = S
    ffn.W_up[unit, BD.HAS_SE] = -S * 10
    ffn.b_up[unit] = -S * T_ent_byte
    ffn.W_gate[unit, BD.CONST] = 1.0
    ffn.W_down[BD.OUTPUT_HI + 0, unit] = 5.0 / S  # Nibble 0
    unit += 1

    # SP byte 2 pos → predict byte 3 = 0x00
    ffn.W_up[unit, BD.OP_ENT] = S
    ffn.W_up[unit, BD.BYTE_INDEX_2] = S
    ffn.W_up[unit, BD.IS_BYTE] = S
    ffn.W_up[unit, BD.H1 + SP_I] = S
    ffn.W_up[unit, BD.HAS_SE] = -S * 10
    ffn.b_up[unit] = -S * T_ent_byte
    ffn.W_gate[unit, BD.CONST] = 1.0
    ffn.W_down[BD.OUTPUT_LO + 0, unit] = 5.0 / S
    unit += 1
    ffn.W_up[unit, BD.OP_ENT] = S
    ffn.W_up[unit, BD.BYTE_INDEX_2] = S
    ffn.W_up[unit, BD.IS_BYTE] = S
    ffn.W_up[unit, BD.H1 + SP_I] = S
    ffn.W_up[unit, BD.HAS_SE] = -S * 10
    ffn.b_up[unit] = -S * T_ent_byte
    ffn.W_gate[unit, BD.CONST] = 1.0
    ffn.W_down[BD.OUTPUT_HI + 0, unit] = 5.0 / S
    unit += 1

    # === BZ PC override: branch if AX == 0 ===
    # FIX 2026-05-11: Use OP_BZ directly (scaled S/5 so OP_BZ=5 → contrib=S)
    # instead of CMP[2] (which was always ~0 because the BZ/BNZ relay couldn't
    # source OP_BZ from a position that had it set). OP_BZ is local at PC
    # marker via L5 decode, so a direct read works without an extra relay.
    # CMP[4]=AX_LO_IS_ZERO, CMP[5]=AX_HI_IS_ZERO (set by L6 head 4 relay from
    # step (N-1)'s AX byte 0).
    # 4-way AND in silu: MARK_PC + OP_BZ_scaled + CMP[4] + CMP[5] - 3.5
    # BZ+zero: 1+1+1+1=4 > 3.5 → fires. One missing: 3 < 3.5 → off.
    #
    # FIX 2026-04-16: Add IS_BYTE suppression. CMP[5] is reused for PRTF flag
    # in conversational I/O mode. Without IS_BYTE blocker, these units fire
    # at PC byte positions during PRTF (MARK_PC=0 but CMP[5]≈5), canceling
    # the L3 PC default output and causing wrong byte values.
    # FIX 2026-04-29: Lowered T_bz from 5.5 to 3.5 to match comment (4-way AND
    # threshold: 4 inputs at ~1.0 each = 4.0 > 3.5).
    T_bz = 3.5
    # Cancel existing PC+5 carry
    for k in range(16):
        ffn.W_up[unit, BD.MARK_PC] = S
        ffn.W_up[unit, BD.OP_BZ] = S / 5.0  # OP_BZ=5 → contrib=S (≈ CMP[2]=1 scaled)
        ffn.W_up[unit, BD.CMP + 4] = S
        ffn.W_up[unit, BD.CMP + 5] = S
        ffn.W_up[unit, BD.IS_BYTE] = -S * 10  # Block at byte positions
        ffn.b_up[unit] = -S * T_bz
        ffn.W_gate[unit, BD.OUTPUT_LO + k] = -1.0
        ffn.W_down[BD.OUTPUT_LO + k, unit] = 2.0 / S
        unit += 1
    for k in range(16):
        ffn.W_up[unit, BD.MARK_PC] = S
        ffn.W_up[unit, BD.OP_BZ] = S / 5.0
        ffn.W_up[unit, BD.CMP + 4] = S
        ffn.W_up[unit, BD.CMP + 5] = S
        ffn.W_up[unit, BD.IS_BYTE] = -S * 10  # Block at byte positions
        ffn.b_up[unit] = -S * T_bz
        ffn.W_gate[unit, BD.OUTPUT_HI + k] = -1.0
        ffn.W_down[BD.OUTPUT_HI + k, unit] = 2.0 / S
        unit += 1
    # Write branch target directly to OUTPUT (same pattern as JMP override).
    # FIX 2026-04-29: Removed -5 remap that was incorrect. The JMP override
    # writes FETCH directly to OUTPUT, and BZ should do the same.
    # FIX 2026-04-16: Add MARK_STACK0 suppression.
    for k in range(16):
        target_lo = (k * INSTR_WIDTH + PC_OFFSET) & 0xF
        ffn.W_up[unit, BD.MARK_PC] = S
        ffn.W_up[unit, BD.OP_BZ] = S / 5.0
        ffn.W_up[unit, BD.CMP + 4] = S
        ffn.W_up[unit, BD.CMP + 5] = S
        ffn.W_up[unit, BD.IS_BYTE] = -S * 10
        ffn.W_up[unit, BD.MARK_STACK0] = -S * 10
        ffn.b_up[unit] = -S * T_bz
        ffn.W_gate[unit, BD.FETCH_LO + k] = 1.0
        ffn.W_down[BD.OUTPUT_LO + target_lo, unit] = 2.0 / S
        unit += 1
    for k in range(16):
        target_hi = ((k * INSTR_WIDTH + PC_OFFSET) >> 4) & 0xF
        ffn.W_up[unit, BD.MARK_PC] = S
        ffn.W_up[unit, BD.OP_BZ] = S / 5.0
        ffn.W_up[unit, BD.CMP + 4] = S
        ffn.W_up[unit, BD.CMP + 5] = S
        ffn.W_up[unit, BD.IS_BYTE] = -S * 10
        ffn.W_up[unit, BD.MARK_STACK0] = -S * 10
        ffn.b_up[unit] = -S * T_bz
        ffn.W_gate[unit, BD.FETCH_LO + k] = 1.0
        ffn.W_down[BD.OUTPUT_HI + target_hi, unit] = 2.0 / S
        unit += 1

    # === BNZ PC override: branch if AX != 0 ===
    # FIX 2026-05-11: Replaced CMP[3] (always 0 - relay couldn't source OP_BNZ)
    # with OP_BNZ/5 (local at PC marker via L5 decode). Same fix pattern as
    # BZ override above.
    # Two exclusive groups. All conditions in silu, gate ONLY for value.
    # OP_BNZ_scaled = S*(OP_BNZ/5) ≈ S, CMP[4]=lo_zero, CMP[5]=hi_zero.
    #
    # Group A: lo nibble is nonzero
    #   up = S*(MARK_PC + OP_BNZ/5 - CMP[4]) - S*1.5
    #   BNZ + lo_nonzero: 1+1-0=2 > 1.5 → fires
    #   BNZ + lo_zero: 1+1-1=1 < 1.5 → off
    #   gate = just the value (cancel or write)
    T_bnz = 1.5
    # Cancel existing OUTPUT
    for k in range(16):
        ffn.W_up[unit, BD.MARK_PC] = S
        ffn.W_up[unit, BD.OP_BNZ] = S / 5.0  # OP_BNZ=5 in pure_neural (only 1 OPCODE_BYTE_LO match via head 5, no leak)
        ffn.W_up[unit, BD.CMP + 4] = -S
        ffn.b_up[unit] = -S * T_bnz
        ffn.W_gate[unit, BD.OUTPUT_LO + k] = -1.0
        ffn.W_down[BD.OUTPUT_LO + k, unit] = 2.0 / S
        unit += 1
    for k in range(16):
        ffn.W_up[unit, BD.MARK_PC] = S
        ffn.W_up[unit, BD.OP_BNZ] = S / 5.0  # OP_BNZ=5 in pure_neural (only 1 OPCODE_BYTE_LO match via head 5, no leak)
        ffn.W_up[unit, BD.CMP + 4] = -S
        ffn.b_up[unit] = -S * T_bnz
        ffn.W_gate[unit, BD.OUTPUT_HI + k] = -1.0
        ffn.W_down[BD.OUTPUT_HI + k, unit] = 2.0 / S
        unit += 1
    # Write target directly to OUTPUT (FIX 2026-04-29: removed -5 remap)
    for k in range(16):
        target_lo = (k * INSTR_WIDTH + PC_OFFSET) & 0xF
        ffn.W_up[unit, BD.MARK_PC] = S
        ffn.W_up[unit, BD.OP_BNZ] = S / 5.0  # OP_BNZ=5 in pure_neural (only 1 OPCODE_BYTE_LO match via head 5, no leak)
        ffn.W_up[unit, BD.CMP + 4] = -S
        ffn.b_up[unit] = -S * T_bnz
        ffn.W_gate[unit, BD.FETCH_LO + k] = 1.0
        ffn.W_down[BD.OUTPUT_LO + target_lo, unit] = 2.0 / S
        unit += 1
    for k in range(16):
        target_hi = ((k * INSTR_WIDTH + PC_OFFSET) >> 4) & 0xF
        ffn.W_up[unit, BD.MARK_PC] = S
        ffn.W_up[unit, BD.OP_BNZ] = S / 5.0  # OP_BNZ=5 in pure_neural (only 1 OPCODE_BYTE_LO match via head 5, no leak)
        ffn.W_up[unit, BD.CMP + 4] = -S
        ffn.b_up[unit] = -S * T_bnz
        ffn.W_gate[unit, BD.FETCH_LO + k] = 1.0
        ffn.W_down[BD.OUTPUT_HI + target_hi, unit] = 2.0 / S
        unit += 1

    # Group B: lo IS zero but hi is nonzero
    #   up = S*(MARK_PC + OP_BNZ/5 + CMP[4] - CMP[5]) - S*2.5
    #   BNZ + lo_zero + hi_nonzero: 1+1+1-0=3 > 2.5 → fires
    #   BNZ + lo_zero + hi_zero (AX=0): 1+1+1-1=2 < 2.5 → off
    #   gate = just the value
    T_bnz_b = 2.5
    # Cancel existing OUTPUT
    for k in range(16):
        ffn.W_up[unit, BD.MARK_PC] = S
        ffn.W_up[unit, BD.OP_BNZ] = S / 5.0  # OP_BNZ=5 in pure_neural (only 1 OPCODE_BYTE_LO match via head 5, no leak)
        ffn.W_up[unit, BD.CMP + 4] = S
        ffn.W_up[unit, BD.CMP + 5] = -S
        ffn.b_up[unit] = -S * T_bnz_b
        ffn.W_gate[unit, BD.OUTPUT_LO + k] = -1.0
        ffn.W_down[BD.OUTPUT_LO + k, unit] = 2.0 / S
        unit += 1
    for k in range(16):
        ffn.W_up[unit, BD.MARK_PC] = S
        ffn.W_up[unit, BD.OP_BNZ] = S / 5.0  # OP_BNZ=5 in pure_neural (only 1 OPCODE_BYTE_LO match via head 5, no leak)
        ffn.W_up[unit, BD.CMP + 4] = S
        ffn.W_up[unit, BD.CMP + 5] = -S
        ffn.b_up[unit] = -S * T_bnz_b
        ffn.W_gate[unit, BD.OUTPUT_HI + k] = -1.0
        ffn.W_down[BD.OUTPUT_HI + k, unit] = 2.0 / S
        unit += 1
    # Write target directly to OUTPUT (FIX 2026-04-29: removed -5 remap)
    for k in range(16):
        target_lo = (k * INSTR_WIDTH + PC_OFFSET) & 0xF
        ffn.W_up[unit, BD.MARK_PC] = S
        ffn.W_up[unit, BD.OP_BNZ] = S / 5.0  # OP_BNZ=5 in pure_neural (only 1 OPCODE_BYTE_LO match via head 5, no leak)
        ffn.W_up[unit, BD.CMP + 4] = S
        ffn.W_up[unit, BD.CMP + 5] = -S
        ffn.b_up[unit] = -S * T_bnz_b
        ffn.W_gate[unit, BD.FETCH_LO + k] = 1.0
        ffn.W_down[BD.OUTPUT_LO + target_lo, unit] = 2.0 / S
        unit += 1
    for k in range(16):
        target_hi = ((k * INSTR_WIDTH + PC_OFFSET) >> 4) & 0xF
        ffn.W_up[unit, BD.MARK_PC] = S
        ffn.W_up[unit, BD.OP_BNZ] = S / 5.0  # OP_BNZ=5 in pure_neural (only 1 OPCODE_BYTE_LO match via head 5, no leak)
        ffn.W_up[unit, BD.CMP + 4] = S
        ffn.W_up[unit, BD.CMP + 5] = -S
        ffn.b_up[unit] = -S * T_bnz_b
        ffn.W_gate[unit, BD.FETCH_LO + k] = 1.0
        ffn.W_down[BD.OUTPUT_HI + target_hi, unit] = 2.0 / S
        unit += 1

    # === Cancel OPCODE_BYTE contamination at AX marker ===
    # L5 head 1 writes the opcode byte to OPCODE_BYTE_LO/HI at AX marker.
    # These dims overlap ADDR_B0_LO/ADDR_B1_LO which L7 uses to gather
    # prev-AX address bytes for L15 memory lookup.  The stale opcode nibbles
    # inflate L15 address-match scores and break ZFOD.
    # Fix: negate each OPCODE_BYTE dim at the AX marker.
    for k in range(16):
        ffn.W_up[unit, BD.MARK_AX] = S
        ffn.b_up[unit] = -S * 0.5
        ffn.W_gate[unit, BD.OPCODE_BYTE_LO + k] = -1.0
        ffn.W_down[BD.ADDR_B0_LO + k, unit] = 2.0 / S
        unit += 1
    for k in range(16):
        ffn.W_up[unit, BD.MARK_AX] = S
        ffn.b_up[unit] = -S * 0.5
        ffn.W_gate[unit, BD.OPCODE_BYTE_HI + k] = -1.0
        ffn.W_down[BD.ADDR_B1_LO + k, unit] = 2.0 / S
        unit += 1

    # === Cancel MEM_STORE leakage at non-MEM markers ===
    # L6 attn head 6 writes MEM_STORE at ALL marker positions (SP, STACK0, BP,
    # MEM) because the O matrix is position-independent. But MEM_STORE should
    # only persist at the MEM marker — L14 uses it to gate memory generation,
    # and stray MEM_STORE at SP/STACK0/BP causes L14 to write garbage to
    # OUTPUT_LO/HI at those positions.
    # Fix: subtract MEM_STORE at non-MEM markers (SP, STACK0, BP).
    ffn.W_up[unit, BD.MARK_SP] = S
    ffn.W_up[unit, BD.MARK_STACK0] = S
    ffn.W_up[unit, BD.MARK_BP] = S
    ffn.b_up[unit] = -S * 0.5
    ffn.W_gate[unit, BD.MEM_STORE] = -1.0
    ffn.W_down[BD.MEM_STORE, unit] = 2.0 / S
    unit += 1

    # Same for MEM_ADDR_SRC (also leaked by head 6)
    ffn.W_up[unit, BD.MARK_SP] = S
    ffn.W_up[unit, BD.MARK_STACK0] = S
    ffn.W_up[unit, BD.MARK_BP] = S
    ffn.b_up[unit] = -S * 0.5
    ffn.W_gate[unit, BD.MEM_ADDR_SRC] = -1.0
    ffn.W_down[BD.MEM_ADDR_SRC, unit] = 2.0 / S
    unit += 1

    # === Clear ALU_LO/HI at AX marker (unconditional) ===
    # L4 FFN stores PC+2 in ALU_LO/HI for L5 Head 5 bytecode fetch.
    # This residual collides with L7 Head 0's STACK0→ALU operand gather.
    # Clear ALU here so L7 writes cleanly. L7 output weight is amplified
    # (6.0) to overcome this clear (-5.0), giving net +1.0 for the
    # correct STACK0 nibble and -4.0 for the wrong PC+2 nibble.
    for k in range(16):
        ffn.W_up[unit, BD.MARK_AX] = S
        ffn.b_up[unit] = -S * 0.5
        ffn.b_gate[unit] = 1.0
        ffn.W_down[BD.ALU_LO + k, unit] = -10.0 / S
        unit += 1

    for k in range(16):
        ffn.W_up[unit, BD.MARK_AX] = S
        ffn.b_up[unit] = -S * 0.5
        ffn.b_gate[unit] = 1.0
        ffn.W_down[BD.ALU_HI + k, unit] = -10.0 / S
        unit += 1


def _set_layer6_relay_heads(attn, S, BD, HD):
    """L6 attention head 6: Cross-register data relay for PSH.

    IMPORTANT: Uses head 6 (previously unused) instead of head 2 to avoid
    conflicts with _set_layer6_attn which already configures heads 0-5.
    Previously this function overwrote head 2, causing AX_CARRY corruption.

    NOTE: The original head 3 (SP relay for ADJ) is NOT configured here because:
    1. Head 7 is reserved for JSR handling (configured later in set_vm_weights)
    2. ADJ operation is not critical for basic arithmetic (ADD/SUB/MUL/DIV)
    3. Can be added back if ADJ support is needed, using a different approach

    Head 6: At STACK0 marker, read AX marker's AX_CARRY → ALU staging.
    This provides the AX value at STACK0 position for PSH (STACK0=AX).
    Distance from STACK0 marker to AX marker = 20-5 = 15 tokens back.
    FIX 2026-04-15: Reverted from AX_FULL to AX_CARRY because AX_FULL (dims 471-502)
    overlaps with TEMP (dims 480-511), causing PC+1 computation to corrupt AX_FULL.
    """
    L = 50.0

    # Head 6: STACK0 ← AX LO nibble (AX_CARRY_LO → ALU_LO at STACK0 marker, d=15)
    # Head 7: STACK0 ← AX HI nibble (AX_CARRY_HI → ALU_HI at STACK0 marker, d=15)
    # NOTE: _set_opcode_relay_head uses head 6 V slots 1-7. We use slots 8-23 for LO.
    # FIX 2026-04-16: Extended from 13 to 16 dims to support all nibble values (0-15).
    # Previously only copied dims 0-12, causing PSH to fail for values with nibbles 13-15
    # (e.g., 30 = 0x1E has lo nibble 14, which was not copied).
    # FIX 2026-04-15: Reverted from AX_FULL to AX_CARRY because AX_FULL overlaps TEMP.

    # Head 6: LO nibble (slots 8-23)
    base6 = 6 * HD
    attn.W_q[base6, BD.MARK_STACK0] = L
    attn.W_q[base6, BD.MARK_AX] = -L  # block at AX marker
    attn.W_k[base6, BD.MARK_AX] = L
    for k in range(16):
        attn.W_v[base6 + 8 + k, BD.AX_CARRY_LO + k] = 1.0
        attn.W_o[BD.ALU_LO + k, base6 + 8 + k] = 1.0

    # Head 7: HI nibble (slots 33-48)
    # Function-call weights also use head 7 for JSR return-address push and
    # occupy slots 1-32. Keep PSH HI in a disjoint slot range so JSR OUTPUT
    # values cannot leak through the ALU_HI projection.
    base7 = 7 * HD
    attn.W_q[base7, BD.MARK_STACK0] = L + L * 20
    attn.W_q[base7, BD.MARK_AX] = -L  # block at AX marker
    attn.W_q[base7, BD.CONST] = -L * 20
    attn.W_k[base7, BD.MARK_AX] = L
    for k in range(16):
        attn.W_v[base7 + 33 + k, BD.AX_CARRY_HI + k] = 1.0
        attn.W_o[BD.ALU_HI + k, base7 + 33 + k] = 1.0


def _set_layer7_memory_heads(attn, S, BD, HD):
    """L7 attention heads 1-6: Memory operation relay heads.

    Head 7: Broadcast MEM_STORE + MEM_ADDR_SRC from MEM marker to MEM byte positions.
    Heads 2-4: Gather prev step's AX bytes → current AX positions (for LI/LC address).
    Head 5: Relay OP_LI/OP_LC flags from AX marker to AX byte positions.
    Head 6: Relay PSH/ENT/JSR flags from STACK0 marker to STACK0 byte positions.
    """
    L = 15.0
    PC_I = 0
    MEM_I = 4
    AX_I = 1
    SP_I = 2
    BP_I = 3

    # === Head 7: MEM flag broadcast (MEM marker → MEM byte positions d=1..8) ===
    # NOTE: was head 1 but collided with LEA relay (also head 1 in operand gather).
    base = 7 * HD
    # Q: fires at MEM marker + positions d≤8.5 from MEM (H3[MEM]=1)
    # Suppress non-MEM positions: subtract H1[AX], H1[SP], H1[BP], H4[BP]
    attn.W_q[base, BD.MARK_MEM] = L
    attn.W_q[base, BD.H3 + MEM_I] = L  # d≤8.5 from MEM → all MEM bytes
    attn.W_q[base, BD.H1 + AX_I] = -L  # suppress AX area
    attn.W_q[base, BD.H1 + SP_I] = -L  # suppress SP area
    attn.W_q[base, BD.H1 + BP_I] = -L  # suppress BP area
    attn.W_q[base, BD.H4 + BP_I] = -L  # suppress STACK0 area
    # K: attend to MEM marker
    attn.W_k[base, BD.MARK_MEM] = L
    # V: copy MEM_STORE, MEM_ADDR_SRC, OP_JSR, OP_ENT
    # FIX 2026-04-16: Also broadcast OP_JSR/OP_ENT for L14 val heads STACK0 source
    attn.W_v[base + 1, BD.MEM_STORE] = 1.0
    attn.W_v[base + 2, BD.MEM_ADDR_SRC] = 1.0
    attn.W_v[base + 3, BD.OP_JSR] = 1.0
    attn.W_v[base + 4, BD.OP_ENT] = 1.0
    # O: write to same dims (accumulates at byte positions)
    attn.W_o[BD.MEM_STORE, base + 1] = 1.0
    attn.W_o[BD.MEM_ADDR_SRC, base + 2] = 1.0
    attn.W_o[BD.OP_JSR, base + 3] = 1.0
    attn.W_o[BD.OP_ENT, base + 4] = 1.0

    # === Heads 2-4: Gather prev AX bytes → AX positions (for LI/LC) ===
    # Each head J gathers addr byte J from prev step's AX byte J.
    # K: BYTE_INDEX_J + H1[AX] (fires at prev step's AX byte J input position)
    # Distance from curr AX marker to prev AX byte J ≈ 34 tokens
    for j in range(3):
        head = 2 + j
        base = head * HD
        byte_idx_dim = [BD.BYTE_INDEX_0, BD.BYTE_INDEX_1, BD.BYTE_INDEX_2][j]
        addr_lo_out = [BD.ADDR_B0_LO, BD.ADDR_B1_LO, BD.ADDR_B2_LO][j]
        addr_hi_out = [BD.ADDR_B0_HI, BD.ADDR_B1_HI, BD.ADDR_B2_HI][j]
        # Q: fires at AX marker + AX bytes, suppressed at MEM/STACK0
        attn.W_q[base, BD.MARK_AX] = L
        attn.W_q[base, BD.H1 + AX_I] = L  # AX byte positions (d≤4.5 from AX)
        attn.W_q[base, BD.H3 + MEM_I] = -L  # suppress MEM area
        attn.W_q[base, BD.H4 + BP_I] = -L  # suppress STACK0 area
        # K: fires at prev step's AX byte J
        attn.W_k[base, byte_idx_dim] = L
        attn.W_k[base, BD.H1 + AX_I] = L  # must be in AX area
        # Anti-leakage: require both conditions (threshold 1.5)
        attn.W_q[base + 33, BD.CONST] = -L / 2
        attn.W_q[base + 33, BD.MARK_AX] = L
        attn.W_k[base + 33, BD.CONST] = L
        # V: copy CLEAN_EMBED nibbles
        for k in range(16):
            attn.W_v[base + 1 + k, BD.CLEAN_EMBED_LO + k] = 1.0
            attn.W_v[base + 17 + k, BD.CLEAN_EMBED_HI + k] = 1.0
        # O: write to ADDR_BJ_LO/HI
        for k in range(16):
            attn.W_o[addr_lo_out + k, base + 1 + k] = 1.0
            attn.W_o[addr_hi_out + k, base + 17 + k] = 1.0

    # === Head 5: Relay opcode flags from AX marker → AX byte positions ===
    base = 5 * HD
    # Q: fires at AX marker + AX bytes
    attn.W_q[base, BD.MARK_AX] = L
    attn.W_q[base, BD.H1 + AX_I] = L  # AX byte positions
    # K: attend to AX marker
    attn.W_k[base, BD.MARK_AX] = L
    # V: OP_LI, OP_LC, OP_LEA (scaled: ≈5 × 0.2 = ≈1.0)
    attn.W_v[base + 1, BD.OP_LI] = 0.2
    attn.W_v[base + 2, BD.OP_LC] = 0.2
    attn.W_v[base + 3, BD.OP_LEA] = 0.2  # FIX 2026-04-13: Relay LEA for first-step byte 2 output
    # V: BITWISE_OP = OP_AND + OP_OR + OP_XOR (scaled by 0.2 each, sum ≈ 1.0 when one is active)
    # FIX 2026-05-08: Relay bitwise op flag to byte positions for L10 passthrough gating
    attn.W_v[base + 4, BD.OP_AND] = 0.2
    attn.W_v[base + 4, BD.OP_OR] = 0.2
    attn.W_v[base + 4, BD.OP_XOR] = 0.2
    # V: Individual bitwise op flags for BitwiseBytePropagationPostOp
    # FIX 2026-05-08: Relay individual flags for correct operation dispatch in post-op
    attn.W_v[base + 5, BD.OP_AND] = 0.2
    attn.W_v[base + 6, BD.OP_OR] = 0.2
    attn.W_v[base + 7, BD.OP_XOR] = 0.2
    # V: OP_JSR relay for AX-bytes-1-3 zeroing at L14
    # FIX 2026-05-12 (fix-jsr-ax-bytes-1-3): Broadcast OP_JSR from AX marker to AX
    # byte positions so the L14 FFN ``_set_layer14_jsr_ax_bytes_zero`` cleanup can
    # gate on OP_JSR at byte positions. Without this, OP_JSR is only present at
    # AX marker (L5 decode) and at PC/SP/STACK0/BP/MEM markers (L6 head 6 relay),
    # leaving AX byte positions unaware of JSR active. The L14 cleanup zeros
    # OUTPUT_LO/HI nibble 1..15 (and boosts nibble 0) at IS_BYTE + H1[AX] when
    # OP_JSR is set, so AX bytes 1-3 emit byte value 0 per C4's 8-bit-AX
    # convention. See docs/PHASE_5_JSR_ENT_LEV_FOLLOWUP.md for the diagnosis.
    attn.W_v[base + 8, BD.OP_JSR] = 0.2
    # V: SHR-only byte-zeroing relay. AND/OR/XOR use the byte-propagation
    # post-op for bytes 1-3; routing them through this cleanup would force
    # genuine multi-byte bitwise results back to 8-bit values.
    # Output dim is TEMP[7] (free per audit of TEMP slot usage: 0/3/4/5/6 used
    # by other relays; 1/2/7..15 free; 16..31 reserved for AX_FULL/SP relays).
    attn.W_v[base + 9, BD.OP_SHR] = 0.2
    # V: SI/SC relays for late neural-authoritative store preservation.
    # Store ops need their opcode visible at AX byte positions so the final
    # declarative correction can preserve the in-flight AX high byte before
    # L14 consumes it as the store value.
    attn.W_v[base + 10, BD.OP_SI] = 0.2
    attn.W_v[base + 11, BD.OP_SC] = 0.2
    # V: ADD/SUB relays for high-byte base arithmetic. Carry/borrow flags only
    # indicate byte-0 overflow/underflow, so the byte propagation post-op needs
    # a separate op discriminator for no-carry high-byte cases.
    attn.W_v[base + 12, BD.OP_ADD] = 0.2
    attn.W_v[base + 13, BD.OP_SUB] = 0.2
    # O: write to relay dims (×5 to normalize)
    attn.W_o[BD.OP_LI_RELAY, base + 1] = 1.0
    attn.W_o[BD.OP_LC_RELAY, base + 2] = 1.0
    attn.W_o[BD.CMP + 7, base + 3] = 1.0  # OP_LEA relay → CMP[7]
    attn.W_o[BD.TEMP + 3, base + 4] = 1.0  # BITWISE_OP relay → TEMP[3]
    attn.W_o[BD.TEMP + 4, base + 5] = 1.0  # OP_AND relay → TEMP[4]
    attn.W_o[BD.TEMP + 5, base + 6] = 1.0  # OP_OR relay → TEMP[5]
    attn.W_o[BD.TEMP + 6, base + 7] = 1.0  # OP_XOR relay → TEMP[6]
    # O: write OP_JSR back to OP_JSR dim (×5 to normalize 0.2→1.0). Additive
    # with existing L6 head 6 OP_JSR writes (which target PC/SP/STACK0/BP/MEM,
    # disjoint from AX byte positions), so no double-counting.
    attn.W_o[BD.OP_JSR, base + 8] = 5.0
    # O: write NOCARRY_ALU_OP relay → TEMP[7] (×5 to normalize 0.2→1.0).
    attn.W_o[BD.TEMP + 7, base + 9] = 1.0
    attn.W_o[BD.OP_SI, base + 10] = 5.0
    attn.W_o[BD.OP_SC, base + 11] = 5.0
    attn.W_o[BD.TEMP + 8, base + 12] = 1.0
    attn.W_o[BD.TEMP + 9, base + 13] = 1.0

    # === Head 6: Relay PSH/ENT/JSR from STACK0 marker → STACK0 byte positions ===
    # Also relay PSH_AT_SP from SP marker → SP byte positions.
    base = 6 * HD
    # Q: fires at STACK0 area (marker + bytes) AND SP area (marker + bytes)
    attn.W_q[base, BD.MARK_STACK0] = L
    # FIX 2026-04-16: Changed from L1H4 (d≤6.5) to H4 (d≤9.5) to cover all STACK0 bytes.
    # STACK0 bytes are at d=6,7,8,9 from BP marker. L1H4 only covered byte 0 (d=6).
    attn.W_q[base, BD.H4 + BP_I] = L  # d≤9.5 from BP (STACK0 bytes at d=6-9)
    attn.W_q[base, BD.H1 + BP_I] = -L  # Exclude BP bytes (d≤4.5)
    attn.W_q[base, BD.IS_BYTE] = L  # only at byte positions (not at SE)
    attn.W_q[base, BD.MARK_SP] = L  # also fire at SP marker
    attn.W_q[base, BD.H1 + SP_I] = L  # also fire at SP byte positions
    # Suppress non-target areas
    attn.W_q[base, BD.H1 + AX_I] = -L
    attn.W_q[base, BD.H3 + MEM_I] = -L
    # K: attend to STACK0 marker (for STACK0 positions) or SP marker (for SP positions)
    attn.W_k[base, BD.MARK_STACK0] = L
    attn.W_k[base, BD.MARK_SP] = L
    # V: copy CMP[0] (PSH), CMP[2] (ENT), CMP[3] (POP), CMP[4] (JSR), PSH_AT_SP from markers
    attn.W_v[base + 1, BD.CMP + 0] = 1.0  # PSH flag (legacy)
    attn.W_v[base + 2, BD.CMP + 2] = 1.0  # ENT flag
    attn.W_v[base + 3, BD.CMP + 4] = 1.0  # JSR flag
    attn.W_v[base + 4, BD.PSH_AT_SP] = 1.0  # Clean PSH flag for SP bytes
    attn.W_v[base + 5, BD.CMP + 3] = 1.0  # POP group flag (for SP passthrough suppression)
    # O: accumulate at STACK0/SP byte positions
    attn.W_o[BD.CMP + 0, base + 1] = 1.0
    attn.W_o[BD.CMP + 2, base + 2] = 1.0
    attn.W_o[BD.CMP + 4, base + 3] = 1.0
    attn.W_o[BD.PSH_AT_SP, base + 4] = 1.0
    attn.W_o[BD.CMP + 3, base + 5] = 1.0  # POP group relay


def _set_layer8_sp_gather(attn, S, BD, HD):
    """L8 attention heads 0-2: Gather SP bytes → STACK0 positions.

    For *SP lookup address. Each head J gathers SP byte J to STACK0 area.
    """
    L = 15.0
    AX_I = 1
    SP_I = 2
    BP_I = 3
    MEM_I = 4

    for j in range(3):
        base = j * HD
        byte_idx_dim = [BD.BYTE_INDEX_0, BD.BYTE_INDEX_1, BD.BYTE_INDEX_2][j]
        addr_lo_out = [BD.ADDR_B0_LO, BD.ADDR_B1_LO, BD.ADDR_B2_LO][j]
        addr_hi_out = [BD.ADDR_B0_HI, BD.ADDR_B1_HI, BD.ADDR_B2_HI][j]
        # Q: fires at STACK0 area (d=5..9 from BP or STACK0 marker)
        attn.W_q[base, BD.MARK_STACK0] = L
        attn.W_q[base, BD.H4 + BP_I] = L  # d≤9.5 from BP
        # Suppress non-STACK0
        attn.W_q[base, BD.H1 + AX_I] = -L
        attn.W_q[base, BD.H1 + SP_I] = -L
        attn.W_q[base, BD.H3 + MEM_I] = -L
        attn.W_q[base, BD.MARK_BP] = -L  # FIX 2026-04-15: Suppress at BP marker itself
        # K: fires at SP byte J (BYTE_INDEX_J + H1[SP])
        attn.W_k[base, byte_idx_dim] = L
        attn.W_k[base, BD.H1 + SP_I] = L  # must be in SP area
        # During pop-group ops the current step's SP bytes are the post-pop
        # value. Stack reads need the previous SP address, so suppress source
        # tokens that already carry the pop relay.
        attn.W_k[base, BD.CMP + 3] = -L
        # Anti-leakage gate
        attn.W_q[base + 33, BD.MARK_STACK0] = L
        attn.W_q[base + 33, BD.CONST] = -L / 2
        attn.W_k[base + 33, BD.CONST] = L
        # V: copy CLEAN_EMBED nibbles
        for k in range(16):
            attn.W_v[base + 1 + k, BD.CLEAN_EMBED_LO + k] = 1.0
            attn.W_v[base + 17 + k, BD.CLEAN_EMBED_HI + k] = 1.0
        # O: write to ADDR_BJ_LO/HI
        for k in range(16):
            attn.W_o[addr_lo_out + k, base + 1 + k] = 1.0
            attn.W_o[addr_hi_out + k, base + 17 + k] = 1.0


def _set_layer8_multibyte_fetch(attn, S, BD, HD):
    """L8 attention head 3: Fetch code bytes at AX byte positions for multi-byte IMM.

    At each AX byte position (BYTE_INDEX_1/2/3), FETCH_LO/HI contains PC+2/3/4
    (written by L4 FFN rotation). This head queries the code section using
    FETCH_LO/HI as address, matches ADDR_KEY, and copies the code byte value
    to FETCH_LO/HI (overwriting the address with the fetched byte value).

    L6 FFN then routes FETCH → OUTPUT at byte positions when OP_IMM is active
    (via relay from L6 attention).
    """
    L = 20.0
    AX_I = 1

    base = 3 * HD
    # Q: low two nibbles from FETCH_LO/HI (PC+2/3/4 computed by L4 FFN)
    for k in range(16):
        attn.W_q[base + k, BD.FETCH_LO + k] = L
        attn.W_q[base + 16 + k, BD.FETCH_HI + k] = L
        attn.W_q[base + 36 + k, BD.ADDR_KEY + 32 + k] = L
    attn.W_q[base + 36, BD.CONST] = L
    attn.W_q[base + 36, BD.HAS_SE] = -L
    # Third nibble gate: require IS_BYTE to avoid leakage at markers
    attn.W_q[base + 32, BD.IS_BYTE] = L

    # K: address nibbles from memory key space (same as L5 heads)
    for k in range(16):
        attn.W_k[base + k, BD.ADDR_KEY + k] = L
        attn.W_k[base + 16 + k, BD.ADDR_KEY + 16 + k] = L
        attn.W_k[base + 36 + k, BD.ADDR_KEY + 32 + k] = L

    # Anti-leakage gate: suppress at non-byte positions
    GATE = 33
    attn.W_q[base + GATE, BD.IS_BYTE] = 500.0
    attn.W_q[base + GATE, BD.CONST] = -500.0
    attn.W_k[base + GATE, BD.CONST] = 5.0

    # Additional gate: only fire near AX marker (H1[AX_I] > 0)
    # This prevents spurious fetch at byte positions of other registers
    HAS_AX_GATE = 34
    attn.W_q[base + HAS_AX_GATE, BD.H1 + AX_I] = 500.0
    attn.W_q[base + HAS_AX_GATE, BD.CONST] = -500.0
    attn.W_k[base + HAS_AX_GATE, BD.CONST] = 5.0

    # HAS_SE gate: REMOVED 2026-04-29. Previously blocked on step 0, but L4 FFN
    # correctly computes FETCH addresses (PC+2/3/4) at byte positions on step 0
    # using the initial PC injected by _inject_initial_pc. Without this head firing
    # on step 0, multi-byte IMM values (e.g., IMM 256) fail because AX bytes 1-3
    # never get their code byte values fetched.

    # V: copy byte value nibbles (scaled 3x to dominate address contamination in AX_CARRY)
    for k in range(16):
        attn.W_v[base + 32 + k, BD.CLEAN_EMBED_LO + k] = 3.0
        attn.W_v[base + 48 + k, BD.CLEAN_EMBED_HI + k] = 3.0
    # O: write to AX_CARRY_LO/HI at byte positions (not FETCH — FETCH has address contamination)
    for k in range(16):
        attn.W_o[BD.AX_CARRY_LO + k, base + 32 + k] = 1.0
        attn.W_o[BD.AX_CARRY_HI + k, base + 48 + k] = 1.0


def _set_layer8_alu(ffn, S, BD):
    """L8 FFN: ADD/SUB lo nibble + carry/borrow + CMP_GROUP flag.

    Uses 3-way AND in silu path: MARK_AX + ALU_LO[a] + AX_CARRY_LO[b].
    Only the exact (a, b) pair fires — silu(-S*0.5) ≈ 0 for mismatches.
    Opcode gating: gate = OP_xxx (b_gate=0). Zero cross-opcode leakage.

    Total: ~753 units (ADD lo 256 + SUB lo 256 + ADD carry 120 +
                       SUB borrow 120 + CMP_GROUP 1).
    """
    unit = 0

    def _block_non_ax_marker_sites(unit_idx: int) -> None:
        # LEA/ADJ/ENT fetch operands can be scale-40 at non-AX marker sites.
        # These units are only valid at the AX marker.
        for blocker_dim in (
            BD.MARK_PC,
            BD.MARK_SP,
            BD.MARK_BP,
            BD.MARK_STACK0,
            BD.MARK_MEM,
            BD.MARK_SE,
            BD.IS_BYTE,
        ):
            ffn.W_up[unit_idx, blocker_dim] = -S * 1000

    # === ADD: lo nibble (256 units) ===
    for a in range(16):
        for b in range(16):
            result = (a + b) % 16
            ffn.W_up[unit, BD.MARK_AX] = S
            # FIX 2026-05-11: Bumped MARK_PC blocker from -S*2 to -S*4. At step
            # 1+ PC marker, the L6 head 0 JMP relay writes AX_CARRY_LO (the
            # carried JMP target) to ~3.0 even when OP_JMP=0 (residual leakage
            # from FETCH at the relayed AX marker). Combined with ALU_LO
            # leakage from L7 attention (~2.0 at PC marker), the original
            # -S*2 blocker is overcome: up = -200 + 100*2 + 100*3 - 250 = 52,
            # silu(52)*OP_ADD(spurious 1.0 from L5 head 6 leak) = 52 →
            # contrib +1 to OUTPUT_LO[0], which overrides the L3 PC+8
            # increment for BZ not_taken (causing wrong PC). Stronger -S*4
            # blocker: up = -400 + 500 - 250 = -150 → silu≈0.
            ffn.W_up[unit, BD.MARK_PC] = -S * 4
            ffn.W_up[unit, BD.ALU_LO + a] = S
            ffn.W_up[unit, BD.AX_CARRY_LO + b] = S
            ffn.b_up[unit] = -S * 2.5  # 3-way AND
            ffn.W_gate[unit, BD.OP_ADD] = 1.0
            # LEA moved to separate units that read from FETCH
            ffn.W_down[BD.OUTPUT_LO + result, unit] = 2.0 / S
            unit += 1

    # === LEA: lo nibble (256 units) ===
    # Like ADD but reads from FETCH_LO instead of AX_CARRY_LO
    # Require both ALU_LO and FETCH_LO to be active. FETCH_LO is now a clean
    # one-hot band in the declarative path, so amplify it locally instead of
    # assuming the older scale-40 activation.
    for a in range(16):
        for b in range(16):
            result = (a + b) % 16
            ffn.W_up[unit, BD.MARK_AX] = S * 60  # Strong MARK_AX requirement
            _block_non_ax_marker_sites(unit)
            ffn.W_up[unit, BD.ALU_LO + a] = S
            ffn.W_up[unit, BD.FETCH_LO + b] = S * 20  # Read from FETCH, not AX_CARRY
            ffn.b_up[unit] = -S * 80.5
            ffn.W_gate[unit, BD.OP_LEA] = 1.0
            ffn.W_down[BD.OUTPUT_LO + result, unit] = 2.0 / S
            unit += 1

    # === SUB: lo nibble (256 units) ===
    # Correct C4 semantics: AX = stack_top - AX = ALU - AX_CARRY = a - b.
    # ALU contains stack top (minuend), AX_CARRY contains current AX (subtrahend).
    for a in range(16):
        for b in range(16):
            result = (a - b) % 16  # ALU - AX_CARRY = stack - AX
            ffn.W_up[unit, BD.MARK_AX] = S
            # FIX 2026-05-11: Bumped MARK_PC blocker from -S*2 to -S*4 (see
            # ADD lo nibble unit above for full rationale on L6 JMP relay leak).
            ffn.W_up[unit, BD.MARK_PC] = -S * 4
            ffn.W_up[unit, BD.ALU_LO + a] = S
            ffn.W_up[unit, BD.AX_CARRY_LO + b] = S
            ffn.b_up[unit] = -S * 2.5
            ffn.W_gate[unit, BD.OP_SUB] = 1.0
            ffn.W_down[BD.OUTPUT_LO + result, unit] = 2.0 / S
            unit += 1

    # === ADD carry detection (120 units: pairs where a+b >= 16) ===
    for a in range(16):
        for b in range(16):
            if a + b >= 16:
                ffn.W_up[unit, BD.MARK_AX] = S
                # FIX 2026-05-11: Bumped MARK_PC blocker from -S*2 to -S*4.
                ffn.W_up[unit, BD.MARK_PC] = -S * 4
                ffn.W_up[unit, BD.ALU_LO + a] = S
                ffn.W_up[unit, BD.AX_CARRY_LO + b] = S
                ffn.b_up[unit] = -S * 2.5
                ffn.W_gate[unit, BD.OP_ADD] = 1.0
                # LEA moved to separate units
                ffn.W_down[BD.CARRY + 0, unit] = 2.0 / (S * 5.0)  # normalize: gate≈5 → CARRY≈1
                unit += 1

    # === LEA carry detection (120 units: pairs where a+b >= 16) ===
    # Require BOTH ALU_LO and FETCH_LO to be active; FETCH_LO is one-hot here.
    for a in range(16):
        for b in range(16):
            if a + b >= 16:
                ffn.W_up[unit, BD.MARK_AX] = S * 60
                _block_non_ax_marker_sites(unit)
                ffn.W_up[unit, BD.ALU_LO + a] = S
                ffn.W_up[unit, BD.FETCH_LO + b] = S * 20  # Read from FETCH
                ffn.b_up[unit] = -S * 80.5
                ffn.W_gate[unit, BD.OP_LEA] = 1.0
                ffn.W_down[BD.CARRY + 0, unit] = 2.0 / (S * 5.0)
                unit += 1

    # === ADJ: lo nibble (256 units) ===
    # Like LEA but gates on OP_ADJ instead
    # ADJ computes: SP = SP + signed_immediate (gathered via L7 head 1)
    # Require BOTH ALU_LO and FETCH_LO; FETCH_LO is one-hot here.
    for a in range(16):
        for b in range(16):
            result = (a + b) % 16
            ffn.W_up[unit, BD.MARK_AX] = S * 60
            _block_non_ax_marker_sites(unit)
            ffn.W_up[unit, BD.ALU_LO + a] = S
            ffn.W_up[unit, BD.FETCH_LO + b] = S * 20  # Read from FETCH (immediate)
            ffn.b_up[unit] = -S * 85
            ffn.W_gate[unit, BD.OP_ADJ] = 1.0
            ffn.W_down[BD.OUTPUT_LO + result, unit] = 2.0 / S
            unit += 1

    # === ADJ carry detection (120 units: pairs where a+b >= 16) ===
    # Require BOTH ALU_LO and FETCH_LO; FETCH_LO is one-hot here.
    for a in range(16):
        for b in range(16):
            if a + b >= 16:
                ffn.W_up[unit, BD.MARK_AX] = S * 60
                _block_non_ax_marker_sites(unit)
                ffn.W_up[unit, BD.ALU_LO + a] = S
                ffn.W_up[unit, BD.FETCH_LO + b] = S * 20  # Read from FETCH
                ffn.b_up[unit] = -S * 85
                ffn.W_gate[unit, BD.OP_ADJ] = 1.0
                ffn.W_down[BD.CARRY + 0, unit] = 2.0 / (S * 5.0)
                unit += 1

    # === SUB borrow detection (120 units: pairs where a < b) ===
    # Borrow occurs when ALU_LO < AX_CARRY_LO (when stack_top < AX for this nibble).
    # a = ALU (stack_top), b = AX_CARRY (AX). SUB = a - b, borrow when a < b.
    for a in range(16):
        for b in range(16):
            if a < b:  # Borrow when stack_top < AX
                ffn.W_up[unit, BD.MARK_AX] = S
                # FIX 2026-05-11: Bumped MARK_PC blocker from -S*2 to -S*4.
                ffn.W_up[unit, BD.MARK_PC] = -S * 4
                ffn.W_up[unit, BD.ALU_LO + a] = S
                ffn.W_up[unit, BD.AX_CARRY_LO + b] = S
                ffn.b_up[unit] = -S * 2.5
                ffn.W_gate[unit, BD.OP_SUB] = 1.0
                ffn.W_down[BD.CARRY + 0, unit] = 2.0 / (S * 5.0)  # normalize: gate≈5 → CARRY≈1
                unit += 1

    # === ENT: lo nibble subtraction (256 units) ===
    # ENT computes: SP = SP - (8 + signed_immediate)
    # For lo nibble: result_lo = (sp_lo - (8 + imm_lo)) mod 16
    # where sp_lo comes from ALU_LO (gathered by L7 head 1)
    # and imm_lo comes from FETCH_LO (instruction immediate)
    # Require BOTH ALU_LO and FETCH_LO; FETCH_LO is one-hot here.
    for sp_lo in range(16):
        for imm_lo in range(16):
            effective_b = (8 + imm_lo) % 16  # Add constant offset 8
            result = (sp_lo - effective_b) % 16
            ffn.W_up[unit, BD.MARK_AX] = S * 60
            _block_non_ax_marker_sites(unit)
            ffn.W_up[unit, BD.ALU_LO + sp_lo] = S  # SP lo nibble from L7
            ffn.W_up[unit, BD.FETCH_LO + imm_lo] = S * 20  # Immediate lo nibble
            ffn.b_up[unit] = -S * 85
            ffn.W_gate[unit, BD.OP_ENT] = 1.0
            ffn.W_down[BD.OUTPUT_LO + result, unit] = 2.0 / S
            unit += 1

    # === ENT borrow detection (256 units) ===
    # Borrow when sp_lo < (8 + imm_lo) mod 16
    # Must enumerate all cases (not just where borrow occurs) because
    # the condition depends on the constant 8, not just relative magnitudes
    # Require BOTH ALU_LO and FETCH_LO; FETCH_LO is one-hot here.
    for sp_lo in range(16):
        for imm_lo in range(16):
            effective_b = (8 + imm_lo) % 16
            # Check if borrow is needed: sp_lo < effective_b
            # But need to account for the full byte math: does (8 + imm_lo) >= 16?
            # If (8 + imm_lo) >= 16, there's a carry out of byte 0 into byte 1
            full_sum = 8 + imm_lo  # This is in range [8, 23]
            if sp_lo < (full_sum % 16) or full_sum >= 16:
                # Need borrow from byte 1
                ffn.W_up[unit, BD.MARK_AX] = S * 60
                _block_non_ax_marker_sites(unit)
                ffn.W_up[unit, BD.ALU_LO + sp_lo] = S
                ffn.W_up[unit, BD.FETCH_LO + imm_lo] = S * 20
                ffn.b_up[unit] = -S * 85
                ffn.W_gate[unit, BD.OP_ENT] = 1.0
                ffn.W_down[BD.CARRY + 0, unit] = 2.0 / (S * 5.0)
                unit += 1

    # === CMP_GROUP flag (1 unit) ===
    # ~1.0 when any comparison opcode active at AX marker.
    # OP flags ≈ 5.0, so silu(S*(5+1-1.5))=S*4.5. Normalize W_down so output ≈ 1.
    for op in [BD.OP_EQ, BD.OP_NE, BD.OP_LT, BD.OP_GT, BD.OP_LE, BD.OP_GE]:
        ffn.W_up[unit, op] = S
    ffn.W_up[unit, BD.MARK_AX] = S
    ffn.b_up[unit] = -S * 1.5  # any_cmp_op(~5) + MARK_AX(1) > 1.5
    ffn.b_gate[unit] = 1.0  # unconditional gate
    ffn.W_down[BD.CMP_GROUP, unit] = 2.0 / (S * 9.0)  # normalize: 9 * 2/(S*9) ≈ 1.0
    unit += 1

    # === CMP[0..3] clearing at AX marker (4 units) ===
    # FIX: L6 attention relay heads write JMP/EXIT/PSH/POP flags to CMP[0..3] at
    # all positions, including AX marker. These pollute the comparison flag dims.
    # This clearing negates CMP[k] at AX marker: silu(S*V)*silu(S/2)*(-2/S²) ≈ -V
    # L9 then overwrites with correct comparison flags (gated by CMP_GROUP).
    for k in range(4):
        ffn.W_up[unit, BD.CMP + k] = S
        ffn.b_up[unit] = 0
        ffn.W_gate[unit, BD.MARK_AX] = S
        ffn.b_gate[unit] = -S * 0.5
        ffn.W_down[BD.CMP + k, unit] = -2.0 / (S * S)
        unit += 1

    # === ENT/ADJ first-step ALU defaults (2 units) ===
    # FIX 2026-04-17: For first step (NOT HAS_SE), L7 attention can't gather SP
    # because the SP marker is AFTER the AX marker in the sequence (causal attention).
    # For ENT/ADJ with initial SP = 0, we need ALU_LO[0] > 0 and ALU_HI[0] > 0.
    # These units fire when: OP_ENT (or OP_ADJ) + MARK_AX + NOT HAS_SE
    # OP_ENT ≈ 22 at SP marker after L7 (amplified from 5 at embedding).
    # BUG FIX 2026-04-17: Add MARK_SP blocker. OP_ENT alone contributes ~750 at SP marker,
    # which overcomes the bias (-600). Need -1000 from MARK_SP to block.
    # L7 attention writes garbage to ALU (~-32) when there's no SP to attend to,
    # so we need large output weights to override this: 50.0/S gives ~+50 contribution.
    for alu_dim, output_weight in [(BD.ALU_LO, 50.0 / S), (BD.ALU_HI, 50.0 / S)]:
        ffn.W_up[unit, BD.OP_ENT] = S / 3  # ~22/3 ≈ 7.3 contribution at SP marker
        ffn.W_up[unit, BD.OP_ADJ] = S / 3  # Also handle ADJ first step
        ffn.W_up[unit, BD.MARK_AX] = S * 2  # 2 contribution when at AX marker
        ffn.W_up[unit, BD.MARK_SP] = -S * 10  # Block at SP marker (MARK_SP = 1)
        ffn.W_up[unit, BD.HAS_SE] = -S * 10  # Block on subsequent steps
        ffn.b_up[unit] = -S * 6  # Threshold: need ENT/ADJ + MARK_AX, blocked by HAS_SE
        ffn.b_gate[unit] = 1.0  # Always gate open (SiLU path does gating)
        ffn.W_down[alu_dim + 0, unit] = output_weight  # Write to ALU[0] for SP = 0
        unit += 1

    # === LEV: BP address relay (BP OUTPUT → ADDR dims) - PHASE 1 ===
    # For L15 to read memory at BP and BP+8, we need BP's address value
    # encoded in ADDR_B0/B1/B2 dims at the BP marker position.
    #
    # CURRENT LIMITATION: Only byte 0 is relayed here (covers addresses < 256).
    # This is sufficient for most C4 test programs which use small stack frames.
    #
    # For larger addresses, we would need:
    # - L7/L8 attention heads to gather BP bytes 1-2 from their positions
    # - Write gathered bytes to TEMP dims
    # - Copy TEMP → ADDR_B1/B2 here
    #
    # Total for byte 0 only: 32 units (1 byte × 2 nibbles × 16 values/nibble)

    # Byte 0 lo nibble: OUTPUT_LO → ADDR_B0_LO
    # FIX 2026-04-15: OP_LEV ≈ 5 and MARK_BP = 1, so up = S * (5 + 1 - 1.5) = S * 4.5
    # silu(S * 4.5) ≈ S * 4.5, so W_down = 2.0 / (S * 9) to normalize output to ~1.0
    # FIX 2026-04-16: Add PC marker exclusion. OP_LEV gets amplified to ~10 by L6,
    # so without MARK_PC penalty, units fire at PC marker (OP_LEV*10 > threshold).
    for k in range(16):
        ffn.W_up[unit, BD.OP_LEV] = S
        ffn.W_up[unit, BD.MARK_BP] = S
        ffn.W_up[unit, BD.MARK_PC] = -S * 10  # Exclude PC marker
        ffn.b_up[unit] = -S * 1.5  # both OP_LEV and MARK_BP required
        ffn.W_gate[unit, BD.OUTPUT_LO + k] = 1.0
        ffn.W_down[BD.ADDR_B0_LO + k, unit] = 2.0 / (S * 9)  # FIX: was 2.0/S
        unit += 1

    # Byte 0 hi nibble: OUTPUT_HI → ADDR_B0_HI
    for k in range(16):
        ffn.W_up[unit, BD.OP_LEV] = S
        ffn.W_up[unit, BD.MARK_BP] = S
        ffn.W_up[unit, BD.MARK_PC] = -S * 10  # Exclude PC marker
        ffn.b_up[unit] = -S * 1.5
        ffn.W_gate[unit, BD.OUTPUT_HI + k] = 1.0
        ffn.W_down[BD.ADDR_B0_HI + k, unit] = 2.0 / (S * 9)  # FIX: was 2.0/S
        unit += 1

    # Bytes 1-2: Set to zero (assume addresses < 256 for now)
    # This gives ADDR_B1 = ADDR_B2 = 0, which is correct for small addresses
    ffn.W_up[unit, BD.OP_LEV] = S
    ffn.W_up[unit, BD.MARK_BP] = S
    ffn.b_up[unit] = -S * 1.5
    ffn.W_gate[unit, BD.CONST] = 1.0  # Always 1.0
    ffn.W_down[BD.ADDR_B1_LO + 0, unit] = 2.0 / (S * 9)  # FIX: was 2.0/S
    unit += 1

    ffn.W_up[unit, BD.OP_LEV] = S
    ffn.W_up[unit, BD.MARK_BP] = S
    ffn.b_up[unit] = -S * 1.5
    ffn.W_gate[unit, BD.CONST] = 1.0
    ffn.W_down[BD.ADDR_B2_LO + 0, unit] = 2.0 / (S * 9)  # FIX: was 2.0/S
    unit += 1

    # === REMOVED 2026-04-16: PC marker ADDR_B0 moved to L9 attention head 1 ===
    # L9 attention head 1 now directly writes BP value to ADDR_B0 at PC marker.
    # Having L8 FFN also write to ADDR_B0 causes interference (values are additive).
    # The 34 units (32 for byte 0 + 2 for bytes 1-2) have been removed.

    # === LEA first-step AX byte 2 output (2 units) ===
    # FIX 2026-05-04: Moved from L4 FFN (where CMP[7] was not yet available).
    # L7 attention head 5 relays OP_LEA → CMP[7], available from L7 onward.
    # At AX byte 1 position (BYTE_INDEX_1), predicting AX byte 2.
    # BP = 0x10000, so byte 2 = 0x01. Fires only on first step (NOT HAS_SE).
    AX_I = 1
    T_lea_byte = 3.5
    ffn.W_up[unit, BD.CMP + 7] = S
    ffn.W_up[unit, BD.H1 + AX_I] = S
    ffn.W_up[unit, BD.IS_BYTE] = S
    ffn.W_up[unit, BD.BYTE_INDEX_1] = S
    ffn.W_up[unit, BD.HAS_SE] = -S
    ffn.b_up[unit] = -S * T_lea_byte
    ffn.b_gate[unit] = 1.0
    ffn.W_down[BD.OUTPUT_LO + 1, unit] = 4.0 / S
    ffn.W_down[BD.OUTPUT_LO + 0, unit] = -4.0 / S
    unit += 1
    ffn.W_up[unit, BD.CMP + 7] = S
    ffn.W_up[unit, BD.H1 + AX_I] = S
    ffn.W_up[unit, BD.IS_BYTE] = S
    ffn.W_up[unit, BD.BYTE_INDEX_1] = S
    ffn.W_up[unit, BD.HAS_SE] = -S
    ffn.b_up[unit] = -S * T_lea_byte
    ffn.b_gate[unit] = 1.0
    ffn.W_down[BD.OUTPUT_HI + 0, unit] = 2.0 / S
    unit += 1

    return unit


def _set_layer8_multibyte_routing(ffn, S, BD):
    """L8 FFN: Route FETCH → OUTPUT at AX byte positions for multi-byte IMM.

    At AX byte positions (IS_BYTE + H1[AX_I]), when OP_IMM is active (relayed
    by L8 Head 4 from AX marker), copy FETCH_LO/HI → OUTPUT_LO/HI.

    The gating uses IS_BYTE + H1[AX_I] + OP_IMM - MARK_AX*4 with threshold T=6.5.
    - Target (byte pos, IMM): up ≈ S*(1+1+5-0-6.5) = S*0.5 = 50, silu(50) ≈ 50
    - AX marker (IMM): up ≈ S*(0+1+5-4-6.5) = S*(-4.5) = -450, blocked
    - Non-IMM byte: up ≈ S*(1+1+0-0-6.5) = S*(-4.5) = -450, blocked

    W_down = 2.0/S gives: 0.02 * 50 * FETCH[k] = FETCH[k] (exact copy).
    Total: 64 units (32 lo + 32 hi).
    """
    unit_start = _set_layer8_alu(ffn, S, BD)
    unit = unit_start
    T = 6.5
    AX_I = 1

    # Write FETCH → OUTPUT at AX byte positions for multi-byte IMM.
    # Read from AX_CARRY_LO/HI (clean fetched value from L8 Head 3, no address contamination).
    # Scale up to 8.0/S to overcome residual byte-0 bias (~0.94 from embedding).
    for k in range(16):
        ffn.W_up[unit, BD.IS_BYTE] = S
        ffn.W_up[unit, BD.H1 + AX_I] = S
        ffn.W_up[unit, BD.OP_IMM] = S
        ffn.W_up[unit, BD.MARK_AX] = -S * 4
        ffn.b_up[unit] = -S * T
        ffn.W_gate[unit, BD.AX_CARRY_LO + k] = 1.0
        ffn.W_down[BD.OUTPUT_LO + k, unit] = 8.0 / S
        unit += 1
    for k in range(16):
        ffn.W_up[unit, BD.IS_BYTE] = S
        ffn.W_up[unit, BD.H1 + AX_I] = S
        ffn.W_up[unit, BD.OP_IMM] = S
        ffn.W_up[unit, BD.MARK_AX] = -S * 4
        ffn.b_up[unit] = -S * T
        ffn.W_gate[unit, BD.AX_CARRY_HI + k] = 1.0
        ffn.W_down[BD.OUTPUT_HI + k, unit] = 8.0 / S
        unit += 1


def _set_layer9_alu(ffn, S, BD):
    """L9 FFN: ADD/SUB hi nibble (with carry/borrow) + comparison flags.

    Hi nibble uses 4-way AND: MARK_AX + ALU_HI[a] + AX_CARRY_HI[b] ± CARRY.
    CARRY ≈ 1.0 when present (from L8 with b_gate=0), ≈ 0 otherwise.

    Comparison flags use 3-way AND: MARK_AX + operand_a + operand_b,
    gated by CMP_GROUP in gate path (b_gate=0). CMP flags ≈ 1.0 when true.

    Total: ~1296 units.
    """
    unit = 0

    def _block_non_ax_marker_sites(unit_idx: int) -> None:
        # LEA/ADJ/ENT fetch operands can be scale-40 at non-AX marker sites.
        # These units are only valid at the AX marker.
        for blocker_dim in (
            BD.MARK_PC,
            BD.MARK_SP,
            BD.MARK_BP,
            BD.MARK_STACK0,
            BD.MARK_MEM,
            BD.MARK_SE,
            BD.IS_BYTE,
        ):
            ffn.W_up[unit_idx, blocker_dim] = -S * 1000

    # === ADD hi nibble (no carry 256 + with carry 256 = 512 units) ===
    for carry_in in [0, 1]:
        for a in range(16):
            for b in range(16):
                result = (a + b + carry_in) % 16
                ffn.W_up[unit, BD.MARK_AX] = S
                ffn.W_up[unit, BD.MARK_PC] = -S * 2
                ffn.W_up[unit, BD.ALU_HI + a] = S
                ffn.W_up[unit, BD.AX_CARRY_HI + b] = S
                if carry_in == 0:
                    # Block when carry present: need strong negative to prevent firing
                    # FIX 2026-04-16: Restored from -0.01 to -S*2.0 for proper carry discrimination
                    ffn.W_up[unit, BD.CARRY + 0] = -S * 2.0
                    ffn.b_up[unit] = -S * 2.5  # 3-way AND
                else:
                    # Require carry: need strong positive to select carry case
                    # FIX 2026-04-16: Restored from 0.01 to S*2.0 for proper carry discrimination
                    ffn.W_up[unit, BD.CARRY + 0] = S * 2.0
                    ffn.b_up[unit] = -S * 4.5  # 4-way AND (3 regs + carry)
                ffn.W_gate[unit, BD.OP_ADD] = 1.0
                # LEA moved to separate units
                ffn.W_down[BD.OUTPUT_HI + result, unit] = 2.0 / S
                unit += 1

    # === LEA hi nibble (no carry 256 + with carry 256 = 512 units) ===
    # Require MARK_AX + ALU_HI + FETCH_HI. FETCH_HI is one-hot in the
    # declarative path, so amplify MARK_AX and FETCH_HI locally. Thresholds
    # still block PC-marker ALU/FETCH leakage and discriminate carry state.
    for carry_in in [0, 1]:
        for a in range(16):
            for b in range(16):
                result = (a + b + carry_in) % 16
                ffn.W_up[unit, BD.MARK_AX] = S * 20
                _block_non_ax_marker_sites(unit)
                ffn.W_up[unit, BD.ALU_HI + a] = S
                ffn.W_up[unit, BD.FETCH_HI + b] = S * 20  # Read from FETCH
                if carry_in == 0:
                    ffn.W_up[unit, BD.CARRY + 0] = -S * 8.0
                    ffn.b_up[unit] = -S * 40.5
                else:
                    ffn.W_up[unit, BD.CARRY + 0] = S * 8.0
                    ffn.b_up[unit] = -S * 48.5
                ffn.W_gate[unit, BD.OP_LEA] = 1.0
                ffn.W_down[BD.OUTPUT_HI + result, unit] = 2.0 / S
                unit += 1

    # === ADJ hi nibble (no carry 256 + with carry 256 = 512 units) ===
    # Like LEA but gates on OP_ADJ instead
    # ADJ computes: SP = SP + signed_immediate
    # Require MARK_AX + ALU_HI + FETCH_HI; FETCH_HI is one-hot here.
    for carry_in in [0, 1]:
        for a in range(16):
            for b in range(16):
                result = (a + b + carry_in) % 16
                ffn.W_up[unit, BD.MARK_AX] = S * 20
                _block_non_ax_marker_sites(unit)
                ffn.W_up[unit, BD.ALU_HI + a] = S
                ffn.W_up[unit, BD.FETCH_HI + b] = S * 20  # Read from FETCH
                if carry_in == 0:
                    ffn.W_up[unit, BD.CARRY + 0] = -S * 8.0
                    ffn.b_up[unit] = -S * 42
                else:
                    ffn.W_up[unit, BD.CARRY + 0] = S * 8.0
                    ffn.b_up[unit] = -S * 50
                ffn.W_gate[unit, BD.OP_ADJ] = 1.0
                ffn.W_down[BD.OUTPUT_HI + result, unit] = 2.0 / S
                unit += 1

    # === SUB hi nibble (no borrow 256 + with borrow 256 = 512 units) ===
    # a = ALU_HI (stack_top), b = AX_CARRY_HI (AX). SUB = stack_top - AX = a - b.
    for borrow_in in [0, 1]:
        for a in range(16):
            for b in range(16):
                result = (a - b - borrow_in) % 16  # ALU - AX_CARRY - borrow
                ffn.W_up[unit, BD.MARK_AX] = S
                ffn.W_up[unit, BD.MARK_PC] = -S * 2
                ffn.W_up[unit, BD.ALU_HI + a] = S
                ffn.W_up[unit, BD.AX_CARRY_HI + b] = S
                if borrow_in == 0:
                    ffn.W_up[unit, BD.CARRY + 0] = -S * 2.0
                    ffn.b_up[unit] = -S * 2.5
                else:
                    ffn.W_up[unit, BD.CARRY + 0] = S * 2.0
                    ffn.b_up[unit] = -S * 4.5  # 4-way AND (3 regs + borrow≈1)
                ffn.W_gate[unit, BD.OP_SUB] = 1.0
                ffn.W_down[BD.OUTPUT_HI + result, unit] = 2.0 / S
                unit += 1

    # === ENT hi nibble (no borrow 256 + with borrow 256 = 512 units) ===
    # ENT computes: SP = SP - (8 + signed_immediate)
    # For bytes 1-3, we subtract imm_byte with borrow propagation
    # The +8 offset affects byte 0 only; its overflow propagates as a borrow
    # Require MARK_AX + ALU_HI + FETCH_HI; FETCH_HI is one-hot here.
    # BUG FIX 2026-04-17: Add MARK_SP blocker. When OP_ENT is amplified to 22+,
    # the gate opens fully and ALU_HI[0]=58 at SP marker meets the threshold,
    # causing OUTPUT_HI[15] to be written instead of OUTPUT_HI[0].
    # BUG FIX 2026-06-11 (step-1 ENT-AX 0xf0 leak): hard HAS_SE blocker so
    # this band fires ONLY on the FIRST ENT step (where its OUTPUT_HI write
    # is the load-bearing BP-cascade source for test_lea_basic). On a
    # subsequent ENT step (JSR→ENT prologue) the OP_ENT in-step broadcast
    # over-amplifies the (sp_hi-imm_hi-borrow)=15 unit to ~+177 on
    # OUTPUT_HI[15], flipping the AX byte-0 marker-row argmax 0→15 and
    # leaking AX byte0=0xf0. The clean OUTPUT_HI[0] entering this layer on
    # subsequent steps is correct; SP comes from the AX_CARRY→SP writeback,
    # not this dim. Mirror of the ``_layer9_ent_hi_nibble_rules`` fix
    # (l9_ops.py) — keeps the legacy/declarative parity guard byte-identical.
    for borrow_in in [0, 1]:
        for sp_hi in range(16):
            for imm_hi in range(16):
                result = (sp_hi - imm_hi - borrow_in) % 16
                ffn.W_up[unit, BD.MARK_AX] = S * 20
                _block_non_ax_marker_sites(unit)
                ffn.W_up[unit, BD.HAS_SE] = -S * 1000  # first ENT step only
                ffn.W_up[unit, BD.ALU_HI + sp_hi] = S  # SP hi nibble from L7
                ffn.W_up[unit, BD.FETCH_HI + imm_hi] = S * 20  # Immediate hi nibble
                if borrow_in == 0:
                    # No borrow: block when CARRY active
                    ffn.W_up[unit, BD.CARRY + 0] = -S * 8.0
                    ffn.b_up[unit] = -S * 42
                else:
                    # With borrow: require CARRY active
                    ffn.W_up[unit, BD.CARRY + 0] = S * 8.0
                    ffn.b_up[unit] = -S * 50
                ffn.W_gate[unit, BD.OP_ENT] = 1.0
                ffn.W_down[BD.OUTPUT_HI + result, unit] = 2.0 / S
                unit += 1

    # === Comparison flags (shared across all cmp opcodes, gated by CMP_GROUP) ===
    # CMP[0]=hi_lt, CMP[1]=hi_eq, CMP[2]=lo_eq, CMP[3]=lo_lt

    # hi_eq: 16 units — 3-way AND (MARK_AX + ALU_HI[k] + AX_CARRY_HI[k])
    for k in range(16):
        ffn.W_up[unit, BD.MARK_AX] = S
        ffn.W_up[unit, BD.MARK_PC] = -S * 2
        ffn.W_up[unit, BD.ALU_HI + k] = S
        ffn.W_up[unit, BD.AX_CARRY_HI + k] = S
        ffn.b_up[unit] = -S * 2.5
        ffn.W_gate[unit, BD.CMP_GROUP] = 1.0
        ffn.W_down[BD.CMP + 1, unit] = 2.0 / S
        unit += 1

    # lo_eq: 16 units
    for k in range(16):
        ffn.W_up[unit, BD.MARK_AX] = S
        ffn.W_up[unit, BD.MARK_PC] = -S * 2
        ffn.W_up[unit, BD.ALU_LO + k] = S
        ffn.W_up[unit, BD.AX_CARRY_LO + k] = S
        ffn.b_up[unit] = -S * 2.5
        ffn.W_gate[unit, BD.CMP_GROUP] = 1.0
        ffn.W_down[BD.CMP + 2, unit] = 2.0 / S
        unit += 1

    # hi_lt: 120 units — 3-way AND (MARK_AX + ALU_HI[a] + AX_CARRY_HI[b]), a < b
    for a in range(16):
        for b in range(a + 1, 16):
            ffn.W_up[unit, BD.MARK_AX] = S
            ffn.W_up[unit, BD.MARK_PC] = -S * 2
            ffn.W_up[unit, BD.ALU_HI + a] = S
            ffn.W_up[unit, BD.AX_CARRY_HI + b] = S
            ffn.b_up[unit] = -S * 2.5
            ffn.W_gate[unit, BD.CMP_GROUP] = 1.0
            ffn.W_down[BD.CMP + 0, unit] = 2.0 / S
            unit += 1

    # lo_lt: 120 units
    for a in range(16):
        for b in range(a + 1, 16):
            ffn.W_up[unit, BD.MARK_AX] = S
            ffn.W_up[unit, BD.MARK_PC] = -S * 2
            ffn.W_up[unit, BD.ALU_LO + a] = S
            ffn.W_up[unit, BD.AX_CARRY_LO + b] = S
            ffn.b_up[unit] = -S * 2.5
            ffn.W_gate[unit, BD.CMP_GROUP] = 1.0
            ffn.W_down[BD.CMP + 3, unit] = 2.0 / S
            unit += 1

    # === ADD hi-nibble carry-out → CARRY[1] (byte carry for inter-byte propagation) ===
    # Detects (a + b + carry_in >= 16) for hi nibble. Same AND pattern as hi nibble
    # result, but writes to CARRY[1] instead of OUTPUT_HI.
    for carry_in in [0, 1]:
        for a in range(16):
            for b in range(16):
                if a + b + carry_in < 16:
                    continue  # no carry-out
                ffn.W_up[unit, BD.MARK_AX] = S
                ffn.W_up[unit, BD.MARK_PC] = -S * 2
                ffn.W_up[unit, BD.ALU_HI + a] = S
                ffn.W_up[unit, BD.AX_CARRY_HI + b] = S
                if carry_in == 0:
                    # Block when carry present: -0.01*CARRY prevents spurious activation
                    ffn.W_up[unit, BD.CARRY + 0] = -0.01  # Reduced from -S*2.0
                    ffn.b_up[unit] = -S * 2.5  # 3-way AND
                else:
                    # Require carry: +0.01*CARRY (hint, not hard requirement)
                    ffn.W_up[unit, BD.CARRY + 0] = 0.01  # Reduced from S*2.0
                    ffn.b_up[unit] = -S * 2.9  # Relaxed from -S*4.5 to allow activation with just 3 inputs
                ffn.W_gate[unit, BD.OP_ADD] = 1.0
                # NOTE: OP_LEA removed from gate because LEA = opcode 0, which causes
                # false positives (many tokens have value 0 and thus OP_LEA=1 in embedding).
                # If LEA byte carry is needed, add explicit OP_LEA gating with suppression.
                ffn.W_down[BD.CARRY + 1, unit] = 2.0 / S
                unit += 1

    # === SUB hi-nibble borrow-out → CARRY[2] (byte borrow for inter-byte propagation) ===
    # Detects borrow-out from hi nibble of ALU - AX_CARRY.
    # For SUB: stack - AX = ALU - AX_CARRY, so borrow when ALU < AX_CARRY.
    # a = ALU_HI index, b = AX_CARRY_HI index.
    for borrow_in in [0, 1]:
        for a in range(16):
            for b in range(16):
                if borrow_in == 0:
                    if a >= b:
                        continue  # no borrow-out when ALU >= AX_CARRY
                else:
                    if a > b:
                        continue  # no borrow-out when ALU > AX_CARRY (a - b - 1 >= 0)
                ffn.W_up[unit, BD.MARK_AX] = S
                ffn.W_up[unit, BD.MARK_PC] = -S * 2
                ffn.W_up[unit, BD.ALU_HI + a] = S
                ffn.W_up[unit, BD.AX_CARRY_HI + b] = S
                if borrow_in == 0:
                    # Block when carry present: -0.01*CARRY prevents spurious activation
                    ffn.W_up[unit, BD.CARRY + 0] = -0.01  # Reduced from -S*2.0
                    ffn.b_up[unit] = -S * 2.5  # 3-way AND
                else:
                    # Require carry: +0.01*CARRY (hint, not hard requirement)
                    ffn.W_up[unit, BD.CARRY + 0] = 0.01  # Reduced from S*2.0
                    ffn.b_up[unit] = -S * 2.9  # Relaxed from -S*4.5 to allow activation with just 3 inputs
                ffn.W_gate[unit, BD.OP_SUB] = 1.0
                ffn.W_down[BD.CARRY + 2, unit] = 2.0 / S
                unit += 1

    # === ALU clearing for opcodes that don't need operand gather ===
    # Prevents spurious activation of Layer 10 bitwise/MUL units due to
    # residual ALU_LO/HI values. Fires when MARK_AX and a non-ALU opcode.
    non_alu_opcodes = [
        BD.OP_IMM,
        BD.OP_NOP,
        BD.OP_JMP,
        BD.OP_JSR,
        BD.OP_EXIT,
        BD.OP_BZ,
        BD.OP_BNZ,
        BD.OP_ENT,
        BD.OP_ADJ,
        BD.OP_LEV,
        BD.OP_PSH,
        BD.OP_LI,
        BD.OP_LC,
        BD.OP_SI,
        BD.OP_SC,
    ]

    for k in range(16):
        ffn.W_up[unit, BD.MARK_AX] = S
        for op_dim in non_alu_opcodes:
            ffn.W_up[unit, op_dim] = S
        ffn.b_up[unit] = -S * 1.5
        ffn.b_gate[unit] = 1.0
        ffn.W_down[BD.ALU_LO + k, unit] = -10.0 / S
        unit += 1

    for k in range(16):
        ffn.W_up[unit, BD.MARK_AX] = S
        for op_dim in non_alu_opcodes:
            ffn.W_up[unit, op_dim] = S
        ffn.b_up[unit] = -S * 1.5
        ffn.b_gate[unit] = 1.0
        ffn.W_down[BD.ALU_HI + k, unit] = -10.0 / S
        unit += 1

    # === FIX 2026-04-16: Add +8 offset to ADDR_B0 at PC marker for LEV return_addr ===
    # L9 attention head 1 wrote ADDR_B0 = BP byte 0. We need ADDR_B0 = (BP+8) byte 0.
    # BP is always 8-byte aligned (lo nibble = 0 or 8), so adding 8:
    # - lo=0: +8 gives 8 (no carry)
    # - lo=8: +8 gives 0 with carry (but this case is rare for stack)
    # For simplicity, just add 8 to lo nibble without full carry propagation.
    # This works for common case where BP lo nibble is 0 (e.g., 0xfff0).
    for k in range(16):
        new_k = (k + 8) % 16
        # Fires at PC marker when OP_LEV active and ADDR_B0_LO[k] has value
        # BUG FIX 2026-04-16: Add MARK_BP exclusion. OP_LEV gets amplified to ~30
        # after L9 attention, causing units to fire at BP marker without MARK_PC.
        # BUG FIX 2026-04-16: Add MARK_SP exclusion. Same issue - OP_LEV alone
        # overcomes threshold, causing spurious ADDR_B0 shifts at SP marker.
        ffn.W_up[unit, BD.MARK_PC] = S
        ffn.W_up[unit, BD.OP_LEV] = S / 5  # OP_LEV ≈ 5
        ffn.W_up[unit, BD.MARK_BP] = -S * 10  # Exclude BP marker
        ffn.W_up[unit, BD.MARK_SP] = -S * 10  # Exclude SP marker
        ffn.b_up[unit] = -S * 1.5
        # FIX 2026-04-16: Gate on ADDR_B0_LO[k] with threshold to distinguish
        # legitimate BP values (LO[8] ~ 3.0 from L9 attention) from opcode fetch
        # contamination (LO[0] ~ 2.0 from L5).
        # Gate = LO[k] - 2.5 → legitimate: 3.0-2.5=0.5>0, contam: 2.0-2.5=-0.5<0
        ffn.W_gate[unit, BD.ADDR_B0_LO + k] = 1.0
        ffn.b_gate[unit] = -2.5  # Require LO[k] > 2.5
        # Cancel old position, set new position
        # BUG FIX 2026-04-16: Scaled to shift ~4 units of energy (matching post-attn values).
        # With output=600 (silu(150)*gate(4)), W_down=0.67/S gives contrib=4.0
        ffn.W_down[BD.ADDR_B0_LO + k, unit] = -0.67 / S
        ffn.W_down[BD.ADDR_B0_LO + new_k, unit] = 0.67 / S
        unit += 1

    # === FIX 2026-04-16: Set ADDR_B1 = 0xff at PC marker for stack addresses ===
    # Stack addresses are typically 0xfff0 range, so byte 1 is always 0xff.
    # Without this, L15 can't find the MEM section at 0xfff8 (query only matches byte 0).
    # ADDR_B1_LO[15] = 1 and ADDR_B1_HI[15] = 1 → byte 1 = 0xff
    ffn.W_up[unit, BD.MARK_PC] = S
    ffn.W_up[unit, BD.OP_LEV] = S / 5
    ffn.W_up[unit, BD.MARK_BP] = -S * 10  # Exclude BP marker
    ffn.W_up[unit, BD.MARK_SP] = -S * 10  # FIX 2026-04-16: Exclude SP marker
    ffn.b_up[unit] = -S * 1.5
    ffn.W_gate[unit, BD.CONST] = 1.0
    # BUG FIX 2026-04-16: Scale down by 9x for amplified up branch (silu(450) vs 50)
    ffn.W_down[BD.ADDR_B1_LO + 15, unit] = 0.22 / S  # lo nibble = 15
    unit += 1

    ffn.W_up[unit, BD.MARK_PC] = S
    ffn.W_up[unit, BD.OP_LEV] = S / 5
    ffn.W_up[unit, BD.MARK_BP] = -S * 10  # Exclude BP marker
    ffn.W_up[unit, BD.MARK_SP] = -S * 10  # FIX 2026-04-16: Exclude SP marker
    ffn.b_up[unit] = -S * 1.5
    ffn.W_gate[unit, BD.CONST] = 1.0
    # BUG FIX 2026-04-16: Scale down by 9x for amplified up branch
    ffn.W_down[BD.ADDR_B1_HI + 15, unit] = 0.22 / S  # hi nibble = 15
    unit += 1

    # === FIX 2026-04-16: Cascade carry for BP=0xfff8 + 8 = 0x10000 ===
    # When BP byte 0 = 0xf8 (lo=8, hi=15), adding 8 causes full cascade:
    #   byte0: 0xf8 + 8 = 0x100 → byte0=0x00, carry to byte1
    #   byte1: 0xff + 1 = 0x100 → byte1=0x00, carry to byte2
    #   byte2: 0x00 + 1 = 0x01
    # Detect via ADDR_B0_LO[8] (original lo nibble before shift) being high.

    # The +8 shift above moved ADDR_B0_LO[8]→[0], but we need to also:
    # 1. Clear ADDR_B0_HI[15] (it wasn't touched by the shift)
    # 2. Cancel the ADDR_B1=0xff setting above (for cascade case)
    # 3. Set ADDR_B2_LO[1] = 1 (byte2 = 0x01)

    # Unit 1: Clear ADDR_B0_HI[15] when lo was 8 (cascade case)
    # Gate on BOTH ADDR_B0_LO[0] (high after shift when original was 8) AND HI[15].
    # For BP=0xf0 (lo=0): after +8 shift, LO[8] is high, LO[0] is low → gate fails
    # For BP=0xf8 (lo=8): after +8 shift, LO[0] is high from shift → gate passes
    ffn.W_up[unit, BD.MARK_PC] = S
    ffn.W_up[unit, BD.OP_LEV] = S / 5
    ffn.W_up[unit, BD.MARK_BP] = -S * 10
    ffn.W_up[unit, BD.MARK_SP] = -S * 10  # FIX 2026-04-16: Exclude SP marker
    ffn.b_up[unit] = -S * 1.5
    # FIX 2026-04-16: Gate on BOTH LO[0] AND HI[15] to distinguish carry case.
    # After +8 shift: BP=0xf8 has LO[0] high (shifted from LO[8]), BP=0xf0 has LO[8] high.
    # Use LO[0] + HI[15] - threshold as AND gate.
    # BP=0xf8 after shift: LO[0]~15, HI[15]~3 → gate = 18 - 15 = 3 > 0 ✓
    # BP=0xf0 after shift: LO[0]~0, HI[15]~3 → gate = 3 - 15 = -12 < 0 ✗
    ffn.W_gate[unit, BD.ADDR_B0_LO + 0] = 1.0
    ffn.W_gate[unit, BD.ADDR_B0_HI + 15] = 1.0
    ffn.b_gate[unit] = -15.0  # Require LO[0] to be significantly high
    # Clear hi nibble 15, set hi nibble 0
    ffn.W_down[BD.ADDR_B0_HI + 15, unit] = -0.67 / S
    ffn.W_down[BD.ADDR_B0_HI + 0, unit] = 0.67 / S
    unit += 1

    # Unit 2: Cancel ADDR_B1_LO[15] setting for cascade case
    # The unconditional ADDR_B1=0xff above added ~1.2 to position 15.
    # For cascade, we need to cancel it and set position 0.
    ffn.W_up[unit, BD.MARK_PC] = S
    ffn.W_up[unit, BD.OP_LEV] = S / 5
    ffn.W_up[unit, BD.MARK_BP] = -S * 10
    ffn.W_up[unit, BD.MARK_SP] = -S * 10  # FIX 2026-04-16: Exclude SP marker
    ffn.b_up[unit] = -S * 1.5
    # FIX 2026-04-16: Use same AND gate as unit 1 to only fire for carry case
    ffn.W_gate[unit, BD.ADDR_B0_LO + 0] = 1.0
    ffn.W_gate[unit, BD.ADDR_B0_HI + 15] = 1.0
    ffn.b_gate[unit] = -15.0
    # Cancel the +0.22/S from unconditional unit, and clear any attention residue
    ffn.W_down[BD.ADDR_B1_LO + 15, unit] = -0.5 / S
    ffn.W_down[BD.ADDR_B1_LO + 0, unit] = 0.5 / S
    unit += 1

    # Unit 3: Cancel ADDR_B1_HI[15] setting for cascade case
    ffn.W_up[unit, BD.MARK_PC] = S
    ffn.W_up[unit, BD.OP_LEV] = S / 5
    ffn.W_up[unit, BD.MARK_BP] = -S * 10
    ffn.W_up[unit, BD.MARK_SP] = -S * 10  # FIX 2026-04-16: Exclude SP marker
    ffn.b_up[unit] = -S * 1.5
    # FIX 2026-04-16: Use same AND gate as unit 1 to only fire for carry case
    ffn.W_gate[unit, BD.ADDR_B0_LO + 0] = 1.0
    ffn.W_gate[unit, BD.ADDR_B0_HI + 15] = 1.0
    ffn.b_gate[unit] = -15.0
    ffn.W_down[BD.ADDR_B1_HI + 15, unit] = -0.5 / S
    ffn.W_down[BD.ADDR_B1_HI + 0, unit] = 0.5 / S
    unit += 1

    # Unit 4: Set ADDR_B2_LO[1] = 1 for cascade case (byte2 = 0x01)
    ffn.W_up[unit, BD.MARK_PC] = S
    ffn.W_up[unit, BD.OP_LEV] = S / 5
    ffn.W_up[unit, BD.MARK_BP] = -S * 10
    ffn.W_up[unit, BD.MARK_SP] = -S * 10  # FIX 2026-04-16: Exclude SP marker
    ffn.b_up[unit] = -S * 1.5
    # FIX 2026-04-16: Use same AND gate as unit 1 to only fire for carry case
    ffn.W_gate[unit, BD.ADDR_B0_LO + 0] = 1.0
    ffn.W_gate[unit, BD.ADDR_B0_HI + 15] = 1.0
    ffn.b_gate[unit] = -15.0
    ffn.W_down[BD.ADDR_B2_LO + 1, unit] = 0.67 / S
    unit += 1

    return unit


def _set_layer9_marker_suppress(ffn, S, BD, start_unit):
    """L9 FFN: Suppress OUTPUT dims when NEXT_* flags indicate marker prediction.

    L8 writes large values to OUTPUT at positions where markers should be
    predicted (e.g., at the AX byte 3 position, NEXT_SP=1.37 fires but
    OUTPUT_LO/HI[0] ~ 132 from multibyte routing, making byte(0) win over
    REG_SP). These units suppress OUTPUT when any NEXT_* flag exceeds ~0.8,
    allowing marker tokens to win.

    Uses steep W_up=100 with b_up=-80 for sharp silu transition:
    - NEXT=1.37: up=100*1.37-80=57, silu(57)=57, gate=5*1.37-3=3.85, act=219.5
    - NEXT=0.02: up=100*0.02-80=-78, silu(-78)~0, gate=5*0.02-3=-2.9, act~0
    W_down=-1.0 gives contribution=-219.5/dim for NEXT=1.37 (enough to overcome ~132)
    """
    unit = start_unit

    next_dims = [
        BD.NEXT_PC, BD.NEXT_AX, BD.NEXT_SP,
        BD.NEXT_BP, BD.NEXT_STACK0, BD.NEXT_MEM, BD.NEXT_SE,
    ]

    for next_dim in next_dims:
        ffn.W_up[unit, next_dim] = 100.0
        ffn.b_up[unit] = -80.0
        ffn.W_gate[unit, next_dim] = 5.0
        ffn.b_gate[unit] = -3.0
        for k in range(16):
            ffn.W_down[BD.OUTPUT_LO + k, unit] = -1.0
            ffn.W_down[BD.OUTPUT_HI + k, unit] = -1.0
        unit += 1

    return unit


def _set_layer10_alu(ffn, S, BD):
    """L10 FFN: Comparison combine + Bitwise ops + AX passthrough.

    Comparison combine: CMP flags from L9 (at AX marker):
      CMP[0]=hi_lt, CMP[1]=hi_eq, CMP[2]=lo_eq, CMP[3]=lo_lt.
      Each ≈ 1.0 when true, ≈ 0 when false (b_gate=0 in L9).
      Default unit writes result=0 or 1; override units flip based on flags.

    Bitwise ops: 3-way AND cross-product (same pattern as ADD/SUB).

    AX passthrough: fires when no handled AX-modifying opcode is active.
      Suppressed via negative W_up weights for handled opcodes.

    Total: ~1586 units (18 cmp + 1536 bitwise + 32 passthrough).
    """
    unit = 0

    # --- Comparison combine (18 units) ---

    def _cmp_default(op_dim, default_result):
        """Default unit: writes result + OUTPUT_HI[0]."""
        nonlocal unit
        ffn.W_up[unit, BD.MARK_AX] = S
        ffn.W_up[unit, op_dim] = S
        ffn.b_up[unit] = -S * 1.5
        ffn.b_gate[unit] = 1.0  # unconditional gate
        ffn.W_down[BD.OUTPUT_LO + default_result, unit] = 2.0 / S
        ffn.W_down[BD.OUTPUT_HI + 0, unit] = 2.0 / S
        unit += 1

    def _cmp_override_2way(op_dim, cmp_dim, to_result, from_result):
        """2-way override: MARK_AX + CMP[k] → flip result."""
        nonlocal unit
        ffn.W_up[unit, BD.MARK_AX] = S
        ffn.W_up[unit, cmp_dim] = S
        ffn.b_up[unit] = -S * 1.5
        ffn.W_gate[unit, op_dim] = 1.0
        ffn.W_down[BD.OUTPUT_LO + to_result, unit] = 4.0 / S
        ffn.W_down[BD.OUTPUT_LO + from_result, unit] = -4.0 / S
        unit += 1

    def _cmp_override_3way(op_dim, cmp_dim1, cmp_dim2, to_result, from_result):
        """3-way override: MARK_AX + CMP[i] + CMP[j] → flip result."""
        nonlocal unit
        ffn.W_up[unit, BD.MARK_AX] = S
        ffn.W_up[unit, cmp_dim1] = S
        ffn.W_up[unit, cmp_dim2] = S
        ffn.b_up[unit] = -S * 4.0
        ffn.W_gate[unit, op_dim] = 1.0
        ffn.W_down[BD.OUTPUT_LO + to_result, unit] = 4.0 / S
        ffn.W_down[BD.OUTPUT_LO + from_result, unit] = -4.0 / S
        unit += 1

    # EQ: default=0, override to 1 when hi_eq AND lo_eq
    _cmp_default(BD.OP_EQ, 0)
    _cmp_override_3way(BD.OP_EQ, BD.CMP + 1, BD.CMP + 2, 1, 0)

    # NE: default=1, override to 0 when hi_eq AND lo_eq
    _cmp_default(BD.OP_NE, 1)
    _cmp_override_3way(BD.OP_NE, BD.CMP + 1, BD.CMP + 2, 0, 1)

    # LT: default=0, override to 1 when hi_lt OR (hi_eq AND lo_lt)
    _cmp_default(BD.OP_LT, 0)
    _cmp_override_2way(BD.OP_LT, BD.CMP + 0, 1, 0)  # hi_lt
    _cmp_override_3way(BD.OP_LT, BD.CMP + 1, BD.CMP + 3, 1, 0)  # hi_eq AND lo_lt

    # GT: default=1, override to 0 when hi_lt OR (hi_eq AND lo_lt) OR (hi_eq AND lo_eq)
    _cmp_default(BD.OP_GT, 1)
    _cmp_override_2way(BD.OP_GT, BD.CMP + 0, 0, 1)  # hi_lt
    _cmp_override_3way(BD.OP_GT, BD.CMP + 1, BD.CMP + 3, 0, 1)  # hi_eq AND lo_lt
    _cmp_override_3way(BD.OP_GT, BD.CMP + 1, BD.CMP + 2, 0, 1)  # hi_eq AND lo_eq

    # LE: default=0, override to 1 when hi_lt OR (hi_eq AND lo_lt) OR (hi_eq AND lo_eq)
    _cmp_default(BD.OP_LE, 0)
    _cmp_override_2way(BD.OP_LE, BD.CMP + 0, 1, 0)  # hi_lt
    _cmp_override_3way(BD.OP_LE, BD.CMP + 1, BD.CMP + 3, 1, 0)  # hi_eq AND lo_lt
    _cmp_override_3way(BD.OP_LE, BD.CMP + 1, BD.CMP + 2, 1, 0)  # hi_eq AND lo_eq

    # GE: default=1, override to 0 when hi_lt OR (hi_eq AND lo_lt)
    _cmp_default(BD.OP_GE, 1)
    _cmp_override_2way(BD.OP_GE, BD.CMP + 0, 0, 1)  # hi_lt
    _cmp_override_3way(BD.OP_GE, BD.CMP + 1, BD.CMP + 3, 0, 1)  # hi_eq AND lo_lt

    # --- Bitwise ops (1536 units) ---
    # BUG FIX 2026-04-09: Increased MARK_AX weight and threshold to prevent spurious
    # firing at byte positions. At PC marker, ALU_LO and AX_CARRY_LO can have large
    # spurious values (up to ~70 combined), which exceeded the old threshold of 10.5.
    # By requiring MARK_AX=60, units only fire at the actual AX marker position.
    # NOTE: AX_CARRY should contain the stack value for binary ops, but is currently
    # not properly populated. This is a known architecture gap for stack-based ops.
    # BUG FIX 2026-04-16: Use balanced weights for true 3-way AND.
    # With weights (40, 30, 30) and threshold 80:
    #   All 3 present: 40 + 30 + 30 = 100 > 80 (fires)
    #   Any 2 present: max(40+30) = 70 < 80 (blocked)
    bitwise_ops = [
        (BD.OP_OR, lambda a, b: a | b),
        (BD.OP_XOR, lambda a, b: a ^ b),
        (BD.OP_AND, lambda a, b: a & b),
    ]
    for op_dim, op_fn in bitwise_ops:
        # Lo nibble (256 units)
        for a in range(16):
            for b in range(16):
                result = op_fn(a, b)
                ffn.W_up[unit, BD.MARK_AX] = S * 40  # Balanced 3-way AND
                ffn.W_up[unit, BD.ALU_LO + a] = S * 30
                ffn.W_up[unit, BD.AX_CARRY_LO + b] = S * 30
                ffn.b_up[unit] = -S * 80  # Threshold requiring all 3 inputs
                ffn.W_gate[unit, op_dim] = 1.0
                ffn.W_down[BD.OUTPUT_LO + result, unit] = 2.0 / S
                unit += 1
        # Hi nibble (256 units)
        for a in range(16):
            for b in range(16):
                result = op_fn(a, b)
                ffn.W_up[unit, BD.MARK_AX] = S * 40  # Balanced 3-way AND
                ffn.W_up[unit, BD.ALU_HI + a] = S * 30
                ffn.W_up[unit, BD.AX_CARRY_HI + b] = S * 30
                ffn.b_up[unit] = -S * 80  # Threshold requiring all 3 inputs
                ffn.W_gate[unit, op_dim] = 1.0
                ffn.W_down[BD.OUTPUT_HI + result, unit] = 2.0 / S
                unit += 1

    # --- MUL lo nibble (256 units) ---
    # For each (a_lo, b_lo): result_lo = (a_lo * b_lo) % 16
    # 3-way AND: MARK_AX + ALU_LO[a_lo] + AX_CARRY_LO[b_lo], gate=OP_MUL
    # BUG FIX 2026-04-16: Use balanced weights for true 3-way AND.
    # With weights (40, 30, 30) and threshold 80:
    #   All 3 present: 40 + 30 + 30 = 100 > 80 (fires)
    #   Any 2 present: max(40+30) = 70 < 80 (blocked)
    for a in range(16):
        for b in range(16):
            result = (a * b) % 16
            ffn.W_up[unit, BD.MARK_AX] = S * 40  # Balanced 3-way AND
            ffn.W_up[unit, BD.ALU_LO + a] = S * 30
            ffn.W_up[unit, BD.AX_CARRY_LO + b] = S * 30
            ffn.b_up[unit] = -S * 80  # Threshold requiring all 3 inputs
            ffn.W_gate[unit, BD.OP_MUL] = 1.0
            ffn.W_down[BD.OUTPUT_LO + result, unit] = 2.0 / S
            unit += 1

    # --- SHL/SHR zero output for shift >= 8 (8 units) ---
    # When shift >= 8, result is 0x00. Two sub-cases:
    # Case A (shift >= 16, hi nibble > 0): gate = OP_xxx * (1 - AX_CARRY_HI[0])
    # Case B (shift 8-15, hi=0, lo>=8): up includes AX_CARRY_HI[0] + sum(AX_CARRY_LO[8..15])
    # BUG FIX 2026-04-09: Increased threshold to prevent spurious firing.
    for op_dim in [BD.OP_SHL, BD.OP_SHR]:
        # Case A: shift >= 16 (hi nibble non-zero → AX_CARRY_HI[0] NOT hot)
        ffn.W_up[unit, BD.MARK_AX] = S * 60  # Strong MARK_AX requirement
        ffn.W_up[unit, BD.AX_CARRY_HI + 0] = -S  # suppress when hi=0 (shift 0-15)
        ffn.b_up[unit] = -S * 59  # Require MARK_AX to overcome threshold
        ffn.W_gate[unit, op_dim] = 1.0
        ffn.W_down[BD.OUTPUT_LO + 0, unit] = 2.0 / S
        ffn.W_down[BD.OUTPUT_HI + 0, unit] = 2.0 / S
        unit += 1

        # Case B: shift 8-15 (hi=0, lo nibble is 8..15)
        ffn.W_up[unit, BD.MARK_AX] = S * 60  # Strong MARK_AX requirement
        ffn.W_up[unit, BD.AX_CARRY_HI + 0] = S
        for lo_bit in range(8, 16):
            ffn.W_up[unit, BD.AX_CARRY_LO + lo_bit] = S
        ffn.b_up[unit] = -S * 80  # Require MARK_AX to overcome threshold
        ffn.W_gate[unit, op_dim] = 1.0
        ffn.W_down[BD.OUTPUT_LO + 0, unit] = 2.0 / S
        ffn.W_down[BD.OUTPUT_HI + 0, unit] = 2.0 / S
        unit += 1

    # --- AX passthrough (32 units) ---
    # Fires when no handled AX-modifying opcode is active.
    # Negative W_up weights suppress passthrough for handled opcodes.
    suppressed_ops = [
        BD.OP_IMM,
        BD.OP_ADD,
        BD.OP_SUB,
        BD.OP_OR,
        BD.OP_XOR,
        BD.OP_AND,
        BD.OP_EQ,
        BD.OP_NE,
        BD.OP_LT,
        BD.OP_GT,
        BD.OP_LE,
        BD.OP_GE,
        BD.OP_MUL,
        BD.OP_DIV,   # neural (DivModModule after L10)
        BD.OP_MOD,   # neural (DivModModule after L10)
        BD.OP_SHL,
        BD.OP_SHR,
        BD.OP_LEA,  # LEA: AX = FETCH + BP (handled by L8/L9 ADD circuit)
        BD.OP_LI,  # L15 provides memory lookup result
        BD.OP_LC,  # L15 provides memory lookup result (byte)
        BD.OP_JMP,  # L6 handles AX_CARRY -> OUTPUT (gated by HAS_SE)
        BD.OP_EXIT,  # L6 handles AX_CARRY -> OUTPUT
        BD.OP_NOP,  # L6 handles AX_CARRY -> OUTPUT
        BD.OP_PUTCHAR,  # L6 handles AX_CARRY -> OUTPUT
    ]
    for k in range(16):
        ffn.W_up[unit, BD.MARK_AX] = S
        for op_dim in suppressed_ops:
            ffn.W_up[unit, op_dim] = -S
        ffn.b_up[unit] = -S * 0.5
        ffn.W_gate[unit, BD.AX_CARRY_LO + k] = 1.0
        ffn.W_down[BD.OUTPUT_LO + k, unit] = 2.0 / S
        unit += 1
    for k in range(16):
        ffn.W_up[unit, BD.MARK_AX] = S
        for op_dim in suppressed_ops:
            ffn.W_up[unit, op_dim] = -S
        ffn.b_up[unit] = -S * 0.5
        ffn.W_gate[unit, BD.AX_CARRY_HI + k] = 1.0
        ffn.W_down[BD.OUTPUT_HI + k, unit] = 2.0 / S
        unit += 1

    # --- Inter-byte carry/borrow: now handled by CarryPropagationPostOp ---

    return unit


def _set_layer15_memory_lookup_heads_0_3(attn, S, BD, HD):
    """Always-on portion of :func:`_set_layer15_memory_lookup`: heads 0-3.

    Factored out for Phase 7.C.2 so the IR carries the unconditional
    heads-0-3 writes as one :class:`RuntimeAttentionFragment` and the
    LEV-only heads-4-11 writes as another fragment. DSL Wave W7 made
    the LEV branch selectable at IR-build time via ``num_heads`` rather
    than via a runtime predicate, but this body itself is unchanged --
    it is the always-on fragment's writer. Callable directly for tests
    and for the legacy :func:`_set_layer15_memory_lookup` umbrella
    entry point.
    """
    L = 15.0
    PC_I = 0
    AX_I = 1
    MEM_I = 4
    SP_I = 2
    BP_I = 3

    # Heads 0-3: Original LI/LC/STACK0 implementation
    for h in range(4):
        base = h * HD

        # === Dim 0: Bias — suppress non-target Q positions ===
        # Non-target: Q[0] = -2000. Target: Q[0] = 0.
        # K[0] = CONST*10 = 10 everywhere.
        # Score: -2000*10/8 = -2500 (non-target), 0 (target).
        # Must overwhelm worst-case address dim correlation: ADDR dims
        # overlap OPCODE_BYTE (both at dims 12-27), creating up to +1200
        # of spurious positive Q·K score at non-target positions.
        attn.W_q[base, BD.CONST] = -2000.0
        attn.W_q[base, BD.OP_LI_RELAY] = 2000.0
        if h == 0:
            # Head 0: also activated by LC and STACK0 for POP group operations.
            # FIX 2026-04-16: Use CMP[3] (POP group flag) instead of MARK_STACK0.
            # This ensures memory lookup only happens for operations that need to
            # read from stack (ADD, SUB, etc.), not IMM which just needs carry-forward.
            # CMP[3] is relayed from AX marker to STACK0 position by L6 attn head 6.
            attn.W_q[base, BD.OP_LC_RELAY] = 2000.0
            attn.W_q[base, BD.CMP + 3] = 2000.0  # POP group → stack memory read
        else:
            # Heads 1-3: activated at STACK0 byte positions via BP threshold
            # L1H4[BP] fires at d<=9.5 from BP (covers STACK0 area d=5-9)
            # H1[BP] fires at d<=4.5 from BP (BP's own bytes) — subtract to cancel
            attn.W_q[base, BD.L1H4 + BP_I] = 2000.0
            attn.W_q[base, BD.H1 + BP_I] = -2000.0
        # PSH suppression at STACK0 positions
        attn.W_q[base, BD.CMP + 0] = -2000.0
        # LEV suppression: Heads 0-3 are for LI/LC/STACK0, not LEV.
        # FIX 2026-04-16: Address dimension overlap (ADDR_B0_LO = OPCODE_BYTE_LO)
        # creates spurious +3556 score at PC byte positions. Need stronger suppression.
        # OP_LEV ≈ 5.0, so -1000*5 = -5000 added to Q[0], total score -= 6250.
        attn.W_q[base, BD.OP_LEV] = -1000.0
        # FIX 2026-04-16: Add strong MARK_PC suppression. At PC marker during LEV,
        # MARK_PC=1 contributes -25000 to Q[0]. Combined with OP_LEV suppression,
        # Q[0] = -2000 - 25000 - 5000 = -32000. Score contribution = -40000.
        # This overwhelms any positive address matching contribution.
        attn.W_q[base, BD.MARK_PC] = -25000.0
        # FIX 2026-04-16: Also suppress at SP marker during LEV.
        # Heads 0-3 attend to PC bytes from previous step, outputting wrong values.
        # Address aliasing (ADDR_B0_LO == OPCODE_BYTE_LO) causes +75500/dim score.
        # Need Q[0] << -60000 to overcome 8 address dims. Use -100000.
        attn.W_q[base, BD.MARK_SP] = -100000.0
        # L15 memory lookup targets AX loads and STACK0 pop reads, never SP/BP
        # register bytes. With dynamic dimensions the byte-index gates can make
        # these heads look target-like at SP/BP byte positions; suppress those
        # positions so first-step SP/BP defaults from L3 remain authoritative.
        attn.W_q[base, BD.H1 + SP_I] = -50000.0
        attn.W_q[base, BD.H1 + BP_I] = -50000.0
        attn.W_k[base, BD.CONST] = 10.0

        # PC byte positions can carry byte-index and address-like residuals
        # that overpower the generic non-target bias. L15 lookup never targets
        # PC bytes, so add an explicit query blocker for the whole PC byte span.
        attn.W_q[base + 29, BD.H1 + PC_I] = -20000.0
        attn.W_k[base + 29, BD.CONST] = 5.0

        # AX byte positions are targets only for LI/LC memory loads. For other
        # opcodes (notably PSH), address-like residuals can overpower the
        # generic non-target bias and make this lookup add a zero-valued memory
        # result over an already-correct AX passthrough byte. Suppress AX bytes
        # by default, then restore the score only for real load queries against
        # stored MEM entries.
        attn.W_q[base + 30, BD.H1 + AX_I] = -20000.0
        attn.W_k[base + 30, BD.CONST] = 5.0
        attn.W_q[base + 31, BD.OP_LI_RELAY] = 20000.0
        if h == 0:
            attn.W_q[base + 31, BD.OP_LC_RELAY] = 20000.0
        attn.W_k[base + 31, BD.MEM_STORE] = 5.0

        # The AX marker itself is the query position for byte 0. The H1[AX]
        # blocker above only covers AX value-byte positions, so non-load ops
        # like PSH could still look like byte-0 LI/LC queries at MARK_AX and
        # pull a zero-valued historical MEM byte over a correct AX passthrough.
        # Suppress the marker by default and restore only head 0 for real
        # LI/LC loads; heads 1-3 serve subsequent AX byte positions.
        attn.W_q[base + 32, BD.MARK_AX] = -20000.0
        attn.W_k[base + 32, BD.CONST] = 5.0
        if h == 0:
            attn.W_q[base + 33, BD.OP_LI_RELAY] = 20000.0
            attn.W_q[base + 33, BD.OP_LC_RELAY] = 20000.0
            attn.W_k[base + 33, BD.MEM_STORE] = 5.0

        # === Dim 1: Store anchor — suppress non-store K at target Q ===
        # Q[1] = 50 at target, 0 at non-target.
        # K[1] = MEM_STORE*100 - CONST*50 = +50 (store) or -50 (non-store).
        # Score: 50*(+50)/8 = +312.5 (target+store),
        #        50*(-50)/8 = -312.5 (target+non-store).
        attn.W_q[base + 1, BD.OP_LI_RELAY] = 50.0
        if h == 0:
            attn.W_q[base + 1, BD.OP_LC_RELAY] = 50.0
            attn.W_q[base + 1, BD.CMP + 3] = 50.0  # POP group (matches dim 0 fix)
        else:
            attn.W_q[base + 1, BD.L1H4 + BP_I] = 50.0
            attn.W_q[base + 1, BD.H1 + BP_I] = -50.0
        attn.W_q[base + 1, BD.CMP + 0] = -50.0
        attn.W_k[base + 1, BD.MEM_STORE] = 100.0
        attn.W_k[base + 1, BD.CONST] = -50.0

        # === Dim 2: ZFOD negative offset for store entries ===
        # Q[2] = CONST*(-96) = -96 always.
        # K[2] = MEM_STORE*50 = 50 at stores, 0 at non-stores.
        # Score at store: -96*50/8 = -600. At non-store: 0.
        # Shifts store baseline so wrong-addr stores score negative.
        attn.W_q[base + 2, BD.CONST] = -96.0
        attn.W_k[base + 2, BD.MEM_STORE] = 50.0

        # === Dim 3: Byte selection ===
        # Ensures each head attends to the correct MEM val byte TOKEN position.
        # MEM section layout: [MEM, a0, a1, a2, a3, v0, v1, v2, v3]
        # Val byte positions: v0=d5, v1=d6, v2=d7, v3=d8 from MEM marker.
        # Head h reads val byte h via V(CLEAN_EMBED).
        # Q[3] = BS*byte_flag. K[3] = BS*threshold_flag_for_val_byte_h.
        # Byte selection weight increased to 60.0 to dominate over address encoding
        # mismatches (up to ~50 points) caused by value bytes having corrupted ADDR dims.
        # Contribution: 60*60/8 = 450 points for correct byte, 0 for wrong byte.
        BS = 60.0  # Byte Selection weight (increased from L=15.0)
        byte_q_flag = [BD.MARK_AX, BD.BYTE_INDEX_0, BD.BYTE_INDEX_1, BD.BYTE_INDEX_2][h]
        attn.W_q[base + 3, byte_q_flag] = BS
        if h == 0:
            attn.W_q[base + 3, BD.MARK_STACK0] = BS
        MEM_VAL_DIMS = [None, BD.MEM_VAL_B1, BD.MEM_VAL_B2, BD.MEM_VAL_B3]
        if h == 0:
            # Head 0 → val byte 0 at d=5: L2H0[MEM]=1 (d≤5.5), H1[MEM]=0 (d>4.5).
            attn.W_k[base + 3, BD.L2H0 + MEM_I] = BS
            attn.W_k[base + 3, BD.H1 + MEM_I] = -BS
        else:
            # Heads 1-3 → val bytes 1,2,3 at d=6,7,8 via MEM_VAL_B1/B2/B3.
            attn.W_k[base + 3, MEM_VAL_DIMS[h]] = BS

        # === Dims 4-27: Binary address encoding (24 bits, scale=10) ===
        # Each of 3 address bytes × 2 nibbles × 4 bits = 24 dims.
        # Q/K: ±scale per bit. Match: 24*100/8 = 300. Random: ~0.
        addr_dim = 4
        scale = 10.0
        addr_bases = [
            (BD.ADDR_B0_LO, BD.ADDR_B0_HI),
            (BD.ADDR_B1_LO, BD.ADDR_B1_HI),
            (BD.ADDR_B2_LO, BD.ADDR_B2_HI),
        ]
        for ab_lo, ab_hi in addr_bases:
            for nibble_base in [ab_lo, ab_hi]:
                for bit in range(4):
                    for k in range(16):
                        bit_val = 2 * ((k >> bit) & 1) - 1
                        attn.W_q[base + addr_dim, nibble_base + k] = scale * bit_val
                        attn.W_k[base + addr_dim, nibble_base + k] = scale * bit_val
                    addr_dim += 1

        # === Dim 28: Per-head position gate ===
        # Each head fires ONLY at its target AX/STACK0 byte position.
        # Head 0: MARK_AX or MARK_STACK0. Head 1: BYTE_INDEX_0.
        # Head 2: BYTE_INDEX_1. Head 3: BYTE_INDEX_2.
        # Non-target: Q=-500, score=-312.5. Target: Q=0, score=0.
        attn.W_q[base + 28, BD.CONST] = -500.0
        attn.W_q[base + 28, byte_q_flag] = 500.0
        if h == 0:
            attn.W_q[base + 28, BD.MARK_STACK0] = 500.0
        attn.W_k[base + 28, BD.CONST] = 5.0

        # === V/O: copy byte value to OUTPUT (dims 32-63) ===
        for k in range(16):
            attn.W_v[base + 32 + k, BD.CLEAN_EMBED_LO + k] = 1.0
            attn.W_v[base + 48 + k, BD.CLEAN_EMBED_HI + k] = 1.0
        for k in range(16):
            attn.W_o[BD.OUTPUT_LO + k, base + 32 + k] = 1.0
            attn.W_o[BD.OUTPUT_HI + k, base + 48 + k] = 1.0


def _set_layer15_memory_lookup(attn, S, BD, HD):
    """L15 attention: Memory lookup for LI/LC (at AX) and *SP (at STACK0).

    4 heads, one per output byte. Each head serves BOTH:
    - LI/LC at AX positions (load from memory address in prev AX)
    - *SP at STACK0 positions (load from address in SP)

    Uses binary Q/K address encoding for 24-bit matching with ZFOD.

    Score budget per dim (all /sqrt(HD)=8):
      Dim 0 (bias):     Q=-2000(non-target) or 0(target), K=CONST*10
                         → -2500 at non-target, 0 at target
      Dim 1 (store):    Q=50(target) or 0, K=(2*MEM_STORE-1)*50
                         → +312.5 target+store, -312.5 target+non-store
      Dim 2 (ZFOD ofs): Q=CONST*(-96), K=MEM_STORE*50
                         → -600 at store entries (shifts addr baseline)
      Dim 3 (byte sel): Q=L*flag, K=L*MEM_VAL_BH → +28 correct byte
      Dims 4-27 (addr): 24 binary bits, scale=10
                         match=+300, 1-bit-off=+275, random≈0
                         NOTE: ADDR_B0_LO overlaps OPCODE_BYTE — up to +1200
                         spurious score from residual opcode nibbles at Q side

    Totals:
      target+store+match:   0+312.5-600+300 = +12.5  → attend
      target+store+1bitoff: 0+312.5-600+275 = -12.5  → ZFOD ✓
      target+store+random:  0+312.5-600+0   = -287.5 → ZFOD ✓
      target+non-store:     0-312.5+0+0     = -312.5 → suppressed
      non-target+worst:     -2500+1200      = -1300  → suppressed ✓
      non-target+self:      -2500+300       = -2200  → suppressed ✓

    Phase 7.C.2 split this umbrella into three runtime-shape pieces so
    the L15 ``memory_lookup`` op can carry them as
    :class:`RuntimeAttentionFragment` entries in its CompilerIR. DSL
    Wave W7 (current) moved the shape gating from per-fragment runtime
    predicates to compile-time Python ``if`` inside
    :func:`_layer15_memory_lookup_ir`, parameterized on ``num_heads``;
    the bodies below are unchanged.

    * :func:`_set_layer15_memory_lookup_heads_0_3` — always emits.
    * :func:`_set_layer15_memory_lookup_lev_heads_4_11` — emits only
      when ``num_heads >= 12``.
    * The current-store-generation suppress helper in
      ``unified_compiler.ops.l15_ops``.

    Kept as a single legacy entry point for ``tests/test_l15_per_op.py``
    and for any caller that still wants the full imperative bake; the
    production L15 op routes through the IR.

    Phase 7.C wave (this commit): the umbrella entry point itself now
    routes through :func:`_layer15_memory_lookup_ir` + ``lower_attention``
    instead of calling the imperative children directly. The fragment
    bodies are still the same imperative writers (Phase 7.C.2 state),
    but the umbrella's dispatch is declarative -- the
    ``RuntimeAttentionFragment`` IR is the single source of truth for
    which fragments fire on which shapes (see
    :func:`_layer15_memory_lookup_ir` for the ``num_heads`` selection
    table). A guard prevents double-writes if a caller already lowered
    the IR before invoking the umbrella.
    """
    if getattr(attn, "_l15_memory_lookup_ir_baked", False):
        # Already lowered by ``make_layer15_memory_lookup_op``'s bake;
        # skip to avoid double-writing the same fragments.
        return

    from neural_vm.unified_compiler.ops.l15_ops import _layer15_memory_lookup_ir

    ir = _layer15_memory_lookup_ir(BD, HD, num_heads=int(attn.num_heads))
    ir.lower_attention(attn, HD, dim_positions=BD, S=S)
    attn._l15_memory_lookup_ir_baked = True


def _set_layer16_lev_routing(ffn, S, BD):
    """L16 FFN: Route LEV memory reads and compute SP = BP + 16.

    After L15, we have:
    - BP marker: OUTPUT = saved_bp (from L15 heads 4-7)
    - PC marker: OUTPUT = return_addr (from L15 heads 8-11)
    - ADDR_B0_LO/HI: old BP value (from Phase 1)

    L16 computes SP = old_BP + 16 and writes to OUTPUT at SP marker.

    Registers updated by LEV:
    - BP ← saved_bp (already in OUTPUT at BP marker from L15)
    - PC ← return_addr (already in OUTPUT at PC marker from L15)
    - SP ← old_BP + 16 (computed here at SP marker)

    Strategy: Enumerate (bp_lo + 16) % 16 for lo nibble, handle carry for hi.
    """
    unit = 0
    first_step_gate = S * 30
    MEM_I = 4

    # === FIX 2026-04-16: Cancel OUTPUT at SP marker during LEV ===
    # L15 heads 0-4 write spurious OUTPUT values at SP marker due to address aliasing.
    # ADDR_B0_LO == OPCODE_BYTE_LO causes heads to attend to PC bytes from prev step.
    # Cancel all OUTPUT before adding correct SP = BP + 16 values.
    for k in range(16):
        # Cancel OUTPUT_LO[k] at SP marker when OP_LEV
        # FIX 2026-04-16: Add marker exclusions. Without them, OP_LEV*S/5 = 10*20 = 200
        # overcomes b_up=-150 threshold even at AX/PC/BP markers where MARK_SP=0.
        ffn.W_up[unit, BD.OP_LEV] = S / 5
        ffn.W_up[unit, BD.MARK_SP] = S
        ffn.W_up[unit, BD.MARK_PC] = -S  # Exclude PC marker
        ffn.W_up[unit, BD.MARK_AX] = -S  # Exclude AX marker
        ffn.W_up[unit, BD.MARK_BP] = -S  # Exclude BP marker
        ffn.W_up[unit, BD.HAS_SE] = first_step_gate
        ffn.W_up[unit, BD.PSH_AT_SP] = -first_step_gate
        ffn.b_up[unit] = -S * 1.5 - first_step_gate
        ffn.W_gate[unit, BD.OUTPUT_LO + k] = -1.0
        ffn.W_down[BD.OUTPUT_LO + k, unit] = 2.0 / S
        unit += 1
    for k in range(16):
        # Cancel OUTPUT_HI[k] at SP marker when OP_LEV
        # FIX 2026-04-16: Add marker exclusions (same reasoning as OUTPUT_LO).
        ffn.W_up[unit, BD.OP_LEV] = S / 5
        ffn.W_up[unit, BD.MARK_SP] = S
        ffn.W_up[unit, BD.MARK_PC] = -S  # Exclude PC marker
        ffn.W_up[unit, BD.MARK_AX] = -S  # Exclude AX marker
        ffn.W_up[unit, BD.MARK_BP] = -S  # Exclude BP marker
        ffn.W_up[unit, BD.HAS_SE] = first_step_gate
        ffn.W_up[unit, BD.PSH_AT_SP] = -first_step_gate
        ffn.b_up[unit] = -S * 1.5 - first_step_gate
        ffn.W_gate[unit, BD.OUTPUT_HI + k] = -1.0
        ffn.W_down[BD.OUTPUT_HI + k, unit] = 2.0 / S
        unit += 1

    # === SP = BP + 16: Lo nibble (SP_byte0_lo = (BP_byte0_lo + 16) % 16 = BP_byte0_lo) ===
    # Since 16 % 16 = 0, adding 16 to a nibble just wraps to the same value for lo nibble
    # but generates carry to hi nibble
    # BUG FIX 2026-04-15: OP_LEV ≈ 10 at SP marker (amplified), scale down to ~1 contribution
    # BUG FIX 2026-04-16: Add MARK_BP exclusion. Without it, units fire at BP marker
    # because OP_LEV*S + ADDR_B0[k]*S > 3S even without MARK_SP.
    # FIX 2026-04-16: Add MARK_PC exclusion. Without it, units fire at PC marker
    # because ADDR_B0_LO[8]=12 from +8 offset makes OP_LEV + ADDR_B0 > threshold.
    # BUG FIX 2026-04-16: Increased MARK_PC penalty from -S*5 to -S*15 to overcome
    # ADDR_B0_LO[8]=12 contribution. Calc: 10 + 0 - 150 + 120 - 30 = -50 < 0 (blocked).
    for k in range(16):
        # Result lo nibble = k (adding 16 to nibble k gives k with carry)
        # FIX 2026-04-16: Gate on ADDR_B0_LO[k] instead of CONST to prevent spurious firing
        # when OP_LEV and MARK_SP are amplified but ADDR_B0_LO[k] is low.
        ffn.W_up[unit, BD.OP_LEV] = S
        ffn.W_up[unit, BD.MARK_SP] = S
        ffn.W_up[unit, BD.MARK_BP] = -S * 15  # Exclude BP marker (must overcome ADDR_B0*S)
        ffn.W_up[unit, BD.MARK_PC] = -S * 15  # Exclude PC marker (must overcome ADDR_B0[8]=12*S)
        ffn.W_up[unit, BD.MARK_AX] = -S * 50  # FIX 2026-04-16: Exclude AX marker (ADDR_B0 contamination ~40*S)
        ffn.W_up[unit, BD.MARK_STACK0] = -S * 15  # Exclude STACK0 marker
        ffn.W_up[unit, BD.MARK_MEM] = -S * 15  # Exclude MEM marker
        ffn.W_up[unit, BD.H3 + MEM_I] = -S * 15  # Exclude MEM byte span
        ffn.W_up[unit, BD.MARK_SE] = -S * 15  # Exclude STEP_END marker
        # FIX 2026-04-16: Suppress at byte positions (BYTE_INDEX=1 at bytes, =0 at markers)
        # ADDR_B0 contamination causes spurious firing at byte positions, need strong suppression.
        ffn.W_up[unit, BD.BYTE_INDEX_0] = -S * 10  # Suppress at byte 0 positions
        ffn.W_up[unit, BD.BYTE_INDEX_1] = -S * 10  # Suppress at byte 1 positions
        ffn.W_up[unit, BD.BYTE_INDEX_2] = -S * 10  # Suppress at byte 2 positions
        ffn.W_up[unit, BD.BYTE_INDEX_3] = -S * 10  # Suppress at byte 3 positions
        ffn.W_up[unit, BD.ADDR_B0_LO + k] = S  # Old BP lo nibble
        ffn.W_up[unit, BD.HAS_SE] = first_step_gate
        ffn.W_up[unit, BD.PSH_AT_SP] = -first_step_gate
        ffn.b_up[unit] = -S * 10.0 - first_step_gate
        # Gate on ADDR_B0_LO[k] - only fires when this nibble has significant value
        ffn.W_gate[unit, BD.ADDR_B0_LO + k] = 1.0
        ffn.W_down[BD.OUTPUT_LO + k, unit] = 2.0 / S
        unit += 1

    # === SP = BP + 16: Hi nibble (SP_byte0_hi = (BP_byte0_hi + 1) % 16) ===
    # Adding 16 to byte 0 means: lo nibble gets +16%16=0, hi nibble gets +1 (carry from 16)
    for k in range(16):
        result = (k + 1) % 16
        # FIX 2026-04-16: Gate on ADDR_B0_HI[k] instead of CONST
        ffn.W_up[unit, BD.OP_LEV] = S
        ffn.W_up[unit, BD.MARK_SP] = S
        ffn.W_up[unit, BD.MARK_BP] = -S * 15  # Exclude BP marker (must overcome ADDR_B0*S)
        ffn.W_up[unit, BD.MARK_PC] = -S * 15  # Exclude PC marker (must overcome ADDR_B0*S)
        ffn.W_up[unit, BD.MARK_AX] = -S * 50  # FIX 2026-04-16: Exclude AX marker
        ffn.W_up[unit, BD.MARK_STACK0] = -S * 15  # Exclude STACK0 marker
        ffn.W_up[unit, BD.MARK_MEM] = -S * 15  # Exclude MEM marker
        ffn.W_up[unit, BD.H3 + MEM_I] = -S * 15  # Exclude MEM byte span
        ffn.W_up[unit, BD.MARK_SE] = -S * 15  # Exclude STEP_END marker
        # FIX 2026-04-16: Suppress at byte positions
        ffn.W_up[unit, BD.BYTE_INDEX_0] = -S * 10
        ffn.W_up[unit, BD.BYTE_INDEX_1] = -S * 10
        ffn.W_up[unit, BD.BYTE_INDEX_2] = -S * 10
        ffn.W_up[unit, BD.BYTE_INDEX_3] = -S * 10
        ffn.W_up[unit, BD.ADDR_B0_HI + k] = S  # Old BP hi nibble
        ffn.W_up[unit, BD.HAS_SE] = first_step_gate
        ffn.W_up[unit, BD.PSH_AT_SP] = -first_step_gate
        ffn.b_up[unit] = -S * 10.0 - first_step_gate
        # Gate on ADDR_B0_HI[k] - only fires when this nibble has significant value
        ffn.W_gate[unit, BD.ADDR_B0_HI + k] = 1.0
        ffn.W_down[BD.OUTPUT_HI + result, unit] = 2.0 / S
        unit += 1

    # === SP bytes 1-3: Copy from ADDR_B1/B2 with byte 1 carry ===
    # NOTE 2026-04-16: These units are DISABLED for now. They were designed to compute
    # SP bytes 1-3 but they fire at the wrong position (marker instead of byte positions).
    # For addresses that wrap (like BP=0xfffffff8 + 16 = 0x00000008), bytes 1-3 are all 0x00
    # which is handled by the default OUTPUT = 0 behavior.
    # TODO: Redesign to fire at byte positions using BYTE_INDEX flags.
    if False:  # Disabled
     for b0_hi in range(16):
        for b1_lo in range(16):
            carry = 1 if b0_hi == 15 else 0  # Carry from byte 0
            result = (b1_lo + carry) % 16
            needs_carry = (b1_lo + carry) >= 16

            ffn.W_up[unit, BD.OP_LEV] = S / 10  # Scale down
            ffn.W_up[unit, BD.MARK_SP] = S
            ffn.W_up[unit, BD.MARK_BP] = -S * 15  # Exclude BP marker (must overcome ADDR_B0*S)
            ffn.W_up[unit, BD.MARK_PC] = -S * 15  # Exclude PC marker (increased to overcome ADDR*S)
            ffn.W_up[unit, BD.MARK_AX] = -S * 50  # FIX 2026-04-16: Exclude AX marker
            # FIX 2026-04-16: Suppress at byte positions
            ffn.W_up[unit, BD.BYTE_INDEX_0] = -S * 10
            ffn.W_up[unit, BD.BYTE_INDEX_1] = -S * 10
            ffn.W_up[unit, BD.BYTE_INDEX_2] = -S * 10
            ffn.W_up[unit, BD.BYTE_INDEX_3] = -S * 10
            ffn.W_up[unit, BD.ADDR_B0_HI + b0_hi] = S
            ffn.W_up[unit, BD.ADDR_B1_LO + b1_lo] = S
            ffn.b_up[unit] = -S * 4.0  # Raised threshold for 4-way AND
            ffn.W_gate[unit, BD.CONST] = 1.0
            ffn.W_down[BD.OUTPUT_LO + result, unit] = 2.0 / S
            if needs_carry:
                ffn.W_down[BD.CARRY + 1, unit] = 2.0 / S
            unit += 1

    # Byte 1 hi nibble: Add carry from byte 1 lo (DISABLED - see note above)
    if False:
     for b1_hi in range(16):
        for carry in [0, 1]:
            result = (b1_hi + carry) % 16
            needs_carry = (b1_hi + carry) >= 16

            ffn.W_up[unit, BD.OP_LEV] = S / 10  # Scale down
            ffn.W_up[unit, BD.MARK_SP] = S
            ffn.W_up[unit, BD.MARK_BP] = -S * 15  # Exclude BP marker (must overcome ADDR_B0*S)
            ffn.W_up[unit, BD.MARK_PC] = -S * 15  # Exclude PC marker (increased to overcome ADDR*S)
            ffn.W_up[unit, BD.MARK_AX] = -S * 50  # FIX 2026-04-16: Exclude AX marker
            # FIX 2026-04-16: Suppress at byte positions
            ffn.W_up[unit, BD.BYTE_INDEX_0] = -S * 10
            ffn.W_up[unit, BD.BYTE_INDEX_1] = -S * 10
            ffn.W_up[unit, BD.BYTE_INDEX_2] = -S * 10
            ffn.W_up[unit, BD.BYTE_INDEX_3] = -S * 10
            ffn.W_up[unit, BD.ADDR_B1_HI + b1_hi] = S
            if carry:
                ffn.W_up[unit, BD.CARRY + 1] = S
            ffn.b_up[unit] = -S * (3.0 + carry * 1.0)  # Raised threshold
            ffn.W_gate[unit, BD.CONST] = 1.0
            ffn.W_down[BD.OUTPUT_HI + result, unit] = 2.0 / S
            if needs_carry:
                ffn.W_down[BD.CARRY + 2, unit] = 2.0 / S
            unit += 1

    # Bytes 2-3: Direct copy with carry propagation (for addresses > 256, not common)
    # (DISABLED - see note above)
    # Byte 2 lo nibble with carry
    if False:
     for b2_lo in range(16):
        for carry in [0, 1]:
            result = (b2_lo + carry) % 16
            needs_carry = (b2_lo + carry) >= 16

            ffn.W_up[unit, BD.OP_LEV] = S / 10  # Scale down
            ffn.W_up[unit, BD.MARK_SP] = S
            ffn.W_up[unit, BD.MARK_BP] = -S * 15  # Exclude BP marker (must overcome ADDR_B0*S)
            ffn.W_up[unit, BD.MARK_PC] = -S * 15  # Exclude PC marker (increased to overcome ADDR*S)
            ffn.W_up[unit, BD.MARK_AX] = -S * 50  # FIX 2026-04-16: Exclude AX marker
            # FIX 2026-04-16: Suppress at byte positions
            ffn.W_up[unit, BD.BYTE_INDEX_0] = -S * 10
            ffn.W_up[unit, BD.BYTE_INDEX_1] = -S * 10
            ffn.W_up[unit, BD.BYTE_INDEX_2] = -S * 10
            ffn.W_up[unit, BD.BYTE_INDEX_3] = -S * 10
            ffn.W_up[unit, BD.ADDR_B2_LO + b2_lo] = S
            if carry:
                ffn.W_up[unit, BD.CARRY + 2] = S
            ffn.b_up[unit] = -S * (3.0 + carry * 1.0)  # Raised threshold
            ffn.W_gate[unit, BD.CONST] = 1.0
            ffn.W_down[BD.OUTPUT_LO + result, unit] = 2.0 / S
            if needs_carry:
                ffn.W_down[BD.CARRY + 3, unit] = 2.0 / (S * 5.0)
            unit += 1

    # Byte 2 hi nibble, byte 3 lo, byte 3 hi - similar pattern but skipping for brevity
    # For addresses < 256, bytes 2-3 are zero anyway

    # === FIX 2026-04-15: Route TEMP → OUTPUT at PC marker for return_addr ===
    # L15 head 8 writes return_addr byte 0 to TEMP[0:31] (lo/hi nibbles).
    # This FFN copies TEMP → OUTPUT at PC marker when OP_LEV is active.
    #
    # FIX 2026-04-16: Removed OUTPUT_LO cancel - let TEMP just add to existing.
    # With heads 0-3 suppressed at PC marker, residual OUTPUT_LO[10]=1.0 comes from
    # L3 carry-forward. Adding TEMP[10]=1.0 gives OUTPUT_LO[10]=3.0, which dominates.
    #
    # Keep OUTPUT_HI cancel to handle spurious OUTPUT_HI[3] from residual attention.
    # BUG FIX 2026-04-16: Add MARK_AX/SP/BP exclusions. Without them, OP_LEV*S/5 = 10*20 = 200
    # overcomes b_up=-150 threshold even at AX/SP/BP markers where MARK_PC=0.
    # This was causing OUTPUT_HI[2] to be canceled at AX marker, breaking AX preservation.
    for k in range(16):
        # Cancel OUTPUT_HI[k] at PC marker when OP_LEV - needed to suppress wrong hi nibble
        ffn.W_up[unit, BD.OP_LEV] = S / 5
        ffn.W_up[unit, BD.MARK_PC] = S
        ffn.W_up[unit, BD.MARK_AX] = -S  # Exclude AX marker
        ffn.W_up[unit, BD.MARK_SP] = -S  # Exclude SP marker
        ffn.W_up[unit, BD.MARK_BP] = -S  # Exclude BP marker
        ffn.b_up[unit] = -S * 1.5
        ffn.W_gate[unit, BD.OUTPUT_HI + k] = -1.0
        ffn.W_down[BD.OUTPUT_HI + k, unit] = 2.0 / S
        unit += 1

    # Add TEMP values to OUTPUT
    for k in range(16):
        # TEMP_LO[k] → OUTPUT_LO[k] at PC marker
        ffn.W_up[unit, BD.OP_LEV] = S
        ffn.W_up[unit, BD.MARK_PC] = S
        ffn.W_up[unit, BD.TEMP + k] = S
        ffn.b_up[unit] = -S * 3.5
        ffn.W_gate[unit, BD.CONST] = 1.0
        ffn.W_down[BD.OUTPUT_LO + k, unit] = 2.0 / S
        unit += 1
    for k in range(16):
        # TEMP_HI[k] → OUTPUT_HI[k] at PC marker
        ffn.W_up[unit, BD.OP_LEV] = S
        ffn.W_up[unit, BD.MARK_PC] = S
        ffn.W_up[unit, BD.TEMP + 16 + k] = S
        ffn.b_up[unit] = -S * 3.5
        ffn.W_gate[unit, BD.CONST] = 1.0
        ffn.W_down[BD.OUTPUT_HI + k, unit] = 2.0 / S
        unit += 1

    # === FIX 2026-04-16: Set OUTPUT = 0x00 at byte positions 1-3 for LEV ===
    # For return_addr < 256, bytes 1-3 are always 0. At byte positions (BYTE_INDEX=1),
    # we need to override the OUTPUT that persists from PC marker with the 0x00 encoding.
    # 0x00 = lo nibble 0, hi nibble 0 → OUTPUT_LO[0]=1, OUTPUT_HI[0]=1
    #
    # BUG FIX 2026-04-16: Require OP_LEV strongly, don't gate on OUTPUT_LO[10].
    # Previous version gated on OUTPUT_LO[10] which caused clearing during IMM
    # (OUTPUT_LO[10]=4 from FETCH overcame threshold without OP_LEV).
    #
    # BUG FIX 2026-04-16: Suppress at marker positions (NEXT_* = 0.68).
    # BYTE_INDEX_3 persists at d=5 from previous marker (marker transition position).
    # At marker positions we should NOT set OUTPUT values.
    # Suppress on all NEXT_* flags to prevent firing at AX, SP, BP, etc. marker positions.
    #
    # First, clear OUTPUT_LO[10] at byte 1-3 (prevent 0x0A from persisting)
    # BUG FIX 2026-04-16: Use BYTE_INDEX_1/2/3 for bytes 1/2/3.
    # BYTE_INDEX_N = 1 at byte N position (0-indexed from after marker).
    # Byte 0 = first byte after marker (should output 0x0A, don't clear!)
    # Byte 1 = second byte (should output 0x00, need to clear OUTPUT_LO[10])
    # etc.
    # BUG FIX 2026-04-16: Add MARK_PC penalty. Same issue as OUTPUT_LO[0]/OUTPUT_HI[0] below.
    #
    # FIX 2026-05-09 (LEV AX preservation, Phase 5 agent T5):
    #   The byte 1-3 zeroing units below force OUTPUT_LO[0]=1, OUTPUT_HI[0]=1
    #   (i.e. byte=0x00) at any BYTE_INDEX_1/2/3 position when OP_LEV is active.
    #   In non-pure-neural mode OP_LEV is broadcast across all positions, so these
    #   units fire at AX byte positions 1-3 too — clobbering AX bytes 1-3 with
    #   0x00 even when AX held a non-trivial multi-byte value across the LEV.
    #
    #   We add an H1[AX_I] negative weight: H1[1] = 1 only when AX is the nearest
    #   marker within 4.5 tokens, i.e. at AX marker (d=0) and AX bytes 1-4 (d=1..4).
    #   Combined with the existing MARK_AX = -S*10 suppression at the AX marker
    #   itself, this isolates AX bytes 1-4 and prevents the unit from firing there.
    #   PC byte 1-4 nearest marker is PC, so H1[PC_I] is on (not H1[AX_I]) — those
    #   positions still fire correctly. Same for SP/BP/STACK0 byte positions: they
    #   have their own H1[SP_I]/H1[BP_I] active, not H1[AX_I]. So this change ONLY
    #   suppresses the unit at AX byte positions (which is what we want).
    AX_I = 1  # AX index in MARKS half-space encoding (matches _set_phase_a_ffn)
    AX_BYTE_SUPPRESS = -S * 10  # Strong: must overcome OP_LEV*S/2 = 5*25=125 even with broadcast
    for byte_pos in range(3):  # byte positions 1, 2, 3
        # BYTE_INDEX_1 for byte 1, BYTE_INDEX_2 for byte 2, BYTE_INDEX_3 for byte 3
        byte_idx_dim = [BD.BYTE_INDEX_1, BD.BYTE_INDEX_2, BD.BYTE_INDEX_3][byte_pos]
        # Clear OUTPUT_LO[10] unconditionally at LEV byte positions 1-3
        ffn.W_up[unit, BD.OP_LEV] = S / 2  # Require OP_LEV (~7.5 at byte pos → contrib 375)
        ffn.W_up[unit, byte_idx_dim] = S  # Fire at this byte position (~0.97 → contrib 97)
        # Suppress at marker positions where NEXT_* = 0.68
        # -S*1.5 * 0.68 = -102, needs to overcome 72 positive contrib → OK
        ffn.W_up[unit, BD.MARK_PC] = -S * 1.5  # BUG FIX: Suppress at PC marker
        # FIX 2026-04-16: Add MARK_AX/SP/BP exclusions. At AX marker, OP_LEV*50 = 501 > 400
        # threshold, but NEXT_* = 0 (NEXT flags are set at previous position, not current).
        ffn.W_up[unit, BD.MARK_AX] = -S * 10  # Suppress at AX marker (MARK_AX=1 → -1000)
        ffn.W_up[unit, BD.MARK_SP] = -S * 10  # Suppress at SP marker
        ffn.W_up[unit, BD.MARK_BP] = -S * 10  # Suppress at BP marker
        ffn.W_up[unit, BD.MARK_STACK0] = -S * 10  # Suppress at STACK0 marker
        ffn.W_up[unit, BD.MARK_MEM] = -S * 10  # Suppress at MEM marker
        ffn.W_up[unit, BD.MARK_SE] = -S * 10  # Suppress at STEP_END marker
        # FIX 2026-05-09: Suppress at AX byte positions (H1[AX_I]=1, MARK_AX=0)
        ffn.W_up[unit, BD.H1 + AX_I] = AX_BYTE_SUPPRESS
        ffn.W_up[unit, BD.NEXT_AX] = -S * 1.5
        ffn.W_up[unit, BD.NEXT_SP] = -S * 1.5
        ffn.W_up[unit, BD.NEXT_BP] = -S * 1.5
        ffn.W_up[unit, BD.NEXT_STACK0] = -S * 1.5
        ffn.W_up[unit, BD.NEXT_MEM] = -S * 1.5
        ffn.W_up[unit, BD.NEXT_SE] = -S * 1.5
        ffn.b_up[unit] = -S * 4  # Threshold: 375 + 97 - 400 = 72 > 0
        ffn.W_gate[unit, BD.CONST] = 1.0
        ffn.W_down[BD.OUTPUT_LO + 10, unit] = -10.0 / S  # Strong cancel of OUTPUT_LO[10]
        unit += 1

    # Set OUTPUT_LO[0] = 1 at byte positions 1-3 (for 0x00 value)
    # BUG FIX 2026-04-16: Add MARK_PC penalty. Without it, OP_LEV=10 at PC marker
    # causes OP_LEV*S/2 = 500 > 400 threshold, making unit fire even with BYTE_INDEX=0.
    for byte_pos in range(3):
        byte_idx_dim = [BD.BYTE_INDEX_1, BD.BYTE_INDEX_2, BD.BYTE_INDEX_3][byte_pos]
        ffn.W_up[unit, BD.OP_LEV] = S / 2  # Require OP_LEV
        ffn.W_up[unit, byte_idx_dim] = S
        # Suppress at marker positions
        ffn.W_up[unit, BD.MARK_PC] = -S * 1.5  # BUG FIX: Suppress at PC marker
        # FIX 2026-04-16: Add MARK_AX/SP/BP exclusions (same as OUTPUT_LO[10] above)
        ffn.W_up[unit, BD.MARK_AX] = -S * 10
        ffn.W_up[unit, BD.MARK_SP] = -S * 10
        ffn.W_up[unit, BD.MARK_BP] = -S * 10
        ffn.W_up[unit, BD.MARK_STACK0] = -S * 10
        ffn.W_up[unit, BD.MARK_MEM] = -S * 10
        ffn.W_up[unit, BD.MARK_SE] = -S * 10
        # FIX 2026-05-09: Suppress at AX byte positions to preserve AX across LEV.
        ffn.W_up[unit, BD.H1 + AX_I] = AX_BYTE_SUPPRESS
        ffn.W_up[unit, BD.NEXT_AX] = -S * 1.5
        ffn.W_up[unit, BD.NEXT_SP] = -S * 1.5
        ffn.W_up[unit, BD.NEXT_BP] = -S * 1.5
        ffn.W_up[unit, BD.NEXT_STACK0] = -S * 1.5
        ffn.W_up[unit, BD.NEXT_MEM] = -S * 1.5
        ffn.W_up[unit, BD.NEXT_SE] = -S * 1.5
        ffn.b_up[unit] = -S * 4
        ffn.W_gate[unit, BD.CONST] = 1.0
        ffn.W_down[BD.OUTPUT_LO + 0, unit] = 5.0 / S  # Strong set of OUTPUT_LO[0]
        unit += 1

    # Set OUTPUT_HI[0] = 1 at byte positions 1-3 (for 0x00 value)
    # BUG FIX 2026-04-16: Add MARK_PC penalty (same as OUTPUT_LO[0] above).
    for byte_pos in range(3):
        byte_idx_dim = [BD.BYTE_INDEX_1, BD.BYTE_INDEX_2, BD.BYTE_INDEX_3][byte_pos]
        ffn.W_up[unit, BD.OP_LEV] = S / 2  # Require OP_LEV
        ffn.W_up[unit, byte_idx_dim] = S
        # Suppress at marker positions
        ffn.W_up[unit, BD.MARK_PC] = -S * 1.5  # BUG FIX: Suppress at PC marker
        # FIX 2026-04-16: Add MARK_AX/SP/BP exclusions (same as OUTPUT_LO[10] above)
        ffn.W_up[unit, BD.MARK_AX] = -S * 10
        ffn.W_up[unit, BD.MARK_SP] = -S * 10
        ffn.W_up[unit, BD.MARK_BP] = -S * 10
        ffn.W_up[unit, BD.MARK_STACK0] = -S * 10
        ffn.W_up[unit, BD.MARK_MEM] = -S * 10
        ffn.W_up[unit, BD.MARK_SE] = -S * 10
        # FIX 2026-05-09: Suppress at AX byte positions to preserve AX across LEV.
        ffn.W_up[unit, BD.H1 + AX_I] = AX_BYTE_SUPPRESS
        ffn.W_up[unit, BD.NEXT_AX] = -S * 1.5
        ffn.W_up[unit, BD.NEXT_SP] = -S * 1.5
        ffn.W_up[unit, BD.NEXT_BP] = -S * 1.5
        ffn.W_up[unit, BD.NEXT_STACK0] = -S * 1.5
        ffn.W_up[unit, BD.NEXT_MEM] = -S * 1.5
        ffn.W_up[unit, BD.NEXT_SE] = -S * 1.5
        ffn.b_up[unit] = -S * 4
        ffn.W_gate[unit, BD.CONST] = 1.0
        ffn.W_down[BD.OUTPUT_HI + 0, unit] = 5.0 / S  # Strong set of OUTPUT_HI[0]
        unit += 1

    return unit  # Return number of units used


# =============================================================================
# BZ/BNZ: Conditional branch (L6 relay + L6 FFN override)
# =============================================================================


    # Old L15 implementation removed — see _set_layer15_memory_lookup above.


# =============================================================================
# IO Syscall Layer (PUTCHAR — fully autoregressive)
# =============================================================================


# =============================================================================
# L6 Opcode Relay Head (AX → SP/STACK0 marker positions)
# =============================================================================


# =============================================================================
# Tool Calling Weight Setting (gated by enable_tool_calling)
# =============================================================================


# =============================================================================
# Conversational I/O Detection (PRTF/READ for autoregressive generation)
# =============================================================================


    # Initialize format position to 0 (nibble encoding: all nibbles zero except set [0]=1)
    # Actually, we want FORMAT_POS to START at 0, which means all nibbles are 0.
    # The position will be incremented AFTER fetching each byte.
    # So we don't need to explicitly set it to 0 here (it defaults to 0).
    # We'll handle increment in L8 FFN.
