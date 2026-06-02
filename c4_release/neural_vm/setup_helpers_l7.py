"""Setup helpers extracted from ``setup_helpers.py`` (Phase 6 wave 1.5).

Layer 7 helpers: operand gather + V18 convo-IO PRTF capture.

Re-exported by ``setup_helpers`` for backward compatibility.
"""

import math

from .constants import PC_OFFSET


def _set_layer7_operand_gather(attn, S, BD, HD):
    """L7 attention: Operand A gather for binary operations.

    Head 0: At AX marker, read previous step's STACK0 byte 0 → ALU_LO/ALU_HI.
    This provides operand A (stack top) for binary ops (ADD, SUB, etc.).

    STACK0 byte 0 is identified by STACK0_BYTE0 flag (from L1 FFN).
    Distance from current AX marker to prev step's STACK0 byte 0:
      AX marker at position 5 in current step.
      STACK0 byte 0 at position 21 in prev step.
      Distance = 35 - 21 + 5 = 19 tokens.

    Head 1: LEA operand gather — BP OUTPUT → ALU at AX marker.
    LEA computes AX = FETCH + BP. Head 1 copies BP's output to ALU_LO/HI
    at AX marker (only when OP_LEA active via Q gating).
    Distance from AX(pos 5) to BP(pos 15 in prev step) = 35 - 15 + 5 = 25.
    With slope=0.5: score = 15^2*0.125 - 0.5*25 = 28.125 - 12.5 = 15.625.

    ALiBi slope should favor d=19 strongly.
    """
    L = 15.0

    # Head 0: AX ← prev STACK0 byte 0 (STACK0_BYTE0 key)
    base = 0 * HD
    attn.W_q[base, BD.MARK_AX] = L
    attn.W_q[base, BD.OP_LEA] = -L  # suppress STACK0→ALU for LEA
    attn.W_k[base, BD.STACK0_BYTE0] = L
    # Anti-leakage gate: keep the head silent at AX byte positions. Without
    # this, q=0 at bytes still averages historical STACK0 values into ALU.
    attn.W_q[base + 33, BD.MARK_AX] = L
    attn.W_q[base + 33, BD.CONST] = -L / 2
    attn.W_k[base + 33, BD.CONST] = L
    # V: copy CLEAN_EMBED_LO/HI from STACK0 byte 0 (pristine, not inflated)
    for k in range(16):
        attn.W_v[base + 1 + k, BD.CLEAN_EMBED_LO + k] = 1.0
        attn.W_v[base + 17 + k, BD.CLEAN_EMBED_HI + k] = 1.0
    # O: write to ALU_LO/ALU_HI at AX marker
    # Amplified to 6.0 to overcome L6 FFN clear (~-5.0), giving net +1.0.
    for k in range(16):
        attn.W_o[BD.ALU_LO + k, base + 1 + k] = 6.0
        attn.W_o[BD.ALU_HI + k, base + 17 + k] = 6.0

    # Head 1: LEA/ADJ/ENT — BP/SP OUTPUT → ALU at AX marker
    # LEA: fires when OP_LEA active, gathers BP
    # ADJ: fires when OP_ADJ active, gathers SP
    # ENT: fires when OP_ENT active, gathers SP (for SP -= 8+imm computation)
    #
    # IMPORTANT: When all opcodes inactive, we need Q < 0 so that softmax1
    # gives near-zero attention. Q=0 would give ~50% weight at d=0 (where
    # exp(0)=1 in softmax1 denominator), causing significant leakage!
    #
    # FIX 2026-04-17: Add MARK_AX gating. With global OP_ENT injection (at all positions),
    # this head was firing at all positions (including PC marker), causing ALU_LO/HI
    # to be written everywhere. The L8 carry computation then explodes at PC marker.
    # Now: Q[0] requires MARK_AX + opcode, with baseline suppression.
    # At AX marker with OP_ENT=5: Q[0] = 150 + 75 - 75 = 150
    # At PC marker with OP_ENT=5: Q[0] = 0 + 75 - 75 = 0
    base = 1 * HD
    attn.W_q[base, BD.MARK_AX] = L * 10  # Strong AX marker requirement
    attn.W_q[base, BD.OP_LEA] = L  # fires when LEA active
    attn.W_q[base, BD.OP_ADJ] = L  # fires when ADJ active
    attn.W_q[base, BD.OP_ENT] = L  # fires when ENT active
    attn.W_q[base, BD.CONST] = -L * 5  # Baseline suppression (cancel OP_* at non-AX)
    # Anti-leakage gate dimension: suppresses when not at AX marker
    attn.W_q[base + 1, BD.CONST] = -L * 2  # -30 baseline
    attn.W_q[base + 1, BD.MARK_AX] = L * 3  # +45 at AX marker → net +15
    attn.W_k[base + 1, BD.CONST] = 1.0  # K[1] = 1 everywhere
    attn.W_k[base, BD.MARK_BP] = L  # attends to BP (for LEA)
    attn.W_k[base, BD.MARK_SP] = L  # attends to SP (for ADJ/ENT)
    # V: copy OUTPUT_LO/HI (BP's or SP's byte-0 output from L6)
    for k in range(16):
        attn.W_v[base + 1 + k, BD.OUTPUT_LO + k] = 1.0
        attn.W_v[base + 17 + k, BD.OUTPUT_HI + k] = 1.0
    # O: write to ALU_LO/ALU_HI at AX marker
    for k in range(16):
        attn.W_o[BD.ALU_LO + k, base + 1 + k] = 1.0
        attn.W_o[BD.ALU_HI + k, base + 17 + k] = 1.0



def _set_convo_io_prtf_capture(ffn, S, BD):
    """L7 FFN addition (V18 Phase 1b, bake 3c): capture PC/SP at PRTF AX marker.

    The companion to ``_set_convo_io_pc_sp_latch`` (3b). At the PRTF AX
    marker (``ACTIVE_OPCODE_PRTF`` AND ``MARK_AX``), this band decomposes
    the current step's PC and SP byte-0 nibbles into the dedicated cache
    dims that the 3b replay band reads at the resumed step's REG_PC /
    REG_SP value-byte positions.

    Source dims at the AX marker position (after L4 / L7 attention has
    run):
      - PC byte 0 nibbles: ``EMBED_LO/HI`` — set by L4 attn head 0
        (``_set_layer4_pc_relay``), which attends to MARK_PC and copies
        the carry-forwarded PC byte 0 nibbles.
      - SP byte 0 nibbles: ``ADDR_B0_HI`` (lo nibble) and ``ADDR_B1_HI``
        (hi nibble) — set by L4 attn heads 2/3
        (``make_layer4_sp_to_addr_key_op``), which attend to SP byte 0
        and copy CLEAN_EMBED nibbles into the ADDR_KEY band.

    Targets (V18 Phase 1b cache dims, declared in ``_SetDim``):
      - ``POST_PRTF_PC_LO/HI`` — aliases AX_FULL_LO/HI (471/487). At the
        PRTF AX marker AX_FULL is normally being staged for PSH AX of the
        same step; PRTF never PSHes AX so the slot is dead.
      - ``POST_PRTF_SP_LO/HI`` — aliases AX_CARRY_LO/HI (328/344). The
        ALU divisor slot; PRTF is not an ALU op so AX_CARRY is dead.

    Unit layout (L7 FFN, starting at unit 800, ending at unit 863):
      800..815  PC lo nibble capture (16 units, one per nibble value k)
      816..831  PC hi nibble capture
      832..847  SP lo nibble capture
      848..863  SP hi nibble capture

    Unit 800 is chosen to sit well above the L7 operand-gather range
    (typically <100 units), leaving room for growth.

    This bake is gated off by default (``enable=False`` in the factory
    ``make_convo_io_prtf_capture_op``); flip to True only when the
    end-to-end neural convo-IO loop is validated.
    """
    # (src lo nibble dim, src hi nibble dim, dst cache lo, dst cache hi)
    capture_groups = [
        (BD.EMBED_LO,   BD.EMBED_HI,   BD.POST_PRTF_PC_LO, BD.POST_PRTF_PC_HI),
        (BD.ADDR_B0_HI, BD.ADDR_B1_HI, BD.POST_PRTF_SP_LO, BD.POST_PRTF_SP_HI),
    ]
    unit = 800
    for src_lo, src_hi, dst_lo, dst_hi in capture_groups:
        for src_dim, dst_dim in ((src_lo, dst_lo), (src_hi, dst_hi)):
            for k in range(16):
                ffn.W_up[unit, BD.ACTIVE_OPCODE_PRTF] = S
                ffn.W_up[unit, BD.MARK_AX] = S
                ffn.b_up[unit] = -S * 1.5  # require both flags hot
                ffn.W_gate[unit, src_dim + k] = 1.0  # one-hot source bit
                ffn.W_down[dst_dim + k, unit] = 2.0 / S
                unit += 1
