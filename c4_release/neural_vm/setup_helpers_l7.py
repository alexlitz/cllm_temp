"""Setup helpers extracted from ``setup_helpers.py`` (Phase 6 wave 1.5).

Layer 7 helpers: operand gather + V18 convo-IO PRTF capture.

Re-exported by ``setup_helpers`` for backward compatibility.
"""

import math

from .constants import PC_OFFSET


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
