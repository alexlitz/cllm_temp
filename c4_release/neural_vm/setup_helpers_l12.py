"""Setup helpers extracted from ``setup_helpers.py`` (Phase 6 wave 1.5).

Layer 12 helpers: MUL combine.

Re-exported by ``setup_helpers`` for backward compatibility.
"""

import math

from .constants import PC_OFFSET


def _set_layer12_mul_combine(ffn, S, BD):
    """L12 FFN: MUL hi nibble from partial + a_hi*b_lo.

    Reads TEMP[partial] from L11 (≈5.0 when hot, ≈0 otherwise).
    Computes: result_hi = (partial + a_hi * b_lo) % 16.

    4-way AND: MARK_AX + TEMP[partial] + ALU_HI[a_hi] + AX_CARRY_LO[b_lo]
    Threshold 7.5 accounts for TEMP ≈ 5.0 (not 1.0):
      All match: 1 + 5 + 1 + 1 = 8 > 7.5 → fires
      Wrong TEMP: 1 + 0 + 1 + 1 = 3 < 7.5 → blocked
      Wrong ALU/AX: 1 + 5 + 0 + 1 = 7 < 7.5 → blocked

    4096 units = 16^3 (fills L12 FFN exactly).
    """
    unit = 0

    for partial in range(16):
        for a_hi in range(16):
            for b_lo in range(16):
                result_hi = (partial + a_hi * b_lo) % 16
                ffn.W_up[unit, BD.MARK_AX] = S
                ffn.W_up[unit, BD.TEMP + partial] = S
                ffn.W_up[unit, BD.ALU_HI + a_hi] = S
                ffn.W_up[unit, BD.AX_CARRY_LO + b_lo] = S
                ffn.b_up[unit] = -S * 7.5
                ffn.W_gate[unit, BD.OP_MUL] = 1.0
                ffn.W_down[BD.OUTPUT_HI + result_hi, unit] = 2.0 / S
                unit += 1

    return unit



