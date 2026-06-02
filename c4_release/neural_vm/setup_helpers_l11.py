"""Setup helpers extracted from ``setup_helpers.py`` (Phase 6 wave 1.5).

Layer 11 helpers: MUL partial products.

Re-exported by ``setup_helpers`` for backward compatibility.
"""

import math

from .constants import PC_OFFSET


def _set_layer11_mul_partial(ffn, S, BD):
    """L11 FFN: MUL partial sum staging for hi nibble computation.

    Schoolbook: result = (a_hi*16+a_lo) * (b_hi*16+b_lo) mod 256
      result_lo = (a_lo * b_lo) % 16           [computed in L10]
      result_hi = (carry + a_lo*b_hi + a_hi*b_lo) % 16

    This layer computes: partial = (carry + a_lo*b_hi) % 16
      where carry = (a_lo * b_lo) // 16
    for all (a_lo, b_lo, b_hi) triples, stored in TEMP[0..15].

    L12 then combines: result_hi = (partial + a_hi*b_lo) % 16.

    Amplitude contract: hot units write TEMP[partial] ≈ 5.0 (silu(50) * 1
    * 10/S = 5.0). L12's 4-way AND threshold (b_up = -S*7.5) is
    calibrated for TEMP[partial] ≈ 5.0 so the L11→L12 chain crosses the
    gate. The earlier 2.0/S writeback produced TEMP[partial] ≈ 1.0,
    which silenced L12 and caused the expr_mul_div_* numeric diagnostic
    failures (e.g. 29*16/8 → 314 instead of 58) because wide-MUL
    products > 255 lost their high byte entirely.

    4096 units = 16^3 (fills L11 FFN exactly).
    """
    unit = 0

    for a_lo in range(16):
        for b_lo in range(16):
            carry = (a_lo * b_lo) // 16
            for b_hi in range(16):
                partial = (carry + a_lo * b_hi) % 16
                # 4-way AND: MARK_AX + ALU_LO[a_lo] + AX_CARRY_LO[b_lo] + AX_CARRY_HI[b_hi]
                ffn.W_up[unit, BD.MARK_AX] = S
                ffn.W_up[unit, BD.ALU_LO + a_lo] = S
                ffn.W_up[unit, BD.AX_CARRY_LO + b_lo] = S
                ffn.W_up[unit, BD.AX_CARRY_HI + b_hi] = S
                ffn.b_up[unit] = -S * 3.5
                ffn.W_gate[unit, BD.OP_MUL] = 1.0
                # 10.0/S so hot TEMP[partial] lands at ~5.0 (L12 threshold).
                ffn.W_down[BD.TEMP + partial, unit] = 10.0 / S
                unit += 1

    return unit



