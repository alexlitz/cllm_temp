"""Setup helpers extracted from ``setup_helpers.py`` (Phase 6 wave 1.5).

Layer 2 FFN helpers (MEM byte flags + extended BYTE_INDEX).

Re-exported by ``setup_helpers`` for backward compatibility.
"""

import math

from .constants import PC_OFFSET


def _set_layer2_mem_byte_flags(ffn, S, BD):
    """Layer 2 FFN: MEM val byte position flags + extended BYTE_INDEX for STACK0.

    MEM val byte flags (4 units): Identify positions d=4..7 from MEM marker.
    These are the QUERY positions that predict MEM val bytes 0-3 (autoregressive shift).
    FIX 2026-04-16: Shifted from d=5..8 to d=4..7 for correct autoregressive prediction.

    Extended BYTE_INDEX for STACK0 bytes 1-3 (3 units): At positions d=7..9
    from BP (where STACK0 bytes live), produce BYTE_INDEX_1/2/3 flags.
    These accumulate with existing BYTE_INDEX (which is 0 at those positions).
    """
    MEM_I = 4  # MEM marker index in MARKS
    BP_I = 3
    NM = BD.NUM_MARKERS
    unit = 0

    # MEM_VAL_B0: d=4 from MEM (addr byte 3 position, predicts val byte 0)
    # H1[MEM]=1 (d≤4.5), H0[MEM]=0 (d>3.5)
    # silu(S*(H1_MEM + IS_BYTE) - S*1.5) × (1 - H0_MEM)
    ffn.W_up[unit, BD.H1 + MEM_I] = S
    ffn.W_up[unit, BD.IS_BYTE] = S
    ffn.b_up[unit] = -S * 1.5
    ffn.W_gate[unit, BD.H0 + MEM_I] = -1.0
    ffn.b_gate[unit] = 1.0
    ffn.W_down[BD.MEM_VAL_B0, unit] = 2.0 / S
    unit += 1

    # MEM_VAL_B1: d=5 from MEM (val byte 0 position, predicts val byte 1)
    # L2H0[MEM]=1 (d≤5.5), H1[MEM]=0 (d>4.5)
    ffn.W_up[unit, BD.L2H0 + MEM_I] = S
    ffn.W_up[unit, BD.IS_BYTE] = S
    ffn.b_up[unit] = -S * 1.5
    ffn.W_gate[unit, BD.H1 + MEM_I] = -1.0
    ffn.b_gate[unit] = 1.0
    ffn.W_down[BD.MEM_VAL_B1, unit] = 2.0 / S
    unit += 1

    # MEM_VAL_B2: d=6 from MEM (val byte 1 position, predicts val byte 2)
    # L1H4[MEM]=1 (d≤6.5), L2H0[MEM]=0 (d>5.5)
    ffn.W_up[unit, BD.L1H4 + MEM_I] = S
    ffn.W_up[unit, BD.IS_BYTE] = S
    ffn.b_up[unit] = -S * 1.5
    ffn.W_gate[unit, BD.L2H0 + MEM_I] = -1.0
    ffn.b_gate[unit] = 1.0
    ffn.W_down[BD.MEM_VAL_B2, unit] = 2.0 / S
    unit += 1

    # MEM_VAL_B3: d=7 from MEM (val byte 2 position, predicts val byte 3)
    # H2[MEM]=1 (d≤7.5), L1H4[MEM]=0 (d>6.5)
    ffn.W_up[unit, BD.H2 + MEM_I] = S
    ffn.W_up[unit, BD.IS_BYTE] = S
    ffn.b_up[unit] = -S * 1.5
    ffn.W_gate[unit, BD.L1H4 + MEM_I] = -1.0
    ffn.b_gate[unit] = 1.0
    ffn.W_down[BD.MEM_VAL_B3, unit] = 2.0 / S
    unit += 1

    # Extended BYTE_INDEX for STACK0 byte 0-3 (at d=6,7,8,9 from BP)
    # BYTE_INDEX_0 at STACK0: d=6 from BP → L1H4[BP]=1 (d≤6.5), H1[BP]=0 (d>4.5)
    ffn.W_up[unit, BD.L1H4 + BP_I] = S
    ffn.W_up[unit, BD.IS_BYTE] = S
    ffn.b_up[unit] = -S * 1.5
    ffn.W_gate[unit, BD.H1 + BP_I] = -1.0
    ffn.b_gate[unit] = 1.0
    ffn.W_down[BD.BYTE_INDEX_0, unit] = 2.0 / S
    unit += 1

    # BYTE_INDEX_1 at STACK0: d=7 from BP → H2[BP]=1 (d≤7.5), L1H4[BP]=0 (d>6.5)
    ffn.W_up[unit, BD.H2 + BP_I] = S
    ffn.W_up[unit, BD.IS_BYTE] = S
    ffn.b_up[unit] = -S * 1.5
    ffn.W_gate[unit, BD.L1H4 + BP_I] = -1.0
    ffn.b_gate[unit] = 1.0
    ffn.W_down[BD.BYTE_INDEX_1, unit] = 2.0 / S
    ffn.W_down[BD.STACK0_BYTE1, unit] = 2.0 / S
    unit += 1

    # BYTE_INDEX_2 at STACK0: d=8 from BP → H3[BP]=1 (d≤8.5), H2[BP]=0 (d>7.5)
    ffn.W_up[unit, BD.H3 + BP_I] = S
    ffn.W_up[unit, BD.IS_BYTE] = S
    ffn.b_up[unit] = -S * 1.5
    ffn.W_gate[unit, BD.H2 + BP_I] = -1.0
    ffn.b_gate[unit] = 1.0
    ffn.W_down[BD.BYTE_INDEX_2, unit] = 2.0 / S
    ffn.W_down[BD.STACK0_BYTE2, unit] = 2.0 / S
    unit += 1

    # BYTE_INDEX_3 at STACK0: d=9 from BP → H4[BP]=1 (d≤9.5), H3[BP]=0 (d>8.5)
    ffn.W_up[unit, BD.H4 + BP_I] = S
    ffn.W_up[unit, BD.IS_BYTE] = S
    ffn.b_up[unit] = -S * 1.5
    ffn.W_gate[unit, BD.H3 + BP_I] = -1.0
    ffn.b_gate[unit] = 1.0
    ffn.W_down[BD.BYTE_INDEX_3, unit] = 2.0 / S
    ffn.W_down[BD.STACK0_BYTE3, unit] = 2.0 / S
    unit += 1


