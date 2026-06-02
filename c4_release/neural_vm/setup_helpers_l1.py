"""Setup helpers extracted from ``setup_helpers.py`` (Phase 6 wave 1.5).

Layer 1 FFN helpers.

Re-exported by ``setup_helpers`` for backward compatibility.
"""

import math

from .constants import PC_OFFSET


# DEPRECATED — parity oracle only, no production use (Phase 6 wave 6A).
def _set_layer1_ffn(ffn, S, BD):
    """Layer 1 FFN: STACK0_BYTE0 flag + BYTE_INDEX flags.

    STACK0 byte 0 is at d=6 from BP marker (nearest IS_MARK).
    Detected by: L1H4[BP] (d<=6.5) AND NOT H1[BP] (d>4.5) AND IS_BYTE.
    L1H4[BP] = dim BD.L1H4 + BP_I (BP index = 3 in MARKS array).
    H1[BP] = dim BD.H1 + BP_I.

    BYTE_INDEX_0-3: Marker-agnostic byte position within a register.
    Derived from threshold heads (summed across all marker types):
      BYTE_INDEX_0: IS_BYTE AND any(L1H1) AND NOT any(L1H0) → d∈(0.5,1.5]
      BYTE_INDEX_1: IS_BYTE AND any(L1H2) AND NOT any(L1H1) → d∈(1.5,2.5]
      BYTE_INDEX_2: IS_BYTE AND any(H0) AND NOT any(L1H2)  → d∈(2.5,3.5]
      BYTE_INDEX_3: IS_BYTE AND any(H1) AND NOT any(H0)    → d∈(3.5,4.5]
    Only one marker type is nearest at any position, so sum ≈ 1 when active.
    """
    BP_I = 3
    NM = BD.NUM_MARKERS  # 7 marker types
    unit = 0

    # STACK0_BYTE0: L1H4[BP] AND NOT H1[BP] AND IS_BYTE
    # silu(S*(L1H4_BP + IS_BYTE - 1.5)) * (1 - H1_BP)
    ffn.W_up[unit, BD.L1H4 + BP_I] = S
    ffn.W_up[unit, BD.IS_BYTE] = S
    ffn.b_up[unit] = -S * 1.5
    ffn.W_gate[unit, BD.H1 + BP_I] = -1.0
    ffn.b_gate[unit] = 1.0
    ffn.W_down[BD.STACK0_BYTE0, unit] = 2.0 / S
    unit += 1

    # BYTE_INDEX_0: IS_BYTE AND any(L1H1[i]) AND NOT any(L1H0[i])
    # up = S*(IS_BYTE + sum(L1H1[0..6])) - S*1.5
    # gate = 1 - sum(L1H0[0..6])
    ffn.W_up[unit, BD.IS_BYTE] = S
    for i in range(NM):
        ffn.W_up[unit, BD.L1H1 + i] = S
    ffn.b_up[unit] = -S * 1.5
    for i in range(NM):
        ffn.W_gate[unit, BD.L1H0 + i] = -1.0
    ffn.b_gate[unit] = 1.0
    ffn.W_down[BD.BYTE_INDEX_0, unit] = 2.0 / S
    unit += 1

    # BYTE_INDEX_1: IS_BYTE AND any(L1H2[i]) AND NOT any(L1H1[i])
    ffn.W_up[unit, BD.IS_BYTE] = S
    for i in range(NM):
        ffn.W_up[unit, BD.L1H2 + i] = S
    ffn.b_up[unit] = -S * 1.5
    for i in range(NM):
        ffn.W_gate[unit, BD.L1H1 + i] = -1.0
    ffn.b_gate[unit] = 1.0
    ffn.W_down[BD.BYTE_INDEX_1, unit] = 2.0 / S
    unit += 1

    # BYTE_INDEX_2: IS_BYTE AND any(H0[i]) AND NOT any(L1H2[i])
    ffn.W_up[unit, BD.IS_BYTE] = S
    for i in range(NM):
        ffn.W_up[unit, BD.H0 + i] = S
    ffn.b_up[unit] = -S * 1.5
    for i in range(NM):
        ffn.W_gate[unit, BD.L1H2 + i] = -1.0
    ffn.b_gate[unit] = 1.0
    ffn.W_down[BD.BYTE_INDEX_2, unit] = 2.0 / S
    unit += 1

    # BYTE_INDEX_3: IS_BYTE AND any(H1[i]) AND NOT any(H0[i])
    ffn.W_up[unit, BD.IS_BYTE] = S
    for i in range(NM):
        ffn.W_up[unit, BD.H1 + i] = S
    ffn.b_up[unit] = -S * 1.5
    for i in range(NM):
        ffn.W_gate[unit, BD.H0 + i] = -1.0
    ffn.b_gate[unit] = 1.0
    ffn.W_down[BD.BYTE_INDEX_3, unit] = 2.0 / S
    unit += 1


