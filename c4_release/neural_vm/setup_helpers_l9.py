"""Setup helpers extracted from ``setup_helpers.py`` (Phase 6 wave 1.5).

Layer 9 helpers: LEV addr / BP-to-PC relay attention heads.

Re-exported by ``setup_helpers`` for backward compatibility.
"""

import math

from .constants import PC_OFFSET


def _set_layer9_lev_addr_relay(attn, S, BD, HD):
    """L9 attention head 0: relay old BP value from prev step's BP byte 0 to SP marker.

    For LEV, L16 FFN computes SP = old_BP + 16. We need old_BP at the SP marker.

    The previous step's BP byte 0 position has CLEAN_EMBED = old_BP value.
    We relay this to ADDR_B0 at the current step's SP marker.

    Key insight: The previous step's BP marker has wrong ADDR_B0 (gets corrupted
    by L8 attention leakage). But the BP byte 0 position has correct CLEAN_EMBED.
    So we attend to BP byte 0 (L1H1[BP_I] AND BYTE_INDEX_0) instead of BP marker.

    Distance from current SP marker to prev BP byte 0 ≈ 29 tokens.

    BUG FIX 2026-04-15: Original approach attended to BP marker which had wrong
    ADDR_B0 due to L8 head 0 leakage. Now attend to BP byte 0 for correct value.
    """
    L = 50.0
    BP_I = 3  # BP marker index
    base = 0 * HD  # head 0

    # Q: fires ONLY at SP marker when OP_LEV active.
    # NOTE 2026-05-09: At L9 input OP_LEV is amplified to ~10 by L6 relays
    # (not ~5 as the original comment said). With threshold = -2*L:
    #   SP marker: Q[0] = L + (10 * L/5) - 2L = L (fires)
    #   any other position (MARK_SP=0): Q[0] = 0 + 2L - 2L = 0 (no fire)
    # The tight threshold prevents firing at PC/BP markers, where head 1 and
    # the L8 FFN do their own LEV writes (avoids cross-contamination).
    attn.W_q[base, BD.MARK_SP] = L
    attn.W_q[base, BD.OP_LEV] = L / 5  # at L9 input, OP_LEV ~= 10 (amplified by L6)
    attn.W_q[base, BD.CONST] = -2 * L  # tight threshold; prevents firing at non-SP markers

    # K: attend to BP byte 0 (L1H1[BP_I] AND BYTE_INDEX_0)
    # L1H1[BP_I] = 1 when within 2.5 tokens of BP marker
    # BYTE_INDEX_0 = 1 when it's byte 0 of a register section
    attn.W_k[base, BD.L1H1 + BP_I] = L
    attn.W_k[base, BD.BYTE_INDEX_0] = L

    # V: copy CLEAN_EMBED_LO/HI (the actual BP byte 0 value)
    # Scale up to dominate over existing values in residual add
    scale = 3.0
    for k in range(16):
        attn.W_v[base + 1 + k, BD.CLEAN_EMBED_LO + k] = scale
        attn.W_v[base + 17 + k, BD.CLEAN_EMBED_HI + k] = scale

    # O: write to ADDR_B0_LO/HI at SP marker
    for k in range(16):
        attn.W_o[BD.ADDR_B0_LO + k, base + 1 + k] = 1.0
        attn.W_o[BD.ADDR_B0_HI + k, base + 17 + k] = 1.0

    # FIX 2026-04-15: Anti-leakage gate to suppress attention at non-SP positions.
    # Without this, positions with K=0 get Q*K=0 scores, giving uniform softmax
    # weights that accumulate and pollute ADDR_B0 at BP marker.
    # At SP marker: Q[gate] = L - L/2 = +L/2, score += +L²/(2*8) ≈ +156
    # At non-SP markers: Q[gate] = -L/2, score += -L²/(2*8) ≈ -156
    GATE = 33
    attn.W_q[base + GATE, BD.MARK_SP] = L
    attn.W_q[base + GATE, BD.CONST] = -L / 2
    attn.W_k[base + GATE, BD.CONST] = L




def _set_layer9_lev_bp_to_pc_relay(attn, S, BD, HD):
    """L9 attention head 1: relay BP value from prev step's BP byte 0 to PC marker.

    For LEV return_addr lookup, L15 heads 8-11 need to know BP at PC marker.
    The previous step's BP byte 0 position has CLEAN_EMBED = old_BP value.
    We relay this to ADDR_B0 at the current step's PC marker.

    FIX 2026-04-16: Changed from attending to BP marker (which has OUTPUT=0)
    to attending to BP byte 0 (which has correct CLEAN_EMBED value).
    Same pattern as head 0 which works correctly at SP marker.
    """
    L = 50.0
    BP_I = 3  # BP marker index
    base = 1 * HD  # head 1

    # Q: fires ONLY at PC marker when OP_LEV active. Mirrors head 0 (see
    # _set_layer9_lev_addr_relay for the OP_LEV=10 / threshold=-2*L derivation).
    # FIX 2026-05-09: Restored -2*L from prior -1.5*L. The 2026-04-16 change
    # assumed OP_LEV ~ 5 at L9 input; in reality OP_LEV ~ 10 (amplified by L6),
    # so -1.5*L gave Q[0]=0.5*L = 25 at SP marker (spurious). That made head 1
    # also write to ADDR_B0 at SP marker, doubling head 0's contribution
    # (LO[8]=6.0 instead of 3.0 for BP=0xE8) and breaking the L9 +8-offset gate
    # and L16 BP+16 gate downstream.
    attn.W_q[base, BD.MARK_PC] = L
    attn.W_q[base, BD.OP_LEV] = L / 5  # at L9 input, OP_LEV ~= 10 (amplified by L6)
    attn.W_q[base, BD.CONST] = -2 * L  # tight threshold matches head 0; prevents SP-marker spurious firing

    # K: attend to BP byte 0 (L1H1[BP_I] AND BYTE_INDEX_0)
    # L1H1[BP_I] = 1 when within 2.5 tokens of BP marker
    # BYTE_INDEX_0 = 1 when it's byte 0 of a register section
    attn.W_k[base, BD.L1H1 + BP_I] = L
    attn.W_k[base, BD.BYTE_INDEX_0] = L

    # V: copy CLEAN_EMBED_LO/HI (the actual BP byte 0 value)
    # Scale up to dominate over existing values in residual add
    scale = 3.0
    for k in range(16):
        attn.W_v[base + 1 + k, BD.CLEAN_EMBED_LO + k] = scale
        attn.W_v[base + 17 + k, BD.CLEAN_EMBED_HI + k] = scale

    # O: write to ADDR_B0_LO/HI at PC marker
    for k in range(16):
        attn.W_o[BD.ADDR_B0_LO + k, base + 1 + k] = 1.0
        attn.W_o[BD.ADDR_B0_HI + k, base + 17 + k] = 1.0

    # Anti-leakage gate to suppress attention at non-PC positions
    # At PC marker: Q[gate] = L - L/2 = +L/2
    # At non-PC markers: Q[gate] = -L/2
    GATE = 33
    attn.W_q[base + GATE, BD.MARK_PC] = L
    attn.W_q[base + GATE, BD.CONST] = -L / 2
    attn.W_k[base + GATE, BD.CONST] = L


