"""Setup helpers extracted from ``setup_helpers.py`` (Phase 6 wave 1.5).

Layer 10 helpers: byte/SP/BP/PSH passthroughs + null-terminator detection.

Re-exported by ``setup_helpers`` for backward compatibility.
"""

import math

from .constants import PC_OFFSET


# DEPRECATED — parity oracle only, no production use (Phase 6 wave 6A).
def _set_layer10_carry_relay(attn, S, BD, HD):
    """L10 attention head 0: relay CARRY[1/2] from AX marker to AX byte positions.

    At AX byte positions (IS_BYTE=1, H1[AX]=1), attends strongly to the
    nearest AX marker and copies CARRY[1] (ADD byte carry) and CARRY[2]
    (SUB byte borrow). Anti-leakage gate via H1[AX_IDX] ensures non-AX
    bytes get negligible attention to the AX marker.
    """
    L = S  # attention scale
    AX_IDX = 1  # AX is register index 1 (PC=0, AX=1, SP=2, ...)
    base = 0  # head 0

    # Q: fires at byte positions, suppressed at markers
    attn.W_q[base + 0, BD.IS_BYTE] = L
    attn.W_q[base + 0, BD.CONST] = -L / 2  # Q[0] = L*(IS_BYTE - 0.5)

    # K: fires at AX marker positions
    attn.W_k[base + 0, BD.MARK_AX] = L

    # Anti-leakage: restrict to AX byte positions (H1[AX_IDX] ≈ 1)
    attn.W_q[base + 33, BD.H1 + AX_IDX] = L
    attn.W_q[base + 33, BD.CONST] = -L / 2
    attn.W_k[base + 33, BD.CONST] = L

    # V: copy CARRY[1] and CARRY[2] from marker
    attn.W_v[base + 1, BD.CARRY + 1] = 1.0
    attn.W_v[base + 2, BD.CARRY + 2] = 1.0

    # O: write to CARRY[1] and CARRY[2] at byte position
    attn.W_o[BD.CARRY + 1, base + 1] = 1.0
    attn.W_o[BD.CARRY + 2, base + 2] = 1.0




# DEPRECATED — parity oracle only, no production use (Phase 6 wave 6A).
def _set_layer10_byte_passthrough(attn, S, BD, HD):
    """L10 attention head 1: AX byte 0-2 passthrough across steps.

    Copies CLEAN_EMBED from previous step's AX bytes 1-3 → OUTPUT at current
    step's AX byte 0-2 positions. Uses **shifted** byte matching because of the
    autoregressive offset: logits at byte K position predict byte K+1's token.

    Mapping (Q byte K attends to K byte K+1 of prev step):
      byte 0 pos → prev byte 1 (predicts byte 1 token)
      byte 1 pos → prev byte 2 (predicts byte 2 token)
      byte 2 pos → prev byte 3 (predicts byte 3 token)
      byte 3 pos → suppressed (predicts SP marker, not a byte)

    Byte 0's token (AX byte 0 value) is handled by the marker-level AX passthrough
    FFN, which reads AX_CARRY_LO/HI at the AX marker position.

    Q[0] gate (default ``byte_passthrough_chain`` coefficients):
        IS_BYTE*3L + HAS_SE*L + suppressed-op terms + CONST*-3.5L
      - AX marker: 0 + L - 3.5L = -2.5L < 0 (blocked)
      - AX byte non-IMM non-bitwise: 3L + L - 3.5L = 0.5L > 0 (fires)
      - First step (HAS_SE=0): 3L + 0 - 3.5L = -0.5L < 0 (blocked)
      - IMM/LI/LC relay/bitwise/binary-pop byte: 3L + L - 3L - 3.5L
        = -2.5L < 0 (blocked, preserving multi-byte OUTPUT and dedicated
        post-op handling)

    Q[33] AND gate: H1[AX] AND HAS_SE (kills leakage at non-AX / first step).

    Strength: W_o=2.0, weaker than carry override (≈5.0) so carry wins when active.
    ALiBi slope=1.0: prefers nearest step (d=35 beats d=70).

    2026-05-10: Refactored to call ``Primitives.byte_passthrough_chain``
    (set 1) for byte-identical weights with the AX-default coefficients.
    """
    from .unified_compiler.primitives import Primitives as _P

    AX_IDX = 1
    _P.byte_passthrough_chain(
        attn,
        head_idx=1,  # head 1 (base = HD)
        source_marker_dim=BD.H1 + AX_IDX,
        target_marker_dim=BD.H1 + AX_IDX,
        value_lo_dim=BD.CLEAN_EMBED_LO,
        value_hi_dim=BD.CLEAN_EMBED_HI,
        # Suppress during multi-byte/relay ops, binary-pop, and bitwise ops
        # (TEMP+3 = BITWISE_OP).
        suppress_op_dims=[
            BD.OP_IMM,
            BD.OP_LI_RELAY,
            BD.OP_LC_RELAY,
            BD.TEMP + 3,
            BD.CMP + 3,
        ],
        S=S,
        HD=HD,
        alibi_slope=1.0,
        # Defaults match AX coefficients (is_byte_strength=3, has_se_strength=1,
        # suppress_strength=3, q0_threshold=3.5, gate_const=-20000, gate_*=10000).
    )




# DEPRECATED — parity oracle only, no production use (Phase 6 wave 6A).
def _set_layer10_sp_byte_passthrough(attn, S, BD, HD):
    """L10 attention head 2: SP byte 0-2 passthrough across steps (when NOT PSH).

    Similar to AX byte passthrough but for SP. Only fires when PSH_AT_SP = 0
    (i.e., when SP doesn't change). When PSH is active, L6/L15 handle SP values.
    During binary POP (CMP[3]), this head still carries old upper SP bytes;
    the declarative pop-carry tail then increments the carried byte values.

    Copies CLEAN_EMBED from previous step's SP bytes 1-3 → OUTPUT at current
    step's SP byte 0-2 positions. Uses shifted byte matching.

    Mapping (Q byte K attends to K byte K+1 of prev step):
      byte 0 pos → prev byte 1 (predicts byte 1 token)
      byte 1 pos → prev byte 2 (predicts byte 2 token)
      byte 2 pos → prev byte 3 (predicts byte 3 token)
      byte 3 pos → suppressed (predicts BP marker, not a byte)

    Q[0] gate (SP-specific coefficients):
        IS_BYTE*L + HAS_SE*2L + (PSH_AT_SP @ -2L) + (MARK_SP @ -2L)
        + CONST*-1.5L
      - SP byte non-PSH non-first: L + 2L - 1.5L = 1.5L > 0 (fires)
      - First step (HAS_SE=0): L + 0 - 1.5L = -0.5L < 0 (blocked)
      - PSH step at SP byte: L + 2L - 2L - 1.5L = -0.5L < 0 (blocked)
      - SP marker row: 0 + 2L - 2L - 1.5L = -1.5L < 0 (blocked;
        marker carry-forward is handled by the explicit Q[34]/Q[35] route)

    Q[33] AND gate: IS_BYTE AND H1[SP] AND HAS_SE AND NOT PSH
    (gate_const=-30000 + 3*10000 base; PSH_AT_SP contributes -10000
    via gate_extras to suppress during PSH).

    2026-05-10: Refactored to call ``Primitives.byte_passthrough_chain``
    (set 1) using SP-specific overrides for the Q[0] / Q[33] coefficients.
    """
    from .unified_compiler.primitives import Primitives as _P

    SP_IDX = 2
    _P.byte_passthrough_chain(
        attn,
        head_idx=2,  # head 2 (base = 2*HD)
        source_marker_dim=BD.H1 + SP_IDX,
        target_marker_dim=BD.H1 + SP_IDX,
        value_lo_dim=BD.CLEAN_EMBED_LO,
        value_hi_dim=BD.CLEAN_EMBED_HI,
        # Q[0] suppression for PSH (SP -= 8 handled elsewhere) and SP
        # marker rows, which are owned by the explicit marker carry route.
        suppress_op_dims=[BD.PSH_AT_SP, BD.MARK_SP],
        S=S,
        HD=HD,
        alibi_slope=1.0,
        # SP Q[0] coefficient overrides (not the AX defaults).
        is_byte_strength=1.0,
        has_se_strength=2.0,
        suppress_strength=2.0,
        q0_threshold=1.5,
        # SP Q[33] gate uses larger negative bias and three extra AND terms.
        gate_const=-30000.0,
        gate_extras=[
            (BD.IS_BYTE, 10000.0),         # require IS_BYTE
            (BD.MARK_SP, 10000.0),         # allow SP marker carry-forward
            (BD.PSH_AT_SP, -10000.0),      # suppress during PSH (SP -= 8)
        ],
    )

    marker_s = 300.0
    base = 2 * HD
    attn.W_q[base + 34, BD.MARK_SP] = marker_s
    attn.W_q[base + 34, BD.HAS_SE] = marker_s
    attn.W_q[base + 34, BD.PSH_AT_SP] = -2.0 * marker_s
    attn.W_q[base + 34, BD.CMP + 3] = -2.0 * marker_s
    attn.W_q[base + 34, BD.CONST] = -marker_s
    attn.W_q[base + 35, BD.MARK_SP] = marker_s
    attn.W_q[base + 35, BD.HAS_SE] = marker_s
    attn.W_q[base + 35, BD.PSH_AT_SP] = -2.0 * marker_s
    attn.W_q[base + 35, BD.CMP + 3] = -2.0 * marker_s
    attn.W_q[base + 35, BD.CONST] = -marker_s
    attn.W_k[base + 34, BD.H1 + SP_IDX] = marker_s
    attn.W_k[base + 35, BD.BYTE_INDEX_0] = marker_s




# DEPRECATED — parity oracle only, no production use (Phase 6 wave 6A).
def _set_layer10_bp_byte_passthrough(attn, S, BD, HD):
    """L10 attention head 7: BP byte 0-2 passthrough across ordinary steps."""
    from .unified_compiler.primitives import Primitives as _P

    BP_IDX = 3
    _P.byte_passthrough_chain(
        attn,
        head_idx=7,
        source_marker_dim=BD.H1 + BP_IDX,
        target_marker_dim=BD.H1 + BP_IDX,
        value_lo_dim=BD.CLEAN_EMBED_LO,
        value_hi_dim=BD.CLEAN_EMBED_HI,
        suppress_op_dims=[BD.OP_ENT, BD.OP_LEV],
        S=S,
        HD=HD,
        alibi_slope=1.0,
        is_byte_strength=1.0,
        has_se_strength=2.0,
        suppress_strength=2.0,
        q0_threshold=1.5,
        gate_const=-30000.0,
        gate_extras=[
            (BD.IS_BYTE, 10000.0),
            (BD.OP_ENT, -10000.0),
            (BD.OP_LEV, -10000.0),
        ],
    )




# DEPRECATED — parity oracle only, no production use (Phase 6 wave 6A).
def _set_layer10_psh_stack0_passthrough(attn, S, BD, HD):
    """L10 attention head 3: PSH STACK0 bytes 1-3 passthrough from AX.

    During PSH, STACK0 = AX. The L6 FFN handles byte 0 at the STACK0 marker.
    This head handles bytes 1-3 by copying AX bytes 1-3 to OUTPUT at STACK0
    byte positions 0-2 (shifted matching for autoregressive generation).

    Mapping (Q at STACK0 byte K attends to K at AX byte K+1 in SAME step):
      STACK0 byte 0 pos → AX byte 1 (predicts STACK0 byte 1 token)
      STACK0 byte 1 pos → AX byte 2 (predicts STACK0 byte 2 token)
      STACK0 byte 2 pos → AX byte 3 (predicts STACK0 byte 3 token)
      STACK0 byte 3 pos → suppressed (predicts MEM marker, not a byte)

    Distances (within same step):
      STACK0 byte 0 (pos 21) → AX byte 1 (pos 7): d = 14
      STACK0 byte 1 (pos 22) → AX byte 2 (pos 8): d = 14
      STACK0 byte 2 (pos 23) → AX byte 3 (pos 9): d = 14

    Only active when PSH_AT_SP = 1 (PSH is executing).
    """
    L = S  # attention scale
    AX_IDX = 1
    BP_IDX = 3
    base = 3 * HD  # head 3 starts at dim 192

    # Q dim 0: IS_BYTE (at STACK0 byte positions)
    attn.W_q[base + 0, BD.IS_BYTE] = L

    # Q dim 1: H4[BP] AND NOT H1[BP] → STACK0 area (d=6-9 from BP)
    # L1H4[BP] fires at d <= 6.5, but we want d=6-9, so use H4[BP] (d <= 9.5)
    attn.W_q[base + 1, BD.H4 + BP_IDX] = L
    attn.W_q[base + 1, BD.H1 + BP_IDX] = -L  # Exclude BP bytes (d <= 4.5)
    attn.W_q[base + 1, BD.CONST] = -L / 2

    # Q dim 2: suppress byte 3 (predicts MEM marker, not a byte)
    attn.W_q[base + 2, BD.BYTE_INDEX_3] = -L
    attn.W_q[base + 2, BD.CONST] = L / 2

    # Q dim 3: PSH_AT_SP (only fire during PSH)
    attn.W_q[base + 3, BD.PSH_AT_SP] = L
    attn.W_q[base + 3, BD.CONST] = -L / 2

    # K dim 0: IS_BYTE (AX byte positions)
    attn.W_k[base + 0, BD.IS_BYTE] = L

    # K dim 1: H1[AX] → AX area (d <= 4.5 from AX marker)
    attn.W_k[base + 1, BD.H1 + AX_IDX] = L

    # K dim 2: suppress byte 0 in K (not a valid target for shifted matching)
    attn.W_k[base + 2, BD.BYTE_INDEX_0] = -L
    attn.W_k[base + 2, BD.CONST] = L / 2

    # Shifted byte matching: Q at STACK0 byte K → K at AX byte K+1
    # STACK0 byte 0 (BYTE_INDEX_0) → AX byte 1 (BYTE_INDEX_1)
    attn.W_q[base + 4, BD.BYTE_INDEX_0] = L
    attn.W_k[base + 4, BD.BYTE_INDEX_1] = L
    # STACK0 byte 1 (BYTE_INDEX_1) → AX byte 2 (BYTE_INDEX_2)
    attn.W_q[base + 5, BD.BYTE_INDEX_1] = L
    attn.W_k[base + 5, BD.BYTE_INDEX_2] = L
    # STACK0 byte 2 (BYTE_INDEX_2) → AX byte 3 (BYTE_INDEX_3)
    attn.W_q[base + 6, BD.BYTE_INDEX_2] = L
    attn.W_k[base + 6, BD.BYTE_INDEX_3] = L

    # Gate dim 33: Enforce 5-way AND (IS_BYTE + H4[BP] - H1[BP] + PSH_AT_SP - MARK_STACK0)
    # At STACK0 byte positions during PSH: all conditions met → Q[33] near 0 → passes
    # At STACK0 marker: MARK_STACK0=1 → Q[33] = -10000 → suppressed (prevents leakage)
    # At other positions: Q[33] large negative → suppressed
    # BUG FIX 2026-04-29: Added MARK_STACK0 suppression. PSH_AT_SP doubles from 1.0 to 2.0
    # between L6 and L7, which neutralized the old 4-way gate at the STACK0 marker position
    # (-30000 + 0 + 10000 - 0 + 20000 = 0). The head was copying CLEAN_EMBED from AX bytes
    # to OUTPUT at the marker, corrupting the correct value from L6 FFN.
    attn.W_q[base + 33, BD.CONST] = -30000.0
    attn.W_q[base + 33, BD.IS_BYTE] = 10000.0
    attn.W_q[base + 33, BD.H4 + BP_IDX] = 10000.0
    attn.W_q[base + 33, BD.H1 + BP_IDX] = -10000.0  # Exclude BP area
    attn.W_q[base + 33, BD.PSH_AT_SP] = 10000.0
    attn.W_q[base + 33, BD.MARK_STACK0] = -10000.0  # Exclude STACK0 marker
    attn.W_k[base + 33, BD.CONST] = 5.0

    # V: copy CLEAN_EMBED nibbles (16 lo + 16 hi = 32 V dims)
    for k in range(16):
        attn.W_v[base + k, BD.CLEAN_EMBED_LO + k] = 1.0
        attn.W_v[base + 16 + k, BD.CLEAN_EMBED_HI + k] = 1.0

    # O: write to OUTPUT_LO/HI (strength 3.0, stronger than default 0)
    for k in range(16):
        attn.W_o[BD.OUTPUT_LO + k, base + k] = 3.0
        attn.W_o[BD.OUTPUT_HI + k, base + 16 + k] = 3.0




# DEPRECATED — parity oracle only, no production use (Phase 6 wave 6A).
def _set_layer10_stack0_byte_relay(attn, S, BD, HD):
    """L10 attention heads 4-6: STACK0 byte relays.

    Heads 4-5 copy stack-memory bytes to ALU at AX bytes. Head 6 carries
    STACK0 bytes 1-3 across non-stack-mutating steps; L3 already carries
    byte 0 at the STACK0 marker.
    """
    AX_IDX = 1
    base = 4 * HD

    attn.W_q[base + 0, BD.CONST] = -3000.0
    attn.W_q[base + 0, BD.IS_BYTE] = 1000.0
    attn.W_q[base + 0, BD.H1 + AX_IDX] = 1000.0
    attn.W_q[base + 0, BD.TEMP + 3] = 1000.0
    attn.W_q[base + 0, BD.BYTE_INDEX_3] = -3000.0
    attn.W_k[base + 0, BD.CONST] = 10.0

    attn.W_q[base + 1, BD.TEMP + 3] = 50.0
    attn.W_k[base + 1, BD.MEM_STORE] = 100.0
    attn.W_k[base + 1, BD.MARK_MEM] = -200.0
    attn.W_k[base + 1, BD.CONST] = -50.0

    for slot, byte_idx_dim, mem_val_dim in (
        (31, BD.BYTE_INDEX_0, BD.MEM_VAL_B2),
        (32, BD.BYTE_INDEX_1, BD.MEM_VAL_B3),
    ):
        attn.W_q[base + slot, byte_idx_dim] = 60.0
        attn.W_k[base + slot, mem_val_dim] = 60.0
    attn.W_k[base + 31, BD.STACK0_BYTE1] = 60.0
    attn.W_k[base + 32, BD.STACK0_BYTE2] = 60.0
    attn.W_q[base + 34, BD.BYTE_INDEX_2] = 60.0
    attn.W_k[base + 34, BD.H3 + 4] = 60.0
    attn.W_k[base + 34, BD.H2 + 4] = -60.0
    attn.W_k[base + 34, BD.STACK0_BYTE3] = 60.0

    attn.W_q[base + 33, BD.CONST] = -30000.0
    attn.W_q[base + 33, BD.IS_BYTE] = 10000.0
    attn.W_q[base + 33, BD.H1 + AX_IDX] = 10000.0
    attn.W_q[base + 33, BD.TEMP + 3] = 10000.0
    attn.W_q[base + 33, BD.BYTE_INDEX_3] = -10000.0
    attn.W_k[base + 33, BD.CONST] = 5.0

    attn.W_v[base + 0, BD.CONST] = 1.0
    for k in range(16):
        attn.W_v[base + 1 + k, BD.CLEAN_EMBED_LO + k] = 1.0
        attn.W_v[base + 17 + k, BD.CLEAN_EMBED_HI + k] = 1.0

    for k in range(16):
        attn.W_o[BD.ALU_LO + k, base + 0] = -8.0
        attn.W_o[BD.ALU_HI + k, base + 0] = -8.0
        attn.W_o[BD.ALU_LO + k, base + 1 + k] = 11.0
        attn.W_o[BD.ALU_HI + k, base + 17 + k] = 11.0

    # Head 5: non-bitwise STACK0 byte relay. This leaves the bitwise-sensitive
    # head 4 scoring untouched while letting ADD/SUB byte post-ops recover
    # higher STACK0 bytes from the stored MEM row. TEMP[3] is the bitwise relay
    # and suppresses this head.
    base = 5 * HD
    attn.W_q[base + 0, BD.CONST] = -3000.0
    attn.W_q[base + 0, BD.IS_BYTE] = 1000.0
    attn.W_q[base + 0, BD.H1 + AX_IDX] = 1000.0
    attn.W_q[base + 0, BD.CMP + 3] = 150000.0
    attn.W_q[base + 0, BD.TEMP + 3] = -500.0
    attn.W_q[base + 0, BD.BYTE_INDEX_3] = -3000.0
    attn.W_k[base + 0, BD.CONST] = 10.0

    attn.W_q[base + 1, BD.CMP + 3] = 1000.0
    attn.W_k[base + 1, BD.MEM_STORE] = 100.0
    attn.W_k[base + 1, BD.MARK_MEM] = -200.0
    attn.W_k[base + 1, BD.CONST] = -50.0

    for slot, byte_idx_dim, mem_val_dim in (
        (31, BD.BYTE_INDEX_0, BD.MEM_VAL_B2),
        (32, BD.BYTE_INDEX_1, BD.MEM_VAL_B3),
    ):
        attn.W_q[base + slot, byte_idx_dim] = 60.0
        attn.W_k[base + slot, mem_val_dim] = 60.0
    attn.W_k[base + 31, BD.STACK0_BYTE1] = 60.0
    attn.W_k[base + 32, BD.STACK0_BYTE2] = 60.0
    attn.W_q[base + 34, BD.BYTE_INDEX_2] = 60.0
    attn.W_k[base + 34, BD.H3 + 4] = 60.0
    attn.W_k[base + 34, BD.H2 + 4] = -60.0
    attn.W_k[base + 34, BD.STACK0_BYTE3] = 60.0

    attn.W_q[base + 33, BD.CONST] = -30000.0
    attn.W_q[base + 33, BD.IS_BYTE] = 10000.0
    attn.W_q[base + 33, BD.H1 + AX_IDX] = 10000.0
    attn.W_q[base + 33, BD.CMP + 3] = 1500000.0
    attn.W_q[base + 33, BD.TEMP + 3] = -5000.0
    attn.W_q[base + 33, BD.BYTE_INDEX_3] = -10000.0
    attn.W_k[base + 33, BD.CONST] = 5.0

    for k in range(16):
        attn.W_v[base + 1 + k, BD.CLEAN_EMBED_LO + k] = 1.0
        attn.W_v[base + 17 + k, BD.CLEAN_EMBED_HI + k] = 1.0
        attn.W_o[BD.ALU_LO + k, base + 1 + k] = 1.0
        attn.W_o[BD.ALU_HI + k, base + 17 + k] = 1.0

    # Head 6: STACK0 carry/update. Query at current STACK0 byte K and read the
    # latest previous STACK0 byte K+1 for non-mutating steps. On STORE rows the
    # popped SP points at the just-written local, so the extra source-match
    # slots read the current AX bytes into STACK0 and dominate persistence.
    base = 6 * HD
    attn.W_q[base + 0, BD.PSH_AT_SP] = -300.0
    attn.W_q[base + 0, BD.OP_PSH] = -300.0
    attn.W_q[base + 0, BD.CMP + 0] = -300.0
    attn.W_q[base + 0, BD.CMP + 1] = -300.0
    attn.W_q[base + 0, BD.CMP + 2] = -300.0
    attn.W_q[base + 0, BD.CMP + 4] = -300.0
    attn.W_q[base + 0, BD.OP_LEV] = -300.0

    # Match the compiler-owned declarative spec: direct STACK0-byte
    # persistence must dominate inactive store-route bias slots.
    byte_match = 50.0 * S
    store_target = 50.0 * S
    store_gate = 50.0 * S
    store_cmp = 5.0 * S
    store_has_se = 5.0 * S
    store_bias = -85.0 * S
    attn.W_q[base + 4, BD.STACK0_BYTE0] = byte_match
    attn.W_k[base + 4, BD.STACK0_BYTE1] = byte_match
    attn.W_q[base + 5, BD.STACK0_BYTE1] = byte_match
    attn.W_k[base + 5, BD.STACK0_BYTE2] = byte_match
    attn.W_q[base + 6, BD.STACK0_BYTE2] = byte_match
    attn.W_k[base + 6, BD.STACK0_BYTE3] = byte_match
    for slot in (7, 8, 9, 10, 11):
        attn.W_q[base + slot, BD.CONST] = store_bias
        attn.W_q[base + slot, BD.MEM_STORE] = store_gate
        attn.W_q[base + slot, BD.MEM_ADDR_SRC] = -store_gate
        attn.W_q[base + slot, BD.CMP + 3] = store_cmp
        attn.W_q[base + slot, BD.HAS_SE] = store_has_se
    attn.W_q[base + 7, BD.MARK_STACK0] = store_target
    attn.W_k[base + 7, BD.BYTE_INDEX_0] = byte_match
    attn.W_q[base + 8, BD.STACK0_BYTE0] = store_target
    attn.W_k[base + 8, BD.BYTE_INDEX_1] = byte_match
    attn.W_q[base + 9, BD.STACK0_BYTE1] = store_target
    attn.W_k[base + 9, BD.BYTE_INDEX_2] = byte_match
    attn.W_q[base + 10, BD.STACK0_BYTE2] = store_target
    attn.W_k[base + 10, BD.BYTE_INDEX_3] = byte_match

    for dim in (
        BD.MARK_STACK0,
        BD.STACK0_BYTE0,
        BD.STACK0_BYTE1,
        BD.STACK0_BYTE2,
    ):
        attn.W_q[base + 11, dim] = store_target
    attn.W_k[base + 11, BD.H1 + AX_IDX] = byte_match

    attn.W_q[base + 33, BD.CONST] = -15000.0
    attn.W_q[base + 33, BD.HAS_SE] = 10000.0
    attn.W_q[base + 33, BD.PSH_AT_SP] = -30000.0
    attn.W_q[base + 33, BD.OP_PSH] = -30000.0
    attn.W_q[base + 33, BD.CMP + 0] = -30000.0
    attn.W_q[base + 33, BD.CMP + 1] = -30000.0
    attn.W_q[base + 33, BD.CMP + 2] = -30000.0
    attn.W_q[base + 33, BD.CMP + 4] = -30000.0
    attn.W_q[base + 33, BD.OP_LEV] = -30000.0
    attn.W_q[base + 33, BD.STACK0_BYTE0] = 10000.0
    attn.W_q[base + 33, BD.STACK0_BYTE1] = 10000.0
    attn.W_q[base + 33, BD.STACK0_BYTE2] = 10000.0
    attn.W_q[base + 33, BD.STACK0_BYTE3] = -30000.0
    attn.W_k[base + 33, BD.CONST] = 100.0

    for k in range(16):
        attn.W_v[base + k, BD.CLEAN_EMBED_LO + k] = 1.0
        attn.W_v[base + 16 + k, BD.CLEAN_EMBED_HI + k] = 1.0
        attn.W_o[BD.OUTPUT_LO + k, base + k] = 3.0
        attn.W_o[BD.OUTPUT_HI + k, base + 16 + k] = 3.0

    if getattr(attn, "alibi_slopes", None) is not None:
        attn.alibi_slopes.data[6] = 1.0




def _set_null_terminator_detection(ffn, S, BD):
    """L10 FFN addition: Detect null terminator (byte = 0) in output.

    When IO_IN_OUTPUT_MODE AND OUTPUT_BYTE == 0 (all nibbles zero):
    - Set IO_OUTPUT_COMPLETE = 1 (format string done)
    - Clear IO_IN_OUTPUT_MODE (exit output mode)
    - Set NEXT_THINKING_START (emit THINKING_START next)

    This detects the end of the format string and prepares to resume normal execution.

    Starts at unit 1864 to avoid conflicts with existing L10 FFN logic.
    _set_layer10_alu uses ~1854 units (comparison, bitwise, MUL, SHL/SHR, passthrough).
    """
    unit = 1864

    # Detect null byte: OUTPUT_BYTE_LO[0] AND OUTPUT_BYTE_HI[0] (both nibbles = 0)
    # AND IO_IN_OUTPUT_MODE (currently in output mode)
    # CRITICAL: Gate on IO_IN_OUTPUT_MODE to prevent firing due to TEMP overlap!
    ffn.W_up[unit, BD.OUTPUT_BYTE_LO] = S  # lo nibble [0] = 1
    ffn.W_up[unit, BD.OUTPUT_BYTE_HI] = S  # hi nibble [0] = 1
    ffn.W_up[unit, BD.IO_IN_OUTPUT_MODE] = S
    ffn.b_up[unit] = -S * 2.5  # need all three active
    # Gate: only fire if IO_IN_OUTPUT_MODE > 5.0 (strongly active)
    # This prevents spurious firing due to OUTPUT_BYTE/TEMP overlap
    ffn.W_gate[unit, BD.IO_IN_OUTPUT_MODE] = 1.0
    ffn.b_gate[unit] = -5.0
    ffn.W_down[BD.IO_OUTPUT_COMPLETE, unit] = 2.0 / S  # set complete flag
    ffn.W_down[BD.IO_IN_OUTPUT_MODE, unit] = -2.0 / S  # clear output mode
    ffn.W_down[BD.NEXT_THINKING_START, unit] = 2.0 / S  # emit THINKING_START
    unit += 1



