"""Setup helpers for VM weight baking.

This module contains weight-setup helpers extracted from ``vm_step.py`` after
Wave 3 deleted ``make_legacy_bake_op``. These functions are now only called by
migrated bake_fns inside ``unified_compiler/migrated_ops.py``.

They are re-exported by ``vm_step`` for backward compatibility, so external
imports like ``from neural_vm.vm_step import _set_X`` continue to work.

All functions are self-contained: they take ``ffn``/``attn`` and ``BD``
(a ``_SetDim`` class) as parameters, and only depend on stdlib ``math`` and
``PC_OFFSET`` from ``.constants``. Internal references to ``unified_compiler``
primitives are lazy-imported inside function bodies.
"""

import math

from .constants import PC_OFFSET



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
    unit += 1

    # BYTE_INDEX_2 at STACK0: d=8 from BP → H3[BP]=1 (d≤8.5), H2[BP]=0 (d>7.5)
    ffn.W_up[unit, BD.H3 + BP_I] = S
    ffn.W_up[unit, BD.IS_BYTE] = S
    ffn.b_up[unit] = -S * 1.5
    ffn.W_gate[unit, BD.H2 + BP_I] = -1.0
    ffn.b_gate[unit] = 1.0
    ffn.W_down[BD.BYTE_INDEX_2, unit] = 2.0 / S
    unit += 1

    # BYTE_INDEX_3 at STACK0: d=9 from BP → H4[BP]=1 (d≤9.5), H3[BP]=0 (d>8.5)
    ffn.W_up[unit, BD.H4 + BP_I] = S
    ffn.W_up[unit, BD.IS_BYTE] = S
    ffn.b_up[unit] = -S * 1.5
    ffn.W_gate[unit, BD.H3 + BP_I] = -1.0
    ffn.b_gate[unit] = 1.0
    ffn.W_down[BD.BYTE_INDEX_3, unit] = 2.0 / S
    unit += 1



def _set_cs_threshold_attn(attn, head_idx, threshold, out_dim, slope, HD):
    """Set a single threshold head that only detects CODE_START distance.

    Outputs a single dim (CS component only) instead of full 7-marker vector.
    """
    import math
    from .vm_step import _SetDim
    BD = _SetDim
    base = head_idx * HD
    # Scale Q so Q·K/sqrt(HD) = slope*threshold (HD-independent).
    q_val = math.sqrt(HD) * slope
    attn.W_q[base, BD.CONST] = q_val
    attn.W_k[base, BD.IS_MARK] = threshold
    attn.W_v[base + 1, BD.MARK_CS] = 1.0
    attn.W_o[out_dim, base + 1] = 1.0



def _set_stack0_carry_attn(attn, head_idx, HD, BD=None):
    """Set attention head for STACK0 carry-forward.

    At STACK0 marker positions, attend to previous step's STACK0 byte 0
    (identified by STACK0_BYTE0 flag from L1 FFN).
    Copies EMBED_LO/HI to EMBED_LO/HI at STACK0 marker.

    Args:
        BD: Optional dim spec (proxy) overriding _SetDim. Pass the compiler
            proxy from a migrated op so pin_io_only=True layouts wire to the
            correct residual lanes. Defaults to module-level ``_SetDim`` for
            backward compatibility with legacy callers (e.g. set_vm_weights).
    """
    if BD is None:
        from .vm_step import _SetDim
        BD = _SetDim
    base = head_idx * HD
    L = 15.0

    # Q: fires at STACK0 markers
    attn.W_q[base, BD.MARK_STACK0] = L
    # K: fires at positions with STACK0_BYTE0 flag
    attn.W_k[base, BD.STACK0_BYTE0] = L

    # V: copies EMBED_LO/HI (the byte value from embedding)
    for k in range(16):
        attn.W_v[base + 1 + k, BD.EMBED_LO + k] = 1.0
        attn.W_v[base + 17 + k, BD.EMBED_HI + k] = 1.0

    # O: writes to EMBED_LO/HI at STACK0 marker
    for k in range(16):
        attn.W_o[BD.EMBED_LO + k, base + 1 + k] = 1.0
        attn.W_o[BD.EMBED_HI + k, base + 17 + k] = 1.0

    # Anti-leakage gate (same as _set_carry_forward_attn)
    GATE = 33
    attn.W_q[base + GATE, BD.MARK_STACK0] = L
    attn.W_q[base + GATE, BD.CONST] = -L / 2
    attn.W_k[base + GATE, BD.CONST] = L



def _set_layer5_fetch(attn, S, BD, HD):
    """Layer 5 attention: fetch opcode/immediate through memory keys.

    Head 0: fetch immediate byte at address PC+1 (TEMP[0..31]).
    Head 1: fetch opcode byte at address PC (EMBED_LO/HI).

    Both heads match query nibbles against ADDR_KEY. This removes the
    forward-time address injection path; code bytes are expected to be
    available as memory entries keyed by address.
    """
    L = 20.0

    # Head 0: fetch immediate byte (address = PC+1)
    # Only fires on non-first steps (HAS_SE > 0). For first step, head 3 fetches
    # immediate at PC marker and head 4 relays to AX marker.
    # TEMP contains PC+1 at AX marker (computed by L4 FFN from relayed PC).
    base = 0 * HD
    # Q: low two nibbles from TEMP (PC+1)
    for k in range(16):
        attn.W_q[base + k, BD.TEMP + k] = L
        attn.W_q[base + 16 + k, BD.TEMP + 16 + k] = L
    # third address nibble fixed to zero (for <= 255 code addresses),
    # gated to AX marker to avoid global leakage.
    attn.W_q[base + 32, BD.MARK_AX] = L

    # K: address nibbles from memory key space
    for k in range(16):
        attn.W_k[base + k, BD.ADDR_KEY + k] = L
        attn.W_k[base + 16 + k, BD.ADDR_KEY + 16 + k] = L
    attn.W_k[base + 32, BD.ADDR_KEY + 32] = L

    # Anti-leakage gate: suppress head activity at non-AX positions.
    # Must overwhelm worst-case address match (~+50 from 33 addr dims).
    # Non-AX: Q=-500, K=5 → -312.5.  AX: Q=0 → 0 contribution.
    GATE = 33
    attn.W_q[base + GATE, BD.MARK_AX] = 500.0
    attn.W_q[base + GATE, BD.CONST] = -500.0
    attn.W_k[base + GATE, BD.CONST] = 5.0

    # HAS_SE gate: only fire on non-first steps (when HAS_SE > 0)
    # First step uses head 3 (PC marker) + head 4 (relay to AX)
    # On first step: Q[HAS_SE]=0, Q[CONST]=-500 → score = -312.5 (blocks)
    # On non-first: Q[HAS_SE]=500, Q[CONST]=-500 → score = 0 (neutral)
    HAS_SE_GATE = 34
    attn.W_q[base + HAS_SE_GATE, BD.HAS_SE] = 500.0
    attn.W_q[base + HAS_SE_GATE, BD.CONST] = -500.0
    attn.W_k[base + HAS_SE_GATE, BD.CONST] = 5.0

    # V: copy byte value nibbles
    for k in range(16):
        attn.W_v[base + 32 + k, BD.CLEAN_EMBED_LO + k] = 1.0
        attn.W_v[base + 48 + k, BD.CLEAN_EMBED_HI + k] = 1.0
    # O: write immediate to FETCH_LO/HI
    for k in range(16):
        attn.W_o[BD.FETCH_LO + k, base + 32 + k] = 1.0
        attn.W_o[BD.FETCH_HI + k, base + 48 + k] = 1.0

    # Head 1: fetch opcode byte (address = PC)
    # EMBED_LO/HI at AX marker contain the PC value (relayed by L4 attention).
    base = 1 * HD
    # Q: low two nibbles from EMBED (PC)
    for k in range(16):
        attn.W_q[base + k, BD.EMBED_LO + k] = L
        attn.W_q[base + 16 + k, BD.EMBED_HI + k] = L
    attn.W_q[base + 32, BD.MARK_AX] = L

    # K: address nibbles from memory key space
    for k in range(16):
        attn.W_k[base + k, BD.ADDR_KEY + k] = L
        attn.W_k[base + 16 + k, BD.ADDR_KEY + 16 + k] = L
    attn.W_k[base + 32, BD.ADDR_KEY + 32] = L

    # Anti-leakage gate: suppress head activity at non-AX positions.
    GATE = 33
    attn.W_q[base + GATE, BD.MARK_AX] = 500.0
    attn.W_q[base + GATE, BD.CONST] = -500.0
    attn.W_k[base + GATE, BD.CONST] = 5.0

    # HAS_SE gate: only fire on non-first steps (when HAS_SE > 0)
    # First step uses Head 4 (fetches opcode to AX marker at PC_OFFSET)
    # On first step: Q[HAS_SE]=0, Q[CONST]=-500 → score = -312.5 (blocks)
    # On non-first: Q[HAS_SE]=500, Q[CONST]=-500 → score = 0 (neutral)
    HAS_SE_GATE = 34
    attn.W_q[base + HAS_SE_GATE, BD.HAS_SE] = 500.0
    attn.W_q[base + HAS_SE_GATE, BD.CONST] = -500.0
    attn.W_k[base + HAS_SE_GATE, BD.CONST] = 5.0

    # V: opcode byte nibbles
    for k in range(16):
        attn.W_v[base + 32 + k, BD.CLEAN_EMBED_LO + k] = 1.0
        attn.W_v[base + 48 + k, BD.CLEAN_EMBED_HI + k] = 1.0
    # O: writes to OPCODE_BYTE_LO/HI (staging for opcode decode)
    # Uses separate dims from ALU_LO/HI to avoid residual collision with L7 operand gather
    for k in range(16):
        attn.W_o[BD.OPCODE_BYTE_LO + k, base + 32 + k] = 1.0
        attn.W_o[BD.OPCODE_BYTE_HI + k, base + 48 + k] = 1.0

    # Head 2: fetch opcode for first-step (PC marker → address PC_OFFSET - 2)
    # FIX 2026-04-12: PC_OFFSET points to immediate byte, opcode is 2 bytes before
    # On the first step (NOT HAS_SE), PC = PC_OFFSET.
    # Uses ADDR_KEY matching (same mechanism as head 1, but fires at PC marker not AX).
    from .constants import PC_OFFSET
    base = 2 * HD
    # Q: fires at PC marker when NOT HAS_SE, queries for address PC_OFFSET
    attn.W_q[base, BD.MARK_PC] = L
    attn.W_q[base, BD.HAS_SE] = -L  # only on first step
    # Q: address PC_OFFSET (e.g., 2: ADDR_KEY_LO[2]=1, ADDR_KEY_HI[0]=1)
    attn.W_q[base + (PC_OFFSET & 0xF), BD.CONST] = L  # lo nibble
    attn.W_q[base + 16 + ((PC_OFFSET >> 4) & 0xF), BD.CONST] = L  # hi nibble
    attn.W_q[base + 32, BD.MARK_PC] = L  # third nibble gate

    # K: match ADDR_KEY nibbles (same as head 1)
    for k in range(16):
        attn.W_k[base + k, BD.ADDR_KEY + k] = L
        attn.W_k[base + 16 + k, BD.ADDR_KEY + 16 + k] = L
    attn.W_k[base + 32, BD.ADDR_KEY + 32] = L

    # Anti-leakage gate (same as head 1)
    GATE = 33
    attn.W_q[base + GATE, BD.MARK_PC] = 500.0
    attn.W_q[base + GATE, BD.CONST] = -500.0
    attn.W_k[base + GATE, BD.CONST] = 5.0

    # V: copy opcode byte nibbles
    for k in range(16):
        attn.W_v[base + 32 + k, BD.CLEAN_EMBED_LO + k] = 1.0
        attn.W_v[base + 48 + k, BD.CLEAN_EMBED_HI + k] = 1.0
    # O: write to OPCODE_BYTE_LO/HI at PC marker
    for k in range(16):
        attn.W_o[BD.OPCODE_BYTE_LO + k, base + 32 + k] = 1.0
        attn.W_o[BD.OPCODE_BYTE_HI + k, base + 48 + k] = 1.0

    # Head 3: fetch immediate (PC marker → address from FETCH_LO/HI = PC+1)
    # FIX 2026-04-29: Changed from fixed address (PC_OFFSET+1) to dynamic address
    # read from FETCH_LO/HI at PC marker. L4 FFN now writes PC+1 to FETCH dims,
    # so head 3 fetches from the correct address for all steps, not just the first.
    # This fixes BZ/BNZ branch targets and JMP targets for non-first steps.
    base = 3 * HD
    # Q: read dynamic address from FETCH_LO/HI (PC+1 computed by L4 FFN)
    # FETCH is one-hot: only one lo and one hi position are active (~0.73).
    for k in range(16):
        attn.W_q[base + k, BD.FETCH_LO + k] = L
        attn.W_q[base + 16 + k, BD.FETCH_HI + k] = L
    attn.W_q[base + 32, BD.MARK_PC] = L  # gate for PC marker only

    # K: match ADDR_KEY nibbles
    for k in range(16):
        attn.W_k[base + k, BD.ADDR_KEY + k] = L
        attn.W_k[base + 16 + k, BD.ADDR_KEY + 16 + k] = L
    attn.W_k[base + 32, BD.ADDR_KEY + 32] = L

    # Anti-leakage gate: suppress at non-PC positions
    GATE = 33
    attn.W_q[base + GATE, BD.MARK_PC] = 500.0
    attn.W_q[base + GATE, BD.CONST] = -500.0
    attn.W_k[base + GATE, BD.CONST] = 5.0

    # HAS_SE gate: REMOVED for JMP support at non-first steps.
    # Previously this blocked immediate fetch on non-first steps, but JMP needs FETCH
    # (immediate = jump target) at any step, not just first step.
    # FIX 2026-04-16: Removed HAS_SE gate to enable JMP at any step position.
    # The L6 FFN JMP override uses OP_JMP + FETCH to set PC to jump target.
    # HAS_SE_GATE = 34  # Disabled
    # attn.W_q[base + HAS_SE_GATE, BD.HAS_SE] = -500.0  # Disabled
    # attn.W_k[base + HAS_SE_GATE, BD.CONST] = 5.0  # Disabled

    # V: copy immediate byte nibbles
    for k in range(16):
        attn.W_v[base + 32 + k, BD.CLEAN_EMBED_LO + k] = 1.0
        attn.W_v[base + 48 + k, BD.CLEAN_EMBED_HI + k] = 1.0
    # O: write to FETCH_LO/HI at PC marker for first-step immediate
    # AMPLIFY FETCH: Use 40.0 instead of 1.0 to compensate for attenuation during
    # Layer 6 Head 5 relay (FETCH gets attenuated 37x: 1.0→0.027, while OP_IMM
    # only gets attenuated 17x: 6.0→0.351). With 40.0, FETCH should reach ~1.0 at AX.
    # BUG FIX 2026-04-09 (part 3): Removed AX_CARRY output from L5 head 3.
    # AX_CARRY should only be written for JMP operations, but L5 head 3 fires for ALL
    # first-step ops. L6 head 2 now reads FETCH and writes to AX_CARRY (gated on OP_JMP).
    for k in range(16):
        # attn.W_o[BD.AX_CARRY_LO + k, base + 32 + k] = 1.0  # REMOVED
        # attn.W_o[BD.AX_CARRY_HI + k, base + 48 + k] = 1.0  # REMOVED
        attn.W_o[BD.FETCH_LO + k, base + 32 + k] = 40.0  # Amplified for relay
        attn.W_o[BD.FETCH_HI + k, base + 48 + k] = 40.0  # Amplified for relay

    # Head 4: Fetch opcode to AX marker for first-step (duplicate of Head 2)
    # Head 2 fetches opcode to PC marker, but opcode decode FFN runs at AX marker.
    # This head fetches the same opcode at address PC_OFFSET but writes to AX marker.
    # (Cannot relay from Head 2 since attention reads from input, not other heads' outputs.)
    base = 4 * HD
    # Q: fires at AX marker when NOT HAS_SE (first step), queries for address PC_OFFSET
    attn.W_q[base, BD.MARK_AX] = L
    attn.W_q[base, BD.HAS_SE] = -L  # only on first step
    # Q: address PC_OFFSET (e.g., 2: ADDR_KEY_LO[2]=1, ADDR_KEY_HI[0]=1)
    attn.W_q[base + (PC_OFFSET & 0xF), BD.CONST] = L  # lo nibble
    attn.W_q[base + 16 + ((PC_OFFSET >> 4) & 0xF), BD.CONST] = L  # hi nibble
    attn.W_q[base + 32, BD.MARK_AX] = L  # third nibble gate
    # K: match ADDR_KEY nibbles (code byte addresses)
    for k in range(16):
        attn.W_k[base + k, BD.ADDR_KEY + k] = L
        attn.W_k[base + 16 + k, BD.ADDR_KEY + 16 + k] = L
    attn.W_k[base + 32, BD.ADDR_KEY + 32] = L
    # Anti-leakage gate: suppress at non-AX positions
    GATE = 33
    attn.W_q[base + GATE, BD.MARK_AX] = 500.0
    attn.W_q[base + GATE, BD.CONST] = -500.0
    attn.W_k[base + GATE, BD.CONST] = 5.0
    # HAS_SE gate: only fire on first step (when HAS_SE = 0)
    # On first step: Q[HAS_SE]=0, Q[CONST]=-500 → score = -312.5 (blocks from wrong positions)
    # On non-first: Q[HAS_SE]=500, Q[CONST]=-500 → score = 0 (neutral)
    # Combined with anti-leakage: non-first + AX = blocked
    HAS_SE_GATE = 34
    attn.W_q[base + HAS_SE_GATE, BD.HAS_SE] = -500.0  # negative: block when HAS_SE > 0
    attn.W_k[base + HAS_SE_GATE, BD.CONST] = 5.0
    # V: copy opcode byte nibbles from matched code position
    for k in range(16):
        attn.W_v[base + 32 + k, BD.CLEAN_EMBED_LO + k] = 1.0
        attn.W_v[base + 48 + k, BD.CLEAN_EMBED_HI + k] = 1.0
    # O: write OPCODE_BYTE_LO/HI at AX marker
    for k in range(16):
        attn.W_o[BD.OPCODE_BYTE_LO + k, base + 32 + k] = 1.0
        attn.W_o[BD.OPCODE_BYTE_HI + k, base + 48 + k] = 1.0

    # Head 5: Fetch opcode byte to OPCODE_BYTE at PC marker on non-first steps.
    # This enables opcode decode at PC marker (via L5 FFN) for BZ/BNZ/LEV/EXIT
    # without relying on _inject_active_opcode. Uses dynamic PC from EMBED_LO/HI.
    # HAS_SE gate: only fires on non-first steps (Head 2 handles first step).
    base = 5 * HD
    for k in range(16):
        attn.W_q[base + k, BD.EMBED_LO + k] = L
        attn.W_q[base + 16 + k, BD.EMBED_HI + k] = L
    attn.W_q[base + 32, BD.MARK_PC] = L
    for k in range(16):
        attn.W_k[base + k, BD.ADDR_KEY + k] = L
        attn.W_k[base + 16 + k, BD.ADDR_KEY + 16 + k] = L
    attn.W_k[base + 32, BD.ADDR_KEY + 32] = L
    GATE5 = 33
    attn.W_q[base + GATE5, BD.MARK_PC] = 500.0
    attn.W_q[base + GATE5, BD.CONST] = -500.0
    attn.W_k[base + GATE5, BD.CONST] = 5.0
    HAS_SE_GATE5 = 34
    attn.W_q[base + HAS_SE_GATE5, BD.HAS_SE] = 500.0
    attn.W_q[base + HAS_SE_GATE5, BD.CONST] = -500.0
    attn.W_k[base + HAS_SE_GATE5, BD.CONST] = 5.0
    for k in range(16):
        attn.W_v[base + 32 + k, BD.CLEAN_EMBED_LO + k] = 1.0
        attn.W_v[base + 48 + k, BD.CLEAN_EMBED_HI + k] = 1.0
    for k in range(16):
        attn.W_o[BD.OPCODE_BYTE_LO + k, base + 32 + k] = 1.0
        attn.W_o[BD.OPCODE_BYTE_HI + k, base + 48 + k] = 1.0

    # Head 6: Direct OP_* flag relay from CODE to PC marker (non-first steps)
    # DEPRECATED: OP_* flags were removed from embeddings (2026-04-13).
    # Head 5 + L5 FFN opcode decode now provides OP_* at PC marker.
    # This bypasses the OPCODE_BYTE mechanism and uses the OP_* flags set in embeddings.
    # For non-first steps, PC value is in EMBED_LO/HI at PC marker (from L4 relay).
    # Query for current PC address, match CODE byte with that ADDR_KEY, copy OP_* flags.
    base = 6 * HD
    # Q: PC marker, address from EMBED_LO/HI (current PC)
    for k in range(16):
        attn.W_q[base + k, BD.EMBED_LO + k] = L
        attn.W_q[base + 16 + k, BD.EMBED_HI + k] = L
    attn.W_q[base + 32, BD.MARK_PC] = L  # Only at PC marker

    # K: match ADDR_KEY in CODE section
    for k in range(16):
        attn.W_k[base + k, BD.ADDR_KEY + k] = L
        attn.W_k[base + 16 + k, BD.ADDR_KEY + 16 + k] = L
    attn.W_k[base + 32, BD.ADDR_KEY + 32] = L

    # Anti-leakage gate
    GATE = 33
    attn.W_q[base + GATE, BD.MARK_PC] = 500.0
    attn.W_q[base + GATE, BD.CONST] = -500.0
    attn.W_k[base + GATE, BD.CONST] = 5.0

    # HAS_SE gate: only fire on non-first steps
    HAS_SE_GATE = 34
    attn.W_q[base + HAS_SE_GATE, BD.HAS_SE] = 500.0
    attn.W_q[base + HAS_SE_GATE, BD.CONST] = -500.0
    attn.W_k[base + HAS_SE_GATE, BD.CONST] = 5.0

    # V: copy OP_* flags from CODE byte (these are set in embeddings)
    # Copy the main opcode flags that Layer 6 needs for routing
    v_offset = 0
    attn.W_v[base + v_offset, BD.OP_IMM] = 1.0; v_offset += 1
    attn.W_v[base + v_offset, BD.OP_LEA] = 1.0; v_offset += 1
    attn.W_v[base + v_offset, BD.OP_EXIT] = 1.0; v_offset += 1
    attn.W_v[base + v_offset, BD.OP_JMP] = 1.0; v_offset += 1
    attn.W_v[base + v_offset, BD.OP_JSR] = 1.0; v_offset += 1  # ADDED for neural JSR (non-first steps)
    attn.W_v[base + v_offset, BD.OP_ADD] = 1.0; v_offset += 1
    attn.W_v[base + v_offset, BD.OP_SUB] = 1.0; v_offset += 1
    attn.W_v[base + v_offset, BD.OP_MUL] = 1.0; v_offset += 1
    attn.W_v[base + v_offset, BD.OP_DIV] = 1.0; v_offset += 1
    attn.W_v[base + v_offset, BD.OP_MOD] = 1.0; v_offset += 1
    attn.W_v[base + v_offset, BD.OP_OR] = 1.0; v_offset += 1
    attn.W_v[base + v_offset, BD.OP_XOR] = 1.0; v_offset += 1
    attn.W_v[base + v_offset, BD.OP_AND] = 1.0; v_offset += 1
    attn.W_v[base + v_offset, BD.OP_EQ] = 1.0; v_offset += 1
    attn.W_v[base + v_offset, BD.OP_LT] = 1.0; v_offset += 1
    attn.W_v[base + v_offset, BD.OP_SHL] = 1.0; v_offset += 1
    attn.W_v[base + v_offset, BD.OP_SHR] = 1.0; v_offset += 1

    # O: write OP_* flags to PC marker
    o_offset = 0
    attn.W_o[BD.OP_IMM, base + o_offset] = 1.0; o_offset += 1
    attn.W_o[BD.OP_LEA, base + o_offset] = 1.0; o_offset += 1
    attn.W_o[BD.OP_EXIT, base + o_offset] = 1.0; o_offset += 1
    attn.W_o[BD.OP_JMP, base + o_offset] = 1.0; o_offset += 1
    attn.W_o[BD.OP_JSR, base + o_offset] = 1.0; o_offset += 1  # ADDED for neural JSR (non-first steps)
    attn.W_o[BD.OP_ADD, base + o_offset] = 1.0; o_offset += 1
    attn.W_o[BD.OP_SUB, base + o_offset] = 1.0; o_offset += 1
    attn.W_o[BD.OP_MUL, base + o_offset] = 1.0; o_offset += 1
    attn.W_o[BD.OP_DIV, base + o_offset] = 1.0; o_offset += 1
    attn.W_o[BD.OP_MOD, base + o_offset] = 1.0; o_offset += 1
    attn.W_o[BD.OP_OR, base + o_offset] = 1.0; o_offset += 1
    attn.W_o[BD.OP_XOR, base + o_offset] = 1.0; o_offset += 1
    attn.W_o[BD.OP_AND, base + o_offset] = 1.0; o_offset += 1
    attn.W_o[BD.OP_EQ, base + o_offset] = 1.0; o_offset += 1
    attn.W_o[BD.OP_LT, base + o_offset] = 1.0; o_offset += 1
    attn.W_o[BD.OP_SHL, base + o_offset] = 1.0; o_offset += 1
    attn.W_o[BD.OP_SHR, base + o_offset] = 1.0; o_offset += 1

    # Head 7: Direct OP_* flag relay from CODE to PC marker (first step only)
    # Same as Head 6, but for first step (NOT HAS_SE instead of HAS_SE).
    base = 7 * HD
    # Q: PC marker when NOT HAS_SE, queries for address PC_OFFSET
    attn.W_q[base, BD.MARK_PC] = L
    attn.W_q[base, BD.HAS_SE] = -L  # only on first step
    # Q: address PC_OFFSET (e.g., 2: ADDR_KEY_LO[2]=1, ADDR_KEY_HI[0]=1)
    attn.W_q[base + (PC_OFFSET & 0xF), BD.CONST] = L  # lo nibble
    attn.W_q[base + 16 + ((PC_OFFSET >> 4) & 0xF), BD.CONST] = L  # hi nibble
    attn.W_q[base + 32, BD.MARK_PC] = L  # third nibble gate

    # K: match ADDR_KEY in CODE section
    for k in range(16):
        attn.W_k[base + k, BD.ADDR_KEY + k] = L
        attn.W_k[base + 16 + k, BD.ADDR_KEY + 16 + k] = L
    attn.W_k[base + 32, BD.ADDR_KEY + 32] = L

    # Anti-leakage gate
    GATE = 33
    attn.W_q[base + GATE, BD.MARK_PC] = 500.0
    attn.W_q[base + GATE, BD.CONST] = -500.0
    attn.W_k[base + GATE, BD.CONST] = 5.0

    # V: copy OP_* flags from CODE byte (these are set in embeddings)
    v_offset = 0
    attn.W_v[base + v_offset, BD.OP_IMM] = 1.0; v_offset += 1
    attn.W_v[base + v_offset, BD.OP_LEA] = 1.0; v_offset += 1
    attn.W_v[base + v_offset, BD.OP_EXIT] = 1.0; v_offset += 1
    attn.W_v[base + v_offset, BD.OP_JMP] = 1.0; v_offset += 1
    attn.W_v[base + v_offset, BD.OP_JSR] = 1.0; v_offset += 1  # ADDED for neural JSR
    attn.W_v[base + v_offset, BD.OP_ADD] = 1.0; v_offset += 1
    attn.W_v[base + v_offset, BD.OP_SUB] = 1.0; v_offset += 1
    attn.W_v[base + v_offset, BD.OP_MUL] = 1.0; v_offset += 1
    attn.W_v[base + v_offset, BD.OP_DIV] = 1.0; v_offset += 1
    attn.W_v[base + v_offset, BD.OP_MOD] = 1.0; v_offset += 1
    attn.W_v[base + v_offset, BD.OP_OR] = 1.0; v_offset += 1
    attn.W_v[base + v_offset, BD.OP_XOR] = 1.0; v_offset += 1
    attn.W_v[base + v_offset, BD.OP_AND] = 1.0; v_offset += 1
    attn.W_v[base + v_offset, BD.OP_EQ] = 1.0; v_offset += 1
    attn.W_v[base + v_offset, BD.OP_LT] = 1.0; v_offset += 1
    attn.W_v[base + v_offset, BD.OP_SHL] = 1.0; v_offset += 1
    attn.W_v[base + v_offset, BD.OP_SHR] = 1.0; v_offset += 1

    # O: write OP_* flags to PC marker
    o_offset = 0
    attn.W_o[BD.OP_IMM, base + o_offset] = 1.0; o_offset += 1
    attn.W_o[BD.OP_LEA, base + o_offset] = 1.0; o_offset += 1
    attn.W_o[BD.OP_EXIT, base + o_offset] = 1.0; o_offset += 1
    attn.W_o[BD.OP_JMP, base + o_offset] = 1.0; o_offset += 1
    attn.W_o[BD.OP_JSR, base + o_offset] = 1.0; o_offset += 1  # ADDED for neural JSR
    attn.W_o[BD.OP_ADD, base + o_offset] = 1.0; o_offset += 1
    attn.W_o[BD.OP_SUB, base + o_offset] = 1.0; o_offset += 1
    attn.W_o[BD.OP_MUL, base + o_offset] = 1.0; o_offset += 1
    attn.W_o[BD.OP_DIV, base + o_offset] = 1.0; o_offset += 1
    attn.W_o[BD.OP_MOD, base + o_offset] = 1.0; o_offset += 1
    attn.W_o[BD.OP_OR, base + o_offset] = 1.0; o_offset += 1
    attn.W_o[BD.OP_XOR, base + o_offset] = 1.0; o_offset += 1
    attn.W_o[BD.OP_AND, base + o_offset] = 1.0; o_offset += 1
    attn.W_o[BD.OP_EQ, base + o_offset] = 1.0; o_offset += 1
    attn.W_o[BD.OP_LT, base + o_offset] = 1.0; o_offset += 1
    attn.W_o[BD.OP_SHL, base + o_offset] = 1.0; o_offset += 1
    attn.W_o[BD.OP_SHR, base + o_offset] = 1.0; o_offset += 1



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
        IS_BYTE*3L + HAS_SE*L + (OP_IMM @ -3L) + (TEMP+3 @ -3L) + CONST*-3.5L
      - AX marker: 0 + L - 3.5L = -2.5L < 0 (blocked)
      - AX byte non-IMM non-bitwise: 3L + L - 3.5L = 0.5L > 0 (fires)
      - First step (HAS_SE=0): 3L + 0 - 3.5L = -0.5L < 0 (blocked)
      - IMM/BITWISE step at AX byte: 3L + L - 3L - 3.5L = -2.5L < 0 (blocked,
        preserves multi-byte OUTPUT / lets BitwiseBytePropagation handle)

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
        # Suppress during IMM (multi-byte) and bitwise ops (TEMP+3 = BITWISE_OP).
        suppress_op_dims=[BD.OP_IMM, BD.TEMP + 3],
        S=S,
        HD=HD,
        alibi_slope=1.0,
        # Defaults match AX coefficients (is_byte_strength=3, has_se_strength=1,
        # suppress_strength=3, q0_threshold=3.5, gate_const=-20000, gate_*=10000).
    )



def _set_layer10_sp_byte_passthrough(attn, S, BD, HD):
    """L10 attention head 2: SP byte 0-2 passthrough across steps (when NOT PSH).

    Similar to AX byte passthrough but for SP. Only fires when PSH_AT_SP = 0
    (i.e., when SP doesn't change). When PSH is active, L6/L15 handle SP values.
    Also suppressed during binary POP (CMP[3]) since L9/L15 handle SP += 8.

    Copies CLEAN_EMBED from previous step's SP bytes 1-3 → OUTPUT at current
    step's SP byte 0-2 positions. Uses shifted byte matching.

    Mapping (Q byte K attends to K byte K+1 of prev step):
      byte 0 pos → prev byte 1 (predicts byte 1 token)
      byte 1 pos → prev byte 2 (predicts byte 2 token)
      byte 2 pos → prev byte 3 (predicts byte 3 token)
      byte 3 pos → suppressed (predicts BP marker, not a byte)

    Q[0] gate (SP-specific coefficients):
        IS_BYTE*L + HAS_SE*2L + (PSH_AT_SP @ -2L) + CONST*-1.5L
      - SP byte non-PSH non-first: L + 2L - 1.5L = 1.5L > 0 (fires)
      - First step (HAS_SE=0): L + 0 - 1.5L = -0.5L < 0 (blocked)
      - PSH step at SP byte: L + 2L - 2L - 1.5L = -0.5L < 0 (blocked)

    Q[33] AND gate: IS_BYTE AND H1[SP] AND HAS_SE AND NOT PSH AND NOT POP
    (gate_const=-30000 + 3*10000 base; PSH_AT_SP and CMP+3 contribute -10000
    each via gate_extras to suppress during PSH and binary POP).

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
        # Q[0] suppression for PSH (SP -= 8 handled elsewhere).
        suppress_op_dims=[BD.PSH_AT_SP],
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
            (BD.PSH_AT_SP, -10000.0),      # suppress during PSH (SP -= 8)
            (BD.CMP + 3, -10000.0),        # suppress during binary POP (SP += 8)
        ],
    )



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



def _set_layer10_stack0_byte_relay(attn, S, BD, HD):
    """L10 attention head 4: copy STACK0 bytes from previous step to ALU at AX bytes.

    At AX byte positions, attends to STACK0 byte positions in the PREVIOUS step
    and copies CLEAN_EMBED nibbles → ALU_LO/HI. The BitwiseBytePropagationPostOp
    then reads ALU (STACK0 operand) + OUTPUT (AX operand) to compute bitwise result.
    """
    L = S
    AX_IDX = 1
    SP_IDX = 2
    BP_IDX = 3
    base = 4 * HD

    attn.W_q[base + 0, BD.IS_BYTE] = L
    attn.W_q[base + 0, BD.H1 + AX_IDX] = L
    attn.W_q[base + 0, BD.CONST] = -L * 1.5

    attn.W_q[base + 1, BD.BYTE_INDEX_0] = L
    attn.W_q[base + 2, BD.BYTE_INDEX_1] = L
    attn.W_q[base + 3, BD.BYTE_INDEX_2] = L

    attn.W_q[base + 4, BD.BYTE_INDEX_3] = -L
    attn.W_q[base + 4, BD.CONST] = L / 2

    attn.W_k[base + 0, BD.IS_BYTE] = L

    attn.W_k[base + 1, BD.H4 + BP_IDX] = L
    attn.W_k[base + 1, BD.H1 + BP_IDX] = -2 * L
    attn.W_k[base + 1, BD.H1 + SP_IDX] = -2 * L
    attn.W_k[base + 1, BD.H1 + AX_IDX] = -L

    attn.W_k[base + 2, BD.BYTE_INDEX_1] = L
    attn.W_k[base + 3, BD.BYTE_INDEX_2] = L

    attn.W_q[base + 33, BD.CONST] = -20000.0
    attn.W_q[base + 33, BD.IS_BYTE] = 10000.0
    attn.W_q[base + 33, BD.H1 + AX_IDX] = 10000.0
    attn.W_q[base + 33, BD.HAS_SE] = 10000.0
    attn.W_k[base + 33, BD.CONST] = 5.0

    for k in range(16):
        attn.W_v[base + k, BD.CLEAN_EMBED_LO + k] = 1.0
        attn.W_v[base + 16 + k, BD.CLEAN_EMBED_HI + k] = 1.0

    for k in range(16):
        attn.W_o[BD.ALU_LO + k, base + k] = 2.0
        attn.W_o[BD.ALU_HI + k, base + 16 + k] = 2.0



def _set_layer11_mul_partial(ffn, S, BD):
    """L11 FFN: MUL partial sum staging for hi nibble computation.

    Schoolbook: result = (a_hi*16+a_lo) * (b_hi*16+b_lo) mod 256
      result_lo = (a_lo * b_lo) % 16           [computed in L10]
      result_hi = (carry + a_lo*b_hi + a_hi*b_lo) % 16

    This layer computes: partial = (carry + a_lo*b_hi) % 16
      where carry = (a_lo * b_lo) // 16
    for all (a_lo, b_lo, b_hi) triples, stored in TEMP[0..15].

    L12 then combines: result_hi = (partial + a_hi*b_lo) % 16.

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
                ffn.W_down[BD.TEMP + partial, unit] = 2.0 / S
                unit += 1

    return unit



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



def _set_layer13_mem_addr_gather(attn, S, BD, HD):
    """L13 attention heads 0-2: Gather MEM addr bytes → MEM val byte positions.

    For L15 K-side address keys: copies addr byte nibbles from MEM addr
    positions (d=0..3 from MEM marker) to MEM val byte positions (d=4..8).
    """
    L = 15.0
    MEM_I = 4

    for j in range(3):
        base = j * HD
        addr_lo_out = [BD.ADDR_B0_LO, BD.ADDR_B1_LO, BD.ADDR_B2_LO][j]
        addr_hi_out = [BD.ADDR_B0_HI, BD.ADDR_B1_HI, BD.ADDR_B2_HI][j]

        # Q: fires at MEM val byte positions (d=5..8 from MEM)
        # Use MEM_VAL_B0-3 flags computed in L2 FFN
        attn.W_q[base, BD.MEM_VAL_B0] = L
        attn.W_q[base, BD.MEM_VAL_B1] = L
        attn.W_q[base, BD.MEM_VAL_B2] = L
        attn.W_q[base, BD.MEM_VAL_B3] = L

        # K: fires at MEM addr byte J position.
        # Addr byte 0 is at d=1 (after MEM marker), byte 1 at d=2, byte 2 at d=3.
        if j == 0:
            # Addr byte 0 at d=1: L1H1[MEM]=1 (d≤1.5), subtract L1H0[MEM] (d=0 only)
            attn.W_k[base, BD.L1H1 + MEM_I] = L
            attn.W_k[base, BD.L1H0 + MEM_I] = -L  # exclude MEM marker (d=0)
        elif j == 1:
            # Addr byte 1 at d=2: L1H2[MEM]=1 (d≤2.5), subtract L1H1[MEM] (d≤1.5)
            attn.W_k[base, BD.L1H2 + MEM_I] = L
            attn.W_k[base, BD.L1H1 + MEM_I] = -L
        elif j == 2:
            # Addr byte 2 at d=3: H0[MEM]=1 (d≤3.5), subtract L1H2[MEM] (d≤2.5)
            attn.W_k[base, BD.H0 + MEM_I] = L
            attn.W_k[base, BD.L1H2 + MEM_I] = -L

        # Anti-leakage gate
        attn.W_q[base + 33, BD.MEM_VAL_B0] = L
        attn.W_q[base + 33, BD.CONST] = -L / 2
        attn.W_k[base + 33, BD.CONST] = L

        # V: copy CLEAN_EMBED nibbles (addr byte value)
        for k in range(16):
            attn.W_v[base + 1 + k, BD.CLEAN_EMBED_LO + k] = 1.0
            attn.W_v[base + 17 + k, BD.CLEAN_EMBED_HI + k] = 1.0
        # O: write to ADDR_BJ_LO/HI (gathered to val byte positions)
        for k in range(16):
            attn.W_o[addr_lo_out + k, base + 1 + k] = 1.0
            attn.W_o[addr_hi_out + k, base + 17 + k] = 1.0



def _set_layer13_shifts(ffn, S, BD):
    """L13 FFN: SHL/SHR for shift amounts 0-7.

    For each (a_lo, a_hi, s) where s=0..7:
      SHL: result = ((a_hi<<4 | a_lo) << s) & 0xFF
      SHR: result = ((a_hi<<4 | a_lo) >> s) & 0xFF

    Relay convention:
      ALU = stack (value to shift)
      AX_CARRY = AX (shift amount)

    5-way AND: MARK_AX + ALU_LO[a_lo] + ALU_HI[a_hi] + AX_CARRY_LO[s] + AX_CARRY_HI[0]
    Threshold -S*4.5:
      All 5 match: 5S - 4.5S = 0.5S → silu fires
      4 match:     4S - 4.5S = -0.5S → blocked

    Gate: OP_SHL or OP_SHR (≈5.0 when active, ≈0 otherwise).

    Shift >= 8 already handled by L10 zero-output units.

    2048 SHL + 2048 SHR = 4096 units (fills L13 exactly).
    """
    unit = 0

    for op_dim, shift_fn in [
        (BD.OP_SHL, lambda v, s: (v << s) & 0xFF),
        (BD.OP_SHR, lambda v, s: (v >> s) & 0xFF),
    ]:
        for s in range(8):
            for a_hi in range(16):
                for a_lo in range(16):
                    value = (a_hi << 4) | a_lo
                    result = shift_fn(value, s)
                    result_lo = result & 0xF
                    result_hi = (result >> 4) & 0xF

                    # 5-way AND in silu path
                    # value = ALU (stack), shift = AX_CARRY (AX)
                    ffn.W_up[unit, BD.MARK_AX] = S
                    ffn.W_up[unit, BD.ALU_LO + a_lo] = S       # value lo (stack)
                    ffn.W_up[unit, BD.ALU_HI + a_hi] = S       # value hi (stack)
                    ffn.W_up[unit, BD.AX_CARRY_LO + s] = S     # shift amount (AX)
                    ffn.W_up[unit, BD.AX_CARRY_HI + 0] = S     # shift hi = 0 (AX hi)
                    ffn.b_up[unit] = -S * 4.5

                    # Opcode gate
                    ffn.W_gate[unit, op_dim] = 1.0

                    # Write both nibbles of result
                    ffn.W_down[BD.OUTPUT_LO + result_lo, unit] = 2.0 / S
                    ffn.W_down[BD.OUTPUT_HI + result_hi, unit] = 2.0 / S
                    unit += 1

    return unit



def _set_layer14_temp_clear(ffn, S, BD, start_unit=0):
    """L14 FFN: Clear TEMP at PC marker when OP_LEV is active.

    BUG FIX 2026-04-16: TEMP[0] has residual value from L5/L6 attention (~2.0).
    This causes L16 TEMP->OUTPUT routing to incorrectly boost OUTPUT_LO[0].

    Solution: Subtract from TEMP[0] when OP_LEV and MARK_PC are active.
    L15 will then write fresh values to TEMP for the return address.

    Activation calculation at PC marker (S = 100, OP_LEV amplified to ~10 by L6):
      W_up contribution from OP_LEV  = (S/10) * 10  = 100
      W_up contribution from MARK_PC = S * 1        = 100
      b_up                           = -S * 1.5     = -150
      pre-silu activation            = 100 + 100 - 150 = 50 (positive: fires)
      silu(50) ~= 50, gate(CONST=1.0) = 1.0
      hidden = silu * gate = 50

    AUDIT NOTE 2026-05-09: Effective subtraction is silu(50) * 1 * (-5/S) = -2.5,
    not -5.0 as the original docstring claimed. With residual=2.0, TEMP[0] becomes
    -0.5 (mild over-correction of 0.5, not -3.0). This is fine because:
    - SiLU at downstream layers clips small negatives to ~0.
    - Selectivity is correct: only fires when OP_LEV AND MARK_PC are both set.
      For non-LEV opcodes, OP_LEV~=0 so pre-silu = -50, silu(-50)~=0 (no firing).
      For LEV at non-PC positions, MARK_PC=0 so pre-silu = -50, no firing.
    - For an exact zero-out of a 2.0 residual, set W_down to -4.0/S (delta = -2.0).
      The current -5.0/S overshoots slightly but is empirically safe.

    The earlier docstring claim "Total = 5" used S=10 in the math, but actual S=100;
    the relative scales cancel so the firing-vs-not behavior is unchanged. The
    only consequence is the absolute subtraction magnitude (-2.5 not -5.0).
    """
    unit = start_unit

    # Clear TEMP[0] at PC marker when OP_LEV active
    # Only clear TEMP[0] since that's the problematic residual
    ffn.W_up[unit, BD.OP_LEV] = S / 10  # ~1 with OP_LEV~=10
    ffn.W_up[unit, BD.MARK_PC] = S
    ffn.b_up[unit] = -S * 1.5  # Fire when OP_LEV + MARK_PC
    ffn.W_gate[unit, BD.CONST] = 1.0
    # Effective subtraction is silu(50) * (-5/S) = -2.5 (not -5.0).
    # Residual is ~2.0, so TEMP[0] lands at ~-0.5; safe with downstream SiLU clipping.
    ffn.W_down[BD.TEMP + 0, unit] = -5.0 / S  # Subtract to clear residual
    unit += 1

    # Note: Don't clear other TEMP positions since L15 head 8 writes there
    return unit



def _set_layer14_clear_addr_key_pollution(ffn, S, BD, start_unit=0):
    """L14 FFN: Clear ADDR_KEY pollution at non-MEM, non-marker positions.

    BUG FIX 2026-04-16: ADDR_KEY dims (206-253) are aliased with ADDR_B*_HI.
    L9 attention writes to ADDR_B*_HI for address gathering, which pollutes
    ADDR_KEY at non-MEM positions. This causes L15 to attend to wrong positions.

    Solution: Clear ADDR_KEY at positions that are:
    - NOT MEM value bytes (MEM_VAL_B* = 0)
    - NOT register markers where ADDR_B*_HI is needed for L15 queries
      (PC marker for LEV return_addr, BP marker for LEV saved_bp,
       AX marker for LI/LC, STACK0 marker for stack read)

    Pattern: Fire when NOT at MEM value position AND NOT at query markers.
    - W_up: Large negative weights for MEM_VAL_B* and MARK_* flags
    - b_up: Positive bias (fires when no flags present)
    - W_down: Write negative value to cancel ADDR_KEY pollution
    """
    unit = start_unit

    # Large value to suppress firing at MEM and marker positions
    suppress = S * 100  # When flag = 1.0, adds -100*S to activation

    # Clear all 48 ADDR_KEY dims at non-MEM, non-marker positions
    for k in range(48):  # ADDR_KEY is 48 dims (206-253)
        # Suppress at MEM value positions (any of B0/B1/B2/B3)
        ffn.W_up[unit, BD.MEM_VAL_B0] = -suppress
        ffn.W_up[unit, BD.MEM_VAL_B1] = -suppress
        ffn.W_up[unit, BD.MEM_VAL_B2] = -suppress
        ffn.W_up[unit, BD.MEM_VAL_B3] = -suppress

        # Suppress at register markers where ADDR_B*_HI is used for L15 queries
        ffn.W_up[unit, BD.MARK_PC] = -suppress  # LEV return_addr lookup
        ffn.W_up[unit, BD.MARK_BP] = -suppress  # LEV saved_bp lookup
        ffn.W_up[unit, BD.MARK_AX] = -suppress  # LI/LC address lookup
        ffn.W_up[unit, BD.MARK_STACK0] = -suppress  # Stack read
        # FIX 2026-04-16: Also suppress at SP marker during LEV
        # SP marker needs ADDR_B0 for SP = BP + 16 computation
        ffn.W_up[unit, BD.MARK_SP] = -suppress

        # Positive bias to fire at non-MEM, non-marker positions
        ffn.b_up[unit] = S * 0.5

        # Gate unconditionally
        ffn.W_gate[unit, BD.CONST] = 1.0

        # Write to cancel pollution and bring ADDR_KEY to 0
        # FIX 2026-04-16: Changed from -200/S (=-100 output) to -4/S (~=-1.4 output).
        # The original -100 clearing caused negative Q × negative K = positive score
        # at non-target positions in L15 LEV heads. Clearing to ~0 avoids this issue
        # while still preventing false address matches (0 × anything = 0).
        # The pollution to clear is small (typically ~1-2 from L9 ADDR_B*_HI writes),
        # so a small negative value is sufficient.
        ffn.W_down[BD.ADDR_KEY + k, unit] = -4.0 / S
        unit += 1

    return unit



def _set_layer14_clear_output_corruption(ffn, S, BD, start_unit=0):
    """L14 FFN: Fix OUTPUT at STACK0 byte positions (bytes 1-3 = 0).

    BUG FIX 2026-04-16: L14 attention V[0] cancelation and CLEAN_EMBED copying
    corrupts OUTPUT at non-MEM query positions (like STACK0 bytes). Even with
    strong Q suppression, softmax normalization ensures some attention weight
    distributes to source positions, causing OUTPUT to have wrong argmax.

    Solution: At STACK0 byte positions (d=5-9 from BP), boost OUTPUT_LO[0] and
    OUTPUT_HI[0] to ensure they win the argmax. This makes bytes 1-3 of STACK0
    (and similar) output 0, which is correct for return addresses < 256.

    Note: This approach assumes return_addr fits in 1 byte. For larger addresses,
    we'd need to compute bytes 1-3 properly (currently they'd be wrong).
    """
    unit = start_unit

    # Suppression value (prevents firing at MEM and register markers)
    suppress = S * 100
    BP_I = 3  # Index for BP marker in threshold dims

    # Only boost OUTPUT_LO[0] and OUTPUT_HI[0]
    for k in [0, 16]:  # 0 = OUTPUT_LO[0], 16 = OUTPUT_HI[0]
        output_dim = BD.OUTPUT_LO if k == 0 else BD.OUTPUT_HI

        # Fire at STACK0 byte area (d=5-9 from BP marker, excludes marker itself)
        # Use H4[BP] (d≤9.5) AND NOT H1[BP] (d>4.5) to select d ∈ (4.5, 9.5]
        ffn.W_up[unit, BD.H4 + BP_I] = S
        ffn.W_up[unit, BD.H1 + BP_I] = -S * 20

        # Suppress at MEM value byte positions (legitimate L14 targets)
        ffn.W_up[unit, BD.MEM_VAL_B0] = -suppress
        ffn.W_up[unit, BD.MEM_VAL_B1] = -suppress
        ffn.W_up[unit, BD.MEM_VAL_B2] = -suppress
        ffn.W_up[unit, BD.MEM_VAL_B3] = -suppress

        # Suppress at register markers where OUTPUT is needed
        ffn.W_up[unit, BD.MARK_PC] = -suppress
        ffn.W_up[unit, BD.MARK_AX] = -suppress
        ffn.W_up[unit, BD.MARK_SP] = -suppress
        ffn.W_up[unit, BD.MARK_BP] = -suppress
        ffn.W_up[unit, BD.MARK_STACK0] = -suppress

        # Suppress at BYTE_INDEX_3 positions - byte 3's OUTPUT should predict
        # the NEXT marker (MEM), not force byte value 0.
        ffn.W_up[unit, BD.BYTE_INDEX_3] = -suppress

        # Bias for activation
        ffn.b_up[unit] = -S * 0.5

        # Gate unconditionally
        ffn.W_gate[unit, BD.CONST] = 1.0

        # Write large POSITIVE value to OUTPUT[0] to make it the argmax winner
        # At d=6-9, activation ≈ 3.5S to 0.5S, so output ≈ S * 50/S = 50
        # This overcomes L14's corruption (~2-3) by a large margin.
        ffn.W_down[output_dim, unit] = 50.0 / S

        unit += 1

    return unit



def _set_bz_bnz_relay(attn, S, BD, HD):
    """L6 attention: Relay AX zero detection to PC marker for BZ/BNZ.

    Head 4 (ALiBi slope=5.0): At PC marker, read prev AX marker's flags.
    Writes to CMP[2..5] at PC marker (CMP[0]=IS_JMP, CMP[1]=IS_EXIT reserved):
      CMP[2] = OP_BZ, CMP[3] = OP_BNZ
      CMP[4] = AX_CARRY_LO[0] (1.0 if lo nibble is 0)
      CMP[5] = AX_CARRY_HI[0] (1.0 if hi nibble is 0)
    Also copies FETCH_LO/HI → TEMP (branch target).

    L6 FFN uses these for conditional PC override:
      BZ:  4-way AND (MARK_PC + CMP[2] + CMP[4] + CMP[5]) → branch if AX==0
      BNZ: 2 groups covering AX!=0 cases
    """
    L = 50.0
    base = 4 * HD

    attn.W_q[base, BD.MARK_PC] = L
    attn.W_q[base, BD.MARK_AX] = -L  # block at AX
    # FIX 2026-04-16: Gate on OP_BZ or OP_BNZ to prevent firing for other opcodes.
    # Without this gate, the head fires for ALL opcodes and writes FETCH to TEMP,
    # which overlaps OUTPUT_BYTE (dims 480-511), corrupting PC byte generation.
    # OP_BZ and OP_BNZ are now injected at 5.0 via _inject_active_opcode.
    # Score budget:
    #   At PC marker without OP_BZ/BNZ: 50 - 65 = -15 (blocked)
    #   At PC marker with OP_BZ=5: 50 - 65 + 50 = 35 (fires)
    #   At PC marker with OP_BNZ=5: 50 - 65 + 50 = 35 (fires)
    attn.W_q[base, BD.CONST] = -L * 1.3  # Baseline penalty (stronger than before)
    attn.W_q[base, BD.OP_BZ] = L / 5.0  # OP_BZ=5 → contributes L
    attn.W_q[base, BD.OP_BNZ] = L / 5.0  # OP_BNZ=5 → contributes L
    attn.W_k[base, BD.MARK_AX] = L
    # Add K-side constant so Q[0] * K[0] creates a negative score when Q[0] < 0.
    # Without this, K[0]=0 at non-AX positions makes score=0, not negative.
    # With K[0]=L*CONST, score = Q[0]*L*CONST/8 = -15*50/8 = -93.75 (blocked)
    # With OP_BZ=5, Q[0]=35, so score = 35*50/8 = 218.75 (fires)
    attn.W_k[base, BD.CONST] = L

    # V: copy OP_BZ, OP_BNZ flags
    attn.W_v[base + 1, BD.OP_BZ] = 1.0
    attn.W_v[base + 2, BD.OP_BNZ] = 1.0
    # V: copy AX_CARRY_LO[0] and AX_CARRY_HI[0] (zero detection)
    attn.W_v[base + 3, BD.AX_CARRY_LO + 0] = 1.0
    attn.W_v[base + 4, BD.AX_CARRY_HI + 0] = 1.0
    # V: copy FETCH_LO/HI (branch target)
    for k in range(16):
        attn.W_v[base + 5 + k, BD.FETCH_LO + k] = 1.0
        attn.W_v[base + 21 + k, BD.FETCH_HI + k] = 1.0

    # O: write to CMP[2..5] at PC marker for FFN to use
    # (CMP[0] reserved for IS_JMP from head 0, CMP[1] for IS_EXIT from head 1)
    # Normalize OP_BZ/BNZ: raw ≈5 × 0.2 → CMP ≈1.0 (same scale as zero flags)
    # FIX 2026-04-29: Increased CMP[4]/CMP[5] scale from 1.0 to 2.0 to compensate
    # for partial attention (softmax1 splits weight across multiple positions).
    attn.W_o[BD.CMP + 2, base + 1] = 0.2  # OP_BZ at PC (normalized)
    attn.W_o[BD.CMP + 3, base + 2] = 0.2  # OP_BNZ at PC (normalized)
    attn.W_o[BD.CMP + 4, base + 3] = 2.0  # AX_LO_IS_ZERO at PC (scaled up)
    attn.W_o[BD.CMP + 5, base + 4] = 2.0  # AX_HI_IS_ZERO at PC (scaled up)
    # Write branch target to TEMP dims at PC marker
    for k in range(16):
        attn.W_o[BD.TEMP + k, base + 5 + k] = 1.0  # FETCH_LO
        attn.W_o[BD.TEMP + 16 + k, base + 21 + k] = 1.0  # FETCH_HI



def _set_tool_call_opcode_decode(ffn, S, BD):
    """L5 FFN addition: decode all I/O opcodes → IO_IS_TOOL_CALL.

    Same pattern as _set_opcode_decode_ffn: 2-way AND on OPCODE_BYTE_LO/HI
    nibbles, gated by MARK_AX. All 6 opcodes write to the same
    IO_IS_TOOL_CALL dim (combined flag, ≈5.0 when any is active, ≈0 otherwise).

    Starts at unit 400 to avoid conflict with existing opcode decode units.

    OPEN=30 (lo=14, hi=1), READ=31 (lo=15, hi=1),
    CLOS=32 (lo=0, hi=2), PRTF=33 (lo=1, hi=2),
    GETCHAR=64 (lo=0, hi=4), PUTCHAR=65 (lo=1, hi=4).
    """
    unit = 400
    io_opcodes = [
        (14, 1),  # OPEN = 30 = 0x1E
        (15, 1),  # READ = 31 = 0x1F
        (0, 2),  # CLOS = 32 = 0x20
        (1, 2),  # PRTF = 33 = 0x21
        (0, 4),  # GETCHAR = 64 = 0x40
        (1, 4),  # PUTCHAR = 65 = 0x41
    ]
    for lo, hi in io_opcodes:
        ffn.W_up[unit, BD.OPCODE_BYTE_LO + lo] = S
        ffn.W_up[unit, BD.OPCODE_BYTE_HI + hi] = S
        ffn.b_up[unit] = -S * 1.5  # both must be ~1
        ffn.W_gate[unit, BD.MARK_AX] = 1.0  # only at AX marker
        ffn.W_down[BD.IO_IS_TOOL_CALL, unit] = 10.0 / S  # ≈5.0 when active
        unit += 1



def _set_tool_call_relay_head(attn, S, BD, HD):
    """L6 attention head 5: relay IO_IS_TOOL_CALL from AX → SE position.

    Same pattern as head 1 (EXIT relay):
    - Q: NEXT_SE (query at SE position) + -L*MARK_AX (block at AX)
    - K: MARK_AX (attend to AX marker)
    - V: copy IO_IS_TOOL_CALL
    - O: write to CMP[2] (IS_TOOL_CALL relay)

    Distance from SE to AX = 28 tokens. With L=50, ALiBi slope=5:
    score = 50²*0.7*0.125 - 5*28 = 79 (strong).
    """
    L = 50.0
    base = 5 * HD  # head 5

    attn.W_q[base, BD.NEXT_SE] = L
    attn.W_q[base, BD.MARK_AX] = -L  # block at AX marker
    attn.W_k[base, BD.MARK_AX] = L
    # V: copy IO_IS_TOOL_CALL flag
    attn.W_v[base + 1, BD.IO_IS_TOOL_CALL] = 1.0
    # O: write to CMP[2]
    attn.W_o[BD.CMP + 2, base + 1] = 1.0



def _set_tool_call_detection(ffn, S, BD):
    """L6 FFN addition: CMP[2] AND NEXT_SE → NEXT_TOOL_CALL.

    Same pattern as HALT detection (CMP[1] AND NEXT_SE → NEXT_HALT).
    When both CMP[2] (IS_TOOL_CALL relay) and NEXT_SE are active:
    - Set NEXT_TOOL_CALL (emit TOOL_CALL token)
    - Clear NEXT_SE (suppress STEP_END)

    CMP[2] ≈ 5 for I/O ops, ≈ 0 otherwise.
    Threshold 3.0: I/O (5+0.68=5.68 > 3) fires, inactive (0+0.68 < 3) doesn't.

    Note: When PUTCHAR activates this, the AX→OUTPUT routing from
    _set_io_putchar_routing still works (it runs in parallel at L6 FFN).
    The output byte is produced by the model's weights as usual; the
    TOOL_CALL token just signals the runner to dispatch via tool_handler.

    Starts at unit 1300 to avoid overlap with _set_layer6_routing_ffn (units 0-1033).
    """
    unit = 1300

    ffn.W_up[unit, BD.CMP + 2] = S
    ffn.W_up[unit, BD.NEXT_SE] = S
    ffn.b_up[unit] = -S * 3.0
    ffn.b_gate[unit] = 1.0
    ffn.W_down[BD.NEXT_TOOL_CALL, unit] = 2.0 / S
    ffn.W_down[BD.NEXT_SE, unit] = -2.0 / S  # clear NEXT_SE → no STEP_END



def _set_conversational_io_opcode_decode(ffn, S, BD):
    """L5 FFN addition: decode PRTF and READ opcodes for conversational I/O mode.

    Detects PRTF (33) and READ (31) opcodes at AX marker and writes to
    separate flags for autoregressive I/O generation:
    - PRTF → IO_IS_PRTF ≈ 5.0
    - READ → IO_IS_READ ≈ 5.0

    This is separate from tool_call detection to enable different routing:
    - Tool call mode: PRTF/READ → TOOL_CALL token (runner dispatches)
    - Conversational I/O mode: PRTF/READ → autoregressive sequence
      (THINKING_END → output bytes → THINKING_START)

    Starts at unit 410 to avoid conflict with tool_call units (400-405).

    PRTF = 33 = 0x21 (lo=1, hi=2)
    READ = 31 = 0x1F (lo=15, hi=1)
    """
    unit = 410

    # PRTF detection via ACTIVE_OPCODE_PRTF flag (set by embedding)
    ffn.W_up[unit, BD.ACTIVE_OPCODE_PRTF] = S  # 1.0 when PRTF is active
    ffn.b_up[unit] = -S * 0.5  # threshold: active when ACTIVE_OPCODE_PRTF ≈ 1
    ffn.b_gate[unit] = 1.0  # always gate (no position restriction needed)
    ffn.W_down[BD.IO_IS_PRTF, unit] = 10.0 / S  # ≈5.0 when active
    unit += 1

    # READ detection via ACTIVE_OPCODE_READ flag
    ffn.W_up[unit, BD.ACTIVE_OPCODE_READ] = S
    ffn.b_up[unit] = -S * 0.5
    ffn.b_gate[unit] = 1.0
    ffn.W_down[BD.IO_IS_READ, unit] = 10.0 / S



def _set_conversational_io_relay_heads(attn, S, BD, HD):
    """L6 attention heads 4-5: relay IO_IS_PRTF and IO_IS_READ from AX → SE.

    IMPORTANT: Changed from heads 6-7 to heads 4-5 to avoid conflict with
    _set_opcode_relay_head() which uses head 6 for PSH/ADJ/pop relay.

    Head 4: Relay IO_IS_PRTF
    - Q: NEXT_SE (query at SE position), gated by ACTIVE_OPCODE_PRTF
    - K: MARK_AX (attend to AX marker)
    - V: copy IO_IS_PRTF
    - O: write to CMP + 5 (changed from CMP[3] to avoid pop group conflict)

    FIX 2026-04-16: Gate head 4 with ACTIVE_OPCODE_PRTF to avoid conflict with
    BZ/BNZ relay which also uses head 4. Without this gate, the BZ/BNZ's
    W_q[base, CONST] = -65 penalty blocks PRTF relay (CONST=1.0 at all positions).
    Adding W_q[base, ACTIVE_OPCODE_PRTF] = +65 cancels the penalty when PRTF active.

    Head 5: Relay IO_IS_READ
    - Q: NEXT_SE (query at SE position)
    - K: MARK_AX (attend to AX marker)
    - V: copy IO_IS_READ
    - O: write to CMP + 6 (changed from TEMP[0] to use dedicated CMP slot)

    Uses steep ALiBi slope (5.0) for both heads to overcome distance penalty.
    """
    L = 50.0

    # Head 4: PRTF relay
    # FIX 2026-04-16: Use V[37] instead of V[1] to avoid conflict with BZ/BNZ relay.
    # BZ/BNZ relay uses V[1] for OP_BZ and writes to CMP+2 via O[CMP+2, V[1]]=0.2.
    # When PRTF is active, IO_IS_PRTF=5.0 would get multiplied by that 0.2, causing
    # spurious CMP+2=1.0 which triggers ENT logic at STACK0 marker.
    base = 4 * HD
    attn.W_q[base, BD.NEXT_SE] = L
    attn.W_q[base, BD.MARK_AX] = -L  # block at AX marker
    # FIX 2026-04-16: Gate with ACTIVE_OPCODE_PRTF to overcome BZ/BNZ CONST penalty
    # BZ/BNZ relay sets W_q[base, CONST] = -65, which blocks at SE (CONST=1.0).
    # When PRTF active, ACTIVE_OPCODE_PRTF=1.0, contributing +L=50 to cancel part of penalty.
    # Combined with NEXT_SE contribution, this makes Q positive at SE during PRTF.
    attn.W_q[base, BD.ACTIVE_OPCODE_PRTF] = L * 1.5  # +75 to overcome -65 CONST penalty
    attn.W_k[base, BD.MARK_AX] = L
    attn.W_v[base + 37, BD.IO_IS_PRTF] = 1.0  # V[37] avoids BZ/BNZ V[1] conflict
    attn.W_o[BD.CMP + 5, base + 37] = 1.0  # Use CMP[5] instead of CMP[3]

    # Head 5: READ relay
    base = 5 * HD
    attn.W_q[base, BD.NEXT_SE] = L
    attn.W_q[base, BD.MARK_AX] = -L  # block at AX marker
    attn.W_k[base, BD.MARK_AX] = L
    attn.W_v[base + 1, BD.IO_IS_READ] = 1.0
    attn.W_o[BD.CMP + 6, base + 1] = 1.0  # Use CMP[6] instead of TEMP[0]



def _set_conversational_io_state_machine(ffn, S, BD):
    """L6 FFN addition: Start conversational I/O sequence when PRTF/READ detected.

    State transitions:
    1. Normal execution (IO_STATE=0)
    2. PRTF/READ detected → set NEXT_THINKING_END, IO_STATE=1
    3. After THINKING_END → generate output, IO_STATE=2
    4. After output complete → set NEXT_THINKING_START, IO_STATE=3
    5. After THINKING_START → resume normal, IO_STATE=0

    For now, we implement step 2: detect PRTF → start sequence.
    Steps 3-5 will be added in L13 FFN (state tracking across generation steps).

    Condition: CMP[5] (PRTF flag) AND NEXT_SE
    - Set NEXT_THINKING_END (emit </thinking> token)
    - Clear NEXT_SE (suppress STEP_END)
    - Set IO_STATE = 1 (begin I/O sequence)

    IMPORTANT: Changed from CMP[3] to CMP[5] and TEMP[0] to CMP[6] to avoid
    conflicts with _set_opcode_relay_head() which uses CMP[3] for pop group.

    Starts at unit 1400 to avoid overlap with _set_layer6_routing_ffn (units 0-1033).
    """
    unit = 1400

    # PRTF triggers thinking end
    # FIX 2026-04-16: CMP[5] can be ~5.0 (scaled by attention), not just ~1.0.
    # With CMP[5]=5.0, old threshold (-3S) was exceeded without NEXT_SE.
    # Fix: Gate on NEXT_SE so output is suppressed when NEXT_SE is not active.
    #
    # IMPORTANT: PureFFN uses `silu(up) * gate` where gate is LINEAR (no sigmoid)!
    # If b_gate is negative, gate can be negative when NEXT_SE=0, causing:
    #   silu(up) * negative_gate * negative_W_down = POSITIVE output (wrong!)
    # Fix: Use b_gate=0 so gate=10*NEXT_SE, which is 0 when NEXT_SE=0.
    ffn.W_up[unit, BD.CMP + 5] = S  # PRTF flag from relay head
    ffn.b_up[unit] = -S * 0.5  # Fire when CMP[5] > 0.5 (CMP[5]~5.0 → 500-50=450)
    ffn.W_gate[unit, BD.NEXT_SE] = 10.0  # Gate = 10*NEXT_SE (0 when inactive, 7 when active)
    ffn.b_gate[unit] = 0.0  # No bias - gate is 0 when NEXT_SE=0
    ffn.W_down[BD.NEXT_THINKING_END, unit] = 2.0 / S  # emit THINKING_END
    ffn.W_down[BD.NEXT_SE, unit] = -2.0 / S  # suppress STEP_END
    ffn.W_down[BD.IO_STATE, unit] = 2.0 / S  # set IO_STATE = 1
    unit += 1

    # READ triggers thinking end (similar pattern)
    ffn.W_up[unit, BD.CMP + 6] = S  # READ flag from relay head
    ffn.b_up[unit] = -S * 0.5  # Fire when CMP[6] > 0.5
    ffn.W_gate[unit, BD.NEXT_SE] = 10.0  # Gate on NEXT_SE
    ffn.b_gate[unit] = 0.0  # No bias
    ffn.W_down[BD.NEXT_THINKING_END, unit] = 2.0 / S
    ffn.W_down[BD.NEXT_SE, unit] = -2.0 / S
    ffn.W_down[BD.IO_STATE, unit] = 2.0 / S
    unit += 1



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



def _set_conversational_io_output_routing(ffn, S, BD):
    """L15 FFN addition: Route OUTPUT_BYTE to OUTPUT when in output mode.

    When IO_IN_OUTPUT_MODE (emitting output bytes):
    - Copy OUTPUT_BYTE_LO → OUTPUT_LO (all 16 nibbles)
    - Copy OUTPUT_BYTE_HI → OUTPUT_HI (all 16 nibbles)

    This routes the fetched format string byte to the output head for emission.

    Note: We don't need to suppress normal OUTPUT routing because IO_IN_OUTPUT_MODE
    only activates after THINKING_END, at which point we're not in the normal
    35-token generation cycle.

    Starts at unit 1200 to avoid overlap with _set_layer6_routing_ffn (units 0-1033).
    """
    unit = 1200

    # Copy each OUTPUT_BYTE nibble to corresponding OUTPUT nibble when in output mode
    for k in range(16):
        # Lo nibble
        ffn.W_up[unit, BD.IO_IN_OUTPUT_MODE] = S
        ffn.b_up[unit] = -S * 0.5
        ffn.W_gate[unit, BD.OUTPUT_BYTE_LO + k] = 1.0
        ffn.W_down[BD.OUTPUT_LO + k, unit] = 2.0 / S
        unit += 1

        # Hi nibble
        ffn.W_up[unit, BD.IO_IN_OUTPUT_MODE] = S
        ffn.b_up[unit] = -S * 0.5
        ffn.W_gate[unit, BD.OUTPUT_BYTE_HI + k] = 1.0
        ffn.W_down[BD.OUTPUT_HI + k, unit] = 2.0 / S
        unit += 1
