"""Test multi-instruction weight baking for AutoregressiveVM.

Supports: IMM, EXIT with proper PC carry-forward, instruction fetch, and AX carry-forward.
Tests: "IMM 42; EXIT" (exit_code=42), "IMM 5; IMM 3; EXIT" (exit_code=3).

NOTE: This file defines its OWN dimension layout (V2Dim) independent of vm_step._BakeDim,
because _BakeDim was rewritten for d_model=512 / 39-token steps. bake_v2 uses d_model=256
with 30-token steps (no STACK0).
"""
import torch
from neural_vm.vm_step import Token, AutoregressiveVM, AutoregressiveAttention
from neural_vm.kv_cache_eviction import softmax1
from neural_vm.embedding import Opcode
from neural_vm.base_layers import PureFFN

# =============================================================================
# Self-contained dimension layout for bake_v2 (d_model=256, 30-token step)
# =============================================================================
# These mirror the OLD _BakeDim layout. They must NOT import from the current
# _BakeDim which requires d_model=512 and 39-token steps.

class V2Dim:
    """Dimension allocation for bake_v2. All dims < 256 for d_model=256."""
    # --- Marker identity flags (set by embedding) ---
    MARK_PC = 0; MARK_AX = 1; MARK_SP = 2; MARK_BP = 3
    MARK_MEM = 4; MARK_SE = 5; IS_BYTE = 6; IS_MARK = 7
    CONST = 8; MARK_CS = 9; MARK_SE_ONLY = 10

    MARKS = [MARK_PC, MARK_AX, MARK_SP, MARK_BP, MARK_MEM, MARK_SE, MARK_CS]
    NUM_MARKERS = 7

    # --- Transition flags (custom positions for d_model=256) ---
    NEXT_PC = 12; NEXT_AX = 13; NEXT_SP = 14; NEXT_BP = 15
    NEXT_MEM = 16; NEXT_SE = 17; NEXT_HALT = 18

    # --- L0 threshold heads (4 heads, thresholds [3.5, 4.5, 7.5, 8.5]) ---
    # Each head outputs 7 marker-type values (NUM_MARKERS)
    H0 = 60; H1 = 67; H2 = 74; H3 = 81

    # --- L1 fine thresholds + HAS_SE ---
    L1H0 = 116; L1H1 = 123; L1H2 = 130
    HAS_SE = 137

    # --- Byte index within register (0-3) ---
    BYTE_INDEX_0 = 138; BYTE_INDEX_1 = 139
    BYTE_INDEX_2 = 140; BYTE_INDEX_3 = 141

    # --- Nibble encoding ---
    EMBED_LO = 142;  EMBED_HI = 158
    OUTPUT_LO = 174; OUTPUT_HI = 190


BD = V2Dim  # Alias used throughout bake_v2

# --- Extended dims for multi-instruction support ---
# Layer 2 CS-only threshold outputs (single dim each, CS component only)
L2H0_CS = 204  # threshold 5.5
L2H1_CS = 205  # threshold 6.5
L2H2_CS = 206  # threshold 10.5
L2H3_CS = 207  # threshold 11.5
L3H0_CS = 208  # threshold 12.5

# Bytecode position flags (set by Layer 2/3 FFN)
FLAG_0_IMM = 112  # d=2 from CS
FLAG_0_OP = 209   # d=1 from CS
FLAG_1_IMM = 210  # d=7 from CS
FLAG_1_OP = 211   # d=6 from CS
FLAG_2_IMM = 212  # d=12 from CS
FLAG_2_OP = 213   # d=11 from CS

# Instruction selection (at AX marker, computed from PC value)
IS_FETCH_0 = 214; IS_FETCH_1 = 215; IS_FETCH_2 = 216

# AX carry-forward staging (at AX marker, from attention)
CARRY_AX_LO = 217  # 217-232 (16 dims)
CARRY_AX_HI = 233  # 233-248 (16 dims)

# Opcode detection staging
IS_EXIT_FETCH = 249  # fetched opcode is EXIT (at AX marker)
IS_IMM_FETCH = 250   # fetched opcode is IMM
IS_EXIT_AT_SE = 251  # EXIT flag relayed to NEXT_SE position (for HALT)

# CS index within the 7-marker MARKS array
CS_IDX = 6

# Dims used by threshold outputs (index into 7-marker array)
# BD.MARKS = [PC=0, AX=1, SP=2, BP=3, MEM=4, SE=5, CS=6]
PC_IDX = 0; AX_IDX = 1


def _bake_cs_threshold_attn(attn, head_idx, threshold, out_dim, slope, HD):
    """Bake a single threshold head that only detects CODE_START distance.

    Outputs a single dim (CS component only) instead of full 7-marker vector.
    """
    base = head_idx * HD
    q_val = 8.0 * slope  # sqrt(HD) = 8
    attn.W_q[base, BD.CONST] = q_val
    attn.W_k[base, BD.IS_MARK] = threshold
    # V only copies MARK_CS
    attn.W_v[base + 1, BD.MARK_CS] = 1.0
    # O writes to single output dim
    attn.W_o[out_dim, base + 1] = 1.0


def _bake_carry_forward_attn(attn, head_idx, marker_dim, l1h1_idx, l1h0_idx,
                              HD, slope=0.5):
    """Bake attention head for register carry-forward.

    At marker positions (e.g., REG_PC), attends to the previous step's
    corresponding byte 0 (identified by L1H1_marker - L1H0_marker = 1).
    Copies EMBED_LO/HI from the target byte token.

    Args:
        marker_dim: dim set at the marker token (e.g., BD.MARK_PC)
        l1h1_idx: index of marker type in L1H1 threshold output
        l1h0_idx: index of marker type in L1H0 threshold output
    """
    base = head_idx * HD
    L = 15.0  # Q*K scale

    # Q: active at marker positions
    attn.W_q[base, marker_dim] = L

    # K: active at "byte 0 of this register" = L1H1[marker] - L1H0[marker] = 1
    # L1H1 + marker_idx is the dim for this marker type in L1H1 output
    attn.W_k[base, BD.L1H1 + l1h1_idx] = L
    attn.W_k[base, BD.L1H0 + l1h0_idx] = -L  # exclude marker itself (d=0)

    # V copies EMBED_LO/HI (nibble encoding of byte value)
    for k in range(16):
        attn.W_v[base + 1 + k, BD.EMBED_LO + k] = 1.0
        attn.W_v[base + 17 + k, BD.EMBED_HI + k] = 1.0

    # O writes to EMBED_LO/HI (overwriting marker's own, which is 0)
    for k in range(16):
        attn.W_o[BD.EMBED_LO + k, base + 1 + k] = 1.0
        attn.W_o[BD.EMBED_HI + k, base + 17 + k] = 1.0


def _bake_carry_forward_to_staging(attn, head_idx, marker_dim, l1h1_idx,
                                    l1h0_idx, out_lo, out_hi, HD, slope=0.5):
    """Like carry-forward but writes to staging dims instead of EMBED."""
    base = head_idx * HD
    L = 15.0

    attn.W_q[base, marker_dim] = L
    attn.W_k[base, BD.L1H1 + l1h1_idx] = L
    attn.W_k[base, BD.L1H0 + l1h0_idx] = -L

    for k in range(16):
        attn.W_v[base + 1 + k, BD.EMBED_LO + k] = 1.0
        attn.W_v[base + 17 + k, BD.EMBED_HI + k] = 1.0

    for k in range(16):
        attn.W_o[out_lo + k, base + 1 + k] = 1.0
        attn.W_o[out_hi + k, base + 17 + k] = 1.0


def _bake_phase_a_ffn_30tok(ffn, S):
    """Step structure FFN for 30-token step (no STACK0).

    30-token step: PC(5)+AX(5)+SP(5)+BP(5)+MEM(9)+SE(1)
    Detects marker transitions using threshold head outputs.
    Thresholds: H0=3.5, H1=4.5, H2=7.5, H3=8.5
    """
    PC_I, AX_I, SP_I, BP_I, MEM_I, SE_I = 0, 1, 2, 3, 4, 5

    transitions = [
        # SE → PC: SE within 3.5 (H0[SE]) → emit REG_PC next
        (BD.H0 + SE_I, None, BD.NEXT_PC),
        # PC → AX: PC within 4.5 (H1) but NOT 3.5 (H0) → d=4 from PC = byte 3
        (BD.H1 + PC_I, BD.H0 + PC_I, BD.NEXT_AX),
        # AX → SP
        (BD.H1 + AX_I, BD.H0 + AX_I, BD.NEXT_SP),
        # SP → BP
        (BD.H1 + SP_I, BD.H0 + SP_I, BD.NEXT_BP),
        # BP → MEM: same pattern (d=4 from BP = byte 3)
        (BD.H1 + BP_I, BD.H0 + BP_I, BD.NEXT_MEM),
        # MEM → SE: MEM within 8.5 (H3) but NOT 7.5 (H2) → d=8 from MEM = last val byte
        (BD.H3 + MEM_I, BD.H2 + MEM_I, BD.NEXT_SE),
    ]
    for i, (up_dim, gate_dim, out_dim) in enumerate(transitions):
        ffn.W_up[i, up_dim] = S
        ffn.b_up[i] = -S * 0.3
        if gate_dim is not None:
            ffn.W_gate[i, gate_dim] = -1.0
            ffn.b_gate[i] = 1.0
        else:
            ffn.b_gate[i] = 1.0
        ffn.W_down[out_dim, i] = 1.0 / S


def bake_v2(model):
    """Bake weights for multi-instruction VM execution.

    Architecture (8 layers):
      L0: Step structure (threshold attention at [3.5,4.5,7.5,8.5])
      L1: Fine thresholds [0.5,1.5,2.5] + HAS_SE detection
      L2: Bytecode thresholds [5.5,6.5,10.5,11.5] (CS-only)
      L3: Threshold [12.5] + PC carry-forward
      L4: AX carry-forward + PC relay to AX marker
      L5: Instruction fetch (imm + opcode)
      L6: Opcode dispatch (AX source selection + HALT)
      L7: reserved

    Supports: IMM, EXIT. Up to 3 instructions.
    """
    assert len(model.blocks) >= 8, f"Need >= 8 layers, got {len(model.blocks)}"

    d = model.d_model
    V = model.vocab_size
    S = 100.0
    HD = d // model.blocks[0].attn.num_heads  # 64
    ALIBI_S = 10.0

    with torch.no_grad():
        # Zero all weights
        for p in model.parameters():
            p.zero_()

        embed = model.embed.weight

        # ===== EMBEDDING =====
        for tok in range(V):
            embed[tok, BD.CONST] = 1.0

        for tok, dim in [(Token.REG_PC, BD.MARK_PC), (Token.REG_AX, BD.MARK_AX),
                         (Token.REG_SP, BD.MARK_SP), (Token.REG_BP, BD.MARK_BP),
                         (Token.MEM, BD.MARK_MEM), (Token.CODE_START, BD.MARK_CS)]:
            embed[tok, dim] = 1.0
            embed[tok, BD.IS_MARK] = 1.0

        for tok in [Token.STEP_END, Token.DATA_END, Token.HALT]:
            embed[tok, BD.MARK_SE] = 1.0
            embed[tok, BD.IS_MARK] = 1.0

        embed[Token.STEP_END, BD.MARK_SE_ONLY] = 1.0

        for b in range(256):
            embed[b, BD.IS_BYTE] = 1.0
            embed[b, BD.EMBED_LO + (b & 0xF)] = 1.0
            embed[b, BD.EMBED_HI + ((b >> 4) & 0xF)] = 1.0

        # ===== LAYER 0: Step structure =====
        attn0 = model.blocks[0].attn
        attn0.alibi_slopes.fill_(ALIBI_S)
        from neural_vm.vm_step import _set_threshold_attn
        _set_threshold_attn(attn0, [3.5, 4.5, 7.5, 8.5],
                            [BD.H0, BD.H1, BD.H2, BD.H3], ALIBI_S, HD)
        _bake_phase_a_ffn_30tok(model.blocks[0].ffn, S)

        # ===== LAYER 1: Fine thresholds + HAS_SE =====
        attn1 = model.blocks[1].attn
        attn1.alibi_slopes.fill_(ALIBI_S)
        attn1.alibi_slopes[3] = 0.0  # global for SE detection
        _set_threshold_attn(attn1, [0.5, 1.5, 2.5],
                            [BD.L1H0, BD.L1H1, BD.L1H2], ALIBI_S, HD,
                            heads=[0, 1, 2])
        # Head 3: HAS_SE (global)
        base = 3 * HD
        attn1.W_q[base, BD.CONST] = 10.0
        attn1.W_k[base, BD.MARK_SE_ONLY] = 10.0
        attn1.W_v[base + 1, BD.MARK_SE_ONLY] = 1.0
        attn1.W_o[BD.HAS_SE, base + 1] = 1.0

        # Layer 1 FFN: compute FLAG_0_IMM (d=2 from CS) and FLAG_0_OP (d=1 from CS)
        ffn1 = model.blocks[1].ffn
        unit = 0

        # FLAG_0_IMM (d=2): IS_BYTE AND L1H2_CS AND NOT L1H1_CS
        ffn1.W_up[unit, BD.IS_BYTE] = S
        ffn1.b_up[unit] = -S * 0.5
        ffn1.W_gate[unit, BD.L1H2 + CS_IDX] = 1.0
        ffn1.W_gate[unit, BD.L1H1 + CS_IDX] = -1.0
        ffn1.W_down[FLAG_0_IMM, unit] = 2.0 / S
        unit += 1

        # FLAG_0_OP (d=1): IS_BYTE AND L1H1_CS AND NOT L1H0_CS
        ffn1.W_up[unit, BD.IS_BYTE] = S
        ffn1.b_up[unit] = -S * 0.5
        ffn1.W_gate[unit, BD.L1H1 + CS_IDX] = 1.0
        ffn1.W_gate[unit, BD.L1H0 + CS_IDX] = -1.0
        ffn1.W_down[FLAG_0_OP, unit] = 2.0 / S
        unit += 1

        # ===== LAYER 2: Bytecode thresholds [5.5, 6.5, 10.5, 11.5] =====
        attn2 = model.blocks[2].attn
        attn2.alibi_slopes.fill_(ALIBI_S)
        _bake_cs_threshold_attn(attn2, 0, 5.5, L2H0_CS, ALIBI_S, HD)
        _bake_cs_threshold_attn(attn2, 1, 6.5, L2H1_CS, ALIBI_S, HD)
        _bake_cs_threshold_attn(attn2, 2, 10.5, L2H2_CS, ALIBI_S, HD)
        _bake_cs_threshold_attn(attn2, 3, 11.5, L2H3_CS, ALIBI_S, HD)

        # Layer 2 FFN: compute FLAG_1_OP, FLAG_1_IMM, FLAG_2_OP, FLAG_2_IMM
        ffn2 = model.blocks[2].ffn
        unit = 0
        H2_CS = BD.H2 + CS_IDX  # Layer 0 threshold 7.5, CS component

        # FLAG_1_OP (d=6): IS_BYTE AND L2H1_CS(6.5) AND NOT L2H0_CS(5.5)
        ffn2.W_up[unit, BD.IS_BYTE] = S
        ffn2.b_up[unit] = -S * 0.5
        ffn2.W_gate[unit, L2H1_CS] = 1.0
        ffn2.W_gate[unit, L2H0_CS] = -1.0
        ffn2.W_down[FLAG_1_OP, unit] = 2.0 / S
        unit += 1

        # FLAG_1_IMM (d=7): IS_BYTE AND H2_CS(7.5) AND NOT L2H1_CS(6.5)
        ffn2.W_up[unit, BD.IS_BYTE] = S
        ffn2.b_up[unit] = -S * 0.5
        ffn2.W_gate[unit, H2_CS] = 1.0
        ffn2.W_gate[unit, L2H1_CS] = -1.0
        ffn2.W_down[FLAG_1_IMM, unit] = 2.0 / S
        unit += 1

        # FLAG_2_OP (d=11): IS_BYTE AND L2H3_CS(11.5) AND NOT L2H2_CS(10.5)
        ffn2.W_up[unit, BD.IS_BYTE] = S
        ffn2.b_up[unit] = -S * 0.5
        ffn2.W_gate[unit, L2H3_CS] = 1.0
        ffn2.W_gate[unit, L2H2_CS] = -1.0
        ffn2.W_down[FLAG_2_OP, unit] = 2.0 / S
        unit += 1

        # FLAG_2_IMM (d=12): IS_BYTE AND L3H0_CS(12.5) AND NOT L2H3_CS(11.5)
        # L3H0_CS is set by Layer 3 attention, so we compute FLAG_2_IMM in Layer 3 FFN
        # For now, skip - we'll handle it in Layer 3.

        # ===== LAYER 3: Threshold 12.5 + PC carry-forward =====
        attn3 = model.blocks[3].attn
        attn3.alibi_slopes.fill_(0.5)  # moderate slope for carry-forward
        # Head 0: threshold 12.5 (CS only)
        _bake_cs_threshold_attn(attn3, 0, 12.5, L3H0_CS, ALIBI_S, HD)
        attn3.alibi_slopes[0] = ALIBI_S  # strong slope for threshold detection

        # Head 1: PC carry-forward (REG_PC marker reads prev step's PC byte 0)
        _bake_carry_forward_attn(attn3, 1, BD.MARK_PC, PC_IDX, PC_IDX, HD)

        # Layer 3 FFN: FLAG_2_IMM + PC first-step default + PC increment
        ffn3 = model.blocks[3].ffn
        unit = 0

        # FLAG_2_IMM (d=12): IS_BYTE AND L3H0_CS(12.5) AND NOT L2H3_CS(11.5)
        ffn3.W_up[unit, BD.IS_BYTE] = S
        ffn3.b_up[unit] = -S * 0.5
        ffn3.W_gate[unit, L3H0_CS] = 1.0
        ffn3.W_gate[unit, L2H3_CS] = -1.0
        ffn3.W_down[FLAG_2_IMM, unit] = 2.0 / S
        unit += 1

        # PC FIRST-STEP DEFAULT: when MARK_PC AND NOT HAS_SE, set OUTPUT for byte 2
        # Uses two units per dim: always-set + conditional-undo (avoids negative gate leak)
        # Unit A: always set OUTPUT_LO[2]=1 at PC markers
        ffn3.W_up[unit, BD.MARK_PC] = S
        ffn3.b_up[unit] = -S * 0.5
        ffn3.b_gate[unit] = 1.0
        ffn3.W_down[BD.OUTPUT_LO + 2, unit] = 2.0 / S
        unit += 1
        # Unit B: undo when HAS_SE (subsequent steps use carry-forward + increment)
        ffn3.W_up[unit, BD.HAS_SE] = S
        ffn3.b_up[unit] = -S * 0.5
        ffn3.W_gate[unit, BD.MARK_PC] = 1.0  # only at PC markers
        ffn3.W_down[BD.OUTPUT_LO + 2, unit] = -2.0 / S
        unit += 1

        # Same for OUTPUT_HI[0]
        ffn3.W_up[unit, BD.MARK_PC] = S
        ffn3.b_up[unit] = -S * 0.5
        ffn3.b_gate[unit] = 1.0
        ffn3.W_down[BD.OUTPUT_HI + 0, unit] = 2.0 / S
        unit += 1
        ffn3.W_up[unit, BD.HAS_SE] = S
        ffn3.b_up[unit] = -S * 0.5
        ffn3.W_gate[unit, BD.MARK_PC] = 1.0
        ffn3.W_down[BD.OUTPUT_HI + 0, unit] = -2.0 / S
        unit += 1

        # PC INCREMENT: when MARK_PC AND HAS_SE, add 5 to carried-forward value
        # The carry-forward wrote prev PC byte's EMBED_LO/HI to current position.
        # We need to rotate lo nibble by +5 and handle carry to hi nibble.
        #
        # For lo nibble k (0-15), old EMBED_LO[k] = 1:
        #   new_lo = (k+5) % 16
        #   carry = 1 if k+5 >= 16 (i.e., k >= 11)
        #
        # Implementation: for each k, SwiGLU unit:
        #   up = S * (HAS_SE - 0.3)  [active when HAS_SE > 0.3]
        #   gate = EMBED_LO[k]       [active when this nibble is hot]
        #   down: +1 to OUTPUT_LO[(k+5)%16], -1 from OUTPUT_LO[k] (cancel residual)
        #         if k >= 11: +1 to CARRY_OUT dim (we'll handle hi carry later)

        # Actually, the EMBED_LO at this position came from carry-forward attention.
        # It's the OLD PC byte's lo nibble. We need to shift it by 5 and write to OUTPUT.
        # But OUTPUT_LO might have residual from EMBED_LO (no, OUTPUT dims are separate).
        # So just write to OUTPUT_LO[(k+5)%16].

        # We also need to copy the hi nibble (for the non-carry case).
        # And handle the carry case (k >= 11) by incrementing hi nibble.

        # Let's use a simpler approach: 16 units for lo nibble shift, 1 for carry detect,
        # 16 for hi nibble copy, 1 for hi nibble carry.

        # Lo nibble: shift by 5 (only at MARK_PC AND HAS_SE)
        for k in range(16):
            new_k = (k + 5) % 16
            ffn3.W_up[unit, BD.HAS_SE] = S
            ffn3.W_up[unit, BD.MARK_PC] = S  # AND with MARK_PC
            ffn3.b_up[unit] = -S * 1.5       # threshold: both must be ~1
            ffn3.W_gate[unit, BD.EMBED_LO + k] = 1.0
            ffn3.W_down[BD.OUTPUT_LO + new_k, unit] = 2.0 / S
            unit += 1

        # Hi nibble: copy (only at MARK_PC AND HAS_SE)
        for k in range(16):
            ffn3.W_up[unit, BD.HAS_SE] = S
            ffn3.W_up[unit, BD.MARK_PC] = S
            ffn3.b_up[unit] = -S * 1.5
            ffn3.W_gate[unit, BD.EMBED_HI + k] = 1.0
            ffn3.W_down[BD.OUTPUT_HI + k, unit] = 2.0 / S
            unit += 1

        # Hi nibble carry: when lo nibble was 11-15 (k+5 >= 16)
        # Sum EMBED_LO[11..15] to get carry signal, then shift hi nibble
        # For simplicity, use a single "carry" signal and add 1 to hi nibble.
        # This only works for single carry (hi nibble < 15), sufficient for PC < 256.
        #
        # Carry = sum(EMBED_LO[11:16]) ≈ 1 when any of 11-15 is hot
        # Then: OUTPUT_HI[(old_hi+1)%16] += carry, OUTPUT_HI[old_hi] -= carry
        #
        # But we don't know old_hi directly. We need per-value handling.
        # Simpler: for each hi nibble value h (0-15), if EMBED_HI[h]=1 AND carry:
        #   subtract from OUTPUT_HI[h], add to OUTPUT_HI[(h+1)%16]

        # First, detect carry: EMBED_LO[11] + ... + EMBED_LO[15] > 0.5
        # Then for each h, conditionally shift:
        # This needs too many units. For simplicity, limit to hi=0 carry (most common).
        # When carry AND EMBED_HI[0]=1: OUTPUT_HI[1] += 1, OUTPUT_HI[0] -= 1

        # Actually for 3 instructions with PC=2,7,12:
        # Step 1: PC=2 (lo=2, hi=0)
        # Step 2: carry-forward PC=2, add 5 → PC=7 (lo=7, hi=0, no carry since 2+5=7<16)
        # Step 3: carry-forward PC=7, add 5 → PC=12 (lo=12, hi=0, no carry since 7+5=12<16)
        # So NO CARRY happens for 3 instructions. Skip carry handling for now.

        # ===== LAYER 4: AX carry-forward + PC relay to AX marker =====
        attn4 = model.blocks[4].attn
        attn4.alibi_slopes.fill_(0.5)

        # Head 0: AX marker reads current step's PC byte 0 (5 tokens back)
        # Q: active at MARK_AX. K: "I am PC byte 0" = L1H1_PC - L1H0_PC
        # The PC byte 0 is 5 positions before AX marker in the step.
        base = 0 * HD
        L = 15.0
        attn4.W_q[base, BD.MARK_AX] = L
        attn4.W_k[base, BD.L1H1 + PC_IDX] = L
        attn4.W_k[base, BD.L1H0 + PC_IDX] = -L
        # V copies EMBED_LO/HI (PC byte value)
        for k in range(16):
            attn4.W_v[base + 1 + k, BD.EMBED_LO + k] = 1.0
        # O writes to EMBED_LO at AX marker (we'll read this in FFN)
        for k in range(16):
            attn4.W_o[BD.EMBED_LO + k, base + 1 + k] = 1.0

        # Head 1: AX carry-forward (AX marker reads prev step's AX byte 0)
        _bake_carry_forward_to_staging(attn4, 1, BD.MARK_AX, AX_IDX, AX_IDX,
                                        CARRY_AX_LO, CARRY_AX_HI, HD)

        # Layer 4 FFN: decode PC value → IS_FETCH_K flags, clear EMBED_LO
        ffn4 = model.blocks[4].ffn
        unit = 0

        # IS_FETCH_0: MARK_AX AND EMBED_LO[2] (PC byte 0 = 2 → instruction 0)
        ffn4.W_up[unit, BD.MARK_AX] = S
        ffn4.b_up[unit] = -S * 0.5
        ffn4.W_gate[unit, BD.EMBED_LO + 2] = 1.0  # PC value 2 → instr 0
        ffn4.W_down[IS_FETCH_0, unit] = 2.0 / S
        unit += 1

        # IS_FETCH_1: MARK_AX AND EMBED_LO[7] (PC byte 0 = 7 → instruction 1)
        ffn4.W_up[unit, BD.MARK_AX] = S
        ffn4.b_up[unit] = -S * 0.5
        ffn4.W_gate[unit, BD.EMBED_LO + 7] = 1.0  # PC value 7 → instr 1
        ffn4.W_down[IS_FETCH_1, unit] = 2.0 / S
        unit += 1

        # IS_FETCH_2: MARK_AX AND EMBED_LO[12] (PC byte 0 = 12 → instruction 2)
        ffn4.W_up[unit, BD.MARK_AX] = S
        ffn4.b_up[unit] = -S * 0.5
        ffn4.W_gate[unit, BD.EMBED_LO + 12] = 1.0  # PC value 12 → instr 2
        ffn4.W_down[IS_FETCH_2, unit] = 2.0 / S
        unit += 1

        # Clear EMBED_LO at AX marker (so Layer 5 can write fetched data)
        # Single unit per dim (no cancel pair — silu(-50) ≈ 0 at non-AX)
        for k in range(16):
            ffn4.W_up[unit, BD.MARK_AX] = S
            ffn4.b_up[unit] = -S * 0.5
            ffn4.W_gate[unit, BD.EMBED_LO + k] = -1.0  # negate to subtract
            ffn4.W_down[BD.EMBED_LO + k, unit] = 2.0 / S
            unit += 1

        # Clear EMBED_HI at AX marker
        for k in range(16):
            ffn4.W_up[unit, BD.MARK_AX] = S
            ffn4.b_up[unit] = -S * 0.5
            ffn4.W_gate[unit, BD.EMBED_HI + k] = -1.0
            ffn4.W_down[BD.EMBED_HI + k, unit] = 2.0 / S
            unit += 1

        # ===== LAYER 5: Instruction fetch (imm byte 0 + opcode byte) =====
        attn5 = model.blocks[5].attn
        attn5.alibi_slopes.fill_(0.1)  # gentle slope for bytecode reading

        # Head 0: Fetch imm byte 0 of selected instruction
        # Q: reads IS_FETCH_K, K: reads FLAG_K_IMM
        base = 0 * HD
        L = 20.0
        for k, (fetch_dim, flag_dim) in enumerate([
            (IS_FETCH_0, FLAG_0_IMM),
            (IS_FETCH_1, FLAG_1_IMM),
            (IS_FETCH_2, FLAG_2_IMM),
        ]):
            attn5.W_q[base + k, fetch_dim] = L
            attn5.W_k[base + k, flag_dim] = L
        # V: copies EMBED_LO/HI (nibble encoding of imm byte)
        for k in range(16):
            attn5.W_v[base + 3 + k, BD.EMBED_LO + k] = 1.0
            attn5.W_v[base + 19 + k, BD.EMBED_HI + k] = 1.0
        # O: writes to EMBED_LO/HI at AX marker
        for k in range(16):
            attn5.W_o[BD.EMBED_LO + k, base + 3 + k] = 1.0
            attn5.W_o[BD.EMBED_HI + k, base + 19 + k] = 1.0

        # Head 1: Fetch opcode byte of selected instruction
        # Q: reads IS_FETCH_K, K: reads FLAG_K_OP
        base = 1 * HD
        for k, (fetch_dim, flag_dim) in enumerate([
            (IS_FETCH_0, FLAG_0_OP),
            (IS_FETCH_1, FLAG_1_OP),
            (IS_FETCH_2, FLAG_2_OP),
        ]):
            attn5.W_q[base + k, fetch_dim] = L
            attn5.W_k[base + k, flag_dim] = L
        # V: detect IMM (byte 1) and EXIT (byte 38)
        # IMM = 0x01: EMBED_LO[1]=1, EMBED_HI[0]=1
        # EXIT = 0x26: EMBED_LO[6]=1, EMBED_HI[2]=1
        # V dim: just copy relevant nibble dims for detection
        attn5.W_v[base + 3, BD.EMBED_LO + 1] = 1.0   # IMM lo indicator
        attn5.W_v[base + 4, BD.EMBED_HI + 0] = 1.0   # IMM hi indicator
        attn5.W_v[base + 5, BD.EMBED_LO + 6] = 1.0   # EXIT lo indicator
        attn5.W_v[base + 6, BD.EMBED_HI + 2] = 1.0   # EXIT hi indicator
        # O: write to staging dims
        attn5.W_o[IS_IMM_FETCH, base + 3] = 0.5   # Will AND in FFN
        attn5.W_o[IS_IMM_FETCH, base + 4] = 0.5   # Average of 2 indicators
        attn5.W_o[IS_EXIT_FETCH, base + 5] = 0.5
        attn5.W_o[IS_EXIT_FETCH, base + 6] = 0.5

        # Layer 5 FFN: Copy fetched imm to OUTPUT_LO/HI at AX marker only
        # (NOT at all IS_MARK positions — that leaks EMBED from DE/SE via attention)
        ffn5 = model.blocks[5].ffn
        unit = 0
        for k in range(16):
            ffn5.W_up[unit, BD.MARK_AX] = S  # Only at AX markers
            ffn5.b_up[unit] = -S * 0.5
            ffn5.W_gate[unit, BD.EMBED_LO + k] = 1.0
            ffn5.W_down[BD.OUTPUT_LO + k, unit] = 2.0 / S
            unit += 1
        for k in range(16):
            ffn5.W_up[unit, BD.MARK_AX] = S
            ffn5.b_up[unit] = -S * 0.5
            ffn5.W_gate[unit, BD.EMBED_HI + k] = 1.0
            ffn5.W_down[BD.OUTPUT_HI + k, unit] = 2.0 / S
            unit += 1

        # ===== LAYER 6: Opcode dispatch + HALT =====
        # For EXIT: override OUTPUT_LO/HI with CARRY_AX values
        # For IMM: keep the fetched imm (already in OUTPUT from Layer 5)

        attn6 = model.blocks[6].attn
        attn6.alibi_slopes.fill_(1.0)
        attn6.alibi_slopes[0] = 0.5  # head 0 needs wider window (AX→last MEM byte ≈ 23 tokens)
        # Head 0: Relay IS_EXIT_FETCH from AX marker to NEXT_SE position
        # (last MEM byte, where HALT decision must be made)
        base = 0 * HD
        L = 15.0
        attn6.W_q[base, BD.NEXT_SE] = L       # active where NEXT_SE is set
        attn6.W_k[base, BD.MARK_AX] = L       # target AX markers
        attn6.W_v[base + 1, IS_EXIT_FETCH] = 1.0
        attn6.W_o[IS_EXIT_AT_SE, base + 1] = 1.0  # separate dim to avoid EXIT override leak

        ffn6 = model.blocks[6].ffn
        unit = 0

        # EXIT AX override: when IS_EXIT_FETCH AND MARK_AX (only at AX markers),
        # subtract OUTPUT_LO/HI (fetched imm) and add CARRY_AX_LO/HI
        for k in range(16):
            # Subtract fetched OUTPUT_LO[k] when EXIT at AX marker
            ffn6.W_up[unit, IS_EXIT_FETCH] = S
            ffn6.W_up[unit, BD.MARK_AX] = S
            ffn6.b_up[unit] = -S * 1.5  # both must be ~1
            ffn6.W_gate[unit, BD.OUTPUT_LO + k] = -1.0
            ffn6.W_down[BD.OUTPUT_LO + k, unit] = 2.0 / S
            unit += 1
            # Add CARRY_AX_LO[k] when EXIT at AX marker
            ffn6.W_up[unit, IS_EXIT_FETCH] = S
            ffn6.W_up[unit, BD.MARK_AX] = S
            ffn6.b_up[unit] = -S * 1.5
            ffn6.W_gate[unit, CARRY_AX_LO + k] = 1.0
            ffn6.W_down[BD.OUTPUT_LO + k, unit] = 2.0 / S
            unit += 1

        for k in range(16):
            # Subtract fetched OUTPUT_HI[k] when EXIT at AX marker
            ffn6.W_up[unit, IS_EXIT_FETCH] = S
            ffn6.W_up[unit, BD.MARK_AX] = S
            ffn6.b_up[unit] = -S * 1.5
            ffn6.W_gate[unit, BD.OUTPUT_HI + k] = -1.0
            ffn6.W_down[BD.OUTPUT_HI + k, unit] = 2.0 / S
            unit += 1
            # Add CARRY_AX_HI[k] when EXIT at AX marker
            ffn6.W_up[unit, IS_EXIT_FETCH] = S
            ffn6.W_up[unit, BD.MARK_AX] = S
            ffn6.b_up[unit] = -S * 1.5
            ffn6.W_gate[unit, CARRY_AX_HI + k] = 1.0
            ffn6.W_down[BD.OUTPUT_HI + k, unit] = 2.0 / S
            unit += 1

        # HALT: when IS_EXIT_AT_SE AND NEXT_SE → emit HALT instead of STEP_END
        ffn6.W_up[unit, IS_EXIT_AT_SE] = S
        ffn6.b_up[unit] = -S * 0.3
        ffn6.W_gate[unit, BD.NEXT_SE] = 1.0
        ffn6.W_down[BD.NEXT_HALT, unit] = 2.0 / S
        ffn6.W_down[BD.NEXT_SE, unit] = -2.0 / S
        unit += 1

        # ===== OUTPUT HEAD =====
        head = model.head
        head.weight.zero_()
        head.bias.zero_()

        for b in range(256):
            lo, hi = b & 0xF, (b >> 4) & 0xF
            head.weight[b, BD.OUTPUT_LO + lo] = 5.0
            head.weight[b, BD.OUTPUT_HI + hi] = 5.0
            head.bias[b] = -5.0
        head.bias[0] = -4.0

        for tok, flag in [(Token.REG_PC, BD.NEXT_PC), (Token.REG_AX, BD.NEXT_AX),
                          (Token.REG_SP, BD.NEXT_SP), (Token.REG_BP, BD.NEXT_BP),
                          (Token.MEM, BD.NEXT_MEM), (Token.STEP_END, BD.NEXT_SE),
                          (Token.HALT, BD.NEXT_HALT)]:
            head.weight[tok, flag] = 20.0
            head.bias[tok] = -10.0

        for tok in [Token.CODE_START, Token.CODE_END, Token.DATA_START,
                    Token.DATA_END, Token.SEP, Token.STACK0]:
            head.bias[tok] = -50.0

    return model


# =============================================================================
# TEST
# =============================================================================
def run_program(model, bytecode, max_steps=5):
    """Run a program and return generated tokens."""
    from neural_vm.run_vm import AutoregressiveVMRunner
    runner = AutoregressiveVMRunner()
    runner.model = model
    context = runner._build_context(bytecode, b'', [])

    names = {
        Token.REG_PC: 'PC', Token.REG_AX: 'AX', Token.REG_SP: 'SP',
        Token.REG_BP: 'BP', Token.MEM: 'MEM', Token.STEP_END: 'SE',
        Token.HALT: 'HALT', Token.DATA_END: 'DE',
        Token.CODE_START: 'CS', Token.CODE_END: 'CE',
    }

    generated = []
    for _ in range(max_steps * 30 + 10):
        tok = model.generate_next(context)
        context.append(tok)
        generated.append(tok)
        if tok == Token.HALT:
            break

    # Parse steps
    print(f"Generated {len(generated)} tokens")
    steps = []
    step_tokens = []
    for tok in generated:
        step_tokens.append(tok)
        if tok in (Token.STEP_END, Token.HALT):
            steps.append(step_tokens)
            step_tokens = []

    for i, step in enumerate(steps):
        print(f"\nStep {i+1} ({len(step)} tokens):")
        display = []
        for tok in step:
            name = names.get(tok, str(tok))
            display.append(name)
        print(f"  {' '.join(display)}")

        if step[0] == Token.REG_PC and len(step) >= 5:
            pc = sum(step[j+1] << (j*8) for j in range(4))
            print(f"  PC = {pc}")
        if len(step) >= 10 and step[5] == Token.REG_AX:
            ax = sum(step[j+6] << (j*8) for j in range(4))
            print(f"  AX = {ax}")

    # Exit code
    exit_code = 0
    for i in range(len(context) - 1, -1, -1):
        if context[i] == Token.REG_AX and i + 4 < len(context):
            exit_code = sum(context[i+1+j] << (j*8) for j in range(4))
            break

    return exit_code, generated


def main():
    print("=" * 60)
    print("MULTI-INSTRUCTION WEIGHT BAKING TEST")
    print("Supports: IMM, EXIT. Up to 3 instructions.")
    print("=" * 60)

    model = AutoregressiveVM(d_model=256, n_layers=8, n_heads=4)
    bake_v2(model)
    results = []

    tests = [
        ("IMM 42; EXIT", [Opcode.IMM | (42 << 8), Opcode.EXIT], 42),
        ("IMM 5; IMM 3; EXIT", [Opcode.IMM | (5 << 8), Opcode.IMM | (3 << 8), Opcode.EXIT], 3),
        ("IMM 100; EXIT", [Opcode.IMM | (100 << 8), Opcode.EXIT], 100),
        ("IMM 255; EXIT", [Opcode.IMM | (255 << 8), Opcode.EXIT], 255),
        ("IMM 0; EXIT", [Opcode.IMM | (0 << 8), Opcode.EXIT], 0),
    ]

    for name, bytecode, expected in tests:
        print(f"\n--- {name} ---")
        exit_code, _ = run_program(model, bytecode)
        result = "PASS" if exit_code == expected else "FAIL"
        print(f"\nExit code: {exit_code} (expected {expected}) → {result}")
        results.append((name, result))

    print(f"\n{'=' * 60}")
    all_pass = all(r == "PASS" for _, r in results)
    for name, result in results:
        print(f"  {result}: {name}")
    print(f"{'=' * 60}")
    print(f"{'ALL PASS' if all_pass else 'SOME FAILED'}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
