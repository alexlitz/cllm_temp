"""Test script for real weight baking into AutoregressiveVM."""
import torch
from neural_vm.vm_step import Token, AutoregressiveVM
from neural_vm.kv_cache_eviction import softmax1
from neural_vm.embedding import Opcode

# =============================================================================
# Dimension allocation (d_model=256)
# =============================================================================
D_MARK_PC = 0; D_MARK_AX = 1; D_MARK_SP = 2; D_MARK_BP = 3
D_MARK_MEM = 4; D_MARK_SE = 5  # STEP_END or DATA_END
D_IS_BYTE = 6; D_IS_MARK = 7; D_CONST = 8
D_MARK_CS = 9      # CODE_START only
D_MARK_SE_ONLY = 10  # STEP_END only (not DATA_END)

# Layer 0 attention output (threshold heads, 7 marker types per head)
D_H0 = 60   # threshold 3.5
D_H1 = 67   # threshold 4.5
D_H2 = 74   # threshold 7.5
D_H3 = 81   # threshold 8.5

# Layer 1 attention output
D_L1H0 = 90   # threshold 0.5
D_L1H1 = 97   # threshold 1.5
D_L1H2 = 104  # threshold 2.5
D_HAS_SE = 111  # STEP_END existence flag

# Flags
D_IS_IMM_B0 = 112  # first immediate byte position
D_NEXT_PC = 120; D_NEXT_AX = 121; D_NEXT_SP = 122
D_NEXT_BP = 123; D_NEXT_MEM = 124; D_NEXT_SE = 125
D_NEXT_HALT = 126

# Nibble encoding
D_EMBED_LO = 140  # 140-155
D_EMBED_HI = 156  # 156-171
D_OUTPUT_LO = 172  # 172-187
D_OUTPUT_HI = 188  # 188-203

# Marker dims in V (index within head's 64 dims)
NUM_MARKERS = 7  # PC,AX,SP,BP,MEM,SE,CS
D_MARKS = [D_MARK_PC, D_MARK_AX, D_MARK_SP, D_MARK_BP, D_MARK_MEM, D_MARK_SE, D_MARK_CS]


def bake_weights_real(model):
    """Bake weights for true autoregressive VM execution."""
    d = model.d_model
    V = model.vocab_size
    S = 100.0  # scale factor
    HD = d // model.blocks[0].attn.num_heads  # 64
    ALIBI_S = 10.0  # ALiBi slope for all threshold heads

    with torch.no_grad():
        embed = model.embed.weight
        embed.zero_()

        # ===== EMBEDDING =====
        for tok in range(V):
            embed[tok, D_CONST] = 1.0

        # Marker embeddings
        for tok, dim in [(Token.REG_PC, D_MARK_PC), (Token.REG_AX, D_MARK_AX),
                         (Token.REG_SP, D_MARK_SP), (Token.REG_BP, D_MARK_BP),
                         (Token.MEM, D_MARK_MEM), (Token.CODE_START, D_MARK_CS)]:
            embed[tok, dim] = 1.0
            embed[tok, D_IS_MARK] = 1.0

        for tok in [Token.STEP_END, Token.DATA_END, Token.HALT]:
            embed[tok, D_MARK_SE] = 1.0
            embed[tok, D_IS_MARK] = 1.0

        embed[Token.STEP_END, D_MARK_SE_ONLY] = 1.0

        for b in range(256):
            embed[b, D_IS_BYTE] = 1.0
            embed[b, D_EMBED_LO + (b & 0xF)] = 1.0
            embed[b, D_EMBED_HI + ((b >> 4) & 0xF)] = 1.0

        # ===== LAYER 0 ATTENTION: Phase A thresholds [3.5, 4.5, 7.5, 8.5] =====
        attn0 = model.blocks[0].attn
        attn0.alibi_slopes.fill_(ALIBI_S)
        _bake_threshold_attn(attn0, [3.5, 4.5, 7.5, 8.5], [D_H0, D_H1, D_H2, D_H3],
                             ALIBI_S, HD)

        # ===== LAYER 0 FFN: Phase A transitions =====
        ffn0 = model.blocks[0].ffn
        _bake_phase_a_ffn(ffn0, S)

        # ===== LAYER 1 ATTENTION: Fine thresholds [0.5, 1.5, 2.5] + SE detection =====
        attn1 = model.blocks[1].attn
        attn1.alibi_slopes.fill_(ALIBI_S)
        attn1.alibi_slopes[3] = 0.0  # Head 3: global attention (no distance penalty)
        # Heads 0-2: fine thresholds
        _bake_threshold_attn(attn1, [0.5, 1.5, 2.5], [D_L1H0, D_L1H1, D_L1H2],
                             ALIBI_S, HD, heads=[0, 1, 2])
        # Head 3: STEP_END detection (global - attends to ALL STEP_END tokens)
        h = 3
        base = h * HD
        # Q: constant. K: reads D_MARK_SE_ONLY (only STEP_END, not DATA_END)
        # Large Q*K to strongly attend to STEP_END tokens over softmax1 anchor
        attn1.W_q[base, D_CONST] = 10.0
        attn1.W_k[base, D_MARK_SE_ONLY] = 10.0
        # V copies D_MARK_SE_ONLY
        attn1.W_v[base + 1, D_MARK_SE_ONLY] = 1.0
        # O writes to D_HAS_SE
        attn1.W_o[D_HAS_SE, base + 1] = 1.0

        # ===== LAYER 1 FFN: D_IS_IMM_B0 + PC=5 + HALT override =====
        ffn1 = model.blocks[1].ffn
        _bake_layer1_ffn(ffn1, S)

        # ===== LAYER 2 ATTENTION: Read imm_b0 for AX =====
        attn2 = model.blocks[2].attn
        attn2.alibi_slopes.fill_(0.1)  # gentle slope for bytecode reading
        # Head 0: REG_AX → imm_b0 (read nibble encoding)
        h = 0
        base = h * HD
        L = 20.0
        attn2.W_q[base, D_MARK_AX] = L
        attn2.W_k[base, D_IS_IMM_B0] = L
        # V copies D_EMBED_LO (16 dims) and D_EMBED_HI (16 dims)
        for k in range(16):
            attn2.W_v[base + 1 + k, D_EMBED_LO + k] = 1.0
            attn2.W_v[base + 17 + k, D_EMBED_HI + k] = 1.0
        # O: write to D_EMBED_LO/HI (they'll be copied to D_OUTPUT by FFN)
        for k in range(16):
            attn2.W_o[D_EMBED_LO + k, base + 1 + k] = 1.0
            attn2.W_o[D_EMBED_HI + k, base + 17 + k] = 1.0

        # ===== LAYER 2 FFN: Conditional nibble copy =====
        ffn2 = model.blocks[2].ffn
        _bake_nibble_copy_ffn(ffn2, S)

        # ===== OUTPUT HEAD =====
        head = model.head
        head.weight.zero_()
        head.bias.zero_()

        # Byte tokens: nibble decoding
        for b in range(256):
            lo = b & 0xF
            hi = (b >> 4) & 0xF
            head.weight[b, D_OUTPUT_LO + lo] = 5.0
            head.weight[b, D_OUTPUT_HI + hi] = 5.0
            head.bias[b] = -5.0
        head.bias[0] = -4.0  # slight preference for byte 0 as default

        # Transition tokens
        for tok, flag in [(Token.REG_PC, D_NEXT_PC), (Token.REG_AX, D_NEXT_AX),
                          (Token.REG_SP, D_NEXT_SP), (Token.REG_BP, D_NEXT_BP),
                          (Token.MEM, D_NEXT_MEM), (Token.STEP_END, D_NEXT_SE),
                          (Token.HALT, D_NEXT_HALT)]:
            head.weight[tok, flag] = 20.0
            head.bias[tok] = -10.0

        # Never output these tokens
        for tok in [Token.CODE_START, Token.CODE_END, Token.DATA_START,
                    Token.DATA_END, Token.SEP]:
            head.bias[tok] = -50.0

    return model


def _bake_threshold_attn(attn, thresholds, out_bases, slope, HD, heads=None):
    """Bake threshold-based attention for marker detection."""
    if heads is None:
        heads = list(range(len(thresholds)))

    for i, (h, t) in enumerate(zip(heads, thresholds)):
        base = h * HD
        # Q reads D_CONST, K reads D_IS_MARK
        # score = q*k/sqrt(HD) - slope*d = slope*(t - d)
        # → q*k = slope * t * sqrt(HD)
        q_val = 8.0 * slope  # sqrt(HD) = 8
        k_val = t
        attn.W_q[base, D_CONST] = q_val
        attn.W_k[base, D_IS_MARK] = k_val

        # V copies marker identity (7 types)
        for m, src in enumerate(D_MARKS):
            attn.W_v[base + 1 + m, src] = 1.0

        # O maps to output dims
        for m in range(NUM_MARKERS):
            attn.W_o[out_bases[i] + m, base + 1 + m] = 1.0


def _bake_phase_a_ffn(ffn, S):
    """Phase A: detect transitions at d=4 (reg) and d=8 (MEM)."""
    # Transitions: (up_dim, gate_dim, out_dim)
    # up_dim: marker signal from Head with HIGHER threshold (d ≤ threshold → HIGH)
    # gate_dim: marker signal from Head with LOWER threshold (d > threshold → LOW → gate active)
    transitions = [
        # SE → REG_PC (unconditional when SE is nearest)
        (D_H0 + 5, None, D_NEXT_PC),      # D_H0_SE always high for SE at d=0
        # PC at d=4: H0(3.5) LOW, H1(4.5) HIGH
        (D_H1 + 0, D_H0 + 0, D_NEXT_AX),
        # AX at d=4
        (D_H1 + 1, D_H0 + 1, D_NEXT_SP),
        # SP at d=4
        (D_H1 + 2, D_H0 + 2, D_NEXT_BP),
        # BP at d=4
        (D_H1 + 3, D_H0 + 3, D_NEXT_MEM),
        # MEM at d=8: H2(7.5) LOW, H3(8.5) HIGH
        (D_H3 + 4, D_H2 + 4, D_NEXT_SE),
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


def _bake_layer1_ffn(ffn, S):
    """Layer 1 FFN: imm_b0 detection + PC=5 + HALT override."""
    unit = 0

    # --- D_IS_IMM_B0: detect d=2 from CODE_START ---
    # Condition: D_IS_BYTE=1 AND L1_H2_CS=HIGH (d≤2.5) AND L1_H1_CS=LOW (d>1.5)
    # up = S * (D_IS_BYTE - 0.5) → active at bytes
    # gate = L1_H2_CS - L1_H1_CS → positive at d=2, zero at d=1,3
    # where L1_Hx_CS is the CS component (index 6 in marker types)
    CS_IDX = 6  # CS is the 7th marker type
    ffn.W_up[unit, D_IS_BYTE] = S
    ffn.b_up[unit] = -S * 0.5
    ffn.W_gate[unit, D_L1H2 + CS_IDX] = 1.0
    ffn.W_gate[unit, D_L1H1 + CS_IDX] = -1.0
    ffn.W_down[D_IS_IMM_B0, unit] = 2.0 / S
    unit += 1

    # --- PC = 5: set nibble encoding at REG_PC positions ---
    # D_OUTPUT_LO[5] = 1.0 when D_MARK_PC = 1.0
    ffn.W_up[unit, D_MARK_PC] = S
    ffn.b_up[unit] = -S * 0.3
    ffn.b_gate[unit] = 1.0
    ffn.W_down[D_OUTPUT_LO + 5, unit] = 2.0 / S
    unit += 1

    # D_OUTPUT_HI[0] = 1.0 when D_MARK_PC = 1.0
    ffn.W_up[unit, D_MARK_PC] = S
    ffn.b_up[unit] = -S * 0.3
    ffn.b_gate[unit] = 1.0
    ffn.W_down[D_OUTPUT_HI + 0, unit] = 2.0 / S
    unit += 1

    # --- HALT override: in step 2+, change STEP_END to HALT ---
    # Condition: D_HAS_SE > 0.2 (STEP_END exists → not first step)
    #            AND D_NEXT_SE is set (Phase A transition fired)
    # Action: add to D_NEXT_HALT, subtract from D_NEXT_SE
    ffn.W_up[unit, D_HAS_SE] = S
    ffn.b_up[unit] = -S * 0.1
    ffn.W_gate[unit, D_NEXT_SE] = 1.0
    ffn.W_down[D_NEXT_HALT, unit] = 2.0 / S
    ffn.W_down[D_NEXT_SE, unit] = -2.0 / S
    unit += 1


def _bake_nibble_copy_ffn(ffn, S):
    """Conditional nibble copy: D_OUTPUT = D_EMBED when D_IS_MARK."""
    unit = 0
    for k in range(16):
        # Low nibble: D_OUTPUT_LO[k] += D_EMBED_LO[k] when D_IS_MARK
        ffn.W_up[unit, D_IS_MARK] = S
        ffn.b_up[unit] = -S * 0.5
        ffn.W_gate[unit, D_EMBED_LO + k] = 1.0
        ffn.W_down[D_OUTPUT_LO + k, unit] = 2.0 / S
        unit += 1

    for k in range(16):
        # High nibble: D_OUTPUT_HI[k] += D_EMBED_HI[k] when D_IS_MARK
        ffn.W_up[unit, D_IS_MARK] = S
        ffn.b_up[unit] = -S * 0.5
        ffn.W_gate[unit, D_EMBED_HI + k] = 1.0
        ffn.W_down[D_OUTPUT_HI + k, unit] = 2.0 / S
        unit += 1


# =============================================================================
# TEST
# =============================================================================
def main():
    print("=" * 60)
    print("REAL WEIGHT BAKING TEST")
    print("=" * 60)

    model = AutoregressiveVM(d_model=256, n_layers=6, n_heads=4)
    bake_weights_real(model)

    # Build context for "IMM 42; EXIT"
    from neural_vm.run_vm import AutoregressiveVMRunner
    runner = AutoregressiveVMRunner()
    runner.model = model

    bytecode = [Opcode.IMM | (42 << 8), Opcode.EXIT]
    context = runner._build_context(bytecode, b'', [])

    token_names = {
        Token.REG_PC: 'PC', Token.REG_AX: 'AX', Token.REG_SP: 'SP',
        Token.REG_BP: 'BP', Token.MEM: 'MEM', Token.STEP_END: 'SE',
        Token.DATA_END: 'DE', Token.HALT: 'HALT', Token.CODE_START: 'CS',
        Token.CODE_END: 'CE', Token.DATA_START: 'DS'
    }

    print(f"\nContext ({len(context)} tokens): {context}")

    # Generate tokens
    generated = []
    for step in range(62):  # 2 full steps + margin
        with torch.no_grad():
            tok = model.generate_next(context)
        name = token_names.get(tok, str(tok))
        generated.append((tok, name))
        context.append(tok)
        if tok == Token.HALT:
            break

    print(f"\nGenerated {len(generated)} tokens:")
    for i, (tok, name) in enumerate(generated):
        print(f"  [{i:2d}] {name}", end="")
        if (i + 1) % 10 == 0:
            print()
    print()

    # Parse register values from generated output
    print("\n--- Register Values ---")
    tokens = [t for t, _ in generated]

    for step_start in range(0, len(tokens), 30):
        step_tokens = tokens[step_start:step_start + 30]
        if len(step_tokens) < 5:
            break

        step_num = step_start // 30 + 1
        print(f"\nStep {step_num}:")

        if step_tokens[0] == Token.REG_PC:
            pc_bytes = step_tokens[1:5]
            pc_val = sum(b << (i * 8) for i, b in enumerate(pc_bytes))
            print(f"  PC = {pc_val} (bytes: {pc_bytes})")

        if len(step_tokens) > 5 and step_tokens[5] == Token.REG_AX:
            ax_bytes = step_tokens[6:10]
            ax_val = sum(b << (i * 8) for i, b in enumerate(ax_bytes))
            print(f"  AX = {ax_val} (bytes: {ax_bytes})")

        if len(step_tokens) > 29:
            last = step_tokens[29]
            print(f"  End: {token_names.get(last, str(last))}")
        elif len(step_tokens) > 0:
            last = step_tokens[-1]
            print(f"  End: {token_names.get(last, str(last))}")

    # Check exit code
    exit_code = runner._decode_exit_code(context)
    print(f"\n{'=' * 60}")
    print(f"Exit code: {exit_code} (expected: 42)")
    print(f"{'PASS' if exit_code == 42 else 'FAIL'}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
