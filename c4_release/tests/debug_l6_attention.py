#!/usr/bin/env python3
"""Debug L6 attention head 6 - PSH STACK0 prediction.

Trace why attention is going to wrong AX marker for PSH.
"""

import torch
import sys
sys.path.insert(0, '/home/alexlitz/Documents/misc/c4_release/c4_release')

from neural_vm.vm_step import AutoregressiveVM, set_vm_weights, _SetDim, Token
from neural_vm.embedding import Opcode, E

def main():
    # Create model
    model = AutoregressiveVM()
    model.eval()
    set_vm_weights(model)

    BD = _SetDim

    # Build PSH test context: IMM 1 + PSH
    # Step 0: IMM 1  (AX becomes 1)
    # Step 1: PSH    (STACK0 should become AX=1)

    # Context layout:
    # CODE_START + [IMM_1 instruction: 8 bytes] + [PSH instruction: 8 bytes] + CODE_END
    # DATA_START + DATA_END
    # Then 35 tokens for step 0 (IMM 1) + N tokens into step 1

    # Build context
    context = [Token.CODE_START]

    # IMM 1 instruction: op=0x04 (OP_IMM), imm=1
    imm1 = 0x04 | (1 << 8)  # IMM with value 1
    op = imm1 & 0xFF
    imm = imm1 >> 8
    context.append(op)
    for i in range(4):  # 4 immediate bytes
        context.append((imm >> (i * 8)) & 0xFF)
    for _ in range(3):  # 3 padding bytes
        context.append(0)

    # PSH instruction: op=0x24 (OP_PSH), imm=0
    psh = Opcode.PSH
    context.append(psh)
    for i in range(4):
        context.append(0)
    for _ in range(3):
        context.append(0)

    context.append(Token.CODE_END)
    context.append(Token.DATA_START)
    context.append(Token.DATA_END)

    # Step 0: 35 tokens (PC(5)+AX(5)+SP(5)+BP(5)+STACK0(5)+MEM(9)+SE(1))
    # Each register: 1 marker + 4 bytes = 5 tokens
    # MEM: 1 marker + 8 bytes = 9 tokens
    # SE: 1 token
    step0 = []
    step0.append(Token.REG_PC)  # PC marker
    step0.extend([1, 0, 0, 0])  # PC = 1 after IMM execution (4 bytes)
    step0.append(Token.REG_AX)  # AX marker
    step0.extend([1, 0, 0, 0])  # AX = 1 after IMM (4 bytes)
    step0.append(Token.REG_SP)  # SP marker
    # SP = 0xFFFFF8 in 4 bytes: 0xF8, 0xFF, 0xFF, 0x00
    step0.extend([0xF8, 0xFF, 0xFF, 0x00])
    step0.append(Token.REG_BP)  # BP marker
    step0.extend([0, 0, 0, 0])  # BP = 0 initially (4 bytes)
    step0.append(Token.STACK0)  # STACK0 marker
    step0.extend([0, 0, 0, 0])  # STACK0 = 0 initially (4 bytes)
    step0.append(Token.MEM)  # MEM marker
    step0.extend([0, 0, 0, 0, 0, 0, 0, 0])  # MEM = 8 bytes of zeros
    step0.append(Token.STEP_END)  # Step-end marker

    # Verify step0 is 35 tokens
    assert len(step0) == 35, f"step0 should be 35 tokens, got {len(step0)}"

    context.extend(step0)

    # Step 1: PSH execution - partially complete (up to STACK0 marker)
    # For PSH: PC += 1, SP -= 4, STACK0 = old AX, *SP = old AX
    # We need to predict STACK0 byte 0 = AX value = 1
    step1_partial = []
    step1_partial.append(Token.REG_PC)  # PC marker
    step1_partial.extend([2, 0, 0, 0])  # PC = 2 after PSH (4 bytes)
    step1_partial.append(Token.REG_AX)  # AX marker
    step1_partial.extend([1, 0, 0, 0])  # AX = 1 unchanged (4 bytes)
    step1_partial.append(Token.REG_SP)  # SP marker
    # SP = old_SP - 4 = 0xFFFFF8 - 4 = 0xFFFFF4 in 4 bytes: F4 FF FF 00
    step1_partial.extend([0xF4, 0xFF, 0xFF, 0x00])
    step1_partial.append(Token.REG_BP)  # BP marker
    step1_partial.extend([0, 0, 0, 0])  # BP unchanged (4 bytes)
    step1_partial.append(Token.STACK0)  # STACK0 marker - prediction starts here!

    context.extend(step1_partial)

    print(f"Context length: {len(context)} tokens")
    print(f"STACK0 marker at position: {len(context) - 1}")

    # Find positions of interest
    stack0_marker_pos = len(context) - 1
    # CODE_START (1) + IMM (8) + PSH (8) + CODE_END (1) + DATA_START (1) + DATA_END (1) = 20
    prefix_len = 1 + 8 + 8 + 1 + 1 + 1  # 20 tokens
    step0_start = prefix_len  # Position 20
    step1_start = step0_start + 35  # Position 55

    # AX marker is at offset 5 from step start (PC marker + 4 bytes)
    step0_ax_marker = step0_start + 5  # PC(5) then AX marker = pos 25
    step1_ax_marker = step1_start + 5  # pos 60

    print(f"\nPositions:")
    print(f"  Step 0 AX marker: {step0_ax_marker}")
    print(f"  Step 1 AX marker: {step1_ax_marker}")
    print(f"  STACK0 marker (query): {stack0_marker_pos}")

    # Run forward pass through layer 6
    token_ids = torch.tensor([context], dtype=torch.long)

    # Get embeddings
    x = model.embed(token_ids)

    # Run through layers 0-5 (blocks 0-5)
    for i in range(6):
        x = model.blocks[i](x)

    # Now extract attention from layer 6
    layer6 = model.blocks[6]
    attn = layer6.attn  # Use .attn instead of .attention

    # Get activations at STACK0 marker position
    x_stack0 = x[0, stack0_marker_pos, :]

    # Check MARK_STACK0 and MARK_AX dims
    print(f"\nActivations at STACK0 marker (pos {stack0_marker_pos}):")
    print(f"  MARK_STACK0: {x_stack0[BD.MARK_STACK0].item():.2f}")
    print(f"  MARK_AX: {x_stack0[BD.MARK_AX].item():.2f}")

    # Check K at AX marker positions
    x_ax0 = x[0, step0_ax_marker, :]
    x_ax1 = x[0, step1_ax_marker, :]

    print(f"\nActivations at step 0 AX marker (pos {step0_ax_marker}):")
    print(f"  MARK_AX: {x_ax0[BD.MARK_AX].item():.2f}")
    print(f"  AX_CARRY_LO[0]: {x_ax0[BD.AX_CARRY_LO + 0].item():.2f}")
    print(f"  AX_CARRY_LO[1]: {x_ax0[BD.AX_CARRY_LO + 1].item():.2f}")

    print(f"\nActivations at step 1 AX marker (pos {step1_ax_marker}):")
    print(f"  MARK_AX: {x_ax1[BD.MARK_AX].item():.2f}")
    print(f"  AX_CARRY_LO[0]: {x_ax1[BD.AX_CARRY_LO + 0].item():.2f}")
    print(f"  AX_CARRY_LO[1]: {x_ax1[BD.AX_CARRY_LO + 1].item():.2f}")

    # Compute Q and K for head 6 manually
    HD = attn.head_dim
    D = attn.dim

    Q_all = torch.nn.functional.linear(x, attn.W_q)  # [B, S, D]
    K_all = torch.nn.functional.linear(x, attn.W_k)  # [B, S, D]

    # Extract head 6
    head = 6
    Q6 = Q_all[0, :, head*HD:(head+1)*HD]  # [S, HD]
    K6 = K_all[0, :, head*HD:(head+1)*HD]  # [S, HD]

    # Q at STACK0 marker
    q_stack0 = Q6[stack0_marker_pos, :]  # [HD]

    # K at all positions
    k_all = K6  # [S, HD]

    # Compute raw scores Q @ K^T
    raw_scores = torch.matmul(q_stack0, k_all.t())  # [S]

    print(f"\n=== Head 6 Attention Analysis ===")
    print(f"Q at STACK0 (pos {stack0_marker_pos}): norm = {q_stack0.norm().item():.2f}")
    print(f"  Q[0] (gate): {q_stack0[0].item():.2f}")

    # Find top scores
    scores_sorted, indices_sorted = raw_scores.sort(descending=True)

    print(f"\nTop 10 raw scores from STACK0 marker:")
    for i in range(min(10, len(indices_sorted))):
        pos = indices_sorted[i].item()
        score = scores_sorted[i].item()
        # Identify what's at this position
        if pos == step0_ax_marker:
            label = "Step0 AX"
        elif pos == step1_ax_marker:
            label = "Step1 AX"
        elif pos == stack0_marker_pos:
            label = "STACK0 (self)"
        elif context[pos] == Token.REG_AX:
            label = "AX marker"
        elif context[pos] == Token.REG_PC:
            label = "PC marker"
        elif context[pos] == Token.REG_SP:
            label = "SP marker"
        elif context[pos] == Token.REG_BP:
            label = "BP marker"
        elif context[pos] == Token.STACK0:
            label = "STACK0 marker"
        elif context[pos] == Token.MEM:
            label = "MEM marker"
        else:
            label = f"token={context[pos]}"

        print(f"  pos={pos:3d} score={score:8.2f}  {label}")

    # Compute attention with ALiBi
    scale = HD ** -0.5
    scores_scaled = raw_scores * scale

    # ALiBi slope for head 6 (8 heads total)
    # slopes = 2^(-8/n * (i+1)) for i in range(n)
    # For 8 heads: slopes = [0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125, 0.00390625]
    n_heads = 8
    slopes = [2 ** (-8.0/n_heads * (i+1)) for i in range(n_heads)]
    slope = slopes[head]

    print(f"\nALiBi slope for head 6: {slope:.6f}")

    # Add ALiBi bias (query position = stack0_marker_pos)
    positions = torch.arange(len(context))
    alibi = -slope * (stack0_marker_pos - positions)
    alibi = alibi.clamp(max=0)  # Only negative bias for future positions

    scores_with_alibi = scores_scaled + alibi

    print(f"\nScores with ALiBi for key positions:")
    print(f"  Step 0 AX (pos {step0_ax_marker}):")
    print(f"    raw={raw_scores[step0_ax_marker].item():.2f}, scaled={scores_scaled[step0_ax_marker].item():.2f}")
    print(f"    alibi={alibi[step0_ax_marker].item():.2f}, total={scores_with_alibi[step0_ax_marker].item():.2f}")
    print(f"  Step 1 AX (pos {step1_ax_marker}):")
    print(f"    raw={raw_scores[step1_ax_marker].item():.2f}, scaled={scores_scaled[step1_ax_marker].item():.2f}")
    print(f"    alibi={alibi[step1_ax_marker].item():.2f}, total={scores_with_alibi[step1_ax_marker].item():.2f}")

    # Apply softmax
    attn_weights = torch.softmax(scores_with_alibi, dim=-1)

    print(f"\nAttention weights for key positions:")
    print(f"  Step 0 AX (pos {step0_ax_marker}): {attn_weights[step0_ax_marker].item():.6f}")
    print(f"  Step 1 AX (pos {step1_ax_marker}): {attn_weights[step1_ax_marker].item():.6f}")

    # Find top attention weights
    weights_sorted, indices_sorted = attn_weights.sort(descending=True)

    print(f"\nTop 10 attention weights from STACK0 marker:")
    for i in range(min(10, len(indices_sorted))):
        pos = indices_sorted[i].item()
        weight = weights_sorted[i].item()
        if weight < 1e-6:
            break
        # Identify what's at this position
        if pos == step0_ax_marker:
            label = "Step0 AX"
        elif pos == step1_ax_marker:
            label = "Step1 AX"
        elif pos == stack0_marker_pos:
            label = "STACK0 (self)"
        elif pos < len(context) and context[pos] == Token.REG_AX:
            label = "AX marker"
        elif pos < len(context) and context[pos] == Token.REG_PC:
            label = "PC marker"
        else:
            label = f"token={context[pos] if pos < len(context) else 'OOB'}"

        print(f"  pos={pos:3d} weight={weight:.6f}  {label}")

    # Check if causal mask is applied
    print(f"\n=== Checking actual layer 6 attention ===")

    # Run actual attention forward
    B, S, D = x.shape
    H = attn.num_heads
    HD = attn.head_dim

    Q = torch.nn.functional.linear(x, attn.W_q).view(B, S, H, HD).transpose(1, 2)
    K = torch.nn.functional.linear(x, attn.W_k).view(B, S, H, HD).transpose(1, 2)
    V = torch.nn.functional.linear(x, attn.W_v).view(B, S, H, HD).transpose(1, 2)

    scores = torch.matmul(Q, K.transpose(-2, -1)) * scale

    # Add the mask
    scores = scores + attn.mask[:S, :S]

    # Check if mask has causal structure
    print(f"Mask shape: {attn.mask.shape}")
    print(f"Mask at query={stack0_marker_pos}, key={step0_ax_marker}: {attn.mask[stack0_marker_pos, step0_ax_marker].item():.2f}")
    print(f"Mask at query={stack0_marker_pos}, key={step1_ax_marker}: {attn.mask[stack0_marker_pos, step1_ax_marker].item():.2f}")

    # Extract head 6 scores
    h6_scores = scores[0, head, stack0_marker_pos, :]  # [S]

    print(f"\nHead 6 scores (with mask) for key positions:")
    print(f"  Step 0 AX (pos {step0_ax_marker}): {h6_scores[step0_ax_marker].item():.2f}")
    print(f"  Step 1 AX (pos {step1_ax_marker}): {h6_scores[step1_ax_marker].item():.2f}")

    # Apply softmax
    h6_attn = torch.softmax(h6_scores, dim=-1)

    print(f"\nHead 6 attention (after mask + softmax):")
    print(f"  Step 0 AX (pos {step0_ax_marker}): {h6_attn[step0_ax_marker].item():.6f}")
    print(f"  Step 1 AX (pos {step1_ax_marker}): {h6_attn[step1_ax_marker].item():.6f}")

    # Find top attention weights
    h6_sorted, h6_indices = h6_attn.sort(descending=True)

    print(f"\nTop 10 head 6 attention weights:")
    for i in range(min(10, len(h6_indices))):
        pos = h6_indices[i].item()
        weight = h6_sorted[i].item()
        if weight < 1e-6:
            break
        if pos == step0_ax_marker:
            label = "Step0 AX"
        elif pos == step1_ax_marker:
            label = "Step1 AX"
        elif pos == stack0_marker_pos:
            label = "STACK0 (self)"
        elif pos < len(context) and context[pos] == Token.REG_AX:
            label = "AX marker"
        elif pos < len(context) and context[pos] == Token.REG_PC:
            label = "PC marker"
        elif pos < len(context) and context[pos] == Token.REG_SP:
            label = "SP marker"
        elif pos < len(context) and context[pos] == Token.REG_BP:
            label = "BP marker"
        elif pos < len(context) and context[pos] == Token.STACK0:
            label = "STACK0 marker"
        elif pos < len(context) and context[pos] == Token.MEM:
            label = "MEM marker"
        else:
            label = f"token={context[pos] if pos < len(context) else 'OOB'}"

        score_val = h6_scores[pos].item()
        print(f"  pos={pos:3d} weight={weight:.6f} score={score_val:8.2f}  {label}")

if __name__ == '__main__':
    main()
