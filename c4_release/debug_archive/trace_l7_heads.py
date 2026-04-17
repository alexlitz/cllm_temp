#!/usr/bin/env python3
"""Trace L7 attention head contributions to ALU_HI[0] at pos 9."""

import torch
import torch.nn.functional as F
from neural_vm.vm_step import AutoregressiveVM, set_vm_weights, Token, _SetDim as BD
from neural_vm.embedding import Opcode
from neural_vm.speculative import DraftVM

def build_context(bytecode):
    tokens = [Token.CODE_START]
    for instr in bytecode:
        op = instr & 0xFF
        imm = instr >> 8
        tokens.append(op)
        for i in range(4):
            tokens.append((imm >> (i * 8)) & 0xFF)
    tokens.append(Token.CODE_END)
    tokens.append(Token.DATA_START)
    tokens.append(Token.DATA_END)
    return tokens

print("Initializing model...")
model = AutoregressiveVM()
set_vm_weights(model)
model.eval()

# LEA 0
bytecode = [Opcode.LEA | (0 << 8)]
context = build_context(bytecode)
draft_vm = DraftVM(bytecode)
draft_vm.step()
draft_tokens = draft_vm.draft_tokens()

ctx_tensor = torch.tensor([context + draft_tokens], dtype=torch.long)
ctx_len = len(context)
pos_9 = 9  # PC marker position

print(f"Checking L7 head contributions to ALU_HI[0] at pos 9 for LEA 0:")

with torch.no_grad():
    emb = model.embed(ctx_tensor)
    x = emb

    # Run through L0-L6
    for i in range(7):
        x = model.blocks[i](x)

    attn7 = model.blocks[7].attn
    NH = attn7.num_heads
    HD = 512 // NH  # 64

    # Manual attention computation
    seq_len = x.size(1)

    # Compute Q, K, V
    Q = torch.matmul(x, attn7.W_q)  # [1, seq, d]
    K = torch.matmul(x, attn7.W_k)  # [1, seq, d]
    V = torch.matmul(x, attn7.W_v)  # [1, seq, d]

    # Reshape for multi-head attention
    Q = Q.view(1, seq_len, NH, HD).transpose(1, 2)  # [1, NH, seq, HD]
    K = K.view(1, seq_len, NH, HD).transpose(1, 2)  # [1, NH, seq, HD]
    V = V.view(1, seq_len, NH, HD).transpose(1, 2)  # [1, NH, seq, HD]

    # Compute attention scores
    scores = torch.matmul(Q, K.transpose(-2, -1))  # [1, NH, seq, seq]
    scores = scores / (HD ** 0.5)

    # Add ALiBi bias
    if hasattr(attn7, 'alibi_slopes') and attn7.alibi_slopes is not None:
        pos = torch.arange(seq_len)
        rel_pos = pos.unsqueeze(0) - pos.unsqueeze(1)  # [seq, seq]
        for h in range(NH):
            scores[0, h] += attn7.alibi_slopes[h] * rel_pos

    # Causal mask
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
    scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))

    # Softmax
    attn_weights = F.softmax(scores, dim=-1)  # [1, NH, seq, seq]

    # Check attention at position 9 for each head
    print(f"\n  Attention weights at pos {pos_9} (top 5 per head):")
    for h in range(min(2, NH)):  # Just check heads 0 and 1
        weights_at_pos = attn_weights[0, h, pos_9, :]  # [seq]
        top_k = torch.topk(weights_at_pos, min(5, seq_len))
        print(f"\n    Head {h}:")
        for i, (idx, w) in enumerate(zip(top_k.indices.tolist(), top_k.values.tolist())):
            if w > 0.001:
                tok = (context + draft_tokens)[idx]
                print(f"      pos {idx} (token {tok}): weight = {w:.4f}")

    # Compute output per head
    attn_out = torch.matmul(attn_weights, V)  # [1, NH, seq, HD]
    attn_out = attn_out.transpose(1, 2).contiguous().view(1, seq_len, -1)  # [1, seq, d]

    # Apply W_o per head and check contribution to dim 376
    target_dim = BD.ALU_HI  # dim 376
    print(f"\n  Per-head contribution to dim {target_dim} at pos {pos_9}:")
    for h in range(NH):
        base = h * HD
        # Get this head's output (before W_o)
        head_out = attn_out[0, pos_9, base:base+HD]  # [HD]
        # Apply W_o for this head
        contribution = 0
        for k in range(HD):
            contribution += attn7.W_o[target_dim, base + k].item() * head_out[k].item()
        if abs(contribution) > 0.1:
            print(f"    Head {h}: {contribution:.2f} ***")
        else:
            print(f"    Head {h}: {contribution:.2f}")
