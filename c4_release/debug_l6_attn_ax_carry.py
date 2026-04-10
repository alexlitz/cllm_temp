#!/usr/bin/env python3
"""Debug which L6 attention head writes AX_CARRY_HI at PC byte 0."""
import torch
import torch.nn.functional as F
import sys
sys.path.insert(0, '.')

from neural_vm.vm_step import AutoregressiveVM, set_vm_weights, Token, _SetDim as BD
from neural_vm.speculative import DraftVM
from neural_vm.embedding import Opcode

bytecode = [Opcode.JMP | (16 << 8)]

def build_context(bytecode):
    tokens = [Token.CODE_START]
    for instr in bytecode:
        op = instr & 0xFF
        imm = instr >> 8
        tokens.append(op)
        for i in range(4): tokens.append((imm >> (i * 8)) & 0xFF)
        for _ in range(3): tokens.append(0)
    tokens.append(Token.CODE_END)
    tokens.append(Token.DATA_START)
    tokens.append(Token.DATA_END)
    return tokens

model = AutoregressiveVM()
set_vm_weights(model)
model.eval()

context = build_context(bytecode)
draft = DraftVM(bytecode)
draft.step()
tokens = draft.draft_tokens()

ctx = context + tokens[:2]
pc_byte0_pos = len(ctx) - 1
pc_marker_pos = len(ctx) - 2

print(f"Context: {ctx}")
print(f"PC marker pos: {pc_marker_pos} (token={ctx[pc_marker_pos]})")
print(f"PC byte 0 pos: {pc_byte0_pos} (token={ctx[pc_byte0_pos]})")

token_ids = torch.tensor([ctx], dtype=torch.long)

with torch.no_grad():
    x = model.embed(token_ids)
    for i in range(6):
        x = model.blocks[i](x)

    print(f"\nBefore blocks[6].attn:")
    print(f"  AX_CARRY_HI[1] at PC byte 0: {x[0, pc_byte0_pos, BD.AX_CARRY_HI + 1].item():.4f}")
    print(f"  AX_CARRY_HI[1] at PC marker: {x[0, pc_marker_pos, BD.AX_CARRY_HI + 1].item():.4f}")

    # Get attention output delta per head manually
    attn = model.blocks[6].attn
    B, S, D = x.shape
    H = attn.num_heads
    HD = D // H

    Q = F.linear(x, attn.W_q).view(B, S, H, HD).transpose(1, 2)  # [B, H, S, HD]
    K = F.linear(x, attn.W_k).view(B, S, H, HD).transpose(1, 2)
    V = F.linear(x, attn.W_v).view(B, S, H, HD).transpose(1, 2)

    # Causal attention with scale
    scores = torch.matmul(Q, K.transpose(-2, -1)) * (HD ** -0.5)

    # Causal mask
    causal_mask = torch.triu(torch.ones(S, S, device=scores.device), diagonal=1) * -1e9
    scores = scores + causal_mask

    # ALiBi (if present)
    if hasattr(attn, 'alibi_slopes') and attn.alibi_slopes is not None:
        positions = torch.arange(S, device=x.device)
        dist = positions.unsqueeze(0) - positions.unsqueeze(1)  # [S, S]
        for h in range(H):
            slope = attn.alibi_slopes[h]
            scores[0, h] += slope * dist

    attn_weights = F.softmax(scores, dim=-1)

    # Check which heads have non-zero W_o for AX_CARRY_HI[1]
    W_o = attn.W_o.data
    for h in range(H):
        o_start = h * HD
        o_slice = W_o[BD.AX_CARRY_HI + 1, o_start:o_start+HD]
        if o_slice.abs().sum() > 0.01:
            print(f"\nHead {h} has W_o writing to AX_CARRY_HI[1]")
            active_dims = (o_slice.abs() > 0.01).nonzero().squeeze(-1)
            print(f"  W_o active dims (in head): {active_dims.tolist()}")

            # Show attention weights at PC byte 0 position
            attn_row = attn_weights[0, h, pc_byte0_pos, :]
            top5 = attn_row.topk(5)
            print(f"  At PC byte 0 (pos {pc_byte0_pos}), attention to:")
            for val, pos in zip(top5.values, top5.indices):
                print(f"    pos {pos.item()} (tok={ctx[pos.item()] if pos.item() < len(ctx) else '?'}): {val.item():.4f}")

            # Show Q at PC byte 0 for this head
            q_byte0 = Q[0, h, pc_byte0_pos, :]
            print(f"  Q[pc_byte0] sum: {q_byte0.sum().item():.4f}, nonzero dims: {(q_byte0.abs() > 0.1).nonzero().squeeze(-1).tolist()}")

            # Show what V values are at attended positions for the active O dims
            for dim_in_head in active_dims.tolist():
                v_dim = V[0, h, :, dim_in_head]  # V values at all positions for this dim
                print(f"  V dim {dim_in_head} at top positions:")
                for val, pos in zip(top5.values[:3], top5.indices[:3]):
                    v_val = v_dim[pos.item()].item()
                    print(f"    pos {pos.item()}: V={v_val:.4f}, attn={val.item():.4f}, contrib={v_val*val.item():.4f}")
