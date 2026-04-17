#!/usr/bin/env python3
"""Debug L7 Head 0 attention pattern directly."""
import torch
import torch.nn.functional as F
import sys
sys.path.insert(0, '.')

from neural_vm.vm_step import AutoregressiveVM, set_vm_weights, Token, _SetDim as BD
from neural_vm.speculative import DraftVM
from neural_vm.embedding import Opcode
from neural_vm.constants import IMMEDIATE_SIZE, PADDING_SIZE

BYTECODE = [Opcode.IMM | (6 << 8), Opcode.PSH, Opcode.IMM | (7 << 8), Opcode.MUL, Opcode.EXIT]

def build_context(bytecode, data=b''):
    context = [Token.CODE_START]
    for instr in bytecode:
        op = instr & 0xFF
        imm = instr >> 8
        context.append(op)
        for i in range(IMMEDIATE_SIZE):
            context.append((imm >> (i * 8)) & 0xFF)
        for _ in range(PADDING_SIZE):
            context.append(0)
    context.append(Token.CODE_END)
    context.append(Token.DATA_START)
    context.extend(list(data))
    context.append(Token.DATA_END)
    return context

def main():
    model = AutoregressiveVM()
    set_vm_weights(model)
    model.eval()

    context = build_context(BYTECODE)
    draft = DraftVM(BYTECODE)

    for step in range(3):
        draft.step()
        context.extend(draft.draft_tokens())

    draft.step()
    step4_tokens = draft.draft_tokens()
    context_for_pred = context + step4_tokens[:6]
    ax_marker_pos = len(context_for_pred) - 1

    token_ids = torch.tensor([context_for_pred], dtype=torch.long)

    with torch.no_grad():
        x = model.embed(token_ids)
        for i in range(7):
            x = model.blocks[i](x)

    # Get L7 attention layer
    attn7 = model.blocks[7].attn
    HD = 64
    seq_len = token_ids.shape[1]

    print("Computing L7 Head 0 attention manually...")
    print("=" * 80)

    # Compute Q, K, V for head 0
    base = 0 * HD
    q = torch.zeros(HD)
    k = torch.zeros(seq_len, HD)
    v = torch.zeros(seq_len, HD)

    for d in range(HD):
        # Q at ax_marker_pos
        for src_dim in range(x.shape[2]):
            q[d] += attn7.W_q[base + d, src_dim].item() * x[0, ax_marker_pos, src_dim].item()

        # K and V at each position
        for pos in range(seq_len):
            for src_dim in range(x.shape[2]):
                k[pos, d] += attn7.W_k[base + d, src_dim].item() * x[0, pos, src_dim].item()
                v[pos, d] += attn7.W_v[base + d, src_dim].item() * x[0, pos, src_dim].item()

    # No biases in this attention implementation

    # Compute attention scores (QK^T / sqrt(HD))
    scores = torch.mv(k, q) / (HD ** 0.5)

    # Add ALiBi
    alibi_slope = 0.5  # From set_vm_weights
    for pos in range(seq_len):
        dist = ax_marker_pos - pos
        scores[pos] -= alibi_slope * dist

    print(f"Top 10 attention scores for Head 0 (before softmax1):")
    top_scores, top_pos = torch.topk(scores, 10)
    for i, (s, p) in enumerate(zip(top_scores, top_pos)):
        token = token_ids[0, p].item()
        stack0_flag = x[0, p, BD.STACK0_BYTE0].item()
        print(f"  {i+1}. pos {p.item():3d}: score={s.item():8.4f}, token={token:3d}, STACK0_BYTE0={stack0_flag:.2f}")

    # Compute softmax1 attention weights
    exp_scores = torch.exp(scores)
    attn_weights = exp_scores / (1.0 + exp_scores.sum())

    print(f"\nTop 10 attention weights (after softmax1):")
    top_weights, top_pos = torch.topk(attn_weights, 10)
    for i, (w, p) in enumerate(zip(top_weights, top_pos)):
        token = token_ids[0, p].item()
        clean_lo_7 = x[0, p, BD.CLEAN_EMBED_LO + 7].item()
        print(f"  {i+1}. pos {p.item():3d}: weight={w.item():.6f}, token={token:3d}, CLEAN_EMBED_LO[7]={clean_lo_7:.2f}")

    # Compute attention output
    attn_out = torch.zeros(HD)
    for pos in range(seq_len):
        attn_out += attn_weights[pos] * v[pos]

    print(f"\nAttention output V slots 1-16 (should go to ALU_LO):")
    for slot in range(1, 17):
        val = attn_out[slot].item()
        if abs(val) > 0.01:
            print(f"  V[{slot:2d}] (→ ALU_LO[{slot-1:2d}]) = {val:.4f}")

    # Apply output projection
    alu_lo_contrib = torch.zeros(16)
    for k in range(16):
        for v_slot in range(HD):
            alu_lo_contrib[k] += attn7.W_o[BD.ALU_LO + k, base + v_slot].item() * attn_out[v_slot].item()

    print(f"\nALU_LO contribution from Head 0:")
    for k in range(16):
        if abs(alu_lo_contrib[k]) > 0.01:
            print(f"  ALU_LO[{k:2d}] += {alu_lo_contrib[k]:.4f}")

if __name__ == "__main__":
    main()
