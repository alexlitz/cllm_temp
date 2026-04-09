#!/usr/bin/env python3
"""Debug L7 Head 0 attention pattern - fast version."""
import torch
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

    print("=" * 80)
    print(f"Debugging L7 (blocks[7]) attention at AX marker (pos {ax_marker_pos})")
    print("=" * 80)

    with torch.no_grad():
        x = model.embed(token_ids)

        # Save state before L7
        x_before_l7 = x.clone()

        # Run through L1-L6
        for i in range(7):
            x = model.blocks[i](x)

        # Now run L7 attention step by step
        attn7 = model.blocks[7].attn

        # Compute Q, K, V using the attention layer's weights
        # x shape: [1, seq_len, dim]
        # W_q, W_k, W_v shape: [dim, dim]

        Q = x @ attn7.W_q.T  # [1, seq_len, dim]
        K = x @ attn7.W_k.T  # [1, seq_len, dim]
        V = x @ attn7.W_v.T  # [1, seq_len, dim]

        # Reshape for multi-head: [1, seq_len, num_heads, head_dim]
        seq_len = x.shape[1]
        num_heads = 8
        head_dim = x.shape[2] // num_heads

        Q = Q.view(1, seq_len, num_heads, head_dim).transpose(1, 2)  # [1, 8, seq_len, 64]
        K = K.view(1, seq_len, num_heads, head_dim).transpose(1, 2)
        V = V.view(1, seq_len, num_heads, head_dim).transpose(1, 2)

        # Focus on head 0
        q0 = Q[0, 0, ax_marker_pos, :]  # [64]
        k0 = K[0, 0, :, :]  # [seq_len, 64]
        v0 = V[0, 0, :, :]  # [seq_len, 64]

        # Compute attention scores for head 0 at AX marker position
        scores = (k0 @ q0) / (head_dim ** 0.5)  # [seq_len]

        # Add ALiBi
        alibi_slope = attn7.alibi_slopes[0].item() if attn7.alibi_slopes is not None else 0.5
        print(f"\nHead 0 ALiBi slope: {alibi_slope}")

        for pos in range(seq_len):
            dist = ax_marker_pos - pos
            if dist > 0:
                scores[pos] -= alibi_slope * dist

        print(f"\nTop 10 attention scores (before softmax1):")
        top_scores, top_pos = torch.topk(scores, min(10, seq_len))
        for s, p in zip(top_scores, top_pos):
            token = token_ids[0, p].item()
            stack0_flag = x[0, p, BD.STACK0_BYTE0].item()
            mark_ax = x[0, p, BD.MARK_AX].item()
            dist = ax_marker_pos - p.item()
            print(f"  pos {p.item():3d} (d={dist:3d}): score={s.item():8.2f}, token={token:3d}, " +
                  f"STACK0_BYTE0={stack0_flag:.2f}, MARK_AX={mark_ax:.2f}")

        # Compute softmax1 attention weights
        exp_scores = torch.exp(scores - scores.max())  # Numerical stability
        attn_weights = exp_scores / (torch.exp(-scores.max()) + exp_scores.sum())

        print(f"\nTop 10 attention weights (after softmax1):")
        top_weights, top_pos = torch.topk(attn_weights, min(10, seq_len))
        for w, p in zip(top_weights, top_pos):
            v_vals = []
            for slot in range(1, 17):  # V slots 1-16 → ALU_LO
                if abs(v0[p, slot].item()) > 0.01:
                    v_vals.append((slot-1, v0[p, slot].item()))
            print(f"  pos {p.item():3d}: weight={w.item():.6f}, V→ALU_LO contributions: {v_vals}")

        # Compute weighted sum (attention output for head 0)
        attn_out = (attn_weights.unsqueeze(1) * v0).sum(dim=0)  # [64]

        print(f"\nHead 0 attention output (V slots 1-16 → ALU_LO via W_o):")
        for slot in range(1, 17):
            if abs(attn_out[slot]) > 0.01:
                print(f"  V[{slot:2d}] = {attn_out[slot].item():.4f}")

        # Apply W_o to get ALU_LO contribution
        print(f"\nProjection W_o: V slots → ALU_LO")
        # W_o[ALU_LO + k, head0_base + slot] maps V[slot] → ALU_LO[k]
        head0_base = 0 * head_dim  # = 0
        for k in range(16):
            contrib = 0
            for slot in range(64):
                contrib += attn7.W_o[BD.ALU_LO + k, head0_base + slot].item() * attn_out[slot].item()
            if abs(contrib) > 0.01:
                print(f"  ALU_LO[{k:2d}] += {contrib:.4f}")

if __name__ == "__main__":
    main()
