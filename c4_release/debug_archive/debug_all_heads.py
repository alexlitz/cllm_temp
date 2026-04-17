#!/usr/bin/env python3
"""Debug ALL L7 heads that write to ALU_LO."""
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
    print(f"Debugging L7 (blocks[7]) ALL heads at AX marker (pos {ax_marker_pos})")
    print("=" * 80)

    with torch.no_grad():
        x = model.embed(token_ids)

        # Run through L1-L6
        for i in range(7):
            x = model.blocks[i](x)

        attn7 = model.blocks[7].attn

        # Compute Q, K, V
        Q = x @ attn7.W_q.T
        K = x @ attn7.W_k.T
        V = x @ attn7.W_v.T

        seq_len = x.shape[1]
        num_heads = 8
        head_dim = x.shape[2] // num_heads

        Q = Q.view(1, seq_len, num_heads, head_dim).transpose(1, 2)
        K = K.view(1, seq_len, num_heads, head_dim).transpose(1, 2)
        V = V.view(1, seq_len, num_heads, head_dim).transpose(1, 2)

        # Accumulate ALU_LO contributions from all heads
        alu_lo_total = torch.zeros(16)

        for head_idx in range(num_heads):
            q = Q[0, head_idx, ax_marker_pos, :]
            k = K[0, head_idx, :, :]
            v = V[0, head_idx, :, :]

            # Attention scores
            scores = (k @ q) / (head_dim ** 0.5)

            # ALiBi
            alibi_slope = attn7.alibi_slopes[head_idx].item() if attn7.alibi_slopes is not None else 0.5
            for pos in range(seq_len):
                dist = ax_marker_pos - pos
                if dist > 0:
                    scores[pos] -= alibi_slope * dist

            # Softmax1
            exp_scores = torch.exp(scores - scores.max())
            attn_weights = exp_scores / (torch.exp(-scores.max()) + exp_scores.sum())

            # Attention output
            attn_out = (attn_weights.unsqueeze(1) * v).sum(dim=0)

            # Check if this head writes to ALU_LO
            head_base = head_idx * head_dim
            alu_contrib = torch.zeros(16)
            for k_idx in range(16):
                for slot in range(head_dim):
                    alu_contrib[k_idx] += attn7.W_o[BD.ALU_LO + k_idx, head_base + slot].item() * attn_out[slot].item()

            if alu_contrib.abs().max() > 0.01:
                print(f"\nHead {head_idx} writes to ALU_LO:")
                for k_idx in range(16):
                    if abs(alu_contrib[k_idx]) > 0.01:
                        print(f"  ALU_LO[{k_idx:2d}] += {alu_contrib[k_idx].item():.4f}")
                alu_lo_total += alu_contrib

        print("\n" + "=" * 80)
        print("Total ALU_LO from ALL L7 heads:")
        for k_idx in range(16):
            if abs(alu_lo_total[k_idx]) > 0.01:
                print(f"  ALU_LO[{k_idx:2d}] = {alu_lo_total[k_idx].item():.4f}")

if __name__ == "__main__":
    main()
