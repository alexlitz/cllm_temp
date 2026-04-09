#!/usr/bin/env python3
"""Debug L7 Head 1 (LEA/ADJ) in detail."""
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
    print(f"Debugging L7 Head 1 (LEA/ADJ) at AX marker (pos {ax_marker_pos})")
    print("=" * 80)

    with torch.no_grad():
        x = model.embed(token_ids)

        # Run through L1-L6
        for i in range(7):
            x = model.blocks[i](x)

        attn7 = model.blocks[7].attn

        # Check OP_LEA and OP_ADJ at AX marker
        print(f"\nAt AX marker (pos {ax_marker_pos}):")
        print(f"  OP_LEA = {x[0, ax_marker_pos, BD.OP_LEA].item():.4f}")
        print(f"  OP_ADJ = {x[0, ax_marker_pos, BD.OP_ADJ].item():.4f}")

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

        # Focus on Head 1
        head_idx = 1
        q = Q[0, head_idx, ax_marker_pos, :]
        k = K[0, head_idx, :, :]
        v = V[0, head_idx, :, :]

        print(f"\nHead 1 Q vector at AX marker (first 10 dims):")
        for d in range(10):
            print(f"  Q[{d}] = {q[d].item():.4f}")

        # Q should be ~0 if OP_LEA and OP_ADJ are 0
        print(f"\n  Q norm = {q.norm().item():.4f} (should be near 0 for MUL)")

        # K at MARK_BP and MARK_SP positions
        print(f"\nK values at BP/SP markers:")
        for pos in range(seq_len):
            mark_bp = x[0, pos, BD.MARK_BP].item()
            mark_sp = x[0, pos, BD.MARK_SP].item()
            if mark_bp > 0.5 or mark_sp > 0.5:
                k_norm = k[pos].norm().item()
                print(f"  pos {pos}: MARK_BP={mark_bp:.2f}, MARK_SP={mark_sp:.2f}, K_norm={k_norm:.4f}")

        # Attention scores
        scores = (k @ q) / (head_dim ** 0.5)

        # ALiBi
        alibi_slope = attn7.alibi_slopes[head_idx].item() if attn7.alibi_slopes is not None else 0.5
        print(f"\nHead 1 ALiBi slope: {alibi_slope}")

        for pos in range(seq_len):
            dist = ax_marker_pos - pos
            if dist > 0:
                scores[pos] -= alibi_slope * dist

        print(f"\nTop 10 attention scores for Head 1:")
        top_scores, top_pos = torch.topk(scores, 10)
        for s, p in zip(top_scores, top_pos):
            token = token_ids[0, p].item()
            mark_bp = x[0, p, BD.MARK_BP].item()
            mark_sp = x[0, p, BD.MARK_SP].item()
            dist = ax_marker_pos - p.item()
            print(f"  pos {p.item():3d} (d={dist:3d}): score={s.item():8.4f}, " +
                  f"MARK_BP={mark_bp:.2f}, MARK_SP={mark_sp:.2f}")

        # Softmax1
        exp_scores = torch.exp(scores - scores.max())
        attn_weights = exp_scores / (torch.exp(-scores.max()) + exp_scores.sum())

        print(f"\nTop 10 attention weights:")
        top_weights, top_pos = torch.topk(attn_weights, 10)
        total_weight = attn_weights.sum().item()
        print(f"Total attention weight: {total_weight:.6f}")

        for w, p in zip(top_weights, top_pos):
            if w > 0.001:
                # Check V values at this position
                output_lo = x[0, p, BD.OUTPUT_LO:BD.OUTPUT_LO+16]
                nonzero = [(i, output_lo[i].item()) for i in range(16) if abs(output_lo[i]) > 0.1]
                print(f"  pos {p.item():3d}: weight={w.item():.6f}, OUTPUT_LO significant: {nonzero}")

        # Attention output
        attn_out = (attn_weights.unsqueeze(1) * v).sum(dim=0)

        print(f"\nHead 1 attention output V slots → ALU_LO:")
        for slot in range(1, 17):
            if abs(attn_out[slot]) > 0.01:
                print(f"  V[{slot:2d}] (→ALU_LO[{slot-1}]) = {attn_out[slot].item():.4f}")

if __name__ == "__main__":
    main()
