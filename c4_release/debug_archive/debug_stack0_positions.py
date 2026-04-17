#!/usr/bin/env python3
"""Debug STACK0_BYTE0 positions and their values."""
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
    print(f"Header length: {len(context)}")

    draft = DraftVM(BYTECODE)

    # Run 3 steps to build context
    for step in range(3):
        draft.step()
        tokens = draft.draft_tokens()
        stack0 = draft._mem_read(draft.sp) if draft.sp < 0xFFFFFF else 0
        print(f"Step {step+1}: AX={draft.ax}, STACK0={stack0}, start_pos={len(context)}")
        context.extend(tokens)

    # Step 4 - MUL execution
    draft.step()
    step4_tokens = draft.draft_tokens()
    print(f"Step 4: AX={draft.ax}, start_pos={len(context)}")

    # Build context up to AX marker
    context_for_pred = context + step4_tokens[:6]
    ax_marker_pos = len(context_for_pred) - 1
    print(f"\nAX marker position: {ax_marker_pos}")

    token_ids = torch.tensor([context_for_pred], dtype=torch.long)

    # Run through first 7 layers to compute STACK0_BYTE0 flags
    with torch.no_grad():
        x = model.embed(token_ids)
        for i in range(7):  # L1-L7
            x = model.blocks[i](x)

    print("\n" + "=" * 60)
    print("Positions with STACK0_BYTE0 flag > 0.5:")
    for pos in range(token_ids.shape[1]):
        flag = x[0, pos, BD.STACK0_BYTE0].item()
        if flag > 0.5:
            clean_lo = []
            for k in range(16):
                v = x[0, pos, BD.CLEAN_EMBED_LO + k].item()
                if abs(v) > 0.3:
                    clean_lo.append((k, v))
            token_val = token_ids[0, pos].item()
            print(f"  pos {pos}: STACK0_BYTE0={flag:.2f}, token={token_val}, CLEAN_EMBED_LO={clean_lo}")

    print("\n" + "=" * 60)
    print("What values are at MARK_AX positions:")
    for pos in range(token_ids.shape[1]):
        mark_ax = x[0, pos, BD.MARK_AX].item()
        if mark_ax > 0.5:
            token_val = token_ids[0, pos].item()
            print(f"  pos {pos}: MARK_AX={mark_ax:.2f}, token={token_val}")

    print("\n" + "=" * 60)
    print("L7 attention head 0 scores for AX marker:")
    # Manually compute attention for head 0 of L7
    attn7 = model.blocks[7].attn
    q_weight = attn7.W_q.data
    k_weight = attn7.W_k.data

    HD = 64
    base = 0 * HD  # Head 0

    # Q at AX marker (pos 154)
    q = torch.zeros(HD)
    q[0] = x[0, ax_marker_pos, BD.MARK_AX].item() * 15.0  # L in the code

    # K at each position
    print("Top K positions (STACK0_BYTE0):")
    scores = []
    for pos in range(token_ids.shape[1]):
        k_score = x[0, pos, BD.STACK0_BYTE0].item() * 15.0
        # Simple score = q[0]*k[0]
        score = q[0] * k_score / 64.0  # divide by sqrt(HD)
        if k_score > 5.0:
            print(f"  pos {pos}: K_score={k_score:.2f}, score={score:.2f}")

if __name__ == "__main__":
    main()
