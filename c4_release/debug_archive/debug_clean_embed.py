#!/usr/bin/env python3
"""Debug CLEAN_EMBED_LO values at STACK0_BYTE0 positions."""
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
        tokens = draft.draft_tokens()
        context.extend(tokens)

    draft.step()
    step4_tokens = draft.draft_tokens()
    context_for_pred = context + step4_tokens[:6]
    ax_marker_pos = len(context_for_pred) - 1

    token_ids = torch.tensor([context_for_pred], dtype=torch.long)

    with torch.no_grad():
        x = model.embed(token_ids)
        # Run through L1-L6 only (before L7 which writes ALU)
        for i in range(7):
            x = model.blocks[i](x)

    print("STACK0_BYTE0 positions and their FULL CLEAN_EMBED_LO values:")
    print("=" * 80)

    stack0_positions = [65, 100, 135]  # From previous trace
    for pos in stack0_positions:
        token_val = token_ids[0, pos].item()
        print(f"\npos {pos} (token={token_val}):")
        print(f"  STACK0_BYTE0 = {x[0, pos, BD.STACK0_BYTE0].item():.4f}")
        print("  CLEAN_EMBED_LO:")
        for k in range(16):
            v = x[0, pos, BD.CLEAN_EMBED_LO + k].item()
            marker = " <<<<" if abs(v) > 0.1 else ""
            print(f"    [{k:2d}] = {v:8.4f}{marker}")

    print("\n" + "=" * 80)
    print("Also check AX marker CLEAN_EMBED_LO (what if it's being read as source?):")
    print(f"\npos {ax_marker_pos} (AX marker):")
    for k in range(16):
        v = x[0, ax_marker_pos, BD.CLEAN_EMBED_LO + k].item()
        marker = " <<<<" if abs(v) > 0.1 else ""
        print(f"  [{k:2d}] = {v:8.4f}{marker}")

if __name__ == "__main__":
    main()
