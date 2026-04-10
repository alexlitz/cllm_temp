#!/usr/bin/env python3
"""Debug ADD 10+32 Step 4 prediction - why is AX byte 0 = 1 instead of 42?"""
import torch
import sys
sys.path.insert(0, '.')

from neural_vm.vm_step import AutoregressiveVM, set_vm_weights, Token, _SetDim as BD
from neural_vm.speculative import DraftVM
from neural_vm.embedding import Opcode
from neural_vm.constants import IMMEDIATE_SIZE, PADDING_SIZE

# ADD 10+32 test
BYTECODE = [Opcode.IMM | (10 << 8), Opcode.PSH, Opcode.IMM | (32 << 8), Opcode.ADD, Opcode.EXIT]

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

    # Run through steps 1-3
    for step in range(3):
        draft.step()
        context.extend(draft.draft_tokens())

    draft.step()  # Step 4 (ADD)
    step4_tokens = draft.draft_tokens()

    # Context up to AX marker (T6)
    context_for_ax = context + step4_tokens[:6]
    ax_marker_pos = len(context_for_ax) - 1

    print("=" * 80)
    print(f"DEBUG ADD 10+32 Step 4 at AX marker (pos {ax_marker_pos})")
    print(f"Expected: 10 + 32 = 42 (AX byte 0 = 42)")
    print("=" * 80)

    token_ids = torch.tensor([context_for_ax], dtype=torch.long)

    with torch.no_grad():
        x = model.embed(token_ids)

        # Run through all layers except final
        for i in range(16):
            x = model.blocks[i](x)

        # Check OUTPUT_LO at AX marker
        print(f"\nOUTPUT_LO at AX marker (expected: [10]=1 for nibble A of 42=0x2A):")
        for k in range(16):
            v = x[0, ax_marker_pos, BD.OUTPUT_LO + k].item()
            if abs(v) > 0.01:
                print(f"  OUTPUT_LO[{k:2d}]: {v:.4f}")

        # Check ALU_LO and AX_CARRY_LO (operands)
        print(f"\nALU_LO at AX marker (should have STACK0 = 10):")
        for k in range(16):
            v = x[0, ax_marker_pos, BD.ALU_LO + k].item()
            if abs(v) > 0.01:
                print(f"  ALU_LO[{k:2d}]: {v:.4f}")

        print(f"\nAX_CARRY_LO at AX marker (should have prev AX = 32):")
        for k in range(16):
            v = x[0, ax_marker_pos, BD.AX_CARRY_LO + k].item()
            if abs(v) > 0.01:
                print(f"  AX_CARRY_LO[{k:2d}]: {v:.4f}")

        # Get logits
        logits = model(token_ids)
        print(f"\nTop 5 predictions for AX byte 0:")
        top5 = torch.topk(logits[0, -1, :], 5)
        for val, idx in zip(top5.values, top5.indices):
            print(f"  token {idx.item()}: logit {val.item():.2f}")

if __name__ == "__main__":
    main()
