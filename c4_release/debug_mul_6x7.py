#!/usr/bin/env python3
"""Debug MUL 6*7 - trace ALU and AX_CARRY at AX marker."""
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

    # Run 3 steps to build context
    print("Building context:")
    for step in range(3):
        draft.step()
        tokens = draft.draft_tokens()
        stack0 = draft._mem_read(draft.sp) if draft.sp < 0xFFFFFF else 0
        print(f"  Step {step+1}: AX={draft.ax}, SP={draft.sp:#x}, STACK0={stack0}")
        context.extend(tokens)

    # Step 4 - MUL execution
    draft.step()
    step4_tokens = draft.draft_tokens()
    print(f"  Step 4: AX={draft.ax} (should be 6*7=42)")

    # Build context up to AX marker
    context_for_pred = context + step4_tokens[:6]  # PC section + AX marker
    ax_marker_pos = len(context_for_pred) - 1

    token_ids = torch.tensor([context_for_pred], dtype=torch.long)

    print(f"\nExpected at AX marker (pos {ax_marker_pos}):")
    print(f"  ALU should have STACK0 value = 6 (a_lo=6, a_hi=0)")
    print(f"  AX_CARRY should have prev AX value = 7 (b_lo=7, b_hi=0)")
    print("=" * 60)

    with torch.no_grad():
        x = model.embed(token_ids)
        for i, block in enumerate(model.blocks):
            x = block(x)

    # Check ALU values
    print(f"\nALU at AX marker:")
    print(f"  ALU_LO (should be one-hot at 6):")
    for k in range(16):
        val = x[0, ax_marker_pos, BD.ALU_LO + k].item()
        if abs(val) > 0.1:
            print(f"    ALU_LO[{k}] = {val:.2f}")

    print(f"  ALU_HI (should be one-hot at 0):")
    for k in range(16):
        val = x[0, ax_marker_pos, BD.ALU_HI + k].item()
        if abs(val) > 0.1:
            print(f"    ALU_HI[{k}] = {val:.2f}")

    # Check AX_CARRY values
    print(f"\nAX_CARRY at AX marker:")
    print(f"  AX_CARRY_LO (should be one-hot at 7):")
    for k in range(16):
        val = x[0, ax_marker_pos, BD.AX_CARRY_LO + k].item()
        if abs(val) > 0.1:
            print(f"    AX_CARRY_LO[{k}] = {val:.2f}")

    print(f"  AX_CARRY_HI (should be one-hot at 0):")
    for k in range(16):
        val = x[0, ax_marker_pos, BD.AX_CARRY_HI + k].item()
        if abs(val) > 0.1:
            print(f"    AX_CARRY_HI[{k}] = {val:.2f}")

    # What's the expected MUL result?
    # a = ALU, b = AX_CARRY
    # For correct operation: a=6, b=7, result=42
    # result_lo = (6 * 7) % 16 = 42 % 16 = 10
    # result_hi = (42 // 16) = 2
    print(f"\nExpected MUL result: 6 * 7 = 42")
    print(f"  result_lo = 10 (0xA), result_hi = 2")
    print(f"  OUTPUT_LO[10] should be ~1, OUTPUT_HI[2] should be ~1")

if __name__ == "__main__":
    main()
