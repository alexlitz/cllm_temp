#!/usr/bin/env python3
"""Verify AND 5 prediction with new L10 bitwise ops fix."""
import torch
import sys
sys.path.insert(0, '.')

from neural_vm.vm_step import AutoregressiveVM, set_vm_weights, Token, _SetDim as BD
from neural_vm.speculative import DraftVM
from neural_vm.embedding import Opcode
from neural_vm.constants import IMMEDIATE_SIZE, PADDING_SIZE

# AND 5: AX = AX & 5 = 0 & 5 = 0
BYTECODE = [Opcode.AND | (5 << 8)]

def build_context(bytecode, data=b""):
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

model = AutoregressiveVM()
set_vm_weights(model)
model.eval()

context = build_context(BYTECODE)
draft = DraftVM(BYTECODE)
draft.step()
step1_tokens = draft.draft_tokens()

print(f"AND 5: AX=0 & 5 = {draft.ax}")
print(f"Expected AX byte 0: {draft.ax & 0xFF}")

# Include up to AX marker (6 tokens: PC marker + 4 PC bytes + AX marker)
ctx = context + step1_tokens[:6]
ax_marker_pos = len(ctx) - 1

token_ids = torch.tensor([ctx], dtype=torch.long)

with torch.no_grad():
    x = model.embed(token_ids)
    for i, block in enumerate(model.blocks):
        x = block(x)

    # Get final OUTPUT_LO/HI
    print(f"\nAt AX marker (pos {ax_marker_pos}) after all layers:")
    output_lo = [x[0, ax_marker_pos, BD.OUTPUT_LO + k].item() for k in range(16)]
    output_hi = [x[0, ax_marker_pos, BD.OUTPUT_HI + k].item() for k in range(16)]

    # Find argmax
    lo_pred = max(range(16), key=lambda k: output_lo[k])
    hi_pred = max(range(16), key=lambda k: output_hi[k])
    byte_pred = lo_pred + (hi_pred << 4)

    expected_lo = draft.ax & 0xF
    expected_hi = (draft.ax >> 4) & 0xF
    expected_byte = draft.ax & 0xFF

    print(f"  OUTPUT_LO argmax: {lo_pred} (expected {expected_lo})")
    print(f"  OUTPUT_HI argmax: {hi_pred} (expected {expected_hi})")
    print(f"  Predicted byte: {byte_pred} (expected {expected_byte})")

    # Show values
    print(f"\n  OUTPUT_LO[{expected_lo}] = {output_lo[expected_lo]:.4f}")
    print(f"  OUTPUT_HI[{expected_hi}] = {output_hi[expected_hi]:.4f}")

    if byte_pred == expected_byte:
        print(f"\n  PASS: AND 5 is correct!")
    else:
        print(f"\n  FAIL: AND 5 is wrong!")

        # Debug: check L10 bitwise unit activation
        print(f"\n  Debugging L10 bitwise unit:")
        # Reload and check values at L10
        x = model.embed(token_ids)
        for i in range(10):
            x = model.blocks[i](x)

        mark_ax = x[0, ax_marker_pos, BD.MARK_AX].item()
        alu_lo_0 = x[0, ax_marker_pos, BD.ALU_LO + 0].item()
        fetch_lo_5 = x[0, ax_marker_pos, BD.FETCH_LO + 5].item()

        print(f"  MARK_AX = {mark_ax:.4f}")
        print(f"  ALU_LO[0] = {alu_lo_0:.4f}")
        print(f"  FETCH_LO[5] = {fetch_lo_5:.4f}")

        # New formula: MARK_AX*60 + ALU_LO[0] + FETCH_LO[5] - 80
        up = mark_ax * 60 + alu_lo_0 + fetch_lo_5 - 80
        print(f"  up = {mark_ax:.2f}*60 + {alu_lo_0:.2f} + {fetch_lo_5:.2f} - 80 = {up:.2f}")
        if up > 0:
            print(f"  Unit would fire")
        else:
            print(f"  Unit does NOT fire")
