#!/usr/bin/env python3
"""Verify LEA 16 prediction after L10 bitwise ops fix."""
import torch
import sys
sys.path.insert(0, '.')

from neural_vm.vm_step import AutoregressiveVM, set_vm_weights, Token, _SetDim as BD
from neural_vm.speculative import DraftVM
from neural_vm.embedding import Opcode
from neural_vm.constants import IMMEDIATE_SIZE, PADDING_SIZE

BYTECODE = [Opcode.LEA | (16 << 8)]

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

print(f"LEA 16: BP={draft.bp}, AX=BP+16={draft.ax}")
print(f"Expected PC byte 0: {(draft.pc & 0xFF)} (lo={draft.pc & 0xF}, hi={(draft.pc >> 4) & 0xF})")

ctx = context + step1_tokens[:1]  # Just PC marker
pc_marker_pos = len(ctx) - 1

token_ids = torch.tensor([ctx], dtype=torch.long)

with torch.no_grad():
    x = model.embed(token_ids)
    for i, block in enumerate(model.blocks):
        x = block(x)

    # Get final OUTPUT_LO/HI
    print(f"\nAt PC marker (pos {pc_marker_pos}) after all layers:")
    output_lo = [x[0, pc_marker_pos, BD.OUTPUT_LO + k].item() for k in range(16)]
    output_hi = [x[0, pc_marker_pos, BD.OUTPUT_HI + k].item() for k in range(16)]

    # Find argmax
    lo_pred = max(range(16), key=lambda k: output_lo[k])
    hi_pred = max(range(16), key=lambda k: output_hi[k])
    byte_pred = lo_pred + (hi_pred << 4)

    expected_lo = draft.pc & 0xF
    expected_hi = (draft.pc >> 4) & 0xF
    expected_byte = draft.pc & 0xFF

    print(f"  OUTPUT_LO argmax: {lo_pred} (expected {expected_lo})")
    print(f"  OUTPUT_HI argmax: {hi_pred} (expected {expected_hi})")
    print(f"  Predicted byte: {byte_pred} (expected {expected_byte})")

    # Show values
    print(f"\n  OUTPUT_LO[{expected_lo}] = {output_lo[expected_lo]:.4f}")
    print(f"  OUTPUT_LO[{lo_pred}] = {output_lo[lo_pred]:.4f}")
    print(f"  OUTPUT_HI[{expected_hi}] = {output_hi[expected_hi]:.4f}")
    print(f"  OUTPUT_HI[{hi_pred}] = {output_hi[hi_pred]:.4f}")

    if byte_pred == expected_byte:
        print(f"\n  PASS: LEA 16 PC byte 0 is correct!")
    else:
        print(f"\n  FAIL: LEA 16 PC byte 0 is wrong!")
