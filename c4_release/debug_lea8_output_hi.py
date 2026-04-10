#!/usr/bin/env python3
"""Trace OUTPUT_HI at AX marker for LEA 8."""
import torch
import sys
sys.path.insert(0, '.')

from neural_vm.vm_step import AutoregressiveVM, set_vm_weights, Token, _SetDim as BD
from neural_vm.speculative import DraftVM
from neural_vm.embedding import Opcode
from neural_vm.constants import IMMEDIATE_SIZE, PADDING_SIZE

BYTECODE = [Opcode.LEA | (8 << 8)]

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

ctx = context + step1_tokens[:6]  # Up to AX marker
ax_marker_pos = len(ctx) - 1

print(f"Expected: AX byte 0 = {step1_tokens[6]} (lo={step1_tokens[6] & 0xF}, hi={step1_tokens[6] >> 4})")

token_ids = torch.tensor([ctx], dtype=torch.long)

with torch.no_grad():
    x = model.embed(token_ids)

    print(f"\nTracing OUTPUT_HI through layers at AX marker (pos {ax_marker_pos}):")
    for i in range(16):
        x_before = x.clone()
        x = model.blocks[i](x)

        delta = x[0, ax_marker_pos, BD.OUTPUT_HI:BD.OUTPUT_HI+16] - x_before[0, ax_marker_pos, BD.OUTPUT_HI:BD.OUTPUT_HI+16]
        active_deltas = [(k, d.item()) for k, d in enumerate(delta) if abs(d.item()) > 0.5]
        if active_deltas:
            current = [(k, x[0, ax_marker_pos, BD.OUTPUT_HI + k].item()) for k in range(16)]
            current.sort(key=lambda x: -x[1])
            print(f"blocks[{i:2d}]: OUTPUT_HI deltas = {active_deltas[:5]}")
            print(f"           Top 3 OUTPUT_HI = {current[:3]}")

    print(f"\nFinal OUTPUT_HI at AX marker:")
    output_hi_vals = [(k, x[0, ax_marker_pos, BD.OUTPUT_HI + k].item()) for k in range(16)]
    output_hi_vals.sort(key=lambda x: -x[1])
    for k, v in output_hi_vals[:5]:
        print(f"  OUTPUT_HI[{k}] = {v:.4f}")

    # Check ALU_HI - what's the hi nibble of BP?
    print(f"\nALU_HI at AX marker (expected BP hi nibble = 0):")
    for k in range(16):
        v = x[0, ax_marker_pos, BD.ALU_HI + k].item()
        if abs(v) > 0.5:
            print(f"  ALU_HI[{k}] = {v:.4f}")

    # Check FETCH_HI - what's the hi nibble of immediate byte 0?
    print(f"\nFETCH_HI at AX marker (expected imm hi nibble = 0):")
    for k in range(16):
        v = x[0, ax_marker_pos, BD.FETCH_HI + k].item()
        if abs(v) > 0.5:
            print(f"  FETCH_HI[{k}] = {v:.4f}")
