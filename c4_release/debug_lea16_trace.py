#!/usr/bin/env python3
"""Trace OUTPUT_LO at PC marker for LEA 16."""
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

ctx = context + step1_tokens[:1]  # Just PC marker
pc_marker_pos = len(ctx) - 1

print(f"PC marker pos: {pc_marker_pos}")
print(f"Expected PC byte 0: {step1_tokens[1]} (lo nibble = {step1_tokens[1] & 0xF})")
print(f"Immediate: 16 (lo nibble = 0, hi nibble = 1)")

token_ids = torch.tensor([ctx], dtype=torch.long)

with torch.no_grad():
    x = model.embed(token_ids)

    print(f"\nTracing OUTPUT_LO through layers at PC marker:")
    for i in range(16):
        x_before = x.clone()
        x = model.blocks[i](x)

        delta_lo = x[0, pc_marker_pos, BD.OUTPUT_LO:BD.OUTPUT_LO+16] - x_before[0, pc_marker_pos, BD.OUTPUT_LO:BD.OUTPUT_LO+16]
        delta_hi = x[0, pc_marker_pos, BD.OUTPUT_HI:BD.OUTPUT_HI+16] - x_before[0, pc_marker_pos, BD.OUTPUT_HI:BD.OUTPUT_HI+16]

        active_lo = [(k, d.item()) for k, d in enumerate(delta_lo) if abs(d.item()) > 0.5]
        active_hi = [(k, d.item()) for k, d in enumerate(delta_hi) if abs(d.item()) > 0.5]

        if active_lo or active_hi:
            print(f"blocks[{i:2d}]:")
            if active_lo:
                active_lo.sort(key=lambda x: -abs(x[1]))
                print(f"  OUTPUT_LO deltas: {active_lo[:3]}")
            if active_hi:
                active_hi.sort(key=lambda x: -abs(x[1]))
                print(f"  OUTPUT_HI deltas: {active_hi[:3]}")

    # Check what's in FETCH at PC marker
    print(f"\nFETCH_LO at PC marker:")
    for k in range(16):
        v = x[0, pc_marker_pos, BD.FETCH_LO + k].item()
        if abs(v) > 0.5:
            print(f"  FETCH_LO[{k}] = {v:.4f}")

    print(f"\nFETCH_HI at PC marker:")
    for k in range(16):
        v = x[0, pc_marker_pos, BD.FETCH_HI + k].item()
        if abs(v) > 0.5:
            print(f"  FETCH_HI[{k}] = {v:.4f}")
