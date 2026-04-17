#!/usr/bin/env python3
"""Trace LEA byte 1 prediction to understand the architecture gap."""
import torch
import sys
sys.path.insert(0, '.')

from neural_vm.vm_step import AutoregressiveVM, set_vm_weights, Token, _SetDim as BD
from neural_vm.speculative import DraftVM
from neural_vm.embedding import Opcode
from neural_vm.constants import IMMEDIATE_SIZE, PADDING_SIZE

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

# LEA 256: AX = BP + 256 = 0x10000 + 0x100 = 0x10100
# Byte 0: 0x00, Byte 1: 0x01, Byte 2: 0x01
bytecode = [Opcode.LEA | (256 << 8)]
context = build_context(bytecode)
draft = DraftVM(bytecode)
draft.step()
step1_tokens = draft.draft_tokens()

print(f"LEA 256: AX = {draft.ax} (0x{draft.ax:08X})")
print(f"  BP = {draft.bp} (0x{draft.bp:08X})")
print(f"  Imm = 256 (0x00000100)")
print(f"  Expected AX bytes: [{draft.ax & 0xFF}, {(draft.ax >> 8) & 0xFF}, {(draft.ax >> 16) & 0xFF}, {(draft.ax >> 24) & 0xFF}]")

# Context up to AX byte 1 position
# PC marker (1) + PC bytes (4) + AX marker (1) + AX byte 0 (1) = 7 tokens
ctx = context + step1_tokens[:7]
ax_byte1_pos = len(ctx) - 1

print(f"\nContext tokens: {ctx[-10:]}")
print(f"AX byte 1 position: {ax_byte1_pos}")

token_ids = torch.tensor([ctx], dtype=torch.long)

with torch.no_grad():
    # Run through layers and check key dimensions
    x = model.embed(token_ids)

    print(f"\nAfter embedding at AX byte 1 (pos {ax_byte1_pos}):")
    print(f"  OP_LEA = {x[0, ax_byte1_pos, BD.OP_LEA].item():.4f}")
    print(f"  MARK_AX = {x[0, ax_byte1_pos, BD.MARK_AX].item():.4f}")

    # After L8 attention - check if ALU is populated
    for i in range(8):
        x = model.blocks[i](x)

    print(f"\nAfter L8 at AX byte 1 (pos {ax_byte1_pos}):")
    print(f"  ALU_LO values:")
    for k in range(16):
        v = x[0, ax_byte1_pos, BD.ALU_LO + k].item()
        if abs(v) > 0.5:
            print(f"    ALU_LO[{k}] = {v:.4f}")

    print(f"  FETCH_LO values:")
    for k in range(16):
        v = x[0, ax_byte1_pos, BD.FETCH_LO + k].item()
        if abs(v) > 0.5:
            print(f"    FETCH_LO[{k}] = {v:.4f}")

    print(f"  OUTPUT_LO values:")
    for k in range(16):
        v = x[0, ax_byte1_pos, BD.OUTPUT_LO + k].item()
        if abs(v) > 0.5:
            print(f"    OUTPUT_LO[{k}] = {v:.4f}")

    # Run remaining layers
    for i in range(8, len(model.blocks)):
        x = model.blocks[i](x)

    print(f"\nAfter all layers at AX byte 1 (pos {ax_byte1_pos}):")
    output_lo = [x[0, ax_byte1_pos, BD.OUTPUT_LO + k].item() for k in range(16)]
    output_hi = [x[0, ax_byte1_pos, BD.OUTPUT_HI + k].item() for k in range(16)]

    lo_pred = max(range(16), key=lambda k: output_lo[k])
    hi_pred = max(range(16), key=lambda k: output_hi[k])
    byte_pred = lo_pred + (hi_pred << 4)

    expected_byte = (draft.ax >> 8) & 0xFF
    print(f"  Predicted: {byte_pred} (lo={lo_pred}, hi={hi_pred})")
    print(f"  Expected: {expected_byte} (lo={expected_byte & 0xF}, hi={(expected_byte >> 4) & 0xF})")

    # Check what's at BP byte 1 position in prev step
    print(f"\n  Checking BP byte 1 position in prev step...")
    # In prev step, BP is at position 15+1=16 (BP marker + 1 byte)
    # In current step (step 1), that's at position 35 - 35 = 0? No, we need to find it.
    # Actually, prev step tokens are embedded as history in context.
