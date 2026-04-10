#!/usr/bin/env python3
"""Check values at AX marker for LEA 16."""
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

S = 100

context = build_context(BYTECODE)
draft = DraftVM(BYTECODE)
draft.step()
step1_tokens = draft.draft_tokens()

# Include up to AX marker
ctx = context + step1_tokens[:6]  # PC marker + 4 PC bytes + AX marker
ax_marker_pos = len(ctx) - 1

print(f"Context ends with: {ctx[-10:]}")
print(f"AX marker pos: {ax_marker_pos}")
print(f"Draft: AX={draft.ax} (0x{draft.ax:08X}), BP={draft.bp}")
print(f"Expected AX byte 0: {draft.ax & 0xFF} (lo={draft.ax & 0xF}, hi={(draft.ax >> 4) & 0xF})")

token_ids = torch.tensor([ctx], dtype=torch.long)

with torch.no_grad():
    x = model.embed(token_ids)
    for i in range(9):
        x = model.blocks[i](x)

    mark_ax = x[0, ax_marker_pos, BD.MARK_AX].item()
    op_lea = x[0, ax_marker_pos, BD.OP_LEA].item()

    print(f"\nAt AX marker before blocks[9]:")
    print(f"  MARK_AX = {mark_ax:.4f}")
    print(f"  OP_LEA = {op_lea:.4f}")

    print(f"\n  ALU_HI values:")
    alu_hi = [x[0, ax_marker_pos, BD.ALU_HI + k].item() for k in range(16)]
    for k, v in enumerate(alu_hi):
        if abs(v) > 0.5:
            print(f"    ALU_HI[{k}] = {v:.4f}")

    print(f"\n  FETCH_HI values:")
    fetch_hi = [x[0, ax_marker_pos, BD.FETCH_HI + k].item() for k in range(16)]
    for k, v in enumerate(fetch_hi):
        if abs(v) > 0.5:
            print(f"    FETCH_HI[{k}] = {v:.4f}")

    # Check what threshold is needed
    # At PC marker: ALU_HI[wrong] + FETCH_HI[correct] ≈ 12 + 43 = 55
    # At AX marker: MARK_AX + ALU_HI[correct] + FETCH_HI[correct]

    # For LEA 16: AX = BP + 16 = 65536 + 16 = 65552 = 0x10010
    # AX byte 0 = 0x10, lo=0, hi=1
    # BP byte 0 = 0x00, lo=0, hi=0

    print(f"\nCalculating threshold needed:")
    print(f"  At PC marker (MARK_AX=0): ALU_HI[0]≈12 + FETCH_HI[1]≈43 = 55")
    print(f"  At AX marker (MARK_AX=1): {mark_ax:.2f} + ALU_HI[0]={alu_hi[0]:.2f} + FETCH_HI[1]={fetch_hi[1]:.2f} = {mark_ax + alu_hi[0] + fetch_hi[1]:.2f}")
    print(f"  Threshold must be > 55 but < {mark_ax + alu_hi[0] + fetch_hi[1]:.2f}")
    print(f"  Suggested: threshold = 58 or 60")
