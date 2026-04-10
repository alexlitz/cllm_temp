#!/usr/bin/env python3
"""Check ALU_LO and AX_CARRY_LO at AX marker for a proper AND operation."""
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

# Include up to AX marker (6 tokens: PC marker + 4 PC bytes + AX marker)
ctx = context + step1_tokens[:6]
ax_marker_pos = len(ctx) - 1
pc_marker_pos = ax_marker_pos - 5

print(f"Context ends with: {ctx[-10:]}")
print(f"PC marker pos: {pc_marker_pos}, AX marker pos: {ax_marker_pos}")
print(f"Draft: AX={draft.ax} (0x{draft.ax:08X})")
print(f"Expected result: AX & 5 = 0 & 5 = 0")

token_ids = torch.tensor([ctx], dtype=torch.long)

with torch.no_grad():
    x = model.embed(token_ids)
    for i in range(10):
        x = model.blocks[i](x)

    print(f"\nAt PC marker (pos {pc_marker_pos}):")
    print(f"  MARK_AX = {x[0, pc_marker_pos, BD.MARK_AX].item():.4f}")
    print(f"  OP_AND = {x[0, pc_marker_pos, BD.OP_AND].item():.4f}")

    print(f"  ALU_LO values (non-zero):")
    for k in range(16):
        v = x[0, pc_marker_pos, BD.ALU_LO + k].item()
        if abs(v) > 0.5:
            print(f"    ALU_LO[{k}] = {v:.4f}")

    print(f"  AX_CARRY_LO values (non-zero):")
    for k in range(16):
        v = x[0, pc_marker_pos, BD.AX_CARRY_LO + k].item()
        if abs(v) > 0.5:
            print(f"    AX_CARRY_LO[{k}] = {v:.4f}")

    print(f"\nAt AX marker (pos {ax_marker_pos}):")
    mark_ax = x[0, ax_marker_pos, BD.MARK_AX].item()
    op_and = x[0, ax_marker_pos, BD.OP_AND].item()
    print(f"  MARK_AX = {mark_ax:.4f}")
    print(f"  OP_AND = {op_and:.4f}")

    print(f"  ALU_LO values (non-zero):")
    alu_lo = []
    for k in range(16):
        v = x[0, ax_marker_pos, BD.ALU_LO + k].item()
        alu_lo.append(v)
        if abs(v) > 0.5:
            print(f"    ALU_LO[{k}] = {v:.4f}")

    print(f"  AX_CARRY_LO values (non-zero):")
    ax_carry_lo = []
    for k in range(16):
        v = x[0, ax_marker_pos, BD.AX_CARRY_LO + k].item()
        ax_carry_lo.append(v)
        if abs(v) > 0.5:
            print(f"    AX_CARRY_LO[{k}] = {v:.4f}")

    # Check thresholds
    print(f"\n  Threshold analysis:")
    print(f"  For unit (a=0, b=5) which computes 0 & 5 = 0:")
    up = mark_ax * 10 + alu_lo[0] + ax_carry_lo[5] - 10.5
    print(f"    up = {mark_ax:.2f}*10 + {alu_lo[0]:.2f} + {ax_carry_lo[5]:.2f} - 10.5 = {up:.2f}")
    if up > 0:
        print(f"    Unit fires (correct)")
    else:
        print(f"    Unit does NOT fire (incorrect!)")
