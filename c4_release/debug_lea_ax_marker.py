#!/usr/bin/env python3
"""Check values at AX marker for LEA to ensure fix doesn't break it."""
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

S = 100  # Scale factor

context = build_context(BYTECODE)
draft = DraftVM(BYTECODE)
draft.step()
step1_tokens = draft.draft_tokens()

# Include up to AX marker
# step1_tokens: [257=PC_MARK, 10, 0, 0, 0, 258=AX_MARK, 8, 0, 1, 0, ...]
ctx = context + step1_tokens[:6]  # Include PC marker + 4 PC bytes + AX marker
ax_marker_pos = len(ctx) - 1

print(f"Context (last 10): {ctx[-10:]}")
print(f"AX marker pos: {ax_marker_pos}, token = {ctx[ax_marker_pos]}")

token_ids = torch.tensor([ctx], dtype=torch.long)

with torch.no_grad():
    x = model.embed(token_ids)
    for i in range(8):
        x = model.blocks[i](x)

    # Get values at AX marker
    mark_ax = x[0, ax_marker_pos, BD.MARK_AX].item()
    op_lea = x[0, ax_marker_pos, BD.OP_LEA].item()

    print(f"\nAt AX marker (pos {ax_marker_pos}):")
    print(f"  MARK_AX = {mark_ax:.4f}")
    print(f"  OP_LEA = {op_lea:.4f}")

    print(f"\n  ALU_LO values (need BP lo nibble = 0):")
    for k in range(16):
        v = x[0, ax_marker_pos, BD.ALU_LO + k].item()
        if abs(v) > 0.5:
            print(f"    ALU_LO[{k}] = {v:.4f}")

    print(f"\n  FETCH_LO values (need immediate lo = 8):")
    for k in range(16):
        v = x[0, ax_marker_pos, BD.FETCH_LO + k].item()
        if abs(v) > 0.5:
            print(f"    FETCH_LO[{k}] = {v:.4f}")

    # Calculate activation for LEA unit (a=0, b=8) at AX marker
    alu_lo_0 = x[0, ax_marker_pos, BD.ALU_LO + 0].item()
    fetch_lo_8 = x[0, ax_marker_pos, BD.FETCH_LO + 8].item()

    # With proposed fix: MARK_AX*60S + ALU*S + FETCH*S - 60.5S
    up_activation_new = mark_ax * 60 * S + alu_lo_0 * S + fetch_lo_8 * S - 60.5 * S
    silu_val = torch.nn.functional.silu(torch.tensor(up_activation_new)).item()

    print(f"\nWith proposed fix (MARK_AX weight = 60S, threshold = -60.5S):")
    print(f"  up_activation = {mark_ax:.2f}*60S + {alu_lo_0:.2f}*S + {fetch_lo_8:.2f}*S - 60.5*S")
    print(f"               = ({mark_ax * 60 + alu_lo_0 + fetch_lo_8:.2f} - 60.5)*S")
    print(f"               = {up_activation_new:.2f}")
    print(f"  silu(up) = {silu_val:.4f}")
    print(f"  gate * silu(up) = {silu_val * op_lea:.4f}")
    print(f"\n  At AX marker (MARK_AX=1): fires = {up_activation_new > 0}")
