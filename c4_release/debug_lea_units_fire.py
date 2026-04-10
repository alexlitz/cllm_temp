#!/usr/bin/env python3
"""Debug why LEA FFN units fire at PC marker despite MARK_AX=0."""
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

ctx = context + step1_tokens[:1]  # Just PC marker
pc_marker_pos = len(ctx) - 1

token_ids = torch.tensor([ctx], dtype=torch.long)

with torch.no_grad():
    x = model.embed(token_ids)
    for i in range(8):
        x = model.blocks[i](x)

    # Get values at PC marker
    mark_ax = x[0, pc_marker_pos, BD.MARK_AX].item()
    alu_lo_0 = x[0, pc_marker_pos, BD.ALU_LO + 0].item()
    fetch_lo_8 = x[0, pc_marker_pos, BD.FETCH_LO + 8].item()
    op_lea = x[0, pc_marker_pos, BD.OP_LEA].item()

    print(f"At PC marker (pos {pc_marker_pos}):")
    print(f"  MARK_AX = {mark_ax:.4f}")
    print(f"  ALU_LO[0] = {alu_lo_0:.4f}")
    print(f"  FETCH_LO[8] = {fetch_lo_8:.4f}")
    print(f"  OP_LEA = {op_lea:.4f}")

    # Calculate activation for LEA unit (a=0, b=8)
    # W_up: MARK_AX*S + ALU_LO[0]*S + FETCH_LO[8]*S - 15.5*S
    up_activation = mark_ax * S + alu_lo_0 * S + fetch_lo_8 * S - 15.5 * S
    silu_val = torch.nn.functional.silu(torch.tensor(up_activation)).item()

    print(f"\nLEA unit (a=0, b=8) → result=8:")
    print(f"  up_activation = {mark_ax:.2f}*S + {alu_lo_0:.2f}*S + {fetch_lo_8:.2f}*S - 15.5*S")
    print(f"               = ({mark_ax + alu_lo_0 + fetch_lo_8:.2f} - 15.5)*S")
    print(f"               = {up_activation:.2f}")
    print(f"  silu(up) = {silu_val:.4f}")
    print(f"  gate = OP_LEA = {op_lea:.4f}")
    print(f"  contribution = silu(up) * gate = {silu_val * op_lea:.4f}")

    # For comparison, LEA unit (a=10, b=8) → result = 18 % 16 = 2
    alu_lo_10 = x[0, pc_marker_pos, BD.ALU_LO + 10].item()
    up_activation_10 = mark_ax * S + alu_lo_10 * S + fetch_lo_8 * S - 15.5 * S
    silu_val_10 = torch.nn.functional.silu(torch.tensor(up_activation_10)).item()

    print(f"\nLEA unit (a=10, b=8) → result=2:")
    print(f"  ALU_LO[10] = {alu_lo_10:.4f}")
    print(f"  up_activation = ({mark_ax + alu_lo_10 + fetch_lo_8:.2f} - 15.5)*S = {up_activation_10:.2f}")
    print(f"  silu(up) = {silu_val_10:.4f}")
    print(f"  contribution = silu(up) * gate = {silu_val_10 * op_lea:.4f}")

    print(f"\nThe problem: MARK_AX=0 but units still fire because")
    print(f"ALU_LO + FETCH_LO = {alu_lo_0 + fetch_lo_8:.2f} > 15.5 threshold")
    print(f"\nFix needed: Increase MARK_AX weight so MARK_AX=1 is required to fire")
