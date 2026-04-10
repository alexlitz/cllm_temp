#!/usr/bin/env python3
"""Check FETCH_LO at PC marker for LEA 16."""
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

token_ids = torch.tensor([ctx], dtype=torch.long)

with torch.no_grad():
    x = model.embed(token_ids)
    for i in range(10):
        x = model.blocks[i](x)

    print(f"At PC marker (pos {pc_marker_pos}):")
    print(f"  MARK_AX = {x[0, pc_marker_pos, BD.MARK_AX].item():.4f}")
    print(f"  OP_AND = {x[0, pc_marker_pos, BD.OP_AND].item():.4f}")

    print(f"\n  FETCH_LO values (non-zero):")
    for k in range(16):
        v = x[0, pc_marker_pos, BD.FETCH_LO + k].item()
        if abs(v) > 0.5:
            print(f"    FETCH_LO[{k}] = {v:.4f}")

    print(f"\n  ALU_LO values (non-zero):")
    for k in range(16):
        v = x[0, pc_marker_pos, BD.ALU_LO + k].item()
        if abs(v) > 0.5:
            print(f"    ALU_LO[{k}] = {v:.4f}")

    # Check threshold with FETCH_LO
    fetch_lo = [x[0, pc_marker_pos, BD.FETCH_LO + k].item() for k in range(16)]
    alu_lo = [x[0, pc_marker_pos, BD.ALU_LO + k].item() for k in range(16)]
    mark_ax = x[0, pc_marker_pos, BD.MARK_AX].item()
    op_and = x[0, pc_marker_pos, BD.OP_AND].item()

    print(f"\n  If bitwise ops use FETCH_LO:")
    max_up = 0
    for a in range(16):
        for b in range(16):
            up = mark_ax * 10 + alu_lo[a] + fetch_lo[b] - 10.5
            if up > max_up:
                max_up = up
    print(f"  Max up value: {max_up:.2f}")
    if max_up > 0:
        print(f"  Some bitwise units would fire (bad!)")
    else:
        print(f"  No bitwise units fire (good)")
