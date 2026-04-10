#!/usr/bin/env python3
"""Debug ALU_LO values at PC marker for LEA."""
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

ctx = context + step1_tokens[:1]  # Just PC marker
pc_marker_pos = len(ctx) - 1

print(f"Context: {ctx}")
print(f"PC marker pos: {pc_marker_pos}")
print(f"Expected PC: {draft.pc} (lo nibble = {draft.pc & 0xF})")
print(f"Immediate: 8")

token_ids = torch.tensor([ctx], dtype=torch.long)

with torch.no_grad():
    x = model.embed(token_ids)

    # Run through blocks[0..7] to get to L8
    for i in range(8):
        x = model.blocks[i](x)

    print(f"\nBefore blocks[8] FFN, at PC marker:")
    print(f"  MARK_AX: {x[0, pc_marker_pos, BD.MARK_AX].item():.4f}")
    print(f"  OP_LEA: {x[0, pc_marker_pos, BD.OP_LEA].item():.4f}")

    print(f"\n  ALU_LO values:")
    for k in range(16):
        v = x[0, pc_marker_pos, BD.ALU_LO + k].item()
        if abs(v) > 0.5:
            print(f"    ALU_LO[{k}] = {v:.4f}")

    print(f"\n  ALU_HI values:")
    for k in range(16):
        v = x[0, pc_marker_pos, BD.ALU_HI + k].item()
        if abs(v) > 0.5:
            print(f"    ALU_HI[{k}] = {v:.4f}")

    print(f"\n  FETCH_LO values:")
    for k in range(16):
        v = x[0, pc_marker_pos, BD.FETCH_LO + k].item()
        if abs(v) > 0.5:
            print(f"    FETCH_LO[{k}] = {v:.4f}")

    # Also check PC_LO values - does this exist?
    if hasattr(BD, 'PC_LO'):
        print(f"\n  PC_LO values:")
        for k in range(16):
            v = x[0, pc_marker_pos, BD.PC_LO + k].item()
            if abs(v) > 0.5:
                print(f"    PC_LO[{k}] = {v:.4f}")
    else:
        print(f"\n  No PC_LO dimension found")

    # Check H1 values (passthrough from prior step)
    print(f"\n  H1 values (passthrough registers):")
    for k in range(4):
        v = x[0, pc_marker_pos, BD.H1 + k].item()
        if abs(v) > 0.1:
            print(f"    H1[{k}] = {v:.4f}")

    # Trace where ALU_LO is supposed to come from
    print(f"\n\nTracing ALU_LO through layers:")
    x = model.embed(token_ids)
    for i in range(8):
        x_before = x.clone()
        x = model.blocks[i](x)
        alu_lo_10 = x[0, pc_marker_pos, BD.ALU_LO + 10].item()
        delta = alu_lo_10 - x_before[0, pc_marker_pos, BD.ALU_LO + 10].item()
        if abs(delta) > 0.1:
            print(f"  blocks[{i}]: ALU_LO[10] delta = {delta:.4f}, current = {alu_lo_10:.4f}")
