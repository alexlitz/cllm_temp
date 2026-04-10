#!/usr/bin/env python3
"""Debug L9 LEA hi nibble units for LEA 16 at PC marker."""
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

ctx = context + step1_tokens[:1]  # Just PC marker
pc_marker_pos = len(ctx) - 1

token_ids = torch.tensor([ctx], dtype=torch.long)

with torch.no_grad():
    x = model.embed(token_ids)
    for i in range(9):
        x = model.blocks[i](x)

    # Get values at PC marker before L9 FFN
    mark_ax = x[0, pc_marker_pos, BD.MARK_AX].item()
    op_lea = x[0, pc_marker_pos, BD.OP_LEA].item()
    carry = x[0, pc_marker_pos, BD.CARRY].item()

    print(f"At PC marker (pos {pc_marker_pos}) before blocks[9]:")
    print(f"  MARK_AX = {mark_ax:.4f}")
    print(f"  OP_LEA = {op_lea:.4f}")
    print(f"  CARRY = {carry:.4f}")

    print(f"\n  ALU_HI values:")
    alu_hi = [x[0, pc_marker_pos, BD.ALU_HI + k].item() for k in range(16)]
    for k, v in enumerate(alu_hi):
        if abs(v) > 0.5:
            print(f"    ALU_HI[{k}] = {v:.4f}")

    print(f"\n  FETCH_HI values:")
    fetch_hi = [x[0, pc_marker_pos, BD.FETCH_HI + k].item() for k in range(16)]
    for k, v in enumerate(fetch_hi):
        if abs(v) > 0.5:
            print(f"    FETCH_HI[{k}] = {v:.4f}")

    # Check L9 LEA hi nibble activation for various (a, b) pairs
    print(f"\nL9 LEA hi nibble activation (threshold=45):")
    print(f"Formula: MARK_AX*S + ALU_HI[a]*S + FETCH_HI[b]*S - CARRY*0.01 - 45*S")

    # For no-carry case, check which units would fire
    activations = []
    for a in range(16):
        for b in range(16):
            # no-carry formula: MARK_AX + ALU_HI[a] + FETCH_HI[b] - 0.01*CARRY - 45
            up = mark_ax * S + alu_hi[a] * S + fetch_hi[b] * S - carry * 0.01 - 45 * S
            if up > 0:
                result = (a + b) % 16
                silu_val = torch.nn.functional.silu(torch.tensor(up)).item()
                contrib = silu_val * op_lea
                activations.append((a, b, result, up, contrib))

    activations.sort(key=lambda x: -x[4])
    for a, b, result, up, contrib in activations[:5]:
        print(f"  (a={a}, b={b}) -> OUTPUT_HI[{result}]: up={up:.2f}, contrib={contrib:.2f}")
        print(f"    ALU_HI[{a}]={alu_hi[a]:.2f}, FETCH_HI[{b}]={fetch_hi[b]:.2f}")

    # Why is MARK_AX + FETCH_HI[1] enough to fire?
    print(f"\nDebug: why units fire with MARK_AX=0?")
    print(f"  MARK_AX({mark_ax:.2f}) + ALU_HI[0]({alu_hi[0]:.2f}) + FETCH_HI[1]({fetch_hi[1]:.2f}) - 45")
    print(f"  = {mark_ax + alu_hi[0] + fetch_hi[1] - 45:.2f}")
