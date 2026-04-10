#!/usr/bin/env python3
"""Find which LEA carry detection units fire in blocks[8]."""
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

S = 100

context = build_context(BYTECODE)
draft = DraftVM(BYTECODE)
draft.step()
step1_tokens = draft.draft_tokens()

ctx = context + step1_tokens[:6]  # Up to AX marker
ax_marker_pos = len(ctx) - 1

token_ids = torch.tensor([ctx], dtype=torch.long)

with torch.no_grad():
    x = model.embed(token_ids)
    for i in range(8):
        x = model.blocks[i](x)

    # Get values at AX marker
    mark_ax = x[0, ax_marker_pos, BD.MARK_AX].item()
    op_lea = x[0, ax_marker_pos, BD.OP_LEA].item()

    print(f"At AX marker:")
    print(f"  MARK_AX = {mark_ax:.4f}")
    print(f"  OP_LEA = {op_lea:.4f}")

    print(f"\n  ALU_LO values:")
    alu_lo = [x[0, ax_marker_pos, BD.ALU_LO + k].item() for k in range(16)]
    for k, v in enumerate(alu_lo):
        if abs(v) > 0.5:
            print(f"    ALU_LO[{k}] = {v:.4f}")

    print(f"\n  FETCH_LO values:")
    fetch_lo = [x[0, ax_marker_pos, BD.FETCH_LO + k].item() for k in range(16)]
    for k, v in enumerate(fetch_lo):
        if abs(v) > 0.5:
            print(f"    FETCH_LO[{k}] = {v:.4f}")

    # Calculate which carry units would fire
    # Carry units: (a, b) where a + b >= 16
    # New formula: MARK_AX*60S + ALU_LO[a]*S + FETCH_LO[b]*S - 60.5*S
    print(f"\nLEA carry detection units (a + b >= 16):")
    carry_activations = []
    for a in range(16):
        for b in range(16):
            if a + b >= 16:
                up = mark_ax * 60 * S + alu_lo[a] * S + fetch_lo[b] * S - 60.5 * S
                if up > 0:
                    silu_val = torch.nn.functional.silu(torch.tensor(up)).item()
                    contrib = silu_val * op_lea
                    carry_activations.append((a, b, up, silu_val, contrib))

    carry_activations.sort(key=lambda x: -x[4])
    for a, b, up, silu_val, contrib in carry_activations[:5]:
        print(f"  (a={a}, b={b}): up={up:.2f}, silu={silu_val:.2f}, gate*silu={contrib:.2f}")
        print(f"    ALU_LO[{a}]={alu_lo[a]:.2f}, FETCH_LO[{b}]={fetch_lo[b]:.2f}")

    # Total expected carry
    total_carry = sum(c[4] for c in carry_activations)
    print(f"\nTotal carry contribution: {total_carry:.4f}")
