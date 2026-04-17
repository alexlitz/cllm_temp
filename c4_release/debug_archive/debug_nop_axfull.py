#!/usr/bin/env python3
"""Debug NOP - check AX_FULL values."""
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

bytecode = [Opcode.NOP]
context = build_context(bytecode)
draft = DraftVM(bytecode)
draft.step()
draft_tokens = draft.draft_tokens()

# Build context up to AX marker
current_ctx = context + draft_tokens[:6]  # Through AX marker

token_ids = torch.tensor([current_ctx], dtype=torch.long)

with torch.no_grad():
    x = model.embed(token_ids)

    # Run through L0-L5
    for i in range(6):
        x = model.blocks[i](x)

    print("After L5, at AX marker:")
    print("\nAX_FULL_LO values:")
    for k in range(16):
        val = x[0, -1, BD.AX_FULL_LO + k].item()
        if abs(val) > 0.1:
            print(f"  [{k}] = {val:.2f}")

    print("\nAX_FULL_HI values:")
    for k in range(16):
        val = x[0, -1, BD.AX_FULL_HI + k].item()
        if abs(val) > 0.1:
            print(f"  [{k}] = {val:.2f}")

    print("\nOP_NOP value:", x[0, -1, BD.OP_NOP].item())
    print("MARK_AX value:", x[0, -1, BD.MARK_AX].item())
    print("MARK_PC value:", x[0, -1, BD.MARK_PC].item())
    print("IS_BYTE value:", x[0, -1, BD.IS_BYTE].item())

    # Run L6
    x = model.blocks[6](x)
    print("\nAfter L6, OUTPUT values:")
    print("OUTPUT_LO:")
    for k in range(16):
        val = x[0, -1, BD.OUTPUT_LO + k].item()
        if abs(val) > 0.1:
            print(f"  [{k}] = {val:.2f}")
    print("OUTPUT_HI:")
    for k in range(16):
        val = x[0, -1, BD.OUTPUT_HI + k].item()
        if abs(val) > 0.1:
            print(f"  [{k}] = {val:.2f}")
