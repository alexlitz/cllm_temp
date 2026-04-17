#!/usr/bin/env python3
"""Debug NOP - find which L4 FFN unit writes to dim 491 and 496."""
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

ffn4 = model.blocks[4].ffn

# Check which units have non-zero W_down entries for dims 491 and 496
print("L4 FFN units with non-zero W_down to dim 491 (AX_FULL_HI[8]):")
W_down = ffn4.W_down.to_dense() if ffn4.W_down.is_sparse else ffn4.W_down
for unit in range(W_down.shape[1]):
    val = W_down[491, unit].item()
    if abs(val) > 0.001:
        print(f"  Unit {unit}: W_down[491, {unit}] = {val:.4f}")
        # Also check what W_up and W_gate for this unit
        W_up = ffn4.W_up.to_dense() if ffn4.W_up.is_sparse else ffn4.W_up
        W_gate = ffn4.W_gate.to_dense() if ffn4.W_gate.is_sparse else ffn4.W_gate
        print(f"    W_up non-zero: ", end="")
        for d in range(512):
            v = W_up[unit, d].item()
            if abs(v) > 0.001:
                print(f"[{d}]={v:.2f} ", end="")
        print()
        print(f"    W_gate non-zero: ", end="")
        for d in range(512):
            v = W_gate[unit, d].item()
            if abs(v) > 0.001:
                print(f"[{d}]={v:.2f} ", end="")
        print()

print("\nL4 FFN units with non-zero W_down to dim 496 (AX_FULL_HI[13]):")
for unit in range(W_down.shape[1]):
    val = W_down[496, unit].item()
    if abs(val) > 0.001:
        print(f"  Unit {unit}: W_down[496, {unit}] = {val:.4f}")
