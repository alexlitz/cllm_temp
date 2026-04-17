#!/usr/bin/env python3
"""Inspect L5 head 7 weights to verify OP_JSR is there."""

from neural_vm.run_vm import AutoregressiveVMRunner
from neural_vm.vm_step import _SetDim as BD

runner = AutoregressiveVMRunner()
model = runner.model

# L5 is blocks[5] (model uses 1-indexed layer naming)
l5_attn = model.blocks[5].attn

# Head 7, HD=64
base = 7 * 64

# Check V matrix
print("L5 head 7 V matrix - checking which OP flags are copied:")
op_flags = [
    ("OP_IMM", BD.OP_IMM),
    ("OP_LEA", BD.OP_LEA),
    ("OP_EXIT", BD.OP_EXIT),
    ("OP_JMP", BD.OP_JMP),
    ("OP_JSR", BD.OP_JSR),
    ("OP_ADD", BD.OP_ADD),
]

for name, dim in op_flags:
    # Check if any V position has weight to this dimension
    weights = l5_attn.W_v[base:base+64, dim].abs()
    max_weight = weights.max().item()
    if max_weight > 0.1:
        pos = weights.argmax().item()
        print(f"  {name:10s}: V[{base}+{pos}] = {weights[pos].item():.3f}")
    else:
        print(f"  {name:10s}: NOT FOUND")

print("\nIf OP_JSR shows 'NOT FOUND', it wasn't added to the V matrix!")
