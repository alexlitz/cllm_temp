#!/usr/bin/env python3
"""Find which L15 FFN unit writes to OUTPUT_HI[12]."""

import torch
from neural_vm.vm_step import AutoregressiveVM, set_vm_weights, _SetDim as BD

print("Initializing model...")
model = AutoregressiveVM()
set_vm_weights(model)
model.eval()

# Check L15 FFN W_down for writes to OUTPUT_HI[12]
ffn15 = model.blocks[15].ffn
target_dim = BD.OUTPUT_HI + 12  # dim 202

print(f"\nLooking for L15 FFN units that write to dim {target_dim} (OUTPUT_HI[12]):")
print()

# Check each FFN unit
for unit in range(4096):
    weight = ffn15.W_down[target_dim, unit].item()
    if abs(weight) > 0.001:
        print(f"  Unit {unit}: W_down[{target_dim}, {unit}] = {weight:.4f}")

        # Also show what this unit reads (W_up)
        w_up = ffn15.W_up[unit, :]
        w_gate = ffn15.W_gate[unit, :]
        b_up = ffn15.b_up[unit].item()
        b_gate = ffn15.b_gate[unit].item()

        # Find non-zero W_up entries
        up_nonzero = [(d, w_up[d].item()) for d in range(512) if abs(w_up[d].item()) > 0.001]
        gate_nonzero = [(d, w_gate[d].item()) for d in range(512) if abs(w_gate[d].item()) > 0.001]

        print(f"    b_up = {b_up:.2f}, b_gate = {b_gate:.2f}")
        print(f"    W_up nonzero: {up_nonzero[:10]}...")
        print(f"    W_gate nonzero: {gate_nonzero[:5]}...")
        print()
