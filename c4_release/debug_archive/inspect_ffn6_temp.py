#!/usr/bin/env python3
"""Inspect Layer 6 FFN weights affecting TEMP[0]."""
import sys
sys.path.insert(0, '.')

import torch
from neural_vm.vm_step import AutoregressiveVM, set_vm_weights, _SetDim as BD

model = AutoregressiveVM()
model.eval()
set_vm_weights(model)

ffn6 = model.blocks[6].ffn

print("Layer 6 FFN - W_down[TEMP+0, :]")
print(f"  Number of hidden units: {ffn6.hidden_dim}")
print(f"  Non-zero units writing to TEMP[0]:")

W_down_temp0 = ffn6.W_down.data[BD.TEMP + 0, :]
nonzero_units = (W_down_temp0.abs() > 1e-6).nonzero(as_tuple=True)[0]

if len(nonzero_units) > 0:
    for unit in nonzero_units[:10]:  # Show first 10
        print(f"    Unit {unit}: W_down[TEMP+0, {unit}] = {W_down_temp0[unit].item():.6f}")
    if len(nonzero_units) > 10:
        print(f"    ... and {len(nonzero_units) - 10} more")
else:
    print("    (none)")

print(f"\n  b_down[TEMP+0] = {ffn6.b_down.data[BD.TEMP + 0].item():.6f}")

# Check if any unit strongly activates at PC marker
print("\nChecking units that might activate at PC marker:")
print("  (Looking for units with W_up[unit, MARK_PC] != 0)")

W_up_mark_pc = ffn6.W_up.data[:, BD.MARK_PC]
pc_units = (W_up_mark_pc.abs() > 1e-6).nonzero(as_tuple=True)[0]

if len(pc_units) > 0:
    for unit in pc_units[:10]:
        w_down_temp = W_down_temp0[unit].item()
        if abs(w_down_temp) > 1e-6:
            print(f"    Unit {unit}: W_up[MARK_PC]={W_up_mark_pc[unit].item():.2f}, W_down[TEMP+0]={w_down_temp:.6f}")
