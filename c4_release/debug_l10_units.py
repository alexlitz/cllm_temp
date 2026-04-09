#!/usr/bin/env python3
"""Check what L10 FFN units write to OUTPUT_LO indices 10-12."""
import os, sys
sys.path.insert(0, os.getcwd())
import torch
from neural_vm.vm_step import AutoregressiveVM, set_vm_weights, _SetDim

model = AutoregressiveVM(vocab_size=512, n_layers=16, n_heads=8, ffn_hidden=4096)
set_vm_weights(model)
BD = _SetDim

ffn10 = model.blocks[10].ffn

# Find all units that write to OUTPUT_LO[10], OUTPUT_LO[11], OUTPUT_LO[12]
print("=== Units writing to OUTPUT_LO[10-12] ===")
for target_idx in [10, 11, 12]:
    w_down_col = ffn10.W_down[BD.OUTPUT_LO + target_idx, :].detach().numpy()
    non_zero_units = [(u, w) for u, w in enumerate(w_down_col) if abs(w) > 0.001]
    print(f"\nOUTPUT_LO[{target_idx}] has {len(non_zero_units)} writing units:")
    for u, w in non_zero_units[:10]:  # Show first 10
        b_up = ffn10.b_up[u].item()
        print(f"  Unit {u}: W_down={w:.4f}, b_up={b_up:.0f}")

# Check byte 1 LO carry units to verify they write to expected indices
print("\n=== Byte 1 LO carry unit W_down pattern ===")
for k in range(16):
    unit = 1633 + k
    # Find all non-zero entries in W_down for this unit
    row = ffn10.W_down[:, unit].detach().numpy()
    non_zero = [(d, v) for d, v in enumerate(row) if abs(v) > 0.001]
    print(f"Unit {unit} (k={k}): {[(d - BD.OUTPUT_LO, v) for d, v in non_zero if BD.OUTPUT_LO <= d < BD.OUTPUT_LO + 16]}")
