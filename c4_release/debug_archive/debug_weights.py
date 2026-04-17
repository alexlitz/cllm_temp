#!/usr/bin/env python3
"""Check L10 FFN weights for carry application."""
import os, sys
sys.path.insert(0, os.getcwd())
import torch
from neural_vm.vm_step import AutoregressiveVM, set_vm_weights, _SetDim

model = AutoregressiveVM(vocab_size=512, n_layers=16, n_heads=8, ffn_hidden=4096)
set_vm_weights(model)
BD = _SetDim

ffn10 = model.blocks[10].ffn

print("=== Byte 0 carry application (units 1600-1632) ===")
# Byte 0 LO: 16 units
for u in range(1600, 1616):
    k = u - 1600
    gate_val = ffn10.W_gate[u, BD.OUTPUT_LO + k].item()
    print(f"  unit {u}: OUTPUT_LO[{k}] gate = {gate_val:.2f}")

print("\n=== Byte 0 HI carry (units 1616-1631) ===")
for u in range(1616, 1632):
    k = u - 1616
    gate_lo15 = ffn10.W_gate[u, BD.OUTPUT_LO + 15].item()
    gate_hik = ffn10.W_gate[u, BD.OUTPUT_HI + k].item()
    print(f"  unit {u}: OUTPUT_HI[{k}] gate = {gate_hik:.2f}, OUTPUT_LO[15] gate = {gate_lo15:.2f}")

print("\n=== Byte 0 carry_1 flag (unit 1632) ===")
u = 1632
gate_lo15 = ffn10.W_gate[u, BD.OUTPUT_LO + 15].item()
gate_hi15 = ffn10.W_gate[u, BD.OUTPUT_HI + 15].item()
print(f"  unit {u}: OUTPUT_LO[15] = {gate_lo15:.2f}, OUTPUT_HI[15] = {gate_hi15:.2f}")

print("\n=== Byte 1 LO carry application (units 1633-1648) ===")
for u in range(1633, 1649):
    k = u - 1633
    gate_val = ffn10.W_gate[u, BD.OUTPUT_LO + k].item()
    up_bi1 = ffn10.W_up[u, BD.BYTE_INDEX_1].item()
    b_up = ffn10.b_up[u].item()
    print(f"  unit {u}: OUTPUT_LO[{k}] gate = {gate_val:.2f}, W_up[BI1] = {up_bi1:.0f}, b_up = {b_up:.0f}")

print("\n=== Check for any non-zero gate weights at OUTPUT_LO for units 1633-1648 ===")
for u in range(1633, 1649):
    gate_row = ffn10.W_gate[u, BD.OUTPUT_LO:BD.OUTPUT_LO+16].tolist()
    non_zero = [(i, v) for i, v in enumerate(gate_row) if abs(v) > 0.01]
    if non_zero:
        print(f"  unit {u}: {non_zero}")
    else:
        print(f"  unit {u}: all zeros")
