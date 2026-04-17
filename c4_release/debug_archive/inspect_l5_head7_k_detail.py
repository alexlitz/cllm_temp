#!/usr/bin/env python3
"""Detailed inspection of L5 head 7 K matrix."""

from neural_vm.run_vm import AutoregressiveVMRunner
from neural_vm.vm_step import _SetDim as BD

import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Force CPU mode

runner = AutoregressiveVMRunner()
model = runner.model

# L5 is blocks[5]
l5_attn = model.blocks[5].attn

# Head 7, HD=64
base = 7 * 64

print("L5 head 7 K matrix - checking ADDR_KEY matching:")

# Check ADDR_KEY dimensions (should match the code pattern)
for k in range(16):
    weight_lo = l5_attn.W_k[base + k, BD.ADDR_KEY + k].item()
    weight_hi = l5_attn.W_k[base + 16 + k, BD.ADDR_KEY + 16 + k].item()
    if abs(weight_lo) > 0.1 or abs(weight_hi) > 0.1:
        print(f"  K[{base}+{k:2d}, ADDR_KEY+{k:2d}] = {weight_lo:+.1f}")
        print(f"  K[{base}+{16+k:2d}, ADDR_KEY+{16+k:2d}] = {weight_hi:+.1f}")

# Check ADDR_KEY top nibble
weight_top = l5_attn.W_k[base + 32, BD.ADDR_KEY + 32].item()
print(f"  K[{base}+32, ADDR_KEY+32] = {weight_top:+.1f}")

# Check anti-leakage gate
GATE = 33
weight_gate_q = l5_attn.W_q[base + GATE, BD.MARK_PC].item()
weight_gate_k = l5_attn.W_k[base + GATE, BD.CONST].item()
print(f"\nAnti-leakage gate:")
print(f"  Q[{base}+{GATE}, MARK_PC] = {weight_gate_q:+.1f}")
print(f"  K[{base}+{GATE}, CONST] = {weight_gate_k:+.1f}")

print("\nExpected: All ADDR_KEY weights should be ≈ +20.0 (L)")
