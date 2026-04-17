#!/usr/bin/env python3
"""Inspect L5 head 7 Q and K matrices to verify attention setup."""

from neural_vm.run_vm import AutoregressiveVMRunner
from neural_vm.vm_step import _SetDim as BD

runner = AutoregressiveVMRunner()
model = runner.model

# L5 is blocks[5] (model uses 1-indexed layer naming)
l5_attn = model.blocks[5].attn

# Head 7, HD=64
base = 7 * 64

print("L5 head 7 Q matrix (query dims for PC marker):")
q_dims = [
    ("HAS_SE", BD.HAS_SE),
    ("MARK_PC", BD.MARK_PC),
    ("OP_JSR", BD.OP_JSR),
    ("OP_IMM", BD.OP_IMM),
    ("OP_JMP", BD.OP_JMP),
]
for name, dim in q_dims:
    weight = l5_attn.W_q[base, dim].item()
    if abs(weight) > 0.1:
        print(f"  {name:10s}: W_q[{base}] = {weight:+.3f}")

print("\nL5 head 7 K matrix (key dims for CODE bytes):")
# Check ADDR_KEY lo nibbles
for k in range(16):
    weight = l5_attn.W_k[base + k, BD.ADDR_KEY + k].item()
    if abs(weight) > 0.1:
        print(f"  W_k[{base}+{k:2d}, ADDR_KEY+{k:2d}] = {weight:+.1f}")

# Check ADDR_KEY hi nibbles
for k in range(16):
    weight = l5_attn.W_k[base + 16 + k, BD.ADDR_KEY + 16 + k].item()
    if abs(weight) > 0.1:
        print(f"  W_k[{base}+{16+k:2d}, ADDR_KEY+{16+k:2d}] = {weight:+.1f}")

# Check ADDR_KEY top nibble
weight = l5_attn.W_k[base + 32, BD.ADDR_KEY + 32].item()
if abs(weight) > 0.1:
    print(f"  W_k[{base}+32, ADDR_KEY+32] = {weight:+.1f}")

# Check anti-leakage gate
GATE = 33
weight_gate = l5_attn.W_k[base + GATE, BD.CONST].item()
if abs(weight_gate) > 0.1:
    print(f"  W_k[{base}+{GATE}, CONST] = {weight_gate:+.1f} (anti-leakage)")

print("\nExpected behavior:")
print("  Q: Fire at PC marker on first step (MARK_PC=L, HAS_SE=-L)")
print("  K: Match CODE byte at ADDR_KEY=2 (opcode position)")
print("  V: Copy OP flags to PC marker")
