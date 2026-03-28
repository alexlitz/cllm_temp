#!/usr/bin/env python3
"""Debug weight loading."""

import torch
from neural_vm.vm_step import AutoregressiveVM
from neural_vm.weight_loader import CompiledWeightLoader
from neural_vm.embedding import Opcode, E

# Create VM
vm = AutoregressiveVM(d_model=1344, n_layers=16, ffn_hidden=20000)
vm.eval()

# Load weights
loader = CompiledWeightLoader(ffn_hidden=20000)

print("OR offset:", loader.unit_allocations[Opcode.OR])
print()

print("Loading weights...")
try:
    result = loader.load_all_weights(vm, verbose=True)
    print("Load result:", result)
except Exception as e:
    print(f"Loading failed: {e}")
    import traceback
    traceback.print_exc()

# Check layer 9 (PRIMARY_ALU)
layer_idx = 9
layer = vm.blocks[layer_idx]

print()
print(f"Checking non-zero parameters in layer {layer_idx}:")
print(f"  W_up non-zero: {(layer.ffn.W_up.data.abs() > 1e-6).sum().item()}")
print(f"  b_up non-zero: {(layer.ffn.b_up.data.abs() > 1e-6).sum().item()}")
print(f"  W_gate non-zero: {(layer.ffn.W_gate.data.abs() > 1e-6).sum().item()}")
print(f"  b_gate non-zero: {(layer.ffn.b_gate.data.abs() > 1e-6).sum().item()}")
print(f"  W_down non-zero: {(layer.ffn.W_down.data.abs() > 1e-6).sum().item()}")

# Check a specific unit for OR(3,5)=7
OR_offset = loader.unit_allocations[Opcode.OR]
# For OR at position 0, unit for (a=3, b=5): OR_offset + 0*256 + (3*16 + 5)
unit_idx = OR_offset + 0 * 256 + (3 * 16 + 5)

print()
print(f"Unit {unit_idx} for OR(3,5) at position 0:")

# Check W_up weights at NIB_A binary bit slots (position 0)
pos = 0
base_idx = pos * 168
print(f"  W_up values at NIB_A binary bit slots (base_idx={base_idx}):")
for i in range(4):
    bit_slot = base_idx + E.NIB_A_BIT0 + i
    weight = layer.ffn.W_up.data[unit_idx, bit_slot].item()
    print(f"    NIB_A_BIT{i} (slot {bit_slot}): {weight}")

print(f"  b_up: {layer.ffn.b_up.data[unit_idx].item()}")

# Check W_gate weights at NIB_B binary bit slots
print(f"  W_gate values at NIB_B binary bit slots:")
for i in range(4):
    bit_slot = base_idx + E.NIB_B_BIT0 + i
    weight = layer.ffn.W_gate.data[unit_idx, bit_slot].item()
    print(f"    NIB_B_BIT{i} (slot {bit_slot}): {weight}")

print(f"  b_gate: {layer.ffn.b_gate.data[unit_idx].item()}")

# Check W_down
result_slot = base_idx + E.RESULT
print(f"  W_down[{result_slot}, {unit_idx}]: {layer.ffn.W_down.data[result_slot, unit_idx].item()}")
