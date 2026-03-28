#!/usr/bin/env python3
"""Debug: Check if weights are being loaded."""

import torch
from neural_vm.weight_loader import CompiledWeightLoader
from neural_vm.vm_step import AutoregressiveVM
from neural_vm.embedding import Opcode

vm = AutoregressiveVM(d_model=1280, n_layers=16, ffn_hidden=40960)
vm.eval()

print("Before loading weights:")
print(f"  Layer 12 W_up non-zero: {(vm.blocks[12].ffn.W_up != 0).sum().item()}")
print(f"  Layer 12 W_gate non-zero: {(vm.blocks[12].ffn.W_gate != 0).sum().item()}")
print(f"  Layer 12 W_down non-zero: {(vm.blocks[12].ffn.W_down != 0).sum().item()}")
print()

loader = CompiledWeightLoader()
print(f"Unit allocations:")
for op_name, op in [("OR", Opcode.OR), ("XOR", Opcode.XOR), ("AND", Opcode.AND),
                     ("MUL", Opcode.MUL), ("DIV", Opcode.DIV), ("MOD", Opcode.MOD),
                     ("SHL", Opcode.SHL), ("SHR", Opcode.SHR)]:
    offset = loader.unit_allocations.get(op, "N/A")
    print(f"  {op_name}: {offset}")
print()

loader.load_all_weights(vm, verbose=True)

print()
print("After loading weights:")
print(f"  Layer 12 W_up non-zero: {(vm.blocks[12].ffn.W_up != 0).sum().item()}")
print(f"  Layer 12 W_gate non-zero: {(vm.blocks[12].ffn.W_gate != 0).sum().item()}")
print(f"  Layer 12 W_down non-zero: {(vm.blocks[12].ffn.W_down != 0).sum().item()}")
print()

# Check specific unit ranges for OR operation
or_offset = loader.unit_allocations[Opcode.OR]
print(f"OR operation units: {or_offset} to {or_offset + 4096}")
or_w_up = vm.blocks[12].ffn.W_up[or_offset:or_offset+100]
or_w_gate = vm.blocks[12].ffn.W_gate[or_offset:or_offset+100]
or_w_down = vm.blocks[12].ffn.W_down[:, or_offset:or_offset+100]

print(f"  OR W_up[{or_offset}:{or_offset+100}] non-zero: {(or_w_up != 0).sum().item()}")
print(f"  OR W_gate[{or_offset}:{or_offset+100}] non-zero: {(or_w_gate != 0).sum().item()}")
print(f"  OR W_down[:, {or_offset}:{or_offset+100}] non-zero: {(or_w_down != 0).sum().item()}")
