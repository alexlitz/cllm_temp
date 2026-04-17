#!/usr/bin/env python3
"""Debug: Check VM layer sizes."""

import torch
from neural_vm.vm_step import AutoregressiveVM

vm = AutoregressiveVM(d_model=1280, n_layers=16, ffn_hidden=40960)

print(f"VM d_model: {vm.d_model}")
print(f"VM n_layers: {len(vm.blocks)}")
print()

for i, block in enumerate(vm.blocks):
    ffn = block.ffn
    print(f"Layer {i}:")
    print(f"  W_up shape: {ffn.W_up.shape}")
    print(f"  W_gate shape: {ffn.W_gate.shape}")
    print(f"  W_down shape: {ffn.W_down.shape}")
    if i == 12:
        print(f"  ← Layer 12 (ALU layer)")
print()
print(f"Expected FFN shapes for ffn_hidden=40960:")
print(f"  W_up: [40960, 1280]")
print(f"  W_gate: [40960, 1280]")
print(f"  W_down: [1280, 40960]")
