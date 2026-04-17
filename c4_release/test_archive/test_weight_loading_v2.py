#!/usr/bin/env python3
"""Test weight loading with graph coloring."""

import torch
from neural_vm.vm_step import AutoregressiveVM
from neural_vm.weight_loader_v2 import CompiledWeightLoader

# Create loader (with layer sharing)
loader = CompiledWeightLoader(use_layer_sharing=True, verbose=True)

print(f"Creating VM with n_layers={loader.n_layers}, ffn_hidden={loader.ffn_hidden}...")
vm = AutoregressiveVM(d_model=1352, n_layers=loader.n_layers, ffn_hidden=loader.ffn_hidden)
vm.eval()

print()
print("Loading weights...")
print()

loader.load_all_weights(vm)

print()
print(f"Layer unit usage:")
for layer_idx, units_used in loader.layer_unit_usage.items():
    utilization = 100 * units_used / loader.ffn_hidden
    print(f"  Layer {layer_idx}: {units_used:5d} / {loader.ffn_hidden} units ({utilization:.1f}%)")
