#!/usr/bin/env python3
"""Test a single operation in isolation."""

import torch
from neural_vm.vm_step import AutoregressiveVM
from neural_vm.weight_loader_v2 import CompiledWeightLoader
from neural_vm.nibble_embedding import NibbleVMEmbedding
from neural_vm.embedding import Opcode

# Create and load VM
loader = CompiledWeightLoader(use_layer_sharing=True, verbose=False)
vm = AutoregressiveVM(d_model=1352, n_layers=loader.n_layers, ffn_hidden=loader.ffn_hidden)
vm.eval()
loader.load_all_weights(vm)

embed = NibbleVMEmbedding(d_model=1352)

print("Testing OR(5, 3) = 7")
print()

# Encode input
input_emb = embed.encode_vm_state(
    pc=0, ax=3, sp=0, bp=0,
    opcode=Opcode.OR, stack_top=5, batch_size=1
)

print("Input embedding non-zero values:")
nonzero_dims = torch.where(input_emb[0].abs() > 1e-9)[0]
for dim in nonzero_dims[:20]:  # Show first 20
    print(f"  dim {dim:4d}: {input_emb[0, dim].item():.2f}")

print()
print("Running through layers...")

with torch.no_grad():
    x = input_emb.unsqueeze(1)

    # Run layer by layer and check result
    for layer_idx, layer in enumerate(vm.blocks):
        x_before = x.clone()
        x = layer(x)
        result = embed.decode_result_nibbles(x.squeeze(1))

        # Check if result changed
        delta = (x - x_before).abs().max().item()
        print(f"  Layer {layer_idx}: result = {result:8d}, max_delta = {delta:.6f}")

        if result == 7:
            print(f"\n✓ Correct result after layer {layer_idx}!")
            break
    else:
        print(f"\n✗ Never got correct result. Final: {result}")
