#!/usr/bin/env python3
"""Test a single OR operation with detailed debugging."""

import torch
from neural_vm.vm_step import AutoregressiveVM
from neural_vm.weight_loader import CompiledWeightLoader
from neural_vm.nibble_embedding import NibbleVMEmbedding
from neural_vm.embedding import Opcode

# Create VM
vm = AutoregressiveVM(d_model=1344, n_layers=16, ffn_hidden=70000)
vm.eval()

# Load weights
loader = CompiledWeightLoader(ffn_hidden=70000)
result = loader.load_all_weights(vm, verbose=False)

embed = NibbleVMEmbedding(d_model=1344)

# Test OR(5, 3) = 7
print("Testing OR(5, 3) = 7")
print("=" * 70)

# Encode input
input_emb = embed.encode_vm_state(
    pc=0, ax=3, sp=0, bp=0,
    opcode=Opcode.OR, stack_top=5, batch_size=1
)

print(f"Input shape: {input_emb.shape}")
print(f"ax=3, stack_top=5")
print()

# Run through layer 12
with torch.no_grad():
    x = input_emb.unsqueeze(1)  # [1, 1, 1344]

    print("Before layer 12:")
    print(f"  x shape: {x.shape}")
    print(f"  x range: [{x.min().item():.2f}, {x.max().item():.2f}]")
    print()

    # Run through layer 12
    output = vm.blocks[12](x)

    print("After layer 12:")
    print(f"  output shape: {output.shape}")
    print(f"  output range: [{output.min().item():.2f}, {output.max().item():.2f}]")
    print()

    # Decode result from position 0
    result = embed.decode_result_nibbles(output.squeeze(1))

    print(f"Decoded result: {result} (expected 7)")

    # Check non-zero values in output
    non_zero = (output.abs() > 1e-3).sum().item()
    print(f"Non-zero output values: {non_zero} / {output.numel()}")
