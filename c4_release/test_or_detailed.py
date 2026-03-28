#!/usr/bin/env python3
"""Detailed debugging of OR operation."""

import torch
from neural_vm.vm_step import AutoregressiveVM
from neural_vm.weight_loader import CompiledWeightLoader
from neural_vm.nibble_embedding import NibbleVMEmbedding
from neural_vm.embedding import Opcode, E

# Create VM
vm = AutoregressiveVM(d_model=1344, n_layers=16, ffn_hidden=70000)
vm.eval()

# Load weights
loader = CompiledWeightLoader(ffn_hidden=70000)
loader.load_all_weights(vm, verbose=False)

embed = NibbleVMEmbedding(d_model=1344)

# Test OR(5, 3) = 7
print("Testing OR(5, 3) = 7")
print("=" * 70)

# Encode input
input_emb = embed.encode_vm_state(
    pc=0, ax=3, sp=0, bp=0,
    opcode=Opcode.OR, stack_top=5, batch_size=1
)

# Run through layer 12
with torch.no_grad():
    x = input_emb.unsqueeze(1)  # [1, 1, 1344]
    output = vm.blocks[12](x)
    output_flat = output.squeeze()  # [1344]

    print("RESULT slot values at each nibble position:")
    for pos in range(8):
        base_idx = pos * 168
        result_slot = base_idx + E.RESULT
        val = output_flat[result_slot].item()
        print(f"  Position {pos} (RESULT slot {result_slot}): {val:.6f}")

    print()
    print("Expected result: 7 = 0b0111 = [7, 0, 0, 0, 0, 0, 0, 0] in nibbles")

    print()
    print("All non-zero output values:")
    non_zero_indices = torch.where(output_flat.abs() > 1e-3)[0]
    for idx in non_zero_indices[:20]:  # Show first 20
        val = output_flat[idx].item()
        pos = idx // 168
        slot_in_pos = idx % 168
        print(f"  Index {idx} (pos {pos}, slot {slot_in_pos}): {val:.2f}")
